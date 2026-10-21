"""Connector checkpoints: one format for every kind and a retention policy.

**Format.** A committed checkpoint is an artifact of kind ``"checkpoint"`` holding ``connector.json`` (the
:class:`ConnectorSpec`'s identity form, so rewording its rationale changes neither the spec's digest nor the
checkpoint's; the identity form of the encoder recipe the connector reads; the receiver depth, ``d_in``, ``d_model``,
the training step, and with an appended extraction its recipe's identity form, ``append_encoder``) and
``weights.safetensors`` (the module's ``state_dict``). It is loaded with :meth:`Connector.from_state`, with no ridge
solutions needed.

**Retention.**

- *Resume state* (weights + optimizer + step, written every ``CheckpointSpec.resume_every`` steps and on SIGUSR1;
  :func:`save_resume_state`, :func:`load_resume_state`) is a run-scoped file (``Paths.runs``), overwritten
  atomically. It is not an artifact and is never hashed; the run's last resume state is deleted when the run's final
  checkpoint is committed (:func:`clear_resume_state`).
- *Committed checkpoints* (weights only, no optimizer) are made at step 0, at every step a suite evaluates, and at the
  end. Only these are content-addressed, cited and kept.
- Files are hashed once, at commit; :meth:`~latent_interconnect.artifacts.store.ArtifactStore.verify` re-checks them
  through the store's persistent hash cache (``artifacts.store``).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import canonical_json
from latent_interconnect.writes.connector import Connector, ConnectorSpec
from latent_interconnect.writes.inputs import EncoderShape

__all__ = [
    "CHECKPOINT_KIND",
    "RESUME_FILE",
    "MissingEncoderError",
    "ResumeState",
    "ResumeSummary",
    "checkpoint_encoder",
    "clear_resume_state",
    "commit_checkpoint",
    "load_connector",
    "load_resume_state",
    "read_checkpoint",
    "read_resume_summary",
    "save_resume_state",
]

CHECKPOINT_KIND = "checkpoint"
_SPEC_FILE = "connector.json"
_WEIGHTS_FILE = "weights.safetensors"
RESUME_FILE = "resume.pt"
"""A run's resume state, in its run directory (:func:`save_resume_state`; its temp files are ``resume.pt.<hex>``)."""


@dataclass(frozen=True)
class ResumeState:
    """A run's resume state (``save_resume_state``).

    Attributes:
        connector: the connector's ``state_dict`` (on CPU).
        optimizer: the optimizer's ``state_dict``.
        step: optimizer steps completed; the sampler's position is this step (its draws are step-indexed).
        extra: what else the trainer needs to continue exactly (e.g. the committed checkpoints by step), as plain
            JSON-like values.
    """

    connector: dict[str, Tensor]
    optimizer: dict[str, Any]
    step: int
    extra: dict[str, Any]


def save_resume_state(
    connector: Connector, optimizer_state: dict[str, Any], step: int, workspace: Path, extra: Mapping[str, Any]
) -> Path:
    """Atomically overwrite the run's resume state in ``workspace`` (weights, optimizer, step, ``extra``).

    The file is written beside its final name, synced to disk and renamed over it, so a job killed mid-write leaves the
    previous state intact; a temp file such a kill left is removed by the next save. It also records the connector
    config's digest, which :func:`load_resume_state` checks.

    Returns:
        The resume file.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    final = workspace / RESUME_FILE
    for orphan in workspace.glob(f"{RESUME_FILE}.*"):  # a temp file a killed save left (this function's own names)
        orphan.unlink()
    tmp = workspace / f"{RESUME_FILE}.{uuid.uuid4().hex}"
    state = {k: v.detach().to("cpu").clone() for k, v in connector.state_dict().items()}
    payload = {
        "connector_sha": connector.spec.digest(),
        "connector": state,
        "optimizer": optimizer_state,
        "step": step,
        "extra": dict(extra),
    }
    with tmp.open("wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(final)
    return final


def load_resume_state(connector: Connector, workspace: Path) -> ResumeState | None:
    """Return the run's resume state in ``workspace``, or None if there is none.

    Raises:
        ValueError: if the state was saved for another connector config (its digest differs from ``connector``'s).
    """
    path = workspace / RESUME_FILE
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["connector_sha"] != connector.spec.digest():
        raise ValueError(f"{path} was saved for connector config {payload['connector_sha'][:12]}, not this run's")
    return ResumeState(payload["connector"], payload["optimizer"], int(payload["step"]), dict(payload["extra"]))


@dataclass(frozen=True)
class ResumeSummary:
    """What a run's resume state says about the run, without its weights or optimizer (:func:`read_resume_summary`).

    Attributes:
        connector_sha: the digest of the connector config it was saved for.
        step: optimizer steps completed when it was saved.
        extra: the trainer's ``extra`` (the committed checkpoints by step, the derived data's digest, the execution).
    """

    connector_sha: str
    step: int
    extra: dict[str, Any]


def read_resume_summary(workspace: Path) -> ResumeSummary | None:
    """Return the run's resume state in ``workspace`` without reading its tensors, or None if there is none.

    The file is memory-mapped (``torch.load(mmap=True)``), so only its pickled structure is read, whatever the size of
    the weights and optimizer state it holds; nothing is written. A publication of a stopped run's achieved
    checkpoints reads it (``run.train``).
    """
    path = workspace / RESUME_FILE
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return ResumeSummary(str(payload["connector_sha"]), int(payload["step"]), dict(payload["extra"]))


def clear_resume_state(workspace: Path) -> None:
    """Delete the run's resume state (done when the run's final checkpoint is committed); no state is not an error."""
    (workspace / RESUME_FILE).unlink(missing_ok=True)


class MissingEncoderError(ValueError):
    """A checkpoint does not record the encoder recipe its connector reads (written before the recipe was recorded)."""


def commit_checkpoint(
    connector: Connector,
    step: int,
    store: ArtifactStore,
    *,
    encoder: EncoderShape,
    append_encoder: EncoderShape | None = None,
) -> ArtifactRef:
    """Commit a weights-only checkpoint artifact.

    ``connector.json`` holds the spec's identity form (rationale fields excluded; they load back at their defaults), the
    identity form of ``encoder`` (the recipe the connector reads: its one source for every consumer, e.g. the suite
    stage's input loader, :func:`checkpoint_encoder`), ``append_encoder``'s when given (the appended extraction's
    recipe; absent otherwise, so a checkpoint without one keeps its identity), the receiver depth, ``d_in``,
    ``d_model`` and ``step``; ``weights.safetensors`` holds the ``state_dict`` on CPU. The manifest metadata names the
    connector, its config digest and the step.

    Raises:
        ValueError: if ``encoder`` is not the recipe the connector reads (``spec.input.recipe``), or
            ``append_encoder`` not the recipe it appends.
        MissingEncoderError: if the connector appends an extraction and ``append_encoder`` is None (its readers would
            otherwise be checked against no recorded recipe).
    """
    if encoder.name != connector.spec.input.recipe:
        raise ValueError(
            f"connector {connector.spec.name!r} reads {connector.spec.input.recipe!r}, not {encoder.name!r}"
        )
    appended = connector.spec.input.append
    if appended is not None and append_encoder is None:
        raise MissingEncoderError(f"connector {connector.spec.name!r} appends {appended.recipe!r}: give its recipe")
    if append_encoder is not None and (appended is None or append_encoder.name != appended.recipe):
        raise ValueError(f"connector {connector.spec.name!r} does not append {append_encoder.name!r}")
    staged = store.staging(f"checkpoint-{uuid.uuid4().hex}")
    header = {
        "spec": connector.spec.identity(),
        "encoder": encoder.identity(),
        "depth": connector.depth,
        "d_in": connector.d_in,
        "d_model": connector.d_model,
        "step": step,
    }
    if append_encoder is not None:
        header["append_encoder"] = append_encoder.identity()
    (staged / _SPEC_FILE).write_text(canonical_json(header))
    state = {k: v.detach().to("cpu").contiguous() for k, v in connector.state_dict().items()}
    save_file(state, str(staged / _WEIGHTS_FILE))
    meta = {"connector": connector.spec.name, "connector_sha": connector.spec.digest(), "step": step}
    return store.commit(staged, CHECKPOINT_KIND, meta)


def checkpoint_encoder(store: ArtifactStore, ref: ArtifactRef, key: str = "encoder") -> dict[str, Any]:
    """Return the identity form of the encoder recipe a checkpoint records (validate it as ``EncoderRecipe``).

    ``key="append_encoder"`` returns its appended extraction's recipe instead (ask only for a connector that appends
    one).

    Raises:
        ValueError: if ``ref`` is not a checkpoint.
        MissingEncoderError: if the checkpoint records no such recipe.
    """
    if ref.kind != CHECKPOINT_KIND:
        raise ValueError(f"{ref.short} is a {ref.kind!r} artifact, not a checkpoint")
    store.manifest(ref)
    header = json.loads((store.path(ref) / _SPEC_FILE).read_text())
    if not isinstance(header.get(key), dict):
        what = "appended recipe" if key == "append_encoder" else "encoder recipe"
        raise MissingEncoderError(f"checkpoint {ref.short} records no {what}: re-commit it through a stage")
    encoder: dict[str, Any] = header[key]
    return encoder


def load_connector(store: ArtifactStore, ref: ArtifactRef) -> tuple[Connector, int]:
    """Load a checkpoint artifact; return the connector (on CPU) and its training step.

    Raises:
        ValueError: if ``ref`` is not a checkpoint.
        IntegrityError: if its manifest does not match ``ref``.
    """
    if ref.kind != CHECKPOINT_KIND:
        raise ValueError(f"{ref.short} is a {ref.kind!r} artifact, not a checkpoint")
    store.manifest(ref)
    return read_checkpoint(store.path(ref))


def read_checkpoint(directory: Path) -> tuple[Connector, int]:
    """Read a checkpoint directory (``connector.json`` + ``weights.safetensors``), e.g. one shipped with a release.

    Returns the connector (on CPU, plugins in its spec loaded) and its training step. No store is needed.

    Raises:
        FileNotFoundError: if either file is missing.
    """
    directory = Path(directory)
    header = json.loads((directory / _SPEC_FILE).read_text())
    spec = ConnectorSpec.model_validate(header["spec"])
    state = load_file(str(directory / _WEIGHTS_FILE))
    connector = Connector.from_state(spec, int(header["depth"]), int(header["d_in"]), int(header["d_model"]), state)
    return connector, int(header["step"])
