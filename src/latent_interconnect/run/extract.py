"""Entry point: extract one encoder over one object set (sharded, resumable); commit the embeddings.

The stage reads the object list (an item set with the one ``object`` role) and the raw inputs from the cited domain's
object table, writes them as the extraction job's object list (``encoders.jobs.stage_objects``, content-addressed under
the store's staging area), and runs the job: in this process, or, with the ``python`` option, as the encoder worker
(``python -m latent_interconnect.encoders.worker``) under that interpreter, the encoder's own environment
(``EncoderRecipe.environment``; pass its interpreter as ``--option python=...``, examples/guides/extending.md
§6 Step 4). It then finalizes the artifact.
A job already committed is returned as it is (``ArtifactStore.committed``), never recomputed, once its worker's
origin is accepted (``encoders.jobs.run_worker``; the worker loads the model to know it). The run declares its
hardware (``device_class``: extraction runs on A100, as capture and scoring do) and refuses another kind or
topology (any accelerator model: ``DeviceClass.matches``), before any work and before a queue job claims
anything (``Stage.target``); the torch device follows from it
(:data:`TORCH_DEVICE`), so what runs is what is declared.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field, model_validator

import latent_interconnect
from latent_interconnect.artifacts.store import read_handoff
from latent_interconnect.canonical import canonical_json
from latent_interconnect.config import ArtifactDigest, StageConfig
from latent_interconnect.data.items import OBJECT_ROLE
from latent_interconnect.domains.base import DOMAINS, DomainSpec
from latent_interconnect.encoders.base import RawObject
from latent_interconnect.encoders.jobs import ExtractionJob, finalize, job_file, run_worker, stage_objects
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.plugins import UnknownNameError, load_plugins
from latent_interconnect.provenance import DeviceClass
from latent_interconnect.run._main import (
    ExecutionTarget,
    Stage,
    StageResult,
    require_hardware,
    stage_main,
    visible_devices,
)
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "TORCH_DEVICE", "ExtractConfig", "main"]

TORCH_DEVICE = {DeviceClass.A100: "cuda", DeviceClass.CPU: "cpu"}
"""The torch device an extraction runs on, by its declared device class (extraction never runs on 8×H200)."""


class ExtractConfig(StageConfig):
    """One extraction.

    Attributes:
        recipe: the encoder recipe (``$ref`` to its YAML).
        domain: the domain at a fixed data version, whose object table supplies the raw inputs (so the extraction's
            identity covers the exact sequences or structures); its kind of object (the domain class's
            ``object_kind``, e.g. ``protein`` for the protein and the PFMbench domains) must be the recipe's domain.
        objects: digest of the object list (usually a ``$pin``).
        part_size: objects per resumable part (the extractor batches within a part, so it is part of the identity).
        device_class: the declared hardware: ``A100`` for every extraction, ``cpu`` for the tests; it fixes the
            torch device (:data:`TORCH_DEVICE`).
    """

    recipe: EncoderRecipe
    domain: DomainSpec
    objects: ArtifactDigest
    part_size: int = Field(ge=1)
    device_class: DeviceClass

    @model_validator(mode="after")
    def _check(self) -> ExtractConfig:
        load_plugins(self.domain.plugins)
        try:
            kind = DOMAINS.get(self.domain.name).object_kind
        except UnknownNameError as err:  # a validation error, not a bare KeyError out of the validator
            raise ValueError(f"unknown domain {self.domain.name!r}") from err
        if kind != self.recipe.domain:
            raise ValueError(
                f"recipe {self.recipe.name!r} is for {self.recipe.domain!r}, not {self.domain.name!r} ({kind})"
            )
        if self.device_class not in TORCH_DEVICE:
            raise ValueError(f"extraction runs on {sorted(TORCH_DEVICE)}, not {self.device_class.value}")
        return self


def _run(
    config: ExtractConfig, ws: Workspace, python: str | None = None, origin_handoff: Path | None = None
) -> StageResult:
    """Output ``embeddings``; inputs ``domain`` and ``objects``.

    Options (execution, not identity): ``python``, the encoder environment's interpreter (default: run the worker in
    this process); ``origin_handoff``, a JSON file declaring an origin handoff (``ArtifactStore.require_origin``), for
    work of other code. The record's ``execution`` is the extractor's fingerprint as the artifact holds it
    (``fingerprint.<key>``; it declares the ``dtype`` the model ran in), the torch device the worker ran on
    (``worker.device``, from ``device_class``), the objects the extractor skipped (``skipped``, their count, and
    ``skipped.objects``, object id → reason, as the artifact lists them), and the job's origin record (``origin``:
    the origin its work was accepted under and every handoff).

    Raises:
        WrongHardwareError: if the visible devices are not the config's ``device_class``.
        ValueError: if the object list is not a single-role list of this domain's objects, or names objects the
            domain does not have.
        subprocess.CalledProcessError: if the encoder worker fails.
        OriginError: if the job's work comes from another origin and no valid handoff covers it.
    """
    devices = visible_devices()
    require_hardware(devices, config.device_class)
    device = TORCH_DEVICE[config.device_class]
    store = ws.artifacts
    items = ws.item_set(config.objects)
    if dict(items.roles) != {OBJECT_ROLE: config.domain.name}:
        raise ValueError(f"objects {config.objects[:12]}: roles {dict(items.roles)}, not a {config.domain.name} list")
    table = ws.domain(config.domain).objects()
    unknown = [o for o in items.object_ids if o not in table.index]
    if unknown:
        raise ValueError(f"{len(unknown)} object(s) not in domain {config.domain.name!r}, e.g. {unknown[:5]}")
    path, sha = stage_objects(store.root, [RawObject(o, str(table.at[o, "value"])) for o in items.object_ids])
    job = ExtractionJob(config.recipe.model_dump(mode="json"), str(path), sha, config.part_size, str(store.root))
    if python is None:
        run_worker(job, device, handoff=read_handoff(origin_handoff))
    else:
        job.dump(job_file(job))
        src = str(Path(latent_interconnect.__file__).resolve().parents[1])
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (src, os.environ.get("PYTHONPATH"))))}
        worker = [python, "-m", "latent_interconnect.encoders.worker", str(job_file(job)), "--device", device]
        worker += [] if origin_handoff is None else ["--handoff", str(origin_handoff)]
        try:
            subprocess.run(worker, check=True, env=env)
        finally:  # the worker's copy of the job, failed or not; a rerun writes it again
            job_file(job).unlink(missing_ok=True)
    ref = finalize(job)
    fingerprint = store.manifest(ref).meta["fingerprint"]
    inputs = {"domain": store.ref(config.domain.data), "objects": store.ref(config.objects)}
    meta = store.manifest(ref).meta
    skipped = {"skipped": str(meta.get("n_skipped", 0)), "skipped.objects": json.dumps(meta.get("skipped", {}))}
    if "notes" in meta:  # the extractor's per-object notes (ExtractedBatch.notes), only where it wrote any
        skipped |= {"notes": str(meta["n_notes"]), "notes.objects": json.dumps(meta["notes"])}
    execution = {**{f"fingerprint.{k}": str(v) for k, v in fingerprint.items()}, "worker.device": device, **skipped}
    execution["origin"] = canonical_json(store.origin(job.key()))
    return StageResult(outputs={"embeddings": ref}, inputs=inputs, devices=devices, execution=execution)


STAGE = Stage("extract", ExtractConfig, _run, lambda c: ExecutionTarget(c.device_class), outputs="embeddings")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.extract --config <yaml> [--pin NAME [--repin REASON]] [--option K=V]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
