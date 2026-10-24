"""The one trainer: any connector, any receiver training backend, any mixture of registered streams.

Per step ``s`` (0-based; the run has done ``s`` updates):

1. the sampler's examples of step ``s`` (``data.sampler``: a function of the run seed and ``s`` alone), rendered;
2. per micro-batch (``RunConfig.micro_batch``, in order): the bound objects' inputs, the connector's writes (each
   entry's map on the device of its layer, from the backend's ``placement()``), resolved per layer, and the backend's
   loss terms; the gradient of ``Σ w·ℓ / Σ w`` over the whole batch is accumulated, with ``Σ w`` taken from the
   rendered targets and checked against the backend's sums;
3. the α budget penalty, the global gradient-norm clip, and one AdamW update at ``OptimizerSpec.factor(s)`` times each
   group's base rate (``lr_rel × lr_reference`` per tensor, from each map kind and the step-0 values; ``lr_alpha`` for
   the α logits), with the decayed tensors (``EntryMap.decayed``) first scaled by ``1 − factor(s)·weight_decay``.

**Step 0.** The step-0 connector is ``build.init_state.initial_connector`` on ``RunConfig.init_spec()``, the one
initial-state procedure shared with the ``init`` stage; the trainer commits it as its step-0 checkpoint and runs the
in-job evaluation on it before the first update.

**Checkpoints and resume.** Committed checkpoints (``writes.checkpoint.commit_checkpoint``, weights only) at step 0,
at every ``checkpoint_every`` step (None: every ``eval_every`` step) and at the end; in-job evaluations at step 0,
every ``eval_every`` step and the end. The resume state (connector, optimizer, step, the committed checkpoints) is
saved every ``CheckpointSpec.resume_every`` steps and, with ``on_signal``, after the step in which SIGUSR1 arrived,
after which :meth:`Trainer.fit` raises :class:`Interrupted`. A run restarted on the same run directory resumes from
it, and because every draw is step-indexed and the update is deterministic, it continues bit-for-bit as the
uninterrupted run would have (on CPU; on GPUs as far as their kernels are deterministic, and only on the same
accelerator model: a resume on another model is allowed and continues the same recipe without a bit-identity
claim). At the run's true final step, after its final checkpoint, the resume state is saved once more, atomically and
without a log line, and only then does the run directory record the completed run (:data:`DONE_FILE`), so a relaunch
returns its checkpoints instead of training again; a save that fails leaves no :data:`DONE_FILE`, and the next job
resumes from the last state saved. The kept final state is what an extension continues (below); it costs the disk of
one resume state per completed run (the connector's weights and the optimizer's two moments: about three times the
weights). When a run starts, the run directory also gets what its streams derived from the training text
(:data:`DERIVED_FILE`); every :meth:`Trainer.fit`, a resume or a relaunch of a completed run, refuses to continue if
the streams now derive anything else, and the resume state carries its digest. It also carries the execution (the
backend's ``execution()`` and the options: :meth:`Trainer.execution`), as does the completed run's :data:`DONE_FILE`,
and a resume or a relaunch under another one (e.g. ``kimi-ep`` with another ``checkpointing`` option, another dtype,
attention implementation or device count) is refused, so the execution the run's record states is every job's. Not
compared (:data:`RESUMABLE_ACROSS`): the weights directory (``execution()["weights"]``: the weights are identified by
the pinned ``revision``, ``build.capture.capture_origin``, and node-local copies differ) and the accelerator model
(``execution()["device_name"]``): every segment records its own devices.

**Extending a finished run.** :meth:`Trainer.extend` initializes the run directory of a run from the finished run of
the config that equals it but for fewer ``steps`` (the ``train`` stage's ``extend_from`` option, ``run.train``, which
authenticates the source's record first): the source's kept final state, log, derived data and segment ledger are
copied, never its :data:`DONE_FILE`, with :data:`EXTENSION_FILE` naming the source (``provenance.Extension``) written
last, into a staging directory that is then renamed to the run directory, so the directory appears initialized or not
at all and the source is left as it was. :meth:`Trainer.fit` then resumes it as any run: at the source's final step,
with its committed checkpoints, optimizer moments and sampler position (the step), and the earlier jobs in its ledger.
Under a deterministic execution (the ``hf`` backend on CPU, as ``tests/unit/test_extend_from.py`` checks; CPU alone does
not make every plugin deterministic) the result is the straight longer run's weights and updates; other executions
continue the same recipe as far as they are deterministic, as a resume does. A source whose final step is off the
checkpoint cadence adds its final checkpoint to the extended run's (a straight run has it only on the cadence). A
schedule that reads the total (``OptimizerSpec.depends_on_total``) is refused.

**Segments.** Each :meth:`Trainer.fit` is one job's share of the run (``provenance.Segment``), and it alone
records the job in the run directory's segment ledger (:data:`SEGMENTS_FILE`, every job that worked on the run): when
it starts, at each resume-state save, and when it leaves however it leaves (completion, a graceful stop, a failure),
so a stopped job counts to its true end and one killed without notice to the state the next job resumes from. The
earlier segments stay in the trainer's ``resumed`` for the run's record (``Provenance.resumed``), so the record's cost
covers every job, not only the last. Segments may run on different accelerator models; each keeps its own
devices, so a run's hardware history is its ledger's, never assumed.

**Log.** One JSON line per update in ``<run directory>/log.jsonl`` (:data:`LOG_FILE`): the objective, the plain token
mean, the position bins of the original log (:data:`POSITION_BINS`), per-stream losses, the α values and their sum, the
penalty, the gradient norm, the rate factor, each map tensor's relative drift from step 0, token and stream counts.
Each update also prints ``train: step N/STEPS loss L loss_w W`` (the line's ``loss`` and ``loss_w``) to stderr;
nothing recorded reads it.
In-job evaluations add ``eval`` lines; committed checkpoints ``checkpoint`` lines and resume-state saves
``resume_state`` lines, each with the seconds it took, so time spent outside the updates shows in the log. A resumed
run drops the lines after its resume step, so the log reads as one run; the dropped lines are first copied to a new
file, ``log.discarded.<UTC time>.<pid>.jsonl`` (:data:`DISCARDED_PREFIX`), never overwritten, and the log is then
replaced atomically.

**One producer, and continuations.** The train stage holds an exclusive ``flock`` on the run directory's
:data:`LOCK_FILE` while it trains (:func:`run_directory_lock`): from before it reads the ledger until the training log
is committed, released before ``run._main.stage_main`` writes the record and the pins. A second job on the same run
directory, under any job name, is refused (:class:`RunDirectoryBusyError`) before it reads a run file or writes
anything else there. A
continuation job (``fit(..., resume_required=True)``) continues the run or does nothing: with no resume state, one that
does not load or one that fails the resume checks (connector config, derived data, execution, optimizer groups) it is
refused (:class:`ResumeRequiredError`) before :meth:`Trainer.fit` writes anything, so it never starts at step 0. Every
check of a resume happens before the job's first write, in either mode.

**A stopped run's achievement.** :func:`achieved_run` reads, without writing or locking, what a run whose
jobs ended short of its steps achieved: its resume state's committed checkpoints, checked against the store, the log
and the run's own connector config and derived data; the ``achieved`` stage publishes them (``run.achieved``).

Collaborators are injected, so tests use fakes. :meth:`Trainer.from_config` wires the real ones. In-job
evaluation is a callback, so ``training`` does not depend on ``evaluation``. The evaluator is given the training
backend, which also scores, so no second model is loaded.
"""

from __future__ import annotations

import fcntl
import json
import os
import pickle
import shutil
import signal
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, get_args

import torch
from torch import Tensor

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore, IntegrityError
from latent_interconnect.build.init_state import initial_connector
from latent_interconnect.build.ridge import RIDGE_KIND, load_ridge
from latent_interconnect.build.zero_shot import check_construction, training_set_construction
from latent_interconnect.canonical import canonical_json, canonical_sha256
from latent_interconnect.data.items import ITEM_SET_KIND, load_item_set
from latent_interconnect.data.sampler import Sampler
from latent_interconnect.data.training_text import TRAINING_SET_KIND, load_training_text
from latent_interconnect.domains.base import open_domain, require_splits
from latent_interconnect.encoders.loader import Consumed, InputLoader
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import EmbeddingReader
from latent_interconnect.plugins import load_plugins
from latent_interconnect.provenance import Extension, Segment
from latent_interconnect.receivers import open_renderer, open_training
from latent_interconnect.receivers.base import ScoringBackend, TrainingBackend
from latent_interconnect.receivers.kernel import resolve_sources
from latent_interconnect.receivers.render import RenderedPrompt
from latent_interconnect.training.config import RunConfig
from latent_interconnect.writes.checkpoint import (
    RESUME_FILE,
    ResumeState,
    commit_checkpoint,
    load_resume_state,
    read_resume_summary,
    save_resume_state,
)
from latent_interconnect.writes.connector import Connector
from latent_interconnect.writes.inputs import ObjectBatch
from latent_interconnect.writes.kinds import EntryMap
from latent_interconnect.writes.sources import LiveSource

__all__ = [
    "DEFAULT_INPUT_CACHE",
    "DERIVED_FILE",
    "DISCARDED_PREFIX",
    "DONE_FILE",
    "EXTENSION_FILE",
    "EXTENSION_STAGING",
    "LOCK_FILE",
    "LOG_FILE",
    "POSITION_BINS",
    "RESUMABLE_ACROSS",
    "SEGMENTS_FILE",
    "AchievedRun",
    "CompletedRun",
    "ExtensionError",
    "InJobEvaluator",
    "InputCache",
    "Interrupted",
    "ResumeRequiredError",
    "RunDirectoryBusyError",
    "Trainer",
    "achieved_run",
    "completed_run",
    "earlier_segments",
    "input_loader",
    "place_connector",
    "read_extension",
    "run_directory_lock",
]

LOG_FILE = "log.jsonl"
"""The run's training log, in its run directory."""
DERIVED_FILE = "derived.json"
"""What the run's streams derived from the training text (``Sampler.derived``; the enabled grammar formats), written
in its run directory when the run starts and checked on every ``fit``; the train stage commits this file."""
DONE_FILE = "done.json"
"""The completed run's committed checkpoints (``committed``: step → digest) and its ``execution``, in its run
directory, written (atomically) after its kept final resume state: a relaunch under the same execution returns them."""
EXTENSION_FILE = "extension.json"
"""In the run directory of an extended run: its source (``provenance.Extension``), written last when
:meth:`Trainer.extend` initializes the directory, so it marks a completed initialization and keeps the run's lineage
for every later job, with or without the ``extend_from`` option."""
EXTENSION_STAGING = ".extend."
"""The prefix of the staging directory :meth:`Trainer.extend` fills beside the run directory before renaming it
(``.extend.<run digest>.<source digest>.<hex>``); one a killed initialization left is removed by the next one."""
SEGMENTS_FILE = "segments.json"
"""The run's segment ledger, in its run directory: every job that worked on the run (``provenance.Segment``)."""
LOCK_FILE = "train.lock"
"""The run directory's producer lock (:func:`run_directory_lock`): never read for content by any other code, and
outside the names the trainer itself writes or clears (``resume.pt``, ``resume.pt.<hex>``, ``.segments.json.<pid>``,
``.log.jsonl.<pid>``, ``.done.json.<pid>``)."""
DISCARDED_PREFIX = "log.discarded."
"""The prefix of a file holding log lines a resume dropped (``log.discarded.<UTC time>.<pid>.jsonl``)."""
_RESUME_ERRORS = (ValueError, TypeError, KeyError, RuntimeError, OSError, EOFError, pickle.UnpicklingError)
"""What loading or checking a resume state raises when the state is unreadable or incompatible."""
RESUMABLE_ACROSS = frozenset({"weights", "device_name", "input_cache"})
"""The execution fields a resume or relaunch may change (``Trainer._check_execution``): the weights directory, the
accelerator model and the input cache (:data:`InputCache`: an I/O policy over immutable artifacts that loads the same
bytes either way, tested). Every other field (backend, dtype, quantization, attention, device count, revision,
backend options) must equal the run's."""
InputCache = Literal["off", "on"]
"""The trainer's ``input_cache`` (``--option input_cache=...`` of ``train``): ``"on"`` (:data:`DEFAULT_INPUT_CACHE`)
keeps the embedding readers' memory maps for the job (``EmbeddingReader``'s ``keep_maps``), ``"off"`` maps the part
files on every batch. The batches are the same bytes either way. :meth:`Trainer.execution` records the effective value
in every segment's facts, and a resume may change it (:data:`RESUMABLE_ACROSS`). It uses host resources only, which no
device memory plan covers: one map and one file descriptor per part file read (295 for a Kimi K2.6 run) and address
space; the pages read stay the operating system's reclaimable page cache (``EmbeddingReader`` states the bound)."""
DEFAULT_INPUT_CACHE: InputCache = "on"
"""The input cache's default, for every backend: the one source of it."""
_INPUT_CACHES = get_args(InputCache)
POSITION_BINS: dict[str, tuple[int, int | None]] = {"1-4": (0, 4), "5-8": (4, 8), "9-32": (8, 32), "33+": (32, None)}
"""Target-token position bins of the logged positional losses (0-based ``[lo, hi)``; originally ``POS_BINS``)."""
_WEIGHT_TOLERANCE = 1e-5
"""Relative tolerance of the check that the backend's Σ w equals the rendered targets' (float32 sums)."""


class InJobEvaluator(Protocol):
    """Scores a checkpoint during training (the run's in-job suite)."""

    def __call__(
        self, backend: ScoringBackend, connector: Connector, checkpoint: ArtifactRef, step: int
    ) -> ArtifactRef:
        """Score through the given backend; return the ``scores`` artifact."""
        ...


class ResumeRequiredError(ValueError):
    """A continuation job (``resume_required``) found no resume state it can continue; nothing was written."""


class RunDirectoryBusyError(RuntimeError):
    """Another process holds the run directory's :data:`LOCK_FILE`: one producer per run directory."""


class ExtensionError(ValueError):
    """A run cannot be extended from the run named as its source (:meth:`Trainer.extend`); nothing was written."""


@contextmanager
def run_directory_lock(workspace: Path) -> Iterator[None]:
    """Hold an exclusive, non-blocking ``flock`` on ``workspace``/:data:`LOCK_FILE` for the ``with`` block.

    The directory and the lock file are made if missing; the lock file is never deleted. The holder writes its host
    and process id into the file, which the refusal shows. The lock ends with the process, however it ends.

    Raises:
        RunDirectoryBusyError: if another open file (another process) holds the lock.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    fd = os.open(workspace / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as err:
            holder = os.pread(fd, 256, 0).decode(errors="replace").strip() or "unknown"
            raise RunDirectoryBusyError(
                f"{workspace / LOCK_FILE} is held by another producer ({holder}); one job per run directory"
            ) from err
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"host={socket.gethostname()} pid={os.getpid()}\n".encode(), 0)
        yield
    finally:
        os.close(fd)  # releases the lock


class Interrupted(Exception):  # noqa: N818  (a requested stop, not an error)
    """SIGUSR1 arrived: the resume state was saved after ``step`` updates and the run stopped.

    Attributes:
        step: the updates completed and saved.
    """

    def __init__(self, step: int) -> None:
        super().__init__(f"stopped by SIGUSR1 after step {step}; resume state saved")
        self.step = step


def _owning_kind(name: str, kinds: Mapping[str, EntryMap]) -> str:
    """Return the module path of the outermost map kind enclosing parameter ``name``.

    Raises:
        TypeError: if no map kind encloses it (it would have no learning-rate reference).
    """
    parts = name.split(".")
    for i in range(1, len(parts)):
        if (prefix := ".".join(parts[:i])) in kinds:
            return prefix
    raise TypeError(f"parameter {name!r} belongs to no map kind, so it has no learning-rate reference")


def _segments(workspace: Path) -> tuple[Segment, ...]:
    """Return the segment ledger of the run in ``workspace``, oldest first; () without one.

    Raises:
        ValueError: if the ledger is not a list of segments.
    """
    path = workspace / SEGMENTS_FILE
    if not path.exists():
        return ()
    try:
        return tuple(Segment.model_validate(entry) for entry in json.loads(path.read_text()))
    except (ValueError, TypeError) as err:
        raise ValueError(f"{path} is not a segment ledger: {err}") from err


def _ledger(workspace: Path, segment: Segment) -> tuple[Segment, ...]:
    """Return the segments of the run in ``workspace`` before ``segment`` (this job), oldest first.

    The ledger's entry for this job itself (the same segment, at any end time) is not an earlier one. A run directory
    without a ledger has none.

    Raises:
        ValueError: if the ledger is not a list of segments.
    """
    return tuple(s for s in _segments(workspace) if s.model_copy(update={"ended": segment.ended}) != segment)


def earlier_segments(workspace: Path, segment: Segment) -> tuple[Segment, ...]:
    """Return the run's segments before ``segment``, as :meth:`Trainer.fit` reads them.

    The train stage reads the ledger before anything loads, so a broken one is refused early. Earlier segments on other
    hardware are not refused: each keeps its own devices.

    Raises:
        ValueError: if the ledger is not a list of segments.
    """
    return _ledger(workspace, segment)


@dataclass(frozen=True)
class AchievedRun:
    """What a stopped run achieved, read from its run directory (:func:`achieved_run`); nothing is trained or written.

    Attributes:
        step: the updates its resume state holds (the step a continuation would resume from): the achieved step.
        committed: step → committed checkpoint, the resume state's, every one up to ``step``.
        execution: the backend's ``execution()`` the run trained under (the resume state's).
        logged_step: the last step in its log (above ``step`` when a job died after its last resume-state save;
            those updates were never saved, and a continuation drops them). Never an achieved step.
        log: the log's lines up to ``step``, as a continuation keeps them (``Trainer._truncate_log``).
        derived: the bytes of :data:`DERIVED_FILE`, checked against the resume state.
    """

    step: int
    committed: Mapping[int, ArtifactRef]
    execution: Mapping[str, str]
    logged_step: int
    log: bytes
    derived: bytes


def achieved_run(config: RunConfig, store: ArtifactStore, workspace: Path) -> AchievedRun:
    """Return what the stopped run in ``workspace`` achieved: its resume state's committed checkpoints, checked.

    Read only (the resume state memory-mapped, :func:`~latent_interconnect.writes.checkpoint.read_resume_summary`):
    the run directory, its ledger and its log are left as they are, and no lock is taken (the ``achieved`` stage takes
    it, ``run.achieved``). It fails closed: the resume state must be the run's (its connector config and derived data)
    and record its execution; it must hold step 0's checkpoint and none past its step; every checkpoint must be
    committed in ``store`` and intact; the resume state and every log line must parse, and the log's ``checkpoint``
    lines up to that step must name exactly those checkpoints.

    Raises:
        ValueError: if the run directory holds no resume state, or a completed run (:data:`DONE_FILE`), or any check
            above fails (a store integrity error and a malformed log line included).
    """
    if (workspace / DONE_FILE).exists():
        raise ValueError(f"run {config.name!r}: {workspace / DONE_FILE} exists: the run is complete, not stopped")
    try:
        summary = read_resume_summary(workspace)
    except (RuntimeError, EOFError, KeyError, ValueError, pickle.UnpicklingError) as err:  # a torn or foreign file
        raise ValueError(f"run {config.name!r}: its {RESUME_FILE} cannot be read: {err!r}") from err
    if summary is None:
        raise ValueError(f"run {config.name!r}: {workspace} holds no {RESUME_FILE}: no achieved state to read")
    if summary.connector_sha != config.connector.digest():
        raise ValueError(f"run {config.name!r}: its resume state was saved for another connector config")
    derived = (workspace / DERIVED_FILE).read_bytes()
    if summary.extra.get("derived_sha256") != canonical_sha256(json.loads(derived)):
        raise ValueError(f"run {config.name!r}: its resume state does not hold the derived data of {DERIVED_FILE}")
    execution = summary.extra.get("execution")
    if not isinstance(execution, dict) or not execution:
        raise ValueError(f"run {config.name!r}: its resume state records no execution")
    try:
        committed = {int(s): store.ref(sha) for s, sha in summary.extra["committed"].items()}
    except (KeyError, FileNotFoundError, ValueError, IntegrityError) as err:
        raise ValueError(f"run {config.name!r}: a committed checkpoint cannot be read: {err}") from err
    if 0 not in committed or max(committed) > summary.step:
        raise ValueError(
            f"run {config.name!r}: its committed steps {sorted(committed)} lack step 0 or pass its step {summary.step}"
        )
    text = (workspace / LOG_FILE).read_text()
    try:
        lines = [(ln, json.loads(ln)) for ln in text.splitlines() if ln]
        steps = [int(x["step"]) for _, x in lines]
        logged = {
            x["step"]: x["checkpoint"] for _, x in lines if x["event"] == "checkpoint" and x["step"] <= summary.step
        }
    except (ValueError, KeyError, TypeError) as err:
        raise ValueError(f"run {config.name!r}: a line of {LOG_FILE} is malformed: {err!r}") from err
    if logged != {s: r.sha256 for s, r in committed.items()}:
        raise ValueError(
            f"run {config.name!r}: the log's checkpoints up to step {summary.step} ({sorted(logged)}) are not the "
            f"resume state's ({sorted(committed)})"
        )
    kept = "".join(f"{ln}\n" for (ln, _), s in zip(lines, steps, strict=True) if s <= summary.step).encode()
    executed = {str(k): str(v) for k, v in execution.items()}
    return AchievedRun(summary.step, committed, executed, max(steps, default=0), kept, derived)


@dataclass(frozen=True)
class CompletedRun:
    """A finished run with its kept final state, read from its run directory (:func:`completed_run`).

    Attributes:
        step: its final step (its config's ``steps``).
        committed: step → committed checkpoint digest, :data:`DONE_FILE`'s and its kept state's.
        execution: the execution it trained under (:data:`DONE_FILE`'s and its kept state's).
        derived: the bytes of :data:`DERIVED_FILE`.
        log: the bytes of :data:`LOG_FILE`.
    """

    step: int
    committed: Mapping[int, str]
    execution: Mapping[str, str]
    derived: bytes
    log: bytes


def completed_run(config: RunConfig, workspace: Path) -> CompletedRun:
    """Return the finished run of ``config`` in ``workspace``, checked; read only (no lock, nothing written).

    It fails closed: :data:`DONE_FILE` must record the committed checkpoints from step 0 to ``config.steps`` and an
    execution; the kept resume state must be at ``config.steps``, saved for the config's connector, and hold the same
    checkpoints, the same execution and the digest of :data:`DERIVED_FILE`; every line of :data:`LOG_FILE` must parse
    and its ``checkpoint`` lines must name exactly those checkpoints.

    Raises:
        ExtensionError: if a file is missing or unreadable, or a check above fails (a run completed without a kept
            final state included).
    """
    name = config.name
    try:
        done = json.loads((workspace / DONE_FILE).read_text())
        committed = {int(s): str(d) for s, d in done["committed"].items()}
        execution = {str(k): str(v) for k, v in done["execution"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as err:
        raise ExtensionError(f"run {name!r}: {workspace / DONE_FILE} does not record a completed run: {err!r}") from err
    if 0 not in committed or max(committed) != config.steps:
        raise ExtensionError(
            f"run {name!r}: its committed steps {sorted(committed)} are not a run completed at {config.steps}"
        )
    try:
        summary = read_resume_summary(workspace)
    except (RuntimeError, EOFError, KeyError, ValueError, pickle.UnpicklingError) as err:
        raise ExtensionError(f"run {name!r}: its {RESUME_FILE} cannot be read: {err!r}") from err
    if summary is None:
        raise ExtensionError(f"run {name!r}: {workspace} holds no kept final state ({RESUME_FILE})")
    if summary.step != config.steps or summary.connector_sha != config.connector.digest():
        raise ExtensionError(
            f"run {name!r}: its {RESUME_FILE} is at step {summary.step} for connector config "
            f"{summary.connector_sha[:12]}, not the final state of step {config.steps} for "
            f"{config.connector.digest()[:12]}"
        )
    if summary.extra.get("committed") != {str(s): d for s, d in committed.items()}:
        raise ExtensionError(f"run {name!r}: its {RESUME_FILE} and {DONE_FILE} hold other checkpoints")
    if summary.extra.get("execution") != execution:
        raise ExtensionError(f"run {name!r}: its {RESUME_FILE} and {DONE_FILE} record other executions")
    try:
        derived = (workspace / DERIVED_FILE).read_bytes()
        log = (workspace / LOG_FILE).read_bytes()
        derived_sha = canonical_sha256(json.loads(derived))
        lines = [json.loads(ln) for ln in log.decode().splitlines() if ln]
        logged = {int(x["step"]): x["checkpoint"] for x in lines if x["event"] == "checkpoint"}
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise ExtensionError(f"run {name!r}: its {DERIVED_FILE} or {LOG_FILE} cannot be read: {err!r}") from err
    if summary.extra.get("derived_sha256") != derived_sha:
        raise ExtensionError(f"run {name!r}: its {RESUME_FILE} does not hold the derived data of {DERIVED_FILE}")
    if logged != committed:
        raise ExtensionError(f"run {name!r}: the checkpoints of its {LOG_FILE} are not its committed checkpoints")
    return CompletedRun(config.steps, committed, execution, derived, log)


def read_extension(workspace: Path) -> Extension | None:
    """Return the source of the extended run in ``workspace`` (its :data:`EXTENSION_FILE`), or None for another run.

    Raises:
        ValueError: if the file does not hold an extension's source.
    """
    path = workspace / EXTENSION_FILE
    if not path.is_file():
        return None
    return Extension.model_validate_json(path.read_text())


def _replace(path: Path, text: str) -> None:
    """Atomically replace ``path`` with ``text`` (written beside it as ``.<name>.<pid>``, synced, renamed over it)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    with tmp.open("w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _sync_file(path: Path) -> None:
    """Flush ``path`` to disk."""
    with path.open("rb") as fh:
        os.fsync(fh.fileno())


def _sync_directory(path: Path) -> None:
    """Flush the entries of the directory ``path`` (a rename in it) to disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_ledger(workspace: Path, segments: Sequence[Segment]) -> None:
    """Atomically replace the run's segment ledger."""
    _replace(workspace / SEGMENTS_FILE, json.dumps([s.model_dump(mode="json") for s in segments]))


def input_loader(
    config: RunConfig, store: ArtifactStore, connector: Connector, *, input_cache: InputCache = DEFAULT_INPUT_CACHE
) -> InputLoader:
    """Return the loader of the run's encoder inputs: every embedding artifact the run cites, as its connector reads.

    It refuses an object whose value in the run's domain is not the one its artifact was extracted from
    (``encoders.loader.Consumed``). With ``input_cache="on"`` its readers keep their memory maps (:data:`InputCache`).
    """
    readers = [EmbeddingReader(store, store.ref(d), keep_maps=input_cache == "on") for d in config.embeddings]
    values = open_domain(config.domain, store).objects()["value"].to_dict()
    consumed = Consumed(f"{config.domain.name} {config.domain.data[:12]}", values, store.root)
    return InputLoader(
        config.connector.input, readers, connector.input_kinds, encoder=config.encoder, consumed=consumed
    )


class Trainer:
    """Runs one training run.

    Args:
        config: the run config.
        backend: the receiver's training backend.
        connector: the step-0 connector (from ``build.init_state``); a resume state is loaded into it.
        load: object ids → their encoder inputs.
        sampler: the step-indexed data sampler.
        store: the artifact store for committed checkpoints.
        workspace: the run directory (``Paths.runs / <run digest>``): resume state, log and the completion record.
        evaluator: the in-job evaluator, or None.
        stop: set to stop after the current step (SIGUSR1 sets it, ``CheckpointSpec.on_signal``); the stage passes the
            event its own handler sets from the start of the job, so a signal during model loading is not lost.
        append_encoder: the recipe of the input's appended extraction (``InputLoader.append``), recorded in
            every committed checkpoint for the suite's loader, or None.
        input_cache: whether ``load`` keeps its readers' memory maps (:data:`InputCache`), recorded in
            :meth:`execution`; any other value is refused here (``ValueError``).

    Attributes:
        resumed: after :meth:`fit`, the run's segments before this job's, oldest first (``Provenance.resumed``).
    """

    def __init__(
        self,
        *,
        config: RunConfig,
        backend: TrainingBackend,
        connector: Connector,
        load: Callable[[Sequence[str]], ObjectBatch],
        sampler: Sampler,
        store: ArtifactStore,
        workspace: Path,
        evaluator: InJobEvaluator | None = None,
        stop: threading.Event | None = None,
        append_encoder: EncoderRecipe | None = None,
        input_cache: InputCache = DEFAULT_INPUT_CACHE,
    ) -> None:
        if input_cache not in _INPUT_CACHES:
            raise ValueError(f"input_cache must be one of {list(_INPUT_CACHES)}; got {input_cache!r}")
        self.append_encoder, self.input_cache = append_encoder, input_cache
        self.config, self.backend, self.connector, self.load = config, backend, connector, load
        self.sampler, self.store, self.workspace, self.evaluator = sampler, store, workspace, evaluator
        self.stop = threading.Event() if stop is None else stop
        self.derived = sampler.derived()  # what the streams derived from the training text
        self.resumed: tuple[Segment, ...] = ()
        named = connector.named_parameters()
        # the step-0 maps, for drift: taken here, before any resume state loads (the connector given is step 0), and
        # moved with the maps when they are placed (_place), so the drift is computed where each map lives
        self._initial = {n: p.detach().clone() for n, p in named if not n.startswith("theta.")}

    @classmethod
    def from_config(
        cls,
        config: RunConfig,
        store: ArtifactStore,
        workspace: Path,
        *,
        evaluator: InJobEvaluator | None = None,
        backend_options: Mapping[str, str] | None = None,
        stop: threading.Event | None = None,
        input_cache: InputCache = DEFAULT_INPUT_CACHE,
    ) -> Trainer:
        """Wire the real collaborators from the config (loads ``config.plugins`` and every embedded config's).

        Checked before the backend loads: the run's domain is the one its training set was built and screened on (the
        same name and data artifact; ``plugins`` only say what to import), so the construction check and the sampler
        read the same domain; the training text's objects are all in the ``train`` split, the mixture passes the
        zero-shot construction check (``build.zero_shot.check_construction``), every object the streams bind
        (``Sampler.objects``) is in ``train`` and loads by the input loader's holder rule (``InputLoader.missing``), and
        so does every object of the in-job suite's item set (``RunConfig.embeddings`` covers both).
        The backend is the receiver's ``train`` backend, built with the run's batch plan and ``backend_options``
        (execution options as text), plus the run's ``checkpointing`` when it sets one; the input loader keeps its
        readers' memory maps with ``input_cache="on"`` (:data:`InputCache`).

        Raises:
            ValueError: if ``backend_options`` repeats the run's ``checkpointing``, or the run's domain is not the one
                its training set was built and screened on (the same name and data artifact; ``plugins`` only say
                what to import; so the zero-shot construction check and the sampler read one domain), a training
                object or an object the streams bind is outside ``train``, an object the streams bind or an object of
                the in-job suite's item set cannot be loaded, the mixture supervises a zero-shot task, or a ridge
                was fit with another content of the appended extraction's recipe (when it recorded one).
        """
        load_plugins(config.plugins)
        text_ref = ArtifactRef(config.training_set, TRAINING_SET_KIND)
        built_on = training_set_construction(store, text_ref).domain  # an approved export's: its definition's
        if (built_on.name, built_on.data) != (config.domain.name, config.domain.data):  # sample where it was screened
            raise ValueError(
                f"run {config.name!r}: its domain {config.domain.name} {config.domain.data[:12]} is not the domain "
                f"its training set was built and screened on ({built_on.name} {built_on.data[:12]})"
            )
        domain = open_domain(config.domain, store)
        text = load_training_text(store, text_ref)
        require_splits(domain, text.object_ids, {"train"}, f"run {config.name!r} training text")
        check_construction(text_ref, config.mixture, store)
        connector = initial_connector(config.init_spec(), store)
        loader = input_loader(config, store, connector, input_cache=input_cache)
        if loader.append is not None:  # the ridges' appended extraction is the run's, by recipe content
            for d in config.ridge.values():
                load_ridge(store, ArtifactRef(d, RIDGE_KIND)).require_fit(
                    f"run {config.name!r}",
                    input=config.connector.input,
                    recipe_digest=config.encoder.digest(),
                    receiver={"model_id": config.receiver.model_id, "revision": config.receiver.revision},
                    append_digest=loader.append.encoder.digest(),
                )
        sampler = Sampler(
            mixture=config.mixture,
            domain=domain,
            renderer=open_renderer(config.receiver),
            batch=config.batch,
            seed=config.seed,
            text=text,
            connector=config.connector.name,
            k=config.connector.k,
            thinking_close=config.receiver.thinking_close,
            separator=config.receiver.separator,
        )
        bound = sorted(sampler.objects())  # the training text plus what streams bind beyond it (wild-type parents)
        require_splits(domain, bound, {"train"}, f"run {config.name!r} objects the streams bind")
        needed = {"objects the streams bind": bound}
        if config.eval_suite is not None:  # the in-job items too, here and not at the step-0 evaluation
            items = load_item_set(store, ArtifactRef(config.eval_suite.item_set, ITEM_SET_KIND))
            scored = {o for slot in config.eval_suite.slots for o in items.objects[slot.role]}
            needed[f"objects of the in-job items {items.name!r}"] = sorted(scored)
        for what, ids in needed.items():
            missing = loader.missing(ids)
            if missing:  # refused at load (the loader's own rule), not by a KeyError at the step that first draws one
                shown = list(missing.values())[:3]
                searched = ", ".join(d[:12] for d in config.embeddings)
                raise ValueError(
                    f"run {config.name!r}: {len(missing)} {what} cannot be loaded from {searched}: {shown}"
                )
        options = dict(backend_options or {})
        if config.checkpointing is not None:  # the run's own setting: never also an option
            if "checkpointing" in options:
                raise ValueError(f"run {config.name!r}: checkpointing is set in the run config; drop the option")
            options["checkpointing"] = str(config.checkpointing).lower()
        backend = open_training(config.receiver, plan=config.plan, options=options or None)
        return cls(
            config=config,
            backend=backend,
            connector=connector,
            load=loader.batch,
            sampler=sampler,
            store=store,
            workspace=workspace,
            evaluator=evaluator,
            stop=stop,
            append_encoder=None if loader.append is None else loader.append.encoder,
            input_cache=input_cache,
        )

    def fit(self, segment: Segment, *, resume_required: bool = False) -> Mapping[int, ArtifactRef]:
        """Train to ``config.steps``; return step → committed checkpoint. Resumes automatically from the run directory.

        The resume state is loaded and checked (:meth:`_restore`) before the job's first write to the run directory.

        Args:
            segment: this job (``Provenance.segment`` of the stage's provenance); it joins the run's segment ledger,
                ended when the job starts, at each resume-state save and when ``fit`` returns or raises, and the
                earlier segments are left in :attr:`resumed`.
            resume_required: a continuation job: a completed run returns its checkpoints, any other run must resume
                from a resume state that loads and passes the resume checks; it never starts at step 0.

        Raises:
            Interrupted: after saving the resume state, when SIGUSR1 arrived (``CheckpointSpec.on_signal``).
            ResumeRequiredError: with ``resume_required``, if the run is not complete and has no resume state that
                loads and passes the checks; nothing was written.
        """
        optimizer = self.optimizer()
        base = [g["lr"] for g in optimizer.param_groups]  # before a resume state's (scheduled) rates load
        try:
            restored = self._restore(optimizer)
        except _RESUME_ERRORS as err:
            if resume_required:
                raise ResumeRequiredError(
                    f"run {self.config.name!r}: its resume state cannot continue it: {err}"
                ) from err
            raise
        if resume_required and restored is None and not (self.workspace / DONE_FILE).exists():
            raise ResumeRequiredError(
                f"run {self.config.name!r}: {self.workspace} has no resume state and no {DONE_FILE}; a continuation "
                "never starts a run at step 0"
            )
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.resumed = _ledger(self.workspace, segment)

        def mark() -> None:  # this job's ledger entry, ended now
            _write_ledger(self.workspace, (*self.resumed, segment.touch()))

        mark()
        try:
            return self._train(mark, optimizer, base, restored)
        finally:
            mark()

    def extend(self, source: RunConfig, directory: Path, *, computation_key: str, record_id: str) -> Extension:
        """Initialize this run's directory from the finished run of ``source`` in ``directory``; return its origin.

        The caller has authenticated ``directory`` as the run of the ``train`` record ``record_id`` (computation
        ``computation_key``) of ``source`` (``run.train``). Checked first, writing nothing: this run's directory does
        not exist; ``source`` is this run's config with fewer ``steps`` and no other change; the schedule does not read
        the total (``OptimizerSpec.depends_on_total``); the source is a finished run with its kept final state
        (:func:`completed_run`), whose derived data the streams derive now and whose execution this job may resume
        (:meth:`_check_execution`, with its permitted differences, :data:`RESUMABLE_ACROSS`).

        Then, under the source's producer lock (:func:`run_directory_lock`), the source is read again and must be
        unchanged; its resume state, log, derived data and segment ledger (never its :data:`DONE_FILE`) are copied into
        a new staging directory beside this run's (:data:`EXTENSION_STAGING`), synced, with :data:`EXTENSION_FILE`
        written last, and the staging directory is renamed to this run's directory. So the run directory appears
        initialized or not at all, and the source keeps every byte (its lock file records this job as its last holder,
        as for any job on it). Staging directories of the same source and run that a killed initialization left are
        removed first, under the same lock.

        Returns:
            The origin written to :data:`EXTENSION_FILE` (``provenance.Extension``): the source's config digest,
            computation key, record id, final step and its number of ledger segments.

        Raises:
            ExtensionError: if a check fails, the source changed meanwhile, or this run's directory appeared meanwhile;
                this run's directory was not written.
            RunDirectoryBusyError: if another job holds the source's lock.
        """
        cfg, name = self.config, self.config.name
        if self.workspace.exists():
            raise ExtensionError(f"run {name!r}: {self.workspace} exists; an extension starts a new run directory")
        if source.steps >= cfg.steps or source.model_copy(update={"steps": cfg.steps}).digest() != cfg.digest():
            raise ExtensionError(
                f"run {name!r}: the source config ({source.steps} steps) is not this config with fewer steps "
                f"({cfg.steps}) and nothing else changed"
            )
        if cfg.optimizer.depends_on_total():
            raise ExtensionError(
                f"run {name!r}: its {cfg.optimizer.schedule} schedule reads the total steps, so the source's updates "
                "are not this run's"
            )
        found = completed_run(source, directory)
        derived = canonical_sha256(json.loads(found.derived))
        if derived != canonical_sha256(self.derived):
            raise ExtensionError(
                f"run {name!r}: the source trained on derived data {derived[:12]}, the streams now derive "
                f"{canonical_sha256(self.derived)[:12]}"
            )
        try:
            self._check_execution(found.execution)
        except ValueError as err:
            raise ExtensionError(f"the source's execution cannot be continued by this job: {err}") from err
        parent = self.workspace.parent
        prefix = f"{EXTENSION_STAGING}{self.workspace.name}.{directory.name}."
        with run_directory_lock(directory):
            if completed_run(source, directory) != found:
                raise ExtensionError(f"run {name!r}: the source run in {directory} changed while it was checked")
            for orphan in parent.glob(f"{prefix}*"):  # a killed initialization's; the lock excludes a live one
                shutil.rmtree(orphan)
            staging = parent / f"{prefix}{uuid.uuid4().hex}"
            staging.mkdir()
            try:
                origin = Extension(
                    config_sha=source.digest(),
                    computation_key=computation_key,
                    record_id=record_id,
                    step=found.step,
                    segments=self._stage_extension(directory, staging, found),
                )
                path = staging / EXTENSION_FILE
                path.write_text(canonical_json(origin.model_dump(mode="json")))
                _sync_file(path)
                _sync_directory(staging)
                if self.workspace.exists():
                    raise ExtensionError(f"run {name!r}: {self.workspace} appeared while it was initialized")
                try:
                    staging.rename(self.workspace)
                except OSError as err:
                    raise ExtensionError(f"run {name!r}: {self.workspace} appeared while it was initialized") from err
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            _sync_directory(parent)
        return origin

    def _stage_extension(self, directory: Path, staging: Path, found: CompletedRun) -> int:
        """Copy the source's run files but :data:`DONE_FILE` into ``staging``, synced and checked (:meth:`extend`).

        Returns:
            The number of segments in the copied ledger.
        """
        names = [RESUME_FILE, LOG_FILE, DERIVED_FILE]
        if (directory / SEGMENTS_FILE).exists():
            names.append(SEGMENTS_FILE)
        for n in names:
            shutil.copyfile(directory / n, staging / n)
            _sync_file(staging / n)
        summary = read_resume_summary(staging)
        if summary is None or summary.step != found.step or (staging / LOG_FILE).read_bytes() != found.log:
            raise ExtensionError(f"run {self.config.name!r}: the copy of the source in {staging} is not the source")
        return len(_segments(staging))

    def _restore(self, optimizer: torch.optim.Optimizer) -> tuple[int, dict[int, ArtifactRef]] | None:
        """Load the run's resume state into the connector and ``optimizer``, checked; writes nothing.

        Returns:
            The step and the committed checkpoints to continue from, or None for a completed run or one without a
            resume state.

        Raises:
            ValueError: if the state was saved for another connector config, other derived data or another execution;
                or what loading an unreadable state or another optimizer's groups raises (:data:`_RESUME_ERRORS`).
        """
        if (self.workspace / DONE_FILE).exists():
            return None
        state = load_resume_state(self.connector, self.workspace)
        if state is None:
            return None
        self._check_derived(state)
        self._check_execution(state.extra.get("execution"))
        self.connector.load_state_dict(state.connector)
        optimizer.load_state_dict(state.optimizer)
        return state.step, {int(s): self.store.ref(sha) for s, sha in state.extra["committed"].items()}

    def _train(
        self,
        mark: Callable[[], None],
        optimizer: torch.optim.Optimizer,
        base: Sequence[float],
        restored: tuple[int, dict[int, ArtifactRef]] | None,
    ) -> dict[int, ArtifactRef]:
        """:meth:`fit` within the job's ledger entry; ``mark`` renews its end at each periodic resume-state save."""
        cfg = self.config
        done = self.workspace / DONE_FILE
        if restored is None:
            if not done.exists():  # a new run: what the streams derived is fixed at its start
                (self.workspace / DERIVED_FILE).write_text(canonical_json(self.derived))
            self._check_derived(None)
        if done.exists():  # a completed run is not retrained: its committed checkpoints are its result
            body = json.loads(done.read_text())
            self._check_execution(body.get("execution"))
            return {int(k): self.store.ref(v) for k, v in body["committed"].items()}
        committed: dict[int, ArtifactRef] = {}
        with self._signal():
            if restored is None:
                step = 0
                self._truncate_log(-1)
                self._commit(0, committed)
            else:
                step, committed = restored
                self._truncate_log(step)
            while step < cfg.steps:
                if self.stop.is_set():
                    self._save(optimizer, step, committed)
                    raise Interrupted(step)
                factor = cfg.optimizer.factor(step, cfg.steps)
                for group, lr in zip(optimizer.param_groups, base, strict=True):
                    group["lr"] = lr * factor
                record = self._update(step, optimizer)
                step += 1
                self._log({"event": "train", "step": step, "lr_factor": factor, **record})
                sys.stderr.write(
                    f"train: step {step}/{cfg.steps} loss {record['loss']:.4f} loss_w {record['loss_w']:.4f}\n"
                )
                every = cfg.checkpoint_every or cfg.eval_every
                if step == cfg.steps or (every is not None and step % every == 0):
                    evaluate = step == cfg.steps or (cfg.eval_every is not None and step % cfg.eval_every == 0)
                    self._commit(step, committed, evaluate)
                if step < cfg.steps and step % cfg.checkpoints.resume_every == 0:
                    self._save(optimizer, step, committed)
                    mark()  # a job killed without notice counts to the state the next job resumes from
            self._keep(optimizer, step, committed)  # the true final state, before the run is declared complete
        body = {"committed": {str(s): r.sha256 for s, r in sorted(committed.items())}}
        body["execution"] = self.execution()
        _replace(done, json.dumps(body))
        return dict(committed)

    def _save(self, optimizer: torch.optim.Optimizer, step: int, committed: Mapping[int, ArtifactRef]) -> None:
        """Save the resume state after ``step`` updates and log the save (``resume_state``) with its seconds."""
        extra = self._extra(committed)
        start = time.perf_counter()
        save_resume_state(self.connector, optimizer.state_dict(), step, self.workspace, extra)
        self._log({"event": "resume_state", "step": step, "sec": time.perf_counter() - start})

    def _keep(self, optimizer: torch.optim.Optimizer, step: int, committed: Mapping[int, ArtifactRef]) -> None:
        """Save the final resume state, after the run's last update, as :meth:`_save` does but logging nothing."""
        save_resume_state(self.connector, optimizer.state_dict(), step, self.workspace, self._extra(committed))

    def _extra(self, committed: Mapping[int, ArtifactRef]) -> dict[str, Any]:
        """What a resume state holds beside the tensors: the committed checkpoints, the derived data, the execution."""
        return {
            "committed": {str(s): r.sha256 for s, r in committed.items()},
            "derived_sha256": canonical_sha256(self.derived),
            "execution": self.execution(),
        }

    def optimizer(self) -> torch.optim.AdamW:
        """Place the maps and their step-0 references (:meth:`_place`); return the run's AdamW, without state yet.

        The one construction of the run's optimizer: :meth:`fit` and the batch measurement (``training.measure``).
        """
        self._place()
        return torch.optim.AdamW(self._groups(), weight_decay=0.0)

    def apply(self, optimizer: torch.optim.Optimizer) -> tuple[float, float]:
        """Finish an update whose loss gradients are accumulated; the gradients are then cleared.

        The α budget penalty's gradient, the global gradient-norm clip and one ``optimizer`` step.

        Returns:
            The penalty and the clipped gradient norm.
        """
        penalty = self.connector.budget_penalty()
        if penalty.requires_grad:
            torch.autograd.backward(penalty)
        params = [p for p in self.connector.parameters() if p.grad is not None]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, self.config.optimizer.clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(penalty.detach()), grad_norm

    def _groups(self) -> list[dict[str, Any]]:
        """AdamW groups, one per tensor: ``lr_rel × reference`` (its map kind's, at step 0) or ``lr_alpha``.

        A parameter belongs to its outermost enclosing map kind, the one the connector holds (a kind may compose
        ``nn.Module`` parts, other kinds included), whose ``lr_reference`` and ``decayed`` get the parameter's name
        relative to it, as ``writes.conformance`` checks it. A decayed tensor's group decay is ``weight_decay / lr``:
        AdamW scales the tensor by ``1 − lr·factor·decay = 1 − factor·weight_decay``, the same for every tensor.
        """
        opt = self.config.optimizer
        kinds = {name: m for name, m in self.connector.named_modules() if isinstance(m, EntryMap)}
        groups = []
        for name, p in self.connector.named_parameters():
            decay = 0.0
            if name.startswith("theta."):
                lr = opt.lr_alpha
            else:
                owner = _owning_kind(name, kinds)
                local = name.removeprefix(owner + ".")
                lr = opt.lr_rel * kinds[owner].lr_reference(local, p.detach())
                if local in kinds[owner].decayed():
                    decay = opt.weight_decay / lr
            groups.append({"params": [p], "lr": lr, "weight_decay": decay, "name": name})
        return groups

    def _update(self, step: int, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """One optimizer update on step ``step``'s batch; return its log record."""
        start = time.perf_counter()
        examples = self.sampler.examples(step)
        prompts = [self.sampler.render(e) for e in examples]
        total = sum(sum(p.target.weights) for p in prompts if p.target is not None)
        if total <= 0:
            raise ValueError(f"step {step}: the batch's target weights sum to {total}; nothing to train on")
        source = {self.config.connector.name: LiveSource(self.connector, self.load)}
        size = self.config.micro_batch or len(prompts)
        weighted, weight_sum = 0.0, 0.0
        token_nll: list[Tensor] = []
        for lo in range(0, len(prompts), size):
            part = prompts[lo : lo + size]
            terms = self.backend.loss(part, resolve_sources(part, source))
            torch.autograd.backward(terms.weighted_sum / total)
            weighted += float(terms.weighted_sum.detach())
            weight_sum += float(terms.weight_sum)
            token_nll += [t.detach().float().cpu() for t in terms.token_nll]
        if abs(weight_sum - total) > _WEIGHT_TOLERANCE * total:
            raise RuntimeError(f"step {step}: the backend's Σw {weight_sum} differs from the targets' Σw {total}")
        penalty, grad_norm = self.apply(optimizer)
        return {
            "loss_w": weighted / total,
            "penalty": penalty,
            "grad_norm": grad_norm,
            "sec": time.perf_counter() - start,
            **_token_stats(prompts, token_nll, [e.stream for e in examples]),
            "scaffolded": sum(bool(e.prefix) for e in examples),
            "rel_drift": self._drift(),
            **self._alphas(),
        }

    def _place(self) -> None:
        """Put the maps on their receiver devices (``place_connector``) and their step-0 references with them."""
        place_connector(self.connector, self.backend.placement())
        params = dict(self.connector.named_parameters())
        self._initial = {n: p0.to(params[n].device) for n, p0 in self._initial.items()}

    def _drift(self) -> dict[str, float]:
        """Each map tensor's relative change from step 0, ``‖p − p₀‖ / ‖p₀‖`` (originally ``rel_drift``)."""
        params = dict(self.connector.named_parameters())
        with torch.no_grad():
            return {n: float((params[n] - p0).norm() / p0.norm()) for n, p0 in self._initial.items()}

    def _alphas(self) -> dict[str, Any]:
        with torch.no_grad():
            alphas = {e.key(): float(self.connector.alpha(e)) for e in self.connector.entries if e.layer > 0}
            learned = [float(torch.sigmoid(p)) for p in self.connector.theta.values()]
        return {"a": alphas, "sum_a": sum(learned)}

    def _commit(self, step: int, committed: dict[int, ArtifactRef], evaluate: bool = True) -> None:
        """Commit the step's checkpoint and, with ``evaluate``, run the in-job evaluation on it."""
        start = time.perf_counter()
        ref = commit_checkpoint(
            self.connector, step, self.store, encoder=self.config.encoder, append_encoder=self.append_encoder
        )
        committed[step] = ref
        self._log({"event": "checkpoint", "step": step, "checkpoint": ref.sha256, "sec": time.perf_counter() - start})
        if evaluate and self.evaluator is not None:
            with torch.no_grad():
                scores = self.evaluator(self.backend, self.connector, ref, step)
            self._log({"event": "eval", "step": step, "scores": scores.sha256})

    def execution(self) -> dict[str, str]:
        """The run's execution facts, saved in its resume state and done file and checked on resume.

        The backend's ``execution()`` as it is (its own options recorded by its own rule) and the trainer's own fact,
        ``input_cache``, its effective value.
        """
        return {**self.backend.execution(), "input_cache": self.input_cache}

    def _check_execution(self, saved: Mapping[str, str] | None) -> None:
        """Refuse to resume or return a run on another execution than the one it saved (its state or done file).

        :meth:`execution` (e.g. ``kimi-ep``'s checkpointing, the dtype, the attention implementation, the device count,
        the input cache) but :data:`RESUMABLE_ACROSS`, the weights directory and the accelerator model (module
        docstring): a run keeps the execution its record states.

        The comparison view leaves out :data:`RESUMABLE_ACROSS` and nothing else; a saved record is never rewritten. The
        input cache is checked first: a value other than ``"on"`` or ``"off"`` is refused, and so is a record without
        one that records any of the backend's options (written by a trainer that records both). A record with neither
        (no option keys and no input cache) resumes, by the exact comparison, only on the backend's reference options.
        The refusal names the ``--option`` values that resume the saved run for the backend's options (its
        ``reference_options``: option → the value a state that does not record the option ran).

        Raises:
            ValueError: if they differ, an input cache is invalid or missing, or ``saved`` is None.
        """
        now_facts = self.execution()
        reference = self.backend.reference_options
        for facts, whose in ((saved or {}, "its state"), (now_facts, "this job")):
            if "input_cache" in facts and facts["input_cache"] not in _INPUT_CACHES:
                cache = facts["input_cache"]
                raise ValueError(f"run {self.config.name!r}: {whose} records input_cache {cache!r}, not on or off")
        if saved and "input_cache" not in saved and any(k in saved for k in reference):
            raise ValueError(f"run {self.config.name!r}: its state records backend options without an input_cache")
        was, now = ({k: v for k, v in e.items() if k not in RESUMABLE_ACROSS} for e in (saved or {}, now_facts))
        if was != now:
            changed = sorted(k for k in {*now, *was} if was.get(k) != now.get(k))
            resume = " ".join(f"--option {k}={was.get(k, v)}" for k, v in reference.items() if k in changed)
            raise ValueError(
                f"run {self.config.name!r}: its state was saved under execution {saved}, this job runs {now} "
                f"(differing: {changed}); resume with the options the run started with"
                + (f": {resume}" if resume else "")
            )

    def _check_derived(self, state: ResumeState | None) -> None:
        """Refuse to continue a run whose streams now derive other data than it started with.

        The run directory's :data:`DERIVED_FILE` is what the run trained on (and what the train stage commits); the
        resume state carries its digest (None: a new or completed run, checked against the file alone). A rerun after
        the format rule's code changed derives something else.

        Raises:
            ValueError: if they differ, or a resume state or a started run lacks them.
        """
        path = self.workspace / DERIVED_FILE
        now = canonical_sha256(self.derived)
        if not path.exists():
            raise ValueError(f"run {self.config.name!r}: {path} is missing (written when the run starts)")
        started = canonical_sha256(json.loads(path.read_text()))
        saved = state.extra.get("derived_sha256") if state is not None else started
        if saved is None:
            raise ValueError(f"run {self.config.name!r}: its resume state predates the derived-data check")
        if not now == started == saved:
            raise ValueError(
                f"run {self.config.name!r}: the streams derive {now[:12]} from the training text, the run started "
                f"with {started[:12]} (resume state {saved[:12]}); a run must train on the data it started with"
            )

    def _log(self, line: Mapping[str, Any]) -> None:
        with (self.workspace / LOG_FILE).open("a") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")

    def _truncate_log(self, step: int) -> None:
        """Keep the log lines of steps up to ``step`` (a resumed run's lines past its saved step never happened).

        The dropped lines are first written to a new ``log.discarded.<UTC time>.<pid>.jsonl``
        (:data:`DISCARDED_PREFIX`; never overwritten), synced, and the log is then replaced atomically; a log with
        nothing to drop is left as it is.

        Raises:
            FileExistsError: if the discarded-lines file exists already.
        """
        path = self.workspace / LOG_FILE
        if not path.exists():
            return
        lines = [ln for ln in path.read_text().splitlines() if ln]
        kept = [ln for ln in lines if json.loads(ln)["step"] <= step]
        if len(kept) == len(lines):
            return
        dropped = [ln for ln in lines if json.loads(ln)["step"] > step]
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        with (self.workspace / f"{DISCARDED_PREFIX}{stamp}.{os.getpid()}.jsonl").open("x") as fh:
            fh.write("".join(f"{ln}\n" for ln in dropped))
            fh.flush()
            os.fsync(fh.fileno())
        _replace(path, "".join(f"{ln}\n" for ln in kept))

    @contextmanager
    def _signal(self) -> Iterator[None]:
        """Turn SIGUSR1 into a stop after the current step (``CheckpointSpec.on_signal``)."""
        if not self.config.checkpoints.on_signal:
            yield
            return

        def handler(_signum: int, _frame: object) -> None:
            self.stop.set()

        previous = signal.signal(signal.SIGUSR1, handler)
        try:
            yield
        finally:
            signal.signal(signal.SIGUSR1, previous)


def place_connector(connector: Connector, placement: Mapping[int, torch.device]) -> None:
    """Put each entry's map on the device of its layer's input."""
    for key, module in connector.maps.items():
        layer = int(key.rsplit("_L", 1)[1])
        module.to(placement[layer])


def _token_stats(prompts: Sequence[RenderedPrompt], nll: Sequence[Tensor], streams: Sequence[str]) -> dict[str, Any]:
    """The plain token-mean NLL, its position bins and per-stream means (``LossTerms.token_nll``, logging only)."""
    if len(nll) != len(prompts):
        raise RuntimeError(f"the backend returned token NLLs for {len(nll)} of {len(prompts)} prompts")
    every = torch.cat(list(nll))
    position = torch.cat([torch.arange(len(t)) for t in nll])
    bins = {}
    for name, (lo, hi) in POSITION_BINS.items():
        mask = position >= lo if hi is None else (position >= lo) & (position < hi)
        bins[name] = float(every[mask].mean()) if bool(mask.any()) else None
    per_stream: dict[str, list[Tensor]] = {}
    for s, t in zip(streams, nll, strict=True):
        per_stream.setdefault(s, []).append(t)
    return {
        "loss": float(every.mean()),
        "tokens": len(every),
        "loss_pos": bins,
        "stream_loss": {s: float(torch.cat(ts).mean()) for s, ts in sorted(per_stream.items())},
        "streams": {s: len(ts) for s, ts in sorted(per_stream.items())},
    }
