"""Entry point: train a connector (``python -m latent_interconnect.run.train --config <run.yaml>``).

The stage checks the declared hardware first (``RunConfig.device_class``), then builds the trainer
(``training.Trainer.from_config``) on the run directory ``Paths.runs / <run digest>``, where its resume state and log
live, so a restarted job resumes the same run. Outputs: ``step_<6 digits>`` per committed checkpoint and ``log`` (the
training log, kind :data:`TRAIN_LOG_KIND`, with ``derived.json``: what the streams derived from the training text,
``Sampler.derived``, the enabled grammar formats, inside the content-addressed output). Inputs: the domain
data, the screened training set, the embeddings and the ridge fits. The committed ``derived.json`` is the file the
trainer wrote into the run directory when the run started and checked on every ``fit``, not a recomputation. SIGUSR1
saves the resume state and exits with status 1, writing no record. The record's provenance lists the earlier jobs of
a resumed run (``Provenance.resumed``, from the trainer's segment ledger), each with the devices it ran on. A run may
resume on other hardware and a config may be executed again on other hardware; the declared class's kind and
topology are still checked, and the trainer still refuses a resume under another execution (precision, attention,
device count, backend options: ``Trainer._check_execution``).

**One producer, and continuations.** The stage holds the run directory's lock
(``training.trainer.run_directory_lock``) from before the ledger is read until the training log is committed (the end
of the stage function); it covers training, not the record or the pins, which ``run._main.stage_main`` writes after the
lock is released. A second job on the same run directory exits :data:`RUN_DIRECTORY_BUSY` at once. A run that finished
training (``done.json``) but whose job died before its record was written stays unrecorded until a relaunch without
the option records it; a continuation cannot duplicate a record, since it writes none
for a completed run. ``--option resume_required=true`` makes the job a continuation:
it exits :data:`RESUME_REFUSED` if the run directory holds neither a resume state nor ``done.json`` (before it creates
anything, the lock file included) or if the resume state does not load or fails the resume checks (before the trainer
writes anything); a run whose ``done.json`` exists is complete, and the job exits 0 without training, writing no record
(the job that completed the run wrote it; a relaunch without the option writes one that is missing). The option is an
execution option, recorded in the provenance (``option.resume_required``), never in an identity.
``--option input_cache=on|off`` (``training.trainer.InputCache``; on by default) keeps the embedding readers' memory
maps for the job or maps the part files on every batch: the same bytes either way. The trainer records the effective
value in its execution facts (the record, the resume state, ``done.json``), and a resume or relaunch may change it.
The in-job evaluator's loader and the ``measure`` stage take the default (they read the same bytes); ``measure`` keys
its measurement by the backend's execution, since input loading is outside the loss calls it measures.

A stopped run's achieved checkpoints are published by the ``achieved`` stage (``run.achieved``), a computation of its
own; this stage's inputs are :func:`run_inputs`.

**The kept final state.** A completed run keeps its final resume state in its run directory (``training.trainer``):
about three times the connector's weights on disk (the weights and the optimizer's two moments; 22–25 GB for a large
multi-token MLP connector, far less for a small one). The save writes no log line, so the records and outputs of a run
are as they were before the state was kept. Deleting ``resume.pt`` of a completed run frees the space; the run then
cannot be extended.

**Extending a finished run.** ``--option extend_from=S`` makes this config, of ``steps`` T, continue the finished
S-step run of the config that equals it but for ``steps = S`` (``1 <= S < T``), instead of training from step 0. The
option is the stage's own, never handed to the backend, and recorded like every option (``option.extend_from``); a
config with more steps and no option trains fresh. The source is authenticated before anything is written: its
``train`` record (the one canonical execution of its computation; the inputs are this config's) and its stored
config's digest; its run directory ``Paths.runs / <source digest>`` holding the completed run, with its kept final state
at S, the record's checkpoints and log, and its derived data (``training.trainer.completed_run``). Then, once the
backend has loaded, ``training.trainer.Trainer.extend`` checks that the streams still derive that data and that this
job's execution may resume the source's (the differences a resume permits, ``RESUMABLE_ACROSS``), and copies the
source's state, log, derived data and segment ledger into this run's new directory, which appears whole with its
``extension.json`` (``EXTENSION_FILE``) or not at all; the source is left as it was. Training then continues to T as a
resume does. A job killed during the copy leaves no run directory, only its staging directory beside it
(``.extend.<run digest>.<source digest>.<hex>``, ``EXTENSION_STAGING``); a relaunch with the same option, source and
target removes it and starts the copy again (another relaunch does not look for it). A job killed after the copy leaves
an extended run that every later job resumes, with or without the option, keeping its lineage from ``extension.json``,
which each job reads again under the run directory's lock before it trains: a job that found no run directory before
another producer published an extension there resumes that extension and records its source. A source run whose lock
another job holds exits :data:`RUN_DIRECTORY_BUSY`, as for any busy run. The job exits
:data:`EXTEND_REFUSED`, writing nothing to either run directory, when: S is out of range; the schedule reads the total
steps (``OptimizerSpec.depends_on_total``, e.g. ``cosine``); there is no such record (another optimizer setting, weight
decay included, or any other change but ``steps`` makes another source config, of which no record exists), or several;
the source is not finished at S, has no kept final state, or disagrees with its record (its log artifact missing or
unreadable included); this run's directory exists and is not an extension of that source; or the streams derive other
data or the execution differs from the source's beyond what a resume permits. A refusal found under the source's lock
(the source changed after it was checked, or this run's directory appeared meanwhile) rewrites the source's
``train.lock``, which names its last holder as for any job on the run; nothing else is written. An origin that
changed, vanished or names another source when read again under this run's lock is refused the same way, rewriting
only this run's ``train.lock``. A continuation (``resume_required``) never starts an extension: without this run's
directory it is refused as ever.

The extended run's record is another execution of this config's computation. Its provenance names the source
(``Provenance.extended_from``: the source's config digest, computation key, record id, final step and number of
segments), its ``resumed`` segments begin with the source's jobs (``Provenance.inherited_segments``, apart from the
cost of its own, ``Provenance.new_segments``), and the source record is among its input records (``extended_from``), so
its purpose is inherited. Its outputs are a straight T-step run's checkpoints, plus the source's final checkpoint at S
when S is off the checkpoint cadence; its log holds the source's lines through S, then its own.

**In-job evaluation.** The run's ``eval_suite`` is scored on the training backend itself at step 0, every
``eval_every`` steps and the end, by the suite kind's registered runner, through the connector being trained. Each
evaluation is its own ``suite`` record with purpose ``monitor`` (or ``anchor``, if its inputs are anchor-made), so it
is never eligible as a result; its config is the ``SuiteRunConfig`` it is equivalent to (the run's receiver, hardware
and batch plan), with the checkpoint the trainer committed at that step.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, get_args

import numpy as np
import torch

from latent_interconnect.artifacts.records import Record, StoredPurpose, computation_key_of
from latent_interconnect.artifacts.store import ArtifactRef, IntegrityError
from latent_interconnect.canonical import canonical_sha256
from latent_interconnect.data.training_text import TRAINING_SET_KIND
from latent_interconnect.domains.base import DOMAIN_DATA_KIND
from latent_interconnect.evaluation.suites import SUITE_KINDS, SuiteContext, SuiteRunConfig
from latent_interconnect.provenance import Extension, code_state
from latent_interconnect.receivers import open_renderer
from latent_interconnect.receivers.base import ScoringBackend
from latent_interconnect.run._main import (
    ExecutionTarget,
    Stage,
    StageResult,
    require_hardware,
    stage_main,
    visible_devices,
)
from latent_interconnect.training.config import RunConfig
from latent_interconnect.training.trainer import (
    DEFAULT_INPUT_CACHE,
    DERIVED_FILE,
    DONE_FILE,
    LOG_FILE,
    ExtensionError,
    InJobEvaluator,
    InputCache,
    Interrupted,
    ResumeRequiredError,
    RunDirectoryBusyError,
    Trainer,
    completed_run,
    earlier_segments,
    input_loader,
    read_extension,
    run_directory_lock,
)
from latent_interconnect.workspace import Workspace
from latent_interconnect.writes.checkpoint import RESUME_FILE
from latent_interconnect.writes.connector import Connector
from latent_interconnect.writes.sources import LiveSource

__all__ = [
    "EXTENDED_FROM",
    "EXTEND_REFUSED",
    "RESUME_REFUSED",
    "RUN_DIRECTORY_BUSY",
    "STAGE",
    "TRAIN_LOG_KIND",
    "in_job_evaluator",
    "main",
    "run_inputs",
    "source_record",
]

TRAIN_LOG_KIND = "train_log"
"""Artifact kind of a run's training log (``training.trainer.LOG_FILE``, one JSON line per update)."""
RESUME_REFUSED = 4
"""Exit code of a continuation job (``resume_required``) refused for a missing, unreadable or incompatible resume state;
it wrote nothing to the run directory."""
RUN_DIRECTORY_BUSY = 5
"""Exit code of a job refused because another process holds the run directory's lock; it wrote nothing else there."""
EXTEND_REFUSED = 6
"""Exit code of a job whose ``extend_from`` is refused (module docstring); it wrote nothing to either run directory but,
for a refusal under the source's lock, the source's ``train.lock``. A busy source exits :data:`RUN_DIRECTORY_BUSY`."""
EXTENDED_FROM = "extended_from"
"""The input-record role of an extended run's source (``StageResult.records``)."""


def in_job_evaluator(config: RunConfig, ws: Workspace) -> InJobEvaluator:
    """Return the run's in-job evaluator: its ``eval_suite`` on the training backend, one ``suite`` record each.

    Raises:
        ValueError: if the run has no ``eval_suite``.
    """
    suite = config.eval_suite
    if suite is None:
        raise ValueError(f"run {config.name!r} has no eval_suite")
    arguments = sorted({slot.connector for slot in suite.slots})

    def evaluate(backend: ScoringBackend, connector: Connector, checkpoint: ArtifactRef, step: int) -> ArtifactRef:
        run = SuiteRunConfig(
            suite=suite,
            receiver=config.receiver,
            checkpoints=dict.fromkeys(arguments, checkpoint.sha256),
            inputs=dict.fromkeys(arguments, config.embeddings),
            step=step,
            device_class=config.device_class,
            plan=config.plan,
        )
        inputs = ws.suite_inputs(run)
        # the checkpoint was made by this run, whose record comes at its end: lineage is judged on the other inputs
        upstream = {k: v for k, v in inputs.items() if not k.startswith("checkpoint/")}
        context = SuiteContext(
            artifacts=ws.artifacts,
            records=ws.records,
            receiver=config.receiver,
            backend_name=config.receiver.backends["train"],
            backend=backend,
            renderer=open_renderer(config.receiver),
            domain=ws.domain,
            device_class=config.device_class,
            purpose=ws.purpose("monitor", upstream, {}),
            items=ws.item_set,
            plan=config.plan,
            code=code_state(ws.repo),
        )
        loader = input_loader(config, ws.artifacts, connector)
        sources = dict.fromkeys(arguments, LiveSource(connector, loader.batch))
        started = ws.provenance({"torch": torch.__version__}, visible_devices())
        scores = SUITE_KINDS.get(suite.kind)(suite).run(context, sources, step)
        provenance = started.model_copy(update={"execution": backend.execution()})
        ws.record("suite", run, inputs, {"scores": scores}, provenance, purpose=context.purpose)
        return scores

    return evaluate


def _refuse(code: int, message: str) -> NoReturn:
    sys.stderr.write(f"train: REFUSED (exit {code}): {message}\n")
    raise SystemExit(code)


@dataclass(frozen=True)
class _Source:
    """The authenticated source of a new extension (:func:`_authenticate`).

    Attributes:
        config: the source's config (this config with ``steps = S``).
        directory: its run directory.
        record: its ``train`` record, the execution read (listed in ``StageResult.records``).
    """

    config: RunConfig
    directory: Path
    record: Record


def _run(
    config: RunConfig,
    ws: Workspace,
    resume_required: bool = False,
    input_cache: InputCache = DEFAULT_INPUT_CACHE,
    extend_from: int | None = None,
    **backend_options: str,
) -> StageResult | None:
    """Outputs ``step_<6 digits>`` per committed checkpoint and ``log``; inputs as the module docstring lists.

    ``backend_options`` (execution options as text) reach the training backend's constructor. ``resume_required``
    makes the job a continuation (module docstring); it returns None, writing no record, for a completed run.
    ``input_cache`` is the trainer's (module docstring). ``extend_from`` extends the finished run of this config with
    that many ``steps`` (module docstring); a run directory that is already an extension keeps its source either way.

    Raises:
        WrongHardwareError: if the visible devices are not the config's ``device_class`` (checked before anything
            loads).
        SystemExit: with status 1 after SIGUSR1, once the resume state is saved; :data:`RUN_DIRECTORY_BUSY` if another
            job holds the run directory (or, while an extension is initialized, its source's);
            :data:`RESUME_REFUSED` for a continuation without a state to continue from; :data:`EXTEND_REFUSED` for an
            extension refused.
    """
    stop = threading.Event()
    if config.checkpoints.on_signal:  # from the job's start, so a SIGUSR1 during loading is kept (the trainer saves)
        signal.signal(signal.SIGUSR1, lambda _signum, _frame: stop.set())
    devices = visible_devices()
    require_hardware(devices, config.device_class)
    workspace = ws.paths.runs / config.digest()
    if resume_required and not any((workspace / name).exists() for name in (RESUME_FILE, DONE_FILE)):
        _refuse(RESUME_REFUSED, f"{workspace} holds no {RESUME_FILE} and no {DONE_FILE}: nothing to continue")
    try:
        origin, pending = _extension(config, ws, workspace, extend_from)
        source = None if origin is None else source_record(ws, origin)
        trainer = None
        if pending is not None:  # loaded before its run directory exists, which the lock would make, then initialized
            trainer = _trainer(
                config, ws, workspace=workspace, stop=stop, backend_options=backend_options, input_cache=input_cache
            )
            origin = trainer.extend(
                pending.config,
                pending.directory,
                computation_key=pending.record.computation_key(),
                record_id=pending.record.record_id(),
            )
            source = pending.record
        with run_directory_lock(workspace):
            return _produce(
                config,
                ws,
                workspace=workspace,
                devices=devices,
                stop=stop,
                resume_required=resume_required,
                backend_options=backend_options,
                input_cache=input_cache,
                trainer=trainer,
                lineage=_locked_lineage(
                    config,
                    ws,
                    workspace,
                    extend_from,
                    None if origin is None or source is None else (origin, source),
                ),
            )
    except RunDirectoryBusyError as err:
        _refuse(RUN_DIRECTORY_BUSY, str(err))
    except ResumeRequiredError as err:
        _refuse(RESUME_REFUSED, str(err))
    except ExtensionError as err:
        _refuse(EXTEND_REFUSED, str(err))


def _extension(
    config: RunConfig, ws: Workspace, workspace: Path, extend_from: int | None
) -> tuple[Extension | None, _Source | None]:
    """Return the origin of the run directory if it is an extension, and the source of a new one to initialize.

    Read only. With ``extend_from``: a run directory that is an extension of the same source is resumed as it is (its
    source is not read again), any other existing one is refused, and without one the source is authenticated
    (:func:`_authenticate`).

    Raises:
        ExtensionError: as the module docstring lists, or for an unreadable ``extension.json``.
    """
    try:
        origin = read_extension(workspace)
    except ValueError as err:
        raise ExtensionError(f"run {config.name!r}: {workspace} holds an unreadable extension origin: {err}") from err
    if extend_from is None:
        return origin, None
    if not 1 <= extend_from < config.steps:
        raise ExtensionError(f"run {config.name!r}: extend_from={extend_from} is not in 1..{config.steps - 1}")
    if config.optimizer.depends_on_total():
        raise ExtensionError(
            f"run {config.name!r}: its {config.optimizer.schedule} schedule reads the total steps, so a shorter run's "
            "updates are not this run's; it cannot be extended"
        )
    source = RunConfig.model_validate(config.model_dump() | {"steps": extend_from})
    if origin is not None:
        if origin.config_sha != source.digest():
            raise ExtensionError(
                f"run {config.name!r}: {workspace} extends the {origin.step}-step run {origin.config_sha[:12]}, "
                f"not the {extend_from}-step run {source.digest()[:12]}"
            )
        return origin, None
    if workspace.exists():
        raise ExtensionError(
            f"run {config.name!r}: {workspace} exists and is not an extension; extend_from starts a new run directory"
        )
    return None, _authenticate(config, source, ws)


def _locked_lineage(
    config: RunConfig,
    ws: Workspace,
    workspace: Path,
    extend_from: int | None,
    read: tuple[Extension, Record] | None,
) -> tuple[Extension, Record] | None:
    """Return the origin and source record of the run directory, read again under its lock, for the record.

    ``read`` is the lineage found before the lock (:func:`_extension`, or the extension this job initialized). Another
    producer may publish an initialized extension between that read and the lock; the run is then resumed as that
    extension and its record names that source, with or without ``extend_from``.

    Raises:
        ExtensionError: if ``extension.json`` is unreadable, is gone or names another source than ``read``, names
            another source than ``extend_from``'s, or its source record is no longer in the store.
    """
    try:
        origin = read_extension(workspace)
    except ValueError as err:
        raise ExtensionError(f"run {config.name!r}: {workspace} holds an unreadable extension origin: {err}") from err
    if origin is None:
        if read is not None:
            raise ExtensionError(f"run {config.name!r}: {workspace} lost the extension origin it was read with")
        return None
    if read is not None and origin != read[0]:
        raise ExtensionError(
            f"run {config.name!r}: the extension origin in {workspace} changed after it was read (source record "
            f"{read[0].record_id[:12]}, now {origin.record_id[:12]})"
        )
    if extend_from is not None:
        expected = RunConfig.model_validate(config.model_dump() | {"steps": extend_from}).digest()
        if origin.config_sha != expected:
            raise ExtensionError(
                f"run {config.name!r}: {workspace} extends the {origin.step}-step run {origin.config_sha[:12]}, "
                f"not the {extend_from}-step run {expected[:12]}"
            )
    source = source_record(ws, origin)
    if read is not None and source.record_id() != read[1].record_id():
        raise ExtensionError(f"run {config.name!r}: the source record of {workspace} changed after it was read")
    return origin, source


def _authenticate(config: RunConfig, source: RunConfig, ws: Workspace) -> _Source:
    """Return the finished source run of an extension, checked against its record (module docstring); read only.

    Raises:
        ExtensionError: if there is no such record or several, or the record, the source's run directory
            (``training.trainer.completed_run``) and its committed log disagree.
    """
    name, digest = config.name, source.digest()
    inputs = {k: ref.sha256 for k, ref in run_inputs(config, ws).items()}
    keys = [
        key
        for purpose in get_args(StoredPurpose)
        if ws.records.exists(key := computation_key_of(STAGE.name, purpose, digest, inputs))
    ]
    if len(keys) != 1:
        raise ExtensionError(
            f"run {name!r}: {len(keys)} train records of the {source.steps}-step run {digest[:12]} (this config with "
            f"steps={source.steps} and nothing else changed); an extension continues exactly one"
        )
    try:
        record = ws.records.canonical(keys[0])
    except LookupError as err:
        raise ExtensionError(f"run {name!r}: the source's record is not resolved: {err}") from err
    try:
        stored = RunConfig.model_validate(record.config).digest()
    except ValueError as err:
        raise ExtensionError(f"run {name!r}: the source record's config does not load: {err}") from err
    if not stored == record.config_sha == digest:
        raise ExtensionError(f"run {name!r}: the source record's stored config is not {digest[:12]}")
    directory = ws.paths.runs / digest
    found = completed_run(source, directory)
    outputs = {int(k.removeprefix("step_")): o.sha256 for k, o in record.outputs.items() if k != "log"}
    if "log" not in record.outputs or outputs != found.committed:
        raise ExtensionError(f"run {name!r}: the source record's checkpoints are not those of its run in {directory}")
    log = ArtifactRef(record.outputs["log"].sha256, TRAIN_LOG_KIND)
    committed = ws.artifacts.path(log)
    try:
        meta = ws.artifacts.manifest(log).meta
        logged, derived = (committed / LOG_FILE).read_bytes(), (committed / DERIVED_FILE).read_bytes()
    except (OSError, ValueError, IntegrityError) as err:
        raise ExtensionError(
            f"run {name!r}: the source record's log artifact {log.short} cannot be read: {type(err).__name__}: {err}"
        ) from err
    if (
        meta.get("run_sha") != digest
        or meta.get("derived_sha256") != canonical_sha256(json.loads(found.derived))
        or logged != found.log
        or derived != found.derived
    ):
        raise ExtensionError(f"run {name!r}: the source record's log is not the log and derived data in {directory}")
    recorded = record.provenance.execution
    if {k: recorded.get(k) for k in found.execution} != dict(found.execution):
        raise ExtensionError(f"run {name!r}: the source record's execution is not its run's in {directory}")
    return _Source(source, directory, record)


def source_record(ws: Workspace, origin: Extension) -> Record:
    """Return the exact source execution an extended run names (``extension.json``); the ``achieved`` stage's too.

    Raises:
        ExtensionError: if the record store no longer holds it.
    """
    found = [r for r in ws.records.executions(origin.computation_key) if r.record_id() == origin.record_id]
    if not found:
        raise ExtensionError(f"the source record {origin.record_id[:12]} of this extended run is not in the store")
    return found[0]


def _trainer(
    config: RunConfig,
    ws: Workspace,
    *,
    workspace: Path,
    stop: threading.Event,
    backend_options: dict[str, str],
    input_cache: InputCache,
) -> Trainer:
    """Build the run's trainer (``Trainer.from_config``) with its in-job evaluator, if any."""
    evaluator = in_job_evaluator(config, ws) if config.eval_suite is not None else None
    return Trainer.from_config(
        config,
        ws.artifacts,
        workspace,
        evaluator=evaluator,
        backend_options=backend_options,
        stop=stop,
        input_cache=input_cache,
    )


def _produce(
    config: RunConfig,
    ws: Workspace,
    *,
    workspace: Path,
    devices: tuple[str, ...],
    stop: threading.Event,
    resume_required: bool,
    backend_options: dict[str, str],
    input_cache: InputCache,
    trainer: Trainer | None,
    lineage: tuple[Extension, Record] | None,
) -> StageResult | None:
    """:func:`_run` under the run directory's lock.

    ``trainer`` is the one a new extension was initialized with (None: built here), ``lineage`` the origin and source
    record of an extended run (None for another run).
    """
    if resume_required and (workspace / DONE_FILE).exists():
        sys.stderr.write(f"train: {workspace / DONE_FILE} exists: the run is complete, nothing to continue\n")
        return None
    segment = ws.provenance({}, devices).finish().segment()
    earlier_segments(workspace, segment)  # the ledger is read (and a broken one refused) before the model loads
    if trainer is None:
        trainer = _trainer(
            config, ws, workspace=workspace, stop=stop, backend_options=backend_options, input_cache=input_cache
        )
    try:
        committed = trainer.fit(segment, resume_required=resume_required)
    except Interrupted as stopped:
        raise SystemExit(1) from stopped
    staged = ws.artifacts.staging(f"train-log-{config.digest()[:16]}")
    (staged / LOG_FILE).write_bytes((workspace / LOG_FILE).read_bytes())
    derived = (workspace / DERIVED_FILE).read_bytes()  # what the run trained on (checked by fit)
    (staged / DERIVED_FILE).write_bytes(derived)
    meta = {"run": config.name, "run_sha": config.digest(), "derived_sha256": canonical_sha256(json.loads(derived))}
    log = ws.artifacts.commit(staged, TRAIN_LOG_KIND, meta)
    outputs = {f"step_{s:06d}": ref for s, ref in sorted(committed.items())} | {"log": log}
    return StageResult(
        outputs=outputs,
        inputs=run_inputs(config, ws),
        packages={"torch": torch.__version__, "numpy": np.__version__},
        devices=devices,
        execution=trainer.execution(),
        resumed=trainer.resumed,
        records={} if lineage is None else {EXTENDED_FROM: lineage[1]},
        extended_from=None if lineage is None else lineage[0],
    )


def run_inputs(config: RunConfig, ws: Workspace) -> dict[str, ArtifactRef]:
    """Return a run's inputs (module docstring): the domain, the training set, the embeddings, the ridge fits.

    The ``achieved`` stage's inputs are the same (``run.achieved``).
    """
    inputs = {
        "domain": ArtifactRef(config.domain.data, DOMAIN_DATA_KIND),
        "training_set": ArtifactRef(config.training_set, TRAINING_SET_KIND),
    }
    inputs |= {f"embeddings/{i}": ws.artifacts.ref(d) for i, d in enumerate(config.embeddings)}
    inputs |= {f"ridge/{t}": ArtifactRef(d, "ridge_init") for t, d in sorted(config.ridge.items())}
    return inputs


STAGE = Stage(
    "train",
    RunConfig,
    _run,
    lambda c: ExecutionTarget(c.device_class, c.receiver, "train"),
    outputs=r"step_\d{6}|log",
)
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.train --config <yaml> [--pin NAME [--repin REASON]] [--option K=V]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
