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
from pathlib import Path
from typing import NoReturn

import numpy as np
import torch

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.canonical import canonical_sha256
from latent_interconnect.data.training_text import TRAINING_SET_KIND
from latent_interconnect.domains.base import DOMAIN_DATA_KIND
from latent_interconnect.evaluation.suites import SUITE_KINDS, SuiteContext, SuiteRunConfig
from latent_interconnect.provenance import code_state
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
    InJobEvaluator,
    InputCache,
    Interrupted,
    ResumeRequiredError,
    RunDirectoryBusyError,
    Trainer,
    earlier_segments,
    input_loader,
    run_directory_lock,
)
from latent_interconnect.workspace import Workspace
from latent_interconnect.writes.checkpoint import RESUME_FILE
from latent_interconnect.writes.connector import Connector
from latent_interconnect.writes.sources import LiveSource

__all__ = ["RESUME_REFUSED", "RUN_DIRECTORY_BUSY", "STAGE", "TRAIN_LOG_KIND", "in_job_evaluator", "main", "run_inputs"]

TRAIN_LOG_KIND = "train_log"
"""Artifact kind of a run's training log (``training.trainer.LOG_FILE``, one JSON line per update)."""
RESUME_REFUSED = 4
"""Exit code of a continuation job (``resume_required``) refused for a missing, unreadable or incompatible resume state;
it wrote nothing to the run directory."""
RUN_DIRECTORY_BUSY = 5
"""Exit code of a job refused because another process holds the run directory's lock; it wrote nothing else there."""


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


def _run(
    config: RunConfig,
    ws: Workspace,
    resume_required: bool = False,
    input_cache: InputCache = DEFAULT_INPUT_CACHE,
    **backend_options: str,
) -> StageResult | None:
    """Outputs ``step_<6 digits>`` per committed checkpoint and ``log``; inputs as the module docstring lists.

    ``backend_options`` (execution options as text) reach the training backend's constructor. ``resume_required``
    makes the job a continuation (module docstring); it returns None, writing no record, for a completed run.
    ``input_cache`` is the trainer's (module docstring).

    Raises:
        WrongHardwareError: if the visible devices are not the config's ``device_class`` (checked before anything
            loads).
        SystemExit: with status 1 after SIGUSR1, once the resume state is saved; :data:`RUN_DIRECTORY_BUSY` if another
            job holds the run directory; :data:`RESUME_REFUSED` for a continuation without a state to continue from.
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
            )
    except RunDirectoryBusyError as err:
        _refuse(RUN_DIRECTORY_BUSY, str(err))
    except ResumeRequiredError as err:
        _refuse(RESUME_REFUSED, str(err))


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
) -> StageResult | None:
    """:func:`_run` under the run directory's lock."""
    if resume_required and (workspace / DONE_FILE).exists():
        sys.stderr.write(f"train: {workspace / DONE_FILE} exists: the run is complete, nothing to continue\n")
        return None
    segment = ws.provenance({}, devices).finish().segment()
    earlier_segments(workspace, segment)  # the ledger is read (and a broken one refused) before the model loads
    evaluator = in_job_evaluator(config, ws) if config.eval_suite is not None else None
    trainer = Trainer.from_config(
        config,
        ws.artifacts,
        workspace,
        evaluator=evaluator,
        backend_options=backend_options,
        stop=stop,
        input_cache=input_cache,
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
