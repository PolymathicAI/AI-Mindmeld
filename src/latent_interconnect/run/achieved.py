"""Entry point: publish a stopped run's achieved checkpoints without a fit, on CPU (``AchievedSpec``).

A long run whose jobs ended short of its steps (stopped at a chosen step, or for good) has committed checkpoints but
no record, so nothing pins them. This stage writes a record of its own for them: kind ``achieved``, config
``training.config.AchievedSpec`` (the run embedded), outputs ``step_<6 digits>`` for every checkpoint the run's resume
state holds as committed (``training.trainer.achieved_run``, which checks them against the store and the log) and
``log`` (the log up to the achieved step and ``derived.json``), inputs the run's (``run.train.run_inputs``), and
``resumed`` every segment of the run's ledger. A stopped extended run (``run.train``, ``extend_from``) keeps its
lineage, as its train record would: ``Provenance.extended_from`` from its ``extension.json``, and the source record
among the input records (``extended_from``, ``run.train.source_record``), so its inherited segments are told apart from
its new ones (``Provenance.inherited_segments``, ``Provenance.new_segments``) and its purpose is inherited; an origin
that does not load or a source record no longer in the store refuses the publication. ``execution`` says what it is:
``publication`` (:data:`PUBLICATION`), ``achieved_step`` (the resume state's step: what was saved), ``logged_step``
(the log's last step, never an achieved one: lines past the saved step are not published) and the run's training
execution under ``training.<key>`` (the training job's backend, not this job's; this job's devices are its own,
none). No ``done.json`` is written and the
run directory is not changed, apart from the lock file (``train.lock``) the publication creates there, so a later
continuation still resumes it.

**Its own identity.** The record is not the run's: its kind and config differ from a fit's, so
``--pin <the config's destination>`` (e.g. ``connectors/kimi-k2.6/esmc-6b-mlp4-staggered``) pins names a completion
never writes, a later completion of the run is the first execution of the run's computation and pins under the run's
destination, a ``--skip-pinned`` train job is not skipped and ``guarded_submit.sh`` does not hold it because of the
publication (neither its config digest nor its ``--pin`` is the run's), and a second publication of the run is refused,
so the published records and pins are never replaced.

**Only a stopped run.** The lock (``training.trainer.run_directory_lock``) is held from before the quiet check to
the end of the stage function, so a continuation from code that takes it is excluded for the whole snapshot (it exits
5 at its start and must be submitted again); a job running code older than the lock takes none. So the stage also
refuses a run whose files still move: it waits until none of ``log.jsonl``, ``segments.json``, ``resume.pt``,
``derived.json`` and ``done.json`` has changed for :data:`QUIET_S` (refusing if one changes while it waits), reads the
run, and refuses if any of them changed while it read. The record and pins are written by ``run._main.stage_main`` after
the lock is released, in the same process, so before the job ends; every output is committed (content-addressed) by
then. The quiet rule sees only a producer that writes: an older job still loading its model when the run is read is not
seen. The exclusion of those writers is
Slurm's, not this module's: the publication job carries its arm's training job name (``JobSpec.slurm_name``) with
``singleton`` and ``afterany`` on every train job of the arm (the runbook, "Publishing a stopped run"), so no job of
that name runs from its start to its exit; a writer under another name is excluded by no check.

Refusals exit :data:`PUBLISH_REFUSED` (``RUN_DIRECTORY_BUSY``, 5, for a held lock), writing nothing: every check,
the run's inputs (``run_inputs``) included, comes before the log artifact is committed, and a run file that vanishes
or cannot be read meanwhile (``OSError``) refuses like a failed check.
``--option dry_run=true`` prints what would be published and how long the files have been quiet, without the lock or
the wait.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

import numpy as np
import torch

from latent_interconnect.artifacts.store import IntegrityError
from latent_interconnect.canonical import canonical_sha256
from latent_interconnect.run._main import (
    ExecutionTarget,
    Stage,
    StageResult,
    require_hardware,
    stage_main,
    visible_devices,
)
from latent_interconnect.run.train import EXTENDED_FROM, RUN_DIRECTORY_BUSY, TRAIN_LOG_KIND, run_inputs, source_record
from latent_interconnect.training.config import AchievedSpec
from latent_interconnect.training.trainer import (
    DERIVED_FILE,
    DONE_FILE,
    LOG_FILE,
    SEGMENTS_FILE,
    AchievedRun,
    RunDirectoryBusyError,
    achieved_run,
    earlier_segments,
    read_extension,
    run_directory_lock,
)
from latent_interconnect.workspace import Workspace
from latent_interconnect.writes.checkpoint import RESUME_FILE

__all__ = ["POLL_S", "PUBLICATION", "PUBLISH_REFUSED", "QUIET_S", "STAGE", "WATCHED", "main"]

PUBLISH_REFUSED = 10
"""Exit code of a refused publication: no stopped run, files that moved, a state that fails the checks, or a run this
config already published; it wrote nothing."""
PUBLICATION = "stopped run: achieved checkpoints, no fit"
"""The ``execution["publication"]`` of the record, which says the run did not reach its steps."""
QUIET_S = 1800
"""Seconds every :data:`WATCHED` file must have been unchanged before the run is read (30 minutes). A live ESM-C 6B
multi-token job writes the log every step and the resume state every 200 steps. Measured on those runs (log.jsonl):
the slowest step 106.8 s (q6; mlp and linear 35 s), the slowest checkpoint commit 54 s and resume-state save 123 s
(q6), the in-job evaluation 115 s (q6 records; 88 s mlp, linear), so the longest silence of a running job is about
5 minutes at an evaluation boundary; the model load before a continuation's first write took about 14 minutes (q6:
from job start to derived.json). 30 minutes is about 6 times the first and 2 times the second."""
POLL_S = 30.0
"""Seconds between checks while the stage waits for :data:`QUIET_S`."""
WATCHED = (LOG_FILE, SEGMENTS_FILE, RESUME_FILE, DERIVED_FILE, DONE_FILE)
"""The run directory's files whose change means a producer is alive (``done.json``: its appearance)."""

Stamp = dict[str, tuple[int, int, int] | None]


def _refuse(code: int, message: str) -> NoReturn:
    sys.stderr.write(f"achieved: REFUSED (exit {code}): {message}\n")
    raise SystemExit(code)


def _stamp(workspace: Path) -> Stamp:
    """Each :data:`WATCHED` file's (modification time in ns, size, inode), or None if it does not exist."""
    out: Stamp = {}
    for name in WATCHED:
        try:
            st = (workspace / name).stat()
        except FileNotFoundError:
            out[name] = None
            continue
        out[name] = (st.st_mtime_ns, st.st_size, st.st_ino)
    return out


def _quiet_for(stamp: Stamp) -> float:
    """Seconds since the newest change among the stamped files.

    This node's clock is compared with mtimes from the writer node's clock: NTP skew is seconds against :data:`QUIET_S`.
    """
    return time.time() - max(s[0] for s in stamp.values() if s is not None) / 1e9


def _wait_quiet(workspace: Path) -> Stamp:
    """Return the files' stamp once none has changed for :data:`QUIET_S`; refuse if one changes meanwhile."""
    first = _stamp(workspace)
    while (quiet := _quiet_for(first)) < QUIET_S:
        sys.stderr.write(f"achieved: {workspace.name[:12]}: files quiet for {quiet:.0f} s, waiting for {QUIET_S} s\n")
        time.sleep(min(POLL_S, QUIET_S - quiet))
        if _stamp(workspace) != first:
            _refuse(PUBLISH_REFUSED, f"{workspace}: a run file changed while waiting: a producer is alive")
    return first


def _achieved(config: AchievedSpec, ws: Workspace, workspace: Path) -> AchievedRun:
    try:
        return achieved_run(config.run, ws.artifacts, workspace)
    except (ValueError, OSError) as err:  # OSError: a run file that vanished or cannot be read meanwhile
        _refuse(PUBLISH_REFUSED, f"{type(err).__name__}: {err}")


def _published(config: AchievedSpec, ws: Workspace, achieved: AchievedRun) -> list[str]:
    """An earlier publication of the run: this config's own records, or the record pinned at its destination.

    The second test (review N2) holds when a code change has altered the config's digest: the published
    ``<destination>@record`` pin still names the earlier record.
    """
    try:
        made = ws.records.produced_by(achieved.committed[0].sha256)
    except IntegrityError as err:
        _refuse(PUBLISH_REFUSED, f"the record store cannot be read: {err}")
    own = [r.computation_key() for r in made if r.kind == STAGE.name and r.config_sha == config.digest()]
    pinned = ws.pins().get(f"{config.destination}@record") if config.destination else None
    return own or ([pinned.sha256] if pinned is not None else [])


def _run(config: AchievedSpec, ws: Workspace, dry_run: bool = False) -> StageResult | None:
    """Outputs ``step_<6 digits>`` per achieved checkpoint and ``log``; the run's inputs (module docstring).

    ``dry_run`` reads and prints without the lock or the wait, returning None (no record).

    Raises:
        WrongHardwareError: if the job sees an accelerator (the config's ``device_class`` is ``cpu``).
        SystemExit: :data:`PUBLISH_REFUSED` for a refused publication, ``RUN_DIRECTORY_BUSY`` for a held lock.
    """
    devices = visible_devices()
    require_hardware(devices, config.device_class)
    workspace = ws.paths.runs / config.run.digest()
    missing = [n for n in (LOG_FILE, DERIVED_FILE, SEGMENTS_FILE, RESUME_FILE) if not (workspace / n).is_file()]
    if missing:  # checked before the lock, which would create the directory
        _refuse(PUBLISH_REFUSED, f"{workspace} lacks {missing}: no stopped run to publish")
    if dry_run:
        achieved = _achieved(config, ws, workspace)
        quiet = _quiet_for(_stamp(workspace))
        out = sys.stdout
        out.write(f"achieved: would publish {workspace} at step {achieved.step} (log to {achieved.logged_step})\n")
        for step, ref in sorted(achieved.committed.items()):
            out.write(f"  step_{step:06d}: {ref.sha256}\n")
        out.write(f"  files quiet for {quiet:.0f} s (a publication waits until {QUIET_S} s)\n")
        made = _published(config, ws, achieved)
        out.write(f"  already published by record(s) {made}: refused\n" if made else "")
        return None
    try:
        with run_directory_lock(workspace):
            return _publish(config, ws, workspace, devices)
    except RunDirectoryBusyError as err:
        _refuse(RUN_DIRECTORY_BUSY, str(err))


def _publish(config: AchievedSpec, ws: Workspace, workspace: Path, devices: tuple[str, ...]) -> StageResult:
    """:func:`_run` under the run directory's lock: wait for quiet, read, check nothing moved, commit the log."""
    before = _wait_quiet(workspace)
    achieved = _achieved(config, ws, workspace)
    try:
        segments = earlier_segments(workspace, ws.provenance({}, devices).finish().segment())
    except ValueError as err:
        _refuse(PUBLISH_REFUSED, str(err))
    if _stamp(workspace) != before:
        _refuse(PUBLISH_REFUSED, f"{workspace}: a run file changed while it was read: a producer is alive")
    if not segments:
        _refuse(PUBLISH_REFUSED, f"{workspace / SEGMENTS_FILE} lists no job: no stopped run to publish")
    if made := _published(config, ws, achieved):
        _refuse(PUBLISH_REFUSED, f"this run is already published (record {made[0][:12]}): published pins never move")
    run = config.run
    try:  # before the log is committed, so a refusal leaves no artifact behind (review S2)
        inputs = run_inputs(run, ws)
    except (IntegrityError, OSError) as err:
        _refuse(PUBLISH_REFUSED, f"an input of the run cannot be read: {type(err).__name__}: {err}")
    try:  # an extended run's lineage, as the train stage records it (run.train): read before the log is committed
        origin = read_extension(workspace)
        source = None if origin is None else source_record(ws, origin)
    except (ValueError, IntegrityError, OSError) as err:  # ExtensionError is a ValueError
        _refuse(PUBLISH_REFUSED, f"the extended run's source cannot be resolved: {type(err).__name__}: {err}")
    key = f"achieved-log-{run.digest()[:16]}-{hashlib.sha256(achieved.log).hexdigest()[:16]}"
    log = ws.artifacts.committed(key)
    if log is None:
        staged = ws.artifacts.staging(key)
        (staged / LOG_FILE).write_bytes(achieved.log)
        (staged / DERIVED_FILE).write_bytes(achieved.derived)
        meta = {
            "run": run.name,
            "run_sha": run.digest(),
            "derived_sha256": canonical_sha256(json.loads(achieved.derived)),
            "achieved_step": achieved.step,
            "logged_step": achieved.logged_step,
        }
        log = ws.artifacts.commit(staged, TRAIN_LOG_KIND, meta)
    execution = {
        "publication": PUBLICATION,
        "achieved_step": str(achieved.step),
        "logged_step": str(achieved.logged_step),
        "quiet_s": str(QUIET_S),
        **{f"training.{k}": v for k, v in sorted(achieved.execution.items())},
    }
    return StageResult(
        outputs={f"step_{s:06d}": ref for s, ref in sorted(achieved.committed.items())} | {"log": log},
        inputs=inputs,
        packages={"torch": torch.__version__, "numpy": np.__version__},
        devices=devices,
        execution=execution,
        resumed=segments,
        records={} if source is None else {EXTENDED_FROM: source},
        extended_from=origin,
    )


STAGE = Stage(
    "achieved",
    AchievedSpec,
    _run,
    lambda c: ExecutionTarget(c.device_class),
    outputs=r"step_\d{6}|log",
)
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.achieved --config <yaml> --pin <destination> [--option dry_run=true]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
