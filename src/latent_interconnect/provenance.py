"""Execution provenance: which code, where, on what hardware, for how long. Recorded by every computation.

A result from a dirty tree is not a result: every entry point that writes a record calls
:meth:`Provenance.require_clean`. An installed package without a git checkout records its version and
``commit = "no-vcs"``. Start and end times with the device list make compute cost an analysis over records.

Stages that run on accelerators declare a :class:`DeviceClass` in their config and check it with
:meth:`DeviceClass.require`. The check is about kind and topology, not the
accelerator model: a GPU class takes any whole accelerators (no MIG slice) in the declared count, so the declared value
names the reference hardware the config was written for and stays part of its identity, while the devices actually
used are recorded (:attr:`Provenance.devices`, :attr:`Segment.devices`).

**Package versions.** :attr:`Provenance.packages` holds what the stage reports (``run._main.StageResult.packages``):
the libraries its numerics depend on, read in the executing process. The ``scan``, ``suite`` and ``analysis`` stages
report the versions of the modules their process loaded (:func:`loaded_packages`); the field is empty for a stage that
reports none and in those three stages' records written before the field was added, whose versions were not recorded.
It is provenance, outside the computation key (``artifacts.records``).

**Segments.** An execution may run in several jobs: a training run stopped by SIGUSR1 or killed resumes in the
next job. Each job is a :class:`Segment` (start, end, host, job, commit, devices); the record's provenance is the last
one, and :attr:`Provenance.resumed` lists the earlier ones, so cost is the sum over :meth:`Provenance.segments`, never
the last job alone.

**Hardware is recorded, not enforced.** An execution's hardware is
:func:`hardware_of` its devices: the distinct accelerator models (``()`` on CPU). A computation may execute again on
other hardware (another execution of the same computation: ``artifacts.records`` keeps the first one canonical
unless a re-pin declares another), and a resumed run may continue on other hardware: every execution and every segment
keeps the devices it actually ran on, so a result's hardware is always its provenance's, never assumed.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field

from latent_interconnect import __version__
from latent_interconnect.config import Config, added_field
from latent_interconnect.paths import NO_VCS, code_state

__all__ = [
    "NO_VCS",
    "DeviceClass",
    "DirtyTreeError",
    "Provenance",
    "Segment",
    "WrongHardwareError",
    "code_state",
    "hardware_of",
    "loaded_packages",
    "visible_devices",
]

_GPU_NODE = 8
"""The device count of an ``8xH200`` stage: one node of eight accelerators (topology, still checked)."""


class DirtyTreeError(RuntimeError):
    """The code tree has uncommitted changes, so a result would not be reproducible."""


class WrongHardwareError(RuntimeError):
    """A stage runs on hardware other than the class its config declares."""


def hardware_of(devices: Sequence[str]) -> tuple[str, ...]:
    """Return an execution's hardware as provenance reports it: the distinct device models, sorted.

    ``()`` is CPU. The device count is not part of it: a stage's declared :class:`DeviceClass` fixes the count where
    it matters (an eight-GPU node).
    """
    return tuple(sorted(set(devices)))


def loaded_packages(names: Sequence[str]) -> dict[str, str]:
    """Return each named top-level module this process has imported → its ``__version__``.

    What a stage records as :attr:`Provenance.packages` when the libraries it uses depend on its backend: the version
    the process actually loaded (``sys.modules``), read after the work, not the installed distribution's metadata.
    Nothing is imported: a module the process did not load, or one without a text ``__version__``, is left out, so a
    stage never needs a library it does not use.

    Args:
        names: top-level module names (e.g. ``"torch"``, ``"compressed_tensors"``).
    """
    found = {name: getattr(sys.modules.get(name), "__version__", None) for name in names}
    return {name: version for name, version in found.items() if isinstance(version, str)}


class DeviceClass(StrEnum):
    """The declared hardware of a stage, checked as kind and topology.

    The value names the reference hardware a config was written for and is part of its identity (configs keep it);
    :meth:`matches` no longer requires that accelerator model (hardware flexibility):

    - ``A100`` and ``gpu``: at least one accelerator, every one whole (no MIG slice), of any model;
    - ``8xH200``: one node of eight whole accelerators, of any model (the count is topology: tensor-parallel
      backends span all eight);
    - ``cpu``: no accelerator (CPU stages and the tests).

    Memory is not a class property: the site preflight checks each device's memory against the receiver's need
    (outside this package), and a backend that does not fit fails as it loads.
    """

    A100 = "A100"
    H200X8 = "8xH200"
    CPU = "cpu"
    GPU = "gpu"

    def matches(self, devices: Sequence[str]) -> bool:
        """Return whether the accelerator names (one per visible device, as provenance records them) are this class."""
        if self is DeviceClass.CPU:
            return not devices
        whole = bool(devices) and all("MIG" not in d for d in devices)
        if self is DeviceClass.H200X8:
            return whole and len(devices) == _GPU_NODE
        return whole

    def require(self, devices: Sequence[str]) -> None:
        """Raise unless ``devices`` are this class.

        Raises:
            WrongHardwareError: naming the declared class and the devices found.
        """
        if not self.matches(devices):
            raise WrongHardwareError(f"declared {self.value}, running on {list(devices) or 'CPU'}")


def visible_devices() -> tuple[str, ...]:
    """Return the names of the visible accelerators (empty on CPU), as provenance records and ``DeviceClass`` checks."""
    import torch  # noqa: PLC0415  (only callers that check hardware pay for it)

    if not torch.cuda.is_available():
        return ()
    return tuple(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))


class Segment(Config):
    """One job's share of an execution: a resumed training run executes in several jobs, on any hardware.

    Attributes:
        started: UTC start time of the job's share (ISO 8601).
        ended: UTC end time; for a job that was killed without notice, the last time it recorded progress.
        host: the machine name.
        job_id: the batch-system job id if any.
        commit: the code commit the job ran.
        devices: accelerator names, one per visible device.
    """

    started: str
    ended: str
    host: str
    job_id: str | None
    commit: str
    devices: tuple[str, ...]

    def hardware(self) -> tuple[str, ...]:
        """Return this segment's hardware (:func:`hardware_of`)."""
        return hardware_of(self.devices)

    def touch(self) -> Segment:
        """Return a copy ended now (the segment's progress mark)."""
        return self.model_copy(update={"ended": _now()})


class Provenance(Config):
    """Code state and execution environment of one run.

    Attributes:
        version: the package version.
        commit: ``git rev-parse HEAD`` of the code tree, or ``"no-vcs"`` for an installed package.
        dirty: whether the tree had uncommitted or untracked changes (False without a checkout).
        argv: the command line.
        host: the machine name.
        job_id: the batch-system job id if any (``$SLURM_JOB_ID`` or ``$LIP_JOB_ID``).
        python: interpreter version.
        packages: versions of the packages that affect numerics (torch, transformers, vllm, numpy, ...), as the
            stage reports them (``run._main.StageResult.packages``): empty for a stage that reports none, and in
            ``scan``, standalone ``suite`` and ``analysis`` records written before they reported theirs
            (:func:`loaded_packages`).
        devices: accelerator names, one per visible device.
        started: UTC start time (ISO 8601).
        ended: UTC end time, set by :meth:`finish` when the record is written.
        execution: how a scoring or training pass executed (backend, precision, batch size, device type), so drift
            across hardware, precision and batching is documented.
        pins: the pin names the run's config cited, with the digests they resolved to (``{$pin: name}``; the
            identity embeds the digests, this keeps the names). An added field (``config.added_field``): empty, it is
            not part of a record's id, so records written before it existed keep theirs.
        resumed: the earlier segments of a resumed execution, oldest first (:class:`Segment`); this provenance's own
            fields are its last segment. An added field: empty for an execution that ran in one job, so records
            written before it existed (all single-job) load and keep their ids.
        input_executions: input role → ``record_id`` of the exact execution the run read for it
            (``run._main.StageResult.records``, resolved once), so the execution an artifact was drawn from stays
            known after a canonical re-point; artifact metas cite computation keys and output digests instead, so
            identical results give identical artifacts. An added field: empty for a run that read no record, and
            records written before it existed keep their ids.
    """

    version: str
    commit: str
    dirty: bool
    argv: tuple[str, ...]
    host: str
    job_id: str | None
    python: str
    packages: dict[str, str]
    devices: tuple[str, ...]
    started: str
    ended: str | None = None
    execution: dict[str, str] = Field(default_factory=dict)
    pins: dict[str, str] = added_field(default_factory=dict)
    resumed: tuple[Segment, ...] = added_field(default=())
    input_executions: dict[str, str] = added_field(default_factory=dict)

    @classmethod
    def capture(cls, repo: Path | None, packages: dict[str, str], devices: tuple[str, ...]) -> Provenance:
        """Capture the provenance of the current process.

        Args:
            repo: the code tree whose commit is recorded, or None for an installed package.
            packages: package versions to record (callers pass what affects their numerics).
            devices: accelerator names (empty on CPU).
        """
        commit, dirty = code_state(repo)
        return cls(
            version=__version__,
            commit=commit,
            dirty=dirty,
            argv=tuple(sys.argv),
            host=platform.node(),
            job_id=os.environ.get("SLURM_JOB_ID") or os.environ.get("LIP_JOB_ID"),
            python=platform.python_version(),
            packages=packages,
            devices=devices,
            started=_now(),
        )

    def finish(self) -> Provenance:
        """Return a copy with ``ended`` set to now."""
        return self.model_copy(update={"ended": _now()})

    def hardware(self) -> tuple[str, ...]:
        """Return this execution's hardware (:func:`hardware_of` its devices)."""
        return hardware_of(self.devices)

    def segment(self) -> Segment:
        """Return this provenance's own segment (its last job).

        Raises:
            ValueError: if it has no end time (:meth:`finish`).
        """
        if self.ended is None:
            raise ValueError("a provenance without an end time is not a segment: call Provenance.finish()")
        return Segment(
            started=self.started,
            ended=self.ended,
            host=self.host,
            job_id=self.job_id,
            commit=self.commit,
            devices=self.devices,
        )

    def segments(self) -> tuple[Segment, ...]:
        """Return every segment of the execution, oldest first: :attr:`resumed`, then its own (the cost is their sum).

        Raises:
            ValueError: if it has no end time.
        """
        return (*self.resumed, self.segment())

    def require_clean(self) -> None:
        """Raise unless the code tree was clean.

        Raises:
            DirtyTreeError: if ``dirty`` is set.
        """
        if self.dirty:
            raise DirtyTreeError(f"uncommitted changes at {self.commit[:12]}: commit before producing a result")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
