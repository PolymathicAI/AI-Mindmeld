"""Entry point: measure what one training ``loss`` call fits (``training.measure``).

``python -m latent_interconnect.run.measure --config <measure.yaml>`` commits the measurements as ``measurements``
(kind :data:`MEASUREMENT_KIND`: one JSON object per candidate and the recommendation) with the stage's record, so the
batch a person writes into a run config (``RunConfig.batch`` / ``micro_batch``) can cite the record it came from. The
declared hardware is the run's (``RunConfig.device_class``), checked first.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import torch

from latent_interconnect.canonical import canonical_sha256
from latent_interconnect.run._main import (
    ExecutionTarget,
    Stage,
    StageResult,
    require_hardware,
    stage_main,
    visible_devices,
)
from latent_interconnect.training.measure import MeasureSpec, measure_batches
from latent_interconnect.training.trainer import Trainer
from latent_interconnect.workspace import Workspace

__all__ = ["MEASUREMENT_KIND", "STAGE", "main"]

MEASUREMENT_KIND = "batch_measurement"
"""Artifact kind of a batch measurement (``measurements.json``)."""


def _run(config: MeasureSpec, ws: Workspace, **backend_options: str) -> StageResult:
    """Output ``measurements``; the training backend's ``execution()`` is recorded.

    The execution (without the weights directory) also keys and labels the artifact (meta ``execution``): the same
    config measured under other execution options is another artifact, cited by its own execution of the record.

    Raises:
        WrongHardwareError: if the visible devices are not the run's ``device_class``.
    """
    devices = visible_devices()
    require_hardware(devices, config.run.device_class)
    trainer = Trainer.from_config(
        config.run, ws.artifacts, ws.paths.runs / config.run.digest(), backend_options=backend_options
    )
    measured = measure_batches(trainer, config)
    best = max((m.size for m in measured if m.fits(config.memory_rule)), default=None)
    rows = [{**m.__dict__, "fits": m.fits(config.memory_rule)} for m in measured]
    execution = trainer.backend.execution()
    # memory and time depend on the execution (kimi-ep's checkpointing): it keys and labels the measurement, so a
    # measurement under other options is never handed the first one's artifact (the weights directory is node-local)
    measured_under = {k: v for k, v in execution.items() if k != "weights"}
    staged = ws.artifacts.staging(f"batch-measurement-{canonical_sha256([config.digest(), measured_under])[:16]}")
    (staged / "measurements.json").write_text(json.dumps({"candidates": rows, "recommended": best}, indent=1))
    meta = {"run_sha": config.run.digest(), "recommended": best, "execution": measured_under}
    out = ws.artifacts.commit(staged, MEASUREMENT_KIND, meta)
    return StageResult(
        outputs={"measurements": out}, packages={"torch": torch.__version__}, devices=devices, execution=execution
    )


STAGE = Stage(
    "measure",
    MeasureSpec,
    _run,
    lambda c: ExecutionTarget(c.run.device_class, c.run.receiver, "train"),
    outputs="measurements",
)
"""This stage's name, config class and function; it runs where the measured run trains."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.measure --config <yaml> [--pin NAME [--repin REASON]] [--option K=V]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
