"""Entry point: run one receiver scan outside a study: an injection scan or a layer-set comparison."""

from __future__ import annotations

from collections.abc import Sequence

from latent_interconnect.evaluation.layer_selection import InjectionScanSpec, ScanRunConfig, injection_scan
from latent_interconnect.evaluation.layer_sets import compare_layer_sets
from latent_interconnect.provenance import loaded_packages
from latent_interconnect.run._main import (
    SCORING_PACKAGES,
    ExecutionTarget,
    Stage,
    StageResult,
    require_hardware,
    stage_main,
    visible_devices,
)
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(config: ScanRunConfig, ws: Workspace, **backend_options: str) -> StageResult:
    """Outputs ``numbers`` (the scan's summary with intervals) and ``scores`` (the per-object scores).

    The declared hardware is checked before anything loads. The inputs are ``Workspace.scan_inputs``; the
    record's purpose is ``result`` unless an input's lineage makes it stricter (``Workspace.purpose``).
    ``backend_options`` reach the scoring backend's constructor; the plan is the config's. The versions of the
    scoring libraries the process loaded are recorded after the scan (``SCORING_PACKAGES``).

    Raises:
        WrongHardwareError: if the visible devices are not the config's ``device_class``.
    """
    devices = visible_devices()
    require_hardware(devices, config.device_class)
    inputs = ws.scan_inputs(config)
    purpose = ws.purpose("result", inputs, {})
    ctx = ws.scoring_context(
        config.scan.receiver,
        "score",
        plan=config.plan,
        device_class=config.device_class,
        purpose=purpose,
        options=backend_options,
    )
    scan = config.scan
    out = injection_scan(scan, ctx) if isinstance(scan, InjectionScanSpec) else compare_layer_sets(scan, ctx)
    return StageResult(
        outputs={"numbers": out.numbers, "scores": out.scores},
        inputs=inputs,
        packages=loaded_packages(SCORING_PACKAGES),
        devices=devices,
        execution=ctx.backend.execution(),
        purpose=purpose,
    )


STAGE = Stage(
    "scan",
    ScanRunConfig,
    _run,
    lambda c: ExecutionTarget(c.device_class, c.scan.receiver, "score"),
    outputs="numbers|scores",
)
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.scan --config <yaml> [--pin NAME]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
