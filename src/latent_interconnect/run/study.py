"""Entry point: run a layer-selection study (``evaluation.layer_selection.run_study``)."""

from __future__ import annotations

from collections.abc import Sequence

from latent_interconnect.evaluation.layer_selection import LayerSelectionStudy, check_study, run_study
from latent_interconnect.run._main import (
    ExecutionTarget,
    Stage,
    StageResult,
    require_hardware,
    stage_main,
    visible_devices,
)
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(config: LayerSelectionStudy, ws: Workspace, **backend_options: str) -> StageResult:
    """Outputs ``run_study``'s (``injection/numbers`` and ``injection/scores``); inputs ``study_inputs``.

    The declared hardware and the train-only rule are checked before anything loads. The record's purpose is
    ``result`` unless an input makes it stricter (``Workspace.purpose``). ``backend_options`` reach the scoring
    backend's constructor; the plan is the config's.

    Raises:
        WrongHardwareError: if the visible devices are not the config's ``device_class``.
        ValueError: if the scan's item set holds an object outside ``train``.
    """
    devices = visible_devices()
    require_hardware(devices, config.device_class)
    check_study(config, ws.item_set, ws.domain)  # the train-only rule, before the receiver loads
    inputs = ws.study_inputs(config)
    purpose = ws.purpose("result", inputs, {})
    ctx = ws.scoring_context(
        config.injection.receiver,
        "score",
        plan=config.plan,
        device_class=config.device_class,
        purpose=purpose,
        options=backend_options,
    )
    outputs = run_study(config, ctx)
    return StageResult(
        outputs=outputs, inputs=inputs, devices=devices, execution=ctx.backend.execution(), purpose=purpose
    )


STAGE = Stage(
    "study",
    LayerSelectionStudy,
    _run,
    lambda c: ExecutionTarget(c.device_class, c.injection.receiver, "score"),
    outputs="injection/numbers|injection/scores",
)
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.study --config <yaml> [--pin NAME [--repin REASON]] [--option K=V]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
