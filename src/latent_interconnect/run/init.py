"""Entry point: build a connector's initial state from its ridge solutions and commit it as step 0."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.build.init_state import InitSpec, commit_initial_state
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(config: InitSpec, ws: Workspace) -> StageResult:
    """Output ``step_000000`` (checkpoint); inputs ``ridge/<target>``, one per ridge the config cites."""
    inputs = {f"ridge/{t}": ArtifactRef(d, "ridge_init") for t, d in sorted(config.ridge.items())}
    checkpoint = commit_initial_state(config, ws.artifacts)
    return StageResult(
        outputs={"step_000000": checkpoint},
        inputs=inputs,
        packages={"torch": torch.__version__, "numpy": np.__version__},
    )


STAGE = Stage("init", InitSpec, _run, outputs="step_000000")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.init --config <yaml> [--pin NAME [--repin REASON]] [--option K=V]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
