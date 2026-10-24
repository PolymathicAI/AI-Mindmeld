"""Entry point: build the zero-shot-screened training set, or an all-tasks one, which screens nothing."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.build.zero_shot import TrainingSetSpec, build_training_set
from latent_interconnect.domains.base import DOMAIN_DATA_KIND
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(config: TrainingSetSpec, ws: Workspace) -> StageResult:
    """Output: ``training_set`` (training_set). Inputs: ``domain``, ``train_ids``."""
    ref = build_training_set(config, ws.artifacts)
    inputs = {
        "domain": ArtifactRef(config.domain.data, DOMAIN_DATA_KIND),
        "train_ids": ws.artifacts.ref(config.train_ids),  # an item set, or a training set whose objects are used
    }
    return StageResult(outputs={"training_set": ref}, inputs=inputs, packages={"pandas": pd.__version__})


STAGE = Stage("training_set", TrainingSetSpec, _run, outputs="training_set")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.training_set --config <yaml> [--pin NAME [--repin REASON]]``.

    Like every stage, it also takes ``[--option K=V]`` (``run._main.stage_options``).
    """
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
