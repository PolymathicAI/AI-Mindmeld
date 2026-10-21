"""Entry point: select training objects from the train split (``build.train_selection``)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.build.train_selection import (
    HomologyExclusion,
    TrainSelectionSpec,
    commit_selection,
    select_train_objects,
)
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(config: TrainSelectionSpec, ws: Workspace) -> StageResult:
    """Outputs ``train_ids``, ``added`` and ``parents`` (item_set).

    Inputs ``domain``, ``search/<reason>`` per homology exclusion and ``base`` when set (the earlier selection's
    ``train_ids`` list).

    The pool and id files are cited by sha256 in the config itself.
    """
    selection = select_train_objects(config, ws.artifacts, ws.paths.data)
    train_ids, added, parents = commit_selection(config, selection, ws.artifacts)
    inputs: dict[str, ArtifactRef] = {"domain": ws.artifacts.ref(config.domain.data)}
    if config.base is not None:
        inputs["base"] = ws.artifacts.ref(config.base)
    for ex in config.exclusions:
        if isinstance(ex, HomologyExclusion):
            inputs[f"search/{ex.reason}"] = ws.artifacts.ref(ex.search)
    return StageResult(
        outputs={"train_ids": train_ids, "added": added, "parents": parents},
        inputs=inputs,
        packages={"pandas": pd.__version__, "numpy": np.__version__},
    )


STAGE = Stage("train_selection", TrainSelectionSpec, _run, outputs="train_ids|added|parents")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.train_selection --config <yaml> [--pin NAME]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
