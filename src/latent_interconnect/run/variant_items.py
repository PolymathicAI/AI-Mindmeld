"""Entry point: build the protein-mutation evaluation cohorts and their object list (``build.variant_items``; E11)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.build.variant_items import VariantItemSpec, build_variant_items
from latent_interconnect.data.items import ITEM_SET_KIND
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(config: VariantItemSpec, ws: Workspace) -> StageResult:
    """Output ``items-<cohort>`` and ``objects``; inputs ``domain``, ``training_set``, ``relational_parents``."""
    inputs = {
        "domain": ws.artifacts.ref(config.domain.data),
        "training_set": ArtifactRef(config.training_set, ITEM_SET_KIND),
        "relational_parents": ArtifactRef(config.relational_parents, ITEM_SET_KIND),
    }
    return StageResult(
        outputs=build_variant_items(config, ws.artifacts, ws.paths.data),
        inputs=inputs,
        packages={"pandas": pd.__version__},
    )


STAGE = Stage("variant_items", VariantItemSpec, _run, outputs="items-[^/]+|objects")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.variant_items --config <yaml> [--pin NAME [--repin REASON]]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
