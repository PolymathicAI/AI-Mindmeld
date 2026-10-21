"""Entry point: select one item set from a domain as whole clusters (``build.item_selection``)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.build.item_selection import ItemSelectionSpec, select_items
from latent_interconnect.data.items import ITEM_SET_KIND
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(config: ItemSelectionSpec, ws: Workspace) -> StageResult:
    """Output ``item_set``.

    Inputs: ``domain``, ``exclude/<i>`` per excluded item set, ``within`` when set (an item set or embeddings),
    ``training_text`` when set (the training set whose kept text bounds the candidates),
    ``within_any/<i>`` and ``without/<i>`` per bound and per dropped set (item sets or embeddings), and per homolog
    screen ``homologs/<reason>/search`` and ``homologs/<reason>/<i>`` per item set it names (a ``pool_clean`` screen:
    ``pool_clean/<reason>/...``).
    """
    inputs = {"domain": ws.artifacts.ref(config.domain.data)}
    inputs |= {f"exclude/{i}": ArtifactRef(d, ITEM_SET_KIND) for i, d in enumerate(config.exclude)}
    for kind, screens in (("homologs", config.homologs), ("pool_clean", config.pool_clean)):
        for h in screens:
            inputs[f"{kind}/{h.reason}/search"] = ws.artifacts.ref(h.search)
            inputs |= {f"{kind}/{h.reason}/{i}": ArtifactRef(d, ITEM_SET_KIND) for i, d in enumerate(h.of)}
    if config.within is not None:
        inputs["within"] = ws.artifacts.ref(config.within)  # an item set or an embeddings artifact
    if config.training_text is not None:
        inputs["training_text"] = ws.artifacts.ref(config.training_text.training_set)
    for field in ("within_any", "without"):
        inputs |= {f"{field}/{i}": ws.artifacts.ref(d) for i, d in enumerate(getattr(config, field))}
    return StageResult(
        outputs={"item_set": select_items(config, ws.artifacts, ws.paths.data)},
        inputs=inputs,
        packages={"pandas": pd.__version__},
    )


STAGE = Stage("items", ItemSelectionSpec, _run, outputs="item_set")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.items --config <yaml> [--pin NAME [--repin REASON]] [--option K=V]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
