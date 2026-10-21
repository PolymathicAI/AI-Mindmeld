"""Entry point: fit the ridge initialization and commit it."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.build.ridge import RidgeSpec, capture_receiver, fit_ridge
from latent_interconnect.data.items import ITEM_SET_KIND
from latent_interconnect.domains.base import DOMAIN_DATA_KIND
from latent_interconnect.encoders.store import EMBEDDINGS_KIND
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(config: RidgeSpec, ws: Workspace) -> StageResult:
    """Output: ``ridge`` (ridge_init). Inputs: ``embeddings_<i>``, ``targets``, ``fit_set``, ``domain``, ``exclude``.

    ``exclude`` is recorded only for a held-out scan ridge, so a production ridge's record is unchanged.
    """
    inputs = {f"embeddings_{i}": ArtifactRef(d, EMBEDDINGS_KIND) for i, d in enumerate(config.embeddings)}
    inputs |= {
        "targets": ArtifactRef(config.targets, EMBEDDINGS_KIND),
        "fit_set": ArtifactRef(config.fit_set, ITEM_SET_KIND),
        "domain": ArtifactRef(config.domain.data, DOMAIN_DATA_KIND),
    }
    if config.exclude is not None:
        inputs["exclude"] = ArtifactRef(config.exclude, ITEM_SET_KIND)
    ref = fit_ridge(config, ws.artifacts, capture_receiver(ws.records, config.targets))
    return StageResult(outputs={"ridge": ref}, inputs=inputs, packages={"numpy": np.__version__})


STAGE = Stage("ridge", RidgeSpec, _run, outputs="ridge")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.ridge --config <yaml> [--pin NAME [--repin REASON]] [--option K=V]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
