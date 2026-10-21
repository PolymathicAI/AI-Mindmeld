"""Entry point: an encoder read's supervised logistic baseline on one suite (``evaluation.logistic_baseline``; CPU)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.data.items import ITEM_SET_KIND
from latent_interconnect.data.training_text import TRAINING_SET_KIND
from latent_interconnect.domains.base import DOMAIN_DATA_KIND
from latent_interconnect.encoders.store import EMBEDDINGS_KIND
from latent_interconnect.evaluation.logistic_baseline import LogisticBaselineSpec, logistic_baseline
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(config: LogisticBaselineSpec, ws: Workspace) -> StageResult:
    """Outputs ``scores`` (scores) and ``numbers`` (numbers).

    Inputs: ``domain/<name>`` (each of the suite's domains), ``items``, ``training_set`` (or ``train_objects`` and
    ``train_domain`` in its place), ``train_embeddings/<i>``, ``eval_embeddings/<i>`` and ``more_items/<i>`` (each
    of ``more_suites``' item sets).
    """
    inputs = {f"domain/{d.name}": ArtifactRef(d.data, DOMAIN_DATA_KIND) for d in config.suite.domains}
    inputs["items"] = ArtifactRef(config.suite.item_set, ITEM_SET_KIND)
    if config.training_set is not None:
        inputs["training_set"] = ArtifactRef(config.training_set, TRAINING_SET_KIND)
    else:
        assert config.train_objects is not None and config.train_domain is not None  # the spec's check
        inputs["train_objects"] = ArtifactRef(config.train_objects, ITEM_SET_KIND)
        inputs["train_domain"] = ArtifactRef(config.train_domain.data, DOMAIN_DATA_KIND)
    inputs |= {f"more_items/{i}": ArtifactRef(s.item_set, ITEM_SET_KIND) for i, s in enumerate(config.more_suites)}
    inputs |= {f"train_embeddings/{i}": ArtifactRef(d, EMBEDDINGS_KIND) for i, d in enumerate(config.train_embeddings)}
    inputs |= {f"eval_embeddings/{i}": ArtifactRef(d, EMBEDDINGS_KIND) for i, d in enumerate(config.eval_embeddings)}
    out = logistic_baseline(config, ws.artifacts)
    return StageResult(
        outputs={"scores": out.scores, "numbers": out.numbers},
        inputs=inputs,
        packages={"numpy": np.__version__, "scipy": scipy.__version__},
    )


STAGE = Stage("logistic_baseline", LogisticBaselineSpec, _run, outputs="scores|numbers")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.logistic_baseline --config <yaml> [--pin NAME [--repin REASON]]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
