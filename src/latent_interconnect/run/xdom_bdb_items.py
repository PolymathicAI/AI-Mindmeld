"""Entry point: build the BindingDB cross-domain 2x2 (``build.bindingdb_items``)."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import pandas as pd

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.build.bindingdb_items import BindingDBItemSpec, build_bindingdb_items
from latent_interconnect.build.xdom_screen import MOLECULE, PROTEIN
from latent_interconnect.data.items import ITEM_SET_KIND
from latent_interconnect.evaluation.cross_domain import swap_quads
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.run.xdom_items import cell_inputs
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(config: BindingDBItemSpec, ws: Workspace) -> StageResult:
    """Build the cells (``build.bindingdb_items``).

    Outputs ``molecule_domain``, ``items-<cell>`` per cell with a quad, ``proteins``, ``molecules``; inputs
    ``run.xdom_items.cell_inputs``' and, for a retained subset, ``retained/within/<cell>`` and
    ``retained/molecule_domain``. A retained subset is held to the suite kind's structure check before it is committed.
    """
    inputs = cell_inputs(config, ws)
    if config.retained is not None:
        inputs |= {f"retained/within/{c}": ArtifactRef(d, ITEM_SET_KIND) for c, d in config.retained.within.items()}
        inputs["retained/molecule_domain"] = ws.artifacts.ref(config.retained.molecule_domain)
    outputs = build_bindingdb_items(
        config, ws.artifacts, ws.paths.data, check=lambda items: swap_quads(items, (PROTEIN, MOLECULE))
    )
    return StageResult(outputs=outputs, inputs=inputs, packages={"pandas": pd.__version__, "networkx": nx.__version__})


STAGE = Stage("xdom_bdb_items", BindingDBItemSpec, _run, outputs="molecule_domain|items-[^/]+|proteins|molecules")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.xdom_bdb_items --config <yaml> [--pin NAME [--repin REASON]]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
