"""Entry point: build one Rhea cross-domain design over its familiar/novel 2x2 cells (``build.cross_domain_items``)."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import pandas as pd

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.build.bindingdb_items import BindingDBItemSpec
from latent_interconnect.build.cross_domain_items import CrossDomainItemSpec, build_cross_domain_items
from latent_interconnect.build.xdom_screen import MOLECULE, PROTEIN
from latent_interconnect.data.items import ITEM_SET_KIND, ItemSet
from latent_interconnect.evaluation.cross_domain import pair_labels, swap_quads
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "cell_inputs", "main"]


def cell_inputs(config: CrossDomainItemSpec | BindingDBItemSpec, ws: Workspace) -> dict[str, ArtifactRef]:
    """Return the inputs of a cross-domain 2x2 build (``xdom_items``, ``xdom_bdb_items``).

    The two domains, the two training sets, the novel-protein selection's domain, exclusions and homolog searches, and
    ``exclude_molecules/<i>``.
    """
    sel = config.novel_proteins
    inputs = {
        "protein": ws.artifacts.ref(config.protein.data),
        "molecule": ws.artifacts.ref(config.molecule.data),
        "familiar_proteins": ws.artifacts.ref(config.familiar_proteins),
        "familiar_molecules": ws.artifacts.ref(config.familiar_molecules),
        "novel_proteins/domain": ws.artifacts.ref(sel.domain.data),
    }
    inputs |= {f"novel_proteins/exclude/{i}": ArtifactRef(d, ITEM_SET_KIND) for i, d in enumerate(sel.exclude)}
    for h in sel.homologs:
        inputs[f"novel_proteins/homologs/{h.reason}/search"] = ws.artifacts.ref(h.search)
        inputs |= {f"novel_proteins/homologs/{h.reason}/{i}": ArtifactRef(d, ITEM_SET_KIND) for i, d in enumerate(h.of)}
    inputs |= {f"exclude_molecules/{i}": ArtifactRef(d, ITEM_SET_KIND) for i, d in enumerate(config.exclude_molecules)}
    return inputs


def _run(config: CrossDomainItemSpec, ws: Workspace) -> StageResult:
    """Build the design's cells (``build.cross_domain_items``).

    Outputs ``items-<cell>`` per cell with an item, ``proteins``, ``molecules``, ``molecule_pool``; inputs
    :func:`cell_inputs`' and (ranking) ``fingerprints``. A retained subset (``CrossDomainItemSpec.retained``) has no
    ``molecule_pool``, also reads ``retained/within/<cell>`` and ``retained/parents/<cell>``, and is held to the
    suite kind's structure check of its design before it is committed.
    """
    inputs = cell_inputs(config, ws)
    if config.fingerprints is not None:
        inputs["fingerprints"] = ws.artifacts.ref(config.fingerprints)
    if config.retained is not None:
        inputs |= {f"retained/within/{c}": ArtifactRef(d, ITEM_SET_KIND) for c, d in config.retained.within.items()}
        inputs |= {f"retained/parents/{c}": ArtifactRef(d, ITEM_SET_KIND) for c, d in config.retained.parents.items()}

    def check(items: ItemSet) -> object:
        return swap_quads(items, (PROTEIN, MOLECULE)) if config.design == "swap_quads" else pair_labels(items)

    outputs = build_cross_domain_items(config, ws.artifacts, ws.paths.data, check=check)
    return StageResult(outputs=outputs, inputs=inputs, packages={"pandas": pd.__version__, "networkx": nx.__version__})


STAGE = Stage("xdom_items", CrossDomainItemSpec, _run, outputs="items-[^/]+|proteins|molecules|molecule_pool")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.xdom_items --config <yaml> [--pin NAME [--repin REASON]] [--option K=V]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
