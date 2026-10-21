"""Entry point: run an encoder read-layer scan (``evaluation.read_scan.read_layer_scan``; CPU).

A long scan can be split statically over workers: ``--option shards=N --option shard=i`` commits the parts of every
N-th read (``ReadLayerScanSpec.reads``: the layers, then the combinations) from the i-th and exits without a record;
a run without the options then reuses every committed part, computes any missing one and records the scan
(``evaluation.read_scan``: parts are reused under the same clean code).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np

from latent_interconnect.evaluation.read_scan import ReadLayerScanSpec, read_label, read_layer_scan, scan_layers
from latent_interconnect.provenance import code_state
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(config: ReadLayerScanSpec, ws: Workspace, shards: int = 1, shard: int = 0) -> StageResult | None:
    """Outputs ``numbers`` and ``table`` (the per-object read_scan table); inputs ``Workspace.read_scan_inputs``.

    The scan refuses objects outside ``train`` itself. With ``shards`` > 1 the run commits the parts of
    ``config.reads[shard::shards]`` and returns None: no record (status 0).

    Raises:
        ValueError: if ``shard`` is not in ``0 .. shards - 1``.
    """
    if not 0 <= shard < shards:
        raise ValueError(f"read scan: shard {shard} of {shards} (0 <= shard < shards)")
    inputs = ws.read_scan_inputs(config)
    if shards > 1:
        reads = config.reads[shard::shards]
        scan_layers(config, ws.artifacts, code_state(ws.repo), reads)
        sys.stdout.write(
            f"read_scan: shard {shard}/{shards} committed reads {list(map(read_label, reads))}; no record\n"
        )
        return None
    out = read_layer_scan(config, ws.artifacts, code_state(ws.repo))
    return StageResult(
        outputs={"numbers": out.numbers, "table": out.table},
        inputs=inputs,
        packages={"numpy": np.__version__},
    )


STAGE = Stage("read_scan", ReadLayerScanSpec, _run, outputs="numbers|table")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.read_scan --config <yaml> [--pin NAME]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
