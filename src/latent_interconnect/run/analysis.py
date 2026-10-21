"""Entry point: run an analysis."""

from __future__ import annotations

from collections.abc import Sequence

from latent_interconnect.evaluation.analysis import AnalysisSpec, analysis_kind, run_analysis
from latent_interconnect.provenance import loaded_packages
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["ANALYSIS_PACKAGES", "STAGE", "main"]

ANALYSIS_PACKAGES = ("numpy", "scipy", "pandas", "pyarrow")
"""The libraries an analysis computes and reads its tables with, by module name: the stage records the versions of
those its process loaded (``provenance.loaded_packages``), never a scoring stack it does not use. A kind that computes
with further libraries names them itself (its ``packages``; ``sae_exploration``: torch and the dictionary's file
format), and they are recorded for that kind's runs only."""


def _run(config: AnalysisSpec, ws: Workspace) -> StageResult:
    """Output ``result``; record inputs: the executions of the spec's ``inputs`` it read (the purpose follows them).

    The versions of the analysis libraries the process loaded are recorded (``ANALYSIS_PACKAGES`` and the kind's own
    ``packages``, if it declares any).
    """
    own: tuple[str, ...] = getattr(analysis_kind(config.kind), "packages", ())
    result = run_analysis(config, ws.artifacts, ws.records, ws.item_set)  # ws.records resolves once (stage_main)
    return StageResult(
        outputs={"result": result},
        records={r: ws.records.canonical(k) for r, k in config.inputs.items()},
        packages=loaded_packages((*ANALYSIS_PACKAGES, *own)),
    )


STAGE = Stage("analysis", AnalysisSpec, _run, outputs="result")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.analysis --config <yaml> [--pin NAME [--repin REASON]] [--option K=V]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
