"""Entry point: run a declared MMseqs2 homology search and commit its hits (``build.homology``)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.build.homology import DomainSequences, HomologySearchSpec, search_homologs
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(
    config: HomologySearchSpec,
    ws: Workspace,
    mmseqs: Path = Path("mmseqs"),
    threads: int | None = None,
    workdir: Path | None = None,
) -> StageResult:
    """Output ``hits`` (homology); inputs ``queries/domain`` and ``targets/domain`` when a side is a domain's.

    Options (execution, not identity): ``mmseqs``, the binary (default: ``mmseqs`` on the path; its version must be
    the config's); ``threads`` (default: MMseqs2's, every core); ``workdir``, where the scratch directory is made
    (default: ``Paths.scratch``; local disk is much faster). Table files are cited by sha256 in the config itself.
    """
    ref = search_homologs(
        config, ws.artifacts, ws.paths.data, mmseqs=mmseqs, threads=threads, workdir=workdir or ws.paths.scratch
    )
    inputs: dict[str, ArtifactRef] = {}
    for side, spec in (("queries", config.queries), ("targets", config.targets)):
        if isinstance(spec, DomainSequences):
            inputs[f"{side}/domain"] = ws.artifacts.ref(spec.domain.data)
    execution = {"mmseqs.version": str(ws.artifacts.manifest(ref).meta["mmseqs_version"])}
    return StageResult(outputs={"hits": ref}, inputs=inputs, packages={"pandas": pd.__version__}, execution=execution)


STAGE = Stage("homology", HomologySearchSpec, _run, outputs="hits")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.homology --config <yaml> [--pin NAME] [--option mmseqs=PATH]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
