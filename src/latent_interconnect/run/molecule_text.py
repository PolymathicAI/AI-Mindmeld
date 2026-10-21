"""Entry point: render a molecule text table from its declared sources and commit it (``domains.molecule_text``).

The table is the molecule domain's ``text`` source: its file (``domains.molecule_text.TEXT_FILE`` in the committed
artifact) is cited by path and sha256 like the paper's field table, so a domain moves from one to the other by a config
change.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pyarrow

from latent_interconnect.domains.molecule_text import MoleculeTextSpec, build_text_table, commit_text_table
from latent_interconnect.receivers import open_renderer
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(config: MoleculeTextSpec, ws: Workspace) -> StageResult:
    """Output ``text`` (molecule_text).

    Its sources are files under the data root, hash-checked by the build; the receiver's tokenizer (at its pinned
    revision) counts the trim's tokens.
    """
    import transformers  # noqa: PLC0415  (the optional `hf` extra, which the receiver's tokenizer needs)

    built = build_text_table(config, ws.paths.data, open_renderer(config.receiver))
    ref = commit_text_table(ws.artifacts, config, built)
    packages = {"pandas": pd.__version__, "pyarrow": pyarrow.__version__, "transformers": transformers.__version__}
    return StageResult(outputs={"text": ref}, packages=packages)


STAGE = Stage("molecule_text", MoleculeTextSpec, _run, outputs="text")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.molecule_text --config <yaml> [--pin NAME]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
