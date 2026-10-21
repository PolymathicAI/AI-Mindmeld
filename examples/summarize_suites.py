"""Summarize the worked pipeline's two suites (``examples/guides/extending.md`` §6): accuracy at score > 0 and AUROC.

    python examples/summarize_suites.py --pin-prefix PREFIX

reads ``PREFIX/suite-init/scores`` and ``PREFIX/suite-trained/scores`` from the workspace of ``LIP_DATA`` (a suite not
pinned yet is skipped) and prints one line per suite, condition and task, e.g. for ``--pin-prefix mine/qwen``::

    suite-init     own  DNA-binding  n=8 pos=4 acc=0.50 AUROC=0.50

``PREFIX`` is the run's prefix, as ``configure_pipeline.py`` names it: ``mine/qwen`` for §6's affine run,
``<pin prefix>-<kind>`` for ``mlp`` and ``xattn3``. The AUROC counts ties as half.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd

from latent_interconnect.workspace import Workspace

SUITES = ("suite-init", "suite-trained")


def summary_lines(scores: pd.DataFrame, suite: str) -> list[str]:
    """Return one line per condition and task of a suite's ``scores.parquet`` table."""
    lines = []
    for (cond, task), g in scores.groupby(["condition.object", "task"]):
        y, s = g["label"].to_numpy(), g["score"].to_numpy()
        auc = np.mean([(p > n) + 0.5 * (p == n) for p in s[y == 1] for n in s[y == 0]])
        lines.append(
            f"{suite:14} {cond:4} {task:12} n={len(g)} pos={int(y.sum())} "
            f"acc={np.mean((s > 0) == (y == 1)):.2f} AUROC={auc:.2f}"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """The command line (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pin-prefix", required=True, help="the run's pin prefix, e.g. mine/qwen or mine/gemma4b-mlp")
    args = parser.parse_args(argv)
    ws = Workspace.from_env()
    pins = ws.pins()
    for suite in SUITES:
        name = f"{args.pin_prefix}/{suite}/scores"
        if name not in pins:
            sys.stdout.write(f"{suite:14} not pinned ({name})\n")
            continue
        table = pd.read_parquet(ws.artifacts.path(ws.artifacts.ref(pins[name].sha256)) / "scores.parquet")
        sys.stdout.write("".join(line + "\n" for line in summary_lines(table, suite)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
