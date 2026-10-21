"""Print a read scan's layer combinations next to its single layers (``evaluation.read_scan.compare_reads``; CPU).

``python -m latent_interconnect.run.read_scan_report NAME [--drop PROBE ...]``: the ``--pin`` name of a ``read_scan``
run with layers and combinations (e.g. ``read-scan-joint/kimi-k2.6_esmc-600m``), whose ``numbers`` output is read
through the pin ``NAME/numbers``, and the object probes declared unstable for the encoder, left out of the means.
Not a stage: it writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.evaluation.read_scan import compare_reads
from latent_interconnect.evaluation.scores import NUMBERS_KIND, read_numbers
from latent_interconnect.workspace import Workspace

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.read_scan_report NAME [--drop PROBE ...]``."""
    parser = argparse.ArgumentParser(prog="python -m latent_interconnect.run.read_scan_report")
    parser.add_argument("name", help="the --pin name of a read scan with layers and combinations")
    parser.add_argument("--drop", action="append", default=[], help="an object probe declared unstable")
    args = parser.parse_args(argv)
    ws = Workspace.from_env()
    pin = ws.pins().get(f"{args.name}/numbers")
    if pin is None:
        parser.error(f"no pin {args.name}/numbers")
    numbers = read_numbers(ws.artifacts, ArtifactRef(pin.sha256, NUMBERS_KIND))
    sys.stdout.write(compare_reads(numbers, args.drop).to_string(float_format="%.3f") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
