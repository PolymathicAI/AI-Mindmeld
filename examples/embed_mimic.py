"""Embed your own protein sequences with the public MIMIC 1.0 for a MIMIC connector.

    python examples/embed_mimic.py BUNDLE.json SEQUENCES.fasta OUT_DIR [--device cpu]

Run it in the MIMIC environment (the public ``mimic`` package, ``env/constraints/mimic.txt``;
``examples/README.md``, "MIMIC from raw inputs"). The connector's ``bundle.json`` names the layers it reads (16
and 20); the recipe is ``configs/encoders/mimic-1.0.yaml`` (``polymathic-ai/MIMIC`` at its pinned commit, residue mean).
Each sequence's vector, the read layers' pooled
states concatenated in the connector's order, is written to ``OUT_DIR/<id>.npy`` for ``examples.ask``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from latent_interconnect.config import load_config
from latent_interconnect.encoders.base import EXTRACTORS, RawObject
from latent_interconnect.encoders.extractors import mimic  # noqa: F401  (registers the extractor)
from latent_interconnect.encoders.recipes import EncoderRecipe

RECIPE = Path(__file__).resolve().parents[1] / "configs" / "encoders" / "mimic-1.0.yaml"


def read_fasta(path: Path) -> dict[str, str]:
    """Return id -> sequence; the id is the first word of the header."""
    records: dict[str, str] = {}
    name = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            records[name] = ""
        elif line and name is not None:
            records[name] += line
    return records


def main() -> None:
    """The command line (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("fasta", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    args = parser.parse_args()
    layers = json.loads(args.bundle.read_text())["extraction"]["read_layers"]
    recipe = load_config(args.recipe, EncoderRecipe).model_dump()
    if not set(layers) <= set(recipe["read"]):
        sys.exit(f"the connector reads layers {layers}; {args.recipe} reads {recipe['read']}")
    extractor = EXTRACTORS.get(recipe["extractor"])(recipe, args.device)
    start = time.perf_counter()
    extractor.load()
    loaded = time.perf_counter()
    batch = extractor.extract([RawObject(k, v) for k, v in read_fasta(args.fasta).items()])
    sys.stdout.write(f"load {loaded - start:.1f} s, extract {time.perf_counter() - loaded:.1f} s\n")
    args.out.mkdir(parents=True, exist_ok=True)
    for i, oid in enumerate(batch.object_ids):
        vec = np.concatenate([batch.pooled[layer][i] for layer in layers]).astype(np.float32)
        np.save(args.out / f"{oid}.npy", vec)
        sys.stdout.write(f"{oid}: {vec.shape[0]} values -> {args.out / f'{oid}.npy'}\n")
    for oid, reason in batch.skipped.items():
        sys.stdout.write(f"{oid}: skipped ({reason})\n")


if __name__ == "__main__":
    main()
