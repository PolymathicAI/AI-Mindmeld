"""Embed your own objects for a connector, with the encoder recipe the connector was trained on (any extractor).

    python examples/embed.py BUNDLE.json SEQUENCES.fasta OUT_DIR [--device cpu]
    python examples/embed.py RECIPE.yaml SEQUENCES.fasta OUT_DIR [--device cpu]

Run it in the interpreter of the recipe's ``environment``: the encoder's own environment (ESM-C: the ``esm`` package,
``examples/README.md`` §6), or the release environment for a recipe with ``environment: lip``. The first
argument says which recipe and layers to use:

- ``BUNDLE.json``, a connector bundle's ``bundle.json`` (``release_download`` keeps it as
  ``$LIP_DATA/lip/bundles/<name>.json``): its ``extraction`` names the recipe (model, revision, pooling) and the
  layers the connector reads;
- ``RECIPE.yaml``, an encoder recipe (``latent_interconnect.encoders.recipes.EncoderRecipe``), for a connector you
  trained yourself: the layers are the recipe's ``read``, in its order (``examples/guides/extending.md`` §6).

The recipe's ``plugins`` are imported first (``latent_interconnect.encoders.jobs.load_extractor``, as the extract
stage does), so a third-party extractor such as ``examples/esm2_extractor.py`` resolves; its module must be importable
(``examples/guides/extending.md`` §2, "Where plugin code lives").

Each object's pooled vector, the layers' pooled states concatenated in that order, is written to ``OUT_DIR/<id>.npy``.
When the recipe has ``residues: true`` (a connector of a residue kind such as ``xattn`` reads per-residue states), the
per-residue states, the same layers concatenated along features, are also written to ``OUT_DIR/<id>.residues.npy``
(float32, one row per residue). ``examples/ask.py`` takes either file and reads what the connector needs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from latent_interconnect.config import load_config
from latent_interconnect.encoders.base import RawObject
from latent_interconnect.encoders.jobs import load_extractor
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import OUTPUT_LAYER

__all__ = ["as_float32", "embed", "main", "read_extraction", "read_fasta"]


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


def _layers(read: Any) -> list[int]:
    return [OUTPUT_LAYER] if read == "output" else [int(layer) for layer in read]


def read_extraction(path: Path) -> tuple[dict[str, Any], list[int]]:
    """Return (recipe, layers) from a connector's ``bundle.json`` or an encoder recipe YAML (module docstring)."""
    if path.suffix in (".yaml", ".yml"):
        recipe = load_config(path, EncoderRecipe).model_dump(mode="json")
        return recipe, _layers(recipe["read"])
    extraction = json.loads(path.read_text())["extraction"]
    return extraction["recipe"], _layers(extraction["read_layers"] or "output")


def as_float32(rows: np.ndarray) -> np.ndarray:
    """Return stored residue rows as float32 (bf16 is stored as ``uint16`` bit patterns)."""
    if rows.dtype == np.uint16:
        return (rows.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return rows.astype(np.float32)


def embed(recipe: dict[str, Any], layers: list[int], objects: dict[str, str], out: Path, device: str = "cpu") -> None:
    """Embed ``objects`` (id -> value) with ``recipe`` and write ``out/<id>.npy`` (and residues; module docstring)."""
    extractor = load_extractor(recipe, device)
    batch = extractor.extract([RawObject(k, v) for k, v in objects.items()])
    out.mkdir(parents=True, exist_ok=True)
    for i, oid in enumerate(batch.object_ids):
        vec = np.concatenate([batch.pooled[layer][i] for layer in layers]).astype(np.float32)
        np.save(out / f"{oid}.npy", vec)
        sys.stdout.write(f"{oid}: {vec.shape[0]} values -> {out / f'{oid}.npy'}\n")
        if batch.residues:
            res = np.concatenate([as_float32(np.asarray(batch.residues[layer][i])) for layer in layers], axis=1)
            np.save(out / f"{oid}.residues.npy", res)
            sys.stdout.write(f"{oid}: {res.shape[0]} residues x {res.shape[1]} -> {out / f'{oid}.residues.npy'}\n")
    for oid, reason in batch.skipped.items():
        sys.stdout.write(f"{oid}: skipped ({reason})\n")


def main(argv: list[str] | None = None) -> None:
    """The command line (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path, help="a connector's bundle.json or an encoder recipe .yaml")
    parser.add_argument("fasta", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args(argv)
    recipe, layers = read_extraction(args.bundle)
    embed(recipe, layers, read_fasta(args.fasta), args.out, args.device)


if __name__ == "__main__":
    main()
