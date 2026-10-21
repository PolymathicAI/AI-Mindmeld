"""Make a small protein domain from your own FASTA file, for the worked pipeline in ``examples/guides/extending.md``.

    python examples/make_domain.py SEQUENCES.fasta ANNOTATIONS.tsv TEMPLATE.yaml OUT.yaml [--subdir mine/proteins]

The domain stage (``latent_interconnect.run.domain``) builds the ``protein`` domain from four source files, cited by
path under ``$LIP_DATA`` and sha256: an entry table, an annotation table, a split table and a cluster table. This
script writes those four files from

- ``SEQUENCES.fasta``: one record per protein. The id is the header's first word, or its middle field for a UniProt
  header (``>sp|P69905|HBA_HUMAN ...``); the rest of the header, up to an ``OS=`` tag, is the protein's name.
- ``ANNOTATIONS.tsv``: tab-separated, one row per FASTA id, columns ``accession``, ``split`` (``train``, ``val`` or
  ``test``), ``keywords`` (``;``-separated UniProt keywords), ``lineage`` (``, ``-separated taxa) and ``caption`` (a
  sentence about the protein); an optional ``cluster`` column groups homologs (each protein is its own cluster
  without it).

It then writes ``OUT.yaml``: the domain-stage config ``TEMPLATE.yaml`` (labels, tasks, frames and capture recipes)
with its four sources filled in. The files go under ``$LIP_DATA/<subdir>/``. The example data in ``examples/data/``
are 32 short UniProtKB/Swiss-Prot entries (CC BY 4.0, https://www.uniprot.org), fetched on 2026-10-19.

Splits matter: training reads only ``train`` objects, and evaluation should read ``val`` or ``test`` objects.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

COLUMNS = ("accession", "split", "keywords", "lineage", "caption")
"""The annotation table's required columns."""


def read_fasta(path: Path) -> dict[str, tuple[str, str]]:
    """Return id → (name, sequence) from a FASTA file (module docstring for the header rule)."""
    records: dict[str, tuple[str, str]] = {}
    oid = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith(">"):
            first, _, rest = line[1:].partition(" ")
            parts = first.split("|")
            oid = parts[1] if len(parts) == 3 else first
            if oid in records:
                raise ValueError(f"{path}: id {oid!r} repeats")
            records[oid] = (rest.split(" OS=")[0].strip() or oid, "")
        elif line and oid is not None:
            name, seq = records[oid]
            records[oid] = (name, seq + line)
    return records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sources(fasta: Path, annotations: Path, root: Path, subdir: str) -> dict[str, dict[str, str]]:
    """Write the four source files under ``root/subdir`` and return their ``{path, sha256}`` by spec field."""
    seqs = read_fasta(fasta)
    notes = pd.read_csv(annotations, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in notes.columns]
    if missing:
        raise ValueError(f"{annotations}: missing columns {missing}")
    notes = notes.set_index("accession")
    if set(notes.index) != set(seqs):
        raise ValueError(f"FASTA and annotation ids differ: {sorted(set(notes.index) ^ set(seqs))[:5]}")
    bad = sorted(set(notes["split"]) - {"train", "val", "test"})
    if bad:
        raise ValueError(f"{annotations}: unknown splits {bad}")
    ids = list(seqs)
    table = pd.DataFrame(
        {
            "accession": ids,
            "protein_names": [seqs[i][0] for i in ids],
            "sequence": [seqs[i][1] for i in ids],
            "length": [len(seqs[i][1]) for i in ids],
            "keywords": [notes.at[i, "keywords"] for i in ids],
            "caption": [notes.at[i, "caption"] for i in ids],
        }
    )
    clusters = notes["cluster"] if "cluster" in notes.columns else pd.Series(ids, index=ids)
    out = root / subdir
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "table": out / "table.parquet",
        "annotations": out / "annotations.parquet",
        "split": out / "split.parquet",
        "clusters": out / "clusters.tsv",
    }
    table.to_parquet(files["table"], index=False)
    pd.DataFrame({"accession": ids, "lineage": [notes.at[i, "lineage"] for i in ids]}).to_parquet(
        files["annotations"], index=False
    )
    pd.DataFrame({"accession": ids, "split": [notes.at[i, "split"] for i in ids]}).to_parquet(
        files["split"], index=False
    )
    pd.DataFrame({"cluster": [clusters[i] for i in ids], "member": ids}).to_csv(
        files["clusters"], sep="\t", header=False, index=False
    )
    return {k: {"path": str(p.relative_to(root)), "sha256": _sha256(p)} for k, p in files.items()}


def main() -> None:
    """The command line (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fasta", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("template", type=Path, help="a domain-stage config whose four sources are filled in")
    parser.add_argument("out", type=Path, help="the domain-stage config to write")
    parser.add_argument("--subdir", default="mine/proteins", help="where under $LIP_DATA the source files go")
    args = parser.parse_args()
    root = os.environ.get("LIP_DATA")
    if not root:
        raise SystemExit("set LIP_DATA to the data root (any writable directory)")
    config = yaml.safe_load(args.template.read_text())
    config["spec"].update(write_sources(args.fasta, args.annotations, Path(root), args.subdir))
    args.out.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=120))
    split = pd.read_csv(args.annotations, sep="\t", dtype=str)["split"].value_counts()
    counts = ", ".join(f"{k} {v}" for k, v in sorted(split.items()))
    sys.stdout.write(f"{sum(split)} proteins ({counts}) -> {args.out}\n")


if __name__ == "__main__":
    main()
