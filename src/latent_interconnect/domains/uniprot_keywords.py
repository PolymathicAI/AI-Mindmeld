"""UniProt's keyword list: each keyword's accession and category, from a hash-pinned ``keywlist.txt``.

The list is UniProt's flat file of keyword entries (``ID`` the keyword,
``AC`` its ``KW-xxxx`` accession, ``CA`` its category) and category entries (``IC``). :func:`read_keywords` reads it
after checking its sha256 (``domains.tables.read_source``); :func:`parse_keywords_list` is the parser. The categories
anchor the protein access figure's meaning bins (the Fig 5a analysis config states each bin's category).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from latent_interconnect.domains.tables import SourceFile, read_source

__all__ = ["CATEGORIES", "UniProtKeyword", "parse_keywords_list", "read_keywords"]

CATEGORIES = frozenset(
    {
        "Biological process",
        "Cellular component",
        "Coding sequence diversity",
        "Developmental stage",
        "Disease",
        "Domain",
        "Ligand",
        "Molecular function",
        "PTM",
        "Technical term",
    }
)
"""UniProt's ten keyword categories (the ``IC`` entries)."""


@dataclass(frozen=True)
class UniProtKeyword:
    """One keyword entry.

    Attributes:
        name: the keyword (``ID``, without its final period).
        accession: ``KW-xxxx``.
        category: its category (``CA``), one of :data:`CATEGORIES`.
    """

    name: str
    accession: str
    category: str


def parse_keywords_list(text: str) -> dict[str, UniProtKeyword]:
    """Return keyword → entry from ``keywlist.txt``'s text (category entries are checked, not returned).

    Raises:
        ValueError: if an entry lacks its accession or category, a category is not one of :data:`CATEGORIES`, the
            categories the file declares differ from them, a keyword repeats, or the last entry has no ``//``
            terminator (a truncated file).
    """
    keywords: dict[str, UniProtKeyword] = {}
    declared: set[str] = set()
    entry: dict[str, str] = {}
    for line in text.splitlines():
        code, value = line[:2], line[5:].strip()
        if code in ("ID", "IC"):
            entry = {"kind": code, "name": value.removesuffix(".")}
        elif code == "AC" and entry:
            entry["accession"] = value
        elif code == "CA" and entry:
            entry["category"] = value.removesuffix(".")
        elif code == "//" and entry:
            if entry["kind"] == "IC":
                declared.add(entry["name"])
            else:
                missing = sorted({"accession", "category"} - set(entry))
                if missing or entry.get("category") not in CATEGORIES:
                    raise ValueError(
                        f"keyword {entry['name']!r}: lacks {missing} or has category {entry.get('category')!r}"
                    )
                if entry["name"] in keywords:
                    raise ValueError(f"keyword {entry['name']!r} repeats")
                keywords[entry["name"]] = UniProtKeyword(entry["name"], entry["accession"], entry["category"])
            entry = {}
    if entry:
        raise ValueError(f"entry {entry.get('name')!r} has no '//' terminator: the file is truncated")
    if declared != CATEGORIES:
        raise ValueError(f"the file declares categories {sorted(declared)}, not UniProt's {sorted(CATEGORIES)}")
    return keywords


def read_keywords(data_root: Path, source: SourceFile) -> dict[str, UniProtKeyword]:
    """Read and parse the pinned list under ``data_root`` (its sha256 checked first)."""
    return parse_keywords_list(read_source(data_root, source).read_text(encoding="utf-8"))
