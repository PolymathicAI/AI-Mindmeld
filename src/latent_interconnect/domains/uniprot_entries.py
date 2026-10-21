"""UniProtKB's flat file (``uniprot_sprot.dat.gz``), read entry by entry: the publisher's structured fields.

The protein domain's entry table holds UniProt's ``protein_names`` text, in which the recommended name, the
alternative names and the bracket blocks are run together, and it holds no fragment flag. The flat file of the same
release states both as fields of the entry: the ``DE   RecName: Full=`` line and the ``DE   Flags:`` line. A protein
data spec that cites the file (``domains.protein.ProteinDataSpec.entries``) reads them from it.

:func:`read_entries` streams the compressed file (never expanded to disk) and yields one :class:`UniProtEntry` per
entry, holding the lines of the line types the package reads; the entry's properties parse the description lines,
and ``domains.keyword_evidence`` reads the comment, feature and GO lines.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

__all__ = ["UniProtEntry", "read_entries"]

_CONTENT = 5  # a line is its two-letter code, three spaces, then the content
_RECOMMENDED = "RecName: Full="
_FLAGS = "Flags:"
_FRAGMENT_FLAGS = frozenset({"Fragment", "Fragments"})
_EVIDENCE = re.compile(r"\s*\{ECO:[^}]*\}")
_ANY_EVIDENCE = re.compile(r"\{[^}]*\}")
_GO = "GO;"
_KEPT_CODES = frozenset({"ID", "AC", "DE", "KW", "OC", "CC", "FT"})


@dataclass(frozen=True)
class UniProtEntry:
    """One entry of the flat file: the lines of the line types the package reads, without their line codes.

    Attributes:
        accession: the primary accession (the first of the ``AC`` lines).
        entry_name: the entry name (the ``ID`` line's first token, e.g. ``"001R_FRG3G"``).
        length: the sequence length the ``ID`` line states.
        description: the ``DE`` lines (names, EC numbers, flags), indentation kept.
        keyword_lines: the ``KW`` lines.
        taxon_lines: the ``OC`` lines (the classification above the species).
        comments: the ``CC`` lines.
        features: the ``FT`` lines.
        go: the ``DR   GO;`` lines' content after ``"GO; "`` (id, aspect and term name, evidence).
    """

    accession: str
    entry_name: str
    length: int
    description: tuple[str, ...]
    keyword_lines: tuple[str, ...]
    taxon_lines: tuple[str, ...]
    comments: tuple[str, ...]
    features: tuple[str, ...]
    go: tuple[str, ...]

    @property
    def recommended_name(self) -> str:
        """Return UniProt's recommended name: the entry's own ``RecName: Full=`` value, evidence tags removed.

        The entry's own line is the unindented one; the names of an ``Includes:`` or ``Contains:`` block are indented.

        Raises:
            ValueError: if the entry has no recommended name.
        """
        for line in self.description:
            if line.startswith(_RECOMMENDED):
                return _EVIDENCE.sub("", line.removeprefix(_RECOMMENDED)).rstrip(";")
        raise ValueError(f"UniProt entry {self.accession}: no 'RecName: Full=' line")

    @property
    def fragment(self) -> bool:
        """Return UniProt's fragment flag: the ``Flags:`` line names ``Fragment`` or ``Fragments``."""
        flags = {f.strip() for line in self.description if line.startswith(_FLAGS) for f in line[6:].split(";")}
        return bool(flags & _FRAGMENT_FLAGS)

    @property
    def keywords(self) -> frozenset[str]:
        """Return the entry's keywords (the ``KW`` lines' ``;``-separated items, evidence tags removed)."""
        text = _ANY_EVIDENCE.sub("", " ".join(self.keyword_lines)).strip().removesuffix(".")
        return frozenset(k.strip() for k in text.split(";") if k.strip())

    @property
    def taxa(self) -> frozenset[str]:
        """Return the taxon nodes of the entry's classification (the ``OC`` lines' ``;``-separated items)."""
        text = " ".join(self.taxon_lines).strip().removesuffix(".")
        return frozenset(t.strip() for t in text.split(";") if t.strip())


def read_entries(path: Path) -> Iterator[UniProtEntry]:
    """Yield the entries of a gzip-compressed UniProtKB flat file, in file order.

    Raises:
        ValueError: if an entry has no ``ID`` or ``AC`` line, or the file ends inside an entry (a truncated file).
    """
    lines: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            code = line[:2]
            if code == "//":
                yield _entry(lines)
                lines = {}
            elif code in _KEPT_CODES:
                lines.setdefault(code, []).append(line[_CONTENT:].rstrip("\n"))
            elif code == "DR" and line.startswith(_GO, _CONTENT):
                lines.setdefault("GO", []).append(line[_CONTENT + len(_GO) :].strip())
    if lines:
        raise ValueError(f"{path.name}: the last entry has no '//' terminator: the file is truncated")


def _entry(lines: dict[str, list[str]]) -> UniProtEntry:
    if "ID" not in lines or "AC" not in lines:
        raise ValueError(f"a UniProt entry lacks its ID or AC line: {lines.get('ID') or lines.get('AC')}")
    identity = lines["ID"][0].split()
    return UniProtEntry(
        accession=lines["AC"][0].split(";")[0].strip(),
        entry_name=identity[0],
        length=int(identity[2]),
        description=tuple(lines.get("DE", ())),
        keyword_lines=tuple(lines.get("KW", ())),
        taxon_lines=tuple(lines.get("OC", ())),
        comments=tuple(lines.get("CC", ())),
        features=tuple(lines.get("FT", ())),
        go=tuple(lines.get("GO", ())),
    )
