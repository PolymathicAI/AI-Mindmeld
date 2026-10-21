"""The molecule text's item rule: an item that names an evaluation protein or an EC number is dropped.

As in the original molecule text, molecule training text must not
state a relation to an evaluation protein, the proteins of the cross-domain candidate tables. The terms are each
protein's names (recommended and alternative, at least 5 characters with 4 letters in a row, not a generic word alone,
matched case-insensitively as whole phrases), its primary gene symbol (at least 3 characters, case-sensitive, a whole
token) and its EC numbers (``EC a.b.c.d``); any EC number at all (``EC 3.``) blocks too. This is the original matcher,
ported; its term counts on the pinned inputs are the original run's (226 accessions, 610 names, 185 genes, 176 EC
numbers).
"""

from __future__ import annotations

import csv
import gzip
import io
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import Field

from latent_interconnect.config import Config
from latent_interconnect.domains.tables import SourceFile, read_source

__all__ = [
    "EC_NUMBER",
    "GENERIC_PROTEIN_WORDS",
    "EvalProteinMatcher",
    "EvalProteins",
    "load_eval_proteins",
    "split_protein_names",
]

EC_NUMBER = re.compile(r"\bEC \d+\.")
"""Any EC number (the original ``EC_RE``)."""
GENERIC_PROTEIN_WORDS = frozenset(
    {
        "protein", "enzyme", "kinase", "phosphatase", "reductase", "oxidase", "dehydrogenase", "transferase",
        "hydrolase", "synthase", "synthetase", "isomerase", "ligase", "lyase", "transporter", "receptor", "channel",
        "subunit", "domain", "factor", "chain", "precursor", "uncharacterized protein", "putative protein",
    }
)  # fmt: skip
"""Names that do not name one protein when they stand alone."""
_COLUMNS = {"entry": "accession", "protein names": "protein_name", "gene names (primary)": "gene", "ec number": "ec"}
_EC_ENTRY = re.compile(r"EC [\d.\-n]+")
_EC_FULL = re.compile(r"\d+\.\d+\.\d+\.\d+")
_MIN_NAME, _MIN_GENE = 5, 3


class EvalProteins(Config):
    """The evaluation proteins' sources.

    Attributes:
        uniprot: the Swiss-Prot table the names, genes and EC numbers are read from (``Entry``, ``Protein names``,
            ``Gene Names (primary)``, ``EC number``; tab-separated, gzipped).
        candidates: the candidate tables whose ``accession`` column lists the evaluation proteins.
    """

    uniprot: SourceFile
    candidates: tuple[SourceFile, ...] = Field(min_length=1)


def split_protein_names(field: str) -> list[str]:
    """Return a UniProt ``Protein names`` cell's names: ``Name (Alt 1) (Alt 2) [Cleaved into: ...]``.

    Split at top-level `` (`` groups only, so a name holding parentheses stays whole; bracketed parts are dropped and
    ``EC ...`` entries are not names.
    """
    s = re.sub(r"\s*\[[^\]]*\]", "", field or "")
    out, cur, depth = [], "", 0
    for i, ch in enumerate(s):
        if ch == "(" and depth == 0 and s[i - 1 : i] == " ":
            out.append(cur.strip())
            cur, depth = "", 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and (i + 1 == len(s) or s[i + 1] == " "):
                out.append(cur.strip())
                cur = ""
                continue
        cur += ch
    out.append(cur.strip())
    return [x for x in out if x and not _EC_ENTRY.fullmatch(x)]


@dataclass(frozen=True)
class EvalProteinMatcher:
    """The evaluation proteins' terms, and the item rule.

    Attributes:
        names: protein names, matched case-insensitively as whole phrases.
        genes: gene symbols, matched case-sensitively as whole tokens.
        ecs: the proteins' full EC numbers.
    """

    names: frozenset[str]
    genes: frozenset[str]
    ecs: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_patterns", self._compile())

    def _compile(self) -> list[re.Pattern[str]]:
        def alt(xs: frozenset[str]) -> str:
            return "|".join(re.escape(x) for x in sorted(xs, key=len, reverse=True))

        out = []
        if self.names:
            out.append(re.compile(r"(?<![A-Za-z0-9\-])(?:" + alt(self.names) + r")(?![A-Za-z0-9])", re.IGNORECASE))
        if self.genes:
            out.append(re.compile(r"(?<![A-Za-z0-9\-])(?:" + alt(self.genes) + r")(?![A-Za-z0-9])"))
        if self.ecs:
            out.append(re.compile(r"\bEC (?:" + alt(self.ecs) + r")(?![\d.])"))
        return out

    def hits(self, text: str) -> list[str]:
        """Return the evaluation-protein names, gene symbols and EC numbers found in ``text``."""
        patterns: list[re.Pattern[str]] = self._patterns  # type: ignore[attr-defined]
        return [m.group(0) for p in patterns for m in p.finditer(text)]

    def blocks(self, text: str) -> bool:
        """Return whether the item rule drops ``text``: it names an evaluation protein or any EC number."""
        return bool(EC_NUMBER.search(text)) or bool(self.hits(text))


def _rows(path: Path) -> Iterator[dict[str, str]]:
    """The UniProt table's rows keyed by short names (header names mapped by content, not position)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        columns = {i: _COLUMNS.get(h.strip().lower()) for i, h in enumerate(header)}
        missing = set(_COLUMNS.values()) - set(columns.values())
        if missing:
            raise ValueError(f"{path}: the UniProt table lacks columns {sorted(missing)}")
        for row in reader:
            yield {key: v for i, v in enumerate(row) if (key := columns.get(i))}


@lru_cache(maxsize=2)
def _matcher(data_root: Path, spec_json: str) -> EvalProteinMatcher:
    spec = EvalProteins.model_validate_json(spec_json)
    accessions: set[str] = set()
    for source in spec.candidates:
        text = read_source(data_root, source).read_text(encoding="utf-8")
        accessions |= {r["accession"] for r in csv.DictReader(io.StringIO(text), delimiter="\t")}
    names: set[str] = set()
    genes: set[str] = set()
    ecs: set[str] = set()
    for row in _rows(read_source(data_root, spec.uniprot)):
        if row.get("accession") not in accessions:
            continue
        gene = row.get("gene", "").split(";")[0].strip()
        if len(gene) >= _MIN_GENE:
            genes.add(gene)
        names |= {
            n
            for n in split_protein_names(row.get("protein_name", ""))
            if n != gene
            and len(n) >= _MIN_NAME
            and n.lower() not in GENERIC_PROTEIN_WORDS
            and re.search(r"[A-Za-z]{4}", n)
        }
        ecs |= {e.strip() for e in row.get("ec", "").split(";") if _EC_FULL.fullmatch(e.strip())}
    return EvalProteinMatcher(frozenset(names), frozenset(genes), frozenset(ecs))


def load_eval_proteins(spec: EvalProteins, data_root: Path) -> EvalProteinMatcher:
    """Return the matcher of the evaluation proteins ``spec`` cites (read once per process).

    Raises:
        IntegrityError: if a source's sha256 differs from its citation.
        ValueError: if the UniProt table lacks a column.
    """
    return _matcher(Path(data_root), spec.model_dump_json())
