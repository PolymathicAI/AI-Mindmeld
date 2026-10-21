r"""A ChEBI release, read from its OBO file: the one ChEBI reader of the package.

The molecule text renderer (``domains.molecule_text``) and the zero-shot screen's ChEBI role detector
(``build.molecule_detectors``) read the same hash-pinned release through :func:`load_release` (read once per process).
The reading is the paper's, with the synonyms the detector needs:

- every ``[Term]`` stanza whose id starts with ``CHEBI:`` is kept, obsolete ones included (a molecule's parent may be
  obsolete and is still named); :meth:`ChebiRelease.descendants` walks the non-obsolete terms only;
- quoted values (names, definitions, synonyms, property values) have their OBO 1.2 escapes resolved (:func:`unescape`:
  ``\n`` a newline, ``\t`` a tab, ``\W`` a space, any other ``\x`` the character ``x``; ChEBI 255 escapes braces);
- relationships are named by their label (``has_role``, ``is_conjugate_base_of``, ...), from the RO id or, in 2026
  releases, from the comment after ``!``;
- property values are typed by their key (SMILES, formula, charge, average mass; the monoisotopic mass is skipped) and
  kept as the release's text, the first of each kind.

ChEBI names as identities: names reach the package from two sources that spell some of them differently. The OBO file
escapes characters (in the pinned file's names: 3,930 ``\{``, 3,926 ``\}`` and 53 ``\"``, nothing else), and the
ChEBI entry table writes them plain and collapses a doubled space; on the paper's training compounds, 150 have a direct
class whose OBO name differs from the entry table's. :func:`chebi_name` is the one written
form (escapes resolved, Unicode NFC, whitespace collapsed); after it the two sources agree on every direct class and
functional parent of the training compounds. :func:`name_key` is the identity (the written form, case-folded): on the
pinned release two of the 218,533 non-obsolete terms share a key (CHEBI:68794 "neoechinulin A" and CHEBI:193004
"neoechinulin  A"), neither a class, a functional parent or a compound of the molecule set. Every builder of a
training-text field holding ChEBI names writes :func:`chebi_name` (``domains.molecule``), and every reader that
compares names compares :func:`name_key` (``data.relational_records_molecule``). Both are idempotent on the pinned
release: no name holds a backslash once unescaped.
"""

from __future__ import annotations

import gzip
import re
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path

from latent_interconnect.domains.tables import SourceFile, read_source

__all__ = ["ChebiEntry", "ChebiRelease", "chebi_name", "load_release", "name_key", "read_release", "unescape"]

_RO = {
    "RO:0000087": "has_role",
    "RO:0018033": "is_conjugate_base_of",
    "RO:0018034": "is_conjugate_acid_of",
    "RO:0018036": "is_tautomer_of",
    "RO:0018038": "has_functional_parent",
    "RO:0018039": "is_enantiomer_of",
    "BFO:0000051": "has_part",
}
_REL_LABELS = (
    "has role",
    "is conjugate base of",
    "is conjugate acid of",
    "is tautomer of",
    "has functional parent",
    "is enantiomer of",
    "has parent hydride",
    "is substituent group from",
    "has part",
)
_PV = re.compile(r'^property_value:\s+(\S+)\s+"(.*)"\s+(\S+)')
_DEF = re.compile(r'^def:\s+"(.*)"\s*\[')
_SYNONYM = re.compile(r'"((?:[^"\\]|\\.)*)" (EXACT|RELATED|BROAD|NARROW)\b')
_STARS = re.compile(r"(\d)[_:]STAR")
_ESCAPE = re.compile(r"\\(.)")
_ESCAPED = {"n": "\n", "t": "\t", "W": " "}
_SPACE = re.compile(r"\s+")
_RELEASE_KEY = "data-version"
_CURATED = 3  # the star subset of the entries ChEBI curated


def unescape(text: str) -> str:
    r"""Return an OBO value with its escapes resolved (OBO 1.2: ``\{`` is ``{``, ``\n`` a newline, ``\W`` a space)."""
    return _ESCAPE.sub(lambda m: _ESCAPED.get(m.group(1), m.group(1)), text)


def chebi_name(text: str) -> str:
    """Return a ChEBI name as written for training text: :func:`unescape`, Unicode NFC, whitespace runs one space."""
    return _SPACE.sub(" ", unicodedata.normalize("NFC", unescape(text))).strip()


def name_key(text: str) -> str:
    """Return a ChEBI name's identity: :func:`chebi_name`, case-folded."""
    return chebi_name(text).casefold()


@dataclass(frozen=True)
class ChebiEntry:
    """One ``[Term]`` of a ChEBI release.

    Attributes:
        name: the name, or ``""``.
        definition: the definition text, or None.
        synonyms: the synonyms as ``(text, scope)`` (``EXACT``, ``RELATED``, ``BROAD``, ``NARROW``), in file order.
        is_a: the ``is_a`` parents, in file order.
        has_role: the ``has_role`` targets, in file order.
        rels: every relationship as ``(relation label, target)``, in file order.
        stars: the star subset (3 for the curated entries), or None.
        obsolete: whether the term is obsolete.
        smiles: the SMILES property value, or None.
        formula: the formula property value, or None.
        charge: the charge property value as written, or None.
        mass: the average-mass property value as written, or None.
    """

    name: str = ""
    definition: str | None = None
    synonyms: tuple[tuple[str, str], ...] = ()
    is_a: tuple[str, ...] = ()
    has_role: tuple[str, ...] = ()
    rels: tuple[tuple[str, str], ...] = ()
    stars: int | None = None
    obsolete: bool = False
    smiles: str | None = None
    formula: str | None = None
    charge: str | None = None
    mass: str | None = None

    def in_universe(self) -> bool:
        """Return whether the entry is a molecule the paper covers: non-obsolete, 3-star, a SMILES without ``*``."""
        return not self.obsolete and self.stars == _CURATED and bool(self.smiles) and "*" not in (self.smiles or "")

    def related(self, relation: str) -> list[str]:
        """Return the targets of ``relation``, in file order."""
        return [target for rel, target in self.rels if rel == relation]


@dataclass(frozen=True)
class ChebiRelease:
    """A parsed ChEBI release.

    Attributes:
        entries: CHEBI id → entry, obsolete ones included.
        version: the header's ``data-version`` (``"255"``), or ``""``.
    """

    entries: Mapping[str, ChebiEntry]
    version: str = ""

    @cached_property
    def _children(self) -> dict[str, frozenset[str]]:
        children: dict[str, set[str]] = {}
        for term_id, entry in self.entries.items():
            if not entry.obsolete:
                for parent in entry.is_a:
                    children.setdefault(parent, set()).add(term_id)
        return {k: frozenset(v) for k, v in children.items()}

    def descendants(self, term: str) -> list[str]:
        """Return ``term`` and every non-obsolete term below it in ``is_a``, ``term`` first, the rest sorted.

        Raises:
            KeyError: if the release has no non-obsolete term ``term``.
        """
        entry = self.entries.get(term)
        if entry is None or entry.obsolete:
            raise KeyError(f"{term} is not a (non-obsolete) term of the ChEBI release")
        seen, stack = {term}, [term]
        while stack:
            for child in self._children.get(stack.pop(), ()):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return [term, *sorted(seen - {term})]


@dataclass
class _Stanza:
    values: dict[str, object] = field(default_factory=dict)
    synonyms: list[tuple[str, str]] = field(default_factory=list)
    is_a: list[str] = field(default_factory=list)
    has_role: list[str] = field(default_factory=list)
    rels: list[tuple[str, str]] = field(default_factory=list)

    def entry(self) -> ChebiEntry:
        v = self.values

        def text(key: str) -> str | None:
            value = v.get(key)
            return value if isinstance(value, str) else None

        stars = v.get("stars")
        return ChebiEntry(
            name=text("name") or "",
            definition=text("def"),
            synonyms=tuple(self.synonyms),
            is_a=tuple(self.is_a),
            has_role=tuple(self.has_role),
            rels=tuple(self.rels),
            stars=stars if isinstance(stars, int) else None,
            obsolete=v.get("obsolete") is True,
            smiles=text("smiles"),
            formula=text("formula"),
            charge=text("charge"),
            mass=text("mass"),
        )


def _relation(rel: str, comment: str) -> str:
    """A relationship's label: a named relation as is; an id by its comment's label, else the RO table."""
    if not re.match(r"^[A-Z]+:\d+$", rel):
        return rel
    c = comment.strip()
    for label in _REL_LABELS:
        if c.startswith(label + " "):
            return label.replace(" ", "_")
    return _RO.get(rel, rel)


def _property_kind(key: str) -> str | None:  # noqa: PLR0911  (the key tests, in the paper's order)
    k = key.lower()
    if "smiles" in k:
        return "smiles"
    if "inchi" in k:
        return None
    if "formula" in k:
        return "formula"
    if k.endswith("charge"):
        return "charge"
    if "monoisotopic" in k:
        return None
    if "mass" in k:
        return "mass"
    return None


def _lines(path: Path) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        yield from fh


def read_release(path: Path) -> ChebiRelease:
    """Parse a ChEBI OBO file (gzipped if it ends in ``.gz``) as the module docstring states."""
    entries: dict[str, ChebiEntry] = {}
    header: dict[str, str] = {}
    stanza: _Stanza | None = None
    in_header = True

    def close(s: _Stanza | None) -> None:
        if s is not None and str(s.values.get("id", "")).startswith("CHEBI:"):
            entries[str(s.values["id"])] = s.entry()

    for raw in _lines(Path(path)):
        line = raw.rstrip("\n")
        if line.startswith("["):
            close(stanza)
            in_header = False
            stanza = _Stanza() if line.strip() == "[Term]" else None
            continue
        if stanza is None:
            if in_header and ":" in line:
                key, value = line.split(":", 1)
                header.setdefault(key.strip(), value.strip())
            continue
        if not line or line.startswith("!"):
            continue
        tag, _, value = line.partition(": ")
        _read_tag(stanza, tag, value, line)
    close(stanza)
    return ChebiRelease(entries, header.get(_RELEASE_KEY, ""))


def _read_tag(stanza: _Stanza, tag: str, value: str, line: str) -> None:  # noqa: PLR0912  (one branch per tag)
    v = stanza.values
    if tag == "id":
        v["id"] = value.strip()
    elif tag == "name":
        v["name"] = unescape(value.strip())
    elif tag == "def":
        m = _DEF.match(line)
        v["def"] = unescape(m.group(1)) if m else value
    elif tag == "synonym":
        m = _SYNONYM.match(value)
        if m:
            stanza.synonyms.append((unescape(m.group(1)), m.group(2)))
    elif tag == "is_a":
        stanza.is_a.append(value.split("!", maxsplit=1)[0].strip())
    elif tag == "relationship":
        body, _, comment = value.partition("!")
        parts = body.split()
        if len(parts) >= 2:  # noqa: PLR2004  (a relation and its target)
            rel = _relation(parts[0], comment)
            stanza.rels.append((rel, parts[1]))
            if rel == "has_role":
                stanza.has_role.append(parts[1])
    elif tag == "subset":
        m = _STARS.match(value.strip())
        if m:
            v["stars"] = int(m.group(1))
    elif tag == "is_obsolete":
        v["obsolete"] = value.strip() == "true"
    elif tag == "property_value":
        m = _PV.match(line)
        kind = _property_kind(m.group(1)) if m else None
        if m and kind and kind not in v:
            v[kind] = unescape(m.group(2))


@lru_cache(maxsize=4)
def _release(data_root: Path, path: str, sha256: str) -> ChebiRelease:
    return read_release(read_source(data_root, SourceFile(path=path, sha256=sha256)))


def load_release(source: SourceFile, data_root: Path) -> ChebiRelease:
    """Return the ChEBI release ``source`` cites, once its sha256 matches (read once per process).

    Raises:
        IntegrityError: if the file's sha256 differs from ``source.sha256``.
    """
    return _release(Path(data_root), source.path, source.sha256)
