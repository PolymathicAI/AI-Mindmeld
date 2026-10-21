"""Protein records for the relational answers: each training protein's facts, parsed from its screened text.

The parsers of the original protein relational generator for the
facts the original runs trained under their task split (family, location, function, pathway, EC, catalytic activity);
the answers built from them are in :mod:`data.relational_answers`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from latent_interconnect.domains.protein import PathwayEntry, parse_ec_set, parse_pathway_set

__all__ = ["EC_CLASS", "LOCATION_WORDS", "Family", "Location", "Protein", "protein_record"]

EC_CLASS = {
    "1": "oxidoreductase",
    "2": "transferase",
    "3": "hydrolase",
    "4": "lyase",
    "5": "isomerase",
    "6": "ligase",
    "7": "translocase",
}

_TOPOLOGY = (
    "single-pass",
    "multi-pass",
    "peripheral membrane",
    "lipid-anchor",
    "gpi-anchor",
    "side",
    "intermembrane",
    "matrix side",
    "lumenal side",
    "cytoplasmic side",
    "extracellular side",
    "beta stranded",
)
_LOC_WORDS = {
    "Secreted": "extracellular space (secreted)",
    "Secreted, extracellular space": "extracellular space (secreted)",
}
LOCATION_WORDS = ("extracellular space (secreted)", "(secreted)")
"""Location wording a record writes rather than copies (a secreted protein's compartment)."""
_LOC_NOTE = re.compile(
    r"\b(is|are|was|were|be|been|may|might|can|could|upon|during|found|when|via|by|to|also|only|in|on|at|of|the|a|an|"
    r"and|or|as|with|from|into|onto|between|near|around|along|within|inside|throughout|then|where|which|that|this|"
    r"these|it|its|not|no|response|associate[sd]?|localize[sd]?|locali[sz]ation|located|present|detected|mainly|"
    r"predominantly|primarily|mostly|partially|partly|exclusively|largely|probably|possibly|likely|weakly|strongly|"
    r"highly|active|shuttles?|translocates?|recruited|accumulates?|forms?|cycles?|colocali[sz]e[sd]?|co-locali[sz]e[sd]?|"
    r"colocates?|displaces?|remains?|resides?|functions?|target(?:s|ed|ing)|depends?|attached|binds?|interacts?|moves?|"
    r"reloca\w*|redistribut\w*|enriched|concentrated|diffuse|observed|seen|visible|appears?|occurs?|tethered|expressed|"
    r"released|transported|imported|exported|sequestered|retained|delivered|internali[sz]\w*|cleaved|processed|"
    r"following|after|before|while|distribution|pattern|staining|foci)\b",
    re.IGNORECASE,
)
"""A location sentence with any of these words is a free-text note, not a compartment (the original prompt_grammar)."""
_NOTE_MAX_CHARS = 80
_NOTE_MAX_WORDS = 6
_MIN_FUNCTION_CHARS = 12
_STEP = re.compile(r"step \d+/\d+")


@dataclass(frozen=True)
class Family:
    """A protein's written family.

    Attributes:
        family: the family phrase (the ``... family`` part, else the last part).
        superfamily: the superfamily part, if any.
    """

    family: str
    superfamily: str | None


@dataclass(frozen=True)
class Location:
    """A protein's cleaned subcellular location.

    Attributes:
        locs: the compartments, in order, free-text notes dropped.
        topo: the topology bits.
    """

    locs: tuple[str, ...]
    topo: tuple[str, ...]


@dataclass(frozen=True)
class Protein:
    """The facts of one training protein, parsed from its screened text.

    Attributes:
        object_id: the protein.
        family: the written family, if any.
        location: the cleaned location, if any.
        function: the first sentence of its function text (at least 12 characters), if any.
        pathway: its pathway as the original ``parse_pathway`` read the whole field, if any.
        pathways: every pathway entry (``domains.protein.parse_pathway_set``).
        ec: its first EC number, if it has four levels and a known class.
        ecs: every EC number (``domains.protein.parse_ec_set``).
        catalytic: the first reaction of its catalytic activity, if any.
    """

    object_id: str
    family: Family | None
    location: Location | None
    function: str | None
    pathway: PathwayEntry | None
    pathways: tuple[PathwayEntry, ...]
    ec: tuple[str, ...] | None
    ecs: tuple[tuple[str, ...], ...]
    catalytic: str | None


def _text(value: str | None) -> str | None:
    return value.strip() if value and value.strip() else None


def _family(text: str | None) -> Family | None:
    pf = _text(text)
    if not pf:
        return None
    parts = [p.strip() for p in pf.split(",")]
    fam = next((p for p in parts if p.lower().endswith(" family")), parts[-1])
    sup = next((p for p in parts if "superfamily" in p.lower()), None)
    return Family(fam, sup)


def _location_note(sent: str) -> bool:
    s = sent.strip()
    if not s or len(s) > _NOTE_MAX_CHARS or _LOC_NOTE.search(s):
        return True
    head = re.sub(r"^Part of (the )?", "", s.split(";")[0].strip(), flags=re.IGNORECASE)
    return "(" in head or any(len(p.split()) > _NOTE_MAX_WORDS for p in head.split(","))


def _location(text: str | None) -> Location | None:
    loc = _text(text)
    if not loc:
        return None
    locs: list[str] = []
    topo: list[str] = []
    for raw in re.split(r"\.\s+|\.$", loc):
        sent = re.sub(r"^(\[[^\]]+\]:\s*)+", "", raw.strip())
        if _location_note(sent):
            continue
        bits = [b.strip() for b in sent.split(";") if b.strip()]
        if not bits:
            continue
        if bits[0].lower().startswith(_TOPOLOGY):
            topo += bits
            continue
        t = re.sub(r"^Part of (the )?", "", bits[0].rstrip("."))
        if t.lower().startswith("secreted"):
            rest = t.split(",", 1)[1].strip() if "," in t else ""
            t = (rest or "extracellular space") + " (secreted)"
        locs.append(_LOC_WORDS.get(t, t))
        topo += bits[1:]
    kept = tuple(dict.fromkeys(locs))
    return Location(kept, tuple(dict.fromkeys(topo))) if kept else None


def _function(text: str | None) -> str | None:
    t = _text(text)
    if not t:
        return None
    s = re.split(r"(?<=[a-z0-9\)])\.\s+(?=[A-Z])", t)[0].rstrip(".")
    return s if len(s) >= _MIN_FUNCTION_CHARS else None


def _pathway(text: str | None) -> PathwayEntry | None:
    """The original ``parse_pathway``: the whole field as one entry (it splits on ``". "``, which never occurs)."""
    pw = _text(text)
    if not pw:
        return None
    parts = [p.strip().rstrip(".") for p in pw.split(". ")[0].rstrip(".").split(";")]
    step = next((p for p in parts if _STEP.search(p)), None)
    found = _STEP.search(step) if step else None
    return PathwayEntry(parts[0], parts[1] if len(parts) > 1 else parts[0], found.group(0) if found else None)


def _ec(text: str | None) -> tuple[str, ...] | None:
    ec = _text(text)
    if not ec:
        return None
    levels = ec.split(";")[0].strip().split(".")
    return tuple(levels) if len(levels) == 4 and levels[0] in EC_CLASS else None  # noqa: PLR2004  (four EC levels)


def protein_record(object_id: str, fields: Mapping[str, str], names: Mapping[str, str]) -> Protein:
    """Parse one protein's screened text.

    ``names`` maps each fact the stream reads to its training-text field; a fact not in it is absent.
    """

    def get(key: str) -> str | None:
        return fields.get(names[key]) if key in names else None

    catalytic = _text(get("catalytic_activity"))
    return Protein(
        object_id=object_id,
        family=_family(get("family")),
        location=_location(get("location")),
        function=_function(get("function")),
        pathway=_pathway(get("pathway")),
        pathways=tuple(parse_pathway_set(get("pathway"))),
        ec=_ec(get("ec")),
        ecs=tuple(parse_ec_set(get("ec"))),
        catalytic=catalytic.split("; ")[0] if catalytic else None,
    )
