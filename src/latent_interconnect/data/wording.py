"""Answer formats and target-wording detectors: rule-based rewrites of a screened field's text.

A port of the original research code's ``transform`` and its helpers and of its target-wording detectors, with the
original's target-wording transform fixes on (its ``transform_fixes`` setting). A transform returns the answer and the
character spans of its announced-format text (template text: weighted at the floor, the positional curve counting
only the other tokens), or None when it does not apply to the text. No LLM rewording, no tokenizer.
"""

from __future__ import annotations

import re

__all__ = [
    "TRANSFORMS",
    "family_generic",
    "names_partners",
    "sentences",
    "transform",
]

_SENT = re.compile(r"(?<=[A-Za-z0-9\)\]])\.\s+(?=[A-Z(])")
"""The sentence boundary with the target-wording fix (also after a capital, "host FGA. Interacts")."""
_FAM_LEVEL = re.compile(
    r"\b(sub-subfamily|subfamily|superfamily|family|subclass|class|clan|subgroup|group|order|subtype|type)$",
    re.IGNORECASE,
)
_LEAD_VERB: dict[str, str | None] = {
    "Involved": "is involved",
    "Required": "is required",
    "Component": "is a component",
    "Part": "is part",
    "Essential": "is essential",
    "Necessary": "is necessary",
    "Responsible": "is responsible",
    "Important": "is important",
    "Probably": "probably",
    "Possibly": "possibly",
    "May": "may",
    "Might": "might",
    "Could": "could",
    "Plays": "plays",
    "Acts": "acts",
    "Functions": "functions",
    "Binds": "binds",
    "Catalyzes": "catalyzes",
    "Mediates": "mediates",
    "Regulates": "regulates",
    "Promotes": "promotes",
    "Inhibits": "inhibits",
    "Participates": "participates",
    "Transports": "transports",
    "Negatively": "negatively",
    "Positively": "positively",
    "Specifically": "specifically",
    "Forms": "forms",
    "Controls": "controls",
    "Contributes": "contributes",
    "Has": "has",
    "Is": "is",
    "Transcription": None,
}
_LOC_NOTE = re.compile(
    r"\b(is|are|was|were|be|been|may|might|can|could|upon|during|found|when|via|by|to|also|only|in|on|at|of|the|a|an|"
    r"and|or|as|with|from|into|onto|between|near|around|along|within|inside|throughout|then|where|which|that|this|"
    r"these|it|its|not|no|response|associate[sd]?|localize[sd]?|locali[sz]ation|located|present|detected|mainly|"
    r"predominantly|primarily|mostly|partially|partly|exclusively|largely|probably|possibly|likely|weakly|strongly|"
    r"highly|active|shuttles?|translocates?|recruited|accumulates?|forms?|cycles?|colocali[sz]e[sd]?|"
    r"co-locali[sz]e[sd]?|"
    r"colocates?|displaces?|remains?|resides?|functions?|target(?:s|ed|ing)|depends?|attached|binds?|interacts?|moves?|"
    r"reloca\w*|redistribut\w*|enriched|concentrated|diffuse|observed|seen|visible|appears?|occurs?|tethered|expressed|"
    r"released|transported|imported|exported|sequestered|retained|delivered|internali[sz]\w*|cleaved|processed|"
    r"following|after|before|while|distribution|pattern|staining|foci)\b",
    re.IGNORECASE,
)
_STEP = re.compile(r"step \d+/\d+")
_MIN_SENTENCE = 12
_MAX_BULLETS = 5
_LOC_NOTE_CHARS = 60
_NOTE_WORDS = 6

TRANSFORMS = (
    "none",
    "verbatim",
    "one_sentence",
    "two_sentences",
    "this_protein",
    "bullets",
    "family_name",
    "loc_list",
    "loc_sentence",
    "reaction_sentence",
    "ec_sentence",
    "pathway_sentence",
)

Formatted = tuple[str, list[tuple[int, int]]]
"""An answer and the character spans of its announced-format text."""


def sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, each ending with one period (the target-wording boundary)."""
    return [s.strip().rstrip(".") + "." for s in _SENT.split(text.strip()) if s.strip()]


def _pieces(parts: list[tuple[str, bool]]) -> Formatted:
    """``[(text, is_template)]`` → the answer and its template spans without surrounding whitespace."""
    answer, spans = "", []
    for txt, template in parts:
        if template and txt.strip():
            a = len(answer) + len(txt) - len(txt.lstrip())
            spans.append((a, a + len(txt.strip())))
        answer += txt
    return answer, spans


def _location_note(sent: str, max_len: int) -> bool:
    s = sent.strip()
    if not s or len(s) > max_len or _LOC_NOTE.search(s):
        return True
    head = re.sub(r"^Part of (the )?", "", s.split(";")[0].strip(), flags=re.IGNORECASE)
    return "(" in head or any(len(p.split()) > _NOTE_WORDS for p in head.split(","))


def _family_sentence(text: str) -> Formatted | None:
    """Target-wording fix: 'Belongs to the A superfamily. B family.' → 'It belongs to the A superfamily (B family).'."""
    s = sentences(text)
    m = re.match(r"^Belongs to the (.+)\.$", s[0]) if s else None
    rest = [x[:-1].strip() for x in s[1:]]
    if not m or any(not _FAM_LEVEL.search(x) for x in rest):
        return None
    return "It belongs to the " + m.group(1) + (f" ({', '.join(rest)})" if rest else "") + ".", [(0, 2)]


def _low(x: str) -> str:
    return x[0].lower() + x[1:] if len(x) > 1 and x[0].isupper() and x[1].islower() else x


def _pathway_sentence(text: str) -> Formatted | None:
    """Target-wording fix: a single-entry PATHWAY field as one sentence; None for multi-entry fields, other shapes."""
    entries = [e.strip() for e in re.split(r"\.;\s*", text.strip().rstrip("."))]
    if len(entries) != 1:
        return None
    parts = [p.strip() for p in entries[0].split(";")]
    if not all(parts) or len(parts) > 3:  # noqa: PLR2004  (category, pathway, sub-pathway)
        return None
    step = None
    if len(parts) == 3:  # noqa: PLR2004
        m = re.match(r"^(.*\S):\s*step (\d+)/(\d+)$", parts[2])
        if m:
            parts[2], step = m.group(1), (m.group(2), m.group(3))
    elif _STEP.search(parts[-1]):
        return None
    if any(_STEP.search(p) for p in parts):
        return None
    out: list[tuple[str, bool]] = [("It is involved in", True)]
    if len(parts) == 3:  # noqa: PLR2004
        if step:
            out += [(" step", True), (" " + step[0], False), (" of", True), (" " + step[1], False), (" of", True)]
        sub = parts[2]
        if " from " in sub:
            product, source = sub.split(" from ", 1)
            out += [(" the sub-pathway that produces", True), (" " + product, False), (" from", True)]
            out += [(" " + source, False)]
        else:
            out += [(" the sub-pathway", True), (" " + sub, False)]
        out += [(", part of", True), (" " + _low(parts[1]), False), (" (", True), (_low(parts[0]), False)]
        out += [(").", True)]
    elif len(parts) == 2:  # noqa: PLR2004
        out += [(" " + _low(parts[1]), False), (" (", True), (_low(parts[0]), False), (").", True)]
    else:
        out += [(" " + _low(parts[0]), False), (".", True)]
    return _pieces(out)


def _locations(text: str) -> list[str] | None:
    locs = []
    for s in sentences(text):
        head_text = re.sub(r"^(\[[^\]]+\]:\s*)+", "", s).rstrip(".")
        if not head_text or _location_note(head_text, _LOC_NOTE_CHARS):
            return None
        head = head_text.split(";")[0].strip()
        if not head:
            return None
        locs.append(head[0].lower() + head[1:] if not head[:2].isupper() else head)
    unique = list(dict.fromkeys(locs))
    return unique or None


def transform(name: str, text: str | None, task: str | None = None) -> Formatted | None:  # noqa: PLR0911, PLR0912
    """Apply the answer format ``name`` to a field's text (with the module's target-wording rules).

    Raises:
        ValueError: for an unknown format.
    """
    if not text or not text.strip():
        return None
    t = text.strip()
    if name == "verbatim" and task == "family":
        return _family_sentence(t)
    if name in ("none", "verbatim"):
        return t, []
    if name == "pathway_sentence":
        return _pathway_sentence(t)
    if name in ("one_sentence", "two_sentences"):
        s = sentences(t)
        if not (s and len(s[0]) >= _MIN_SENTENCE):
            return None
        a = " ".join(s[: 1 if name == "one_sentence" else 2])
        return None if task == "pathway" and ";" in a else (a, [])
    if name == "this_protein":
        w = t.split(" ", 1)
        verb = _LEAD_VERB.get(w[0]) if len(w) > 1 else None
        return ("This protein " + verb + " " + w[1], [(0, len("This protein"))]) if verb else None
    if name == "bullets":
        s = sentences(t)
        if len(s) < 2:  # noqa: PLR2004
            return None
        lines = ["- " + x for x in s[:_MAX_BULLETS]]
        spans, p = [], 0
        for line in lines:
            spans.append((p, p + 1))
            p += len(line) + 1
        return "\n".join(lines), spans
    if name == "family_name":
        s = sentences(t)
        m = re.search(r"Belongs to the (.+?)\.?$", s[0] if s else t)
        return (m.group(1)[0].upper() + m.group(1)[1:], []) if m else None
    if name in ("loc_list", "loc_sentence"):
        locs = _locations(t)
        if locs is None:
            return None
        if name == "loc_list":
            return ", ".join(locs), []
        joined = locs[0] if len(locs) == 1 else ", ".join(locs[:-1]) + " and " + locs[-1]
        return f"It is found in: {joined}.", [(0, len("It is found in:"))]
    if name == "reaction_sentence":
        return None if "=" not in t else ("It catalyzes the reaction: " + t.rstrip("."), [(0, 26)])
    if name == "ec_sentence":
        return f"Its EC number is {t.split(';')[0].strip()}.", [(0, len("Its EC number is"))]
    raise ValueError(f"unknown answer format {name!r}; known: {TRANSFORMS}")


# ------------------------------------------------------------------------------------------ target-wording detectors

_FIRST = re.compile(r"(?<=[a-z0-9\)\]A-Z])\.\s+(?=[A-Z(\[])")
_OLIGO = re.compile(
    r"^([\w\-]+ ){0,3}?((Homo|Hetero)?(di|tri|tetra|penta|hexa|hepta|octa|nona|deca|dodeca|oligo|multi|"
    r"poly)mer(ic)?s?|Monomers?)\b|^(Probabl[ey] |Likely |Can |May )?(Oligomeri[sz]es|(Homo|Hetero)(di|tri|"
    r"oligo|multi)?"
    r"meri[sz]es|Self-associates|Forms? (a |an )?(homo|hetero)?(di|tri|tetra|hexa|oligo|multi|poly)mer)",
    re.IGNORECASE,
)
_PARTNER = re.compile(
    r"^(It |Probabl[ey] |Possibly |May |Likely |Directly |Specifically |Also |Weakly |Strongly |Constitutively |Can )?"
    r"(Interacts?|Binds?|Associates?|associated|Heterodimeri[sz]es|Found in a complex|Identified in|Does not interact|"
    r"forms?|Forms?)\b",
    re.IGNORECASE,
)
_PARTNER_ANY = re.compile(
    r"\b(Interacts?|Binds?|Associates?|with|Component of|Part of|Hetero\w*|complex)\b", re.IGNORECASE
)
_MISSING_SUBJECT = re.compile(
    r"^(Probably |Possibly |Likely |Also |Specifically |Negatively |Positively )?(Catalyzes|Binds|Involved|"
    r"Required|Plays|Acts|"
    r"Functions|Mediates|Regulates|Promotes|Inhibits|Participates|Transports|Controls|Contributes|Forms|Has|Is|"
    r"May|Might|"
    r"Could|Essential|Necessary|Responsible|Important|Component|Part|Interacts|Associates|Hydrolyzes|Converts|"
    r"Transfers|"
    r"Cleaves|Phosphorylates|Degrades|Represses|Activates|Induces|Protects|Confers|Seems|Appears|Needed)\b"
)
_PLURAL_SUBJ = re.compile(r"^(The )?[A-Z][\w\-()]*s( and [\w\-]+s)? (are|have|act|block|inhibit|bind|consist|form)\b")
_COMPONENT = re.compile(
    r"^(Probabl[ey] |Putative |Probable |May be (a )?)?(\w+ )?(\w+ )?(component|part|subunit|member|"
    r"constituent) (of|in)\b",
    re.IGNORECASE,
)
_GENERIC_SUBJ = re.compile(
    r"^(The |A |An |In )?[^.;]{0,80}?\b(complex(es)?|system|machinery|family|superfamily|channels|toxins|"
    r"conotoxins|ATPases|"
    r"core|holoenzyme|polymerase|synthase|reductase|oxidase|particle|reaction center|nucleosome|connexon|transporter|"
    r"proteins|enzymes|members|NDH-1|F\(1\)F\(0\)|CRISPR|units?|cluster)\b[^.;]{0,40}?\b(is|are|has|have|consists?|"
    r"comprises?|contains?|catalyzes?|produces?|shuttles?|acts?|blocks?|mediates?|forms?|composed|characterized|may)\b"
)
_CATALOG_PREFIX = re.compile(r"^(\[[^\]]+\]:|\(Microbial infection\))")
_FUNC_MISMATCH = re.compile(r"^(Does not|Not |The protein is truncated|Belongs to|Expressed|Lacks|No )")
_FUNC_OWN_SUBJ = re.compile(r"^[A-Z][\w\-/,() ]*?(protein|subunit|kinase|factor|enzyme|receptor|ase)\b")


def _first(text: str) -> str:
    return _FIRST.split(re.sub(r"^- ", "", (text or "").strip()), maxsplit=1)[0]


def family_generic(task: str, text: str | None) -> bool:  # noqa: PLR0911  (the rubric, in its order)
    """The family-generic detector: the first sentence is about a family, class, complex or system."""
    if not text or not text.strip():
        return False
    fs = _first(text.strip())
    if _CATALOG_PREFIX.match(fs):
        return False
    if task == "subunit":
        specific = _OLIGO.search(fs) or _PARTNER.search(fs) or _COMPONENT.search(fs)
        return not specific and bool(_GENERIC_SUBJ.search(fs))
    if task in ("function", "describe"):
        if _FUNC_MISMATCH.match(fs):
            return False
        if _PLURAL_SUBJ.search(fs):
            return True
        own = re.match(r"^(This protein|It)\b", fs) or _MISSING_SUBJECT.search(fs) or _FUNC_OWN_SUBJ.match(fs)
        return bool(_GENERIC_SUBJ.search(fs) and not own)
    return False


def names_partners(text: str | None) -> bool:
    """Whether a text names binding partners or a complex (the partner routing)."""
    return bool(_PARTNER_ANY.search(text or ""))
