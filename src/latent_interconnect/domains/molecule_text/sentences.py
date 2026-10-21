"""Sentence-level rules of the molecule text.

Splitting, the connective after a removed sentence, the own-name rewrite (with its organism guard) and the
sentence-boundary trim to a token cap.

The port of the original ``split_sentences``, ``strip_connective``, ``own_name_rewrite`` (``organism_guard=True``)
and of ``trim`` and ``trim_describe``, kept
exactly but for one extension: the entity suffix ``zwitterion`` of the own-name rule
covers every ``-zwitterion`` form (``dizwitterion``), which the paper's list missed; the paper's rows are unchanged by
it (the reproduction gate). The trim counts tokens with the receiver's renderer, the one user of the tokenizer
(``receivers.render.Renderer.target_token_count`` and ``Renderer.target_prefix``).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Protocol

__all__ = [
    "OWN_NAME",
    "TokenCounter",
    "own_name_rewrite",
    "split_sentences",
    "strip_connective",
    "trim",
    "trim_roles_first",
]

OWN_NAME = "this compound"
"""What a mention of the compound's own name becomes (``"This compound"`` at a sentence or line start)."""

_ABBREVIATIONS = frozenset(
    {
        "sp.",
        "spp.",
        "subsp.",
        "var.",
        "fig.",
        "figs.",
        "e.g.",
        "i.e.",
        "cf.",
        "al.",
        "approx.",
        "no.",
        "vs.",
        "ca.",
        "st.",
        "dr.",
        "etc.",
        "pv.",
        "str.",
    }
)
_SENTENCE_START = re.compile(r"(?:It|This|These|The|A|An|In|Its|They)\b")
_CONNECTIVE = re.compile(
    r"^(?:Therefore|Thus|Hence|Consequently|Accordingly|Also|In addition|Furthermore),?\s+|"
    r"(?<=^It) also\b|(?<= is) therefore\b|(?<= are) therefore\b"
)
_MIN_NAME = 3
# a mention followed by one of these is part of another entity's name ('X free base', 'X acid', 'X(1-)'): kept
_ENTITY_NEXT = re.compile(
    r"(?:\(|\[|\+| (?:acid|acids|base|bases|salt|salts|ester|esters|(?:di|tri|tetra|penta)?(?:anion|cation)s?|"
    r"[a-z]*zwitterions?|radicals?|ion|ions|form|forms|"
    r"moiety|moieties|residue|residues|group|groups|unit|units|core|derivative|derivatives|analogue|analogues|analog|"
    r"free|aglycone|hydrate|dihydrate|monohydrate|sodium|potassium|calcium|magnesium|zinc|"
    r"[a-z]+(?:ate|ide|yl|il|ine|ase)|[IVX]+|[A-Z])\b)"
)
# ... or preceded by one of these ('hydrogen cyanide', 'ferric X', 'methyl X', 'poly X', '(R)- and X')
_ENTITY_PREV = re.compile(
    r"(?:(?:^|[\s(-])(?:hydrogen|poly|di|tri|ferric|ferrous|cupric|cuprous|[A-Za-z]+(?:ium|yl))|-\s(?:and|or)|-)\s$"
)
# a mention inside an organism or strain designation ('Nocardiopsis sp K-252a') names the organism: kept
_ORGANISM_PREV = re.compile(
    r"(?:(?<![\w-])(?:sp|spp|subsp|ssp|str|strain|strains|isolate|var|pv|cv)\.?|(?<=[a-z,;] )[A-Z][a-z]{2,})\s$"
)
_CLASS_NOUN_PREV = re.compile(r"(?:(?:^|[\s(])(?:a|an|A|An)(?:\s[\w-]+)?|\b[Ss]implest)\s$")
_ARTICLE_PREV = re.compile(r"(?:^|(?<=[\s(]))(?:the|The)\s$")
_AT_START = re.compile(r"(?:^|[.!?]\s|\n)\s*$")
_BOUNDARY = re.compile(r"\.(?= |\n)")
_INITIAL = re.compile(r"\(?[a-z]\.")


def split_sentences(text: str) -> list[str]:
    """Return the sentences of ``text``; a break after an abbreviation is undone unless a sentence opener follows."""
    out: list[str] = []
    for part in (s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(0-9])", text) if s):
        if out and out[-1].split()[-1].lower() in _ABBREVIATIONS and not _SENTENCE_START.match(part):
            out[-1] += " " + part
        else:
            out.append(part)
    return out


def strip_connective(sentence: str) -> str:
    """Return the sentence without the connective that tied it to a removed sentence (first letter upper-cased)."""
    out = _CONNECTIVE.sub("", sentence, count=1)
    return out[:1].upper() + out[1:] if out != sentence else sentence


def own_name_rewrite(text: str, name: str) -> tuple[str, Counter[str]]:
    """Return ``text`` with each mention of the compound's own ``name`` rewritten to :data:`OWN_NAME`, and the counts.

    A mention is the name whole (no word character or hyphen on either side), case-insensitive, for names of at least
    three characters; a ``the`` before it is absorbed. Kept, and counted by kind: a mention that is part of another
    entity's name (``kept_entity_name``), one inside an organism or strain designation (``kept_organism``), and a
    class-noun use after ``a`` / ``an`` (one word between allowed) or ``simplest`` (``kept_class_noun``). Rewritten
    mentions count as ``rewritten``.
    """
    counts: Counter[str] = Counter()
    if not name or len(name) < _MIN_NAME:
        return text, counts
    mention = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])", re.I)
    out, pos = [], 0
    for m in mention.finditer(text):
        start, end = m.start(), m.end()
        before, after = text[:start], text[end:]
        if _ENTITY_NEXT.match(after) or (before.endswith("(") and after.startswith(")")) or _ENTITY_PREV.search(before):
            counts["kept_entity_name"] += 1
            continue
        if _ORGANISM_PREV.search(before):
            counts["kept_organism"] += 1
            continue
        if _CLASS_NOUN_PREV.search(before):
            counts["kept_class_noun"] += 1
            continue
        article = _ARTICLE_PREV.search(before)
        if article:
            start -= len(article.group(0))
        at_start = _AT_START.search(text[:start]) is not None
        out += [text[pos:start], OWN_NAME[0].upper() + OWN_NAME[1:] if at_start else OWN_NAME]
        pos = end
        counts["rewritten"] += 1
    out.append(text[pos:])
    return "".join(out), counts


class TokenCounter(Protocol):
    """The part of the renderer the trim uses (``receivers.render.Renderer``)."""

    def target_token_count(self, text: str) -> int:
        """Return how many tokens ``text`` takes as a target part."""
        ...

    def target_prefix(self, text: str, tokens: int) -> str:
        """Return the text of the first ``tokens`` tokens of ``text`` as a target part."""
        ...


def trim(text: str | None, counter: TokenCounter, cap: int) -> str | None:
    """Return ``text`` cut at the last sentence end within its first ``cap`` tokens.

    A text within the cap is whole. A sentence end is a ``.`` before a space or a line break, not after an
    abbreviation or a one-letter initial (``E.``). Without one, the text stays whole (the trainer's cap cuts it).
    """
    if not text or counter.target_token_count(text) <= cap:
        return text
    head = counter.target_prefix(text, cap)
    cut = -1
    for m in _BOUNDARY.finditer(head):
        words = head[: m.end()].split()
        word = words[-1].lower() if words else ""
        if word in _ABBREVIATIONS or _INITIAL.fullmatch(word):
            continue
        cut = m.end()
    if cut <= 0 or not text.startswith(head[:cut]):
        return text
    return head[:cut]


def trim_roles_first(text: str | None, roles: str | None, counter: TokenCounter, cap: int) -> str | None:
    """Return :func:`trim` of a roles-first describe text.

    The text stays whole when the trim would leave its roles line alone (the field table's fallback: the trainer's cap
    then cuts it).
    """
    cut = trim(text, counter, cap)
    if roles and cut is not None and cut.strip() == roles.strip() and cut != text:
        return text
    return cut
