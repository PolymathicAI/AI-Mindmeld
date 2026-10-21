"""The domain-agnostic pieces of the relational stream (``data.relational_stream``).

- **Marked answers.** An answer is fixed template text with its facts marked while it is built (:func:`fill`, one
  fact of a kind; :func:`group`, one object's fact span); :func:`clean` turns it into its text and the character
  spans, fact kinds and fact groups of its fills (the loss weights read them). This is the original
  ``clean_full`` of both relational generators (protein and molecule, where it is the same function).
- **Weighted draws** (:class:`WeightedPool`): draws within a list of objects, weighted by each object's draw
  weight, as the original ``Pool.pick`` / ``pick2`` / ``pickn`` of both generators.
- **A domain's answers** (:class:`RelationalAnswers`, in :data:`RELATIONAL_ANSWERS`): its records, its answer
  templates per family, its verdicts, its fixed wording, its pool, and the checks of its records against the domain.
  The data package registers them (``data.relational_protein.PROTEIN`` as ``protein``,
  ``data.relational_answers_molecule.MOLECULE`` as ``molecule``); the stream config names one
  (``RelationalParams.answers``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from latent_interconnect.domains.views import DomainView
from latent_interconnect.plugins import Registry

__all__ = [
    "RELATIONAL_ANSWERS",
    "Answer",
    "Clean",
    "Pool",
    "RelationalAnswers",
    "RelationalRecord",
    "TripleAnswer",
    "WeightedPool",
    "clean",
    "fill",
    "group",
    "plain",
    "sentences",
]

_FO, _FC, _KS, _GO, _GC = "\x01", "\x02", "\x03", "\x04", "\x05"


# ---------------------------------------------------------------------------------------------------- marked answers


def fill(text: object, kind: str) -> str:
    """Mark ``text`` as an object's fact of ``kind``."""
    return f"{_FO}{kind}{_KS}{text}{_FC}"


def group(text: str) -> str:
    """Mark one object's fact span: the fills inside share one fact group."""
    return f"{_GO}{text}{_GC}"


@dataclass(frozen=True)
class Clean:
    """A marked answer, cleaned.

    Attributes:
        text: the answer.
        spans: each fill's ``[start, end)``.
        kinds: each fill's fact kind.
        groups: each fill's fact group.
    """

    text: str
    spans: tuple[tuple[int, int], ...]
    kinds: tuple[str, ...]
    groups: tuple[int, ...]


def clean(marked: str) -> Clean:
    """Strip the markers; return the text and the spans, kinds and groups of its fills (the original ``clean_full``)."""
    out: list[str] = []
    spans: list[tuple[int, int]] = []
    kinds: list[str] = []
    groups: list[int] = []
    depth, start, kind, gdepth, gid, n_groups = 0, 0, "", 0, 0, 0
    i = 0
    while i < len(marked):
        ch = marked[i]
        if ch == _FO:
            j = marked.index(_KS, i)
            if depth == 0:
                start, kind = len(out), marked[i + 1 : j]
            depth += 1
            i = j + 1
            continue
        if ch == _FC:
            depth -= 1
            if depth == 0 and len(out) > start:
                spans.append((start, len(out)))
                kinds.append(kind)
                if gdepth > 0:
                    groups.append(gid)
                else:
                    groups.append(n_groups)
                    n_groups += 1
        elif ch == _GO:
            if gdepth == 0:
                gid, n_groups = n_groups, n_groups + 1
            gdepth += 1
        elif ch == _GC:
            gdepth -= 1
        else:
            out.append(ch)
        i += 1
    return Clean("".join(out), tuple(spans), tuple(kinds), tuple(groups))


def plain(marked: str) -> str:
    """The answer text without markers."""
    return clean(marked).text


def sentences(items: Sequence[str]) -> str:
    """Each item as a sentence: its first letter upper-cased, a period after it (the original ``_sentences``)."""
    return " ".join(x[0].upper() + x[1:] + "." for x in items)


# ---------------------------------------------------------------------------------------------------- weighted draws


class WeightedPool:
    """Weighted draws within lists of objects (the original relational ``Pool``'s draws).

    A draw within a list is weighted by each object's weight (``cluster_sqrt``: 1/sqrt(its cluster's size among the
    training objects), originally ``set_weights``). A list the pool owns caches its cumulative weights; a list built
    for one draw is weighted on the spot.

    Args:
        weight: object id → its draw weight.
    """

    def __init__(self, weight: Mapping[str, float]) -> None:
        self.weight = weight
        self._cum: dict[int, tuple[list[str], np.ndarray]] = {}

    def _cumulative(self, among: list[str], fixed: bool) -> np.ndarray:
        if not fixed:
            return np.cumsum([self.weight[a] for a in among])
        cached = self._cum.get(id(among))
        if cached is None or cached[0] is not among:
            cached = self._cum[id(among)] = (among, np.cumsum([self.weight[a] for a in among]))
        return cached[1]

    def _index(self, g: np.random.Generator, cum: np.ndarray) -> int:
        """The original ``weighted_index``: one uniform draw."""
        return int(min(np.searchsorted(cum, g.random() * cum[-1], side="right"), len(cum) - 1))

    def pick(self, g: np.random.Generator, among: list[str], fixed: bool = True) -> str:
        """One object of ``among``, weighted (``fixed``: a list the pool owns, whose weights are cached)."""
        return among[self._index(g, self._cumulative(among, fixed))]

    def pick2(self, g: np.random.Generator, among: list[str], fixed: bool = True) -> tuple[str, str]:
        """Two distinct objects of ``among`` (the second redrawn until it differs)."""
        cum = self._cumulative(among, fixed)
        i = self._index(g, cum)
        j = i
        while j == i:
            j = self._index(g, cum)
        return among[i], among[j]

    def pickn(self, g: np.random.Generator, among: list[str], n: int) -> list[str]:
        """``n`` weighted draws with replacement from a list the pool owns (first-fit scans); none from no list."""
        if not among:
            return []
        cum = self._cumulative(among, True)
        u = g.random(n) * cum[-1]
        return [among[int(i)] for i in np.minimum(np.searchsorted(cum, u, side="right"), len(cum) - 1)]


# ---------------------------------------------------------------------------------------------- a domain's answers

Answer = Callable[[Sequence[Any], Sequence[str], bool], str | None]
"""(records, allowed facts, compact) → a marked answer, or None when the records give none."""
TripleAnswer = Callable[[Sequence[Any], Sequence[str], bool, int], str | None]
"""(records, allowed facts, compact, the odd one's position) → a marked answer, or None."""
Record = Callable[[str, Mapping[str, str], Mapping[str, str], Mapping[str, str]], Any]
"""(object id, its training-text fields, fact → field, list fact → separator) → the object's record."""


class RelationalRecord(Protocol):
    """A training object's record: its facts, parsed from its screened text."""

    @property
    def object_id(self) -> str:
        """The object."""
        ...


class Pool(Protocol):
    """A domain's pool of training-object records, indexed for its pair and triplet draws."""

    acc: list[str]

    @property
    def P(self) -> Mapping[str, RelationalRecord]:  # noqa: N802  (the original pool's name)
        """Object id → its record."""
        ...

    def pick(self, g: np.random.Generator, among: list[str], fixed: bool = True) -> str:
        """One object of ``among``, weighted."""
        ...

    def pair(self, g: np.random.Generator, family: str) -> tuple[str, str]:
        """Two objects for a pair family (related pairs, hard negatives and random ones)."""
        ...

    def triplet(self, g: np.random.Generator, family: str) -> list[str] | None:
        """``[a, b, odd]`` for a triplet family, or None."""
        ...


@dataclass(frozen=True)
class RelationalAnswers:
    """One domain's relational answers: everything the relational stream needs that is not domain-agnostic.

    Attributes:
        noun: the noun its fixed wording uses for an object; it must be the domain's noun.
        facts: the facts a family may state (``RelationalParams.families`` cites them).
        requires: fact → the further facts its answers read with it (e.g. the class graph, for a class's parents).
        lists: the facts whose training-text field is an item list (their separator is config).
        fill_kinds: the fact kinds of the fills (``RelationalParams.fact_curves`` gives each a curve).
        single: the one-object families' answers.
        pair: the pair families' answers.
        triple: the triplet families' answers.
        verdict_words: family → label → its verdict word (evidence first, a one-word verdict last).
        verdict_sentence: family → its verdict sentence, with ``{w}`` for the word.
        wording: the answers' fixed wording (``Supervision.texts``, screened as the stream's wording).
        record: builds an object's record from its screened training text.
        pool: builds the pool from the records, the draw weights and the hard-negative share.
        tables: fact → the domain's extra tables its records are checked against (``Supervision.tables``).
        verify: checks the records against those tables when the stream is built, raising ``ValueError`` on data that
            breaks the answers' contract; None for no check.
        yes_no: the families whose answer opens with a Yes/No verdict (``fill("Yes" | "No", "choice")``), asked as a
            direct Yes/No question under an answer convention (``RelationalParams.answer``).
    """

    noun: str
    facts: tuple[str, ...]
    requires: Mapping[str, tuple[str, ...]]
    lists: frozenset[str]
    fill_kinds: tuple[str, ...]
    single: Mapping[str, Answer]
    pair: Mapping[str, Answer]
    triple: Mapping[str, TripleAnswer]
    verdict_words: Mapping[str, Mapping[str, str]]
    verdict_sentence: Mapping[str, str]
    wording: tuple[str, ...]
    record: Record
    pool: Callable[[Mapping[str, Any], Mapping[str, float], float], Pool]
    tables: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    verify: Callable[[Mapping[str, Any], DomainView], None] | None = None
    yes_no: frozenset[str] = frozenset()

    def families(self) -> frozenset[str]:
        """Return every family it answers."""
        return frozenset(self.single) | frozenset(self.pair) | frozenset(self.triple)


RELATIONAL_ANSWERS: Registry[RelationalAnswers] = Registry("relational answers")
"""Each domain's relational answers, by name (``RelationalParams.answers``)."""
