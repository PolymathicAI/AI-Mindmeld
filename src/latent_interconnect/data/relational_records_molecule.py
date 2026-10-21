"""Compound records for the molecule relational answers, parsed from the screened text, and the pool drawing them.

The facts of the original molecule relational generator (the one the paper's molecule runs used with all
relational families) for the fields those runs
trained: a compound's direct ChEBI classes, its definition, its functional parents and its trained mechanism roles
(the generator states no conjugate, so none is ported). The original read them from the ChEBI entry table and the
ontology file. Here every fact, the class graph included, is a screened training-text field, so held-out content is
kept out by the zero-shot screen alone:

- ``classes``, ``functional_parent``, ``roles``: item lists, read in field order;
- ``definition``: free text, of which the first sentence is used (the original ``first_sentence``);
- ``class_graph``: an item list of the ``is_a`` edges above the compound's direct classes, each
  ``"<class> > <parent>"`` (:data:`EDGE`). The edges give the parents of each direct class and every class above them,
  as the ontology did, and every class name an answer can state is the compound's own screened text;
- ``parent_classes``: an item list of ``"<functional parent> > <class>"`` edges: the direct classes of a functional
  parent. **Contract:** only for a parent that is in the train split or is no molecule of the domain; a parent outside
  the train split (validation, test, or no split) is a held-out compound, whose classes the original never stated and
  the stream refuses (:func:`verify_parents`, against the domain's :data:`HELD_OUT_PARENTS` table).

Every name is read through ``domains.chebi``: written as its ``chebi_name`` (OBO escapes removed, so the
ontology's and the entry table's spellings agree) and compared by its ``name_key``. ChEBI names identify classes, so
the original's ids are not needed (``domains.chebi`` has the measurement). The answers are built from the records by
:mod:`data.relational_answers_molecule`. :class:`CompoundPool` indexes the records for pair and triplet draws as the
original ``Pool`` did.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from latent_interconnect.data.relational_base import WeightedPool
from latent_interconnect.domains.chebi import chebi_name, name_key
from latent_interconnect.domains.views import DomainView

__all__ = [
    "EDGE",
    "HELD_OUT_PARENTS",
    "Compound",
    "CompoundPool",
    "class_relation",
    "compound_record",
    "fp_classes",
    "fp_key",
    "fp_set",
    "shared_class_pair",
    "shared_parent",
    "verify_parents",
]

HELD_OUT_PARENTS = "functional_parents"
"""The molecule domain's extra table (by ``object_id``; column ``held_out``: the compound's functional parents that are
molecules of the domain outside the train split) that :func:`verify_parents` checks ``parent_classes`` against."""
EDGE = " > "
"""Between the two names of a graph edge (``"<class> > <parent>"``; no ChEBI name contains it)."""
_FIRST_SENTENCE = re.compile(r"(?<=[a-z0-9\)\]])\.\s+(?=[A-Z])")
_MIN_DEFINITION_CHARS = 12
_SIB_CAP = 5000
"""Hard-negative candidates gathered per draw before the scan stops (the original ``_sib_cand``)."""


@dataclass(frozen=True)
class Compound:
    """The facts of one training compound, parsed from its screened text.

    Attributes:
        object_id: the compound.
        classes: its direct classes, as written.
        definition: the first sentence of its definition (at least 12 characters), if any.
        parents: its functional parents, as written.
        roles: its trained mechanism roles.
        graph: a class above or at its direct classes (its name key) → that class's parents, as written.
        parent_classes: a functional parent (its name key) → its direct classes, as written.
        above: every class above one of its direct classes (its name key).
    """

    object_id: str
    classes: tuple[str, ...]
    definition: str | None
    parents: tuple[str, ...]
    roles: tuple[str, ...]
    graph: Mapping[str, tuple[str, ...]]
    parent_classes: Mapping[str, tuple[str, ...]]
    above: frozenset[str]

    def class_parents(self, cls: str) -> tuple[str, ...]:
        """Return a class's parents, as written (none for a class outside the compound's graph)."""
        return self.graph.get(name_key(cls), ())

    def ancestors(self, cls: str) -> frozenset[str]:
        """Return every class above ``cls`` in the compound's graph (its name key)."""
        return _closure(self.graph, (name_key(cls),))


def _closure(graph: Mapping[str, tuple[str, ...]], start: Sequence[str]) -> frozenset[str]:
    seen: set[str] = set()
    todo = [name_key(p) for c in start for p in graph.get(c, ())]
    while todo:
        c = todo.pop()
        if c not in seen:
            seen.add(c)
            todo += [name_key(p) for p in graph.get(c, ())]
    return frozenset(seen)


def _items(text: str | None, separator: str | None) -> tuple[str, ...]:
    if not text or not text.strip() or separator is None:
        return ()
    return tuple(chebi_name(x) for x in text.split(separator) if x.strip())  # every name, edges included


def _edges(object_id: str, items: Sequence[str], what: str) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for item in items:
        head, sep, tail = item.partition(EDGE)
        if not sep or not head.strip() or not tail.strip() or EDGE in tail:
            raise ValueError(f"{object_id} {what}: {item!r} is not an edge '<a>{EDGE}<b>'")
        out.setdefault(name_key(head), []).append(tail.strip())
    return {k: tuple(dict.fromkeys(v)) for k, v in out.items()}


def _first_sentence(text: str | None) -> str | None:
    """The original ``first_sentence``: the text up to its first sentence end, if at least 12 characters long."""
    if not text or not text.strip():
        return None
    s = _FIRST_SENTENCE.split(text.strip())[0].rstrip(".")
    return s if len(s) >= _MIN_DEFINITION_CHARS else None


def compound_record(
    object_id: str, fields: Mapping[str, str], names: Mapping[str, str], lists: Mapping[str, str]
) -> Compound:
    """Parse one compound's screened text.

    ``names`` maps each fact the stream reads to its training-text field (a fact not in it is absent) and ``lists`` each
    item-list fact to its separator.

    Raises:
        ValueError: if a graph item is not an edge.
    """

    def get(fact: str) -> str | None:
        return fields.get(names[fact]) if fact in names else None

    def items(fact: str) -> tuple[str, ...]:
        return _items(get(fact), lists.get(fact))

    classes = items("classes")
    graph = _edges(object_id, items("class_graph"), "class_graph")
    return Compound(
        object_id=object_id,
        classes=classes,
        definition=_first_sentence(get("definition")),
        parents=items("functional_parent"),
        roles=items("roles"),
        graph=graph,
        parent_classes=_edges(object_id, items("parent_classes"), "parent_classes"),
        above=_closure(graph, [name_key(c) for c in classes]),
    )


# ---------------------------------------------------------------------------------------------------- relations


def class_relation(a: Compound, b: Compound) -> str:
    """Return how two compounds' direct classes relate (the original ``class_relation``).

    ``shared``: a common direct class; ``nested``: a direct class of one is above one of the other's (ambiguous, the
    written-identity backstop); else ``disjoint``.
    """
    na, nb = {name_key(n) for n in a.classes}, {name_key(n) for n in b.classes}
    if na & nb:
        return "shared"
    if na & b.above or nb & a.above:
        return "nested"
    return "disjoint"


def shared_parent(a: Compound, shown_a: Sequence[str], b: Compound, shown_b: Sequence[str]) -> str | None:
    """A parent shared by a class of ``shown_a`` (in ``a``'s graph) and one of ``shown_b`` (in ``b``'s), as written.

    The most specific first (the most classes above it), then by name (originally ``shared_parent``, whose tie-break was
    the ChEBI id).
    """
    written = {name_key(p): p for c in shown_a for p in a.class_parents(c)}
    common = set(written) & {name_key(p) for c in shown_b for p in b.class_parents(c)}
    if not common:
        return None
    best = sorted(common, key=lambda p: (-len(a.ancestors(p)), p))[0]
    return written[best]


def fp_set(p: Compound) -> frozenset[str]:
    """The compound's functional parents, by name key (parent_relation compares these sets)."""
    return frozenset(name_key(n) for n in p.parents)


def fp_key(p: Compound) -> tuple[str, ...]:
    """The ordered key of :func:`fp_set` (the pool's same-parent groups)."""
    return tuple(sorted(fp_set(p)))


def shared_class_pair(
    a: Sequence[tuple[str, ...] | None], b: Sequence[tuple[str, ...] | None]
) -> tuple[int, int, str] | None:
    """The first class two lists of class lists share: (its list in ``a``, its list in ``b``, the class as in ``a``).

    ``a``'s lists in order, then ``b``'s; None when no pair of lists shares a class (ported from
    ``shared_fp_class_any`` and the scan of ``parent_relation_label``).
    """
    for i, xa in enumerate(a):
        for j, xb in enumerate(b):
            keys = {name_key(c) for c in xb or ()}
            found = next((c for c in xa or () if name_key(c) in keys), None)
            if found:
                return i, j, found
    return None


def verify_parents(compounds: Mapping[str, Compound], domain: DomainView) -> None:
    """Refuse a compound whose ``parent_classes`` names a functional parent outside the train split.

    Raises:
        ValueError: naming the count and the first compounds that break the contract.
    """
    if not any(p.parent_classes for p in compounds.values()):
        return
    table = domain.table(HELD_OUT_PARENTS)
    bad = []
    for o, p in compounds.items():
        held = table["held_out"].get(o) if o in table.index else None
        keys = {name_key(str(n)) for n in (held if held is not None else ())}
        if keys & set(p.parent_classes):
            bad.append(o)
    if bad:
        raise ValueError(
            f"{len(bad)} compounds state the classes of a functional parent outside the train split, e.g. {bad[:3]}: "
            "parent_classes may hold only parents in the train split or outside the domain"
        )


def fp_classes(p: Compound) -> list[tuple[str, ...] | None]:
    """Per functional parent, in order: its direct classes as written, or None when they are not known."""
    return [p.parent_classes.get(name_key(n)) or None for n in p.parents]


# ---------------------------------------------------------------------------------------------------- pool


class CompoundPool(WeightedPool):
    """The training compounds indexed for pair and triplet draws (the original ``Pool`` of ``build_relational_mol``).

    Classes are indexed by their written name; a class's siblings are the pool's classes that share a parent
    with it. Hard negatives (``p_hard`` of the odd-one-out triplets and a quarter of the pairs) come from a
    sibling class, with no direct class in common with (nor above or below) the classes of the compounds they are
    drawn against.

    Args:
        compounds: id → record.
        weight: id → draw weight (``cluster_sqrt`` over the Butina clusters).
        p_hard: the share of hard odd-one-out triplets.
    """

    def __init__(self, compounds: Mapping[str, Compound], weight: Mapping[str, float], p_hard: float) -> None:
        super().__init__(weight)
        self.P, self.p_hard = compounds, p_hard
        self.acc = list(compounds)
        self.by_cls: defaultdict[str, list[str]] = defaultdict(list)
        self.by_fp: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
        self.by_fpcls: defaultdict[str, list[str]] = defaultdict(list)
        parents: defaultdict[str, set[str]] = defaultdict(set)
        for a, p in compounds.items():
            for n in p.classes:
                self.by_cls[name_key(n)].append(a)
                parents[name_key(n)].update(name_key(x) for x in p.class_parents(n))
            if p.parents:
                self.by_fp[fp_key(p)].append(a)
                for c in dict.fromkeys(name_key(c) for x in fp_classes(p) for c in (x or ())):
                    self.by_fpcls[c].append(a)
        self.cls2 = [k for k, v in self.by_cls.items() if len(v) >= 2]  # noqa: PLR2004  (a pair)
        self.fp2 = [k for k, v in self.by_fp.items() if len(v) >= 2]  # noqa: PLR2004
        self.fpp = [a for v in self.by_fp.values() for a in v]
        kids: defaultdict[str, set[str]] = defaultdict(set)
        for k, ps in parents.items():
            for q in ps:
                kids[q].add(k)
        self.sib = {k: sorted({s for q in parents[k] for s in kids[q]} - {k}) for k in self.by_cls}
        self.cls2_hard = [k for k in self.cls2 if self.sib[k]]

    def _sib_cand(self, k: str, avoid: Sequence[str]) -> list[str]:
        """Compounds of the sibling classes of ``k`` whose classes are disjoint from every compound in ``avoid``."""
        out: list[str] = []
        for s in self.sib.get(k, ()):
            out += self.by_cls[s]
            if len(out) > _SIB_CAP:
                break
        P = self.P
        return [x for x in dict.fromkeys(out) if all(class_relation(P[x], P[y]) == "disjoint" for y in avoid)]

    def pair(self, g: np.random.Generator, family: str) -> tuple[str, str]:
        """~50 % related pairs, ~25 % hard negatives, ~25 % random (the original ``sample_pair``)."""
        u, P = g.random(), self.P
        if family in ("same_class", "commonalities", "differences") and u < 0.5 and self.cls2:  # noqa: PLR2004
            return self.pick2(g, self.by_cls[self.cls2[int(g.integers(len(self.cls2)))]])
        if family == "parent_relation":
            if u < 0.5 and self.fp2:  # noqa: PLR2004
                return self.pick2(g, self.by_fp[self.fp2[int(g.integers(len(self.fp2)))]])
            a = self.pick(g, self.fpp)
            cls_a = [name_key(c) for x in fp_classes(P[a]) for c in (x or ())]
            if u < 0.75 and cls_a:  # noqa: PLR2004  (a parent in the same class, another parent set)
                ka = fp_set(P[a])
                cand = list(dict.fromkeys(x for c in cls_a for x in self.by_fpcls[c] if not fp_set(P[x]) & ka))
                if cand:
                    return a, self.pick(g, cand, fixed=False)
            return a, self.pick(g, self.fpp)
        if u < 0.75:  # noqa: PLR2004  (a hard negative: a compound of a sibling class)
            a = self.pick(g, self.acc)
            p = P[a]
            cand = self._sib_cand(name_key(p.classes[0]), [a]) if p.classes else []
            if cand:
                return a, self.pick(g, cand, fixed=False)
        return self.pick2(g, self.acc)

    def triplet(self, g: np.random.Generator, family: str) -> list[str] | None:
        """``[a, b, odd]`` (the original ``sample_triplet``), or None if no odd compound fits.

        ``p_hard`` of them take the odd one from a sibling class of the pair's shared class; the others, and a hard one
        that finds no candidate, a random compound whose classes are disjoint from the pair's.
        """
        hard = g.random() < self.p_hard
        if family != "odd_one_out_class" or not self.cls2:
            return None
        ks = self.cls2_hard if hard and self.cls2_hard else self.cls2
        k = ks[int(g.integers(len(ks)))]
        a, b = self.pick2(g, self.by_cls[k])
        if hard and self.cls2_hard:
            cand = self._sib_cand(k, [a, b])
            if cand:
                return [a, b, self.pick(g, cand, fixed=False)]
        P = self.P
        c = next(
            (
                x
                for x in self.pickn(g, self.acc, 50)
                if P[x].classes and all(class_relation(P[x], P[y]) == "disjoint" for y in (a, b))
            ),
            None,
        )
        return [a, b, c] if c else None
