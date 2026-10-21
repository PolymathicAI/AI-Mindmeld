"""Molecule relational answers (the paper's molecule runs): templates over compound records.

A port of the answers of the original molecule relational generator (the one the paper's molecule
runs used), registered as the ``molecule`` relational answers (:data:`MOLECULE`). Its six families
are the protein families with a ChEBI analogue in the trained fields, and all six were trained:

- one compound: ``profile`` (classes, definition, functional parents, trained mechanism roles; three facts at least);
- pairs: ``same_class`` (Yes / No, the written classes, a shared parent class), ``differences`` and
  ``commonalities`` (classes, functional parents, roles), ``parent_relation`` (the functional-parent sets and their
  classes, then a verdict);
- triplets: ``odd_one_out_class`` (the three classes, then the odd one out).

Answers follow the protein answers' grammar (evidence first, a one-word verdict last; written identity,
fact kinds and groups, a compact form without parentheticals) and call the object a "compound". Every answer is
fixed template text (all of it in :data:`ANSWER_WORDING`, which the stream declares to the zero-shot screen) with
facts from the records (:mod:`data.relational_records_molecule`), marked with :mod:`data.relational_base`.
"""

from __future__ import annotations

from collections.abc import Sequence

from latent_interconnect.data.relational_base import (
    RelationalAnswers,
    fill,
    group,
    sentences,
)
from latent_interconnect.data.relational_records_molecule import (
    HELD_OUT_PARENTS,
    Compound,
    CompoundPool,
    class_relation,
    compound_record,
    fp_classes,
    fp_set,
    shared_class_pair,
    shared_parent,
    verify_parents,
)
from latent_interconnect.domains.chebi import name_key

__all__ = [
    "ANSWER_WORDING",
    "FACTS",
    "FILL_KINDS",
    "MOLECULE",
    "ORDINALS",
    "PAIR",
    "SINGLE",
    "TRIPLE",
    "VERDICT_SENTENCE",
    "VERDICT_WORDS",
    "odd_classes",
    "parent_relation_label",
]

ORDINALS = ("first", "second", "third")
FACTS = ("classes", "definition", "functional_parent", "roles")
"""The facts a family may state (the generator's fields; the trained mechanism roles are ``roles``)."""
FILL_KINDS = ("class", "superclass", "definition", "functional_parent", "parent_class", "role", "choice", "verdict")
"""Fact kinds of the fills (the loss weights map each to a curve, ``RelationalParams.fact_curves``)."""

VERDICT_WORDS = {
    "parent_relation": {"same": "identical", "partial": "related", "different": "unrelated"},
    "odd_one_out_class": {o: o for o in ORDINALS},
}
"""The verdict word of each label (evidence first, a one-word verdict last)."""
VERDICT_SENTENCE = {
    "parent_relation": "So their functional parents are {w}.",
    "odd_one_out_class": "So the {w} is the odd one out.",
}

_TEMPLATES = (
    "It belongs to the class …. It belongs to the classes …, … and ….",
    "It is derived from … and ….",
    "It has the role …. It has the roles …, … and ….",
    "Yes. Both compounds belong to the class …. Both compounds belong to the classes … and ….",
    "No. The first belongs to the class …, whereas the second belongs to the classes … and …. Their classes do share "
    "the parent class ….",
    "Both belong to the class …. The first belongs to the class … and the second to the class …. Both fall under the "
    "parent class ….",
    "Both are derived from …. The first is also derived from …, the second also from …. Only the first is also derived "
    "from …. Only the second is also derived from ….",
    "Both are derived from compounds of the class …. Within it, the first is derived from … and the second from ….",
    "The first is derived from …, the second from …. Only the first is annotated as derived from …. Only the second is "
    "annotated as derived from ….",
    "Both have the role …. The first has the roles … and … and the second the roles … and ….",
    "On the other hand, …. However, ….",
    "They have little in common.",
    "The first is derived from … (class …), and the second from … (class …) and … (class …).",
    "The first belongs to the class … (a subclass of …), the second to the class …, and the third to the class ….",
)
"""Every answer template's fixed text, fills shown as "…" (the screen reads the words around them)."""

ANSWER_WORDING: tuple[str, ...] = (
    *_TEMPLATES,
    *(VERDICT_SENTENCE[f].format(w=w) for f, words in VERDICT_WORDS.items() for w in words.values()),
)
"""The molecule relational answers' fixed wording: the templates and the verdict sentences. No word is derived from a
fact (every class, parent and role name is copied from the screened text)."""


def _verdict(family: str, label: str) -> str:
    return VERDICT_SENTENCE[family].format(w=fill(VERDICT_WORDS[family][label], "verdict"))


def _and(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def class_list(names: Sequence[str], kind: str = "class") -> str:
    """'the class X' / 'the classes X and Y' / 'the classes X, Y and Z' (one fact group)."""
    return group(f"the class{'es' if len(names) > 1 else ''} " + _and([fill(n, kind) for n in names]))


def class_one(name: str, sup: str | None) -> str:
    """'the class X (a subclass of Y)' (one fact group); no parenthetical without ``sup`` (compact answers)."""
    return group(f"the class {fill(name, 'class')}" + (f" (a subclass of {fill(sup, 'superclass')})" if sup else ""))


def role_list(roles: Sequence[str]) -> str:
    """'the role X' / 'the roles X, Y and Z' (one fact group)."""
    return group(f"the role{'s' if len(roles) > 1 else ''} " + _and([fill(r, "role") for r in roles]))


def fp_names(names: Sequence[str]) -> str:
    """'X' / 'X and Y' / 'X, Y and Z' over functional parents (several: one fact group)."""
    ns = [fill(n, "functional_parent") for n in names]
    return ns[0] if len(ns) == 1 else group(_and(ns))


def _shared_classes(a: Compound, b: Compound) -> list[str]:
    nb = {name_key(n) for n in b.classes}
    return [c for c in a.classes if name_key(c) in nb]


# ---------------------------------------------------------------------------------------------------- one compound


def ans_profile(ps: Sequence[Compound], ok: Sequence[str], compact: bool) -> str | None:
    """A short profile: classes, definition, functional parents, roles (three sentences at least)."""
    del compact
    p, s = ps[0], []
    if "classes" in ok and p.classes:
        s.append(f"It belongs to {class_list(p.classes[:3])}.")
    if "definition" in ok and p.definition:
        s.append(fill(p.definition, "definition") + ".")
    if "functional_parent" in ok and p.parents:
        s.append("It is derived from " + group(" and ".join(fill(n, "functional_parent") for n in p.parents[:2])) + ".")
    if "roles" in ok and p.roles:
        s.append(f"It has {role_list(p.roles[:3])}.")
    return " ".join(s) if len(s) >= 3 else None  # noqa: PLR2004  (three facts at least)


# ---------------------------------------------------------------------------------------------------- pairs


def ans_same_class(ps: Sequence[Compound], ok: Sequence[str], compact: bool) -> str | None:
    """Yes / No, then the written classes (and a parent class the shown classes share)."""
    del ok, compact
    a, b = ps
    if not (a.classes and b.classes):
        return None
    relation = class_relation(a, b)
    if relation == "shared":
        return f"{fill('Yes', 'choice')}. Both compounds belong to {class_list(_shared_classes(a, b)[:2])}."
    if relation == "nested":  # one class is above the other's: ambiguous (the written-identity rule's analogue)
        return None
    sa, sb = a.classes[:3], b.classes[:3]
    sup = shared_parent(a, sa, b, sb)
    tail = f" Their classes do share the parent class {fill(sup, 'superclass')}." if sup else ""
    return (
        f"{fill('No', 'choice')}. The first belongs to {class_list(sa)}, whereas the second belongs to "
        f"{class_list(sb)}.{tail}"
    )


def _parent_facts(a: Compound, b: Compound, sh: list[str], df: list[str]) -> None:
    """The functional-parent facts of a pair over the parent sets, every parent named."""
    fa, fb = a.parents, b.parents
    if fa and fb:
        sa, sb = fp_set(a), fp_set(b)
        if sa == sb:
            sh.append(f"both are derived from {fp_names(fa)}")
        elif sa & sb:
            ra, rb = [x for x in fa if name_key(x) not in sb], [x for x in fb if name_key(x) not in sa]
            sh.append(f"both are derived from {fp_names([x for x in fa if name_key(x) in sb])}")
            if ra and rb:
                df.append(f"the first is also derived from {fp_names(ra)}, the second also from {fp_names(rb)}")
            else:
                df.append(
                    f"only the {fill('first' if ra else 'second', 'choice')} is also derived from {fp_names(ra or rb)}"
                )
        else:
            found = shared_class_pair(fp_classes(a), fp_classes(b))
            pc = found[2] if found else None
            if pc:
                sh.append(f"both are derived from compounds of the class {fill(pc, 'parent_class')}")
            if pc and len(fa) == len(fb) == 1:
                df.append(f"within it, the first is derived from {fp_names(fa)} and the second from {fp_names(fb)}")
            else:
                df.append(f"the first is derived from {fp_names(fa)}, the second from {fp_names(fb)}")
    elif fa or fb:
        df.append(
            f"only the {fill('first' if fa else 'second', 'choice')} is annotated as derived from {fp_names(fa or fb)}"
        )


def _pair_facts(a: Compound, b: Compound, ok: Sequence[str]) -> tuple[list[str], list[str]]:
    """(shared facts, differing facts) over the allowed facts."""
    sh: list[str] = []
    df: list[str] = []
    if "classes" in ok and a.classes and b.classes:
        relation = class_relation(a, b)
        if relation == "shared":
            sh.append(f"both belong to {class_list(_shared_classes(a, b)[:2])}")
        elif relation == "disjoint":  # a stated difference is not a subsumption
            sa, sb = a.classes[:3], b.classes[:3]
            df.append(f"the first belongs to {class_list(sa)} and the second to {class_list(sb)}")
            sup = shared_parent(a, sa, b, sb)
            if sup:
                sh.append(f"both fall under the parent class {fill(sup, 'superclass')}")
    if "functional_parent" in ok:
        _parent_facts(a, b, sh, df)
    if "roles" in ok and a.roles and b.roles:
        common = [r for r in a.roles if r in set(b.roles)][:2]
        if common:
            sh.append(f"both have {role_list(common)}")
        else:
            df.append(f"the first has {role_list(a.roles[:2])} and the second {role_list(b.roles[:2])}")
    return sh, df


def ans_differences(ps: Sequence[Compound], ok: Sequence[str], compact: bool) -> str | None:
    """At least two differences (then one shared fact, if any)."""
    del compact
    sh, df = _pair_facts(ps[0], ps[1], ok)
    if len(df) < 2:  # noqa: PLR2004  (two differences at least)
        return None
    return sentences(df[:4]) + (f" On the other hand, {sh[0]}." if sh else "")


def ans_commonalities(ps: Sequence[Compound], ok: Sequence[str], compact: bool) -> str | None:
    """Shared facts (then one difference), or 'little in common' and two differences."""
    del compact
    sh, df = _pair_facts(ps[0], ps[1], ok)
    if not sh:
        return (f"They have {fill('little', 'verdict')} in common. " + sentences(df[:2])) if len(df) >= 2 else None  # noqa: PLR2004
    return sentences(sh[:4]) + (f" However, {df[0]}." if df else "")


def parent_relation_label(a: Compound, b: Compound) -> tuple[str, list[str | None], list[str | None]]:
    """(label, the class shown for each of a's parents, for each of b's), over the parent sets (as in the original).

    ``same`` iff the two sets of parent names are equal; ``partial`` if they share a parent, or a parent of each has a
    direct class in common (the pathway category's analogue); else ``different``. Each parent shows its first class,
    except that the pair sharing a class shows it.
    """
    la, lb = fp_classes(a), fp_classes(b)
    ca: list[str | None] = [x[0] if x else None for x in la]
    cb: list[str | None] = [x[0] if x else None for x in lb]
    sa, sb = fp_set(a), fp_set(b)
    if sa == sb:
        return "same", ca, cb
    if sa & sb:
        return "partial", ca, cb
    found = shared_class_pair(la, lb)
    if found:
        i, j, cls = found
        ca[i], cb[j] = cls, cls
        return "partial", ca, cb
    return "different", ca, cb


def _parents_phrase(p: Compound, shown: Sequence[str | None], compact: bool) -> str:
    """'X (class C) and Y (class D)': each functional parent with its class, one fact group each."""
    return _and(
        [
            group(fill(n, "functional_parent") + ("" if compact or not c else f" (class {fill(c, 'parent_class')})"))
            for n, c in zip(p.parents, shown, strict=True)
        ]
    )


def ans_parent_relation(ps: Sequence[Compound], ok: Sequence[str], compact: bool) -> str | None:
    """Both functional-parent sets (with their classes), then the verdict."""
    del ok
    a, b = ps
    if not (a.parents and b.parents):
        return None
    label, ca, cb = parent_relation_label(a, b)
    return (
        f"The first is derived from {_parents_phrase(a, ca, compact)}, and the second from "
        f"{_parents_phrase(b, cb, compact)}. {_verdict('parent_relation', label)}"
    )


# ---------------------------------------------------------------------------------------------------- triplets


def odd_classes(x: Compound, y: Compound, odd: Compound) -> tuple[str, str, str | None]:
    """(the pair's shown class, the odd one's shown class, their shared parent or None).

    The first class the pair shares and the first class of the odd one that have a parent in common (a hard negative's
    sibling classes), else the first of each without a parent. The original showed the class the sampler had drawn.
    """
    shared = _shared_classes(x, y)
    for k in shared:
        for s in odd.classes:
            sup = shared_parent(x, [k], odd, [s])
            if sup:
                return k, s, sup
    return shared[0], odd.classes[0], None


def ans_odd_class(ps: Sequence[Compound], ok: Sequence[str], compact: bool, odd: int) -> str | None:
    """The three classes (with their shared parent), then the odd one out."""
    del ok
    if not all(p.classes for p in ps):
        return None
    pair = [i for i in range(3) if i != odd]
    x, y = ps[pair[0]], ps[pair[1]]
    if class_relation(x, y) != "shared" or any(class_relation(ps[odd], ps[i]) != "disjoint" for i in pair):
        return None  # written-identity backstop: the pair shares a class, the odd one is unrelated to both
    k, s, sup = odd_classes(x, y, ps[odd])
    shown = [s if i == odd else k for i in range(3)]
    ph = [class_one(shown[i], None if compact else sup) for i in range(3)]
    return (
        f"The first belongs to {ph[0]}, the second to {ph[1]}, and the third to {ph[2]}. "
        f"{_verdict('odd_one_out_class', ORDINALS[odd])}"
    )


SINGLE = {"profile": ans_profile}
PAIR = {
    "same_class": ans_same_class,
    "differences": ans_differences,
    "commonalities": ans_commonalities,
    "parent_relation": ans_parent_relation,
}
TRIPLE = {"odd_one_out_class": ans_odd_class}

MOLECULE = RelationalAnswers(
    noun="compound",
    facts=FACTS,
    requires={"classes": ("class_graph",), "functional_parent": ("parent_classes",)},
    lists=frozenset({"classes", "functional_parent", "roles", "class_graph", "parent_classes"}),
    fill_kinds=FILL_KINDS,
    single=SINGLE,
    pair=PAIR,
    triple=TRIPLE,
    verdict_words=VERDICT_WORDS,
    verdict_sentence=VERDICT_SENTENCE,
    wording=ANSWER_WORDING,
    record=compound_record,
    pool=CompoundPool,
    tables={"parent_classes": (HELD_OUT_PARENTS,)},
    verify=verify_parents,
    yes_no=frozenset({"same_class"}),
)
"""The molecule relational answers, registered as ``molecule`` by the data package."""
