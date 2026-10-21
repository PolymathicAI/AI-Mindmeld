"""Relational answers: the answer templates over protein records, with their facts marked.

A port of the original protein relational generator, for the fields
the original runs trained under their task split (family, location, function, pathway, EC, catalytic activity); the
families that read only blocked fields (``length_compare``, ``taxonomy_compare``) and the blocked facts (cofactor, GO)
are not ported. The records the answers read are parsed by :mod:`data.relational_records`. Every answer
is fixed template text (all of it in :data:`ANSWER_WORDING`, which the stream declares to the zero-shot screen) with
filled-in protein facts. Fills are marked while an answer is built and cleaned into its text, spans, fact kinds and
fact groups by the domain-agnostic :mod:`data.relational_base` (a group is one protein's fact span: the curve of the
loss weights restarts per group). Proteins are referred to by position
("the first", "the second", "the third"), never by name. The templates stay in code, one function per original one:
each decides whether an answer exists and computes its verdict, which a config template language would split from
its wording.
The module stays above the 400-line guideline because it is one unit, reviewed against the one original module it
ports (records, the other concern, are in :mod:`data.relational_records`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from latent_interconnect.data.relational_base import Clean, clean, fill, group, plain
from latent_interconnect.data.relational_base import sentences as _sentences
from latent_interconnect.data.relational_records import EC_CLASS, LOCATION_WORDS, Protein
from latent_interconnect.domains.protein import PathwayEntry

__all__ = [
    "ANSWER_WORDING",
    "FILL_KINDS",
    "ORDINALS",
    "PAIR",
    "SINGLE",
    "TRIPLE",
    "VERDICT_SENTENCE",
    "VERDICT_WORDS",
    "Clean",
    "clean",
    "coarse",
    "fam_key",
    "fam_name",
    "top_compartments",
]

ORDINALS = ("first", "second", "third")
_KEEP_CAPS = ("Golgi", "Cajal", "Z line", "Z disc", "PML", "P-body")

FILL_KINDS = (
    "family",
    "superfamily",
    "location",
    "topology",
    "function",
    "pathway",
    "category",
    "step",
    "ec_number",
    "ec_class",
    "catalytic",
    "choice",
    "verdict",
)
"""Fact kinds of the fills (the loss weights map each to a curve, ``RelationalParams.fact_curves``)."""


def lc(term: str) -> str:
    """Lower-case a location or family term for mid-sentence use, keeping proper names and acronyms."""
    w = term.split(" ", maxsplit=1)[0]
    if re.match(r"^(T|R|M|Sn|Sno)RNA", term):
        return term[0].lower() + term[1:]
    keep = re.match(r"^[A-Z][0-9,']*-", term) or any(term.startswith(k) for k in _KEEP_CAPS)
    if keep or "/" in w or any(c.isupper() for c in w[1:]):
        return term
    return term[0].lower() + term[1:]


def _an(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def fam_phrase(p: Protein, sup: bool = True) -> str:
    """The written family: 'the X family (part of the Y superfamily)'."""
    f = p.family
    if f is None:
        raise ValueError(f"{p.object_id} has no family")
    tail = ""
    if sup and f.superfamily and f.superfamily != f.family:
        tail = f" (part of the {fill(lc(f.superfamily), 'superfamily')})"
    return group(f"the {fill(lc(f.family), 'family')}" + tail)


def fam_key(p: Protein) -> str:
    """The written family identity: the phrase an answer shows."""
    return plain(fam_phrase(p)).lower()


def fam_name(p: Protein) -> str:
    """The written family name without the superfamily."""
    if p.family is None:
        raise ValueError(f"{p.object_id} has no family")
    return lc(p.family.family).lower()


def shown_locs(p: Protein) -> tuple[str, ...]:
    """The locations an answer displays (at most three)."""
    return p.location.locs[:3] if p.location else ()


def _shown_topo(p: Protein) -> str | None:
    starts = ("single-pass", "multi-pass", "peripheral", "lipid-anchor")
    topo = [x[0].lower() + x[1:] for x in (p.location.topo if p.location else ()) if x.lower().startswith(starts)]
    return topo[0] if topo else None


def loc_phrase(p: Protein, topo: bool = True) -> str:
    """The displayed locations (and topology), one fact group."""
    s = " and ".join(fill(lc(x), "location") for x in shown_locs(p))
    t = _shown_topo(p) if topo else None
    return group(s + (f" ({fill(t, 'topology')})" if t else ""))


def _is_membrane(p: Protein) -> bool:
    return bool(p.location) and (
        any("membrane" in x.lower() for x in shown_locs(p)) or "membrane" in (_shown_topo(p) or "").lower()
    )


def _pw_phrase(pw: PathwayEntry, cat: bool = True) -> str:
    c = f" ({fill(lc(pw.category), 'category')})" if cat and pw.category != pw.pathway else ""
    return group(f"{fill(lc(pw.pathway), 'pathway')}{c}")


def top_compartments(p: Protein) -> set[str]:
    """The top-level compartments of a protein's locations."""
    return {x.split(",")[0].strip().lower() for x in (p.location.locs if p.location else ())}


_COARSE = (
    ("nucleus", ("nucle", "chromosome")),
    ("mitochondrion", ("mitochondri",)),
    ("plastid", ("plastid", "chloroplast", "thylakoid", "amyloplast", "chromoplast")),
    (
        "endomembrane",
        (
            "endoplasmic reticulum",
            "golgi",
            "vesicle",
            "endosome",
            "lysosome",
            "vacuole",
            "peroxisome",
            "microsome",
            "secreted",
            "extracellular",
            "cell wall",
            "secretory",
        ),
    ),
    (
        "cell_surface",
        ("cell membrane", "cell surface", "cell junction", "cell projection", "synapse", "cilium", "flagellum"),
    ),
    ("cytoplasm", ("cytoplasm", "cytosol", "cytoskeleton")),
)


def coarse(loc: str) -> str:
    """The coarse compartment class of a location (odd-one-out hard negatives)."""
    top = loc.split(",", maxsplit=1)[0].strip().lower()
    return next((c for c, keys in _COARSE if any(k in top for k in keys)), "other")


# ---------------------------------------------------------------------------------------------------- answers

_TEMPLATES = (
    "It belongs to … (part of the …).",
    "It is found in the … and … (…).",
    "It takes part in … (…), ….",
    "It is a … (EC …). It catalyzes the reaction: ….",
    "It is an … (EC …).",
    "Yes. Both proteins belong to the ….",
    "No. The first belongs to the …, whereas the second belongs to the …. They do share the ….",
    "The first is found in the …, and the second in the …. Only the first is membrane-associated.",
    "Only the second is membrane-associated.",
    "Both belong to the …. The first belongs to the … and the second to the ….",
    "Both are in the …. Both are found in the … and ….",
    "The first is found in the …, the second in the ….",
    "Both are …s. The first is a … and the second an ….",
    "The first is an … and the second a ….",
    "Only the first is annotated as an enzyme (EC …, a …). Only the second is annotated as an enzyme (EC …, an …).",
    "Both take part in …. Both act in …. Within it, the first is involved in … and the second in ….",
    "The first acts in …, the second in ….",
    "On the other hand, …. However, ….",
    "They have little in common.",
    "The first is EC … (a …), and the second EC … (an …).",
    "The first takes part in … (…), and the second in … (…).",
    "The first catalyzes … and the second ….",
    "The first belongs to the …, the second to the …, and the third to the ….",
    "The first is found in the …, the second in the …, and the third in the ….",
)
"""Every answer template's fixed text, fills shown as "…" (the screen reads the words around them)."""

VERDICT_WORDS = {
    "compare_location": {"same": "identical", "partial": "overlapping", "different": "disjoint"},
    "pathway_relation": {"same": "identical", "partial": "related", "different": "unrelated"},
    "ec_depth": {f"d{k}": w for k, w in enumerate(("none", "one", "two", "three", "all four"))},
    "odd_one_out_family": {o: o for o in ORDINALS},
    "odd_one_out_location": {o: o for o in ORDINALS},
}
"""The verdict word of each label (evidence first, a one-word verdict last)."""
VERDICT_SENTENCE = {
    "compare_location": "So their localizations are {w}.",
    "pathway_relation": "So their pathways are {w}.",
    "ec_depth": "So they agree on {w} of the four EC levels.",
    "odd_one_out_family": "So the {w} is the odd one out.",
    "odd_one_out_location": "So the {w} localizes differently from the other two.",
}


ANSWER_WORDING: tuple[str, ...] = (
    *_TEMPLATES,
    *(VERDICT_SENTENCE[f].format(w=w) for f, words in VERDICT_WORDS.items() for w in words.values()),
    *(f"{c} {c}s" for c in EC_CLASS.values()),
    *LOCATION_WORDS,
)
"""The relational answers' fixed wording: the templates, the verdict sentences, and the words an answer derives from
a fact rather than copying it (the EC class names from the EC number, the secreted location), so the zero-shot screen
of the stream's wording sees what the field screen cannot."""


def _verdict(family: str, label: str) -> str:
    return VERDICT_SENTENCE[family].format(w=fill(VERDICT_WORDS[family][label], "verdict"))


Answer = Callable[[Sequence[Protein], Sequence[str], bool], str | None]
"""(proteins, allowed facts, compact) → a marked answer, or None when the proteins do not give one."""


def ans_profile(ps: Sequence[Protein], ok: Sequence[str], compact: bool) -> str | None:
    """A short profile: family, location, function, pathway (three sentences at least)."""
    del compact
    p, s = ps[0], []
    if "family" in ok and p.family:
        s.append(f"It belongs to {fam_phrase(p)}.")
    if "location" in ok and p.location:
        s.append(f"It is found in the {loc_phrase(p)}.")
    if "function" in ok and p.function:
        s.append(fill(p.function, "function") + ".")
    if "pathway" in ok and p.pathway:
        pw = p.pathway
        cat = f" ({fill(lc(pw.category), 'category')})" if pw.category != pw.pathway else ""
        step = f", {fill(pw.step, 'step')}" if pw.step else ""
        s.append("It takes part in " + group(f"{fill(lc(pw.pathway), 'pathway')}{cat}{step}") + ".")
    return " ".join(s) if len(s) >= 3 else None  # noqa: PLR2004  (three facts at least)


def ans_enzyme(ps: Sequence[Protein], ok: Sequence[str], compact: bool) -> str | None:
    """The EC class and number and the reaction."""
    del compact
    p = ps[0]
    if not p.ec or not (p.catalytic and "catalytic_activity" in ok):
        return None
    c = EC_CLASS[p.ec[0]]
    s = "It is " + group(f"{_an(c)} {fill(c, 'ec_class')} (EC {fill('.'.join(p.ec), 'ec_number')})") + "."
    return s + f" It catalyzes the reaction: {fill(p.catalytic.rstrip('.'), 'catalytic')}."


def ans_same_family(ps: Sequence[Protein], ok: Sequence[str], compact: bool) -> str | None:
    """Yes / No, then the written family or families."""
    del ok, compact
    a, b = ps
    if not (a.family and b.family):
        return None
    if fam_key(a) == fam_key(b):
        return f"{fill('Yes', 'choice')}. Both proteins belong to {fam_phrase(a)}."
    if fam_name(a) == fam_name(b):
        return None
    sa, sb = a.family.superfamily, b.family.superfamily
    tail = f" They do share the {fill(lc(sa), 'superfamily')}." if sa and sa == sb else ""
    return (
        f"{fill('No', 'choice')}. The first belongs to {fam_phrase(a)}, whereas the second belongs to "
        f"{fam_phrase(b)}.{tail}"
    )


def ans_compare_location(ps: Sequence[Protein], ok: Sequence[str], compact: bool) -> str | None:
    """Both locations, then the verdict (and which one is membrane-associated)."""
    del ok
    a, b = ps
    if not (a.location and b.location):
        return None
    pa, pb = loc_phrase(a, topo=not compact), loc_phrase(b, topo=not compact)
    la, lb = set(shown_locs(a)), set(shown_locs(b))
    label = "same" if (la == lb or plain(pa) == plain(pb)) else ("partial" if la & lb else "different")
    s = f"The first is found in the {pa}, and the second in the {pb}. {_verdict('compare_location', label)}"
    ma, mb = _is_membrane(a), _is_membrane(b)
    if ma != mb and not compact:
        s += f" Only the {fill('first' if ma else 'second', 'verdict')} is membrane-associated."
    return s


def _pair_facts(a: Protein, b: Protein, ok: Sequence[str]) -> tuple[list[str], list[str]]:  # noqa: PLR0912  (original)
    """(shared facts, differing facts) over the allowed facts."""
    sh: list[str] = []
    df: list[str] = []
    if "family" in ok and a.family and b.family:
        if fam_key(a) == fam_key(b):
            sh.append(f"both belong to {fam_phrase(a)}")
        elif fam_name(a) != fam_name(b):
            df.append(f"the first belongs to {fam_phrase(a)} and the second to {fam_phrase(b)}")
            if a.family.superfamily and a.family.superfamily == b.family.superfamily:
                sh.append(f"both are in the {fill(lc(a.family.superfamily), 'superfamily')}")
    if "location" in ok and a.location and b.location:
        common = set(a.location.locs) & set(b.location.locs)
        if common:
            sh.append("both are found in the " + group(" and ".join(fill(lc(x), "location") for x in sorted(common))))
        if set(shown_locs(a)) != set(shown_locs(b)) and plain(loc_phrase(a)) != plain(loc_phrase(b)):
            df.append(f"the first is found in the {loc_phrase(a)}, the second in the {loc_phrase(b)}")
    if "ec" in ok and a.ec and b.ec:
        ca, cb = EC_CLASS[a.ec[0]], EC_CLASS[b.ec[0]]
        if ca == cb:
            sh.append(f"both are {fill(ca, 'ec_class')}s")
        else:
            df.append(f"the first is {_an(ca)} {fill(ca, 'ec_class')} and the second {_an(cb)} {fill(cb, 'ec_class')}")
    elif "ec" in ok and bool(a.ec) != bool(b.ec):
        e = a if a.ec else b
        ec = e.ec or ()
        c = EC_CLASS[ec[0]]
        df.append(
            f"only the {fill('first' if a.ec else 'second', 'choice')} is annotated as an enzyme "
            + group(f"(EC {fill('.'.join(ec), 'ec_number')}, {_an(c)} {fill(c, 'ec_class')})")
        )
    if "pathway" in ok and a.pathway and b.pathway:
        if a.pathway.pathway == b.pathway.pathway:
            sh.append(f"both take part in {fill(lc(a.pathway.pathway), 'pathway')}")
        elif a.pathway.category == b.pathway.category:
            sh.append(f"both act in {fill(lc(a.pathway.category), 'category')}")
            df.append(
                f"within it, the first is involved in {fill(lc(a.pathway.pathway), 'pathway')} and the second in "
                f"{fill(lc(b.pathway.pathway), 'pathway')}"
            )
        else:
            df.append(
                f"the first acts in {fill(lc(a.pathway.pathway), 'pathway')}, the second in "
                f"{fill(lc(b.pathway.pathway), 'pathway')}"
            )
    return sh, df


def ans_differences(ps: Sequence[Protein], ok: Sequence[str], compact: bool) -> str | None:
    """At least two differences (then one shared fact, if any)."""
    del compact
    sh, df = _pair_facts(ps[0], ps[1], ok)
    if len(df) < 2:  # noqa: PLR2004  (two differences at least)
        return None
    s = _sentences(df[:4])
    return s + (f" On the other hand, {sh[0]}." if sh else "")


def ans_commonalities(ps: Sequence[Protein], ok: Sequence[str], compact: bool) -> str | None:
    """Shared facts (then one difference), or 'little in common' and two differences."""
    del compact
    sh, df = _pair_facts(ps[0], ps[1], ok)
    if not sh:
        return (f"They have {fill('little', 'verdict')} in common. " + _sentences(df[:2])) if len(df) >= 2 else None  # noqa: PLR2004
    return _sentences(sh[:4]) + (f" However, {df[0]}." if df else "")


def _ec_prefix(x: Sequence[str], y: Sequence[str]) -> int:
    depth = 0
    for u, v in zip(x, y, strict=False):
        if u != v or u == "-":
            break
        depth += 1
    return depth


def ans_ec_depth(ps: Sequence[Protein], ok: Sequence[str], compact: bool) -> str | None:
    """Both EC numbers (the closest pair over the EC sets), then how many levels they share."""
    del ok, compact
    a, b = ps
    if not (a.ec and b.ec):
        return None
    best: tuple[int, Sequence[str], Sequence[str]] = (-1, a.ec, b.ec)
    for x in a.ecs or (a.ec,):
        for y in b.ecs or (b.ec,):
            d = _ec_prefix(x, y)
            if d > best[0]:
                best = (d, x, y)
    depth, xa, xb = best
    ca, cb = EC_CLASS[xa[0]], EC_CLASS[xb[0]]
    ea = group(f"EC {fill('.'.join(xa), 'ec_number')} ({_an(ca)} {fill(ca, 'ec_class')})")
    eb = group(f"EC {fill('.'.join(xb), 'ec_number')} ({_an(cb)} {fill(cb, 'ec_class')})")
    return f"The first is {ea}, and the second {eb}. {_verdict('ec_depth', f'd{depth}')}"


def _pathway_label(a: Protein, b: Protein) -> tuple[str, PathwayEntry, PathwayEntry]:
    ea = a.pathways or ((a.pathway,) if a.pathway else ())
    eb = b.pathways or ((b.pathway,) if b.pathway else ())
    names = {e.pathway for e in eb}
    x = next((e for e in ea if e.pathway in names), None)
    if x is not None:
        return "same", x, next(e for e in eb if e.pathway == x.pathway)
    cats = {e.category for e in eb}
    x = next((e for e in ea if e.category in cats), None)
    if x is not None:
        return "partial", x, next(e for e in eb if e.category == x.category)
    return "different", ea[0], eb[0]


def ans_pathway_relation(ps: Sequence[Protein], ok: Sequence[str], compact: bool) -> str | None:
    """Both pathways (over the pathway sets), then the verdict (and the steps of a shared pathway)."""
    del ok
    a, b = ps
    if not (a.pathway and b.pathway):
        return None
    label, pa, pb = _pathway_label(a, b)
    s = (
        f"The first takes part in {_pw_phrase(pa, not compact)}, and the second in {_pw_phrase(pb, not compact)}. "
        f"{_verdict('pathway_relation', label)}"
    )
    if label == "same" and pa.step and pb.step and pa.step != pb.step and not compact:
        s += f" The first catalyzes {fill(pa.step, 'step')} and the second {fill(pb.step, 'step')}."
    return s


def ans_odd_family(ps: Sequence[Protein], ok: Sequence[str], compact: bool, odd: int) -> str | None:
    """The three written families, then the odd one out."""
    del ok
    if not all(p.family for p in ps):
        return None
    pair = [i for i in range(3) if i != odd]
    if fam_key(ps[pair[0]]) != fam_key(ps[pair[1]]) or fam_name(ps[odd]) in (
        fam_name(ps[pair[0]]),
        fam_name(ps[pair[1]]),
    ):
        return None
    sup = not compact
    return (
        f"The first belongs to {fam_phrase(ps[0], sup)}, the second to {fam_phrase(ps[1], sup)}, and the third to "
        f"{fam_phrase(ps[2], sup)}. {_verdict('odd_one_out_family', ORDINALS[odd])}"
    )


def ans_odd_location(ps: Sequence[Protein], ok: Sequence[str], compact: bool, odd: int) -> str | None:
    """The three locations, then which one localizes differently."""
    del ok
    if not all(p.location for p in ps):
        return None
    pair = [i for i in range(3) if i != odd]
    common = set(shown_locs(ps[pair[0]])) & set(shown_locs(ps[pair[1]]))
    if not common or top_compartments(ps[odd]) & (top_compartments(ps[pair[0]]) | top_compartments(ps[pair[1]])):
        return None
    t = not compact
    return (
        f"The first is found in the {loc_phrase(ps[0], t)}, the second in the {loc_phrase(ps[1], t)}, and the third "
        f"in the {loc_phrase(ps[2], t)}. {_verdict('odd_one_out_location', ORDINALS[odd])}"
    )


SINGLE: dict[str, Answer] = {"profile": ans_profile, "enzyme": ans_enzyme}
PAIR: dict[str, Answer] = {
    "same_family": ans_same_family,
    "compare_location": ans_compare_location,
    "differences": ans_differences,
    "commonalities": ans_commonalities,
    "ec_depth": ans_ec_depth,
    "pathway_relation": ans_pathway_relation,
}
TRIPLE = {"odd_one_out_family": ans_odd_family, "odd_one_out_location": ans_odd_location}
