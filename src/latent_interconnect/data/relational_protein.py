"""The protein relational answers as one bundle: the pool of protein records and :data:`PROTEIN`.

:class:`ProteinPool` indexes the training proteins for the pair and triplet draws of the original ``Pool`` /
``sample_pair`` / ``sample_triplet`` of the original protein relational generator.
:data:`PROTEIN` bundles it with the records (:mod:`data.relational_records`) and the answers
(:mod:`data.relational_answers`); the data package registers it as the ``protein`` relational answers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from latent_interconnect.data.relational_answers import (
    ANSWER_WORDING,
    FILL_KINDS,
    PAIR,
    SINGLE,
    TRIPLE,
    VERDICT_SENTENCE,
    VERDICT_WORDS,
    coarse,
    fam_key,
    fam_name,
    top_compartments,
)
from latent_interconnect.data.relational_base import RelationalAnswers, WeightedPool
from latent_interconnect.data.relational_records import Protein, protein_record

__all__ = ["FACTS", "PROTEIN", "ProteinPool"]

FACTS = ("family", "location", "function", "pathway", "ec", "catalytic_activity")
"""The protein facts the port supports (the original runs' trained relational fields)."""


def _superfamily(p: Protein) -> str | None:
    return p.family.superfamily if p.family else None


def _locs(p: Protein) -> tuple[str, ...]:
    return p.location.locs if p.location else ()


class ProteinPool(WeightedPool):
    """The training proteins indexed for pair and triplet draws (the original ``Pool``), with weighted protein draws.

    A draw within a list is weighted by each protein's weight (:class:`~latent_interconnect.data.relational_base.
    WeightedPool`); the choice of group (family, location, EC class, pathway) is uniform.
    """

    def __init__(self, proteins: Mapping[str, Protein], weight: Mapping[str, float], p_hard: float) -> None:
        super().__init__(weight)
        self.P, self.p_hard = proteins, p_hard
        self.acc = list(proteins)
        idx: dict[str, defaultdict[str, list[str]]] = {
            k: defaultdict(list) for k in ("fam", "sup", "loc", "ec1", "pw", "cat")
        }
        for a, p in proteins.items():
            if p.family:
                idx["fam"][fam_key(p)].append(a)
                if p.family.superfamily:
                    idx["sup"][p.family.superfamily].append(a)
            if p.location:
                idx["loc"][p.location.locs[0]].append(a)
            if p.ec:
                idx["ec1"][p.ec[0]].append(a)
            if p.pathway:
                idx["pw"][p.pathway.pathway].append(a)
                idx["cat"][p.pathway.category].append(a)
        self.by = idx
        self.fam2 = [k for k, v in idx["fam"].items() if len(v) >= 2]  # noqa: PLR2004  (a pair)
        self.loc2 = [k for k, v in idx["loc"].items() if len(v) >= 2]  # noqa: PLR2004
        self.pw2 = [k for k, v in idx["pw"].items() if len(v) >= 2]  # noqa: PLR2004
        self.ec2 = [k for k, v in idx["ec1"].items() if len(v) >= 2]  # noqa: PLR2004
        sup_fams = {s: {fam_name(proteins[x]) for x in v} for s, v in idx["sup"].items()}
        self.fam2_hard = []
        for k in self.fam2:
            sup = _superfamily(proteins[idx["fam"][k][0]])
            if sup and len(sup_fams[sup]) >= 2:  # noqa: PLR2004  (its superfamily holds another family)
                self.fam2_hard.append(k)
        self.by_coarse: defaultdict[str, list[str]] = defaultdict(list)
        for a, p in proteins.items():
            if p.location:
                self.by_coarse[coarse(p.location.locs[0])].append(a)
        self.pwp = [a for v in idx["pw"].values() for a in v]
        self.enz = [a for v in idx["ec1"].values() for a in v]

    def pair(self, g: np.random.Generator, family: str) -> tuple[str, str]:  # noqa: PLR0911  (the original branches)
        """~50 % related pairs, ~25 % hard negatives, ~25 % random (the original ``sample_pair``)."""
        u, P = g.random(), self.P
        if family in ("same_family", "commonalities", "differences") and u < 0.5 and self.fam2:  # noqa: PLR2004
            return self.pick2(g, self.by["fam"][self.fam2[int(g.integers(len(self.fam2)))]])
        if family == "pathway_relation":
            if u < 0.5 and self.pw2:  # noqa: PLR2004
                return self.pick2(g, self.by["pw"][self.pw2[int(g.integers(len(self.pw2)))]])
            a = self.pick(g, self.pwp)
            pa = P[a].pathway
            if u < 0.75 and pa is not None:  # noqa: PLR2004  (same category, another pathway)
                same = [
                    x
                    for x in self.by["cat"][pa.category]
                    if (q := P[x].pathway) is not None and q.pathway != pa.pathway
                ]
                if same:
                    return a, self.pick(g, same, fixed=False)
            return a, self.pick(g, self.pwp)
        if family == "compare_location" and u < 0.5 and self.loc2:  # noqa: PLR2004
            return self.pick2(g, self.by["loc"][self.loc2[int(g.integers(len(self.loc2)))]])
        if family == "ec_depth":
            if u < 0.6 and self.ec2:  # noqa: PLR2004
                return self.pick2(g, self.by["ec1"][self.ec2[int(g.integers(len(self.ec2)))]])
            return self.pick2(g, self.enz)
        if u < 0.75:  # noqa: PLR2004  (a hard negative: a shared superfamily or EC class, another family)
            a = self.pick(g, self.acc)
            p = P[a]
            cand: list[str] = []
            sup = _superfamily(p)
            if sup:
                cand = [x for x in self.by["sup"][sup] if fam_name(P[x]) != fam_name(p)]
            if not cand and p.ec and p.family:
                cand = [x for x in self.by["ec1"][p.ec[0]] if P[x].family and fam_name(P[x]) != fam_name(p)][:5000]
            if cand:
                return a, self.pick(g, cand, fixed=False)
        return self.pick2(g, self.acc)

    def triplet(self, g: np.random.Generator, family: str) -> list[str] | None:
        """``[a, b, odd]`` (the original ``sample_triplet``), or None."""
        hard = g.random() < self.p_hard
        P = self.P
        if family == "odd_one_out_family" and self.fam2:
            fams = self.fam2_hard if hard and self.fam2_hard else self.fam2
            a, b = self.pick2(g, self.by["fam"][fams[int(g.integers(len(fams)))]])
            name = fam_name(P[a])
            if hard and self.fam2_hard:
                cand = [x for x in self.by["sup"][_superfamily(P[a]) or ""] if fam_name(P[x]) != name]
                return [a, b, self.pick(g, cand, fixed=False)]
            c = next((x for x in self.pickn(g, self.acc, 50) if P[x].family and fam_name(P[x]) != name), None)
            return [a, b, c] if c else None
        if family == "odd_one_out_location" and self.loc2:
            k = self.loc2[int(g.integers(len(self.loc2)))]
            a, b = self.pick2(g, self.by["loc"][k])
            locs_ab = set(_locs(P[a])) | set(_locs(P[b]))
            top_ab = top_compartments(P[a]) | top_compartments(P[b])

            def fits(x: str) -> bool:
                locs = _locs(P[x])
                return bool(locs) and k not in locs and not set(locs) & locs_ab

            if hard:
                grp = self.by_coarse.get(coarse(k), [])
                c = next((x for x in self.pickn(g, grp, 400) if fits(x) and not top_compartments(P[x]) & top_ab), None)
                if c:
                    return [a, b, c]
            c = next((x for x in self.pickn(g, self.acc, 400) if fits(x)), None)
            return [a, b, c] if c else None
        return None


def _protein(object_id: str, fields: Mapping[str, str], names: Mapping[str, str], lists: Mapping[str, str]) -> Protein:
    """A protein's record (no protein fact is an item list, so ``lists`` is empty)."""
    del lists
    return protein_record(object_id, fields, names)


PROTEIN = RelationalAnswers(
    noun="protein",
    facts=FACTS,
    requires={},
    lists=frozenset(),
    fill_kinds=FILL_KINDS,
    single=SINGLE,
    pair=PAIR,
    triple=TRIPLE,
    verdict_words=VERDICT_WORDS,
    verdict_sentence=VERDICT_SENTENCE,
    wording=ANSWER_WORDING,
    record=_protein,
    pool=ProteinPool,
    yes_no=frozenset({"same_family"}),
)
"""The protein relational answers: the pool, the records and the answers."""
