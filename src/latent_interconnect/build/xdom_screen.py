"""The machinery the cross-domain 2x2 builders share: the cells, the exposure screen, the quad draw.

Both builders (``build.bindingdb_items``, ``build.cross_domain_items`` for Rhea) cut positives (protein, molecule key)
into the cells of :data:`CELLS`, screen every edge with :class:`ExposureScreen` and draw swap quads per cell with
:func:`draw_quads`; each states its own label rules, molecule key (a BindingDB form, a ChEBI id) and pair
admissibility.

**Cells**. A protein is *familiar* when it is an object of the connector's protein training set, *novel* when
it is in the clean holdout (the selection's eligible pool, before its draw); a molecule key is *familiar* or *novel*
by its builder's rule. Each cell is named ``<protein side>P-<molecule side>M``.

**Exposure screen** (before quads are drawn, on every edge of a quad). A (protein, key) relation is *stated*
when

- the training text of the protein (if it is a training object), or of any training protein with the same name,
  mentions a name of the key (exact, word-bounded, case-insensitive: :class:`Mentions`), or holds a UniProt
  ``Reaction=`` equation whose ChEBI cross-references include a molecule of the key;
- or the key is familiar and the training text of ANY training object of the key mentions the protein (a non-generic
  protein name of five or more characters, the primary gene of four or more, case-exact, or the accession).

A stated positive is removed (counted by the first reason that holds, in that order: ``protein_text``,
``same_name_protein_text``, ``molecule_text``); a pair of positives whose swap is stated forms no quad.

**Quads.** Positives are selected in rounds (:func:`select`): each round is a maximum bipartite matching of protein
clusters to molecule groups over the positives not yet selected (one positive per edge: the first by protein, then
molecule), so each unit enters at most ``per_unit_cap`` positives. Within a round, positives are paired by a
maximum-cardinality matching over the admissible pairs the builder weighs; each edge's weight carries a seeded jitter
of 1e-3 (seed + round), so the matching is reproducible.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.data.training_text import TRAINING_SET_KIND
from latent_interconnect.domains.base import Domain
from latent_interconnect.domains.eval_proteins import GENERIC_PROTEIN_WORDS, split_protein_names

__all__ = [
    "CELLS",
    "MOLECULE",
    "PROTEIN",
    "Cell",
    "ExposureScreen",
    "Mentions",
    "draw_quads",
    "name_variants",
    "screen_positives",
    "select",
    "training_objects",
    "training_text",
]

CELLS = {"famP-famM": ("familiar", "familiar"), "famP-novM": ("familiar", "novel"),
         "novP-famM": ("novel", "familiar"), "novP-novM": ("novel", "novel")}  # fmt: skip
"""Cell name → (protein side, molecule side)."""
PROTEIN, MOLECULE = "protein", "molecule"
_JITTER = 1e-3  # the matching's tie-break: below any weight difference, so it only orders equal-cardinality choices
_MIN_TERM = 3
_CHARGE = re.compile(r"\((?:\d*[+-]|\d+[+-]?)\)$")
_TOKEN = re.compile(r"[A-Za-z0-9]+")
_ACCESSION = re.compile(
    r"(?<![A-Za-z0-9])([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?![A-Za-z0-9])"
)
_EQUATION = re.compile(r"Reaction=(.*?); Xref=([^;]*);")
_MIN_NAME, _MIN_GENE = 5, 4


class Mentions:
    """Exact, word-bounded mentions of terms in text (the xdom_leak scan rule), indexed by each term's first token.

    A term matches where it starts at a word start (no letter or digit before it) and ends at a word end; terms shorter
    than three characters are ignored. Case-insensitive unless ``case_sensitive``.
    """

    def __init__(self, terms: Iterable[str], case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive
        self.index: dict[str, dict[tuple[int, int], set[str]]] = defaultdict(lambda: defaultdict(set))
        for term in terms:
            key = term if case_sensitive else term.lower()
            first = _TOKEN.search(key)
            if len(key) >= _MIN_TERM and first is not None:
                self.index[first.group()][(first.start(), len(key))].add(key)

    def find(self, text: str) -> set[str]:
        """Return the terms (as indexed: lower-cased unless case-sensitive) mentioned in ``text``."""
        s = text if self.case_sensitive else text.lower()
        found: set[str] = set()
        for token in _TOKEN.finditer(s):
            for (offset, length), terms in self.index.get(token.group(), {}).items():
                i, j = token.start() - offset, token.start() - offset + length
                if i < 0 or (i and s[i - 1].isalnum()) or (j < len(s) and s[j].isalnum()):
                    continue
                if s[i:j] in terms:
                    found.add(s[i:j])
        return found


def name_variants(name: object) -> set[str]:
    """Return a molecule-domain name and, when it ends in a charge (``(1-)``, ``(2+)``), the name without it."""
    if not isinstance(name, str) or not name:
        return set()
    base = _CHARGE.sub("", name).strip()
    return {name} | ({base} if base != name and len(base) >= _MIN_TERM else set())


def training_text(store: ArtifactStore, digest: str, fields: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Return a training set's ``fields.parquet`` (``object_id``, ``field``, ``text``), all or the named fields."""
    frame = pd.read_parquet(store.path(store.ref(digest)) / "fields.parquet")
    return frame if fields is None else frame[frame["field"].isin(fields)]


def training_objects(store: ArtifactStore, digest: str) -> pd.Index:
    """Return a training set's objects.

    Raises:
        ValueError: if the artifact is not a training set.
    """
    ref = store.ref(digest)
    if ref.kind != TRAINING_SET_KIND:
        raise ValueError(f"{ref.short} is a {ref.kind!r} artifact, not a training set")
    return pd.Index(pd.read_parquet(store.path(ref) / "objects.parquet")["object_id"].astype(str))


class ExposureScreen:
    """The exposure screen's stated relations (module docstring) for the candidate pairs.

    Args:
        store: the store holding the training sets.
        protein: the protein domain (its ``name`` column is the same-name rule's name).
        familiar_proteins: the protein training set (a ``training_set`` digest).
        familiar_molecules: the molecule training set (a ``training_set`` digest).
        uniprot: Swiss-Prot's table (gzipped TSV: ``Entry``, ``Protein names``, ``Gene Names (primary)``,
            ``Catalytic activity``).
        candidates: the pairs to screen (``protein``, ``molecule``: the molecule key).
        names: key → its names.
        chebi_keys: ChEBI id → its key (a trained equation's cross-references name keys through it).
        trained_objects: familiar key → the molecule training objects of the key.

    Attributes:
        named: training protein → the keys its training text names.
        term_hits: name → the training-text fields that mention it.
        equations: keys named only through a trained equation.
        protein_names: familiar key → the proteins the training text of any of its objects names.
    """

    def __init__(
        self,
        store: ArtifactStore,
        *,
        protein: Domain,
        familiar_proteins: str,
        familiar_molecules: str,
        uniprot: Path,
        candidates: pd.DataFrame,
        names: Mapping[str, set[str]],
        chebi_keys: Mapping[str, str],
        trained_objects: Mapping[str, tuple[str, ...]],
    ) -> None:
        self.familiar_p = set(training_objects(store, familiar_proteins))
        pnames = protein.objects()["name"].fillna("").astype(str).str.lower()
        ptext = training_text(store, familiar_proteins)
        trained_names = ptext[ptext["field"] == "name"].assign(k=lambda d: d["text"].str.lower())
        want = {pnames.get(p, "") for p in candidates[PROTEIN]} - {""}
        same = trained_names[trained_names["k"].isin(want)]
        self.same_name: dict[str, tuple[str, ...]] = (
            same.groupby("k")["object_id"].agg(lambda v: tuple(sorted(v))).to_dict()
        )
        self.pname = pnames
        relevant = set(same["object_id"]) | (set(candidates[PROTEIN]) & self.familiar_p)
        ptext = ptext[ptext["object_id"].isin(relevant)]
        by_term: dict[str, set[str]] = defaultdict(set)
        for key, terms in names.items():
            for term in terms:
                by_term[term.lower()].add(key)
        scanner = Mentions(by_term)
        self.named: dict[str, set[str]] = defaultdict(set)
        self.term_hits: dict[str, int] = defaultdict(int)
        for oid, text in zip(ptext["object_id"], ptext["text"], strict=True):
            if isinstance(text, str):
                for term in scanner.find(text):
                    self.named[oid] |= by_term[term]
                    self.term_hits[term] += 1
        catalytic = ptext[ptext["field"] == "catalytic_activity"].set_index("object_id")["text"]
        table = pd.read_csv(
            uniprot,
            sep="\t",
            usecols=["Entry", "Protein names", "Gene Names (primary)", "Catalytic activity"],
            dtype=str,
            quoting=3,
        ).fillna("")
        self.equations = 0
        with_text = table[table["Entry"].isin(set(catalytic.index))]
        for acc, activity in zip(with_text["Entry"], with_text["Catalytic activity"], strict=True):
            trained = catalytic.get(acc)
            if not isinstance(trained, str):
                continue
            for equation, xref in _EQUATION.findall(activity):
                if equation.strip() and equation.strip() in trained:
                    ids = {x.split("ChEBI:")[1].strip() for x in xref.split(",") if "ChEBI:" in x}
                    hit = {chebi_keys[c] for c in ids if c in chebi_keys}
                    self.equations += len(hit - self.named[acc])
                    self.named[acc] |= hit
        self.protein_names = self._molecule_side(store, familiar_molecules, table, candidates, trained_objects)

    @staticmethod
    def _molecule_side(
        store: ArtifactStore,
        familiar_molecules: str,
        table: pd.DataFrame,
        candidates: pd.DataFrame,
        trained_objects: Mapping[str, tuple[str, ...]],
    ) -> dict[str, set[str]]:
        """Return familiar key → the proteins the training text of any of its training objects names."""
        familiar = candidates[candidates[MOLECULE].isin(set(trained_objects))]
        accs = set(familiar[PROTEIN])
        table = table[table["Entry"].isin(accs)]
        names: dict[str, set[str]] = defaultdict(set)
        genes: dict[str, set[str]] = defaultdict(set)
        for acc, field, gene_field in zip(
            table["Entry"], table["Protein names"], table["Gene Names (primary)"], strict=True
        ):
            gene = gene_field.split(";")[0].strip()
            if len(gene) >= _MIN_GENE:
                genes[gene].add(acc)
            for n in split_protein_names(field):
                generic = n == gene or len(n) < _MIN_NAME or n.lower() in GENERIC_PROTEIN_WORDS
                if not generic and re.search(r"[A-Za-z]{4}", n):
                    names[n.lower()].add(acc)
        name_scan, gene_scan = Mentions(names), Mentions(genes, case_sensitive=True)
        owners = {k: trained_objects[k] for k in sorted(set(familiar[MOLECULE]))}
        objects = {o for t in owners.values() for o in t}
        mtext = training_text(store, familiar_molecules)
        mtext = mtext[mtext["object_id"].isin(objects)]
        by_object: dict[str, set[str]] = defaultdict(set)
        for oid, text in zip(mtext["object_id"], mtext["text"], strict=True):
            if not isinstance(text, str):
                continue
            for t in name_scan.find(text):
                by_object[oid] |= names[t]
            for t in gene_scan.find(text):
                by_object[oid] |= genes[t]
            by_object[oid] |= {m.group(1) for m in _ACCESSION.finditer(text) if m.group(1) in accs}
        return {k: set().union(*(by_object.get(o, set()) for o in t)) for k, t in owners.items()}

    def stated(self, p: str, key: str) -> str | None:
        """Return why the relation (p, key) is stated (the first reason that holds), or None."""
        if p in self.familiar_p and key in self.named.get(p, ()):
            return "protein_text"
        for t in self.same_name.get(self.pname.get(p, "") or "\0", ()):
            if key in self.named.get(t, ()):
                return "same_name_protein_text"
        if p in self.protein_names.get(key, ()):
            return "molecule_text"
        return None

    def evidence(self) -> dict[str, Any]:
        """Return the screen's own counts (training proteins naming a key, equation hits, the 25 most hit names)."""
        return {
            "training_proteins_scanned_naming_a_form": sum(1 for v in self.named.values() if v),
            "catalytic_equation_hits": self.equations,
            "top_terms": dict(sorted(self.term_hits.items(), key=lambda kv: (-kv[1], kv[0]))[:25]),
        }


@dataclass
class Cell:
    """One cell's positives after the screen, its quads and its counts.

    Attributes:
        positives: the screened positives (``protein``, ``molecule``, ``cluster``, ``group``, the builder's columns),
            sorted by protein, molecule.
        quads: pairs of positive indices.
        evidence: the cell's counts (before and after the screen, removals, blocked pairings, quads, units).
    """

    positives: pd.DataFrame
    quads: list[tuple[int, int]]
    evidence: dict[str, Any]


def screen_positives(pos: pd.DataFrame, screen: ExposureScreen) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the positives no stated relation removes (sorted by protein, molecule) and the removals by reason."""
    reasons = [screen.stated(p, m) for p, m in zip(pos[PROTEIN], pos[MOLECULE], strict=True)]
    removed = pd.Series([r for r in reasons if r is not None], dtype=object).value_counts().to_dict()
    kept = pos[np.array([r is None for r in reasons], dtype=bool)].sort_values([PROTEIN, MOLECULE])
    return kept.reset_index(drop=True), {str(k): int(v) for k, v in removed.items()}


def select(pos: pd.DataFrame, cap: int) -> list[list[int]]:
    """Return the selection rounds (positive indices; ``cluster`` and ``group`` columns); each is unit-disjoint."""
    remaining = list(range(len(pos)))
    rounds: list[list[int]] = []
    for _ in range(cap):
        if not remaining:
            break
        sub = pos.iloc[remaining]
        clusters = sorted(set(sub["cluster"]))
        groups = sorted(set(sub["group"]))
        ci, gi = {c: n for n, c in enumerate(clusters)}, {g: n for n, g in enumerate(groups)}
        edge: dict[tuple[int, int], int] = {}
        for k, c, g in zip(remaining, sub["cluster"], sub["group"], strict=True):
            edge.setdefault((ci[c], gi[g]), k)  # the first positive by (protein, molecule) holds the edge
        r, c = zip(*edge, strict=True)
        graph = csr_matrix((np.ones(len(r)), (r, c)), shape=(len(clusters), len(groups)))
        match = maximum_bipartite_matching(graph, perm_type="column")
        chosen = sorted(edge[(row, int(col))] for row, col in enumerate(match) if col >= 0)
        if not chosen:
            break
        rounds.append(chosen)
        taken = set(chosen)
        remaining = [k for k in remaining if k not in taken]
    return rounds


def draw_quads(
    pos: pd.DataFrame, cap: int, seed: int, weigh: Callable[[list[int]], Iterable[tuple[int, int, float]]]
) -> list[tuple[int, int]]:
    """Return the quads as pairs of positive indices, round by round (module docstring).

    Args:
        pos: the positives (``cluster`` and ``group`` columns).
        cap: the most positives one unit enters.
        seed: the jitter's seed (round ``n`` draws from ``seed + n``).
        weigh: a round's members → its admissible pairs ``(a, b, weight)`` (positions in ``members``, ``a < b``, in
            ascending order: each draws its jitter in turn).
    """
    quads: list[tuple[int, int]] = []
    for n, members in enumerate(select(pos, cap)):
        rng = np.random.default_rng(seed + n)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(members)))
        for a, b, weight in weigh(members):
            graph.add_edge(a, b, weight=weight + _JITTER * rng.random())
        matching = nx.max_weight_matching(graph, maxcardinality=True)
        quads += sorted((members[min(e)], members[max(e)]) for e in matching)
    return quads
