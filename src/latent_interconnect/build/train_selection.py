"""Training-object selection: grow an existing training pool to a target size, cluster by cluster.

:func:`select_train_objects` applies one :class:`TrainSelectionSpec`: a fill (the five steps below), or, with ``base``,
the subset of an earlier selection (the last section).

1. **Pool.** The union of the ids the pool sources list. Ids that are not train-split proteins of the domain are
   dropped and counted (``pool_outside_train``); the rest are kept whole.
2. **Candidates.** The train-split proteins (``Domain.population``) outside the pool, no longer than ``max_length``
   (when set), with every ``required_fields`` field non-empty.
3. **Exclusions**, each counted under its reason:
   - :class:`HomologyExclusion`: a protein that is the target of a hit of a homology search (``build.homology``) that
     passes any of its :class:`HitRule` thresholds, from any query or from the listed ones only. The search must have
     searched this domain's train split as its targets, so no candidate escapes unsearched;
   - :class:`IdExclusion`: the listed ids;
   - :class:`LengthExclusion`: the proteins longer than its residue count.

   An exclusion removes candidates only: the pool is never screened here (a :class:`ShortExclusion`, the minimum
   length, is therefore a base's subset's alone).
4. **Cluster fill.** Candidates are ranked within their cluster, richest first (:class:`Richness`; ties by a seeded
   key per protein, :func:`tie_key`), and a candidate's *level* is the number of pool members in its cluster plus its
   rank. Candidates are taken by level, then richness, then the key, until the selection holds ``target_size``
   proteins. So every cluster new to the pool gets its first protein before any cluster gets another, and clusters
   fill one level at a time.
5. **Output.** The ``train_ids`` object list: the selected proteins (pool and added, by id), then, with
   ``with_derived``, every train-split derived object of the domain (outside ``Domain.population``: the protein
   variants), by id. Its ``origin`` field says ``pool``, ``added`` or ``derived``. A second object list, ``added``,
   holds the added proteins alone (the objects that still need embeddings). A third, ``parents``, holds the derived
   objects' parents outside the selection, by id: the variant stream's relational rows bind them, so the runs
   need their embeddings, but they are not training objects. With ``drop_excluded_parents``,
   a derived object whose parent any exclusion names is left out (counted, ``derived_dropped``), whatever
   the parent's origin: a pool protein (the pool is never screened) or one outside the selection. Its parent is then
   no longer bound, so it leaves ``parents`` too. The pool proteins themselves stay.

**Evidence.** The counts of every step, the cluster statistics of the pool and of the selection (:class:`ClusterStats`)
and the parameters of every search an exclusion reads are the :class:`SelectionEvidence`. It is stored in each
artifact's metadata. A spec may state it (``evidence``, rationale: not identity); the selection then refuses to commit
a result whose evidence differs (:class:`EvidenceMismatchError`), as the zero-shot screen's detector evidence is held
to its scopes (``build.zero_shot.DetectorSpec``).

**A base's subset** (``base``: objects leave new training with no refill). The selection is an earlier
selection's ``train_ids`` list, in its order and with its origins, less

- every object an exclusion names, whatever its origin: a pool protein leaves as an added one does (a
  :class:`ShortExclusion` and a :class:`LengthExclusion` name proteins by their residues, an :class:`IdExclusion` any
  listed object, a derived one too);
- every derived object whose parent an exclusion names (a variant follows its parent), whether the parent was in the
  base or outside it.

Nothing is added: no candidate is read, no cluster is filled and no target size is stated, so the selection is the
base minus those objects and nothing else. The ``added`` list is the selection's objects of origin ``added`` and the
``parents`` list the remaining derived objects' parents outside the selection. The domain may be a later data version
than the base's; a base object that it does not hold in its train split is refused. The ``train_ids`` artifact's
metadata holds ``removed``: each object of the base that left, with the reasons that name it (an exclusion's
``reason``, or ``parent:<reason>`` for a derived object whose parent it names), so base = selection + ``removed``.
The evidence is a :class:`SubsetEvidence`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.build.homology import HOMOLOGY_KIND, DomainSequences, HomologySearchSpec, load_hits, sequences
from latent_interconnect.build.selection_evidence import (
    ClusterStats,
    EvidenceMismatchError,
    ExclusionEvidence,
    SearchEvidence,
    SelectionEvidence,
    SubsetEvidence,
    cluster_stats,
)
from latent_interconnect.build.selection_spec import (
    PARENT_REASON,
    ElementGroup,
    HitRule,
    HomologyExclusion,
    IdExclusion,
    IdSource,
    LengthExclusion,
    Richness,
    ShortExclusion,
    TrainSelectionSpec,
)
from latent_interconnect.config import Config
from latent_interconnect.data.items import (
    ITEM_SET_KIND,
    OBJECT_ROLE,
    ItemSet,
    commit_item_set,
    load_item_set,
    object_list,
)
from latent_interconnect.domains.base import DomainSpec, open_domain
from latent_interconnect.domains.protein import ProteinDomain

__all__ = [
    "ORIGINS",
    "PARENT_REASON",
    "ClusterStats",
    "ElementGroup",
    "EvidenceMismatchError",
    "ExclusionEvidence",
    "HitRule",
    "HomologyExclusion",
    "IdExclusion",
    "IdSource",
    "LengthExclusion",
    "Richness",
    "SearchEvidence",
    "Selection",
    "SelectionEvidence",
    "ShortExclusion",
    "SubsetEvidence",
    "TrainSelectionSpec",
    "cluster_stats",
    "commit_selection",
    "passing_hits",
    "present",
    "select_train_objects",
    "tie_key",
]

ORIGINS = ("pool", "added", "derived")
"""The values of the ``origin`` field of a ``train_ids`` list."""


class Selection(Config):
    """A selection's result before it is committed.

    Attributes:
        train_ids: the selected objects, in list order.
        origin: each object's origin (:data:`ORIGINS`).
        added: the added proteins, in the order they were taken (a base's subset: in list order).
        parents: the derived objects' parents outside the selection, by id.
        evidence: the counts.
        removed: a base's subset only: each object of the base that left → the reasons that name it, by id.
    """

    train_ids: tuple[str, ...]
    origin: tuple[str, ...]
    added: tuple[str, ...]
    parents: tuple[str, ...]
    evidence: SelectionEvidence | SubsetEvidence
    removed: dict[str, tuple[str, ...]] = Field(default_factory=dict)


def tie_key(seed: int, object_id: str) -> float:
    """Return an object's tie-breaking key in [0, 1): the first 8 bytes of sha256(``"<seed>:<object_id>"``).

    A key depends on the seed and the object alone, not on which other objects compete, so a change of the candidate
    set (one more exclusion, say) moves only the objects it touches.
    """
    digest = hashlib.sha256(f"{seed}:{object_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def select_train_objects(spec: TrainSelectionSpec, store: ArtifactStore, data_root: Path) -> Selection:
    """Apply the selection rule (module docstring) and check its evidence against the spec's.

    Raises:
        IntegrityError: if a cited file's sha256 differs from its citation, or an artifact from its digest.
        FileNotFoundError: if a cited file or artifact is missing.
        ValueError: for an unknown field, a train protein without a cluster, a search that did not search the train
            split or lacks a listed query, or a target that the pool already exceeds or the candidates cannot reach;
            with ``base``: a base that is no ``train_ids`` list of this domain, a base object outside the domain's
            train split, or exclusions that leave no protein.
        EvidenceMismatchError: if ``spec.evidence`` is stated and differs.
    """
    if spec.base is not None:
        return _held(spec, _subset(spec, spec.base, store, data_root))
    target = _target(spec)
    domain = open_domain(spec.domain, store)
    objects, fields, split, clusters = domain.objects(), domain.fields(), domain.split(), domain.clusters()
    known = set(fields.columns)
    unknown = sorted({*spec.required_fields, *spec.richness.fields, *(g.field for g in spec.richness.groups)} - known)
    if unknown:
        raise ValueError(f"unknown fields {unknown}; the domain has {sorted(known)}")
    proteins = domain.population()
    train = proteins[(split.reindex(proteins) == "train").to_numpy()]
    unclustered = train[clusters.reindex(train).isna().to_numpy()]
    if len(unclustered):
        raise ValueError(f"{len(unclustered)} train proteins have no cluster, e.g. {list(unclustered[:3])}")
    listed = set().union(*(s.read(data_root) for s in spec.pool))
    pool = train[train.isin(listed)]
    if len(pool) > target:
        raise ValueError(f"the pool holds {len(pool)} proteins, more than the target {target}")
    length = objects["value"].str.len()
    outside = train[~train.isin(pool)]
    short = outside if spec.max_length is None else outside[(length.reindex(outside) <= spec.max_length).to_numpy()]
    complete = np.ones(len(short), dtype=bool)
    for f in spec.required_fields:
        complete &= present(fields[f].reindex(short)).to_numpy()
    candidates = short[complete]
    exclusions, removed, barred = {}, pd.Index([]), set[str]()
    for ex in spec.exclusions:
        named, search = _named(
            ex, length.reindex(proteins), set(objects.index), spec.domain, store, data_root=data_root
        )
        hit = candidates[candidates.isin(named)]
        exclusions[ex.reason] = ExclusionEvidence(train=int(train.isin(named).sum()), removed=len(hit), search=search)
        removed, barred = removed.union(hit), barred | named
    eligible = candidates.difference(removed).sort_values()
    added, level = _fill(spec, clusters, pool, eligible, fields)
    chosen = pool.union(pd.Index(added)).sort_values()
    derived, kept, parents, dropped = pd.Index([]), pd.Index([]), pd.Index([]), None
    if spec.with_derived:
        others = objects.index.difference(proteins)
        kept = derived = others[(split.reindex(others) == "train").to_numpy()].sort_values()
        if isinstance(domain, ProteinDomain):
            parent = domain.variants()["parent"].astype(str).reindex(derived)
            if spec.drop_excluded_parents:  # variants of excluded parents leave
                parent = parent[~parent.isin(barred).to_numpy()]
                kept, dropped = pd.Index(parent.index), len(derived) - len(parent)
            parents = pd.Index(parent.unique()).difference(chosen).sort_values()
    evidence = SelectionEvidence(
        pool_listed=len(listed),
        pool_outside_train=len(listed) - len(pool),
        pool=len(pool),
        train_proteins=len(train),
        candidates=len(outside),
        over_max_length=len(outside) - len(short),
        missing_required=len(short) - len(candidates),
        exclusions=exclusions,
        eligible=len(eligible),
        added=len(added),
        level_reached=level,
        new_clusters=len(set(clusters.reindex(chosen)) - set(clusters.reindex(pool))),
        new_residues=int(length.reindex(pd.Index(added)).sum()),
        derived=len(derived),
        derived_dropped=dropped,
        pool_stats=cluster_stats(clusters.reindex(pool)),
        selected_stats=cluster_stats(clusters.reindex(chosen)),
    )
    in_pool = set(pool)
    origin = ["pool" if o in in_pool else "added" for o in chosen] + ["derived"] * len(kept)
    return _held(
        spec,
        Selection(
            train_ids=(*chosen, *kept),
            origin=tuple(origin),
            added=tuple(added),
            parents=tuple(parents),
            evidence=evidence,
        ),
    )


def _held(spec: TrainSelectionSpec, selection: Selection) -> Selection:
    """Return ``selection``, its evidence checked against the spec's.

    Raises:
        EvidenceMismatchError: if ``spec.evidence`` is stated and differs.
    """
    if spec.evidence is not None and spec.evidence != selection.evidence:
        stated, found = spec.evidence.model_dump(), selection.evidence.model_dump()
        diff = {k: (stated.get(k), found[k]) for k in found if stated.get(k) != found[k]}
        raise EvidenceMismatchError(f"selection {spec.name!r}: evidence differs (stated, found): {diff}")
    return selection


def _target(spec: TrainSelectionSpec) -> int:
    """Return a fill's target size.

    Raises:
        ValueError: if the spec states none (a base's subset has none).
    """
    if spec.target_size is None:
        raise ValueError(f"selection {spec.name!r} states no target size")
    return spec.target_size


def _subset(spec: TrainSelectionSpec, base: str, store: ArtifactStore, data_root: Path) -> Selection:
    """Return the ``base`` list less what the exclusions name (module docstring, "A base's subset").

    Raises:
        ValueError: as :func:`select_train_objects` for a base.
    """
    domain = open_domain(spec.domain, store)
    objects, split = domain.objects(), domain.split()
    listed = load_item_set(store, ArtifactRef(base, ITEM_SET_KIND))
    if dict(listed.roles) != {OBJECT_ROLE: spec.domain.name} or "origin" not in listed.fields:
        raise ValueError(f"base {base[:12]} ({listed.name!r}) is no train_ids list of {spec.domain.name!r}")
    ids = pd.Index(listed.object_ids)
    origin = pd.Series([str(o) for o in listed.fields["origin"]], index=ids)
    outside = ids[(split.reindex(ids) != "train").to_numpy()]
    if len(outside):
        raise ValueError(
            f"selection {spec.name!r}: {len(outside)} objects of the base are outside the domain's train split, "
            f"e.g. {list(outside[:3])}"
        )
    derived = ids[(origin == "derived").to_numpy()]
    if len(derived) and not isinstance(domain, ProteinDomain):
        raise ValueError(f"derived objects are protein variants; {spec.domain.name!r} has none")
    parent = (
        domain.variants()["parent"].astype(str).reindex(derived)
        if isinstance(domain, ProteinDomain)
        else pd.Series([], index=derived, dtype=str)
    )
    proteins = domain.population()
    train = proteins[(split.reindex(proteins) == "train").to_numpy()]
    length = objects["value"].str.len().reindex(proteins)
    exclusions, reasons, followed = {}, dict[str, list[str]](), set[str]()
    named, counts = set[str](), dict[str, int]()
    for ex in spec.exclusions:
        names, search = _named(ex, length, set(objects.index), spec.domain, store, data_root=data_root)
        own = ids[ids.isin(names)]
        counts[ex.reason] = len(names)
        exclusions[ex.reason] = ExclusionEvidence(train=int(train.isin(names).sum()), removed=len(own), search=search)
        for o in own:
            reasons.setdefault(str(o), []).append(ex.reason)
        for o in parent.index[parent.isin(names).to_numpy()]:
            reasons.setdefault(str(o), []).append(PARENT_REASON + ex.reason)
            followed.add(str(o))
        named |= set(own)
    stays = ~ids.isin(list(reasons))
    kept, left = ids[stays], ids[~stays]
    chosen = kept[(origin[stays] != "derived").to_numpy()]
    if not len(chosen):
        raise ValueError(f"selection {spec.name!r}: the exclusions leave no protein of the base")
    parents = pd.Index(parent.reindex(kept[(origin[stays] == "derived").to_numpy()]).unique()).difference(kept)
    evidence = SubsetEvidence(
        base=_by_origin(origin),
        listed=counts,
        exclusions=exclusions,
        removed=_by_origin(origin[~stays]),
        followed_parent=len(followed - named),
        selected=_by_origin(origin[stays]),
        parents=len(parents),
        selected_stats=cluster_stats(domain.clusters().reindex(chosen)),
    )
    return Selection(
        train_ids=tuple(kept),
        origin=tuple(origin[stays]),
        added=tuple(kept[(origin[stays] == "added").to_numpy()]),
        parents=tuple(parents.sort_values()),
        evidence=evidence,
        removed={str(o): tuple(reasons[str(o)]) for o in left.sort_values()},
    )


def _by_origin(origin: pd.Series) -> dict[str, int]:
    """Return origin → how many objects have it, in :data:`ORIGINS` order (an origin with none has no key)."""
    counts = origin.value_counts()
    return {o: int(counts[o]) for o in ORIGINS if o in counts.index}


def commit_selection(
    spec: TrainSelectionSpec, selection: Selection, store: ArtifactStore
) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef]:
    """Commit the ``train_ids``, ``added`` and ``parents`` lists, each with the spec and evidence in its metadata.

    A fill's stored spec leaves ``base`` out (a field added after selections were stored: the artifacts of an older
    selection are unchanged). A base's subset stores ``removed`` (object → reasons) with its ``train_ids`` list.
    """
    domain = open_domain(spec.domain, store)
    stored = spec.model_dump(mode="json", exclude={"base"} if spec.base is None else None)
    meta: dict[str, Any] = {"spec": stored, "spec_digest": spec.digest()}
    meta["evidence"] = selection.evidence.model_dump(mode="json")
    base = object_list(spec.name, domain, selection.train_ids)
    train_ids = ItemSet(base.name, base.roles, base.objects, {**base.fields, "origin": selection.origin})
    added = object_list(f"{spec.name}/added", domain, selection.added)
    parents = object_list(f"{spec.name}/parents", domain, selection.parents)
    removed = {} if spec.base is None else {"removed": {o: list(r) for o, r in selection.removed.items()}}
    return (
        commit_item_set(store, train_ids, {**meta, **removed}),
        commit_item_set(store, added, meta),
        commit_item_set(store, parents, meta),
    )


def passing_hits(hits: pd.DataFrame, rules: Sequence[HitRule]) -> pd.DataFrame:
    """Return the hits that pass any of ``rules`` (:class:`HitRule`: identity, and the coverage of both sequences)."""
    passing = np.zeros(len(hits), dtype=bool)
    for r in rules:
        cover = (hits["qcov"] >= r.min_coverage) & (hits["tcov"] >= r.min_coverage)
        passing |= ((hits["fident"] >= r.min_identity) & cover).to_numpy()
    return hits[passing]


def present(values: pd.Series) -> pd.Series:
    """Return which cells hold a value: a non-blank text or a non-empty item list."""

    def has(v: object) -> bool:
        if isinstance(v, str):
            return bool(v.strip())
        if isinstance(v, list | tuple | np.ndarray):
            return len(v) > 0
        return v is not None and not pd.isna(v)

    return values.map(has).astype(bool)


def _richness(spec: Richness, fields: pd.DataFrame, ids: pd.Index) -> pd.Series:
    rich = pd.Series(0, index=ids, dtype=np.int64)
    for f in spec.fields:
        rich += present(fields[f].reindex(ids)).astype(np.int64)
    for g in spec.groups:
        rich += fields[g.field].reindex(ids).map(_meets(g)).astype(np.int64)
    return rich


def _meets(group: ElementGroup) -> Callable[[object], bool]:
    """Return the test of one cell: a text whose elements (split at ``group.separator``) include a wanted one."""
    wanted = set(group.elements)

    def meets(value: object) -> bool:
        return isinstance(value, str) and any(e.strip() in wanted for e in value.split(group.separator))

    return meets


def _fill(
    spec: TrainSelectionSpec, clusters: pd.Series, pool: pd.Index, eligible: pd.Index, fields: pd.DataFrame
) -> tuple[list[str], int]:
    """Return the added proteins in the order taken, and the highest level among them (-1 if none)."""
    target = _target(spec)
    need = target - len(pool)
    if need > len(eligible):
        raise ValueError(f"the target {target} needs {need} candidates; {len(eligible)} are eligible")
    have = clusters.reindex(pool).value_counts()
    c = pd.DataFrame(
        {
            "cluster": clusters.reindex(eligible).to_numpy(),
            "rich": _richness(spec.richness, fields, eligible).to_numpy(),
            "key": [tie_key(spec.seed, str(o)) for o in eligible],
        },
        index=eligible,
    )
    c = c.sort_values(["cluster", "rich", "key"], ascending=[True, False, True], kind="stable")
    c["level"] = c.groupby("cluster").cumcount().to_numpy() + c["cluster"].map(have).fillna(0).astype(np.int64)
    taken = c.sort_values(["level", "rich", "key"], ascending=[True, False, True], kind="stable").iloc[:need]
    return [str(o) for o in taken.index], int(taken["level"].max()) if need else -1


def _named(
    ex: HomologyExclusion | IdExclusion | LengthExclusion | ShortExclusion,
    length: pd.Series,
    known: set[str],
    spec: DomainSpec,
    store: ArtifactStore,
    *,
    data_root: Path,
) -> tuple[set[str], SearchEvidence | None]:
    """Return the ids an exclusion names, and the search it read; ``length`` is each protein's residue count."""
    if isinstance(ex, LengthExclusion):
        return set(length.index[(length > ex.longer_than).to_numpy()]), None
    if isinstance(ex, ShortExclusion):
        return set(length.index[(length < ex.shorter_than).to_numpy()]), None
    return _excluded(ex, known, spec, store, data_root)


def _excluded(
    ex: HomologyExclusion | IdExclusion, known: set[str], spec: DomainSpec, store: ArtifactStore, data_root: Path
) -> tuple[set[str], SearchEvidence | None]:
    """Return the ids an exclusion names, and the search it read."""
    if isinstance(ex, IdExclusion):
        return set().union(*(s.read(data_root) for s in ex.ids)), None
    ref = store.ref(ex.search)
    if ref.kind != HOMOLOGY_KIND:
        raise ValueError(f"exclusion {ex.reason!r}: {ref.short} is a {ref.kind!r} artifact, not a homology search")
    meta = store.manifest(ref).meta
    search = HomologySearchSpec.model_validate(meta["spec"])
    targets = search.targets
    if not (isinstance(targets, DomainSequences) and targets.domain.data == spec.data and "train" in targets.splits):
        raise ValueError(f"exclusion {ex.reason!r}: search {search.name!r} did not search this domain's train split")
    hits = load_hits(store, ref)
    if ex.queries:
        wanted = set().union(*(s.read(data_root) for s in ex.queries))
        searched = set(sequences(search.queries, store, data_root).index)
        if wanted - searched:
            raise ValueError(
                f"exclusion {ex.reason!r}: {len(wanted - searched)} listed queries were not searched, "
                f"e.g. {sorted(wanted - searched)[:3]}"
            )
        hits = hits[hits["query"].isin(wanted)]
    named = set(passing_hits(hits, ex.rules)["target"])
    stray = named - known
    if stray:
        raise ValueError(f"exclusion {ex.reason!r}: targets outside the domain, e.g. {sorted(stray)[:3]}")
    evidence = SearchEvidence(
        name=search.name,
        mmseqs_version=meta["mmseqs_version"],
        arguments=tuple(meta["arguments"]),
        both_directions=search.both_directions,
        queries=meta["queries"]["n"],
        targets=meta["targets"]["n"],
        hits=meta["hits"],
    )
    return named, evidence
