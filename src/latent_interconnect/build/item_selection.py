"""Item selection: an object list drawn as whole clusters from a domain's splits (the ``items`` stage).

:func:`select_items` applies one :class:`ItemSelectionSpec`:

1. **Candidates.** The domain's population (``Domain.population``: no derived objects) in one of ``splits``, no
   shorter than ``min_length`` and no longer than ``max_length`` (when set), with every ``required_fields`` field
   non-empty, with a ``where`` value (when set: a column of one of the domain's extra tables, e.g. the molecule
   split's stratum), and listed in ``within`` (when set: an item set's objects, or the objects an embeddings
   artifact holds, e.g. a capture's: its screened coverage), or in any of ``within_any`` (the union of their objects:
   a population made of several sets). With ``derived``, the candidates are instead the derived objects (the protein
   variants) in ``splits`` and the bound whose parent is such a candidate; a variant has its parent's cluster. Last,
   a candidate that an ``excluded_ids`` list names (ids listed with a reason),
   or, with ``derived``, whose parent one names, is dropped, and so is a candidate listed in ``without`` (item sets
   or embeddings artifacts, e.g. the objects existing extractions hold), object by object, where ``exclude`` drops
   whole clusters; and, with ``training_text``, a candidate that lacks one of its fields in a screened training
   set's text (``data.training_text``: the zero-shot screen removed it, or the object is not in the set), so that a
   text a suite scores is one the training set may train on (the monitor scores training proteins with
   training-eligible captions).
2. **Exclusions.** A candidate whose cluster holds an object of an ``exclude`` item set is dropped, so the selection
   shares no cluster with those sets (the test populations come from the held-out pool outside the
   monitoring set's clusters). Then a candidate with a homolog among the objects of a ``homologs`` screen's item sets
   is dropped (:class:`HomologScreen`: no rule-E homolog in the training set). With ``shared_names``, last, a
   candidate is kept only if its name is held by the remaining candidates of at least that many clusters (for
   Fig 10b's held-out proteins sharing a name across clusters).
   "Cluster" is the domain's cluster (``Domain.clusters``) or, with ``units``, a column of one of its extra tables
   (e.g. the molecule split's group, its independence unit); every step below reads the same unit.
3. **Whole clusters.** Clusters are visited in the order of a seeded key per cluster
   (``build.train_selection.tie_key`` on the cluster id). A cluster's candidates are taken together if they fit in
   what ``size`` leaves and skipped otherwise, until the selection holds ``size`` objects. With ``per_cluster``, a
   cluster contributes at most that many candidates, the first by a seeded key per object (``tie_key`` on the object
   id), so the selection spans more clusters (one per cluster: ``size`` independent units).
   With ``min_positives`` (one per cluster only), a declared stratified top-up comes first: task by task, the
   task with the fewest positives among the eligible candidates first (ties by id), clusters are visited in the seeded
   order and an unused cluster gives its first positive of the task (by the object key) until the selection holds the
   task's count of positives (in the task's scope, label > 0, as an AUROC reads it). The rest is filled as above from
   the unused clusters. Labels only: nothing scored is read.
   With ``size: all`` (a subset of the ``within`` set; objects leave an evaluated set with no redraw) there is
   no draw: the selection is every eligible candidate, as it is. Nothing is topped up and no name is paired again:
   ``min_positives`` and ``shared_names`` are then measured on the selection and reported (the evidence's
   ``below_min_positives``: the tasks under their count; ``unpaired_names``: the objects whose name the selection
   holds in fewer clusters), and the objects stay.
4. **Output.** The ``item_set`` object list, by id, with its clusters (``data.items.object_list``; with ``units``, the
   units the selection read) and, with ``names``, each object's name as ``name.object`` (``data.items.with_names``).
   With ``pool_clean``, each screen marks instead of dropping: the field ``pool_clean.<reason>`` is True for an object
   with no passing hit to an object of the screen's sets (for PFMbench, whose proteins are another domain than
   the training set's: the paper's pool-clean subsets, no MMseqs2 30 % / 80 % hit either way).

The counts of every step and the selection's cluster statistics (``build.train_selection.cluster_stats``) are the
evidence, stored in the artifact's metadata with the spec. With ``size: all`` a selected object's unit must be the
one the ``within`` item set stores for it (the old selection's ``units`` are stated again, or the subset is refused).
With ``excluded_ids`` or ``size: all`` the metadata also holds ``removed``, object → reasons: every candidate an
``excluded_ids`` list took (its ``reason``, or ``parent:<reason>`` for a derived object whose parent it names) and,
with ``size: all``, every other object of the ``within`` set that is not selected (:data:`MIN_LENGTH_REASON`, or
:data:`OTHER_REASON` for any other rule of the spec), each with every reason that applies. So the ``within`` set =
the selection + ``removed`` (with ``derived``: the set's derived objects; a variant is as long as its parent).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.build.homology import (
    HOMOLOGY_KIND,
    DomainSequences,
    HomologySearchSpec,
    TableSequences,
    load_hits,
    sequences,
)
from latent_interconnect.build.selection_spec import PARENT_REASON, HitRule, IdExclusion
from latent_interconnect.build.train_selection import cluster_stats, passing_hits, present, tie_key
from latent_interconnect.config import ArtifactDigest, Config, StageConfig, added_field
from latent_interconnect.data.items import (
    ITEM_SET_KIND,
    OBJECT_ROLE,
    ItemSet,
    cluster_field,
    commit_item_set,
    load_item_set,
    object_list,
    with_names,
)
from latent_interconnect.data.training_text import load_training_text
from latent_interconnect.domains.base import Domain, DomainSpec, open_domain
from latent_interconnect.domains.protein import ProteinDomain
from latent_interconnect.encoders.store import EMBEDDINGS_KIND, EmbeddingReader

__all__ = [
    "MIN_LENGTH_REASON",
    "OTHER_REASON",
    "ColumnIn",
    "EligiblePool",
    "HomologScreen",
    "ItemSelectionSpec",
    "TableColumn",
    "TopUpShortfallError",
    "TrainingTextBound",
    "build_items",
    "eligible_pool",
    "select_items",
]

_ADDED = (
    "within",
    "derived",
    "per_cluster",
    "min_positives",
    "homologs",
    "units",
    "where",
    "pool_clean",
    "within_any",
    "without",
    "shared_names",
    "training_text",
    "min_length",
    "excluded_ids",
)
"""Fields added after selections were stored: left out of the stored spec at their defaults, so the artifact of an
older selection is unchanged (its metadata holds the spec)."""


MIN_LENGTH_REASON = "min_length"
"""The reason, in ``removed``, of an object shorter than ``min_length``."""
OTHER_REASON = "candidate_rules"
"""The reason, in ``removed``, of an object of the ``within`` set that another rule of the spec dropped."""


class TopUpShortfallError(ValueError):
    """The top-up cannot give a task its ``min_positives`` (step 3): the candidates hold fewer positives of it.

    Attributes:
        task: the task.
        have: the positives the top-up reached (the selection's, scarcer tasks first).
        want: the task's ``min_positives``.
    """

    def __init__(self, selection: str, task: str, have: int, want: int) -> None:
        super().__init__(f"selection {selection!r}: task {task!r} reaches {have} of {want} positives")
        self.task, self.have, self.want = task, have, want


class TableColumn(Config):
    """A column of one of a domain's extra tables (``Domain.table``), e.g. the molecule split's ``strata``.

    Attributes:
        table: the extra table's name.
        column: the column.
    """

    table: str = Field(min_length=1)
    column: str = Field(min_length=1)


class ColumnIn(TableColumn):
    """The objects whose value in a :class:`TableColumn` is one of ``values``.

    Attributes:
        values: the values kept, as text.
    """

    values: tuple[str, ...] = Field(min_length=1)


class HomologScreen(Config):
    """Drop a candidate with a homolog among the objects of some item sets (rule E against the training set).

    A hit of ``search`` that passes any of ``rules`` (``build.train_selection.passing_hits``, the training selection's
    matcher) names a candidate when its other sequence is an object of an ``of`` set, a derived object (a protein
    variant) through its parent. The search must have searched this domain's proteins in the selection's splits on one
    side and in the splits of those objects on the other, so no pair escapes unsearched.

    Attributes:
        reason: the name its count is recorded under.
        search: the ``homology`` artifact.
        rules: a hit passes when it passes any of them.
        of: the item sets whose homologs are dropped; each has a role of this domain.
    """

    reason: str = Field(min_length=1)
    search: ArtifactDigest
    rules: tuple[HitRule, ...] = Field(min_length=1)
    of: tuple[ArtifactDigest, ...] = Field(min_length=1)


class TrainingTextBound(Config):
    """The objects with every one of ``fields`` in a screened training set's text (``data.training_text``).

    The training text holds, per training object, the fields that survived the zero-shot screen; a removed field is
    absent (``TrainingText.fields``). An object is inside the bound when the training text holds each of ``fields`` for
    it, so the text is one the set's streams may train on. That is eligibility: it does not say a sampler drew it.

    Attributes:
        training_set: the ``training_set`` artifact.
        fields: the training-text fields an object must have kept.
    """

    training_set: ArtifactDigest
    fields: tuple[str, ...] = Field(min_length=1)


class ItemSelectionSpec(StageConfig):
    """One item selection (the module docstring gives the rule).

    Attributes:
        name: the object list's name.
        domain: the domain at a fixed data version.
        splits: the splits candidates come from.
        max_length: the longest candidate value (a protein's residues), or None for no cap.
        required_fields: fields a candidate must have non-empty (e.g. the field a suite's text task scores).
        exclude: item sets whose objects' clusters the selection avoids; each has a role of this domain.
        size: objects in the selection, or ``all``: every eligible candidate of the ``within`` (or ``within_any``)
            set, with no draw and no top-up (step 3).
        seed: the seed of the cluster order.
        names: add each object's name as the field ``name.object``, for an analysis that groups by the name input
            (Fig 10b's ``stratified_contrast``); an object without a name is then refused.
        within: an item set whose objects (any role of this domain), or an embeddings artifact whose objects, bound
            the candidates, or None (added field).
        derived: select the derived objects (the protein variants) of the candidates instead (added field).
        per_cluster: the most candidates one cluster contributes, chosen by a seeded key per object, or None for whole
            clusters (added field).
        min_positives: task id → the fewest positives the selection holds, topped up first (step 3; needs
            ``per_cluster`` 1; with ``size: all``, only reported; added field).
        homologs: the homolog screens of step 2 (added field).
        units: the column read as each object's cluster, or None for ``Domain.clusters`` (added field).
        where: the candidates' required value of an extra-table column, or None (added field).
        pool_clean: screens that mark the selected objects instead of dropping them (step 4); their ``of`` sets may be
            of another domain, the one the search's other side holds (added field).
        within_any: item sets or embeddings artifacts, as ``within``, whose objects together bound the candidates (the
            union of their objects); not with ``within`` (added field).
        without: item sets or embeddings artifacts whose objects leave the candidates one by one, not by cluster
            (e.g. the objects existing extractions already hold, so an extraction covers only the rest; added field).
        shared_names: keep only candidates whose name the eligible candidates (after the exclusions and homolog
            screens) hold in at least this many clusters, or None (step 2; Fig 10b; with ``size: all``, only
            reported; added field).
        training_text: keep only candidates with these fields in a screened training set's text, object by object, or
            None (step 1; added field).
        min_length: the shortest candidate value, or None for no minimum (step 1; added field).
        excluded_ids: listed ids that leave the candidates one by one, each list with its reason (step 1; reasons
            are unique; added field).
    """

    name: str = Field(min_length=1)
    domain: DomainSpec
    splits: tuple[str, ...] = Field(min_length=1)
    max_length: int | None = Field(ge=1)
    required_fields: tuple[str, ...] = ()
    exclude: tuple[ArtifactDigest, ...] = ()
    size: Annotated[int, Field(ge=1)] | Literal["all"]
    seed: int
    names: bool = False
    within: ArtifactDigest | None = added_field(default=None)
    derived: bool = added_field(default=False)
    per_cluster: int | None = added_field(default=None, ge=1)
    min_positives: dict[str, Annotated[int, Field(ge=1)]] = added_field(default_factory=dict)
    homologs: tuple[HomologScreen, ...] = added_field(default=())
    units: TableColumn | None = added_field(default=None)
    where: ColumnIn | None = added_field(default=None)
    pool_clean: tuple[HomologScreen, ...] = added_field(default=())
    within_any: tuple[ArtifactDigest, ...] = added_field(default=())
    without: tuple[ArtifactDigest, ...] = added_field(default=())
    shared_names: int | None = added_field(default=None, ge=2)
    training_text: TrainingTextBound | None = added_field(default=None)
    min_length: int | None = added_field(default=None, ge=1)
    excluded_ids: tuple[IdExclusion, ...] = added_field(default=())

    @model_validator(mode="after")
    def _check_strata(self) -> ItemSelectionSpec:
        if self.within is not None and self.within_any:
            raise ValueError("within and within_any both bound the candidates; give the bounds as within_any")
        if self.size == "all":
            if self.within is None and not self.within_any:
                raise ValueError("size all takes every remaining object of a set; give the set as within")
            if self.per_cluster is not None:
                raise ValueError(f"size all takes the set's objects as they are; per_cluster is {self.per_cluster}")
        elif self.min_positives and self.per_cluster != 1:
            raise ValueError(f"min_positives tops up one object per cluster; per_cluster is {self.per_cluster}")
        listed = [e.reason for e in self.excluded_ids]
        reserved = [r for r in listed if r in (MIN_LENGTH_REASON, OTHER_REASON) or r.startswith(PARENT_REASON)]
        if len(set(listed)) != len(listed) or reserved:
            raise ValueError(f"excluded_ids reasons repeat or are reserved ({reserved}): {listed}")
        reasons = [h.reason for h in (*self.homologs, *self.pool_clean)]
        if len(set(reasons)) != len(reasons):
            raise ValueError(f"homolog screens share a reason: {reasons}")
        return self


def select_items(spec: ItemSelectionSpec, store: ArtifactStore, data_root: Path | None = None) -> ArtifactRef:
    """Apply the selection rule (module docstring) and commit the object list (:func:`build_items`, then commit).

    Raises:
        ValueError: as :func:`build_items`.
    """
    return commit_item_set(store, *build_items(spec, store, data_root))


def build_items(
    spec: ItemSelectionSpec, store: ArtifactStore, data_root: Path | None = None
) -> tuple[ItemSet, dict[str, Any]]:
    """Apply the selection rule (module docstring); return the object list and its metadata, uncommitted.

    ``data_root`` (``Paths.data``) reads the table files of a ``pool_clean`` search's table side.

    Raises:
        ValueError: for an unknown field, split or extra-table column, a candidate without a unit, an ``exclude`` or
            ``of`` set without a role of this domain, a ``training_text`` artifact that is no training set, a homolog
            search that does not cover the candidates and those objects, a selection that cannot reach ``size`` with
            whole clusters, a ``size: all`` subset that nothing remains of or whose units are not the set's,
            ``excluded_ids`` without the data root, (with ``names``) an object without a name, or a ``pool_clean``
            search that did not search every selected object against every object of its sets.
    """
    pool = eligible_pool(spec, store, data_root)
    domain, clusters, eligible, screened = pool.domain, pool.units, pool.eligible, pool.screened
    in_splits, short, complete, inside, candidates = (
        pool.in_splits,
        pool.short,
        pool.complete,
        pool.inside,
        pool.candidates,
    )
    dropped, unshared, outside_text = pool.without, pool.unshared, pool.outside_text
    groups = {str(c): [str(o) for o in g.index] for c, g in eligible.groupby(eligible)}
    ids = tuple(sorted(eligible.index if spec.size == "all" else _draw(spec, spec.size, domain, groups)))
    if not ids:
        raise ValueError(f"selection {spec.name!r}: no object of the set remains")
    evidence = {
        "in_splits": len(in_splits),
        "over_max_length": len(in_splits) - pool.too_short - len(short),
        "missing_required": len(short) - int(complete.sum()),
        "excluded": len(candidates) - len(eligible) - sum(screened.values()) - unshared,
        "eligible": len(eligible),
        "eligible_clusters": len(groups),
        "selected": cluster_stats(clusters.reindex(pd.Index(ids))).model_dump(),
    }
    bounded = spec.within is not None or spec.within_any or spec.without or spec.training_text is not None
    if bounded or spec.derived or spec.per_cluster is not None or spec.where is not None:
        evidence["candidates"] = len(candidates)  # older selections: none of them
    if spec.without:
        evidence["without"] = dropped
    if spec.training_text is not None:
        evidence["outside_training_text"] = outside_text
    if spec.where is not None:
        evidence["outside_where"] = int(complete.sum() - inside.sum())
    if spec.homologs:
        evidence["homologs"] = screened
    evidence |= {"unshared_names": unshared} if spec.shared_names is not None and spec.size != "all" else {}
    removed = _removed(spec, pool, store, ids) if spec.excluded_ids or spec.size == "all" else None
    evidence |= _later_evidence(spec, pool, removed)
    marks = {s.reason: _marked(s, pd.Index(ids), store, data_root) for s in spec.pool_clean}
    clean = {f"pool_clean.{r}": tuple(o not in m for o in ids) for r, m in marks.items()}
    if marks:
        evidence["pool_clean"] = {r: sum(c) for r, c in zip(marks, clean.values(), strict=True)}
    if spec.min_positives:
        have = {t: len(p) for t, p in _positives(spec, domain, pd.Index(ids)).items()}
        evidence["positives"] = have
        if spec.size == "all":  # reported, never topped up
            evidence["below_min_positives"] = {t: n for t, n in have.items() if n < spec.min_positives[t]}
    unset = {f for f in _ADDED if getattr(spec, f) == ItemSelectionSpec.model_fields[f].default}
    meta = {"spec": spec.model_dump(mode="json", exclude=unset), "spec_digest": spec.digest(), "evidence": evidence}
    if removed is not None:
        meta["removed"] = removed
    items = object_list(spec.name, domain, ids)
    unit = tuple(str(c) for c in clusters.reindex(pd.Index(ids)))  # the unit read (with ``units``, not Domain.clusters)
    _require_stored_units(spec, store, ids, unit)
    items = replace(items, fields={**items.fields, cluster_field(OBJECT_ROLE): unit, **clean})
    if spec.names:
        items = with_names(items, {domain.name: domain}, OBJECT_ROLE)
    return items, meta


def _later_evidence(
    spec: ItemSelectionSpec, pool: EligiblePool, removed: Mapping[str, list[str]] | None
) -> dict[str, Any]:
    """Return the evidence of ``min_length``, ``excluded_ids`` and, with ``size: all``, ``shared_names`` (when set).

    A draw: ``under_min_length``, the objects of the splits under the minimum (as ``over_max_length`` counts), and
    ``excluded_ids``, reason → the candidates it dropped (an object, or its parent, named by several counts under
    each). With ``size: all`` those two would count other objects than the ``removed`` table holds (the whole
    splits; the candidates left after the length rule), so the evidence is instead ``removed``: reason → the entries
    of the ``removed`` table that carry it. ``unpaired_names``: :func:`_unpaired`.
    """
    out: dict[str, Any] = {}
    if spec.size == "all":
        carried = [r for reasons in (removed or {}).values() for r in reasons]
        out["removed"] = {r: carried.count(r) for r in sorted(set(carried))}
        if spec.shared_names is not None:
            out["unpaired_names"] = _unpaired(spec.shared_names, pool.domain, pool.eligible)
        return out
    if spec.min_length is not None:
        out["under_min_length"] = pool.too_short
    if spec.excluded_ids:
        dropped = [r.removeprefix(PARENT_REASON) for reasons in pool.listed.values() for r in reasons]
        out["excluded_ids"] = {e.reason: dropped.count(e.reason) for e in spec.excluded_ids}
    return out


def _require_stored_units(
    spec: ItemSelectionSpec, store: ArtifactStore, ids: tuple[str, ...], unit: tuple[str, ...]
) -> None:
    """Refuse a ``size: all`` subset whose objects' units differ from the ones its ``within`` item set stores.

    A subset declared without the old selection's ``units`` would read the domain's clusters and store another
    ``cluster.<role>`` for the same objects, so an analysis would resample other units. Only the unit is held to the
    old set: a name may change with the domain's data version. A ``within`` embeddings artifact stores no unit.

    Raises:
        ValueError: if a selected object's unit differs from the stored one.
    """
    if spec.size != "all" or spec.within is None or store.ref(spec.within).kind != ITEM_SET_KIND:
        return
    old = load_item_set(store, ArtifactRef(spec.within, ITEM_SET_KIND))
    stored: dict[str, str] = {}
    for role, domain in old.roles.items():
        if domain == spec.domain.name and cluster_field(role) in old.fields:
            stored |= {o: str(c) for o, c in zip(old.objects[role], old.fields[cluster_field(role)], strict=True)}
    moved = [o for o, u in zip(ids, unit, strict=True) if o in stored and stored[o] != u]
    if moved:
        raise ValueError(
            f"selection {spec.name!r}: {len(moved)} objects have another unit than the within set stores, e.g. "
            f"{moved[:3]} (state the old selection's units)"
        )


@dataclass(frozen=True)
class EligiblePool:
    """The candidates of a selection before its draw (module docstring, steps 1 and 2), and each step's objects.

    Attributes:
        domain: the selection's domain.
        units: object → unit (the domain's clusters, or the ``units`` column).
        eligible: object → unit of every eligible candidate: the pool the draw takes whole units from.
        screened: objects each homolog screen dropped, by reason, in order.
        in_splits: the population in ``splits``.
        short: of those, no longer than ``max_length``.
        complete: per ``short`` object, every ``required_fields`` field present.
        inside: ``complete`` and inside ``where``.
        candidates: the bounded candidates (``within``, ``derived``), less ``without``, inside ``training_text``.
        without: how many ``without`` dropped.
        unshared: how many ``shared_names`` dropped.
        outside_text: how many ``training_text`` dropped (their fields are not all in the training text).
        too_short: how many of ``in_splits`` are shorter than ``min_length``.
        named: ``excluded_ids`` reason → the ids its lists name.
        listed: each candidate ``excluded_ids`` dropped → the reasons that name it (``parent:<reason>``: its parent).
    """

    domain: Domain
    units: pd.Series
    eligible: pd.Series
    screened: dict[str, int]
    in_splits: pd.Index
    short: pd.Index
    complete: pd.Series
    inside: pd.Series
    candidates: pd.Index
    without: int
    unshared: int
    outside_text: int
    too_short: int = 0
    named: Mapping[str, frozenset[str]] = field(default_factory=dict)
    listed: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def eligible_pool(spec: ItemSelectionSpec, store: ArtifactStore, data_root: Path | None = None) -> EligiblePool:
    """Return the selection's eligible candidates (steps 1 and 2 of the module docstring), before the draw.

    The clean holdout pool (e.g. the protein evaluation population before its 2,000 draw), which the cross-domain
    2x2 reads as its novel proteins (``build.bindingdb_items``). ``data_root`` reads the ``excluded_ids`` tables.

    Raises:
        ValueError: as :func:`build_items` for its steps 1 and 2, or ``excluded_ids`` without the data root.
    """
    domain = open_domain(spec.domain, store)
    fields, population, split = _checked(spec, domain)
    in_splits = population[split.isin(spec.splits).to_numpy()]
    long = in_splits
    if spec.min_length is not None:
        long = in_splits[(domain.objects()["value"].str.len().reindex(in_splits) >= spec.min_length).to_numpy()]
    short = long
    if spec.max_length is not None:
        short = long[(domain.objects()["value"].str.len().reindex(long) <= spec.max_length).to_numpy()]
    complete = pd.Series(True, index=short)
    for f in spec.required_fields:
        complete &= present(fields[f].reindex(short))
    inside = complete.copy()
    if spec.where is not None:
        inside &= _column(domain, spec.where).reindex(short).isin(spec.where.values)
    named = _named(spec, data_root)
    candidates, dropped, listed = _bounded(spec, domain, store, short[inside.to_numpy()], named)
    candidates, outside_text = _in_training_text(spec, store, candidates)
    clusters = domain.clusters() if spec.units is None else _column(domain, spec.units)
    if clusters.reindex(candidates).isna().any():
        raise ValueError(f"selection {spec.name!r}: candidates without a unit, e.g. {list(candidates[:3])}")
    avoided: set[str] = set()
    for digest in spec.exclude:
        avoided |= {str(c) for c in clusters.reindex(_objects_of(store, digest, spec.domain.name)).dropna()}
    of = clusters.reindex(candidates).astype(str)
    eligible, screened = _screened(spec, domain, store, of[~of.isin(avoided)])
    eligible, unshared = _shared(spec, domain, eligible)
    return EligiblePool(
        domain,
        clusters,
        eligible,
        screened,
        in_splits,
        short,
        complete,
        inside,
        candidates,
        dropped,
        unshared,
        outside_text,
        len(in_splits) - len(long),
        named,
        listed,
    )


def _checked(spec: ItemSelectionSpec, domain: Domain) -> tuple[pd.DataFrame, pd.Index, pd.Series]:
    """Return the domain's fields, population and the population's splits, the spec's fields and splits checked.

    Raises:
        ValueError: for an unknown field, or a split that holds no object.
    """
    fields = domain.fields()
    unknown = sorted(set(spec.required_fields) - set(fields.columns))
    if unknown:
        raise ValueError(f"unknown fields {unknown}; the domain has {sorted(fields.columns)}")
    population = domain.population()
    split = domain.split().reindex(population)
    absent = sorted(set(spec.splits) - set(split.dropna()))
    if absent:
        raise ValueError(f"splits {absent} hold no object of {spec.domain.name!r}")
    return fields, population, split


def _objects_of(store: ArtifactStore, digest: str, domain: str) -> pd.Index:
    """Return the objects of an item set's roles of ``domain``.

    Raises:
        ValueError: if the item set has no role of ``domain``.
    """
    items = load_item_set(store, ArtifactRef(digest, ITEM_SET_KIND))
    roles = [r for r, d in items.roles.items() if d == domain]
    if not roles:
        raise ValueError(f"item set {digest[:12]} ({items.name!r}) has no {domain} role")
    return pd.Index([o for r in roles for o in items.objects[r]])


def _column(domain: Domain, col: TableColumn) -> pd.Series:
    """Return an extra-table column, as text.

    Raises:
        ValueError: if the domain has no such table or column.
    """
    table = domain.table(col.table)
    if table is None or col.column not in table.columns:
        raise ValueError(f"{domain.name!r} has no extra table {col.table!r} with a column {col.column!r}")
    return table[col.column].dropna().astype(str)


def _screened(
    spec: ItemSelectionSpec, domain: Domain, store: ArtifactStore, eligible: pd.Series
) -> tuple[pd.Series, dict[str, int]]:
    """Return the eligible candidates after the homolog screens, and how many each screen dropped (in order)."""
    screened = {}
    for screen in spec.homologs:
        named = eligible.index.isin(list(_homologs(screen, spec, domain, store)))
        screened[screen.reason] = int(named.sum())
        eligible = eligible[~named]
    return eligible, screened


def _shared(spec: ItemSelectionSpec, domain: Domain, eligible: pd.Series) -> tuple[pd.Series, int]:
    """Return the candidates (object → cluster) whose name they hold in ``shared_names`` clusters, and the rest's count.

    Without ``shared_names`` every candidate stays, and with ``size: all`` (step 3: names are only reported, by
    :func:`_unpaired`); an empty name is never shared.
    """
    if spec.shared_names is None or spec.size == "all":
        return eligible, 0
    name = domain.objects()["name"].reindex(eligible.index).fillna("")
    kept = (eligible.groupby(name.to_numpy()).transform("nunique") >= spec.shared_names) & (name != "")
    return eligible[kept], int((~kept).sum())


def _unpaired(shared: int, domain: Domain, selected: pd.Series) -> list[str]:
    """Return the selected objects (object → cluster) whose name the selection holds in fewer than ``shared`` clusters.

    An object with an empty name is one of them. By id.
    """
    name = domain.objects()["name"].reindex(selected.index).fillna("")
    paired = (selected.groupby(name.to_numpy()).transform("nunique") >= shared) & (name != "")
    return sorted(str(o) for o in selected.index[~paired.to_numpy()])


def _named(spec: ItemSelectionSpec, data_root: Path | None) -> dict[str, frozenset[str]]:
    """Return each ``excluded_ids`` reason's listed ids.

    Raises:
        ValueError: if the spec has ``excluded_ids`` and ``data_root`` is None.
    """
    if spec.excluded_ids and data_root is None:
        raise ValueError(f"selection {spec.name!r}: excluded_ids tables need the data root")
    root = data_root or Path()
    return {e.reason: frozenset[str]().union(*(s.read(root) for s in e.ids)) for e in spec.excluded_ids}


def _reasons(
    spec: ItemSelectionSpec, domain: Domain, objects: pd.Index, named: Mapping[str, frozenset[str]]
) -> dict[str, tuple[str, ...]]:
    """Return each of ``objects`` that an ``excluded_ids`` list names → its reasons, in the spec's order.

    With ``spec.derived`` an object is also named through its parent, under ``parent:<reason>``.
    """
    parent = pd.Series([], dtype=str)
    if spec.derived and isinstance(domain, ProteinDomain):
        parent = domain.variants()["parent"].astype(str).reindex(objects)
    out: dict[str, list[str]] = {}
    for reason, ids in named.items():
        for o in objects[objects.isin(ids)]:
            out.setdefault(str(o), []).append(reason)
        for o in parent.index[parent.isin(ids).to_numpy()]:
            out.setdefault(str(o), []).append(PARENT_REASON + reason)
    return {o: tuple(r) for o, r in out.items()}


def _removed(
    spec: ItemSelectionSpec, pool: EligiblePool, store: ArtifactStore, ids: tuple[str, ...]
) -> dict[str, list[str]]:
    """Return the metadata's ``removed`` table (module docstring): object → reasons, by id."""
    out = {o: list(r) for o, r in pool.listed.items()}
    if spec.size == "all":
        bounds = (spec.within,) if spec.within is not None else spec.within_any
        left = _held(spec, store, bounds).difference(pd.Index(ids))
        if spec.derived and isinstance(pool.domain, ProteinDomain):  # the set's variants: its parents only bound them
            left = left[left.isin(pool.domain.variants().index)]
        length = _length(spec, pool.domain, left)
        short = set() if spec.min_length is None else set(left[(length < spec.min_length).to_numpy()])
        listed = _reasons(spec, pool.domain, left, pool.named)
        for o in left:
            reasons = [*listed.get(str(o), ()), *([MIN_LENGTH_REASON] if o in short else [])]
            out[str(o)] = reasons or [OTHER_REASON]
    return dict(sorted(out.items()))


def _length(spec: ItemSelectionSpec, domain: Domain, objects: pd.Index) -> pd.Series:
    """Return each object's value length as the candidates read it: with ``spec.derived``, its parent's."""
    of = pd.Series(objects, index=objects)
    if spec.derived and isinstance(domain, ProteinDomain):
        of = domain.variants()["parent"].astype(str).reindex(objects).fillna(of)
    return pd.Series(domain.objects()["value"].str.len().reindex(pd.Index(of)).to_numpy(), index=objects)


def _homologs(screen: HomologScreen, spec: ItemSelectionSpec, domain: Domain, store: ArtifactStore) -> set[str]:
    """Return the objects of the selection's splits with a passing hit to an object of ``screen.of``.

    Raises:
        ValueError: if the artifact is no homology search, or the search does not cover both sides.
    """
    ref = store.ref(screen.search)
    if ref.kind != HOMOLOGY_KIND:
        raise ValueError(f"homologs {screen.reason!r}: {ref.short} is a {ref.kind!r} artifact, not a homology search")
    search = HomologySearchSpec.model_validate(store.manifest(ref).meta["spec"])
    of = pd.Index([o for d in screen.of for o in _objects_of(store, d, spec.domain.name)]).unique()
    if isinstance(domain, ProteinDomain):  # a variant is matched through its parent (searches hold no variant)
        parent = domain.variants()["parent"].astype(str)
        of = of.difference(parent.index).union(pd.Index(parent.reindex(of).dropna()))
    held = set(domain.split().reindex(of).fillna("in no split"))
    sides = {"query": search.queries, "target": search.targets}
    for mine, other in (("query", "target"), ("target", "query")):
        if _covers(sides[mine], spec, set(spec.splits)) and _covers(sides[other], spec, held):
            hits = passing_hits(load_hits(store, ref), screen.rules)
            return set(hits[mine][hits[other].isin(of)])
    raise ValueError(
        f"homologs {screen.reason!r}: search {search.name!r} did not search splits {list(spec.splits)} of "
        f"{spec.domain.name!r} against the splits of its objects {sorted(held)}"
    )


def _marked(screen: HomologScreen, ids: pd.Index, store: ArtifactStore, data_root: Path | None) -> set[str]:
    """Return the objects of ``ids`` with a passing hit of ``screen.search`` to an object of ``screen.of``.

    One side of the search must hold every object of ``ids`` (a domain's splits, or table rows by id); the other is a
    domain's proteins whose splits hold every object of ``screen.of`` (a variant through its parent).

    Raises:
        ValueError: if the artifact is no homology search, or no orientation covers both.
    """
    ref = store.ref(screen.search)
    if ref.kind != HOMOLOGY_KIND:
        raise ValueError(f"pool_clean {screen.reason!r}: {ref.short} is a {ref.kind!r} artifact, not a homology search")
    search = HomologySearchSpec.model_validate(store.manifest(ref).meta["spec"])
    sides = {"query": search.queries, "target": search.targets}
    for mine, other in (("query", "target"), ("target", "query")):
        theirs = sides[other]
        if not isinstance(theirs, DomainSequences) or not _holds(sides[mine], ids, store, data_root):
            continue
        domain = open_domain(theirs.domain, store)
        of = pd.Index([o for d in screen.of for o in _objects_of(store, d, domain.name)]).unique()
        if isinstance(domain, ProteinDomain):  # a variant is matched through its parent (searches hold no variant)
            parent = domain.variants()["parent"].astype(str)
            of = of.difference(parent.index).union(pd.Index(parent.reindex(of).dropna()))
        if set(domain.split().reindex(of).fillna("in no split")) <= set(theirs.splits):
            hits = passing_hits(load_hits(store, ref), screen.rules)
            return set(hits[mine][hits[other].isin(of)]) & set(ids)
    raise ValueError(
        f"pool_clean {screen.reason!r}: search {search.name!r} did not search the selection against its sets"
    )


def _holds(side: DomainSequences | TableSequences, ids: pd.Index, store: ArtifactStore, data_root: Path | None) -> bool:
    """Return whether a search side searched every object of ``ids`` (table rows are read from ``data_root``)."""
    if isinstance(side, TableSequences) and data_root is None:
        raise ValueError("a pool_clean search with a table side needs the data root")
    held = sequences(side, store, data_root or Path()).index
    return bool(ids.isin(held).all())


def _covers(side: object, spec: ItemSelectionSpec, splits: set[str]) -> bool:
    """Return whether a search side holds this domain's proteins in every one of ``splits``."""
    return isinstance(side, DomainSequences) and side.domain.data == spec.domain.data and splits <= set(side.splits)


def _bounded(
    spec: ItemSelectionSpec,
    domain: Domain,
    store: ArtifactStore,
    candidates: pd.Index,
    named: Mapping[str, frozenset[str]],
) -> tuple[pd.Index, int, dict[str, tuple[str, ...]]]:
    """Return the bounded candidates, less the objects ``excluded_ids`` names and those of ``without``.

    The bound is ``within`` or ``within_any`` (when set); with ``spec.derived``, the candidates' derived objects.
    Also returned: how many ``without`` dropped, and each candidate ``named`` (the ``excluded_ids`` lists) dropped
    with its reasons.
    """
    bounds = (spec.within,) if spec.within is not None else spec.within_any
    bound = _held(spec, store, bounds) if bounds else None
    if bound is not None:
        candidates = candidates[candidates.isin(bound)]
    if spec.derived:
        candidates = _derived(domain, candidates, spec.splits, bound)
    listed = _reasons(spec, domain, candidates, named)
    candidates = candidates[~candidates.isin(list(listed))]
    held = candidates.isin(_held(spec, store, spec.without))
    return candidates[~held], int(held.sum()), listed


def _in_training_text(spec: ItemSelectionSpec, store: ArtifactStore, candidates: pd.Index) -> tuple[pd.Index, int]:
    """Return the candidates with every ``training_text`` field in the training text, and how many lack one.

    Without ``training_text`` every candidate stays. The training text is read with its own loader, whose contract
    (a removed field is absent) is the one definition of a kept text; an object outside the training set has none.

    Raises:
        ValueError: if the artifact is not a training set.
    """
    bound = spec.training_text
    if bound is None:
        return candidates, 0
    kept = load_training_text(store, store.ref(bound.training_set), bound.fields).fields
    inside = np.array([all(f in kept.get(str(o), {}) for f in bound.fields) for o in candidates], dtype=bool)
    return candidates[inside], int((~inside).sum())


def _held(spec: ItemSelectionSpec, store: ArtifactStore, digests: tuple[str, ...]) -> pd.Index:
    """Return the objects the artifacts hold together, each once.

    An item set holds its objects of this domain; an embeddings artifact the objects it holds states for (a capture's
    or an extraction's).
    """
    out: list[str] = []
    for digest in digests:
        ref = store.ref(digest)
        if ref.kind == EMBEDDINGS_KIND:
            out += EmbeddingReader(store, ref).object_ids
        else:
            out += list(_objects_of(store, digest, spec.domain.name))
    return pd.Index(out).unique()


def _derived(domain: Domain, parents: pd.Index, splits: tuple[str, ...], bound: pd.Index | None) -> pd.Index:
    """Return the derived objects (variants) in ``splits`` (and ``bound``, when set) whose parent is in ``parents``.

    Raises:
        ValueError: if the domain has no derived objects.
    """
    if not isinstance(domain, ProteinDomain):
        raise ValueError(f"derived objects are protein variants; {domain.name!r} has none")
    table = domain.variants()
    out = table.index[table["parent"].astype(str).isin(parents).to_numpy()]
    out = out[domain.split().reindex(out).isin(splits).to_numpy()]
    return out if bound is None else out[out.isin(bound)]


def _draw(spec: ItemSelectionSpec, size: int, domain: Domain, groups: dict[str, list[str]]) -> list[str]:
    """Return ``size`` objects from the eligible clusters' candidates (steps 3: the top-up, then the seeded fill).

    Raises:
        ValueError: if whole clusters (or the top-up) cannot reach ``size``.
    """
    ranked = {c: sorted(m, key=lambda o: (tie_key(spec.seed, o), o)) for c, m in groups.items()}
    if spec.per_cluster is not None:
        groups = {c: m[: spec.per_cluster] for c, m in ranked.items()}
    order = sorted(groups, key=lambda c: (tie_key(spec.seed, c), c))
    taken = _top_up(spec, size, domain, ranked, order) if spec.min_positives else {}
    chosen = [o for m in taken.values() for o in m]
    for c in order:
        if len(chosen) == size:
            break
        if c not in taken and len(chosen) + len(groups[c]) <= size:
            chosen += groups[c]
    if len(chosen) != size:
        raise ValueError(f"selection {spec.name!r}: whole clusters reach {len(chosen)} of {size} objects")
    return chosen


def _positives(spec: ItemSelectionSpec, domain: Domain, objects: pd.Index) -> dict[str, set[str]]:
    """Return each ``min_positives`` task's positives among ``objects``: in the task's scope, label > 0.

    Raises:
        KeyError: for a task the domain does not have.
    """
    catalog, labels = domain.tasks(), domain.labels()
    out = {}
    for t in spec.min_positives:
        task = catalog[t]
        in_scope = domain.scope_mask(task, objects).reindex(objects).fillna(False).to_numpy(bool)
        y = labels[task.label].reindex(objects).to_numpy(np.float64)
        out[t] = {str(o) for o in objects[in_scope & (y > 0)]}
    return out


def _top_up(
    spec: ItemSelectionSpec, size: int, domain: Domain, ranked: dict[str, list[str]], order: list[str]
) -> dict[str, list[str]]:
    """Return cluster → its one object taken by the stratified top-up (step 3), scarcest task first.

    Raises:
        TopUpShortfallError: if a task cannot reach its count.
        ValueError: if the top-up alone exceeds ``size``.
    """
    positives = _positives(spec, domain, pd.Index([o for c in order for o in ranked[c]]))
    taken: dict[str, list[str]] = {}
    for t in sorted(positives, key=lambda t: (len(positives[t]), t)):
        have = sum(m[0] in positives[t] for m in taken.values())
        for c in order:
            if have >= spec.min_positives[t]:
                break
            hit = None if c in taken else next((o for o in ranked[c] if o in positives[t]), None)
            if hit is not None:
                taken[c] = [hit]
                have += 1
        if have < spec.min_positives[t]:
            raise TopUpShortfallError(spec.name, t, have, spec.min_positives[t])
    if len(taken) > size:
        raise ValueError(f"selection {spec.name!r}: the top-up takes {len(taken)} objects, more than {size}")
    return taken
