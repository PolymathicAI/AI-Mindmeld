"""Analyses: records (scores, provenance, counts) → the numbers a figure, table or sentence shows.

An analysis is declared as data: an :class:`AnalysisSpec` names a registered analysis *kind*, its inputs (record
computation keys, usually ``$pin``-ed) and the kind's parameters. A kind declares ``Params`` (validated when the spec
loads, if the kind is registered) and ``Result`` (a typed model; ``evaluation.results`` holds those built so far,
and each further kind's model is defined with the kind). Its output is an
``analysis`` record whose artifact holds the result; the pipeline ends at these records, and every plot,
table and number reads them. Statistics come only from
``stats``; a kind composes them. Kinds are an open set registered in :data:`ANALYSIS_KINDS`, like map kinds and suite
kinds (``grouped_contrasts`` is in ``evaluation.contrasts``; ``counts`` and ``removal_counts`` in
``evaluation.counts``, ``cost`` in ``evaluation.cost``, ``cross_domain_quads`` in ``evaluation.cross_domain_quads``,
``cross_domain_ranks`` in ``evaluation.cross_domain_ranks``,
``description_exposure`` in ``evaluation.exposure``,
``execution_drift`` in ``evaluation.drift``, ``excerpts`` in ``evaluation.excerpts``, ``explicit_pairs`` in
``evaluation.explicit_pairs``, ``generation_decline`` and
``name_guess`` in ``evaluation.generation_analyses``, ``layer_selection`` in ``evaluation.layer_selection_analysis``,
``sae_exploration`` in ``evaluation.sae_exploration``, ``stratified_contrast`` in ``evaluation.stratified``).

Kinds:

========================  =================================================================================
``grouped_contrasts``     equal-weight task groups, means of cells, paired contrasts (Figs 5–7, 8a-b)
``stratified_contrast``   a contrast pooled within strata, e.g. lineage within name groups (Fig 10b)
``cross_domain_quads``    E1 over swap quads, crossed bootstrap, paired contrasts between cells (Fig 8c)
``cross_domain_ranks``    E4 ranking (MRR, top-1, unique top) and E5 decoy-pair ordering per cell
``explicit_pairs``        fixed discordant-pair accuracy, paired controls and state change (mutations, E11)
``generation_decline``    direct vs reasoning readouts, cap rate, completion, finished-subset contrast (Fig 9b)
``excerpts``              a declared, seeded selection rule for qualitative excerpts (Fig 9a)
``name_guess``            the share of exactly elicited names, with its interval (name recovery)
``counts``                counts of objects, items and examples from artifacts (Figs 1–4, methods numbers)
``removal_counts``        the zero-shot screen's removal counts per task and field
``description_exposure``  expected description-target exposure per zero-shot task
``cost``                  compute cost from provenance start, end and devices
``execution_drift``       score drift across execution configs: receiver and batch plan
``layer_selection``       a setup's layer-selection records (scan plots)
``sae_exploration``       a frozen sparse dictionary on a capture's own states (diagnostic, never paper)
========================  =================================================================================

**Legacy kind names.** A kind renamed after records were written keeps its old name readable: an analysis config or
record whose ``kind`` is an old name (:data:`LEGACY_ANALYSIS_KINDS`: ``cross_domain_e1`` → ``cross_domain_quads``)
validates, runs and loads as the current kind (:func:`analysis_kind`), and its ``kind`` stays as written, so its
config digest, computation key and record id are unchanged. New configs write the current name.

Tasks are never silently dropped by an analysis: zero-shot tasks are clean by construction, and a thin cell
(``evaluation.contrasts.THIN_CELL``) is left out of its means and listed in the result.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Protocol, TypeVar

from pydantic import Field, model_validator

from latent_interconnect.artifacts.records import Record, RecordStore, drawn_from
from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import ArtifactDigest, Config, RecordKey, StageConfig, added_field
from latent_interconnect.data.items import ItemSet
from latent_interconnect.evaluation.results import AnalysisResult, Cell
from latent_interconnect.plugins import Registry
from latent_interconnect.stats.bootstrap import ResamplingUnits

__all__ = [
    "ANALYSIS_KIND",
    "ANALYSIS_KINDS",
    "LEGACY_ANALYSIS_KINDS",
    "RESULT_FILE",
    "RESULT_OUTPUT",
    "AnalysisKind",
    "AnalysisSpec",
    "Cell",
    "CellMean",
    "Contrast",
    "Group",
    "GroupedContrasts",
    "SupportFloor",
    "analysis_kind",
    "load_result",
    "run_analysis",
]


class AnalysisSpec(StageConfig):
    """One analysis.

    Attributes:
        id: e.g. ``"fig6/protein"``.
        kind: registered analysis kind.
        inputs: input name → record computation key (scores, provenance, counts), usually a ``$pin``. The names are
            labels the kind's parameters cite (a ``grouped_contrasts`` cell's ``inputs``), never parsed.
        params: the kind's parameters, validated by its ``Params``.
        units: the resampling units for any interval, and the interval method of a crossed design
            (``ResamplingUnits.interval``; a kind accepts a method other than the bootstrap only if it lists it in
            ``interval_methods``).
    """

    id: str
    kind: str
    inputs: dict[str, RecordKey] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    units: ResamplingUnits = ResamplingUnits()

    @model_validator(mode="after")
    def _check_params(self) -> AnalysisSpec:
        if self.kind in ANALYSIS_KINDS or self.kind in LEGACY_ANALYSIS_KINDS:
            kind = analysis_kind(self.kind)
            kind.Params.model_validate(self.params)
            if self.units.interval and self.units.interval not in getattr(kind, "interval_methods", ()):
                raise ValueError(
                    f"analysis kind {self.kind!r} computes its intervals by the bootstrap only, "
                    f"not units.interval {self.units.interval!r}"
                )
        return self


class AnalysisKind(Protocol):
    """One analysis kind: ``Params``, ``Result`` and a run function over the input records.

    A kind that computes with libraries beyond the analysis stage's own (``run.analysis.ANALYSIS_PACKAGES``) names
    their modules in a class attribute ``packages``; the stage records the versions its process loaded.
    """

    Params: ClassVar[type[Config]]
    Result: ClassVar[type[AnalysisResult]]

    def run(
        self,
        params: Any,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> AnalysisResult:
        """Return the result.

        Args:
            params: an instance of ``Params``.
            units: the resampling units.
            inputs: role → the canonical record of each input.
            artifacts: the store holding the input records' outputs.
            items: resolves an item-set digest to its item set (the resampling units are item-set columns).
        """
        ...


ANALYSIS_KINDS: Registry[type[AnalysisKind]] = Registry("analysis kind")
"""Analysis kinds by name."""

LEGACY_ANALYSIS_KINDS: Mapping[str, str] = MappingProxyType({"cross_domain_e1": "cross_domain_quads"})
"""Old kind name → current name, for configs and records written before a rename (module docstring)."""


def analysis_kind(name: str) -> type[AnalysisKind]:
    """Return the kind registered under ``name``, or under the current name of a legacy name.

    The legacy names are :data:`LEGACY_ANALYSIS_KINDS`.

    The name itself is never rewritten: a spec keeps the ``kind`` it was written with, so its identity is unchanged.

    Raises:
        UnknownNameError: if neither names a registered kind.
    """
    return ANALYSIS_KINDS.get(LEGACY_ANALYSIS_KINDS.get(name, name))


ANALYSIS_KIND = "analysis"
"""Artifact kind of an analysis result."""
RESULT_OUTPUT = "result"
"""Output name of the ``analysis`` stage's artifact (``latent_interconnect.run``)."""
RESULT_FILE = "result.json"
"""The file of an ``analysis`` artifact that holds the result."""

R = TypeVar("R", bound=AnalysisResult)


class Group(Config):
    """An equal-weight mean over tasks (a figure axis), or over other groups of the same analysis (a mean of strata).

    Attributes:
        name: the axis name.
        tasks: task ids pooled (a task group).
        parts: instead of ``tasks``, the names of two or more task groups of the analysis whose values this group
            averages with equal weight, per draw (e.g. half the trained-task mean plus half the held-out-task mean,
            the connector score); its tasks are the union of theirs (:meth:`GroupedContrasts.pooled`).
    """

    name: str
    tasks: tuple[str, ...] = ()
    parts: tuple[str, ...] = added_field(default=())

    @model_validator(mode="after")
    def _one_kind(self) -> Group:
        if bool(self.tasks) == bool(self.parts):
            raise ValueError(f"group {self.name!r} pools either tasks or parts (two or more groups), not both or none")
        if len(self.parts) == 1 or len(set(self.parts)) != len(self.parts):
            raise ValueError(f"group {self.name!r}: parts are two or more distinct groups, not {list(self.parts)}")
        return self


class Contrast(Config):
    """``a − b`` between two cells (or means of cells, :class:`CellMean`), paired over the shared bootstrap draws.

    ``a`` or ``b`` may name an earlier contrast of the same analysis (a contrast of contrasts): its draws are then that
    contrast's per-draw differences, so ``(d2 − o2) − (d0 − o0)``, the change of a gap between two steps, is one
    signed sum of four cells on the same resample, never a difference of reported limits.

    Attributes:
        name: the contrast name (distinct from every cell, mean and other contrast if a later contrast cites it).
        a: first cell (its name in ``cells``), a mean's (in ``means``) or an earlier contrast's.
        b: second cell, mean or earlier contrast.
    """

    name: str
    a: str
    b: str


class SupportFloor(Config):
    """A ``grouped_contrasts`` support floor (no interval under 10 items or 8 clusters).

    A task of a cell with fewer than ``items`` scored items or fewer than ``clusters`` distinct units of ``factor``
    is left out of that cell's means and listed with its support and its full-sample point (only the interval and its
    part in the group means are withheld), as a thin task is listed; a contrast is then
    reported only between cells that leave out the same tasks. Used by the mutation source subgroups,
    whose scored unit is the item (one row per item, task and cell) and whose clusters are the parents' id30 clusters.

    Attributes:
        items: the minimum scored items.
        clusters: the minimum distinct units of ``factor``.
        factor: the resampling factor counted (one of the analysis's ``units.factors``, e.g. ``cluster.object``).
    """

    items: int = Field(ge=1)
    clusters: int = Field(ge=1)
    factor: str


class CellMean(Config):
    """An equal-weight mean of named cells' group values, on the shared bootstrap draws.

    Its interval is the percentile interval of the per-draw mean, so it keeps the dependence between its cells (shared
    objects, the same draws); it is never the mean of the cells' interval endpoints. A contrast may cite it as ``a`` or
    ``b``: the difference of two means of equally many cells is, draw by draw, the mean of the paired differences of
    their cells (e.g. the gain of a mean over experts: their trained cells' mean minus their starts' mean).

    Attributes:
        name: the mean's name, distinct from every cell's.
        cells: the cells averaged (names in ``cells``), at least two, distinct.
    """

    name: str
    cells: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _distinct(self) -> CellMean:
        if len(set(self.cells)) != len(self.cells):
            raise ValueError(f"mean {self.name!r} averages each cell once, not {list(self.cells)}")
        return self


class GroupedContrasts(Config):
    """Parameters of the ``grouped_contrasts`` kind.

    A per-task analysis is the case of one group per task.

    Attributes:
        cells: cell name → :class:`~latent_interconnect.evaluation.results.Cell` (encoder, inputs, step,
            conditions); the name is a free-form label (results and contrasts cite it), never parsed.
        groups: the pooled axes: task groups, and means of task groups (:class:`Group` ``parts``).
        means: equal-weight means of listed cells, per group (:class:`CellMean`); a mean is reported on a group only
            when every cell it averages has the group and all leave out the same thin tasks there, never over the
            cells that have it (``evaluation.contrasts.grouped_result``).
        contrasts: the contrasts to report (between listed cells, means or earlier contrasts, :class:`Contrast`),
            each a paired difference with its interval.
        metric: the per-task statistic: ``auroc`` of the scores against the labels (Yes/No suites) or ``mean`` of the
            scores (likelihood suites).
        item_filters: suite id → item-set field → accepted value (or list of values): every input of that suite keeps
            only those items (e.g. ``{pfmbench: {pool_clean.<reason>/train_ids: true}}``, the pool-clean
            PFMbench items against the run's training pool).
        min_support: a support floor on every task of every cell (:class:`SupportFloor`), or None, the default and
            every analysis's but the mutation source subgroups': only the thin-cell rule.
        task_aliases: input key → stored task name → the task name the groups cite, applied to that input's rows as
            they are read (``evaluation.contrasts.prepare``), so an input stored under an earlier task id joins the
            groups unchanged (e.g. a reference input's mass task, stored under its earlier id). One to one
            and declared per input: a selector is one input key a cell reads; within an input two stored names
            never map to one name, every stored name must be the input's and no target may already be one of its
            tasks (both checked on the rows). Nothing stored is relabelled; the result records the mapping. Empty,
            the default and every other analysis's.
        object_set: an item set's digest (usually a ``$pin``): every input keeps only its items whose object of each
            of that set's roles is among the set's objects of the role, applied as the rows are read, after the task
            aliases and before the item filters (``evaluation.contrasts.prepare``), so every cell, reference and arms
            alike, is on the set's objects before any contrast (e.g. a reference input on the retained
            evaluation objects its arms score). An input it would empty is refused; the result records it. None, the
            default and every other analysis's.
    """

    cells: dict[str, Cell] = Field(min_length=1)
    groups: tuple[Group, ...] = Field(min_length=1)
    means: tuple[CellMean, ...] = ()
    contrasts: tuple[Contrast, ...] = ()
    metric: Literal["auroc", "mean"] = "auroc"
    item_filters: dict[str, dict[str, str | int | float | bool | list[str | int | float | bool]]] = Field(
        default_factory=dict
    )
    min_support: SupportFloor | None = added_field(default=None)
    task_aliases: dict[str, dict[str, str]] = added_field(default_factory=dict)
    object_set: ArtifactDigest | None = added_field(default=None)

    @model_validator(mode="after")
    def _check(self) -> GroupedContrasts:
        names = [m.name for m in self.means]
        clash = sorted({n for n in names if names.count(n) > 1} | (set(names) & set(self.cells)))
        if clash:
            raise ValueError(f"means {clash} are named twice or as a cell")
        members = sorted({c for m in self.means for c in m.cells} - set(self.cells))
        if members:
            raise ValueError(f"mean cells {members} are not among the listed cells")
        earlier: list[str] = []
        for x in self.contrasts:
            cited = {x.a, x.b} - set(self.cells) - set(names)
            unknown = sorted(cited - set(earlier))
            if unknown:
                raise ValueError(
                    f"contrast cells {unknown} are not among the listed cells or means, or earlier contrasts"
                )
            earlier.append(x.name)
        twice = sorted({c for x in self.contrasts for c in (x.a, x.b)} & {n for n in earlier if earlier.count(n) > 1})
        if twice:
            raise ValueError(f"contrasts cite contrasts {twice}, which are named twice")
        ambiguous = sorted(
            {c for x in self.contrasts for c in (x.a, x.b)} & set(earlier) & (set(self.cells) | set(names))
        )
        if ambiguous:
            raise ValueError(f"contrasts cite {ambiguous}, which name both a contrast and a cell or mean")
        task_groups = {g.name for g in self.groups if g.tasks}
        bad = sorted({p for g in self.groups for p in g.parts} - task_groups)
        if bad:
            raise ValueError(f"group parts {bad} are not task groups of this analysis")
        read = {k for c in self.cells.values() for k in c.inputs}
        stray = sorted(set(self.task_aliases) - read)
        if stray:
            raise ValueError(f"task_aliases select {stray}: a selector is one input key a cell reads")
        for key, aliases in self.task_aliases.items():
            if not aliases or len(set(aliases.values())) != len(aliases):
                raise ValueError(f"task_aliases of input {key!r} must map stored tasks one to one, not {aliases}")
        return self

    def pooled(self, group: Group) -> tuple[str, ...]:
        """Return the tasks a group pools: its own, or the union of its parts' (in their order)."""
        if group.tasks:
            return group.tasks
        by_name = {g.name: g for g in self.groups}
        return tuple(dict.fromkeys(t for p in group.parts for t in by_name[p].tasks))


def run_analysis(
    spec: AnalysisSpec, artifacts: ArtifactStore, records: RecordStore, items: Callable[[str], ItemSet]
) -> ArtifactRef:
    """Resolve the inputs to their canonical records, run the kind, and commit its result as an ``analysis`` artifact.

    It writes no record: the ``analysis`` stage records the run with the spec's inputs as its record inputs, so the
    record's purpose follows them (``result`` only if every input record is a result; ``Workspace.purpose``), and the
    exact executions read (``Provenance.input_executions``). The artifact's meta ``inputs`` is
    ``artifacts.records.drawn_from`` them (computation keys and output digests), so identical results give identical
    artifacts. It is written in a fresh staging directory. ``items`` resolves item-set digests (the stage passes the
    item-set loader of ``data.items``).

    Raises:
        LookupError: if an input has no canonical execution.
        UnknownNameError: if the kind is not registered.
    """
    kind = analysis_kind(spec.kind)
    inputs = {role: records.canonical(key) for role, key in sorted(spec.inputs.items())}
    result = kind().run(kind.Params.model_validate(spec.params), spec.units, inputs, artifacts, items)
    meta = {"spec": spec.digest(), "kind": spec.kind, "inputs": drawn_from(inputs)}
    staged = artifacts.staging(f"analysis-{uuid.uuid4().hex}")  # a scratch name, never part of the artifact
    (staged / RESULT_FILE).write_text(result.model_dump_json(indent=1))
    for name, text in result.companion_files().items():
        (staged / name).write_text(text)
    return artifacts.commit(staged, ANALYSIS_KIND, meta)


def load_result(artifacts: ArtifactStore, record: Record, result: type[R]) -> R:
    """Return the result of an ``analysis`` record, validated as its kind's ``Result``.

    The record's own config names the kind, so passing ``AnalysisResult`` returns the kind's full result model.

    Raises:
        TypeError: if that model is not a ``result`` (pass ``AnalysisResult`` to accept any kind).
        ValueError: if the record is not an ``analysis`` record.
    """
    if record.kind != "analysis":
        raise ValueError(f"record {record.record_id()[:12]} is a {record.kind!r} record, not an analysis")
    spec = AnalysisSpec.model_validate(record.config)
    model = analysis_kind(spec.kind).Result
    if not issubclass(model, result):
        raise TypeError(f"analysis kind {spec.kind!r} returns {model.__name__}, not {result.__name__}")
    out = record.outputs[RESULT_OUTPUT]
    ref = ArtifactRef(out.sha256, out.kind)
    artifacts.manifest(ref)
    loaded: R = model.model_validate_json((artifacts.path(ref) / RESULT_FILE).read_text())
    return loaded
