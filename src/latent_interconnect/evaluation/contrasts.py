"""The ``grouped_contrasts`` analysis kind: cells × equal-weight task groups, with paired contrasts between cells.

A *cell* is named and declared by fields (:class:`~latent_interconnect.evaluation.results.Cell`): the rows of its
``inputs``' score tables, concatenated (no task in two of them), at its ``step`` under its ``conditions`` (slot role →
condition). A suite run scores one step, so Fig 7's initial and fitted points of one connector are two inputs and two
cells, and Fig 7b's PFMbench localization column is a second input of each cell. Input keys and cell names are
labels: nothing parses them. Every input a cell names must be declared, and every input must be read by some cell.
``item_filters`` are keyed by suite id (the ``id`` of the suite an input record scored): every input of that suite keeps
only its items whose item-set fields match, so a population's filter is declared once (e.g. the pool-clean PFMbench
items, ``pool_clean.<training-ids pin>``). ``task_aliases`` (input key → stored task → the groups' name) renames an
input's tasks one to one as its rows are read, e.g. a reference stored under an earlier id of a renamed task;
it changes no row, label or unit, so the cohort and pairing are those of the unaliased rows, and the result records it
(``GroupedContrastsResult.task_aliases``). ``object_set`` (an item set's digest) keeps, in every input, only the items
whose objects are that set's, as the rows are read (:func:`_restricted`), so a reference scored on a wider item set is
on its arms' objects before any contrast; the result records it.
Per cell and task the statistic is the AUROC of the scores against the labels
(``metric: auroc``; unlabelled rows are left out) or the mean score (``metric: mean``); a group's value is the mean over
its tasks, and a group of ``parts`` (a mean of strata) is the equal-weight mean of its parts' values, every draw on the
unstratified matrix (:func:`parts_value`). **Thin cells, the one rule wherever cells are aggregated:**
under ``auroc``, a task with fewer than :data:`THIN_CELL` positive or negative rows in a cell is left out of that cell's
means and listed with its support (``GroupedContrastsResult.unavailable``; :func:`split_thin`); a group all of whose
tasks are thin has no value for the cell, and a contrast on a group is reported only when its two cells leave out the
same tasks there, so it always compares like with like (:func:`grouped_result`). Any other task without a value on the
full sample is an error, never dropped; a draw in which a task has no value (undefined by design, e.g. a crossed
resample without positives) leaves the group's mean undefined in that draw, never averaged over fewer tasks, and
``stats.bootstrap.summarize`` drops and counts it. A per-task analysis is one group per task. Every task's rows
in every cell are recorded with the result (``GroupedContrastsResult.support``, :func:`support_of`), so each
value is read with its ``n``. A declared support floor (``GroupedContrasts.min_support``, :func:`split_floor`; the
mutation source subgroups' only) leaves out of the means, and lists the same way with its point, a task under its
minimum items or clusters; no other analysis has one.

Intervals come from the one bootstrap (``stats.bootstrap``). Each resampling factor of ``units`` names an item-set
column (:func:`unit_columns`: a field such as ``cluster.<role>``, or ``protein_cluster`` / ``molecule_group`` for swap
quads, or a role's objects), joined to the score rows by ``item``; the item set is the one the input's suite scored (the
``item_set`` of the suite in ``SuiteRunConfig``), resolved by the caller's ``items``. A factor's units are its distinct
values over all inputs, resampled independently per factor (crossed), and a row counts as many times as the product of
its units' draw counts. A cell with several inputs draws over the union of their objects (each input's own resample size
then varies), and an object shared by inputs is one unit, so it is paired. A one-task group's AUROC draws resample its
positive and negative units separately (``stats.auroc.class_strata``; a task with fewer than two of either is
refused), so they are the same in every cell whose rows give its units the same classes. A group of several tasks
resamples the units jointly: one unstratified draw matrix, shared by its tasks and by every cell, so the group's
interval keeps the correlation between tasks that share objects (:func:`group_draws`); a draw in which a task lacks a
class is undefined by design. A ``mean`` task always uses that one unstratified draw matrix. A
contrast is the difference of the two cells' values with the percentile interval of the per-draw difference (paired),
under either metric; it is reported as an estimate, never graded. A contrast may cite an earlier
contrast (``Contrast``): its draws are that contrast's per-draw differences, so a change of a gap, ``(d2 − o2) −
(d0 − o0)``, is one signed sum of four cells on the same draws, its thin tasks the union of both sides'.
A declared *mean of cells*
(``GroupedContrasts.means``; e.g. the three experts of a receiver, or the nine receiver × expert cells
of a domain, one analysis listing every cell) is the equal-weight mean of its cells' group values, its interval the
percentile interval of the per-draw mean on the same shared draws, so it keeps the dependence between its cells and is
never the mean of their endpoints; a contrast may cite it, so the gain of a mean is the per-draw mean of its cells'
paired differences. A mean is reported on a group only when every cell it averages has the group and all leave out the
same thin tasks there, like a contrast: never over the cells that have a value (their thin tasks are listed with their
support in ``unavailable``). :func:`prepare` and
:func:`cell_tasks` are the kind's own steps, public so that ``evaluation.stratified`` reads the same draws.

**Shared-unit interval** (``units.interval: shared_unit``; ``stats.shared_units``). An opt-in alternative
for a crossed design, the bootstrap staying the default: defined here for ``metric: auroc`` and groups of one task. The
observations are the task's rows, their influence values the placement-value influence
(``stats.auroc.auroc_influence``), their units the row's unit of each factor; a contrast is read from the paired
differences of the two cells' influence values, which must hold the same items (:class:`Observed`,
:func:`shared_unit_result`). A group of several tasks, ``metric: mean`` or a mean of cells is refused when the method
is selected, and only then. The point estimates are the same. Empirically conservative and approximately 95%, with one
unresolved shortfall: the module docstring of ``stats.shared_units``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from latent_interconnect.artifacts.records import Record, record_output
from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.config import Config
from latent_interconnect.data.items import ItemSet
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS, GroupedContrasts, SupportFloor
from latent_interconnect.evaluation.results import (
    AnalysisResult,
    Cell,
    ContrastSpec,
    Estimate,
    GroupedContrastsResult,
    GroupPoint,
    Support,
    Unavailable,
)
from latent_interconnect.evaluation.scores import SCORES_KIND, read_scores
from latent_interconnect.stats.auroc import auroc_counts, auroc_influence, class_strata
from latent_interconnect.stats.bootstrap import Bootstrap, ResamplingUnits, summarize, unit_index
from latent_interconnect.stats.shared_units import SHARED_UNIT, shared_unit_interval

__all__ = [
    "THIN_CELL",
    "UNIT_PREFIX",
    "CellDraws",
    "GroupedContrastsKind",
    "Observed",
    "Prepared",
    "Task",
    "cell_draws",
    "cell_tasks",
    "group_draws",
    "grouped_result",
    "is_thin",
    "parts_value",
    "prepare",
    "shared_unit_result",
    "split_floor",
    "split_thin",
    "suite_of",
    "support_of",
    "task_draws",
    "task_estimate",
    "tasks_of",
]

UNIT_PREFIX = "unit."
"""Prefix of the per-row unit columns joined from the item set (``unit.<factor>``)."""
_UNIT = UNIT_PREFIX
THIN_CELL = 10
"""An AUROC task with fewer positive (or negative) rows than this in a cell is thin: left out of that
cell's means and listed as unavailable with its support, never silently dropped."""


@dataclass(frozen=True)
class Task:
    """One task's rows in one cell: scores, labels and each row's unit index per factor.

    Attributes:
        items: the rows' item indices.
        scores: the rows' scores.
        labels: the rows' labels (True for the positive class).
        units: factor → each row's unit index.
    """

    items: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    units: Mapping[str, np.ndarray]


@ANALYSIS_KINDS.register("grouped_contrasts")
class GroupedContrastsKind:
    """Equal-weight task groups per cell, with paired contrasts (params :class:`GroupedContrasts`)."""

    Params: ClassVar[type[Config]] = GroupedContrasts
    Result: ClassVar[type[AnalysisResult]] = GroupedContrastsResult
    interval_methods: ClassVar[tuple[str, ...]] = (SHARED_UNIT,)

    def run(
        self,
        params: GroupedContrasts,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> GroupedContrastsResult:
        """Compute every cell's group values and every contrast, with intervals from the shared draws.

        Raises:
            KeyError: if a cell names an undeclared input, a factor or filter field is not a column of an input's item
                set, or an item filter names a suite no input scored.
            ValueError: if an input is read by no cell, an input is not a suite record, two inputs of a cell share a
                task, or a cell or a task has no rows.
        """
        prepared = prepare(params, units, inputs, artifacts, items)
        if params.min_support is not None and (
            units.interval == SHARED_UNIT or params.min_support.factor not in units.factors
        ):
            raise ValueError(
                f"min_support counts factor {params.min_support.factor!r}: one of the units {units.factors}, under "
                "the bootstrap"
            )
        if units.interval == SHARED_UNIT:
            return _shared_unit_run(params, units, prepared)
        per_group: dict[str, dict[str, tuple[float, np.ndarray]]] = {}
        unavailable: dict[str, dict[str, Unavailable]] = {}
        support: dict[str, dict[str, Support]] = {}
        for c in params.cells:
            rows = cell_tasks(prepared, params, c)
            support[c] = support_of(rows, params.metric)
            tasks, thin = split_thin(rows, params.metric)
            if params.min_support is not None:
                tasks, low = split_floor(tasks, params.min_support, params.metric)
                thin |= low
            tasks = _valued(tasks, params.metric, f"cell {c!r}")
            per_group[c] = {}
            for g in params.groups:
                kept = [t for t in g.tasks if t in tasks]
                if kept:
                    per_group[c][g.name] = (
                        _group_value(tasks, kept, params.metric),
                        group_draws(tasks, kept, params.metric, prepared.n_units),
                    )
            by_name = {g.name: g for g in params.groups}
            for g in params.groups:
                parts = [[t for t in by_name[p].tasks if t in tasks] for p in g.parts]
                if parts and all(parts):
                    per_group[c][g.name] = parts_value(tasks, parts, params.metric, prepared.n_units)
            if thin:
                unavailable[c] = thin
        return grouped_result(params, units, per_group, unavailable, support)


@dataclass(frozen=True)
class Observed:
    """One cell's value of a group with what its shared-unit interval reads (``stats.shared_units``).

    Attributes:
        estimate: the group's value on the full sample.
        influence: each observation's influence value on it.
        keys: each observation's identity (distinct); a paired contrast needs the same keys in both cells.
        units: factor → each observation's unit indices, ``(n,)`` or ``(n, m)`` padded with the unit count.
        unit_counts: factor → number of units.
        bounds: the range of the statistic.
        classes: each observation's label class for an AUROC, else None.
    """

    estimate: float
    influence: np.ndarray
    keys: np.ndarray
    units: Mapping[str, np.ndarray]
    unit_counts: Mapping[str, int]
    bounds: tuple[float, float]
    classes: np.ndarray | None = None

    def interval(self) -> Estimate:
        """Return the value with its shared-unit interval."""
        return Estimate.of(
            shared_unit_interval(
                self.estimate, self.influence, self.units, self.unit_counts, self.bounds, classes=self.classes
            )
        )

    def minus(self, other: Observed, estimate: float, what: str) -> Estimate:
        """Return the paired contrast ``self − other`` with its shared-unit interval.

        Raises:
            ValueError: if the two cells do not hold the same observations with the same units and classes.
        """
        same = np.array_equal(self.keys, other.keys) and all(
            np.array_equal(ix, other.units[f]) for f, ix in self.units.items()
        )
        if self.classes is not None and other.classes is not None:
            same = same and np.array_equal(self.classes, other.classes)
        if not same or dict(self.unit_counts) != dict(other.unit_counts):
            raise ValueError(f"{what}: a paired shared-unit contrast needs the same observations in both cells")
        return Estimate.of(
            shared_unit_interval(
                estimate,
                self.influence - other.influence,
                self.units,
                self.unit_counts,
                self.bounds,
                contrast=True,
                classes=self.classes,
            )
        )


def shared_unit_result(
    params: GroupedContrasts,
    units: ResamplingUnits,
    per_group: Mapping[str, Mapping[str, Observed]],
    unavailable: Mapping[str, Mapping[str, Unavailable]] | None = None,
    support: Mapping[str, Mapping[str, Support]] | None = None,
) -> GroupedContrastsResult:
    """Return a result from each cell's per-group :class:`Observed`: :func:`grouped_result` with shared-unit intervals.

    The same cells, groups, contrast names and thin rule as :func:`grouped_result`; a contrast's interval is read from
    the paired differences of the two cells' influence values.
    """
    thin = {c: set(t) for c, t in (unavailable or {}).items()}
    values = {
        cell: {g.name: groups[g.name].interval() for g in params.groups if g.name in groups}
        for cell, groups in per_group.items()
    }
    contrasts, specs = {}, {}
    for c in params.contrasts:
        for group in params.groups:
            left_out = [thin.get(x, set()) & set(params.pooled(group)) for x in (c.a, c.b)]
            if group.name not in values[c.a] or group.name not in values[c.b] or left_out[0] != left_out[1]:
                continue
            name = c.name if len(params.groups) == 1 else f"{c.name}/{group.name}"
            difference = values[c.a][group.name].estimate - values[c.b][group.name].estimate
            specs[name] = ContrastSpec(a=c.a, b=c.b, group=group.name)
            contrasts[name] = per_group[c.a][group.name].minus(
                per_group[c.b][group.name], difference, f"contrast {name!r}"
            )
    return GroupedContrastsResult(
        units=units,
        cells=values,
        cell_specs=params.cells,
        contrasts=contrasts,
        contrast_specs=specs,
        unavailable={c: dict(t) for c, t in (unavailable or {}).items()},
        support={c: dict(t) for c, t in (support or {}).items()},
        task_aliases=params.task_aliases,
        object_set=params.object_set,
    )


def _shared_unit_run(params: GroupedContrasts, units: ResamplingUnits, prepared: Prepared) -> GroupedContrastsResult:
    """Return the result with shared-unit intervals: one single-task AUROC per group, the rows as observations.

    Raises:
        ValueError: if the metric is not ``auroc``, a group holds several tasks, the analysis declares means of
            cells or a contrast of contrasts (the shared-unit interval is not defined for them), or a task holds an
            item twice.
    """
    if params.means:
        raise ValueError(
            f"the shared-unit interval is not defined for a mean of cells: {[m.name for m in params.means]}"
        )
    nested = sorted(c.name for c in params.contrasts if {c.a, c.b} - set(params.cells))
    if nested:
        raise ValueError(f"the shared-unit interval is not defined for a contrast of contrasts: {nested}")
    if params.metric != "auroc":
        raise ValueError(
            f"the shared-unit interval of grouped_contrasts is defined for metric 'auroc', not {params.metric!r}"
        )
    several = sorted(g.name for g in params.groups if len(g.tasks) != 1)
    if several:
        raise ValueError(f"the shared-unit interval is not defined for a mean over several tasks: groups {several}")
    per_group: dict[str, dict[str, Observed]] = {}
    unavailable: dict[str, dict[str, Unavailable]] = {}
    support: dict[str, dict[str, Support]] = {}
    for c in params.cells:
        rows = cell_tasks(prepared, params, c)
        support[c] = support_of(rows, params.metric)
        tasks, thin = split_thin(rows, params.metric)
        tasks = _valued(tasks, params.metric, f"cell {c!r}")
        per_group[c] = {
            g.name: _observed_auroc(tasks[g.tasks[0]], prepared.n_units, f"cell {c!r}")
            for g in params.groups
            if g.tasks[0] in tasks
        }
        if thin:
            unavailable[c] = thin
    return shared_unit_result(params, units, per_group, unavailable, support)


def _observed_auroc(task: Task, n_units: Mapping[str, int], what: str) -> Observed:
    """Return a task's AUROC with its rows' placement-value influence, the rows in item order."""
    order = np.argsort(task.items, kind="stable")
    if np.unique(task.items).size != task.items.size:
        raise ValueError(f"{what}: a task holds an item twice; its rows are not paired observations")
    return Observed(
        estimate=_task_stat(task, "auroc", None),
        influence=auroc_influence(task.scores, task.labels)[order],
        keys=task.items[order],
        units={f: ix[order] for f, ix in task.units.items()},
        unit_counts=dict(n_units),
        bounds=(0.0, 1.0),
        classes=task.labels[order],
    )


def support_of(tasks: Mapping[str, Task], metric: str) -> dict[str, Support]:
    """Return task → the rows it is computed from: every row read, and under ``auroc`` its two classes.

    ``tasks`` are a cell's rows per task (:func:`cell_tasks`: filtered, unlabelled rows already left out for AUROC),
    before the thin rule, so a thin task's support is listed too.
    """
    out = {}
    for name, task in tasks.items():
        positives = int(task.labels.sum())
        if metric == "auroc":
            out[name] = Support(
                n=int(task.labels.size), positives=positives, negatives=int(task.labels.size) - positives
            )
        else:
            out[name] = Support(n=int(task.labels.size))
    return out


def grouped_result(
    params: GroupedContrasts,
    units: ResamplingUnits,
    per_group: Mapping[str, Mapping[str, tuple[float, np.ndarray]]],
    unavailable: Mapping[str, Mapping[str, Unavailable]] | None = None,
    support: Mapping[str, Mapping[str, Support]] | None = None,
) -> GroupedContrastsResult:
    """Return a result from each cell's per-group (value, draws): group estimates, means of cells, paired contrasts.

    A cell lacks the groups whose tasks are all thin (``unavailable``: cell → thin task → support). A mean of
    cells on a group, and a contrast on a group, is reported only when each of its cells (or means) has the group and
    all leave out the same thin tasks of it, so they pool the same tasks; a mean is the per-draw mean of its cells'
    draws (a draw undefined in one cell is undefined in the mean). A contrast citing an earlier contrast reads its
    per-draw differences and point, and leaves out the union of its sides' thin tasks. ``summarize`` drops and counts
    a value's undefined draws. ``support`` (cell → task → its rows, :func:`support_of`) is recorded as given. A group
    of exactly one task that the support floor left out with a defined point gets that point, without an interval, in
    ``points``; it joins no mean or contrast, and a group of several tasks is unchanged.
    """
    thin = {c: set(t) for c, t in (unavailable or {}).items()}
    values: dict[str, dict[str, Estimate]] = {}
    draws: dict[tuple[str, str], np.ndarray] = {}
    for cell, groups in per_group.items():
        values[cell] = {}
        for group in params.groups:
            if group.name in groups:
                estimate, draws[(cell, group.name)] = groups[group.name]
                values[cell][group.name] = Estimate.of(summarize(draws[(cell, group.name)], estimate))
    means: dict[str, dict[str, Estimate]] = {}
    for m in params.means:
        means[m.name] = {}
        thin[m.name] = set().union(*(thin.get(c, set()) for c in m.cells))
        for group in params.groups:
            alike = {frozenset(thin.get(c, set()) & set(params.pooled(group))) for c in m.cells}
            if any((c, group.name) not in draws for c in m.cells) or len(alike) != 1:
                continue
            draws[(m.name, group.name)] = np.mean([draws[(c, group.name)] for c in m.cells], axis=0)
            estimate = float(np.mean([values[c][group.name].estimate for c in m.cells]))
            means[m.name][group.name] = Estimate.of(summarize(draws[(m.name, group.name)], estimate))
    stated = values | means
    paired = {(x, g): (d, stated[x][g].estimate) for (x, g), d in draws.items()}
    contrasts, specs = {}, {}
    for c in params.contrasts:
        for group in params.groups:
            left_out = [thin.get(x, set()) & set(params.pooled(group)) for x in (c.a, c.b)]
            if (c.a, group.name) not in paired or (c.b, group.name) not in paired or left_out[0] != left_out[1]:
                continue
            name = c.name if len(params.groups) == 1 else f"{c.name}/{group.name}"
            (draws_a, point_a), (draws_b, point_b) = paired[(c.a, group.name)], paired[(c.b, group.name)]
            diff = draws_a - draws_b
            specs[name] = ContrastSpec(a=c.a, b=c.b, group=group.name)
            contrasts[name] = Estimate.of(summarize(diff, point_a - point_b))
            if c.name not in stated:  # a later contrast may cite it (``Contrast``): its draws, point and thin tasks
                paired[(c.name, group.name)] = (diff, point_a - point_b)
        if c.name not in stated:
            thin[c.name] = thin.get(c.a, set()) | thin.get(c.b, set())
    points = {
        c: {
            g.name: GroupPoint(
                point=u.point,
                n=u.n,
                clusters=u.clusters,
                unavailable=f"under the support floor: {u.n} items, {u.clusters} clusters",
            )
            for g in params.groups
            if len(g.tasks) == 1
            and (u := tasks.get(g.tasks[0])) is not None
            and u.clusters is not None
            and u.point is not None
        }
        for c, tasks in (unavailable or {}).items()
    }
    return GroupedContrastsResult(
        units=units,
        cells=values,
        cell_specs=params.cells,
        points={c: p for c, p in points.items() if p},
        means=means,
        mean_specs={m.name: m.cells for m in params.means},
        contrasts=contrasts,
        contrast_specs=specs,
        unavailable={c: dict(t) for c, t in (unavailable or {}).items()},
        support={c: dict(t) for c, t in (support or {}).items()},
        task_aliases=params.task_aliases,
        object_set=params.object_set,
    )


def is_thin(positives: int, negatives: int, min_class: int | None = None) -> bool:
    """Return whether a task with these labeled rows is thin.

    Thin: fewer than ``min_class`` (default :data:`THIN_CELL`) positive or negative rows.
    """
    return min(positives, negatives) < (THIN_CELL if min_class is None else min_class)


def split_thin(
    tasks: Mapping[str, Task], metric: str, min_class: int | None = None
) -> tuple[dict[str, Task], dict[str, Unavailable]]:
    """Return (the tasks kept, task → support of each thin task, :func:`is_thin`): the thin-cell rule, ``auroc`` only.

    A ``mean`` task has no classes, so none is thin.
    """
    kept, thin = {}, {}
    for name, task in tasks.items():
        positives = int(task.labels.sum())
        negatives = int(task.labels.size) - positives
        if metric == "auroc" and is_thin(positives, negatives, min_class):
            thin[name] = Unavailable(n=int(task.labels.size), positives=positives, negatives=negatives)
        else:
            kept[name] = task
    return kept, thin


def split_floor(
    tasks: Mapping[str, Task], floor: SupportFloor, metric: str
) -> tuple[dict[str, Task], dict[str, Unavailable]]:
    """Return (the tasks kept, task → support of each task under the support floor), :class:`SupportFloor`'s rule.

    A task under the floor keeps its point (its statistic on the full sample, None when undefined); only its interval
    and its part in the cell's group means and contrasts are withheld.
    """
    kept, low = {}, {}
    for name, task in tasks.items():
        n, clusters = int(np.unique(task.items).size), int(np.unique(task.units[floor.factor]).size)
        if n < floor.items or clusters < floor.clusters:
            positives = int(task.labels.sum())
            point = _task_stat(task, metric, None)
            low[name] = Unavailable(
                n=n,
                positives=positives,
                negatives=n - positives,
                clusters=clusters,
                point=float(point) if np.isfinite(point) else None,
            )
        else:
            kept[name] = task
    return kept, low


@dataclass(frozen=True)
class Prepared:
    """An analysis's score tables (filtered, with unit columns) and its unit index.

    Attributes:
        tables: input key → the input's score rows.
        index: factor → unit value → unit index (over every input).
    """

    tables: Mapping[str, pd.DataFrame]
    index: Mapping[str, Mapping[str, int]]

    @property
    def n_units(self) -> dict[str, int]:
        """Factor → number of units: the bootstrap's (:func:`task_draws`)."""
        return {f: len(ix) for f, ix in self.index.items()}


def prepare(
    params: GroupedContrasts,
    units: ResamplingUnits,
    inputs: Mapping[str, Record],
    artifacts: ArtifactStore,
    items: Callable[[str], ItemSet],
) -> Prepared:
    """Read every input's scores, apply its task aliases, the object set and its suite's item filters; join the units.

    Raises:
        KeyError: if a cell names an undeclared input, a factor or filter field is not a column of an input's item
            set, a filter names a suite no input scored, or a role of the object set is not one of an input's.
        ValueError: if an input is read by no cell, an input is not a suite record, a task alias names a task the
            input lacks or a target it already holds, or the object set leaves an input no rows.
    """
    _check_inputs(params, inputs)
    unknown = sorted(set(params.item_filters) - {suite_of(r)["id"] for r in inputs.values()})
    if unknown:
        raise KeyError(f"item filters name suites no input scored: {unknown}")
    within = None if params.object_set is None else items(params.object_set)
    tables = {}
    for key, rec in inputs.items():
        table = _aliased(read_scores(artifacts, record_output(rec, SCORES_KIND)), params.task_aliases.get(key), key)
        suite = suite_of(rec)
        item_set = items(str(suite["item_set"]))
        if within is not None:
            table = _restricted(table, item_set, within, key)
        for field, accepted in params.item_filters.get(str(suite["id"]), {}).items():
            field_values = _item_field(item_set, field)
            ok = accepted if isinstance(accepted, list) else [accepted]
            table = table[table["item"].map(lambda i, v=field_values, a=ok: v[i] in a)]
        columns = unit_columns(item_set, units.factors)
        tables[key] = table.assign(**{_UNIT + f: table["item"].map(lambda i, c=c: c[i]) for f, c in columns.items()})
    return Prepared(tables, _unit_index(tables, units.factors))


def _aliased(table: pd.DataFrame, aliases: Mapping[str, str] | None, key: str) -> pd.DataFrame:
    """Return an input's score rows with its declared task aliases applied (``GroupedContrasts.task_aliases``).

    Raises:
        ValueError: if an aliased stored task has no rows in the input, or a target is already one of its tasks (the
            mapping would merge two tasks).
    """
    if not aliases:
        return table
    stored = set(table["task"])
    missing, clash = sorted(set(aliases) - stored), sorted(set(aliases.values()) & stored)
    if missing or clash:
        raise ValueError(f"input {key!r}: task_aliases name stored tasks {missing} it lacks, targets {clash} it holds")
    return table.assign(task=table["task"].map(lambda t: aliases.get(t, t)))


def _restricted(table: pd.DataFrame, item_set: ItemSet, within: ItemSet, key: str) -> pd.DataFrame:
    """Return an input's score rows whose item's object of each role of ``within`` is among ``within``'s of the role.

    Raises:
        KeyError: if a role of ``within`` is not a role of the input's item set.
        ValueError: if no row is left.
    """
    missing = sorted(set(within.roles) - set(item_set.roles))
    if missing:
        raise KeyError(f"input {key!r}: object set {within.name!r} has roles {missing} its item set lacks")
    keep = pd.Series(True, index=table.index)
    for role, objects in within.objects.items():
        of_item, allowed = item_set.objects[role], set(objects)
        keep &= table["item"].map(lambda i, o=of_item, a=allowed: o[i] in a)
    if not keep.any():
        raise ValueError(f"input {key!r}: object set {within.name!r} leaves it no rows")
    return table[keep]


def cell_tasks(prepared: Prepared, params: GroupedContrasts, name: str) -> dict[str, Task]:
    """Return the rows per task of the analysis's groups of the cell ``name``, on the prepared units.

    Raises:
        ValueError: if two inputs of the cell share a task, or the cell or a task has no rows.
    """
    cell = params.cells[name]
    keys = _cell_inputs(cell)
    parts = [prepared.tables[k] for k in keys]
    seen = [t for p in parts for t in set(p["task"])]
    shared = {t for t in seen if seen.count(t) > 1}
    if shared:
        raise ValueError(f"cell {name!r}: inputs {keys} share tasks {sorted(shared)}")
    rows = pd.concat(parts, ignore_index=True)
    rows = rows[rows["step"].isna() if cell.step is None else rows["step"].eq(cell.step).fillna(False)]
    for slot_role, condition in cell.conditions.items():
        rows = rows[rows[f"condition.{slot_role}"] == condition]
    if rows.empty:
        raise ValueError(f"cell {name!r} has no rows")
    return tasks_of(rows, params, prepared.index, f"cell {name!r}")


def tasks_of(
    rows: pd.DataFrame, params: GroupedContrasts, index: Mapping[str, Mapping[str, int]], what: str
) -> dict[str, Task]:
    """Return the rows per task of the analysis's groups (unlabelled rows left out for AUROC), on ``index``.

    The rows carry a ``unit.<factor>`` column per factor of ``index``.

    Raises:
        ValueError: if a task has no rows, or a row's unit is not in ``index``.
    """
    return _tasks(rows, sorted({t for g in params.groups for t in g.tasks}), params.metric, index, what)


def _group_value(tasks: Mapping[str, Task], group_tasks: Sequence[str], metric: str) -> float:
    """Return a group's equal-weight mean over its tasks on the full sample."""
    return float(np.mean([_task_stat(tasks[t], metric, None) for t in group_tasks]))


def group_draws(
    tasks: Mapping[str, Task], group_tasks: Sequence[str], metric: str, n_units: Mapping[str, int]
) -> np.ndarray:
    """Return a group's equal-weight mean over its tasks on every draw (undefined where a task is, never fewer).

    A one-task group is the task's own draws (:func:`task_draws`). A group of several tasks resamples the units jointly
    (one unstratified draw matrix shared by its tasks), so its interval keeps the correlation between tasks that share
    objects; a draw in which a task lacks a class is undefined by design.
    """
    if len(group_tasks) == 1:
        return task_draws(tasks[group_tasks[0]], metric, n_units)
    boot = Bootstrap(n_units)
    return np.mean([boot.draws(partial(_task_stat, tasks[t], metric)) for t in group_tasks], axis=0)


def parts_value(
    tasks: Mapping[str, Task], parts: Sequence[Sequence[str]], metric: str, n_units: Mapping[str, int]
) -> tuple[float, np.ndarray]:
    """Return a mean of strata (a group with ``parts``): the equal-weight mean of its parts' task means, with its draws.

    Every task is drawn on the one unstratified draw matrix (the rule of a group of several tasks, :func:`group_draws`),
    so a draw is one resample of the shared units for every part and every cell, and a contrast between cells is
    paired; a draw in which a task is undefined leaves the mean undefined.
    """
    boot = Bootstrap(n_units)
    draws = [np.mean([boot.draws(partial(_task_stat, tasks[t], metric)) for t in p], axis=0) for p in parts]
    value = float(np.mean([_group_value(tasks, p, metric) for p in parts]))
    return value, np.mean(draws, axis=0)


def task_draws(task: Task, metric: str, n_units: Mapping[str, int]) -> np.ndarray:
    """Return one task's statistic on every bootstrap draw over ``n_units`` (factor → number of units).

    An AUROC's draws resample the task's positive and negative units separately; a mean's use the one
    unstratified draw matrix (as a group of several tasks does, :func:`group_draws`).

    Raises:
        ValueError: for an AUROC, if fewer than two units of a factor hold a positive or a negative row.
    """
    strata = class_strata(task.units, task.labels, n_units) if metric == "auroc" else None
    return Bootstrap(n_units, strata=strata).draws(partial(_task_stat, task, metric))


def task_estimate(
    rows: pd.DataFrame, metric: Literal["auroc", "mean"], columns: Mapping[str, Sequence[str]]
) -> Estimate:
    """Return one task's statistic over score-table rows with its interval (the kind's per-task computation).

    Args:
        rows: the task's score rows.
        metric: ``auroc`` (unlabelled rows left out) or ``mean``.
        columns: factor → per-item unit value (:func:`unit_columns`), joined by ``item``.

    Raises:
        ValueError: if no rows remain (for ``auroc``: no labeled rows), the statistic has no value on the full
            sample (e.g. one label class), as for a cell's task (:func:`cell_draws`), or (``auroc``) fewer than two
            units hold either class.
    """
    if metric == "auroc":
        rows = rows[rows["label"].notna()]
    if rows.empty:
        raise ValueError("no rows to estimate from")
    rows = rows.assign(**{_UNIT + f: rows["item"].map(lambda i, c=c: c[i]) for f, c in columns.items()})
    index = _unit_index({"rows": rows}, tuple(columns))
    task = _checked_task("the rows", _task(rows, index), metric)
    draws = task_draws(task, metric, {f: len(ix) for f, ix in index.items()})
    return Estimate.of(summarize(draws, _task_stat(task, metric, None)))


@dataclass(frozen=True)
class CellDraws:
    """One cell's statistics on the full sample and on every draw of a bootstrap shared with the other cells.

    Attributes:
        tasks: task → (its statistic, its draws), thin tasks left out.
        mean: (the equal-weight mean over the tasks kept, its draws), the grouped contrasts' task mean.
        unavailable: thin task → its support (:func:`split_thin`).
    """

    tasks: dict[str, tuple[float, np.ndarray]]
    mean: tuple[float, np.ndarray]
    unavailable: dict[str, Unavailable]


def cell_draws(
    tables: Mapping[str, pd.DataFrame],
    tasks: Sequence[str],
    metric: Literal["auroc", "mean"],
    columns: Mapping[str, Sequence[str]],
    min_class: int | None = None,
) -> dict[str, CellDraws]:
    """Return each cell's per-task and mean statistics with draws from one bootstrap over every cell's units.

    One unit index covers every cell and task (the grouped contrasts' rule): a task's draws are :func:`task_draws`
    and the mean's are :func:`group_draws` over it, so a per-draw difference between cells is paired; a mean over
    tasks is undefined in a draw where a task is. Thin tasks are left out of a cell and listed.

    Args:
        tables: cell → its score rows (``item``, ``task``, ``score``, ``label``), items of one item set.
        tasks: the tasks of every cell.
        metric: ``auroc`` (unlabelled rows left out) or ``mean``.
        columns: factor → per-item unit value (:func:`unit_columns`), joined by ``item``.
        min_class: the thin-cell rule's minimum per class (:func:`is_thin`): None (:data:`THIN_CELL`) for paper
            cells; 0 for a layer-selection tool, whose population is checked apart (``evaluation.layer_sets``).

    Raises:
        ValueError: if a cell has no rows for a task, every task of a cell is thin, or a task kept has no value on
            the full sample or (``auroc``) fewer than two units of either class.
    """
    keyed = {
        cell: t.assign(**{_UNIT + f: t["item"].map(lambda i, c=c: c[i]) for f, c in columns.items()})
        for cell, t in tables.items()
    }
    index = _unit_index(keyed, tuple(columns))
    n_units = {f: len(ix) for f, ix in index.items()}
    out = {}
    for cell, table in keyed.items():
        what = f"cell {cell!r}"
        members, thin = split_thin(_tasks(table, tasks, metric, index, what), metric, min_class)
        if not members:
            raise ValueError(f"{what}: every task is thin: {sorted(thin)}")
        members = _valued(members, metric, what)
        per_task = {t: (_task_stat(m, metric, None), task_draws(m, metric, n_units)) for t, m in members.items()}
        mean = (_group_value(members, list(members), metric), group_draws(members, list(members), metric, n_units))
        out[cell] = CellDraws(per_task, mean, thin)
    return out


def unit_columns(items: ItemSet, factors: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Return factor → the unit of every item (in item order): an item-set field, else a role's objects.

    Raises:
        KeyError: if a factor is neither a field nor a role of the item set.
    """
    out = {}
    for f in factors:
        if f in items.fields:
            out[f] = tuple(str(v) for v in items.fields[f])
        elif f in items.objects:
            out[f] = tuple(items.objects[f])
        else:
            raise KeyError(f"resampling factor {f!r} is not a column of item set {items.name!r}")
    return out


def _cell_inputs(cell: Cell) -> list[str]:
    """Return the input keys a cell reads, in its declared order."""
    return list(cell.inputs)


def _check_inputs(params: GroupedContrasts, inputs: Mapping[str, object]) -> None:
    missing = sorted({(n, k) for n, c in params.cells.items() for k in _cell_inputs(c) if k not in inputs})
    if missing:
        raise KeyError(f"cells read inputs the analysis does not declare: {missing}")
    read = {k for c in params.cells.values() for k in _cell_inputs(c)}
    unread = sorted(set(inputs) - read)
    if unread:
        raise ValueError(f"inputs {unread} are read by no cell")


def _item_field(items: ItemSet, field: str) -> tuple[object, ...]:
    if field not in items.fields:
        raise KeyError(f"item filter field {field!r} is not a field of item set {items.name!r}")
    return tuple(items.fields[field])


def suite_of(record: Record) -> Mapping[str, object]:
    """Return the suite config of a suite record (its ``id`` and ``item_set`` among the keys).

    Raises:
        ValueError: if the record is not a suite run (it names no item set).
    """
    suite = record.config.get("suite")
    if not isinstance(suite, dict) or "item_set" not in suite or "id" not in suite:
        raise ValueError(f"record {record.record_id()[:12]} ({record.kind}) is not a suite run; it names no item set")
    return suite


def _task(rows: pd.DataFrame, index: Mapping[str, Mapping[str, int]]) -> Task:
    mapped = {f: rows[_UNIT + f].map(ix) for f, ix in index.items()}
    missing = sorted({f for f, m in mapped.items() if m.isna().any()})
    if missing:
        raise ValueError(f"rows with units outside the index for factors {missing}")
    unit_ix = {f: m.to_numpy(dtype=np.int64) for f, m in mapped.items()}
    labels = rows["label"].to_numpy(np.float64) > 0
    return Task(rows["item"].to_numpy(np.int64), rows["score"].to_numpy(np.float64), labels, unit_ix)


def _tasks(
    rows: pd.DataFrame, tasks: Sequence[str], metric: str, index: Mapping[str, Mapping[str, int]], what: str
) -> dict[str, Task]:
    """Return the rows per task (unlabelled rows left out for ``auroc``), on ``index``: the per-cell core.

    Raises:
        ValueError: if a task has no rows, or a row's unit is not in ``index``.
    """
    out = {}
    for task in tasks:
        t = rows[rows["task"] == task]
        if metric == "auroc":
            t = t[t["label"].notna()]
        if t.empty:
            raise ValueError(f"{what}: task {task!r} has no rows")
        out[task] = _task(t, index)
    return out


def _valued(tasks: dict[str, Task], metric: str, what: str) -> dict[str, Task]:
    """Return ``tasks``, refusing one without a value on the full sample (:func:`_checked_task`): never dropped.

    Raises:
        ValueError: if a task's statistic is NaN on the full sample.
    """
    for name, task in tasks.items():
        _checked_task(f"{what}: task {name!r}", task, metric)
    return tasks


def _checked_task(where: str, task: Task, metric: str) -> Task:
    """Return ``task``, refusing one without a value on the full sample (the one refusal of every path).

    Raises:
        ValueError: naming ``where``, if the statistic is NaN on the full sample (e.g. one label class).
    """
    if np.isnan(_task_stat(task, metric, None)):
        raise ValueError(f"{where} has no {metric} on the full sample (one label class?)")
    return task


def _unit_index(tables: Mapping[str, pd.DataFrame], factors: tuple[str, ...]) -> dict[str, dict[str, int]]:
    index = {}
    for f in factors:
        index[f] = unit_index(v for t in tables.values() for v in t[_UNIT + f].dropna())
    return index


def _weights(units: Mapping[str, np.ndarray], n: int, draw: Mapping[str, np.ndarray] | None) -> np.ndarray:
    """Each row's multiplicity in a draw: the product over factors of its unit's count (1 without a draw)."""
    weights = np.ones(n, dtype=np.int64)
    if draw is not None:
        for f, ix in units.items():
            weights *= np.bincount(draw[f], minlength=int(ix.max()) + 1)[ix]
    return weights


def _task_stat(task: Task, metric: str, draw: Mapping[str, np.ndarray] | None) -> float:
    weights = _weights(task.units, task.scores.size, draw)
    if metric == "mean":
        total = float(weights.sum())
        return float((task.scores * weights).sum()) / total if total else float("nan")
    return auroc_counts(task.scores, task.labels, weights)
