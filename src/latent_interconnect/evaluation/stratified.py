"""The ``stratified_contrast`` analysis kind: within-stratum concordance, pooled over strata (Fig 10b).

Fig 10b asks whether the embedding orders proteins *within* a group whose name input is identical (lineage questions,
where the name alone cannot rank them). Its statistic is the paper's:

- per task, the *within-stratum concordance*: over every (positive, negative) pair of rows in the same stratum, the
  share the score orders correctly (ties count one half), pooled over strata by pairs; a task with no such pair
  (undefined on the full sample, e.g. no name shared by a positive and a negative) is left out of the cell's means
  and listed as unavailable with its support and ``pairs=0``, as thin tasks are;
- per group, the equal-weight mean over its tasks, and paired contrasts between cells (``grouped_result``, as
  ``grouped_contrasts``).

The strata are the analysis's one resampling factor (``units``; e.g. ``name.object``, the literal name input,
``data.items.with_names``), and the bootstrap resamples them unstratified by class, as the paper's group bootstrap did.
A draw in which no stratum holding a pair of a task is drawn leaves that task, and its groups, undefined in that draw:
undefined by design, dropped and counted (``stats.bootstrap.summarize``). The parameters are
:class:`~latent_interconnect.evaluation.analysis.GroupedContrasts` with ``metric: auroc`` (a within-stratum AUROC).
The result also counts, per cell and group, the informative strata (holding a pair for some task of the group) and
their objects (the manuscript's ``numbers_lineage_display``), and per cell and task its rows (``support``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import ClassVar

import numpy as np
from pydantic import Field

from latent_interconnect.artifacts.records import Record
from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.config import Config
from latent_interconnect.data.items import ItemSet
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS, GroupedContrasts
from latent_interconnect.evaluation.contrasts import (
    Task,
    cell_tasks,
    grouped_result,
    prepare,
    split_thin,
    support_of,
)
from latent_interconnect.evaluation.results import AnalysisResult, GroupedContrastsResult, Support, Unavailable
from latent_interconnect.stats.bootstrap import Bootstrap, ResamplingUnits

__all__ = ["StratifiedContrastKind", "StratifiedContrastResult", "StratumCounts"]


class StratumCounts(Config):
    """The informative strata of one cell and group.

    Attributes:
        strata: strata holding a (positive, negative) pair for some task of the group.
        objects: the distinct items of the cell's rows in those strata.
    """

    strata: int = Field(ge=1)
    objects: int = Field(ge=2)


class StratifiedContrastResult(GroupedContrastsResult):
    """Result of ``stratified_contrast``: a ``grouped_contrasts`` result of within-stratum concordances.

    Attributes:
        informative: cell → group → its informative strata and their objects.
    """

    informative: dict[str, dict[str, StratumCounts]]


@ANALYSIS_KINDS.register("stratified_contrast")
class StratifiedContrastKind:
    """Within-stratum concordance per task, equal-weight groups and paired contrasts (params ``GroupedContrasts``)."""

    Params: ClassVar[type[Config]] = GroupedContrasts
    Result: ClassVar[type[AnalysisResult]] = StratifiedContrastResult

    def run(
        self,
        params: GroupedContrasts,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> StratifiedContrastResult:
        """Compute every cell's group concordances and every contrast over the resampled strata.

        Raises:
            ValueError: if ``units`` is not one factor (the strata), the metric is not ``auroc``, a group has
                ``parts``, or as
                ``grouped_contrasts`` (e.g. a cell or task without rows).
        """
        if len(units.factors) != 1 or params.metric != "auroc":
            raise ValueError(f"stratified_contrast needs one stratum factor and metric auroc, got {units.factors}")
        if any(g.parts for g in params.groups):
            raise ValueError("stratified_contrast pools tasks only; a group of parts is a grouped_contrasts mean")
        (factor,) = units.factors
        prepared = prepare(params, units, inputs, artifacts, items)
        n = prepared.n_units[factor]
        counts = np.stack([np.bincount(ix, minlength=n) for ix in Bootstrap(prepared.n_units).indices[factor]])
        per_group: dict[str, dict[str, tuple[float, np.ndarray]]] = {}
        informative: dict[str, dict[str, StratumCounts]] = {}
        unavailable: dict[str, dict[str, Unavailable]] = {}
        support: dict[str, dict[str, Support]] = {}
        for c in params.cells:
            rows = cell_tasks(prepared, params, c)
            support[c] = support_of(rows, params.metric)
            tasks, thin = split_thin(rows, params.metric)
            pairs = {t: _pairs(task, factor) for t, task in tasks.items()}
            for t in [t for t, (v, _) in pairs.items() if not v.size]:  # no within-stratum pair: unavailable too
                y = tasks.pop(t).labels
                thin[t] = Unavailable(n=y.size, positives=int(y.sum()), negatives=int((~y).sum()), pairs=0)
                del pairs[t]
            if thin:
                unavailable[c] = thin
            per_task = {t: _concordance(v, s, counts) for t, (v, s) in pairs.items()}
            per_group[c] = {}
            informative[c] = {}
            for g in params.groups:
                kept = [t for t in g.tasks if t in tasks]
                if not kept:
                    continue
                per_group[c][g.name] = (
                    float(np.mean([per_task[t][0] for t in kept])),
                    np.mean([per_task[t][1] for t in kept], axis=0),
                )
                strata = np.unique(np.concatenate([pairs[t][1] for t in kept]))
                objects = {i for t in kept for i in tasks[t].items[np.isin(tasks[t].units[factor], strata)]}
                informative[c][g.name] = StratumCounts(strata=strata.size, objects=len(objects))
        grouped = grouped_result(params, units, per_group, unavailable, support)
        return StratifiedContrastResult(**dict(grouped), informative=informative)


def _pairs(task: Task, factor: str) -> tuple[np.ndarray, np.ndarray]:
    """Return each within-stratum (positive, negative) pair's concordance (1, ½ for a tie, 0) and its stratum.

    Both arrays are empty if the task has no such pair.
    """
    values, strata = [], []
    unit = task.units[factor]
    for s in np.unique(unit):
        rows = unit == s
        pos, neg = task.scores[rows & task.labels], task.scores[rows & ~task.labels]
        d = (pos[:, None] - neg[None, :]).ravel()
        values.append(np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0)))
        strata.append(np.full(d.size, s))
    return np.concatenate(values), np.concatenate(strata)


def _concordance(values: np.ndarray, strata: np.ndarray, counts: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the pair-pooled concordance and its draws (NaN where no stratum holding a pair is drawn)."""
    n = counts.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (counts @ np.bincount(strata, values, minlength=n)) / (counts @ np.bincount(strata, minlength=n))
    return float(values.mean()), draws
