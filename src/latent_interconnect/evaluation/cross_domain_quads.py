"""The ``cross_domain_quads`` analysis kind: E1 over swap quads, the crossed bootstrap, and paired contrasts of cells.

**E1** (the frozen protocol's primary endpoint). In one cell (``evaluation.results.Cell``: the
rows of its inputs at one step under one condition per slot role; its name is a label, never parsed), quad q's
statistic ``Q_q``
is the mean over its four comparisons of ``1[s(positive) > s(swap)]``, ties ½: the AUROC of its four scores
(``stats.auroc``). E1 is the unweighted mean of ``Q_q`` over the quads. Any single-slot score gives exactly 0.5
(``evaluation.cross_domain``), so E1 above 0.5 is what joint use of both slots would show.

**Crossed bootstrap** (replaces the frozen design's jackknife). The resampling factors (``AnalysisSpec.units``) are
item-set columns, normally ``protein_cluster`` and ``molecule_group`` (written with the item set): a quad's units
are the distinct values of each factor over its four items. Each factor's units are resampled independently by the
one bootstrap (``stats.Bootstrap``); in a draw, a quad weighs the product of its units' draw counts (the pigeonhole
weighting of the frozen protocol's ``prod`` scheme), and E1 is the weighted mean of the quad statistics. A draw in
which every quad weighs 0 has no E1: the frozen code drew again, the one fixed scheme cannot, so such draws are dropped
and counted (``CrossDomainQuadsResult.empty_draws``). Every cell sees the same draws, so a contrast's interval is the
percentile interval of the per-draw difference.

**Shared-unit interval** (``units.interval: shared_unit``; ``stats.shared_units``). An opt-in
alternative to the crossed bootstrap, which stays the default; an analysis config selects this interval by name.
The observations are the
quads, their influence values ``(Q_q − E1)/n`` (for a contrast, the difference of the two cells' quad statistics), their
units the quad's units as above, and the statistic's range ``[0, 1]``. The point estimates are the same; no draw is
made (``empty_draws`` is 0); an interval that is a fallback is marked (``Estimate.interval_flag``). Empirically
conservative and approximately 95%, with one unresolved shortfall: the module docstring of ``stats.shared_units``.

**Estimates only.** The kind reports each cell's E1 and each declared paired contrast with its
interval (``stats.LEVEL``); it holds no test, gate, margin or p-value. The statistic is not ``grouped_contrasts``' (a
mean of per-quad AUROCs weighted per quad, not a pooled AUROC per task), so it keeps its own kind.

**Legacy name.** Configs and records written before this name hold kind ``cross_domain_e1``; they validate, run and
load as this kind with their ``kind`` unchanged (``evaluation.analysis.LEGACY_ANALYSIS_KINDS``).

**Refusals.** Every input must be a ``cross_domain`` score table on one item set (contrasts are paired over it); a
cell must hold exactly one score per item, of one task (scores are finite: ``scores.write_scores``); each factor
needs at least two units.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.records import Record, record_output
from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.config import Config, added_field
from latent_interconnect.data.items import ItemSet
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS, Contrast
from latent_interconnect.evaluation.contrasts import suite_of, unit_columns
from latent_interconnect.evaluation.cross_domain import QUAD_SIZE, swap_quads
from latent_interconnect.evaluation.results import AnalysisResult, Cell, Estimate
from latent_interconnect.evaluation.scores import SCORES_KIND, read_scores, score_roles
from latent_interconnect.stats.auroc import auroc
from latent_interconnect.stats.bootstrap import Bootstrap, Interval, ResamplingUnits, summarize
from latent_interconnect.stats.shared_units import SHARED_UNIT, shared_unit_interval

__all__ = ["CrossDomainQuads", "CrossDomainQuadsKind", "CrossDomainQuadsResult", "QuadDesign", "cell_rows"]

_MIN_UNITS = 2
_ROLES = 2  # a swap quad pairs two roles


class CrossDomainQuads(Config):
    """Parameters of the ``cross_domain_quads`` kind.

    Attributes:
        cells: cell name → :class:`~latent_interconnect.evaluation.results.Cell` (encoder, inputs, step,
            conditions), the cells whose E1 is reported; the name is a label (results and contrasts cite it).
        contrasts: paired contrasts ``a − b`` between listed cells.
    """

    cells: dict[str, Cell] = Field(min_length=1)
    contrasts: tuple[Contrast, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> CrossDomainQuads:
        unknown = sorted({c for x in self.contrasts for c in (x.a, x.b)} - set(self.cells))
        if unknown:
            raise ValueError(f"cells {unknown} are not among the listed cells")
        names = [x.name for x in self.contrasts]
        if len(names) != len(set(names)):
            raise ValueError(f"contrast names repeat: {sorted(names)}")
        return self


class CrossDomainQuadsResult(AnalysisResult):
    """Result of ``cross_domain_quads``.

    Attributes:
        quads: quads in the item set.
        unit_counts: factor → its units in the item set.
        empty_draws: bootstrap draws in which every quad weighed 0 (dropped from every interval).
        cells: cell → E1 with its interval.
        contrasts: contrast name → the paired difference with its interval.
        item_set: the scored item set's name (``<build>/<cell>`` for a cross-domain build), so a reader checks that
            a record is of the cell its role names; empty in a record written before it was recorded.
    """

    quads: int
    unit_counts: dict[str, int]
    empty_draws: int
    cells: dict[str, Estimate]
    contrasts: dict[str, Estimate] = Field(default_factory=dict)
    item_set: str = added_field(default="")


@dataclass(frozen=True)
class QuadDesign:
    """The swap quads of an item set, or any grouping of its items (:meth:`from_groups`), and their crossed units.

    Attributes:
        quad: each item's quad (group) index.
        units: factor → ``(quads, m)`` unit indices of each quad's distinct units, padded with ``unit_counts[f]``.
        unit_counts: factor → number of units.
    """

    quad: np.ndarray
    units: Mapping[str, np.ndarray]
    unit_counts: Mapping[str, int]

    @classmethod
    def of(cls, items: ItemSet, roles: tuple[str, str], factors: tuple[str, ...]) -> QuadDesign:
        """Return the design of an item set (its quads checked by ``evaluation.cross_domain.swap_quads``).

        Raises:
            KeyError: if a factor is not a column of the item set.
            ValueError: if the set breaks the swap-quad structure or a factor has fewer than two units.
        """
        return cls.from_groups(items, swap_quads(items, roles), factors)

    @classmethod
    def from_groups(cls, items: ItemSet, quad: np.ndarray, factors: tuple[str, ...]) -> QuadDesign:
        """Return the design of the item groups ``quad`` (each item's group index, 0 to the number of groups − 1).

        Raises:
            KeyError: if a factor is not a column of the item set.
            ValueError: if a factor has fewer than two units.
        """
        n_quads = int(quad.max()) + 1
        units, counts = {}, {}
        for f, column in unit_columns(items, factors).items():
            index = {u: k for k, u in enumerate(sorted(set(column)))}
            if len(index) < _MIN_UNITS:
                raise ValueError(f"factor {f!r} has {len(index)} unit(s); the bootstrap needs at least {_MIN_UNITS}")
            per_quad: list[set[int]] = [set() for _ in range(n_quads)]
            for q, u in zip(quad, column, strict=True):
                per_quad[q].add(index[u])
            width = max(len(s) for s in per_quad)
            pad = len(index)
            units[f] = np.array([sorted(s) + [pad] * (width - len(s)) for s in per_quad], dtype=np.int64)
            counts[f] = len(index)
        return cls(quad=quad, units=units, unit_counts=counts)

    @property
    def n_quads(self) -> int:
        """Number of quads."""
        return int(self.quad.max()) + 1

    def weights(self, draw: Mapping[str, np.ndarray] | None) -> np.ndarray:
        """Return each quad's weight in one draw: the product of its units' draw counts (all 1 without a draw)."""
        w = np.ones(self.n_quads, dtype=np.float64)
        if draw is None:
            return w
        for f, idx in self.units.items():
            counts = np.append(np.bincount(draw[f], minlength=self.unit_counts[f]), 1)  # the padding slot weighs 1
            w *= counts[idx].prod(axis=1)
        return w

    def shared_unit(
        self, q: np.ndarray, estimate: float, bounds: tuple[float, float] = (0.0, 1.0), *, contrast: bool = False
    ) -> Interval:
        """Return ``estimate``, the mean of the group statistics ``q``, with its shared-unit interval.

        The groups are the observations and each group's distinct units of every factor are its units
        (``stats.shared_units``). ``bounds`` is the range of one group statistic; ``contrast`` says that ``q`` holds
        paired differences of two cells.
        """
        influence = (q - q.mean()) / q.size
        return shared_unit_interval(estimate, influence, self.units, self.unit_counts, bounds, contrast=contrast)

    def weighted_mean(self, q: np.ndarray, draw: Mapping[str, np.ndarray] | None) -> float:
        """Return the weighted mean of the quad statistics ``q`` (E1; NaN when every quad weighs 0).

        The weights are :meth:`weights` of ``draw`` (all 1 without a draw: the unweighted mean, the point estimate).
        """
        w = self.weights(draw)
        total = float(w.sum())
        return float(w @ q) / total if total else float("nan")


@ANALYSIS_KINDS.register("cross_domain_quads")
class CrossDomainQuadsKind:
    """E1 per cell and paired contrasts between cells, from shared crossed draws (params :class:`CrossDomainQuads`)."""

    Params: ClassVar[type[Config]] = CrossDomainQuads
    Result: ClassVar[type[AnalysisResult]] = CrossDomainQuadsResult
    interval_methods: ClassVar[tuple[str, ...]] = (SHARED_UNIT,)

    def run(
        self,
        params: CrossDomainQuads,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> CrossDomainQuadsResult:
        """Compute every cell's E1 and every contrast from the shared crossed draws.

        Raises:
            KeyError: if a cell names an unknown input, or a factor is not a column of the item set.
            ValueError: if the inputs are not suite records on one item set, the item set breaks the swap-quad
                structure, or a cell does not hold exactly one score per item of one task.
        """
        digests = {str(suite_of(rec)["item_set"]) for rec in inputs.values()}
        if len(digests) != 1:
            raise ValueError(f"cross_domain_quads needs every input on one item set, got {len(digests)}")
        tables = {role: read_scores(artifacts, record_output(rec, SCORES_KIND)) for role, rec in inputs.items()}
        roles = {score_roles(t) for t in tables.values()}
        if len(roles) != 1 or len(next(iter(roles))) != _ROLES:
            raise ValueError(f"cross_domain_quads needs two-role score tables of one shape, got roles {sorted(roles)}")
        item_set = items(digests.pop())
        pair = next(iter(roles))
        design = QuadDesign.of(item_set, (pair[0], pair[1]), units.factors)
        quad_stats = {c: _quad_stats(design, tables, c, spec) for c, spec in params.cells.items()}
        estimates = {c: design.weighted_mean(q, None) for c, q in quad_stats.items()}
        if units.interval == SHARED_UNIT:
            return CrossDomainQuadsResult(
                units=units,
                quads=design.n_quads,
                unit_counts=dict(design.unit_counts),
                empty_draws=0,
                cells={c: Estimate.of(design.shared_unit(q, estimates[c])) for c, q in quad_stats.items()},
                contrasts={
                    x.name: Estimate.of(
                        design.shared_unit(
                            quad_stats[x.a] - quad_stats[x.b], estimates[x.a] - estimates[x.b], contrast=True
                        )
                    )
                    for x in params.contrasts
                },
                item_set=item_set.name,
            )
        boot = Bootstrap(dict(design.unit_counts))
        draws = {c: boot.draws(partial(design.weighted_mean, q)) for c, q in quad_stats.items()}
        contrasts = {
            x.name: Estimate.of(summarize(draws[x.a] - draws[x.b], estimates[x.a] - estimates[x.b]))
            for x in params.contrasts
        }
        return CrossDomainQuadsResult(
            units=units,
            quads=design.n_quads,
            unit_counts=dict(design.unit_counts),
            empty_draws=int(np.isnan(next(iter(draws.values()))).sum()),
            cells={c: Estimate.of(summarize(draws[c], estimates[c])) for c in params.cells},
            contrasts=contrasts,
            item_set=item_set.name,
        )


def _quad_stats(design: QuadDesign, tables: Mapping[str, pd.DataFrame], cell: str, spec: Cell) -> np.ndarray:
    """Return each quad's statistic (the AUROC of its four scores) in the cell ``cell``, defined by ``spec``."""
    scores, labels = cell_rows(tables, cell, spec, design.quad.size)
    order = np.argsort(design.quad, kind="stable")
    by_quad = scores[order].reshape(design.n_quads, QUAD_SIZE)
    label_by_quad = labels[order].reshape(design.n_quads, QUAD_SIZE)
    return np.array([auroc(s, y) for s, y in zip(by_quad, label_by_quad, strict=True)])


def cell_rows(tables: Mapping[str, pd.DataFrame], cell: str, spec: Cell, n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the cell's scores and positive flags in item order, one row per item of the ``n_items``.

    Raises:
        ValueError: if the cell does not hold exactly one row per item, or mixes tasks.
    """
    rows = pd.concat([tables[key] for key in spec.inputs], ignore_index=True)
    rows = rows[rows["step"] == spec.step]
    for slot_role, condition in spec.conditions.items():
        rows = rows[rows[f"condition.{slot_role}"] == condition]
    if len(rows) != n_items or set(rows["item"]) != set(range(n_items)):
        raise ValueError(f"cell {cell!r} holds {len(rows)} rows, not one per item of the {n_items}")
    if rows["task"].nunique() != 1:
        raise ValueError(f"cell {cell!r} mixes tasks {sorted(rows['task'].unique())}")
    rows = rows.sort_values("item")
    return rows["score"].to_numpy(np.float64), rows["label"].to_numpy(np.float64) > 0
