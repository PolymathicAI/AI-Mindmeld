"""Typed analysis results: what each analysis kind writes, the end of the pipeline.

Every analysis kind declares its ``Result`` (a subclass of :class:`AnalysisResult`); the ``analysis`` record's output
artifact holds it as JSON, and every reader (the site's review page, a figure drawn on request) loads it validated
(``evaluation.analysis.load_result``). Only the models the vertical slice needs are defined here; each further kind's
``Result`` is defined, and reviewed, when that kind is built (the kind table in ``evaluation.analysis`` lists them).

Shared pieces:

- :class:`Estimate`: a statistic with its bootstrap interval (``stats.Bootstrap``); a paired contrast is an
  :class:`Estimate` of the difference ``a − b`` with its own interval (estimates, not judgements);
- :class:`Unavailable`: a thin cell's support (a task with fewer than ``evaluation.contrasts.THIN_CELL``
  positive or negative rows in a cell is left out of that cell's means and listed, never silently dropped);
  :meth:`AnalysisResult.unavailable_lines` lists a result's (the review page shows them in place);
- :class:`Support`: a task's labeled rows in a cell, the ``n`` every value is read with;
- :data:`FINITE`: the model configuration that refuses NaN and infinities ("A NaN is a bug");
  the kinds built since (``evaluation.counts``, ``evaluation.cost``, ``evaluation.drift``, ``evaluation.excerpts``)
  set it on their results and every model inside them, so an undefined value fails where it is made, never in a
  reader.
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from latent_interconnect.config import Config, added_field
from latent_interconnect.stats.bootstrap import Interval, IntervalFlag, ResamplingUnits

__all__ = [
    "FINITE",
    "AnalysisResult",
    "Cell",
    "ContrastSpec",
    "Estimate",
    "GroupPoint",
    "GroupedContrastsResult",
    "Support",
    "Unavailable",
]


FINITE = ConfigDict(allow_inf_nan=False)
"""``model_config`` of a result model that refuses NaN and infinities (merged with ``Config``'s)."""


class Cell(Config):
    """One cell of a ``grouped_contrasts`` analysis: the rows of named inputs at one step under given conditions.

    Declared by name in the analysis parameters and recorded with the result, so a reader takes a cell's encoder,
    inputs, step and conditions from these fields, never from its name (a free-form label).

    Attributes:
        encoder: the encoder whose connector produced the rows (the encoder the input runs are pinned under); None
            for rows no connector produced (the name references). A label for consumers; the statistic does not use
            it.
        inputs: the analysis inputs (keys of ``AnalysisSpec.inputs``) whose score rows the cell reads, concatenated;
            several when a cell draws on several suites of one connector and step (Fig 7b's held-out tasks and its
            PFMbench localization column). Distinct, with disjoint tasks.
        step: the checkpoint step of the rows; None for rows of a suite that reads no connector (the name
            references, ``SuiteRunConfig.step``).
        conditions: slot role → condition of the rows (e.g. ``{object: own}``).
        receiver: the receiver that scored the rows, a label for consumers like ``encoder``, set where one analysis
            lists cells of several receivers (an aggregate analysis that pools cells across receivers); empty
            where the analysis reads one receiver. An added field: empty in every config and record written before
            it.
    """

    encoder: str | None = Field(default=None, min_length=1)
    inputs: tuple[str, ...] = Field(min_length=1)
    step: int | None = None
    conditions: dict[str, str] = Field(min_length=1)
    receiver: str = added_field(default="")

    @model_serializer(mode="wrap")
    def _omit_an_empty_receiver(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Leave ``receiver`` out while empty, so a cell without one is stored as before."""
        out: dict[str, Any] = handler(self)
        if not out.get("receiver"):
            out.pop("receiver", None)
        return out

    @model_validator(mode="after")
    def _distinct(self) -> Cell:
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError(f"a cell reads each input once, not {list(self.inputs)}")
        return self


class Estimate(Config):
    """A statistic with its interval; every value finite (a NaN is a bug).

    The interval is the bootstrap's percentile interval unless the analysis selected the shared-unit interval
    (``ResamplingUnits.interval``, ``stats.shared_units``).

    Attributes:
        estimate: the statistic on the original sample.
        lo: lower bound of the interval (``stats.LEVEL``).
        hi: upper bound.
        dropped: bootstrap draws left out as undefined by design (``stats.bootstrap.summarize``); 0 for a shared-unit
            interval.
        interval_flag: empty for an ordinary interval. For a shared-unit interval that is a fallback:
            ``"full_range"`` (the sample is degenerate, e.g. a constant statistic: the interval is the statistic's
            whole range and is uninformative) or ``"independent_variance"`` (the shared-unit variance was not
            positive). No other value loads. A defined estimate either way: an estimate with no value is refused,
            never flagged. An added field: empty in every record written before it.
    """

    model_config = FINITE

    estimate: float
    lo: float
    hi: float
    dropped: int = Field(ge=0)
    interval_flag: IntervalFlag = added_field(default="")

    @model_serializer(mode="wrap")
    def _omit_an_empty_flag(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Leave ``interval_flag`` out while empty, so a result without one is stored byte for byte as before."""
        out: dict[str, Any] = handler(self)
        if not out.get("interval_flag"):
            out.pop("interval_flag", None)
        return out

    @classmethod
    def of(cls, interval: Interval) -> Estimate:
        """Return the estimate of a ``stats.Interval``."""
        return cls(
            estimate=interval.estimate,
            lo=interval.lo,
            hi=interval.hi,
            dropped=interval.dropped,
            interval_flag=interval.flag,
        )


class Unavailable(Config):
    """The support of a task left out of a cell's means as thin.

    Attributes:
        n: the task's labeled rows in the cell.
        positives: its positive rows.
        negatives: its negative rows.
        pairs: its within-stratum (positive, negative) pairs, for a ``stratified_contrast`` task left out for having
            none (None elsewhere).
        clusters: its distinct units of the floor's factor, for a ``grouped_contrasts`` task left out under the
            analysis's support floor (``GroupedContrasts.min_support``; None elsewhere).
        point: that task's statistic on the full sample: under the support floor the point is kept and only the
            interval (and the task's part in the cell's group means and contrasts) withheld; None when undefined or
            for any other entry.
    """

    n: int = Field(ge=0)
    positives: int = Field(ge=0)
    negatives: int = Field(ge=0)
    pairs: int | None = added_field(default=None, ge=0)
    clusters: int | None = added_field(default=None, ge=0)
    point: float | None = added_field(default=None)

    def text(self) -> str:
        """Return ``unavailable (n=200: 7 positives, 193 negatives)``, as the review page prints it.

        ``; 0 within-stratum pairs`` is added before the parenthesis when ``pairs`` is set; a task under a support floor
        reads ``interval unavailable (n=9 items in 6 clusters: under the support floor; point 0.612)``.
        """
        if self.clusters is not None:
            point = "" if self.point is None else f"; point {self.point:.3f}"
            return (
                f"interval unavailable (n={self.n} items in {self.clusters} clusters: under the support floor{point})"
            )
        pairs = "" if self.pairs is None else f"; {self.pairs} within-stratum pairs"
        return f"unavailable (n={self.n}: {self.positives} positives, {self.negatives} negatives{pairs})"


class Support(Config):
    """The rows one task of one cell is computed from, after the analysis's object set and item filters.

    Attributes:
        n: the task's rows the statistic reads (under ``auroc`` its labeled rows; under ``mean`` every row).
        positives: its positive rows under ``auroc``; None under ``mean`` (no classes).
        negatives: its negative rows under ``auroc``; None under ``mean``.
    """

    n: int = Field(ge=0)
    positives: int | None = Field(default=None, ge=0)
    negatives: int | None = Field(default=None, ge=0)


class AnalysisResult(Config):
    """Base of every kind's result.

    Attributes:
        units: the resampling units the intervals used.
    """

    units: ResamplingUnits = ResamplingUnits()

    def companion_files(self) -> dict[str, str]:
        """Return file name → text of the readable files written beside the result (``excerpts``' candidates).

        They render the result for a person; nothing reads them back. None by default.
        """
        return {}

    def unavailable_lines(self) -> list[str]:
        """Return one line per thin cell left out of the result's means; none by default."""
        return []


class ContrastSpec(Config):
    """What one reported contrast of a ``grouped_contrasts`` result is: ``a − b`` on one group.

    Recorded with the result, so a reader selects a contrast by its cells' definitions and its group, never by its
    name (a free-form label, suffixed ``/<group>`` when the analysis has several groups).

    Attributes:
        a: the first cell (a key of ``cells``), mean (a key of ``means``) or earlier contrast (its declared name, a
            key of ``contrast_specs`` without the ``/<group>`` suffix).
        b: the second cell, mean or earlier contrast.
        group: the group the contrast is taken on.
    """

    a: str
    b: str
    group: str


class GroupPoint(Config):
    """A one-task group's point where its task is under the analysis's support floor (``GroupedContrasts.min_support``).

    Attributes:
        point: the task's statistic on the full sample, the group's value (no interval).
        n: the task's scored items.
        clusters: its distinct units of the floor's factor.
        unavailable: why there is no interval (``under the support floor: <n> items, <k> clusters``).
    """

    point: float
    n: int = Field(ge=0)
    clusters: int = Field(ge=0)
    unavailable: str


class GroupedContrastsResult(AnalysisResult):
    """Result of ``grouped_contrasts``.

    Attributes:
        cells: cell name → group name → equal-weight mean over the group's tasks.
        cell_specs: cell name → the cell's definition (encoder, inputs, step and conditions); the same names as
            ``cells``.
        means: mean name → group name → the equal-weight mean of its cells' values, with the percentile interval of
            the per-draw mean (``GroupedContrasts.means``); a group a mean is not reported on is
            absent (a cell lacks it, or its cells leave out different thin tasks there). Empty, it is not stored, so
            a result without means is stored as before.
        mean_specs: mean name → the cells it averages; the same names as ``means``.
        contrasts: contrast name → the paired difference ``a − b`` with its interval.
        contrast_specs: contrast name → its definition (cells, means or earlier contrasts, and group); the same names as
            ``contrasts``.
        unavailable: cell name → task → its support, for each task left out of the cell's means as thin. A
            group all of whose tasks are thin has no value in ``cells``, and a contrast on a group is not reported
            when its two cells leave out different tasks there (``evaluation.contrasts.grouped_result``).
        support: cell name → task (``cross_domain_ranks``: endpoint) → the rows it is computed from (every task of
            the cell's groups, thin ones too), so a group's ``n`` is read per task, never summed over tasks
            that share objects. Empty in records written before it was added.
        points: cell name → group name → :class:`GroupPoint`, for each group of exactly one task whose task the
            support floor left out with a defined point: the group's point without an interval. Such
            a group is in no mean or contrast; a group of several tasks never has one. Empty, it is not stored.
        task_aliases: input key → stored task → the task name it was read under (``GroupedContrasts.task_aliases``),
            as applied; the stored score rows keep their names. Empty, it is not stored.
        object_set: the item set whose objects every input was restricted to (``GroupedContrasts.object_set``), as
            applied. None, it is not stored.
    """

    cells: dict[str, dict[str, Estimate]]
    cell_specs: dict[str, Cell]
    means: dict[str, dict[str, Estimate]] = Field(default_factory=dict)
    mean_specs: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    contrasts: dict[str, Estimate] = Field(default_factory=dict)
    contrast_specs: dict[str, ContrastSpec] = Field(default_factory=dict)
    unavailable: dict[str, dict[str, Unavailable]] = Field(default_factory=dict)
    support: dict[str, dict[str, Support]] = added_field(default_factory=dict)
    points: dict[str, dict[str, GroupPoint]] = Field(default_factory=dict)
    task_aliases: dict[str, dict[str, str]] = added_field(default_factory=dict)
    object_set: str | None = added_field(default=None)

    @model_serializer(mode="wrap")
    def _omit_empty_means(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Leave ``means``, ``mean_specs``, ``points``, ``task_aliases``, ``object_set`` out while empty: as before."""
        out: dict[str, Any] = handler(self)
        for key in ("means", "mean_specs", "points", "task_aliases", "object_set"):
            if not out.get(key):
                out.pop(key, None)
        return out

    def unavailable_lines(self) -> list[str]:
        """Return ``cell 'x', task 'y': unavailable (n=..: .. positives, .. negatives)`` per thin cell."""
        return [
            f"cell {c!r}, task {t!r}: {u.text()}"
            for c, tasks in sorted(self.unavailable.items())
            for t, u in sorted(tasks.items())
        ]

    @model_validator(mode="after")
    def _same_cells(self) -> GroupedContrastsResult:
        if set(self.cell_specs) != set(self.cells):
            raise ValueError(f"cell_specs name {sorted(self.cell_specs)}, but the values name {sorted(self.cells)}")
        if not set(self.support) <= set(self.cells):
            raise ValueError(f"support cells {sorted(set(self.support) - set(self.cells))} are not in the result")
        if not set(self.unavailable) <= set(self.cells):
            raise ValueError(
                f"unavailable cells {sorted(set(self.unavailable) - set(self.cells))} are not in the result"
            )
        if set(self.mean_specs) != set(self.means):
            raise ValueError(f"mean_specs name {sorted(self.mean_specs)}, but the means name {sorted(self.means)}")
        for name, members in self.mean_specs.items():
            if name in self.cells or not set(members) <= set(self.cells):
                raise ValueError(f"mean {name!r}: named as a cell, or cells {list(members)} are not in the result")
        if set(self.contrast_specs) != set(self.contrasts):
            raise ValueError(
                f"contrast_specs name {sorted(self.contrast_specs)}, but the contrasts name {sorted(self.contrasts)}"
            )
        on = {(x, g) for x, by_group in (self.cells | self.means).items() for g in by_group}
        for name, c in self.contrast_specs.items():  # in order: a contrast may cite an earlier one by its declared name
            missing = sorted({c.a, c.b} - {x for x, g in on if g == c.group})
            if missing:
                raise ValueError(f"contrast {name!r}: cells {missing} or group {c.group!r} are not in the result")
            on |= {(name, c.group), (name.removesuffix(f"/{c.group}"), c.group)}
        return self
