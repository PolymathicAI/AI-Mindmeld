"""The bootstrap: one scheme everywhere, shared draws, paired contrasts.

The one other interval of the package is the shared-unit interval of a crossed design (``stats.shared_units``),
computed only when a config selects it (``ResamplingUnits.interval``); it is no part of this scheme.

The scheme is fixed, not configurable:

- multinomial draws (``rng.multinomial(n, 1/n)`` counts per draw, the paper's main-figure generator);
- :data:`DRAWS` draws, seed :data:`SEED`, :data:`LEVEL` percentile intervals.

What varies is only the *resampling units*, which follow the item set's structure. An item set's units are one or
more crossed *factors* (columns of the item set):

- ``object`` (or a cluster column) for single-domain sets;
- ``protein_cluster`` and ``molecule_group`` for cross-domain swap quads.

Each factor is resampled independently (the pigeonhole bootstrap for crossed designs, Owen 2007), so a design whose
items share proteins and molecules is not treated as if its items were independent.

One draw matrix per (item set, factor) is shared by every condition, connector and step scored on that item set, so
contrasts are paired by construction: a contrast's interval is the percentile interval of the per-draw difference.

A factor may be *stratified*: its units are resampled within their stratum, each stratum keeping its size. A single-task
AUROC resamples positive and negative units separately (``stats.auroc.class_strata``), so every draw has
both classes; a mean over several tasks uses the one unstratified matrix, shared by its tasks
(``evaluation.contrasts.group_draws``), since no one stratification fits tasks with different labels. Each stratum's
draws are seeded by ``(SEED, factor index, stratum)``, so two tasks with the same positive (or negative) units draw them
identically.

**Undefined draws.** A NaN is a bug, except a draw undefined *by design* (e.g. a crossed resample in
which no positive row is drawn, or a mean over tasks in whose joint draw a task lacks a class). :func:`summarize` is the
one place that drops such draws: it records their count in :attr:`Interval.dropped`, and it refuses a non-finite
full-sample value, an infinite draw, and fewer than half the draws defined.

:meth:`Bootstrap.draws` and :meth:`Bootstrap.contrast_draws` return the per-draw values; :meth:`Bootstrap.interval`
and :meth:`Bootstrap.contrast` summarize them, so there is one resampling loop.

**Estimates, not judgements.** The package reports point estimates, paired contrasts (differences) and
their :data:`LEVEL` intervals only: no grade, margin, p-value or test. A reading of a result (e.g. a pre-registered
verdict from an interval's lower bound) is made on top of the recorded intervals, outside the package.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import SerializerFunctionWrapHandler, model_serializer, model_validator

from latent_interconnect.config import Config, added_field

__all__ = [
    "DRAWS",
    "LEVEL",
    "SEED",
    "Bootstrap",
    "Interval",
    "IntervalFlag",
    "ResamplingUnits",
    "Statistic",
    "defined_draws",
    "summarize",
    "unit_index",
]

IntervalFlag = Literal["", "full_range", "independent_variance"]
"""The marker of an interval: empty for an ordinary one, else the fallback it is (``stats.shared_units``)."""
DRAWS = 2000
"""Bootstrap draws for every interval."""
SEED = 0
"""Seed of every draw matrix; a factor's stream is seeded by ``(SEED, factor index)``."""
LEVEL = 0.95
"""Two-sided coverage of every percentile interval: the one declared interval level."""

Statistic = Callable[[Mapping[str, np.ndarray]], float]
"""A statistic of one draw: factor → resampled unit indices ↦ a number."""


class ResamplingUnits(Config):
    """The resampling units of an analysis.

    Attributes:
        factors: item-set columns whose distinct values are the independent units, resampled independently (crossed).
            ``("object",)`` for a single-domain set; ``("protein_cluster", "molecule_group")`` for swap quads.
        interval: the interval method of a crossed design. Empty (the default): this module's bootstrap.
            ``"shared_unit"``: the shared-unit interval (``stats.shared_units``: empirically
            conservative, approximately 95%, one unresolved shortfall; selected by a named list of cross-domain
            analyses, its extension, their declared successors and 288 typed-pairing configs),
            which needs at least two factors and is accepted only by the analysis kinds that declare it
            (``evaluation.analysis.AnalysisSpec``). An added
            field (``config.added_field``): empty, it is outside the identity, so every config written before it keeps
            its digest.
    """

    factors: tuple[str, ...] = ("object",)
    interval: Literal["", "shared_unit"] = added_field(default="", repr=False)  # the default repr stays as stored

    @model_serializer(mode="wrap")
    def _omit_an_empty_interval(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Leave ``interval`` out while empty, so every config, record and result without one is stored as before."""
        out: dict[str, Any] = handler(self)
        if not out.get("interval"):
            out.pop("interval", None)
        return out

    @model_validator(mode="after")
    def _crossed(self) -> ResamplingUnits:
        if self.interval and len(self.factors) < 2:  # noqa: PLR2004 - a crossed design has two factors or more
            raise ValueError(
                f"interval {self.interval!r} is for a crossed design; factors {list(self.factors)} name one"
            )
        return self


@dataclass(frozen=True)
class Interval:
    """A point estimate with its percentile interval.

    Attributes:
        estimate: the statistic on the original sample.
        lo: lower percentile bound.
        hi: upper percentile bound.
        dropped: draws left out as undefined by design (:func:`summarize`).
        flag: empty for an ordinary interval; else why the interval is a fallback (``stats.shared_units``:
            ``full_range`` for a degenerate sample, ``independent_variance``). The bootstrap never sets it.
    """

    estimate: float
    lo: float
    hi: float
    dropped: int
    flag: IntervalFlag = ""


class Bootstrap:
    """Fixed draw matrices over the units of one item set.

    Args:
        n_units: factor name → number of units of that factor in the item set.
        draws: number of draws (tests only; analyses use :data:`DRAWS`).
        strata: factor → each unit's stratum (non-negative integers); a factor not listed is one stratum.
    """

    def __init__(
        self, n_units: Mapping[str, int], draws: int = DRAWS, strata: Mapping[str, np.ndarray] | None = None
    ) -> None:
        self.indices: dict[str, np.ndarray] = {}
        for k, (factor, n) in enumerate(n_units.items()):
            if strata is None or factor not in strata:
                counts = _multinomial(n, draws, np.random.default_rng([SEED, k]))
            else:
                codes = np.asarray(strata[factor], dtype=np.int64)
                if codes.shape != (n,):
                    raise ValueError(f"factor {factor!r}: {codes.shape[0]} strata codes for {n} units")
                counts = np.zeros((draws, n), dtype=np.int64)
                for s in np.unique(codes):
                    members = np.flatnonzero(codes == s)
                    counts[:, members] = _multinomial(members.size, draws, np.random.default_rng([SEED, k, int(s)]))
            self.indices[factor] = np.stack([np.repeat(np.arange(n), c) for c in counts])
        self.n_draws = draws

    def _draw(self, b: int) -> dict[str, np.ndarray]:
        return {f: ix[b] for f, ix in self.indices.items()}

    def draws(self, statistic: Statistic) -> np.ndarray:
        """Return ``statistic`` on every draw, shape ``(draws,)`` float64 (NaN where a draw is undefined by design).

        Args:
            statistic: maps one draw (factor → resampled unit indices) to a number.
        """
        return np.array([statistic(self._draw(b)) for b in range(self.n_draws)], dtype=np.float64)

    def contrast_draws(self, stat_a: Statistic, stat_b: Statistic) -> np.ndarray:
        """Return ``stat_a − stat_b`` on every draw: both statistics see the same draw each time (paired)."""
        return self.draws(lambda d: stat_a(d) - stat_b(d))

    def interval(self, statistic: Statistic, estimate: float) -> Interval:
        """Return ``estimate`` with the percentile interval of ``statistic`` over the draws (:func:`summarize`)."""
        return summarize(self.draws(statistic), estimate)

    def contrast(self, stat_a: Statistic, stat_b: Statistic, estimate: float) -> Interval:
        """Return the paired interval of ``stat_a − stat_b``."""
        return summarize(self.contrast_draws(stat_a, stat_b), estimate)


def _multinomial(n: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    counts: np.ndarray = rng.multinomial(n, np.full(n, 1.0 / n), size=draws)
    return counts


def defined_draws(values: np.ndarray) -> np.ndarray:
    """Return the defined draws of ``values``, dropping NaN draws (undefined by design): the one place that does.

    Raises:
        ValueError: if a draw is infinite, or fewer than half the draws are defined.
    """
    v = np.asarray(values, dtype=np.float64)
    if np.isinf(v).any():
        raise ValueError(f"{int(np.isinf(v).sum())} bootstrap draws are infinite (a bug, never a value)")
    kept = v[~np.isnan(v)]
    if 2 * kept.size < v.size:
        raise ValueError(f"only {kept.size} of {v.size} bootstrap draws are defined (fewer than half)")
    return kept


def summarize(values: np.ndarray, estimate: float) -> Interval:
    """Return ``estimate`` with the :data:`LEVEL` percentile interval of the defined per-draw ``values``.

    Undefined (NaN) draws are dropped and counted in :attr:`Interval.dropped` (:func:`defined_draws`).

    Raises:
        ValueError: if ``estimate`` is not finite (the full-sample value is undefined: a bug), a draw is infinite, or
            fewer than half the draws are defined.
    """
    if not np.isfinite(estimate):
        raise ValueError(f"the full-sample value is {estimate} (undefined on the full sample is a bug)")
    v = defined_draws(values)
    tail = (1 - LEVEL) / 2 * 100
    lo, hi = np.percentile(v, [tail, 100 - tail])
    return Interval(estimate=float(estimate), lo=float(lo), hi=float(hi), dropped=int(np.size(values) - v.size))


def unit_index(values: Iterable[object]) -> dict[str, int]:
    """Return each distinct unit (as text, NaN left out) → its index 0..n-1, in sorted order: one factor's units."""
    distinct = sorted({str(v) for v in values if not (isinstance(v, float) and math.isnan(v))})
    return {u: i for i, u in enumerate(distinct)}
