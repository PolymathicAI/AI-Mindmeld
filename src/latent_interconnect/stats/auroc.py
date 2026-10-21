"""AUROC with average-rank ties (the Mann–Whitney form) and its bootstrap strata: the one implementation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import rankdata

__all__ = [
    "MIN_CLASS_UNITS",
    "auroc",
    "auroc_columns",
    "auroc_counts",
    "auroc_influence",
    "class_strata",
    "has_class_units",
]

MIN_CLASS_UNITS = 2
"""Fewest units holding each class for an AUROC with an interval: one unit would never vary."""


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return the area under the ROC curve of ``scores`` for binary ``labels``; ties count one half.

    Returns NaN when one class is absent: on the full sample a caller refuses it; in a bootstrap draw of a crossed
    design (several factors) it is a draw undefined by design, which ``stats.bootstrap.summarize`` drops and counts.

    Args:
        scores: real-valued scores, shape ``(n,)``.
        labels: booleans (or 0/1), shape ``(n,)``.
    """
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auroc_columns(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return :func:`auroc` of every column of ``scores`` against the same ``labels``: the same value, column by column.

    For many score columns over one sample (the features of a sparse dictionary, ``evaluation.sae_exploration``); ties
    count one half, so a column that is zero for most rows is ranked as :func:`auroc` ranks it.

    Args:
        scores: real-valued scores, shape ``(n, k)``.
        labels: booleans (or 0/1), shape ``(n,)``.

    Returns:
        ``(k,)`` float64; NaN in every column when one class is absent.

    Raises:
        ValueError: if ``scores`` is not ``(n, k)`` with one row per label.
    """
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != y.size:  # noqa: PLR2004 - a matrix of columns
        raise ValueError(f"scores of shape {s.shape} for {y.size} labels: one row per label, one column per score")
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(s.shape[1], np.nan)
    ranks = rankdata(s, axis=0)
    out: np.ndarray = (ranks[y].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def auroc_counts(scores: np.ndarray, labels: np.ndarray, counts: np.ndarray) -> float:
    """Return :func:`auroc` of the sample in which row ``i`` appears ``counts[i]`` times (a bootstrap draw's rows).

    Args:
        scores: real-valued scores, shape ``(n,)``.
        labels: booleans (or 0/1), shape ``(n,)``.
        counts: non-negative integer multiplicities, shape ``(n,)``.
    """
    return auroc(np.repeat(scores, counts), np.repeat(labels, counts))


def auroc_influence(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return each row's influence on the AUROC (ties one half): the centered placement value over its class size.

    For a positive row ``i``, ``(V10_i − A) / n_pos`` with ``V10_i`` the share of negative rows scored below it; for a
    negative row ``j``, ``(V01_j − A) / n_neg`` with ``V01_j`` the share of positive rows scored above it; ``A`` is the
    AUROC. The values sum to zero within each class; the shared-unit interval reads them (``stats.shared_units``).

    Args:
        scores: real-valued scores, shape ``(n,)``.
        labels: booleans (or 0/1), shape ``(n,)``.

    Raises:
        ValueError: if a class is absent.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if not y.any() or y.all():
        raise ValueError("an AUROC's influence values need both classes")
    wins = (s[y][:, None] > s[~y][None, :]) + 0.5 * (s[y][:, None] == s[~y][None, :])
    area = float(wins.mean())
    out = np.empty(s.size, dtype=np.float64)
    out[y] = (wins.mean(axis=1) - area) / int(y.sum())
    out[~y] = (wins.mean(axis=0) - area) / int((~y).sum())
    return out


def has_class_units(units: Mapping[str, np.ndarray], labels: np.ndarray) -> bool:
    """Return whether every factor has :data:`MIN_CLASS_UNITS` units holding a positive row and as many a negative.

    The one class-count rule of an AUROC with an interval.

    Args:
        units: factor → each row's unit (any comparable values).
        labels: each row's label (True for the positive class).
    """
    y = np.asarray(labels, dtype=bool)
    return all(min(np.unique(ix[y]).size, np.unique(ix[~y]).size) >= MIN_CLASS_UNITS for ix in units.values())


def class_strata(
    units: Mapping[str, np.ndarray], labels: np.ndarray, n_units: Mapping[str, int]
) -> dict[str, np.ndarray]:
    """Return factor → each unit's stratum in a stratified AUROC bootstrap: the label classes of its rows.

    Codes: 0 no row, 1 negatives only, 2 positives only, 3 both (a cluster holding both classes).

    Positive and negative units are then resampled separately, so a single-factor draw always has both classes.

    Args:
        units: factor → each row's unit index.
        labels: each row's label (True for the positive class).
        n_units: factor → number of units (the bootstrap's).

    Raises:
        ValueError: if fewer than :data:`MIN_CLASS_UNITS` units of a factor hold a positive or a negative row.
    """
    y = np.asarray(labels, dtype=bool)
    out = {}
    for factor, ix in units.items():
        n = n_units[factor]
        pos = np.bincount(ix[y], minlength=n) > 0
        neg = np.bincount(ix[~y], minlength=n) > 0
        if not has_class_units({factor: ix}, y):
            raise ValueError(
                f"factor {factor!r}: {int(pos.sum())} units hold positives and {int(neg.sum())} negatives; an AUROC "
                f"interval needs at least {MIN_CLASS_UNITS} of each"
            )
        out[factor] = neg.astype(np.int64) + 2 * pos
    return out
