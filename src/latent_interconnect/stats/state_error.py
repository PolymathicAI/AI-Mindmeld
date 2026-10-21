"""The summaries of a reconstruction's error over a set of states, as statistics of a bootstrap draw.

A reconstruction ``ĥ`` of states ``h`` (a sparse dictionary's, ``receivers.dictionary.encode_states``) is summarized
three ways, each a ratio whose terms are recomputed on the resampled states (the mean state and the denominator
included, never carried over from the full sample):

- the **fraction unexplained**, ``Σ|h - ĥ|² / Σ|h - mean(h)|²`` (the headline of
  ``receivers.dictionary.reconstruction_check``: 0 exact, 1 no better than the mean state);
- the **norm-normalized aggregate**, ``Σ|h - ĥ|² / Σ|h|²``;
- the **mean relative error**, the mean over states of ``|h - ĥ| / |h|``.

:class:`StateErrors` keeps per-unit sums (a unit: a resampling unit, e.g. a cluster of objects), so a draw costs one
product with the units' counts and no state is gathered again: ``Σ|h - m|² = Σ|h|² - n |m|²`` with ``m`` the mean of
the drawn states, in float64. A draw whose denominator is not positive is NaN (undefined by design: every drawn state
equal), which ``stats.bootstrap.summarize`` drops and counts.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["StateErrors"]


class StateErrors:
    """Per-unit sums of states and of their reconstruction errors; each method is a ``stats.Statistic`` of one factor.

    Args:
        states: ``(n, d)`` the states.
        error_sq: ``(n,)`` each state's squared reconstruction error ``|h - ĥ|²``.
        units: ``(n,)`` each state's unit index, ``0 .. n_units - 1``.
        n_units: the number of units.
        factor: the bootstrap factor the units are (the key of a draw).

    Raises:
        ValueError: if the shapes disagree, a unit index is out of range, or a state has zero norm (its relative error
            is undefined).
    """

    def __init__(self, states: np.ndarray, error_sq: np.ndarray, units: np.ndarray, n_units: int, factor: str) -> None:
        h = np.asarray(states, dtype=np.float64)
        e = np.asarray(error_sq, dtype=np.float64)
        u = np.asarray(units, dtype=np.int64)
        if h.ndim != 2 or e.shape != (h.shape[0],) or u.shape != (h.shape[0],):  # noqa: PLR2004 - a matrix of states
            raise ValueError(f"states {h.shape}, errors {e.shape}, units {u.shape}: one error and one unit per state")
        if u.size and (u.min() < 0 or u.max() >= n_units):
            raise ValueError(f"unit indices {int(u.min())}..{int(u.max())} outside 0..{n_units - 1}")
        norm_sq = np.square(h).sum(axis=1)
        if not (norm_sq > 0).all():
            raise ValueError(f"{int((norm_sq <= 0).sum())} states have zero norm: their relative error is undefined")
        self.factor = factor
        self._n = np.bincount(u, minlength=n_units).astype(np.float64)
        self._error = np.bincount(u, weights=e, minlength=n_units)
        self._norm = np.bincount(u, weights=norm_sq, minlength=n_units)
        self._relative = np.bincount(u, weights=np.sqrt(e / norm_sq), minlength=n_units)
        self._states = np.zeros((n_units, h.shape[1]), dtype=np.float64)
        np.add.at(self._states, u, h)
        self._units = n_units

    def all_units(self) -> dict[str, np.ndarray]:
        """Return the draw that is the full sample: every unit once."""
        return {self.factor: np.arange(self._units)}

    def _counts(self, draw: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.bincount(draw[self.factor], minlength=self._units).astype(np.float64)

    def centered_total(self, draw: Mapping[str, np.ndarray]) -> float:
        """Return ``Σ|h - mean(h)|²`` over the drawn states (the denominator of the fraction unexplained)."""
        c = self._counts(draw)
        n = float(c @ self._n)
        mean = (c @ self._states) / n
        return float(c @ self._norm - n * (mean @ mean))

    def unexplained(self, draw: Mapping[str, np.ndarray]) -> float:
        """Return the fraction unexplained on the drawn states; NaN if they have no variance."""
        total = self.centered_total(draw)
        return float(self._counts(draw) @ self._error) / total if total > 0 else float("nan")

    def norm_normalized(self, draw: Mapping[str, np.ndarray]) -> float:
        """Return ``Σ|h - ĥ|² / Σ|h|²`` on the drawn states."""
        c = self._counts(draw)
        return float(c @ self._error) / float(c @ self._norm)

    def relative(self, draw: Mapping[str, np.ndarray]) -> float:
        """Return the mean of ``|h - ĥ| / |h|`` over the drawn states."""
        c = self._counts(draw)
        return float(c @ self._relative) / float(c @ self._n)

    def mean_baseline(self, draw: Mapping[str, np.ndarray]) -> float:
        """Return ``Σ|h - mean(h)|² / Σ|h|²``: what the mean state alone scores under the norm-normalized aggregate."""
        return self.centered_total(draw) / float(self._counts(draw) @ self._norm)
