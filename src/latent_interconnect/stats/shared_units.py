"""The shared-unit interval: an interval for a crossed design whose observations share units.

A cross-domain statistic is computed from *observations* (swap quads, ranking groups, or the rows of a row-level AUROC)
that touch protein clusters and molecule groups; two observations that share a unit are not independent. This module
computes the interval a config selects with ``ResamplingUnits.interval: shared_unit``. The crossed bootstrap
(``stats.bootstrap``) stays the default; the committed cross-domain analysis configs that select this interval are
a named list of cross-domain analyses, its extension, their declared versioned successors, and 288 typed-pairing
configs.

**Status.** The method is *empirically conservative and approximately 95%*: it has no proof of coverage. In its
one frozen confirmation (339 simulated scenarios on the real sharing structure of six cells, 10,000 datasets each) the
median coverage of the nominal 95% interval was 0.969, the minimum 0.944, and one scenario was more than two Monte
Carlo standard errors below 0.95: a paired contrast of the protein-ranking reciprocal rank with twelve dominant decoy
clusters, 0.9438 ± 0.0023. That shortfall is unresolved; no claim of uniform 95% coverage is made. The two ingredients
that distinguish it from the plain shared-unit variance (the pooled guard and the working-covariance degrees of
freedom) were confirmed only together.

**The rule.** For ``n`` observations with influence values ``e`` (the estimate's sampling error is the sum of the
observations' contributions: ``(q_g − mean)/n`` for a mean of per-observation statistics, the placement-value influence
for an AUROC, :func:`~latent_interconnect.stats.auroc.auroc_influence`; the difference of two arms' values for a paired
contrast) and factors ``f`` (item-set columns) whose units each observation touches:

- *dependence pattern*: ``A[g, h] = 1`` when ``g`` and ``h`` share a unit of any factor, and ``A[g, g] = 1``;
  observations that share no unit are taken as independent;
- *shared-unit variance*: ``Q = Σ_{A[g,h]=1} e_g e_h`` and ``V_Q = Q / (1 − c)``, where ``c`` corrects for centering at
  the estimate: ``N_s / n²`` with ``N_s = ΣA`` (for an AUROC, per label class, weighted by each class's share of
  ``Σe²``);
- *pooled guard*: per factor, with ``S_u`` and ``ss_u`` the sum and the sum of squares of ``e`` over the ``m_u``
  observations of unit ``u`` (units with ``m_u ≥ 2``), ``t_u = (S_u² − ss_u) / (m_u (m_u − 1))`` and
  ``τ_f = Σ w_u t_u / Σ w_u`` with ``w_u = 1 / (τ0_f + s0 / (m_u − 1))²``,
  ``τ0_f = max(0, Σ(S_u² − ss_u) / Σ m_u(m_u − 1))``, ``s0 = Σe² / n``; then
  ``V_pooled = (Σe² + Σ_f τ_f Σ_u m_u(m_u − 1)) / (1 − c)`` and the variance used is ``max(V_Q, V_pooled)``;
- *degrees of freedom*: the smaller of ``2Q² / Var̂(Q)`` (``Var̂(Q) = Σ_{A[g,h]=1} (z_g − z̄)(z_h − z̄)``,
  ``z_g = e_g (Ae)_g``; the independent-observation value ``s0² (n(κ − 3) + 2N_s)`` when it is not positive) and the
  Satterthwaite value ``2 tr(AΣ)² / (2 tr(AΣAΣ) + n(κ − 3)σ⁴)`` under the positive-semi-definite working covariance
  ``Σ = σ² I + Σ_f max(τ_f, 0) K_f``, ``K_f = M_f M_fᵀ`` (``M_f`` the observation-by-unit membership),
  ``σ² = max(s0 − Σ_f max(τ_f, 0) k̄_f, 0.05 s0)``, ``k̄_f`` the mean number of units of ``f`` per observation and ``κ``
  the sample kurtosis of ``e``; both clipped to ``[1, n − 1]``;
- *interval*: ``estimate ± t · √V`` with the Student ``t`` quantile of those degrees of freedom; for a single quantity
  bounded in ``[a, b]`` on the logit scale of ``(estimate − a) / (b − a)``, mapped back.

**Flagged intervals** (:attr:`~latent_interconnect.stats.bootstrap.Interval.flag`), never silent:

- :data:`FULL_RANGE`: every influence value is zero (the statistic is constant in the sample: at a bound, at an
  interior value, or a contrast whose paired differences are all zero). The sample carries no information about the
  variance; the interval is the statistic's full range (``[a, b]``, or ``[−(b − a), b − a]`` for a contrast), which is
  uninformative. This is a defined estimate with a degenerate sample; an input with no value (a missing class, no
  observation) is refused as before, never flagged.
- :data:`INDEPENDENT_VARIANCE`: the variance used is not positive (the masked plug-in is not a positive-semi-definite
  form, so ``Q`` can be zero or negative under negative dependence); the independent-observation variance is used,
  with the degrees of freedom of the working correlation ``r = clip(mean cross-product of dependent pairs / s0, 0, 1)``.

**Refused**, never flagged: a sample in which every pair of observations shares a unit. The centering correction ``c``
is then 1 and the rule has no value (``V_Q`` and ``V_pooled`` divide by ``1 − c``); no interval is invented for it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.stats import t as student_t

from latent_interconnect.stats.bootstrap import LEVEL, Interval, IntervalFlag

__all__ = [
    "FULL_RANGE",
    "INDEPENDENT_VARIANCE",
    "SHARED_UNIT",
    "SharedUnitVariance",
    "shared_unit_interval",
    "shared_unit_variance",
]

SHARED_UNIT = "shared_unit"
"""The value of ``ResamplingUnits.interval`` that selects this module's interval."""
FULL_RANGE: Final = "full_range"
"""Flag of an interval that is the statistic's full range: the sample is degenerate (module docstring)."""
INDEPENDENT_VARIANCE: Final = "independent_variance"
"""Flag of an interval built on the independent-observation variance: the variance used was not positive."""

_MIN_OBSERVATIONS = 3
_NO_INDEPENDENT_PAIR = 1e-12  # the centering correction is 1 up to rounding (a weighted sum per class for an AUROC)
_SIGMA_FLOOR = 0.05  # the working covariance keeps at least this share of the per-observation variance


@dataclass(frozen=True)
class SharedUnitVariance:
    """The variance of an estimate under the shared-unit rule, with what the interval needs.

    Attributes:
        variance: the variance the interval uses (the independent-observation variance under
            :data:`INDEPENDENT_VARIANCE`; 0 under :data:`FULL_RANGE`).
        df: the degrees of freedom of the Student reference.
        flag: ``""``, :data:`FULL_RANGE` or :data:`INDEPENDENT_VARIANCE`.
        shared: the shared-unit variance ``V_Q`` (may be zero or negative).
        pooled: the pooled-component variance ``V_pooled``.
        independent: the independent-observation variance.
    """

    variance: float
    df: float
    flag: IntervalFlag
    shared: float
    pooled: float
    independent: float


def _memberships(units: Mapping[str, np.ndarray], unit_counts: Mapping[str, int], n: int) -> list[np.ndarray]:
    """Return each factor's observation-by-unit membership (0/1, units no observation touches left out)."""
    out = []
    for factor, raw in units.items():
        ix = np.asarray(raw, dtype=np.int64)
        ix = ix[:, None] if ix.ndim == 1 else ix
        count = int(unit_counts[factor])
        if ix.shape[0] != n or ix.min() < 0:
            raise ValueError(f"factor {factor!r}: unit indices of shape {ix.shape} for {n} observations")
        member = np.zeros((n, count + 1), dtype=np.float64)
        np.put_along_axis(member, np.minimum(ix, count), 1.0, axis=1)  # an index ≥ count is padding
        member = member[:, :count]
        if not member.any(axis=1).all():
            raise ValueError(f"factor {factor!r}: an observation touches no unit")
        out.append(member[:, member.any(axis=0)])
    if not out:
        raise ValueError("the shared-unit interval needs at least one factor")
    return out


def _pooled_components(e: np.ndarray, members: list[np.ndarray], s0: float) -> tuple[list[float], float]:
    """Return each factor's pooled between-unit component ``τ_f`` and ``Σ_f τ_f Σ_u m_u(m_u − 1)``."""
    taus, total = [], 0.0
    for member in members:
        size = member.sum(axis=0)
        keep = size >= 2  # noqa: PLR2004 - a unit of one observation has no cross-product
        if not keep.any():
            taus.append(0.0)
            continue
        size = size[keep]
        cross = (e @ member[:, keep]) ** 2 - (e**2) @ member[:, keep]
        pairs = size * (size - 1.0)
        tau0 = max(float(cross.sum() / pairs.sum()), 0.0)
        weight = 1.0 / (tau0 + s0 / (size - 1.0)) ** 2
        taus.append(float((weight * cross / pairs).sum() / weight.sum()))
        total += taus[-1] * float(pairs.sum())
    return taus, total


def _working_df(
    pattern: np.ndarray,
    shares: list[np.ndarray],
    members: list[np.ndarray],
    taus: list[float],
    moments: tuple[float, float],
) -> float:
    """Return the Satterthwaite df of ``Q`` under ``σ² I + Σ_f τ_f⁺ K_f`` (``moments``: ``s0``, ``κ``)."""
    n = pattern.shape[0]
    s0, kurtosis = moments
    positive = [max(tau, 0.0) for tau in taus]
    explained = sum(tau * float(m.sum()) / n for tau, m in zip(positive, members, strict=True))
    sigma = max(s0 - explained, _SIGMA_FLOOR * s0)
    coefficient = np.array([sigma, *positive])
    products = [pattern] + [pattern @ k for k in shares]  # A X for X in (I, K_f)
    traces = np.array([float(pattern.trace())] + [float((pattern * k).sum()) for k in shares])  # tr(A X)
    cross_traces = np.array([[float((a * b.T).sum()) for b in products] for a in products])  # tr(A X A Y)
    mean_q = float(coefficient @ traces)
    var_q = 2.0 * float(coefficient @ cross_traces @ coefficient) + n * (kurtosis - 3.0) * sigma**2
    return float(np.clip(2.0 * mean_q**2 / max(var_q, 1e-300), 1.0, n - 1.0))


def _correlation_df(pattern: np.ndarray, q: float, ss: float, kurtosis: float) -> float:
    """Return the degrees of freedom under the working correlation ``r`` of dependent pairs (the fallback's)."""
    n = pattern.shape[0]
    n_dependent = float(pattern.sum())
    n_pairs = n_dependent - n
    r = float(np.clip((q - ss) / n_pairs / (ss / n), 0.0, 1.0)) if n_pairs > 0 else 0.0
    square = pattern @ pattern
    cubic, quartic = float((square * pattern).sum()), float((square**2).sum())
    spread = n * (kurtosis - 3.0) + 2.0 * ((1 - r) ** 2 * n_dependent + 2 * r * (1 - r) * cubic + r**2 * quartic)
    return float(np.clip(2.0 * ((1 - r) * n + r * n_dependent) ** 2 / max(spread, 1e-12), 1.0, n - 1.0))


def _centring(e: np.ndarray, pattern: np.ndarray, classes: np.ndarray | None) -> tuple[float, float]:
    """Return the centering correction ``c`` and the independent-observation variance (per class for an AUROC)."""
    n = e.size
    ss = float(e @ e)
    if classes is None:
        return float(pattern.sum()) / n**2, ss * n / (n - 1)
    label = np.asarray(classes, dtype=bool)
    sizes = (int(label.sum()), int((~label).sum()))
    if label.shape != (n,) or min(sizes) < 2:  # noqa: PLR2004 - a class of one observation has no variance
        raise ValueError(f"a two-sample statistic needs two observations of each class, got {sizes}")
    parts = [
        (float(e[k] @ e[k]), float(pattern[np.ix_(k, k)].sum()) / m**2, m)
        for k, m in zip((label, ~label), sizes, strict=True)
    ]
    centring = sum(s * c for s, c, _ in parts) / ss if ss > 0 else 0.0
    return centring, sum(s * m / (m - 1) for s, _, m in parts)


def shared_unit_variance(
    influence: np.ndarray,
    units: Mapping[str, np.ndarray],
    unit_counts: Mapping[str, int],
    classes: np.ndarray | None = None,
) -> SharedUnitVariance:
    """Return the variance and degrees of freedom of an estimate with influence values ``influence`` (module docstring).

    Args:
        influence: each observation's influence value, shape ``(n,)``.
        units: factor → each observation's unit indices, shape ``(n,)`` or ``(n, m)``; an index equal to or above
            the factor's unit count is padding (an observation with fewer than ``m`` distinct units).
        unit_counts: factor → number of units.
        classes: for a two-sample statistic (an AUROC), each observation's label class; the centering correction and
            the independent-observation variance are then taken per class. None for a mean of observations.

    Raises:
        ValueError: if a value is not finite, there are fewer than three observations, a class holds fewer than two,
            an observation touches no unit of a factor, or every pair of observations shares a unit (the centering
            correction is then 1 and the rule has no value; no interval is invented for that case).
    """
    e = np.asarray(influence, dtype=np.float64)
    n = e.size
    if e.ndim != 1 or n < _MIN_OBSERVATIONS:
        raise ValueError(
            f"the shared-unit interval needs at least {_MIN_OBSERVATIONS} observations, got shape {e.shape}"
        )
    if not np.isfinite(e).all():
        raise ValueError("an influence value is not finite (a NaN is a bug)")
    members = _memberships(units, unit_counts, n)
    shares = [m @ m.T for m in members]  # K_f: the number of units of factor f two observations share
    pattern = (sum(shares) > 0).astype(np.float64)
    np.fill_diagonal(pattern, 1.0)
    centring, independent = _centring(e, pattern, classes)
    ss = float(e @ e)
    if ss <= 0:
        return SharedUnitVariance(variance=0.0, df=1.0, flag=FULL_RANGE, shared=0.0, pooled=0.0, independent=0.0)
    if 1.0 - centring <= _NO_INDEPENDENT_PAIR:  # a ratio of counts: 1 when no pair of observations is independent
        raise ValueError(
            "the shared-unit variance is undefined when every pair of observations shares a unit "
            f"(centring correction {centring:.6g}; {n} observations)"
        )
    s0 = ss / n
    kurtosis = n * float((e**4).sum()) / ss**2
    spread = pattern @ e
    q = float(e @ spread)
    z = e * spread
    zc = z - z.mean()
    var_q = float(zc @ (pattern @ zc))
    if not var_q > 0:
        var_q = s0**2 * max(n * (kurtosis - 3.0) + 2.0 * float(pattern.sum()), 1e-12)
    df_pieces = float(np.clip(2.0 * q**2 / var_q, 1.0, n - 1.0))
    taus, pooled_sum = _pooled_components(e, members, s0)
    shared = q / (1.0 - centring)
    pooled = (ss + pooled_sum) / (1.0 - centring)
    variance = max(shared, pooled)
    if variance > 0:
        df = min(df_pieces, _working_df(pattern, shares, members, taus, (s0, kurtosis)))
        return SharedUnitVariance(variance, df, "", shared, pooled, independent)
    return SharedUnitVariance(
        independent, _correlation_df(pattern, q, ss, kurtosis), INDEPENDENT_VARIANCE, shared, pooled, independent
    )


def shared_unit_interval(
    estimate: float,
    influence: np.ndarray,
    units: Mapping[str, np.ndarray],
    unit_counts: Mapping[str, int],
    bounds: tuple[float, float],
    *,
    contrast: bool = False,
    classes: np.ndarray | None = None,
) -> Interval:
    """Return ``estimate`` with its shared-unit interval (module docstring); ``Interval.flag`` marks a fallback.

    Args:
        estimate: the statistic on the sample (for a contrast, the difference of the two arms).
        influence: each observation's influence value (for a contrast, the difference of the two arms' values).
        units: factor → each observation's unit indices (:func:`shared_unit_variance`).
        unit_counts: factor → number of units.
        bounds: the range ``(a, b)`` of the single statistic (of each arm, for a contrast).
        contrast: whether ``estimate`` is a paired difference: no logit scale, and the full range is ``±(b − a)``.
        classes: each observation's label class for a two-sample statistic, else None.

    Raises:
        ValueError: if ``estimate`` is not finite or outside the statistic's range, or as
            :func:`shared_unit_variance`.
    """
    a, b = float(bounds[0]), float(bounds[1])
    low, high = (-(b - a), b - a) if contrast else (a, b)
    if not np.isfinite(estimate) or not low - 1e-12 <= estimate <= high + 1e-12:
        raise ValueError(f"the estimate {estimate} is outside the statistic's range [{low}, {high}]")
    v = shared_unit_variance(influence, units, unit_counts, classes)
    if v.flag == FULL_RANGE:
        return Interval(estimate=float(estimate), lo=low, hi=high, dropped=0, flag=FULL_RANGE)
    half = float(student_t.ppf(1 - (1 - LEVEL) / 2, v.df)) * float(np.sqrt(v.variance))
    u = (estimate - a) / (b - a)
    if contrast or not 0 < u < 1:
        lo, hi = estimate - half, estimate + half
    else:
        eta, h = float(np.log(u / (1 - u))), half / (b - a) / (u * (1 - u))
        lo, hi = a + (b - a) / (1 + float(np.exp(-(eta - h)))), a + (b - a) / (1 + float(np.exp(-(eta + h))))
    return Interval(estimate=float(estimate), lo=float(lo), hi=float(hi), dropped=0, flag=v.flag)
