"""L2-penalized binary logistic regression on a fixed design, its penalty chosen by cluster-grouped cross-validation.

The fit behind the supervised logistic baseline (``evaluation.logistic_baseline``). Every task is its own
binary model: its own rows (``rows[:, t]``), labels, weights and intercept, and its own penalty. Per task and penalty
``C`` the model minimizes the mean log-loss over its training rows plus ``‖w‖² / (2 C n)`` (``n`` its training rows;
the intercept is not penalized), which has the minimizer of ``½‖w‖² + C Σ log-loss``, so ``C`` is the familiar inverse
penalty. The features are standardized on the training objects of the fit: every row of the design outside the
held-out fold (or every row, for the all-rows fit), whatever its labels, by the population mean and σ plus
:data:`~latent_interconnect.build.ridge.SD_EPSILON`, so the penalty acts on standardized weights. The standardization
reads no label, and it is folded into each model's weights, so the design is held once and never copied per fold.

**Selection.** Each object's fold is given (``evaluation.read_scan.cluster_folds`` over its cluster, so a cluster is
never split). Per penalty, each task is fit on every fold's complement and once on all its rows; its cross-validated
log-loss is the held-out log-loss summed over the folds, over its rows; the penalty with the least wins (the smallest,
the strongest penalty, on a tie), and the all-rows fit at that penalty is the task's model. Only the training rows
and their labels enter: nothing is fit or chosen on another set.

**Solver.** Per penalty, every (task, fold) model and every task's all-rows model is solved at once by one
L-BFGS-B minimization (``scipy.optimize``) of the sum of their objectives. The sum is separable, so its minimizer is
each model's own; the shared solve only lets every model's two products with the design be one matrix product each.
The solve runs in preconditioned coordinates: a model's standardized weights are ``w = U (s ⊙ θ)``, with ``U`` the
eigenvectors of its fold's standardized covariance (label-free, as the standardization) and ``s_j = (e_j / 4 + λ)^-½``
from their eigenvalues ``e_j`` and the penalty ``λ = 1 / (C n)``: the inverse square root of the objective's curvature
at ``w = 0``. The penalty is rotation-invariant, so this changes the path of the solver, never the objective or its
minimizer; it makes the strongly correlated features of an embedding cheap to fit. It stops when every model's largest
gradient component with respect to its standardized weights and intercept is at most ``tol``, or after ``max_iter``
iterations; each model's convergence is checked on its own gradient and returned. A penalty that is not chosen may
stop unconverged (it is recorded, ``PenaltyFit.converged``); a task whose chosen penalty has an unconverged fold or
all-rows fit is refused, since an early stop acts as extra regularization and biases its cross-validated loss. The
intercepts start at the log-odds of their training rows, the weights at zero, and each penalty, solved in increasing
order of ``C``, starts from the previous one's solution. BLAS uses the CPUs of the process's affinity mask, so the
result can differ in the last bits with the thread count (as every BLAS fit here); it never depends on anything else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit, logit

from latent_interconnect.build.ridge import SD_EPSILON

__all__ = ["FoldBases", "LogisticPath", "PenaltyFit", "fit_logistic_path", "fold_bases"]

_CHUNK = 8192
"""Rows per block when accumulating a design's moments per fold (bounds the block's temporaries)."""
_CURVATURE = 0.25
"""The log-loss's curvature at log-odds 0 (``p (1 - p)`` at ``p = ½``): the preconditioner's scale of the covariance."""


@dataclass(frozen=True)
class FoldBases:
    """Per fit ``k`` (fold ``k``'s complement, ``0 .. K-1``; ``K``: every row): its standardization and its basis.

    Attributes:
        mu: ``(d, K + 1)``: the mean of each feature.
        sd: ``(d, K + 1)``: its σ plus ``SD_EPSILON``.
        eigenvalues: ``(d, K + 1)``: of the standardized covariance (clipped at 0).
        eigenvectors: ``(K + 1, d, d)``: the standardized covariance's eigenvectors, as columns.
    """

    mu: np.ndarray
    sd: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass(frozen=True)
class PenaltyFit:
    """One penalty's fits.

    Attributes:
        C: the inverse penalty.
        coef: ``(d, T)``: each task's all-rows weights on the raw features (the standardization folded in).
        intercept: ``(T,)``: their intercepts on the raw features.
        cv_log_loss: ``(T,)``: each task's cross-validated log-loss (the module docstring).
        converged: ``(T, K + 1)``: whether each task's fold fits (``0 .. K-1``) and its all-rows fit (``K``) reached
            ``tol``.
        iterations: the shared solve's L-BFGS iterations.
        evaluations: its objective evaluations.
    """

    C: float
    coef: np.ndarray
    intercept: np.ndarray
    cv_log_loss: np.ndarray
    converged: np.ndarray
    iterations: int
    evaluations: int


@dataclass(frozen=True)
class LogisticPath:
    """Every penalty's fits of a task set, and each task's choice.

    Attributes:
        fits: per penalty, in increasing ``C``.
        chosen: ``(T,)``: each task's index into ``fits`` (least cross-validated log-loss, the first on a tie).
    """

    fits: tuple[PenaltyFit, ...]
    chosen: np.ndarray

    def decision(self, X: np.ndarray) -> np.ndarray:
        """Return ``(m, T)``: each task's chosen all-rows model's log-odds for the rows of ``X`` (raw features)."""
        out = np.empty((len(X), len(self.chosen)))
        for j, fit in enumerate(self.fits):
            tasks = np.flatnonzero(self.chosen == j)
            if tasks.size:
                out[:, tasks] = X @ fit.coef[:, tasks] + fit.intercept[tasks]
        return out


def fold_bases(X: np.ndarray, folds: np.ndarray, n_folds: int) -> FoldBases:
    """Return each fit's standardization and the eigenbasis of its standardized covariance (label-free).

    The rows' blocks are accumulated per fold (sums and Gram matrices), so no fold's rows are copied.
    """
    d = X.shape[1]
    sums, grams = np.zeros((n_folds, d)), np.zeros((n_folds, d, d))
    for start in range(0, len(X), _CHUNK):
        block, f = X[start : start + _CHUNK], folds[start : start + _CHUNK]
        for k in np.unique(f):
            part = block[f == k]
            sums[k] += part.sum(0)
            grams[k] += part.T @ part
    counts = np.bincount(folds, minlength=n_folds).astype(np.float64)
    total, gram, n = sums.sum(0), grams.sum(0), counts.sum()
    mus, sds, values, vectors = [], [], [], []
    for k in range(n_folds + 1):
        m = n if k == n_folds else n - counts[k]
        mu = (total if k == n_folds else total - sums[k]) / m
        cov = (gram if k == n_folds else gram - grams[k]) / m - np.outer(mu, mu)
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None)) + SD_EPSILON
        e, U = np.linalg.eigh(cov / np.outer(sd, sd))
        mus.append(mu)
        sds.append(sd)
        values.append(np.clip(e, 0.0, None))
        vectors.append(U)
    return FoldBases(np.stack(mus, 1), np.stack(sds, 1), np.stack(values, 1), np.stack(vectors))


def fit_logistic_path(
    X: np.ndarray,
    rows: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    n_folds: int,
    grid: Sequence[float],
    max_iter: int,
    tol: float,
) -> LogisticPath:
    """Fit every task at every penalty of ``grid`` and choose each task's penalty (the module docstring).

    Args:
        X: ``(n, d)`` float64 design: the training objects' features.
        rows: ``(n, T)`` bool: the rows each task trains on (its scope, its known labels).
        labels: ``(n, T)`` bool: each task's labels (read on its rows only).
        folds: ``(n,)``: each row's fold, ``0 .. n_folds - 1``.
        n_folds: the number of folds.
        grid: the inverse penalties ``C``, increasing.
        max_iter: the most L-BFGS iterations of one penalty's solve.
        tol: the largest gradient component a converged model may have.

    Raises:
        ValueError: if the grid is not increasing, a fold is empty, a task's rows of some fit hold one class, or a
            fit of a task's chosen penalty (a fold fit, whose held-out loss chose it, or its all-rows model) did not
            converge.
    """
    C = np.asarray(grid, dtype=np.float64)
    if C.size == 0 or (np.diff(C) <= 0).any():
        raise ValueError(f"logistic fit: the C grid must be non-empty and increasing, not {list(grid)}")
    T, K = rows.shape[1], n_folds
    if (np.bincount(folds, minlength=K) == 0).any():
        raise ValueError(f"logistic fit: a fold of {K} holds no row")
    fold_of = np.tile(np.arange(K + 1), T)  # column c is (task c // (K+1), fold c % (K+1)); fold K is all rows
    task_of = np.repeat(np.arange(T), K + 1)
    member = rows[:, task_of] & (folds[:, None] != fold_of[None, :])
    n_rows = member.sum(0).astype(np.float64)
    y = labels[:, task_of] & member
    positives = y.sum(0)
    one_class = np.flatnonzero((positives == 0) | (positives == n_rows))
    if one_class.size:
        bad = sorted({(int(task_of[c]), int(fold_of[c])) for c in one_class})
        raise ValueError(f"logistic fit: (task, fold) fits {bad[:5]} have one label class (fold {K}: all rows)")
    bases = fold_bases(X, folds, K)
    solve = _Solve(X, weight=member / n_rows, target=y.astype(np.float64), bases=bases, fold_of=fold_of, tol=tol)
    held = (folds[:, None] == fold_of[None, :]) & rows[:, task_of] & (fold_of < K)[None, :]
    W, b = np.zeros((X.shape[1], fold_of.size)), logit(positives / n_rows)
    fits = []
    for c_value in C:
        W, b, result = solve.run(1.0 / (c_value * n_rows), W, b, max_iter)
        V, offset = solve.raw(W, b)
        Z = X @ V + offset
        losses = np.where(held, np.logaddexp(0.0, Z) - labels[:, task_of] * Z, 0.0).sum(0)
        full = fold_of == K
        fits.append(
            PenaltyFit(
                C=float(c_value),
                coef=V[:, full],
                intercept=offset[full],
                cv_log_loss=losses.reshape(T, K + 1).sum(1) / rows.sum(0),
                converged=(solve.last_gradient <= tol).reshape(T, K + 1),
                iterations=int(result.nit),
                evaluations=int(result.nfev),
            )
        )
    chosen = np.stack([f.cv_log_loss for f in fits]).argmin(0)  # argmin: the first (smallest C) on a tie
    unconverged = [
        (t, fits[j].C, [int(k) for k in np.flatnonzero(~fits[j].converged[t])], fits[j].iterations)
        for t, j in enumerate(chosen)
        if not fits[j].converged[t].all()
    ]
    if unconverged:
        raise ValueError(
            "logistic fit: the chosen penalty's fits did not reach tol "
            f"{tol:g} within max_iter {max_iter}; (task, C, unconverged folds (fold {K}: all rows), iterations): "
            f"{unconverged[:5]}"
        )
    return LogisticPath(tuple(fits), chosen)


class _Solve:
    """One penalty's shared solve in preconditioned coordinates (the module docstring)."""

    def __init__(
        self,
        X: np.ndarray,
        *,
        weight: np.ndarray,
        target: np.ndarray,
        bases: FoldBases,
        fold_of: np.ndarray,
        tol: float,
    ) -> None:
        self.X, self.weight, self.target, self.bases, self.tol = X, weight, target, bases, tol
        self.mu, self.sd = bases.mu[:, fold_of], bases.sd[:, fold_of]
        self.groups = [(k, np.flatnonzero(fold_of == k)) for k in range(bases.mu.shape[1])]
        self.last_gradient = np.full(fold_of.size, np.inf)
        self._last_x: np.ndarray | None = None

    def raw(self, W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the raw-feature weights and intercepts of standardized weights ``W`` and intercepts ``b``."""
        V = W / self.sd
        return V, b - (self.mu * V).sum(0)

    def run(
        self, ridge: np.ndarray, W: np.ndarray, b: np.ndarray, max_iter: int
    ) -> tuple[np.ndarray, np.ndarray, OptimizeResult]:
        """Solve at penalties ``ridge`` (per model) from ``(W, b)``; return the solution and the optimizer's result."""
        d, cols = W.shape
        scale = np.empty_like(W)
        for k, c in self.groups:
            scale[:, c] = 1.0 / np.sqrt(_CURVATURE * self.bases.eigenvalues[:, k, None] + ridge[None, c])

        def weights(theta: np.ndarray) -> np.ndarray:
            S = scale * theta[: d * cols].reshape(d, cols)
            out: np.ndarray = np.empty_like(S)
            for k, c in self.groups:
                out[:, c] = self.bases.eigenvectors[k] @ S[:, c]
            return out

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            W, b = weights(theta), theta[d * cols :]
            V, offset = self.raw(W, b)
            Z = self.X @ V + offset
            loss = (self.weight * (np.logaddexp(0.0, Z) - self.target * Z)).sum() + 0.5 * (ridge * (W * W).sum(0)).sum()
            R = self.weight * (expit(Z) - self.target)
            s = R.sum(0)
            grad_w = (self.X.T @ R - self.mu * s) / self.sd + ridge * W
            self.last_gradient = np.maximum(np.abs(grad_w).max(0), np.abs(s))
            self._last_x = theta.copy()
            grad_theta = np.empty_like(grad_w)
            for k, c in self.groups:
                grad_theta[:, c] = self.bases.eigenvectors[k].T @ grad_w[:, c]
            return float(loss), np.concatenate([(scale * grad_theta).ravel(), s])

        def stop(intermediate_result: OptimizeResult) -> None:
            if np.array_equal(intermediate_result.x, self._last_x) and (self.last_gradient <= self.tol).all():
                raise StopIteration

        theta0 = np.empty_like(W)
        for k, c in self.groups:
            theta0[:, c] = self.bases.eigenvectors[k].T @ W[:, c]
        x0 = np.concatenate([(theta0 / scale).ravel(), b])
        result = minimize(
            objective,
            x0,
            jac=True,
            method="L-BFGS-B",
            callback=stop,
            options={"maxiter": max_iter, "gtol": 0.0, "ftol": 0.0},
        )
        if not np.array_equal(result.x, self._last_x):
            objective(result.x)  # the converged flags are the returned solution's
        return weights(result.x), result.x[d * cols :], result
