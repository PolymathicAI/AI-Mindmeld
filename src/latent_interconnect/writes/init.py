"""Initialization inputs for connector maps: ridge solutions, per-entry seeds and shared factorizations.

The construction stage (``build.ridge``) fits, per receiver layer L and per target (the receiver's state at the
object's name, or at its description), a ridge map from the standardized encoder input to that state. A map kind
initializes itself from its entry's ridge solution: ``affine`` copies it; ``mlp`` / ``xattn`` / ``xblock`` start at its
best rank-``d_s`` approximation (:func:`rank_factor`), sharing one factorization per (target, layer) through
:class:`InitContext`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from latent_interconnect.writes.inputs import InputSpec

__all__ = ["EntryInit", "InitContext", "RidgeLayer", "RidgeSolution", "entry_seed", "rank_factor"]


@dataclass(frozen=True)
class RidgeLayer:
    """The ridge map to the receiver state at one layer, in standardized input coordinates.

    Attributes:
        W: ``(d_model, d_in)`` float32.
        b: ``(d_model,)`` float32 (a mean state, so ``W = 0`` writes it: the paper's is the mean over the name tokens,
            ``build.ridge``).
    """

    W: np.ndarray
    b: np.ndarray


@dataclass(frozen=True)
class RidgeSolution:
    """All layers of one ridge fit (one encoder input → one receiver target), sharing one standardization.

    Attributes:
        target: what the receiver state is (``"name"``, ``"caption"``, ...).
        mu: ``(d_in,)`` input mean used for standardization.
        sd: ``(d_in,)`` input standard deviation (population, + 1e-8).
        layers: receiver layer → ridge map. Layer 0 is the token-embedding target.
        input: the fit's encoder input (``InputSpec.identity()``: recipe, read layers in order, combine, appended
            extraction), or None when unrecorded (a fit made in memory, or committed before inputs were recorded:
            refused by :meth:`require_fit`, refit it).
        recipe_digest: the digest of the encoder recipe the fit's input names, or None as ``input``.
        receiver: the receiver whose states the fit's target holds, ``{model_id, revision}``, from the capture's own
            record (``build.ridge.capture_receiver``), or None as ``input``.
        append_digest: the appended extraction's recipe digest a committed fit recorded, or None (no append, or a
            fit committed before the digest was recorded: refused with an append, :meth:`require_fit`); a run or
            scan reading another refuses the fit.
    """

    target: str
    mu: np.ndarray
    sd: np.ndarray
    layers: Mapping[int, RidgeLayer]
    input: Mapping[str, Any] | None = None
    recipe_digest: str | None = None
    receiver: Mapping[str, str] | None = None
    append_digest: str | None = None

    def require_fit(
        self,
        use: str,
        *,
        input: InputSpec,
        recipe_digest: str,
        receiver: Mapping[str, str],
        append_digest: str | None,
    ) -> None:
        """Refuse a fit made for another input or receiver than the caller's.

        Refused: a fit that records no input, recipe digest or receiver (refit it; nothing passes unbound), another
        input (recipe, read layers or their order, combine, appended extraction), another content of the input's
        recipe (``recipe_digest``: the digest of the recipe the caller reads), another receiver (``{model_id,
        revision}`` of the caller's receiver), a fit with an appended extraction but no
        recorded ``append_digest`` (refit it), or an appended recipe of another digest. ``append_digest`` is the digest
        of the appended recipe the caller loaded; None only where nothing appended is loaded (the init stage:
        ``Trainer.from_config`` checks the run's content at load).

        Raises:
            ValueError: naming ``use`` (whose input it is: ``"connector's input"``).
        """
        if self.input is None or self.recipe_digest is None or self.receiver is None:
            raise ValueError(f"ridge {self.target!r} records no input, recipe digest or receiver; refit it")
        if dict(self.receiver) != dict(receiver):
            raise ValueError(
                f"ridge {self.target!r} was fit to receiver {dict(self.receiver)}; the {use} is {receiver}"
            )
        theirs = input.identity()
        if self.input != theirs:
            raise ValueError(f"ridge {self.target!r} was fit on input {dict(self.input)}; the {use} is {theirs}")
        if self.recipe_digest != recipe_digest:
            raise ValueError(
                f"ridge {self.target!r} was fit on recipe content {self.recipe_digest[:12]}; the {use} reads "
                f"{recipe_digest[:12]}"
            )
        fit = self.append_digest
        if input.append is not None and fit is None:
            raise ValueError(f"ridge {self.target!r} records no appended recipe digest; refit it")
        if append_digest is not None and fit != append_digest:
            raise ValueError(
                f"ridge {self.target!r} was fit with appended recipe content {str(fit)[:12]}; the {use} "
                f"appends {append_digest[:12]}"
            )

    def reexpress(self, mu: np.ndarray, sd: np.ndarray) -> RidgeSolution:
        """Return the same affine maps written for another standardization (μ', σ').

        A connector has one standardization; a second target's ridge (fit with its own μ, σ) is re-expressed on it
        exactly: W' = W diag(σ'/σ), b' = b + W ((μ' - μ)/σ).
        """
        scale = (sd / self.sd).astype(np.float32)
        shift = ((mu - self.mu) / self.sd).astype(np.float32)
        layers = {L: RidgeLayer(W=r.W * scale[None, :], b=r.b + r.W @ shift) for L, r in self.layers.items()}
        return replace(self, mu=mu, sd=sd, layers=layers)


def entry_seed(base_seed: int, token: int, layer: int) -> int:
    """Derive the random seed of one (token, layer) entry: independent streams, stable across runs and machines.

    Raises:
        ValueError: for a negative ``token`` or ``layer`` (seed entropy must be non-negative).
    """
    return int(np.random.SeedSequence([base_seed, token, layer]).generate_state(1, dtype=np.uint64)[0] >> 1)


@dataclass(frozen=True)
class EntryInit:
    """Everything a map kind needs to initialize one entry.

    Attributes:
        token: slot-token index.
        layer: receiver layer.
        ridge: the entry's ridge map (its token's target at its layer), already in the connector's standardization.
        target: the ridge target name, used as part of the shared-factor cache key.
        generator: a torch generator seeded with :func:`entry_seed` (random parameters, e.g. attention queries).
    """

    token: int
    layer: int
    ridge: RidgeLayer
    target: str
    generator: torch.Generator


@dataclass
class InitContext:
    """Connector-level state shared across entries during initialization.

    Attributes:
        d_in: encoder input width.
        d_model: receiver hidden width.
        factors: cache of shared factorizations keyed by (target, layer, rank), filled by :meth:`factor`.
    """

    d_in: int
    d_model: int
    factors: dict[tuple[str, int, int], tuple[Tensor, Tensor]] = field(default_factory=dict)

    def factor(
        self, target: str, layer: int, rank: int, compute: Callable[[], tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        """Return the cached rank-``rank`` factorization ``(U·S, Vᵀ)`` of a ridge W, computing it once.

        The cached tensors are shared by every entry with the same key: callers copy them before training them.
        """
        key = (target, layer, rank)
        if key not in self.factors:
            self.factors[key] = compute()
        return self.factors[key]


def rank_factor(W: np.ndarray, rank: int) -> tuple[Tensor, Tensor]:
    """Return the best rank-``rank`` factorization of ``W`` as ``(W V_k, V_kᵀ)``, both float32.

    ``V_k`` holds the top ``rank`` eigenvectors of ``WᵀW`` (float64 ``eigh``), so ``W V_k V_kᵀ`` is the rank-``rank``
    truncation of ``W`` (Eckart–Young) and ``W V_k = U_k S_k``. Each column's sign is fixed so that its
    largest-magnitude entry is positive, which makes the factors a deterministic function of ``W`` (up to ties in the
    spectrum).

    Args:
        W: ``(m, n)`` map, e.g. a ridge W ``(d_model, d_in)``.
        rank: target rank, ``0 < rank <= min(m, n)``.

    Raises:
        ValueError: for a rank outside ``(0, min(m, n)]``.
    """
    W64 = torch.as_tensor(np.asarray(W), dtype=torch.float64)
    m, n = W64.shape
    if not 0 < rank <= min(m, n):
        raise ValueError(f"rank {rank} is outside (0, {min(m, n)}] for a {m} x {n} map")
    _, V = torch.linalg.eigh(W64.T @ W64)
    top = V[:, -rank:].flip(1)
    sign = torch.sign(top.gather(0, top.abs().argmax(0, keepdim=True)))
    top = top * torch.where(sign == 0, torch.ones_like(sign), sign)
    return (W64 @ top).to(torch.float32), top.T.contiguous().to(torch.float32)
