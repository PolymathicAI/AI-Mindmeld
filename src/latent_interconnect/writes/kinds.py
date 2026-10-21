"""Map kinds: the open set of architectures that turn an object's encoder input into one entry's write vector.

A map kind is an :class:`EntryMap` subclass registered in :data:`MAP_KINDS` (one map per entry). It declares the input
it needs (pooled or residues), a ``Params`` config model for its hyperparameters, the learning-rate reference of each
parameter, how it initializes from ridge solutions and at random, and which tensors receive per-token init noise.

Registered here: ``constant`` (an object-independent vector; the layer-0 constant) and ``affine`` (δ = W z + b; the
paper's connectors and the Linear arm of the stream-kind comparison). The stream kinds are registered in
``writes.kinds_mlp`` (``mlp``, the mean-pool + MLP arm, on pooled input) and ``writes.kinds_residue`` (``xattn`` and
``xblock``, xa and tr, on residues).
A third party adds a kind the same way (``examples/guides/extending.md``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
from torch import Tensor, nn

from latent_interconnect.config import Config
from latent_interconnect.plugins import Registry
from latent_interconnect.writes.init import EntryInit, InitContext
from latent_interconnect.writes.inputs import InputKind

__all__ = [
    "MAP_KINDS",
    "Affine",
    "Constant",
    "EntryMap",
    "NoParams",
    "fan_in_normal",
    "random_init",
]


class NoParams(Config):
    """Hyperparameters of a kind that has none."""


class EntryMap(nn.Module, ABC):
    """Computes the write vector δ of one (token, layer) entry for a batch of objects.

    Subclasses set ``input_kind`` (the encoder input they read) and ``Params`` (their hyperparameter model), and
    implement :meth:`create` (initialize from the ridge solution), :meth:`empty` (shapes only, for loading a
    checkpoint) and :meth:`forward`; :meth:`init_random`, :meth:`noise_targets`, :meth:`lr_reference` and
    :meth:`decayed` have defaults.
    """

    input_kind: ClassVar[InputKind]
    Params: ClassVar[type[Config]] = NoParams

    def lr_reference(self, name: str, init_value: Tensor) -> float:
        """Return the reference scale of parameter ``name``: its learning rate is ``lr_rel × reference``.

        ``name`` is the parameter's dotted path relative to this kind (``"W"``, ``"ff.ff2.weight"``), as
        ``named_parameters()`` of the kind gives it; a kind may compose ``nn.Module`` parts.

        The default is the RMS of its initial value. A parameter that starts at zero has no such scale, so
        the default refuses it and the kind must override this method for it (each such kind states its rule).

        Raises:
            ValueError: if the initial value is all zeros.
        """
        rms = float(init_value.float().pow(2).mean().sqrt())
        if rms == 0.0:
            raise ValueError(f"{type(self).__name__}.{name} starts at zero: override lr_reference for it")
        return rms

    def decayed(self) -> tuple[str, ...]:
        """Return the parameters that take weight decay (``OptimizerSpec.weight_decay``); default: none.

        A kind decays the weight of δ's input-dependent part only, never its bias: δ is used by direction only
        (``writes.rule``), so a decayed weight with a kept bias is the intended pressure toward ``δ = b`` for an entry
        whose input no gradient holds (a pressure, not a guarantee: a needed bias may shrink with its weight).
        Decaying both together would only shrink the scale, which the loss cannot see. Names are relative to the
        kind, as in :meth:`lr_reference`.
        """
        return ()

    @classmethod
    @abstractmethod
    def create(cls, params: Any, init: EntryInit, ctx: InitContext) -> EntryMap:
        """Build and initialize the map of one entry.

        Args:
            params: an instance of ``cls.Params`` (validated from the connector config's ``kind_params``).
            init: the entry's ridge solution, seed and noise level.
            ctx: connector-level dimensions and shared factorizations.
        """

    @classmethod
    @abstractmethod
    def empty(cls, params: Any, d_in: int, d_model: int) -> EntryMap:
        """Build the map with the right shapes and uninitialized values (a checkpoint's state is loaded next)."""

    @classmethod
    def init_random(cls, params: Any, d_in: int, d_model: int, generator: torch.Generator) -> EntryMap:
        """Build the map with random values (the kind conformance check, ``writes.conformance``).

        The default draws every parameter from N(0, 1/fan_in) with fan_in its last dimension; kinds override it
        when a different start is principled.
        """
        m = cls.empty(params, d_in, d_model)
        random_init(m, generator)
        return m

    @abstractmethod
    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Return δ, shape ``(B, d_model)``, float32.

        Args:
            x: standardized input: ``(B, d_in)`` for pooled kinds; for residue kinds the real residues packed object
                by object, ``(N, d_in)`` (:attr:`~latent_interconnect.writes.inputs.InputKind.RESIDUES`).
            mask: ``(B, T)`` validity mask for residue kinds (its row sums split ``x``), None for pooled kinds.
        """

    def noise_targets(self) -> tuple[str, ...]:
        """Return the parameters that receive per-token init noise (default: none).

        The connector applies the noise after :meth:`create`: ``p += noise × RMS(p) × N(0, 1)``, drawn from the
        entry's generator, so multi-token connectors start with distinct but equally scaled tokens.
        """
        return ()


def fan_in_normal(shape: torch.Size | tuple[int, ...], generator: torch.Generator, dtype: torch.dtype) -> Tensor:
    """Draw a tensor from N(0, 1/fan_in), fan_in being its last dimension: the package's one random rule.

    The draw is on the CPU (seeded generators are CPU generators) in ``dtype``; callers copy it onto the parameter's
    device. For a parameter in the default dtype it equals ``torch.randn(shape, generator=g) * shape[-1] ** -0.5``.
    """
    return torch.randn(shape, generator=generator, dtype=dtype, device="cpu") * math.pow(shape[-1], -0.5)


def random_init(module: nn.Module, generator: torch.Generator) -> None:
    """Draw every parameter of ``module`` from N(0, 1/fan_in) (:func:`fan_in_normal`, in the parameter's dtype)."""
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(fan_in_normal(p.shape, generator, p.dtype))


MAP_KINDS: Registry[type[EntryMap]] = Registry("map kind")
"""Per-entry map kinds by name."""


@MAP_KINDS.register("constant")
class Constant(EntryMap):
    """δ = e, the same vector for every object; initialized at the ridge bias (the mean target state)."""

    input_kind = InputKind.NONE

    def __init__(self, e: Tensor) -> None:
        super().__init__()
        self.e = nn.Parameter(e)

    @classmethod
    def create(cls, params: Any, init: EntryInit, ctx: InitContext) -> Constant:
        """Initialize ``e`` from the ridge bias."""
        del params, ctx
        return cls(torch.as_tensor(init.ridge.b, dtype=torch.float32).clone())

    @classmethod
    def empty(cls, params: Any, d_in: int, d_model: int) -> Constant:
        """Return a zero constant of width ``d_model``."""
        del params, d_in
        return cls(torch.zeros(d_model))

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Return ``e`` for every object."""
        del mask
        return self.e.float().expand(x.shape[0], -1)


@MAP_KINDS.register("affine")
class Affine(EntryMap):
    """δ = W z + b on the pooled standardized input; initialized at the full ridge solution."""

    input_kind = InputKind.POOLED

    def __init__(self, W: Tensor, b: Tensor) -> None:
        super().__init__()
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)

    @classmethod
    def create(cls, params: Any, init: EntryInit, ctx: InitContext) -> Affine:
        """Copy the ridge map."""
        del params, ctx
        W = torch.as_tensor(init.ridge.W, dtype=torch.float32).clone()
        return cls(W, torch.as_tensor(init.ridge.b, dtype=torch.float32).clone())

    @classmethod
    def empty(cls, params: Any, d_in: int, d_model: int) -> Affine:
        """Return a zero ``(d_model, d_in)`` map."""
        del params
        return cls(torch.zeros(d_model, d_in), torch.zeros(d_model))

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Return ``W z + b``."""
        del mask
        return x.float() @ self.W.T + self.b

    def noise_targets(self) -> tuple[str, ...]:
        """Init noise goes on W only."""
        return ("W",)

    def decayed(self) -> tuple[str, ...]:
        """Weight decay goes on W only: the intended pressure toward δ = b where no gradient holds W."""
        return ("W",)
