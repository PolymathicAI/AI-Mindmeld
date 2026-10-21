"""Building blocks of the stream map kinds (``mlp``, ``xattn`` and ``xblock``).

A stream kind reads the standardized encoder input into an internal stream of width ``d_s``, transforms it there, and
writes ``δ = W_out x + b``. Built from a ridge solution, it starts at the best rank-``d_s`` approximation of its
entry's ridge map: the input map is ``V_kᵀ``, the output map ``W V_k``
(:func:`~latent_interconnect.writes.init.rank_factor`, one factorization per (target, layer) shared through
``InitContext``), and every other part starts as the identity. So
step 0 writes ``W V_k V_kᵀ z + b`` (attention kinds: at uniform attention, i.e. ``q_init = 0``).

- :class:`StreamParams` / :class:`FeedForwardParams`: the hyperparameters every stream kind shares.
- :class:`FeedForward`: the pre-LayerNorm residual block, its output layer at zero.
- :class:`StreamMap`: the shared construction (ridge, random, empty), output map, init noise and learning-rate
  references; a kind supplies its stream modules (:meth:`StreamMap.initialize`) and its features
  (:meth:`StreamMap.features`).
- :func:`linear`: an ``nn.Linear`` built without drawing from the global random generator (every value is set from the
  entry's seeded generator or a checkpoint).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar, cast

import torch
from pydantic import Field
from torch import Tensor, nn

from latent_interconnect.config import Config
from latent_interconnect.writes.init import EntryInit, InitContext, rank_factor
from latent_interconnect.writes.kinds import EntryMap, fan_in_normal, random_init

__all__ = ["FeedForward", "FeedForwardParams", "StreamMap", "StreamParams", "linear"]


def linear(d_in: int, d_out: int) -> nn.Linear:
    """Return an uninitialized float32 ``nn.Linear(d_in, d_out)`` with a bias; the caller sets every value.

    Like the map's other parameters it is float32 whatever the default dtype, and lives on the default device (so
    ``torch.device("meta")`` sizes a map without allocating it); values are drawn on the CPU and copied in.
    """
    return nn.Linear(d_in, d_out, device="meta", dtype=torch.float32).to_empty(device=torch.get_default_device())


class StreamParams(Config):
    """Hyperparameters every stream kind shares.

    Attributes:
        d_s: the stream width; a ridge-built map needs ``d_s <= min(d_in, d_model)`` (its rank). At ``d_s = d_in``
            the input map is still factored (an orthogonal ``V``, not the identity); the step-0 map is the same.
    """

    d_s: int = Field(gt=0)


class FeedForwardParams(StreamParams):
    """Hyperparameters of a stream kind with a feed-forward block.

    Attributes:
        ff_mult: the feed-forward hidden width as a multiple of ``d_s``.
    """

    ff_mult: int = Field(gt=0)


class FeedForward(nn.Module):
    """``x + W₂ gelu(W₁ LN(x) + b₁) + b₂``: a pre-LayerNorm residual feed-forward block on the stream.

    :meth:`initialize` sets LN to the identity affine map, ``W₁`` to N(0, 1/d) and ``W₂``, ``b₁``, ``b₂`` to zero, so
    the block starts as the identity and its first step moves only ``W₂`` and ``b₂``.

    Args:
        d: the stream width.
        mult: the hidden width as a multiple of ``d``.
    """

    def __init__(self, d: int, mult: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d, dtype=torch.float32)
        self.ff1 = linear(d, mult * d)
        self.act = nn.GELU()
        self.ff2 = linear(mult * d, d)

    def initialize(self, generator: torch.Generator) -> None:
        """Set every value: LN gain 1 and bias 0, ``W₁`` from ``generator``, the rest zero."""
        with torch.no_grad():
            self.ln.reset_parameters()
            self.ff1.weight.copy_(fan_in_normal(self.ff1.weight.shape, generator, torch.float32))
            self.ff1.bias.zero_()
            self.ff2.weight.zero_()
            self.ff2.bias.zero_()

    def references(self) -> dict[str, float]:
        """Return the learning-rate reference of each zero-initialized parameter.

        The LN output, the hidden pre-activation and the stream all have unit scale (the input is standardized and
        ``W₁``'s rows have norm about 1), so their biases take 1; ``W₂`` takes the RMS a default ``1/sqrt(fan_in)``
        initialization of a ``d_ff → d`` map would have.
        """
        return {"ln.bias": 1.0, "ff1.bias": 1.0, "ff2.weight": self.ff2.in_features**-0.5, "ff2.bias": 1.0}

    def forward(self, x: Tensor) -> Tensor:
        """Return the block's output, the same shape as ``x``."""
        return x + cast(Tensor, self.ff2(self.act(self.ff1(self.ln(x)))))


class StreamMap(EntryMap):
    """A per-entry map ``δ = W_out features(x) + b`` whose features live in a stream of width ``d_s``.

    Subclasses set ``input_kind`` and ``Params``, build their stream modules in ``__init__`` (after calling this
    ``__init__``), and implement :meth:`initialize`, :meth:`features` and, for zero-initialized parameters,
    :meth:`references`.

    Args:
        params: the kind's hyperparameters.
        d_in: the encoder input width.
        d_model: the receiver hidden width.
    """

    Params: ClassVar[type[StreamParams]]

    def __init__(self, params: Any, d_in: int, d_model: int) -> None:
        super().__init__()
        del d_in  # the subclasses' input maps read it; the output map does not
        self.out = linear(params.d_s, d_model)

    @abstractmethod
    def initialize(self, params: Any, in_map: Tensor, generator: torch.Generator) -> None:
        """Set every stream parameter except ``out``.

        The input map is set to ``in_map`` ``(d_s, d_in)``; the rest is drawn from ``generator`` or set to the identity.
        """

    @abstractmethod
    def features(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Return the stream features ``(B, d_s)`` of a standardized input (see :meth:`EntryMap.forward`)."""

    def references(self) -> dict[str, float]:
        """Return the learning-rate reference of each parameter that does not take its own initial RMS."""
        return {}

    @classmethod
    def create(cls, params: Any, init: EntryInit, ctx: InitContext) -> StreamMap:
        """Start at the rank-``d_s`` approximation of the entry's ridge map (shared factors per (target, layer)).

        Raises:
            ValueError: if ``d_s`` exceeds ``min(d_in, d_model)``.
        """
        out_map, in_map = ctx.factor(init.target, init.layer, params.d_s, lambda: rank_factor(init.ridge.W, params.d_s))
        m = cls(params, ctx.d_in, ctx.d_model)
        m.initialize(params, in_map, init.generator)
        with torch.no_grad():
            m.out.weight.copy_(out_map)
            m.out.bias.copy_(torch.as_tensor(init.ridge.b, dtype=torch.float32))
        return m

    @classmethod
    def empty(cls, params: Any, d_in: int, d_model: int) -> StreamMap:
        """Return the map with every value zero (a checkpoint's state is loaded next)."""
        m = cls(params, d_in, d_model)
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
        return m

    @classmethod
    def init_random(cls, params: Any, d_in: int, d_model: int, generator: torch.Generator) -> StreamMap:
        """Build the map with random values.

        The input map and ``out`` follow the package's random rule (:func:`random_init`, N(0, 1/fan_in)); the rest is
        set as in :meth:`create`.
        """
        m = cls(params, d_in, d_model)
        m.initialize(params, fan_in_normal((params.d_s, d_in), generator, torch.float32), generator)
        random_init(m.out, generator)
        return m

    def lr_reference(self, name: str, init_value: Tensor) -> float:
        """Return the declared reference of ``name`` (:meth:`references`), else its initial RMS."""
        declared = self.references().get(name)
        return declared if declared is not None else super().lr_reference(name, init_value)

    def noise_targets(self) -> tuple[str, ...]:
        """Init noise goes on the output map only, as on ``affine``'s W."""
        return ("out.weight",)

    def decayed(self) -> tuple[str, ...]:
        """Weight decay goes on the output map only, as on ``affine``'s W: the intended pressure toward ``δ = b``.

        Not a guarantee: the stream's features are trained too and can compensate for the output map's shrink.
        """
        return ("out.weight",)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Return ``δ = W_out features(x) + b``, ``(B, d_model)``, in the map's dtype (float32)."""
        return cast(Tensor, self.out(self.features(x, mask)))
