"""The ``mlp`` map kind: the mean-pool + MLP arm, a feed-forward block on the pooled input.

``δ = W_out FF(W_in z + b_in) + b`` on the pooled standardized input ``z`` (the object's mean residue state), with the
stream construction of :mod:`latent_interconnect.writes.blocks`: from a ridge solution, ``W_in = V_kᵀ``, ``b_in = 0``,
FF the identity and ``W_out = W V_k``, so step 0 writes the rank-``d_s`` ridge ``W V_k V_kᵀ z + b`` exactly.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from latent_interconnect.writes.blocks import FeedForward, FeedForwardParams, StreamMap, linear
from latent_interconnect.writes.inputs import InputKind
from latent_interconnect.writes.kinds import MAP_KINDS

__all__ = ["MLP"]


@MAP_KINDS.register("mlp")
class MLP(StreamMap):
    """``δ = W_out FF(W_in z + b_in) + b`` on the pooled input (the mean-pool + MLP arm)."""

    input_kind = InputKind.POOLED
    Params = FeedForwardParams

    def __init__(self, params: Any, d_in: int, d_model: int) -> None:
        super().__init__(params, d_in, d_model)
        self.w_in = linear(d_in, params.d_s)
        self.ff = FeedForward(params.d_s, params.ff_mult)

    def initialize(self, params: Any, in_map: Tensor, generator: torch.Generator) -> None:
        """Set ``W_in`` to ``in_map``, ``b_in`` to zero and the FF block to the identity."""
        del params
        with torch.no_grad():
            self.w_in.weight.copy_(in_map)
            self.w_in.bias.zero_()
        self.ff.initialize(generator)

    def references(self) -> dict[str, float]:
        """``b_in`` starts at zero in the stream's unit scale; the FF block declares its own."""
        return {"w_in.bias": 1.0, **{f"ff.{k}": v for k, v in self.ff.references().items()}}

    def features(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Return ``FF(W_in z + b_in)``."""
        del mask
        return cast(Tensor, self.ff(self.w_in(x)))
