"""Residue map kinds: ``xattn`` (xa) and ``xblock`` (tr), attention over an object's per-residue states.

Both pool the standardized residues ``h̃_i`` with :class:`AttentionPool`: learned constant queries ``q_h`` (no ``W_q``:
a projection of a constant is a constant), keys ``k_hi = W_k h̃_i + b_k``, values ``v_hi = W_v h̃_i + b_v`` and
``o = concat_h Σ_i softmax_i(q_h · k_hi / √d_k) v_hi`` over the real residues. Then

- ``xattn``: ``δ = W_out o + b``;
- ``xblock``: ``x₁ = x₀ + W_o o + b_o``, ``δ = W_out FF(x₁) + b`` (a transformer block without self-attention).

From a ridge solution (:mod:`latent_interconnect.writes.blocks`): ``W_v = V_kᵀ``, ``b_v = b_k = 0``, ``x₀ = 0``,
``W_o = I``, FF the identity, ``W_out = W V_k``. At ``q = 0`` the attention is uniform over the real residues (``b_k``
shifts every logit of a head equally), so ``o`` is the standardized residue mean ``z`` projected by ``V_kᵀ`` and step 0
writes the rank-``d_s`` ridge ``W V_k V_kᵀ z + b``; ``q_init`` > 0 starts the queries near zero, not at it.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from latent_interconnect.writes.blocks import FeedForward, FeedForwardParams, StreamMap, StreamParams, linear
from latent_interconnect.writes.inputs import InputKind
from latent_interconnect.writes.kinds import MAP_KINDS, fan_in_normal

__all__ = ["AttentionParams", "AttentionPool", "BlockParams", "XAttn", "XBlock"]


class AttentionParams(StreamParams):
    """Hyperparameters of an attention-pooling kind.

    Attributes:
        heads: attention heads; each carries ``d_s / heads`` value features.
        d_k: query and key width per head.
        q_init: the queries start from N(0, (q_init / √d_k)²): relative to a unit-norm query, small so that step-0
            attention is close to uniform; 0 starts exactly at the rank-``d_s`` ridge.
    """

    heads: int = Field(gt=0)
    d_k: int = Field(gt=0)
    q_init: float = Field(default=0.01, ge=0.0)

    @model_validator(mode="after")
    def _check_heads(self) -> AttentionParams:
        if self.d_s % self.heads:
            raise ValueError(f"d_s {self.d_s} is not divisible by {self.heads} heads")
        return self


class BlockParams(AttentionParams, FeedForwardParams):
    """Hyperparameters of ``xblock``: attention pooling plus a feed-forward block (fields of both parents)."""


class AttentionPool(nn.Module):
    """Learned constant queries attending over the standardized residues: packed ``(N, d_in)`` → ``(B, d_s)``.

    Args:
        d_in: the residue width.
        params: the attention hyperparameters.
    """

    def __init__(self, d_in: int, params: AttentionParams) -> None:
        super().__init__()
        self.heads, self.d_k, self.d_v = params.heads, params.d_k, params.d_s // params.heads
        self.q = nn.Parameter(torch.empty(params.heads, params.d_k, dtype=torch.float32))
        self.k = linear(d_in, params.heads * params.d_k)
        self.v = linear(d_in, params.d_s)

    def initialize(self, in_map: Tensor, q_init: float, generator: torch.Generator) -> None:
        """Set every value.

        ``q`` (scale ``q_init / √d_k``) then ``W_k`` (N(0, 1/d_in)) are drawn from ``generator``; ``W_v = in_map``; the
        biases are zero.
        """
        with torch.no_grad():
            self.q.copy_(fan_in_normal(self.q.shape, generator, torch.float32) * q_init)
            self.k.weight.copy_(fan_in_normal(self.k.weight.shape, generator, torch.float32))
            self.k.bias.zero_()
            self.v.weight.copy_(in_map)
            self.v.bias.zero_()

    def references(self) -> dict[str, float]:
        """Return the learning-rate references of the small or zero-initialized parameters.

        ``q`` takes the RMS of a unit-norm ``d_k`` query (its own small start would all but freeze it); ``b_k`` and
        ``b_v`` start at zero, and the logits and values have unit scale.
        """
        return {"q": self.d_k**-0.5, "k.bias": 1.0, "v.bias": 1.0}

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Return ``o``, ``(B, d_s)``, from the packed real residues ``x`` ``(N, d_in)`` and the batch's ``mask``.

        Pooled first, projected after. Each head's weights sum to one over an object's residues, so
        ``Σ_i w_hi v_hi = W_v,h (Σ_i w_hi h̃_i) + b_v,h``, and ``q_h · k_hi = (W_k,hᵀ q_h) · h̃_i + q_h · b_k,h``: the
        function, and every parameter's gradient, of projecting each residue to ``k_hi`` and ``v_hi`` and pooling those
        (the code before this form), up to float rounding. Only the logits ``(N, heads)`` are computed per residue;
        backward keeps the softmax weights and each object's pooled residues ``(B, heads, d_in)`` beside ``x``, which
        the maps on a device share (``Connector.forward`` packs and standardizes once per device). Projecting first
        kept per map a float32 copy of ``x`` and ``k``, ``v`` and ``w v``, 45,256 B per real residue at the HM widths:
        the memory that split the 8xH200 HM arms' batch of 85 into calls of 32. The constant ``q_h · b_k,h`` stays in
        the logits although the softmax cancels it, so ``b_k`` keeps its gradient, zero in exact arithmetic and
        rounding-level in float, as before. Each object is pooled by its own softmax and matrix product, with no
        atomic adds.

        Args:
            x: the real residues, packed object by object (``residues[mask]``), standardized.
            mask: ``(B, T)`` validity mask; its row sums are the objects' residue counts in ``x``. A CPU mask avoids a
                device synchronization per map (the connector passes the batch's own).

        Raises:
            ValueError: without a mask, or if ``x`` does not hold the mask's real residues.
        """
        if mask is None:
            raise ValueError("attention pooling reads residues: it needs their (B, T) mask")
        lengths = mask.sum(dim=1).tolist()
        if tuple(x.shape) != (sum(lengths), self.k.in_features):
            raise ValueError(
                f"expected the mask's real residues, ({sum(lengths)}, {self.k.in_features}): {tuple(x.shape)}"
            )
        scale = self.d_k**-0.5
        u = torch.einsum("hkd,hk->hd", self.k.weight.view(self.heads, self.d_k, -1), self.q) * scale
        c = torch.einsum("hk,hk->h", self.k.bias.view(self.heads, self.d_k), self.q) * scale
        logits = x @ u.T + c  # (N, heads)
        pooled = torch.stack(
            [
                torch.softmax(lg, dim=0).T @ r
                for lg, r in zip(torch.split(logits, lengths), torch.split(x, lengths), strict=True)
            ]
        )  # (B, heads, d_in): each head's weighted mean residue
        o = torch.einsum("hvd,bhd->bhv", self.v.weight.view(self.heads, self.d_v, -1), pooled)
        return (o + self.v.bias.view(self.heads, self.d_v)).reshape(len(lengths), self.heads * self.d_v)


@MAP_KINDS.register("xattn")
class XAttn(StreamMap):
    """``δ = W_out o + b`` with ``o`` the attention pooling of the residues (xa)."""

    input_kind = InputKind.RESIDUES
    Params = AttentionParams

    def __init__(self, params: Any, d_in: int, d_model: int) -> None:
        super().__init__(params, d_in, d_model)
        self.attn = AttentionPool(d_in, params)

    def initialize(self, params: Any, in_map: Tensor, generator: torch.Generator) -> None:
        """Initialize the attention pooling (``W_v = in_map``)."""
        self.attn.initialize(in_map, params.q_init, generator)

    def references(self) -> dict[str, float]:
        """The attention pooling's references."""
        return {f"attn.{k}": v for k, v in self.attn.references().items()}

    def features(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Return ``o``."""
        return cast(Tensor, self.attn(x, mask))


@MAP_KINDS.register("xblock")
class XBlock(StreamMap):
    """``δ = W_out FF(x₀ + W_o o + b_o) + b`` with ``o`` the attention pooling of the residues (tr)."""

    input_kind = InputKind.RESIDUES
    Params = BlockParams

    def __init__(self, params: Any, d_in: int, d_model: int) -> None:
        super().__init__(params, d_in, d_model)
        self.attn = AttentionPool(d_in, params)
        self.x0 = nn.Parameter(torch.empty(params.d_s, dtype=torch.float32))
        self.o_in = linear(params.d_s, params.d_s)
        self.ff = FeedForward(params.d_s, params.ff_mult)

    def initialize(self, params: Any, in_map: Tensor, generator: torch.Generator) -> None:
        """Initialize the attention pooling, then ``x₀ = 0``, ``W_o = I``, ``b_o = 0`` and the FF block."""
        self.attn.initialize(in_map, params.q_init, generator)
        with torch.no_grad():
            self.x0.zero_()
            self.o_in.weight.copy_(torch.eye(self.o_in.in_features))
            self.o_in.bias.zero_()
        self.ff.initialize(generator)

    def references(self) -> dict[str, float]:
        """``x₀`` and ``b_o`` start at zero in the stream's unit scale; the blocks declare their own."""
        return {
            "x0": 1.0,
            "o_in.bias": 1.0,
            **{f"attn.{k}": v for k, v in self.attn.references().items()},
            **{f"ff.{k}": v for k, v in self.ff.references().items()},
        }

    def features(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Return ``FF(x₀ + W_o o + b_o)``."""
        return cast(Tensor, self.ff(self.x0 + self.o_in(self.attn(x, mask))))
