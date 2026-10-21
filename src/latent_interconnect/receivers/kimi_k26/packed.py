"""Packed sequences for ``kimi-ep``: one stream of a batch's real tokens, and Kimi's MLA attention on it.

A right-padded ``(B, T)`` batch becomes one ``(1, N)`` stream of its N real tokens (row 0's tokens, then row 1's, ...)
with per-sequence position ids restarting at 0. Every decoder layer runs on the stream: no padding token reaches the
attention, the dense MLP or the experts, and sequences never attend to each other (variable-length causal attention
over the cumulative lengths). Kimi's remote code cannot do this itself: its eager attention needs a ``(B, 1, T, T)``
mask, and its flash-attention class derives the lengths only from a 2-d padding mask.

:func:`mla_packed` re-implements the remote ``DeepseekV3Attention.forward`` math on the stream with the module's own
projections, rotary and softmax scale. ``"varlen"`` runs FlashAttention-2's variable-length kernel
(``torch.nn.attention.varlen``, GPU); ``"eager"`` scatters q, k and v back to the padded layout and runs the eager math
verbatim with the padded mask, which isolates the packing machinery from the kernel (CPU tests, the GPU parity job's
reference).

Ported from the original research trainer (``Packing``, ``mla_packed``, ``packed_layer``).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["Attention", "Packing", "packed_layer"]

Attention = Literal["varlen", "eager"]
"""The packed attention: FlashAttention-2 variable-length (GPU) or the eager math on the padded layout (any device)."""


class Packing:
    """A right-padded batch as one packed stream of its real tokens.

    Args:
        lengths: each row's token count.

    Attributes:
        lengths: each row's token count.
        start: ``(B,)`` each row's first index in the stream.
        cu: ``(B + 1,)`` int32 cumulative lengths (the variable-length kernel's ``cu_seqlens``).
        positions: ``(1, N)`` each token's position within its row.
        keep: ``(N,)`` each token's flat index in the padded ``(B, T)`` layout.
        longest: the longest row.

    Raises:
        ValueError: if a row is empty.
    """

    def __init__(self, lengths: list[int]) -> None:
        if not lengths or min(lengths) < 1:
            raise ValueError(f"packing needs non-empty rows, got lengths {lengths}")
        self.lengths = lengths
        self.longest = max(lengths)
        n = torch.tensor(lengths)
        self.cu = F.pad(n.cumsum(0), (1, 0)).to(torch.int32)
        self.start = self.cu[:-1].long()
        self.positions = torch.cat([torch.arange(k) for k in lengths]).unsqueeze(0)
        self.keep = torch.cat([r * self.longest + torch.arange(k) for r, k in enumerate(lengths)])

    @property
    def size(self) -> int:
        """N, the stream's token count."""
        return int(self.cu[-1])

    def index(self, rows: Tensor, positions: Tensor) -> Tensor:
        """Return the stream index of each (row, position) site."""
        return self.start[rows.long()] + positions.long()


def _rotary(attn: nn.Module) -> Any:
    """The remote code's ``apply_rotary_pos_emb`` (the module that defines the attention's class)."""
    return sys.modules[type(attn).__module__].apply_rotary_pos_emb


def mla_packed(attn: nn.Module, x: Tensor, pack: Packing, impl: Attention, mask: Tensor | None) -> Tensor:
    """The remote ``DeepseekV3Attention.forward`` on a packed stream ``x`` ``(1, N, d)``.

    Same projections (q LoRA with its layernorm; ``kv_a_proj_with_mqa`` → ``kv_a_layernorm`` → ``kv_b_proj``), q and
    k as ``[nope | rope]`` with the rotary applied at the per-sequence positions by the remote code's own function, one
    shared ``k_pe`` head broadcast over the heads, the module's ``softmax_scale`` (YaRN mscale included), causal within
    each sequence, ``o_proj``.

    Args:
        attn: the layer's attention module.
        x: the normed stream.
        pack: the stream's layout.
        impl: ``"varlen"`` or ``"eager"``.
        mask: the padded additive mask (``common.attention_mask``), for ``"eager"`` only.
    """
    a = cast(Any, attn)
    _, n, _ = x.shape
    heads, dn, dr, dv = a.num_heads, a.qk_nope_head_dim, a.qk_rope_head_dim, a.v_head_dim
    dev = x.device
    q = a.q_proj(x) if a.q_lora_rank is None else a.q_b_proj(a.q_a_layernorm(a.q_a_proj(x)))
    q_nope, q_pe = torch.split(q.view(1, n, heads, dn + dr), [dn, dr], dim=-1)
    ckv, k_pe = torch.split(a.kv_a_proj_with_mqa(x), [a.kv_lora_rank, dr], dim=-1)
    k_pe = k_pe.view(1, n, 1, dr)
    kv = a.kv_b_proj(a.kv_a_layernorm(ckv)).view(1, n, heads, dn + dv)
    k_nope, v = torch.split(kv, [dn, dv], dim=-1)
    cos, sin = a.rotary_emb(v, seq_len=pack.longest)  # a position's values do not depend on seq_len
    q_pe, k_pe = _rotary(attn)(q_pe, k_pe, cos, sin, pack.positions.to(dev), unsqueeze_dim=2)
    qs = torch.cat([q_nope, q_pe], dim=-1)
    ks = torch.cat([k_nope, k_pe.expand(1, n, heads, dr)], dim=-1)
    if impl == "varlen":
        from torch.nn.attention.varlen import varlen_attn  # noqa: PLC0415  (CUDA kernels; imported where used)

        vs = F.pad(v, [0, dn + dr - dv]) if dn + dr != dv else v  # the kernel needs one head dim for q, k and v
        cu = pack.cu.to(dev)
        attended = cast(
            Tensor,
            varlen_attn(
                qs[0], ks[0], vs[0], cu, cu, pack.longest, pack.longest, scale=a.softmax_scale, window_size=(-1, 0)
            ),
        )
        o = attended[..., :dv]
        out: Tensor = a.o_proj(o.reshape(1, n, heads * dv))
        return out
    if mask is None:
        raise ValueError("the eager packed attention needs the padded mask")
    b, t = pack.cu.shape[0] - 1, pack.longest  # the row count from a tensor's shape: no list-length guard compiled
    keep = pack.keep.to(dev)

    def unpack(z: Tensor) -> Tensor:  # (1, N, H, D) → (B, H, T, D), zeros at padding
        full = z.new_zeros(b * t, heads, z.shape[-1]).index_copy(0, keep, z[0])
        return full.view(b, t, heads, -1).transpose(1, 2)

    query, key, value = unpack(qs), unpack(ks), unpack(v)
    weights = torch.matmul(query, key.transpose(2, 3)) * a.softmax_scale + mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)  # as the remote eager attention
    o = torch.matmul(weights, value).transpose(1, 2).contiguous().reshape(b * t, heads * dv)[keep].unsqueeze(0)
    return cast(Tensor, a.o_proj(o))


def packed_layer(
    layer: nn.Module,
    h: Tensor,
    pack: Packing,
    impl: Attention,
    mask: Tensor | None,
    *,
    attend: Callable[..., Tensor] = mla_packed,
) -> Tensor:
    """The remote ``DeepseekV3DecoderLayer.forward`` on a packed stream (the same residual structure).

    ``attend`` is :func:`mla_packed`, or its compiled version (``kimi-ep``'s ``compile="regional"``).
    """
    lay = cast(Any, layer)
    h = h + attend(lay.self_attn, lay.input_layernorm(h), pack, impl, mask)
    out: Tensor = h + lay.mlp(lay.post_attention_layernorm(h))
    return out
