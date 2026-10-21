"""The write rule ``interp_v1``: how a write vector enters the receiver's residual stream.

This is the only write rule (a norm-preserving interpolation). At a
slot token's position and a write layer L (the input of decoder layer L; L = 0 is the token-embedding output)::

    u = h / |h|,  v = δ / |δ|,  m = (1 - α) u + α v,  h' = |h| · m / |m|

computed in at least float32 (float64 inputs stay float64) and cast back to the stream dtype. Consequences, part of
the definition: ``|h'| = |h|``; α = 0 leaves h unchanged; α = 1 is δ's direction at h's norm; h = 0 gives 0;
δ = 0 with α < 1 gives h.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["EPS", "WRITE_RULE", "interp"]

WRITE_RULE = "interp_v1"
EPS = 1e-6


def interp(h: Tensor, delta: Tensor, alpha: Tensor | float) -> Tensor:
    """Apply ``interp_v1`` row-wise.

    Args:
        h: residual-stream states, shape ``(..., d)``, any float dtype.
        delta: write vectors, broadcastable to ``h``.
        alpha: mixing weight(s) in [0, 1]: a float, or a tensor broadcastable to ``h[..., :1]``.

    Returns:
        The written states, same shape and dtype as ``h``. Differentiable in ``h``, ``delta`` and ``alpha``. Math
        runs in ``promote_types(h.dtype, float32)``.
    """
    dt = torch.promote_types(h.dtype, torch.float32)
    hf, df = h.to(dt), delta.to(dt)
    hn = hf.norm(dim=-1, keepdim=True)
    u = hf / hn.clamp_min(EPS)
    v = df / df.norm(dim=-1, keepdim=True).clamp_min(EPS)
    m = (1 - alpha) * u + alpha * v
    out: Tensor = (hn * m / m.norm(dim=-1, keepdim=True).clamp_min(EPS)).to(h.dtype)
    return out
