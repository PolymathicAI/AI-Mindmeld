"""The one-pass Triton INT4 unpack of Kimi K2.6's experts on CUDA (``dequant="triton"``), bit-identical to eager.

The eager unpack makes five elementwise passes over the block (shift, mask, subtract, convert, multiply), each reading
and writing a full-size temporary. This kernel reads each packed int32 word and its group's scale once, unpacks the
eight nibbles in registers and writes the eight outputs contiguously: one read of the packed weights and the scales and
one write of the result, the memory floor of the operation.

**Bit-identity.** Each output is ``(nibble - 8) · scale``. The eager path converts ``nibble - 8`` (an integer in
[-8, 7]) to the scale's dtype, which is exact, and multiplies two tensors of that dtype. For bf16 and fp16 PyTorch's
elementwise multiply (CUDA and CPU alike) computes in fp32 and rounds once to the output dtype, round to nearest even.
The kernel does the same: both factors converted to fp32 (exact), one fp32 multiply, one round to nearest even (Triton's
default downcast). The fp32 product is moreover exact (a 4-bit integer times an 8- or 11-bit significand fits in fp32's
24 bits), so both paths produce the correctly rounded value of the exact product. An fp32 or fp64 scale is multiplied in
its own dtype, as PyTorch does, so those agree too.

The kernel works on the flattened, contiguous tensors: word ``i`` of the ``(E·out·in/8)`` words holds outputs
``8i .. 8i+7`` of the contiguous ``(E, out, in)`` result, and its group's scale is the flat scale element ``8i // g``
(each row's ``in`` is a multiple of ``g``, and ``g`` a multiple of 8, so a word never straddles a group or a row).

Triton is imported by this module only, and ``experts.dequantize_int4`` imports the module on its first ``"triton"``
unpack, so the eager unpack never needs Triton. Triton ships with torch's CUDA wheels. The bit-identity is checked on
a GPU (``tests/unit/test_int4_triton.py``, which skips without CUDA).
"""

# ruff: noqa: ANN001, ANN202, PLR0917
# (a Triton kernel: its arguments are pointers and constexprs Triton types itself, passed positionally)
from __future__ import annotations

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
from torch import Tensor

__all__ = ["dequantize_int4_triton"]

_BLOCK_WORDS = 1024  # int32 words per program: 8192 outputs, 16 KiB of bf16 written per program
_DIMS = 3  # (E, out, in / 8) and (E, out, in / group_size)
_COMPUTE = {torch.float64: tl.float64}  # every other scale dtype multiplies in fp32 (PyTorch's opmath)


@triton.jit  # type: ignore[untyped-decorator]
def _unpack_kernel(  # type: ignore[no-untyped-def]
    packed_ptr,
    scale_ptr,
    out_ptr,
    n_words,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPUTE: tl.constexpr,
):
    words = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    live = words < n_words
    word = tl.load(packed_ptr + words, mask=live, other=0)
    scale = tl.load(scale_ptr + (words * 8) // GROUP, mask=live, other=0).to(COMPUTE)
    shifts = tl.arange(0, 8) * 4
    nibbles = ((word[:, None] >> shifts[None, :]) & 15) - 8
    values = nibbles.to(COMPUTE) * scale[:, None]
    targets = words[:, None] * 8 + tl.arange(0, 8)[None, :].to(tl.int64)
    tl.store(out_ptr + targets, values.to(out_ptr.dtype.element_ty), mask=live[:, None])


def dequantize_int4_triton(packed: Tensor, scale: Tensor, group_size: int) -> Tensor:
    """Dequantize a stack of ``pack-quantized`` INT4 weights in one Triton pass (CUDA tensors).

    Args:
        packed: ``(E, out, in / 8)`` int32, eight 4-bit values per word along the input dimension, low nibble first.
        scale: ``(E, out, in / group_size)``, on ``packed``'s device.
        group_size: inputs per scale, a multiple of 8.

    Returns:
        ``(E, out, in)`` contiguous, in ``scale``'s dtype: ``(nibble - 8) · scale``, bit-identical to
        ``experts.dequantize_int4``'s eager unpack.

    Raises:
        ValueError: on a non-int32 ``packed``, a group size that is not a positive multiple of 8, a CPU tensor, or
            shapes that disagree.
    """
    if packed.dtype != torch.int32 or packed.dim() != _DIMS or scale.dim() != _DIMS:
        raise ValueError(f"packed must be (E, out, in/8) int32 and scale 3-D; got {packed.dtype} {tuple(packed.shape)}")
    if group_size <= 0 or group_size % 8:
        raise ValueError(f"group_size must be a positive multiple of 8; got {group_size}")
    if not packed.is_cuda or scale.device != packed.device:
        raise ValueError(f"CUDA tensors on one device only; got {packed.device} and {scale.device}")
    n, out, words = packed.shape
    if tuple(scale.shape) != (n, out, words * 8 // group_size) or (words * 8) % group_size:
        raise ValueError(f"scale {tuple(scale.shape)} does not match packed {tuple(packed.shape)}, group {group_size}")
    result = torch.empty(n, out, words * 8, dtype=scale.dtype, device=packed.device)
    n_words = packed.numel()
    if n_words:
        grid = (triton.cdiv(n_words, _BLOCK_WORDS),)
        _unpack_kernel[grid](
            packed.contiguous(),
            scale.contiguous(),
            result,
            n_words,
            GROUP=group_size,
            BLOCK=_BLOCK_WORDS,
            COMPUTE=_COMPUTE.get(scale.dtype, tl.float32),
        )
    return result
