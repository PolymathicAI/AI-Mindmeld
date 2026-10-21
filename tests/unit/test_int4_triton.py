"""``dequant="triton"``: the one-pass Triton INT4 unpack (``kimi_k26.int4_triton``) is bit-identical to the eager one,
and ``experts.dequantize_int4`` never falls back from it.

The kernel tests need CUDA and skip without it (the CPU suite hides the devices; a GPU job runs them with ``--gpu``), so
on CPU they prove nothing about the kernel. The CPU tests check the dispatch: the eager unpack needs no Triton, and
``"triton"`` refuses a CPU tensor, a group size that is not a positive multiple of 8 and a missing Triton instead of
unpacking eagerly.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from latent_interconnect.receivers.kimi_k26 import experts

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton kernel needs CUDA")

# (E, out, in, group): odd expert and row counts, inputs one group wide, word counts that do not fill the last
# program (1024 words), one stack of many programs, and group size 16
SHAPES = [(1, 1, 32, 32), (3, 5, 96, 32), (7, 33, 224, 32), (2, 129, 1056, 32), (5, 300, 2080, 32), (9, 64, 512, 16)]


def _random(shape: tuple[int, int, int, int], dtype: torch.dtype, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n, out, inp, group = shape
    gen = torch.Generator().manual_seed(n * 1_000_003 + out * 1009 + inp)
    packed = torch.randint(-(2**31), 2**31 - 1, (n, out, inp // 8), dtype=torch.int32, generator=gen)
    # scales of both signs over nine binades, so the products round every way; every product stays normal in fp16
    size = (n, out, inp // group)
    sign = torch.randint(0, 2, size, generator=gen) * 2 - 1
    scale = sign * (0.5 + torch.rand(size, generator=gen)) * torch.randint(-10, -1, size, generator=gen).exp2()
    return packed.to(device), scale.to(dtype).to(device)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


@CUDA
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_triton_unpack_is_bit_identical_to_eager(shape: tuple[int, int, int, int], dtype: torch.dtype) -> None:
    packed, scale = _random(shape, dtype, "cuda")
    ours = experts.dequantize_int4(packed, scale, shape[3], "triton")
    assert ours.is_contiguous() and ours.device == packed.device
    assert _same_bits(ours, experts.dequantize_int4(packed, scale, shape[3]))


@CUDA
def test_triton_unpack_of_every_nibble() -> None:
    """Every nibble value at every position of a word, against the definition ``(nibble - 8) · scale``."""
    nibbles = torch.arange(16, dtype=torch.int64).repeat_interleave(8).view(16, 8)  # word k holds nibble k eight times
    rand = torch.randint(0, 16, (16, 8), generator=torch.Generator().manual_seed(0))
    nibbles = torch.cat([nibbles, nibbles.roll(1, 0), rand], 0)  # (48 words, 8 nibbles)
    words = (nibbles << torch.arange(0, 32, 4)).sum(-1)
    packed = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32).view(1, 12, 4)  # 12 rows of 32 inputs
    scale = torch.tensor([0.3, -1.7, 2.5e-3]).repeat(4).view(1, 12, 1).to(torch.bfloat16)  # one group per row
    expected = ((nibbles.reshape(1, 12, 32) - 8).float() * scale.float()).to(torch.bfloat16)
    got = experts.dequantize_int4(packed.cuda(), scale.cuda(), 32, "triton").cpu()
    assert _same_bits(got, expected)


@CUDA
def test_triton_unpack_refuses_mismatched_shapes() -> None:
    from latent_interconnect.receivers.kimi_k26.int4_triton import dequantize_int4_triton

    packed, scale = _random((2, 3, 64, 32), torch.bfloat16, "cuda")
    with pytest.raises(ValueError, match="does not match"):
        dequantize_int4_triton(packed, scale[:, :, :1], 32)


def test_the_default_unpack_needs_no_triton(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "triton", None)  # importing triton now raises ImportError
    monkeypatch.delitem(sys.modules, "latent_interconnect.receivers.kimi_k26.int4_triton", raising=False)
    packed, scale = _random(SHAPES[1], torch.bfloat16, "cpu")
    assert _same_bits(experts.dequantize_int4(packed, scale, 32), experts.dequantize_int4(packed, scale, 32, "eager"))


def test_triton_refuses_a_cpu_tensor_rather_than_unpacking_eagerly() -> None:
    packed, scale = _random(SHAPES[1], torch.bfloat16, "cpu")
    with pytest.raises(ValueError, match="CUDA tensors"):
        experts.dequantize_int4(packed, scale, 32, "triton")


@pytest.mark.parametrize("group_size", [0, -8, 12])
def test_triton_refuses_a_group_size_that_is_not_a_positive_multiple_of_8(group_size: int) -> None:
    packed, scale = _random(SHAPES[1], torch.bfloat16, "cpu")
    with pytest.raises(ValueError, match="positive multiple of 8"):
        experts.dequantize_int4(packed, scale, group_size, "triton")


def test_triton_without_triton_fails_rather_than_unpacking_eagerly(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.delitem(sys.modules, "latent_interconnect.receivers.kimi_k26.int4_triton", raising=False)
    cuda_tensor = types.SimpleNamespace(is_cuda=True, device="cuda:0")  # the CPU suite has no CUDA tensor to give
    with pytest.raises(ImportError, match="needs Triton"):
        experts.dequantize_int4(cuda_tensor, cuda_tensor, 32, "triton")  # type: ignore[arg-type]
