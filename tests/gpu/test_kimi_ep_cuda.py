"""``kimi-ep``'s CUDA-only speed options on the tiny Kimi model, on a GPU (``--gpu``; ``tests/conftest.py``).

The CUDA-only choices (``ep.NEEDS_CUDA``), on a GPU:

- ``compile="regional"`` computes the original's function to rounding (the criterion of
  ``tests/unit/test_kimi_ep_options.check_same_function``, float64 and float32 on one GPU, eager attention), and does
  not recompile across batches of other sizes and lengths (counted by ``regional.dynamo_counts``, from a reset dynamo);
- ``expert_dispatch="streams"`` gives the serial dispatch's bits on two GPUs (deterministic mode), skipped with fewer.
- checkpointing with ``compile="regional"`` updates at growing longest sequences, first update included.

Optional: needs the Kimi CPU test setup (``CONTRIBUTING.md``) and a GPU.
"""

# ruff: noqa: E402
# (the imports follow the optional-dependency skip)
from __future__ import annotations

import pytest

pytest.importorskip("compressed_tensors", reason="the Kimi tests need compressed-tensors (CONTRIBUTING.md)")
pytestmark = pytest.mark.gpu

import contextlib
from collections.abc import Iterator

import torch
from tests.unit.test_kimi_ep_options import _prompts, _same_bits, _step, _writes_factory, check_same_function

from latent_interconnect.receivers.kernel import resolve
from latent_interconnect.receivers.kimi_k26.ep import KimiEP
from latent_interconnect.receivers.kimi_k26.regional import dynamo_counts

ONE_GPU = {"devices": ["cuda:0", "cuda:0"], "layout": "packed", "attention": "eager", "expert_gemm": "loop"}


@pytest.fixture
def fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_regional_compilation_computes_the_same_function(tiny, fresh_dynamo):
    spec, _ = tiny
    print(check_same_function(lambda dtype: KimiEP(spec, dtype=dtype, **ONE_GPU), "compile", "regional"))


def test_regional_compilation_does_not_recompile_across_batches(tiny, fresh_dynamo):
    spec, _ = tiny
    ep = KimiEP(spec, dtype=torch.bfloat16, devices=["cuda:0", "cuda:0"], compile="regional")
    prompts = _prompts()
    frames = []
    for batch in (prompts, prompts[:3], prompts[1:], prompts[:2], prompts[2:3]):  # other sizes and lengths
        writes = _writes_factory(torch.float32)()
        ep.loss(batch, resolve(batch, writes))
        frames.append(dynamo_counts()["frames_compiled"])
    print(f"frames compiled after each batch: {frames}")
    assert frames[0] > 0 and frames[1:] == [frames[1]] * (len(frames) - 1)


@contextlib.contextmanager
def deterministic_cublas(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """torch's deterministic mode with cuBLAS's deterministic workspace, within the block only.

    ``CUBLAS_WORKSPACE_CONFIG`` is set for the block only (``monkeypatch.setenv``, undone by the caller's monkeypatch).
    torch checks it on every cuBLAS call in deterministic mode; cuBLAS applies it to the handles created after it is
    set, so it binds fully when the block is the process's first to use cuBLAS on those devices (run the test alone:
    ``pytest --gpu -k streams``).
    """
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)


@pytest.fixture
def deterministic(monkeypatch):
    with deterministic_cublas(monkeypatch):
        yield


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="side-stream copies need two GPUs")
def test_streams_give_the_serial_bits(tiny, deterministic):
    spec, _ = tiny
    ep = KimiEP(spec, dtype=torch.bfloat16, devices=["cuda:0", "cuda:1"], expert_dispatch="serial", compile="none")
    serial = _step(ep)
    ep.configure(expert_dispatch="streams")
    assert _same_bits(_step(ep), serial)


def test_checkpointed_regional_updates_at_growing_lengths(tiny, fresh_dynamo):
    """Checkpointing on (the GPU default) and ``compile="regional"``: the first update and later ones at growing
    longest sequences run (no ``CheckpointError`` from a rotary rebuild inside a checkpointed layer)."""
    spec, _ = tiny
    ep = KimiEP(spec, dtype=torch.bfloat16, devices=["cuda:0", "cuda:0"], checkpointing=True, compile="regional")
    assert ep.checkpointing and ep.options()["compile"] == "regional"
    prompts = _prompts()
    for batch in ([prompts[2]], [prompts[0], prompts[2]], [prompts[3]], prompts):  # longest 10, 11, 19, 26
        writes = _writes_factory(torch.float32)()
        leaves = [*writes["c"].deltas.values(), *writes["c"].alphas.values()]
        terms = ep.loss(batch, resolve(batch, writes))
        (terms.weighted_sum / terms.weight_sum).backward()
        assert all(leaf.grad is not None and bool(torch.isfinite(leaf.grad).all()) for leaf in leaves)
