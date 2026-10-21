"""``kimi-ep`` on CPU, float64, on the tiny Kimi model: its loss, connector gradients and one optimizer update equal
the ``hf`` reference's (``receivers.conformance``), in every configuration that runs on CPU (padded; packed with the
eager attention; expert parallelism of degree 1 and 2); its in-job scores equal ``kimi-stream``'s.

The reference: ``hf`` on the same model, with Kimi's remote ``moe_infer`` unwrapped for the reference's own computation
only (``weights.moe_gradients``, restored right after). Without that, ``hf`` returns the same loss but drops the routed
experts from every gradient (measured: gradients off by more than their own scale; the parity check refuses it,
``test_the_unprepared_reference_fails_parity``). ``kimi-ep`` never needs it (its
expert-parallel ``moe_infer`` replaces the remote one), and every test computes ``kimi-ep`` with the remote code as
shipped, which :func:`_unprepared` asserts. The grouped expert GEMM and the variable-length attention need CUDA and
are not tested here. Optional: needs the Kimi CPU test setup (``CONTRIBUTING.md``), and skips without it.
"""

# ruff: noqa: E402
# (the imports follow the optional-dependency skip)
from __future__ import annotations

import pytest

pytest.importorskip("compressed_tensors", reason="the Kimi CPU tests need compressed-tensors (CONTRIBUTING.md)")
pytestmark = pytest.mark.kimi_ep

import contextlib
import math
from collections.abc import Iterator
from pathlib import Path

import torch
from compressed_tensors.compressors import BaseCompressor, ModelCompressor
from huggingface_hub import snapshot_download
from tests.kimi_tiny import SHRINK, TINY_LAYERS, build_tiny_kimi, write_dense

from latent_interconnect.config import OptionError
from latent_interconnect.receivers import open_training
from latent_interconnect.receivers.base import Readout, TrainingBackend
from latent_interconnect.receivers.conformance import check_training_parity, training_step
from latent_interconnect.receivers.kernel import BatchPlan, resolve
from latent_interconnect.receivers.kimi_k26.ep import KimiEP
from latent_interconnect.receivers.kimi_k26.experts import dequantize_int4
from latent_interconnect.receivers.kimi_k26.stream import KimiStream
from latent_interconnect.receivers.kimi_k26.weights import TransientLinear
from latent_interconnect.receivers.kimi_k26.weights import moe_gradients as moe_gradients_for
from latent_interconnect.receivers.render import RenderedPrompt, SlotBinding, SlotOccurrence, Target
from latent_interconnect.run import measure as measure_stage
from latent_interconnect.run import train as train_stage
from latent_interconnect.run._main import stage_options
from latent_interconnect.writes import Entry, SlotWrites

D = SHRINK["hidden_size"]
VOCAB = SHRINK["vocab_size"] - 3
OBJECTS = ("o0", "o1", "o2", "o3")
CPU_CONFIGS = {
    "padded": {"devices": ["cpu"], "layout": "padded"},
    "packed-eager": {"devices": ["cpu"], "layout": "packed", "attention": "eager"},
    "padded-ep2": {"devices": ["cpu", "cpu"], "layout": "padded"},
    "packed-ep2": {"devices": ["cpu", "cpu"], "layout": "packed", "attention": "eager"},
    # activation checkpointing recomputes each layer in backward
    "padded-checkpointed": {"devices": ["cpu"], "layout": "padded", "checkpointing": True},
    "packed-ep2-checkpointed": {
        "devices": ["cpu", "cpu"],
        "layout": "packed",
        "attention": "eager",
        "checkpointing": True,
    },
}
TOL = 1e-12  # float64 against the same arithmetic; measured exactly 0 on this fixture


def _prompts(with_target: bool = True) -> list[RenderedPrompt]:
    """Four prompts of different lengths, each with a 2-token slot, a target of 2-5 tokens and positional weights."""
    g = torch.Generator().manual_seed(3)
    out = []
    for i, n in enumerate((9, 23, 6, 14)):
        ids = tuple(int(x) for x in torch.randint(0, VOCAB, (n,), generator=g))
        slot = (SlotOccurrence("p", SlotBinding(OBJECTS[i], "c", 2), (1, 2)),)
        tokens = tuple(int(x) for x in torch.randint(0, VOCAB, (2 + i,), generator=g))
        target = Target(tokens, tuple(0.5 + 0.1 * k for k in range(len(tokens)))) if with_target else None
        out.append(RenderedPrompt(ids, slot, target))
    return out


def _slot_writes() -> SlotWrites:
    """Leaf writes at every layer for both slot tokens (layer 0 at α = 1, deeper ones at α = 0.4)."""
    g = torch.Generator().manual_seed(5)
    entries = [Entry(t, layer) for t in range(2) for layer in range(TINY_LAYERS)]
    deltas = {e: torch.randn(len(OBJECTS), D, generator=g, dtype=torch.float64).requires_grad_(True) for e in entries}
    alphas = {e: torch.tensor(1.0 if e.layer == 0 else 0.4, dtype=torch.float64, requires_grad=True) for e in entries}
    return SlotWrites(OBJECTS, deltas, alphas)


def _writes() -> dict[str, SlotWrites]:
    """The conformance factory: fresh leaf writes, equal on every call."""
    return {"c": _slot_writes()}


def _leaves(writes: SlotWrites) -> list[torch.Tensor]:
    return [*writes.deltas.values(), *writes.alphas.values()]


def _loss_and_grads(backend: TrainingBackend) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """The loss terms' weighted sum and every leaf gradient of the normalized loss, on a fresh set of writes."""
    prompts, writes = _prompts(), _slot_writes()
    terms = backend.loss(prompts, resolve(prompts, {"c": writes}))
    (terms.weighted_sum / terms.weight_sum).backward()
    grads = torch.cat([leaf.grad.flatten() for leaf in _leaves(writes)])
    return terms.weighted_sum.detach(), terms.weight_sum.detach(), [grads]


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def eps(tiny):
    spec, _ = tiny
    return {name: KimiEP(spec, dtype=torch.float64, **kw) for name, kw in CPU_CONFIGS.items()}


@pytest.fixture
def prepared(tiny):
    """A context for the reference's own computation: Kimi's remote ``moe_infer`` unwrapped, restored on exit."""
    spec, _ = tiny
    path = Path(snapshot_download(spec.model_id, revision=spec.revision, local_files_only=True))
    return lambda: moe_gradients_for(path)


def _unprepared(tiny) -> None:
    """Assert the remote ``moe_infer`` is as shipped (still ``@torch.no_grad``): no test left the process prepared."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    spec, _ = tiny
    path = snapshot_download(spec.model_id, revision=spec.revision, local_files_only=True)
    moe = get_class_from_dynamic_module("modeling_deepseek.DeepseekV3MoE", path)
    assert hasattr(moe.moe_infer, "__wrapped__"), "the remote moe_infer was left unwrapped"


# ---------------------------------------------------------------- equal to hf


@pytest.mark.parametrize("config", CPU_CONFIGS)
def test_loss_gradients_and_one_update_equal_hf(tiny, eps, hf_on_tiny, prepared, config):
    """``receivers.conformance``: the loss, every write gradient and the writes after one Adam step equal the prepared
    ``hf`` reference's, in every CPU configuration."""
    prompts = _prompts()
    _unprepared(tiny)
    got = training_step(eps[config], prompts, _writes, lr=1e-2)
    with prepared():
        want = training_step(hf_on_tiny, prompts, _writes, lr=1e-2)
    _unprepared(tiny)
    print(
        f"{config}: |Δ loss| {float((got.loss - want.loss).abs()):.3e}, max |Δ grad| "
        f"{float((got.gradients - want.gradients).abs().max()):.3e}"
        f" (grad scale {float(want.gradients.abs().max()):.3e})"
    )
    check_training_parity(got, want, atol=TOL)


def test_the_unprepared_reference_fails_parity(tiny, eps, hf_on_tiny):
    """Kimi's remote ``moe_infer`` is ``@torch.no_grad``, so hf through it as shipped gives kimi-ep's loss but not
    its gradients. hf refuses that model (its declared preparation is not applied); with
    the declaration removed, as if a receiver config had forgotten it, parity is the next line that catches it."""
    prompts = _prompts()
    _unprepared(tiny)
    got = training_step(eps["padded"], prompts, _writes, lr=1e-2)
    with pytest.raises(ValueError, match="enable_moe_gradients"):
        training_step(hf_on_tiny, prompts, _writes, lr=1e-2)
    hf_on_tiny.spec = hf_on_tiny.spec.model_copy(update={"training_preparation": None})
    blind = training_step(hf_on_tiny, prompts, _writes, lr=1e-2)
    assert float((got.loss - blind.loss).abs()) <= TOL  # the loss is right
    with pytest.raises(ValueError, match=r"^the backend differs from the reference: gradients: max \|difference\|"):
        check_training_parity(got, blind, atol=TOL)


def test_loss_weights_follow_the_target_weights(eps):
    """``Σ w·ℓ`` and ``Σ w`` are un-normalized: doubling every weight doubles both (the trainer normalizes)."""
    prompts = _prompts()
    doubled = [
        RenderedPrompt(p.input_ids, p.slots, Target(p.target.token_ids, tuple(2 * w for w in p.target.weights)))
        for p in prompts
    ]
    with torch.no_grad():
        once = eps["padded"].loss(prompts, resolve(prompts, {"c": _slot_writes()}))
        twice = eps["padded"].loss(doubled, resolve(doubled, {"c": _slot_writes()}))
    assert torch.allclose(twice.weighted_sum, 2 * once.weighted_sum, rtol=0, atol=TOL)
    assert torch.equal(twice.weight_sum, 2 * once.weight_sum)


# ---------------------------------------------------------------- in-job scores equal kimi-stream's


class _Fixed:
    """A write source over fixed writes."""

    def __init__(self, writes: SlotWrites) -> None:
        self._writes = writes

    @property
    def bank(self):
        return None

    @property
    def entries(self):
        return tuple(self._writes.deltas)

    def writes_for(self, object_ids):
        rows = [self._writes.object_ids.index(o) for o in object_ids]
        return SlotWrites(
            tuple(object_ids),
            {e: d[rows].detach() for e, d in self._writes.deltas.items()},
            {e: a.detach() for e, a in self._writes.alphas.items()},
        )


@pytest.mark.parametrize("config", ["padded", "packed-eager"])
def test_scores_equal_kimi_streams(tiny, eps, config):
    spec, _ = tiny
    whole = BatchPlan(budget_tokens=10**6)
    stream = KimiStream(spec, device="cpu", dtype=torch.float64, plan=whole)
    ep = KimiEP(spec, dtype=torch.float64, plan=whole, **CPU_CONFIGS[config])
    sources = {"c": _Fixed(_slot_writes())}
    for readout, with_target in (
        (Readout(kind="next_token", token_ids=(5, 9)), False),
        (Readout(kind="target_nll"), True),
    ):
        prompts = _prompts(with_target)
        got, want = ep.score(prompts, sources, readout), stream.score(prompts, sources, readout)
        assert got.dtype == want.dtype == torch.float64
        assert float((got - want).abs().max()) <= TOL


def test_target_nll_is_the_unweighted_per_prompt_mean_of_the_loss_tokens(eps):
    """The readout and the loss read the same token NLLs: with unit weights, Σ w·ℓ is the sum of per-prompt NLL sums."""
    prompts = _prompts()
    unit = [
        RenderedPrompt(p.input_ids, p.slots, Target(p.target.token_ids, (1.0,) * len(p.target.token_ids)))
        for p in prompts
    ]
    sources = {"c": _Fixed(_slot_writes())}
    nll = eps["padded"].score(unit, sources, Readout(kind="target_nll"))
    with torch.no_grad():
        terms = eps["padded"].loss(unit, resolve(unit, {"c": sources["c"].writes_for(OBJECTS)}))
    total = sum(float(nll[i]) * len(p.target.token_ids) for i, p in enumerate(unit))
    assert math.isclose(float(terms.weighted_sum), total, rel_tol=0, abs_tol=1e-10)


# ---------------------------------------------------------------- the INT4 experts


def test_int4_dequantization_is_compressed_tensors_own():
    compression = ModelCompressor.from_compression_config(
        {
            "config_groups": {
                "g": {
                    "targets": ["Linear"],
                    "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 32},
                }
            },
            "format": "pack-quantized",
            "quant_method": "compressed-tensors",
        }
    ).quantization_config
    args = compression.config_groups["g"].weights
    compressor = BaseCompressor.load_from_registry("pack-quantized")
    g = torch.Generator().manual_seed(0)
    packed, scales, dense = [], [], []
    for _ in range(3):
        w = torch.randn(48, 64, generator=g)
        scale = (w.abs().reshape(48, -1, 32).amax(-1) / 7).to(torch.bfloat16)
        c = compressor.compress_weight(w, scale.float(), args)
        packed.append(c["weight_packed"])
        scales.append(scale)
        dense.append(compressor.decompress_weight({**c, "weight_scale": scale}, args))
    got = dequantize_int4(torch.stack(packed), torch.stack(scales), 32)
    assert torch.equal(got, torch.stack(dense))  # bit-identical, in the scale's dtype


def test_int4_experts_equal_their_dequantized_twin(hub):
    spec_q, twin = build_tiny_kimi(hub, "kimi-tiny-int4", layers=TINY_LAYERS, quantized=True)
    spec_d = write_dense(twin, hub, "kimi-tiny-twin", layers=TINY_LAYERS, seed=0)
    results = []
    for spec in (spec_q, spec_d):
        ep = KimiEP(spec, dtype=torch.float64, **CPU_CONFIGS["packed-ep2"])
        results.append(_loss_and_grads(ep))
    (q_sum, _, (q_grads,)), (d_sum, _, (d_grads,)) = results
    assert float((q_sum - d_sum).abs()) <= TOL
    assert float((q_grads - d_grads).abs().max()) <= TOL


class _Compressed:
    """A stand-in compressed linear: counts how often its weight is decompressed."""

    def __init__(self, w: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        self.w, self.bias, self.compressor, self.decompressed = w, bias, self, 0

    def decompress_module(self, _module):
        self.decompressed += 1
        return self.w


@pytest.mark.parametrize("with_bias", [False, True])
def test_transient_linear_gives_the_dense_input_gradient_and_keeps_no_weight(with_bias):
    """TransientLinear's output and input gradient equal the dense linear's (bias included, frozen), and backward
    decompresses the weight again: nothing dense is kept for it (the memory claim)."""
    g = torch.Generator().manual_seed(1)
    w = torch.randn(5, 7, generator=g, dtype=torch.float64)
    b = torch.randn(5, generator=g, dtype=torch.float64) if with_bias else None
    x1 = torch.randn(3, 7, generator=g, dtype=torch.float64, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    module = _Compressed(w, b)
    y1 = TransientLinear.apply(x1, module)
    assert module.decompressed == 1
    y2 = torch.nn.functional.linear(x2, w, b)
    assert torch.equal(y1, y2)
    y1.pow(2).sum().backward()
    y2.pow(2).sum().backward()
    assert torch.equal(x1.grad, x2.grad)
    assert module.decompressed == 2  # backward decompressed again: the forward kept no dense weight


def test_expert_blocks_dequantize_again_in_backward(eps):
    """The expert blocks keep no dense weight for backward: each projection is dequantized afresh in backward."""
    from latent_interconnect.receivers.kimi_k26.experts import ExpertBlock

    calls: list[str] = []
    original = ExpertBlock.weight

    def counting(self, name):
        calls.append(name)
        return original(self, name)

    prompts, writes = _prompts(), _slot_writes()
    ExpertBlock.weight = counting
    try:
        terms = eps["padded"].loss(prompts, resolve(prompts, {"c": writes}))
        forward = len(calls)
        (terms.weighted_sum / terms.weight_sum).backward()
    finally:
        ExpertBlock.weight = original
    assert forward > 0
    assert len(calls) - forward == forward  # three per block call in forward, three again in backward


# ---------------------------------------------------------------- contract details


def test_opened_through_the_receiver_config(tiny):
    spec, _ = tiny
    backend = open_training(spec, devices=["cpu"], dtype=torch.float64, layout="padded")
    assert isinstance(backend, KimiEP)


def test_open_training_runs_the_runtime_gradient_check_once(tiny, monkeypatch):
    """``open_training`` runs ``receivers.gradients.check_gradients`` once, at open; the loss does not repeat it."""
    import latent_interconnect.receivers as receivers

    spec, _ = tiny
    calls = []
    real = receivers.check_gradients
    monkeypatch.setattr(receivers, "check_gradients", lambda b, s: (calls.append(1), real(b, s))[1])
    backend = open_training(spec, devices=["cpu", "cpu"], dtype=torch.float64, layout="packed", attention="eager")
    assert len(calls) == 1
    backend.loss(_prompts(), resolve(_prompts(), {"c": _slot_writes()}))
    assert len(calls) == 1


@contextlib.contextmanager
def _routes(ep: KimiEP) -> Iterator[dict[int, set[int]]]:
    """Record, per MoE layer, the experts its ``moe_infer`` dispatches tokens to (what the experts really receive)."""
    seen: dict[int, set[int]] = {}
    saved = {}
    for li, layer in enumerate(ep.layers):
        if hasattr(layer.mlp, "_ep_blocks"):
            saved[li] = inner = layer.mlp.moe_infer

            def record(x, ids, w, inner=inner, li=li):
                seen.setdefault(li, set()).update(ids.flatten().tolist())
                return inner(x, ids, w)

            layer.mlp.moe_infer = record
    try:
        yield seen
    finally:
        for li, inner in saved.items():
            ep.layers[li].mlp.moe_infer = inner


def test_the_parity_prompts_reach_every_expert_in_every_layer(eps):
    """The parity tests are what check each routed expert (inside ``experts._Experts``, a custom Function): their
    prompts must route tokens to all of them (measured: all 16 experts of both MoE layers)."""
    ep = eps["packed-ep2"]
    prompts = _prompts()
    with _routes(ep) as seen:
        ep.loss(prompts, resolve(prompts, {"c": _slot_writes()}))
    assert len(seen) == 2 and seen == {li: set(range(SHRINK["n_routed_experts"])) for li in seen}


def test_execution_and_placement(eps):
    ep = eps["packed-ep2"]
    assert ep.execution()["attention"] == "packed-eager"
    assert ep.execution()["experts"] == "loop, 2 blocks"
    assert ep.execution()["device_count"] == "1"  # one distinct device, two expert blocks on it
    assert ep.execution()["checkpointing"] == "none"  # off by default on CPU
    assert eps["packed-ep2-checkpointed"].execution()["checkpointing"] == "every layer"
    assert set(ep.placement()) == set(range(TINY_LAYERS))


def test_checkpointing_on_and_off_agree(eps):
    """Checkpointing is execution only: recomputing each layer in backward gives the same loss, gradients and update
    (float64, CPU; measured equal exactly)."""
    off = training_step(eps["packed-ep2"], _prompts(), _writes, lr=1e-2)
    on = training_step(eps["packed-ep2-checkpointed"], _prompts(), _writes, lr=1e-2)
    assert torch.equal(on.loss, off.loss) and torch.equal(on.gradients, off.gradients)
    assert torch.equal(on.update, off.update) and not torch.equal(off.update, off.start)


def test_the_checkpointing_option_reaches_kimi_ep_through_measure_and_is_the_run_configs_in_train(tiny):
    """``--option checkpointing=...`` of ``measure``: the stage hands it on as a backend option, which ``open_training``
    binds to the constructor (as ``Trainer.from_config`` opens the backend), and ``execution()``, which both stages
    record, reports it. ``train`` refuses it: a run sets ``RunConfig.checkpointing``, which
    ``Trainer.from_config`` hands on as the same option."""
    spec, _ = tiny
    for text, shown in (("true", "every layer"), ("false", "none")):
        options = stage_options({"checkpointing": text}, measure_stage.STAGE.run, measure_stage.STAGE.config)
        backend = open_training(spec, options=options, devices=["cpu"], dtype=torch.float64, layout="padded")
        assert backend.execution()["checkpointing"] == shown
    with pytest.raises(OptionError, match="checkpointing"):
        stage_options({"checkpointing": "false"}, train_stage.STAGE.run, train_stage.STAGE.config)


def test_options_are_checked(tiny):
    spec, _ = tiny
    with pytest.raises(ValueError, match="'varlen' needs the packed layout"):
        KimiEP(spec, devices=["cpu"], layout="padded", attention="varlen")
    with pytest.raises(ValueError, match="need CUDA devices"):
        KimiEP(spec, devices=["cpu"], expert_gemm="grouped")
    with pytest.raises(ValueError, match="do not split evenly"):
        KimiEP(spec, devices=["cpu"] * 3, dtype=torch.float64)


def test_the_loss_checks_writes_before_targets(eps):
    prompts = _prompts(with_target=False)
    deep = {TINY_LAYERS: resolve(prompts, {"c": _slot_writes()})[0]}
    with pytest.raises(ValueError, match=rf"writes at layers \[{TINY_LAYERS}\]"):
        eps["padded"].loss(prompts, deep)
    with pytest.raises(ValueError, match=r"loss: prompts \[0, 1, 2, 3\] have no target tokens"):
        eps["padded"].loss(prompts, {})


def test_token_nll_equals_hfs_and_is_detached(tiny, eps, hf_on_tiny, prepared):
    """``LossTerms.token_nll``: per prompt, detached, equal to hf's; with the weights it sums to ``weighted_sum``."""
    prompts = _prompts()
    _unprepared(tiny)
    got = eps["packed-eager"].loss(prompts, resolve(prompts, {"c": _slot_writes()}))
    with prepared():
        want = hf_on_tiny.loss(prompts, resolve(prompts, {"c": _slot_writes()}))
    assert len(got.token_nll) == len(prompts)
    for g, w, p in zip(got.token_nll, want.token_nll, prompts, strict=True):
        assert g.shape == (len(p.target.token_ids),) and not g.requires_grad
        assert float((g - w).abs().max()) <= TOL
    weights = [torch.tensor(p.target.weights, dtype=g.dtype) for g, p in zip(got.token_nll, prompts, strict=True)]
    rebuilt = sum(float((w * g).sum()) for w, g in zip(weights, got.token_nll, strict=True))
    assert math.isclose(rebuilt, float(got.weighted_sum), rel_tol=0, abs_tol=1e-10)
    assert got.weighted_sum.requires_grad


def test_configure_switches_execution_options_on_the_loaded_model(tiny):
    """A reconfigured instance of one loaded model equals a fresh one with those options."""
    spec, _ = tiny
    switched = KimiEP(spec, dtype=torch.float64, **CPU_CONFIGS["packed-ep2"])
    switched.configure(layout="padded")
    assert switched.execution()["attention"] == "padded-eager"
    fresh = KimiEP(spec, dtype=torch.float64, **CPU_CONFIGS["padded-ep2"])
    got, want = _loss_and_grads(switched), _loss_and_grads(fresh)
    assert torch.equal(got[0], want[0]) and torch.equal(got[2][0], want[2][0])
    with pytest.raises(ValueError, match="'varlen' needs the packed layout"):
        switched.configure(attention="varlen")
