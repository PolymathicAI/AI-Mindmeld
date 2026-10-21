"""``kimi-ep``'s speed options on the tiny Kimi model, on CPU: each is an explicit, recorded choice, by default on
CUDA devices the one of ``receivers.kimi_k26.ep.CUDA_DEFAULTS`` (the fast one, but the original for ``combine`` and
``compile``) and the original behaviour otherwise and whenever its value is named (``REFERENCE_OPTIONS``); the
options and their values have one definition (``SPEED_OPTIONS``).

- The choices that change no number and run on CPU (``expert_dispatch="concurrent"``, ``expert_rows="split"``) give
  bit for bit the original loss, write gradients and Adam update, in float64 and in bf16, on two CPU expert blocks,
  each alone and together; ``configure`` switches each one way and back on the loaded model, and the way back gives
  the original bits again.
- ``combine="expert"`` computes the same function in another order: it matches the original per tensor (the loss and
  every write gradient), float64 elementwise within a relative 1e-10 scaled to the tensor, and float64 differences at
  most 1e-6 times the float32 ones. The float64 model runs Kimi's float32 islands (gate, RMSNorm, attention softmax)
  in float64 for this, the original and the candidate alike.
- ``head="bf16"`` changes the training loss's rounding by design.
- ``execution()`` is the original dict when every option is the original and records every option otherwise; the
  trainer's resume check (``Trainer._check_execution``) resumes an original state only on the original options,
  refuses every other choice and the GPU defaults on it, naming the ``--option`` values that resume it, and resumes a
  state saved under the same choices. A backend on the GPU choices is a stand-in whose ``execution()`` is what
  ``KimiEP.execution`` returns for them (``ep.recorded_execution``): the suite has no CUDA device.

The CUDA-only choices (``ep.NEEDS_CUDA``) are tested in ``tests/gpu/test_kimi_ep_cuda.py`` and
``test_int4_triton.py``. Optional: needs the Kimi CPU test setup (``CONTRIBUTING.md``).
"""

# ruff: noqa: E402
# (the imports follow the optional-dependency skip)
from __future__ import annotations

import pytest

pytest.importorskip("compressed_tensors", reason="the Kimi CPU tests need compressed-tensors (CONTRIBUTING.md)")
pytestmark = pytest.mark.kimi_ep

import contextlib
import inspect
import textwrap
import types
from collections.abc import Callable, Iterator
from typing import Any, cast

import torch
from tests.kimi_tiny import SHRINK, TINY_LAYERS

from latent_interconnect.config import OptionError
from latent_interconnect.receivers import open_training
from latent_interconnect.receivers.conformance import TrainingStep, training_step
from latent_interconnect.receivers.kernel import resolve
from latent_interconnect.receivers.kimi_k26 import ep as ep_module
from latent_interconnect.receivers.kimi_k26.ep import (
    CUDA_DEFAULTS,
    NEEDS_CUDA,
    REFERENCE_OPTIONS,
    SPEED_OPTIONS,
    KimiEP,
    MemoryBudgetError,
    available_bytes,
    plan_memory,
    recorded_execution,
    resolve_options,
)
from latent_interconnect.receivers.kimi_k26.experts import moe_modules
from latent_interconnect.receivers.kimi_k26.regional import Region, compile_moe_parts, recompile_limit
from latent_interconnect.receivers.render import RenderedPrompt, SlotBinding, SlotOccurrence, Target
from latent_interconnect.run import measure as measure_stage
from latent_interconnect.run import train as train_stage
from latent_interconnect.run._main import stage_options
from latent_interconnect.training.trainer import Trainer
from latent_interconnect.writes import Entry, SlotWrites

D = SHRINK["hidden_size"]
VOCAB = SHRINK["vocab_size"] - 3
OBJECTS = ("o0", "o1", "o2", "o3")
EP2 = {"devices": ["cpu", "cpu"], "layout": "packed", "attention": "eager"}
LAYOUTS = {"packed-ep2": EP2, "packed-ep2-checkpointed": {**EP2, "checkpointing": True}}
DTYPES = {"float64": torch.float64, "bf16": torch.bfloat16}
LEGACY_KEYS = (
    "backend",
    "dtype",
    "quantization",
    "device_name",
    "device_count",
    "attention",
    "experts",
    "checkpointing",
    "revision",
    "weights",
)
"""The keys of ``execution()`` at the base commit (087e9f9): on the original options it must stay exactly these."""
EXACT = ("expert_dispatch", "expert_rows", "dequant")
"""The options whose choices change no number."""
SAME_FUNCTION = ("combine", "compile")
"""The options whose choices compute the same function in another order (rounding differs)."""
PRECISION = ("head",)
"""The options whose choices change the precision by design."""
ALL_CHOICES = [(name, value) for name, values in SPEED_OPTIONS.items() for value in values[1:]]
"""Every option's values other than the original (``ep.SPEED_OPTIONS``, the one definition)."""
CPU_CHOICES = [c for c in ALL_CHOICES if c not in NEEDS_CUDA]
"""The choices a CPU backend runs (the others need CUDA, ``ep.NEEDS_CUDA``)."""
EXACT_ON_CPU = [c for c in CPU_CHOICES if c[0] in EXACT]
SAME_FUNCTION_ON_CPU = [c for c in CPU_CHOICES if c[0] in SAME_FUNCTION]


def _prompts() -> list[RenderedPrompt]:
    """Four prompts of different lengths, each with a 2-token slot and a weighted target (``test_kimi_ep``'s)."""
    g = torch.Generator().manual_seed(3)
    out = []
    for i, n in enumerate((9, 23, 6, 14)):
        ids = tuple(int(x) for x in torch.randint(0, VOCAB, (n,), generator=g))
        slot = (SlotOccurrence("p", SlotBinding(OBJECTS[i], "c", 2), (1, 2)),)
        tokens = tuple(int(x) for x in torch.randint(0, VOCAB, (2 + i,), generator=g))
        out.append(RenderedPrompt(ids, slot, Target(tokens, tuple(0.5 + 0.1 * k for k in range(len(tokens))))))
    return out


def _writes_factory(dtype: torch.dtype) -> Any:
    """Fresh leaf writes at every layer for both slot tokens, in the body's state dtype, equal on every call."""

    def make() -> dict[str, SlotWrites]:
        g = torch.Generator().manual_seed(5)
        entries = [Entry(t, layer) for t in range(2) for layer in range(TINY_LAYERS)]
        deltas = {e: torch.randn(len(OBJECTS), D, generator=g, dtype=torch.float64).to(dtype) for e in entries}
        alphas = {e: torch.tensor(1.0 if e.layer == 0 else 0.4, dtype=dtype) for e in entries}
        for t in [*deltas.values(), *alphas.values()]:
            t.requires_grad_(True)
        return {"c": SlotWrites(OBJECTS, deltas, alphas)}

    return make


def _step(ep: KimiEP) -> TrainingStep:
    state = torch.float64 if ep.dtype == torch.float64 else torch.float32
    return training_step(ep, _prompts(), _writes_factory(state), lr=1e-2)


def _same_bits(a: TrainingStep, b: TrainingStep) -> bool:
    return all(
        torch.equal(x, y) for x, y in ((a.loss, b.loss), (a.gradients, b.gradients), (a.update, b.update))
    ) and torch.equal(a.start, b.start)


@pytest.fixture(scope="module")
def backends(tiny):
    spec, _ = tiny
    return {(d, layout): KimiEP(spec, dtype=DTYPES[d], **kw) for d in DTYPES for layout, kw in LAYOUTS.items()}


# ---------------------------------------------------------------- the exact choices


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("name", "value"), EXACT_ON_CPU)
def test_an_exact_choice_gives_the_original_bits_and_switches_back(backends, dtype, layout, name, value):
    """Loss, every write gradient and the writes after one Adam step: bit for bit the original's, and again after
    ``configure`` switches the loaded model back to the original."""
    ep = backends[(dtype, layout)]
    legacy = _step(ep)
    ep.configure(**{name: value})
    try:
        got = _step(ep)
    finally:
        ep.configure(**{name: REFERENCE_OPTIONS[name]})
    again = _step(ep)
    assert ep.options() == REFERENCE_OPTIONS
    assert not torch.equal(legacy.update, legacy.start)  # an informative step
    assert _same_bits(got, legacy)
    assert _same_bits(again, legacy)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_the_exact_choices_together_give_the_original_bits(backends, dtype, layout):
    ep = backends[(dtype, layout)]
    legacy = _step(ep)
    together = dict(EXACT_ON_CPU)  # the last CPU value of each exact option
    try:
        ep.configure(**together)
        got = _step(ep)
    finally:
        ep.configure(**REFERENCE_OPTIONS)
    assert _same_bits(got, legacy)


RTOL = 1e-10
"""The float64 criterion: every element within ``atol + RTOL·|b|``, ``atol = RTOL · max|b|`` per tensor."""
SCALING = 1e-6
"""The precision scaling: each tensor's float64 difference is at most ``SCALING`` times its float32 one."""


def _loss_and_gradients(ep: KimiEP) -> dict[str, torch.Tensor]:
    """The normalized loss and every write leaf's gradient (each delta and each α), by name."""
    writes = _writes_factory(ep.dtype)()["c"]
    terms = ep.loss(_prompts(), resolve(_prompts(), {"c": writes}))
    loss = terms.weighted_sum / terms.weight_sum
    loss.backward()
    out = {"loss": loss.detach()}
    for e, d in writes.deltas.items():
        out[f"delta {e.token} {e.layer}"] = d.grad
    for e, a in writes.alphas.items():
        out[f"alpha {e.token} {e.layer}"] = a.grad
    return out


def _differences(ep: KimiEP, name: str, value: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """``(candidate, original)``: the loss and gradients under one choice and under the original, on one backend."""
    want = _loss_and_gradients(ep)
    ep.configure(**{name: value})
    try:
        got = _loss_and_gradients(ep)
    finally:
        ep.configure(**{name: REFERENCE_OPTIONS[name]})
    return got, want


def _in_float64(fn: Any) -> Any:
    """``fn`` with its float32 casts made float64: the same source, compiled again with ``torch.float64``."""
    source = textwrap.dedent(inspect.getsource(fn))
    assert "torch.float32" in source
    namespace: dict[str, Any] = {}
    exec(
        compile(source.replace("torch.float32", "torch.float64"), f"<{fn.__qualname__} in float64>", "exec"),
        dict(fn.__globals__),
        namespace,
    )
    return namespace[fn.__name__]


@contextlib.contextmanager
def _float32_islands_in_float64(ep: KimiEP) -> Iterator[None]:
    """Run the model's float32 islands in float64 (the original and the candidate alike), so the float64 model is
    float64 throughout: Kimi's gate (its weights) and RMSNorm compute in float32 whatever the body (the remote
    ``MoEGate.forward``, ``DeepseekV3RMSNorm.forward``), and the packed attention's softmax does too (``mla_packed``,
    as the remote eager attention). Without this the comparison sees float32 rounding, not the function (measured as
    shipped: loss differences of order 1e-8, printed by the test)."""
    gate = next(layer.mlp.gate for layer in ep.layers if hasattr(layer.mlp, "_ep_blocks"))
    patched = [(type(gate), "forward"), (type(ep.layers[0].input_layernorm), "forward"), (ep_module, "mla_packed")]
    saved = [(owner, name, getattr(owner, name)) for owner, name in patched]
    for owner, name, fn in saved:
        setattr(owner, name, _in_float64(fn))
    try:
        yield
    finally:
        for owner, name, fn in saved:
            setattr(owner, name, fn)


def check_same_function(make: Callable[[torch.dtype], KimiEP], name: str, value: str) -> str:
    """Require one choice to compute the original's function to rounding; return the measured differences.

    Per compared tensor (the loss and every write gradient), on a float64 backend whose float32 islands run in float64
    (:func:`_float32_islands_in_float64`): every element within ``atol + rtol·|b|`` (rtol 1e-10, atol 1e-10 ·
    max|b|); and the float64 difference at most 1e-6 times the float32 backend's (rounding shrinks with the unit
    roundoff, a wrong formula does not), or zero where the float32 difference is zero. ``make(dtype)`` builds a
    backend (each call a fresh one, so a compiled region is the patched one).
    """
    shipped64, shipped_want64 = _differences(make(torch.float64), name, value)  # with the float32 islands
    fresh = make(torch.float64)
    with _float32_islands_in_float64(fresh):
        got64, want64 = _differences(fresh, name, value)
    got32, want32 = _differences(make(torch.float32), name, value)
    assert got64.keys() == want64.keys() == got32.keys()
    worst64 = worst32 = 0.0
    for key, b in want64.items():
        a = got64[key]
        atol = RTOL * float(b.abs().max())
        excess = (a - b).abs() - (atol + RTOL * b.abs())
        assert float(excess.max()) <= 0, f"{key}: outside the float64 tolerance by {float(excess.max()):.3e}"
        d64 = float((a - b).abs().max())
        d32 = float((got32[key] - want32[key]).abs().max())
        if d32 == 0:
            assert d64 == 0, f"{key}: float64 difference {d64:.3e} where float32's is zero"
        else:
            assert d64 <= SCALING * d32, f"{key}: float64 difference {d64:.3e} vs float32 {d32:.3e}"
        worst64 = max(worst64, d64 / float(b.abs().max()))
        worst32 = max(worst32, d32 / float(want32[key].abs().max()))
    assert worst32 > 0  # the choice reorders arithmetic: some float32 difference shows it is exercised
    loss64 = float((got64["loss"] - want64["loss"]).abs())
    loss32 = float((got32["loss"] - want32["loss"]).abs())
    shipped = float((shipped64["loss"] - shipped_want64["loss"]).abs())
    return (
        f"{name}={value}: loss |diff| float64 {loss64:.3e} (float32 islands as shipped: {shipped:.3e}), "
        f"float32 {loss32:.3e}; largest relative (to max|b|) tensor difference float64 {worst64:.3e}, "
        f"float32 {worst32:.3e}"
    )


@pytest.mark.parametrize(("name", "value"), SAME_FUNCTION_ON_CPU)
def test_a_reordering_choice_computes_the_same_function(tiny, name, value):
    """:func:`check_same_function` on two CPU expert blocks (the measured differences are printed)."""
    spec, _ = tiny
    print(check_same_function(lambda dtype: KimiEP(spec, dtype=dtype, **EP2), name, value))


def test_regional_compilation_switches_on_and_off_on_the_loaded_model(tiny):
    """``compile="regional"`` needs CUDA (refused here, ``test_choices_that_need_cuda_are_refused_on_cpu``); its module
    switch is checked here without compiling anything: on, every gate and shared expert gets a compiled ``forward``
    over its class's; off, none is left and the step is the original's bits. Running compiled:
    ``tests/gpu/test_kimi_ep_cuda.py``."""
    spec, _ = tiny
    ep = KimiEP(spec, dtype=torch.float64, **EP2)
    parts = [getattr(mlp, n) for mlp in moe_modules(ep.layers, converted=True) for n in ("gate", "shared_experts")]
    legacy = _step(ep)
    limit = recompile_limit(len(ep.layers), 1)
    compile_moe_parts(ep.layers, True, limit)
    assert parts and all(isinstance(vars(part)["forward"], Region) for part in parts)
    compile_moe_parts(ep.layers, False, limit)
    assert not any("forward" in vars(part) for part in parts)
    assert _same_bits(_step(ep), legacy)


def test_the_memory_plan_holds_the_bf16_head_copy_and_follows_configure(tiny):
    """The options' one persistent buffer, the bf16 head copy, is in the load plan: the plan under the GPU defaults
    differs from the original's by exactly that copy, in the final phase of the head's device, and ``configure`` keeps
    the plan equal to the allocated copy both ways; no other option changes the plan."""
    spec, _ = tiny
    ep = KimiEP(spec, dtype=torch.float64, **EP2)
    original = ep.memory
    defaults = plan_memory(ep.snapshot, ep.devices, ep.dtype, head=cast(Any, resolve_options(cuda=True)["head"]))
    copy = ep.head.numel() * torch.bfloat16.itemsize
    last = defaults.devices.index(ep.device_of[-1])
    assert defaults.build == original.build and defaults.convert == original.convert
    assert [f - o for f, o in zip(defaults.final, original.final, strict=True)] == [
        copy if i == last else 0 for i in range(len(defaults.devices))
    ]
    ep.configure(head="bf16")
    assert ep._head_bf16 is not None and ep._head_bf16.numel() * ep._head_bf16.element_size() == copy
    assert ep.memory == defaults
    ep.configure(head="fp32")
    assert ep._head_bf16 is None and ep.memory == original
    for name, value in [c for c in CPU_CHOICES if c[0] != "head"]:
        ep.configure(**{name: value})
        assert ep.memory == original, name
        ep.configure(**{name: REFERENCE_OPTIONS[name]})


def _rotaries(ep: KimiEP) -> list[Any]:
    return [layer.self_attn.rotary_emb for layer in ep.layers]


def _uncached(ep: KimiEP) -> None:
    """Every layer's rotary table as a fresh model has it: built on the next call (the remote module's ``None``)."""
    for rotary in _rotaries(ep):
        rotary.max_seq_len_cached = None


@pytest.mark.parametrize("dtype", DTYPES)
def test_covering_the_rotary_positions_first_changes_no_number(backends, monkeypatch, dtype):
    """The tables built ahead (here for twice the longest sequence) hold, for every position used, the bits the
    layer's own lazy build makes, and the loss, write gradients and update are bit for bit the lazy build's."""
    ep = backends[(dtype, "packed-ep2-checkpointed")]
    longest = max(len(p.input_ids) + len(p.target.token_ids) for p in _prompts())
    _uncached(ep)
    with monkeypatch.context() as m:
        m.setattr(ep, "_cover_positions", lambda _longest: None)  # the remote module's own lazy build, in the layers
        lazy = _step(ep)
    lazy_tables = [(r.cos_cached[:longest].clone(), r.sin_cached[:longest].clone()) for r in _rotaries(ep)]
    _uncached(ep)
    ep._cover_positions(2 * longest)
    ahead = _step(ep)
    assert all(r.max_seq_len_cached == 2 * longest for r in _rotaries(ep))
    for r, (cos, sin) in zip(_rotaries(ep), lazy_tables, strict=True):
        assert torch.equal(r.cos_cached[:longest], cos) and torch.equal(r.sin_cached[:longest], sin)
    assert _same_bits(ahead, lazy)


@pytest.mark.parametrize("layout", ["padded-checkpointed", "packed-ep2-checkpointed"])
def test_no_rotary_table_is_rebuilt_inside_a_checkpointed_layer(tiny, layout):
    """With checkpointing, over batches of growing longest sequences: every rotary rebuild happens before the layers
    run, none inside a layer's forward or its recompute in backward (where it would make the two differ)."""
    spec, _ = tiny
    kw = (
        {"devices": ["cpu"], "layout": "padded", "checkpointing": True}
        if layout.startswith("padded")
        else LAYOUTS[layout]
    )
    ep = KimiEP(spec, dtype=torch.float64, **kw)
    inside = [False]
    rebuilds: list[tuple[int, bool]] = []
    for rotary in _rotaries(ep):
        build = rotary._set_cos_sin_cache

        def counting(seq_len: int, device: Any, dtype: Any, build: Any = build) -> Any:
            rebuilds.append((seq_len, inside[0]))
            return build(seq_len=seq_len, device=device, dtype=dtype)

        rotary._set_cos_sin_cache = counting
    run = ep._run

    def in_layer(fn: Any, h: torch.Tensor) -> torch.Tensor:
        inside[0] = True
        try:
            return run(fn, h)
        finally:
            inside[0] = False

    ep._run = cast(Any, in_layer)
    prompts = _prompts()
    for batch in ([prompts[2]], [prompts[0], prompts[2]], [prompts[3]], prompts):  # longest 10, 11, 19, 26
        writes = _writes_factory(torch.float64)()
        terms = ep.loss(batch, resolve(batch, writes))
        inside[0] = True  # the recompute runs in backward
        (terms.weighted_sum / terms.weight_sum).backward()
        inside[0] = False
    assert [n for n, _ in rebuilds] == [
        n for n in (10, 11, 19, 26) for _ in range(TINY_LAYERS)
    ]  # each layer, each growth
    assert not any(within for _, within in rebuilds)


def test_choices_that_need_cuda_are_refused_on_cpu(tiny):
    spec, _ = tiny
    for name, value in sorted(NEEDS_CUDA):
        with pytest.raises(ValueError, match="needs CUDA devices"):
            KimiEP(spec, dtype=torch.float64, **EP2, **{name: value})
    ep = KimiEP(spec, dtype=torch.float64, **EP2)
    for name, value in sorted(NEEDS_CUDA):
        with pytest.raises(ValueError, match="needs CUDA devices"):
            ep.configure(**{name: value})
        assert ep.options() == REFERENCE_OPTIONS  # a refused choice leaves the backend as it was


# ---------------------------------------------------------------- execution facts and resume


def _execution_under(ep: KimiEP, name: str, value: str) -> dict[str, str]:
    """``ep.execution()`` with one option at ``value``: by ``configure``, or, for a choice that needs CUDA, as the
    backend records it (``ep.recorded_execution`` of its facts and options, the function ``execution()`` returns)."""
    if (name, value) in NEEDS_CUDA:
        return recorded_execution(ep.execution(), {**ep.options(), name: value})
    ep.configure(**{name: value})
    try:
        return ep.execution()
    finally:
        ep.configure(**{name: REFERENCE_OPTIONS[name]})


def _on(facts: dict[str, str]) -> Any:
    """A backend whose ``execution()`` is ``facts`` (a kimi-ep backend on other options or devices)."""
    return types.SimpleNamespace(execution=lambda: facts, reference_options=KimiEP.reference_options)


def _trainer(backend: Any) -> Any:
    """A trainer on ``backend`` with the input cache off, for its execution facts and resume check."""
    trainer: Any = types.SimpleNamespace(config=types.SimpleNamespace(name="r"), backend=backend, input_cache="off")
    trainer.execution = lambda: Trainer.execution(trainer)
    return trainer


def _check(ep: KimiEP, saved: dict[str, str]) -> None:
    """The trainer's resume check (``Trainer._check_execution``) against ``ep`` of the state a trainer saved on a
    backend whose ``execution()`` was ``saved`` (both trainers with the input cache off)."""
    then = types.SimpleNamespace(execution=lambda: saved, reference_options=KimiEP.reference_options)
    Trainer._check_execution(_trainer(ep), _trainer(then).execution())


def test_the_original_options_keep_the_original_execution_facts(backends):
    ep = backends[("float64", "packed-ep2")]
    legacy = ep.execution()
    assert tuple(legacy) == LEGACY_KEYS
    assert legacy["experts"] == "loop, 2 blocks" and legacy["attention"] == "packed-eager"
    for name, value in CPU_CHOICES:  # away and back: the same dict
        ep.configure(**{name: value})
        ep.configure(**{name: REFERENCE_OPTIONS[name]})
        assert ep.execution() == legacy


@pytest.mark.parametrize(("name", "value"), ALL_CHOICES)
def test_any_other_choice_records_every_option(backends, name, value):
    ep = backends[("float64", "packed-ep2")]
    legacy = ep.execution()
    got = _execution_under(ep, name, value)
    assert {k: got[k] for k in LEGACY_KEYS} == legacy
    assert {k: got[k] for k in got if k not in LEGACY_KEYS} == {**REFERENCE_OPTIONS, name: value}


RESUME_LEGACY = " ".join(f"--option {k}={v}" for k, v in REFERENCE_OPTIONS.items())
"""What a refused resume of an original state names: every option's original value."""


def test_the_defaults_are_the_cuda_defaults_on_cuda_and_the_original_otherwise(backends):
    assert sorted((*EXACT, *SAME_FUNCTION, *PRECISION)) == sorted(SPEED_OPTIONS)  # every option is classified here
    assert resolve_options(cuda=True) == CUDA_DEFAULTS
    assert resolve_options(cuda=False) == REFERENCE_OPTIONS
    assert resolve_options(cuda=True, **REFERENCE_OPTIONS) == REFERENCE_OPTIONS  # the original, by naming its values
    assert list(CUDA_DEFAULTS) == list(REFERENCE_OPTIONS)
    # every option is fast on GPUs but combine and compile, which keep the original
    assert [k for k, v in REFERENCE_OPTIONS.items() if CUDA_DEFAULTS[k] == v] == ["combine", "compile"]
    assert backends[("float64", "packed-ep2")].options() == REFERENCE_OPTIONS  # a CPU backend on its defaults


def test_an_original_state_resumes_on_the_original_options(backends):
    ep = backends[("float64", "packed-ep2")]
    _check(ep, ep.execution())


def test_the_gpu_defaults_refuse_an_original_state_naming_the_options_that_resume_it(backends):
    ep = backends[("float64", "packed-ep2")]
    legacy = ep.execution()
    gpu = recorded_execution(legacy, resolve_options(cuda=True))  # what a backend on the CUDA defaults records
    assert {k: gpu[k] for k in CUDA_DEFAULTS} == CUDA_DEFAULTS  # recorded, never absent or "default"
    with pytest.raises(ValueError) as refused:
        _check(_on(gpu), legacy)
    assert str(refused.value).endswith(f"resume with the options the run started with: {RESUME_LEGACY}")
    _check(_on(gpu), gpu)  # the defaults save and resume
    ep.configure(**REFERENCE_OPTIONS)  # the explicit original options resume the original state
    _check(ep, legacy)


@pytest.mark.parametrize(("name", "value"), ALL_CHOICES)
def test_every_other_choice_refuses_an_original_state_and_resumes_its_own(backends, name, value):
    ep = backends[("float64", "packed-ep2")]
    legacy = ep.execution()
    other = _execution_under(ep, name, value)
    with pytest.raises(ValueError, match=rf"differing: \[.*'{name}'.*\]\); resume with the options") as refused:
        _check(_on(other), legacy)
    assert str(refused.value).endswith(f": {RESUME_LEGACY}")
    _check(_on(other), other)  # the same choices save and resume
    with pytest.raises(ValueError, match=rf"differing: \[.*'{name}'.*\]"):  # and the original options refuse its state
        _check(ep, other)


@pytest.mark.parametrize(("name", "value"), CPU_CHOICES)
def test_an_option_reaches_kimi_ep_through_train_and_measure(tiny, name, value):
    """``--option NAME=VALUE`` of ``train`` and ``measure``: the stage hands it on as a backend option, which
    ``open_training`` binds to the constructor (as ``Trainer.from_config`` opens the backend)."""
    spec, _ = tiny
    for stage in (train_stage.STAGE, measure_stage.STAGE):
        options = stage_options({name: value}, stage.run, stage.config)
        backend = open_training(spec, options=options, devices=["cpu"], dtype=torch.float64, layout="padded")
        assert backend.execution()[name] == value
    with pytest.raises(OptionError, match=name):
        open_training(spec, options={name: "fastest"}, devices=["cpu"], dtype=torch.float64, layout="padded")


def test_the_gpu_defaults_are_recorded_as_the_explicit_options_and_resume_themselves(backends):
    """On CUDA the defaults resolve to ``CUDA_DEFAULTS``: the same execution facts as naming each of them, every
    effective value recorded (no preset label), and they resume themselves; naming the original values records the
    original facts exactly. The facts are the CPU backend's (the options' recording does not depend on them)."""
    facts = backends[("float64", "packed-ep2")].execution()
    defaults = recorded_execution(facts, resolve_options(cuda=True))
    assert defaults == recorded_execution(facts, resolve_options(cuda=True, **CUDA_DEFAULTS))
    assert defaults == {**facts, **CUDA_DEFAULTS}
    assert recorded_execution(facts, resolve_options(cuda=True, **REFERENCE_OPTIONS)) == facts
    backend = types.SimpleNamespace(execution=lambda: defaults, reference_options=KimiEP.reference_options)
    _check(cast(Any, backend), defaults)


def test_every_caller_gets_the_values_checked():
    """``resolve_options`` (the constructor's, ``configure``'s and a driver's way in) refuses an unknown name, an
    unknown value and an empty one, naming the allowed values; None takes the default."""
    assert resolve_options(cuda=True, head=None) == CUDA_DEFAULTS
    with pytest.raises(ValueError, match=r"expert_rows='splitt': one of \['slice', 'split'\]"):
        resolve_options(cuda=False, expert_rows="splitt")
    with pytest.raises(ValueError, match=r"head='': one of \['fp32', 'bf16'\]"):
        resolve_options(cuda=True, head="")
    with pytest.raises(ValueError, match="not kimi-ep speed options: \\['fast'\\]"):
        resolve_options(cuda=True, fast="yes")


def test_the_constructor_and_configure_check_the_values(tiny):
    spec, _ = tiny
    with pytest.raises(ValueError, match="one of"):
        KimiEP(spec, dtype=torch.float64, **EP2, combine=cast(Any, "anywhere"))
    ep = KimiEP(spec, dtype=torch.float64, **EP2)
    with pytest.raises(ValueError, match="one of"):
        ep.configure(expert_dispatch=cast(Any, "parallel"))
    assert ep.options() == REFERENCE_OPTIONS


def test_configure_restores_the_options_when_applying_them_fails(tiny, monkeypatch):
    """Any failure while the options are applied (here an injected one) leaves the backend on its previous options,
    applied again, and is raised."""
    spec, _ = tiny
    ep = KimiEP(spec, dtype=torch.float64, **EP2)
    legacy = _step(ep)
    real = ep._apply_options
    calls = []

    def failing() -> None:
        calls.append(dict(ep._options))
        if len(calls) == 1:
            raise RuntimeError("injected")
        real()

    monkeypatch.setattr(ep, "_apply_options", failing)
    with pytest.raises(RuntimeError, match="injected"):
        ep.configure(head="bf16", expert_rows="split", layout="padded")
    assert calls[0]["head"] == "bf16" and calls[1] == REFERENCE_OPTIONS  # tried, then restored and re-applied
    assert ep.options() == REFERENCE_OPTIONS and ep.layout == "packed" and ep._head_bf16 is None
    monkeypatch.undo()
    assert _same_bits(_step(ep), legacy)


def test_the_bf16_head_copy_counts_the_allocators_reusable_cache(monkeypatch):
    """``configure``'s check of the bf16 head copy on a GPU (monkeypatched device queries; no GPU): the allocator's
    reusable cache counts beside the driver's free memory, and a device without room for the copy refuses, naming
    both."""
    gib = 1 << 30
    head = types.SimpleNamespace(device=torch.device("cuda:3"), numel=lambda: gib)  # a 2 GiB bf16 copy
    backend: Any = types.SimpleNamespace(head=head, _head_bf16=None)
    memory = {"free": 0, "reserved": 0, "allocated": 0}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _d: (memory["free"], 80 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _d: memory["reserved"])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _d: memory["allocated"])
    memory.update(free=gib // 2, reserved=60 * gib, allocated=57 * gib)  # 0.5 GiB free, 3 GiB reusable in the cache
    KimiEP._check_head_copy(backend, {"head": "bf16"})
    memory.update(free=gib // 2, reserved=60 * gib, allocated=60 * gib)  # truly full
    with pytest.raises(MemoryBudgetError, match=r"0.50 GiB free on the device and 0.00 GiB reusable"):
        KimiEP._check_head_copy(backend, {"head": "bf16"})
    assert available_bytes(torch.device("cuda:3")) == (gib // 2, 0)
