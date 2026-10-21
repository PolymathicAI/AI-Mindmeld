"""The prepared ``hf`` reference on the tiny Kimi model against central finite differences, in float64 on CPU.

``test_kimi_ep.py`` compares ``kimi-ep`` with this reference, so the reference itself must be right: with Kimi's
declared training preparation applied its gradient equals finite differences, and without it (the remote ``moe_infer``
as shipped, ``@torch.no_grad``) the loss is right and every write still gets a non-zero gradient, yet the gradient is
wrong, which finite differences refuse. Kimi's own code keeps float32 islands (RMSNorm, router logits, attention
softmax, YaRN tables) whatever the weight dtype, hence the step and tolerances below.

Optional: needs the Kimi CPU test setup (``CONTRIBUTING.md``), and skips without it.
"""

# ruff: noqa: E402
# (the imports follow the optional-dependency skip)
from __future__ import annotations

import pytest

pytest.importorskip("compressed_tensors", reason="the Kimi CPU tests need compressed-tensors (CONTRIBUTING.md)")
pytestmark = pytest.mark.kimi_ep

from collections.abc import Sequence
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tests.kimi_tiny import SHRINK, TINY_LAYERS, VOCAB

from latent_interconnect.receivers.render import RenderedPrompt, SlotBinding, SlotOccurrence, Target
from latent_interconnect.writes import Entry, SlotWrites

LAYERS = TINY_LAYERS
D = SHRINK["hidden_size"]


class FixedSource:
    """A write source over fixed per-object writes (what a bank holds)."""

    def __init__(self, writes: SlotWrites) -> None:
        self._writes = writes

    @property
    def bank(self):
        return None

    @property
    def entries(self):
        return tuple(self._writes.deltas)

    def writes_for(self, object_ids: Sequence[str]) -> SlotWrites:
        row = {o: i for i, o in enumerate(self._writes.object_ids)}
        idx = torch.tensor([row[o] for o in object_ids])
        return SlotWrites(
            tuple(object_ids), {e: d[idx] for e, d in self._writes.deltas.items()}, dict(self._writes.alphas)
        )


def _writes(objects: Sequence[str], entries: Sequence[tuple[int, int]], seed: int) -> FixedSource:
    g = torch.Generator().manual_seed(seed)
    deltas, alphas = {}, {}
    for token, layer in entries:
        e = Entry(token, layer)
        deltas[e] = torch.randn(len(objects), D, generator=g, dtype=torch.float64)
        alphas[e] = torch.tensor(1.0 if layer == 0 else 0.2 + 0.6 * float(torch.rand((), generator=g)))
    return FixedSource(SlotWrites(tuple(objects), deltas, alphas))


SOURCES = {
    # connector "a": 2 tokens writing every layer; connector "b": 1 token writing layers 0 and 2
    "a": _writes(["p0", "p1", "p2"], [(t, L) for t in range(2) for L in range(LAYERS)], seed=1),
    "b": _writes(["m0", "m1"], [(0, 0), (0, 2)], seed=2),
}
K = {"a": 2, "b": 1}


def _prompts(with_target: bool, with_slots: bool = True) -> list[RenderedPrompt]:
    g = torch.Generator().manual_seed(7)
    layout = [  # (length, slots as (connector, object, first position))
        (9, [("a", "p0", 3)]),
        (23, [("a", "p1", 5), ("b", "m0", 15)]),
        (6, []),
        (14, [("a", "p0", 1)]),
        (18, [("b", "m1", 17)]),
        (11, [("a", "p2", 8), ("b", "m0", 2)]),
    ]
    out = []
    for i, (n, slots) in enumerate(layout):
        ids = tuple(int(x) for x in torch.randint(0, VOCAB - 3, (n,), generator=g))
        occ = [
            SlotOccurrence(name, SlotBinding(obj, name, K[name]), tuple(range(first, first + K[name])))
            for name, obj, first in slots
        ]
        target = None
        if with_target:
            t = tuple(int(x) for x in torch.randint(0, VOCAB - 3, (2 + i % 4,), generator=g))
            target = Target(t, tuple(1.0 for _ in t))
        slots_ = tuple(sorted(occ, key=lambda o: o.positions)) if with_slots else ()
        span = tuple(range(max(0, n - 2 - i % 3), n - 1))  # 1 to 3 tokens ending at the capture position
        out.append(RenderedPrompt(ids, slots_, target, capture_position=n - 2, capture_span=span))
    return out


def _leaf_writes() -> dict[str, SlotWrites]:
    """Fresh float64 leaf copies of every connector's fixed writes (the conformance factory)."""
    out = {}
    for name, objects in (("a", ["p0", "p1", "p2"]), ("b", ["m0", "m1"])):
        w = SOURCES[name].writes_for(objects)
        deltas = {e: d.clone().requires_grad_(True) for e, d in w.deltas.items()}
        alphas = {e: a.clone().double().requires_grad_(True) for e, a in w.alphas.items()}
        out[name] = SlotWrites(w.object_ids, deltas, alphas)
    return out


def test_hf_through_kimis_remote_moe_needs_its_preparation_and_finite_differences_show_it(tiny, hf_on_tiny):
    """Kimi's remote ``moe_infer`` is ``@torch.no_grad``, so ``hf`` on Kimi without its preparation returns
    the right loss and silently drops the routed experts from the gradient. ``hf`` refuses it by the declared
    preparation (``test_kimi_ep.py::test_the_unprepared_reference_fails_parity``). With the declaration removed, as if
    a config forgot it: the runtime check ``open_training`` runs passes it (every write still gets a non-zero
    gradient); central finite differences (``receivers.conformance``) refuse it. Prepared
    (``weights.enable_moe_gradients``; scoped here by ``weights.moe_gradients``), both pass. The module's state is
    restored after."""

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from latent_interconnect.receivers.conformance import check_finite_differences
    from latent_interconnect.receivers.gradients import check_gradients
    from latent_interconnect.receivers.kimi_k26.weights import moe_gradients

    spec, _ = tiny
    path = Path(snapshot_download(spec.model_id, revision=spec.revision, local_files_only=True))
    moe = get_class_from_dynamic_module("modeling_deepseek.DeepseekV3MoE", str(path))
    assert hasattr(moe.moe_infer, "__wrapped__")  # as shipped: @torch.no_grad
    prompts = [p for p in _prompts(with_target=True) if p.slots][:2]
    # δ of connector a's first token at layer 0, object p0 (prompt 0's slot): dimensions 0, 3 and 7. Kimi's float32
    # islands (router, norms) make a smaller step noisy; the step's own error is O(h²). Measured gaps (autograd vs
    # central difference, normalized loss): prepared 2.0e-6 .. 4.4e-5; as shipped 7.4e-4 .. 1.9e-2.
    fd = {"coordinates": (0, 3, 7), "step": 1e-3, "atol": 1e-4, "rtol": 5e-4}

    hf_on_tiny.spec = hf_on_tiny.spec.model_copy(update={"training_preparation": None})
    check_gradients(hf_on_tiny, hf_on_tiny.spec)  # the runtime check's documented limit: it passes
    with pytest.raises(ValueError, match="differs from finite differences") as blind:
        check_finite_differences(hf_on_tiny, prompts, _leaf_writes, **fd)
    print(blind.value)
    with moe_gradients(path):
        measured = check_finite_differences(hf_on_tiny, prompts, _leaf_writes, **fd)
    for c, got, want in measured:
        print(f"[{c}]: autograd {got:.6g}, finite difference {want:.6g}")
    assert hasattr(moe.moe_infer, "__wrapped__")  # restored
