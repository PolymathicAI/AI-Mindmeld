"""The runtime gradient check every training backend passes when it is opened: finite, and reaching every write.

:func:`~latent_interconnect.receivers.open_training` runs :func:`check_gradients` once, on the opened backend, before
any training step. It runs the backend's own ``loss`` on :func:`probe_batch` (two short prompts of unequal lengths,
with a write at every receiver layer in each) and requires:

- a finite loss that depends on the writes;
- a finite gradient into every write vector and every mixing weight;
- a non-zero gradient into each of them, so a connector parameter at any layer can learn.

It is deliberately small. It does not show that a gradient is *correct*: a model that drops part
of the gradient (code run with gradients disabled on the residual stream, such as Kimi K2.6's remote ``moe_infer``,
which is ``@torch.no_grad``) still passes it, because the rest of the model carries a non-zero gradient to every
write. Correctness is established by the receiver's preparation and its tests, never by this check:

- the reference backend's gradient against central finite differences, and every other training backend's loss,
  gradients and one optimizer update against that reference (``receivers.conformance``, on CPU in float64);
- the GPU parity gate for the configurations CPU cannot run (``kimi-ep``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from latent_interconnect.receivers.base import ReceiverSpec, TrainingBackend
from latent_interconnect.receivers.kernel import LayerWrites
from latent_interconnect.receivers.render import RenderedPrompt, Target

__all__ = ["check_gradients", "probe_batch"]

PROBE_LENGTHS = (6, 3)
"""Prompt tokens of the probe's two prompts (unequal, so a padding or packing backend takes that path); each prompt
adds a two-token target. Token ids are ``0 .. 8``, valid in any real vocabulary."""
PROBE_ALPHA = 0.5
"""The probe's mixing weight at every site: inside (0, 1), where ``interp_v1`` depends on both δ and α."""


def probe_batch(
    spec: ReceiverSpec, placement: Mapping[int, torch.device]
) -> tuple[list[RenderedPrompt], dict[int, LayerWrites]]:
    """Return the probe's prompts and its writes: position 0 of each prompt, at every layer ``0 .. depth - 1``.

    The write vectors are fixed (``cos`` of their index, a different frequency per layer and phase per prompt), float32,
    on the device of their layer's input, and require grad, as do the mixing weights. Nothing is random.

    Args:
        spec: the receiver (``depth``, ``d_model``).
        placement: layer → the device of that layer's input (``TrainingBackend.placement``).
    """
    prompts = [
        RenderedPrompt(tuple(range(k)), (), Target((k + 1, k + 2), (1.0, 1.0)))  # each target follows its prompt
        for k in PROBE_LENGTHS
    ]
    rows = torch.arange(len(prompts), dtype=torch.int64)
    positions = torch.zeros(len(prompts), dtype=torch.int64)
    index = torch.arange(1, spec.d_model + 1, dtype=torch.float32)
    writes = {}
    for layer in range(spec.depth):
        dev = placement[layer]
        deltas = torch.stack([torch.cos(0.37 * (layer + 2) * index + r) for r in range(len(prompts))])
        alphas = torch.full((len(prompts),), PROBE_ALPHA)
        writes[layer] = LayerWrites(
            rows=rows,
            positions=positions,
            deltas=deltas.to(dev).requires_grad_(True),
            alphas=alphas.to(dev).requires_grad_(True),
        )
    return prompts, writes


def check_gradients(backend: TrainingBackend, spec: ReceiverSpec) -> None:
    """Run the probe through ``backend.loss`` and require a finite loss and a finite, non-zero gradient everywhere.

    Args:
        backend: the opened training backend.
        spec: its receiver config.

    Raises:
        ValueError: if the probe loss is not finite or does not depend on the writes, or a write's δ or α at some
            layer gets no gradient, a non-finite one or an exactly zero one (each named, with its layer).
    """
    prompts, writes = probe_batch(spec, backend.placement())
    terms = backend.loss(prompts, writes)
    loss = terms.weighted_sum / terms.weight_sum
    if not math.isfinite(float(loss.detach())):
        raise ValueError(f"the gradient probe's loss is {float(loss.detach())}, not finite")
    if not loss.requires_grad:
        raise ValueError("the gradient probe's loss does not depend on the writes: no gradient reaches them")
    torch.autograd.backward(loss)
    bad = [
        f"layer {layer} {name}: {problem}"
        for layer, w in writes.items()
        for name, leaf in (("delta", w.deltas), ("alpha", w.alphas))
        if (problem := _problem(leaf.grad))
    ]
    if bad:
        raise ValueError(f"the gradient probe's writes are not all reached by a usable gradient: {'; '.join(bad)}")


def _problem(grad: Tensor | None) -> str:
    """What is wrong with one leaf's gradient (one row per probe prompt), or ``""``."""
    if grad is None:
        return "no gradient"
    if not bool(torch.isfinite(grad).all()):
        return "a non-finite gradient"
    per_site = grad.reshape(grad.shape[0], -1).abs().amax(dim=1)
    zero = [int(i) for i in torch.nonzero(per_site == 0).flatten()]
    return f"a zero gradient at probe rows {zero}" if zero else ""
