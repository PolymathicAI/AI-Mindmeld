"""What a training backend must satisfy in its tests: its gradients are correct. Run by the tests, never at run time.

Gradient correctness is established here, by comparison, and not by the small runtime check at ``open_training``
(``receivers.gradients``):

- **The reference** (``hf`` on a model, float64, on CPU) is checked against central finite differences of its own
  loss: :func:`check_finite_differences`. Code that runs part of the forward with gradients disabled gives the right
  loss and a partial gradient, and fails it.
- **Every other training backend** of that model is checked against the reference on the same prompts and the same
  starting writes: the loss terms, the gradient into every write and the writes after one optimizer step, by
  :func:`training_step` on each and :func:`check_training_parity` on the two. The optimizer step makes a gradient's
  sign pattern visible: Adam's first step is about ``-lr * sign(g)``.

A configuration that runs only on GPU (packed variable-length attention, grouped expert GEMMs) is compared with the
backend's own reference configuration on the real model by a GPU parity job (``kimi-ep``).

Writes are given by a factory (``make_writes``) that returns fresh leaf writes, connector name →
:class:`~latent_interconnect.writes.connector.SlotWrites`, the same values on every call: each comparison starts both
backends from equal leaves. The leaves are taken in the factory's order: per connector, its write vectors, then its
mixing weights.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from latent_interconnect.receivers.base import TrainingBackend
from latent_interconnect.receivers.kernel import resolve
from latent_interconnect.receivers.render import RenderedPrompt
from latent_interconnect.writes.connector import SlotWrites

__all__ = ["TrainingStep", "check_finite_differences", "check_training_parity", "training_step"]

WritesFactory = Callable[[], Mapping[str, SlotWrites]]
"""Returns fresh leaf writes (``requires_grad``), connector name → its writes; equal values on every call."""


@dataclass(frozen=True)
class TrainingStep:
    """One backend's loss, gradients and one optimizer update on fixed prompts, from the factory's writes.

    Attributes:
        loss: ``Σ w·ℓ / Σ w``, detached.
        weight_sum: ``Σ w``, detached.
        gradients: the gradient of ``loss`` into every write leaf, flattened and concatenated in the factory's order.
        start: the leaves before the step, in the same layout.
        update: the leaves after one Adam step, in the same layout.
    """

    loss: Tensor
    weight_sum: Tensor
    gradients: Tensor
    start: Tensor
    update: Tensor


def _leaves(writes: Mapping[str, SlotWrites]) -> list[Tensor]:
    return [t for w in writes.values() for t in (*w.deltas.values(), *w.alphas.values())]


def _flat(tensors: Sequence[Tensor]) -> Tensor:
    return torch.cat([t.detach().flatten().cpu() for t in tensors])


def training_step(
    backend: TrainingBackend, prompts: Sequence[RenderedPrompt], make_writes: WritesFactory, *, lr: float
) -> TrainingStep:
    """Run one forward, backward and Adam step (learning rate ``lr``) of ``backend`` on the factory's writes.

    Raises:
        ValueError: if a write leaf gets no gradient (the comparison would be undefined).
    """
    writes = make_writes()
    leaves = _leaves(writes)
    start = _flat(leaves)
    optimizer = torch.optim.Adam(leaves, lr=lr)
    terms = backend.loss(prompts, resolve(prompts, writes))
    loss = terms.weighted_sum / terms.weight_sum
    torch.autograd.backward(loss)
    missing = [i for i, t in enumerate(leaves) if t.grad is None]
    if missing:
        raise ValueError(f"write leaves {missing} (in the factory's order) got no gradient")
    gradients = _flat([t.grad for t in leaves if t.grad is not None])
    optimizer.step()
    return TrainingStep(loss.detach().cpu(), terms.weight_sum.detach().cpu(), gradients, start, _flat(leaves))


def check_training_parity(got: TrainingStep, want: TrainingStep, *, atol: float) -> None:
    """Require ``got`` (a backend under test) to equal ``want`` (the reference) within ``atol``, quantity by quantity.

    The weight sums must be equal exactly (they are the targets' weights); the loss, every gradient element and every
    updated leaf within ``atol``. The reference must be informative: some gradient element and some update must be
    larger than ``atol``, or equality would say nothing.

    Raises:
        ValueError: naming each quantity that differs, with its largest difference, or an uninformative reference.
    """
    if float(want.gradients.abs().max()) <= atol or float((want.update - want.start).abs().max()) <= atol:
        raise ValueError(f"the reference's gradients or update are within {atol} of zero: the comparison is empty")
    bad = []
    if not torch.equal(got.weight_sum, want.weight_sum):
        bad.append(f"weight sum {float(got.weight_sum)} != {float(want.weight_sum)}")
    for name, a, b in (
        ("loss", got.loss, want.loss),
        ("gradients", got.gradients, want.gradients),
        ("update", got.update, want.update),
    ):
        if a.shape != b.shape:
            bad.append(f"{name}: shapes {tuple(a.shape)} != {tuple(b.shape)}")
        elif not (gap := float((a.double() - b.double()).abs().max())) <= atol:  # NaN fails too
            bad.append(f"{name}: max |difference| {gap:.3e} > {atol:.1e}")
    if bad:
        raise ValueError(f"the backend differs from the reference: {'; '.join(bad)}")


def check_finite_differences(
    backend: TrainingBackend,
    prompts: Sequence[RenderedPrompt],
    make_writes: WritesFactory,
    *,
    coordinates: Sequence[int],
    step: float,
    atol: float,
    rtol: float,
) -> list[tuple[int, float, float]]:
    """Compare the autograd gradient of ``Σ w·ℓ / Σ w`` with central finite differences at chosen leaf coordinates.

    A leaf that gets no gradient counts as a zero gradient, deliberately: autograd says the loss does not depend on
    it, so a loss that does fails at that leaf's coordinates (unlike :func:`training_step`, which refuses it).

    Args:
        backend: the backend under test (in float64, so the difference quotient is accurate).
        prompts: the prompts, with targets.
        make_writes: the writes factory.
        coordinates: indices into the flattened leaves (the factory's order).
        step: the finite-difference step ``h``; the quotient is ``(L(x + h) - L(x - h)) / 2h``.
        atol: allowed absolute gap.
        rtol: allowed gap relative to the finite difference.

    Returns:
        ``(coordinate, autograd, finite difference)`` for every coordinate, for the test's log.

    Raises:
        ValueError: if any coordinate's gap exceeds ``atol + rtol * |finite difference|``, listing each.
    """
    writes = make_writes()
    terms = backend.loss(prompts, resolve(prompts, writes))
    torch.autograd.backward(terms.weighted_sum / terms.weight_sum)
    gradients = _flat([t.grad if t.grad is not None else torch.zeros_like(t) for t in _leaves(writes)])

    def loss_at(coordinate: int, shift: float) -> float:
        moved = make_writes()
        offset = coordinate
        for leaf in _leaves(moved):
            if offset < leaf.numel():
                with torch.no_grad():
                    leaf.view(-1)[offset] += shift
                break
            offset -= leaf.numel()
        else:
            raise ValueError(f"coordinate {coordinate} is outside the {int(gradients.numel())} leaf elements")
        with torch.no_grad():
            t = backend.loss(prompts, resolve(prompts, moved))
            return float(t.weighted_sum / t.weight_sum)

    measured = []
    for c in coordinates:
        fd = (loss_at(c, step) - loss_at(c, -step)) / (2 * step)
        measured.append((c, float(gradients[c]), fd))
    bad = [(c, g, fd) for c, g, fd in measured if not abs(g - fd) <= atol + rtol * abs(fd)]
    if bad:
        shown = ", ".join(f"[{c}] autograd {g:.6g} vs {fd:.6g}" for c, g, fd in bad)
        raise ValueError(f"the gradient differs from finite differences at {len(bad)} coordinates: {shown}")
    return measured
