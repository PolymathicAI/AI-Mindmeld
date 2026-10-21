"""From connector writes and slot positions to per-layer writes, and the one kernel every backend applies.

:func:`resolve` turns each bound connector's :class:`~latent_interconnect.writes.connector.SlotWrites` and the slot
occurrences of rendered prompts into :class:`LayerWrites`: for each receiver layer, the (row, position) sites with
their δ and α. :func:`resolve_sources` does the same starting from write sources (live connector or bank). Every
backend applies the result with :func:`apply_writes` at the input of decoder layer L, so the write semantics are
identical across the generic, streaming, expert-parallel and vLLM paths.

The read side has one implementation too: :func:`reduce_spans` turns a span capture into :class:`SpanStates` on the
device, :func:`readout_logits` is the one logit head (float32 or wider on the native-precision body), and
:func:`state_dtype` is the dtype of every readout and captured state, empty results included.

Batching has one implementation as well. A :class:`BatchPlan` is the configured way a backend forms micro-batches from
the prompts of one call; :func:`plan_batches` is the one planner every backend uses, and :func:`partition_digest`
names the partition it produced. The plan is a configuration value (it changes numbers,
``examples/guides/batching.md``), so the configs that score or capture carry it in their identity; what a run observed
while executing stays in its provenance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, assert_never

import torch
from pydantic import Field
from torch import Tensor

from latent_interconnect.canonical import canonical_sha256
from latent_interconnect.config import Config, added_field
from latent_interconnect.receivers.render import RenderedPrompt, Target
from latent_interconnect.writes.connector import SlotWrites
from latent_interconnect.writes.rule import interp
from latent_interconnect.writes.sources import WriteSource

__all__ = [
    "BatchPlan",
    "LayerWrites",
    "Partition",
    "PlannedPass",
    "ReadoutKind",
    "SpanStates",
    "apply_writes",
    "partition_digest",
    "plan_batches",
    "planned_lengths",
    "planned_lengths_and_targets",
    "readout_logits",
    "reduce_spans",
    "require_targets",
    "resolve",
    "resolve_sources",
    "select_writes",
    "state_dtype",
]

Partition = tuple[tuple[int, ...], ...]
"""Micro-batches as tuples of prompt indices (positions in the call's prompt list), in execution order."""


class BatchPlan(Config):
    """How a backend forms micro-batches from the prompts of one call.

    The planner ``length-stable/v1`` takes the prompts in their given order, sorts them stably by token count (equal
    counts keep their order), and closes a micro-batch when adding the next prompt would exceed ``budget_tokens``
    padded tokens (micro-batch size × longest prompt); a prompt longer than the budget gets a micro-batch of its own.
    It is the original streaming engine's rule. The partition is therefore a function of the prompts' token counts, in
    order, and of this plan alone.

    Attributes:
        planner: the planning rule, by name and version.
        budget_tokens: the padded-token budget of a micro-batch.
        pad_every_batch: whether every forward pass of a backend built with this plan is right-padded by one masked
            column beyond its longest prompt, so that no batch is free of padding (Gemma 3 scoring). With
            transformers 4.57.6 a batch without padding gets no attention mask and one with any padding an explicit
            one (``examples/guides/batching.md``, the padding class): the same mathematics, rounded differently in bf16
            on some cards. The partition and the budget's count are unchanged; the column lies causally after every
            real token, so positions, write sites and read positions are the prompts' own. Only a backend that pads
            serves it (``hf``; ``receivers._open`` refuses the others). The default off is the historical meaning of
            every stored plan; a fresh scoring run on a backend that pads must state the field
            (``run.suite.require_stated_padding``).
    """

    planner: Literal["length-stable/v1"] = "length-stable/v1"
    budget_tokens: int = Field(default=32_768, ge=1)
    pad_every_batch: bool = added_field(default=False)


def plan_batches(lengths: Sequence[int], plan: BatchPlan) -> Partition:
    """Return the micro-batches of one call: the one planner every backend uses.

    Args:
        lengths: each prompt's token count, in the call's prompt order.
        plan: the configured plan.
    """
    batches: list[tuple[int, ...]] = []
    current: list[int] = []
    for i in sorted(range(len(lengths)), key=lambda k: lengths[k]):
        if current and max(lengths[j] for j in (*current, i)) * (len(current) + 1) > plan.budget_tokens:
            batches.append(tuple(current))
            current = []
        current.append(i)
    if current:
        batches.append(tuple(current))
    return tuple(batches)


ReadoutKind = Literal["next_token", "target_nll"]
"""The readouts a scoring pass returns (``receivers.base.Readout.kind``): the one list of their names."""
PlannedPass = ReadoutKind | Literal["capture"]
"""What a forward pass feeds the model, which decides the token count the planner sees (:func:`planned_lengths`)."""


def require_targets(prompts: Sequence[RenderedPrompt], what: str) -> list[Target]:
    """Return each prompt's target, checked: the one non-empty-target rule (``target_nll`` readouts and the loss).

    Args:
        prompts: the prompts.
        what: the pass, for the message (``"target_nll"``, ``"loss"``).

    Raises:
        ValueError: naming the prompts without target tokens (no target, or an empty one).
    """
    missing = [i for i, p in enumerate(prompts) if p.target is None or not p.target.token_ids]
    if missing:
        raise ValueError(f"{what}: prompts {missing[:5]} have no target tokens")
    return [p.target for p in prompts if p.target is not None]


def planned_lengths(prompts: Sequence[RenderedPrompt], kind: PlannedPass) -> list[int]:
    """Return each prompt's token count as the planner sees it: the one rule every backend and runner uses.

    A ``next_token`` readout and a capture feed the prompt only; a ``target_nll`` readout feeds the prompt followed
    by its target.

    Args:
        prompts: the call's prompts, in call order.
        kind: the pass (``Readout.kind``, or ``"capture"``).

    Raises:
        ValueError: for ``target_nll`` on a prompt without target tokens (:func:`require_targets`).
    """
    return planned_lengths_and_targets(prompts, kind)[0]


def planned_lengths_and_targets(
    prompts: Sequence[RenderedPrompt], kind: PlannedPass
) -> tuple[list[int], list[Target] | None]:
    """Return :func:`planned_lengths` and, for a ``target_nll`` pass, the checked targets (None otherwise).

    The targets are checked once (:func:`require_targets`), so a backend that needs them does not check again.

    Raises:
        ValueError: for ``target_nll`` on a prompt without target tokens.
    """
    match kind:
        case "next_token" | "capture":
            return [len(p.input_ids) for p in prompts], None
        case "target_nll":
            targets = require_targets(prompts, "target_nll")
            return [len(p.input_ids) + len(t.token_ids) for p, t in zip(prompts, targets, strict=True)], targets
        case _:  # a new pass kind must state what it feeds (mypy checks that this is unreachable)
            assert_never(kind)


def partition_digest(partition: Partition) -> str:
    """Return the sha256 of a partition, so that "the same micro-batches" is checkable from a result."""
    return canonical_sha256([list(b) for b in partition])


@dataclass(frozen=True)
class LayerWrites:
    """The writes at one receiver layer.

    Attributes:
        rows: ``(M,)`` int64, prompt index of each site.
        positions: ``(M,)`` int64, token position of each site within its prompt (prompt-relative, i.e. as in
            right-padded batches; a left-padding or packed backend adds its own offsets).
        deltas: ``(M, d_model)`` write vectors (may require grad).
        alphas: ``(M,)`` mixing weights (may require grad).
    """

    rows: Tensor
    positions: Tensor
    deltas: Tensor
    alphas: Tensor


def resolve(prompts: Sequence[RenderedPrompt], writes: Mapping[str, SlotWrites]) -> dict[int, LayerWrites]:
    """Gather per-layer writes for a batch of prompts.

    Args:
        prompts: rendered prompts; each slot occurrence names its connector and object.
        writes: connector name → that connector's writes for (at least) the objects bound to it in ``prompts``.

    Returns:
        layer → the sites written at that layer, over all prompts and slots.

    Raises:
        KeyError: if a slot names a connector or object missing from ``writes``.
    """
    row_of = {name: {oid: i for i, oid in enumerate(w.object_ids)} for name, w in writes.items()}
    sites: dict[int, list[tuple[int, int, Tensor, Tensor]]] = {}
    for prompt_index, prompt in enumerate(prompts):
        for occ in prompt.slots:
            w = writes[occ.binding.connector]
            i = row_of[occ.binding.connector][occ.binding.object_id]
            for entry, delta in w.deltas.items():
                site = occ.positions[entry.token]
                sites.setdefault(entry.layer, []).append((prompt_index, site, delta[i], w.alphas[entry]))
    out = {}
    for layer, items in sorted(sites.items()):
        dev = items[0][2].device
        out[layer] = LayerWrites(
            rows=torch.tensor([it[0] for it in items], dtype=torch.int64),
            positions=torch.tensor([it[1] for it in items], dtype=torch.int64),
            deltas=torch.stack([it[2] for it in items]),
            alphas=torch.stack([it[3].reshape(()).to(dev) for it in items]),
        )
    return out


def resolve_sources(prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource]) -> dict[int, LayerWrites]:
    """Fetch each connector's writes for the objects bound to it in ``prompts``, then :func:`resolve`."""
    wanted: dict[str, dict[str, None]] = {}
    for prompt in prompts:
        for occ in prompt.slots:
            wanted.setdefault(occ.binding.connector, {})[occ.binding.object_id] = None
    return resolve(prompts, {name: sources[name].writes_for(list(ids)) for name, ids in wanted.items()})


def select_writes(writes: Mapping[int, LayerWrites], rows: Sequence[int]) -> dict[int, LayerWrites]:
    """Return the writes of the prompts ``rows`` (indices into the call), re-indexed to their places in ``rows``.

    A micro-batch's writes: every backend that splits a call takes them from here. Layers with no site in ``rows``
    are left out.
    """
    place = {r: i for i, r in enumerate(rows)}
    out = {}
    for layer, w in writes.items():
        keep = [k for k, r in enumerate(w.rows.tolist()) if r in place]
        if keep:
            idx = torch.tensor(keep, dtype=torch.long)
            out[layer] = LayerWrites(
                rows=torch.tensor([place[r] for r in w.rows[idx].tolist()], dtype=torch.long),
                positions=w.positions[idx],
                deltas=w.deltas[idx.to(w.deltas.device)],
                alphas=w.alphas[idx.to(w.alphas.device)],
            )
    return out


def apply_writes(hidden: Tensor, writes: LayerWrites) -> Tensor:
    """Return ``hidden`` with ``interp_v1`` applied at the write sites (out of place; differentiable).

    Args:
        hidden: ``(N, S, d_model)`` right-padded states at the input of the layer the writes belong to.
        writes: that layer's sites; ``rows`` index N and ``positions`` index S.
    """
    out = hidden.clone()
    h = hidden[writes.rows, writes.positions]
    d = writes.deltas.to(device=h.device)
    a = writes.alphas.to(device=h.device).unsqueeze(-1)
    out[writes.rows, writes.positions] = interp(h, d, a)
    return out


@dataclass(frozen=True)
class SpanStates:
    """A span capture at one layer, reduced on the device.

    No per-token state leaves the device: each span is reduced to its last and mean states there.

    Attributes:
        last: ``(N, d_model)``, the state at each prompt's last span token.
        mean: ``(N, d_model)``, the mean state over each prompt's span (e.g. the ridge bias over a name's tokens).

    Both are :func:`state_dtype` of the compute dtype, separate and contiguous (safetensors refuses views that share
    storage).
    """

    last: Tensor
    mean: Tensor


def readout_logits(hidden: Tensor, head_weight: Tensor) -> Tensor:
    """Return the logits of the final (normed) hidden states: the one logit head every readout uses.

    The receiver's body runs in its native precision; the head is computed in :func:`state_dtype` of it (float32 for
    a bf16 body, float64 in the CPU parity tests), so log-odds are not quantized to the body's precision.

    Args:
        hidden: ``(..., d_model)`` final hidden states, after the model's final norm.
        head_weight: ``(vocab, d_model)`` the output projection (untied or tied embedding).
    """
    dtype = state_dtype(hidden.dtype)
    return hidden.to(dtype) @ head_weight.to(device=hidden.device, dtype=dtype).T


def state_dtype(dtype: torch.dtype) -> torch.dtype:
    """The dtype of readouts and captured states: the compute dtype promoted to at least float32 (float64 stays)."""
    return torch.promote_types(dtype, torch.float32)


def reduce_spans(hidden: Tensor, spans: Sequence[Sequence[int]]) -> SpanStates:
    """Reduce each row's span to its last-token state and its mean, on ``hidden``'s device.

    The span states are gathered into a ``(B, S, d)`` block whose filler entries (spans shorter than S) are replaced by
    zero by selection, so a non-finite value outside a span never reaches the mean, then summed along S: a fixed order
    on every device (an atomic scatter-add would let the mean differ between runs on a GPU). Run-to-run identity on a
    GPU rests on that order.

    Args:
        hidden: ``(B, T, d)`` states of one layer.
        spans: exactly one span per row of ``hidden``: its positions, in order (the last one is the span's last
            token).

    Returns:
        The reduction, in :func:`state_dtype` of ``hidden``, on ``hidden``'s device.

    Raises:
        ValueError: if ``spans`` is empty or does not have one span per row, a span is empty, or a position is outside
            ``0 .. T - 1``.
    """
    if not spans or len(spans) != hidden.shape[0]:
        raise ValueError(f"{len(spans)} spans for {hidden.shape[0]} rows: give exactly one span per row (at least one)")
    t = hidden.shape[1]
    bad = [r for r, span in enumerate(spans) if not span or not all(0 <= at < t for at in span)]
    if bad:
        raise ValueError(f"rows {bad[:5]}: a span must be non-empty and inside 0 .. {t - 1}")
    dtype, dev = state_dtype(hidden.dtype), hidden.device
    n = torch.tensor([len(span) for span in spans], device=dev)
    at = torch.zeros(len(spans), int(n.max()), dtype=torch.int64)
    for r, span in enumerate(spans):
        at[r, : len(span)] = torch.tensor(span)
    rows = torch.arange(len(spans), device=dev)
    valid = torch.arange(at.shape[1], device=dev).unsqueeze(0) < n.unsqueeze(1)
    block = torch.where(valid.unsqueeze(-1), hidden[rows.unsqueeze(1), at.to(dev)].to(dtype), 0.0)
    last = hidden[rows, torch.tensor([span[-1] for span in spans], device=dev)].to(dtype)
    return SpanStates(last=last.contiguous(), mean=(block.sum(1) / n.to(dtype).unsqueeze(1)).contiguous())
