"""The ``kimi-stream`` backend: Kimi K2.6 scoring and state capture on one GPU, streaming one decoder layer at a time.

The whole batch of prompts advances together: each decoder layer is built from disk, applied to every micro-batch,
and freed before the next is built, so device memory holds about one layer plus the activations, while the next
layer's shard files are read into the page cache in the background. The residual stream of every micro-batch waits on
the host between layers.

Semantics (the same as the generic ``hf`` backend; ``tests/unit/test_kimi_ep.py`` checks that ``kimi-ep``'s
in-job scores equal this backend's):

- prompts are right-padded; position ids run 0 .. T-1; the attention mask is the causal mask with padded keys masked;
- writes are resolved by :func:`~latent_interconnect.receivers.kernel.resolve_sources` and applied with
  :func:`~latent_interconnect.receivers.kernel.apply_writes` at the input of decoder layer L (L = 0 is the
  token-embedding output);
- ``capture`` and ``capture_span`` read ``hidden_states[L]``: the input of decoder layer L for L < depth, and the
  final-norm output for L = depth, as the model's own ``output_hidden_states`` defines them;
  ``capture_span`` reduces each span to its last token and its mean on the device, in a fixed summation order;
- every logit comes from the one head, :func:`~latent_interconnect.receivers.kernel.readout_logits` (float32 on the
  bf16 body; float64 stays float64), and readouts and returned states are in
  :func:`~latent_interconnect.receivers.kernel.state_dtype`.

Every call is partitioned into micro-batches by the one planner,
:func:`~latent_interconnect.receivers.kernel.plan_batches`, under the configured
:class:`~latent_interconnect.receivers.kernel.BatchPlan` (``plan``; the default is the deterministic plan). The
plan changes numbers (``examples/guides/batching.md``), so it is configuration; :meth:`KimiStream.execution` records
only static facts (precision, device, weights).
"""

from __future__ import annotations

import gc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import torch
from torch import Tensor, nn

from latent_interconnect.receivers.base import (
    RECEIVER_BACKENDS,
    CaptureBackend,
    Readout,
    ReceiverSpec,
    ScoringBackend,
    SpanStates,
    capture_layers,
    capture_positions,
    check_score_request,
)
from latent_interconnect.receivers.kernel import (
    BatchPlan,
    LayerWrites,
    apply_writes,
    plan_batches,
    planned_lengths,
    reduce_spans,
    select_writes,
    state_dtype,
)
from latent_interconnect.receivers.kimi_k26.common import attention_mask, padded_ids, padding_mask, read_ahead, read_out
from latent_interconnect.receivers.kimi_k26.weights import (
    EMBED_PREFIX,
    HEAD_PREFIX,
    NORM_PREFIX,
    open_snapshot,
)
from latent_interconnect.receivers.render import RenderedPrompt
from latent_interconnect.writes.sources import WriteSource

__all__ = ["KimiStream"]


@dataclass(frozen=True)
class _MicroBatch:
    """One micro-batch: which sequences, their right-padded mask, and the writes landing in it, by layer."""

    rows: list[int]
    mask: Tensor
    writes: dict[int, LayerWrites]


@RECEIVER_BACKENDS.register("kimi-stream")
class KimiStream:
    """Kimi K2.6 ``score`` and ``capture`` with the model streamed one decoder layer at a time from local disk.

    Implements :class:`~latent_interconnect.receivers.base.ScoringBackend` and
    :class:`~latent_interconnect.receivers.base.CaptureBackend`.

    Args:
        spec: the receiver config (``configs/receivers/kimi-k2.6.yaml``).
        weights: the directory to read the weights from, e.g. a node-local copy the job passes (e.g. a site's
            copy for the job's device class). It must hold the same model files, by name and size, as the
            pinned snapshot ``snapshots/<spec.revision>`` of ``spec.model_id`` in the Hugging Face cache
            (:func:`~latent_interconnect.receivers.kimi_k26.weights.check_weights`). When None, the pinned snapshot
            itself is read. Nothing is ever downloaded.
        device: the one device the layers run on.
        dtype: compute precision of the dense weights and the residual stream (bf16 is Kimi's native precision;
            the CPU parity test uses float64).
        plan: the batch plan (None: ``BatchPlan()``, the deterministic default).

    Raises:
        FileNotFoundError: if ``weights`` (or the pinned snapshot) has no safetensors index.
        SnapshotMismatchError: if ``weights`` differs from the pinned snapshot.
        ValueError: if the snapshot's shape differs from the receiver config's.
    """

    reads_banks: ClassVar[bool] = False  # takes live connectors and banks alike (in-process)
    pads_every_batch: ClassVar[bool] = False  # a capability, not an exemption: it forms no right-padded batch

    def __init__(
        self,
        spec: ReceiverSpec,
        *,
        weights: Path | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        plan: BatchPlan | None = None,
    ) -> None:
        self.spec = spec
        self.snapshot = open_snapshot(spec, weights)
        self.weights = self.snapshot.path
        self.device = torch.device(device)
        self.dtype = dtype
        self.plan = BatchPlan() if plan is None else plan

    def execution(self) -> dict[str, str]:
        """Return the static execution facts of :mod:`latent_interconnect.receivers.base`: no history."""
        name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else self.device.type
        quantization = self.snapshot.quantization
        return {
            "backend": "kimi-stream",
            "dtype": str(self.dtype).removeprefix("torch."),
            "quantization": "none" if quantization is None else f"compressed-tensors {quantization.format}",
            "device_name": name,
            "device_count": "1",
            "attention": "eager",
            "revision": self.spec.revision,
            "weights": str(self.weights),
        }

    def score(self, prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource], readout: Readout) -> Tensor:
        """Teacher-forced readout with writes (see :class:`~latent_interconnect.receivers.base.ScoringBackend`).

        Raises:
            ValueError: by the one score-request rule (``receivers.base.check_score_request``), checked before any
                model work: a write at a layer the receiver does not have, then a ``target_nll`` prompt without target
                tokens. A ``Readout`` is valid by construction.
        """
        with torch.inference_mode():
            call = check_score_request(prompts, sources, readout, self.spec.depth, self.plan)
            if not prompts:  # nothing to stream the model for
                width = readout.width
                return torch.empty((0,) if width is None else (0, width), dtype=state_dtype(self.dtype))
            if call.targets is None:
                sequences = [p.input_ids for p in prompts]
            else:
                sequences = [p.input_ids + t.token_ids for p, t in zip(prompts, call.targets, strict=True)]
            batches = self._micro_batches(call.lengths, call.writes)
            hidden, _ = self._stream(sequences, batches, layers=self.spec.depth)
            # the head is a body weight (self.dtype), promoted once here as readout_logits would on every call
            head = self.snapshot.matrix(HEAD_PREFIX, self.device, self.dtype).to(state_dtype(self.dtype))
            out = read_out(prompts, sequences, [b.rows for b in batches], hidden, readout=readout, head=head)
            del head
        if self.device.type == "cuda":
            torch.cuda.empty_cache()  # the head (vocabulary × d_model) was freed with the readout
        return out

    def capture(self, prompts: Sequence[RenderedPrompt], layers: Sequence[int]) -> dict[int, Tensor]:
        """Return layer → ``(N, d_model)`` states ``hidden_states[L]`` at each prompt's ``capture_position``.

        Only the layers below the deepest requested one are streamed.

        Raises:
            ValueError: by the one capture rule (``receivers.base.capture_layers`` and ``capture_positions``): no
                layer, a layer outside 0 .. depth, a prompt with slots, or a prompt without a capture position
                inside its ids.
        """
        return self._capture(prompts, layers, "position", reduce=False)

    def capture_span(self, prompts: Sequence[RenderedPrompt], layers: Sequence[int]) -> dict[int, SpanStates]:
        """Return layer → the last-token and mean states of each prompt's ``capture_span``, reduced on the device.

        Only the layers below the deepest requested one are streamed.

        Raises:
            ValueError: by the one capture rule (``receivers.base.capture_layers`` and ``capture_positions``): no
                layer, a layer outside 0 .. depth, a prompt with slots, or a prompt without a non-empty span inside
                its ids.
        """
        flat = self._capture(prompts, layers, "span", reduce=True)
        # separate contiguous copies: the two may be saved together, and safetensors refuses shared storage
        return {L: SpanStates(last=both[:, 0].contiguous(), mean=both[:, 1].contiguous()) for L, both in flat.items()}

    def _capture(
        self,
        prompts: Sequence[RenderedPrompt],
        layers: Sequence[int],
        what: Literal["position", "span"],
        *,
        reduce: bool,
    ) -> dict[int, Tensor]:
        """Check a capture call by the one capture rule, then stream it.

        Returns ``(N, d)`` states at single positions, or ``(N, 2, d)`` (last, mean) of spans.
        """
        capture_layers(layers, self.spec.depth)
        spans = capture_positions(prompts, what)
        if not prompts:  # nothing to stream the model for
            shape = (0, 2, self.spec.d_model) if reduce else (0, self.spec.d_model)
            return {layer: torch.empty(shape, dtype=state_dtype(self.dtype)) for layer in layers}
        with torch.inference_mode():
            sequences = [p.input_ids for p in prompts]
            batches = self._micro_batches(planned_lengths(prompts, "capture"), {})
            _, flat = self._stream(
                sequences, batches, layers=max(layers), capture=layers, positions=spans, reduce=reduce
            )
        return flat

    def _micro_batches(self, lengths: Sequence[int], writes: Mapping[int, LayerWrites]) -> list[_MicroBatch]:
        """Split a call into micro-batches by its planned lengths, re-indexing each layer's write sites.

        ``lengths`` come from ``kernel.planned_lengths``, the one length rule; write rows become micro-batch rows.
        """
        out = []
        for batch in plan_batches(lengths, self.plan):
            rows = list(batch)
            out.append(_MicroBatch(rows, padding_mask([lengths[i] for i in rows]), select_writes(writes, rows)))
        return out

    def _stream(
        self,
        sequences: Sequence[Sequence[int]],
        batches: Sequence[_MicroBatch],
        *,
        layers: int,
        capture: Sequence[int] = (),
        positions: Sequence[Sequence[int]] = (),
        reduce: bool = False,
    ) -> tuple[list[Tensor], dict[int, Tensor]]:
        """Run the first ``layers`` decoder layers (and the final norm when ``layers`` is the depth).

        Args:
            sequences: token ids per sequence.
            batches: the micro-batches with their writes.
            layers: how many decoder layers to run.
            capture: hidden-state indices to read.
            positions: per sequence, the positions to read at each of them.
            reduce: reduce each sequence's positions on the device to (last, mean) instead of returning every state.

        Returns:
            Each micro-batch's last state (post-norm when every layer ran) on the host, and per captured index the
            states of every sequence's positions concatenated in sequence order, or, with ``reduce``, an
            ``(N, 2, d_model)`` tensor of each sequence's (last, mean).
        """
        snapshot, dev, dtype = self.snapshot, self.device, self.dtype
        out_dtype = state_dtype(dtype)
        offsets = [0]
        for at in positions:
            offsets.append(offsets[-1] + len(at))
        shape = (len(sequences), 2, self.spec.d_model) if reduce else (offsets[-1], self.spec.d_model)
        captured = {layer: torch.empty(shape, dtype=out_dtype) for layer in sorted(capture)}
        pad = snapshot.config.pad_token_id
        embedding = snapshot.matrix(EMBED_PREFIX, dev, dtype)
        hidden = []
        for b in batches:
            ids = padded_ids([sequences[i] for i in b.rows], int(pad))
            hidden.append(nn.functional.embedding(ids.to(dev), embedding).to("cpu"))
        del embedding
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        def edit(index: int, k: int, h: Tensor) -> Tensor:
            site = batches[k].writes.get(index)
            if site is not None:
                h = apply_writes(h, site)
            if index in captured and reduce:
                rows = batches[k].rows
                reduced = reduce_spans(h, [positions[i] for i in rows])
                captured[index][rows, 0] = reduced.last.to("cpu")
                captured[index][rows, 1] = reduced.mean.to("cpu")
            elif index in captured:
                rows = batches[k].rows
                row = torch.tensor([r for r, i in enumerate(rows) for _ in positions[i]], device=h.device)
                at = torch.tensor([a for i in rows for a in positions[i]], device=h.device)
                dest = [offsets[i] + j for i in rows for j in range(len(positions[i]))]
                captured[index][dest] = h[row, at].to("cpu", out_dtype)
            return h

        prefixes = [snapshot.layer_prefix(i) for i in range(layers)]
        if layers == self.spec.depth:  # the final norm's shard is read during the last layer
            prefixes.append(NORM_PREFIX)
        for index in read_ahead(snapshot, prefixes, layers):
            layer = snapshot.build_layer(index, dev, dtype)
            for k in range(len(batches)):
                h = edit(index, k, hidden[k].to(dev))
                # built per micro-batch and layer, so device memory never holds every micro-batch's (t × t) mask
                mask = attention_mask(batches[k].mask, self.dtype, self.device)
                position = torch.arange(batches[k].mask.shape[1], device=dev).unsqueeze(0)
                hidden[k] = layer(h, attention_mask=mask, position_ids=position, use_cache=False)[0].to("cpu")
                del mask
            del layer
            # The remote layers hold reference cycles; without a collection each freed layer leaks (runbook).
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        if layers < self.spec.depth:
            for k in range(len(batches)):
                edit(layers, k, hidden[k])
            return hidden, captured
        norm = snapshot.build_norm(dev, dtype)
        for k in range(len(batches)):
            hidden[k] = norm(hidden[k].to(dev)).to("cpu")
            if self.spec.depth in captured:
                edit(self.spec.depth, k, hidden[k])
        return hidden, captured


# Static protocol conformance, checked by mypy: the registry hands backends out through a cast, which would hide a
# signature that drifted from the protocol.
_SCORING: type[ScoringBackend] = KimiStream
_CAPTURE: type[CaptureBackend] = KimiStream
