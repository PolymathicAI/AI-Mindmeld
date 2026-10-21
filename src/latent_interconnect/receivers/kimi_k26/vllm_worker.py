"""The worker side of ``kimi-vllm``: what runs inside vLLM's worker processes, one per tensor-parallel rank.

vLLM 0.28's V2 model runner serves Kimi K2.6 with vLLM's own model code (``KimiK25ForConditionalGeneration`` around
``DeepseekV2Model``). :class:`WriteExtension` is mixed into every worker (vLLM's ``worker_extension_cls``) and
:meth:`WriteExtension.lip_install` installs, once:

- a forward pre-hook on every decoder layer that applies the engine step's writes at the input of decoder layer L
  (:func:`write_fused`);
- the one logit head in place of vLLM's ``compute_logits`` (:func:`head_logits`), so every logit, sampled or read,
  is ``kernel.readout_logits``: float32 on the bf16 body, with no exception for this backend;
- wrappers of three runner methods that map each request's writes onto the rows of each engine step
  (:func:`step_writes`).

**The fused residual.** vLLM's decoder loop carries the residual stream as a pair ``(hidden_states, residual)``. At
the input of layer 0 the stream is ``hidden_states`` (the token embedding) and ``residual`` is None. At the input of
layer L > 0 the stream is ``hidden_states + residual``: layer L's input norm adds them (fused add and RMSNorm) and
keeps the sum as its residual. A write at L > 0 therefore sets, at each written row, ``residual`` to the written
stream (``kernel.apply_writes`` on the sum, as the fused add stores it, in the residual's dtype) and ``hidden_states``
to zero, so the fused add yields exactly the written stream. At layer 0 the write replaces ``hidden_states``. Both
use the one kernel; nothing here restates the write rule.

**Rows and prefill only.** vLLM flattens every token a step schedules into one row axis. Request i's tokens are rows
``query_start_loc[i] ..`` at prompt positions ``num_computed[i] .. num_computed[i] + num_scheduled[i] - 1``. A write
at prompt position p lands at row ``query_start_loc[i] + p - num_computed[i]`` of the one step that computes p. So a
chunked prefill splits a request's writes over its steps; a decode step (positions past the prompt) carries none, so
writes apply during prefill only; and a request recomputed from scratch gets its writes again. Every hook checks that
the step's ``positions`` at its rows are the slots' positions.

**Writes by reference.** A request carries its slot occurrences (connector, object, positions) in
``SamplingParams.extra_args`` under :data:`REQUEST_KEY` (:func:`encode_slots`). Each worker resolves them with
``kernel.resolve_sources`` against its own ``BankSource`` per connector, opened from the bank's store and digest
(:meth:`WriteExtension.lip_sources`). No write tensor crosses a process boundary.

**Refused at install** (each would break one of the above): a model runner other than V2; sequence-parallel MoE (rows
would not be global token rows); pipeline parallelism; prefix caching (a cached prefix skips its slot positions, and
two requests sharing a prefix with different writes would share keys and values); CUDA graphs (the hooks are Python:
``enforce_eager``); speculative decoding; async scheduling (an engine step must follow its scheduling); a body dtype
other than bf16; float32 matmuls below full precision (the head would run in TF32); a head with a bias, a scale or a
soft cap (the one head would not reproduce it); a decoder-layer count other than the receiver's.

The pieces that need no vLLM are :func:`encode_slots`, :func:`request_writes`, :func:`step_writes`,
:func:`write_fused` and :func:`head_logits`; the engine path is checked on the real model by the GPU parity job
against ``kimi-stream``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.receivers.kernel import LayerWrites, apply_writes, readout_logits, resolve_sources
from latent_interconnect.receivers.render import RenderedPrompt, SlotBinding, SlotOccurrence
from latent_interconnect.writes.sources import BANK_KIND, BankSource, WriteSource

__all__ = [
    "REQUEST_KEY",
    "StepRequest",
    "StepWrites",
    "WriteExtension",
    "encode_slots",
    "head_logits",
    "request_writes",
    "step_writes",
    "write_fused",
]

REQUEST_KEY = "lip_slots"
"""The key of a request's slot occurrences in ``SamplingParams.extra_args``."""
V2_RUNNER = "vllm.v1.worker.gpu.model_runner"
"""The module of vLLM's V2 model runner, which the runner wrappers are written for (``VLLM_USE_V2_MODEL_RUNNER=1``)."""

SlotData = list[list[Any]]
"""A request's slot occurrences as plain data: ``[slot, connector, object_id, k, [positions]]`` each."""


def encode_slots(prompt: RenderedPrompt) -> SlotData:
    """Return the prompt's slot occurrences as plain data, for ``SamplingParams.extra_args`` (:data:`REQUEST_KEY`)."""
    return [[o.slot, o.binding.connector, o.binding.object_id, o.binding.k, list(o.positions)] for o in prompt.slots]


def request_writes(slots: SlotData, sources: Mapping[str, WriteSource], prompt_len: int) -> dict[int, LayerWrites]:
    """Return one request's writes by layer: rows all 0, positions prompt-relative (``kernel.resolve_sources``).

    Every slot position must lie inside the request's prompt, so a write can only land in a prefill step: the prefill
    computes positions ``0 .. prompt_len - 1``, decoding the positions after them.

    Args:
        slots: the request's slot occurrences (:func:`encode_slots`).
        sources: connector → its write source.
        prompt_len: the request's prompt length in tokens.

    Raises:
        ValueError: if a slot position lies outside the prompt.
        KeyError: if a slot names a connector or object the sources lack.
    """
    occurrences = tuple(
        SlotOccurrence(str(slot), SlotBinding(str(obj), str(conn), int(k)), tuple(int(p) for p in at))
        for slot, conn, obj, k, at in slots
    )
    outside = sorted({p for o in occurrences for p in o.positions if not 0 <= p < prompt_len})
    if outside:
        raise ValueError(
            f"slot positions {outside[:5]} lie outside the {prompt_len}-token prompt (writes are prefill only)"
        )
    return resolve_sources([RenderedPrompt((), occurrences)], sources)


@dataclass(frozen=True)
class StepRequest:
    """One request's share of an engine step (vLLM's ``InputBatch``, batch order).

    Attributes:
        req_id: the request.
        first_row: its first row in the step's flattened tokens (``query_start_loc``).
        computed: its tokens computed before this step (``num_computed_tokens``): the step starts at that position.
        scheduled: its tokens in this step (``num_scheduled_tokens``).
    """

    req_id: str
    first_row: int
    computed: int
    scheduled: int


@dataclass(frozen=True)
class StepWrites:
    """One layer's writes in one engine step.

    Attributes:
        writes: the sites as ``kernel.LayerWrites`` over the step's flattened tokens: ``rows`` all 0 (one sequence),
            ``positions`` the step's token rows.
        positions: the prompt position of each site, which the step's ``positions`` must show at its row.
    """

    writes: LayerWrites
    positions: Tensor


def step_writes(
    requests: Sequence[StepRequest], writes_of: Mapping[str, Mapping[int, LayerWrites]]
) -> dict[int, StepWrites]:
    """Return the writes an engine step applies, by layer: each request's sites at the positions this step computes.

    Args:
        requests: the step's requests, in batch order.
        writes_of: request id → its writes (:func:`request_writes`); every request of the step has an entry.

    Raises:
        KeyError: if a request of the step has no writes entry.
        ValueError: if two sites of one layer land on one row.
    """
    parts: dict[int, list[tuple[Tensor, Tensor, Tensor, Tensor]]] = {}
    for r in requests:
        for layer, w in writes_of[r.req_id].items():
            here = (w.positions >= r.computed) & (w.positions < r.computed + r.scheduled)
            if bool(here.any()):
                at = w.positions[here]
                idx = here.nonzero().squeeze(1)
                parts.setdefault(layer, []).append((at - r.computed + r.first_row, at, w.deltas[idx], w.alphas[idx]))
    out = {}
    for layer, items in sorted(parts.items()):
        rows = torch.cat([it[0] for it in items])
        if len(set(rows.tolist())) != len(rows):
            raise ValueError(f"layer {layer}: two writes land on one row of the step")
        writes = LayerWrites(
            rows=torch.zeros_like(rows),
            positions=rows,
            deltas=torch.cat([it[2] for it in items]),
            alphas=torch.cat([it[3] for it in items]),
        )
        out[layer] = StepWrites(writes, torch.cat([it[1] for it in items]))
    return out


def write_fused(hidden: Tensor, residual: Tensor | None, writes: LayerWrites) -> tuple[Tensor, Tensor | None]:
    """Apply one layer's writes to vLLM's fused residual pair at the input of that layer (new tensors).

    Args:
        hidden: ``(T, d)`` the step's ``hidden_states``.
        residual: ``(T, d)`` the step's ``residual``, or None at the input of layer 0 (the stream is ``hidden``).
        writes: the layer's sites over the step's rows (:class:`StepWrites` ``writes``).

    Returns:
        The pair whose fused sum is the written stream: at layer 0 ``(written, None)``; above it, the residual's rows
        hold the written stream and the hidden rows zero.
    """
    if residual is None:
        return apply_writes(hidden.unsqueeze(0), writes).squeeze(0), None
    rows = writes.positions.to(hidden.device)
    wide = torch.promote_types(residual.dtype, torch.float32)
    stream = (hidden[rows].to(wide) + residual[rows].to(wide)).to(residual.dtype)  # as the fused add stores the sum
    local = LayerWrites(torch.zeros_like(rows), torch.arange(len(rows)), writes.deltas, writes.alphas)
    new_residual, new_hidden = residual.clone(), hidden.clone()
    new_residual[rows] = apply_writes(stream.unsqueeze(0), local).squeeze(0)
    new_hidden[rows] = 0
    return new_hidden, new_residual


def head_logits(hidden: Tensor, weight: Tensor, gather: Callable[[Tensor], Tensor | None], vocab: int) -> Tensor | None:
    """Return the logits vLLM samples and reads from, by the one head.

    That is ``kernel.readout_logits`` on this rank's vocabulary shard, gathered across tensor-parallel ranks as vLLM
    gathers them, then cut to the vocabulary.

    Args:
        hidden: ``(n, d)`` final (normed) states.
        weight: this rank's ``(vocab shard, d)`` rows of the output projection.
        gather: vLLM's gather of shard logits across ranks (None on a rank that receives nothing).
        vocab: the vocabulary size (the shards may be padded past it).
    """
    logits = gather(readout_logits(hidden, weight))
    return None if logits is None else logits[..., :vocab]


def _plain_head(language_model: Any) -> tuple[Tensor, Callable[[Tensor], Tensor | None], int]:
    """Return the head's weight shard, vLLM's gather and the vocabulary size, refusing a head the one head is not."""
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod  # noqa: PLC0415

    head, processor = language_model.lm_head, language_model.logits_processor
    if not isinstance(head.quant_method, UnquantizedEmbeddingMethod) or getattr(head, "bias", None) is not None:
        raise ValueError(
            "kimi-vllm: the output projection is quantized or has a bias; the one head reads a plain matrix"
        )
    if processor.scale != 1.0 or processor.soft_cap is not None or processor.logits_as_input:
        raise ValueError(f"kimi-vllm: vLLM's head scales ({processor.scale}) or caps ({processor.soft_cap}) logits")
    if head.tp_size > 1:
        # vLLM's own gather (gather or all-gather, its platform's choice); vLLM is pinned (env/locks/vllm.txt)
        return head.weight, processor._gather_logits, int(processor.org_vocab_size)
    return head.weight, lambda logits: logits, int(processor.org_vocab_size)


class WriteExtension:
    """Mixed into each vLLM worker (``worker_extension_cls``); its methods are called with ``LLM.collective_rpc``.

    vLLM's worker provides ``model_runner``, ``vllm_config`` and ``rank``.
    """

    model_runner: Any
    vllm_config: Any
    rank: int

    def lip_install(self, depth: int) -> dict[str, Any]:
        """Install the decoder-layer hooks, the one head and the runner wrappers, after checking the engine.

        Args:
            depth: the receiver's decoder-layer count (``ReceiverSpec.depth``).

        Returns:
            This rank's facts, plain ``int`` and ``str`` only (vLLM returns them through msgspec): ``rank``,
            ``device_name``, ``layers``, ``kv_tokens`` (the KV cache's token capacity, the least over its groups) and
            the ``vllm`` and ``torch`` versions.

        Raises:
            ValueError: on any engine setting this backend refuses (module docstring).
        """
        import vllm  # noqa: PLC0415
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer  # noqa: PLC0415

        runner, config = self.model_runner, self.vllm_config
        _check_engine(type(runner).__module__, config)
        layers = {}
        for name, module in runner.model.named_modules():
            if isinstance(module, DeepseekV2DecoderLayer):
                if not name.endswith(f"language_model.model.layers.{module.layer_idx}") or module.layer_idx in layers:
                    raise ValueError(f"kimi-vllm: unexpected decoder layer {name!r} (index {module.layer_idx})")
                layers[module.layer_idx] = module
        if sorted(layers) != list(range(depth)):
            raise ValueError(f"kimi-vllm: the engine has decoder layers {sorted(layers)[:3]}..., the receiver {depth}")
        weight, gather, vocab = _plain_head(runner.model.language_model)
        self._sources: dict[str, WriteSource] = {}
        self._writes: dict[str, dict[int, LayerWrites]] = {}
        self._step: dict[int, StepWrites] = {}
        self._log: list[dict[str, int]] = []
        for index, module in layers.items():
            module.register_forward_pre_hook(self._hook(index))
        runner.model.compute_logits = lambda hidden_states, **_: head_logits(hidden_states, weight, gather, vocab)
        self._wrap(runner)
        kv = runner.kv_cache_config
        return {
            "rank": int(self.rank),
            "device_name": str(torch.cuda.get_device_name()),
            "layers": len(layers),
            "kv_tokens": int(min(kv.num_blocks * g.kv_cache_spec.block_size for g in kv.kv_cache_groups)),
            "vllm": str(vllm.__version__),
            "torch": str(torch.__version__),  # a TorchVersion: vLLM's msgspec RPC encoder refuses str subclasses
        }

    def lip_sources(self, banks: Mapping[str, tuple[str, str]]) -> None:
        """Open the call's write banks: connector → (store root, bank digest). Banks already open are kept."""
        for name, (root, sha) in banks.items():
            current = self._sources.get(name)
            if current is None or current.bank is None or current.bank.sha256 != sha:
                self._sources[name] = BankSource(ArtifactStore(Path(root)), ArtifactRef(sha, BANK_KIND))

    def lip_steps(self) -> list[dict[str, int]]:
        """Return the engine steps since the last call (``requests``, ``tokens``, ``sites``) and forget them.

        The finished requests' writes are forgotten too: no request is in flight between engine calls.
        """
        log, self._log = self._log, []
        self._writes.clear()
        return log

    def _hook(self, index: int) -> Callable[[nn.Module, tuple[Any, ...]], tuple[Any, ...] | None]:
        def pre_hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            site = self._step.get(index)
            if site is None:
                return None
            positions, hidden, residual = args[0], args[1], args[2]
            if hidden.shape[0] != positions.shape[0]:
                raise RuntimeError(f"layer {index}: hidden states are not one row per token (sequence parallel?)")
            rows = site.writes.positions.to(positions.device)
            if not torch.equal(positions[rows].cpu(), site.positions):
                raise RuntimeError(
                    f"layer {index}: rows {rows.tolist()} hold positions {positions[rows].tolist()}, "
                    f"the slots are at {site.positions.tolist()}"
                )
            hidden, residual = write_fused(hidden, residual, site.writes)
            return (positions, hidden, residual, *args[3:])

        return pre_hook

    def _wrap(self, runner: Any) -> None:
        """Wrap the runner's ``add_requests``, ``prepare_inputs`` and ``execute_model`` (instance attributes only)."""
        add, prepare, execute = runner.add_requests, runner.prepare_inputs, runner.execute_model

        def add_requests(scheduler_output: Any) -> Any:
            for r in scheduler_output.scheduled_new_reqs:
                extra = getattr(r.sampling_params, "extra_args", None) or {}
                if REQUEST_KEY not in extra:
                    raise ValueError(
                        f"request {r.req_id}: no {REQUEST_KEY!r}; every kimi-vllm request declares its slots"
                    )
                self._writes[r.req_id] = request_writes(extra[REQUEST_KEY], self._sources, len(r.prompt_token_ids))
            return add(scheduler_output)

        def prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            batch = prepare(*args, **kwargs)
            requests = [
                StepRequest(
                    rid,
                    int(batch.query_start_loc_np[i]),
                    int(batch.num_computed_tokens_np[i]),
                    int(batch.num_scheduled_tokens[i]),
                )
                for i, rid in enumerate(batch.req_ids)
            ]
            self._step = step_writes(requests, self._writes)
            sites = sum(len(s.positions) for s in self._step.values())
            self._log.append({"requests": len(requests), "tokens": int(batch.num_tokens), "sites": sites})
            return batch

        def execute_model(*args: Any, **kwargs: Any) -> Any:
            self._step = {}  # a dummy or profiling run never sees a stale step
            try:
                return execute(*args, **kwargs)
            finally:
                self._step = {}

        runner.add_requests, runner.prepare_inputs, runner.execute_model = add_requests, prepare_inputs, execute_model


def _check_engine(runner_module: str, config: Any) -> None:
    """Refuse an engine setting that breaks the row map, the hooks or the head (module docstring)."""
    parallel, model, scheduler = config.parallel_config, config.model_config, config.scheduler_config
    refusals = {
        f"the V2 model runner ({V2_RUNNER}, VLLM_USE_V2_MODEL_RUNNER=1), not {runner_module}": runner_module
        != V2_RUNNER,
        "no sequence-parallel MoE": bool(parallel.use_sequence_parallel_moe),
        "no pipeline parallelism": parallel.pipeline_parallel_size != 1,
        "no prefix caching": bool(config.cache_config.enable_prefix_caching),
        "enforce_eager (no CUDA graphs)": not model.enforce_eager,
        "no speculative decoding": config.speculative_config is not None,
        "no async scheduling": bool(scheduler.async_scheduling),
        f"a bf16 body, not {model.dtype}": model.dtype != torch.bfloat16,
        "float32 matmuls at full precision (torch 'highest'; VLLM_FLOAT32_MATMUL_PRECISION unset)": (
            torch.get_float32_matmul_precision() != "highest"
        ),
    }
    bad = [need for need, broken in refusals.items() if broken]
    if bad:
        raise ValueError(f"kimi-vllm needs: {'; '.join(bad)}")
