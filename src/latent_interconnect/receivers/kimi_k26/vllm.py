"""The ``kimi-vllm`` backend: Kimi K2.6 generation and teacher-forced scoring on a resident vLLM engine (8×H200).

One vLLM 0.28 engine (V2 model runner, tensor parallel over the node's GPUs, eager) holds the model; the worker
extension (``vllm_worker.py``) applies the writes at the input of decoder layer L in vLLM's fused-residual form, during
prefill only, and replaces vLLM's logit computation with the one head. It runs in the ``vllm`` environment
(``env/locks/vllm.txt``); vLLM is imported when the
backend is constructed, so the plugin imports anywhere.

Semantics shared with ``kimi-stream`` and ``hf``, each by its one rule:

- a score request is checked whole first (``receivers.base.check_score_request``), a generation request's write layers
  too (``receivers.base.checked_write_layers``), before any engine work;
- writes are ``kernel.apply_writes`` at the input of decoder layer L (L = 0 is the token embedding), at the slots'
  prompt positions only; the workers read them from the write banks by reference (``reads_banks``: every source is a
  ``writes.sources.BankSource``), and every engine call checks that each write site was applied exactly once;
- every logit is ``kernel.readout_logits`` (float32 on the bf16 body), sampled or read;
- a ``score`` call is partitioned by ``kernel.plan_batches`` under ``plan``; each micro-batch is one engine call that
  must run as one engine step (checked before and after), so a score depends on the batch plan as ``kimi-stream``'s
  does. ``next_token`` reads vLLM's log-probabilities of the requested ids at the prompt's last position
  (``logprob_token_ids``, the raw log-softmax of the head's float32 logits); ``target_nll`` reads the prompt
  log-probabilities of prompt + target (``prompt_logprobs``) and averages the target tokens' negatives;
- every engine call queues all its requests, in order, while vLLM's scheduling is paused (``LLM.sleep(level=0)``),
  and only then lets the engine step: vLLM's engine core otherwise steps on the first requests to arrive;
- a ``generate`` call submits its prompts in order to the engine, which batches them under the declared
  ``GenerationPlan`` (sequences in flight, tokens per step: identity); the engine refuses to start when its KV cache
  cannot hold every in-flight sequence at full length, so a request is never preempted and recomputed under memory
  pressure. Each prompt samples with its own generator seeded from ``GenerationParams.seed`` (vLLM's per-request
  seed) and ends by ``receivers.base.end_of_generation``.
- a ``stream`` serves one request alone (``llm_engine.add_request``, then ``step`` for its delta outputs as the caller
  reads them); vLLM's engine core runs ahead in its own process, so tokens queue until read. Closing it aborts the
  request, so none is in flight between calls; its step log is checked as every engine call's.

vLLM does not run Kimi on CPU; the GPU parity job compares it with ``kimi-stream`` on the real model.
"""

from __future__ import annotations

import json
from collections.abc import Generator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import Tensor

from latent_interconnect.provenance import DeviceClass, visible_devices
from latent_interconnect.receivers.base import (
    RECEIVER_BACKENDS,
    Generation,
    GenerationBackend,
    GenerationParams,
    GenerationPlan,
    Readout,
    ReceiverSpec,
    ScoringBackend,
    check_score_request,
    checked_write_layers,
    end_of_generation,
    generated_text,
)
from latent_interconnect.receivers.hf import load_tokenizer, read_vocabularies_from_their_files
from latent_interconnect.receivers.kernel import (
    BatchPlan,
    LayerWrites,
    plan_batches,
    resolve_sources,
    select_writes,
    state_dtype,
)
from latent_interconnect.receivers.kimi_k26 import vllm_worker
from latent_interconnect.receivers.kimi_k26.weights import snapshot_path
from latent_interconnect.receivers.progress import Progress
from latent_interconnect.receivers.render import RenderedPrompt
from latent_interconnect.writes.sources import BankSource, WriteSource

__all__ = ["EXTENSION", "MAX_READ_IDS", "KimiVLLM", "bank_references", "step_budget"]

EXTENSION = f"{vllm_worker.__name__}.WriteExtension"
"""The worker extension vLLM mixes into every worker (``worker_extension_cls``)."""
MAX_READ_IDS = 128
"""The most token ids one ``next_token`` readout reads: vLLM 0.28's ``MAX_LOGPROB_TOKEN_IDS``."""


def bank_references(
    prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource]
) -> dict[str, tuple[str, str]]:
    """Return connector → (store root, bank digest) for every connector the prompts' slots name.

    Raises:
        ValueError: naming a connector whose source is not a write bank (``kimi-vllm`` reads banks by reference).
    """
    out = {}
    for name in sorted({o.binding.connector for p in prompts for o in p.slots}):
        source = sources[name]
        if not isinstance(source, BankSource) or source.bank is None:
            raise ValueError(f"kimi-vllm reads write banks by reference; connector {name!r} has a live source")
        out[name] = (str(source.store.root), source.bank.sha256)
    return out


def step_budget(plan: GenerationPlan, max_model_len: int) -> int:
    """The most tokens one engine step processes: the plan's, or ``max_model_len`` when it sets none (no chunking)."""
    return max_model_len if plan.max_num_batched_tokens is None else plan.max_num_batched_tokens


def _sites(writes: Mapping[int, LayerWrites]) -> int:
    return sum(len(w.positions) for w in writes.values())


@RECEIVER_BACKENDS.register("kimi-vllm")
class KimiVLLM:
    """Kimi K2.6 ``generate`` and ``score`` on a resident vLLM engine, writes read from banks by reference.

    Implements :class:`~latent_interconnect.receivers.base.GenerationBackend` and
    :class:`~latent_interconnect.receivers.base.ScoringBackend`.

    Args:
        spec: the receiver config (``configs/receivers/kimi-k2.6.yaml``).
        weights: the weights directory (e.g. a node-local copy), checked against the pinned snapshot
            (``weights.snapshot_path``); the pinned snapshot itself when None.
        max_model_len: the longest prompt plus continuation the engine accepts, in tokens. The default, 4,096, holds a
            generation suite's prompt and its 2,048-token cap with room, and lets the concurrent default plan's
            sequences fit the KV cache together (the start-up check below); raise it for longer prompts, lowering
            ``max_num_seqs`` if the check refuses.
        gpu_memory_utilization: the fraction of each GPU's memory vLLM may take (weights, activations, KV cache).
        plan: the batch plan of ``score`` calls (None: ``BatchPlan()``).
        generation_plan: the generation plan (None: ``GenerationPlan()``, the concurrent default).

    Raises:
        WrongHardwareError: unless the visible devices are one node of eight whole accelerators (``8xH200``;
            any accelerator model, since the count is the engine's topology: it spans all eight, tensor parallel).
            Memory is the engine's own start-up check.
        ValueError: if the engine is not one this backend serves (``vllm_worker``), or its KV cache cannot hold
            ``generation_plan.max_num_seqs`` sequences of ``max_model_len`` tokens (preemption would be possible).
    """

    reads_banks: ClassVar[bool] = True

    def __init__(
        self,
        spec: ReceiverSpec,
        *,
        weights: Path | None = None,
        max_model_len: int = 4_096,
        gpu_memory_utilization: float = 0.9,
        plan: BatchPlan | None = None,
        generation_plan: GenerationPlan | None = None,
    ) -> None:
        from vllm import LLM  # noqa: PLC0415  (the vllm environment only)

        devices = visible_devices()
        DeviceClass.H200X8.require(devices)
        self.spec = spec
        self.plan = BatchPlan() if plan is None else plan
        self.generation_plan = GenerationPlan() if generation_plan is None else generation_plan
        self.weights = snapshot_path(spec, weights)
        self.max_model_len = max_model_len
        self.tensor_parallel = len(devices)
        self.gpu_memory_utilization = gpu_memory_utilization
        gp = self.generation_plan
        read_vocabularies_from_their_files()  # before the engine starts: its processes load the tokenizer too
        self._llm = LLM(
            model=str(self.weights),
            trust_remote_code=spec.trust_remote_code,
            dtype="bfloat16",
            tensor_parallel_size=self.tensor_parallel,
            enforce_eager=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            max_model_len=max_model_len,
            max_num_seqs=gp.max_num_seqs,
            max_num_batched_tokens=step_budget(gp, max_model_len),
            enable_chunked_prefill=gp.max_num_batched_tokens is not None,
            logprobs_mode="raw_logprobs",
            gpu_memory_utilization=gpu_memory_utilization,
            worker_extension_cls=EXTENSION,
            seed=0,
        )
        self.facts: list[dict[str, Any]] = self._llm.collective_rpc("lip_install", args=(spec.depth,))
        kv = min(int(f["kv_tokens"]) for f in self.facts)
        if kv < gp.max_num_seqs * max_model_len:
            raise ValueError(
                f"kimi-vllm: the KV cache holds {kv} tokens, fewer than {gp.max_num_seqs} sequences × {max_model_len}: "
                "a request could be preempted and recomputed; lower max_num_seqs or max_model_len"
            )
        config = json.loads((self.weights / "config.json").read_text())
        quantization = config.get("text_config", config).get("quantization_config")
        self._quantization = "none" if quantization is None else f"compressed-tensors {quantization['format']}"
        eos = json.loads((self.weights / "generation_config.json").read_text())["eos_token_id"]
        self._eos = frozenset([eos] if isinstance(eos, int) else eos)
        self._tokenizer = load_tokenizer(spec)
        self._streams = 0

    def execution(self) -> dict[str, str]:
        """Return the static execution facts of :mod:`latent_interconnect.receivers.base`, plus ``engine``."""
        return {
            "backend": "kimi-vllm",
            "dtype": "bfloat16",
            "quantization": self._quantization,
            "device_name": str(self.facts[0]["device_name"]),
            "device_count": str(self.tensor_parallel),
            "attention": "vllm",
            "revision": self.spec.revision,
            "weights": str(self.weights),
            "engine": f"vllm {self.facts[0]['vllm']} (V2 model runner, eager), torch {self.facts[0]['torch']}",
        }

    def score(self, prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource], readout: Readout) -> Tensor:
        """Teacher-forced readout with writes (see :class:`~latent_interconnect.receivers.base.ScoringBackend`).

        Raises:
            ValueError: by the one score-request rule first; then a live source, a readout of more than
                :data:`MAX_READ_IDS` ids, or a micro-batch the engine could not run in one step.
            RuntimeError: if a micro-batch ran in more than one engine step or applied other than its write sites.
        """
        call = check_score_request(prompts, sources, readout, self.spec.depth, self.plan)
        banks = bank_references(prompts, sources)
        width = readout.width
        out = torch.empty(
            (len(prompts),) if width is None else (len(prompts), width), dtype=state_dtype(torch.bfloat16)
        )
        if not prompts:
            return out
        if len(readout.token_ids) > MAX_READ_IDS:
            raise ValueError(
                f"kimi-vllm reads at most {MAX_READ_IDS} token ids per prompt, not {len(readout.token_ids)}"
            )
        targets = call.targets
        sequences = [
            p.input_ids if targets is None else p.input_ids + targets[i].token_ids for i, p in enumerate(prompts)
        ]
        budget = step_budget(self.generation_plan, self.max_model_len)
        progress = Progress("kimi-vllm score", len(prompts))
        for batch in plan_batches(call.lengths, self.plan):
            tokens = sum(call.lengths[i] for i in batch)
            if len(batch) > self.generation_plan.max_num_seqs or tokens > budget:
                raise ValueError(
                    f"a micro-batch of {len(batch)} prompts and {tokens} tokens exceeds one engine step "
                    f"({self.generation_plan.max_num_seqs} sequences, {budget} tokens): declare a larger generation "
                    "plan"
                )
            params = [self._score_params(prompts[i], readout) for i in batch]
            results = self._run([sequences[i] for i in batch], params, banks, _sites(select_writes(call.writes, batch)))
            if [step["requests"] for step in results.steps] != [len(batch)]:
                raise RuntimeError(
                    f"a micro-batch of {len(batch)} prompts ran as engine steps {results.steps}, not one"
                )
            for i, result in zip(batch, results.outputs, strict=True):
                out[i] = self._read(result, readout, len(prompts[i].input_ids), sequences[i])
            progress.advance(len(batch))
        return out

    def generate(
        self, prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource], params: GenerationParams
    ) -> list[Generation]:
        """Generate one continuation per prompt, writes during prefill only, under the declared generation plan.

        Raises:
            ValueError: a write past the last layer (the one rule, first), a live source, or a prompt whose
                continuation could exceed ``max_model_len``.
            RuntimeError: if the engine applied other than the call's write sites.
        """
        checked_write_layers(prompts, sources, self.spec.depth)
        banks = bank_references(prompts, sources)
        if not prompts:
            return []
        long = [i for i, p in enumerate(prompts) if len(p.input_ids) + params.max_new_tokens > self.max_model_len]
        if long:
            raise ValueError(f"prompts {long[:5]} plus {params.max_new_tokens} new tokens exceed {self.max_model_len}")
        expected = _sites(resolve_sources(prompts, sources))
        sampling = [self._generation_params(p, params) for p in prompts]
        results = self._run([p.input_ids for p in prompts], sampling, banks, expected)
        decode = partial(generated_text, self._tokenizer)
        out = []
        for result in results.outputs:
            ids = tuple(int(t) for t in result.outputs[0].token_ids)
            ended = end_of_generation(ids, eos=self._eos, stop=params.stop, decode=decode)
            text = decode(ids) if ended is None else ended
            out.append(Generation(text=text, token_ids=ids, finished=ended is not None))
        return out

    def stream(
        self, prompt: RenderedPrompt, sources: Mapping[str, WriteSource], params: GenerationParams
    ) -> Generator[tuple[int, ...], None, None]:
        """Yield one prompt's continuation per engine step (``base.GenerationBackend.stream``), one request at a time.

        The request is added to the engine (``llm_engine.add_request``) and its delta outputs are read by ``step`` as
        the caller iterates; the engine core may run ahead in its own process (the outputs queue until read). Nothing
        else is in flight; closing the iterator aborts the request. Its checks are :meth:`generate`'s, before the
        request is added. Its step log is read by :meth:`_steps`, the one check of every engine call: the ranks ran
        the same steps and, once the request has finished, its write sites were applied exactly once.

        Raises:
            ValueError: as :meth:`generate`.
            RuntimeError: if a finished request's engine steps applied other than its write sites.
        """
        checked_write_layers([prompt], sources, self.spec.depth)
        banks = bank_references([prompt], sources)
        if len(prompt.input_ids) + params.max_new_tokens > self.max_model_len:
            raise ValueError(f"the prompt plus {params.max_new_tokens} new tokens exceeds {self.max_model_len}")
        sites = _sites(resolve_sources([prompt], sources))
        return self._stream(prompt, banks, sites, params)

    def _stream(
        self, prompt: RenderedPrompt, banks: Mapping[str, tuple[str, str]], sites: int, params: GenerationParams
    ) -> Generator[tuple[int, ...], None, None]:
        from vllm.inputs import TokensPrompt  # noqa: PLC0415
        from vllm.sampling_params import RequestOutputKind  # noqa: PLC0415

        sampling = self._generation_params(prompt, params, output_kind=RequestOutputKind.DELTA)
        engine = self._llm.llm_engine
        self._llm.collective_rpc("lip_sources", args=(dict(banks),))
        self._streams += 1
        request = f"lip-stream-{self._streams}"
        engine.add_request(request, TokensPrompt(prompt_token_ids=list(prompt.input_ids)), sampling)
        finished = False
        try:
            while not finished:
                for output in engine.step():
                    if output.request_id != request:
                        continue
                    finished = bool(output.finished)
                    ids = tuple(int(t) for t in output.outputs[0].token_ids)
                    if ids:
                        yield ids
        finally:
            if not finished:
                engine.abort_request([request])
            self._steps(sites if finished else None)  # forgets the request's writes, finished or not

    # -- internals ---------------------------------------------------------------------------------------------------

    def _run(
        self,
        sequences: Sequence[Sequence[int]],
        params: Sequence[Any],
        banks: Mapping[str, tuple[str, str]],
        sites: int,
    ) -> _Results:
        """One engine call: the banks opened on every worker, requests queued with scheduling paused, steps checked."""
        from vllm.inputs import TokensPrompt  # noqa: PLC0415

        self._llm.collective_rpc("lip_sources", args=(dict(banks),))
        # vLLM's engine core steps as soon as it holds a request, while later requests are still arriving (job
        # 7178206: 64 prompts ran as steps of 3 and 61). Scheduling is paused (vLLM's level-0 sleep) while the call's
        # requests are queued, so the engine schedules them together; ``enqueue`` aborts its own requests on failure.
        self._llm.sleep(level=0, mode="keep")
        try:
            self._llm.enqueue([TokensPrompt(prompt_token_ids=list(s)) for s in sequences], list(params), use_tqdm=False)
        finally:
            self._llm.wake_up(tags=["scheduling"])
        outputs = self._llm.wait_for_completion(use_tqdm=False)
        return _Results(outputs, self._steps(sites))

    def _steps(self, sites: int | None) -> list[dict[str, int]]:
        """Read (and clear) the engine's step log: the ranks must agree and, unless None, ``sites`` must be applied.

        Raises:
            RuntimeError: if the tensor-parallel ranks ran different steps, or the steps applied other than ``sites``.
        """
        logs = self._llm.collective_rpc("lip_steps")
        if any(log != logs[0] for log in logs):
            raise RuntimeError("kimi-vllm: the tensor-parallel ranks ran different engine steps")
        applied = sum(step["sites"] for step in logs[0])
        if sites is not None and applied != sites:
            raise RuntimeError(f"kimi-vllm: the engine applied {applied} write sites, the call has {sites}")
        steps: list[dict[str, int]] = logs[0]
        return steps

    def _score_params(self, prompt: RenderedPrompt, readout: Readout) -> Any:
        from vllm import SamplingParams  # noqa: PLC0415

        extra = {vllm_worker.REQUEST_KEY: vllm_worker.encode_slots(prompt)}
        if readout.kind == "next_token":
            return SamplingParams(
                max_tokens=1, temperature=0.0, logprob_token_ids=list(readout.token_ids), extra_args=extra
            )
        return SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, extra_args=extra)

    def _generation_params(self, prompt: RenderedPrompt, params: GenerationParams, **kind: Any) -> Any:
        from vllm import SamplingParams  # noqa: PLC0415

        return SamplingParams(
            max_tokens=params.max_new_tokens,
            temperature=params.temperature,
            seed=params.seed,
            stop=list(params.stop),
            stop_token_ids=sorted(self._eos),
            skip_special_tokens=False,
            extra_args={vllm_worker.REQUEST_KEY: vllm_worker.encode_slots(prompt)},
            **kind,
        )

    @staticmethod
    def _read(result: Any, readout: Readout, prompt_len: int, sequence: Sequence[int]) -> Tensor:
        """One prompt's readout from its engine output (log-probabilities in float32 or wider)."""
        if readout.kind == "next_token":
            logprobs = result.outputs[0].logprobs[0]
            return torch.tensor([logprobs[t].logprob for t in readout.token_ids], dtype=torch.float64)
        per_token = result.prompt_logprobs
        nll = [-per_token[at][sequence[at]].logprob for at in range(prompt_len, len(sequence))]
        return torch.tensor(nll, dtype=torch.float64).mean()


class _Results:
    """One engine call's outputs (vLLM's, in request order) and its engine steps (``vllm_worker`` step log)."""

    def __init__(self, outputs: Sequence[Any], steps: Sequence[Mapping[str, int]]) -> None:
        self.outputs = outputs
        self.steps = steps


# Static protocol conformance, checked by mypy: the registry hands backends out through a cast, which would hide a
# signature that drifted from the protocol.
_SCORING: type[ScoringBackend] = KimiVLLM
_GENERATION: type[GenerationBackend] = KimiVLLM
