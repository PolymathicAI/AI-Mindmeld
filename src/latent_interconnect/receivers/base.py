"""Receivers: the frozen language model a connector writes into, described by a config and served by backends.

A receiver is a :class:`ReceiverSpec` plus one backend per *capability*:

==============  ======================================================  =====================================
capability      what it does                                            protocol
==============  ======================================================  =====================================
``score``       teacher-forced readouts (next-token log-probs, NLL)     :class:`ScoringBackend`
``train``       loss terms with gradients into the writes               :class:`TrainingBackend`
``generate``    free generation with writes during prefill              :class:`GenerationBackend`
``capture``     the receiver's own hidden states (ridge targets)        :class:`CaptureBackend`
==============  ======================================================  =====================================

Backends are registered in :data:`RECEIVER_BACKENDS`. ``hf`` (``receivers.hf``) serves every capability for any
Hugging Face causal LM. ``kimi_k26`` is a plugin package with faster Kimi backends. A training backend also
implements :class:`ScoringBackend`, so the trainer's in-job evaluation scores on the same loaded model.

Scoring and generation take write *sources* (live connector or bank, ``writes.sources``) per connector name;
training takes resolved per-layer writes, because the trainer owns the connector and its gradients.

Semantics every backend shares:

- ``execution()`` describes how the backend executes, as strings, with no history: ``backend``, ``dtype`` (the
  body's native precision), ``quantization`` (``"none"`` if none), ``device_name``, ``device_count``,
  ``attention``, ``revision`` (the weights' revision) and ``weights`` (the directory, or the hub id, the weights were
  read from). A backend with execution choices beyond these adds keys naming them (``kimi-ep``: ``experts``, its
  expert GEMM and blocks, and ``checkpointing``). The stage copies it into its record's provenance
  (``StageResult.execution``); it is never part of an identity;
- every ``score`` and ``capture`` call is batched by the one planner, ``kernel.plan_batches``, under the backend's
  configured ``kernel.BatchPlan`` (the constructor's ``plan`` parameter, given explicitly from the config by
  ``receivers.open_*``, never as an execution option; the default is the deterministic plan). The plan
  changes numbers (``examples/guides/batching.md``), so the configs that score or capture carry it in their identity,
  and each result records the digest of the partition it was computed with (``kernel.partition_digest``). A score call's
  writes are computed per micro-batch of that partition (:func:`check_score_request`). A ``loss`` call is
  the trainer's micro-batch (the trainer fixes its size) and is not re-planned;
- every ``generate`` call is batched under the backend's configured :class:`GenerationPlan` (the constructor's
  ``generation_plan`` parameter, given from the config like ``plan``; the default is concurrent): how
  many sequences decode together and how many tokens one engine step holds. It changes numbers as the batch plan
  does, so it is part of the identity of every generation config, and a backend refuses a plan it cannot honor;
- every readout (next-token log-probs, target NLL, generation) takes its logits from the one head,
  ``kernel.readout_logits``: float32 or wider on the native-precision body;
- ``target_nll`` is the unweighted mean NLL over the target tokens (a readout; training loss weights do not apply);
- writes enter at layers ``0 .. depth - 1`` only: :func:`write_layers` is the one write-layer rule, and every
  backend applies it to every call that writes (score, loss, generation), so a write past the last layer is refused,
  never ignored;
- a score request is checked whole before any model work, in one order (write layers, then targets):
  :func:`check_score_request` is the one rule, so every backend refuses the same request with the same first error
  and runs nothing (not even a connector) first; generation checks its write layers the same way
  (:func:`checked_write_layers`);
- ``measure`` reports memory as NaN on CPU;
- generation applies writes during prefill only, and samples with a generator seeded from ``GenerationParams.seed``.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import torch
from pydantic import Field, model_validator
from torch import Tensor

from latent_interconnect.config import Config, added_field
from latent_interconnect.plugins import Registry
from latent_interconnect.receivers.kernel import (
    BatchPlan,
    LayerWrites,
    ReadoutKind,
    SpanStates,
    plan_batches,
    planned_lengths_and_targets,
    resolve_sources,
)
from latent_interconnect.receivers.render import RenderedPrompt, Target, Tokenizer
from latent_interconnect.writes.sources import WriteSource

__all__ = [
    "IDENTITY_PARAMETERS",
    "PREPARATIONS",
    "RECEIVER_BACKENDS",
    "SEQUENTIAL_GENERATION",
    "Capability",
    "CaptureBackend",
    "Generation",
    "GenerationBackend",
    "GenerationParams",
    "GenerationPlan",
    "LossTerms",
    "Readout",
    "ReceiverSpec",
    "ScoreRequest",
    "ScoringBackend",
    "SpanStates",
    "TrainingBackend",
    "capture_layers",
    "capture_positions",
    "check_score_request",
    "checked_write_layers",
    "end_of_generation",
    "generated_text",
    "write_layers",
]

Capability = Literal["score", "train", "generate", "capture"]


class ReceiverSpec(Config):
    """A receiver model, as data.

    Attributes:
        name: registry name.
        model_id: Hugging Face model id, or a local model directory (the test fixtures build a random tiny model into
            one and load a copy of ``configs/receivers/tiny-llama.yaml`` pointing at it).
        revision: immutable revision (commit sha) of the weights.
        depth: number of decoder layers. Writable layers are 0 .. depth - 1 (input of decoder layer L).
        d_model: hidden width.
        placeholder: text of one slot token; it must encode to exactly one token id (checked at load).
        layers_path: attribute path of the decoder-layer list, for the generic backend.
        template_kwargs: chat-template arguments per conversation flag and state, both stated, e.g. for Kimi
            ``{"thinking": {"on": {"thinking": true}, "off": {"thinking": false}}}`` (a template's own default is never
            relied on: Kimi's renders thinking ON when the argument is absent).
        trust_remote_code: whether the model's own code is loaded (needed by Kimi; the revision is pinned).
        thinking_close: the text that closes the receiver's thinking block (``"</think>"`` for Kimi K2.6), a target part
            between a reasoning scaffold and the answer, and where a generated trace is cut before its readout
            (``evaluation.runners``); None for a receiver without a thinking mode, whose scaffold is ordinary assistant
            text closed by its ``separator`` and whose traces are cut only at a finished trace's end id.
        separator: for a receiver without a thinking close, the text between a reasoning written as ordinary assistant
            text and its answer (Gemma 3: one blank line): a scaffold's close (``data.scaffold``) and an
            after-trace prefill's ``{close}`` (``evaluation.runners``). At most one of ``thinking_close`` and
            ``separator`` is set; a receiver with neither can be used where no reasoning is closed, and the scaffold
            and ``{close}`` refuse it.
        thinking_request: words appended, after one space, to the last user message of a prompt in thinking mode
            (``receivers.render.Renderer``): how a receiver without a thinking template argument is asked to reason
            (Gemma 3: ``"Reason first; give the answer only at the end."``); None for none (Kimi K2.6's
            thinking mode is its template argument). It needs ``template_kwargs`` for ``thinking`` (``{}`` for both
            states when the template takes none). Training rows never carry it: a scaffold on a receiver without a
            thinking close is assistant text in the row's own mode, and no stream turns thinking on
            (``data.scaffold``).
        backends: capability → registered backend name.
        plugins: modules to import before resolving backends.
        training_preparation: the preparation the model's own code needs before a backend that runs that code can
            train through it (a name in :data:`PREPARATIONS`, registered by the plugin that provides it), or None.
            Kimi K2.6 declares ``kimi-moe-gradients``: its remote ``moe_infer`` is ``@torch.no_grad``. ``hf`` checks
            it on every ``loss`` call, so ``open_training`` refuses an unprepared model (fail closed); a backend that
            replaces that code (``kimi-ep``) does not need it. The name is resolved when any backend opens
            (``receivers._open``): an unregistered one is refused there.
    """

    name: str
    model_id: str
    revision: str
    depth: int = Field(ge=1)
    d_model: int = Field(ge=1)
    placeholder: str
    layers_path: str = "model.layers"
    template_kwargs: dict[str, dict[Literal["on", "off"], dict[str, Any]]] = Field(default_factory=dict)
    trust_remote_code: bool = False
    thinking_close: str | None = None
    separator: str | None = added_field(default=None, min_length=1)
    thinking_request: str | None = added_field(default=None, min_length=1)
    backends: dict[Capability, str]
    plugins: tuple[str, ...] = ()
    training_preparation: str | None = None

    @model_validator(mode="after")
    def _one_close(self) -> ReceiverSpec:
        if self.thinking_close is not None and self.separator is not None:
            raise ValueError(
                f"a reasoning is closed by the thinking close or, without one, the separator, not both: "
                f"{self.thinking_close!r}, {self.separator!r}"
            )
        if self.thinking_request is not None and "thinking" not in self.template_kwargs:
            raise ValueError("a thinking request needs template arguments for 'thinking' ({} for both states)")
        return self


class Readout(Config):
    """What a scoring pass returns.

    Attributes:
        kind: ``"next_token"`` (log-probs of ``token_ids`` at the prompt's last position) or ``"target_nll"``
            (per-token NLL over the prompt's target span).
        token_ids: vocabulary ids to read for ``"next_token"`` (at least one); empty for ``"target_nll"``, which reads
            the target's own tokens. A readout that breaks this is refused at construction, so no backend checks it.

    The backend computes in its native fast precision; the precision it used is recorded with the scores.
    """

    kind: ReadoutKind
    token_ids: tuple[int, ...] = ()

    @property
    def width(self) -> int | None:
        """Values per prompt: ``len(token_ids)`` for ``next_token``, None for ``target_nll`` (one value, the NLL).

        A score result is ``(N, width)``, or ``(N,)`` when the width is None, an empty call included.
        """
        return len(self.token_ids) if self.kind == "next_token" else None

    @model_validator(mode="after")
    def _token_ids_fit_the_kind(self) -> Readout:
        if self.kind == "next_token" and not self.token_ids:
            raise ValueError("a next_token readout reads at least one token id; token_ids is empty")
        if self.kind == "target_nll" and self.token_ids:
            raise ValueError(f"a target_nll readout reads the target's own tokens; token_ids {self.token_ids} unused")
        return self


class GenerationParams(Config):
    """Free-generation settings.

    Attributes:
        max_new_tokens: the trace cap, recorded with every generated row (2,048 for the
            paper's generation suites; the residual cap rate reported).
        temperature: sampling temperature (0 = greedy).
        stop: stop strings.
        seed: sampling seed.
    """

    max_new_tokens: int = Field(ge=1)
    temperature: float = 0.0
    stop: tuple[str, ...] = ()
    seed: int = 0


class GenerationPlan(Config):
    """How a generation backend batches the prompts of one call.

    A configured value, as :class:`~latent_interconnect.receivers.kernel.BatchPlan` is for scoring. Requests enter in
    the call's prompt order and are served first come, first served. At most ``max_num_seqs`` sequences are in flight
    at once (prefilling or decoding); one engine step processes at most ``max_num_batched_tokens`` tokens, so a longer
    prefill is split into chunks over several steps (chunked prefill). Writes still apply only to prompt positions,
    whichever step computes them. The partition of a generation call into engine steps follows from the prompts, the
    outputs and this plan alone; a backend whose engine could also preempt a request when memory runs short refuses to
    start unless memory for the whole plan is guaranteed. With ``prompts_per_call`` a generation run is split, in prompt
    order, into consecutive calls of that many prompts, each committed as it finishes, so a killed run resumes by
    whole calls (``evaluation.generating``); a call's last sequences decode with fewer in flight, so this too is part
    of the plan.

    The default is concurrent, for simplicity and as the original vLLM runs batched their
    generations: 64 sequences in flight and 16,384 tokens per engine step (chunked prefill). Both values are this
    package's choice; the old engine used vLLM's own defaults. Concurrency moves bf16 numbers as other
    batch plans do (``examples/guides/batching.md``); the plan is recorded with every generation.
    :data:`SEQUENTIAL_GENERATION` decodes one sequence at a time with each prefill in one pass, the only engine ``hf``
    serves (with any ``prompts_per_call``: the call partition is the runner's).

    Attributes:
        max_num_seqs: the most sequences in flight at once.
        max_num_batched_tokens: the most tokens one engine step processes (a prefill longer than this is chunked);
            None: no limit, so every prefill is one pass.
        prompts_per_call: the most prompts one ``generate`` call of a generation run submits (the run's calls are
            committed one by one and resumed); None: the whole run in one call, committed at its end (added field).
    """

    max_num_seqs: int = Field(default=64, ge=1)
    max_num_batched_tokens: int | None = Field(default=16_384, ge=1)
    prompts_per_call: int | None = added_field(default=None, ge=1)


SEQUENTIAL_GENERATION = GenerationPlan(max_num_seqs=1, max_num_batched_tokens=None)
"""One sequence at a time, each prefill in one pass: every output independent of the call's other prompts (``hf``'s
engine; ``hf`` also takes these engine fields with a ``prompts_per_call``)."""


@dataclass(frozen=True)
class Generation:
    """One generated continuation.

    Attributes:
        text: the decoded continuation (:func:`generated_text`).
        token_ids: its token ids.
        finished: True if it ended on EOS or a stop string, False if it hit ``max_new_tokens``.
    """

    text: str
    token_ids: tuple[int, ...]
    finished: bool

    @property
    def n_tokens(self) -> int:
        """Length of the continuation in tokens."""
        return len(self.token_ids)


def end_of_generation(
    token_ids: Sequence[int], *, eos: Collection[int], stop: Sequence[str], decode: Callable[[Sequence[int]], str]
) -> str | None:
    """The one end-of-generation rule: the final text if a continuation has ended, else None.

    A continuation ends on an end-of-sequence id as its last token (kept in ``Generation.token_ids``, not in the text)
    or when its decoded text contains a stop string (the text is cut before the first one; the token ids keep the
    tokens that formed it). Every generation backend applies it, to each continuation it returns.

    Args:
        token_ids: the continuation so far.
        eos: the receiver's end-of-sequence ids (its generation config's ``eos_token_id``).
        stop: ``GenerationParams.stop``.
        decode: the continuation decoder (:func:`generated_text` with the receiver's tokenizer).
    """
    if token_ids and token_ids[-1] in eos:
        return decode(token_ids[:-1])
    if stop:
        text = decode(token_ids)
        cut = min((i for i in (text.find(s) for s in stop) if i >= 0), default=-1)
        if cut >= 0:
            return text[:cut]
    return None


def generated_text(tokenizer: Tokenizer, token_ids: Sequence[int]) -> str:
    """Decode a continuation's ids to its text, the one decoding every generation backend applies.

    Special tokens (end of turn, padding) are dropped, and no space is inserted around the added tokens that remain
    (Kimi's ``<think>`` / ``</think>`` are added, not special, so a trace keeps them as generated). The ids, special
    ones included, stay in ``Generation.token_ids``.
    """
    return tokenizer.decode(list(token_ids), skip_special_tokens=True, spaces_between_special_tokens=False)


@dataclass(frozen=True)
class LossTerms:
    """Un-normalized loss terms: ``Σ w·ℓ`` and ``Σ w`` over target tokens, and the per-token NLLs for logging.

    Micro-batching and packing therefore never change the weighting; the trainer forms ``Σ w·ℓ / Σ w`` over the
    whole optimizer batch. The loss is the weighted sums alone: ``token_nll`` is detached, so it is never part of
    the gradient; it serves the trainer's logs (the unweighted mean, positional bins, per-stream losses). Given the
    prompts' target weights, ``Σ_prompts Σ_tokens w·token_nll`` equals ``weighted_sum`` (up to summation order).

    Attributes:
        weighted_sum: ``Σ w·ℓ``, a 0-d tensor with gradients into the writes.
        weight_sum: ``Σ w``, a 0-d tensor.
        token_nll: per prompt, in call order, a detached 1-D tensor of the NLL ``ℓ`` of each of its target tokens
            (in the target's order; the state dtype, ``kernel.state_dtype``).
    """

    weighted_sum: Tensor
    weight_sum: Tensor
    token_nll: tuple[Tensor, ...]


@runtime_checkable
class ScoringBackend(Protocol):
    """Teacher-forced scoring with writes.

    A score request is checked whole before any model work by :func:`check_score_request`, which a backend calls
    first and from which it takes the resolved writes, the planned lengths and the checked targets (a ``target_nll``
    prompt has a non-empty target, ``kernel.require_targets``; a training backend's ``loss`` applies that rule too).

    Attributes:
        reads_banks: True for a backend that runs the receiver in other processes and reads write banks by
            reference: every source it is given has a ``bank`` (``writes.sources.BankSource``), and the ``suite``
            stage builds the banks for it. False for a backend that takes any source, live connectors included.
    """

    reads_banks: ClassVar[bool]

    @property
    def plan(self) -> BatchPlan:
        """The configured batch plan every ``score`` call is partitioned with (``kernel.plan_batches``)."""
        ...

    def execution(self) -> dict[str, str]:
        """Return how this backend executes (precision, device type, kernels): recorded with every score."""
        ...

    def score(self, prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource], readout: Readout) -> Tensor:
        """Return ``(N, len(token_ids))`` log-probs (``next_token``) or ``(N,)`` mean target NLL (``target_nll``)."""
        ...


@runtime_checkable
class TrainingBackend(ScoringBackend, Protocol):
    """Loss with gradients into the writes; the receiver's own weights are frozen. Also scores (in-job evaluation).

    ``receivers.open_training`` runs the runtime gradient check (``receivers.gradients.check_gradients``) on every
    training backend it opens; a backend implements nothing for it. Its gradients are shown correct by its tests
    (``receivers.conformance``: parity with the reference backend, which itself passes finite differences).

    Attributes:
        reference_options: the backend's execution options that ``execution()`` records only when one is not its
            original value, each at that value (empty for a backend without such options): a run whose record
            lacks them ran these values, and the trainer's resume refusal names them (``Trainer._check_execution``).
    """

    reference_options: ClassVar[Mapping[str, str]]

    def placement(self) -> Mapping[int, torch.device]:
        """Return layer → device where that layer's input lives (connectors put each entry's map there)."""
        ...

    def loss(self, prompts: Sequence[RenderedPrompt], writes: Mapping[int, LayerWrites]) -> LossTerms:
        """Return the loss terms of the prompts' target tokens (the trainer's micro-batch, not re-planned)."""
        ...


@runtime_checkable
class GenerationBackend(Protocol):
    """Free generation; writes are applied during prefill only.

    Attributes:
        reads_banks: whether the backend reads write banks by reference, as for :class:`ScoringBackend`.
    """

    reads_banks: ClassVar[bool]

    @property
    def generation_plan(self) -> GenerationPlan:
        """The configured generation plan every ``generate`` call is batched under."""
        ...

    def execution(self) -> dict[str, str]:
        """Return how this backend executes (precision, device type, engine version): recorded with every output."""
        ...

    def generate(
        self, prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource], params: GenerationParams
    ) -> list[Generation]:
        """Return one :class:`Generation` per prompt."""
        ...

    def stream(
        self, prompt: RenderedPrompt, sources: Mapping[str, WriteSource], params: GenerationParams
    ) -> Generator[tuple[int, ...], None, None]:
        """Yield one prompt's continuation as the engine produces it: the token ids each engine step adds.

        The ids, joined, are the ``token_ids`` :meth:`generate` returns for the prompt alone; the caller applies the one
        end rule and decoding to them (:func:`end_of_generation`, :func:`generated_text`). The prompt's write layers are
        checked before anything runs. Closing the iterator (``close()``, or dropping it) cancels the request: the
        backend stops decoding and frees what it holds for it. One stream at a time per backend (chat serves one
        request at a time; ``latent_interconnect.chat``).
        """
        ...


def capture_positions(prompts: Sequence[RenderedPrompt], what: Literal["position", "span"]) -> list[tuple[int, ...]]:
    """Return each capture prompt's positions to read, checked: the one capture-prompt rule every backend applies.

    A capture reads the receiver's own states, so a capture prompt has no slots (nothing is written into it). For a
    ``position`` capture each prompt has a ``capture_position`` inside its ids; for a ``span`` capture a non-empty
    ``capture_span`` inside its ids.

    Args:
        prompts: the call's prompts.
        what: ``"position"`` (``CaptureBackend.capture``) or ``"span"`` (``CaptureBackend.capture_span``).

    Returns:
        Per prompt, ``(capture_position,)`` or its ``capture_span``.

    Raises:
        ValueError: naming the first prompt that has slots or lacks the positions.
    """
    out = []
    for i, p in enumerate(prompts):
        at = (() if p.capture_position is None else (p.capture_position,)) if what == "position" else p.capture_span
        if p.slots:
            raise ValueError(f"prompt {i}: a capture prompt has no slots (capture reads the receiver's own states)")
        if not at or not all(0 <= j < len(p.input_ids) for j in at):
            raise ValueError(f"prompt {i}: a capture {what} inside its {len(p.input_ids)} ids is needed, got {at}")
        out.append(at)
    return out


@dataclass(frozen=True)
class ScoreRequest:
    """A score request, checked by :func:`check_score_request` before any model work.

    Attributes:
        writes: layer → the call's resolved writes (``kernel.resolve_sources``), at checked layers.
        lengths: each prompt's planned length (``kernel.planned_lengths``), for ``kernel.plan_batches``.
        targets: each prompt's checked target for a ``target_nll`` readout (``kernel.require_targets``), else None.
    """

    writes: dict[int, LayerWrites]
    lengths: list[int]
    targets: list[Target] | None


def checked_write_layers(
    prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource], depth: int
) -> list[int]:
    """Return the layers a request writes at, checked before any write is computed: the whole request's write check.

    Each slot's connector writes its source's ``entries`` (the same for every object), so the request's layers are
    known without running a connector or reading a bank; :func:`write_layers` checks them. Every backend calls this
    first on a request that writes (``check_score_request``, generation).

    Raises:
        KeyError: if a slot names a connector the sources lack.
        ValueError: naming the layers outside ``0 .. depth - 1``.
    """
    names = {occ.binding.connector for p in prompts for occ in p.slots}
    return write_layers({e.layer for name in sorted(names) for e in sources[name].entries}, depth)


def check_score_request(
    prompts: Sequence[RenderedPrompt],
    sources: Mapping[str, WriteSource],
    readout: Readout,
    depth: int,
    plan: BatchPlan,
) -> ScoreRequest:
    """Check a whole score request before any model work: the one score-request rule every backend applies first.

    The order is fixed, so every backend reports the same error first: the write layers
    (:func:`checked_write_layers`, before any write is computed), then the targets (``kernel.require_targets``, once)
    and the planned lengths. Only then are the writes computed (``kernel.resolve_sources``); their layers are checked
    again, which catches a source whose ``entries`` understate what it writes. A ``Readout`` is valid by
    construction.

    The writes are computed per micro-batch of ``plan`` (``kernel.plan_batches``, the partition the backend then
    scores), so a live connector's peak memory scales with the configured scoring batch, not the call. Each object's
    write depends on that object alone, so this equals one resolution of the whole call up to float32 rounding: an
    execution detail of the plan, already part of the scoring identity, with no identity of its own.

    Args:
        prompts: the call's prompts.
        sources: connector name → its write source.
        readout: what the call reads.
        depth: the receiver's decoder-layer count.
        plan: the backend's batch plan.

    Raises:
        KeyError: if a slot names a connector or object the sources lack.
        ValueError: a write past the last layer, or a ``target_nll`` prompt without target tokens.
    """
    checked_write_layers(prompts, sources, depth)
    lengths, targets = planned_lengths_and_targets(prompts, readout.kind)
    parts = [(b, resolve_sources([prompts[i] for i in b], sources)) for b in plan_batches(lengths, plan)]
    writes = {}
    for layer in sorted({layer for _, w in parts for layer in w}):
        got = [(torch.tensor(b)[w[layer].rows], w[layer]) for b, w in parts if layer in w]
        order = torch.argsort(torch.cat([rows for rows, _ in got]), stable=True)  # call order, as one resolution
        writes[layer] = LayerWrites(
            rows=torch.cat([rows for rows, _ in got])[order],
            positions=torch.cat([w.positions for _, w in got])[order],
            deltas=torch.cat([w.deltas for _, w in got])[order.to(got[0][1].deltas.device)],
            alphas=torch.cat([w.alphas for _, w in got])[order.to(got[0][1].alphas.device)],
        )
    write_layers(writes, depth)
    return ScoreRequest(writes=writes, lengths=lengths, targets=targets)


def write_layers(layers: Iterable[int], depth: int) -> list[int]:
    """Return the layers a call writes at, checked: the one write-layer rule every backend applies.

    A write at layer L enters at the input of decoder layer L (L = 0 is the token-embedding output), so the writable
    layers are ``0 .. depth - 1``; a write past them would have no layer to enter and is refused, never ignored.

    Args:
        layers: the layers of a call's resolved writes (e.g. the keys of ``kernel.resolve_sources``' result).
        depth: the receiver's decoder-layer count (``ReceiverSpec.depth``).

    Raises:
        ValueError: naming the layers outside ``0 .. depth - 1``.
    """
    out = sorted(layers)
    bad = [layer for layer in out if not 0 <= layer < depth]
    if bad:
        raise ValueError(f"writes at layers {bad}; writable layers are 0 .. {depth - 1} (the input of decoder layer L)")
    return out


def capture_layers(layers: Sequence[int], depth: int) -> list[int]:
    """Return the layers a capture reads, checked: the one capture-layer rule every backend applies.

    A capture reads ``hidden_states[L]`` for each requested L, so at least one layer is needed,
    and every layer lies in ``0 .. depth`` (``depth`` is the final norm's output).

    Args:
        layers: the requested layers.
        depth: the receiver's decoder-layer count (``ReceiverSpec.depth``).

    Raises:
        ValueError: if ``layers`` is empty or names a layer outside ``0 .. depth``.
    """
    if not layers:
        raise ValueError("a capture reads at least one layer; none was requested")
    bad = [layer for layer in layers if not 0 <= layer <= depth]
    if bad:
        raise ValueError(f"capture layers {bad} are outside 0 .. {depth} (hidden_states indices)")
    return list(layers)


@runtime_checkable
class CaptureBackend(Protocol):
    """The receiver's hidden states at chosen positions (targets for ridge initialization).

    A capture prompt has no slots, and its capture position (``capture``) or span (``capture_span``) lies inside its
    ids; a capture reads at least one layer, each in ``0 .. depth``. :func:`capture_positions` and
    :func:`capture_layers` are the one rule, and every backend applies both before running the model.
    """

    @property
    def plan(self) -> BatchPlan:
        """The configured batch plan every ``capture`` call is partitioned with (``kernel.plan_batches``)."""
        ...

    def execution(self) -> dict[str, str]:
        """Return how this backend executes (precision, device type): recorded with every capture."""
        ...

    def capture(self, prompts: Sequence[RenderedPrompt], layers: Sequence[int]) -> dict[int, Tensor]:
        """Return layer → ``(N, d_model)`` states at each prompt's ``capture_position``."""
        ...

    def capture_span(self, prompts: Sequence[RenderedPrompt], layers: Sequence[int]) -> dict[int, SpanStates]:
        """Return layer → the :class:`SpanStates` (last token, mean) of each prompt's ``capture_span``.

        The reduction happens on the device, so a pass's host memory is two ``(N, d_model)`` tensors per layer
        whatever the span lengths; a recipe's target picks ``last`` or ``mean``.
        """
        ...


RECEIVER_BACKENDS: Registry[type[Any]] = Registry("receiver backend")
"""Backend classes by name. Each is constructed by ``receivers.open_*`` as ``cls(spec, plan=..., **kwargs)``.

The constructor contract: after ``spec``, keyword parameters with defaults; ``plan`` accepted by every scoring,
capture and training backend and ``generation_plan`` by every generation backend (identity, passed from the config;
:data:`IDENTITY_PARAMETERS`); every parameter settable from a command line (``open_*(options=...)``,
bound by ``config.bind_options``) annotated with ``str``, ``int``, ``float``, ``bool``, ``Path``, ``None`` or
``Literal`` in annotations that resolve at run time. Other parameters (a built model, a ``torch.dtype``) are given as
values from code (``open_*(**construction)``)."""


PREPARATIONS: Registry[Callable[[Any], None]] = Registry("training preparation")
"""Training preparations by name (``ReceiverSpec.training_preparation``), registered by the plugin that provides them.

Each is a check on the loaded model's root module that raises ``ValueError``, naming the remedy, if the preparation
has not been applied. A backend that runs the model's own code calls it before every loss (``hf``)."""

IDENTITY_PARAMETERS: dict[str, str] = {
    "plan": "the batch plan is part of a result's identity: it is set in the "
    "config (SuiteRunConfig.plan, CaptureSpec.plan) and passed explicitly, never as an execution option",
    "generation_plan": "the generation plan is part of a result's identity: it is set in the "
    "config (SuiteRunConfig.generation_plan) and passed explicitly, never as an execution option",
    "edit": "a state edit is part of a result's identity: it is set in the config (SuiteRunConfig.edit) and "
    "passed explicitly, never as an execution option",
}
"""Backend parameters that are part of a result's identity, with the reason: never set from a command line.

The batch plan is here because a deterministic default plan is part of the identity of every scoring and capture
config; the generation plan for the same reason, for generation; a state edit
(``receivers.dictionary.StateEdit``) because it changes the states the receiver computes. The precision
(``dtype``) is not: native fast precision is an execution choice, recorded in provenance.
``config.bind_options`` refuses these names on every stage and every backend."""
