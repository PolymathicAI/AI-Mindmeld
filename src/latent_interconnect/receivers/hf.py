"""The generic receiver backend: any Hugging Face causal LM, every capability, through decoder-layer pre-hooks.

It loads ``AutoModelForCausalLM`` (frozen, ``requires_grad`` off, eval mode), finds the decoder layers at
``ReceiverSpec.layers_path``, and registers a forward pre-hook on layer L that applies
:func:`~latent_interconnect.receivers.kernel.apply_writes` to the layer's input hidden states. Layer 0's input is
the token-embedding output, so a layer-0 write needs no special case. Batches are right-padded, so write positions
are the prompts' own. Every readout (next token, target NLL, generation) runs the model's headless base (the
decoder stack and its final norm, no head; one resolver, ``_headless_base``, which follows a multimodal wrapper such as
Gemma 3's to its text decoder) and applies the one head, :func:`~latent_interconnect.receivers.kernel.readout_logits`,
to the positions it reads only, so neither every layer's states nor the full ``(B, T, vocab)`` logits are kept. The
width checked against ``ReceiverSpec.d_model`` is the text config's (``config.get_text_config()``, which a multimodal
config nests). A model whose head adds a bias or scales or caps its
logits is refused, since the one head would not reproduce it (by name for the known config fields, and by a
construction-time probe whose floor on a bf16 body lets a logit scale within about a percent of 1 through:
``_check_plain_head``). Score and capture calls are partitioned by the planner (the ``plan`` option); a ``loss`` call is
the trainer's micro-batch, run as one pass; generation runs one prompt at a time (no padding; the engine fields of
``base.SEQUENTIAL_GENERATION``, the only engine it serves) and applies writes during prefill only, through one decode
loop that ``generate`` collects and ``stream`` yields. A backend built with
a state edit (``edit``, ``receivers.dictionary.StateEdit``; none by default) has one more pre-hook, on the edit's
layer, registered after the write hooks, so it reads the state the write leaves: the order is write, then edit, then
whatever reads the layer's input (:meth:`HFReceiver.pop_edit_reports`). It is correct for
every model that exposes its decoder layers as a list; it is not fast. Efficient backends for a specific model
(``kimi_k26``) register under other names and are selected in the receiver config.

Its CPU tests run on a tiny random ``LlamaForCausalLM`` saved into a local directory (no download), which is also the
receiver of the tiny end-to-end test (``configs/receivers/tiny-llama.yaml``), and on a tiny random Gemma 3 built through
``AutoModelForCausalLM`` (``Gemma3ForConditionalGeneration``, the class the real Gemma 3 models load as;
``tests/unit/tiny_gemma.py``).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator, Iterator, Mapping, Sequence
from functools import partial
from typing import Any, ClassVar, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from latent_interconnect.receivers.base import (
    PREPARATIONS,
    RECEIVER_BACKENDS,
    SEQUENTIAL_GENERATION,
    CaptureBackend,
    Generation,
    GenerationBackend,
    GenerationParams,
    GenerationPlan,
    LossTerms,
    Readout,
    ReceiverSpec,
    ScoringBackend,
    SpanStates,
    TrainingBackend,
    capture_layers,
    capture_positions,
    check_score_request,
    checked_write_layers,
    end_of_generation,
    generated_text,
    write_layers,
)
from latent_interconnect.receivers.dictionary import (
    Dictionary,
    EditReport,
    StateEdit,
    load_dictionary,
    splice,
    written_sites,
)
from latent_interconnect.receivers.kernel import (
    BatchPlan,
    LayerWrites,
    Partition,
    apply_writes,
    plan_batches,
    planned_lengths,
    readout_logits,
    reduce_spans,
    require_targets,
    resolve_sources,
    select_writes,
    state_dtype,
)
from latent_interconnect.receivers.progress import Progress
from latent_interconnect.receivers.render import RenderedPrompt, Target, Tokenizer
from latent_interconnect.receivers.tokenizer_setups import SETUPS, setup_key
from latent_interconnect.writes.sources import WriteSource

__all__ = ["HFReceiver", "load_tokenizer", "read_vocabularies_from_their_files", "set_up_tokenizer"]

_PAD_ID = 0
"""Id written into right padding: masked, and causally invisible to every real position, so its value never matters."""


def read_vocabularies_from_their_files() -> None:
    """Make tiktoken read every vocabulary from the file it is given, for this process and its children.

    A tiktoken-backed tokenizer (Kimi's remote code calls ``tiktoken.load.load_tiktoken_bpe`` with no expected hash)
    otherwise takes the vocabulary from a node-local cache file, ``<tmp>/data-gym-cache/<sha1 of the path string>``,
    whose bytes nothing checks: an input outside the hash-addressed snapshot. ``TIKTOKEN_CACHE_DIR=""`` turns that
    cache off (tiktoken 0.14.0, ``tiktoken/load.py``; the variable takes precedence over ``DATA_GYM_CACHE_DIR``), so
    the file opened is the snapshot's own. The setting replaces any value the caller's environment had and stays in
    ``os.environ``, so that a process started later (the vLLM engine and its workers) inherits it.
    """
    os.environ["TIKTOKEN_CACHE_DIR"] = ""


def load_tokenizer(spec: ReceiverSpec) -> Tokenizer:
    """Load the receiver's tokenizer at ``spec.revision`` (``trust_remote_code`` from the spec).

    Every file it reads is the pinned snapshot's (:func:`read_vocabularies_from_their_files`). The setup of the loaded
    class, if it has one, is applied (:func:`set_up_tokenizer`); no receiver plugin is imported.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415  (the optional `hf` extra; core imports without it)

    read_vocabularies_from_their_files()
    auto: Any = AutoTokenizer
    tok: Tokenizer = auto.from_pretrained(
        spec.model_id, revision=spec.revision, trust_remote_code=spec.trust_remote_code
    )
    set_up_tokenizer(tok)
    return tok


def set_up_tokenizer(tok: Tokenizer) -> None:
    """Apply the setup of the tokenizer's class (``tokenizer_setups.SETUPS``), if it has one; nothing otherwise."""
    setup = SETUPS.get(setup_key(tok))
    if setup is not None:
        setup(tok)


@RECEIVER_BACKENDS.register("hf")
class HFReceiver:
    """Scoring, training, generation and capture for a Hugging Face causal LM (implements every backend protocol).

    Args:
        spec: the receiver config.
        model: an already built model; loaded from ``spec.model_id`` at ``spec.revision`` when None.
        tokenizer: an already built tokenizer (used to decode generations); when None it is loaded like the model,
            on the first generation (scoring, training and capture never need it).
        device_map: passed to ``from_pretrained`` when loading (None: ``"auto"``, which shards a large model across
            the visible GPUs).
        dtype: weight dtype when loading (None: the checkpoint's native dtype).
        plan: the batch plan every ``score`` and ``capture`` call is partitioned with
            (:func:`~latent_interconnect.receivers.kernel.plan_batches`; None: ``BatchPlan()``, the deterministic
            default).
        generation_plan: the generation plan (None: ``base.SEQUENTIAL_GENERATION``), stored as accepted
            (:attr:`generation_plan`, which a generation runner compares with the run's). ``hf`` generates one prompt
            at a time and prefills it in one pass, so it checks only the engine fields: ``max_num_seqs`` and
            ``max_num_batched_tokens`` must be ``SEQUENTIAL_GENERATION``'s. ``prompts_per_call`` partitions a run
            into calls, which the runner owns (``evaluation.generating``); ``hf`` takes it as given.
        edit: a state edit applied in every pass that writes (``receivers.dictionary.StateEdit``; identity, from
            the config: ``base.IDENTITY_PARAMETERS``), or None, the default: no edit, no further hook. Its dictionary
            is loaded from local files and checked when the backend is built (``dictionary.load_dictionary``).

    Raises:
        ValueError: if ``device_map`` or ``dtype`` is given with a built model, the model does not match the spec
            (depth, width, decoder-layer path, which must lie inside its headless base), its head has a bias or its
            logits are scaled or capped, ``generation_plan``'s engine fields are not ``SEQUENTIAL_GENERATION``'s, or
            the edit's layer is not below the depth or its dictionary does not fit the receiver and that layer.
        FileNotFoundError: if the edit's dictionary is not in local files.
    """

    reads_banks: ClassVar[bool] = False  # takes live connectors and banks alike (in-process)
    reference_options: ClassVar[Mapping[str, str]] = {}  # no execution option that execution() records conditionally
    pads_every_batch: ClassVar[bool] = True  # serves BatchPlan.pad_every_batch (_forward)

    def __init__(
        self,
        spec: ReceiverSpec,
        *,
        model: nn.Module | None = None,
        tokenizer: Tokenizer | None = None,
        device_map: str | None = None,
        dtype: torch.dtype | None = None,
        plan: BatchPlan | None = None,
        generation_plan: GenerationPlan | None = None,
        edit: StateEdit | None = None,
    ) -> None:
        engine = None if generation_plan is None else generation_plan.model_copy(update={"prompts_per_call": None})
        if engine not in (None, SEQUENTIAL_GENERATION):
            raise ValueError(
                f"hf generates one prompt at a time in one prefill pass ({SEQUENTIAL_GENERATION}, any "
                f"prompts_per_call), not {generation_plan}"
            )
        if model is None:
            from transformers import AutoModelForCausalLM  # noqa: PLC0415  (the optional `hf` extra)

            model = AutoModelForCausalLM.from_pretrained(
                spec.model_id,
                revision=spec.revision,
                trust_remote_code=spec.trust_remote_code,
                device_map=device_map or "auto",
                dtype=dtype or "auto",
            )
        elif device_map is not None or dtype is not None:
            raise ValueError("device_map and dtype apply to loading; a built model is used as it is")
        self.spec = spec
        self.plan = BatchPlan() if plan is None else plan
        self.generation_plan = SEQUENTIAL_GENERATION if generation_plan is None else generation_plan
        self.model: nn.Module = model.eval().requires_grad_(False)
        self._tokenizer = tokenizer
        self.layers = _decoder_layers(self.model, spec)
        width = getattr(cast(Any, self.model).config.get_text_config(), "hidden_size", None)
        if width != spec.d_model:
            raise ValueError(f"model hidden size {width} (its text config's) != spec d_model {spec.d_model}")
        self._active: Mapping[int, LayerWrites] = {}
        _check_plain_head(self.model)
        self.edit = edit
        self._edit_reports: list[EditReport] = []
        dictionary = None if edit is None else self._dictionary(edit)  # loaded and checked before any hook exists
        self._hooks = [
            module.register_forward_pre_hook(self._hook(layer), with_kwargs=True)
            for layer, module in enumerate(self.layers)
        ]
        if edit is not None and dictionary is not None:
            # after the write hooks: pre-hooks run in registration order, each on the previous one's result
            edited = self.layers[edit.layer]
            self._hooks.append(edited.register_forward_pre_hook(self._edit_hook(dictionary), with_kwargs=True))

    @property
    def tokenizer(self) -> Tokenizer:
        """The tokenizer that decodes generations (loaded on first use when none was given)."""
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.spec)
        return self._tokenizer

    def close(self) -> None:
        """Remove this backend's hooks from the model (another backend may then share it without stacking them)."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    def pop_edit_reports(self) -> list[EditReport]:
        """Return the edit reports gathered since the last call, in the order of the passes, and forget them.

        One report per forward pass that the edit changed: each micro-batch of a ``score`` call in execution order
        (``kernel.plan_batches`` under :attr:`plan`; a report's ``rows`` index that micro-batch), a ``loss`` call's one
        pass, each prompt's prefill of a ``generate`` call. Empty for a backend without an edit. The reports are
        kept until taken, so a caller that edits takes them after each call.
        """
        reports, self._edit_reports = self._edit_reports, []
        return reports

    # -- capabilities ------------------------------------------------------------------------------------------------

    def execution(self) -> dict[str, str]:
        """Return the static execution facts of :mod:`latent_interconnect.receivers.base` (no history).

        ``weights`` is the directory or hub id the model was loaded from (``"in-memory"`` for a built model without
        one).
        """
        params = list(self.model.parameters())
        devices = sorted({str(p.device) for p in params})
        names = sorted({_device_name(p.device) for p in params})
        quant = getattr(self.model.config, "quantization_config", None)
        method = quant.get("quant_method") if isinstance(quant, dict) else getattr(quant, "quant_method", None)
        return {
            "backend": "hf",
            "dtype": str(_body_dtype(self._base())).removeprefix("torch."),
            "quantization": "none" if quant is None else str(method),
            "device_name": ",".join(names),
            "device_count": str(len(devices)),
            "attention": str(getattr(self.model.config, "_attn_implementation", None) or "unknown"),
            "revision": self.spec.revision,
            "weights": str(getattr(self.model, "name_or_path", "") or "in-memory"),
        }

    def placement(self) -> Mapping[int, torch.device]:
        """Return the device of each decoder layer's input."""
        return {layer: next(module.parameters()).device for layer, module in enumerate(self.layers)}

    def score(self, prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource], readout: Readout) -> Tensor:
        """Teacher-forced readout with writes (see :class:`~latent_interconnect.receivers.base.ScoringBackend`).

        Returns, on CPU in the state dtype (``receivers.kernel.state_dtype``): ``(N, len(token_ids))`` log-probs at
        each prompt's last position (``next_token``), or ``(N,)`` unweighted mean NLL over each prompt's target tokens
        (``target_nll``), both from the one head (:func:`~latent_interconnect.receivers.kernel.readout_logits`). Each
        call is partitioned by :attr:`plan`; an empty call returns an empty result in the state dtype.

        Raises:
            ValueError: by the one score-request rule (``receivers.base.check_score_request``), checked before any
                model work: a write at a layer the model does not have, then a ``target_nll`` prompt without target
                tokens.
        """
        parts: dict[int, Tensor] = {}
        with torch.no_grad():
            call = check_score_request(prompts, sources, readout, self.spec.depth, self.plan)
            if not prompts:
                width = readout.width
                return torch.zeros((0,) if width is None else (0, width), dtype=self._state_dtype())
            progress = Progress("hf score", len(prompts))
            for batch in self._partition(call.lengths):
                part, w = [prompts[i] for i in batch], select_writes(call.writes, batch)
                if call.targets is not None:
                    nll, _ = self._target_nll(part, [call.targets[i] for i in batch], w)
                    parts.update({i: x.mean().cpu() for i, x in zip(batch, nll, strict=True)})
                    progress.advance(len(batch))
                    continue
                final = self._final([p.input_ids for p in part], w)
                last = torch.tensor([len(p.input_ids) - 1 for p in part], device=final.device)
                logits = readout_logits(final[torch.arange(len(part), device=final.device), last], self._head())
                lp = torch.log_softmax(logits, -1)[:, list(readout.token_ids)].cpu()
                parts.update({i: lp[r] for r, i in enumerate(batch)})
                progress.advance(len(batch))
        return torch.stack([parts[i] for i in range(len(prompts))])

    def loss(self, prompts: Sequence[RenderedPrompt], writes: Mapping[int, LayerWrites]) -> LossTerms:
        """Weighted LM loss terms ``Σ w·ℓ`` and ``Σ w`` over the target tokens, with gradients into the writes.

        ``token_nll`` holds each prompt's per-token NLLs, detached (logging only; never part of the gradient).

        A model whose own code runs part of its forward with gradients disabled (Kimi K2.6's remote ``moe_infer`` is
        ``@torch.no_grad``) would return the right loss and a silently partial gradient. Such a model declares its
        preparation (``ReceiverSpec.training_preparation``), and every call checks it first (``base.PREPARATIONS``):
        an unprepared model is refused, so ``open_training``, whose runtime check calls this, refuses it at open.

        Raises:
            ValueError: if the model's declared training preparation has not been applied, or a prompt has no target
                tokens (``kernel.require_targets``, the readouts' rule): an empty target would give ``Σ w = 0`` and a
                NaN loss downstream.
            UnknownNameError: if the declared preparation is not registered (its plugin is not loaded).
        """
        if self.spec.training_preparation is not None:
            PREPARATIONS.get(self.spec.training_preparation)(self.model)
        write_layers(writes, self.spec.depth)  # the writes first, then the targets: check_score_request's order
        targets = require_targets(prompts, "loss")
        with torch.enable_grad():  # the loss records a graph even when the caller is under no_grad
            nll, weights = self._target_nll(prompts, targets, writes)
            ell = torch.cat(nll)
            w = torch.cat(weights).to(ell.device)
            return LossTerms(weighted_sum=(w * ell).sum(), weight_sum=w.sum(), token_nll=tuple(x.detach() for x in nll))

    def generate(
        self, prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource], params: GenerationParams
    ) -> list[Generation]:
        """Greedy (temperature 0) or sampled generation, one prompt at a time (``SEQUENTIAL_GENERATION``).

        Writes apply during prefill only.

        The next token is read from the one head at the last position (``receivers.kernel.readout_logits``).

        The whole request's write layers are checked first, before anything runs
        (``receivers.base.checked_write_layers``); each prompt's writes are then computed for its own objects alone,
        so they never depend on the other prompts in the call.

        Every prompt samples from a CPU generator seeded with ``params.seed``, so its output does not depend on the
        other prompts. Generation ends by the one rule (``receivers.base.end_of_generation``: an end-of-sequence id of
        the model's generation config, or a stop string) or at ``max_new_tokens``. Each prompt's tokens come from the
        one decode loop :meth:`stream` also serves.
        """
        decode = partial(generated_text, self.tokenizer)
        # the whole request's write layers are checked before the first prompt generates, from the sources' entries
        # (no write computed); each prompt's writes are then computed from its own objects alone, so they never depend
        # on the other prompts and memory does not grow with the call
        checked_write_layers(prompts, sources, self.spec.depth)
        out = []
        progress = Progress("hf generate", len(prompts))
        for prompt in prompts:
            new = [token for step in self._decode(prompt, sources, params) for token in step]
            ended = end_of_generation(new, eos=_eos_ids(self.model), stop=params.stop, decode=decode)
            text = decode(new) if ended is None else ended
            out.append(Generation(text=text, token_ids=tuple(new), finished=ended is not None))
            progress.advance()
        return out

    def stream(
        self, prompt: RenderedPrompt, sources: Mapping[str, WriteSource], params: GenerationParams
    ) -> Generator[tuple[int, ...], None, None]:
        """Yield one prompt's continuation as it is decoded, one token per step (``base.GenerationBackend.stream``).

        The tokens :meth:`generate` returns for this prompt, from the same loop; closing the iterator stops it.
        """
        checked_write_layers([prompt], sources, self.spec.depth)
        return self._decode(prompt, sources, params)

    @torch.no_grad()
    def _decode(
        self, prompt: RenderedPrompt, sources: Mapping[str, WriteSource], params: GenerationParams
    ) -> Generator[tuple[int, ...], None, None]:
        """The one decode loop: prefill with the prompt's writes, then one token per step until the end rule or cap.

        A generator under ``torch.no_grad`` (torch's decorator restores the caller's grad mode at every yield).
        """
        eos = _eos_ids(self.model)
        decode = partial(generated_text, self.tokenizer)
        writes = resolve_sources([prompt], sources)
        gen = torch.Generator().manual_seed(params.seed)
        ids = torch.tensor([prompt.input_ids], device=self._input_device())
        base = self._base()
        with self._writing(writes):
            step = base(input_ids=ids, use_cache=True)
        new: list[int] = []
        while len(new) < params.max_new_tokens:
            token = _pick(readout_logits(step.last_hidden_state[0, -1], self._head()), params.temperature, gen)
            new.append(token)
            yield (token,)
            if end_of_generation(new, eos=eos, stop=params.stop, decode=decode) is not None:
                return
            next_ids = torch.tensor([[token]], device=ids.device)
            step = base(input_ids=next_ids, past_key_values=step.past_key_values, use_cache=True)

    def capture(self, prompts: Sequence[RenderedPrompt], layers: Sequence[int]) -> dict[int, Tensor]:
        """Hidden states ``hidden_states[L]`` at each prompt's capture position, on CPU in the state dtype.

        For L < depth that is the input of decoder layer L (where a write at L applies); L = depth is the model's
        last hidden state, which Hugging Face models return after the final norm.

        Raises:
            ValueError: if a prompt has slots (capture reads the receiver's own states) or no capture position, or a
                layer is outside ``0 .. depth``.
        """
        return {L: states.last for L, states in self._states_at(prompts, layers, "position").items()}

    def capture_span(self, prompts: Sequence[RenderedPrompt], layers: Sequence[int]) -> dict[int, SpanStates]:
        """Per layer, each prompt's state at its span's last token and its mean over the span, on CPU (state dtype).

        The reduction runs on the model's device (``receivers.kernel.reduce_spans``): the span states are summed in a
        fixed order (never an atomic scatter-add), and only two ``(N, d_model)`` tensors per layer leave the device.
        Run-to-run identity rests on that fixed order. Layers as in :meth:`capture`.

        Raises:
            ValueError: if a prompt has slots or no span, or a layer is outside ``0 .. depth``.
        """
        return self._states_at(prompts, layers, "span")

    # -- internals ---------------------------------------------------------------------------------------------------

    def _hook(self, layer: int) -> Any:
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            writes = self._active.get(layer)
            if writes is None:
                return None
            if args:
                return (apply_writes(args[0], writes), *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": apply_writes(kwargs["hidden_states"], writes)}

        return hook

    def _dictionary(self, edit: StateEdit) -> Dictionary:
        """Load the edit's dictionary onto the device of the edited layer's input, checked against the receiver."""
        if edit.layer >= self.spec.depth:
            raise ValueError(f"edit at layer {edit.layer}; editable layers are 1 .. {self.spec.depth - 1}")
        dictionary = load_dictionary(
            edit.dictionary, model_id=self.spec.model_id, layer=edit.layer, d_model=self.spec.d_model
        )
        return dictionary.to(next(self.layers[edit.layer].parameters()).device)

    def _edit_hook(self, dictionary: Dictionary) -> Any:
        """The pre-hook of the edited layer: splice the pass's written positions, after that layer's write hook."""

        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            rows, positions = written_sites(self._active)
            if not len(rows):  # a pass without writes (a text-only prompt, a decoding step, a capture): no edit
                return None
            hidden, report = splice(args[0] if args else kwargs["hidden_states"], rows, positions, dictionary)
            self._edit_reports.append(report)
            if args:
                return (hidden, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": hidden}

        return hook

    @contextlib.contextmanager
    def _writing(self, writes: Mapping[int, LayerWrites]) -> Iterator[None]:
        # A backstop, not the rule's home: every public path checked its whole request before any model work
        # (check_score_request, checked_write_layers, loss); this catches an internal caller that did not.
        write_layers(writes, self.spec.depth)
        self._active = writes
        try:
            yield
        finally:
            self._active = {}

    def _states_at(
        self, prompts: Sequence[RenderedPrompt], layers: Sequence[int], what: Literal["position", "span"]
    ) -> dict[int, SpanStates]:
        """Return layer → each prompt's state at the last of its capture positions and their mean, on the CPU.

        The prompts are checked by the one capture-prompt rule (``receivers.base.capture_positions``).

        One forward pass per micro-batch of :attr:`plan`; the reduction runs on the device
        (``receivers.kernel.reduce_spans``). An empty call returns ``(0, d_model)`` tensors in the model's state dtype
        (``receivers.kernel.state_dtype``).
        """
        capture_layers(layers, self.spec.depth)
        at = capture_positions(prompts, what)
        dtype = self._state_dtype()
        if not prompts:
            return {
                L: SpanStates(
                    last=torch.zeros(0, self.spec.d_model, dtype=dtype),
                    mean=torch.zeros(0, self.spec.d_model, dtype=dtype),
                )
                for L in layers
            }
        last: dict[int, dict[int, Tensor]] = {L: {} for L in layers}
        mean: dict[int, dict[int, Tensor]] = {L: {} for L in layers}
        with torch.no_grad():
            for batch in self._partition(planned_lengths(prompts, "capture")):
                states = self._layer_states([prompts[i].input_ids for i in batch], layers)
                for L in layers:
                    reduced = reduce_spans(states[L], [at[i] for i in batch])
                    got_last, got_mean = reduced.last.cpu(), reduced.mean.cpu()
                    last[L].update({i: got_last[r] for r, i in enumerate(batch)})
                    mean[L].update({i: got_mean[r] for r, i in enumerate(batch)})
        order = range(len(prompts))
        return {
            L: SpanStates(last=torch.stack([last[L][i] for i in order]), mean=torch.stack([mean[L][i] for i in order]))
            for L in layers
        }

    def _partition(self, lengths: Sequence[int]) -> Partition:
        """The call's micro-batches under :attr:`plan`."""
        return plan_batches(lengths, self.plan)

    def _head(self) -> Tensor:
        """The output projection's weight, for :func:`~latent_interconnect.receivers.kernel.readout_logits`."""
        head: nn.Module = cast(Any, self.model).get_output_embeddings()
        return cast(Tensor, head.weight)

    def _base(self) -> nn.Module:
        """The model without its head: the decoder stack and its final norm (``_headless_base``)."""
        return _headless_base(self.model)

    def _state_dtype(self) -> torch.dtype:
        return state_dtype(_body_dtype(self._base()))

    def _final(self, seqs: Sequence[Sequence[int]], writes: Mapping[int, LayerWrites]) -> Tensor:
        """The final (normed) hidden states of right-padded sequences, with the writes applied; no logits."""
        out: Tensor = self._forward(seqs, writes, base=True).last_hidden_state
        return out

    def _input_device(self) -> torch.device:
        embeddings: nn.Module = cast(Any, self.model).get_input_embeddings()
        return cast(torch.device, embeddings.weight.device)

    def _forward(
        self, seqs: Sequence[Sequence[int]], writes: Mapping[int, LayerWrites], *, base: bool = False, **kwargs: Any
    ) -> Any:
        """Run the model (or, with ``base``, its base without the head) on right-padded sequences with the writes.

        Under a plan with ``pad_every_batch`` the batch is one masked column wider than its longest sequence, so
        every batch has padding and transformers builds the explicit mask for all of them
        (``examples/guides/batching.md``).
        """
        width = max(len(s) for s in seqs) + int(self.plan.pad_every_batch)
        ids = torch.full((len(seqs), width), _PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        dev = self._input_device()
        model = self._base() if base else self.model
        with self._writing(writes):
            return model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False, **kwargs)

    def _layer_states(self, seqs: Sequence[Sequence[int]], layers: Sequence[int]) -> dict[int, Tensor]:
        """``hidden_states[L]`` of right-padded sequences for the requested layers only.

        A forward pre-hook on decoder layer L keeps its input (``hidden_states[L]`` for L < depth); the body runs only
        as deep as the deepest requested layer (the hook of that layer stops the forward), and L = depth (the final
        norm's output) runs the whole base model. No other layer's states are kept, as kimi-stream streams only up to
        the deepest requested layer.
        """
        deepest = max(layers)
        got: dict[int, Tensor] = {}

        def keep(layer: int) -> Any:
            def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                got[layer] = args[0] if args else kwargs["hidden_states"]
                if layer == deepest:
                    raise _DeepEnough

            return hook

        handles = [
            self.layers[L].register_forward_pre_hook(keep(L), with_kwargs=True)
            for L in set(layers)
            if self.spec.depth > L
        ]
        try:
            final = self._forward(seqs, {}, base=True).last_hidden_state
            if self.spec.depth in layers:
                got[self.spec.depth] = final
        except _DeepEnough:
            pass
        finally:
            for handle in handles:
                handle.remove()
        return got

    def _target_nll(
        self, prompts: Sequence[RenderedPrompt], targets: Sequence[Target], writes: Mapping[int, LayerWrites]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Per prompt: the NLL of each target token (one head, state dtype), and the target's loss weights.

        ``targets`` are the prompts' targets, already checked by ``kernel.require_targets`` (once, by the caller).
        """
        final = self._final([(*p.input_ids, *t.token_ids) for p, t in zip(prompts, targets, strict=True)], writes)
        head = self._head()
        nll, weights = [], []
        for i, (p, t) in enumerate(zip(prompts, targets, strict=True)):
            n = len(p.input_ids)
            pred = readout_logits(final[i, n - 1 : n - 1 + len(t.token_ids)], head)
            labels = torch.tensor(t.token_ids, device=pred.device)
            nll.append(F.cross_entropy(pred, labels, reduction="none"))
            weights.append(torch.tensor(t.weights, dtype=pred.dtype))
        return nll, weights


class _DeepEnough(Exception):  # noqa: N818  (control flow, not an error: the requested layers are read)
    """Raised by a capture hook at the deepest requested layer, to stop the forward there."""


_LOGIT_TRANSFORMS = ("final_logit_softcapping", "logit_scale", "logits_scaling")
"""Config fields by which some models change their logits after the head (Gemma 2, Cohere, Granite): a fast first
check with a clear message; the probe in :func:`_check_plain_head` is the rule itself."""

_PROBE_TOKENS = 8
"""Tokens of the construction-time probe (ids 0 .. 7): enough to exercise the head, cheap at load."""
_PROBE_MARGIN = 8
"""The probe's margin over its rounding bound (about eps·max|z|): the gap allowed is this × eps × max(1, max|z|)."""


def _headless_base(model: nn.Module) -> nn.Module:
    """Return the model without its head (the decoder stack and its final norm): the one resolver, ``get_decoder()``.

    Transformers' ``get_decoder`` follows a multimodal wrapper to its text decoder (``Gemma3ForConditionalGeneration``
    → ``Gemma3TextModel``, where ``base_model`` is the model itself: its ``base_model_prefix`` is empty); on a plain
    causal LM (Llama; Kimi K2.6's remote DeepSeek-V3) it is the module ``base_model`` names. That the module returns
    the final states the head reads is not assumed: :func:`_check_plain_head` checks it when the backend is built.

    Raises:
        ValueError: if the model has no ``get_decoder``, it names no module, or it names the model itself
            (transformers' answer for a model with no decoder inside it).
    """
    get = getattr(model, "get_decoder", None)
    decoder = get() if callable(get) else None
    if not isinstance(decoder, nn.Module):
        raise ValueError(
            f"the model ({type(model).__name__}) names no decoder (get_decoder() gives {type(decoder).__name__}): "
            "no headless base to read final states from"
        )
    if decoder is model:
        raise ValueError(
            "the model's decoder (get_decoder()) is the model itself: no headless base to read final states from"
        )
    return decoder


def _check_plain_head(model: nn.Module) -> None:
    """Refuse a model whose logits are not the one head applied to its headless base's final states.

    The rule is behavioral: one short forward compares the model's own log-probabilities with
    ``log_softmax(kernel.readout_logits(_headless_base(model)(ids, attention_mask=ones).last_hidden_state, head))``. It
    catches any transform after the head, named or not (remote code included), a resolved base that is not what the
    head reads, and a base whose run alone differs from its run inside the model's forward. The base gets an all-ones
    mask, as the runtime path always passes a mask (``_forward``): on Gemma 3 that gives the base alone the same mask
    form as the wrapper's own forward (with no mask the text model builds an explicit one, U12a). Before
    it, the structure is checked (output embeddings exist; ``_headless_base`` resolves a separate module), and the
    known config fields that transform logits (in ``config.get_text_config()``) are refused by name: they are the first
    line of defense, the probe the backstop.

    The bound: rounding a logit z to the coarser of the body's and the model's own logit dtype (resolution eps) moves it
    by at most eps·|z|/2, so the two log-softmaxes differ by at most about eps·max|z|; the probe allows
    :data:`_PROBE_MARGIN` × eps × max(1, max|z|). What it lets through: on a bf16 body (eps = 2⁻⁷) at logits near 24,
    the allowance is about 1.5 nats, so a hidden logit scale of ×1.01 (a shift of about 0.5 nats there) passes; on a
    float32 body (eps ≈ 1.2e-7) the allowance is about 2e-5 nats and such a scale is refused.

    Raises:
        ValueError: for no output embeddings, no headless base (``_headless_base``), a head with a bias, a config
            that caps or scales the logits, or logits the one head does not reproduce.
    """
    m = cast(Any, model)
    head = m.get_output_embeddings()
    if head is None:
        raise ValueError("the model has no output embeddings (get_output_embeddings() is None): no head to read")
    base = _headless_base(model)
    if getattr(head, "bias", None) is not None:
        raise ValueError("the model's head has a bias; the one head (kernel.readout_logits) has none")
    config = m.config.get_text_config()
    changed = {k: getattr(config, k) for k in _LOGIT_TRANSFORMS if getattr(config, k, None) not in (None, 1, 1.0)}
    if changed:
        raise ValueError(f"the model transforms its logits after the head ({changed}); the one head does not")
    ids = torch.arange(min(_PROBE_TOKENS, int(head.weight.shape[0])), device=m.get_input_embeddings().weight.device)
    with torch.no_grad():
        own = m(input_ids=ids[None], use_cache=False).logits[0]
        alone = base(input_ids=ids[None], attention_mask=torch.ones_like(ids[None]), use_cache=False)
        one = readout_logits(alone.last_hidden_state[0], head.weight)
    # the coarser of the text body's and the model's own logit dtype (Kimi's remote code casts logits to float32)
    eps = max(torch.finfo(_body_dtype(base)).eps, torch.finfo(own.dtype).eps)
    tol = _PROBE_MARGIN * eps * max(1.0, float(one.abs().max()))
    gap = float(
        (torch.log_softmax(own.to(device=one.device, dtype=one.dtype), -1) - torch.log_softmax(one, -1)).abs().max()
    )
    if not gap <= tol:
        raise ValueError(f"the model's logits are not the one head's: log-prob gap {gap:.3g} > {tol:.3g}")


def _body_dtype(model: nn.Module) -> torch.dtype:
    """The body's compute dtype: that of its first floating-point parameter (packed integer weights are skipped).

    ``hf`` passes its headless base (``_headless_base``), the text decoder: a multimodal model's first parameter is its
    vision tower's (Gemma 3), whose dtype may differ from the text body's.

    Raises:
        ValueError: if the model has no floating-point parameter.
    """
    for p in model.parameters():
        if p.dtype.is_floating_point:
            return p.dtype
    raise ValueError("the model has no floating-point parameter: no body dtype to compute readouts from")


def _decoder_layers(model: nn.Module, spec: ReceiverSpec) -> nn.ModuleList:
    """The decoder-layer list at ``spec.layers_path``, inside the headless base the readouts run (``_headless_base``).

    Raises:
        ValueError: if the path is missing, is not a ``ModuleList`` of ``spec.depth`` layers, or lies outside the
            headless base (a write hooked there would never run on a readout).
    """
    obj: Any = model
    for part in spec.layers_path.split("."):
        if not hasattr(obj, part):
            raise ValueError(f"the model has no {spec.layers_path!r} (missing {part!r})")
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise ValueError(f"{spec.layers_path!r} is a {type(obj).__name__}, not a list of decoder layers")
    if len(obj) != spec.depth:
        raise ValueError(f"the model has {len(obj)} decoder layers, the spec says depth {spec.depth}")
    decoder = _headless_base(model)
    if not any(m is obj for m in decoder.modules()):  # e.g. Gemma 3's vision layers: hooks there never see the text
        raise ValueError(f"{spec.layers_path!r} is not inside the text decoder ({type(decoder).__name__}) hf reads")
    return obj


def _eos_ids(model: nn.Module) -> frozenset[int]:
    config = getattr(model, "generation_config", None)
    eos = getattr(config, "eos_token_id", None)
    if eos is None:
        return frozenset()
    return frozenset([eos] if isinstance(eos, int) else eos)


def _pick(logits: Tensor, temperature: float, gen: torch.Generator) -> int:
    if temperature == 0.0:
        return int(logits.argmax())
    probs = torch.softmax(logits.float().cpu() / temperature, -1)
    return int(torch.multinomial(probs, 1, generator=gen))


def _device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else device.type


# Static protocol conformance, checked by mypy: the registry hands backends out through a cast, which would hide a
# signature that drifted from the protocol.
_SCORING: type[ScoringBackend] = HFReceiver
_TRAINING: type[TrainingBackend] = HFReceiver
_GENERATION: type[GenerationBackend] = HFReceiver
_CAPTURE: type[CaptureBackend] = HFReceiver
