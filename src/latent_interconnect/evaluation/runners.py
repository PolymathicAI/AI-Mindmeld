"""The readout runners: ``yes_no``, ``caption_nll`` and ``composition`` (registered in ``SUITE_KINDS``).

Each scores every (item, task, condition) of the item set, in that order, and commits a score table
(``evaluation.scores``). ``yes_no`` and ``caption_nll`` fill one slot. ``composition`` fills two or more slots of one
role in one prompt: the item's one object through several connectors (Figs 8a-b), each slot written by its own connector
at its own positions, in the slot order of the frame. The same connector may fill several slots (the repeated-source
controls, the original ``MMM``). Its ``readout`` param picks the Yes/No or the target-NLL readout below, and its rows
have the single-slot shape: one role, so one set of role columns; which connector fills which slot is the suite spec's
(part of the score's identity), so each arrangement is its own suite. Several objects in one prompt (distinct roles: the
original part B relational pairs and triplets) are refused: their per-item prompts, labels and targets have no item-set
form yet.

**One call, a configured plan** (``examples/guides/batching.md``). A run hands all its prompts to the backend
in one call, item-major (items in item-set order, then tasks and conditions in spec order: the original engine's
sequence order), never re-sorted. The backend forms micro-batches with the one planner (``kernel.plan_batches``)
under the configured plan (``SuiteContext.plan``, from ``SuiteRunConfig.plan``, part of the suite run's identity).
The runner checks that the backend was built with that plan, computes the same partition from the prompts' token
counts (``kernel.planned_lengths``, the one rule by pass kind) and records it in the score artifact's
metadata (``batch_plan``: the plan, and per call its prompt count and ``kernel.partition_digest``); each row's
``batch`` column is the index of its micro-batch in that partition. Nothing is read from ``execution()``, which is
provenance.

**Prompts.** The user message is the suite's frame (``SuiteSpec.frame_in``: a domain frame, or the suite's own) with
the task's bare question in place (``Frame.ask``): the frame carries the wording around the question and the answer
instruction (the A5 frame "Here is an unknown protein, given to you as an embedding:<SLOT:x>. {question} Answer Yes or
No."), so the rendered message is the paper's, and the runners add no words. Under the suite's answer convention
(``SuiteSpec.answer``, ``data.answers``) the frame carries only the task and object wording; the convention's
instruction follows it, its prefix opens the answer turn and its answers are read. There are no in-context examples.
Under ``SuiteSpec.frame_cycle`` each item is asked in its unit's frame (``SuiteSpec.item_frames``), the
same under every condition and at every step. Every rendered prompt is vetted (``evaluation.vetting``) before anything
is scored, within its frame.

**Conditions** of the slots' role (each applies to every slot):

- ``own``: the item's object, written by each slot's connector;
- ``mismatched`` (likelihood suites only: ``caption_nll``, and ``composition`` with ``readout: target_nll``):
  the object at the item's position in a seeded derangement of the item set (``data.items.derangement``, seed
  ``mismatch_seed``), in every slot (the original part A's all-slots-shuffled caption baseline);
- ``name``: the object's name (``Domain.objects`` ``name``) in place of the marker, rendered as its own text piece
  (``Renderer.render`` ``texts``); nothing is written. Single-slot kinds only: one name in several markers is no paper
  condition. A suite of name conditions only reads no connector (``SuiteSpec.reads_connector``, the one definition):
  the runner then takes no write source and reads no k;
- ``elicited_name`` (``yes_no`` reading a ``name_guess`` run, ``SuiteSpec.upstream_kind``): the name the receiver
  gave for the item's object from the slot (the upstream run's ``own`` generation, its ``answer``), in place of the
  marker as ``name`` renders a name; an empty answer is refused unless ``empty_elicited_name`` states the text shown
  instead (the paper: "an uncharacterized protein").

- the variant conditions (``data.suites.FIELD_CONDITIONS``; an item set of ``build.variant_items``, whose fields
  name each item's partners): ``sibling``, ``wt`` and ``derange`` write the field's object
  (``<condition>.<role>``) through the connector in place of the item's own; ``name_substitution`` shows the field's
  text as ``name`` renders a name. ``name`` shows the item set's ``name.<role>`` field when it has one (a variant's
  parent name; variants have none in the domain). A slot with ``fill: wt`` (``SlotSpec.fill``) always holds the item's
  wild-type parent (its name under a text condition), so the relational prompt's wild-type slot stays fixed while the
  conditions vary the mutant.

``value`` and ``placeholder``, the cross-domain factorial's structure-text and omitted cells
(``evaluation.cross_domain``), are refused.

**Unavailable objects** (``SuiteContext.unavailable``: the objects this run's embedding inputs recorded as skipped,
with the recorded reason; ``run.suite``). A row whose
prompt writes one is left out and listed in the score table's metadata (``unavailable``); a row that writes none (a
text condition, or a partner that is held) is scored as usual. Nothing is refilled or redrawn.

**Suite-declared tasks** (``SuiteSpec.task_specs``) join the domain's catalog for the suite: the variant labels and the
variant effect text, which the domain asks no question of. Each is in scope for every item.

**Reading a trace** (``yes_no`` reading a ``generation`` run). Each prompt continues the upstream run's trace of the
same item, task and condition: the runner re-renders the trace's prompt and matches its digest
(``evaluation.suites.Upstream.check_prompt``), then appends the trace's token ids (``render.TokenIds``) up to the first
thinking close (``ReceiverSpec.thinking_close``), or, for a trace that never closed it and for every trace of a receiver
without a thinking close (whose traces are ordinary assistant text: the Gemma recipe), all of
them without the final end-of-sequence id of a finished trace, and then the prefill text, and reads the answer words.
The prefill's ``{close}`` is the receiver's thinking close, or else its separator (``ReceiverSpec.separator``), so one
spec serves every receiver: ``"{close}Answer:"`` is Kimi's ``"</think>Answer:"`` and, on a receiver without a thinking
close, the same assistant turn continued with the separator and ``"Answer:"``. A capped trace is closed this
way too, never dropped, and keeps every token; its row records ``finished`` false. The upstream run must have no
stop strings, so a finished trace ends on its end-of-sequence id, the only id removed. Each row carries the trace's
``text``, ``n_tokens``, ``finished`` and ``max_new_tokens`` (``scores.GENERATION_COLUMNS``). Vetting reads the
re-rendered prompts before the traces. Under ``YesNoParams.cut`` (the 200-token
protocol) a capped trace that never closed its thinking keeps its ids only up to its
last complete sentence (``evaluation.trace_cut``; a trace with none is kept whole, the ``fallback``) before the close
and the forced answer, and every row records ``n_tokens_kept``, ``cut_rule`` and ``n_dropped``
(``scores.CUT_COLUMNS``); a trace that closed its thinking, or finished, is continued as above (rule ``none``),
except that under ``cut: answer_line`` a finished trace of a receiver without a thinking close keeps its ids only up to
its own answer (``trace_cut.answer_line_cut``, rule ``answer_line``), so the readout does not copy it. The
readout's score is the forced answer's; the trace's own stated answer stays in its generation table and ``text``
(``generating.stated_answer``), never merged with it.

Under an answer convention the trace's prompt carries no answer instruction (a generation takes none:
``data.answers``), and the trace's close (the receiver's thinking close, else its separator) and the convention's
prefix follow the trace, each its own piece, as a training scaffold row ends (``data.scaffold``).

**Readouts.** Yes/No (``yes_no``; ``composition`` with ``readout: yes_no``): ``log p(yes) − log p(no)`` of the next
token after the prompt and its prefill (``yes_no`` params ``prefill``; the paper's direct readouts have none),
for the answer words ``yes`` / ``no`` (params, or the answer convention's; each must be one token,
``Renderer.token_id``). The row also keeps the two log-probabilities themselves (``logp_yes``, ``logp_no``:
``scores.ANSWER_COLUMNS``), as the backend returned them from the same call, so the answer mass
``exp(logp_yes) + exp(logp_no)`` is in the record; nothing is checked on it.
Target NLL (``caption_nll``; ``composition`` with ``readout: target_nll``): the unweighted mean NLL of the task's text
field (``TaskSpec.label``) cut to ``max_target_tokens`` tokens by the renderer, passed as one full-weight span (the
renderer weighs tokens outside every span 0): text tokens only, with no end-of-turn token (the original caption
NLLs included it). Items whose text is empty are not scored (as the original in-job set excluded them). Items outside a
task's scope (``Domain.scope_mask``) are not scored; an in-scope item of a Yes/No task without a label is an error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.config import Config, added_field
from latent_interconnect.data.answers import Answering, answering
from latent_interconnect.data.items import ItemSet, derangement
from latent_interconnect.data.suites import FIELD_CONDITIONS, LIKELIHOOD_READOUT, Condition, SlotSpec, SuiteSpec
from latent_interconnect.domains.base import Domain, TaskKind, TaskSpec
from latent_interconnect.evaluation.scores import write_scores
from latent_interconnect.evaluation.suites import (
    SUITE_KINDS,
    SuiteContext,
    Upstream,
    UpstreamError,
    execution_devices,
    upstream_of,
)
from latent_interconnect.evaluation.trace_cut import CutRule, answer_line_cut, last_sentence_cut
from latent_interconnect.evaluation.vetting import PromptRecord, vet_prompts
from latent_interconnect.receivers.base import Readout, ScoringBackend, generated_text
from latent_interconnect.receivers.kernel import partition_digest, plan_batches, planned_lengths
from latent_interconnect.receivers.render import (
    Conversation,
    Message,
    RenderedPrompt,
    Renderer,
    SlotBinding,
    TokenIds,
    WeightedSpan,
)
from latent_interconnect.writes.sources import WriteSource

__all__ = [
    "CaptionNLL",
    "CaptionParams",
    "Composition",
    "CompositionParams",
    "PromptVettingError",
    "YesNo",
    "YesNoParams",
    "answer_log_probs",
    "answer_readout",
    "log_odds",
    "slot_fill",
    "slot_tokens",
    "yes_no_prompt",
]

_MIN_COMPOSITION_SLOTS = 2
_TEXT_CONDITIONS = (Condition.NAME, Condition.ELICITED_NAME, Condition.NAME_SUBSTITUTION)
"""Conditions that put a name in place of the marker: their prompt text depends on the item by design (vetting), and
nothing is written."""


class _NoClose(Enum):
    """The trace close of a receiver without a thinking close (``ReceiverSpec.thinking_close`` None).

    Its own value, not None: a ``close`` of None means the suite continues no trace (no upstream run, or a
    ``name_guess`` one), while this one means it continues traces that have no thinking delimiter to cut at.
    """

    NO_CLOSE = "no_close"


_NO_CLOSE = _NoClose.NO_CLOSE
_CLOSE = "{close}"
"""In an after-trace prefill: the receiver's thinking close, or else its separator (:class:`YesNoParams`)."""


class PromptVettingError(ValueError):
    """A suite's rendered prompts failed vetting (``evaluation.vetting``); the findings are the message."""


class YesNoParams(Config):
    """Parameters of the ``yes_no`` kind (``SuiteSpec.params``).

    Attributes:
        yes: the inline positive answer word, read as one token (e.g. ``"Yes"``; the post-reasoning readout
            ``" Yes"``); None under the suite's answer convention (``SuiteSpec.answer``), which then gives it.
        no: the inline negative answer word, as ``yes``.
        prefill: the inline text the answer turn starts with before the answer is read ("" for none); after a trace,
            the text after its token ids, where ``{close}`` (a literal replace) stands for the receiver's thinking
            close or, without one, its separator (e.g. ``"{close}Answer:"``). A suite that continues no trace
            refuses ``{close}``. Under an answer convention it carries no answer words: ``""`` for a direct readout
            (the convention's prefix follows the prompt) and exactly ``"{close}"``, the structural marker, after a
            trace (the receiver's close, then the convention's prefix).
        empty_elicited_name: the text shown in the ``elicited_name`` condition when the elicited name is empty
            (the paper: ``"an uncharacterized protein"``); None refuses an empty one.
        cut: after a trace, ``"last_sentence"`` cuts a capped trace that never closed its thinking at its last
            complete sentence before the forced answer (``evaluation.trace_cut``), and
            each row records the cut ("Reading a trace"); ``"answer_line"`` cuts so too and, on a receiver without a
            thinking close, cuts a finished trace before its own answer (``trace_cut.answer_line_cut``; the label is the
            prefill's text after ``{close}``); None keeps every id of a capped trace,
            as before.
    """

    yes: str | None = added_field(default=None)
    no: str | None = added_field(default=None)
    prefill: str = added_field(default="")
    empty_elicited_name: str | None = added_field(default=None)
    cut: Literal["last_sentence", "answer_line"] | None = added_field(default=None)


class CaptionParams(Config):
    """Parameters of the ``caption_nll`` kind (``SuiteSpec.params``).

    Attributes:
        max_target_tokens: the text is cut to this many tokens (the paper: 128); no end-of-turn token follows.
        mismatch_seed: seed of the derangement behind the ``mismatched`` condition (required when it is scored).
    """

    max_target_tokens: int = Field(ge=1)
    mismatch_seed: int | None = None


class CompositionParams(Config):
    """Parameters of the ``composition`` kind (``SuiteSpec.params``).

    Attributes:
        readout: ``yes_no`` (Yes/No tasks: the original part A keyword and held-out questions) or ``target_nll`` (text
            tasks: the original part A caption NLL; a likelihood suite, so ``mismatched`` is allowed).
        yes: the inline positive answer word, read as one token (``yes_no`` without an answer convention only,
            ``SuiteSpec.answer``).
        no: the inline negative answer word, as ``yes``.
        max_target_tokens: the target is cut to this many tokens (``target_nll`` only; no end-of-turn token).
        mismatch_seed: seed of the derangement behind ``mismatched`` (``target_nll`` only; required when it is
            scored).
    """

    readout: Literal["yes_no", "target_nll"]
    yes: str | None = None
    no: str | None = None
    max_target_tokens: int | None = Field(default=None, ge=1)
    mismatch_seed: int | None = None

    @model_validator(mode="after")
    def _check(self) -> CompositionParams:
        answer = (self.yes, self.no)
        target = (self.max_target_tokens, self.mismatch_seed)
        if self.readout == "yes_no" and target != (None, None):
            raise ValueError(f"readout yes_no takes no target params, got {target}")
        if self.readout == LIKELIHOOD_READOUT and (self.max_target_tokens is None or answer != (None, None)):
            raise ValueError(f"readout target_nll takes max_target_tokens (and mismatch_seed), got answer {answer}")
        return self


@dataclass(frozen=True)
class _Row:
    task: str
    condition: Condition
    item: int
    object_id: str
    filler: str
    label: float
    prompt: RenderedPrompt
    vetted: tuple[int, ...]
    trace: pd.Series | None = None
    cut: tuple[int, CutRule] | None = None
    frame: str = ""


def slot_fill(
    slot_name: str, connector: str, condition: Condition, *, filler: str, name: str, k: int | None
) -> tuple[dict[str, SlotBinding], dict[str, str]]:
    """Return the renderer bindings and texts that fill one slot under ``condition`` (the conditions above).

    A name condition puts ``name`` in place of the marker (its own text piece); every other condition binds ``filler``
    through ``connector`` with ``k`` tokens (None only for a suite that writes nothing, whose conditions are all text).
    """
    if condition in _TEXT_CONDITIONS:
        return {}, {slot_name: name}
    if k is None:
        raise ValueError(f"slot {slot_name!r}: condition {condition} writes, and no slot-token count was read")
    return {slot_name: SlotBinding(object_id=filler, connector=connector, k=k)}, {}


class _SlotRunner(ABC):
    """What every readout runner shares: spec checks, prompt construction, vetting, batching, the table.

    The prompt's slots all fill the one role of the spec's first slot; subclasses fix how many slots they take.
    """

    task_kind: TaskKind
    likelihood: bool
    answering: Answering | None = None
    """The Yes/No answer policy (``data.answers``) of a Yes/No readout; None for a likelihood readout."""
    reads_upstream: ClassVar[bool] = False
    """Whether the kind reads an upstream generation run (``SuiteSpec.upstream_kind``)."""
    leaves_out_unavailable: ClassVar[bool] = True
    """The kind leaves out the rows that write an object its inputs recorded as skipped (``SuiteContext.unavailable``);
    the ``suite`` stage refuses such objects for a kind without this attribute."""
    max_target_tokens: int | None = None

    def __init__(self, spec: SuiteSpec) -> None:
        if spec.frame is None and not spec.frame_cycle:
            raise ValueError(f"suite {spec.id!r}: {spec.kind} needs a frame (it places the task's question)")
        if spec.capability != "score":
            raise ValueError(f"suite {spec.id!r}: {spec.kind} needs the score capability, not {spec.capability!r}")
        if not spec.tasks:
            raise ValueError(f"suite {spec.id!r}: no tasks")
        names = [s.name for s in spec.slots]
        if len(set(names)) != len(names):
            raise ValueError(f"suite {spec.id!r}: slot names repeat: {names}")
        self.spec = spec
        self.slots = spec.slots
        self.role = spec.slots[0].role
        roles = sorted({s.role for s in spec.slots})
        if roles != [self.role]:
            raise ValueError(
                f"suite {spec.id!r}: slots fill roles {roles}; {spec.kind} fills one role (one object per prompt)"
            )
        conditions = [c.get(self.role) for c in spec.conditions]
        if any(c is None or set(cs) != {self.role} for c, cs in zip(conditions, spec.conditions, strict=True)):
            raise ValueError(f"suite {spec.id!r}: every condition combination must name exactly role {self.role!r}")
        if spec.upstream_kind is not None and not self.reads_upstream:
            raise ValueError(f"suite {spec.id!r}: {spec.kind} reads no upstream run, not {spec.upstream_kind!r}")
        for refused in (Condition.VALUE, Condition.PLACEHOLDER):
            if refused in conditions:
                raise ValueError(
                    f"suite {spec.id!r}: the {refused} condition is the cross-domain factorial's (cross_domain)"
                )
        self.conditions: tuple[Condition, ...] = tuple(c for c in conditions if c is not None)
        self._wt_slot = any(s.fill == "wt" for s in spec.slots)
        self.connectors = sorted({s.connector for s in spec.slots}) if spec.reads_connector() else []

    @abstractmethod
    def readout(self, renderer: Renderer) -> Readout:
        """Return the backend readout of this kind."""

    @abstractmethod
    def target(self, domain: Domain, task: TaskSpec, object_id: str) -> str | None:
        """Return the item's target text, or None if the kind scores no target (or the item has none)."""

    def fillers(self, items: ItemSet) -> tuple[str, ...] | None:
        """Return the mismatched object of every item, or None if the condition is not scored."""
        del items
        return None

    @abstractmethod
    def to_score(self, readout: Any) -> np.ndarray:
        """Turn one backend call's readout into per-prompt scores."""

    def answer_logp(self, readout: Any) -> np.ndarray | None:
        """Return a Yes/No readout's ``(N, 2)`` answer log-probabilities (yes, no) as read; None for a likelihood."""
        return None if self.answering is None else answer_log_probs(readout)

    def prefill(self) -> str:
        """Return the text the answer turn starts with ("" for none): the answer policy's prefix."""
        return self.answering.prefix if self.answering is not None else ""

    def trace_ids(
        self, trace: pd.Series, close: int | _NoClose, decode: Callable[[Sequence[int]], str]
    ) -> tuple[tuple[int, ...], tuple[int, CutRule] | None]:
        """Return the trace ids a continued trace keeps, and the recorded cut (None: none recorded).

        The base rule (:func:`_cut_trace`) for every kind; ``yes_no`` cuts further when its params say so.
        """
        del decode
        return _cut_trace(trace, close), None

    def after_trace(self, close: str | None) -> tuple[str, ...]:
        """Return the prefill parts after a continued trace, each encoded on its own (``render.TokenIds`` tail).

        Under an answer convention: the receiver's close (its thinking close, else its separator, ``close``), then the
        convention's prefix, as a training scaffold row ends (``data.scaffold``). Inline: the configured prefill with
        ``{close}`` replaced.

        Raises:
            ValueError: if the policy needs a close and the receiver has neither.
        """
        if self.answering is None or not self.answering.convention:
            return (_continuation(self.prefill(), close, self.spec.id),)
        if close is None:
            raise ValueError(f"suite {self.spec.id!r}: a trace is closed by the receiver's close; it has none")
        return tuple(p for p in (close, self.answering.prefix) if p)

    def elicited(self, upstream: Upstream, item: int) -> str:
        """Return the text the ``elicited_name`` condition shows for ``item`` (only ``yes_no`` reads names)."""
        del upstream, item
        raise ValueError(f"suite {self.spec.id!r}: {self.spec.kind} does not read elicited names")

    def run(self, ctx: SuiteContext, sources: Mapping[str, WriteSource], step: int | None) -> ArtifactRef:
        """Score every task × condition × item and commit the score table (``SuiteRunner.run``).

        One call of :meth:`score_groups` with one group, committed with its metadata.

        Raises:
            WrongHardwareError: if the backend runs on hardware other than ``ctx.device_class``.
            PromptVettingError: if a prompt's text depends on its item.
            KeyError: if ``sources`` lacks a slot's connector (when a condition writes), or a task is not in the
                domain's catalog.
            ValueError: if a task has the wrong kind for this suite, or the backend's plan is not ``ctx.plan``.
            UpstreamError: if the upstream run is missing, of another kind, lacks a row the suite reads, its prompt
                differs from the re-rendered one, or it has stop strings.
        """
        tables, meta = self.score_groups(ctx, [sources], step)
        return write_scores(ctx.artifacts, tables[0], meta)

    def score_groups(
        self, ctx: SuiteContext, groups: Sequence[Mapping[str, WriteSource]], step: int | None
    ) -> tuple[list[pd.DataFrame], dict[str, Any]]:
        """Score the suite's rows once per group of write sources, all groups in ONE backend call; commit nothing.

        The rows are rendered and vetted once; every group scores the same prompts, and each group's slots are bound
        to that group's sources only (in the call, group j > 0 names connector ``c`` as ``c#j``), so a row receives
        exactly its own group's writes, never another group's. The call's prompts are group-major (group 0's rows in
        :meth:`run`'s order, then group 1's, ...), and the plan partitions the whole call, so each group's ``batch``
        column indexes the call's one partition. Every group's sources must give the slots the same token count.

        Returns:
            Each group's score table (:meth:`run`'s columns), and the metadata a committed table of this call carries
            (the suite digest, kind, step and ``batch_plan``: the plan, and the call's prompt count and partition
            digest).

        Raises:
            WrongHardwareError: if the backend runs on hardware other than ``ctx.device_class``.
            PromptVettingError: if a prompt's text depends on its item.
            KeyError: if a group lacks a slot's connector, or a task is not in the domain's catalog.
            ValueError: if there is no group, the groups' slot-token counts differ, a task has the wrong kind for this
                suite, or the backend's plan is not ``ctx.plan``.
            UpstreamError: as :meth:`run`.
        """
        backend = ctx.backend
        if not isinstance(backend, ScoringBackend):
            raise TypeError(f"suite {self.spec.id!r}: backend {ctx.backend_name!r} cannot score")
        ctx.device_class.require(execution_devices(backend.execution()))
        if ctx.plan is None or backend.plan != ctx.plan:
            raise ValueError(f"suite {self.spec.id!r}: the backend's plan {backend.plan} is not the run's {ctx.plan}")
        if not groups:
            raise ValueError(f"suite {self.spec.id!r}: no group of write sources to score")
        connectors = self.connectors
        for j, sources in enumerate(groups):
            missing = sorted(set(connectors) - set(sources))
            if missing:
                raise KeyError(f"suite {self.spec.id!r}: group {j} has no write source for connector(s) {missing}")
        writes = [{c: sources[c] for c in connectors} for sources in groups]
        upstream = upstream_of(self.spec, ctx)
        close: tuple[int | _NoClose, str | None] | None = None
        if upstream is not None and upstream.run.suite.kind == "generation":
            close = _trace_close(ctx, upstream)
        items = ctx.items(self.spec.item_set)
        domain = self._domain(ctx, items)
        first = self._first_held(items, ctx.unavailable)
        rows = self._rows(ctx.renderer, domain, items, writes[0], upstream=upstream, close=close, first=first)
        rows, unavailable = _held_rows(rows, ctx.unavailable)
        self._vet(rows)
        ks = [{c: slot_tokens(w[c], first) for c in connectors} for w in writes]
        if any(k != ks[0] for k in ks):
            raise ValueError(f"suite {self.spec.id!r}: the groups' sources give the slots different token counts {ks}")
        names = [{c: c if j == 0 else f"{c}#{j}" for c in connectors} for j in range(len(groups))]
        prompts = [_bound(r.prompt, names[j]) for j in range(len(groups)) for r in rows]
        call: dict[str, WriteSource] = {names[j][c]: w[c] for j, w in enumerate(writes) for c in connectors}
        readout = self.readout(ctx.renderer)
        partition = plan_batches(planned_lengths(prompts, readout.kind), ctx.plan)
        batch = np.empty(len(prompts), dtype=np.int64)
        for b, members in enumerate(partition):
            batch[list(members)] = b
        scores, logp = np.empty(0), None if self.answering is None else np.empty((0, 2))
        if prompts:
            read = backend.score(prompts, call, readout)
            scores, logp = self.to_score(read), self.answer_logp(read)
        n = len(rows)
        cut = [slice(j * n, (j + 1) * n) for j in range(len(groups))]
        tables = [self._table(rows, [scores[c]], [batch[c]], step, None if logp is None else logp[c]) for c in cut]
        plan = {
            "plan": ctx.plan.identity(),  # an added field at its default left out (config.added_field)
            "calls": [{"prompts": len(prompts), "partition": partition_digest(partition)}],
        }
        meta = {**self._meta(step), "batch_plan": plan}
        if unavailable:
            meta["unavailable"] = unavailable
        return tables, meta

    def _domain(self, ctx: SuiteContext, items: ItemSet) -> Domain:
        name = items.roles[self.role]
        specs = [d for d in self.spec.domains if d.name == name]
        if len(specs) != 1:
            raise ValueError(f"suite {self.spec.id!r}: role {self.role!r} is domain {name!r}, cited {len(specs)}x")
        return ctx.domain(specs[0])

    def _rows(
        self,
        renderer: Renderer,
        domain: Domain,
        items: ItemSet,
        sources: Mapping[str, WriteSource],
        *,
        upstream: Upstream | None = None,
        close: tuple[int | _NoClose, str | None] | None = None,
        first: str | None = None,
    ) -> list[_Row]:
        """Return every task × condition × item row.

        Slot-token counts are read from ``first``, a held object (None: the first item's, for a caller with every
        object held).
        """
        objects = items.objects[self.role]
        ks = {s.name: slot_tokens(sources[s.connector], first or objects[0]) for s in self.slots if self.connectors}
        names = domain.objects()["name"]
        labels = domain.labels()
        fillers = self.fillers(items)
        catalog = dict(domain.tasks())
        declared = {t.id: t for t in self.spec.task_specs}
        if clash := sorted(set(declared) & set(catalog)):
            raise ValueError(f"suite {self.spec.id!r}: task_specs {clash} repeat the domain catalog's")
        catalog |= declared
        frames = self.spec.item_frames(items, domain.frames())
        tasks = []
        for task_id in self.spec.tasks:
            task = catalog[task_id]
            if task.kind != self.task_kind:
                raise ValueError(f"suite {self.spec.id!r}: task {task_id!r} is {task.kind}, not {self.task_kind}")
            label = labels[task.label] if task.kind == TaskKind.YES_NO else None
            targets = {obj: self.target(domain, task, obj) for obj in dict.fromkeys(objects)}
            index = pd.Index(objects)
            scope = pd.Series(True, index=index) if task_id in declared else domain.scope_mask(task, index)
            tasks.append((task, scope, label, targets))
        partners = _partner_fields(self.spec.id, items, self.role, self.conditions, self.slots)
        rows = []
        for i, obj in enumerate(objects):  # item-major: the original engine's sequence order
            for task, in_scope, label, targets in tasks:
                target = targets[obj]
                if not in_scope[obj] or (self.likelihood and target is None):
                    continue
                if label is not None and pd.isna(label.get(obj)):
                    raise ValueError(f"suite {self.spec.id!r}: task {task.id!r} has no label for {obj!r}")
                y = float("nan") if label is None else float(label[obj])
                for condition in self.conditions:
                    filler = fillers[i] if condition == Condition.MISMATCHED and fillers is not None else obj
                    name = self._name(condition, item=i, obj=obj, partners=partners, names=names, upstream=upstream)
                    if condition in FIELD_CONDITIONS:
                        filler = str(partners[str(condition)][i])
                    frame = frames[i]
                    asked = frame.ask(task.question)
                    render = partial(
                        self._render,
                        renderer,
                        asked if self.answering is None else self.answering.user(asked),
                        condition,
                        system=frame.system,
                        filler=filler,
                        name=name,
                        ks=ks,
                        target=target,
                        wt=(str(partners["wt"][i]), str(partners["name"][i])) if self._wt_slot else None,
                    )
                    trace, cut = None, None
                    if close is None or upstream is None:
                        prompt = render(prefill=self.prefill() or None)
                        vetted = prompt.input_ids
                    else:  # continue the upstream trace of this item, task and condition, by its token ids
                        vetted = render(prefill=None).input_ids
                        trace = upstream.row(i, self.role, str(condition), task.id)
                        upstream.check_prompt(trace, vetted)
                        ids, cut = self.trace_ids(trace, close[0], partial(generated_text, renderer.tokenizer))
                        prompt = render(prefill=(TokenIds(ids), *self.after_trace(close[1])))
                    rows.append(_Row(task.id, condition, i, obj, filler, y, prompt, vetted, trace, cut, frame.id))
        return rows

    def _first_held(self, items: ItemSet, unavailable: Mapping[str, str]) -> str:
        """Return the first item object the run holds (slot-token counts are read from it)."""
        held = [o for o in items.objects[self.role] if o not in unavailable]
        if not held:
            raise ValueError(f"suite {self.spec.id!r}: every item's object is declared unavailable")
        return held[0]

    def _name(
        self,
        condition: Condition,
        *,
        item: int,
        obj: str,
        partners: Mapping[str, tuple[object, ...]],
        names: pd.Series,
        upstream: Upstream | None,
    ) -> str:
        """Return the text a text condition shows for an item ("" for a condition that writes)."""
        if condition == Condition.NAME_SUBSTITUTION:
            return str(partners[str(condition)][item])
        if condition == Condition.NAME:
            return str(partners["name"][item]) if "name" in partners else str(names[obj])
        if condition == Condition.ELICITED_NAME and upstream is not None:
            return self.elicited(upstream, item)
        return ""

    def _render(
        self,
        renderer: Renderer,
        text: str,
        condition: Condition,
        *,
        system: str | None,
        filler: str,
        name: str,
        ks: Mapping[str, int],
        target: str | None,
        wt: tuple[str, str] | None = None,
        prefill: str | tuple[str | TokenIds, ...] | None = None,
    ) -> RenderedPrompt:
        bindings: dict[str, SlotBinding] = {}
        texts: dict[str, str] = {}
        for slot in self.slots:
            if slot.fill == "wt":  # the item's wild type, its name under a text condition
                if wt is None:
                    raise ValueError(
                        f"suite {self.spec.id!r}: wild-type slot {slot.name!r} without the item's wild type"
                    )
                fixed = _wt_condition(condition)
                b, t = slot_fill(slot.name, slot.connector, fixed, filler=wt[0], name=wt[1], k=ks.get(slot.name))
            else:
                b, t = slot_fill(slot.name, slot.connector, condition, filler=filler, name=name, k=ks.get(slot.name))
            bindings |= b
            texts |= t
        messages = (Message("system", system),) if system is not None else ()
        conversation = Conversation(
            messages=(*messages, Message("user", text)), prefill=prefill, thinking=self.spec.thinking
        )
        if target is None:
            return renderer.render(conversation, bindings, texts=texts)
        # tokens outside every span weigh 0 (Renderer.render), so the whole target is one full-weight span
        return renderer.render(
            conversation,
            bindings,
            texts=texts,
            target=target,
            spans=(WeightedSpan(0, len(target), 1.0),),
            max_target_tokens=self.max_target_tokens,
        )

    def _vet(self, rows: Sequence[_Row]) -> None:
        cycled = bool(self.spec.frame_cycle)  # then compared within a frame: the wording differs across units by design
        records = [
            PromptRecord(f"{r.task}@{r.frame}" if cycled else r.task, f"{self.role}={r.condition}", r.item, r.vetted)
            for r in rows
        ]
        exempt = [f"{self.role}={c}" for c in _TEXT_CONDITIONS]
        findings = vet_prompts(records, name_conditions=exempt)
        if findings:
            raise PromptVettingError(f"suite {self.spec.id!r}: " + "; ".join(findings))

    def _table(
        self,
        rows: Sequence[_Row],
        scores: Sequence[np.ndarray],
        batches: Sequence[np.ndarray],
        step: int | None,
        logp: np.ndarray | None = None,
    ) -> pd.DataFrame:
        role = self.role
        table = pd.DataFrame(
            {
                "suite": self.spec.digest(),
                "step": step,
                "task": [r.task for r in rows],
                "item": [r.item for r in rows],
                "score": np.concatenate(scores) if scores else np.empty(0),
                "label": [r.label for r in rows],
                "batch": np.concatenate(batches) if batches else np.empty(0, dtype=np.int64),
                f"object.{role}": [r.object_id for r in rows],
                f"filler.{role}": [r.filler for r in rows],
                f"condition.{role}": [str(r.condition) for r in rows],
                f"connector.{role}": None,
            }
        )
        if logp is not None:  # the two answer log-probabilities the score is the difference of (ANSWER_COLUMNS)
            table["logp_yes"], table["logp_no"] = logp[:, 0], logp[:, 1]
        if self.likelihood:
            table["n_target"] = [len(r.prompt.target.token_ids) if r.prompt.target else 0 for r in rows]
        traces = [r.trace for r in rows if r.trace is not None]
        if traces and len(traces) != len(rows):
            raise ValueError(f"suite {self.spec.id!r}: {len(traces)} of {len(rows)} rows continue a trace")
        if traces:  # the trace's length, completion and cap next to every post-reasoning score
            for column in ("text", "n_tokens", "finished", "max_new_tokens"):
                table[column] = [t[column] for t in traces]
        cuts = [r.cut for r in rows if r.cut is not None]
        if cuts and len(cuts) != len(rows):
            raise ValueError(f"suite {self.spec.id!r}: {len(cuts)} of {len(rows)} rows record a cut")
        if cuts:  # where each trace was cut before the forced answer (CUT_COLUMNS)
            table["n_tokens_kept"] = [k for k, _ in cuts]
            table["cut_rule"] = [rule for _, rule in cuts]
            table["n_dropped"] = table["n_tokens"] - table["n_tokens_kept"]
        return table

    def _meta(self, step: int | None) -> dict[str, Any]:
        return {"suite": self.spec.digest(), "kind": self.spec.kind, "step": step}


class _SingleSlotRunner(_SlotRunner):
    """A runner of exactly one slot the conditions fill (``yes_no``, ``caption_nll``).

    Besides it, wild-type slots (``SlotSpec.fill``) may hold the item's parent: the variant relational prompt.
    """

    def __init__(self, spec: SuiteSpec) -> None:
        varied = [s for s in spec.slots if s.fill is None]
        if len(varied) != 1:
            raise ValueError(f"suite {spec.id!r}: {spec.kind} scores exactly one slot, got {len(varied)}")
        super().__init__(spec)


@SUITE_KINDS.register("yes_no")
class YesNo(_SingleSlotRunner):
    """``yes_no``: log p(yes) − log p(no) of the next token per task × condition × item (:class:`YesNoParams`)."""

    task_kind = TaskKind.YES_NO
    likelihood = False
    reads_upstream = True

    def __init__(self, spec: SuiteSpec) -> None:
        super().__init__(spec)
        self.params = YesNoParams.model_validate(spec.params)
        p, where = self.params, f"suite {spec.id!r}"
        if _CLOSE in p.prefill and spec.upstream_kind != "generation":
            raise ValueError(f"{where}: the prefill's {_CLOSE} closes a trace, and the suite continues none")
        inline = p.prefill
        if spec.answer is not None and spec.upstream_kind == "generation":
            if p.prefill != _CLOSE:
                raise ValueError(f"{where}: after a trace an answer convention takes prefill {_CLOSE!r} alone")
            inline = ""  # the structural marker: where the trace is closed, no answer words
        self.answering = answering(spec.answer, yes=p.yes, no=p.no, prefix=inline, where=where)
        if spec.answer is not None and spec.upstream_kind == "generation":
            self.answering = replace(self.answering, instruction="")  # the trace's prompt has none (data.answers)
        self.label = self.prefill().replace(_CLOSE, "").strip()  # the answer prefix an answer_line cut looks for

    def elicited(self, upstream: Upstream, item: int) -> str:
        """Return the name the upstream ``name_guess`` run elicited for ``item`` from its own slot.

        Raises:
            UpstreamError: if the item has no ``own`` generation, or its answer is empty and no
                ``empty_elicited_name`` is stated.
        """
        answer = str(upstream.row(item, self.role, str(Condition.OWN))["answer"])
        if answer:
            return answer
        if self.params.empty_elicited_name is None:
            raise UpstreamError(
                f"suite {self.spec.id!r}: item {item}'s elicited name is empty; set empty_elicited_name"
            )
        return self.params.empty_elicited_name

    def readout(self, renderer: Renderer) -> Readout:
        """Return the next-token readout of the two answer words."""
        return answer_readout(renderer, self.answering)

    def trace_ids(
        self, trace: pd.Series, close: int | _NoClose, decode: Callable[[Sequence[int]], str]
    ) -> tuple[tuple[int, ...], tuple[int, CutRule] | None]:
        """Return the kept trace ids and, under ``params.cut``, the recorded cut ("Reading a trace").

        A capped trace that never closed its thinking is cut at its last sentence (``trace_cut.last_sentence_cut``;
        none found: kept whole, ``fallback``); a trace that closed it keeps its ids before the close, and a finished
        one all but its end id (rule ``none``), except under ``answer_line`` on a receiver without a thinking close,
        where it is cut before its own answer (``trace_cut.answer_line_cut``).
        """
        ids = _cut_trace(trace, close)
        closed = close is not _NO_CLOSE and close in tuple(trace["token_ids"])
        if bool(trace["finished"]) and close is _NO_CLOSE and self.params.cut == "answer_line":
            if not self.label:  # only a receiver without a thinking close cuts, so only it needs the prefix
                raise ValueError(f"suite {self.spec.id!r}: cut answer_line needs an answer prefix to cut before")
            k, rule = answer_line_cut(ids, decode, self.label)
            return ids[:k], (k, rule)
        if not bool(trace["finished"]) and not closed and self.params.cut is not None:
            k, rule = last_sentence_cut(ids, decode)
            return ids[:k], (k, rule)
        return ids, None if self.params.cut is None else (len(ids), "none")

    def target(self, domain: Domain, task: TaskSpec, object_id: str) -> str | None:
        """A Yes/No prompt has no target."""
        del domain, task, object_id
        return None

    def to_score(self, readout: Any) -> np.ndarray:
        """Return ``log p(yes) − log p(no)`` per prompt."""
        return log_odds(readout)


@SUITE_KINDS.register("caption_nll")
class CaptionNLL(_SingleSlotRunner):
    """``caption_nll``: mean NLL of the task's text field, cut, caption tokens only (:class:`CaptionParams`)."""

    task_kind = TaskKind.TEXT
    likelihood = True

    def __init__(self, spec: SuiteSpec) -> None:
        super().__init__(spec)
        self.params = CaptionParams.model_validate(spec.params)
        if spec.answer is not None:
            raise ValueError(
                f"suite {spec.id!r}: caption_nll asks no Yes/No question, so it takes no answer convention"
            )
        self.max_target_tokens = self.params.max_target_tokens
        if Condition.MISMATCHED in self.conditions and self.params.mismatch_seed is None:
            raise ValueError(f"suite {spec.id!r}: the mismatched condition needs params.mismatch_seed")

    def readout(self, renderer: Renderer) -> Readout:
        """Return the target-NLL readout."""
        del renderer
        return Readout(kind="target_nll")

    def target(self, domain: Domain, task: TaskSpec, object_id: str) -> str | None:
        """Return the object's text field (the renderer cuts it to ``max_target_tokens``), or None if it is empty."""
        return _text_target(domain, task, object_id)

    def fillers(self, items: ItemSet) -> tuple[str, ...] | None:
        """Return the deranged object of every item (None when ``mismatched`` is not scored)."""
        if Condition.MISMATCHED not in self.conditions or self.params.mismatch_seed is None:
            return None
        return _deranged(items.objects[self.role], self.params.mismatch_seed)

    def to_score(self, readout: Any) -> np.ndarray:
        """Return the mean target NLL per prompt."""
        return _target_nll(readout)


@SUITE_KINDS.register("composition")
class Composition(_SlotRunner):
    """``composition``: one object through two or more slots, each slot its own connector (:class:`CompositionParams`).

    The readout is Yes/No or target NLL (``params.readout``); rows have the single-slot shape (module docstring).
    """

    def __init__(self, spec: SuiteSpec) -> None:
        if len(spec.slots) < _MIN_COMPOSITION_SLOTS:
            raise ValueError(f"suite {spec.id!r}: composition fills two or more slots, got {len(spec.slots)}")
        super().__init__(spec)
        if Condition.NAME in self.conditions:
            raise ValueError(f"suite {spec.id!r}: the name condition fills one slot; composition fills several")
        self.params = CompositionParams.model_validate(spec.params)
        self.likelihood = self.params.readout == LIKELIHOOD_READOUT
        self.task_kind = TaskKind.TEXT if self.likelihood else TaskKind.YES_NO
        self.max_target_tokens = self.params.max_target_tokens
        if Condition.MISMATCHED in self.conditions and self.params.mismatch_seed is None:
            raise ValueError(f"suite {spec.id!r}: the mismatched condition needs params.mismatch_seed")
        where = f"suite {spec.id!r}"
        if self.likelihood:
            if spec.answer is not None:
                raise ValueError(f"{where}: readout target_nll takes no answer convention")
        else:
            self.answering = answering(spec.answer, yes=self.params.yes, no=self.params.no, where=where)

    def readout(self, renderer: Renderer) -> Readout:
        """Return the Yes/No next-token readout or the target-NLL readout."""
        if self.answering is None:
            return Readout(kind="target_nll")
        return answer_readout(renderer, self.answering)

    def target(self, domain: Domain, task: TaskSpec, object_id: str) -> str | None:
        """Return the object's text field for the target-NLL readout (None if empty); None for Yes/No."""
        return _text_target(domain, task, object_id) if self.likelihood else None

    def fillers(self, items: ItemSet) -> tuple[str, ...] | None:
        """Return the deranged object of every item (None when ``mismatched`` is not scored)."""
        if Condition.MISMATCHED not in self.conditions or self.params.mismatch_seed is None:
            return None
        return _deranged(items.objects[self.role], self.params.mismatch_seed)

    def to_score(self, readout: Any) -> np.ndarray:
        """Return ``log p(yes) − log p(no)`` or the mean target NLL per prompt."""
        return _target_nll(readout) if self.likelihood else log_odds(readout)


def answer_readout(renderer: Renderer, policy: Answering | None) -> Readout:
    """Return the next-token readout of the policy's two answers (each one token)."""
    if policy is None:  # unreachable: a Yes/No runner sets its policy at construction
        raise ValueError("a Yes/No readout without an answer policy")
    return Readout(kind="next_token", token_ids=policy.ids(renderer.token_id))


def yes_no_prompt(
    renderer: Renderer, policy: Answering, question: str, bindings: Mapping[str, SlotBinding]
) -> tuple[RenderedPrompt, Readout]:
    """Return the prompt of one Yes/No ``question`` asked alone, and the readout of its two answers.

    The prompt is a ``yes_no`` suite's for that wording: one user message, ``question`` with the policy's instruction
    after one space (``Answering.user``), no system message, the answer turn opened with its prefix and thinking off;
    the readout is the next token's (yes, no) ids (:func:`answer_readout`). Each ``<SLOT:name>`` of ``question`` takes
    its binding in ``bindings``.

    Raises:
        ValueError: if an answer text is not exactly one token of the receiver's tokenizer.
    """
    readout = answer_readout(renderer, policy)
    conversation = Conversation((Message("user", policy.user(question)),), prefill=policy.prefill(), thinking=False)
    return renderer.render(conversation, bindings), readout


def answer_log_probs(readout: Any) -> np.ndarray:
    """Return the ``(N, 2)`` log-probabilities of (yes, no) of a next-token readout, in float64."""
    out: np.ndarray = readout.detach().to("cpu").double().numpy()
    return out


def log_odds(readout: Any) -> np.ndarray:
    """Return ``log p(yes) − log p(no)`` per prompt from a next-token readout of (yes, no)."""
    lp = answer_log_probs(readout)
    out: np.ndarray = lp[:, 0] - lp[:, 1]
    return out


def _target_nll(readout: Any) -> np.ndarray:
    """Return the mean target NLL per prompt."""
    out: np.ndarray = readout.detach().to("cpu").double().numpy()
    return out


def _text_target(domain: Domain, task: TaskSpec, object_id: str) -> str | None:
    """Return the object's text field ``task.label`` (the renderer cuts it), or None if it is empty."""
    text = domain.fields()[task.label].get(object_id)
    if text is None or pd.isna(text) or not str(text):
        return None
    return str(text)


def _deranged(objects: Sequence[str], seed: int) -> tuple[str, ...]:
    """Return each item's object at its position in the seeded derangement of the item set."""
    perm = derangement(len(objects), seed)
    return tuple(objects[j] for j in perm)


def _trace_close(ctx: SuiteContext, upstream: Upstream) -> tuple[int | _NoClose, str | None]:
    """Return the receiver's thinking-close id and a prefill's ``{close}`` text, once the upstream run may be continued.

    A receiver without a thinking close (``ReceiverSpec.thinking_close`` None) gets :data:`_NO_CLOSE`: its traces are
    cut by :func:`_cut_trace`'s rule for a trace without a close. The text is the thinking close, else the separator,
    else None.

    Raises:
        UpstreamError: if the upstream run has stop strings (a finished trace would not end on its end id).
        ValueError: if the receiver's thinking close is not one token.
    """
    generation = upstream.run.suite.generation
    if generation is None or generation.stop:
        raise UpstreamError(f"upstream {upstream.run.suite.id!r} has stop strings; its traces cannot be continued")
    if ctx.receiver.thinking_close is None:
        return _NO_CLOSE, ctx.receiver.separator
    return ctx.renderer.token_id(ctx.receiver.thinking_close), ctx.receiver.thinking_close


def _continuation(prefill: str, close: str | None, suite: str) -> str:
    """Return the prefill after a trace, its ``{close}`` the receiver's thinking close or separator (``close``).

    Raises:
        ValueError: if the prefill holds ``{close}`` and the receiver has neither.
    """
    if _CLOSE in prefill and close is None:
        raise ValueError(f"suite {suite!r}: {_CLOSE} needs a receiver thinking close or separator; it has neither")
    return prefill if close is None else prefill.replace(_CLOSE, close)


def _cut_trace(trace: pd.Series, close: int | _NoClose) -> tuple[int, ...]:
    """Return a trace's ids before its first thinking close; with none, all but a finished trace's end id.

    With :data:`_NO_CLOSE` no close is looked for. A finished trace ends on its end-of-sequence id (the upstream run has
    no stop strings, :func:`_trace_close`), so that is the only id removed; a capped trace keeps every id.
    """
    ids: tuple[int, ...] = trace["token_ids"]
    if close is not _NO_CLOSE and close in ids:
        return ids[: ids.index(close)]
    return ids[:-1] if bool(trace["finished"]) else ids


def _held_rows(rows: list[_Row], unavailable: Mapping[str, str]) -> tuple[list[_Row], list[dict[str, Any]]]:
    """Return the rows whose prompt writes no unavailable object, and one record per row left out.

    Each record: the row's ``item``, ``task`` and ``condition``, and the unavailable ``objects`` it would write with
    their reasons (``SuiteRunConfig.unavailable``).
    """
    kept, out = [], []
    for row in rows:
        written = {s.binding.object_id for s in row.prompt.slots}
        if lost := sorted(written & set(unavailable)):
            out.append(
                {
                    "item": row.item,
                    "task": row.task,
                    "condition": str(row.condition),
                    "objects": {o: unavailable[o] for o in lost},
                }
            )
        else:
            kept.append(row)
    return kept, out


def _wt_condition(condition: Condition) -> Condition:
    """Return how a wild-type slot (``SlotSpec.fill``) is filled under ``condition``.

    Its name under a text condition, else the wild type through the connector (``placeholder`` is refused).
    """
    return Condition.NAME if condition in _TEXT_CONDITIONS else Condition.OWN


def _partner_fields(
    suite: str, items: ItemSet, role: str, conditions: Sequence[Condition], slots: Sequence[SlotSpec]
) -> dict[str, tuple[object, ...]]:
    """Return the item fields the variant conditions, ``name`` and a wild-type slot read, keyed by their prefix.

    Condition (or ``wt``, ``name``) → the per-item column ``<prefix>.<role>``; empty for a suite that uses none.
    ``name`` reads the item set's ``name.<role>`` whenever the set has one: a variant has no name in the domain.

    Raises:
        ValueError: if a variant condition or a wild-type slot is scored on an item set without its field.
    """
    wanted = {str(c) for c in conditions if c in FIELD_CONDITIONS}
    if any(s.fill == "wt" for s in slots):
        wanted |= {"wt", "name"}  # a wild type shows its name under a text condition
    if Condition.NAME in conditions and f"name.{role}" in items.fields:
        wanted.add("name")
    missing = sorted(w for w in wanted if f"{w}.{role}" not in items.fields)
    if missing:
        raise ValueError(f"suite {suite!r}: item set {items.name!r} has no fields {[f'{m}.{role}' for m in missing]}")
    return {w: items.fields[f"{w}.{role}"] for w in wanted}


def slot_tokens(source: WriteSource, object_id: str) -> int:
    """Return a connector's slot-token count k from its writes for one object (entries are (token, layer))."""
    tokens = {entry.token for entry in source.writes_for([object_id]).deltas}
    if not tokens:
        raise ValueError("the slot's write source has no entries, so its token count is unknown")
    return max(tokens) + 1


def _bound(prompt: RenderedPrompt, names: Mapping[str, str]) -> RenderedPrompt:
    """Return ``prompt`` with each slot bound to connector ``names[c]`` instead of ``c``."""
    slots = tuple(replace(o, binding=replace(o.binding, connector=names[o.binding.connector])) for o in prompt.slots)
    return replace(prompt, slots=slots)
