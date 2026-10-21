"""The generation runners: ``generation`` and ``name_guess`` (registered in ``SUITE_KINDS``).

Both generate from one slot (``SuiteSpec.slots`` has exactly one entry) for every (item, task, condition) of the item
set, item-major as the scoring runners order their rows, and commit a generations table
(``evaluation.generations``). The settings are the suite's ``generation`` (``receivers.base.GenerationParams``: the
paper's cap is 2,048 tokens, decoded greedily, as every original generation was), and the backend batches each call
under the run's generation plan (``SuiteRunConfig.generation_plan``), which the runner checks against
``backend.generation_plan``. There are no in-context examples; every prompt is vetted before anything is
generated (``evaluation.vetting``).

**Calls and resume.** Without ``GenerationPlan.prompts_per_call`` the run is one ``generate`` call. With it, the rows
are split in order into consecutive calls of that many prompts (:func:`generation_calls`, a function of the row count
and the plan alone), and each call's rows are committed as their own generations table (a *part*) as soon as the call
returns, remembered under a key of (suite digest, receiver, backend, plan, step, write banks, the call's span, origin)
(``ArtifactStore.remember``). A rerun of the same run after a kill reuses the parts already committed and generates
the rest, a call cut short whole: since the calls are fixed by the row count and the plan (never by what finished),
the final table is the one an uninterrupted run of the same plan gives, up to the backend's numerics (vLLM does not
promise bitwise-equal decoding of a repeated call; the fake backends do), and names its parts in order (``parts``).
The origin is the code (commit, dirty) and the execution (device class, ``backend.execution()`` but the weights
directory), as the injection scan's (``evaluation.layer_selection``), so a rerun on other code or hardware regenerates
every call and a dirty tree never reuses. Only parts whose writes are named by a digest are reused: every connector a
prompt can bind reads a write bank (``WriteSource.bank``; the ``kimi-vllm`` backend, which generates the paper's
traces); a live source regenerates.
A reused part is checked first (files against its manifest; the same suite, step, plan and span; the same task, item,
condition and prompt digest on every row).

**Prompts.** The user message is the suite's frame with the task's bare question in place (``Frame.ask``), as for
``yes_no``, with no answer instruction: a generation takes no answer convention (``data.answers``). The receiver's
thinking mode is the suite's (``SuiteSpec.thinking``; the paper's free reasoning runs with it on and no prefill).
Conditions of the slot's role: ``own`` and ``name``, filled as the scoring runners fill them
(``runners.slot_fill``). Items outside a task's scope (``Domain.scope_mask``) are not generated, so a reader scores
exactly the generated rows.

**Kinds.**

- ``generation`` (:class:`GenerationSuiteParams`): a trace per item, task and condition, e.g. the reasoning a scoring
  suite then continues (the post-reasoning readout, ``runners`` "Reading a trace"). Its ``answer`` column is empty,
  or, when the frame asks the trace to state a final answer, the answer it states (:class:`StatedAnswer`; "" for an
  unanswered, cut or unparsable trace, which stays a row).
- ``name_guess`` (:class:`NameGuessParams`): the receiver's name for the object, from a frame that asks for it (no
  task: rows carry task :data:`NAME_TASK`). The ``answer`` column holds the elicited name, read from the text by the
  original rule (the original ``collect`` and ``clean_name``, the 10a elicitation): the
  first line of the stripped text, stripped of spaces, periods, asterisks, quotes and backticks at both ends, cut
  before the first ``" ("`` and then the first ``" ["``, stripped, and truncated to ``max_answer_chars``. An empty
  name stays empty here; a reader decides what an empty name shows (``runners.YesNoParams.empty_elicited_name``).
  With ``NameGuessParams.prefill`` the answer turn starts with that text (e.g. a field label); the rule reads the
  generated continuation only, so a label the model repeats stays in the name.
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pandas as pd
from pydantic import Field

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.config import Config, canonical_sha256
from latent_interconnect.data.suites import Condition, SuiteSpec
from latent_interconnect.evaluation.generations import prompt_digest, read_generations, write_generations
from latent_interconnect.evaluation.runners import PromptVettingError, slot_fill, slot_tokens
from latent_interconnect.evaluation.suites import SUITE_KINDS, SuiteContext, execution_devices, upstream_of
from latent_interconnect.evaluation.vetting import PromptRecord, vet_prompts
from latent_interconnect.receivers.base import GenerationBackend, GenerationParams, GenerationPlan
from latent_interconnect.receivers.progress import Progress
from latent_interconnect.receivers.render import Conversation, Message, RenderedPrompt
from latent_interconnect.writes.sources import WriteSource

__all__ = [
    "NAME_TASK",
    "GenerationSuite",
    "GenerationSuiteParams",
    "NameGuess",
    "NameGuessParams",
    "StatedAnswer",
    "generation_calls",
    "normalize_name",
    "read_name",
    "stated_answer",
]

NAME_TASK = "name"
"""The task column of a ``name_guess`` row (the frame asks for the name; there is no catalog task)."""
_CONDITIONS = frozenset({Condition.OWN, Condition.NAME})
_GenRow = tuple[str, Condition, int, str, RenderedPrompt]
"""One prompt of a generation run: task, condition, item, object and the rendered prompt."""
_STRIP = " .*\"'`"
_CUTS = (" (", " [")


_GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
    "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "ο": "omicron", "π": "pi",
    "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi", "χ": "chi", "ψ": "psi",
    "ω": "omega",
}  # fmt: skip
"""Greek letters (after casefolding) → their names."""
_ROMAN = {
    "i": "1",
    "ii": "2",
    "iii": "3",
    "iv": "4",
    "v": "5",
    "vi": "6",
    "vii": "7",
    "viii": "8",
    "ix": "9",
    "x": "10",
}
"""Roman numerals read as a token's end."""
_ARTICLES = frozenset({"the", "a", "an"})
"""Words dropped from a name's start only: a trailing "a" is part of the name ("Cyclin A", "Coenzyme A")."""
_HEDGES = frozenset({"probable", "putative", "possible"})
"""Words dropped from either end of a name (the original rule's hedges)."""
_SEPARATORS = re.compile(r"[\s\-\u2010-\u2015\u2212_/\\.,;:'\"`\u2018-\u201f()\[\]{}<>]+")
"""Whitespace, every dash and hyphen, underscores, slashes, periods, commas, quotes, apostrophes and brackets."""


def normalize_name(name: str) -> str:
    """Return a name's comparison key: names equal up to case, spacing and punctuation share it.

    The rule, in order: Unicode NFKC and casefold; Greek letters to their names; split at separators (whitespace,
    dashes and hyphens of every kind, underscores, slashes, periods, commas, quotes, apostrophes, brackets); articles
    (:data:`_ARTICLES`) dropped from the start and hedges (:data:`_HEDGES`) from both ends; every token that is a
    Roman numeral up to ten becomes its digits ("kinase II" and "kinase 2" match); the tokens joined with nothing
    between them. The same rule applies to both sides of a comparison, so it is symmetric. Deterministic; "" for a
    name with no letters or digits.
    """
    text = unicodedata.normalize("NFKC", name).casefold()
    text = "".join(f" {_GREEK[c]} " if c in _GREEK else c for c in text)
    tokens = [t for t in _SEPARATORS.split(text) if t]
    while tokens and tokens[0] in _ARTICLES | _HEDGES:
        tokens.pop(0)
    while tokens and tokens[-1] in _HEDGES:
        tokens.pop()
    return "".join(_ROMAN.get(t, t) for t in tokens)


class StatedAnswer(Config):
    """How a trace states its own final answer, for the trace's ``answer`` column and its support counts.

    A trace states ``yes`` (or ``no``) when its last line that starts with ``prefix`` (after leading spaces and
    asterisks) continues with that word (case-insensitive; spaces, asterisks and quotes between) and does not offer
    the other as an alternative: a restated format line ("Answer: Yes or Answer: No") states nothing. On a receiver
    with a thinking close (Kimi's ``</think>``) only the text after the first close is read, and a trace that never
    closed its
    thinking states nothing: an ``Answer:`` line inside the thinking is not a final answer. A trace that states
    nothing (``""``: unanswered, cut, unclosed or unparsable) stays a row (:func:`stated_answer`). This is the trace's
    own generated answer; a post-reasoning readout's forced answer is a separate score, never merged with it.

    Attributes:
        prefix: the answer line's label (the instruction's, e.g. ``"Answer:"``).
        yes: the word for yes.
        no: the word for no.
    """

    prefix: str = Field(min_length=1)
    yes: str = Field(min_length=1)
    no: str = Field(min_length=1)


def stated_answer(text: str, rule: StatedAnswer, close: str | None = None) -> str:
    """Return the final answer a trace states under ``rule`` (``rule.yes``, ``rule.no``, or ``""`` for none).

    Args:
        text: the trace's decoded text.
        rule: how the answer is stated.
        close: the receiver's thinking close (``ReceiverSpec.thinking_close``), if it has one.
    """
    if close is not None:
        if close not in text:
            return ""
        text = text.split(close, 1)[1]
    lines = [line.strip().lstrip("*").strip() for line in text.splitlines()]
    stated = [line[len(rule.prefix) :] for line in lines if line.startswith(rule.prefix)]
    if not stated:
        return ""
    if _offers_both(stated[-1], rule):
        return ""  # the format restated ("Answer: Yes or Answer: No"), not answered
    word = re.match(r"[\s*\"'`]*([A-Za-z]+)", stated[-1])
    found = word.group(1).casefold() if word else ""
    return {rule.yes.casefold(): rule.yes, rule.no.casefold(): rule.no}.get(found, "")


def _offers_both(line: str, rule: StatedAnswer) -> bool:
    """Whether an answer line offers both words as alternatives (``Yes or No``, ``Yes / Answer: No``, either order).

    Only that counts: an answer that merely mentions the other word ("Yes, there is no doubt", "Yes (no signal
    peptide)") states its first word.
    """
    gap = r"[\s*\"'`]*(?:or|/)[\s*\"'`]*(?:" + re.escape(rule.prefix) + r")?[\s*\"'`]*"
    words = [(rule.yes, rule.no), (rule.no, rule.yes)]
    return any(re.search(rf"\b{re.escape(a)}\b{gap}\b{re.escape(b)}\b", line, re.I) for a, b in words)


class GenerationSuiteParams(Config):
    """Parameters of the ``generation`` kind (``SuiteSpec.params``).

    Attributes:
        prefill: text the answer turn starts with ("" for none; the paper's free reasoning has none).
        stated_answer: how the trace states its final answer, when its frame asks for one (the trace's ``answer``
            column: :func:`stated_answer`); None: no answer column (``""``), as every trace before it. A suite without
            the key keeps its digest (``params`` is stored as written).
    """

    prefill: str = ""
    stated_answer: StatedAnswer | None = None


class NameGuessParams(Config):
    """Parameters of the ``name_guess`` kind (``SuiteSpec.params``).

    Attributes:
        max_answer_chars: the elicited name is truncated to this many characters (the original rule: 200).
        prefill: text the answer turn starts with, in the assistant continuation after the template's generation
            prompt (``Conversation.prefill``), e.g. ``"Protein name: "``; "" for none, the original elicitation. The
            backend returns only the new tokens, so the prefill is neither in the text nor in the token budget, and
            :func:`read_name` reads the continuation alone. A suite without the key keeps its digest (``params``
            is stored as written).
    """

    max_answer_chars: int = Field(ge=1)
    prefill: str = ""


def generation_calls(n: int, plan: GenerationPlan) -> tuple[tuple[int, int], ...]:
    """Return a run's calls as (start, stop) spans of its ``n`` rows: consecutive, ``plan.prompts_per_call`` each.

    One span of all rows when the plan sets none (no rows: one empty span); the last may be shorter. A function of
    ``n`` and the plan alone (never of what a run finished), so a resumed run makes the same calls.
    """
    size = plan.prompts_per_call or max(n, 1)
    return tuple((start, min(start + size, n)) for start in range(0, max(n, 1), size))


def read_name(text: str, max_chars: int) -> str:
    """Return the name a ``name_guess`` generation states, by the original rule (module docstring); "" if none."""
    name = text.strip().split("\n", 1)[0].strip(_STRIP)
    for cut in _CUTS:
        name = name.split(cut, 1)[0]
    return name.strip()[:max_chars]


class _GenerationRunner(ABC):
    """What ``generation`` and ``name_guess`` share: spec checks, prompts, vetting, the call and the table."""

    asks_tasks: ClassVar[bool]
    """Whether the suite's tasks place a question in the frame (``generation``) or the frame asks alone."""

    def __init__(self, spec: SuiteSpec) -> None:
        if spec.capability != "generate" or spec.generation is None:
            raise ValueError(f"suite {spec.id!r}: {spec.kind} needs the generate capability and generation settings")
        if len(spec.slots) != 1:
            raise ValueError(f"suite {spec.id!r}: {spec.kind} generates from exactly one slot, got {len(spec.slots)}")
        if spec.frame is None:
            raise ValueError(f"suite {spec.id!r}: {spec.kind} needs a frame")
        if bool(spec.tasks) != self.asks_tasks:
            raise ValueError(f"suite {spec.id!r}: {spec.kind} takes {'tasks' if self.asks_tasks else 'no tasks'}")
        if spec.answer is not None:  # a convention's instruction asks for a direct answer (data.answers)
            raise ValueError(f"suite {spec.id!r}: {spec.kind} generates, so it takes no answer convention")
        self.spec, self.slot = spec, spec.slots[0]
        self.close: str | None = None
        """The receiver's thinking close, set when a run starts (a trace's stated answer follows it)."""
        self.generation: GenerationParams = spec.generation
        conditions = [c.get(self.slot.role) for c in spec.conditions]
        if any(c is None or set(cs) != {self.slot.role} for c, cs in zip(conditions, spec.conditions, strict=True)):
            raise ValueError(f"suite {spec.id!r}: every condition must name exactly role {self.slot.role!r}")
        self.conditions: tuple[Condition, ...] = tuple(c for c in conditions if c is not None)
        unsupported = sorted(set(self.conditions) - _CONDITIONS)
        if unsupported:
            raise ValueError(f"suite {spec.id!r}: {spec.kind} generates under {sorted(_CONDITIONS)}, not {unsupported}")

    @abstractmethod
    def prefill(self) -> str:
        """Return the text the answer turn starts with ("" for none)."""

    @abstractmethod
    def answer(self, text: str) -> str:
        """Return the kind's reading of a generated text (the ``answer`` column; "" for none)."""

    def run(self, ctx: SuiteContext, sources: Mapping[str, WriteSource], step: int | None) -> ArtifactRef:
        """Generate every task × condition × item and commit the generations table (``SuiteRunner.run``).

        In the plan's calls (:func:`generation_calls`), each committed and resumed when the plan sets
        ``prompts_per_call`` (module docstring).

        Raises:
            WrongHardwareError: if the backend runs on hardware other than ``ctx.device_class``.
            PromptVettingError: if a prompt's text depends on its item.
            TypeError: if the backend cannot generate.
            ValueError: if the backend's generation plan is not ``ctx.generation_plan``, it returns another number
                of generations than prompts, or a remembered part is not this run's.
            UpstreamError: if the context carries an upstream run (a generation suite reads none).
        """
        backend = ctx.backend
        self.close = ctx.receiver.thinking_close
        if not isinstance(backend, GenerationBackend):
            raise TypeError(f"suite {self.spec.id!r}: backend {ctx.backend_name!r} cannot generate")
        ctx.device_class.require(execution_devices(backend.execution()))
        plan = ctx.generation_plan
        if plan is None or backend.generation_plan != plan:
            raise ValueError(
                f"suite {self.spec.id!r}: the backend's generation plan {backend.generation_plan} is not the run's "
                f"{plan}"
            )
        upstream_of(self.spec, ctx)
        # the slot's connector only when a condition writes (``own``); a name-only suite reads none
        writes = {c: sources[c] for c in (self.slot.connector,) if self.spec.reads_connector()}
        rows = self._rows(ctx, writes)
        records = [PromptRecord(t, f"{self.slot.role}={c}", i, p.input_ids) for t, c, i, _, p in rows]
        findings = vet_prompts(records, name_conditions=[f"{self.slot.role}={Condition.NAME}"])
        if findings:
            raise PromptVettingError(f"suite {self.spec.id!r}: " + "; ".join(findings))
        meta = {
            "suite": self.spec.digest(),
            "kind": self.spec.kind,
            "step": step,
            "generation": self.generation.model_dump(mode="json"),
            "generation_plan": plan.identity(),  # an added field at its default left out, as in the digest
        }
        if plan.prompts_per_call is None:
            return write_generations(ctx.artifacts, self._table(backend, rows, writes, step), meta)
        calls = generation_calls(len(rows), plan)
        progress = Progress(f"suite {self.spec.id} generation calls", len(rows))  # across calls (logging only)
        parts: list[ArtifactRef] = []
        for span in calls:
            parts.append(self._part(ctx, backend, rows, writes, meta=meta, span=span))
            progress.advance(span[1] - span[0])
        table = pd.concat([read_generations(ctx.artifacts, r) for r in parts], ignore_index=True)
        return write_generations(ctx.artifacts, table, meta | {"parts": [r.sha256 for r in parts]})

    def _part(
        self,
        ctx: SuiteContext,
        backend: GenerationBackend,
        rows: Sequence[_GenRow],
        writes: Mapping[str, WriteSource],
        *,
        meta: Mapping[str, Any],
        span: tuple[int, int],
    ) -> ArtifactRef:
        """Return the committed part of one call: remembered and checked, else generated, committed and remembered."""
        commit, dirty = ctx.code
        banks = {c: w.bank.sha256 for c, w in sorted(writes.items()) if w.bank is not None}
        reuse = not dirty and len(banks) == len(writes)  # every connector's writes named by a digest
        execution = {k: v for k, v in backend.execution().items() if k != "weights"}
        origin = {"commit": commit, "dirty": dirty, "device_class": ctx.device_class, "execution": execution}
        step = meta["step"]
        identity = [meta["suite"], ctx.receiver.digest(), ctx.backend_name, meta["generation_plan"], step, banks]
        key = "generation-call-" + canonical_sha256([*identity, list(span), origin])
        part_meta = {**meta, "call": list(span)}
        ref = ctx.artifacts.committed(key) if reuse else None
        if ref is not None:
            self._check_part(ctx, ref, rows[span[0] : span[1]], part_meta)
            return ref
        ref = write_generations(ctx.artifacts, self._table(backend, rows[span[0] : span[1]], writes, step), part_meta)
        return ctx.artifacts.remember(key, ref) if reuse else ref  # a duplicate writer's part, if it was first

    def _table(
        self, backend: GenerationBackend, rows: Sequence[_GenRow], writes: Mapping[str, WriteSource], step: int | None
    ) -> pd.DataFrame:
        """Generate ``rows`` in one call and return their generations table."""
        prompts = [p for *_, p in rows]
        generations = backend.generate(prompts, writes, self.generation) if prompts else []
        if len(generations) != len(prompts):
            raise ValueError(f"suite {self.spec.id!r}: {len(generations)} generations for {len(prompts)} prompts")
        role = self.slot.role
        return pd.DataFrame(
            {
                "suite": self.spec.digest(),
                "step": step,
                "task": [t for t, *_ in rows],
                "item": [i for _, _, i, _, _ in rows],
                "prompt": [prompt_digest(p.input_ids) for p in prompts],
                "text": [g.text for g in generations],
                "token_ids": [list(g.token_ids) for g in generations],
                "n_tokens": [g.n_tokens for g in generations],
                "finished": [g.finished for g in generations],
                "max_new_tokens": self.generation.max_new_tokens,
                "answer": [self.answer(g.text) for g in generations],
                f"object.{role}": [obj for _, _, _, obj, _ in rows],
                f"filler.{role}": [obj for _, _, _, obj, _ in rows],
                f"condition.{role}": [str(c) for _, c, *_ in rows],
                f"connector.{role}": None,
            }
        )

    def _check_part(
        self, ctx: SuiteContext, ref: ArtifactRef, rows: Sequence[_GenRow], meta: Mapping[str, Any]
    ) -> None:
        """Refuse a remembered part that is not this run's intact table of ``rows`` (module docstring)."""
        ctx.artifacts.verify(ref)
        table = read_generations(ctx.artifacts, ref)
        found = {k: ctx.artifacts.manifest(ref).meta.get(k) for k in meta}
        expected = [(t, i, str(c), prompt_digest(p.input_ids)) for t, c, i, _, p in rows]
        role = self.slot.role
        got = list(zip(table["task"], table["item"], table[f"condition.{role}"], table["prompt"], strict=True))
        if found != dict(meta) or got != expected:
            raise ValueError(
                f"suite {self.spec.id!r}: remembered part {ref.short} (call {meta['call']}) is not this run's; "
                "delete its pointer"
            )

    def _rows(self, ctx: SuiteContext, writes: Mapping[str, WriteSource]) -> list[_GenRow]:
        items = ctx.items(self.spec.item_set)
        name = items.roles[self.slot.role]
        specs = [d for d in self.spec.domains if d.name == name]
        if len(specs) != 1:
            raise ValueError(f"suite {self.spec.id!r}: role {self.slot.role!r} is domain {name!r}, cited {len(specs)}x")
        domain = ctx.domain(specs[0])
        objects = items.objects[self.slot.role]
        frame = self.spec.frame_in(domain.frames())
        names = domain.objects()["name"]
        catalog = domain.tasks()
        index = pd.Index(objects)
        tasks = [(catalog[t].id, catalog[t].question, domain.scope_mask(catalog[t], index)) for t in self.spec.tasks]
        if not self.asks_tasks:
            tasks = [(NAME_TASK, "", pd.Series(True, index=index))]
        k = slot_tokens(writes[self.slot.connector], objects[0]) if writes else None
        system = (Message("system", frame.system),) if frame.system is not None else ()
        rows = []
        for i, obj in enumerate(objects):  # item-major, as the scoring runners
            for task, question, in_scope in tasks:
                if not in_scope[obj]:
                    continue
                for condition in self.conditions:
                    bindings, texts = slot_fill(
                        self.slot.name, self.slot.connector, condition, filler=obj, name=str(names[obj]), k=k
                    )
                    conversation = Conversation(
                        messages=(*system, Message("user", frame.ask(question))),
                        prefill=self.prefill() or None,
                        thinking=self.spec.thinking,
                    )
                    rows.append((task, condition, i, obj, ctx.renderer.render(conversation, bindings, texts=texts)))
        return rows


@SUITE_KINDS.register("generation")
class GenerationSuite(_GenerationRunner):
    """``generation``: a trace per item × task × condition (:class:`GenerationSuiteParams`)."""

    asks_tasks = True

    def __init__(self, spec: SuiteSpec) -> None:
        super().__init__(spec)
        self.params = GenerationSuiteParams.model_validate(spec.params)

    def prefill(self) -> str:
        """Return the configured prefill."""
        return self.params.prefill

    def answer(self, text: str) -> str:
        """Return the trace's stated final answer (:class:`StatedAnswer`), or ``""`` when the suite declares none.

        A reader continues the trace and reads its own answer in either case.
        """
        rule = self.params.stated_answer
        return "" if rule is None else stated_answer(text, rule, self.close)


@SUITE_KINDS.register("name_guess")
class NameGuess(_GenerationRunner):
    """``name_guess``: the receiver's name for each item's object (:class:`NameGuessParams`; :func:`read_name`)."""

    asks_tasks = False

    def __init__(self, spec: SuiteSpec) -> None:
        super().__init__(spec)
        self.params = NameGuessParams.model_validate(spec.params)

    def prefill(self) -> str:
        """Return the configured prefill ("" by default: the original elicitation has none)."""
        return self.params.prefill

    def answer(self, text: str) -> str:
        """Return the elicited name (:func:`read_name`)."""
        return read_name(text, self.params.max_answer_chars)
