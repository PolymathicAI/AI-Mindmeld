"""The single-object question streams: ``text`` (a trained text field as the answer) and ``yes_no`` (a trained label).

Both ask about one training object bound to the template's slot, with the training wording from the stream's own
config (``configs/streams/``; the domain catalog's questions are the evaluation wording): a template is a
user message with one ``<SLOT:x>`` marker (and, for ``yes_no``, a ``{question}`` marker), citing a trained task of
the domain catalog. Both may carry a reasoning scaffold (``data.scaffold``) and in-context examples
(``data.context``), which the sampler applies to their rows. Objects come from an :class:`~latent_interconnect.data.
streams.ObjectPool` over the screened training text's objects (uniform or ``cluster_sqrt``).

- ``text`` (originally ``draw``): an object first, among those whose screened text answers at least one template's task
  (the field ``TaskSpec.label``); then a template uniformly from all the stream's templates, drawn again until the
  object answers it (at most :data:`TEMPLATE_TRIES` draws, then a new object: the original dropped that row), so given
  the object a template is uniform among those it answers and a task is weighted by its template count; the answer is
  that text, capped at ``max_target_tokens``, weighted by the task's loss curve.
- ``yes_no`` (originally ``pick_yesno``): a task uniformly, then Yes or No with equal probability, then an
  object with that label; a template uniformly and one of the task's questions uniformly; the answer is the configured
  answer word. An object whose label is unknown (a null: no label, or a cell outside the task's scope,
  ``data.streams.ScopedDomain``) is in neither pool of that task; a task left without a Yes or without a No object is
  refused, and the pools of a task that lost objects are recorded with the run (``YesNoStream.derived``). Under an
  answer convention (``YesNoParams.answer``, ``data.answers``) the template or grammar row carries only the task and
  object wording: the convention's instruction follows it, its prefix opens the assistant turn
  (``Conversation.prefill``: context, not a target) and its answer text is the target, so the row ends on the tokens a
  scored prompt of the receiver ends on.
"""

from __future__ import annotations

import re
from typing import ClassVar, Protocol

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.config import Config, added_field
from latent_interconnect.data.answers import AnswerConvention, Answering, answering
from latent_interconnect.data.context import ContextSpec, context_supervision
from latent_interconnect.data.grammar import SLOT, FormatCoverage, Grammar, GrammarSpec, UserTurn, fit_coverage
from latent_interconnect.data.scaffold import ScaffoldSpec, scaffold_supervision
from latent_interconnect.data.streams import (
    STREAMS,
    AnswerLine,
    Example,
    LossCurve,
    ObjectPool,
    Sampling,
    SegmentBoost,
    Supervision,
    population,
    worded,
)
from latent_interconnect.data.training_text import TrainingText
from latent_interconnect.domains.base import QUESTION_MARKER, TaskKind, TaskScope, TaskSpec
from latent_interconnect.domains.views import CatalogView, DomainView
from latent_interconnect.receivers.render import SLOT_PATTERN, Conversation, Message, WeightedSpan

__all__ = ["TEMPLATE_TRIES", "Template", "TextParams", "TextStream", "YesNoAnswers", "YesNoParams", "YesNoStream"]

TEMPLATE_TRIES = 20
"""Template draws per object before the text stream draws another object (the original ``draw``: 20)."""
SLOT_NAME = SLOT.removeprefix("<SLOT:").removesuffix(">")
"""The slot of a grammar-worded row (``data.grammar.SLOT``)."""


class TokenOffsets(Protocol):
    """The renderer's token offsets of a target part (``Renderer.target_token_offsets``)."""

    def __call__(self, text: str) -> tuple[tuple[int, int], ...]:
        """Return each token's ``[start, end)`` in ``text``."""
        ...


class Template(Config):
    """One training prompt template.

    Attributes:
        id: the template's id (the original template ids).
        task: the trained task it asks (``yes_no``: None, the task comes with the question).
        user: the user message, with one ``<SLOT:x>`` marker (``yes_no``: and one ``{question}`` marker).
    """

    id: str
    task: str | None = None
    user: str

    @model_validator(mode="after")
    def _check(self) -> Template:
        slots = re.findall(SLOT_PATTERN, self.user)
        if len(slots) != 1:
            raise ValueError(f"template {self.id!r} has slot markers {slots}; a single-object template needs one")
        return self

    @property
    def slot(self) -> str:
        """The name of the template's slot marker."""
        found: list[str] = re.findall(SLOT_PATTERN, self.user)
        return found[0]


class TextParams(Config):
    """Parameters of the ``text`` stream.

    Attributes:
        templates: the training templates, each citing a trained text task (whose ``label`` names its field).
        sampling: object draws (``data.streams.Sampling``).
        max_target_tokens: the answer's token cap (originally ``--cap-tokens``).
        weights: task id → its loss curve (every task of a template has one).
        segment_boost: task id → the measured head ratio ``r_f`` of its ';'-segment boost; the boost's floor
            is the task's curve floor. Template rows answer with the field text verbatim, so every row of a listed
            task is boosted.
        segment_boost_formats: the answer formats the boost applies to (the field text verbatim, a
            grammar row's ``none`` / ``verbatim`` / ``source``, and the catalytic ``reaction_sentence``); template rows
            are always boosted.
        scaffold: the reasoning scaffold attempted on this stream's eligible rows, or None.
        context: the in-context examples given on a share of this stream's rows, or None.
        grammar: the prompt grammar and target wording, or None.
        answer_lines: field → how its items are worded as the answer, for item-list fields (the molecule fields;
            ``data.streams.AnswerLine``); the in-context examples cite the same part (``ContextSpec.answer_lines``).
    """

    templates: tuple[Template, ...] = Field(min_length=1)
    sampling: Sampling
    max_target_tokens: int = Field(ge=1)
    weights: dict[str, LossCurve]
    segment_boost: dict[str, float] = Field(default_factory=dict)
    segment_boost_formats: tuple[str, ...] = ()
    scaffold: ScaffoldSpec | None = None
    context: ContextSpec | None = None
    grammar: GrammarSpec | None = None
    answer_lines: dict[str, AnswerLine] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> TextParams:
        missing = sorted({t.task or "" for t in self.templates} - set(self.weights))
        if missing:
            raise ValueError(f"text stream: templates without a task or a loss curve: {missing}")
        stray = sorted(set(self.segment_boost) - {t.task or "" for t in self.templates})
        if stray:
            raise ValueError(f"text stream: segment boosts for tasks without a template: {stray}")
        return self


class YesNoAnswers(Config):
    """The answer words of the ``yes_no`` stream.

    Attributes:
        yes: the answer for a true label.
        no: the answer for a false label.
    """

    yes: str
    no: str


class YesNoParams(Config):
    """Parameters of the ``yes_no`` stream.

    Attributes:
        templates: the training frames, each with a ``{question}`` marker (their ``task`` is unset).
        questions: trained Yes/No task id → its training questions.
        sampling: object draws (``data.streams.Sampling``).
        answers: the inline answer words; None under ``answer``.
        weights: the answer's loss curve.
        scaffold: the reasoning scaffold attempted on this stream's eligible rows, or None.
        context: the in-context examples given on a share of this stream's rows, or None.
        grammar: the prompt grammar, or None. Under ``answer`` its rows end on the question, the convention's
            instruction replacing the grammar's own (``GrammarSpec.yes_no_formats``).
        answer: the receiver's answer convention (``data.answers``); the templates then carry no answer
            instruction. None: the inline policy (``answers``, the templates' and grammar's own instructions).
    """

    templates: tuple[Template, ...] = Field(min_length=1)
    questions: dict[str, tuple[str, ...]] = Field(min_length=1)
    sampling: Sampling
    answers: YesNoAnswers | None = added_field(default=None)
    weights: LossCurve
    scaffold: ScaffoldSpec | None = None
    context: ContextSpec | None = None
    grammar: GrammarSpec | None = None
    answer: AnswerConvention | None = added_field(default=None)

    def policy(self) -> Answering:
        """Return the stream's answer policy (``data.answers.answering``: exactly one of ``answer``, ``answers``)."""
        words = self.answers
        yes, no = (words.yes, words.no) if words is not None else (None, None)
        return answering(self.answer, yes=yes, no=no, where="yes_no stream")

    @model_validator(mode="after")
    def _check(self) -> YesNoParams:
        bad = [t.id for t in self.templates if t.task is not None or t.user.count(QUESTION_MARKER) != 1]
        empty = sorted(k for k, qs in self.questions.items() if not qs)
        if bad or empty:
            raise ValueError(
                f"yes_no stream: templates {bad} need one {QUESTION_MARKER} and no task; {empty} no question"
            )
        self.policy()
        return self


def _trained(domain: CatalogView, ids: list[str], kind: TaskKind) -> dict[str, TaskSpec]:
    """The catalog's tasks ``ids``, checked to be trained tasks of ``kind`` (never a zero-shot one)."""
    catalog = domain.tasks()
    unknown = sorted(t for t in set(ids) if t not in catalog)
    if unknown:
        raise ValueError(f"tasks {unknown} are not in the domain catalog")
    bad = sorted(t for t in set(ids) if catalog[t].kind != kind or catalog[t].scope != TaskScope.TRAINED)
    if bad:
        raise ValueError(f"tasks {bad} are not trained {kind.value} tasks: a stream trains trained tasks only")
    return {t: catalog[t] for t in sorted(set(ids))}


def _grammar_slots(params: TextParams | YesNoParams) -> None:
    """With a grammar: every template binds the grammar's slot, and every text task has grammar cores and formats."""
    if params.grammar is None:
        return
    other = [t.id for t in params.templates if t.slot != SLOT_NAME]
    if other:
        raise ValueError(f"with a grammar, templates bind <SLOT:{SLOT_NAME}>; these do not: {other}")
    worded = set(params.grammar.cores) & set(params.grammar.formats)
    missing = sorted({t.task for t in params.templates if t.task} - worded)
    if missing:
        raise ValueError(f"the grammar has no cores or formats for tasks {missing}")


def _supervision(
    domain: CatalogView, tasks: dict[str, TaskSpec], texts: list[str], params: TextParams | YesNoParams
) -> Supervision:
    """A question stream's declaration: its tasks, their labels or fields, its wording and its transforms'."""
    yes_no = {t.label for t in tasks.values() if t.kind == TaskKind.YES_NO}
    fields = {t.label for t in tasks.values()} - yes_no
    scaffold, context, grammar = params.scaffold, params.context, params.grammar
    wording = Grammar(grammar).texts() if grammar is not None else ()
    out = Supervision(frozenset(tasks), frozenset(yes_no), frozenset(fields), (*texts, *wording))
    if scaffold is not None:
        out = out | scaffold_supervision(scaffold, domain)
    if context is not None:
        out = out | context_supervision(context, domain)
    return out


@STREAMS.register("text")
class TextStream:
    """Trained text fields as answers to their tasks' templates (see the module docstring)."""

    Params: ClassVar[type[Config]] = TextParams

    def __init__(self, domain: DomainView, params: Config, text: TrainingText) -> None:
        if not isinstance(params, TextParams):
            raise TypeError(f"expected TextParams, got {type(params).__name__}")
        self.params, self.scaffold, self.context = params, params.scaffold, params.context
        self.context_yes_no = None
        self.tasks = _trained(domain, [t.task or "" for t in params.templates], TaskKind.TEXT)
        self.text = text
        self._supervision = self.declare(domain, params)

        def answers(o: str, t: Template) -> bool:
            return bool(text.fields.get(o, {}).get(self.tasks[t.task or ""].label, "").strip())

        stray = sorted(set(params.answer_lines) - {t.label for t in self.tasks.values()})
        if stray:
            raise ValueError(f"text stream: answer lines for fields none of its tasks answers with: {stray}")
        objects = population(domain, text.object_ids)
        self.answerable = {o: tuple(t for t in params.templates if answers(o, t)) for o in objects}
        having = [o for o, ts in self.answerable.items() if ts]
        if not having:
            raise ValueError(f"text stream: no training object answers a template of tasks {list(self.tasks)}")
        self.pool = ObjectPool(objects, params.sampling, domain.clusters()).subset(having)
        _grammar_slots(params)
        self.formats: FormatCoverage | None = None
        self.grammar: Grammar | None = None
        if params.grammar is not None:  # the formats the rule enables on this training text
            texts = {
                t: [self._answer(o, spec.label) for o in objects]
                for t, spec in self.tasks.items()
                if t in params.grammar.formats
            }
            self.formats = fit_coverage(params.grammar, texts)
            self.grammar = Grammar(params.grammar, self.formats.enabled)
        self.offsets: TokenOffsets | None = None
        partners = set(params.grammar.partner_templates) if params.grammar else set()
        self.plain = {
            t: tuple((x.id, x.user) for x in params.templates if x.task == t and x.id not in partners)
            for t in self.tasks
        }

    def _answer(self, oid: str, field: str) -> str:
        """An object's field as the answer text: worded as its answer line if it is an item list."""
        return worded(self.params.answer_lines, field, self.text.fields.get(oid, {}).get(field, ""))

    def bind(self, offsets: TokenOffsets) -> None:
        """Bind the renderer's token offsets (the grammar checks and cuts answers in receiver tokens)."""
        self.offsets = offsets

    def derived(self) -> dict[str, object]:
        """Return what this stream derived from its training text (the grammar formats it enabled)."""
        if self.formats is None:
            return {}
        return {"grammar_formats": {"coverage": self.formats.coverage, "enabled": self.formats.enabled}}

    def _tokens(self) -> TokenOffsets:
        if self.offsets is None:
            raise RuntimeError("text stream with a grammar: not bound to a renderer (Sampler binds it)")
        return self.offsets

    def _capped(self, text: str) -> str:
        """The text as trained: cut at the answer cap (the original ``_capped_text``)."""
        offsets, cap = self._tokens()(text), self.params.max_target_tokens
        return text if len(offsets) <= cap else text[: offsets[cap - 1][1]]

    def _fits(self, text: str) -> bool:
        return len(self._tokens()(text)) <= self.params.max_target_tokens

    def sample(self, rng: np.random.Generator, n: int) -> list[Example]:
        """Draw ``n`` examples."""
        out: list[Example] = []
        while len(out) < n:
            worded = self.grammar is not None and rng.random() < self.grammar.spec.rate
            oid = self.pool.draw(rng)
            example = self._grammar_row(oid, rng) if worded else None
            example = example or self._template_row(oid, rng)
            if example is not None:
                out.append(example)
        return out

    def _template(self, oid: str, rng: np.random.Generator) -> Template | None:
        """A template uniformly, redrawn until the object answers it (the original ``draw``), or None."""
        templates = self.params.templates
        for _ in range(TEMPLATE_TRIES):
            template = templates[int(rng.integers(len(templates)))]
            if template in self.answerable[oid]:
                return template
        return None

    def _template_row(self, oid: str, rng: np.random.Generator) -> Example | None:
        template = self._template(oid, rng)
        if template is None:
            return None
        task = template.task or ""
        answer = self._answer(oid, self.tasks[task].label)
        cap = self.params.max_target_tokens
        curve = self.params.weights[task]
        user, context_question, suffix = template.user, None, ""
        if self.grammar is not None:  # target wording on template rows
            plain = self.plain[task]
            user, context_question, suffix = self.grammar.template_wording(
                rng, task, template.id, template.user, answer, self._capped, plain
            )
        r_f = self.params.segment_boost.get(task)
        boost = None if r_f is None else SegmentBoost(r_f=r_f, floor=curve.floor)
        return Example(
            Conversation((Message("user", user),)),
            {template.slot: oid},
            answer,
            (curve.span(answer, cap),),
            "text",
            cap,
            (),
            task,
            template.slot,
            boost,
            context_question,
            suffix,
        )

    def _grammar_row(self, oid: str, rng: np.random.Generator) -> Example | None:
        """A grammar-worded row (the original ``gdraw``), or None.

        A template picks the task and the grammar words it; None if nothing applies in :data:`TEMPLATE_TRIES` draws
        (the row is then a template row, as in the original code).
        """
        grammar = self.grammar
        if grammar is None:
            return None
        for _ in range(TEMPLATE_TRIES):
            template = self._template(oid, rng)
            if template is None:
                return None
            task = template.task or ""
            field_text = self._answer(oid, self.tasks[task].label)
            turn = grammar.row(rng, task, field_text, self._capped, self._fits)
            if turn is None:
                continue
            role = grammar.system_role(rng)
            messages = ((Message("system", role),) if role else ()) + (Message("user", turn.user),)
            weights = self._grammar_weights(task, turn)
            span = WeightedSpan(0, len(turn.answer), weights[0], tuple(weights))
            context_format = None if turn.answer_format == "source" else turn.answer_format
            return Example(
                Conversation(messages),
                {SLOT_NAME: oid},
                turn.answer,
                (span,),
                "text",
                self.params.max_target_tokens,
                (),
                task,
                SLOT_NAME,
                None,
                turn.context_question,
                turn.context_suffix,
                context_format,
            )
        return None

    def _grammar_weights(self, task: str, turn: UserTurn) -> list[float]:
        """Per-token weights of the capped answer (the original ``LossWeights._weights``, template mask, segment boost).

        A token overlapping announced-format text gets the floor; the others the task's curve, counted over them only; a
        boosted format raises each fact token after a fact token holding ';'.
        """
        curve = self.params.weights[task]
        offsets = self._tokens()(turn.answer)[: self.params.max_target_tokens]
        template = [any(a < max(e, s + 1) and b > s for a, b in turn.template_spans) for s, e in offsets]
        facts = [i for i, t in enumerate(template) if not t]
        weights = [curve.floor * curve.multiplier] * len(offsets)
        along = curve.weights(len(facts)) if facts else ()
        for k, i in enumerate(facts):
            weights[i] = along[k]
        r_f = self.params.segment_boost.get(task)
        if r_f is not None and turn.answer_format in self.params.segment_boost_formats:
            head = SegmentBoost(r_f=r_f, floor=curve.floor).head
            for j in range(1, len(facts)):
                s, e = offsets[facts[j - 1]]
                if ";" in turn.answer[s:e]:
                    weights[facts[j]] = max(weights[facts[j]], head)
        return weights

    @classmethod
    def declare(cls, catalog: CatalogView, params: Config) -> Supervision:
        """Return the tasks, their fields, the templates and the transforms' declarations."""
        if not isinstance(params, TextParams):
            raise TypeError(f"expected TextParams, got {type(params).__name__}")
        tasks = _trained(catalog, [t.task or "" for t in params.templates], TaskKind.TEXT)
        texts = [t.user for t in params.templates]
        texts += [h for line in params.answer_lines.values() for h in (line.one, line.many)]  # the answer lines' heads
        return _supervision(catalog, tasks, texts, params)

    def supervision(self) -> Supervision:
        """Return :meth:`declare`'s declaration."""
        return self._supervision


@STREAMS.register("yes_no")
class YesNoStream:
    """Trained labels as Yes/No answers, balanced per question (see the module docstring)."""

    Params: ClassVar[type[Config]] = YesNoParams

    def __init__(self, domain: DomainView, params: Config, text: TrainingText) -> None:
        if not isinstance(params, YesNoParams):
            raise TypeError(f"expected YesNoParams, got {type(params).__name__}")
        self.params, self.scaffold, self.context = params, params.scaffold, params.context
        self.grammar = Grammar(params.grammar) if params.grammar is not None else None
        _grammar_slots(params)
        self.tasks = _trained(domain, list(params.questions), TaskKind.YES_NO)
        self.order = tuple(params.questions)
        self.policy = params.policy()
        # an in-context example shows its answer as the assistant turn reads it: the prefix, then the answer text
        shown = (self.policy.prefix + self.policy.yes, self.policy.prefix + self.policy.no)
        self.context_yes_no = (params.questions, shown)
        self.context_cue = self.policy.prefix
        self.answer_instruction = self.policy.instruction  # a direct-answer instruction: a scaffolded row drops it
        self._supervision = self.declare(domain, params)
        ids = list(population(domain, text.object_ids))
        pool = ObjectPool(ids, params.sampling, domain.clusters())
        self.pools: dict[tuple[str, bool], ObjectPool] = {}
        self.unknown: dict[str, int] = {}  # task → its objects in neither pool (no label, or outside its scope)
        for t, task in self.tasks.items():
            column = domain.labels([task.label])[task.label].reindex(ids)
            self.unknown[t] = int(column.isna().sum())
            for value in (True, False):
                having = [o for o, v in zip(ids, column, strict=True) if not pd.isna(v) and bool(v) is value]
                if not having:
                    raise ValueError(
                        f"yes_no stream: no training object has label {value} for task {t!r} "
                        f"({self.unknown[t]} of {len(ids)} have no label or are outside its scope)"
                    )
                self.pools[(t, value)] = pool.subset(having)

    def derived(self) -> dict[str, object]:
        """Return what this stream derived from its training text: the pools of the tasks that lost objects.

        Per task with an object in neither pool (no label, or a cell outside the task's scope): its Yes objects, its
        No objects and those left out, so a pool thinned by label scope is on the run's record. Nothing when every
        object has a label in scope for every task.
        """
        pools = {
            t: {"yes": len(self.pools[(t, True)].ids), "no": len(self.pools[(t, False)].ids), "unknown": n}
            for t, n in sorted(self.unknown.items())
            if n
        }
        return {"label_pools": pools} if pools else {}

    def sample(self, rng: np.random.Generator, n: int) -> list[Example]:
        """Draw ``n`` examples; a share :attr:`GrammarSpec.rate` worded by the grammar."""
        out = []
        for _ in range(n):
            worded = self.grammar is not None and rng.random() < self.grammar.spec.rate
            t = self.order[int(rng.integers(len(self.order)))]
            value = bool(rng.random() < 0.5)  # noqa: PLR2004  (the balanced Yes/No draw)
            oid = self.pools[(t, value)].draw(rng)
            questions = self.params.questions[t]
            question = questions[int(rng.integers(len(questions)))]
            answer = self.policy.word(value)
            span = self.params.weights.span(answer, len(answer))
            if self.grammar is not None and worded:
                role = self.grammar.system_role(rng)
                system = (Message("system", role),) if role else ()
                user = self.grammar.user(rng, t, question=question, instructed=not self.policy.convention)
                conversation = Conversation((*system, Message("user", self.policy.user(user))), self.policy.prefill())
                slot = SLOT_NAME
            else:
                template = self.params.templates[int(rng.integers(len(self.params.templates)))]
                user = self.policy.user(template.user.replace(QUESTION_MARKER, question))
                conversation = Conversation((Message("user", user),), self.policy.prefill())
                slot = template.slot
            out.append(Example(conversation, {slot: oid}, answer, (span,), "yes_no", None, (), t, slot))
        return out

    @classmethod
    def declare(cls, catalog: CatalogView, params: Config) -> Supervision:
        """Return the tasks, their labels, every template with every question, the answers and the transforms'."""
        if not isinstance(params, YesNoParams):
            raise TypeError(f"expected YesNoParams, got {type(params).__name__}")
        tasks = _trained(catalog, list(params.questions), TaskKind.YES_NO)
        asked = [q for qs in params.questions.values() for q in qs]
        policy = params.policy()
        texts = [policy.user(t.user.replace(QUESTION_MARKER, q)) for t in params.templates for q in asked]
        texts += [t for t in (policy.prefix, policy.yes, policy.no) if t]
        return _supervision(catalog, tasks, texts, params)

    def supervision(self) -> Supervision:
        """Return :meth:`declare`'s declaration."""
        return self._supervision
