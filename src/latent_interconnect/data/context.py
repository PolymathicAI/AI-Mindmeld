"""In-context examples (preambles): a row transform of the ``text`` and ``yes_no`` streams, training only.

On a share of a stream's rows (:attr:`ContextSpec.rate`, originally ``--p-context``), a system turn is added before the
user turn: one role sentence, a header, and one or two examples in one example format, chosen by a weighted
*structure*. Each example is ``named`` (the object's name as text) or ``inject`` (the object at its own slot,
``<SLOT:c1>``, ``<SLOT:c2>``, … in order of appearance, through the run's connector, so the query's slot comes
last), and is the ``same`` question as the row's or an ``other`` one. The loss stays on the row's own answer.

- **same** (text): an entry whose capped answer to the row's field differs from the row's answer, asked with the
  task's example question; (Yes/No): a trained keyword drawn uniformly and a label drawn 50/50 independently of the
  row's, an entry with that label, one of the keyword's training questions, and its answer word. An entry whose label
  is unknown (a null: no label, or a cell outside the keyword's scope, ``data.streams.ScopedDomain``) has neither
  label, so it is never a Yes example and never a No example. When none is found the example becomes an ``other`` one.
- **other**: an entry and one of its tasks whose field is not the row's, with the capped answer.
- Every example: never the row's object, never in its cluster, distinct from the other example.

**The pool** (as the original preamble builder drew it): the training objects visited in a weighted random order without
replacement (``ObjectPool.weighted_order``: the ``cluster_sqrt`` weights, seeded by :attr:`PoolSpec.seed`), keeping an
object whose name is specific (not :attr:`PoolSpec.generic_names`, at most ``max_name_chars``) and that has at least
``min_tasks`` capped answers among the pool tasks, until ``size`` are kept. An answer is the screened training text,
capped at ``answer_sentences`` sentences and ``answer_tokens`` receiver tokens without cutting a sentence (a field in
``unsplit_fields`` is never sentence-split: whole, or its first ``"; "`` item), or no answer. Token counts come from the
renderer (``Renderer.target_token_count``). Yes/No answers are the domain's labels of the trained keywords.

Evaluation never uses in-context examples; this transform exists only in training streams.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.config import Config
from latent_interconnect.data.grammar import Grammar
from latent_interconnect.data.streams import AnswerLine, Example, ObjectPool, Supervision, population, worded
from latent_interconnect.data.training_text import TrainingText
from latent_interconnect.data.wording import transform
from latent_interconnect.domains.base import TaskKind, TaskScope
from latent_interconnect.domains.views import CatalogView, DomainView
from latent_interconnect.receivers.render import Message

__all__ = ["Context", "ContextSpec", "ExampleFormat", "PoolSpec", "Structure", "context_supervision", "sentences"]

_SENTENCE = re.compile(r"(?<=[a-z0-9\)\]])\.\s+(?=[A-Z(])")
"""The sentence boundary of the original pool builder (the original prompt grammar's ``SENT``)."""


def sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, each ending with one period (the original prompt_grammar sentences)."""
    return [s.strip().rstrip(".") + "." for s in _SENTENCE.split(text.strip()) if s.strip()]


class PoolSpec(Config):
    """How the example pool is drawn from the training objects.

    Attributes:
        size: the entries kept.
        seed: the seed of the visiting order.
        generic_names: a regular expression (case-insensitive) of names too generic to show; None: every name is
            specific enough (the molecule pool's rule).
        max_name_chars: the longest name shown.
        min_tasks: the fewest distinct answer sources an entry has among the pool tasks (a task, or the group
            ``sources`` puts it in).
        sources: pool task → its answer source, for tasks that count as one (the molecule rule: describe_def counts as
            describe, functional-group classes and parent family as one "classes" source); a task not listed is its own.
        answer_sentences: the most sentences of an answer.
        answer_tokens: the most receiver tokens of an answer.
        unsplit_fields: fields never sentence-split (reactions, EC numbers).
    """

    size: int = Field(ge=1)
    seed: int
    generic_names: str | None = None
    max_name_chars: int = Field(ge=1)
    min_tasks: int = Field(ge=1)
    answer_sentences: int = Field(ge=1)
    answer_tokens: int = Field(ge=1)
    unsplit_fields: tuple[str, ...] = ()
    sources: dict[str, str] = Field(default_factory=dict)


class ExampleFormat(Config):
    """One example line format and the separator between examples.

    Attributes:
        text: the line, with ``{object}`` (the example's slot, or a space and its name), ``{question}`` and
            ``{answer}``.
        separator: between two examples.
    """

    text: str
    separator: str

    @model_validator(mode="after")
    def _check(self) -> ExampleFormat:
        missing = [k for k in ("{object}", "{question}", "{answer}") if k not in self.text]
        if missing:
            raise ValueError(f"example format {self.text!r} lacks {missing}")
        return self


class Structure(Config):
    """A weighted preamble structure.

    Attributes:
        id: its name.
        weight: its draw weight.
        examples: per example, ``(inject | named, same | other)``.
    """

    id: str
    weight: float = Field(gt=0.0)
    examples: tuple[tuple[Literal["inject", "named"], Literal["same", "other"]], ...] = Field(min_length=1)


class ContextSpec(Config):
    """In-context examples, as data (``configs/streams/context-*.yaml``; the training wording only).

    Attributes:
        rate: the share of the stream's rows that get examples.
        roles: the role sentences.
        headers: per header, its text before two examples and before one.
        formats: the example formats.
        structures: the weighted structures.
        example_questions: task id → its example question (text tasks; a ``same`` example of a text row asks its own).
        pool_tasks: the text tasks an entry is tagged with (``other`` examples and the pool rule).
        name_field: the screened training-text field a named example shows (anything shown is training text, so a
            name is read from the screened set, never from the domain's object table).
        pool: the pool rule.
        answer_lines: field → how its items are worded as an example's answer, for item-list fields (the molecule
            fields; ``data.streams.AnswerLine``): the same part the text stream's ``TextParams.answer_lines`` cites, so
            examples and rows word an answer alike, on every stream that carries these examples.
    """

    rate: float = Field(gt=0.0, le=1.0)
    roles: tuple[str, ...] = Field(min_length=1)
    headers: tuple[tuple[str, str], ...] = Field(min_length=1)
    formats: tuple[ExampleFormat, ...] = Field(min_length=1)
    structures: tuple[Structure, ...] = Field(min_length=1)
    example_questions: dict[str, str]
    pool_tasks: tuple[str, ...] = Field(min_length=1)
    name_field: str
    pool: PoolSpec
    answer_lines: dict[str, AnswerLine] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ContextSpec:
        missing = sorted(set(self.pool_tasks) - set(self.example_questions))
        if missing:
            raise ValueError(f"pool tasks {missing} have no example question")
        return self


def context_supervision(spec: ContextSpec, domain: CatalogView) -> Supervision:
    """Return what the examples put into a prompt: the pool tasks, their fields, and the fixed wording."""
    catalog = domain.tasks()
    tasks = set(spec.pool_tasks) | set(spec.example_questions)
    missing = sorted(tasks - set(catalog))
    if missing:
        raise ValueError(f"context tasks the domain catalog lacks: {missing}")
    fields = frozenset({catalog[t].label for t in tasks} | {spec.name_field})
    wording = [*spec.roles, *(h for pair in spec.headers for h in pair), *(f.text for f in spec.formats)]
    wording += [h for line in spec.answer_lines.values() for h in (line.one, line.many)]
    return Supervision(frozenset(tasks), frozenset(), fields, (*wording, *spec.example_questions.values()))


@dataclass(frozen=True)
class _Entry:
    object_id: str
    name: str
    cluster: object
    answers: Mapping[str, str]  # field → capped answer
    tasks: tuple[str, ...]  # pool tasks it answers


class Context:
    """Applies a :class:`ContextSpec` to a stream's examples.

    Args:
        spec: the examples' config.
        domain: names, clusters, the task catalog and the Yes/No labels.
        text: the screened training text (answers and the pool's objects).
        count: the renderer's token count.
        yes_no: for a ``yes_no`` stream, its questions per trained Yes/No task and its answer words as an example line
            shows them (under an answer convention its prefix and answer text, as the assistant turn reads them);
            None for text.
        grammar: the stream's prompt grammar, whose answer formats (with the pathway part limit) a same-question example
            of a formatted grammar row is put in; None formats with ``data.wording.transform``.
        cue: the answer convention's prefix of a ``yes_no`` stream (``data.answers``), which leads every text answer
            of an example line as it leads the Yes/No ones (one space between), so a line format carries no answer
            label of its own; "" for none.

    Raises:
        ValueError: if a task is not a trained task of the domain, or the pool is smaller than ``pool.size``.
    """

    def __init__(
        self,
        spec: ContextSpec,
        domain: DomainView,
        text: TrainingText,
        count: Callable[[str], int],
        yes_no: tuple[Mapping[str, Sequence[str]], tuple[str, str]] | None,
        *,
        grammar: Grammar | None = None,
        cue: str = "",
    ) -> None:
        catalog = domain.tasks()
        self.lines = spec.answer_lines
        self.formatted = grammar.transform_draw if grammar is not None else transform
        self.cue = cue
        bad = sorted(
            t
            for t in set(spec.pool_tasks) | set(spec.example_questions)
            if t not in catalog or catalog[t].kind != TaskKind.TEXT or catalog[t].scope != TaskScope.TRAINED
        )
        if bad:
            raise ValueError(f"context: tasks {bad} are not trained text tasks of the domain")
        self.spec, self.count, self.yes_no = spec, count, yes_no
        self.field = {t: catalog[t].label for t in set(spec.pool_tasks) | set(spec.example_questions)}
        stray = sorted(set(spec.answer_lines) - set(self.field.values()))
        if stray:
            raise ValueError(f"context: answer lines for fields no pool task answers with: {stray}")
        self.clusters = domain.clusters()
        self._labels: dict[str, frozenset[str]] = {}
        self._unknown: dict[str, frozenset[str]] = {}  # neither Yes nor No: never drawn for either
        if yes_no is not None:
            for t in yes_no[0]:
                column = domain.labels([catalog[t].label])[catalog[t].label].reindex(list(text.object_ids))
                self._labels[t] = frozenset(o for o, v in column.items() if not pd.isna(v) and bool(v))
                self._unknown[t] = frozenset(o for o, v in column.items() if pd.isna(v))
        self.pool = self._build_pool(text, population(domain, text.object_ids))
        weights = np.array([s.weight for s in spec.structures])
        self._structure_p = weights / weights.sum()

    def _build_pool(self, text: TrainingText, objects: Sequence[str]) -> list[_Entry]:
        rule = self.spec.pool
        generic = re.compile(rule.generic_names, re.IGNORECASE) if rule.generic_names else None
        order = ObjectPool(objects, "cluster_sqrt", self.clusters).weighted_order(np.random.default_rng(rule.seed))
        pool = []
        for o in order:
            fields = text.fields.get(o, {})
            name = fields.get(self.spec.name_field, "").strip()
            if not name or (generic is not None and generic.search(name)) or len(name) > rule.max_name_chars:
                continue
            answers = {}
            for f in sorted({self.field[t] for t in self.spec.pool_tasks}):
                capped = self._cap(worded(self.lines, f, fields.get(f) or ""), f)
                if capped is not None:
                    answers[f] = capped
            tasks = tuple(t for t in self.spec.pool_tasks if self.field[t] in answers)
            if len({rule.sources.get(t, t) for t in tasks}) >= rule.min_tasks:
                pool.append(_Entry(o, name, self.clusters[o], answers, tasks))
            if len(pool) == rule.size:
                return pool
        raise ValueError(f"context pool: {len(pool)} qualifying training objects, {rule.size} needed")

    def _cap(self, text: str | None, field: str) -> str | None:
        """The original cap: sentences and tokens, never cutting a sentence, or None."""
        if not text or not text.strip():
            return None
        rule, text = self.spec.pool, text.strip()
        if field in rule.unsplit_fields:
            first = text.split("; ")[0].strip()
            if self.count(text) <= rule.answer_tokens:
                return text
            return first if self.count(first) <= rule.answer_tokens else None
        parts = sentences(text)
        if len(parts) <= rule.answer_sentences and self.count(text) <= rule.answer_tokens:
            return text
        for k in range(rule.answer_sentences, 0, -1):
            capped = " ".join(parts[:k])
            if capped and self.count(capped) <= rule.answer_tokens:
                return capped
        return None

    def apply(self, example: Example, rng: np.random.Generator) -> Example:
        """Return the example with the system turn of its examples (the row's own answer unchanged)."""
        query = example.objects[example.subject or ""]
        used: set[str] = set()

        def ok(e: _Entry) -> bool:
            return e.object_id != query and e.object_id not in used and e.cluster != self.clusters.get(query)

        structure = self.spec.structures[int(rng.choice(len(self.spec.structures), p=self._structure_p))]
        chosen = []
        for kind, role in structure.examples:
            got = self._same(example, rng, ok) if role == "same" else None
            got = got or self._other(example, rng, ok)
            used.add(got[0].object_id)
            chosen.append((kind, *got))
        chosen = [chosen[i] for i in rng.permutation(len(chosen))]
        role_text = self.spec.roles[int(rng.integers(len(self.spec.roles)))]
        header = self.spec.headers[int(rng.integers(len(self.spec.headers)))][0 if len(chosen) > 1 else 1]
        fmt = self.spec.formats[int(rng.integers(len(self.spec.formats)))]
        objects, lines = dict(example.objects), list[str]()
        for kind, entry, question, answer in chosen:
            if kind == "inject":
                slot = f"c{sum(k == 'inject' for k, *_ in chosen[: len(lines) + 1])}"
                objects[slot] = entry.object_id
                shown = f"<SLOT:{slot}>"
            else:
                shown = " " + entry.name
            lines.append(fmt.text.format(object=shown, question=question, answer=answer))
        system = Message("system", f"{role_text} {header}\n" + fmt.separator.join(lines))
        rest = tuple(m for m in example.conversation.messages if m.role != "system")  # it replaces a grammar role
        conversation = dataclasses.replace(example.conversation, messages=(system, *rest))
        return dataclasses.replace(example, conversation=conversation, objects=objects)

    def _same(
        self, example: Example, rng: np.random.Generator, ok: Callable[[_Entry], bool]
    ) -> tuple[_Entry, str, str] | None:
        if self.yes_no is not None:
            questions, (yes, no) = self.yes_no
            keys = tuple(questions)
            task = keys[int(rng.integers(len(keys)))]
            want = bool(rng.random() < 0.5)  # noqa: PLR2004  (independent of the row's label)
            true, unknown = self._labels[task], self._unknown[task]
            labelled = [e for e in self.pool if e.object_id not in unknown and (e.object_id in true) is want]
            entry = _draw(labelled, rng, ok)
            if entry is None:
                return None
            asked = questions[task]
            return entry, asked[int(rng.integers(len(asked)))], yes if want else no
        task = example.question or ""
        if task not in self.field:
            return None
        field = self.field[task]
        own = " ".join(example.target.split())

        def answer(e: _Entry) -> str | None:
            text = e.answers.get(field)
            if text is None or example.context_format is None:
                return text
            formatted = self.formatted(example.context_format, text, task)
            return formatted[0] if formatted else None

        memo: dict[str, str] = {}

        def good(e: _Entry) -> bool:
            if not ok(e):
                return False
            a = answer(e)
            if a is None:
                return False
            memo[e.object_id] = a
            a = " ".join(a.split())
            return not (a == own or own.startswith(a) or a.startswith(own))

        entry = _draw(self.pool, rng, good)
        if entry is None:
            return None
        question = (example.context_question or self.spec.example_questions[task]) + example.context_suffix
        return entry, question, memo[entry.object_id]

    def _other(
        self, example: Example, rng: np.random.Generator, ok: Callable[[_Entry], bool]
    ) -> tuple[_Entry, str, str]:
        own_field = self.field.get(example.question or "") if self.yes_no is None else None
        for _ in range(_OTHER_TRIES):
            entry = self.pool[int(rng.integers(len(self.pool)))]
            tasks = [t for t in entry.tasks if self.field[t] != own_field]
            if tasks and ok(entry):
                t = tasks[int(rng.integers(len(tasks)))]
                text = entry.answers[self.field[t]]
                return entry, self.spec.example_questions[t], f"{self.cue} {text}" if self.cue else text
        raise RuntimeError(f"context: no 'other' example found in {_OTHER_TRIES} draws")


_OTHER_TRIES = 200
"""Draws for an ``other`` example before giving up (the original _other_example)."""
_DRAW_TRIES = 64
"""Rejection draws before a full scan (the original TemplateData _draw)."""


def _draw(candidates: Sequence[_Entry], rng: np.random.Generator, ok: Callable[[_Entry], bool]) -> _Entry | None:
    """A uniform candidate passing ``ok`` (rejection sampling, then a full scan), or None."""
    if not candidates:
        return None
    for _ in range(_DRAW_TRIES):
        e = candidates[int(rng.integers(len(candidates)))]
        if ok(e):
            return e
    rest = [e for e in candidates if ok(e)]
    return rest[int(rng.integers(len(rest)))] if rest else None
