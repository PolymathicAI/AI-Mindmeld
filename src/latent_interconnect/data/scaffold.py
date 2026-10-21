r"""The property-first reasoning scaffold: a row transform on the ``yes_no`` and ``text`` streams.

A scaffold is attempted on a share (:attr:`ScaffoldSpec.rate`) of a stream's *eligible* rows: rows whose question is
listed in :attr:`ScaffoldSpec.questions`. An attempted row becomes a reasoning row:

- **prompt**: the row's own conversation with one of the training instructions appended to its last user message (slot
  positions are unchanged: every slot lies before it), in thinking mode on a receiver with a thinking close. A row
  under an answer convention first loses the convention's instruction, which asks for a direct answer
  (``data.answers``);
- **target**: three parts, each encoded on its own (``receivers.render.TargetPart``): the scaffold, its close, and the
  row's own answer with its own spans and cap. A row whose answer turn opens with a prefill (an answer convention's
  prefix, ``data.answers``) has it moved after the close as its own part with weight 0 (context, not a target), so
  the row ends as a post-reasoning readout does: close, prefix, answer. The close is the receiver's end-of-thinking text
  (``ReceiverSpec.thinking_close``; Kimi K2.6, the paper); a receiver without a thinking mode (Gemma)
  takes the scaffold as ordinary assistant text, in the row's own mode (so without the receiver's thinking
  request), closed by the receiver's separator (``ReceiverSpec.separator``). Both live in the
  receiver config, which refuses both; a scaffold refuses neither, so each row has exactly one close;
- **scaffold**: one of the stance lines, then 1-3 property lines ``\n<Label>: <value>.`` in random order, with
  values from the object's *trained* facts only: screened training-text fields and the labels of trained Yes/No
  tasks (:class:`FactSpec`). A row never states its own answer: the facts the config omits for its question, the
  values it lists as give-aways for that question (P(Yes | fact) above 0.9 or below 0.1 on training objects, data in
  the config), and, for a Yes/No question, its own label are left out;
- **weights**: the stance and the values get :attr:`ScaffoldSpec.value_weight`; labels, punctuation and the close
  get the floor; the answer keeps its own weights, its curve starting at its first token.

A row whose object has fewer than ``n_facts[0]`` usable facts, or whose scaffold exceeds ``max_tokens``, stays plain
(originally ``FieldScaffold.build``; facts are never dropped to fit). Token counts come from the
renderer (``Renderer.target_token_count``), the only tokenizer user; a value over ``fact_max_tokens`` is not used. The
transform draws from its own generator, so scaffolding never moves a stream's draws.

The mechanism is domain-agnostic: the protein scaffold and the molecule scaffold (trained ChEBI fields) are two
configs (``configs/streams/``).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from latent_interconnect.config import Config
from latent_interconnect.data.streams import Example, Supervision
from latent_interconnect.data.training_text import TrainingText
from latent_interconnect.domains.base import TaskKind, TaskScope
from latent_interconnect.domains.views import CatalogView, DomainView
from latent_interconnect.receivers.render import Conversation, Message, TargetPart, WeightedSpan

__all__ = ["FactSpec", "GiveawayRule", "Scaffold", "ScaffoldSpec", "TokenCount", "scaffold_supervision"]

TokenCount = Callable[[str], int]
"""Text → its number of receiver tokens when encoded on its own (``Renderer.target_token_count``)."""


class FactSpec(Config):
    """One property line of the scaffold, ``<label>: <value>``.

    Sources:

    - ``field``: a screened training-text field: its first item (cut at ``split``), first sentence, without the final
      period;
    - ``labels``: the trained Yes/No tasks in ``tasks`` whose label is true for the object, in config order, at most
      ``max_items``, joined by ``", "``;
    - ``choice``: ``choices[task]`` for the true ones among ``choices`` (``both`` when several are true);
    - ``field_class``: ``classes[<the field's text before its first ".">]`` (e.g. an EC number's class);
    - ``labels_or_field``: as ``labels``, else as ``field``;
    - ``field_list``: a list field's items (cut at ``split``), at most ``max_items``, joined by ``join`` (e.g. a
      molecule's parent classes).

    Attributes:
        key: the fact's name (``ScaffoldSpec.omit`` cites it).
        label: the line's label.
        source: where the value comes from.
        field: the training-text field (``field``, ``field_class``, ``labels_or_field``, ``field_list``).
        split: the item separator of the field (``field``: its first item is used; ``field_list``: required), or None.
        tasks: trained Yes/No task ids (``labels``, ``labels_or_field``).
        choices: trained Yes/No task id → its text (``choice``).
        both: the text when several choices are true (``choice``); None joins them with ``" and "``.
        classes: the field's leading code → its text (``field_class``).
        none: the value when the source gives nothing; None omits the line.
        max_items: the most items of a list value.
        lower_first: lower-case the value's first letter unless its second is upper-case (``"DNA"`` stays).
        join: between the items of a list value (``labels``, ``labels_or_field``, ``field_list``); a drop entry
            matches its items one by one (``ScaffoldSpec.itemwise``).
    """

    key: str
    label: str
    source: Literal["field", "labels", "choice", "field_class", "labels_or_field", "field_list"]
    field: str | None = None
    split: str | None = None
    tasks: tuple[str, ...] = ()
    choices: dict[str, str] = Field(default_factory=dict)
    both: str | None = None
    classes: dict[str, str] = Field(default_factory=dict)
    none: str | None = None
    max_items: int = Field(default=3, ge=1)
    lower_first: bool = True
    join: str = Field(default=", ", min_length=1)

    @model_validator(mode="after")
    def _check(self) -> FactSpec:
        needs_field = self.source in ("field", "field_class", "labels_or_field", "field_list")
        if self.source == "field_list" and not self.split:
            raise ValueError(f"fact {self.key!r}: source 'field_list' needs split")
        if needs_field != (self.field is not None):
            verb = "needs" if needs_field else "takes no"
            raise ValueError(f"fact {self.key!r}: source {self.source!r} {verb} field")
        if self.source in ("labels", "labels_or_field") and not self.tasks:
            raise ValueError(f"fact {self.key!r}: source {self.source!r} needs tasks")
        if (self.source == "choice") != bool(self.choices) or (self.source == "field_class") != bool(self.classes):
            raise ValueError(f"fact {self.key!r}: choices go with 'choice' and classes with 'field_class' only")
        return self

    def cited_tasks(self) -> tuple[str, ...]:
        """Return the Yes/No task ids whose labels this fact reads."""
        return (*self.tasks, *self.choices)


class GiveawayRule(Config):
    """Which property values give a Yes/No question's answer away, measured on the training text.

    A fact value v is dropped from the rows asking a trained Yes/No question q when its balanced posterior
    ``P(Yes | v) = r_yes / (r_yes + r_no)`` (``r_yes``, ``r_no``: the share of q's Yes and No objects that show v; the
    rows draw Yes and No 50/50) is above ``high`` or below ``low``, on at least ``min_support`` objects with a known
    label that show v. The original protein rule (the original reasoning scaffold's ``yesno_exclusions``: 0.9,
    0.1, 30 proteins), here applied to every training object of the run.

    Attributes:
        high: above it, v states Yes.
        low: below it, v states No.
        min_support: the fewest objects showing v for its posterior to be measured (fewer: kept).
    """

    high: float = Field(default=0.9, gt=0.5, le=1.0)
    low: float = Field(default=0.1, ge=0.0, lt=0.5)
    min_support: int = Field(default=30, ge=1)


class ScaffoldSpec(Config):
    """A reasoning scaffold, as data (``configs/streams/scaffold-*.yaml``).

    Attributes:
        rate: the share of eligible rows a scaffold is attempted on (0.2 in the scaffold configs).
        questions: the eligible question (task) ids.
        instructions: the training instructions; one is drawn per row.
        stances: the target's opening lines (accepting the prompt's frame); one is drawn per row; empty for none.
        facts: the property lines, in config order.
        n_facts: the least and most lines per row (``[1, 3]`` in the shipped scaffold configs).
        value_weight: the loss weight of the stance and the values.
        floor: the loss weight of every other scaffold token and of the close (the receiver's end-of-thinking text or
            separator).
        fact_max_tokens: a value longer than this (receiver tokens) is not used.
        max_tokens: a scaffold longer than this stays off its row.
        omit: question id → fact keys left out of its rows (the question's own facts).
        drop: question id → ``"<label>=<value>"`` give-aways left out of its rows (measured once and listed here).
        giveaway: the declared rule that measures a Yes/No question's give-aways on the run's training text instead
            (:class:`GiveawayRule`); its drops add to ``drop``. None: only the listed drops.
        itemwise: the fact sources whose list values a drop entry matches item by item (``labels`` and
            ``labels_or_field``); a value of another source is dropped only when it equals a drop entry whole.
            The original runs matched ``labels`` (originally ``keywords_list``) item by item and ``labels_or_field``
            (originally ``keywords_or_field``) whole. Required, so every scaffold config states it and a run's digest
            (which embeds the stream config as written) records it.
    """

    rate: float = Field(gt=0.0, le=1.0)
    questions: tuple[str, ...] = Field(min_length=1)
    instructions: tuple[str, ...] = Field(min_length=1)
    stances: tuple[str, ...] = ()
    facts: tuple[FactSpec, ...] = Field(min_length=1)
    n_facts: tuple[int, int]
    value_weight: float = Field(gt=0.0)
    floor: float = Field(ge=0.0)
    fact_max_tokens: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    omit: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    drop: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    itemwise: tuple[Literal["labels", "labels_or_field", "field_list"], ...]
    giveaway: GiveawayRule | None = None

    @model_validator(mode="after")
    def _check(self) -> ScaffoldSpec:
        lo, hi = self.n_facts
        if not 1 <= lo <= hi:
            raise ValueError(f"n_facts must be 1 <= min <= max, got {self.n_facts}")
        keys = [f.key for f in self.facts]
        unknown = sorted({k for ks in self.omit.values() for k in ks} - set(keys))
        if len(set(keys)) != len(keys) or unknown:
            raise ValueError(f"fact keys must be unique and omit must cite them: keys {keys}, unknown {unknown}")
        stray = sorted((set(self.omit) | set(self.drop)) - set(self.questions))
        if stray:  # a key that names no question would silently switch its give-aways off
            raise ValueError(f"omit and drop keys must be questions of the scaffold; not questions: {stray}")
        labels = [f.label for f in self.facts]
        if len(set(labels)) != len(labels):
            raise ValueError(f"fact labels must be unique (a drop entry names its fact by label): {labels}")
        by_label = {f.label: f for f in self.facts}
        stale = [
            f"{question}: {entry!r}"
            for question, entries in self.drop.items()
            for entry in entries
            if not _producible(by_label, entry, self.itemwise)
        ]
        if stale:
            raise ValueError(
                f"drop entries no fact can produce (they would never match, so a give-away leaks): {stale}"
            )
        return self


def _producible(facts: Mapping[str, FactSpec], entry: str, itemwise: Collection[str]) -> bool:
    """Whether a ``"<label>=<value>"`` drop entry is a value its fact can produce, as the drop matches it.

    Exact for the values a fact makes from its own config: a ``labels`` item (when ``itemwise`` matches its source
    item by item; else a list value whole: one to ``max_items`` label items in config order, joined by ``join``), a
    ``choice`` text (``both``, or two choices joined by ``" and "``), a ``field_class`` class, ``none``. A value from
    field text can be any text, so it is checked against the formatting the fact applies (its first sentence,
    whitespace collapsed): no surrounding whitespace, no final period, no ``". "`` inside, no double space, no other
    whitespace (a newline), not the field's ``split`` separator (the fact keeps the first item), not ``join`` when
    ``itemwise`` matches the fact's source item by item, the casing rule (a ``lower_first`` value never starts
    upper-case then lower-case), and, for ``labels_or_field``, a value equal to one of its label values up to case
    must equal it exactly (the "dNA repair" case).
    """
    label, sep, value = entry.partition("=")
    f = facts.get(label)
    if not sep or f is None:
        return False
    choices = list(f.choices.values())
    shown = [_shown(t, f.lower_first) for t in f.tasks]
    own = set(shown) | set(choices) | set(f.classes.values())
    own |= {v for v in (f.both, f.none) if v is not None}
    if f.both is None:
        own |= {f"{a} and {b}" for i, a in enumerate(choices) for b in choices[i + 1 :]}
    items = f.source in itemwise
    if not items and f.source in ("labels", "labels_or_field") and _joined(shown, value, f.max_items, f.join):
        return True
    if f.source in ("labels", "choice", "field_class"):
        return value in own
    if value in own:
        return True
    malformed = (
        value != value.strip()
        or not value
        or value.endswith(".")
        or ". " in value
        or "  " in value
        or any(c.isspace() and c != " " for c in value)
        or (f.split is not None and f.split.strip() in value)
        or (items and f.join in value)
    )
    if malformed or any(value.casefold() == o.casefold() for o in own):
        return False
    return not f.lower_first or _lower(value) == value


def _joined(shown: Sequence[str], value: str, max_items: int, join: str) -> bool:
    """Whether ``value`` is a whole list value of these label items: one to ``max_items`` of them, in config order."""
    parts = value.split(join)
    if not 1 <= len(parts) <= max_items or not set(parts) <= set(shown):
        return False
    at = [shown.index(p) for p in parts]
    return at == sorted(set(at))  # distinct, in config order


def _shown(task: str, lower_first: bool) -> str:
    """A label item as a fact shows it."""
    return _lower(task) if lower_first else task


class Scaffold:
    """Applies a :class:`ScaffoldSpec` to examples.

    Args:
        spec: the scaffold.
        domain: the domain (the trained Yes/No labels).
        text: the screened training text (the field values).
        count: the renderer's token count.
        thinking_close: the receiver's end-of-thinking text (``ReceiverSpec.thinking_close``), None for a receiver
            without a thinking mode.
        separator: the receiver's separator (``ReceiverSpec.separator``), the close without a thinking close.
        instruction: the stream's answer-convention instruction (``Answering.instruction``), removed from the end of a
            scaffolded row's last user message; ``""`` for none (an inline policy).

    Raises:
        ValueError: if a fact or the spec cites a task that is not a trained Yes/No task of the domain (a fact never
            reads a held-out label), or the receiver has neither a thinking close nor a separator.
    """

    def __init__(
        self,
        spec: ScaffoldSpec,
        domain: DomainView,
        text: TrainingText,
        count: TokenCount,
        thinking_close: str | None,
        *,
        separator: str | None,
        instruction: str = "",
    ) -> None:
        tasks = domain.tasks()
        cited = {t for f in spec.facts for t in f.cited_tasks()}
        trained = {t for t, task in tasks.items() if task.kind == TaskKind.YES_NO and task.scope == TaskScope.TRAINED}
        bad = sorted(cited - trained)
        if bad:
            raise ValueError(f"scaffold facts cite tasks that are not trained Yes/No tasks: {bad}")
        close = thinking_close if thinking_close is not None else separator
        if close is None:
            raise ValueError("a scaffold is closed by the receiver's thinking close or separator; it has neither")
        self.close, self.thinking = close, thinking_close is not None
        self.instruction = instruction
        self.spec, self.text, self.count = spec, text, count
        self._tasks = tasks
        ids = list(text.object_ids)
        self._true: dict[str, frozenset[str]] = {}
        # a null: a variant object has no keyword labels; a cell outside its task's scope (streams.ScopedDomain)
        self._unknown: dict[str, frozenset[str]] = {}
        for t in sorted(cited):
            col = domain.labels([tasks[t].label])[tasks[t].label].reindex(ids)
            known = col.dropna()
            self._true[t] = frozenset(str(o) for o, v in known.items() if v is True or v == 1)
            self._unknown[t] = frozenset(str(o) for o in col.index[col.isna()])
        self.measured: dict[str, frozenset[str]] = {}
        if spec.giveaway is not None:
            self.measured = self._measure(spec.giveaway, domain, ids)

    def _measure(self, rule: GiveawayRule, domain: DomainView, ids: list[str]) -> dict[str, frozenset[str]]:
        """The give-aways of every trained Yes/No question of the scaffold (:class:`GiveawayRule`)."""
        shown: dict[str, set[str]] = {}  # object → the "<label>=<item>" values it can show
        for o in ids:
            fields = self.text.fields.get(o, {})
            entries = set()
            for f in self.spec.facts:
                value = self._value(f, o, fields, None)
                if value:
                    items = value.split(f.join) if f.source in self.spec.itemwise else [value]
                    entries |= {f"{f.label}={v}" for v in items}
            shown[o] = entries
        out = {}
        for q in self.spec.questions:
            task = self._tasks.get(q)
            if task is None or task.kind != TaskKind.YES_NO:
                continue
            col = domain.labels([task.label])[task.label].reindex(ids).dropna()
            yes = [str(o) for o, v in col.items() if bool(v)]
            no = [str(o) for o, v in col.items() if not bool(v)]
            if not yes or not no:
                continue
            n_yes = Counter(e for o in yes for e in shown.get(o, ()))
            n_no = Counter(e for o in no for e in shown.get(o, ()))
            dropped = set()
            for entry in set(n_yes) | set(n_no):
                support = n_yes[entry] + n_no[entry]
                if support < rule.min_support:
                    continue
                r_yes, r_no = n_yes[entry] / len(yes), n_no[entry] / len(no)
                posterior = r_yes / (r_yes + r_no)
                if posterior > rule.high or posterior < rule.low:
                    dropped.add(entry)
            out[q] = frozenset(dropped)
        return out

    def derived(self) -> dict[str, object]:
        """Return what the scaffold measured on the training text (the give-aways per question), or nothing."""
        if self.spec.giveaway is None:
            return {}
        return {"giveaways": {q: sorted(d) for q, d in sorted(self.measured.items())}}

    def eligible(self, example: Example) -> bool:
        """Return whether the example's question is eligible and it is a plain single-answer row."""
        return example.question in self.spec.questions and not example.prefix and example.subject is not None

    def apply(self, example: Example, rng: np.random.Generator) -> Example | None:
        """Return the scaffolded example, or None when no scaffold fits (too few facts, or over ``max_tokens``)."""
        question = example.question
        if question is None or example.subject is None:
            return None
        values = self.values(example.objects[example.subject], question)
        lo = self.spec.n_facts[0]
        if len(values) < lo:
            return None
        n = int(rng.integers(lo, min(len(values), self.spec.n_facts[1]) + 1))
        chosen = [values[i] for i in rng.permutation(len(values))[:n]]
        scaffold, spans = self._render(chosen, rng)
        if self.count(scaffold) > self.spec.max_tokens:
            return None
        instruction = self.spec.instructions[int(rng.integers(len(self.spec.instructions)))]
        prefix: tuple[TargetPart, ...] = (
            TargetPart(scaffold, spans),
            TargetPart(self.close, (WeightedSpan(0, len(self.close), self.spec.floor),)),
        )
        lead = example.conversation.prefill
        if lead is not None and not isinstance(lead, str):
            raise ValueError(f"a scaffold moves a text prefill after its close, not {lead!r}")
        if lead:
            prefix += (TargetPart(lead),)
        conversation = _instructed(example.conversation, instruction, thinking=self.thinking, direct=self.instruction)
        conversation = dataclasses.replace(conversation, prefill=None)
        return dataclasses.replace(example, conversation=conversation, prefix=prefix)

    def values(self, object_id: str, question: str) -> list[tuple[str, str]]:
        """Return the object's usable ``(label, value)`` lines for a row asking ``question``, in config order."""
        omitted = set(self.spec.omit.get(question, ()))
        drop = set(self.spec.drop.get(question, ())) | self.measured.get(question, frozenset())
        asked = question if question in self._tasks and self._tasks[question].kind == TaskKind.YES_NO else None
        fields = self.text.fields.get(object_id, {})
        out = []
        for f in self.spec.facts:
            if f.key in omitted:
                continue
            value = self._value(f, object_id, fields, asked)
            if value is not None and drop:
                items = value.split(f.join) if f.source in self.spec.itemwise else [value]
                kept = [v for v in items if f"{f.label}={v}" not in drop]
                value = f.join.join(kept) if kept else None
            if value and self.count(value) <= self.spec.fact_max_tokens:
                out.append((f.label, value))
        return out

    def _value(self, f: FactSpec, object_id: str, fields: Mapping[str, str], asked: str | None) -> str | None:
        """The fact's value for the object (None: no line), the asked Yes/No task's own label never among them."""

        def true(tasks: tuple[str, ...]) -> list[str]:
            return [t for t in tasks if t != asked and object_id in self._true[t]]

        value: str | None = None
        if f.source in ("labels", "labels_or_field"):
            have = true(f.tasks)[: f.max_items]
            value = f.join.join(_shown(t, f.lower_first) for t in have) if have else None
        elif f.source == "choice":
            hit = [f.choices[t] for t in true(tuple(f.choices))]
            value = (f.both or " and ".join(hit)) if len(hit) > 1 else (hit[0] if hit else None)
        raw = (fields.get(f.field) or "").strip() if f.field is not None else ""
        if value is None and raw and f.source in ("field", "labels_or_field"):
            first = _first_sentence(raw.split(f.split)[0] if f.split else raw)
            value = _lower(first) if f.lower_first else first
        elif value is None and raw and f.source == "field_class":
            value = f.classes.get(raw.split(".")[0])
        elif raw and f.source == "field_list":
            items = [i.strip() for i in raw.split(f.split or "") if i.strip()][: f.max_items]
            value = f.join.join(_shown(i, f.lower_first) for i in items) if items else None
        if value is None:  # a fact whose labels are unknown for the object states nothing, never its negative
            unknown = any(object_id in self._unknown[t] for t in f.cited_tasks() if t != asked)
            return None if unknown else f.none
        return value

    def _render(self, values: list[tuple[str, str]], rng: np.random.Generator) -> tuple[str, tuple[WeightedSpan, ...]]:
        text, spans = "", []
        w = self.spec.value_weight
        if self.spec.stances:
            text = self.spec.stances[int(rng.integers(len(self.spec.stances)))]
            spans.append((0, len(text)))
        for label, value in values:
            text += f"\n{label}: " if text else f"{label}: "
            # the space before a value belongs to it: a token joining the two takes the value's weight, as the original
            # flags did (a token overlapping a value), since the renderer gives a token its first span's weight
            spans.append((len(text) - 1, len(text) + len(value)))
            text += value + "."
        weighted, at = [], 0
        for a, b in spans:  # the floor between the weighted spans: every scaffold token is supervised
            if a > at:
                weighted.append(WeightedSpan(at, a, self.spec.floor))
            weighted.append(WeightedSpan(a, b, w))
            at = b
        if at < len(text):
            weighted.append(WeightedSpan(at, len(text), self.spec.floor))
        return text, tuple(weighted)


def scaffold_supervision(spec: ScaffoldSpec, domain: CatalogView) -> Supervision:
    """Return what a scaffold puts into its rows: the facts' tasks, labels and fields, and its fixed wording."""
    catalog = domain.tasks()
    tasks = frozenset(t for f in spec.facts for t in f.cited_tasks())
    if spec.giveaway is not None:  # the rule reads the questions' labels
        tasks |= {q for q in spec.questions if q in catalog and catalog[q].kind == TaskKind.YES_NO}
    missing = sorted(tasks - set(catalog))
    if missing:
        raise ValueError(f"scaffold facts cite tasks the domain catalog lacks: {missing}")
    fields = frozenset(f.field for f in spec.facts if f.field is not None)
    wording = [*spec.instructions, *spec.stances]
    for f in spec.facts:
        wording += [f.label, *f.choices.values(), *f.classes.values(), *(t for t in (f.both, f.none) if t is not None)]
    labels = frozenset(catalog[t].label for t in tasks if t in catalog)
    return Supervision(tasks, labels, fields, tuple(wording))


def _instructed(conversation: Conversation, instruction: str, *, thinking: bool, direct: str) -> Conversation:
    """The conversation with ``instruction`` appended to its last user message, in thinking mode if ``thinking``.

    The message's ending ``" " + direct`` (the answer convention's instruction) is removed first; ``direct`` ``""``
    removes nothing. Without ``thinking`` the row keeps its own mode (a scaffold as ordinary assistant text).

    Raises:
        ValueError: if the last user message does not end with ``" " + direct``.
    """
    messages = list(conversation.messages)
    last = max(i for i, m in enumerate(messages) if m.role == "user")
    content = messages[last].content
    if direct:
        if not content.endswith(f" {direct}"):
            raise ValueError(f"a scaffolded row's user turn does not end with its answer instruction {direct!r}")
        content = content[: -len(direct) - 1]
    messages[last] = Message("user", f"{content} {instruction}")
    return dataclasses.replace(conversation, messages=tuple(messages), thinking=thinking or conversation.thinking)


def _first_sentence(text: str) -> str:
    """The first sentence (to the first ``". "``), whitespace normalized, without its final period."""
    t = " ".join(text.split())
    i = t.find(". ")
    t = t[: i + 1] if i >= 0 else t
    return t[:-1] if t.endswith(".") else t


def _lower(text: str) -> str:
    """Lower-case the first letter unless the second is upper-case (an acronym such as ``"DNA"`` stays)."""
    return text[0].lower() + text[1:] if text[:1].isupper() and not text[1:2].isupper() else text
