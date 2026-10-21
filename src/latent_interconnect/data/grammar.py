"""The prompt grammar with its target wording, as data: compositional wording of training rows.

A port of the original research code's ``Grammar`` and of its grammar sampling for text rows and template rows,
with the original prompt grammar and target wording merged into one config (``configs/streams/grammar-protein.yaml``).
Only the training entries are carried: the last entry of every pool is the original's held-out entry for its
unseen-wording evaluation and is never drawn in training.

- **A grammar row** (a share :attr:`GrammarSpec.rate` of the text and Yes/No rows, originally ``--p-grammar``): the user
  turn is form A (``[lead] + core`` with ``{P}`` the slot phrase), B (``intro + sep + pronoun core``) or C (``pronoun
  core + ref``), plus, for a text task, the instruction of an answer format drawn among the task's enabled formats
  (:func:`fit_coverage`) that apply to the object's text (:mod:`data.wording`); for a Yes/No task the question
  and a Yes/No format. A draw that reproduces a held-out evaluation template is redrawn. Without in-context examples the
  row gets a grammar system role (or none).
- **Target wording**: a family-generic target (``wording.family_generic``) gets a source-naming core and its verbatim
  text; a subunit core or template that asks for partners is used only when the answer names partners; the UniProt
  instruction is announced for the configured tasks and an answer whose format must fit the cap is checked;
  the target wording disables formats (:attr:`GrammarSpec.disabled`) and the 3-part pathway sentence. On template
  rows the same rules apply through :meth:`Grammar.template_wording`.
- **Relational rows** (the same share): a lead and a system role around the relational prompt
  (:meth:`Grammar.wrap_relational`).
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import Field, model_validator

from latent_interconnect.config import Config
from latent_interconnect.data.wording import TRANSFORMS, Formatted, family_generic, names_partners, transform

__all__ = ["Cores", "Format", "FormatCoverage", "Grammar", "GrammarSpec", "UserTurn", "fit_coverage"]

SLOT = "<SLOT:x>"
"""The slot marker of grammar wording (the single-object streams' slot)."""
_TRIES = 50
"""Draws of a user turn before giving up on a non-colliding one (the original ``Grammar.user``)."""


class Cores(Config):
    """A task's question cores.

    Attributes:
        inline: cores with ``{P}``, the slot phrase (form A).
        pronoun: cores asking about "it" (forms B and C).
    """

    inline: tuple[str, ...] = Field(min_length=1)
    pronoun: tuple[str, ...] = Field(min_length=1)


class Format(Config):
    """An enabled answer format of a task.

    Attributes:
        transform: the format (``data.wording.TRANSFORMS``).
        instructions: its instructions (each with its leading space); ``none`` has the empty one.
    """

    transform: str
    instructions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Format:
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unknown answer format {self.transform!r}; known: {TRANSFORMS}")
        return self


class GrammarSpec(Config):
    """The prompt grammar and the target wording, training entries only.

    Attributes:
        rate: the share of text, Yes/No and relational rows worded by the grammar (originally ``--p-grammar``).
        system_roles: system roles of a grammar row without in-context examples; null for none.
        leads: form A's leads.
        slot_phrases: form A's slot phrases (``{P}``), each with the slot marker.
        intros: form B's intros, each with the slot marker.
        seps: form B's separators.
        refs: form C's references, each with the slot marker.
        yes_no_inline: form A's Yes/No prefixes with ``{P}``.
        yes_no_formats: the Yes/No answer instructions.
        cores: task → its question cores.
        source_cores: task → its source-naming cores.
        formats: task → its candidate answer formats with their instructions (every format its group allows the task).
        min_coverage: a candidate is enabled when it applies to at least this share of the task's training texts
            (:func:`fit_coverage`); ``none`` always is.
        coverage_sample: the most training texts per task the coverage is measured on (a seeded sample).
        coverage_seed: the seed of that sample.
        disabled: ``"task:format"`` → why it is off whatever its coverage (a measured vetting result).
        reject: the held-out evaluation templates a training draw must not reproduce (with the slot marker; a Yes/No
            template with ``{question}``).
        lexicon: a regular expression of property words no pool entry may contain (cores and instructions exempt).
        announce_uniprot: tasks whose answers announce the UniProt format.
        flag_tasks: tasks whose family-generic targets get the source-naming wording.
        fit_cap_formats: formats used only when their answer fits the answer cap.
        partner_cores: task → indices of its ``inline`` and ``pronoun`` cores that ask for partners.
        template_rewrites: training template id → its rewritten user text.
        partner_templates: training template ids that ask for partners.
        source_example_questions: task → the in-context example question of a source-naming row.
        format_example_questions: format → the in-context example question of a row in that format.
        pathway_sentence_max_parts: the most ``;`` parts a pathway sentence may have; None: no limit.

    The target-wording fields (``source_cores`` through ``pathway_sentence_max_parts``) are optional: a
    grammar without target wording (the molecule grammar) leaves them empty.
    """

    rate: float = Field(gt=0.0, le=1.0)
    system_roles: tuple[str | None, ...] = Field(min_length=1)
    leads: tuple[str, ...] = Field(min_length=1)
    slot_phrases: tuple[str, ...] = Field(min_length=1)
    intros: tuple[str, ...] = Field(min_length=1)
    seps: tuple[str, ...] = Field(min_length=1)
    refs: tuple[str, ...] = Field(min_length=1)
    yes_no_inline: tuple[str, ...] = Field(min_length=1)
    yes_no_formats: tuple[str, ...] = Field(min_length=1)
    cores: dict[str, Cores]
    source_cores: dict[str, Cores] = Field(default_factory=dict)
    formats: dict[str, tuple[Format, ...]]
    reject: tuple[str, ...]
    lexicon: str
    announce_uniprot: tuple[str, ...] = ()
    flag_tasks: tuple[str, ...] = ()
    fit_cap_formats: tuple[str, ...] = ()
    partner_cores: dict[str, dict[str, tuple[int, ...]]] = Field(default_factory=dict)
    template_rewrites: dict[str, str] = Field(default_factory=dict)
    partner_templates: tuple[str, ...] = ()
    source_example_questions: dict[str, str] = Field(default_factory=dict)
    format_example_questions: dict[str, str] = Field(default_factory=dict)
    pathway_sentence_max_parts: int | None = Field(default=None, ge=1)
    min_coverage: float = Field(ge=0.0, le=1.0)
    coverage_sample: int = Field(ge=1)
    coverage_seed: int
    disabled: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> GrammarSpec:
        lexicon = re.compile(self.lexicon, re.IGNORECASE)
        pools = (*self.leads, *self.slot_phrases, *self.intros, *self.refs, *self.yes_no_inline)
        pools += tuple(r for r in self.system_roles if r)
        leaks = [e for e in pools if lexicon.search(e.replace(SLOT, ""))]
        problems = {
            "pool entries stating a property": leaks,
            "tasks without cores": sorted(set(self.formats) - set(self.cores)),
            "source tasks without an example question": sorted(
                set(self.source_cores) - set(self.source_example_questions)
            ),
            "flagged tasks without source cores": sorted(set(self.flag_tasks) - set(self.source_cores)),
            "announced tasks without a verbatim format": sorted(
                t for t in self.announce_uniprot if "verbatim" not in {f.transform for f in self.formats.get(t, ())}
            ),
        }
        problems["tasks without a none format"] = sorted(
            t for t, fs in self.formats.items() if "none" not in {f.transform for f in fs}
        )  # fail closed: a task always has a plain answer, so its enabled set is never empty
        declared = {f"{t}:{f.transform}" for t, fs in self.formats.items() for f in fs}
        problems["disabled formats that are not candidates"] = sorted(set(self.disabled) - declared)
        problems["disabled entries naming none"] = sorted(d for d in self.disabled if d.endswith(":none"))
        if any(problems.values()):
            raise ValueError(f"grammar config: {problems}")
        return self


@dataclass(frozen=True)
class UserTurn:
    """A grammar-worded row's wording.

    Attributes:
        user: the user message (with the format instruction).
        answer: the answer text (formatted).
        template_spans: the answer's announced-format text spans.
        answer_format: the format (``"source"`` for a source-naming row).
        context_question: the question of a same-question in-context example (None: the task's own).
        context_suffix: appended to that question (the row's format instruction, so the example demonstrates it).
    """

    user: str
    answer: str
    template_spans: tuple[tuple[int, int], ...]
    answer_format: str
    context_question: str | None
    context_suffix: str


Fits = Callable[[str], bool]
"""Whether an answer fits the answer cap (receiver tokens; the stream asks the renderer)."""


@dataclass(frozen=True)
class FormatCoverage:
    """The answer formats a run enables, recorded with the run.

    Attributes:
        coverage: ``"task:format"`` → the share of the task's sampled training texts the format applies to.
        enabled: task → its enabled formats, in config order.
    """

    coverage: dict[str, float]
    enabled: dict[str, tuple[str, ...]]


def fit_coverage(spec: GrammarSpec, texts: Mapping[str, Sequence[str]]) -> FormatCoverage:
    """Enable each task's formats by the declared rule (the original ``fit_coverage``, with one generator per task).

    ``texts``: task → its training texts (the task's field over the stream's objects); only these tasks are measured and
    recorded. Up to :attr:`GrammarSpec.coverage_sample` of a task's texts, drawn with its own generator
    (``[coverage_seed, crc32(task)]``, so one task's sample never depends on another task's data or existence), measure
    each candidate's coverage (``data.wording.transform`` gives an answer); a candidate is enabled at
    :attr:`GrammarSpec.min_coverage` or above unless :attr:`GrammarSpec.disabled` names it, and ``none`` always is. A
    task the stream does not train keeps only ``none`` (:class:`Grammar`).

    Raises:
        ValueError: if a measured task ends with no enabled format.
    """
    coverage: dict[str, float] = {}
    enabled: dict[str, tuple[str, ...]] = {}
    for task in sorted(spec.formats):
        if task not in texts:
            continue
        have = sorted(t for t in texts[task] if t and t.strip())
        if len(have) > spec.coverage_sample:
            rng = np.random.default_rng([spec.coverage_seed, zlib.crc32(task.encode())])
            have = [have[int(j)] for j in sorted(rng.choice(len(have), spec.coverage_sample, replace=False))]
        on = []
        for f in spec.formats[task]:
            share = sum(transform(f.transform, t, task) is not None for t in have) / max(len(have), 1)
            coverage[f"{task}:{f.transform}"] = round(share, 4)
            if f.transform == "none" or (share >= spec.min_coverage and f"{task}:{f.transform}" not in spec.disabled):
                on.append(f.transform)
        if not on:  # a backstop: GrammarSpec requires a never-disabled `none` per task, so this cannot happen
            raise ValueError(f"grammar: no answer format enabled for task {task!r}")
        enabled[task] = tuple(on)
    return FormatCoverage(coverage, enabled)


class Grammar:
    """Draws the grammar's wording (see the module docstring).

    Args:
        spec: the grammar config.
        enabled: task → the formats a row may take (:func:`fit_coverage`); None: every candidate.
    """

    def __init__(self, spec: GrammarSpec, enabled: Mapping[str, Sequence[str]] | None = None) -> None:
        self.spec = spec
        self.enabled = enabled
        self._reject = [
            (h.split(SLOT)[0], h.split(SLOT)[1] if SLOT in h else "", h) for h in spec.reject
        ]  # (text before the slot, after it, whole)

    @staticmethod
    def pick(g: np.random.Generator, pool: Sequence[str] | Sequence[str | None]) -> str | None:
        """A uniform entry of a training pool (no draw for a one-entry pool, as the original ``pick``)."""
        return pool[0] if len(pool) == 1 else pool[int(g.integers(len(pool)))]

    def _text(self, g: np.random.Generator, pool: Sequence[str]) -> str:
        entry = self.pick(g, pool)
        return entry if entry is not None else ""

    def system_role(self, g: np.random.Generator) -> str | None:
        """A system role of a grammar row without in-context examples (None: no system turn)."""
        return self.pick(g, self.spec.system_roles)

    def wrap_relational(self, g: np.random.Generator, prompt: str) -> tuple[str, str | None]:
        """A lead before the relational prompt (without its trailing space before a slot) and a system role."""
        lead = self._text(g, self.spec.leads)
        role = self.system_role(g)
        if lead and prompt.startswith("<SLOT"):
            lead = lead.rstrip()
        return lead + prompt, role

    # ---- the user turn

    def _collides(self, user: str, question: str | None) -> bool:
        u = _norm(user)
        for pre, post, whole in self._reject:
            h = _norm(whole.replace("{question}", question or ""))
            if h in u:
                return True
            post_q = _norm(post.replace("{question}", question or "")) if post else ""
            head = _norm(pre) + SLOT if pre.strip() else SLOT
            if u.startswith(head) and post_q[:25] and post_q[:25] in u:
                return True
        return False

    def user(
        self,
        g: np.random.Generator,
        task: str,
        question: str | None = None,
        answer: str | None = None,
        source: bool = False,
        *,
        instructed: bool = True,
    ) -> str:
        """A user turn without the format instruction, redrawn while it reproduces a held-out template.

        A Yes/No turn (``question`` given) ends on one of the Yes/No answer instructions when ``instructed``; without,
        on the question (a stream under an answer convention, ``data.answers``, whose instruction replaces them).

        Raises:
            RuntimeError: if no draw avoids the held-out templates.
        """
        for _ in range(_TRIES):
            s = self._user(g, task, question, answer, source, instructed=instructed)
            if not self._collides(s, question):
                return s
        raise RuntimeError(f"grammar: no prompt for {task!r} avoids the held-out templates")

    def _user(
        self,
        g: np.random.Generator,
        task: str,
        question: str | None,
        answer: str | None,
        source: bool,
        *,
        instructed: bool,
    ) -> str:
        sp = self.spec
        form = "ABC"[int(g.integers(3))]
        if question is not None:
            fmt = self._text(g, sp.yes_no_formats) if instructed else ""
            if form == "A":
                lead, phrase = self._text(g, sp.leads), self._text(g, sp.slot_phrases)
                pre = self._text(g, sp.yes_no_inline)
                s = (lead + pre.replace("{P}", phrase) + question + fmt).replace("," + SLOT + ",:", SLOT + ":")
            elif form == "B":
                s = f"{self._text(g, sp.intros)}{sp.seps[int(g.integers(len(sp.seps)))]}{question}{fmt}"
            else:
                s = f"{question}{self._text(g, sp.refs)}{fmt}"
            return s[0].upper() + s[1:] if s[0].islower() else s
        cores = sp.source_cores[task] if source else sp.cores[task]
        kind = "inline" if form == "A" else "pronoun"
        pool = getattr(cores, kind)
        allowed = None if source else self._routed(task, kind, len(pool), answer)
        core = pool[allowed[int(g.integers(len(allowed)))]] if allowed else self._text(g, pool)
        if form == "A":
            lead, phrase = self._text(g, sp.leads), self._text(g, sp.slot_phrases)
            s = lead + core.replace("{P}", phrase)
            s = s[0].upper() + s[1:] if s[0] != "<" else s
        elif form == "B":
            s = f"{self._text(g, sp.intros)}{sp.seps[int(g.integers(len(sp.seps)))]}{core}"
        else:
            s = core + self._text(g, sp.refs)
        for a, b in (
            ("," + SLOT + ",?", SLOT + "?"),
            ("," + SLOT + ",.", SLOT + "."),
            (SLOT + ",?", SLOT + "?"),
            (SLOT + ",.", SLOT + "."),
        ):
            s = s.replace(a, b)
        return s

    def _routed(self, task: str, kind: str, n: int, answer: str | None) -> list[int] | None:
        """Partner routing: the core indices allowed for an answer that names no partners, or None (a plain draw)."""
        partners = self.spec.partner_cores.get(task, {}).get(kind)
        if partners is None or answer is None or names_partners(answer):
            return None
        ok = [i for i in range(n) if i not in partners]
        return ok or None

    # ---- answer formats

    def transform_draw(self, name: str, text: str | None, task: str) -> Formatted | None:
        """``data.wording.transform`` for a drawn answer, None for a pathway sentence of too many ``;`` parts."""
        parts = len(text.strip().rstrip(".").split(";")) if text else 0
        limit = self.spec.pathway_sentence_max_parts
        if name == "pathway_sentence" and limit is not None and parts > limit:
            return None
        return transform(name, text, task)

    def instruction(self, g: np.random.Generator, task: str, name: str) -> str:
        """An instruction of ``task``'s format ``name``."""
        for f in self.spec.formats[task]:
            if f.transform == name:
                return self._text(g, f.instructions)
        raise KeyError(f"task {task!r} has no format {name!r}")

    def row(
        self, g: np.random.Generator, task: str, text: str, capped: Callable[[str], str], fits: Fits
    ) -> UserTurn | None:
        """A grammar-worded text row for a field's text, or None when no enabled format applies.

        ``capped`` cuts a text to the answer cap (partner routing reads the answer as trained); ``fits`` tells whether
        an answer fits the cap.
        """
        if task in self.spec.flag_tasks and family_generic(task, text):  # source naming
            user = self.user(g, task, source=True)
            return UserTurn(user, text.strip(), (), "source", self.spec.source_example_questions[task], "")
        ok = [
            (f.transform, r)
            for f in self.spec.formats[task]
            if self.enabled is None or f.transform in self.enabled.get(task, ("none",))
            for r in [self.transform_draw(f.transform, text, task)]
            if r is not None and (f.transform not in self.spec.fit_cap_formats or fits(r[0]))
        ]
        if not ok:
            return None
        name, (answer, spans) = ok[int(g.integers(len(ok)))]
        if name == "none" and task in self.spec.announce_uniprot:  # UniProt announcement
            name = "verbatim"
        instruction = self.instruction(g, task, name)
        user = self.user(g, task, answer=capped(answer)) + instruction
        question = self.spec.format_example_questions.get(name)
        return UserTurn(user, answer, tuple(spans), name, question, instruction)

    def template_wording(  # noqa: PLR0917  (the row's parts, as the original template wording)
        self,
        g: np.random.Generator,
        task: str,
        template_id: str,
        user: str,
        text: str,
        capped: Callable[[str], str],
        plain_templates: Sequence[tuple[str, str]],
    ) -> tuple[str, str | None, str]:
        """Target wording on a template row: the user turn, in-context example question (None: the task's own), suffix.

        A family-generic target gets a source-naming grammar turn; a partner template whose capped answer names
        no partners is replaced by a non-partner template of the task (``plain_templates``: id and user text); a
        rewritten template takes its rewrite; an announced task gets the UniProt instruction (on its own line before
        a completion prompt ending in ``:``).
        """
        if task in self.spec.flag_tasks and family_generic(task, text):
            return self.user(g, task, source=True), self.spec.source_example_questions[task], ""
        tid, turn = template_id, user
        if template_id in self.spec.partner_templates and not names_partners(capped(text)) and plain_templates:
            tid, turn = plain_templates[int(g.integers(len(plain_templates)))]
        turn = self.spec.template_rewrites.get(tid, turn)
        if task not in self.spec.announce_uniprot:
            return turn, None, ""
        instruction = self.instruction(g, task, "verbatim")
        turn = instruction.strip() + "\n" + turn if turn.rstrip().endswith(":") else turn + instruction
        return turn, None, instruction

    def texts(self) -> tuple[str, ...]:
        """Every fixed wording a grammar row can show (the zero-shot screen runs over it)."""
        sp = self.spec
        out = [*(r for r in sp.system_roles if r), *sp.leads, *sp.slot_phrases, *sp.intros, *sp.refs, *sp.yes_no_inline]
        out += [*sp.yes_no_formats, *sp.template_rewrites.values(), *sp.source_example_questions.values()]
        out += list(sp.format_example_questions.values())
        for c in (*sp.cores.values(), *sp.source_cores.values()):
            out += [*c.inline, *c.pronoun]
        out += [i for fs in sp.formats.values() for f in fs for i in f.instructions if i]
        return tuple(out)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
