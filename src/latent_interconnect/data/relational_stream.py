"""The relational stream (``relational``): one to three objects, a multi-sentence answer.

A port of the original ``Pool`` / ``generate`` of the protein relational generator and of ``rel_item`` /
``TemplateData.relational`` of the original trainer; molecule rows use the molecule relational answers in place of
the protein ones.
Per row: a family uniformly from the configured ones, objects drawn for it (cluster-weighted, with related
pairs, hard negatives and random ones as the original ``sample_pair`` / ``sample_triplet``), the answer built from their
screened facts, and one of the family's prompts with the objects at ``<SLOT:p1>``, ``<SLOT:p2>``, ``<SLOT:p3>``.

The stream is domain-agnostic; a domain's records, answer templates, verdicts, fixed wording and pair and triplet
draws are its registered :class:`~latent_interconnect.data.relational_base.RelationalAnswers`, which the config names
(:attr:`RelationalParams.answers`): ``protein`` (``data.relational_records``, ``data.relational_answers``,
``data.relational_protein``) and ``molecule`` (``data.relational_records_molecule``,
``data.relational_answers_molecule``). An answer that names one of its objects is redrawn when the config names
the screened name field (:attr:`RelationalParams.own_name`; the molecule generator's rule).

**Answer length**: an answer over :attr:`RelationalParams.max_target_tokens` that carries a verdict
is replaced by its compact form (no parentheticals such as a protein's superfamily, nothing after the verdict), and
if that is still too long another row is drawn; any other long answer is cut at its last sentence boundary within the
cap. **Weights**: each fact gets the curve of the single-protein format stating the same fact
(:attr:`RelationalParams.fact_curves`), its position counted over the fact tokens of its fact group (one object's
facts), so the curve restarts per object; template text and verdict words get the floor. Both need receiver tokens,
so the stream is bound to the renderer (``bind``, by the sampler) and asks it for token offsets.

**Yes/No-led families** (``RelationalAnswers.yes_no``: ``same_class``, ``same_family``). Under an answer convention
(:attr:`RelationalParams.answer`, ``data.answers``) such a row is asked as a Yes/No question, answer first: the
instruction (:attr:`RelationalParams.instruction`) follows the prompt after one space, the
convention's prefix opens the assistant turn (context, not a target), and the answer's leading verdict is the
convention's answer text, weighted by the ``choice`` curve, with the explanation after it, its weights unchanged.
Without ``answer`` the rows are as before.

Held-out content is kept out by the zero-shot screen of the training text the facts come from; the original per-answer
leak regexes are not ported.
"""

from __future__ import annotations

import dataclasses
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, ClassVar, Protocol

import numpy as np
from pydantic import Field, model_validator

from latent_interconnect.config import Config, added_field
from latent_interconnect.data.answers import AnswerConvention, Answering
from latent_interconnect.data.grammar import Grammar, GrammarSpec
from latent_interconnect.data.relational_base import (
    RELATIONAL_ANSWERS,
    Clean,
    RelationalAnswers,
    RelationalRecord,
    clean,
    fill,
)
from latent_interconnect.data.streams import STREAMS, Example, LossCurve, Supervision, population
from latent_interconnect.data.training_text import TrainingText
from latent_interconnect.domains.views import CatalogView, DomainView
from latent_interconnect.receivers.render import Conversation, Message, WeightedSpan

__all__ = ["RelationalParams", "RelationalStream", "TokenOffsets"]


_DRAWS = 5
"""Family draws per attempt (the original ``rel_item``; the original then gave the row to the text stream)."""
_ATTEMPTS = 20
"""Attempts per row before the stream refuses (a configuration whose families yield no rows)."""
_TRIES = 40
"""Object draws per family draw (the original ``generate``'s ``max_tries``)."""
_MIN_WORDS = 8
"""An answer of fewer words is not used."""
_MIN_NAME_CHARS = 3
"""A shorter own name is not looked for in the answers (the original ``names_in``)."""


class TokenOffsets(Protocol):
    """The renderer's token offsets of a target part (``Renderer.target_token_offsets``)."""

    def __call__(self, text: str) -> tuple[tuple[int, int], ...]:
        """Return each token's ``[start, end)`` in ``text``."""
        ...


class RelationalParams(Config):
    """Parameters of the ``relational`` stream.

    Attributes:
        answers: the domain's relational answers (``data.relational_base.RELATIONAL_ANSWERS``: ``protein``,
            ``molecule``); their noun must be the domain's.
        families: family → the facts it may state (the original task split: cofactor, GO, length and lineage blocked).
        prompts: family → its prompts, with ``<SLOT:p1>`` .. ``<SLOT:p3>``.
        fields: fact → its training-text field (``family`` → ``protein_families``, ...), for every fact the families
            state or read (``RelationalAnswers.requires``).
        lists: list fact → the item separator its training-text field is stored with (``ScreenSpec.list_fields``);
            every used fact the answers read as an item list needs one.
        own_name: the screened training-text field of the object's name: an answer that names one of its objects is
            redrawn, or None for no such check.
        sampling: object draws (``data.streams.Sampling``; the original runs: cluster_sqrt).
        p_hard: the share of odd-one-out triplets whose odd object comes from a related group.
        max_target_tokens: the answer cap (originally ``--cap-tokens-long``).
        floor: the weight of template text and verdict words.
        fact_curves: fact kind → the loss curve of the single-object format stating it, or None for the floor.
        grammar: the prompt grammar whose lead and system role wrap a share of the prompts, or None.
        answer: the receiver's answer convention (``data.answers``) of the Yes/No-led families' rows (module
            docstring), or None for their rows as before.
        instruction: the user-turn instruction of those rows, required with ``answer`` (e.g. "Answer Yes
            or No, then explain.", the target explains), None without it.
    """

    answers: str
    families: dict[str, tuple[str, ...]] = Field(min_length=1)
    prompts: dict[str, tuple[str, ...]]
    fields: dict[str, str]
    lists: dict[str, str] = Field(default_factory=dict)
    own_name: str | None = None
    sampling: str
    p_hard: float = Field(ge=0.0, le=1.0)
    max_target_tokens: int = Field(ge=1)
    floor: float = Field(ge=0.0, le=1.0)
    fact_curves: dict[str, LossCurve | None]
    grammar: GrammarSpec | None = None
    answer: AnswerConvention | None = added_field(default=None)
    instruction: str | None = added_field(default=None)

    @model_validator(mode="after")
    def _check(self) -> RelationalParams:
        if self.answers not in RELATIONAL_ANSWERS:
            raise ValueError(f"relational stream: unknown answers {self.answers!r} ({RELATIONAL_ANSWERS.names()})")
        kit = self.kit()
        stated = {f for fs in self.families.values() for f in fs}
        used = self.used_facts()
        problems = {
            "unknown families": sorted(set(self.families) - kit.families()),
            "families without prompts": sorted(set(self.families) - set(self.prompts)),
            "unsupported facts": sorted(stated - set(kit.facts)),
            "facts without a field": sorted(used - set(self.fields)),
            "list facts without a separator": sorted((kit.lists & used) - set(self.lists)),
            "separators of facts that are not lists": sorted(set(self.lists) - kit.lists),
            "fill kinds without a curve": sorted(set(kit.fill_kinds) - set(self.fact_curves)),
        }
        if any(problems.values()) or self.sampling not in ("uniform", "cluster_sqrt"):
            raise ValueError(f"relational stream: {problems}, sampling {self.sampling!r}")
        if (self.answer is None) != (self.instruction is None) or self.instruction == "":
            raise ValueError("relational stream: an answer convention takes a non-empty instruction, and only it")
        return self

    def policy(self) -> Answering | None:
        """Return the Yes/No-led families' answer policy (the convention's, with :attr:`instruction`), or None."""
        if self.answer is None or self.instruction is None:
            return None
        return dataclasses.replace(self.answer.policy(), instruction=self.instruction)

    def kit(self) -> RelationalAnswers:
        """Return the configured domain's relational answers."""
        return RELATIONAL_ANSWERS.get(self.answers)

    def used_facts(self) -> set[str]:
        """Return the facts the families state and the facts their answers read with them."""
        stated = {f for fs in self.families.values() for f in fs}
        kit = self.kit()
        return stated | {r for f in stated for r in kit.requires.get(f, ())}


def _draw_weights(ids: Sequence[str], sampling: str, domain: DomainView) -> dict[str, float]:
    """Each object's draw weight: 1, or 1/sqrt(its cluster's size among ``ids``) (``cluster_sqrt``)."""
    if sampling == "uniform":
        return dict.fromkeys(ids, 1.0)
    clusters = domain.clusters().reindex(list(ids))
    if clusters.isna().any():  # a NaN weight would poison the cumulative sums silently (as ObjectPool refuses)
        missing = list(clusters.index[clusters.isna()][:3])
        n = int(clusters.isna().sum())
        raise ValueError(f"relational stream: {n} {domain.noun}s without a cluster: {missing}")
    sizes = clusters.map(clusters.value_counts())
    return {o: 1.0 / float(np.sqrt(n)) for o, n in zip(ids, sizes, strict=True)}


@STREAMS.register("relational")
class RelationalStream:
    """Relational and long-form answers about one to three objects (see the module docstring)."""

    Params: ClassVar[type[Config]] = RelationalParams

    def __init__(self, domain: DomainView, params: Config, text: TrainingText) -> None:
        if not isinstance(params, RelationalParams):
            raise TypeError(f"expected RelationalParams, got {type(params).__name__}")
        self.params = params
        self._supervision = self.declare(domain, params)
        self.kit = params.kit()
        ids = population(domain, text.object_ids)
        used = params.used_facts()
        names = {f: field for f, field in params.fields.items() if f in used}  # the declared fields only
        records = {o: self.kit.record(o, text.fields.get(o, {}), names, params.lists) for o in ids}
        if self.kit.verify is not None:
            self.kit.verify(records, domain)
        self.pool = self.kit.pool(records, _draw_weights(ids, params.sampling, domain), params.p_hard)
        self.own_names: dict[str, str] = {}  # object → its screened name, when the config names the field
        if params.own_name is not None:
            for o in ids:
                name = text.fields.get(o, {}).get(params.own_name)
                if name and name.strip():
                    self.own_names[o] = name.strip()
        self.unnamed = len(ids) - len(self.own_names) if params.own_name is not None else 0
        self.families = tuple(params.families)
        self.policy = params.policy()
        self.grammar = Grammar(params.grammar) if params.grammar is not None else None
        self.offsets: TokenOffsets | None = None

    def derived(self) -> dict[str, object]:
        """Return what the stream derived from the training text.

        That is the count of objects without a screened name, which the own-name check cannot cover (nothing when the
        config names no name field).
        """
        return {"objects_without_own_name": self.unnamed} if self.params.own_name is not None else {}

    def bind(self, offsets: TokenOffsets) -> None:
        """Bind the renderer's token offsets (the sampler does, before any draw)."""
        self.offsets = offsets

    def sample(self, rng: np.random.Generator, n: int) -> list[Example]:
        """Draw ``n`` examples.

        Raises:
            RuntimeError: if the stream was not bound to a renderer, or no family yields a row.
        """
        return [self._row(rng) for _ in range(n)]

    def _row(self, rng: np.random.Generator) -> Example:
        for _ in range(_DRAWS * _ATTEMPTS):
            family = self.families[int(rng.integers(len(self.families)))]
            example = self._draw(family, rng)
            if example is not None:
                return example
        raise RuntimeError(f"relational stream: no row in {_DRAWS * _ATTEMPTS} family draws")

    def _draw(self, family: str, g: np.random.Generator) -> Example | None:
        """One row of ``family`` (the original ``generate`` with n = 1, then ``TemplateData.relational``), or None."""
        ok, P, kit = self.params.families[family], self.pool.P, self.kit
        for _ in range(_TRIES):
            compact: str | None = None
            ps: list[Any]
            if family in kit.single:
                ps = [P[self.pool.pick(g, self.pool.acc)]]
                answer = kit.single[family](ps, ok, False)
            elif family in kit.pair:
                a, b = self.pool.pair(g, family)
                ps = [P[a], P[b]]
                answer = kit.pair[family](ps, ok, False)
                if answer and family in kit.verdict_words:
                    compact = kit.pair[family](ps, ok, True)
            else:
                trio = self.pool.triplet(g, family)
                if not trio:
                    continue
                odd = int(g.integers(3))
                order = [0, 1]
                order.insert(odd, 2)
                ps = [P[trio[i]] for i in order]
                answer = kit.triple[family](ps, ok, False, odd)
                if answer:
                    compact = kit.triple[family](ps, ok, True, odd)
            if not answer:
                continue
            asked = self.policy is not None and family in kit.yes_no
            if asked:
                answer, compact = self._verdict(answer), (self._verdict(compact) if compact else compact)
            full = clean(answer)
            if len(full.text.split()) < _MIN_WORDS or self._names_an_object(full.text, ps):
                continue
            short = clean(compact) if compact else None
            prompts = self.params.prompts[family]
            prompt = prompts[int(g.integers(len(prompts)))]
            role = None
            if self.grammar is not None and g.random() < self.grammar.spec.rate:  # the grammar's lead and system role
                prompt, role = self.grammar.wrap_relational(g, prompt)
            return self._finish(family, prompt, [p.object_id for p in ps], full, short, role=role, asked=asked)
        return None

    def _verdict(self, marked: str) -> str:
        """Return a Yes/No-led answer with its leading verdict as the convention's answer text (module docstring)."""
        if self.policy is None:
            raise RuntimeError("relational stream: a verdict is replaced under an answer convention only")
        for label, word in ((True, "Yes"), (False, "No")):
            lead = fill(word, "choice")
            if marked.startswith(lead):
                return fill(self.policy.word(label), "choice") + marked[len(lead) :]
        raise ValueError(f"relational stream: a Yes/No-led answer opens with its verdict, not {clean(marked).text!r}")

    def _names_an_object(self, text: str, records: Sequence[RelationalRecord]) -> bool:
        """Whether ``text`` contains one of its objects' own names, as a whole word (the original ``names_in``)."""
        for r in records:
            name = self.own_names.get(r.object_id, "")
            if len(name) >= _MIN_NAME_CHARS and re.search(
                rf"(?<![\w-]){re.escape(name)}(?![\w-])", text, re.IGNORECASE
            ):
                return True
        return False

    def _finish(
        self,
        family: str,
        prompt: str,
        objects: Sequence[str],
        full: Clean,
        short: Clean | None,
        *,
        role: str | None = None,
        asked: bool = False,
    ) -> Example | None:
        """Cap the answer (compact form or a sentence cut) and weight its tokens, with the renderer's offsets."""
        if self.offsets is None:
            raise RuntimeError("relational stream: not bound to a renderer (Sampler binds it)")
        cap = self.params.max_target_tokens
        answer, offsets = full, self.offsets(full.text)
        if len(offsets) > cap and family in self.kit.verdict_words:
            if short is None or short.text == full.text:
                return None
            answer, offsets = short, self.offsets(short.text)
            if len(offsets) > cap:
                return None
        text = answer.text
        if len(offsets) > cap:
            within = text[: offsets[cap - 1][1]]
            cut = within.rfind(". ")
            text = within[: cut + 1] if cut > 0 else within
            offsets = self.offsets(text)[:cap]
        weights = self._weights(answer, offsets)
        system = (Message("system", role),) if role else ()
        slots = {f"p{i + 1}": o for i, o in enumerate(objects)}
        span = WeightedSpan(0, len(text), weights[0], tuple(weights))
        policy = self.policy if asked else None
        user = prompt if policy is None else policy.user(prompt)
        conversation = Conversation((*system, Message("user", user)), None if policy is None else policy.prefill())
        return Example(conversation, slots, text, (span,), "relational", cap)

    def _weights(self, answer: Clean, offsets: Sequence[tuple[int, int]]) -> list[float]:
        """Return each token's weight (the original ``token_fills`` and the per-fact-group relational weights).

        A token takes the first fill it overlaps; a fact token gets its kind's curve at its position among its
        group's fact tokens; every other token the floor.
        """
        floor, out = self.params.floor, []
        seen: defaultdict[int, int] = defaultdict(int)
        for start, end in offsets:
            hit = next(
                (j for j, (a, b) in enumerate(answer.spans) if a < max(end, start + 1) and b > start),
                None,
            )
            if hit is None:
                out.append(floor)
                continue
            seen[answer.groups[hit]] += 1
            curve = self.params.fact_curves[answer.kinds[hit]]
            out.append(floor if curve is None else curve.weights(seen[answer.groups[hit]])[-1])
        return out

    @classmethod
    def declare(cls, catalog: CatalogView, params: Config) -> Supervision:
        """Return the fact fields, the prompts and the answers' fixed wording (templates, verdicts, derived words).

        Raises:
            ValueError: if the answers' noun is not the domain's (their wording names the object by it).
        """
        if not isinstance(params, RelationalParams):
            raise TypeError(f"expected RelationalParams, got {type(params).__name__}")
        kit = params.kit()
        if kit.noun != catalog.noun:
            raise ValueError(
                f"relational answers {params.answers!r} say {kit.noun!r}; the domain's noun is {catalog.noun!r}"
            )
        own = {params.own_name} if params.own_name else set()
        fields = frozenset({params.fields[f] for f in params.used_facts()} | own)
        verdicts = [
            kit.verdict_sentence[f].format(w=w)
            for f in params.families
            if f in kit.verdict_words
            for w in kit.verdict_words[f].values()
        ]
        texts = (*(p for f in params.families for p in params.prompts[f]), *verdicts, *kit.wording)
        policy = params.policy()
        if policy is not None and kit.yes_no & set(params.families):
            texts = (*texts, *(t for t in (policy.instruction, policy.prefix, policy.yes, policy.no) if t))
        if params.grammar is not None:
            texts = (*texts, *Grammar(params.grammar).texts())
        tables = frozenset(t for f in params.used_facts() for t in kit.tables.get(f, ()))
        return Supervision(frozenset(), frozenset(), fields, texts, tables)

    def supervision(self) -> Supervision:
        """Return :meth:`declare`'s declaration."""
        return self._supervision
