"""The ``cross_domain`` suite kind: swap quads scored with a protein slot and a molecule slot in one prompt.

**Swap quads** (the frozen Rhea design, as implemented; kept for BindingDB). For two positive
pairs ``(P_i, M_i)`` and ``(P_j, M_j)`` the quad adds the two swapped pairs ``(P_i, M_j)`` and ``(P_j, M_i)`` as its
negatives: the same objects with only the pairing broken. Within a quad each positive is compared with each negative
(four comparisons), so any score that adds a protein term and a molecule term orders exactly half of them: a
single-slot readout gives a quad statistic of exactly 0.5, and only joint use of both slots moves it (E1,
``evaluation.cross_domain_quads``). The swapped cells are assumed, not measured, negatives.

**Item-set contract.** Two roles, one per slot (e.g. ``protein`` and ``molecule``), and the fields:

- :data:`QUAD_FIELD`: the item's quad;
- :data:`LABEL_FIELD`: 1 for a positive pair, 0 for a swapped one;
- the crossed resampling units the analysis names (``AnalysisSpec.units``): ``protein_cluster`` and
  ``molecule_group``, written with the item set when it is built, next to every set's ``cluster.<role>``.

Each quad is exactly four items with the swap structure above (two different objects per role); :func:`swap_quads`
checks it, and the runner refuses a set that breaks it before anything is scored.

**Secondary designs** (``CrossDomainParams.design: pairs``): labeled pairs without the quad structure, whose
analyses group the items themselves: the ranking sets and decoy pairs of E4 and E5 (``evaluation.cross_domain_ranks``)
and the substrate-versus-product task of E6 (``grouped_contrasts``). Only :data:`LABEL_FIELD` is required and checked
(:func:`pair_labels`). A pair that recurs across quads is
one item per occurrence; the runner scores each distinct prompt once and gives every occurrence that score.

**Prompts.** The user message is ``CrossDomainParams.user`` with each slot's sentence in place of ``{<slot name>}``;
a slot's sentence depends on its condition (``CrossDomainParams.sentences``), so a name or a SMILES string is
introduced as such (the frozen protocol's type-aware frames). There are no in-context examples. Every rendered
prompt is vetted (``evaluation.vetting``); a combination that puts an object's text in the prompt (``name``,
``value``) is exempt, as the name condition is in the single-slot kinds.

**Conditions** per role (the frozen design's 2 × 2 factorial and its diagnostics):

- ``own``: the item's object through the slot's connector;
- ``name``: the object's name (``Domain.objects`` ``name``) as its own text piece; nothing written;
- ``value``: the object's encoder input as text (``Domain.objects`` ``value``: a molecule's SMILES); nothing written;
- ``placeholder``: the connector's k placeholder tokens with nothing written (the omitted cells).

There is no shuffled condition: the swap contrast inside every quad is the test's own reference.
``elicited_name`` is refused.

**Readout and batching.** ``log p(yes) − log p(no)`` of the next token, as ``yes_no``; one backend call for all
distinct prompts, item-major (items in item-set order, then conditions in spec order), planned with the run's batch
plan and recorded as ``yes_no`` records it, the two answer log-probabilities included
(``scores.ANSWER_COLUMNS``). A non-finite score is refused when the table is written
(``scores.write_scores``; the frozen protocol's validity gate).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import model_validator

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.config import Config
from latent_interconnect.data.answers import answering
from latent_interconnect.data.items import ItemSet
from latent_interconnect.data.suites import Condition, SlotSpec, SuiteSpec
from latent_interconnect.domains.base import Domain
from latent_interconnect.evaluation.runners import PromptVettingError, slot_tokens
from latent_interconnect.evaluation.scores import write_scores
from latent_interconnect.evaluation.suites import SUITE_KINDS, SuiteContext, execution_devices
from latent_interconnect.evaluation.vetting import PromptRecord, vet_prompts
from latent_interconnect.receivers.base import Readout, ScoringBackend
from latent_interconnect.receivers.kernel import partition_digest, plan_batches, planned_lengths
from latent_interconnect.receivers.render import Conversation, Message, RenderedPrompt, Renderer, SlotBinding
from latent_interconnect.writes.connector import SlotWrites
from latent_interconnect.writes.layers import Entry
from latent_interconnect.writes.sources import WriteSource

__all__ = ["LABEL_FIELD", "QUAD_FIELD", "QUAD_SIZE", "CrossDomain", "CrossDomainParams", "pair_labels", "swap_quads"]

QUAD_FIELD = "quad"
"""Item-set field naming each item's swap quad."""
LABEL_FIELD = "pair_label"
"""Item-set field holding each item's pair label: 1 for a positive pair, 0 for a swapped one."""
QUAD_SIZE = 4
"""Items per swap quad: two positive pairs and their two swaps."""
_PAIR = 2  # roles per pair, positives per quad, objects per role in a quad
_TEXT_CONDITIONS = frozenset({Condition.NAME, Condition.VALUE})
_SERVED = frozenset({Condition.OWN, Condition.NAME, Condition.VALUE, Condition.PLACEHOLDER})
NO_WRITE = "__no_write__"
"""Connector name of the omitted (``placeholder``) slots: bound to a source with no writes, so the slot keeps its
placeholder state."""


class NoWrite:
    """A write source with no entries: slots bound to it keep the placeholder tokens' own states."""

    @property
    def bank(self) -> ArtifactRef | None:
        """No bank."""
        return None

    @property
    def entries(self) -> tuple[Entry, ...]:
        """None: nothing is written."""
        return ()

    def writes_for(self, object_ids: Sequence[str]) -> SlotWrites:
        """Return empty writes for the objects."""
        return SlotWrites(object_ids=tuple(object_ids), deltas={}, alphas={})


def pair_labels(items: ItemSet) -> tuple[int, ...]:
    """Return each item's pair label (:data:`LABEL_FIELD`), checked to be 0 or 1.

    Raises:
        KeyError: if the field is missing.
        ValueError: if a label is not 0 or 1.
    """
    if LABEL_FIELD not in items.fields:
        raise KeyError(f"item set {items.name!r} has no {LABEL_FIELD!r} field (the cross-domain item-set contract)")
    labels = items.fields[LABEL_FIELD]
    bad = sorted({repr(v) for v in labels if v not in (0, 1) or isinstance(v, bool)})
    if bad:
        raise ValueError(f"item set {items.name!r}: {LABEL_FIELD} must be 0 or 1, found {bad}")
    return tuple(1 if v == 1 else 0 for v in labels)


def swap_quads(items: ItemSet, roles: tuple[str, str]) -> np.ndarray:
    """Return each item's quad index (quads numbered in sorted order of their :data:`QUAD_FIELD` values).

    Args:
        items: the item set (fields :data:`QUAD_FIELD` and :data:`LABEL_FIELD`).
        roles: the two roles of the pairs, e.g. ``("protein", "molecule")``.

    Raises:
        KeyError: if a field or a role is missing.
        ValueError: if a label is not 0 or 1, or a quad is not two positive pairs of four distinct objects (two per
            role) and exactly their two swaps.
    """
    if QUAD_FIELD not in items.fields:
        raise KeyError(f"item set {items.name!r} has no {QUAD_FIELD!r} field (the cross-domain item-set contract)")
    labels = pair_labels(items)
    a, b = (items.objects[r] for r in roles)
    keys = [str(q) for q in items.fields[QUAD_FIELD]]
    index = {q: n for n, q in enumerate(sorted(set(keys)))}
    members: dict[str, list[int]] = {q: [] for q in index}
    for i, q in enumerate(keys):
        members[q].append(i)
    for q, rows in members.items():
        pos = sorted((a[i], b[i]) for i in rows if labels[i] == 1)
        neg = sorted((a[i], b[i]) for i in rows if labels[i] == 0)
        ok = len(rows) == QUAD_SIZE and len(pos) == len(neg) == _PAIR
        if ok:
            (pa, pb), (qa, qb) = pos
            ok = pa != qa and pb != qb and neg == sorted([(pa, qb), (qa, pb)])
        if not ok:
            raise ValueError(f"item set {items.name!r}: quad {q!r} is not a swap quad: positives {pos}, swaps {neg}")
    return np.array([index[q] for q in keys], dtype=np.int64)


class CrossDomainParams(Config):
    """Parameters of the ``cross_domain`` kind (``SuiteSpec.params``).

    Attributes:
        user: the user message: ``{<slot name>}`` once per slot where that slot's sentence goes, then the question
            and, without an answer convention (``SuiteSpec.answer``, which appends its own), the answer instruction
            (e.g. ``"{p} {m} Does this protein bind this compound? Answer Yes or No."``).
        sentences: slot name → condition → the slot's sentence under that condition, holding the slot's marker
            ``<SLOT:<slot name>>`` once (e.g. ``own``: ``"Here is an unknown protein, given to you as an
            embedding:<SLOT:p>."``; ``name``: ``"Here is a protein, given by its name:<SLOT:p>."``).
        yes: the inline positive answer word, read as one token; None under an answer convention.
        no: the inline negative answer word, as ``yes``.
        design: the item set's design: ``swap_quads`` (the E1 designs, checked by :func:`swap_quads` before scoring)
            or ``pairs`` (the secondary designs: labeled pairs, checked by :func:`pair_labels`).
    """

    user: str
    sentences: dict[str, dict[Condition, str]]
    yes: str | None = None
    no: str | None = None
    design: Literal["swap_quads", "pairs"] = "swap_quads"

    @model_validator(mode="after")
    def _check(self) -> CrossDomainParams:
        for slot, by_condition in self.sentences.items():
            if self.user.count("{" + slot + "}") != 1:
                raise ValueError(f"user message must hold {{{slot}}} once: {self.user!r}")
            for condition, text in by_condition.items():
                if text.count(f"<SLOT:{slot}>") != 1:
                    raise ValueError(f"slot {slot!r} sentence for {condition} must hold <SLOT:{slot}> once: {text!r}")
        return self


@dataclass(frozen=True)
class _Row:
    item: int
    combo: tuple[Condition, ...]
    objects: tuple[str, ...]
    label: float
    prompt: int


@SUITE_KINDS.register("cross_domain")
class CrossDomain:
    """``cross_domain``: swap quads, one prompt per item × condition combination (:class:`CrossDomainParams`)."""

    def __init__(self, spec: SuiteSpec) -> None:
        if len(spec.slots) != len({s.role for s in spec.slots}) or len(spec.slots) != _PAIR:
            raise ValueError(f"suite {spec.id!r}: cross_domain needs two slots with different roles")
        if len({s.connector for s in spec.slots}) != len(spec.slots):
            raise ValueError(f"suite {spec.id!r}: the two slots need different connectors")
        if spec.capability != "score":
            raise ValueError(f"suite {spec.id!r}: cross_domain needs the score capability, not {spec.capability!r}")
        if not spec.reads_connector():
            raise ValueError(
                f"suite {spec.id!r}: cross_domain reads each slot's token count; an all-name suite has none"
            )
        if spec.frame is not None:
            raise ValueError(f"suite {spec.id!r}: cross_domain takes its sentences from params, not frame")
        if len(spec.tasks) != 1:
            raise ValueError(f"suite {spec.id!r}: cross_domain scores exactly one pair task, got {list(spec.tasks)}")
        self.spec = spec
        self.slots: tuple[SlotSpec, ...] = spec.slots
        self.params = CrossDomainParams.model_validate(spec.params)
        p = self.params
        self.answering = answering(spec.answer, yes=p.yes, no=p.no, where=f"suite {spec.id!r}")
        roles = [s.role for s in self.slots]
        combos = []
        for c in spec.conditions:
            if set(c) != set(roles):
                raise ValueError(f"suite {spec.id!r}: condition {dict(c)} must name exactly the roles {roles}")
            combo = tuple(c[r] for r in roles)
            unserved = sorted(set(combo) - _SERVED)
            if unserved:
                raise ValueError(f"suite {spec.id!r}: cross_domain does not serve {unserved}")
            for slot, condition in zip(self.slots, combo, strict=True):
                if condition not in self.params.sentences.get(slot.name, {}):
                    raise ValueError(f"suite {spec.id!r}: no sentence for slot {slot.name!r} under {condition}")
            combos.append(combo)
        if set(self.params.sentences) != {s.name for s in self.slots}:
            raise ValueError(f"suite {spec.id!r}: sentences cover {sorted(self.params.sentences)}, not the slots")
        self.combos: tuple[tuple[Condition, ...], ...] = tuple(combos)

    def run(self, ctx: SuiteContext, sources: Mapping[str, WriteSource], step: int | None) -> ArtifactRef:
        """Score every item × condition combination and commit the score table (``SuiteRunner.run``).

        Raises:
            WrongHardwareError: if the backend runs on hardware other than ``ctx.device_class``.
            PromptVettingError: if a prompt's text depends on its item outside the text conditions.
            KeyError: if ``sources`` lacks a slot's connector, or the item set lacks a role or a contract field.
            ValueError: if the item set breaks its design (``CrossDomainParams.design``), the backend's plan is not
                ``ctx.plan``, or a score is not finite.
        """
        backend = ctx.backend
        if not isinstance(backend, ScoringBackend):
            raise TypeError(f"suite {self.spec.id!r}: backend {ctx.backend_name!r} cannot score")
        ctx.device_class.require(execution_devices(backend.execution()))
        if ctx.plan is None or backend.plan != ctx.plan:
            raise ValueError(f"suite {self.spec.id!r}: the backend's plan {backend.plan} is not the run's {ctx.plan}")
        items = ctx.items(self.spec.item_set)
        roles = (self.slots[0].role, self.slots[1].role)
        if self.params.design == "swap_quads":  # refuse a malformed design before anything is scored
            swap_quads(items, roles)
        else:
            pair_labels(items)
        domains = [self._domain(ctx, items, s.role) for s in self.slots]
        slot_sources = [sources[s.connector] for s in self.slots]
        rows, prompts = self._rows(ctx.renderer, items, domains, slot_sources)
        self._vet(rows, prompts)
        readout = Readout(kind="next_token", token_ids=self.answering.ids(ctx.renderer.token_id))
        writes: dict[str, WriteSource] = {s.connector: src for s, src in zip(self.slots, slot_sources, strict=True)}
        writes[NO_WRITE] = NoWrite()
        partition = plan_batches(planned_lengths(prompts, readout.kind), ctx.plan)
        batch = np.empty(len(prompts), dtype=np.int64)
        for b, members in enumerate(partition):
            batch[list(members)] = b
        lp = backend.score(prompts, writes, readout).detach().to("cpu").double().numpy()
        scores: np.ndarray = lp[:, 0] - lp[:, 1]
        plan = {
            "plan": ctx.plan.identity(),  # an added field at its default left out (config.added_field)
            "calls": [{"prompts": len(prompts), "partition": partition_digest(partition)}],
        }
        meta = {"suite": self.spec.digest(), "kind": self.spec.kind, "step": step, "batch_plan": plan}
        return write_scores(ctx.artifacts, self._table(rows, scores, batch, step, lp), meta)

    def _domain(self, ctx: SuiteContext, items: ItemSet, role: str) -> Domain:
        name = items.roles[role]
        specs = [d for d in self.spec.domains if d.name == name]
        if len(specs) != 1:
            raise ValueError(f"suite {self.spec.id!r}: role {role!r} is domain {name!r}, cited {len(specs)}x")
        return ctx.domain(specs[0])

    def _rows(
        self, renderer: Renderer, items: ItemSet, domains: Sequence[Domain], sources: Sequence[WriteSource]
    ) -> tuple[list[_Row], list[RenderedPrompt]]:
        columns = [items.objects[s.role] for s in self.slots]
        ks = [slot_tokens(src, col[0]) for src, col in zip(sources, columns, strict=True)]
        texts = [d.objects()[["name", "value"]] for d in domains]
        labels = items.fields[LABEL_FIELD]
        rows: list[_Row] = []
        prompts: list[RenderedPrompt] = []
        seen: dict[tuple[Any, ...], int] = {}
        for i in range(len(items)):  # item-major, then conditions in spec order
            objects = tuple(col[i] for col in columns)
            for combo in self.combos:
                # a placeholder's prompt does not depend on its object, so its key drops it
                shown = tuple("" if c == Condition.PLACEHOLDER else o for c, o in zip(combo, objects, strict=True))
                key = (combo, shown)
                if key not in seen:
                    seen[key] = len(prompts)
                    prompts.append(self._render(renderer, combo, objects, ks, texts))
                rows.append(_Row(i, combo, objects, 1.0 if labels[i] == 1 else 0.0, seen[key]))  # 0 or 1 (pair_labels)
        return rows, prompts

    def _render(
        self,
        renderer: Renderer,
        combo: tuple[Condition, ...],
        objects: tuple[str, ...],
        ks: Sequence[int],
        texts: Sequence[pd.DataFrame],
    ) -> RenderedPrompt:
        user = self.params.user
        bindings: dict[str, SlotBinding] = {}
        supplied: dict[str, str] = {}
        for slot, condition, obj, k, table in zip(self.slots, combo, objects, ks, texts, strict=True):
            user = user.replace("{" + slot.name + "}", self.params.sentences[slot.name][condition])
            if condition in _TEXT_CONDITIONS:
                supplied[slot.name] = str(table.at[obj, "name" if condition == Condition.NAME else "value"])
            else:
                connector = NO_WRITE if condition == Condition.PLACEHOLDER else slot.connector
                bindings[slot.name] = SlotBinding(object_id=obj, connector=connector, k=k)
        conversation = Conversation(
            messages=(Message("user", self.answering.user(user)),),
            prefill=self.answering.prefill(),
            thinking=self.spec.thinking,
        )
        return renderer.render(conversation, bindings, texts=supplied)

    def _label(self, combo: tuple[Condition, ...]) -> str:
        return ",".join(f"{s.role}={c}" for s, c in zip(self.slots, combo, strict=True))

    def _vet(self, rows: Sequence[_Row], prompts: Sequence[RenderedPrompt]) -> None:
        task = self.spec.tasks[0]
        records = [PromptRecord(task, self._label(r.combo), r.item, prompts[r.prompt].input_ids) for r in rows]
        exempt = [self._label(c) for c in self.combos if _TEXT_CONDITIONS & set(c)]
        findings = vet_prompts(records, name_conditions=exempt)
        if findings:
            raise PromptVettingError(f"suite {self.spec.id!r}: " + "; ".join(findings))

    def _table(
        self, rows: Sequence[_Row], scores: np.ndarray, batch: np.ndarray, step: int | None, logp: np.ndarray
    ) -> pd.DataFrame:
        index = np.array([r.prompt for r in rows], dtype=np.int64)
        table: dict[str, Any] = {
            "suite": self.spec.digest(),
            "step": step,
            "task": self.spec.tasks[0],
            "item": [r.item for r in rows],
            "score": scores[index],
            "label": [r.label for r in rows],
            "batch": batch[index],
            "logp_yes": logp[index, 0],
            "logp_no": logp[index, 1],
        }
        for n, slot in enumerate(self.slots):
            table[f"object.{slot.role}"] = [r.objects[n] for r in rows]
            table[f"filler.{slot.role}"] = ["" if r.combo[n] == Condition.PLACEHOLDER else r.objects[n] for r in rows]
            table[f"condition.{slot.role}"] = [str(r.combo[n]) for r in rows]
            table[f"connector.{slot.role}"] = None
        return pd.DataFrame(table)
