"""The variant stream (``variant``): single-substitution mutants in the slot, asked about their mutation.

The original variant stream, as data: the variants are objects of the domain (its
``variants`` table links each to its parent, ``domains.protein``), their effect texts are a screened training-text
field, and their Yes/No answers are label columns. Training draws use the training text's variant objects (the
training set holds only train variants of train proteins).

Kinds, drawn by :attr:`VariantParams.kinds` share:

- ``effect``: a variant with an effect text, one of the kind's templates; the answer is the effect text (capped);
- ``relational``: the same rows, two slots: the parent (wild type) then the variant; the answer is the effect text;
- ``clinical`` / ``functional``: Yes or No 50/50, then a variant with that label; the answer word. With
  :attr:`VariantParams.pair_frac` (kinds in :attr:`VariantParams.pair_kinds`) a draw instead takes a parent that has
  variants of both labels, the label 50/50, then a variant of that parent with it (within-protein balance, so a
  protein's label prevalence is no shortcut).

Variant rows are never scaffolded and carry no in-context examples (as in the original runs). Each kind's answer is
weighted by its loss curve, whose ``multiplier`` raises the stream's share of Σw (the original runs: 3.0).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.config import Config, added_field
from latent_interconnect.data.answers import AnswerConvention, Answering, answering
from latent_interconnect.data.streams import STREAMS, Example, LossCurve, Supervision
from latent_interconnect.data.training_text import TrainingText
from latent_interconnect.domains.views import CatalogView, DomainView
from latent_interconnect.receivers.render import SLOT_PATTERN, Conversation, Message

__all__ = ["VariantKind", "VariantParams", "VariantStream", "VariantTemplate"]

VariantKind = Literal["effect", "clinical", "functional", "relational"]
_YES_NO: tuple[VariantKind, ...] = ("clinical", "functional")
_WILD_TYPE, _VARIANT = "w", "x"
"""The slot names: a relational template has both (wild type first), every other template only the variant's."""


class VariantTemplate(Config):
    """One variant question template.

    Attributes:
        id: the template's id in the original variant templates.
        kind: its question kind.
        user: the user message: ``<SLOT:x>`` the variant (and, for ``relational``, ``<SLOT:w>`` its parent before it).
    """

    id: str
    kind: VariantKind
    user: str

    @model_validator(mode="after")
    def _check(self) -> VariantTemplate:
        slots = re.findall(SLOT_PATTERN, self.user)
        want = [_WILD_TYPE, _VARIANT] if self.kind == "relational" else [_VARIANT]
        if slots != want:
            raise ValueError(f"variant template {self.id!r} ({self.kind}) has slots {slots}, not {want}")
        return self


class VariantParams(Config):
    """Parameters of the ``variant`` stream.

    Attributes:
        table: the domain's variant table (``domains.protein.VARIANTS_TABLE``).
        templates: the training templates, by kind.
        kinds: kind → its share of the stream's rows (summing to 1).
        effect_field: the training-text field of the effect text.
        labels: Yes/No kind → its label column.
        answers: the inline answer words, ``yes`` and ``no``; None under ``answer``.
        max_target_tokens: the effect text's token cap.
        weights: kind → its loss curve.
        pair_frac: the share of a pair kind's draws balanced within a parent.
        pair_kinds: the Yes/No kinds the within-parent balance applies to.
        answer: the receiver's answer convention (``data.answers``) for the Yes/No kinds, whose templates then carry
            no answer instruction: it follows the template, its prefix opens the assistant turn and its answer text is
            the target. None: the inline policy (``answers``).
    """

    table: str
    templates: tuple[VariantTemplate, ...] = Field(min_length=1)
    kinds: dict[VariantKind, float]
    effect_field: str
    labels: dict[Literal["clinical", "functional"], str]
    answers: dict[Literal["yes", "no"], str] | None = added_field(default=None)
    max_target_tokens: int = Field(ge=1)
    weights: dict[VariantKind, LossCurve]
    pair_frac: float = Field(ge=0.0, le=1.0)
    pair_kinds: tuple[Literal["clinical", "functional"], ...] = ()
    answer: AnswerConvention | None = added_field(default=None)

    def policy(self) -> Answering:
        """Return the Yes/No kinds' answer policy (``data.answers.answering``: exactly one of the two)."""
        words = self.answers or {}
        return answering(self.answer, yes=words.get("yes"), no=words.get("no"), where="variant stream")

    @model_validator(mode="after")
    def _check(self) -> VariantParams:
        drawn = {k for k, share in self.kinds.items() if share > 0}
        if not np.isclose(sum(self.kinds.values()), 1.0) or min(self.kinds.values(), default=0.0) < 0:
            raise ValueError(f"variant kinds must be non-negative shares summing to 1, got {self.kinds}")
        missing = {
            "templates": sorted(drawn - {t.kind for t in self.templates}),
            "weights": sorted(drawn - set(self.weights)),
            "labels": sorted((drawn & set(_YES_NO)) - set(self.labels)),
        }
        if any(missing.values()) or (self.answers is not None and set(self.answers) != {"yes", "no"}):
            raise ValueError(f"variant stream: kinds drawn without {missing}, or answers not yes/no: {self.answers}")
        self.policy()
        return self


@STREAMS.register("variant")
class VariantStream:
    """Variant objects in the slot, asked about their mutation (see the module docstring)."""

    Params: ClassVar[type[Config]] = VariantParams

    def __init__(self, domain: DomainView, params: Config, text: TrainingText) -> None:
        if not isinstance(params, VariantParams):
            raise TypeError(f"expected VariantParams, got {type(params).__name__}")
        self.params = params
        self.policy = params.policy()
        self._supervision = self.declare(domain, params)
        link = domain.table(params.table)
        ours = [o for o in text.object_ids if o in link.index]
        self.parent: Mapping[str, str] = link["parent"].reindex(ours).to_dict()
        effect = [o for o in ours if text.fields.get(o, {}).get(params.effect_field, "").strip()]
        self.effect = {o: text.fields[o][params.effect_field] for o in effect}
        labels = domain.labels(list(params.labels.values()))
        self.rows: dict[tuple[str, bool | None], tuple[str, ...]] = {("text", None): tuple(effect)}
        self.by_parent: dict[str, dict[tuple[str, bool], tuple[str, ...]]] = {}
        for yes_no, column in params.labels.items():
            values = labels[column].reindex(ours)
            for want in (True, False):
                self.rows[(yes_no, want)] = tuple(o for o, v in values.items() if not pd.isna(v) and bool(v) is want)
            grouped: dict[tuple[str, bool], list[str]] = {}
            for want in (True, False):
                for o in self.rows[(yes_no, want)]:
                    grouped.setdefault((self.parent[o], want), []).append(o)
            self.by_parent[yes_no] = {k: tuple(v) for k, v in grouped.items()}
        for kind, share in params.kinds.items():
            keys = [("text", None)] if kind in ("effect", "relational") else [(kind, True), (kind, False)]
            if share > 0 and any(not self.rows[k] for k in keys):
                raise ValueError(f"variant kind {kind!r}: an empty training pool")
        self.pairs = {
            kind: tuple(
                sorted({p for p, w in self.by_parent[kind] if w} & {p for p, w in self.by_parent[kind] if not w})
            )
            for kind in params.pair_kinds
        }
        empty = [k for k, parents in self.pairs.items() if params.kinds.get(k, 0) > 0 and not parents]
        if empty and params.pair_frac > 0:
            raise ValueError(f"variant kinds {empty}: no training parent has variants of both labels")
        self.kind_order = tuple(k for k in params.kinds if params.kinds[k] > 0)
        shares = np.array([params.kinds[k] for k in self.kind_order])
        self.cumulative = np.cumsum(shares / shares.sum())
        self.templates = {k: tuple(t for t in params.templates if t.kind == k) for k in self.kind_order}

    def derived(self) -> dict[str, object]:
        """Return what this stream derived from its training text: nothing."""
        return {}

    def sample(self, rng: np.random.Generator, n: int) -> list[Example]:
        """Draw ``n`` examples."""
        return [self._draw(rng) for _ in range(n)]

    def bound_objects(self) -> frozenset[str]:
        """Return every object a row can bind.

        The variants and, in relational rows, their parents (not necessarily training-text objects), so the run can
        check each loads before training.
        """
        relational = self.params.kinds.get("relational", 0) > 0
        parents = {self.parent[o] for o in self.rows[("text", None)]} if relational else set()
        return frozenset(self.parent) | parents

    def _draw(self, rng: np.random.Generator) -> Example:
        p = self.params
        kind = self.kind_order[int(np.searchsorted(self.cumulative, rng.random(), side="right"))]
        if kind in ("effect", "relational"):
            pool = self.rows[("text", None)]
            oid = pool[int(rng.integers(len(pool)))]
            answer, cap = self.effect[oid], p.max_target_tokens
        else:
            if kind in self.pairs and p.pair_frac > 0 and rng.random() < p.pair_frac:
                parent = self.pairs[kind][int(rng.integers(len(self.pairs[kind])))]
                want = bool(rng.random() < 0.5)  # noqa: PLR2004  (the balanced label)
                pool = self.by_parent[kind][(parent, want)]
            else:
                want = bool(rng.random() < 0.5)  # noqa: PLR2004
                pool = self.rows[(kind, want)]
            oid = pool[int(rng.integers(len(pool)))]
            answer, cap = self.policy.word(want), None
        templates = self.templates[kind]
        template = templates[int(rng.integers(len(templates)))]
        objects = {_VARIANT: oid} | ({_WILD_TYPE: self.parent[oid]} if kind == "relational" else {})
        span = p.weights[kind].span(answer, cap or len(answer))
        if kind in _YES_NO:
            conversation = Conversation((Message("user", self.policy.user(template.user)),), self.policy.prefill())
        else:
            conversation = Conversation((Message("user", template.user),))
        return Example(conversation, objects, answer, (span,), "variant", cap, (), None, _VARIANT)

    @classmethod
    def declare(cls, catalog: CatalogView, params: Config) -> Supervision:
        """Return the label columns, the effect field and every template and answer word."""
        if not isinstance(params, VariantParams):
            raise TypeError(f"expected VariantParams, got {type(params).__name__}")
        policy = params.policy()
        users = (policy.user(t.user) if t.kind in _YES_NO else t.user for t in params.templates)
        texts = (*users, *(t for t in (policy.prefix, policy.yes, policy.no) if t))
        if params.table not in catalog.table_names():
            raise ValueError(f"the variant stream needs the domain's {params.table!r} table; {catalog.name!r} has none")
        labels = frozenset(params.labels.values())
        return Supervision(frozenset(), labels, frozenset({params.effect_field}), texts, frozenset({params.table}))

    def supervision(self) -> Supervision:
        """Return :meth:`declare`'s declaration."""
        return self._supervision
