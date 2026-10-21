"""Example streams: the open set of training-text generators, mixed by share in a run config.

A stream turns domain data and the screened training text into examples (:class:`Example`: a conversation with slot
markers, the objects bound to the slots, and a target with weighted spans). The built-in streams are
``data.qa_streams`` (``text``, ``yes_no``); ``relational`` and ``variant`` follow with their domain data
(``data.relational_stream``, ``data.variant_stream``). A new corpus or task family is a new registered stream
(``examples/guides/extending.md``).

Rules every stream keeps:

- **Screened text only.** A stream reads text from :class:`~latent_interconnect.data.training_text.TrainingText`
  alone, never from ``Domain.fields``, and asks only trained questions (``TaskScope.TRAINED``), so no
  zero-shot question or label is ever training content (``build.zero_shot.check_construction`` checks the mixture).
- **Determinism.** ``sample`` draws only from the generator it is given, so a step's examples are a function of the
  run seed and the step (``data.sampler``).
- **Declared supervision.** :meth:`Stream.declare` (and an instance's :meth:`Stream.supervision`, the same) states
  every task, label column, training-text field and fixed wording the stream can put into an example. A stream is
  built on the training text restricted to its declared fields (:func:`build_stream`), and
  ``build.zero_shot.check_construction`` refuses a mixture whose streams train a zero-shot task or label, read an
  unscreened field, or write wording the screen matches.
- **Label scope.** A stream is built on a :class:`ScopedDomain`: a label cell outside its task's scope
  (``Domain.scope_mask``) reads as unknown, so it is never a Yes and never a No, as a target, a stated fact or an
  in-context example.
- **No tokenizer.** Streams produce text and character spans; the renderer is the only module that tokenizes. Token
  caps are stated per target part (``receivers.render.TargetPart``) and applied by the renderer.

Shared pieces live here: :class:`LossCurve` (the positional loss-weight curves) and :class:`ObjectPool` (object
draws, uniform or cluster-weighted).
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.config import Config
from latent_interconnect.data.training_text import TrainingText
from latent_interconnect.domains.base import Domain, TaskKind
from latent_interconnect.domains.views import CatalogView, DomainCatalog, DomainView, RestrictedDomain
from latent_interconnect.plugins import Registry
from latent_interconnect.receivers.render import Conversation, TargetPart, WeightedSpan

__all__ = [
    "STREAMS",
    "AnswerLine",
    "Built",
    "Example",
    "LossCurve",
    "ObjectPool",
    "Sampling",
    "ScopedDomain",
    "SegmentBoost",
    "Stream",
    "Supervision",
    "boosted",
    "build_stream",
    "population",
    "restrict",
    "worded",
]

Sampling = Literal["uniform", "cluster_sqrt"]
"""How a stream draws objects: ``uniform`` over its objects, or ``cluster_sqrt``: a cluster with probability
proportional to the square root of its size among the stream's objects, then an object uniformly within it."""


@dataclass(frozen=True)
class Example:
    """One training example before rendering.

    Attributes:
        conversation: messages with ``<SLOT:name>`` markers.
        objects: slot name → object id.
        target: the supervised answer text.
        spans: loss weights over ``target``; the renderer maps them onto tokens.
        stream: the producing stream's name (for logging the mixture).
        max_target_tokens: the answer's token cap (cut by the renderer; no end-of-turn token either way).
        prefix: target parts supervised before the answer, each encoded on its own (e.g. a reasoning scaffold and its
            close, ``data.scaffold``); empty for a plain example.
        question: the trained task the example asks (a task id of the domain catalog), or None for a stream without
            one; scaffold eligibility and the mixture log read it.
        subject: the slot of the object the question is about (the query; in-context examples bind other slots).
        boost: the answer's segment-head boost, applied by the sampler with the renderer's token offsets, or
            None.
        context_question: the question a same-question in-context example asks (None: the task's example question).
        context_suffix: appended to that question (a grammar row's format instruction).
        context_format: the answer format a same-question example's answer is put in (None: verbatim).
    """

    conversation: Conversation
    objects: Mapping[str, str]
    target: str
    spans: tuple[WeightedSpan, ...]
    stream: str
    max_target_tokens: int | None = None
    prefix: tuple[TargetPart, ...] = ()
    question: str | None = None
    subject: str | None = None
    boost: SegmentBoost | None = None
    context_question: str | None = None
    context_suffix: str = ""
    context_format: str | None = None

    def parts(self) -> tuple[TargetPart, ...]:
        """Return the target as its parts in order: the prefix, then the answer."""
        return (*self.prefix, TargetPart(self.target, self.spans, self.max_target_tokens))


@dataclass(frozen=True)
class Supervision:
    """Everything a stream can put into an example, declared for the zero-shot construction check.

    Attributes:
        tasks: the task ids (domain catalog) whose questions or facts the stream trains.
        labels: the label columns (``Domain.labels``) it reads, as answers or as stated facts.
        fields: the training-text fields (``TrainingText``) it reads, as answers or as stated facts.
        texts: its fixed wording, verbatim: frames with their questions placed, answer words, instructions, stances
            and fact labels (the screen is run over them).
        tables: the domain's extra tables it reads (``DomainView.table``, e.g. ``variants``).
    """

    tasks: frozenset[str]
    labels: frozenset[str]
    fields: frozenset[str]
    texts: tuple[str, ...]
    tables: frozenset[str] = frozenset()

    def __or__(self, other: Supervision) -> Supervision:
        """Return the union of two declarations (a stream's own and its scaffold's)."""
        return Supervision(
            self.tasks | other.tasks,
            self.labels | other.labels,
            self.fields | other.fields,
            self.texts + other.texts,
            self.tables | other.tables,
        )


class Stream(Protocol):
    """An example generator.

    It declares first (``declare``, from a values-free ``domains.views.CatalogView``) and is then constructed as
    ``cls(domain, params, text)`` (:func:`build_stream`): ``domain`` a ``domains.views.DomainView`` restricted to its
    declared tasks, label columns and tables, ``params`` the validated ``Params`` from the mixture entry, and ``text``
    the screened training text restricted to its declared fields. An undeclared read raises ``KeyError`` naming it.

    Optional hooks the sampler calls when a stream has them: ``bind(offsets)`` (the renderer's token offsets, before
    any draw) and ``bound_objects()`` (objects it binds beyond the training text, checked at load). ``derived()`` is
    required (what it derived from the training text, recorded with the run; ``{}`` for none).
    """

    Params: ClassVar[type[Config]]

    @classmethod
    def declare(cls, catalog: CatalogView, params: Config) -> Supervision:
        """Return everything a stream with these params can put into an example, before it reads any data."""
        ...

    def __init__(self, domain: DomainView, params: Config, text: TrainingText) -> None: ...

    def sample(self, rng: np.random.Generator, n: int) -> list[Example]:
        """Return ``n`` examples drawn with ``rng`` (deterministic given its state)."""
        ...

    def derived(self) -> dict[str, object]:
        """Return what the stream derived from its training text (``{}`` for nothing), recorded with the run."""
        ...

    def supervision(self) -> Supervision:
        """Return everything this stream can put into an example (:class:`Supervision`)."""
        ...


STREAMS: Registry[type[Stream]] = Registry("example stream")
"""Stream classes by name."""


def population(domain: DomainView, object_ids: Sequence[str]) -> tuple[str, ...]:
    """Return the objects a single-object stream draws and counts cluster sizes over.

    That is ``object_ids`` in the domain's population (``Domain.population``, the one rule: the protein domain
    leaves out its variants). As the original runs drew and weighted proteins only: a variant carries its parent's
    cluster, so counting it would down-weight variant-rich clusters; the variant stream draws its own objects.
    """
    members = set(domain.population())
    return tuple(o for o in object_ids if o in members)


class _ObjectFields(Mapping[str, str]):
    """One object's fields, restricted: reading an undeclared field raises ``KeyError`` naming the reader."""

    def __init__(self, fields: Mapping[str, str], declared: frozenset[str], user: str) -> None:
        self._fields, self._declared, self._user = fields, declared, user

    def _check(self, field: object) -> None:
        if field not in self._declared:
            raise KeyError(f"{self._user} did not declare training-text field {field!r}")

    def __getitem__(self, field: str) -> str:
        self._check(field)
        return self._fields[field]

    def get(self, field: str, default: str | None = None) -> str | None:  # type: ignore[override]
        """Return the field's text, or ``default`` when the object lacks it (an undeclared field raises)."""
        self._check(field)
        return self._fields.get(field, default)

    def __contains__(self, field: object) -> bool:
        self._check(field)
        return field in self._fields

    def __iter__(self) -> Iterator[str]:
        return (f for f in self._fields if f in self._declared)

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _Fields(Mapping[str, Mapping[str, str]]):
    """Every object's fields, each restricted (:class:`_ObjectFields`), built on access."""

    def __init__(self, fields: Mapping[str, Mapping[str, str]], declared: frozenset[str], user: str) -> None:
        self._fields, self._declared, self._user = fields, declared, user

    def __getitem__(self, object_id: str) -> Mapping[str, str]:
        return _ObjectFields(self._fields[object_id], self._declared, self._user)

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)


def restrict(text: TrainingText, fields: Collection[str], user: str) -> TrainingText:
    """Return ``text`` as ``user`` may read it: only ``fields``; reading another raises ``KeyError`` naming ``user``."""
    return TrainingText(text.object_ids, _Fields(text.fields, frozenset(fields), user), text.removed)


@dataclass(frozen=True)
class Built:
    """A stream built on its views (:func:`build_stream`).

    Attributes:
        stream: the stream.
        domain: the domain view it reads (its declared tasks, label columns and tables).
        text: the training text it reads (its declared fields).
    """

    stream: Stream
    domain: DomainView
    text: TrainingText


class ScopedDomain(RestrictedDomain):
    """The view a stream is built on: a ``RestrictedDomain`` whose label columns honor the declared tasks' scope.

    A label cell outside its task's scope (``Domain.scope_mask``: a label the domain version declares unusable for
    that object) is neither Yes nor No. The label column of a declared Yes/No task therefore reads such a cell as
    unknown (a null), which every reader of the view keeps out of its Yes objects and its No objects alike: the
    ``yes_no`` stream's pools, the scaffold's stated labels and give-away counts, the in-context examples. A column
    whose tasks are scored on every object is returned as the domain holds it, and a column no declared task reads
    (the variant stream's) has no scope.
    """

    def labels(self, columns: Collection[str]) -> pd.DataFrame:
        """Return the label ``columns``, each cell outside its declared task's scope unknown.

        Raises:
            KeyError: naming the stream and the columns it did not declare.
            ValueError: if two declared tasks read one column on different scopes (one column cannot show both).
        """
        found = super().labels(columns)
        catalog = self._domain.tasks()
        yes_no = [t for t in sorted(self._tasks) if catalog[t].kind is TaskKind.YES_NO]
        scoped = {}
        for column in found.columns:
            reading = [t for t in yes_no if catalog[t].label == column]
            masks = [
                self._domain.scope_mask(catalog[t], found.index).reindex(found.index).fillna(False).to_numpy(bool)
                for t in reading
            ]
            if all(m.all() for m in masks):
                continue
            if any(not np.array_equal(m, masks[0]) for m in masks[1:]):
                raise ValueError(f"{self._user}: tasks {reading} read label column {column!r} on different scopes")
            scoped[column] = found[column].where(masks[0])
        if scoped:
            found = found.copy()  # the domain's own table is never written
            for column, values in scoped.items():
                found[column] = values
        return found


def build_stream(name: str, params: Mapping[str, Any], domain: Domain, text: TrainingText) -> Built:
    """Build a registered stream on views of the domain and the training text restricted to what it declares.

    The declaration (``Stream.declare``) comes first, from a values-free catalog of the domain and the params, so
    the stream (and the row transforms the sampler builds for it, which read the same views) cannot read a task,
    label column, table or field it did not declare. The zero-shot construction check builds streams the same way.
    The domain view honors label scope (:class:`ScopedDomain`).
    """
    cls = STREAMS.get(name)
    validated = cls.Params.model_validate(dict(params))
    sup = cls.declare(DomainCatalog(domain), validated)
    user = f"stream {name!r}"
    view = ScopedDomain(
        domain, tasks=sup.tasks, labels=sup.labels, tables=sup.tables, objects=text.object_ids, user=user
    )
    restricted = restrict(text, sup.fields, user)
    return Built(cls(view, validated, restricted), view, restricted)


class AnswerLine(Config):
    """How an item-list field's items are worded as an answer line (the stream's wording; the data hold the items).

    A line is ``one + item + end`` for one item and ``many + items joined by separator + end`` for more (e.g.
    ``"Parent families: a; b."``), the inverse of the domain's item-list reading (``domains.molecule.ItemList``).

    Attributes:
        split: the separator between the items in the training text (the training set's list separator).
        one: the line's head before a single item.
        many: the line's head before several items.
        separator: between items in the line.
        end: after the last item.
    """

    split: str = Field(min_length=1)
    one: str
    many: str
    separator: str = Field(min_length=1)
    end: str

    def render(self, text: str) -> str:
        """Return the answer line of a training-text value (its items, as the training set joined them)."""
        items = [i.strip() for i in text.split(self.split) if i.strip()]
        if not items:
            return ""
        return (self.many if len(items) > 1 else self.one) + self.separator.join(items) + self.end


def worded(lines: Mapping[str, AnswerLine], field: str, text: str) -> str:
    """Return a field's training text as an answer: its answer line if it is an item list, else the text itself."""
    rule = lines.get(field)
    return rule.render(text) if rule is not None and text.strip() else text


class SegmentBoost(Config):
    """The ';'-segment head boost of an answer (scheme e_meas; the original loss weights' ``segment_head_boost``).

    Each answer token right after a token whose text contains :attr:`marker` gets
    ``max(w, floor + (1 − floor)·r_f)``; every other token keeps its curve weight (:func:`boosted`).

    Attributes:
        r_f: the measured head ratio of the answer format (pathway 0.2109, catalytic 0.1647).
        floor: the curve's floor (``LossCurve.floor``).
        marker: the segment separator.
    """

    r_f: float = Field(ge=0.0, le=1.0)
    floor: float = Field(ge=0.0, le=1.0)
    marker: str = ";"

    @property
    def head(self) -> float:
        """The head weight ``floor + (1 − floor)·r_f``."""
        return self.floor + (1.0 - self.floor) * self.r_f


def boosted(
    text: str, offsets: Sequence[tuple[int, int]], weights: Sequence[float], boost: SegmentBoost
) -> tuple[float, ...]:
    """Return per-token weights with every segment head raised to the boost's head weight.

    Args:
        text: the answer.
        offsets: each token's ``[start, end)`` in ``text`` (``Renderer.target_token_offsets``, cut to the cap).
        weights: each token's curve weight.
        boost: the boost.

    A token's text is ``text[start:end]``. The original tested ``decode([id])`` of the previous token; the two agree,
    because ``";"`` is one byte, so no token splits it (a token that splits a multi-byte character decodes to U+FFFD,
    which never holds the marker).
    """
    out = list(weights)
    for i in range(1, len(out)):
        start, end = offsets[i - 1]
        if boost.marker in text[start:end]:
            out[i] = max(out[i], boost.head)
    return tuple(out)


class LossCurve(Config):
    """Per-token loss weights along an answer: ``w(t) = floor + (1 − floor)·g(t)``, t = 1 at its first token.

    Families: ``flat`` g = 1; ``exp`` g = exp(−(t − 1)/tau); ``power`` g = ((tau + 1)/(t + tau))^p; ``single_token``
    g = 1 at t = 1, 0 after. The fitted values per answer format are data (``configs/streams/``, from the original
    fitted loss weights).

    Attributes:
        family: the curve family.
        floor: the weight every token keeps.
        tau: the scale of ``exp`` and ``power`` (required by them, absent otherwise).
        p: the exponent of ``power`` (required by it, absent otherwise).
        multiplier: a factor on every weight of the answer (a stream's share of Σw, e.g. the variant stream's 3.0).
    """

    family: Literal["flat", "exp", "power", "single_token"]
    floor: float = Field(ge=0.0, le=1.0)
    tau: float | None = Field(default=None, gt=0.0)
    p: float | None = Field(default=None, gt=0.0)
    multiplier: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> LossCurve:
        need = {"flat": set(), "single_token": set(), "exp": {"tau"}, "power": {"tau", "p"}}[self.family]
        given = {k for k in ("tau", "p") if getattr(self, k) is not None}
        if given != need:
            raise ValueError(f"loss curve {self.family!r} takes {sorted(need) or 'no parameters'}, got {sorted(given)}")
        return self

    def weights(self, n: int) -> tuple[float, ...]:
        """Return the weights of tokens t = 1 .. n."""
        t = np.arange(1, n + 1, dtype=np.float64)
        if self.family == "flat":
            g = np.ones(n)
        elif self.family == "single_token":
            g = (t == 1).astype(np.float64)
        elif self.family == "exp":
            g = np.exp(-(t - 1) / float(self.tau or 1.0))
        else:
            tau = float(self.tau or 0.0)
            g = ((tau + 1) / (t + tau)) ** float(self.p or 1.0)
        return tuple(float(w) for w in self.multiplier * (self.floor + (1 - self.floor) * g))

    def span(self, text: str, n: int) -> WeightedSpan:
        """Return the span over all of ``text`` whose token i gets ``weights(n)[i]`` (tokens past n the last one)."""
        curve = self.weights(n)
        return WeightedSpan(0, len(text), curve[0], curve)


class ObjectPool:
    """Weighted draws of objects, from a fixed order (the training text's), so draws are reproducible.

    Each object's weight is fixed when the pool is built: 1 (``uniform``), or 1/sqrt(n_c) (``cluster_sqrt``, n_c the
    size of its cluster among the pool's objects). A :meth:`subset` keeps those weights, as the original sampler's
    Yes/No re-draw did (a label first, then a weighted object with that label).

    Args:
        object_ids: the stream's objects, in a fixed order.
        sampling: ``uniform`` or ``cluster_sqrt`` (:data:`Sampling`).
        clusters: object id → cluster id (``Domain.clusters``), read by ``cluster_sqrt``.

    Raises:
        ValueError: if the pool is empty, or ``cluster_sqrt`` meets an object without a cluster.
    """

    def __init__(self, object_ids: Sequence[str], sampling: Sampling, clusters: pd.Series) -> None:
        weight = np.ones(len(object_ids))
        if sampling == "cluster_sqrt":
            of = clusters.reindex(list(object_ids))
            if of.isna().any():
                raise ValueError(f"cluster_sqrt: objects without a cluster, e.g. {list(of.index[of.isna()][:3])}")
            weight = 1.0 / np.sqrt(of.map(of.value_counts()).to_numpy(dtype=np.float64))
        self._init(tuple(object_ids), weight)

    def _init(self, ids: tuple[str, ...], weight: np.ndarray) -> None:
        if not ids:
            raise ValueError("an object pool needs at least one object")
        self.ids = ids
        self._weight = weight
        self._cum = np.cumsum(weight / weight.sum())
        self._cum[-1] = 1.0

    def subset(self, object_ids: Collection[str]) -> ObjectPool:
        """Return the pool restricted to ``object_ids`` (in the pool's order), with the same per-object weights.

        Raises:
            ValueError: if none of them is in the pool.
        """
        keep = set(object_ids)
        rows = [i for i, o in enumerate(self.ids) if o in keep]
        out = ObjectPool.__new__(ObjectPool)
        out._init(tuple(self.ids[i] for i in rows), self._weight[rows])
        return out

    def probabilities(self) -> np.ndarray:
        """Return each object's draw probability, in :attr:`ids` order: its weight over the pool's total weight."""
        return np.asarray(self._weight / self._weight.sum(), dtype=np.float64)

    def weighted_order(self, rng: np.random.Generator) -> tuple[str, ...]:
        """Return the pool's objects in a weighted random order without replacement (Efraimidis–Spirakis keys)."""
        keys = rng.random(len(self.ids)) ** (1.0 / self._weight)
        return tuple(self.ids[i] for i in np.argsort(-keys, kind="stable"))

    def draw(self, rng: np.random.Generator) -> str:
        """Draw one object (weighted), with one uniform draw of ``rng``."""
        return self.ids[int(np.searchsorted(self._cum, rng.random(), side="right"))]
