"""Object domains: one contract that proteins, molecules and any new domain implement in full.

A domain owns everything about its objects that is not an embedding: the object table (ids, raw inputs, names), the
split, the clusters used for sampling and the bootstrap, the materialized label table, the text fields that training
targets are built from, the task catalog with each task's scope, and the prompt frames. Every evaluator and stream is
domain-agnostic: it asks the domain.

Domain *data* (object tables, splits, labels, fields, templates, frames, task catalog) is one artifact per domain
version, cited by digest through :class:`DomainSpec`, so every run and suite identity covers the exact questions,
frames and labels it used. Domain *code* implements :class:`Domain`, is constructed as ``cls(data_root: Path)`` on
that artifact's directory, and registers in :data:`DOMAINS`; :func:`open_domain` is the one way callers get the domain
a spec cites. The artifact format is ``domains.tables``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import ArtifactDigest, Config
from latent_interconnect.plugins import Registry, load_plugins

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "DOMAINS",
    "DOMAIN_DATA_KIND",
    "QUESTION_MARKER",
    "SLOT_MARKER",
    "CaptureRecipe",
    "Domain",
    "DomainSpec",
    "Frame",
    "RecipeMessage",
    "TaskKind",
    "TaskScope",
    "TaskSpec",
    "open_domain",
    "require_splits",
]

QUESTION_MARKER = "{question}"
"""Where a frame's user message takes the task's bare question (:meth:`Frame.ask`)."""
SLOT_MARKER = re.compile(r"<SLOT:[A-Za-z0-9_]+>")
"""A slot marker in a frame or a question."""

DOMAIN_DATA_KIND = "domain_data"
"""Artifact kind of a domain-data artifact (the directory a :class:`Domain` is constructed on)."""


class TaskKind(StrEnum):
    """How a task is answered and scored."""

    YES_NO = "yes_no"
    """Next-token log p(Yes) − log p(No); scored by AUROC against a binary label."""
    TEXT = "text"
    """Free text; scored by NLL of the reference (training targets, caption suites)."""


class TaskScope(StrEnum):
    """A task's relation to training."""

    TRAINED = "trained"
    HELD_OUT = "held_out"
    IMPLIED = "implied"
    """Not trained directly, but implied by a trained field (reported next to held-out, never pooled with it)."""


class TaskSpec(Config):
    """One evaluation or training task.

    Attributes:
        id: unique within the domain (e.g. ``"kw.ribosomal_protein"``).
        kind: answer kind.
        question: the bare question (no slot marker); the frame places it (:meth:`Frame.ask`). Empty for a task
            asked by the frame alone (e.g. a caption target under the ``describe`` frame).
        label: label-table column (YES_NO) or field name (TEXT).
        scope: trained / held-out / implied.
        group: the axis a figure pools it into (e.g. ``"lineage"``, ``"roles.metabolite"``).
    """

    id: str
    kind: TaskKind
    question: str
    label: str
    scope: TaskScope
    group: str = ""


class Frame(Config):
    """A prompt frame: the messages around a slot and a question.

    The frame carries the wording around the question. Without an answer convention it includes the answer
    instruction, so a Yes/No frame such as ``"Here is an unknown protein, given to you as an embedding:<SLOT:x>.
    {question} Answer Yes or No."`` renders the paper's evaluation message exactly once a task's bare question is
    placed; under one (``data.answers``) it carries only the task and object wording and the convention adds the rest.

    Attributes:
        id: frame id within the domain data (a suite's own frame: the stream template's id, ``SuiteSpec.frame``).
        system: the system message, if any.
        user: the user message, with one ``<SLOT:name>`` marker per slot (one for single-slot suites, several for
            ``composition``) and at most one :data:`QUESTION_MARKER`.
    """

    id: str
    system: str | None = None
    user: str

    @model_validator(mode="after")
    def _check(self) -> Frame:
        slots = SLOT_MARKER.findall(self.user)
        if not slots or len(set(slots)) != len(slots) or self.user.count(QUESTION_MARKER) > 1:
            raise ValueError(
                f"frame {self.id!r}: needs one or more distinct slot markers and at most one question marker"
            )
        return self

    def ask(self, question: str) -> str:
        """Return the user message with ``question`` in place of :data:`QUESTION_MARKER`.

        Raises:
            ValueError: if the frame has no marker but the question is not empty, or more than one marker.
        """
        count = self.user.count(QUESTION_MARKER)
        if count > 1 or (count == 0 and question):
            raise ValueError(f"frame {self.id!r} has {count} {QUESTION_MARKER} markers for question {question!r}")
        return self.user.replace(QUESTION_MARKER, question)


class RecipeMessage(Config):
    """One message of a capture recipe.

    Attributes:
        role: ``system``, ``user`` or ``assistant``.
        text: the text, with any of the placeholders ``{noun}``, ``{name}`` and ``{context}``.
    """

    role: Literal["system", "user", "assistant"]
    text: str


class CaptureRecipe(Config):
    """How one ridge-init target state is captured: a declared prompt and the tokens read.

    The capture stage renders the recipe for every object with the receiver's chat template and reads the receiver's
    states at the tokens of the ``read`` placeholder: the state at its last token is the stored target (or the mean,
    with ``target = "mean"``), and the mean over all its tokens is the ridge bias; the capture stores both, and a
    consumer may select the mean instead of the stored target (``build.ridge.RidgeSpec.pool``). A trailing assistant
    message is the prefill: the answer the receiver reads, with no end of turn. Every text that fills ``{context}``
    comes from the zero-shot-screened training text (``data.training_text``), so no zero-shot property enters a
    target. Recipes are domain data, so a capture's identity covers its exact prompt; a new target is a new recipe,
    never a new code path.

    Attributes:
        name: the target name connectors and ridges cite (``TokenSpec.target``), e.g. ``"name"``,
            ``"name_described"``, ``"caption"``.
        messages: the prompt.
        read: the placeholder whose tokens are read (``name`` or ``context``); it occurs exactly once.
        context_field: the screened training-text field that fills ``{context}``, or None for a recipe without one.
        context_tokens: the cap on the context in receiver tokens: whole sentences up to the cap; a first sentence
            longer than the cap is cut at its last word boundary within it. Required with ``context_field``.
        thinking: the receiver's thinking mode for this prompt.
        target: the read tokens' state the capture stores as its pooled state, which a ridge maps to by default:
            ``last`` (the last token; the name states) or ``mean`` (the mean over them; the original caption state).
    """

    name: str
    messages: tuple[RecipeMessage, ...] = Field(min_length=1)
    read: Literal["name", "context"]
    context_field: str | None = None
    context_tokens: int | None = Field(default=None, ge=1)
    thinking: bool = False
    target: Literal["last", "mean"] = "last"

    @model_validator(mode="after")
    def _check(self) -> CaptureRecipe:
        text = "".join(m.text for m in self.messages)
        if text.count("{" + self.read + "}") != 1:
            raise ValueError(f"recipe {self.name!r}: {{{self.read}}} must occur exactly once")
        has_context = "{context}" in text
        if has_context != (self.context_field is not None) or has_context != (self.context_tokens is not None):
            raise ValueError(f"recipe {self.name!r}: {{context}}, context_field and context_tokens go together")
        return self


class DomainSpec(Config):
    """A domain at a fixed data version, as cited by runs and suites.

    Attributes:
        name: registered domain name.
        data: digest of the domain-data artifact.
        plugins: modules to import before resolving ``name`` (third-party domains).
    """

    name: str
    data: ArtifactDigest
    plugins: tuple[str, ...] = ()


class Domain(ABC):
    """The contract every object domain implements.

    Attributes:
        name: registry name (``"protein"``, ``"molecule"``).
        noun: how prompts refer to an object (``"protein"``, ``"compound"``).
        object_kind: the kind of object, as encoder recipes name it (``EncoderRecipe.domain``): the encoders that
            can embed this domain's objects are those whose recipes name it. Declared by every domain: the protein
            domain and the PFMbench benchmark domain are both ``"protein"``, the molecule domain ``"molecule"``.
    """

    name: ClassVar[str]
    noun: ClassVar[str]
    object_kind: ClassVar[str]

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)

    @abstractmethod
    def objects(self) -> pd.DataFrame:
        """Return the object table: ``object_id`` (index), ``value`` (raw encoder input), ``name``."""

    @abstractmethod
    def split(self) -> pd.Series:
        """Return ``object_id`` → split (``"train"``, ``"val"``, ``"test"``), homology- or similarity-clean."""

    @abstractmethod
    def clusters(self) -> pd.Series:
        """Return ``object_id`` → cluster id (sampling weights; cluster bootstrap)."""

    @abstractmethod
    def labels(self) -> pd.DataFrame:
        """Return the materialized label table: ``object_id`` × label columns (bool / float, NaN = unknown)."""

    @abstractmethod
    def fields(self) -> pd.DataFrame:
        """Return the fields training targets are built from: ``object_id`` × field columns.

        A field is either a text field (each cell a ``str``) or an item-list field (each cell a ``tuple[str, ...]``
        of items, e.g. a molecule's parent classes, with no presentation wording: streams word them from their own
        config); a missing value is None. A capture recipe's context is a text field; a text task's label is a field
        of either kind (an item-list target, e.g. a molecule's parent families, is worded by the stream).
        """

    @abstractmethod
    def tasks(self) -> Mapping[str, TaskSpec]:
        """Return the task catalog by id."""

    @abstractmethod
    def frames(self) -> Mapping[str, Frame]:
        """Return the prompt frames by id."""

    @abstractmethod
    def capture_recipes(self) -> Mapping[str, CaptureRecipe]:
        """Return the declared capture recipes by name."""

    def table(self, name: str) -> pd.DataFrame | None:
        """Return the domain-specific extra table ``name`` (indexed by ``object_id``), or None (the default: none)."""
        del name
        return None

    def table_names(self) -> frozenset[str]:
        """Return the names of the domain's extra tables (the default: none)."""
        return frozenset()

    def population(self) -> pd.Index:
        """Return the objects population draws (evaluation item sets, sampling plans) draw from.

        By default every object; a domain with derived objects (the protein variants) leaves them out.
        """
        return self.objects().index

    @abstractmethod
    def scope_mask(self, task: TaskSpec, object_ids: pd.Index) -> pd.Series:
        """Return which objects a task may be scored on (e.g. molecules whose trained text does not state it)."""

    def training_mask(self, task: TaskSpec, object_ids: pd.Index) -> pd.Series:
        """Return which objects a task's label may train on when no task is held out (an all-tasks training set).

        The scope (:meth:`scope_mask`) without the rules that only keep a zero-shot score fair (the molecule domain's
        unstated subset); the rules that make a label unusable stay. By default the scope itself: a domain with such a
        rule overrides this.
        """
        return self.scope_mask(task, object_ids)


DOMAINS: Registry[type[Domain]] = Registry("domain")


def open_domain(spec: DomainSpec, store: ArtifactStore) -> Domain:
    """Return the domain a spec cites: its plugins loaded, constructed on its data artifact's directory.

    Raises:
        IntegrityError: if the data artifact's manifest does not match the digest.
        UnknownNameError: if no domain is registered under ``spec.name``.
    """
    load_plugins(spec.plugins)
    ref = ArtifactRef(spec.data, DOMAIN_DATA_KIND)
    store.manifest(ref)
    return DOMAINS.get(spec.name)(store.path(ref))


def require_splits(domain: Domain, object_ids: Iterable[str], allowed: Collection[str], use: str) -> None:
    """Raise unless every object is in one of the ``allowed`` splits of ``domain``.

    Args:
        domain: the domain whose split is checked.
        object_ids: the objects a computation uses.
        allowed: the splits it may use (layer selection: ``train`` only).
        use: what the objects are for, for the message.

    Raises:
        ValueError: naming the use and up to five offending objects with their splits (unknown ids included).
    """
    split = domain.split()
    bad = [(o, split.get(o, "unknown")) for o in object_ids if split.get(o) not in allowed]
    if bad:
        shown = ", ".join(f"{o} ({s})" for o, s in bad[:5])
        raise ValueError(f"{use}: {len(bad)} object(s) outside {sorted(allowed)}: {shown}")
