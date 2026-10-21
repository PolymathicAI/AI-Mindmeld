"""The zero-shot screen: the training set contains no text that states a zero-shot property.

The screen runs once, when the training set is built (:func:`build_training_set`), not in the sampler. Every rule of
it is in this module.

- **What is screened.** Every training-text field (the descriptions and other free-text fields were the only
  spoiling source), each in full. Truncation at sample time only removes text, so screening the full field is
  conservative.
- **What happens to a match** (the training set holds no text that states a zero-shot property):
  - a *free-text* field whose text states, for its object, the property of a declared zero-shot task is removed
    whole;
  - in a *list* field (``ScreenSpec.list_fields``, e.g. the keyword list), only the elements that state it are
    removed, which removes exactly the equivalent statement; a list left empty is removed. A list field's cells are
    either text joined by its separator or structured item lists (a list, tuple or array of strings per object, e.g.
    the molecule scaffold's fields); both are screened per element, and the kept elements are stored joined by the
    field's separator (the training text holds strings), so an item that contains the separator is refused.
  - in a list field whose elements are properties of the object (``ScreenSpec.property_list_fields``, e.g. the
    keywords), a zero-shot task's own label is always an element that states it (:func:`own_labels`), whatever the
    detectors say. Every other element is trained unless a measured detector removes it as an exact equivalent, by
    the same evidence rule as any other text (train on as much text as possible, as long as no zero-shot task is
    touched). The elements that must be read (:func:`element_read_required`) and are
    kept have their reads recorded (``ScreenSpec.kept_elements``, :func:`element_precision`).
- **What counts as stating the property.** Best-effort regex, alias and ontology-equivalent matching of *exact
  equivalents*, not correlates, by the registered detector kinds (:data:`EXPOSURE_DETECTORS`):
  - ``lexical`` (:class:`LexicalDetector`): the property's name, its exact synonyms and the names of its ontology
    descendants, and exact list elements;
  - ``numeric`` (:class:`NumericDetector`): for a length threshold, a stated residue count above it
    (``a 400-residue domain``).

  A detector reads the text, not whom it is about, so the same term can state the property in one field and mention
  another object in the next. Each detector is therefore scoped to the fields where it states the property
  (``DetectorSpec.fields``), and the shipped configs cite the measured precision behind every scope. Descendant tasks
  are declared once (``ScreenSpec.descendants``: Metazoa ⊃ Vertebrata ⊃ Mammalia): a task is also screened by
  its descendants' detectors, each in its own scope. A stream's fixed wording is always about the slot's object, so
  it is screened by every detector whose scope is empty or names :data:`WORDING`.
- **Which tasks.** Every held-out task is zero-shot and must be screened. An *implied* task
  (``TaskScope.IMPLIED``: answered by a trained field, e.g. a molecule's net charge or element content stated by its
  ChEBI classes) is not zero-shot and is not screened by default; a spec may still list it, and then its
  equivalents are removed like any other. A trained task is never zero-shot.
- **Recorded.** The training-set artifact's metadata holds the spec, the removed fields per task and field
  (``removed``, :class:`~latent_interconnect.data.training_text.TrainingText`), the removed list elements per task and
  field (``removed_elements``), the distinct elements removed per list field (``screened_elements``), and up to
  :data:`N_EXAMPLES` matched snippets per task (``examples``), for the paper's honesty statement. A field matched by
  several tasks counts under each.

Every other channel is controlled by construction: zero-shot questions and labels never appear as training
content. :func:`check_construction` asserts this for a run's training set and mixture: each stream declares what it
supervises (its ``supervision()``: tasks, label columns, training-text fields and its fixed wording), and a run is
refused when a stream trains a zero-shot task or a task related to one (:func:`find_conflicts`), reads a field the
training set did not screen, or writes wording a zero-shot detector matches (e.g. a template that names the species).
A conflicting template is untrained by default; the list is reviewed, and keeping a template makes that task no
longer zero-shot.

The three consumers of a training set's construction (``check_construction``, the trainer's domain check and the
logistic baseline's) read it through :func:`training_set_construction`: an ordinary training set's producing spec, or
the registry-bound construction definition of an approved release export (``build.metadata_export``), which never
claims to be what produced the training set. :func:`training_set_spec` stays the producing spec and refuses an export.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.build.metadata_export import MARKER, release_registry, resolve_release_export
from latent_interconnect.config import ArtifactDigest, Config, StageConfig, rationale
from latent_interconnect.data.items import ITEM_SET_KIND, load_item_set
from latent_interconnect.data.sampler import MixtureEntry
from latent_interconnect.data.streams import build_stream
from latent_interconnect.data.training_text import (
    TRAINING_SET_KIND,
    TrainingText,
    commit_training_text,
    load_training_text,
)
from latent_interconnect.domains.base import (
    Domain,
    DomainSpec,
    TaskKind,
    TaskScope,
    TaskSpec,
    open_domain,
    require_splits,
)
from latent_interconnect.plugins import Registry

__all__ = [
    "EXPOSURE_DETECTORS",
    "LOW_EVIDENCE_N",
    "MIN_PRECISION",
    "MIN_READ_SHARE",
    "NAME_FIELD",
    "N_EXAMPLES",
    "WORDING",
    "Conflict",
    "DetectorSpec",
    "ExposureDetector",
    "FieldEvidence",
    "LexicalDetector",
    "NumericDetector",
    "ReadSample",
    "ScreenSpec",
    "Screened",
    "TrainingSetSpec",
    "build_training_set",
    "check_construction",
    "detector_precision",
    "element_precision",
    "element_read_required",
    "find_conflicts",
    "own_labels",
    "read_sample",
    "screen_fields",
    "training_set_construction",
    "training_set_spec",
    "wording_screen",
]

N_EXAMPLES = 5
"""Matched snippets recorded per task in the training set's metadata."""
_SNIPPET = 60
NAME_FIELD = "name"
"""A training-text field read from the object table (``Domain.objects``'s ``name``), not from ``Domain.fields``: a
stream that shows an object's name (e.g. an in-context example) declares it, so the name is screened like any text."""
MIN_PRECISION = 0.90
"""A scope's first criterion: P(label | match) at least this (an exact equivalent sits near 1)."""
MIN_READ_SHARE = 0.80
"""A scope's second criterion: at least this share of a read sample states the property of the object itself.

Precision shows that the label co-occurs with a match, not that the text is about the object; the read sample shows
what the matches say. 0.80 (16 of 20) admits the odd incidental mention inside a field that is otherwise about the
object, and refuses any field where a fifth of the matches name something else."""
LOW_EVIDENCE_N = 10
"""A scope resting on at most this many matches is flagged low-evidence (``FieldEvidence.low_evidence``)."""
WORDING = "wording"
"""The scope name of a stream's fixed wording (templates, instructions) in ``DetectorSpec.fields``."""


def element_read_required(matches: int, precision: float) -> bool:
    """Return whether a property-list element must be read.

    It must when its precision against a zero-shot task reaches :data:`MIN_PRECISION`, or when it rests on at most
    :data:`LOW_EVIDENCE_N` matches and any of them is positive: on so few matches one label moves the precision by a
    tenth, so the precision cannot rule an equivalent out, and the read decides (an equivalent below the criterion is
    screened by an ``exception``).
    """
    return precision >= MIN_PRECISION or (matches <= LOW_EVIDENCE_N and precision > 0)


class ReadSample(Config):
    """A read sample of one detector's matches in one field (the second criterion of a scope).

    Attributes:
        n: matches read (up to 20, drawn with ``seed``).
        seed: the draw's seed (``detector_precision(..., sample=...)``).
        equivalent: matches that state the property of the object itself (a family defined by the property counts).
        correlate: matches that imply it only through something else (a family or class name, a route).
        other: matches that name another object (a substrate, partner, target, assay organism).
        note: what the matches say.
    """

    n: int = Field(ge=1)
    seed: int
    equivalent: int = Field(ge=0)
    correlate: int = Field(ge=0)
    other: int = Field(ge=0)
    note: str = ""

    @model_validator(mode="after")
    def _sum(self) -> ReadSample:
        if self.equivalent + self.correlate + self.other != self.n:
            raise ValueError(f"read sample: {self.equivalent} + {self.correlate} + {self.other} != n = {self.n}")
        return self

    @property
    def share(self) -> float:
        """Return the share of matches that state the property of the object."""
        return self.equivalent / self.n


class FieldEvidence(Config):
    """The measured evidence for one detector in one field (``detector_precision`` and a read sample).

    Attributes:
        matches: matches on the measured objects.
        precision: P(task label true | match).
        read: the read sample; required where the precision reaches :data:`MIN_PRECISION`.
        exception: why the rule's decision is reversed for this field (kept although a criterion fails, or dropped
            although both pass); empty when the rule decides.
    """

    matches: int = Field(ge=1)
    precision: float = Field(ge=0.0, le=1.0)
    read: ReadSample | None = None
    exception: str = ""

    @property
    def low_evidence(self) -> bool:
        """Return whether the scope rests on at most :data:`LOW_EVIDENCE_N` matches."""
        return self.matches <= LOW_EVIDENCE_N

    @property
    def states_property(self) -> bool:
        """Return whether the rule (without the exception) reads the matches as stating the property.

        Both criteria: precision at least :data:`MIN_PRECISION`, and a read sample with at least
        :data:`MIN_READ_SHARE` matches that state it.
        """
        return self.precision >= MIN_PRECISION and self.read is not None and self.read.share >= MIN_READ_SHARE

    @property
    def screened(self) -> bool:
        """Return whether the matches are screened: the rule's decision, reversed where an exception is stated."""
        return self.states_property != bool(self.exception)


class DetectorSpec(Config):
    """One detector applied to one zero-shot task.

    Attributes:
        task: the zero-shot task id (domain catalog).
        detector: registered detector kind.
        params: detector parameters, validated by the detector.
        fields: the training-text fields (and :data:`WORDING`, a stream's fixed wording) it screens; empty = all.
        evidence: field → the measurement its scope rests on (rationale: not part of the identity), one entry per
            field the detector matches on the measured objects. The rule: a field is in scope when its precision is at
            least :data:`MIN_PRECISION` and its read sample at least :data:`MIN_READ_SHARE` equivalents
            (``FieldEvidence.states_property``), reversed where an ``exception`` is stated; a detector with evidence
            must be scoped exactly so (checked at load), plus :data:`WORDING`. A detector whose scope is declared (e.g.
            the molecule roles) has no evidence.
    """

    task: str
    detector: str
    params: dict[str, Any] = Field(default_factory=dict)
    fields: tuple[str, ...] = ()
    evidence: dict[str, FieldEvidence] = rationale(default_factory=dict)

    @model_validator(mode="after")
    def _scope_follows_evidence(self) -> DetectorSpec:
        if not self.evidence:
            return self
        if not self.fields:
            raise ValueError(
                f"detector of {self.task!r} {self.params}: evidence without fields would screen every field"
            )
        expected = {f for f, e in self.evidence.items() if e.screened}
        scoped = set(self.fields) - {WORDING}
        if scoped != expected:
            raise ValueError(
                f"detector of {self.task!r} {self.params}: fields {sorted(scoped)} differ from what its evidence keeps"
                f" {sorted(expected)}"
            )
        return self


class ScreenSpec(Config):
    """The zero-shot screen of a domain's training text: what is screened, for which tasks, by which detectors.

    One screen serves every training set of a domain version (``TrainingSetSpec.screen``; in YAML a shared
    ``{$ref: ...}``), so training sets on different objects differ only in their objects and what depends on them.

    Attributes:
        fields: the training-text fields (each screened); the training set holds only these. :data:`NAME_FIELD`
            is the object's name from the object table, unless the domain has a text field of that name.
        list_fields: field → element separator, for the fields that are lists (e.g. ``keywords: ";"``); only their
            matching elements are removed. Every other field is free text, removed whole when it matches.
        property_list_fields: the list fields whose elements are properties of the object (an element present = the
            object has it, e.g. the keywords). Required, so that a spec states it (empty: no list is one): in these
            fields every zero-shot task's own label (:func:`own_labels`) is removed whatever the detectors say, and a
            trained Yes/No task's label value may not be screened (:func:`build_training_set`).
        element_parts: list field → the separator inside its elements (e.g. ``" > "`` in a class-graph edge
            ``"a > b"``): a detector that matches any part, or the whole element, removes the whole element.
        zero_shot_tasks: every task scored zero-shot; it must cover the domain's held-out tasks.
        descendants: zero-shot task → its descendant zero-shot tasks (membership in one implies the task), whose
            detectors also screen it, each in its own scope.
        detectors: the detectors per task (at least one per zero-shot task).
        kept_elements: property-list field → zero-shot task → element → its evidence (rationale): the elements that
            must be read against the task (:func:`element_read_required`, on :func:`element_precision`) and that the
            read keeps (a correlate, not a statement of the property). None may be one the rule screens: an element
            the rule screens is an ``elements`` detector of the task, with its evidence there. Measured on the
            domain's whole train split, so it holds for every training set's objects.
    """

    fields: tuple[str, ...] = Field(min_length=1)
    list_fields: dict[str, str] = Field(default_factory=dict)
    property_list_fields: tuple[str, ...]
    element_parts: dict[str, str] = Field(default_factory=dict)
    zero_shot_tasks: tuple[str, ...] = Field(min_length=1)
    descendants: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    detectors: tuple[DetectorSpec, ...] = Field(min_length=1)
    kept_elements: dict[str, dict[str, dict[str, FieldEvidence]]] = rationale(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ScreenSpec:
        if len(set(self.fields)) != len(self.fields) or len(set(self.zero_shot_tasks)) != len(self.zero_shot_tasks):
            raise ValueError("fields and zero_shot_tasks must not repeat")
        stray = sorted(set(self.list_fields) - set(self.fields))
        if stray or any(not sep for sep in self.list_fields.values()):
            raise ValueError(f"list_fields must be screened fields with a separator; not fields: {stray}")
        cited = {d.task for d in self.detectors}
        if cited - set(self.zero_shot_tasks):
            raise ValueError(f"detectors for tasks that are not zero-shot: {sorted(cited - set(self.zero_shot_tasks))}")
        if set(self.zero_shot_tasks) - cited:
            raise ValueError(f"zero-shot tasks without a detector: {sorted(set(self.zero_shot_tasks) - cited)}")
        if len(set(self.property_list_fields)) != len(self.property_list_fields):
            raise ValueError("property_list_fields must not repeat")
        if set(self.property_list_fields) - set(self.list_fields):
            stray = sorted(set(self.property_list_fields) - set(self.list_fields))
            raise ValueError(f"property_list_fields names fields that are not lists: {stray}")
        if set(self.element_parts) - set(self.list_fields) or not all(self.element_parts.values()):
            raise ValueError(f"element_parts must name list fields with a separator: {self.element_parts}")
        self._check_kept_elements()
        scopes = set(self.fields) | {WORDING}
        for d in self.detectors:
            if set(d.fields) - scopes:
                raise ValueError(f"detector of {d.task!r} is scoped to unknown fields {sorted(set(d.fields) - scopes)}")
        named = set(self.descendants) | {t for ts in self.descendants.values() for t in ts}
        if named - set(self.zero_shot_tasks):
            raise ValueError(
                f"descendants name tasks that are not zero-shot: {sorted(named - set(self.zero_shot_tasks))}"
            )
        for task in self.descendants:
            self.descendant_closure(task)  # refuses a cycle
        return self

    def _check_kept_elements(self) -> None:
        stray = sorted(set(self.kept_elements) - set(self.property_list_fields))
        if stray:
            raise ValueError(f"kept_elements names fields that are not property lists: {stray}")
        problems = []
        for field, by_task in self.kept_elements.items():
            unknown = sorted(set(by_task) - set(self.zero_shot_tasks))
            if unknown:
                raise ValueError(f"kept_elements of {field!r} name tasks that are not zero-shot: {unknown}")
            for task, by_element in by_task.items():
                for element, e in by_element.items():
                    if e.read is None or not element_read_required(e.matches, e.precision):
                        problems.append(f"{field} {task} {element!r}: a kept element is recorded only when read")
                    elif e.screened:
                        problems.append(f"{field} {task} {element!r}: its evidence screens it; make it a detector")
        if problems:
            raise ValueError("kept_elements:\n- " + "\n- ".join(problems))

    def descendant_closure(self, task: str) -> list[str]:
        """Return ``task``'s descendants, transitively, in declaration order.

        Raises:
            ValueError: on a cycle.
        """
        out: list[str] = []

        def walk(t: str, path: tuple[str, ...]) -> None:
            for child in self.descendants.get(t, ()):
                if child in path:
                    raise ValueError(f"descendants cycle: {' -> '.join((*path, child))}")
                if child not in out:
                    out.append(child)
                walk(child, (*path, child))

        walk(task, (task,))
        return out


class TrainingSetSpec(StageConfig):
    """A screened training set: a screen applied to a list of training objects.

    Attributes:
        domain: the domain at a fixed data version.
        train_ids: the training objects: an object list (``item_set``), all in the ``train`` split.
        screen: the screen (:class:`ScreenSpec`).
        screened_elements: list field → the distinct elements the screen removes from these training objects
            (rationale; it depends on ``train_ids``, so each training set records its own, and the slice test checks it
            against the data).
    """

    domain: DomainSpec
    train_ids: ArtifactDigest
    screen: ScreenSpec
    screened_elements: dict[str, tuple[str, ...]] = rationale(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> TrainingSetSpec:
        stray = sorted(set(self.screened_elements) - set(self.screen.list_fields))
        if stray:
            raise ValueError(f"screened_elements names fields that are not lists: {stray}")
        return self


class ExposureDetector(Protocol):
    """Decides whether a text states one task's property.

    Constructed as ``cls(domain, task, params)``.
    """

    def __init__(self, domain: Domain, task: TaskSpec, params: Mapping[str, Any]) -> None: ...

    def matches(self, object_id: str, text: str) -> list[tuple[int, int]]:
        """Return the character spans of free text that state the property (empty if none)."""
        ...

    def matches_element(self, object_id: str, element: str) -> bool:
        """Return whether one element of a list field states the property."""
        ...


EXPOSURE_DETECTORS: Registry[type[ExposureDetector]] = Registry("exposure detector")


class _LexicalParams(Config):
    terms: tuple[str, ...] = ()
    case_sensitive: tuple[str, ...] = ()
    elements: tuple[str, ...] = ()


@EXPOSURE_DETECTORS.register("lexical")
class LexicalDetector:
    """Exact names, synonyms and descendant names of a property, as regular expressions.

    Params:
        terms: patterns matched case-insensitively (write the word boundaries in the pattern).
        case_sensitive: patterns matched case-sensitively (symbols such as ``Mg(2+)``, taxon names).
        elements: list elements that state the property exactly (case-insensitive equality), besides elements a
            pattern matches.

    Raises (at construction):
        pydantic.ValidationError: for unknown params; ``re.error`` for a bad pattern.
        ValueError: if it has nothing to match.
    """

    Params: ClassVar[type[Config]] = _LexicalParams

    def __init__(self, domain: Domain, task: TaskSpec, params: Mapping[str, Any]) -> None:
        del domain
        p = _LexicalParams.model_validate(dict(params))
        if not (p.terms or p.case_sensitive or p.elements):
            raise ValueError(f"task {task.id!r}: a lexical detector needs terms or elements")
        self.patterns = [re.compile(t, re.IGNORECASE) for t in p.terms] + [re.compile(t) for t in p.case_sensitive]
        self.elements = frozenset(e.casefold() for e in p.elements)

    def matches(self, object_id: str, text: str) -> list[tuple[int, int]]:
        """Return the spans any pattern matches."""
        del object_id
        return sorted(m.span() for pattern in self.patterns for m in pattern.finditer(text))

    def matches_element(self, object_id: str, element: str) -> bool:
        """Return whether the element is a listed one or a pattern matches it."""
        return element.strip().casefold() in self.elements or bool(self.matches(object_id, element))


class _NumericParams(Config):
    threshold: int = Field(ge=0)


_AMINO_ACIDS = "Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val|Sec|Pyl"
_COUNT = re.compile(
    # a length, not a position: no range end ("330-346"), ratio ("777/804"), named residue ("Asp-827") or digit before
    r"(?<![\d,.])(?<!\d[-–/])(?<!\d -)(?<!\d to )(?<!\d and )(?<!(?:" + _AMINO_ACIDS + r")[- ])"
    r"\b(\d[\d,]*)(?:-residue\b|\s+residues\s+long\b|[- ]amino[- ]acids?\b)",
    re.IGNORECASE,
)


@EXPOSURE_DETECTORS.register("numeric")
class NumericDetector:
    """A stated length above a threshold: ``a 400-residue protein``, ``412 residues long``, ``1,200 amino acids``.

    Params:
        threshold: the task's length threshold in residues.

    A count of the object's own residues above the threshold entails "longer than ``threshold``"; the detector cannot
    tell whose residues are counted, so its scope rests on the measured precision. Residue
    positions are not matched (``Lys-412``, ``position 2552``, a range end ``330-346 aa``, a named residue
    ``Asp-827 residue``, a ratio ``777/804 residues``): in the descriptions they mostly name other molecules' residues
    (rRNA nucleotides, partners, substrates), so they do not entail the object's length. A count at or below the
    threshold entails nothing, and a mass in kDa only correlates with length.
    """

    Params: ClassVar[type[Config]] = _NumericParams

    def __init__(self, domain: Domain, task: TaskSpec, params: Mapping[str, Any]) -> None:
        del domain, task
        self.threshold = _NumericParams.model_validate(dict(params)).threshold

    def matches(self, object_id: str, text: str) -> list[tuple[int, int]]:
        """Return the spans of counts above the threshold."""
        del object_id
        return [m.span() for m in _COUNT.finditer(text) if int(m.group(1).replace(",", "")) > self.threshold]

    def matches_element(self, object_id: str, element: str) -> bool:
        """Return whether the element states a number above the threshold."""
        return bool(self.matches(object_id, element))


@dataclass(frozen=True)
class Conflict:
    """A trained task or label related to a zero-shot task.

    Attributes:
        trained: the trained task id or label column.
        zero_shot: the zero-shot task id.
        relation: ``"same"`` (the same label, or equal on every object where both are known), ``"ancestor"`` (the
            zero-shot property implies the trained one) or ``"descendant"`` (the trained property implies it).
        support: the positives of the implying property where both are known (0 for the same column).
    """

    trained: str
    zero_shot: str
    relation: str
    support: int = 0


@dataclass(frozen=True)
class Screened:
    """The result of screening (:func:`screen_fields`).

    Attributes:
        kept: object → field → kept text (a removed field is absent).
        removed: task → field → whole fields removed (free text, and lists left empty).
        removed_elements: task → field → list elements removed.
        examples: task → up to :data:`N_EXAMPLES` matches (``object``, ``field``, ``match``, ``context``).
        screened_elements: list field → the distinct elements removed (stripped, sorted).
    """

    kept: dict[str, dict[str, str]]
    removed: dict[str, dict[str, int]]
    removed_elements: dict[str, dict[str, int]]
    examples: dict[str, list[dict[str, str]]]
    screened_elements: dict[str, list[str]]


@dataclass(frozen=True)
class ScopedDetector:
    """A constructed detector with its field scope (:attr:`DetectorSpec.fields`).

    Attributes:
        detector: the detector.
        fields: the fields (and :data:`WORDING`) it screens; None = all.
    """

    detector: ExposureDetector
    fields: frozenset[str] | None = None

    def applies(self, field: str) -> bool:
        """Return whether it screens ``field``."""
        return self.fields is None or field in self.fields


def own_labels(task: TaskSpec) -> frozenset[str]:
    """Return the list elements that are ``task``'s own label: its id, and its label column's value after ``:``.

    A label column ``kw:Coiled coil`` gives ``Coiled coil``; a column without ``:`` (``len>320``) gives only the id.
    """
    names = {task.id}
    if ":" in task.label:
        names.add(task.label.partition(":")[2])
    return frozenset(n.strip() for n in names if n.strip())


def _detectors(spec: ScreenSpec, domain: Domain) -> dict[str, list[ScopedDetector]]:
    """Construct every detector of ``spec`` by task, each task's descendants' detectors included.

    Besides the declared detectors, each zero-shot task gets one that matches its own labels (:func:`own_labels`) as
    elements of the property lists (``ScreenSpec.property_list_fields``).
    """
    tasks = domain.tasks()
    unknown = sorted(set(spec.zero_shot_tasks) - set(tasks))
    if unknown:
        raise ValueError(f"zero-shot tasks not in the domain catalog: {unknown}")
    declared: dict[str, list[DetectorSpec]] = {t: [] for t in spec.zero_shot_tasks}
    for d in spec.detectors:
        declared[d.task].append(d)
    out: dict[str, list[ScopedDetector]] = {}
    for task in spec.zero_shot_tasks:
        lineage = (task, *spec.descendant_closure(task))
        out[task] = [
            ScopedDetector(
                EXPOSURE_DETECTORS.get(d.detector)(domain, tasks[task], d.params),
                frozenset(d.fields) if d.fields else None,
            )
            for t in lineage
            for d in declared[t]
        ]
        if spec.property_list_fields:
            out[task] += [
                ScopedDetector(
                    LexicalDetector(domain, tasks[t], {"elements": sorted(own_labels(tasks[t]))}),
                    frozenset(spec.property_list_fields),
                )
                for t in lineage
            ]
    return out


def _field_table(domain: Domain, fields: Sequence[str], object_ids: Sequence[str]) -> pd.DataFrame:
    """Return the training-text fields of the objects.

    The domain's text fields, and :data:`NAME_FIELD` from the object table when the domain has no text field of that
    name.

    Raises:
        ValueError: for a field the domain does not have.
    """
    table = domain.fields()
    unknown = sorted(set(fields) - set(table.columns) - {NAME_FIELD})
    if unknown:
        raise ValueError(f"fields not in the domain: {unknown}")
    ids = list(object_ids)
    columns = {}
    for f in fields:
        source = table if f in table.columns else domain.objects().rename(columns={"name": NAME_FIELD})
        columns[f] = source[f].reindex(ids)
    return pd.DataFrame(columns, index=pd.Index(ids, name=table.index.name))


def _snippet(text: str, span: tuple[int, int]) -> str:
    a, b = span
    return text[max(0, a - _SNIPPET) : b + _SNIPPET]


class _Tally:
    """Removal counts and examples while screening."""

    def __init__(self, tasks: Collection[str]) -> None:
        self.removed: dict[str, dict[str, int]] = {t: {} for t in tasks}
        self.elements: dict[str, dict[str, int]] = {t: {} for t in tasks}
        self.examples: dict[str, list[dict[str, str]]] = {t: [] for t in tasks}
        self.values: dict[str, set[str]] = {}

    def count(self, counts: dict[str, dict[str, int]], task: str, field: str, match: tuple[str, str, str]) -> None:
        counts[task][field] = counts[task].get(field, 0) + 1
        obj, found, context = match
        if len(self.examples[task]) < N_EXAMPLES:
            self.examples[task].append({"object": obj, "field": field, "match": found, "context": context})


def _screen_text(
    obj: str, field: str, value: str, detectors: Mapping[str, Sequence[ScopedDetector]], tally: _Tally
) -> str | None:
    """Return the free text if no detector matches it, else None (counted under every matching task)."""
    hit = False
    for task, dets in detectors.items():
        spans = [s for d in dets if d.applies(field) for s in d.detector.matches(obj, value)]
        if spans:
            hit = True
            a, b = spans[0]
            tally.count(tally.removed, task, field, (obj, value[a:b], _snippet(value, spans[0])))
    return None if hit else value


def _screen_list(
    obj: str,
    field: str,
    items: Sequence[str],
    detectors: Mapping[str, Sequence[ScopedDetector]],
    *,
    sep: str,
    tally: _Tally,
    parts: str | None = None,
) -> str | None:
    """Return the list without its matching elements, joined by ``sep`` (None if none is left; counted per task)."""
    value = sep.join(items)
    keep = []
    for element in items:
        if not element.strip():
            continue
        tasks = [t for t, dets in detectors.items() if any(_element(d, obj, field, element, parts) for d in dets)]
        for task in tasks:
            tally.count(tally.elements, task, field, (obj, element.strip(), value))
        if tasks:
            tally.values.setdefault(field, set()).add(element.strip())
        else:
            keep.append(element)
    if keep:
        return sep.join(keep)
    for task, dets in detectors.items():  # the list is gone: the whole field counts as removed for its tasks
        if any(_element(d, obj, field, e, parts) for d in dets for e in items if e.strip()):
            tally.removed[task][field] = tally.removed[task].get(field, 0) + 1
    return None


def _element(d: ScopedDetector, obj: str, field: str, element: str, parts: str | None = None) -> bool:
    return d.applies(field) and any(d.detector.matches_element(obj, e) for e in _pieces(element, parts))


def _pieces(element: str, parts: str | None) -> list[str]:
    """The element, and its parts when its field declares a part separator (``ScreenSpec.element_parts``)."""
    return [element, *(p for p in element.split(parts) if p.strip())] if parts else [element]


def _items(obj: str, field: str, value: object, sep: str | None) -> list[str] | None:
    """Return a cell as its text (one item) or its list items; None for an empty or missing cell.

    Raises:
        ValueError: for an item list in a field that is not a list, a non-string item, an item that contains the
            separator (it would split differently when read back), or a cell that is neither text nor a list.
    """
    if isinstance(value, str):
        if not value.strip():
            return None
        return [value] if sep is None else value.split(sep)
    if isinstance(value, list | tuple | np.ndarray):
        if sep is None:
            raise ValueError(f"{obj} {field}: an item list in a field not declared in list_fields")
        if not all(isinstance(v, str) for v in value):
            raise ValueError(f"{obj} {field}: list items must be strings")
        joined = [str(v) for v in value if sep in str(v)]
        if joined:
            raise ValueError(f"{obj} {field}: item {joined[0]!r} contains the separator {sep!r}")
        return [str(v) for v in value] or None
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    raise ValueError(f"{obj} {field}: a cell must be text or a list of strings, not {type(value).__name__}")


def screen_fields(
    fields: pd.DataFrame,
    list_fields: Mapping[str, str],
    detectors: Mapping[str, Sequence[ScopedDetector]],
    element_parts: Mapping[str, str] | None = None,
) -> Screened:
    """Screen a field table (``object_id`` index × field columns; NaN or empty = no text).

    A free-text field that any task's detector matches is removed whole and counted under every matching task; in a
    list field (separator ``list_fields[field]``) each matching element is removed (counted per task) and the rest is
    kept in order, joined by the same separator; a list left empty is removed and counted as a field. A list field's
    cell may also be a structured item list (list, tuple or array of strings), screened the same way. Each detector
    screens only the fields of its scope. In a field of ``element_parts`` each element's parts are screened too, and a
    match on any part removes the whole element.

    Raises:
        ValueError: as the cells allow (an item list in a field that is not a list, a non-string item, an item that
            contains the separator, or a cell that is neither text nor a list).
    """
    kept: dict[str, dict[str, str]] = {}
    tally = _Tally(list(detectors))
    for field in fields.columns:
        f = str(field)
        sep = list_fields.get(f)
        for obj, value in fields[field].items():
            o = str(obj)
            items = _items(o, f, value, sep)
            if items is None:
                continue
            if sep is None:
                out = _screen_text(o, f, items[0], detectors, tally)
            else:
                parts = (element_parts or {}).get(f)
                out = _screen_list(o, f, items, detectors, sep=sep, tally=tally, parts=parts)
            if out is not None:
                kept.setdefault(o, {})[f] = out
    screened = {f: sorted(v) for f, v in sorted(tally.values.items())}
    return Screened(kept, tally.removed, tally.elements, tally.examples, screened)


def build_training_set(spec: TrainingSetSpec, store: ArtifactStore) -> ArtifactRef:
    """Screen every training field and commit the ``training_set`` artifact.

    The artifact holds the training objects and their kept texts (``data.training_text``); its metadata holds the
    spec (``spec``, its ``spec_digest``), ``removed``, ``removed_elements``, ``screened_elements`` and ``examples``
    (:class:`Screened`).

    Raises:
        ValueError: if a training object is outside the ``train`` split, a held-out task of the domain is not
            zero-shot here, a trained task is, a detector screens a trained Yes/No task's label value in a property
            list (the trained label would lose its text), a zero-shot task or a field is unknown to the domain, or a
            detector is misconfigured.
    """
    domain = open_domain(spec.domain, store)
    ids = load_item_set(store, ArtifactRef(spec.train_ids, ITEM_SET_KIND)).object_ids
    require_splits(domain, ids, {"train"}, "the training set")
    held_out = [t.id for t in domain.tasks().values() if t.scope == TaskScope.HELD_OUT]
    screen = spec.screen
    missing = sorted(set(held_out) - set(screen.zero_shot_tasks))
    if missing:
        raise ValueError(f"held-out tasks of the domain that are not screened as zero-shot: {missing}")
    trained = sorted(
        t for t in screen.zero_shot_tasks if t in domain.tasks() and domain.tasks()[t].scope == TaskScope.TRAINED
    )
    if trained:
        raise ValueError(f"trained tasks cannot be zero-shot: {trained}")
    detectors = _detectors(screen, domain)
    lost = _screened_trained_labels(screen, domain, detectors)
    if lost:
        raise ValueError("detectors screen trained Yes/No labels in property lists:\n- " + "\n- ".join(lost))
    screened = screen_fields(
        _field_table(domain, screen.fields, ids), screen.list_fields, detectors, screen.element_parts
    )
    text = TrainingText(ids, screened.kept, screened.removed)
    meta = {
        "spec": spec.model_dump(mode="json"),
        "spec_digest": spec.digest(),
        "removed_elements": screened.removed_elements,
        "screened_elements": screened.screened_elements,
        "examples": screened.examples,
    }
    return commit_training_text(store, text, meta)


def _screened_trained_labels(
    spec: ScreenSpec, domain: Domain, detectors: Mapping[str, Sequence[ScopedDetector]]
) -> list[str]:
    """Return each (trained Yes/No task, zero-shot task, field) where a detector screens the trained label's value.

    A trained Yes/No task whose label column is ``prefix:value`` (``kw:Kinase``) is trained as that value in the
    property lists; a detector that matches the value as an element there would remove the trained label's text.
    """
    out = []
    for t in domain.tasks().values():
        if t.kind != TaskKind.YES_NO or t.scope != TaskScope.TRAINED or ":" not in t.label:
            continue
        value = t.label.partition(":")[2]
        for field in spec.property_list_fields:
            for z, dets in detectors.items():
                if any(_element(d, "", field, value) for d in dets):
                    out.append(f"trained {t.id!r} ({value!r} in {field}) is screened for zero-shot {z!r}")
    return out


def detector_precision(
    spec: ScreenSpec,
    domain: Domain,
    object_ids: Sequence[str],
    proxies: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Measure every detector of ``spec`` in every field: how often a match comes with a true label.

    The evidence behind a detector's scope (``DetectorSpec.fields``): an exact equivalent of a property sits near
    precision 1, a mention that carries no information at the label's base rate. Each detector is run unscoped on
    the given objects' fields (a list field matches when any element does); its own task's label is read from the
    domain, through ``proxies`` (object → the object whose label stands in, e.g. a variant's parent) where it is
    unknown.

    Returns:
        One row per detector and field with at least one match: ``detector`` (its index in ``spec.detectors``),
        ``task``, ``field``, ``matches``, ``known`` (matches with a known label), ``positives`` and ``precision``
        (positives / known); and one row per task with ``field = "(base rate)"``: the label's base rate over the
        objects.
    """
    tasks = domain.tasks()
    table = _field_table(domain, spec.fields, object_ids)
    rows: list[dict[str, Any]] = []
    for task in spec.zero_shot_tasks:
        label = _task_label(domain, task, table.index, proxies)
        known = label.dropna()
        rows.append(
            {
                "detector": -1,
                "task": task,
                "field": "(base rate)",
                "matches": 0,
                "known": len(known),
                "positives": int(known.sum()),
                "precision": float(known.mean()) if len(known) else float("nan"),
            }
        )
        for i, d in enumerate(spec.detectors):
            if d.task != task:
                continue
            det = EXPOSURE_DETECTORS.get(d.detector)(domain, tasks[task], d.params)
            for field in spec.fields:
                sep = spec.list_fields.get(field)
                hits = []
                for obj, value in table[field].items():
                    items = _items(str(obj), field, value, sep)
                    if items is None:
                        continue
                    if sep is None:
                        hit = bool(det.matches(str(obj), items[0]))
                    else:
                        parts = spec.element_parts.get(field)
                        hit = any(
                            det.matches_element(str(obj), x) for e in items if e.strip() for x in _pieces(e, parts)
                        )
                    if hit:
                        hits.append(obj)
                if not hits:
                    continue
                y = label.reindex(hits).dropna()
                rows.append(
                    {
                        "detector": i,
                        "task": task,
                        "field": field,
                        "matches": len(hits),
                        "known": len(y),
                        "positives": int(y.sum()),
                        "precision": float(y.mean()) if len(y) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _task_label(domain: Domain, task: str, index: pd.Index, proxies: Mapping[str, str] | None) -> pd.Series:
    """Return ``task``'s label on ``index`` as a nullable boolean, read through ``proxies`` where it is unknown.

    Raises:
        ValueError: if the task has no binary label column.
    """
    column = _label_of(domain, task)
    binary = None if column is None else _binary(domain.labels()[column])
    if binary is None:
        raise ValueError(f"zero-shot task {task!r} has no binary label column")
    label = binary.reindex(index)
    if proxies:
        stand_in = pd.Series({o: proxies[o] for o in index if o in proxies}, dtype=object)
        if len(stand_in):
            fill = pd.Series(binary.reindex(stand_in.to_numpy()).to_numpy(), stand_in.index)
            label.loc[stand_in.index] = label.loc[stand_in.index].fillna(fill)
    return label


def element_precision(
    spec: ScreenSpec,
    domain: Domain,
    object_ids: Sequence[str],
    field: str,
    proxies: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Measure every element of a list field against every zero-shot task, as :func:`detector_precision` measures.

    The precision of an ``elements`` detector of one element, for all elements at once: on the given objects, an
    element matches an object whose ``field`` holds it (stripped, case-insensitive, as :class:`LexicalDetector`
    compares elements), and the task's label is read from the domain, through ``proxies`` where it is unknown. This
    finds the elements a read must classify (``ScreenSpec.kept_elements``).

    Returns:
        One row per element and zero-shot task: ``element`` (its first spelling), ``task``, ``matches``, ``known``,
        ``positives`` and ``precision`` (positives / known; NaN when no label is known).

    Raises:
        ValueError: if ``field`` is not a list field of ``spec``, or a zero-shot task has no binary label column.
    """
    sep = spec.list_fields.get(field)
    if sep is None:
        raise ValueError(f"{field!r} is not a list field of the training set")
    table = _field_table(domain, [field], object_ids)
    rows_of: dict[str, list[int]] = {}
    spelling: dict[str, str] = {}
    for row, (obj, value) in enumerate(table[field].items()):
        for element in {e.strip() for e in _items(str(obj), field, value, sep) or () if e.strip()}:
            key = element.casefold()
            spelling.setdefault(key, element)
            rows_of.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for task in spec.zero_shot_tasks:
        label = _task_label(domain, task, table.index, proxies)
        known = label.notna().to_numpy()
        positive = label.fillna(False).to_numpy(dtype=bool) & known
        for key, rows in rows_of.items():
            idx = np.unique(np.asarray(rows))
            with_label = idx[known[idx]]
            hits = int(positive[with_label].sum())
            out.append(
                {
                    "element": spelling[key],
                    "task": task,
                    "matches": len(idx),
                    "known": len(with_label),
                    "positives": hits,
                    "precision": hits / len(with_label) if len(with_label) else float("nan"),
                }
            )
    return pd.DataFrame(out, columns=["element", "task", "matches", "known", "positives", "precision"])


def read_sample(
    spec: ScreenSpec,
    domain: Domain,
    object_ids: Sequence[str],
    *,
    detector: int,
    field: str,
    n: int = 20,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Draw up to ``n`` matches of one detector in one field, for reading (:class:`ReadSample`).

    The objects whose ``field`` the detector (``spec.detectors[detector]``, unscoped) matches are drawn uniformly with
    ``numpy.random.default_rng(seed)``; each draw gives the ``object``, the ``match`` and its ``context``.
    """
    d = spec.detectors[detector]
    det = EXPOSURE_DETECTORS.get(d.detector)(domain, domain.tasks()[d.task], d.params)
    sep = spec.list_fields.get(field)
    found: list[dict[str, str]] = []
    for obj, value in _field_table(domain, [field], object_ids)[field].items():
        items = _items(str(obj), field, value, sep)
        if items is None:
            continue
        if sep is None:
            spans = det.matches(str(obj), items[0])
            if spans:
                a, b = spans[0]
                found.append({"object": str(obj), "match": items[0][a:b], "context": _snippet(items[0], spans[0])})
        else:
            parts = spec.element_parts.get(field)
            hits = [
                e.strip()
                for e in items
                if e.strip() and any(det.matches_element(str(obj), x) for x in _pieces(e, parts))
            ]
            if hits:
                found.append({"object": str(obj), "match": hits[0], "context": sep.join(items)})
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(len(found), size=min(n, len(found)), replace=False)) if found else []
    return [found[int(i)] for i in picks]


def training_set_spec(store: ArtifactStore, training_set: ArtifactRef) -> TrainingSetSpec:
    """Return the spec a training set was built from (its domain, objects, fields and screen).

    Raises:
        ValueError: if the artifact is not a training set built by :func:`build_training_set`, or is a release export
            (``build.metadata_export``: its source production spec is unavailable; :func:`training_set_construction`
            reads its construction definition).
    """
    if training_set.kind != TRAINING_SET_KIND:
        raise ValueError(f"{training_set.short} is a {training_set.kind!r} artifact, not a training set")
    meta = store.manifest(training_set).meta
    if MARKER in meta:
        raise ValueError(
            f"training set {training_set.short} is a release export: its source production spec is unavailable"
        )
    if "spec" not in meta:
        raise ValueError(f"training set {training_set.short} was not built by build_training_set (no spec)")
    return TrainingSetSpec.model_validate(meta["spec"])


def training_set_construction(store: ArtifactStore, training_set: ArtifactRef) -> TrainingSetSpec:
    """Return what a training set's construction is checked against: its domain, objects, fields and screen.

    An ordinary training set (historical or newly built) gives its producing spec (:func:`training_set_spec`), without
    reading the registry. A training set whose metadata holds the export marker (``build.metadata_export.MARKER``)
    gives the construction definition of an approved release export, only after every registry binding holds
    (``build.metadata_export.resolve_release_export``); it is not what produced the training set.

    Raises:
        ValueError: if the artifact is not a training set, an ordinary one has no spec, or an export (or an id the
            registry lists) fails a binding; an export never falls back to a producing spec.
    """
    if training_set.kind != TRAINING_SET_KIND:
        raise ValueError(f"{training_set.short} is a {training_set.kind!r} artifact, not a training set")
    manifest = store.manifest(training_set)
    if MARKER in manifest.meta:
        definition = resolve_release_export(store, training_set, manifest)
        try:
            return TrainingSetSpec.model_validate(definition)
        except ValueError as e:
            raise ValueError(
                f"release export {training_set.short}: its definition is not a valid construction: {e}"
            ) from e
    if training_set.sha256 in _registered():
        raise ValueError(f"training set {training_set.short} is a registered release export without its marker")
    return training_set_spec(store, training_set)


def _registered() -> frozenset[str]:
    """Return the registered export ids, or none when the registry cannot be read (an ordinary set needs none)."""
    try:
        return frozenset(release_registry())
    except ValueError:
        return frozenset()


def check_construction(training_set: ArtifactRef, mixture: Sequence[MixtureEntry], store: ArtifactStore) -> None:
    """Refuse a training run whose mixture supervises a zero-shot task.

    Each stream of ``mixture`` is built as the run builds it (``data.streams.build_stream``: on the domain of the
    training set's construction, :func:`training_set_construction`, and on its text restricted to the fields the
    stream declares), and its ``supervision()`` is checked against that construction's screen.
    Refused, all at once:

    - a trained task or label that is a zero-shot task's, or related to one (:func:`find_conflicts`);
    - a trained field the training set did not screen;
    - fixed wording (``texts``) that a zero-shot detector scoped to :data:`WORDING` (or unscoped) matches, or that
      contains a zero-shot question.

    Raises:
        ValueError: naming each stream and what it supervises.
    """
    spec = training_set_construction(store, training_set)
    domain = open_domain(spec.domain, store)
    text = load_training_text(store, training_set)
    screen = wording_screen(spec.screen, domain)
    problems: list[str] = []
    for entry in mixture:
        sup = build_stream(entry.stream, entry.params, domain, text).stream.supervision()
        trained = sorted({*sup.tasks, *sup.labels})
        for c in find_conflicts(domain, trained, spec.screen.zero_shot_tasks):
            problems.append(f"stream {entry.stream!r} trains {c.trained!r}: {c.relation} of zero-shot {c.zero_shot!r}")
        for f in sorted(set(sup.fields) - set(spec.screen.fields)):
            problems.append(f"stream {entry.stream!r} reads field {f!r}, which the training set did not screen")
        for task, wording, shown in screen(sup.texts):
            problems.append(f"stream {entry.stream!r} wording {wording!r} states {task!r} ({shown!r})")
    if problems:
        raise ValueError("the training construction supervises zero-shot tasks:\n- " + "\n- ".join(problems))


class _Catalog(Protocol):
    def tasks(self) -> Mapping[str, TaskSpec]: ...


WordingScreen = Callable[[Sequence[str]], list[tuple[str, str, str]]]
"""Fixed wordings → ``(zero-shot task, wording, what states it)`` for each match."""


def wording_screen(spec: ScreenSpec, catalog: _Catalog) -> WordingScreen:
    """Return the training set's screen of a stream's fixed wording (the one rule ``check_construction`` applies).

    A wording is refused for a zero-shot task when one of the task's detectors (descendants' included) scoped to
    :data:`WORDING`, or unscoped, matches it, or when it contains the task's question. ``catalog`` needs only the
    task catalog (a domain, or a domain's names without data).
    """
    detectors = _detectors(spec, catalog)  # type: ignore[arg-type]  # the detectors read only tasks()
    tasks = catalog.tasks()
    questions = {t: tasks[t].question for t in spec.zero_shot_tasks if tasks[t].question}

    def screen(texts: Sequence[str]) -> list[tuple[str, str, str]]:
        out = []
        for wording in texts:
            for task, dets in detectors.items():
                spans = [s for d in dets if d.applies(WORDING) for s in d.detector.matches("", wording)]
                if spans or (task in questions and questions[task] in wording):
                    out.append((task, wording, wording[spans[0][0] : spans[0][1]] if spans else questions[task]))
        return out

    return screen


def _label_of(domain: Domain, name: str) -> str | None:
    """Return the label column a trained task id or label column stands for (None for a text task)."""
    tasks = domain.tasks()
    labels = domain.labels().columns
    if name in tasks:
        return tasks[name].label if tasks[name].label in labels else None
    if name in labels:
        return name
    raise ValueError(f"{name!r} is neither a task nor a label column of domain {domain.name!r}")


def _binary(column: pd.Series) -> pd.Series | None:
    """Return the column as a nullable boolean, or None if it is not binary."""
    values = column.dropna()
    if values.empty or not values.isin([True, False, 0, 1]).all():
        return None
    return column.astype("boolean")


def find_conflicts(domain: Domain, trained: Sequence[str], zero_shot: Sequence[str]) -> list[Conflict]:
    """List trained tasks or labels whose property is a zero-shot property, its ancestor or its descendant.

    Relations are read from the domain's materialized labels, the domain's own ontology made explicit: on the objects
    where both labels are known, a property that holds wherever another holds is its ancestor (lineage taxa, length
    thresholds, a keyword and its parent keyword). Equal columns are ``same``. Trained text tasks (no label) relate to
    nothing; their texts are screened instead.

    Raises:
        ValueError: if a name is neither a task nor a label column, or a zero-shot task has no label column.
    """
    labels = domain.labels()
    out = []
    for z in zero_shot:
        zcol = _label_of(domain, z)
        if zcol is None:
            raise ValueError(f"zero-shot task {z!r} has no label column")
        zb = _binary(labels[zcol])
        for t in trained:
            tcol = _label_of(domain, t)
            if tcol is None:
                continue
            if tcol == zcol:
                out.append(Conflict(t, z, "same"))
                continue
            tb = _binary(labels[tcol])
            if zb is None or tb is None:
                continue
            known = zb.notna() & tb.notna()
            zv, tv = zb[known].to_numpy(dtype=bool), tb[known].to_numpy(dtype=bool)
            z_in_t = bool(zv.any()) and not bool((zv & ~tv).any())
            t_in_z = bool(tv.any()) and not bool((tv & ~zv).any())
            if z_in_t and t_in_z:
                out.append(Conflict(t, z, "same", int(np.sum(zv))))
            elif z_in_t:
                out.append(Conflict(t, z, "ancestor", int(np.sum(zv))))
            elif t_in_z:
                out.append(Conflict(t, z, "descendant", int(np.sum(tv))))
    return out
