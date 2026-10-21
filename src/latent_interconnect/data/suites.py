"""Suite specs: what an evaluation scores and how, as data (``configs/suites/*.yaml``).

The spec lives here, below ``training`` and ``evaluation``, because both use it: the trainer runs an in-job suite and
``evaluation`` runs every suite. Runners per suite kind are in ``evaluation.suites``.

A suite fixes *what* is scored (an item set, tasks, conditions per role), *how* (backend capability, generation
settings) and the domain data version. Precision and batching are not part of a suite: runs use the fastest
convenient execution, and each score record carries its precision, batch size, device and backend. A suite says
nothing about *which* connector: it is run against connector checkpoints, and each (suite, checkpoints) execution is
one ``score`` record.

**Upstream record.** A scoring suite may read one generation run's record: the post-reasoning readout continues each
generated trace (``upstream_kind: generation``), and ``Condition.ELICITED_NAME`` shows the name a ``name_guess`` run
elicited (``upstream_kind: name_guess``). Generate and score are two records chained through the record store, never
two backends in one run. The spec declares only the kind it reads, since one spec serves every connector and step; the
record itself is cited by the run, like its checkpoints (``evaluation.suites.SuiteRunConfig.upstream``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, model_validator

from latent_interconnect.config import ArtifactDigest, Config, added_field
from latent_interconnect.data.answers import AnswerConvention
from latent_interconnect.data.items import ItemSet
from latent_interconnect.domains.base import DomainSpec, Frame, TaskSpec
from latent_interconnect.receivers.base import GenerationParams

__all__ = [
    "FIELD_CONDITIONS",
    "LIKELIHOOD_KINDS",
    "LIKELIHOOD_READOUT",
    "SLOT_READOUT_KINDS",
    "UPSTREAM_KINDS",
    "Condition",
    "SlotSpec",
    "SuiteSpec",
]

LIKELIHOOD_KINDS = frozenset({"caption_nll"})
"""Suite kinds whose metric is a likelihood or perplexity, where a shuffled baseline is meaningful."""
LIKELIHOOD_READOUT = "target_nll"
"""The ``params.readout`` value that makes a suite of a kind with a readout choice (``composition``) a likelihood
suite (:meth:`SuiteSpec.likelihood`)."""
SLOT_READOUT_KINDS = frozenset({"yes_no", "caption_nll"})
"""The single-slot readout kinds (``evaluation.runners``): the only kinds that fill a wild-type slot
(``SlotSpec.fill``) or cycle frames over the items (``SuiteSpec.frame_cycle``)."""
UPSTREAM_KINDS = frozenset({"generation", "name_guess"})
"""The generation suite kinds whose records a scoring suite may read (``SuiteSpec.upstream_kind``)."""


class Condition(StrEnum):
    """What fills a slot for an item."""

    OWN = "own"
    """The item's own object for that role, through the connector."""
    MISMATCHED = "mismatched"
    """Another item's object (a seeded random derangement): the shuffled baseline, allowed only in likelihood and
    perplexity suites (:data:`LIKELIHOOD_KINDS`), the one exception to scoring without a shuffled baseline."""
    NAME = "name"
    """The object's name as text in place of the slot; no write (the name reference)."""
    VALUE = "value"
    """The object's encoder input as text in place of the slot (``Domain.objects`` ``value``: a molecule's SMILES);
    no write. The structure-text cells of the cross-domain factorial (``cross_domain``); other kinds refuse it."""
    ELICITED_NAME = "elicited_name"
    """The name the receiver itself gave for the object (from a ``name_guess`` record), as text; no write."""
    PLACEHOLDER = "placeholder"
    """The slot's placeholder tokens with nothing written: the cross-domain factorial's omitted cells
    (``cross_domain``); other kinds refuse it."""
    SIBLING = "sibling"
    """A variant item's sibling (another substitution of the same parent, at another position; the item set's field
    ``sibling.<role>``), through the connector: the within-protein mutation control."""
    WT = "wt"
    """A variant item's wild-type parent (the item set's field ``wt.<role>``), through the connector."""
    DERANGE = "derange"
    """A variant item's label-independent derangement partner (another parent's variant; the item set's field
    ``derange.<role>``), through the connector: a fixed partner map with no same-parent pairing. It is scored on the
    clinical and functional binary readouts as well as the likelihood ones (no shuffled-object control when the
    unshuffled AUROC is itself meaningful), the different-parent comparison being the mutation test itself."""
    NAME_SUBSTITUTION = "name_substitution"
    """A variant item's parent name with its substitution, as text (the item set's field ``name_substitution.<role>``,
    e.g. "Methylenetetrahydrofolate reductase with the amino-acid substitution E153Q"); no write."""


FIELD_CONDITIONS = frozenset({Condition.SIBLING, Condition.WT, Condition.DERANGE, Condition.NAME_SUBSTITUTION})
"""The variant conditions (``build.variant_items``): each reads the item set's field ``<condition>.<role>``, the
object (or, for ``name_substitution``, the text) that fills the slot instead of the item's own object."""


def _stream_frame(value: Any) -> Any:
    """Turn ``{stream: <config>, template: <id>}`` into that template's frame; else ``value``.

    ``<config>`` is any config with ``templates`` (each an ``id`` and its ``user`` text): a training stream (a QA or
    the variant stream, whose templates the suite then evaluates seen) or a frames file of held-out wordings
    (``configs/frames/``). The frame checks the slots.
    """
    if not (isinstance(value, dict) and set(value) == {"stream", "template"}):
        return value
    found = [t for t in value["stream"].get("templates", ()) if t.get("id") == value["template"]]
    if len(found) != 1:
        raise ValueError(f"the stream has {len(found)} templates {value['template']!r}, not 1")
    return {"id": found[0]["id"], "user": found[0]["user"]}


class SlotSpec(Config):
    """One slot of the suite's prompts.

    Attributes:
        name: the marker name used in the questions (``<SLOT:name>``).
        role: the item-set role whose object fills the slot.
        connector: which connector argument of the run fills it (e.g. ``"a"``, ``"b"`` for composition).
        fill: ``"wt"``: the slot always holds the item's wild-type parent (the item set's field ``wt.<role>``, and
            under a text condition the parent's name, ``name.<role>``), whatever the condition does to the other
            slots: the wild-type slot of the variant relational prompt (``vrel_*``), beside one slot the conditions
            fill, in a suite of :data:`SLOT_READOUT_KINDS`. None: the condition's object.
    """

    name: str
    role: str
    connector: str
    fill: Literal["wt"] | None = added_field(default=None)


class SuiteSpec(Config):
    """An evaluation suite.

    Attributes:
        id: e.g. ``"protein/transfer@union2036"``.
        kind: registered suite kind (``evaluation.suites.SUITE_KINDS``).
        domains: the domains of the item set's roles, at fixed data versions.
        item_set: the item-set artifact digest.
        tasks: task ids from the domain catalog, or from ``task_specs``.
        task_specs: tasks the suite declares itself, for labels or fields the domain catalog asks no question of (the
            variant labels and effect text, ``build.variant_items``); each is in scope for every item (the item
            producer selected them). An id may not repeat a catalog task.
        frame: the frame, if the kind uses one: a frame id from the domain data, or the suite's own :class:`Frame`
            (its text is then part of the suite's identity), given inline or as ``{stream: {$ref: <config>},
            template: <id>}``, that template's wording: a training stream's (seen) or a frames file's (held out);
            :meth:`frame_in` returns it.
        frame_cycle: several frames, given as ``frame`` is, assigned to the item set's units in turn instead of one
            frame for every item (:meth:`item_frames`): the declared seen wordings of the variant questions, each unit
            asked in exactly one, the same under every condition and at every step. Empty: ``frame``.
        frame_unit: the item-set field whose values are the units ``frame_cycle`` assigns (``pair``: both members of
            a fixed pair in one wording); None: each item is a unit.
        slots: the prompt's slots.
        conditions: the condition combinations scored, one mapping role → condition each.
        thinking: the receiver's thinking mode for every prompt of the suite (``Conversation.thinking``, the one home
            of the flag; mapped through ``ReceiverSpec.template_kwargs``).
        capability: backend capability used (``score`` or ``generate``).
        generation: generation settings (generation suites only).
        upstream_kind: the suite kind of the generation run a scoring suite reads (:data:`UPSTREAM_KINDS`), or None.
        params: kind-specific parameters, validated by the kind's runner (e.g. ``evaluation.runners.YesNoParams``).
        answer: the receiver's Yes/No answer convention (``data.answers``) for a kind that asks a Yes/No question
            (``yes_no``, ``composition`` with ``readout: yes_no``, ``cross_domain``, ``generation`` over Yes/No
            tasks); its frame then carries only the task and object wording, and its params no answer words or
            prefill. None: the inline policy (frame instruction, params words), until Kimi's convention.
    """

    id: str
    kind: str
    domains: tuple[DomainSpec, ...] = Field(min_length=1)
    item_set: ArtifactDigest
    tasks: tuple[str, ...] = ()
    task_specs: tuple[TaskSpec, ...] = added_field(default=())
    frame: Annotated[str | Frame | None, BeforeValidator(_stream_frame)] = None
    frame_cycle: tuple[Annotated[Frame, BeforeValidator(_stream_frame)], ...] = added_field(default=())
    frame_unit: str | None = added_field(default=None)
    slots: tuple[SlotSpec, ...] = Field(min_length=1)
    conditions: tuple[dict[str, Condition], ...] = Field(min_length=1)
    thinking: bool = False
    capability: Literal["score", "generate"]
    generation: GenerationParams | None = None
    upstream_kind: str | None = added_field(default=None)
    params: dict[str, Any] = Field(default_factory=dict)
    answer: AnswerConvention | None = added_field(default=None)

    def reads_connector(self) -> bool:
        """Return whether the suite reads a connector: False exactly when every condition is a name.

        A name is ``name`` or ``name_substitution`` (text; a wild-type slot shows its parent's name under either).
        The one definition of a suite that writes nothing and needs no slot-token count: its run cites no checkpoint,
        inputs or step (``evaluation.suites.SuiteRunConfig``) and the runners read no write source
        (``evaluation.runners``). ``value`` and ``elicited_name`` also write nothing, but a ``value`` suite is
        ``cross_domain``'s, which reads every slot's token count, and an ``elicited_name`` suite reads its connector's
        upstream run (a suite whose conditions are all ``elicited_name`` reads one: it cites its connector, as its
        upstream does; none such is declared); ``cross_domain`` refuses a suite that reads none.
        """
        names = (Condition.NAME, Condition.NAME_SUBSTITUTION)
        return any(c not in names for combo in self.conditions for c in combo.values())

    def likelihood(self) -> bool:
        """Return whether the suite's metric is a likelihood (a likelihood kind, or the likelihood readout).

        True for a :data:`LIKELIHOOD_KINDS` kind, or ``params.readout`` equal to :data:`LIKELIHOOD_READOUT`.
        """
        return self.kind in LIKELIHOOD_KINDS or self.params.get("readout") == LIKELIHOOD_READOUT

    def frame_in(self, frames: Mapping[str, Frame]) -> Frame:
        """Return the suite's frame: its own, or the one it names in ``frames`` (``Domain.frames``).

        Raises:
            ValueError: if the suite has no frame.
            KeyError: if the named frame is not in ``frames``.
        """
        if self.frame is None:
            raise ValueError(f"suite {self.id!r}: no frame")
        return self.frame if isinstance(self.frame, Frame) else frames[self.frame]

    def item_frames(self, items: ItemSet, frames: Mapping[str, Frame]) -> tuple[Frame, ...]:
        """Return each item's frame: :meth:`frame_in` for every item, or ``frame_cycle``'s by unit.

        The units (the values of ``frame_unit``, or the items) take the frames in turn, in their order of first
        appearance in the item set, so the counts per frame differ by at most one.

        Raises:
            KeyError: if the item set has no ``frame_unit`` field.
        """
        if not self.frame_cycle:
            return (self.frame_in(frames),) * len(items)
        keys: Sequence[object] = items.fields[self.frame_unit] if self.frame_unit else range(len(items))
        order = {k: j for j, k in enumerate(dict.fromkeys(keys))}
        return tuple(self.frame_cycle[order[k] % len(self.frame_cycle)] for k in keys)

    @model_validator(mode="after")
    def _check(self) -> SuiteSpec:
        uses_mismatch = any(Condition.MISMATCHED in c.values() for c in self.conditions)
        if uses_mismatch and not self.likelihood():
            raise ValueError(f"suite {self.id!r}: the mismatched baseline is for likelihood suites only")
        if (self.capability == "generate") != (self.generation is not None):
            raise ValueError(f"suite {self.id!r}: generation settings come with the generate capability, and only")
        if self.upstream_kind is not None and (self.capability != "score" or self.upstream_kind not in UPSTREAM_KINDS):
            raise ValueError(
                f"suite {self.id!r}: a scoring suite reads one of {sorted(UPSTREAM_KINDS)}, not {self.upstream_kind!r}"
            )
        declared = [t.id for t in self.task_specs]
        if len(set(declared)) != len(declared) or not set(declared) <= set(self.tasks):
            raise ValueError(f"suite {self.id!r}: task_specs {declared} must be distinct tasks the suite scores")
        filled = [s.name for s in self.slots if s.fill is not None]
        if filled and (len(filled) == len(self.slots) or self.kind not in SLOT_READOUT_KINDS):
            raise ValueError(
                f"suite {self.id!r}: wild-type slots {filled} need a slot the conditions fill, in one of "
                f"{sorted(SLOT_READOUT_KINDS)}"
            )
        if self.frame_cycle and (
            self.frame is not None
            or len({f.id for f in self.frame_cycle}) != len(self.frame_cycle)
            or len(self.frame_cycle) < 2  # noqa: PLR2004 - one frame is ``frame``
            or self.kind not in SLOT_READOUT_KINDS
        ):
            raise ValueError(
                f"suite {self.id!r}: frame_cycle takes two or more distinct frames in place of frame, in one of "
                f"{sorted(SLOT_READOUT_KINDS)}"
            )
        if self.frame_unit is not None and not self.frame_cycle:
            raise ValueError(f"suite {self.id!r}: frame_unit comes with frame_cycle")
        elicits = any(Condition.ELICITED_NAME in c.values() for c in self.conditions)
        if elicits != (self.upstream_kind == "name_guess"):
            raise ValueError(f"suite {self.id!r}: elicited_name and upstream_kind name_guess come together")
        return self
