"""What the receiver scans write: each object's state, from a capture or a ridge, at chosen entries.

The two receiver scans (``layer_selection.injection_scan``, one layer at a time; ``layer_sets.compare_layer_sets``,
whole write sets) share:

- :class:`WrittenScan`: the fields that say what is written and how it is scored;
- :func:`scan_states`: layer → the objects' states for the scan's source;
- :class:`StateWrites`: a write source that writes those states at fixed (token, layer) entries with fixed α;
- :func:`scan_suite`: the ``yes_no`` suite a scan scores through (prompts vetted, hardware checked);
- :func:`require_train`: the train-only check for study evidence (the study's scans, a set comparison);
- :func:`require_label_classes`: the check that every task's AUROC is defined on the scored objects, before scoring;
- :class:`ScanOutputs`: the summary numbers and the per-object scores a scan commits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal, NamedTuple, Self

import numpy as np
import pandas as pd
import torch
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.build.ridge import load_ridge, source_layer
from latent_interconnect.config import ArtifactDigest, Config, added_field
from latent_interconnect.data.answers import AnswerConvention
from latent_interconnect.data.items import ItemSet
from latent_interconnect.data.suites import Condition, SlotSpec, SuiteSpec
from latent_interconnect.domains.base import Domain, DomainSpec, Frame, require_splits
from latent_interconnect.encoders.loader import InputLoader
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import EmbeddingReader
from latent_interconnect.evaluation.runners import YesNoParams
from latent_interconnect.evaluation.suites import SuiteContext
from latent_interconnect.receivers.base import ReceiverSpec
from latent_interconnect.stats.bootstrap import ResamplingUnits
from latent_interconnect.writes.connector import SlotWrites
from latent_interconnect.writes.inputs import InputKind, InputSpec
from latent_interconnect.writes.layers import Entry

__all__ = [
    "SCAN_CONNECTOR",
    "ScanOutputs",
    "StateWrites",
    "WrittenScan",
    "require_label_classes",
    "require_train",
    "scan_states",
    "scan_suite",
]

SCAN_CONNECTOR = "scan"
"""Connector name of the written states in a scan's suite."""


class WrittenScan(Config):
    """What a receiver scan writes and how it is scored (the fields both receiver scans share).

    Attributes:
        receiver: the receiver (embedded).
        domain: the domain at a fixed data version (questions, frame).
        source: ``oracle`` writes the receiver's own captured state (the own-state reference); ``ridge`` a fitted
            ridge's prediction from the object's embedding (a study's: the held-out scan ridge, which never saw the
            scanned objects).
        states: the capture (``oracle``) or the ridge fit (``ridge``, a ``ridge_init`` artifact).
        item_set: the objects scored (one role): a study's are ``train`` objects (``run_study``); a standalone
            scan's (the ``scan`` stage) may be any set.
        tasks: Yes/No task ids asked.
        units: the resampling units; ``units.interval`` is refused (a scan's intervals are the bootstrap's).
        frame: the frame that places the tasks' questions (as for a ``yes_no`` suite): a domain frame id, or the
            scan's own :class:`~latent_interconnect.domains.base.Frame`.
        slot: the frame's slot marker name (``<SLOT:name>``).
        answers: the inline Yes/No answer words (as a ``yes_no`` suite's params); None under ``answer``.
        answer: the receiver's answer convention (``data.answers``, as ``SuiteSpec.answer``); None: ``answers``.
        input: the encoder input the ridge reads (``ridge`` only).
        encoder: the encoder recipe ``input`` names (embedded; ``ridge`` only): every load is checked against it
            (``encoders.loader.InputLoader``, ``writes.inputs.check_read_layers``).
        embeddings: the encoder's embedding artifacts covering the item set (``ridge`` only).
        pool: the capture state an ``oracle`` scan writes: None, the capture's stored pooled state (its recipe's
            ``target``); ``mean``, the mean over the read tokens (a capture without means fails; nothing falls back).
            Refused for ``ridge``, whose fit chose the state (``RidgeSpec.pool``).
        source_floor: the source-layer floor p* of an ``oracle`` scan: at layer ``L`` it writes the ``pool``
            state of layer ``build.ridge.source_layer(L, p*)``; None, each layer's own. Refused for ``ridge``
            (``RidgeSpec.source_floor``).
        fixed_source: the fixed source layer S of an ``oracle`` scan (the fixed-source curves): at every layer
            ``L`` > 0 it writes the ``pool`` state of layer S, and at layer 0 layer 0's own (the source-floor rule,
            ``build.ridge.source_layer``); None, as ``source_floor`` says. S is a capture layer, 1 .. the receiver's
            ``depth`` (its hidden states; whether the capture holds S is checked when the state is read). Refused with
            ``source_floor`` and for ``ridge``. A fixed-source curve is never a study's own-state reference (the
            ridge has no fixed source; ``layer_selection_analysis``).
    """

    receiver: ReceiverSpec
    domain: DomainSpec
    source: Literal["oracle", "ridge"]
    states: ArtifactDigest
    item_set: ArtifactDigest
    tasks: tuple[str, ...] = Field(min_length=1)
    units: ResamplingUnits = ResamplingUnits()
    frame: str | Frame
    slot: str
    answers: YesNoParams | None = added_field(default=None)
    answer: AnswerConvention | None = added_field(default=None)
    input: InputSpec | None = None
    encoder: EncoderRecipe | None = None
    embeddings: tuple[ArtifactDigest, ...] = ()
    pool: Literal["mean"] | None = added_field(default=None)
    source_floor: int | None = added_field(default=None, ge=1)
    fixed_source: int | None = added_field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_answer(self) -> Self:
        if (self.answers is None) == (self.answer is None):
            raise ValueError("a scan gives exactly one of answers (inline words) and answer (a convention)")
        return self

    @model_validator(mode="after")
    def _check_units(self) -> Self:
        if self.units.interval:
            raise ValueError(f"a scan's intervals are the bootstrap's, not units.interval {self.units.interval!r}")
        return self

    @model_validator(mode="after")
    def _check_source(self) -> Self:
        ridge = self.source == "ridge"
        given = (self.input is not None, self.encoder is not None, bool(self.embeddings))
        if any(g != ridge for g in given):
            raise ValueError("input, encoder and embeddings are given exactly when the source is 'ridge'")
        if ridge and (self.pool is not None or self.source_floor is not None or self.fixed_source is not None):
            raise ValueError(
                "pool, source_floor and fixed_source select a capture's state: a ridge scan writes what its ridge fit"
            )
        if self.source_floor is not None and self.fixed_source is not None:
            raise ValueError(
                f"source_floor {self.source_floor} and fixed_source {self.fixed_source}: a scan has one source rule"
            )
        if self.fixed_source is not None and self.fixed_source > self.receiver.depth:
            raise ValueError(
                f"fixed_source {self.fixed_source} past {self.receiver.name}'s hidden states 0..{self.receiver.depth}"
            )
        return self


class ScanOutputs(NamedTuple):
    """What a scan commits: its summary and the per-object scores behind it (a stage records both).

    Attributes:
        numbers: the summary with intervals (a ``numbers`` artifact).
        scores: the per-object scores, one ``scores`` artifact with the scan's columns.
    """

    numbers: ArtifactRef
    scores: ArtifactRef


def require_train(items: ItemSet, domain: Domain, use: str) -> None:
    """Refuse an item set holding an object outside ``train`` (study evidence never sees ``val`` or ``test``).

    Raises:
        ValueError: naming ``use`` and the offending objects (``domains.require_splits``).
    """
    require_splits(domain, [o for role in sorted(items.objects) for o in items.objects[role]], {"train"}, use)


def scan_suite(spec: WrittenScan, role: str, suite_id: str) -> SuiteSpec:
    """Return the ``yes_no`` suite a scan scores through: the item set's ``own`` condition, the scan's connector."""
    return SuiteSpec(
        id=suite_id,
        kind="yes_no",
        domains=(spec.domain,),
        item_set=spec.item_set,
        tasks=spec.tasks,
        frame=spec.frame,
        slots=(SlotSpec(name=spec.slot, role=role, connector=SCAN_CONNECTOR),),
        conditions=({role: Condition.OWN},),
        capability="score",
        params=spec.answers.model_dump() if spec.answers is not None else {},
        answer=spec.answer,
    )


class StateWrites:
    """A write source of fixed per-object states at fixed entries, each with a fixed α.

    Args:
        objects: the objects, in the row order of ``states``.
        states: receiver layer → ``(n, d_model)`` states of the objects.
        alphas: entry → its α; an entry writes the state of its layer.
    """

    def __init__(self, objects: Sequence[str], states: Mapping[int, np.ndarray], alphas: Mapping[Entry, float]) -> None:
        self.row = {o: i for i, o in enumerate(objects)}
        self.states = {L: torch.as_tensor(s, dtype=torch.float32) for L, s in states.items()}
        missing = sorted({e.layer for e in alphas} - set(self.states))
        if missing:
            raise ValueError(f"entries write layers {missing} that have no states")
        self.alphas = {e: torch.tensor(a, dtype=torch.float32) for e, a in alphas.items()}

    @property
    def bank(self) -> ArtifactRef | None:
        """No bank."""
        return None

    @property
    def entries(self) -> tuple[Entry, ...]:
        """The written entries."""
        return tuple(sorted(self.alphas))

    def writes_for(self, object_ids: Sequence[str]) -> SlotWrites:
        """Return the objects' states at every entry."""
        rows = [self.row[o] for o in object_ids]
        deltas = {e: self.states[e.layer][rows] for e in self.alphas}
        return SlotWrites(object_ids=tuple(object_ids), deltas=deltas, alphas=dict(self.alphas))


def scan_states(
    spec: WrittenScan, ctx: SuiteContext, objects: Sequence[str], *, bias: bool = True
) -> Callable[[int], np.ndarray]:
    """Return layer → ``(n, d_model)`` states of the objects, from the capture or the ridge.

    An oracle scan writes the capture's ``pool`` state of the layer's source layer (``source_floor``, or
    ``fixed_source``; ``build.ridge.source_layer``).

    A ridge state is the fit's prediction ``W z + b`` (``b``: the fit-set mean of its target state); with
    ``bias`` off it is ``W z`` alone, the object-specific part (a ridge source only).

    Raises:
        FileNotFoundError: (at a layer) if an oracle scan's ``pool`` is ``mean`` and the capture holds no means.
        ValueError: if the ridge was fit on another input (``RidgeSolution.require_fit``), the ridge input loads
            no pooled vectors, or ``bias`` is off for an oracle scan.
    """
    if spec.source == "oracle" and not bias:
        raise ValueError("an oracle scan writes captured states: there is no ridge intercept to leave out")
    if spec.source == "oracle":
        reader = EmbeddingReader(ctx.artifacts, ctx.artifacts.ref(spec.states))
        state = reader.pooled if spec.pool is None else reader.means
        return lambda layer: state(objects, source_layer(layer, spec.source_floor, spec.fixed_source))
    if spec.input is None or spec.encoder is None:
        raise ValueError("a ridge scan needs its input and encoder")  # the spec validator guarantees it; narrows types
    ridge = load_ridge(ctx.artifacts, ctx.artifacts.ref(spec.states))
    readers = [EmbeddingReader(ctx.artifacts, ctx.artifacts.ref(d)) for d in spec.embeddings]
    loader = InputLoader(spec.input, readers, {InputKind.POOLED}, encoder=spec.encoder)
    digest = None if loader.append is None else loader.append.encoder.digest()
    ridge.require_fit(
        "scan",
        input=spec.input,
        recipe_digest=spec.encoder.digest(),
        receiver={"model_id": spec.receiver.model_id, "revision": spec.receiver.revision},
        append_digest=digest,
    )
    pooled = loader.batch(objects).pooled
    if pooled is None:
        raise ValueError("the ridge input loaded no pooled vectors")
    z = (pooled.double().numpy() - ridge.mu) / ridge.sd

    def predict(layer: int) -> np.ndarray:
        fit = ridge.layers[layer]
        wz = z @ fit.W.T.astype(np.float64)
        out: np.ndarray = (wz + fit.b if bias else wz).astype(np.float32)
        return out

    return predict


def require_label_classes(domain: Domain, objects: Sequence[str], tasks: Sequence[str], use: str) -> None:
    """Refuse tasks whose labeled in-scope objects hold one label class (no AUROC), before any scoring.

    The objects a ``yes_no`` suite scores for a task are those in its scope (``Domain.scope_mask``); a label ``> 0``
    is a positive, as the AUROC reads it (``evaluation.contrasts``).

    Raises:
        ValueError: naming ``use`` and each task with fewer than two label classes among its scored objects.
    """
    catalog, labels = domain.tasks(), domain.labels()
    index = pd.Index(list(dict.fromkeys(objects)))
    short = {}
    for task_id in tasks:
        task = catalog[task_id]
        in_scope = domain.scope_mask(task, index).reindex(index).fillna(False).to_numpy(bool)
        known = labels[task.label].reindex(index)[in_scope].dropna()
        classes = sorted({bool(v > 0) for v in known.to_numpy(np.float64)})
        if classes != [False, True]:
            short[task_id] = classes
    if short:
        raise ValueError(
            f"{use}: tasks with one label class among their in-scope objects (no AUROC): "
            + ", ".join(f"{t!r} (classes {c})" for t, c in short.items())
        )
