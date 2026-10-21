"""Suite runners, one per suite kind; the specs themselves are data (``data.suites.SuiteSpec``).

Suite kinds (a new kind is a new registered runner; ``yes_no``, ``caption_nll`` and
``composition`` are in ``evaluation.runners``, ``cross_domain`` in ``evaluation.cross_domain``):

==================  ==========================================================================================
``yes_no``          Yes/No next-token readout per task (access on trained tasks; transfer to held-out)
``caption_nll``     teacher-forced NLL of the reference description; the only kind with a mismatched baseline
``composition``     several slots in one prompt: one object through several connectors (several objects: refused)
``cross_domain``    swap quads with protein and molecule roles: own pairing vs swapped pairing (E1)
``generation``      free generation (slot-name reasoning, reports); trace cap, length and completion recorded
``name_guess``      the receiver's elicited name for the object, then scored as text against the embedding
==================  ==========================================================================================

The layer-selection injection scan is a function (``evaluation.layer_selection.injection_scan``), not a suite kind.

**Identity of a score.** A score record's config is the suite spec, the receiver config and the name of the backend
that scored, so the same checkpoint scored on another receiver revision or backend is a different computation.

**Generate, then score (an upstream record).** A generation suite (capability ``generate``) writes a generations table
(``evaluation.generations``), not scores. A scoring suite that needs the generations declares the kind it reads
(``SuiteSpec.upstream_kind``), and its run cites that run's record (:class:`SuiteRunConfig` ``upstream``, like its
checkpoints): the workspace resolves it to its canonical execution, :func:`check_upstream` refuses an upstream run
that is not a generation run of that kind on the same receiver, checkpoints, inputs, step, item set and domains, and
the runner gets it as :class:`Upstream` (``SuiteContext.upstream``). A trace is continued by its token ids
(``receivers.render.TokenIds``), never re-tokenized, after the runner has re-rendered the trace's prompt and matched
its digest (:meth:`Upstream.check_prompt`). The suite stage records the upstream record as an input record and its
generations artifact as an input, so the scores' purpose follows it (``Workspace.purpose``).

**Hardware and purpose.** A standalone suite run (:class:`SuiteRunConfig`) declares its
:class:`~latent_interconnect.provenance.DeviceClass` (A100). The runner refuses another kind or topology of hardware,
not another accelerator model (``ctx.device_class.require``), and the stage records its scores with the
context's purpose: ``result``, or ``anchor`` when an input has no producing record (``Workspace.purpose``). In-job
evaluation builds a context with the training hardware's class and purpose ``monitor``. Only ``result`` records are
eligible as results.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.records import Purpose, Record, RecordStore
from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import ArtifactDigest, RecordKey, StageConfig, added_field
from latent_interconnect.data.items import ItemSet
from latent_interconnect.data.suites import SuiteSpec
from latent_interconnect.domains.base import Domain, DomainSpec
from latent_interconnect.evaluation.generations import GENERATIONS_KIND, prompt_digest
from latent_interconnect.plugins import Registry
from latent_interconnect.provenance import DeviceClass
from latent_interconnect.receivers.base import GenerationBackend, GenerationPlan, ReceiverSpec, ScoringBackend
from latent_interconnect.receivers.dictionary import StateEdit
from latent_interconnect.receivers.kernel import BatchPlan
from latent_interconnect.receivers.render import Renderer
from latent_interconnect.writes.sources import WriteSource

__all__ = [
    "SUITE_KINDS",
    "SUITE_RECORD_KIND",
    "SuiteContext",
    "SuiteRunConfig",
    "SuiteRunner",
    "Upstream",
    "UpstreamError",
    "check_upstream",
    "execution_devices",
    "identity_differences",
    "upstream_of",
]

SUITE_RECORD_KIND = "suite"
"""Record kind of a suite run (the ``suite`` stage's name)."""


class UpstreamError(ValueError):
    """An upstream generation record is missing, is not a generation run, or does not match the run reading it."""


class SuiteRunConfig(StageConfig):
    """One standalone suite execution (the ``suite`` stage's config).

    Attributes:
        suite: the suite (``$ref``).
        receiver: the receiver (``$ref``).
        checkpoints: slot connector argument → checkpoint artifact digest (usually ``$pin``, e.g.
            ``{$pin: connectors/gemma-3-4b-it/esmc-600m/step_002000}``), one per slot connector; none for a suite whose
            every condition is a name (``SuiteSpec.reads_connector``): it writes nothing, so it reads no connector (the
            name references, the name input's reasoning traces and readouts; an upstream run it reads is connector-free
            too, :func:`check_upstream`).
        inputs: slot connector argument → the embedding artifacts its connector reads for the item set's objects
            (composed by the connector's ``InputSpec``; the stage builds ``InputLoader`` → a write source per
            connector). Same keys as ``checkpoints``.
        step: the training step of the checkpoints; None exactly when there are none.
        device_class: the declared hardware (``A100`` for every scoring suite).
        plan: the batch plan the scoring backend is built with, exactly for a scoring suite (capability
            ``score``; None for ``generate``); part of the identity, since it changes numbers.
        generation_plan: the generation plan the generation backend is built with (``receivers.base.GenerationPlan``),
            exactly for a generation suite (capability ``generate``; None for ``score``); part of the identity, since
            it changes numbers. Each run states only its own capability's plan, so the other plan's defaults never
            reach its digest.
        upstream: the record (computation key, usually ``$pin``-ed ``<name>@record``) of the generation run the suite
            reads, exactly when the suite declares ``upstream_kind``; part of the identity.
        edit: a state edit the scoring backend applies in every pass that writes
            (``receivers.dictionary.StateEdit``: operation, layer, order, positions, error, dictionary;
            diagnostic), or None, the default and every paper run's: no edit. Scoring suites only; part of the
            identity when set (added field), so the record of an edited run names its recipe.
    """

    suite: SuiteSpec
    receiver: ReceiverSpec
    checkpoints: dict[str, ArtifactDigest] = Field(default_factory=dict)
    inputs: dict[str, tuple[ArtifactDigest, ...]] = Field(default_factory=dict)
    step: int | None = None
    device_class: DeviceClass
    plan: BatchPlan | None = None
    generation_plan: GenerationPlan | None = None
    upstream: RecordKey | None = added_field(default=None)
    edit: StateEdit | None = added_field(default=None)

    @model_validator(mode="after")
    def _check(self) -> SuiteRunConfig:
        if set(self.inputs) != set(self.checkpoints):
            raise ValueError(f"inputs cover {sorted(self.inputs)}, checkpoints {sorted(self.checkpoints)}")
        names_only = not self.suite.reads_connector()
        connectors = set() if names_only else {s.connector for s in self.suite.slots}
        if set(self.checkpoints) != connectors or (self.step is None) != (not connectors):
            raise ValueError(
                f"suite {self.suite.id!r} cites checkpoints {sorted(self.checkpoints)} at step {self.step}; it takes "
                + ("none and no step (every condition is a name)" if names_only else f"{sorted(connectors)} and a step")
            )
        if (self.upstream is None) != (self.suite.upstream_kind is None):
            raise ValueError(
                f"suite {self.suite.id!r} reads upstream kind {self.suite.upstream_kind}; the run cites {self.upstream}"
            )
        own, other = ("plan", "generation_plan") if self.suite.capability == "score" else ("generation_plan", "plan")
        if getattr(self, own) is None or getattr(self, other) is not None:
            raise ValueError(f"suite {self.suite.id!r} ({self.suite.capability}) takes {own} and no {other}")
        if self.edit is not None and self.suite.capability != "score":
            raise ValueError(f"suite {self.suite.id!r} ({self.suite.capability}): a state edit applies to scoring only")
        return self


def identity_differences(a: SuiteRunConfig, b: SuiteRunConfig) -> list[str]:
    """Return the experimental-identity fields on which two suite runs differ (sorted; empty when they match).

    The identity is what makes two runs read the same connector on the same rows: the receiver, checkpoints, embedding
    inputs, step, item set and domains. Prompting (frame, thinking, params) and plans are not part of it.
    """
    pairs = {
        "receiver": (a.receiver.digest(), b.receiver.digest()),
        "checkpoints": (a.checkpoints, b.checkpoints),
        "inputs": (a.inputs, b.inputs),
        "step": (a.step, b.step),
        "item_set": (a.suite.item_set, b.suite.item_set),
        "domains": (a.suite.domains, b.suite.domains),
    }
    return sorted(name for name, (x, y) in pairs.items() if x != y)


def check_upstream(run: SuiteRunConfig, record: Record) -> SuiteRunConfig:
    """Return the upstream run's config after checking that ``run`` may read ``record`` (``SuiteRunConfig.upstream``).

    The upstream must be a ``suite`` record of a generation run of the kind the suite reads
    (``SuiteSpec.upstream_kind``) with a generations output, whose config is the one its digest names, on the same
    receiver (so the trace's token ids mean the same tokens), checkpoints, embedding
    inputs and step (so a trace or elicited name is the same connector's), and the same item set and domains (so rows
    join on the item).

    Raises:
        UpstreamError: naming every mismatch.
    """
    if record.kind != SUITE_RECORD_KIND:
        raise UpstreamError(f"upstream record {record.computation_key()[:12]} is a {record.kind!r} record, not a suite")
    upstream = SuiteRunConfig.model_validate(record.config)
    if upstream.digest() != record.config_sha:
        raise UpstreamError(f"upstream record {record.computation_key()[:12]}: its config is not the one it names")
    if upstream.suite.capability != "generate" or record.outputs.get(GENERATIONS_KIND) is None:
        raise UpstreamError(f"upstream suite {upstream.suite.id!r} is not a generation run with a generations output")
    if upstream.suite.kind != run.suite.upstream_kind:
        raise UpstreamError(
            f"suite {run.suite.id!r} reads a {run.suite.upstream_kind!r} run; upstream {upstream.suite.id!r} is "
            f"{upstream.suite.kind!r}"
        )
    differ = identity_differences(upstream, run)
    if differ:
        raise UpstreamError(f"suite {run.suite.id!r} reads upstream {upstream.suite.id!r}, which differs in {differ}")
    return upstream


@dataclass(frozen=True)
class Upstream:
    """A scoring suite's upstream generation run, resolved to its canonical record and checked (:func:`check_upstream`).

    Attributes:
        key: the record's computation key (the run's ``SuiteRunConfig.upstream``).
        run: the upstream suite run's config.
        generations: its generations table (``evaluation.generations``).
    """

    key: str
    run: SuiteRunConfig
    generations: pd.DataFrame

    def row(self, item: int, role: str, condition: str, task: str | None = None) -> pd.Series:
        """Return the one generation of ``item`` under ``condition`` of ``role`` (and ``task``, if given).

        Raises:
            UpstreamError: if no row or several match: every item a reader scores has exactly one generation.
        """
        g = self.generations
        mask = (g["item"] == item) & (g[f"condition.{role}"] == condition)
        if task is not None:
            mask &= g["task"] == task
        found = g[mask]
        if len(found) != 1:
            raise UpstreamError(
                f"upstream {self.run.suite.id!r} has {len(found)} generations for item {item}, {role}={condition}"
                + ("" if task is None else f", task {task!r}")
            )
        return found.iloc[0]

    def check_prompt(self, row: pd.Series, input_ids: tuple[int, ...]) -> None:
        """Refuse a reader whose re-rendered prompt is not the one the trace continued (by :func:`prompt_digest`).

        Raises:
            UpstreamError: if the digests differ (another frame, question, thinking flag or slot binding).
        """
        if prompt_digest(input_ids) != row["prompt"]:
            raise UpstreamError(
                f"item {row['item']}, task {row['task']!r}: the re-rendered prompt is not the prompt of upstream "
                f"{self.run.suite.id!r}"
            )


@dataclass(frozen=True)
class SuiteContext:
    """Everything a suite run needs besides its spec and write sources.

    The trainer builds one around its training backend for in-job evaluation; standalone evaluation builds one around
    the receiver's scoring or generation backend.

    Attributes:
        artifacts: the artifact store (item sets in, scores out).
        records: the record store.
        receiver: the receiver config (part of every score record's identity).
        backend_name: the registered backend that scores (part of every score record's identity).
        backend: the backend instance.
        renderer: the receiver's renderer (``receivers.open_renderer``).
        domain: resolves a domain spec to a domain instance.
        device_class: the declared hardware; runners refuse any other (``device_class.require``).
        purpose: the purpose of the records made from this context's scores.
        items: resolves an item-set artifact digest to its :class:`~latent_interconnect.data.items.ItemSet` (as
            ``domain`` resolves domains, so runners never read artifact files of another module's format).
        plan: the batch plan the scoring backend was constructed with (``SuiteRunConfig.plan``, or the training
            backend's for in-job evaluation; None for a generation backend): runners compute each call's partition
            with it (``kernel.plan_batches``) and record its digest. It changes numbers
            (``examples/guides/batching.md``), so it is part of the suite run's identity.
        code: the code tree's (commit, dirty) the scores are computed with (``provenance.code_state``); a resumed
            injection scan reuses only parts of the same code and execution (``layer_selection.injection_scan``).
        generation_plan: the generation plan the generation backend was constructed with
            (``SuiteRunConfig.generation_plan``; None for a scoring backend); a generation runner checks it against
            ``backend.generation_plan``.
        upstream: the upstream generation run the suite reads (``SuiteRunConfig.upstream``), or None.
        unavailable: object → the reason the run's embedding inputs recorded for skipping it (their ``skipped``
            metadata, ``encoders.store``; set by the ``suite`` stage); empty in-job.
    """

    artifacts: ArtifactStore
    records: RecordStore
    receiver: ReceiverSpec
    backend_name: str
    backend: ScoringBackend | GenerationBackend
    renderer: Renderer
    domain: Callable[[DomainSpec], Domain]
    device_class: DeviceClass
    purpose: Purpose
    items: Callable[[str], ItemSet]
    plan: BatchPlan | None
    code: tuple[str, bool]
    generation_plan: GenerationPlan | None = None
    upstream: Upstream | None = None
    unavailable: Mapping[str, str] = field(default_factory=dict)


def upstream_of(spec: SuiteSpec, ctx: SuiteContext) -> Upstream | None:
    """Return the context's upstream run, checked against the kind the suite reads (``SuiteSpec.upstream_kind``).

    Raises:
        UpstreamError: if the suite reads an upstream kind the context does not carry, or the context carries an
            upstream run the suite does not read.
    """
    found = None if ctx.upstream is None else ctx.upstream.run.suite.kind
    if found != spec.upstream_kind:
        raise UpstreamError(f"suite {spec.id!r} reads upstream kind {spec.upstream_kind}, the context carries {found}")
    return ctx.upstream


class SuiteRunner(Protocol):
    """Runs one suite kind."""

    def __init__(self, spec: SuiteSpec) -> None: ...

    def run(self, ctx: SuiteContext, sources: Mapping[str, WriteSource], step: int | None) -> ArtifactRef:
        """Score every item × task × condition combination and commit a ``scores`` artifact.

        The caller records it: the ``suite`` stage (purpose ``result``, or ``anchor`` on inputs of unknown lineage) or
        the trainer (``monitor``).

        Args:
            ctx: stores, receiver, backend, renderer and domains.
            sources: slot connector argument → write source (none for a suite that writes nothing).
            step: training step of the checkpoints scored (None: no checkpoints, ``SuiteRunConfig.step``).

        Returns:
            The ``scores`` artifact (``evaluation.scores`` schema).
        """
        ...


SUITE_KINDS: Registry[type[SuiteRunner]] = Registry("suite kind")
"""Suite runner classes by kind name."""


def execution_devices(execution: Mapping[str, str]) -> tuple[str, ...]:
    """Return the accelerator names of a backend's ``execution()`` (one per device; empty on CPU).

    The form ``DeviceClass.require`` and provenance take: ``device_name`` repeated ``device_count`` times, or nothing
    when the device is the CPU.

    Raises:
        KeyError: if ``execution`` lacks ``device_name`` or ``device_count`` (``receivers.base`` requires both).
    """
    name, count = execution["device_name"], int(execution["device_count"])
    return () if count == 0 or name.lower() == "cpu" else (name,) * count
