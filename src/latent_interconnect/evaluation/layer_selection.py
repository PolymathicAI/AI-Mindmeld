"""Layer-selection tools: measurements that help a person choose write and read layers. They choose nothing.

The process: the ``read_scan`` stage runs the encoder's read-layer scan (CPU); a person picks
the read layers; the ``ridge`` stage fits the held-out scan ridge from them (``build.ridge`` with ``exclude``: the
selection objects minus the scan items); the ``study`` stage runs the setup's injection scan of that ridge's
predictions (:class:`LayerSelectionStudy`); and the ``scan`` stage runs one standalone scan (an injection scan or a
set comparison, :class:`ScanRunConfig`). The own-state reference injection scan depends on no encoder (only the
receiver, its captured states, the items, the tasks, the prompt, the write settings and the execution config), so it
is a ``scan`` record run once per domain: every setup that shares the spec cites the one record, and the
``layer_selection`` analysis checks it matches the setup's study (``layer_selection_analysis``). Each
scan commits its per-layer (or per-set) results with confidence intervals and the per-object table behind them, which
its stage records, and its setup's ``layer_selection`` analysis reads them (``evaluation.layer_selection_analysis``).
How to read them and choose layers is described in prose in ``examples/guides/choosing_layers.md``; the chosen layers
are then written explicitly into the connector config.

- :func:`injection_scan`: for one receiver, write a state into the slot at one layer at a time and score Yes/No
  answers. It shows where a written state is usable: the onset, the peak and the fade with depth. The written state is
  the receiver's own name state (``source = "oracle"``, the own-state reference; no encoder needed; a mapped state can
  beat it) or a fitted ridge's prediction (``source = "ridge"``; a study's is the held-out scan ridge, which never saw
  the scanned objects).
- :func:`~latent_interconnect.evaluation.read_scan.read_layer_scan` (``evaluation.read_scan``): for one encoder, fit
  cross-validated ridge probes from each encoder layer's pooled states to the domain's Yes/No task labels and, for
  protein encoders, to variant labels. It shows which encoder layers carry the domain's properties.
- :func:`~latent_interconnect.evaluation.layer_sets.compare_layer_sets` (``evaluation.layer_sets``): score whole
  candidate write-layer sets (at the connector's step-0 α) on the same items, with paired
  contrasts between sets.

**Training objects only.** A study never sees ``test`` (the final evaluation's) or
``val`` (the in-job monitoring set). It runs inside the ``train`` split: the read scan's probes are cross-validated
over cluster folds; the held-out scan ridge is fit on the selection objects minus the scan items and predicts only
them; the own-state reference fits nothing and scores the study's item set, which the analysis requires, so it is
train-only through the study's check. The studies run before any training, so a training object is special only to a
map that saw it. :func:`run_study` checks every object list with ``domains.require_splits`` (``{"train"}``) before
running anything. The rule is the studies': a standalone scan (the ``scan`` stage, :class:`ScanRunConfig`) scores the
set it declares.

For one (receiver, encoder) setup the diagnostic is the study's record, the shared own-state reference record and the
read scan's record; a set comparison adds another once candidate sets exist. A connector config cites them in
``layer_choice.records``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.canonical import canonical_sha256
from latent_interconnect.config import StageConfig
from latent_interconnect.data.items import ItemSet
from latent_interconnect.domains.base import Domain, DomainSpec
from latent_interconnect.evaluation.contrasts import task_estimate, unit_columns
from latent_interconnect.evaluation.layer_sets import LayerSetComparisonSpec
from latent_interconnect.evaluation.runners import YesNo
from latent_interconnect.evaluation.scan_states import (
    SCAN_CONNECTOR,
    ScanOutputs,
    StateWrites,
    WrittenScan,
    require_label_classes,
    require_train,
    scan_states,
    scan_suite,
)
from latent_interconnect.evaluation.scores import read_scores, write_numbers, write_scores
from latent_interconnect.evaluation.suites import SuiteContext
from latent_interconnect.provenance import DeviceClass
from latent_interconnect.receivers.kernel import BatchPlan
from latent_interconnect.writes.layers import Entry

__all__ = [
    "InjectionScanSpec",
    "LayerSelectionStudy",
    "ScanRunConfig",
    "check_study",
    "injection_scan",
    "run_study",
    "scan_chunks",
    "study_output",
]


class InjectionScanSpec(WrittenScan):
    """Single-layer write scan of one receiver.

    The shared fields are :class:`~latent_interconnect.evaluation.scan_states.WrittenScan`'s.

    Attributes:
        layers: layers to scan (every layer for a new receiver).
        alpha: the write weight of the one write rule at the scanned layer (1 = the state's direction at the stream's
            norm). The original scans overwrote the stream, which ``interp_v1`` cannot express, so no default.
        layers_per_call: how many scanned layers share one backend call: ``layers``, in order, cut into
            consecutive chunks of this many (the last may be shorter; :func:`scan_chunks`). Each row of a call still
            writes at its own one layer. It changes the micro-batches, so the numbers (bf16), and is identity.
    """

    layers: tuple[int, ...] = Field(min_length=1)
    alpha: float = Field(gt=0.0, le=1.0)
    layers_per_call: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_layers(self) -> InjectionScanSpec:
        if len(set(self.layers)) != len(self.layers):
            raise ValueError(f"the scan's layers repeat: {self.layers}")
        return self


class ScanRunConfig(StageConfig):
    """One standalone receiver scan (the ``scan`` stage's config), as ``SuiteRunConfig`` is one suite execution.

    The scan says what is measured; this says where and how it runs. A study's scan runs under the study's own
    ``device_class`` (:class:`LayerSelectionStudy`), so the scan spec carries none. A domain's own-state reference is
    one of these (source ``oracle``), shared by every setup with the same receiver-side spec. An injection scan scores
    the set it declares; a set comparison is study evidence and refuses objects outside ``train`` itself
    (``layer_sets.compare_layer_sets``).

    Attributes:
        scan: an injection scan or a layer-set comparison (inline or ``$ref``).
        device_class: the declared hardware (``A100``).
        plan: the batch plan the scoring backend is built with (the deterministic default); part of the
            identity.
    """

    scan: InjectionScanSpec | LayerSetComparisonSpec
    device_class: DeviceClass
    plan: BatchPlan = BatchPlan()


class LayerSelectionStudy(StageConfig):
    """A setup's encoder-dependent receiver diagnostic: its held-out ridge injection scan (the ``study`` stage).

    With the shared own-state reference (a ``scan`` record) and the encoder's read-layer scan (the ``read_scan`` stage)
    these are the full diagnostic a person reads before choosing layers (the protocol: a thorough diagnostic scan,
    then a declared choice for coverage and the high-value layers). Candidate write sets are compared once they exist,
    by the ``scan`` stage (``layer_sets.LayerSetComparisonSpec``), without re-running the diagnostic. The
    ``layer_selection`` analysis (``evaluation.layer_selection_analysis``) assembles these records and checks they are
    one setup's.

    Attributes:
        name: e.g. ``"kimi-k2.6/esmc-600m"``.
        injection: injection scan with this encoder's held-out scan ridge's predictions (source ``ridge``).
        device_class: the declared scoring hardware (``A100``); the scan refuses another kind or topology, not
            another accelerator model (``DeviceClass.matches``).
        plan: the batch plan the scoring backend is built with (the deterministic default); part of the
            identity.
    """

    name: str
    injection: InjectionScanSpec
    device_class: DeviceClass
    plan: BatchPlan = BatchPlan()

    @model_validator(mode="after")
    def _check_scan(self) -> LayerSelectionStudy:
        if self.injection.source != "ridge":
            raise ValueError(
                f"study {self.name!r}: the study's scan writes source 'ridge' (the held-out scan ridge; the "
                f"own-state reference is a scan record), not {self.injection.source!r}"
            )
        return self


def scan_chunks(spec: InjectionScanSpec) -> tuple[tuple[int, ...], ...]:
    """Return the scan's canonical chunk plan: its layers, in order, in consecutive chunks of ``layers_per_call``.

    A function of the spec alone (never of what a run finished), so a resumed or distributed run scores the same calls.
    """
    n = spec.layers_per_call
    return tuple(spec.layers[i : i + n] for i in range(0, len(spec.layers), n))


def injection_scan(spec: InjectionScanSpec, ctx: SuiteContext) -> ScanOutputs:
    """Run the scan; commit per-layer AUROC with intervals and the per-object scores.

    At each layer the object's state (``oracle``: its captured state at that layer; ``ridge``: the ridge prediction
    for that layer) is written at the one slot token with weight ``alpha``, and the tasks are scored as a ``yes_no``
    suite on the item set's ``own`` condition (prompts vetted, hardware checked). The layers are scored in the chunks
    of :func:`scan_chunks`, one backend call per chunk (``YesNo.score_groups``, one group per layer): a row writes at
    its own layer only. The per-object scores of all layers are one ``scores`` artifact (``step`` 0) with the scan
    columns ``layer`` and ``source`` (``evaluation.scores.SCAN_COLUMNS``), in chunk order (layer order within a chunk,
    then the suite's row order). The ``numbers`` artifact holds, per layer and task, the AUROC with its interval over
    ``units`` (``layer``, ``task``, ``value``, ``lo``, ``hi``, ``dropped``; key ``<layer>/<task>``) and cites the
    scores in its metadata (``scores``). Both are returned (:class:`ScanOutputs`), and a stage that runs a scan records
    both as outputs. The combined table's metadata holds the chunk plan (``chunks``) and names each chunk's own score
    artifact (``parts``, in chunk order), whose metadata holds that call's batch plan and partition digest (``batch``
    indexes that call's partition) and its layers.

    Resumable by whole chunks: each chunk's score artifact is remembered under a key of (scan digest, backend, plan,
    chunk, origin) (``ArtifactStore.remember``) once its call has finished, so a rerun of the same scan after a kill
    reuses the chunks already committed and reruns the rest, a partly scored chunk whole; the chunks never change
    with what finished, so the combined record equals an uninterrupted run's. The origin is the code (commit, dirty)
    and the execution (device class, ``backend.execution()`` but the weights directory, as
    ``build.capture.capture_origin``): a rerun on other code or hardware finds no part and recomputes every chunk, so
    parts are never mixed, and a dirty tree never reuses. A reused chunk is checked first (files against its
    manifest, the same suite, step, plan and layers, finite scores for exactly the scan's tasks). Before anything is
    committed, the assembled scores must cover each (layer, item, task) once, with the same (item, task) rows at
    every layer.

    The scan scores whatever single-role item set it is given: the train-only rule belongs to studies
    (:func:`run_study`); a standalone scan (the ``scan`` stage) may score any set.

    Raises:
        ValueError: if the item set has more than one role, or a task has one label class among its in-scope objects
            (checked before anything is scored), a remembered chunk fails its check, or the assembled scores do not
            cover the scan exactly.
    """
    items = ctx.items(spec.item_set)
    if len(items.roles) != 1:
        raise ValueError(f"an injection scan scores one role, the item set has {sorted(items.roles)}")
    role = next(iter(items.roles))
    objects = items.objects[role]
    require_label_classes(ctx.domain(spec.domain), objects, spec.tasks, "injection scan")
    suite = scan_suite(spec, role, f"injection_scan/{spec.source}")
    runner = YesNo(suite)
    columns = unit_columns(items, spec.units.factors)
    vectors = scan_states(spec, ctx, objects)
    chunks = scan_chunks(spec)
    tables, parts = [], []
    plan = None if ctx.plan is None else ctx.plan.identity()  # an added field at its default left out
    commit, dirty = ctx.code
    execution = {k: v for k, v in ctx.backend.execution().items() if k != "weights"}
    origin = {"commit": commit, "dirty": dirty, "device_class": ctx.device_class, "execution": execution}
    for chunk in chunks:
        key = "scan-chunk-" + canonical_sha256([spec.digest(), ctx.backend_name, plan, list(chunk), origin])
        ref = None if dirty else ctx.artifacts.committed(key)
        if ref is None:
            groups = [{SCAN_CONNECTOR: StateWrites(objects, {L: vectors(L)}, {Entry(0, L): spec.alpha})} for L in chunk]
            scored, meta = runner.score_groups(ctx, groups, 0)
            table = pd.concat([t.assign(layer=L, source=spec.source) for L, t in zip(chunk, scored, strict=True)])
            ref = write_scores(ctx.artifacts, table, {**meta, "layers": list(chunk)})
            if not dirty:  # a duplicate writer's chunk, if one was remembered first, is the key's
                ref = ctx.artifacts.remember(key, ref)
        else:
            _check_resumed(ctx.artifacts, ref, suite.digest(), plan, chunk=chunk, tasks=spec.tasks)
        parts.append(ref)
        tables.append(read_scores(ctx.artifacts, ref))
    scores_table = pd.concat(tables, ignore_index=True)
    _check_coverage(scores_table, spec.layers)
    rows = []
    for layer in spec.layers:
        table = scores_table[scores_table["layer"] == layer]
        estimates = {t: task_estimate(table[table["task"] == t], "auroc", columns) for t in spec.tasks}
        rows += [
            {
                "key": f"{layer}/{t}",
                "layer": layer,
                "task": t,
                "value": e.estimate,
                "lo": e.lo,
                "hi": e.hi,
                "dropped": e.dropped,
            }
            for t, e in estimates.items()
        ]
    meta = {"scan": spec.digest(), "chunks": [list(c) for c in chunks], "parts": [r.sha256 for r in parts]}
    scores = write_scores(ctx.artifacts, scores_table, meta)
    numbers = write_numbers(ctx.artifacts, pd.DataFrame(rows), {"scan": spec.digest(), "scores": scores.sha256})
    return ScanOutputs(numbers=numbers, scores=scores)


def _check_resumed(
    store: ArtifactStore,
    ref: ArtifactRef,
    suite: str,
    plan: object,
    *,
    chunk: tuple[int, ...],
    tasks: tuple[str, ...],
) -> None:
    """Refuse a remembered chunk that is not this scan's complete, finite, intact score table of ``chunk``."""
    store.verify(ref)
    meta = store.manifest(ref).meta
    table = read_scores(store, ref)
    ok = meta.get("suite") == suite and meta.get("step") == 0 and meta.get("batch_plan", {}).get("plan") == plan
    ok = ok and meta.get("layers") == list(chunk) and set(table.get("layer", ())) == set(chunk)
    if not (ok and set(table["task"]) == set(tasks) and np.isfinite(table["score"].to_numpy(np.float64)).all()):
        raise ValueError(f"injection scan: remembered chunk scores {ref.short} are not this scan's; delete its pointer")


def _check_coverage(table: pd.DataFrame, layers: tuple[int, ...]) -> None:
    """Refuse assembled scan scores unless each (layer, item, task) occurs once, at every layer the same rows.

    Raises:
        ValueError: naming the repeated rows, or the layers missing, extra or with other rows than the first.
    """
    keys = table[["layer", "item", "task"]]
    repeated = keys[keys.duplicated()]
    if not repeated.empty:
        raise ValueError(f"injection scan: rows scored twice, e.g. {repeated.head(3).to_dict('records')}")
    rows = {int(L): set(zip(g["item"], g["task"], strict=True)) for L, g in keys.groupby("layer")}
    expected = rows.get(layers[0], set())
    wrong = sorted(set(rows) ^ set(layers)) + sorted(L for L in rows if L in layers and rows[L] != expected)
    if wrong or not expected:
        raise ValueError(
            f"injection scan: layers {wrong} are missing, extra or cover other rows than layer {layers[0]}"
        )


def run_study(study: LayerSelectionStudy, ctx: SuiteContext) -> dict[str, ArtifactRef]:
    """Run the study's injection scan and return its outputs by name.

    The object list is checked against the ``train`` split first, so nothing runs on a study that would see
    ``val`` or ``test`` (the ``study`` stage checks it again before it loads the receiver). Output names
    (:func:`study_output`): ``injection/numbers``, ``injection/scores``.

    Raises:
        ValueError: if the item set holds an object outside ``train`` (``domains.require_splits``).
    """
    check_study(study, ctx.items, ctx.domain)
    scan = injection_scan(study.injection, ctx)
    return {study_output("numbers"): scan.numbers, study_output("scores"): scan.scores}


def study_output(artifact: Literal["numbers", "scores"]) -> str:
    """Return the output name of one of :func:`run_study`'s artifacts (e.g. ``injection/numbers``)."""
    return f"injection/{artifact}"


def check_study(
    study: LayerSelectionStudy, items: Callable[[str], ItemSet], domain: Callable[[DomainSpec], Domain]
) -> None:
    """Refuse a study whose scan's objects are not all ``train`` objects.

    Raises:
        ValueError: naming the offending objects.
    """
    spec = study.injection
    require_train(items(spec.item_set), domain(spec.domain), f"study {study.name!r}")
