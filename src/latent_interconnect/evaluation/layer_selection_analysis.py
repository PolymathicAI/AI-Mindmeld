"""The ``layer_selection`` analysis kind: one (receiver, encoder) setup's layer-selection records, checked and typed.

The layer-selection stages (``evaluation.layer_selection``) each record their own numbers. For one
setup, the plots and the table of chosen layers read them through this analysis. It takes
these inputs (roles of ``AnalysisSpec.inputs``):

- ``study``: a ``study`` record (:class:`~latent_interconnect.evaluation.layer_selection.LayerSelectionStudy`): the
  injection scan of the encoder's held-out scan ridge;
- ``oracle``: a ``scan`` record (:class:`~latent_interconnect.evaluation.layer_selection.ScanRunConfig`) whose scan is
  an oracle :class:`~latent_interconnect.evaluation.layer_selection.InjectionScanSpec`: the own-state reference (the
  receiver's own states; a mapped state can beat it), run once per domain and cited by every setup that shares it;
- ``ridge``: the ``ridge`` record (:class:`~latent_interconnect.build.ridge.RidgeSpec` with ``exclude``) whose output
  the study's injection scan writes;
- ``read_scan``: the encoder's ``read_scan`` record
  (:class:`~latent_interconnect.evaluation.read_scan.ReadLayerScanSpec`);
- ``sets`` (optional): a ``scan`` record (:class:`~latent_interconnect.evaluation.layer_selection.ScanRunConfig`)
  whose scan is a :class:`~latent_interconnect.evaluation.layer_sets.LayerSetComparisonSpec`.

It refuses records that are not one setup's (:func:`check_setup`; for the shared own-state reference, every field of
its scan but the written states and its execution config must be the study's). Each record's committed ``numbers``
artifact (``evaluation.scores.read_numbers``) becomes a field of :class:`LayerSelectionResult`. It computes no new
statistic; layers are chosen by hand from these diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import ClassVar, NamedTuple

import pandas as pd
from pydantic import Field

from latent_interconnect.artifacts.records import Record, record_output
from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.build.ridge import RIDGE_KIND, RidgeSpec
from latent_interconnect.config import Config
from latent_interconnect.data.items import ItemSet
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS
from latent_interconnect.evaluation.layer_selection import (
    InjectionScanSpec,
    LayerSelectionStudy,
    ScanRunConfig,
    study_output,
)
from latent_interconnect.evaluation.layer_sets import LayerSetComparisonSpec
from latent_interconnect.evaluation.read_scan import ReadLayerScanSpec
from latent_interconnect.evaluation.results import AnalysisResult, Estimate
from latent_interconnect.evaluation.scan_states import WrittenScan
from latent_interconnect.evaluation.scores import NUMBERS_KIND, read_numbers
from latent_interconnect.stats.bootstrap import ResamplingUnits

__all__ = [
    "LayerSelection",
    "LayerSelectionKind",
    "LayerSelectionResult",
    "LayerSetsResult",
    "SetEstimates",
    "Setup",
    "check_setup",
    "mean_per_layer",
]

_STAGES = {"study": "study", "oracle": "scan", "ridge": "ridge", "read_scan": "read_scan", "sets": "scan"}
"""Input role → the stage (``Record.kind``) whose record it takes."""
_REQUIRED = ("study", "oracle", "ridge", "read_scan")
_WRITTEN = frozenset({"source", "states", "input", "embeddings", "encoder", "pool", "source_floor"})
"""The fields of an injection scan that say what is written (``WrittenScan``); the own-state reference shares the
rest. Its ``pool`` and ``source_floor`` are checked against the ridge's instead (:func:`check_setup`); its
``fixed_source`` is not here, so a fixed-source curve (the ridge has none) is never a study's reference."""


class LayerSelection(Config):
    """Parameters of the ``layer_selection`` kind: none (it reads the records' numbers; their configs say the rest)."""


class SetEstimates(Config):
    """One candidate write set's AUROC.

    Attributes:
        tasks: task → AUROC with its interval.
        mean: the equal-weight mean over the tasks, with its interval.
    """

    tasks: dict[str, Estimate]
    mean: Estimate


class LayerSetsResult(Config):
    """A layer-set comparison's numbers.

    Attributes:
        sets: set name → its AUROC per task and mean.
        contrasts: contrast name (``"<a> - <b>"``) → the paired contrast of the means with its interval.
    """

    sets: dict[str, SetEstimates]
    contrasts: dict[str, Estimate] = Field(default_factory=dict)


class LayerSelectionResult(AnalysisResult):
    """Result of ``layer_selection``: one setup's scans, as the plots read them.

    ``units`` are the receiver scans' resampling units.

    Attributes:
        oracle: receiver layer → task → AUROC with its interval, writing the receiver's own states (the shared
            own-state reference record).
        injection: the same, writing the encoder's held-out scan ridge's predictions (the study's scan).
        read: population (``objects``, ``variants``) → probe (a task id or a variant label) → read (its
            ``read_layer`` label: a layer, or a joint read's layers joined by ``+``;
            ``evaluation.read_scan.read_label``) → AUROC with its interval (the read scan).
        read_units: the read scan's resampling units (its objects' clusters).
        sets: the layer-set comparison, or None if the analysis has no ``sets`` input.
    """

    oracle: dict[int, dict[str, Estimate]]
    injection: dict[int, dict[str, Estimate]]
    read: dict[str, dict[str, dict[str, Estimate]]]
    read_units: ResamplingUnits
    sets: LayerSetsResult | None = None


class Setup(NamedTuple):
    """The configs of one setup's input records.

    Attributes:
        study: the study.
        oracle: the own-state reference (the ``oracle`` scan record's scan).
        ridge: the held-out scan ridge whose predictions the study's injection scan writes.
        read_scan: the encoder's read-layer scan.
        sets: the layer-set comparison, or None.
    """

    study: LayerSelectionStudy
    oracle: InjectionScanSpec
    ridge: RidgeSpec
    read_scan: ReadLayerScanSpec
    sets: LayerSetComparisonSpec | None


@ANALYSIS_KINDS.register("layer_selection")
class LayerSelectionKind:
    """A setup's study, own-state reference, ridge, read-scan and optional set-comparison records."""

    Params: ClassVar[type[Config]] = LayerSelection
    Result: ClassVar[type[AnalysisResult]] = LayerSelectionResult

    def run(
        self,
        params: LayerSelection,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> LayerSelectionResult:
        """Check the records are one setup's and type their numbers.

        Raises:
            ValueError: if a role is unknown or missing, a record is not of its role's stage, or the records are not
                one setup's (:func:`check_setup`).
        """
        del params, items  # no parameters; nothing is resampled
        setup = check_setup(inputs, units)
        study, oracle, read_scan = inputs["study"], inputs["oracle"], inputs["read_scan"]
        sets: LayerSetsResult | None = None
        if setup.sets is not None:
            sets = _sets(read_numbers(artifacts, record_output(inputs["sets"], NUMBERS_KIND)), setup.sets)
        injection = _layer_scan(read_numbers(artifacts, _output(study, study_output("numbers"))))
        own = _layer_scan(read_numbers(artifacts, record_output(oracle, NUMBERS_KIND)))
        return LayerSelectionResult(
            units=units,
            oracle=own,
            injection=injection,
            read=_read(read_numbers(artifacts, record_output(read_scan, NUMBERS_KIND))),
            read_units=setup.read_scan.units,
            sets=sets,
        )


def check_setup(inputs: Mapping[str, Record], units: ResamplingUnits) -> Setup:
    """Return the inputs' configs, refusing records that are not one (receiver, encoder) setup's.

    The checks: the study's injection scan writes the ridge record's output, reading the ridge's input and encoder; the
    ridge leaves out exactly the study's item set (``exclude``), fits the read scan's object list and the capture state
    the own-state reference writes (the same capture, ``pool`` and ``source_floor``), and reads the read scan's
    encoder; the own-state reference is the study's scan in every other field (receiver, domain, item set, tasks,
    units, frame, slot, answers, layers, alpha; so no ``fixed_source``) and in its execution config (device class,
    batch plan); every record reads one domain at one data version; a set comparison scores the study's receiver with
    the study's states for its source (an oracle one: the reference's capture, ``pool``, ``source_floor`` and
    ``fixed_source``); the receiver scans score one item set with ``units`` (the read scan has its own units).

    Raises:
        ValueError: naming the unknown, missing or mis-staged input, or the mismatch.
    """
    setup = _configs(inputs)
    _check_ridge(setup, record_output(inputs["ridge"], RIDGE_KIND).sha256)
    _check_oracle(setup, ScanRunConfig.model_validate(inputs["oracle"].config))
    _check_domain(setup)
    _check_receiver_scans(setup, units)
    return setup


def _configs(inputs: Mapping[str, Record]) -> Setup:
    """Return the inputs' configs, refusing an unknown, missing or mis-staged input.

    Also refused: an ``oracle`` scan record that is not an oracle injection scan, and a ``sets`` one that is not a set
    comparison.
    """
    unknown = sorted(set(inputs) - set(_STAGES))
    missing = [r for r in _REQUIRED if r not in inputs]
    if unknown or missing:
        raise ValueError(
            f"layer_selection takes inputs {sorted(_STAGES)} ('sets' optional): unknown {unknown}, missing {missing}"
        )
    for role, record in inputs.items():
        if record.kind != _STAGES[role]:
            raise ValueError(f"input {role!r} is a {record.kind!r} record, not a {_STAGES[role]!r} record")
    oracle = ScanRunConfig.model_validate(inputs["oracle"].config).scan
    if not isinstance(oracle, InjectionScanSpec) or oracle.source != "oracle":
        raise ValueError("input 'oracle' is a scan record of another scan, not of an oracle injection scan")
    sets = None
    if "sets" in inputs:
        scan = ScanRunConfig.model_validate(inputs["sets"].config).scan
        if not isinstance(scan, LayerSetComparisonSpec):
            raise ValueError("input 'sets' is a scan record of an injection scan, not of a layer-set comparison")
        sets = scan
    return Setup(
        study=LayerSelectionStudy.model_validate(inputs["study"].config),
        oracle=oracle,
        ridge=RidgeSpec.model_validate(inputs["ridge"].config),
        read_scan=ReadLayerScanSpec.model_validate(inputs["read_scan"].config),
        sets=sets,
    )


def _check_ridge(setup: Setup, fitted: str) -> None:
    """Refuse a study, ridge or read scan that is not one held-out scan ridge's (``fitted``: the ridge's output)."""
    study, ridge, read_scan = setup.study.injection, setup.ridge, setup.read_scan
    if study.states != fitted:
        raise ValueError(f"the study's injection scan writes {study.states[:12]}, not the ridge's output {fitted[:12]}")
    if study.input != ridge.input or study.encoder is None or study.encoder.digest() != ridge.encoder.digest():
        raise ValueError("the study's injection scan does not read the ridge's input and encoder")
    if ridge.exclude != study.item_set:
        raise ValueError(
            f"the ridge leaves out {str(ridge.exclude)[:12]}, not the study's item set {study.item_set[:12]} "
            "(the held-out scan ridge never sees the scanned objects)"
        )
    if setup.oracle.states != ridge.targets:
        raise ValueError(
            f"the own-state reference writes capture {setup.oracle.states[:12]}, the ridge fits {ridge.targets[:12]}"
        )
    if setup.oracle.pool != ridge.pool:
        raise ValueError(
            f"the own-state reference writes the capture's pool {setup.oracle.pool!r} state, the ridge fits its "
            f"pool {ridge.pool!r} state (None: the capture's stored pooled state)"
        )
    if setup.oracle.source_floor != ridge.source_floor:
        raise ValueError(
            f"the own-state reference writes source_floor {setup.oracle.source_floor!r} states, the ridge fits "
            f"source_floor {ridge.source_floor!r} states (None: each layer's own state)"
        )
    if read_scan.encoder.digest() != ridge.encoder.digest():
        raise ValueError(
            f"the read scan scans encoder {read_scan.encoder.name!r}, the ridge reads {ridge.encoder.name!r} "
            "(or another recipe of it)"
        )
    if read_scan.objects != ridge.fit_set:
        raise ValueError(
            f"the read scan fits object list {read_scan.objects[:12]}, the ridge {ridge.fit_set[:12]} "
            "(one study population)"
        )


def _check_oracle(setup: Setup, run: ScanRunConfig) -> None:
    """Refuse an own-state reference (``run``: its record's config) unless it is the study's scan, run the same way.

    Every field but the written states' (``_WRITTEN``) must equal the study's scan's, and the device class and batch
    plan the study's.
    """
    study = setup.study
    mine, theirs = study.injection.model_dump(exclude=set(_WRITTEN)), setup.oracle.model_dump(exclude=set(_WRITTEN))
    execution = {"device_class": run.device_class, "plan": run.plan}
    differ = sorted(k for k in mine if mine[k] != theirs[k])
    differ += [k for k, v in execution.items() if v != getattr(study, k)]
    if differ:
        raise ValueError(f"the own-state reference record is not this study's spec: it differs in {differ}")


def _check_domain(setup: Setup) -> None:
    """Refuse inputs that read different domains or data versions."""
    domains = {
        "study": setup.study.injection.domain,
        "ridge": setup.ridge.domain,
        "read_scan": setup.read_scan.domain,
    }
    if setup.sets is not None:
        domains["sets"] = setup.sets.domain
    if len({d.digest() for d in domains.values()}) != 1:
        read = {role: f"{d.name}@{d.data[:12]}" for role, d in domains.items()}
        raise ValueError(f"the inputs read different domains (or data versions): {read}")


def _check_receiver_scans(setup: Setup, units: ResamplingUnits) -> None:
    """Refuse a set comparison of another receiver or other states, and receiver scans of other items or units."""
    study, sets = setup.study, setup.sets
    scans: dict[str, WrittenScan] = {"oracle": setup.oracle, "injection": study.injection}
    if sets is not None:
        if sets.receiver.digest() != study.injection.receiver.digest():
            raise ValueError(f"the set comparison scores receiver {sets.receiver.name!r}, the study another")
        written = study.injection.states if sets.source == "ridge" else setup.oracle.states
        if sets.states != written:
            raise ValueError(
                f"the set comparison (source {sets.source!r}) writes {sets.states[:12]}, the study's {sets.source} "
                f"states are {written[:12]}"
            )
        mine = (sets.pool, sets.source_floor, sets.fixed_source)
        ref = (setup.oracle.pool, setup.oracle.source_floor, setup.oracle.fixed_source)
        if sets.source == "oracle" and mine != ref:
            raise ValueError(
                f"the set comparison writes the capture's pool {sets.pool!r} state, the own-state reference its "
                f"pool {setup.oracle.pool!r} state (source_floor {sets.source_floor!r}, "
                f"{setup.oracle.source_floor!r}; fixed_source {sets.fixed_source!r}, {setup.oracle.fixed_source!r})"
            )
        scans["sets"] = sets
    for field, expected in (("item_set", study.injection.item_set), ("units", units)):
        differ = {name: getattr(scan, field) for name, scan in scans.items() if getattr(scan, field) != expected}
        if differ:
            raise ValueError(f"the receiver scans {sorted(differ)} have {field} {differ}, expected {expected!r}")


def _output(record: Record, name: str) -> ArtifactRef:
    out = record.outputs[name]
    return ArtifactRef(out.sha256, out.kind)


def _estimate(row: pd.Series) -> Estimate:
    return Estimate(estimate=float(row["value"]), lo=float(row["lo"]), hi=float(row["hi"]), dropped=int(row["dropped"]))


def mean_per_layer(scan: Mapping[int, Mapping[str, Estimate]]) -> dict[int, float]:
    """Return layer → the equal-weight mean of the estimates at that layer: a scan's curve (the rule's, the plots')."""
    return {L: sum(e.estimate for e in tasks.values()) / len(tasks) for L, tasks in sorted(scan.items())}


def _layer_scan(numbers: pd.DataFrame) -> dict[int, dict[str, Estimate]]:
    """Return layer → task → estimate of an injection scan's numbers (``layer_selection.injection_scan``)."""
    out: dict[int, dict[str, Estimate]] = {}
    for _, row in numbers.iterrows():
        out.setdefault(int(row["layer"]), {})[str(row["task"])] = _estimate(row)
    return out


def _read(numbers: pd.DataFrame) -> dict[str, dict[str, dict[str, Estimate]]]:
    """Return population → probe → read label → estimate of a read scan's numbers (``ReadScanOutputs``)."""
    out: dict[str, dict[str, dict[str, Estimate]]] = {}
    for _, row in numbers.iterrows():
        by_name = out.setdefault(str(row["population"]), {})
        by_name.setdefault(str(row["name"]), {})[str(row["read_layer"])] = _estimate(row)
    return out


def _sets(numbers: pd.DataFrame, spec: LayerSetComparisonSpec) -> LayerSetsResult:
    """Return a set comparison's numbers (``layer_sets.compare_layer_sets``).

    Raises:
        ValueError: if a row's ``kind`` is not ``task``, ``mean`` or ``contrast``.
    """
    tasks: dict[str, dict[str, Estimate]] = {}
    means: dict[str, Estimate] = {}
    contrasts: dict[str, Estimate] = {}
    for _, row in numbers.iterrows():
        name = str(row["layer_set"])
        match row["kind"]:
            case "task":
                tasks.setdefault(name, {})[str(row["task"])] = _estimate(row)
            case "mean":
                means[name] = _estimate(row)
            case "contrast":
                contrasts[str(row["key"])] = _estimate(row)
            case other:
                raise ValueError(f"set comparison numbers: row {row['key']!r} has kind {other!r}")
    sets = {s.name: SetEstimates(tasks=tasks[s.name], mean=means[s.name]) for s in spec.sets}  # the spec's order
    return LayerSetsResult(sets=sets, contrasts=contrasts)
