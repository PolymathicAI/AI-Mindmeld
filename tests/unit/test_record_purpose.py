"""Record purpose classes (result, monitor, anchor) and stored records written under the earlier names.

The golden texts and their ids and keys below were produced by the code before the rename (``paper``/``in_job``,
commit a34c683a): reading them under the current code must verify them byte for byte and classify them afterwards.
A stored analysis record of the earlier kind name ``cross_domain_e1`` is checked the same way.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from tests.fakes import make_workspace

from latent_interconnect.artifacts.records import (
    LEGACY_PURPOSE,
    OMITTED_PURPOSE,
    Record,
    RecordRef,
    RecordStore,
    purpose_class,
    strictest_purpose,
)
from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.config import ArtifactDigest, Config, Pin, canonical_json
from latent_interconnect.evaluation.analysis import (
    ANALYSIS_KINDS,
    LEGACY_ANALYSIS_KINDS,
    AnalysisSpec,
    analysis_kind,
    load_result,
)
from latent_interconnect.evaluation.cross_domain_quads import CrossDomainQuadsKind, CrossDomainQuadsResult, QuadDesign
from latent_interconnect.evaluation.results import Estimate
from latent_interconnect.provenance import Provenance
from latent_interconnect.run._main import StageResult
from latent_interconnect.workspace import Workspace

_B, _C, _D, _E = "b" * 64, "c" * 40, "d" * 64, "e" * 64
_PROV = (
    '"provenance":{"argv":["suite"],"commit":"' + _C + '","devices":[],"dirty":false,'
    '"ended":"2026-10-19T00:00:01+00:00","execution":{},"host":"h","input_executions":{},"job_id":null,'
    '"packages":{},"pins":{},"python":"3.11","resumed":[],"started":"2026-10-19T00:00:00+00:00","version":"0"}'
)
_SUITE = (
    '{"config":{},"config_sha":"'
    + "a" * 64
    + '","input_records":{},"inputs":{"x":{"kind":"scores","sha256":"'
    + _B
    + '"}},"kind":"suite","outputs":{},'
    + _PROV
)
GOLDEN = {  # stored value → (stored text, record_id, computation_key), all from the code before the rename
    "paper": (
        _SUITE + ',"purpose":"paper"}',
        "a4b02804d9b4f7a4c594f44948cc236ef6d79ec1c61d5a37e1bcfc630304ef27",
        "cd736ad3f837e7d94315a23d67887ac8c0741bee99bab76dfb6398c0098409ee",
    ),
    "anchor": (
        _SUITE + ',"purpose":"anchor"}',
        "4fefe179983c7dc2a957fccc6cdf609b8ee835cffd9b9e7fbd3ed0fe9f7b5289",
        "0de2aeea4ef3a79b4fb33cbb0c2b37117ec749b447bf65862492381f3509ea05",
    ),
    "in_job": (
        _SUITE + ',"purpose":"in_job"}',
        "97482c13747c796a93fe3459aab3a19e3a55ede8b37a4ee9b5c19efe5049fe7b",
        "0f7db91ea4c37296c24970500330b3611e394588f457ec903967acc30090b822",
    ),
    None: (  # the field omitted: the old default, so the same identity as a stored "paper"
        _SUITE + "}",
        "a4b02804d9b4f7a4c594f44948cc236ef6d79ec1c61d5a37e1bcfc630304ef27",
        "cd736ad3f837e7d94315a23d67887ac8c0741bee99bab76dfb6398c0098409ee",
    ),
}
OLD_SPEC = (
    '{"id":"toy/xdom","inputs":{"xd":"' + _D + '"},"kind":"cross_domain_e1","params":{"cells":{"a":{"conditions":'
    '{"protein":"own"},"encoder":"xd","inputs":["xd"],"step":0},"b":{"conditions":{"protein":"name"},"encoder":"xd",'
    '"inputs":["xd"],"step":0}},"contrasts":[{"a":"a","b":"b","name":"a minus b"}]},"units":{"factors":'
    '["protein_cluster","molecule_group"]}}'
)
OLD_SPEC_DIGEST = "47d33b9438f9f888cc90cb106f075fc61379d283841443f2b2ebb0481094f443"


def _store_as_written(root: Path, text: str, record_id: str, key: str) -> Path:
    """Place a stored record's exact bytes where the store keeps it (as older code wrote it)."""
    path = root / key / f"{record_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _prov() -> Provenance:
    return Provenance(
        version="0", commit=_C, dirty=False, argv=("suite",), host="h", job_id=None, python="3.11", packages={},
        devices=(), started="2026-10-19T00:00:00+00:00", ended="2026-10-19T00:00:01+00:00",
    )  # fmt: skip


def _record(purpose: str, *, outputs: dict[str, RecordRef] | None = None, config_sha: str = "a" * 64) -> Record:
    return Record(
        kind="suite", purpose=purpose, config_sha=config_sha, config={}, provenance=_prov(),
        inputs={"x": RecordRef(sha256=_B, kind="scores")}, outputs=outputs or {},
    )  # fmt: skip


@pytest.mark.parametrize("stored", list(GOLDEN))
def test_golden_stored_records_verify_unchanged_and_classify_after(tmp_path, stored):
    """Each golden record loads and verifies with its old id and key, keeps its stored value, and is then classified
    through the legacy map; its file is never rewritten."""
    text, record_id, key = GOLDEN[stored]
    path = _store_as_written(tmp_path, text, record_id, key)
    record = RecordStore(tmp_path).get(record_id)
    assert (record.record_id(), record.computation_key()) == (record_id, key)
    assert record.purpose == (stored or OMITTED_PURPOSE)
    assert record.purpose_class() == LEGACY_PURPOSE[stored or OMITTED_PURPOSE]
    assert RecordStore(tmp_path).canonical(key).record_id() == record_id
    assert path.read_text() == text
    if stored is not None:  # an explicit stored value dumps back to the very bytes it was read from
        assert canonical_json(record.model_dump(mode="json")) == text


def test_an_omitted_purpose_keeps_the_old_default_identity():
    """An omitted purpose decodes as ``paper`` (the old default), never as the new writers' ``result``."""
    text, record_id, key = GOLDEN[None]
    record = Record.model_validate_json(text)
    assert record.purpose == OMITTED_PURPOSE == "paper"
    assert (record.record_id(), record.computation_key()) == (record_id, key) == GOLDEN["paper"][1:]
    as_result = record.model_copy(update={"purpose": "result"})
    assert as_result.record_id() != record_id and as_result.computation_key() != key


def test_new_writers_write_current_values_and_refuse_legacy_ones(tmp_path):
    """``RecordStore.put`` writes result/monitor/anchor explicitly and refuses paper/in_job; the writers' defaults
    are ``result``."""
    store = RecordStore(tmp_path)
    for value in ("result", "monitor", "anchor"):
        rid = store.put(_record(value))
        assert f'"purpose":"{value}"' in (tmp_path / _record(value).computation_key() / f"{rid}.json").read_text()
    for legacy in ("paper", "in_job"):
        with pytest.raises(ValueError, match="legacy value"):
            store.put(_record(legacy))
    assert inspect.signature(Workspace.record).parameters["purpose"].default == "result"
    assert StageResult(outputs={}).purpose == "result"


@pytest.mark.parametrize("value", ["", "Paper", "dry_run", "monitored", "results"])
def test_unknown_purpose_values_fail_and_never_count_as_result(value):
    """An unknown value fails validation of a record and classification; it never falls through to ``result``."""
    with pytest.raises(ValidationError):
        _record(value)
    with pytest.raises(ValidationError):
        Record.model_validate_json(GOLDEN["paper"][0].replace('"purpose":"paper"', f'"purpose":"{value}"'))
    with pytest.raises(ValueError, match="unknown record purpose"):
        purpose_class(value)
    with pytest.raises(ValueError, match="unknown record purpose"):
        strictest_purpose(["result", value])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["result"], "result"),
        (["paper"], "result"),
        (["paper", "result"], "result"),
        (["result", "in_job"], "monitor"),
        (["paper", "monitor"], "monitor"),
        (["in_job", "monitor", "paper"], "monitor"),
        (["result", "anchor"], "anchor"),
        (["paper", "in_job", "anchor"], "anchor"),
        (["monitor", "anchor", "result"], "anchor"),
    ],
)
def test_mixed_old_and_new_lineage_takes_the_strictest_class(values, expected):
    """anchor > monitor > result over current and legacy values alike."""
    assert strictest_purpose(values) == expected


def test_workspace_purpose_over_mixed_lineage_and_missing_producers(tmp_path):
    """Workspace.purpose: input records and producing records count by class, legacy ones classified; an input
    artifact with no producing record makes the record anchor; neither restricted class is ever promoted."""
    ws = make_workspace(tmp_path)
    made = {}
    for value, sha in (("paper", "1" * 64), ("in_job", "2" * 64), ("anchor", "3" * 64), ("result", "4" * 64)):
        rec = _record(value, outputs={"scores": RecordRef(sha256=sha, kind="scores")}, config_sha=sha)
        _store_as_written(ws.paths.records, canonical_json(rec.model_dump(mode="json")), rec.record_id(),
                          rec.computation_key())  # fmt: skip
        made[value] = (ArtifactRef(sha, "scores"), rec)
    art = {v: {"x": ref} for v, (ref, _) in made.items()}
    rec = {v: {"x": r} for v, (_, r) in made.items()}
    assert ws.purpose("result", art["paper"], {}) == "result"
    assert ws.purpose("result", art["result"], rec["paper"]) == "result"
    assert ws.purpose("result", art["in_job"], rec["result"]) == "monitor"
    assert ws.purpose("result", art["paper"], rec["in_job"]) == "monitor"
    assert ws.purpose("monitor", art["paper"], rec["result"]) == "monitor"
    assert ws.purpose("monitor", art["anchor"], {}) == "anchor"
    assert ws.purpose("result", {"x": ArtifactRef("5" * 64, "scores")}, rec["paper"]) == "anchor"  # no producer
    assert ws.purpose("anchor", art["paper"], rec["result"]) == "anchor"
    both = art["paper"] | {"y": made["in_job"][0]}
    assert ws.purpose("result", both, {}) == "monitor"


class _Cites(Config):
    x: ArtifactDigest


def test_input_check_accepts_a_legacy_result_and_refuses_the_restricted_classes(tmp_path):
    """Workspace.input_problems: an artifact made by a stored ``paper`` or a new ``result`` record is eligible; one
    made only by ``in_job``/``monitor``/``anchor`` records, or by none, is refused."""
    ws = make_workspace(tmp_path)
    for value in ("paper", "result", "in_job", "monitor", "anchor", None):
        staged = ws.artifacts.staging(f"t-{value}")
        (staged / "x.txt").write_text(str(value))
        ref = ws.artifacts.commit(staged, "scores", {})
        if value is not None:
            r = _record(value, outputs={"scores": RecordRef(sha256=ref.sha256, kind="scores")}, config_sha=ref.sha256)
            _store_as_written(ws.paths.records, canonical_json(r.model_dump(mode="json")), r.record_id(),
                              r.computation_key())  # fmt: skip
        ws.pin_all({f"p-{value}": Pin(target="artifact", sha256=ref.sha256)})
        config = tmp_path / f"{value}.yaml"
        config.write_text(f"x: {{$pin: p-{value}}}\n")
        problems = ws.input_problems(config, _Cites)
        if value in ("paper", "result"):
            assert problems == [], value
        else:
            assert len(problems) == 1 and "not produced by the new code" in problems[0], (value, problems)


def test_a_legacy_analysis_kind_reads_and_dispatches_with_its_identity_unchanged(tmp_path):
    """An analysis config and record of kind ``cross_domain_e1`` keep their bytes, config digest, computation key and
    record id; the kind dispatches to cross_domain_quads, validates its params and loads its result."""
    assert LEGACY_ANALYSIS_KINDS["cross_domain_e1"] == "cross_domain_quads"
    assert "cross_domain_e1" not in ANALYSIS_KINDS  # only the current name is registered
    assert analysis_kind("cross_domain_e1") is analysis_kind("cross_domain_quads") is CrossDomainQuadsKind
    spec = AnalysisSpec.model_validate_json(OLD_SPEC)
    assert spec.kind == "cross_domain_e1" and spec.digest() == OLD_SPEC_DIGEST
    assert canonical_json(spec.model_dump(mode="json")) == OLD_SPEC
    renamed = spec.model_copy(update={"kind": "cross_domain_quads"})
    assert renamed.digest() != OLD_SPEC_DIGEST  # a new config is a new identity
    with pytest.raises(ValidationError):  # the old name still validates the kind's params
        AnalysisSpec.model_validate_json(OLD_SPEC.replace('"contrasts":[{"a":"a"', '"contrasts":[{"a":"zz"'))

    ws = make_workspace(tmp_path)
    result = CrossDomainQuadsResult(
        quads=2, unit_counts={"protein_cluster": 2}, empty_draws=0,
        cells={
            "a": Estimate(estimate=0.75, lo=0.5, hi=1.0, dropped=0),
            "b": Estimate(estimate=0.5, lo=0.25, hi=0.75, dropped=0),
        },
    )  # fmt: skip
    staged = ws.artifacts.staging("legacy-analysis")
    (staged / "result.json").write_text(result.model_dump_json(indent=1))
    ref = ws.artifacts.commit(staged, "analysis", {"spec": OLD_SPEC_DIGEST, "kind": "cross_domain_e1"})
    record = Record(
        kind="analysis", purpose="paper", config_sha=OLD_SPEC_DIGEST, config=spec.model_dump(mode="json"),
        input_records={"xd": _D}, outputs={"result": RecordRef(sha256=ref.sha256, kind=ref.kind)}, provenance=_prov(),
    )  # fmt: skip
    text = canonical_json(record.model_dump(mode="json"))
    _store_as_written(ws.paths.records, text, record.record_id(), record.computation_key())
    stored = ws.records.get(record.record_id())
    assert canonical_json(stored.model_dump(mode="json")) == text
    loaded = load_result(ws.artifacts, stored, CrossDomainQuadsResult)
    assert isinstance(loaded, CrossDomainQuadsResult) and loaded == result
    assert stored.config["kind"] == "cross_domain_e1" and stored.purpose_class() == "result"


def test_the_golden_legacy_analysis_record_keeps_its_old_identity(tmp_path):
    """The golden ``cross_domain_e1`` analysis record (written before both renames) verifies with its old id and key."""
    text = (
        '{"config":' + OLD_SPEC + ',"config_sha":"' + OLD_SPEC_DIGEST + '","input_records":{"xd":"' + _D + '"},'
        '"inputs":{},"kind":"analysis","outputs":{"result":{"kind":"analysis","sha256":"'
        + _E
        + '"}},'
        + _PROV
        + ',"purpose":"paper"}'
    )
    record_id = "602d1bfa55c3bb098d20e409148ba70926081144b26ff9e9ad4e75da0344615b"
    key = "1a1f88be2b3090189d021b4cdf19391b1ab46f0cca1348eb974da023dab83cbe"
    _store_as_written(tmp_path, text, record_id, key)
    record = RecordStore(tmp_path).get(record_id)
    assert (record.record_id(), record.computation_key()) == (record_id, key)
    assert AnalysisSpec.model_validate(record.config).digest() == record.config_sha == OLD_SPEC_DIGEST


def test_weighted_mean_is_the_former_statistic():
    """QuadDesign.weighted_mean: the unweighted mean without a draw, the product-of-counts weighting in a draw, and
    NaN when every quad weighs 0."""
    quad = np.array([0, 0, 1, 1, 2, 2])
    design = QuadDesign(
        quad=quad,
        units={"f": np.array([[0], [1], [1]])},
        unit_counts={"f": 2},
    )
    q = np.array([1.0, 0.5, 0.0])
    assert design.weighted_mean(q, None) == pytest.approx(0.5)
    assert design.weighted_mean(q, {"f": np.array([0, 0])}) == pytest.approx(1.0)  # unit 1 drawn 0 times
    assert design.weighted_mean(q, {"f": np.array([1, 1])}) == pytest.approx(0.25)
    assert design.weighted_mean(q, {"f": np.array([0, 1])}) == pytest.approx(0.5)
    empty = QuadDesign(quad=np.array([0, 1]), units={"f": np.array([[1], [1]])}, unit_counts={"f": 2})
    assert np.isnan(empty.weighted_mean(np.array([1.0, 0.0]), {"f": np.array([0, 0])}))
