"""The ``cost`` analysis kind (``evaluation.cost``) on an extended run and its source.

An extended run's provenance lists its source's jobs as its first segments; cited with its source, those jobs are
charged once, to the source.
"""

import pytest

from latent_interconnect.artifacts.records import Record, RecordRef
from latent_interconnect.evaluation.cost import Cost, CostKind, cost_of
from latent_interconnect.provenance import Extension, Provenance, Segment
from latent_interconnect.stats.bootstrap import ResamplingUnits

A100, H100 = "NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"


def _record(job: str, started: str, ended: str, devices: tuple[str, ...], **fields) -> Record:
    provenance = Provenance(
        version="1",
        commit="c" * 40,
        dirty=False,
        argv=("train",),
        host="h",
        job_id=job,
        python="3",
        packages={},
        devices=devices,
        started=started,
        ended=ended,
        **fields,
    )
    return Record(
        kind="train",
        purpose="result",
        config_sha=job * 64,
        config={"name": job},
        outputs={"log": RecordRef(sha256="b" * 64, kind="train_log")},
        provenance=provenance,
    )


SOURCE = _record("1", "2026-01-01T00:00:00+00:00", "2026-01-01T02:00:00+00:00", (A100, A100))
"""A 2-hour job on two A100s."""
LEDGER = Segment(
    started="2026-01-01T00:00:30+00:00",
    ended="2026-01-01T02:00:00+00:00",
    host="h",
    job_id="1",
    commit="c" * 40,
    devices=(A100, A100),
)
"""The source's job as its segment ledger holds it (the trainer's segment starts after the stage's own start)."""
EXTENDED = _record(
    "2",
    "2026-01-01T03:00:00+00:00",
    "2026-01-01T04:00:00+00:00",
    (H100,),
    resumed=(LEDGER,),
    extended_from=Extension(
        config_sha="1" * 64,
        computation_key=SOURCE.computation_key(),
        record_id=SOURCE.record_id(),
        step=4,
        segments=1,
    ),
)
"""Its extension: the source's job inherited, then a 1-hour job on one H100."""
INHERITED_H = 2.0 - 30 / 3600


def test_the_source_alone_is_charged_its_job():
    entry = cost_of([SOURCE])
    assert (entry.records, entry.wall_hours, entry.device_hours, entry.cpu_hours) == (1, 2.0, {A100: 4.0}, 0.0)


def test_the_extension_alone_is_charged_its_whole_lineage():
    entry = cost_of([EXTENDED])
    assert entry.wall_hours == pytest.approx(INHERITED_H + 1.0)
    assert entry.device_hours == pytest.approx({A100: 2 * INHERITED_H, H100: 1.0})


@pytest.mark.parametrize("order", [(SOURCE, EXTENDED), (EXTENDED, SOURCE)])
def test_the_source_and_its_extension_count_the_inherited_job_once(order):
    entry = cost_of(list(order))
    assert (entry.records, entry.wall_hours, entry.device_hours) == (2, 3.0, {A100: 4.0, H100: 1.0})


def test_in_separate_rows_the_inherited_job_is_the_source_rows_and_the_rows_sum_to_the_total():
    params = Cost.model_validate({"rows": [{"name": "short", "inputs": ["a"]}, {"name": "long", "inputs": ["b"]}]})
    inputs = {"a": SOURCE, "b": EXTENDED}
    result = CostKind().run(params, ResamplingUnits(factors=("object",)), inputs, None, None)
    assert result.rows["short"].device_hours == {A100: 4.0}
    assert (result.rows["long"].wall_hours, result.rows["long"].device_hours) == (1.0, {H100: 1.0})
    assert (result.total.wall_hours, result.total.device_hours) == (3.0, {A100: 4.0, H100: 1.0})
