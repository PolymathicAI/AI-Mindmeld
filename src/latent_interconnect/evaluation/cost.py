"""The ``cost`` analysis kind: compute cost from the provenance of records.

A cost table's rows (a connector's training, its preparation, shared work) are declared as groups of input records
(:class:`Cost`). Each record is one execution; its cost comes from its provenance alone
(:class:`~latent_interconnect.provenance.Provenance`), summed over every job it ran in
(:meth:`~latent_interconnect.provenance.Provenance.segments`): per segment the wall time ``ended − started``,
and per accelerator model that segment's wall time times its number of devices of that model (device-hours), so a
resumed run that continued on other hardware charges each job to the devices it actually ran on. A segment
without accelerators is a CPU task: its wall time counts as CPU task-hours. A row sums its records; the total sums
the rows, so a record belongs to one row only.

The table has no per-step timing columns: the original table's "Updates (h)" and "Mean s/step" are dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import ClassVar

from pydantic import Field, model_validator

from latent_interconnect.artifacts.records import Record
from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.config import Config
from latent_interconnect.data.items import ItemSet
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS
from latent_interconnect.evaluation.results import FINITE, AnalysisResult
from latent_interconnect.stats.bootstrap import ResamplingUnits

__all__ = ["Cost", "CostEntry", "CostKind", "CostResult", "CostRow", "cost_of"]

_HOUR = 3600.0


class CostRow(Config):
    """One row of a cost table.

    Attributes:
        name: the row label (a connector, "protein-shared", ...); a label, never parsed.
        inputs: the analysis inputs (keys of ``AnalysisSpec.inputs``) whose records the row sums.
    """

    name: str = Field(min_length=1)
    inputs: tuple[str, ...] = Field(min_length=1)


class Cost(Config):
    """Parameters of the ``cost`` kind.

    Attributes:
        rows: the table's rows, in display order; names distinct, and each input in exactly one row.
    """

    rows: tuple[CostRow, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Cost:
        names = [r.name for r in self.rows]
        if len(set(names)) != len(names):
            raise ValueError(f"cost rows repeat names: {names}")
        keys = [k for r in self.rows for k in r.inputs]
        repeated = sorted({k for k in keys if keys.count(k) > 1})
        if repeated:
            raise ValueError(f"inputs {repeated} are in more than one row (the total would count them twice)")
        return self


class CostEntry(Config):
    """The cost of one or more executions.

    Attributes:
        records: executions summed.
        wall_hours: the sum of their wall times, over every segment (job) of each.
        device_hours: accelerator model (as provenance names it) → wall hours times devices of that model.
        cpu_hours: wall hours of the executions without accelerators (CPU task-hours).
    """

    model_config = FINITE

    records: int = Field(ge=0)
    wall_hours: float = Field(ge=0)
    device_hours: dict[str, float] = Field(default_factory=dict)
    cpu_hours: float = Field(ge=0)


class CostResult(AnalysisResult):
    """Result of ``cost``.

    Attributes:
        rows: row name → its cost, in the declared order.
        total: the sum over the rows.
    """

    model_config = FINITE

    rows: dict[str, CostEntry]
    total: CostEntry


@ANALYSIS_KINDS.register("cost")
class CostKind:
    """Compute cost per declared row from the input records' provenance (params :class:`Cost`)."""

    Params: ClassVar[type[Config]] = Cost
    Result: ClassVar[type[AnalysisResult]] = CostResult

    def run(
        self,
        params: Cost,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> CostResult:
        """Return each row's cost and the total.

        Raises:
            KeyError: if a row names an undeclared input.
            ValueError: if an input is in no row, or a record ends before it starts.
        """
        del artifacts, items  # provenance is in the records themselves
        named = {k for r in params.rows for k in r.inputs}
        missing = sorted(named - set(inputs))
        if missing:
            raise KeyError(f"cost rows name inputs the analysis does not declare: {missing}")
        unread = sorted(set(inputs) - named)
        if unread:
            raise ValueError(f"inputs {unread} are in no cost row")
        rows = {r.name: cost_of([inputs[k] for k in r.inputs]) for r in params.rows}
        return CostResult(units=units, rows=rows, total=cost_of([inputs[k] for r in params.rows for k in r.inputs]))


def cost_of(records: list[Record]) -> CostEntry:
    """Return the summed cost of executions, from their provenance: every segment of each, on its own devices.

    Raises:
        ValueError: if a record has no end time or one of its segments ends before it starts.
    """
    wall, cpu = 0.0, 0.0
    devices: dict[str, float] = {}
    for rec in records:
        p = rec.provenance
        if p.ended is None:
            raise ValueError(f"record {rec.record_id()[:12]} has no end time")
        for seg in p.segments():  # every job of the execution, each on its own devices
            hours = (datetime.fromisoformat(seg.ended) - datetime.fromisoformat(seg.started)).total_seconds() / _HOUR
            if hours < 0:
                raise ValueError(
                    f"record {rec.record_id()[:12]} has a segment that ends ({seg.ended}) before it starts "
                    f"({seg.started})"
                )
            wall += hours
            if not seg.devices:
                cpu += hours
            for model in seg.devices:
                devices[model] = devices.get(model, 0.0) + hours
    return CostEntry(records=len(records), wall_hours=wall, device_hours=dict(sorted(devices.items())), cpu_hours=cpu)
