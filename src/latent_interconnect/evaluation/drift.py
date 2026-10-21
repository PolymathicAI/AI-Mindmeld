"""The ``execution_drift`` analysis kind: how much scores move across execution configs.

The inputs are suite records of one suite run that differ only in their execution config: a *reference* (the input
named by ``ExecutionDrift.reference``) and one or more *alternates*. An alternate's config may differ from the
reference's only in :data:`EXECUTION_KEYS` (the receiver, which fixes the backend and precision, and the batch plan);
anything else (the suite, items, checkpoints, inputs, step, device class) is refused, so the drift is the config
difference alone. Both sides must score exactly the same rows, with the same labels. Rows are matched on the tables'
identity columns, never on what was read (the score, the answer log-probabilities, the batch), so a table written
before the answer columns existed compares with one written after.

Per alternate:

- per row, the absolute score difference, summarized (max, median, mean), and the *answer flips*: rows whose sign of
  the score differs (for Yes/No suites the score is log p(Yes) − log p(No), so a flip changes the answer; likelihood
  scores never flip);
- per task and condition combination, the statistic (``metric``: the AUROC of the scores against the labels,
  ``stats.auroc``, or the mean score) on each side and its absolute difference, with the largest over the groups.

Every value is defined or refused ("A NaN is a bug"): a group without both label classes under
``auroc`` is an error. The device class is not a drift dimension (it is refused above). The accelerator model may be:
a config's device class admits any accelerator model of its kind and topology, so a reference and an
alternate may have run on different models. The result does not hide it: each side's ``execution`` (its
``device_name`` included) is in it (``reference_execution``, ``AlternateDrift.execution``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from latent_interconnect.artifacts.records import Record, record_output
from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.config import Config
from latent_interconnect.data.items import ItemSet
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS
from latent_interconnect.evaluation.contrasts import suite_of
from latent_interconnect.evaluation.results import FINITE, AnalysisResult
from latent_interconnect.evaluation.scores import (
    ANSWER_COLUMNS,
    GENERATION_COLUMNS,
    LIKELIHOOD_COLUMNS,
    SCORES_KIND,
    read_scores,
)
from latent_interconnect.stats.auroc import auroc
from latent_interconnect.stats.bootstrap import ResamplingUnits

__all__ = [
    "EXECUTION_KEYS",
    "AlternateDrift",
    "DriftGroup",
    "Execution",
    "ExecutionDrift",
    "ExecutionDriftKind",
    "ExecutionDriftResult",
    "RowDrift",
]

EXECUTION_KEYS = frozenset({"receiver", "plan"})
"""The suite-run config keys (``SuiteRunConfig``) an alternate may change: what executes, not what is scored."""
_NOT_IDENTITY = frozenset({"score", "label", "batch", *GENERATION_COLUMNS, *LIKELIHOOD_COLUMNS, *ANSWER_COLUMNS})


class ExecutionDrift(Config):
    """Parameters of the ``execution_drift`` kind.

    Attributes:
        reference: the input (a key of ``AnalysisSpec.inputs``) every other input is compared with.
        metric: the per-group statistic: ``auroc`` (unlabelled rows left out) or ``mean`` of the scores.
    """

    reference: str = Field(min_length=1)
    metric: Literal["auroc", "mean"] = "auroc"


class Execution(Config):
    """How one input executed (from its record), the provenance of its execution config.

    Attributes:
        execution: backend, precision, batch size and device type (``Provenance.execution``).
    """

    model_config = FINITE

    execution: dict[str, str]


class RowDrift(Config):
    """Per-row score drift of one alternate.

    Attributes:
        rows: rows compared.
        max_abs: the largest absolute score difference.
        median_abs: the median.
        mean_abs: the mean.
        answer_flips: rows whose score changes sign.
    """

    model_config = FINITE

    rows: int = Field(gt=0)
    max_abs: float = Field(ge=0)
    median_abs: float = Field(ge=0)
    mean_abs: float = Field(ge=0)
    answer_flips: int = Field(ge=0)


class DriftGroup(Config):
    """One task and condition combination.

    Attributes:
        task: the task.
        conditions: ``condition.<role>`` column → value.
        n: rows the statistic used.
        reference: the statistic on the reference.
        alternate: the statistic on the alternate.
        abs_diff: their absolute difference.
    """

    model_config = FINITE

    task: str
    conditions: dict[str, str]
    n: int = Field(gt=0)
    reference: float
    alternate: float
    abs_diff: float = Field(ge=0)


class AlternateDrift(Config):
    """One alternate against the reference.

    Attributes:
        differs_in: the config keys that differ (a subset of :data:`EXECUTION_KEYS`).
        execution: how the alternate executed.
        scores: the per-row drift.
        groups: the per-group drift, sorted by task and conditions.
        max_group_abs_diff: the largest ``abs_diff`` over the groups.
    """

    model_config = FINITE

    differs_in: tuple[str, ...]
    execution: Execution
    scores: RowDrift
    groups: tuple[DriftGroup, ...] = Field(min_length=1)
    max_group_abs_diff: float = Field(ge=0)


class ExecutionDriftResult(AnalysisResult):
    """Result of ``execution_drift``.

    Attributes:
        metric: the per-group statistic.
        reference: the reference input.
        reference_execution: how the reference executed.
        alternates: input → its drift against the reference.
    """

    model_config = FINITE

    metric: Literal["auroc", "mean"]
    reference: str
    reference_execution: Execution
    alternates: dict[str, AlternateDrift] = Field(min_length=1)


@ANALYSIS_KINDS.register("execution_drift")
class ExecutionDriftKind:
    """Score drift of alternate executions of one suite run against a reference (params :class:`ExecutionDrift`)."""

    Params: ClassVar[type[Config]] = ExecutionDrift
    Result: ClassVar[type[AnalysisResult]] = ExecutionDriftResult

    def run(
        self,
        params: ExecutionDrift,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> ExecutionDriftResult:
        """Compare every alternate with the reference.

        Raises:
            KeyError: if the reference is not an input.
            ValueError: if there is no alternate, an input is not a suite run, an alternate differs from the reference
                beyond :data:`EXECUTION_KEYS` (or not at all), the rows or labels differ, or a group's statistic is
                undefined.
        """
        del items  # rows align on the score tables' own columns
        if params.reference not in inputs:
            raise KeyError(f"reference {params.reference!r} is not among the inputs {sorted(inputs)}")
        if len(inputs) < 2:  # noqa: PLR2004 - a reference and at least one alternate
            raise ValueError("execution_drift needs at least one alternate besides the reference")
        ref = inputs[params.reference]
        suite_of(ref)
        ref_table = read_scores(artifacts, record_output(ref, SCORES_KIND))
        alternates = {}
        for key, rec in sorted(inputs.items()):
            if key == params.reference:
                continue
            suite_of(rec)
            differs = sorted(k for k in {*ref.config, *rec.config} if ref.config.get(k) != rec.config.get(k))
            beyond = sorted(set(differs) - EXECUTION_KEYS)
            if beyond or not differs:
                raise ValueError(
                    f"input {key!r} is not another execution of the reference's suite run: it differs in "
                    f"{differs or 'nothing'} (only {sorted(EXECUTION_KEYS)} may differ)"
                )
            table = read_scores(artifacts, record_output(rec, SCORES_KIND))
            alternates[key] = _alternate(
                ref_table, table, differs=tuple(differs), execution=_execution(rec), metric=params.metric, what=key
            )
        return ExecutionDriftResult(
            units=units,
            metric=params.metric,
            reference=params.reference,
            reference_execution=_execution(ref),
            alternates=alternates,
        )


def _execution(record: Record) -> Execution:
    return Execution(execution=dict(record.provenance.execution))


def _alternate(
    ref: pd.DataFrame,
    alt: pd.DataFrame,
    *,
    differs: tuple[str, ...],
    execution: Execution,
    metric: Literal["auroc", "mean"],
    what: str,
) -> AlternateDrift:
    align = sorted(c for c in ref.columns if c not in _NOT_IDENTITY)
    if sorted(c for c in alt.columns if c not in _NOT_IDENTITY) != align:
        raise ValueError(f"{what}: the score tables have different columns")
    for side, t in (("reference", ref), (what, alt)):
        if t.duplicated(align).any():
            raise ValueError(f"{side}: rows are not unique on {align}")
    joined = ref[[*align, "score", "label"]].merge(
        alt[[*align, "score", "label"]], on=align, how="outer", suffixes=("_ref", "_alt"), indicator=True
    )
    unmatched = int((joined["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(f"{what}: {unmatched} rows are scored on one side only")
    if not joined["label_ref"].fillna(-1).equals(joined["label_alt"].fillna(-1)):
        raise ValueError(f"{what}: the labels differ from the reference's")
    diff = (joined["score_ref"] - joined["score_alt"]).abs().to_numpy(np.float64)
    flips = int((np.sign(joined["score_ref"]) != np.sign(joined["score_alt"])).sum())
    rows = RowDrift(
        rows=len(diff),
        max_abs=float(diff.max()),
        median_abs=float(np.median(diff)),
        mean_abs=float(diff.mean()),
        answer_flips=flips,
    )
    conditions = sorted(c for c in align if c.startswith("condition."))
    groups = []
    for key, group in joined.groupby(["task", *conditions], sort=True):
        task, *values = key
        g = group[group["label_ref"].notna()] if metric == "auroc" else group
        a, b = _statistic(g, "score_ref", metric), _statistic(g, "score_alt", metric)
        if a is None or b is None:
            raise ValueError(f"{what}: task {task!r} {dict(zip(conditions, values, strict=True))} has no {metric}")
        groups.append(
            DriftGroup(
                task=str(task),
                conditions={c: str(v) for c, v in zip(conditions, values, strict=True)},
                n=len(g),
                reference=a,
                alternate=b,
                abs_diff=abs(a - b),
            )
        )
    return AlternateDrift(
        differs_in=differs,
        execution=execution,
        scores=rows,
        groups=tuple(groups),
        max_group_abs_diff=max(x.abs_diff for x in groups),
    )


def _statistic(rows: pd.DataFrame, column: str, metric: str) -> float | None:
    """Return the group's statistic, or None where it is undefined (no rows, or one label class for AUROC)."""
    if rows.empty:
        return None
    scores = rows[column].to_numpy(np.float64)
    if metric == "mean":
        return float(scores.mean())
    labels = rows["label_ref"].to_numpy(np.float64) > 0
    if labels.all() or not labels.any():
        return None
    return auroc(scores, labels)
