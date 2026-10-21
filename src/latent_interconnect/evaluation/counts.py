"""The ``counts`` and ``removal_counts`` analysis kinds: the paper's counts, read from records.

**``counts``** (:class:`Counts`). Every count a figure, table or sentence shows (Figs 1–4, the task tables, the
methods numbers) is a declared query over one input record, never a hand-entered value:

- a *count* (:class:`CountQuery`) reads one table of the record (:data:`TABLES`), keeps the rows whose columns match
  ``where`` (the one row filter, ``evaluation.scores.select_rows``) and, on ``scores``, whose item's fields match
  ``item_filter`` (``grouped_contrasts``' ``item_filters`` of the scored item set, e.g. pool-clean), and counts the
  rows or the distinct values of one column, broken down by the ``by`` columns;
- a *value* (:class:`ValueQuery`) reads one entry of the record's config (a connector's width or write layers, a
  run's steps: what the run configs declare, e.g. ``connector_instances`` and ``supp_training_changes``).

The tables a count reads:

=====================  ==============================================================================================
``scores``             the record's score table (``evaluation.scores``): rows per task, positives (``label: 1.0``)
``items``              the record's ``item_set`` output: ``item``, ``object.<role>`` and every item field
``suite_items``        the item set a suite record scored (its suite's ``item_set``), same columns
``training_objects``   the record's ``training_set`` output: one row per training object (``object``)
``training_fields``    the same: one row per kept training-text field (``object``, ``field``)
=====================  ==============================================================================================

**``removal_counts``** (no parameters; one input: the ``training_set`` stage's record). The zero-shot screen's
removal counts for the paper's honesty statement: per zero-shot task and field, the whole fields removed and the
list elements removed, their totals per task, the list elements outside an allow-list, and the number of training
objects (the denominator of a share). They are the counts the screen recorded in the training set's metadata
(``build.zero_shot.build_training_set``), not a re-screen.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import ClassVar, Literal

import pandas as pd
from pydantic import Field, JsonValue, model_validator

from latent_interconnect.artifacts.records import Record, record_output
from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.config import Config
from latent_interconnect.data.items import ITEM_SET_KIND, ItemSet, load_item_set
from latent_interconnect.data.training_text import TRAINING_SET_KIND, load_training_text
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS
from latent_interconnect.evaluation.contrasts import suite_of
from latent_interconnect.evaluation.results import FINITE, AnalysisResult
from latent_interconnect.evaluation.scores import SCORES_KIND, read_scores, select_rows
from latent_interconnect.stats.bootstrap import ResamplingUnits

__all__ = [
    "ROWS",
    "TABLES",
    "CountQuery",
    "CountRow",
    "Counts",
    "CountsKind",
    "CountsResult",
    "RemovalCounts",
    "RemovalCountsKind",
    "RemovalCountsResult",
    "TaskRemovals",
    "ValueQuery",
    "item_table",
]

TABLES = ("scores", "items", "suite_items", "training_objects", "training_fields")
"""The tables a count can read (module docstring)."""
ROWS = "rows"
"""``CountQuery.count`` that counts rows (any other value names the column whose distinct values are counted)."""

Table = Literal["scores", "items", "suite_items", "training_objects", "training_fields"]
Scalar = str | int | float | bool


class CountQuery(Config):
    """One declared count over one table of one input record.

    Attributes:
        input: the analysis input (a key of ``AnalysisSpec.inputs``) whose record holds the table.
        table: which of the record's tables (:data:`TABLES`).
        where: column → accepted value (or list of values); only matching rows are counted.
        item_filter: on ``scores`` only: field of the scored item set → accepted value (or list of values); only rows
            whose item matches are counted (the same filter as ``GroupedContrasts.item_filters``).
        count: :data:`ROWS`, or a column whose distinct (non-missing) values are counted (e.g. ``object.object``).
        by: columns the count is broken down by (one result row per combination present); none gives one total.
    """

    input: str = Field(min_length=1)
    table: Table
    where: dict[str, Scalar | list[Scalar]] = Field(default_factory=dict)
    item_filter: dict[str, Scalar | list[Scalar]] = Field(default_factory=dict)
    count: str = ROWS
    by: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _filter_on_scores(self) -> CountQuery:
        if self.item_filter and self.table != "scores":
            raise ValueError(f"item_filter applies to the scores table, not {self.table!r} (use where)")
        return self


class ValueQuery(Config):
    """One entry of an input record's config.

    Attributes:
        input: the analysis input whose record's config is read.
        path: keys (and list indices) from the config's root to the entry, e.g. ``["run", "steps"]``.
    """

    input: str = Field(min_length=1)
    path: tuple[str | int, ...] = Field(min_length=1)


class Counts(Config):
    """Parameters of the ``counts`` kind.

    Attributes:
        counts: count name → query; the name is a label readers cite, never parsed.
        values: value name → query.
    """

    counts: dict[str, CountQuery] = Field(default_factory=dict)
    values: dict[str, ValueQuery] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _some(self) -> Counts:
        if not (self.counts or self.values):
            raise ValueError("a counts analysis declares at least one count or value")
        clash = sorted(set(self.counts) & set(self.values))
        if clash:
            raise ValueError(f"names {clash} are both a count and a value")
        return self


class CountRow(Config):
    """One row of a count.

    Attributes:
        by: ``by`` column → its value in this row, as text (empty for a total).
        n: the count.
    """

    model_config = FINITE

    by: dict[str, str] = Field(default_factory=dict)
    n: int = Field(ge=0)


class CountsResult(AnalysisResult):
    """Result of ``counts``.

    Attributes:
        counts: count name → its rows, sorted by their ``by`` values (one row with empty ``by`` for a total; a total
            over no matching rows is 0, while a breakdown lists only the combinations present).
        values: value name → the config entry (JSON).
    """

    model_config = FINITE

    counts: dict[str, tuple[CountRow, ...]] = Field(default_factory=dict)
    values: dict[str, JsonValue] = Field(default_factory=dict)


@ANALYSIS_KINDS.register("counts")
class CountsKind:
    """Declared counts and config values over the input records (params :class:`Counts`)."""

    Params: ClassVar[type[Config]] = Counts
    Result: ClassVar[type[AnalysisResult]] = CountsResult

    def run(
        self,
        params: Counts,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> CountsResult:
        """Return every declared count and value.

        Raises:
            KeyError: if a query names an undeclared input, a ``where``, ``count`` or ``by`` column the table lacks,
                an ``item_filter`` field the item set lacks, or a config path that does not resolve.
            ValueError: if an input is read by no query, or a record lacks the table's artifact.
        """
        read = {q.input for q in params.counts.values()} | {q.input for q in params.values.values()}
        missing = sorted(read - set(inputs))
        if missing:
            raise KeyError(f"queries name inputs the analysis does not declare: {missing}")
        unread = sorted(set(inputs) - read)
        if unread:
            raise ValueError(f"inputs {unread} are read by no query")
        tables: dict[tuple[str, str], pd.DataFrame] = {}
        counts = {}
        for name, q in sorted(params.counts.items()):
            if (q.input, q.table) not in tables:
                tables[(q.input, q.table)] = _table(q.table, inputs[q.input], artifacts, items)
            table = tables[(q.input, q.table)]
            if q.item_filter:
                fields = item_table(items(str(suite_of(inputs[q.input])["item_set"])))
                keep = select_rows(fields, q.item_filter, f"count {name!r} item_filter")["item"]
                table = table[table["item"].isin(keep)]
            counts[name] = _count(table, q, f"count {name!r}")
        values = {name: _value(inputs[q.input].config, q.path, f"value {name!r}") for name, q in params.values.items()}
        return CountsResult(units=units, counts=counts, values=values)


def item_table(item_set: ItemSet) -> pd.DataFrame:
    """Return an item set as a table: ``item`` (the index), ``object.<role>`` per role, and every item field."""
    columns: dict[str, object] = {"item": range(len(item_set))}
    columns |= {f"object.{role}": list(ids) for role, ids in item_set.objects.items()}
    columns |= {f: list(v) for f, v in item_set.fields.items()}
    return pd.DataFrame(columns)


def _table(table: Table, record: Record, artifacts: ArtifactStore, items: Callable[[str], ItemSet]) -> pd.DataFrame:
    if table == "scores":
        return read_scores(artifacts, record_output(record, SCORES_KIND))
    if table == "items":
        return item_table(load_item_set(artifacts, record_output(record, ITEM_SET_KIND)))
    if table == "suite_items":
        return item_table(items(str(suite_of(record)["item_set"])))
    text = load_training_text(artifacts, record_output(record, TRAINING_SET_KIND))
    if table == "training_objects":
        return pd.DataFrame({"object": list(text.object_ids)})
    rows = [(o, f) for o in text.object_ids for f in sorted(text.fields.get(o, {}))]
    return pd.DataFrame(rows, columns=["object", "field"])


def _count(table: pd.DataFrame, q: CountQuery, what: str) -> tuple[CountRow, ...]:
    for column in (*q.by, *(() if q.count == ROWS else (q.count,))):
        if column not in table.columns:
            raise KeyError(f"{what}: column {column!r} not in {q.table} ({sorted(table.columns)})")
    rows = select_rows(table, q.where, what)

    def n(part: pd.DataFrame) -> int:
        return len(part) if q.count == ROWS else int(part[q.count].nunique(dropna=True))

    if not q.by:
        return (CountRow(n=n(rows)),)
    out = []
    for key, g in rows.groupby(list(q.by), sort=True):
        values = key if isinstance(key, tuple) else (key,)
        out.append(CountRow(by={c: str(v) for c, v in zip(q.by, values, strict=True)}, n=n(g)))
    return tuple(out)


def _value(config: object, path: tuple[str | int, ...], what: str) -> JsonValue:
    node = config
    for step in path:
        if isinstance(node, dict) and isinstance(step, str) and step in node:
            node = node[step]
            continue
        if isinstance(node, list) and isinstance(step, int) and -len(node) <= step < len(node):
            node = node[step]
            continue
        raise KeyError(f"{what}: config path {list(path)} does not resolve at {step!r}")
    value: JsonValue = node  # type: ignore[assignment]  # a record's config is JSON by construction
    return value


class RemovalCounts(Config):
    """Parameters of the ``removal_counts`` kind: none (its one input is the ``training_set`` stage's record)."""


class TaskRemovals(Config):
    """One zero-shot task's removals.

    Attributes:
        removed: field → whole fields removed (free text, and lists left empty).
        removed_elements: field → list elements removed.
        total_removed: the sum of ``removed``.
        total_elements: the sum of ``removed_elements``.
    """

    model_config = FINITE

    removed: dict[str, int] = Field(default_factory=dict)
    removed_elements: dict[str, int] = Field(default_factory=dict)
    total_removed: int = Field(ge=0)
    total_elements: int = Field(ge=0)


class RemovalCountsResult(AnalysisResult):
    """Result of ``removal_counts``.

    Attributes:
        training_set: the digest of the ``training_set`` artifact counted.
        objects: training objects (the denominator of a share).
        tasks: zero-shot task → its removals (every screened task, with zeros where nothing was removed).
        not_allowed: list field → elements outside its allow-list (not training text).
    """

    model_config = FINITE

    training_set: str
    objects: int = Field(ge=0)
    tasks: dict[str, TaskRemovals]
    not_allowed: dict[str, int] = Field(default_factory=dict)


@ANALYSIS_KINDS.register("removal_counts")
class RemovalCountsKind:
    """The screened training set's removal counts (params :class:`RemovalCounts`)."""

    Params: ClassVar[type[Config]] = RemovalCounts
    Result: ClassVar[type[AnalysisResult]] = RemovalCountsResult

    def run(
        self,
        params: RemovalCounts,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> RemovalCountsResult:
        """Return the removal counts the screen recorded.

        Raises:
            ValueError: if there is not exactly one input, it has no ``training_set`` output, or the training set's
                metadata lacks the screen's counts.
        """
        del params, items  # no parameters; the training set is the record's output
        if len(inputs) != 1:
            raise ValueError(f"removal_counts reads one training_set record, not {sorted(inputs)}")
        (record,) = inputs.values()
        ref = record_output(record, TRAINING_SET_KIND)
        meta = artifacts.manifest(ref).meta
        if "removed_elements" not in meta or "spec" not in meta:
            raise ValueError(f"training set {ref.short} has no screen counts (built before the zero-shot screen?)")
        objects = len(load_training_text(artifacts, ref).object_ids)
        removed, elements = meta["removed"], meta["removed_elements"]
        tasks = {}
        for task in sorted({*removed, *elements}):
            r, e = dict(removed.get(task, {})), dict(elements.get(task, {}))
            tasks[task] = TaskRemovals(
                removed=r, removed_elements=e, total_removed=sum(r.values()), total_elements=sum(e.values())
            )
        return RemovalCountsResult(
            units=units,
            training_set=ref.sha256,
            objects=objects,
            tasks=tasks,
            not_allowed=dict(meta.get("not_allowed", {})),
        )
