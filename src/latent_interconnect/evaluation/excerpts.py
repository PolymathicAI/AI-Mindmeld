"""The ``excerpts`` analysis kind: seeded candidate traces and the declared picks among them.

A paper excerpt (Fig 9a's generated traces) is never drawn by hand from the whole table. The rule is data
(:class:`Excerpts`), applied to the one input, a generation suite's record (its score table carries
``evaluation.scores.GENERATION_COLUMNS``):

1. the *eligible* rows are the labeled rows whose columns match ``where`` (the one row filter,
   ``evaluation.scores.select_rows``; e.g. a task, a condition, ``finished: true``) and, if ``pattern`` is set,
   whose generated text contains a match of that regular expression (``re.search``);
2. each eligible row is *correct* when its answer agrees with its label: the answer is Yes when the score
   (log p(Yes) − log p(No)) is above 0, so a score of exactly 0 answers No;
3. the correct and the incorrect eligible rows are each put in a fixed order (by every identity column of the table)
   and permuted by one generator seeded with ``seed`` (correct first); the first ``per_outcome`` of each are the
   *candidates*, so both outcomes are shown;
4. ``picked`` declares the choice among the candidates (task, item and conditions); the figure reads those, in
   the declared order. It is empty until a pick is made.

Fewer than ``per_outcome`` eligible rows of either outcome is an error, never a shorter set, and a pick outside the
candidates is an error. The result records how many rows were eligible per outcome, so the base rate is reported with
the selection, and :meth:`ExcerptsResult.companion_files` writes the candidates as :data:`CANDIDATES_FILE` beside
the result, the file read to make the picks.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from latent_interconnect.artifacts.records import Record, record_output
from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.config import Config
from latent_interconnect.data.items import ItemSet
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS
from latent_interconnect.evaluation.results import FINITE, AnalysisResult
from latent_interconnect.evaluation.scores import (
    ANSWER_COLUMNS,
    GENERATION_COLUMNS,
    SCORES_KIND,
    read_scores,
    select_rows,
)
from latent_interconnect.stats.bootstrap import ResamplingUnits

__all__ = ["CANDIDATES_FILE", "Excerpt", "Excerpts", "ExcerptsKind", "ExcerptsResult", "Pick"]

CANDIDATES_FILE = "candidates.md"
"""The file of an ``excerpts`` analysis artifact that lists the candidates."""
_ORDER_SKIP = frozenset({"score", "label", "batch", *GENERATION_COLUMNS, *ANSWER_COLUMNS})


class Pick(Config):
    """One picked excerpt, named by what identifies a row of one suite run.

    Attributes:
        task: the task.
        item: the item index in the suite's item set.
        conditions: role → the slot's condition.
    """

    task: str
    item: int
    conditions: dict[str, str] = Field(min_length=1)


class Excerpts(Config):
    """Parameters of the ``excerpts`` kind: the candidate rule and the picks.

    Attributes:
        where: column → accepted value (or list of values) of an eligible row.
        pattern: a regular expression the generated text must contain, or None for any text.
        seed: the seed of the permutation.
        per_outcome: the number of candidates drawn among the correct rows, and again among the incorrect ones.
        picked: the picks among the candidates, in display order; empty until a pick is made.
    """

    where: dict[str, str | int | float | bool | list[str | int | float | bool]] = Field(default_factory=dict)
    pattern: str | None = None
    seed: int
    per_outcome: int = Field(gt=0)
    picked: tuple[Pick, ...] = ()

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                re.compile(value)
            except re.error as e:
                raise ValueError(f"pattern {value!r} is not a regular expression: {e}") from e
        return value

    @model_validator(mode="after")
    def _distinct_picks(self) -> Excerpts:
        keys = [_key(p.task, p.item, p.conditions) for p in self.picked]
        if len(set(keys)) != len(keys):
            raise ValueError("a row is picked twice")
        return self


class Excerpt(Config):
    """One candidate row.

    Attributes:
        task: the task.
        item: the item index in the suite's item set.
        step: the checkpoint step.
        objects: role → the item's object.
        conditions: role → the slot's condition.
        text: the generated continuation, whole.
        n_tokens: its length in tokens.
        finished: whether it ended before the cap.
        max_new_tokens: the cap in force.
        score: the row's score.
        label: the task label.
        correct: whether the answer (Yes iff ``score`` > 0) agrees with the label.
    """

    model_config = FINITE

    task: str
    item: int
    step: int
    objects: dict[str, str]
    conditions: dict[str, str]
    text: str
    n_tokens: int
    finished: bool
    max_new_tokens: int
    score: float
    label: float
    correct: bool


class ExcerptsResult(AnalysisResult):
    """Result of ``excerpts``.

    Attributes:
        rows: rows of the generation table.
        eligible_correct: eligible rows answered correctly.
        eligible_incorrect: eligible rows answered incorrectly.
        candidates: the drawn candidates, the correct ones first, each outcome in draw order.
        picked: the declared picks, in their declared order (empty until a pick is made).
    """

    model_config = FINITE

    rows: int = Field(ge=0)
    eligible_correct: int = Field(ge=0)
    eligible_incorrect: int = Field(ge=0)
    candidates: tuple[Excerpt, ...] = Field(min_length=2)
    picked: tuple[Excerpt, ...] = ()

    def companion_files(self) -> dict[str, str]:
        """Return the candidates as a Markdown file for review, with each one's ``picked`` entry to copy."""
        lines = [
            "# Excerpt candidates",
            "",
            f"Eligible rows: {self.eligible_correct} correct, {self.eligible_incorrect} incorrect, of {self.rows}. "
            "To pick a candidate, copy its `picked` line into the analysis config's `params.picked`.",
        ]
        for n, e in enumerate(self.candidates, start=1):
            conditions = ", ".join(f"{r}: {c}" for r, c in sorted(e.conditions.items()))
            lines += [
                "",
                f"## {n}. {'correct' if e.correct else 'incorrect'}: {e.task}, item {e.item}",
                "",
                f"- picked: `{{task: {e.task!r}, item: {e.item}, conditions: {{{conditions}}}}}`",
                f"- objects: {e.objects}; step {e.step}; label {e.label:g}; score {e.score:+.3f}",
                f"- {e.n_tokens} tokens, {'finished' if e.finished else 'cut at the cap'} (cap {e.max_new_tokens})",
                "",
                "```text",
                e.text,
                "```",
            ]
        return {CANDIDATES_FILE: "\n".join(lines) + "\n"}


@ANALYSIS_KINDS.register("excerpts")
class ExcerptsKind:
    """Seeded candidates of both outcomes and the declared picks (params :class:`Excerpts`)."""

    Params: ClassVar[type[Config]] = Excerpts
    Result: ClassVar[type[AnalysisResult]] = ExcerptsResult

    def run(
        self,
        params: Excerpts,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> ExcerptsResult:
        """Apply the rule to the one input's generation rows and resolve the picks among the candidates.

        Raises:
            KeyError: if a ``where`` column is not in the table.
            ValueError: if there is not exactly one input, its table has no generation columns, fewer than
                ``per_outcome`` rows of an outcome are eligible, or a pick is not a candidate.
        """
        del items  # the table's own columns are the rule's
        if len(inputs) != 1:
            raise ValueError(f"excerpts reads one generation suite record, not {sorted(inputs)}")
        (record,) = inputs.values()
        table = read_scores(artifacts, record_output(record, SCORES_KIND))
        missing = sorted(set(GENERATION_COLUMNS) - set(table.columns))
        if missing:
            raise ValueError(f"record {record.record_id()[:12]} is not a generation suite: no columns {missing}")
        eligible = select_rows(table, params.where, "excerpts")
        eligible = eligible[eligible["label"].notna()]
        if params.pattern is not None:
            rx = re.compile(params.pattern)
            eligible = eligible[eligible["text"].map(lambda t: rx.search(str(t)) is not None)]
        correct = (eligible["score"] > 0) == (eligible["label"] > 0)
        order = sorted(c for c in table.columns if c not in _ORDER_SKIP)
        rng = np.random.default_rng(params.seed)
        candidates: list[Excerpt] = []
        for outcome, rows in (("correct", eligible[correct]), ("incorrect", eligible[~correct])):
            if len(rows) < params.per_outcome:
                raise ValueError(f"{len(rows)} {outcome} rows meet the rule; {params.per_outcome} are declared")
            ordered = rows.sort_values(order, kind="stable").reset_index(drop=True)
            drawn = rng.permutation(len(ordered))[: params.per_outcome]
            candidates += [_excerpt(ordered.iloc[int(i)], outcome == "correct") for i in drawn]
        by_key = {_key(e.task, e.item, e.conditions): e for e in candidates}
        absent = [p for p in params.picked if _key(p.task, p.item, p.conditions) not in by_key]
        if absent:
            raise ValueError(f"picks {[p.model_dump() for p in absent]} are not among the candidates")
        return ExcerptsResult(
            units=units,
            rows=len(table),
            eligible_correct=int(correct.sum()),
            eligible_incorrect=int((~correct).sum()),
            candidates=tuple(candidates),
            picked=tuple(by_key[_key(p.task, p.item, p.conditions)] for p in params.picked),
        )


def _key(task: str, item: int, conditions: Mapping[str, str]) -> tuple[str, int, tuple[tuple[str, str], ...]]:
    return task, item, tuple(sorted(conditions.items()))


def _excerpt(row: pd.Series, correct: bool) -> Excerpt:
    def per_role(prefix: str) -> dict[str, str]:
        return {str(c).removeprefix(prefix): str(row[c]) for c in row.index if str(c).startswith(prefix)}

    label = float(row["label"])
    if math.isnan(label):
        raise ValueError(f"item {row['item']} of task {row['task']!r} has no label")
    return Excerpt(
        task=str(row["task"]),
        item=int(row["item"]),
        step=int(row["step"]),
        objects=per_role("object."),
        conditions=per_role("condition."),
        text=str(row["text"]),
        n_tokens=int(row["n_tokens"]),
        finished=bool(row["finished"]),
        max_new_tokens=int(row["max_new_tokens"]),
        score=float(row["score"]),
        label=label,
        correct=correct,
    )
