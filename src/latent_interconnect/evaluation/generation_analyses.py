"""The ``generation_decline`` and ``name_guess`` analysis kinds: estimates and paired 95% intervals only.

**generation_decline** (Fig 9b, the reasoning tables and controls). Inputs ``direct`` (the direct ``yes_no`` readout)
and ``after`` (the post-reasoning readout, a ``yes_no`` suite reading a ``generation`` run), both score records of one
experiment (receiver, checkpoints, inputs, step, item set, domains) with the same unique labeled rows; only the
prompting (reasoning, prefill) differs. Per input the statistic is the mean over ``tasks`` of the per-task AUROC under
``role=condition``, thin tasks left out and listed (``evaluation.contrasts.split_thin``); the decline is
``after − direct``, paired over the shared bootstrap draws. The two readouts must be scored on one batch route (both
with ``pad_every_batch`` or both without, read from the input records' plans): a mixed pair is refused unless
the params declare it (``mixed_route``), and the result then says so; it never passes as a clean contrast. The cap
rate is the share of traces that hit the cap; the
finished-subset decline repeats the contrast on the (item, task) rows whose trace finished, over the same tasks, and is
None, with the tasks named, when a task's finished rows hold fewer than two units of a label class (never NaN)
or are thin. Capped traces stay in the headline decline (the after readout closes them and scores its own appended
answer). When the params say how a trace states its own final answer (``GenerationDecline.stated_answer``), the result
also counts the stated answers, the missing ones and those among capped traces (``StatedSupport``): a trace that states
none, or is cut before its answer, is counted as missing, never dropped and never scored as wrong.

**name_guess** (the name-recovery table and numbers). Input ``names``: a ``name_guess`` run. The statistic is the share
of items whose elicited name equals the object's name under ``evaluation.generating.normalize_name`` (case, spacing,
dashes, punctuation, Greek letters, numerals, leading articles, hedges do not count, so the automatic match does not
penalize case, spacing or dashes); the domain data lists no synonyms, so the gold name alone is
matched. The share of empty names is reported beside it. The paper's recovery table was graded
by LLM judges; this is the package's automatic measure.

Intervals: the grouped contrasts' bootstrap (``evaluation.contrasts.cell_draws`` and ``task_estimate``, over
``stats.bootstrap``) on the units of ``AnalysisSpec.units`` (item-set columns, ``evaluation.contrasts.unit_columns``):
``direct`` and ``after`` are two cells on one unit index, so the decline is paired; a one-task mean resamples its
classes separately and a several-task mean resamples the units jointly, a draw in which a task lacks a class being
undefined, dropped and counted. A task with fewer than two units of a class on the full sample is an error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import Field

from latent_interconnect.artifacts.records import Record, record_output
from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.config import Config
from latent_interconnect.data.items import ItemSet
from latent_interconnect.domains.base import DomainSpec, open_domain
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS
from latent_interconnect.evaluation.contrasts import CellDraws, cell_draws, is_thin, task_estimate, unit_columns
from latent_interconnect.evaluation.generating import StatedAnswer, normalize_name, stated_answer
from latent_interconnect.evaluation.generations import GENERATIONS_KIND, read_generations
from latent_interconnect.evaluation.results import AnalysisResult, Estimate, Unavailable
from latent_interconnect.evaluation.scores import SCORES_KIND, read_scores
from latent_interconnect.evaluation.suites import SuiteRunConfig, identity_differences
from latent_interconnect.stats.auroc import has_class_units
from latent_interconnect.stats.bootstrap import ResamplingUnits, summarize

__all__ = [
    "GenerationDecline",
    "GenerationDeclineKind",
    "GenerationDeclineResult",
    "NameGuessAnalysis",
    "NameGuessKind",
    "NameGuessResult",
    "StatedSupport",
]


class GenerationDecline(Config):
    """Parameters of ``generation_decline``.

    Attributes:
        tasks: the tasks averaged (equal weight).
        role: the slot role whose condition selects the rows.
        condition: that role's condition (``own``, or ``name`` for the name-input control).
        stated_answer: how the traces state their own final answer, when their frame asks for one: the result then
            counts the stated answers (:class:`StatedSupport`); None: no count, as every analysis before it.
        mixed_route: the inputs' batch plans differ in ``pad_every_batch`` (the padded route against the original
            unpadded one), declared: the result is then marked mixed-route. False: such a pair is refused, and a pair
            on one route must not declare it.
    """

    tasks: tuple[str, ...] = Field(min_length=1)
    role: str = "object"
    condition: str = "own"
    stated_answer: StatedAnswer | None = None
    mixed_route: bool = False


class StatedSupport(Config):
    """How the traces a ``generation_decline`` reads state their own final answer (:class:`StatedAnswer`).

    Every trace is counted once, in ``yes``, ``no`` or ``missing``; nothing is dropped, and the decline itself never
    reads the stated answer (the after readout scores its own appended answer on every row).

    Attributes:
        yes: traces that state yes.
        no: traces that state no.
        missing: traces that state no answer (none, or unparsable).
        missing_capped: of ``missing``, the traces that hit the cap (cut before or inside their answer line).
        capped_stated: traces that hit the cap yet state an answer before it.
    """

    yes: int = Field(ge=0)
    no: int = Field(ge=0)
    missing: int = Field(ge=0)
    missing_capped: int = Field(ge=0)
    capped_stated: int = Field(ge=0)


class GenerationDeclineResult(AnalysisResult):
    """Result of ``generation_decline``.

    Attributes:
        direct: mean per-task AUROC of the direct readout.
        after: mean per-task AUROC after reasoning.
        decline: ``after − direct``, paired.
        n_traces: the traces read.
        cap_rate: the share of traces that hit the cap.
        finished: the decline on the rows whose trace finished, or None when a task's finished rows lack class units.
        finished_undefined: the tasks that make ``finished`` undefined (empty when it is defined).
        unavailable: thin task → its support, left out of every mean.
        per_task: task → its ``direct``, ``after`` and paired ``decline`` (the supplement's per-task values),
            from the same bootstrap as the means; thin tasks left out.
        stated: the traces' stated answers (:class:`StatedSupport`), when the params declare how they are stated.
        mixed_route: the direct and after readouts were scored on different batch routes (``GenerationDecline``):
            not a clean contrast.
    """

    direct: Estimate
    after: Estimate
    decline: Estimate
    n_traces: int
    cap_rate: float
    finished: Estimate | None
    finished_undefined: tuple[str, ...] = ()
    unavailable: dict[str, Unavailable] = Field(default_factory=dict)
    per_task: dict[str, dict[str, Estimate]] = Field(default_factory=dict)
    stated: StatedSupport | None = None
    mixed_route: bool = False

    def unavailable_lines(self) -> list[str]:
        """Return one line per thin task."""
        return [f"task {t!r}: {u.text()}" for t, u in sorted(self.unavailable.items())]


class NameGuessAnalysis(Config):
    """Parameters of the ``name_guess`` analysis (none: the rule is fixed, module docstring)."""


class NameGuessResult(AnalysisResult):
    """Result of the ``name_guess`` analysis.

    Attributes:
        exact: the share of items whose elicited name matches the object's name (normalized).
        empty: the share of items with an empty elicited name.
        n: the items read.
    """

    exact: Estimate
    empty: float
    n: int


def _item_set(record: Record) -> str:
    suite = record.config.get("suite")
    if not isinstance(suite, dict) or "item_set" not in suite:
        raise ValueError(f"record {record.record_id()[:12]} is not a suite run")
    return str(suite["item_set"])


def _rows(table: pd.DataFrame, params: GenerationDecline) -> pd.DataFrame:
    return table[(table[f"condition.{params.role}"] == params.condition) & table["task"].isin(params.tasks)]


def _check_rows(direct: pd.DataFrame, after: pd.DataFrame) -> None:
    """Refuse readouts whose (item, task, condition) rows are not unique, not the same, or labeled differently."""
    keys = ["item", "task", *sorted(c for c in direct.columns if c.startswith("condition."))]
    for name, t in (("direct", direct), ("after", after)):
        if not set(keys) <= set(t.columns) or t.duplicated(keys).any():
            raise ValueError(f"generation_decline: {name}'s rows are not unique on {keys}")
    both = direct[[*keys, "label"]].merge(after[[*keys, "label"]], on=keys, how="outer", indicator=True)
    if (both["_merge"] != "both").any():
        raise ValueError(
            f"generation_decline: direct and after read different rows ({(both['_merge'] != 'both').sum()})"
        )
    x, y = both["label_x"], both["label_y"]
    if not ((x == y) | (x.isna() & y.isna())).all():
        raise ValueError("generation_decline: direct and after label a row differently")


def _thin(rows: pd.DataFrame, task: str) -> bool:
    """Return whether a task's labeled rows are thin (``evaluation.contrasts.is_thin``)."""
    labels = rows.loc[(rows["task"] == task) & rows["label"].notna(), "label"] > 0
    return is_thin(int(labels.sum()), int((~labels).sum()))


def _decline(cells: Mapping[str, CellDraws]) -> tuple[Estimate, Estimate, Estimate]:
    """Return the direct and after means and the paired ``after − direct`` (one bootstrap, one unit index)."""
    (d, d_draws), (a, a_draws) = cells["direct"].mean, cells["after"].mean
    return (
        Estimate.of(summarize(d_draws, d)),
        Estimate.of(summarize(a_draws, a)),
        Estimate.of(summarize(a_draws - d_draws, a - d)),
    )


def _per_task(cells: Mapping[str, CellDraws], task: str) -> dict[str, Estimate]:
    """Return one task's direct and after AUROC and the paired ``after − direct``."""
    (d, d_draws), (a, a_draws) = cells["direct"].tasks[task], cells["after"].tasks[task]
    return {
        "direct": Estimate.of(summarize(d_draws, d)),
        "after": Estimate.of(summarize(a_draws, a)),
        "decline": Estimate.of(summarize(a_draws - d_draws, a - d)),
    }


@ANALYSIS_KINDS.register("generation_decline")
class GenerationDeclineKind:
    """``generation_decline``: direct vs post-reasoning mean AUROC, cap rate, finished-subset decline."""

    Params: ClassVar[type[Config]] = GenerationDecline
    Result: ClassVar[type[AnalysisResult]] = GenerationDeclineResult

    def run(
        self,
        params: GenerationDecline,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> GenerationDeclineResult:
        """Return the result (module docstring).

        Raises:
            ValueError: if the inputs are not ``direct`` and ``after`` of one experiment (the same receiver,
                checkpoints, inputs, step, item set and domains: ``suites.identity_differences``; prompting may differ)
                with the same unique (item, task, condition) rows and labels, ``after`` carries no traces, or a task
                lacks a label class.
        """
        if set(inputs) != {"direct", "after"}:
            raise ValueError(f"generation_decline reads inputs direct and after, not {sorted(inputs)}")
        direct_run, after_run = (SuiteRunConfig.model_validate(inputs[k].config) for k in ("direct", "after"))
        if differ := identity_differences(direct_run, after_run):
            raise ValueError(f"generation_decline: direct and after differ in {differ}")
        padded = {
            k: bool(r.plan is not None and r.plan.pad_every_batch)
            for k, r in (("direct", direct_run), ("after", after_run))
        }
        mixed = padded["direct"] != padded["after"]
        if mixed != params.mixed_route:
            raise ValueError(
                f"generation_decline: the inputs' pad_every_batch is {padded} (records "
                f"{inputs['direct'].record_id()[:8]}, {inputs['after'].record_id()[:8]}), so the pair is "
                f"{'mixed-route' if mixed else 'on one route'}; params.mixed_route is {params.mixed_route}"
            )
        columns = unit_columns(items(_item_set(inputs["direct"])), units.factors)
        direct = _rows(read_scores(artifacts, record_output(inputs["direct"], SCORES_KIND)), params)
        after = _rows(read_scores(artifacts, record_output(inputs["after"], SCORES_KIND)), params)
        _check_rows(direct, after)
        if "finished" not in after.columns:
            raise ValueError("generation_decline: the after input carries no traces (not a post-reasoning readout)")
        full = cell_draws({"direct": direct, "after": after}, list(params.tasks), "auroc", columns)
        d, a, decline = _decline(full)
        tasks = list(full["direct"].tasks)
        keys = ["item", "task"]
        done = after.loc[after["finished"].astype(bool), keys]
        pair = {name: t.merge(done, on=keys) for name, t in (("direct", direct), ("after", after))}
        undefined = tuple(
            t
            for t in tasks
            if not all(
                has_class_units({f: np.asarray(c)[q["item"].to_numpy()] for f, c in columns.items()}, q["label"] > 0)
                for p in pair.values()
                for q in [p[(p["task"] == t) & p["label"].notna()]]
            )
        )
        undefined += tuple(t for t in tasks if t not in undefined and _thin(pair["direct"], t))
        finished = None if undefined else _decline(cell_draws(pair, tasks, "auroc", columns))[2]
        return GenerationDeclineResult(
            units=units,
            direct=d,
            after=a,
            decline=decline,
            n_traces=len(after),
            cap_rate=float((~after["finished"].astype(bool)).mean()),
            finished=finished,
            finished_undefined=undefined,
            unavailable=full["direct"].unavailable,
            per_task={t: _per_task(full, t) for t in tasks},
            stated=None
            if params.stated_answer is None
            else _stated(after, params.stated_answer, after_run.receiver.thinking_close),
            mixed_route=mixed,
        )


def _stated(after: pd.DataFrame, rule: StatedAnswer, close: str | None) -> StatedSupport:
    """Count the after rows' traces by the answer they state (``text``, carried from the trace) and the cap."""
    stated = after["text"].map(lambda text: stated_answer(str(text), rule, close))
    capped = ~after["finished"].astype(bool)
    return StatedSupport(
        yes=int((stated == rule.yes).sum()),
        no=int((stated == rule.no).sum()),
        missing=int((stated == "").sum()),
        missing_capped=int(((stated == "") & capped).sum()),
        capped_stated=int(((stated != "") & capped).sum()),
    )


@ANALYSIS_KINDS.register("name_guess")
class NameGuessKind:
    """``name_guess``: the share of exact (normalized) elicited names, with its interval, and the empty share."""

    Params: ClassVar[type[Config]] = NameGuessAnalysis
    Result: ClassVar[type[AnalysisResult]] = NameGuessResult

    def run(
        self,
        params: NameGuessAnalysis,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> NameGuessResult:
        """Return the result (module docstring).

        Raises:
            ValueError: if the input is not one ``name_guess`` run named ``names``, or it holds no ``own`` row.
        """
        del params
        if set(inputs) != {"names"}:
            raise ValueError(f"the name_guess analysis reads input names, not {sorted(inputs)}")
        record = inputs["names"]
        suite = record.config.get("suite")
        if not isinstance(suite, dict) or suite.get("kind") != "name_guess":
            raise ValueError(f"record {record.record_id()[:12]} is not a name_guess run")
        table = read_generations(artifacts, record_output(record, GENERATIONS_KIND))
        role = next(c.split(".", 1)[1] for c in table.columns if c.startswith("object."))
        rows = table[table[f"condition.{role}"] == "own"]
        if rows.empty:
            raise ValueError("the name_guess analysis: no own generations")
        item_set = items(_item_set(record))
        domain = open_domain(DomainSpec.model_validate(suite["domains"][0]), artifacts)
        names = domain.objects()["name"]
        hit = np.array(
            [
                normalize_name(a) == normalize_name(str(names[o]))
                for a, o in zip(rows["answer"], rows[f"object.{role}"], strict=True)
            ],
            dtype=np.float64,
        )
        scored = pd.DataFrame({"item": rows["item"].to_numpy(np.int64), "score": hit, "label": hit})
        empty = float((rows["answer"] == "").mean())
        return NameGuessResult(
            units=units,
            exact=task_estimate(scored, "mean", unit_columns(item_set, units.factors)),
            empty=empty,
            n=len(scored),
        )
