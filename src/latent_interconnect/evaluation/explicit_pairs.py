"""The ``explicit_pairs`` analysis kind: accuracy over fixed discordant pairs, with paired controls and state changes.

The protein-mutation clinical and functional readouts (E11; the within-parent discordant evaluation). The
item set is a ``build.variant_items`` fixed-pair cohort: every item names its pair (field ``pair``), its parent
(``wt.object``) and its annotation source (``source.object``), each pair is one yes and one no variant of one parent.
Unlike ``stratified_contrast``, which forms every positive × negative product within a stratum, only the listed pairs
are compared, each once and with equal weight: a pair is correct when its yes member scores above its no member
(log p(yes) − log p(no)), half correct on a tie. The pooled own AUROC of the same score records is
``grouped_contrasts``' (``metric: auroc``, units ``cluster.object``): the two answer different questions, and neither
replaces the other.

**Inputs** are ``yes_no`` score records of one experiment on the cohort (the same receiver, embedding inputs, item set
and domains: ``suites.identity_differences``; checkpoints and step may differ, as two states of one run do), all on
one batch route (``plan.pad_every_batch`` the same: a mixed route is refused, never reported). For every input and
every condition of ``params.conditions`` the result holds the pair accuracy; for every input, each control's paired
difference to ``own``; and for ``params.change`` (two inputs: a run's own start and endpoint) each condition's paired
difference between them. The binary declarations score ``own``, ``sibling``, ``wt`` and ``derange``; the
text references ``name`` and ``name_substitution`` are scored by their own connector-free suite (no checkpoint), whose
analysis reports them without ``own`` and so without a difference to it.

**A logistic baseline's record** (``evaluation.logistic_baseline``: a supervised read of the fixed encoder,
no connector) is read as one input alone, under ``own`` only (its one condition): its config's suite, or with
``params.suite`` the one of its scored suites of that id (a baseline scores every cohort of a task with one fit, each
suite's rows under its digest), names the item set and the rows read.

**Availability.** A pair member the score record lists as ``unavailable`` under a condition (its prompt would write an
object the run's inputs cannot hold, ``SuiteRunConfig.unavailable``) leaves that pair out of that condition only; any
other unscored member is an error. Each estimate states its pairs; a difference is computed on the pairs available on
both sides, so it stays paired.

**Uncertainty.** The one bootstrap (``stats.bootstrap``) over the pairs' parent clusters (the item set's field
``cluster.object``, which a pair's members share with their parent): ``units`` must name exactly that factor. Within
a group (all pairs, or one source) one draw matrix serves every input, condition and contrast, so every difference is
paired; a draw weights each pair by the times its cluster is drawn, so several pairs of one parent, and several parents
of one cluster, move together.

**Groups.** The pooled group and one per annotation source (``by_source``): a pair's source is its members' common
source, or ``mixed`` when they differ (the rule of the variant text baselines).

**Support** (the rule of the variant text baselines: no interval under 10 items or 8 clusters; scoped
here to the pair readouts). The readout's scored unit is the pair (one comparison of
its two rows, never two independent items), so the rule's item floor counts pairs (``min_pairs``); its cluster floor
counts the parents' id30 clusters (``min_clusters``); parents are counted and reported, not floored. An estimate under
either floor keeps its point and counts and withholds its interval, saying why (``PairValue.unavailable``); only
an undefined point (no pair held) has none. The pairs the cohort dropped (its item set's metadata ``dropped``,
``build.variant_items``: never refilled) are always reported. No power claim is made.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.records import Record, record_output
from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import Config, added_field
from latent_interconnect.data.items import ITEM_SET_KIND, OBJECT_ROLE, ItemSet, cluster_field
from latent_interconnect.data.suites import Condition
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS
from latent_interconnect.evaluation.logistic_baseline import LogisticBaselineSpec
from latent_interconnect.evaluation.results import AnalysisResult, Estimate
from latent_interconnect.evaluation.scores import SCORES_KIND, read_scores
from latent_interconnect.evaluation.suites import SuiteRunConfig, identity_differences
from latent_interconnect.stats.bootstrap import Bootstrap, ResamplingUnits, Statistic, unit_index

__all__ = [
    "ExplicitPairs",
    "ExplicitPairsKind",
    "ExplicitPairsResult",
    "PairCell",
    "PairGroup",
    "PairValue",
    "fixed_pairs",
    "pair_correct",
]

PAIR_FIELD = "pair"
"""The item-set field naming each item's fixed pair (``build.variant_items``)."""
PARENT_FIELD = f"wt.{OBJECT_ROLE}"
"""The item-set field naming each item's parent (``build.variant_items``)."""
SOURCE_FIELD = f"source.{OBJECT_ROLE}"
"""The item-set field naming each item's annotation source (``build.variant_items``)."""
MIXED = "mixed"
"""The source of a pair whose members' sources differ."""
UNITS = ResamplingUnits(factors=(cluster_field(OBJECT_ROLE),))
"""The one resampling factor of the kind: the parents' clusters."""
BASELINE_KIND = "logistic_baseline"
"""The record kind of a logistic baseline (``run.logistic_baseline``), read by its own route (module docstring)."""
TEXT_CONDITIONS = frozenset({Condition.NAME, Condition.NAME_SUBSTITUTION})
"""The text references, the one condition set reported without ``own`` (their connector-free suite)."""
_TIE = 0.5


class ExplicitPairs(Config):
    """Parameters of ``explicit_pairs``.

    Attributes:
        task: the scored task (one Yes/No variant label).
        conditions: the conditions reported, distinct; with ``own`` among them every other one is also reported as
            its paired difference to ``own`` (the controls' reference); without it, every condition must be a text
            reference (:data:`TEXT_CONDITIONS`, their connector-free suite), and none is.
        change: two input names ``(from, to)``: each condition's paired difference ``to − from`` (a run's own start
            and endpoint); None for none.
        min_pairs: the support rule's item floor, counted in pairs, the readout's scored unit: with fewer the point
            is reported without an interval (the rule of the variant text baselines:
            no interval under 10 items or 8 clusters).
        min_clusters: the rule's cluster floor.
        suite: for a logistic baseline's record, the id of the scored suite read (added field); None: its ``suite``.
            Refused for a suite record.
    """

    task: str
    conditions: tuple[Condition, ...] = Field(min_length=1)
    change: tuple[str, str] | None = None
    min_pairs: int = Field(default=10, ge=1)
    min_clusters: int = Field(default=8, ge=1)
    suite: str | None = added_field(default=None)

    @model_validator(mode="after")
    def _own(self) -> ExplicitPairs:
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError(f"explicit_pairs: conditions {list(self.conditions)} must be distinct")
        if Condition.OWN not in self.conditions and not set(self.conditions) <= TEXT_CONDITIONS:
            raise ValueError(f"explicit_pairs: conditions {list(self.conditions)} without own must be text references")
        return self


class PairValue(Config):
    """One estimate: its point, its interval when the support rule allows one, and the pairs it is computed on.

    Attributes:
        point: the value on the full sample (accuracy, or a paired difference); None only when it is undefined (no
            pair held).
        interval: the point with its bootstrap interval, or None under the support rule (or with no point).
        n_pairs: the pairs it is computed on (for a difference, those held on both sides).
        n_clusters: their distinct parent id30 clusters.
        unavailable: why there is no interval (``under the support floor: <n> pairs, <k> clusters``, or ``no pair
            held``), or "".
    """

    point: float | None
    interval: Estimate | None
    n_pairs: int
    n_clusters: int
    unavailable: str = ""


class PairCell(Config):
    """One input's values under one condition.

    Attributes:
        accuracy: the pair accuracy.
        minus_own: its paired difference to ``own`` (None for ``own``, and when ``own`` is not reported).
    """

    accuracy: PairValue
    minus_own: PairValue | None = None


class PairGroup(Config):
    """The estimates of one group of pairs.

    Attributes:
        n_pairs: the group's fixed pairs.
        n_parents: their distinct parents.
        n_clusters: their distinct parent clusters (the bootstrap's units).
        cells: input name → condition → :class:`PairCell`.
        change: condition → the paired ``to − from`` difference of ``params.change`` (empty without one).
    """

    n_pairs: int
    n_parents: int
    n_clusters: int
    cells: dict[str, dict[str, PairCell]]
    change: dict[str, PairValue] = Field(default_factory=dict)


class ExplicitPairsResult(AnalysisResult):
    """Result of ``explicit_pairs``.

    Attributes:
        pooled: every fixed pair.
        by_source: annotation source (``mutagen``, ``variant``, ``mixed``) → its pairs.
        dropped: the cohort's dropped pairs, each with its reason (the item set's metadata; never refilled).
    """

    pooled: PairGroup
    by_source: dict[str, PairGroup]
    dropped: tuple[dict[str, str], ...] = ()


def pair_correct(
    table: pd.DataFrame, pairs: pd.DataFrame, condition: str, task: str, unavailable: frozenset[int] = frozenset()
) -> np.ndarray:
    """Return each fixed pair's correctness under one condition: 1, 0.5 on a tie, 0, or NaN when unavailable.

    Args:
        table: a score table (``item``, ``task``, ``condition.object``, ``label``, ``score``).
        pairs: one row per pair, ``yes`` and ``no`` item indices (:func:`fixed_pairs`).
        condition: the condition.
        task: the task.
        unavailable: the items the score record lists as unavailable under this task and condition.

    Raises:
        ValueError: if a pair member has two rows, or none and is not listed unavailable, or a row's label
            contradicts its pair side.
    """
    rows = table[(table["task"] == task) & (table[f"condition.{OBJECT_ROLE}"] == condition)].set_index("item")
    if rows.index.duplicated().any():
        twice = sorted(set(rows.index[rows.index.duplicated()]))
        raise ValueError(f"explicit_pairs: items {twice[:3]} have two score rows under {condition}; not paired")
    missing = sorted((set(pairs["yes"]) | set(pairs["no"])) - set(rows.index) - unavailable)
    if missing:
        raise ValueError(f"explicit_pairs: {len(missing)} pair members unscored under {condition}, e.g. {missing[:3]}")
    held = (pairs["yes"].isin(rows.index) & pairs["no"].isin(rows.index)).to_numpy()
    yes = rows["score"].reindex(pairs["yes"]).to_numpy(dtype=float)
    no = rows["score"].reindex(pairs["no"]).to_numpy(dtype=float)
    labels = (rows["label"].reindex(pairs["yes"]).to_numpy(), rows["label"].reindex(pairs["no"]).to_numpy())
    if (labels[0][held] != 1).any() or (labels[1][held] != 0).any():
        raise ValueError(f"explicit_pairs: a pair's members are not labelled (yes, no) under {condition}")
    return np.where(held, np.where(yes > no, 1.0, np.where(yes == no, _TIE, 0.0)), np.nan)


def fixed_pairs(items: ItemSet, labels: pd.Series) -> pd.DataFrame:
    """Return the item set's fixed pairs: ``pair``, ``yes`` and ``no`` items, ``parent``, ``cluster`` and ``source``.

    A pair's ``source`` is its members' common annotation source, or ``mixed`` when they differ.

    Args:
        items: a fixed-pair cohort (``build.variant_items``).
        labels: item index → label (1 yes, 0 no).

    Raises:
        ValueError: if the set lacks the ``pair``, parent, cluster or source field, or a pair is not one yes and one no
            item of one named parent in one cluster.
    """
    fields = {"pair": PAIR_FIELD, "parent": PARENT_FIELD, "cluster": cluster_field(OBJECT_ROLE), "source": SOURCE_FIELD}
    if missing := [f for f in fields.values() if f not in items.fields]:
        raise ValueError(f"explicit_pairs: item set {items.name!r} has no fields {missing}")
    frame = pd.DataFrame({k: items.fields[f] for k, f in fields.items()}).assign(
        item=range(len(items)), label=lambda d: d["item"].map(labels)
    )
    out = []
    for pair, g in frame.groupby("pair", sort=False):
        if sorted(g["label"].tolist()) != [0, 1] or g["parent"].nunique() != 1 or g["cluster"].nunique() != 1:
            raise ValueError(f"explicit_pairs: pair {pair!r} is not one yes and one no item of one parent and cluster")
        if not str(g["parent"].iloc[0]):
            raise ValueError(f"explicit_pairs: pair {pair!r} names no parent")
        out.append(
            {
                "pair": pair,
                "yes": int(g.loc[g["label"] == 1, "item"].iloc[0]),
                "no": int(g.loc[g["label"] == 0, "item"].iloc[0]),
                "parent": str(g["parent"].iloc[0]),
                "cluster": str(g["cluster"].iloc[0]),
                "source": str(g["source"].iloc[0]) if g["source"].nunique() == 1 else MIXED,
            }
        )
    return pd.DataFrame(out, columns=["pair", "yes", "no", "parent", "cluster", "source"])


@ANALYSIS_KINDS.register("explicit_pairs")
class ExplicitPairsKind:
    """``explicit_pairs``: fixed-pair accuracy per input and condition, controls minus own, and the state change."""

    Params: ClassVar[type[Config]] = ExplicitPairs
    Result: ClassVar[type[AnalysisResult]] = ExplicitPairsResult

    def run(
        self,
        params: ExplicitPairs,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> ExplicitPairsResult:
        """Return the result (module docstring).

        Raises:
            ValueError: if ``units`` is not the parents' clusters, the inputs are not one experiment on one route,
                ``change`` names an input not given, a pair is malformed, or a pair member is unscored.
        """
        if units != UNITS:
            raise ValueError(f"explicit_pairs: units {units.factors} must be {UNITS.factors} (the parents' clusters)")
        if any(r.kind == BASELINE_KIND for r in inputs.values()):
            return self._baseline(params, inputs, artifacts, items)
        if params.suite is not None:
            raise ValueError(f"explicit_pairs: suite {params.suite!r} is for a logistic baseline's record")
        runs = {name: SuiteRunConfig.model_validate(r.config) for name, r in inputs.items()}
        first = next(iter(runs.values()))
        for name, run in runs.items():
            if differ := [d for d in identity_differences(first, run) if d not in ("checkpoints", "step")]:
                raise ValueError(f"explicit_pairs: input {name!r} differs in {differ}")
        padded = {name: bool(r.plan is not None and r.plan.pad_every_batch) for name, r in runs.items()}
        if len(set(padded.values())) > 1:
            raise ValueError(f"explicit_pairs: the inputs are on two batch routes (pad_every_batch {padded})")
        if params.change is not None and not set(params.change) <= set(inputs):
            raise ValueError(f"explicit_pairs: change {params.change} names inputs not given ({sorted(inputs)})")
        tables = {n: read_scores(artifacts, record_output(r, SCORES_KIND)) for n, r in inputs.items()}
        lost = {n: _unavailable(artifacts, record_output(r, SCORES_KIND), params.task) for n, r in inputs.items()}
        return _result(params, tables, lost, digest=str(first.suite.item_set), artifacts=artifacts, items=items)

    def _baseline(
        self,
        params: ExplicitPairs,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> ExplicitPairsResult:
        """Return the result on one logistic baseline's record: its scored suite's rows (module docstring)."""
        if len(inputs) != 1 or params.change is not None or tuple(params.conditions) != (Condition.OWN,):
            raise ValueError("explicit_pairs: a logistic baseline's record is read alone, under own, with no change")
        ((name, rec),) = inputs.items()
        spec = LogisticBaselineSpec.model_validate(rec.config)
        wanted = spec.suite.id if params.suite is None else params.suite
        found = [s for s in spec.suites if s.id == wanted]
        if len(found) != 1:
            raise ValueError(f"explicit_pairs: the baseline scores {[s.id for s in spec.suites]}, not {wanted!r}")
        table = read_scores(artifacts, record_output(rec, SCORES_KIND))
        table = table[table["suite"] == found[0].digest()].reset_index(drop=True)
        return _result(
            params, {name: table}, {name: {}}, digest=str(found[0].item_set), artifacts=artifacts, items=items
        )


def _result(
    params: ExplicitPairs,
    tables: Mapping[str, pd.DataFrame],
    lost: Mapping[str, Mapping[str, frozenset[int]]],
    *,
    digest: str,
    artifacts: ArtifactStore,
    items: Callable[[str], ItemSet],
) -> ExplicitPairsResult:
    """Return the result of the inputs' score tables on the fixed pairs of item set ``digest``."""
    dropped = tuple(artifacts.manifest(ArtifactRef(digest, ITEM_SET_KIND)).meta.get("dropped", ()))
    some = next(iter(tables.values()))
    labels = some[some["task"] == params.task].drop_duplicates("item").set_index("item")["label"]
    pairs = fixed_pairs(items(digest), labels)
    correct = {  # validated before the support rule: a scoring fault is an error, never "unavailable"
        (n, str(c)): pair_correct(t, pairs, str(c), params.task, lost[n].get(str(c), frozenset()))
        for n, t in tables.items()
        for c in params.conditions
    }
    group = partial(_group, params, pairs=pairs, correct=correct, inputs=tuple(tables))
    return ExplicitPairsResult(
        units=UNITS,
        pooled=group(np.ones(len(pairs), dtype=bool)),
        by_source={str(k): group((pairs["source"] == k).to_numpy()) for k in sorted(pairs["source"].unique())},
        dropped=dropped,
    )


def _unavailable(artifacts: ArtifactStore, scores: ArtifactRef, task: str) -> dict[str, frozenset[int]]:
    """Return condition → the items a score record lists as unavailable for ``task`` (``evaluation.runners``)."""
    out: dict[str, set[int]] = {}
    for row in artifacts.manifest(scores).meta.get("unavailable", ()):
        if row["task"] == task:
            out.setdefault(str(row["condition"]), set()).add(int(row["item"]))
    return {c: frozenset(v) for c, v in out.items()}


def _group(
    params: ExplicitPairs,
    member: np.ndarray,
    *,
    pairs: pd.DataFrame,
    correct: Mapping[tuple[str, str], np.ndarray],
    inputs: tuple[str, ...],
) -> PairGroup:
    """Return the estimates of the pairs ``member`` selects, on their own clusters' bootstrap (module docstring)."""
    sub = pairs[member]
    index = unit_index(sub["cluster"])
    cluster = sub["cluster"].map(index).to_numpy()
    values = {k: v[member] for k, v in correct.items()}
    counts = {"n_pairs": len(sub), "n_parents": sub["parent"].nunique(), "n_clusters": len(index)}
    boot = Bootstrap({"cluster": max(len(index), 1)})

    def accuracy(key: tuple[str, str], keep: np.ndarray) -> Statistic:
        def stat(draw: Mapping[str, np.ndarray]) -> float:
            w = np.bincount(draw["cluster"], minlength=len(index))[cluster][keep]
            return float((w * values[key][keep]).sum() / w.sum()) if w.sum() else float("nan")

        return stat

    def value(a: tuple[str, str], b: tuple[str, str] | None = None) -> PairValue:
        keep = ~np.isnan(values[a]) if b is None else ~np.isnan(values[a]) & ~np.isnan(values[b])
        n, k = int(keep.sum()), int(sub["cluster"][keep].nunique())
        if not n:
            return PairValue(point=None, interval=None, n_pairs=0, n_clusters=0, unavailable="no pair held")
        point = float(values[a][keep].mean() - (0.0 if b is None else values[b][keep].mean()))
        if n < params.min_pairs or k < params.min_clusters:  # the point stands; only the interval is withheld
            why = f"under the support floor: {n} pairs, {k} clusters"
            return PairValue(point=point, interval=None, n_pairs=n, n_clusters=k, unavailable=why)
        if b is None:
            interval = boot.interval(accuracy(a, keep), point)
        else:
            interval = boot.contrast(accuracy(a, keep), accuracy(b, keep), point)
        return PairValue(point=point, interval=Estimate.of(interval), n_pairs=n, n_clusters=k)

    own = str(Condition.OWN)
    cells = {
        n: {
            str(c): PairCell(
                accuracy=value((n, str(c))),
                minus_own=None
                if c == Condition.OWN or Condition.OWN not in params.conditions
                else value((n, str(c)), (n, own)),
            )
            for c in params.conditions
        }
        for n in inputs
    }
    change = {}
    if params.change is not None:
        a, b = params.change
        change = {str(c): value((b, str(c)), (a, str(c))) for c in params.conditions}
    return PairGroup(cells=cells, change=change, **counts)
