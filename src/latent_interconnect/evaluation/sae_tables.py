"""The feature tables of the ``sae_exploration`` analysis: support per question, strength and separation per feature.

Exploratory and descriptive: every table is computed on the same objects its lists are selected on, so a
listed feature is an **unvalidated discovery ranking**, never a tested claim. No interval, threshold or test is
computed here.

Inputs are one dictionary's sparse activations ``z`` of ``n`` objects (``receivers.dictionary.StateCodes``), the norm
of each active feature's decoder row, each object's group (the resampling unit, e.g. its Butina cluster) and the Yes/No
tasks' labels and scope masks. An activation that is not stored is zero, and zeros count everywhere (means, prevalence,
ranks).

- **Support** (:func:`task_cohorts`, before any feature is ranked). Per task: the raw label counts over all objects
  (yes, no, unknown), the objects outside the task's scope (``Domain.scope_mask``; excluded, never counted as
  negatives), and the *eligible* cohorts: in scope with a known label, yes where the label is ``> 0``; their objects and
  distinct groups. The tier is ``primary`` when both eligible cohorts hold at least ``Reporting.primary_groups``
  distinct groups, ``low_support`` when both hold at least ``Reporting.low_support_groups``, else ``thin``.
- **Strength.** A feature's strength on an object is ``|z_j| · ‖decoder row j‖₂``, a contribution-magnitude proxy (not
  additive explained energy); a cohort's is its mean over the cohort's objects.
- **Separation.** A feature's effect on a task is the signed rank-biserial ``2 · AUROC(z_j, yes) − 1`` over the task's
  eligible objects (``stats.auroc_columns``: ties, the shared zeros included, count one half), beside the prevalence
  (share of objects with ``z_j > 0``) in each cohort. No classifier is fitted and no feature combination searched.
- **The activity floor.** A feature has effect rows for a task when it is active in at least
  ``Reporting.active_groups`` distinct groups among the task's eligible objects, the yes and no cohorts pooled.
- **Compact lists** (``primary`` tasks only): the ``Reporting.top`` strongest features of each cohort, and the
  ``Reporting.top`` features of largest positive and of largest negative effect among those meeting the floor. Ties
  break by feature index. Every task with both cohorts keeps its effect rows (:class:`TaskEffects`) whatever its tier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
from pydantic import Field

from latent_interconnect.config import Config
from latent_interconnect.domains.base import Domain, TaskKind
from latent_interconnect.evaluation.results import FINITE
from latent_interconnect.stats.auroc import auroc_columns

__all__ = [
    "Activations",
    "Cohort",
    "FeatureSummary",
    "FeatureTables",
    "RankedFeature",
    "Reporting",
    "Separator",
    "TaskEffects",
    "TaskLists",
    "TaskSupport",
    "feature_tables",
    "listed_features",
    "task_cohorts",
]

Tier = Literal["primary", "low_support", "thin"]


class Reporting(Config):
    """The reporting recipe of the feature tables (module docstring); every value changes what is listed.

    Attributes:
        top: the length of each compact list.
        primary_groups: the least distinct eligible groups in each cohort for a task's compact lists.
        low_support_groups: the least in each cohort for the ``low_support`` tier (below it: ``thin``).
        active_groups: the activity floor: the least distinct groups, among a task's eligible objects, in which a
            feature is active for it to have effect rows.
    """

    top: int = Field(ge=1)
    primary_groups: int = Field(ge=1)
    low_support_groups: int = Field(ge=1)
    active_groups: int = Field(ge=1)


class TaskSupport(Config):
    """One task's support, counted before any feature is ranked.

    Attributes:
        scope: the task's declared relation to training (``trained``, ``held_out``, ``implied``).
        raw_yes: objects labeled yes, whatever the task's scope mask.
        raw_no: objects labeled no.
        unknown: objects without a label.
        out_of_scope: objects outside the task's scope mask (excluded from the cohorts).
        yes: eligible yes objects.
        no: eligible no objects.
        yes_groups: their distinct groups.
        no_groups: the eligible no objects' distinct groups.
        tier: ``primary``, ``low_support`` or ``thin`` (module docstring).
    """

    scope: str
    raw_yes: int = Field(ge=0)
    raw_no: int = Field(ge=0)
    unknown: int = Field(ge=0)
    out_of_scope: int = Field(ge=0)
    yes: int = Field(ge=0)
    no: int = Field(ge=0)
    yes_groups: int = Field(ge=0)
    no_groups: int = Field(ge=0)
    tier: Tier


class Cohort(NamedTuple):
    """One task's eligible objects.

    Attributes:
        eligible: ``(n,)`` bool, in scope with a known label.
        yes: ``(n,)`` bool, eligible and labeled yes.
    """

    eligible: np.ndarray
    yes: np.ndarray


class FeatureSummary(Config):
    """Every feature active on at least one object, over the whole population (columns of equal length).

    Attributes:
        feature: the feature indices, ascending.
        active_objects: objects on which each is active.
        active_groups: distinct groups in which each is active.
        mean_activation: each one's mean activation over all objects.
        mean_strength: each one's mean strength over all objects.
        decoder_norm: the norm of each one's decoder row.
    """

    model_config = FINITE

    feature: tuple[int, ...]
    active_objects: tuple[int, ...]
    active_groups: tuple[int, ...]
    mean_activation: tuple[float, ...]
    mean_strength: tuple[float, ...]
    decoder_norm: tuple[float, ...]


class Activations(Config):
    """The sparse activations (columns of equal length, sorted by object then feature); any other activation is zero.

    Attributes:
        row: the object's position in the result's ``objects``.
        feature: the feature index.
        value: the activation.
    """

    model_config = FINITE

    row: tuple[int, ...]
    feature: tuple[int, ...]
    value: tuple[float, ...]


class RankedFeature(Config):
    """One feature of a strength list, with its numbers on the list's objects (the population, or one cohort).

    Attributes:
        feature: the feature index.
        mean_strength: its mean strength over those objects (the ranking value).
        mean_activation: its mean raw activation over them.
        prevalence: the share of them on which it is active.
        active_groups: their distinct groups in which it is active.
    """

    model_config = FINITE

    feature: int
    mean_strength: float
    mean_activation: float
    prevalence: float
    active_groups: int


class Separator(Config):
    """One feature of a separator list (an unvalidated discovery ranking).

    Attributes:
        feature: the feature index.
        effect: the signed rank-biserial effect (positive: higher in the yes cohort).
        prevalence_yes: the share of eligible yes objects on which it is active.
        prevalence_no: the share of eligible no objects.
        prevalence_difference: ``prevalence_yes − prevalence_no``.
        active_groups: the distinct groups among the task's eligible objects in which it is active.
    """

    model_config = FINITE

    feature: int
    effect: float
    prevalence_yes: float
    prevalence_no: float
    prevalence_difference: float
    active_groups: int


class TaskEffects(Config):
    """One task's rows for every feature meeting the activity floor (columns of equal length, by feature index).

    Attributes:
        feature: the feature indices.
        effect: the signed rank-biserial effects.
        prevalence_yes: the prevalence in the eligible yes cohort.
        prevalence_no: the prevalence in the eligible no cohort.
        strength_yes: the mean strength over the eligible yes cohort.
        strength_no: the mean strength over the eligible no cohort.
        active_groups: the distinct eligible groups in which each is active.
    """

    model_config = FINITE

    feature: tuple[int, ...]
    effect: tuple[float, ...]
    prevalence_yes: tuple[float, ...]
    prevalence_no: tuple[float, ...]
    strength_yes: tuple[float, ...]
    strength_no: tuple[float, ...]
    active_groups: tuple[int, ...]


class TaskLists(Config):
    """A ``primary`` task's compact lists.

    Attributes:
        strongest_yes: the strongest features of the eligible yes cohort.
        strongest_no: the strongest of the eligible no cohort.
        higher_in_yes: the features of largest positive effect, among those meeting the activity floor.
        higher_in_no: the features of largest negative effect, most negative first.
    """

    strongest_yes: tuple[RankedFeature, ...]
    strongest_no: tuple[RankedFeature, ...]
    higher_in_yes: tuple[Separator, ...]
    higher_in_no: tuple[Separator, ...]


class FeatureTables(Config):
    """One dictionary's feature tables on one set of objects.

    Attributes:
        reporting: the recipe they were made with.
        support: task → its support (every task asked).
        features: the population-wide summary of every active feature.
        activations: the sparse activations.
        strongest: the population's ``top`` strongest features (one list per dictionary, not per task).
        effects: task → its effect rows; a task with an empty eligible cohort has none (its support says so).
        lists: ``primary`` task → its compact lists.
    """

    reporting: Reporting
    support: dict[str, TaskSupport]
    features: FeatureSummary
    activations: Activations
    strongest: tuple[RankedFeature, ...]
    effects: dict[str, TaskEffects]
    lists: dict[str, TaskLists]


def _distinct(groups: np.ndarray, mask: np.ndarray) -> int:
    return int(np.unique(groups[mask]).size)


def task_cohorts(
    domain: Domain, objects: Sequence[str], groups: np.ndarray, tasks: Sequence[str], reporting: Reporting
) -> tuple[dict[str, TaskSupport], dict[str, Cohort]]:
    """Return each task's support and eligible cohorts on ``objects`` (module docstring).

    Args:
        domain: the domain (its catalog, labels and scope masks).
        objects: the objects, in the order of every per-object array.
        groups: ``(n,)`` each object's group index.
        tasks: the Yes/No task ids.
        reporting: the recipe (its group floors set the tiers).

    Raises:
        KeyError: if a task is not in the domain's catalog.
        ValueError: if a task is not a Yes/No task, or the objects repeat.
    """
    index = pd.Index(list(objects))
    if not index.is_unique:
        raise ValueError("the objects repeat: each object is one row of the tables")
    catalog, labels = domain.tasks(), domain.labels()
    support, cohorts = {}, {}
    for task_id in tasks:
        task = catalog[task_id]
        if task.kind != TaskKind.YES_NO:
            raise ValueError(f"task {task_id!r} is a {task.kind.value} task: the tables take Yes/No tasks only")
        column = labels[task.label].reindex(index)
        known = column.notna().to_numpy(bool)
        positive = known & (column.fillna(0).to_numpy(np.float64) > 0)
        in_scope = domain.scope_mask(task, index).reindex(index).fillna(False).to_numpy(bool)
        eligible = known & in_scope
        yes, no = eligible & positive, eligible & ~positive
        yes_groups, no_groups = _distinct(groups, yes), _distinct(groups, no)
        least = min(yes_groups, no_groups)
        tier: Tier = (
            "primary"
            if least >= reporting.primary_groups
            else "low_support"
            if least >= reporting.low_support_groups
            else "thin"
        )
        support[task_id] = TaskSupport(
            scope=task.scope.value,
            raw_yes=int(positive.sum()),
            raw_no=int((known & ~positive).sum()),
            unknown=int((~known).sum()),
            out_of_scope=int((~in_scope).sum()),
            yes=int(yes.sum()),
            no=int(no.sum()),
            yes_groups=yes_groups,
            no_groups=no_groups,
            tier=tier,
        )
        cohorts[task_id] = Cohort(eligible, yes)
    return support, cohorts


def _active_groups(active: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Return, per column of the ``(m, k)`` bool ``active``, the number of distinct ``groups`` with an active row."""
    if active.shape[0] == 0:
        return np.zeros(active.shape[1], dtype=np.int64)
    order = np.argsort(groups, kind="stable")
    ordered = groups[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    counts: np.ndarray = np.logical_or.reduceat(active[order], starts, axis=0).sum(axis=0)
    return counts


def _first(values: np.ndarray, features: np.ndarray, top: int) -> np.ndarray:
    """Return the positions of the ``top`` largest positive ``values``, ties by feature index."""
    order = np.lexsort((features, -values))
    return order[values[order] > 0][:top]


def _ranked(
    z: np.ndarray, norms: np.ndarray, features: np.ndarray, groups: np.ndarray, top: int
) -> list[RankedFeature]:
    """Return the ``top`` strongest features of the objects whose activations are the rows of ``z``."""
    if z.shape[0] == 0:
        return []
    strength = np.abs(z).mean(axis=0) * norms
    at = _first(strength, features, top)
    in_groups = _active_groups(z[:, at] > 0, groups)
    return [
        RankedFeature(
            feature=int(features[j]),
            mean_strength=float(strength[j]),
            mean_activation=float(z[:, j].mean()),
            prevalence=float((z[:, j] > 0).mean()),
            active_groups=int(g),
        )
        for j, g in zip(at, in_groups, strict=True)
    ]


def _task(
    z: np.ndarray, yes: np.ndarray, groups: np.ndarray, *, norms: np.ndarray, features: np.ndarray, reporting: Reporting
) -> tuple[TaskEffects, tuple[list[Separator], list[Separator]]]:
    """Return one task's effect rows and its two separator lists, from its eligible objects' activations ``z``."""
    in_groups = _active_groups(z > 0, groups)
    keep = np.flatnonzero(in_groups >= reporting.active_groups)
    kept, ids = z[:, keep], features[keep]
    effect = 2.0 * auroc_columns(kept, yes) - 1.0
    prevalence_yes, prevalence_no = (kept[yes] > 0).mean(axis=0), (kept[~yes] > 0).mean(axis=0)
    rows = TaskEffects(
        feature=tuple(int(f) for f in ids),
        effect=tuple(float(v) for v in effect),
        prevalence_yes=tuple(float(v) for v in prevalence_yes),
        prevalence_no=tuple(float(v) for v in prevalence_no),
        strength_yes=tuple(float(v) for v in np.abs(kept[yes]).mean(axis=0) * norms[keep]),
        strength_no=tuple(float(v) for v in np.abs(kept[~yes]).mean(axis=0) * norms[keep]),
        active_groups=tuple(int(g) for g in in_groups[keep]),
    )

    def separators(signed: np.ndarray) -> list[Separator]:
        return [
            Separator(
                feature=int(ids[j]),
                effect=float(effect[j]),
                prevalence_yes=float(prevalence_yes[j]),
                prevalence_no=float(prevalence_no[j]),
                prevalence_difference=float(prevalence_yes[j] - prevalence_no[j]),
                active_groups=int(in_groups[keep][j]),
            )
            for j in _first(signed, ids, reporting.top)
        ]

    return rows, (separators(effect), separators(-effect))


def feature_tables(
    rows: np.ndarray,
    features: np.ndarray,
    values: np.ndarray,
    *,
    decoder_norms: Mapping[int, float],
    groups: np.ndarray,
    support: Mapping[str, TaskSupport],
    cohorts: Mapping[str, Cohort],
    reporting: Reporting,
) -> FeatureTables:
    """Return the feature tables of one dictionary's sparse activations (module docstring).

    Args:
        rows: ``(m,)`` the object (its row, ``0 .. n - 1``) of each stored activation.
        features: ``(m,)`` its feature index.
        values: ``(m,)`` the activation.
        decoder_norms: active feature → the norm of its decoder row.
        groups: ``(n,)`` each object's group index.
        support: task → its support (:func:`task_cohorts`).
        cohorts: task → its eligible cohorts.
        reporting: the recipe.

    Raises:
        KeyError: if an active feature has no decoder norm.
    """
    n = int(groups.shape[0])
    ids = np.unique(features)
    z = np.zeros((n, ids.size), dtype=np.float32)
    z[rows, np.searchsorted(ids, features)] = values
    norms = np.array([decoder_norms[int(f)] for f in ids], dtype=np.float64)
    summary = FeatureSummary(
        feature=tuple(int(f) for f in ids),
        active_objects=tuple(int(c) for c in (z > 0).sum(axis=0)),
        active_groups=tuple(int(c) for c in _active_groups(z > 0, groups)),
        mean_activation=tuple(float(v) for v in z.mean(axis=0, dtype=np.float64)),
        mean_strength=tuple(float(v) for v in np.abs(z).mean(axis=0, dtype=np.float64) * norms),
        decoder_norm=tuple(float(v) for v in norms),
    )
    order = np.lexsort((features, rows))
    effects, lists = {}, {}
    for task_id, cohort in cohorts.items():
        yes = cohort.yes[cohort.eligible]
        if not yes.any() or yes.all():
            continue  # an empty eligible cohort: no effect is defined, and the task's support row says so
        eligible, in_groups = z[cohort.eligible], groups[cohort.eligible]
        effects[task_id], (higher_yes, higher_no) = _task(
            eligible, yes, in_groups, norms=norms, features=ids, reporting=reporting
        )
        if support[task_id].tier == "primary":
            lists[task_id] = TaskLists(
                strongest_yes=tuple(_ranked(eligible[yes], norms, ids, in_groups[yes], reporting.top)),
                strongest_no=tuple(_ranked(eligible[~yes], norms, ids, in_groups[~yes], reporting.top)),
                higher_in_yes=tuple(higher_yes),
                higher_in_no=tuple(higher_no),
            )
    return FeatureTables(
        reporting=reporting,
        support=dict(support),
        features=summary,
        activations=Activations(
            row=tuple(int(r) for r in rows[order]),
            feature=tuple(int(f) for f in features[order]),
            value=tuple(float(v) for v in values[order]),
        ),
        strongest=tuple(_ranked(z, norms, ids, groups, reporting.top)),
        effects=effects,
        lists=lists,
    )


def listed_features(tables: FeatureTables) -> tuple[int, ...]:
    """Return every feature that appears in a compact list (the population's or a task's), ascending."""
    listed = {f.feature for f in tables.strongest}
    for task in tables.lists.values():
        listed |= {f.feature for f in (*task.strongest_yes, *task.strongest_no)}
        listed |= {f.feature for f in (*task.higher_in_yes, *task.higher_in_no)}
    return tuple(sorted(listed))
