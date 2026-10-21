"""The supervised logistic baseline: how accessible a suite's tasks are in an encoder's fixed read.

For one encoder read (the connector's ``input``: its layers and their combination, before any learned map) and one
Yes/No suite, each of the suite's tasks gets its own binary L2 logistic model (``build.logistic``), trained on the
connectors' training set and scored on the suite's own items, so its rows have the suite records' support:

- **training rows:** the training set's objects (``training_set``, the one the connectors trained on) that have a
  known label on the task within the task's scope (``Domain.scope_mask``, as the suite scores it); objects with no
  such label for any of the suite's tasks (e.g. the protein variants, for catalog tasks) are not in the design. Where
  the connectors' training set holds no label of the suite's tasks (PFMbench's localization: subcellular location is
  trained, but not as PFMbench task labels, and the PFMbench domain held only its test split), ``train_objects``
  instead lists the training objects of ``train_domain``, another data version of the suite's domain whose ``train``
  split holds the benchmark's own training proteins; the suite's
  tasks must be the same there (id, question, label, scope, group), and the design's labels, scope and clusters are
  read from it, the suite items' from the suite's domain;
- **selection:** the inverse penalty ``C`` from ``grid`` by cluster-grouped cross-validation on the training rows only
  (``cluster_folds`` over the domain's clusters, ``folds`` folds, ``seed``), by log-loss; then the fit on all of them.
  A task whose chosen ``C`` has a fit that did not reach ``tol`` within ``max_iter`` refuses the stage (a ``C`` not
  chosen may stop unconverged; its flag is in ``numbers``);
- **scores:** the chosen model's log-odds for every suite item in the task's scope, under the suite's one ``own``
  condition. As a suite run, an in-scope item without a label is an error. Unweighted: the class balance is the
  training rows' (reported, with the items', in the ``numbers`` output). ``more_suites`` (added field) are scored by
  the same models: their rows join the table under their own suite digest (its ``suite`` column), their items
  indexed in their own item set, so one fit serves several cohorts of one task (the protein-mutation cohorts);
- **suite-declared tasks** (``SuiteSpec.task_specs``: the variant labels) join the domain's catalog as a suite run
  joins them (``evaluation.runners``): each is in scope for every object, training objects included, so the design
  is every training object with a known label;
- **features:** the read of each object (``input``), or with ``minus_wt`` (added field) the read of each object minus
  its wild-type parent's (``ProteinDomain.variants``' ``parent``), so a model sees only the mutation's change: the
  training objects' parents are read from ``train_embeddings``, the items' from ``eval_embeddings``; an object
  without a parent is refused.

The ``scores`` output has the score-table shape (``evaluation.scores``) and the record's config carries the suite, so a
``grouped_contrasts`` analysis reads a single-suite baseline record as it reads a suite record (``step`` NA: no
checkpoint; the cells' ``step: null``); the item set and its resampling units are the suite's. A record with
``more_suites`` holds several suites' rows, each indexed in its own item set: only ``explicit_pairs`` reads it, choosing
the suite with ``params.suite``; ``grouped_contrasts`` must not. Nothing here depends on a receiver:
one fit of an (encoder read, suite) serves every receiver's connector of that read. It is a supervised accessibility
measure of the fixed representation, not a zero-shot score and not an upper bound on a connector.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, NamedTuple, Self

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.build.logistic import LogisticPath, fit_logistic_path
from latent_interconnect.build.zero_shot import training_set_construction
from latent_interconnect.config import ArtifactDigest, StageConfig, added_field
from latent_interconnect.data.items import ITEM_SET_KIND, OBJECT_ROLE, ItemSet, load_item_set
from latent_interconnect.data.suites import Condition, SuiteSpec
from latent_interconnect.data.training_text import TRAINING_SET_KIND, training_objects
from latent_interconnect.domains.base import Domain, DomainSpec, TaskKind, TaskSpec, open_domain, require_splits
from latent_interconnect.encoders.loader import InputLoader
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import EMBEDDINGS_KIND, EmbeddingReader
from latent_interconnect.evaluation.read_scan import MIN_FOLDS, cluster_folds
from latent_interconnect.evaluation.scores import write_numbers, write_scores
from latent_interconnect.writes.inputs import InputKind, InputSpec

__all__ = ["BaselineOutputs", "LogisticBaselineSpec", "logistic_baseline"]

_USE = "logistic baseline"
PositiveC = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class LogisticBaselineSpec(StageConfig):
    """One encoder read's logistic baseline on one Yes/No suite.

    Attributes:
        suite: the suite whose tasks and items are scored (``$ref``, the connectors' suite spec): a ``yes_no`` suite of
            one slot and the one condition ``own``.
        encoder: the read's encoder recipe (as the connectors' runs embed it).
        input: the read (``writes.inputs.InputSpec``, as the connectors' ``input``: layers, combination, an appended
            extraction).
        train_embeddings: the embedding artifacts covering the training objects of the design (the read's recipe and,
            with ``append``, the appended one's), and with ``minus_wt`` their parents.
        eval_embeddings: the embedding artifacts covering the suites' items' objects, and with ``minus_wt`` their
            parents.
        training_set: the connectors' training set (a ``training_set`` artifact of the suite's domain), or None
            with ``train_objects``.
        grid: the inverse penalties ``C`` tried, increasing (``build.logistic``).
        folds: the cluster folds of the selection.
        seed: the fold-hash seed (``cluster_folds``).
        max_iter: the most L-BFGS iterations of one penalty's solve.
        tol: the largest gradient component of a converged model.
        train_objects: the training objects as an object list (``item_set``) of ``train_domain``, in place of
            ``training_set`` (added field).
        train_domain: the data version of the suite's domain that holds ``train_objects`` and their labels (added
            field).
        more_suites: further suites the chosen models score (added field): each a Yes/No suite of ``suite``'s
            domains, slot, condition, tasks and declared tasks, on another item set.
        minus_wt: the features are each object's read minus its wild-type parent's (added field; the module
            docstring); False: the read.
    """

    suite: SuiteSpec
    encoder: EncoderRecipe
    input: InputSpec
    train_embeddings: tuple[ArtifactDigest, ...] = Field(min_length=1)
    eval_embeddings: tuple[ArtifactDigest, ...] = Field(min_length=1)
    training_set: ArtifactDigest | None = None
    grid: tuple[PositiveC, ...] = Field(min_length=1)
    folds: int = Field(ge=MIN_FOLDS)
    seed: int = 0
    max_iter: int = Field(ge=1)
    tol: float = Field(gt=0)
    train_objects: ArtifactDigest | None = added_field(default=None)
    train_domain: DomainSpec | None = added_field(default=None)
    more_suites: tuple[SuiteSpec, ...] = added_field(default=())
    minus_wt: bool = added_field(default=False)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.training_set is None) == (self.train_objects is None):
            raise ValueError(f"{_USE}: give exactly one of training_set and train_objects")
        if (self.train_objects is None) != (self.train_domain is None):
            raise ValueError(f"{_USE}: train_objects and train_domain come together")
        if list(self.grid) != sorted(set(self.grid)):
            raise ValueError(f"{_USE}: grid {list(self.grid)} must be strictly increasing")
        for suite in self.suites:
            if suite.kind != "yes_no" or suite.likelihood():
                raise ValueError(f"{_USE}: suite {suite.id!r} is {suite.kind!r}, not a Yes/No suite")
            if len(suite.slots) != 1 or suite.conditions != ({self.role: Condition.OWN},):
                raise ValueError(f"{_USE}: suite {suite.id!r} must have one slot and the one condition own")
        first = self.suite
        for suite in self.more_suites:
            same = (suite.domains, suite.slots, suite.tasks, suite.task_specs)
            if same != (first.domains, first.slots, first.tasks, first.task_specs):
                raise ValueError(f"{_USE}: suite {suite.id!r} differs from {first.id!r} in domains, slot or tasks")
        if len({s.item_set for s in self.suites}) != len(self.suites):
            raise ValueError(f"{_USE}: the suites must score distinct item sets")
        return self

    @property
    def suites(self) -> tuple[SuiteSpec, ...]:
        """Every suite scored: ``suite``, then ``more_suites``."""
        return (self.suite, *self.more_suites)

    @property
    def role(self) -> str:
        """The suite's one slot role (the score table's ``<column>.<role>``)."""
        return self.suite.slots[0].role


class BaselineOutputs(NamedTuple):
    """What the baseline commits.

    Attributes:
        scores: the score table (``scores``): one row per task and in-scope suite item.
        numbers: per task and penalty (key ``<task>/<C>``): ``value`` the cross-validated log-loss, ``C``, ``chosen``
            (1 for the task's chosen penalty), ``converged`` (1 if all its fits reached ``tol``; always 1 on a chosen
            row, an unconverged chosen penalty being refused), ``iterations`` (the penalty's shared solve), and the
            task's ``n_train``, ``positives_train``, ``n_eval``, ``positives_eval``.
    """

    scores: ArtifactRef
    numbers: ArtifactRef


def logistic_baseline(spec: LogisticBaselineSpec, store: ArtifactStore) -> BaselineOutputs:
    """Fit every task of the suite on the training set, choose its penalty, score the suite's items; commit both.

    The suite items' labels are read only after every model is chosen and fit, to fill the score table.

    Raises:
        ValueError: if the training set is of another domain data version than the suite's (``train_domain``: of
            another domain, or a suite task differs there), a design object is outside ``train`` or has no cluster,
            a training object is a suite item's, a task is not a Yes/No task of the domain or the suite's (or a
            declared task repeats a catalog one), an in-scope suite item has no label, with ``minus_wt`` an object
            has no parent, or a fit is refused (``build.logistic.fit_logistic_path``: one label class, or a chosen
            penalty's fit unconverged).
    """
    sets = [load_item_set(store, ArtifactRef(s.item_set, ITEM_SET_KIND)) for s in spec.suites]
    domain_spec = _suite_domain(spec, sets[0])
    domain = open_domain(domain_spec, store)
    tasks = _tasks(spec, domain)
    trained, listed = _training(spec, store, domain_spec, domain, tasks)
    evaluated = [tuple(items.objects[spec.role]) for items in sets]
    if shared := sorted(set(listed) & set().union(*evaluated)):
        raise ValueError(f"{_USE}: training objects {shared[:5]} are suite items")
    ids, rows, labels = _design(trained, tasks, listed, _declared(spec))
    require_splits(trained, ids, {"train"}, _USE)
    clusters = trained.clusters().reindex(list(ids))
    if clusters.isna().any():
        raise ValueError(f"{_USE}: design objects without a cluster {list(clusters.index[clusters.isna()])[:5]}")
    folds = cluster_folds([str(c) for c in clusters], spec.folds, spec.seed)
    X = _features(spec, store, spec.train_embeddings, ids, trained)
    path = fit_logistic_path(
        X, rows, labels, folds, n_folds=spec.folds, grid=spec.grid, max_iter=spec.max_iter, tol=spec.tol
    )
    del X
    parts, n_eval = [], {t.id: (0, 0) for t in tasks}
    for suite, objects in zip(spec.suites, evaluated, strict=True):
        scored = path.decision(_features(spec, store, spec.eval_embeddings, objects, domain))
        part, support = _score_rows(spec, suite, domain, tasks=tasks, objects=objects, scored=scored)
        parts.append(part)
        n_eval = {t: (n + support[t][0], k + support[t][1]) for t, (n, k) in n_eval.items()}
    table = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    meta: dict[str, object] = {"spec": spec.digest(), "suite": spec.suite.digest()}
    if spec.more_suites:
        meta["more_suites"] = [s.digest() for s in spec.more_suites]
    numbers = _numbers(tasks, path, rows, labels, n_eval)
    return BaselineOutputs(write_scores(store, table, meta), write_numbers(store, numbers, meta))


def _training(
    spec: LogisticBaselineSpec, store: ArtifactStore, domain_spec: DomainSpec, domain: Domain, tasks: Sequence[TaskSpec]
) -> tuple[Domain, tuple[str, ...]]:
    """Return the domain the design's labels come from and the training objects, in order."""
    if spec.training_set is not None:
        ref = ArtifactRef(spec.training_set, TRAINING_SET_KIND)
        if (built := training_set_construction(store, ref).domain) != domain_spec:
            raise ValueError(
                f"{_USE}: training set of {built.name} {built.data[:12]}, suite of {domain_spec.data[:12]}"
            )
        return domain, training_objects(store, ref)
    assert spec.train_domain is not None and spec.train_objects is not None  # the spec's check
    if spec.train_domain.name != domain_spec.name:
        raise ValueError(f"{_USE}: train_domain is {spec.train_domain.name!r}, the suite's {domain_spec.name!r}")
    trained = open_domain(spec.train_domain, store)
    catalog = trained.tasks()
    if moved := [t.id for t in tasks if t.id not in _declared(spec) and catalog.get(t.id) != t]:
        raise ValueError(f"{_USE}: tasks {moved} differ between train_domain and the suite's domain")
    listed = load_item_set(store, ArtifactRef(spec.train_objects, ITEM_SET_KIND))
    if dict(listed.roles) != {OBJECT_ROLE: domain_spec.name}:
        raise ValueError(f"{_USE}: train_objects {listed.name!r} is not an object list of {domain_spec.name!r}")
    return trained, listed.object_ids


def _suite_domain(spec: LogisticBaselineSpec, items: ItemSet) -> DomainSpec:
    """Return the suite's domain of its slot role (as a suite run finds it)."""
    name = items.roles[spec.role]
    found = [d for d in spec.suite.domains if d.name == name]
    if len(found) != 1:
        raise ValueError(f"{_USE}: role {spec.role!r} is domain {name!r}, cited {len(found)}x by the suite")
    return found[0]


def _declared(spec: LogisticBaselineSpec) -> frozenset[str]:
    """Return the ids of the suite's declared tasks (``SuiteSpec.task_specs``), in scope for every object."""
    return frozenset(t.id for t in spec.suite.task_specs)


def _tasks(spec: LogisticBaselineSpec, domain: Domain) -> tuple[TaskSpec, ...]:
    """Return the suite's tasks from the domain catalog and its declared tasks, each a Yes/No task."""
    catalog = dict(domain.tasks())
    declared = {t.id: t for t in spec.suite.task_specs}
    if clash := sorted(set(declared) & set(catalog)):
        raise ValueError(f"{_USE}: task_specs {clash} repeat the domain catalog's")
    catalog |= declared
    out = []
    for task_id in spec.suite.tasks:
        task = catalog.get(task_id)
        if task is None or task.kind is not TaskKind.YES_NO:
            raise ValueError(f"{_USE}: {task_id!r} is not a Yes/No task of the domain")
        out.append(task)
    return tuple(out)


def _in_scope(domain: Domain, task: TaskSpec, objects: Sequence[str], declared: frozenset[str]) -> np.ndarray:
    """Return whether each object is in the task's scope (``Domain.scope_mask``; every object for a declared task)."""
    if task.id in declared:
        return np.ones(len(objects), dtype=bool)
    index = pd.Index(list(objects))
    mask: np.ndarray = domain.scope_mask(task, index).reindex(index).fillna(False).to_numpy(bool)
    return mask


def _design(
    domain: Domain, tasks: Sequence[TaskSpec], listed: Sequence[str], declared: frozenset[str]
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return the design's objects (training objects with a known in-scope label on some task), rows and labels."""
    table = domain.labels()
    known = np.zeros((len(listed), len(tasks)), dtype=bool)
    values = np.zeros_like(known)
    for t, task in enumerate(tasks):
        column = table[task.label].reindex(list(listed))
        known[:, t] = column.notna().to_numpy() & _in_scope(domain, task, listed, declared)
        values[:, t] = np.where(known[:, t], column.fillna(False).to_numpy().astype(bool), False)
    keep = known.any(1)
    return tuple(o for o, k in zip(listed, keep, strict=True) if k), known[keep], values[keep]


def _features(
    spec: LogisticBaselineSpec, store: ArtifactStore, digests: Sequence[str], ids: Sequence[str], domain: Domain
) -> np.ndarray:
    """Return the objects' features from ``digests``, float64, in the order of ``ids``.

    The read (the spec's ``input``), or with ``minus_wt`` the read minus the read of each object's parent in
    ``domain`` (each parent read once).
    """
    readers = [EmbeddingReader(store, ArtifactRef(d, EMBEDDINGS_KIND)) for d in digests]
    loader = InputLoader(spec.input, readers, {InputKind.POOLED}, encoder=spec.encoder)

    def read(objects: Sequence[str]) -> np.ndarray:
        out: np.ndarray = loader.batch(tuple(objects)).require(InputKind.POOLED).numpy().astype(np.float64)
        return out

    X = read(ids)
    if not spec.minus_wt:
        return X
    variants = getattr(domain, "variants", None)  # ProteinDomain.variants: the variant objects and their parents
    if variants is None:
        raise ValueError(f"{_USE}: minus_wt: domain {domain.name!r} holds no variants")
    parents = variants()["parent"].reindex(list(ids))
    if parents.isna().any():
        raise ValueError(f"{_USE}: minus_wt: objects without a parent {list(parents.index[parents.isna()])[:5]}")
    unique = tuple(dict.fromkeys(parents.astype(str)))
    where = {p: i for i, p in enumerate(unique)}
    X -= read(unique)[[where[str(p)] for p in parents]]
    return X


def _score_rows(
    spec: LogisticBaselineSpec,
    suite: SuiteSpec,
    domain: Domain,
    *,
    tasks: Sequence[TaskSpec],
    objects: Sequence[str],
    scored: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, tuple[int, int]]]:
    """Return one suite's score table and each task's (items scored, positives): as a suite run scores the items."""
    table, parts, support = domain.labels(), [], {}
    for t, task in enumerate(tasks):
        items = np.flatnonzero(_in_scope(domain, task, objects, _declared(spec)))
        obj = [objects[int(i)] for i in items]
        values = table[task.label].reindex(obj)
        if values.isna().any():
            raise ValueError(f"{_USE}: task {task.id!r} has no label for {list(values.index[values.isna()])[:5]}")
        y = values.to_numpy().astype(bool).astype(np.float64)
        support[task.id] = (len(items), int(y.sum()))
        parts.append(
            pd.DataFrame(
                {
                    "suite": suite.digest(),
                    "step": pd.array([pd.NA] * len(items), dtype="Int64"),
                    "task": task.id,
                    "item": items.astype(np.int64),
                    "score": scored[items, t],
                    "label": y,
                    "batch": 0,
                    f"object.{spec.role}": obj,
                    f"filler.{spec.role}": obj,
                    f"condition.{spec.role}": str(Condition.OWN),
                    f"connector.{spec.role}": pd.array([pd.NA] * len(items), dtype="string"),
                }
            )
        )
    return pd.concat(parts, ignore_index=True), support


def _numbers(
    tasks: Sequence[TaskSpec],
    path: LogisticPath,
    rows: np.ndarray,
    labels: np.ndarray,
    n_eval: dict[str, tuple[int, int]],
) -> pd.DataFrame:
    """Return the number rows (``BaselineOutputs.numbers``)."""
    out = []
    for t, task in enumerate(tasks):
        for j, fit in enumerate(path.fits):
            out.append(
                {
                    "key": f"{task.id}/{fit.C!r}",
                    "task": task.id,
                    "value": float(fit.cv_log_loss[t]),
                    "C": fit.C,
                    "chosen": int(path.chosen[t] == j),
                    "converged": int(fit.converged[t].all()),
                    "iterations": fit.iterations,
                    "n_train": int(rows[:, t].sum()),
                    "positives_train": int((rows[:, t] & labels[:, t]).sum()),
                    "n_eval": n_eval[task.id][0],
                    "positives_eval": n_eval[task.id][1],
                }
            )
    return pd.DataFrame(out)
