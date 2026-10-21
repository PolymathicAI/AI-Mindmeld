"""The PFMbench domain: the paper's PFMbench test proteins and its 15 Yes/No evaluation tasks.

PFMbench's proteins are not SwissProt entries: they have no name, no split of ours and no clustering, and each task
has its own population. :class:`PfmbenchDataSpec` (registered in ``DOMAIN_DATA`` as ``pfmbench``) builds them the way
the paper's evaluation read them:

- the five binary tasks (DeepLocBinary, DeepSol, metal_ion_binding, cloning_clf, material_production) come from
  ``pfm_binary_test.parquet`` (``id``, ``aa_seq``, ``task``, ``label``): per task, the evaluation draw of at most
  ``max_per_task`` proteins (``sample(min(n, max_per_task), random_state=seed)`` over the task's rows, grouped by
  task), each answered Yes where its label is the task's ``yes_label`` (DeepLocBinary: 0, a membrane protein);
- the ten DeepLoc2 location tasks come from ``dl2multi_test_n600.parquet`` (``id``, ``aa_seq``, ``labels``: the
  comma-separated compartment indices): every protein, answered Yes where its labels hold the task's ``index``;
- the population is ``r11_eval_ids.parquet`` (``id``, ``task``), which the build checks equals that draw and the 600
  DeepLoc2 proteins exactly (5,166 proteins).

Objects are keyed by the benchmark's ids (``DeepSol::...``, ``DL2::...``); the same sequence may occur under two or
three ids of different benchmarks (132 sequences; 5,166 objects, 5,032 distinct sequences), and is then that many
objects. Each object's ``value`` is its sequence, it has no name and every object is in ``test``. Objects with the same
sequence share one cluster (named by the first of their ids), every other object is its own
(identical sequences are one protein, so the bootstrap resamples them as one unit; the paper's bootstrap treated them as
independent, which changes interval widths only). A task is scored on its own population only: the scope table
(``scope.parquet``) marks it, and its label is unknown (null) elsewhere. The four non-localization tasks (DeepSol,
metal_ion_binding, cloning_clf, material_production) are held out (``TaskScope.HELD_OUT``, zero-shot transfer); the
eleven localization tasks (the ten DeepLoc2 compartments and DeepLocBinary, the binary task in the ``localization``
group) are ``TaskScope.TRAINED``, because subcellular location is part of the ordinary protein connectors' training
(asked here of the benchmark's proteins, in its own wording). The scope changes no scoring (a suite names its tasks
and scores each on its scope table), and a domain-data artifact keeps the catalog it was built with. The domain's
objects are proteins (``object_kind = "protein"``), so the protein encoders extract them.

A version may also hold PFMbench's own training split of some tasks (``binary_train``, ``locations_train``; added
fields, so a version without them is the test-only domain it was): those proteins are in ``train``, each in the scope
of its own tasks only, with labels defined exactly as the test rows' (the same ``yes_label``, the same compartment
indices). They serve supervised baselines fit on the benchmark's own training proteins; no connector evaluation reads
them. The test objects, labels, scope and clusters are the test-only version's, unchanged: a cluster (identical
sequences) is named by its first test object when it holds one, else by its first object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.config import Config, added_field
from latent_interconnect.domains.base import DOMAINS, Frame, TaskKind, TaskScope, TaskSpec
from latent_interconnect.domains.tables import (
    DOMAIN_DATA,
    DomainDataSpec,
    DomainTables,
    SourceFile,
    TableDomain,
    read_source,
)

__all__ = ["BinaryTask", "LocationTask", "PfmbenchDataSpec", "PfmbenchDomain"]

_SPLIT = "test"
_TRAIN = "train"
_LOCATION_SET = "DeepLoc2Multi"
_LABEL_SEP = ","
_LOCALIZATION = "localization"


@DOMAINS.register("pfmbench")
class PfmbenchDomain(TableDomain):
    """PFMbench's test proteins, encoded from their sequences (objects of kind ``protein``)."""

    name = "pfmbench"
    noun = "protein"
    object_kind = "protein"


class BinaryTask(Config):
    """One of PFMbench's binary tasks.

    Attributes:
        id: the task id (the benchmark's task name).
        question: the bare question (the frame places it).
        yes_label: the label value that means Yes (not ``yes``: YAML 1.1 reads a bare ``yes`` key as a boolean).
        group: the figure group; a task in the ``localization`` group is a trained task (``TaskScope.TRAINED``),
            any other is held out.
    """

    id: str
    question: str
    yes_label: Literal[0, 1]
    group: str


class LocationTask(Config):
    """One DeepLoc2 compartment.

    Attributes:
        id: the task id.
        question: the bare question.
        index: the compartment's index in the ``labels`` column.
        group: the figure group.
    """

    id: str
    question: str
    index: int = Field(ge=0)
    group: str


@DOMAIN_DATA.register("pfmbench")
class PfmbenchDataSpec(DomainDataSpec):
    """A PFMbench domain-data version.

    Attributes:
        binary: the binary tasks' test table (``id``, ``aa_seq``, ``task``, ``label``).
        locations: DeepLoc2's 600 test proteins (``id``, ``aa_seq``, ``labels``).
        population: the evaluated proteins (``id``, ``task``), checked against the draw.
        max_per_task: the draw's cap per binary task.
        seed: the draw's ``random_state``.
        binary_tasks: the binary tasks, in catalog order.
        location_tasks: the DeepLoc2 compartments, in catalog order.
        frames: the prompt frames.
        binary_train: binary task id -> its training split (``id``, ``aa_seq``, ``label``: the test table's label
            values), every row in ``train`` (added field).
        locations_train: DeepLoc2's training split (``id``, ``aa_seq``, ``label``: the comma-separated compartment
            indices, as the test table's ``labels``), every row in ``train``, or None (added field).
    """

    binary: SourceFile
    locations: SourceFile
    population: SourceFile
    max_per_task: int = Field(ge=1)
    seed: int
    binary_tasks: tuple[BinaryTask, ...] = Field(min_length=1)
    location_tasks: tuple[LocationTask, ...] = Field(min_length=1)
    frames: tuple[Frame, ...] = Field(min_length=1)
    binary_train: dict[str, SourceFile] = added_field(default_factory=dict)
    locations_train: SourceFile | None = added_field(default=None)

    @model_validator(mode="after")
    def _check(self) -> PfmbenchDataSpec:
        ids = [t.id for t in self.binary_tasks] + [t.id for t in self.location_tasks]
        if len(set(ids)) != len(ids):
            raise ValueError(f"a task id repeats in {ids}")
        if len({t.index for t in self.location_tasks}) != len(self.location_tasks):
            raise ValueError("a DeepLoc2 index repeats")
        stray = sorted(set(self.binary_train) - {t.id for t in self.binary_tasks})
        if stray:
            raise ValueError(f"binary_train names tasks that are not binary tasks: {stray}")
        return self

    def sources(self) -> dict[str, SourceFile]:
        """Return the binary table, the DeepLoc2 table and the population, then the training splits, in order."""
        out = {"binary": self.binary, "locations": self.locations, "population": self.population}
        out |= {f"binary_train/{task}": source for task, source in self.binary_train.items()}
        if self.locations_train is not None:
            out["locations_train"] = self.locations_train
        return out

    def tables(self, data_root: Path) -> DomainTables:
        """Read and check the sources and derive the domain tables.

        Raises:
            IntegrityError: if a source's sha256 differs from the spec.
            ValueError: if a binary label is not 0 or 1, a declared binary task is not in the table, an id repeats
                (test and training rows together), a DeepLoc2 label index is out of the declared compartments, or the
                population is not exactly the draw and the DeepLoc2 proteins.
        """
        binary = pd.read_parquet(read_source(data_root, self.binary), columns=["id", "aa_seq", "task", "label"])
        locations = pd.read_parquet(read_source(data_root, self.locations), columns=["id", "aa_seq", "labels"])
        population = pd.read_parquet(read_source(data_root, self.population), columns=["id", "task"])
        odd = sorted(int(v) for v in set(binary["label"].unique()) - {0, 1})
        if odd:
            raise ValueError(f"{self.binary.path}: labels {odd} are not 0 or 1")
        missing = sorted({t.id for t in self.binary_tasks} - set(binary["task"]))
        if missing:
            raise ValueError(f"binary tasks {missing} are not in {self.binary.path}")
        draw = pd.concat(
            [g.sample(min(len(g), self.max_per_task), random_state=self.seed) for _, g in binary.groupby("task")]
        )
        draw = draw[draw["task"].isin({t.id for t in self.binary_tasks})]
        drawn = set(zip(draw["id"], draw["task"], strict=True)) | {(i, _LOCATION_SET) for i in locations["id"]}
        if set(zip(population["id"], population["task"], strict=True)) != drawn or not population["id"].is_unique:
            raise ValueError(f"{self.population.path} is not the draw (max {self.max_per_task}, seed {self.seed})")
        if not (draw["id"].is_unique and locations["id"].is_unique) or set(draw["id"]) & set(locations["id"]):
            raise ValueError("a PFMbench id repeats")
        train_binary = {t: _train_table(data_root, s, binary=True) for t, s in self.binary_train.items()}
        train_locations = (
            _train_table(data_root, self.locations_train, binary=False).rename(columns={"label": "labels"})
            if self.locations_train is not None
            else locations.iloc[:0]
        )
        test_ids = [*draw["id"], *locations["id"]]
        train_ids = [*(i for t in train_binary.values() for i in t["id"]), *train_locations["id"]]
        ids = pd.Index([*test_ids, *train_ids], name="object_id")
        if not ids.is_unique:
            raise ValueError(
                f"PFMbench ids repeat across the test and training rows: {list(ids[ids.duplicated()][:5])}"
            )
        values = [
            *draw["aa_seq"],
            *locations["aa_seq"],
            *(v for t in train_binary.values() for v in t["aa_seq"]),
            *train_locations["aa_seq"],
        ]
        objects = pd.DataFrame({"value": values, "name": None}, index=ids)
        split = pd.Series([_SPLIT] * len(test_ids) + [_TRAIN] * len(train_ids), index=ids)
        labels = pd.DataFrame(index=ids)
        scope = pd.DataFrame(index=ids)
        tasks = []
        for task in self.binary_tasks:
            trained = train_binary.get(task.id, draw.iloc[:0])
            rows = pd.concat([draw[draw["task"] == task.id][["id", "label"]], trained[["id", "label"]]]).set_index("id")
            labels[task.id] = pd.Series(rows["label"] == task.yes_label, dtype="boolean").reindex(ids)
            scope[task.id] = ids.isin(rows.index)
            tasks.append(_task(task.id, task.question, task.group, trained=task.group == _LOCALIZATION))
        located = pd.concat([locations[["id", "labels"]], train_locations[["id", "labels"]]]).set_index("id")
        compartments = located["labels"].map(lambda v: {int(x) for x in str(v).split(_LABEL_SEP)})
        stray = set().union(*compartments) - {t.index for t in self.location_tasks}
        if stray:
            raise ValueError(f"DeepLoc2 label indices {sorted(stray)} are not declared compartments")
        for location in self.location_tasks:
            known = compartments.map(lambda held, k=location.index: k in held)
            labels[location.id] = pd.Series(known, dtype="boolean").reindex(ids)
            scope[location.id] = ids.isin(compartments.index)
            tasks.append(_task(location.id, location.question, location.group, trained=True))
        return DomainTables(
            domain="pfmbench",
            objects=objects,
            split=split,
            clusters=_clusters(objects["value"], split),
            labels=labels,
            fields=pd.DataFrame(index=ids),
            tasks=tuple(tasks),
            frames=self.frames,
            recipes=(),
            scope=scope,
        )


def _train_table(data_root: Path, source: SourceFile, *, binary: bool) -> pd.DataFrame:
    """Read one training split (``id``, ``aa_seq``, ``label``): ids unique and, for a binary task, labels 0 or 1."""
    table = pd.read_parquet(read_source(data_root, source), columns=["id", "aa_seq", "label"])
    if not table["id"].is_unique:
        raise ValueError(f"{source.path}: ids repeat")
    if binary and (odd := sorted(int(v) for v in set(table["label"].unique()) - {0, 1})):
        raise ValueError(f"{source.path}: labels {odd} are not 0 or 1")
    return table


def _clusters(values: pd.Series, split: pd.Series) -> pd.Series:
    """One cluster per distinct sequence (one protein, one unit), named by its first test object.

    A cluster with no test object is named by its first object, so the test objects' clusters are the test-only
    version's.
    """
    frame = values.rename("value").reset_index()
    frame["train"] = split.reindex(values.index).ne(_SPLIT).to_numpy()
    first = frame.sort_values(["train", "object_id"]).groupby("value", sort=False)["object_id"].first()
    return pd.Series(frame["value"].map(first).to_numpy(), index=values.index)


def _task(task_id: str, question: str, group: str, *, trained: bool) -> TaskSpec:
    """Return a Yes/No task: trained scope for a localization task (subcellular location is trained), else held out."""
    scope = TaskScope.TRAINED if trained else TaskScope.HELD_OUT
    return TaskSpec(id=task_id, kind=TaskKind.YES_NO, question=question, label=task_id, scope=scope, group=group)
