"""The domain-data artifact format, and :class:`TableDomain`, a :class:`Domain` read from it.

A domain-data artifact (kind :data:`~latent_interconnect.domains.base.DOMAIN_DATA_KIND`) is the directory a domain is
constructed on. Every domain's data has the same files, each table keyed by ``object_id``:

- ``objects.parquet``: ``object_id``, ``value`` (raw encoder input), ``name``;
- ``split.parquet``: ``object_id``, ``split`` (``train`` / ``val`` / ``test``; an object in no split has no row);
- ``clusters.parquet``: ``object_id``, ``cluster``;
- ``labels.parquet``: ``object_id`` and one column per label;
- ``fields.parquet``: ``object_id`` and one column per field: text (string) or an item list (list of strings);
- ``catalog.json``: ``tasks``, ``frames`` and ``recipes`` (:class:`TaskSpec`, :class:`Frame` and
  :class:`CaptureRecipe` dumps, in catalog order);
- ``scope.parquet``, only when some task is scored on a subset: ``object_id`` and one bool column per such task id,
  True where the task may be scored (``Domain.scope_mask``; e.g. an unstated subset). A task without a column is
  scored on every object;
- one ``<name>.parquet`` per extra table a domain adds (e.g. the protein domain's ``variants``), keyed by
  ``object_id`` and read with :meth:`TableDomain.table`.

The manifest metadata holds ``domain``, the object count, the label and field columns, the extra tables' names, and
whatever the builder adds (its spec digest, its source hashes).

A domain's data is built from its sources by a :class:`DomainDataSpec`: the domain's own subclass, registered in
:data:`DOMAIN_DATA` under the domain's name, cites every source file by a path under the data root and its sha256
(:class:`SourceFile`, checked by :func:`read_source` before reading) and derives the :class:`DomainTables`;
:meth:`DomainDataSpec.build` commits them with :func:`commit_domain_data`. The domain stage
(``latent_interconnect.run.domain``) builds any registered domain this way, so a third-party domain adds a spec class
and registers it, with no change to shared code (``domains.protein``, ``domains.molecule``).
"""

from __future__ import annotations

import json
import re
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore, IntegrityError, sha256_file
from latent_interconnect.config import Config
from latent_interconnect.domains.base import DOMAIN_DATA_KIND, CaptureRecipe, Domain, Frame, TaskSpec
from latent_interconnect.plugins import Registry

__all__ = [
    "DOMAIN_DATA",
    "DomainDataSpec",
    "DomainTables",
    "SourceFile",
    "TableDomain",
    "commit_domain_data",
    "read_source",
]

_ID = "object_id"
_RESERVED_META = {"domain", "n_objects", "labels", "fields", "tables"}
_SCOPE = "scope.parquet"
_CORE_TABLES = {"objects", "split", "clusters", "labels", "fields", "catalog", "scope"}
_TABLE_NAME = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class DomainTables:
    """The content of one domain-data artifact.

    Attributes:
        domain: the registered domain name the data is for.
        objects: ``object_id`` (index) → ``value``, ``name``.
        split: ``object_id`` → split.
        clusters: ``object_id`` → cluster id (string).
        labels: ``object_id`` (index) × label columns.
        fields: ``object_id`` (index) × text-field columns.
        tasks: the task catalog, in order.
        frames: the prompt frames, in order.
        recipes: the capture recipes, in order.
        scope: ``object_id`` (index) × task id: True where the task may be scored, for the tasks scored on a subset
            only; None (no ``scope.parquet``) when every task is scored on every object.
        extra: table name → a domain-specific table indexed by ``object_id`` (e.g. ``variants``); each is written as
            ``<name>.parquet``.
    """

    domain: str
    objects: pd.DataFrame
    split: pd.Series
    clusters: pd.Series
    labels: pd.DataFrame
    fields: pd.DataFrame
    tasks: tuple[TaskSpec, ...]
    frames: tuple[Frame, ...]
    recipes: tuple[CaptureRecipe, ...]
    scope: pd.DataFrame | None = None
    extra: dict[str, pd.DataFrame] = field(default_factory=dict)


def commit_domain_data(store: ArtifactStore, tables: DomainTables, meta: Mapping[str, Any]) -> ArtifactRef:
    """Write the tables in the domain-data format and commit them.

    Args:
        store: the artifact store.
        tables: the content.
        meta: builder metadata (e.g. its spec digest and source hashes); must not use the reserved keys.

    Raises:
        ValueError: if ``meta`` uses a reserved key, ids repeat, a task, frame or recipe name repeats, or the scope
            table names an unknown task, is not bool or does not cover every object, a field mixes text and item
            lists, or an extra table's name is not a lower-case identifier, is a core table's, or it holds unknown
            object ids.
    """
    clash = _RESERVED_META & set(meta)
    if clash:
        raise ValueError(f"meta uses reserved keys {sorted(clash)}")
    if not tables.objects.index.is_unique:
        raise ValueError(f"{tables.domain}: object ids repeat")
    catalogued = (
        ("task", [t.id for t in tables.tasks]),
        ("frame", [f.id for f in tables.frames]),
        ("recipe", [r.name for r in tables.recipes]),
    )
    for what, ids in catalogued:
        if len(set(ids)) != len(ids):
            raise ValueError(f"{tables.domain}: a {what} id repeats in {ids}")
    _check_fields(tables)
    if tables.scope is not None:
        _check_scope(tables, tables.scope)
    known = set(tables.objects.index)
    for name, table in tables.extra.items():
        if not _TABLE_NAME.fullmatch(name) or name in _CORE_TABLES:
            raise ValueError(f"{tables.domain}: extra table name {name!r} is not allowed")
        stray = set(table.index) - known
        if stray:
            raise ValueError(f"{tables.domain}: extra table {name!r} has unknown object ids {sorted(stray)[:5]}")
    staged = store.staging(f"domain-{uuid.uuid4().hex}")
    tables.objects[["value", "name"]].rename_axis(_ID).reset_index().to_parquet(staged / "objects.parquet", index=False)
    tables.split.rename("split").rename_axis(_ID).reset_index().to_parquet(staged / "split.parquet", index=False)
    tables.clusters.rename("cluster").rename_axis(_ID).reset_index().to_parquet(
        staged / "clusters.parquet", index=False
    )
    tables.labels.rename_axis(_ID).reset_index().to_parquet(staged / "labels.parquet", index=False)
    tables.fields.rename_axis(_ID).reset_index().to_parquet(staged / "fields.parquet", index=False)
    if tables.scope is not None:
        tables.scope.rename_axis(_ID).reset_index().to_parquet(staged / _SCOPE, index=False)
    for name, table in tables.extra.items():
        table.rename_axis(_ID).reset_index().to_parquet(staged / f"{name}.parquet", index=False)
    catalog = {
        "tasks": [t.model_dump(mode="json") for t in tables.tasks],
        "frames": [f.model_dump(mode="json") for f in tables.frames],
        "recipes": [r.model_dump(mode="json") for r in tables.recipes],
    }
    (staged / "catalog.json").write_text(json.dumps(catalog, indent=1, ensure_ascii=False))
    return store.commit(
        staged,
        DOMAIN_DATA_KIND,
        {
            **meta,
            "domain": tables.domain,
            "n_objects": len(tables.objects),
            "labels": [str(c) for c in tables.labels.columns],
            "fields": [str(c) for c in tables.fields.columns],
            "tables": sorted(tables.extra),
        },
    )


def _check_fields(tables: DomainTables) -> None:
    for column, values in tables.fields.items():
        present = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]
        text = all(isinstance(v, str) for v in present)
        items = all(isinstance(v, tuple | list) and all(isinstance(i, str) for i in v) for v in present)
        if not (text or items):
            raise ValueError(f"{tables.domain}: field {column!r} must hold only text or only item lists of strings")


def _check_scope(tables: DomainTables, scope: pd.DataFrame) -> None:
    unknown = set(scope.columns) - {t.id for t in tables.tasks}
    if unknown:
        raise ValueError(f"{tables.domain}: the scope table names unknown tasks {sorted(unknown)}")
    if not all(pd.api.types.is_bool_dtype(scope[c]) for c in scope.columns):
        raise ValueError(f"{tables.domain}: the scope table must be bool")
    if not scope.index.sort_values().equals(tables.objects.index.sort_values()):
        raise ValueError(f"{tables.domain}: the scope table must cover exactly the objects")


class SourceFile(Config):
    """One source file of a domain-data build.

    Attributes:
        path: the path under the data root (``Paths.data``).
        sha256: its expected sha256 (checked before it is read).
    """

    path: str
    sha256: str


def read_source(data_root: Path, source: SourceFile) -> Path:
    """Return the source's path under ``data_root`` once its sha256 matches.

    Raises:
        IntegrityError: if the file's sha256 differs from ``source.sha256``.
    """
    path = Path(data_root) / source.path
    found = sha256_file(path)
    if found != source.sha256:
        raise IntegrityError(f"{source.path}: sha256 {found}, expected {source.sha256}")
    return path


class DomainDataSpec(Config, ABC):
    """A domain-data version: the sources one domain's tables are derived from, and its catalog.

    Each domain subclasses it with its own fields and registers the subclass in :data:`DOMAIN_DATA` under the domain's
    name; the domain stage builds any registered spec.
    """

    @abstractmethod
    def sources(self) -> dict[str, SourceFile]:
        """Return the source files by role, in a fixed order (their hashes go into the artifact's metadata)."""

    @abstractmethod
    def tables(self, data_root: Path) -> DomainTables:
        """Read and check the sources under ``data_root`` and derive the domain tables (no store needed).

        Raises:
            IntegrityError: if a source's sha256 differs from the spec.
        """

    def artifact_inputs(self, store: ArtifactStore) -> dict[str, ArtifactRef]:
        """Return the artifacts the build reads, by role (recorded as the domain stage's inputs; default: none)."""
        del store
        return {}

    def build(self, store: ArtifactStore, data_root: Path) -> ArtifactRef:
        """Derive the tables and commit the domain-data artifact, with this spec's digest and the sources' hashes.

        Args:
            store: the artifact store.
            data_root: the data root the sources' paths are under (``Paths.data``).

        Raises:
            IntegrityError: if a source's sha256 differs from the spec.
            ValueError: as the subclass's :meth:`tables`, or :func:`commit_domain_data`.
        """
        meta = {"spec": self.digest(), "sources": {k: s.sha256 for k, s in self.sources().items()}}
        return commit_domain_data(store, self.tables(data_root), meta)


DOMAIN_DATA: Registry[type[DomainDataSpec]] = Registry("domain data spec")
"""Domain name → the :class:`DomainDataSpec` subclass its data is built from."""


class TableDomain(Domain):
    """A domain served from a domain-data artifact directory (tables read once, on first use).

    Subclasses set ``name`` and ``noun`` and register in :data:`~latent_interconnect.domains.base.DOMAINS`. A task
    scored on a subset has a column in the artifact's scope table, which :meth:`scope_mask` reads.

    Args:
        data_root: the artifact directory.
    """

    def _table(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.data_root / f"{name}.parquet").set_index(_ID)

    @cached_property
    def _objects(self) -> pd.DataFrame:
        return self._table("objects")

    @cached_property
    def _split(self) -> pd.Series:
        return self._table("split")["split"]

    @cached_property
    def _clusters(self) -> pd.Series:
        return self._table("clusters")["cluster"]

    @cached_property
    def _labels(self) -> pd.DataFrame:
        return self._table("labels")

    @cached_property
    def _fields(self) -> pd.DataFrame:
        fields = self._table("fields")
        for column in fields.columns:  # an item-list column comes back as arrays; the contract's cells are tuples
            if fields[column].map(lambda v: hasattr(v, "__len__") and not isinstance(v, str)).any():
                fields[column] = fields[column].map(lambda v: None if v is None else tuple(str(i) for i in v))
        return fields

    @cached_property
    def _catalog(self) -> tuple[dict[str, TaskSpec], dict[str, Frame], dict[str, CaptureRecipe]]:
        body = json.loads((self.data_root / "catalog.json").read_text())
        tasks = {t["id"]: TaskSpec.model_validate(t) for t in body["tasks"]}
        frames = {f["id"]: Frame.model_validate(f) for f in body["frames"]}
        recipes = {r["name"]: CaptureRecipe.model_validate(r) for r in body["recipes"]}
        return tasks, frames, recipes

    def objects(self) -> pd.DataFrame:
        """Return the object table: ``object_id`` (index), ``value``, ``name``."""
        return self._objects

    def table(self, name: str) -> pd.DataFrame | None:
        """Return the extra table ``name`` (indexed by ``object_id``), or None if this data version has none."""
        path = self.data_root / f"{name}.parquet"
        if not _TABLE_NAME.fullmatch(name) or name in _CORE_TABLES or not path.exists():
            return None
        return pd.read_parquet(path).set_index(_ID)

    def table_names(self) -> frozenset[str]:
        """Return the names of this data version's extra tables."""
        names = (p.stem for p in self.data_root.glob("*.parquet"))
        return frozenset(n for n in names if _TABLE_NAME.fullmatch(n) and n not in _CORE_TABLES)

    def split(self) -> pd.Series:
        """Return ``object_id`` → split; objects in no split are absent."""
        return self._split

    def clusters(self) -> pd.Series:
        """Return ``object_id`` → cluster id."""
        return self._clusters

    def labels(self) -> pd.DataFrame:
        """Return the materialized label table."""
        return self._labels

    def fields(self) -> pd.DataFrame:
        """Return the field table: text fields and item-list fields (``Domain.fields``)."""
        return self._fields

    def tasks(self) -> Mapping[str, TaskSpec]:
        """Return the task catalog by id, in catalog order."""
        return self._catalog[0]

    def frames(self) -> Mapping[str, Frame]:
        """Return the prompt frames by id, in catalog order."""
        return self._catalog[1]

    def capture_recipes(self) -> Mapping[str, CaptureRecipe]:
        """Return the capture recipes by name, in catalog order."""
        return self._catalog[2]

    @cached_property
    def _scope(self) -> pd.DataFrame | None:
        return self._table("scope") if (self.data_root / _SCOPE).exists() else None

    def scope_mask(self, task: TaskSpec, object_ids: pd.Index) -> pd.Series:
        """Return the task's column of the scope table, or all True for a task without one.

        Raises:
            KeyError: for an object id the domain does not hold.
        """
        unknown = pd.Index(object_ids).difference(self._objects.index)
        if len(unknown):
            raise KeyError(f"{self.name}: unknown objects {list(unknown[:5])}")
        if self._scope is None or task.id not in self._scope.columns:
            return pd.Series(True, index=object_ids)
        return self._scope[task.id].reindex(object_ids).astype(bool)
