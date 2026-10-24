"""Views of a domain for training streams: a values-free catalog to declare from, and a restricted view to draw from.

A training stream first declares what it may put into an example (``data.streams.Supervision``) from a
:class:`CatalogView`, which holds names only: the domain's name and noun, its task catalog, its label column names and
its extra table names, and no data value. The stream is then built on a :class:`RestrictedDomain` (a
:class:`DomainView`): the declared tasks, the declared label columns, the declared extra tables, and the object
clusters and population, each over the training objects' rows only, so no validation or test object's label is
reachable from a stream (an extra table too: one not keyed by ``object_id`` is refused). Object values and
names, the domain's text fields and its frames are not part of the view: anything a training prompt shows is training
text (``data.training_text``, restricted to the declared fields) or the stream's own wording. An undeclared task,
label column or table raises ``KeyError`` naming the stream.

Both views show each task with its catalog scope, except for the streams of an all-tasks training set
(``build.zero_shot.TrainingSetSpec.all_tasks``), which no task is held out of: built with ``all_trained``, they show
every task as trained (``TaskScope.TRAINED``), so the streams' own refusal of untrained tasks lets them all through.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator, Mapping
from typing import TYPE_CHECKING, Protocol

from latent_interconnect.domains.base import Domain, TaskScope, TaskSpec

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["CatalogView", "DomainCatalog", "DomainView", "RestrictedDomain"]


class CatalogView(Protocol):
    """What a stream may read to declare its supervision: names, never data values."""

    @property
    def name(self) -> str:
        """The domain's registry name."""
        ...

    @property
    def noun(self) -> str:
        """How prompts refer to an object."""
        ...

    def tasks(self) -> Mapping[str, TaskSpec]:
        """Return the task catalog by id."""
        ...

    def label_columns(self) -> frozenset[str]:
        """Return the names of the label columns."""
        ...

    def table_names(self) -> frozenset[str]:
        """Return the names of the extra tables."""
        ...


class DomainView(CatalogView, Protocol):
    """What a training stream may read of its domain.

    Its declared tasks, label columns and tables (as a :class:`CatalogView` of those), their values, the clusters and
    the population.
    """

    def labels(self, columns: Collection[str]) -> pd.DataFrame:
        """Return the declared label ``columns``."""
        ...

    def table(self, name: str) -> pd.DataFrame:
        """Return the declared extra table ``name``."""
        ...

    def clusters(self) -> pd.Series:
        """Return ``object_id`` → cluster id."""
        ...

    def population(self) -> pd.Index:
        """Return the objects population draws draw from (``Domain.population``)."""
        ...


def _catalog(domain: Domain, all_trained: bool) -> Mapping[str, TaskSpec]:
    """Return the domain's task catalog; with ``all_trained``, each task as a trained one (its scope alone changed)."""
    tasks = domain.tasks()
    if not all_trained:
        return tasks
    return {t: spec.model_copy(update={"scope": TaskScope.TRAINED}) for t, spec in tasks.items()}


class DomainCatalog:
    """A :class:`CatalogView` of a domain.

    Args:
        domain: the domain.
    """

    def __init__(self, domain: Domain, *, all_trained: bool = False) -> None:
        self._domain, self._all_trained = domain, all_trained

    @property
    def name(self) -> str:
        """The domain's registry name."""
        return self._domain.name

    @property
    def noun(self) -> str:
        """How prompts refer to an object."""
        return self._domain.noun

    def tasks(self) -> Mapping[str, TaskSpec]:
        """Return the task catalog by id."""
        return _catalog(self._domain, self._all_trained)

    def label_columns(self) -> frozenset[str]:
        """Return the names of the label columns."""
        return frozenset(str(c) for c in self._domain.labels().columns)

    def table_names(self) -> frozenset[str]:
        """Return the names of the extra tables."""
        return self._domain.table_names()


class _Declared(Mapping[str, TaskSpec]):
    """The declared tasks of a catalog; an undeclared one raises ``KeyError`` naming the user."""

    def __init__(self, tasks: Mapping[str, TaskSpec], declared: frozenset[str], user: str) -> None:
        self._tasks, self._declared, self._user = tasks, declared, user

    def __getitem__(self, task: str) -> TaskSpec:
        if task not in self._declared:
            raise KeyError(f"{self._user} did not declare task {task!r}")
        return self._tasks[task]

    def __iter__(self) -> Iterator[str]:
        return (t for t in self._tasks if t in self._declared)

    def __len__(self) -> int:
        return sum(1 for _ in self)


class RestrictedDomain:
    """A :class:`DomainView` (and a :class:`CatalogView`) of a domain, restricted to what one stream declared.

    Args:
        domain: the domain.
        tasks: the declared task ids.
        labels: the declared label columns.
        tables: the declared extra tables.
        objects: the objects whose rows the view shows (the training text's objects).
        user: who reads (named in every refusal, e.g. ``"stream 'yes_no'"``).
        all_trained: show every task as trained (the streams of an all-tasks training set).

    Raises:
        KeyError: if a declared task, label column or table is not in the domain.
    """

    def __init__(
        self,
        domain: Domain,
        *,
        tasks: Collection[str],
        labels: Collection[str],
        tables: Collection[str],
        objects: Collection[str],
        user: str,
        all_trained: bool = False,
    ) -> None:
        self._domain, self._user, self._all_trained = domain, user, all_trained
        self._objects = frozenset(objects)
        self._tasks, self._labels, self._tables = frozenset(tasks), frozenset(labels), frozenset(tables)
        missing = {
            "tasks": sorted(self._tasks - set(domain.tasks())),
            "label columns": sorted(self._labels - {str(c) for c in domain.labels().columns}),
            "tables": sorted(self._tables - domain.table_names()),
        }
        if any(missing.values()):
            raise KeyError(f"{user} declares what domain {domain.name!r} lacks: {missing}")

    @property
    def name(self) -> str:
        """The domain's registry name."""
        return self._domain.name

    @property
    def noun(self) -> str:
        """How prompts refer to an object."""
        return self._domain.noun

    def tasks(self) -> Mapping[str, TaskSpec]:
        """Return the declared tasks by id; looking up another raises ``KeyError`` naming the stream."""
        return _Declared(_catalog(self._domain, self._all_trained), self._tasks, self._user)

    def label_columns(self) -> frozenset[str]:
        """Return the declared label columns (a restricted view is also a :class:`CatalogView`)."""
        return self._labels

    def table_names(self) -> frozenset[str]:
        """Return the declared extra tables."""
        return self._tables

    def labels(self, columns: Collection[str]) -> pd.DataFrame:
        """Return the label ``columns``.

        Raises:
            KeyError: naming the stream and the columns it did not declare.
        """
        undeclared = sorted(set(columns) - self._labels)
        if undeclared:
            raise KeyError(f"{self._user} did not declare label columns {undeclared}")
        found = self._domain.labels()[list(columns)]
        return found[found.index.isin(self._objects)]

    def table(self, name: str) -> pd.DataFrame:
        """Return the extra table ``name``.

        Raises:
            KeyError: naming the stream, if it did not declare the table (even when the domain has it), or if the table
                is not keyed by ``object_id`` (its rows could not be restricted to the view's objects).
        """
        if name not in self._tables:
            raise KeyError(f"{self._user} did not declare table {name!r}")
        found = self._domain.table(name)
        if found is None:
            raise KeyError(f"domain {self._domain.name!r} has no table {name!r}")
        if found.index.name != "object_id":  # fail closed: rows the view cannot restrict are not shown
            raise KeyError(f"{self._user}: table {name!r} is not keyed by object_id, so the view cannot restrict it")
        return found[found.index.isin(self._objects)]

    def clusters(self) -> pd.Series:
        """Return ``object_id`` → cluster id, for the view's objects."""
        found = self._domain.clusters()
        return found[found.index.isin(self._objects)]

    def population(self) -> pd.Index:
        """Return the view's objects that population draws draw from."""
        found = self._domain.population()
        return found[found.isin(self._objects)]
