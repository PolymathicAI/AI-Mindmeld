"""Named registries: the one extension mechanism of the package.

Every open set (map kinds, receivers and their backends, encoders, domains, example streams, suite and analysis
kinds) is a :class:`Registry`. An extension implements the relevant interface and registers it under a name; configs
refer to it by that name. Third-party code is loaded by listing its modules under ``plugins:`` in a config, which
:func:`load_plugins` imports. Core code never imports a plugin directly.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Iterator
from typing import Generic, TypeVar

__all__ = ["DuplicateRegistrationError", "Registry", "UnknownNameError", "load_plugins"]

T = TypeVar("T")


class UnknownNameError(KeyError):
    """A config named something that is not registered."""


class DuplicateRegistrationError(ValueError):
    """Two extensions tried to register under the same name."""


class Registry(Generic[T]):
    """A mapping from names to registered objects (usually classes) of one kind.

    Args:
        kind: what the registry holds, used in error messages (e.g. ``"map kind"``).
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Return a decorator that registers its argument under ``name``.

        Raises:
            DuplicateRegistrationError: if ``name`` is already taken.
        """

        def decorator(obj: T) -> T:
            self.add(name, obj)
            return obj

        return decorator

    def add(self, name: str, obj: T) -> None:
        """Register ``obj`` under ``name`` (the non-decorator form)."""
        if name in self._items:
            raise DuplicateRegistrationError(f"{self._kind} {name!r} is already registered")
        self._items[name] = obj

    def get(self, name: str) -> T:
        """Return the object registered under ``name``.

        Raises:
            UnknownNameError: with the list of known names.
        """
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "none"
            raise UnknownNameError(f"unknown {self._kind} {name!r} (registered: {known})") from None

    def names(self) -> list[str]:
        """Return the registered names, sorted."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


def load_plugins(modules: Iterable[str]) -> None:
    """Import each module so that its registrations run (idempotent: Python caches imports)."""
    for module in modules:
        importlib.import_module(module)
