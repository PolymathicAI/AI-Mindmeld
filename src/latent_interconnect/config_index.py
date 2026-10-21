"""The declared producers of pins: which config makes a pin, read from the configs' own declarations.

A stage config declares the name its stage run is pinned under (``config.StageConfig.destination``), and its stage
names its outputs (``run._main.Stage.outputs``), so a run with ``--pin D`` makes ``D@record``, ``D/<output>`` per output
and ``D`` when it has one output (``run._main.stage_pins``). :func:`index` reads the destinations of the configs under
explicit roots, never the store, and :meth:`PinIndex.resolve` names the config that makes a pin, by the first rule that
applies:

1. ``D@record`` is made by the config declaring ``D``;
2. a pin equal to a destination ``D`` is made by that config;
3. ``D/<rest>`` is made by the config declaring ``D`` when its stage's output naming accepts ``<rest>``.

A pin that more than one producer makes (two output claims, or a destination that is also another producer's output) is
an error, and :func:`index` refuses a destination of that kind when it builds.

There is no prediction of which outputs exist: whether a producer has made a pin is the site's check, from the store.
A config's stage is the one whose config class takes a file of its top-level keys (:func:`stage_of`). Excluded roots
(preserved fixtures, generated sets) are not indexed, and an indexed config may not ``$ref`` a file under one.
"""

from __future__ import annotations

import hashlib
import importlib
import pkgutil
import re
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import latent_interconnect.run as run_pkg
from latent_interconnect.config import DESTINATION, PIN_KEY, REF_KEY, Pin, load_resolved, load_yaml
from latent_interconnect.run._main import Stage

__all__ = [
    "STAND_IN",
    "AmbiguousPinError",
    "DestinationCollisionError",
    "ExcludedCitationError",
    "InvalidDestinationError",
    "PinIndex",
    "Producer",
    "UnknownStageError",
    "cited_pins",
    "declared",
    "destination",
    "index",
    "nodes",
    "stage_of",
    "stages",
    "stand_in_pins",
    "tree_pins",
]

RECORD = "@record"
"""The suffix of a record pin (``run._main.stage_pins``)."""
STAND_IN = "f" * 16
"""A stand-in pin's digest starts with this, then its name's own hash (:func:`stand_in_pins`)."""


class DestinationCollisionError(ValueError):
    """Two indexed configs declare the same destination."""


class AmbiguousPinError(ValueError):
    """More than one declared producer accepts a pin."""


class ExcludedCitationError(ValueError):
    """An indexed config ``$ref``s a file under an excluded root (a preserved fixture or a generated set)."""


class InvalidDestinationError(ValueError):
    """A config file's ``destination`` that ``StageConfig`` would refuse (checked before indexing or comparing)."""


class UnknownStageError(ValueError):
    """A config declares a destination, but no one stage's config class fits its keys."""


@dataclass(frozen=True)
class Producer:
    """The config that declares one destination.

    Attributes:
        destination: the declared destination (``StageConfig.destination``).
        path: the config file.
        stage: its stage (:func:`stage_of`).
    """

    destination: str
    path: Path
    stage: Stage

    def makes(self, rest: str) -> bool:
        """Return whether ``<destination>/<rest>`` is an output name of this producer's stage (``Stage.outputs``)."""
        return re.fullmatch(self.stage.outputs, rest) is not None


@dataclass(frozen=True)
class PinIndex:
    """Destination → its declared producer (:func:`index`).

    Attributes:
        producers: destination → the config that declares it.
    """

    producers: Mapping[str, Producer]

    def resolve(self, pin: str) -> Producer | None:
        """Return the declared producer of ``pin`` (module docstring), or None if no indexed config makes it.

        Raises:
            AmbiguousPinError: if several producers make ``pin``: output claims, or a destination and another
                producer's output.
        """
        base = pin.removesuffix(RECORD)
        if base != pin:
            return self.producers.get(base)
        found = [self.producers[pin]] if pin in self.producers else []
        found += [
            p
            for i in range(1, len(pin))
            if pin[i] == "/" and (p := self.producers.get(pin[:i])) is not None and p.makes(pin[i + 1 :])
        ]
        if len(found) > 1:
            raise AmbiguousPinError(f"pin {pin!r} is made by {[str(p.path) for p in found]}")
        return found[0] if found else None

    def citing(self, stage: str, key: str, target: Path) -> list[Producer]:
        """Return the producers of ``stage`` whose top-level ``key`` ``$ref``s ``target`` (an init: its connector)."""
        found = []
        for p in self.producers.values():
            if p.stage.name == stage:
                node = load_yaml(p.path.read_text()).get(key)
                if isinstance(node, dict) and REF_KEY in node and (p.path.parent / node[REF_KEY]).resolve() == target:
                    found.append(p)
        return found


def destination(path: Path) -> str | None:
    """Return the destination a config file declares (its top-level ``destination``, as its stage loads it), or None.

    Raises:
        InvalidDestinationError: as :func:`declared`.
    """
    return declared(_top(Path(path)), Path(path))


def declared(tree: Any, path: Path) -> str | None:
    """Return the destination ``tree`` (the content of ``path``) declares at its top level, or None.

    Raises:
        InvalidDestinationError: for a value that is not a destination by ``StageConfig``'s own rule (a string of
            ``/``-separated name parts, no ``@``), checked as written, never coerced.
    """
    value = tree.get("destination") if isinstance(tree, dict) else None
    if value is not None and not (isinstance(value, str) and re.fullmatch(DESTINATION, value)):
        raise InvalidDestinationError(f"{path}: destination {value!r} is not a destination (pattern {DESTINATION})")
    return value


def stages() -> dict[str, Stage]:
    """Return every stage (``latent_interconnect.run.<module>.STAGE``), by name."""
    found = {}
    for module in sorted(m.name for m in pkgutil.iter_modules(run_pkg.__path__)):
        stage = getattr(importlib.import_module(f"latent_interconnect.run.{module}"), "STAGE", None)
        if isinstance(stage, Stage):
            found[stage.name] = stage
    return found


def stage_of(tree: Mapping[str, Any], known: Mapping[str, Stage]) -> Stage:
    """Return the one stage whose config class takes a file with ``tree``'s top-level keys (``StageConfig.fits``).

    Raises:
        UnknownStageError: if no stage, or more than one, fits.
    """
    fits = [s for s in known.values() if s.config.fits(set(tree))]
    if len(fits) != 1:
        raise UnknownStageError(f"keys {sorted(tree)} fit {len(fits)} stages ({[s.name for s in fits]}), not one")
    return fits[0]


def index(roots: Sequence[Path], exclude: Collection[Path] = ()) -> PinIndex:
    """Index the destinations declared by the ``*.yaml`` files under ``roots``, outside the ``exclude`` roots.

    Raises:
        DestinationCollisionError: if two indexed configs declare the same destination.
        AmbiguousPinError: if a destination is also an output name of another producer (``D/<output>``).
        InvalidDestinationError: for a declared destination that is not one (:func:`declared`).
        ExcludedCitationError: if an indexed config ``$ref``s a file under an excluded root.
        UnknownStageError: if a config declaring a destination fits no one stage (:func:`stage_of`).
    """
    excluded = tuple(Path(p).resolve() for p in exclude)
    known = stages()
    producers: dict[str, Producer] = {}
    closures: dict[Path, frozenset[Path]] = {}
    for root in roots:
        for path in sorted(Path(root).resolve().rglob("*.yaml")):
            if any(path.is_relative_to(e) for e in excluded):
                continue
            for cited in sorted(_closure(path, closures)):
                if any(cited.is_relative_to(e) for e in excluded):
                    raise ExcludedCitationError(f"{path} $refs {cited}, under an excluded root")
            tree = _top(path)
            name = declared(tree, path)
            if name is None:
                continue
            if name in producers:
                raise DestinationCollisionError(
                    f"destination {name!r} is declared by {producers[name].path} and {path}"
                )
            producers[name] = Producer(name, path, stage_of(tree, known))
    found = PinIndex(producers)
    for name in producers:  # a destination another producer also publishes as an output is refused now
        found.resolve(name)
    return found


def _closure(path: Path, cache: dict[Path, frozenset[Path]]) -> frozenset[Path]:
    """Every file ``path`` ``$ref``s, through the referenced files too (each file's closure computed once)."""
    if path not in cache:
        cache[path] = frozenset()  # a cycle (refused when a config loads) ends here
        direct = {
            (path.parent / str(node[REF_KEY])).resolve()
            for node in nodes(load_yaml(path.read_text()))
            if isinstance(node, dict) and set(node) == {REF_KEY}
        }
        cache[path] = frozenset(direct).union(*(_closure(t, cache) for t in direct if t.is_file()))
    return cache[path]


def _top(path: Path) -> Any:
    """A config file's content as its stage sees its top level (a whole-file ``$ref`` followed, ``load_resolved``)."""
    tree = load_yaml(path.read_text())
    if isinstance(tree, dict) and set(tree) == {REF_KEY}:
        return load_resolved(path, unknown_pins="keep")
    return tree


def cited_pins(path: Path) -> list[str]:
    """Return every pin name a config file cites, through the files it ``$ref``s, once each in file order.

    A structured walk (:func:`tree_pins` of the parsed file); comments and other text are not read.
    """
    path = Path(path).resolve()
    return tree_pins(load_yaml(path.read_text()), path)


def tree_pins(tree: Any, path: Path) -> list[str]:
    """Return every pin name ``tree`` cites, once each in the order ``load_resolved`` would inline them.

    ``tree`` is the content of the file ``path``, as parsed or as rewritten in memory: each ``{$ref: ...}`` node is
    followed relative to ``path``'s folder, as ``load_resolved`` follows it (a missing file or a cycle raises), and a
    resolved tree has none. Comments and other text are not read.
    """
    out: dict[str, None] = {}

    def walk(node: Any, here: Path, seen: tuple[Path, ...]) -> None:
        for n in nodes(node):
            if isinstance(n, dict) and set(n) == {PIN_KEY}:
                out[str(n[PIN_KEY])] = None
            elif isinstance(n, dict) and set(n) == {REF_KEY}:
                target = (here.parent / str(n[REF_KEY])).resolve()
                if target in seen:
                    raise ValueError(f"$ref cycle: {' -> '.join(str(p) for p in (*seen, target))}")
                walk(load_yaml(target.read_text()), target, (*seen, target))

    path = Path(path).resolve()
    walk(tree, path, (path,))
    return list(out)


def stand_in_pins(names: Iterable[str]) -> dict[str, Pin]:
    """Return a stand-in pin for each of ``names``, to load a config's schema without the store's pins.

    Finite: a name not given stays unknown to ``load_config``. ``NAME@record`` stands in as a record, any other name
    as an artifact; the digest is :data:`STAND_IN` then the name's own hash, so a name stands in alike in every config
    and a stand-in is never a produced pin's digest (real execution resolves the store's pins only).
    """
    return {
        n: Pin(
            target="record" if n.endswith(RECORD) else "artifact",
            sha256=STAND_IN + hashlib.sha256(n.encode()).hexdigest()[: 64 - len(STAND_IN)],
        )
        for n in names
    }


def nodes(tree: Any) -> Iterator[Any]:
    """Every node of a parsed config tree, the tree first, depth first."""
    yield tree
    children = tree.values() if isinstance(tree, dict) else tree if isinstance(tree, list) else ()
    for child in children:
        yield from nodes(child)
