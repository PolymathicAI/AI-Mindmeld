"""Configuration base: every run, connector, suite and figure is a frozen, validated model loaded from YAML.

Configs over flags: entry points take one config file. Configs are immutable values identified by
:meth:`Config.digest`, the sha256 of their canonical *identity* form, which run records cite.

- **Identity covers content, not names.** A config that uses another config embeds it. In YAML, write
  ``{$ref: ../connectors/<name>.yaml}`` and :func:`load_config` inlines the referenced file before validation. So
  editing any byte of a referenced file changes the digest of every config that uses it.
- **Named pins instead of copied hashes.** A config cites an upstream artifact or record as ``{$pin: <name>}``;
  :func:`load_config` replaces it with the digest the pins map gives for that name (the workspace keeps the map and
  stages add to it when they commit). The resolved digest is what the identity embeds, so pins are a convenience
  that never weakens identity. An unknown pin is an error.
- **Typed digests.** A field that cites an artifact is typed :data:`ArtifactDigest`; one that cites a record (by
  computation key) is typed :data:`RecordKey`. Both must be sha256 hex digests, and a pin naming the other kind
  (:class:`Pin` ``target``) is refused when the config loads.
- **Rationale is not identity.** Fields declared with :func:`rationale` (e.g. ``ConnectorSpec.layer_choice``) are
  kept and validated, but excluded from the digest: rewording a justification never changes a model's identity.
  Rationale fields have defaults, so a config's identity form (:meth:`Config.identity`) loads back as the config;
  artifacts store that form (e.g. a checkpoint's ``connector.json``), so their bytes, like the digest, ignore
  rewording. Whether the repository's configs state their rationale is checked by the tests.
- **Execution metadata is neither identity nor content.** A field declared with :func:`execution_metadata` says how a
  config is executed, not what it computes: a stage config's ``destination`` (:class:`StageConfig`), the pin name its
  stage run publishes under. It carries the rationale flag, so :meth:`Config.identity` and the digest leave it out, and
  it is never dumped (``model_dump``, so neither a record's ``config``, a data builder's artifact metadata nor
  :func:`dump_config` holds it); a path that writes a config file meant to keep it writes it explicitly.
- **Schema evolution keeps identities.** A field added to a model after values of it were stored (records, a
  checkpoint's ``connector.json``) is declared with :func:`added_field`: it has a default, and while it holds that
  default it is left out of the identity form. So everything written before the field existed keeps its digest (a
  record's id, ``artifacts.records``), and a value that sets the field gets a new one. Every field added to a stored
  model is declared this way; fields that were there when values were first stored stay in the identity as they are.
"""

from __future__ import annotations

import inspect
import json
import sys
import types
import typing
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

from latent_interconnect.canonical import SHA256_HEX_LENGTH, canonical_json, canonical_sha256, is_sha256

__all__ = [
    "DESTINATION",
    "PIN_KEY",
    "REF_KEY",
    "SHA256_HEX_LENGTH",
    "ArtifactDigest",
    "Config",
    "OptionError",
    "Pin",
    "PinTarget",
    "RecordKey",
    "StageConfig",
    "UnknownPinError",
    "added_field",
    "bind_options",
    "canonical_json",
    "canonical_sha256",
    "dump_config",
    "execution_metadata",
    "is_sha256",
    "load_config",
    "load_resolved",
    "load_yaml",
    "rationale",
]

REF_KEY = "$ref"
PIN_KEY = "$pin"

PinTarget = Literal["artifact", "record"]
"""What a pin or a typed digest field names: an artifact (by digest) or a record (by computation key)."""


class UnknownPinError(KeyError):
    """A config cites a pin name that the pins map does not define."""


class _Pinned(str):
    """A digest that came from a pin, carrying what the pin names until the field validator checks it."""

    pin: str
    target: PinTarget

    def __new__(cls, digest: str, pin: str, target: PinTarget) -> _Pinned:
        out = super().__new__(cls, digest)
        out.pin, out.target = pin, target
        return out


_CITED = {"artifact": "an artifact digest", "record": "a record computation key"}


def _digest_of(target: PinTarget) -> Callable[[Any], Any]:
    def check(value: Any) -> Any:
        if isinstance(value, _Pinned) and value.target != target:
            raise ValueError(f"pin {value.pin!r} is {_CITED[value.target]}, but this field takes {_CITED[target]}")
        if isinstance(value, str):
            if not is_sha256(value):
                raise ValueError(f"not a sha256 hex digest: {value!r}")
            return str(value)
        return value

    return check


ArtifactDigest = Annotated[str, BeforeValidator(_digest_of("artifact"))]
"""A field citing an artifact by its sha256 digest (a ``$pin`` to it must target an artifact)."""
RecordKey = Annotated[str, BeforeValidator(_digest_of("record"))]
"""A field citing a record by its computation key (a ``$pin`` to it must target a record)."""


_RATIONALE = "rationale"
_ADDED = "added"


def rationale(**kwargs: Any) -> Any:
    """Declare a field that documents *why* (evidence, notes) and is excluded from :meth:`Config.digest`.

    Give it a default (``rationale(default=...)``), so the identity form loads back.
    """
    return Field(json_schema_extra={_RATIONALE: True}, **kwargs)


def execution_metadata(**kwargs: Any) -> Any:
    """Declare a field of execution metadata: excluded from :meth:`Config.digest` and from every ``model_dump``.

    It carries the rationale flag (so :meth:`Config.identity` and the stored-schema snapshot leave it out) and
    ``exclude=True`` (so records, artifact metadata and :func:`dump_config` never hold it). Give it a default.
    """
    return Field(json_schema_extra={_RATIONALE: True}, exclude=True, **kwargs)


def added_field(**kwargs: Any) -> Any:
    """Declare a field added to a stored model: left out of :meth:`Config.identity` while it holds its default.

    Give it a default (``added_field(default=...)`` or ``default_factory=...``), the value every older stored value
    stands for; see the module docstring (schema evolution). The default must be empty (None, False, ``""``, ``()``,
    ``{}``): a reader at an older commit keeps a newer record's unknown provenance field in the id only when it is not
    empty (``artifacts.records``).
    """
    if "default" not in kwargs and "default_factory" not in kwargs:
        raise TypeError("an added field needs the default that older stored values stand for")
    default = kwargs["default"] if "default" in kwargs else kwargs["default_factory"]()
    if not (default is None or default is False or default in ("", (), [], {})):
        raise TypeError("an added field's default must be empty (None, False, '', () or {}): older readers assume it")
    return Field(json_schema_extra={_ADDED: True}, **kwargs)


class Config(BaseModel):
    """Base class of every configuration model: immutable, and unknown keys are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    def identity(self) -> dict[str, Any]:
        """Return the JSON form that defines identity, recursively.

        Every field is in it except rationale fields (execution metadata included, :func:`execution_metadata`) and
        added fields (:func:`added_field`) that hold their default.
        """
        out = {}
        for name, field in type(self).model_fields.items():
            extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
            value = getattr(self, name)
            if extra.get(_RATIONALE) or (extra.get(_ADDED) and value == field.get_default(call_default_factory=True)):
                continue
            out[name] = _identity(value)
        return out

    def canonical(self) -> str:
        """Return the canonical JSON (``latent_interconnect.canonical``) of :meth:`identity`.

        Defaults are included, except an added field's (:func:`added_field`) while it holds its default.
        """
        return canonical_json(self.identity())

    def digest(self) -> str:
        """Return the sha256 of the canonical form: the identity of this configuration in records."""
        return canonical_sha256(self.identity())


DESTINATION = r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$"
"""A destination: a pin name of ``/``-separated parts, without ``@`` (``--pin D`` makes ``D@record``)."""


class StageConfig(Config):
    """Base class of every stage's config (``latent_interconnect.run._main.Stage.config``).

    Attributes:
        destination: the name its stage run is pinned under (``--pin``, which must equal it unless the run's
            ``--destination`` overrides it; the stage names the outputs, ``latent_interconnect.config_index``), or None
            if the config declares none. Execution metadata
            (:func:`execution_metadata`): where the outputs are published, never part of the identity or of a dump.
    """

    destination: str | None = execution_metadata(default=None, pattern=DESTINATION)

    @classmethod
    def fits(cls, keys: Collection[str]) -> bool:
        """Return whether a config file with these top-level keys is one of this class's (``config_index.stage_of``).

        It is when the class has a field for every key and finds each required field among them. A class that also
        reads another file shape (a before-validator) says so by overriding this.
        """
        required = {name for name, field in cls.model_fields.items() if field.is_required()}
        return set(keys) <= set(cls.model_fields) and required <= set(keys)


def _identity(value: Any) -> Any:
    if isinstance(value, Config):
        return value.identity()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _identity(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_identity(v) for v in value]
    return json.loads(json.dumps(value, default=str))


C = TypeVar("C", bound=Config)


class Pin(Config):
    """One named pin: what a ``{$pin: <name>}`` resolves to.

    Attributes:
        target: ``artifact`` (``sha256`` is an artifact digest) or ``record`` (a record computation key).
        sha256: the sha256 hex digest.
    """

    target: PinTarget
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _hex64(cls, v: str) -> str:
        if not is_sha256(v):
            raise ValueError(f"not a sha256 hex digest: {v!r}")
        return v


def load_config(path: Path, model: type[C], pins: Mapping[str, Pin] | None = None) -> C:
    """Load a YAML config, inline every ``{$ref: <relative path>}``, resolve every ``{$pin: <name>}``, validate.

    Args:
        path: the YAML file.
        model: the config class.
        pins: pin name → pin (``Workspace.pins``); required only if the config uses pins.

    Raises:
        pydantic.ValidationError: if the result does not satisfy the schema (unknown keys included, and a pin whose
            target is not the kind its field cites).
        UnknownPinError: for a pin name not in ``pins``.
        ValueError: on a reference cycle.
        yaml.YAMLError: if a mapping repeats a key (:func:`load_yaml`).
    """
    return model.model_validate(load_resolved(path, pins))


_SafeLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
"""libyaml's parser where PyYAML was built with it (about 9× faster on the config tree), else the pure-Python one; the
two construct the same values."""


class _UniqueKeyLoader(_SafeLoader):  # type: ignore[misc,valid-type]  # PyYAML ships no type stubs
    """The safe loader, refusing a mapping key given twice (the safe loader silently keeps the last)."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue  # a ``<<`` merge key: explicit keys override merged ones by design
            key = self.construct_object(key_node, deep=True)
            if isinstance(key, typing.Hashable):
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping",
                        node.start_mark,
                        f"found duplicate key {key!r}",
                        key_node.start_mark,
                    )
                seen.add(key)
        mapping: dict[Any, Any] = super().construct_mapping(node, deep=deep)
        return mapping


def load_yaml(stream: str | typing.IO[str]) -> Any:
    """Parse YAML as ``yaml.safe_load`` does, but refuse a mapping that repeats a key (fail closed).

    ``yaml.safe_load`` keeps the last of two equal keys, so a repeated cell, task or pin name would silently replace
    the first. Every YAML file the package reads (configs, pins, site tables) goes through this function.

    Raises:
        yaml.YAMLError: on a repeated key (``ConstructorError``, naming the key and its line) or invalid YAML.
    """
    return yaml.load(stream, Loader=_UniqueKeyLoader)


def load_resolved(
    path: Path, pins: Mapping[str, Pin] | None = None, *, unknown_pins: Literal["error", "keep"] = "error"
) -> Any:
    """Load a YAML config with every ``{$ref: ...}`` inlined and every known ``{$pin: ...}`` resolved, unvalidated.

    ``load_config`` validates this. ``unknown_pins="keep"`` leaves a pin not in ``pins`` as its ``{$pin: name}`` node,
    for reading a config's structure without a workspace's pins (e.g. which files are run configs).

    Raises:
        UnknownPinError: for a pin name not in ``pins`` (``unknown_pins="error"``).
        ValueError: on a reference cycle.
        yaml.YAMLError: if a mapping repeats a key (:func:`load_yaml`).
    """
    return _load_resolved(Path(path), (), pins or {}, unknown_pins == "keep")


def _load_resolved(path: Path, seen: tuple[Path, ...], pins: Mapping[str, Pin], keep: bool = False) -> Any:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"$ref cycle: {' -> '.join(str(p) for p in (*seen, path))}")
    with path.open() as fh:
        data = load_yaml(fh)
    return _resolve(data, path.parent, (*seen, path), pins, keep)


def _resolve(node: Any, base: Path, seen: tuple[Path, ...], pins: Mapping[str, Pin], keep: bool) -> Any:
    if isinstance(node, dict):
        if set(node) == {REF_KEY}:
            return _load_resolved(base / str(node[REF_KEY]), seen, pins, keep)
        if set(node) == {PIN_KEY}:
            name = str(node[PIN_KEY])
            if name not in pins:
                if keep:
                    return dict(node)
                raise UnknownPinError(f"unknown pin {name!r} in {seen[-1]}")
            return _Pinned(pins[name].sha256, name, pins[name].target)
        return {k: _resolve(v, base, seen, pins, keep) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, base, seen, pins, keep) for v in node]
    return node


def dump_config(cfg: Config, path: Path) -> None:
    """Write ``cfg`` as YAML (keys in schema order, references inlined; execution metadata left out).

    A caller writing a config file that must keep its execution metadata (a stage config's ``destination``) writes
    that field explicitly.
    """
    Path(path).write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))


class OptionError(ValueError):
    """A command-line option names no settable parameter, a refused one, or a value its parameter does not accept."""


_CLI_ATOMS: tuple[type, ...] = (str, int, float, bool, Path, type(None))
"""The types a command-line text can express; a parameter of any other type (a ``torch.dtype``, a module, a tokenizer)
is never an option."""


def bind_options(
    target: Callable[..., Any], raw: Mapping[str, str], refused: Mapping[str, str] | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """Bind command-line option texts to ``target``'s parameters: the one rule for execution options.

    An option sets a parameter of ``target`` (a stage function, or a backend class) that has a default; parameters
    without one (the config, the workspace, the receiver spec) are the caller's. The text is validated against the
    parameter's annotation (pydantic, lax: ``"4096"`` → ``4096``, ``"/nvme/kimi"`` → ``Path``; ``"4e9"`` is no
    ``int``). An annotation must be built from ``str``, ``int``, ``float``, ``bool``, ``Path``, ``None`` and
    ``Literal`` only, and resolve at run time; only the parameters actually given are resolved, so a parameter
    annotated with a ``TYPE_CHECKING``-only import is fine until someone sets it from a command line. A text is never
    empty (omit the option to keep the default). Names ``target`` does not have go to its ``**kwargs``, untouched,
    when it has them (a stage hands them to its backend, which binds them in turn).

    Args:
        target: the callable whose parameters the options set (a class: its constructor).
        raw: option name → its text, as given.
        refused: names that are never options, with the reason (identity: a config field, a backend parameter that
            is part of a result's identity).

    Returns:
        The typed values of ``target``'s own parameters, and the remaining texts for its ``**kwargs``.

    Raises:
        OptionError: for a refused name, an empty text, a parameter without a default or an annotation, an annotation
            that does not resolve or that no text expresses, a value its type refuses, or an unknown name where
            ``target`` has no ``**kwargs``.
    """
    if not raw:
        return {}, {}
    params = inspect.signature(target).parameters  # a class: its constructor's
    open_ended = any(p.kind is p.VAR_KEYWORD for p in params.values())
    bound: dict[str, Any] = {}
    rest: dict[str, str] = {}
    for name, text in raw.items():
        if refused and name in refused:
            raise OptionError(f"option {name!r} is refused: {refused[name]}")
        if not text:
            raise OptionError(f"option {name!r} is empty: omit it to keep the default")
        param = params.get(name)
        if param is None or param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            if not open_ended:
                settable = sorted(n for n, p in params.items() if p.default is not p.empty)
                raise OptionError(f"option {name!r}: {_name(target)} takes {settable or 'no options'}")
            rest[name] = text
            continue
        if param.default is param.empty:
            raise OptionError(f"option {name!r}: a required argument of {_name(target)}, not an option")
        if param.annotation is param.empty:
            raise OptionError(f"option {name!r}: {_name(target)} does not annotate it, so its value cannot be checked")
        annotation = _resolved(target, name, param.annotation)
        if not _expressible(annotation):
            raise OptionError(f"option {name!r}: its type {annotation} cannot be given as text")
        try:
            bound[name] = TypeAdapter(annotation).validate_python(text)
        except ValidationError as err:
            raise OptionError(f"option {name}={text!r}: not a {annotation} ({err.errors()[0]['msg']})") from None
    return bound, rest


def _resolved(target: Callable[..., Any], name: str, annotation: Any) -> Any:
    """Return one parameter's annotation, evaluated in ``target``'s module if it is a string (postponed evaluation)."""
    if not isinstance(annotation, str):
        return annotation
    holder = type("_Option", (), {"__annotations__": {name: annotation}})
    module = sys.modules.get(getattr(target, "__module__", ""), None)
    try:
        return typing.get_type_hints(holder, globalns=vars(module) if module else None)[name]
    except NameError as err:
        raise OptionError(
            f"option {name!r}: its annotation {annotation!r} does not resolve at run time ({err})"
        ) from None


def _expressible(annotation: Any) -> bool:
    if annotation in _CLI_ATOMS:
        return True
    origin = typing.get_origin(annotation)
    if origin is Literal:
        return all(isinstance(v, str | int | bool) for v in typing.get_args(annotation))
    if origin in (typing.Union, types.UnionType):
        return all(_expressible(a) for a in typing.get_args(annotation))
    return False


def _name(target: Callable[..., Any]) -> str:
    return getattr(target, "__qualname__", repr(target))
