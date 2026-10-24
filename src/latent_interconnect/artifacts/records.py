"""Result records: what was computed, from which inputs, by which code, with which outputs.

A :class:`Record` ties a computation (its kind, its config digest and its input artifacts) to the output artifacts of
one execution and that execution's provenance. Figures, tables and the paper's numbers read records, never paths.

- ``computation_key``: sha256 of (kind, purpose, config digest, input digests). Two executions of one computation
  share it.
- ``purpose``: the record's lineage class (:data:`Purpose`). ``result`` records are eligible as results: built by the
  stages from stage-made inputs. ``anchor`` records (built on an input no stage produced, of unknown lineage) and
  ``monitor`` records (in-job evaluation on the training hardware) are not: the input check
  (``Workspace.input_problems``) refuses them, and a record built on them inherits their class
  (``Workspace.purpose``: anchor > monitor > result). Neither ``result`` nor ``anchor`` says whether anything was
  computed, nor that a result is scientifically accepted.
- **Stored purpose values.** Records written before these names hold ``paper`` (now ``result``) or ``in_job`` (now
  ``monitor``), and a record written before the field existed omits it, which decodes as ``paper``
  (:data:`OMITTED_PURPOSE`). A stored value is part of the record's identity (``computation_key``, ``record_id``), so
  it is read and verified exactly as written, never rewritten; :func:`purpose_class` classifies it afterwards
  through :data:`LEGACY_PURPOSE`. New records always write a current value (:meth:`RecordStore.put` refuses a legacy
  one), and an unknown value fails validation.
- ``record_id``: sha256 of the whole record, provenance included; unique per execution. It is the record's identity
  form (``Config.identity``), so a field added to ``Record`` or ``Provenance`` after records were written is declared
  with ``config.added_field`` and left out while at its default: a stored record keeps its id under newer code (the
  schema-evolution rule; a golden record written by older code is tested).
- Reading tolerates newer provenance: a stored record whose provenance (or a resumed segment, or its extension
  source) has a field this code does not know, written by newer code, still loads, and the field is kept, so its id
  verifies (an added field holds an empty default, ``config.added_field``, which the id leaves out). An unknown field
  anywhere else is refused, and writing (:meth:`RecordStore.put`) stays strict. So a long job at an older commit can
  finish after newer code wrote records.

The normal case is one execution per computation, which is canonical without any declaration. If a computation was
executed more than once, :meth:`RecordStore.declare_canonical` names the one to use, with a reason; it may be
re-declared, and the file keeps a one-line history.

**The first execution stays canonical.** A computation may be executed again, on the same or on other
hardware (an earlier one-hardware refusal is lifted; every record keeps the devices it actually ran on, and a resumed
execution keeps each segment's). When :meth:`RecordStore.put` writes the second execution of a computation that had
exactly one and no declaration, it declares that first one canonical (reason :data:`FIRST_EXECUTION`), so what the
computation's pins and readers resolve to never changes implicitly. Only a declared re-pin (``--repin REASON``,
``run._main.adopt``) moves it. A computation that already had several executions and no declaration (written before
this rule) stays ambiguous until one is declared.

On disk (:class:`RecordStore`):

- ``<root>/<computation_key>/<record_id>.json``: one execution, written once and read-only. The writer prepares the
  file (synced) in a private hidden directory. For a new computation it renames that directory to the computation's
  name, so a computation directory never exists without a record; otherwise it hard-links the file in, which fails if
  the name exists. So readers never see a partial record, nothing is overwritten, and concurrent identical writes all
  succeed (same content, same name).
- ``<root>/<computation_key>/CANONICAL.jsonl``: the declarations, one :class:`CanonicalChoice` per line, each appended
  in one ``O_APPEND`` write; the last line is in force.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import ConfigDict, Field, ValidationError

from latent_interconnect.artifacts.store import ArtifactRef, IntegrityError
from latent_interconnect.config import (
    ArtifactDigest,
    Config,
    RecordKey,
    added_field,
    canonical_json,
    canonical_sha256,
    is_sha256,
)
from latent_interconnect.provenance import Extension, Provenance, Segment

__all__ = [
    "CANONICAL_NAME",
    "FIRST_EXECUTION",
    "LEGACY_PURPOSE",
    "OMITTED_PURPOSE",
    "PURPOSE_ORDER",
    "CanonicalChoice",
    "Purpose",
    "Record",
    "RecordRef",
    "RecordStore",
    "ResolvedRecords",
    "StoredPurpose",
    "computation_key_of",
    "drawn_from",
    "purpose_class",
    "record_output",
    "strictest_purpose",
]

CANONICAL_NAME = "CANONICAL.jsonl"
_RECORD_FILE = re.compile(r"^[0-9a-f]{64}\.json$")

Purpose = Literal["result", "monitor", "anchor"]
"""A record's lineage class, the value new records write: ``result`` (made by the stages from stage-made inputs),
``monitor`` (in-job evaluation) or ``anchor`` (unknown lineage). Only ``result`` records are eligible as results."""

StoredPurpose = Literal["result", "monitor", "anchor", "paper", "in_job"]
"""A purpose value as a stored record holds it: a current one or a legacy one (:data:`LEGACY_PURPOSE`)."""

LEGACY_PURPOSE: Mapping[str, Purpose] = MappingProxyType({"paper": "result", "in_job": "monitor", "anchor": "anchor"})
"""The class of each purpose value written before the current names (``anchor`` kept its name and meaning)."""

OMITTED_PURPOSE = "paper"
"""What a stored record without a ``purpose`` field holds: the default it was written under, part of its identity."""

PURPOSE_ORDER: tuple[Purpose, ...] = ("anchor", "monitor", "result")
"""The classes, strictest first: a record built on several inputs takes the strictest of theirs."""


def purpose_class(value: str) -> Purpose:
    """Return the lineage class of a purpose value, current or legacy (:data:`LEGACY_PURPOSE`).

    Raises:
        ValueError: if the value is neither (it never counts as ``result``).
    """
    if value in PURPOSE_ORDER:
        return value  # type: ignore[return-value]
    if value in LEGACY_PURPOSE:
        return LEGACY_PURPOSE[value]
    raise ValueError(f"unknown record purpose {value!r} (known: {sorted({*PURPOSE_ORDER, *LEGACY_PURPOSE})})")


def strictest_purpose(values: Iterable[str]) -> Purpose:
    """Return the strictest class of the purpose values (anchor > monitor > result; legacy values classified first).

    Raises:
        ValueError: if a value is unknown, or there is none.
    """
    found = {purpose_class(v) for v in values}
    if not found:
        raise ValueError("no purpose to classify")
    return next(p for p in PURPOSE_ORDER if p in found)


class RecordRef(Config):
    """A JSON-serializable :class:`~latent_interconnect.artifacts.store.ArtifactRef` for use inside records.

    Attributes:
        sha256: the artifact digest.
        kind: the artifact kind.
    """

    sha256: ArtifactDigest
    kind: str


class Record(Config):
    """One execution of one computation.

    Attributes:
        kind: the stage that ran (``latent_interconnect.run`` lists them with their output names, e.g. ``"extract"``,
            ``"init"``, ``"suite"``, ``"analysis"``).
        purpose: the stored purpose value (:data:`StoredPurpose`; part of the computation key), exactly as written:
            ``result``, ``monitor`` or ``anchor``, or a legacy ``paper`` or ``in_job``. Omitted, it is
            :data:`OMITTED_PURPOSE`, the default older records were written under. :meth:`purpose_class` classifies
            it.
        config_sha: :meth:`Config.digest` of the config that defined the computation.
        config: the config itself, as JSON, so a record is self-describing.
        inputs: named input artifacts.
        input_records: named input records (computation keys), e.g. an analysis's score records; lineage only, since
            the config that cites them is already part of the identity.
        outputs: named output artifacts.
        provenance: code state, host, job and environment of this execution.
    """

    kind: str
    purpose: StoredPurpose = OMITTED_PURPOSE  # the decoding default of older records; new writers write a current value
    config_sha: str
    config: dict[str, object]
    inputs: dict[str, RecordRef] = Field(default_factory=dict)
    input_records: dict[str, RecordKey] = Field(default_factory=dict)
    outputs: dict[str, RecordRef] = Field(default_factory=dict)
    provenance: Provenance

    def computation_key(self) -> str:
        """Return the identity shared by every execution of this computation (:func:`computation_key_of`)."""
        inputs = {k: v.sha256 for k, v in self.inputs.items()}
        return computation_key_of(self.kind, self.purpose, self.config_sha, inputs)

    def record_id(self) -> str:
        """Return the identity of this execution."""
        return self.digest()

    def purpose_class(self) -> Purpose:
        """Return this record's lineage class: its stored purpose classified (:func:`purpose_class`)."""
        return purpose_class(self.purpose)


def computation_key_of(kind: str, purpose: str, config_sha: str, inputs: Mapping[str, str]) -> str:
    """Return the computation key of a record from its parts: :meth:`Record.computation_key`'s one rule.

    ``inputs`` maps each input name to its artifact digest. A stage finds a record by key with it, without a record.
    """
    return canonical_sha256({"kind": kind, "purpose": purpose, "config_sha": config_sha, "inputs": dict(inputs)})


def _with_newer(identity: dict[str, object], extra: dict[str, object] | None) -> dict[str, object]:
    """Add the non-empty fields newer code wrote (an added field at its empty default is not part of the id)."""
    return identity | {k: v for k, v in (extra or {}).items() if not (v is None or v is False or v in ("", [], {}))}


class _StoredSegment(Segment):
    model_config = ConfigDict(extra="allow")

    def identity(self) -> dict[str, object]:
        return _with_newer(super().identity(), self.model_extra)


class _StoredExtension(Extension):
    model_config = ConfigDict(extra="allow")

    def identity(self) -> dict[str, object]:
        return _with_newer(super().identity(), self.model_extra)


class _StoredProvenance(Provenance):
    model_config = ConfigDict(extra="allow")
    resumed: tuple[_StoredSegment, ...] = added_field(default=())
    extended_from: _StoredExtension | None = added_field(default=None)

    def identity(self) -> dict[str, object]:
        return _with_newer(super().identity(), self.model_extra)


class _StoredRecord(Record):
    """A record as read back: unknown provenance fields (newer code) are kept; other unknown fields are refused."""

    provenance: _StoredProvenance


class CanonicalChoice(Config):
    """Names which execution of a multiply-executed computation to use, and why.

    Attributes:
        computation_key: the computation.
        record_id: the chosen execution.
        reason: why this one.
    """

    computation_key: RecordKey
    record_id: str
    reason: str


FIRST_EXECUTION = "first execution stays canonical: declared when a second execution was recorded"
"""The reason :meth:`RecordStore.put` declares a computation's first execution canonical with (module docstring)."""


class RecordStore:
    """Records on disk: ``<root>/<computation_key>/<record_id>.json`` (one small file per execution).

    :meth:`put` refuses a record whose provenance is dirty or has no end time (a result from uncommitted code is not a
    result), so an entry-point run needs a committed tree (or an installed package, ``no-vcs``), and a record whose
    purpose is a legacy value (``paper``, ``in_job``: :data:`LEGACY_PURPOSE`; stored records keep theirs, new ones
    write a current one). It keeps the first execution of a computation canonical when a second is written (module
    docstring). Tests and fakes use a real store on a temporary directory; there is no separate in-memory store.

    Reads by computation key list one directory. :meth:`get` and :meth:`find` scan every computation directory:
    records are small and far fewer than artifact files. :meth:`produced_by` parses only the files whose text holds the
    digest it looks for, so a record it does not need cannot fail it.

    Args:
        root: the store root (``Paths.records``).
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def put(self, record: Record) -> str:
        """Write a record (idempotent and race-safe) and return its ``record_id``.

        Raises:
            DirtyTreeError: if the provenance is dirty.
            ValueError: if the provenance has no end time, the purpose is a legacy value, or the record does not
                survive a JSON round trip.
            IntegrityError: if the file already stored under this id holds a different record.
        """
        prov = record.provenance
        prov.require_clean()
        if prov.ended is None:
            raise ValueError(f"the {record.kind!r} record's provenance has no end time: call Provenance.finish()")
        if record.purpose not in PURPOSE_ORDER:
            raise ValueError(
                f"the {record.kind!r} record's purpose {record.purpose!r} is a legacy value: a new record writes "
                f"one of {list(PURPOSE_ORDER)} ({record.purpose!r} is {purpose_class(record.purpose)!r})"
            )
        record_id, key = record.record_id(), record.computation_key()
        text = canonical_json(record.model_dump(mode="json"))
        if Record.model_validate_json(text).record_id() != record_id:
            raise ValueError(f"the {record.kind!r} record does not survive JSON (e.g. a NaN or infinity in its config)")
        earlier = [rid for rid in self._ids(key) if rid != record_id]
        if not _write_once(self.root, key, f"{record_id}.json", text):
            _load_record(self._path(key, record_id), record_id, key)
        elif len(earlier) == 1 and not self._declarations(key):
            # the computation's first execution was canonical by being the only one: it stays so
            self.declare_canonical(CanonicalChoice(computation_key=key, record_id=earlier[0], reason=FIRST_EXECUTION))
        return record_id

    def declare_canonical(self, choice: CanonicalChoice) -> None:
        """Declare (or re-declare) a computation's canonical execution; the previous choice stays as a history line.

        Raises:
            ValueError: if the record does not exist (as an execution of ``choice.computation_key``).
            IntegrityError: if the named record's file does not hold that record.
        """
        key, rid = choice.computation_key, choice.record_id
        if not (is_sha256(rid) and self._path(key, rid).is_file()):
            raise ValueError(f"no record {rid!r} of computation {key}")
        _load_record(self._path(key, rid), rid, key)
        line = (canonical_json(choice.model_dump(mode="json")) + "\n").encode()
        fd = os.open(self.root / key / CANONICAL_NAME, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            written = os.write(fd, line)
        finally:
            os.close(fd)
        if written != len(line):
            raise OSError(f"short write ({written} of {len(line)} bytes) to {self.root / key / CANONICAL_NAME}")

    def _declarations(self, computation_key: str) -> list[CanonicalChoice]:
        """Return the canonical declarations, oldest first (the last is in force); IntegrityError on a bad line."""
        path = self.root / computation_key / CANONICAL_NAME
        if not (is_sha256(computation_key) and path.is_file()):
            return []
        choices = []
        for n, line in enumerate(path.read_text().splitlines(), 1):
            try:
                choice = CanonicalChoice.model_validate_json(line)
            except ValueError as e:
                raise IntegrityError(f"{path}:{n} is not a canonical declaration: {e}") from e
            if choice.computation_key != computation_key:
                raise IntegrityError(f"{path}:{n} declares computation {choice.computation_key}")
            choices.append(choice)
        return choices

    def get(self, record_id: str) -> Record:
        """Load one record by id.

        Raises:
            ValueError: if ``record_id`` is not a sha256 hex digest.
            LookupError: if no record has this id.
            IntegrityError: if the stored file does not hold that record.
        """
        if not is_sha256(record_id):
            raise ValueError(f"not a record id: {record_id!r}")
        for key in self._computation_keys():
            path = self._path(key, record_id)
            if path.is_file():
                return _load_record(path, record_id, key)
        raise LookupError(f"no record {record_id} in {self.root}")

    def executions(self, computation_key: str) -> list[Record]:
        """Return every recorded execution of a computation, ordered by ``record_id`` (empty if none).

        Raises:
            IntegrityError: if a stored file does not hold the record its name and directory say.
        """
        return [
            _load_record(self._path(computation_key, rid), rid, computation_key) for rid in self._ids(computation_key)
        ]

    def canonical(self, computation_key: str) -> Record:
        """Return the computation's execution: the only one, or the declared one if there are several.

        A declaration is followed even when only one execution exists (it names that one).

        Raises:
            LookupError: if the computation has no execution, or several and none is declared.
            IntegrityError: if the declared record is missing or does not hold that record.
        """
        declared = self._declarations(computation_key)
        if declared:
            rid = declared[-1].record_id
            path = self._path(computation_key, rid)
            if not path.is_file():
                raise IntegrityError(f"computation {computation_key} declares {rid}, which is missing")
            return _load_record(path, rid, computation_key)
        ids = self._ids(computation_key)
        if not ids:
            raise LookupError(f"computation {computation_key} has no execution in {self.root}")
        if len(ids) > 1:
            raise LookupError(
                f"computation {computation_key} has {len(ids)} executions ({', '.join(ids)}) and none is declared "
                "canonical (RecordStore.declare_canonical)"
            )
        return _load_record(self._path(computation_key, ids[0]), ids[0], computation_key)

    def exists(self, computation_key: str) -> bool:
        """Return whether the computation has at least one recorded execution."""
        return (self.root / computation_key).is_dir()

    def produced_by(self, artifact_sha256: str) -> list[Record]:
        """Return the records whose outputs include the artifact (lineage; purpose inheritance reads it).

        Ordered by (``computation_key``, ``record_id``); every execution is included.
        """
        return [r for r in self._all(artifact_sha256) if any(o.sha256 == artifact_sha256 for o in r.outputs.values())]

    def find(self, kind: str, /, **config_fields: object) -> Iterator[Record]:
        """Yield records of ``kind`` whose config matches the given top-level fields.

        Values compare as JSON (a tuple matches a list; a nested config is passed as its JSON form). Records come in
        (``computation_key``, ``record_id``) order, every execution included. ``kind`` is positional-only, so a config
        field named ``kind`` can be matched too.

        Raises:
            TypeError: if a value is not JSON-serializable.
        """
        wanted = {name: json.loads(json.dumps(value)) for name, value in config_fields.items()}
        for record in self._all():
            if record.kind == kind and all(n in record.config and record.config[n] == v for n, v in wanted.items()):
                yield record

    def _path(self, computation_key: str, record_id: str) -> Path:
        return self.root / computation_key / f"{record_id}.json"

    def _ids(self, computation_key: str) -> list[str]:
        folder = self.root / computation_key
        if not (is_sha256(computation_key) and folder.is_dir()):
            return []
        return sorted(f.name.removesuffix(".json") for f in folder.iterdir() if _RECORD_FILE.match(f.name))

    def _computation_keys(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(d.name for d in self.root.iterdir() if is_sha256(d.name) and d.is_dir())

    def _all(self, containing: str = "") -> Iterator[Record]:
        """Yield every record (in key, id order) whose file's text contains ``containing``."""
        for key in self._computation_keys():
            for rid in self._ids(key):
                path = self._path(key, rid)
                if not containing or containing in path.read_text():
                    yield _load_record(path, rid, key)


def _write_once(root: Path, key: str, name: str, text: str) -> bool:
    """Atomically create ``root/key/name``, read-only, holding ``text``; return False (writing nothing) if it exists.

    The file is prepared in a hidden directory under ``root``. A new ``key`` directory appears by renaming that
    directory, so it always holds a record; an existing one gets the file by a hard link.
    """
    root.mkdir(parents=True, exist_ok=True)
    prepared = root / f".new-{uuid.uuid4().hex}"
    prepared.mkdir()
    tmp = prepared / name
    try:
        with tmp.open("w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.chmod(0o444)
        try:
            prepared.rename(root / key)
        except OSError:
            if not (root / key).is_dir():
                raise
        else:
            return True
        try:
            os.link(tmp, root / key / name)
        except FileExistsError:
            return False
        return True
    finally:
        if prepared.exists():
            tmp.unlink(missing_ok=True)
            prepared.rmdir()


def _load_record(path: Path, record_id: str, computation_key: str) -> Record:
    text = path.read_text()
    try:
        record = Record.model_validate_json(text)
    except ValidationError:  # newer code's provenance fields? Anything else unknown is still refused
        record = _StoredRecord.model_validate_json(text)
    if record.record_id() != record_id or record.computation_key() != computation_key:
        raise IntegrityError(f"{path} does not hold record {record_id} of computation {computation_key}")
    return record


class ResolvedRecords(RecordStore):
    """A record store that resolves each computation key once: the first :meth:`canonical` lookup is kept.

    ``stage_main`` gives every stage run one of these over the workspace's store, so a stage reads each record it
    cites once and records exactly the executions it used (their ``record_id``, outputs and purpose, from
    :attr:`resolved`; the record's ``Provenance.input_executions``): a canonical choice declared while it runs cannot
    change what it computed from, what it says it was drawn from, or the purpose it inherits. Everything else is
    :class:`RecordStore` on the same root.

    Args:
        root: the store root (``Paths.records``).
    """

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self._resolved: dict[str, Record] = {}

    @classmethod
    def of(cls, store: RecordStore) -> ResolvedRecords:
        """Return ``store`` if it already resolves once, else a new :class:`ResolvedRecords` on its root."""
        return store if isinstance(store, ResolvedRecords) else cls(store.root)

    @property
    def resolved(self) -> Mapping[str, Record]:
        """Computation key → the execution resolved for it, in lookup order (read-only)."""
        return MappingProxyType(self._resolved)

    def canonical(self, computation_key: str) -> Record:
        """Return the computation's execution, resolved on first use (:meth:`RecordStore.canonical`) and kept."""
        if computation_key not in self._resolved:
            self._resolved[computation_key] = super().canonical(computation_key)
        return self._resolved[computation_key]


def record_output(record: Record, kind: str) -> ArtifactRef:
    """Return the one output of ``record`` whose artifact kind is ``kind``.

    Raises:
        ValueError: if the record has no such output, or several.
    """
    found = [ArtifactRef(o.sha256, o.kind) for o in record.outputs.values() if o.kind == kind]
    if len(found) != 1:
        raise ValueError(f"record {record.record_id()[:12]} ({record.kind}) has {len(found)} {kind!r} outputs")
    return found[0]


def drawn_from(records: Mapping[str, Record]) -> dict[str, dict[str, object]]:
    """Return what an artifact made from ``records`` (role → the execution read) was drawn from (its meta ``inputs``).

    Per role: the computation key and the digest of each output of the execution read. No ``record_id``: two
    executions with the same outputs give the same meta, so identical results commit identical artifacts, while
    another result under the same key changes it. The exact executions are in the writing record
    (``Provenance.input_executions``, set by ``run._main.stage_main``).
    """
    return {
        role: {
            "computation_key": rec.computation_key(),
            "outputs": {n: o.sha256 for n, o in sorted(rec.outputs.items())},
        }
        for role, rec in sorted(records.items())
    }
