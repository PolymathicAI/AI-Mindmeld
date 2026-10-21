"""Content-addressed artifact store (dependency-light: standard library only, Python 3.11).

An *artifact* is a directory of files with a ``MANIFEST.json`` listing every file's sha256 and size, plus free-form
metadata. The artifact's identity is the sha256 of its canonical manifest, so it changes whenever any file or metadata
changes. Artifacts are written once:

1. a writer stages files in ``<root>/_staging/<key>/``, where ``key`` identifies the computation (resumable: committed
   parts survive a crash, and a restarted job skips them);
2. :meth:`ArtifactStore.commit` claims the directory (an atomic rename to ``<key>.claim-<writer>``, under an exclusive
   ``flock`` on ``<key>.claim-<writer>.lock`` held until the commit ends), hashes the files, writes the manifest and
   renames the directory to ``<root>/<sha[:2]>/<sha>/`` (atomic within one filesystem);
3. the artifact is then read-only (files are made non-writable).

Hashing: a writer may record its files' hashes as it finishes them (:func:`record_hashes`, a ``HASHES.json`` sidecar
per directory: name -> ``[sha256, size, mtime_ns]``); commit uses a recorded hash when the file's size and mtime still
match and hashes every other file in parallel threads. Sidecars are removed before the move and never listed, so the
manifest (and the digest) is the same whether or not hashes were recorded.

Recovery: a committer that dies leaves its claim behind. :meth:`ArtifactStore.recover` (called by
:meth:`ArtifactStore.staging`) renames a dead claim back to the staging directory, so the next writer finds its parts
and commits. A claim is dead when its lock can be taken without blocking, the rule ``encoders.store`` applies to
temporary parts. While a claim is left or its lock is held, :meth:`ArtifactStore.committed`,
:meth:`ArtifactStore.staging` and a second committer wait: until the committer's pointer appears, or it dies and its
claim is restored (bounded by liveness, and in any case by the site's longest walltime), so no writer starts an empty
staging directory beside a live commit and no waiter gives up on one. The moves of the staging directory (a claim, a
restore, its creation) are serialized by a short ``flock`` on ``<key>.lock``, so no waiter creates an empty staging
directory between another's restore and a third's claim. A key's pointer is written once: a late duplicate writer gets
the committed artifact (:meth:`ArtifactStore.resume`, or from its own commit) and never re-points the key.

Verification: files are hashed once, at commit. Readers check the manifest against the reference (cheap).
:meth:`ArtifactStore.verify` checks every file against the manifest through the persistent hash cache
``<root>/_hash_cache.jsonl``, which the store writes itself (so verifying needs write access to the root): a file
whose size, times and inode are unchanged since it was last hashed is not read again, so bit rot that leaves the
metadata alone is caught only after the cache file is removed. ``meta`` must be
deterministic (no timestamps or hostnames; those belong in records), or identical content would get two identities.

Origin: a resumable computation (extraction, capture) reuses its staged parts or committed artifact only after
:meth:`ArtifactStore.require_origin` accepts this run's origin (the code and the execution facts the running process
records) against the one recorded for the key; a code-only change is accepted only through a declared handoff.

This module is imported by encoder workers that run in foreign Python environments (``encoders.store``), so it must
not import pydantic, pandas or torch.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from latent_interconnect.canonical import canonical_json, is_sha256

__all__ = [
    "HARDWARE_HANDOFF",
    "HASHES_NAME",
    "HASH_CACHE_NAME",
    "MANIFEST_NAME",
    "ORIGIN_CODE",
    "ORIGIN_HARDWARE",
    "TMP_MARK",
    "ArtifactRef",
    "ArtifactStore",
    "HashCache",
    "IntegrityError",
    "Manifest",
    "OriginError",
    "hash_files",
    "read_handoff",
    "record_hashes",
    "sha256_file",
]

MANIFEST_NAME = "MANIFEST.json"
_CHUNK = 1 << 24
HASH_CACHE_NAME = "_hash_cache.jsonl"
HASHES_NAME = "HASHES.json"
"""A directory's writer-recorded file hashes (:func:`record_hashes`); never part of a manifest."""
COMMIT_WAIT_S = 7 * 24 * 3600.0
"""The bound on waiting for a live committer: 7 days, a long cluster job time limit that no job is expected to
outlive."""
ORIGIN_CODE = ("commit", "dirty")
"""The origin fields a declared handoff may change (the code); any other difference but the hardware (dtype, weights,
libraries) is never accepted."""
ORIGIN_HARDWARE = "hardware"
"""The origin field naming the accelerator models; with an ``execution`` mapping's ``device_name`` it is the hardware,
whose accelerator model may change without a declared handoff: the change is appended to the origin record
with :data:`HARDWARE_HANDOFF`, so the record keeps every hardware the key's work ran on. Its kind may not: CPU work
(``[]``) and GPU work are other origins."""
HARDWARE_HANDOFF = "hardware change: accepted without a declared handoff, kept in the origin record"
_CLAIM = ".claim-"
TMP_MARK = ".tmp-"
"""Marks a writer's unfinished temporary entry in a staging directory (``part-00000.tmp-<writer>``): not work."""
_HANDOFF_FIELDS = {"previous", "current", "reason"}


class IntegrityError(RuntimeError):
    """A file or artifact does not match its recorded hash."""


class OriginError(ValueError):
    """A computation's work was recorded under another origin than this run's, and no valid handoff covers it."""


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of a file, read in 16 MiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _threads() -> int:
    return min(32, len(os.sched_getaffinity(0)))


def hash_files(paths: Iterable[Path]) -> dict[Path, str]:
    """Return the sha256 of each file, hashed in parallel threads (``hashlib`` and reads release the GIL)."""
    paths = list(paths)
    with ThreadPoolExecutor(max_workers=_threads()) as pool:
        return dict(zip(paths, pool.map(sha256_file, paths), strict=True))


def record_hashes(directory: Path) -> None:
    """Record the sha256, size and mtime of every file directly in ``directory`` in its ``HASHES.json`` sidecar.

    A writer calls it once the directory's files are complete (while they are still in the page cache), so
    :meth:`ArtifactStore.commit` need not read them again.
    """
    directory = Path(directory)
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.name not in (HASHES_NAME, MANIFEST_NAME))
    rows = {}
    for p, sha in hash_files(files).items():
        st = p.stat()
        rows[p.name] = [sha, st.st_size, st.st_mtime_ns]
    (directory / HASHES_NAME).write_text(json.dumps(rows, sort_keys=True))


def _tree_hashes(root: Path) -> dict[str, tuple[str, int]]:
    """Return relative path -> (sha256, size) for every file under ``root`` except manifests and sidecars.

    A file's recorded hash (``HASHES.json`` in its directory) is used when its size and mtime still match; the
    other files are hashed in parallel.
    """
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in (MANIFEST_NAME, HASHES_NAME))
    recorded: dict[Path, list[Any]] = {}
    for sidecar in root.rglob(HASHES_NAME):
        recorded |= {sidecar.parent / name: row for name, row in json.loads(sidecar.read_text()).items()}
    stats = {p: p.stat() for p in files}
    known = {
        p: str(row[0])
        for p in files
        if (row := recorded.get(p)) and [row[1], row[2]] == [stats[p].st_size, stats[p].st_mtime_ns]
    }
    known |= hash_files(p for p in files if p not in known)
    return {str(p.relative_to(root)): (known[p], stats[p].st_size) for p in files}


@dataclass(frozen=True)
class ArtifactRef:
    """A reference to an artifact by content: the only way configs and records name data.

    Attributes:
        sha256: hash of the artifact's canonical manifest.
        kind: what the artifact holds (e.g. ``"embeddings"``, ``"ridge_init"``, ``"checkpoint"``, ``"scores"``).
    """

    sha256: str
    kind: str

    def __post_init__(self) -> None:
        if not is_sha256(self.sha256):
            raise ValueError(f"not a sha256 hex digest: {self.sha256!r}")

    @property
    def short(self) -> str:
        """Return the first 12 hex digits, for logs and file names."""
        return self.sha256[:12]


@dataclass(frozen=True)
class Manifest:
    """The canonical description of an artifact's content.

    Attributes:
        kind: artifact kind (see :class:`ArtifactRef`).
        files: relative path -> (sha256, size in bytes), for every file except the manifest itself.
        meta: JSON-compatible metadata (the producing spec, shapes, dtypes, ...); part of the identity.
    """

    kind: str
    files: Mapping[str, tuple[str, int]]
    meta: Mapping[str, Any] = field(default_factory=dict)

    def canonical(self) -> str:
        """Return the canonical JSON text whose sha256 is the artifact identity."""
        body = {
            "kind": self.kind,
            "files": {k: list(v) for k, v in sorted(self.files.items())},
            "meta": self.meta,
        }
        return canonical_json(body)

    def sha256(self) -> str:
        """Return the artifact identity."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Read a manifest file written by :meth:`ArtifactStore.commit`."""
        body = json.loads(Path(path).read_text())
        files = {k: (str(v[0]), int(v[1])) for k, v in body["files"].items()}
        return cls(kind=body["kind"], files=files, meta=body["meta"])


class HashCache:
    """Memoizes file hashes keyed by (path, size, mtime, ctime, inode), so verifying multi-GB inputs is cheap.

    With a ``path`` the cache is persistent: it is read from that file on first use, and each new hash is appended to
    it as one JSON line ``[path, size, mtime_ns, ctime_ns, inode, sha256]`` in a single ``O_APPEND`` write, so
    concurrent processes share it without a lock. The key includes ``ctime``, which tools cannot set back, so a file
    rewritten with its old size and mtime is hashed again (unless the rewrite falls in the same filesystem timestamp
    tick as the hashed version, a few milliseconds). A line that does not parse (a torn concurrent append) is
    not loaded: a missing entry only costs a re-hash. Without a ``path`` the cache lives in this process only.

    Args:
        path: the persistent cache file, or None for a per-process cache.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = None if path is None else Path(path)
        self._seen: dict[_CacheKey, str] | None = None

    def sha256(self, path: Path) -> str:
        """Return the sha256 of ``path``, hashing only if the file changed since it was last hashed."""
        sha = self.cached(path)
        if sha is None:
            sha = sha256_file(path)
            self.record(path, sha)
        return sha

    def cached(self, path: Path) -> str | None:
        """Return the cached sha256 of ``path`` as it is now, or None if it was not hashed in this state."""
        return self._entries().get(_cache_key(path))

    def record(self, path: Path, sha: str) -> None:
        """Remember ``sha`` as the sha256 of ``path`` in its current state (appending it to a persistent cache)."""
        seen, key = self._entries(), _cache_key(path)
        if seen.get(key) != sha:
            seen[key] = sha
            if self.path is not None:
                _append_cache_line(self.path, key, sha)

    def _entries(self) -> dict[_CacheKey, str]:
        if self._seen is None:
            self._seen = {}
            if self.path is not None and self.path.is_file():
                for line in self.path.read_text().splitlines():
                    if entry := _parse_cache_line(line):
                        self._seen[entry[0]] = entry[1]
        return self._seen


_CacheKey = tuple[str, int, int, int, int]  # resolved path, size, mtime_ns, ctime_ns, inode
_CACHE_LINE_FIELDS = 6  # the key and the sha256


def _cache_key(path: Path) -> _CacheKey:
    st = Path(path).stat()
    return (str(Path(path).resolve()), st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)


def _append_cache_line(cache: Path, key: _CacheKey, sha: str) -> None:
    """Append one entry to a persistent hash cache in a single ``O_APPEND`` write."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps([*key, sha], separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    fd = os.open(cache, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _parse_cache_line(line: str) -> tuple[_CacheKey, str] | None:
    """Parse one persistent-cache line; None if it is torn or malformed."""
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not (isinstance(row, list) and len(row) == _CACHE_LINE_FIELDS and isinstance(row[0], str)):
        return None
    path, size, mtime, ctime, ino, sha = row
    if not all(isinstance(v, int) for v in (size, mtime, ctime, ino)) or not is_sha256(sha):
        return None
    return (path, size, mtime, ctime, ino), sha


class ArtifactStore:
    """A directory of committed, content-addressed artifacts plus a staging area for writers.

    Args:
        root: the store root (``Paths.artifacts``); :meth:`verify` keeps its persistent :class:`HashCache` there, in
            ``HASH_CACHE_NAME``.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._hashes = HashCache(self.root / HASH_CACHE_NAME)

    def staging(self, key: str) -> Path:
        """Return (and create) the staging directory for the computation identified by ``key``.

        The same key always maps to the same directory, which makes interrupted writers resumable. A live committer
        of the key is waited for, and a dead committer's claim is restored (:meth:`recover`) first.

        Raises:
            TimeoutError: if a committer of the key stays alive past ``COMMIT_WAIT_S``.
        """
        path = self.root / "_staging" / key
        while True:
            _settle(path)
            if _made(path):
                return path

    def resume(self, key: str) -> ArtifactRef | Path:
        """Return the artifact ``key`` committed, or else its staging directory (:meth:`staging`), after one wait.

        A resumable writer calls it after its own planning, so a duplicate writer whose key was committed meanwhile
        returns that artifact instead of recomputing it into an empty staging directory.

        Raises:
            TimeoutError: if a committer of the key stays alive past ``COMMIT_WAIT_S``.
        """
        path = self.root / "_staging" / key
        while True:
            pointer = _settle(path)
            if pointer.is_file():
                return self.ref(pointer.read_text().strip())
            if _made(path):
                return path

    def recover(self, key: str) -> bool:
        """Rename a dead committer's claim of ``key`` back to its staging directory; return whether one was.

        Nothing happens while the staging directory exists. A claim ``<key>.claim-<writer>`` is dead when its lock
        ``<claim>.lock`` can be taken without blocking (its committer holds it from before the claim until the
        artifact is moved); the lock is held across the rename, so two recoverers never take one claim. The only file
        removed is a dead claim's ``MANIFEST.json``, which its committer wrote before dying and the next commit writes
        again.

        Args:
            key: the computation identity (the staging directory name).
        """
        return _recover(self.root / "_staging" / key)

    def committed(self, key: str) -> ArtifactRef | None:
        """Return the artifact the computation ``key`` already committed (from its ``<staging>.ref`` pointer), or None.

        A writer that finds one returns it instead of recomputing: a recomputation on non-deterministic hardware
        would commit a second digest for one computation. While a committer of the key is alive, this waits
        for its end; a dead one's claim is restored (:meth:`recover`). It never creates an empty staging directory.

        Raises:
            TimeoutError: if a committer stays alive past ``COMMIT_WAIT_S``.
        """
        pointer = _settle(self.root / "_staging" / key)
        return self.ref(pointer.read_text().strip()) if pointer.is_file() else None

    def require_origin(
        self, key: str, origin: Mapping[str, Any], handoff: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Accept this run's ``origin`` for the computation ``key`` before it reuses any of its work; return the record.

        The record, ``<staging>.origin.json``, holds ``origin`` (the origin the key's work is accepted under) and
        ``handoffs`` (the declared handoffs, oldest first). The first run of a key writes it exclusively. A later run
        reuses the key's parts or committed artifact only if its origin equals the recorded one, if the two differ only
        in the accelerator model (:data:`ORIGIN_HARDWARE`: the change is appended to the record as a handoff
        with reason :data:`HARDWARE_HANDOFF`; CPU against GPU is not a model change and is refused), or if ``handoff``
        declares the change: ``{"previous": <the recorded origin>, "current": <this run's origin>, "reason": ...}``,
        and the two differ only in :data:`ORIGIN_CODE` (and the accelerator model). A declared handoff's ``current``
        may name another accelerator model than the one the run lands on (it is written before the job is placed):
        the declared handoff is appended, then the hardware change from its ``current`` to this run's origin. The
        record then holds this run's origin; it is read, compared and rewritten under the key's lock, so concurrent
        runs on other hardware lose no handoff. Work with no recorded origin (parts, a claim or a pointer left without
        one) is never reused: it is refused; the same key is recomputed only in a clean artifact store (no bypass).

        Args:
            key: the computation identity (the staging directory name).
            origin: JSON-compatible facts of this run, at least :data:`ORIGIN_CODE`.
            handoff: the declared handoff (:func:`read_handoff`), or None.

        Raises:
            OriginError: naming both origins, if they differ beyond the code and the hardware, or in the code without a
                handoff that names both and a reason.
        """
        path = self.root / "_staging" / f"{key}.origin.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        origin = json.loads(json.dumps(dict(origin)))  # as the record holds it (tuples become lists)
        handoff = None if handoff is None else json.loads(json.dumps(dict(handoff)))
        while True:
            held = json.loads(path.read_text()) if path.is_file() else None
            if held is not None and held["origin"] == origin:
                return dict(held)
            if held is None:
                if self._has_work(key):
                    raise OriginError(
                        f"{key}: its work has no recorded origin, so it cannot be reused (no handoff adopts it); "
                        "recompute this key in a clean artifact store"
                    )
                record: dict[str, Any] = {"origin": origin, "handoffs": []}
                if _create(path, canonical_json(record)):
                    return record
                continue  # another run recorded it first: compare with that
            with _key_lock(path.parent / key):  # read, compare and append under the key's lock: no handoff is lost
                held = json.loads(path.read_text())
                if held["origin"] == origin:
                    return dict(held)
                added = _handoffs(key, held["origin"], origin, handoff)
                record = {"origin": origin, "handoffs": [*held["handoffs"], *added]}
                _write_atomic(path, canonical_json(record))
            return record

    def origin(self, key: str) -> dict[str, Any] | None:
        """Return the origin record of the computation ``key`` (:meth:`require_origin`), or None if none was made."""
        path = self.root / "_staging" / f"{key}.origin.json"
        return dict(json.loads(path.read_text())) if path.is_file() else None

    def _has_work(self, key: str) -> bool:
        """Whether ``key`` has a pointer, a claim or a finished staged entry (unfinished temporaries are not work)."""
        staging = self.root / "_staging"
        staged = staging / key
        if (staging / f"{key}.ref").is_file() or (
            staged.is_dir() and any(TMP_MARK not in p.name for p in staged.iterdir())
        ):
            return True
        return any(p.is_dir() for p in staging.glob(f"{key}{_CLAIM}*"))

    def exists(self, sha256: str) -> bool:
        """Return whether an artifact with this digest is in the store."""
        return self._dir(sha256).is_dir()

    def path(self, ref: ArtifactRef) -> Path:
        """Return the directory of a committed artifact (it may not exist)."""
        return self._dir(ref.sha256)

    def _dir(self, sha256: str) -> Path:
        return self.root / sha256[:2] / sha256

    def commit(self, staged: Path, kind: str, meta: Mapping[str, Any]) -> ArtifactRef:
        """Hash a staged directory, write its manifest and move it into the store.

        Safe when several writers finalize the same staging directory: each first claims it by an atomic rename to a
        writer-unique name, holding the claim's lock until the end. The one that wins commits and leaves a
        ``<staging>.ref`` pointer; the others wait while its lock is held and return the same reference, or, if it
        died, restore its claim (:meth:`recover`) and claim it themselves. Recorded hashes (:func:`record_hashes`) are
        used where valid; the other files are hashed in parallel. If an identical artifact already exists, the staged
        copy is removed and the existing one is returned. If the key's pointer already names an artifact (a late
        duplicate writer), that artifact is returned and the pointer is kept.

        Returns:
            The reference of the committed artifact.

        Raises:
            FileNotFoundError: if the staging directory vanished with neither a commit pointer nor a claim to restore.
            TimeoutError: if another committer stays alive past ``COMMIT_WAIT_S``.
        """
        staged = Path(staged)
        pointer = staged.with_name(staged.name + ".ref")
        while True:
            claim = staged.with_name(
                f"{staged.name}{_CLAIM}{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            )
            lock = claim.with_name(claim.name + ".lock")
            fd = _lock(lock, blocking=True)
            if fd is None:  # a blocking lock is always taken; this keeps the type exact
                raise RuntimeError(f"{lock}: not locked")
            try:
                try:
                    with _key_lock(staged):
                        staged.rename(claim)
                except FileNotFoundError:
                    pass  # another committer claimed it: wait for it below, without holding a lock of our own
                else:
                    return self._commit_claim(claim, pointer, kind, meta)
            finally:
                _unlock(lock, fd)
            if _settle(staged).is_file():
                return self.ref(pointer.read_text().strip())
            # a restored claim another waiter took before this one looked: claim or settle again
            if not staged.is_dir() and not _busy(staged) and not pointer.is_file():
                raise FileNotFoundError(f"{staged} vanished and no commit pointer appeared")

    def _commit_claim(self, claim: Path, pointer: Path, kind: str, meta: Mapping[str, Any]) -> ArtifactRef:
        files = _tree_hashes(claim)
        manifest = Manifest(kind=kind, files=files, meta=dict(meta))
        (claim / MANIFEST_NAME).write_text(manifest.canonical())
        for sidecar in list(claim.rglob(HASHES_NAME)):
            sidecar.unlink()
        ref = ArtifactRef(manifest.sha256(), kind)
        dest = self.path(ref)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            claim.rename(dest)
        except OSError:
            if not (dest / MANIFEST_NAME).is_file():
                raise
            _remove_tree(claim)
        else:
            for rel, (sha, _) in files.items():
                (dest / rel).chmod(0o444)
                self._hashes.record(dest / rel, sha)  # after the chmod, which changes the ctime in the cache key
        if not _create(pointer, ref.sha256):  # a late duplicate writer never re-points a committed key
            return self.ref(pointer.read_text().strip())
        return ref

    def remember(self, key: str, ref: ArtifactRef) -> ArtifactRef:
        """Point the computation ``key`` at a committed artifact, so :meth:`committed` finds it; return what it names.

        The pointer is written once (exclusively, as a commit's): a key that already names an artifact (a duplicate
        writer's) keeps it, and that artifact is returned instead of ``ref``.
        """
        pointer = self.root / "_staging" / f"{key}.ref"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        return ref if _create(pointer, ref.sha256) else self.ref(pointer.read_text().strip())

    def ref(self, sha256: str) -> ArtifactRef:
        """Return the reference of a committed artifact from its digest alone (the kind is read from its manifest).

        Raises:
            FileNotFoundError: if no such artifact is committed.
            IntegrityError: if the manifest does not hash to ``sha256``.
        """
        ref = ArtifactRef(sha256, Manifest.load(self._dir(sha256) / MANIFEST_NAME).kind)
        self.manifest(ref)  # checks that the manifest hashes to the digest
        return ref

    def manifest(self, ref: ArtifactRef) -> Manifest:
        """Load and check a committed artifact's manifest.

        Raises:
            IntegrityError: if the manifest does not hash to ``ref.sha256``.
        """
        manifest = Manifest.load(self.path(ref) / MANIFEST_NAME)
        if manifest.sha256() != ref.sha256 or manifest.kind != ref.kind:
            raise IntegrityError(f"manifest of {ref.short} does not match its reference")
        return manifest

    def verify(self, ref: ArtifactRef) -> None:
        """Check every file of an artifact against its manifest.

        Files not in the hash cache are hashed in parallel (:func:`hash_files`) and their hashes are cached.

        Raises:
            IntegrityError: on the first file that is missing or differs.
        """
        root = self.path(ref)
        files = self.manifest(ref).files
        for rel, (_, size) in files.items():
            p = root / rel
            if not p.is_file() or p.stat().st_size != size:
                raise IntegrityError(f"{ref.short}/{rel} is missing or differs from its manifest")
        known = {rel: self._hashes.cached(root / rel) for rel in files}
        for p, sha in hash_files(root / rel for rel, cached in known.items() if cached is None).items():
            self._hashes.record(p, sha)
            known[str(p.relative_to(root))] = sha
        for rel, (sha, _) in files.items():
            if known[rel] != sha:
                raise IntegrityError(f"{ref.short}/{rel} is missing or differs from its manifest")


def read_handoff(path: Path | None) -> dict[str, Any] | None:
    """Read a declared origin handoff (:meth:`ArtifactStore.require_origin`) from a JSON file; None for no path."""
    return None if path is None else dict(json.loads(Path(path).read_text()))


def _check_handoff(
    key: str, recorded: Mapping[str, Any], origin: Mapping[str, Any], handoff: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    """Return ``handoff`` if it validly hands ``key``'s work from its ``recorded`` origin to ``origin``; else refuse."""
    shown = (
        f"{key}: its work was recorded under {canonical_json(recorded)}; this run's origin is {canonical_json(origin)}"
    )
    was, now = _hardware_free(recorded), _hardware_free(origin)
    fixed = sorted(k for k in was.keys() | now.keys() if k not in ORIGIN_CODE and was.get(k) != now.get(k))
    if fixed:
        raise OriginError(f"{shown}: they differ in {fixed}, which no handoff waives; this is another computation")
    if handoff is None:
        raise OriginError(f"{shown}: reuse needs a declared handoff, a JSON file {{previous, current, reason}}")
    problems = []
    if set(handoff) != _HANDOFF_FIELDS or not str(handoff.get("reason", "")).strip():
        problems.append(f"it must hold exactly {sorted(_HANDOFF_FIELDS)} with a reason")
    if handoff.get("previous") != recorded:
        problems.append("its previous is not the recorded origin")
    current = handoff.get("current")
    if not isinstance(current, Mapping) or _hardware_free(current) != now:
        problems.append("its current is not this run's origin (up to the accelerator model)")
    if problems:
        raise OriginError(f"{shown}: the handoff is refused: {'; '.join(problems)}")
    return handoff


def _handoffs(
    key: str, recorded: Mapping[str, Any], origin: Mapping[str, Any], handoff: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """Return the handoffs that take ``key``'s work from its ``recorded`` origin to ``origin`` (OriginError if none).

    A change of accelerator model alone is one :data:`HARDWARE_HANDOFF`. Otherwise the declared ``handoff`` must cover
    it (:func:`_check_handoff`), followed by a hardware handoff when the run landed on another model than it names.
    """
    if _hardware_free(recorded) == _hardware_free(origin):
        return [{"previous": dict(recorded), "current": dict(origin), "reason": HARDWARE_HANDOFF}]
    declared = dict(_check_handoff(key, recorded, origin, handoff))
    if declared["current"] == origin:
        return [declared]
    return [declared, {"previous": declared["current"], "current": dict(origin), "reason": HARDWARE_HANDOFF}]


def _hardware_free(origin: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``origin`` without its accelerator model (:data:`ORIGIN_HARDWARE`, an ``execution``'s ``device_name``).

    The kind stays (kind and topology are kept): ``hardware`` becomes ``"cpu"`` (no accelerator) or
    ``"accelerator"``, so CPU work never continues on a GPU, nor GPU work on the CPU, without being another origin.
    """
    out = {k: v for k, v in origin.items() if k != ORIGIN_HARDWARE}
    if ORIGIN_HARDWARE in origin:
        out[ORIGIN_HARDWARE] = "accelerator" if origin[ORIGIN_HARDWARE] else "cpu"
    execution = out.get("execution")
    if isinstance(execution, Mapping):
        out["execution"] = {k: v for k, v in execution.items() if k != "device_name"}
    return out


def _create(path: Path, text: str) -> bool:
    """Write ``path`` whole and exclusively (a hard link of a complete temporary file); False if it exists."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}")
    tmp.write_text(text)
    try:
        os.link(tmp, path)
    except FileExistsError:
        return False
    finally:
        tmp.unlink()
    return True


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}")
    tmp.write_text(text)
    tmp.rename(path)


def _made(staged: Path) -> bool:
    """Create the staging directory ``staged`` if absent; return whether it is there now.

    False when a committer owns it (another waiter claimed a restored directory before this one looked), or when it
    existed at the ``mkdir`` but is gone at the check: in both cases the caller settles again.
    """
    staged.parent.mkdir(parents=True, exist_ok=True)
    with _key_lock(staged):
        if not staged.is_dir() and _busy(staged):
            return False
        try:
            staged.mkdir()
        except FileExistsError:
            return staged.is_dir()
        return True


@contextmanager
def _key_lock(staged: Path) -> Iterator[None]:
    """Hold ``<key>.lock`` (blocking): the moves of a staging directory (claim, restore, creation) happen under it."""
    lock = staged.with_name(staged.name + ".lock")
    fd = _lock(lock, blocking=True)
    if fd is None:  # a blocking lock is always taken; this keeps the type exact
        raise RuntimeError(f"{lock}: not locked")
    try:
        yield
    finally:
        _unlock(lock, fd)


def _settle(staged: Path, wait_s: float = COMMIT_WAIT_S) -> Path:
    """Wait until no committer owns ``staged``; return its pointer path (the pointer exists if it was committed).

    The wait ends when the pointer appears, when the staging directory exists (a dead committer's claim is restored
    first, :func:`_recover`), or when no claim of it is left (:func:`_busy`). It is bounded by the committer's life,
    whose lock the kernel releases when it dies, and in any case by ``wait_s``.

    Raises:
        TimeoutError: if a committer is still alive after ``wait_s``.
    """
    pointer = staged.with_name(staged.name + ".ref")
    deadline, poll = time.monotonic() + wait_s, 0.05
    while not pointer.is_file() and not _recover(staged) and not staged.is_dir() and _busy(staged):
        if time.monotonic() > deadline:
            raise TimeoutError(f"{staged}: its committer is still alive after {wait_s:.0f} s")
        time.sleep(poll)
        poll = min(2 * poll, 5.0)
    return pointer


def _busy(staged: Path) -> bool:
    """Return whether a committer may still own ``staged``.

    It may while a claim of it is left (one a concurrent recoverer holds), or while a claim's lock is held (its
    committer moved the claim into the store and has not yet written the pointer).
    """
    for path in staged.parent.glob(f"{staged.name}.*"):
        suffix = path.name[len(staged.name) :]
        if suffix.startswith(_CLAIM) and not suffix.endswith(".lock"):
            return True
        if suffix.startswith(_CLAIM):
            fd = _lock(path, blocking=False)
            if fd is None:
                return True
            _unlock(path, fd)  # a lock left by a committer that ended
    return False


def _recover(staged: Path) -> bool:
    """:meth:`ArtifactStore.recover` for the staging directory ``staged``."""
    if staged.exists() or not staged.parent.is_dir():
        return False
    with _key_lock(staged):
        return not staged.exists() and _restore_dead_claim(staged)


def _restore_dead_claim(staged: Path) -> bool:
    key = staged.name
    for claim in sorted(staged.parent.glob(f"{key}.*")):
        suffix = claim.name[len(key) :]
        if suffix.startswith(_CLAIM) and not suffix.endswith(".lock"):
            lock = claim.with_name(claim.name + ".lock")
            fd = _lock(lock, blocking=False)
            if fd is None:  # a live committer
                continue
            try:
                (claim / MANIFEST_NAME).unlink(missing_ok=True)  # uncommitted; commit writes it again
                if _restore(claim, staged):
                    return True
            finally:
                _unlock(lock, fd)
    return False


def _restore(claim: Path, staged: Path) -> bool:
    """Rename a dead claim back to its staging directory; False if another process moved either first."""
    try:
        claim.rename(staged)
    except OSError:  # the claim is gone, or the staging directory was recreated meanwhile
        return False
    return True


def _lock(path: Path, *, blocking: bool) -> int | None:
    """Take an exclusive ``flock`` on ``path`` (created if absent); return its descriptor, or None if it is held.

    The lock counts only if ``path`` still names the locked file afterwards: a holder unlinks its lock file before it
    releases it, so a lock taken on an unlinked file is retried on the new one. ``flock`` is honored on the site's
    CephFS across hosts (measured).
    """
    while True:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            os.close(fd)
            return None
        try:
            current = path.stat().st_ino == os.fstat(fd).st_ino
        except FileNotFoundError:
            current = False
        if current:
            return fd
        os.close(fd)


def _unlock(path: Path, fd: int) -> None:
    path.unlink()  # before the release, so no one locks a file that is about to go
    os.close(fd)


def _remove_tree(path: Path) -> None:
    """Delete a staging directory this process created (bottom-up, no glob outside it)."""
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for name in filenames:
            (Path(dirpath) / name).unlink()
        for name in dirnames:
            (Path(dirpath) / name).rmdir()
    Path(path).rmdir()
