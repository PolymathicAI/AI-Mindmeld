"""The embedding store: one content-addressed artifact per (encoder recipe, object set), written in resumable parts.

Layout of an embedding artifact (kind ``"embeddings"``)::

    MANIFEST.json                 file hashes + meta: recipe, fingerprint, object-set sha, layers, dims, dtypes, parts
    part-00000/ids.json           object ids of the part, in row order
    part-00000/pooled.L53.npy     (n, d) float32, one file per read layer
    part-00000/residues.L53.npy   (Σ L_i, d) bf16-as-uint16 or float32, concatenated in row order (optional)
    part-00000/offsets.npy        (n + 1,) int64 row offsets into the residue arrays (with residues only)
    part-00000/mean.L53.npy       (n, d) float32 mean over a capture's read tokens (captures only)
    part-00000/notes.json         object id → how the extractor computed it (``ExtractedBatch.notes``; when any)
    ...

Writing (``jobs`` drives it): every part is staged in ``<staging>/part-XXXXX.tmp-<writer>/`` and renamed to
``part-XXXXX/`` when complete, so a part is either absent or whole; a restarted job skips the parts that exist. Each
writer holds an exclusive lock, ``<staging>.writer-<writer>.lock``, while it writes (:func:`writer_lock`), so a
temporary part whose writer's lock is free belongs to a writer that died (a killed job) and is removed when the next
writer starts (:func:`sweep_dead_writers`); a live writer's never is. ``flock`` is honored on the site's CephFS across
hosts (measured between two hosts, both directions). A finished part records its files'
hashes (``HASHES.json``, ``artifacts.store.record_hashes``); when every part exists, :meth:`ArtifactStore.commit`
uses those hashes (hashing in parallel any part written without them) and moves the artifact into the store. Reading
memory-maps the part files, on every read or, with ``keep_maps``, once per reader (:class:`EmbeddingReader`); the index
id → (part, row) is built from the ``ids.json`` files.

Layers are stored separately, so one pass serves several read layers; a connector's input (e.g. ``concat`` of layers
53 and 79) is composed at read time (``encoders.loader``). The encoder's defined output (recipe ``read = "output"``,
e.g. MIMIC's 5 register tokens flattened to one vector) is stored as layer :data:`OUTPUT_LAYER`.

The manifest metadata holds at least ``layers`` (the stored layer keys) and ``dim`` (the vector width). Encoder
embeddings also hold ``recipe`` (the recipe dict, at least its ``name``), which ``encoders.loader`` checks. A
receiver-state capture (``build.capture``) uses the same format: its pooled vectors are the ridge targets and its
``mean`` files the read-token means (the ridge bias); its metadata names the capture ``recipe`` (a string) and the
``target`` state, and it has no encoder recipe.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from latent_interconnect.artifacts.store import TMP_MARK as _TMP
from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore, _lock, _unlock, record_hashes
from latent_interconnect.encoders.base import ExtractedBatch

__all__ = [
    "EMBEDDINGS_KIND",
    "OUTPUT_LAYER",
    "EmbeddingReader",
    "PartWriter",
    "commit_embeddings",
    "finalize_embeddings",
    "part_dir_name",
    "sweep_dead_writers",
    "writer_lock",
    "writer_token",
]

EMBEDDINGS_KIND = "embeddings"
"""Artifact kind of encoder embeddings and of receiver-state captures."""
OUTPUT_LAYER = -1
"""The layer key of the encoder's defined output (``read = "output"``): the last ``hidden_states`` index (the final
output). The actual index is resolved in the artifact's metadata where it is known (an extractor's fingerprint
``output_layer``)."""
_SKIPPED_FILE = "skipped.json"
"""A part's skipped objects (object id → reason), when it has any."""
_NOTES_FILE = "notes.json"
"""A part's object notes (object id → how it was computed, ``ExtractedBatch.notes``), when it has any."""


def part_dir_name(index: int) -> str:
    """Return the directory name of part ``index``."""
    return f"part-{index:05d}"


class PartWriter:
    """Writes one part atomically into a staging directory.

    Args:
        staging: the artifact's staging directory (``ArtifactStore.staging(job_key)``).
        index: part number.
        writer: a token unique to this process (host + pid), so concurrent writers never share a temp directory.
    """

    def __init__(self, staging: Path, index: int, writer: str) -> None:
        self.final = Path(staging) / part_dir_name(index)
        self.tmp = Path(staging) / f"{part_dir_name(index)}{_TMP}{writer}"

    def done(self) -> bool:
        """Return True if this part is already committed (resume)."""
        return self.final.is_dir()

    def _write_object_maps(self, batch: ExtractedBatch) -> None:
        """Write the part's skipped objects and object notes, each only when it has any."""
        if set(batch.skipped) & set(batch.object_ids):
            raise ValueError(f"objects both embedded and skipped: {sorted(set(batch.skipped) & set(batch.object_ids))}")
        if set(batch.notes) - set(batch.object_ids):
            raise ValueError(f"notes for objects not embedded: {sorted(set(batch.notes) - set(batch.object_ids))[:5]}")
        for name, objects in ((_SKIPPED_FILE, batch.skipped), (_NOTES_FILE, batch.notes)):
            if objects:
                (self.tmp / name).write_text(json.dumps(dict(objects), sort_keys=True))

    def write(self, batch: ExtractedBatch) -> None:
        """Write the part's arrays and ids, then rename the directory into place.

        If another writer committed the same part first, this writer's copy is discarded.

        Raises:
            ValueError: if a pooled vector or mean is not finite (a NaN is a bug).
        """
        self.tmp.mkdir(parents=True)
        (self.tmp / "ids.json").write_text(json.dumps(list(batch.object_ids)))
        self._write_object_maps(batch)
        if batch.means and set(batch.means) != set(batch.pooled):
            raise ValueError(f"means at layers {sorted(batch.means)}, pooled at {sorted(batch.pooled)}")
        for layer, arr in batch.pooled.items():
            np.save(self.tmp / f"pooled.L{layer}.npy", _finite(arr, f"pooled vectors at layer {layer}"))
        for layer, arr in batch.means.items():
            np.save(self.tmp / f"mean.L{layer}.npy", _finite(arr, f"read-token means at layer {layer}"))
        offsets: np.ndarray | None = None
        for layer, rows in batch.residues.items():
            lengths = np.array([len(r) for r in rows], dtype=np.int64)
            this = np.concatenate([[0], np.cumsum(lengths)])
            if offsets is not None and not np.array_equal(offsets, this):
                raise ValueError(f"layer {layer}: residue lengths differ from the other layers")
            offsets = this
            np.save(self.tmp / f"residues.L{layer}.npy", np.concatenate(rows))
        if offsets is not None:
            np.save(self.tmp / "offsets.npy", offsets)
        record_hashes(self.tmp)  # while the files are in the page cache, so the commit need not read them
        try:
            self.tmp.rename(self.final)
        except OSError:
            if not self.final.is_dir():
                raise
            for f in self.tmp.iterdir():
                f.unlink()
            self.tmp.rmdir()


@dataclass(frozen=True)
class _Loc:
    part: int
    row: int


class EmbeddingReader:
    """Read access to a committed embedding artifact (memory-mapped, read-only).

    By default every read memory-maps the part files it touches afresh (``np.load(mmap_mode="r")``). With
    ``keep_maps`` each part file is mapped on its first read and the map is kept for the reader's lifetime, so a batch
    opens no file (without it, the trainer's loader opens every part file a batch touches, on every batch, each open a
    round trip to the file system). A kept map cannot go stale:
    an artifact is written once, is read-only from its commit on (``artifacts.store``) and is named by its content, so
    the map shows the same bytes for as long as it is held. The cache is bounded by the artifact: at most one map per
    part file read, each holding one file descriptor (Python's ``mmap`` keeps its own) and address space, not memory
    (295 descriptors for that run). The pages a map has read are the operating system's page cache, shared and
    reclaimable, as when every batch opened its own maps (about 2.7 GB of pooled vectors for that run). A missing file
    is not remembered: it is looked for again on the next read.

    Args:
        store: the artifact store.
        ref: the embedding artifact; its manifest is checked on open.
        keep_maps: keep each part file's map for the reader's lifetime (the trainer's ``input_cache``).
    """

    def __init__(self, store: ArtifactStore, ref: ArtifactRef, *, keep_maps: bool = False) -> None:
        self.root = store.path(ref)
        self.meta = store.manifest(ref).meta
        recipe = self.meta.get("recipe")
        self.recipe: str | None = str(recipe["name"]) if isinstance(recipe, Mapping) else None
        self.layers: tuple[int, ...] = tuple(int(x) for x in self.meta["layers"])
        self._parts = sorted(p.name for p in self.root.iterdir() if p.name.startswith("part-"))
        self._index: dict[str, _Loc] = {}
        self.keep_maps = keep_maps
        self._maps: dict[tuple[int, str], np.ndarray] = {}
        for i, name in enumerate(self._parts):
            for row, oid in enumerate(json.loads((self.root / name / "ids.json").read_text())):
                self._index[oid] = _Loc(i, row)

    @property
    def object_ids(self) -> tuple[str, ...]:
        """Return every object id the artifact holds, in part and row order."""
        return tuple(self._index)

    def __contains__(self, object_id: object) -> bool:
        return object_id in self._index

    def __len__(self) -> int:
        return len(self._index)

    def pooled(self, object_ids: Sequence[str], layer: int) -> np.ndarray:
        """Return ``(len(object_ids), d)`` float32 pooled vectors at ``layer``, in the given order.

        Raises:
            KeyError: for an id that is not in the artifact.
            ValueError: if a vector is not finite (a NaN is a bug).
        """
        rows = self._rows(object_ids, f"pooled.L{layer}.npy")
        return _finite(rows, f"{self.root.name}: pooled vectors at layer {layer}")

    def means(self, object_ids: Sequence[str], layer: int) -> np.ndarray:
        """Return a capture's ``(len(object_ids), d)`` float32 read-token means at ``layer``, in the given order.

        Raises:
            KeyError: for an id that is not in the artifact.
            FileNotFoundError: if the artifact holds no means (an encoder's embeddings).
            ValueError: if a mean is not finite (a NaN is a bug).
        """
        rows = self._rows(object_ids, f"mean.L{layer}.npy")
        return _finite(rows, f"{self.root.name}: read-token means at layer {layer}")

    def residues(self, object_ids: Sequence[str], layer: int) -> list[np.ndarray]:
        """Return each object's ``(L_i, d)`` residue states at ``layer`` (stored dtype; bf16 as ``uint16``)."""
        out: list[np.ndarray] = [np.empty(0)] * len(object_ids)
        for part, (positions, rows) in self._by_part(object_ids).items():
            offsets, states = self._load(part, "offsets.npy"), self._load(part, f"residues.L{layer}.npy")
            for k, r in zip(positions, rows, strict=True):
                out[k] = states[offsets[r] : offsets[r + 1]]
        return out

    def _by_part(self, object_ids: Sequence[str]) -> dict[int, tuple[list[int], list[int]]]:
        """Group the request by part: part → (positions in ``object_ids``, rows in the part); each file opens once."""
        groups: dict[int, tuple[list[int], list[int]]] = {}
        for k, o in enumerate(object_ids):
            loc = self._index[o]
            positions, rows = groups.setdefault(loc.part, ([], []))
            positions.append(k)
            rows.append(loc.row)
        return groups

    def _rows(self, object_ids: Sequence[str], name: str) -> np.ndarray:
        """Return the rows of file ``name`` for these objects, in the given order (repeats kept)."""
        out: np.ndarray | None = None
        for part, (positions, rows) in self._by_part(object_ids).items():
            arr = self._load(part, name)
            if out is None:
                out = np.empty((len(object_ids), *arr.shape[1:]), arr.dtype)
            out[positions] = arr[rows]
        return np.stack([]) if out is None else out  # an empty request raises, as before

    def _load(self, part: int, name: str) -> np.ndarray:
        """Return part ``part``'s file ``name``, memory-mapped read-only (kept with ``keep_maps``: class docstring).

        Raises:
            FileNotFoundError: if the part has no such file.
        """
        arr = self._maps.get((part, name))
        if arr is None:
            arr = np.load(self.root / self._parts[part] / name, mmap_mode="r")
            if self.keep_maps:
                self._maps[(part, name)] = arr
        return arr


def _finite(arr: np.ndarray, what: str) -> np.ndarray:
    """Return ``arr`` as contiguous float32, refusing a non-finite value (a NaN is a bug, never a vector)."""
    out = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError(f"{what}: {int((~np.isfinite(out)).sum())} values are not finite")
    return out


def writer_token() -> str:
    """Return a token unique to this process (host + pid), for :class:`PartWriter`."""
    return f"{socket.gethostname()}-{os.getpid()}"


def _lock_path(staging: Path, token: str) -> Path:
    return staging.parent / f"{staging.name}.writer-{token}.lock"


def sweep_dead_writers(staging: Path, own: str) -> list[str]:
    """Remove the temporary parts in ``staging`` whose writers are dead; return their names.

    A writer is dead when its lock (``<staging>.writer-<token>.lock``) can be taken without blocking. The temporary
    parts of ``own``, the caller's token, are removed too: the caller holds that lock and has not written yet, so they
    were left by an earlier process with the same host and pid.

    Args:
        staging: the staging directory.
        own: the caller's writer token (whose lock it holds).
    """
    removed = []
    for tmp in sorted(Path(staging).glob(f"part-*{_TMP}*")):
        token = tmp.name.partition(_TMP)[2]
        if token == own:
            shutil.rmtree(tmp)
            removed.append(tmp.name)
            continue
        path = _lock_path(Path(staging), token)
        fd = _lock(path, blocking=False)
        if fd is None:  # a live writer
            continue
        try:
            if tmp.is_dir():  # another sweeper may have removed it first
                shutil.rmtree(tmp)
                removed.append(tmp.name)
        finally:
            _unlock(path, fd)
    return removed


@contextmanager
def writer_lock(staging: Path) -> Iterator[str]:
    """Hold this process's writer lock on ``staging`` for the block, after removing dead writers' temporary parts.

    Yields:
        The writer token (:func:`writer_token`) to give :class:`PartWriter`.
    """
    staging = Path(staging)
    token = writer_token()
    path = _lock_path(staging, token)
    fd = _lock(path, blocking=True)
    if fd is None:  # a blocking lock is always taken; this keeps the type exact
        raise RuntimeError(f"{path}: not locked")
    try:
        sweep_dead_writers(staging, token)
        yield token
    finally:
        _unlock(path, fd)


def commit_embeddings(
    store: ArtifactStore, key: str, batches: Iterable[ExtractedBatch], meta: Mapping[str, Any]
) -> ArtifactRef:
    """Write each batch as one part (in order) into the staging directory ``key`` and commit the artifact.

    Parts that already exist are kept (resume), so the batches must be the same on a re-run; a computation that was
    already committed returns its artifact without consuming ``batches``.

    Args:
        store: the artifact store.
        key: the computation identity (the staging directory name): it must cover the content of every input (an
            import's file digest, not its path), since a committed key returns its artifact unread.
        batches: the parts' contents, in part order.
        meta: the manifest metadata; ``layers`` and ``dim`` are added (:func:`finalize_embeddings`).

    Raises:
        ValueError: as :func:`finalize_embeddings`.
    """
    staging = store.resume(key)
    if isinstance(staging, ArtifactRef):
        return staging
    count = 0
    with writer_lock(staging) as token:
        for index, batch in enumerate(batches):
            writer = PartWriter(staging, index, token)
            if not writer.done():
                writer.write(batch)
            count = index + 1
    return finalize_embeddings(store, staging, count, meta)


def finalize_embeddings(store: ArtifactStore, staging: Path, n_parts: int, meta: Mapping[str, Any]) -> ArtifactRef:
    """Check that parts ``0 .. n_parts - 1`` are complete and consistent, then commit the embedding artifact.

    Args:
        store: the artifact store.
        staging: the staging directory holding the parts.
        n_parts: the number of parts the computation has.
        meta: the manifest metadata; ``layers`` (the pooled layers) and ``dim`` (their width) are added.

    Raises:
        ValueError: if there are no parts, a part is missing, a temporary part directory is left, an unexpected
            entry is present, or the parts disagree on layers, width or which layers have means.
    """
    if n_parts < 1:
        raise ValueError("no part to commit")
    expected = {part_dir_name(i) for i in range(n_parts)}
    present = {p.name for p in Path(staging).iterdir()}
    if present != expected:
        temporary = sorted(p for p in present if _TMP in p)
        hint = (
            f"; temporary parts {temporary[:5]} are a live writer's, or a dead one's that the next writer removes"
            if temporary
            else ""
        )
        raise ValueError(
            f"{staging}: missing parts {sorted(expected - present)[:5]}, extra entries {sorted(present - expected)[:5]}"
            + hint
        )
    layers: set[tuple[int, ...]] = set()
    widths: set[int] = set()
    means: set[tuple[int, ...]] = set()
    skipped: dict[str, str] = {}
    notes: dict[str, str] = {}
    conflicts: list[str] = []
    for name in sorted(expected):
        part = Path(staging) / name
        files = [f.name for f in part.iterdir()]
        these = tuple(sorted(int(f[len("pooled.L") : -len(".npy")]) for f in files if f.startswith("pooled.L")))
        layers.add(these)
        means.add(tuple(sorted(int(f[len("mean.L") : -len(".npy")]) for f in files if f.startswith("mean.L"))))
        arrays = [np.load(part / f"pooled.L{L}.npy", mmap_mode="r") for L in these]
        widths |= {int(a.shape[1]) for a in arrays if len(a)}  # a part whose objects were all skipped has no width
        for file, gathered in ((_SKIPPED_FILE, skipped), (_NOTES_FILE, notes)):
            if file in files:
                for object_id, text in json.loads((part / file).read_text()).items():
                    if gathered.setdefault(object_id, text) != text:
                        conflicts.append(object_id)
    if conflicts:
        raise ValueError(
            f"{staging}: objects noted or skipped in two parts for different reasons: {sorted(conflicts)[:5]}"
        )
    if not widths:
        examples = dict(list(skipped.items())[:3])
        raise ValueError(f"{staging}: every object was skipped ({len(skipped)}), e.g. {examples}")
    if len(layers) != 1 or len(widths) != 1 or len(means) != 1:
        raise ValueError(f"{staging}: parts disagree on layers {sorted(layers)}, widths {sorted(widths)} or means")
    counts: dict[str, Any] = {"skipped": dict(sorted(skipped.items())), "n_skipped": len(skipped)}
    if notes:  # an extraction without notes keeps the manifest it always had
        counts |= {"notes": dict(sorted(notes.items())), "n_notes": len(notes)}
    return store.commit(
        Path(staging), EMBEDDINGS_KIND, {**meta, "layers": list(layers.pop()), "dim": widths.pop(), **counts}
    )
