"""Extraction jobs: the wire format between the orchestrator and encoder workers (dependency-light, Python 3.11).

An :class:`ExtractionJob` is written as JSON by the orchestrator (``encoders.recipes`` validates it first) and read by
``python -m latent_interconnect.encoders.worker <job.json>`` inside the encoder's environment. The object list is cut
into parts of ``part_size`` objects, in its order. Workers share a job by static sharding: worker ``i`` of ``n`` takes
the parts whose index is ``i`` modulo ``n``, so two workers never compute the same part and a restarted worker skips the
parts that exist. Each part is written with :class:`~latent_interconnect.encoders.store.PartWriter`. The worker imports
``recipe["plugins"]`` first, so third-party extractors resolve, loads the extractor and, before it reuses any part or
the committed artifact, has the store accept its origin (:func:`worker_origin`, ``ArtifactStore.require_origin``):
workers of other code, kind of hardware (CPU or GPU) or fingerprint never share a job, a code-only change needs a
declared handoff, and another accelerator model is accepted and kept in the origin record. A
worker holds its writer lock while it runs and, at start, removes the temporary parts of workers that died
(``encoders.store.writer_lock``), so a killed job never blocks the commit. :func:`finalize` commits the artifact once
every part exists (the worker entry point calls it after each worker); a second call returns the committed reference
from the store's commit pointer.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore, sha256_file
from latent_interconnect.encoders.base import EXTRACTORS, Extractor, RawObject
from latent_interconnect.encoders.store import (
    EMBEDDINGS_KIND,
    PartWriter,
    finalize_embeddings,
    part_dir_name,
    writer_lock,
    writer_token,
)
from latent_interconnect.paths import running_code
from latent_interconnect.plugins import load_plugins

__all__ = [
    "ExtractionJob",
    "finalize",
    "job_file",
    "load_extractor",
    "missing_parts",
    "objects_sha256",
    "read_objects",
    "run_worker",
    "stage_objects",
    "staged_values",
    "worker_origin",
    "write_objects",
]

_BUILTIN = "latent_interconnect.encoders.extractors"
"""The package whose import registers the built-in extractors."""


@dataclass(frozen=True)
class ExtractionJob:
    """One extraction: a recipe applied to an object set, split into parts.

    Attributes:
        recipe: the validated encoder recipe, as a JSON-compatible dict (``EncoderRecipe.model_dump()``).
        objects: path of the object list (JSON lines: ``{"object_id", "value", optional "extra"}``; no pyarrow
            needed in encoder environments).
        objects_sha256: its sha256; workers refuse a table that differs.
        part_size: objects per part.
        artifacts_root: the artifact store root.
    """

    recipe: dict[str, Any]
    objects: str
    objects_sha256: str
    part_size: int
    artifacts_root: str

    def key(self) -> str:
        """Return the computation identity: the staging directory name, stable across restarts."""
        # Declared exception to latent_interconnect.canonical (sorted keys, default separators): committed extractions
        # are found by this key (store.committed), so changing the form would orphan them and recompute.
        body = json.dumps(
            {"recipe": self.recipe, "objects": self.objects_sha256, "part_size": self.part_size}, sort_keys=True
        )
        return "embed-" + hashlib.sha256(body.encode()).hexdigest()[:32]

    def dump(self, path: Path) -> None:
        """Write the job as JSON."""
        Path(path).write_text(json.dumps(asdict(self), sort_keys=True, indent=1))

    @classmethod
    def load(cls, path: Path) -> ExtractionJob:
        """Read a job written by :meth:`dump`."""
        return cls(**json.loads(Path(path).read_text()))


def _line(o: RawObject) -> str:
    row: dict[str, Any] = {"object_id": o.object_id, "value": o.value}
    if o.extra is not None:
        row["extra"] = dict(o.extra)
    return json.dumps(row, sort_keys=True) + "\n"


def write_objects(path: Path, objects: Iterable[RawObject]) -> str:
    """Write an object list as JSON lines (in the given order) and return its sha256."""
    with Path(path).open("w") as fh:
        for o in objects:
            fh.write(_line(o))
    return sha256_file(Path(path))


def objects_sha256(objects: Iterable[RawObject]) -> str:
    """Return the sha256 :func:`write_objects` gives this list (the lines are ASCII), without writing it."""
    digest = hashlib.sha256()
    for o in objects:
        digest.update(_line(o).encode("ascii"))
    return digest.hexdigest()


def read_objects(job: ExtractionJob) -> list[RawObject]:
    """Read the job's object list after checking its sha256.

    Raises:
        ValueError: if the file's sha256 differs from ``job.objects_sha256``.
    """
    found = sha256_file(Path(job.objects))
    if found != job.objects_sha256:
        raise ValueError(f"{job.objects}: sha256 {found}, the job expects {job.objects_sha256}")
    rows = [json.loads(line) for line in Path(job.objects).read_text().splitlines() if line]
    return [RawObject(r["object_id"], r["value"], r.get("extra")) for r in rows]


def _n_parts(n_objects: int, part_size: int) -> int:
    return max(1, math.ceil(n_objects / part_size))


def stage_objects(root: Path, objects: Iterable[RawObject]) -> tuple[Path, str]:
    """Write an object list once into the staging area of the store at ``root``, named by its own sha256.

    Returns:
        The file and its sha256 (an :class:`ExtractionJob`'s ``objects`` and ``objects_sha256``). The same list
        written again is the same file.
    """
    staging = Path(root) / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    tmp = staging / f"objects.{writer_token()}.jsonl.tmp"
    sha = write_objects(tmp, objects)
    path = _staged_list(root, sha)
    if path.is_file():
        tmp.unlink()  # content-addressed: the file there is this list
    else:
        tmp.rename(path)
    return path, sha


def _staged_list(root: Path, sha256: str) -> Path:
    return Path(root) / "_staging" / f"objects-{sha256}.jsonl"


def staged_values(root: Path, sha256: str) -> dict[str, str] | None:
    """Return object id → raw value of the object list :func:`stage_objects` wrote under ``sha256`` at ``root``.

    This is what an extraction was made from: its artifact records the list's sha256 (:func:`finalize`), and the list
    stays in the staging area. None if the list is not there.

    Raises:
        ValueError: if the file's sha256 is not its name's.
    """
    path = _staged_list(root, sha256)
    if not path.is_file():
        return None
    found = sha256_file(path)
    if found != sha256:
        raise ValueError(f"{path}: sha256 {found}, not the {sha256} it is named by")
    rows = (json.loads(line) for line in path.read_text().splitlines() if line)
    return {r["object_id"]: r["value"] for r in rows}


def job_file(job: ExtractionJob) -> Path:
    """Return where the job's JSON goes for a worker in another process (``<staging>.job.json``)."""
    return _staging_sibling(ArtifactStore(Path(job.artifacts_root)), job, "job.json")


def _staging_sibling(store: ArtifactStore, job: ExtractionJob, suffix: str) -> Path:
    """Return ``<staging>.<suffix>`` without creating the staging directory (``ArtifactStore.staging`` would)."""
    return store.root / "_staging" / f"{job.key()}.{suffix}"


def worker_origin(fingerprint: Mapping[str, str], device: str) -> dict[str, Any]:
    """Return a worker's origin as it runs: its code (commit, dirty), its hardware and the extractor's fingerprint.

    ``hardware`` is the accelerator model the worker runs on (``[]`` on the CPU), as ``provenance.hardware_of``
    lists devices; the fingerprint holds the weights' revision, the library versions and the dtype.

    Raises:
        ValueError: if the fingerprint does not declare ``dtype`` (the precision the model ran in).
    """
    if "dtype" not in fingerprint:
        raise ValueError(f"the extractor's fingerprint {dict(fingerprint)} does not declare its dtype")
    hardware: list[str] = []
    if device.startswith("cuda"):
        import torch  # noqa: PLC0415  (the extractor's load imported it)

        hardware = [torch.cuda.get_device_name(torch.device(device))]
    commit, dirty = running_code()
    return {"commit": commit, "dirty": dirty, "hardware": hardware, "fingerprint": dict(fingerprint)}


def load_extractor(recipe: Mapping[str, Any], device: str) -> Extractor:
    """Import the recipe's plugins, construct its registered extractor on ``device`` and load the model."""
    load_plugins((_BUILTIN, *recipe.get("plugins", ())))
    extractor: Extractor = EXTRACTORS.get(str(recipe["extractor"]))(dict(recipe), device)
    extractor.load()
    return extractor


def run_worker(
    job: ExtractionJob,
    device: str,
    shard: tuple[int, int] = (0, 1),
    deadline_s: float | None = None,
    handoff: Mapping[str, Any] | None = None,
    *,
    extractor: Extractor | None = None,
) -> int:
    """Extract this worker's parts until none is left (or the deadline passes); return the number written.

    The extractor is loaded first, even for a committed job: its parts or artifact are reused only once the store
    accepts this worker's origin (:func:`worker_origin`). A resident worker (``encoders.worker --serve``) passes the
    extractor it loaded for this recipe, so the model loads once for many jobs.

    Args:
        job: the job.
        device: the torch device string the extractor runs on.
        shard: ``(index, count)``: this worker takes the parts whose index is ``index`` modulo ``count``.
        deadline_s: stop starting new parts after this many seconds.
        handoff: a declared origin handoff (``ArtifactStore.require_origin``), for work of other code.
        extractor: the job's recipe's extractor, loaded (:func:`load_extractor`); None loads one.

    Raises:
        ValueError: for an invalid shard, or an object list that differs from the job.
        OriginError: if the job's work comes from another origin and no valid handoff covers it.
    """
    index, count = shard
    if not 0 <= index < count:
        raise ValueError(f"shard {shard}: index must be in [0, {count})")
    start = time.monotonic()
    store = ArtifactStore(Path(job.artifacts_root))
    if extractor is None:
        extractor = load_extractor(job.recipe, device)
    origin = worker_origin(extractor.fingerprint(), device)
    store.require_origin(job.key(), origin, handoff)
    if store.committed(job.key()) is not None:
        return 0
    objects = read_objects(job)
    staging = store.resume(job.key())
    if isinstance(staging, ArtifactRef):  # another worker's finalizer committed it while this one read the objects
        return 0
    written = 0
    with writer_lock(staging) as token:  # held while this worker lives; a killed worker's parts are swept by the next
        for part in range(index, _n_parts(len(objects), job.part_size), count):
            if deadline_s is not None and time.monotonic() - start > deadline_s:
                break
            writer = PartWriter(staging, part, token)
            if writer.done():
                continue
            chunk = objects[part * job.part_size : (part + 1) * job.part_size]
            batch = extractor.extract(chunk)
            expected = [o.object_id for o in chunk]
            embedded = [o for o in expected if o not in batch.skipped]
            if embedded != list(batch.object_ids) or not set(batch.skipped) <= set(expected):
                raise RuntimeError(
                    f"part {part}: the extractor returned {len(batch.object_ids)} objects and "
                    f"{len(batch.skipped)} skips for {len(expected)}, not each once in order"
                )
            writer.write(batch)
            written += 1
    return written


def missing_parts(job: ExtractionJob) -> list[int]:
    """Return the parts not yet written (empty once the job is complete or committed)."""
    store = ArtifactStore(Path(job.artifacts_root))
    if store.committed(job.key()) is not None:
        return []
    n_objects = sum(1 for line in Path(job.objects).read_text().splitlines() if line)
    staging = store.root / "_staging" / job.key()
    return [i for i in range(_n_parts(n_objects, job.part_size)) if not (staging / part_dir_name(i)).is_dir()]


def finalize(job: ExtractionJob) -> ArtifactRef:
    """Check that every part exists and no temp directory is left, then commit the artifact.

    The artifact's metadata holds the recipe, the object list's sha256, the part size and the extractor fingerprint
    (from the job's origin record, ``ArtifactStore.require_origin``).
    If the job was already committed, its reference is returned. Several finalizers may race: the store lets one
    commit, and the others wait for its pointer and return the same reference.

    Raises:
        RuntimeError: listing missing parts or stale temp directories, or when no worker recorded an origin.
    """
    store = ArtifactStore(Path(job.artifacts_root))
    key = job.key()
    done = store.committed(key)
    if done is not None:
        return done
    record = store.origin(key)
    if record is None:
        raise RuntimeError(f"job {key}: no worker has run this job")
    meta = {
        "recipe": job.recipe,
        "objects": job.objects_sha256,
        "part_size": job.part_size,
        "fingerprint": record["origin"]["fingerprint"],
    }
    staging = store.root / "_staging" / key
    missing = missing_parts(job)
    if missing and not staging.is_dir():
        # a worker ran, so the parts existed: another finalizer has claimed them and is committing
        return store.commit(staging, EMBEDDINGS_KIND, meta)
    if missing:
        raise RuntimeError(f"job {key}: {len(missing)} part(s) missing, e.g. {missing[:5]}")
    n_objects = sum(1 for line in Path(job.objects).read_text().splitlines() if line)
    try:
        return finalize_embeddings(store, staging, _n_parts(n_objects, job.part_size), meta)
    except FileNotFoundError:
        return store.commit(staging, EMBEDDINGS_KIND, meta)  # claimed by another finalizer mid-check: await it
    except ValueError as err:
        raise RuntimeError(str(err)) from err
