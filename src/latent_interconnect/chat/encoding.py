"""Live encoding for chat: one resident encoder worker per environment, committing ordinary embedding artifacts.

A worker is ``<interpreter> -m latent_interconnect.encoders.worker --serve --device <device>`` (``encoders.worker``):
the encoder environment's own interpreter (``EncoderRecipe.environment``, resolved by the site launcher; the package
names no environment path), with this package's source on its path, as the extract stage runs its worker. A job is one
object list staged in the pool's store, chat's own (``encoders.jobs.stage_objects``), so an object encoded before is
found again by its key and never recomputed, and the artifact is the one an ``extract`` run of the same list would
commit.
A worker starts on first use, loads each recipe's model once and stays up until the pool closes; one that died
starts again on the next job. Its state per environment: ``cold``, ``starting``, ``warm`` or ``error``.

With ``max_resident`` (a local deployment: one GPU for the receiver and the encoders), a worker serves one recipe and
at most that many workers stay up: before another starts, the least recently used idle one is stopped (its process
ends, so its GPU memory is released), and a worker serving a job is never stopped (the new one waits for it, within
the job timeout). Encoders otherwise accumulate on the GPU (with Gemma 3 12B and four encoders resident, Uni-Mol2
ran out of memory). A stopped encoder loads again on its next raw input.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import IO, Literal

import latent_interconnect
from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.encoders.base import RawObject
from latent_interconnect.encoders.jobs import ExtractionJob, stage_objects
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import EMBEDDINGS_KIND
from latent_interconnect.encoders.worker import REPLY

__all__ = ["EncoderError", "EncoderPool"]

Status = Literal["cold", "starting", "warm", "error"]


class EncoderError(RuntimeError):
    """A live encoding failed (no interpreter for the environment, the worker died or timed out, or the job failed)."""


class _Worker:
    def __init__(self, python: str, device: str, environment: str = "") -> None:
        self.python, self.device, self.environment = python, device, environment
        self.users = 0  # jobs holding it (the pool's lock guards this count)
        self.process: subprocess.Popen[str] | None = None
        self.replies: queue.Queue[dict[str, object] | None] = queue.Queue()
        self.status: Status = "cold"
        self.error: str | None = None
        self.lock = threading.Lock()

    def _start(self, timeout_s: float) -> subprocess.Popen[str]:
        src = str(Path(latent_interconnect.__file__).resolve().parents[1])
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (src, os.environ.get("PYTHONPATH"))))}
        command = [self.python, "-m", "latent_interconnect.encoders.worker", "--serve", "--device", self.device]
        self.status = "starting"
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        self.process, self.replies = process, queue.Queue()
        threading.Thread(target=_read_replies, args=(process.stdout, self.replies), daemon=True).start()
        self._reply(process, timeout_s)  # the ready line
        return process

    def _reply(self, process: subprocess.Popen[str], timeout_s: float) -> dict[str, object]:
        try:
            reply = self.replies.get(timeout=timeout_s)
        except queue.Empty:
            reply = None
        if reply is None:
            process.kill()
            process.wait(timeout=30)  # reaped, so its GPU memory is free before another worker starts
            raise EncoderError(f"the encoder worker ({self.python}) exited or did not answer within {timeout_s:.0f} s")
        return reply

    def run(self, job: ExtractionJob, timeout_s: float) -> str:
        with self.lock:
            try:
                process = self.process
                if process is None or process.poll() is not None:
                    process = self._start(timeout_s)
                assert process.stdin is not None
                process.stdin.write(json.dumps(asdict(job)) + "\n")
                process.stdin.flush()
                reply = self._reply(process, timeout_s)
            except (OSError, EncoderError) as err:
                self.status, self.error, self.process = "error", str(err), None
                raise EncoderError(str(err)) from err
            self.status, self.error = "warm", None
            if "error" in reply:
                raise EncoderError(str(reply["error"]))
            return str(reply["embeddings"])

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            assert self.process.stdin is not None
            self.process.stdin.close()  # the worker ends at the end of its input
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()  # reaped: its GPU memory is released before another worker starts
        self.process, self.status = None, "cold"


def _read_replies(out: IO[str] | None, replies: queue.Queue[dict[str, object] | None]) -> None:
    """Pass a worker's reply lines on as they come (its other output is dropped); None when its output ends."""
    assert out is not None
    for line in out:
        if line.startswith(REPLY):
            replies.put(json.loads(line[len(REPLY) :]))
    replies.put(None)


class EncoderPool:
    """Resident encoder workers: one per environment, or one per recipe and at most ``max_resident`` of them.

    Args:
        store: chat's own artifact store, never the production one (jobs are staged and committed there).
        interpreters: environment name → its Python interpreter (from the site's environment table).
        device: the torch device the workers run on (``cuda`` or ``cpu``).
        timeout_s: the longest one job may take, a model load included; also the longest a job waits for a worker
            slot under ``max_resident``.
        max_resident: None: one worker per environment, each kept until the pool closes (a cluster deployment).
            N >= 1: one worker per recipe, at most N up at once; the least recently used idle one is stopped before
            another starts (module docstring).

    Raises:
        ValueError: for a ``max_resident`` below 1.
    """

    def __init__(
        self,
        store: ArtifactStore,
        interpreters: Mapping[str, str],
        device: str,
        timeout_s: float = 900.0,
        max_resident: int | None = None,
    ) -> None:
        if max_resident is not None and max_resident < 1:
            raise ValueError(f"max_resident must be at least 1 (or None for no limit), not {max_resident}")
        self.store, self.interpreters, self.device, self.timeout_s = store, dict(interpreters), device, timeout_s
        self.max_resident = max_resident
        self._workers: OrderedDict[str, _Worker] = OrderedDict()  # least recently used first
        self._cond = threading.Condition()  # job slots: users counts, starting and stopping workers
        self._index = threading.Lock()  # the worker map's structure, held briefly (status reads never wait on a stop)

    def _key(self, recipe: EncoderRecipe) -> str:
        return recipe.environment if self.max_resident is None else f"{recipe.environment}/{recipe.digest()}"

    def status(self, environment: str) -> Status:
        """The state of an environment's workers (``cold`` until its first job, or when every one was stopped)."""
        with self._index:
            states = {w.status for w in self._workers.values() if w.environment == environment}
        return next((s for s in ("starting", "warm", "error") if s in states), "cold")

    def warm(self, recipe: EncoderRecipe) -> bool:
        """Whether the worker that would encode ``recipe`` is up with its model loaded (no cold start)."""
        with self._index:
            worker = self._workers.get(self._key(recipe))
        return worker is not None and worker.status == "warm"

    def encode(self, recipe: EncoderRecipe, objects: Sequence[RawObject]) -> ArtifactRef:
        """Commit ``objects`` encoded by ``recipe`` as one embedding artifact (or find it committed); return it.

        Raises:
            EncoderError: if the recipe's environment has no interpreter, no worker slot frees up in time, or the
                worker or the job failed.
        """
        python = self.interpreters.get(recipe.environment)
        if python is None:
            raise EncoderError(f"no interpreter for environment {recipe.environment!r} (recipe {recipe.name!r})")
        worker = self._acquire(self._key(recipe), python, recipe)
        try:
            path, sha = stage_objects(self.store.root, objects)
            job = ExtractionJob(
                recipe.model_dump(mode="json"), str(path), sha, max(1, len(objects)), str(self.store.root)
            )
            return ArtifactRef(worker.run(job, self.timeout_s), EMBEDDINGS_KIND)
        finally:
            with self._cond:
                worker.users -= 1
                self._cond.notify_all()

    def _acquire(self, key: str, python: str, recipe: EncoderRecipe) -> _Worker:
        """The worker for ``key``, held for one job; under ``max_resident``, room made first (idle workers stopped)."""
        deadline = time.monotonic() + self.timeout_s
        with self._cond:
            while (worker := self._workers.get(key)) is None and not self._make_room():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EncoderError(
                        f"no encoder slot freed within {self.timeout_s:.0f} s for {recipe.name!r}: "
                        f"{len(self._workers)} resident encoder(s), all in use (max_resident {self.max_resident})"
                    )
                self._cond.wait(remaining)
            with self._index:
                if worker is None:
                    worker = self._workers[key] = _Worker(python, self.device, recipe.environment)
                self._workers.move_to_end(key)
            worker.users += 1
            return worker

    def _make_room(self) -> bool:
        """Stop least recently used idle workers until another fits; False if every resident one is in use."""
        if self.max_resident is None:
            return True
        while len(self._workers) >= self.max_resident:
            idle = next((k for k, w in self._workers.items() if w.users == 0), None)
            if idle is None:
                return False
            worker = self._workers[idle]
            sys.stderr.write(f"[chat] stopping the encoder worker {idle} to make room (max_resident)\n")
            try:
                worker.close()
            except OSError:
                if worker.process is not None:
                    worker.process.kill()
                    worker.process.wait(timeout=30)
            with self._index:
                self._workers.pop(idle)
        return True

    def close(self) -> None:
        """Stop every worker."""
        with self._index:
            workers = list(self._workers.values())
        for worker in workers:
            worker.close()
