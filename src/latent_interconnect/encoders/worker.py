"""Encoder worker entry point: ``python -m latent_interconnect.encoders.worker <job.json> [--device D] [--shard I/N]``.

``--handoff H.json`` passes a declared origin handoff (``ArtifactStore.require_origin``). Runs inside the encoder's
own environment (standard library + numpy at import; the extractor imports its model library in ``load``). Prints the
number of parts it wrote and, when it committed the artifact, the artifact digest.

``--serve`` keeps the worker resident instead (live encoding, ``latent_interconnect.chat``): it reads one
:class:`~latent_interconnect.encoders.jobs.ExtractionJob` per line of standard input (its fields as a JSON object),
runs it whole and commits it as the one-shot worker does (``run_worker``, ``finalize``: an ordinary embedding artifact
in the job's store, found again by its key), loading each recipe's extractor once and keeping it. Each job is answered
by one line on standard output, :data:`REPLY` and a JSON object: ``{"embeddings": <sha256>}`` or ``{"error": <text>}``;
the first such line, ``{"ready": true}``, says the worker is reading jobs. Other output lines (a library's) carry no
prefix and are not replies.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import IO, Any

from latent_interconnect.artifacts.store import read_handoff
from latent_interconnect.encoders.base import Extractor
from latent_interconnect.encoders.jobs import ExtractionJob, finalize, load_extractor, missing_parts, run_worker

__all__ = ["REPLY", "main", "serve"]

REPLY = "lip-encoder-reply "
"""The prefix of a resident worker's reply lines."""


def _reply(out: IO[str], body: dict[str, Any]) -> None:
    out.write(REPLY + json.dumps(body, sort_keys=True) + "\n")
    out.flush()


def serve(device: str, lines: Iterable[str], out: IO[str]) -> int:
    """Run jobs, one per line, until the input ends; reply to each (module docstring). Return the exit code."""
    loaded: dict[str, Extractor] = {}
    _reply(out, {"ready": True})
    for line in lines:
        if not line.strip():
            continue
        try:
            job = ExtractionJob(**json.loads(line))
            key = json.dumps(job.recipe, sort_keys=True)
            if key not in loaded:
                loaded[key] = load_extractor(job.recipe, device)
            run_worker(job, device, extractor=loaded[key])
            _reply(out, {"embeddings": finalize(job).sha256})
        except Exception as err:  # one job's failure is its reply; the worker stays up
            _reply(out, {"error": f"{type(err).__name__}: {err}"})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one worker; return the process exit code."""
    parser = argparse.ArgumentParser(prog="python -m latent_interconnect.encoders.worker")
    parser.add_argument("job", type=Path, nargs="?", help="the job JSON (ExtractionJob.dump)")
    parser.add_argument("--device", default="cuda", help="torch device string")
    parser.add_argument("--shard", default="0/1", help="I/N: take the parts whose index is I modulo N")
    parser.add_argument("--deadline-s", type=float, default=None, help="stop starting parts after this many seconds")
    parser.add_argument("--handoff", type=Path, default=None, help="a declared origin handoff (JSON)")
    parser.add_argument("--serve", action="store_true", help="stay resident: jobs from stdin, one JSON per line")
    args = parser.parse_args(argv)
    if args.serve:
        return serve(args.device, sys.stdin, sys.stdout)
    if args.job is None:
        parser.error("a job JSON is needed (or --serve)")
    index, sep, count = str(args.shard).partition("/")
    if not (sep and index.isdigit() and count.isdigit()):
        parser.error(f"--shard {args.shard!r}: expected I/N")
    job = ExtractionJob.load(args.job)
    written = run_worker(job, args.device, (int(index), int(count)), args.deadline_s, read_handoff(args.handoff))
    sys.stdout.write(f"wrote {written} part(s)\n")
    missing = missing_parts(job)
    if missing:
        sys.stdout.write(f"{len(missing)} part(s) left for other workers\n")
        return 0
    sys.stdout.write(f"embeddings: {finalize(job).sha256}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
