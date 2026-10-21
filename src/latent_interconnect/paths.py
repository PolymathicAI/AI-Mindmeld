"""Filesystem roots, taken from the environment: no absolute path is ever written in code.

Set ``LIP_DATA`` (and optionally the other ``LIP_*`` variables) to any directory. A site maps its own conventions onto
these variables in its own environment file. The code's own root is found, not set:
:func:`code_checkout` (and its state, :func:`code_state`; standard library only, so encoder workers record it too).
"""

from __future__ import annotations

import functools
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = ["NO_VCS", "MissingEnvironmentError", "Paths", "code_checkout", "code_state", "running_code"]

NO_VCS = "no-vcs"


class MissingEnvironmentError(RuntimeError):
    """A required environment variable is not set."""


@dataclass(frozen=True)
class Paths:
    """Roots of the on-disk layout.

    Attributes:
        data: root of all large data (``$LIP_DATA``).
        artifacts: content-addressed artifact store (``<data>/lip/artifacts``).
        records: result records (``<data>/lip/records``).
        runs: run-scoped state that outlives one job but is never an artifact: a training run's resume state lives
            under ``<runs>/<run config digest>/`` (``<data>/lip/runs``; ``LIP_RUNS`` overrides).
        scratch: per-job scratch space; jobs delete only paths they created.
        hf_home: Hugging Face cache for receiver and encoder weights.
    """

    data: Path
    artifacts: Path
    records: Path
    runs: Path
    scratch: Path
    hf_home: Path

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Paths:
        """Build the layout from environment variables.

        Raises:
            MissingEnvironmentError: if ``LIP_DATA`` is not set.
        """
        env = os.environ if environ is None else environ
        root = env.get("LIP_DATA")
        if not root:
            raise MissingEnvironmentError("set LIP_DATA to the data root (any writable directory)")
        data = Path(root)
        lip = data / "lip"
        return cls(
            data=data,
            artifacts=Path(env.get("LIP_ARTIFACTS", lip / "artifacts")),
            records=Path(env.get("LIP_RECORDS", lip / "records")),
            runs=Path(env.get("LIP_RUNS", lip / "runs")),
            scratch=Path(env.get("LIP_SCRATCH", lip / "scratch")),
            hf_home=Path(env.get("HF_HOME", data / "hf_cache")),
        )


def code_checkout() -> Path | None:
    """Return the git checkout whose ``src/`` holds this package, or None for an installed package."""
    package = Path(__file__).resolve().parent
    return next(
        (p for p in package.parents if (p / ".git").exists() and (p / "src" / package.name).resolve() == package), None
    )


def code_state(repo: Path | None) -> tuple[str, bool]:
    """Return the code tree's (commit, dirty), or ``(NO_VCS, False)`` for an installed package (``repo`` None)."""
    if repo is None:
        return NO_VCS, False
    return _git(repo, "rev-parse", "HEAD"), bool(_git(repo, "status", "--porcelain", "--untracked-files=all"))


@functools.cache
def running_code() -> tuple[str, bool]:
    """Return the (commit, dirty) of the code this process runs (:func:`code_checkout`), read once per process."""
    return code_state(code_checkout())


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()
