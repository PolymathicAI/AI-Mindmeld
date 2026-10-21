"""The one test-side definition of "a receiver's local snapshot is available", the same for every receiver.

Tests that read a receiver's real tokenizer, config or remote code (no weights) use :func:`requires` or
:func:`snapshot`. Without the snapshot they skip with :func:`reason`, and ``pytest --require-snapshots`` (the merge
gate, CONTRIBUTING.md) turns every such skip into a failure.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
REASON = "needs the receiver's local snapshot (HF_HOME; examples/README.md)"


def snapshot(receiver: str) -> Path | None:
    """Return the cached snapshot of ``configs/receivers/<receiver>.yaml``'s pinned revision (``$HF_HOME``), or None."""
    home = os.environ.get("HF_HOME")
    if not home:
        return None
    spec = yaml.safe_load((REPO / "configs" / "receivers" / f"{receiver}.yaml").read_text())
    snap = Path(home) / "hub" / ("models--" + spec["model_id"].replace("/", "--")) / "snapshots" / spec["revision"]
    return snap if (snap / "config.json").is_file() and (snap / "tokenizer_config.json").is_file() else None


def reason(receiver: str) -> str:
    """Return the skip reason of a test that needs ``receiver``'s snapshot (it contains :data:`REASON`)."""
    return f"{REASON}: {receiver}"


def requires(receiver: str) -> pytest.MarkDecorator:
    """Mark a test that needs ``receiver``'s snapshot: it skips without it (:func:`reason`)."""
    return pytest.mark.skipif(snapshot(receiver) is None, reason=reason(receiver))
