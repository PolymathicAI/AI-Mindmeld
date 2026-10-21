"""Opt-in: existing trained connector checkpoints load with this checkout's code, unchanged.

Set ``AIMM_CHECKPOINT_DIRS`` to checkpoint artifact directories (``connector.json``, ``weights.safetensors``,
``MANIFEST.json``), separated by ``:`` (``os.pathsep``). Without it every test here is skipped. Each directory is read
with :func:`latent_interconnect.writes.checkpoint.read_checkpoint` on CPU; the checks are identity evidence (the state
is the stored tensors, the spec digest and the manifest digest are the recorded ones), not a run of the model.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from latent_interconnect.artifacts.store import MANIFEST_NAME, Manifest
from latent_interconnect.writes.checkpoint import read_checkpoint

DIRS = [Path(p) for p in os.environ.get("AIMM_CHECKPOINT_DIRS", "").split(os.pathsep) if p]

pytestmark = pytest.mark.skipif(not DIRS, reason="set AIMM_CHECKPOINT_DIRS to checkpoint directories")


@pytest.mark.parametrize("directory", DIRS, ids=[d.name[:12] for d in DIRS])
def test_an_existing_checkpoint_loads_unchanged(directory):
    manifest = Manifest.load(directory / MANIFEST_NAME)
    assert manifest.kind == "checkpoint"
    assert manifest.sha256() == directory.name  # the artifact id is its manifest's digest

    connector, step = read_checkpoint(directory)
    assert step == json.loads((directory / "connector.json").read_text())["step"]
    assert connector.spec.digest() == manifest.meta["connector_sha"]

    raw = load_file(str(directory / "weights.safetensors"))
    state = connector.state_dict()
    assert sorted(state) == sorted(raw)
    for key, tensor in raw.items():
        assert state[key].dtype == tensor.dtype, key
        assert torch.equal(state[key].cpu(), tensor), key
