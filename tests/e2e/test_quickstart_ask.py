"""The Quickstart's ask path in miniature (examples/README.md steps 3-4): download a connector by name, ask by name.

A tiny connector of the e2e chain's receiver is packaged (``bundle_connector``), installed by ``release_download``
from a local source, and ``examples/ask.py`` is run as the Quickstart runs it, with the connector's name in place of
its checkpoint directory. The answer is the one the checkpoint directory gives.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from tests.fakes import build_tiny_llama
from tests.unit.test_bundle_connector import D_IN, _commit

from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.bundle_connector import package
from latent_interconnect.config import dump_config
from latent_interconnect.release_download import main as download_main

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
NAME = "tiny-llama__toy-linear"


def test_the_quickstart_asks_an_installed_connector_by_name(tmp_path):
    configs, store_root, ref, _ = _commit(tmp_path)
    repo = tmp_path / "hub"
    package(
        ArtifactStore(store_root).path(ref),
        configs / "train_affine.yaml",
        repo / "connectors",
        license="MIT",
        pin="c/x",
    )
    dest = tmp_path / "aimm-data"
    assert download_main(["--local-source", str(repo), "--connector", NAME, "--dest", str(dest)]) == 0
    receiver = tmp_path / "tiny-llama-local.yaml"
    dump_config(build_tiny_llama(tmp_path / "tiny-llama"), receiver)
    work = tmp_path / "work"
    work.mkdir()
    np.save(work / "protein.npy", np.ones(D_IN, np.float32))
    env = os.environ | {"LIP_DATA": str(dest)}
    ask = [sys.executable, str(ROOT / "examples" / "ask.py"), str(receiver)]
    question = ["protein.npy", "What does the protein <SLOT:x> do?", "--max-new-tokens", "4"]
    by_name = subprocess.run([*ask, NAME, *question], capture_output=True, text=True, cwd=work, env=env, check=False)
    assert by_name.returncode == 0, by_name.stderr
    assert f"(bundle {NAME}): kind affine, reads pooled" in by_name.stderr
    checkpoint = str(ArtifactStore(dest / "lip" / "artifacts").path(ref))
    by_path = subprocess.run(
        [*ask, checkpoint, *question], capture_output=True, text=True, cwd=work, env=env, check=False
    )
    assert by_path.returncode == 0, by_path.stderr
    assert by_name.stdout == by_path.stdout and by_name.stdout.strip()
