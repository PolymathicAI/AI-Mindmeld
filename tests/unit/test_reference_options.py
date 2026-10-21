"""The reference-options names (``ep.REFERENCE_OPTIONS``, ``TrainingBackend.reference_options``) and the scope of the
GPU tests' cuBLAS setting: the old names are gone everywhere in the package, the modules that use the new ones import,
and importing the GPU test module does not set ``CUBLAS_WORKSPACE_CONFIG``. Names only: the option values, defaults and
execution facts are the tests' of ``test_kimi_ep_options``, ``test_input_cache`` and ``test_existing_runs``.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from latent_interconnect.receivers.base import TrainingBackend

REPO = Path(__file__).resolve().parents[2]
ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO), os.environ.get("PYTHONPATH", "")]),
}


def test_the_reference_options_replace_the_legacy_names():
    pytest.importorskip("compressed_tensors", reason="kimi-ep's module needs compressed-tensors")
    ep = importlib.import_module("latent_interconnect.receivers.kimi_k26.ep")
    assert "REFERENCE_OPTIONS" in ep.__all__ and isinstance(ep.REFERENCE_OPTIONS, type(ep.CUDA_DEFAULTS))
    assert not hasattr(ep, "LEGACY_OPTIONS")
    assert ep.KimiEP.reference_options is ep.REFERENCE_OPTIONS
    assert "reference_options" in TrainingBackend.__annotations__
    assert "legacy_options" not in TrainingBackend.__annotations__
    old = re.compile(r"\blegacy_options\b|\bLEGACY_OPTIONS\b")
    hits = [str(f) for f in (REPO / "src").rglob("*.py") if old.search(f.read_text())]
    assert hits == []


def test_the_modules_import_and_hf_exports_set_up_tokenizer():
    for name in ("latent_interconnect.receivers.hf", "latent_interconnect.training.trainer"):
        importlib.import_module(name)
    hf = importlib.import_module("latent_interconnect.receivers.hf")
    assert "set_up_tokenizer" in hf.__all__
    pytest.importorskip("compressed_tensors", reason="kimi-ep's module needs compressed-tensors")
    importlib.import_module("latent_interconnect.receivers.kimi_k26.ep")


def test_the_gpu_tests_set_cublas_workspace_config_within_their_block_only():
    """Importing ``tests/gpu/test_kimi_ep_cuda.py`` does not set ``CUBLAS_WORKSPACE_CONFIG``; its deterministic block
    sets ``CUBLAS_WORKSPACE_CONFIG`` inside the block and removes it after (in a subprocess without the variable)."""
    pytest.importorskip("compressed_tensors", reason="the Kimi test modules need compressed-tensors")
    script = textwrap.dedent(
        """
        import json, os
        import pytest, torch
        import tests.gpu.test_kimi_ep_cuda as gpu
        facts = {"after_import": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
        with pytest.MonkeyPatch.context() as mp, gpu.deterministic_cublas(mp):
            facts["inside"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            facts["deterministic"] = torch.are_deterministic_algorithms_enabled()
        facts["after"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        facts["deterministic_after"] = torch.are_deterministic_algorithms_enabled()
        print(json.dumps(facts))
        """
    )
    env = {k: v for k, v in ENV.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=REPO, env=env)
    facts = json.loads(out.stdout.strip().splitlines()[-1])
    assert facts == {
        "after_import": None,
        "inside": ":4096:8",
        "deterministic": True,
        "after": None,
        "deterministic_after": False,
    }
