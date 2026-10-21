"""Opt-in: existing ``kimi-ep`` training runs continue under this checkout's code only on the original speed options.

Set ``AIMM_RUN_DIRS`` to run directories (``Paths.runs / <run digest>``, holding a ``done.json`` or a ``resume.pt``),
separated by ``:`` (``os.pathsep``). Without it every test here is skipped. Each run's recorded ``execution`` (the
completed run's ``done.json``, else its resume state's) is fed through the trainer's own resume and relaunch check
(``Trainer._check_execution``) against what this checkout's ``kimi-ep`` records on that run's hardware: the recorded
hardware facts with the speed options as ``KimiEP.execution`` adds them (``ep.recorded_execution``). With every option
named at its original value the check must accept the run, with the input cache on or off, which needs the recorded
keys exactly; with the GPU defaults, and with each backend option alone at its fast value (its last), it must
refuse, naming the ``--option`` values that resume it.
No model is loaded.
"""

import json
import os
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("compressed_tensors", reason="kimi-ep's module needs compressed-tensors")

from latent_interconnect.receivers.kimi_k26.ep import (
    REFERENCE_OPTIONS,
    SPEED_OPTIONS,
    KimiEP,
    recorded_execution,
    resolve_options,
)
from latent_interconnect.training.trainer import DONE_FILE, Trainer
from latent_interconnect.writes.checkpoint import read_resume_summary

DIRS = [Path(p) for p in os.environ.get("AIMM_RUN_DIRS", "").split(os.pathsep) if p]

pytestmark = pytest.mark.skipif(not DIRS, reason="set AIMM_RUN_DIRS to run directories")


def _recorded(directory: Path) -> dict[str, str]:
    if (directory / DONE_FILE).exists():
        execution = json.loads((directory / DONE_FILE).read_text())["execution"]
    else:
        summary = read_resume_summary(directory)
        assert summary is not None, f"{directory} holds no {DONE_FILE} and no resume state"
        execution = summary.extra["execution"]
    if execution.get("backend") != "kimi-ep":
        pytest.skip(f"{directory.name[:12]} ran {execution.get('backend')!r}, not kimi-ep")
    return {str(k): str(v) for k, v in execution.items()}


def _check(recorded: dict[str, str], options: dict[str, str], input_cache: str = "off") -> None:
    backend = types.SimpleNamespace(
        execution=lambda: recorded_execution(recorded, options), reference_options=KimiEP.reference_options
    )
    trainer: Any = types.SimpleNamespace(
        config=types.SimpleNamespace(name=recorded["backend"]), backend=backend, input_cache=input_cache
    )
    trainer.execution = lambda: Trainer.execution(trainer)
    Trainer._check_execution(trainer, recorded)


@pytest.mark.parametrize("directory", DIRS, ids=[d.name[:12] for d in DIRS])
def test_an_existing_kimi_ep_run_continues_only_on_the_original_options(directory):
    recorded = _recorded(directory)
    assert not set(REFERENCE_OPTIONS) & set(recorded)  # recorded before the speed options
    _check(recorded, resolve_options(cuda=True, **REFERENCE_OPTIONS))
    _check(recorded, resolve_options(cuda=True, **REFERENCE_OPTIONS), "on")
    resume = " ".join(f"--option {k}={v}" for k, v in REFERENCE_OPTIONS.items())
    mixed = [
        resolve_options(cuda=True, **{**REFERENCE_OPTIONS, name: SPEED_OPTIONS[name][-1]}) for name in REFERENCE_OPTIONS
    ]
    for options in (resolve_options(cuda=True), *mixed):
        for input_cache in ("on", "off"):
            with pytest.raises(ValueError) as refused:
                _check(recorded, options, input_cache)
            assert str(refused.value).endswith(f"resume with the options the run started with: {resume}")
