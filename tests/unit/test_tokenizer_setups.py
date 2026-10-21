"""``hf.load_tokenizer`` applies a loaded tokenizer's setup (``receivers.tokenizer_setups``) and imports no receiver
plugin to do so: loading Kimi K2.6's tokenizer imports neither compressed-tensors nor the ``kimi_k26`` backends. No hub
is read: transformers' loader is replaced by one that returns a stand-in of Kimi's tokenizer class (its module named as
the remote code's, ``…tokenization_kimi``).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import transformers

from latent_interconnect.config import load_config
from latent_interconnect.receivers import hf
from latent_interconnect.receivers.base import ReceiverSpec
from latent_interconnect.receivers.tokenizer_setups import KIMI_ENCODE_WARNING, setup_key

REPO = Path(__file__).resolve().parents[2]
KIMI = REPO / "configs" / "receivers" / "kimi-k2.6.yaml"
STAND_IN = "transformers_modules.stand_in.tokenization_kimi"
"""The module name of the stand-in tokenizer class: the remote code's module name, with any revision directory."""


def _stand_in() -> Any:
    return type("TikTokenTokenizer", (), {"__module__": STAND_IN})()


def test_load_tokenizer_applies_set_up_tokenizer(monkeypatch):
    loaded, set_up = _stand_in(), []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: loaded)
    monkeypatch.setattr(hf, "set_up_tokenizer", set_up.append)
    assert hf.load_tokenizer(load_config(KIMI, ReceiverSpec)) is loaded
    assert set_up == [loaded]


def test_only_the_kimi_class_gets_the_filter():
    assert setup_key(_stand_in()) == "tokenization_kimi.TikTokenTokenizer"
    other = type("TikTokenTokenizer", (), {"__module__": "transformers_modules.x.tokenization_other"})()
    hf.set_up_tokenizer(other)
    assert not logging.getLogger(type(other).__module__).filters


def test_a_tokenizer_only_load_imports_no_receiver_plugin():
    script = textwrap.dedent(
        f"""
        import json, logging, sys
        import transformers
        from latent_interconnect.config import load_config
        from latent_interconnect.receivers import hf
        from latent_interconnect.receivers.base import ReceiverSpec

        tok = type("TikTokenTokenizer", (), {{"__module__": {STAND_IN!r}}})()
        transformers.AutoTokenizer.from_pretrained = lambda *a, **k: tok
        hf.load_tokenizer(load_config(__import__("pathlib").Path({str(KIMI)!r}), ReceiverSpec))
        logger = logging.getLogger({STAND_IN!r})
        record = logger.makeRecord(logger.name, logging.WARNING, "f", 1, {KIMI_ENCODE_WARNING!r} + " {{}}", (), None)
        print(json.dumps({{
            "compressed_tensors": "compressed_tensors" in sys.modules,
            "kimi_k26": [m for m in sys.modules if m.startswith("latent_interconnect.receivers.kimi_k26")],
            "dropped": not logger.filter(record),
        }}))
        """
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO), os.environ.get("PYTHONPATH", "")]),
    }
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, cwd=REPO, env=env)
    facts = json.loads(out.stdout.strip().splitlines()[-1])
    assert facts == {"compressed_tensors": False, "kimi_k26": [], "dropped": True}


@pytest.mark.parametrize("message", ["another warning", "Calling encode"])
def test_the_filter_passes_other_records(message):
    from latent_interconnect.receivers.tokenizer_setups import quiet_kimi_encode_warning

    tok = _stand_in()
    quiet_kimi_encode_warning(tok)
    quiet_kimi_encode_warning(tok)
    logger = logging.getLogger(STAND_IN)
    assert len(logger.filters) == 1
    assert logger.filter(logger.makeRecord(logger.name, logging.WARNING, "f", 1, message, (), None))
