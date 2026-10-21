"""Kimi K2.6's tokenizer setup (``receivers.tokenizer_setups``): the remote tokenizer's per-encode warning is dropped
by a filter on its module's logger, and nothing else changes: the tokens of a set of renders are the same with and
without the filter, the logger's other warnings still pass, and a second load adds no second filter.

The tokenizer is the pinned snapshot's own remote code and vocabulary, from the Kimi CPU test setup
(``CONTRIBUTING.md``); no weights are read.
"""

# ruff: noqa: E402
# (the imports follow the optional-dependency skip)
from __future__ import annotations

import pytest

pytest.importorskip("compressed_tensors", reason="the Kimi CPU tests need compressed-tensors (CONTRIBUTING.md)")
pytestmark = pytest.mark.kimi_ep

import logging
from typing import Any

from latent_interconnect.receivers import open_renderer
from latent_interconnect.receivers.hf import set_up_tokenizer
from latent_interconnect.receivers.render import Conversation, Message, SlotBinding
from latent_interconnect.receivers.tokenizer_setups import KIMI_ENCODE_WARNING, _DropKimiEncodeWarning

TEXTS = ("A protein of 120 residues.", "Is it an enzyme? Answer Yes or No.", "Ångström · 3′-UTR — ünïcode", "")


def _filters(tok: Any) -> list[logging.Filter]:
    return [f for f in logging.getLogger(type(tok).__module__).filters if isinstance(f, _DropKimiEncodeWarning)]


def _renders(renderer: Any) -> list[tuple[int, ...]]:
    """The ids of a set of renders and of the pieces the renderer encodes (every encode passes a keyword)."""
    out = [tuple(renderer._encode_part(t)) for t in TEXTS]
    conversation = Conversation(messages=(Message(role="user", content="Describe <SLOT:p>. " + TEXTS[1]),))
    rendered = renderer.render(conversation, {"p": SlotBinding("o0", "c", 2)}, target=TEXTS[0])
    out.append(tuple(rendered.input_ids))
    out.append(tuple(rendered.target.token_ids))
    return out


def _load(spec: Any) -> Any:
    """The fixture's tokenizer as ``hf.load_tokenizer`` loads it, from its snapshot directory, then set up.

    ``load_tokenizer`` resolves the model id through transformers' own hub cache, which the tests do not point at the
    fixture's hub, so the tokenizer is loaded from the snapshot path (the same remote code and files) and
    ``hf.set_up_tokenizer``, the step ``load_tokenizer`` ends with, is applied to it.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download(spec.model_id, revision=spec.revision, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    set_up_tokenizer(tok)
    return tok


@pytest.fixture
def tokenizer(tiny):
    spec, _ = tiny
    return spec, _load(spec)


def test_the_setup_filters_exactly_the_tokenizer_modules_logger_once(tokenizer):
    spec, tok = tokenizer
    assert type(tok).__name__ == "TikTokenTokenizer"
    assert len(_filters(tok)) == 1
    again = _load(spec)  # a second load
    assert type(again).__module__ == type(tok).__module__
    assert len(_filters(tok)) == 1


def test_the_encode_warning_is_dropped_and_other_warnings_pass(tokenizer, caplog):
    _, tok = tokenizer
    logger = logging.getLogger(type(tok).__module__)
    with caplog.at_level(logging.WARNING, logger=logger.name):
        tok.encode(TEXTS[0], add_special_tokens=False)
        logger.warning("another warning from the tokenizer")
    messages = [r.getMessage() for r in caplog.records if r.name == logger.name]
    assert messages == ["another warning from the tokenizer"]
    (kept,) = _filters(tok)
    logger.removeFilter(kept)
    try:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=logger.name):
            tok.encode(TEXTS[0], add_special_tokens=False)
        assert next(r.getMessage() for r in caplog.records if r.name == logger.name).startswith(KIMI_ENCODE_WARNING)
    finally:
        logger.addFilter(kept)


def test_the_tokens_are_the_same_with_and_without_the_filter(tokenizer):
    spec, tok = tokenizer
    renderer = open_renderer(spec, tok)
    filtered = _renders(renderer)
    (kept,) = _filters(tok)
    logger = logging.getLogger(type(tok).__module__)
    logger.removeFilter(kept)
    try:
        unfiltered = _renders(renderer)
    finally:
        logger.addFilter(kept)
    assert filtered == unfiltered and all(filtered[:-3])
