"""Setups of a loaded tokenizer, by its class (:data:`SETUPS`), applied by ``hf.load_tokenizer``: logging only.

This module imports only ``logging``, so a tokenizer-only load imports no receiver plugin (a receiver's plugin package,
e.g. ``kimi_k26``, registers its backends when imported and needs their dependencies). A setup changes no token.

**Kimi K2.6** (:func:`quiet_kimi_encode_warning`). Kimi's remote ``TikTokenTokenizer.encode``
(``tokenization_kimi.py``) logs a warning, ``"Calling super().encode with {kwargs}"``, on every call that passes any
keyword argument, then calls ``super().encode``. The renderer passes one on every encode (``add_special_tokens=False``),
so a training update logs the warning once per rendered piece (hundreds of lines per update), with the same content
every time and no information, into job logs that are flushed per line. The setup adds a filter to exactly that
module's logger that drops only these records: the logger's level, its other records and the encode call (its
arguments, its path through ``super().encode``, its tokens) are unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

__all__ = ["KIMI_ENCODE_WARNING", "SETUPS", "quiet_kimi_encode_warning", "setup_key"]

KIMI_ENCODE_WARNING = "Calling super().encode with"
"""The start of Kimi's remote ``encode`` warning message."""


class _DropKimiEncodeWarning(logging.Filter):
    """Drop the records whose message starts with :data:`KIMI_ENCODE_WARNING`; pass every other one."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(KIMI_ENCODE_WARNING)


def quiet_kimi_encode_warning(tokenizer: Any) -> None:
    """Add the filter to the logger of the loaded tokenizer class's module, once (a later load adds none)."""
    logger = logging.getLogger(type(tokenizer).__module__)
    if not any(isinstance(f, _DropKimiEncodeWarning) for f in logger.filters):
        logger.addFilter(_DropKimiEncodeWarning())


def setup_key(tokenizer: Any) -> str:
    """A tokenizer class's key in :data:`SETUPS`: its module's last component and its name (no revision in it)."""
    cls = type(tokenizer)
    return f"{cls.__module__.rsplit('.', 1)[-1]}.{cls.__qualname__}"


SETUPS: dict[str, Callable[[Any], None]] = {"tokenization_kimi.TikTokenTokenizer": quiet_kimi_encode_warning}
"""Each tokenizer class that needs a setup (:func:`setup_key`) and its setup; other tokenizers get none."""
