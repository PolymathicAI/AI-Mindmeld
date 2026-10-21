"""The canonical JSON form: the one rule behind content identities (stdlib only, so worker-side code may use it).

Sorted keys, compact separators (``,`` and ``:``), and non-ASCII characters written as themselves (UTF-8 when
encoded), never as ASCII escapes. Config digests (``config.Config.digest``), artifact manifests
(``artifacts.store``), item-set bodies (``data.items``) and batch-partition digests (``receivers.kernel``) all
serialize through :func:`canonical_json`, so two spellings of the rule cannot drift apart.

One declared exception: an embedding job's key and its fingerprint text (``encoders.jobs``) keep their older form
(sorted keys, default separators). ``store.committed(job.key())`` finds committed extractions by that key, so moving
it would orphan them and recompute; a migration would need a declared re-key or an alias lookup.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["SHA256_HEX_LENGTH", "canonical_json", "canonical_sha256", "is_sha256"]

SHA256_HEX_LENGTH = 64
"""Length of a sha256 hex digest (artifact digests, record computation keys)."""


def canonical_json(obj: Any) -> str:
    """Serialize a JSON-compatible object deterministically (sorted keys, compact separators, non-ASCII as is)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(obj: Any) -> str:
    """Return the sha256 hex digest of the UTF-8 bytes of :func:`canonical_json` of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def is_sha256(value: object) -> bool:
    """Return whether ``value`` is a lowercase sha256 hex digest."""
    return isinstance(value, str) and len(value) == SHA256_HEX_LENGTH and all(c in "0123456789abcdef" for c in value)
