"""Extractors: the per-encoder code that turns raw objects into hidden states (dependency-light, Python 3.11).

An extractor wraps one specialist model and returns, for a batch of raw objects, the states at the requested
layers: one pooled vector per object and layer, and optionally the per-residue states. Extractors run inside
*encoder environments*, the encoder's own Python environment (the extract stage's ``--option python=...``). So
this module, ``store`` and ``jobs`` import only the standard library and numpy; an extractor imports its model library
lazily in :meth:`Extractor.load`.

Layer convention (both sides of the system): layer ``i`` is ``hidden_states[i]``: 0 = the
embedding output, ``i`` = the output of block ``i``, and ``D`` = the final output after the final norm when the model
applies one. ``read = "output"`` means the encoder's defined output (MIMIC's 5 register tokens; a molecule encoder's
summary vector).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from latent_interconnect.plugins import Registry

__all__ = ["EXTRACTORS", "ExtractedBatch", "Extractor", "RawObject"]


@dataclass(frozen=True)
class RawObject:
    """One object to embed.

    Attributes:
        object_id: stable id within its domain (UniProt accession, ChEBI id, ...).
        value: the raw input (amino-acid sequence, SMILES, ...).
        extra: extractor-specific fields.
    """

    object_id: str
    value: str
    extra: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ExtractedBatch:
    """States for a batch of objects.

    Attributes:
        object_ids: batch order.
        pooled: layer → ``(B, d)`` float32 pooled vectors.
        residues: layer → per-object ``(L_i, d)`` arrays (bf16 stored as ``uint16`` views, or float32), or empty.
        means: layer → ``(B, d)`` float32 means over a capture's read tokens (the ridge bias), or empty: captures
            only (``build.capture``), at the same layers as ``pooled``.
        skipped: object id → reason, for the batch's objects the extractor could not embed on their own (a bad
            input, not a failed run); they are absent from ``object_ids``, and the artifact lists and counts them.
        notes: object id → how the extractor computed it, for embedded objects that left the extractor's default
            route (Uni-Mol2's opt-in conformer retry); the artifact lists and counts them, and has no ``notes`` when
            no object has one.
    """

    object_ids: tuple[str, ...]
    pooled: Mapping[int, np.ndarray]
    residues: Mapping[int, Sequence[np.ndarray]]
    means: Mapping[int, np.ndarray] = field(default_factory=dict)
    skipped: Mapping[str, str] = field(default_factory=dict)
    notes: Mapping[str, str] = field(default_factory=dict)


class Extractor(Protocol):
    """One encoder's extraction code. Constructed as ``cls(recipe: dict, device: str)``."""

    def load(self) -> None:
        """Load the model (and import its library). Called once per worker process."""
        ...

    def fingerprint(self) -> Mapping[str, str]:
        """Return what identifies the computation: model revision, weights sha256, library versions, pooling code.

        It must hold ``dtype``, the precision the model actually ran in (native fast precision, declared and
        reported); the job refuses a fingerprint without it, and the artifact and the extract record carry it.
        """
        ...

    def extract(self, objects: Sequence[RawObject]) -> ExtractedBatch:
        """Return the recipe's layers for a batch of objects (pooled always; residues if the recipe asks)."""
        ...


EXTRACTORS: Registry[type] = Registry("extractor")
"""Extractor classes by name (the package registers the six paper encoders and ESM-C 6B; third parties add theirs).

The receiver's own states (ridge targets) are not an extractor: they are captured by ``build.capture`` through the
renderer and a capture backend, and written in the same store format."""
