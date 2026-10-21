"""Extension path 2b: a new encoder, registered as extractor ``example-kmer``.

Extractors are imported inside the encoder's own environment, so this module imports only the standard library,
numpy and the worker-side package modules.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np

from latent_interconnect.encoders.base import EXTRACTORS, ExtractedBatch, RawObject
from latent_interconnect.encoders.store import OUTPUT_LAYER

KMERS = ["".join(p) for p in itertools.product("ACGU", repeat=2)]


@EXTRACTORS.register("example-kmer")
class KmerExtractor:
    """Dinucleotide frequencies as a 16-d 'embedding' (no model; runs anywhere)."""

    def __init__(self, recipe: Mapping[str, object], device: str) -> None:
        self.recipe, self.device = recipe, device

    def load(self) -> None:
        """Nothing to load."""

    def fingerprint(self) -> Mapping[str, str]:
        """Identify the computation."""
        return {"extractor": "example-kmer", "k": "2", "dtype": "float32"}

    def extract(self, objects: Sequence[RawObject]) -> ExtractedBatch:
        """Return pooled dinucleotide frequencies, stored as the encoder's output (``read: output``)."""
        rows = []
        for o in objects:
            pairs = [o.value[i : i + 2] for i in range(len(o.value) - 1)]
            rows.append([pairs.count(k) / max(len(pairs), 1) for k in KMERS])
        return ExtractedBatch(tuple(o.object_id for o in objects), {OUTPUT_LAYER: np.asarray(rows, np.float32)}, {})
