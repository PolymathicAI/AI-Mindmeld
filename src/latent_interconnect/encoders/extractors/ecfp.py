"""ECFP extractor (``ecfp``): RDKit Morgan fingerprint bits as a molecule's output vector (worker side, no model).

The frozen cross-domain design ranks held-out metabolites by ECFP4 Tanimoto to a positive's metabolite
(``build.cross_domain_items``: the matched decoys): RDKit's Morgan generator
at ``options["radius"]`` (2 for ECFP4) and ``options["bits"]`` (2,048), the SMILES parsed with RDKit's defaults. Each
molecule's bits are stored as a float32 0/1 vector at :data:`~latent_interconnect.encoders.store.OUTPUT_LAYER`
(``read: output``), so the extract stage commits and records them like any encoder's states and the builder computes
Tanimoto from the vectors. A SMILES RDKit cannot parse is skipped with its reason. It runs in the ``smallmol``
environment, where RDKit is installed (passed to the extract stage as ``--option python=...``).
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping, Sequence

import numpy as np

from latent_interconnect.encoders.base import EXTRACTORS, ExtractedBatch, RawObject
from latent_interconnect.encoders.store import OUTPUT_LAYER

__all__ = ["EcfpExtractor"]


@EXTRACTORS.register("ecfp")
class EcfpExtractor:
    """Morgan fingerprint bits of each SMILES (the recipe's ``radius`` and ``bits`` options)."""

    def __init__(self, recipe: Mapping[str, object], device: str) -> None:
        self.recipe, self.device = recipe, device
        options = recipe.get("options")
        if not isinstance(options, Mapping) or set(options) != {"radius", "bits"}:
            raise ValueError(f"the ecfp recipe's options are radius and bits, got {options!r}")
        self.radius, self.bits = int(options["radius"]), int(options["bits"])
        if recipe.get("read") != "output":
            raise ValueError(f"the ecfp extractor has only the output read, not {recipe.get('read')!r}")

    def load(self) -> None:
        """Import RDKit and build the generator."""
        from rdkit import Chem, RDLogger  # noqa: PLC0415 - the smallmol environment's RDKit, imported only here
        from rdkit.Chem import rdFingerprintGenerator  # noqa: PLC0415

        RDLogger.DisableLog("rdApp.*")
        self._parse = Chem.MolFromSmiles
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.bits)

    def fingerprint(self) -> Mapping[str, str]:
        """Return the RDKit version, the radius and the width."""
        return {
            "extractor": "ecfp",
            "rdkit": importlib.metadata.version("rdkit"),
            "radius": str(self.radius),
            "bits": str(self.bits),
            "dtype": "float32",
        }

    def extract(self, objects: Sequence[RawObject]) -> ExtractedBatch:
        """Return each parsable SMILES's bits as a float32 0/1 vector; skip the others."""
        ids, rows, skipped = [], [], {}
        for o in objects:
            mol = self._parse(o.value) if o.value else None
            if mol is None:
                skipped[o.object_id] = "RDKit cannot parse the SMILES"
                continue
            ids.append(o.object_id)
            rows.append(np.asarray(self._generator.GetFingerprintAsNumPy(mol), dtype=np.float32))
        pooled = np.stack(rows) if rows else np.zeros((0, self.bits), np.float32)
        return ExtractedBatch(tuple(ids), {OUTPUT_LAYER: pooled}, {}, skipped=skipped)
