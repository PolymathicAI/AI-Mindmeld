"""The built-in extractors, registered in ``EXTRACTORS`` on import (worker side: model libraries load lazily).

- ``esmc``: ESM-C 600M and 6B (``extractors.esmc``);
- ``mimic``: MIMIC's EVA register tokens (``extractors.mimic``);
- ``pglm``: xTrimoPGLM (ProteinGLM-MLM) 1B (``extractors.pglm``);
- ``molformer``, ``chemfm``: the SMILES models MoLFormer-c3 and ChemFM-1B (``extractors.molformer``,
  ``extractors.chemfm``);
- ``unimol2``: Uni-Mol2 through unimol_tools (``extractors.unimol2``);
- ``ecfp``: RDKit Morgan fingerprint bits, no model (``extractors.ecfp``; the cross-domain matched decoys).

``WindowedExtractor`` (``extractors._windows``) is re-exported here as the public base of a protein language model's
extractor: a third-party plugin subclasses it, loads its model in ``load`` and returns the read layers' states and the
residue mask from ``_forward`` (``examples/esm2_extractor.py``; ``examples/guides/extending.md`` §2).
"""

from latent_interconnect.encoders.extractors._windows import WindowedExtractor
from latent_interconnect.encoders.extractors.chemfm import ChemfmExtractor
from latent_interconnect.encoders.extractors.ecfp import EcfpExtractor
from latent_interconnect.encoders.extractors.esmc import EsmcExtractor
from latent_interconnect.encoders.extractors.mimic import MimicExtractor
from latent_interconnect.encoders.extractors.molformer import MolformerExtractor
from latent_interconnect.encoders.extractors.pglm import PglmExtractor
from latent_interconnect.encoders.extractors.unimol2 import Unimol2Extractor

__all__ = [
    "ChemfmExtractor",
    "EcfpExtractor",
    "EsmcExtractor",
    "MimicExtractor",
    "MolformerExtractor",
    "PglmExtractor",
    "Unimol2Extractor",
    "WindowedExtractor",
]
