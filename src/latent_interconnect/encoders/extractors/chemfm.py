"""ChemFM extractor (``chemfm``): ChemFM-1B or -3B, LLaMA-architecture causal SMILES models (worker side, lazy imports).

The procedure is the paper's: ``AutoTokenizer`` and
``AutoModel`` (the ``LlamaModel`` body) at the pinned revision (the paper's run loaded the unpinned head and recorded
this revision); bf16 on CUDA, float32 on CPU, pooling from float32 states. The SMILES models' shared rules (one
molecule at a time, truncation, layers, the molecule's tokens, skips) are
:class:`~latent_interconnect.encoders.extractors._molecules.SmilesExtractor`'s.

Layers: ``hidden_states[L]`` for ``L < D`` (0 = the embeddings), and layer ``D = num_hidden_layers`` is
``last_hidden_state``, after the final RMSNorm. The tokenizer adds no special tokens, so the mean is over every token
and ``last`` is the last SMILES token (the causal model's summary), ``cls`` the first. The paper's input,
``last+mean`` at ``read = "output"``, is ``chemfm_1b_cat``. ChemFM is CC BY-NC 4.0.
"""

from __future__ import annotations

from collections.abc import Mapping

from latent_interconnect.encoders.base import EXTRACTORS
from latent_interconnect.encoders.extractors._molecules import SmilesExtractor

__all__ = ["ChemfmExtractor"]


@EXTRACTORS.register("chemfm")
class ChemfmExtractor(SmilesExtractor):
    """ChemFM through transformers' LLaMA implementation."""

    name = "chemfm"

    def load(self) -> None:
        """Load the tokenizer and model at the pinned revision."""
        import torch  # noqa: PLC0415 - the encoder environment's torch, imported only in the worker
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        repo, revision = str(self.recipe["model_id"]), str(self.recipe["revision"])
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)  # type: ignore[no-untyped-call]
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(repo, revision=revision, dtype=dtype).to(self.device).eval()
        self._set_depth(int(self.model.config.num_hidden_layers))

    def fingerprint(self) -> Mapping[str, str]:
        """Return the model id and revision, the library versions, the dtype, the layers and the pooling."""
        return self._fingerprint("bfloat16" if self.device.startswith("cuda") else "float32")
