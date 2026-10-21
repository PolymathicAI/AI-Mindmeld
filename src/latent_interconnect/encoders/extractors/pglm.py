"""PGLM extractor (``pglm``): pooled and per-residue states of xTrimoPGLM (ProteinGLM-MLM; worker side, lazy imports).

The procedure is the paper's: the model's own remote code at the pinned revision
(``AutoModelForMaskedLM`` with ``trust_remote_code``; transformers 4.57 in the ``esmc`` environment), a window costs its
residues plus ``<eos>`` (no ``<cls>``; right padding), and the windows, batches and residue-weighted mean pooling are
ESM-C's (``extractors._windows``). The model runs in bf16 on CUDA, float32 on CPU.

Layers: the model returns ``hidden_states`` sequence-first (``[T, n, d]``), ``num_layers + 1`` of them: index
``i < num_layers`` is the input of block ``i + 1`` (0 = the embeddings), and index ``num_layers`` is the last block's
output after the final LayerNorm, which replaces the un-normed one. So the paper's embedding (its
``return_last_hidden_state``) is layer ``num_layers`` (24 for the 1B model).

Recipe: ``read`` is a tuple of layers or ``"all"`` (``"output"`` is refused: name the final layer's index);
``pooling`` is ``mean`` or ``eos`` (``extractors._windows``; the special-token diagnostic);
``options["max_residues"]`` is required (1,023: the 1,024-token context minus ``<eos>``); ``residues`` adds per-residue
states as ESM-C's do.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from latent_interconnect.encoders.base import EXTRACTORS
from latent_interconnect.encoders.extractors._windows import WindowedExtractor

__all__ = ["PglmExtractor"]


@EXTRACTORS.register("pglm")
class PglmExtractor(WindowedExtractor):
    """xTrimoPGLM through its Hugging Face remote code (recipe checks: ``WindowedExtractor``)."""

    name = "pglm"
    specials = 1  # <eos> after every window
    library = "transformers"
    context = "<eos>"
    poolings = ("mean", "eos")
    model: Any

    def load(self) -> None:
        """Load the tokenizer and model (with their remote code) at the pinned revision."""
        import torch  # noqa: PLC0415 - the encoder environment's torch, imported only in the worker
        from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: PLC0415

        repo, revision = str(self.recipe["model_id"]), str(self.recipe["revision"])
        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            repo, revision=revision, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.padding_side != "right":
            raise ValueError(f"pglm: the tokenizer pads on the {self.tokenizer.padding_side}; the masks assume right")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        model = AutoModelForMaskedLM.from_pretrained(repo, revision=revision, trust_remote_code=True, dtype=dtype)
        self.model = model.to(self.device).eval()
        self._set_depth(int(self.model.config.num_layers))

    def _forward(self, seqs: list[str]) -> tuple[Sequence[Any], Any]:
        """Return the read layers' ``(n, T, d)`` states and the residue mask (``<eos>`` and padding out)."""
        import torch  # noqa: PLC0415

        enc = self.tokenizer(seqs, add_special_tokens=True, padding=True, return_tensors="pt")
        ids, mask = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        residue = mask.clone()
        residue[torch.arange(len(seqs)), mask.sum(1) - 1] = 0  # <eos>: the last real token under right padding
        return [out.hidden_states[L].transpose(0, 1) for L in self.layers], residue.bool()
