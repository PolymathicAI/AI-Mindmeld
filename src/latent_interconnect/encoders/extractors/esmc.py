"""ESM-C extractor (``esmc``): pooled and per-residue states of ESM-C 600M / 6B (worker side; ``esm`` imported in load).

The procedure is the paper's, shared with PGLM (``extractors._windows``): windows of
at most ``options["max_residues"]`` residues, length-sorted batches under ``batch_tokens`` (each window costs its
residues plus ``<cls>`` and ``<eos>``), residue-weighted mean pooling.

- the model is loaded in bf16 on CUDA (float32 on CPU) with SDPA attention, as the original 6B extractor loaded it
  (``EsmcModel.from_pretrained(..., dtype=...)``). Both checkpoints are float32; the
  original 600M extractor cast after loading, which gives bit-identical bf16 parameters and buffers (checked on 600M).
  Pooling accumulates in float64 on the device (``extractors._windows``);
- layer ``i`` is ``hidden_states[i]`` of ``EsmcModel`` (0 = embeddings, ``i`` = the output of block ``i``, the raw
  residual stream; the last index is the final post-LayerNorm output), so the paper's final-layer embedding is
  layer ``num_hidden_layers``: 36 for 600M, 80 for 6B (whose all-layer scan reads 0..80; the original layer sweep's
  extra pre-norm state of block 80 is not a ``hidden_states`` index and is not read).

Recipe: ``read`` is a tuple of layers or ``"all"`` (``"output"`` is refused: name the final layer's index);
``pooling`` is ``mean``, ``cls`` or ``eos`` (``extractors._windows``); ``residues`` adds per-residue states (bf16
stored as ``uint16`` bit patterns, or float32), concatenated over windows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from latent_interconnect.encoders.base import EXTRACTORS
from latent_interconnect.encoders.extractors._windows import WindowedExtractor

__all__ = ["EsmcExtractor"]


@EXTRACTORS.register("esmc")
class EsmcExtractor(WindowedExtractor):
    """ESM-C through the ``esm`` package's Hugging Face-style ``EsmcModel`` (recipe checks: ``WindowedExtractor``)."""

    name = "esmc"
    specials = 2  # <cls> and <eos> around every window
    library = "esm"
    context = "<cls> and <eos>"
    poolings = ("mean", "cls", "eos")
    model: Any

    def load(self) -> None:
        """Load the tokenizer and model at the pinned revision."""
        import torch  # noqa: PLC0415 - the encoder environment's torch, imported only in the worker
        from esm.models.esmc.model import EsmcModel  # noqa: PLC0415
        from esm.models.esmc.tokenizer import EsmcTokenizer  # noqa: PLC0415

        repo, revision = str(self.recipe["model_id"]), str(self.recipe["revision"])
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = EsmcTokenizer.from_pretrained(repo, revision=revision)
        self.model = EsmcModel.from_pretrained(
            repo, revision=revision, device=self.device, dtype=dtype, attn_implementation="sdpa"
        ).eval()
        self._set_depth(int(self.model.config.num_hidden_layers))

    def _forward(self, seqs: list[str]) -> tuple[Sequence[Any], Any]:
        """Return the read layers' ``(n, T, d)`` states and the residue mask (``<cls>``, ``<eos>``, padding out)."""
        import torch  # noqa: PLC0415

        enc = self.tokenizer(seqs, return_tensors="pt", padding=True)
        ids, mask = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        residue = mask.clone()
        residue[:, 0] = 0
        residue[torch.arange(len(seqs)), mask.sum(1) - 1] = 0
        return [out.hidden_states[L] for L in self.layers], residue.bool()
