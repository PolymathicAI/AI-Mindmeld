"""A third-party encoder as a plugin: ESM-2 through Hugging Face ``transformers``, registered as extractor ``esm2``.

A worked example of ``examples/guides/extending.md`` §2: the recipe ``examples/configs/esm2-8m.yaml`` names this module
under ``plugins:`` (``examples.esm2_extractor``), so the extract stage and ``examples/embed.py`` import it before
resolving ``extractor: esm2``. It subclasses the public
:class:`~latent_interconnect.encoders.extractors.WindowedExtractor`, which holds the windows, batching, pooling and
fingerprint; only loading and the forward pass are ESM-2's. Layer ``i`` is ``hidden_states[i]`` (0 = embeddings; the
last, ``num_hidden_layers``, the final post-LayerNorm output). Like every extractor it imports only the standard
library, numpy and worker-side package modules at module level.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from latent_interconnect.encoders.base import EXTRACTORS
from latent_interconnect.encoders.extractors import WindowedExtractor


@EXTRACTORS.register("esm2")
class Esm2Extractor(WindowedExtractor):
    """ESM-2 (``facebook/esm2_*``) via ``transformers.EsmModel``, in float32."""

    name = "esm2"
    specials = 2  # <cls> and <eos> around every window
    library = "transformers"
    context = "<cls> and <eos>"
    poolings = ("mean", "cls", "eos")

    def load(self) -> None:
        """Load the tokenizer and model at the pinned revision."""
        import torch  # noqa: PLC0415  (model libraries load inside load)
        from transformers import AutoTokenizer, EsmModel  # noqa: PLC0415

        repo, revision = str(self.recipe["model_id"]), str(self.recipe["revision"])
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
        model = EsmModel.from_pretrained(repo, revision=revision, dtype=torch.float32, add_pooling_layer=False)
        self.model = model.to(self.device).eval()
        self._set_depth(int(self.model.config.num_hidden_layers))

    def _forward(self, seqs: list[str]) -> tuple[Sequence[Any], Any]:
        """Return the read layers' ``(n, T, d)`` states and the residue mask (``<cls>``, ``<eos>``, padding out)."""
        import torch  # noqa: PLC0415

        enc = self.tokenizer(seqs, return_tensors="pt", padding=True)
        ids, mask = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        residue = mask.clone()
        residue[:, 0] = 0  # <cls>
        residue[torch.arange(len(seqs)), mask.sum(1) - 1] = 0  # <eos>
        return [out.hidden_states[layer] for layer in self.layers], residue.bool()
