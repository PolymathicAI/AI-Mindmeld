"""MoLFormer extractor (``molformer``): DeepChem's MoLFormer-c3 weights run with IBM's MoLFormer code (worker side).

The procedure is the paper's: the weights and tokenizer
(``tokenizer.json`` with ``<bos>``, ``<eos>``, ``<pad>``) at ``model_id`` / ``revision``; the model classes from
``options["code_model_id"]`` at ``options["code_revision"]`` (the weights' own config points at IBM's code without a
revision, so the code is pinned separately); ``deterministic_eval`` (fixed random features); float32 on any device.
The SMILES models' shared rules (one molecule at a time, truncation, layers, the molecule's tokens, skips) are
:class:`~latent_interconnect.encoders.extractors._molecules.SmilesExtractor`'s.

Layers: ``hidden_states[L]`` for ``L < D`` (0 = the embeddings), and layer ``D = num_hidden_layers`` is
``last_hidden_state``, the last block's output after the model's final LayerNorm (``hidden_states[D]`` is before it).
``cls`` is ``<bos>`` (position 0), ``last`` is ``<eos>``; the mean excludes ``<bos>``, ``<eos>`` and padding
(MoLFormer's own ``pooler_output`` would include them). The paper's input, ``cls+mean`` at ``read = "output"``, is
``molformer_c3_cat``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from latent_interconnect.encoders.base import EXTRACTORS
from latent_interconnect.encoders.extractors._molecules import SmilesExtractor

__all__ = ["MolformerExtractor"]

_SPECIAL_TOKENS = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>"}


@EXTRACTORS.register("molformer")
class MolformerExtractor(SmilesExtractor):
    """MoLFormer through IBM's Hugging Face remote code; ``options`` also needs ``code_model_id``, ``code_revision``."""

    name = "molformer"
    required = frozenset({"code_model_id", "code_revision"})

    def load(self) -> None:
        """Load the tokenizer and weights at the pinned revision, the model classes at the pinned code revision."""
        from huggingface_hub import snapshot_download  # noqa: PLC0415 - the encoder environment's libraries
        from transformers import PreTrainedTokenizerFast  # noqa: PLC0415
        from transformers.dynamic_module_utils import get_class_from_dynamic_module  # noqa: PLC0415

        options = self.recipe["options"]
        code, code_revision = str(options["code_model_id"]), str(options["code_revision"])
        config_class: Any = get_class_from_dynamic_module(
            "configuration_molformer.MolformerConfig", code, revision=code_revision
        )
        model_class: Any = get_class_from_dynamic_module(
            "modeling_molformer.MolformerModel", code, revision=code_revision
        )
        weights = snapshot_download(str(self.recipe["model_id"]), revision=str(self.recipe["revision"]))
        config = config_class.from_pretrained(weights, deterministic_eval=True)
        self.tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
            tokenizer_file=f"{weights}/tokenizer.json", model_max_length=self.max_tokens, **_SPECIAL_TOKENS
        )
        self.model = model_class.from_pretrained(weights, config=config).to(self.device).eval()
        self._set_depth(int(self.model.config.num_hidden_layers))

    def fingerprint(self) -> Mapping[str, str]:
        """Return the weights' and the code's ids and revisions, the versions, the dtype, the layers and pooling."""
        options = self.recipe["options"]
        return self._fingerprint("float32") | {"code": f"{options['code_model_id']}@{options['code_revision']}"}
