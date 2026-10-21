"""MIMIC extractor (``mimic``): the EVA encoder's register tokens or residue mean (worker side; ``eva`` in load).

Runs in central-dogma's environment (``cd-xl-central``). The procedure is the paper's:
sequences go to ``EVAInterface.input`` as ``{"aa_seq": sequence}`` and
``EVAInterface.encode(sep_encodings=False)`` gives the encoder output ``full`` (registers first, then the sequence's
tokens, then padding), its per-position modality ids ``mask`` (-1 at padding) and ``register``, the first
``num_register_tokens`` rows (5 × 1536 for MIMIC), in the model's own precision; vectors are stored in float32.

Poolings (the representation diagnostics, superseding an earlier register scan):

- ``registers``: the register rows flattened (5 × 1536 = 7,680), at the output layer only (the paper's input);
- ``registers_mean``: the mean of the register rows (1,536), at the output layer only;
- ``mean``: the mean over the residue positions (after the registers, before the padding; one token per residue,
  checked against each sequence's length), at any layer: the layer scan.

Layers: the EVA encoder of MIMIC is an ``x_transformers`` ``AttentionLayers`` stack (``encoder.torso``,
20 blocks) ending in a final norm. Layer 0 is the stack's input, layer ``i`` the residual stream after block ``i``, and
layer ``D`` (20) the normed output, which ``encode`` returns. Layers below ``D`` (``mean`` only) are the stack's
``return_hiddens`` states (``hiddens[i]``, the input of block ``i + 1``), taken by two hooks on the torso registered at
load when the recipe reads one; they share the output's positions.

Recipe: ``model_id`` is ``"<name>/<version>"`` (``EVAInterface.load_model``); ``revision`` is the sha256 of the
checkpoint file the interface loads (checked before loading); ``options["chkpt_number"]`` selects a numbered
checkpoint instead of ``checkpoint-final.pth``; ``read`` is ``"output"`` (layer ``D`` stored as ``OUTPUT_LAYER``), a
tuple of layers or ``"all"`` (``0..D``). Batches hold at most ``batch_tokens`` residues counted as ``n * longest``, in
the given order. A sequence that is empty or longer than the model's ``tok_aa_seq`` limit (10,000 residues for MIMIC
5.7) is skipped with the old script's reason (``length>10000``, ``empty``): the paper's extraction flagged such
sequences and never cut them.

Public model (``options["source"] = "hub"``, recipe ``mimic-1.0``): ``model_id`` is the Hugging Face repository
(``polymathic-ai/MIMIC``) and ``revision`` its commit; the public ``mimic`` package
(https://github.com/PolymathicAI/MIMIC) loads it with ``mimic.load_pretrained`` instead of ``EVAInterface``, and
:class:`_HubModel` gives it EVAInterface's names (``model.encoder``, ``encode(sep_encodings=False)``, ``mask``), so the
read, the poolings and the skips are the ones above. MIMIC 1.0 is the bf16 rounding of the 5.7 checkpoint
(``examples/README.md``); the package holds the weights in float32 and autocasts to bf16 on CUDA only.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from latent_interconnect.artifacts.store import sha256_file
from latent_interconnect.encoders.base import EXTRACTORS, ExtractedBatch, RawObject
from latent_interconnect.encoders.extractors._layers import check_read, read_layers

__all__ = ["MimicExtractor"]

_FINAL = "checkpoint-final.pth"
_POOLINGS = ("registers", "registers_mean", "mean")
_OUTPUT_ONLY = ("registers", "registers_mean")


@EXTRACTORS.register("mimic")
class MimicExtractor:
    """MIMIC (EVA) register tokens or residue mean.

    Args:
        recipe: the recipe dict.
        device: ``cuda`` or ``cpu`` (``EVAInterface.load_model``; select a GPU with ``CUDA_VISIBLE_DEVICES``).

    Raises:
        ValueError: for ``read = "pending"``, a pooling not in :data:`_POOLINGS`, a ``model_id`` not of the form
            ``name/version``, or a device other than ``cuda`` / ``cpu``.
    """

    def __init__(self, recipe: Mapping[str, Any], device: str) -> None:
        check_read(recipe["read"], "mimic", output=True)
        if recipe["pooling"] not in _POOLINGS:
            raise ValueError(f"mimic: pooling {recipe['pooling']!r}; expected one of {_POOLINGS}")
        if device not in ("cuda", "cpu"):
            raise ValueError(
                f"mimic: device {device!r}; EVA takes 'cuda' or 'cpu' (pick the GPU by CUDA_VISIBLE_DEVICES)"
            )
        name, sep, version = str(recipe["model_id"]).partition("/")
        if not sep:
            raise ValueError(f"mimic: model_id {recipe['model_id']!r} is not '<name>/<version>'")
        self.recipe, self.device = recipe, device
        self.name, self.version = name, version
        number = recipe.get("options", {}).get("chkpt_number")
        self.chkpt_number = int(number) if number is not None else None
        self.eva: Any = None
        self.checkpoint: Path | None = None
        self.keys: dict[int, int] = {}
        self.hiddens: list[Any] = []

    def load(self) -> None:
        """Check the checkpoint file's sha256 against the recipe's revision, then load the model.

        With ``options["source"] = "hub"``, load the public model at the pinned Hugging Face commit instead (module
        docstring).

        Raises:
            ValueError: if the checkpoint's sha256 differs from ``revision``, or a register pooling reads a layer
                below the output.
        """
        if self.recipe.get("options", {}).get("source") == "hub":
            from mimic import load_pretrained  # noqa: PLC0415 - the mimic environment only

            model_id, revision = str(self.recipe["model_id"]), str(self.recipe["revision"])
            self.eva = _HubModel(load_pretrained(hf_repo=model_id, revision=revision, device=self.device))
            self.checkpoint = Path(f"hf:{model_id}@{revision}")
            self._read_and_hook()
            return
        from central_dogma.config import paths  # noqa: PLC0415 - central-dogma's environment only
        from eva.utils.model_interface import EVAInterface  # noqa: PLC0415

        folder = Path(paths.get_path(data_type="model", name=self.name, version=self.version, stage=None))
        self.checkpoint = folder / (f"checkpoint-{self.chkpt_number}.pth" if self.chkpt_number else _FINAL)
        found = sha256_file(self.checkpoint)
        if found != self.recipe["revision"]:
            raise ValueError(f"mimic: {self.checkpoint} has sha256 {found}, the recipe pins {self.recipe['revision']}")
        self.eva = EVAInterface()
        self.eva.load_model(
            name=self.name, version=self.version, chkpt_number=self.chkpt_number, run_scripts=False, device=self.device
        )
        self._read_and_hook()

    def _read_and_hook(self) -> None:
        """Resolve the recipe's layers against the loaded encoder and hook its torso for layers below the output."""
        depth = int(self.eva.model.encoder.get_num_layers())
        layers, self.keys = read_layers(self.recipe["read"], depth, "mimic", output=True)
        if self.recipe["pooling"] in _OUTPUT_ONLY and min(layers) < depth:
            raise ValueError(f"mimic: pooling {self.recipe['pooling']!r} reads the output layer {depth} only")
        if min(layers) < depth:
            torso = self.eva.model.encoder.torso
            torso.register_forward_pre_hook(
                lambda _m, args, kw: (args, {**kw, "return_hiddens": True}), with_kwargs=True
            )
            torso.register_forward_hook(self._keep_hiddens)

    def _keep_hiddens(self, _module: Any, _args: Any, output: tuple[Any, Any]) -> Any:
        """Keep the torso's per-block inputs and hand EVA the output alone (forward hook)."""
        x, intermediates = output
        self.hiddens = list(intermediates.hiddens)
        return x

    def fingerprint(self) -> Mapping[str, str]:
        """Return the checkpoint (path and sha256), the EVA dtype, the library versions and ``output_layer``.

        ``output_layer`` is the encoder's layer count, the ``hidden_states`` index that ``OUTPUT_LAYER`` stands for.
        """
        return {
            "extractor": "mimic",
            "model_id": str(self.recipe["model_id"]),
            "checkpoint": str(self.checkpoint),
            "revision": str(self.recipe["revision"]),
            "dtype": str(self.eva.dtype),
            "torch": importlib.metadata.version("torch"),
            "registers": str(self.eva.model.num_register_tokens),
            "output_layer": str(self.eva.model.encoder.get_num_layers()),
        }

    def extract(self, objects: Sequence[RawObject]) -> ExtractedBatch:
        """Return each object's pooled vector at the recipe's layers (poolings and skips: the module docstring).

        Raises:
            RuntimeError: if the torso returned another number of block inputs than the encoder has blocks, or a
                sequence's residue positions are not its length.
        """
        import torch  # noqa: PLC0415

        limit = int(self.eva.modality_info["tok_aa_seq"]["max_tokens"])
        skipped = {
            o.object_id: "empty" if not o.value else f"length>{limit}" for o in objects if not 0 < len(o.value) <= limit
        }
        kept = [o for o in objects if o.object_id not in skipped]
        depth = int(self.eva.model.encoder.get_num_layers())
        budget = int(self.recipe["batch_tokens"])
        rows: dict[int, list[np.ndarray]] = {L: [] for L in self.keys}
        start = 0
        while start < len(kept):
            end = start + 1
            while end < len(kept) and (end - start + 1) * max(len(o.value) for o in kept[start : end + 1]) <= budget:
                end += 1  # grow the batch while n * longest stays within the budget
            self.eva.input([{"aa_seq": o.value} for o in kept[start:end]], bypass_input_length_check=False)
            self.hiddens = []
            with torch.inference_mode():
                enc = self.eva.encode(sep_encodings=False)
            for L, vectors in self._pool(enc, kept[start:end], depth).items():
                rows[L].append(vectors.float().cpu().numpy().reshape(end - start, -1))
            start = end
        pooled = {self.keys[L]: np.concatenate(r) if r else np.zeros((0, 0), np.float32) for L, r in rows.items()}
        return ExtractedBatch(tuple(o.object_id for o in kept), pooled, {}, skipped=skipped)

    def _pool(self, enc: Mapping[str, Any], batch: Sequence[RawObject], depth: int) -> dict[int, Any]:
        """Return layer → one batch's ``(n, ...)`` pooled tensors (the module docstring's poolings)."""
        import torch  # noqa: PLC0415

        pooling = self.recipe["pooling"]
        if pooling == "registers":
            return {depth: enc["register"]}
        if pooling == "registers_mean":
            return {depth: enc["register"].double().mean(1)}
        residue = enc["mask"] != -1
        residue[:, : int(self.eva.model.num_register_tokens)] = False
        counts = residue.sum(1)
        if counts.tolist() != [len(o.value) for o in batch]:
            raise RuntimeError(f"mimic: residue positions {counts.tolist()} are not the sequence lengths")
        states = {depth: enc["full"]}
        if min(self.keys) < depth:
            if len(self.hiddens) != depth:
                raise RuntimeError(f"mimic: the torso returned {len(self.hiddens)} block inputs, not {depth}")
            states.update(enumerate(self.hiddens))
        keep = residue.unsqueeze(-1)
        # where, not a mask product: a non-finite state at a register or padding position must not reach the mean
        return {L: torch.where(keep, states[L].double(), 0).sum(1) / counts.unsqueeze(-1) for L in self.keys}


class _HubModel:
    """The public ``mimic.MIMIC`` model under the EVAInterface names the extractor uses (module docstring)."""

    def __init__(self, model: Any) -> None:
        import torch  # noqa: PLC0415

        self.model = model
        self.modality_info = model.modality_info
        self.dtype = model.autocast_dtype if model.device.type == "cuda" else torch.float32

    def input(self, batch: Sequence[Mapping[str, str]], bypass_input_length_check: bool) -> None:
        """Tokenize and stage a batch (``mimic.MIMIC.input``)."""
        self.model.input(list(batch), bypass_input_length_check=bypass_input_length_check)

    def encode(self, sep_encodings: bool) -> dict[str, Any]:
        """Return ``full``, ``mask`` (``embed``'s ``mod_ids``) and ``register``, as ``EVAInterface.encode``."""
        if sep_encodings:
            raise ValueError("mimic: the hub model encodes with sep_encodings=False only")
        enc = self.model.embed(return_register=True)
        return {"full": enc["full"], "mask": enc["mod_ids"], "register": enc["register"]}
