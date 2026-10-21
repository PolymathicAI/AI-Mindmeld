"""The molecule extractors' shared rules: token pooling (all three), and the SMILES models' common extractor.

Pooling, used by MoLFormer, ChemFM and Uni-Mol2 at every read layer: ``mean`` over the molecule's own tokens
(a SMILES model's added special tokens, or Uni-Mol2's CLS, excluded), ``cls`` the first position's state, ``last`` the
last position's, and ``cls+mean`` / ``last+mean`` the summary state concatenated with the mean, summary first; the mean
is summed in float64 from float32 states. What the first and last positions are is the model's: MoLFormer's ``<bos>``
and ``<eos>``, ChemFM's first and last SMILES tokens (its tokenizer adds no special token), Uni-Mol2's CLS token.
Two poolings reproduce a paper's own representation, SMILES models only: ``mean_all`` the mean over every
token, the tokenizer's added specials included (MoLFormer's ``pooler_output``, ``masked_avg_pool1d``), and ``eos`` the
state of an ``<eos>`` (the tokenizer's ``eos_token_id``) appended after the SMILES (ChemFM's pretraining input
``{SMILES}<eos>``); its SMILES are truncated to leave room for it.

:class:`SmilesExtractor` holds what MoLFormer and ChemFM share, as the paper's run did them:
one molecule at a time (``options["batch_molecules"]`` must be 1, so no padding
ever enters), truncation at ``options["max_tokens"]`` tokens, the read layers (``extractors._layers``; layer ``D`` is
the model's ``last_hidden_state``, after its final norm), and one rule for which tokens are the molecule's: those the
tokenizer did not add (``special_tokens_mask``), so an ``<unk>`` (a SMILES character outside the vocabulary; 330 of
the paper's 45,970 molecules for MoLFormer, none for ChemFM) counts as the position it is. A molecule with no token
of its own is skipped with its reason (``ExtractedBatch.skipped``).

Per-token rows (``residues: true``; the residue map kinds' input, the molecule tr arm): one row per token of the
pooling's mean, in order, each row the pooled vector's parts with the mean replaced by that token's state, so
``last+mean`` gives ``[h_last | h_t]`` and ``mean`` gives ``h_t`` (:func:`token_rows`). The rows run over exactly the
tokens the mean runs over (the molecule's own tokens; every token for ``mean_all``; no padding, one molecule at a
time), so their mean is the pooled vector: the pooled vector's ridge and standardization hold for them unchanged.
They are stored as the protein residues are (``residue_dtype``, bf16 by default: ``_windows.store_dtype``); the pooled
vectors are the same with or without them. Poolings with no mean (``cls``, ``last``, ``eos``) have no such rows.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

from latent_interconnect.encoders.base import ExtractedBatch, RawObject
from latent_interconnect.encoders.extractors._layers import check_read, read_layers
from latent_interconnect.encoders.extractors._windows import store_dtype

__all__ = ["POOLINGS", "TOKEN_ROW_POOLINGS", "SmilesExtractor", "pool_tokens", "token_rows"]

POOLINGS = ("mean", "cls", "last", "cls+mean", "last+mean", "mean_all", "eos")
"""The token poolings a molecule model supports."""

TOKEN_ROW_POOLINGS = ("mean", "cls+mean", "last+mean", "mean_all")
"""The poolings with per-token rows (:func:`token_rows`): those with a mean part."""


def pool_tokens(states: np.ndarray, keep: np.ndarray, pooling: str) -> np.ndarray:
    """Pool one molecule's ``(T, d)`` float32 states; ``keep`` marks the tokens the mean runs over.

    Raises:
        ValueError: for an unknown pooling, or no token to pool.
    """
    if pooling not in POOLINGS:
        raise ValueError(f"pooling {pooling!r}: expected one of {POOLINGS}")
    if not keep.any():
        raise ValueError("no token of the molecule to pool")
    if pooling == "mean_all":
        keep = np.ones(len(states), bool)
    mean = states[keep].astype(np.float64).mean(0).astype(np.float32)
    summary = {"cls": states[0], "last": states[-1], "eos": states[-1]}
    parts = [summary.get(p, mean) for p in pooling.split("+")]
    return np.concatenate(parts).astype(np.float32)


def token_rows(states: np.ndarray, keep: np.ndarray, pooling: str) -> np.ndarray:
    """Return one molecule's ``(n, width)`` float32 per-token rows, whose mean is :func:`pool_tokens`'s vector.

    One row per token the mean runs over (``keep``; every token for ``mean_all``), in order: the pooling's parts with
    the mean replaced by the token's state, the summary state repeated (``last+mean``: ``[h_last | h_t]``).

    Raises:
        ValueError: for a pooling without a mean part, or no token to pool.
    """
    if pooling not in TOKEN_ROW_POOLINGS:
        raise ValueError(f"pooling {pooling!r} has no per-token rows; expected one of {TOKEN_ROW_POOLINGS}")
    if not keep.any():
        raise ValueError("no token of the molecule to pool")
    if pooling == "mean_all":
        keep = np.ones(len(states), bool)
    own = states[keep]
    summary = {"cls": states[0], "last": states[-1]}
    parts = [np.broadcast_to(summary[p], own.shape) if p in summary else own for p in pooling.split("+")]
    return np.concatenate(parts, axis=1).astype(np.float32)


_OWN = "over its own tokens (added specials out)"
_TOKENS = {"mean_all": "over every token (added specials in)", "eos": "at an appended <eos>"}
"""How the fingerprint states the tokens a pooling reads (the earlier wording kept for the earlier poolings)."""


class SmilesExtractor:
    """A SMILES language model run one molecule at a time; a subclass loads the model (setting ``depth``).

    Class attributes: ``name`` (the registry name) and ``required`` (the options it needs besides ``max_tokens``).

    Args:
        recipe: the recipe dict; ``options["max_tokens"]`` is required.
        device: the torch device string.

    Raises:
        ValueError: for an unknown pooling, ``read = "pending"``, a missing option, ``batch_molecules`` other
            than 1, or ``residues`` with a pooling that has no per-token rows.
    """

    name: ClassVar[str]
    required: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, recipe: Mapping[str, Any], device: str) -> None:
        check_read(recipe["read"], self.name, output=True)
        if recipe["pooling"] not in POOLINGS:
            raise ValueError(f"{self.name}: pooling {recipe['pooling']!r}; expected one of {POOLINGS}")
        options = recipe.get("options", {})
        missing = sorted(({"max_tokens"} | self.required) - set(options))
        if missing:
            raise ValueError(f"{self.name}: options {missing} are required")
        if int(options.get("batch_molecules", 1)) != 1:
            raise ValueError(f"{self.name}: batch_molecules must be 1 (the paper embeds one molecule at a time)")
        self.residues = bool(recipe.get("residues"))
        if self.residues and recipe["pooling"] not in TOKEN_ROW_POOLINGS:
            raise ValueError(f"{self.name}: residues need a pooling with a mean part, one of {TOKEN_ROW_POOLINGS}")
        self.recipe = recipe
        self.device = device
        self.max_tokens = int(options["max_tokens"])
        self.model: Any = None
        self.tokenizer: Any = None
        self.layers: tuple[int, ...] = ()
        self.keys: dict[int, int] = {}
        self.depth = 0

    def _set_depth(self, depth: int) -> None:
        self.depth = depth
        self.layers, self.keys = read_layers(self.recipe["read"], depth, self.name, output=True)

    def _fingerprint(self, dtype: str) -> dict[str, str]:
        pooling = str(self.recipe["pooling"])
        rows = {"residues": f"per token: {pooling} with the token's state for the mean, over the mean's tokens"}
        return {
            "extractor": self.name,
            "model_id": str(self.recipe["model_id"]),
            "revision": str(self.recipe["revision"]),
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
            "dtype": dtype,
            "output_layer": str(self.depth),
            "batching": "one molecule at a time (no padding)",
            "pooling": f"{self.recipe['pooling']} {_TOKENS.get(pooling, _OWN)}, <= {self.max_tokens}",
        } | (rows if self.residues else {})

    def _states(self, smiles: str) -> tuple[list[np.ndarray], np.ndarray]:
        """One molecule's states per read layer (``(T, d)`` float32) and its own-token mask."""
        import torch  # noqa: PLC0415

        suffix = [int(self.tokenizer.eos_token_id)] if self.recipe["pooling"] == "eos" else []
        enc = self.tokenizer(
            smiles,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens - len(suffix),
            return_special_tokens_mask=True,
        )
        if suffix and bool(enc["special_tokens_mask"].any()):
            raise ValueError(f"{self.name}: pooling 'eos' appends <eos> itself; this tokenizer adds special tokens")
        ids = torch.cat([enc["input_ids"][0], torch.tensor(suffix, dtype=torch.long)])
        own = torch.cat([enc["special_tokens_mask"][0] == 0, torch.zeros(len(suffix))])
        with torch.no_grad():
            out = self.model(
                input_ids=ids[None].to(self.device),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=self.device),
                output_hidden_states=True,
            )
        if len(out.hidden_states) != self.depth + 1:
            raise RuntimeError(f"{self.name}: {len(out.hidden_states)} hidden states, expected {self.depth + 1}")
        every = (*out.hidden_states[: self.depth], out.last_hidden_state)
        return [every[L][0].float().cpu().numpy() for L in self.layers], own.bool().numpy()

    def extract(self, objects: Sequence[RawObject]) -> ExtractedBatch:
        """Return the recipe's layers, pooled (and per token with ``residues``), one molecule at a time.

        A molecule with no token is skipped.
        """
        pooling, dtype = str(self.recipe["pooling"]), str(self.recipe.get("residue_dtype", "bf16"))
        rows: dict[int, list[np.ndarray]] = {L: [] for L in self.layers}
        per_token: dict[int, list[np.ndarray]] = {L: [] for L in self.layers}
        kept: list[str] = []
        skipped: dict[str, str] = {}
        for obj in objects:
            states, keep = self._states(obj.value)
            if not keep.any():
                skipped[obj.object_id] = f"no token of the molecule in {obj.value!r}"[:300]
                continue
            kept.append(obj.object_id)
            for L, h in zip(self.layers, states, strict=True):
                rows[L].append(pool_tokens(h, keep, pooling))
                if self.residues:
                    per_token[L].append(store_dtype(token_rows(h, keep, pooling), dtype))
        pooled = {self.keys[L]: np.stack(v) if v else np.zeros((0, 0), np.float32) for L, v in rows.items()}
        residues = {self.keys[L]: v for L, v in per_token.items()} if self.residues and kept else {}
        return ExtractedBatch(tuple(kept), pooled, residues, skipped=skipped)
