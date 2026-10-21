"""The protein language models' shared procedure: windows, padded-token batches, residue-weighted mean pooling.

ESM-C (``extractors.esmc``) and PGLM (``extractors.pglm``) embed a sequence the paper's way:

- a sequence longer than ``max_residues`` is cut into consecutive windows of at most that many residues;
- windows are sorted by length and batched under a padded-token budget: a batch of ``n`` windows whose longest has
  ``m`` residues costs ``n * (m + specials)`` (the special tokens each window carries);
- a pooled vector is the mean over the residue states of all of a sequence's windows (specials and padding excluded),
  accumulated in float64 from the bf16 or float32 states, so the windows are weighted by their residue counts;
- residue states, when kept, are the windows' residue rows as float32, concatenated in order;
- with ``pooling: cls`` (a model with a ``<cls>`` token: ``WindowedExtractor.poolings``; the special-token
  diagnostic), the pooled vector is instead the ``<cls>`` state (position 0) of the sequence's first window, as float32;
  with ``pooling: eos`` it is the ``<eos>`` state of the sequence's last window (the position after its last residue,
  which both models' tokenizers put there), as float32.

Only the model differs: :class:`WindowedExtractor` holds the rest (the recipe checks, the read layers, the
fingerprint, the pooling and the residue storage), and a model's subclass loads it and runs its forward pass, which
returns, per read layer, ``(n, T, d)`` states and an ``(n, T)`` boolean mask of the residue positions, as tensors on the
model's device. The residue sums are taken there (:func:`reduce_batch`): only ``(n, d)`` sums per layer leave the
device, so an all-layer scan (81 layers of ESM-C 6B) does not move every padded state to the host.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

from latent_interconnect.encoders.base import ExtractedBatch, RawObject
from latent_interconnect.encoders.extractors._layers import check_read, read_layers

__all__ = ["BatchSums", "Forward", "Pooled", "WindowedExtractor", "pool_windows", "reduce_batch", "store_dtype"]

Forward = Callable[[list[str]], tuple[Sequence[Any], Any]]
"""Window sequences → per read layer ``(n, T, d)`` states and the ``(n, T)`` boolean residue mask: torch tensors on
the model's device (typed ``Any``: worker-side modules import torch only inside functions)."""
Pooled = tuple[dict[int, np.ndarray], dict[int, list[np.ndarray]]]
BatchSums = tuple[list[np.ndarray], np.ndarray, list[list[np.ndarray]]]
"""Per read layer ``(n, d)`` float64 residue sums, the ``(n,)`` residue counts, and per read layer the windows'
float32 residue rows (empty unless kept)."""


def reduce_batch(states: Sequence[Any], residue: Any, *, keep_residues: bool) -> BatchSums:
    """Reduce one batch's states over its residue positions, on the states' device (:data:`BatchSums`)."""
    import torch  # noqa: PLC0415 - the encoder environment's torch, imported only in the worker

    keep = residue.unsqueeze(-1)
    zero = torch.zeros((), dtype=torch.float64, device=residue.device)
    # where, not a mask product: a non-finite state at a special or padding position must not reach the sums
    sums = torch.stack([torch.where(keep, s.to(torch.float64), zero).sum(1) for s in states]).cpu().numpy()
    counts = residue.sum(1).cpu().numpy().astype(np.int64)
    rows: list[list[np.ndarray]] = []
    if keep_residues:
        rows = [[s[i][residue[i]].to(torch.float32).cpu().numpy() for i in range(len(residue))] for s in states]
    return list(sums), counts, rows


def pool_windows(
    objects: Sequence[RawObject],
    forward: Forward,
    *,
    layers: Sequence[int],
    width: int,
    max_residues: int,
    specials: int,
    budget: int,
    keep_residues: bool,
    name: str,
    cls: bool = False,
    eos: bool = False,
) -> Pooled:
    """Return layer → ``(B, d)`` float32 pooled vectors, and layer → per-object residue rows (empty unless kept).

    The pooled vector is the residue mean, or with ``cls`` the first window's position-0 state, or with ``eos`` the
    last window's state at the position after its last residue.

    Raises:
        RuntimeError: if the residues the masks count differ from the sequence lengths.
    """
    windows = [
        (i, o.value[a : a + max_residues]) for i, o in enumerate(objects) for a in range(0, len(o.value), max_residues)
    ]
    order = sorted(range(len(windows)), key=lambda k: len(windows[k][1]))
    sums = {L: np.zeros((len(objects), width)) for L in layers}
    heads = {L: np.zeros((len(objects), width), np.float32) for L in layers}
    counts = np.zeros(len(objects), dtype=np.int64)
    pieces: dict[int, dict[int, np.ndarray]] = {L: {} for L in layers}
    batch: list[int] = []
    for k in [*order, None]:
        if k is not None and (not batch or (len(batch) + 1) * (len(windows[k][1]) + specials) <= budget):
            batch.append(k)
            continue
        states, residue = forward([windows[j][1] for j in batch])
        batch_sums, batch_counts, rows = reduce_batch(states, residue, keep_residues=keep_residues)
        for row, j in enumerate(batch):
            owner = windows[j][0]
            counts[owner] += int(batch_counts[row])
            after = int(residue[row].nonzero()[-1]) + 1 if eos else 0  # the <eos> position
            for li, L in enumerate(layers):
                sums[L][owner] += batch_sums[li][row]
                if cls and (j == 0 or windows[j - 1][0] != owner):  # the sequence's first window
                    heads[L][owner] = states[li][row, 0].float().cpu().numpy()
                if eos and (j == len(windows) - 1 or windows[j + 1][0] != owner):  # the sequence's last window
                    heads[L][owner] = states[li][row, after].float().cpu().numpy()
                if keep_residues:
                    pieces[L][j] = rows[li][row]
        batch = [k] if k is not None else []
    lengths = np.array([len(o.value) for o in objects])
    if not np.array_equal(counts, lengths):
        raise RuntimeError(f"{name}: residue counts {counts.tolist()} differ from lengths {lengths.tolist()}")
    pooled = heads if cls or eos else {L: (sums[L] / counts[:, None]).astype(np.float32) for L in layers}
    residues: dict[int, list[np.ndarray]] = {}
    if keep_residues:
        for L in layers:
            residues[L] = [
                np.concatenate([pieces[L][j] for j, w in enumerate(windows) if w[0] == i]) for i in range(len(objects))
            ]
    return pooled, residues


def store_dtype(rows: np.ndarray, dtype: str) -> np.ndarray:
    """Return float32 rows as stored: float32, or bf16 bit patterns in ``uint16`` (round to nearest even)."""
    if dtype == "fp32":
        return rows.astype(np.float32)
    bits = rows.astype(np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    out: np.ndarray = (rounded >> np.uint32(16)).astype(np.uint16)
    return out


_FINGERPRINTS = {
    "mean": "mean over residues, windows of {}, residue-weighted",
    "cls": "<cls> of the first window, windows of {}",
    "eos": "<eos> of the last window, windows of {}",
}
"""The fingerprint's statement of each pooling (the mean's wording kept from before ``cls`` existed)."""


class WindowedExtractor:
    """A protein language model embedded the paper's way; a subclass loads the model and runs its forward pass.

    Class attributes: ``name`` (the registry name), ``specials`` (special tokens per window), ``library`` (the model
    library whose version the fingerprint records), ``context`` (what ``max_residues`` leaves room for) and
    ``poolings`` (``mean``, ``cls`` for a model with a ``<cls>`` token, ``eos`` for one with an ``<eos>``).

    Args:
        recipe: the recipe dict; ``options["max_residues"]`` (int) is required.
        device: the torch device string.

    Raises:
        ValueError: for ``read = "output"`` or ``"pending"``, a pooling not in ``poolings``, or a missing
            ``max_residues``.
    """

    name: ClassVar[str]
    specials: ClassVar[int]
    library: ClassVar[str]
    context: ClassVar[str]
    poolings: ClassVar[tuple[str, ...]] = ("mean",)

    def __init__(self, recipe: Mapping[str, Any], device: str) -> None:
        check_read(recipe["read"], self.name, output=False)
        if recipe["pooling"] not in self.poolings:
            raise ValueError(f"{self.name}: pooling {recipe['pooling']!r}; expected one of {self.poolings}")
        options = recipe.get("options", {})
        if "max_residues" not in options:
            raise ValueError(f"{self.name}: options.max_residues is required (the model context minus {self.context})")
        self.recipe = recipe
        self.device = device
        self.max_residues = int(options["max_residues"])
        self.model: Any = None
        self.tokenizer: Any = None
        self.layers: tuple[int, ...] = ()

    def _set_depth(self, depth: int) -> None:
        """Resolve the recipe's read layers against the loaded model's depth."""
        self.layers, _ = read_layers(self.recipe["read"], depth, self.name, output=False)

    def fingerprint(self) -> Mapping[str, str]:
        """Return the model id and revision, the library versions, the dtype and the pooling procedure."""
        return {
            "extractor": self.name,
            "model_id": str(self.recipe["model_id"]),
            "revision": str(self.recipe["revision"]),
            self.library: importlib.metadata.version(self.library),
            "torch": importlib.metadata.version("torch"),
            "dtype": "bfloat16" if self.device.startswith("cuda") else "float32",
            "pooling": _FINGERPRINTS[str(self.recipe["pooling"])].format(self.max_residues),
        }

    def _forward(self, seqs: list[str]) -> tuple[Sequence[Any], Any]:
        """Return the read layers' ``(n, T, d)`` states and the ``(n, T)`` residue mask (:data:`Forward`)."""
        raise NotImplementedError

    def extract(self, objects: Sequence[RawObject]) -> ExtractedBatch:
        """Return the recipe's layers for a batch of objects (pooled always; residues if the recipe asks)."""
        pooled, residues = pool_windows(
            objects,
            self._forward,
            layers=self.layers,
            width=int(self.model.config.hidden_size),
            max_residues=self.max_residues,
            specials=self.specials,
            budget=int(self.recipe["batch_tokens"]),
            keep_residues=bool(self.recipe.get("residues")),
            name=self.name,
            cls=self.recipe["pooling"] == "cls",
            eos=self.recipe["pooling"] == "eos",
        )
        dtype = str(self.recipe.get("residue_dtype", "bf16"))
        stored = {L: [store_dtype(r, dtype) for r in rows] for L, rows in residues.items()}
        return ExtractedBatch(tuple(o.object_id for o in objects), pooled, stored)
