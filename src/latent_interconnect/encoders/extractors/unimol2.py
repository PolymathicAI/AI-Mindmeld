"""Uni-Mol2 extractor (``unimol2``): unimol_tools' Uni-Mol2 on its own conformers, at every layer (worker side).

The model is the paper's: ``unimol_tools.UniMolRepr``
(``model_name`` and ``model_size`` from ``model_id = "unimolv2/<size>"``, ``remove_hs=False``), float32, in the
version ``options["unimol_tools"]`` names (its conformer and featurizing code: RDKit ETKDG with seed 42 and MMFF with
hydrogens, else 2D coordinates; the model sees heavy atoms only). The weights are
``$UNIMOL_WEIGHT_DIR/modelzoo/<size>/checkpoint.pt``, whose sha256 must be the recipe's ``revision``. unimol_tools
writes ``./logs``, so it runs in a scratch working directory.

Two of unimol_tools' behaviors are recipe options, because the paper's vectors have them and new extractions do not
(new extractions normalize by the unpadded length and never crop at random):

- ``padding``. unimol_tools 0.1.6 computes a molecule differently in a batch than alone (relative L2 up to 0.92 on
  the paper's first molecules), for three reasons, measured one at a time:
  its collate pads the attention mask (``attn_bias``) with zeros, so every atom attends to the padding atoms (0.92 →
  0.023 once masked); every block's outer-product update divides by the batch's padded length
  (``TransformerEncoderWithPairV2.forward``: ``op_mask * op_mask.size(-2) ** -0.5`` and ``op_norm``; → 3.8e-5 once
  each molecule's own length is used); and the triangle multiplication scales its mask by the padded length too
  (``TriangleMultiplication.forward``, before a LayerNorm whose eps keeps the scale; → 1.9e-7, float32 noise).
  These steps were measured once (the 1.1B weights on the paper's first six molecules), not by a test; the slice test
  checks the combined result.
  ``"as_shipped"`` keeps all three (the paper's vectors, which reproduce only in the paper's own batches of 32, in its
  order). Not even the paper's own job reproduces around its 23 ``IndexError`` molecules: the paper's run failed
  their batches and ran those batch-mates alone, whereas here those molecules are skipped before batching, so the
  batch-mates are batched and the batch boundaries after them shift. ``"isolated"`` computes each molecule as if
  alone, with three forward pre-hooks and no change to unimol_tools: the model's hook sets the attention mask to -inf
  at padding atoms (:func:`mask_padding`), and each block's and each triangle multiplication's hook put the molecule's
  own length (its atoms plus the CLS token) in place of the padded one (:func:`own_length_opm`,
  :func:`own_length_trimul`). A molecule alone is unchanged, bit for bit.
- ``crop``. With ``crop: true``, unimol_tools crops a molecule above ``max_atoms`` heavy atoms at random, unseeded
  (91 molecules in the paper's run, 257 to 598 heavy atoms). With ``crop: false`` every molecule is processed whole:
  the model has no positional table, its shortest-path table is clipped at 510 hops (the paper's molecules span at
  most 218), and its atom-type and degree tables do not depend on the atom count, so nothing limits the size but
  memory, which grows with the square of the padded length.

Batching (recorded in the fingerprint): the recipe's ``batch_tokens`` is the batch budget in *pairs*, molecules ×
padded length² (the length counts the CLS token), because the pair representation and its outer-product update
(64² float32 per pair) dominate the memory. With ``"as_shipped"`` the molecules go to the model in the given order in
batches of ``batch_molecules``, as in the paper's run, and a batch over the budget is refused; only the same job
(objects, order, part size, batch size) reproduces its vectors. With ``"isolated"`` the batch no longer matters, so
molecules are sorted by heavy-atom count and packed under both ``batch_molecules`` and the budget. Each batch is
pooled as soon as it ran, so a part holds only its pooled vectors. The objects a crop shortens are named in a warning.

Skips (the rules, :data:`SKIP_RULES`, are in the fingerprint; each skipped object is recorded with its reason in
``ExtractedBatch.skipped``):

- before the model: an empty SMILES; a SMILES RDKit cannot parse; one with no heavy atom (the featurizer raises
  ``IndexError`` on it: 26 of the paper's molecules, such as ``[H+]``); and one with helium among other heavy atoms.
  unimol_tools 0.1.6 drops atomic numbers 1 and 2 as special tokens when it counts a molecule's atoms, then slices
  the first ``n`` atom states, so a helium among other atoms would silently shift them (none in ChEBI-3,
  measured);
- after the model: a molecule left without atom states (every heavy atom helium, as ``[3He+]``);
- a batch that hits RDKit's ``Bad Conformer Id`` (a molecule without a conformer) is retried one molecule at a time,
  named in a warning (as shipped, that changes the retried molecules' padding, as in the paper's run), and the
  molecule that raises it alone is skipped.

A failed 3D embedding is not an error in unimol_tools: it falls back silently to 2D or zero coordinates. Any other
failure is raised, never recorded as a skip. The paper's run skipped 84 of
ChEBI-3's 46,013 molecules; these rules give 83 of them (by RDKit classification). The 84th, CHEBI:5931 (405 heavy
atoms), hit the paper's 900 s conformer timeout; this extractor has no conformer timeout and attempts it.

Conformer retry (opt-in, recipe option ``conformer_retry_timeout_s``; absent, the default, the extractor hands
unimol_tools the SMILES as before). With it, the extractor builds each molecule's conformer itself with
:func:`~latent_interconnect.encoders.extractors._conformer_retry.conformer` (unimol_tools' own ``inner_smi2coords``
first; only a 2D fallback is embedded again, from random starting coordinates, chirality enforced, within the limit)
and hands unimol_tools the molecules (its ``ROMol`` input, which featurizes each with the same ``mol2unimolv2``), so a
molecule that embedded the first time gets the same model input. Each molecule not on the first-try route is named
with its route in ``ExtractedBatch.notes`` (the manifest's ``notes``); a molecule without a note took the first try.

Layers: the states are read with forward hooks on the encoder. Layer 0 is the input of the first block (the
atom features after the encoder's input LayerNorm), layer ``i`` the output of block ``i``, and layer ``D`` (the block
count) is the last block's output, which is what unimol_tools returns (``cls_repr``, ``atomic_reprs``; the encoder
applies no final norm); the hooked layer ``D`` is checked against it. Pooling: ``cls`` is the CLS token (position 0),
``mean`` the mean over the atom states; ``cls+mean`` at ``read = "output"`` is the paper's ``unimol2_1.1b_cat``.
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
import tempfile
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from latent_interconnect.artifacts.store import sha256_file
from latent_interconnect.encoders.base import EXTRACTORS, ExtractedBatch, RawObject
from latent_interconnect.encoders.extractors._conformer_retry import FIRST_3D, conformer
from latent_interconnect.encoders.extractors._layers import read_layers
from latent_interconnect.encoders.extractors._molecules import pool_tokens

__all__ = ["OPM_EPS", "SKIP_RULES", "Unimol2Extractor", "mask_padding", "own_length_opm", "own_length_trimul"]

_POOLINGS = ("mean", "cls", "cls+mean")
_PADDINGS = ("as_shipped", "isolated")
_NO_HEAVY_ATOMS = "no heavy atoms (unimol_tools 0.1.6's featuriser raises IndexError)"
_NO_ATOM_STATES = "no atom states (unimol_tools 0.1.6 drops atoms of atomic number 1 and 2 as special tokens)"
_HELIUM = 2
_NO_SMILES = "no SMILES"
_UNPARSABLE = "RDKit cannot parse the SMILES"
_HELIUM_AMONG_ATOMS = (
    "helium among other heavy atoms (unimol_tools 0.1.6 drops atomic number 2 when counting atoms, shifting the states)"
)
_BAD_CONFORMER = "Bad Conformer Id"
"""The one bad-molecule error unimol_tools 0.1.6 lets through after the RDKit filter (RDKit's ``ValueError`` when a
molecule has no conformer); a failed 3D embedding otherwise falls back silently to 2D or zero coordinates."""


SKIP_RULES = (
    "before the model: an empty SMILES, RDKit cannot parse it, no heavy atom, helium among other heavy atoms; "
    "after: no atom states (every heavy atom helium); alone: RDKit's Bad Conformer Id"
)
"""The skip rules, as the fingerprint records them (the module docstring explains each)."""


def _helium_among_atoms(numbers: Sequence[int]) -> bool:
    heavy = [z for z in numbers if z > 1]
    return _HELIUM in heavy and any(z != _HELIUM for z in heavy)


def _bad_molecule(err: BaseException) -> bool:
    return isinstance(err, ValueError) and _BAD_CONFORMER in str(err)


OPM_EPS = 1e-3
"""The ``eps`` of unimol_tools' ``op_norm`` (``TransformerEncoderWithPairV2.forward``), kept as it is."""


def mask_padding(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Forward pre-hook of the Uni-Mol2 model: no atom attends to a padding atom.

    unimol_tools' collate pads each molecule's zero ``attn_bias`` (``(1 + atoms)²``, the attention mask) with zeros;
    the hook puts -inf in the padding atoms' key columns (the CLS token, column 0, is never padding).
    """
    import torch  # noqa: PLC0415

    if "atom_mask" not in kwargs or "attn_bias" not in kwargs:
        raise RuntimeError(f"unimol2: the model was called without atom_mask / attn_bias ({sorted(kwargs)})")
    atoms = kwargs["atom_mask"]
    keep = torch.cat([torch.ones_like(atoms[:, :1]), atoms], dim=1).bool()
    kwargs["attn_bias"] = kwargs["attn_bias"].masked_fill(~keep[:, None, :], float("-inf"))
    return args, kwargs


def own_length_opm(
    _module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Forward pre-hook of a Uni-Mol2 block: its outer-product normalization from each molecule's own length.

    unimol_tools passes ``op_mask = mask * T ** -0.5`` with ``T`` the padded length; the hook keeps the mask and puts
    each molecule's length ``n`` (its unmasked tokens) in place of ``T``, in the same float64-then-cast arithmetic, so
    an unpadded molecule (``n = T``) gets the same numbers.
    """
    import torch  # noqa: PLC0415

    if "op_mask" not in kwargs:
        raise RuntimeError(f"unimol2: a block was called without op_mask ({sorted(kwargs)})")
    op_mask = kwargs["op_mask"]
    mask = (op_mask > 0).to(op_mask.dtype)
    length = mask.sum(dim=-2, keepdim=True)
    scaled = mask * (length.double() ** -0.5).to(op_mask.dtype)
    kwargs["op_mask"] = scaled
    kwargs["op_norm"] = 1.0 / (OPM_EPS + torch.einsum("...bc,...dc->...bdc", scaled, scaled))
    return args, kwargs


def own_length_trimul(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Forward pre-hook of a Uni-Mol2 triangle multiplication: its mask scaled by the molecule's own length.

    The module scales its pair mask by ``T ** -0.5`` (``T`` the padded length); passing the mask times
    ``sqrt(T / n)`` makes that ``n ** -0.5`` for a molecule of ``n`` tokens, and leaves an unpadded molecule as it is.
    """
    if len(args) != 2:  # noqa: PLR2004 - (pair, mask)
        raise RuntimeError(
            f"unimol2: a triangle multiplication was called with {len(args)} arguments, not (pair, mask)"
        )
    pair, mask = args
    padded = mask.shape[-1]
    length = mask.sum(dim=-1, keepdim=True).amax(dim=-2, keepdim=True)
    return pair, mask * (padded / length.double()).sqrt().to(mask.dtype)


@EXTRACTORS.register("unimol2")
class Unimol2Extractor:
    """Uni-Mol2 through unimol_tools.

    Args:
        recipe: the recipe dict; ``model_id`` is ``unimolv2/<size>``; ``options`` needs ``unimol_tools``,
            ``padding``, ``crop`` (and ``max_atoms`` when cropping) and ``batch_molecules``, and may set
            ``conformer_retry_timeout_s`` (seconds, a positive integer: the conformer retry; absent: off);
            ``batch_tokens`` is the pair budget.
        device: the torch device string (unimol_tools uses ``cuda:0`` for any CUDA device).

    Raises:
        ValueError: for another ``model_id`` form, an unknown pooling or ``padding``, a missing option, or a
            ``conformer_retry_timeout_s`` that is not a positive integer.
    """

    def __init__(self, recipe: Mapping[str, Any], device: str) -> None:
        name, _, size = str(recipe["model_id"]).partition("/")
        if name != "unimolv2" or not size:
            raise ValueError(f"unimol2: model_id {recipe['model_id']!r}; expected 'unimolv2/<size>'")
        if recipe["pooling"] not in _POOLINGS:
            raise ValueError(f"unimol2: pooling {recipe['pooling']!r}; expected one of {_POOLINGS}")
        options = recipe.get("options", {})
        needed = {"unimol_tools", "padding", "crop", "batch_molecules"}
        needed |= {"max_atoms"} if options.get("crop") else set()
        missing = sorted(needed - set(options))
        if missing:
            raise ValueError(f"unimol2: options {missing} are required")
        if options["padding"] not in _PADDINGS:
            raise ValueError(f"unimol2: padding {options['padding']!r}; expected one of {_PADDINGS}")
        retry = options.get("conformer_retry_timeout_s")
        if retry is not None and (isinstance(retry, bool) or not isinstance(retry, int) or retry < 1):
            raise ValueError(f"unimol2: conformer_retry_timeout_s {retry!r}; expected a positive integer (seconds)")
        self.retry_timeout_s: int | None = retry
        self.recipe = recipe
        self.device = device
        self.size = size
        self.isolated = options["padding"] == "isolated"
        self.max_atoms = int(options["max_atoms"]) if options["crop"] else sys.maxsize
        self.repr: Any = None
        self.layers: tuple[int, ...] = ()
        self.keys: dict[int, int] = {}
        self.depth = 0
        self.budget = int(recipe["batch_tokens"])
        self.workdir: tempfile.TemporaryDirectory[str] | None = None
        self._hooked: dict[int, list[Any]] = {}

    @contextmanager
    def _in_workdir(self) -> Iterator[None]:
        if self.workdir is None:
            raise RuntimeError("unimol2: load() first")
        cwd = Path.cwd()
        os.chdir(self.workdir.name)
        try:
            yield
        finally:
            os.chdir(cwd)

    def load(self) -> None:
        """Check the library version and the weights' sha256, load the model, hook its encoder blocks."""
        options = self.recipe["options"]
        version = importlib.metadata.version("unimol_tools")
        if version != str(options["unimol_tools"]):
            raise ValueError(f"unimol2: unimol_tools {version}, the recipe pins {options['unimol_tools']}")
        weights = Path(os.environ.get("UNIMOL_WEIGHT_DIR", "")) / "modelzoo" / self.size / "checkpoint.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"unimol2: no weights at {weights} (set UNIMOL_WEIGHT_DIR)")
        digest = sha256_file(weights)
        if digest != str(self.recipe["revision"]):
            raise ValueError(f"unimol2: {weights} has sha256 {digest}, the recipe pins {self.recipe['revision']}")
        self.workdir = tempfile.TemporaryDirectory(prefix="unimol2-")  # its ./logs; removed with the extractor
        with self._in_workdir():
            from unimol_tools import UniMolRepr  # noqa: PLC0415 - writes ./logs at import

            self.repr = UniMolRepr(
                data_type="molecule",
                remove_hs=False,
                model_name="unimolv2",
                model_size=self.size,
                use_cuda=self.device.startswith("cuda"),
                max_atoms=self.max_atoms,
                batch_size=int(options["batch_molecules"]),
                pretrained_model_path=str(weights),
            )
        encoder = self.repr.model.encoder
        self.depth = len(encoder.layers)
        self.layers, self.keys = read_layers(self.recipe["read"], self.depth, "unimol2", output=True)
        if self.isolated:
            self.repr.model.register_forward_pre_hook(mask_padding, with_kwargs=True)
            for block in encoder.layers:
                block.register_forward_pre_hook(own_length_opm, with_kwargs=True)
                block.pair_tri_mul.register_forward_pre_hook(own_length_trimul)
        modules = {0: encoder.layer_norm, **{i + 1: block for i, block in enumerate(encoder.layers)}}
        for L in set(self.layers) | {self.depth}:
            self._hooked[L] = []
            modules[L].register_forward_hook(self._hook(L))

    def _hook(self, layer: int) -> Any:
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            x = output[0] if isinstance(output, tuple) else output
            self._hooked[layer].append(x.detach().float().cpu().numpy())

        return record

    def fingerprint(self) -> Mapping[str, str]:
        """Return the model, weights' sha256, library versions, dtype, overrides, pooling and skip rules."""
        options = self.recipe["options"]
        crop = f"random crop above {options['max_atoms']} heavy atoms (unseeded)" if options["crop"] else "none"
        return {
            "extractor": "unimol2",
            "model_id": str(self.recipe["model_id"]),
            "revision": str(self.recipe["revision"]),
            "unimol_tools": importlib.metadata.version("unimol_tools"),
            "torch": importlib.metadata.version("torch"),
            "dtype": "float32",
            "padding": (
                "isolated: padding masked in attention, own-length outer product and triangle multiplication (hooks)"
                if self.isolated
                else "as shipped: the batch's padding enters every molecule"
            ),
            "crop": crop,
            "batching": (
                f"length-sorted, <= {options['batch_molecules']} molecules and <= {self.budget} pairs (batch-free)"
                if self.isolated
                else f"{options['batch_molecules']} per batch in object order within parts (vectors depend on it)"
            ),
            "output_layer": str(self.depth),
            "pooling": f"{self.recipe['pooling']} over heavy atoms",
            "skips": SKIP_RULES,
        } | (
            {}
            if self.retry_timeout_s is None
            else {
                "conformer_retry": f"a 2D fallback is embedded again from random starting coordinates, chirality "
                f"enforced, within {self.retry_timeout_s} s, and kept if the stereo is unchanged (_conformer_retry)"
            }
        )

    def _batches(self, objects: Sequence[RawObject]) -> tuple[list[list[int]], dict[str, str]]:
        """Return the model batches (indices of ``objects``) and the objects skipped before the model (id → reason).

        Batches keep the given order as shipped, or are length-sorted under the budgets; the objects skipped before
        the model (an empty or unparsable SMILES, no heavy atom, helium among other heavy atoms) never reach it.

        Raises:
            ValueError: as shipped, for a batch over the pair budget.
        """
        from rdkit import Chem  # noqa: PLC0415 - the encoder environment's RDKit (unimol_tools' own)

        mols: list[Any] = [Chem.MolFromSmiles(o.value) if o.value else None for o in objects]
        pre_skipped = {}
        for o, m in zip(objects, mols, strict=True):
            if not o.value:
                pre_skipped[o.object_id] = _NO_SMILES
            elif m is None:
                pre_skipped[o.object_id] = _UNPARSABLE
            elif m.GetNumHeavyAtoms() == 0:
                pre_skipped[o.object_id] = _NO_HEAVY_ATOMS
            elif _helium_among_atoms([a.GetAtomicNum() for a in m.GetAtoms()]):
                pre_skipped[o.object_id] = _HELIUM_AMONG_ATOMS
        index = [k for k, o in enumerate(objects) if o.object_id not in pre_skipped]  # back to the caller's positions
        atoms = [int(mols[k].GetNumHeavyAtoms()) for k in index]
        objects = [objects[k] for k in index]
        cropped = [o.object_id for o, n in zip(objects, atoms, strict=True) if n > self.max_atoms]
        if cropped:
            warnings.warn(f"unimol2: cropped at random to {self.max_atoms} atoms: {cropped}", stacklevel=2)
        length = [min(n, self.max_atoms) + 1 for n in atoms]  # + the CLS token
        size = int(self.recipe["options"]["batch_molecules"])
        if not self.isolated:
            batches = [list(range(a, min(a + size, len(objects)))) for a in range(0, len(objects), size)]
            over = [b for b in batches if len(b) * max(length[k] for k in b) ** 2 > self.budget]
            if over:
                ids = [objects[k].object_id for k in over[0]]
                raise ValueError(f"unimol2: a batch of {ids} is over the {self.budget}-pair budget (batch_tokens)")
            return [[index[k] for k in b] for b in batches], pre_skipped
        batches = []
        for k in sorted(range(len(objects)), key=lambda j: length[j]):
            last = batches[-1] if batches else []
            if last and len(last) < size and (len(last) + 1) * length[k] ** 2 <= self.budget:
                last.append(k)
            else:
                batches.append([k])
        return [[index[k] for k in b] for b in batches], pre_skipped

    def _molecules(self, smiles: list[str]) -> tuple[Any, list[str]]:
        """The retry's model input: unimol_tools' ``ROMol`` frame of the molecules' conformers, and their routes.

        The conformers are built in a pool of unimol_tools' own size (``UniMolV2Feature.transform``: at most 8
        processes), with unimol_tools' own seed, mode and ``remove_hs``.
        """
        from multiprocessing import Pool  # noqa: PLC0415

        import pandas as pd  # noqa: PLC0415 - unimol_tools' own dependency
        from rdkit import Chem  # noqa: PLC0415
        from unimol_tools.data.conformer import UniMolV2Feature  # noqa: PLC0415

        if self.retry_timeout_s is None:
            raise RuntimeError("unimol2: the conformer retry is off")
        feature = UniMolV2Feature(**self.repr.params)
        build = partial(
            conformer, seed=feature.seed, mode=feature.mode, remove_hs=feature.remove_hs, timeout_s=self.retry_timeout_s
        )
        with Pool(processes=min(8, os.cpu_count() or 1)) as pool:
            built = pool.map(build, smiles)
        return pd.DataFrame({"ROMol": [Chem.Mol(b) for b, _ in built]}), [route for _, route in built]

    def _run(self, objects: list[RawObject]) -> tuple[list[dict[int, np.ndarray] | str], dict[str, str]]:
        """One model call: per molecule, layer → its pooled vector, or a skip reason (no atom states); and the notes.

        The notes name, with the conformer retry on, each molecule not on the first-try route and its route.

        Raises:
            ValueError: RDKit's ``Bad Conformer Id`` for a molecule without a conformer (the bad-input error).
            RuntimeError: if the hooked final layer is not what unimol_tools returned (a broken hook: never a skip).
        """
        smiles = [o.value for o in objects]
        for calls in self._hooked.values():
            calls.clear()
        notes: dict[str, str] = {}
        with self._in_workdir():
            self.repr.params["batch_size"] = len(smiles)  # one unimol_tools batch per call
            if self.retry_timeout_s is None:
                out = self.repr.get_repr(smiles, return_atomic_reprs=True)
            else:
                frame, routes = self._molecules(smiles)
                notes = {o.object_id: f"conformer: {r}" for o, r in zip(objects, routes, strict=True) if r != FIRST_3D}
                out = self.repr.get_repr(frame, return_atomic_reprs=True)
        atoms = [np.asarray(a, dtype=np.float32) for a in out["atomic_reprs"]]
        if len(atoms) != len(smiles):
            raise RuntimeError(f"unimol2: {len(atoms)} representations for {len(smiles)} molecules")
        rows = {L: [m for batch in calls for m in batch] for L, calls in self._hooked.items()}
        pooling = str(self.recipe["pooling"])
        pooled: list[dict[int, np.ndarray] | str] = []
        for i, a in enumerate(atoms):  # the hooked last block is what unimol_tools returned
            final = rows[self.depth][i]
            if not (
                np.array_equal(final[1 : len(a) + 1], a)
                and np.array_equal(final[0], np.asarray(out["cls_repr"][i], dtype=np.float32))
            ):
                raise RuntimeError(
                    f"unimol2: the hooked final layer differs from unimol_tools' output ({objects[i].object_id})"
                )
            if len(a) == 0:  # e.g. [3He+]: helium's atomic number is one of the featurizer's special tokens
                pooled.append(_NO_ATOM_STATES)
                continue
            keep = np.arange(len(a) + 1) > 0  # the atoms, not the CLS token
            pooled.append({L: pool_tokens(rows[L][i][: len(a) + 1], keep, pooling) for L in self.layers})
        for calls in self._hooked.values():
            calls.clear()
        return pooled, notes

    def extract(self, objects: Sequence[RawObject]) -> ExtractedBatch:
        """Return the recipe's layers, pooled; molecules that are bad inputs are skipped with their reason.

        A bad input is one of :data:`SKIP_RULES`: skipped before the model, left without atom states, or raising
        RDKit's ``Bad Conformer Id`` when run alone. Every other failure (a broken hook, the
        final-layer check, running out of memory) is raised: it is a failed run, not a bad molecule. With the conformer
        retry on, the batch's ``notes`` name each embedded molecule not on the first-try route, with its route.
        """
        results: dict[int, dict[int, np.ndarray] | str] = {}
        notes: dict[str, str] = {}
        batches, skipped = self._batches(objects)
        retried: list[str] = []
        for batch in batches:
            try:
                rows, batch_notes = self._run([objects[k] for k in batch])
                results.update(zip(batch, rows, strict=True))
                notes |= batch_notes
                continue
            except ValueError as err:
                if not _bad_molecule(err):
                    raise
                retried += [objects[k].object_id for k in batch]
            for k in batch:  # a bad molecule in the batch: find it by running each alone
                try:
                    alone, alone_notes = self._run([objects[k]])
                    results[k] = alone[0]
                    notes |= alone_notes
                except ValueError as err:
                    if not _bad_molecule(err):
                        raise
                    results[k] = f"{type(err).__name__}: {err}"[:300]
        if retried:
            warnings.warn(f"unimol2: retried one at a time after a bad molecule: {retried}", stacklevel=2)
        vectors = {k: r for k, r in results.items() if not isinstance(r, str)}
        skipped |= {objects[k].object_id: r for k, r in results.items() if isinstance(r, str)}
        kept = [k for k in range(len(objects)) if k in vectors]
        pooled = {
            self.keys[L]: np.stack([vectors[k][L] for k in kept]) if kept else np.zeros((0, 0), np.float32)
            for L in self.layers
        }
        kept_ids = tuple(objects[k].object_id for k in kept)
        notes = {o: notes[o] for o in kept_ids if o in notes}  # a skipped molecule has its reason instead
        return ExtractedBatch(kept_ids, pooled, {}, skipped=skipped, notes=notes)
