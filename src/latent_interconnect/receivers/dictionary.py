"""A frozen sparse dictionary and the one state edit a backend may apply right after a write (opt-in).

A **dictionary** is a published sparse autoencoder of the receiver's residual stream (Gemma Scope 2's JumpReLU
dictionaries): :class:`DictionarySpec` names it (repository, revision, folder, the sha256 of its parameter file and,
where a config states it, of its ``config.json``), :func:`load_dictionary` reads it from local files only and checks it
against the receiver and the layer it is used at, and :class:`Dictionary` encodes and decodes. :func:`encode_states`
applies it to a set of states in bounded chunks and keeps the active features only, with each state's reconstruction
error; :func:`reconstruction_check` summarizes that. Its parameters never require gradients; gradients pass through it
to the states it reads.

A **state edit** (:class:`StateEdit`) is the opted-in recipe: which operation, at which receiver layer, at which
positions, in which order with the write, with which dictionary, and what becomes of the reconstruction error. The one
operation is ``splice`` (:func:`splice`): at the input of decoder layer L, after the write at L if there is one, the
states at the pass's written positions (:func:`written_sites`: every position a connector writes at, at any layer, so
a layer without a write of its own can be edited too) are replaced by the dictionary's reconstruction, and the
reconstruction error is dropped. Every edit returns an :class:`EditReport` (per position: the norm of the dropped
error, the norm of the state, the number of active features). Positions are prompt positions of prefill and
teacher-forced passes, as writes are; generated tokens are never edited.

Diagnostic, not paper: nothing here is on unless a config sets ``edit`` (``evaluation.suites.SuiteRunConfig``),
and a config without it keeps its digest (``config.added_field``).

**The file format**, from the published files and loader: a folder
``<site>/layer_<r>_width_<w>_l0_<s>`` holding ``config.json`` (keys read here: ``architecture`` = ``"jump_relu"``,
``type`` = ``"sae"``, ``affine_connection`` = false, ``model_name``, ``width``, and ``hf_hook_point_in`` =
``hf_hook_point_out`` = ``"model.layers.<r>.output"``) and ``params.safetensors`` with ``w_enc`` ``(d_model, width)``,
``b_enc`` ``(width,)``, ``threshold`` ``(width,)``, ``w_dec`` ``(width, d_model)`` and ``b_dec`` ``(d_model,)``. The
published loader (SAELens' Gemma 3 loader) applies no input normalization and does not subtract the decoder bias from
the input, so neither is applied here.

**Which layer.** The output of decoder layer r is ``hidden_states[r + 1]``, the input of decoder layer r + 1, where a
write at r + 1 applies. So an edit at receiver layer L needs a dictionary whose own ``config.json`` names the output of
layer L - 1; :func:`load_dictionary` reads that from the file and refuses any other (no offset is assumed), and an edit
at layer 0 is refused (no dictionary reads the embedding output).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field, field_validator
from safetensors.torch import load_file
from torch import Tensor

from latent_interconnect.canonical import is_sha256
from latent_interconnect.config import Config, added_field
from latent_interconnect.receivers.kernel import LayerWrites

__all__ = [
    "CONFIG_FILE",
    "PARAMS_FILE",
    "Dictionary",
    "DictionarySpec",
    "EditReport",
    "ReconstructionCheck",
    "StateCodes",
    "StateEdit",
    "encode_states",
    "load_dictionary",
    "read_config",
    "reconstruction_check",
    "splice",
    "written_sites",
]

CONFIG_FILE = "config.json"
PARAMS_FILE = "params.safetensors"
_HOOK = re.compile(r"^model\.layers\.(\d+)\.output$")
_SHAPES = {"w_enc": "dw", "b_enc": "w", "threshold": "w", "w_dec": "wd", "b_dec": "d"}
"""The parameter tensors and their shapes (d: the receiver's width, w: the dictionary's)."""
_HASH_BLOCK = 1 << 20
CHUNK_ELEMENTS = 1 << 26
"""The most feature activations :func:`encode_states` holds densely at once (states per chunk times the dictionary's
width): 256 MB in float32 per dense temporary, so a dictionary of width 2^20 is applied 64 states at a time."""


class DictionarySpec(Config):
    """One published dictionary, as data: where it is and which bytes it is.

    Attributes:
        repository: the Hugging Face repository id (``google/gemma-scope-2-4b-it``), or a local directory holding the
            same layout (the tests' synthetic dictionary).
        revision: the repository's immutable revision (commit sha); for a local directory, a label of its origin.
        folder: the dictionary's folder inside the repository (``resid_post_all/layer_6_width_16k_l0_small``): it names
            the family, layer, width and sparsity, and two families do not share feature indices.
        sha256: the sha256 of the folder's ``params.safetensors``, checked when the dictionary loads.
        config_sha256: the sha256 of the folder's ``config.json`` (the hook, the model, the width and the declared
            ``l0`` are read from it), checked when the dictionary loads; None: not checked. An added field
            (``config.added_field``): a spec without it keeps its digest.

    For a Hugging Face repository the files are read from the local cache's snapshot of exactly ``revision``: a
    revision that is not that snapshot's commit (a branch name, a tag) is refused (:func:`load_dictionary`).
    """

    repository: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    folder: str = Field(min_length=1)
    sha256: str
    config_sha256: str | None = added_field(default=None)

    @field_validator("sha256", "config_sha256")
    @classmethod
    def _hex64(cls, value: str | None) -> str | None:
        if value is not None and not is_sha256(value):
            raise ValueError(f"not a sha256 hex digest: {value!r}")
        return value


class StateEdit(Config):
    """An opted-in edit of the receiver's states: the recipe a config states and a record keeps.

    Attributes:
        operation: ``"splice"``: replace each edited state by the dictionary's reconstruction of it, dropping the
            reconstruction error (:func:`splice`).
        layer: the receiver layer L, at least 1: the edit applies at the input of decoder layer L
            (``hidden_states[L]``).
        order: ``"after_write"``: the edit reads the state the write at ``layer`` leaves (the write rule first, then
            the edit); at a layer the pass does not write at, the state as it arrives.
        positions: ``"written"``: every position the pass writes at, at any layer (:func:`written_sites`: the slot
            tokens its connectors write, the layer-0 constant's included), and no other. A pass that writes nothing
            (a prompt without a bound slot, a decoding step after prefill) is not edited.
        error: ``"dropped"``: the reconstruction error ``h - decode(encode(h))`` is not added back.
        dictionary: the dictionary; its own config must name the output of decoder layer ``layer - 1``
            (:func:`load_dictionary`).
    """

    operation: Literal["splice"]
    layer: int = Field(ge=1)
    order: Literal["after_write"]
    positions: Literal["written"]
    error: Literal["dropped"]
    dictionary: DictionarySpec


@dataclass(frozen=True)
class Dictionary:
    """A loaded JumpReLU dictionary: frozen parameters, differentiable in its input.

    Attributes:
        w_enc: ``(d_model, width)`` encoder weights.
        b_enc: ``(width,)`` encoder bias.
        threshold: ``(width,)`` the JumpReLU thresholds.
        w_dec: ``(width, d_model)`` decoder weights.
        b_dec: ``(d_model,)`` decoder bias.

    No parameter requires gradients. The math runs in ``promote_types(input dtype, float32)``, as the write rule's.
    """

    w_enc: Tensor
    b_enc: Tensor
    threshold: Tensor
    w_dec: Tensor
    b_dec: Tensor

    @property
    def d_model(self) -> int:
        """The width of the states it reads and reconstructs."""
        return int(self.w_enc.shape[0])

    @property
    def width(self) -> int:
        """The number of features."""
        return int(self.w_enc.shape[1])

    def to(self, device: torch.device) -> Dictionary:
        """Return the dictionary with its parameters on ``device``."""
        return Dictionary(*(getattr(self, name).to(device) for name in _SHAPES))

    def encode(self, states: Tensor) -> Tensor:
        """Return the feature activations ``relu(pre) · [pre > threshold]`` with ``pre = states @ w_enc + b_enc``.

        Equal to ``pre · [pre > threshold]`` wherever the threshold is positive; a feature whose threshold is negative
        is zero at a negative pre-activation. A pre-activation equal to its threshold is inactive. The gate passes no
        gradient: an inactive feature gives its input none.

        Args:
            states: ``(..., d_model)``.

        Returns:
            ``(..., width)`` in ``promote_types(states.dtype, float32)``.
        """
        dtype = torch.promote_types(states.dtype, torch.float32)
        pre = states.to(dtype) @ self.w_enc.to(dtype) + self.b_enc.to(dtype)
        return torch.relu(pre) * (pre > self.threshold.to(dtype))

    def decode(self, features: Tensor) -> Tensor:
        """Return the reconstruction ``features @ w_dec + b_dec``, ``(..., d_model)`` in ``features``' dtype."""
        dtype = features.dtype
        return features @ self.w_dec.to(dtype) + self.b_dec.to(dtype)


def _file(spec: DictionarySpec, name: str) -> Path:
    """Return the local path of one of the dictionary's files; nothing is ever downloaded."""
    local = Path(spec.repository)
    if local.is_dir():
        path = local / spec.folder / name
        if not path.is_file():
            raise FileNotFoundError(f"dictionary file {path} does not exist")
        return path
    from huggingface_hub import hf_hub_download  # noqa: PLC0415  (the optional `hf` extra; core imports without it)

    path = Path(
        hf_hub_download(spec.repository, f"{spec.folder}/{name}", revision=spec.revision, local_files_only=True)
    )
    if spec.revision not in path.parts:  # the cache resolved a branch or tag: the bytes' commit is not the spec's
        raise ValueError(
            f"dictionary {spec.folder!r}: revision {spec.revision!r} is not the commit of the cached snapshot "
            f"{path.parent}; name the commit itself"
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def read_config(spec: DictionarySpec) -> dict[str, Any]:
    """Return the dictionary's ``config.json`` (its hook, model, width, declared ``l0``), from local files only.

    Raises:
        FileNotFoundError: if the file is not local.
        ValueError: if the spec names a ``config_sha256`` and the file is not those bytes, or (a Hugging Face
            repository) ``spec.revision`` is not the cached snapshot's commit.
    """
    path = _file(spec, CONFIG_FILE)
    if spec.config_sha256 is not None and (found := _sha256(path)) != spec.config_sha256:
        raise ValueError(
            f"dictionary {spec.folder!r}: {CONFIG_FILE} has sha256 {found}, the spec names {spec.config_sha256}"
        )
    config: dict[str, Any] = json.loads(path.read_text())
    return config


def load_dictionary(spec: DictionarySpec, *, model_id: str, layer: int, d_model: int) -> Dictionary:
    """Load a dictionary from local files and check it against the receiver and the layer it is used at.

    The files come from the local directory ``spec.repository``, or from the Hugging Face cache at ``spec.revision``
    (``local_files_only``: a file that is not there is an error, never a download).

    Args:
        spec: the dictionary.
        model_id: the receiver's ``ReceiverSpec.model_id``; the dictionary's ``model_name`` must equal it.
        layer: the receiver layer L whose input the dictionary is applied to; the dictionary's hook must be the
            output of decoder layer L - 1.
        d_model: the receiver's width.

    Returns:
        The dictionary, float32 on the CPU, no parameter requiring gradients.

    Raises:
        FileNotFoundError: if a file is not local (``huggingface_hub``'s ``LocalEntryNotFoundError`` is one).
        ValueError: if the parameter file (or, where the spec names its sha256, the config file) is not the bytes the
            spec names, the revision is not the cached snapshot's commit (:func:`read_config`), the config is not a
            JumpReLU dictionary without a skip connection for this model and layer, or the tensors are not the five
            expected ones in float32 with matching shapes.
    """
    params_path = _file(spec, PARAMS_FILE)
    found = _sha256(params_path)
    if found != spec.sha256:
        raise ValueError(f"dictionary {spec.folder!r}: {PARAMS_FILE} has sha256 {found}, the spec names {spec.sha256}")
    config = read_config(spec)
    want = {
        "architecture": "jump_relu",
        "type": "sae",
        "affine_connection": False,
        "model_name": model_id,
        "hf_hook_point_in": f"model.layers.{layer - 1}.output",
        "hf_hook_point_out": f"model.layers.{layer - 1}.output",
    }
    wrong = {key: config.get(key) for key, value in want.items() if config.get(key) != value}
    if wrong:
        raise ValueError(
            f"dictionary {spec.folder!r} cannot be applied at the input of layer {layer} of {model_id!r}: its config "
            f"has {wrong}, needed { ({k: want[k] for k in wrong}) }"
        )
    if _HOOK.match(str(config["hf_hook_point_in"])) is None:
        raise ValueError(f"dictionary {spec.folder!r}: hook {config['hf_hook_point_in']!r} is not a layer's output")
    tensors = load_file(params_path)
    if set(tensors) != set(_SHAPES):
        raise ValueError(f"dictionary {spec.folder!r}: tensors {sorted(tensors)}, expected {sorted(_SHAPES)}")
    width = config.get("width")
    sizes = {"d": d_model, "w": width}
    bad = {
        name: (tuple(tensors[name].shape), str(tensors[name].dtype))
        for name, dims in _SHAPES.items()
        if tuple(tensors[name].shape) != tuple(sizes[c] for c in dims) or tensors[name].dtype != torch.float32
    }
    if bad:
        raise ValueError(
            f"dictionary {spec.folder!r}: tensors {bad} are not float32 of the shapes of width {width} (its config's) "
            f"and d_model {d_model} (the receiver's)"
        )
    return Dictionary(*(tensors[name].requires_grad_(False) for name in _SHAPES))


@dataclass(frozen=True)
class EditReport:
    """What one edit of one forward pass did, per edited position (detached, on the CPU).

    Attributes:
        rows: ``(M,)`` int64, the row of each edited position in the pass's batch (a micro-batch of the call).
        positions: ``(M,)`` int64, its token position.
        error_norm: ``(M,)`` the norm of the dropped reconstruction error, ``|h - decode(encode(h))|``, with h the
            state after the write; computed before the reconstruction is cast to the stream dtype.
        state_norm: ``(M,)`` the norm of h.
        active: ``(M,)`` int64, the number of active features.
    """

    rows: Tensor
    positions: Tensor
    error_norm: Tensor
    state_norm: Tensor
    active: Tensor


def written_sites(writes: Mapping[int, LayerWrites]) -> tuple[Tensor, Tensor]:
    """Return the positions a pass writes at, at any layer: ``(rows, positions)``, each once, by row then position.

    Args:
        writes: layer → the pass's writes (a micro-batch's, ``kernel.select_writes``).

    Returns:
        Two ``(M,)`` int64 tensors on the CPU; empty when the pass writes nothing.
    """
    sites = sorted({(int(r), int(p)) for w in writes.values() for r, p in zip(w.rows, w.positions, strict=True)})
    rows = torch.tensor([r for r, _ in sites], dtype=torch.int64)
    return rows, torch.tensor([p for _, p in sites], dtype=torch.int64)


def splice(hidden: Tensor, rows: Tensor, positions: Tensor, dictionary: Dictionary) -> tuple[Tensor, EditReport]:
    """Return ``hidden`` with the states at the sites replaced by their reconstruction, and the report.

    Out of place and differentiable in ``hidden``: gradients reach the sites' states through the dictionary's active
    features, and no other position's state changes.

    Args:
        hidden: ``(N, S, d_model)`` states at the input of the edited layer, after its write.
        rows: ``(M,)`` int64, the row of each site (distinct sites).
        positions: ``(M,)`` int64, its token position.
        dictionary: the dictionary, on ``hidden``'s device.
    """
    at = (rows.to(hidden.device), positions.to(hidden.device))
    rows, positions = rows.cpu(), positions.cpu()
    states = hidden[at].to(torch.promote_types(hidden.dtype, torch.float32))
    features = dictionary.encode(states)
    rebuilt = dictionary.decode(features)
    out = hidden.clone()
    out[at] = rebuilt.to(hidden.dtype)
    report = EditReport(
        rows=rows,
        positions=positions,
        error_norm=(states - rebuilt).detach().norm(dim=-1).cpu(),
        state_norm=states.detach().norm(dim=-1).cpu(),
        active=(features.detach() > 0).sum(-1).cpu(),
    )
    return out, report


@dataclass(frozen=True)
class StateCodes:
    """A dictionary applied to a set of states: the active features (sparse) and each state's reconstruction error.

    Attributes:
        rows: ``(m,)`` int64, the state (its row) of each active feature; sorted by row, then feature.
        features: ``(m,)`` int64, the feature index.
        values: ``(m,)`` float32, its activation (positive; every other activation is zero).
        error_sq: ``(n,)`` float64, each state's squared reconstruction error ``|h - decode(encode(h))|²``.
        norm_sq: ``(n,)`` float64, each state's squared norm ``|h|²``.
        active: ``(n,)`` int64, each state's number of active features.
    """

    rows: Tensor
    features: Tensor
    values: Tensor
    error_sq: Tensor
    norm_sq: Tensor
    active: Tensor


def encode_states(dictionary: Dictionary, states: Tensor) -> StateCodes:
    """Encode and reconstruct ``states`` in chunks of at most :data:`CHUNK_ELEMENTS` dense activations.

    The reconstruction is ``decode(encode(h))``; the error is not added back anywhere. No gradient is kept.

    Args:
        dictionary: the dictionary.
        states: ``(n, d_model)`` states.

    Raises:
        ValueError: if ``states`` is not ``(n, d_model)`` of the dictionary's width.
    """
    if states.ndim != 2 or states.shape[1] != dictionary.d_model:  # noqa: PLR2004 - a matrix of states
        raise ValueError(f"states of shape {tuple(states.shape)}: (n, {dictionary.d_model}) is needed")
    step = max(1, CHUNK_ELEMENTS // dictionary.width)
    rows, features, values, error_sq, norm_sq, active = [], [], [], [], [], []
    with torch.no_grad():
        for start in range(0, states.shape[0], step):
            x = states[start : start + step].to(torch.promote_types(states.dtype, torch.float32))
            z = dictionary.encode(x)
            error_sq.append((x - dictionary.decode(z)).double().square().sum(-1))
            norm_sq.append(x.double().square().sum(-1))
            at = z.nonzero(as_tuple=True)
            rows.append(at[0] + start)
            features.append(at[1])
            values.append(z[at].float())
            active.append((z > 0).sum(-1))
    return StateCodes(*(torch.cat(part) for part in (rows, features, values, error_sq, norm_sq, active)))


@dataclass(frozen=True)
class ReconstructionCheck:
    """How well a dictionary reconstructs a set of states (:func:`reconstruction_check`).

    Attributes:
        n: the number of states.
        unexplained: the fraction of variance unexplained, ``Σ|h - ĥ|² / Σ|h - mean(h)|²`` (0: exact; 1: no better
            than the mean state).
        relative_error: the mean over states of ``|h - ĥ| / |h|``.
        active: the mean number of active features per state (to set beside the ``l0`` its config declares).
    """

    n: int
    unexplained: float
    relative_error: float
    active: float

    @classmethod
    def of(cls, states: Tensor, codes: StateCodes) -> ReconstructionCheck:
        """Return the summary of ``codes``, a dictionary's :func:`encode_states` of ``states`` (sums in float64).

        Raises:
            ValueError: if ``states`` is not ``(n ≥ 2, d)`` with one row per coded state, or has no variance.
        """
        if states.ndim != 2 or states.shape[0] < 2 or states.shape[0] != codes.error_sq.shape[0]:  # noqa: PLR2004
            raise ValueError(f"states of shape {tuple(states.shape)}: at least two, one per coded state, are needed")
        x = states.detach().double()
        total = float((x - x.mean(0)).square().sum())
        if not total > 0:
            raise ValueError("the states have no variance: the fraction unexplained is undefined")
        return cls(
            n=int(x.shape[0]),
            unexplained=float(codes.error_sq.sum()) / total,
            relative_error=float((codes.error_sq / codes.norm_sq).sqrt().mean()),
            active=float(codes.active.double().mean()),
        )


def reconstruction_check(dictionary: Dictionary, states: Tensor) -> ReconstructionCheck:
    """Measure a dictionary's reconstruction of states it was trained to read: the first check on real weights.

    The summary (:meth:`ReconstructionCheck.of`) of :func:`encode_states`. Run it on the receiver's own states at the
    dictionary's site on ordinary text (``hidden_states[L]`` of an unwritten pass, e.g. a capture): a dictionary
    applied at the right site with the right preprocessing leaves a small fraction unexplained and activates about as
    many features as its config declares; a wrong layer or a missing normalization shows as a large one. It reports
    the measurement; the reader judges it.

    Args:
        dictionary: the dictionary.
        states: ``(n, d_model)`` states, at least two, not all equal.

    Raises:
        ValueError: if ``states`` is not ``(n ≥ 2, d_model)`` or has no variance.
    """
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] != dictionary.d_model:  # noqa: PLR2004
        raise ValueError(
            f"states of shape {tuple(states.shape)}: at least two of width {dictionary.d_model} are needed"
        )
    return ReconstructionCheck.of(states, encode_states(dictionary, states))
