"""Encoder recipes (orchestrator side): what to extract, validated, as data in ``configs/encoders/``.

A recipe names a registered extractor and fixes everything that changes the numbers: model revision, read layers,
pooling, dtypes, and the Python environment that runs it. The recipe's sha256 plus the object set's sha256 identify
the embedding artifact.

A connector reads the recipe's artifact through its own
:class:`~latent_interconnect.writes.inputs.InputSpec` (which layers, how combined).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from latent_interconnect.config import Config

__all__ = ["EncoderRecipe"]


class EncoderRecipe(Config):
    """How to extract one encoder's states.

    Attributes:
        name: registry name, e.g. ``"esmc-600m"``, ``"chemfm-1b"``.
        extractor: registered extractor (``encoders.base.EXTRACTORS``).
        domain: object domain (``"protein"``, ``"molecule"``, ...).
        model_id: model identifier (Hugging Face id or checkpoint name).
        revision: immutable model revision (``"pending"`` until it is pinned; a pending recipe cannot run).
        read: ``hidden_states`` indices to keep, ``"all"`` for every index (the read-layer scans), ``"output"`` for the
            encoder's defined output, or ``"pending"`` until the read-layer scan decides.
        pooling: how a pooled vector is formed from residue (token, atom) states. ``cls+mean`` and ``last+mean`` are a
            summary state (the first or the last token's) concatenated with the mean, summary first: the paper's
            molecule input. ``mean_all`` and ``eos`` are SMILES models' paper representations
            (``extractors._molecules``), ``registers`` and ``registers_mean`` MIMIC's register tokens concatenated and
            averaged (``extractors.mimic``).
        residues: also store per-residue states (needed by residue map kinds).
        residue_dtype: storage dtype of residue states.
        environment: name of the Python environment that runs the extractor (its interpreter is passed to the
            extract stage as ``--option python=...``, examples/guides/extending.md §6 Step 4).
        batch_tokens: token budget per extraction batch (batch composition can move low-precision states).
        options: extractor-specific options.
        plugins: modules the worker imports before resolving ``extractor`` (third-party extractors).
    """

    name: str
    extractor: str
    domain: str
    model_id: str
    revision: str
    read: tuple[int, ...] | Literal["all", "output", "pending"]
    pooling: Literal["mean", "cls", "last", "registers", "cls+mean", "last+mean", "mean_all", "eos", "registers_mean"]
    residues: bool = False
    residue_dtype: Literal["bf16", "fp32"] = "bf16"
    environment: str
    batch_tokens: int = Field(ge=1)
    options: dict[str, str | int | float | bool] = Field(default_factory=dict)
    plugins: tuple[str, ...] = ()
