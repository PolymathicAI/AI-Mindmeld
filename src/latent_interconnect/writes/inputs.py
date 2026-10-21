"""What a connector reads: one batch of objects as pooled vectors and/or per-residue (per-token) states.

The encoder store produces :class:`ObjectBatch`; connectors consume it. Raw (unstandardized) values: standardization
is part of the connector (one μ, σ per connector, fixed at initialization).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol, Self

import torch
from pydantic import model_validator
from torch import Tensor

from latent_interconnect.config import Config, added_field

__all__ = ["AppendedLayer", "EncoderShape", "InputKind", "InputSpec", "ObjectBatch", "check_read_layers"]


class InputKind(StrEnum):
    """The shape of encoder input a map kind needs."""

    NONE = "none"
    """Object-independent (the layer-0 constant): the map receives a ``(B, 0)`` tensor."""
    POOLED = "pooled"
    """One vector per object, ``(B, d_in)``."""
    RESIDUES = "residues"
    """The real residues of the batch packed object by object, ``(N, d_in)`` (``residues[mask]`` of
    :class:`ObjectBatch`), with its validity mask ``(B, T)``, whose row sums are the objects' counts."""


@dataclass(frozen=True)
class ObjectBatch:
    """Encoder outputs for a batch of B objects.

    Attributes:
        object_ids: the B object ids, in batch order.
        pooled: ``(B, d_in)`` float32 pooled vectors, or None if not loaded.
        residues: ``(B, T, d_in)`` per-residue states (bf16 or float32), or None if not loaded.
        mask: ``(B, T)`` bool, True at real residues, at least one per object; required with ``residues``.
    """

    object_ids: tuple[str, ...]
    pooled: Tensor | None = None
    residues: Tensor | None = None
    mask: Tensor | None = None

    def __post_init__(self) -> None:
        if self.pooled is None and self.residues is None:
            raise ValueError("an ObjectBatch needs pooled vectors, residues, or both")
        if (self.residues is None) != (self.mask is None):
            raise ValueError("residues and mask go together")
        b = len(self.object_ids)
        for name, t in (("pooled", self.pooled), ("residues", self.residues), ("mask", self.mask)):
            if t is not None and t.shape[0] != b:
                raise ValueError(f"{name} has batch size {t.shape[0]}, expected {b}")
        if self.residues is not None and self.mask is not None:
            if self.mask.dtype != torch.bool or self.mask.shape != self.residues.shape[:2]:
                raise ValueError(
                    f"mask must be bool of shape {tuple(self.residues.shape[:2])}, not {self.mask.dtype} "
                    f"{tuple(self.mask.shape)}"
                )
            empty = [o for o, real in zip(self.object_ids, self.mask.any(dim=1).tolist(), strict=True) if not real]
            if empty:
                raise ValueError(f"objects without residues: {empty}")

    def require(self, kind: InputKind) -> Tensor:
        """Return the tensor a map kind needs: ``(B, 0)`` for NONE, else ``pooled`` or ``residues``.

        Raises:
            ValueError: if that input was not loaded.
        """
        if kind is InputKind.NONE:
            return torch.empty(len(self.object_ids), 0)
        t = self.pooled if kind is InputKind.POOLED else self.residues
        if t is None:
            raise ValueError(f"this batch has no {kind.value} input")
        return t


class AppendedLayer(Config):
    """A second extraction's pooled vector at one layer, appended after an input's.

    Its artifacts are among the ones the input is loaded from (a ridge's, a scan's or a run's ``embeddings``), told
    apart by the recipe name they record; all of them must record one recipe (``encoders.loader.InputLoader``).

    Attributes:
        recipe: the second extraction's encoder recipe name (not the input's own).
        layer: the one layer read, its pooled vector whole.
    """

    recipe: str
    layer: int


class InputSpec(Config):
    """What a connector reads from the embedding store.

    Attributes:
        recipe: the encoder recipe name.
        layers: the ``hidden_states`` indices read, explicitly, a subset of the recipe's ``read`` without repeats
            (:func:`check_read_layers`); ``()`` only for a recipe whose ``read`` is ``"output"``; ``"pending"`` until
            the read-layer scan decides.
        combine: how several layers form one input (``concat`` along features).
        append: a second extraction's pooled vector at one layer, concatenated after the input's per object (aligned
            by object id; pooled inputs only), or None.

    Whether pooled vectors, residues or both are loaded follows from the connector's map kinds
    (:attr:`~latent_interconnect.writes.connector.Connector.input_kinds`), so it is not stated twice.
    """

    recipe: str
    layers: tuple[int, ...] | Literal["pending"] = ()
    combine: Literal["single", "concat"] = "single"
    append: AppendedLayer | None = added_field(default=None)

    @model_validator(mode="after")
    def _check_append(self) -> Self:
        if self.append is not None and self.append.recipe == self.recipe:
            raise ValueError(f"the appended extraction must be another recipe than {self.recipe!r}")
        return self


class EncoderShape(Protocol):
    """What the read-layer check and a checkpoint need of an encoder recipe.

    ``encoders.recipes.EncoderRecipe`` satisfies it; ``writes`` comes before ``encoders`` in the import order.
    """

    @property
    def name(self) -> str:
        """The recipe's name (what a connector's ``input.recipe`` cites)."""
        ...

    @property
    def read(self) -> tuple[int, ...] | str:
        """The recipe's ``hidden_states`` indices, or ``"all"``, ``"output"`` or ``"pending"``."""
        ...

    def identity(self) -> dict[str, Any]:
        """The recipe's identity form (``Config.identity``): what a checkpoint records of its encoder."""
        ...


def check_read_layers(spec: InputSpec, encoder: EncoderShape) -> tuple[int, ...]:
    """Check a connector's input against its encoder recipe: the one read-layer rule.

    Refuses, in this order: another recipe; a ``"pending"`` read or ``"pending"`` layers; repeated indices; indices
    outside an explicit ``read``; layers against an ``"output"`` recipe; no layers against any other recipe. Called
    wherever a connector's input meets its recipe: ``InitSpec`` validation (the ``init`` stage, and every run through
    ``RunConfig.init_spec()``), the input loader
    (``encoders.loader.InputLoader``, so every load) and the configs test.

    Returns:
        The checked layers (``()`` for an ``"output"`` recipe), so callers hold them as indices, never ``"pending"``.

    Raises:
        ValueError: naming the rule and the offending values.
    """
    layers, read = spec.layers, encoder.read
    if spec.recipe != encoder.name:
        raise ValueError(f"the input reads recipe {spec.recipe!r}, not {encoder.name!r}")
    if read == "pending" or layers == "pending":
        raise ValueError(f"{encoder.name!r}: read {read!r}, layers {layers!r}; choose the layers first")
    if len(set(layers)) != len(layers):
        raise ValueError(f"{encoder.name!r}: layers {layers} repeat an index")
    if isinstance(read, tuple) and not set(layers) <= set(read):
        raise ValueError(f"{encoder.name!r}: layers {layers} are not all within its read {read}")
    if read == "output" and layers:
        raise ValueError(f"{encoder.name!r} reads its output: its layers are (), not {layers}")
    if read != "output" and not layers:
        raise ValueError(f"{encoder.name!r} reads {read!r}: state the layers explicitly, not ()")
    return layers
