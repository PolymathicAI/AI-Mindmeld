"""Extension path 1: a new connector architecture (a gated two-layer map), registered as map kind ``example-gated``."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from latent_interconnect.config import Config
from latent_interconnect.writes.init import EntryInit, InitContext
from latent_interconnect.writes.inputs import InputKind
from latent_interconnect.writes.kinds import MAP_KINDS, EntryMap


class GatedParams(Config):
    """Hyperparameters of the gated map.

    Attributes:
        hidden: width of the gate.
    """

    hidden: int


@MAP_KINDS.register("example-gated")
class Gated(EntryMap):
    """δ = (W z + b) · sigmoid(g(z)); starts as the ridge map with the gate fully open."""

    input_kind = InputKind.POOLED
    Params = GatedParams

    def __init__(self, W: Tensor, b: Tensor, hidden: int) -> None:
        super().__init__()
        self.W, self.b = nn.Parameter(W), nn.Parameter(b)
        self.gate = nn.Sequential(nn.Linear(W.shape[1], hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, 6.0)  # sigmoid(6) ≈ 1: the gate starts open

    @classmethod
    def create(cls, params: Any, init: EntryInit, ctx: InitContext) -> Gated:
        """Start from the entry's ridge map."""
        del ctx
        return cls(torch.as_tensor(init.ridge.W).clone(), torch.as_tensor(init.ridge.b).clone(), params.hidden)

    @classmethod
    def empty(cls, params: Any, d_in: int, d_model: int) -> Gated:
        """Shapes only, for loading a checkpoint."""
        return cls(torch.zeros(d_model, d_in), torch.zeros(d_model), params.hidden)

    def lr_reference(self, name: str, init_value: Tensor) -> float:
        """The gate's output layer starts at zero, so it takes the scale of a unit-variance input projection."""
        if name == "gate.2.weight":
            return float(self.gate[2].in_features ** -0.5)
        return super().lr_reference(name, init_value)

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        """Return the gated affine write."""
        del mask
        return (x @ self.W.T + self.b) * torch.sigmoid(self.gate(x))
