"""Write layers: which (slot token, receiver layer) entries a connector writes, stated explicitly in its config.

Every connector writes at layer 0 (the slot token's input embedding, ``ConnectorSpec.layer0``) and at the entries
listed here, all at layers ≥ 1:

- ``explicit``: per-token layer lists;
- ``none``: no entries above layer 0 (the layer-0 ablation);
- ``pending``: not chosen yet (chosen with the help of the layer-selection scans). A pending connector config
  validates but cannot be built, and an init or run config that embeds it is refused at load
  (``build.init_state.InitSpec``).

There is deliberately no formula that derives layers. Layers are chosen by a person with the help of the scans in
``evaluation.layer_selection`` and the guidance in ``examples/guides/choosing_layers.md``, and written into the config.

Layer numbers follow the receiver convention: layer L is ``hidden_states[L]``, the input of decoder layer L.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from latent_interconnect.config import Config

__all__ = ["Entry", "ExplicitLayers", "LayerRule", "NoLayers", "PendingError", "PendingLayers", "entries"]


class Entry(tuple[int, int]):
    """A write site of a connector: ``(token, layer)``. Sorting entries orders them by token, then layer."""

    __slots__ = ()

    def __new__(cls, token: int, layer: int) -> Entry:
        """Create the entry of slot token ``token`` at receiver layer ``layer``."""
        return super().__new__(cls, (token, layer))

    @property
    def token(self) -> int:
        """Index of the slot token (0 .. k-1) within the slot."""
        return self[0]

    @property
    def layer(self) -> int:
        """Receiver layer: the write is applied at the input of decoder layer ``layer``."""
        return self[1]

    def key(self) -> str:
        """Return a stable name for parameter and tensor keys, e.g. ``t2_L12`` (a valid module name)."""
        return f"t{self.token}_L{self.layer}"


class ExplicitLayers(Config):
    """Per-token layer lists, written out.

    Attributes:
        rule: always ``"explicit"``.
        per_token: ``per_token[t]`` = token t's layers, strictly increasing, all ≥ 1.
    """

    rule: Literal["explicit"] = "explicit"
    per_token: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def _check(self) -> ExplicitLayers:
        for t, layers in enumerate(self.per_token):
            if list(layers) != sorted(set(layers)) or any(L < 1 for L in layers):
                raise ValueError(f"token {t}: layers must be strictly increasing and >= 1, got {layers}")
        return self


class NoLayers(Config):
    """No entries above layer 0: the connector writes only at the slot's input embedding.

    Attributes:
        rule: always ``"none"``.
    """

    rule: Literal["none"] = "none"


class PendingLayers(Config):
    """Write layers still to be chosen from the layer-selection scans.

    Attributes:
        rule: always ``"pending"``.
    """

    rule: Literal["pending"] = "pending"


class PendingError(ValueError):
    """A config still has a pending choice, so it cannot be built or run."""


LayerRule = Annotated[ExplicitLayers | NoLayers | PendingLayers, Field(discriminator="rule")]


def entries(rule: ExplicitLayers | NoLayers | PendingLayers, k: int, depth: int) -> tuple[Entry, ...]:
    """Resolve the configured layers into the connector's entries above layer 0, sorted by (token, layer).

    Args:
        rule: the layers from the connector config.
        k: number of slot tokens.
        depth: number of receiver decoder layers; valid write layers are 1 .. depth - 1.

    Raises:
        ValueError: if the lists do not fit k or the receiver depth.
        PendingError: if the layers are still pending.
    """
    if isinstance(rule, PendingLayers):
        raise PendingError("write layers are pending: choose them from the layer-selection scans")
    if isinstance(rule, NoLayers):
        return ()
    if len(rule.per_token) != k:
        raise ValueError(f"explicit layers list {len(rule.per_token)} tokens, the connector has k = {k}")
    out = [Entry(t, L) for t, layers in enumerate(rule.per_token) for L in layers]
    bad = sorted({e.layer for e in out if not 1 <= e.layer < depth})
    if bad:
        raise ValueError(f"layers {bad} are outside 1 .. {depth - 1} for a receiver with {depth} decoder layers")
    return tuple(sorted(out))
