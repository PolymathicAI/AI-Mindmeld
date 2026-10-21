"""The write rule and the connector model.

- ``rule``: ``interp_v1``, the one write rule.
- ``layers``: explicit write layers → (token, layer) entries.
- ``inputs``: :class:`ObjectBatch` and :class:`InputSpec`, what connectors read.
- ``init``: ridge solutions and per-entry seeds.
- ``kinds``: the registry of map kinds and the core kinds ``constant`` and ``affine``.
- ``blocks``: the building blocks of the stream kinds.
- ``kinds_mlp`` / ``kinds_residue``: the stream kinds ``mlp`` (pooled) and ``xattn`` / ``xblock`` (residues).
- ``connector``: :class:`ConnectorSpec` and :class:`Connector`.
- ``sources``: where a backend gets writes: a live connector or a precomputed bank.
- ``checkpoint``: the checkpoint format and retention policy.
"""

from latent_interconnect.writes import kinds_mlp as _kinds_mlp  # noqa: F401  (registers mlp)
from latent_interconnect.writes import kinds_residue as _kinds_residue  # noqa: F401  (registers xattn, xblock)
from latent_interconnect.writes.connector import (
    AlphaSpec,
    Connector,
    ConnectorSpec,
    LayerChoice,
    SlotWrites,
    TokenSpec,
)
from latent_interconnect.writes.inputs import InputKind, InputSpec, ObjectBatch
from latent_interconnect.writes.kinds import MAP_KINDS, EntryMap
from latent_interconnect.writes.layers import Entry, ExplicitLayers, NoLayers, PendingError, PendingLayers, entries
from latent_interconnect.writes.rule import WRITE_RULE, interp

__all__ = [
    "MAP_KINDS",
    "WRITE_RULE",
    "AlphaSpec",
    "Connector",
    "ConnectorSpec",
    "Entry",
    "EntryMap",
    "ExplicitLayers",
    "InputKind",
    "InputSpec",
    "LayerChoice",
    "NoLayers",
    "ObjectBatch",
    "PendingError",
    "PendingLayers",
    "SlotWrites",
    "TokenSpec",
    "entries",
    "interp",
]
