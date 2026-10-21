"""Connectors: a config (:class:`ConnectorSpec`) and the module it builds (:class:`Connector`).

A connector with k slot tokens writes:

- at layer 0, for each token, a map of kind ``layer0`` (any registered kind; ``"none"`` for no write), with α = 1;
- at each entry (token, layer ≥ 1) of its explicit ``layers``, a map of its ``kind``. α is fixed at ``alpha.first``
  at each token's first entry (when set; None: that entry is like the others); elsewhere it is learned as
  sigmoid(θ) from ``alpha.init``, or from ``alpha.per_layer``'s value at the entry's layer (or fixed at that start
  when ``alpha.learned`` is off).

Two ways to build one:

- :meth:`Connector.from_ridge`: the initial state from ridge solutions, the one initial state of every run
  (``build.init_state``). Each token declares its own ridge target (``name``, ``caption``, ...), which initializes
  every entry of that token, its layer-0 constant included, whatever layers it writes to;
- :meth:`Connector.from_state`: a checkpoint's tensors (no ridge needed).

Every entry has its own map, of a kind registered in ``MAP_KINDS``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from latent_interconnect.config import Config, RecordKey, added_field, rationale
from latent_interconnect.plugins import load_plugins
from latent_interconnect.writes.init import EntryInit, InitContext, RidgeSolution, entry_seed
from latent_interconnect.writes.inputs import InputKind, InputSpec, ObjectBatch
from latent_interconnect.writes.kinds import MAP_KINDS, EntryMap
from latent_interconnect.writes.layers import Entry, LayerRule, PendingError, PendingLayers, entries

__all__ = [
    "AlphaSpec",
    "Connector",
    "ConnectorSpec",
    "LayerChoice",
    "SlotWrites",
    "TokenSpec",
    "initial_alphas",
    "learned_alpha_entries",
]


class TokenSpec(Config):
    """One slot token.

    Attributes:
        target: the ridge target that initializes every entry of this token, its layer-0 constant included (e.g.
            ``"name"``, ``"caption"``).
        noise: relative init noise on the ``noise_targets`` of this token's maps.
    """

    target: str
    noise: float = Field(default=0.0, ge=0.0)


class LayerChoice(Config):
    """Why the connector writes and reads where it does. Rationale only: excluded from the connector's digest.

    Attributes:
        records: computation keys of the layer-scan records the choice rests on.
        decisions: ids of the design decisions that fixed or constrain the choice.
        note: one or two sentences applying ``examples/guides/choosing_layers.md`` to those records.
    """

    records: tuple[RecordKey, ...] = ()
    decisions: tuple[str, ...] = ()
    note: str


class AlphaSpec(Config):
    """Mixing weights of the entries above layer 0.

    Attributes:
        first: fixed α at each token's first entry, or None: the first entry is like the others.
        init: initial α of every other entry (of every entry when ``first`` is None).
        learned: learn the other entries' α as sigmoid(θ); if false they stay at their start.
        budget: soft budget C on the sum of learned α (penalty ``penalty_k·relu(Σα − C)²``); None disables it.
        penalty_k: the penalty's k.
        per_layer: receiver layer → initial α of the entries at that layer, in (0, 1), in place of ``init`` (a
            per-layer start, e.g. lower mixing at the late write layers); every layer named must be written
            (``initial_alphas``). Not combined with ``first``. Empty: every entry starts at ``init`` (added field).
    """

    first: float | None = Field(default=None, gt=0.0, le=1.0)
    init: float = Field(gt=0.0, lt=1.0)
    learned: bool = True
    budget: float | None = None
    penalty_k: float = 5.0
    per_layer: dict[int, float] = added_field(default_factory=dict)

    @model_validator(mode="after")
    def _check_per_layer(self) -> AlphaSpec:
        bad = {L: a for L, a in self.per_layer.items() if L < 1 or not 0.0 < a < 1.0}
        if bad:
            raise ValueError(f"per_layer {bad}: layers >= 1, each start in (0, 1) (a learned sigmoid start)")
        if self.per_layer and self.first is not None:
            raise ValueError("per_layer and first are not combined: give the first entry's start in per_layer")
        return self


class ConnectorSpec(Config):
    """A connector, as data.

    Attributes:
        name: registry name of the connector.
        receiver: registered receiver name.
        input: what the connector reads from the embedding store.
        tokens: one entry per slot token (k = len(tokens)).
        init: always ``ridge``: every connector starts from its tokens' ridge targets (:meth:`Connector.from_ridge`);
            the field is part of every stored connector's identity.
        layer0: registered map kind written at layer 0, or ``"none"``.
        layers: explicit write layers above layer 0, ``none``, or pending.
        kind: registered map kind of the entries above layer 0.
        kind_params: the kind's hyperparameters (validated by ``kind.Params``).
        alpha: mixing weights of the entries above layer 0 (required when there are any).
        seed: base seed of every entry's initialization (the run config's seed governs data order only).
        plugins: modules to import before resolving kinds and the connector class.
        layer_choice: the evidence for ``layers`` and ``input.layers`` (rationale; not part of the digest, and absent
            from stored identity forms; every config in ``configs/connectors`` states it, checked by the tests).
    """

    name: str
    receiver: str
    input: InputSpec
    tokens: tuple[TokenSpec, ...]
    init: Literal["ridge"] = "ridge"
    layer0: str = "constant"
    layers: LayerRule
    kind: str
    kind_params: dict[str, Any] = Field(default_factory=dict)
    alpha: AlphaSpec | None = None
    seed: int = 0
    plugins: tuple[str, ...] = ()
    layer_choice: LayerChoice | None = rationale(default=None)

    def ridge_targets(self) -> frozenset[str]:
        """Return the ridge targets the tokens cite; a run supplies one ridge each."""
        return frozenset(t.target for t in self.tokens)

    def pending(self) -> list[str]:
        """Return the names of fields whose choice is still pending."""
        out = []
        if isinstance(self.layers, PendingLayers):
            out.append("layers")
        if self.input.layers == "pending":
            out.append("input.layers")
        return out

    def ready_tokens(self) -> tuple[TokenSpec, ...]:
        """Return the tokens of a ready config.

        Raises:
            PendingError: if any choice is pending.
        """
        pending = self.pending()
        if pending:
            raise PendingError(f"connector {self.name!r}: {', '.join(pending)} pending (run the layer-selection study)")
        return self.tokens

    @property
    def k(self) -> int:
        """Number of slot tokens (ready configs only)."""
        return len(self.ready_tokens())


@dataclass(frozen=True)
class SlotWrites:
    """A connector's writes for a batch of objects.

    Attributes:
        object_ids: batch order.
        deltas: entry → δ, ``(B, d_model)`` (layer-0 entries included).
        alphas: entry → α, a 0-d tensor on the entry's device.
    """

    object_ids: tuple[str, ...]
    deltas: Mapping[Entry, Tensor]
    alphas: Mapping[Entry, Tensor]


class Connector(nn.Module):
    """The module a :class:`ConnectorSpec` describes.

    Build it with :meth:`from_ridge` or :meth:`from_state`.

    Args:
        spec: the connector config (ready).
        depth: number of receiver decoder layers.
        d_in: encoder input width.
        d_model: receiver hidden width.
        maps: entry → its map, for every entry the spec implies.
    """

    def __init__(
        self,
        spec: ConnectorSpec,
        depth: int,
        d_in: int,
        d_model: int,
        *,
        maps: Mapping[Entry, EntryMap],
    ) -> None:
        super().__init__()
        self.spec, self.depth, self.d_in, self.d_model = spec, depth, d_in, d_model
        self.entries: tuple[Entry, ...] = expected_entries(spec, depth)
        if set(maps) != set(self.entries):
            raise ValueError(f"maps cover {sorted(maps)}; the spec implies {self.entries}")
        self.maps = nn.ModuleDict({e.key(): m for e, m in sorted(maps.items())})
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        fixed, free = self._alpha_plan(spec)
        for e, a in fixed.items():
            self.register_buffer(f"alpha_{e.key()}", torch.tensor(a))
        self.theta = nn.ParameterDict({e.key(): nn.Parameter(torch.tensor(logit)) for e, logit in free.items()})

    def _alpha_plan(self, spec: ConnectorSpec) -> tuple[dict[Entry, float], dict[Entry, float]]:
        """Return (fixed α per entry, initial logit per learned entry).

        Both follow :func:`initial_alphas` and :func:`learned_alpha_entries`.
        """
        if spec.alpha is None and any(e.layer > 0 for e in self.entries):  # checked here to name the connector
            raise ValueError(f"connector {spec.name!r} writes above layer 0, so it needs an alpha spec")
        start = initial_alphas(self.entries, spec.alpha)
        free = learned_alpha_entries(self.entries, spec.alpha)
        fixed = {e: a for e, a in start.items() if e not in free}
        return fixed, {e: math.log(start[e] / (1 - start[e])) for e in self.entries if e in free}

    @classmethod
    def from_ridge(cls, spec: ConnectorSpec, depth: int, ridge: Mapping[str, RidgeSolution]) -> Connector:
        """Build the initial (step-0) connector from the ridge solutions of its tokens' targets.

        Every entry of token t starts from ``ridge[tokens[t].target]`` at the entry's layer. The first token's target
        defines the standardization; other targets are re-expressed on it exactly.

        """
        load_plugins(spec.plugins)
        tokens = spec.ready_tokens()
        all_entries = expected_entries(spec, depth)
        base = ridge[_target(tokens, all_entries[0])]
        solved = {t: r.reexpress(base.mu, base.sd) for t, r in ridge.items()}
        some = next(iter(base.layers.values()))
        ctx = InitContext(d_in=some.W.shape[1], d_model=some.W.shape[0])
        inits = {}
        for e in all_entries:
            target = _target(tokens, e)
            gen = torch.Generator().manual_seed(entry_seed(spec.seed, e.token, e.layer))
            inits[e] = EntryInit(e.token, e.layer, solved[target].layers[e.layer], target, gen)
        maps = {}
        for e in all_entries:
            kind = _kind(spec, e)
            maps[e] = kind.create(_params(spec, e, kind), inits[e], ctx)
            _add_noise(maps[e], tokens[e.token].noise, inits[e].generator)
        out = cls(spec, depth, ctx.d_in, ctx.d_model, maps=maps)
        out.get_buffer("mu").copy_(torch.as_tensor(base.mu))
        out.get_buffer("sd").copy_(torch.as_tensor(base.sd))
        return out

    @classmethod
    def from_state(
        cls, spec: ConnectorSpec, depth: int, d_in: int, d_model: int, state: Mapping[str, Tensor]
    ) -> Connector:
        """Build a connector from a checkpoint's tensors (no ridge solutions needed)."""
        load_plugins(spec.plugins)
        maps = {
            e: _kind(spec, e).empty(_params(spec, e, _kind(spec, e)), d_in, d_model)
            for e in expected_entries(spec, depth)
        }
        out = cls(spec, depth, d_in, d_model, maps=maps)
        out.load_state_dict(dict(state))
        return out

    @property
    def input_kinds(self) -> frozenset[InputKind]:
        """The encoder inputs this connector reads (the loader loads exactly these)."""
        kinds = {cast(EntryMap, m).input_kind for m in self.maps.values()}
        return frozenset(kinds) - {InputKind.NONE}

    def alpha(self, entry: Entry) -> Tensor:
        """Return the entry's α as a 0-d tensor on the entry's device (differentiable when learned)."""
        if entry.key() in self.theta:
            return torch.sigmoid(self.theta[entry.key()])
        return self.get_buffer(f"alpha_{entry.key()}")

    def budget_penalty(self) -> Tensor:
        """Return the soft budget penalty on the learned α (0 when ``alpha.budget`` is None)."""
        a = self.spec.alpha
        if a is None or a.budget is None or not len(self.theta):
            return torch.zeros(())
        total = torch.stack([torch.sigmoid(p) for p in self.theta.values()]).sum()
        return a.penalty_k * torch.relu(total - a.budget) ** 2

    def forward(self, batch: ObjectBatch) -> SlotWrites:
        """Compute every entry's write vector; each map runs on its own device.

        Inputs are standardized once per device and shared by the maps there. Residue maps get the real residues only,
        packed once per call (``residues[mask]``, :attr:`InputKind.RESIDUES`), with the batch's own mask.
        """
        cache: dict[tuple[InputKind, torch.device], Tensor] = {}
        raw: dict[InputKind, Tensor] = {}

        def source(kind: InputKind) -> Tensor:
            if kind not in raw:
                t = batch.require(kind)
                raw[kind] = t[batch.mask] if kind is InputKind.RESIDUES else t
            return raw[kind]

        def inputs(kind: InputKind, module: nn.Module) -> tuple[Tensor, Tensor | None]:
            dev = _device(module)
            if kind is InputKind.NONE:
                return batch.require(InputKind.NONE).to(dev), None
            if (kind, dev) not in cache:
                cache[(kind, dev)] = self._standardize(source(kind), dev)
            return cache[(kind, dev)], batch.mask if kind is InputKind.RESIDUES else None

        deltas: dict[Entry, Tensor] = {}
        for key, module in self.maps.items():
            m = cast(EntryMap, module)
            deltas[_entry(key)] = m(*inputs(m.input_kind, m))
        return SlotWrites(batch.object_ids, deltas, {e: self.alpha(e) for e in self.entries})

    def _standardize(self, x: Tensor, dev: torch.device) -> Tensor:
        mu, sd = self.get_buffer("mu").to(dev), self.get_buffer("sd").to(dev)
        return (x.to(dev).float() - mu) / sd


def initial_alphas(entries: Sequence[Entry], alpha: AlphaSpec | None) -> dict[Entry, float]:
    """Return each entry's α at step 0 (the connector's schedule).

    1 at layer 0, ``alpha.first`` (if set) at each token's first entry above layer 0 and, at its other entries,
    ``alpha.per_layer``'s value at the entry's layer or else ``alpha.init``; a layer-selection set comparison writes the
    same.

    Raises:
        ValueError: if an entry is above layer 0 and ``alpha`` is None, or ``alpha.per_layer`` names a layer no entry
            writes.
    """
    out = {e: 1.0 for e in entries if e.layer == 0}
    deep = [e for e in entries if e.layer > 0]
    unwritten = sorted(set(alpha.per_layer) - {e.layer for e in deep}) if alpha is not None else []
    if unwritten:  # checked before the early return: a setting is never silently ignored
        raise ValueError(f"alpha.per_layer names layers {unwritten} that no entry writes")
    if not deep:
        return out
    if alpha is None:
        raise ValueError(f"entries above layer 0 ({deep[:3]}) need an alpha spec")
    firsts = {t: min(e.layer for e in deep if e.token == t) for t in {e.token for e in deep}}
    start = {e: alpha.per_layer.get(e.layer, alpha.init) for e in deep}
    if alpha.first is not None:
        start |= {e: alpha.first for e in deep if e.layer == firsts[e.token]}
    return out | start


def learned_alpha_entries(entries: Sequence[Entry], alpha: AlphaSpec | None) -> frozenset[Entry]:
    """Return the entries whose α is learned, when ``alpha.learned``: above layer 0 (but each token's first if set)."""
    if alpha is None or not alpha.learned:
        return frozenset()
    deep = [e for e in entries if e.layer > 0]
    if alpha.first is None:
        return frozenset(deep)
    firsts = {t: min(e.layer for e in deep if e.token == t) for t in {e.token for e in deep}}
    return frozenset(e for e in deep if e.layer != firsts[e.token])


def expected_entries(spec: ConnectorSpec, depth: int) -> tuple[Entry, ...]:
    """Return every entry a ready spec implies: layer 0 per token (unless ``layer0 == "none"``) plus ``layers``."""
    k = spec.k
    zero = () if spec.layer0 == "none" else tuple(Entry(t, 0) for t in range(k))
    return tuple(sorted(zero + entries(spec.layers, k, depth)))


def _target(tokens: tuple[TokenSpec, ...], e: Entry) -> str:
    target = tokens[e.token].target
    if target is None:
        raise ValueError(f"token {e.token} has no ridge target")
    return target


def _kind(spec: ConnectorSpec, e: Entry) -> type[EntryMap]:
    return MAP_KINDS.get(spec.layer0 if e.layer == 0 else spec.kind)


def _params(spec: ConnectorSpec, e: Entry, kind: type[EntryMap]) -> Config:
    return kind.Params.model_validate(spec.kind_params if e.layer else {})


def _entry(key: str) -> Entry:
    t, layer = key.removeprefix("t").split("_L")
    return Entry(int(t), int(layer))


def _device(m: nn.Module) -> torch.device:
    for p in m.parameters():
        return p.device
    return torch.device("cpu")


def _add_noise(m: EntryMap, noise: float, gen: torch.Generator) -> None:
    """Add ``noise × RMS(p) × N(0, 1)`` to each of the map's noise targets."""
    if not noise:
        return
    with torch.no_grad():
        for name in m.noise_targets():
            p = m.get_parameter(name)
            p.add_(noise * p.pow(2).mean().sqrt() * torch.randn(p.shape, generator=gen))
