"""The layer-set comparison: whole candidate write sets scored at initialization on the same items, paired.

Each candidate (:class:`LayerSet`) names the layers each slot token writes; every entry writes the object's state for
its layer (:mod:`latent_interconnect.evaluation.scan_states`: a capture or a ridge) at the connector's
step-0 α (``writes.connector.initial_alphas``: 1 at layer 0, ``alpha.first`` (if set) at each token's first entry
above it, ``alpha.init`` elsewhere). Only the α follow the connector: a set that lists layer 0 writes the object's
state there (a connector's default layer-0 write is a per-token constant), and a set that omits it writes nothing
there, unless it writes the connector's layer-0 constant (``layer0: constant``: the ridge's layer-0 intercept at α = 1,
as ``Connector.from_ridge`` initializes a ``constant`` layer 0). A ridge set may leave out the fit's intercept above
layer 0 (``bias: zero``: ``W z`` alone). So one comparison also scores whole connectors' step-0 α schedules, a
train-free test; :func:`connector_start` maps a set that is a trainable start onto its connector config, so the
``init`` stage and the trainer start from exactly the α it was scored at, and refuses the diagnostics (an omitted
entry, a fixed α, no intercept). :func:`auroc_weighted_alphas` computes one such schedule from single-layer scores
(data-selected, so exploratory), :func:`clipped_auroc_weighted_alphas` its literal clip to [0.1, 0.9] (the
same).
Every set is scored as a ``yes_no`` suite on the item set's ``own`` condition; per set the scan reports each task's
AUROC and the equal-weight mean over tasks, and per declared pair of sets the paired contrast of the means, all from
one bootstrap over every set's units (``evaluation.contrasts.cell_draws``), as estimates with intervals. It
compares; it chooses nothing (``examples/guides/choosing_layers.md``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.build.ridge import load_ridge
from latent_interconnect.config import Config, added_field
from latent_interconnect.evaluation.contrasts import cell_draws, unit_columns
from latent_interconnect.evaluation.runners import YesNo
from latent_interconnect.evaluation.scan_states import (
    SCAN_CONNECTOR,
    ScanOutputs,
    StateWrites,
    WrittenScan,
    require_label_classes,
    require_train,
    scan_states,
    scan_suite,
)
from latent_interconnect.evaluation.scores import read_scores, write_numbers, write_scores
from latent_interconnect.evaluation.suites import SuiteContext
from latent_interconnect.receivers.base import write_layers
from latent_interconnect.stats.bootstrap import Interval, summarize
from latent_interconnect.writes.connector import AlphaSpec, ConnectorSpec, initial_alphas
from latent_interconnect.writes.layers import Entry, ExplicitLayers

__all__ = [
    "LayerSet",
    "LayerSetComparisonSpec",
    "auroc_weighted_alphas",
    "check_sets",
    "clipped_auroc_weighted_alphas",
    "compare_layer_sets",
    "connector_start",
    "set_sources",
]

_NO_SIGNAL = 0.5
"""The AUROC of a score that does not order the two classes: the zero of :func:`auroc_weighted_alphas`."""


class LayerSet(Config):
    """One candidate write configuration.

    Attributes:
        name: the set's label (unique within its comparison).
        per_token: per slot token, the receiver layers it writes, increasing as ``ExplicitLayers`` lists them (one
            set, one identity; at layer 0 too, the object's state, not a constant).
        alpha: the α schedule its entries start at (``writes.connector.initial_alphas``); a ``per_layer`` start at a
            layer the set does not write is refused when the set is read, never ignored.
        layer0: ``constant``: each token also writes the connector's layer-0 constant, the ridge's layer-0 intercept
            ``b`` (the mean layer-0 target state), at α = 1, as a connector with ``layer0: constant`` does at step 0;
            ``per_token`` then lists no layer 0 (ridge source only). None: layer 0 as ``per_token`` says (added field).
        bias: ``zero``: the entries above layer 0 write ``W z``, the ridge prediction without its intercept (the
            layer-0 constant is unchanged); None: the full prediction ``W z + b``, ``b`` from the target state
            (ridge source only; added field).
    """

    name: str
    per_token: tuple[tuple[int, ...], ...] = Field(min_length=1)
    alpha: AlphaSpec
    layer0: Literal["constant"] | None = added_field(default=None)
    bias: Literal["zero"] | None = added_field(default=None)

    @model_validator(mode="after")
    def _check_layers(self) -> Self:
        for t, layers in enumerate(self.per_token):
            if not layers or list(layers) != sorted(set(layers)):
                raise ValueError(
                    f"set {self.name!r}: token {t} writes {layers}: at least one layer, none repeated, in increasing "
                    "order"
                )
        if self.layer0 == "constant" and any(0 in layers for layers in self.per_token):
            raise ValueError(f"set {self.name!r}: the layer-0 constant is written, so per_token lists no layer 0")
        try:  # the schedule's own rule, when the set is read: a per_layer start no entry takes would fail in the job
            initial_alphas(self.entries(), self.alpha)
        except ValueError as e:
            raise ValueError(f"set {self.name!r}: {e}") from e
        return self

    def entries(self) -> tuple[Entry, ...]:
        """Return the set's (token, layer) entries, sorted."""
        return tuple(sorted(Entry(t, L) for t, layers in enumerate(self.per_token) for L in layers))


class LayerSetComparisonSpec(WrittenScan):
    """Score several candidate write sets at initialization on the same items.

    The shared fields are :class:`~latent_interconnect.evaluation.scan_states.WrittenScan`'s.

    Attributes:
        sets: the candidates (at least two; :func:`check_sets`: names unique, every layer one the receiver writes).
            The source is ``oracle`` or ``ridge``. A study's ridge is its held-out scan ridge, which never saw the
            items (the ``layer_selection`` analysis checks it is the study's). A comparison outside a study may
            name another ridge, e.g. a run's production ridge, whose fit holds the items (allowed): its scores are then
            in-sample, which the config that names it states; the spec does not check which ridge it is given.
        contrasts: the pairs of set names compared (a − b), each a set's name, no pair twice and no set with
            itself; empty compares every pair, in the sets' order.
    """

    sets: tuple[LayerSet, ...] = Field(min_length=2)
    contrasts: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def _check_sets(self) -> Self:
        check_sets(self.sets, self.contrasts, source=self.source, depth=self.receiver.depth)
        return self

    def pairs(self) -> tuple[tuple[str, str], ...]:
        """Return the compared pairs: the declared ones, or every pair in the sets' order."""
        return self.contrasts or tuple(combinations([s.name for s in self.sets], 2))


def check_sets(sets: Sequence[LayerSet], contrasts: Sequence[tuple[str, str]], *, source: str, depth: int) -> None:
    """Refuse candidates that are not one comparison's on a receiver: the one statement of a comparison's rules.

    :class:`LayerSetComparisonSpec` applies it when a config is read; whatever writes a comparison's candidates
    (e.g. a port's generator) applies it before writing them. Each set's own rules are
    :class:`LayerSet`'s.

    Args:
        sets: the candidates.
        contrasts: the compared pairs of set names.
        source: the comparison's source (``WrittenScan.source``).
        depth: the receiver's decoder-layer count (``ReceiverSpec.depth``).

    Raises:
        ValueError: naming the repeated set names; the sets that write a ridge's intercept or leave it out
            (``layer0``, ``bias``) under a source other than ``ridge``; the set and its layers outside the receiver's
            writable layers (``receivers.base.write_layers``, the one write-layer rule); the contrast names no set
            has; the pairs listed twice (one key twice in the numbers table, which ``scores.write_numbers`` refuses
            only after every set is scored); or the sets compared with themselves.
    """
    names = [s.name for s in sets]
    if len(set(names)) != len(names):
        raise ValueError(f"set names repeat: {names}")
    ridge_only = [s.name for s in sets if s.layer0 is not None or s.bias is not None]
    if source != "ridge" and ridge_only:
        raise ValueError(f"sets {ridge_only} write a ridge's intercept or leave it out: the source must be 'ridge'")
    for s in sets:
        try:
            write_layers({e.layer for e in s.entries()}, depth)
        except ValueError as e:
            raise ValueError(f"set {s.name!r}: {e}") from e
    unknown = sorted({n for pair in contrasts for n in pair} - set(names))
    if unknown:
        raise ValueError(f"contrasts name sets {unknown} the comparison does not have")
    repeated = sorted({pair for pair in contrasts if contrasts.count(pair) > 1})
    if repeated:
        raise ValueError(f"contrasts repeat the pairs {repeated}: each is one row of the numbers table")
    selves = sorted({a for a, b in contrasts if a == b})
    if selves:
        raise ValueError(f"contrasts compare the sets {selves} with themselves")


def auroc_weighted_alphas(means: Mapping[int, float], *, top: float = 0.99) -> dict[int, float]:
    """Return the AUROC-weighted α start of a connector's write layers: data-selected, exploratory.

    ``α_l = top · max(m_l − 0.5, 0) / max_j max(m_j − 0.5, 0)``: each layer's start is proportional to how far its
    single-layer mean AUROC ``m_l`` lies above 0.5, the best layer at ``top``. A layer at or below 0.5 gets no weight
    and is left out of the result: a set cannot write at α = 0 (``AlphaSpec.per_layer`` is a sigmoid start in (0, 1)),
    so the candidate omits that layer and is then a diagnostic, not a trainable start (:func:`connector_start`). A
    layer above 0.5 keeps its weight however small; no interval is consulted. The layer-0 constant is not a weighted
    layer: it is not in ``means`` and stays at α = 1 (``layer0: constant``).

    The weights are chosen from scores of the same items and tasks the candidate is then scored on, so a comparison
    that uses them is selected on its own data and is not independent evidence.

    Args:
        means: write layer (above 0) → its mean AUROC over the comparison's tasks, written alone.
        top: the best layer's α, in (0, 1).

    Raises:
        ValueError: if ``means`` is empty, names layer 0 or holds a value that is not a finite AUROC in [0, 1] (the
            evidence is missing or wrong: nothing is imputed), or if no layer is above 0.5 (the candidate is
            unavailable: there is nothing to normalize by).
    """
    if not 0.0 < top < 1.0:
        raise ValueError(f"top {top}: a learned sigmoid start in (0, 1)")
    bad = {L: m for L, m in means.items() if L < 1 or not (np.isfinite(m) and 0.0 <= m <= 1.0)}
    if not means or bad:
        raise ValueError(f"single-layer means {dict(means)}: one finite AUROC in [0, 1] per write layer above 0")
    above = {L: m - _NO_SIGNAL for L, m in means.items() if m > _NO_SIGNAL}
    if not above:
        raise ValueError(f"no layer is above 0.5 ({dict(means)}): the AUROC-weighted candidate is unavailable")
    best = max(above.values())
    return {L: top * d / best for L, d in sorted(above.items())}


def clipped_auroc_weighted_alphas(
    means: Mapping[int, float], *, floor: float = 0.1, ceiling: float = 0.9, top: float = 0.99
) -> dict[int, float]:
    """Return the clipped AUROC-weighted α start of a connector's write layers: data-selected, exploratory.

    The literal clip of :func:`auroc_weighted_alphas`: ``α_l = min(max(a_l, floor), ceiling)``, ``a_l`` that
    function's start at full precision (callers round the result, never ``a_l`` before the clip) and 0 for a layer it
    leaves out. Nothing is rescaled: the best layer starts at ``ceiling``, a layer at or below 0.5 at ``floor``, and a
    start already between the two is unchanged. Every layer of ``means`` is therefore written, so the candidate omits
    no write of its connector (whether it is a trainable start is :func:`connector_start`'s to say). The layer-0
    constant is not a weighted layer: it is not in ``means`` and stays at α = 1 (``layer0: constant``).

    As the unclipped weights, these are chosen from scores of the same items and tasks the candidate is then scored
    on: a second data-selected candidate, not independent evidence and not a correction of the first.

    Args:
        means: write layer (above 0) → its mean AUROC over the comparison's tasks, written alone.
        floor: the smallest start, in (0, ``ceiling``).
        ceiling: the largest start, in (``floor``, 1).
        top: the unclipped best layer's α (:func:`auroc_weighted_alphas`).

    Raises:
        ValueError: if not ``0 < floor < ceiling < 1``, or as :func:`auroc_weighted_alphas` (missing or impossible
            evidence, or no layer above 0.5: the unclipped candidate is unavailable, and so is its clip).
    """
    if not 0.0 < floor < ceiling < 1.0:
        raise ValueError(f"floor {floor}, ceiling {ceiling}: sigmoid starts with 0 < floor < ceiling < 1")
    unclipped = auroc_weighted_alphas(means, top=top)
    return {L: min(max(unclipped.get(L, 0.0), floor), ceiling) for L in sorted(means)}


def compare_layer_sets(spec: LayerSetComparisonSpec, ctx: SuiteContext) -> ScanOutputs:
    """Score each set; commit per-set AUROC with intervals, the paired contrasts and the per-object scores.

    The ``scores`` artifact holds every set's per-object scores with the scan columns
    ``layer_set`` and ``source`` (``evaluation.scores.SET_SCAN_COLUMNS``). The ``numbers`` artifact has one row per
    (set, task) (key ``<set>/<task>``), per set mean (``<set>/mean``) and per contrast (``<a> - <b>``): ``kind``
    (``task``, ``mean``, ``contrast``), ``layer_set``, ``other`` (a contrast's b), ``task``, ``value``, ``lo``,
    ``hi`` and ``dropped`` (``stats.bootstrap.summarize``).

    Raises:
        ValueError: if the item set has more than one role, an object outside ``train``, or a task has one
            label class among its in-scope objects (checked before anything is scored).
    """
    items = ctx.items(spec.item_set)
    if len(items.roles) != 1:
        raise ValueError(f"a layer-set comparison scores one role, the item set has {sorted(items.roles)}")
    domain = ctx.domain(spec.domain)
    require_train(items, domain, "layer-set comparison")
    role = next(iter(items.roles))
    objects = items.objects[role]
    require_label_classes(domain, objects, spec.tasks, "layer-set comparison")
    suite = scan_suite(spec, role, f"layer_sets/{spec.source}")
    columns = unit_columns(items, spec.units.factors)
    tables: dict[str, pd.DataFrame] = {}
    for name, source in set_sources(spec, ctx, objects).items():
        tables[name] = read_scores(ctx.artifacts, YesNo(suite).run(ctx, {SCAN_CONNECTOR: source}, 0))
    # a layer-selection tool, not a paper cell: no thin-cell rule; require_label_classes checked the population
    draws = cell_draws(tables, spec.tasks, "auroc", columns, min_class=0)
    rows = []
    for name, cell in draws.items():
        for t, (estimate, task_draws) in cell.tasks.items():
            rows.append(_row(f"{name}/{t}", "task", name, summarize(task_draws, estimate), task=t))
        rows.append(_row(f"{name}/mean", "mean", name, summarize(cell.mean[1], cell.mean[0])))
    for a, b in spec.pairs():
        interval = summarize(draws[a].mean[1] - draws[b].mean[1], draws[a].mean[0] - draws[b].mean[0])
        rows.append(_row(f"{a} - {b}", "contrast", a, interval, other=b))
    scored = [t.assign(layer_set=name, source=spec.source) for name, t in tables.items()]
    scores = write_scores(ctx.artifacts, pd.concat(scored, ignore_index=True), {"scan": spec.digest()})
    numbers = write_numbers(ctx.artifacts, pd.DataFrame(rows), {"scan": spec.digest(), "scores": scores.sha256})
    return ScanOutputs(numbers=numbers, scores=scores)


def set_sources(spec: LayerSetComparisonSpec, ctx: SuiteContext, objects: Sequence[str]) -> dict[str, StateWrites]:
    """Return each set's write source for the objects: its entries' states at its step-0 α.

    An entry writes its layer's state (:func:`~latent_interconnect.evaluation.scan_states.scan_states`; ``W z`` alone
    for a ``bias: zero`` set) at ``initial_alphas``; a ``layer0: constant`` set adds, per token, the ridge's layer-0
    intercept at α = 1. Nothing is scored.

    Raises:
        ValueError: as ``scan_states``, or if a set writes the layer-0 constant and the ridge has no layer 0.
    """
    predict = {True: scan_states(spec, ctx, objects)}
    if any(s.bias == "zero" for s in spec.sets):
        predict[False] = scan_states(spec, ctx, objects, bias=False)
    cache: dict[tuple[int, bool], np.ndarray] = {}

    def state(layer: int, bias: bool) -> np.ndarray:
        if (layer, bias) not in cache:
            cache[(layer, bias)] = predict[bias](layer)
        return cache[(layer, bias)]

    def constant() -> np.ndarray:
        layers = load_ridge(ctx.artifacts, ctx.artifacts.ref(spec.states)).layers
        if 0 not in layers:
            raise ValueError(
                f"the layer-0 constant is the ridge's layer-0 intercept; ridge {spec.states[:12]} has none"
            )
        return np.tile(np.asarray(layers[0].b, dtype=np.float32), (len(objects), 1))

    b0 = constant() if any(s.layer0 == "constant" for s in spec.sets) else None
    out = {}
    for s in spec.sets:
        entries = s.entries()
        alphas = initial_alphas(entries, s.alpha)
        states = {L: state(L, s.bias is None) for L in sorted({e.layer for e in entries})}
        if b0 is not None and s.layer0 == "constant":
            alphas |= {Entry(t, 0): 1.0 for t in range(len(s.per_token))}
            states[0] = b0
        out[s.name] = StateWrites(objects, states, alphas)
    return out


def connector_start(layer_set: LayerSet, connector: ConnectorSpec) -> ConnectorSpec:
    """Return the connector config whose step-0 writes are the set's: the connector with the set's α start.

    The set's α start replaces the connector's whole start: ``first`` (None: a start never fixes an α), ``init`` and
    ``per_layer``; the connector's ``learned``, ``budget`` and ``penalty_k`` stay. So the ``init`` stage
    (``build.init_state``) and the trainer start every α as sigmoid(θ) from exactly the values the set was scored at
    (``initial_alphas``, one function for both). With the same ridge as the scan's, an affine connector's step-0
    writes are the set's (``W z + b`` per entry, the layer-0 intercept at α = 1).

    Raises:
        ValueError: for a set that is a diagnostic, not a trainable start: its layers are not the connector's (an
            omitted entry, α = 0, is a layer choice), it fixes an α (``alpha.first``, e.g. α = 1), it leaves
            out the intercept (``bias: zero``: no initialization writes it) or its layer 0 is not the connector's
            constant; or for a connector whose step-0 writes are not ridge predictions (a kind other than
            ``affine``, ``init`` not ``ridge``, a layer-0 kind other than ``constant``, init noise on a token).
    """
    c = connector
    if c.init != "ridge" or c.kind != "affine" or c.layer0 != "constant" or c.alpha is None:
        raise ValueError(
            f"connector {c.name!r}: only a ridge-initialized affine connector with a layer-0 constant and an alpha "
            "spec writes the scan's states at step 0"
        )
    noisy = [t for t, token in enumerate(c.ready_tokens()) if token.noise]
    if noisy:
        raise ValueError(
            f"connector {c.name!r}: tokens {noisy} add init noise, so its step-0 writes are not the scan's"
        )
    if not isinstance(c.layers, ExplicitLayers) or tuple(c.layers.per_token) != layer_set.per_token:
        raise ValueError(
            f"set {layer_set.name!r} writes {layer_set.per_token}, connector {c.name!r} {c.layers}: a diagnostic "
            "(an omitted entry is a layer choice), not a start"
        )
    if layer_set.layer0 != "constant":
        raise ValueError(f"set {layer_set.name!r} does not write the layer-0 constant the connector writes")
    if layer_set.bias is not None:
        raise ValueError(f"set {layer_set.name!r} leaves out the ridge intercept: a diagnostic, no init path writes it")
    if layer_set.alpha.first is not None:
        raise ValueError(f"set {layer_set.name!r} fixes alpha.first: a diagnostic, not a learned sigmoid start")
    start = {"first": None, "init": layer_set.alpha.init, "per_layer": dict(layer_set.alpha.per_layer)}
    alpha = c.alpha.model_copy(update=start)
    return c.model_copy(update={"alpha": AlphaSpec.model_validate(alpha.model_dump())})


def _row(
    key: str,
    kind: str,
    layer_set: str,
    interval: Interval,
    *,
    task: str = "",
    other: str = "",
) -> dict[str, object]:
    """Return one numbers row (string columns empty where they do not apply)."""
    return {
        "key": key,
        "kind": kind,
        "layer_set": layer_set,
        "other": other,
        "task": task,
        "value": interval.estimate,
        "lo": interval.lo,
        "hi": interval.hi,
        "dropped": interval.dropped,
    }
