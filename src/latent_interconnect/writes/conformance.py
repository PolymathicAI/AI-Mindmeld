"""Conformance checks for map kinds: what any kind, core or third-party, must satisfy. Run by the tests on every kind.

A kind conforms (:func:`check_kind`) if, for a random ridge and batch:

- ``create`` and ``init_random`` build maps whose ``forward`` returns ``(B, d_model)`` float32;
- a ``state_dict`` round trip through ``empty`` reproduces the outputs exactly;
- every parameter has a learning-rate reference (so no parameter silently gets learning rate 0), and every
  parameter ``decayed()`` names exists.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from latent_interconnect.config import Config
from latent_interconnect.writes.init import EntryInit, InitContext, RidgeLayer
from latent_interconnect.writes.inputs import InputKind
from latent_interconnect.writes.kinds import EntryMap
from latent_interconnect.writes.layers import Entry

__all__ = ["check_kind"]

_BATCH, _RESIDUES = 3, 5


def check_kind(kind: type[EntryMap], params: Config, d_in: int, d_model: int) -> None:
    """Raise ``AssertionError`` describing the first way ``kind`` breaks the map-kind contract."""
    init = _init(Entry(0, 1), d_in, d_model)
    x, mask = _inputs(kind.input_kind, d_in)
    for how, m in (
        ("create", kind.create(params, init, InitContext(d_in=d_in, d_model=d_model))),
        ("init_random", kind.init_random(params, d_in, d_model, torch.Generator().manual_seed(0))),
    ):
        out = m(x, mask)
        _check_delta(f"{kind.__name__}.forward after {how}", out, d_model)
        loaded = kind.empty(params, d_in, d_model)
        loaded.load_state_dict(m.state_dict())
        _check_round_trip(kind, how, torch.equal(loaded(x, mask), out))
        _check_lr(m)


def _init(e: Entry, d_in: int, d_model: int) -> EntryInit:
    rng = np.random.default_rng([e.token, e.layer])
    ridge = RidgeLayer(rng.normal(size=(d_model, d_in)).astype(np.float32), rng.normal(size=d_model).astype(np.float32))
    return EntryInit(e.token, e.layer, ridge, "name", torch.Generator().manual_seed(0))


def _check_delta(where: str, out: Tensor, d_model: int) -> None:
    if out.shape != (_BATCH, d_model) or out.dtype != torch.float32:
        raise AssertionError(f"{where} returned {tuple(out.shape)} {out.dtype}, not ({_BATCH}, {d_model}) float32")


def _check_round_trip(kind: type[EntryMap], how: str, same: bool) -> None:
    if not same:
        raise AssertionError(f"{kind.__name__}: a state_dict round trip through empty() changes the output ({how})")


def _check_lr(m: EntryMap) -> None:
    names = {name for name, _ in m.named_parameters()}
    for name, p in m.named_parameters():
        m.lr_reference(name, p.detach())
    stray = sorted(set(m.decayed()) - names)
    if stray:
        raise AssertionError(f"{type(m).__name__}.decayed() names {stray}, which are not its parameters")


def _inputs(kind: InputKind, d_in: int) -> tuple[Tensor, Tensor | None]:
    g = torch.Generator().manual_seed(1)
    if kind is InputKind.NONE:
        return torch.empty(_BATCH, 0), None
    if kind is InputKind.POOLED:
        return torch.randn(_BATCH, d_in, generator=g), None
    mask = torch.arange(_RESIDUES)[None, :] < torch.tensor([_RESIDUES, 2, 1])[:, None]
    return torch.randn(int(mask.sum()), d_in, generator=g), mask
