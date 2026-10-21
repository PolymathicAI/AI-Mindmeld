"""Write sources: where a backend gets a connector's writes for the objects in a batch.

A connector's writes depend only on the object, never on the receiver state, so they can come from two places:

- :class:`LiveSource`: run the connector on inputs from a loader (training, and in-process scoring);
- :class:`BankSource`: rows of a precomputed *write bank* artifact, selected by object id. A backend running the
  receiver in other processes (vLLM) passes the bank by reference (``bank``) and its workers select rows themselves,
  so per-request tensors never cross a process boundary.

Backends take ``Mapping[connector name, WriteSource]`` and resolve per-layer writes with
:func:`latent_interconnect.receivers.kernel.resolve`. A bank is built from a checkpoint's connector by
:func:`build_bank`.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.canonical import canonical_json, canonical_sha256
from latent_interconnect.writes.connector import Connector, SlotWrites
from latent_interconnect.writes.inputs import ObjectBatch
from latent_interconnect.writes.layers import Entry

__all__ = ["BANK_KIND", "BankSource", "LiveSource", "WriteSource", "build_bank"]


class WriteSource(Protocol):
    """Gives a connector's writes for a list of objects."""

    @property
    def bank(self) -> ArtifactRef | None:
        """The bank artifact behind this source, if any (lets out-of-process backends read it directly)."""
        ...

    @property
    def entries(self) -> tuple[Entry, ...]:
        """The (token, layer) entries this source writes, the same for every object, known without computing writes.

        A backend checks a request's write layers from them before any write is computed
        (``receivers.base.checked_write_layers``).
        """
        ...

    def writes_for(self, object_ids: Sequence[str]) -> SlotWrites:
        """Return the writes of these objects, in this order."""
        ...


class LiveSource:
    """Writes computed by a live connector from a loader (gradients flow when the caller enables them).

    Args:
        connector: the connector module.
        load: object ids → their encoder inputs (``encoders.loader.InputLoader.batch``).
    """

    def __init__(self, connector: Connector, load: Callable[[Sequence[str]], ObjectBatch]) -> None:
        self.connector = connector
        self.load = load

    @property
    def bank(self) -> ArtifactRef | None:
        """A live source has no bank."""
        return None

    @property
    def entries(self) -> tuple[Entry, ...]:
        """The connector's entries."""
        return self.connector.entries

    def writes_for(self, object_ids: Sequence[str]) -> SlotWrites:
        """Run the connector on the objects' inputs."""
        writes: SlotWrites = self.connector(self.load(object_ids))
        return writes


BANK_KIND = "write_bank"
"""The artifact kind of a write bank (:func:`build_bank`)."""
_BANK_FILE = "bank.json"
_DELTAS_DIR = "deltas"


class BankSource:
    """Writes read from a committed write-bank artifact (inference only; α fixed at the checkpoint's values).

    The bank is opened by reference: its header is read at construction and each entry's ``(N, d_model)`` float32
    array is memory-mapped, so selecting a batch's rows reads those rows only. A backend whose receiver runs in other
    processes passes :attr:`bank` (with :attr:`store`) and its workers build their own ``BankSource`` from it.

    Args:
        store: the artifact store holding the bank.
        bank: the bank artifact (:data:`BANK_KIND`).

    Raises:
        ValueError: if ``bank`` is not a write bank.
        IntegrityError: if its manifest does not match the reference.
    """

    def __init__(self, store: ArtifactStore, bank: ArtifactRef) -> None:
        if bank.kind != BANK_KIND:
            raise ValueError(f"{bank.short} is a {bank.kind!r} artifact, not a {BANK_KIND!r}")
        store.manifest(bank)
        self.store = store
        self._bank = bank
        root = store.path(bank)
        header = json.loads((root / _BANK_FILE).read_text())
        self.object_ids: tuple[str, ...] = tuple(header["object_ids"])
        self._row = {oid: i for i, oid in enumerate(self.object_ids)}
        self._entries = tuple(Entry(int(t), int(layer)) for t, layer in header["entries"])
        self._alphas = {e: float(header["alphas"][e.key()]) for e in self._entries}
        self._deltas = {e: np.load(root / _DELTAS_DIR / f"{e.key()}.npy", mmap_mode="r") for e in self._entries}

    @property
    def bank(self) -> ArtifactRef | None:
        """The bank artifact."""
        return self._bank

    @property
    def entries(self) -> tuple[Entry, ...]:
        """The bank's entries (from its header), sorted by (token, layer)."""
        return self._entries

    def writes_for(self, object_ids: Sequence[str]) -> SlotWrites:
        """Select the objects' rows from the bank (float32 on the host), in the given order.

        Raises:
            KeyError: naming the first object the bank lacks.
        """
        missing = [oid for oid in object_ids if oid not in self._row]
        if missing:
            raise KeyError(f"objects {missing[:5]} are not in write bank {self._bank.short}")
        rows = np.array([self._row[oid] for oid in object_ids], dtype=np.int64)
        deltas = {e: torch.from_numpy(np.ascontiguousarray(d[rows])) for e, d in self._deltas.items()}
        alphas = {e: torch.tensor(a, dtype=torch.float32) for e, a in self._alphas.items()}
        return SlotWrites(tuple(object_ids), deltas, alphas)


def build_bank(
    connector: Connector, checkpoint: ArtifactRef, batches: Sequence[ObjectBatch], store: ArtifactStore
) -> ArtifactRef:
    """Run the connector over every batch (no grad, float32) and commit a ``write_bank`` artifact.

    The bank holds, per entry, a ``(N, d_model)`` float32 array of δ with one row per object, in the batches' order
    (``deltas/<entry key>.npy``), and ``bank.json``: the object ids in that order, the entries, each entry's α and the
    checkpoint reference it was computed from. Its metadata names the checkpoint, the connector and the digest of the
    object ids. A connector's writes depend only on the object, so a bank's rows are the live connector's writes;
    the bank is deterministic for given batches, so building it twice commits one artifact.

    Args:
        connector: the checkpoint's connector (on the host).
        checkpoint: the checkpoint artifact the connector was loaded from (recorded in the bank).
        batches: the objects' encoder inputs; every object at most once over all batches.
        store: where the bank is committed.

    Raises:
        ValueError: if no batch is given, an object appears twice, or a learned α differs between batches.
    """
    if not batches:
        raise ValueError("a write bank needs at least one batch of objects")
    ids = [oid for b in batches for oid in b.object_ids]
    repeated = sorted(oid for oid, n in Counter(ids).items() if n > 1)
    if repeated:
        raise ValueError(f"objects appear more than once in the bank's batches: {repeated[:5]}")
    parts: dict[Entry, list[Tensor]] = {e: [] for e in connector.entries}
    alphas: dict[Entry, float] = {}
    with torch.no_grad():
        for b in batches:
            writes = connector(b)
            for e in connector.entries:
                parts[e].append(writes.deltas[e].detach().to("cpu", torch.float32))
                a = float(writes.alphas[e])
                if alphas.setdefault(e, a) != a:
                    raise ValueError(f"entry {e.key()}: α changed between batches ({alphas[e]} vs {a})")
    staged = store.staging(f"{BANK_KIND}-{uuid.uuid4().hex}")
    (staged / _DELTAS_DIR).mkdir()
    for e, rows in parts.items():
        np.save(staged / _DELTAS_DIR / f"{e.key()}.npy", torch.cat(rows).numpy())
    header = {
        "object_ids": ids,
        "entries": [[e.token, e.layer] for e in connector.entries],
        "alphas": {e.key(): a for e, a in alphas.items()},
        "checkpoint": checkpoint.sha256,
        "connector": connector.spec.name,
        "d_model": connector.d_model,
    }
    (staged / _BANK_FILE).write_text(canonical_json(header))
    meta = {"checkpoint": checkpoint.sha256, "connector": connector.spec.name, "objects": canonical_sha256(ids)}
    return store.commit(staged, BANK_KIND, meta)
