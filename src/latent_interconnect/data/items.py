"""Item sets: the fixed, content-addressed tables that suites score.

An item is one row: an object id per *role* (``object`` for single-object suites; ``protein`` and ``molecule`` for
cross-domain items; ``a`` and ``b`` for composition), plus optional fields (e.g. a pair label). There is no shuffled
control: eval prompts are vetted instead (``evaluation.vetting``).

The one exception is likelihood metrics such as caption NLL, where a mismatched-object baseline is meaningful. Their
suites derive the mismatch from :func:`derangement` with a seed declared in the suite and record the object actually
written in every score row.

Every committed item set carries each role's cluster as the field ``cluster.<role>`` (:func:`cluster_field`), taken
from the role's domain when the set is built (:func:`with_clusters`), so analyses resample by cluster from the item
set alone (``ResamplingUnits.factors`` name these columns). A set whose analysis compares objects within a name
(Fig 10b's ``stratified_contrast``) also carries the role's name input as ``name.<role>`` (:func:`with_names`).

The **object list** is the artifact that ``CaptureSpec.objects``, ``RidgeSpec.fit_set`` and an extraction's objects
cite: an item set with the one role :data:`OBJECT_ROLE` (built by :func:`object_list`).
Its order is part of its content, since fold assignment and part layout follow it.

Artifact format (kind :data:`ITEM_SET_KIND`): ``items.json`` holds ``name``, ``roles`` (``[role, domain]`` pairs, in
role order), ``objects`` and ``fields`` (column-wise, in item order); the manifest metadata repeats ``name``, ``roles``
and the item count, plus any extra metadata of its builder. Read and write with :func:`load_item_set` and
:func:`commit_item_set`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.canonical import canonical_json
from latent_interconnect.domains.base import Domain

__all__ = [
    "ITEM_SET_KIND",
    "OBJECT_ROLE",
    "ItemSet",
    "cluster_field",
    "commit_item_set",
    "derangement",
    "load_item_set",
    "object_list",
    "with_clusters",
    "with_names",
]

ITEM_SET_KIND = "item_set"
"""Artifact kind of an item set (and of an object list)."""
OBJECT_ROLE = "object"
"""The role of single-object item sets and object lists."""
_FILE = "items.json"

_MAX_TRIES = 10_000
_MIN_DERANGEMENT = 2  # a single element has no derangement


def derangement(n: int, seed: int) -> np.ndarray:
    """Return a uniformly random permutation of ``range(n)`` with no fixed point.

    Rejection sampling: a uniform permutation is a derangement with probability → 1/e, so the expected number of
    draws is about 2.7 and the result is exactly uniform over derangements.

    Raises:
        ValueError: if ``n < 2`` (no derangement exists).
    """
    if n < _MIN_DERANGEMENT:
        raise ValueError(f"no derangement of {n} element(s)")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    for _ in range(_MAX_TRIES):
        p = rng.permutation(n)
        if not np.any(p == idx):
            return p
    raise RuntimeError("derangement sampling did not terminate")  # probability ~ (1 - 1/e)^10000


@dataclass(frozen=True)
class ItemSet:
    """A table of items.

    Attributes:
        name: e.g. ``"protein/union"``.
        roles: role name → domain name, in a fixed order.
        objects: role → object id per item (all the same length).
        fields: extra per-item columns (e.g. ``pair_label``, ``stratum``).
    """

    name: str
    roles: Mapping[str, str]
    objects: Mapping[str, tuple[str, ...]]
    fields: Mapping[str, tuple[object, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = {len(v) for v in (*self.objects.values(), *self.fields.values())}
        if len(n) != 1:
            raise ValueError(f"item set {self.name!r}: columns have different lengths {sorted(n)}")
        if set(self.objects) != set(self.roles):
            raise ValueError(f"item set {self.name!r}: objects must cover exactly the roles")

    def __len__(self) -> int:
        return len(next(iter(self.objects.values())))

    @property
    def object_ids(self) -> tuple[str, ...]:
        """Return the objects of a single-role set (an object list) in item order.

        Raises:
            ValueError: if the set has more than one role.
        """
        if len(self.roles) != 1:
            raise ValueError(f"item set {self.name!r} has roles {sorted(self.roles)}; an object list has one")
        return next(iter(self.objects.values()))


def cluster_field(role: str) -> str:
    """Return the item field holding ``role``'s cluster ids."""
    return f"cluster.{role}"


def with_clusters(items: ItemSet, domains: Mapping[str, Domain]) -> ItemSet:
    """Return the set with ``cluster.<role>`` fields from each role's domain (existing cluster fields replaced).

    Args:
        items: the item set.
        domains: domain name → domain, covering every role's domain.

    Raises:
        KeyError: if a role's domain is not given.
        ValueError: if an object has no cluster in its domain.
    """
    fields = dict(items.fields)
    for role, domain_name in items.roles.items():
        clusters = domains[domain_name].clusters()
        ids = pd.Index(items.objects[role])
        found = clusters.reindex(ids)
        if found.isna().any():
            raise ValueError(
                f"{items.name}: {int(found.isna().sum())} {role} object(s) without a cluster in "
                f"{domain_name!r}, e.g. {list(ids[found.isna()][:3])}"
            )
        fields[cluster_field(role)] = tuple(str(c) for c in found)
    return ItemSet(items.name, items.roles, items.objects, fields)


def with_names(items: ItemSet, domains: Mapping[str, Domain], role: str) -> ItemSet:
    """Return the set with the field ``name.<role>``: each object's name in its domain (the name condition's text).

    Raises:
        KeyError: if the role or its domain is not given.
        ValueError: if an object has no name (an empty name would group unnamed objects as if they shared one).
    """
    names = domains[items.roles[role]].objects()["name"].reindex(pd.Index(items.objects[role]))
    if names.isna().any() or (names == "").any():
        raise ValueError(
            f"{items.name}: {role} objects without a name, e.g. {list(names.index[names.fillna('') == ''][:3])}"
        )
    return ItemSet(items.name, items.roles, items.objects, {**items.fields, f"name.{role}": tuple(names)})


def object_list(name: str, domain: Domain, object_ids: tuple[str, ...]) -> ItemSet:
    """Return an object list: an item set with the one role :data:`OBJECT_ROLE`, with its clusters.

    Raises:
        ValueError: if an object id repeats (an object list names each object once), or has no cluster.
    """
    if len(set(object_ids)) != len(object_ids):
        dup = pd.Index(object_ids)[pd.Index(object_ids).duplicated()]
        raise ValueError(f"object list {name!r} repeats {len(dup)} id(s), e.g. {list(dup[:3])}")
    return with_clusters(ItemSet(name, {OBJECT_ROLE: domain.name}, {OBJECT_ROLE: object_ids}), {domain.name: domain})


def commit_item_set(store: ArtifactStore, items: ItemSet, meta: Mapping[str, Any] | None = None) -> ArtifactRef:
    """Commit an item set as an :data:`ITEM_SET_KIND` artifact.

    Args:
        store: the artifact store.
        items: the item set; field values must be JSON-compatible.
        meta: extra metadata (e.g. the selection that built it); it must not repeat ``name``, ``roles`` or ``n``.

    Raises:
        ValueError: if ``meta`` repeats a reserved key, or a role has no ``cluster.<role>`` field.
    """
    unclustered = [r for r in items.roles if cluster_field(r) not in items.fields]
    if unclustered:
        raise ValueError(f"item set {items.name!r}: roles {unclustered} have no cluster field (with_clusters)")
    extra = dict(meta or {})
    reserved = {"name", "roles", "n"} & set(extra)
    if reserved:
        raise ValueError(f"meta repeats reserved keys {sorted(reserved)}")
    staged = store.staging(f"items-{uuid.uuid4().hex}")
    body = {
        "name": items.name,
        "roles": [[r, d] for r, d in items.roles.items()],
        "objects": {r: list(v) for r, v in items.objects.items()},
        "fields": {f: list(v) for f, v in items.fields.items()},
    }
    (staged / _FILE).write_text(canonical_json(body))
    roles = [[r, d] for r, d in items.roles.items()]
    return store.commit(staged, ITEM_SET_KIND, {"name": items.name, "roles": roles, "n": len(items), **extra})


def load_item_set(store: ArtifactStore, ref: ArtifactRef) -> ItemSet:
    """Load a committed item set (its manifest is checked against ``ref``).

    Raises:
        ValueError: if ``ref`` is not an item set.
    """
    if ref.kind != ITEM_SET_KIND:
        raise ValueError(f"{ref.short} is a {ref.kind!r} artifact, not an item set")
    store.manifest(ref)
    body = json.loads((store.path(ref) / _FILE).read_text())
    return ItemSet(
        name=body["name"],
        roles=dict(body["roles"]),
        objects={r: tuple(v) for r, v in body["objects"].items()},
        fields={f: tuple(v) for f, v in body["fields"].items()},
    )
