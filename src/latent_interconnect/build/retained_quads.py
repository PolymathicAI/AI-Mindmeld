"""A retained subset of a swap-quad item set: whole quads kept as they are, the others removed with a reason.

Correcting an evaluated cross-domain set removes units and never redraws (the no-redraw rule of the single-domain sets,
``build.item_selection`` ``size: all``; here for quads). :func:`retain_quads` takes a built set (``within``), the
quads declared removed, each with its one reason, and the declared replacements of objects, and returns the other
quads untouched: the same items in the same order with the same quad keys and fields. Nothing is paired again, so a
quad whose member leaves is removed whole, and the caller's declaration must say so:

- **Removal is declared, not inferred.** A kept quad that holds an object outside ``members`` (the objects still
  eligible for its role: e.g. the corrected training set) is refused, with the quads named: the declaration removes
  it, or the build stops. A declared quad the set does not hold is refused too (a list written for another set).
- **A replacement touches only the item it names.** A replacement is keyed by quad, role and stored object, and
  changes that object in that quad's items (the positive and the swap that carry it), nowhere else. One naming a
  removed quad, or an object its quad does not hold at that role, is refused.

**Other whole units.** The Rhea ranking and direction sets (``build.cross_domain_items``) group their items by
another field (``anchor``): ``field`` names it, and a unit of that field is kept or removed as a quad is.

The structure of what is returned (two positives of four different objects and exactly their two swaps) is the suite
kind's own check, ``evaluation.cross_domain.swap_quads``, which lies above this layer: the stage passes it to the
builder that calls this (``build.bindingdb_items``).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from latent_interconnect.data.items import ItemSet

__all__ = ["QUAD", "RetainedItems", "retain_quads"]

QUAD = "quad"
"""The item-set field naming each item's quad (the cross-domain item-set contract)."""
_SHOWN = 5


@dataclass(frozen=True)
class RetainedItems:
    """The retained subset of one set.

    Attributes:
        items: the kept quads' items, in the set's order, with the declared replacements applied.
        removed: removed quad → its reasons (one declared reason each; a list, as ``build.item_selection`` stores
            ``removed``).
        replaced: kept quad → role → stored object → the object the items now hold.
    """

    items: ItemSet
    removed: dict[str, list[str]]
    replaced: dict[str, dict[str, dict[str, str]]]


def retain_quads(
    within: ItemSet,
    removed: Mapping[str, str],
    replacements: Mapping[tuple[str, str, str], str],
    *,
    name: str,
    members: Mapping[str, Collection[str]],
    field: str = QUAD,
) -> RetainedItems:
    """Return the quads of ``within`` that ``removed`` does not name, with ``replacements`` applied (module docstring).

    Args:
        within: the built set (field :data:`QUAD`).
        removed: quad → the reason it leaves.
        replacements: (quad, role, stored object) → the object that replaces it in that quad.
        name: the returned set's name.
        members: role → the objects a kept item may hold at that role, after the replacements.
        field: the item-set field naming each item's unit (:data:`QUAD`; ``anchor`` for a ranking or direction set).

    Raises:
        KeyError: if the set has no ``field`` field.
        ValueError: if a removed or replaced quad is not in the set, a replacement names a removed quad or an object
            its quad does not hold, or a kept quad holds an object outside ``members``.
    """
    if field not in within.fields:
        raise KeyError(f"item set {within.name!r} has no {field!r} field (the cross-domain item-set contract)")
    quads = [str(q) for q in within.fields[field]]
    known = set(quads)
    unknown = sorted(set(removed) - known)
    if unknown:
        raise ValueError(f"{name}: {len(unknown)} removed {field}s are not in {within.name!r}, e.g. {unknown[:_SHOWN]}")
    stray = sorted(k for k in replacements if k[0] not in known or k[0] in removed or k[1] not in within.roles)
    if stray:
        raise ValueError(f"{name}: replacements name a removed or unknown {field} or role, e.g. {stray[:_SHOWN]}")
    keep = [i for i, q in enumerate(quads) if q not in removed]
    objects = {role: [within.objects[role][i] for i in keep] for role in within.roles}
    replaced: dict[str, dict[str, dict[str, str]]] = {}
    for (quad, role, old), new in sorted(replacements.items()):
        rows = [n for n, i in enumerate(keep) if quads[i] == quad and within.objects[role][i] == old]
        if not rows:
            raise ValueError(f"{name}: {field} {quad!r} holds no {role} {old!r} to replace")
        for n in rows:
            objects[role][n] = new
        replaced.setdefault(quad, {}).setdefault(role, {})[old] = new
    for role, allowed in members.items():
        outside = sorted({(quads[i], o) for i, o in zip(keep, objects[role], strict=True) if o not in allowed})
        if outside:
            raise ValueError(
                f"{name}: {len({q for q, _ in outside})} kept {field}s hold a {role} that is no longer eligible, e.g. "
                f"{outside[:_SHOWN]}: the declaration must remove them (a {field} leaves whole; nothing is refilled)"
            )
    items = ItemSet(
        name,
        within.roles,
        {role: tuple(v) for role, v in objects.items()},
        {f: tuple(column[i] for i in keep) for f, column in within.fields.items()},
    )
    return RetainedItems(items, {q: [removed[q]] for q in sorted(removed)}, replaced)
