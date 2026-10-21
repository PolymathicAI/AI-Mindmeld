"""The B4 role split of ChEBI roles and the trained mechanism roles of a molecule.

The port of the original ``role_depths``, ``role_ancestors`` and ``role_split_b4``, and of ``nearest_trained`` and
``b4_roles``, kept exactly. Every term below the role
root (``CHEBI:50906``) gets one assignment, by its name and ``is_a`` ancestry, first match wins:

- ``ec_dropped``: its cleaned name names an EC number;
- ``organisational``: the root, its three branches (depth 1) and ``... role`` categories;
- ``heldout_provenance``: the metabolite subtree (``CHEBI:25212``);
- ``heldout_application``: any ancestor in the application branch (``CHEBI:33232``);
- ``heldout_chemical``: any ancestor in the chemical-role branch (``CHEBI:51086``);
- ``trained_mechanism``: the rest (the biological-role branch, outside the metabolite subtree).

A molecule's trained roles (:meth:`RoleSplit.molecule_roles`) are its direct ``trained_mechanism`` roles, then, for each
direct EC-named role, its nearest ``trained_mechanism`` ``is_a`` ancestors (breadth-first, every one at the first level
that has one), a mapped role equal to or an ancestor of another listed role dropped. Its held-out roles are its direct
roles in the three held-out groups.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

__all__ = ["HELD_OUT_GROUPS", "TRAINED", "MoleculeRoles", "RoleSplit", "ancestors"]

ROOT = "CHEBI:50906"
METABOLITE = "CHEBI:25212"
_BRANCHES = {"CHEBI:24432": "bio", "CHEBI:33232": "app", "CHEBI:51086": "chem"}
_EC = re.compile(r"\bEC \d")
_EC_DROPPED = "ec_dropped"
TRAINED = "trained_mechanism"
"""The assignment of the trained roles."""
HELD_OUT_GROUPS = ("heldout_provenance", "heldout_application", "heldout_chemical")
"""The assignments of the held-out roles."""


def ancestors(is_a: Mapping[str, tuple[str, ...]]) -> dict[str, frozenset[str]]:
    """Return every term's transitive ``is_a`` ancestors (memoized; a cycle is cut where it closes)."""
    memo: dict[str, frozenset[str]] = {}

    def of(term: str) -> frozenset[str]:
        if term not in memo:
            memo[term] = frozenset()  # the cycle guard
            found: set[str] = set()
            for parent in is_a.get(term, ()):
                found |= {parent} | of(parent)
            memo[term] = frozenset(found)
        return memo[term]

    return {t: of(t) for t in is_a}


def _depths(is_a: Mapping[str, tuple[str, ...]]) -> dict[str, int]:
    """Shortest ``is_a`` depth below the role root of every term reachable from it."""
    children: dict[str, list[str]] = {}
    for term, parents in is_a.items():
        for parent in parents:
            children.setdefault(parent, []).append(term)
    depth, frontier = {ROOT: 0}, [ROOT]
    while frontier:
        following = []
        for parent in frontier:
            for child in children.get(parent, []):
                if child not in depth:
                    depth[child] = depth[parent] + 1
                    following.append(child)
        frontier = following
    return depth


@dataclass(frozen=True)
class MoleculeRoles:
    """One molecule's roles under the split.

    Attributes:
        trained: the trained role ids (direct trained roles, then the mapped ones).
        held_out: the direct roles in a held-out group.
    """

    trained: tuple[str, ...]
    held_out: tuple[str, ...]


class RoleSplit:
    """The B4 assignment of every role term, and the role lists of a molecule.

    Args:
        is_a: every term's ``is_a`` parents.
        name: a term's cleaned name.
        ancestry: every term's transitive ancestors (:func:`ancestors` of ``is_a``).
    """

    def __init__(
        self,
        is_a: Mapping[str, tuple[str, ...]],
        name: Callable[[str], str],
        ancestry: Mapping[str, frozenset[str]],
    ) -> None:
        self.is_a, self.ancestry = is_a, ancestry
        self.assignment = {term: _assign(term, depth, name(term), ancestry) for term, depth in _depths(is_a).items()}

    def nearest_trained(self, role: str) -> list[str]:
        """Return the nearest ``trained_mechanism`` ``is_a`` ancestors of ``role`` (sorted), or ``[]``."""
        seen, front = {role}, [role]
        while front:
            following = []
            for term in front:
                for parent in self.is_a.get(term, ()):
                    if parent not in seen:
                        seen.add(parent)
                        following.append(parent)
            hit = sorted(p for p in following if self.assignment.get(p) == TRAINED)
            if hit:
                return hit
            front = following
        return []

    def molecule_roles(self, direct: tuple[str, ...]) -> MoleculeRoles:
        """Return a molecule's trained and held-out roles from its direct ``has_role`` targets (module docstring)."""
        trained = list(dict.fromkeys(r for r in direct if self.assignment.get(r) == TRAINED))
        mapped_by_ec = [self.nearest_trained(r) for r in direct if self.assignment.get(r) == _EC_DROPPED]
        mapped = list(dict.fromkeys(m for ms in mapped_by_ec for m in ms if m not in trained))
        listed = trained + mapped
        mapped = [m for m in mapped if not any(m in self.ancestry.get(o, ()) for o in listed if o != m)]
        held_out = tuple(r for r in direct if self.assignment.get(r) in HELD_OUT_GROUPS)
        return MoleculeRoles(tuple(trained + mapped), held_out)


def _assign(term: str, depth: int, name: str, ancestry: Mapping[str, frozenset[str]]) -> str:
    above = ancestry.get(term, frozenset())
    branches = {label for root, label in _BRANCHES.items() if root == term or root in above}
    if _EC.search(name):
        return _EC_DROPPED
    if depth <= 1 or name.endswith(" role"):
        return "organisational"
    if term == METABOLITE or METABOLITE in above:
        return "heldout_provenance"
    if "app" in branches:
        return "heldout_application"
    if "chem" in branches:
        return "heldout_chemical"
    return TRAINED
