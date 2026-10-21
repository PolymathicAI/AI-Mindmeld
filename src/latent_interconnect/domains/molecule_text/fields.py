"""One molecule's rendered text fields: the paper's molecule templates, field by field.

The port of the original template builder's field rendering as run for the paper (the B4 role split, roles first in
``describe``, the class split, the own-name rewrite with its organism guard, unnamed conjugate labels),
with the conjugates answer split into its two lines as the original field table does.
Item names are cleaned ChEBI names (``definitions.clean_text``); an item that is empty or contains ``;`` (the item
separator) is left out of every list, and a list field without items is None. The fields (:data:`FIELDS`):

- ``definition``: the usable definition (``definitions.usable_definition``) sentence by sentence; a sentence that names
  an evaluation protein or an EC number is removed (``domains.eval_proteins``) and the connective opening the next kept
  sentence with it; a kept sentence ends with ``.``, ``!`` or ``?`` (``.`` is added); the own name is rewritten
  (``sentences.own_name_rewrite``). A charged form described by its neutral parent keeps its own line (its own cleaned
  definition, or its name, with ``.``; its name alone if the own line is blocked), then the parent's kept sentences not
  already in the own line under ``Its conjugate acid: `` / ``Its conjugate base: ``, or
  ``Tautomer (<name>): ``;
- ``classes`` (``ChEBI parent class(es): a; b.``), and their partition (``classes.ClassPartition``) into
  ``functional_group_classes`` (``Functional-group class(es)``) and ``parent_family`` (``Parent family`` /
  ``Parent families``);
- ``mechanism_roles``: the trained roles (``roles.RoleSplit``), ``Role: a.`` / ``Roles: a; b.``;
- ``describe``: the roles line, then the definition (roles first), untrimmed; only with a definition;
- ``functional_parent`` (``Functional parent(s)``), ``conjugate_acid`` (``Conjugate acid(s)``), ``conjugate_base``
  (``Conjugate base(s)``);
- ``heldout_roles``: the held-out roles (the role split's held-out groups; ``Role(s)``); ``formula``:
  ``Formula: <f>. Net charge: <c>. Average mass: <m> Da.`` (a polymer's formula alone). Both are held-out text, never
  a training target.

Every list field drops items that name an evaluation protein or an EC number (the item filter), ``heldout_roles``
included; the formula is not filtered.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from latent_interconnect.domains.chebi import ChebiRelease
from latent_interconnect.domains.eval_proteins import EvalProteinMatcher
from latent_interconnect.domains.molecule_text.classes import ClassPartition
from latent_interconnect.domains.molecule_text.definitions import PARENT_RELS, clean_text, name_of, usable_definition
from latent_interconnect.domains.molecule_text.roles import RoleSplit
from latent_interconnect.domains.molecule_text.sentences import (
    own_name_rewrite,
    split_sentences,
    strip_connective,
)

__all__ = ["FIELDS", "HELD_OUT_FIELDS", "MoleculeRenderer"]

FIELDS = (
    "describe",
    "mechanism_roles",
    "classes",
    "functional_parent",
    "conjugate_acid",
    "conjugate_base",
    "definition",
    "functional_group_classes",
    "parent_family",
    "formula",
    "heldout_roles",
)
"""The rendered fields, in the paper's field-table column order."""
HELD_OUT_FIELDS = ("formula", "heldout_roles")
"""The held-out fields: references for held-out evaluation, never training targets."""

_UNNAMED_LABEL = {"is_conjugate_base_of": "Its conjugate acid", "is_conjugate_acid_of": "Its conjugate base"}
_TAUTOMER = "is_tautomer_of"
_SENTENCE_END = (".", "!", "?")
_POLYMER = re.compile(r"\)n\b|\bn$|\)n$")
_LABELS = {
    "classes": ("ChEBI parent class", "ChEBI parent classes"),
    "functional_group_classes": ("Functional-group class", "Functional-group classes"),
    "parent_family": ("Parent family", "Parent families"),
    "mechanism_roles": ("Role", "Roles"),
    "heldout_roles": ("Role", "Roles"),
    "functional_parent": ("Functional parent", "Functional parents"),
    "conjugate_acid": ("Conjugate acid", "Conjugate acids"),
    "conjugate_base": ("Conjugate base", "Conjugate bases"),
}


def _line(field: str, items: list[str]) -> str | None:
    if not items:
        return None
    one, many = _LABELS[field]
    return f"{many if len(items) > 1 else one}: {'; '.join(items)}."


@dataclass(frozen=True)
class _Definition:
    text: str
    own_line: str


class MoleculeRenderer:
    """Renders a molecule's fields from a ChEBI release (the module docstring's rules).

    Args:
        release: the ChEBI release.
        proteins: the evaluation-protein matcher (``domains.eval_proteins``).
        roles: the role split.
        partition: the class partition.

    Attributes:
        counts: what the rules did over every molecule rendered (removed sentences and items, own-name mentions by
            kind, ...), for the build's record.
    """

    def __init__(
        self, release: ChebiRelease, proteins: EvalProteinMatcher, roles: RoleSplit, partition: ClassPartition
    ) -> None:
        self.release, self.proteins, self.roles, self.partition = release, proteins, roles, partition
        self.counts: Counter[str] = Counter()

    def _items(self, field: str, ids: Iterable[str]) -> list[str]:
        """Cleaned names, list-safe, without the blocked ones (counted)."""
        names = [n for n in (name_of(self.release, i) for i in ids) if n and ";" not in n]
        kept = [n for n in names if not self.proteins.blocks(n)]
        self.counts[f"removed_{field}_items"] += len(names) - len(kept)
        return kept

    def render(self, chebi_id: str) -> dict[str, str | None]:
        """Return the molecule's fields (:data:`FIELDS`; None where a field has no text).

        Raises:
            KeyError: if the release has no entry ``chebi_id``.
        """
        entry = self.release.entries[chebi_id]
        out: dict[str, str | None] = dict.fromkeys(FIELDS)
        definition = self._definition(chebi_id)
        out["classes"] = _line("classes", self._items("classes", entry.is_a))
        parts = self.partition.split(entry.is_a)
        out["functional_group_classes"] = _line(
            "functional_group_classes", self._items("functional_group_classes", parts["A"])
        )
        out["parent_family"] = _line("parent_family", self._items("parent_family", parts["B"]))
        roles = self.roles.molecule_roles(entry.has_role)
        out["mechanism_roles"] = _line("mechanism_roles", self._items("mechanism_roles", roles.trained))
        out["heldout_roles"] = _line("heldout_roles", self._items("heldout_roles", roles.held_out))
        if definition is not None:
            out["definition"] = definition.text
            own, _, rest = definition.text.partition("\n")
            head = f"{out['mechanism_roles']} {own}" if out["mechanism_roles"] else own
            out["describe"] = head + ("\n" + rest if rest else "")
        for field, relation in (
            ("functional_parent", "has_functional_parent"),
            ("conjugate_acid", "is_conjugate_base_of"),
            ("conjugate_base", "is_conjugate_acid_of"),
        ):
            out[field] = _line(field, self._items(field, entry.related(relation)))
        if entry.formula:
            out["formula"] = _formula(entry.formula, entry.charge, entry.mass)
        return out

    def _rewrite(self, text: str, name: str) -> str:
        rewritten, counts = own_name_rewrite(text, name)
        self.counts.update({f"own_name_{k}": v for k, v in counts.items()})
        return rewritten

    def _definition(self, chebi_id: str) -> _Definition | None:
        """The definition field, or None (no usable definition, or every sentence removed)."""
        text, source = usable_definition(self.release, chebi_id)
        if not text:
            return None
        name = name_of(self.release, chebi_id)
        keep, removed = [], False
        for sentence in split_sentences(text):
            if self.proteins.blocks(sentence):
                removed = True
                self.counts["removed_definition_sentences"] += 1
                continue
            ended = sentence if sentence.endswith(_SENTENCE_END) else sentence + "."
            keep.append(strip_connective(ended) if removed else ended)
            removed = False
        keep = [self._rewrite(s, name) for s in keep]
        if not keep:
            return None
        if source == "own":
            joined = " ".join(keep)
            return _Definition(joined, joined)
        parent = source.split()[1]
        entry = self.release.entries[chebi_id]
        relation = next(r for r, target in entry.rels if target == parent and r in PARENT_RELS)
        own = clean_text(entry.definition) or f"{name}."
        own = own if own.endswith(".") else own + "."
        if self.proteins.blocks(own):
            self.counts["removed_definition_sentences"] += 1
            own = f"{name}."
        own = self._rewrite(own, name)
        own_sentences = {s.rstrip(".") for s in split_sentences(own)}
        rest = [s for s in keep if s.rstrip(".") not in own_sentences]
        self.counts["parent_sentences_already_in_own_line"] += len(keep) - len(rest)
        if not rest:
            return _Definition(own, own)
        label = _UNNAMED_LABEL[relation] if relation != _TAUTOMER else f"Tautomer ({name_of(self.release, parent)})"
        return _Definition(f"{own}\n{label}: {' '.join(rest)}", own)


def _formula(formula: str, charge: str | None, mass: str | None) -> str:
    """The formula line; a polymer's (per repeat unit) has no charge or mass."""
    polymer = bool(_POLYMER.search(formula))
    parts = [f"Formula: {formula}."]
    if charge is not None and not polymer:
        parts.append(f"Net charge: {charge}.")
    if mass and not polymer:
        parts.append(f"Average mass: {mass} Da.")
    return " ".join(parts)
