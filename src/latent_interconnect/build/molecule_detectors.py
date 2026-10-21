"""The molecule domain's exposure detectors, registered in ``build.zero_shot.EXPOSURE_DETECTORS``.

The zero-shot screen (``build.zero_shot``) is domain-generic; these two detector kinds let it state what a molecule's
held-out properties look like in text, as ``lexical`` and ``numeric`` do for proteins:

- ``chebi_role`` (:class:`ChebiRoleDetector`): a held-out role (a task whose label is
  ``@role:<ChEBI id>``) is stated in text by its lexicon: its ChEBI name, its synonyms of the declared scopes and the
  same for every role below it in ChEBI's ``is_a`` hierarchy (a molecule with a descendant role has the role: the
  label is the role closure), their spelling and adjective variants, and the task's declared paraphrase stems;
  and in a list of ChEBI terms by an element whose term is the role or below it, or has it (``has_role``). The
  ontology is a hash-pinned ChEBI release under the data root (``domains.chebi``), the one the label table was built
  from;
- ``formula_mass`` (:class:`FormulaMassDetector`): a molecular formula states its compound's mass exactly, so a
  supported formula (:func:`formula_mass`) whose computed average mass is strictly greater than the configured positive
  threshold in Da states the property.

Like every detector, they read the text, not whom it is about (a formula may be a fragment's, a role a partner's), so
each is scoped to the fields where its measured evidence says it states the property of the object
(``build.zero_shot.DetectorSpec``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal

from pydantic import Field

from latent_interconnect.build.zero_shot import EXPOSURE_DETECTORS
from latent_interconnect.config import Config
from latent_interconnect.domains.base import Domain, TaskSpec
from latent_interconnect.domains.chebi import load_release, name_key
from latent_interconnect.domains.tables import SourceFile
from latent_interconnect.paths import Paths

__all__ = [
    "ATOMIC_WEIGHTS",
    "ChebiRoleDetector",
    "FormulaMassDetector",
    "formula_mass",
]

SynonymScope = Literal["EXACT", "RELATED", "BROAD", "NARROW"]
_ROLE_KEY = "@role:"
_EDGE = r"(?<![A-Za-z0-9])"
_END = r"s?(?![A-Za-z0-9])"  # a plural "s" is the same statement
_MIN_CORE = 6  # the shortest adjective form kept (the study's rule): a shorter core is too generic a word


def _alternation(words: Iterable[str]) -> str:
    """Return a regular expression matching exactly ``words``, as a character trie (fast on thousands of names)."""
    trie: dict[str, Any] = {}
    for word in words:
        node = trie
        for ch in word:
            node = node.setdefault(ch, {})
        node[""] = {}

    def emit(node: dict[str, Any]) -> str:
        branches = [re.escape(ch) + emit(sub) for ch, sub in sorted(node.items()) if ch]
        if not branches:
            return ""
        body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
        if "" in node:
            return (body if len(branches) > 1 else "(?:" + body + ")") + "?"
        return body

    return emit(trie)


def _variants(names: Iterable[str]) -> re.Pattern[str]:
    """Return the spelling and adjective variants of ``names`` (the screen study's rule).

    Spelling: ``anti`` with or without a hyphen, ``ae`` / ``e``, ``our`` / ``or``, a space or a hyphen between words.
    Adjective: a multi-word name ending in ``agent`` or ``drug`` whose last other word is an adjective (``anti...``,
    ``-ic``, ``-ive``, ``-ant``, ``-ory``, ``-al``) also matches without that word (``antineoplastic``), and an ``-ide``
    ending also as ``-idal`` (``insecticidal``). Whole words, case-insensitive, an optional plural ``s``.
    """
    forms: set[str] = set()
    for name in names:
        core = re.sub(r"\s+(agents?|drugs?)$", "", name)
        own = {name}
        if core != name and len(core) >= _MIN_CORE and re.search(r"(^anti|ic|ive|ant|ory|al)$", core.split()[-1]):
            own.add(core)
        own |= {f[:-1] + "al" for f in own if f.endswith("ide")}
        forms |= own
    pats = set()
    for f in forms:
        p = re.sub(r"anti(\\-)?", r"anti-?", re.escape(f))
        pats.add(p.replace("ae", "a?e").replace("our", "ou?r").replace("\\ ", r"[\s\-]"))
    return re.compile(_EDGE + "(?:" + "|".join(sorted(pats, key=lambda q: (-len(q), q))) + ")" + _END, re.IGNORECASE)


class _ChebiRoleParams(Config):
    ontology: SourceFile
    synonyms: tuple[SynonymScope, ...]
    min_length: int = Field(ge=1)
    stems: tuple[str, ...] = ()


@EXPOSURE_DETECTORS.register("chebi_role")
class ChebiRoleDetector:
    """A ChEBI role in text (its lexicon) and in a list of ChEBI terms (the ontology).

    The task's role is its label's ChEBI id (``@role:CHEBI:35610``). Its names are its own and its ``is_a``
    descendants', with their synonyms of the declared scopes; names shorter than ``min_length`` characters are left out
    (a short abbreviation names too many things to state one role). Text matches a name, a spelling or adjective
    variant of one (:func:`_variants`), whole and case-insensitively with an optional plural ``s``, or a paraphrase stem
    (``stems``: regular expressions, case-insensitive, anywhere). A list element matches when it is the ChEBI name
    (``domains.chebi.name_key``) of a non-obsolete term that is the role or below it in ``is_a``, or whose
    ``has_role`` names one of those, or when a name of the role is in it.

    Params:
        ontology: the ChEBI OBO release (``SourceFile``: path under the data root, sha256).
        synonyms: the synonym scopes that count (the OBO scopes ``EXACT``, ``RELATED``, ``BROAD``, ``NARROW``).
        min_length: the shortest name kept, in characters.
        stems: the task's paraphrase stems (the screen study's, for the application roles); none = role words.

    Raises (at construction):
        pydantic.ValidationError: for unknown or missing params.
        ValueError: if the task's label is not ``@role:<id>``, or the role has no name of ``min_length``.
        KeyError: if the release has no such role.
        IntegrityError: if the release's sha256 differs from its citation.
    """

    Params: ClassVar[type[Config]] = _ChebiRoleParams

    def __init__(self, domain: Domain, task: TaskSpec, params: Mapping[str, Any]) -> None:
        del domain
        p = _ChebiRoleParams.model_validate(dict(params))
        if not task.label.startswith(_ROLE_KEY):
            raise ValueError(
                f"task {task.id!r}: a chebi_role detector needs a label {_ROLE_KEY}<id>, not {task.label!r}"
            )
        release = load_release(p.ontology, Paths.from_env().data)
        role = task.label.removeprefix(_ROLE_KEY)
        below = frozenset(release.descendants(role))
        names: set[str] = set()
        for term_id in below:
            term = release.entries[term_id]
            names.add(term.name)
            names.update(text for text, scope in term.synonyms if scope in p.synonyms)
        self.role = role
        self.names = frozenset(n.lower() for n in names if len(n) >= p.min_length)
        if not self.names:
            raise ValueError(f"task {task.id!r}: role {role} has no name of at least {p.min_length} characters")
        self.pattern = re.compile(_EDGE + "(?:" + _alternation(sorted(self.names)) + ")" + _END, re.IGNORECASE)
        self.lexicon = [self.pattern, _variants(self.names), *(re.compile(s, re.IGNORECASE) for s in p.stems)]
        self.terms = frozenset(
            name_key(e.name)
            for t, e in release.entries.items()
            if not e.obsolete and e.name and (t in below or not below.isdisjoint(e.has_role))
        )

    def matches(self, object_id: str, text: str) -> list[tuple[int, int]]:
        """Return the spans of the role's lexicon in ``text``."""
        del object_id
        return sorted({m.span() for pattern in self.lexicon for m in pattern.finditer(text)})

    def matches_element(self, object_id: str, element: str) -> bool:
        """Return whether a list element is a term that states the role, or holds one of the role's names."""
        del object_id
        return name_key(element) in self.terms or bool(self.pattern.search(element))


# fmt: off
ATOMIC_WEIGHTS: Mapping[str, float] = {
    # IUPAC CIAAW abridged standard atomic weights (2021; Pure Appl. Chem. 94, 573), the conventional value for an
    # element given as an interval. Elements without a standard atomic weight (Tc, Pm, Po, At, Rn, Fr, Ra, Ac, Np on)
    # are absent, so a formula naming one is not read.
    "H": 1.0080, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Ar": 39.95, "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630,
    "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.95, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82,
    "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29, "Cs": 132.91, "Ba": 137.33, "La": 138.91,
    "Ce": 140.12, "Pr": 140.91, "Nd": 144.24, "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50,
    "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05, "Lu": 174.97, "Hf": 178.49, "Ta": 180.95, "W": 183.84,
    "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59, "Tl": 204.38, "Pb": 207.2,
    "Bi": 208.98, "Th": 232.04, "Pa": 231.04, "U": 238.03,
}
# fmt: on
"""Element symbol → standard atomic weight (Da), for :func:`formula_mass`."""

# A carbon compound's formula in Hill notation (C21H24N2O3): carbon, hydrogen, then the other elements in alphabetical
# order, each once; checked by formula_mass.
_FORMULA = re.compile(r"(?<![A-Za-z0-9])C\d*H\d*(?:[A-Z][a-z]?\d*)*(?![A-Za-z0-9])")
_GROUP = re.compile(r"([A-Z][a-z]?)(\d*)")
_HALOGENS = ("F", "Cl", "Br", "I")
_MIN_UNSATURATION = -1.0


def formula_mass(token: str) -> float | None:
    """Return the average mass of a carbon compound's formula in Hill notation (``C6H12O6``), or None.

    Hill notation, the form ChEBI writes formulas in: ``C`` and ``H`` first, then the other elements in alphabetical
    order, each element once, every count a positive integer without a leading zero (1 is not written), and a
    hydrogen count a structure can have (rings plus double bonds at least -1). A token that is not one (``IC50``,
    ``PI3K``, ``SU5402``, ``CB104``, ``CH5126766``), or names an element without a standard atomic weight, is not a
    formula. The mass is the sum of the standard atomic weights (:data:`ATOMIC_WEIGHTS`).
    """
    groups = _GROUP.findall(token)
    symbols = [s for s, _ in groups]
    if "".join(s + n for s, n in groups) != token or symbols[:2] != ["C", "H"]:
        return None
    rest = symbols[2:]
    if rest != sorted(set(rest)) or "C" in rest or "H" in rest or any(n.startswith("0") for _, n in groups):
        return None
    if any(s not in ATOMIC_WEIGHTS for s in rest):
        return None
    count = {s: int(n) if n else 1 for s, n in groups}
    # The rings plus double bonds a formula implies (halogens count as hydrogen, trivalent N and P add half a bond
    # each): below -1 (a cation's -0.5 allowed) no structure has it, so the token is a code name (CH5126766).
    unsaturation = count["C"] + 1 + (count.get("N", 0) + count.get("P", 0)) / 2
    unsaturation -= (count["H"] + sum(count.get(x, 0) for x in _HALOGENS)) / 2
    if unsaturation < _MIN_UNSATURATION:
        return None
    return sum(ATOMIC_WEIGHTS[s] * n for s, n in count.items())


class _FormulaMassParams(Config):
    threshold: float = Field(gt=0)


@EXPOSURE_DETECTORS.register("formula_mass")
class FormulaMassDetector:
    """A molecular formula whose average mass is above a threshold: ``with formula C23H28O12`` (480.5 Da).

    A formula fixes its compound's mass, so the object's own formula entails "mass above ``threshold``" when its mass
    is above it; the detector cannot tell whose formula it is, so its scope rests on the measured evidence. A formula
    at or below the threshold is not matched: it states the negation (a protein length at or below its threshold is
    not matched either), and the screen removes statements of the property only.

    Params:
        threshold: the configured mass threshold in Da, a positive number; a formula matches when its computed
            average mass is strictly greater.
    """

    Params: ClassVar[type[Config]] = _FormulaMassParams

    def __init__(self, domain: Domain, task: TaskSpec, params: Mapping[str, Any]) -> None:
        del domain, task
        self.threshold = _FormulaMassParams.model_validate(dict(params)).threshold

    def matches(self, object_id: str, text: str) -> list[tuple[int, int]]:
        """Return the spans of formulas heavier than the threshold."""
        del object_id
        out = []
        for m in _FORMULA.finditer(text):
            mass = formula_mass(m.group(0))
            if mass is not None and mass > self.threshold:
                out.append(m.span())
        return out

    def matches_element(self, object_id: str, element: str) -> bool:
        """Return whether a list element holds such a formula."""
        return bool(self.matches(object_id, element))
