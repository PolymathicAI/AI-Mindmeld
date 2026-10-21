"""The partition of a molecule's direct ChEBI parent classes into the two class fields.

The port of the original ``Partition`` as the paper's molecule templates apply it, kept exactly. Each
class gets one letter, first match wins:

- **X** (in neither field: charge-state and element classes proper):
  X1, ontology: ``is_a*`` ion, atom, elemental molecular entity or inorganic radical; ``is_a*`` inorganic molecular
  entity unless it is a salt; or a non-zero ChEBI charge. X2, name: a charge or element class name;
- **B** ``parent_family`` (scaffold, ring-system or natural-product family, or parent compound): B1 the class has a
  complete structure; then A1 / A2 below; B2 ``is_a*`` a natural-product family root; B3 a plural core word; B4 the
  definition names a skeleton or parent hydride; B5 ``is_a*`` organic heterocyclic compound unless the ring is the
  functional group's own (peptides, ``-olide`` / ``-lactone`` / ``-lactam`` / ``-peptide`` names);
- **A** ``functional_group_classes``: A1 a functional-group head noun; A2 a ring-count class; A3 everything else.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Literal

from latent_interconnect.domains.chebi import ChebiEntry

__all__ = ["ClassPartition", "Letter"]

Letter = Literal["A", "B", "X"]

_ION, _ATOM, _ELEMENTAL = "CHEBI:24870", "CHEBI:33250", "CHEBI:33259"
_INORGANIC, _INORGANIC_RADICAL, _SALT = "CHEBI:24835", "CHEBI:36871", "CHEBI:24866"
_X_ROOTS = frozenset({_ION, _ATOM, _ELEMENTAL, _INORGANIC_RADICAL})
_FAMILY_ROOTS = frozenset(
    {
        "CHEBI:35341",  # steroid
        "CHEBI:26873",  # terpenoid
        "CHEBI:35186",  # terpene
        "CHEBI:22315",  # alkaloid
        "CHEBI:72544",  # flavonoids
        "CHEBI:47916",  # flavonoid
        "CHEBI:26188",  # polyketide
        "CHEBI:25036",  # lignan
        "CHEBI:23044",  # carotenoid
        "CHEBI:23403",  # coumarins
        "CHEBI:51149",  # xanthones
        "CHEBI:16646",  # carbohydrate
        "CHEBI:63299",  # carbohydrate derivative
        "CHEBI:35381",  # monosaccharide
        "CHEBI:50699",  # oligosaccharide
        "CHEBI:18154",  # polysaccharide
        "CHEBI:23899",  # icosanoid
        "CHEBI:26333",  # prostaglandin
        "CHEBI:17387",  # O-acylcarnitine
        "CHEBI:176910",  # fatty acyl-L-carnitine
        "CHEBI:85034",  # ladderane
    }
)
_RING_EXEMPT = frozenset({"CHEBI:23643", "CHEBI:16670"})  # depsipeptide, peptide
_RING_EXEMPT_NAME = re.compile(r"(?:olide|lactone|lactam|peptide)s?$", re.I)
_HETEROCYCLIC = "CHEBI:24532"
_X_NAME = re.compile(r"^elemental |\bhydride$|coordination entity$|\(\d*[+\-−]\)$|(?:tri|tetra|penta)cation", re.I)
_ELEMENT_WORD = (
    r"(?:hydrogen|carbon|nitrogen|oxygen|fluorine|chlorine|bromine|iodine|astatine|sulfur|selenium|tellurium|"
    r"phosphorus|arsenic|antimony|boron|silicon|germanium|tin|lead|halogen|chalcogen|pnictogen|carbon group|"
    r"[spdf]-block|main group|transition element|alkali metal|alkaline earth metal|metal|lanthanoid|actinoid|"
    r"heteroatomic|polyatomic|inorganic|\w+ group)"
)
_ELEMENT_CLASS = re.compile(
    rf"^{_ELEMENT_WORD} molecular entity$|^organo(?:nitrogen|oxygen|sulfur|halogen|chlorine|fluorine|bromine|iodine|"
    r"phosphorus|chalcogen|selenium|boron|silicon|metallic|pnictogen|tin|arsenic) compound$|"
    r"^heteroorganic entity$|^(?:halide|hydrides|oxide|inorganic hydride)$"
)
_CHARGE_CLASS = re.compile(
    r"\b(?:ion|anion|cation|oxoanion|dianion|trianion|tetraanion|zwitterion|"
    r"monoanion|monocation|dication)s?\b|^dipolar compound$|ammonium ion"
)
_HEADS = frozenset(
    [
        "acid",
        "ester",
        "ether",
        "alcohol",
        "ketone",
        "aldehyde",
        "amide",
        "amine",
        "lactone",
        "lactam",
        "nitrile",
        "oxime",
        "glycoside",
        "sulfide",
        "thiol",
        "imine",
        "phenol",
        "acetamide",
        "formamide",
        "carboxamide",
        "urea",
        "thiourea",
        "guanidine",
        "formamidine",
        "hydroxylamine",
        "isocyanate",
        "thiocyanate",
        "anhydride",
        "peroxide",
        "peroxol",
        "epoxide",
        "acetal",
        "ketal",
        "hemiacetal",
        "hemiketal",
        "enol",
        "enone",
        "diol",
        "triol",
        "tetrol",
        "polyol",
        "carbamate",
        "sulfonamide",
        "sulfone",
        "sulfoxide",
        "azide",
        "hydrazine",
        "hydrazone",
        "imide",
        "thioester",
        "lactol",
        "aldimine",
        "ketimine",
        "alkene",
        "alkyne",
        "olefin",
        "hydrocarbon",
        "ketoester",
        "nitro",
        "disulfide",
        "hydroperoxide",
        "semicarbazone",
        "thioether",
        "sulfonate",
        "phosphate",
        "diphosphate",
        "triphosphate",
        "phosphonate",
        "sulfate",
        "halide",
        "n-oxide",
        "monophosphate",
        "bisphosphate",
        "trisphosphate",
    ]
)
_HEAD_SUFFIX = re.compile(
    r"(?:gluc|galact|mann|xyl|rhamn|fuc|arabin|rib|fruct|glyc|lyx|all|alt|gul|id|tal|quinov|api|hex|pent|hept)"
    r"osides?$|osaminides?$|ester$"
)
_RING_COUNT = re.compile(
    r"^(?:organic |inorganic )?(?:carbo|hetero)?(?:mono|bi|tri|tetra|penta|hexa|hepta|octa|poly)cyclic compound$"
)
_NOT_PLURAL = re.compile(r"(?:ss|us|is|ous|ics)$")
_SKELETON = re.compile(r"\bskeleton\b|\bparent hydride\b", re.I)
_QUALIFIER = re.compile(r"[A-Z0-9]{1,3}|[IVX]+")
_ZERO = ("0", "0.0")


def _head_noun(name: str) -> str:
    """The last word of a class name, its last hyphen segment, lower case; ``(s)`` and a known head's plural dropped."""
    word = name.strip().split()[-1].lower() if name.strip() else ""
    word = re.sub(r"\(s\)$", "", word)
    if word in _HEADS:  # hyphenated heads ('n-oxide')
        return word
    word = word.split("-")[-1]
    return word[:-1] if word.endswith("s") and word[:-1] in _HEADS else word


def _core_word(name: str) -> str:
    """The last word, or the word before a short trailing qualifier (``"prostaglandins D"``)."""
    words = name.strip().split()
    if len(words) >= 2 and _QUALIFIER.fullmatch(words[-1]):  # noqa: PLR2004  (a word and its qualifier)
        return words[-2]
    return words[-1] if words else ""


class ClassPartition:
    """The rule (module docstring) over a release's entries.

    Args:
        entries: CHEBI id → entry.
        ancestry: every term's transitive ``is_a`` ancestors.
        clean: the text cleaning applied to names and definitions.
    """

    def __init__(
        self,
        entries: Mapping[str, ChebiEntry],
        ancestry: Mapping[str, frozenset[str]],
        clean: Callable[[str | None], str],
    ) -> None:
        self.entries, self.ancestry, self.clean = entries, ancestry, clean

    def letter(self, class_id: str) -> Letter:  # noqa: PLR0911  (the rule's steps, first match wins, in its order)
        """Return the class's letter."""
        entry = self.entries.get(class_id, ChebiEntry())
        name = self.clean(entry.name)
        above = self.ancestry.get(class_id, frozenset()) | {class_id}
        charge = entry.charge
        if (
            above & _X_ROOTS
            or (_INORGANIC in above and _SALT not in above)
            or (charge not in (None, "") and charge not in _ZERO)
        ):
            return "X"
        if _CHARGE_CLASS.search(name) or _ELEMENT_CLASS.search(name) or _X_NAME.search(name):
            return "X"
        if entry.smiles and "*" not in entry.smiles:
            return "B"
        head = _head_noun(name)
        if head in _HEADS or _HEAD_SUFFIX.search(head):
            return "A"
        if _RING_COUNT.search(name):
            return "A"
        if above & _FAMILY_ROOTS:
            return "B"
        core = _core_word(name)
        if core.endswith("s") and not _NOT_PLURAL.search(core):
            return "B"
        if _SKELETON.search(self.clean(entry.definition)):
            return "B"
        if _HETEROCYCLIC in above and not (above & _RING_EXEMPT or _RING_EXEMPT_NAME.search(name)):
            return "B"
        return "A"

    def split(self, class_ids: Iterable[str]) -> dict[Letter, list[str]]:
        """Return the classes by letter, each list in the given order."""
        out: dict[Letter, list[str]] = {"A": [], "B": [], "X": []}
        for class_id in class_ids:
            out[self.letter(class_id)].append(class_id)
        return out
