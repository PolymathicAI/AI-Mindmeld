"""Which keyword properties another structured field of the same UniProt entry states.

A protein's keyword label is the keyword's presence in the entry, so a "no" is the absence of one keyword. Where
another field of the same entry states the property (a transmembrane feature under a "no" for ``Membrane``), that
"no" is contradicted, and the protein domain marks the (protein, task) cell out of scope
(``domains.protein.ProteinDataSpec.keyword_list``). The rule is frozen: per
keyword (:data:`STATED_BY`), the feature keys, subcellular-location words, GO term names, EC numbers, cofactors,
pathways and residue modifications that state it, and for every keyword a more specific keyword of UniProt's keyword
hierarchy carried by the entry.

Its known limits are accepted and are part of the rule: GO terms are matched by the name printed in
the entry, so a child term whose name lacks the pattern is missed; ``Cytoplasm`` also matches the topology word
"Cytoplasmic", and "ligase activity" matches ubiquitin ligases.

:func:`stated_keywords` applies the rule to one entry and names the stating sources, which the domain keeps as the
reason of each exclusion; :func:`keyword_ancestors` reads the hierarchy from UniProt's ``keywlist.txt``.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from latent_interconnect.domains.uniprot_entries import UniProtEntry

__all__ = ["STATED_BY", "StatedBy", "keyword_ancestors", "stated_keywords"]

HIERARCHY = "KW hierarchy"
"""The source named when the entry carries a more specific keyword of the hierarchy."""

_LOCATION = "SUBCELLULAR LOCATION"
_COFACTOR = "COFACTOR"
_PATHWAY = "PATHWAY"
_BINDING = "BINDING"
_MODIFIED = "MOD_RES"
_QUALIFIER = re.compile(r'/(\w+)="([^"]*)"')
_EC = re.compile(r"EC=([0-9n.\-]+)")
_GO_PARTS = 3  # id, "<aspect>:<term name>", evidence


@dataclass(frozen=True)
class StatedBy:
    """The fields of an entry, other than the keyword itself, that state one keyword's property.

    Attributes:
        features: feature keys (``FT``) whose presence states it.
        location: words whose presence in the subcellular-location comments states it.
        location_any_case: whether the location words match in any letter case.
        go_aspect: the GO aspect read (``C``, ``F`` or ``P``), or ``""`` for none.
        go_contains: lower-case text; a GO term name of the aspect containing one states it.
        go_exact: lower-case GO term names of the aspect that state it.
        go_without: lower-case text; a term name containing one is not counted (``go_contains`` and ``go_exact``).
        ec: EC-number prefixes; an EC number of the description lines starting with one states it.
        ligand: a ligand whose cofactor comment or binding-site feature states it, or ``""`` for none.
        modification: lower-case text; a modified-residue note containing one states it.
        modification_without: lower-case text; a modified-residue note containing one is not counted.
        pathway: the text a pathway comment starts with to state it, or ``""`` for none.
    """

    features: tuple[str, ...] = ()
    location: tuple[str, ...] = ()
    location_any_case: bool = False
    go_aspect: str = ""
    go_contains: tuple[str, ...] = ()
    go_exact: tuple[str, ...] = ()
    go_without: tuple[str, ...] = ()
    ec: tuple[str, ...] = ()
    ligand: str = ""
    modification: tuple[str, ...] = ()
    modification_without: tuple[str, ...] = ()
    pathway: str = ""


def _enzyme(ec_class: str, activity: str) -> StatedBy:
    return StatedBy(ec=(f"{ec_class}.",), go_aspect="F", go_contains=(f"{activity} activity",))


_KINASE_EC = ("2.7.1.", "2.7.2.", "2.7.3.", "2.7.4.", "2.7.10.", "2.7.11.", "2.7.12.", "2.7.13.", "2.7.14.")

STATED_BY: Mapping[str, StatedBy] = {
    "Membrane": StatedBy(
        features=("TRANSMEM", "INTRAMEM"),
        location=("membrane",),
        location_any_case=True,
        go_aspect="C",
        go_contains=("membrane",),
    ),
    "Cell membrane": StatedBy(
        location=("Cell membrane", "Cell inner membrane"), go_aspect="C", go_contains=("plasma membrane",)
    ),
    "Cytoplasm": StatedBy(location=("Cytoplasm",), go_aspect="C", go_exact=("cytoplasm", "cytosol")),
    "Secreted": StatedBy(
        location=("Secreted",), go_aspect="C", go_exact=("extracellular region", "extracellular space")
    ),
    "Nucleus": StatedBy(location=("Nucleus",), go_aspect="C", go_exact=("nucleus", "nucleoplasm", "nucleolus")),
    "Mitochondrion": StatedBy(location=("Mitochondrion",), go_aspect="C", go_contains=("mitochondri",)),
    "Plastid": StatedBy(location=("Plastid",), go_aspect="C", go_contains=("plastid", "chloroplast")),
    "Oxidoreductase": _enzyme("1", "oxidoreductase"),
    "Transferase": _enzyme("2", "transferase"),
    "Hydrolase": _enzyme("3", "hydrolase"),
    "Lyase": _enzyme("4", "lyase"),
    "Isomerase": _enzyme("5", "isomerase"),
    "Ligase": _enzyme("6", "ligase"),
    "Kinase": StatedBy(
        ec=_KINASE_EC,
        go_aspect="F",
        go_contains=("kinase activity",),
        go_without=("regulator", "inhibitor", "activator", "binding"),
    ),
    "DNA-binding": StatedBy(features=("DNA_BIND",), go_aspect="F", go_contains=("dna binding", "dna-binding")),
    "RNA-binding": StatedBy(go_aspect="F", go_contains=("rna binding", "rna-binding")),
    "Ribosomal protein": StatedBy(go_aspect="F", go_exact=("structural constituent of ribosome",)),
    "Chaperone": StatedBy(go_aspect="F", go_contains=("chaperone",), go_exact=("unfolded protein binding",)),
    "Transport": StatedBy(go_aspect="P", go_contains=("transport",)),
    "Transcription regulation": StatedBy(
        go_aspect="P", go_contains=("regulation of transcription", "regulation of dna-templated transcription")
    ),
    "DNA repair": StatedBy(go_aspect="P", go_contains=("repair",)),
    "Photosynthesis": StatedBy(go_aspect="P", go_contains=("photosynthe",)),
    "Protein biosynthesis": StatedBy(go_aspect="P", go_contains=("translational",), go_exact=("translation",)),
    "Amino-acid biosynthesis": StatedBy(
        pathway="Amino-acid biosynthesis", go_aspect="P", go_contains=("amino acid biosynthetic",)
    ),
    "Magnesium": StatedBy(ligand="Mg(2+)", go_aspect="F", go_exact=("magnesium ion binding",)),
    "Zinc": StatedBy(ligand="Zn(2+)", features=("ZN_FING",), go_aspect="F", go_exact=("zinc ion binding",)),
    "Repeat": StatedBy(features=("REPEAT",)),
    "Coiled coil": StatedBy(features=("COILED",)),
    "Acetylation": StatedBy(modification=("acetyl",)),
    "Phosphoprotein": StatedBy(modification=("phospho",), modification_without=("pantethein",)),
    "Disulfide bond": StatedBy(features=("DISULFID",)),
    "Glycoprotein": StatedBy(features=("CARBOHYD",)),
}
"""Keyword → the other fields that state it: the frozen rule."""


def keyword_ancestors(text: str) -> dict[str, frozenset[str]]:
    """Return keyword → the keywords above it in UniProt's hierarchy, from ``keywlist.txt``'s text.

    Each ``HI`` line of a keyword entry is one path from its category down to the keyword
    (``"Biological process: Transport; Ion transport; Calcium transport."``); the keyword's ancestors are the other
    keywords of its paths. A keyword without a path has no entry.
    """
    ancestors: dict[str, set[str]] = {}
    keyword = ""
    for line in text.splitlines():
        code, value = line[:2], line[5:].strip()
        if code == "ID":
            keyword = value.removesuffix(".")
        elif code == "//":
            keyword = ""
        elif code == "HI" and keyword:
            path = {node.strip().removesuffix(".") for node in value.partition(":")[2].split(";")}
            ancestors.setdefault(keyword, set()).update(path - {keyword, ""})
    return {k: frozenset(v) for k, v in ancestors.items()}


def stated_keywords(
    entry: UniProtEntry, keywords: Collection[str], ancestors: Mapping[str, frozenset[str]]
) -> dict[str, tuple[str, ...]]:
    """Return, for each of ``keywords`` that another field of ``entry`` states, the stating sources in a fixed order.

    A source is named by its line type and kind (``"FT TRANSMEM"``, ``"CC SUBCELLULAR LOCATION"``, ``"GO C"``,
    ``"DE EC"``, ``"CC COFACTOR"``, ``"FT BINDING"``, ``"FT MOD_RES"``, ``"CC PATHWAY"``) or is
    :data:`HIERARCHY`. Whether the entry carries the keyword itself is not read: the caller holds the label.

    Args:
        entry: the UniProt entry.
        keywords: the keywords to test, each a key of :data:`STATED_BY`.
        ancestors: keyword → its ancestors (:func:`keyword_ancestors`).

    Raises:
        KeyError: for a keyword :data:`STATED_BY` has no rule for.
    """
    features = _features(entry.features)
    comments = _comments(entry.comments)
    location = " ".join(comments.get(_LOCATION, ()))
    go = _go(entry.go)
    ecs = [number for line in entry.description for number in _EC.findall(line)]
    above = frozenset[str]().union(*(ancestors.get(k, frozenset()) for k in entry.keywords))
    out = {}
    for keyword in keywords:
        rule = STATED_BY[keyword]
        found = [f"FT {key}" for key in rule.features if key in features]
        text = location.lower() if rule.location_any_case else location
        if any(word in text for word in rule.location):
            found.append(f"CC {_LOCATION}")
        if any(_go_states(rule, name) for name in go.get(rule.go_aspect, ())):
            found.append(f"GO {rule.go_aspect}")
        if any(number.startswith(rule.ec) for number in ecs if rule.ec):
            found.append("DE EC")
        if rule.ligand:
            if any(
                part.startswith(f"{rule.ligand};")
                for comment in comments.get(_COFACTOR, ())
                for part in comment.split("Name=")[1:]
            ):
                found.append(f"CC {_COFACTOR}")
            if any(q.get("ligand") == rule.ligand for q in features.get(_BINDING, ())):
                found.append(f"FT {_BINDING}")
        if any(_modification_states(rule, q.get("note", "").lower()) for q in features.get(_MODIFIED, ())):
            found.append(f"FT {_MODIFIED}")
        if rule.pathway and any(c.startswith(rule.pathway) for c in comments.get(_PATHWAY, ())):
            found.append(f"CC {_PATHWAY}")
        if keyword in above:
            found.append(HIERARCHY)
        if found:
            out[keyword] = tuple(found)
    return out


def _go_states(rule: StatedBy, name: str) -> bool:
    named = name in rule.go_exact or any(text in name for text in rule.go_contains)
    return named and not any(text in name for text in rule.go_without)


def _modification_states(rule: StatedBy, note: str) -> bool:
    named = any(text in note for text in rule.modification)
    return named and not any(text in note for text in rule.modification_without)


def _features(lines: tuple[str, ...]) -> dict[str, list[dict[str, str]]]:
    """Feature key → each such feature's qualifiers (a feature's continuation lines are indented).

    Qualifiers are parsed for binding sites and modified residues only, the two kinds whose qualifiers the rule reads.
    """
    found: list[tuple[str, list[str]]] = []
    for line in lines:
        if line and not line.startswith(" "):
            found.append((line.split(None, 1)[0], []))
        elif found:
            found[-1][1].append(line.strip())
    out: dict[str, list[dict[str, str]]] = {}
    for key, qualifiers in found:
        read = key in (_BINDING, _MODIFIED)
        out.setdefault(key, []).append(dict(_QUALIFIER.findall(" ".join(qualifiers))) if read else {})
    return out


def _comments(lines: tuple[str, ...]) -> dict[str, list[str]]:
    """Comment topic → each such comment's text, its wrapped lines joined (the copyright block is no comment)."""
    found: list[list[str]] = []
    open_block = False
    for line in lines:
        if line.startswith("-!- "):
            topic, _, rest = line[4:].partition(":")
            found.append([topic.strip(), rest.strip()])
            open_block = True
        elif line.startswith("---"):
            open_block = False
        elif open_block:
            found[-1][1] += " " + line.strip()
    out: dict[str, list[str]] = {}
    for topic, text in found:
        out.setdefault(topic, []).append(text)
    return out


def _go(lines: tuple[str, ...]) -> dict[str, list[str]]:
    """GO aspect → the lower-cased term names of the entry's GO lines (``"GO:id; P:name; evidence."``)."""
    out: dict[str, list[str]] = {}
    for line in lines:
        parts = line.split("; ")
        if len(parts) >= _GO_PARTS:
            out.setdefault(parts[1][0], []).append(parts[1][2:].lower())
    return out
