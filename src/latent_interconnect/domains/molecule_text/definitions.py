"""ChEBI text for the molecule text: cleaning, names and the usable definition.

The port of the original ``clean_text`` and ``usable_definition``, kept exactly; the release is read
by ``domains.chebi``.
"""

from __future__ import annotations

import re

from latent_interconnect.domains.chebi import ChebiEntry, ChebiRelease

__all__ = ["PARENT_RELS", "clean_text", "name_of", "usable_definition"]

PARENT_RELS = ("is_conjugate_base_of", "is_conjugate_acid_of", "is_tautomer_of")
"""The relations whose target's definition may stand for a charged form's boilerplate one (in this order of names)."""

_GREEK = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ω": "omega",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "Δ": "Delta",
}
# ChEBI definitions of charged microspecies are often boilerplate that only names the parent acid or base
_BOILERPLATE = re.compile(
    r"conjugate (?:base|acid) of|major (?:micro)?species at pH|zwitterionic form of|"
    r"resulting from the (?:removal|addition|transfer|(?:de)?protonation) of|"
    r"obtained by (?:de)?protonation|"
    r"arising from (?:de)?protonation|tautomer of",
    re.I,
)


def clean_text(text: str | None) -> str:
    """Return ChEBI text for prompts: stereo tags unwrapped, HTML tags and links removed, Greek letters spelled out."""
    out = re.sub(r"<stereo>\s*(.*?)\s*</stereo>", r"\1", text or "")
    out = re.sub(r"</?(?:i|b|u|sub|sup|small|em|a|br)\b[^>]*>", "", out)
    for letter, word in _GREEK.items():
        out = out.replace(letter, word)
    return re.sub(r"\s+", " ", out).strip()


def name_of(release: ChebiRelease, chebi_id: str) -> str:
    """Return the cleaned name of ``chebi_id`` (:func:`clean_text`), or ``""`` for an unknown id."""
    entry = release.entries.get(chebi_id)
    return clean_text(entry.name if entry else "")


def usable_definition(release: ChebiRelease, chebi_id: str) -> tuple[str | None, str]:
    """Return a molecule's usable definition and its source.

    The entry's own cleaned definition unless it is charged-form boilerplate (source ``"own"``); else the first
    :data:`PARENT_RELS` target's non-boilerplate definition, in relationship order (source ``"parent <id>"``); else
    ``(None, "boilerplate")`` (an own definition exists) or ``(None, "none")``.
    """
    entry = release.entries.get(chebi_id, ChebiEntry())
    own = clean_text(entry.definition)
    if own and not _BOILERPLATE.search(own):
        return own, "own"
    for rel, target in entry.rels:
        other = release.entries.get(target)
        text = clean_text(other.definition if other else None)
        if rel in PARENT_RELS and text and not _BOILERPLATE.search(text):
            return text, f"parent {target}"
    return None, "boilerplate" if own else "none"
