"""The config schema of a training-object selection (``build.train_selection``)."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.build.selection_evidence import SelectionEvidence, SubsetEvidence
from latent_interconnect.config import ArtifactDigest, Config, StageConfig, added_field, rationale
from latent_interconnect.domains.base import DomainSpec
from latent_interconnect.domains.tables import SourceFile, read_source

__all__ = [
    "ElementGroup",
    "HitRule",
    "HomologyExclusion",
    "IdExclusion",
    "IdSource",
    "LengthExclusion",
    "Richness",
    "ShortExclusion",
    "TrainSelectionSpec",
]


class IdSource(Config):
    """Object ids from one column of a table file.

    Attributes:
        table: the parquet file, under the data root, with its sha256.
        column: the id column.
    """

    table: SourceFile
    column: str

    def read(self, data_root: Path) -> set[str]:
        """Return the ids.

        Raises:
            IntegrityError: if the file's sha256 differs from its citation.
            ValueError: if a column is missing.
        """
        frame = pd.read_parquet(read_source(data_root, self.table))
        if self.column not in frame.columns:
            raise ValueError(f"{self.table.path}: no column {self.column!r}")
        return set(frame[self.column].dropna().astype(str))


class HitRule(Config):
    """A hit passes when its identity and the coverage of both sequences reach these values.

    Attributes:
        min_identity: the smallest ``fident``.
        min_coverage: the smallest ``qcov`` and ``tcov`` (0: any coverage).
    """

    min_identity: float = Field(ge=0, le=1)
    min_coverage: float = Field(ge=0, le=1)


class HomologyExclusion(Config):
    """Exclude the targets of a homology search's passing hits.

    Attributes:
        reason: the name its count is recorded under.
        search: the ``homology`` artifact; its targets must be this domain's proteins, the train split included.
        rules: a hit passes when it passes any of them.
        queries: count only hits from these query ids (each must be a query of the search); empty: every query.
    """

    reason: str = Field(min_length=1)
    search: ArtifactDigest
    rules: tuple[HitRule, ...] = Field(min_length=1)
    queries: tuple[IdSource, ...] = ()


class IdExclusion(Config):
    """Exclude listed ids.

    Attributes:
        reason: the name its count is recorded under.
        ids: the sources of the ids.
    """

    reason: str = Field(min_length=1)
    ids: tuple[IdSource, ...] = Field(min_length=1)


class LengthExclusion(Config):
    """Exclude the domain's proteins longer than a residue count (what an encoder of the domain cannot hold).

    A variant is its parent with one residue substituted, so it is as long; with ``drop_excluded_parents`` the
    variants of an excluded protein leave too.

    Attributes:
        reason: the name its count is recorded under.
        longer_than: a protein with more residues than this is excluded.
    """

    reason: str = Field(min_length=1)
    longer_than: int = Field(ge=1)


class ShortExclusion(Config):
    """Exclude the domain's proteins shorter than a residue count (e.g. proteins of 20 residues or fewer leave).

    Only in a selection taken from a ``base`` (:class:`TrainSelectionSpec`): a fill never screens its pool, so there
    the rule would reach the candidates alone.

    Attributes:
        reason: the name its count is recorded under.
        shorter_than: a protein with fewer residues than this is excluded.
    """

    reason: str = Field(min_length=1)
    shorter_than: int = Field(ge=1)


class ElementGroup(Config):
    """One richness point for a list field holding any of some elements.

    Attributes:
        field: the list field (text joined by ``separator``).
        separator: the element separator.
        elements: the elements that count (exact, after stripping).
    """

    field: str
    separator: str = Field(min_length=1)
    elements: tuple[str, ...] = Field(min_length=1)


class Richness(Config):
    """How rich a protein's annotation is: one point per non-empty field, one per element group it meets.

    Attributes:
        fields: the fields that count when non-empty.
        groups: the element groups.
    """

    fields: tuple[str, ...] = ()
    groups: tuple[ElementGroup, ...] = ()


PARENT_REASON = "parent:"
"""What a reason starts with when it is the parent's: a derived object that left with its parent (``removed``)."""

_FILL = ("pool", "max_length", "richness", "target_size", "seed", "with_derived")
"""The fields a fill states and a selection taken from a ``base`` does not (with ``_FILL_ONLY``)."""
_FILL_ONLY = (*_FILL, "required_fields", "drop_excluded_parents")


class TrainSelectionSpec(StageConfig):
    """One selection of training objects (the module docstring gives the two rules: a fill, or a ``base``'s subset).

    A file states either a fill (every field of ``pool``, ``max_length``, ``richness``, ``target_size``, ``seed`` and
    ``with_derived``; they have defaults only so that the other shape loads) or a ``base`` and none of the fill's
    fields.

    Attributes:
        name: the ``train_ids`` object list's name (the ``added`` list's is ``<name>/added``).
        domain: the domain at a fixed data version.
        pool: the sources of the existing training proteins (their union).
        max_length: the longest candidate sequence; None: no length cap (a selection that keeps its pool and adds
            nothing, ``target_size`` = the pool, states none).
        required_fields: fields a candidate must have non-empty.
        exclusions: the exclusions, applied together; reasons are unique.
        richness: how candidates are ordered within a cluster.
        target_size: proteins in the selection, the pool included.
        seed: the seed of the tie-breaking keys (:func:`tie_key`).
        with_derived: add every train-split derived object of the domain (the original variant stream draws every train
            variant).
        drop_excluded_parents: with ``with_derived``, leave out every derived object whose parent an exclusion names,
            whatever the parent's origin (pool, or outside the selection).
        base: an earlier selection's ``train_ids`` list; the selection is then that list less what the exclusions
            name, with nothing added (no refill), or None for a fill (added field).
        evidence: the evidence the selection must reproduce (rationale: not identity), or None.
    """

    name: str = Field(min_length=1)
    domain: DomainSpec
    pool: tuple[IdSource, ...] = ()
    max_length: int | None = Field(default=None, ge=1)
    required_fields: tuple[str, ...] = ()
    exclusions: tuple[HomologyExclusion | IdExclusion | LengthExclusion | ShortExclusion, ...] = ()
    richness: Richness = Richness()
    target_size: int | None = Field(default=None, ge=1)
    seed: int = 0
    with_derived: bool = False
    drop_excluded_parents: bool = added_field(default=False)
    base: ArtifactDigest | None = added_field(default=None)
    evidence: SelectionEvidence | SubsetEvidence | None = rationale(default=None)

    @classmethod
    def fits(cls, keys: Collection[str]) -> bool:
        """Return whether a file with these top-level keys is a selection: a fill's fields, or ``base`` without them."""
        shape = {"name", "domain", *(("base",) if "base" in keys else _FILL)}
        return set(keys) <= set(cls.model_fields) and shape <= set(keys)

    @model_validator(mode="after")
    def _check(self) -> TrainSelectionSpec:
        reasons = [e.reason for e in self.exclusions]
        if len(set(reasons)) != len(reasons):
            raise ValueError(f"exclusion reasons repeat: {reasons}")
        stated = self.model_fields_set
        if self.base is None:
            missing = [f for f in _FILL if f not in stated]
            if missing or not self.pool or self.target_size is None:
                raise ValueError(
                    f"a fill states {list(_FILL)} (a pool source at least); missing: {missing or ['pool, target_size']}"
                )
            short = [e.reason for e in self.exclusions if isinstance(e, ShortExclusion)]
            if short:
                raise ValueError(
                    f"a fill does not screen its pool, so it takes no minimum length ({short}); give a base"
                )
        else:
            extra = [f for f in _FILL_ONLY if f in stated]
            if extra:
                raise ValueError(f"a selection taken from a base adds nothing; it does not read {extra}")
            if any(r.startswith(PARENT_REASON) for r in reasons):
                raise ValueError(f"a reason starting {PARENT_REASON!r} is a derived object's (its parent's): {reasons}")
        return self
