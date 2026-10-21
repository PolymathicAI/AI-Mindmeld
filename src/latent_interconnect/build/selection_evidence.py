"""The evidence of a training-object selection: its counts, cluster statistics and the searches it read.

A selection stores its :class:`SelectionEvidence` in its artifacts' metadata; a spec that states one is held to it
(:class:`EvidenceMismatchError`, ``build.train_selection``). A selection taken from a base stores a
:class:`SubsetEvidence` instead.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from latent_interconnect.config import Config, added_field

__all__ = [
    "ClusterStats",
    "EvidenceMismatchError",
    "ExclusionEvidence",
    "SearchEvidence",
    "SelectionEvidence",
    "SubsetEvidence",
    "cluster_stats",
]

_TOP_SHARE = 0.01  # the top 1% of clusters (TSA's "top-1%" column)
_SHARE_DIGITS, _NEFF_DIGITS = 4, 1  # evidence rounding, so a stated value compares exactly


class EvidenceMismatchError(ValueError):
    """A selection's evidence differs from the evidence its spec states."""


class ClusterStats(Config):
    """Cluster statistics of a protein set.

    Attributes:
        proteins: proteins in the set.
        clusters: clusters they fall in.
        singletons: clusters holding one of them.
        largest: proteins in the largest cluster.
        top1_share: share of the proteins in the largest 1% of clusters (rounded up to whole clusters).
        n_eff: the effective number of clusters under ``cluster_sqrt`` draws (a cluster with probability ∝
            √size), the exponential of their entropy.
    """

    proteins: int
    clusters: int
    singletons: int
    largest: int
    top1_share: float
    n_eff: float


class SearchEvidence(Config):
    """What one homology search ran: copied from its artifact.

    Attributes:
        name: the search's name.
        mmseqs_version: the version the binary reported.
        arguments: the search arguments.
        both_directions: whether it also searched the other way.
        queries: query sequences.
        targets: target sequences.
        hits: hits per direction.
    """

    name: str
    mmseqs_version: str
    arguments: tuple[str, ...]
    both_directions: bool
    queries: int
    targets: int
    hits: dict[str, int]


class ExclusionEvidence(Config):
    """One exclusion's counts.

    Attributes:
        train: train-split proteins it names (pool included).
        removed: candidates it removes (one removed by several reasons counts under each).
        search: the search it reads, or None for an id list or a length rule.
    """

    train: int
    removed: int
    search: SearchEvidence | None = None


class SelectionEvidence(Config):
    """The counts behind a selection (the module docstring's steps).

    Attributes:
        pool_listed: distinct ids the pool sources list.
        pool_outside_train: of them, not train-split proteins of the domain (dropped).
        pool: the pool kept.
        train_proteins: train-split proteins of the domain.
        candidates: train-split proteins outside the pool.
        over_max_length: candidates longer than ``max_length``.
        missing_required: candidates within ``max_length`` that lack a required field.
        exclusions: reason → its counts.
        eligible: candidates left after every exclusion.
        added: proteins added.
        level_reached: the highest level an added protein has.
        new_clusters: clusters the pool did not touch that the selection does.
        new_residues: residues of the added proteins.
        derived: train-split derived objects (``with_derived``).
        derived_dropped: of them, left out because an exclusion names their parent (``drop_excluded_parents``).
        pool_stats: the pool's clusters.
        selected_stats: the selection's clusters (pool and added).
    """

    pool_listed: int
    pool_outside_train: int
    pool: int
    train_proteins: int
    candidates: int
    over_max_length: int
    missing_required: int
    exclusions: dict[str, ExclusionEvidence]
    eligible: int
    added: int
    level_reached: int
    new_clusters: int
    new_residues: int
    derived: int
    derived_dropped: int | None = added_field(default=None)
    pool_stats: ClusterStats
    selected_stats: ClusterStats


class SubsetEvidence(Config):
    """The counts behind a selection taken from a base (``TrainSelectionSpec.base``): the base less its exclusions.

    Origins are the ``train_ids`` list's (``pool``, ``added``, ``derived``); an origin with no object has no key.

    Attributes:
        base: origin → objects of the base.
        listed: reason → the ids the exclusion names, in the base or not (a list may be domain-wide).
        exclusions: reason → its counts: ``train`` as in a fill, ``removed`` the base objects it names itself (a
            derived object that leaves with its parent is counted in ``followed_parent``, not here).
        removed: origin → base objects that left, for any reason.
        followed_parent: derived objects that left because an exclusion names their parent and none names them.
        selected: origin → objects of the selection.
        parents: the selection's derived objects' parents outside it.
        selected_stats: the selection's clusters (pool and added).
    """

    base: dict[str, int]
    listed: dict[str, int]
    exclusions: dict[str, ExclusionEvidence]
    removed: dict[str, int]
    followed_parent: int
    selected: dict[str, int]
    parents: int
    selected_stats: ClusterStats


def cluster_stats(clusters: pd.Series) -> ClusterStats:
    """Return the :class:`ClusterStats` of a protein set, given each protein's cluster."""
    sizes = clusters.value_counts().sort_values(ascending=False, kind="stable").to_numpy(dtype=np.int64)
    top = math.ceil(_TOP_SHARE * len(sizes))
    p = np.sqrt(sizes) / np.sqrt(sizes).sum()
    return ClusterStats(
        proteins=int(sizes.sum()),
        clusters=len(sizes),
        singletons=int((sizes == 1).sum()),
        largest=int(sizes[0]),
        top1_share=round(float(sizes[:top].sum() / sizes.sum()), _SHARE_DIGITS),
        n_eff=round(float(np.exp(-(p * np.log(p)).sum())), _NEFF_DIGITS),
    )
