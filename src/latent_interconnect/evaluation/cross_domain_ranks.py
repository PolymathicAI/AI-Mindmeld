"""The ``cross_domain_ranks`` analysis kind: E4 (ranking) and E5 (decoy pairs) per cell, crossed draws.

E4 ranks the true partner among candidates; E5 orders the true pair against decoys. Both come with the crossed
bootstrap and paired contrasts between cells.

**Design** (the frozen Rhea design's secondary endpoints). The item set is a ``cross_domain`` set of design ``pairs``
(``evaluation.cross_domain``) with one item per (group, candidate); a candidate serving several groups is one item per
group, and the runner scores its prompt once. An :class:`Endpoint` keeps the items whose item-set fields match its
``select``, groups them by its ``group`` field and takes one statistic per group from the true pair's score ``s0`` (the
group's one item labeled 1) and the other candidates' scores, with ``n_gt`` of them above ``s0`` and ``n_eq`` tied:

- ``reciprocal_rank``: ``1 / (1 + n_gt + n_eq / 2)``, ties at the mean rank; its mean over groups is E4's MRR;
- ``top1``: ``1 / (1 + n_eq)`` when ``n_gt = 0``, else 0: the expected top-1 under uniform random tie-breaking;
- ``unique_top``: 1 when ``n_gt = n_eq = 0``: the strict unique-top rate, reported beside top-1;
- ``auroc``: the group's AUROC (``stats.auroc``), the share of candidates ordered below ``s0``, ties ½: E5's pair
  ordering, E1's statistic on a group with one positive.

The endpoint's value in a cell is the unweighted mean over its groups.

**The Rhea item set** (``xdom/rhea-2x2-ranking/items-<cell>``; built by ``build.cross_domain_items``, the frozen
builder's rules, per cell). Per positive pair ``(P_i, M_i)`` of the
cell's swap quads, with quad partner ``(P_j, M_j)``, two ranking sets of k = 4 candidates, one item each:

- ``ranking: compound`` (``P_i`` with four compounds): the ``pair``; the ``swap`` ``(P_i, M_j)``; two ``matched``
  decoys ``(P_i, M)``: the novel metabolites nearest ``M_i`` by ECFP4 Tanimoto (radius 2, 2,048 bits), one per
  molecule group, that are admissible (a Rhea small-molecule participant somewhere, not a currency metabolite, not a
  participant of ``P_i``'s reactions) and strict (not a participant of any Swiss-Prot protein sharing a full EC
  number with ``P_i``), never ``M_j``;
- ``ranking: protein`` (``M_i`` with four proteins): the ``pair``; the ``swap`` ``(P_j, M_i)``; two
  ``protein_decoy`` s ``(P, M_i)``: novel proteins with a Rhea reaction and none on ``M_i``, outside ``P_i``'s
  cluster, never ``P_j``, drawn (seeded) from ``P_i``'s EC sub-subclass, then its EC class, then any.

Fields: ``pair_label`` (1 for the pair), ``anchor`` (the positive pair's id: the group of both rankings), ``ranking``,
``candidate`` (``pair``, ``swap``, ``matched``, ``protein_decoy``), and the units ``protein_cluster`` and
``molecule_group`` with every set's ``cluster.<role>``. E4 groups one ranking by ``anchor``; E5 keeps one ranking's
``pair`` and decoys.

**Crossed bootstrap**, as E1's (``evaluation.cross_domain_quads``): the factors (``AnalysisSpec.units``) are item-set
columns; a group's units are its items' distinct values, and in a draw it weighs the product of their draw counts
(``QuadDesign.from_groups``). Every cell of an endpoint sees the same draws, so a contrast is the percentile interval
of the per-draw difference. A draw in which every group weighs 0 has no value; it is dropped and counted
(``Estimate.dropped``).

**Shared-unit interval** (``units.interval: shared_unit``; ``stats.shared_units``). An opt-in
alternative to the crossed bootstrap, which stays the default: the observations are an endpoint's groups, their
influence values ``(statistic − mean)/n`` (for a contrast, the two cells' difference), their units the group's units
as above (every candidate's, decoys included), and the statistic's range ``[0, 1]``, or ``[1/k, 1]`` for the
reciprocal rank of ``k`` candidates (the largest group's ``k``). The method's one confirmation used a fixed lower
bound of 0.25 (every group of four candidates): groups of unequal sizes, where the bound comes from the largest group,
were not part of it. The point estimates are the same. Empirically
conservative and approximately 95%, with one unresolved shortfall, found on this kind's reciprocal rank: the module
docstring of ``stats.shared_units``.

**Estimates only.** The result is a ``GroupedContrastsResult`` whose groups are the endpoints, so a
reader reads it as it reads ``grouped_contrasts``. Its ``support`` is keyed cell → endpoint: the endpoint's
selected items (``n``), its true pairs (``positives``, one per group) and its other candidates (``negatives``). It
depends on the item set and the endpoint only (:func:`endpoint_support`), so a reader derives it for a record written
before the field was added.

**Refusals.** Every input must be a suite record on one item set; a cell must hold exactly one score per item, of one
task; a group must hold exactly one true pair and at least one other candidate; each factor needs at least two units
among an endpoint's items.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.records import Record, record_output
from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.config import Config
from latent_interconnect.data.items import ItemSet
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS, Contrast, Group, GroupedContrasts
from latent_interconnect.evaluation.contrasts import Observed, grouped_result, shared_unit_result, suite_of
from latent_interconnect.evaluation.cross_domain_quads import QuadDesign, cell_rows
from latent_interconnect.evaluation.results import AnalysisResult, Cell, GroupedContrastsResult, Support
from latent_interconnect.evaluation.scores import SCORES_KIND, read_scores
from latent_interconnect.stats.auroc import auroc
from latent_interconnect.stats.bootstrap import Bootstrap, ResamplingUnits
from latent_interconnect.stats.shared_units import SHARED_UNIT

__all__ = ["CrossDomainRanks", "CrossDomainRanksKind", "Endpoint", "endpoint_support"]

FieldValue = str | int | float | bool
_MIN_CANDIDATES = 2  # the true pair and at least one other candidate


class Endpoint(Config):
    """One E4 or E5 endpoint: which items, grouped how, and the per-group statistic.

    Attributes:
        group: the item-set field naming each item's group (a ranking set or an anchor's decoy pairs).
        select: item-set field → accepted values; the endpoint keeps the items matching every entry (e.g.
            ``{ranking: [compound], candidate: [true, matched]}``).
        statistic: ``reciprocal_rank``, ``top1``, ``unique_top`` or ``auroc`` (module docstring).
    """

    group: str
    select: dict[str, tuple[FieldValue, ...]] = Field(default_factory=dict)
    statistic: Literal["reciprocal_rank", "top1", "unique_top", "auroc"]


class CrossDomainRanks(Config):
    """Parameters of the ``cross_domain_ranks`` kind.

    Attributes:
        cells: cell name → :class:`~latent_interconnect.evaluation.results.Cell`; the name is a label.
        endpoints: endpoint name → :class:`Endpoint` (the result's groups).
        contrasts: paired contrasts ``a − b`` between listed cells, reported on every endpoint (named
            ``<name>/<endpoint>`` when there are several, as ``grouped_contrasts`` names them).
    """

    cells: dict[str, Cell] = Field(min_length=1)
    endpoints: dict[str, Endpoint] = Field(min_length=1)
    contrasts: tuple[Contrast, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> CrossDomainRanks:
        unknown = sorted({c for x in self.contrasts for c in (x.a, x.b)} - set(self.cells))
        if unknown:
            raise ValueError(f"cells {unknown} are not among the listed cells")
        return self


@ANALYSIS_KINDS.register("cross_domain_ranks")
class CrossDomainRanksKind:
    """E4 and E5 endpoints per cell, with crossed draws shared by the cells (params :class:`CrossDomainRanks`)."""

    Params: ClassVar[type[Config]] = CrossDomainRanks
    Result: ClassVar[type[AnalysisResult]] = GroupedContrastsResult
    interval_methods: ClassVar[tuple[str, ...]] = (SHARED_UNIT,)

    def run(
        self,
        params: CrossDomainRanks,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> GroupedContrastsResult:
        """Compute every endpoint of every cell and the contrasts.

        Raises:
            KeyError: if a cell names an unknown input, or a group, selection or factor field is not in the item set.
            ValueError: if the inputs are not on one item set, a cell does not hold one score per item of one task,
                or a group does not hold one true pair and another candidate.
        """
        digests = {str(suite_of(rec)["item_set"]) for rec in inputs.values()}
        if len(digests) != 1:
            raise ValueError(f"cross_domain_ranks needs every input on one item set, got {len(digests)}")
        item_set = items(digests.pop())
        tables = {key: read_scores(artifacts, record_output(rec, SCORES_KIND)) for key, rec in inputs.items()}
        rows = {c: cell_rows(tables, c, spec, len(item_set)) for c, spec in params.cells.items()}
        per_cell: dict[str, dict[str, tuple[float, np.ndarray]]] = {c: {} for c in params.cells}
        observed: dict[str, dict[str, Observed]] = {c: {} for c in params.cells}
        support: dict[str, dict[str, Support]] = {c: {} for c in params.cells}
        shared = units.interval == SHARED_UNIT
        for name, endpoint in params.endpoints.items():
            keep, group = _groups(item_set, endpoint)
            design = QuadDesign.from_groups(_subset(item_set, keep), group, units.factors)
            boot = None if shared else Bootstrap(dict(design.unit_counts))
            for c, (scores, positive) in rows.items():
                stats = _group_stats(scores[keep], positive[keep], group, endpoint.statistic, f"{c}/{name}")
                if boot is None:
                    observed[c][name] = Observed(
                        estimate=design.weighted_mean(stats, None),
                        influence=(stats - stats.mean()) / stats.size,
                        keys=np.arange(stats.size),
                        units=design.units,
                        unit_counts=design.unit_counts,
                        bounds=_range(endpoint.statistic, group),
                    )
                else:
                    estimate = design.weighted_mean(stats, None)
                    per_cell[c][name] = (estimate, boot.draws(partial(design.weighted_mean, stats)))
                support[c][name] = _support(keep, group)
        task = next(iter(tables.values()))["task"].iloc[0]
        as_groups = GroupedContrasts(
            cells=params.cells,
            groups=tuple(Group(name=e, tasks=(str(task),)) for e in params.endpoints),
            contrasts=params.contrasts,
        )
        if shared:
            return shared_unit_result(as_groups, units, observed, support=support)
        return grouped_result(as_groups, units, per_cell, support=support)


def endpoint_support(items: ItemSet, endpoint: Endpoint) -> Support:
    """Return an endpoint's support on an item set, as :meth:`CrossDomainRanksKind.run` records it in every cell.

    ``n`` is the selected items, ``positives`` the groups (each holds exactly one true pair, or the run refuses) and
    ``negatives`` the other candidates. It reads the item set only, never a score.

    Raises:
        KeyError: if a group or selection field is not in the item set.
        ValueError: if no item matches the selection.
    """
    return _support(*_groups(items, endpoint))


def _range(statistic: str, group: np.ndarray) -> tuple[float, float]:
    """Return a group statistic's range: ``[1/k, 1]`` for a reciprocal rank (``k``: largest group), else ``[0, 1]``."""
    return (1.0 / int(np.max(np.bincount(group))), 1.0) if statistic == "reciprocal_rank" else (0.0, 1.0)


def _support(keep: np.ndarray, group: np.ndarray) -> Support:
    groups = int(group.max()) + 1  # one true pair per group, checked by _group_stats
    return Support(n=int(keep.size), positives=groups, negatives=int(keep.size) - groups)


def _groups(items: ItemSet, endpoint: Endpoint) -> tuple[np.ndarray, np.ndarray]:
    """Return the kept items' indices and their group indices (groups numbered in sorted order of their keys)."""
    mask = np.ones(len(items), dtype=bool)
    for f, accepted in endpoint.select.items():
        mask &= np.array([v in accepted for v in _field(items, f)])
    keep = np.flatnonzero(mask)
    if keep.size == 0:
        raise ValueError(f"no item matches the selection {endpoint.select}")
    keys = [str(_field(items, endpoint.group)[i]) for i in keep]
    index = {k: n for n, k in enumerate(sorted(set(keys)))}
    return keep, np.array([index[k] for k in keys], dtype=np.int64)


def _field(items: ItemSet, name: str) -> tuple[object, ...]:
    if name not in items.fields:
        raise KeyError(f"item set {items.name!r} has no field {name!r}")
    return items.fields[name]


def _subset(items: ItemSet, keep: np.ndarray) -> ItemSet:
    def take(column: tuple[object, ...]) -> tuple[object, ...]:
        return tuple(column[i] for i in keep)

    return ItemSet(
        items.name,
        items.roles,
        {r: tuple(str(o) for o in take(c)) for r, c in items.objects.items()},
        {f: take(c) for f, c in items.fields.items()},
    )


def _group_stats(scores: np.ndarray, positive: np.ndarray, group: np.ndarray, statistic: str, where: str) -> np.ndarray:
    """Return each group's statistic (module docstring)."""
    frame = pd.DataFrame({"g": group, "s": scores, "y": positive})
    out = np.empty(int(group.max()) + 1, dtype=np.float64)
    for key, rows in frame.groupby("g", sort=True):
        g = int(key)
        s, y = rows["s"].to_numpy(), rows["y"].to_numpy(bool)
        if int(y.sum()) != 1 or len(y) < _MIN_CANDIDATES:
            raise ValueError(f"{where}: group {g} holds {int(y.sum())} true pair(s) among {len(y)} candidates")
        s0, others = s[y][0], s[~y]
        n_gt, n_eq = int((others > s0).sum()), int((others == s0).sum())
        if statistic == "auroc":
            out[g] = auroc(s, y)
        elif statistic == "reciprocal_rank":
            out[g] = 1.0 / (1.0 + n_gt + 0.5 * n_eq)
        elif statistic == "top1":
            out[g] = 1.0 / (1.0 + n_eq) if n_gt == 0 else 0.0
        else:
            out[g] = float(n_gt == 0 and n_eq == 0)
    return out
