"""Cross-domain item sets on Rhea per cell: the swap quads (E1), the ranking (E4, E5) and direction (E6) sets.

The rules are the frozen Rhea design's, ported (the original funnel for the positives and the quads, the original
item builder for the secondary sets), cut into familiar/novel cells and screened for exposure with the machinery
the BindingDB 2x2 shares (``build.xdom_screen``: the cells, the screen, the quad draw). One
:class:`CrossDomainItemSpec` builds one design for its ``cells``; every design re-derives the same positives and
quads per cell from the same inputs, so the secondary sets sit on the E1 set's quads of their cell.

**Sources** (:class:`RheaSources`, each a path under the data root with its sha256): UniProt's Swiss-Prot table
(``Entry``, ``Protein names``, ``Gene Names (primary)``, ``EC number``, ``Catalytic activity``; the screen reads its
names and equations too), Rhea's reaction participants (``rhea``, ``side``, ``chebi``) and directions (master, LR,
RL, BI ids), and the currency-metabolite list (``chebi_id``; every listed id is excluded). Every Rhea id is read as
its master reaction. A molecule's unit is its form group, the molecule domain's ``strata`` column ``group_id``
(the independence unit; a ChEBI id outside the domain is its own group).

**Cells** (``build.xdom_screen.CELLS``). A protein is *familiar* when it is an object of the protein training
set (``familiar_proteins``), *novel* when it is in the pool ``novel_proteins`` states before its draw (the clean
holdout; ``build.item_selection.eligible_pool``). A molecule is *familiar* when it is an object of the molecule
training set (``familiar_molecules``), *novel* when it is strict-held-out: in the molecule domain's population, split
val or test, with a value, of a ``strata`` stratum (strict N), outside the groups of the ``exclude_molecules`` sets
(the in-job set).

**Positives.** A (protein, molecule) pair of a built cell is a positive when the protein's catalytic-activity block
cites the Rhea reaction with experimental evidence (``ECO:0000269``, tier B; no stated direction needed, since the
question is substrate-or-product) and the molecule is a ChEBI participant of that reaction, not a currency
metabolite.

**Exposure screen** (``build.xdom_screen``, keyed by ChEBI id). A molecule's names are its molecule-domain name
with and without a trailing charge; a trained equation names the ChEBI ids its cross-references hold; a familiar
molecule's training object is itself. Stated positives are removed per cell, by reason.

**Quads** (per cell; ``build.xdom_screen.draw_quads``, units the protein's cluster and the molecule's group, at most
``per_unit_cap`` positives per unit). A pair is admissible when neither molecule's group is the group of any
participant of any annotated Rhea reaction of the other's protein, whatever its evidence, and neither
swap is stated. Each pair ``(i, j)`` is a swap quad (``evaluation.cross_domain``). A cell with no quad is reported,
not built.

**Designs** (``design``):

- ``swap_quads``: per quad the two positives and the two swaps; fields ``quad`` and ``pair_label``.
- ``ranking`` (E4, E5; ``evaluation.cross_domain_ranks`` holds the contract; novel-molecule cells only): per quad
  positive (the ``anchor``), a ``compound`` ranking (the pair, the swap ``(P_i, M_j)``, ``decoys`` ``matched``
  molecules) and a ``protein`` ranking (the pair, the swap ``(P_j, M_i)``, ``decoys`` ``protein_decoy`` proteins).
  Matched molecules are the novel molecules nearest ``M_i`` by Tanimoto on the ``fingerprints`` (ECFP4,
  ``encoders.extractors.ecfp``), one per group: a participant of some Rhea reaction, not currency, and strict, i.e.
  not in the group of ``M_i``, ``M_j``, a participant of ``P_i``'s reactions, or a participant of any Swiss-Prot
  protein sharing a full EC number with ``P_i``. Protein decoys are novel proteins with an annotated Rhea reaction,
  none on ``M_i``'s group, outside ``P_i``'s cluster, never ``P_i`` or ``P_j``, drawn (seeded per cell) from
  ``P_i``'s EC sub-subclass, then the first two EC levels, then any. A decoy edge (``(P_i, matched)``,
  ``(protein_decoy, M_i)``) the exposure screen states is rejected before selection: a ranking build's
  screen covers every such edge (each cell protein with each pool molecule, each decoy protein with each cell
  molecule) besides the positives. A ranking the supply cannot fill stays short (counted; rejected edges counted as
  ``decoys_stated``).
- ``direction`` (E6): every quad positive whose role is defined, labeled 1 for a substrate and 0 for a product. A
  role is defined when each tier-B reaction of the protein that has the molecule annotates exactly one physiological
  direction, with experimental evidence, and they agree (the frozen T4 rule).

Every item carries the units ``protein_cluster`` and ``molecule_group``, each role's ``cluster.<role>``
(``data.items.with_clusters``) and the molecule's strata columns but ``group_id`` as ``<column>.molecule``. Besides
each built cell's item set (``<name>/<cell>``), a build commits the object lists of its proteins and molecules (the
extractions' inputs) and ``molecule_pool``, every candidate matched molecule (the fingerprints' input). The funnel's
and each cell's counts are the evidence, in the cell's item-set metadata with the spec.

**Retained units** (``CrossDomainItemSpec.retained``, :class:`RetainedUnits`: a corrected version of the evaluated
sets, as the retained quads of ``build.bindingdb_items`` for BindingDB). With ``retained`` the spec reads no source
and draws nothing: it keeps the units of
already built cells of its design (``within``) that a declared table does not remove, as they are
(``build.retained_quads``: the same items in the same order with the same unit keys and fields). A unit is a swap
quad (its four items), a ranking anchor (both rankings: every candidate row) or a direction group (its one item); it
leaves whole, and nothing is paired, matched or drawn again, so a ranking never loses a single candidate. The table
(parquet: ``design``, ``cell``, ``unit``, ``reason``; one row per removed unit, one file for the three designs) is
the declaration, and the build holds it to the rows:

- **Standing reasons are what the rows show.** A declared reason of the closed list below must hold for a member of
  the unit, or the build refuses; the manifest stores every standing reason the unit carries, the declared one first
  (``removed``), and the members that carry each (``removed_members``). The list: :data:`MIN_LENGTH_REASON` and the
  ``excluded_ids`` reasons of ``novel_proteins`` (a novel protein); the ``excluded_molecules`` reasons (a molecule on
  the list); :data:`LEFT_TRAINING_REASON` (a familiar object that is no object of its training set);
  :data:`NOW_TRAINED_REASON` (a novel object that is one); :data:`PARENT_REASON` (below). Any other reason is the
  declaration's own and is stored as declared.
- **A kept unit holds only eligible objects.** A familiar object is an object of its training set; a novel protein is
  in the pool ``novel_proteins`` states (its ``min_length`` and ``excluded_ids`` applied); a novel molecule is
  strict-held-out in ``molecule`` as a draw's is, and on no ``excluded_molecules`` list. Otherwise the build
  refuses: the declaration removes the unit, or the build stops. (A retained ranking is of novel-protein cells
  only, so its protein decoys are held as the cell's proteins.)
- **A ranking anchor and a direction group follow their parent quad** (``parents``: the cell's swap-quad set). The
  parent is the quad whose positive is the unit's pair; a unit without one is refused, and a kept unit whose parent
  the table removes (its ``swap_quads`` rows) is refused: it is declared with :data:`PARENT_REASON`. The suite kind
  and the analyses read a direction group on its own (design ``pairs``); that it leaves with its parent is the
  retained set's rule, stated in the config that declares it.
- **The units are those the set stores.** Every kept protein has the pool's sequence in ``protein``, and each item's
  ``protein_cluster``, ``molecule_group``, strata columns and ``cluster.<role>`` equal what the two domains give now.
- **The draw's fields** (:data:`_DRAW`) repeat those of the ``within`` sets and are not applied.
- **Structure.** Each ``within`` set and each kept set passes the design's check: the suite kind's
  (``evaluation.cross_domain``: ``swap_quads`` or ``pair_labels``, passed in by the stage, since this layer lies
  below it) and, for a ranking or direction set, :func:`design_structure`.

A retained build commits ``items-<cell>`` and the object lists ``proteins`` and ``molecules`` of the kept items (no
``molecule_pool``: nothing is matched).
"""

from __future__ import annotations

import csv
import gzip
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.build.item_selection import MIN_LENGTH_REASON, ItemSelectionSpec, eligible_pool
from latent_interconnect.build.retained_quads import QUAD, retain_quads
from latent_interconnect.build.selection_spec import IdExclusion
from latent_interconnect.build.xdom_screen import (
    CELLS,
    MOLECULE,
    PROTEIN,
    Cell,
    ExposureScreen,
    draw_quads,
    name_variants,
    screen_positives,
    training_objects,
)
from latent_interconnect.config import ArtifactDigest, Config, StageConfig, added_field
from latent_interconnect.data.items import (
    ITEM_SET_KIND,
    ItemSet,
    commit_item_set,
    load_item_set,
    object_list,
    with_clusters,
)
from latent_interconnect.domains.base import Domain, DomainSpec, open_domain
from latent_interconnect.domains.tables import SourceFile, read_source
from latent_interconnect.encoders.store import OUTPUT_LAYER, EmbeddingReader

__all__ = [
    "LEFT_TRAINING_REASON",
    "NOW_TRAINED_REASON",
    "PARENT_REASON",
    "CrossDomainItemSpec",
    "RetainedUnits",
    "RheaSources",
    "build_cross_domain_items",
    "design_structure",
    "retained_cells",
    "rhea_cells",
]

_HELD_OUT = ("val", "test")
_EXPERIMENTAL = "ECO:0000269"
_RHEA = re.compile(r"Rhea:RHEA:(\d+)")
_DIRECTION = {"left-to-right": "LR", "right-to-left": "RL"}
_EC = re.compile(r"^\d+\.\d+\.\d+\.[\dn\-]+$")
_EC_FULL = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
_GROUP = "group_id"
_PROTEIN, _MOLECULE = PROTEIN, MOLECULE

_ANCHOR = "anchor"
_UNIT = {"swap_quads": QUAD, "ranking": _ANCHOR, "direction": _ANCHOR}
"""Design → the item-set field naming each item's unit."""
_DRAW = ("design", "exclude_molecules", "sources", "strata", "per_unit_cap", "fingerprints", "decoys", "seed")
"""The spec fields that state the draw: a retained subset repeats those of its ``within`` sets."""
_RANKINGS = {"compound": "matched", "protein": "protein_decoy"}
"""Ranking → its decoys' ``candidate`` kind (beside the ``pair`` and the ``swap``)."""
LEFT_TRAINING_REASON = "left-training"
"""The reason, in ``removed``, of a unit with a familiar object that is no object of its training set."""
NOW_TRAINED_REASON = "now-trained"
"""The reason, in ``removed``, of a unit with a novel object that is an object of its training set."""
PARENT_REASON = "parent-quad-removed"
"""The reason, in ``removed``, of a ranking anchor or direction group whose parent quad the declaration removes."""
_SHOWN = 5

CellName = Literal["famP-famM", "famP-novM", "novP-famM", "novP-novM"]


class RheaSources(Config):
    """The outside sources of the Rhea item sets (module docstring), each checked by sha256 before it is read.

    Attributes:
        catalytic: UniProt's Swiss-Prot table (gzipped TSV: ``Entry``, ``Protein names``, ``Gene Names (primary)``,
            ``EC number``, ``Catalytic activity``).
        participants: Rhea's reaction participants (TSV: ``rhea``, ``side``, ``chebi``).
        directions: Rhea's direction table (TSV: ``RHEA_ID_MASTER``, ``RHEA_ID_LR``, ``RHEA_ID_RL``, ``RHEA_ID_BI``).
        currency: the currency metabolites (TSV with ``#`` comments; column ``chebi_id``).
    """

    catalytic: SourceFile
    participants: SourceFile
    directions: SourceFile
    currency: SourceFile


class RetainedUnits(Config):
    """The units kept of already built cells (module docstring, "Retained units").

    Attributes:
        within: cell → the built cell's item set of the spec's design.
        excluded: the removed units, a parquet table: ``design``, ``cell``, ``unit``, ``reason`` (one row per unit;
            the spec reads its design's rows and, for the parents, the ``swap_quads`` rows; other columns are the
            declaration's evidence).
        excluded_molecules: listed molecules no kept unit holds, each list with its reason.
        parents: cell → the cell's swap-quad item set (``ranking`` and ``direction``: a unit follows its parent
            quad); empty for ``swap_quads``.
    """

    within: dict[CellName, ArtifactDigest] = Field(min_length=1)
    excluded: SourceFile
    excluded_molecules: tuple[IdExclusion, ...] = ()
    parents: dict[CellName, ArtifactDigest] = Field(default_factory=dict)


class CrossDomainItemSpec(StageConfig):
    """One Rhea cross-domain design over familiar/novel cells (the module docstring gives the rules).

    Attributes:
        name: the item sets' name prefix (each cell's set is ``<name>/<cell>``).
        design: ``swap_quads`` (E1), ``ranking`` (E4, E5) or ``direction`` (E6).
        cells: the cells built (``build.xdom_screen.CELLS``; ``ranking``: novel-molecule cells only).
        protein: the protein domain at a fixed data version.
        molecule: the molecule domain at a fixed data version (its ``strata`` table has ``stratum`` and ``group_id``).
        familiar_proteins: the protein training set (a ``training_set`` artifact: objects and their trained text).
        novel_proteins: the selection whose eligible pool, before its draw, is the novel proteins.
        familiar_molecules: the molecule training set (a ``training_set`` artifact).
        exclude_molecules: item sets whose molecules' groups no novel molecule is in (the in-job set).
        sources: the outside sources.
        strata: the ``stratum`` values a novel molecule may have (``N``, strict).
        per_unit_cap: the most positives one protein cluster or one molecule group enters.
        fingerprints: the ECFP4 embeddings of ``molecule_pool`` (``ranking`` only).
        decoys: matched molecules and protein decoys per ranking (``ranking`` only).
        seed: the seed of the matching's jitter and of the protein-decoy draws.
        retained: draw nothing: keep the units of built cells that a declared table does not remove, held to the
            fields above (module docstring, "Retained units"; added field).
    """

    name: str = Field(min_length=1)
    design: Literal["swap_quads", "ranking", "direction"]
    cells: tuple[CellName, ...] = Field(min_length=1)
    protein: DomainSpec
    molecule: DomainSpec
    familiar_proteins: ArtifactDigest
    novel_proteins: ItemSelectionSpec
    familiar_molecules: ArtifactDigest
    exclude_molecules: tuple[ArtifactDigest, ...] = ()
    sources: RheaSources
    strata: tuple[str, ...] = Field(min_length=1)
    per_unit_cap: int = Field(ge=1)
    fingerprints: ArtifactDigest | None = None
    decoys: int | None = Field(default=None, ge=1)
    seed: int
    retained: RetainedUnits | None = added_field(default=None)

    @model_validator(mode="after")
    def _design_inputs(self) -> CrossDomainItemSpec:
        if len(set(self.cells)) != len(self.cells):
            raise ValueError(f"cells repeat: {self.cells}")
        ranking = self.design == "ranking"
        if ranking != (self.fingerprints is not None) or ranking != (self.decoys is not None):
            raise ValueError("fingerprints and decoys are set exactly when design is ranking")
        familiar = [c for c in self.cells if CELLS[c][1] != "novel"]
        if ranking and familiar:
            raise ValueError(
                f"a ranking draws matched molecules from the novel pool; familiar-molecule cells {familiar}"
            )
        if self.retained is not None:
            parents = set(self.cells) if self.design != "swap_quads" else set()
            if set(self.retained.within) != set(self.cells) or set(self.retained.parents) != parents:
                raise ValueError(
                    f"retained: within names exactly the cells {list(self.cells)}, and parents the same cells for a "
                    "ranking or direction design and none for swap quads"
                )
            if ranking and any(CELLS[c][0] != "novel" for c in self.cells):
                raise ValueError(
                    f"retained: a ranking's protein decoys are novel proteins, held as the cell's; cells {self.cells}"
                )
            reasons = [e.reason for e in (*self.novel_proteins.excluded_ids, *self.retained.excluded_molecules)]
            reserved = {MIN_LENGTH_REASON, LEFT_TRAINING_REASON, NOW_TRAINED_REASON, PARENT_REASON}
            if len(set(reasons)) != len(reasons) or reserved & set(reasons):
                raise ValueError(f"retained: list reasons repeat or are reserved ({sorted(reserved)}): {reasons}")
        return self


@dataclass
class _Rhea:
    """What the sources say per protein: tier-B reactions, all reactions, EC numbers, directions."""

    tier_b: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    every: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    ec: dict[str, list[str]] = field(default_factory=dict)
    direction: dict[tuple[str, int], str | None] = field(default_factory=dict)  # tier-B (protein, reaction)
    sides: dict[int, dict[str, str]] = field(default_factory=dict)  # reaction -> chebi -> side (L / R)


def _masters(path: Path) -> dict[int, int]:
    table = pd.read_csv(path, sep="\t", dtype=int)
    return {int(v): int(r[0]) for r in table.itertuples(index=False) for v in r}


def _direction(segments: list[str]) -> str | None:
    """The one experimentally evidenced physiological direction of a reaction, or None (none, several, unevidenced)."""
    found = set()
    for seg in segments:
        m = re.match(r"\s*(left-to-right|right-to-left)", seg)
        if m is None:
            return None
        found.add((_DIRECTION[m.group(1)], _EXPERIMENTAL in seg))
    if len(found) != 1:
        return None
    ((tag, experimental),) = found
    return tag if experimental else None


def _read_rhea(sources: RheaSources, data_root: Path) -> _Rhea:
    masters = _masters(read_source(data_root, sources.directions))
    out = _Rhea()
    with gzip.open(read_source(data_root, sources.catalytic), "rt", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        entry, ec, activity = (header.index(c) for c in ("Entry", "EC number", "Catalytic activity"))
        for row in reader:
            acc = row[entry]
            out.ec[acc] = [e.strip() for e in row[ec].replace(",", ";").split(";") if _EC.match(e.strip())]
            for block in row[activity].split("CATALYTIC ACTIVITY:")[1:] if len(row) > activity else ():
                head, *segments = block.split("PhysiologicalDirection=")
                for rid in _RHEA.findall(head):
                    master = masters.get(int(rid), int(rid))
                    out.every[acc].add(master)
                    if _EXPERIMENTAL in head:
                        out.tier_b[acc].add(master)
                        out.direction[(acc, master)] = _direction(segments)
    parts = pd.read_csv(read_source(data_root, sources.participants), sep="\t", dtype=str)
    parts = parts[parts["chebi"].fillna("").str.len() > 0]  # a compound with a ChEBI id (rhea_v2b_funnel.py)
    for r in parts.itertuples(index=False):
        out.sides.setdefault(masters.get(int(r.rhea), int(r.rhea)), {})[str(r.chebi)] = str(r.side)
    return out


def _role(rhea: _Rhea, acc: str, chebi: str) -> str | None:
    roles = set()
    for master in rhea.tier_b.get(acc, ()):
        side = rhea.sides.get(master, {}).get(chebi)
        if side is None:
            continue
        d = rhea.direction.get((acc, master))
        roles.add(None if d is None else "substrate" if (d == "LR") == (side == "L") else "product")
    return roles.pop() if len(roles) == 1 else None


@dataclass
class _Funnel:
    """The positives of the built cells, their units and every lookup the designs need."""

    positives: pd.DataFrame  # protein, molecule, p_side, m_side, cluster, group (sorted by protein, molecule)
    protein_cluster: pd.Series
    group: Callable[[str], str]
    partner_groups: Callable[[str], set[str]]  # groups of every participant of every annotated reaction of a protein
    ec_groups: dict[str, set[str]]  # full EC number -> groups of its proteins' participants
    decoy_proteins: list[str]  # novel proteins with an annotated Rhea reaction (sorted)
    pool: list[str]  # candidate matched molecules: novel Rhea participants, not currency (sorted)
    rhea: _Rhea
    strata: pd.DataFrame
    evidence: dict[str, Any]


@dataclass
class _Built:
    """The funnel, each built cell, the two domains and the exposure screen."""

    funnel: _Funnel
    cells: dict[str, Cell]
    protein: Domain
    molecule: Domain
    screen: ExposureScreen


def _monitor_groups(store: ArtifactStore, exclude: Iterable[str], molecule: Domain, groups: pd.Series) -> set[str]:
    out: set[str] = set()
    for digest in exclude:
        items = load_item_set(store, ArtifactRef(digest, ITEM_SET_KIND))
        for role, name in items.roles.items():
            if name == molecule.name:
                out |= set(groups.reindex(pd.Index(items.objects[role])).dropna())
    return out


def _novel_molecules(
    spec: CrossDomainItemSpec, molecule: Domain, strata: pd.DataFrame, groups: pd.Series, monitor: set[str]
) -> pd.Index:
    population = molecule.population()
    split = molecule.split().reindex(population)
    ids = population[split.isin(_HELD_OUT).to_numpy()]
    values = molecule.objects()["value"].reindex(ids).fillna("").astype(str)
    ids = ids[(values.str.len() > 0).to_numpy()]
    ids = ids[strata["stratum"].reindex(ids).isin(spec.strata).to_numpy()]
    return ids[~groups.reindex(ids).isin(monitor).to_numpy()]


def _funnel(spec: CrossDomainItemSpec, store: ArtifactStore, data_root: Path) -> tuple[_Funnel, Domain, Domain]:
    protein, molecule = open_domain(spec.protein, store), open_domain(spec.molecule, store)
    strata = molecule.table("strata")
    if strata is None or not {"stratum", _GROUP} <= set(strata.columns):
        raise ValueError(f"molecule domain {spec.molecule.data[:12]} has no strata table with stratum and {_GROUP}")
    rhea = _read_rhea(spec.sources, data_root)
    gid = strata[_GROUP].dropna()
    groups = pd.Series([f"g{int(g)}" for g in gid], index=gid.index.astype(str))
    currency = set(
        pd.read_csv(read_source(data_root, spec.sources.currency), sep="\t", comment="#")["chebi_id"].astype(str)
    )
    familiar_p = set(training_objects(store, spec.familiar_proteins)) & set(protein.population())
    novel_p = set(eligible_pool(spec.novel_proteins, store).eligible.index)
    if familiar_p & novel_p:
        raise ValueError(f"proteins both familiar and novel, e.g. {sorted(familiar_p & novel_p)[:3]}")
    familiar_m = set(training_objects(store, spec.familiar_molecules)) & set(molecule.population())
    monitor = _monitor_groups(store, spec.exclude_molecules, molecule, groups)
    novel_m = set(_novel_molecules(spec, molecule, strata, groups, monitor))
    if familiar_m & novel_m:
        raise ValueError(f"molecules both familiar and novel, e.g. {sorted(familiar_m & novel_m)[:3]}")

    def group(c: str) -> str:
        return str(groups.get(c, c))

    participants = {m: set(s) for m, s in rhea.sides.items()}
    partner_cache: dict[str, set[str]] = {}

    def partner_groups(acc: str) -> set[str]:
        if acc not in partner_cache:
            partner_cache[acc] = {group(c) for m in rhea.every.get(acc, ()) for c in participants.get(m, ())}
        return partner_cache[acc]

    ec_groups: dict[str, set[str]] = defaultdict(set)
    for acc, ecs in rhea.ec.items():
        for e in ecs:
            if _EC_FULL.match(e):
                ec_groups[e] |= partner_groups(acc)
    side_p = {"familiar": familiar_p, "novel": novel_p}
    side_m = {"familiar": familiar_m - currency, "novel": novel_m - currency}
    p_sides = {CELLS[c][0] for c in spec.cells}
    m_sides = {CELLS[c][1] for c in spec.cells}
    rows = sorted(
        {
            (a, c, ps, ms)
            for ps in p_sides
            for a in side_p[ps]
            for m in rhea.tier_b.get(a, ())
            for c in participants.get(m, ())
            for ms in m_sides
            if c in side_m[ms]
        }
    )
    pos = pd.DataFrame(rows, columns=[_PROTEIN, _MOLECULE, "p_side", "m_side"])
    pos["cluster"] = protein.clusters().reindex(pos[_PROTEIN]).astype(str).to_numpy()
    pos["group"] = pos[_MOLECULE].map(group)
    rhea_molecules = {c for s in participants.values() for c in s}
    pool = sorted(novel_m & rhea_molecules - currency)
    evidence = {
        "pools": {
            "familiar_proteins": len(familiar_p),
            "novel_proteins": len(novel_p),
            "familiar_molecules": len(familiar_m),
            "novel_molecules": len(novel_m),
            "injob_groups": len(monitor),
        },
        "positives_by_side": {
            f"{ps}P-{ms}M": int(((pos["p_side"] == ps) & (pos["m_side"] == ms)).sum())
            for ps in p_sides
            for ms in m_sides
        },
        "molecule_pool": len(pool),
    }
    funnel = _Funnel(
        pos,
        protein.clusters(),
        group,
        partner_groups,
        ec_groups,
        sorted(a for a in novel_p if rhea.every.get(a)),
        pool,
        rhea,
        strata,
        evidence,
    )
    return funnel, protein, molecule


def _cell_quads(
    pos: pd.DataFrame, funnel: _Funnel, screen: ExposureScreen, spec: CrossDomainItemSpec
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    blocked = {"partner_reaction": 0, "swap_stated": 0}
    p, m, g = pos[_PROTEIN].to_numpy(), pos[_MOLECULE].to_numpy(), pos["group"].to_numpy()

    def weigh(members: list[int]) -> Iterator[tuple[int, int, float]]:
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if g[j] in funnel.partner_groups(p[i]) or g[i] in funnel.partner_groups(p[j]):
                    blocked["partner_reaction"] += 1
                    continue
                if screen.stated(p[i], m[j]) or screen.stated(p[j], m[i]):
                    blocked["swap_stated"] += 1
                    continue
                yield a, b, 1.0

    return draw_quads(pos, spec.per_unit_cap, spec.seed, weigh), blocked


def _units(pos: pd.DataFrame) -> dict[str, int]:
    return {
        "positives": len(pos),
        "proteins": int(pos[_PROTEIN].nunique()),
        "protein_clusters": int(pos["cluster"].nunique()),
        "molecules": int(pos[_MOLECULE].nunique()),
        "molecule_groups": int(pos["group"].nunique()),
    }


def _decoy_edges(funnel: _Funnel) -> pd.DataFrame:
    """Return every edge a ranking's decoys may form, which its screen covers.

    Each positive's protein with each pool molecule (matched molecules), each decoy protein with each positive's
    molecule (protein decoys).
    """
    proteins, molecules = sorted(set(funnel.positives[_PROTEIN])), sorted(set(funnel.positives[_MOLECULE]))
    matched = pd.MultiIndex.from_product([proteins, funnel.pool], names=[_PROTEIN, _MOLECULE])
    decoys = pd.MultiIndex.from_product([funnel.decoy_proteins, molecules], names=[_PROTEIN, _MOLECULE])
    return pd.concat([e.to_frame(index=False) for e in (matched, decoys)], ignore_index=True)


def rhea_cells(spec: CrossDomainItemSpec, store: ArtifactStore, data_root: Path) -> _Built:
    """Apply the rules (module docstring) and return the funnel and the cells' screened positives and quads.

    Raises:
        IntegrityError: if a source's sha256 differs from the spec.
        ValueError: if the molecule domain has no strata with ``stratum`` and ``group_id``, or an object is both
            familiar and novel.
    """
    funnel, protein, molecule = _funnel(spec, store, data_root)
    positives = funnel.positives
    candidates = positives
    if spec.design == "ranking":  # the screen also covers the decoy edges
        candidates = pd.concat([positives[[_PROTEIN, _MOLECULE]], _decoy_edges(funnel)], ignore_index=True)
    used = sorted(set(candidates[_MOLECULE]))
    names = molecule.objects()["name"].reindex(pd.Index(used))
    familiar = set(positives.loc[positives["m_side"] == "familiar", _MOLECULE])
    screen = ExposureScreen(
        store,
        protein=protein,
        familiar_proteins=spec.familiar_proteins,
        familiar_molecules=spec.familiar_molecules,
        uniprot=read_source(data_root, spec.sources.catalytic),
        candidates=candidates,
        names={c: name_variants(n) for c, n in zip(used, names, strict=True)},
        chebi_keys={c: c for c in used},
        trained_objects={c: (c,) for c in sorted(familiar)},
    )
    funnel.evidence["screen"] = screen.evidence()
    cells: dict[str, Cell] = {}
    for name in spec.cells:
        ps, ms = CELLS[name]
        pos = positives[(positives["p_side"] == ps) & (positives["m_side"] == ms)]
        kept, removed = screen_positives(pos, screen)
        quads, blocked = _cell_quads(kept, funnel, screen, spec) if len(kept) else ([], {})
        q = kept.iloc[sorted({k for pair in quads for k in pair})]
        cells[name] = Cell(
            kept,
            quads,
            {
                "before_screen": _units(pos),
                "removed_by_screen": removed,
                "after_screen": _units(kept),
                "swap_pairings_blocked": blocked,
                "quads": len(quads),
                "quad_units": _units(q),
            },
        )
    return _Built(funnel, cells, protein, molecule, screen)


def _row(funnel: _Funnel, p: str, m: str, label: int, **extra: object) -> dict[str, object]:
    return {
        _PROTEIN: p,
        _MOLECULE: m,
        "pair_label": label,
        "protein_cluster": str(funnel.protein_cluster[p]),
        "molecule_group": funnel.group(m),
        **extra,
    }


def _swap_quads(funnel: _Funnel, cell: Cell) -> list[dict[str, object]]:
    pos = cell.positives
    rows = []
    for q, (i, j) in enumerate(cell.quads):
        pi, mi, pj, mj = pos[_PROTEIN].iat[i], pos[_MOLECULE].iat[i], pos[_PROTEIN].iat[j], pos[_MOLECULE].iat[j]
        for p, m, y in ((pi, mi, 1), (pj, mj, 1), (pi, mj, 0), (pj, mi, 0)):
            rows.append(_row(funnel, p, m, y, quad=f"q{q:04d}"))
    return rows


def _anchors(quads: list[tuple[int, int]]) -> list[tuple[str, int, int]]:
    """(anchor id, positive, its quad partner) per quad positive."""
    return [(f"a{2 * q + s}", (i, j)[s], (j, i)[s]) for q, (i, j) in enumerate(quads) for s in (0, 1)]


def _direction_rows(funnel: _Funnel, cell: Cell) -> list[dict[str, object]]:
    pos = cell.positives
    rows, undefined = [], 0
    for anchor, i, _ in _anchors(cell.quads):
        p, m = pos[_PROTEIN].iat[i], pos[_MOLECULE].iat[i]
        role = _role(funnel.rhea, p, m)
        if role is None:
            undefined += 1
            continue
        rows.append(_row(funnel, p, m, int(role == "substrate"), anchor=anchor))
    cell.evidence |= {"direction_items": len(rows), "direction_undefined": undefined}
    return rows


def _tanimoto(store: ArtifactStore, digest: str, pool: list[str]) -> tuple[list[str], np.ndarray]:
    reader = EmbeddingReader(store, ArtifactRef(digest, "embeddings"))
    recipe = reader.meta.get("recipe")
    if not isinstance(recipe, Mapping) or recipe.get("extractor") != "ecfp":
        raise ValueError(f"fingerprints {digest[:12]} are not an ecfp extraction (recipe {recipe!r})")
    ids = [c for c in pool if c in reader]
    return ids, reader.pooled(ids, OUTPUT_LAYER).astype(np.float64) if ids else np.zeros((0, 0))


def _matched(
    built: _Built, pos: pd.DataFrame, fingerprints: tuple[list[str], np.ndarray], pair: tuple[int, int], k: int
) -> tuple[list[tuple[str, float]], int]:
    """Return the matched molecules of positive ``pair[0]``, nearest first, and the number of stated ones passed."""
    funnel = built.funnel
    (ids, bits), (i, j) = fingerprints, pair
    p, m = pos[_PROTEIN].iat[i], pos[_MOLECULE].iat[i]
    if m not in ids:
        return [], 0
    strict = funnel.partner_groups(p) | {funnel.group(m), funnel.group(pos[_MOLECULE].iat[j])}
    for e in funnel.rhea.ec.get(p, ()):
        strict |= funnel.ec_groups.get(e, set())
    x = bits[ids.index(m)]
    inter = bits @ x
    sims = inter / np.maximum(bits.sum(1) + x.sum() - inter, 1.0)
    chosen: list[tuple[str, float]] = []
    seen: set[str] = set()
    stated = 0
    for n in np.argsort(-sims, kind="stable"):
        g = funnel.group(ids[n])
        if g in strict or g in seen:
            continue
        if built.screen.stated(p, ids[n]):
            stated += 1
            continue
        seen.add(g)
        chosen.append((ids[n], round(float(sims[n]), 4)))
        if len(chosen) == k:
            break
    return chosen, stated


def _protein_decoys(
    built: _Built, pos: pd.DataFrame, pair: tuple[int, int], k: int, rng: np.random.Generator
) -> tuple[list[tuple[str, str]], int]:
    """Return the protein decoys of positive ``pair[0]`` with their EC level, and how many eligible ones are stated."""
    funnel = built.funnel
    i, j = pair
    p, m = pos[_PROTEIN].iat[i], pos[_MOLECULE].iat[i]
    ecs = funnel.rhea.ec.get(p, [])
    levels = (
        ("ec3", {".".join(e.split(".")[:3]) for e in ecs}, 3),
        ("ec2", {".".join(e.split(".")[:2]) for e in ecs}, 2),
        ("any", None, 0),
    )
    excluded = {p, pos[_PROTEIN].iat[j]}
    stated: set[str] = set()
    chosen: list[tuple[str, str]] = []
    for level, prefixes, depth in levels:
        eligible = [
            b
            for b in funnel.decoy_proteins
            if b not in excluded
            and funnel.protein_cluster[b] != funnel.protein_cluster[p]
            and funnel.group(m) not in funnel.partner_groups(b)
            and (prefixes is None or any(".".join(e.split(".")[:depth]) in prefixes for e in funnel.rhea.ec.get(b, ())))
        ]
        stated |= {b for b in eligible if built.screen.stated(b, m)}  # rejected before the draw
        cands = [b for b in eligible if b not in stated]
        take = min(k - len(chosen), len(cands))
        for x in rng.choice(len(cands), size=take, replace=False) if take else ():
            chosen.append((cands[int(x)], level))
            excluded.add(cands[int(x)])
        if len(chosen) == k:
            break
    return chosen, len(stated)


def _ranking_rows(
    built: _Built, cell: Cell, spec: CrossDomainItemSpec, fingerprints: tuple[list[str], np.ndarray]
) -> list[dict[str, object]]:
    if spec.decoys is None:  # the validator guarantees it; this narrows the type
        raise ValueError("a ranking build needs decoys")
    funnel = built.funnel
    rng = np.random.default_rng(spec.seed)
    pos = cell.positives
    rows: list[dict[str, object]] = []
    short = {"compound": 0, "protein": 0}
    rejected = {"matched": 0, "protein_decoy": 0}
    levels: dict[str, int] = defaultdict(int)
    for anchor, i, j in _anchors(cell.quads):
        p, m = pos[_PROTEIN].iat[i], pos[_MOLECULE].iat[i]
        pj, mj = pos[_PROTEIN].iat[j], pos[_MOLECULE].iat[j]
        matched, n_matched = _matched(built, pos, fingerprints, (i, j), spec.decoys)
        decoys, n_decoys = _protein_decoys(built, pos, (i, j), spec.decoys, rng)
        rejected["matched"] += n_matched
        rejected["protein_decoy"] += n_decoys
        short["compound"] += len(matched) < spec.decoys
        short["protein"] += len(decoys) < spec.decoys
        for _, level in decoys:
            levels[level] += 1
        compound = [(p, m, 1, "pair"), (p, mj, 0, "swap")] + [(p, c, 0, "matched") for c, _ in matched]
        protein = [(p, m, 1, "pair"), (pj, m, 0, "swap")] + [(b, m, 0, "protein_decoy") for b, _ in decoys]
        for ranking, cands in (("compound", compound), ("protein", protein)):
            for a, c, y, kind in cands:
                rows.append(_row(funnel, a, c, y, anchor=anchor, ranking=ranking, candidate=kind))
    cell.evidence |= {
        "short_rankings": short,
        "protein_decoy_levels": dict(sorted(levels.items())),
        "decoys_stated": rejected,
    }
    return rows


def _items(name: str, rows: list[dict[str, object]], built: _Built) -> ItemSet:
    table = pd.DataFrame(rows)
    extra = {c: tuple(table[c]) for c in table.columns if c not in (_PROTEIN, _MOLECULE)}
    strata = built.funnel.strata
    for column in strata.columns.drop(_GROUP):
        values = strata[column].reindex(table[_MOLECULE])
        extra[f"{column}.molecule"] = tuple(None if pd.isna(v) else str(v) for v in values)
    items = ItemSet(
        name,
        {_PROTEIN: built.protein.name, _MOLECULE: built.molecule.name},
        {_PROTEIN: tuple(table[_PROTEIN]), _MOLECULE: tuple(table[_MOLECULE])},
        extra,
    )
    return with_clusters(items, {built.protein.name: built.protein, built.molecule.name: built.molecule})


def design_structure(items: ItemSet, design: str) -> None:
    """Check a ranking or direction set against its design (swap quads have the suite kind's own check).

    The designs are the module docstring's ("Designs"). A ranking anchor holds a ``compound`` and a ``protein``
    ranking on one pair: each is that pair (label 1), its one
    swap and at least one decoy of the ranking's kind (label 0), on one protein with distinct molecules (compound) or
    one molecule with distinct proteins (protein). A direction group is one item.

    Raises:
        KeyError: if a field of the design is missing.
        ValueError: if a unit breaks its design.
    """
    if design == "swap_quads":
        return
    missing = sorted(
        {_ANCHOR, "pair_label", *(("ranking", "candidate") if design == "ranking" else ())} - set(items.fields)
    )
    if missing:
        raise KeyError(f"item set {items.name!r} has no {missing} field (the {design} design)")
    table = pd.DataFrame({**items.objects, **{f: items.fields[f] for f in items.fields if "." not in f}})
    for anchor, unit in table.groupby(_ANCHOR, sort=False):
        ok = len(unit) == 1 and unit["pair_label"].iat[0] in (0, 1)
        if design == "ranking":
            pairs = set()
            ok = set(unit["ranking"]) == set(_RANKINGS)
            for ranking, rows in unit.groupby("ranking", sort=False):
                kinds = list(rows["candidate"])
                fixed, varied = (_PROTEIN, _MOLECULE) if ranking == "compound" else (_MOLECULE, _PROTEIN)
                pair = rows[rows["candidate"] == "pair"]
                pairs |= set(zip(pair[_PROTEIN], pair[_MOLECULE], strict=True))
                ok = (
                    ok
                    and kinds.count("pair") == kinds.count("swap") == 1
                    and kinds.count(_RANKINGS.get(str(ranking), "")) == len(kinds) - 2 >= 1
                    and list(rows["pair_label"]) == [int(k == "pair") for k in kinds]
                    and rows[fixed].nunique() == 1
                    and rows[varied].is_unique
                )
            ok = ok and len(pairs) == 1
        if not ok:
            raise ValueError(f"item set {items.name!r}: {_ANCHOR} {anchor!r} is not a unit of the {design} design")


@dataclass
class _Standing:
    """What a kept unit's objects are held to: the eligible objects per side, and the standing reasons' lists."""

    protein: Domain
    molecule: Domain
    proteins: dict[str, set[str]]  # side → the eligible proteins
    molecules: dict[str, set[str]]  # side → the eligible molecules
    pool_sequences: pd.Series  # the novel pool's domain: object → sequence
    min_length: int | None
    protein_lists: Mapping[str, frozenset[str]]  # reason → the novel pool's listed ids
    molecule_lists: dict[str, set[str]]  # reason → the listed molecules
    group: Callable[[str], str]
    strata: pd.DataFrame

    @property
    def reasons(self) -> tuple[str, ...]:
        """The standing reasons, in the order a manifest lists a unit's."""
        lists = (*self.protein_lists, *self.molecule_lists)
        return (MIN_LENGTH_REASON, *lists, LEFT_TRAINING_REASON, NOW_TRAINED_REASON, PARENT_REASON)

    def of(self, role: str, obj: str, side: str) -> list[str]:
        """Return the standing reasons the object carries at ``role`` on ``side`` (familiar or novel)."""
        out = []
        if role == _PROTEIN:
            trained = obj in self.proteins["familiar"]
            if side == "novel":
                sequence = self.pool_sequences.get(obj)
                if self.min_length is not None and isinstance(sequence, str) and len(sequence) < self.min_length:
                    out.append(MIN_LENGTH_REASON)
                out += [r for r, ids in self.protein_lists.items() if obj in ids]
        else:
            trained = obj in self.molecules["familiar"]
            out += [r for r, ids in self.molecule_lists.items() if obj in ids]
        if side == "familiar" and not trained:
            out.append(LEFT_TRAINING_REASON)
        if side == "novel" and trained:
            out.append(NOW_TRAINED_REASON)
        return out


def _standing(spec: CrossDomainItemSpec, retained: RetainedUnits, store: ArtifactStore, data_root: Path) -> _Standing:
    protein, molecule = open_domain(spec.protein, store), open_domain(spec.molecule, store)
    strata = molecule.table("strata")
    if strata is None or not {"stratum", _GROUP} <= set(strata.columns):
        raise ValueError(f"molecule domain {spec.molecule.data[:12]} has no strata table with stratum and {_GROUP}")
    gid = strata[_GROUP].dropna()
    groups = pd.Series([f"g{int(g)}" for g in gid], index=gid.index.astype(str))
    lists = {e.reason: set[str]().union(*(s.read(data_root) for s in e.ids)) for e in retained.excluded_molecules}
    listed = set[str]().union(*lists.values())
    familiar_p = set(training_objects(store, spec.familiar_proteins)) & set(protein.population())
    pool = eligible_pool(spec.novel_proteins, store, data_root)
    familiar_m = set(training_objects(store, spec.familiar_molecules)) & set(molecule.population())
    monitor = _monitor_groups(store, spec.exclude_molecules, molecule, groups)
    novel_m = set(_novel_molecules(spec, molecule, strata, groups, monitor)) - familiar_m
    return _Standing(
        protein,
        molecule,
        {"familiar": familiar_p, "novel": set(pool.eligible.index) - familiar_p},
        {"familiar": familiar_m - listed, "novel": novel_m - listed},
        pool.domain.objects()["value"],
        spec.novel_proteins.min_length,
        pool.named,
        lists,
        lambda c: str(groups.get(c, c)),
        strata,
    )


def _declared(
    spec: CrossDomainItemSpec, retained: RetainedUnits, data_root: Path
) -> dict[str, dict[str, dict[str, str]]]:
    """Return the declaration: design → cell → unit → reason, for the spec's design and the parents'.

    Raises:
        IntegrityError: if the file's sha256 differs from its citation.
        ValueError: if a column is missing, a row of the spec's design names a cell that is not retained, or a unit
            is removed twice.
    """
    columns = ["design", "cell", "unit", "reason"]
    table = pd.read_parquet(read_source(data_root, retained.excluded))
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"{retained.excluded.path}: no column {missing}")
    table = table[columns].astype(str)
    twice = table[table.duplicated(["design", "cell", "unit"])]
    if len(twice):
        raise ValueError(f"{retained.excluded.path}: a unit is removed twice, e.g. {list(twice.iloc[0][:3])}")
    unknown = sorted(set(table.loc[table["design"] == spec.design, "cell"]) - set(retained.within))
    if unknown:
        raise ValueError(
            f"{retained.excluded.path}: cells {unknown} are not among the retained cells {sorted(retained.within)}"
        )
    out: dict[str, dict[str, dict[str, str]]] = {d: {c: {} for c in spec.cells} for d in {spec.design, "swap_quads"}}
    for design, cell, unit, reason in table.itertuples(index=False):
        if design in out and cell in out[design]:
            out[design][cell][unit] = reason
    return out


def _parents(name: str, within: ItemSet, parent: ItemSet, design: str) -> dict[str, str]:
    """Return unit → its parent quad: the quad of ``parent`` whose positive is the unit's pair.

    Raises:
        ValueError: if a unit's pair is no positive of ``parent``.
    """
    positives = {
        (p, m): str(q)
        for p, m, q, y in zip(
            parent.objects[_PROTEIN],
            parent.objects[_MOLECULE],
            parent.fields[QUAD],
            parent.fields["pair_label"],
            strict=True,
        )
        if y == 1
    }
    out: dict[str, str] = {}
    lost = []
    kinds = within.fields["candidate"] if design == "ranking" else ("pair",) * len(within)
    rows = zip(within.fields[_ANCHOR], within.objects[_PROTEIN], within.objects[_MOLECULE], kinds, strict=True)
    for unit, p, m, kind in rows:
        if kind != "pair":
            continue
        if (p, m) not in positives:
            lost.append(str(unit))
        else:
            out[str(unit)] = positives[(p, m)]
    if lost:
        raise ValueError(
            f"{name}: {len(set(lost))} units have no parent quad in {parent.name!r}, e.g. {sorted(set(lost))[:_SHOWN]}"
        )
    return out


def _held(items: ItemSet, standing: _Standing) -> dict[str, int]:
    """Hold a cell's kept items to the domains now (module docstring) and return its units.

    Raises:
        ValueError: if a protein has another sequence than the pool's, or an item another unit or strata value than
            the set stores.
    """
    ids = pd.Index(items.objects[_PROTEIN])
    was, now = standing.pool_sequences.reindex(ids), standing.protein.objects()["value"].reindex(ids)
    changed = sorted(set(ids[(was != now).to_numpy() & was.notna().to_numpy()]))
    if changed:
        raise ValueError(f"{items.name}: proteins with another sequence than the pool's, e.g. {changed[:_SHOWN]}")
    molecules = pd.Index(items.objects[_MOLECULE])
    fresh: dict[str, list[object]] = {
        "protein_cluster": [str(c) for c in standing.protein.clusters().reindex(ids)],
        "molecule_group": [standing.group(m) for m in molecules],
    }
    for column in standing.strata.columns.drop(_GROUP):
        values = standing.strata[column].reindex(molecules)
        fresh[f"{column}.molecule"] = [None if pd.isna(v) else str(v) for v in values]
    domains = {standing.protein.name: standing.protein, standing.molecule.name: standing.molecule}
    clustered = with_clusters(items, domains).fields
    fresh |= {f: list(v) for f, v in clustered.items() if f.startswith("cluster.")}
    moved = sorted(f for f, v in fresh.items() if f not in items.fields or list(items.fields[f]) != v)
    if moved:
        raise ValueError(f"{items.name}: the fields {moved} differ from what the domains give now (the set's units)")
    return {
        "items": len(items),
        "proteins": int(ids.nunique()),
        "protein_clusters": len(set(items.fields["protein_cluster"])),
        "molecules": int(molecules.nunique()),
        "molecule_groups": len(set(items.fields["molecule_group"])),
    }


def _shown(cell: str, within: ItemSet, units: list[str], standing: _Standing) -> dict[str, dict[str, list[str]]]:
    """Return unit → standing reason → the members of the unit that carry it (the member reasons)."""
    shown: dict[str, dict[str, list[str]]] = {}
    side = dict(zip((_PROTEIN, _MOLECULE), CELLS[cell], strict=True))  # every item's objects are of the cell's sides
    for unit, p, m in zip(units, within.objects[_PROTEIN], within.objects[_MOLECULE], strict=True):
        for role, obj in ((_PROTEIN, p), (_MOLECULE, m)):
            for reason in standing.of(role, obj, side[role]):
                members = shown.setdefault(unit, {}).setdefault(reason, [])
                if obj not in members:
                    members.append(obj)
    return shown


def _follow(
    spec: CrossDomainItemSpec,
    cell: CellName,
    within: ItemSet,
    store: ArtifactStore,
    *,
    removed_quads: Mapping[str, str],
    shown: dict[str, dict[str, list[str]]],
) -> dict[str, str]:
    """Return unit → parent quad, and add :data:`PARENT_REASON` to ``shown`` for a unit whose parent is removed.

    Raises:
        ValueError: if a removed quad is not in the parent set, or a unit has no parent quad.
    """
    if spec.retained is None:  # the caller guarantees it; this narrows the type
        raise ValueError(f"{spec.name}: no retained units are declared")
    parent = load_item_set(store, ArtifactRef(spec.retained.parents[cell], ITEM_SET_KIND))
    stray = sorted(set(removed_quads) - {str(q) for q in parent.fields[QUAD]})
    if stray:
        raise ValueError(
            f"{spec.name}/{cell}: {len(stray)} removed quads are not in {parent.name!r}, e.g. {stray[:_SHOWN]}"
        )
    parents = _parents(f"{spec.name}/{cell}", within, parent, spec.design)
    for unit, quad in parents.items():
        if quad in removed_quads:
            shown.setdefault(unit, {})[PARENT_REASON] = [quad]
    return parents


def retained_cells(
    spec: CrossDomainItemSpec, store: ArtifactStore, data_root: Path, check: Callable[[ItemSet], object]
) -> dict[str, tuple[ItemSet, dict[str, Any]]]:
    """Return, per cell, the kept units of ``spec.retained`` and their metadata, uncommitted (module docstring).

    The metadata: ``evidence`` (the cell's counts), ``removed`` (unit → its reasons, the declared one first) and
    ``removed_members`` (unit → standing reason → the members that carry it).

    Raises:
        IntegrityError: if a table's sha256 differs from the spec.
        ValueError: if the declaration is not one the cells can keep (module docstring: a kept unit with an object
            that is no longer eligible or a removed parent, a standing reason the rows do not show, a unit removed
            twice or not in the set, a moved unit field or sequence, another draw than the ``within`` sets'), or a
            set breaks its design's structure.
    """
    retained = spec.retained
    if retained is None:
        raise ValueError(f"{spec.name}: no retained units are declared")
    standing = _standing(spec, retained, store, data_root)
    declared = _declared(spec, retained, data_root)
    field_, draw = _UNIT[spec.design], spec.model_dump(mode="json")
    out: dict[str, tuple[ItemSet, dict[str, Any]]] = {}
    for cell in spec.cells:
        ref = ArtifactRef(retained.within[cell], ITEM_SET_KIND)
        within = load_item_set(store, ref)
        check(within)
        design_structure(within, spec.design)
        built = store.manifest(ref).meta.get("spec", {})
        other = [f for f in _DRAW if built.get(f) != draw[f]]
        if other:
            raise ValueError(f"{spec.name}: {within.name!r} was drawn with another {other} than this spec states")
        gone = declared[spec.design][cell]
        units = [str(u) for u in within.fields[field_]]
        shown = _shown(cell, within, units, standing)
        parents: dict[str, str] = {}
        if spec.design != "swap_quads":
            parents = _follow(spec, cell, within, store, removed_quads=declared["swap_quads"][cell], shown=shown)
            orphaned = sorted(u for u in dict.fromkeys(units) if PARENT_REASON in shown.get(u, {}) and u not in gone)
            if orphaned:
                raise ValueError(
                    f"{spec.name}/{cell}: {len(orphaned)} kept units have a removed parent quad, e.g. "
                    f"{orphaned[:_SHOWN]}: the declaration must remove them ({PARENT_REASON})"
                )
        unshown = sorted((u, r) for u, r in gone.items() if r in standing.reasons and r not in shown.get(u, {}))
        if unshown:
            raise ValueError(
                f"{spec.name}/{cell}: {len(unshown)} units are declared with a reason their rows do not show, e.g. "
                f"{unshown[:_SHOWN]}"
            )
        ps, ms = CELLS[cell]
        members = {_PROTEIN: standing.proteins[ps], _MOLECULE: standing.molecules[ms]}
        kept = retain_quads(within, gone, {}, name=f"{spec.name}/{cell}", members=members, field=field_)
        check(kept.items)
        design_structure(kept.items, spec.design)
        removed = {
            u: [r, *(x for x in standing.reasons if x in shown.get(u, {}) and x != r)] for u, r in sorted(gone.items())
        }
        evidence = {
            "cell": cell,
            "design": spec.design,
            "within": ref.sha256,
            "units": {"within": len(set(units)), "retained": len(set(kept.items.fields[field_])), "removed": len(gone)},
            "removed": dict(sorted(Counter(r[0] for r in removed.values()).items())),
            "carried": dict(sorted(Counter(x for r in removed.values() for x in r).items())),
            "kept": _held(kept.items, standing),
        }
        if parents:
            evidence["parents"] = retained.parents[cell]
        members_of = {u: dict(shown[u]) for u in sorted(gone) if u in shown}
        out[cell] = (kept.items, {"evidence": evidence, "removed": removed, "removed_members": members_of})
    return out


def _build_retained(
    spec: CrossDomainItemSpec, store: ArtifactStore, data_root: Path, check: Callable[[ItemSet], object] | None
) -> dict[str, ArtifactRef]:
    """Commit the kept units of ``spec.retained``: ``items-<cell>``, ``proteins`` and ``molecules``."""
    if check is None:
        raise ValueError(f"{spec.name}: retained units are committed only with the suite kind's structure check")
    kept = retained_cells(spec, store, data_root, check)
    if not any(len(items) for items, _ in kept.values()):
        raise ValueError(f"{spec.name}: no cell keeps a unit")
    meta = {"spec": spec.model_dump(mode="json"), "spec_digest": spec.digest()}
    protein, molecule = open_domain(spec.protein, store), open_domain(spec.molecule, store)
    out: dict[str, ArtifactRef] = {}
    proteins: set[str] = set()
    molecules: set[str] = set()
    for cell, (items, extra) in kept.items():
        if not len(items):
            continue
        proteins |= set(items.objects[_PROTEIN])
        molecules |= set(items.objects[_MOLECULE])
        out[f"items-{cell}"] = commit_item_set(store, items, meta | extra)
    lists = {
        "proteins": object_list(f"{spec.name}/proteins", protein, tuple(sorted(proteins))),
        "molecules": object_list(f"{spec.name}/molecules", molecule, tuple(sorted(molecules))),
    }
    return out | {k: commit_item_set(store, v) for k, v in lists.items()}


def build_cross_domain_items(
    spec: CrossDomainItemSpec,
    store: ArtifactStore,
    data_root: Path,
    *,
    check: Callable[[ItemSet], object] | None = None,
) -> dict[str, ArtifactRef]:
    """Apply the rules (module docstring) and commit each built cell's item set and the object lists.

    Returns:
        ``items-<cell>`` per cell with at least one item (with the stage's ``--pin NAME``: ``NAME/items-<cell>``, the
        pin the suites cite), ``proteins`` and ``molecules`` (every built cell's objects: the extractions' inputs) and
        ``molecule_pool``. With ``retained``: ``items-<cell>``, ``proteins`` and ``molecules`` of the kept units.

    Args:
        spec: the design, drawn or (``retained``) kept of built cells.
        store: the artifact store.
        data_root: the data root the sources are under.
        check: the suite kind's structure check a retained subset is held to before it is committed
            (``evaluation.cross_domain``: ``swap_quads`` or ``pair_labels``, passed by the stage); unused by a draw.

    Raises:
        IntegrityError: if a source's sha256 differs from the spec.
        ValueError: as :func:`rhea_cells`; if no cell has a quad, or the fingerprints are not ECFP bits; with
            ``retained``, as :func:`retained_cells`, or if no ``check`` is given.
    """
    if spec.retained is not None:
        return _build_retained(spec, store, data_root, check)
    built = rhea_cells(spec, store, data_root)
    funnel = built.funnel
    if not any(c.quads for c in built.cells.values()):
        raise ValueError(f"{spec.name}: no cell has a swap quad ({funnel.evidence})")
    fingerprints = None if spec.fingerprints is None else _tanimoto(store, spec.fingerprints, funnel.pool)
    meta = {"spec": spec.model_dump(mode="json"), "spec_digest": spec.digest()}
    out: dict[str, ArtifactRef] = {}
    proteins: set[str] = set()
    molecules: set[str] = set()
    for name, cell in built.cells.items():
        if not cell.quads:
            continue
        if spec.design == "swap_quads":
            rows = _swap_quads(funnel, cell)
        elif spec.design == "direction":
            rows = _direction_rows(funnel, cell)
        else:
            if fingerprints is None:  # the validator guarantees them; this narrows the type
                raise ValueError("a ranking build needs fingerprints")
            rows = _ranking_rows(built, cell, spec, fingerprints)
        if not rows:  # a direction cell whose quad positives define no role
            continue
        items = _items(f"{spec.name}/{name}", rows, built)
        proteins |= set(items.objects[_PROTEIN])
        molecules |= set(items.objects[_MOLECULE])
        evidence = {**funnel.evidence, "cell": name, **cell.evidence, "items": len(items)}
        out[f"items-{name}"] = commit_item_set(store, items, meta | {"evidence": evidence})
    lists = {
        "proteins": object_list(f"{spec.name}/proteins", built.protein, tuple(sorted(proteins))),
        "molecules": object_list(f"{spec.name}/molecules", built.molecule, tuple(sorted(molecules))),
        "molecule_pool": object_list(f"{spec.name}/molecule_pool", built.molecule, tuple(funnel.pool)),
    }
    return out | {k: commit_item_set(store, v) for k, v in lists.items()}
