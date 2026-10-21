"""The BindingDB cross-domain 2x2: swap quads in four familiar/novel cells, and their molecule data.

One :class:`BindingDBItemSpec` builds the four cells' item sets and one domain-data artifact on the molecule schema
(``domains.tables``; it opens as the ``molecule`` domain). The label rules, the quad design and the E1 contract are the
BindingDB design's (the original item builder's); the cells and the exposure screen are the
cross-domain 2x2's (``build.xdom_screen``), and the novelty rule is stated below.

**Sources** (:class:`BindingDBSources`, each a path under the data root with its sha256): the BindingDB 202609 rows and
the ligands' chemistry keys, prepared outside this package (RDKit: InChIKey, form key, nearest
training-pool ECFP4 Tanimoto, the forms' fingerprints), the same keys for the molecule domain's objects, and the
Swiss-Prot table (protein names, genes, catalytic activity) the screen reads.

**Compounds.** A ligand's *form* is its form key (the first InChIKey block of its largest fragment, uncharged: salts,
charge states and stereoisomers are one form). Forms are the molecule unit, so familiarity is decided once per form,
from the same key on both sides:

- *familiar*: some object of the molecule training set (``familiar_molecules``) has the form;
- *novel*: no training object has it, no object of an ``exclude_molecules`` set (the in-job set) has it, and every
  SMILES of the form lies below ``novelty_max`` ECFP4 Tanimoto to every training object (the Tanimoto novelty rule,
  against the stated training pool);
- otherwise the form is in neither pool (counted).

**Proteins.** A target is its chain-1 Swiss-Prot primary id (the first one listed), an object of the protein domain:
*familiar* if it is an object of the protein training set (``familiar_proteins``), *novel* if it is in the pool
``novel_proteins`` states before its draw (``build.item_selection.eligible_pool``).

**Labels** (the design's §3.2). Per row, the first assay of ``assays`` (IC50, Ki, Kd) with a value; pAct = 9 −
log10(nM). A relational value is interval-censored: ``>X`` is a determinate inactive when pAct(X) < ``threshold``,
``<X`` a determinate active when pAct(X) ≥ ``threshold``, and any other censored row is dropped. Single-chain targets
only. Per (target, form): only the rows of the highest-priority assay present, their median pAct; a pair whose rows
disagree about the label is dropped; a positive has pAct ≥ ``threshold``.

**Exposure screen** (``build.xdom_screen``, keyed by form). A form's names are each BindingDB synonym of its
SMILES and the name of every molecule-domain object with the form, with and without a trailing charge; a trained
equation names the forms of the molecule-domain objects its ChEBI cross-references hold; a familiar form's training
objects are every training object with the form.

**Quads** (the design's §3.3-3.4, per cell; ``build.xdom_screen.draw_quads``, units protein clusters and forms, at
most ``per_unit_cap`` positives per unit). The admissible pairs: the two forms' ECFP4 Tanimoto below ``partner_max``,
neither swap measured active in BindingDB, neither swap stated. "Measured" counts every row of the target, single-chain
or not (complex rows included), keyed as a target is, by its first listed accession: a swap whose protein is a later
chain of a measured complex is not seen (the pinned source lists a second accession on 274 of its 2,897,051 rows). Edge
weights prefer swaps measured inactive (+0.1 each). The swapped cells are assumed negatives. A cell with no quad
is reported, not built.

**The molecule data**: one object per form of any cell's items. A familiar form's object is a training object
with the form (one with the ligand's exact InChIKey first, then the smallest id): its id, its SMILES and its name, so
the connector reads exactly what it was trained on. A novel form's object is ``bdb:<form key>``, its representative
SMILES (from the prepared keys) and that SMILES's BindingDB name. Split: ``train`` (familiar) or ``test``
(novel); cluster: the form. Its extra table ``forms`` holds each object's form, cell side and nearest-training
Tanimoto. No tasks, frames or recipes.

Each item carries ``quad``, ``pair_label``, the units ``protein_cluster`` and ``molecule_group``, ``cluster.<role>``,
and for a positive its ``pact``, ``assay``, ``measurements`` and ``sources`` (BindingDB's Curation/DataSource of every
contributing row: the license class); ``swap_measured_inactive`` marks a swap BindingDB measured inactive.

**Retained quads** (``BindingDBItemSpec.retained``, :class:`RetainedQuads`). With ``retained`` the spec draws
nothing: it keeps the quads of already built cells (``within``) that a declared table does not remove, as they are
(``build.retained_quads``: the same items, quad keys and fields; a quad leaves whole, nothing is paired again), and
writes the molecule data of what is kept. The spec's other fields then state what the kept quads are held to, and
the build refuses a kept quad that fails (the declaration removes it, with its reason):

- ``protein``, ``molecule``: the domains now (a corrected data version). A protein keeps the cluster the set stores
  and, where ``novel_proteins`` reads another data version, its sequence.
- ``familiar_proteins``, ``familiar_molecules``: a familiar protein or molecule is an object of these training sets;
  ``novel_proteins``: a novel protein is in this selection's pool (``build.item_selection.eligible_pool``, read with
  the data root so the selection may list excluded ids).
- A novel molecule is a novel object of the built molecule data whose form no object of ``familiar_molecules`` has
  (``sources.molecule_keys``); every kept molecule has the form its item's ``molecule_group`` names.
- ``exclude_molecules``, ``sources``, ``assays``, ``threshold``, ``novelty_max``, ``partner_max``, ``per_unit_cap``,
  ``seed``: the draw of the ``within`` sets, equal to the spec their manifests store (never applied again).

The declared ``replacements`` are of two kinds, by row. *Another object* (``to`` differs from ``molecule``): the
quad's items hold ``to`` in place of ``molecule``: a familiar positive whose assayed compound is another training
object of the same form (the form-identity reading). *Another string* (``to`` equals ``molecule``, a novel object):
the object's ``value`` and ``name`` in the molecule data are the row's: the assayed structure in place of the form's
representative. An object has one string, so every kept quad that holds such an object declares it, with the same
string, or the build is refused. Which compound a row assayed is the declaration's evidence (its other columns);
this module checks membership and structure, not chemistry.

The molecule data then holds one object per kept molecule: a familiar one with its string and name in ``molecule``
(the string it is trained on now), a novel one as built or as replaced. Each cell's manifest holds ``removed`` (quad
→ its reason) and ``replaced`` (quad → role → stored object → kept object); the molecule data's holds ``restrung``
(object → the string it replaces). The cells' structure is checked before anything is committed, by the check the
stage passes (``evaluation.cross_domain.swap_quads``: a layer above this one).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.build.item_selection import ItemSelectionSpec, eligible_pool
from latent_interconnect.build.retained_quads import RetainedItems, retain_quads
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
from latent_interconnect.domains.tables import DomainTables, SourceFile, commit_domain_data, read_source

__all__ = [
    "CELLS",
    "BindingDBItemSpec",
    "BindingDBSources",
    "RetainedQuads",
    "bindingdb_cells",
    "build_bindingdb_items",
]

_PROTEIN, _MOLECULE = PROTEIN, MOLECULE
_INACTIVE_BONUS = 0.1
_DRAW = ("exclude_molecules", "sources", "assays", "threshold", "novelty_max", "partner_max", "per_unit_cap", "seed")
"""The spec fields that state the draw: a retained subset repeats those of its ``within`` sets."""
_SHOWN = 5


class BindingDBSources(Config):
    """The outside sources (module docstring), each checked by sha256 before it is read.

    Attributes:
        rows: BindingDB's rows (``rows.parquet``).
        ligands: per ligand SMILES its ``formkey``, ``inchikey`` and ``nn_train`` (``ligands.parquet``).
        molecule_keys: per molecule-domain object its ``formkey`` and ``inchikey`` (``molecule_keys.parquet``).
        forms: per form its representative ``smiles`` and packed ECFP4 ``bits`` (``forms.parquet``).
        uniprot: Swiss-Prot's table (gzipped TSV: ``Entry``, ``Protein names``, ``Gene Names (primary)``,
            ``Catalytic activity``).
    """

    rows: SourceFile
    ligands: SourceFile
    molecule_keys: SourceFile
    forms: SourceFile
    uniprot: SourceFile


class RetainedQuads(Config):
    """The quads kept of already built cells (module docstring, "Retained quads").

    Attributes:
        within: cell → the built cell's item set.
        molecule_domain: the molecule data those sets were built with.
        excluded: the removed quads, a parquet table: ``cell``, ``quad``, ``reason`` (one row per quad; its other
            columns are the declaration's evidence).
        replacements: the replaced objects, a parquet table: ``cell``, ``quad``, ``molecule`` (the stored object),
            ``to`` (the object the quad holds instead, or the same object) and, for the same object, its ``value`` and
            ``name``.
    """

    within: dict[str, ArtifactDigest] = Field(min_length=1)
    molecule_domain: ArtifactDigest
    excluded: SourceFile
    replacements: SourceFile

    @model_validator(mode="after")
    def _cells(self) -> RetainedQuads:
        unknown = sorted(set(self.within) - set(CELLS))
        if unknown:
            raise ValueError(f"retained cells {unknown} are not among {sorted(CELLS)}")
        return self


class BindingDBItemSpec(StageConfig):
    """The BindingDB 2x2 (the module docstring gives the rules).

    Attributes:
        name: the item sets' name prefix (each cell's set is ``<name>/<cell>``).
        protein: the protein domain at a fixed data version.
        molecule: the molecule domain whose objects the familiar forms' objects are (names, SMILES).
        familiar_proteins: the protein training set (a ``training_set`` artifact: objects and their trained text).
        novel_proteins: the selection whose eligible pool, before its draw, is the novel proteins.
        familiar_molecules: the molecule training set (a ``training_set`` artifact).
        exclude_molecules: item sets whose molecules' forms are never novel (the in-job set).
        sources: the outside sources.
        assays: the assay priority (``ic50``, ``ki``, ``kd``).
        threshold: the pAct of a positive.
        novelty_max: the ECFP4 Tanimoto to the training pool a novel form's every SMILES stays below.
        partner_max: the ECFP4 Tanimoto a quad's two forms stay below.
        per_unit_cap: the most positives one protein cluster or one form enters.
        seed: the seed of the matching's jitter.
        retained: draw nothing: keep the quads of built cells that a declared table does not remove, held to the
            fields above (module docstring, "Retained quads"; added field).
    """

    name: str = Field(min_length=1)
    protein: DomainSpec
    molecule: DomainSpec
    familiar_proteins: ArtifactDigest
    novel_proteins: ItemSelectionSpec
    familiar_molecules: ArtifactDigest
    exclude_molecules: tuple[ArtifactDigest, ...] = ()
    sources: BindingDBSources
    assays: tuple[Literal["ic50", "ki", "kd"], ...] = Field(min_length=1)
    threshold: float
    novelty_max: float = Field(gt=0, le=1)
    partner_max: float = Field(gt=0, le=1)
    per_unit_cap: int = Field(ge=1)
    seed: int
    retained: RetainedQuads | None = added_field(default=None)


@dataclass
class _Built:
    """Every cell, the pair tables the items read, the forms' objects and the evidence."""

    cells: dict[str, Cell]
    inactive: set[tuple[str, str]]
    objects: pd.DataFrame  # form → object_id, value, name, side, nn_train
    protein_domain: Domain
    evidence: dict[str, Any]


def _labelled_rows(spec: BindingDBItemSpec, data_root: Path, proteins: pd.Index) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return the determinate rows (``acc``, ``form``, ``pact``, ``label``, ``assay``, ``single``, ``source``)."""
    rows = pd.read_parquet(read_source(data_root, spec.sources.rows))
    counts = {"rows": len(rows)}
    rows["acc"] = rows["uniprot"].str.split(r"[,; ]").str[0]
    rows = rows[rows["acc"].isin(proteins)]
    counts["target_in_protein_domain"] = len(rows)
    ligands = pd.read_parquet(read_source(data_root, spec.sources.ligands), columns=["smiles", "formkey"])
    rows = rows.assign(form=rows["smiles"].map(ligands.set_index("smiles")["formkey"]))
    rows = rows[rows["form"].notna()]
    counts["with_form"] = len(rows)
    choice = pd.Series("", index=rows.index)
    value = pd.Series(np.nan, index=rows.index)
    relation = pd.Series(0, index=rows.index)
    for assay in reversed(spec.assays):  # the first assay with a value wins
        text = rows[assay].str.strip()
        number = pd.to_numeric(text.str.replace(r"^[<>]=?", "", regex=True).str.replace(",", ""), errors="coerce")
        number = number.where(number > 0)
        has = number.notna()
        choice[has] = assay
        value[has] = number[has]
        relation[has] = np.where(text[has].str.startswith(">"), 1, np.where(text[has].str.startswith("<"), -1, 0))
    rows = rows.assign(assay=choice, pact=9 - np.log10(value), relation=relation)
    rows = rows[rows["assay"] != ""]
    counts["with_affinity"] = len(rows)
    counts["censored"] = int((rows["relation"] != 0).sum())
    t = spec.threshold
    determinate = (
        (rows["relation"] == 0)
        | ((rows["relation"] > 0) & (rows["pact"] < t))
        | ((rows["relation"] < 0) & (rows["pact"] >= t))
    )
    rows = rows[determinate]
    counts["indeterminate_dropped"] = counts["with_affinity"] - len(rows)
    rows = rows.assign(label=(rows["pact"] >= t).astype(int), single=rows["chains"] == "1")
    return rows[["acc", "form", "smiles", "pact", "label", "assay", "single", "source"]], counts


def _pairs(spec: BindingDBItemSpec, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Pool the single-chain rows per (target, form): the top assay's median pAct; label conflicts dropped."""
    rank = {a: n for n, a in enumerate(spec.assays)}
    x = rows[rows["single"]].assign(rank=lambda d: d["assay"].map(rank))
    x = x[x["rank"] == x.groupby(["acc", "form"])["rank"].transform("min")]
    pairs = (
        x.groupby(["acc", "form"], sort=True)
        .agg(
            pact=("pact", "median"),
            measurements=("pact", "size"),
            lmin=("label", "min"),
            lmax=("label", "max"),
            assay=("assay", "first"),
            sources=("source", lambda v: "|".join(sorted({str(s) for s in v if str(s)}))),
        )
        .reset_index()
    )
    conflict = pairs["lmin"] != pairs["lmax"]
    pairs = pairs[~conflict].assign(label=lambda d: (d["pact"] >= spec.threshold).astype(int))
    counts = {"single_chain_rows": len(x), "pairs_conflicting": int(conflict.sum()), "pairs": len(pairs)}
    return pairs.drop(columns=["lmin", "lmax"]).reset_index(drop=True), counts


def _forms(
    spec: BindingDBSources,
    data_root: Path,
    keys: pd.DataFrame,
    *,
    train: pd.Index,
    monitor: set[str],
    novelty_max: float,
) -> pd.DataFrame:
    """Per BindingDB form: its side (familiar, novel or neither), its training objects and its largest NN Tanimoto."""
    ligands = pd.read_parquet(
        read_source(data_root, spec.ligands), columns=["smiles", "inchikey", "formkey", "nn_train"]
    )
    ligands = ligands[ligands["formkey"].notna()]
    grouped = ligands.groupby("formkey")["nn_train"]
    nn = grouped.max().where(~ligands["nn_train"].isna().groupby(ligands["formkey"]).any())
    trained = keys[keys["object_id"].isin(train) & keys["formkey"].notna()]
    members = trained.groupby("formkey")["object_id"].agg(lambda v: tuple(sorted(v)))
    monitor_forms = set(keys.loc[keys["object_id"].isin(monitor), "formkey"].dropna())
    out = pd.DataFrame({"nn_train": nn})
    out["training_objects"] = members.reindex(out.index)
    familiar = out["training_objects"].notna()
    novel = ~familiar & ~out.index.isin(monitor_forms) & (out["nn_train"] < novelty_max)
    out["side"] = np.where(familiar, "familiar", np.where(novel, "novel", "neither"))
    out["injob"] = out.index.isin(monitor_forms)
    return out


def _names(data_root: Path, spec: BindingDBItemSpec, keys: pd.DataFrame, forms: set[str]) -> dict[str, set[str]]:
    """Return form → its names: every BindingDB synonym of its SMILES and every molecule-domain name with the form."""
    raw = pd.read_parquet(read_source(data_root, spec.sources.rows), columns=["smiles", "ligand_name"])
    ligands = pd.read_parquet(read_source(data_root, spec.sources.ligands), columns=["smiles", "formkey"])
    raw = raw.drop_duplicates().merge(ligands, on="smiles")
    raw = raw[raw["formkey"].isin(forms)]
    out: dict[str, set[str]] = defaultdict(set)
    for form, name in zip(raw["formkey"], raw["ligand_name"], strict=True):
        out[form] |= {n.strip() for n in str(name).split("::") if n.strip()}
    chebi = keys[keys["formkey"].isin(forms)]
    for form, name in zip(chebi["formkey"], chebi["name"], strict=True):
        out[form] |= name_variants(name)
    return out


def _bits(data_root: Path, spec: BindingDBItemSpec, forms: set[str]) -> dict[str, bytes]:
    """Return form → its representative's packed ECFP4 bits."""
    table = pd.read_parquet(read_source(data_root, spec.sources.forms))
    table = table[table["formkey"].isin(forms)]
    return dict(zip(table["formkey"], table["bits"], strict=True))


def _tanimoto(packed: list[bytes]) -> np.ndarray:
    """Return the Tanimoto matrix of packed fingerprints (exact: integer counts)."""
    x = np.unpackbits(np.frombuffer(b"".join(packed), np.uint8).reshape(len(packed), -1), axis=1).astype(np.float32)
    inter = (x @ x.T).astype(np.float64)
    count = x.sum(1).astype(np.float64)
    return inter / np.maximum(count[:, None] + count[None, :] - inter, 1.0)


def _cell_quads(
    pos: pd.DataFrame,
    spec: BindingDBItemSpec,
    *,
    bits: Mapping[str, bytes],
    active: set[tuple[str, str]],
    inactive: set[tuple[str, str]],
    screen: ExposureScreen,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    blocked = {"partner_tanimoto": 0, "swap_measured_active": 0, "swap_stated": 0}
    p, f = pos[_PROTEIN].to_numpy(), pos[_MOLECULE].to_numpy()

    def weigh(members: list[int]) -> Iterator[tuple[int, int, float]]:
        sim = _tanimoto([bits[f[k]] for k in members])
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if sim[a, b] >= spec.partner_max:
                    blocked["partner_tanimoto"] += 1
                    continue
                swaps = ((p[i], f[j]), (p[j], f[i]))
                if any(s in active for s in swaps):
                    blocked["swap_measured_active"] += 1
                    continue
                if any(screen.stated(*s) for s in swaps):
                    blocked["swap_stated"] += 1
                    continue
                yield a, b, 1.0 + _INACTIVE_BONUS * sum(s in inactive for s in swaps)

    return draw_quads(pos, spec.per_unit_cap, spec.seed, weigh), blocked


def bindingdb_cells(spec: BindingDBItemSpec, store: ArtifactStore, data_root: Path) -> _Built:
    """Apply the rules (module docstring) and return the cells, uncommitted.

    Raises:
        IntegrityError: if a source's sha256 differs from the spec.
        ValueError: if a protein is both familiar and novel.
    """
    protein = open_domain(spec.protein, store)
    familiar_p = set(training_objects(store, spec.familiar_proteins)) & set(protein.population())
    novel_p = set(eligible_pool(spec.novel_proteins, store).eligible.index)
    if familiar_p & novel_p:
        raise ValueError(f"proteins both familiar and novel, e.g. {sorted(familiar_p & novel_p)[:3]}")
    molecule = open_domain(spec.molecule, store)
    keys = pd.read_parquet(read_source(data_root, spec.sources.molecule_keys))
    keys = keys.assign(name=keys["object_id"].map(molecule.objects()["name"]))
    monitor: set[str] = set()
    for digest in spec.exclude_molecules:
        items = load_item_set(store, ArtifactRef(digest, ITEM_SET_KIND))
        monitor |= {o for r, d in items.roles.items() if d == molecule.name for o in items.objects[r]}
    train_m = training_objects(store, spec.familiar_molecules)
    forms = _forms(spec.sources, data_root, keys, train=train_m, monitor=monitor, novelty_max=spec.novelty_max)
    rows, row_counts = _labelled_rows(spec, data_root, protein.objects().index)
    evidence: dict[str, Any] = dict(row_counts)
    measured = rows.groupby(["acc", "form"])["label"].max()
    active = set(measured.index[measured.to_numpy() == 1])
    inactive = set(measured.index[measured.to_numpy() == 0])
    pairs, counts = _pairs(spec, rows)
    evidence |= counts
    pairs = pairs.rename(columns={"acc": _PROTEIN, "form": _MOLECULE})
    pairs["p_side"] = np.where(
        pairs[_PROTEIN].isin(familiar_p), "familiar", np.where(pairs[_PROTEIN].isin(novel_p), "novel", "neither")
    )
    pairs["m_side"] = forms["side"].reindex(pairs[_MOLECULE]).fillna("neither").to_numpy()
    positives = pairs[(pairs["label"] == 1) & (pairs["p_side"] != "neither") & (pairs["m_side"] != "neither")]
    evidence["pools"] = {
        "familiar_proteins": len(familiar_p),
        "novel_proteins": len(novel_p),
        "forms": {str(k): int(v) for k, v in forms["side"].value_counts().items()},
        "forms_injob": int(forms["injob"].sum()),
        "positive_pairs_by_side": {
            f"{a}P-{b}M": int(((pairs["label"] == 1) & (pairs["p_side"] == a) & (pairs["m_side"] == b)).sum())
            for a in ("familiar", "novel", "neither")
            for b in ("familiar", "novel", "neither")
        },
    }
    used = set(positives[_MOLECULE])
    chebi_forms = dict(zip(keys["object_id"], keys["formkey"], strict=True))
    chebi_forms = {k: v for k, v in chebi_forms.items() if isinstance(v, str) and v in used}
    familiar_forms = forms[forms["side"] == "familiar"]
    screen = ExposureScreen(
        store,
        protein=protein,
        familiar_proteins=spec.familiar_proteins,
        familiar_molecules=spec.familiar_molecules,
        uniprot=read_source(data_root, spec.sources.uniprot),
        candidates=positives,
        names=_names(data_root, spec, keys, used),
        chebi_keys=chebi_forms,
        trained_objects=dict(zip(familiar_forms.index, familiar_forms["training_objects"], strict=True)),
    )
    evidence["screen"] = screen.evidence()
    bits = _bits(data_root, spec, used)
    clusters = protein.clusters()
    cells: dict[str, Cell] = {}
    for name, (ps, ms) in CELLS.items():
        pos = positives[(positives["p_side"] == ps) & (positives["m_side"] == ms)].copy()
        before = {
            "positives": len(pos),
            "proteins": int(pos[_PROTEIN].nunique()),
            "forms": int(pos[_MOLECULE].nunique()),
        }
        kept, removed = screen_positives(pos, screen)
        kept["cluster"] = clusters.reindex(kept[_PROTEIN]).astype(str).to_numpy()
        kept["group"] = "f:" + kept[_MOLECULE]
        quads, blocked = (
            _cell_quads(kept, spec, bits=bits, active=active, inactive=inactive, screen=screen)
            if len(kept)
            else ([], {})
        )
        idx = sorted({k for q in quads for k in q})
        q = kept.iloc[idx]
        cells[name] = Cell(
            kept,
            quads,
            {
                "before_screen": before,
                "removed_by_screen": removed,
                "after_screen": {
                    "positives": len(kept),
                    "proteins": int(kept[_PROTEIN].nunique()),
                    "protein_clusters": int(kept["cluster"].nunique()),
                    "forms": int(kept[_MOLECULE].nunique()),
                },
                "swap_pairings_blocked": blocked,
                "quads": len(quads),
                "quad_units": {
                    "proteins": int(q[_PROTEIN].nunique()),
                    "protein_clusters": int(q["cluster"].nunique()),
                    "forms": int(q[_MOLECULE].nunique()),
                },
                "quads_with_a_swap_measured_inactive": sum(
                    any(
                        s in inactive
                        for s in (
                            (kept[_PROTEIN].iat[i], kept[_MOLECULE].iat[j]),
                            (kept[_PROTEIN].iat[j], kept[_MOLECULE].iat[i]),
                        )
                    )
                    for i, j in quads
                ),
            },
        )
    objects = _objects(data_root, spec, keys, forms, {f for c in cells.values() for f in _quad_forms(c)})
    return _Built(cells, inactive, objects, protein, evidence)


def _quad_forms(cell: Cell) -> set[str]:
    return {cell.positives[_MOLECULE].iat[k] for q in cell.quads for k in q}


def _objects(
    data_root: Path, spec: BindingDBItemSpec, keys: pd.DataFrame, forms: pd.DataFrame, used: set[str]
) -> pd.DataFrame:
    """Return form → its object (``object_id``, ``value``, ``name``) in the molecule data (module docstring)."""
    reps = pd.read_parquet(read_source(data_root, spec.sources.forms), columns=["formkey", "smiles"]).set_index(
        "formkey"
    )
    raw = pd.read_parquet(read_source(data_root, spec.sources.rows), columns=["smiles", "ligand_name"])
    named = raw["ligand_name"].notna() & (raw["ligand_name"] != "")  # never a name "nan"
    raw = raw[named].drop_duplicates("smiles").set_index("smiles")["ligand_name"]
    ligand_keys = pd.read_parquet(
        read_source(data_root, spec.sources.ligands), columns=["smiles", "inchikey", "formkey"]
    )
    exact = ligand_keys.groupby("formkey")["inchikey"].agg(set)
    by_id = keys.set_index("object_id")
    out = []
    for form in sorted(used):
        if forms.at[form, "side"] == "familiar":
            members = forms.at[form, "training_objects"]
            oid = min(members, key=lambda o: (by_id.at[o, "inchikey"] not in exact.get(form, set()), o))
            out.append((form, oid, by_id.at[oid, "value"], by_id.at[oid, "name"], "familiar"))
        else:
            smiles = reps.at[form, "smiles"]
            out.append((form, f"bdb:{form}", smiles, str(raw.get(smiles, "")), "novel"))
    frame = pd.DataFrame(out, columns=["form", "object_id", "value", "name", "side"]).set_index("form")
    frame["nn_train"] = forms["nn_train"].reindex(frame.index)
    return frame


def _domain_tables(objects: pd.DataFrame) -> DomainTables:
    ids = pd.Index(objects["object_id"], name="object_id")
    table = pd.DataFrame(
        {"value": objects["value"].to_numpy(), "name": objects["name"].fillna("").to_numpy()}, index=ids
    )
    return DomainTables(
        domain=_MOLECULE,
        objects=table,
        split=pd.Series(np.where(objects["side"] == "familiar", "train", "test"), index=ids),
        clusters=pd.Series(("f:" + objects.index).to_numpy(), index=ids),
        labels=pd.DataFrame(index=ids),
        fields=pd.DataFrame(index=ids),
        tasks=(),
        frames=(),
        recipes=(),
        extra={
            "forms": pd.DataFrame(
                {
                    "formkey": objects.index.to_numpy(),
                    "side": objects["side"].to_numpy(),
                    "nn_train": objects["nn_train"].to_numpy(),
                },
                index=ids,
            )
        },
    )


def _items(name: str, cell: Cell, objects: pd.DataFrame, inactive: set[tuple[str, str]]) -> ItemSet:
    pos = cell.positives
    oid = objects["object_id"]
    rows: list[dict[str, object]] = []
    for q, (i, j) in enumerate(cell.quads):
        for a, b, y in ((i, i, 1), (j, j, 1), (i, j, 0), (j, i, 0)):
            p, f = pos[_PROTEIN].iat[a], pos[_MOLECULE].iat[b]
            positive = pos.iloc[a] if y else None
            rows.append(
                {
                    _PROTEIN: p,
                    _MOLECULE: oid[f],
                    "quad": f"q{q:04d}",
                    "pair_label": y,
                    "protein_cluster": pos["cluster"].iat[a],
                    "molecule_group": pos["group"].iat[b],
                    "pact": None if positive is None else round(float(positive["pact"]), 4),
                    "assay": None if positive is None else str(positive["assay"]),
                    "measurements": None if positive is None else int(positive["measurements"]),
                    "sources": None if positive is None else str(positive["sources"]),
                    "swap_measured_inactive": None if y else (p, f) in inactive,
                }
            )
    table = pd.DataFrame(rows)
    fields = {c: tuple(table[c]) for c in table.columns if c not in (_PROTEIN, _MOLECULE)}
    return ItemSet(
        name, {_PROTEIN: _PROTEIN, _MOLECULE: _MOLECULE}, {r: tuple(table[r]) for r in (_PROTEIN, _MOLECULE)}, fields
    )


@dataclass
class _Retained:
    """The kept quads of every cell, the kept molecules' objects and the strings replaced."""

    cells: dict[str, RetainedItems]
    # one row per object, indexed by its form: object_id, value, name, side, nn_train. The index repeats where a
    # form holds two objects (a positive re-pointed to another training object beside one that keeps the first)
    objects: pd.DataFrame
    restrung: dict[str, str]  # object → the string it replaces
    protein_domain: Domain
    evidence: dict[str, dict[str, Any]]


@dataclass
class _Sides:
    """What a kept quad's objects are held to: the eligible objects per side, and every molecule's form."""

    protein: Domain
    molecule: Domain
    built_with: Domain
    forms: pd.DataFrame  # the built molecule data's table: formkey, side, nn_train
    proteins: dict[str, set[str]]  # side → the eligible proteins
    molecules: dict[str, set[str]]  # side → the eligible molecules
    form: pd.Series  # object → its form (training objects by the source's keys, built objects by ``forms``)
    pool_sequences: pd.Series  # the novel pool's domain: object → sequence


def _sides(spec: BindingDBItemSpec, retained: RetainedQuads, store: ArtifactStore, data_root: Path) -> _Sides:
    protein = open_domain(spec.protein, store)
    molecule = open_domain(spec.molecule, store)
    built_with = open_domain(DomainSpec(name=_MOLECULE, data=retained.molecule_domain), store)
    forms = built_with.table("forms")
    if forms is None:
        raise ValueError(f"{spec.name}: the molecule data {retained.molecule_domain[:12]} has no forms table")
    familiar_p = set(training_objects(store, spec.familiar_proteins)) & set(protein.population())
    pool = eligible_pool(spec.novel_proteins, store, data_root)
    novel_p = set(pool.eligible.index)
    if familiar_p & novel_p:
        raise ValueError(f"proteins both familiar and novel, e.g. {sorted(familiar_p & novel_p)[:3]}")
    familiar_m = set(training_objects(store, spec.familiar_molecules)) & set(molecule.objects().index)
    keys = pd.read_parquet(read_source(data_root, spec.sources.molecule_keys)).set_index("object_id")["formkey"]
    trained = keys.reindex(sorted(familiar_m))
    novel_m = set(forms.index[(forms["side"] == "novel") & ~forms["formkey"].isin(set(trained.dropna()))])
    return _Sides(
        protein,
        molecule,
        built_with,
        forms,
        {"familiar": familiar_p, "novel": novel_p},
        {"familiar": familiar_m, "novel": novel_m},
        pd.concat([trained, forms["formkey"][~forms.index.isin(familiar_m)]]),
        pool.domain.objects()["value"],
    )


def _declared(
    data_root: Path, source: SourceFile, columns: tuple[str, ...], cells: Mapping[str, object]
) -> pd.DataFrame:
    """Return a declaration table's ``columns`` as text.

    Raises:
        IntegrityError: if the file's sha256 differs from its citation.
        ValueError: if a column is missing or a row names a cell that is not retained.
    """
    table = pd.read_parquet(read_source(data_root, source))
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"{source.path}: no column {missing}")
    table = table[list(columns)].astype(str)
    unknown = sorted(set(table["cell"]) - set(cells))
    if unknown:
        raise ValueError(f"{source.path}: cells {unknown} are not among the retained cells {sorted(cells)}")
    return table


def _declarations(
    retained: RetainedQuads, data_root: Path, novel: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the removed quads, the replacements by another object, those by another string, and object → string.

    Raises:
        ValueError: if a quad is removed twice, a row that brings another object also brings a string, or an object's
            string is replaced by none, by two, or is not a novel object's.
    """
    excluded = _declared(data_root, retained.excluded, ("cell", "quad", "reason"), retained.within)
    if excluded.duplicated(["cell", "quad"]).any():
        twice = excluded[excluded.duplicated(["cell", "quad"])].iloc[0]
        raise ValueError(f"{retained.excluded.path}: a quad is removed twice, e.g. {twice['cell']} {twice['quad']}")
    columns = ("cell", "quad", "molecule", "to", "value", "name")
    replacements = _declared(data_root, retained.replacements, columns, retained.within)
    same = replacements["molecule"] == replacements["to"]
    moved, strung = replacements[~same], replacements[same]
    strings = strung.groupby("molecule")[["value", "name"]].agg(lambda v: tuple(sorted(set(v))))
    bad = sorted(m for m, v in strings["value"].items() if len(v) != 1 or not v[0] or m not in novel)
    if bad or (moved["value"] != "").any():
        raise ValueError(
            f"{retained.replacements.path}: a string replaces one novel object's string with one string, and another "
            f"object brings its own, e.g. {bad[:_SHOWN]}"
        )
    return excluded, moved, strung, strings.map(lambda v: v[0])


def _held(spec: BindingDBItemSpec, items: ItemSet, sides: _Sides) -> dict[str, int]:
    """Hold a cell's kept items to the domains now (module docstring) and return its units.

    Raises:
        ValueError: if a protein has another cluster or sequence, a molecule another form, or a positive is below
            the threshold.
    """
    ids = pd.Index(items.objects[_PROTEIN])
    if list(sides.protein.clusters().reindex(ids).astype(str)) != [str(c) for c in items.fields["protein_cluster"]]:
        raise ValueError(f"{items.name}: a protein has another cluster in the domain than the set stores")
    was, now = sides.pool_sequences.reindex(ids), sides.protein.objects()["value"].reindex(ids)
    changed = sorted(set(ids[(was != now).to_numpy() & was.notna().to_numpy()]))
    if changed:
        raise ValueError(f"{items.name}: proteins with another sequence than the pool's, e.g. {changed[:_SHOWN]}")
    groups = "f:" + sides.form.reindex(pd.Index(items.objects[_MOLECULE])).astype(str)
    if list(groups) != [str(g) for g in items.fields["molecule_group"]]:
        raise ValueError(f"{items.name}: a molecule has another form than its item's molecule_group")
    pact = [p for p, y in zip(items.fields["pact"], items.fields["pair_label"], strict=True) if y == 1]
    if any(not isinstance(p, int | float) or p < spec.threshold for p in pact):
        raise ValueError(f"{items.name}: a positive is below the threshold {spec.threshold}")
    return {
        "proteins": int(ids.nunique()),
        "protein_clusters": len(set(items.fields["protein_cluster"])),
        "forms": len(set(items.fields["molecule_group"])),
    }


def _kept_objects(cells: Mapping[str, RetainedItems], sides: _Sides, strings: pd.DataFrame) -> pd.DataFrame:
    """Return the kept molecules' objects, indexed by form (``object_id``, ``value``, ``name``, ``side``)."""
    trained, built = sides.molecule.objects(), sides.built_with.objects()
    out = []
    for oid in sorted({m for kept in cells.values() for m in kept.items.objects[_MOLECULE]}):
        if oid in sides.molecules["familiar"]:
            out.append((sides.form[oid], oid, trained.at[oid, "value"], trained.at[oid, "name"], "familiar"))
        elif oid in strings.index:
            out.append((sides.form[oid], oid, strings.at[oid, "value"], strings.at[oid, "name"], "novel"))
        else:
            out.append((sides.form[oid], oid, built.at[oid, "value"], built.at[oid, "name"], "novel"))
    objects = pd.DataFrame(out, columns=["form", "object_id", "value", "name", "side"]).set_index("form")
    nn = sides.forms.drop_duplicates("formkey").set_index("formkey")["nn_train"]
    objects["nn_train"] = nn.reindex(objects.index).to_numpy()
    return objects


def retained_cells(
    spec: BindingDBItemSpec, store: ArtifactStore, data_root: Path, check: Callable[[ItemSet], object]
) -> _Retained:
    """Return the kept quads of ``spec.retained`` and their molecule objects, uncommitted (module docstring).

    Raises:
        IntegrityError: if a table's sha256 differs from the spec.
        ValueError: if the declaration is not one the cells can keep (module docstring: a kept quad with an object
            that is no longer eligible, a quad removed twice, a replacement of the wrong kind or an object with two
            strings, a moved cluster or form, another draw than the ``within`` sets'), or ``check`` refuses a set.
    """
    retained = spec.retained
    if retained is None:
        raise ValueError(f"{spec.name}: no retained quads are declared")
    sides = _sides(spec, retained, store, data_root)
    excluded, moved, strung, strings = _declarations(retained, data_root, sides.molecules["novel"])
    cells: dict[str, RetainedItems] = {}
    evidence: dict[str, dict[str, Any]] = {}
    draw = spec.model_dump(mode="json")
    for cell, (ps, ms) in CELLS.items():
        if cell not in retained.within:
            continue
        ref = ArtifactRef(retained.within[cell], ITEM_SET_KIND)
        within = load_item_set(store, ref)
        check(within)
        built = store.manifest(ref).meta.get("spec", {})
        other = [f for f in _DRAW if built.get(f) != draw[f]]
        if other:
            raise ValueError(f"{spec.name}: {within.name!r} was drawn with another {other} than this spec states")
        gone, to, new = (t[t["cell"] == cell] for t in (excluded, moved, strung))
        kept = retain_quads(
            within,
            dict(zip(gone["quad"], gone["reason"], strict=True)),
            {(q, _MOLECULE, m): t for q, m, t in zip(to["quad"], to["molecule"], to["to"], strict=True)},
            name=f"{spec.name}/{cell}",
            members={_PROTEIN: sides.proteins[ps], _MOLECULE: sides.molecules[ms]},
        )
        check(kept.items)
        declared = set(zip(new["quad"], new["molecule"], strict=True))
        held = {
            (str(q), m)
            for q, m in zip(kept.items.fields["quad"], kept.items.objects[_MOLECULE], strict=True)
            if m in strings.index
        }
        if declared != held:
            raise ValueError(
                f"{spec.name}/{cell}: an object has one string: exactly the kept quads that hold a replaced object "
                f"declare it, e.g. {sorted(declared ^ held)[:_SHOWN]}"
            )
        cells[cell] = kept
        evidence[cell] = {
            "cell": cell,
            "within": ref.sha256,
            "quads": {"within": len(within) // 4, "retained": len(kept.items) // 4},
            "removed": dict(sorted(Counter(r for reasons in kept.removed.values() for r in reasons).items())),
            "replaced": {
                "objects": sum(len(m) for roles in kept.replaced.values() for m in roles.values()),
                "strings": int(new["molecule"].nunique()),
            },
            "units": _held(spec, kept.items, sides),
        }
    objects = _kept_objects(cells, sides, strings)
    built = sides.built_with.objects()["value"]
    restrung = {o: str(built[o]) for o in objects["object_id"] if o in strings.index}
    return _Retained(cells, objects, restrung, sides.protein, evidence)


def _build_retained(
    spec: BindingDBItemSpec, store: ArtifactStore, data_root: Path, check: Callable[[ItemSet], object] | None
) -> dict[str, ArtifactRef]:
    """Commit the kept quads of ``spec.retained``: the outputs of :func:`build_bindingdb_items`."""
    if check is None:
        raise ValueError(f"{spec.name}: retained quads are committed only with the swap-quad structure check")
    kept = retained_cells(spec, store, data_root, check)
    if not any(len(c.items) for c in kept.cells.values()):
        raise ValueError(f"{spec.name}: no cell keeps a swap quad ({kept.evidence})")
    meta = {"spec": spec.model_dump(mode="json"), "spec_digest": spec.digest()}
    domain_ref = commit_domain_data(store, _domain_tables(kept.objects), meta | {"restrung": kept.restrung})
    domain = open_domain(DomainSpec(name=_MOLECULE, data=domain_ref.sha256), store)
    out = {"molecule_domain": domain_ref}
    domains = {_PROTEIN: kept.protein_domain, _MOLECULE: domain}
    proteins: set[str] = set()
    for name, cell in kept.cells.items():
        if not len(cell.items):
            continue
        items = with_clusters(cell.items, domains)
        proteins |= set(items.objects[_PROTEIN])
        extra = {"evidence": kept.evidence[name], "removed": cell.removed, "replaced": cell.replaced}
        out[f"items-{name}"] = commit_item_set(store, items, meta | extra)
    out["proteins"] = commit_item_set(
        store, object_list(f"{spec.name}/proteins", kept.protein_domain, tuple(sorted(proteins)))
    )
    out["molecules"] = commit_item_set(
        store, object_list(f"{spec.name}/molecules", domain, tuple(sorted(kept.objects["object_id"])))
    )
    return out


def build_bindingdb_items(
    spec: BindingDBItemSpec,
    store: ArtifactStore,
    data_root: Path,
    *,
    check: Callable[[ItemSet], object] | None = None,
) -> dict[str, ArtifactRef]:
    """Apply the rules (module docstring) and commit the molecule data, each supported cell's set and the lists.

    Args:
        spec: the 2x2, drawn or (``retained``) kept of built cells.
        store: the artifact store.
        data_root: the data root the sources are under.
        check: the swap-quad structure check a retained subset is held to before it is committed
            (``evaluation.cross_domain.swap_quads`` on the two roles, passed by the stage); unused by a draw.

    Returns:
        ``molecule_domain`` (the molecule data the suites' molecule domain cites), ``items-<cell>`` per cell with at
        least one quad (with the stage's ``--pin NAME``: ``NAME/items-<cell>``), ``proteins`` and ``molecules`` (every
        cell's objects: the extractions' inputs).

    Raises:
        IntegrityError: if a source's sha256 differs from the spec.
        ValueError: if a protein is both familiar and novel, or no cell has a quad; with ``retained``, as
            :func:`retained_cells`, or if no ``check`` is given.
    """
    if spec.retained is not None:
        return _build_retained(spec, store, data_root, check)
    built = bindingdb_cells(spec, store, data_root)
    if not any(c.quads for c in built.cells.values()):
        raise ValueError(f"{spec.name}: no cell has a swap quad ({built.evidence})")
    meta = {"spec": spec.model_dump(mode="json"), "spec_digest": spec.digest()}
    domain_ref = commit_domain_data(store, _domain_tables(built.objects), meta)
    domain = open_domain(DomainSpec(name=_MOLECULE, data=domain_ref.sha256), store)
    out = {"molecule_domain": domain_ref}
    domains = {_PROTEIN: built.protein_domain, _MOLECULE: domain}
    proteins: set[str] = set()
    for name, cell in built.cells.items():
        if not cell.quads:
            continue
        items = with_clusters(_items(f"{spec.name}/{name}", cell, built.objects, built.inactive), domains)
        proteins |= set(items.objects[_PROTEIN])
        evidence = {**built.evidence, "cell": name, **cell.evidence}
        out[f"items-{name}"] = commit_item_set(store, items, meta | {"evidence": evidence})
    out["proteins"] = commit_item_set(
        store, object_list(f"{spec.name}/proteins", built.protein_domain, tuple(sorted(proteins)))
    )
    out["molecules"] = commit_item_set(
        store, object_list(f"{spec.name}/molecules", domain, tuple(sorted(built.objects["object_id"])))
    )
    return out
