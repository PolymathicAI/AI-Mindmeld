"""The protein domain (SwissProt), and the spec its domain-data artifact is built from.

:class:`ProteinDataSpec` (registered in ``DOMAIN_DATA`` as ``protein``) turns the source tables it names into a
domain-data artifact (``domains.tables``). Every source is cited by a path under the data root (``Paths.data``) and
its sha256, checked before reading. The paper's procedure is kept:

- the object's ``value`` is its amino-acid sequence and its ``name`` the ``protein_names`` text before the first
  ``" ("``, which the paper's procedure took for
  the UniProt recommended name;
- the split is the homology split's ``split`` column; objects it marks otherwise (``discarded``) are in no split;
- clusters are the MMseqs2 clustering's representatives;
- labels follow the paper's label keys: a keyword label is the keyword's presence in the
  ``;``-separated ``keywords``; a lineage label is a substring match on ``lineage``; a length label is
  ``length > value``. Missing text counts as absent, so every label is a bool;
- the task catalog, the frames (the question wording) and the capture recipes (``name``, ``caption``,
  ``name_described``) are data in the spec (the domain stage's config).

A spec may add the original runs' protein variants (:class:`VariantSource`; both shipped protein domains do): each
single substitution is an object of this domain whose ``value`` is its parent's sequence with the substitution applied
(the parent residue is checked), with no name, its parent's cluster, and its original variant split when that is
``train`` / ``val`` / ``test`` (which must be its parent's split; ``heldpos`` and the excluded rows are in no split).
Their free text (the effect text) is a field, so it passes the zero-shot screen like every training text, and their
Yes/No answers are labels (unknown on proteins, as protein labels are on variants; a data version with variants stores
labels as nullable booleans). The ``variants`` extra table (:meth:`ProteinDomain.variants`) links each variant to its
parent. Population draws of proteins exclude the variant objects (:meth:`ProteinDomain.population`).

The relational families' EC-number and pathway sets and the keyword lists the scaffold reads
are parsed from the screened ``ec``, ``pathway`` and ``keywords`` texts by :func:`parse_ec_set`,
:func:`parse_pathway_set` and :func:`parse_keywords`, the one home of these rules (the original
``parse_ecs`` and ``parse_pathways``).

A spec may correct four things a data audit found (``protein-domain`` does), each by a field that is off by default,
so an older spec's digest and tables are unchanged:

- ``entries`` cites UniProt's flat file of the same release (``domains.uniprot_entries``). The ``name`` is then the
  entry's recommended-name field (the cut text is wrong where a name holds ``" ("`` or carries bracket blocks),
  and the ``entry_flags`` extra table (:meth:`ProteinDomain.fragments`) carries UniProt's fragment flag. The
  domain keeps the flagged proteins: a selection that excludes them declares it.
- ``taxon_lineage`` makes a lineage label a match on the taxon nodes of the lineage text, not a substring of it
  ("Mammalia" is a substring of "Mammalian orthoreovirus", "Fungi" of "Fungiidae").
- ``keyword_list`` cites UniProt's keyword list; a keyword task is then out of scope (``scope.parquet``,
  ``Domain.scope_mask``) on the proteins whose "no" another field of the same entry contradicts (the rule is
  ``domains.keyword_evidence``). The label is kept, and the ``out_of_scope`` extra table
  (:meth:`ProteinDomain.out_of_scope`) holds the reason of each excluded cell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.config import Config, added_field
from latent_interconnect.domains.base import DOMAINS, CaptureRecipe, Frame, TaskKind, TaskSpec
from latent_interconnect.domains.keyword_evidence import STATED_BY, keyword_ancestors, stated_keywords
from latent_interconnect.domains.tables import (
    DOMAIN_DATA,
    DomainDataSpec,
    DomainTables,
    SourceFile,
    TableDomain,
    read_source,
)
from latent_interconnect.domains.uniprot_entries import read_entries

__all__ = [
    "ENTRY_FLAGS_TABLE",
    "OUT_OF_SCOPE_COLUMNS",
    "OUT_OF_SCOPE_TABLE",
    "VARIANTS_TABLE",
    "VARIANT_COLUMNS",
    "LabelRule",
    "PathwayEntry",
    "ProteinDataSpec",
    "ProteinDomain",
    "VariantLabel",
    "VariantSource",
    "parse_ec_set",
    "parse_keywords",
    "parse_pathway_set",
]

_NAME_END = " ("
_KEYWORD_SEP = ";"
_ACCESSION = "accession"
_SPLITS = ("train", "val", "test")
_EC_CLASSES = frozenset("1234567")
_EC_LEVELS = 4
_STEP = re.compile(r"step \d+/\d+")
_ENTRY_SEP = re.compile(r"\.;\s*")
_RESIDUE = re.compile(r"[A-Z]")
_TAXON_SEP = ", "
_RANK_START = " ("
_SOURCE_SEP = "; "
_FRAGMENT = "fragment"

VARIANTS_TABLE = "variants"
"""The extra table (``domains.tables``) linking each variant object to its parent protein."""
VARIANT_COLUMNS = ("parent", "position", "wt", "mut", "vsplit")
"""Columns of :meth:`ProteinDomain.variants`: the parent's object id, the 1-based position, the parent and mutant
residues, and the original variant split (``train``, ``val``, ``test``, ``heldpos``, ...)."""
ENTRY_FLAGS_TABLE = "entry_flags"
"""The extra table of UniProt's entry flags, one row per protein: ``fragment`` (bool)."""
OUT_OF_SCOPE_TABLE = "out_of_scope"
"""The extra table of the reasons of the scope table's excluded cells, one row per (protein, task) cell."""
OUT_OF_SCOPE_COLUMNS = ("task", "rule", "sources")
"""Columns of :meth:`ProteinDomain.out_of_scope`: the task id, the rule that excludes the cell and the entry's
fields that fire it (``domains.keyword_evidence.stated_keywords``, ``"; "``-joined)."""
_CONTRADICTED = "keyword 'no' contradicted by another field of the same UniProt entry"


@DOMAINS.register("protein")
class ProteinDomain(TableDomain):
    """Proteins: SwissProt entries, encoded from their amino-acid sequence."""

    name = "protein"
    noun = "protein"
    object_kind = "protein"

    def population(self) -> pd.Index:
        """Return the proteins: every object but the variants (population draws never pick a variant)."""
        objects = self.objects().index
        return objects.difference(self.variants().index, sort=False)

    def variants(self) -> pd.DataFrame:
        """Return the variant objects: ``object_id`` (index) → :data:`VARIANT_COLUMNS`; empty without variants."""
        table = self.table(VARIANTS_TABLE)
        if table is None:
            return pd.DataFrame(columns=list(VARIANT_COLUMNS), index=pd.Index([], name="object_id"))
        return table

    def fragments(self) -> pd.Index:
        """Return the proteins UniProt flags as fragments (a variant is not listed: see its parent).

        Raises:
            ValueError: if this data version does not carry the flag (its spec cites no UniProt entries).
        """
        flags = self.table(ENTRY_FLAGS_TABLE)
        if flags is None:
            raise ValueError(f"{self.name}: this data version does not carry UniProt's fragment flag")
        return flags.index[flags[_FRAGMENT].astype(bool)]

    def out_of_scope(self) -> pd.DataFrame:
        """Return the reason of every cell the scope table excludes; empty when the data version excludes none.

        One row per excluded (protein, task) cell: ``object_id`` (index, repeating) → :data:`OUT_OF_SCOPE_COLUMNS`.
        """
        table = self.table(OUT_OF_SCOPE_TABLE)
        if table is None:
            return pd.DataFrame(columns=list(OUT_OF_SCOPE_COLUMNS), index=pd.Index([], name="object_id"))
        return table


class LabelRule(Config):
    """How one label column is computed from the source tables.

    Attributes:
        column: the label column name (tasks cite it as ``TaskSpec.label``).
        rule: ``keyword`` (``value`` is one of the entry's keywords), ``lineage`` (``value`` is a substring of the
            lineage text, or one of its taxon nodes under ``ProteinDataSpec.taxon_lineage``) or ``length_gt`` (the
            sequence is longer than ``value`` residues).
        value: the keyword, the lineage text or taxon, or the length threshold.
    """

    column: str
    rule: Literal["keyword", "lineage", "length_gt"]
    value: str | int


class VariantLabel(Config):
    """One variant label column, read from a column of the variant table.

    Attributes:
        column: the label column name.
        source: the variant-table column.
        positive: the source values that mean True.
        negative: the source values that mean False (every other value is unknown).
    """

    column: str
    source: str
    positive: tuple[str, ...] = Field(min_length=1)
    negative: tuple[str, ...] = Field(min_length=1)


class VariantSource(Config):
    """The protein-variant table: single substitutions of domain proteins, added as variant objects.

    Attributes:
        table: the variant table (``vid``, ``accession``, ``pos``, ``wt``, ``mut``, ``vsplit`` and the field and label
            source columns; the original variant build's ``variants.parquet``).
        fields: text-field name → variant-table column.
        labels: the variant label columns.
    """

    table: SourceFile
    fields: dict[str, str]
    labels: tuple[VariantLabel, ...] = ()


@DOMAIN_DATA.register("protein")
class ProteinDataSpec(DomainDataSpec):
    """A protein domain-data version.

    Attributes:
        table: the SwissProt entry table (``accession``, ``protein_names``, ``sequence``, ``length``, ``keywords``,
            ``caption``, ...).
        annotations: the per-entry annotation table (``accession``, ``lineage``, the cleaned comment fields, ...).
        split: the homology split (``accession``, ``split``).
        clusters: the MMseqs2 cluster table (tab-separated ``representative``, ``member``; no header).
        fields: text-field name → ``"table:<column>"`` or ``"annotations:<column>"``.
        labels: the label columns.
        tasks: the task catalog.
        frames: the prompt frames.
        recipes: the capture recipes.
        variants: the variant objects (``protein-domain-base`` and ``protein-domain``), or None.
        entries: UniProt's flat file of the release the entry table is of (``uniprot_sprot.dat.gz``), or None
            (``protein-domain``). With it, an object's ``name`` is the entry's recommended-name field and the
            ``entry_flags`` extra table carries UniProt's fragment flag.
        taxon_lineage: whether a ``lineage`` label matches a taxon node of the lineage text (``", "``-separated
            ``"<taxon> (<rank>)"`` nodes) instead of a substring of it.
        keyword_list: UniProt's keyword list of the same release (``keywlist.txt``), or None; needs ``entries``. With
            it, each Yes/No task on a keyword label is out of scope on the proteins whose "no" another field of the
            same entry contradicts (``domains.keyword_evidence``).
    """

    table: SourceFile
    annotations: SourceFile
    split: SourceFile
    clusters: SourceFile
    fields: dict[str, str]
    labels: tuple[LabelRule, ...] = Field(min_length=1)
    tasks: tuple[TaskSpec, ...] = Field(min_length=1)
    frames: tuple[Frame, ...] = Field(min_length=1)
    recipes: tuple[CaptureRecipe, ...] = Field(min_length=1)
    variants: VariantSource | None = None
    entries: SourceFile | None = added_field(default=None)
    taxon_lineage: bool = added_field(default=False)
    keyword_list: SourceFile | None = added_field(default=None)

    @model_validator(mode="after")
    def _check_keyword_list(self) -> ProteinDataSpec:
        if self.keyword_list is not None and self.entries is None:
            raise ValueError("keyword_list needs entries: the contradicting fields are read from the UniProt entries")
        return self

    def sources(self) -> dict[str, SourceFile]:
        """Return the source files by role, in a fixed order.

        The entry table, the annotations, the split and the clusters; then the variant table, the UniProt entries and
        the keyword list, each when the spec cites it.
        """
        out = {"table": self.table, "annotations": self.annotations, "split": self.split, "clusters": self.clusters}
        if self.variants is not None:
            out["variants"] = self.variants.table
        if self.entries is not None:
            out["entries"] = self.entries
        if self.keyword_list is not None:
            out["keyword_list"] = self.keyword_list
        return out

    def tables(self, data_root: Path) -> DomainTables:
        """Read and check the sources and derive the domain tables (no store needed).

        Raises:
            IntegrityError: if a source's sha256 differs from the spec.
            ValueError: for a field source that is not ``table:`` / ``annotations:``, a task citing an unknown label,
                an object without a cluster, a protein the UniProt entries lack or give another length (another
                release), or a keyword task whose keyword ``domains.keyword_evidence`` has no rule for.
        """
        table = pd.read_parquet(read_source(data_root, self.table)).set_index(_ACCESSION)
        annotations = pd.read_parquet(read_source(data_root, self.annotations)).set_index(_ACCESSION)
        split = pd.read_parquet(read_source(data_root, self.split), columns=[_ACCESSION, "split"]).set_index(_ACCESSION)
        clusters = pd.read_csv(
            read_source(data_root, self.clusters), sep="\t", header=None, names=["cluster", _ACCESSION], dtype=str
        ).set_index(_ACCESSION)
        ids = table.index
        names = table["protein_names"].fillna("").str.split(_NAME_END, regex=False).str[0]
        extra: dict[str, pd.DataFrame] = {}
        stated: dict[str, dict[str, str]] = {}
        if self.entries is not None:
            names, fragment, stated = self._uniprot(data_root, table)
            extra[ENTRY_FLAGS_TABLE] = pd.DataFrame({_FRAGMENT: fragment})
        objects = pd.DataFrame({"value": table["sequence"], "name": names})
        sources = {"table": table, "annotations": annotations}
        fields = {}
        for field, where in self.fields.items():
            src, _, column = where.partition(":")
            if src not in sources:
                raise ValueError(f"field {field!r}: source {where!r} is not 'table:<column>' or 'annotations:<column>'")
            fields[field] = sources[src][column].reindex(ids)
        entries = pd.DataFrame(
            {"keywords": table["keywords"], "length": table["length"], "lineage": annotations["lineage"].reindex(ids)}
        )
        labels = pd.DataFrame(
            {rule.column: _label(rule, entries, self.taxon_lineage) for rule in self.labels}, index=ids
        )
        scope = None
        if self.keyword_list is not None:
            scope, extra[OUT_OF_SCOPE_TABLE] = _keyword_scope(self._keyword_tasks(), labels, stated)
        columns = {*labels.columns, *fields}
        if self.variants is not None:
            columns |= {*self.variants.fields, *(rule.column for rule in self.variants.labels)}
        unknown = {t.id: t.label for t in self.tasks if t.label not in columns}
        if unknown:
            raise ValueError(f"tasks cite unknown label columns: {unknown}")
        split_col = split["split"].reindex(ids)
        cluster = clusters["cluster"].reindex(ids)
        if cluster.isna().any():
            raise ValueError(
                f"{int(cluster.isna().sum())} objects have no cluster, e.g. {list(ids[cluster.isna()][:3])}"
            )
        tables = DomainTables(
            domain=ProteinDomain.name,
            objects=objects,
            split=split_col[split_col.isin(_SPLITS)],
            clusters=cluster,
            labels=labels,
            fields=pd.DataFrame(fields, index=ids),
            tasks=self.tasks,
            frames=self.frames,
            recipes=self.recipes,
            scope=scope,
            extra=extra,
        )
        if self.variants is None:
            return tables
        return _with_variants(tables, self.variants, pd.read_parquet(read_source(data_root, self.variants.table)))

    def _keyword_tasks(self) -> dict[str, tuple[str, str]]:
        """Yes/No task id → its label column and the keyword the column tests, for the tasks on a ``keyword`` label."""
        keywords = {rule.column: str(rule.value) for rule in self.labels if rule.rule == "keyword"}
        return {
            t.id: (t.label, keywords[t.label]) for t in self.tasks if t.kind is TaskKind.YES_NO and t.label in keywords
        }

    def _uniprot(self, data_root: Path, table: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[str, dict[str, str]]]:
        """Read the UniProt entries of the table's proteins: the recommended names, the fragment flags and the evidence.

        The evidence (with the keyword list only) is keyword → accession → the ``"; "``-joined sources that state it.
        """
        if self.entries is None:
            raise ValueError("the spec cites no UniProt entries")
        tested = sorted({k for _, k in self._keyword_tasks().values()}) if self.keyword_list is not None else []
        unruled = sorted(set(tested) - set(STATED_BY))
        if unruled:
            raise ValueError(f"keyword tasks on {unruled}: domains.keyword_evidence has no rule for them")
        ancestors = {}
        if self.keyword_list is not None:
            ancestors = keyword_ancestors(read_source(data_root, self.keyword_list).read_text(encoding="utf-8"))
        length = table["length"].to_dict()
        names: dict[str, str] = {}
        fragment: dict[str, bool] = {}
        stated: dict[str, dict[str, str]] = {keyword: {} for keyword in tested}
        for entry in read_entries(read_source(data_root, self.entries)):
            accession = entry.accession
            if accession not in length:
                continue
            if entry.length != length[accession]:
                raise ValueError(
                    f"{accession}: the UniProt entries give length {entry.length}, the entry table "
                    f"{length[accession]}: the two are not of one release"
                )
            names[accession] = entry.recommended_name
            fragment[accession] = entry.fragment
            for keyword, sources in stated_keywords(entry, tested, ancestors).items():
                stated[keyword][accession] = _SOURCE_SEP.join(sources)
        missing = table.index.difference(pd.Index(names))
        if len(missing):
            raise ValueError(f"the UniProt entries lack {len(missing)} proteins, e.g. {list(missing[:3])}")
        ids = table.index
        return pd.Series(names).reindex(ids), pd.Series(fragment, dtype=bool).reindex(ids), stated


def _label(rule: LabelRule, entries: pd.DataFrame, taxon_lineage: bool) -> pd.Series:
    if rule.rule == "keyword":
        word = str(rule.value)
        return entries["keywords"].fillna("").map(lambda s: word in {k.strip() for k in s.split(_KEYWORD_SEP)})
    if rule.rule == "lineage":
        if taxon_lineage:
            taxon = str(rule.value)
            return entries["lineage"].fillna("").map(lambda s: taxon in _taxon_nodes(s))
        return entries["lineage"].fillna("").str.contains(str(rule.value), regex=False)
    return entries["length"] > int(rule.value)


def _taxon_nodes(lineage: str) -> set[str]:
    """The taxa of a lineage text: its ``", "``-separated nodes, each without its ``" (<rank>)"`` suffix."""
    nodes = set()
    for node in lineage.split(_TAXON_SEP):
        taxon, ranked, _ = node.rpartition(_RANK_START)
        nodes.add(taxon if ranked else node)
    return nodes


def _keyword_scope(
    tasks: dict[str, tuple[str, str]], labels: pd.DataFrame, stated: dict[str, dict[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The scope table of the keyword tasks, and the reasons of its excluded cells.

    A cell is excluded where the protein's keyword label is "no" and another field of its entry states the keyword.
    """
    scope = {}
    reasons = []
    for task, (column, keyword) in tasks.items():
        sources = pd.Series(stated[keyword], dtype=object).reindex(labels.index)
        excluded = ~labels[column].astype(bool) & sources.notna()
        scope[task] = ~excluded
        reasons.append(pd.DataFrame({"task": task, "rule": _CONTRADICTED, "sources": sources[excluded]}))
    table = pd.concat(reasons) if reasons else pd.DataFrame(columns=list(OUT_OF_SCOPE_COLUMNS))
    return pd.DataFrame(scope, index=labels.index).astype(bool), table[list(OUT_OF_SCOPE_COLUMNS)]


def _with_variants(proteins: DomainTables, source: VariantSource, table: pd.DataFrame) -> DomainTables:
    """Return ``proteins`` with the variant objects of ``table`` added (checked against their parents)."""
    missing = sorted({"vid", _ACCESSION, "pos", "wt", "mut", "vsplit"} - set(table.columns))
    missing += sorted({*source.fields.values(), *(rule.source for rule in source.labels)} - set(table.columns))
    if missing:
        raise ValueError(f"the variant table lacks columns {missing}")
    t = table.set_index("vid")
    if not t.index.is_unique or t.index.isin(proteins.objects.index).any():
        raise ValueError("variant ids repeat, or name a protein")
    pos = t["pos"].astype(int)
    wt, mut, parent = t["wt"].astype(str), t["mut"].astype(str), t[_ACCESSION].astype(str)
    named = parent + ":" + wt + pos.astype(str) + mut
    bad = t.index[(named != t.index) | ~wt.str.fullmatch(_RESIDUE) | ~mut.str.fullmatch(_RESIDUE) | (wt == mut)]
    if len(bad):
        raise ValueError(f"{len(bad)} variant(s) are not single substitutions named <parent>:<wt><pos><mut>: {bad[:3]}")
    orphans = t.index[~parent.isin(proteins.objects.index)]
    if len(orphans):
        raise ValueError(f"{len(orphans)} variant(s) of proteins not in the domain, e.g. {list(orphans[:3])}")
    sequence = proteins.objects["value"].reindex(parent).to_numpy()
    values = []
    for vid, seq, p, w, m in zip(t.index, sequence, pos, wt, mut, strict=True):
        if not 1 <= p <= len(seq) or seq[p - 1] != w:
            raise ValueError(f"variant {vid}: the parent has no {w} at position {p}")
        values.append(seq[: p - 1] + m + seq[p:])
    vsplit = t["vsplit"].astype(str)
    split = vsplit[vsplit.isin(_SPLITS)]
    parent_split = proteins.split.reindex(parent[split.index]).to_numpy()
    if (split.to_numpy() != parent_split).any():
        raise ValueError("a variant's split differs from its parent's (rebuild the variant table on this split)")
    ids = t.index.rename(None)
    objects = pd.DataFrame({"value": values, "name": ""}, index=ids)
    labels = pd.DataFrame(
        {
            rule.column: t[rule.source].map(
                lambda v, r=rule: True if v in r.positive else (False if v in r.negative else None)
            )
            for rule in source.labels
        },
        index=ids,
    )
    fields = pd.DataFrame({name: t[column] for name, column in source.fields.items()}, index=ids)
    link = pd.DataFrame({"parent": parent, "position": pos, "wt": wt, "mut": mut, "vsplit": vsplit}, index=ids)[
        list(VARIANT_COLUMNS)
    ]
    all_labels = pd.concat([proteins.labels, labels]).astype("boolean")
    return DomainTables(
        domain=proteins.domain,
        objects=pd.concat([proteins.objects, objects]),
        split=pd.concat([proteins.split, split.rename(None)]),
        clusters=pd.concat([proteins.clusters, proteins.clusters.reindex(parent).set_axis(ids)]),
        labels=all_labels,
        fields=pd.concat([proteins.fields, fields]),
        tasks=proteins.tasks,
        frames=proteins.frames,
        recipes=proteins.recipes,
        scope=None if proteins.scope is None else proteins.scope.reindex(all_labels.index, fill_value=True),
        extra={**proteins.extra, VARIANTS_TABLE: link},
    )


@dataclass(frozen=True)
class PathwayEntry:
    """One entry of a SwissProt PATHWAY field.

    Attributes:
        category: its first ``;``-part (e.g. ``"Carbohydrate degradation"``).
        pathway: its second part, else the category.
        step: its own ``"step n/m"``, if any.
    """

    category: str
    pathway: str
    step: str | None


def parse_ec_set(text: str | None) -> list[tuple[str, str, str, str]]:
    """Return every EC number of an ``ec`` field (``"2.7.7.-; 2.7.7.108"``), in order, without repeats.

    Each is its four levels; an entry without four levels or with an unknown class (``1``–``7``) is skipped
    (the relational families' EC sets).
    """
    out: list[tuple[str, str, str, str]] = []
    for entry in (text or "").split(";"):
        levels = entry.strip().split(".")
        if len(levels) == _EC_LEVELS and levels[0] in _EC_CLASSES:
            ec = (levels[0], levels[1], levels[2], levels[3])
            if ec not in out:
                out.append(ec)
    return out


def parse_pathway_set(text: str | None) -> list[PathwayEntry]:
    """Return every entry of a ``pathway`` field, in order (the relational families' pathway sets).

    Entries are separated by ``".;"`` (``"A; B; C: step 1/2.; D; E."``) and each is parsed on its own.
    """
    out = []
    for entry in _ENTRY_SEP.split((text or "").strip()):
        parts = [p.strip().rstrip(".") for p in entry.strip().rstrip(".").split(";")]
        if not parts[0]:
            continue
        step = next((m.group(0) for p in parts if (m := _STEP.search(p))), None)
        out.append(PathwayEntry(parts[0], parts[1] if len(parts) > 1 else parts[0], step))
    return out


def parse_keywords(text: str | None) -> list[str]:
    """Return the keywords of a ``keywords`` field (``;``-separated), stripped, in order."""
    return [k.strip() for k in (text or "").split(_KEYWORD_SEP) if k.strip()]
