"""The molecule domain (ChEBI 3-star compounds), and the spec its domain-data artifact is built from.

:class:`MoleculeDataSpec` (registered in ``DOMAIN_DATA`` as ``molecule``) derives the domain tables from the paper's
molecule sources, each cited by a path under the data root (``Paths.data``) and its sha256, checked before reading.
The paper's procedure is kept (the original research code's molecule domain, and its evaluation tasks):

- objects are the ChEBI entry table's molecules, keyed by ChEBI id. The ``value`` is the entry's SMILES, the string
  the paper's encoders read; an entry without one has no value, and no encoder
  can read it. The ``name`` is the ChEBI name, used whole;
- the split is the similarity split's ``split`` column, its labels mapped to train / val / test by
  ``MoleculeDataSpec.split_labels`` when the split names its own (the band-stratified split's ``N_val``,
  ``A_test``, ...); molecules it marks otherwise (``discarded``) are in no split. The split's declared ``strata``
  columns (the band-stratified split's stratum, Tanimoto band and annotation bin) are the extra table ``strata``.
  Clusters are its Butina ``cluster_id`` (the paper's sampling unit; the split's form groups never span two clusters);
- the fields are columns of the training field table (the paper's training targets), restricted to the fields the
  spec names. A text field is the column's text. An item-list field (:class:`ItemList`) is the column's items with
  the source table's presentation wording removed (``"ChEBI parent classes: a; b."`` → ``("a", "b")``): data are
  data, and streams word the items from their own config. Each line is parsed by its declared format and must
  re-render to itself exactly;
- declared ontology fields (:class:`OntologyFields`) add item lists from a hash-pinned ChEBI release (the class graph
  above each compound, its functional parents' classes) and the table of its held-out functional parents;
- the Yes/No tasks are imported from the paper's evaluation item table: every task of the named groups, with its
  question and label key as the paper asked them, in the table's
  order. Further tasks, the frames and the capture recipes are data in the spec (the domain stage's config);
- a task's label follows its key ('No' = not annotated, open world; every label is a bool):

  - ``@class:<id>``: the class is in the molecule's transitive ``is_a`` closure (the label table's ``class_ids``);
  - ``@role:<id>``: the role is in its role closure (the label table's ``role_ids``, closed upward);
  - ``@mass:median``: its mass is above the median mass of the training molecules, or above the declared
    ``MoleculeDataSpec.mass_threshold`` when one is set, so that a new split cannot move the labels (no mass: No);
  - ``@charge:nonzero``: its net charge is not zero (no charge: zero);
  - ``@element:<g>``: it contains an element of group ``g`` (``MoleculeDataSpec.element_groups``; a group not listed
    is the element symbol itself);

- **the unstated subset** is a declared rule, :class:`StatedRule`: a task the item table marks
  ``has_pattern`` is scored only on molecules whose trained text does not match the task's stated pattern. The
  trained text is the lower-cased join of the entry table's trained columns and the line ``Roles: a; b.`` of the
  molecule's direct roles that are trained roles; the patterns are the task split's. The rule is materialized as the
  artifact's scope table (``Domain.scope_mask``); a task without a pattern is scored on every molecule.

A data version may opt in to three data corrections (``molecule-domain`` uses all three; each is off by default and
then leaves the spec's digest and the artifact as they were):

- ``missing_out_of_scope``: a molecule without a mass is out of scope for the ``@mass:median`` tasks and one
  without a charge for the ``@charge:nonzero`` tasks, instead of reading "No";
- ``inherited_roles``: a molecule's role closure also holds the roles of its parent compounds, as ChEBI's own
  classification gives them: the ``has_role`` targets of every class in its ``is_a`` closure, closed upward over the
  role hierarchy of the declared ontology release;
- ``out_of_scope_cells``: cited cell lists (``id``, ``task``, ``rule``, ``source``), each cell out of scope (the
  class "hydroxides", whose list is derived from the structures).

The cells these rules exclude are False in the scope table beside the unstated subset's; their labels are kept, and
the extra table ``out_of_scope`` (:meth:`MoleculeDomain.out_of_scope`) holds the reason of each.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import ArtifactDigest, Config, added_field
from latent_interconnect.domains.base import DOMAINS, CaptureRecipe, Frame, TaskKind, TaskScope, TaskSpec
from latent_interconnect.domains.chebi import ChebiRelease, chebi_name, load_release
from latent_interconnect.domains.eval_proteins import EvalProteins, load_eval_proteins
from latent_interconnect.domains.tables import (
    DOMAIN_DATA,
    DomainDataSpec,
    DomainTables,
    SourceFile,
    TableDomain,
    commit_domain_data,
    read_source,
)

__all__ = [
    "FUNCTIONAL_PARENTS",
    "OUT_OF_SCOPE_COLUMNS",
    "OUT_OF_SCOPE_TABLE",
    "ArtifactFile",
    "ItemList",
    "MoleculeDataSpec",
    "MoleculeDomain",
    "OntologyFields",
    "StatedRule",
    "TaskImport",
]

_ID = "chebi_id"
_SPLITS = ("train", "val", "test")
_ID_SEP = ";"
_ELEMENT_SEP = ";"
_TEXT_JOIN = " \n "
_KEY_KINDS = ("@class:", "@role:", "@element:", "@mass:median", "@charge:nonzero")
_MASS, _CHARGE = "@mass:median", "@charge:nonzero"
_MISSING = {_MASS: ("mass", "no mass in ChEBI"), _CHARGE: ("charge", "no charge in ChEBI")}
_CELL_COLUMNS = ("id", "task", "rule", "source")

OUT_OF_SCOPE_TABLE = "out_of_scope"
"""The extra table of the reasons of the cells a data version's opt-in rules exclude, one row per cell and rule."""
OUT_OF_SCOPE_COLUMNS = ("task", "rule", "sources")
"""Columns of :meth:`MoleculeDomain.out_of_scope`: the task id, the rule that excludes the cell and what it rests on
(the entry table's missing column, or the cell list's ``source``)."""


@DOMAINS.register("molecule")
class MoleculeDomain(TableDomain):
    """Molecules: ChEBI 3-star compounds, encoded from their SMILES."""

    name = "molecule"
    noun = "compound"  # "compound", not "molecule"
    object_kind = "molecule"

    def out_of_scope(self) -> pd.DataFrame:
        """Return the reason of every cell the opt-in rules exclude; empty when the data version has none.

        One row per excluded (molecule, task) cell and rule: ``object_id`` (index, repeating) →
        :data:`OUT_OF_SCOPE_COLUMNS`. The unstated subset is a rule of the task, not listed here.
        """
        table = self.table(OUT_OF_SCOPE_TABLE)
        if table is None:
            return pd.DataFrame(columns=list(OUT_OF_SCOPE_COLUMNS), index=pd.Index([], name="object_id"))
        return table

    def training_mask(self, task: TaskSpec, object_ids: pd.Index) -> pd.Series:
        """Return the scope without the unstated subset: every molecule but the cells the opt-in rules exclude.

        Those cells (:meth:`out_of_scope`: a missing mass or charge, the cited cell lists) have no usable label, so
        they stay out; a molecule whose trained text states the property is a training label like any other.

        Raises:
            KeyError: for an object id the domain does not hold.
        """
        scope = self.scope_mask(task, object_ids)  # refuses an object the domain does not hold
        listed = self.out_of_scope()
        excluded = scope.index.isin(listed.index[listed["task"] == task.id])
        return pd.Series(~excluded, index=scope.index)


class TaskImport(Config):
    """Which tasks of the evaluation item table enter the catalog, and how.

    Attributes:
        item_group: the item table's ``group`` whose tasks are imported (e.g. ``trained_yesno``).
        scope: the imported tasks' scope.
        group: the imported tasks' figure group (``TaskSpec.group``).
        implied: tasks of the group whose scope is ``implied`` instead (implied by training, evaluated but never
            pooled with the held-out tasks; the screened roles, as the implied-by-trained-fields group).
    """

    item_group: str
    scope: TaskScope
    group: str
    implied: tuple[str, ...] = ()


class ItemList(Config):
    """How an item-list field is read from a line of the training field table.

    A line is ``one + item + end`` for one item and ``many + items joined by separator + end`` for more; no item
    contains the separator, so the items are unambiguous.

    Attributes:
        column: the training field table's column.
        one: the line's head before a single item (e.g. ``"ChEBI parent class: "``).
        many: the line's head before several items (e.g. ``"ChEBI parent classes: "``).
        separator: between items.
        end: after the last item.
    """

    column: str
    one: str
    many: str
    separator: str = Field(min_length=1)
    end: str

    def items(self, line: str) -> tuple[str, ...]:
        """Return the line's items.

        Raises:
            ValueError: if the line does not re-render to itself from its items.
        """
        for head, several in ((self.many, True), (self.one, False)):
            if line.startswith(head) and line.endswith(self.end) and len(line) >= len(head) + len(self.end):
                body = line[len(head) : len(line) - len(self.end)]
                items = tuple(body.split(self.separator))
                if all(items) and (len(items) > 1) == several and self.render(items) == line:
                    return items
        raise ValueError(f"column {self.column!r}: {line!r} is not an item line of the declared format")

    def render(self, items: tuple[str, ...]) -> str:
        """Return the line of ``items`` in this format (the inverse of :meth:`items`)."""
        head = self.many if len(items) > 1 else self.one
        return head + self.separator.join(items) + self.end


class ArtifactFile(Config):
    """A file of an artifact, cited by the artifact's digest (e.g. the molecule text table, ``molecule_text``).

    Attributes:
        artifact: the artifact's digest.
        kind: its kind (checked).
        file: the file's name inside the artifact.
    """

    artifact: ArtifactDigest
    kind: str = Field(min_length=1)
    file: str = Field(min_length=1)

    def path(self, store: ArtifactStore) -> Path:
        """Return the file's path in ``store`` once the artifact's manifest matches its digest.

        Raises:
            IntegrityError: if the artifact does not match its digest.
            ValueError: if it is of another kind.
        """
        ref = store.ref(self.artifact)
        if ref.kind != self.kind:
            raise ValueError(f"{ref.short} is a {ref.kind!r} artifact, not {self.kind!r}")
        store.manifest(ref)
        return store.path(ref) / self.file


class OntologyFields(Config):
    """Item-list fields and a table derived from a ChEBI release (``domains.chebi``), for the relational families.

    Every name is written as its :func:`~latent_interconnect.domains.chebi.chebi_name`; an item is two names joined by
    ``arrow``. A molecule is *held out* when it is a molecule of the domain outside the train split (val, test or no
    split); a ChEBI term that is no molecule of the domain is not.

    - ``class_graph``: every ``is_a`` edge above the compound's direct classes (the entry table's ``class_ids``),
      ``"<class> > <parent>"``, breadth-first from the direct classes in the table's order, each class's parents in
      the release's order, each edge once; obsolete terms are not walked;
    - ``functional_parent_classes``: each functional parent's direct classes, ``"<parent> > <class>"``, the parents in
      the entry table's order (``functional_parent_ids``), a held-out parent left out (no held-out molecule's classes
      are shown, the relational stream's contract);
    - the extra table ``functional_parents``: per compound with one, its held-out functional parents' names (column
      ``held_out``), against which the relational stream checks the contract.

    An item that names an evaluation protein or any EC number (``domains.eval_proteins``) is dropped: the original item
    rule of the molecule text.

    Attributes:
        ontology: the ChEBI OBO release (the one the label table was built from).
        class_graph: the class-graph field's name.
        functional_parent_classes: the functional-parent-classes field's name.
        arrow: between the two names of an item.
        eval_proteins: the evaluation proteins whose names, genes and EC numbers block an item.
    """

    ontology: SourceFile
    class_graph: str = Field(min_length=1)
    functional_parent_classes: str = Field(min_length=1)
    arrow: str = Field(min_length=1)
    eval_proteins: EvalProteins


FUNCTIONAL_PARENTS = "functional_parents"
"""The extra table of the held-out functional parents (:class:`OntologyFields`)."""


class StatedRule(Config):
    """The unstated-subset rule: which trained text states a held-out task, so the task is scored on the rest.

    Attributes:
        columns: the entry-table columns of the trained text, in order (the paper's: definition, classes, functional
            parent, conjugates).
        role_assignment: the role-set ``assignment`` whose roles are trained; a molecule's direct roles with it form
            the text's ``Roles: a; b.`` line.
    """

    columns: tuple[str, ...] = Field(min_length=1)
    role_assignment: str


@DOMAIN_DATA.register("molecule")
class MoleculeDataSpec(DomainDataSpec):
    """A molecule domain-data version.

    Attributes:
        split: the similarity split (``chebi_id``, ``split``, ``cluster_id``, ...).
        entries: the ChEBI entry table (``chebi_id``, ``name``, ``smiles``, ...).
        labels: the label table (``chebi_id``, ``class_ids``: the ``;``-separated transitive ``is_a`` closure).
        text: the training field table (``chebi_id``, ``split``, one column per text field): a file under the data
            root, or a file of an artifact (the package's rendered text table, read through the store in
            :meth:`build`).
        items: the evaluation item table the tasks are imported from (``task``, ``group``, ``question``, ``key``,
            ``has_pattern``, one row per task and molecule).
        patterns: the task split's held-out task table (``task``, ``stated_pattern``).
        role_sets: the task split's role sets (``id``, ``name``, ``assignment``).
        split_labels: the split's own labels → ``train`` / ``val`` / ``test`` (a label not listed: no split); empty:
            the split's labels are already those (the ECFP4-split molecule domain).
        strata: columns of the split copied into the extra table ``strata`` (empty: no table).
        text_split_matches: whether the text table's ``split`` column must equal the split (True: the table was
            rendered on this split, the ECFP4-split domain; False: it was rendered on another, e.g. the
            band-stratified split on the ECFP4-split domain's table, and its
            rows are read whatever their split).
        mass_threshold: the ``@mass:median`` threshold in Da, frozen; None: the median mass of the training
            molecules (the ECFP4-split molecule domain).
        ontology_fields: the fields and table derived from a ChEBI release; None: none (the ECFP4-split
            molecule domain).
        fields: the text fields: field name → column of ``text``.
        item_lists: the item-list fields: field name → how its items are read from ``text``.
        imported: the task groups imported from ``items``.
        stated: the unstated-subset rule.
        element_groups: element group → its element symbols, for ``@element:<group>`` keys.
        tasks: further tasks (e.g. the caption target), after the imported ones.
        frames: the prompt frames.
        recipes: the capture recipes.
        missing_out_of_scope: whether a molecule without a mass (a charge) is out of scope for the ``@mass:median``
            (``@charge:nonzero``) tasks; False: its label reads "No" and it is scored (the ECFP4-split,
            band-split and group-split domains).
        inherited_roles: whether the role closure also holds the roles of the molecule's parent compounds (the
            module docstring); needs ``ontology_fields``, whose release it reads. False: the label table's closure.
        out_of_scope_cells: cell lists (parquet: ``id``, ``task``, ``rule``, ``source``), each cell out of scope.
    """

    split: SourceFile
    entries: SourceFile
    labels: SourceFile
    text: SourceFile | ArtifactFile
    items: SourceFile
    patterns: SourceFile
    role_sets: SourceFile
    split_labels: dict[str, Literal["train", "val", "test"]] = Field(default_factory=dict)
    strata: tuple[str, ...] = ()
    text_split_matches: bool = True
    mass_threshold: float | None = Field(default=None, gt=0)
    ontology_fields: OntologyFields | None = None
    fields: dict[str, str] = Field(min_length=1)
    item_lists: dict[str, ItemList] = Field(default_factory=dict)
    imported: tuple[TaskImport, ...] = Field(min_length=1)
    stated: StatedRule
    element_groups: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    tasks: tuple[TaskSpec, ...] = ()
    frames: tuple[Frame, ...] = Field(min_length=1)
    recipes: tuple[CaptureRecipe, ...] = Field(min_length=1)
    missing_out_of_scope: bool = added_field(default=False)
    inherited_roles: bool = added_field(default=False)
    out_of_scope_cells: tuple[SourceFile, ...] = added_field(default=())

    @model_validator(mode="after")
    def _check_inherited_roles(self) -> MoleculeDataSpec:
        if self.inherited_roles and self.ontology_fields is None:
            raise ValueError("inherited_roles needs ontology_fields: the parents' roles are read from its release")
        return self

    def sources(self) -> dict[str, SourceFile]:
        """Return the source files by role, in order.

        The split, the entries, the labels, the text (when a file), the items, the patterns, the role sets, the
        ontology and the evaluation proteins' tables when ontology fields are declared, and the cell lists.
        """
        text = {"text": self.text} if isinstance(self.text, SourceFile) else {}
        onto: dict[str, SourceFile] = {}
        if self.ontology_fields is not None:
            proteins = self.ontology_fields.eval_proteins
            onto = {"ontology": self.ontology_fields.ontology, "uniprot": proteins.uniprot}
            onto |= {f"candidates/{i}": c for i, c in enumerate(proteins.candidates)}
        return {
            "split": self.split,
            "entries": self.entries,
            "labels": self.labels,
            **text,
            "items": self.items,
            "patterns": self.patterns,
            "role_sets": self.role_sets,
            **onto,
            **{f"out_of_scope_cells/{i}": c for i, c in enumerate(self.out_of_scope_cells)},
        }

    def artifact_inputs(self, store: ArtifactStore) -> dict[str, ArtifactRef]:
        """Return the text table's artifact when the text is one (recorded as the domain stage's input)."""
        return {"text": store.ref(self.text.artifact)} if isinstance(self.text, ArtifactFile) else {}

    def tables(self, data_root: Path, store: ArtifactStore | None = None) -> DomainTables:
        """Read and check the sources and derive the domain tables (no store needed).

        Raises:
            IntegrityError: if a source's sha256 differs from the spec.
            ValueError: if the tables do not cover the same molecules, the text table's split disagrees with the split
                (when ``text_split_matches``), ``split_labels`` names a label the split does not use,
                a field is named twice or an item line does not parse, an imported task's question, key or
                ``has_pattern`` varies, its key is of an unknown kind or a marked task has no pattern, a task cites an
                unknown label column, a task id repeats, or a cell list lacks a column, names a molecule or a Yes/No
                task the domain does not have, or repeats a cell.
        """
        split = pd.read_parquet(read_source(data_root, self.split)).set_index(_ID)
        if self.split_labels:
            unused = sorted(set(self.split_labels) - set(split["split"].unique()))
            if unused:
                raise ValueError(f"split_labels names labels the split does not use: {unused}")
            split["split"] = split["split"].map(self.split_labels)
        entries = pd.read_parquet(read_source(data_root, self.entries)).set_index(_ID)
        closures = pd.read_parquet(read_source(data_root, self.labels), columns=[_ID, "class_ids", "role_ids"])
        closures = closures.set_index(_ID)
        text = pd.read_parquet(self._text_path(data_root, store)).set_index(_ID)
        items = pd.read_parquet(
            read_source(data_root, self.items), columns=["task", "group", "question", "key", "has_pattern"]
        )
        patterns = pd.read_csv(read_source(data_root, self.patterns), sep="\t", dtype=str).set_index("task")
        role_sets = pd.read_csv(read_source(data_root, self.role_sets), sep="\t", dtype=str).set_index("id")
        ids = entries.index
        for what, table in (("split", split), ("labels", closures)):
            if not table.index.sort_values().equals(ids.sort_values()):
                raise ValueError(f"the {what} table does not hold the entry table's {len(ids)} molecules")
        stray = text.index.difference(ids)
        if len(stray):
            raise ValueError(
                f"the text table has {len(stray)} molecules not in the entry table, e.g. {list(stray[:3])}"
            )
        disagree = text.index[text["split"] != split["split"].reindex(text.index)]
        if self.text_split_matches and len(disagree):
            raise ValueError(f"the text table's split differs for {len(disagree)} molecules, e.g. {list(disagree[:3])}")
        imported, marked = _imported_tasks(items, self.imported)
        tasks = (*imported, *self.tasks)
        task_ids = [t.id for t in tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError(f"a task id repeats in {task_ids}")
        train = split.index[split["split"] == "train"]
        with_roles = self._closures(data_root, closures.reindex(ids))
        rules = _LabelRules(entries.reindex(ids), with_roles, train, self.element_groups, self.mass_threshold)
        labels = pd.DataFrame({t.label: rules.label(t.label) for t in imported}, index=ids)
        clash = set(self.fields) & set(self.item_lists)
        if clash:
            raise ValueError(f"fields named both text and item list: {sorted(clash)}")
        columns = {name: text[column].reindex(ids) for name, column in self.fields.items()}
        for name, rule in self.item_lists.items():
            columns[name] = text[rule.column].reindex(ids).map(lambda v, r=rule: None if pd.isna(v) else r.items(v))
        extra = {"strata": split[list(self.strata)].reindex(ids)} if self.strata else {}
        if self.ontology_fields is not None:
            derived, parents = _ontology_fields(self.ontology_fields, entries.reindex(ids), split["split"], data_root)
            columns |= derived
            extra[FUNCTIONAL_PARENTS] = parents
        fields = pd.DataFrame(columns, index=ids)
        unknown = {
            t.id: t.label
            for t in tasks
            if t.label not in (fields.columns if t.kind is TaskKind.TEXT else labels.columns)
        }
        if unknown:
            raise ValueError(f"tasks cite unknown label columns: {unknown}")
        trained = _trained_text(entries.reindex(ids), self.stated, role_sets)
        stated = {task: ~trained.str.contains(_pattern(patterns, task), regex=True) for task in marked}
        scope, reasons = self._scope(data_root, pd.DataFrame(stated, index=ids), entries.reindex(ids), imported)
        split_col = split["split"].reindex(ids)
        return DomainTables(
            domain=MoleculeDomain.name,
            objects=pd.DataFrame({"value": entries["smiles"], "name": entries["name"]}),
            split=split_col[split_col.isin(_SPLITS)],
            clusters=split["cluster_id"].reindex(ids).astype(str),
            labels=labels,
            fields=fields,
            tasks=tasks,
            frames=self.frames,
            recipes=self.recipes,
            scope=scope.astype(bool) if len(scope.columns) else None,
            extra=extra | reasons,
        )

    def _closures(self, data_root: Path, closures: pd.DataFrame) -> pd.DataFrame:
        """The label table's closures, the role closure with the parents' roles when ``inherited_roles``."""
        if not self.inherited_roles or self.ontology_fields is None:
            return closures
        release = load_release(self.ontology_fields.ontology, data_root)
        return closures.assign(role_ids=_with_parent_roles(closures, release))

    def _scope(
        self, data_root: Path, stated: pd.DataFrame, entries: pd.DataFrame, imported: list[TaskSpec]
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """The unstated subset's scope table less the opt-in rules' cells, and their reasons as an extra table.

        A task that gains an excluded cell and has no column gets one (True elsewhere), after the others; without
        such a rule the table is ``stated`` and there is no extra table.
        """
        excluded = self._excluded_cells(data_root, entries, imported)
        if excluded is None:
            return stated, {}
        scope = stated.copy()
        for task in dict.fromkeys(excluded["task"]):
            allowed = scope[task] if task in scope.columns else pd.Series(True, index=scope.index)
            scope[task] = allowed & ~scope.index.isin(excluded.index[excluded["task"] == task])
        return scope, {OUT_OF_SCOPE_TABLE: excluded}

    def _excluded_cells(self, data_root: Path, entries: pd.DataFrame, imported: list[TaskSpec]) -> pd.DataFrame | None:
        """The cells the opt-in rules put out of scope, with their reasons; None when no such rule is declared.

        ``object_id`` (index, repeating) → :data:`OUT_OF_SCOPE_COLUMNS`: the missing-value cells in catalog order of
        their tasks, then each cell list's rows in its order.
        """
        if not (self.missing_out_of_scope or self.out_of_scope_cells):
            return None
        parts = []
        if self.missing_out_of_scope:
            for task in imported:
                if task.label in _MISSING:
                    column, rule = _MISSING[task.label]
                    missing = entries.index[entries[column].isna()]
                    reason = {"task": task.id, "rule": rule, "sources": f"entries.{column} is missing"}
                    parts.append(pd.DataFrame(reason, index=missing))
        known = {t.id for t in imported}
        for i, cited in enumerate(self.out_of_scope_cells):
            cells = pd.read_parquet(read_source(data_root, cited))
            lacking = [c for c in _CELL_COLUMNS if c not in cells.columns]
            if lacking:
                raise ValueError(f"cell list {i} ({cited.path}) lacks the columns {lacking}")
            stray = sorted(set(cells["id"]) - set(entries.index))[:3]
            unknown = sorted(set(cells["task"]) - known)
            if stray or unknown:
                raise ValueError(f"cell list {i} ({cited.path}) names unknown molecules {stray} or tasks {unknown}")
            if cells.duplicated(["id", "task"]).any():
                raise ValueError(f"cell list {i} ({cited.path}) repeats a cell")
            renamed = cells.rename(columns={"source": "sources"}).set_index("id")
            parts.append(renamed[list(OUT_OF_SCOPE_COLUMNS)].astype(str))
        table = pd.concat(parts) if parts else pd.DataFrame(columns=list(OUT_OF_SCOPE_COLUMNS))
        return table[list(OUT_OF_SCOPE_COLUMNS)]

    def build(self, store: ArtifactStore, data_root: Path) -> ArtifactRef:
        """Commit the domain data as :meth:`DomainDataSpec.build` does, and record which molecules have no text.

        The metadata's ``without_text`` counts, per split (``none``: in no split), the molecules for which the text
        table has no row: the rendered training text covers the molecules it was rendered for, and a split it was not
        rendered on leaves the others without text fields.

        Raises:
            IntegrityError: if a source's sha256 differs from the spec.
            ValueError: as :meth:`tables`, or :func:`commit_domain_data`.
        """
        tables = self.tables(data_root, store)
        text = pd.read_parquet(self._text_path(data_root, store), columns=[_ID])[_ID]
        missing = tables.objects.index.difference(pd.Index(text))
        by_split = tables.split.reindex(missing).fillna("none").value_counts()
        meta = {
            "spec": self.digest(),
            "sources": {k: f.sha256 for k, f in self.sources().items()},
            "artifacts": {k: r.sha256 for k, r in self.artifact_inputs(store).items()},
            "without_text": {str(k): int(v) for k, v in sorted(by_split.items())},
        }
        return commit_domain_data(store, tables, meta)

    def _text_path(self, data_root: Path, store: ArtifactStore | None) -> Path:
        """Return the text table's path, checked: a source file by its sha256, an artifact file by its digest.

        Raises:
            ValueError: for an artifact text without a store (build the domain through :meth:`build`).
        """
        if isinstance(self.text, SourceFile):
            return read_source(data_root, self.text)
        if store is None:
            raise ValueError("the text table is an artifact: build the domain with its store")
        return self.text.path(store)


class _LabelRules:
    """A label key's rule over the molecules (the module docstring lists the kinds)."""

    def __init__(
        self,
        entries: pd.DataFrame,
        closures: pd.DataFrame,
        train: pd.Index,
        element_groups: dict[str, tuple[str, ...]],
        mass_threshold: float | None,
    ) -> None:
        self.entries, self.groups = entries, element_groups
        self.classes = closures["class_ids"].fillna("").map(_id_set)
        self.roles = closures["role_ids"].fillna("").map(_id_set)
        median = float(entries.loc[entries.index.intersection(train), "mass"].median())
        self.median_mass = median if mass_threshold is None else mass_threshold

    def label(self, key: str) -> pd.Series:
        kind, _, arg = key.partition(":")
        if kind == "@class":
            return self.classes.map(lambda c: arg in c)
        if kind == "@role":
            return self.roles.map(lambda r: arg in r)
        if key == "@mass:median":
            return (self.entries["mass"] > self.median_mass).fillna(False).astype(bool)
        if key == "@charge:nonzero":
            return self.entries["charge"].fillna(0).astype(float) != 0
        if kind == "@element":
            want = frozenset(self.groups.get(arg, (arg,)))
            found = self.entries["elements"].fillna("").map(lambda e: frozenset(e.split(_ELEMENT_SEP)))
            return found.map(lambda e: bool(e & want))
        raise ValueError(f"label key {key!r} is of none of the kinds {_KEY_KINDS}")


def _with_parent_roles(closures: pd.DataFrame, release: ChebiRelease) -> pd.Series:
    """Each molecule's ``role_ids`` with the roles of its parent compounds added, ``;``-joined and sorted.

    A parent compound is a class of the molecule's ``is_a`` closure (``class_ids``); its roles are its ``has_role``
    targets in ``release``, closed upward over ``is_a`` (obsolete terms are not walked). The molecule's own closure
    (``role_ids``) is kept whole, so no role is lost.
    """
    live = {t: e for t, e in release.entries.items() if not e.obsolete}
    above: dict[str, frozenset[str]] = {}
    given: dict[str, frozenset[str]] = {}

    def closed(role: str) -> frozenset[str]:
        if role not in above:
            seen: set[str] = set()
            queue = [role]
            while queue:
                term = queue.pop()
                if term in live and term not in seen:
                    seen.add(term)
                    queue.extend(live[term].is_a)
            above[role] = frozenset(seen)
        return above[role]

    def roles_of(compound: str) -> frozenset[str]:
        if compound not in given:
            own = live[compound].has_role if compound in live else ()
            given[compound] = frozenset().union(*(closed(r) for r in own))
        return given[compound]

    def joined(row: tuple[object, object]) -> str:
        classes, roles = row
        inherited = frozenset().union(*(roles_of(c) for c in _id_list(classes)))
        return _ID_SEP.join(sorted(inherited | frozenset(_id_list(roles))))

    pairs = zip(closures["class_ids"], closures["role_ids"], strict=True)
    return pd.Series([joined(pair) for pair in pairs], index=closures.index, dtype=object)


def _trained_text(entries: pd.DataFrame, rule: StatedRule, role_sets: pd.DataFrame) -> pd.Series:
    """The unstated subset's trained text per molecule, lower-cased: the rule's columns, then its trained roles line."""
    trained = set(role_sets.index[role_sets["assignment"] == rule.role_assignment])
    names = role_sets["name"].to_dict()
    text = entries[list(rule.columns)].fillna("").astype(str).agg(_TEXT_JOIN.join, axis=1)

    def role_line(joined: str) -> str:
        own = [r.strip() for r in joined.split(_ID_SEP) if r.strip() in trained]
        return f"Roles: {'; '.join(names[r] for r in own)}." if own else ""

    return (text + _TEXT_JOIN + entries["role_ids"].fillna("").map(role_line)).str.lower()


def _ontology_fields(
    spec: OntologyFields, entries: pd.DataFrame, split: pd.Series, data_root: Path
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """The class graph and the functional parents' classes of every molecule, and the held-out parents' table."""
    release = load_release(spec.ontology, data_root)
    proteins = load_eval_proteins(spec.eval_proteins, data_root)
    live = {t: e for t, e in release.entries.items() if not e.obsolete}
    train = set(split.index[split == "train"])
    held_out = set(entries.index) - train

    def name(term: str) -> str:
        return chebi_name(live[term].name)

    def keep(items: list[str]) -> tuple[str, ...] | None:
        return tuple(dict.fromkeys(i for i in items if not proteins.blocks(i))) or None

    def graph(direct: list[str]) -> tuple[str, ...] | None:
        edges: list[str] = []
        seen: set[str] = set()
        queue = [c for c in direct if c in live]
        while queue:
            term = queue.pop(0)
            if term in seen:
                continue
            seen.add(term)
            for parent in live[term].is_a:
                if parent in live:
                    edges.append(f"{name(term)}{spec.arrow}{name(parent)}")
                    queue.append(parent)
        return keep(edges)

    def parent_classes(parents: list[str]) -> tuple[str, ...] | None:
        shown = [p for p in parents if p in live and p not in held_out]
        return keep([f"{name(p)}{spec.arrow}{name(c)}" for p in shown for c in live[p].is_a if c in live])

    parents = entries["functional_parent_ids"].map(_id_list)
    names = {p: chebi_name(e.name) for p, e in release.entries.items() if p in held_out}
    held = parents.map(lambda ps: [names.get(p, p) for p in ps if p in held_out])
    held = held[held.map(len) > 0]
    fields = {
        spec.class_graph: entries["class_ids"].map(lambda v: graph(_id_list(v))),
        spec.functional_parent_classes: parents.map(parent_classes),
    }
    return fields, pd.DataFrame({"held_out": held.map(tuple)}, index=held.index)


def _id_list(joined: object) -> list[str]:
    """The ids of a ``;``-joined cell, in order (none for a missing cell)."""
    return [i.strip() for i in joined.split(_ID_SEP) if i.strip()] if isinstance(joined, str) else []


def _pattern(patterns: pd.DataFrame, task: str) -> str:
    found = patterns["stated_pattern"].get(task)
    if not isinstance(found, str) or not found:
        raise ValueError(f"task {task!r} is marked has_pattern but the pattern table has no stated_pattern for it")
    return found


def _id_set(joined: str) -> frozenset[str]:
    return frozenset(i.strip() for i in joined.split(_ID_SEP) if i.strip())


def _imported_tasks(items: pd.DataFrame, imports: tuple[TaskImport, ...]) -> tuple[list[TaskSpec], list[str]]:
    """The imported groups' Yes/No tasks in the item table's order, and the ids of the tasks marked ``has_pattern``."""
    out, marked = [], []
    for spec in imports:
        rows = items[items["group"] == spec.item_group]
        if rows.empty:
            raise ValueError(f"the item table has no task in group {spec.item_group!r}")
        varying = rows.groupby("task", sort=False)[["question", "key", "has_pattern"]].nunique()
        bad = varying.index[(varying > 1).any(axis=1)]
        if len(bad):
            raise ValueError(f"group {spec.item_group!r}: question, key or has_pattern varies within tasks {list(bad)}")
        first = rows.drop_duplicates("task")[["task", "question", "key", "has_pattern"]]
        unknown = sorted(set(spec.implied) - set(first["task"]))
        if unknown:
            raise ValueError(f"group {spec.item_group!r}: implied names tasks it does not have: {unknown}")
        for task, question, key, has_pattern in first.itertuples(index=False):
            if not str(key).startswith(_KEY_KINDS):
                raise ValueError(f"task {task!r}: label key {key!r} is of none of the kinds {_KEY_KINDS}")
            out.append(
                TaskSpec(
                    id=str(task),
                    kind=TaskKind.YES_NO,
                    question=str(question),
                    label=str(key),
                    scope=TaskScope.IMPLIED if task in spec.implied else spec.scope,
                    group=spec.group,
                )
            )
            if bool(has_pattern):
                marked.append(str(task))
    return out, marked
