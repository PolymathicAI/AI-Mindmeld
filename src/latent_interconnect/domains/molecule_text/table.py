"""The molecule text table: every molecule of a split rendered from declared sources, as one artifact.

:class:`MoleculeTextSpec` cites every source by a path under the data root and its sha256 (``domains.tables``
:class:`~latent_interconnect.domains.tables.SourceFile`): the ChEBI release, the reviewed UniProt table and the
evaluation-protein candidate tables, and the molecule split. :func:`build_text_table` renders one row per molecule of
the declared splits (``fields.MoleculeRenderer``) and adds the trimmed describe forms of the paper's field table
(with a cap of 48 tokens):

- ``describe_def``: the definition cut at a sentence end within ``cap_tokens`` receiver tokens
  (``sentences.trim``);
- ``describe``: the roles-first describe text cut the same way, kept whole when only its roles line would remain
  (``sentences.trim_roles_first``).

The paper's split rule is kept as data: every row carries the trained fields; the held-out fields (``formula``,
``heldout_roles``) are filled only on rows of the ``held_out`` splits, so no held-out text can become a training target.
The table's columns and their order are the paper's field table's, so the molecule
domain reads either one as its ``text`` source (``domains.molecule.MoleculeDataSpec``).

:func:`commit_text_table` writes it as an artifact of kind :data:`MOLECULE_TEXT_KIND`: ``text.parquet``, with the
spec's digest, the sources' hashes, the release version, the row counts per split and field, and the rules' counts in
its metadata.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import StageConfig
from latent_interconnect.domains.chebi import ChebiRelease, load_release
from latent_interconnect.domains.eval_proteins import EvalProteins, load_eval_proteins
from latent_interconnect.domains.molecule_text.classes import ClassPartition
from latent_interconnect.domains.molecule_text.definitions import clean_text, name_of
from latent_interconnect.domains.molecule_text.fields import FIELDS, HELD_OUT_FIELDS, MoleculeRenderer
from latent_interconnect.domains.molecule_text.roles import RoleSplit, ancestors
from latent_interconnect.domains.molecule_text.sentences import TokenCounter, own_name_rewrite, trim, trim_roles_first
from latent_interconnect.domains.tables import SourceFile, read_source
from latent_interconnect.receivers.base import ReceiverSpec

__all__ = [
    "CHECKS",
    "COLUMNS",
    "MOLECULE_TEXT_KIND",
    "TEXT_FILE",
    "MoleculeTextSpec",
    "TextTable",
    "build_text_table",
    "check_text_table",
    "clear_held_out",
    "commit_text_table",
]

MOLECULE_TEXT_KIND = "molecule_text"
"""Artifact kind of a molecule text table."""
TEXT_FILE = "text.parquet"
"""The table's file in the artifact."""
COLUMNS = ("chebi_id", "split", "describe", "describe_def", *(f for f in FIELDS if f != "describe"))
"""The table's columns, in the paper's field-table order."""

_TAG = re.compile(r"</?[a-z]+>")
_EC = re.compile(r"\bEC \d+\.")
_NAMED_CONJUGATE = re.compile(r"(?:^|\n)(?:Conjugate (?:acid|base) \(|Its conjugate (?:acid|base) \()")
_ROLE_LINE = re.compile(r"^Roles?: (.*)\.$")
CHECKS = (
    "duplicate_id",
    "format",
    "eval_protein_or_ec_left",
    "heldout_or_ec_role_in_trained",
    "describe_presence",
    "describe_def_prefix",
    "own_name_left",
    "conjugate_name_in_label",
)
"""The table's checks (:func:`check_text_table`), named as the paper's builders named them where they had a name:
ids unique; every text without HTML tags or double spaces, ending in ``.``, with no empty item; no evaluation protein
or EC number in any field but ``formula``; no held-out or EC role among the trained roles; ``describe`` and
``describe_def`` present exactly with ``definition``; ``describe_def`` a prefix of it; no rewritable own-name mention
left in ``definition``; no conjugate label naming the conjugate."""


class MoleculeTextSpec(StageConfig):
    """One molecule text table: its sources, its rows and the trim.

    Attributes:
        ontology: the ChEBI OBO release (gzipped OBO).
        proteins: the evaluation proteins' sources (``domains.eval_proteins``), whose names, gene symbols and EC
            numbers no training text may state.
        split: the molecule split (``chebi_id``, ``split``); it must hold every molecule the release covers
            (``domains.chebi.ChebiEntry.in_universe``).
        rows: the splits whose molecules get a row.
        held_out: the splits whose rows carry the held-out fields; a subset of ``rows``.
        receiver: the receiver whose tokens the trim counts.
        cap_tokens: the trim's cap in receiver tokens.
    """

    ontology: SourceFile
    proteins: EvalProteins
    split: SourceFile
    rows: tuple[str, ...] = Field(min_length=1)
    held_out: tuple[str, ...]
    receiver: ReceiverSpec
    cap_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def _check(self) -> MoleculeTextSpec:
        stray = sorted(set(self.held_out) - set(self.rows))
        if stray:
            raise ValueError(f"held_out splits {stray} are not among the rows {list(self.rows)}")
        return self

    def sources(self) -> dict[str, SourceFile]:
        """Return the source files by role, in a fixed order."""
        out = {"ontology": self.ontology, "uniprot": self.proteins.uniprot, "split": self.split}
        out |= {f"candidates/{i}": c for i, c in enumerate(self.proteins.candidates)}
        return out


class TextTable:
    """A built table and what its rules did.

    Args:
        table: the rows (:data:`COLUMNS`, sorted by ``chebi_id``).
        release: the ChEBI release version.
        counts: the rules' counts (removed sentences and items, own-name mentions by kind, ...).
    """

    def __init__(self, table: pd.DataFrame, release: str, counts: Mapping[str, int]) -> None:
        self.table, self.release, self.counts = table, release, dict(sorted(counts.items()))


def _renderer(spec: MoleculeTextSpec, data_root: Path) -> tuple[ChebiRelease, MoleculeRenderer]:
    release = load_release(spec.ontology, data_root)
    proteins = load_eval_proteins(spec.proteins, data_root)
    is_a = {t: e.is_a for t, e in release.entries.items()}
    ancestry = ancestors(is_a)
    return release, MoleculeRenderer(
        release,
        proteins,
        RoleSplit(is_a, lambda t: name_of(release, t), ancestry),
        ClassPartition(release.entries, ancestry, clean_text),
    )


def build_text_table(spec: MoleculeTextSpec, data_root: Path, counter: TokenCounter) -> TextTable:
    """Render the table (the module docstring's rules) and check it (:func:`check_text_table`).

    Args:
        spec: the table's spec.
        data_root: the data root the sources' paths are under (``Paths.data``).
        counter: the receiver's renderer (``receivers.open_renderer`` of ``spec.receiver``), which counts the trim's
            tokens.

    Raises:
        IntegrityError: if a source's sha256 differs from the spec.
        ValueError: if a molecule the release covers is not in the split, a row's molecule is not one it covers, or
            the table fails a check.
    """
    split = pd.read_parquet(read_source(data_root, spec.split), columns=["chebi_id", "split"]).set_index("chebi_id")
    release, renderer = _renderer(spec, data_root)
    universe = {t for t, e in release.entries.items() if e.in_universe()}
    unsplit = sorted(universe - set(split.index))
    if unsplit:
        raise ValueError(
            f"{len(unsplit)} molecules of ChEBI {release.version} are not in the split, e.g. {unsplit[:5]}"
        )
    ids = sorted(split.index[split["split"].isin(spec.rows)])
    outside = [i for i in ids if i not in universe]
    if outside:
        raise ValueError(f"{len(outside)} rows are not 3-star molecules with a structure, e.g. {outside[:5]}")
    table = pd.DataFrame([{"chebi_id": i, "split": split.at[i, "split"], **renderer.render(i)} for i in ids])
    table = _missing_as_none(table.reindex(columns=["chebi_id", "split", *FIELDS]))
    table["describe_def"] = [trim(t, counter, spec.cap_tokens) for t in table["definition"]]
    table["describe"] = [
        trim_roles_first(t, r, counter, spec.cap_tokens)
        for t, r in zip(table["describe"], table["mechanism_roles"], strict=True)
    ]
    table = _missing_as_none(table[list(COLUMNS)])
    check_text_table(table, renderer.proteins.blocks, release)
    return TextTable(clear_held_out(table, spec.held_out).reset_index(drop=True), release.version, renderer.counts)


def clear_held_out(table: pd.DataFrame, held_out: Collection[str]) -> pd.DataFrame:
    """Return a copy of ``table`` whose held-out fields are None on every row outside the ``held_out`` splits."""
    out = table.copy()
    out.loc[~out["split"].isin(list(held_out)), list(HELD_OUT_FIELDS)] = None
    return out


def _missing_as_none(table: pd.DataFrame) -> pd.DataFrame:
    """The table as object columns with None for a missing text (pandas infers NaN-valued string columns)."""
    table = table.astype(object)
    return table.where(table.notna(), None)


def check_text_table(  # noqa: PLR0912  (one branch per check)
    table: pd.DataFrame,
    blocked: Callable[[str], bool],
    release: ChebiRelease,
) -> None:
    """Check a table, before its held-out fields are cleared, against the paper's builders' checks (:data:`CHECKS`).

    Args:
        table: the table (:data:`COLUMNS`).
        blocked: the item rule's test (``domains.eval_proteins.EvalProteinMatcher.blocks``).
        release: the ChEBI release (for the own names).

    Raises:
        ValueError: listing every failed check with its count and first ids.
    """
    failed: dict[str, set[str]] = {}

    def fail(check: str, chebi_id: str) -> None:
        failed.setdefault(check, set()).add(chebi_id)

    for chebi_id in table["chebi_id"][table["chebi_id"].duplicated()]:
        fail("duplicate_id", chebi_id)
    for row in table.to_dict("records"):
        chebi_id = row["chebi_id"]
        for field in COLUMNS[2:]:
            text = row[field]
            if text is None:
                continue
            if _TAG.search(text) or "  " in text or not text.endswith(".") or "; ." in text or ": ." in text:
                fail("format", chebi_id)
            if field != "formula" and blocked(text):
                fail("eval_protein_or_ec_left", chebi_id)
        trained = set(_items(row["mechanism_roles"]))
        if trained & set(_items(row["heldout_roles"])) or any(_EC.search(x) for x in trained):
            fail("heldout_or_ec_role_in_trained", chebi_id)
        definition = row["definition"]
        if (row["describe"] is None) != (definition is None) or (row["describe_def"] is None) != (definition is None):
            fail("describe_presence", chebi_id)
        if definition is None:
            continue
        if row["describe_def"] is not None and not definition.startswith(row["describe_def"]):
            fail("describe_def_prefix", chebi_id)
        if own_name_rewrite(definition, name_of(release, chebi_id))[1]["rewritten"]:
            fail("own_name_left", chebi_id)
        if _NAMED_CONJUGATE.search(definition):
            fail("conjugate_name_in_label", chebi_id)
    problems = [f"{check}: {len(ids)} (e.g. {sorted(ids)[:3]})" for check in CHECKS if (ids := failed.get(check))]
    if problems:
        raise ValueError(f"the molecule text table fails its checks: {'; '.join(problems)}")


def _items(line: str | None) -> list[str]:
    m = _ROLE_LINE.match(line or "")
    return m.group(1).split("; ") if m else []


def commit_text_table(store: ArtifactStore, spec: MoleculeTextSpec, built: TextTable) -> ArtifactRef:
    """Write the table as a :data:`MOLECULE_TEXT_KIND` artifact (the module docstring's metadata) and commit it."""
    staged = store.staging(f"molecule-text-{uuid.uuid4().hex}")
    built.table.to_parquet(staged / TEXT_FILE, index=False)
    table = built.table
    per_split: dict[str, Any] = {
        str(s): {"rows": len(g), **{c: int(g[c].notna().sum()) for c in COLUMNS[2:]}}
        for s, g in table.groupby("split", sort=True)
    }
    meta = {
        "spec": spec.digest(),
        "sources": {k: s.sha256 for k, s in spec.sources().items()},
        "release": built.release,
        "counts": per_split,
        "rules": built.counts,
    }
    return store.commit(staged, MOLECULE_TEXT_KIND, meta)
