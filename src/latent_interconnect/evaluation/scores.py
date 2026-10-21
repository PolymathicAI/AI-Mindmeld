"""The score table and the number table: the long-format schemas suites and scans write.

**Score table** (artifact kind ``scores``, one file ``scores.parquet``). One row per (item, task, condition
combination). Per-role columns are named ``<column>.<role>`` (e.g. ``object.protein``, ``filler.molecule``,
``condition.a``), so single-object, composition and cross-domain suites share one shape. Generation suites add the
generation columns, so the trace cap and completion are always next to the score; likelihood
suites add the target length, so token-weighted means can be formed; an injection scan's table adds the scan columns,
the written layer and the written state's source, so all its layers are one table
(``evaluation.layer_selection.injection_scan``). A Yes/No readout adds the answer columns
(:data:`ANSWER_COLUMNS`): the two answer log-probabilities whose difference is the score, so the answer mass
``exp(logp_yes) + exp(logp_no)`` can be read from the record.

The answer columns are an additive change of the format. The score column and every other column are what they were.
A table written before the columns existed does not have them and reads as before; a reader that needs them must say
so, and no reader here does. They are values read, never row keys: readers that match or order rows by a table's
identity columns leave them out (``evaluation.drift``, ``evaluation.excerpts``). Which tables have them follows from
the code that wrote them, the commit in the record's provenance: every Yes/No table (``yes_no``, ``composition``
with the Yes/No readout, ``cross_domain``, and the scans' score tables, which run ``yes_no``) written by code that
defines :data:`ANSWER_COLUMNS`, and none written by earlier code. Nothing stored is rewritten or rescored. Running an
already-run Yes/No evaluation again with this code gives a table with the same scores and other file bytes, so
another artifact digest: under a pin name that already names the earlier artifact that is a declared re-pin
(``Workspace.pin_all``), never an automatic one. Code older than the columns is not claimed to read the new tables.

**Number table** (artifact kind ``numbers``, one file ``numbers.parquet``). Per-set numbers: a unique ``key`` per row, a
float ``value``, descriptive key columns (``layer``, ``task``, ``condition.object``, ...) that analyses filter on
(:func:`select_rows`), and float companions such as an interval's ``lo`` / ``hi``. Scans write it.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import canonical_json

__all__ = [
    "ANSWER_COLUMNS",
    "CUT_COLUMNS",
    "GENERATION_COLUMNS",
    "LIKELIHOOD_COLUMNS",
    "NUMBERS_KIND",
    "ROLE_COLUMNS",
    "SCAN_COLUMNS",
    "SCORES_KIND",
    "SCORE_COLUMNS",
    "SET_SCAN_COLUMNS",
    "commit_table",
    "read_numbers",
    "read_scores",
    "read_table",
    "score_roles",
    "select_rows",
    "write_numbers",
    "write_scores",
]

SCORES_KIND = "scores"
"""Artifact kind of a score table."""
NUMBERS_KIND = "numbers"
"""Artifact kind of a number table."""
_SCORES_FILE = "scores.parquet"
_NUMBERS_FILE = "numbers.parquet"

SCORE_COLUMNS: dict[str, str] = {
    "suite": "string",  # SuiteSpec digest
    "step": "Int64",  # training step of the checkpoints (NA: none, a suite that reads no connector)
    "task": "string",  # task id
    "item": "int64",  # item index in the item set
    "score": "float64",  # log p(Yes) - log p(No), or mean target NLL
    "label": "float64",  # the task label for the item (NaN = unknown)
    "batch": "int64",  # index of the forward batch in the batch plan
}
"""Columns of every score table, with their pandas dtypes."""

ROLE_COLUMNS: dict[str, str] = {
    "object": "string",  # the item's own object for the role
    "filler": "string",  # the object actually written (own, or the mismatched one in likelihood baselines)
    "condition": "string",  # data.suites.Condition
    "connector": "string",  # checkpoint digest (NA for name / placeholder / value)
}
"""Per-role columns, written once per role as ``<column>.<role>``."""

GENERATION_COLUMNS: dict[str, str] = {
    "text": "string",  # the generated continuation
    "n_tokens": "int64",  # its length
    "finished": "bool",  # ended on EOS / stop, not on the cap
    "max_new_tokens": "int64",  # the cap in force
}
"""Additional columns of generation suites."""

CUT_COLUMNS: dict[str, str] = {
    "n_tokens_kept": "int64",  # the trace ids kept before the forced answer
    "cut_rule": "string",  # none / last_sentence / fallback / answer_line (evaluation.trace_cut.CutRule)
    "n_dropped": "int64",  # n_tokens - n_tokens_kept
}
"""Additional columns of a post-reasoning readout that records its cut (``runners.YesNoParams.cut``)."""

LIKELIHOOD_COLUMNS: dict[str, str] = {
    "n_target": "int64",  # number of target tokens the mean NLL averages over
}
"""Additional columns of likelihood suites (``data.suites.SuiteSpec.likelihood``)."""

SCAN_COLUMNS: dict[str, str] = {
    "layer": "int64",  # the receiver layer the state was written at
    "source": "string",  # the written state's source (WrittenScan.source: oracle, ridge)
}
"""Additional columns of injection-scan score tables (both or neither)."""

SET_SCAN_COLUMNS: dict[str, str] = {
    "layer_set": "string",  # the candidate write set (LayerSet.name)
    "source": "string",  # the written states' source (WrittenScan.source)
}
"""Additional columns of layer-set comparison score tables (both or neither; never with ``layer``)."""

ANSWER_COLUMNS: dict[str, str] = {
    "logp_yes": "float64",  # log p(yes) of the next token at the read position, under the full softmax
    "logp_no": "float64",  # log p(no), likewise; score = logp_yes - logp_no
}
"""Additional columns of Yes/No readouts (both or neither): the backend's two answer log-probabilities, as read."""

_OPTIONAL = {
    **GENERATION_COLUMNS,
    **CUT_COLUMNS,
    **LIKELIHOOD_COLUMNS,
    **SCAN_COLUMNS,
    **SET_SCAN_COLUMNS,
    **ANSWER_COLUMNS,
}


def score_roles(table: pd.DataFrame) -> tuple[str, ...]:
    """Return the roles of a score table (from its ``object.<role>`` columns), sorted."""
    return tuple(sorted(c.split(".", 1)[1] for c in table.columns if c.startswith("object.")))


def _schema(table: pd.DataFrame) -> dict[str, str]:
    roles = score_roles(table)
    if not roles:
        raise ValueError("a score table needs at least one object.<role> column")
    schema = dict(SCORE_COLUMNS)
    schema |= {f"{c}.{r}": t for r in roles for c, t in ROLE_COLUMNS.items()}
    schema |= {c: t for c, t in _OPTIONAL.items() if c in table.columns}
    if "layer" in table.columns and "layer_set" in table.columns:
        raise ValueError("a score table is an injection scan's (layer) or a set comparison's (layer_set), not both")
    if "layer_set" in table.columns:
        schema |= SET_SCAN_COLUMNS  # a set comparison's rows name both their set and their source
    elif any(c in table.columns for c in SCAN_COLUMNS):
        schema |= SCAN_COLUMNS  # a scan's rows name both their layer and their source
    if any(c in table.columns for c in ANSWER_COLUMNS):
        schema |= ANSWER_COLUMNS  # a Yes/No row keeps both answer log-probabilities
    return schema


def _check_columns(table: pd.DataFrame, schema: Mapping[str, str], what: str) -> pd.DataFrame:
    missing = sorted(set(schema) - set(table.columns))
    unknown = sorted(set(table.columns) - set(schema))
    if missing or unknown:
        raise ValueError(f"{what}: missing columns {missing}, unknown columns {unknown}")
    return table[list(schema)].astype(dict(schema)).reset_index(drop=True)


def write_scores(store: ArtifactStore, table: pd.DataFrame, meta: Mapping[str, Any]) -> ArtifactRef:
    """Validate a score table against the schema and commit it as a ``scores`` artifact.

    Args:
        store: the artifact store.
        table: the rows; columns exactly the schema for its roles (plus the generation, likelihood, scan or answer
            columns).
        meta: deterministic metadata (e.g. the suite digest and step); part of the identity.

    Raises:
        ValueError: on a missing or unknown column, or a score that is not finite (a NaN is a bug).
    """
    table = _check_columns(table, _schema(table), "score table")
    bad = ~np.isfinite(table["score"].to_numpy(np.float64))
    if bad.any():
        raise ValueError(
            f"score table: {int(bad.sum())} scores are not finite, e.g. rows {table.index[bad][:5].tolist()}"
        )
    return commit_table(store, table, SCORES_KIND, _SCORES_FILE, meta)


def read_scores(store: ArtifactStore, ref: ArtifactRef) -> pd.DataFrame:
    """Return the score table of a ``scores`` artifact (manifest checked)."""
    return read_table(store, ref, SCORES_KIND, _SCORES_FILE)


def write_numbers(store: ArtifactStore, table: pd.DataFrame, meta: Mapping[str, Any]) -> ArtifactRef:
    """Validate a number table and commit it as a ``numbers`` artifact (rows sorted by key).

    Args:
        store: the artifact store.
        table: a unique string ``key`` per row, a float ``value``, string or integer key columns, and float
            companion columns (e.g. an interval's ``lo`` and ``hi``).
        meta: deterministic metadata; part of the identity.

    Raises:
        ValueError: if ``key`` or ``value`` is missing, a key repeats, a column is not string, integer or float, or a
            row's ``value``, ``lo`` or ``hi`` is not finite (a NaN is a bug upstream, never a number to record).
    """
    if not {"key", "value"} <= set(table.columns):
        raise ValueError(f"number table: needs 'key' and 'value', has {sorted(table.columns)}")
    if table["key"].duplicated().any():
        raise ValueError(f"number table: repeated keys {sorted(set(table['key'][table['key'].duplicated()]))[:5]}")
    for c in ("value", "lo", "hi"):
        if c in table.columns:
            bad = ~np.isfinite(pd.to_numeric(table[c], errors="coerce").to_numpy(np.float64))
            if bad.any():
                raise ValueError(f"number table: {c!r} is not finite at keys {sorted(table['key'][bad])[:5]}")
    types: dict[str, str] = {"key": "string", "value": "float64"}
    for c in table.columns:
        if c in types:
            continue
        if pd.api.types.is_integer_dtype(table[c]):
            types[c] = "int64"
        elif pd.api.types.is_float_dtype(table[c]):
            types[c] = "float64"
        elif pd.api.types.is_string_dtype(table[c]) or pd.api.types.is_object_dtype(table[c]):
            types[c] = "string"
        else:
            raise ValueError(f"number table: column {c!r} has dtype {table[c].dtype}")
    out = table.astype(types).sort_values("key").reset_index(drop=True)
    return commit_table(store, out, NUMBERS_KIND, _NUMBERS_FILE, meta)


def read_numbers(store: ArtifactStore, ref: ArtifactRef) -> pd.DataFrame:
    """Return the number table of a ``numbers`` artifact (manifest checked)."""
    return read_table(store, ref, NUMBERS_KIND, _NUMBERS_FILE)


def commit_table(
    store: ArtifactStore, table: pd.DataFrame, kind: str, name: str, meta: Mapping[str, Any]
) -> ArtifactRef:
    """Commit one table as the parquet file ``name`` of a ``kind`` artifact (the one table writer).

    The staging key hashes the body and the metadata, so the same table and metadata give the same artifact.
    """
    buf = io.BytesIO()
    table.to_parquet(buf, index=False)
    body = buf.getvalue()
    key = hashlib.sha256(body + canonical_json(dict(meta)).encode()).hexdigest()
    staged = store.staging(f"{kind}-{key}")
    (staged / name).write_bytes(body)
    return store.commit(staged, kind, dict(meta))


def read_table(store: ArtifactStore, ref: ArtifactRef, kind: str, name: str) -> pd.DataFrame:
    """Return the table a :func:`commit_table` artifact holds (its kind and manifest checked).

    Raises:
        ValueError: if ``ref`` is not a ``kind`` artifact.
    """
    if ref.kind != kind:
        raise ValueError(f"{ref.short} is a {ref.kind!r} artifact, not {kind!r}")
    store.manifest(ref)
    return pd.read_parquet(store.path(ref) / name)


def select_rows(table: pd.DataFrame, where: Mapping[str, Any], what: str) -> pd.DataFrame:
    """Return the rows meeting every condition of ``where`` (column → value, or a list of accepted values).

    Raises:
        KeyError: naming ``what`` if a filter column is not in the table.
    """
    for column, value in where.items():
        if column not in table.columns:
            raise KeyError(f"{what}: filter column {column!r} not in {sorted(table.columns)}")
        table = table[table[column].isin(value)] if isinstance(value, list) else table[table[column] == value]
    return table
