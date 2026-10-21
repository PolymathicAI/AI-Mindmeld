"""The generations table: what a generation suite writes (artifact kind ``generations``, one file).

One row per (item, task, condition combination), in the suite's prompt order. A generation has no score, so it is not
a score table; a scoring suite that reads a generation record (``evaluation.suites.SuiteRunConfig.upstream``) writes
the score table, and copies each trace's length, completion and cap into its rows (``scores.GENERATION_COLUMNS``), so
the cap and completion stay next to every score.

Columns: the score table's keys (``suite``, ``step``, ``task``, ``item``) and per-role columns
(``scores.ROLE_COLUMNS``, as ``<column>.<role>``), then:

==================  ========  ==========================================================================================
``prompt``          string    :func:`prompt_digest` of the rendered prompt's ids: a reader re-renders the prompt and
                              refuses a trace whose prompt differs (``evaluation.suites.Upstream.check_prompt``)
``text``            string    the decoded continuation (``receivers.base.Generation.text``)
``token_ids``       int list  its token ids, exactly as generated (a reader's token-id prefill, ``render.TokenIds``)
``n_tokens``        int64     ``len(token_ids)``
``finished``        bool      ended on end-of-sequence or a stop string, not on the cap
``max_new_tokens``  int64     the cap in force
``answer``          string    the kind's reading of the text (``name_guess``: the elicited name), "" if it reads none
==================  ========  ==========================================================================================

The artifact's metadata holds the suite digest, the step, the generation parameters and the generation plan
(``receivers.base.GenerationPlan``), both part of the suite run's identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import canonical_sha256
from latent_interconnect.evaluation.scores import ROLE_COLUMNS, commit_table, read_table, score_roles

__all__ = ["GENERATIONS_COLUMNS", "GENERATIONS_KIND", "prompt_digest", "read_generations", "write_generations"]

GENERATIONS_KIND = "generations"
"""Artifact kind of a generations table."""
_FILE = "generations.parquet"

GENERATIONS_COLUMNS: dict[str, str] = {
    "suite": "string",
    "step": "Int64",
    "task": "string",
    "item": "int64",
    "prompt": "string",
    "text": "string",
    "token_ids": "object",
    "n_tokens": "int64",
    "finished": "bool",
    "max_new_tokens": "int64",
    "answer": "string",
}
"""Columns of every generations table besides the per-role ones, with their pandas dtypes (``token_ids``: int lists)."""


def prompt_digest(input_ids: Sequence[int]) -> str:
    """Return the sha256 of a rendered prompt's ids (canonical JSON of the id list): the ``prompt`` column."""
    return canonical_sha256([int(i) for i in input_ids])


def _check(table: pd.DataFrame) -> pd.DataFrame:
    roles = score_roles(table)
    if not roles:
        raise ValueError("a generations table needs at least one object.<role> column")
    schema = GENERATIONS_COLUMNS | {f"{c}.{r}": t for r in roles for c, t in ROLE_COLUMNS.items()}
    missing, unknown = sorted(set(schema) - set(table.columns)), sorted(set(table.columns) - set(schema))
    if missing or unknown:
        raise ValueError(f"generations table: missing columns {missing}, unknown columns {unknown}")
    out = table[list(schema)].astype(schema).reset_index(drop=True)
    ids = [tuple(int(i) for i in row) for row in out["token_ids"]]
    lengths = pd.Series([len(row) for row in ids], dtype="int64")
    if not lengths.equals(out["n_tokens"]):
        raise ValueError("generations table: n_tokens differs from the length of token_ids")
    over = out.index[out["n_tokens"] > out["max_new_tokens"]].tolist()
    if over:
        raise ValueError(f"generations table: rows {over[:5]} are longer than their cap")
    capped = out.index[~out["finished"] & (out["n_tokens"] != out["max_new_tokens"])].tolist()
    if capped:
        raise ValueError(f"generations table: unfinished rows {capped[:5]} stopped before their cap")
    out["token_ids"] = [list(row) for row in ids]
    return out


def write_generations(store: ArtifactStore, table: pd.DataFrame, meta: Mapping[str, Any]) -> ArtifactRef:
    """Validate a generations table and commit it as a ``generations`` artifact.

    Args:
        store: the artifact store.
        table: the rows; columns exactly :data:`GENERATIONS_COLUMNS` plus the per-role columns.
        meta: deterministic metadata (suite digest, step, generation parameters and plan); part of the identity.

    Raises:
        ValueError: on a missing or unknown column, ``n_tokens`` that is not the length of ``token_ids``, a row longer
            than its cap, or an unfinished row shorter than its cap.
    """
    return commit_table(store, _check(table), GENERATIONS_KIND, _FILE, meta)


def read_generations(store: ArtifactStore, ref: ArtifactRef) -> pd.DataFrame:
    """Return the generations table of a ``generations`` artifact (manifest checked; ``token_ids`` as int tuples).

    Raises:
        ValueError: if ``ref`` is not a ``generations`` artifact.
    """
    table = read_table(store, ref, GENERATIONS_KIND, _FILE)
    table["token_ids"] = [tuple(int(i) for i in row) for row in table["token_ids"]]
    return table
