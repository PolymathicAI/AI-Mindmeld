"""The screened training text: the only text streams may train on.

A ``training_set`` artifact (built by ``build.zero_shot.build_training_set``) holds the training objects and, per
object, the text fields that survived the zero-shot screen, with the removal counts per task and field in its
metadata. Streams receive it as :class:`TrainingText`; they never read domain fields directly, so unscreened text
cannot reach a target. Captures read their recipe's context from it too (``build.capture``). An all-tasks
training set (``build.zero_shot.TrainingSetSpec.all_tasks``) screened nothing; its text says so
(:attr:`TrainingText.all_tasks`), and the streams built on it may train every task (``data.streams.build_stream``).

Artifact format (kind :data:`TRAINING_SET_KIND`): ``objects.parquet`` holds ``object_id`` in order, and
``fields.parquet`` the kept texts in long form (``object_id``, ``field``, ``text``; a removed field has no row), so
a reader may load only the fields it needs; the manifest metadata holds ``removed``, ``all_tasks`` (only when true:
an artifact without it screened its zero-shot tasks) and the builder's own entries.
Write it with :func:`commit_training_text`, read it with :func:`load_training_text` (its objects alone:
:func:`training_objects`).
"""

from __future__ import annotations

import uuid
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore

__all__ = ["TRAINING_SET_KIND", "TrainingText", "commit_training_text", "load_training_text", "training_objects"]

TRAINING_SET_KIND = "training_set"
"""Artifact kind of the screened training text."""
_OBJECTS = "objects.parquet"
_FIELDS = "fields.parquet"
_ALL_TASKS = "all_tasks"


@dataclass(frozen=True)
class TrainingText:
    """Screened training text.

    Attributes:
        object_ids: the training objects, in a fixed order.
        fields: object id → field name → kept text (a removed field is absent).
        removed: zero-shot task → field → number of removed fields (the honesty statement's counts).
        all_tasks: True when the training set holds no task out (nothing screened), so its streams may train every
            task of the domain's catalog; False when its domain's held-out tasks are zero-shot.
    """

    object_ids: tuple[str, ...]
    fields: Mapping[str, Mapping[str, str]]
    removed: Mapping[str, Mapping[str, int]]
    all_tasks: bool = False


def commit_training_text(store: ArtifactStore, text: TrainingText, meta: Mapping[str, Any]) -> ArtifactRef:
    """Commit screened training text as a :data:`TRAINING_SET_KIND` artifact.

    Raises:
        ValueError: if ``meta`` sets ``removed`` (it is taken from ``text``), or a field names an object that is not
            in ``object_ids``.
    """
    if {"removed", _ALL_TASKS} & set(meta):
        raise ValueError(f"meta must not set 'removed' or {_ALL_TASKS!r}; they come from the training text")
    stray = sorted(set(text.fields) - set(text.object_ids))
    if stray:
        raise ValueError(f"fields of objects not in the training set: {stray[:5]}")
    staged = store.staging(f"training-set-{uuid.uuid4().hex}")
    pd.DataFrame({"object_id": list(text.object_ids)}).to_parquet(staged / _OBJECTS, index=False)
    rows = [(o, f, t) for o in text.object_ids for f, t in sorted(text.fields.get(o, {}).items())]
    pd.DataFrame(rows, columns=["object_id", "field", "text"]).to_parquet(staged / _FIELDS, index=False)
    removed = {t: dict(c) for t, c in text.removed.items()}
    all_tasks = {_ALL_TASKS: True} if text.all_tasks else {}  # absent when false: a screened set's digest is without it
    return store.commit(staged, TRAINING_SET_KIND, {**meta, "removed": removed, **all_tasks})


def load_training_text(store: ArtifactStore, ref: ArtifactRef, fields: Collection[str] | None = None) -> TrainingText:
    """Load a committed ``training_set`` artifact (its manifest checked against ``ref``).

    Args:
        store: the artifact store.
        ref: the training set.
        fields: load only these fields (a parquet filter; e.g. a capture recipe's context field); None loads all.

    Raises:
        ValueError: if ``ref`` is not a training set.
    """
    if ref.kind != TRAINING_SET_KIND:
        raise ValueError(f"{ref.short} is a {ref.kind!r} artifact, not a training set")
    manifest = store.manifest(ref)
    ids = tuple(str(o) for o in pd.read_parquet(store.path(ref) / _OBJECTS)["object_id"])
    filters = None if fields is None else [("field", "in", sorted(fields))]
    long = pd.read_parquet(store.path(ref) / _FIELDS, filters=filters)
    kept: dict[str, dict[str, str]] = {}
    for o, f, t in zip(long["object_id"], long["field"], long["text"], strict=True):
        kept.setdefault(str(o), {})[str(f)] = str(t)
    return TrainingText(ids, kept, manifest.meta["removed"], all_tasks=bool(manifest.meta.get(_ALL_TASKS, False)))


def training_objects(store: ArtifactStore, ref: ArtifactRef) -> tuple[str, ...]:
    """Return a committed ``training_set`` artifact's objects, in order, without its text (manifest checked).

    Raises:
        ValueError: if ``ref`` is not a training set.
    """
    if ref.kind != TRAINING_SET_KIND:
        raise ValueError(f"{ref.short} is a {ref.kind!r} artifact, not a training set")
    store.manifest(ref)
    return tuple(str(o) for o in pd.read_parquet(store.path(ref) / _OBJECTS)["object_id"])
