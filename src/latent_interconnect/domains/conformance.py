"""Conformance checks for domains: what any domain, core or third-party, must satisfy. Run by the tests."""

from __future__ import annotations

import pandas as pd

from latent_interconnect.domains.base import QUESTION_MARKER, SLOT_MARKER, Domain, TaskKind

__all__ = ["check_domain"]

_SPLITS = {"train", "val", "test"}


def _check_object_kind(domain: Domain) -> None:
    kind = getattr(type(domain), "object_kind", None)
    if not isinstance(kind, str) or not kind:
        raise AssertionError(f"{domain.name} declares no object_kind (the encoder-recipe domain of its objects)")


def check_domain(domain: Domain) -> None:
    """Raise ``AssertionError`` describing the first way ``domain`` breaks the domain contract.

    Checked: the domain declares its ``object_kind``; every table is indexed by known object ids; splits are train /
    val / test; every task's label column exists in the label table (Yes/No) or is a field (text: a text
    field, or an item-list field whose items the stream words); every field is a text field (``str`` cells) or an
    item-list field (``tuple`` of ``str`` cells), missing values None; a question is bare (no slot marker; the frame
    places it); every task's scope mask is a bool series over the objects; every capture recipe is filed under its name
    and its context field is a text field. A frame's markers are checked when it is built (:class:`Frame`).
    """
    _check_object_kind(domain)
    ids = set(domain.objects().index)
    for name, table in (
        ("split", domain.split()),
        ("clusters", domain.clusters()),
        ("labels", domain.labels()),
        ("fields", domain.fields()),
    ):
        extra = set(table.index) - ids
        if extra:
            raise AssertionError(f"{domain.name}.{name}() has unknown object ids {sorted(extra)[:5]}")
    bad = set(domain.split().unique()) - _SPLITS
    if bad:
        raise AssertionError(f"{domain.name}.split() has values {sorted(bad)}; expected {sorted(_SPLITS)}")
    labels, fields = set(domain.labels().columns), _text_fields(domain)
    for tid, task in domain.tasks().items():
        source = set(domain.fields().columns) if task.kind is TaskKind.TEXT else labels
        if task.label not in source:
            raise AssertionError(f"{domain.name} task {tid!r}: label column {task.label!r} not found")
        if SLOT_MARKER.search(task.question) or QUESTION_MARKER in task.question:
            raise AssertionError(f"{domain.name} task {tid!r}: the question must be bare (the frame places it)")
        mask = domain.scope_mask(task, domain.objects().index)
        if not (mask.index.equals(domain.objects().index) and pd.api.types.is_bool_dtype(mask)):
            raise AssertionError(f"{domain.name} task {tid!r}: scope_mask must be a bool series over the given objects")
    for rid, recipe in domain.capture_recipes().items():
        if rid != recipe.name:
            raise AssertionError(f"{domain.name} recipe {rid!r} is named {recipe.name!r}")
        if recipe.context_field is not None and recipe.context_field not in fields:
            raise AssertionError(f"{domain.name} recipe {rid!r}: context {recipe.context_field!r} is no text field")


def _text_fields(domain: Domain) -> set[str]:
    """The text fields; raise unless every other field is an item-list field."""
    text = set()
    for column, values in domain.fields().items():
        present = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]
        if all(isinstance(v, str) for v in present):
            text.add(str(column))
        elif not all(isinstance(v, tuple) and all(isinstance(i, str) for i in v) for v in present):
            raise AssertionError(f"{domain.name} field {column!r}: cells must all be str or all be tuples of str")
    return text
