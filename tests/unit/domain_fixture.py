"""A tiny protein domain-data artifact for the unit tests (every object its own cluster), with its capture recipes."""

import pandas as pd

from latent_interconnect.domains.base import (
    CaptureRecipe,
    DomainSpec,
    Frame,
    RecipeMessage,
    TaskKind,
    TaskScope,
    TaskSpec,
    open_domain,
)
from latent_interconnect.domains.tables import DomainTables, commit_domain_data

FRAME = "Here is an unknown protein, given to you as an embedding:<SLOT:x>. Describe what this protein is."
LEGACY = (
    "Here is an unknown {noun}, given to you as an embedding: {name}. Describe what this {noun} is and what it does."
)
RECIPES = (
    CaptureRecipe(name="name", messages=(RecipeMessage(role="user", text=LEGACY),), read="name"),
    CaptureRecipe(
        name="caption",
        messages=(RecipeMessage(role="user", text=LEGACY), RecipeMessage(role="assistant", text="{context}")),
        read="context",
        context_field="caption",
        context_tokens=4096,
        target="mean",
    ),
    CaptureRecipe(
        name="name_described",
        messages=(RecipeMessage(role="user", text="{context} This is the description of the {noun}: {name}"),),
        read="name",
        context_field="caption",
        context_tokens=64,
    ),
)


def tiny_domain(store, ids, split="train", names=None):
    """Commit domain data over ``ids`` and return (its spec, the opened domain)."""
    index = pd.Index(list(ids), name="object_id")
    names = names or {}
    task = TaskSpec(id="t", kind=TaskKind.YES_NO, question="Yes?", label="y", scope=TaskScope.TRAINED)
    tables = DomainTables(
        domain="protein",
        objects=pd.DataFrame({"value": ["MA"] * len(index), "name": [names.get(i, i) for i in index]}, index=index),
        split=pd.Series(split if isinstance(split, list) else [split] * len(index), index=index),
        clusters=pd.Series([f"c{i}" for i in index], index=index),
        labels=pd.DataFrame({"y": [True] * len(index)}, index=index),
        fields=pd.DataFrame({"caption": ["c"] * len(index)}, index=index),
        tasks=(task,),
        frames=(Frame(id="describe", user=FRAME), Frame(id="ask", user="Is<SLOT:x> it? {question}")),
        recipes=RECIPES,
    )
    spec = DomainSpec(name="protein", data=commit_domain_data(store, tables, {}).sha256)
    return spec, open_domain(spec, store)
