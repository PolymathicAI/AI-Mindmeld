"""Extension path 2a: a new object domain (``example-rna``). Its encoder is in ``examples/kmer_extractor.py``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from latent_interconnect.domains.base import (
    DOMAINS,
    CaptureRecipe,
    Domain,
    Frame,
    RecipeMessage,
    TaskKind,
    TaskScope,
    TaskSpec,
)


@DOMAINS.register("example-rna")
class RNA(Domain):
    """A three-object RNA domain held in memory (a real domain reads its data artifact at ``data_root``)."""

    name, noun = "example-rna", "RNA"
    object_kind = "example-rna"

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root
        self._objects = pd.DataFrame(
            {"value": ["ACGUACGU", "GGGCCCAU", "UUUUACGA"], "name": ["r1", "r2", "r3"]},
            index=pd.Index(["r1", "r2", "r3"], name="object_id"),
        )

    def objects(self) -> pd.DataFrame:
        """Return the object table."""
        return self._objects

    def split(self) -> pd.Series:
        """Return the split."""
        return pd.Series(["train", "train", "test"], index=self._objects.index)

    def clusters(self) -> pd.Series:
        """Return one cluster per object."""
        return pd.Series(range(3), index=self._objects.index)

    def labels(self) -> pd.DataFrame:
        """Return a GC-rich label."""
        gc = self._objects["value"].map(lambda s: (s.count("G") + s.count("C")) / len(s) > 0.5)
        return pd.DataFrame({"gc_rich": gc})

    def fields(self) -> pd.DataFrame:
        """Return a description field."""
        return pd.DataFrame({"description": ["a short RNA"] * 3}, index=self._objects.index)

    def tasks(self) -> Mapping[str, TaskSpec]:
        """Return one Yes/No task."""
        q = "Is this RNA GC-rich?"
        return {"gc": TaskSpec(id="gc", kind=TaskKind.YES_NO, question=q, label="gc_rich", scope=TaskScope.HELD_OUT)}

    def frames(self) -> Mapping[str, Frame]:
        """Return the carrier frame and the Yes/No frame (which places a task's bare question)."""
        return {
            "carrier": Frame(id="carrier", user="Tell me about the RNA<SLOT:x>."),
            "yes_no": Frame(id="yes_no", user="Here is an RNA:<SLOT:x>. {question} Answer Yes or No."),
        }

    def capture_recipes(self) -> Mapping[str, CaptureRecipe]:
        """Return a description-primed name recipe."""
        text = "{context} This is the description of the {noun}: {name}"
        recipe = CaptureRecipe(
            name="name_described",
            messages=(RecipeMessage(role="user", text=text),),
            read="name",
            context_field="description",
            context_tokens=64,
        )
        return {recipe.name: recipe}

    def scope_mask(self, task: TaskSpec, object_ids: pd.Index) -> pd.Series:
        """Every object is in scope."""
        del task
        return pd.Series(True, index=object_ids)
