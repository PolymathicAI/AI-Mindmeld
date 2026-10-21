"""Variant item sets: the protein-mutation evaluation cohorts and their fixed partners (E10/E11).

One :class:`VariantItemSpec` builds the cohorts of one training contract from the domain's variant objects (single
substitutions, ``domains.protein``), each on one generalization axis, and commits per cohort and once:

- ``items-<cohort>``: the scored variants (role :data:`~latent_interconnect.data.items.OBJECT_ROLE`), with the fields
  every variant condition reads (``data.suites.FIELD_CONDITIONS``): ``wt.object`` (the parent), ``sibling.object``,
  ``derange.object``, ``name.object`` (the parent's name), ``name_substitution.object`` (the parent's name with the
  substitution, worded by ``substitution``), ``pair`` (the fixed pair's id, or "" for a sampled cohort),
  ``source.object`` (the annotation source, below), ``parent_represented`` (bool) and ``cluster.object`` (the
  parent's cluster, so analyses resample parent clusters). Its metadata lists what was ``dropped`` (never refilled),
  one record per fixed pair (``pair``: its id as the ``pair`` field writes it, ``parent``, ``reason``); for a sampled
  cohort its ``eligible_parents``; and for every cohort its ``overlap`` with each other cohort of its axis (shared
  ``variants`` and ``parents``: reported, not excluded);
- ``objects``: every object a condition of any cohort writes (the items, their parents, siblings and partners),
  sorted: the list an extraction reads, so each recipe extracts all the cohorts once.

**Axes.** ``protein``: variants of held-out parents (variant split ``val``/``test``), none of whose parents is
trained in any form (a training-text object, a parent with a trained variant, or a relational wild type,
``relational_parents``). ``position``: held-out positions (``heldpos``) of a parent that training represents in one of
those three forms, with no trained variant at the tested parent and position. A violation is refused, except one: a
position item whose parent training does not represent, which ``unrepresented_parents`` drops (the pair whole,
recorded in the metadata, no refill) or keeps (``parent_represented`` false). It is the cohort definition's one switch:
such a pair is never relabelled as an unseen-protein case.

**Cohorts.** Fixed pairs (``pairs``): listed discordant pairs ``(yes, no)`` of one parent under the label
column ``label``, each member the other's sibling, in the listed order. Or a sampled effect cohort (``effect``): ``n``
variants with a non-empty effect text, one per parent: ``n`` parents drawn with ``seed`` among those with such a
variant and another eligible variant at another position on the axis, then one of the parent's variants and its
sibling (another position) with the same generator. One per parent keeps a parent's identity from dominating the
cohort, and lets the derangement pair every item with another parent. Nothing excludes the fixed pairs' variants or
parents (no exclusion; the overlap is reported). Whether the effect and the WT-relational readouts share a
draw is the config's: one cohort read by both suites, or two cohorts with their own seeds. Too few eligible parents is
refused, never filled. Partners are drawn once, on the whole cohort, for every encoder: an object an encoder cannot
embed makes the bindings that write it unavailable at scoring (``evaluation.suites.SuiteRunConfig.unavailable``),
never a redrawn map.

**Annotation source** (``provenance``: the domain's own variant table, checked against the domain's recorded source
hash): ``effect_source`` (``mutagen`` or ``variant``, the feature the effect text came from) or, where the row has
no effect text, ``variant`` when its ``sources`` include a VARIANT feature (the humsavar-labeled substitutions; the
rule of the original variant text-baseline results, ``variant_text_baselines.md``); a row with neither is refused.

**Derangement** (``derange_seed``): each item's partner is another item's variant of a different parent, by the first
seeded derangement of the cohort with no same-parent match; label-independent, so its binary control is not a
same-label shuffle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import ArtifactDigest, Config, StageConfig
from latent_interconnect.data.items import (
    ITEM_SET_KIND,
    OBJECT_ROLE,
    ItemSet,
    commit_item_set,
    derangement,
    load_item_set,
    object_list,
    with_clusters,
)
from latent_interconnect.domains.base import DomainSpec, open_domain
from latent_interconnect.domains.protein import ProteinDomain
from latent_interconnect.domains.tables import SourceFile, read_source

__all__ = ["EffectDraw", "VariantCohort", "VariantItemSpec", "build_variant_items"]

_AXIS_SPLITS = {"protein": frozenset({"val", "test"}), "position": frozenset({"heldpos"})}
"""Axis → the variant splits (``ProteinDomain.variants`` ``vsplit``) its items come from."""
_DERANGE_TRIES = 1000
_COLUMNS = ["object", "wt", "sibling", "pair"]
_SOURCES = frozenset({"mutagen", "variant"})
"""The annotation sources of an effect text (the provenance table's ``effect_source``)."""


class EffectDraw(Config):
    """A sampled effect/relational cohort.

    Attributes:
        n: the number of variants (one per parent).
        seed: the draw's seed (parents, then each one's variant and sibling).
        field: the effect-text field the items must carry (the stream's ``effect_field``).
    """

    n: int = Field(ge=2)
    seed: int
    field: str


class VariantCohort(Config):
    """One cohort of a :class:`VariantItemSpec`.

    Attributes:
        axis: ``protein`` (held-out parents) or ``position`` (held-out positions of represented parents).
        label: the variant label column the fixed pairs are discordant on (pairs only).
        pairs: fixed discordant pairs ``(yes, no)``, in order; empty for a sampled cohort.
        effect: the sampled effect cohort; None for fixed pairs.
    """

    axis: Literal["protein", "position"]
    label: str | None = None
    pairs: tuple[tuple[str, str], ...] = ()
    effect: EffectDraw | None = None

    @model_validator(mode="after")
    def _one_cohort(self) -> VariantCohort:
        if bool(self.pairs) == (self.effect is not None) or bool(self.pairs) != (self.label is not None):
            raise ValueError("a variant cohort takes fixed pairs with a label, or an effect draw, not both")
        return self


class VariantItemSpec(StageConfig):
    """The protein-mutation cohorts of one training contract (the module docstring gives the rules).

    Attributes:
        name: the item sets' name prefix (a cohort's set is ``<name>/<cohort>``, the object list
            ``<name>/objects``).
        domain: the protein domain, at a version with variants.
        training_set: the trained objects (a training selection's ``train_ids``: proteins and trained variants).
        relational_parents: the wild types the variant stream binds outside the training objects (the selection's
            ``parents`` list).
        cohorts: cohort name (letters, digits, dashes) → :class:`VariantCohort`; output ``items-<cohort>``.
        unrepresented_parents: ``drop`` (whole pairs, recorded, no refill) or ``keep`` (flagged) a position item whose
            parent training does not represent: the cohort definition's one switch; such a pair is
            never relabelled as an unseen-protein case.
        derange_seed: seed of each cohort's derangement partners.
        substitution: the ``name_substitution`` wording, with ``{name}``, ``{wt}``, ``{position}`` and ``{mut}``.
        provenance: the domain's variant table (its declared source file), read for the annotation source.
    """

    name: str
    domain: DomainSpec
    training_set: ArtifactDigest
    relational_parents: ArtifactDigest
    cohorts: dict[str, VariantCohort] = Field(min_length=1)
    unrepresented_parents: Literal["drop", "keep"]
    derange_seed: int
    substitution: str
    provenance: SourceFile

    @model_validator(mode="after")
    def _wording(self) -> VariantItemSpec:
        for key in ("{name}", "{wt}", "{position}", "{mut}"):
            if key not in self.substitution:
                raise ValueError(f"variant items {self.name!r}: substitution lacks {key}")
        bad = [c for c in self.cohorts if not c.replace("-", "").isalnum()]
        if bad:
            raise ValueError(f"variant items {self.name!r}: cohort names {bad} are not letters, digits and dashes")
        return self


class _Training:
    """What the training contract trains: the represented parents and the trained (parent, position) sites."""

    def __init__(self, spec: VariantItemSpec, store: ArtifactStore, variants: pd.DataFrame) -> None:
        trained = set(load_item_set(store, ArtifactRef(spec.training_set, ITEM_SET_KIND)).object_ids)
        bound = set(load_item_set(store, ArtifactRef(spec.relational_parents, ITEM_SET_KIND)).object_ids)
        trained_variants = variants[variants.index.isin(trained)]
        self.represented = trained | set(trained_variants["parent"]) | bound
        self.sites = set(zip(trained_variants["parent"], trained_variants["position"], strict=True))

    def eligible(self, axis: str, parent: str, position: int) -> bool:
        """Whether a variant may test its axis.

        A protein-axis parent is trained in no form; a position-axis position is untrained and its parent represented.
        """
        if axis == "protein":
            return parent not in self.represented
        return parent in self.represented and (parent, position) not in self.sites


def build_variant_items(spec: VariantItemSpec, store: ArtifactStore, data_root: Path) -> dict[str, ArtifactRef]:
    """Commit each cohort's ``items-<cohort>`` and the union ``objects`` (module docstring).

    Args:
        spec: the cohorts.
        store: the artifact store (domain, training lists in; item sets out).
        data_root: the data root the provenance table's path is under (``Paths.data``).

    Raises:
        ValueError: if a pair is not two variants of one parent with the labels yes and no, or a variant is in two
            pairs; an item is off its axis, its parent is trained on the protein axis, or its tested position is
            trained on the position axis; a sampled cohort has fewer eligible parents than ``n``; no derangement
            avoids same-parent partners; the provenance table is not the domain's, or an item has no source.
        IntegrityError: if the provenance table's sha256 differs from the spec's.
    """
    domain = open_domain(spec.domain, store)
    if not isinstance(domain, ProteinDomain):
        raise ValueError(f"variant items {spec.name!r}: domain {spec.domain.name!r} holds no variants")
    recorded = store.manifest(store.ref(spec.domain.data)).meta.get("sources", {}).get("variants")
    if recorded != spec.provenance.sha256:
        raise ValueError(f"variant items {spec.name!r}: provenance {spec.provenance.path} is not the domain's table")
    variants = domain.variants()
    training = _Training(spec, store, variants)
    names = domain.objects()["name"]
    labels = domain.labels()
    chosen: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict[str, Any]] = {}
    for key, cohort in spec.cohorts.items():
        where = f"variant items {spec.name}/{key}"
        if cohort.pairs:
            table, dropped = _fixed_pairs(
                where,
                cohort,
                unrepresented=spec.unrepresented_parents,
                labels=labels,
                variants=variants,
                training=training,
            )
            meta[key] = {"axis": cohort.axis, "dropped": dropped}
        else:
            table, eligible = _sampled(where, cohort, fields=domain.fields(), variants=variants, training=training)
            meta[key] = {"axis": cohort.axis, "dropped": [], "eligible_parents": eligible}
        _check_axis(where, cohort.axis, variants, table["object"], training)
        table["derange"] = _derange(where, spec.derange_seed, table)
        chosen[key] = table
    for key, table in chosen.items():
        meta[key]["overlap"] = {
            other: {
                "variants": len(set(table["object"]) & set(t["object"])),
                "parents": len(set(table["wt"]) & set(t["wt"])),
            }
            for other, t in chosen.items()
            if other != key and meta[other]["axis"] == meta[key]["axis"]
        }
    source = _sources(spec, data_root, sorted({o for t in chosen.values() for o in t["object"]}))
    outputs: dict[str, ArtifactRef] = {}
    written: set[str] = set()
    for key, table in chosen.items():
        where = f"variant items {spec.name}/{key}"
        parents = names.reindex(table["wt"])
        if parents.isna().any() or (parents == "").any():
            raise ValueError(f"{where}: parents without a name, e.g. {list(table['wt'][:3])}")
        site = variants.loc[table["object"]]
        wording = [
            spec.substitution.format(name=n, wt=w, position=int(p), mut=m)
            for n, w, p, m in zip(parents, site["wt"], site["position"], site["mut"], strict=True)
        ]
        fields: dict[str, tuple[Any, ...]] = {
            "wt.object": tuple(table["wt"]),
            "sibling.object": tuple(table["sibling"]),
            "derange.object": tuple(table["derange"]),
            "name.object": tuple(str(n) for n in parents),
            "name_substitution.object": tuple(wording),
            "pair": tuple(table["pair"]),
            "source.object": tuple(source[o] for o in table["object"]),
            "parent_represented": tuple(bool(p in training.represented) for p in table["wt"]),
        }
        items = with_clusters(
            ItemSet(f"{spec.name}/{key}", {OBJECT_ROLE: domain.name}, {OBJECT_ROLE: tuple(table["object"])}, fields),
            {domain.name: domain},
        )
        outputs[f"items-{key}"] = commit_item_set(store, items, meta[key])
        written |= set(table["object"]) | set(table["wt"]) | set(table["sibling"])  # derange partners are items
    union = object_list(f"{spec.name}/objects", domain, tuple(sorted(written)))
    outputs["objects"] = commit_item_set(store, union, {"dropped": {k: m["dropped"] for k, m in meta.items()}})
    return outputs


def _sources(spec: VariantItemSpec, data_root: Path, objects: list[str]) -> dict[str, str]:
    """Return each item's annotation source (module docstring) from the provenance table."""
    columns = ["vid", "effect_source", "sources"]
    table = pd.read_parquet(read_source(data_root, spec.provenance), columns=columns).set_index("vid")
    rows = table.reindex(objects)
    out: dict[str, str] = {}
    for vid, effect, sources in zip(rows.index, rows["effect_source"], rows["sources"], strict=True):
        if isinstance(effect, str) and effect in _SOURCES:
            out[str(vid)] = effect
        elif pd.isna(effect) and isinstance(sources, str) and "variant" in sources.split(","):
            out[str(vid)] = "variant"
        else:
            raise ValueError(f"variant items {spec.name!r}: {vid!r} has no annotation source ({effect!r}, {sources!r})")
    return out


def _fixed_pairs(
    where: str,
    cohort: VariantCohort,
    *,
    unrepresented: str,
    labels: pd.DataFrame,
    variants: pd.DataFrame,
    training: _Training,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Return the items of the fixed pairs (each member the other's sibling) and the pairs dropped."""
    rows: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    label = labels[str(cohort.label)]
    listed = [v for pair in cohort.pairs for v in pair]
    if twice := sorted({v for v in listed if listed.count(v) > 1}):
        raise ValueError(f"{where}: variants in more than one pair (or a pair listed twice): {twice[:3]}")
    for yes, no in cohort.pairs:
        for vid in (yes, no):
            if vid not in variants.index:
                raise ValueError(f"{where}: {vid!r} is no variant of the domain")
            if variants.at[vid, "vsplit"] not in _AXIS_SPLITS[cohort.axis]:
                raise ValueError(f"{where}: {vid!r} is off the {cohort.axis} axis")
        parent = str(variants.at[yes, "parent"])
        if variants.at[no, "parent"] != parent or variants.at[yes, "position"] == variants.at[no, "position"]:
            raise ValueError(f"{where}: pair {yes}/{no} is not two positions of one parent")
        if not (_labelled(label.get(yes), want=True) and _labelled(label.get(no), want=False)):
            raise ValueError(f"{where}: pair {yes}/{no} is not ({cohort.label}: yes, no)")
        pair = f"{parent}:{yes}|{no}"
        if cohort.axis == "position" and parent not in training.represented and unrepresented == "drop":
            dropped.append({"pair": pair, "parent": parent, "reason": "parent_unrepresented"})
            continue
        rows += [
            {"object": yes, "wt": parent, "sibling": no, "pair": pair},
            {"object": no, "wt": parent, "sibling": yes, "pair": pair},
        ]
    return pd.DataFrame(rows, columns=_COLUMNS), dropped


def _labelled(value: object, *, want: bool) -> bool:
    """Whether a nullable label value is known and equals ``want``."""
    return not pd.isna(value) and bool(value) is want


def _sampled(
    where: str,
    cohort: VariantCohort,
    *,
    fields: pd.DataFrame,
    variants: pd.DataFrame,
    training: _Training,
) -> tuple[pd.DataFrame, int]:
    """Return a seeded effect cohort and the number of eligible parents it was drawn from.

    One variant per parent, each with a sibling at another eligible position.
    """
    effect = cohort.effect
    if effect is None:
        raise ValueError(f"{where}: neither pairs nor an effect draw")
    on_axis = variants[variants["vsplit"].isin(_AXIS_SPLITS[cohort.axis])]
    ok = [
        training.eligible(cohort.axis, p, int(q)) for p, q in zip(on_axis["parent"], on_axis["position"], strict=True)
    ]
    on_axis = on_axis[ok]
    text = fields[effect.field].reindex(on_axis.index)
    with_text = on_axis[text.notna() & (text.astype(str).str.len() > 0)]
    positions = on_axis.groupby("parent")["position"].nunique()
    pool = with_text[with_text["parent"].map(positions) >= 2]  # noqa: PLR2004 - a sibling needs another position
    parents = sorted(pool["parent"].unique())
    if len(parents) < effect.n:
        raise ValueError(f"{where}: {len(parents)} eligible parents for n {effect.n}")
    rng = np.random.default_rng(effect.seed)
    rows = []
    for parent in sorted(parents[j] for j in rng.choice(len(parents), size=effect.n, replace=False)):
        own = sorted(pool.index[pool["parent"] == parent])
        vid = own[int(rng.integers(len(own)))]
        position = variants.at[vid, "position"]
        others = sorted(on_axis.index[(on_axis["parent"] == parent) & (on_axis["position"] != position)])
        rows.append({"object": vid, "wt": parent, "sibling": others[int(rng.integers(len(others)))], "pair": ""})
    return pd.DataFrame(rows, columns=_COLUMNS), len(parents)


def _check_axis(where: str, axis: str, variants: pd.DataFrame, objects: pd.Series, training: _Training) -> None:
    """Refuse an item off its axis, a trained protein-axis parent or a trained tested position."""
    site = variants.loc[list(objects)]
    off = site.index[~site["vsplit"].isin(_AXIS_SPLITS[axis])]
    if len(off):
        raise ValueError(f"{where}: {len(off)} items off the {axis} axis, e.g. {list(off[:3])}")
    if axis == "protein":
        bad = site.index[site["parent"].isin(training.represented)]
        what = "parents trained in some form"
    else:
        sites = zip(site["parent"], site["position"], strict=True)
        bad = site.index[[(p, int(q)) in training.sites for p, q in sites]]
        what = "tested positions trained"
    if len(bad):
        raise ValueError(f"{where}: {len(bad)} items with {what}, e.g. {list(bad[:3])}")


def _derange(where: str, seed: int, chosen: pd.DataFrame) -> list[str]:
    """Return each item's derangement partner: the first seeded derangement with no same-parent match."""
    parents = chosen["wt"].to_numpy()
    for attempt in range(_DERANGE_TRIES):
        perm = derangement(len(chosen), seed + attempt)
        if not (parents[perm] == parents).any():
            return [str(chosen["object"].iloc[j]) for j in perm]
    raise ValueError(f"{where}: no derangement avoids same-parent partners")
