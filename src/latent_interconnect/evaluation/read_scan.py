"""The encoder read-layer scan: which encoder layers carry the domain's properties.

For one encoder, at every scanned layer L (``hidden_states[L]``, pooled), on a study's ``train`` objects:

- **property probes:** a cross-validated ridge probe of each declared Yes/No task's label (±1); the measure is the
  held-out AUROC, on the task's scope (``Domain.scope_mask``, as Yes/No suites score it), unlabelled objects dropped;
- **variant probes** (protein encoders, :class:`VariantProbes`): a cross-validated ridge probe of each declared variant
  label on ``train`` variant objects, each represented by its embedding minus its parent protein's (the change the
  substitution makes); the measure is the held-out AUROC.

The scan needs no receiver states (no R² to the receiver). Each object is predicted by the map fit on the other
cluster folds (:func:`cluster_folds`; a variant carries its parent's cluster), so the scan needs no separate report set.
Per fold the input is standardized and each probe centered on the training folds, and the training design is
factorized once (:func:`_basis`) for every probe that shares the fold's rows and every α of the grid; each probe takes
the α with the least held-out squared error pooled over the folds (as ``build.ridge`` chooses α by CV error; no
nested α). The measure reads the **centered cross-validated discrimination scores**: each row's held-out prediction
minus its fold's training target mean (the intercept the prediction adds back), so folds of uneven prevalence do not
rank by intercept when pooled (pure noise read 0.45 at a large α); α is chosen on the
uncentred predictions, and no held-out label enters a score. Each probe also records its mean within-fold AUROC, a
diagnostic free of cross-fold comparison. Probes sharing a row set share the factorization; a task's scope or unknown
labels make its own row set.
Intervals come from the one bootstrap over the item set's units
(:func:`~latent_interconnect.evaluation.contrasts.task_estimate`), which must be the folds' clusters. The variant
probes read the domain's variant labels and add no dataset. The scan outputs the full diagnostic and chooses
nothing (``examples/guides/choosing_layers.md``).

**Joint reads.** A scan may also declare hand-chosen ``combinations`` of layers: each is probed exactly as a
layer is, on the per-object concatenation of its layers' pooled vectors (``InputSpec`` ``combine: concat``, as a
connector reads several layers). Limits on combinations are guidance for whoever writes the config, not checked here.
:func:`compare_reads` sets the combinations next to the single layers.

**A second extraction.** ``append`` reads another extraction jointly: its pooled vector at one layer is
concatenated after every read's (each object's two rows aligned by object id). It covers every object the scan reads
(proteins, variants and parents), and its recipe and artifacts are in the spec's identity and the record's inputs.

**Per-read parts.** Each read's (a layer's or a combination's) rows are committed as their own ``read_scan`` artifact,
with its numbers (each probe's AUROC and interval, the costly bootstrap) in its metadata, and remembered under a key of
(scan digest, read, code), so a killed scan resumes from its committed reads and a scan can be split statically over
workers (:func:`scan_layers` on a slice of the reads, then :func:`read_layer_scan` merges every read). A dirty tree
never reuses a part. BLAS uses the CPUs of the process's affinity mask (OpenBLAS), so each worker stays within its
allocation.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import NamedTuple, Self

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.build.ridge import SD_EPSILON, AlphaGrid
from latent_interconnect.canonical import canonical_sha256
from latent_interconnect.config import ArtifactDigest, Config, StageConfig, added_field
from latent_interconnect.data.items import ITEM_SET_KIND, OBJECT_ROLE, ItemSet, cluster_field, load_item_set
from latent_interconnect.domains.base import Domain, DomainSpec, TaskKind, open_domain, require_splits
from latent_interconnect.domains.protein import ProteinDomain
from latent_interconnect.encoders.loader import InputLoader
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import EMBEDDINGS_KIND, EmbeddingReader
from latent_interconnect.evaluation.contrasts import task_estimate, unit_columns
from latent_interconnect.evaluation.scores import commit_table, read_table, write_numbers
from latent_interconnect.stats.auroc import auroc
from latent_interconnect.stats.bootstrap import ResamplingUnits
from latent_interconnect.writes.inputs import InputKind, InputSpec

__all__ = [
    "MIN_FOLDS",
    "PROBE_TARGETS",
    "READ_SCAN_KIND",
    "AppendedInput",
    "Read",
    "ReadLayerScanSpec",
    "ReadScanOutputs",
    "VariantProbes",
    "cluster_folds",
    "compare_reads",
    "read_label",
    "read_layer_scan",
    "scan_layers",
]

READ_SCAN_KIND = "read_scan"
"""Artifact kind of the scan's per-object table (and of each layer's part)."""
_TABLE_FILE = "read_scan.parquet"
PROBE_TARGETS = (-1.0, 1.0)
"""The ridge probe's targets for a No and a Yes label."""
MIN_FOLDS = 2
"""The fewest folds a cross-validated probe needs: each fold is predicted by the map of the others."""
_EIGEN_TOLERANCE = 1e-8
"""The most negative eigenvalue of ``ZᵀZ`` accepted as rounding, relative to the largest (else a loud failure)."""
_USE = "read-layer scan"


Read = int | tuple[int, ...]
"""One read of the scan: a layer, or a combination of layers."""


def read_label(read: Read) -> int | str:
    """Return a read's ``read_layer`` label: the layer, or a combination's layers joined by ``+`` (``"34+36"``)."""
    return read if isinstance(read, int) else "+".join(map(str, read))


def cluster_folds(clusters: Sequence[str], folds: int, seed: int) -> np.ndarray:
    """Return each object's fold: a seeded hash of its cluster id, modulo ``folds``.

    The assignment depends on the cluster id and the seed only, so it is the same in every object list.

    Raises:
        ValueError: for fewer than :data:`MIN_FOLDS` folds, or if a fold receives no object.
    """
    if folds < MIN_FOLDS:
        raise ValueError(f"a cross-validated probe needs at least {MIN_FOLDS} folds, not {folds}")
    of = {c: int(canonical_sha256([seed, c]), 16) % folds for c in set(clusters)}
    assigned = np.array([of[c] for c in clusters], dtype=np.int64)
    empty = sorted(set(range(folds)) - set(assigned.tolist()))
    if empty:
        raise ValueError(f"folds {empty} of {folds} received no object ({len(of)} clusters, seed {seed})")
    return assigned


class VariantProbes(Config):
    """The read scan's variant probes (protein encoders).

    Attributes:
        objects: the ``train`` variant objects (an object list of the domain's variant objects).
        embeddings: the encoder's artifacts covering the variants and any parent the scan's ``embeddings`` do not
            cover (each object at each layer in exactly one artifact, as every load requires); an artifact of the
            scan's ``embeddings`` may be named again (one extraction of proteins and variants), and is read once.
        labels: the variant label columns probed (e.g. ``variant:clinical``, ``variant:functional``).
    """

    objects: ArtifactDigest
    embeddings: tuple[ArtifactDigest, ...] = Field(min_length=1)
    labels: tuple[str, ...] = Field(min_length=1)


class AppendedInput(Config):
    """A second extraction appended to every read of a scan.

    Attributes:
        encoder: its recipe (its artifacts' recorded recipe, as every load checks).
        embeddings: its artifacts covering every object the scan reads (proteins, variants, parents).
        layer: the one layer read (its pooled vector, whole).
    """

    encoder: EncoderRecipe
    embeddings: tuple[ArtifactDigest, ...] = Field(min_length=1)
    layer: int


class ReadLayerScanSpec(StageConfig):
    """Encoder read-layer scan of one (receiver, encoder) setup.

    Attributes:
        domain: the objects' domain at a fixed data version (clusters, splits, labels, tasks).
        encoder: the scanned encoder recipe (embedded; ``read: all`` or a list covering ``layers``).
        embeddings: the recipe's embedding artifacts covering the objects (pooled at every scanned layer).
        layers: the encoder layers scanned one by one (``hidden_states`` indices); may be empty when
            ``combinations`` are given.
        labels: the domain's Yes/No task ids probed.
        objects: the study's object list (``train`` objects only).
        folds: the number of cluster folds.
        seed: the fold-hash seed.
        alphas: the ridge penalty grid.
        units: the resampling units (default: the objects' clusters, which the folds use too).
        variants: the variant probes, or None.
        combinations: the layer combinations scanned: each read as the concatenation of its layers' pooled
            vectors, in the order given.
        part: the one part of a composite pooled vector probed, as the recipe's ``pooling`` declares its
            layout: ``cls`` or ``mean`` of ``cls+mean``, ``last`` or ``mean`` of ``last+mean`` (equal widths, in
            that order); None probes the whole vector.
        append: a second extraction's pooled vector appended after every read's (``part`` does not apply to
            it), or None.
    """

    domain: DomainSpec
    encoder: EncoderRecipe
    embeddings: tuple[ArtifactDigest, ...] = Field(min_length=1)
    layers: tuple[int, ...]
    labels: tuple[str, ...] = ()
    objects: ArtifactDigest
    folds: int = Field(ge=MIN_FOLDS)
    seed: int = 0
    alphas: AlphaGrid
    units: ResamplingUnits = ResamplingUnits(factors=(cluster_field(OBJECT_ROLE),))
    variants: VariantProbes | None = None
    combinations: tuple[tuple[int, ...], ...] = added_field(default=())
    part: str | None = added_field(default=None)
    append: AppendedInput | None = added_field(default=None)

    @model_validator(mode="after")
    def _check_probes(self) -> Self:
        if self.units.interval:
            raise ValueError(f"a read scan's intervals are the bootstrap's, not units.interval {self.units.interval!r}")
        if not self.labels and self.variants is None:
            raise ValueError("a read scan probes something: give labels, variants or both")
        if not self.layers and not self.combinations:
            raise ValueError("a read scan reads something: give layers, combinations or both")
        if short := [c for c in self.combinations if len(set(c)) <= 1]:
            raise ValueError(f"combinations {short} read fewer than 2 distinct layers")
        if repeated := sorted({c for c in self.combinations if self.combinations.count(c) > 1}):
            raise ValueError(f"combinations {repeated} repeat")
        if self.part is not None and self.part not in self.parts:
            raise ValueError(f"part {self.part!r} is not a part of pooling {self.encoder.pooling!r}: {self.parts}")
        return self

    @property
    def parts(self) -> tuple[str, ...]:
        """The parts of the recipe's pooled vector, in its stored order (one part unless the pooling is composite)."""
        return tuple(self.encoder.pooling.split("+"))

    @property
    def reads(self) -> tuple[Read, ...]:
        """The scan's reads, in order: every layer, then every combination."""
        return (*self.layers, *self.combinations)


class ReadScanOutputs(NamedTuple):
    """What the read-layer scan commits.

    Attributes:
        numbers: per scanned read (a layer or a combination) and probe, the AUROC with its interval (``numbers``:
            ``read_layer`` (the layer, or a combination's label: :func:`read_label`), ``population`` (``objects`` or
            ``variants``), ``name`` (a task id or a variant label), ``value``, ``lo``, ``hi``, ``dropped``, and the
            diagnostic ``within_fold_auroc``: the mean over the folds holding both classes of the AUROC within the
            fold, NaN when none does, with ``within_fold_folds`` the folds it averages; key
            ``<read_layer>/<population>/<name>``).
        table: the per-object table behind them (``read_scan``: ``read_layer``, ``population``, ``name``, ``item``
            (the index in that population's object list), ``score`` (the centered cross-validated discrimination
            score: the held-out probe output minus its fold's training target mean) and ``label``).
    """

    numbers: ArtifactRef
    table: ArtifactRef


@dataclass(frozen=True)
class _Population:
    """One object list of the scan: its objects, folds, probe row sets and how a layer's features are formed."""

    name: str
    items: ItemSet
    ids: tuple[str, ...]
    folds: np.ndarray
    probes: dict[str, tuple[np.ndarray, np.ndarray]]
    parents: tuple[str, ...] | None = None

    def features(self, pooled: Callable[[Sequence[str]], np.ndarray]) -> np.ndarray:
        """Return the rows' features: the objects' pooled states, or a variant's minus its parent's."""
        X = pooled(self.ids)
        return X if self.parents is None else X - pooled(self.parents)


def read_layer_scan(spec: ReadLayerScanSpec, store: ArtifactStore, code: tuple[str, bool]) -> ReadScanOutputs:
    """Run the scan (CPU; float64 solves) and commit its per-object table and its numbers.

    Every read's part is reused when committed under the same code (``code``: the tree's commit and dirty flag),
    else computed (:func:`scan_layers`); the table and the numbers are the parts' in the spec's read order
    (:attr:`ReadLayerScanSpec.reads`), so they are the same however the reads were split over runs. With combinations,
    every ``read_layer`` is written as its label string (:func:`read_label`), so the column has one type.

    Raises:
        ValueError: as :func:`scan_layers`.
    """
    reads = spec.reads
    parts = scan_layers(spec, store, code, reads)
    table = pd.concat([read_table(store, parts[r], READ_SCAN_KIND, _TABLE_FILE) for r in reads]).reset_index(drop=True)
    numbers = pd.DataFrame([row for r in reads for row in store.manifest(parts[r]).meta["numbers"]])
    if spec.combinations:
        table["read_layer"] = table["read_layer"].astype(str)
        numbers["read_layer"] = numbers["read_layer"].astype(str)
    meta = {"scan": spec.digest(), "parts": {str(read_label(r)): parts[r].sha256 for r in reads}}
    ref = commit_table(store, table, READ_SCAN_KIND, _TABLE_FILE, meta)
    meta = {"scan": spec.digest(), "table": ref.sha256}
    return ReadScanOutputs(numbers=write_numbers(store, numbers, meta), table=ref)


def scan_layers(
    spec: ReadLayerScanSpec, store: ArtifactStore, code: tuple[str, bool], reads: Sequence[Read]
) -> dict[Read, ArtifactRef]:
    """Commit (or reuse) the per-object part of each of ``reads`` (layers or combinations); return read → its part.

    A probe is scored on its task's scope with known labels, a variant probe on the variants with a known label. Each
    object's fold comes from its cluster in the spec's domain, so every row set keeps each cluster's fold. A part is
    remembered under (scan digest, read, ``code``) and reused by a later run of the same clean code (its manifest,
    scan and read checked); a dirty tree recomputes every read.

    Raises:
        ValueError: if an object is outside ``train`` or has no cluster, an item set has more than one role or clusters
            other than the domain's, a task is not a Yes/No task of the domain, variant probes are declared for a
            domain without variants or name an object that is not a variant, a probe's rows fill fewer than
            :data:`MIN_FOLDS` folds or hold one label class, a read is not one of the spec's (or a combination
            repeats a layer: ``InputSpec``), a remembered part is not this scan's, or a factorization is not
            numerically sound (:func:`_basis`).
    """
    unknown = [read_label(r) for r in reads if r not in spec.reads]
    if unknown:
        raise ValueError(f"{_USE}: reads {unknown} are not scanned by the spec")
    commit, dirty = code
    keys = {r: "read-scan-layer-" + canonical_sha256([spec.digest(), r, commit, dirty]) for r in reads}
    parts = {r: ref for r in reads if not dirty and (ref := store.committed(keys[r])) is not None}
    for r, ref in parts.items():
        _check_part(store, ref, spec, r)
    todo = [r for r in reads if r not in parts]
    if not todo:
        return parts
    populations = _populations(spec, open_domain(spec.domain, store), store)
    digests = spec.embeddings
    if spec.variants is not None:
        digests = tuple(dict.fromkeys(digests + spec.variants.embeddings))  # one artifact may hold both populations
    readers = [EmbeddingReader(store, ArtifactRef(d, EMBEDDINGS_KIND)) for d in digests]
    extra = None
    if (a := spec.append) is not None:
        extra = InputLoader(
            InputSpec(recipe=a.encoder.name, layers=(a.layer,), combine="single"),
            [EmbeddingReader(store, ArtifactRef(d, EMBEDDINGS_KIND)) for d in a.embeddings],
            {InputKind.POOLED},
            encoder=a.encoder,
        )
    for read in todo:
        layers = (read,) if isinstance(read, int) else read
        combine = "single" if len(layers) == 1 else "concat"
        loader = InputLoader(
            InputSpec(recipe=spec.encoder.name, layers=layers, combine=combine),
            readers,
            {InputKind.POOLED},
            encoder=spec.encoder,
        )

        def pooled(ids: Sequence[str], loader: InputLoader = loader, n: int = len(layers)) -> np.ndarray:
            X = loader.batch(tuple(ids)).require(InputKind.POOLED).numpy().astype(np.float64)
            if spec.part is not None:
                k, parts = spec.parts.index(spec.part), len(spec.parts)  # each layer's block: its parts, side by side
                X = X.reshape(len(X), n, parts, -1)[:, :, k].reshape(len(X), -1)
            if extra is None:
                return X
            return np.hstack([X, extra.batch(tuple(ids)).require(InputKind.POOLED).numpy().astype(np.float64)])

        label = read_label(read)
        rows = [_probe_rows(pop, pop.features(pooled), label, spec.alphas) for pop in populations]
        numbers = [_estimates(pop, rows_, spec.units, label) for pop, rows_ in zip(populations, rows, strict=True)]
        meta = {"scan": spec.digest(), "read_layer": label, "numbers": [n for ns in numbers for n in ns]}
        parts[read] = commit_table(store, pd.concat(rows, ignore_index=True), READ_SCAN_KIND, _TABLE_FILE, meta)
        if not dirty:  # a duplicate writer's part, if one was remembered first, is the key's
            parts[read] = store.remember(keys[read], parts[read])
    return parts


def compare_reads(numbers: pd.DataFrame, drop: Collection[str] = ()) -> pd.DataFrame:
    """Return a scan's combinations next to its single layers, one row per read.

    Rows: the best single layer (the most ``probe_mean``), the other single layers, then each combination. Columns:
    ``kind`` (``best layer``, ``layer``, ``combination``), ``probe_mean`` and ``probe_within_fold`` (the mean AUROC and
    mean within-fold AUROC over the object probes not in ``drop``), per variant label its AUROC with ``lo`` / ``hi``
    and its within-fold AUROC, and ``probes`` (how many the means average). It computes no new statistic.

    Args:
        numbers: the numbers of a scan with layers and combinations (:class:`ReadScanOutputs`).
        drop: the object probes declared unstable for this encoder, left out of the means.

    Raises:
        ValueError: if the scan has no single layer or no combination, or ``drop`` names a probe it does not have.
    """
    numbers = numbers.assign(read_layer=numbers["read_layer"].astype(str))
    reads = list(dict.fromkeys(numbers["read_layer"]))
    layers = [r for r in reads if "+" not in r]
    combos = [r for r in reads if "+" in r]
    if not combos or not layers:
        raise ValueError(f"compare_reads: layers {layers} and combinations {combos}; each needs one")
    names = set(numbers.loc[numbers["population"] == "objects", "name"])
    if set(drop) - names:
        raise ValueError(f"compare_reads: dropped probes {sorted(set(drop) - names)} are not the scan's")
    out: dict[str, dict[str, object]] = {}
    for read, part in numbers.groupby("read_layer", sort=False):
        probes = part[(part["population"] == "objects") & ~part["name"].isin(list(drop))]
        row: dict[str, object] = {
            "kind": "combination" if read in combos else "layer",
            "probe_mean": probes["value"].mean(),
            "probe_within_fold": probes["within_fold_auroc"].mean(),
        }
        for v in part[part["population"] == "variants"].itertuples():
            row |= {
                v.name: v.value,
                f"{v.name} lo": v.lo,
                f"{v.name} hi": v.hi,
                f"{v.name} within": v.within_fold_auroc,
            }
        out[str(read)] = row
    table = pd.DataFrame.from_dict(out, orient="index").rename_axis("read")
    best = str(table.loc[layers, "probe_mean"].astype(float).idxmax())
    table.loc[best, "kind"] = "best layer"
    return table.loc[[best, *[r for r in layers if r != best], *combos]].assign(probes=len(names - set(drop)))


def _check_part(store: ArtifactStore, ref: ArtifactRef, spec: ReadLayerScanSpec, read: Read) -> None:
    """Refuse a remembered part that is not this scan's intact part of ``read``."""
    store.verify(ref)
    meta = store.manifest(ref).meta
    if (meta.get("scan"), meta.get("read_layer")) != (spec.digest(), read_label(read)) or "numbers" not in meta:
        raise ValueError(
            f"{_USE}: remembered part {ref.short} is not read {read_label(read)} of this scan; delete its pointer"
        )


def _estimates(
    pop: _Population, rows: pd.DataFrame, units: ResamplingUnits, label: int | str
) -> list[dict[str, object]]:
    """Return one read's number rows of a population: per probe, the AUROC, its interval and diagnostics.

    ``label`` is the read's :func:`read_label`. The interval is over ``units``; the diagnostic is the mean within-fold
    AUROC (:class:`ReadScanOutputs`).
    """
    columns = unit_columns(pop.items, units.factors)
    out: list[dict[str, object]] = []
    for name, part in rows.groupby("name", sort=True):
        est = task_estimate(part, "auroc", columns)
        folds, score, y = pop.folds[part["item"].to_numpy()], part["score"].to_numpy(), part["label"].to_numpy()
        within = [
            auroc(score[folds == k], y[folds == k])
            for k in np.unique(folds)
            if 0 < y[folds == k].sum() < (folds == k).sum()
        ]
        out.append(
            {
                "key": f"{label}/{pop.name}/{name}",
                "read_layer": label,
                "population": pop.name,
                "name": str(name),
                "value": est.estimate,
                "lo": est.lo,
                "hi": est.hi,
                "dropped": est.dropped,
                "within_fold_auroc": float(np.mean(within)) if within else float("nan"),
                "within_fold_folds": len(within),
            }
        )
    return out


def _populations(spec: ReadLayerScanSpec, domain: Domain, store: ArtifactStore) -> list[_Population]:
    """Return the scan's populations: the objects with their task probes, then the variants with theirs."""
    proteins = _population("objects", spec.objects, domain, store, spec)
    out = [
        _Population(
            proteins.name, proteins.items, proteins.ids, proteins.folds, _label_rows(spec, domain, proteins.ids)
        )
    ]
    if spec.variants is not None:
        out.append(_variant_population(spec, spec.variants, domain, store))
    return [pop for pop in out if pop.probes]


def _population(name: str, digest: str, domain: Domain, store: ArtifactStore, spec: ReadLayerScanSpec) -> _Population:
    """Return an object list's population without probes: its items, train-checked, and folds from its clusters.

    Raises:
        ValueError: if the item set has more than one role, an object is outside ``train`` or has no cluster, or the
            item set's clusters (the bootstrap units) are not the domain's (the folds').
    """
    items = load_item_set(store, ArtifactRef(digest, ITEM_SET_KIND))
    if set(items.objects) != {OBJECT_ROLE}:
        raise ValueError(f"{_USE}: an object list has one role, {name!r} has {sorted(items.objects)}")
    ids = tuple(items.objects[OBJECT_ROLE])
    require_splits(domain, ids, {"train"}, f"{_USE} ({name})")
    clusters = domain.clusters()
    unclustered = [o for o in ids if pd.isna(clusters.get(o))]
    if unclustered:
        raise ValueError(f"{_USE}: objects without a cluster {unclustered[:5]} (they would share one fold)")
    listed = [str(c) for c in items.fields[cluster_field(OBJECT_ROLE)]]
    if any(str(clusters.get(o)) != c for o, c in zip(ids, listed, strict=True)):
        raise ValueError(f"{_USE}: {name!r}'s item-set clusters are not the domain's (another domain data version?)")
    folds = cluster_folds([str(clusters[o]) for o in ids], spec.folds, spec.seed)
    return _Population(name, items, ids, folds, {})


def _variant_population(
    spec: ReadLayerScanSpec, variants: VariantProbes, domain: Domain, store: ArtifactStore
) -> _Population:
    """Return the variants' population: each row's parent and, per label, the variants whose label is known.

    Raises:
        ValueError: if the domain has no variant objects or an object of the list is not one.
    """
    base = _population("variants", variants.objects, domain, store, spec)
    if not isinstance(domain, ProteinDomain):
        raise ValueError(f"{_USE}: variant probes need a domain with variant objects, not {spec.domain.name!r}")
    table = domain.variants()
    missing = [v for v in base.ids if v not in table.index]
    if missing:
        raise ValueError(f"{_USE}: {missing[:5]} are not variant objects of the domain")
    labels = domain.labels()
    probes = {}
    for column in variants.labels:
        if column not in labels.columns:
            raise ValueError(f"{_USE}: {column!r} is not a label column of the domain")
        rows, targets = _known_targets(labels[column].reindex(list(base.ids)), None)
        if not rows.size:
            raise ValueError(f"{_USE}: label {column!r} is known on none of the variants")
        probes[column] = (rows, targets)
    parents = tuple(str(table.loc[v, "parent"]) for v in base.ids)
    return _Population(base.name, base.items, base.ids, base.folds, probes, parents)


def _probe_rows(pop: _Population, X: np.ndarray, label: int | str, alphas: Sequence[float]) -> pd.DataFrame:
    """Return one read's per-object rows of every probe of a population (one fit per row set, all its probes)."""
    groups: dict[tuple[int, ...], list[str]] = {}
    for name, (rows, y) in pop.probes.items():
        folds = pop.folds[rows]
        if len(np.unique(folds)) < MIN_FOLDS:
            raise ValueError(f"{_USE}: {name!r} fills {len(np.unique(folds))} fold, a probe needs {MIN_FOLDS}")
        if len(np.unique(y)) < len(PROBE_TARGETS):
            raise ValueError(f"{_USE}: {name!r} has one label class over its objects (no AUROC)")
        groups.setdefault(tuple(rows.tolist()), []).append(name)
    out = []
    for key, names in groups.items():
        rows = np.asarray(key, dtype=np.int64)
        Y = np.stack([pop.probes[n][1] for n in names], 1)
        held = _held_out(X[rows], pop.folds[rows], Y, alphas)
        if not np.isfinite(held).all():  # a NaN is a bug
            raise ValueError(f"read {label}, {pop.name}/{names}: held-out predictions are not finite")
        for j, name in enumerate(names):
            base = {"read_layer": label, "population": pop.name, "name": name, "item": rows}
            out.append(pd.DataFrame(base | {"score": held[:, j], "label": (Y[:, j] == PROBE_TARGETS[1]) * 1.0}))
    return pd.concat(out, ignore_index=True)


def _held_out(X: np.ndarray, folds: np.ndarray, Y: np.ndarray, alphas: Sequence[float]) -> np.ndarray:
    """Return ``(n, p)``: the centered cross-validated discrimination scores, each column at its own α.

    Each row is predicted by the map fit on the other folds; per fold: the training rows' standardization and target
    means (the offset), one factorization (:func:`_basis`) serving every column and α. A column's α is the grid value
    with the least held-out squared error of the predictions (offset included) summed over the folds (the first on a
    tie). The score returned is the chosen prediction minus its fold's offset (module docstring).
    """
    grid = np.asarray(alphas, dtype=np.float64)
    held = np.empty((len(grid), *Y.shape))
    offsets = np.empty(Y.shape)
    for k in np.unique(folds):
        train = folds != k
        mu, sd = X[train].mean(0), X[train].std(0) + SD_EPSILON
        Z = (X[train] - mu) / sd
        offsets[~train] = Y[train].mean(0)
        lam, V = _basis(Z)
        B = V.T @ (Z.T @ (Y[train] - Y[train].mean(0)))
        C = ((X[~train] - mu) / sd) @ V
        for i, a in enumerate(grid):
            held[i, ~train] = C @ (B / (lam + a)[:, None])
    best = ((held + offsets - Y) ** 2).sum(1).argmin(0)
    return held[best, :, np.arange(Y.shape[1])].T


def _basis(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (λ, V) of a standardized design: ``W(α) = V diag(1/(λ + α)) Vᵀ ZᵀY`` for every α, with no division.

    With more rows than columns, the eigendecomposition of ``ZᵀZ`` (3-4× faster than the SVD; equal to it to ~1e-13 on
    real layers); otherwise the thin SVD (``λ = s²``), never the Gram of a wide design.

    Raises:
        ValueError: if an eigenvalue is not finite or more negative than rounding (:data:`_EIGEN_TOLERANCE`).
    """
    n, d = Z.shape
    if n < d:
        _, s, Vt = np.linalg.svd(Z, full_matrices=False)
        return s**2, Vt.T
    lam, V = np.linalg.eigh(Z.T @ Z)
    if not np.isfinite(lam).all() or lam[0] < -_EIGEN_TOLERANCE * max(float(lam[-1]), 1.0):
        raise ValueError(f"{_USE}: the design's Gram has eigenvalues {lam[0]:.3g}..{lam[-1]:.3g} (not a Gram)")
    return np.clip(lam, 0.0, None), V


def _label_rows(
    spec: ReadLayerScanSpec, domain: Domain, objects: Sequence[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return task id → (the object rows whose label is known and in scope, their ±1 labels)."""
    tasks, labels = domain.tasks(), domain.labels()
    out = {}
    for task_id in spec.labels:
        task = tasks.get(task_id)
        if task is None or task.kind is not TaskKind.YES_NO:
            raise ValueError(f"{_USE}: {task_id!r} is not a Yes/No task of the domain")
        in_scope = domain.scope_mask(task, pd.Index(objects)).reindex(list(objects)).fillna(False).to_numpy(bool)
        out[task_id] = _known_targets(labels[task.label].reindex(list(objects)), in_scope)
    return out


def _known_targets(values: pd.Series, in_scope: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (the rows whose label is known and in scope, their probe targets, :data:`PROBE_TARGETS`)."""
    known = values.notna().to_numpy() if in_scope is None else values.notna().to_numpy() & in_scope
    rows = np.flatnonzero(known).astype(np.int64)
    targets = np.where(values.to_numpy()[rows].astype(bool), PROBE_TARGETS[1], PROBE_TARGETS[0])
    return rows, targets
