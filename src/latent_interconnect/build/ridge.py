"""Ridge initialization: encoder input → receiver state, per receiver layer and target (construction stage).

Inputs are embedding artifacts of the encoder (as the connector reads them, through
:class:`~latent_interconnect.encoders.loader.InputLoader`) and capture artifacts of the receiver's states for the same
objects (``build.capture``). The procedure is the paper's:

- the input is standardized over the fit set (mean, population σ + 1e-8) and the target is centered;
- the K CV folds are ``permutation(n) % K`` from ``numpy.random.default_rng(seed)``, over the fit set in its stored
  order; the standardization and the centering use the whole fit set;
- per layer, one α serves every output dimension: the grid value with the least squared CV error summed over folds
  and dimensions (the first on a tie), with the closed-form ridge solved through each fold's SVD;
- ``W`` is refitted on the whole fit set at that α; the CV R² is ``1 − cv_mse / mean(centered target²)``;
- the target is the capture state the spec's ``pool`` selects: by default the capture's stored pooled state (its
  recipe's target: the last read token, or the mean for ``caption``), with ``pool: mean`` the mean over the read
  tokens (``EmbeddingReader.means``; the capture stores both);
- the bias has two conventions, chosen by ``source_floor``. Unset, the paper's split (every fit recorded before the
  source floor): ``W`` maps to the layer's own state and ``b`` is the fit-set mean of the capture's read-token means
  whatever ``pool`` is (for the name recipes by default: ``W`` to the centered last-name-token state, ``b`` the mean
  over all name tokens). Set to p*, the source-floor rule: the target at layer ``L > 0`` is the ``pool`` state of layer
  ``max(L, p*)`` (layer 0 its own; :func:`source_layer`) and ``b`` is the fit-set mean of that same state.

All solves are float64 on CPU; the stored maps are float32.

The same fit serves the layer-selection study's held-out scan ridge: a spec with ``exclude`` (the injection
scan's items) fits on the fit set minus those objects, so no fit-derived quantity (standardization, centering, ``b``,
α) sees them, and the scan writes its predictions for them (``evaluation.scan_states``, source ``ridge``). A
production ridge declares no ``exclude``.

Artifact format (kind :data:`RIDGE_KIND`): ``mu.npy`` and ``sd.npy`` ``(d_in,)``, then per layer ``W.L<L>.npy``
``(d_model, d_in)`` and ``b.L<L>.npy`` ``(d_model,)``, all float32. The metadata holds ``target``, ``layers``, ``d_in``,
``d_model``, ``n_fit`` (and, for a fit, ``fit_left_out``; with ``exclude``, ``excluded``; with ``pool``, ``pool``;
with ``source_floor``, ``source_floor``), the encoder ``input`` (``InputSpec.identity()``: recipe, read layers in
order, combine, appended extraction) and its recipe's digest ``recipe_digest`` (and an appended extraction's recipe
digest, ``append_digest``) and the capture's receiver ``{model_id, revision}`` (:func:`capture_receiver`), which every
consumer checks (``RidgeSolution.require_fit``), ``alpha`` and ``cv_r2`` per layer, and the producing spec's digest.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import Field

from latent_interconnect.artifacts.records import RecordStore
from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import ArtifactDigest, StageConfig, added_field
from latent_interconnect.data.items import ITEM_SET_KIND, load_item_set
from latent_interconnect.domains.base import DomainSpec, open_domain, require_splits
from latent_interconnect.encoders.loader import InputLoader
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import EMBEDDINGS_KIND, EmbeddingReader
from latent_interconnect.writes.init import RidgeLayer, RidgeSolution
from latent_interconnect.writes.inputs import InputKind, InputSpec

__all__ = [
    "RIDGE_KIND",
    "AlphaCV",
    "AlphaGrid",
    "RidgeSpec",
    "RidgeStats",
    "capture_receiver",
    "fit_objects",
    "fit_ridge",
    "load_ridge",
    "load_ridge_stats",
    "source_layer",
]

RIDGE_KIND = "ridge_init"
"""Artifact kind of a ridge fit."""
SD_EPSILON = 1e-8
"""Added to the population σ of every input feature (the paper's standardization; constant features stay finite)."""
AlphaGrid = Annotated[tuple[Annotated[float, Field(gt=0, allow_inf_nan=False)], ...], Field(min_length=1)]
"""A ridge penalty grid: one or more finite α > 0 (the one type of every config's ``alphas``)."""


class RidgeSpec(StageConfig):
    """A ridge fit.

    Attributes:
        domain: the objects' domain at a fixed data version (the fit set must be in its ``train`` split).
        input: the encoder input, as the connector will read it.
        encoder: the encoder recipe ``input`` names (embedded, usually a ``$ref``): the loader checks the input
            against it (``writes.inputs.check_read_layers``) and the embeddings' artifacts against its name.
        embeddings: digests of the encoder's embedding artifacts covering the fit set (and those of the input's
            appended extraction, if any).
        targets: digest of the capture artifact (``build.capture``): ``W`` maps to its centered states (``pool``) and
            ``b`` is the fit-set mean of its read-token means.
        target: the target name connectors cite.
        layers: receiver layers to fit (layer 0 included for a layer-0 map).
        fit_set: digest of the object list the fit uses (``data.items``; ``train`` objects only); its order
            sets the folds.
        alphas: ridge penalty grid (the paper: ``numpy.logspace(0, 9, 28)``).
        folds: CV folds.
        seed: fold assignment seed.
        exclude: digest of an object list left out of the fit (the held-out scan ridge's scan items), or None
            (a production ridge). Every excluded object must be in the fit set.
        pool: the capture state ``W`` maps to: None, the capture's stored pooled state (its recipe's ``target``);
            ``mean``, the mean over the read tokens (a capture without means fails; nothing falls back).
        source_floor: the source-layer floor p*, or None. It also sets the bias convention:
            set (the source-floor rule): ``W`` and ``b`` both come from one state, the ``pool`` state of layer
            :func:`source_layer` ``(L, p*)`` (``b`` = its fit-set mean);
            None (the paper's split; every config recorded before the floor): ``W`` maps to the layer's own ``pool``
            state, but ``b`` is always the fit-set mean of the layer's read-token MEANS, even when ``W``'s target is
            the last token. None keeps that split so stored fits and digests are unchanged; it is not the
            source-floor rule.
    """

    domain: DomainSpec
    input: InputSpec
    encoder: EncoderRecipe
    embeddings: tuple[ArtifactDigest, ...] = Field(min_length=1)
    targets: ArtifactDigest
    target: str
    layers: tuple[int, ...] = Field(min_length=1)
    fit_set: ArtifactDigest
    alphas: AlphaGrid
    folds: int = Field(ge=2)
    seed: int = 0
    exclude: ArtifactDigest | None = added_field(default=None)
    pool: Literal["mean"] | None = added_field(default=None)
    source_floor: int | None = added_field(default=None, ge=1)


def source_layer(layer: int, floor: int | None, fixed: int | None = None) -> int:
    """Return the receiver layer whose state is the target at ``layer``; layer 0 is always its own.

    ``floor`` p*: ``max(layer, p*)`` (the floor rule); ``fixed`` S: S at every layer above 0 (an oracle scan's fixed
    source, ``evaluation.scan_states.WrittenScan.fixed_source``); neither: ``layer``.

    Raises:
        ValueError: if both ``floor`` and ``fixed`` are given.
    """
    if floor is not None and fixed is not None:
        raise ValueError(f"source floor {floor} and fixed source {fixed}: a source layer has one rule")
    if layer == 0:
        return 0
    return fixed if fixed is not None else layer if floor is None else max(layer, floor)


@dataclass(frozen=True)
class RidgeStats:
    """What a ridge fit chose, per layer.

    Attributes:
        alpha: layer → the CV-chosen α.
        cv_r2: layer → the CV R².
        n_fit: number of fit objects.
    """

    alpha: Mapping[int, float]
    cv_r2: Mapping[int, float]
    n_fit: int


class AlphaCV:
    """Ridge solves on one standardized design, α chosen by cross-validation over given fold labels.

    The SVDs of the design, per CV fold and in full, and each fold's held-out Gram ``(Z_h V)ᵀ(Z_h V)``, are computed
    once and shared by every target fit on it (every receiver layer of a ridge fit).

    Args:
        Z: ``(n, d)`` standardized design, float64.
        labels: ``(n,)`` CV fold of each row; each distinct label is held out once, in sorted order.
    """

    def __init__(self, Z: np.ndarray, labels: np.ndarray) -> None:
        self.cv = []
        for f in np.unique(labels):
            train, held = labels != f, labels == f
            U, s, Vt = np.linalg.svd(Z[train], full_matrices=False)
            ZhV = Z[held] @ Vt.T
            self.cv.append((train, held, U, s, ZhV, ZhV.T @ ZhV))
        self.U, self.s, self.Vt = np.linalg.svd(Z, full_matrices=False)

    def fit(self, Y: np.ndarray, alphas: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Return (chosen α, CV mean squared error per entry, ``W`` of shape ``(d_in, d_out)``) for centered ``Y``.

        A fold's error at α, with ``B = UᵀY_train``, ``f = s / (s² + α)`` and the fold's ``G = (Z_h V)ᵀ(Z_h V)``, is
        ``‖Y_h‖² − 2 Σ_i f_i ((Z_h V)ᵀY_h ∘ B)_i· + fᵀ(G ∘ BBᵀ)f``: the per-α cost is ``r²``, not a prediction of
        every held-out entry (equal to the direct sum to rounding).
        """
        err = np.zeros(len(alphas))
        for train, held, U, s, ZhV, G in self.cv:
            B = U.T @ Y[train]
            total = float((Y[held] ** 2).sum())
            cross = ((ZhV.T @ Y[held]) * B).sum(1)
            quad = G * (B @ B.T)
            for i, a in enumerate(alphas):
                f = s / (s**2 + a)
                err[i] += total - 2.0 * f @ cross + f @ quad @ f
        best = int(np.argmin(err))
        alpha = float(alphas[best])
        W = self.Vt.T @ ((self.s / (self.s**2 + alpha))[:, None] * (self.U.T @ Y))
        return alpha, float(err[best] / Y.size), W


def fit_objects(
    store: ArtifactStore, objects: str, domain: DomainSpec, targets: EmbeddingReader, use: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (the listed objects, those the capture holds, in list order): the objects a fit on a capture uses.

    A capture leaves out objects without a screened description; a capture that left nothing out yet lacks
    listed objects was made for another object list.

    Args:
        store: the artifact store.
        objects: the object list's digest (``train`` objects only).
        domain: the objects' domain.
        targets: the capture.
        use: what the objects are for, in error messages.

    Raises:
        ValueError: if ``objects`` is not an object list, an object is outside ``train``, the capture holds none of
            them, or it is another list's capture.
    """
    listed = load_item_set(store, ArtifactRef(objects, ITEM_SET_KIND)).object_ids
    require_splits(open_domain(domain, store), listed, {"train"}, use)
    held = tuple(o for o in listed if o in targets)
    if not held:
        raise ValueError(f"{use}: the capture holds none of the {len(listed)} objects")
    if len(held) < len(listed) and not targets.meta.get("left_out"):
        raise ValueError(f"{use}: {len(listed) - len(held)} objects are not in the capture (another list's capture)")
    return listed, held


def capture_receiver(records: RecordStore, capture: str) -> dict[str, str]:
    """Return the receiver whose states a capture holds, ``{model_id, revision}``, from the capture's own records.

    ``model_id`` is the capture config's receiver; ``revision`` is what its execution loaded (``provenance.execution``,
    which every capture backend checks against the pinned snapshot), not a label of the asking config.

    Raises:
        ValueError: if no record produced the capture, one lacks an execution revision, or its records disagree.
    """
    seen = set()
    for record in records.produced_by(capture):
        receiver = record.config.get("receiver")
        revision = record.provenance.execution.get("revision")
        if record.kind != "capture" or not isinstance(receiver, dict) or not revision:
            raise ValueError(f"capture {capture[:12]}: record {record.record_id()[:12]} names no receiver revision")
        seen.add((str(receiver["model_id"]), str(revision)))
    if len(seen) != 1:
        raise ValueError(f"capture {capture[:12]}: its records name receivers {sorted(seen)} (one expected)")
    ((model_id, revision),) = seen
    return {"model_id": model_id, "revision": revision}


def fit_ridge(spec: RidgeSpec, store: ArtifactStore, receiver: Mapping[str, str]) -> ArtifactRef:
    """Fit every layer and commit a ``ridge_init`` artifact (CPU; float64 solves).

    ``receiver`` is the capture's (:func:`capture_receiver`), recorded so every consumer checks it.

    The fit runs on the fit-set objects the capture holds, in the fit set's order: a capture leaves out objects
    without a screened description, as the old caption fit kept only proteins with a caption state. The count
    left out is recorded (``fit_left_out``) beside ``n_fit``. A spec with ``exclude`` also leaves out those objects
    before anything is fit; the count removed is recorded as ``excluded`` (those the capture holds: ``n_fit`` +
    ``fit_left_out`` + ``excluded`` is the fit set's size).

    Raises:
        KeyError: for a fit object missing from the embeddings.
        FileNotFoundError: if the capture holds no read-token means.
        ValueError: if the fit set is not an object list, holds an object outside the ``train`` split, the capture
            holds none of it, the capture left nothing out yet lacks fit objects (a capture of another object list),
            the embeddings are not from the input's recipe, a target layer is constant over the fit set, or an
            excluded object is not in the fit set.
    """
    targets = EmbeddingReader(store, ArtifactRef(spec.targets, EMBEDDINGS_KIND))
    listed, fit = fit_objects(store, spec.fit_set, spec.domain, targets, f"ridge fit {spec.target!r}")
    left_out = len(listed) - len(fit)
    excluded: dict[str, int] = {}
    if spec.exclude is not None:
        out = set(load_item_set(store, ArtifactRef(spec.exclude, ITEM_SET_KIND)).object_ids)
        stray = sorted(out - set(listed))
        if stray:
            raise ValueError(f"ridge fit {spec.target!r}: excluded objects {stray[:5]} are not in the fit set")
        kept = tuple(o for o in fit if o not in out)
        excluded = {"excluded": len(fit) - len(kept)}  # removed from the fit (an object the capture lacks is left out)
        fit = kept
    readers = [EmbeddingReader(store, ArtifactRef(d, EMBEDDINGS_KIND)) for d in spec.embeddings]
    loader = InputLoader(spec.input, readers, {InputKind.POOLED}, encoder=spec.encoder)
    batch = loader.batch(fit)
    X = batch.require(InputKind.POOLED).numpy().astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + SD_EPSILON
    folds = AlphaCV((X - mu) / sd, np.random.default_rng(spec.seed).permutation(len(X)) % spec.folds)
    del X
    staged = store.staging(f"ridge-{uuid.uuid4().hex}")
    np.save(staged / "mu.npy", mu.astype(np.float32))
    np.save(staged / "sd.npy", sd.astype(np.float32))
    alphas, r2 = {}, {}
    d_model = 0
    state = targets.pooled if spec.pool is None else targets.means
    for L in spec.layers:
        Y = state(fit, source_layer(L, spec.source_floor)).astype(np.float64)
        Yc = Y - Y.mean(0)
        variance = float((Yc**2).mean())
        if variance == 0.0:
            raise ValueError(f"ridge fit {spec.target!r}: the target at layer {L} is constant over the fit set (no R²)")
        alpha, cv_mse, W = folds.fit(Yc, np.asarray(spec.alphas, dtype=np.float64))
        alphas[L], r2[L] = alpha, 1.0 - cv_mse / variance
        d_model = Y.shape[1]
        np.save(staged / f"W.L{L}.npy", np.ascontiguousarray(W.T, dtype=np.float32))
        # bias: the floor rule (floor set) takes it from W's own target state; the paper's split (None) from the means
        b = Y if spec.source_floor is not None else targets.means(fit, L).astype(np.float64)
        np.save(staged / f"b.L{L}.npy", b.mean(0).astype(np.float32))
    meta = {
        "target": spec.target,
        "layers": list(spec.layers),
        "d_in": int(mu.shape[0]),
        "d_model": int(d_model),
        "n_fit": len(fit),
        "fit_left_out": left_out,
        "input": spec.input.identity(),
        "recipe_digest": spec.encoder.digest(),
        "receiver": dict(receiver),
        "alpha": {str(L): a for L, a in alphas.items()},
        "cv_r2": {str(L): v for L, v in r2.items()},
        "spec": spec.digest(),
    } | excluded
    if spec.pool is not None:  # a default fit's metadata, so its digest, stays what it was
        meta["pool"] = spec.pool
    if spec.source_floor is not None:
        meta["source_floor"] = spec.source_floor
    if spec.input.append is not None and loader.append is not None:
        meta["append_digest"] = loader.append.encoder.digest()
    return store.commit(staged, RIDGE_KIND, meta)


def load_ridge(store: ArtifactStore, ref: ArtifactRef) -> RidgeSolution:
    """Load a committed ridge fit (arrays memory-mapped, read-only).

    Raises:
        ValueError: if ``ref`` is not a ridge fit.
    """
    meta = _meta(store, ref)
    root = store.path(ref)
    layers = {
        int(L): RidgeLayer(W=np.load(root / f"W.L{L}.npy", mmap_mode="r"), b=np.load(root / f"b.L{L}.npy"))
        for L in meta["layers"]
    }
    return RidgeSolution(
        target=str(meta["target"]),
        mu=np.load(root / "mu.npy"),
        sd=np.load(root / "sd.npy"),
        layers=layers,
        input=meta.get("input"),
        recipe_digest=meta.get("recipe_digest"),
        receiver=meta.get("receiver"),
        append_digest=meta.get("append_digest"),
    )


def load_ridge_stats(store: ArtifactStore, ref: ArtifactRef) -> RidgeStats:
    """Return the α and CV R² a ridge fit recorded per layer.

    Raises:
        ValueError: if ``ref`` is not a ridge fit.
    """
    meta = _meta(store, ref)
    return RidgeStats(
        alpha={int(k): v for k, v in meta["alpha"].items()},
        cv_r2={int(k): v for k, v in meta["cv_r2"].items()},
        n_fit=int(meta["n_fit"]),
    )


def _meta(store: ArtifactStore, ref: ArtifactRef) -> Mapping[str, Any]:
    if ref.kind != RIDGE_KIND:
        raise ValueError(f"{ref.short} is a {ref.kind!r} artifact, not a ridge fit")
    return store.manifest(ref).meta
