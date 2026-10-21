"""Ridge fits: a planted map and the paper's fold and α rules."""

import dataclasses

import numpy as np
import pytest
from pydantic import ValidationError
from tests.unit.domain_fixture import tiny_domain

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.build.ridge import (
    RidgeSpec,
    fit_ridge,
    load_ridge,
    load_ridge_stats,
    source_layer,
)
from latent_interconnect.data.items import commit_item_set, object_list
from latent_interconnect.encoders.base import ExtractedBatch
from latent_interconnect.encoders.loader import InputLoader
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import OUTPUT_LAYER, EmbeddingReader, commit_embeddings
from latent_interconnect.writes.inputs import AppendedLayer, InputKind, InputSpec

ENC = EncoderRecipe(
    name="enc",
    extractor="mimic",
    domain="protein",
    model_id="5.7_main/v",
    revision="r",
    read="output",
    pooling="registers",
    environment="e",
    batch_tokens=1,
)
ALPHAS = tuple(float(a) for a in np.logspace(-3, 6, 10))
RECEIVER = {"model_id": "toy/receiver", "revision": "r1"}
"""The receiver a test capture holds (``build.ridge.capture_receiver`` reads it from a capture's records)."""


def planted(store: ArtifactStore, n: int = 400, d_in: int = 6, d_out: int = 5, noise: float = 0.01):
    rng = np.random.default_rng(0)
    ids = tuple(f"p{i}" for i in range(n))
    X = rng.normal(size=(n, d_in)) * rng.uniform(0.5, 3, d_in) + rng.normal(size=d_in)
    maps = {L: rng.normal(size=(d_out, d_in)) for L in (0, 7)}
    last = {L: (X @ M.T + noise * rng.normal(size=(n, d_out))).astype(np.float32) for L, M in maps.items()}
    mean = {L: (v + 1.0).astype(np.float32) for L, v in last.items()}
    batch = ExtractedBatch(ids, {OUTPUT_LAYER: X.astype(np.float32)}, {})
    emb = commit_embeddings(store, "x", [batch], {"recipe": ENC.model_dump(mode="json")})
    tgt = commit_embeddings(store, "t", [ExtractedBatch(ids, last, {}, mean)], {"recipe": "name", "target": "last"})
    domain_spec, domain = tiny_domain(store, ids)
    fit = commit_item_set(store, object_list("fit", domain, ids))
    spec = RidgeSpec(
        domain=domain_spec,
        input=InputSpec(recipe="enc"),
        encoder=ENC,
        embeddings=(emb.sha256,),
        targets=tgt.sha256,
        target="name",
        layers=(0, 7),
        fit_set=fit.sha256,
        alphas=ALPHAS,
        folds=5,
    )
    return spec, X, maps, mean


def test_planted_map_is_recovered_with_the_capture_means_as_bias(tmp_path):
    store = ArtifactStore(tmp_path)
    spec, X, maps, mean = planted(store)
    ref = fit_ridge(spec, store, RECEIVER)
    sol = load_ridge(store, ref)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    np.testing.assert_allclose(sol.mu, mu, rtol=1e-6)
    np.testing.assert_allclose(sol.sd, sd, rtol=1e-6)
    for L, M in maps.items():
        # W acts on standardized inputs: W_std = M diag(sd)
        np.testing.assert_allclose(sol.layers[L].W, M * sd[None, :], rtol=2e-2, atol=2e-2)
        np.testing.assert_allclose(sol.layers[L].b, mean[L].astype(np.float64).mean(0), rtol=1e-6)
    stats = load_ridge_stats(store, ref)
    assert stats.n_fit == 400 and set(stats.alpha) == {0, 7}
    assert all(v > 0.99 for v in stats.cv_r2.values())
    assert all(a in ALPHAS and a < 10 for a in stats.alpha.values())  # little noise: a small α wins


def test_fit_follows_the_paper_fold_and_alpha_rule(tmp_path):
    store = ArtifactStore(tmp_path)
    spec, X, _, _ = planted(store, n=120, noise=3.0)
    ref = fit_ridge(spec, store, RECEIVER)
    Y = EmbeddingReader(store, ArtifactRef(spec.targets, "embeddings"))
    Yc = Y.pooled([f"p{i}" for i in range(120)], 7).astype(np.float64)
    Yc -= Yc.mean(0)
    X32 = X.astype(np.float32).astype(np.float64)  # the stored embeddings are float32
    Z = (X32 - X32.mean(0)) / (X32.std(0) + 1e-8)
    folds = np.random.default_rng(0).permutation(120) % 5
    err = []
    for a in ALPHAS:  # direct ridge per fold, summed over folds and outputs
        e = 0.0
        for f in range(5):
            tr, te = folds != f, folds == f
            B = np.linalg.solve(Z[tr].T @ Z[tr] + a * np.eye(Z.shape[1]), Z[tr].T @ Yc[tr])
            e += ((Yc[te] - Z[te] @ B) ** 2).sum()
        err.append(e)
    best = int(np.argmin(err))
    stats = load_ridge_stats(store, ref)
    assert stats.alpha[7] == ALPHAS[best]
    assert stats.cv_r2[7] == pytest.approx(1 - err[best] / Yc.size / (Yc**2).mean(), rel=1e-9)
    B = np.linalg.solve(Z.T @ Z + ALPHAS[best] * np.eye(Z.shape[1]), Z.T @ Yc)
    np.testing.assert_allclose(load_ridge(store, ref).layers[7].W, B.T, rtol=1e-5, atol=1e-6)


def test_ridge_stage_reports_its_inputs(tmp_path, monkeypatch):
    from latent_interconnect.run import ridge as ridge_stage
    from latent_interconnect.run.ridge import _run
    from latent_interconnect.workspace import Workspace

    monkeypatch.setenv("LIP_DATA", str(tmp_path))
    ws = Workspace.from_env()
    spec, *_ = planted(ws.artifacts, n=60)
    seen = []
    monkeypatch.setattr(ridge_stage, "capture_receiver", lambda records, capture: seen.append(capture) or RECEIVER)
    result = _run(spec, ws)
    assert seen == [spec.targets] and ws.artifacts.manifest(result.outputs["ridge"]).meta["receiver"] == RECEIVER
    assert result.outputs["ridge"].kind == "ridge_init"
    assert set(result.inputs) == {"embeddings_0", "targets", "fit_set", "domain"}
    assert result.inputs["domain"].sha256 == spec.domain.data
    assert result.inputs["fit_set"].sha256 == spec.fit_set


def test_fit_set_must_be_train_only(tmp_path):
    store = ArtifactStore(tmp_path)
    spec, *_ = planted(store, n=20)
    ids = tuple(f"p{i}" for i in range(20))
    other, _ = tiny_domain(store, ids, split=["train"] * 19 + ["val"])
    with pytest.raises(ValueError, match="outside"):
        fit_ridge(spec.model_copy(update={"domain": other}), store, RECEIVER)


def test_the_fit_keeps_the_captured_objects_and_counts_the_rest(tmp_path):
    store = ArtifactStore(tmp_path)
    spec, X, _, mean = planted(store, n=60)
    ids = tuple(f"p{i}" for i in range(60))
    kept = ids[5:]  # the capture left out p0..p4 (no screened description)
    rng = np.random.default_rng(3)
    last = {L: rng.normal(size=(55, 5)).astype(np.float32) for L in (0, 7)}
    means = {L: m[5:] for L, m in mean.items()}
    tgt = commit_embeddings(
        store, "t2", [ExtractedBatch(kept, last, {}, means)], {"recipe": "caption", "target": "mean", "left_out": 5}
    )
    ref = fit_ridge(spec.model_copy(update={"targets": tgt.sha256}), store, RECEIVER)
    meta = store.manifest(ref).meta
    assert meta["n_fit"] == 55 and meta["fit_left_out"] == 5
    np.testing.assert_allclose(load_ridge(store, ref).sd, X[5:].astype(np.float32).std(0) + 1e-8, rtol=1e-5)
    nothing = commit_embeddings(store, "t3", [ExtractedBatch(("zz",), {0: last[0][:1]}, {}, {0: last[0][:1]})], {})
    with pytest.raises(ValueError, match="holds none"):
        fit_ridge(spec.model_copy(update={"targets": nothing.sha256}), store, RECEIVER)


def test_the_held_out_scan_ridge_fits_without_the_excluded_objects(tmp_path):
    """With ``exclude`` every fit-derived quantity (μ, σ, ``b``, α, ``W``) is the fit on the rest of the fit set;
    a production ridge (no ``exclude``) keeps its identity."""
    store = ArtifactStore(tmp_path)
    spec, *_ = planted(store, n=80, noise=1.0)
    ids = tuple(f"p{i}" for i in range(80))
    scan = ids[::4]
    rest = tuple(o for o in ids if o not in scan)
    domain_spec, domain = tiny_domain(store, ids)
    held_out = spec.model_copy(
        update={"domain": domain_spec, "exclude": commit_item_set(store, object_list("scan", domain, scan)).sha256}
    )
    rest_spec = spec.model_copy(
        update={"domain": domain_spec, "fit_set": commit_item_set(store, object_list("rest", domain, rest)).sha256}
    )
    a, b = fit_ridge(held_out, store, RECEIVER), fit_ridge(rest_spec, store, RECEIVER)
    sa, sb = load_ridge(store, a), load_ridge(store, b)
    np.testing.assert_array_equal(sa.mu, sb.mu)
    np.testing.assert_array_equal(sa.sd, sb.sd)
    for L in (0, 7):
        np.testing.assert_array_equal(sa.layers[L].W, sb.layers[L].W)
        np.testing.assert_array_equal(sa.layers[L].b, sb.layers[L].b)
    assert load_ridge_stats(store, a) == load_ridge_stats(store, b)
    meta = store.manifest(a).meta
    assert meta["n_fit"] == len(rest) and meta["excluded"] == len(scan) and "excluded" not in store.manifest(b).meta
    assert "exclude" not in spec.identity() and "exclude" in held_out.identity()
    stray = commit_item_set(store, object_list("stray", domain, scan))
    with pytest.raises(ValueError, match="are not in the fit set"):
        fit_ridge(rest_spec.model_copy(update={"exclude": stray.sha256}), store, RECEIVER)


def test_pool_mean_fits_the_read_token_means_and_the_default_keeps_its_identity(tmp_path):
    """``pool: mean`` fits ``W`` to the capture's read-token means (``b`` unchanged): the fit equals the default fit
    of a capture storing those means as its pooled state. Unset, the field is left out of the identity and the
    metadata, so a stored ridge keeps its digests."""
    store = ArtifactStore(tmp_path)
    spec, X, _, _ = planted(store, n=120, noise=1.0)
    ids = tuple(f"p{i}" for i in range(120))
    rng = np.random.default_rng(6)
    last = EmbeddingReader(store, ArtifactRef(spec.targets, "embeddings"))
    mean = {L: (X @ rng.normal(size=(6, 5)) + rng.normal(size=(120, 5))).astype(np.float32) for L in (0, 7)}
    both = commit_embeddings(
        store, "both", [ExtractedBatch(ids, {L: last.pooled(ids, L) for L in (0, 7)}, {}, mean)], {"target": "last"}
    )
    as_pooled = commit_embeddings(store, "as-pooled", [ExtractedBatch(ids, mean, {}, mean)], {"target": "mean"})
    default = spec.model_copy(update={"targets": both.sha256})
    pooled = RidgeSpec.model_validate(default.model_dump() | {"pool": "mean"})
    a, b = (
        fit_ridge(pooled, store, RECEIVER),
        fit_ridge(default.model_copy(update={"targets": as_pooled.sha256}), store, RECEIVER),
    )
    sa, sb = load_ridge(store, a), load_ridge(store, b)
    for L in (0, 7):
        np.testing.assert_array_equal(sa.layers[L].W, sb.layers[L].W)
        np.testing.assert_array_equal(sa.layers[L].b, sb.layers[L].b)
        np.testing.assert_allclose(sa.layers[L].b, mean[L].astype(np.float64).mean(0), rtol=1e-6)
    assert load_ridge_stats(store, a) == load_ridge_stats(store, b)
    assert load_ridge_stats(store, a) != load_ridge_stats(store, fit_ridge(default, store, RECEIVER))  # another target
    assert store.manifest(a).meta["pool"] == "mean" and "pool" not in store.manifest(b).meta
    assert "pool" not in default.identity() and pooled.digest() != default.digest()
    assert RidgeSpec.model_validate(default.model_dump() | {"pool": None}).digest() == default.digest()


def test_a_mean_fit_on_a_capture_without_means_fails(tmp_path):
    """Never a fallback to the pooled state."""
    store = ArtifactStore(tmp_path)
    spec, X, *_ = planted(store, n=40)
    ids = tuple(f"p{i}" for i in range(40))
    bare = commit_embeddings(store, "bare", [ExtractedBatch(ids, {0: X[:, :5].astype(np.float32)}, {})], {})
    pooled = RidgeSpec.model_validate(spec.model_dump() | {"targets": bare.sha256, "layers": [0], "pool": "mean"})
    with pytest.raises(FileNotFoundError, match=r"mean\.L0\.npy"):
        fit_ridge(pooled, store, RECEIVER)


def test_the_cv_error_in_the_svd_basis_equals_the_direct_sum():
    """``AlphaCV.fit`` computes each fold's error from the SVD factors; it equals summing the held-out residuals."""
    from latent_interconnect.build.ridge import AlphaCV

    g = np.random.default_rng(1)
    Z, Y = g.normal(size=(90, 7)), g.normal(size=(90, 12))
    Y -= Y.mean(0)
    labels = g.permutation(90) % 5
    alpha, mse, _ = AlphaCV(Z, labels).fit(Y, np.asarray(ALPHAS))
    err = []
    for a in ALPHAS:
        e = 0.0
        for f in range(5):
            tr, te = labels != f, labels == f
            B = np.linalg.solve(Z[tr].T @ Z[tr] + a * np.eye(7), Z[tr].T @ Y[tr])
            e += ((Y[te] - Z[te] @ B) ** 2).sum()
        err.append(e)
    assert alpha == ALPHAS[int(np.argmin(err))] and mse == pytest.approx(min(err) / Y.size, rel=1e-10)


def test_the_folds_follow_the_fit_set_order_not_the_capture_order(tmp_path):
    """The folds follow the fit set's order (the paper's fold rule), not the capture's storage order."""
    store = ArtifactStore(tmp_path)
    spec, *_ = planted(store, n=120, noise=3.0)
    ids = tuple(f"p{i}" for i in range(120))
    shuffled = tuple(np.random.default_rng(5).permutation(ids))
    reader = EmbeddingReader(store, ArtifactRef(spec.targets, "embeddings"))
    last = {L: reader.pooled(list(shuffled), L) for L in (0, 7)}
    means = {L: reader.means(list(shuffled), L) for L in (0, 7)}
    meta = {"recipe": "name", "target": "last"}
    reordered = commit_embeddings(store, "t-shuffled", [ExtractedBatch(shuffled, last, {}, means)], meta)
    a = load_ridge_stats(store, fit_ridge(spec, store, RECEIVER))
    b = load_ridge_stats(store, fit_ridge(spec.model_copy(update={"targets": reordered.sha256}), store, RECEIVER))
    assert a.cv_r2 == b.cv_r2 and a.alpha == b.alpha  # the capture's storage order does not move the folds
    domain_spec, domain = tiny_domain(store, shuffled)
    refit = spec.model_copy(
        update={"fit_set": commit_item_set(store, object_list("fit2", domain, shuffled)).sha256, "domain": domain_spec}
    )
    c = load_ridge_stats(store, fit_ridge(refit, store, RECEIVER))
    assert c.cv_r2 != a.cv_r2  # the fit set's order does


def test_a_capture_of_another_object_list_is_refused(tmp_path):
    store = ArtifactStore(tmp_path)
    spec, *_ = planted(store, n=40)
    ids = tuple(f"p{i}" for i in range(40))
    x = {L: np.zeros((35, 5), np.float32) for L in (0, 7)}
    other = commit_embeddings(store, "t-other", [ExtractedBatch(ids[5:], x, {}, x)], {"recipe": "name", "left_out": 0})
    with pytest.raises(ValueError, match="not in the capture"):
        fit_ridge(spec.model_copy(update={"targets": other.sha256}), store, RECEIVER)


def test_a_constant_target_layer_is_refused_by_name(tmp_path):
    """A target with no variance over the fit set has no R²: a ValueError naming the layer, not a ZeroDivisionError."""
    store = ArtifactStore(tmp_path)
    spec, _, _, mean = planted(store, n=60)
    ids = tuple(f"p{i}" for i in range(60))
    rng = np.random.default_rng(4)
    last = {0: rng.normal(size=(60, 5)).astype(np.float32), 7: np.full((60, 5), 0.25, np.float32)}
    tgt = commit_embeddings(store, "flat", [ExtractedBatch(ids, last, {}, mean)], {"recipe": "name", "target": "last"})
    with pytest.raises(ValueError, match="target at layer 7 is constant"):
        fit_ridge(spec.model_copy(update={"targets": tgt.sha256}), store, RECEIVER)


def test_an_appended_extraction_is_the_manual_concatenation(tmp_path):
    """``input.append`` concatenates a second extraction's pooled vector (another recipe, one layer) after the
    input's, per object by id: a fit on two artifacts splitting X's columns (the second in another object order) is the
    fit on X whole; the ridge records the append and an input without one keeps its earlier digest."""
    store = ArtifactStore(tmp_path)
    spec, X, *_ = planted(store)
    ids = tuple(f"p{i}" for i in range(len(X)))
    head = commit_embeddings(
        store,
        "h",
        [ExtractedBatch(ids, {OUTPUT_LAYER: X[:, :4].astype(np.float32)}, {})],
        {"recipe": ENC.model_dump(mode="json")},
    )
    enc2 = ENC.model_copy(update={"name": "enc2", "read": (3,), "pooling": "mean"})
    back = ids[::-1]
    tail = commit_embeddings(
        store,
        "t2",
        [ExtractedBatch(back, {3: X[::-1, 4:].astype(np.float32)}, {})],
        {"recipe": enc2.model_dump(mode="json")},
    )
    joint = spec.model_copy(
        update={
            "input": InputSpec(recipe="enc", append=AppendedLayer(recipe="enc2", layer=3)),
            "embeddings": (head.sha256, tail.sha256),
        }
    )
    readers = [EmbeddingReader(store, ArtifactRef(d, "embeddings")) for d in joint.embeddings]
    got = InputLoader(joint.input, readers, {InputKind.POOLED}, encoder=ENC).batch(ids[5:9]).pooled
    np.testing.assert_array_equal(got.numpy(), X[5:9].astype(np.float32))
    a, b = load_ridge(store, fit_ridge(joint, store, RECEIVER)), load_ridge(store, fit_ridge(spec, store, RECEIVER))
    np.testing.assert_array_equal(a.mu, b.mu)
    for L in (0, 7):
        np.testing.assert_allclose(a.layers[L].W, b.layers[L].W, rtol=1e-6, atol=1e-6)
    assert store.manifest(fit_ridge(joint, store, RECEIVER)).meta["input"]["append"] == {"recipe": "enc2", "layer": 3}
    assert InputSpec(recipe="enc", layers=(3,), combine="concat").digest() == (
        "023a1a859c9aab51d1f4985a6ba5e3c49d90188396932b1f0770015775e79ace"  # at da372260, before `append`
    )
    with pytest.raises(ValueError, match="another recipe"):
        InputSpec(recipe="enc", append=AppendedLayer(recipe="enc", layer=3))


def test_an_appended_extraction_is_bound_by_its_recipe_content(tmp_path):
    """The loader refuses an appended extraction it cannot read (no reader of its recipe;
    residues), checks its readers by digest against a given ``append_encoder``, and a ridge records that digest, so a
    run or scan appending another content of the same-named recipe is refused."""
    store = ArtifactStore(tmp_path)
    spec, X, *_ = planted(store)
    ids = tuple(f"p{i}" for i in range(len(X)))
    head = commit_embeddings(
        store,
        "h",
        [ExtractedBatch(ids, {OUTPUT_LAYER: X[:, :4].astype(np.float32)}, {})],
        {"recipe": ENC.model_dump(mode="json")},
    )
    enc2 = ENC.model_copy(update={"name": "enc2", "read": (3,), "pooling": "mean"})
    tail = commit_embeddings(
        store,
        "t2",
        [ExtractedBatch(ids, {3: X[:, 4:].astype(np.float32)}, {})],
        {"recipe": enc2.model_dump(mode="json")},
    )
    joint = InputSpec(recipe="enc", append=AppendedLayer(recipe="enc2", layer=3))
    h, t = (EmbeddingReader(store, ArtifactRef(d.sha256, "embeddings")) for d in (head, tail))
    with pytest.raises(ValueError, match="none of the readers is of the appended 'enc2'"):
        InputLoader(joint, [h], {InputKind.POOLED}, encoder=ENC)
    with pytest.raises(ValueError, match="pooled vectors only"):
        InputLoader(joint, [h, t], {InputKind.POOLED, InputKind.RESIDUES}, encoder=ENC)
    revised = enc2.model_copy(update={"revision": "r2"})
    with pytest.raises(ValueError, match="digest"):
        InputLoader(joint, [h, t], {InputKind.POOLED}, encoder=ENC, append_encoder=revised)
    assert InputLoader(joint, [h, t], {InputKind.POOLED}, encoder=ENC, append_encoder=enc2).append is not None
    fit = load_ridge(
        store,
        fit_ridge(spec.model_copy(update={"input": joint, "embeddings": (head.sha256, tail.sha256)}), store, RECEIVER),
    )
    assert fit.append_digest == enc2.digest()

    loaded = enc2.digest()

    def check(fit, input=joint, digest=loaded):
        fit.require_fit("scan", input=input, recipe_digest=ENC.digest(), receiver=RECEIVER, append_digest=digest)

    check(fit)
    check(fit, digest=None)  # the init: loads nothing appended
    legacy = dataclasses.replace(fit, append_digest=None)  # a fit committed before the digest was recorded
    for digest in (enc2.digest(), None):
        with pytest.raises(ValueError, match="records no appended recipe digest"):
            check(legacy, digest=digest)
    with pytest.raises(ValueError, match="appended recipe content"):
        check(fit, digest=revised.digest())
    with pytest.raises(ValueError, match=r"fit on input .*'append'"):
        check(fit, input=InputSpec(recipe="enc"), digest=None)


@pytest.mark.parametrize("pool", [None, "mean"])
def test_a_source_floor_fits_each_layer_to_the_state_of_max_layer_floor_with_its_own_bias(tmp_path, pool):
    """With ``source_floor`` p* = 5, layer 3 (< p*) is fitted to layer 5's state, layers 5 and 7 (>= p*) to
    their own and layer 0 to its own, ``W`` and ``b`` both from the ``pool`` state (no last / mean split): the fit
    equals a default fit on a capture whose pooled state and means at each layer are that source state. Unset, the
    field is left out of the identity and the metadata."""
    store = ArtifactStore(tmp_path)
    spec, X, _, _ = planted(store, n=120, noise=1.0)
    ids = tuple(f"p{i}" for i in range(120))
    rng = np.random.default_rng(9)
    layers = (0, 3, 5, 7)
    last, mean = (
        {L: (X @ rng.normal(size=(6, 5)) + rng.normal(size=(120, 5)) + k).astype(np.float32) for L in layers}
        for k in (0.0, 2.0)
    )
    capture = commit_embeddings(store, "c", [ExtractedBatch(ids, last, {}, mean)], {"target": "last"})
    source = {0: 0, 3: 5, 5: 5, 7: 7}
    chosen = last if pool is None else mean
    moved = {L: chosen[source[L]] for L in layers}
    expected = commit_embeddings(store, "moved", [ExtractedBatch(ids, moved, {}, moved)], {"target": "last"})
    base = spec.model_copy(update={"targets": capture.sha256, "layers": layers})
    floored = RidgeSpec.model_validate(base.model_dump() | {"pool": pool, "source_floor": 5})
    a = fit_ridge(floored, store, RECEIVER)
    b = fit_ridge(base.model_copy(update={"targets": expected.sha256}), store, RECEIVER)
    sa, sb = load_ridge(store, a), load_ridge(store, b)
    for L in layers:
        np.testing.assert_array_equal(sa.layers[L].W, sb.layers[L].W)
        np.testing.assert_array_equal(sa.layers[L].b, sb.layers[L].b)
        np.testing.assert_allclose(sa.layers[L].b, chosen[source[L]].astype(np.float64).mean(0), rtol=1e-6)
    assert load_ridge_stats(store, a) == load_ridge_stats(store, b)
    assert store.manifest(a).meta["source_floor"] == 5 and "source_floor" not in store.manifest(b).meta
    assert "source_floor" not in base.identity() and floored.digest() != base.digest()
    assert RidgeSpec.model_validate(base.model_dump() | {"source_floor": None}).digest() == base.digest()
    assert [source_layer(L, 5) for L in layers] == [0, 5, 5, 7] and source_layer(3, None) == 3
    with pytest.raises(ValidationError):
        RidgeSpec.model_validate(base.model_dump() | {"source_floor": 0})


def test_the_bias_source_is_the_means_unset_and_the_target_state_with_a_source_floor(tmp_path):
    """The two bias conventions, pinned: unset (the paper's split), a last-token fit's ``b`` is the read-token means'
    mean, not the last token's; with ``source_floor``, ``b`` is the mean of the same state ``W`` maps to."""
    store = ArtifactStore(tmp_path)
    spec, _, _, mean = planted(store, n=80)  # the capture's means are its last-token states + 1
    last = EmbeddingReader(store, ArtifactRef(spec.targets, "embeddings"))
    ids = tuple(f"p{i}" for i in range(80))
    split = load_ridge(store, fit_ridge(spec, store, RECEIVER))
    floored = load_ridge(store, fit_ridge(spec.model_copy(update={"source_floor": 7}), store, RECEIVER))
    for L, source in ((0, 0), (7, 7)):
        own, means = last.pooled(ids, source).astype(np.float64).mean(0), mean[source].astype(np.float64).mean(0)
        np.testing.assert_allclose(split.layers[L].b, means, rtol=1e-6)
        np.testing.assert_allclose(floored.layers[L].b, own, rtol=1e-6)
        assert not np.allclose(own, means)
        np.testing.assert_array_equal(split.layers[L].W, floored.layers[L].W)  # same target state at L >= p* and 0


def test_capture_receiver_reads_the_captures_records_and_refuses_none_or_two():
    """The receiver a fit records comes from the capture's own record: its config's model id and the revision its
    execution loaded; no record, a record without a revision, or two disagreeing records refuse the fit."""
    from types import SimpleNamespace

    from latent_interconnect.build.ridge import capture_receiver

    def record(revision, model_id="org/m"):
        provenance = SimpleNamespace(execution={} if revision is None else {"revision": revision})
        return SimpleNamespace(
            kind="capture",
            config={"receiver": {"model_id": model_id}},
            provenance=provenance,
            record_id=lambda: "e" * 64,
        )

    def store(*records):
        return SimpleNamespace(produced_by=lambda _sha: list(records))

    assert capture_receiver(store(record("abc"), record("abc")), "c" * 64) == {"model_id": "org/m", "revision": "abc"}
    for records, why in (((), r"receivers \[\]"), ((record(None),), "names no receiver revision")):
        with pytest.raises(ValueError, match=why):
            capture_receiver(store(*records), "c" * 64)
    with pytest.raises(ValueError, match="one expected"):
        capture_receiver(store(record("abc"), record("def")), "c" * 64)
