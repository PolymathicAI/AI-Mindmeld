import numpy as np
import pytest
import torch
from pydantic import ValidationError

from latent_interconnect.writes import (
    AlphaSpec,
    Connector,
    ConnectorSpec,
    Entry,
    ExplicitLayers,
    InputSpec,
    LayerChoice,
    NoLayers,
    ObjectBatch,
    PendingError,
    PendingLayers,
    TokenSpec,
)
from latent_interconnect.writes.init import RidgeLayer, RidgeSolution

D_IN, D_MODEL, DEPTH = 6, 5, 8


def ridge(target, layers, seed):
    g = np.random.default_rng(seed)
    mu, sd = g.normal(size=D_IN).astype(np.float32), (g.random(D_IN) + 0.5).astype(np.float32)
    fits = {
        L: RidgeLayer(g.normal(size=(D_MODEL, D_IN)).astype(np.float32), g.normal(size=D_MODEL).astype(np.float32))
        for L in layers
    }
    return RidgeSolution(target, mu, sd, fits)


def spec(**kw):
    base = {
        "name": "t",
        "receiver": "tiny",
        "input": InputSpec(recipe="enc"),
        "tokens": (TokenSpec(target="name"),),
        "layers": ExplicitLayers(per_token=((2, 4, 6),)),
        "kind": "affine",
        "alpha": AlphaSpec(first=1.0, init=0.4, budget=1.0),
        "layer_choice": LayerChoice(note="test"),
    }
    base.update(kw)
    return ConnectorSpec(**base)


@pytest.fixture
def batch():
    x = torch.from_numpy(np.random.default_rng(9).normal(size=(3, D_IN)).astype(np.float32))
    return ObjectBatch(object_ids=("a", "b", "c"), pooled=x)


def test_affine_initial_state_is_the_ridge(batch):
    r = ridge("name", range(DEPTH), 0)
    c = Connector.from_ridge(spec(), DEPTH, {"name": r})
    assert c.entries == (Entry(0, 0), Entry(0, 2), Entry(0, 4), Entry(0, 6))
    w = c(batch)
    z = (batch.pooled.numpy() - r.mu) / r.sd
    for e in c.entries[1:]:
        expected = z @ r.layers[e.layer].W.T + r.layers[e.layer].b
        np.testing.assert_allclose(w.deltas[e].detach().numpy(), expected, rtol=1e-5, atol=1e-5)
    const = w.deltas[Entry(0, 0)].detach().numpy()
    np.testing.assert_allclose(const, np.broadcast_to(r.layers[0].b, const.shape))


def test_alpha_schedule(batch):
    c = Connector.from_ridge(spec(), DEPTH, {"name": ridge("name", range(DEPTH), 0)})
    assert float(c.alpha(Entry(0, 0))) == 1.0
    assert float(c.alpha(Entry(0, 2))) == 1.0  # first entry: fixed
    assert float(c.alpha(Entry(0, 4)).detach()) == pytest.approx(0.4)
    assert c.budget_penalty().item() == 0.0  # 0.8 < budget 1.0
    with torch.no_grad():
        for p in c.theta.values():
            p.fill_(5.0)
    assert c.budget_penalty().item() > 0


def test_r170_without_first_every_write_alpha_is_learned_from_init():
    c = Connector.from_ridge(spec(alpha=AlphaSpec(init=0.5)), DEPTH, {"name": ridge("name", range(DEPTH), 0)})
    assert float(c.alpha(Entry(0, 0))) == 1.0  # the layer-0 constant stays fixed
    assert set(c.theta) == {e.key() for e in c.entries[1:]}  # the first write layer is learned too
    assert all(float(c.alpha(e).detach()) == pytest.approx(0.5) for e in c.entries[1:])
    assert c.budget_penalty().item() == 0.0  # budget None


def test_layer0_ablation_reduces_to_the_constant_when_w0_is_zero(batch):
    r = ridge("name", range(DEPTH), 0)
    a = Connector.from_ridge(spec(layer0="affine", layers=NoLayers(), alpha=None), DEPTH, {"name": r})
    assert a.entries == (Entry(0, 0),)
    assert float(a.alpha(Entry(0, 0))) == 1.0
    with torch.no_grad():
        a.maps["t0_L0"].W.zero_()
    out = a(batch).deltas[Entry(0, 0)].detach().numpy()
    np.testing.assert_allclose(out, np.broadcast_to(r.layers[0].b, out.shape), rtol=1e-6)


def test_k_tokens_get_distinct_noisy_maps_and_targets(batch):
    name, caption = ridge("name", range(DEPTH), 0), ridge("caption", range(DEPTH), 1)
    s = spec(
        tokens=(TokenSpec(target="name", noise=0.03), TokenSpec(target="caption", noise=0.03)),
        layers=ExplicitLayers(per_token=((2, 4), (3, 5))),
    )
    c = Connector.from_ridge(s, DEPTH, {"name": name, "caption": caption})
    assert len(c.entries) == 6
    # caption token re-expressed on the name standardization: same function as its own ridge (noise aside)
    c0 = Connector.from_ridge(
        spec(tokens=(TokenSpec(target="name"), TokenSpec(target="caption")), layers=s.layers),
        DEPTH,
        {"name": name, "caption": caption},
    )
    x = batch.pooled.numpy()
    own = ((x - caption.mu) / caption.sd) @ caption.layers[3].W.T + caption.layers[3].b
    np.testing.assert_allclose(c0(batch).deltas[Entry(1, 3)].detach().numpy(), own, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(c(batch).deltas[Entry(1, 3)], c0(batch).deltas[Entry(1, 3)])


def test_unknown_kind_is_a_clear_error():
    with pytest.raises(KeyError, match="unknown map kind 'nope'"):
        Connector.from_ridge(spec(kind="nope"), DEPTH, {"name": ridge("name", range(DEPTH), 0)})


def test_checkpoint_round_trip_needs_no_ridge(batch):
    c = Connector.from_ridge(spec(), DEPTH, {"name": ridge("name", range(DEPTH), 0)})
    loaded = Connector.from_state(c.spec, DEPTH, D_IN, D_MODEL, c.state_dict())
    for e in c.entries:
        torch.testing.assert_close(loaded(batch).deltas[e], c(batch).deltas[e])
        torch.testing.assert_close(loaded.alpha(e), c.alpha(e))


def test_pending_configs_cannot_be_built():
    s = spec(layers=PendingLayers())
    assert s.pending() == ["layers"]
    with pytest.raises(PendingError, match="layers pending"):
        Connector.from_ridge(s, DEPTH, {"name": ridge("name", range(DEPTH), 0)})


def test_rationale_is_not_identity():
    a = spec(layer_choice=LayerChoice(note="one reason"))
    b = spec(layer_choice=LayerChoice(note="another reason", records=("a" * 64,)))
    assert a.digest() == b.digest()
    assert a.digest() != spec(seed=1).digest()


def test_every_token_has_a_ridge_target():
    with pytest.raises(ValidationError, match="target"):
        spec(tokens=(TokenSpec.model_validate({}),))
    with pytest.raises(ValidationError, match="init"):
        spec(init="random")
    two = spec(tokens=(TokenSpec(target="name"), TokenSpec(target="caption")), layers=PendingLayers())
    assert two.ridge_targets() == {"name", "caption"}


def test_each_token_inits_from_its_own_target():
    """The ESM-C 6B multi-token q6 run uses per-token init sources: token 1 starts from the even-layer caption ridge."""
    targets = ("name", "caption_even", "name", "caption_even")
    q6 = spec(
        tokens=tuple(TokenSpec(target=t) for t in targets),
        layers=ExplicitLayers(per_token=((4, 12, 20, 28, 36, 44, 52, 60),) * 4),
        layer0="constant",
    )
    depth = 61
    name, caption = ridge("name", range(depth), 0), ridge("caption_even", range(depth), 1)
    c = Connector.from_ridge(q6, depth, {"name": name, "caption_even": caption})
    on_name = caption.reexpress(name.mu, name.sd)
    for layer, key in ((0, "e"), (4, "b")):
        torch.testing.assert_close(
            getattr(c.maps[f"t1_L{layer}"], key).detach(), torch.as_tensor(on_name.layers[layer].b)
        )
        torch.testing.assert_close(getattr(c.maps[f"t0_L{layer}"], key).detach(), torch.as_tensor(name.layers[layer].b))
