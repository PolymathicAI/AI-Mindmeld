"""Checkpoints (``writes.checkpoint``), the initial state (``build.init_state``) and the ``init`` stage."""

import json

import numpy as np
import pytest
import torch
from tests.unit.tiny_receiver import ROOT

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.build import init_state
from latent_interconnect.build.init_state import InitSpec, commit_initial_state, initial_connector
from latent_interconnect.config import load_config
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.receivers.base import ReceiverSpec
from latent_interconnect.run import init as init_stage
from latent_interconnect.workspace import Workspace
from latent_interconnect.writes import Connector, ConnectorSpec, ObjectBatch
from latent_interconnect.writes.checkpoint import (
    commit_checkpoint,
    load_connector,
    read_checkpoint,
)
from latent_interconnect.writes.init import RidgeLayer, RidgeSolution
from latent_interconnect.writes.inputs import InputSpec

D_IN, D_MODEL, DEPTH = 12, 16, 8
LAYERS = (0, 2, 3, 5, 6)  # 0 = the constant; 2 = the first layer (α = 1); 3, 5, 6 = later layers (learned α)
ENCODER = load_config(ROOT / "configs" / "encoders" / "esmc-600m-l36.yaml", EncoderRecipe)


def spec(**update):
    base = {
        "name": "tiny-affine",
        "receiver": "tiny-llama",
        "input": {"recipe": "esmc-600m-l36", "layers": [36]},  # explicit read indices (one representation)
        "tokens": [{"target": "name"}],
        "layer0": "constant",
        "layers": {"rule": "explicit", "per_token": [list(LAYERS[1:])]},
        "kind": "affine",
        "alpha": {"first": 1.0, "init": 0.4, "learned": True, "budget": 2.0, "penalty_k": 5.0},
        "layer_choice": {"note": "test"},
    }
    return ConnectorSpec.model_validate(base | update)


def receiver(**update):
    tiny = load_config(ROOT / "configs" / "receivers" / "tiny-llama.yaml", ReceiverSpec)
    return tiny.model_copy(update={"depth": DEPTH, "d_model": D_MODEL} | update)


def bound_to(receiver_spec):
    """A fit's recorded receiver, as ``build.ridge.capture_receiver`` reads it."""
    return {"model_id": receiver_spec.model_id, "revision": receiver_spec.revision}


def ridge(seed=0, layers=(*LAYERS, 4), input=None, encoder=ENCODER):
    g = np.random.default_rng(seed)
    per = {
        L: RidgeLayer(W=g.normal(size=(D_MODEL, D_IN)).astype(np.float32), b=g.normal(size=D_MODEL).astype(np.float32))
        for L in layers
    }
    mu = g.normal(size=D_IN).astype(np.float32)
    sd = (g.random(D_IN) + 0.5).astype(np.float32)
    fit_input = InputSpec.model_validate(input or spec().input.model_dump(mode="json"))
    return RidgeSolution(
        target="name",
        mu=mu,
        sd=sd,
        layers=per,
        input=fit_input.identity(),
        recipe_digest=encoder.digest(),
        receiver=bound_to(receiver()),
    )


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


def test_commit_and_load_round_trip_deterministically(store):
    c = Connector.from_ridge(spec(), DEPTH, {"name": ridge()})
    ref = commit_checkpoint(c, 7, store, encoder=ENCODER)
    assert ref.kind == "checkpoint" and store.manifest(ref).meta["step"] == 7
    assert commit_checkpoint(c, 7, store, encoder=ENCODER) == ref  # same content, same identity
    loaded, step = load_connector(store, ref)
    assert step == 7 and loaded.spec.digest() == c.spec.digest()
    assert loaded.state_dict().keys() == c.state_dict().keys()
    for k, v in c.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v), k
    again, _ = read_checkpoint(store.path(ref))
    batch = ObjectBatch(("a", "b"), pooled=torch.randn(2, D_IN))
    for e in c.entries:
        assert torch.equal(again(batch).deltas[e], c(batch).deltas[e])
        assert torch.equal(again.alpha(e), c.alpha(e))
    with pytest.raises(ValueError, match="not a checkpoint"):
        load_connector(store, ArtifactRef(ref.sha256, "embeddings"))


def test_a_checkpoint_records_its_encoder_recipe(store):
    """One source of a connector's encoder recipe: the checkpoint (the suite stage reads it there)."""
    from latent_interconnect.writes.checkpoint import MissingEncoderError, checkpoint_encoder

    c = Connector.from_ridge(spec(), DEPTH, {"name": ridge()})
    ref = commit_checkpoint(c, 0, store, encoder=ENCODER)
    assert EncoderRecipe.model_validate(checkpoint_encoder(store, ref)) == ENCODER
    with pytest.raises(ValueError, match="reads 'esmc-600m-l36', not 'other'"):
        commit_checkpoint(c, 0, store, encoder=ENCODER.model_copy(update={"name": "other"}))
    old = store.staging("old-checkpoint")  # a checkpoint from before the recipe was recorded
    header = json.loads((store.path(ref) / "connector.json").read_text())
    del header["encoder"]
    (old / "connector.json").write_text(json.dumps(header))
    (old / "weights.safetensors").write_bytes((store.path(ref) / "weights.safetensors").read_bytes())
    with pytest.raises(MissingEncoderError, match="records no encoder recipe"):
        checkpoint_encoder(store, store.commit(old, "checkpoint", {}))


def test_rewording_the_rationale_changes_no_checkpoint(store):
    sol = ridge()
    a = Connector.from_ridge(spec(), DEPTH, {"name": sol})
    b = Connector.from_ridge(spec(layer_choice={"note": "reworded", "decisions": ["other"]}), DEPTH, {"name": sol})
    assert commit_checkpoint(a, 0, store, encoder=ENCODER) == commit_checkpoint(b, 0, store, encoder=ENCODER)


def init_spec(encoder=ENCODER, **update):
    return InitSpec(connector=spec(), receiver=receiver(**update), encoder=encoder, ridge={"name": "ab" * 32})


def test_initial_connector_is_from_ridge_on_the_cited_ridge(store, monkeypatch):
    sol = ridge()
    seen = []

    def load(_store, ref):
        seen.append(ref)
        return sol

    monkeypatch.setattr(init_state, "load_ridge", load)
    c = initial_connector(init_spec(), store)
    assert seen == [ArtifactRef("ab" * 32, "ridge_init")]
    for k, v in Connector.from_ridge(spec(), DEPTH, {"name": sol}).state_dict().items():
        assert torch.equal(c.state_dict()[k], v)
    ref = commit_initial_state(init_spec(), store)
    assert load_connector(store, ref)[1] == 0
    monkeypatch.setattr(init_state, "load_ridge", lambda _s, _r: ridge(layers=(0, 2, 3)))
    with pytest.raises(ValueError, match="no layer 5"):
        initial_connector(init_spec(), store)
    monkeypatch.setattr(init_state, "load_ridge", lambda _s, _r: sol)
    with pytest.raises(ValueError, match="receiver is 32"):
        initial_connector(init_spec(d_model=32), store)
    with pytest.raises(ValueError, match="not all within its read"):  # the init spec applies the rule at load
        init_spec(encoder=ENCODER.model_copy(update={"read": (12,)}))
    with pytest.raises(ValueError, match="reads recipe 'esmc-600m-l36', not 'other'"):
        init_spec(encoder=ENCODER.model_copy(update={"name": "other"}))
    with pytest.raises(ValueError, match="choose the layers first"):  # an init config cannot wait for its layers
        InitSpec(
            connector=spec(input={"recipe": "esmc-600m-l36", "layers": "pending"}),
            receiver=receiver(),
            encoder=ENCODER,
            ridge={"name": "ab" * 32},
        )
    with pytest.raises(ValueError, match="layers pending"):  # nor for its write layers
        InitSpec(
            connector=spec(layers={"rule": "pending"}), receiver=receiver(), encoder=ENCODER, ridge={"name": "ab" * 32}
        )


def test_a_ridge_fit_on_another_recipe_is_refused(store, monkeypatch):
    """Every ridge's recorded recipe is the connector's input recipe."""
    scan = ENCODER.model_copy(update={"name": "esmc-600m-l36-scan", "read": "all"})
    on_scan = ridge(input={"recipe": scan.name, "layers": [36]}, encoder=scan)
    monkeypatch.setattr(init_state, "load_ridge", lambda _s, _r: on_scan)
    with pytest.raises(ValueError, match=r"fit on input .*'esmc-600m-l36-scan'.*; the connector is .*'esmc-600m-l36'"):
        initial_connector(init_spec(), store)


def test_an_appended_extraction_binds_the_ridges_and_the_checkpoint(store, monkeypatch):
    """A ridge fit with another appended extraction, or recording no appended recipe digest,
    is refused at init, and a checkpoint records the appended recipe (absent without one, so other checkpoints keep
    their identity); a connector that appends is never committed without it."""
    import dataclasses

    from latent_interconnect.writes.checkpoint import MissingEncoderError, checkpoint_encoder

    joint = {"recipe": "esmc-600m-l36", "layers": [36], "append": {"recipe": "esmc-600m-l36-scan", "layer": 20}}
    other16 = joint | {"append": {"recipe": "esmc-600m-l36-scan", "layer": 16}}
    appended = InitSpec(connector=spec(input=joint), receiver=receiver(), encoder=ENCODER, ridge={"name": "ab" * 32})
    scan = ENCODER.model_copy(update={"name": "esmc-600m-l36-scan", "read": "all"})
    fit = ridge(input=joint)
    monkeypatch.setattr(init_state, "load_ridge", lambda _s, _r: fit)  # no recorded digest: refit
    with pytest.raises(ValueError, match="records no appended recipe digest"):
        initial_connector(appended, store)
    fit = dataclasses.replace(fit, append_digest=scan.digest())
    monkeypatch.setattr(
        init_state, "load_ridge", lambda _s, _r: dataclasses.replace(fit, input=ridge(input=other16).input)
    )
    with pytest.raises(ValueError, match=r"fit on input .*'layer': 16.*; the connector is .*'layer': 20"):
        initial_connector(appended, store)
    monkeypatch.setattr(init_state, "load_ridge", lambda _s, _r: fit)
    c = initial_connector(appended, store)
    ref = commit_checkpoint(c, 0, store, encoder=ENCODER, append_encoder=scan)
    assert EncoderRecipe.model_validate(checkpoint_encoder(store, ref, "append_encoder")) == scan
    plain = commit_checkpoint(Connector.from_ridge(spec(), DEPTH, {"name": ridge()}), 0, store, encoder=ENCODER)
    with pytest.raises(MissingEncoderError, match="records no appended recipe"):  # omitted without an append
        checkpoint_encoder(store, plain, "append_encoder")
    with pytest.raises(ValueError, match="does not append"):
        commit_checkpoint(c, 0, store, encoder=ENCODER, append_encoder=ENCODER)
    with pytest.raises(MissingEncoderError, match="appends 'esmc-600m-l36-scan': give its recipe"):
        commit_checkpoint(c, 0, store, encoder=ENCODER)


def test_the_init_step_0_records_the_appended_recipe_as_the_trainer_does(store, monkeypatch):
    """The step-0 gap: the init stage checks ``append_encoder`` against the ridges' recorded digest and records it,
    so its step 0 is the trainer's checkpoint (``commit_checkpoint`` with ``append_encoder``) and a step-0 suite checks
    the appended inputs against it instead of taking the recipe from their artifacts."""
    import dataclasses

    from latent_interconnect.writes.checkpoint import checkpoint_encoder

    joint = {"recipe": "esmc-600m-l36", "layers": [36], "append": {"recipe": "esmc-600m-l36-scan", "layer": 20}}
    scan = ENCODER.model_copy(update={"name": "esmc-600m-l36-scan", "read": "all"})
    fit = dataclasses.replace(ridge(input=joint), append_digest=scan.digest())
    monkeypatch.setattr(init_state, "load_ridge", lambda _s, _r: fit)
    base = {"connector": spec(input=joint), "receiver": receiver(), "encoder": ENCODER, "ridge": {"name": "ab" * 32}}
    ref = commit_initial_state(InitSpec(**base, append_encoder=scan), store)
    assert EncoderRecipe.model_validate(checkpoint_encoder(store, ref, "append_encoder")) == scan
    trained = commit_checkpoint(
        initial_connector(InitSpec(**base), store), 0, store, encoder=ENCODER, append_encoder=scan
    )
    assert ref == trained
    with pytest.raises(ValueError, match="give its append_encoder"):
        commit_initial_state(InitSpec(**base), store)
    with pytest.raises(ValueError, match="appended recipe content"):  # another content than the ridges were fit with
        commit_initial_state(InitSpec(**base, append_encoder=scan.model_copy(update={"revision": "other"})), store)
    with pytest.raises(ValueError, match="not 'esmc-600m-l36'"):
        InitSpec(**base, append_encoder=ENCODER)
    with pytest.raises(ValueError, match="appends no extraction"):
        InitSpec(**base | {"connector": spec()}, append_encoder=scan)


def test_init_stage_commits_step_0_and_cites_its_ridge(tmp_path, monkeypatch):
    monkeypatch.setenv("LIP_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(init_state, "load_ridge", lambda _s, _r: ridge())
    ws = Workspace.from_env()
    result = init_stage._run(init_spec(), ws)
    assert set(result.outputs) == {"step_000000"}
    assert result.inputs == {"ridge/name": ArtifactRef("ab" * 32, "ridge_init")}
    assert load_connector(ws.artifacts, result.outputs["step_000000"])[1] == 0


def test_a_committed_fit_initializes_only_its_own_input_target_and_receiver(tmp_path):
    """A fit on read layer 1 of a two-layer extraction, for target ``name``, from one
    receiver's capture. Its own connector initializes; one reading layer 2, citing it for ``caption``, or on another
    receiver revision is refused."""
    from tests.unit.test_ridge import ENC, planted

    from latent_interconnect.build.ridge import fit_ridge
    from latent_interconnect.encoders.base import ExtractedBatch
    from latent_interconnect.encoders.store import commit_embeddings

    store = ArtifactStore(tmp_path)
    original, X, _, _ = planted(store, n=20)
    encoder = ENC.model_copy(update={"read": (1, 2)})
    ids = tuple(f"p{i}" for i in range(20))
    layers = {1: X.astype(np.float32), 2: (X + 10).astype(np.float32)}
    emb = commit_embeddings(
        store, "two", [ExtractedBatch(ids, layers, {})], {"recipe": encoder.model_dump(mode="json")}
    )
    update = {"encoder": encoder, "input": InputSpec(recipe="enc", layers=(1,)), "embeddings": (emb.sha256,)}
    r = receiver(depth=8, d_model=5)
    fitted = fit_ridge(original.model_copy(update=update), store, bound_to(r))

    def init(read_layer=1, target="name", on=r):
        c = ConnectorSpec.model_validate(
            {
                "name": "probe",
                "receiver": on.name,
                "input": {"recipe": "enc", "layers": [read_layer]},
                "tokens": [{"target": target}],
                "layer0": "constant",
                "layers": {"rule": "explicit", "per_token": [[7]]},
                "kind": "affine",
                "alpha": {"init": 0.5},
            }
        )
        return initial_connector(
            InitSpec(connector=c, receiver=on, encoder=encoder, ridge={target: fitted.sha256}), store
        )

    assert init().d_in == 6  # the matched control
    with pytest.raises(ValueError, match=r"fit on input .*'layers': \[1\].*; the connector is .*'layers': \[2\]"):
        init(read_layer=2)
    with pytest.raises(ValueError, match="the ridge given for target 'caption' was fit to target 'name'"):
        init(target="caption")
    with pytest.raises(ValueError, match=r"fit to receiver .*; the connector is .*'revision': 'other'"):
        init(on=r.model_copy(update={"revision": "other"}))  # same shape, another receiver


RETRY = "conformer_retry_timeout_s"
"""The one option the runtime recipe adds to its source in these tests (the Uni-Mol2 conformer retry's)."""


def _source_fit(recipe=ENCODER, option=RETRY, ridges=("ab" * 32,)):
    from latent_interconnect.build.init_state import SourceFit

    return SourceFit(recipe=recipe, option=option, ridges=ridges)


def test_a_declared_source_fit_keeps_the_source_start_on_the_runtime_recipe(store, monkeypatch):
    """No refit on the runtime inputs: the ridge fit on the source recipe initialises a connector that reads
    the source recipe plus one declared option; the state is the source recipe's start, bit for bit, and the step-0
    checkpoint records the runtime recipe its inputs are checked against."""
    from latent_interconnect.writes.checkpoint import checkpoint_encoder

    sol = ridge()  # fit on ENCODER
    monkeypatch.setattr(init_state, "load_ridge", lambda _s, _r: sol)
    runtime = ENCODER.model_copy(update={"options": {**ENCODER.options, RETRY: 120}})
    assert runtime.name == ENCODER.name and runtime.digest() != ENCODER.digest()
    with pytest.raises(ValueError, match="fit on recipe content"):  # undeclared: refused as before
        initial_connector(init_spec(encoder=runtime), store)
    declared = init_spec(encoder=runtime).model_copy(update={"source_fit": _source_fit()})
    declared = InitSpec.model_validate(declared.model_dump())  # validated as a config file would be
    source_start = initial_connector(init_spec(), store)
    rebound = initial_connector(declared, store)
    assert source_start.state_dict().keys() == rebound.state_dict().keys()
    for k, v in source_start.state_dict().items():
        assert torch.equal(rebound.state_dict()[k], v), k
    ref = commit_initial_state(declared, store)
    assert EncoderRecipe.model_validate(checkpoint_encoder(store, ref)) == runtime  # its inputs: the runtime ones
    assert ref != commit_initial_state(init_spec(), store)  # another start identity, the same numbers
    assert declared.identity()["source_fit"]["recipe"] == ENCODER.identity()  # the true fit, in the identity
    assert "source_fit" not in init_spec().identity()  # the default leaves every existing identity as it was


def test_a_source_fit_covers_only_its_declared_ridges_and_its_one_option(store, monkeypatch):
    """The exception is scoped: the exact ridge digests, the source recipe the ridge records, and one added option."""
    monkeypatch.setattr(init_state, "load_ridge", lambda _s, _r: ridge())
    runtime = ENCODER.model_copy(update={"options": {**ENCODER.options, RETRY: 120}})

    def declared(encoder=runtime, **fit):
        return InitSpec(
            connector=spec(),
            receiver=receiver(),
            encoder=encoder,
            ridge={"name": "ab" * 32},
            source_fit=_source_fit(**fit),
        )

    with pytest.raises(ValueError, match="source_fit covers ridges"):  # another ridge than the one declared
        declared(ridges=("cd" * 32,))
    with pytest.raises(ValueError, match="beyond 'conformer_retry_timeout_s'"):  # anything else differs
        declared(encoder=runtime.model_copy(update={"revision": "other"}))
    with pytest.raises(ValueError, match="beyond 'conformer_retry_timeout_s'"):
        declared(encoder=runtime.model_copy(update={"options": {**runtime.options, "other": 1}}))
    with pytest.raises(ValueError, match="must add option 'other'"):  # the declared option is not the one added
        declared(option="other")
    with pytest.raises(ValueError, match="must add option"):  # no option added at all
        declared(encoder=ENCODER)
    wrong_source = ENCODER.model_copy(update={"pooling": "cls"})
    wrong = declared(
        encoder=wrong_source.model_copy(update={"options": {**ENCODER.options, RETRY: 120}}), recipe=wrong_source
    )
    with pytest.raises(ValueError, match="fit on recipe content"):  # the ridge records another source recipe
        initial_connector(wrong, store)
    other_receiver = InitSpec(
        connector=spec(),
        receiver=receiver(revision="other"),
        encoder=runtime,
        ridge={"name": "ab" * 32},
        source_fit=_source_fit(),
    )
    with pytest.raises(ValueError, match="was fit to receiver"):  # every other check stays
        initial_connector(other_receiver, store)


def test_a_run_passes_its_source_fit_to_its_step_0():
    """RunConfig.source_fit reaches the trainer's step 0 through init_spec (the propagation the trainer relies on)."""
    from latent_interconnect.training.config import RunConfig

    fields = RunConfig.model_fields
    assert "source_fit" in fields and fields["source_fit"].default is None
    import inspect

    assert "source_fit=self.source_fit" in inspect.getsource(RunConfig.init_spec)


def test_a_source_fit_may_name_residues_turned_on_and_nothing_else(store, monkeypatch):
    """The molecule tr arm: the pooled ridge covers a recipe that only turns on per-residue storage (``residues``
    names the field, not an option); the start is the source recipe's, and every other difference is still refused."""
    sol = ridge()
    monkeypatch.setattr(init_state, "load_ridge", lambda _s, _r: sol)
    assert not ENCODER.residues
    tokens = ENCODER.model_copy(update={"residues": True})

    def declared(encoder=tokens, **fit):
        return InitSpec(
            connector=spec(),
            receiver=receiver(),
            encoder=encoder,
            ridge={"name": "ab" * 32},
            source_fit=_source_fit(**{"option": "residues", **fit}),
        )

    rebound = initial_connector(InitSpec.model_validate(declared().model_dump()), store)
    source_start = initial_connector(init_spec(), store)
    for k, v in source_start.state_dict().items():
        assert torch.equal(rebound.state_dict()[k], v), k
    with pytest.raises(ValueError, match="must turn on residues"):  # the runtime recipe has them off
        declared(encoder=ENCODER)
    with pytest.raises(ValueError, match="must turn on residues"):  # the source recipe already has them on
        declared(recipe=tokens)
    with pytest.raises(ValueError, match="beyond 'residues'"):  # anything else differs
        declared(encoder=tokens.model_copy(update={"residue_dtype": "fp32"}))
