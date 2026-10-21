"""The examples of examples/guides/extending.md run and behave (extension paths 1-4, through `plugins:` alone)."""

from pathlib import Path

import numpy as np
import torch

from latent_interconnect.config import load_config
from latent_interconnect.data import STREAMS, TrainingText
from latent_interconnect.domains import DOMAINS
from latent_interconnect.domains.conformance import check_domain
from latent_interconnect.encoders.base import EXTRACTORS, RawObject
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import OUTPUT_LAYER
from latent_interconnect.plugins import load_plugins
from latent_interconnect.receivers.base import RECEIVER_BACKENDS, ReceiverSpec
from latent_interconnect.writes import MAP_KINDS, Connector, ConnectorSpec, Entry, ObjectBatch
from latent_interconnect.writes.conformance import check_kind
from latent_interconnect.writes.init import RidgeLayer, RidgeSolution
from latent_interconnect.writes.kinds import NoParams

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "examples" / "configs"


def test_core_kinds_conform():
    for name in ("constant", "affine"):
        check_kind(MAP_KINDS.get(name), NoParams(), 16, 8)


def test_path1_new_connector_kind():
    spec = load_config(CONF / "gated-connector.yaml", ConnectorSpec)
    load_plugins(spec.plugins)
    kind = MAP_KINDS.get("example-gated")
    check_kind(kind, kind.Params(hidden=8), 16, 64)
    rng = np.random.default_rng(0)
    layers = {L: RidgeLayer(rng.normal(size=(64, 16)).astype(np.float32), np.zeros(64, np.float32)) for L in range(4)}
    ridge = RidgeSolution("name", np.zeros(16, np.float32), np.ones(16, np.float32), layers)
    c = Connector.from_ridge(spec, 4, {"name": ridge})
    x = torch.ones(2, 16)
    w = c(ObjectBatch(("a", "b"), pooled=x))
    assert {e.layer for e in w.deltas} == {0, 1, 2, 3}
    gated = w.deltas[Entry(0, 2)]
    ridge_out = x @ torch.as_tensor(layers[2].W).T
    torch.testing.assert_close(gated, ridge_out * torch.sigmoid(torch.tensor(6.0)), rtol=1e-5, atol=1e-5)  # gate open


def test_path2_new_domain_and_encoder():
    load_plugins(["examples.rna_domain"])
    domain = DOMAINS.get("example-rna")()
    check_domain(domain)
    assert domain.labels()["gc_rich"].tolist() == [False, True, False]  # GC fractions 4/8, 6/8, 2/8
    recipe = load_config(CONF / "rna-kmer.yaml", EncoderRecipe)
    load_plugins(recipe.plugins)
    extractor = EXTRACTORS.get(recipe.extractor)(recipe.model_dump(), "cpu")
    extractor.load()
    out = extractor.extract([RawObject(i, v) for i, v in domain.objects()["value"].items()])
    pooled = out.pooled[OUTPUT_LAYER]
    assert pooled.shape == (3, 16) and pooled.dtype == np.float32
    np.testing.assert_allclose(pooled.sum(axis=1), 1.0, rtol=1e-6)  # frequencies
    assert pooled[0, 1] == np.float32(2 / 7)  # "ACGUACGU" has AC twice among 7 pairs


def test_path3_new_text_stream():
    load_plugins(["examples.corpus_stream", "examples.rna_domain"])
    cls = STREAMS.get("example-corpus")
    domain = DOMAINS.get("example-rna")()
    text = TrainingText(("r1", "r2"), {"r1": {"description": "one"}, "r2": {}}, {})
    params = cls.Params(question="Describe the RNA<SLOT:x>.", field="description")
    examples = cls(domain, params, text).sample(np.random.default_rng(0), 3)
    assert [e.objects for e in examples] == [{"x": "r1"}] * 3  # r2's field was screened out
    assert all(e.target == "one" and e.spans[0].end == 3 for e in examples)


def test_path4_new_language_model():
    spec = load_config(CONF / "llama-receiver.yaml", ReceiverSpec)
    assert set(spec.backends) == {"score", "train", "generate", "capture"}
    assert all(name in RECEIVER_BACKENDS for name in spec.backends.values())


def test_ask_a_question(tmp_path):
    """examples/ask.py on plain files: a local tiny Llama and a checkpoint directory, built here (no store, no pins)."""
    from examples.ask import ask
    from tests.fakes import build_tiny_llama

    from latent_interconnect.artifacts.store import ArtifactStore
    from latent_interconnect.config import dump_config
    from latent_interconnect.writes.checkpoint import commit_checkpoint

    receiver = tmp_path / "tiny-llama.yaml"
    spec = build_tiny_llama(tmp_path / "tiny-llama")
    dump_config(spec, receiver)
    connector_spec = load_config(CONF / "gated-connector.yaml", ConnectorSpec)
    load_plugins(connector_spec.plugins)
    rng = np.random.default_rng(0)
    d_in, d_model = 16, spec.d_model
    layers = {
        L: RidgeLayer(rng.normal(size=(d_model, d_in)).astype(np.float32), np.zeros(d_model, np.float32))
        for L in range(spec.depth)
    }
    ridge = RidgeSolution("name", np.zeros(d_in, np.float32), np.ones(d_in, np.float32), layers)
    store = ArtifactStore(tmp_path / "store")
    encoder = load_config(CONF / "rna-kmer.yaml", EncoderRecipe)  # the recipe the gated connector reads
    connector = Connector.from_ridge(connector_spec, spec.depth, {"name": ridge})
    checkpoint = commit_checkpoint(connector, 0, store, encoder=encoder)
    lines: list[str] = []
    answer = ask(
        receiver,
        store.path(checkpoint),
        np.ones(d_in, np.float32),
        "Is <SLOT:x> GC-rich?",
        max_new_tokens=4,
        log=lines.append,
    )
    assert isinstance(answer, str)
    assert lines == [f"connector {connector_spec.name}: kind {connector_spec.kind}, reads pooled"]
    again = ask(receiver, store.path(checkpoint), np.ones(d_in, np.float32), "Is <SLOT:x> GC-rich?", max_new_tokens=4)
    assert again == answer  # greedy by default: deterministic


def test_ask_reports_an_unusable_input_in_one_line(tmp_path):
    """examples/ask.py's command line: a checkpoint that does not load is one `error:` line and exit status 1."""
    import subprocess
    import sys

    np.save(tmp_path / "x.npy", np.ones(4, np.float32))
    receiver = CONF / "pipeline" / "qwen2.5-0.5b-instruct.yaml"
    cmd = [
        sys.executable,
        str(ROOT / "examples" / "ask.py"),
        str(receiver),
        str(tmp_path / "none"),
        str(tmp_path / "x.npy"),
        "Is <SLOT:x> GC-rich?",
    ]
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr.startswith("error: ") and done.stderr.count("\n") == 1 and "Traceback" not in done.stderr


def test_ask_names_the_installed_bundle_of_a_checkpoint(tmp_path):
    """examples/ask.py's stderr line: a checkpoint under <root>/lip/artifacts names the bundle that installed it."""
    import json

    from examples.ask import installed_bundle

    sha = "ab" + "0" * 62
    ck = tmp_path / "lip" / "artifacts" / "ab" / sha
    ck.mkdir(parents=True)
    assert installed_bundle(ck) is None  # no bundles directory
    (tmp_path / "lip" / "bundles").mkdir()
    (tmp_path / "lip" / "bundles" / "other.json").write_text(json.dumps({"name": "other", "artifacts": []}))
    bundle = {"name": "gemma-3-4b-it__toy", "artifacts": [{"sha256": sha}]}
    (tmp_path / "lip" / "bundles" / "gemma-3-4b-it__toy.json").write_text(json.dumps(bundle))
    assert installed_bundle(ck) == "gemma-3-4b-it__toy"
    assert installed_bundle(tmp_path) is None  # a directory outside a store


def _installed(root, name, kind="checkpoint"):
    """Install one artifact of ``kind`` under ``root`` as release_download does; return its directory."""
    import json

    from latent_interconnect.artifacts.store import ArtifactStore

    store = ArtifactStore(root / "lip" / "artifacts")
    staged = store.staging(name)
    (staged / "connector.json").write_text(json.dumps({"name": name}))
    ref = store.commit(staged, kind, {"name": name})
    bundle = {"name": name, "artifacts": [{"sha256": ref.sha256, "kind": kind}]}
    (root / "lip" / "bundles").mkdir(parents=True, exist_ok=True)
    (root / "lip" / "bundles" / f"{name}.json").write_text(json.dumps(bundle))
    return store.path(ref)


def test_ask_takes_an_installed_connector_by_name(tmp_path, monkeypatch):
    """examples/ask.py's CONNECTOR: an installed name is its bundle's checkpoint directory; a path stays a path."""
    from examples.ask import connector_checkpoint, installed_connectors

    ck = _installed(tmp_path, "gemma-3-4b-it__toy")
    _installed(tmp_path, "toy-data", kind="domain")  # a data bundle is kept there too, and is no connector
    assert installed_connectors(tmp_path) == ["gemma-3-4b-it__toy"]
    assert connector_checkpoint("gemma-3-4b-it__toy", tmp_path) == ck
    monkeypatch.setenv("LIP_DATA", str(tmp_path))
    assert connector_checkpoint("gemma-3-4b-it__toy") == ck  # the data root defaults to $LIP_DATA
    assert connector_checkpoint(ck) == ck  # a directory, as given
    assert connector_checkpoint(str(tmp_path / "absent")) == tmp_path / "absent"  # a path, even if absent
    monkeypatch.chdir(tmp_path)
    (tmp_path / "gemma-3-4b-it__toy").mkdir()
    assert connector_checkpoint("gemma-3-4b-it__toy") == Path("gemma-3-4b-it__toy")  # an existing path wins


def test_ask_names_the_installed_connectors_for_an_unknown_name(tmp_path):
    """examples/ask.py's command line: a name not installed is one `error:` line naming the installed connectors."""
    import os
    import subprocess
    import sys

    _installed(tmp_path, "gemma-3-4b-it__toy")
    np.save(tmp_path / "x.npy", np.ones(4, np.float32))
    receiver = CONF / "pipeline" / "qwen2.5-0.5b-instruct.yaml"
    cmd = [sys.executable, str(ROOT / "examples" / "ask.py"), str(receiver), "gemma-3-4b-it__nope"]
    cmd += [str(tmp_path / "x.npy"), "Is <SLOT:x> GC-rich?"]
    env = os.environ | {"LIP_DATA": str(tmp_path)}
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, env=env, check=False)
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr.startswith("error: ") and done.stderr.count("\n") == 1, done.stderr
    assert "(installed: gemma-3-4b-it__toy)" in done.stderr
    assert "release_download --connector gemma-3-4b-it__nope --dest $LIP_DATA" in done.stderr
