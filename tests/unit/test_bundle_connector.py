"""Contributor packaging (``bundle_connector``): a tiny checkpoint packaged, installed through the downloader, loaded.

The checkpoint is the e2e chain's protein ``affine`` connector for its tiny Llama receiver (``tests/e2e``'s configs,
written by its :func:`write_chain`), built from a random ridge fit instead of trained, and committed as a stage does.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from tests.e2e.test_tiny_pipeline import PROTEIN_CHAIN, ROOT, write_chain

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore, Manifest, sha256_file
from latent_interconnect.bundle_connector import BundleError, main, package
from latent_interconnect.config import load_config
from latent_interconnect.config_index import cited_pins, stand_in_pins
from latent_interconnect.release_download import local_fetch
from latent_interconnect.release_download import main as download_main
from latent_interconnect.training import RunConfig
from latent_interconnect.workspace import Workspace
from latent_interconnect.writes import Connector, ObjectBatch
from latent_interconnect.writes.checkpoint import commit_checkpoint, load_connector
from latent_interconnect.writes.init import RidgeLayer, RidgeSolution

D_IN = 6
STEP = 50


def _commit(tmp_path: Path, kind: str = "affine") -> tuple[Path, Path, ArtifactRef, Connector]:
    """Write the e2e chain's configs in ``tmp_path`` and commit one ``kind`` checkpoint of its tiny receiver."""
    receiver = tmp_path / "receiver.yaml"
    receiver.write_text((ROOT / "configs" / "receivers" / "tiny-llama.yaml").read_text())
    write_chain(tmp_path, receiver, PROTEIN_CHAIN)
    config = tmp_path / f"train_{kind}.yaml"
    run = load_config(config, RunConfig, stand_in_pins(cited_pins(config)))
    g = np.random.default_rng(0)
    d_model = run.receiver.d_model
    layers = {
        L: RidgeLayer(W=g.normal(size=(d_model, D_IN)).astype(np.float32), b=g.normal(size=d_model).astype(np.float32))
        for L in (0, 1, 2)
    }
    fit = RidgeSolution("name", g.normal(size=D_IN).astype(np.float32), np.ones(D_IN, np.float32), layers)
    connector = Connector.from_ridge(run.connector, run.receiver.depth, {"name": fit})
    store = ArtifactStore(tmp_path / "contributor-store")
    ref = commit_checkpoint(connector, STEP, store, encoder=run.encoder)
    return tmp_path, store.root, ref, connector


@pytest.fixture
def trained(tmp_path: Path) -> tuple[Path, Path, ArtifactRef, Connector]:
    """One affine checkpoint of the e2e chain's tiny receiver (:func:`_commit`)."""
    return _commit(tmp_path)


def _ws(dest: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    monkeypatch.setenv("LIP_DATA", str(dest))
    for var in ("LIP_ARTIFACTS", "LIP_RECORDS", "LIP_RUNS", "LIP_SCRATCH"):
        monkeypatch.delenv(var, raising=False)
    return Workspace.from_env()


def test_a_packaged_checkpoint_installs_and_loads(trained, tmp_path, monkeypatch):
    configs, store_root, ref, connector = trained
    store = ArtifactStore(store_root)
    repo = tmp_path / "community"
    argv = ["--checkpoint", str(store.path(ref)), "--run-config", str(configs / "train_affine.yaml")]
    argv += ["--out", str(repo / "connectors"), "--license", "MIT", "--card-url", "https://example.org/card"]
    argv += ["--install-pin", f"contrib/toy-affine/step_{STEP:06d}"]
    assert main(argv) == 0
    assert main(argv) == 0  # again: the same checkpoint under the same name is a no-op
    bundle = json.loads((repo / "connectors" / "tiny-llama__toy-linear" / "bundle.json").read_text())
    assert bundle["format"] == "aimm-bundle/1"
    assert bundle["receiver"] == {"name": "tiny-llama", "model_id": "lip-test/tiny-llama", "revision": "random-seed-0"}
    assert bundle["specialist"] == {"name": "fake-toy", "model_id": "none", "revision": "v1", "domain": "fake-toy"}
    assert bundle["checkpoint_manifest_sha256"] == ref.sha256
    assert bundle["connector_sha"] == connector.spec.digest()
    assert (bundle["license"], bundle["card_url"]) == ("MIT", "https://example.org/card")
    pin = f"contrib/toy-affine/step_{STEP:06d}"
    assert bundle["pins"] == {pin: {"target": "artifact", "sha256": ref.sha256}}
    assert set(bundle["artifacts"][0]["files"]) == {"connector.json", "weights.safetensors"}
    index = json.loads((repo / "connectors" / "connectors.json").read_text())
    assert [e["name"] for e in index["connectors"]] == ["tiny-llama__toy-linear"]

    dest = tmp_path / "user-data"
    install = ["--local-source", str(repo), "--connector", "tiny-llama__toy-linear", "--dest", str(dest)]
    assert download_main(install) == 0
    ws = _ws(dest, monkeypatch)
    installed = ws.artifacts.ref(ws.pins()[pin].sha256)
    assert installed.sha256 == ref.sha256
    loaded, step = load_connector(ws.artifacts, installed)
    assert step == STEP
    assert loaded.spec.digest() == connector.spec.digest()
    pooled = torch.randn(3, D_IN, generator=torch.Generator().manual_seed(1))
    want = connector(ObjectBatch(("a", "b", "c"), pooled=pooled))
    got = loaded(ObjectBatch(("a", "b", "c"), pooled=pooled))
    assert set(got.deltas) == set(want.deltas)
    for entry, delta in want.deltas.items():
        assert torch.equal(got.deltas[entry], delta)


def test_a_checkpoint_of_another_connector_is_refused_and_nothing_written(trained, tmp_path):
    configs, store_root, ref, _ = trained
    out = tmp_path / "community" / "connectors"
    with pytest.raises(BundleError, match="connector differ"):
        package(ArtifactStore(store_root).path(ref), configs / "train_mlp.yaml", out, license="MIT", pin="p")
    assert not out.exists()


def test_a_corrupt_or_missing_checkpoint_file_writes_nothing_and_a_good_retry_succeeds(trained, tmp_path):
    configs, store_root, ref, _ = trained
    src = ArtifactStore(store_root).path(ref)
    config = configs / "train_affine.yaml"
    out = tmp_path / "community" / "connectors"
    corrupt = tmp_path / "corrupt"
    shutil.copytree(src, corrupt)
    weights = corrupt / "weights.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF  # same size, other bytes
    weights.chmod(0o644)
    weights.write_bytes(bytes(data))
    with pytest.raises(BundleError, match=r"weights\.safetensors differs"):
        package(corrupt, config, out, license="MIT", pin="contrib/x")
    assert not out.exists()
    weights.unlink()
    with pytest.raises(BundleError, match=r"weights\.safetensors is missing"):
        package(corrupt, config, out, license="MIT", pin="contrib/x")
    assert not out.exists()
    bundle_dir = package(src, config, out, license="MIT", pin="contrib/x")  # the retry with the good checkpoint
    assert json.loads((bundle_dir / "bundle.json").read_text())["checkpoint_manifest_sha256"] == ref.sha256


def test_missing_metadata_is_named(trained, tmp_path, capsys):
    configs, store_root, ref, _ = trained
    src = ArtifactStore(store_root).path(ref)
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "MANIFEST.json").write_text((src / "MANIFEST.json").read_text())  # no connector.json
    config = str(configs / "train_affine.yaml")
    out = tmp_path / "out"
    assert main(["--checkpoint", str(bare), "--run-config", config, "--out", str(out), "--license", "MIT"]) == 2
    assert "has no connector.json" in capsys.readouterr().err
    with pytest.raises(BundleError, match="license"):
        package(src, configs / "train_affine.yaml", out, license=" ")
    with pytest.raises(BundleError, match="names no destination"):  # the e2e run configs name none (--pin at run)
        package(src, configs / "train_affine.yaml", out, license="MIT")
    assert Manifest.load(src / "MANIFEST.json").sha256() == ref.sha256
    assert not out.exists()


def test_a_community_repo_needs_its_revision_and_is_read_at_it(trained, tmp_path, monkeypatch):
    import huggingface_hub

    configs, store_root, ref, _ = trained
    repo = tmp_path / "community"
    src = ArtifactStore(store_root).path(ref)
    package(src, configs / "train_affine.yaml", repo / "connectors", license="MIT", pin="contrib/x")
    base = ["--repo", "someone/their-connectors", "--connector", "tiny-llama__toy-linear"]
    base += ["--dest", str(tmp_path / "d")]
    with pytest.raises(SystemExit):
        download_main(base)  # no --revision: the release commit is not theirs
    asked = []

    def fake(repo_id, filename, revision, repo_type):
        asked.append((repo_id, revision, repo_type))
        return str(local_fetch(repo / "connectors")(filename))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    assert download_main([*base, "--revision", "0123456789abcdef0123456789abcdef01234567"]) == 0
    assert set(asked) == {("someone/their-connectors", "0123456789abcdef0123456789abcdef01234567", "model")}
    assert _ws(tmp_path / "d", monkeypatch).artifacts.exists(ref.sha256)


@pytest.mark.parametrize(
    ("kind", "reads"), [("affine", "pooled"), ("mlp", "pooled"), ("xattn", "residues"), ("xblock", "residues")]
)
def test_a_bundle_states_its_kind_and_receiver_and_any_kind_loads_for_ask(kind, reads, tmp_path, monkeypatch, capsys):
    """bundle.json carries the kind, layout, plugins and receiver config; the downloader prints them and installs the
    receiver config; ``examples/ask.py``'s input path serves every kind from one residue file, with no kind flag."""
    from examples.ask import object_batch

    from latent_interconnect.receivers.base import ReceiverSpec
    from latent_interconnect.writes.checkpoint import read_checkpoint

    configs, store_root, ref, connector = _commit(tmp_path, kind)
    store = ArtifactStore(store_root)
    repo = tmp_path / "community"
    argv = ["--checkpoint", str(store.path(ref)), "--run-config", str(configs / f"train_{kind}.yaml")]
    assert main([*argv, "--out", str(repo / "connectors"), "--license", "MIT", "--install-pin", "c/x"]) == 0
    bundle = json.loads((repo / "connectors" / "tiny-llama__toy-linear" / "bundle.json").read_text())
    layout = bundle["connector_kind"]
    assert (layout["kind"], layout["input"], layout["tokens"]) == (kind, reads, connector.spec.k)
    assert layout["kind_params"] == connector.spec.kind_params
    assert layout["write_layers"] == [sorted(t) for t in connector.spec.layers.per_token]
    assert bundle["plugins"] == []
    run = load_config(
        configs / f"train_{kind}.yaml", RunConfig, stand_in_pins(cited_pins(configs / f"train_{kind}.yaml"))
    )
    assert ReceiverSpec.model_validate(bundle["receiver_config"]) == run.receiver

    dest = tmp_path / "user-data"
    capsys.readouterr()
    assert (
        download_main(["--local-source", str(repo), "--connector", "tiny-llama__toy-linear", "--dest", str(dest)]) == 0
    )
    out = capsys.readouterr().out
    assert f"kind {kind} " in out and f"reads {reads}" in out and "plugins to have importable: none" in out
    installed = dest / "lip" / "receivers" / "tiny-llama.yaml"
    assert f"receiver config: {installed}" in out
    assert load_config(installed, ReceiverSpec) == run.receiver

    ws = _ws(dest, monkeypatch)
    ck_dir = ws.artifacts.path(ws.artifacts.ref(ws.pins()["c/x"].sha256))
    assert f"checkpoint directory: {ck_dir}\n" in out
    loaded, _ = read_checkpoint(ws.artifacts.path(ws.artifacts.ref(ws.pins()["c/x"].sha256)))
    residues = np.random.default_rng(2).normal(size=(5, D_IN)).astype(np.float32)
    got = loaded(object_batch(residues, loaded.input_kinds))
    assert got.deltas and all(torch.isfinite(d).all() for d in got.deltas.values())
    if reads == "residues":
        with pytest.raises(ValueError, match=r"residues\.npy"):
            object_batch(residues.mean(0), loaded.input_kinds)
        with pytest.raises(ValueError, match=r"give emb/P1\.residues\.npy, which examples/embed\.py writes"):
            object_batch(residues.mean(0), loaded.input_kinds, source="emb/P1.npy")
    else:
        pooled = loaded(object_batch(residues.mean(0), loaded.input_kinds))
        for entry, delta in got.deltas.items():
            torch.testing.assert_close(pooled.deltas[entry], delta)


def test_a_bundle_without_the_new_fields_is_summarized_from_its_connector_json(trained, tmp_path, capsys):
    """The released bundles predate ``connector_kind``, ``plugins`` and ``receiver_config``: still installed, and
    described from the stored checkpoint."""
    configs, store_root, ref, _ = trained
    store = ArtifactStore(store_root)
    repo = tmp_path / "community" / "connectors"
    argv = ["--checkpoint", str(store.path(ref)), "--run-config", str(configs / "train_affine.yaml")]
    assert main([*argv, "--out", str(repo), "--license", "MIT", "--install-pin", "c/x"]) == 0
    path = repo / "tiny-llama__toy-linear" / "bundle.json"
    bundle = json.loads(path.read_text())
    for field in ("connector_kind", "plugins", "receiver_config"):
        del bundle[field]
    path.write_text(json.dumps(bundle))
    index = json.loads((repo / "connectors.json").read_text())
    index["connectors"][0]["bundle_sha256"] = sha256_file(path)
    (repo / "connectors.json").write_text(json.dumps(index))
    capsys.readouterr()
    dest = tmp_path / "user-data"
    assert (
        download_main(
            ["--local-source", str(repo.parent), "--connector", "tiny-llama__toy-linear", "--dest", str(dest)]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "kind affine {}, reads pooled" in out and "plugins to have importable: none" in out
    assert "receiver config: not in the bundle; the release's is configs/receivers/tiny-llama.yaml" in out
    assert not (dest / "lip" / "receivers").exists()


def test_a_run_whose_source_fit_lists_its_ridge_by_digest_packages_as_the_pin_spelling(tmp_path):
    """A residue run started from a pooled ridge declares ``source_fit``; its ridge listed by digest packages without a
    workaround, to the same bundle as the ``$pin`` spelling, whose run identity is the same; unequal lists stay
    refused."""
    from tests.fakes import toy_recipe

    from latent_interconnect.config import Pin, dump_config, load_yaml

    configs, store_root, ref, _ = _commit(tmp_path, "xattn")
    dump_config(toy_recipe(residues=False), configs / "source.yaml")
    base = (configs / "train_xattn.yaml").read_text()
    ridge_pin = load_yaml(base)["ridge"]["name"]["$pin"]
    digest = "ab" * 32
    spellings = {"literal": f"[{digest}]", "pin": f"[{{$pin: {ridge_pin}}}]", "unequal": f"[{digest}, {'cd' * 32}]"}
    for tag, ridges in spellings.items():
        fit = f"source_fit:\n  recipe: {{$ref: source.yaml}}\n  option: residues\n  ridges: {ridges}\n"
        (configs / f"train_fit_{tag}.yaml").write_text(base + fit)
    src = ArtifactStore(store_root).path(ref)
    bundles = {
        tag: package(src, configs / f"train_fit_{tag}.yaml", tmp_path / tag, license="MIT", pin="c/x")
        for tag in ("literal", "pin")
    }
    assert (bundles["literal"] / "bundle.json").read_bytes() == (bundles["pin"] / "bundle.json").read_bytes()
    real = stand_in_pins(cited_pins(configs / "train_fit_pin.yaml")) | {
        ridge_pin: Pin(target="artifact", sha256=digest)
    }
    runs = {tag: load_config(configs / f"train_fit_{tag}.yaml", RunConfig, real) for tag in ("literal", "pin")}
    assert runs["literal"].digest() == runs["pin"].digest()
    with pytest.raises(ValueError, match="source_fit covers ridges"):
        package(src, configs / "train_fit_unequal.yaml", tmp_path / "unequal", license="MIT", pin="c/x")
    assert not (tmp_path / "unequal").exists()


def test_a_note_and_the_demo_mark_are_recorded_only_when_given(trained, tmp_path):
    configs, store_root, ref, _ = trained
    argv = ["--checkpoint", str(ArtifactStore(store_root).path(ref))]
    argv += ["--run-config", str(configs / "train_affine.yaml"), "--license", "MIT", "--install-pin", "c/x"]
    assert main([*argv, "--out", str(tmp_path / "plain")]) == 0
    assert main([*argv, "--out", str(tmp_path / "demo"), "--note", "trained briefly for testing", "--demo"]) == 0
    plain, demo = (
        json.loads((tmp_path / out / "tiny-llama__toy-linear" / "bundle.json").read_text()) for out in ("plain", "demo")
    )
    assert "note" not in plain and "demo" not in plain
    assert (demo["note"], demo["demo"]) == ("trained briefly for testing", True)
    assert {k: v for k, v in demo.items() if k not in ("note", "demo")} == plain


def test_a_pinned_checkpoint_records_its_pinning_record_and_run_config_is_the_primary_name(
    trained, tmp_path, monkeypatch
):
    """--pin finds the record that pinned the checkpoint: source.pinning_record, run_record_key, research_commit."""
    from latent_interconnect.artifacts.records import Record, RecordRef
    from latent_interconnect.config import Pin
    from latent_interconnect.provenance import Provenance

    configs, store_root, ref, _ = trained
    ws = _ws(tmp_path / "data", monkeypatch)
    src = ArtifactStore(store_root).path(ref)
    shutil.copytree(src, ws.artifacts.path(ref))
    prov = Provenance(
        version="0",
        commit="c" * 40,
        dirty=False,
        argv=("train",),
        host="h",
        job_id=None,
        python="3.11",
        packages={},
        devices=(),
        started="2026-10-19T00:00:00+00:00",
        ended="2026-10-19T00:00:01+00:00",
    )
    record = Record(
        kind="train",
        purpose="result",
        config_sha="a" * 64,
        config={},
        provenance=prov,
        outputs={"step_000050": RecordRef(sha256=ref.sha256, kind=ref.kind)},
    )
    record_id = ws.records.put(record)
    pin = f"alice/toy/train/step_{STEP:06d}"
    ws.pin_all({pin: Pin(target="artifact", sha256=ref.sha256)}, record=record_id)
    out = tmp_path / "hub" / "connectors"
    argv = ["--pin", pin, "--run-config", str(configs / "train_affine.yaml"), "--out", str(out), "--license", "MIT"]
    assert main(argv) == 0
    bundle = json.loads((out / "tiny-llama__toy-linear" / "bundle.json").read_text())
    assert bundle["source"]["pinning_record"] == record_id == record.record_id()
    assert bundle["source"]["run_record_key"] == record.computation_key()
    assert bundle["source"]["research_commit"] == "c" * 40
    assert bundle["pins"] == {pin: {"target": "artifact", "sha256": ref.sha256}}
    out2 = tmp_path / "hub2" / "connectors"  # --source-commit wins
    argv2 = ["--pin", pin, "--run-config", str(configs / "train_affine.yaml"), "--out", str(out2)]
    assert main([*argv2, "--license", "MIT", "--source-commit", "d" * 40]) == 0
    bundle2 = json.loads((out2 / "tiny-llama__toy-linear" / "bundle.json").read_text())
    assert bundle2["source"]["research_commit"] == "d" * 40 and bundle2["source"]["pinning_record"] == record_id


def test_a_checkpoint_given_by_directory_has_no_record_fields(trained, tmp_path):
    configs, store_root, ref, _ = trained
    out = tmp_path / "hub" / "connectors"
    argv = [
        "--checkpoint",
        str(ArtifactStore(store_root).path(ref)),
        "--run-config",
        str(configs / "train_affine.yaml"),
    ]
    assert main([*argv, "--out", str(out), "--license", "MIT", "--install-pin", "alice/x/step_000050"]) == 0
    source = json.loads((out / "tiny-llama__toy-linear" / "bundle.json").read_text())["source"]
    assert source["pinning_record"] is None and source["run_record_key"] is None and source["research_commit"] is None


def test_the_run_config_flag_has_no_other_spelling(tmp_path, capsys):
    argv = [
        "--checkpoint",
        str(tmp_path),
        "--connector-config",
        "x.yaml",
        "--out",
        str(tmp_path / "o"),
        "--license",
        "MIT",
    ]
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2 and "required: --run-config" in capsys.readouterr().err
