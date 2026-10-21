"""The release downloader: a local directory stands in for the hub (``--local-source``, a test path)."""

import json
import re
import shlex
import shutil
from pathlib import Path

import pytest

from latent_interconnect import release_download
from latent_interconnect.artifacts.store import MANIFEST_NAME, ArtifactStore, IntegrityError, sha256_file
from latent_interconnect.release_download import BUNDLE_FILE, download, local_fetch, main
from latent_interconnect.workspace import Workspace


def _release(tmp_path: Path) -> tuple[Path, str]:
    """A release layout with one connector bundle holding one committed artifact; return its root and the id."""
    store = ArtifactStore(tmp_path / "src-store")
    staged = store.staging("x")
    (staged / "weights.bin").write_bytes(b"\x00\x01" * 1000)
    (staged / "sub").mkdir()
    (staged / "sub" / "a.json").write_text('{"a": 1}')
    ref = store.commit(staged, "checkpoint", {"connector": "toy", "step": 3})
    root = tmp_path / "release"
    bundle_dir = root / "connectors" / "toy"
    shutil.copytree(store.path(ref), bundle_dir)
    manifest = json.loads((bundle_dir / MANIFEST_NAME).read_text())
    files = {rel: {"sha256": sha, "bytes": size} for rel, (sha, size) in manifest["files"].items()}
    bundle = {
        "name": "toy",
        "type": "connector",
        "artifacts": [{"sha256": ref.sha256, "kind": "checkpoint", "path": ".", "files": files}],
        "pins": {"toy/step_000003": {"target": "artifact", "sha256": ref.sha256}},
        "license": "CC BY 4.0 (toy bundle)",
    }
    (bundle_dir / BUNDLE_FILE).write_text(json.dumps(bundle))
    index = {"connectors": [{"name": "toy", "path": "toy", "bundle_sha256": sha256_file(bundle_dir / BUNDLE_FILE)}]}
    (root / "connectors" / "connectors.json").write_text(json.dumps(index))
    (root / "data").mkdir()
    (root / "data" / "data.json").write_text(json.dumps({"data": []}))
    return root, ref.sha256


def _ws(dest: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    monkeypatch.setenv("LIP_DATA", str(dest))
    for var in ("LIP_ARTIFACTS", "LIP_RECORDS", "LIP_RUNS", "LIP_SCRATCH"):
        monkeypatch.delenv(var, raising=False)
    return Workspace.from_env()


def test_download_installs_verifies_and_pins(tmp_path, monkeypatch):
    root, sha = _release(tmp_path)
    dest = tmp_path / "data"
    assert main(["--local-source", str(root), "--connector", "toy", "--dest", str(dest)]) == 0
    ws = _ws(dest, monkeypatch)
    ref = ws.artifacts.ref(sha)
    ws.artifacts.verify(ref)
    assert ws.pins()["toy/step_000003"].sha256 == sha
    assert json.loads((dest / "lip" / "bundles" / "toy.json").read_text())["license"] == "CC BY 4.0 (toy bundle)"
    assert download(local_fetch(root / "connectors"), dest, ["toy"]) == {
        "toy/step_000003": ws.pins()["toy/step_000003"]
    }  # again


def test_a_changed_file_is_refused_before_anything_is_pinned(tmp_path, monkeypatch):
    root, sha = _release(tmp_path)
    weights = root / "connectors" / "toy" / "weights.bin"
    weights.chmod(0o644)  # committed files are read only
    weights.write_bytes(b"\x00\x02" * 1000)  # same size, other bytes
    dest = tmp_path / "data"
    with pytest.raises(IntegrityError, match=r"weights\.bin"):
        download(local_fetch(root / "connectors"), dest, ["toy"])
    ws = _ws(dest, monkeypatch)
    assert not ws.artifacts.exists(sha)
    assert ws.pins() == {}


def test_a_changed_bundle_file_is_refused_by_the_index(tmp_path):
    root, _ = _release(tmp_path)
    path = root / "connectors" / "toy" / BUNDLE_FILE
    path.chmod(0o644)
    path.write_text(path.read_text().replace("toy bundle", "changed"))
    with pytest.raises(IntegrityError, match=re.escape(BUNDLE_FILE)):
        download(local_fetch(root / "connectors"), tmp_path / "data", ["toy"])


def test_an_unknown_bundle_names_the_listed_ones(tmp_path):
    root, _ = _release(tmp_path)
    with pytest.raises(KeyError, match="toy"):
        download(local_fetch(root / "connectors"), tmp_path / "data", ["nope"])


def test_the_hub_path_fetches_each_file_by_its_release_path(tmp_path, monkeypatch):
    import huggingface_hub

    root, sha = _release(tmp_path)
    asked = []

    def fake(repo, filename, revision, repo_type):
        asked.append((repo, filename, revision, repo_type))
        return str(root / ("connectors" if repo_type == "model" else "data") / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    dest = tmp_path / "data"
    assert (
        main(
            [
                "--repo",
                "org/aimm",
                "--revision",
                "0123456789abcdef0123456789abcdef01234567",
                "--connector",
                "toy",
                "--dest",
                str(dest),
            ]
        )
        == 0
    )
    assert {(a[0], a[2], a[3]) for a in asked} == {("org/aimm", "0123456789abcdef0123456789abcdef01234567", "model")}
    assert {"connectors.json", "toy/bundle.json", "toy/weights.bin"} <= {a[1] for a in asked}
    assert _ws(dest, monkeypatch).artifacts.exists(sha)


COMMIT = "0123456789abcdef0123456789abcdef01234567"
GETTING_STARTED = Path(__file__).resolve().parents[2] / "examples" / "README.md"


def _record_hub(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    """Replace the hub and the install with recorders; return the (repo, revision, type) each fetcher was made for."""
    made: list[tuple[str, str, str]] = []

    def hub(repo, revision, repo_type="model"):
        made.append((repo, revision, repo_type))
        return Path

    monkeypatch.setattr(release_download, "hub_fetch", hub)
    monkeypatch.setattr(release_download, "_download", lambda *a, **k: ({}, 0))
    monkeypatch.setattr(release_download, "list_bundles", lambda *a, **k: [])
    monkeypatch.setattr(release_download, "all_data_names", lambda *a, **k: [])  # --all-data reads the data index
    monkeypatch.setattr(release_download, "_plan", lambda *a, **k: [])  # --dry-run reads both indexes
    return made


@pytest.mark.parametrize("flags", [["--revision", "main"], ["--revision", COMMIT[:7]]])
def test_another_repo_at_a_branch_or_short_sha_is_refused(flags, tmp_path, monkeypatch, capsys):
    made = _record_hub(monkeypatch)
    with pytest.raises(SystemExit):
        main(["--repo", "someone/theirs", *flags, "--connector", "c", "--dest", str(tmp_path)])
    err = capsys.readouterr().err
    assert f"--revision {flags[1]!r} is not a commit" in err
    assert "40-hex commit sha" in err
    assert made == []


def test_another_data_repo_needs_a_full_commit_only_when_data_is_asked(tmp_path, monkeypatch, capsys):
    made = _record_hub(monkeypatch)
    data = ["--data-repo", "someone/data", "--data", "d", "--dest", str(tmp_path)]
    with pytest.raises(SystemExit):
        main([*data, "--data-revision", "main"])
    assert "--data-revision 'main' is not a commit" in capsys.readouterr().err
    assert main([*data, "--data-revision", COMMIT]) == 0
    assert ("someone/data", COMMIT, "dataset") in made
    made.clear()
    assert main(["--data-repo", "someone/data", "--connector", "c", "--dest", str(tmp_path)]) == 0  # no --data
    assert made == [(release_download.CONNECTORS_REPO, release_download.CONNECTORS_REVISION, "model")]


def test_the_getting_started_download_commands_run_as_written(tmp_path, monkeypatch):
    blocks = re.findall(r"```bash\n(.*?)```", GETTING_STARTED.read_text(), flags=re.S)
    commands = [
        line
        for block in blocks
        for line in block.replace("\\\n", " ").splitlines()
        if line.startswith("python -m latent_interconnect.release_download")
        and "$(" not in line  # the all-bundles form builds its flags in a shell substitution that reads the hub
    ]
    assert len(commands) >= 2
    for command in commands:
        made = _record_hub(monkeypatch)
        argv = shlex.split(command.replace("$LIP_DATA", str(tmp_path)), comments=True)[3:]  # a trailing # comment
        assert main(argv) == 0, command
        assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for _, revision, _ in made), made


def test_the_command_reports_how_many_files_it_verified(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root, _ = _release(tmp_path)
    assert main(["--local-source", str(root), "--connector", "toy", "--dest", str(tmp_path / "data")]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "verified 2 installed files (sha256)" in out  # weights.bin and sub/a.json
    assert out[-1].startswith("export LIP_DATA=")


def test_the_command_prints_each_connectors_full_checkpoint_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, sha = _release(tmp_path)
    dest = tmp_path / "data"
    assert main(["--local-source", str(root), "--connector", "toy", "--dest", str(dest)]) == 0
    out = capsys.readouterr().out.splitlines()
    prefix = "  checkpoint directory: "  # under the "connector toy: ..." line
    [line] = [x for x in out if x.startswith(prefix)]
    assert out[out.index(line) - 1].startswith("connector toy: ")
    path = Path(line.removeprefix(prefix))
    assert path == (dest / "lip" / "artifacts" / sha[:2] / sha).resolve()  # the full id, not a 12-hex prefix
    assert (path / "weights.bin").is_file()
    license_line = "  license (this bundle): CC BY 4.0 (toy bundle)"  # the bundle's own string, then the one code line
    assert out[out.index(license_line) + 1] == (
        "  The project code is Apache-2.0; check this bundle's license and its upstream terms before use."
    )


def _add_data(root: Path, tmp_path: Path, name: str, group: str | None) -> str:
    """Add one data bundle of one committed artifact to the release's data index; return the artifact id."""
    store = ArtifactStore(tmp_path / f"src-{name}")
    staged = store.staging(name)
    (staged / "t.json").write_text(json.dumps({"name": name}))
    ref = store.commit(staged, "domain", {"name": name})
    bundle_dir = root / "data" / name
    shutil.copytree(store.path(ref), bundle_dir)
    manifest = json.loads((bundle_dir / MANIFEST_NAME).read_text())
    files = {rel: {"sha256": h, "bytes": n} for rel, (h, n) in manifest["files"].items()}
    bundle = {
        "name": name,
        "type": "data",
        "artifacts": [{"sha256": ref.sha256, "kind": "domain", "path": ".", "files": files}],
        "pins": {name: {"target": "artifact", "sha256": ref.sha256}},
        "license": "CC BY 4.0 (toy bundle)",
    }
    (bundle_dir / BUNDLE_FILE).write_text(json.dumps(bundle))
    index_path = root / "data" / "data.json"
    index = json.loads(index_path.read_text())
    entry = {"name": name, "path": name, "bundle_sha256": sha256_file(bundle_dir / BUNDLE_FILE), "bytes": 123}
    index["data"].append(entry | ({"group": group} if group else {}))
    index_path.write_text(json.dumps(index))
    return ref.sha256


def test_all_data_installs_every_data_bundle_but_the_demo_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _ = _release(tmp_path)
    full = _add_data(root, tmp_path, "protein-domain", "domains")
    mini = _add_data(root, tmp_path, "protein-esmc-600m-mini", None)
    demo = _add_data(root, tmp_path, "protein-residues-mini-demo", "demo")
    other_demo = _add_data(root, tmp_path, "named-only-demo", None)
    dest = tmp_path / "data"
    assert main(["--local-source", str(root), "--all-data", "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert [x.split()[3] for x in out[:-1]] == ["protein-domain", "protein-esmc-600m-mini"]
    assert out[-1] == "total 2 bundles, 0.00 GB (nothing downloaded: --dry-run)"
    assert not dest.exists()  # a dry run installs nothing
    assert main(["--local-source", str(root), "--all-data", "--dest", str(dest)]) == 0
    ws = _ws(dest, monkeypatch)
    assert set(ws.pins()) == {"protein-domain", "protein-esmc-600m-mini"}
    assert ws.artifacts.exists(full) and ws.artifacts.exists(mini)
    assert not ws.artifacts.exists(demo) and not ws.artifacts.exists(other_demo)


def test_a_failed_check_prints_one_line_naming_the_file_and_the_cache_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import huggingface_hub
    import huggingface_hub.constants

    root, _ = _release(tmp_path)
    weights = root / "connectors" / "toy" / "weights.bin"
    weights.chmod(0o644)
    weights.write_bytes(b"\x00\x02" * 1000)

    def fake(repo: str, filename: str, revision: str, repo_type: str) -> str:
        return str(root / "connectors" / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hf" / "hub"))
    argv = ["--repo", "org/aimm", "--revision", COMMIT, "--connector", "toy", "--dest", str(tmp_path / "data")]
    assert main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    assert "connector bundle toy" in lines[0]
    assert "weights.bin" in lines[0]
    assert str(tmp_path / "hf" / "hub" / "models--org--aimm") in lines[0]
    assert "rerun" in lines[0]


def test_list_prints_each_bundle_with_its_size_and_installs_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _ = _release(tmp_path)
    index = root / "connectors" / "connectors.json"
    entries = json.loads(index.read_text())
    entries["connectors"][0] |= {"bytes": 8055912051, "description": "a toy"}
    index.write_text(json.dumps(entries))
    (root / "data" / "data.json").write_text(json.dumps({"data": [{"name": "mini", "path": "mini", "bytes": 173225}]}))
    assert main(["--local-source", str(root), "--list"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2
    assert out[0].split() == ["connector", "toy", "8.06", "GB", "a", "toy"]
    assert out[1].split() == ["data", "mini", "0.17", "MB"]  # below 0.1 GB, in MB
    assert not (tmp_path / "data").exists()


def test_list_kinds_reads_each_connector_bundle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--list --kinds: kind, input, tokens and DEMO from each bundle.json (a fake hub); the index stays as it is."""
    root, _ = _release(tmp_path)
    bundle_file = root / "connectors" / "toy" / BUNDLE_FILE
    bundle = json.loads(bundle_file.read_text()) | {
        "demo": True,
        "write": {"kind": "xattn", "tokens": [{}, {}, {}], "layers": {"per_token": [[1], [2], [3]]}},
    }
    bundle_file.write_text(json.dumps(bundle))
    index_file = root / "connectors" / "connectors.json"
    index = json.loads(index_file.read_text())
    index["connectors"][0]["bundle_sha256"] = sha256_file(bundle_file)
    index["connectors"].append(dict(index["connectors"][0], name="toy2", path="toy2"))
    (root / "connectors" / "toy2").mkdir()
    layout = {"kind": "mlp", "input": "pooled", "tokens": 1, "kind_params": {}, "write_layers": [[1]]}
    (root / "connectors" / "toy2" / BUNDLE_FILE).write_text(
        json.dumps(bundle | {"demo": False, "connector_kind": layout})
    )
    index["connectors"][1]["bundle_sha256"] = sha256_file(root / "connectors" / "toy2" / BUNDLE_FILE)
    index_file.write_text(json.dumps(index))
    before = index_file.read_bytes()
    assert main(["--local-source", str(root), "--list", "--kinds"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split()[:2] == ["connector", "toy"] and out[0].endswith("[xattn, residues, 3 token(s), DEMO]")
    assert out[1].endswith("[mlp, pooled, 1 token(s)]")  # connector_kind, when the bundle has it
    assert index_file.read_bytes() == before
    assert main(["--local-source", str(root), "--list"]) == 0
    assert "[" not in capsys.readouterr().out  # without --kinds, the index alone
    with pytest.raises(SystemExit):
        main(["--local-source", str(root), "--kinds", "--connector", "toy", "--dest", str(tmp_path / "d")])
