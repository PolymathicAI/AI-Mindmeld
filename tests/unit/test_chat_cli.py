"""``aimm-chat`` (``chat.cli``) on CPU.

Gemma and generic: connector selection by installed pins, the narrowed config, ``check``, and a served deployment
(the tiny receiver of ``test_chat``) that answers, then stops with its encoder worker on SIGTERM or SIGINT.

Kimi K2.6 (``kimi-vllm``): SOFTWARE EVIDENCE ONLY, never a full-model execution. vLLM, the GPUs and the weights are
replaced by fakes; the tests show the public selection (``--model kimi-k2.6``), the forwarding of the context, the
options and the engine's environment variables to the server, and the refusals before any load (vLLM environment
missing, not exactly eight whole GPUs, weights missing).
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import types
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
from tests.unit.test_chat import deployment  # noqa: F401  (the module fixture: workspace, chat config, receiver)
from tests.unit.tiny_receiver import ROOT

import latent_interconnect.receivers
from latent_interconnect.chat import __main__ as chat_main
from latent_interconnect.chat import cli
from latent_interconnect.chat.config import ChatConfig
from latent_interconnect.config import Pin, load_config, load_resolved

KIMI = "kimi-k2.6"
STAGGERED = "esmc-6b-mlp4-staggered"
KIMI_WEIGHTS = cli._kimi_weights  # the real check, before the fixture fakes it
ALL_TASKS = [
    ("gemma-3-12b-it", "esmc-6b-mlp4-all-tasks", 5000, "protein", "esmc-6b"),
    ("gemma-3-12b-it", "chemfm-3b-mlp4-all-tasks", 5000, "molecule", "chemfm-3b"),
    (KIMI, f"{STAGGERED}-all-tasks", 4800, "protein", "esmc-6b"),
    (KIMI, "chemfm-3b-mlp4-all-tasks", 4800, "molecule", "chemfm-3b"),
]
"""The four connectors trained on all tasks: receiver, connector id, released step, domain, specialist."""


def test_the_shipped_config_offers_both_gemma_models_and_kimi_with_named_connectors():
    raw = cli.load_yaml(cli.DEFAULT_CONFIG.read_text())
    assert sorted(raw["receivers"]) == ["gemma-3-12b-it", "gemma-3-4b-it", KIMI]
    assert "esmc-600m" in raw["receivers"]["gemma-3-4b-it"]["connectors"]
    assert "esmc-600m" in raw["receivers"]["gemma-3-12b-it"]["connectors"]
    assert list(raw["receivers"][KIMI]["connectors"]) == [STAGGERED, *(c for r, c, *_ in ALL_TASKS if r == KIMI)]
    for rid, cid, step, *_ in ALL_TASKS:
        assert raw["receivers"][rid]["connectors"][cid]["step"] == step, cid
    for receiver in raw["receivers"].values():
        for cid, entry in receiver["connectors"].items():
            assert "tested" not in entry["label"], cid  # display labels carry no test status
            assert cid in cli.BUNDLES
            if cid in {"mimic", "pglm-1b"}:  # the release install cannot encode their raw sequences
                assert "raw sequence unavailable" in entry["label"]


def test_every_connector_of_the_full_chat_config_has_a_download_hint():
    """Each connector of ``configs/chat/all-connectors.yaml`` has its data bundles in ``BUNDLES``; the all-tasks ones
    name the all-tasks training set of their domain, never the released one."""
    raw = cli.load_yaml((cli.DEFAULT_CONFIG.parent / "all-connectors.yaml").read_text())
    for rid, receiver in raw["receivers"].items():
        for cid in receiver["connectors"]:
            names = cli._bundles(rid, cid)
            assert names[0] == f"{rid}__{cid}"
            assert len(names) == 6, cid  # the connector, domain, training set, items, embeddings, ridge start
            (training_set,) = [n for n in names if "-training-set" in n]
            assert training_set.endswith("-all-tasks") == cid.endswith("-all-tasks"), cid
    hint = cli._download_hint(KIMI, [f"{STAGGERED}-all-tasks"])
    assert "--connector kimi-k2.6__esmc-6b-mlp4-staggered-all-tasks" in hint
    assert "--data protein-training-set-all-tasks" in hint
    assert "--data ridge-kimi-k2.6__esmc-6b" in hint


@pytest.mark.parametrize(("receiver", "connector", "step", "domain", "specialist"), ALL_TASKS)
def test_the_default_command_selects_each_connector_trained_on_all_tasks(
    monkeypatch, tmp_path, receiver, connector, step, domain, specialist
):
    """Through the default config (``cli.DEFAULT_CONFIG``, ``configs/chat/release.yaml``), with stand-in pins for
    every pin it needs: the connector is selectable, and its narrowed config loads at its released checkpoint with the
    receiver's settings of the default config."""
    monkeypatch.setattr(cli, "_workspace_pins", _all_pins)
    assert cli._choose(cli.DEFAULT_CONFIG, receiver, [connector]) == [connector]
    narrowed = tmp_path / "chat.json"
    narrowed.write_text(json.dumps(cli.narrowed_config(cli.DEFAULT_CONFIG, receiver, [connector])))
    entry = load_config(narrowed, ChatConfig, _all_pins()).receivers[receiver]
    assert list(entry.connectors) == [connector]
    assert entry.connectors[connector].checkpoint_pin == f"connectors/{receiver}/{connector}/step_{step:06d}"
    assert entry.connectors[connector].run.encoder.name.startswith(specialist)
    default = cli.load_yaml(cli.DEFAULT_CONFIG.read_text())["receivers"][receiver]
    assert entry.context == default["context"]
    if receiver == KIMI:  # the Kimi generation plan of the default config
        assert entry.receiver.backends["generate"] == "kimi-vllm"
        assert entry.generation_plan is not None and entry.generation_plan.max_num_seqs == 1


@pytest.mark.parametrize(("receiver", "connector", "step", "domain", "specialist"), ALL_TASKS)
def test_the_default_command_prints_the_download_of_each_connector_trained_on_all_tasks(
    monkeypatch, receiver, connector, step, domain, specialist
):
    """Without the connector's checkpoint, the default command refuses it and prints the bundles that install it: the
    connector, its domain, the all-tasks training set of its domain (never the released one), its items, its
    specialist's embeddings and its receiver's ridge start."""
    checkpoint = f"connectors/{receiver}/{connector}/step_{step:06d}"
    monkeypatch.setattr(cli, "_workspace_pins", lambda: {n: p for n, p in _all_pins().items() if n != checkpoint})
    with pytest.raises(SystemExit) as refused:
        cli._choose(cli.DEFAULT_CONFIG, receiver, [connector])
    message = str(refused.value)
    assert f"not installed for {receiver}: {[connector]}" in message and checkpoint in message
    expected = (
        f"python -m latent_interconnect.release_download --connector {receiver}__{connector} --data {domain}-domain "
        f"--data {domain}-training-set-all-tasks --data {domain}-items --data embeddings-{specialist} "
        f"--data ridge-{receiver}__{specialist} --dest $LIP_DATA"
    )
    assert expected in message
    assert cli._download_hint(receiver, [connector]) == expected


def test_every_pin_of_the_shipped_config_is_reported_missing_from_an_empty_workspace():
    missing = cli.missing_pins(cli.DEFAULT_CONFIG, {})
    esmc = missing["gemma-3-4b-it"]["esmc-600m"]
    assert "connectors/gemma-3-4b-it/esmc-600m/step_002000" in esmc  # the checkpoint, from the run's destination
    assert {"protein-domain", "items/protein-training-monitor/item_set"} <= set(esmc)
    kimi = missing[KIMI][STAGGERED]
    assert "connectors/kimi-k2.6/esmc-6b-mlp4-staggered/step_002000" in kimi  # the released bundle's checkpoint pin
    assert "ridges/kimi-k2.6/esmc-6b" in kimi  # ridge-kimi-k2.6__esmc-6b's


def test_a_narrowed_config_keeps_one_receiver_and_loads_with_the_installed_pins(deployment, tmp_path):  # noqa: F811
    ws, path, _ = deployment
    assert cli.missing_pins(path, ws.pins())["tiny"] == {
        "foreign": [],
        "toy": [],
        "unpinned": ["toy/run/step_000060"],
    }
    narrowed = tmp_path / "elsewhere" / "chat.yaml"
    narrowed.parent.mkdir()
    narrowed.write_text(json.dumps(cli.narrowed_config(path, "tiny", ["toy"])))
    config = load_config(narrowed, ChatConfig, ws.pins())  # its $refs are absolute: it loads from anywhere
    assert list(config.receivers) == ["tiny"] and list(config.receivers["tiny"].connectors) == ["toy"]


def test_serve_refuses_an_unknown_model_and_an_uninstalled_connector(deployment, monkeypatch, capsys):  # noqa: F811
    ws, path, _ = deployment
    monkeypatch.setenv("LIP_DATA", str(ws.paths.data))
    with pytest.raises(SystemExit, match="no model 'gemma'"):
        cli.main(["serve", "--config", str(path), "--model", "gemma", "--no-auth"])
    with pytest.raises(SystemExit, match=r"not installed for tiny: \['unpinned'\]"):
        cli.main(["serve", "--config", str(path), "--model", "tiny", "--connectors", "unpinned", "--no-auth"])


def test_check_reports_the_installed_connectors(deployment, monkeypatch, capsys):  # noqa: F811
    ws, path, _ = deployment
    monkeypatch.setenv("LIP_DATA", str(ws.paths.data))
    status = cli.main(["check", "--config", str(path), "--encoder-python", f"lip={sys.executable}"])
    out = capsys.readouterr().out
    assert status == 0, out
    assert "tiny: installed connectors toy, foreign" in out and "unpinned: not installed (1 pins missing)" in out
    assert f"encoder env 'lip': {sys.executable}" in out


def test_check_warns_when_an_empty_cuda_visible_devices_hides_the_gpus(deployment, monkeypatch, capsys):  # noqa: F811
    ws, path, _ = deployment
    monkeypatch.setenv("LIP_DATA", str(ws.paths.data))
    argv = ["check", "--config", str(path), "--encoder-python", f"lip={sys.executable}"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert cli.main(argv) == 0
    assert "[warn] CUDA_VISIBLE_DEVICES is set but empty" in capsys.readouterr().out
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cli.main(argv)
    assert "CUDA_VISIBLE_DEVICES is set but empty" not in capsys.readouterr().out
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    cli.main(argv)
    assert "CUDA_VISIBLE_DEVICES is set but empty" not in capsys.readouterr().out


# --- Kimi K2.6: software evidence only (fakes for vLLM, the GPUs and the weights) -----------------------------------


@pytest.mark.parametrize("connector", [STAGGERED, f"{STAGGERED}-all-tasks"])
def test_the_kimi_chat_run_is_the_released_runs_but_for_its_destination(connector):
    chat_run = cli.DEFAULT_CONFIG.parent / "runs" / f"{KIMI}__{connector}.yaml"
    ours = load_resolved(chat_run, {}, unknown_pins="keep")
    source = load_resolved(ROOT / f"configs/runs/kimi-k2.6/{connector}.yaml", {}, unknown_pins="keep")
    achieved = cli.load_yaml((ROOT / f"configs/runs/kimi-k2.6/{connector}-achieved.yaml").read_text())
    assert ours.pop("destination") == achieved["destination"] == f"connectors/kimi-k2.6/{connector}"
    source.pop("destination")
    assert ours == source


def _all_pins():
    names = {n for c in cli.missing_pins(cli.DEFAULT_CONFIG, {}).values() for m in c.values() for n in m}
    return {n: Pin(target="artifact", sha256="0" * 64) for n in names}


def test_the_narrowed_kimi_config_validates_as_a_chat_config(tmp_path):
    narrowed = tmp_path / "kimi.yaml"
    narrowed.write_text(json.dumps(cli.narrowed_config(cli.DEFAULT_CONFIG, KIMI, [STAGGERED])))
    config = load_config(narrowed, ChatConfig, _all_pins())
    entry = config.receivers[KIMI]
    assert entry.receiver.backends["generate"] == "kimi-vllm" and entry.context == 16384
    assert entry.connectors[STAGGERED].checkpoint_pin == "connectors/kimi-k2.6/esmc-6b-mlp4-staggered/step_002000"
    assert entry.connectors[STAGGERED].run.encoder.name == "esmc-6b-l51l80"


class _FakeKimiVLLM:
    """The constructor signature that matters here: kimi-vllm takes max_model_len (its engine limit)."""

    def __init__(self, spec, *, weights=None, max_model_len=4096, gpu_memory_utilization=0.9):
        pass


@pytest.fixture
def kimi_node(monkeypatch):
    """A fake node: vLLM importable, eight whole H200s, the weights present, every pin installed; os.environ copied."""
    monkeypatch.setattr(os, "environ", dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
    monkeypatch.setenv("LIP_DATA", "/nonexistent")
    monkeypatch.setattr(cli, "_workspace_pins", _all_pins)
    real = cli.importlib.util.find_spec
    monkeypatch.setattr(
        cli.importlib.util, "find_spec", lambda m: object() if m in {"vllm", "compressed_tensors"} else real(m)
    )
    monkeypatch.setattr(cli, "_devices", lambda: ("NVIDIA H200",) * 8)
    monkeypatch.setattr(cli, "_kimi_weights", lambda entry, options: [])
    monkeypatch.setattr(latent_interconnect.receivers, "backend_class", lambda spec, capability: _FakeKimiVLLM)
    seen = {}

    def server(argv):
        seen["argv"] = argv
        seen["config"] = cli.load_yaml(Path(argv[argv.index("--config") + 1]).read_text())
        seen["environ"] = dict(os.environ)
        return 0

    monkeypatch.setattr(chat_main, "main", server)
    return seen


def test_kimi_is_served_without_the_yes_no_readout_it_cannot_score(kimi_node, capsys):
    kimi = ROOT / "configs" / "answers" / "kimi-k2.6.yaml"
    assert cli.main(["serve", "--model", KIMI, "--yes-no", str(kimi)]) == 0
    assert "--yes-no" not in kimi_node["argv"]  # scored by kimi-stream, generated by kimi-vllm: not offered
    assert "--yes-no: not offered for kimi-k2.6" in capsys.readouterr().err


def test_kimi_is_selected_by_model_and_gets_its_context_options_and_environment(kimi_node):
    status = cli.main(["serve", "--model", KIMI, "--option", "gpu_memory_utilization=0.85", "--option", "weights=/w"])
    argv = kimi_node["argv"]
    assert status == 0 and argv[argv.index("--receiver") + 1] == KIMI
    options = [a for a in argv if a.startswith("--option=")]
    assert options == ["--option=max_model_len=16384", "--option=gpu_memory_utilization=0.85", "--option=weights=/w"]
    assert (
        "--serial-requests" in argv
        and "--encoder-device" in argv
        and argv[argv.index("--encoder-device") + 1] == "cuda"
    )
    receiver = kimi_node["config"]["receivers"][KIMI]
    every = [STAGGERED, *(c for r, c, *_ in ALL_TASKS if r == KIMI)]  # every installed one
    assert list(kimi_node["config"]["receivers"]) == [KIMI] and list(receiver["connectors"]) == every
    assert receiver["generation_plan"] == {"max_num_seqs": 1, "max_num_batched_tokens": 16384}
    environ = kimi_node["environ"]
    assert environ["VLLM_USE_V2_MODEL_RUNNER"] == "1" and environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in environ


@pytest.mark.parametrize("connector", [c for r, c, *_ in ALL_TASKS if r == KIMI])
def test_kimi_serves_one_connector_trained_on_all_tasks_by_its_id(kimi_node, connector):
    """``serve --model kimi-k2.6 --connectors <id>`` with the default config: the narrowed config handed to the
    server holds that connector alone, with the Kimi generation plan and the context of the default config."""
    assert cli.main(["serve", "--model", KIMI, "--connectors", connector]) == 0
    receiver = kimi_node["config"]["receivers"][KIMI]
    assert list(receiver["connectors"]) == [connector] and receiver["connectors"][connector]["step"] == 4800
    assert receiver["generation_plan"] == {"max_num_seqs": 1, "max_num_batched_tokens": 16384}
    assert receiver["context"] == 16384


@pytest.mark.parametrize(
    ("devices", "modules", "match"),
    [
        (("NVIDIA H200",) * 8, {"compressed_tensors"}, "needs the module 'vllm': run aimm-chat from the separate vLLM"),
        (("NVIDIA H200",) * 4, {"vllm", "compressed_tensors"}, "exactly 8 visible whole GPUs on one node"),
        (("NVIDIA H200 MIG 1g.18gb",) * 8, {"vllm", "compressed_tensors"}, "exactly 8 visible whole GPUs"),
        ((), {"vllm", "compressed_tensors"}, "visible: none"),
    ],
)
def test_kimi_is_refused_before_any_load_off_its_environment_or_topology(
    kimi_node, monkeypatch, devices, modules, match
):
    monkeypatch.setattr(cli, "_devices", lambda: devices)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda m: object() if m in modules else None)
    with pytest.raises(SystemExit, match=match):
        cli.main(["serve", "--model", KIMI])
    assert "argv" not in kimi_node  # the server never started


def test_kimi_without_its_weights_is_refused_naming_the_download(kimi_node, monkeypatch):
    def missing(spec, weights):
        raise FileNotFoundError(f"{spec.model_id} at revision {spec.revision} is not in the Hugging Face cache")

    monkeypatch.setitem(
        sys.modules, "latent_interconnect.receivers.kimi_k26.weights", types.SimpleNamespace(snapshot_path=missing)
    )
    monkeypatch.setattr(cli, "_kimi_weights", KIMI_WEIGHTS)
    with pytest.raises(SystemExit, match=r"hf download moonshotai/Kimi-K2\.6 --revision 7eb5002f"):
        cli.main(["serve", "--model", KIMI])


def test_check_names_what_kimi_lacks_in_the_gemma_environment(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_workspace_pins", _all_pins)
    monkeypatch.setenv("LIP_DATA", "/nonexistent")
    assert cli.main(["check", "--model", KIMI]) == 1
    out = capsys.readouterr().out
    assert f"{KIMI}: installed connectors {STAGGERED}" in out
    assert "needs the module 'vllm'" in out


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _children(pid):
    """Every descendant process id of ``pid`` (Linux ``/proc``)."""
    out = []
    for task in Path(f"/proc/{pid}/task").iterdir():
        try:
            children = (task / "children").read_text().split()
        except FileNotFoundError:  # a thread that ended meanwhile
            continue
        for child in children:
            out += [int(child), *_children(int(child))]
    return out


def _alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except FileNotFoundError:
        return False
    return state != "Z"


@pytest.mark.skipif(not Path("/proc/self/task").exists(), reason="reads /proc for the worker processes")
@pytest.mark.parametrize("stop", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("entry_point", ["aimm-chat", "module"])
def test_a_served_deployment_answers_then_stops_with_its_encoder_worker(
    deployment,  # noqa: F811
    tmp_path,
    stop,
    entry_point,
):
    """Both entry points: ``aimm-chat serve`` and ``python -m latent_interconnect.chat`` (``chat.__main__``).

    The encoder worker is one that ignores the end of its input (it serves, then hangs), so it is gone after the stop
    only if the server's pool stopped it (``EncoderPool.close``: input closed, then a kill after 30 s), not because the
    server's exit closed its input.
    """
    ws, path, _ = deployment
    port = _free_port()
    # this checkout's package and tests, never an installed copy
    env = {**os.environ, "LIP_DATA": str(ws.paths.data), "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    hung = tmp_path / "hung-python"
    worker = (
        "import sys, time; from latent_interconnect.encoders.worker import main; main(sys.argv[1:]); time.sleep(600)"
    )
    hung.write_text(f'#!/bin/sh\nshift 2\nexec "{sys.executable}" -c "{worker}" "$@"\n')  # drops "-m <module>"
    hung.chmod(0o755)
    module = "latent_interconnect.chat.cli" if entry_point == "aimm-chat" else "latent_interconnect.chat.__main__"
    entry = f"import sys, tests.fakes; from {module} import main; sys.exit(main(sys.argv[1:]))"
    command = [sys.executable, "-c", entry]  # tests.fakes: the fake-toy domain
    if entry_point == "aimm-chat":
        command += ["serve", "--config", str(path), "--model", "tiny", "--connectors", "toy", "--no-auth"]
        command += ["--port", str(port), "--encoder-python", f"lip={hung}"]
    else:
        command += ["--config", str(path), "--receiver", "tiny", "--host", "127.0.0.1", "--port", str(port)]
        command += ["--no-auth", f"--python=lip={hung}", "--encoder-device", "cpu"]
    command += ["--store", str(tmp_path / "store")]
    log = (tmp_path / "server.log").open("w")
    server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{port}"
    workers: list[int] = []
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                with urlopen(url + "/api/health", timeout=5) as response:
                    if json.load(response)["ready"]:
                        break
            except OSError:
                pass
            assert server.poll() is None, (tmp_path / "server.log").read_text()
            time.sleep(0.2)
        body = {
            "messages": [{"role": "user", "content": "Here is a protein:[[P1]]. Describe it."}],
            "placeholders": {"P1": {"connector": "toy", "input_type": "sequence", "value": "MKWWKKCCAA"}},
        }
        request = Request(
            url + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urlopen(request, timeout=120) as response:
            events = [
                json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith("data: ")
            ]
        assert events[-1]["type"] == "done", events[-1]
        assert events[-1]["injections"]["P1"]["source"] == "live"
        workers = _children(server.pid)
        assert workers, "the raw value should have started an encoder worker"
        server.send_signal(stop)
        assert server.wait(timeout=120) == 0
        time.sleep(1)
        assert not [p for p in workers if _alive(p)], "the pool did not stop its encoder worker"
    finally:
        for pid in workers:
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
        if server.poll() is None:
            server.kill()
            server.wait()
        log.close()


# --- the defaults reach the server (a fake server: argv only) -------------------------------------------------------


@pytest.fixture
def captured(deployment, monkeypatch):  # noqa: F811
    ws, path, _ = deployment
    monkeypatch.setenv("LIP_DATA", str(ws.paths.data))
    seen = {}
    monkeypatch.setattr(chat_main, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    return path, seen


def test_the_local_defaults_reach_the_server_and_zero_lifts_them(captured):
    path, seen = captured
    assert cli.main(["serve", "--config", str(path), "--model", "tiny", "--no-auth"]) == 0
    argv = seen.pop("argv")
    for flag in ("--max-tokens-cap=256", "--max-prompt-tokens=2056", "--max-resident-encoders=1", "--serial-requests"):
        assert flag in argv
    assert "--secure-cookie" not in argv and argv[argv.index("--host") + 1] == "127.0.0.1"
    lifted = ["--max-tokens-cap", "0", "--max-prompt-tokens", "0", "--max-resident-encoders", "0", "--secure-cookie"]
    cli.main(["serve", "--config", str(path), "--model", "tiny", "--no-auth", *lifted])
    argv = seen.pop("argv")
    assert not [a for a in argv if a.startswith(("--max-tokens-cap", "--max-prompt-tokens", "--max-resident"))]
    assert "--secure-cookie" in argv


def test_the_answer_convention_reaches_the_server_checked(captured, tmp_path):
    path, seen = captured
    gemma = ROOT / "configs" / "answers" / "gemma-3.yaml"
    assert cli.main(["serve", "--config", str(path), "--model", "tiny", "--no-auth", "--yes-no", str(gemma)]) == 0
    argv = seen.pop("argv")
    assert argv[argv.index("--yes-no") + 1] == str(gemma.resolve())
    (tmp_path / "bad.yaml").write_text("instruction: Answer.\n'yes': ' Yes'\n'no': ' Yes'\n")
    for bad, match in ((tmp_path / "missing.yaml", "--yes-no"), (tmp_path / "bad.yaml", "must differ")):
        with pytest.raises(SystemExit, match=match):
            cli.main(["serve", "--config", str(path), "--model", "tiny", "--no-auth", "--yes-no", str(bad)])
    assert "argv" not in seen


def test_negative_limits_a_wildcard_bind_and_no_auth_off_loopback_are_refused(captured, capsys):
    path, seen = captured
    serve = ["serve", "--config", str(path), "--model", "tiny"]
    with pytest.raises(SystemExit):
        cli.main([*serve, "--max-prompt-tokens", "-1"])
    assert "must be 0 (no limit) or positive" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="wildcard bind"):
        cli.main([*serve, "--host", "0.0.0.0"])
    with pytest.raises(SystemExit, match="loopback only"):
        cli.main([*serve, "--host", "10.1.2.3", "--no-auth"])
    assert "argv" not in seen


def test_a_non_loopback_address_is_served_with_a_warning(captured, tmp_path, capsys):
    path, seen = captured
    cli.main(
        [
            "serve",
            "--config",
            str(path),
            "--model",
            "tiny",
            "--host",
            "10.1.2.3",
            "--password-file",
            str(tmp_path / "pw"),
        ]
    )
    assert seen["argv"][seen["argv"].index("--host") + 1] == "10.1.2.3"
    assert "not a loopback address" in capsys.readouterr().err


def test_a_missing_default_config_says_how_to_install(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "DEFAULT_CONFIG", tmp_path / "configs" / "chat" / "release.yaml")
    with pytest.raises(SystemExit, match="install the package from a clone"):
        cli.main(["check"])


def test_check_without_a_model_skips_the_models_of_another_environment(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_workspace_pins", _all_pins)
    monkeypatch.setenv("LIP_DATA", "/nonexistent")
    cli.main(["check"])  # here: no vLLM, so Kimi K2.6 is skipped, not failed
    out = capsys.readouterr().out
    assert f"[skip] {KIMI}: backend 'kimi-vllm' needs the module 'vllm'" in out
    assert "gemma-3-4b-it: installed connectors" in out and f"{KIMI}: installed connectors" not in out
