"""``aimm-chat``: serve the chat (``serve``) and check a machine for it, offline (``check``).

A thin command over ``python -m latent_interconnect.chat`` (``chat.__main__``), which does all the serving.
``aimm-chat serve --model <receiver id>`` serves one receiver of a chat config (default ``configs/chat/release.yaml``:
Gemma 3 4B, Gemma 3 12B and Kimi K2.6) on a loopback address, in the foreground, with these defaults: one whole
request at a time (``--serial-requests``), at most one resident live encoder (``--max-resident-encoders 1``), a
256-token answer cap and a 2,056-token prompt limit (the conditions of the measured Gemma runs: a starting point, not
a guarantee). Changing the model is a restart with another ``--model``. Ctrl-C or SIGTERM stops
it: the server shuts down and stops every encoder worker before the process exits. ``--yes-no ANSWER.yaml`` (the
model's answer convention, ``configs/answers/``) offers the UI's Yes/No probability, scored as the evaluation suites
score, for a model scored by its generation engine (Gemma; not Kimi K2.6, whose score backend is another engine).

The generation backend is the receiver config's (``backends.generate``, chosen by ``receivers.open_generation``):
``hf`` for Gemma, ``kimi-vllm`` for Kimi K2.6. Before anything loads, ``serve`` checks what a backend needs
(:data:`ENGINES`): for ``kimi-vllm`` the vLLM environment (vLLM and compressed-tensors importable, so ``aimm-chat`` runs
from that environment), exactly eight visible whole accelerators (``DeviceClass.H200X8``, the engine's tensor
parallel topology, which the backend checks again) and the pinned weights snapshot in the Hugging Face cache (or a
checked copy, ``--option weights=DIR``). It forwards the config's ``context`` as the engine's ``max_model_len`` to a
backend that takes one, and every ``--option K=V`` (a later one wins).

A chat config cites, through each connector's training run, pins that only the connector's release bundles install
(its checkpoint, cached embeddings, domain table, ridge start, training set and monitor items), and the server refuses
a config with a pin it lacks. ``serve`` therefore serves only the connectors whose pins are all installed (or exactly
those of ``--connectors``), from a copy of the config narrowed to them and to the served receiver. ``check`` reports,
without loading a model, the config, the installed connectors (and the bundles a missing one needs), what each
model's backend needs, the GPUs and their free memory (with a warning when an empty ``CUDA_VISIBLE_DEVICES`` hides
them), the Hugging Face login and the encoder interpreters.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

import latent_interconnect
from latent_interconnect.config import PIN_KEY, REF_KEY, load_config, load_resolved, load_yaml

__all__ = ["DEFAULT_CONFIG", "ENGINES", "Engine", "main", "missing_pins", "narrowed_config", "preflight"]

DEFAULT_CONFIG = Path(latent_interconnect.__file__).resolve().parents[2] / "configs" / "chat" / "release.yaml"
"""The shipped config, in the checkout the package is installed from (``pip install -e``)."""
DEFAULT_PASSWORD_FILE = Path("~/.config/aimm/chat-password")
DEFAULT_PORT = 8811
LOCAL_MAX_TOKENS = 256
LOCAL_MAX_PROMPT_TOKENS = 2056
LOCAL_MAX_RESIDENT_ENCODERS = 1


@dataclass(frozen=True)
class Engine:
    """What a generation backend needs before it loads.

    Attributes:
        modules: modules its environment must import (the hint names the environment).
        environment: the environment's name in the docs (``examples/guides/chat.md``).
        whole_gpus: the exact number of visible whole accelerators it runs on (None: any, CPU included).
        variables: environment variables set for it unless already set.
        unset: environment variables removed for it.
    """

    modules: tuple[str, ...]
    environment: str
    whole_gpus: int | None = None
    variables: Mapping[str, str] = field(default_factory=dict)
    unset: tuple[str, ...] = ()


ENGINES: Mapping[str, Engine] = {
    "hf": Engine(("transformers", "accelerate"), "the release environment (pip install -e '.[hf,chat]')"),
    "kimi-vllm": Engine(
        ("vllm", "compressed_tensors"),
        "the separate vLLM environment (env/locks/vllm.txt; examples/guides/chat.md, Kimi K2.6)",
        whole_gpus=8,
        # the worker extension wraps vLLM's V2 model runner; a 1 h engine start; flashinfer's sampler off (its JIT
        # build failed on mixed CUDA wheels); no float32 matmul precision or allocator override
        variables={
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        },
        unset=("VLLM_FLOAT32_MATMUL_PRECISION", "PYTORCH_CUDA_ALLOC_CONF"),
    ),
}
"""Generation backend name → what it needs (an unlisted backend: nothing checked)."""

_PROTEIN = ("protein-domain", "protein-training-set", "protein-items")
_MOLECULE = ("molecule-domain", "molecule-training-set", "molecule-items")
_PROTEIN_ALL_TASKS = ("protein-domain", "protein-training-set-all-tasks", "protein-items")
_MOLECULE_ALL_TASKS = ("molecule-domain", "molecule-training-set-all-tasks", "molecule-items")
BUNDLES: Mapping[str, tuple[str, ...]] = {
    "esmc-600m": (*_PROTEIN, "embeddings-esmc-600m", "ridge-{receiver}__esmc-600m"),
    "esmc-6b": (*_PROTEIN, "embeddings-esmc-6b", "ridge-{receiver}__esmc-6b"),
    "esmc-6b-mlp4-staggered": (*_PROTEIN, "embeddings-esmc-6b", "ridge-{receiver}__esmc-6b"),
    "esmc-6b-mlp4-staggered-all-tasks": (*_PROTEIN_ALL_TASKS, "embeddings-esmc-6b", "ridge-{receiver}__esmc-6b"),
    "esmc-6b-mlp4-all-tasks": (*_PROTEIN_ALL_TASKS, "embeddings-esmc-6b", "ridge-{receiver}__esmc-6b"),
    "chemfm-3b-mlp4-all-tasks": (*_MOLECULE_ALL_TASKS, "embeddings-chemfm-3b", "ridge-{receiver}__chemfm-3b"),
    "mimic": (*_PROTEIN, "embeddings-mimic", "ridge-{receiver}__mimic"),
    "pglm-1b": (*_PROTEIN, "embeddings-pglm-1b", "ridge-{receiver}__pglm-1b"),
    "chemfm-1b": (*_MOLECULE, "embeddings-chemfm-1b", "ridge-{receiver}__chemfm-1b"),
    "molformer-c3": (*_MOLECULE, "embeddings-molformer-c3", "ridge-{receiver}__molformer-c3"),
    "unimol2-1.1b": (*_MOLECULE, "embeddings-unimol2-1.1b", "ridge-{receiver}__unimol2-1.1b"),
}
"""Connector id → the data bundles its run cites (``release/data_full.md``, "Retrainable from released data"); the
connector bundle itself is ``<receiver>__<connector id>``."""


# -- the config ------------------------------------------------------------------------------------------------------


def _pins_in(node: Any, out: set[str]) -> set[str]:
    if isinstance(node, dict):
        if set(node) == {PIN_KEY}:
            out.add(str(node[PIN_KEY]))
        else:
            for value in node.values():
                _pins_in(value, out)
    elif isinstance(node, list):
        for value in node:
            _pins_in(value, out)
    return out


def missing_pins(config: Path, pins: Mapping[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Receiver id → connector id → the pins it needs that ``pins`` lacks (its checkpoint's included); [] if none."""
    raw = load_resolved(config, pins, unknown_pins="keep")
    out: dict[str, dict[str, list[str]]] = {}
    for rid, receiver in raw["receivers"].items():
        out[rid] = {}
        for cid, entry in receiver["connectors"].items():
            needed = _pins_in(entry, set())
            needed.add(f"{entry['run']['destination']}/step_{int(entry['step']):06d}")
            out[rid][cid] = sorted(n for n in needed if n not in pins)
    return out


def _absolute_refs(node: Any, base: Path) -> Any:
    if isinstance(node, dict):
        if set(node) == {REF_KEY}:
            return {REF_KEY: str((base / str(node[REF_KEY])).resolve())}
        return {k: _absolute_refs(v, base) for k, v in node.items()}
    if isinstance(node, list):
        return [_absolute_refs(v, base) for v in node]
    return node


def narrowed_config(config: Path, receiver: str, connectors: Sequence[str]) -> dict[str, Any]:
    """The chat config with only ``receiver`` and its ``connectors`` (in order), its ``$ref`` paths made absolute."""
    raw = load_yaml(config.read_text())
    entry = raw["receivers"][receiver]
    entry["connectors"] = {c: entry["connectors"][c] for c in connectors}
    raw["receivers"] = {receiver: entry}
    narrowed: dict[str, Any] = _absolute_refs(raw, config.resolve().parent)
    return narrowed


def _receiver(config: Path, receiver: str) -> dict[str, Any]:
    """The receiver entry, resolved (pins kept as names)."""
    entry: dict[str, Any] = load_resolved(config, {}, unknown_pins="keep")["receivers"][receiver]
    return entry


def _bundles(receiver: str, connector: str) -> list[str]:
    extra = [b.format(receiver=receiver) for b in BUNDLES.get(connector, ())]
    return [f"{receiver}__{connector}", *extra]


def _download_hint(receiver: str, connectors: Sequence[str]) -> str:
    args = []
    for cid in connectors:
        names = _bundles(receiver, cid)
        args += [f"--connector {names[0]}", *(f"--data {d}" for d in names[1:])]
    return "python -m latent_interconnect.release_download " + " ".join(dict.fromkeys(args)) + " --dest $LIP_DATA"


def _workspace_pins() -> dict[str, Any]:
    from latent_interconnect.workspace import Workspace  # noqa: PLC0415

    if not os.environ.get("LIP_DATA"):
        raise SystemExit("aimm-chat: set LIP_DATA to your data directory (examples/README.md)")
    return dict(Workspace.from_env().pins())


def _choose(config: Path, receiver: str, asked: Sequence[str]) -> list[str]:
    """The connectors to serve: those asked (each must be installed), else every installed one."""
    missing = missing_pins(config, _workspace_pins())[receiver]
    if asked:
        unknown = [c for c in asked if c not in missing]
        if unknown:
            raise SystemExit(f"aimm-chat: {receiver} has no connectors {unknown}; it has {sorted(missing)}")
        lacking = [c for c in asked if missing[c]]
        if lacking:
            raise SystemExit(
                f"aimm-chat: not installed for {receiver}: {lacking} (missing pins "
                f"{ {c: missing[c] for c in lacking} }).\nInstall them with:\n  {_download_hint(receiver, lacking)}"
            )
        return list(asked)
    chosen = [c for c, m in missing.items() if not m]
    if not chosen:
        first = "esmc-600m" if "esmc-600m" in missing else sorted(missing)[0]
        raise SystemExit(
            f"aimm-chat: no connector of {receiver} is installed in LIP_DATA. Install {first} with:\n  "
            + _download_hint(receiver, [first])
        )
    if skipped := [c for c, m in missing.items() if m]:
        sys.stderr.write(f"[aimm-chat] not installed, not served: {', '.join(skipped)} (aimm-chat check lists why)\n")
    return chosen


# -- the backend -----------------------------------------------------------------------------------------------------


def _devices() -> tuple[str, ...]:
    from latent_interconnect.provenance import visible_devices  # noqa: PLC0415

    return visible_devices()


def _hidden_gpus() -> str | None:
    """Why no GPU is visible, when ``CUDA_VISIBLE_DEVICES`` is set but empty (it hides every GPU), else None."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "unset").strip():
        return None
    return "CUDA_VISIBLE_DEVICES is set but empty, which hides every GPU: unset it (unset CUDA_VISIBLE_DEVICES)"


def preflight(entry: Mapping[str, Any], options: Mapping[str, str]) -> list[str]:
    """What keeps a receiver's generation backend from loading here, before any load ([]: nothing found).

    Args:
        entry: the receiver's resolved chat config entry.
        options: the backend options given (``weights`` names a checked copy of a Kimi snapshot).
    """
    backend = str(entry["receiver"]["backends"]["generate"])
    engine = ENGINES.get(backend)
    if engine is None:
        return []
    problems = _missing_modules(entry)
    if engine.whole_gpus is not None:
        devices = _devices()
        whole = [d for d in devices if "MIG" not in d]
        if len(devices) != engine.whole_gpus or len(whole) != len(devices):
            problems.append(
                f"backend {backend!r} runs tensor parallel over exactly {engine.whole_gpus} visible whole GPUs on "
                f"one node; visible: {list(devices) or 'none'} (set CUDA_VISIBLE_DEVICES or ask for a full node)"
            )
    if backend == "kimi-vllm" and not problems:
        problems += _kimi_weights(entry, options)
    return problems


def _missing_modules(entry: Mapping[str, Any]) -> list[str]:
    """The modules a receiver's generation backend needs that this environment lacks, as problems."""
    backend = str(entry["receiver"]["backends"]["generate"])
    engine = ENGINES.get(backend)
    if engine is None:
        return []
    return [
        f"backend {backend!r} needs the module {m!r}: run aimm-chat from {engine.environment}"
        for m in engine.modules
        if importlib.util.find_spec(m) is None
    ]


def _kimi_weights(entry: Mapping[str, Any], options: Mapping[str, str]) -> list[str]:
    from latent_interconnect.receivers.base import ReceiverSpec  # noqa: PLC0415
    from latent_interconnect.receivers.kimi_k26.weights import snapshot_path  # noqa: PLC0415

    spec = ReceiverSpec.model_validate(entry["receiver"])
    weights = options.get("weights")
    try:
        snapshot_path(spec, Path(weights) if weights else None)
    except (OSError, ValueError) as err:
        return [
            f"Kimi K2.6 weights: {err}. Download them first: hf download {spec.model_id} --revision {spec.revision}"
        ]
    return []


def _context_options(entry: Mapping[str, Any]) -> list[str]:
    """``max_model_len=<context>`` for a generation backend with an engine limit (``kimi-vllm``), else none."""
    from latent_interconnect.receivers import backend_class  # noqa: PLC0415
    from latent_interconnect.receivers.base import ReceiverSpec  # noqa: PLC0415

    cls = backend_class(ReceiverSpec.model_validate(entry["receiver"]), "generate")
    return [f"max_model_len={entry['context']}"] if "max_model_len" in inspect.signature(cls).parameters else []


def _engine_environment(backend: str) -> None:
    """Set (unless set) and unset the backend's environment variables, in this process (its workers inherit them)."""
    engine = ENGINES.get(backend)
    if engine is None:
        return
    for key, value in engine.variables.items():
        os.environ.setdefault(key, value)
    for key in engine.unset:
        os.environ.pop(key, None)
    cuda = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
    if backend == "kimi-vllm" and "CUDA_HOME" not in os.environ and cuda.is_dir():
        os.environ["CUDA_HOME"] = str(cuda)  # the vLLM environment's CUDA 13 wheels


def _pairs(items: Sequence[str], what: str) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not (sep and key and value):
            raise SystemExit(f"aimm-chat: {what} {item!r}: expected KEY=VALUE")
        out[key] = value
    return out


# -- the commands ----------------------------------------------------------------------------------------------------


def serve(args: argparse.Namespace) -> int:
    """Serve one receiver with the installed (or the asked) connectors; return the server's exit status."""
    from latent_interconnect.chat import __main__ as chat_main  # noqa: PLC0415
    from latent_interconnect.chat.server import check_bind, is_loopback  # noqa: PLC0415

    config = args.config.expanduser()
    try:
        check_bind(args.host, args.no_auth)
    except ValueError as err:
        raise SystemExit(f"aimm-chat: {err}") from err
    receivers = load_yaml(config.read_text()).get("receivers", {})
    if args.model not in receivers:
        raise SystemExit(f"aimm-chat: no model {args.model!r} in {config}; it has {sorted(receivers)}")
    entry = _receiver(config, args.model)
    options = _pairs(args.option, "--option")
    backend = str(entry["receiver"]["backends"]["generate"])
    _engine_environment(backend)  # before the checks, which may import the backend's libraries
    if problems := preflight(entry, options):
        raise SystemExit(f"aimm-chat: cannot serve {args.model}:\n  " + "\n  ".join(problems))
    connectors = _choose(config, args.model, [c for c in (args.connectors or "").split(",") if c])
    answer = _answer(args.yes_no, entry, args.model)
    if not is_loopback(args.host):
        sys.stderr.write(
            f"[aimm-chat] warning: serving on {args.host}, not a loopback address: anyone who can reach it and has "
            "the password can use it; the traffic is plain HTTP (behind HTTPS, add --secure-cookie)\n"
        )
    devices = _devices()
    if not devices:
        hidden = _hidden_gpus()
        sys.stderr.write(
            "[aimm-chat] no CUDA GPU visible: the model runs on CPU (slow)"
            + (f"; {hidden} to serve on a GPU" if hidden else "")
            + "\n"
        )
    with tempfile.TemporaryDirectory(prefix="aimm-chat-") as tmp:
        narrowed = Path(tmp) / f"{args.model}.yaml"
        narrowed.write_text(yaml.safe_dump(narrowed_config(config, args.model, connectors), sort_keys=False))
        argv = ["--config", str(narrowed), "--receiver", args.model, "--host", args.host, "--port", str(args.port)]
        argv += ["--no-auth"] if args.no_auth else ["--password-file", str(args.password_file.expanduser())]
        argv += ["--secure-cookie"] if args.secure_cookie else []
        argv += ["--deployment", f"aimm-chat-{args.model}-{os.getpid()}"]
        argv += ["--encoder-device", args.encoder_device or ("cuda" if devices else "cpu")]
        argv += [f"--python={k}={v}" for k, v in sorted(_pairs(args.encoder_python, "--encoder-python").items())]
        argv += [f"--option={o}" for o in [*_context_options(entry), *args.option]]  # a later option wins
        argv += ["--serial-requests"]
        argv += [f"--max-tokens-cap={args.max_tokens_cap}"] if args.max_tokens_cap > 0 else []
        argv += [f"--max-prompt-tokens={args.max_prompt_tokens}"] if args.max_prompt_tokens > 0 else []
        argv += [f"--max-resident-encoders={args.max_resident_encoders}"] if args.max_resident_encoders > 0 else []
        argv += ["--store", str(args.store.expanduser())] if args.store is not None else []
        argv += ["--yes-no", str(answer)] if answer is not None else []
        sys.stderr.write(
            f"[aimm-chat] model {args.model} ({backend}), connectors {', '.join(connectors)}; "
            f"open http://{args.host}:{args.port}/"
            + ("" if args.no_auth else f" (password: {args.password_file.expanduser()})")
            + "; readiness: /api/health; stop: Ctrl-C\n"
        )
        return chat_main.main(argv)  # Ctrl-C and SIGTERM: a clean exit there, encoder workers stopped


def _answer(path: Path | None, entry: Mapping[str, Any], model: str) -> Path | None:
    """The answer convention to pass on (checked), or None: none given, or a model scored by another engine."""
    if path is None:
        return None
    from latent_interconnect.data.answers import AnswerConvention  # noqa: PLC0415

    path = path.expanduser().resolve()
    try:
        load_config(path, AnswerConvention)
    except (OSError, ValueError) as err:
        raise SystemExit(f"aimm-chat: --yes-no {path}: {err}") from err
    backends = entry["receiver"]["backends"]
    if backends.get("score") != backends["generate"]:
        sys.stderr.write(
            f"[aimm-chat] --yes-no: not offered for {model}: it is scored by {backends.get('score')!r}, another engine "
            f"than its generation's {backends['generate']!r}\n"
        )
        return None
    return path


def _line(status: str, text: str) -> None:
    sys.stdout.write(f"[{status:>4}] {text}\n")


def _check_connectors(config: Path, models: Sequence[str], pins: Mapping[str, Any]) -> tuple[bool, dict[str, str]]:
    """Report each model's installed connectors; return whether each has one, and environment → a connector."""
    ok, environments = True, dict[str, str]()
    missing = missing_pins(config, pins)
    resolved = load_resolved(config, {}, unknown_pins="keep")
    for rid in models:
        installed = [c for c, m in missing[rid].items() if not m]
        ok = ok and bool(installed)
        _line("ok" if installed else "FAIL", f"{rid}: installed connectors {', '.join(installed) or 'none'}")
        for cid, lacking in missing[rid].items():
            if lacking:
                hint = _download_hint(rid, [cid])
                _line("info", f"{rid} {cid}: not installed ({len(lacking)} pins missing); install: {hint}")
            else:
                env = resolved["receivers"][rid]["connectors"][cid]["run"]["encoder"].get("environment", "?")
                environments.setdefault(str(env), cid)
    return ok, environments


def _check_encoders(environments: Mapping[str, str], given: Mapping[str, str]) -> bool:
    ok = True
    for env, cid in sorted(environments.items()):
        python = given.get(env)
        if python is None:
            _line("info", f"encoder env {env!r} ({cid}): no --encoder-python {env}=PATH: raw inputs refused, ids work")
            continue
        try:
            probe = subprocess.run(
                [shutil.which(python) or python, "-c", "import latent_interconnect"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            works, why = probe.returncode == 0, "" if probe.returncode == 0 else ": cannot import latent_interconnect"
        except (OSError, subprocess.TimeoutExpired) as err:
            works, why = False, f": {type(err).__name__}: {err}"
        ok = ok and works
        _line("ok" if works else "FAIL", f"encoder env {env!r}: {python}{why}")
    return ok


def _servable_here(config: Path, models: Sequence[str]) -> list[str]:
    """The models whose backend's modules this environment has; the others reported as skipped."""
    here = []
    for rid in models:
        if missing := _missing_modules(_receiver(config, rid)):
            _line("skip", f"{rid}: {missing[0]} (check it there with --model {rid})")
        else:
            here.append(rid)
    return here


def _check_gpus(models: Sequence[str]) -> None:
    """Report the visible GPUs and their free memory, and warn when none is visible (and why, if known)."""
    import torch  # noqa: PLC0415

    if hidden := _hidden_gpus():
        _line("warn", f"{hidden} or set it to the GPU ids before serving {', '.join(models)} on a GPU")
    if not torch.cuda.is_available():
        _line("warn", "no CUDA GPU visible: Gemma would run on CPU")
    for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
        free, total = torch.cuda.mem_get_info(i)
        _line("ok", f"GPU {i} {torch.cuda.get_device_name(i)}: {free / 2**30:.1f} GiB free of {total / 2**30:.1f}")


def check(args: argparse.Namespace) -> int:
    """Report what serving needs, without loading a model; return 1 if a checked model cannot be served."""
    config = args.config.expanduser()
    try:
        receivers = sorted(load_yaml(config.read_text())["receivers"])
    except (OSError, KeyError, TypeError, yaml.YAMLError) as err:
        _line("FAIL", f"config {config}: {err}")
        return 1
    _line("ok", f"config {config}: models {', '.join(receivers)}")
    if args.model is not None and args.model not in receivers:
        _line("FAIL", f"no model {args.model!r} in the config")
        return 1
    models = receivers if args.model is None else [args.model]
    if args.model is None:  # every model this environment can serve; another environment's are named, not failed
        models = _servable_here(config, models)
    ok = True
    for module in ("fastapi", "uvicorn"):
        found = importlib.util.find_spec(module) is not None
        ok = ok and found
        _line("ok" if found else "FAIL", f"python package {module}" + ("" if found else ": pip install -e '.[chat]'"))
    pins: dict[str, Any] = {}
    if os.environ.get("LIP_DATA"):
        pins = _workspace_pins()
        _line("ok", f"LIP_DATA {os.environ['LIP_DATA']}: {len(pins)} pins")
    else:
        ok = False
        _line("FAIL", "LIP_DATA is not set (examples/README.md)")
    installed, environments = _check_connectors(config, models, pins)
    options = _pairs(args.option, "--option")
    for rid in models:
        problems = preflight(_receiver(config, rid), options)
        installed = installed and not problems
        for problem in problems:
            _line("FAIL", f"{rid}: {problem}")
        if not problems:
            _line("ok", f"{rid}: its generation backend's requirements are met here")
    _check_gpus(models)
    if importlib.util.find_spec("huggingface_hub") is not None:
        from huggingface_hub import get_token  # noqa: PLC0415

        found = get_token() is not None
        _line("ok" if found else "warn", "Hugging Face token " + ("found" if found else "not found: hf auth login"))
    encoders = _check_encoders(environments, _pairs(args.encoder_python, "--encoder-python"))
    return 0 if ok and installed and encoders else 1


def main(argv: Sequence[str] | None = None) -> int:
    """The ``aimm-chat`` command."""
    parser = argparse.ArgumentParser(prog="aimm-chat", description="Chat with objects (examples/guides/chat.md).")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="chat config (default: %(default)s)")
    common.add_argument(
        "--encoder-python", action="append", default=[], metavar="ENV=PATH", help="an encoder environment's python"
    )
    common.add_argument("--option", action="append", default=[], metavar="K=V", help="a generation backend option")
    s = sub.add_parser("serve", parents=[common], help="serve one model in the foreground (Ctrl-C stops it)")
    s.add_argument("--model", required=True, help="a receiver id of the config, e.g. gemma-3-4b-it")
    s.add_argument("--connectors", default=None, help="comma-separated connector ids (default: every installed one)")
    s.add_argument("--host", default="127.0.0.1", help="address to serve on (default: %(default)s)")
    s.add_argument("--port", type=int, default=DEFAULT_PORT, help="(default: %(default)s)")
    s.add_argument("--password-file", type=Path, default=DEFAULT_PASSWORD_FILE, help="made if absent (mode 600)")
    s.add_argument("--no-auth", action="store_true", help="no password (loopback addresses only)")
    s.add_argument("--encoder-device", default=None, help="live encoders' device (default: cuda if visible, else cpu)")
    s.add_argument("--store", type=Path, default=None, help="chat's own store (default: $LIP_DATA/chat/store)")
    s.add_argument("--max-tokens-cap", type=int, default=LOCAL_MAX_TOKENS, help="answer cap; 0: the config's")
    s.add_argument("--max-prompt-tokens", type=int, default=LOCAL_MAX_PROMPT_TOKENS, help="0: the model's context")
    s.add_argument("--max-resident-encoders", type=int, default=LOCAL_MAX_RESIDENT_ENCODERS, help="0: no limit")
    s.add_argument(
        "--yes-no",
        type=Path,
        default=None,
        metavar="ANSWER.yaml",
        help="offer Yes/No probabilities, scored as the suites score: the model's answer convention "
        "(configs/answers/gemma-3.yaml for Gemma 3)",
    )
    c = sub.add_parser("check", parents=[common], help="check this machine for serving, offline")
    c.add_argument(
        "--model", default=None, help="check one receiver id (default: every model this environment can serve)"
    )
    s.add_argument("--secure-cookie", action="store_true", help="mark the session cookie Secure (behind HTTPS)")
    args = parser.parse_args(argv)
    if args.command == "serve":
        for flag in ("max_tokens_cap", "max_prompt_tokens", "max_resident_encoders"):
            if getattr(args, flag) < 0:
                parser.error(f"--{flag.replace('_', '-')} must be 0 (no limit) or positive")
    if not args.config.expanduser().is_file():
        missing = f"aimm-chat: no chat config at {args.config}"
        if args.config == DEFAULT_CONFIG:
            missing += (
                ": the shipped config lives in the checkout, so install the package from a clone "
                "(pip install -e .) or give --config"
            )
        raise SystemExit(missing)
    return serve(args) if args.command == "serve" else check(args)


if __name__ == "__main__":
    raise SystemExit(main())
