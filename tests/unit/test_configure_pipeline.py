"""``examples/configure_pipeline.py``: one command writes a consistent set of §6 configs for a receiver, an encoder
and a connector kind; on the shipped Qwen/ESM-C example it reproduces the shipped configs' identities."""

import importlib
import shlex
from pathlib import Path

import pytest
import yaml
from examples.configure_pipeline import auto_layers, configure, inspect, main, parse_layers

from latent_interconnect.config import load_config
from latent_interconnect.config_index import cited_pins, stand_in_pins
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.receivers.base import ReceiverSpec
from latent_interconnect.writes.connector import ConnectorSpec

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "examples" / "configs" / "pipeline"
QWEN, ESMC = PIPELINE / "qwen2.5-0.5b-instruct.yaml", PIPELINE / "esmc-300m.yaml"
STAGES = {
    "extract.yaml": "extract",
    "capture.yaml": "capture",
    "ridge.yaml": "ridge",
    "init.yaml": "init",
    "suite-init.yaml": "suite",
    "train.yaml": "train",
    "suite-trained.yaml": "suite",
    "training_set.yaml": "training_set",
    "items-train.yaml": "items",
}
# kind -> generated file -> the shipped file with the same identity (capture/ridge of affine and mlp: below)
SHIPPED = {
    "affine": {
        "connector.yaml": "connector.yaml",
        "init.yaml": "init.yaml",
        "train.yaml": "train.yaml",
        "suite-init.yaml": "suite-init.yaml",
        "suite-trained.yaml": "suite-trained.yaml",
        "extract.yaml": "extract.yaml",
    },
    "mlp": {"connector.yaml": "connector-mlp.yaml", "init.yaml": "init-mlp.yaml", "train.yaml": "train-mlp.yaml"},
    "xattn3": {
        "connector.yaml": "connector-xattn-3tok-staggered.yaml",
        "init.yaml": "init-xattn.yaml",
        "train.yaml": "train-xattn.yaml",
        "capture.yaml": "capture-staggered.yaml",
        "ridge.yaml": "ridge-staggered.yaml",
        "extract.yaml": "extract-res.yaml",
        "esmc-300m-res.yaml": "esmc-300m-res.yaml",
    },
}
NAMES = {"affine": "qwen-esmc300m", "mlp": "qwen-esmc300m-mlp", "xattn3": "qwen-esmc300m-xattn3"}


def _model(fname):
    if fname in STAGES:
        return importlib.import_module(f"latent_interconnect.run.{STAGES[fname]}").STAGE.config
    return ConnectorSpec if fname.startswith("connector") else EncoderRecipe


def _identity(path, model):
    return load_config(path, model, stand_in_pins(cited_pins(path))).identity()


@pytest.mark.parametrize(
    ("depth", "tokens", "layers"),
    [
        (24, 1, [[4, 8, 12, 16]]),  # Qwen2.5-0.5B: connector.yaml
        (24, 3, [[4, 10, 16], [6, 12, 18], [8, 14, 20]]),  # Qwen: connector-xattn-3tok-staggered.yaml
        (32, 1, [[5, 10, 15, 20]]),  # SmolLM2-360M
        (34, 1, [[6, 12, 18]]),  # Gemma 3 4B
        (16, 1, [[3, 6, 9]]),  # OLMo-2 1B
        (16, 3, [[3, 6, 9], [4, 7, 10], [5, 8, 11]]),
        (34, 3, [[6, 15], [9, 18], [12, 21]]),
        (30, 3, [[5, 11, 17], [7, 13, 19], [9, 15, 21]]),  # SmolLM2-135M
        (4, 1, [[1, 2]]),  # the tests' tiny Llama
    ],
)
def test_auto_layers_is_the_documented_rule(depth, tokens, layers):
    assert auto_layers(depth, tokens) == layers


def test_parse_layers():
    assert parse_layers("auto", 24, 1) == [[4, 8, 12, 16]]
    assert parse_layers("3,9", 24, 1) == [[3, 9]]
    assert parse_layers("4,10/6,12/8,14", 24, 3) == [[4, 10], [6, 12], [8, 14]]
    with pytest.raises(ValueError, match="give 3"):
        parse_layers("4,10", 24, 3)
    with pytest.raises(ValueError, match=r"outside 1 \.\. 23"):
        parse_layers("0,24", 24, 1)


@pytest.mark.parametrize("kind", sorted(SHIPPED))
def test_configure_reproduces_the_shipped_example(kind, tmp_path):
    out = tmp_path / kind
    files = configure(QWEN, ESMC, kind, out, name=NAMES[kind], pin_prefix="mine/qwen")
    for gen, shipped in SHIPPED[kind].items():
        model = _model(gen)
        assert _identity(out / gen, model) == _identity(PIPELINE / shipped, model), (kind, gen)
    for fname, stage in STAGES.items():  # every stage config loads with its refs
        load_config(out / fname, _model(fname), stand_in_pins(cited_pins(out / fname)))
        assert stage
    if kind != "xattn3":  # the shipped capture and ridge also fit layer 20, which no write uses
        for fname in ("capture.yaml", "ridge.yaml"):
            gen, shipped = _identity(out / fname, _model(fname)), _identity(PIPELINE / fname, _model(fname))
            assert gen.pop("layers") == [0, 4, 8, 12, 16] and shipped.pop("layers") == [0, 4, 8, 12, 16, 20]
            assert gen == shipped
    commands = files["commands.sh"].read_text()
    import re
    import subprocess

    assert subprocess.run(["bash", "-n", str(files["commands.sh"])], check=False).returncode == 0
    assert not [m for m in re.findall(r"\$\{[^}]*\}", commands) if "'" in m]  # a quote in ${...} spans lines
    run = "mine/qwen" if kind == "affine" else f"mine/qwen-{kind}"
    assert f"--pin {run}/train/step_000010 --run-config" in commands
    assert "--option python=" in commands  # esmc runs in its own environment
    assert "PYTHONPATH" not in commands  # esmc-300m names no plugins
    assert 'summarize_suites.py" --pin-prefix ' + run in commands
    assert '[ -s "$FASTA" ] ||' in commands  # step 12's sequence is checked before step 1
    assert commands.rstrip().splitlines()[-2].endswith(" $demo") and "demo=--demo; fi" in commands  # only DEMO=1


def test_configure_a_new_receiver_and_encoder(tmp_path):
    """Gemma 3 4B and ESM-2 8M, mlp: one command; layers from the rule, mlp defaults written, pins under the prefix."""
    out = tmp_path / "g"
    configure(
        ROOT / "configs/receivers/gemma-3-4b-it.yaml",
        ROOT / "examples/configs/esm2-8m.yaml",
        "mlp",
        out,
        pin_prefix="alice/gemma4b",
    )
    spec = load_config(out / "connector.yaml", ConnectorSpec)
    assert spec.layers.per_token == ((6, 12, 18),) and spec.kind_params == {"d_s": 256, "ff_mult": 2}
    assert spec.input.recipe == "esm2-8m" and list(spec.input.layers) == [6] and spec.receiver == "gemma-3-4b-it"
    assert spec.name == "esm2-8m-mlp"  # <recipe>-<kind>: its bundle is gemma-3-4b-it__esm2-8m-mlp
    train = yaml.safe_load((out / "train.yaml").read_text())
    assert train["destination"] == "alice/gemma4b-mlp/train" and train["ridge"] == {
        "name": {"$pin": "alice/gemma4b/ridge"}
    }
    commands = (out / "commands.sh").read_text()
    assert "--option python=" not in commands  # environment: lip runs in-process
    pythonpath, check = (line for line in commands.splitlines() if "PYTHONPATH" in line)  # before step 1
    assert pythonpath.startswith('export PYTHONPATH="$AIMM${PYTHONPATH:+:$PYTHONPATH}"')  # examples.esm2_extractor
    assert check.startswith("python -c 'import importlib") and "examples.esm2_extractor" in check
    assert commands.index(check) < commands.index("steps 1-2")
    assert yaml.safe_load((out / "capture.yaml").read_text())["layers"] == [0, 6, 12, 18]


def test_configure_xattn3_on_a_residue_recipe_keeps_it(tmp_path):
    configure(QWEN, PIPELINE / "esmc-300m-res.yaml", "xattn3", tmp_path / "x", d_s=128)
    spec = load_config(tmp_path / "x" / "connector.yaml", ConnectorSpec)
    assert spec.input.recipe == "esmc-300m-res" and spec.kind_params["d_s"] == 128
    assert not (tmp_path / "x" / "esmc-300m-res-res.yaml").exists()


def test_configure_refuses(tmp_path):
    with pytest.raises(ValueError, match="divisible by 4"):
        configure(QWEN, ESMC, "xattn3", tmp_path / "a", d_s=130)
    configure(QWEN, ESMC, "affine", tmp_path / "b")
    with pytest.raises(FileExistsError):
        configure(QWEN, ESMC, "affine", tmp_path / "b")
    with pytest.raises(SystemExit):
        main(["--receiver", str(QWEN)])
    with pytest.raises(ValueError, match="--data-fasta and --data-tsv together"):
        configure(QWEN, ESMC, "affine", tmp_path / "c", data_fasta=PIPELINE / "esmc-300m.yaml")
    with pytest.raises(ValueError, match=r"--data-tsv .*nope\.tsv: no such file"):
        configure(QWEN, ESMC, "affine", tmp_path / "c", data_fasta=QWEN, data_tsv=tmp_path / "nope.tsv")
    assert not (tmp_path / "c").exists()  # checked before anything is written
    base = ["--receiver", str(QWEN), "--encoder", str(ESMC), "--kind", "affine", "--out", str(tmp_path / "d")]
    with pytest.raises(SystemExit) as stop:  # an argparse error (exit 2)
        main([*base, "--data-tsv", str(QWEN)])
    assert stop.value.code == 2


def test_inspect_a_local_model(tmp_path):
    from tests.fakes import build_tiny_llama

    spec = build_tiny_llama(tmp_path / "tiny")
    text = inspect(str(tmp_path / "tiny"), revision="0" * 40)
    got = ReceiverSpec.model_validate(yaml.safe_load(text))
    assert (got.depth, got.d_model, got.layers_path, got.revision) == (
        spec.depth,
        spec.d_model,
        "model.layers",
        "0" * 40,
    )
    assert "one token" in text and "auto write layers: one token [[1, 2]]" in text
    assert "as an encoder recipe" not in text  # the encoder line is for encoder-only models


def test_inspect_cli_placeholder_of_one_and_of_two_tokens(tmp_path, capsys):
    """A one-token placeholder prints the receiver YAML (exit 0); a two-token one prints nothing on stdout, the reason
    and a one-token candidate on stderr, and exits 3, so ``> my-receiver.yaml`` holds no misleading config."""
    from examples.configure_pipeline import NOT_USABLE
    from tests.fakes import build_tiny_llama

    build_tiny_llama(tmp_path / "tiny")  # the stub tokenizer: one token per character, " c" one token
    base = ["--inspect", str(tmp_path / "tiny"), "--revision", "0" * 40]
    assert main([*base, "--placeholder", " Q"]) == 0
    out, err = capsys.readouterr()
    assert ReceiverSpec.model_validate(yaml.safe_load(out)).placeholder == " Q" and err == ""
    assert main([*base, "--placeholder", " XY"]) == NOT_USABLE == 3
    out, err = capsys.readouterr()
    assert out == ""
    assert "no receiver config: placeholder ' XY' encodes to 2 tokens" in err
    assert "try --placeholder ' X' (one token, id " in err, err  # the default comes first among the candidates


@pytest.mark.parametrize("kind", ["llava", "t5"])
def test_inspect_an_unbuildable_causal_lm_is_not_usable(tmp_path, capsys, kind):
    """``architectures`` naming a causal LM that AutoModelForCausalLM cannot build (Llava, T5): nothing on stdout,
    the reason on stderr, exit 3; no encoder-recipe line (stub configs only, no download)."""
    from examples.configure_pipeline import NOT_USABLE
    from transformers import LlavaConfig, T5Config

    if kind == "llava":
        stub = LlavaConfig(text_config={"model_type": "llama", "num_hidden_layers": 2, "hidden_size": 32})
        stub.architectures = ["LlavaForConditionalGeneration"]
    else:
        stub = T5Config(num_layers=2, d_model=32, d_ff=64, num_heads=2, d_kv=16)
        stub.architectures = ["T5ForConditionalGeneration"]
    stub.save_pretrained(tmp_path / kind)
    assert main(["--inspect", str(tmp_path / kind), "--revision", "0" * 40]) == NOT_USABLE
    out, err = capsys.readouterr()
    assert out == "", out
    assert "no receiver config: " in err and "AutoModelForCausalLM cannot build it" in err, err
    assert stub.architectures[0] in err and "as an encoder recipe" not in err


@pytest.mark.parametrize("architectures", [["BertForMaskedLM"], None])
def test_inspect_an_encoder_only_model_prints_no_receiver_config(tmp_path, architectures):
    """A BERT-like encoder (as Rostlab/prot_bert): encoder facts and no receiver config, from a stub config only."""
    from transformers import BertConfig

    stub = BertConfig(num_hidden_layers=3, hidden_size=32, num_attention_heads=2, intermediate_size=64)
    stub.architectures = architectures
    stub.save_pretrained(tmp_path / "bert")
    text = inspect(str(tmp_path / "bert"), revision="0" * 40)
    assert "num_hidden_layers 3, hidden_size 32" in text and "read: [3]" in text
    assert "not a causal LM: no receiver config printed" in text
    assert all(line.startswith("#") for line in text.splitlines()), text  # no YAML body
    assert "# as an encoder recipe: revision: " + "0" * 40 + ", read: [3]" in text
    assert main(["--inspect", str(tmp_path / "bert"), "--revision", "0" * 40]) == 0  # informative, exit 0


def _steps_1_to_3(tmp_path, out, *, prefix):
    """Write the configs under ``out`` and return a copy of their ``commands.sh`` cut before step 4 (no model)."""
    configure(QWEN, ESMC, "affine", out, pin_prefix=prefix)
    text = (out / "commands.sh").read_text()
    script = out / "steps-1-3.sh"
    script.write_text(text[: text.index("t step 4")])
    return script


def _run(script, env, cwd):
    import subprocess

    done = subprocess.run(["bash", str(script)], env=env, cwd=cwd, capture_output=True, text=True, check=False)
    return done.returncode, done.stdout + done.stderr


def test_commands_sh_skips_only_its_own_computation(tmp_path):
    """Steps 1-3 run for real (the domain and item stages, CPU, no model) against one LIP_DATA:

    a fresh root runs them; an unchanged rerun skips them; a second ``--pin-prefix`` with the same data skips the shared
    data steps; a changed setting or annotation stops the script with the one-line refusal and exit 1.
    """
    import os
    import shutil
    import sys

    pkg = tmp_path / "installed"  # an installed copy of the package: its records are no-vcs, so a dirty tree runs too
    skip_caches = shutil.ignore_patterns("__pycache__")
    shutil.copytree(ROOT / "src" / "latent_interconnect", pkg / "latent_interconnect", ignore=skip_caches)
    aimm = tmp_path / "aimm"  # a checkout copy whose example data the test may edit
    shutil.copytree(ROOT / "examples", aimm / "examples", ignore=skip_caches)
    (tmp_path / "my.fasta").write_text(">P05204\nMPK\n")
    env = os.environ | {
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(pkg),
        "AIMM": str(aimm),
        "LIP_DATA": str(tmp_path / "lip"),
    }
    first = _steps_1_to_3(tmp_path, tmp_path / "a", prefix="mine/qwen")
    rc, log = _run(first, env, tmp_path)
    assert rc == 0, log
    assert "domain: record" in log and log.count("items: record") == 3 and "(pinned from this config)" not in log
    sources = {p.name: p.read_bytes() for p in (tmp_path / "lip" / "mine" / "proteins").iterdir()}

    rc, log = _run(first, env, tmp_path)  # (b) unchanged: a true resume
    assert rc == 0, log
    assert log.count("(pinned from this config)") == 4 and ": record " not in log

    second = _steps_1_to_3(tmp_path, tmp_path / "b", prefix="alice/qwen")  # (d) another prefix, the same data
    rc, log = _run(second, env, tmp_path)
    assert rc == 0, log
    assert log.count("(pinned from this config)") == 4

    items = first.parent / "items-train.yaml"  # (c) a setting changed
    original = items.read_text()
    items.write_text(original.replace("size: 24", "size: 20"))
    rc, log = _run(first, env, tmp_path)
    refusal = (
        "commands.sh: LIP_DATA already holds mine/items-train from different inputs/settings; "
        "use a fresh LIP_DATA, and for route (a) a fresh run directory (keep the old ones)"
    )
    assert rc == 1 and refusal in log and "skip mine/protein-domain" in log, log
    items.write_text(original.replace("size: 24", "size: many"))  # a config error is no reuse conflict
    rc, log = _run(first, env, tmp_path)
    assert rc == 1 and "items-train.yaml does not load (ValidationError: " in log, log
    assert "fix the config and rerun" in log and "already holds" not in log and "recover with" not in log, log
    items.write_text(original)

    tsv = aimm / "examples" / "data" / "proteins.tsv"  # (c) one annotation changed
    tsv.write_text(tsv.read_text().replace("\ttrain\t3D-structure;", "\ttrain\t3D-structure;Transferase;", 1))
    rc, log = _run(first, env, tmp_path)
    assert rc == 1 and "LIP_DATA already holds mine/protein-domain from different inputs/settings" in log, log
    assert "step 3: item sets" not in log.split("holds mine/protein-domain")[1]  # it stopped at the domain
    assert {p.name: p.read_bytes() for p in (tmp_path / "lip" / "mine" / "proteins").iterdir()} == sources


def test_verify_pin_refuses_what_it_cannot_verify(tmp_path, monkeypatch, capsys):
    from examples.configure_pipeline import ABSENT, verify_pin

    monkeypatch.setenv("LIP_DATA", str(tmp_path / "lip"))
    configure(QWEN, ESMC, "affine", tmp_path / "c")
    config = tmp_path / "c" / "items-train.yaml"
    assert verify_pin("items", config, "mine/items-train") == ABSENT  # nothing pinned: the step runs
    pins = tmp_path / "lip" / "lip" / "pins.yaml"
    pins.parent.mkdir(parents=True)
    digest = "ab" * 32
    pins.write_text(yaml.safe_dump({"mine/items-train": {"target": "artifact", "sha256": digest}}))
    assert verify_pin("items", config, "mine/items-train") == 1  # pinned without its record pin
    assert "no mine/items-train@record pin" in capsys.readouterr().err
    pins.write_text(
        yaml.safe_dump(
            {
                "mine/items-train": {"target": "artifact", "sha256": digest},
                "mine/items-train@record": {"target": "record", "sha256": digest},
            }
        )
    )
    assert verify_pin("items", config, "mine/items-train") == 1  # its record is missing
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("commands.sh: LIP_DATA already holds mine/items-train from")
    assert main(["--pinned", "mine/items-train"]) == 0 and main(["--pinned", "mine/items"]) == 1


def _tree(*roots):
    """Return path -> sha256 of every file under ``roots``."""
    import hashlib

    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for r in roots for p in sorted(r.rglob("*")) if p.is_file()}


def test_a_refused_rerun_rewrites_nothing_and_prints_the_recovery(tmp_path):
    """A changed annotation on a pinned domain is checked from a candidate under ``domain-check``: the rerun is refused
    with the two recovery routes, and every file of the run directory and ``LIP_DATA`` keeps its bytes (the run's
    ``domain-filled.yaml`` and the pinned sources included); new files appear only under the candidate directory.
    The training data are given by ``--data-fasta``/``--data-tsv``, which ``fill()`` and the recovery carry. Every
    path holds a space: the printed commands are quoted, parse, and route (a)'s configure and copy lines run, carrying
    a hand edit of ``train.yaml`` over the regenerated configs."""
    import os
    import shutil
    import subprocess
    import sys

    pkg = tmp_path / "installed"  # an installed copy of the package: its records are no-vcs, so a dirty tree runs too
    shutil.copytree(
        ROOT / "src" / "latent_interconnect", pkg / "latent_interconnect", ignore=shutil.ignore_patterns("__pycache__")
    )
    base = tmp_path / "a b"
    work = base / "work 1"  # the run directory; route (a)'s new one goes beside it, under ``base``
    data = base / "my data"
    work.mkdir(parents=True)
    data.mkdir()
    fasta, tsv = data / "p s.fasta", data / "p s.tsv"
    shutil.copy(ROOT / "examples" / "data" / "proteins.fasta", fasta)
    shutil.copy(ROOT / "examples" / "data" / "proteins.tsv", tsv)
    (work / "my seq.fasta").write_text(">P05204\nMPK\n")
    lip, run = base / "lip 1", work / "configs"
    env = os.environ | {
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(pkg),
        "AIMM": str(ROOT),  # read only: make_domain.py writes under LIP_DATA and the run directory
        "LIP_DATA": str(lip),
        "FASTA": "my seq.fasta",
        "LICENSE": "PENDING: demo only",
    }
    configure(QWEN, ESMC, "xattn3", run, pin_prefix="alice/qwen", data_fasta=fasta, data_tsv=tsv)
    text = (run / "commands.sh").read_text()
    assert f'make_domain.py" {shlex.quote(str(fasta))} {shlex.quote(str(tsv))} ' in text
    script = run / "steps-1-3.sh"
    script.write_text(text[: text.index("t step 4")])
    rc, log = _run(script, env, work)  # a fresh root: the domain is filled into the run directory
    assert rc == 0 and "domain: record" in log, log
    assert not (run / "domain-check").exists()
    before = _tree(run, lip)
    assert run / "domain-filled.yaml" in before

    tsv.write_text(tsv.read_text().replace("\ttrain\t3D-structure;", "\ttrain\t3D-structure;Transferase;", 1))
    rc, log = _run(script, env, work)
    assert rc == 1 and "LIP_DATA already holds mine/protein-domain from different inputs/settings" in log, log
    recovery = log.split("(keep the old ones)")[1]
    assert "commands.sh: recover with (a) a new run beside this one" in recovery
    assert "The cp line carries this run's editable configs over to the new run" in recovery
    assert "do not carry over" not in recovery
    keep = "Keep or set ENCODER_PY (the encoder environment's interpreter), which this recipe uses."
    assert recovery.count(keep) == 2, recovery  # ESM-C runs in its own environment: both routes say so
    q = shlex.quote
    configure_line = (
        f'  cd "$(mktemp -d {str(base).replace(" ", chr(92) + " ")}/pipeline.XXXXXX)" && '
        'export LIP_DATA=$PWD/data && python "$AIMM/examples/configure_pipeline.py" '
        f"--receiver {QWEN} --encoder {ESMC} --kind xattn3 --pin-prefix alice/qwen "
        f"--data-fasta {q(str(fasta))} --data-tsv {q(str(tsv))} --out configs"
    )
    assert configure_line in recovery, recovery
    lines = [line for line in recovery.splitlines() if line.startswith("  ")]
    cp_line = next(line for line in lines if line.startswith("  cp -- "))
    not_editable = {"commands.sh", "steps-1-3.sh", "domain-filled.yaml", "domain-check"}
    editable = sorted({f.name for f in run.iterdir()} - not_editable)
    assert cp_line.endswith("/{" + ",".join(editable) + "} configs/"), cp_line
    assert all(n in cp_line for n in ("domain.yaml", "train.yaml", "items-train.yaml", "connector.yaml"))
    assert not any(n in cp_line for n in ("domain-filled", "domain-check", "commands.sh"))
    assert "# not copied: domain-filled.yaml and domain-check/ (filled by fill()) and commands.sh" in recovery
    run_line = next(line for line in lines if line.startswith("  export FASTA="))
    assert run_line.endswith(" ID=P05204 LICENSE=PENDING:\\ demo\\ only && bash configs/commands.sh"), run_line
    variables = run_line.strip().removeprefix("export ").split(" && ")[0]  # FASTA, ID and LICENSE, quoted
    route_b = (
        f"LIP_DATA=<new dir> OUT=<new hub dir> {variables} bash {str(run).replace(' ', chr(92) + ' ')}/commands.sh"
    )
    assert route_b in recovery, recovery  # route (b) carries LICENSE (and DEMO) as route (a) does
    for line in lines:  # every printed command parses
        done = subprocess.run(["bash", "-n", "-c", line], capture_output=True, text=True, check=False)
        assert done.returncode == 0, (line, done.stderr)
    after = _tree(run, lip)
    assert {p: after.get(p) for p in before} == before  # every old file keeps its bytes
    assert all((run / "domain-check") in p.parents for p in set(after) - set(before)), sorted(set(after) - set(before))
    assert (run / "domain-check" / "domain-filled.yaml") in after

    train = run / "train.yaml"  # route (a)'s configure and copy lines, pasted: a hand edit carries over
    train.write_text(train.read_text() + "# a hand edit\n")
    route = "set -e; " + configure_line.strip() + "; " + cp_line.strip() + "; pwd; " + run_line.strip().split(" && ")[0]
    done = subprocess.run(["bash", "-c", route], env=env, cwd=work, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    new = Path(done.stdout.strip().splitlines()[-1]) / "configs"
    assert new.parent.parent == base
    assert {n: (new / n).read_bytes() for n in editable} == {n: (run / n).read_bytes() for n in editable}


def test_the_recovery_names_encoder_py_only_for_a_recipe_outside_lip():
    from examples.configure_pipeline import _recovery

    keep = "Keep or set ENCODER_PY"
    assert keep not in "\n".join(_recovery(["--kind", "mlp"], ["train.yaml"]))
    assert "\n".join(_recovery(["--kind", "mlp"], ["train.yaml"], encoder_py=True)).count(keep) == 2
