"""Write the worked pipeline's configs (``examples/guides/extending.md`` §6) for a receiver, encoder, connector kind.

    python examples/configure_pipeline.py --receiver RECEIVER.yaml --encoder RECIPE.yaml --kind affine|mlp|xattn3 \
        --out DIR [--layers auto|L,L,...|L,L/L,L/...] [--name NAME] [--pin-prefix PREFIX] [--d-s N] \
        [--data-fasta SEQUENCES.fasta --data-tsv ANNOTATIONS.tsv]
    python examples/configure_pipeline.py --inspect HF_MODEL_ID [--revision REV] [--placeholder " X"]
    python examples/configure_pipeline.py --verify-pin STAGE CONFIG PIN | --pinned PIN   (commands.sh's checks)

**Configure.** From the shipped templates in ``examples/configs/pipeline/``, write into ``DIR`` one consistent set of
stage configs, named as in §6's table: ``domain.yaml`` (the template ``make_domain.py`` fills in), ``items-*.yaml``,
``extract.yaml``, ``capture.yaml``, ``ridge.yaml``, ``connector.yaml``, ``init.yaml``, ``suite-init.yaml``,
``training_set.yaml``, ``train.yaml`` and ``suite-trained.yaml``, plus copies of the receiver config and the encoder
recipe, and ``commands.sh``, the stage commands in order (steps 1 to 12, the suites' summary by
``examples/summarize_suites.py`` and packaging, ``--demo`` only with ``DEMO=1``). Before step 1 it checks that the
recipe's ``plugins`` import (putting ``$AIMM`` on ``PYTHONPATH`` for the checkout's ``examples.*``) and that step 12's
``FASTA`` exists. Its ``fill()`` builds the domain from ``--data-fasta`` and ``--data-tsv`` (``make_domain.py``'s
inputs; default the example data in ``examples/data/``). The connector's name defaults to ``<recipe>-<kind>``, so its
bundle is ``<receiver>__<recipe>-<kind>``. The kind is one argument:

- ``affine`` and ``mlp``: one slot token, pooled embeddings (``mlp``: ``kind_params`` ``d_s`` and ``ff_mult: 2``);
- ``xattn3``: three slot tokens of ``xattn`` (``d_s``, ``heads: 4``, ``d_k: 32``) with staggered write layers,
  reading per-residue states. A recipe without ``residues: true`` gets a residue copy, ``<name>-res.yaml`` (the same
  recipe with ``residues: true`` and float32 rows, a new recipe name), which the extraction and the ridge fits read.

``d_s`` is ``min(256, d_model)`` (for ``xattn3`` rounded down to a multiple of the 4 heads) unless ``--d-s`` sets it;
it must also be at most the connector's input width ``d_in`` (the encoder width times the read layers), which this
script does not know: give a smaller ``--d-s`` for an encoder narrower than 256.

**Write layers.** ``--layers auto`` (the default) applies :func:`auto_layers`, the starting point of §6 (it is not a
layer choice: ``examples/guides/choosing_layers.md``); explicit layers are ``4,8,12,16`` for one token or
``4,10,16/6,12,18/8,14,20`` per token. The layers are part of the connector's identity, so they are written into
``connector.yaml``; ``capture.yaml`` and ``ridge.yaml`` take layer 0 and every write layer.

**Pins.** The data steps keep the templates' pins (``mine/protein-domain``, ``mine/items-*``,
``mine/protein/training-set``, ``mine/<recipe>/embeddings``). The receiver's capture and ridge are pinned under
``PREFIX`` (``PREFIX/capture``, ``PREFIX/ridge``; ``-staggered`` for ``xattn3``), and the connector's init, training and
suites under ``PREFIX`` for ``affine`` and ``PREFIX-<kind>`` otherwise, as §6 names them (``mine/qwen/...``,
``mine/qwen-mlp/...``, ``mine/qwen-xattn3/...``). ``PREFIX`` defaults to ``mine/<receiver name>``; a connector you will
share is better pinned under your own name (``--pin-prefix alice/gemma4b``), since the bundle installs its
checkpoint under ``<train destination>/step_NNNNNN``.

**Reruns.** ``commands.sh`` skips a step only when its pin holds this step's own computation: before each step,
:func:`verify_pin` (``--verify-pin``) loads the step's config with the workspace's pins and compares its digest with
the ``config_sha`` of the record pinned at ``<pin>@record``. A different digest, or a pin it cannot verify, stops the
script with one line, then the two recovery routes (:func:`_recovery`), and exit 1. A pinned domain is checked from
a candidate filled under ``DIR/domain-check``. The domain, item and training-set pins are fixed names, so
``--pin-prefix`` alone does not isolate them: use a fresh ``LIP_DATA`` when the data, the recipe or the settings change,
and share one only between pipelines whose upstream data inputs are identical (then the shared steps are skipped).

**Inspect.** ``--inspect`` prints, for a Hugging Face model id, its revision (the commit of ``--revision``, default
``main``), ``num_hidden_layers`` and ``hidden_size`` (the text config's, for a multimodal model), and for a causal LM
the decoder-layer path, that the placeholder encodes to one token, the auto write layers and a receiver config
ready to save (``> my-receiver.yaml``); for an encoder, the revision and ``num_hidden_layers`` are what its recipe's
``revision`` and final ``read`` layer take (its ``as an encoder recipe`` line). A causal LM it can write no usable
receiver config for (a ``--placeholder``, default ``" X"``, that encodes to more than one token, no decoder-layer
list, or one ``AutoModelForCausalLM`` cannot build, as for Llava or T5) prints nothing on stdout, the report and the
reason on stderr, and exits :data:`NOT_USABLE` (3); for a placeholder, the reason names the first of
:data:`PLACEHOLDER_CANDIDATES` that is one token, to rerun with as ``--placeholder``. The placeholder is encoded on
its own, without special tokens, as the renderer encodes it, and its one id is inserted for each slot token (a space
before ``<SLOT:x>`` is removed), so a leading space only selects which token id that is. Only an encoder-only model
(its ``architectures`` name no ``CausalLM`` or ``ForConditionalGeneration``, or, when it lists none, it is neither a
decoder nor an encoder-decoder and its causal-LM class is not one or cannot be built, as for ``Rostlab/prot_bert``)
gets these encoder lines and ``not a causal LM: no receiver config printed``, with exit 0. It reads the model's
config and tokenizer only (no weights; a gated model needs a login).
"""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

TEMPLATES = Path(__file__).resolve().parent / "configs" / "pipeline"
KINDS = ("affine", "mlp", "xattn3")
TOKENS = {"affine": 1, "mlp": 1, "xattn3": 3}
CONNECTOR_TEMPLATES = {
    "affine": "connector.yaml",
    "mlp": "connector-mlp.yaml",
    "xattn3": "connector-xattn-3tok-staggered.yaml",
}
CPU_STEPS = 10
"""The templates' training steps (``train.yaml``): suite-trained reads the checkpoint of this step."""
ABSENT = 10
"""Exit code of ``--verify-pin`` when nothing is pinned under the name: the step runs."""
CAUSAL_MARKERS = ("CausalLM", "ForConditionalGeneration")
"""``--inspect``: a model whose architecture names hold none of these is an encoder (no receiver config printed)."""
NOT_USABLE = 3
"""Exit code of ``--inspect`` for a causal LM it can write no usable receiver config for (nothing on stdout)."""
PLACEHOLDER_CANDIDATES = (" X", "X", " Z", " Q", " @", "Z", "Q", "@")
"""``--inspect``: the texts tried as a one-token placeholder when ``--placeholder`` is not one token."""
REUSE_REFUSED = (
    "LIP_DATA already holds {pin} from different inputs/settings; "
    "use a fresh LIP_DATA, and for route (a) a fresh run directory (keep the old ones)"
)
"""The one line ``--verify-pin`` stops ``commands.sh`` with (exit 1)."""
BAD_CONFIG = 4
"""Exit code of ``--verify-pin`` for a pinned step whose config does not load: fix the config (no new-run advice)."""
GENERATED = ("domain-filled.yaml", "domain-check/", "commands.sh")
"""The run directory's files a new run does not copy back: filled by ``fill()``, or written for this directory."""


def pinned_names(names: Sequence[str], pin: str) -> list[str]:
    """Return the names a ``--pin PIN`` stage run makes that are among ``names``: ``PIN``, ``PIN@record``, ``PIN/*``."""
    return sorted(n for n in names if n in (pin, f"{pin}@record") or n.startswith(f"{pin}/"))


def verify_pin(stage: str, config: Path, pin: str) -> int:
    """Return whether ``commands.sh`` may skip the step that pins ``pin``: 0 skip, :data:`ABSENT` run, 1 refuse.

    0 when the record pinned at ``PIN@record`` is the computation this config defines in this workspace (its
    ``config_sha`` is the digest of the config loaded with the current pins, so a changed upstream step or source file
    changes it), :data:`ABSENT` when no name of the step is pinned, and 1, with :data:`REUSE_REFUSED` on stderr, when
    the digests differ or the pin cannot be verified (no record pin, no record). A config that does not load is no
    reuse conflict: :data:`BAD_CONFIG`, with the error and "fix the config" on stderr.
    """
    import importlib  # noqa: PLC0415

    from latent_interconnect.workspace import Workspace  # noqa: PLC0415

    ws = Workspace.from_env()
    pins = ws.pins()
    if not pinned_names(list(pins), pin):
        return ABSENT
    why = ""
    try:
        record_pin = pins.get(f"{pin}@record")
        if record_pin is None:
            raise LookupError(f"no {pin}@record pin")
        executions = ws.records.executions(record_pin.sha256)
        if not executions:
            raise LookupError(f"no record {record_pin.sha256[:12]}")
    except Exception as err:  # any failure to verify refuses the reuse
        why = f" (not verified: {_first_line(err)})"
    else:
        try:
            model = importlib.import_module(f"latent_interconnect.run.{stage}").STAGE.config
            digest = ws.load(config, model).digest()
        except Exception as err:  # a config error, not a reuse conflict
            sys.stderr.write(f"commands.sh: {config} does not load ({_first_line(err)}): fix the config and rerun\n")
            return BAD_CONFIG
        if executions[0].kind == stage and executions[0].config_sha == digest:
            return 0
    sys.stderr.write("commands.sh: " + REUSE_REFUSED.format(pin=pin) + why + "\n")
    return 1


def _first_line(err: BaseException) -> str:
    return f"{type(err).__name__}: {str(err).splitlines()[0] if str(err) else ''}"


def auto_layers(depth: int, tokens: int = 1) -> list[list[int]]:
    """Return the starting-point write layers for ``depth`` decoder layers (``examples/guides/extending.md`` §6).

    ``s`` is ``depth / 6`` rounded to the nearest integer (halves up, at least 1), ``t = floor(2 depth / 3)``:

    - one token writes at ``s, 2s, 3s, ...`` up to ``t``;
    - ``k > 1`` staggered tokens use the offset ``o = max(1, floor(s / 2))``: the first token writes at
      ``s, s + k o, s + 2 k o, ...`` up to ``t``, and token ``i`` at the first token's layers plus ``i o``, so the
      tokens interleave every ``o`` layers and the last one reaches ``(k - 1) o`` past ``t``.

    Qwen2.5-0.5B (24): ``[[4, 8, 12, 16]]``, three tokens ``[[4, 10, 16], [6, 12, 18], [8, 14, 20]]``;
    Gemma 3 4B (34): ``[[6, 12, 18]]``. Layers at or past ``depth`` are dropped.
    """
    if depth < 2 or tokens < 1:
        raise ValueError(f"need depth >= 2 and tokens >= 1, got depth {depth}, tokens {tokens}")
    s = max(1, int(depth / 6 + 0.5))
    top = (2 * depth) // 3
    if tokens == 1:
        return [list(range(s, top + 1, s))]
    o = max(1, s // 2)
    first = list(range(s, top + 1, tokens * o))
    out = [[L + i * o for L in first if L + i * o < depth] for i in range(tokens)]
    if not all(out):
        raise ValueError(f"depth {depth} is too shallow for {tokens} staggered tokens: give --layers")
    return out


def parse_layers(text: str, depth: int, tokens: int) -> list[list[int]]:
    """Return the per-token write layers ``--layers`` gives (``auto``, ``4,8,12`` or ``4,10/6,12/...``).

    Raises:
        ValueError: for a token count other than ``tokens``, or a layer outside ``1 .. depth - 1``.
    """
    if text == "auto":
        return auto_layers(depth, tokens)
    lists = [[int(x) for x in part.split(",") if x.strip()] for part in text.split("/")]
    if len(lists) != tokens or not all(lists):
        raise ValueError(f"--layers {text!r}: give {tokens} non-empty list(s) of layers, separated by '/'")
    bad = sorted({L for t in lists for L in t if not 1 <= L < depth})
    if bad:
        raise ValueError(f"--layers {text!r}: layers {bad} are outside 1 .. {depth - 1} (depth {depth})")
    return lists


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return data


def _dump(path: Path, header: str, data: dict[str, Any]) -> None:
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=None, allow_unicode=True, width=120)
    path.write_text("".join(f"# {line}\n" for line in header.splitlines()) + body)


def _stream_width(d_s: int | None, d_model: int, heads: int) -> int:
    """Return ``d_s``, by default ``min(256, d_model)`` rounded down to a multiple of ``heads``; check it."""
    width = d_s if d_s is not None else min(256, d_model) // heads * heads
    if not 0 < width <= d_model or width % heads:
        raise ValueError(f"d_s {width}: must be in 1 .. d_model ({d_model}) and divisible by {heads}")
    return width


def residue_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    """Return ``recipe`` with per-residue states kept: a new name ``<name>-res``, ``residues: true``, float32 rows."""
    out = dict(recipe)
    out["name"] = f"{recipe['name']}-res"
    out["residues"] = True
    out["residue_dtype"] = "fp32"
    return out


def configure(
    receiver_path: Path,
    encoder_path: Path,
    kind: str,
    out: Path,
    *,
    layers: str = "auto",
    name: str | None = None,
    pin_prefix: str | None = None,
    d_s: int | None = None,
    data_fasta: Path | None = None,
    data_tsv: Path | None = None,
) -> dict[str, Path]:
    """Write the pipeline's configs and ``commands.sh`` into ``out`` (module docstring); return file name -> path.

    ``data_fasta`` and ``data_tsv`` are the training data ``commands.sh``'s ``fill()`` passes to ``make_domain.py``
    (default ``examples/data/proteins.fasta`` and ``proteins.tsv`` of ``$AIMM``).

    Raises:
        ValueError: for an unknown kind, bad layers, a ``d_s`` the receiver's width or the heads do not allow, or only
            one of ``data_fasta`` and ``data_tsv`` (or one that is not a file).
        FileExistsError: if ``out`` already holds a file this would write.
    """
    if kind not in KINDS:
        raise ValueError(f"--kind {kind!r}: one of {', '.join(KINDS)}")
    data_flags, data_words = _data_words(data_fasta, data_tsv)  # checked before anything is written
    receiver, recipe = _load(receiver_path), _load(encoder_path)
    depth, d_model, rname = int(receiver["depth"]), int(receiver["d_model"]), str(receiver["name"])
    per_token = parse_layers(layers, depth, TOKENS[kind])
    residues = kind == "xattn3"
    width = _stream_width(d_s, d_model, heads=4 if residues else 1)
    prefix = pin_prefix or f"mine/{rname}"
    run_prefix = prefix if kind == "affine" else f"{prefix}-{kind}"
    rec_file, enc_file, stag = receiver_path.name, encoder_path.name, "-staggered" if residues else ""
    files: dict[str, Path] = {}
    out.mkdir(parents=True, exist_ok=True)

    def target(fname: str) -> Path:
        p = out / fname
        if p.exists():
            raise FileExistsError(f"{p} exists: write into a new directory")
        files[fname] = p
        return p

    (target(rec_file)).write_text(receiver_path.read_text())
    target(enc_file).write_text(encoder_path.read_text())
    if residues and not recipe.get("residues"):
        recipe = residue_recipe(recipe)
        enc_file = f"{recipe['name']}.yaml"
        _dump(target(enc_file), f"{encoder_path.name} with per-residue states kept, for the xattn3 connector.", recipe)
    rname_enc, read = str(recipe["name"]), list(recipe["read"])
    src = f"generated by examples/configure_pipeline.py ({kind}, {rname} x {rname_enc}) from examples/configs/pipeline/"
    emb_pin = f"mine/{rname_enc}/embeddings"
    rec_ref, enc_ref = {"$ref": rec_file}, {"$ref": enc_file}
    fit_layers = sorted({0, *(L for t in per_token for L in t)})
    conn_name = name or f"{_load(encoder_path)['name']}-{kind}"  # the given recipe's name, not its -res copy

    for t in ("domain.yaml", "items-train.yaml", "items-eval.yaml", "items-all.yaml", "training_set.yaml"):
        target(t).write_text(
            f"# copied by examples/configure_pipeline.py from examples/configs/pipeline/{t}\n"
            + (TEMPLATES / t).read_text()
        )

    extract = _load(TEMPLATES / "extract.yaml") | {"destination": emb_pin, "recipe": enc_ref}
    _dump(target("extract.yaml"), f"Embeddings of every object with {enc_file}; {src}extract.yaml", extract)
    capture = _load(TEMPLATES / "capture.yaml") | {
        "destination": f"{prefix}/capture{stag}",
        "receiver": rec_ref,
        "layers": fit_layers,
    }
    _dump(
        target("capture.yaml"),
        f"The receiver's name states at layer 0 and every write layer; {src}capture.yaml",
        capture,
    )
    ridge = _load(TEMPLATES / "ridge.yaml") | {
        "destination": f"{prefix}/ridge{stag}",
        "input": {"recipe": rname_enc, "layers": read},
        "encoder": enc_ref,
        "embeddings": [{"$pin": emb_pin}],
        "targets": {"$pin": f"{prefix}/capture{stag}"},
        "layers": fit_layers,
    }
    _dump(target("ridge.yaml"), f"Ridge maps from the embedding to the captured states; {src}ridge.yaml", ridge)

    connector = _connector(
        kind,
        name=conn_name,
        receiver=rname,
        recipe=rname_enc,
        read=read,
        per_token=per_token,
        note=f"configure_pipeline.py --layers {layers} on {rname} ({depth} layers)",
        d_s=width,
    )
    _dump(
        target("connector.yaml"),
        f"The {kind} connector, {len(per_token)} slot token(s); {src}{CONNECTOR_TEMPLATES[kind]}",
        connector,
    )
    ridge_ref = {"name": {"$pin": f"{prefix}/ridge{stag}"}}
    init = {
        "destination": f"{run_prefix}/init",
        "connector": {"$ref": "connector.yaml"},
        "receiver": rec_ref,
        "encoder": enc_ref,
        "ridge": ridge_ref,
    }
    _dump(target("init.yaml"), f"The step-0 connector from the ridge maps; {src}init.yaml", init)
    train = _load(TEMPLATES / "train.yaml") | {
        "destination": f"{run_prefix}/train",
        "name": f"{conn_name}-tiny",
        "connector": {"$ref": "connector.yaml"},
        "receiver": rec_ref,
        "encoder": enc_ref,
        "ridge": ridge_ref,
        "embeddings": [{"$pin": emb_pin}],
    }
    if int(train["steps"]) != CPU_STEPS:
        raise ValueError(f"the train.yaml template runs {train['steps']} steps; this script expects {CPU_STEPS}")
    _dump(target("train.yaml"), f"Ten training steps on CPU; {src}train.yaml", train)
    _suites(target, run_prefix=run_prefix, receiver=rec_ref, embeddings=emb_pin, src=src)

    commands = _commands(
        out,
        rec=rec_file,
        enc=enc_file,
        recipe=recipe,
        prefix=prefix,
        run=run_prefix,
        stag=stag,
        emb=emb_pin,
        editable=sorted(files),
        flags=[
            *("--receiver", str(receiver_path.resolve()), "--encoder", str(encoder_path.resolve()), "--kind", kind),
            *(["--layers", layers] if layers != "auto" else []),
            *(["--name", name] if name else []),
            *(["--pin-prefix", pin_prefix] if pin_prefix else []),
            *(["--d-s", str(d_s)] if d_s is not None else []),
            *data_flags,
        ],
        data=data_words,
    )
    target("commands.sh").write_text(commands)  # not in ``editable``: written for this run directory
    return files


def _data_words(fasta: Path | None, tsv: Path | None) -> tuple[list[str], list[str]]:
    """Return the ``--data-fasta``/``--data-tsv`` flags given and ``fill()``'s two shell words (default the example).

    Raises:
        ValueError: for only one of the two, or one that is not a file.
    """
    if (fasta is None) != (tsv is None):
        raise ValueError("give --data-fasta and --data-tsv together (make_domain.py reads both)")
    for flag, f in (("--data-fasta", fasta), ("--data-tsv", tsv)):
        if f is not None and not f.is_file():
            raise ValueError(f"{flag} {f}: no such file")
    given = [(flag, f.resolve()) for flag, f in (("--data-fasta", fasta), ("--data-tsv", tsv)) if f is not None]
    words = [
        shlex.quote(str(f.resolve())) if f else f'"$AIMM/examples/data/proteins.{ext}"'
        for f, ext in ((fasta, "fasta"), (tsv, "tsv"))
    ]
    return [w for flag, f in given for w in (flag, str(f))], words


def _suites(
    target: Callable[[str], Path], *, run_prefix: str, receiver: dict[str, str], embeddings: str, src: str
) -> None:
    """Write ``suite-init.yaml`` and ``suite-trained.yaml`` for the run under ``run_prefix``."""
    for which, ck, step in (
        ("init", f"{run_prefix}/init", 0),
        ("trained", f"{run_prefix}/train/step_{CPU_STEPS:06d}", CPU_STEPS),
    ):
        suite = _load(TEMPLATES / f"suite-{which}.yaml") | {
            "destination": f"{run_prefix}/suite-{which}",
            "receiver": receiver,
            "checkpoints": {"x": {"$pin": ck}},
            "inputs": {"x": [{"$pin": embeddings}]},
            "step": step,
        }
        _dump(target(f"suite-{which}.yaml"), f"Yes/No scores of the {which} connector; {src}suite-{which}.yaml", suite)


def _connector(
    kind: str,
    *,
    name: str,
    receiver: str,
    recipe: str,
    read: list[int],
    per_token: list[list[int]],
    note: str,
    d_s: int,
) -> dict[str, Any]:
    """Return the connector config of ``kind`` from its template, with these names, layers and ``kind_params``."""
    connector = _load(TEMPLATES / CONNECTOR_TEMPLATES[kind]) | {
        "name": name,
        "receiver": receiver,
        "input": {"recipe": recipe, "layers": read},
        "layers": {"rule": "explicit", "per_token": per_token},
        "layer_choice": {"note": f"{note}, without a layer study (examples/guides/choosing_layers.md)"},
    }
    if kind == "mlp":
        connector["kind_params"] = {"d_s": d_s, "ff_mult": 2}
    elif kind == "xattn3":
        connector["kind_params"] = {"d_s": d_s, "heads": 4, "d_k": 32}
    return connector


def _commands(
    out: Path,
    *,
    rec: str,
    enc: str,
    recipe: dict[str, Any],
    prefix: str,
    run: str,
    stag: str,
    emb: str,
    editable: list[str],
    flags: list[str],
    data: list[str],
) -> str:
    """Return ``commands.sh``: §6's steps for these configs, each skipped when its pin holds its record.

    ``flags`` are this configuration's ``configure_pipeline.py`` arguments but ``--out``, printed in the recovery a
    refused step prints (:func:`_recovery`) with ``editable``, the configs a new run copies back; ``data`` the shell
    words of ``fill()``'s FASTA and annotation table.
    """
    lip = recipe.get("environment") == "lip"
    py_enc = "python" if lip else '"${ENCODER_PY:?set ENCODER_PY to the encoder environment interpreter}"'
    opt = "" if lip else ' --option python="${ENCODER_PY:?set ENCODER_PY to the encoder environment interpreter}"'
    p = shlex.quote(str(out.resolve()))
    plugins = [str(m) for m in recipe.get("plugins") or ()]
    lines = [
        "#!/usr/bin/env bash",
        "# The worked pipeline (examples/guides/extending.md §6) for these configs,",
        "# written by examples/configure_pipeline.py.",
        "# Run it from a work directory outside the checkout, with LIP_DATA, AIMM (the checkout) and, for an encoder",
        "# environment other than lip, ENCODER_PY set; FASTA (default my.fasta) and ID (default P05204) name the",
        "# sequence step 12 asks about. Use a fresh run directory and LIP_DATA (keep the old ones) when the data,",
        "# recipe or settings change: the domain, item and training-set pins are fixed names (--pin-prefix does not",
        "# isolate them), and packaging will not overwrite another checkpoint's bundle; so share a LIP_DATA only",
        "# between pipelines with identical upstream data inputs. A step is skipped only if its pin holds this",
        "# config's record (configure_pipeline.py --verify-pin); a pin from other inputs or settings stops the script",
        "# (exit 1).",
        "# DEMO=1 marks the bundle demo: true (a software demo). OUT (default hub/connectors,",
        "# relative to the work directory) is packaging's output: the bundle goes to $OUT/<receiver>__<connector>.",
        "set -euo pipefail",
        f"P={p}",
        ': "${AIMM:?set AIMM to the checkout}" "${LIP_DATA:?set LIP_DATA to your data root}"',
        "FASTA=${FASTA:-my.fasta} ID=${ID:-P05204}",
        *_preflight(plugins, py_enc),
        't() { echo "[$(date +%H:%M:%S)] $*"; }',
        'guard() { python "$AIMM/examples/configure_pipeline.py" "$@"; }',
        'stage() { local s=$1 c=$2 pin=$3 rc=0; shift 3; guard --verify-pin "$s" "$P/$c" "$pin" || rc=$?; '
        'if [ "$rc" = 0 ]; then echo "skip $pin (pinned from this config)"; elif [ "$rc" = '
        f"{ABSENT}"
        ' ]; then python -m latent_interconnect.run.$s --config "$P/$c" --pin "$pin" "$@"; elif [ "$rc" = '
        f"{BAD_CONFIG}"
        " ]; then exit 1; else refused; fi; }",
        *_recovery(flags, editable, encoder_py=not lip),
        'ck() { python -c "from latent_interconnect.workspace import Workspace as W; w = W.from_env(); '
        "print(w.artifacts.path(w.artifacts.ref(w.pins()['$1'].sha256)))\"; }",
        "",
        "t steps 1-2: the domain",
        "# steps 1-2: the domain (examples/data; for your own proteins, your FASTA and annotation table)",
        f'fill() {{ python "$AIMM/examples/make_domain.py" {data[0]} {data[1]} "$P/domain.yaml" "$P/$1"; }}',
        "# a pinned domain is checked from a candidate under $P/domain-check (sources and filled config): its pinned",
        "# sources and this run's domain-filled.yaml are never rewritten",
        "dom=domain-filled.yaml",
        "if guard --pinned mine/protein-domain; then dom=domain-check/domain-filled.yaml; "
        'LIP_DATA="$P/domain-check" fill "$dom"; else fill "$dom"; fi',
        'stage domain "$dom" mine/protein-domain',
        "t step 3: item sets",
        "for s in train eval all; do stage items items-$s.yaml mine/items-$s; done",
        "t step 4: embeddings",
        f"stage extract extract.yaml {emb}{opt}",
        "t steps 5-8: capture, ridge, the step-0 connector and its scores",
        f"stage capture capture.yaml {prefix}/capture{stag}",
        f"stage ridge ridge.yaml {prefix}/ridge{stag}",
        f"stage init init.yaml {run}/init",
        f"stage suite suite-init.yaml {run}/suite-init",
        "t steps 9-11: the training set, ten training steps, the trained scores",
        "stage training_set training_set.yaml mine/protein/training-set",
        f"stage train train.yaml {run}/train",
        f"stage suite suite-trained.yaml {run}/suite-trained",
        "t the summary of both suites",
        f'python "$AIMM/examples/summarize_suites.py" --pin-prefix {run}',
        "t step 12: embed a sequence and ask about it",
        f'{py_enc} "$AIMM/examples/embed.py" "$P/{enc}" "$FASTA" embedded',
        f"EMB=embedded/$ID{'.residues' if recipe.get('residues') else ''}.npy",
        f"CK=$(ck {run}/train/step_{CPU_STEPS:06d})",
        'echo "the trained checkpoint (ask.py takes this path): $CK"',
        'python "$AIMM/examples/ask.py" "$P/' + rec + '" "$CK" "$EMB" \\',
        '  "Does the protein <SLOT:x> bind DNA? Answer yes or no, then one sentence." --max-new-tokens 48',
        'echo "ask your own question (any text with <SLOT:x> where the protein goes):"',
        'echo "  python \\"\\$AIMM/examples/ask.py\\" \\"$P/' + rec + '\\" \\"$CK\\" \\"$EMB\\" '
        '\\"What is known about the function of the protein <SLOT:x>?\\" --max-new-tokens 96"',
        "t packaging",
        "# packaging (examples/guides/contributing-a-connector.md): LICENSE is your connector's own license;",
        "# DEMO=1 adds --demo",
        'demo=""; if [ "${DEMO:-0}" = 1 ]; then demo=--demo; fi',
        f"python -m latent_interconnect.bundle_connector --pin {run}/train/step_{CPU_STEPS:06d} "
        '--run-config "$P/train.yaml" \\',
        '  --out "${OUT:-hub/connectors}" --license "${LICENSE:?set LICENSE to the license of your connector}" $demo',
        "t done",
        "",
    ]
    return "\n".join(lines)


def _recovery(flags: list[str], editable: list[str], *, encoder_py: bool = False) -> list[str]:
    """Return ``commands.sh``'s ``refused``: after :data:`REUSE_REFUSED`, print the two recovery routes and exit 1.

    (a) a new run directory beside the current one (``$PWD``) with its own ``LIP_DATA``, from these
    ``configure_pipeline.py`` flags; then ``editable`` (the run directory's configs but :data:`GENERATED`) copied
    over the regenerated ones, which carries this run's editable configs, hand edits included, to the new run; then
    the run with ``FASTA``, ``ID`` and, when set, ``LICENSE`` and ``DEMO``. (b) this run with a fresh ``LIP_DATA``
    and ``OUT``, and the same ``FASTA``, ``ID`` and, when set, ``LICENSE`` and ``DEMO``. With ``encoder_py`` (a
    recipe whose environment is not ``lip``), each route reminds the user to keep or set ``ENCODER_PY``. Paths
    are quoted to paste (``printf %q``); nothing is deleted or rebound.
    """
    q = shlex.quote
    names = ",".join(q(n) for n in editable)
    keep = (
        " Keep or set ENCODER_PY (the encoder environment's interpreter), which this recipe uses." if encoder_py else ""
    )
    words = [
        q(
            "commands.sh: recover with (a) a new run beside this one (this one is kept). The cp line carries this"
            " run's editable configs over to the new run; the --data-* flags carry over too." + keep
        ),
        q('  cd "$(mktemp -d ')
        + '"$here"'
        + q(
            f'/pipeline.XXXXXX)" && export LIP_DATA=$PWD/data && python "$AIMM/examples/configure_pipeline.py" '
            f"{shlex.join(flags)} --out configs"
        ),
        q("  cp -- ") + '"$p"' + q(f"/{{{names}}} configs/"),
        q(
            "  # not copied: domain-filled.yaml and domain-check/ (filled by fill()) and commands.sh (written for its"
            " run directory): redo a hand edit of commands.sh, e.g. of fill(), in configs/commands.sh"
        ),
        q("  export ") + '"$vars"' + q(" && bash configs/commands.sh"),
        q(
            "or (b) this run with a fresh data root and bundle directory (its configs as they are)."
            + keep
            + " LIP_DATA=<new dir> OUT=<new hub dir> "
        )
        + '"$vars"'
        + q(" bash ")
        + '"$p"'
        + q("/commands.sh"),
    ]
    return [
        "refused() { local here p vars",
        '  here=$(printf %q "$(dirname "$PWD")") p=$(printf %q "$P")',
        '  vars="FASTA=$(printf %q "$(realpath "$FASTA")") ID=$(printf %q "$ID")"',
        '  if [ -n "${LICENSE:-}" ]; then vars+=" LICENSE=$(printf %q "$LICENSE")"; fi',
        '  if [ -n "${DEMO:-}" ]; then vars+=" DEMO=$(printf %q "$DEMO")"; fi',
        "  printf '%s\\n' " + " \\\n    ".join(words) + " >&2",
        "  exit 1; }",
    ]


def _preflight(plugins: list[str], py_enc: str) -> list[str]:
    """Return ``commands.sh``'s checks before step 1: the recipe's plugins import, step 12's FASTA exists."""
    lines = []
    if any(m.startswith("examples.") for m in plugins):  # the example plugins live in the checkout's examples/
        lines.append('export PYTHONPATH="$AIMM${PYTHONPATH:+:$PYTHONPATH}"  # for the recipe\'s examples.* plugins')
    if plugins:
        mods = " ".join(shlex.quote(m) for m in plugins)
        lines.append(
            f"{py_enc} -c 'import importlib, sys; [importlib.import_module(m) for m in sys.argv[1:]]' {mods} "
            "2>/dev/null || { echo \"commands.sh: the recipe's plugins"
            f" ({', '.join(plugins)})"
            f" do not import (python -c 'import {plugins[0]}' shows why): put their package on PYTHONPATH\" >&2;"
            " exit 1; }"
        )
    seq = "MPKRKAEGDAKGDKAKVKDEPQRRSARLSAKPAPPKPEPKPKKAPAKKGEKVPKGKKGKADAGKEGNNPAENGDAKTDQAQKAEGAGDAK"
    lines += [
        '[ -s "$FASTA" ] || { echo "commands.sh: step 12 embeds sequence $ID of FASTA=$FASTA, which is missing;'
        ' for the example protein (P05204, HMG-17), run first:" >&2;',
        f"  echo \"  printf '>P05204\\n{seq}\\n' > $FASTA\" >&2; exit 1; }}",
    ]
    return lines


class NotUsableError(ValueError):
    """``--inspect`` of a causal LM that yields no usable receiver config: the report so far and the reason."""

    def __init__(self, lines: list[str], reason: str) -> None:
        super().__init__(reason)
        self.lines, self.reason = lines, reason


def inspect(model_id: str, revision: str | None = None, placeholder: str = " X") -> str:
    """Return the ``--inspect`` report for a Hugging Face model (module docstring) as YAML text with comments.

    Raises:
        NotUsableError: for a causal LM whose placeholder is not one token or whose decoder layers are not found, and
            for a model named or configured as one (architectures with a :data:`CAUSAL_MARKERS` name, or a decoder or
            encoder-decoder config) that ``AutoModelForCausalLM`` cannot build.
    """
    import torch  # noqa: PLC0415
    from huggingface_hub import HfApi  # noqa: PLC0415
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    local = Path(model_id).is_dir()  # a local model directory has no hub revision: give --revision yourself
    sha = (revision or "local-directory") if local else HfApi().model_info(model_id, revision=revision or "main").sha
    kw = {} if local else {"revision": sha}
    config = AutoConfig.from_pretrained(model_id, **kw)
    text = config.get_text_config()
    depth, width = int(text.num_hidden_layers), int(text.hidden_size)
    lines = [f"# {model_id}: revision {sha} ({revision or 'main'}), num_hidden_layers {depth}, hidden_size {width}"]
    as_encoder = (
        f"# as an encoder recipe: revision: {sha}, read: [{depth}] (the final layer; hidden_states index {depth})"
    )
    archs = [str(a) for a in getattr(config, "architectures", None) or ()]
    encoder_only = not getattr(config, "is_decoder", False) and not getattr(config, "is_encoder_decoder", False)
    if archs and not any(m in a for a in archs for m in CAUSAL_MARKERS):  # e.g. BertForMaskedLM
        lines += [
            as_encoder,
            f"# {', '.join(archs)}: not a causal LM: no receiver config printed (use it as an encoder)",
        ]
        return "\n".join(lines) + "\n"
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
    except (ValueError, KeyError) as e:
        if archs or not encoder_only:  # e.g. LlavaForConditionalGeneration, T5ForConditionalGeneration
            what = ", ".join(archs) if archs else type(config).__name__
            raise NotUsableError(
                lines,
                f"{what}: AutoModelForCausalLM cannot build it ({_first_line(e)}): not a receiver for this package",
            ) from e
        lines += [
            as_encoder,
            f"# {type(config).__name__}: not a causal LM: no receiver config printed (use it as an encoder)",
        ]
        return "\n".join(lines) + "\n"
    name = type(model).__name__
    if not archs and encoder_only and not any(m in name for m in CAUSAL_MARKERS):  # e.g. BertLMHeadModel
        lines += [as_encoder, f"# {name}: not a causal LM: no receiver config printed (use it as an encoder)"]
        return "\n".join(lines) + "\n"
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else None
    path = next(
        (
            n
            for n, m in model.named_modules()
            if isinstance(m, torch.nn.ModuleList)
            and len(m) == depth
            and decoder is not None
            and any(m is d for d in decoder.modules())
        ),
        None,
    )
    if path is None:
        raise NotUsableError(lines, "get_decoder() names no decoder layer list: the hf backend refuses this model (§4)")
    tok = AutoTokenizer.from_pretrained(model_id, **kw)

    def encode(text: str) -> list[int]:  # as the renderer encodes the placeholder: on its own, no special tokens
        return list(tok(text, add_special_tokens=False)["input_ids"])

    ids = encode(placeholder)
    if len(ids) != 1:
        good = [(c, encode(c)[0]) for c in PLACEHOLDER_CANDIDATES if c != placeholder and len(encode(c)) == 1]
        hint = (
            f"try --placeholder {shlex.quote(good[0][0])} (one token, id {good[0][1]}); one token: "
            + ", ".join(f"{c!r}" for c, _ in good)
            if good
            else f"none of {', '.join(map(repr, PLACEHOLDER_CANDIDATES))} is one token: try --placeholder with another"
        )
        raise NotUsableError(lines, f"placeholder {placeholder!r} encodes to {len(ids)} tokens {ids}, not one; {hint}")
    lines.append(f"# placeholder {placeholder!r}: one token (id {ids[0]})")
    lines.append(f"# auto write layers: one token {auto_layers(depth)}, three staggered {auto_layers(depth, 3)}")
    spec = {
        "name": model_id.rstrip("/").split("/")[-1].lower(),
        "model_id": model_id,
        "revision": sha,
        "depth": depth,
        "d_model": width,
        "placeholder": placeholder,
        "layers_path": path,
        "backends": {"score": "hf", "train": "hf", "generate": "hf", "capture": "hf"},
    }
    body = yaml.safe_dump(spec, sort_keys=False, default_flow_style=None, allow_unicode=True, width=120)
    return "\n".join(lines) + "\n" + body


def main(argv: Sequence[str] | None = None) -> int:
    """The command line (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inspect", metavar="HF_MODEL_ID", help="print a model's revision, depth, width and more")
    parser.add_argument("--revision", help="with --inspect: the branch, tag or commit to resolve (default main)")
    parser.add_argument("--placeholder", default=" X", help="with --inspect: the slot token text to check")
    parser.add_argument("--receiver", type=Path, help="the receiver config (a ReceiverSpec YAML)")
    parser.add_argument("--encoder", type=Path, help="the encoder recipe (an EncoderRecipe YAML)")
    parser.add_argument("--kind", choices=KINDS, help="the connector kind")
    parser.add_argument("--out", type=Path, help="the new directory to write the configs into")
    parser.add_argument("--layers", default="auto", help="auto (default), L,L,... or one list per token split by /")
    parser.add_argument("--name", help="the connector's name (default <recipe>-<kind>; its bundle <receiver>__<name>)")
    parser.add_argument("--pin-prefix", help="the receiver's and run's pin prefix (default mine/<receiver>)")
    parser.add_argument("--d-s", type=int, help="the stream width of mlp and xattn3 (default min(256, d_model))")
    parser.add_argument(
        "--data-fasta", type=Path, help="fill()'s training FASTA (default examples/data/proteins.fasta)"
    )
    parser.add_argument("--data-tsv", type=Path, help="fill()'s annotation table (default examples/data/proteins.tsv)")
    parser.add_argument(
        "--verify-pin",
        nargs=3,
        metavar=("STAGE", "CONFIG", "PIN"),
        help=f"commands.sh's check: exit 0 if PIN holds this config's record, {ABSENT} if unpinned, else 1",
    )
    parser.add_argument("--pinned", metavar="PIN", help="commands.sh's check: exit 0 if any name of PIN is pinned")
    args = parser.parse_args(argv)
    if args.verify_pin:
        stage, config, pin = args.verify_pin
        return verify_pin(stage, Path(config), pin)
    if args.pinned:
        from latent_interconnect.workspace import Workspace  # noqa: PLC0415

        return 0 if pinned_names(list(Workspace.from_env().pins()), args.pinned) else 1
    if args.inspect:
        try:
            report = inspect(args.inspect, args.revision, args.placeholder)
        except NotUsableError as e:
            sys.stderr.write("\n".join([*e.lines, f"configure_pipeline.py --inspect: no receiver config: {e.reason}"]))
            sys.stderr.write("\n")
            return NOT_USABLE
        sys.stdout.write(report)
        return 0
    missing = [f"--{k}" for k in ("receiver", "encoder", "kind", "out") if getattr(args, k) is None]
    if missing:
        parser.error(f"give {', '.join(missing)} (or --inspect)")
    try:
        files = configure(
            args.receiver,
            args.encoder,
            args.kind,
            args.out,
            layers=args.layers,
            name=args.name,
            pin_prefix=args.pin_prefix,
            d_s=args.d_s,
            data_fasta=args.data_fasta,
            data_tsv=args.data_tsv,
        )
    except (ValueError, FileExistsError) as e:
        parser.error(str(e))
    connector = _load(files["connector.yaml"])
    sys.stdout.write(
        f"wrote {len(files)} files to {args.out}: connector {connector['name']} ({connector['kind']}), "
        f"write layers {connector['layers']['per_token']}\nrun: bash {args.out / 'commands.sh'}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
