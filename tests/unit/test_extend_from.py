"""The kept final state and ``--option extend_from=S`` of the train stage (``run.train``, ``training.trainer``).

On CPU, through the train stage's ``main`` on the tiny Llama (``hf`` backend), with a fixed clock and fixed provenance
(:func:`fixed`: the commit, host, version, argv and times are constants, so two runs of one config write the same
bytes). Checked:

- a fresh run without the option writes the records, outputs, log and run files the base code wrote, plus its kept
  final state: against the same run with the final save patched to the 1.0.0 behaviour (in every environment), and
  against frozen goldens (:data:`GOLDEN`, in the release lock's library versions);
- extending a finished 4-step run to 6 steps gives the straight 6-step run's checkpoints, log and outputs, with the
  source's lineage in the record and its run directory untouched; an extension from an off-cadence step adds that
  step's checkpoint and may run with another input cache (a permitted execution difference);
- each refusal, which writes nothing to either run directory: S out of range, a schedule that reads the total, no
  source record (another optimizer setting included), a source that is not finished or has no kept state, an existing
  run directory, another source for an extended one, other derived data, another execution, a source log artifact
  that is missing or unreadable;
- a killed initialization leaves no run directory and the next one removes its staging; a job stopped after it is
  resumed without the option and keeps its lineage, and so does its publication by the ``achieved`` stage;
- the origin is read again under the run directory's lock: an extension another producer publishes between a job's
  first read and its lock keeps its lineage in that job's record, and an origin that changed meanwhile is refused;
- a failed final save leaves no ``done.json``, and the relaunch completes the run as an uninterrupted one;
- provenance without ``extended_from`` is stored as before, with it round-trips, and a newer reader's fields load.
"""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import transformers
from tests.fakes import (
    TOY_OBJECTS,
    ToyDomain,
    build_tiny_llama,
    commit_toy_domain,
    commit_toy_embeddings,
    make_workspace,
    toy_recipe,
)

import latent_interconnect.provenance as provenance_module
import latent_interconnect.training.trainer as trainer_module
from latent_interconnect.artifacts.records import Record, RecordRef, RecordStore, _StoredRecord
from latent_interconnect.artifacts.store import MANIFEST_NAME
from latent_interconnect.build.zero_shot import ScreenSpec, TrainingSetSpec, build_training_set
from latent_interconnect.canonical import canonical_json
from latent_interconnect.config import dump_config
from latent_interconnect.data import MixtureEntry
from latent_interconnect.data.items import commit_item_set, object_list
from latent_interconnect.data.sampler import Sampler
from latent_interconnect.domains.base import DomainSpec
from latent_interconnect.provenance import Extension, Provenance, Segment
from latent_interconnect.run import achieved as achieved_stage
from latent_interconnect.run import train as train_stage
from latent_interconnect.training import RunConfig
from latent_interconnect.training.config import AchievedSpec, CheckpointSpec, OptimizerSpec
from latent_interconnect.training.trainer import (
    DONE_FILE,
    EXTENSION_FILE,
    EXTENSION_STAGING,
    LOCK_FILE,
    LOG_FILE,
    Trainer,
)
from latent_interconnect.writes.checkpoint import RESUME_FILE, clear_resume_state, read_resume_summary
from latent_interconnect.writes.connector import ConnectorSpec

TRAIN = tuple(sorted(TOY_OBJECTS)[:6])
RECEIVER_DIR = "tiny-llama"
"""The tiny Llama's directory, relative to the world root (the tests run there), so no config holds a temporary path."""
CONNECTOR = {
    "name": "toy-linear",
    "receiver": "tiny-llama",
    "input": {"recipe": "fake-toy", "layers": [0]},
    "tokens": [{"target": "name"}, {"target": "name", "noise": 0.03}],
    "layers": {"rule": "explicit", "per_token": [[1, 2], [1, 3]]},
    "kind": "affine",
    "alpha": {"first": 1.0, "init": 0.4, "budget": 0.5, "penalty_k": 5.0},
    "layer_choice": {"note": "unit test"},
}
YES_NO = {
    "templates": [{"id": "y1", "user": "Here is a protein:<SLOT:x>. {question} Answer Yes or No."}],
    "questions": {"has_c": ["Does it contain cysteine?"]},
    "sampling": "uniform",
    "answers": {"yes": "Y", "no": "N"},
    "weights": {"family": "single_token", "floor": 0.1},
}
TEXT = {
    "templates": [{"id": "d1", "task": "describe", "user": "Describe<SLOT:x>."}],
    "sampling": "cluster_sqrt",
    "max_target_tokens": 8,
    "weights": {"describe": {"family": "power", "floor": 0.1, "tau": 0.001, "p": 1.2}},
}
CONSTANT = {
    "lr_rel": 0.05,
    "lr_alpha": 0.05,
    "warmup_steps": 2,
    "schedule": "constant",
    "weight_decay": 0.01,
    "clip": 1.0,
}
COSINE = {**CONSTANT, "schedule": "cosine", "lr_floor": 0.1}
GOLDEN = {
    "computation_key": "47e74c3c94bb57945906e78f92eadb24f024ae1da5bc423dda5042b8c555930f",
    "record_id": "d5abf44a587f72b4b8ba05c6384435a2fcc91f1c9754e84815fb559b2ae3b510",
    "record_file": "431e0de76e9cd93924b6325e37b17a77f14e02da08dbe4e7278dafd8caca58db",
    "outputs": {
        "log": "b68e83b9563047609b211c42b879c5acd4b2bd718c11388c100684dd6f56423a",
        "step_000000": "47fe68712e6a16e33c53e8b375fcf83e02c706fa42a5f9bf66123ad4585689fe",
        "step_000002": "77a1b9158bd8d23725cde167a2bac35075f0d7bca4c09c14dbf8aa38ac668d17",
        "step_000004": "527a0957bb086e2c77c57f10995b7c805239e6b07e3d5bb8e2c673bc70ae2c0d",
        "step_000006": "3efc49d8cb7319119d701009964e8d210418d62c2ef93a78a775788b05f4b435",
    },
    "run_files": {
        "derived.json": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        "done.json": "83087982d764daec0986f0d4ff6e64d5751c43da05485c1f17e16f0d64bf9af8",
        "log.jsonl": "3f4bfce77d9282ba8d2af0d71b7c4d23f4f3fa6639d349d312a6d23684bd8303",
        "segments.json": "0058d3fa778171da7fa30a06e08e486e1761475f9b2eb55222675195344e9c0c",
    },
}
"""A fresh 6-step run (:func:`config`, job 1, :func:`fixed`) as the 1.0.0 code (``33e9909``) wrote it, before the
final state was kept: its computation key, record id, record file, outputs and run files (sha256 of their bytes; the
lock file aside, which holds the host and process)."""
GOLDEN_ENV = {"torch": "2.11.0+cu128", "numpy": "2.5.2", "transformers": "4.57.6"}
"""The library versions :data:`GOLDEN` was computed with: the weights, so every digest, depend on them."""


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The world root (the tiny Llama in :data:`RECEIVER_DIR`), its workspace and a run's fields but steps, optimizer.

    The workspace holds the toy domain, its embeddings, a screened training set and a ridge fit; each test gives the
    stage its own records and run directories (:func:`stage`) over these artifacts.
    """
    root = tmp_path_factory.mktemp("world")
    ws = make_workspace(root / "data")
    store = ws.artifacts
    domain_ref = commit_toy_domain(store)
    receiver = build_tiny_llama(root / RECEIVER_DIR).model_copy(update={"model_id": RECEIVER_DIR})
    ids = commit_item_set(store, object_list("train", ToyDomain(store.path(domain_ref)), TRAIN))
    screen = ScreenSpec(
        fields=("description",),
        property_list_fields=(),
        zero_shot_tasks=("long_name",),
        detectors=({"task": "long_name", "detector": "lexical", "params": {"terms": [r"\blong-named\b"]}},),
    )
    domain = DomainSpec(name="fake-toy", data=domain_ref.sha256)
    text_ref = build_training_set(TrainingSetSpec(domain=domain, train_ids=ids.sha256, screen=screen), store)
    connector = ConnectorSpec.model_validate(CONNECTOR)
    rng = np.random.default_rng(0)
    staged = store.staging("unit-ridge")
    np.save(staged / "mu.npy", np.zeros(8, np.float32))
    np.save(staged / "sd.npy", np.ones(8, np.float32))
    for layer in range(4):
        np.save(staged / f"W.L{layer}.npy", (0.1 * rng.normal(size=(64, 8))).astype(np.float32))
        np.save(staged / f"b.L{layer}.npy", rng.normal(size=64).astype(np.float32))
    bound = {
        "input": connector.input.identity(),
        "recipe_digest": toy_recipe().digest(),
        "receiver": {"model_id": receiver.model_id, "revision": receiver.revision},
    }
    ridge = store.commit(staged, "ridge_init", {"target": "name", "layers": [0, 1, 2, 3]} | bound)
    base = {
        "name": "unit",
        "connector": connector,
        "receiver": receiver,
        "encoder": toy_recipe(),
        "domain": domain,
        "ridge": {"name": ridge.sha256},
        "embeddings": (commit_toy_embeddings(store, {i: v for i, (v, _) in TOY_OBJECTS.items()}).sha256,),
        "training_set": text_ref.sha256,
        "mixture": (
            MixtureEntry(stream="yes_no", share=0.75, params=YES_NO),
            MixtureEntry(stream="text", share=0.25, params=TEXT),
        ),
        "batch": 4,
        "device_class": "cpu",
        "checkpoints": CheckpointSpec(resume_every=3),
        "checkpoint_every": 2,
    }
    return root, ws, base


def config(world, steps: int, optimizer: dict = CONSTANT) -> RunConfig:
    return RunConfig.model_validate({**world[2], "steps": steps, "optimizer": OptimizerSpec(**optimizer)})


@pytest.fixture
def fixed(monkeypatch):
    """Fix what a record and a log take from the process: the clock, the commit, the host, the version, the argv."""
    clock = SimpleNamespace(perf_counter=lambda: 0.0, strftime=time.strftime, gmtime=time.gmtime)
    monkeypatch.setattr(trainer_module, "time", clock)
    monkeypatch.setattr(provenance_module, "_now", lambda: "2026-01-01T00:00:00+00:00")
    monkeypatch.setattr(provenance_module, "code_state", lambda repo: ("f" * 40, False))
    monkeypatch.setattr(provenance_module, "__version__", "0.0.0")
    monkeypatch.setattr(
        provenance_module, "platform", SimpleNamespace(node=lambda: "host", python_version=lambda: "3.0.0")
    )
    monkeypatch.setattr(sys, "argv", ["train"])


def stage(
    world,
    here: Path,
    monkeypatch,
    cfg: RunConfig | AchievedSpec,
    *options: str,
    job: int = 1,
    main=train_stage.main,
    artifacts: Path | None = None,
) -> int:
    """Run a stage's ``main`` (the train stage's) as job ``job`` on ``cfg`` with records and run directories under
    ``here``, over the world's artifact store or ``artifacts``.

    Returns:
        The exit status (a refusal's ``SystemExit`` code included).
    """
    root, ws, _ = world
    monkeypatch.chdir(root)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("LIP_JOB_ID", str(job))
    monkeypatch.setenv("LIP_DATA", str(here / "data"))
    monkeypatch.setenv("LIP_ARTIFACTS", str(artifacts or ws.artifacts.root))
    monkeypatch.setenv("LIP_RECORDS", str(here / "records"))
    monkeypatch.setenv("LIP_RUNS", str(here / "runs"))
    here.mkdir(parents=True, exist_ok=True)
    path = here / f"run-{cfg.digest()[:12]}.yaml"
    dump_config(cfg, path)
    argv = ["--config", str(path)] + [x for o in options for x in ("--option", o)]
    try:
        return main(argv)
    except SystemExit as exit_:
        return int(exit_.code or 0)


def records(here: Path, cfg: RunConfig) -> list[Record]:
    """The train records of ``cfg`` under ``here``, by job."""
    found = [r for r in RecordStore(here / "records").find("train") if r.config_sha == cfg.digest()]
    return sorted(found, key=lambda r: r.provenance.job_id or "")


def run_dir(here: Path, cfg: RunConfig) -> Path:
    return here / "runs" / cfg.digest()


def outputs(record: Record) -> dict[str, str]:
    return {k: v.sha256 for k, v in record.outputs.items()}


def files(directory: Path) -> dict[str, tuple[bytes, int, int]]:
    """Every file's bytes, inode and modification time (a rewrite with the same bytes still shows), the lock aside.

    The lock file is every job's on the directory: its holder writes its host and process into it.
    """
    if not directory.exists():
        return {}
    return {
        p.name: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
        for p in sorted(directory.iterdir())
        if p.name != LOCK_FILE
    }


def train_lines(directory: Path) -> list[dict]:
    lines = [json.loads(x) for x in (directory / LOG_FILE).read_text().splitlines()]
    return [x for x in lines if x["event"] == "train"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_a_fresh_run_writes_what_the_base_code_wrote_and_keeps_its_final_state(world, tmp_path, monkeypatch, fixed):
    """The record (its file's bytes), the outputs and the run files of a fresh run are the base code's, and the run
    directory also keeps the resume state of its last step, beside a done file that names the same checkpoints."""
    env = {"torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__}
    if env != GOLDEN_ENV:
        pytest.skip(f"the goldens were computed with {GOLDEN_ENV}, this environment has {env}")
    cfg = config(world, 6)
    assert stage(world, tmp_path, monkeypatch, cfg) == 0
    (record,) = records(tmp_path, cfg)
    stored = tmp_path / "records" / record.computation_key() / f"{record.record_id()}.json"
    assert (record.computation_key(), record.record_id()) == (GOLDEN["computation_key"], GOLDEN["record_id"])
    assert sha(stored.read_bytes()) == GOLDEN["record_file"]
    assert outputs(record) == GOLDEN["outputs"]
    run = run_dir(tmp_path, cfg)
    kept = {n: sha(b) for n, (b, _, _) in files(run).items() if n != RESUME_FILE}
    assert kept == GOLDEN["run_files"] and sorted(files(run)) == sorted([*GOLDEN["run_files"], RESUME_FILE])
    summary = read_resume_summary(run)
    assert summary is not None and summary.step == 6
    assert summary.extra["committed"] == json.loads((run / DONE_FILE).read_text())["committed"]


def test_a_fresh_run_writes_what_the_1_0_0_final_save_wrote(world, tmp_path, monkeypatch, fixed):
    """In any environment: a fresh run's records, outputs and run files, apart from ``resume.pt``, are byte for byte
    those of the same run with the final save as 1.0.0 made it (no kept state: the resume state deleted at completion,
    ``done.json`` written in place)."""
    cfg = config(world, 6)
    kept, base = tmp_path / "kept", tmp_path / "base"
    assert stage(world, kept, monkeypatch, cfg) == 0
    replace = trainer_module._replace

    def done_as_in_1_0_0(path: Path, text: str) -> None:
        if path.name != DONE_FILE:
            replace(path, text)
            return
        path.write_text(text)
        clear_resume_state(path.parent)

    monkeypatch.setattr(Trainer, "_keep", lambda self, optimizer, step, committed: None)
    monkeypatch.setattr(trainer_module, "_replace", done_as_in_1_0_0)
    assert stage(world, base, monkeypatch, cfg) == 0
    (new,), (old,) = records(kept, cfg), records(base, cfg)
    assert outputs(new) == outputs(old)
    written = {here: {f: (here / f).read_bytes() for f in stored(here)} for here in (kept, base)}
    assert written[kept] == written[base] and len(written[kept]) == 1
    new_files = {n: b for n, (b, _, _) in files(run_dir(kept, cfg)).items()}
    old_files = {n: b for n, (b, _, _) in files(run_dir(base, cfg)).items()}
    assert RESUME_FILE not in old_files and read_resume_summary(run_dir(kept, cfg)).step == 6
    assert {n: b for n, b in new_files.items() if n != RESUME_FILE} == old_files


def test_extending_4_to_6_steps_is_the_straight_6_step_run(world, tmp_path, monkeypatch, fixed):
    """On the deterministic CPU backend the extension's checkpoints, log and outputs are the straight run's; the record
    names its source and keeps its jobs apart; the source's run directory keeps every byte."""
    short, long = config(world, 4), config(world, 6)
    assert stage(world, tmp_path, monkeypatch, short, job=1) == 0
    (source,) = records(tmp_path, short)
    before = files(run_dir(tmp_path, short))
    assert stage(world, tmp_path, monkeypatch, long, "extend_from=4", job=2) == 0
    straight = tmp_path / "straight"
    assert stage(world, straight, monkeypatch, long, job=3) == 0
    (extended,), (plain,) = records(tmp_path, long), records(straight, long)
    assert outputs(extended) == outputs(plain)  # every checkpoint and the log artifact
    assert extended.computation_key() == plain.computation_key()  # another execution of the same computation
    run = run_dir(tmp_path, long)
    assert (run / LOG_FILE).read_bytes() == (run_dir(straight, long) / LOG_FILE).read_bytes()
    assert files(run_dir(tmp_path, short)) == before  # the source is untouched
    origin = Extension(
        config_sha=short.digest(),
        computation_key=source.computation_key(),
        record_id=source.record_id(),
        step=4,
        segments=1,
    )
    prov = extended.provenance
    assert prov.extended_from == origin and json.loads((run / EXTENSION_FILE).read_text()) == origin.model_dump()
    assert extended.input_records == {"extended_from": source.computation_key()}
    assert prov.input_executions == {"extended_from": source.record_id()}
    assert prov.execution["option.extend_from"] == "4"
    assert [s.job_id for s in prov.inherited_segments()] == ["1"]
    assert [s.job_id for s in prov.new_segments()] == ["2"]
    assert [s.job_id for s in prov.segments()] == ["1", "2"]
    assert plain.provenance.extended_from is None and plain.provenance.inherited_segments() == ()
    assert read_resume_summary(run).step == 6


def test_an_off_cadence_source_adds_its_checkpoint_and_the_input_cache_may_change(world, tmp_path, monkeypatch, fixed):
    """From step 3 (checkpoints every 2): the outputs add ``step_000003``, the rest are the straight run's; the source
    ran with the input cache on, the extension off (an execution difference a resume permits)."""
    short, long = config(world, 3), config(world, 6)
    assert stage(world, tmp_path, monkeypatch, short, job=1) == 0
    assert stage(world, tmp_path, monkeypatch, long, "extend_from=3", "input_cache=off", job=2) == 0
    straight = tmp_path / "straight"
    assert stage(world, straight, monkeypatch, long, job=3) == 0
    extended, plain = outputs(records(tmp_path, long)[0]), outputs(records(straight, long)[0])
    assert sorted(extended) == ["log", "step_000000", "step_000002", "step_000003", "step_000004", "step_000006"]
    assert {k: v for k, v in extended.items() if k not in ("log", "step_000003")} == {
        k: v for k, v in plain.items() if k != "log"
    }
    assert train_lines(run_dir(tmp_path, long)) == train_lines(run_dir(straight, long))


def stored(here: Path) -> list[str]:
    """The record files under ``here``."""
    return sorted(str(p.relative_to(here)) for p in (here / "records").glob("*/*.json"))


def refused(world, here: Path, monkeypatch, capfd, cfg: RunConfig, *options: str, match: str, **kwargs) -> None:
    """Run a job that must exit ``EXTEND_REFUSED`` naming ``match``, writing nothing to any run directory or record."""
    runs = here / "runs"
    before = ({d.name: files(d) for d in runs.iterdir()} if runs.exists() else {}), stored(here)
    capfd.readouterr()
    assert stage(world, here, monkeypatch, cfg, *options, job=9, **kwargs) == train_stage.EXTEND_REFUSED
    err = capfd.readouterr().err
    assert match in err, err
    after = ({d.name: files(d) for d in runs.iterdir()} if runs.exists() else {}), stored(here)
    assert after == before


@pytest.fixture
def finished(world, tmp_path, monkeypatch, fixed):
    """A finished 4-step run (job 1) under ``tmp_path``; returns its config."""
    short = config(world, 4)
    assert stage(world, tmp_path, monkeypatch, short, job=1) == 0
    return short


@pytest.mark.parametrize("step", ["0", "6", "7"])
def test_a_step_outside_1_to_steps_is_refused(world, tmp_path, monkeypatch, capfd, finished, step):
    refused(world, tmp_path, monkeypatch, capfd, config(world, 6), f"extend_from={step}", match="is not in 1..5")


def test_a_schedule_that_reads_the_total_is_refused(world, tmp_path, monkeypatch, capfd, fixed):
    assert OptimizerSpec(**COSINE).depends_on_total() and not OptimizerSpec(**CONSTANT).depends_on_total()
    constant = OptimizerSpec(**CONSTANT)
    assert all(constant.factor(s, 4) == constant.factor(s, 6) for s in range(4))
    cfg = config(world, 6, COSINE)
    refused(world, tmp_path, monkeypatch, capfd, cfg, "extend_from=4", match="cosine schedule reads the total steps")


def test_no_source_record_is_refused(world, tmp_path, monkeypatch, capfd, fixed):
    refused(world, tmp_path, monkeypatch, capfd, config(world, 6), "extend_from=4", match="0 train records")


def test_another_optimizer_setting_finds_no_source(world, tmp_path, monkeypatch, capfd, finished):
    """A source config differs in nothing but steps: another weight decay (as one set from the run's length) has no
    finished run to continue."""
    cfg = config(world, 6, {**CONSTANT, "weight_decay": 0.02})
    refused(world, tmp_path, monkeypatch, capfd, cfg, "extend_from=4", match="0 train records")


@pytest.mark.parametrize(
    ("name", "match"), [(DONE_FILE, "does not record a completed run"), (RESUME_FILE, "holds no kept final state")]
)
def test_a_source_not_finished_or_without_its_kept_state_is_refused(
    world, tmp_path, monkeypatch, capfd, finished, name, match
):
    """Without its done file the source is not finished; without its resume state (a run completed by code that did
    not keep it) there is nothing to continue. The file is moved aside, not deleted."""
    source = run_dir(tmp_path, finished)
    (source / name).rename(tmp_path / name)
    refused(world, tmp_path, monkeypatch, capfd, config(world, 6), "extend_from=4", match=match)


def test_a_source_whose_files_disagree_with_its_record_is_refused(world, tmp_path, monkeypatch, capfd, finished):
    log = run_dir(tmp_path, finished) / LOG_FILE
    log.write_text(log.read_text() + json.dumps({"event": "note", "step": 4}) + "\n")
    refused(world, tmp_path, monkeypatch, capfd, config(world, 6), "extend_from=4", match="the source record's log")


@pytest.mark.parametrize("damage", ["missing", "unparsable", "altered"])
def test_an_unreadable_source_log_artifact_is_refused(world, tmp_path, monkeypatch, capfd, fixed, damage):
    """The source's committed log artifact is moved aside (FileNotFoundError), its manifest made unparsable
    (ValueError) or altered (IntegrityError): a refusal, not a traceback. On a copy of the artifact store, so the
    world's store, shared by the other tests, is untouched."""
    store = tmp_path / "artifacts"
    shutil.copytree(world[1].artifacts.root, store, symlinks=True)
    short = config(world, 4)
    assert stage(world, tmp_path, monkeypatch, short, job=1, artifacts=store) == 0
    (source,) = records(tmp_path, short)
    digest = source.outputs["log"].sha256
    directory = store / digest[:2] / digest
    for path in (directory.parent, directory, directory / MANIFEST_NAME):
        path.chmod(path.stat().st_mode | 0o200)
    manifest = directory / MANIFEST_NAME
    if damage == "missing":
        directory.rename(tmp_path / "aside")
    elif damage == "unparsable":
        manifest.write_text(manifest.read_text()[:-1])
    else:
        body = json.loads(manifest.read_text())
        body["meta"]["run"] = "another"
        manifest.write_text(json.dumps(body))
    cfg = config(world, 6)
    refused(world, tmp_path, monkeypatch, capfd, cfg, "extend_from=4", match="log artifact", artifacts=store)


def test_an_existing_run_directory_is_refused(world, tmp_path, monkeypatch, capfd, finished):
    long = config(world, 6)
    assert stage(world, tmp_path, monkeypatch, long, job=2) == 0  # the 6-step run trained fresh
    refused(world, tmp_path, monkeypatch, capfd, long, "extend_from=4", match="exists and is not an extension")


def test_an_extension_of_another_source_is_refused(world, tmp_path, monkeypatch, capfd, finished):
    long = config(world, 6)
    assert stage(world, tmp_path, monkeypatch, long, "extend_from=4", job=2) == 0
    refused(world, tmp_path, monkeypatch, capfd, long, "extend_from=3", match="extends the 4-step run")


def test_other_derived_data_is_refused(world, tmp_path, monkeypatch, capfd, finished):
    derived = Sampler.derived
    monkeypatch.setattr(Sampler, "derived", lambda self: {**derived(self), "changed": True})
    refused(world, tmp_path, monkeypatch, capfd, config(world, 6), "extend_from=4", match="the streams now derive")


def test_another_execution_is_refused(world, tmp_path, monkeypatch, capfd, finished):
    execution = Trainer.execution
    monkeypatch.setattr(Trainer, "execution", lambda self: {**execution(self), "dtype": "float16"})
    refused(world, tmp_path, monkeypatch, capfd, config(world, 6), "extend_from=4", match="differing: ['dtype']")


def test_a_continuation_never_starts_an_extension(world, tmp_path, monkeypatch, capfd, finished):
    long = config(world, 6)
    assert stage(world, tmp_path, monkeypatch, long, "extend_from=4", "resume_required=true", job=2) == 4
    assert not run_dir(tmp_path, long).exists()


def test_a_killed_initialization_leaves_no_run_directory_and_the_next_one_clears_its_staging(
    world, tmp_path, monkeypatch, finished
):
    """A failure while the source is copied removes the staging directory; one a killed job left (simulated here) is
    removed by the next initialization, which succeeds."""
    long = config(world, 6)
    before = files(run_dir(tmp_path, finished))
    stage_copy = Trainer._stage_extension

    def fails(self, directory, staging, found):
        stage_copy(self, directory, staging, found)
        raise RuntimeError("killed while copying")

    monkeypatch.setattr(Trainer, "_stage_extension", fails)
    with pytest.raises(RuntimeError, match="killed while copying"):
        stage(world, tmp_path, monkeypatch, long, "extend_from=4", job=2)
    runs = tmp_path / "runs"
    assert sorted(p.name for p in runs.iterdir()) == [finished.digest()]  # no run directory, no staging
    assert files(run_dir(tmp_path, finished)) == before
    monkeypatch.setattr(Trainer, "_stage_extension", stage_copy)
    left = runs / f"{EXTENSION_STAGING}{long.digest()}.{finished.digest()}.0123abcd"
    left.mkdir()
    (left / LOG_FILE).write_text("partial\n")
    assert stage(world, tmp_path, monkeypatch, long, "extend_from=4", job=3) == 0
    assert sorted(p.name for p in runs.iterdir()) == sorted([finished.digest(), long.digest()])
    assert records(tmp_path, long)[0].provenance.extended_from.segments == 1


def test_a_stopped_extension_resumes_without_the_option_and_keeps_its_lineage(world, tmp_path, monkeypatch, finished):
    """SIGUSR1 after update 5 of the extension: the relaunch without ``extend_from`` resumes it as any run, ends as the
    straight run, and its record names the source; a relaunch with it on the completed run records it again."""
    long = config(world, 6)
    update = Trainer._update

    def stop_after_update_5(self, step, optimizer):
        out = update(self, step, optimizer)
        if step == 4:
            self.stop.set()
        return out

    monkeypatch.setattr(Trainer, "_update", stop_after_update_5)
    assert stage(world, tmp_path, monkeypatch, long, "extend_from=4", job=2) == 1
    run = run_dir(tmp_path, long)
    assert not (run / DONE_FILE).exists() and read_resume_summary(run).step == 5 and (run / EXTENSION_FILE).exists()
    monkeypatch.setattr(Trainer, "_update", update)
    assert stage(world, tmp_path, monkeypatch, long, job=3) == 0
    straight = tmp_path / "straight"
    assert stage(world, straight, monkeypatch, long, job=4) == 0
    (resumed,), (plain,) = records(tmp_path, long), records(straight, long)
    (source,) = records(tmp_path, finished)
    checkpoints = {k: v for k, v in outputs(resumed).items() if k != "log"}
    assert checkpoints == {k: v for k, v in outputs(plain).items() if k != "log"}
    assert train_lines(run) == train_lines(run_dir(straight, long))
    prov = resumed.provenance
    assert prov.extended_from is not None and prov.extended_from.record_id == source.record_id()
    assert "option.extend_from" not in prov.execution and resumed.input_records == {
        "extended_from": source.computation_key()
    }
    assert [s.job_id for s in prov.inherited_segments()] == ["1"]
    assert [s.job_id for s in prov.new_segments()] == ["2", "3"]
    assert stage(world, tmp_path, monkeypatch, long, "extend_from=4", job=5) == 0  # complete: returns, records again
    assert [r.provenance.extended_from for r in records(tmp_path, long)] == [prov.extended_from] * 2


def test_an_extension_published_between_the_origin_read_and_the_lock_keeps_its_lineage(
    world, tmp_path, monkeypatch, finished
):
    """Another producer publishes an initialized extension after this job's first read of the origin (none: no run
    directory) and before it takes the run directory's lock: the job resumes that extension, and its record names the
    source, read again under the lock."""
    long = config(world, 6)
    update = Trainer._update

    def stop_after_update_5(self, step, optimizer):
        out = update(self, step, optimizer)
        if step == 4:
            self.stop.set()
        return out

    monkeypatch.setattr(Trainer, "_update", stop_after_update_5)
    assert stage(world, tmp_path, monkeypatch, long, "extend_from=4", job=2) == 1
    monkeypatch.setattr(Trainer, "_update", update)
    run = run_dir(tmp_path, long)
    elsewhere = run.with_name("initialized-elsewhere")
    run.rename(elsewhere)
    first_read = train_stage._extension

    def published_after_the_read(cfg, ws, workspace, extend_from):
        found = first_read(cfg, ws, workspace, extend_from)
        assert found == (None, None)
        elsewhere.rename(run)  # the other producer's atomic publication of the initialized extension
        return found

    monkeypatch.setattr(train_stage, "_extension", published_after_the_read)
    assert stage(world, tmp_path, monkeypatch, long, job=3) == 0
    (resumed,), (source,) = records(tmp_path, long), records(tmp_path, finished)
    assert (run / EXTENSION_FILE).exists()
    prov = resumed.provenance
    assert prov.extended_from is not None and prov.extended_from.record_id == source.record_id()
    assert resumed.input_records == {"extended_from": source.computation_key()}
    assert prov.input_executions == {"extended_from": source.record_id()}


def test_an_origin_that_changes_before_the_lock_is_refused(world, tmp_path, monkeypatch, capfd, finished):
    """An extension's origin read before the lock and another one found under it: the job is refused before training,
    and the run directory keeps what it held."""
    long = config(world, 6)
    update = Trainer._update

    def stop_after_update_5(self, step, optimizer):
        out = update(self, step, optimizer)
        if step == 4:
            self.stop.set()
        return out

    monkeypatch.setattr(Trainer, "_update", stop_after_update_5)
    assert stage(world, tmp_path, monkeypatch, long, "extend_from=4", job=2) == 1
    monkeypatch.setattr(Trainer, "_update", update)
    run = run_dir(tmp_path, long)
    origin_file = run / EXTENSION_FILE
    first_read = train_stage._extension

    def changed_after_the_read(cfg, ws, workspace, extend_from):
        found = first_read(cfg, ws, workspace, extend_from)
        origin = json.loads(origin_file.read_text())
        origin_file.write_text(json.dumps(origin | {"record_id": "0" * 64}))
        return found

    monkeypatch.setattr(train_stage, "_extension", changed_after_the_read)
    before = read_resume_summary(run).step
    assert stage(world, tmp_path, monkeypatch, long, job=3) == train_stage.EXTEND_REFUSED
    assert "changed" in capfd.readouterr().err
    assert read_resume_summary(run).step == before and not (run / DONE_FILE).exists()
    assert len(records(tmp_path, long)) == 0


def test_a_stopped_extension_published_by_the_achieved_stage_keeps_its_lineage(world, tmp_path, monkeypatch, finished):
    """The extension 4 -> 6 stopped by SIGUSR1 at step 5 and published by the ``achieved`` stage: the record names
    the source, lists its record among the input records and splits the source's job from its own."""
    long = config(world, 6)
    update = Trainer._update

    def stop_after_update_5(self, step, optimizer):
        out = update(self, step, optimizer)
        if step == 4:
            self.stop.set()
        return out

    monkeypatch.setattr(Trainer, "_update", stop_after_update_5)
    assert stage(world, tmp_path, monkeypatch, long, "extend_from=4", job=2) == 1
    assert read_resume_summary(run_dir(tmp_path, long)).step == 5
    monkeypatch.setattr(achieved_stage, "QUIET_S", 0)
    spec = AchievedSpec(run=long, device_class="cpu")
    assert stage(world, tmp_path, monkeypatch, spec, job=3, main=achieved_stage.main) == 0
    (published,) = RecordStore(tmp_path / "records").find("achieved")
    (source,) = records(tmp_path, finished)
    prov = published.provenance
    assert prov.extended_from == Extension(
        config_sha=finished.digest(),
        computation_key=source.computation_key(),
        record_id=source.record_id(),
        step=4,
        segments=1,
    )
    assert published.input_records == {"extended_from": source.computation_key()}
    assert prov.input_executions == {"extended_from": source.record_id()}
    assert prov.execution["achieved_step"] == "5"
    assert [s.job_id for s in prov.inherited_segments()] == ["1"]
    assert [s.job_id for s in prov.new_segments()] == ["2", "3"]


def test_a_failed_final_save_leaves_no_done_file_and_the_relaunch_completes(world, tmp_path, monkeypatch, fixed):
    """The final state's save fails: no ``done.json`` certifies the run, the state on disk is the periodic one of step
    3, and the relaunch resumes there and ends with the uninterrupted run's checkpoints and kept state."""
    cfg = config(world, 4)
    keep = Trainer._keep

    def fails(self, optimizer, step, committed):
        raise OSError("disk full")

    monkeypatch.setattr(Trainer, "_keep", fails)
    with pytest.raises(OSError, match="disk full"):
        stage(world, tmp_path, monkeypatch, cfg, job=1)
    run = run_dir(tmp_path, cfg)
    assert not (run / DONE_FILE).exists() and read_resume_summary(run).step == 3 and stored(tmp_path) == []
    monkeypatch.setattr(Trainer, "_keep", keep)
    assert stage(world, tmp_path, monkeypatch, cfg, job=2) == 0
    straight = tmp_path / "straight"
    assert stage(world, straight, monkeypatch, cfg, job=3) == 0
    assert {k: v for k, v in outputs(records(tmp_path, cfg)[0]).items() if k != "log"} == {
        k: v for k, v in outputs(records(straight, cfg)[0]).items() if k != "log"
    }
    assert (run / DONE_FILE).exists() and read_resume_summary(run).step == 4


SEGMENT = Segment(started="t0", ended="t1", host="h", job_id="1", commit="c" * 40, devices=())


def _record(extended_from: Extension | None) -> Record:
    provenance = Provenance(
        version="1",
        commit="c" * 40,
        dirty=False,
        argv=("train",),
        host="h",
        job_id="2",
        python="3",
        packages={},
        devices=(),
        started="t2",
        ended="t3",
        resumed=(SEGMENT,),
        extended_from=extended_from,
    )
    return Record(
        kind="train",
        purpose="result",
        config_sha="a" * 64,
        config={"name": "x"},
        outputs={"log": RecordRef(sha256="b" * 64, kind="train_log")},
        provenance=provenance,
    )


ORIGIN = Extension(config_sha="d" * 64, computation_key="e" * 64, record_id="f" * 64, step=4, segments=1)


def test_provenance_without_an_extension_is_stored_as_before_and_with_one_round_trips(tmp_path):
    """No ``extended_from`` key in the stored form of a record that extended nothing (so its bytes and id are the
    ones written before the field existed); one that sets it stores it, gets another id and loads back equal."""
    plain, extended = _record(None), _record(ORIGIN)
    assert "extended_from" not in plain.model_dump(mode="json")["provenance"]
    assert "extended_from" not in plain.provenance.identity()
    assert extended.record_id() != plain.record_id() and extended.computation_key() == plain.computation_key()
    store = RecordStore(tmp_path)
    for record in (plain, extended):
        assert store.get(store.put(record)) == record
    assert extended.provenance.inherited_segments() == (SEGMENT,)
    assert [s.started for s in extended.provenance.new_segments()] == ["t2"]
    assert plain.provenance.inherited_segments() == () and len(plain.provenance.new_segments()) == 2


def test_a_newer_field_of_the_extension_loads_and_keeps_its_id(tmp_path):
    """A record whose ``extended_from`` carries a field this code does not know (newer code) still loads, and the
    field is kept in its id, as for any newer provenance field."""
    body = _record(ORIGIN).model_dump(mode="json")
    body["provenance"]["extended_from"]["copied_bytes"] = 123
    text = canonical_json(body)
    record = _StoredRecord.model_validate_json(text)
    key, record_id = record.computation_key(), record.record_id()
    assert record_id != _record(ORIGIN).record_id()
    (tmp_path / key).mkdir()
    (tmp_path / key / f"{record_id}.json").write_text(text)
    loaded = RecordStore(tmp_path).get(record_id)
    assert loaded.record_id() == record_id and loaded.provenance.extended_from.model_extra == {"copied_bytes": 123}
