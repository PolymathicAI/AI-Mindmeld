"""The all-tasks runs: the multi-token MLP connectors trained on every task, on the released training sets' objects.

Checked on the configs, against the released pins (``test_train_from_release``) and the new training sets' pins:

- each training set (``configs/training_sets/{protein,molecule}-all-tasks.yaml``) holds no task out and takes the
  released training set's objects (``train_ids``);
- each all-tasks Yes/No stream is its source stream with the questions of every held-out and implied Yes/No task added
  (the tasks of the domain's transfer suite) and nothing else changed;
- each all-tasks mixture is its source mixture but for that stream;
- each Kimi K2.6 run is its multi-token MLP run but for name, destination, training set and mixture, and the
  molecule run for its 5,000 steps with the protein run's optimizer (its weight decay is for 5,000 steps); the
  protein run's destination is ``...-all-tasks-training`` and its publication's (``run.achieved``) the plain
  ``...-all-tasks``, as the released ESM-C 6B four-token MLP connector's; ``destination`` is not in the run digest;
- each Gemma 3 12B run is the ordinary Gemma 3 12B run of its read but for the connector (the Kimi run's, with every
  token at the ordinary connector's write layers, from its starts), the training set, the mixture, micro-batch 4, its
  names, and 5,000 steps with the Kimi protein run's optimizer (its weight decay is for 5,000 steps);
- only the named suites read an all-tasks connector, each at its released step (its access and transfer suites): none
  of its evaluations is zero-shot.
"""

import hashlib
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from latent_interconnect.build.zero_shot import TrainingSetSpec
from latent_interconnect.config import Pin, load_config, load_resolved, load_yaml
from latent_interconnect.config_index import cited_pins
from latent_interconnect.data.qa_streams import YesNoParams
from latent_interconnect.training import RunConfig
from latent_interconnect.training.config import AchievedSpec

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RELEASED_SETS = {
    "protein": "6314cf2ca50c589a229303a5991ec6e83b5fb76e871276857088d48543f89e77",
    "molecule": "feabbac061169488f31d19cf0f4a7a08a5bb7290709f91641022aa76057690e8",
}
"""The released training sets (data ``a27d7808``), whose objects the all-tasks sets take."""
KIMI = {"protein": "esmc-6b-mlp4-staggered", "molecule": "chemfm-3b-mlp4"}
"""The Kimi K2.6 multi-token MLP run trained on all tasks, by domain."""
GEMMA = {"protein": "esmc-6b", "molecule": "chemfm-3b"}
"""The ordinary Gemma 3 12B run (and connector) of the read, by domain."""
TRANSFER = {
    "protein": CONFIGS / "suite_specs" / "kimi-k2.6" / "protein-transfer.yaml",
    "molecule": CONFIGS / "suite_specs" / "kimi-k2.6" / "molecule-transfer-novel-scaffold.yaml",
}
"""The suite of each domain's held-out and implied Yes/No tasks."""


class _Pins(Mapping[str, Pin]):
    """The released training sets by id; any other name a stand-in (a hash of the name)."""

    def __getitem__(self, name: str) -> Pin:
        released = {f"{d}-training-set/training_set": s for d, s in RELEASED_SETS.items()}
        sha = released.get(name, hashlib.sha256(name.encode()).hexdigest())
        return Pin(target="record" if name.endswith("@record") else "artifact", sha256=sha)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str)

    def __iter__(self) -> Iterator[str]:  # the released names; non-empty, so the loader keeps this mapping
        return iter(f"{d}-training-set/training_set" for d in RELEASED_SETS)

    def __len__(self) -> int:
        return len(RELEASED_SETS)


def _resolved(path: Path):
    return load_resolved(path, unknown_pins="keep")


def _words(text: str) -> str:
    """``text`` with shell line continuations joined and every run of whitespace one space."""
    return " ".join(text.replace("\\\n", " ").split())


def _yes_no(mixture):
    (entry,) = [e for e in mixture if e["stream"] == "yes_no"]
    return entry["params"]


@pytest.mark.parametrize("domain", sorted(KIMI))
def test_the_training_set_holds_no_task_out_on_the_released_objects(domain):
    spec = load_config(CONFIGS / "training_sets" / f"{domain}-all-tasks.yaml", TrainingSetSpec, pins=_Pins())
    assert spec.all_tasks and spec.screen.zero_shot_tasks == () and spec.screen.detectors == ()
    assert spec.train_ids == RELEASED_SETS[domain] and spec.domain.name == domain
    assert spec.destination == f"{domain}-training-set-all-tasks"


@pytest.mark.parametrize("domain", sorted(KIMI))
@pytest.mark.parametrize("receiver", ["kimi-k2.6", "gemma-3-12b-it"])
def test_the_stream_adds_every_untrained_yes_no_task_and_nothing_else(domain, receiver):
    sub = "" if receiver == "kimi-k2.6" else f"{receiver}/"
    tail = "" if receiver == "kimi-k2.6" else "-answer-free"
    source = _resolved(CONFIGS / "streams" / f"{sub}yes-no-{domain}{tail}.yaml")
    mine = _resolved(CONFIGS / "streams" / f"{sub}yes-no-{domain}-all-tasks{tail}.yaml")
    assert {k: v for k, v in mine.items() if k != "questions"} == {k: v for k, v in source.items() if k != "questions"}
    assert {k: mine["questions"][k] for k in source["questions"]} == source["questions"]  # the source's, unchanged
    added = {k: v for k, v in mine["questions"].items() if k not in source["questions"]}
    assert set(added) == set(load_yaml(TRANSFER[domain].read_text())["tasks"])
    assert all(len(qs) == 2 and len(set(qs)) == 2 for qs in added.values())  # two wordings each, as the trained ones
    YesNoParams.model_validate(mine)


@pytest.mark.parametrize("domain", sorted(KIMI))
@pytest.mark.parametrize("sub", ["", "gemma-3-12b-it/"])
def test_the_mixture_is_its_source_but_for_the_yes_no_stream(domain, sub):
    source = _resolved(CONFIGS / "training" / f"{sub}mixture-{domain}.yaml")
    mine = _resolved(CONFIGS / "training" / f"{sub}mixture-{domain}-all-tasks.yaml")
    assert [(e["stream"], e["share"]) for e in mine] == [(e["stream"], e["share"]) for e in source]
    assert [e for e in mine if e["stream"] != "yes_no"] == [e for e in source if e["stream"] != "yes_no"]
    assert _yes_no(mine)["questions"] != _yes_no(source)["questions"]


@pytest.mark.parametrize("domain", sorted(KIMI))
def test_the_kimi_run_is_its_mlp_run_but_for_the_training_set_and_mixture(domain):
    run = KIMI[domain]
    mine = _resolved(CONFIGS / "runs" / "kimi-k2.6" / f"{run}-all-tasks.yaml")
    theirs = _resolved(CONFIGS / "runs" / "kimi-k2.6" / f"{run}.yaml")
    # the released runs' naming: the protein run's destination ends in -training (its publication takes the plain name)
    assert mine["name"] == f"{run}-all-tasks"
    assert mine["destination"] == theirs["destination"].replace(run, f"{run}-all-tasks")
    assert mine["destination"] == f"connectors/kimi-k2.6/{run}-all-tasks" + ("-training" if domain == "protein" else "")
    assert mine["training_set"] == {"$pin": f"{domain}-training-set-all-tasks/training_set"}
    assert mine["connector"] == theirs["connector"]  # the same connector, so the same step 0
    drop = ("name", "destination", "training_set", "mixture", "steps", "optimizer")
    assert {k: v for k, v in mine.items() if k not in drop} == {k: v for k, v in theirs.items() if k not in drop}
    protein = _resolved(CONFIGS / "runs" / "kimi-k2.6" / f"{KIMI['protein']}-all-tasks.yaml")
    assert mine["steps"] == 5000 and mine["optimizer"] == protein["optimizer"]
    load_config(CONFIGS / "runs" / "kimi-k2.6" / f"{run}-all-tasks.yaml", RunConfig, pins=_Pins())


@pytest.mark.parametrize("domain", sorted(KIMI))
def test_the_gemma_run_is_the_ordinary_run_with_the_mlp_connector_at_its_layers(domain):
    read, kimi = GEMMA[domain], KIMI[domain]
    path = CONFIGS / "runs" / "gemma-3-12b-it" / f"{read}-mlp4-all-tasks.yaml"
    mine, ordinary = _resolved(path), _resolved(CONFIGS / "runs" / "gemma-3-12b-it" / f"{read}.yaml")
    kimi_run = _resolved(CONFIGS / "runs" / "kimi-k2.6" / f"{kimi}-all-tasks.yaml")
    connector = _resolved(CONFIGS / "connectors" / "gemma-3-12b-it" / f"{read}.yaml")
    (layers,) = connector["layers"]["per_token"]
    mlp = mine["connector"]
    assert mlp["layers"] == {"rule": "explicit", "per_token": [layers] * 4}  # every token, not staggered
    assert mlp["alpha"] == connector["alpha"] and mlp["receiver"] == "gemma-3-12b-it"
    same = ("input", "tokens", "layer0", "kind", "kind_params", "seed")
    assert {k: mlp[k] for k in same} == {k: kimi_run["connector"][k] for k in same}
    assert {k: mine[k] for k in ("training_set", "batch", "seed")} == {
        k: kimi_run[k] for k in ("training_set", "batch", "seed")
    }
    protein = _resolved(CONFIGS / "runs" / "kimi-k2.6" / f"{KIMI['protein']}-all-tasks.yaml")
    assert mine["steps"] == protein["steps"] == 5000 and mine["optimizer"] == protein["optimizer"]
    for key in ("receiver", "encoder", "domain", "ridge", "embeddings", "device_class", "checkpoints", "eval_suite"):
        assert mine[key] == ordinary[key], key
    assert mine["micro_batch"] == 4 and "checkpointing" not in mine and "checkpointing" not in ordinary
    assert _yes_no(mine["mixture"]) == _resolved(
        CONFIGS / "streams" / "gemma-3-12b-it" / f"yes-no-{domain}-all-tasks-answer-free.yaml"
    )
    assert mine["destination"] == f"connectors/gemma-3-12b-it/{read}-mlp4-all-tasks"
    load_config(path, RunConfig, pins=_Pins())


@pytest.mark.parametrize("domain", sorted(KIMI))
def test_the_documented_build_command_pins_the_set_the_runs_read(domain):
    """The build command in the config's header pins the destination the runs cite: without ``--pin`` the stage pins
    nothing and the runs do not load."""
    path = CONFIGS / "training_sets" / f"{domain}-all-tasks.yaml"
    destination = load_yaml(path.read_text())["destination"]
    command = f"--config configs/training_sets/{domain}-all-tasks.yaml"
    header = "\n".join(line.removeprefix("#") for line in path.read_text().splitlines() if line.startswith("#"))
    assert f"{command} --pin {destination}" in _words(header)
    runs = [
        p
        for p in sorted((CONFIGS / "runs").rglob("*-all-tasks.yaml"))
        if f"{destination}/training_set" in p.read_text()
    ]
    assert len(runs) == 2  # the Kimi K2.6 and the Gemma 3 12B run of the domain


SUITES = {  # a suite -> (its source suite, the source's checkpoint pin, the step, its own checkpoint pin)
    f"suites/{receiver}/{mine}-{suite}-trained.yaml": (
        f"suites/{source}/{base}-{suite}-trained.yaml",
        f"connectors/{source}/{base}/step_002000",
        step,
        f"connectors/{receiver}/{mine}{published}/step_{step:06d}",
    )
    for receiver, source, base, mine, domain, step, published in (
        ("gemma-3-12b-it", "gemma-3-12b-it", "chemfm-3b", "chemfm-3b-mlp4-all-tasks", "molecule", 5000, ""),
        ("gemma-3-12b-it", "gemma-3-12b-it", "esmc-6b", "esmc-6b-mlp4-all-tasks", "protein", 5000, ""),
        (
            "kimi-k2.6",
            "kimi-k2.6",
            "esmc-6b-mlp4-staggered",
            "esmc-6b-mlp4-staggered-all-tasks",
            "protein",
            4800,
            "",
        ),
        ("kimi-k2.6", "gemma-3-12b-it", "chemfm-3b", "chemfm-3b-mlp4-all-tasks", "molecule", 4800, ""),
    )
    for suite in (
        [f"{domain}-{kind}-novel-scaffold" for kind in ("access", "transfer")]
        if domain == "molecule"
        else [f"{domain}-{kind}" for kind in ("access", "transfer")]
    )
}
"""The all-tasks connectors' trained-task (access) and historically held-out task (transfer) suites at each connector's
released step (Kimi K2.6: 4,800, the ESM-C 6B one from the publication of its stopped run; Gemma 3 12B: 5,000): each is
its source suite at step 2000 but for destination, checkpoint and step (Kimi K2.6's ChemFM-3B connector has no released
suite: its source is Gemma 3 12B's, on Kimi K2.6's suite spec, receiver and plan). None of their tasks is zero-shot
for these connectors."""


RELEASED_STEP = {"kimi-k2.6": 4800, "gemma-3-12b-it": 5000}
"""The released step of each receiver's all-tasks connectors."""


def test_only_the_named_suites_read_an_all_tasks_connector_and_only_at_its_released_step():
    """The access and transfer suites above, read from the pins each config cites (comments are not read)."""
    readers, steps = [], set()
    for p in sorted(CONFIGS.rglob("*.yaml")):
        cited = [n for n in cited_pins(p) if re.search(r"-all-tasks[^/]*/step_", n)]
        if cited:
            readers.append(str(p.relative_to(CONFIGS)))
            steps |= {(n.split("/")[1], int(n.rsplit("_", 1)[1])) for n in cited}
    assert sorted(readers) == sorted(SUITES)
    assert steps == set(RELEASED_STEP.items())


@pytest.mark.parametrize("path", sorted(SUITES))
def test_each_suite_is_its_source_but_for_checkpoint_destination_and_step(path):
    source, old_pin, step, pin = SUITES[path]
    mine, theirs = _resolved(CONFIGS / path), _resolved(CONFIGS / source)
    connector = path.split("/")[-1].removesuffix("-trained.yaml")
    receiver = path.split("/")[1]
    name = next(
        n
        for n in ("chemfm-3b-mlp4-all-tasks", "esmc-6b-mlp4-staggered-all-tasks", "esmc-6b-mlp4-all-tasks")
        if connector.startswith(n)
    )
    assert mine["checkpoints"] == {"m": {"$pin": pin}}
    assert theirs["checkpoints"] == {"m": {"$pin": old_pin}} and mine["step"] == step and theirs["step"] == 2000
    assert mine["destination"] == f"connectors/{receiver}/{name}/{connector.removeprefix(name + '-')}/trained"
    drop = {"destination", "checkpoints", "step"}
    if receiver not in source:  # the source is another receiver's: its suite spec, receiver and plan are this one's
        drop |= {"suite", "receiver", "plan"}
        kimi = _resolved(CONFIGS / "suites" / receiver / "esmc-6b-mlp4-staggered-protein-transfer-trained.yaml")
        assert mine["receiver"] == kimi["receiver"] and mine["plan"] == kimi["plan"]
        assert mine["suite"]["id"] == theirs["suite"]["id"] and mine["suite"]["tasks"] == theirs["suite"]["tasks"]
    assert {k: v for k, v in mine.items() if k not in drop} == {k: v for k, v in theirs.items() if k not in drop}


def test_the_kimi_protein_publication_publishes_its_run_under_the_plain_name():
    """The stopped Kimi K2.6 ESM-C 6B run is published (run.achieved) on CPU under the connector's plain name, and the
    run's own destination is that name with ``-training``, as the released pair
    ``esmc-6b-mlp4-staggered{,-achieved}.yaml``."""
    runs = CONFIGS / "runs" / "kimi-k2.6"
    for stem in ("esmc-6b-mlp4-staggered", "esmc-6b-mlp4-staggered-all-tasks"):
        published = _resolved(runs / f"{stem}-achieved.yaml")
        assert published["run"] == _resolved(runs / f"{stem}.yaml") and published["device_class"] == "cpu"
        assert published["destination"] == f"connectors/kimi-k2.6/{stem}"
        assert published["run"]["destination"] == f"connectors/kimi-k2.6/{stem}-training"
    load_config(runs / "esmc-6b-mlp4-staggered-all-tasks-achieved.yaml", AchievedSpec, pins=_Pins())


@pytest.mark.parametrize(
    "rel", ["esmc-6b-mlp4-staggered-all-tasks.yaml", "esmc-6b-mlp4-staggered-all-tasks-achieved.yaml"]
)
def test_the_destination_is_not_in_the_digest(rel):
    """Renaming the destination keeps the config's identity: the run (and its publication) under any other
    destination has the same digest, so its run folder and records are the same."""
    path = CONFIGS / "runs" / "kimi-k2.6" / rel
    cls = AchievedSpec if rel.endswith("-achieved.yaml") else RunConfig
    tree = load_resolved(path, _Pins())
    config = cls.model_validate(tree)
    renamed = cls.model_validate({**tree, "destination": "connectors/kimi-k2.6/another-name"})
    assert renamed.destination != config.destination and renamed.digest() == config.digest()
    if cls is AchievedSpec:  # the run it publishes keeps its identity under its own renamed destination
        run = {**tree["run"], "destination": "connectors/kimi-k2.6/another-name"}
        assert RunConfig.model_validate(run).digest() == config.run.digest()
