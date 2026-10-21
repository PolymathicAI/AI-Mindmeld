"""The pipeline on CPU, end to end through the ``run`` entry points, on tiny inputs.

Two chains share one workspace, one per domain (:class:`Chain`): the toy protein-like domain (``tests.fakes``) and the
real ``molecule`` domain on twelve stand-in compounds (:func:`commit_toy_molecules`, with the molecule frames and
capture recipes of ``molecule_prompts.yaml``). Each runs ``capture`` on the local tiny Llama → ``ridge`` → ``init`` (the
train-free step-0 connector) → ``suite`` (``yes_no``) → ``training_set`` → ``train`` (:data:`STEPS` steps per map kind;
the ``affine`` run is interrupted by SIGUSR1 and resumed by a second job, :func:`interrupted_then_resumed`) → ``suite``
on the trained ``affine`` checkpoint → ``analysis`` (initial and fitted cells per connector, with their paired change,
under the training-changes table's groups). The protein chain trains every registered map kind (:data:`KIND_PARAMS`).
Then the two protein wording analyses; the pipeline ends at the analysis records, so the checks read
the results (cells read by their fields, the support of each task). Last, the inputs check
(``Workspace.input_problems``, the preflight's ``--inputs``). Each stage's ``main(argv)`` runs in-process, so the
worktree's own code is what the records cite, and each stage cites its upstream outputs by ``{$pin: ...}``. The domains,
embeddings and object lists are fixture inputs, committed outside any stage.

- :func:`test_stage_configs_validate` checks every config of the chain against its schema now, with stand-in pins.
- :func:`test_chain_runs` runs the chain. Records refuse a dirty tree, so it is skipped with uncommitted changes
  (:func:`_dirty`); ``--require-e2e`` (the CPU gate, ``tests/conftest.py``) makes that skip a failure.
"""

import itertools
import json
import math
import os
import signal
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from tests.fakes import TOY_OBJECTS, build_tiny_llama, commit_toy_domain, commit_toy_embeddings, toy_recipe

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.build.capture import CaptureSpec
from latent_interconnect.build.init_state import InitSpec
from latent_interconnect.build.ridge import RidgeSpec
from latent_interconnect.build.zero_shot import TrainingSetSpec
from latent_interconnect.config import Config, Pin, dump_config, load_config
from latent_interconnect.data.items import ItemSet, commit_item_set, object_list, with_clusters
from latent_interconnect.domains.base import (
    CaptureRecipe,
    DomainSpec,
    Frame,
    TaskKind,
    TaskScope,
    TaskSpec,
    open_domain,
)
from latent_interconnect.domains.tables import DomainTables, commit_domain_data
from latent_interconnect.evaluation.analysis import AnalysisSpec, load_result
from latent_interconnect.evaluation.results import AnalysisResult, GroupedContrastsResult
from latent_interconnect.evaluation.suites import SuiteRunConfig
from latent_interconnect.paths import running_code
from latent_interconnect.run import analysis, capture, init, ridge, suite, train, training_set
from latent_interconnect.training import RunConfig, Trainer
from latent_interconnect.training.trainer import LOG_FILE
from latent_interconnect.workspace import Workspace
from latent_interconnect.writes.kinds import MAP_KINDS

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "latent_interconnect."
STEPS = 50
"""Training steps of every run in the chain."""
SIGNAL_AFTER = 9
"""The interrupted run signals itself after this update (0-based), so its first job stops at step 10."""
WORDINGS = {"seen": "has_c", "unseen": "long_name"}
"""The wording table's two records (one analysis record per wording set), toy stand-ins: one task each."""
KIND_PARAMS: dict[str, str] = {
    "affine": "{}",
    "mlp": "{d_s: 4, ff_mult: 2}",
    "xattn": "{d_s: 4, heads: 2, d_k: 2}",
    "xblock": "{d_s: 4, heads: 2, d_k: 2, ff_mult: 2}",
}
"""Every registered kind's tiny hyperparameters above layer 0; a newly registered kind fails
:func:`test_every_kind_is_trained`. ``constant`` is every run's ``layer0`` kind: a connector of constants alone reads no
input, which the input loader refuses."""
PROTEIN = ("mimic", "esmc-600m", "pglm-1b")
"""The protein connectors of the training-changes analysis: each reads the trained toy run's scores."""
MOLECULE = ("molformer-c3", "unimol2-1.1b", "chemfm-1b")
"""The molecule connectors of the training-changes analysis, likewise."""
PROTEIN_TRAINED = ("DNA-binding", "RNA-binding", "Ribosomal protein", "Transferase", "Membrane")
"""The protein training-changes groups (Fig 7a's), each one toy task here."""
MOLECULE_TRAINED = ("Biomolecules", "Oxygen functional groups", "Heterocycles", "Ring systems", "Other classes")
"""The molecule training-changes groups (Fig 7c's), likewise."""
MOLECULES = {  # id → (SMILES, name, lipid, metabolite, definition); names end in distinct letters (capture targets)
    "CHEBI:01": ("CCCCCCCCCC(=O)O", "toy compound a", True, False, "A medium-chain fatty acid."),
    "CHEBI:02": ("NCC(=O)O", "toy compound b", False, True, "An amino acid and human metabolite."),
    "CHEBI:03": ("CCCCCCCCCCCCCCCC(=O)O", "toy compound c", True, False, "A long-chain fatty acid."),
    "CHEBI:04": ("c1ccccc1O", "toy compound d", False, True, "A phenol and plant metabolite."),
    "CHEBI:05": ("OCC(O)CO", "toy compound e", True, False, "A triol and glycerol backbone."),
    "CHEBI:06": ("c1ccncc1", "toy compound f", False, True, "An azaarene and bacterial metabolite."),
    "CHEBI:07": ("CCCCCC=CCC(=O)O", "toy compound g", True, True, "An unsaturated fatty acid."),
    "CHEBI:08": ("CCCCCCCCCCCC(=O)OC", "toy compound h", True, False, "A fatty acid methyl ester."),
    "CHEBI:09": ("CCCCCCCCO", "toy compound i", True, True, "A fatty alcohol."),
    "CHEBI:10": ("c1ccc2ccccc2c1", "toy compound j", False, False, "A polycyclic arene."),
    "CHEBI:11": ("NC(=O)N", "toy compound k", False, True, "A carbonyl compound."),
    "CHEBI:12": ("ClC(Cl)Cl", "toy compound l", False, False, "A chloroalkane."),
}
"""Twelve stand-in compounds: CHEBI:01-06 train, 07-09 val, 10-12 test; two clusters by parity. The evaluation objects
hold three of each class in both Yes/No tasks, as an AUROC interval needs."""


@dataclass(frozen=True)
class Chain:
    """One domain's chain: its domain, tasks and wording, and the prefix of its pins and config files.

    Attributes:
        prefix: of the chain's pins, config files and fixture pins (``toy/<prefix>domain``, ...).
        domain: the registered domain.
        trained: the trained Yes/No task (stream and suites).
        held_out: the held-out Yes/No task (suites; the zero-shot screen's task).
        question: the trained task's training question.
        frame: the suites' frame.
        noun: the stream templates' noun.
        text_task: the trained text task (text stream).
        field: the text field the screen reads.
        term: the held-out task's lexical detector.
        connectors: the training-changes analysis's connectors of the domain.
        groups: its groups (each one task here).
        kinds: the map kinds trained.
    """

    prefix: str
    domain: str
    trained: str
    held_out: str
    question: str
    frame: str
    noun: str
    text_task: str
    field: str
    term: str
    connectors: tuple[str, ...]
    groups: tuple[str, ...]
    kinds: tuple[str, ...]

    def stages(self) -> list[tuple[object, type, str, str]]:
        """Return (stage module, config class, config file, pin name) per stage, in run order."""
        stages = [
            (capture, CaptureSpec, "capture.yaml", "cap"),
            (ridge, RidgeSpec, "ridge.yaml", "ridge"),
            (init, InitSpec, "init.yaml", "init"),
            (suite, SuiteRunConfig, "suite.yaml", "s0"),
            (training_set, TrainingSetSpec, "training_set.yaml", "tset"),
            *[(train, RunConfig, f"train_{kind}.yaml", f"train-{kind}") for kind in self.kinds],
            (suite, SuiteRunConfig, "suite_trained.yaml", f"s{STEPS}"),
            (analysis, AnalysisSpec, "analysis_changes.yaml", "changes"),
        ]
        return [(m, c, self.prefix + f, self.prefix + p) for m, c, f, p in stages]


PROTEIN_CHAIN = Chain(
    "",
    "fake-toy",
    "has_c",
    "long_name",
    "Does it contain cysteine?",
    "yes_no",
    "protein",
    "describe",
    "description",
    r"\blong-named\b",
    PROTEIN,
    PROTEIN_TRAINED,
    tuple(KIND_PARAMS),
)
MOLECULE_CHAIN = Chain(
    "mol-",
    "molecule",
    "lipid",
    "metabolite",
    "Is this compound a lipid?",
    "a5",
    "compound",
    "caption",
    "definition",
    r"\bmetabolite\b",
    MOLECULE,
    MOLECULE_TRAINED,
    ("affine",),
)
CHAINS = (PROTEIN_CHAIN, MOLECULE_CHAIN)
FINAL = [(analysis, AnalysisSpec, f"analysis_{w}.yaml", f"wording-{w}") for w in WORDINGS]
TRAINED = {"x": f"train-affine/step_{STEPS:06d}"}
"""The trained protein connector every composition and generation suite reads (one slot ``x``)."""
PATHS = [
    (suite, SuiteRunConfig, "composition.yaml", "comp"),  # within-domain: affine and mlp through one object
    (analysis, AnalysisSpec, "analysis_composition.yaml", "comp-changes"),
    (suite, SuiteRunConfig, "xdom.yaml", "xdom"),  # cross-domain: the protein and molecule connectors in one prompt
    (analysis, AnalysisSpec, "analysis_xdom.yaml", "xdom-quads"),
    (suite, SuiteRunConfig, "generation.yaml", "gen"),  # a trace, its post-reasoning readout, the decline
    (suite, SuiteRunConfig, "after_trace.yaml", "after"),
    (analysis, AnalysisSpec, "analysis_decline.yaml", "decline"),
    (suite, SuiteRunConfig, "name_guess.yaml", "names"),  # the elicited name and its exact share
    (analysis, AnalysisSpec, "analysis_names.yaml", "names-share"),
]
"""The composition, cross-domain and generation paths on the trained connectors of both chains."""
CHAIN = [*PROTEIN_CHAIN.stages(), *MOLECULE_CHAIN.stages(), *FINAL, *PATHS]
"""Every stage, in run order."""
XDOM_PAIRS = tuple(itertools.combinations(range(6), 2))
"""Swap quads over the positives (P_i, M_i), the two evaluation sets' i-th objects; quad (i, j) adds two swaps. Every
pair, so most draws of the per-object bootstrap keep a quad (the interval refuses under half the draws defined)."""
INPUT_PINS = {f"toy/{c.prefix}{name}": "artifact" for c in CHAINS for name in ("domain", "embeddings", "fit", "eval")}
INPUT_PINS["toy/xdom"] = "artifact"
"""Pins the fixture sets from committed fakes; every other pin is a stage output."""


class MoleculePrompts(Config):
    """The molecule frames and capture recipes of ``molecule_prompts.yaml``."""

    frames: tuple[Frame, ...]
    recipes: tuple[CaptureRecipe, ...]


def commit_toy_molecules(store: ArtifactStore) -> ArtifactRef:
    """Commit :data:`MOLECULES` as ``molecule`` domain data (the real domain class; the frames and recipes of
    ``molecule_prompts.yaml``)."""
    spec = load_config(Path(__file__).with_name("molecule_prompts.yaml"), MoleculePrompts)
    rows = pd.DataFrame(list(MOLECULES.values()), index=list(MOLECULES), columns=["value", "name", "l", "m", "d"])
    tasks = (
        TaskSpec(
            id="lipid",
            kind=TaskKind.YES_NO,
            question="Is this compound a lipid?",
            label="@class:CHEBI:18059",
            scope=TaskScope.TRAINED,
        ),
        TaskSpec(
            id="metabolite",
            kind=TaskKind.YES_NO,
            question="Is this compound a metabolite?",
            label="@role:CHEBI:25212",
            scope=TaskScope.HELD_OUT,
        ),
        TaskSpec(id="caption", kind=TaskKind.TEXT, question="", label="definition", scope=TaskScope.TRAINED),
    )
    tables = DomainTables(
        domain="molecule",
        objects=rows[["value", "name"]],
        split=pd.Series(["train"] * 6 + ["val"] * 3 + ["test"] * 3, index=rows.index),
        clusters=pd.Series([str(n % 2) for n in range(len(rows))], index=rows.index),
        labels=pd.DataFrame({tasks[0].label: rows["l"], tasks[1].label: rows["m"]}),
        fields=pd.DataFrame({"definition": rows["d"]}),
        tasks=tasks,
        frames=spec.frames,
        recipes=spec.recipes,
    )
    return commit_domain_data(store, tables, {})


def write_chain(directory: Path, receiver: Path, c: Chain) -> None:
    """Write the encoder recipe, the connectors and every stage config of one chain."""
    dump_config(toy_recipe(residues=True), directory / "encoder.yaml")
    p = c.prefix
    domain = f"{{name: {c.domain}, data: {{$pin: toy/{p}domain}}}}"
    configs = {
        "capture.yaml": f"""recipe: name
receiver: {{$ref: {receiver}}}
domain: {domain}
layers: [0, 1, 2]
objects: {{$pin: toy/{p}fit}}
device_class: cpu
""",
        "ridge.yaml": f"""domain: {domain}
input: {{recipe: fake-toy, layers: [0]}}
encoder: {{$ref: encoder.yaml}}
embeddings: [{{$pin: toy/{p}embeddings}}]
targets: {{$pin: {p}cap}}
target: name
layers: [0, 1, 2]
fit_set: {{$pin: toy/{p}fit}}
alphas: [0.1, 1.0, 10.0]
folds: 2
""",
        "connector.yaml": """name: toy-linear
receiver: tiny-llama
input: {recipe: fake-toy, layers: [0]}
tokens:
  - {target: name}
layer0: constant
layers:
  rule: explicit
  per_token:
    - [1, 2]
kind: affine
alpha: {first: 1.0, init: 0.4}
layer_choice: {note: "unit-scale e2e"}
""",
        "init.yaml": f"""connector: {{$ref: {p}connector.yaml}}
receiver: {{$ref: {receiver}}}
encoder: {{$ref: encoder.yaml}}
ridge: {{name: {{$pin: {p}ridge}}}}
""",
        "suite.yaml": f"""suite:
  id: toy/{p}yes_no@eval
  kind: yes_no
  domains: [{domain}]
  item_set: {{$pin: toy/{p}eval}}
  tasks: [{c.trained}, {c.held_out}]
  frame: {c.frame}
  slots: [{{name: x, role: object, connector: x}}]
  conditions: [{{object: own}}, {{object: name}}]
  capability: score
  params: {{"yes": "Y", "no": "N"}}  # quoted keys (YAML 1.1 reads yes/no as booleans); one token each
receiver: {{$ref: {receiver}}}
checkpoints: {{x: {{$pin: {p}init}}}}
inputs: {{x: [{{$pin: toy/{p}embeddings}}]}}
step: 0
device_class: cpu
plan: {{pad_every_batch: true}}  # a backend that pads states it
""",
        "training_set.yaml": f"""domain: {domain}
train_ids: {{$pin: toy/{p}fit}}
screen:
  fields: [{c.field}]
  property_list_fields: []
  zero_shot_tasks: [{c.held_out}]
  detectors: [{{task: {c.held_out}, detector: lexical, params: {{terms: ['{c.term}']}}}}]
""",
    }
    for kind in c.kinds:
        configs[f"connector_{kind}.yaml"] = configs["connector.yaml"].replace(
            "kind: affine", f"kind: {kind}\nkind_params: {KIND_PARAMS[kind]}"
        )
        configs[f"train_{kind}.yaml"] = f"""name: toy-{p}{kind}
connector: {{$ref: {p}connector_{kind}.yaml}}
receiver: {{$ref: {receiver}}}
encoder: {{$ref: encoder.yaml}}
domain: {domain}
ridge: {{name: {{$pin: {p}ridge}}}}
embeddings: [{{$pin: toy/{p}embeddings}}]
training_set: {{$pin: {p}tset}}
mixture:
  - stream: yes_no
    share: 0.75
    params:
      templates: [{{id: y1, user: "Here is a {c.noun}:<SLOT:x>. {{question}} Answer Yes or No."}}]
      questions: {{{c.trained}: ["{c.question}"]}}
      sampling: uniform
      answers: {{"yes": "Y", "no": "N"}}
      weights: {{family: single_token, floor: 0.1}}
  - stream: text
    share: 0.25
    params:
      templates: [{{id: d1, task: {c.text_task}, user: "Describe<SLOT:x>."}}]
      sampling: cluster_sqrt
      max_target_tokens: 8
      weights: {{{c.text_task}: {{family: power, floor: 0.1, tau: 0.001, p: 1.2}}}}
steps: {STEPS}
batch: 4
device_class: cpu
optimizer: {{lr_rel: 0.05, lr_alpha: 0.05, warmup_steps: 2, schedule: constant, weight_decay: 0.0, clip: 1.0}}
checkpoints: {{resume_every: 25}}
"""
    configs["suite_trained.yaml"] = (
        configs["suite.yaml"]
        .replace(f"{{$pin: {p}init}}", f"{{$pin: {p}train-affine/step_{STEPS:06d}}}")
        .replace("step: 0", f"step: {STEPS}")
    )
    tasks = (c.trained, c.held_out)
    configs["analysis_changes.yaml"] = (
        f"id: toy/{p}training-changes\nkind: grouped_contrasts\n"
        f"inputs: {{s0: {{$pin: {p}s0@record}}, s{STEPS}: {{$pin: {p}s{STEPS}@record}}}}\nparams:\n  cells:\n"
        + "".join(
            f"    {e} {name}: {{encoder: {e}, inputs: [s{step}], step: {step}, conditions: {{object: own}}}}\n"
            for e in c.connectors
            for name, step in (("initial", 0), ("fitted", STEPS))
        )
        + "  groups:\n"
        + "".join(f"    - {{name: {g}, tasks: [{tasks[n % 2]}]}}\n" for n, g in enumerate(c.groups))
        + "  contrasts:\n"
        + "".join(f"    - {{name: {e} change, a: {e} fitted, b: {e} initial}}\n" for e in c.connectors)
    )
    for name, text in configs.items():
        (directory / (p + name)).write_text(text)


def write_final(directory: Path) -> None:
    """Write the two protein wording analyses."""
    configs = {
        **{
            f"analysis_{w}.yaml": f"id: toy/wording-{w}\nkind: grouped_contrasts\ninputs: {{"
            + ", ".join(f"{e}: {{$pin: s{STEPS}@record}}" for e in PROTEIN)
            + "}\nparams:\n  cells:\n"
            + "".join(
                f"    {e}: {{encoder: {e}, inputs: [{e}], step: {STEPS}, conditions: {{object: own}}}}\n"
                for e in PROTEIN
            )
            + f"  groups: [{{name: all trained, tasks: [{task}]}}]\n"
            for w, task in WORDINGS.items()
        },
    }
    for name, text in configs.items():
        (directory / name).write_text(text)


def commit_toy_xdom(ws: Workspace, domains: dict[str, DomainSpec]) -> ArtifactRef:
    """Commit the cross-domain swap-quad set (:data:`XDOM_PAIRS`) over the two chains' evaluation objects."""
    roles = {"protein": "", "molecule": "mol-"}
    ids = {r: ws.item_set(ws.pins()[f"toy/{p}eval"].sha256).object_ids for r, p in roles.items()}
    rows = []
    for q, (i, j) in enumerate(XDOM_PAIRS):
        rows += [(q, i, i, 1), (q, j, j, 1), (q, i, j, 0), (q, j, i, 0)]
    items = ItemSet(
        "toy-xdom",
        {r: domains[p].name for r, p in roles.items()},
        {
            "protein": tuple(ids["protein"][i] for _, i, _, _ in rows),
            "molecule": tuple(ids["molecule"][j] for *_, j, _ in rows),
        },
        {"quad": tuple(q for q, *_ in rows), "pair_label": tuple(y for *_, y in rows)},
    )
    items = with_clusters(items, {d.name: open_domain(d, ws.artifacts) for d in domains.values()})
    units = {"protein_cluster": items.fields["cluster.protein"], "molecule_group": items.objects["molecule"]}
    return commit_item_set(ws.artifacts, ItemSet(items.name, items.roles, items.objects, {**items.fields, **units}))


def write_paths(directory: Path, receiver: Path) -> None:
    """Write the composition, cross-domain and generation paths' configs (:data:`PATHS`)."""
    domain = "{name: fake-toy, data: {$pin: toy/domain}}"
    run = f"""receiver: {{$ref: {receiver}}}
inputs: {{x: [{{$pin: toy/embeddings}}]}}
checkpoints: {{x: {{$pin: {TRAINED["x"]}}}}}
step: {STEPS}
device_class: cpu
"""
    one_slot = f"""  domains: [{domain}]
  item_set: {{$pin: toy/eval}}
  slots: [{{name: x, role: object, connector: x}}]
"""
    yes_no = "{question} Answer Yes or No."
    sequential = "generation_plan: {max_num_seqs: 1, max_num_batched_tokens: null}  # hf: one prompt at a time\n"
    scoring = "plan: {pad_every_batch: true}  # a backend that pads states it (run.suite.require_stated_padding)\n"
    configs = {
        "composition.yaml": f"""suite:
  id: toy/composition@eval
  kind: composition
  domains: [{domain}]
  item_set: {{$pin: toy/eval}}
  tasks: [has_c, long_name]
  frame: {{id: two, user: "Here is a protein:<SLOT:a><SLOT:b>. {yes_no}"}}
  slots: [{{name: a, role: object, connector: a}}, {{name: b, role: object, connector: b}}]
  conditions: [{{object: own}}]
  capability: score
  params: {{readout: yes_no, "yes": "Y", "no": "N"}}
receiver: {{$ref: {receiver}}}
checkpoints: {{a: {{$pin: {TRAINED["x"]}}}, b: {{$pin: train-mlp/step_{STEPS:06d}}}}}
inputs: {{a: [{{$pin: toy/embeddings}}], b: [{{$pin: toy/embeddings}}]}}
step: {STEPS}
device_class: cpu
{scoring}""",
        "analysis_composition.yaml": f"""id: toy/composition
kind: grouped_contrasts
inputs: {{single: {{$pin: s{STEPS}@record}}, both: {{$pin: comp@record}}}}
params:
  cells:
    affine: {{encoder: affine, inputs: [single], step: {STEPS}, conditions: {{object: own}}}}
    affine+mlp: {{encoder: affine+mlp, inputs: [both], step: {STEPS}, conditions: {{object: own}}}}
  groups: [{{name: trained, tasks: [has_c]}}, {{name: held out, tasks: [long_name]}}]
  contrasts: [{{name: composed minus single, a: affine+mlp, b: affine}}]
""",
        "xdom.yaml": f"""suite:
  id: toy/xdom
  kind: cross_domain
  domains: [{domain}, {{name: molecule, data: {{$pin: toy/mol-domain}}}}]
  item_set: {{$pin: toy/xdom}}
  tasks: [pair]
  slots: [{{name: p, role: protein, connector: p}}, {{name: m, role: molecule, connector: m}}]
  conditions:
    - {{protein: own, molecule: own}}
    - {{protein: name, molecule: name}}
    - {{protein: own, molecule: placeholder}}
  capability: score
  params:
    user: "{{p}} {{m}} Do they pair? Answer Yes or No."
    sentences:
      p: {{own: "A protein:<SLOT:p>.", placeholder: "A protein:<SLOT:p>.", name: "A protein named<SLOT:p>."}}
      m: {{own: "A compound:<SLOT:m>.", placeholder: "A compound:<SLOT:m>.", name: "A compound named<SLOT:m>."}}
    "yes": "Y"
    "no": "N"
receiver: {{$ref: {receiver}}}
checkpoints: {{p: {{$pin: {TRAINED["x"]}}}, m: {{$pin: mol-train-affine/step_{STEPS:06d}}}}}
inputs: {{p: [{{$pin: toy/embeddings}}], m: [{{$pin: toy/mol-embeddings}}]}}
step: {STEPS}
device_class: cpu
{scoring}""",
        "analysis_xdom.yaml": f"""id: toy/xdom-quads
kind: cross_domain_quads
inputs: {{xd: {{$pin: xdom@record}}}}
params:
  cells:
    own-own: {{encoder: xd, inputs: [xd], step: {STEPS}, conditions: {{protein: own, molecule: own}}}}
    name-name: {{encoder: xd, inputs: [xd], step: {STEPS}, conditions: {{protein: name, molecule: name}}}}
    own-placeholder: {{encoder: xd, inputs: [xd], step: {STEPS}, conditions: {{protein: own, molecule: placeholder}}}}
  contrasts: [{{name: own minus names, a: own-own, b: name-name}}]
units: {{factors: [protein_cluster, molecule_group]}}
""",
        "generation.yaml": f"""suite:
  id: toy/trace
  kind: generation
{one_slot}  tasks: [has_c, long_name]
  frame: yes_no
  conditions: [{{object: own}}]
  capability: generate
  generation: {{max_new_tokens: 6, temperature: 0.0}}
{run}{sequential}""",
        "after_trace.yaml": f"""suite:
  id: toy/after-trace
  kind: yes_no
{one_slot}  tasks: [has_c, long_name]
  frame: yes_no
  conditions: [{{object: own}}]
  capability: score
  upstream_kind: generation
  params: {{"yes": "Y", "no": "N", prefill: "@A:"}}
upstream: {{$pin: gen@record}}
{run}{scoring}""",
        "analysis_decline.yaml": f"""id: toy/decline
kind: generation_decline
inputs: {{direct: {{$pin: s{STEPS}@record}}, after: {{$pin: after@record}}}}
params: {{tasks: [has_c, long_name]}}
""",
        "name_guess.yaml": f"""suite:
  id: toy/names
  kind: name_guess
{one_slot}  frame: {{id: identify, user: "Here is a protein:<SLOT:x>. What is its name?"}}
  conditions: [{{object: own}}]
  capability: generate
  generation: {{max_new_tokens: 8, temperature: 0.0}}
  params: {{max_answer_chars: 200}}
{run}{sequential}""",
        "analysis_names.yaml": """id: toy/names
kind: name_guess
inputs: {names: {$pin: names@record}}
""",
    }
    for name, text in configs.items():
        (directory / name).write_text(text)


def _stand_in_pins() -> dict[str, Pin]:
    pins = {name: Pin(target="artifact", sha256=f"{n:064x}") for n, name in enumerate(INPUT_PINS, start=1)}
    for n, (_, _, _, name) in enumerate(CHAIN, start=10):
        for output in ("", f"/step_{STEPS:06d}"):
            pins[name + output] = Pin(target="artifact", sha256=f"{n:063x}{len(output):x}")
        pins[f"{name}@record"] = Pin(target="record", sha256=f"{n:063x}f")
    return pins


def test_stage_configs_validate(tmp_path):
    receiver = tmp_path / "receiver.yaml"
    receiver.write_text((ROOT / "configs" / "receivers" / "tiny-llama.yaml").read_text())
    for c in CHAINS:
        write_chain(tmp_path, receiver, c)
    write_final(tmp_path)
    write_paths(tmp_path, receiver)
    pins = _stand_in_pins()
    for _, model, name, _ in CHAIN:
        load_config(tmp_path / name, model, pins)


def test_every_kind_is_trained():
    registered = [n for n in MAP_KINDS if MAP_KINDS.get(n).__module__.startswith(PACKAGE)]
    assert sorted([*KIND_PARAMS, "constant"]) == sorted(registered)  # the package's kinds, not tests' own


def _dirty() -> bool:
    """Whether a record of this run would be refused: the dirty flag the records themselves read (``running_code``).

    False in a copy without git (the CPU gate's ``git archive`` of a commit), as for an installed package, so the
    chain runs there.
    """
    return running_code()[1]


def interrupted_then_resumed(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run ``train`` as two jobs: the first signals itself (SIGUSR1) and exits 1, the second resumes and finishes."""
    update = Trainer._update

    def signalled(self: Trainer, step: int, optimizer: object) -> dict:
        record = update(self, step, optimizer)
        if step == SIGNAL_AFTER:
            os.kill(os.getpid(), signal.SIGUSR1)
        return record

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)  # two jobs, two ids (segments are second-resolution)
    monkeypatch.setenv("LIP_JOB_ID", "1")
    monkeypatch.setattr(Trainer, "_update", signalled)
    with pytest.raises(SystemExit) as stop:
        train.main(argv)
    assert stop.value.code == 1  # resume state saved, no record written
    monkeypatch.setattr(Trainer, "_update", update)
    monkeypatch.setenv("LIP_JOB_ID", "2")
    assert train.main(argv) == 0


def run_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: str = "data") -> Workspace:
    """Run both chains and the wording analyses in a fresh workspace ``tmp_path / data``; return it."""
    from latent_interconnect.evaluation import contrasts

    monkeypatch.setattr(contrasts, "THIN_CELL", 2)  # toy populations; the thin-cell rule itself: test_thin_cells.py
    monkeypatch.setenv("LIP_DATA", str(tmp_path / data))
    ws = Workspace.from_env()
    receiver = tmp_path / "receiver.yaml"
    tiny = build_tiny_llama(tmp_path / "tiny-llama").model_copy(update={"thinking_close": "@"})  # a trace's close
    dump_config(tiny, receiver)
    domains = {"": commit_toy_domain(ws.artifacts), "mol-": commit_toy_molecules(ws.artifacts)}
    values = {"": {i: v for i, (v, _) in TOY_OBJECTS.items()}, "mol-": {i: v[0] for i, v in MOLECULES.items()}}
    for c in CHAINS:
        p = c.prefix
        ws.pin(f"toy/{p}domain", domains[p].sha256, "artifact")
        ws.pin(f"toy/{p}embeddings", commit_toy_embeddings(ws.artifacts, values[p], residues=True).sha256, "artifact")
        domain = open_domain(DomainSpec(name=c.domain, data=domains[p].sha256), ws.artifacts)
        for name, splits in (("fit", ("train",)), ("eval", ("val", "test"))):
            ids = tuple(sorted(domain.split()[domain.split().isin(splits)].index))
            ws.pin(f"toy/{p}{name}", commit_item_set(ws.artifacts, object_list(name, domain, ids)).sha256, "artifact")
        write_chain(tmp_path, receiver, c)
    write_final(tmp_path)
    specs = {p: DomainSpec(name=c.domain, data=domains[p].sha256) for p, c in zip(domains, CHAINS, strict=True)}
    ws.pin("toy/xdom", commit_toy_xdom(ws, specs).sha256, "artifact")
    write_paths(tmp_path, receiver)
    for module, _, name, pin in CHAIN:
        argv = ["--config", str(tmp_path / name), "--pin", pin]
        if pin.endswith("train-affine"):
            interrupted_then_resumed(argv, monkeypatch)
        else:
            assert module.main(argv) == 0, name
    return ws


@pytest.mark.skipif(_dirty(), reason="records refuse a dirty tree: commit, then run the e2e")
def test_chain_runs(tmp_path, monkeypatch, capfd):
    ws = run_chain(tmp_path, monkeypatch)
    assert f"train: step {STEPS}/{STEPS} loss " in capfd.readouterr().err  # the per-step line of the training CLI
    pins = ws.pins()
    scores = ws.records.canonical(pins["s0@record"].sha256)
    assert scores.purpose == "anchor"  # its item set is a fixture, committed outside any stage
    assert scores.provenance.execution and not scores.provenance.dirty

    def output(name: str) -> Path:
        return ws.artifacts.path(ws.artifacts.ref(pins[name].sha256))

    for c in CHAINS:
        for kind in c.kinds:
            log = [json.loads(x) for x in (output(f"{c.prefix}train-{kind}/log") / LOG_FILE).read_text().splitlines()]
            steps = [x["step"] for x in log if x["event"] == "train"]
            assert steps == list(range(1, STEPS + 1)), (c.domain, kind)  # every step once, across both jobs
            assert all(math.isfinite(x["loss"]) for x in log if x["event"] == "train"), (c.domain, kind)
            timed = [x["event"] for x in log if x["event"] in ("checkpoint", "resume_state") and x["sec"] >= 0]
            stopped = ["resume_state"] if kind == "affine" else []  # the interrupted job's save
            assert timed == ["checkpoint", *stopped, "resume_state", "checkpoint"], (c.domain, kind)
        run = ws.records.canonical(pins[f"{c.prefix}train-affine@record"].sha256)
        assert [s.job_id for s in run.provenance.resumed] == ["1"] and run.provenance.job_id == "2", c.domain

    # the inputs check: the analysis reads records made by the new code; the suite reads fixture inputs (no producing
    # record) and the init checkpoint built on them (an anchor record)
    assert ws.input_problems(tmp_path / "analysis_changes.yaml", AnalysisSpec) == []
    problems = ws.input_problems(tmp_path / "suite.yaml", SuiteRunConfig)
    assert sorted(p.split(":")[0] for p in problems) == ["init", "toy/domain", "toy/embeddings", "toy/eval"], problems

    def result(name: str) -> AnalysisResult:
        return load_result(ws.artifacts, ws.records.canonical(pins[f"{name}@record"].sha256), AnalysisResult)

    def estimates(value: object) -> list[dict]:
        if isinstance(value, dict):
            return [value] if "estimate" in value else [e for v in value.values() for e in estimates(v)]
        return []

    for w in WORDINGS:  # one record per wording set, every connector's cell on 'all trained'
        wording = result(f"wording-{w}")
        assert isinstance(wording, GroupedContrastsResult)
        assert {s.encoder for s in wording.cell_specs.values()} == set(PROTEIN)
    for c in CHAINS:  # the initial and fitted cells by their fields, each task's rows recorded
        changes = result(f"{c.prefix}changes")
        assert isinstance(changes, GroupedContrastsResult)
        fields = {(s.encoder, s.step) for s in changes.cell_specs.values()}
        assert fields == {(e, step) for e in c.connectors for step in (0, STEPS)}
        assert set(changes.support) == set(changes.cells)
        assert all(sup.n > 0 and sup.positives for cell in changes.support.values() for sup in cell.values())
    for name in ("comp-changes", "xdom-quads", "decline", "names-share"):  # each path's estimates and intervals
        found = estimates(result(name).model_dump())
        assert found and all(e["lo"] <= e["estimate"] <= e["hi"] for e in found), name
    xdom = result("xdom-quads")
    assert xdom.quads == len(XDOM_PAIRS) and set(xdom.cells) == {"own-own", "name-name", "own-placeholder"}
    assert xdom.cells["own-placeholder"].estimate == 0.5  # a single-slot score orders exactly half of each quad


PER_EXECUTION = ("started", "ended", "sec", "job_id")
"""What differs between two executions by design: provenance times and job id, the training log's seconds."""


def untimed(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in PER_EXECUTION}


@pytest.mark.skipif(_dirty(), reason="records refuse a dirty tree: commit, then run the e2e")
def test_chain_is_deterministic(tmp_path, monkeypatch):
    """Two fresh workspaces (one receiver path, which configs cite) give the same artifacts (digests, so contents and
    metas) and computation keys; only the record ids differ, which each record lists for the executions it
    read (``Provenance.input_executions``)."""
    a, b = (run_chain(tmp_path, monkeypatch, data) for data in ("data-a", "data-b"))
    pins, other = a.pins(), b.pins()
    records = [n for n, p in pins.items() if p.target == "record"]

    def manifest(ws: Workspace, name: str) -> dict:
        m = ws.artifacts.manifest(ws.artifacts.ref(ws.pins()[name].sha256))
        return {"kind": m.kind, "files": m.files, "meta": m.meta}

    def body(ws: Workspace, name: str) -> dict:
        record = ws.records.canonical(ws.pins()[name].sha256).model_dump(mode="json")
        record["outputs"].pop("log", None)  # the training log's digest covers its seconds: compared untimed below
        prov = record["provenance"]
        prov.pop("input_executions")  # record ids, per execution: checked below
        return record | {"provenance": untimed(prov | {"resumed": [untimed(s) for s in prov["resumed"]]})}

    def log(ws: Workspace, name: str) -> list[dict]:
        text = (ws.artifacts.path(ws.artifacts.ref(ws.pins()[name].sha256)) / LOG_FILE).read_text()
        return [untimed(json.loads(x)) for x in text.splitlines()]

    logs = [n for n in pins if n.endswith("/log")]
    assert logs and all(log(a, n) == log(b, n) for n in logs)  # every logged number, the seconds aside
    artifacts = [n for n, p in pins.items() if p.target == "artifact" and n not in logs]
    assert all(manifest(a, n) == manifest(b, n) for n in artifacts)  # every file byte-identical, every meta
    assert sorted(n for n in [*artifacts, *records] if pins[n] != other[n]) == []  # digests and computation keys
    assert all(body(a, n) == body(b, n) for n in records)  # config, inputs, outputs, code, execution

    def lineage(ws: Workspace, name: str) -> dict[str, tuple[str, str]]:
        rec = ws.records.canonical(ws.pins()[name].sha256)
        return {r: (ws.records.get(i).computation_key(), i) for r, i in rec.provenance.input_executions.items()}

    readers = [n for n in records if lineage(a, n)]
    assert readers  # the analyses name the executions they read: same computations, other executions
    assert all(
        {r: k for r, (k, _) in lineage(a, n).items()} == {r: k for r, (k, _) in lineage(b, n).items()} for n in readers
    )
    assert all(lineage(a, n)[r][1] != lineage(b, n)[r][1] for n in readers for r in lineage(a, n))
