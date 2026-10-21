"""``examples/ask.py --yes-no`` reads exactly what a ``yes_no`` suite records, on the tiny Llama.

One object set is scored by the ``suite`` stage's own run function (``run.suite``: the checkpoint, the embedding
inputs, the runner) and then asked again, object by object, through ``ask_yes_no`` on plain files: the checkpoint's
directory and the object's stored embedding. Every row's ``logp_yes``, ``logp_no`` and ``score`` must be equal, for a
pooled kind (``affine``, given the pooled vector) and a residue kind (``xattn``, given the residue states), under an
answer convention (instruction, prefix ``"Answer:"``, answers with a leading space, as Gemma's) and under inline words
(instruction inside the frame, bare answers, as Kimi K2.6's paper suites). The plan gives each prompt its own
micro-batch, so the suite's forward passes are the ones ``ask_yes_no`` makes. The answer token ids and the rendered
prompt's ids are the runner's, on the tiny tokenizer and (with its local snapshot, ``tests/snapshots.py``) on Gemma 3
4B's for the release suite: ``" Yes"`` 8438 and ``" No"`` 2301 after ``"Answer:"``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests import snapshots
from tests.e2e.test_tiny_pipeline import PROTEIN_CHAIN, write_chain
from tests.fakes import TOY_DIM, TOY_OBJECTS, build_tiny_llama, commit_toy_domain, commit_toy_embeddings

from latent_interconnect.config import dump_config, load_config
from latent_interconnect.config_index import cited_pins, stand_in_pins
from latent_interconnect.data.answers import AnswerConvention
from latent_interconnect.data.items import commit_item_set, object_list
from latent_interconnect.data.suites import Condition
from latent_interconnect.domains.base import DomainSpec, Frame, open_domain
from latent_interconnect.encoders.store import EmbeddingReader
from latent_interconnect.evaluation.runners import YesNo
from latent_interconnect.evaluation.suites import SuiteRunConfig
from latent_interconnect.receivers import open_renderer
from latent_interconnect.receivers.render import RenderedPrompt, Renderer
from latent_interconnect.run import suite
from latent_interconnect.training import RunConfig
from latent_interconnect.workspace import Workspace
from latent_interconnect.writes import Connector
from latent_interconnect.writes.checkpoint import commit_checkpoint
from latent_interconnect.writes.init import RidgeLayer, RidgeSolution

ROOT = Path(__file__).resolve().parents[1]
WORDING = "Here is a protein:<SLOT:x>. {question}"
"""The suites' frame without an answer instruction: what a user of ``ask.py --yes-no`` writes around a question."""
CONVENTION = AnswerConvention(instruction="Answer with only Y or N.", prefix="Answer:", yes=" Y", no=" N")
"""Gemma's convention shape (``configs/answers/gemma-3.yaml``) in the tiny tokenizer's one-character tokens."""
INLINE = AnswerConvention(instruction="Answer Y or N.", yes="Y", no="N")
"""Kimi's (``configs/answers/kimi-k2.6.yaml``): no prefix, bare answers; an inline suite's frame ends on the
instruction, which this convention appends after one space."""
SUITES = {
    "convention": (
        f"frame: {{id: q, user: '{WORDING}'}}\n  params: {{}}\n  answer: "
        "{instruction: 'Answer with only Y or N.', prefix: 'Answer:', 'yes': ' Y', 'no': ' N'}\n",
        CONVENTION,
    ),
    "inline": (f"frame: {{id: q, user: '{WORDING} Answer Y or N.'}}\n  params: {{'yes': Y, 'no': N}}\n", INLINE),
}


def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> tuple[Workspace, Path, Path]:
    """Return a workspace holding the toy domain, its embeddings, an item set and one ``kind`` checkpoint (step 0),
    pinned; the tiny receiver's config; the checkpoint's directory."""
    monkeypatch.setenv("LIP_DATA", str(tmp_path / "data"))
    for var in ("LIP_ARTIFACTS", "LIP_RECORDS", "LIP_RUNS", "LIP_SCRATCH"):
        monkeypatch.delenv(var, raising=False)
    ws = Workspace.from_env()
    receiver = tmp_path / "receiver.yaml"
    dump_config(build_tiny_llama(tmp_path / "tiny-llama"), receiver)
    write_chain(tmp_path, receiver, PROTEIN_CHAIN)
    domain = commit_toy_domain(ws.artifacts)
    ws.pin("toy/domain", domain.sha256, "artifact")
    values = {i: v for i, (v, _) in TOY_OBJECTS.items()}
    ws.pin("toy/embeddings", commit_toy_embeddings(ws.artifacts, values, residues=True).sha256, "artifact")
    toy = open_domain(DomainSpec(name="fake-toy", data=domain.sha256), ws.artifacts)
    ids = tuple(sorted(TOY_OBJECTS))[:4]
    ws.pin("toy/eval", commit_item_set(ws.artifacts, object_list("eval", toy, ids)).sha256, "artifact")
    config = tmp_path / f"train_{kind}.yaml"
    run = load_config(config, RunConfig, stand_in_pins(cited_pins(config)))
    g = np.random.default_rng(0)
    d_model = run.receiver.d_model
    layers = {
        L: RidgeLayer(g.normal(size=(d_model, TOY_DIM)).astype(np.float32), g.normal(size=d_model).astype(np.float32))
        for L in (0, 1, 2)
    }
    fit = RidgeSolution("name", g.normal(size=TOY_DIM).astype(np.float32), np.ones(TOY_DIM, np.float32), layers)
    checkpoint = commit_checkpoint(
        Connector.from_ridge(run.connector, run.receiver.depth, {"name": fit}), 0, ws.artifacts, encoder=run.encoder
    )
    ws.pin("toy/checkpoint", checkpoint.sha256, "artifact")
    return ws, receiver, ws.artifacts.path(checkpoint)


def _suite(tmp_path: Path, receiver: Path, wording: str) -> Path:
    path = tmp_path / "suite_yes_no.yaml"
    path.write_text(f"""suite:
  id: toy/ask-yes-no
  kind: yes_no
  domains: [{{name: fake-toy, data: {{$pin: toy/domain}}}}]
  item_set: {{$pin: toy/eval}}
  tasks: [has_c, long_name]
  {wording}  slots: [{{name: x, role: object, connector: x}}]
  conditions: [{{object: own}}]
  thinking: false
  capability: score
receiver: {{$ref: {receiver}}}
checkpoints: {{x: {{$pin: toy/checkpoint}}}}
inputs: {{x: [{{$pin: toy/embeddings}}]}}
step: 0
device_class: cpu
plan: {{budget_tokens: 1, pad_every_batch: true}}  # one prompt per micro-batch, padded as ask_yes_no pads
""")
    return path


def _suite_prompt(runner: YesNo, renderer: Renderer, question: str) -> RenderedPrompt:
    """Return the prompt the runner renders for one ``own`` row of a direct readout (``_SlotRunner._rows``)."""
    frame = runner.spec.frame
    assert isinstance(frame, Frame)
    policy = runner.answering
    assert policy is not None
    return runner._render(
        renderer,
        policy.user(frame.ask(question)),
        Condition.OWN,
        system=frame.system,
        filler="object",
        name="",
        ks={s.name: 1 for s in runner.slots},
        target=None,
        prefill=runner.prefill() or None,
    )


@pytest.mark.parametrize("wording", sorted(SUITES))
@pytest.mark.parametrize("kind", ["affine", "xattn"])
def test_ask_yes_no_reads_the_suite_score(kind, wording, tmp_path, monkeypatch):
    from examples.ask import ask_yes_no, yes_no_prompt

    ws, receiver, checkpoint = _workspace(tmp_path, monkeypatch, kind)
    text, convention = SUITES[wording]
    config = load_config(_suite(tmp_path, receiver, text), SuiteRunConfig, ws.pins())
    result = suite._run(config, ws)
    table = pd.read_parquet(ws.artifacts.path(result.outputs["scores"]) / "scores.parquet")
    assert len(table) == 8  # 4 objects x 2 tasks
    assert table["score"].nunique() == len(table)  # the objects and questions are told apart

    runner = YesNo(config.suite)
    renderer = open_renderer(config.receiver)
    ids = runner.readout(renderer).token_ids
    assert ids == (renderer.token_id(convention.yes), renderer.token_id(convention.no))

    reader = EmbeddingReader(ws.artifacts, ws.artifacts.ref(ws.pins()["toy/embeddings"].sha256))
    tasks = dict(open_domain(config.suite.domains[0], ws.artifacts).tasks())
    for _, row in table.iterrows():
        oid = row["object.object"]
        stored = reader.residues([oid], 0)[0] if kind == "xattn" else reader.pooled([oid], 0)[0]
        bare = tasks[row["task"]].question
        prompt, readout = yes_no_prompt(renderer, convention, WORDING.replace("{question}", bare), 1)
        assert readout.token_ids == ids
        assert prompt.input_ids == _suite_prompt(runner, renderer, bare).input_ids
        got = ask_yes_no(receiver, checkpoint, np.array(stored), WORDING.replace("{question}", bare), convention)
        assert (got.logp_yes, got.logp_no, got.score) == (row["logp_yes"], row["logp_no"], row["score"])
        assert (got.yes, got.no, got.yes_id, got.no_id) == (convention.yes, convention.no, *ids)
        assert got.pair_normalized == pytest.approx(got.p_yes / (got.p_yes + got.p_no))


@snapshots.requires("gemma-3-4b-it")
def test_ask_yes_no_reads_the_gemma_suites_tokens():
    """On Gemma 3 4B's tokenizer: the release suite's answer ids are " Yes" 8438 and " No" 2301, the next tokens after
    the prefix "Answer:", and ``yes_no_prompt`` renders that suite's prompt."""
    from examples.ask import yes_no_prompt

    path = ROOT / "configs" / "release" / "suites" / "gemma-3-4b-it-esmc-600m-protein-access-mini-cpu.yaml"
    config = load_config(path, SuiteRunConfig, stand_in_pins(cited_pins(path)))
    snap = snapshots.snapshot("gemma-3-4b-it")
    renderer = open_renderer(config.receiver.model_copy(update={"model_id": str(snap)}))
    runner = YesNo(config.suite)
    convention = load_config(ROOT / "configs" / "answers" / "gemma-3.yaml", AnswerConvention)
    assert config.suite.answer == convention
    question = "Is it a kinase?"
    frame = config.suite.frame
    assert isinstance(frame, Frame)
    prompt, readout = yes_no_prompt(renderer, convention, frame.ask(question), 1)
    assert readout.token_ids == runner.readout(renderer).token_ids == (8438, 2301)
    assert [renderer.tokenizer.decode([i]) for i in readout.token_ids] == [" Yes", " No"]
    assert prompt.input_ids == _suite_prompt(runner, renderer, question).input_ids
    assert renderer.tokenizer.decode(prompt.input_ids[-3:]) == "\nAnswer:"
