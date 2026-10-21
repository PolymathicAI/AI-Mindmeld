"""``latent_interconnect.chat`` on CPU: the tiny Llama, the ``fake-toy`` extractor and fixture pins.

The deployment is real but tiny: a step-50 checkpoint of a toy connector on the tiny receiver, pinned under its run's
destination, the run's cached embeddings for six objects, the toy domain for the rest, and a resident encoder worker
(this interpreter, ``fake-toy``) for live encoding. A second entry's checkpoint pin is missing and a third's checkpoint
is a connector for another receiver: both are refused, the others still serve. The service is given an answer
convention of Gemma's shape (instruction, prefix ``"Answer:"``, answers with a leading space), so it also serves the
Yes/No readout, which must read exactly what ``examples/ask.py --yes-no`` reads.
"""

import contextlib
import json
import math
import socket
import sys
import threading
import time
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from tests.fakes import TOY_OBJECTS, ToyExtractor, commit_toy_domain, make_workspace, toy_recipe
from tests.unit.tiny_receiver import ROOT, build_tiny_receiver

from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.chat import encoding, smoke
from latent_interconnect.chat.config import ChatConfig
from latent_interconnect.chat.encoding import EncoderPool
from latent_interconnect.chat.server import check_bind, create_app
from latent_interconnect.chat.service import ChatService, _yes_no_event
from latent_interconnect.config import Pin, dump_config, load_yaml
from latent_interconnect.data.answers import AnswerConvention
from latent_interconnect.encoders.base import RawObject
from latent_interconnect.encoders.store import EmbeddingReader, commit_embeddings
from latent_interconnect.evaluation.runners import yes_no_prompt
from latent_interconnect.receivers.base import GenerationParams
from latent_interconnect.receivers.render import Conversation, Message, SlotBinding
from latent_interconnect.writes import Connector, ConnectorSpec
from latent_interconnect.writes.checkpoint import commit_checkpoint
from latent_interconnect.writes.init import RidgeLayer, RidgeSolution
from latent_interconnect.writes.inputs import InputSpec

CACHED = ("P01", "P02", "P03", "P04", "P05", "P06")
CONNECTOR = """name: toy-chat
receiver: {receiver}
input: {{recipe: fake-toy, layers: [0]}}
tokens:
  - {{target: name}}
layer0: constant
layers:
  rule: explicit
  per_token:
    - [1, 2]
kind: affine
alpha: {{first: 1.0, init: 0.4}}
layer_choice: {{note: "chat test"}}
"""
RUN = """destination: toy/run
name: toy-chat
connector: {$ref: connector.yaml}
receiver: {$ref: receiver.yaml}
encoder: {$ref: encoder.yaml}
domain: {name: fake-toy, data: {$pin: toy/domain}}
ridge: {name: {$pin: toy/ridge}}
embeddings: [{$pin: toy/embeddings}]
training_set: {$pin: toy/training-set}
mixture:
  - stream: yes_no
    share: 1.0
    params:
      templates: [{id: y1, user: "Here is a protein:<SLOT:x>. {question} Answer Yes or No."}]
      questions: {has_c: ["Does it contain cysteine?"]}
      sampling: uniform
      answers: {"yes": "Y", "no": "N"}
      weights: {family: single_token, floor: 0.1}
steps: 50
batch: 4
device_class: cpu
optimizer: {lr_rel: 0.05, lr_alpha: 0.05, warmup_steps: 2, schedule: constant, weight_decay: 0.0, clip: 1.0}
checkpoints: {resume_every: 25}
"""
CHAT = """receivers:
  tiny:
    label: Tiny Llama
    receiver: {$ref: receiver.yaml}
    context: 256
    connectors:
      toy: {label: Toy, run: {$ref: run.yaml}, step: 50}
      unpinned: {label: Unpinned, run: {$ref: run.yaml}, step: 60}
      foreign: {label: Foreign, run: {$ref: run.yaml}, step: 70, operational_only: true}
inputs:
  fake-toy: {id: toy-id, value: sequence, max_chars: 64}
defaults: {max_tokens: 12, temperature: 0.0}
max_tokens_cap: 64
"""
ANSWER = AnswerConvention(instruction="Answer with only Yes or No.", prefix="Answer:", yes=" Yes", no=" No")
"""Gemma's convention (``configs/answers/gemma-3.yaml``); " Yes" and " No" are one token each of the tiny one."""
QUESTION = "Here is an unknown protein, given to you as an embedding:[[P1]]. Is it an enzyme?"


def _ridge(receiver, recipe, d_model):
    g = np.random.default_rng(0)
    layers = {
        L: RidgeLayer(W=g.normal(size=(d_model, 8)).astype(np.float32), b=np.zeros(d_model, np.float32))
        for L in (0, 1, 2)
    }
    return RidgeSolution(
        target="name",
        mu=np.zeros(8, np.float32),
        sd=np.ones(8, np.float32),
        layers=layers,
        input=InputSpec(recipe="fake-toy", layers=(0,)).identity(),
        recipe_digest=recipe.digest(),
        receiver={"model_id": receiver.model_id, "revision": receiver.revision},
    )


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    """The workspace, the chat config file, and the tiny receiver."""
    root = tmp_path_factory.mktemp("chat")
    receiver = build_tiny_receiver(root / "tiny")
    ws = make_workspace(root / "data")
    store = ws.artifacts
    recipe = toy_recipe().model_copy(update={"plugins": ("tests.fakes",)})  # the worker process registers fake-toy
    dump_config(receiver, root / "receiver.yaml")
    dump_config(recipe, root / "encoder.yaml")
    (root / "connector.yaml").write_text(CONNECTOR.format(receiver="tiny-llama"))
    (root / "run.yaml").write_text(RUN)
    (root / "chat.yaml").write_text(CHAT)
    domain = commit_toy_domain(store)
    extractor = ToyExtractor(recipe.model_dump(mode="json"), "cpu")
    batch = extractor.extract([RawObject(o, TOY_OBJECTS[o][0]) for o in CACHED])
    meta = {"recipe": recipe.model_dump(mode="json"), "fingerprint": dict(extractor.fingerprint())}
    embeddings = commit_embeddings(store, "chat-test-embeddings", [batch], meta)
    pins = {"toy/domain": domain, "toy/embeddings": embeddings, "toy/ridge": domain, "toy/training-set": domain}
    for name, step, receiver_name in (
        ("toy/run/step_000050", 50, "tiny-llama"),
        ("toy/run/step_000070", 70, "kimi-k2.6"),
    ):
        spec = ConnectorSpec.model_validate(load_yaml(CONNECTOR.format(receiver=receiver_name)))
        connector = Connector.from_ridge(spec, receiver.depth, {"name": _ridge(receiver, recipe, receiver.d_model)})
        pins[name] = commit_checkpoint(connector, step, store, encoder=recipe)
    ws.pin_all({n: Pin(target="artifact", sha256=r.sha256) for n, r in pins.items()})
    return ws, root / "chat.yaml", receiver


@pytest.fixture(scope="module")
def service(deployment):
    ws, path, _ = deployment
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYTHONPATH", str(ROOT))  # the worker imports tests.fakes (the recipe's plugin)
        pool = EncoderPool(chat_store(ws), {"lip": sys.executable}, "cpu", timeout_s=300)
        svc = ChatService(ws.load(path, ChatConfig), "tiny", ws, pool, deployment="test-1", answer=ANSWER)
        svc.load()
        yield svc
        pool.close()


def chat_store(ws):
    """Chat's own store, beside (never inside) the workspace's."""
    return ArtifactStore(ws.paths.data.parent / "chat-store")


def body(text, placeholders=None, **kw):
    return {"messages": [{"role": "user", "content": text}], "placeholders": placeholders or {}, **kw}


def run(service, request, rid="r1"):
    return list(service.events(request, rid))


def text_of(events, channel="content"):
    return "".join(e["delta"] for e in events if e["type"] == channel)


def test_the_service_loads_and_reports_the_refused_connectors(service):
    assert service.ready and service.failed is None
    view = {c["id"]: c for c in service.config_view()["connectors"]}
    assert "missing" in view["unpinned"]["error"] and "is a connector for 'kimi-k2.6'" in view["foreign"]["error"]
    assert view["toy"]["status"] in {"cold", "warm"} and view["foreign"]["operational_only"]
    assert service.config_view()["receivers"] == [{"id": "tiny", "label": "Tiny Llama", "deployed": True}]
    assert service.health()["job"] == "test-1" and service.health()["startup"]["chat_open"]


def test_health_counts_the_requests_in_flight_for_the_doors_picker(service):
    assert service.health()["active_requests"] == 0
    service.stop_event("r-flight")  # the server registers a request before it starts it
    assert service.health()["active_requests"] == 1
    events = list(service.events({}, "r-flight"))  # an invalid body: an error event, then the request is gone
    assert events[-1]["type"] == "error" and service.health()["active_requests"] == 0


def test_a_reply_streams_the_tokens_generate_returns(service):
    events = run(
        service,
        body(
            "Here is a protein:[[P1]]. Describe it.",
            {"P1": {"connector": "toy", "input_type": "toy-id", "value": "P02"}},
        ),
    )
    done = events[-1]
    assert done["type"] == "done" and done["injections"]["P1"]["source"] == "cache"
    c = service.connectors["toy"]
    conversation = Conversation((Message("user", "Here is a protein:<SLOT:P1>. Describe it."),))
    prompt = service.renderer.render(conversation, {"P1": SlotBinding("P02", "toy", 1)})
    source, _ = c.checked.source(c.loader, ["P02"], service.ws.artifacts, banked=False)
    (expected,) = service.backend.generate([prompt], {"toy": source}, GenerationParams(max_new_tokens=12))
    assert text_of(events) == expected.text.lstrip()
    assert done["usage"] == {"prompt_tokens": len(prompt.input_ids), "completion_tokens": expected.n_tokens}
    assert sum(e["type"] == "content" for e in events) > 1  # sent as it grew, not in one piece


def test_multi_turn_conversations_render_through_the_template(service):
    request = {
        "messages": [
            {"role": "user", "content": "Here is a protein:[[P1]]."},
            {"role": "assistant", "content": "Noted.", "reasoning": "ignored"},
            {"role": "user", "content": "And [[P2]]? Describe both."},
        ],
        "placeholders": {
            "P1": {"connector": "toy", "input_type": "toy-id", "value": "P01"},
            "P2": {"connector": "toy", "input_type": "toy-id", "value": "P03", "label": "a label"},
        },
    }
    done = run(service, request)[-1]
    assert done["type"] == "done" and set(done["injections"]) == {"P1", "P2"}


def test_a_new_object_is_encoded_live_once_into_an_ordinary_artifact(service):
    sequence = {"P1": {"connector": "toy", "input_type": "sequence", "value": "MKC LLW\nAVA"}}
    first = run(service, body("Describe [[P1]].", sequence))[-1]
    assert first["type"] == "done" and first["injections"]["P1"]["source"] == "live"
    assert service.pool.status("lip") == "warm"
    held = service.connectors["toy"].live
    again = run(service, body("Describe [[P1]].", sequence))[-1]
    assert again["injections"]["P1"] == first["injections"]["P1"] and service.connectors["toy"].live == held
    uncached = run(
        service, body("Describe [[P1]].", {"P1": {"connector": "toy", "input_type": "toy-id", "value": "P10"}})
    )
    assert uncached[-1]["injections"]["P1"]["source"] == "live"  # a domain object the run's embeddings lack


@pytest.mark.parametrize(
    ("placeholder", "match"),
    [
        ({"connector": "toy", "input_type": "toy-id", "value": "P99"}, "not an object"),
        ({"connector": "toy", "input_type": "smiles", "value": "CCO"}, "takes input types"),
        ({"connector": "toy", "input_type": "sequence", "value": "M" * 65}, "at most 64"),
        ({"connector": "toy", "input_type": "sequence", "value": "MK!"}, "unreadable"),  # the encoder skips it
        ({"connector": "toy", "input_type": "sequence", "value": "MK#"}, "encoding failed: RuntimeError: toy encoder"),
        ({"connector": "unpinned", "input_type": "toy-id", "value": "P01"}, "pin 'toy/run/step_000060' is missing"),
        ({"connector": "foreign", "input_type": "toy-id", "value": "P01"}, "is a connector for 'kimi-k2.6'"),
        ({"connector": "nope", "input_type": "toy-id", "value": "P01"}, "unknown connector"),
    ],
)
def test_refusals_are_error_events_never_a_substitute(service, placeholder, match):
    events = run(service, body("Describe [[P1]].", {"P1": placeholder}))
    assert events[-1]["type"] == "error" and match in events[-1]["message"], events[-1]
    assert not any(e["type"] == "content" for e in events)


@pytest.mark.parametrize(
    ("request_body", "match"),
    [
        (body("Describe [[P1]]."), "no entry"),
        (body("<SLOT:x>"), "reserved"),
        (body("Hi", sampling={"max_tokens": 65}), "over the cap"),
        (body("Hi", sampling={"top_p": 0.9}), "invalid request"),
        ({"messages": [{"role": "assistant", "content": "Hi"}]}, "last message"),
    ],
)
def test_malformed_requests_are_refused(service, request_body, match):
    assert match in run(service, request_body)[-1]["message"]


def test_an_envelope_lowers_the_answer_cap_and_refuses_longer_prompts(deployment, service, monkeypatch):
    ws, path, _ = deployment
    capped = ChatService(
        ws.load(path, ChatConfig), "tiny", ws, EncoderPool(chat_store(ws), {}, "cpu"), max_tokens_cap=8
    )
    view = capped.config_view()
    assert view["defaults"]["max_tokens"] == 8 and view["defaults"]["max_tokens_cap"] == 8  # the config's 12 and 64
    assert view["limits"]["max_prompt_tokens"] is None and service.config_view()["defaults"]["max_tokens_cap"] == 64
    short = run(service, body("Hi"))[-1]
    assert short["type"] == "done"
    monkeypatch.setattr(service, "config", capped.config)
    monkeypatch.setattr(service, "max_prompt_tokens", short["usage"]["prompt_tokens"])
    assert run(service, body("Hi"))[-1]["type"] == "done"  # within the envelope: served
    assert "over the cap 8" in run(service, body("Hi", sampling={"max_tokens": 9}))[-1]["message"]
    longer = run(service, body("Hi there, a longer prompt."))[-1]
    assert (
        longer["type"] == "error"
        and "prompt limit of" in longer["message"]
        and "measured envelope" in longer["message"]
    )


def test_a_stop_ends_a_request_before_or_during_generation(service):
    service.stop_event("early").set()  # the server registers a request before it starts it
    events = run(service, body("Describe it."), rid="early")
    assert events[-1]["finish_reason"] == "abort" and not any(e["type"] == "content" for e in events)
    rid, out = "during", []
    for event in service.events(body("Describe it.", sampling={"max_tokens": 64}), rid):
        out.append(event)
        if event["type"] == "content":
            service.stop(rid)
    assert out[-1]["type"] == "done" and out[-1]["finish_reason"] == "abort"
    assert out[-1]["usage"]["completion_tokens"] < 64


def test_closing_the_events_cancels_the_generation(service):
    events = service.events(body("Describe it.", sampling={"max_tokens": 64}), "closed")
    while next(events)["type"] != "content":
        pass
    events.close()
    assert service._generation.acquire(blocking=False)  # the request let the receiver go
    service._generation.release()


class _Watched:
    """A lock that counts the acquires that found it held (a request waiting for the one before)."""

    def __init__(self):
        self.lock, self.waited, self.changed = threading.Lock(), 0, threading.Condition()

    def acquire(self, timeout=-1):
        if self.lock.acquire(blocking=False):
            return True
        with self.changed:
            self.waited += 1
            self.changed.notify_all()
        return self.lock.acquire(timeout=timeout)

    def release(self):
        self.lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *_):
        self.release()

    def waiters(self, n):
        with self.changed:
            return self.changed.wait_for(lambda: self.waited >= n, timeout=30)


@pytest.fixture
def blocked(service, monkeypatch):
    """``begin(lock)``: a first request on ``service`` (with ``lock`` as its whole-request lock, None: none), held in
    generation until released; the placeholder resolutions and streams are logged in order."""
    log, generating, resolving, release = [], threading.Event(), threading.Event(), threading.Event()
    real_stream, real_resolve = service.backend.stream, service.resolve
    threads, results = [], {}

    def stream(prompt, sources, params):
        log.append("generate")
        if not generating.is_set():
            generating.set()
            assert release.wait(30)
        yield from real_stream(prompt, sources, params)
        log.append("generated")

    def resolve(*args):
        log.append("resolve")
        resolving.set()
        return real_resolve(*args)

    def start(rid, request):
        threads.append(threading.Thread(target=lambda: results.__setitem__(rid, run(service, request, rid=rid))))
        threads[-1].start()
        return threads[-1]

    def begin(lock):
        monkeypatch.setattr(service, "_serial", lock)
        start("first", body("Describe it."))
        assert generating.wait(30)
        return SimpleNamespace(log=log, resolving=resolving, release=release, start=start, results=results)

    monkeypatch.setattr(service.backend, "stream", stream)
    monkeypatch.setattr(service, "resolve", resolve)
    yield begin
    release.set()
    for thread in threads:
        thread.join(30)


CACHED_P = {"P1": {"connector": "toy", "input_type": "toy-id", "value": "P02"}}


def test_serial_requests_is_off_unless_asked(deployment):
    ws, path, _ = deployment
    for kw, serial in (({}, False), ({"serial_requests": True}, True)):
        svc = ChatService(ws.load(path, ChatConfig), "tiny", ws, EncoderPool(chat_store(ws), {}, "cpu"), **kw)
        assert (svc._serial is not None) == serial


def test_a_serial_service_serves_one_whole_request_at_a_time(service, blocked):
    watched = _Watched()
    first = blocked(watched)  # the first request holds the lock while it generates
    stopped = first.start("stopped", body("Describe [[P1]].", CACHED_P))
    assert watched.waiters(1)  # waiting for the request before, nothing resolved
    service.stop("stopped")
    stopped.join(30)
    assert first.results["stopped"][-1]["finish_reason"] == "abort" and "resolve" not in first.log
    check = threading.Thread(target=lambda: first.results.__setitem__("check", service.check(CACHED_P["P1"])))
    check.start()
    assert watched.waiters(2) and "resolve" not in first.log  # a placeholder check waits too
    second = first.start("second", body("Describe [[P1]].", CACHED_P))
    assert watched.waiters(3) and "resolve" not in first.log
    first.release.set()  # the first request finishes, then releases the lock
    for thread in (check, second):
        thread.join(30)
    assert first.log[:2] == ["generate", "generated"] and first.log.count("resolve") == 2
    assert first.results["second"][-1]["type"] == "done" and first.results["check"]["ok"]


def test_without_serial_requests_only_generation_waits(service, blocked):
    first = blocked(None)
    second = first.start("second", body("Describe [[P1]].", CACHED_P))
    assert first.resolving.wait(30) and first.log == ["generate", "resolve"]  # resolved while the first generates
    assert service.check(CACHED_P["P1"])["ok"]  # a check is not held up either
    first.release.set()
    second.join(30)
    assert first.log[:3] == ["generate", "resolve", "resolve"] and "generated" in first.log
    assert first.results["second"][-1]["type"] == "done"


def test_the_entry_point_passes_serial_requests_only_when_given(monkeypatch, tmp_path):
    from latent_interconnect.chat import __main__ as entry

    class BuiltError(Exception):
        pass

    def built(*args, **kw):
        raise BuiltError(kw["serial_requests"], pools[-1])

    ws = SimpleNamespace(paths=SimpleNamespace(data=tmp_path), load=lambda path, kind: None)
    monkeypatch.setattr(entry.Workspace, "from_env", staticmethod(lambda: ws))
    pools = []
    monkeypatch.setattr(entry, "EncoderPool", lambda *a, **kw: pools.append(kw.get("max_resident")))
    monkeypatch.setattr(entry, "ChatService", built)
    args = ["--config", "c.yaml", "--receiver", "r", "--host", "127.0.0.1", "--port", "1", "--no-auth"]
    for extra, serial, limit in (([], False, None), (["--serial-requests", "--max-resident-encoders", "1"], True, 1)):
        with pytest.raises(BuiltError) as err:
            entry.main([*args, *extra])
        assert err.value.args == (serial, limit)


# --- the Yes/No readout ---------------------------------------------------------------------------------------------


def yes_no_of(events):
    (event,) = [e for e in events if e["type"] == "yes_no"]
    assert events[-1]["type"] == "done" and events[-1]["finish_reason"] == "stop", events[-1]
    return event


def test_a_yes_no_request_reads_exactly_what_ask_yes_no_reads(deployment, service):
    from examples.ask import ask_yes_no

    ws, path, _ = deployment
    events = run(service, body(QUESTION, CACHED_P, readout="yes_no"))
    got = yes_no_of(events)
    done = events[-1]
    assert done["usage"]["completion_tokens"] == 0 and done["injections"]["P1"]["source"] == "cache"
    assert not any(e["type"] in {"content", "reasoning"} for e in events)
    store = ws.artifacts
    checkpoint = store.path(store.ref(ws.pins()["toy/run/step_000050"].sha256))
    vector = EmbeddingReader(store, store.ref(ws.pins()["toy/embeddings"].sha256)).pooled(["P02"], 0)[0]
    asked = ask_yes_no(
        path.parent / "receiver.yaml", checkpoint, vector, QUESTION.replace("[[P1]]", "<SLOT:x>"), ANSWER
    )
    assert (got["logp_yes"], got["logp_no"], got["score"]) == (asked.logp_yes, asked.logp_no, asked.score)
    assert (got["yes"], got["no"], got["yes_id"], got["no_id"]) == (asked.yes, asked.no, asked.yes_id, asked.no_id)
    assert got["pair_normalized"] == asked.pair_normalized
    assert (got["instruction"], got["prefix"]) == (ANSWER.instruction, ANSWER.prefix)
    # the placeholder's name does not reach the prompt: ask.py's slot x and the chat's P1 render the same ids
    chat, _ = yes_no_prompt(
        service.renderer, ANSWER.policy(), QUESTION.replace("[[P1]]", "<SLOT:P1>"), {"P1": SlotBinding("P02", "toy", 1)}
    )
    alone, _ = yes_no_prompt(
        service.renderer, ANSWER.policy(), QUESTION.replace("[[P1]]", "<SLOT:x>"), {"x": SlotBinding("x", "x", 1)}
    )
    assert chat.input_ids == alone.input_ids and done["usage"]["prompt_tokens"] == len(chat.input_ids)


def test_a_yes_no_request_scores_its_last_turn_alone(service):
    alone = yes_no_of(run(service, body(QUESTION, CACHED_P, readout="yes_no")))
    request = {
        "messages": [
            {"role": "user", "content": "Describe [[P2]]."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": QUESTION},
        ],
        "placeholders": {**CACHED_P, "P2": {"connector": "toy", "input_type": "toy-id", "value": "P03"}},
        "system": "Answer briefly.",
        "thinking": True,
        "sampling": {"temperature": 1.0, "max_tokens": 64},
        "readout": "yes_no",
    }
    events = run(service, request)
    assert yes_no_of(events) == alone  # no earlier turn, system message, thinking or sampling reaches the score
    assert set(events[-1]["injections"]) == {"P1"}  # only the scored turn's placeholders are bound
    assert "last message" in run(service, {**request, "messages": request["messages"][:2]})[-1]["message"]


def test_a_generation_after_a_score_sends_the_history_without_the_scored_turn(service):
    yes_no_of(run(service, body(QUESTION, CACHED_P, readout="yes_no")))
    history = [  # the UI's history (static/history.js): the scored turn and its card are both left out
        {"role": "user", "content": "Here is a protein:[[P1]]. Describe it."},
        {"role": "assistant", "content": "A protein."},
        {"role": "user", "content": "Is [[P1]] an enzyme?"},
    ]
    done = run(service, {"messages": history, "placeholders": CACHED_P})[-1]
    assert done["type"] == "done" and done["usage"]["completion_tokens"] > 0


def test_a_yes_no_request_waits_for_a_running_generation(service, blocked, monkeypatch):
    watched = _Watched()
    monkeypatch.setattr(service, "_generation", watched)
    first = blocked(None)
    real = service.scorer.score
    monkeypatch.setattr(service.scorer, "score", lambda *a: first.log.append("score") or real(*a))
    scored = first.start("scored", body(QUESTION, CACHED_P, readout="yes_no"))
    assert watched.waiters(1) and "score" not in first.log  # resolved, then waiting for the receiver
    first.release.set()
    scored.join(30)
    assert first.log.index("generated") < first.log.index("score")
    yes_no_of(first.results["scored"])


def test_a_stopped_yes_no_request_scores_nothing(service, monkeypatch):
    service.stop_event("early-score").set()
    events = run(service, body(QUESTION, CACHED_P, readout="yes_no"), rid="early-score")
    assert events[-1]["finish_reason"] == "abort" and not any(e["type"] == "yes_no" for e in events)
    real = service.scorer.score

    def stopped_during(*args):
        service.stop("during-score")  # the stop arrives while the forward pass runs
        return real(*args)

    monkeypatch.setattr(service.scorer, "score", stopped_during)
    events = run(service, body(QUESTION, CACHED_P, readout="yes_no"), rid="during-score")
    assert events[-1]["finish_reason"] == "abort" and not any(e["type"] == "yes_no" for e in events)


@pytest.mark.parametrize("question", ["", "   \n\t"])
def test_an_empty_yes_no_question_is_refused(service, question):
    events = run(service, body(question, readout="yes_no"))
    assert events[-1]["type"] == "error" and events[-1]["message"] == "the Yes/No question is empty"


def test_yes_no_is_offered_only_with_a_convention_and_a_receiver_scored_by_its_generation_engine(deployment, service):
    ws, path, _ = deployment
    assert service.config_view()["yes_no"] == ANSWER.model_dump()
    config = ws.load(path, ChatConfig)
    without = ChatService(config, "tiny", ws, EncoderPool(chat_store(ws), {}, "cpu"))
    without.load()
    assert without.config_view()["yes_no"] is None and without.scorer is None
    assert "without an answer convention" in run(without, body(QUESTION, CACHED_P, readout="yes_no"))[-1]["message"]
    # Kimi K2.6's backends: scored by kimi-stream, generated by kimi-vllm (another engine): not offered
    kimi = load_yaml((ROOT / "configs" / "receivers" / "kimi-k2.6.yaml").read_text())["backends"]
    assert kimi["score"] != kimi["generate"]
    entry = config.receivers["tiny"]
    other = entry.model_copy(update={"receiver": entry.receiver.model_copy(update={"backends": kimi})})
    hidden = ChatService(
        config.model_copy(update={"receivers": {"tiny": other}}), "tiny", ws, without.pool, answer=ANSWER
    )
    assert not hidden.yes_no_offered and hidden.config_view()["yes_no"] is None


def test_an_answer_that_is_not_one_token_fails_the_load(deployment):
    ws, path, _ = deployment
    two = ANSWER.model_copy(update={"yes": " Yes indeed"})
    svc = ChatService(ws.load(path, ChatConfig), "tiny", ws, EncoderPool(chat_store(ws), {}, "cpu"), answer=two)
    svc.load()
    assert not svc.ready and "Yes indeed" in svc.failed


def test_the_yes_no_event_never_holds_a_non_finite_json_number():
    import torch

    policy = ANSWER.policy()
    zero_yes = _yes_no_event("r", policy, (1, 2), torch.tensor([[-math.inf, -0.5]], dtype=torch.float64))
    assert (zero_yes["logp_yes"], zero_yes["logp_no"], zero_yes["score"], zero_yes["pair_normalized"]) == (
        None,
        -0.5,
        None,
        0.0,
    )
    with np.errstate(invalid="ignore"):  # -inf - -inf: the score is NaN, so null
        both = _yes_no_event("r", policy, (1, 2), torch.tensor([[-math.inf, -math.inf]]))
    assert both["pair_normalized"] is None
    far = _yes_no_event("r", policy, (1, 2), torch.tensor([[-720.0, -0.5]], dtype=torch.float64))
    assert far["score"] == -719.5 and 0.0 < far["pair_normalized"] < 1e-300  # exp(719.5) would overflow
    for event in (zero_yes, both, far):
        json.dumps(event, allow_nan=False)


def test_the_entry_point_passes_the_answer_convention(monkeypatch, tmp_path):
    from latent_interconnect.chat import __main__ as entry

    class BuiltError(Exception):
        pass

    def built(*args, **kw):
        raise BuiltError(kw["answer"])

    ws = SimpleNamespace(paths=SimpleNamespace(data=tmp_path), load=lambda path, kind: None)
    monkeypatch.setattr(entry.Workspace, "from_env", staticmethod(lambda: ws))
    monkeypatch.setattr(entry, "EncoderPool", lambda *a, **kw: None)
    monkeypatch.setattr(entry, "ChatService", built)
    args = ["--config", "c.yaml", "--receiver", "r", "--host", "127.0.0.1", "--port", "1", "--no-auth"]
    gemma = ROOT / "configs" / "answers" / "gemma-3.yaml"
    for extra, answer in (
        ([], None),
        (["--yes-no", str(gemma)], AnswerConvention.model_validate(load_yaml(gemma.read_text()))),
    ):
        with pytest.raises(BuiltError) as err:
            entry.main([*args, *extra])
        assert err.value.args == (answer,)


# --- HTTP ------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(service):
    from fastapi.testclient import TestClient

    return TestClient(create_app(service, "pw"))


def sse(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_every_api_route_but_health_and_login_needs_the_password(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/config").status_code == 401
    assert client.get("/api/config", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/api/login", json={"token": "wrong"}).status_code == 401
    assert client.get("/api/config", headers={"Authorization": "Bearer pw"}).json()["model"] == "Tiny Llama"
    login = client.post("/api/login", json={"token": "pw"})
    assert login.status_code == 200 and "chat_session" in login.cookies
    assert client.get("/api/config").status_code == 200  # the session cookie
    client.post("/api/logout")
    client.cookies.clear()
    assert client.get("/api/config").status_code == 401
    assert client.get("/").status_code == 200 and "AI Mindmeld" in client.get("/").text


def test_chat_is_served_as_server_sent_events(client):
    auth = {"Authorization": "Bearer pw"}
    response = client.post("/api/chat", json=body("Describe it."), headers=auth)
    events = sse(response)
    assert response.headers["content-type"].startswith("text/event-stream")
    assert events[0]["type"] == "status" and events[-1]["type"] == "done" and text_of(events)
    assert client.post("/api/chat", content=b"[1]", headers=auth).status_code == 400
    assert client.post("/api/stop", json={"request_id": "bad id!"}, headers=auth).status_code == 400
    check = {"connector": "toy", "input_type": "toy-id", "value": "P04"}
    assert client.post("/api/placeholder/check", json=check, headers=auth).json()["info"]["source"] == "cache"


def test_a_yes_no_request_is_served_as_server_sent_events(client, service):
    response = client.post(
        "/api/chat", json=body(QUESTION, CACHED_P, readout="yes_no"), headers={"Authorization": "Bearer pw"}
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "Infinity" not in response.text and "NaN" not in response.text
    events = sse(response)
    assert [e["type"] for e in events] == ["status", "status", "yes_no", "done"]  # streamed: no reply deltas
    assert [e["stage"] for e in events[:2]] == ["embedding", "prefill"]
    sent = {k: v for k, v in yes_no_of(events).items() if k != "request_id"}
    direct = {
        k: v
        for k, v in yes_no_of(run(service, body(QUESTION, CACHED_P, readout="yes_no"))).items()
        if k != "request_id"
    }
    assert sent == direct
    assert client.get("/api/config", headers={"Authorization": "Bearer pw"}).json()["yes_no"] == ANSWER.model_dump()


def test_no_auth_is_loopback_only_and_a_wildcard_bind_is_refused():
    check_bind("127.0.0.1", True)
    check_bind("10.0.0.5", False)
    for host, no_auth in (("10.0.0.5", True), ("0.0.0.0", False), ("::", False)):
        with pytest.raises(ValueError):
            check_bind(host, no_auth)


def test_a_config_refuses_a_run_of_another_receiver(deployment, tmp_path):
    ws, path, receiver = deployment
    other = receiver.model_copy(update={"name": "other"})
    dump_config(other, path.parent / "other.yaml")
    text = path.read_text().replace("receiver: {$ref: receiver.yaml}", "receiver: {$ref: other.yaml}", 1)
    (path.parent / "other-chat.yaml").write_text(text)
    with pytest.raises(ValueError, match="connectors of runs on other receivers"):
        ws.load(path.parent / "other-chat.yaml", ChatConfig)


def test_the_service_stays_responsive_while_the_receiver_loads(deployment):
    ws, path, _ = deployment
    pool = EncoderPool(chat_store(ws), {}, "cpu")
    svc = ChatService(ws.load(path, ChatConfig), "tiny", ws, pool)
    assert not svc.ready and svc.health()["startup"]["phases"][0]["status"] == "pending"
    assert "still loading" in run(svc, body("Hi"))[-1]["message"]
    loader = threading.Thread(target=svc.load)
    loader.start()
    loader.join()
    assert svc.ready
    assert (
        "no interpreter"
        in run(svc, body("[[P]]", {"P": {"connector": "toy", "input_type": "sequence", "value": "MKA"}}))[-1]["message"]
    )


def _tree(root):
    return {(str(p.relative_to(root)), p.stat().st_mtime_ns) for p in root.rglob("*")}


def test_chat_never_writes_to_the_production_store(service):
    production, before = service.ws.artifacts.root, _tree(service.ws.artifacts.root)
    live = {"P1": {"connector": "toy", "input_type": "sequence", "value": "MWWKLC"}}
    assert run(service, body("Describe [[P1]].", live))[-1]["injections"]["P1"]["source"] == "live"
    c = service.connectors["toy"]
    _, bank = c.checked.source(c.loader, ["P01", "chat-" + _sha("MWWKLC")], service.pool.store, banked=True)
    assert _tree(production) == before
    assert bank is not None and service.pool.store.exists(bank.sha256) and not service.ws.artifacts.exists(bank.sha256)


def _sha(value):
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()[:16]


@pytest.mark.parametrize("where", ["same", "inside production", "around production"])
def test_a_chat_store_overlapping_the_production_store_is_refused(deployment, where):
    ws, path, _ = deployment
    root = ws.artifacts.root
    store = {"same": root, "inside production": root / "chat", "around production": root.parent}[where]
    with pytest.raises(ValueError, match="overlap"):
        ChatService(ws.load(path, ChatConfig), "tiny", ws, EncoderPool(ArtifactStore(store), {}, "cpu"))


def test_a_max_model_len_below_the_context_fails_before_any_load(deployment):
    ws, path, _ = deployment
    with pytest.raises(ValueError, match="below the config's context 256"):
        ChatService(
            ws.load(path, ChatConfig),
            "tiny",
            ws,
            EncoderPool(chat_store(ws), {}, "cpu"),
            options={"max_model_len": "100"},
        )


def test_readiness_is_refused_below_the_configs_context(deployment, monkeypatch):
    import latent_interconnect.chat.service as module

    ws, path, _ = deployment
    real = module.open_generation

    def short(spec, **kw):
        backend = real(spec, **kw)
        backend.max_model_len = 100  # an engine limit below the config's 256
        return backend

    monkeypatch.setattr(module, "open_generation", short)
    svc = ChatService(ws.load(path, ChatConfig), "tiny", ws, EncoderPool(chat_store(ws), {}, "cpu"))
    svc.load()
    assert not svc.ready and "below the config's context 256" in svc.failed


def test_a_stop_for_an_unknown_request_leaves_nothing_behind(service):
    service.stop("never-ran")
    assert "never-ran" not in service._stops


def test_an_oversized_chunked_body_is_413(client):
    chunks = iter([b'{"x": "' + b"a" * (1 << 20), b"a" * (1 << 20), b"a" * 1024 + b'"}'])
    response = client.post("/api/chat", content=chunks, headers={"Authorization": "Bearer pw"})
    assert response.status_code == 413


# --- the smoke client (chat.smoke) against a served deployment ------------------------------------------------------

SMOKE = {
    "inputs": {"fake-toy": {"id": "P02", "value": "MKCLLW"}},
    "prompts": {"fake-toy": "Here is a protein:[[P1]]. Describe it."},
    "sampling": {"temperature": 0.0, "max_tokens": 8},
    "ready_timeout_s": 30,
    "request_timeout_s": 120,
}


@contextlib.contextmanager
def serving(app):
    """Serve ``app`` on a free loopback port in a thread; yield its URL."""
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(30)


def test_the_smoke_client_passes_a_working_deployment_through_its_entry_point(deployment, tmp_path):
    ws, path, _ = deployment
    text = path.read_text()
    only_toy = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("unpinned:", "foreign:")))
    (path.parent / "toy-only.yaml").write_text(only_toy + "\n")
    pool = EncoderPool(chat_store(ws), {"lip": sys.executable}, "cpu", timeout_s=300)
    svc = ChatService(ws.load(path.parent / "toy-only.yaml", ChatConfig), "tiny", ws, pool, deployment="smoke-1")
    threading.Thread(target=svc.load, daemon=True).start()  # the client waits for readiness
    (tmp_path / "smoke.yaml").write_text(json.dumps(SMOKE))
    (tmp_path / "pw").write_text("pw\n")
    receipt = tmp_path / "out" / "receipt.json"
    with pytest.MonkeyPatch.context() as mp, serving(create_app(svc, "pw")) as url:
        mp.setenv("PYTHONPATH", str(ROOT))  # the encoder worker imports tests.fakes
        args = ["--config", str(tmp_path / "smoke.yaml"), "--url", url, "--out", str(receipt)]
        status = smoke.main([*args, "--password-file", str(tmp_path / "pw")])
    pool.close()
    out = json.loads(receipt.read_text())
    assert status == 0 and out["passed"] and out["failed"] == [], out.get("failed")
    access, cached, raw = out["steps"]
    assert access["step"] == "access" and access["no_credentials"] == 401 and access["with_cookie"] == 200
    assert (cached["path"], cached["events"][-1]["injections"]["P1"]["source"]) == ("cached", "cache")
    assert (raw["path"], raw["events"][-1]["injections"]["P1"]["source"]) == ("raw", "live")
    assert raw["text"] and out["health_at_ready"]["job"] == "smoke-1"


def test_the_smoke_client_runs_every_request_and_fails_on_any_failure(service, tmp_path):
    cfg = smoke.SmokeConfig.model_validate(SMOKE | {"order": ["foreign", "missing-id"]})
    (tmp_path / "pw").write_text("wrong\n")
    with serving(create_app(service, "pw")) as url:
        assert smoke.run_smoke(cfg, url, tmp_path / "pw", tmp_path / "bad-password.json") == 1
        (tmp_path / "pw").write_text("pw\n")
        status = smoke.run_smoke(cfg, url, tmp_path / "pw", tmp_path / "receipt.json")
    out = json.loads((tmp_path / "receipt.json").read_text())
    assert status == 1 and not out["passed"]
    assert out["failed"] == ["chat foreign cached", "chat unpinned cached", "chat foreign raw", "chat unpinned raw"]
    assert [s.get("path") for s in out["steps"][1:]] == ["cached"] * 3 + ["raw"] * 3  # none skipped after a failure
    assert "is a connector for 'kimi-k2.6'" in out["steps"][1]["why"]
    assert "HTTP 401" in json.loads((tmp_path / "bad-password.json").read_text())["why"]


def test_the_smoke_client_stops_waiting_when_its_server_exits(tmp_path):
    cfg = smoke.SmokeConfig.model_validate(SMOKE | {"ready_timeout_s": 600})
    port = _free_port()
    t0 = time.monotonic()
    status = smoke.run_smoke(
        cfg,
        f"http://127.0.0.1:{port}",
        None,
        tmp_path / "receipt.json",
        server=[sys.executable, "-c", "import sys; sys.exit(7)"],
        revision="abc123",
    )
    out = json.loads((tmp_path / "receipt.json").read_text())
    assert status == smoke.NOT_READY and time.monotonic() - t0 < 60  # not the 600 s ready wait
    assert out["why"] == "server exited (rc=7) before it became ready" and out["server"]["exit"] == 7
    assert out["revision"] == "abc123" and out["steps"] == []


@pytest.mark.parametrize(("ignores_term", "signal_number"), [(False, 15), (True, 9)])
def test_the_smoke_client_stops_the_server_it_started_within_a_bound(
    tmp_path, monkeypatch, ignores_term, signal_number
):
    monkeypatch.setattr(smoke, "STOP_GRACE_S", 1.0)
    handler = "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignores_term else ""
    cfg = smoke.SmokeConfig.model_validate(SMOKE | {"ready_timeout_s": 1})
    status = smoke.run_smoke(
        cfg,
        f"http://127.0.0.1:{_free_port()}",
        None,
        tmp_path / "receipt.json",
        server=[sys.executable, "-c", handler + "import time; time.sleep(600)"],
    )
    out = json.loads((tmp_path / "receipt.json").read_text())
    assert status == smoke.NOT_READY and out["why"] == "the deployment did not become ready"
    assert out["server"]["exit"] == -signal_number


def test_a_smoke_usage_error_is_not_the_never_ready_status(tmp_path):
    with pytest.raises(SystemExit) as usage:
        smoke.main(["--config", str(tmp_path / "smoke.yaml"), "--url", "http://127.0.0.1:1", "--out", "r.json"])
    assert usage.value.code == 2 != smoke.NOT_READY  # neither --password-file nor --no-auth


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_the_smoke_client_reports_a_deployment_that_never_becomes_ready(deployment, tmp_path):
    ws, path, _ = deployment
    svc = ChatService(ws.load(path, ChatConfig), "tiny", ws, EncoderPool(chat_store(ws), {}, "cpu"))  # never loaded
    cfg = smoke.SmokeConfig.model_validate(SMOKE | {"ready_timeout_s": 1})
    with serving(create_app(svc, None, no_auth=True)) as url:
        assert smoke.run_smoke(cfg, url, None, tmp_path / "receipt.json") == smoke.NOT_READY
    out = json.loads((tmp_path / "receipt.json").read_text())
    assert not out["passed"] and out["health_at_ready"]["ready"] is False and out["steps"] == []


# --- the encoder residency limit (a local deployment: max_resident) --------------------------------------------------


class _FakeWorker:
    """A worker without a process: logs its runs and stops; a run can block on ``gate`` or fail."""

    log: ClassVar[list[tuple[str, str]]] = []
    fail: ClassVar[set[str]] = set()
    gate: ClassVar[threading.Event | None] = None

    def __init__(self, python, device, environment=""):
        self.environment, self.users, self.status, self.name = environment, 0, "cold", None

    def run(self, job, timeout_s):
        self.name = job.recipe["name"]
        _FakeWorker.log.append(("run", self.name))
        if _FakeWorker.gate is not None:
            assert _FakeWorker.gate.wait(30)
        if self.name in _FakeWorker.fail:
            self.status = "error"
            raise encoding.EncoderError(f"{self.name} failed")
        self.status = "warm"
        return "0" * 64

    def close(self):
        _FakeWorker.log.append(("close", self.name))
        self.status = "cold"


@pytest.fixture
def fake_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(encoding, "_Worker", _FakeWorker)
    monkeypatch.setattr(_FakeWorker, "log", [])
    monkeypatch.setattr(_FakeWorker, "fail", set())
    monkeypatch.setattr(_FakeWorker, "gate", None)
    pythons = {"esmc": "py", "smallmol": "py"}
    return lambda **kw: EncoderPool(ArtifactStore(tmp_path / "store"), pythons, "cpu", **kw)


def recipe(name, environment="esmc"):
    return toy_recipe().model_copy(update={"name": name, "environment": environment})


def encode(pool, name, environment="esmc"):
    return pool.encode(recipe(name, environment), [RawObject("x", "MKV")])


def test_without_a_limit_one_worker_per_environment_stays_resident(fake_pool):
    pool = fake_pool()
    for name, env in (("esmc-600m", "esmc"), ("pglm-1b", "esmc"), ("chemfm-1b", "smallmol"), ("esmc-600m", "esmc")):
        encode(pool, name, env)
    assert [e for e in _FakeWorker.log if e[0] == "close"] == []  # the cluster behavior: nothing is stopped
    assert len(pool._workers) == 2 and pool.status("esmc") == pool.status("smallmol") == "warm"


def test_the_least_recently_used_idle_worker_is_stopped_before_another_starts(fake_pool):
    pool = fake_pool(max_resident=2)
    for name in ("a", "b", "a", "c"):  # c needs room: b is the least recently used
        encode(pool, name)
    assert _FakeWorker.log == [("run", "a"), ("run", "b"), ("run", "a"), ("close", "b"), ("run", "c")]
    one = fake_pool(max_resident=1)
    _FakeWorker.log.clear()
    encode(one, "a", "smallmol")
    assert one.warm(recipe("a", "smallmol")) and one.status("smallmol") == "warm"
    encode(one, "b", "smallmol")  # one environment, two recipes: two workers, never both up
    assert _FakeWorker.log == [("run", "a"), ("close", "a"), ("run", "b")]
    assert not one.warm(recipe("a", "smallmol")) and one.warm(recipe("b", "smallmol"))


def test_a_worker_serving_a_job_is_never_stopped(fake_pool):
    pool = fake_pool(max_resident=1)
    _FakeWorker.gate = threading.Event()
    first = threading.Thread(target=encode, args=(pool, "a"))
    first.start()
    while ("run", "a") not in _FakeWorker.log:
        time.sleep(0.01)
    second = threading.Thread(target=encode, args=(pool, "b"))
    second.start()
    time.sleep(0.3)
    assert _FakeWorker.log == [("run", "a")]  # b waits: a is in use
    _FakeWorker.gate.set()
    first.join(30)
    second.join(30)
    assert _FakeWorker.log == [("run", "a"), ("close", "a"), ("run", "b")]


def test_a_job_that_finds_no_free_slot_in_time_is_an_encoder_error(fake_pool):
    pool = fake_pool(max_resident=1, timeout_s=0.3)
    _FakeWorker.gate = threading.Event()
    first = threading.Thread(target=encode, args=(pool, "a"))
    first.start()
    while ("run", "a") not in _FakeWorker.log:
        time.sleep(0.01)
    with pytest.raises(encoding.EncoderError, match="no encoder slot freed"):
        encode(pool, "b")
    _FakeWorker.gate.set()
    first.join(30)
    assert ("close", "a") not in _FakeWorker.log


def test_a_failed_job_releases_its_worker_and_the_next_recipe_still_runs(fake_pool):
    pool = fake_pool(max_resident=1)
    _FakeWorker.fail = {"a"}
    with pytest.raises(encoding.EncoderError, match="a failed"):
        encode(pool, "a")
    assert pool.status("esmc") == "error" and not pool.warm(recipe("a"))
    encode(pool, "b")  # the failed worker is idle, so it is stopped and b starts
    assert _FakeWorker.log == [("run", "a"), ("close", "a"), ("run", "b")] and pool.status("esmc") == "warm"
    with pytest.raises(ValueError, match="at least 1"):
        fake_pool(max_resident=0)
