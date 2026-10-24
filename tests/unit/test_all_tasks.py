"""All-tasks training sets (``TrainingSetSpec.all_tasks``) on synthetic domains: the build screens nothing, a screened
set keeps its identity and stored form, the construction check has no zero-shot task to refuse, streams train held-out
tasks only on an all-tasks text, and the molecule domain's training scope drops the unstated subset alone."""

import dataclasses
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from tests.fakes import TOY_OBJECTS, ToyDomain, commit_toy_domain

from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.build.zero_shot import (
    DetectorSpec,
    TrainingSetSpec,
    build_training_set,
    check_construction,
)
from latent_interconnect.config import Config
from latent_interconnect.data.items import commit_item_set, object_list
from latent_interconnect.data.sampler import MixtureEntry
from latent_interconnect.data.streams import STREAMS, ScopedDomain, Supervision, build_stream
from latent_interconnect.data.training_text import TrainingText, load_training_text
from latent_interconnect.domains.base import DomainSpec, Frame, TaskKind, TaskScope, TaskSpec, open_domain
from latent_interconnect.domains.tables import DomainTables, commit_domain_data
from latent_interconnect.domains.views import DomainCatalog

IDS = pd.Index(["P1", "P2", "P3", "P4", "P5"], name="object_id")
FUNGI = {"terms": [r"\bfung(?:us|i|al)\b"]}
NOTHING_HELD_OUT = {"zero_shot_tasks": (), "detectors": (), "all_tasks": True}
"""A screen of no zero-shot task, on an all-tasks training set."""


def protein_domain(store):
    """Five proteins (P5 in ``test``): one trained Yes/No task, one held-out, a caption and a keyword list."""
    tasks = (
        TaskSpec(id="Kinase", kind=TaskKind.YES_NO, question="Is it a kinase?", label="kw:Kinase", scope="trained"),
        TaskSpec(id="Fungi", kind=TaskKind.YES_NO, question="Is it fungal?", label="lin:Fungi", scope="held_out"),
        TaskSpec(id="caption", kind=TaskKind.TEXT, question="", label="caption", scope="trained"),
    )
    tables = DomainTables(
        domain="protein",
        objects=pd.DataFrame({"value": ["MA"] * 5, "name": list(IDS)}, index=IDS),
        split=pd.Series(["train"] * 4 + ["test"], index=IDS),
        clusters=pd.Series(list(IDS), index=IDS),
        labels=pd.DataFrame(
            {"kw:Kinase": [False, False, True, False, True], "lin:Fungi": [True, False, False, False, False]}, index=IDS
        ),
        fields=pd.DataFrame(
            {
                "caption": ["A fungal kinase.", "A toxin.", "Binds ATP.", "A long kinase.", "x"],
                "keywords": ["Repeat;Fungi", "Toxin", "Kinase", "Repeat", None],
                "secret": ["s"] * 5,
            },
            index=IDS,
        ),
        tasks=tasks,
        frames=(Frame(id="ask", user="Is<SLOT:x> it? {question}"),),
        recipes=(),
    )
    spec = DomainSpec(name="protein", data=commit_domain_data(store, tables, {}).sha256)
    return spec, open_domain(spec, store)


def training_spec(store, domain_spec, domain, **update):
    train = commit_item_set(store, object_list("train", domain, ("P1", "P2", "P3", "P4"))).sha256
    screen = {
        "fields": ("caption", "keywords"),
        "list_fields": {"keywords": ";"},
        "property_list_fields": ("keywords",),
        "zero_shot_tasks": ("Fungi",),
        "detectors": (DetectorSpec(task="Fungi", detector="lexical", params=FUNGI),),
    }
    top = {k: update.pop(k) for k in ("screened_elements", "all_tasks") if k in update}
    return TrainingSetSpec.model_validate(
        {"domain": domain_spec, "train_ids": train, "screen": {**screen, **update}, **top}
    )


def test_an_all_tasks_build_holds_no_task_out_and_removes_nothing(tmp_path):
    store = ArtifactStore(tmp_path)
    domain_spec, domain = protein_domain(store)
    spec = training_spec(store, domain_spec, domain, **NOTHING_HELD_OUT)
    ref = build_training_set(spec, store)
    text = load_training_text(store, ref)
    assert text.all_tasks and text.removed == {}
    assert text.fields == {  # every field the screen names, as the domain holds it (P5 is no training object)
        "P1": {"caption": "A fungal kinase.", "keywords": "Repeat;Fungi"},
        "P2": {"caption": "A toxin.", "keywords": "Toxin"},
        "P3": {"caption": "Binds ATP.", "keywords": "Kinase"},
        "P4": {"caption": "A long kinase.", "keywords": "Repeat"},
    }
    meta = store.manifest(ref).meta
    assert meta["all_tasks"] is True and meta["spec"]["all_tasks"] is True
    assert TrainingSetSpec.model_validate(meta["spec"]) == spec


def test_a_screened_set_keeps_its_identity_and_stored_form(tmp_path):
    """At False, ``all_tasks`` is in no digest, stored spec or text."""
    store = ArtifactStore(tmp_path)
    domain_spec, domain = protein_domain(store)
    spec = training_spec(store, domain_spec, domain)
    assert "all_tasks" not in spec.identity() and spec == training_spec(store, domain_spec, domain, all_tasks=False)
    ref = build_training_set(spec, store)
    text, meta = load_training_text(store, ref), store.manifest(ref).meta
    assert "all_tasks" not in meta and "all_tasks" not in meta["spec"] and not text.all_tasks
    assert text.fields["P1"] == {"keywords": "Repeat"} and text.removed == {"Fungi": {"caption": 1}}  # still screened


def test_train_ids_may_name_a_training_set_of_the_same_domain(tmp_path):
    """An all-tasks set on a screened set's objects: ``train_ids`` names that set (released sets ship no object list),
    so the two hold the same objects in the same order; a training set of another domain is refused."""
    store = ArtifactStore(tmp_path)
    domain_spec, domain = protein_domain(store)
    screened = build_training_set(training_spec(store, domain_spec, domain), store)
    spec = training_spec(store, domain_spec, domain, **NOTHING_HELD_OUT)
    spec = spec.model_copy(update={"train_ids": screened.sha256})
    all_tasks = load_training_text(store, build_training_set(spec, store))
    assert all_tasks.object_ids == load_training_text(store, screened).object_ids and all_tasks.all_tasks
    assert all_tasks.fields["P1"] == {"caption": "A fungal kinase.", "keywords": "Repeat;Fungi"}  # unscreened
    molecule_spec, *_ = molecule_domain(store)
    with pytest.raises(ValueError, match="its objects cannot be this training set's"):
        build_training_set(spec.model_copy(update={"domain": molecule_spec}), store)


def test_only_an_all_tasks_set_has_a_screen_of_no_zero_shot_task(tmp_path):
    store = ArtifactStore(tmp_path)
    domain_spec, domain = protein_domain(store)
    with pytest.raises(ValueError, match="names no zero-shot task; only an all-tasks"):
        training_spec(store, domain_spec, domain, **{**NOTHING_HELD_OUT, "all_tasks": False})
    with pytest.raises(ValueError, match="an all-tasks training set holds no task out"):
        training_spec(store, domain_spec, domain, all_tasks=True)  # the screen still names a zero-shot task
    with pytest.raises(ValueError, match="an all-tasks training set holds no task out"):
        training_spec(store, domain_spec, domain, **NOTHING_HELD_OUT, screened_elements={"keywords": ("Fungi",)})


class _Params(Config):
    tasks: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()


class _Declared:
    """A test stream that declares what its params say it trains."""

    Params: ClassVar[type[Config]] = _Params

    def __init__(self, domain, params, text):
        self.params = params

    def sample(self, rng: np.random.Generator, n: int):
        raise NotImplementedError

    @classmethod
    def declare(cls, domain, params):
        return Supervision(frozenset(params.tasks), frozenset(), frozenset(params.fields), ())

    def supervision(self):
        return self.declare(None, self.params)


STREAMS.add("all-tasks-test-declared", _Declared)


def test_the_construction_check_passes_held_out_tasks_on_an_all_tasks_set(tmp_path):
    """Nothing is zero-shot on an all-tasks set; a field it does not hold is still refused."""
    store = ArtifactStore(tmp_path)
    domain_spec, domain = protein_domain(store)
    screened = build_training_set(training_spec(store, domain_spec, domain), store)
    all_tasks = build_training_set(training_spec(store, domain_spec, domain, **NOTHING_HELD_OUT), store)
    held_out = [MixtureEntry(stream="all-tasks-test-declared", share=1.0, params={"tasks": ["Fungi", "Kinase"]})]
    check_construction(all_tasks, held_out, store)
    with pytest.raises(ValueError, match="trains 'Fungi'"):
        check_construction(screened, held_out, store)
    unread = [MixtureEntry(stream="all-tasks-test-declared", share=1.0, params={"fields": ["secret"]})]
    with pytest.raises(ValueError, match="reads field 'secret'"):
        check_construction(all_tasks, unread, store)


YES_NO = {
    "templates": [{"id": "y1", "user": "Here is a protein:<SLOT:x>. {question} Answer Yes or No."}],
    "questions": {"long_name": ["Is it long-named?"]},
    "sampling": "uniform",
    "answers": {"yes": "Y", "no": "N"},
    "weights": {"family": "single_token", "floor": 0.1},
}


def test_a_stream_trains_a_held_out_task_only_on_an_all_tasks_text(tmp_path):
    """The toy domain's held-out ``long_name`` is refused on a screened text and asked on an all-tasks one; the
    stream's views show it as trained there, and the domain's own catalog is unchanged."""
    store = ArtifactStore(tmp_path)
    domain = ToyDomain(store.path(commit_toy_domain(store)))
    train = tuple(sorted(TOY_OBJECTS)[:5])
    screened = TrainingText(train, {}, {})
    with pytest.raises(ValueError, match="not trained"):
        build_stream("yes_no", YES_NO, domain, screened)
    built = build_stream("yes_no", YES_NO, domain, dataclasses.replace(screened, all_tasks=True))
    assert built.domain.tasks()["long_name"].scope is TaskScope.TRAINED and built.text.all_tasks
    assert domain.tasks()["long_name"].scope is TaskScope.HELD_OUT
    assert {row.question for row in built.stream.sample(np.random.default_rng(0), 20)} == {"long_name"}
    assert DomainCatalog(domain, all_trained=True).tasks()["long_name"].scope is TaskScope.TRAINED


def molecule_domain(store):
    """Four molecules and the mass task: M3 out of its scope by the unstated subset alone, M4 also listed out of scope
    (no mass), as the molecule domain stores both rules (the scope table, and the listed cells with their reasons)."""
    ids = pd.Index(["M1", "M2", "M3", "M4"], name="object_id")
    mass = TaskSpec(id="mass", kind=TaskKind.YES_NO, question="Is it heavy?", label="@mass:median", scope="held_out")
    tables = DomainTables(
        domain="molecule",
        objects=pd.DataFrame({"value": ["C"] * 4, "name": list(ids)}, index=ids),
        split=pd.Series(["train"] * 4, index=ids),
        clusters=pd.Series(list(ids), index=ids),
        labels=pd.DataFrame({"@mass:median": [True, False, True, False]}, index=ids),
        fields=pd.DataFrame({"definition": ["a", "b", "a heavy one", "d"]}, index=ids),
        tasks=(mass,),
        frames=(Frame(id="ask", user="Is<SLOT:x> it? {question}"),),
        recipes=(),
        scope=pd.DataFrame({"mass": [True, True, False, False]}, index=ids),
        extra={
            "out_of_scope": pd.DataFrame(
                {"task": ["mass"], "rule": ["no mass"], "sources": ["entries.mass is missing"]},
                index=pd.Index(["M4"], name="object_id"),
            )
        },
    )
    spec = DomainSpec(name="molecule", data=commit_domain_data(store, tables, {}).sha256)
    return spec, open_domain(spec, store), ids, mass


def test_the_molecule_training_scope_drops_the_unstated_subset_alone(tmp_path):
    store = ArtifactStore(tmp_path)
    _, domain, ids, mass = molecule_domain(store)
    assert domain.scope_mask(mass, ids).tolist() == [True, True, False, False]
    assert domain.training_mask(mass, ids).tolist() == [True, True, True, False]  # M4: no usable label
    with pytest.raises(KeyError, match="unknown objects"):
        domain.training_mask(mass, pd.Index(["M9"]))
    for all_trained, unknown in ((False, [False, False, True, True]), (True, [False, False, False, True])):
        view = ScopedDomain(
            domain, tasks={"mass"}, labels={mass.label}, tables=(), objects=ids, user="t", all_trained=all_trained
        )
        assert view.labels([mass.label])[mass.label].reindex(ids).isna().tolist() == unknown, all_trained
