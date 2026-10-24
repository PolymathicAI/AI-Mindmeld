"""Frozen identities of default (screened) training sets and of the streams built on them.

A training set that holds its domain's held-out tasks out (``TrainingSetSpec.all_tasks`` False) keeps its spec
digest, stored spec and artifact digest, and a stream built on it draws the same examples and reads the same scoped
labels. The values below are the ones these builds give; a change to any of them changes every stored training set
and every run that cites one."""

import hashlib

import numpy as np
import pandas as pd
from tests.fakes import TOY_OBJECTS, commit_toy_domain

from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.build.zero_shot import DetectorSpec, TrainingSetSpec, build_training_set
from latent_interconnect.data.items import commit_item_set, object_list
from latent_interconnect.data.streams import build_stream
from latent_interconnect.data.training_text import TrainingText, load_training_text
from latent_interconnect.domains.base import DomainSpec, Frame, TaskKind, TaskSpec, open_domain
from latent_interconnect.domains.tables import DomainTables, commit_domain_data

IDS = pd.Index(["P1", "P2", "P3", "P4", "P5"], name="object_id")
FUNGI = {"terms": [r"\bfung(?:us|i|al)\b"]}
LONG = {"terms": [r"\blong\b"]}
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
MASS = {
    "templates": [{"id": "m1", "user": "Here is a molecule:<SLOT:x>. {question} Answer Yes or No."}],
    "questions": {"mass": ["Is it heavy?"]},
    "sampling": "uniform",
    "answers": {"yes": "Y", "no": "N"},
    "weights": {"family": "single_token", "floor": 0.1},
}

PROTEIN = {
    "domain": "bf1894dd67397925d926b947c29d3b89d141260ab40e80b7f809d4d3374618a0",
    "train_ids": "a30026c14961fcf85a36440227249bca0ecfadb60a6eda067ef001c84b3d394c",
    "spec_digest": "6e62363fc6e7c60345129563a30eab31cec1bac0f41f504adbfb176e769c0a9e",
    "training_set": "64053371ebc49b07f174ca817129c17b88abb28c52b44ed2e27c3984e144fd73",
}
TOY = {
    "training_set": "90a1f5c027441e87e0773eff00ac19a9727037522209928b15797ea1412b6840",
    "yes_no": "32737e7307c8f555480facda84dcb583f6eaf36e7fc2b001747e6e1862004dac",
    "text": "d098afd1d267c0a71f7ad4b2c35d0c331fce81147fa1a359e804ef7d675d8637",
}
MOLECULE = {
    "yes_no": "f757e2af74b751c93f2ea5055d34bde1b00d55d34173b1e7260a90b42cac27b6",
    "unknown": [False, False, True, True],  # M3 (unstated subset) and M4 (no mass) read as unknown
}


def _sha(value: object) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _samples(name, params, domain, text) -> str:
    """The digest of 24 examples a stream draws with seed 0 (their ``repr``: every field of every example)."""
    built = build_stream(name, params, domain, text)
    return _sha(built.stream.sample(np.random.default_rng(0), 24))


def protein_build(store: ArtifactStore) -> dict[str, str]:
    """Five proteins (P5 in ``test``): a trained and a held-out Yes/No task; the screen removes fungal text."""
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
            },
            index=IDS,
        ),
        tasks=tasks,
        frames=(Frame(id="ask", user="Is<SLOT:x> it? {question}"),),
        recipes=(),
    )
    domain_spec = DomainSpec(name="protein", data=commit_domain_data(store, tables, {}).sha256)
    domain = open_domain(domain_spec, store)
    train = commit_item_set(store, object_list("train", domain, ("P1", "P2", "P3", "P4"))).sha256
    spec = TrainingSetSpec.model_validate(
        {
            "domain": domain_spec,
            "train_ids": train,
            "screen": {
                "fields": ("caption", "keywords"),
                "list_fields": {"keywords": ";"},
                "property_list_fields": ("keywords",),
                "zero_shot_tasks": ("Fungi",),
                "detectors": (DetectorSpec(task="Fungi", detector="lexical", params=FUNGI),),
            },
            "screened_elements": {"keywords": ("Fungi",)},
        }
    )
    ref = build_training_set(spec, store)
    return {"domain": domain_spec.data, "train_ids": train, "spec_digest": spec.digest(), "training_set": ref.sha256}


def toy_build(store: ArtifactStore) -> dict[str, str]:
    """The toy domain's training objects, screened for its held-out ``long_name``; a Yes/No and a text stream on it."""
    domain_spec = DomainSpec(name="fake-toy", data=commit_toy_domain(store).sha256)
    domain = open_domain(domain_spec, store)
    train = tuple(o for o in sorted(TOY_OBJECTS) if domain.split()[o] == "train")
    item_set = commit_item_set(store, object_list("train", domain, train)).sha256
    spec = TrainingSetSpec.model_validate(
        {
            "domain": domain_spec,
            "train_ids": item_set,
            "screen": {
                "fields": ("description",),
                "property_list_fields": (),
                "zero_shot_tasks": ("long_name",),
                "detectors": (DetectorSpec(task="long_name", detector="lexical", params=LONG),),
            },
        }
    )
    ref = build_training_set(spec, store)
    text = load_training_text(store, ref)
    return {
        "training_set": ref.sha256,
        "yes_no": _samples("yes_no", YES_NO, domain, text),
        "text": _samples("text", TEXT, domain, text),
    }


def molecule_build(store: ArtifactStore) -> dict[str, object]:
    """Four molecules and a trained mass task: M3 out of its scope by the unstated subset, M4 listed (no mass)."""
    ids = pd.Index(["M1", "M2", "M3", "M4"], name="object_id")
    mass = TaskSpec(id="mass", kind=TaskKind.YES_NO, question="Is it heavy?", label="@mass:median", scope="trained")
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
    domain = open_domain(DomainSpec(name="molecule", data=commit_domain_data(store, tables, {}).sha256), store)
    text = TrainingText(tuple(ids), {}, {})
    built = build_stream("yes_no", MASS, domain, text)
    unknown = built.domain.labels([mass.label])[mass.label].reindex(ids).isna().tolist()
    return {"yes_no": _samples("yes_no", MASS, domain, text), "unknown": unknown}


def test_a_screened_protein_training_set_keeps_its_identity(tmp_path):
    assert protein_build(ArtifactStore(tmp_path)) == PROTEIN


def test_a_screened_toy_training_set_and_its_streams_keep_their_output(tmp_path):
    assert toy_build(ArtifactStore(tmp_path)) == TOY


def test_a_molecule_stream_keeps_its_scoped_labels_and_examples(tmp_path):
    assert molecule_build(ArtifactStore(tmp_path)) == MOLECULE
