"""The strict construction reader of approved release exports (``build.metadata_export``) and its three consumers.

An ordinary training set (built here by ``build_training_set``) keeps its producing spec and never reads the registry;
an export of it onto a domain whose held-out task id was renamed resolves only through an exact registry entry, and
every consumer (the zero-shot construction check, the trainer's domain check, the logistic baseline's) then checks the
same definition with its original refusals. Missing, malformed, contradictory, forged, tampered and rehashed exports
refuse; nothing falls back to ``meta.spec``.
"""

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore, IntegrityError
from latent_interconnect.build import metadata_export as mx
from latent_interconnect.build.zero_shot import (
    TrainingSetSpec,
    build_training_set,
    check_construction,
    training_set_construction,
    training_set_spec,
)
from latent_interconnect.canonical import canonical_json, canonical_sha256
from latent_interconnect.data.items import commit_item_set, object_list
from latent_interconnect.data.sampler import MixtureEntry
from latent_interconnect.data.training_text import TRAINING_SET_KIND
from latent_interconnect.domains.base import (
    DOMAIN_DATA_KIND,
    DomainSpec,
    Frame,
    TaskKind,
    TaskScope,
    TaskSpec,
    open_domain,
)
from latent_interconnect.domains.tables import DomainTables, commit_domain_data
from latent_interconnect.evaluation import logistic_baseline
from latent_interconnect.training import trainer as trainer_module

IDS = ("o1", "o2", "o3", "o4")


def _domain(store, held_out_id):
    index = pd.Index(list(IDS), name="object_id")
    tasks = (
        TaskSpec(id="t", kind=TaskKind.YES_NO, question="Is it t?", label="y", scope=TaskScope.TRAINED),
        TaskSpec(
            id=held_out_id, kind=TaskKind.YES_NO, question="Does it bind zinc?", label="z", scope=TaskScope.HELD_OUT
        ),
        TaskSpec(id="tz", kind=TaskKind.YES_NO, question="Is it zinc-ish?", label="z", scope=TaskScope.TRAINED),
    )
    tables = DomainTables(
        domain="protein",
        objects=pd.DataFrame({"value": ["MA"] * 4, "name": list(IDS)}, index=index),
        split=pd.Series(["train"] * 4, index=index),
        clusters=pd.Series([f"c{i}" for i in IDS], index=index),
        labels=pd.DataFrame({"y": [True, False, True, False], "z": [True, True, False, False]}, index=index),
        fields=pd.DataFrame({"caption": ["binds zinc", "a zinc finger", "plain", "plain too"]}, index=index),
        tasks=tasks,
        frames=(Frame(id="ask", user="Is<SLOT:x> it? {question}"),),
        recipes=(),
    )
    return DomainSpec(name="protein", data=commit_domain_data(store, tables, {}).sha256)


def _mixture(task):
    params = {
        "templates": [{"id": "y1", "user": "Here is a protein:<SLOT:x>. {question} Answer Yes or No."}],
        "questions": {task: ["Is it?"]},
        "sampling": "uniform",
        "answers": {"yes": "Y", "no": "N"},
        "weights": {"family": "single_token", "floor": 0.1},
    }
    return [MixtureEntry(stream="yes_no", share=1.0, params=params)]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A training set on domain A (held-out task ``h``), domain B (``h`` renamed ``h2``) and an empty registry."""
    store = ArtifactStore(tmp_path / "store")
    a = _domain(store, "h")
    b = _domain(store, "h2")
    train = commit_item_set(store, object_list("train", open_domain(a, store), IDS))
    spec = TrainingSetSpec.model_validate(
        {
            "domain": a.model_dump(mode="json"),
            "train_ids": train.sha256,
            "screen": {
                "fields": ["caption"],
                "property_list_fields": [],
                "zero_shot_tasks": ["h"],
                "detectors": [{"task": "h", "detector": "lexical", "params": {"terms": ["zinc"]}}],
            },
        }
    )
    ref = build_training_set(spec, store)
    registry = {}
    monkeypatch.setattr(mx, "_registry_text", lambda: json.dumps({"format": mx.REGISTRY_FORMAT, "exports": registry}))
    mx.release_registry.cache_clear()
    yield SimpleNamespace(store=store, a=a, b=b, ref=ref, spec=spec, registry=registry, tmp=tmp_path)
    mx.release_registry.cache_clear()


def _rename(obj, old, new):
    if isinstance(obj, dict):
        return {(new if k == old else k): _rename(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rename(x, old, new) for x in obj]
    return new if obj == old else obj


def make_export(
    w, *, register=True, edit_definition=None, edit_descriptor=None, edit_receipt=None, edit_meta=None, target=None
):
    """Export ``w.ref`` onto domain B as the staging exporter does; register it unless told not to; return its ref."""
    man = w.store.manifest(w.ref)
    spec = man.meta["spec"]
    target = (target or w.b).model_dump(mode="json")
    definition = _rename(copy.deepcopy(spec), "h", "h2")
    definition["domain"] = target
    if edit_definition:
        edit_definition(definition)
    exported_from = {
        "training_set": w.ref.sha256,
        "domain": spec["domain"],
        "spec_digest": man.meta["spec_digest"],
        "spec_sha256": canonical_sha256(spec),
    }
    transformations = {"task_ids": {"new_values": ["h2"], "dictionary_sha256": canonical_sha256({"h": "h2"})}}
    receipt = {
        "format": mx.RECEIPT_FORMAT,
        "exported_from": copy.deepcopy(exported_from),
        "target_domain": copy.deepcopy(target),
        "definition_sha256": canonical_sha256(definition),
        "transformations": copy.deepcopy(transformations),
        "verification": {"scope": "test"},
    }
    if edit_receipt:
        edit_receipt(receipt)
    rbytes = canonical_json(receipt).encode()
    descriptor = {
        "format": mx.EXPORT_FORMAT,
        "exported_from": exported_from,
        "target_domain": target,
        "definition": definition,
        "definition_sha256": canonical_sha256(definition),
        "transformations_sha256": canonical_sha256(transformations),
        "receipt_sha256": hashlib.sha256(rbytes).hexdigest(),
    }
    if edit_descriptor:
        edit_descriptor(descriptor)
    meta = {k: _rename(v, "h", "h2") for k, v in man.meta.items() if k not in ("spec", "spec_digest")}
    meta[mx.MARKER] = descriptor
    if edit_meta:
        edit_meta(meta)
    staged = w.store.staging(
        f"export-{len(w.registry)}-{hashlib.sha256(canonical_json(meta).encode()).hexdigest()[:8]}"
    )
    for fn in man.files:
        shutil.copyfile(w.store.path(w.ref) / fn, staged / fn)
    (staged / mx.RECEIPT_FILE).write_bytes(rbytes)
    ref = w.store.commit(staged, TRAINING_SET_KIND, meta)
    if register:
        w.registry[ref.sha256] = {
            "exported_from": w.ref.sha256,
            "target_domain": w.b.model_dump(mode="json"),
            "descriptor_sha256": canonical_sha256(descriptor) if isinstance(descriptor, dict) else "0" * 64,
            "receipt_sha256": hashlib.sha256(rbytes).hexdigest(),
            "transformations_sha256": canonical_sha256(transformations),
        }
        mx.release_registry.cache_clear()
    return ref


# ---------------------------------------------------------------------------------------------- ordinary path


def test_ordinary_training_set_keeps_its_spec_and_never_reads_the_registry(world, monkeypatch):
    def broken():
        raise OSError("no registry")

    monkeypatch.setattr(mx, "_registry_text", broken)
    mx.release_registry.cache_clear()
    assert training_set_construction(world.store, world.ref) == training_set_spec(world.store, world.ref) == world.spec
    check_construction(world.ref, _mixture("t"), world.store)  # the registry is unreadable and not needed
    with pytest.raises(ValueError, match="supervises zero-shot tasks"):
        check_construction(world.ref, _mixture("tz"), world.store)


def test_ordinary_training_set_without_spec_still_refuses(world):
    man = world.store.manifest(world.ref)
    staged = world.store.staging("nospec")
    for fn in man.files:
        shutil.copyfile(world.store.path(world.ref) / fn, staged / fn)
    ref = world.store.commit(staged, TRAINING_SET_KIND, {"removed": man.meta["removed"]})
    for read in (training_set_spec, training_set_construction):
        with pytest.raises(ValueError, match="no spec"):
            read(world.store, ref)


# ---------------------------------------------------------------------------------------------- approved export


def test_approved_export_resolves_its_definition_and_the_historical_api_refuses(world):
    ref = make_export(world)
    built = training_set_construction(world.store, ref)
    assert built.domain == world.b and built.screen.zero_shot_tasks == ("h2",)
    assert built.train_ids == world.spec.train_ids and built.screen.detectors[0].params == {"terms": ["zinc"]}
    with pytest.raises(ValueError, match="source production spec is unavailable"):
        training_set_spec(world.store, ref)


def test_zero_shot_check_screens_the_export_with_its_original_refusals(world):
    ref = make_export(world)
    check_construction(ref, _mixture("t"), world.store)
    with pytest.raises(ValueError, match="trains 'tz': same of zero-shot 'h2'"):
        check_construction(ref, _mixture("tz"), world.store)


class _StopError(Exception):
    pass


def _trainer_checks(store, ref, domain, mixture, monkeypatch):
    """Run ``Trainer.from_config`` up to the connector (its checks before any model loads)."""

    def stop(*_a, **_k):
        raise _StopError

    monkeypatch.setattr(trainer_module, "initial_connector", stop)
    config = SimpleNamespace(
        plugins=(), training_set=ref.sha256, domain=domain, name="run", mixture=mixture, init_spec=lambda: None
    )
    trainer_module.Trainer.from_config(config, store, store.root)


def test_trainer_and_logistic_baseline_agree_with_the_export_target(world, monkeypatch):
    ref = make_export(world)
    with pytest.raises(_StopError):  # domain B passes the trainer's domain check and the construction check
        _trainer_checks(world.store, ref, world.b, _mixture("t"), monkeypatch)
    with pytest.raises(ValueError, match="is not the domain its training set was built and screened on"):
        _trainer_checks(world.store, ref, world.a, _mixture("t"), monkeypatch)  # export_of's domain is not accepted
    with pytest.raises(ValueError, match="supervises zero-shot tasks"):
        _trainer_checks(world.store, ref, world.b, _mixture("tz"), monkeypatch)
    spec = SimpleNamespace(training_set=ref.sha256)
    trained, listed = logistic_baseline._training(spec, world.store, world.b, None, ())
    assert trained is None and listed == IDS
    with pytest.raises(ValueError, match="training set of protein"):
        logistic_baseline._training(spec, world.store, world.a, None, ())


# ---------------------------------------------------------------------------------------------- refusals


def test_unregistered_export_refuses(world):
    ref = make_export(world, register=False)
    with pytest.raises(ValueError, match="not an approved release export"):
        training_set_construction(world.store, ref)
    with pytest.raises(ValueError, match="not an approved release export"):
        check_construction(ref, _mixture("t"), world.store)


def test_missing_or_malformed_registry_refuses_every_export(world, monkeypatch):
    ref = make_export(world)
    for text in ("{", json.dumps({"format": "aimm-release-export-registry/2", "exports": {}})):
        monkeypatch.setattr(mx, "_registry_text", lambda text=text: text)
        mx.release_registry.cache_clear()
        with pytest.raises(ValueError, match="missing or malformed"):
            training_set_construction(world.store, ref)
    assert training_set_construction(world.store, world.ref) == world.spec  # the ordinary path is unaffected


@pytest.mark.parametrize(
    ("case", "kwargs", "message"),
    [
        ("unsupported schema", {"edit_descriptor": lambda d: d.update(format="aimm-release-export/2")}, "unsupported"),
        ("malformed marker", {"edit_meta": lambda m: m.update({mx.MARKER: "approved"})}, "not a mapping"),
        (
            "marker and producing spec",
            {"edit_meta": lambda m: m.update(spec={"domain": {}})},
            "contradict",
        ),
        (
            "forged original",
            {"edit_descriptor": lambda d: d["exported_from"].update(training_set="1" * 64)},
            "original training set is not the registered one",
        ),
        (
            "changed target domain",
            {"edit_descriptor": lambda d: d["target_domain"].update(data="2" * 64)},
            "target domain is not the registered one",
        ),
        (
            "definition digest",
            {"edit_descriptor": lambda d: d.update(definition_sha256="3" * 64)},
            "different definitions",
        ),
        (
            "receipt digest",
            {"edit_descriptor": lambda d: d.update(receipt_sha256="4" * 64)},
            "receipt is not the registered one",
        ),
        (
            "receipt names another original",
            {"edit_receipt": lambda r: r["exported_from"].update(spec_digest="5" * 64)},
            "different originals",
        ),
        ("receipt schema", {"edit_receipt": lambda r: r.update(format="other/1")}, "supported schema"),
        (
            "transformation inventory",
            {"edit_receipt": lambda r: r["transformations"].update(extra=1)},
            "transformation inventory is not the descriptor's",
        ),
    ],
)
def test_contradictory_export_refuses(world, case, kwargs, message):
    del case
    ref = make_export(world, **kwargs)
    with pytest.raises(ValueError, match=message):
        training_set_construction(world.store, ref)


def test_rehashed_tampered_metadata_fails_the_registry_binding(world):
    good = make_export(world)

    def swap(definition):  # a self-consistent rehash: a weaker detector, every digest and the receipt recomputed
        definition["screen"]["detectors"][0]["params"] = {"terms": ["nothing"]}

    forged = make_export(world, register=False, edit_definition=swap)
    assert forged.sha256 != good.sha256
    with pytest.raises(ValueError, match="not an approved release export"):
        training_set_construction(world.store, forged)
    # registering the forged id under the good entry's digests is refused as well
    world.registry[forged.sha256] = dict(world.registry[good.sha256])
    mx.release_registry.cache_clear()
    with pytest.raises(ValueError, match="descriptor is not the registered one"):
        training_set_construction(world.store, forged)


def test_tampered_receipt_or_definition_files_refuse(world):
    ref = make_export(world)
    receipt = world.store.path(ref) / mx.RECEIPT_FILE
    receipt.chmod(0o644)
    receipt.write_bytes(receipt.read_bytes().replace(b'"test"', b'"tset"'))
    with pytest.raises(IntegrityError):
        training_set_construction(world.store, ref)


def test_changed_target_domain_payload_refuses(world):
    ref = make_export(world)
    target = world.store.path(ArtifactRef(world.b.data, DOMAIN_DATA_KIND))
    labels = target / "labels.parquet"  # the target's manifest is unchanged and genuine; one payload file is not
    labels.chmod(0o644)
    labels.write_bytes(labels.read_bytes() + b"\0")
    with pytest.raises(IntegrityError):
        training_set_construction(world.store, ref)


def test_registered_definition_must_be_a_valid_construction(world):
    def bad(definition):  # a zero-shot task without a detector; every digest consistent and registered
        definition["screen"]["zero_shot_tasks"] = ["h2", "t"]

    ref = make_export(world, edit_definition=bad)
    with pytest.raises(ValueError, match="not a valid construction"):
        training_set_construction(world.store, ref)


def test_packaged_registry_is_well_formed_and_shipped():
    mx.release_registry.cache_clear()
    entries = mx.release_registry()
    assert len(entries) == 2
    assert {e.target_domain.name for e in entries.values()} == {"protein", "molecule"}
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    assert mx.REGISTRY_FILE in pyproject
