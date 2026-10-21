"""The training stage's ``input_cache`` option: embedding readers that keep their memory maps (``keep_maps``) load the
same bytes as readers that map every part file on every read, and open each part file once; the trainer records its
effective value in its execution facts beside the backend's, and its resume check (the full comparator,
``Trainer._check_execution``) lets the cache change, the one narrow exception of ``RESUMABLE_ACROSS``, while every other
fact, the backend's options included, still refuses; an invalid or missing input cache refuses. The ``kimi-ep`` cases
need its module's dependencies (compressed-tensors) and skip without them.
"""

from __future__ import annotations

import inspect
import types
from typing import Any, cast

import numpy as np
import pytest
import torch
from tests.fakes import FakeTrainingBackend, fake_receiver

from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.config import OptionError
from latent_interconnect.encoders.base import ExtractedBatch
from latent_interconnect.encoders.loader import InputLoader
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import EmbeddingReader, commit_embeddings
from latent_interconnect.run import train as train_stage
from latent_interconnect.run._main import stage_options
from latent_interconnect.training.trainer import DEFAULT_INPUT_CACHE, RESUMABLE_ACROSS, Trainer, input_loader
from latent_interconnect.writes.inputs import InputKind, InputSpec


def _recipe() -> EncoderRecipe:
    return EncoderRecipe(
        name="m",
        extractor="x",
        domain="protein",
        model_id="m",
        revision="r",
        read=(1, 2),
        pooling="mean",
        environment="e",
        batch_tokens=1,
    )


def _readers(store: ArtifactStore, keep_maps: bool) -> list[EmbeddingReader]:
    """Two artifacts of three and two parts (layers 1 and 2 apart): bf16 residues, pooled vectors and means."""
    rng, meta = np.random.default_rng(3), {"recipe": _recipe().model_dump(mode="json")}
    length = dict(zip("abcdefgh", rng.integers(1, 6, 8).tolist(), strict=True))  # an object's, at every layer
    readers = []
    for layer, groups in (
        (1, (("a", "b", "c"), ("d", "e"), ("f", "g", "h"))),
        (2, (("a", "b", "c", "d"), ("e", "f", "g", "h"))),
    ):
        parts = []
        for ids in groups:
            bf16 = [rng.integers(0, 2**16, (length[o], 3)).astype(np.uint16) for o in ids]
            pooled = rng.normal(size=(len(ids), 3)).astype(np.float32)
            parts.append(ExtractedBatch(ids, {layer: pooled}, {layer: bf16}, {layer: pooled * 2}))
        ref = commit_embeddings(store, f"p{layer}", parts, meta)
        readers.append(EmbeddingReader(store, ref, keep_maps=keep_maps))
    return readers


BATCHES = [["h", "a", "e", "a"], ["b"], ["g", "c", "f", "d", "b", "h"], ["a", "a", "a"], ["e", "d"]]


def _loader(readers: list[EmbeddingReader]) -> InputLoader:
    spec = InputSpec(recipe="m", layers=(2, 1), combine="concat")
    return InputLoader(spec, readers, {InputKind.POOLED, InputKind.RESIDUES}, encoder=_recipe())


def _bytes(t: torch.Tensor) -> tuple[str, tuple[int, ...], bytes]:
    bits = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
    return str(t.dtype), tuple(t.shape), bits.numpy().tobytes()


def test_kept_maps_load_the_bytes_of_maps_opened_on_every_read(tmp_path):
    store = ArtifactStore(tmp_path)
    kept, fresh = _loader(_readers(store, keep_maps=True)), _loader(_readers(store, keep_maps=False))
    for _ in range(2):  # the second pass reads through the kept maps
        for ids in BATCHES:
            got, want = kept.batch(ids), fresh.batch(ids)
            assert got.residues is not None and got.residues.dtype == torch.bfloat16
            for a, b in ((got.pooled, want.pooled), (got.residues, want.residues), (got.mask, want.mask)):
                assert a is not None and b is not None and _bytes(a) == _bytes(b)
    for r, f in zip(kept.readers, fresh.readers, strict=True):
        assert r.means(BATCHES[0], r.layers[0]).tobytes() == f.means(BATCHES[0], f.layers[0]).tobytes()


@pytest.mark.parametrize("keep_maps", [True, False])
def test_kept_maps_open_each_part_file_once(tmp_path, monkeypatch, keep_maps):
    store = ArtifactStore(tmp_path)
    readers = _readers(store, keep_maps)
    loader = _loader(readers)
    opened: list[str] = []
    load = np.load

    def counting(path: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(str(path))
        return load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", counting)
    for _ in range(3):
        for ids in BATCHES:
            loader.batch(ids)
    every = {str(f) for r in readers for f in r.root.glob("part-*/*.npy") if not f.name.startswith("mean.")}
    if keep_maps:
        assert len(opened) == len(set(opened))  # no file opened twice
        assert set(opened) == every  # every part's offsets, residues and pooled vectors (no means): each read once
    else:
        assert len(opened) > len(set(opened))  # the original reads open the files on every batch
    with pytest.raises(FileNotFoundError):  # a missing file is not remembered: looked for on each read
        readers[0].pooled(["a"], 2)
    with pytest.raises(FileNotFoundError):
        readers[0].pooled(["a"], 2)


# ---------------------------------------------------------------- the trainer's execution facts and resume

KIMI = {
    "backend": "kimi-ep",
    "dtype": "bfloat16",
    "quantization": "compressed-tensors pack-quantized",
    "device_name": "NVIDIA H200",
    "device_count": "8",
    "attention": "packed-varlen",
    "experts": "grouped, 8 blocks",
    "checkpointing": "none",
    "revision": "r",
    "weights": "/w",
}
"""A ``kimi-ep`` backend's facts on its original options (the keys of a record from before the speed options)."""


@pytest.fixture(scope="module")
def kimi() -> tuple[dict[str, str], dict[str, str]]:
    """``kimi-ep``'s original and CUDA-default options (``receivers.kimi_k26.ep``)."""
    ep = pytest.importorskip("latent_interconnect.receivers.kimi_k26.ep")
    return dict(ep.REFERENCE_OPTIONS), dict(ep.CUDA_DEFAULTS)


def _trainer(facts: dict[str, str], input_cache: str, legacy: Any) -> Any:
    """A trainer whose backend's ``execution()`` is ``facts``, with ``input_cache``: its real execution and check."""
    backend = types.SimpleNamespace(execution=lambda: facts, reference_options=legacy)
    trainer: Any = types.SimpleNamespace(
        config=types.SimpleNamespace(name="r"), backend=backend, input_cache=input_cache
    )
    trainer.execution = lambda: Trainer.execution(trainer)
    return trainer


def _check(trainer: Any, saved: dict[str, Any] | None) -> None:
    Trainer._check_execution(trainer, saved)


def test_the_trainer_records_the_input_cache_beside_the_backends_facts(kimi):
    legacy, fast = kimi
    for cache in ("on", "off"):  # the backend's keys follow its own rule, never the cache
        assert _trainer(KIMI, cache, legacy).execution() == {**KIMI, "input_cache": cache}
        assert _trainer({**KIMI, **fast}, cache, legacy).execution() == {**KIMI, **fast, "input_cache": cache}
    assert "input_cache" in RESUMABLE_ACROSS and DEFAULT_INPUT_CACHE == "on"


@pytest.mark.parametrize("input_cache", ["on", "off"])
def test_a_record_from_before_the_options_resumes_with_the_cache_on_or_off(kimi, input_cache):
    legacy, _ = kimi
    _check(_trainer(KIMI, input_cache, legacy), dict(KIMI))  # an old state: no option key, no input_cache


@pytest.mark.parametrize(("then", "now"), [("on", "off"), ("off", "on"), ("on", "on")])
def test_a_run_resumes_with_another_input_cache_in_the_same_mode(kimi, then, now):
    legacy, fast = kimi
    for facts in (KIMI, {**KIMI, **fast}):
        _check(_trainer(facts, now, legacy), _trainer(facts, then, legacy).execution())


def test_a_backend_without_options_continues_in_the_same_mode():
    """A non-Kimi backend (the test fake, standing in for hf): its record continues with the cache either way."""
    fake = FakeTrainingBackend(fake_receiver())
    for then, now in (("on", "off"), ("off", "on")):
        saved = _trainer(fake.execution(), then, fake.reference_options).execution()
        _check(_trainer(fake.execution(), now, fake.reference_options), saved)
        _check(_trainer(fake.execution(), now, fake.reference_options), fake.execution())  # an old record


def test_every_semantic_option_change_still_refuses(kimi):
    legacy, fast = kimi
    resume = " ".join(f"--option {k}={v}" for k, v in legacy.items())
    for cache in ("on", "off"):
        for name, value in fast.items():
            with pytest.raises(ValueError, match="differing") as refused:
                _check(_trainer({**KIMI, **legacy, name: value}, cache, legacy), dict(KIMI))
            assert str(refused.value).endswith(f"resume with the options the run started with: {resume}")
        with pytest.raises(ValueError, match="differing"):
            _check(_trainer(KIMI, cache, legacy), _trainer({**KIMI, **fast}, cache, legacy).execution())


def test_an_input_identity_change_still_refuses(kimi):
    legacy, _ = kimi
    with pytest.raises(ValueError, match=r"differing: \['dtype'\]"):
        _check(_trainer({**KIMI, "dtype": "float32"}, "on", legacy), _trainer(KIMI, "on", legacy).execution())


@pytest.mark.parametrize("bad", ["maybe", "", "ON", 1, None])
def test_an_invalid_input_cache_refuses(kimi, bad):
    legacy, _ = kimi
    with pytest.raises(ValueError, match="not on or off"):
        _check(_trainer(KIMI, "on", legacy), {**KIMI, "input_cache": bad})
    with pytest.raises(ValueError, match="not on or off"):
        _check(_trainer(KIMI, cast(Any, bad), legacy), dict(KIMI))


def test_a_record_of_backend_options_without_an_input_cache_refuses(kimi):
    legacy, fast = kimi
    with pytest.raises(ValueError, match="without an input_cache"):
        _check(_trainer({**KIMI, **fast}, "on", legacy), {**KIMI, **fast})


def test_the_train_stage_takes_input_cache_on_by_default():
    assert inspect.signature(train_stage.STAGE.run).parameters["input_cache"].default == DEFAULT_INPUT_CACHE
    for name in ("__init__", "from_config"):
        assert inspect.signature(getattr(Trainer, name)).parameters["input_cache"].default == DEFAULT_INPUT_CACHE
    assert inspect.signature(input_loader).parameters["input_cache"].default == DEFAULT_INPUT_CACHE
    assert stage_options({"input_cache": "off"}, train_stage.STAGE.run, train_stage.STAGE.config) == {
        "input_cache": "off"
    }
    with pytest.raises(OptionError, match="input_cache"):
        stage_options({"input_cache": "maybe"}, train_stage.STAGE.run, train_stage.STAGE.config)


@pytest.mark.parametrize("bad", ["maybe", "", "ON"])
def test_the_trainer_refuses_an_invalid_input_cache_when_built(bad):
    """A value other than on or off is refused at construction, never run as either."""
    with pytest.raises(ValueError, match="input_cache must be one of"):
        Trainer(
            config=cast(Any, None),
            backend=cast(Any, None),
            connector=cast(Any, None),
            load=cast(Any, None),
            sampler=cast(Any, None),
            store=cast(Any, None),
            workspace=cast(Any, None),
            input_cache=cast(Any, bad),
        )
