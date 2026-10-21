"""Extraction jobs and the built-in extractors' pooling, with stand-in models (the real models run on GPU)."""

import json
import multiprocessing
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latent_interconnect.artifacts.store import ArtifactStore, OriginError
from latent_interconnect.config import load_config
from latent_interconnect.encoders.base import RawObject
from latent_interconnect.encoders.extractors._windows import store_dtype as _store_dtype
from latent_interconnect.encoders.extractors.esmc import EsmcExtractor
from latent_interconnect.encoders.extractors.mimic import MimicExtractor
from latent_interconnect.encoders.jobs import ExtractionJob, finalize, missing_parts, run_worker, write_objects
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import OUTPUT_LAYER, EmbeddingReader, sweep_dead_writers, writer_token
from latent_interconnect.encoders.worker import main as worker_main

KMER = "examples/configs/rna-kmer.yaml"


def job(tmp_path, repo, n=5, part_size=2):
    recipe = load_config(repo / KMER, EncoderRecipe).model_dump(mode="json")
    objects = [RawObject(f"r{i}", "ACGU" * (i + 1)) for i in range(n)]
    sha = write_objects(tmp_path / "objects.jsonl", objects)
    return ExtractionJob(recipe, str(tmp_path / "objects.jsonl"), sha, part_size, str(tmp_path / "store"))


def test_sharded_workers_cover_the_job_and_finalize_once(tmp_path, repo):
    j = job(tmp_path, repo)
    assert missing_parts(j) == [0, 1, 2]
    assert run_worker(j, "cpu", shard=(1, 2)) == 1  # part 1
    with pytest.raises(RuntimeError, match="missing"):
        finalize(j)
    assert run_worker(j, "cpu", shard=(0, 2)) == 2  # parts 0 and 2
    assert run_worker(j, "cpu", shard=(0, 2)) == 0  # resume: nothing left
    ref = finalize(j)
    assert finalize(j) == ref and missing_parts(j) == [] and run_worker(j, "cpu") == 0  # committed: no rework
    reader = EmbeddingReader(ArtifactStore(tmp_path / "store"), ref)
    assert len(reader) == 5 and reader.layers == (OUTPUT_LAYER,)
    meta = ArtifactStore(tmp_path / "store").manifest(ref).meta
    assert meta["fingerprint"] == {"extractor": "example-kmer", "k": "2", "dtype": "float32"} and meta["part_size"] == 2


def test_concurrent_finalizers_commit_once_and_agree(tmp_path, repo):
    import threading

    j = job(tmp_path, repo, n=8, part_size=1)
    assert run_worker(j, "cpu") == 8
    barrier = threading.Barrier(4)
    refs, errors = [], []

    def finalizer():
        barrier.wait()
        try:
            refs.append(finalize(j))
        except Exception as err:
            errors.append(err)

    threads = [threading.Thread(target=finalizer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and len(refs) == 4 and len(set(refs)) == 1
    staging = tmp_path / "store" / "_staging"
    assert not (staging / j.key()).exists()  # committed once; no claim directory left behind
    assert [p.name for p in staging.iterdir() if p.name.startswith(j.key()) and ".claim-" in p.name] == []


def test_a_commit_killed_before_publishing_is_recovered_and_finalizes_to_the_same_digest(tmp_path, repo, monkeypatch):
    (tmp_path / "clean").mkdir()
    clean = job(tmp_path / "clean", repo)
    run_worker(clean, "cpu")
    expected = finalize(clean)
    j = job(tmp_path, repo)
    run_worker(j, "cpu")
    rename = Path.rename

    def killed_before_publishing(path, target):  # the claim's MANIFEST.json is written; the move never happens
        if ".claim-" in path.name and Path(target).parent.name != "_staging":
            raise SystemExit("killed")
        return rename(path, target)

    with monkeypatch.context() as m:
        m.setattr(Path, "rename", killed_before_publishing)
        with pytest.raises(SystemExit):
            finalize(j)
    assert not (tmp_path / "store" / "_staging" / j.key()).exists()  # left as a dead claim
    assert finalize(j) == expected


def _hold_a_writer_lock(staging, ready):
    from latent_interconnect.encoders.store import writer_lock

    with writer_lock(staging) as token:
        (staging / f"part-00000.tmp-{token}").mkdir()
        ready.put(token)
        time.sleep(120)


def test_a_dead_writers_temporary_part_is_swept_and_a_live_ones_never(tmp_path, repo):
    """A killed worker's part-*.tmp-<token> no longer blocks the commit; a live writer's is left alone."""
    j = job(tmp_path, repo)
    staging = ArtifactStore(Path(j.artifacts_root)).staging(j.key())
    ctx = multiprocessing.get_context("fork")
    ready = ctx.Queue()
    live = ctx.Process(target=_hold_a_writer_lock, args=(staging, ready))
    live.start()
    try:
        token = ready.get(timeout=30)
        tmp = staging / f"part-00000.tmp-{token}"
        assert sweep_dead_writers(staging, writer_token()) == [] and tmp.is_dir()  # its writer holds the lock
        assert run_worker(j, "cpu", shard=(1, 2)) == 1 and tmp.is_dir()  # nor does a worker's start remove it
    finally:
        live.kill()
        live.join()
    assert tmp.is_dir()  # killed: the lock is released, the directory left
    with pytest.raises(RuntimeError, match="missing"):
        finalize(j)
    own = staging / f"part-00002.tmp-{writer_token()}"  # an earlier process with this host and pid
    own.mkdir()
    assert run_worker(j, "cpu", shard=(0, 2)) == 2
    assert not tmp.exists() and not own.exists()
    assert sorted(p.name for p in staging.parent.glob(f"{staging.name}.writer-*")) == []  # every lock file removed
    finalize(j)


def test_worker_refuses_a_changed_object_list_and_bad_shards(tmp_path, repo):
    j = job(tmp_path, repo)
    (tmp_path / "objects.jsonl").write_text(json.dumps({"object_id": "x", "value": "A"}) + "\n")
    with pytest.raises(ValueError, match="sha256"):
        run_worker(j, "cpu")
    with pytest.raises(ValueError, match="shard"):
        run_worker(j, "cpu", shard=(2, 2))


def test_worker_entry_point_commits(tmp_path, repo, capsys):
    j = job(tmp_path, repo, n=3, part_size=3)
    j.dump(tmp_path / "job.json")
    assert worker_main([str(tmp_path / "job.json"), "--device", "cpu"]) == 0
    assert "embeddings: " in capsys.readouterr().out
    for bad in ("1", "a/2", "0/"):
        with pytest.raises(SystemExit):
            worker_main([str(tmp_path / "job.json"), "--shard", bad])


@pytest.mark.parametrize("completed", [False, True], ids=["partial", "completed"])
def test_a_worker_reuses_work_only_under_its_origin_or_a_declared_code_handoff(tmp_path, repo, monkeypatch, completed):
    import latent_interconnect.encoders.jobs as jobs

    (tmp_path / "clean").mkdir()
    clean = job(tmp_path / "clean", repo)
    run_worker(clean, "cpu")
    expected = finalize(clean)
    j = job(tmp_path, repo)
    assert run_worker(j, "cpu", shard=(0, 2)) == 2  # parts 0 and 2
    if completed:
        run_worker(j, "cpu")
        assert finalize(j) == expected
    store = ArtifactStore(tmp_path / "store")
    recorded = store.origin(j.key())["origin"]
    real = jobs.worker_origin
    for change in ({"fingerprint": {**recorded["fingerprint"], "dtype": "bfloat16"}},):  # hardware alone is accepted
        monkeypatch.setattr(jobs, "worker_origin", lambda f, d, change=change: real(f, d) | change)
        with pytest.raises(OriginError, match="no handoff waives"):  # even declared
            run_worker(j, "cpu", handoff={"previous": recorded, "current": recorded | change, "reason": "same"})
    monkeypatch.setattr(jobs, "worker_origin", real)
    assert run_worker(j, "cpu", shard=(0, 2)) == 0  # the same origin: reused
    monkeypatch.setattr(jobs, "running_code", lambda: ("c" * 40, False))
    with pytest.raises(OriginError, match="declared handoff"):
        run_worker(j, "cpu")
    moved = recorded | {"commit": "c" * 40, "dirty": False}
    handoff = {"previous": recorded, "current": moved, "reason": "reviewed: no numerical change"}
    assert run_worker(j, "cpu", handoff=handoff) == (0 if completed else 1)  # earlier parts are not recomputed
    assert store.origin(j.key()) == {"origin": moved, "handoffs": [handoff]}  # the handoff is recorded
    assert finalize(j) == expected


class FakeEsmTokenizer:
    """``<cls>`` + one token per residue + ``<eos>``, right-padded; ``<cls>`` is 1000 + the window's first code,
    ``<eos>`` 2000 + its last."""

    def __call__(self, seqs, return_tensors, padding):
        width = max(len(s) for s in seqs) + 2
        mask = torch.zeros(len(seqs), width, dtype=torch.long)
        for i, s in enumerate(seqs):
            mask[i, : len(s) + 2] = 1
        ids = torch.zeros_like(mask)
        for i, s in enumerate(seqs):
            ids[i, 1 : len(s) + 1] = torch.tensor([ord(c) for c in s])
            ids[i, 0] = 1000 + ord(s[0])
            ids[i, len(s) + 1] = 2000 + ord(s[-1])
        return {"input_ids": ids, "attention_mask": mask}


class FakeEsm:
    """State at layer L of a residue with code c is (c + L, 1); specials and padding get large values."""

    config = SimpleNamespace(num_hidden_layers=2, hidden_size=2)

    def __call__(self, input_ids, attention_mask, output_hidden_states):
        assert output_hidden_states
        code = input_ids.float().masked_fill(input_ids == 0, 1e6)
        states = [torch.stack([code + L, torch.ones_like(code)], dim=-1) for L in range(3)]
        return SimpleNamespace(hidden_states=states)


def esmc(residues=False, max_residues=3, pooling="mean"):
    recipe = {
        "read": [0, 2],
        "pooling": pooling,
        "options": {"max_residues": max_residues},
        "batch_tokens": 10,
        "residues": residues,
        "residue_dtype": "fp32",
        "model_id": "m",
        "revision": "r",
    }
    ex = EsmcExtractor(recipe, "cpu")
    ex.model, ex.tokenizer, ex.layers = FakeEsm(), FakeEsmTokenizer(), (0, 2)
    return ex


def test_esmc_pools_over_residues_and_windows_weighted_by_length():
    seqs = ["ABCDEFG", "AB"]  # 7 residues: windows of 3, 3, 1
    out = esmc().extract([RawObject("p", seqs[0]), RawObject("q", seqs[1])])
    expect = [np.mean([ord(c) for c in s]) for s in seqs]
    np.testing.assert_allclose(out.pooled[0][:, 0], expect, rtol=1e-6)
    np.testing.assert_allclose(out.pooled[2][:, 0], np.array(expect) + 2, rtol=1e-6)
    assert out.residues == {}
    with_res = esmc(residues=True).extract([RawObject("p", seqs[0])])
    np.testing.assert_array_equal(with_res.residues[2][0][:, 0], [ord(c) + 2 for c in seqs[0]])
    cls = esmc(pooling="cls").extract([RawObject("p", seqs[0]), RawObject("q", seqs[1])])  # special-token diagnostic
    np.testing.assert_array_equal(cls.pooled[2], [[1000 + ord("A") + 2, 1], [1000 + ord("A") + 2, 1]])
    assert cls.pooled[0][0, 0] == 1000 + ord("A")  # the first window's <cls>, not the second's ("D")
    eos = esmc(pooling="eos").extract([RawObject("p", seqs[0]), RawObject("q", seqs[1])])
    np.testing.assert_array_equal(eos.pooled[2], [[2000 + ord("G") + 2, 1], [2000 + ord("B") + 2, 1]])  # last windows


def test_esmc_refuses_unpinned_reads_and_rounds_bf16_to_nearest_even():
    with pytest.raises(ValueError, match="name the layers"):
        EsmcExtractor({"read": "output", "pooling": "mean", "options": {"max_residues": 3}}, "cpu")
    with pytest.raises(ValueError, match="max_residues"):
        EsmcExtractor({"read": [1], "pooling": "mean", "options": {}}, "cpu")
    x = np.array([[1.0, -2.5, 3.14159, 1e-3]], np.float32)
    bits = _store_dtype(x, "bf16")
    back = torch.from_numpy(bits.astype(np.int16)).view(torch.bfloat16).float().numpy()
    np.testing.assert_array_equal(back, torch.from_numpy(x).to(torch.bfloat16).float().numpy())


class FakeTorso(torch.nn.Module):
    """``x_transformers.AttentionLayers``' ``return_hiddens`` contract: ``hiddens[i]`` is block ``i + 1``'s input and
    the output is normed (here doubled). Each block adds the first residue row (row 2) to every row, as attention
    would, so after ``L`` blocks a row has gained ``(2**L - 1)`` times the first residue."""

    def __init__(self, depth):
        super().__init__()
        self.depth = depth

    def forward(self, x, mask=None, return_hiddens=False):
        hiddens = []
        for _ in range(self.depth):
            hiddens.append(x)
            x = x + x[:, 2:3]
        return (2 * x, SimpleNamespace(hiddens=hiddens)) if return_hiddens else 2 * x


class FakeEva:
    """EVA's interface: the learned registers [1, 0] and [0, 1], then one row [code, 0] per residue, then padding
    (modality id -1)."""

    dtype = "torch.bfloat16"

    def __init__(self, depth, max_tokens):
        self.modality_info = {"tok_aa_seq": {"max_tokens": max_tokens}}
        torso = FakeTorso(depth)
        self.model = SimpleNamespace(
            num_register_tokens=2, encoder=SimpleNamespace(get_num_layers=lambda: depth, torso=torso)
        )
        self.batches = []

    def load_model(self, **kwargs):
        self.loaded = kwargs

    def input(self, batch, bypass_input_length_check):
        self.batches.append([b["aa_seq"] for b in batch])

    def encode(self, sep_encodings):
        seqs = self.batches[-1]
        x = torch.zeros(len(seqs), 2 + max(map(len, seqs)), 2)
        mask = torch.full(x.shape[:2], -1, dtype=torch.int16)
        x[:, 0, 0] = x[:, 1, 1] = 1
        mask[:, :2] = 7  # the register modality's id
        for i, s in enumerate(seqs):
            x[i, 2 : 2 + len(s), 0] = torch.tensor([float(ord(c)) for c in s])
            mask[i, 2 : 2 + len(s)] = 0
        full = self.model.encoder.torso(x, mask=None)
        return {"full": full, "mask": mask, "register": full[:, :2]}


def mimic(read, budget=8, depth=3, max_tokens=6, pooling="registers", monkeypatch=None, tmp_path=None):
    """A MIMIC extractor loaded through the fake EVA (the checkpoint's sha256 is the recipe's revision)."""
    import hashlib
    import sys

    ckpt = tmp_path / "checkpoint-final.pth"
    ckpt.write_bytes(b"weights")
    eva = FakeEva(depth, max_tokens)
    paths = SimpleNamespace(get_path=lambda **kw: str(tmp_path))
    monkeypatch.setitem(sys.modules, "central_dogma.config", SimpleNamespace(paths=paths))
    monkeypatch.setitem(sys.modules, "eva.utils.model_interface", SimpleNamespace(EVAInterface=lambda: eva))
    recipe = {
        "read": read,
        "pooling": pooling,
        "model_id": "5.7_main/E20",
        "batch_tokens": budget,
        "options": {},
        "revision": hashlib.sha256(b"weights").hexdigest(),
    }
    ex = MimicExtractor(recipe, "cpu")
    ex.load()
    return ex


def test_mimic_flattens_registers_batches_by_budget_and_skips_over_long_sequences(monkeypatch, tmp_path):
    ex = mimic("output", monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert ex.fingerprint()["output_layer"] == "3"
    out = ex.extract([RawObject("a", "MAAA"), RawObject("b", "MK"), RawObject("x", "M" * 7), RawObject("c", "MKKKK")])
    assert ex.eva.batches == [["MAAA", "MK"], ["MKKKK"]]  # 2 x 4 fits in 8; adding 5 residues would not
    assert out.object_ids == ("a", "b", "c") and out.skipped == {"x": "length>6"}  # the old script's reason
    assert list(out.pooled) == [OUTPUT_LAYER] and out.pooled[OUTPUT_LAYER].shape == (3, 4)
    m = ord("M")
    assert out.pooled[OUTPUT_LAYER][0].tolist() == [2 + 14 * m, 0, 14 * m, 2]  # 2 x ([1, 0] + 7 M, [0, 1] + 7 M)
    empty = ex.extract([RawObject("y", "")])
    assert empty.object_ids == () and empty.skipped == {"y": "empty"} and empty.pooled[OUTPUT_LAYER].shape == (0, 0)
    averaged = mimic("output", pooling="registers_mean", monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert averaged.extract([RawObject("a", "MAAA")]).pooled[OUTPUT_LAYER].tolist() == [[1 + 14 * m, 1]]
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        MimicExtractor({"read": "output", "pooling": "registers", "model_id": "a/b"}, "cuda:1")
    with pytest.raises(ValueError, match="pending"):
        MimicExtractor({"read": "pending", "pooling": "registers", "model_id": "a/b"}, "cpu")
    with pytest.raises(ValueError, match="expected one of"):
        MimicExtractor({"read": "output", "pooling": "cls", "model_id": "a/b"}, "cpu")


def test_mimic_scans_the_residue_mean_at_every_layer_and_the_registers_at_the_output_only(monkeypatch, tmp_path):
    """``mean`` = the residue positions' mean (registers and padding out) at 0..D: the torso's block inputs
    and, at D, the normed output; the register poolings read the output layer only (no register
    scan)."""
    seqs = [RawObject("a", "MAAA"), RawObject("b", "MK"), RawObject("c", "MKKKK")]
    out = mimic("all", pooling="mean", monkeypatch=monkeypatch, tmp_path=tmp_path).extract(seqs)
    assert sorted(out.pooled) == [0, 1, 2, 3]
    for i, o in enumerate(seqs):
        mean, first = np.mean([ord(c) for c in o.value]), ord(o.value[0])
        want = [[mean + (2**L - 1) * first, 0] for L in range(3)] + [[2 * (mean + 7 * first), 0]]
        np.testing.assert_allclose([out.pooled[L][i] for L in range(4)], want, rtol=1e-6)
    single = mimic((3,), pooling="mean", monkeypatch=monkeypatch, tmp_path=tmp_path).extract(seqs)
    np.testing.assert_array_equal(out.pooled[3], single.pooled[3])
    for pooling in ("registers", "registers_mean"):
        with pytest.raises(ValueError, match="reads the output layer 3 only"):
            mimic("all", pooling=pooling, monkeypatch=monkeypatch, tmp_path=tmp_path)
    with pytest.raises(ValueError, match=r"layers \[4\] outside 0..3"):
        mimic((4,), pooling="mean", monkeypatch=monkeypatch, tmp_path=tmp_path)


def test_mimic_loads_the_public_model_through_the_mimic_package_with_the_same_read(monkeypatch, tmp_path):
    """``options.source: hub`` (recipe mimic-1.0): ``mimic.load_pretrained`` at the pinned Hugging Face commit, without
    eva or central-dogma; the public model's names (``encoder``, ``embed``, ``mod_ids``) give the EVA path's vectors."""
    import sys

    seqs = [RawObject("a", "MAAA"), RawObject("b", "MK"), RawObject("x", "M" * 7)]
    want = mimic("all", pooling="mean", monkeypatch=monkeypatch, tmp_path=tmp_path).extract(seqs)
    eva, calls = FakeEva(3, 6), []

    def embed(return_register):
        enc = eva.encode(sep_encodings=False)
        return {"full": enc["full"], "mod_ids": enc["mask"], "register": enc["register"]}

    public = SimpleNamespace(
        encoder=eva.model.encoder,
        num_register_tokens=2,
        modality_info=eva.modality_info,
        autocast_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        input=eva.input,
        embed=embed,
    )

    def load_pretrained(**kw):
        calls.append(kw)
        return public

    monkeypatch.setitem(sys.modules, "mimic", SimpleNamespace(load_pretrained=load_pretrained))
    for private in ("central_dogma.config", "eva.utils.model_interface"):
        monkeypatch.setitem(sys.modules, private, None)  # importing either fails
    recipe = load_config(
        Path(__file__).resolve().parents[2] / "configs/encoders/mimic-1.0.yaml", EncoderRecipe
    ).model_dump()
    assert recipe["read"] == (16, 20) and recipe["pooling"] == "mean" and recipe["options"] == {"source": "hub"}
    ex = MimicExtractor({**recipe, "read": "all"}, "cpu")
    ex.load()
    assert calls == [{"hf_repo": "polymathic-ai/MIMIC", "revision": recipe["revision"], "device": "cpu"}]
    got = ex.extract(seqs)
    assert got.object_ids == want.object_ids == ("a", "b") and got.skipped == want.skipped == {"x": "length>6"}
    for L in range(4):
        np.testing.assert_array_equal(got.pooled[L], want.pooled[L])
    fp = ex.fingerprint()
    assert fp["checkpoint"] == f"hf:polymathic-ai/MIMIC@{recipe['revision']}" and fp["dtype"] == "torch.float32"
    assert fp["output_layer"] == "3" and fp["registers"] == "2"


def test_a_fingerprint_must_declare_its_dtype():
    from latent_interconnect.encoders.jobs import worker_origin

    with pytest.raises(ValueError, match="dtype"):
        worker_origin({"extractor": "x"}, "cpu")
    origin = worker_origin({"extractor": "x", "dtype": "bfloat16"}, "cpu")
    assert origin["fingerprint"]["dtype"] == "bfloat16" and origin["hardware"] == []


class FakePglmTokenizer:
    """One token per residue + ``<eos>`` (no ``<cls>``), right-padded."""

    padding_side = "right"

    def __call__(self, seqs, add_special_tokens, padding, return_tensors):
        assert add_special_tokens and padding
        width = max(len(s) for s in seqs) + 1
        mask = torch.zeros(len(seqs), width, dtype=torch.long)
        ids = torch.zeros_like(mask)
        for i, s in enumerate(seqs):
            mask[i, : len(s) + 1] = 1
            ids[i, : len(s)] = torch.tensor([ord(c) for c in s])
            ids[i, len(s)] = 2  # <eos>
        return {"input_ids": ids, "attention_mask": mask}


class FakePglm:
    """Sequence-first states ``[T, n, d]``: at layer L a residue with code c is (c + L, 1); <eos> and padding huge."""

    config = SimpleNamespace(num_layers=2, hidden_size=2)

    def __call__(self, input_ids, attention_mask, output_hidden_states):
        assert output_hidden_states
        code = input_ids.float().masked_fill(input_ids == 0, 1e6).masked_fill(input_ids == 2, 2e6)  # pad 0, <eos> 2
        states = [torch.stack([code + L, torch.ones_like(code)], dim=-1).transpose(0, 1) for L in range(3)]
        return SimpleNamespace(hidden_states=states)

    def to(self, device):
        return self

    def eval(self):
        return self


def pglm_recipe(read=(0, 2), residues=False):
    return {
        "read": read,
        "pooling": "mean",
        "options": {"max_residues": 3},
        "batch_tokens": 8,
        "residues": residues,
        "residue_dtype": "fp32",
        "model_id": "m",
        "revision": "r",
    }


def test_pglm_pools_sequence_first_states_over_residues_and_windows(monkeypatch):
    import transformers

    from latent_interconnect.encoders.extractors.pglm import PglmExtractor

    seen = {}
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakePglmTokenizer())

    def model(repo, **kwargs):
        seen.update(kwargs, repo=repo)
        return FakePglm()

    monkeypatch.setattr(transformers.AutoModelForMaskedLM, "from_pretrained", model)
    ex = PglmExtractor(pglm_recipe(read="all", residues=True), "cpu")
    ex.load()
    assert ex.layers == (0, 1, 2)  # all hidden_states indices: embeddings .. the final post-LayerNorm output
    assert seen["revision"] == "r" and seen["trust_remote_code"] and seen["dtype"] == torch.float32
    seqs = ["ABCDEFG", "AB"]  # 7 residues: windows of 3, 3, 1
    out = ex.extract([RawObject("p", seqs[0]), RawObject("q", seqs[1])])
    expect = np.array([np.mean([ord(c) for c in s]) for s in seqs])
    for L in (0, 1, 2):
        np.testing.assert_allclose(out.pooled[L][:, 0], expect + L, rtol=1e-6)
    np.testing.assert_array_equal(out.residues[2][0][:, 0], [ord(c) + 2 for c in seqs[0]])
    assert ex.fingerprint()["dtype"] == "float32"
    ex.recipe = {**ex.recipe, "pooling": "eos", "residues": False}  # the <eos> state, not padding (1e6)
    eos = ex.extract([RawObject("p", seqs[0]), RawObject("q", seqs[1])])
    np.testing.assert_array_equal(eos.pooled[1], [[2e6 + 1, 1]] * 2)


def test_pglm_refuses_unpinned_reads_other_poolings_and_out_of_range_layers(monkeypatch):
    import transformers

    from latent_interconnect.encoders.extractors.pglm import PglmExtractor

    with pytest.raises(ValueError, match="name the layers"):
        PglmExtractor(pglm_recipe(read="output"), "cpu")
    with pytest.raises(ValueError, match="pooling"):
        PglmExtractor({**pglm_recipe(), "pooling": "cls"}, "cpu")
    with pytest.raises(ValueError, match="max_residues"):
        PglmExtractor({**pglm_recipe(), "options": {}}, "cpu")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakePglmTokenizer())
    monkeypatch.setattr(transformers.AutoModelForMaskedLM, "from_pretrained", lambda *a, **k: FakePglm())
    with pytest.raises(ValueError, match=r"layers \[3\] outside 0..2"):
        PglmExtractor(pglm_recipe(read=(3,)), "cpu").load()


@pytest.mark.parametrize(("part_size", "parts"), [(2, 2), (1, 3)])  # 1: the skipped object's part holds nothing else
def test_a_skipped_object_does_not_fail_its_part_and_reaches_the_artifact(tmp_path, part_size, parts):
    """The toy extractor skips an unreadable object: the part is written, the artifact lists and counts the skip."""
    from tests.fakes import toy_recipe

    recipe = toy_recipe((0,)).model_copy(update={"plugins": ("tests.fakes",)}).model_dump(mode="json")
    objects = [RawObject("a", "MKCL"), RawObject("b", "MK!L"), RawObject("c", "MSTN")]
    sha = write_objects(tmp_path / "objects.jsonl", objects)
    j = ExtractionJob(recipe, str(tmp_path / "objects.jsonl"), sha, part_size, str(tmp_path / "store"))
    assert run_worker(j, "cpu") == parts
    meta = ArtifactStore(tmp_path / "store").manifest(finalize(j)).meta
    assert meta["skipped"] == {"b": "unreadable: '!'"} and meta["n_skipped"] == 1
    assert EmbeddingReader(ArtifactStore(tmp_path / "store"), finalize(j)).object_ids == ("a", "c")


def test_an_extractor_that_drops_or_reorders_objects_is_refused(tmp_path, monkeypatch):
    from tests.fakes import ToyExtractor, toy_recipe

    recipe = toy_recipe((0,)).model_copy(update={"plugins": ("tests.fakes",)}).model_dump(mode="json")
    objects = [RawObject("a", "MKCL"), RawObject("b", "MKAL")]
    sha = write_objects(tmp_path / "objects.jsonl", objects)
    j = ExtractionJob(recipe, str(tmp_path / "objects.jsonl"), sha, 2, str(tmp_path / "store"))
    real = ToyExtractor.extract
    monkeypatch.setattr(ToyExtractor, "extract", lambda self, objs: real(self, objs[::-1]))
    with pytest.raises(RuntimeError, match="not each once in order"):
        run_worker(j, "cpu")


def test_pool_windows_batches_under_the_budget_and_keeps_each_objects_residue_order():
    """The shared windowing (ESM-C, PGLM): windows sorted by length, packed while n * (m + specials) <= budget."""
    from latent_interconnect.encoders.extractors._windows import pool_windows

    calls = []

    def forward(seqs):
        calls.append(list(seqs))
        width = max(len(s) for s in seqs)
        states = torch.full((len(seqs), width, 1), 1e6)  # padding: huge, so a leak shows
        mask = torch.zeros((len(seqs), width), dtype=torch.bool)
        for i, s in enumerate(seqs):
            states[i, : len(s), 0] = torch.tensor([float(ord(c)) for c in s])
            mask[i, : len(s)] = True
        return [states], mask

    objects = [RawObject("p", "ABCDE"), RawObject("q", "WXY")]  # windows of 2: AB CD E | WX Y
    pooled, residues = pool_windows(
        objects, forward, layers=(0,), width=1, max_residues=2, specials=1, budget=6, keep_residues=True, name="t"
    )
    assert calls == [["E", "Y"], ["AB", "CD"], ["WX"]]  # (1+1)*2 = 4 <= 6; then 2 * 3 = 6; a third would cost 9
    np.testing.assert_allclose(pooled[0][:, 0], [np.mean([ord(c) for c in "ABCDE"]), np.mean([ord(c) for c in "WXY"])])
    np.testing.assert_array_equal(residues[0][0][:, 0], [ord(c) for c in "ABCDE"])  # windows put back in order
    np.testing.assert_array_equal(residues[0][1][:, 0], [ord(c) for c in "WXY"])

    def short(seqs):
        states, mask = forward(seqs)
        mask[:, 0] = False  # the model's mask loses a residue per window
        return states, mask

    with pytest.raises(RuntimeError, match="residue counts"):
        pool_windows(
            objects, short, layers=(0,), width=1, max_residues=2, specials=1, budget=6, keep_residues=False, name="t"
        )


def test_reduce_batch_sums_bf16_states_in_float64_over_the_residues_only():
    """The device-side reduction: float64 sums of the bf16 states at the residue positions only (a NaN or inf at a
    special or padding position never reaches them), float32 rows."""
    from latent_interconnect.encoders.extractors._windows import reduce_batch

    gen = torch.Generator().manual_seed(0)
    states = [(torch.randn(3, 5, 4, generator=gen) * 100).to(torch.bfloat16) for _ in range(2)]
    residue = torch.tensor([[0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 1, 1, 1]], dtype=torch.bool)
    states[0][0, 4] = float("nan")  # <eos>
    states[1][1, 3] = float("inf")  # padding
    states[1][0, 0] = float("-inf")  # <cls>
    sums, counts, rows = reduce_batch(states, residue, keep_residues=True)
    assert counts.tolist() == [3, 1, 4] and all(s.dtype == np.float64 for s in sums)
    for li, s in enumerate(states):
        exact = s.double().numpy()
        for i in range(3):
            assert np.isfinite(sums[li][i]).all()
            np.testing.assert_allclose(sums[li][i], exact[i][residue[i].numpy()].sum(0), rtol=1e-12)
            assert rows[li][i].dtype == np.float32
            np.testing.assert_array_equal(rows[li][i], s[i][residue[i]].float().numpy())
    assert reduce_batch(states, residue, keep_residues=False)[2] == []


def test_esmc_loads_the_pinned_revision_in_the_device_precision_and_reads_every_layer(monkeypatch):
    """ESM-C 6B's scan: the model loaded at the recipe's revision with SDPA, in float32 on CPU (bf16 on CUDA, as the
    original 6B extractor loaded it); ``read: all`` resolves to hidden_states 0..80 for an 80-block model."""
    import sys

    seen = {}

    class Model:
        config = SimpleNamespace(num_hidden_layers=80, hidden_size=2560)

        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            seen.update(kwargs, repo=repo)
            return cls()

        def eval(self):
            return self

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, repo, revision):
            seen["tokenizer"] = (repo, revision)
            return cls()

    monkeypatch.setitem(sys.modules, "esm.models.esmc.model", SimpleNamespace(EsmcModel=Model))
    monkeypatch.setitem(sys.modules, "esm.models.esmc.tokenizer", SimpleNamespace(EsmcTokenizer=Tokenizer))
    recipe = {"read": "all", "pooling": "mean", "options": {"max_residues": 2046}, "model_id": "m6b", "revision": "r"}
    ex = EsmcExtractor(recipe, "cpu")
    ex.load()
    assert seen == {
        "revision": "r",
        "device": "cpu",
        "dtype": torch.float32,
        "attn_implementation": "sdpa",
        "repo": "m6b",
        "tokenizer": ("m6b", "r"),
    }
    assert ex.layers == tuple(range(81))
    with pytest.raises(ValueError, match=r"layers \[81\] outside 0..80"):
        EsmcExtractor({**recipe, "read": [80, 81]}, "cpu").load()
