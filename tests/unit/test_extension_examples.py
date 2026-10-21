"""The worked pipeline's helpers (``examples/guides/extending.md`` §6): ``examples/embed.py`` reads a bundle or a
recipe, and ``examples/make_domain.py`` writes source files the domain stage builds a domain from;
``examples/summarize_suites.py`` prints the suites' summary."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from examples.embed import main as embed_main
from examples.embed import read_extraction
from examples.make_domain import read_fasta, write_sources
from examples.summarize_suites import summary_lines

from latent_interconnect.config import load_config
from latent_interconnect.run.domain import DomainBuild

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "examples" / "configs" / "pipeline"
DATA = ROOT / "examples" / "data"


def test_embed_reads_a_recipe_yaml():
    recipe, layers = read_extraction(PIPELINE / "esmc-300m.yaml")
    assert layers == [30]
    assert recipe["model_id"] == "biohub/ESMC-300M" and recipe["extractor"] == "esmc"


def test_embed_reads_a_bundle_like_the_recipe(tmp_path):
    recipe, _ = read_extraction(ROOT / "configs" / "encoders" / "esmc-600m-l36.yaml")
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"extraction": {"recipe": recipe, "read_layers": [36, 33]}}))
    from_bundle, layers = read_extraction(bundle)
    assert from_bundle == recipe
    assert layers == [36, 33]  # the connector's order, not the recipe's


TOY_PLUGIN = """
import numpy as np
from latent_interconnect.encoders.base import EXTRACTORS, ExtractedBatch


@EXTRACTORS.register("toy-third-party")
class Toy:
    def __init__(self, recipe, device):
        self.recipe = recipe

    def load(self):
        pass

    def fingerprint(self):
        return {"extractor": "toy-third-party", "dtype": "float32"}

    def extract(self, objects):
        layers = [int(x) for x in self.recipe["read"]]
        rows = {o.object_id: np.array([[ord(c), i] for i, c in enumerate(o.value)], np.float32) for o in objects}
        pooled = {L: np.stack([rows[o.object_id].mean(0) + L for o in objects]) for L in layers}
        residues = {L: [rows[o.object_id] + L for o in objects] for L in layers} if self.recipe.get("residues") else {}
        return ExtractedBatch(tuple(o.object_id for o in objects), pooled, residues)
"""


@pytest.mark.parametrize("residues", [False, True])
def test_embed_loads_a_third_party_extractor_from_the_recipe_plugins(residues, tmp_path, monkeypatch, capsys):
    """The recipe's ``plugins`` are imported before the extractor is resolved (else
    ``unknown extractor 'esm2'``); the vector is the read layers in the recipe's order; residues are written too."""
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "toy_third_party_ext.py").write_text(TOY_PLUGIN)
    monkeypatch.syspath_prepend(str(plugins))  # a plugin is any importable module (PYTHONPATH)
    recipe = tmp_path / "toy.yaml"
    recipe.write_text(
        "name: toy\nextractor: toy-third-party\ndomain: protein\nmodel_id: none\nrevision: v1\nread: [3, 1]\n"
        f"pooling: mean\nenvironment: lip\nbatch_tokens: 64\nresidues: {str(residues).lower()}\n"
        "plugins: [toy_third_party_ext]\n"
    )
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">a\nMK\n>b\nACD\n")
    embed_main([str(recipe), str(fasta), str(tmp_path / "out")])
    a = np.load(tmp_path / "out" / "a.npy")
    np.testing.assert_allclose(a, [(77 + 75) / 2 + 3, 0.5 + 3, (77 + 75) / 2 + 1, 0.5 + 1])
    res = tmp_path / "out" / "b.residues.npy"
    assert res.exists() == residues
    if residues:
        b = np.load(res)
        assert b.shape == (3, 4) and b.dtype == np.float32
        np.testing.assert_allclose(b[:, 1], [3, 4, 5])  # layer 3's position column, then layer 1's
        np.testing.assert_allclose(b[:, 3], [1, 2, 3])
    assert "a: 4 values" in capsys.readouterr().out


def test_embed_help_runs_as_a_script(tmp_path):
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "embed.py"), "--help"], capture_output=True, text=True, check=True
    ).stdout
    assert "any extractor" in out


def test_the_esm2_example_recipe_names_its_plugin():
    from latent_interconnect.encoders.base import EXTRACTORS
    from latent_interconnect.encoders.recipes import EncoderRecipe
    from latent_interconnect.plugins import load_plugins

    recipe = load_config(ROOT / "examples" / "configs" / "esm2-8m.yaml", EncoderRecipe)
    load_plugins(recipe.plugins)
    assert recipe.environment == "lip" and "esm2" in EXTRACTORS


def test_read_fasta_takes_uniprot_ids_and_names(tmp_path):
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">sp|P69905|HBA_HUMAN Hemoglobin subunit alpha OS=Homo sapiens OX=9606\nMVLS\nPADK\n>plain\nMK\n")
    assert read_fasta(fasta) == {"P69905": ("Hemoglobin subunit alpha", "MVLSPADK"), "plain": ("plain", "MK")}


def test_make_domain_builds_the_example_domain(tmp_path):
    config = yaml.safe_load((PIPELINE / "domain.yaml").read_text())
    config["spec"].update(write_sources(DATA / "proteins.fasta", DATA / "proteins.tsv", tmp_path, "mine/proteins"))
    out = tmp_path / "domain.yaml"
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    build = load_config(out, DomainBuild)
    tables = build.spec.tables(tmp_path)
    assert len(tables.objects) == 32
    assert tables.split.value_counts().to_dict() == {"train": 24, "val": 4, "test": 4}
    assert tables.objects.loc["P62805", "name"] == "Histone H4"
    assert bool(tables.labels.loc["P62805", "kw:DNA-binding"]) and bool(tables.labels.loc["P62805", "lineage:Mammalia"])
    assert not tables.labels.loc["P0A7T7", "lineage:Mammalia"]


STAGE_CONFIGS = {
    "extract-res.yaml": "extract",
    "capture-staggered.yaml": "capture",
    "ridge-staggered.yaml": "ridge",
    "init-mlp.yaml": "init",
    "init-xattn.yaml": "init",
    "train-mlp.yaml": "train",
    "train-xattn.yaml": "train",
}


@pytest.mark.parametrize("name", sorted(STAGE_CONFIGS))
def test_the_connector_kind_variants_load_and_validate(name):
    """§6's mlp and staggered-xattn variants: each stage config loads with its connector, recipe and receiver."""
    import importlib

    from latent_interconnect.config_index import cited_pins, stand_in_pins

    stage = importlib.import_module(f"latent_interconnect.run.{STAGE_CONFIGS[name]}").STAGE
    cfg = load_config(PIPELINE / name, stage.config, stand_in_pins(cited_pins(PIPELINE / name)))
    if hasattr(cfg, "connector"):
        assert cfg.connector.input.recipe == cfg.encoder.name


@pytest.mark.parametrize(
    ("name", "kind", "layout"),
    [
        ("connector-mlp.yaml", "mlp", [[4, 8, 12, 16]]),
        ("connector-xattn-3tok-staggered.yaml", "xattn", [[4, 10, 16], [6, 12, 18], [8, 14, 20]]),
    ],
)
def test_the_example_connector_kinds(name, kind, layout):
    """Both example connectors build from a ridge fit at their layers and write what their kind reads."""
    from latent_interconnect.bundle_connector import connector_layout
    from latent_interconnect.encoders.recipes import EncoderRecipe
    from latent_interconnect.writes import Connector, ConnectorSpec, ObjectBatch
    from latent_interconnect.writes.init import RidgeLayer, RidgeSolution

    spec = load_config(PIPELINE / name, ConnectorSpec)
    recipe = load_config(PIPELINE / f"{spec.input.recipe}.yaml", EncoderRecipe)
    assert (spec.kind, spec.layers.per_token) == (kind, tuple(tuple(t) for t in layout))
    assert recipe.residues == (kind == "xattn")
    info = connector_layout(spec.model_dump(mode="json"), 24)
    assert info["write_layers"] == layout and info["input"] == ("residues" if kind == "xattn" else "pooled")
    d_in, d_model = 960, 896  # ESM-C 300M, Qwen2.5-0.5B
    g = np.random.default_rng(0)
    fit = {
        L: RidgeLayer(g.normal(size=(d_model, d_in)).astype(np.float32) * 0.01, np.zeros(d_model, np.float32))
        for L in {0, *(L for t in layout for L in t)}
    }
    ridge = RidgeSolution("name", np.zeros(d_in, np.float32), np.ones(d_in, np.float32), fit)
    connector = Connector.from_ridge(spec, 24, {"name": ridge})
    import torch

    x = torch.randn(1, 7, d_in)
    batch = ObjectBatch(("a",), pooled=x.mean(1), residues=x, mask=torch.ones(1, 7, dtype=torch.bool))
    writes = connector(batch)
    assert {e.layer for e in writes.deltas} == {0, *(L for t in layout for L in t)}


def test_summarize_suites_prints_accuracy_and_auroc():
    import pandas as pd

    scores = pd.DataFrame(
        {
            "condition.object": ["own"] * 4,
            "task": ["DNA-binding"] * 4,
            "label": [1.0, 1.0, 0.0, 0.0],
            "score": [2.0, -1.0, -1.0, -3.0],  # a tie between a positive and a negative counts half
        }
    )
    assert summary_lines(scores, "suite-init") == ["suite-init     own  DNA-binding  n=4 pos=2 acc=0.75 AUROC=0.88"]
