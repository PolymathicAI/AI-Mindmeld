"""The Rhea cross-domain test (the paper's Fig 8c): reaction participation in swap quads, cells novP-famM (novel
proteins with familiar molecules) and novP-novM (novel proteins with novel molecules), for the released Kimi K2.6
multi-token MLP pair at step 2000 (the paper's pair and step).

Checked on the configs:

- each suite spec is the paper's Rhea cell spec (task, slots, conditions, wording, inline answer words) on the protein
  and molecule domains and the cell's item set, by its pin (``items/rhea-novP-<cell>``);
- each suite scores the pair's released checkpoints, by pin, on the Rhea embeddings, by pin;
- each analysis is the paper's analysis of its suite (nine cells, six contrasts, swap-quad units, shared-unit
  interval) on the suite's record, at the suite's step;
- every config loads.
"""

from pathlib import Path

import pytest

from latent_interconnect.config import load_config, load_resolved
from latent_interconnect.config_index import cited_pins, stand_in_pins
from latent_interconnect.evaluation.analysis import AnalysisSpec
from latent_interconnect.evaluation.suites import SuiteRunConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
CELLS = {"famM": {"$pin": "items/rhea-novP-famM"}, "novM": {"$pin": "items/rhea-novP-novM"}}
"""Each cell's Rhea item set, by its pin (a release data bundle installs it)."""
INPUTS = {
    "p": [{"$pin": "embeddings/esmc-6b/rhea-proteins"}],  # ESM-C 6B [51|80] of the Rhea proteins
    "m": [{"$pin": "embeddings/chemfm-3b/rhea-molecules"}],  # ChemFM-3B L27 of the Rhea molecules
}
"""The embeddings of the Rhea objects, by pin (a release data bundle installs them)."""
KIMI_PLAN = {"planner": "length-stable/v1", "budget_tokens": 32768}
RECEIVER, PAIR, STEP = "kimi-k2.6", "esmc-6b-mlp4-staggered_x_chemfm-3b-mlp4", 2000
CHECKPOINTS = {
    "p": {"$pin": "connectors/kimi-k2.6/esmc-6b-mlp4-staggered/step_002000"},
    "m": {"$pin": "connectors/kimi-k2.6/chemfm-3b-mlp4/step_002000"},
}
"""The paper's pair, its step and its released checkpoints."""


def _stem(cell: str) -> str:
    return f"{PAIR}-xdom-rhea-novP-{cell}-trained"


def _destination(cell: str) -> str:
    return f"connectors/{RECEIVER}/{PAIR}/xdom-rhea-novP-{cell}/trained"


SUITES = sorted(f"suites/{RECEIVER}/{_stem(c)}.yaml" for c in CELLS)
ANALYSES = sorted(f"analyses/{RECEIVER}/{_stem(c)}.yaml" for c in CELLS)

QUESTION = "Does this protein catalyse a reaction in which this compound is a substrate or a product?"
SENTENCES = {
    "p": {
        "own": "Here is an unknown protein, given to you as an embedding:<SLOT:p>.",
        "placeholder": "Here is an unknown protein, given to you as an embedding:<SLOT:p>.",
        "name": "Here is a protein, given by its name:<SLOT:p>.",
    },
    "m": {
        "own": "Here is an unknown compound, given to you as an embedding:<SLOT:m>.",
        "placeholder": "Here is an unknown compound, given to you as an embedding:<SLOT:m>.",
        "name": "Here is a compound, given by its name:<SLOT:m>.",
        "value": "Here is a compound, given as a SMILES string:<SLOT:m>.",
    },
}
PAPER_CELLS = {  # cell -> (protein condition, molecule condition)
    "D own-own": ("own", "own"),
    "A name-value": ("name", "value"),
    "B name-own": ("name", "own"),
    "C own-value": ("own", "value"),
    "name-name": ("name", "name"),
    "own-name": ("own", "name"),
    "placeholder-own": ("placeholder", "own"),
    "own-placeholder": ("own", "placeholder"),
    "placeholder-placeholder": ("placeholder", "placeholder"),
}
"""The paper's cross-domain cells, in its spec's condition order (Fig 8c plots D own-own)."""
PAPER_CONTRASTS = [
    {"name": "B minus A", "a": "B name-own", "b": "A name-value"},
    {"name": "D minus C", "a": "D own-own", "b": "C own-value"},
    {"name": "C minus A", "a": "C own-value", "b": "A name-value"},
    {"name": "D minus B", "a": "D own-own", "b": "B name-own"},
    {"name": "D minus A", "a": "D own-own", "b": "A name-value"},
    {"name": "own minus names", "a": "D own-own", "b": "name-name"},
]


def _resolved(path: Path):
    return load_resolved(path, unknown_pins="keep")


def _paper_spec(cell: str) -> dict:
    """The paper's Kimi K2.6 Rhea spec of ``cell`` on this repository's domains and the cell's item set."""
    return {
        "id": f"xdom/rhea/novP-{cell}",
        "kind": "cross_domain",
        "domains": [
            {"name": "protein", "data": {"$pin": "protein-domain"}},
            {"name": "molecule", "data": {"$pin": "molecule-domain"}},
        ],
        "item_set": CELLS[cell],
        "tasks": ["participation"],
        "slots": [
            {"name": "p", "role": "protein", "connector": "p"},
            {"name": "m", "role": "molecule", "connector": "m"},
        ],
        "conditions": [{"protein": p, "molecule": m} for p, m in PAPER_CELLS.values()],
        "thinking": False,
        "capability": "score",
        "params": {
            "user": f"{{p}} {{m}} {QUESTION} Answer Yes or No.",
            "sentences": SENTENCES,
            "yes": "Yes",
            "no": "No",
        },
    }


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_the_spec_is_the_papers_cell_spec(cell):
    assert _resolved(CONFIGS / "suite_specs" / RECEIVER / f"xdom-rhea-novP-{cell}.yaml") == _paper_spec(cell)


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_each_suite_is_the_papers_pair_suite(cell):
    mine = _resolved(CONFIGS / "suites" / RECEIVER / f"{_stem(cell)}.yaml")
    assert mine["destination"] == _destination(cell)
    assert mine["suite"] == _resolved(CONFIGS / "suite_specs" / RECEIVER / f"xdom-rhea-novP-{cell}.yaml")
    assert mine["receiver"] == _resolved(CONFIGS / "receivers" / f"{RECEIVER}.yaml")
    assert mine["checkpoints"] == CHECKPOINTS and mine["step"] == STEP
    assert mine["inputs"] == INPUTS and mine["device_class"] == "A100" and mine["plan"] == KIMI_PLAN
    assert set(mine) == {"destination", "suite", "receiver", "checkpoints", "inputs", "step", "device_class", "plan"}


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_each_analysis_is_the_papers_on_its_suites_record(cell):
    mine = _resolved(CONFIGS / "analyses" / RECEIVER / f"{_stem(cell)}.yaml")
    suite = _destination(cell)
    assert mine["destination"] == f"{suite}/cross-domain-quads"
    assert mine["kind"] == "cross_domain_quads" and mine["inputs"] == {"xd": {"$pin": f"{suite}@record"}}
    assert mine["id"] == f"xdom/rhea/novP-{cell}/{RECEIVER}/{PAIR}"
    assert mine["params"] == {
        "cells": {
            name: {
                "encoder": "esmc-6b-l51l80 x chemfm-3b",
                "inputs": ["xd"],
                "step": STEP,
                "conditions": {"protein": p, "molecule": m},
            }
            for name, (p, m) in PAPER_CELLS.items()
        },
        "contrasts": PAPER_CONTRASTS,
    }
    assert mine["units"] == {"factors": ["protein_cluster", "molecule_group"], "interval": "shared_unit"}


@pytest.mark.parametrize("path", SUITES + ANALYSES)
def test_each_config_loads(path):
    config = CONFIGS / path
    load_config(
        config, SuiteRunConfig if path.startswith("suites/") else AnalysisSpec, stand_in_pins(cited_pins(config))
    )
