"""The 1.1.0 configs resolve on the pins the release installs, and only on those (config resolution only).

The pins are the 1.0.0 bundles' (``test_train_from_release``: the ESM-C 6B, ChemFM-3B and Kimi K2.6 ChemFM-3B mlp
connectors' inputs and checkpoints) and the 1.1.0 bundles' names listed once in :data:`SHIPPED`; an analysis also cites
the record of the suite it reads (``<suite destination>@record``), which a run of that suite config makes. Checked:

- every config in :data:`CONFIGS` cites only those names and loads on stand-ins for exactly them;
- every name in :data:`SHIPPED` is cited, so the list is the set the 1.1.0 bundles must install;
- the chat entries of the four all-tasks connectors (``configs/chat/all-connectors.yaml``) need only those names,
  their checkpoint each a :data:`SHIPPED` connector pin;
- the configs' destinations index without a clash (``config_index.index``; ``configs/chat`` is not indexed: its run
  copies declare their connector's published destination on purpose).
"""

import json
from pathlib import Path

import pytest
from tests.test_train_from_release import CHEMFM3B, ESMC6B, MOL_MLP

from latent_interconnect.chat import cli
from latent_interconnect.chat.config import ChatConfig
from latent_interconnect.config import load_config, load_yaml
from latent_interconnect.config_index import cited_pins, index, stage_of, stages, stand_in_pins

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

SHIPPED = {
    "protein-training-set-all-tasks/training_set": (
        "the protein training set on all tasks (configs/training_sets/protein-all-tasks.yaml)"
    ),
    "molecule-training-set-all-tasks/training_set": (
        "the molecule training set on all tasks (configs/training_sets/molecule-all-tasks.yaml)"
    ),
    "connectors/kimi-k2.6/esmc-6b-mlp4-staggered-all-tasks/step_004800": (
        "Kimi K2.6 x ESM-C 6B four-token MLP, all tasks: step 4800, pinned by its run's publication "
        "(runs/kimi-k2.6/esmc-6b-mlp4-staggered-all-tasks-achieved.yaml)"
    ),
    "connectors/kimi-k2.6/chemfm-3b-mlp4-all-tasks/step_004800": (
        "Kimi K2.6 x ChemFM-3B four-token MLP, all tasks: step 4800 of runs/kimi-k2.6/chemfm-3b-mlp4-all-tasks.yaml"
    ),
    "connectors/gemma-3-12b-it/esmc-6b-mlp4-all-tasks/step_005000": (
        "Gemma 3 12B x ESM-C 6B four-token MLP, all tasks: step 5000 of runs/gemma-3-12b-it/esmc-6b-mlp4-all-tasks.yaml"
    ),
    "connectors/gemma-3-12b-it/chemfm-3b-mlp4-all-tasks/step_005000": (
        "Gemma 3 12B x ChemFM-3B four-token MLP, all tasks: step 5000 of "
        "runs/gemma-3-12b-it/chemfm-3b-mlp4-all-tasks.yaml"
    ),
    "items/rhea-novP-famM": "the Rhea cross-domain item set, cell novP-famM (novel proteins, familiar molecules)",
    "items/rhea-novP-novM": "the Rhea cross-domain item set, cell novP-novM (novel proteins, novel molecules)",
    "embeddings/esmc-6b/rhea-proteins": "ESM-C 6B [51|80] pooled embeddings of the Rhea item sets' proteins",
    "embeddings/chemfm-3b/rhea-molecules": "ChemFM-3B L27 last+mean embeddings of the Rhea item sets' molecules",
}
"""Every pin name the 1.1.0 data and connector bundles add, with what it holds: the one list of them."""
RELEASED = set(ESMC6B) | set(CHEMFM3B) | set(MOL_MLP)
"""The 1.0.0 bundles' pin names the 1.1.0 configs also cite."""

_RHEA = [
    f"{kind}/kimi-k2.6/esmc-6b-mlp4-staggered_x_chemfm-3b-mlp4-xdom-rhea-novP-{cell}-trained.yaml"
    for kind in ("suites", "analyses")
    for cell in ("famM", "novM")
]
_SINGLE = [
    f"suites/{receiver}/{connector}-{suite}-trained.yaml"
    for receiver, connector, domain in (
        ("kimi-k2.6", "esmc-6b-mlp4-staggered-all-tasks", "protein"),
        ("kimi-k2.6", "chemfm-3b-mlp4-all-tasks", "molecule"),
        ("gemma-3-12b-it", "esmc-6b-mlp4-all-tasks", "protein"),
        ("gemma-3-12b-it", "chemfm-3b-mlp4-all-tasks", "molecule"),
    )
    for suite in (
        [f"molecule-{k}-novel-scaffold" for k in ("access", "transfer")]
        if domain == "molecule"
        else [f"protein-{k}" for k in ("access", "transfer")]
    )
]
CONFIGS_NEW = sorted(
    [
        "training_sets/protein-all-tasks.yaml",
        "training_sets/molecule-all-tasks.yaml",
        "runs/kimi-k2.6/esmc-6b-mlp4-staggered-all-tasks.yaml",
        "runs/kimi-k2.6/esmc-6b-mlp4-staggered-all-tasks-achieved.yaml",
        "runs/kimi-k2.6/chemfm-3b-mlp4-all-tasks.yaml",
        "runs/gemma-3-12b-it/esmc-6b-mlp4-all-tasks.yaml",
        "runs/gemma-3-12b-it/chemfm-3b-mlp4-all-tasks.yaml",
        *_SINGLE,
        *_RHEA,
    ]
)
"""The 1.1.0 stage configs (relative to ``configs``)."""
CHAT = {
    "kimi-k2.6": ["esmc-6b-mlp4-staggered-all-tasks", "chemfm-3b-mlp4-all-tasks"],
    "gemma-3-12b-it": ["esmc-6b-mlp4-all-tasks", "chemfm-3b-mlp4-all-tasks"],
}
"""The four all-tasks connectors' entries in ``configs/chat/all-connectors.yaml``."""


def _index():
    return index([CONFIGS], exclude=[CONFIGS / "chat"])


def _produced(names, configs) -> set[str]:
    """The ``@record`` names among ``names`` that one of ``configs`` makes (a suite record an analysis reads)."""
    found = _index()
    paths = {(CONFIGS / c).resolve() for c in configs}
    out = set()
    for n in names:
        producer = found.resolve(n) if n.endswith("@record") else None
        if producer is not None and producer.path in paths:
            out.add(n)
    return out


def test_the_shipped_names_are_new():
    assert not set(SHIPPED) & RELEASED and len(CONFIGS_NEW) == 19


@pytest.mark.parametrize("rel", CONFIGS_NEW)
def test_each_config_resolves_on_the_released_and_shipped_pins_only(rel):
    path = CONFIGS / rel
    cited = set(cited_pins(path))
    allowed = RELEASED | set(SHIPPED) | _produced(cited, CONFIGS_NEW)
    assert cited <= allowed, sorted(cited - allowed)
    config_class = stage_of(load_yaml(path.read_text()), stages()).config
    load_config(path, config_class, stand_in_pins(allowed))


def test_every_shipped_name_is_cited():
    cited = set().union(*(cited_pins(CONFIGS / rel) for rel in CONFIGS_NEW))
    assert set(SHIPPED) <= cited


def test_the_chat_entries_need_only_released_and_shipped_pins(tmp_path):
    path = CONFIGS / "chat" / "all-connectors.yaml"
    missing = cli.missing_pins(path, {})
    for receiver, connectors in CHAT.items():
        for cid in connectors:
            assert set(missing[receiver][cid]) <= RELEASED | set(SHIPPED), (receiver, cid)
        narrowed = tmp_path / f"{receiver}.yaml"
        narrowed.write_text(json.dumps(cli.narrowed_config(path, receiver, connectors)))
        config = load_config(narrowed, ChatConfig, stand_in_pins(RELEASED | set(SHIPPED)))
        for cid in connectors:
            assert config.receivers[receiver].connectors[cid].checkpoint_pin in SHIPPED, (receiver, cid)


def test_the_config_destinations_index_without_a_clash():
    found = _index()
    for rel in CONFIGS_NEW:
        name = load_yaml((CONFIGS / rel).read_text())["destination"]
        assert found.resolve(name).path == (CONFIGS / rel).resolve(), rel
    for name in SHIPPED:  # a connector or training-set pin is made by its config; items and embeddings by no config
        producer = found.resolve(name)
        assert (producer is None) == name.startswith(("items/", "embeddings/")), name
