"""The ``all_tasks`` class of ``release/library-ranking.json`` and its totals in ``library/README.md`` (offline).

Checked: the class holds a description and its bundles, apart from the ranked bundles; each entry has the fields of a
ranked bundle's entry and its total is the ranking's equal-weight total; the library README shows each total, to two
decimals, marked ``*``.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RANKING = json.loads((ROOT / "release" / "library-ranking.json").read_text())
ALL_TASKS = RANKING["all_tasks"]
BUNDLES = {b["bundle"]: b for b in ALL_TASKS["bundles"]}
NAMES = [
    "kimi-k2.6__esmc-6b-mlp4-staggered-all-tasks",
    "kimi-k2.6__chemfm-3b-mlp4-all-tasks",
    "gemma-3-12b-it__esmc-6b-mlp4-all-tasks",
    "gemma-3-12b-it__chemfm-3b-mlp4-all-tasks",
]


def test_the_class_is_listed_apart_from_the_ranked_bundles():
    assert set(ALL_TASKS) == {"description", "bundles"} and list(BUNDLES) == NAMES
    assert not set(BUNDLES) & {b["bundle"] for b in RANKING["bundles"]}


@pytest.mark.parametrize("name", NAMES)
def test_each_entry_has_a_ranked_entrys_fields(name):
    bundle = BUNDLES[name]
    assert set(bundle) in [set(b) for b in RANKING["bundles"]]
    assert set(bundle["sources"]) == {"trained", "unseen"}
    groups = RANKING["unseen_groups"][bundle["domain"]]
    assert list(bundle["unseen_group_aurocs"]) == list(groups)
    unseen = sum(bundle["unseen_group_aurocs"].values()) / len(groups)
    assert bundle["total_auroc"] == pytest.approx((bundle["trained_auroc"] + unseen) / 2, abs=1e-12)


def test_the_library_readme_shows_each_total_marked():
    text = (ROOT / "library" / "README.md").read_text()
    for name in NAMES:
        (line,) = [x for x in text.splitlines() if x.startswith("|") and f"[{name}]" in x]
        cells = [c.strip() for c in line.strip("|").split("|")]
        assert cells[2] == f"{BUNDLES[name]['total_auroc']:.2f}\\*", name
