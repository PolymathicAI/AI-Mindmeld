"""The CPU training smoke on released data (``examples/configs/train-from-release``, examples/README.md step 9).

Both configs load and validate against the pins the release installs (the artifact ids below), and the smoke run is
the paper run (``configs/runs/gemma-3-4b-it/esmc-600m.yaml``) in everything but the
declared overrides: its connector, receiver, encoder, inputs, mixture and optimizer are the paper's.

The two Gemma 3 ChemFM-3B runs and their four step-2000 suites (``configs/runs/gemma-3-{4b,12b}-it/chemfm-3b.yaml``,
``configs/suites/gemma-3-{4b,12b}-it/chemfm-3b-*``) load on the pins their release bundles install, and only on those
(config resolution only; nothing is trained or scored). So do the three ESM-C 6B connectors' runs (Kimi K2.6 and
Gemma 3 12B affine, Kimi K2.6's four-token mlp) and their six step-2000 suites, and the run of Kimi K2.6's ChemFM-3B
four-token mlp connector. Each run starts from the released ridge and writes to the released connector's destination;
each suite scores the released connector on the released eval embeddings and item set.
"""

import hashlib
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from latent_interconnect.config import Pin, load_config
from latent_interconnect.evaluation.suites import SuiteRunConfig
from latent_interconnect.provenance import DeviceClass
from latent_interconnect.training import RunConfig
from latent_interconnect.training.config import AchievedSpec

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "configs" / "train-from-release"
PAPER = ROOT / "configs" / "runs" / "gemma-3-4b-it" / "esmc-600m.yaml"
RELEASED = {
    "protein-domain": "1c3c1d8cf604930f72a309217cd17d4728fdc011ed9e15ec874a1abf99569e5a",
    "protein-training-set/training_set": "6314cf2ca50c589a229303a5991ec6e83b5fb76e871276857088d48543f89e77",
    "embeddings/esmc-600m/protein-train/embeddings": "864d2f0dd51842c618220c7c67bfeffabf85a098d197fb5d754d5c772fabc8d8",
    "embeddings/esmc-600m/protein-monitor/embeddings": (
        "f710e9afe9af241eb0f6f0b468f8f273c16b533e8e2a745cc183460cd6cb2c4d"
    ),
    "embeddings/esmc-600m/protein-parents/embeddings": (
        "ff95d5e58583951abbc3745652bdc46b007603f85a7b3f1a4c29b17ce9f2293f"
    ),
    "ridges/gemma-3-4b-it/esmc-600m/ridge": "1b6128fc9d4880fb7f8badd53101a7acc07458f312daf746f81d5e8105d4d43b",
    "release/protein/esmc-600m/domain-mini": "d1d2a566a8d394d572b003a3bdbd6005ca8bbbd0b1bc8e3ca5fa6c489108696d",
    "release/protein/esmc-600m/embeddings-mini": "8593324b7249b5dd858a013225a1edc94c9d86aa21949a55f36fd9fe96d4620d",
    "release/protein/esmc-600m/items-mini": "79fc0dd86d298886a55be63b338c9f385c5b5400206242a74d78fe52238712b2",
}
"""The released training inputs the smoke reads (``sgolkar/ai-mindmeld-data`` at ``a27d7808``, by id) and the minimal
eval set's pins; the ridge start is released sliced to its written layers. Both configs resolve on these same ids, so
the comparison below checks pin names and settings, not which records the paper run cited."""
CHECKPOINT = "train-from-release/gemma-3-4b-it/esmc-600m/cpu-smoke/step_000003"


class _Pins(Mapping[str, Pin]):
    """The released pins; any other name (the paper run's monitor suite inputs, the smoke's checkpoint) a stand-in."""

    def __getitem__(self, name: str) -> Pin:
        sha = RELEASED.get(name, hashlib.sha256(name.encode()).hexdigest())
        return Pin(target="record" if name.endswith("@record") else "artifact", sha256=sha)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str)

    def __iter__(self) -> Iterator[str]:
        return iter(RELEASED)

    def __len__(self) -> int:
        return len(RELEASED)


def test_smoke_run_is_the_paper_run_on_released_inputs():
    smoke = load_config(EXAMPLE / "gemma-3-4b-it-esmc-600m-cpu-smoke.yaml", RunConfig, pins=_Pins())
    paper = load_config(PAPER, RunConfig, pins=_Pins())
    overrides = {"destination", "name", "steps", "batch", "micro_batch", "device_class", "checkpoints", "eval_suite"}
    overrides.add("eval_every")
    assert overrides <= set(RunConfig.model_fields)
    for field in set(RunConfig.model_fields) - overrides:
        assert getattr(smoke, field) == getattr(paper, field), field
    assert set(smoke.embeddings) | {smoke.training_set, smoke.domain.data, *smoke.ridge.values()} <= set(
        RELEASED.values()
    )
    assert (smoke.steps, smoke.batch, smoke.micro_batch, smoke.device_class) == (3, 2, None, DeviceClass.CPU)
    assert smoke.eval_suite is None and paper.device_class is DeviceClass.A100


def test_smoke_suite_loads():
    run = load_config(EXAMPLE / "protein-access-mini-cpu-smoke.yaml", SuiteRunConfig, pins=_Pins())
    assert run.checkpoints == {"m": _Pins()[CHECKPOINT].sha256}
    assert run.inputs == {"m": (RELEASED["release/protein/esmc-600m/embeddings-mini"],)}
    assert run.device_class is DeviceClass.CPU and run.step == 3


CHEMFM3B_RIDGE = "ridges/{}/chemfm-3b"
"""The pin of a ChemFM-3B run's ridge start, by receiver."""
CHEMFM3B = {
    "molecule-domain": "d680ca774ffdbb7c8604d37366d077023df33b53a1751efd81d99ea83861f3a5",
    "molecule-training-set/training_set": ("feabbac061169488f31d19cf0f4a7a08a5bb7290709f91641022aa76057690e8"),
    "items/molecule-training-monitor/item_set": ("fabad10e5ac1e5c167998f63f6929448839bfe88ca26f84ba5b8d7ddf46878d1"),
    "items/molecule-eval-novel-scaffold": "419053360b60f9c2634d67b18dcb35cc4cd8ba992906ffa8fdc48971dd1d1f02",
    "embeddings/chemfm-3b/molecule-train": ("5d3c841af51f3ac0b3bcf6a3c9f1620a05a6f72bb471f6daefe5220600bfa647"),
    "embeddings/chemfm-3b/molecule-monitor": ("f4a8f91ae42b59599c602b0f09d4acf67197c4c8b7ff7178a1c84b2f90c81033"),
    "embeddings/chemfm-3b/molecule-eval-novel-scaffold": (
        "71c0f7e4aab147155f4a125f2453a8f069dad28d5ef839c635a138826d63e82f"
    ),
    CHEMFM3B_RIDGE.format("gemma-3-4b-it"): "13d77a65357ef9fa52f994de871a7f030ef3b18e481ace4c7914b850dc0d69e1",
    CHEMFM3B_RIDGE.format("gemma-3-12b-it"): "e908228debd3a5288254345a5079117b49d11d84340093fdf8dba037678f1458",
    "connectors/gemma-3-4b-it/chemfm-3b/step_002000": (
        "ca65d8c31fb392bc33ac365520fdb31ac6f7649489c5551791be35dbdfe0eb60"
    ),
    "connectors/gemma-3-12b-it/chemfm-3b/step_002000": (
        "b499ba25c1564529c4015a01d8a1ddc24ab801c5a5a26b89a3a15764f08fad14"
    ),
}
"""The pins the ChemFM-3B bundles install (data ``a27d7808``, connectors ``72a79954``): every pin the two Gemma 3
ChemFM-3B run configs and their four step-2000 suite configs cite, and nothing else."""


def _released(names: Mapping[str, str]) -> dict[str, Pin]:
    return {n: Pin(target="artifact", sha256=s) for n, s in names.items()}


@pytest.mark.parametrize("receiver", ["gemma-3-4b-it", "gemma-3-12b-it"])
def test_chemfm3b_run_resolves_on_released_pins(receiver):
    """The Gemma 3 ChemFM-3B run loads on the released pins alone: it starts from the released ridge and writes the
    released connector."""
    path = ROOT / "configs" / "runs" / receiver / "chemfm-3b.yaml"
    run = load_config(path, RunConfig, pins=_released(CHEMFM3B))
    ridge = CHEMFM3B_RIDGE.format(receiver)
    assert run.ridge == {"name_described": CHEMFM3B[ridge]}
    assert run.destination == f"connectors/{receiver}/chemfm-3b" and f"{run.destination}/step_002000" in CHEMFM3B


@pytest.mark.parametrize("receiver", ["gemma-3-4b-it", "gemma-3-12b-it"])
@pytest.mark.parametrize("task", ["access", "transfer"])
def test_chemfm3b_suite_resolves_on_released_pins(receiver, task):
    """Each step-2000 ChemFM-3B suite loads on the released pins, scoring the released connector on the released eval
    embeddings and item set."""
    name = f"chemfm-3b-molecule-{task}-novel-scaffold-trained.yaml"
    run = load_config(ROOT / "configs" / "suites" / receiver / name, SuiteRunConfig, pins=_released(CHEMFM3B))
    assert run.checkpoints == {"m": CHEMFM3B[f"connectors/{receiver}/chemfm-3b/step_002000"]}
    assert run.inputs == {"m": (CHEMFM3B["embeddings/chemfm-3b/molecule-eval-novel-scaffold"],)}
    assert run.suite.item_set == CHEMFM3B["items/molecule-eval-novel-scaffold"] and run.step == 2000


ESMC6B = {
    "protein-domain": "1c3c1d8cf604930f72a309217cd17d4728fdc011ed9e15ec874a1abf99569e5a",
    "protein-training-set/training_set": "6314cf2ca50c589a229303a5991ec6e83b5fb76e871276857088d48543f89e77",
    "items/protein-training-monitor/item_set": "2d562135ab01c4ab541bb18b383ae3ec01b7e30c4b4dcbfadd929353ffab3a7d",
    "items/protein-eval": "fedf984ea427b3f6c632953dcc76015c3b857dc12634bcf415724145abddf74b",
    "embeddings/esmc-6b/protein-train": "edcbe5ec9ce25286c3b3c1449ebba0ba1c488e9aba64e1027b710a5be6dc5a6f",
    "embeddings/esmc-6b/protein-monitor": "866752843f70366a93bd312a100cea66c3be042d09cd233ccaa568f66f118877",
    "embeddings/esmc-6b/protein-parents": "1f0f3e8ff417da02fd4a4d90e73f8432c93021e180d1c03bd0296715089ed233",
    "embeddings/esmc-6b/protein-eval": "d30900df57593beec66b9fbe703b3fcac116ab151e9180b6a481f64d88b6b614",
    "ridges/kimi-k2.6/esmc-6b": "5669f201a5ad5054a81328907db6432b6fcb56415266070f1f4052c5c4481b00",
    "ridges/gemma-3-12b-it/esmc-6b": ("23707a8e90e9162c39053e310bf9413870a8c74a115fa6a64ddb4c096136e45b"),
    "connectors/kimi-k2.6/esmc-6b/step_002000": "43906432dfd67abf9096c538131845c2368e60ccf61fd3cc5daf6191247ab3af",
    "connectors/gemma-3-12b-it/esmc-6b/step_002000": "8f94b9dbc34bdfbee07d814038dc82ece8d4d8082965c9ac1e9241767503d454",
    "connectors/kimi-k2.6/esmc-6b-mlp4-staggered/step_002000": (
        "5d4655c43fbe70ef2cd9050fec802d9dd0d50a4f309f2bd238e67459909c977a"
    ),
}
"""The pins the ESM-C 6B connectors' release bundles install (data ``a27d7808``, connectors ``72a79954``): every
pin their three run configs and six step-2000 suite configs cite. The ESM-C 6B embeddings and the two ridge starts
are slices (new ids, same pins); Kimi K2.6's ridge pin serves both the affine and the mlp connector."""
ESMC6B_RUNS = {
    "kimi-k2.6__esmc-6b": "runs/kimi-k2.6/esmc-6b.yaml",
    "gemma-3-12b-it__esmc-6b": "runs/gemma-3-12b-it/esmc-6b.yaml",
    "kimi-k2.6__esmc-6b-mlp4-staggered": "runs/kimi-k2.6/esmc-6b-mlp4-staggered-achieved.yaml",
}
"""Each ESM-C 6B connector's run config. The mlp connector's is the ``achieved`` publication of its stopped run
(``configs/runs/kimi-k2.6/esmc-6b-mlp4-staggered.yaml``)."""
ACHIEVED = {"runs/kimi-k2.6/esmc-6b-mlp4-staggered-achieved.yaml"}
"""The run configs that are ``achieved`` publications of a stopped run (``AchievedSpec``)."""
ESMC6B_SUITES = {
    "suites/kimi-k2.6/esmc-6b-protein-access-trained.yaml": "connectors/kimi-k2.6/esmc-6b/step_002000",
    "suites/kimi-k2.6/esmc-6b-protein-transfer-trained.yaml": "connectors/kimi-k2.6/esmc-6b/step_002000",
    "suites/gemma-3-12b-it/esmc-6b-protein-access-trained.yaml": "connectors/gemma-3-12b-it/esmc-6b/step_002000",
    "suites/gemma-3-12b-it/esmc-6b-protein-transfer-trained.yaml": "connectors/gemma-3-12b-it/esmc-6b/step_002000",
    "suites/kimi-k2.6/esmc-6b-mlp4-staggered-protein-access-trained.yaml": (
        "connectors/kimi-k2.6/esmc-6b-mlp4-staggered/step_002000"
    ),
    "suites/kimi-k2.6/esmc-6b-mlp4-staggered-protein-transfer-trained.yaml": (
        "connectors/kimi-k2.6/esmc-6b-mlp4-staggered/step_002000"
    ),
}
"""Each step-2000 suite's checkpoint pin."""


@pytest.mark.parametrize("connector", sorted(ESMC6B_RUNS))
def test_esmc6b_run_resolves_on_released_pins(connector):
    """Each ESM-C 6B run loads on the released pins alone: its ridge start and embeddings are released pins, and it
    writes the released connector."""
    rel = ESMC6B_RUNS[connector]
    cls = AchievedSpec if rel in ACHIEVED else RunConfig
    top = load_config(ROOT / "configs" / rel, cls, pins=_released(ESMC6B))
    run = top.run if cls is AchievedSpec else top
    assert set(run.ridge.values()) <= set(ESMC6B.values())
    assert set(run.embeddings) <= set(ESMC6B.values())
    assert f"{top.destination}/step_002000" in ESMC6B


@pytest.mark.parametrize("path", sorted(ESMC6B_SUITES))
def test_esmc6b_suite_resolves_on_released_pins(path):
    """Each step-2000 suite loads on the released pins, scoring the released connector on the released eval
    embeddings and item set."""
    pin = ESMC6B_SUITES[path]
    run = load_config(ROOT / "configs" / path, SuiteRunConfig, pins=_released(ESMC6B))
    assert run.checkpoints == {"m": ESMC6B[pin]} and run.step == 2000
    assert run.inputs == {"m": (ESMC6B["embeddings/esmc-6b/protein-eval"],)}
    assert run.suite.item_set == ESMC6B["items/protein-eval"]


MOL_MLP = {
    "molecule-domain": "d680ca774ffdbb7c8604d37366d077023df33b53a1751efd81d99ea83861f3a5",
    "molecule-training-set/training_set": ("feabbac061169488f31d19cf0f4a7a08a5bb7290709f91641022aa76057690e8"),
    "items/molecule-training-monitor/item_set": ("fabad10e5ac1e5c167998f63f6929448839bfe88ca26f84ba5b8d7ddf46878d1"),
    "embeddings/chemfm-3b/molecule-train": ("5d3c841af51f3ac0b3bcf6a3c9f1620a05a6f72bb471f6daefe5220600bfa647"),
    "embeddings/chemfm-3b/molecule-monitor": ("f4a8f91ae42b59599c602b0f09d4acf67197c4c8b7ff7178a1c84b2f90c81033"),
    "ridges/kimi-k2.6/chemfm-3b": "17b5c94e7ed3757f43937a7e516be25e6f61f0116e90cfcf3f6803e7d6c42de1",
    "connectors/kimi-k2.6/chemfm-3b-mlp4/step_002000": (
        "7263d8669969e3219d925e8cac254d2264524c23b08b4e88911636361395a35d"
    ),
}
"""The pins Kimi K2.6's ChemFM-3B four-token mlp connector's bundles install (data ``a27d7808``, connectors
``72a79954``): every pin its run config cites. The ridge start is a slice to the written layers (new id, same pin)."""


def test_mol_mlp_run_resolves_on_released_pins():
    """Kimi K2.6's ChemFM-3B mlp run loads on the released pins alone: it starts from the released ridge and writes
    the released connector. The four tokens write the same six layers (not staggered)."""
    path = ROOT / "configs" / "runs" / "kimi-k2.6" / "chemfm-3b-mlp4.yaml"
    run = load_config(path, RunConfig, pins=_released(MOL_MLP))
    assert run.ridge == {"name_described": MOL_MLP["ridges/kimi-k2.6/chemfm-3b"]}
    assert run.connector.layers.per_token == ((7, 13, 22, 32, 49, 57),) * 4
    assert run.destination == "connectors/kimi-k2.6/chemfm-3b-mlp4" and f"{run.destination}/step_002000" in MOL_MLP
