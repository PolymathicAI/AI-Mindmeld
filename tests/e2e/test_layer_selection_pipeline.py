"""The worked pipeline's layer-selection diagnostics on CPU, through the ``run`` entry points, on tiny stand-ins.

The templates in ``examples/configs/pipeline/`` run as shipped, in a copy of that directory in which two files are
replaced by stand-ins under their own names: ``qwen2.5-0.5b-instruct.yaml`` by a random 24-layer tiny Llama (the
receiver's depth, so every layer list runs unchanged; ``tests.fakes.build_tiny_llama``) and ``esmc-300m-layers.yaml``
by the toy extractor's recipe (``tests.fakes.ToyExtractor``) at the same read layers. The answer words become ``Y`` and
``N``, one token each in the tiny tokenizer. The domain is the example data (``examples/data``, as
``examples/make_domain.py`` fills it).

The chain: ``domain`` → ``items`` (``items-train.yaml``, ``items-scan.yaml``) → ``extract`` (``extract-layers.yaml``)
→ ``capture`` (``capture.yaml``) → ``ridge`` (``ridge-heldout.yaml``) → ``read_scan`` (``read-scan.yaml``) → ``scan``
(``scan-reference.yaml``) → ``study`` (``study.yaml``) → ``analysis`` (``analysis-layer-selection.yaml``). The
analysis must accept the records, and refuse an own-state reference scanned on another receiver (another setup's).

- :func:`test_diagnostic_templates_validate` loads every diagnostic template against its stage's schema, stand-in pins.
- :func:`test_diagnostic_chain_runs` runs the chain. Records refuse a dirty tree, so it is skipped with uncommitted
  changes; ``--require-e2e`` makes that skip a failure.
"""

import importlib
import shutil
from pathlib import Path

import pytest
import yaml
from examples.make_domain import write_sources
from tests.fakes import build_tiny_llama

from latent_interconnect.config import dump_config, load_config
from latent_interconnect.config_index import cited_pins, stand_in_pins
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.evaluation.analysis import load_result
from latent_interconnect.evaluation.layer_selection_analysis import LayerSelectionResult
from latent_interconnect.paths import running_code
from latent_interconnect.workspace import Workspace

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "examples" / "configs" / "pipeline"
DATA = ROOT / "examples" / "data"
DIAGNOSTICS = [
    ("items", "items-train.yaml", "mine/items-train"),
    ("items", "items-scan.yaml", "mine/items-scan"),
    ("extract", "extract-layers.yaml", "mine/esmc-300m-layers/embeddings"),
    ("capture", "capture.yaml", "mine/qwen/capture"),
    ("ridge", "ridge-heldout.yaml", "mine/qwen/ridge-heldout"),
    ("read_scan", "read-scan.yaml", "mine/esmc-300m-layers/read-scan"),
    ("scan", "scan-reference.yaml", "mine/qwen/scan-reference"),
    ("study", "study.yaml", "mine/qwen/study"),
    ("analysis", "analysis-layer-selection.yaml", "mine/qwen/layer-selection"),
]
"""(stage, template, pin) in run order, after the domain stage."""
SCAN_TEMPLATES = ("scan-reference.yaml", "study.yaml")
"""The receiver scans, whose answer words the tiny tokenizer needs as one token each."""


def _stage(name: str):
    return importlib.import_module(f"latent_interconnect.run.{name}")


@pytest.mark.parametrize(("stage", "template", "pin"), DIAGNOSTICS)
def test_diagnostic_templates_validate(stage, template, pin):
    config = load_config(
        PIPELINE / template, _stage(stage).STAGE.config, stand_in_pins(cited_pins(PIPELINE / template))
    )
    assert config.destination == pin  # the pin each stage command names (stage_main refuses another)


def _stand_ins(run: Path, receiver_dir: Path) -> None:
    """Copy the templates into ``run`` and replace the receiver and encoder files by tiny stand-ins (module doc)."""
    for template in PIPELINE.glob("*.yaml"):
        shutil.copy(template, run / template.name)
    dump_config(build_tiny_llama(receiver_dir, depth=24), run / "qwen2.5-0.5b-instruct.yaml")
    recipe = load_config(PIPELINE / "esmc-300m-layers.yaml", EncoderRecipe)
    toy = recipe.model_copy(
        update={"extractor": "fake-toy", "model_id": "none", "revision": "v1", "environment": "lip", "options": {}}
    )
    dump_config(toy, run / "esmc-300m-layers.yaml")
    for name in SCAN_TEMPLATES:
        text = (run / name).read_text()
        assert text.count('{"yes": "Yes", "no": "No"}') == 1
        (run / name).write_text(text.replace('{"yes": "Yes", "no": "No"}', '{"yes": "Y", "no": "N"}'))


def _main(stage: str, config: Path, pin: str) -> None:
    assert _stage(stage).main(["--config", str(config), "--pin", pin]) == 0, config.name


@pytest.mark.skipif(running_code()[1], reason="records refuse a dirty tree: commit, then run the e2e")
def test_diagnostic_chain_runs(tmp_path, monkeypatch):
    lip = tmp_path / "data"
    monkeypatch.setenv("LIP_DATA", str(lip))
    run = tmp_path / "run"
    run.mkdir()
    _stand_ins(run, tmp_path / "tiny-llama-24")
    domain = yaml.safe_load((run / "domain.yaml").read_text())  # filled as examples/make_domain.py fills it
    domain["spec"].update(write_sources(DATA / "proteins.fasta", DATA / "proteins.tsv", lip, "mine/proteins"))
    (run / "domain-filled.yaml").write_text(yaml.safe_dump(domain, sort_keys=False))
    _main("domain", run / "domain-filled.yaml", "mine/protein-domain")
    for stage, template, pin in DIAGNOSTICS:
        _main(stage, run / template, pin)

    ws = Workspace.from_env()
    pins = ws.pins()
    record = ws.records.canonical(pins["mine/qwen/layer-selection@record"].sha256)
    result = load_result(ws.artifacts, record, LayerSelectionResult)
    tasks = {"DNA-binding", "Membrane", "Transferase", "Mammalia"}
    layers = {0, 4, 8, 12, 16, 20}
    for scan in (result.oracle, result.injection):  # every layer and task of both receiver scans, with intervals
        assert set(scan) == layers and all(set(by_task) == tasks for by_task in scan.values())
        assert all(e.lo <= e.estimate <= e.hi for by_task in scan.values() for e in by_task.values())
    reads = result.read["objects"]
    assert set(reads) == tasks and all(
        set(by_read) == {str(L) for L in (0, 6, 12, 18, 24, 30)} for by_read in reads.values()
    )
    assert result.sets is None  # no layer-set comparison in this chain
    ridge = ws.artifacts.manifest(ws.artifacts.ref(pins["mine/qwen/ridge-heldout"].sha256)).meta
    assert ridge["excluded"] == 12  # the scan items are left out of the held-out scan ridge's fit

    # another setup's own-state reference (another receiver, the same capture): the analysis refuses it
    dump_config(build_tiny_llama(tmp_path / "tiny-llama-24-b", seed=1, depth=24), run / "other-receiver.yaml")
    reference = (run / "scan-reference.yaml").read_text()
    (run / "scan-reference-other.yaml").write_text(
        reference.replace("{$ref: qwen2.5-0.5b-instruct.yaml}", "{$ref: other-receiver.yaml}").replace(
            "destination: mine/qwen/scan-reference", "destination: mine/other/scan-reference"
        )
    )
    _main("scan", run / "scan-reference-other.yaml", "mine/other/scan-reference")
    analysis = (run / "analysis-layer-selection.yaml").read_text()
    (run / "analysis-mixed.yaml").write_text(
        analysis.replace("mine/qwen/scan-reference@record", "mine/other/scan-reference@record").replace(
            "mine/qwen/layer-selection", "mine/mixed/layer-selection"
        )
    )
    with pytest.raises(ValueError, match=r"not this study's spec: it differs in \['receiver'\]"):
        _stage("analysis").main(["--config", str(run / "analysis-mixed.yaml"), "--pin", "mine/mixed/layer-selection"])
    assert "mine/mixed/layer-selection@record" not in ws.pins()
