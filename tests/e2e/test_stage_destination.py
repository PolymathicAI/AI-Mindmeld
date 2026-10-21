"""``--destination NAME`` on a stage run (``run._main.stage_main``, ``check_pin``), on a one-output toy stage.

``--destination`` overrides the config's destination for the run, pins the outputs and the record under ``NAME`` as
``--pin NAME`` would, and the record keeps the override in its provenance's ``execution``; neither flag pins nothing.
The tests that write records skip on a dirty tree, as records refuse one (``--require-e2e`` fails the skip).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from latent_interconnect.config import StageConfig
from latent_interconnect.paths import running_code
from latent_interconnect.run._main import PIN_REFUSED, PinDestinationError, StageResult, check_pin, stage_main
from latent_interconnect.workspace import Workspace

pytestmark = pytest.mark.e2e
records = pytest.mark.skipif(running_code()[1], reason="records refuse a dirty tree: commit, then run the e2e")


class _Toy(StageConfig):
    value: int = 1


SEEN: list[str | None] = []
"""The destination each toy run saw in its config."""


def _run(config: _Toy, ws: Workspace) -> StageResult:
    SEEN.append(config.destination)
    staged = ws.artifacts.staging(f"toy-{config.value}")
    (staged / "x.txt").write_text(str(config.value))
    return StageResult(outputs={"out": ws.artifacts.commit(staged, "toy", {})})


def _main(argv: list[str]) -> int:
    return stage_main(argv, "toy", _Toy, _run)


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    SEEN.clear()
    root = tmp_path / "data"
    monkeypatch.setenv("LIP_DATA", str(root))
    return root


def _config(tmp_path: Path, destination: str | None, value: int = 1) -> str:
    path = tmp_path / f"toy-{destination is not None}-{value}.yaml"
    path.write_text((f"destination: {destination}\n" if destination else "") + f"value: {value}\n")
    return str(path)


def _key(out: str) -> str:
    return re.search(r"toy: record ([0-9a-f]{64})", out).group(1)


@records
def test_the_override_names_the_outputs_and_is_recorded(data, tmp_path, capsys, monkeypatch):
    """The outputs are pinned under --destination, not the config's destination; the stage sees the override; the
    record notes both names in its provenance's execution, and the computation is the one the config makes without."""
    config = _config(tmp_path, "declared/toy")
    assert _main(["--config", config, "--destination", "quickstart/toy"]) == 0
    key = _key(capsys.readouterr().out)
    ws = Workspace.from_env()
    pins = ws.pins()
    assert set(pins) == {"quickstart/toy", "quickstart/toy/out", "quickstart/toy@record"}
    assert pins["quickstart/toy@record"].sha256 == key
    assert SEEN == ["quickstart/toy"]
    record = ws.records.canonical(key)
    assert record.provenance.execution == {"destination.declared": "declared/toy", "destination.used": "quickstart/toy"}
    assert "destination" not in record.config  # execution metadata stays out of the record's config
    monkeypatch.setenv("LIP_DATA", str(tmp_path / "other"))  # the same config without the flag: the same computation
    assert _main(["--config", config]) == 0
    assert _key(capsys.readouterr().out) == key


@records
def test_destination_without_pin_pins_even_when_the_config_declares_none(data, tmp_path, capsys):
    assert _main(["--config", _config(tmp_path, None), "--destination", "quickstart/toy"]) == 0
    key = _key(capsys.readouterr().out)
    ws = Workspace.from_env()
    assert ws.pins()["quickstart/toy@record"].sha256 == key
    assert ws.records.canonical(key).provenance.execution == {"destination.used": "quickstart/toy"}


@records
def test_a_matching_pin_and_destination_are_accepted(data, tmp_path):
    assert _main(["--config", _config(tmp_path, "declared/toy"), "--destination", "quickstart/toy",
                  "--pin", "quickstart/toy"]) == 0  # fmt: skip
    assert set(Workspace.from_env().pins()) == {"quickstart/toy", "quickstart/toy/out", "quickstart/toy@record"}


@pytest.mark.parametrize(
    ("declared", "pin", "destination"),
    [
        ("declared/toy", "declared/toy", "quickstart/toy"),  # --pin equal to the config's, not to --destination
        (None, "other/toy", "quickstart/toy"),
        ("declared/toy", None, "bad@name"),  # not a destination
    ],
)
def test_a_mismatching_pin_or_an_invalid_destination_is_refused_before_the_stage_runs(
    data, tmp_path, capsys, declared, pin, destination
):
    argv = ["--config", _config(tmp_path, declared), "--destination", destination]
    with pytest.raises(SystemExit) as exit_:
        _main(argv + (["--pin", pin] if pin else []))
    assert exit_.value.code == 2
    assert SEEN == []
    assert not (data / "lip" / "pins.yaml").exists()
    assert "--destination" in capsys.readouterr().err


def test_check_pin_returns_the_name_and_refuses_a_mismatch():
    """The renderer-facing check: --destination is the name; --pin alone keeps its rule; neither names nothing."""
    assert check_pin("d/a", None) is None
    assert check_pin(None, "x/y") == "x/y"
    assert check_pin("d/a", "d/a") == "d/a"
    assert check_pin("d/a", None, "q/b") == "q/b"
    assert check_pin(None, "q/b", "q/b") == "q/b"
    with pytest.raises(PinDestinationError, match="is not --destination"):
        check_pin("d/a", "d/a", "q/b")
    with pytest.raises(PinDestinationError, match="is not the config's destination"):
        check_pin("d/a", "q/b")


@records
def test_neither_flag_pins_nothing_and_records_no_destination(data, tmp_path, capsys):
    assert _main(["--config", _config(tmp_path, "declared/toy")]) == 0
    key = _key(capsys.readouterr().out)
    ws = Workspace.from_env()
    assert ws.pins() == {}
    assert SEEN == ["declared/toy"]
    assert ws.records.canonical(key).provenance.execution == {}


@records
def test_pin_alone_keeps_its_rule_and_records_no_destination(data, tmp_path, capsys):
    assert _main(["--config", _config(tmp_path, "declared/toy"), "--pin", "declared/toy"]) == 0
    key = _key(capsys.readouterr().out)
    ws = Workspace.from_env()
    assert ws.pins()["declared/toy@record"].sha256 == key
    assert ws.records.canonical(key).provenance.execution == {}


@records
def test_repin_with_destination(data, tmp_path, capsys):
    """A second computation under a taken --destination is refused (exit 3), and pinned with --repin alone."""
    assert _main(["--config", _config(tmp_path, None, value=1), "--destination", "quickstart/toy"]) == 0
    first = _key(capsys.readouterr().out)
    second = _config(tmp_path, None, value=2)
    assert _main(["--config", second, "--destination", "quickstart/toy"]) == PIN_REFUSED
    assert Workspace.from_env().pins()["quickstart/toy@record"].sha256 == first
    assert _main(["--config", second, "--destination", "quickstart/toy", "--repin", "the second toy"]) == 0
    assert Workspace.from_env().pins()["quickstart/toy@record"].sha256 == _key(capsys.readouterr().out) != first


def test_repin_without_pin_or_destination_is_refused(data, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _main(["--config", _config(tmp_path, None), "--repin", "why"])
    assert "--pin or --destination" in capsys.readouterr().err
    assert SEEN == []
