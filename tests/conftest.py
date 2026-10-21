"""Shared fixtures. CPU only by default: GPU tests carry the ``gpu`` marker.

Tests marked ``slice`` read the real original data (checks against a full data root, LIP_DATA), tests
marked ``gpu`` load real models, and tests marked ``long`` take tens of minutes (a recorded measurement). Each kind
runs only with its explicit opt-in flag, ``--slice``, ``--gpu`` or ``--long``, and a test of two kinds needs both
(``--slice --long``); without it the tests are skipped. So the default suite needs no data and never loads a real
model, even on a GPU workstation, and ``--slice`` stays quick. Which tests run is otherwise pytest's own selection
(paths, ``-k``, ``-m``).

Tests that read a receiver's local snapshot (tokenizer, config, remote code; ``tests/snapshots.py``) skip without it.
The summary always counts those skips, and ``--require-snapshots`` makes them failures, for every
receiver alike.

The ``e2e`` tests that write records skip on a dirty tree (``tests/e2e``). ``--require-e2e`` (the CPU gate, whose copy
of a commit is never dirty) is for the run that is meant to include them, so the gate never passes without the pipeline
run. With it these fail:

- an ``e2e`` test that skips (an xfail ran, so it stays an xfail);
- a module under ``tests/e2e`` skipped at collection (``importorskip``, ``allow_module_level``);
- a run in which no ``e2e`` test reported, because none is marked, selected or collected (exit status 4, with a
  line saying so; so the flag goes on no other run).

**Device policy.** Fixed once, in ``pytest_configure`` (before any test module is imported, so before anything can
query CUDA), by ``--gpu`` alone. A run without it hides every accelerator (``CUDA_VISIBLE_DEVICES=""``), so the CPU
suite gives one result on any host. A run with it keeps the host's devices untouched and never changes them afterwards,
whichever tests it selects: ``pytest --gpu tests/unit`` runs the CPU tests with the devices visible (how a GPU job
runs a CPU test beside its own).
In a ``--gpu`` run these fail, so a GPU job never passes having tested nothing:

- a ``gpu`` test that skips (a module-level ``skipif`` included; an xfail ran, so it stays an xfail);
- any module skipped at collection (``importorskip``, ``allow_module_level``).

Tests marked ``kimi_ep`` (the Kimi K2.6 CPU tests, ``CONTRIBUTING.md``) need an optional setup and skip without it.
``--require-kimi-ep`` (their acceptance run) passes only if every test of :data:`KIMI_EP_TESTS` passed. It refuses
``-k`` and ``--deselect``, and it fails a ``kimi_ep`` test that skips, xfails or xpasses, a ``kimi_ep`` module skipped
at collection, and a run in which a listed test did not pass or an unlisted ``kimi_ep`` test reported.

A run whose selected tests are all opt-in tests without their flag (``pytest -m gpu`` with no ``--gpu``) is refused
rather than passed with every test skipped, naming the flags those tests need; ``--collect-only`` lists them instead.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from huggingface_hub import constants
from tests.snapshots import REASON

REPO = Path(__file__).resolve().parents[1]


_OPT_IN = {
    "slice": "needs the real research data: run with --slice",
    "gpu": "a GPU job (it loads real models): run with --gpu, from its queue job",
    "long": "a long measurement (tens of minutes): run with --long (the slice measurement: --slice --long)",
}
"""Opt-in marker -> the reason its tests skip without their flag (``--<marker>``)."""
KIMI_EP_TESTS = frozenset((REPO / "tests" / "kimi_ep_tests.txt").read_text().split())
"""The node ids of the ``kimi_ep`` tests (``tests/kimi_ep_tests.txt``): under ``--require-kimi-ep`` each must pass."""
KIMI_EP_MODULES = frozenset(node.split("::")[0] for node in KIMI_EP_TESTS)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-snapshots",
        action="store_true",
        help="fail, not skip, every test that needs a receiver's local snapshot",
    )
    parser.addoption(
        "--require-e2e",
        action="store_true",
        help="fail a skipped e2e test, and a run that selects none (the CPU gate: its copy is never a dirty tree)",
    )
    parser.addoption(
        "--require-kimi-ep",
        action="store_true",
        help="fail a skipped kimi_ep test, and a run in which a kimi_ep module reported none (CONTRIBUTING.md)",
    )
    parser.addoption("--gpu", action="store_true", help="run the gpu tests and keep the host's devices (a GPU job)")
    parser.addoption("--slice", action="store_true", help="run the slice tests (they read the real research data)")
    parser.addoption("--long", action="store_true", help="run the long measurements (tens of minutes each)")


def _opted_in(config: pytest.Config, marker: str) -> bool:
    return bool(config.getoption(f"--{marker}"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "kimi_ep: a Kimi K2.6 CPU test; needs the optional setup of CONTRIBUTING.md")
    if config.getoption("--require-kimi-ep") and (config.option.keyword or config.option.deselect):
        raise pytest.UsageError("--require-kimi-ep runs every kimi_ep test: drop -k and --deselect")
    if not _opted_in(config, "gpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # CUDA reads the variable once, at its first query (is_available included), and keeps the answer: this query
        # fixes it hidden, and sees a device only if something queried CUDA before this hook could hide it
        if torch.cuda.is_available():
            raise pytest.UsageError("CUDA was queried before the device policy could hide the devices")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[None]:
    outcome = yield
    report = outcome.get_result()
    if report.skipped and item.config.getoption("--require-snapshots") and REASON in str(report.longrepr):
        report.outcome = "failed"
        report.longrepr = f"--require-snapshots: skipped ({report.longrepr})"
    elif (
        report.skipped
        and not hasattr(report, "wasxfail")
        and item.config.getoption("--require-e2e")
        and item.get_closest_marker("e2e")
    ):
        report.outcome = "failed"
        report.longrepr = f"--require-e2e: skipped ({report.longrepr})"
    elif (
        (report.skipped or hasattr(report, "wasxfail"))
        and item.config.getoption("--require-kimi-ep")
        and item.get_closest_marker("kimi_ep")
    ):
        kind = "xfailed or xpassed" if hasattr(report, "wasxfail") else "skipped"
        report.outcome = "failed"
        report.longrepr = f"--require-kimi-ep: {kind} ({report.longrepr})"
    elif (
        report.skipped
        and not hasattr(report, "wasxfail")
        and _opted_in(item.config, "gpu")
        and item.get_closest_marker("gpu")
    ):
        report.outcome = "failed"
        report.longrepr = (
            f"a selected gpu test skipped ({report.longrepr}): a GPU job must not pass having tested nothing"
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector: pytest.Collector) -> Iterator[None]:
    outcome = yield
    report = outcome.get_result()
    if not (report.skipped and isinstance(collector, pytest.Module)):
        return
    # a module skipped at collection (importorskip, allow_module_level) has no test report to fail
    if _opted_in(collector.config, "gpu"):
        report.outcome = "failed"
        report.longrepr = f"{collector.path.name} skipped at collection in a gpu run ({report.longrepr})"
    elif collector.config.getoption("--require-e2e") and "e2e" in collector.path.parts:
        report.outcome = "failed"
        report.longrepr = f"{collector.path.name} skipped at collection under --require-e2e ({report.longrepr})"
    elif collector.config.getoption("--require-kimi-ep") and _relative(collector.path) in KIMI_EP_MODULES:
        report.outcome = "failed"
        report.longrepr = f"{collector.path.name} skipped at collection under --require-kimi-ep ({report.longrepr})"


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix() if path.resolve().is_relative_to(REPO) else str(path)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Refuse a ``--require-e2e`` run in which no ``e2e`` test reported (none marked, selected or collected), and a
    ``--require-kimi-ep`` run in which a test of :data:`KIMI_EP_TESTS` did not pass or an unlisted ``kimi_ep`` test
    reported.

    Checked on the reports, at the end, because under pytest-xdist only the controller can stop the run with a
    readable message and it sees no items, only the workers' reports.
    """
    config = session.config
    if config.option.collectonly or hasattr(config, "workerinput"):
        return
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    reports = [r for stat in reporter.stats.values() for r in stat if isinstance(r, pytest.TestReport)]
    if config.getoption("--require-e2e") and not any("e2e" in r.keywords for r in reports):
        reporter.ensure_newline()
        reporter.write_line("ERROR: --require-e2e: no e2e test ran, so the toy pipeline did not run", red=True)
        session.exitstatus = pytest.ExitCode.USAGE_ERROR
    if config.getoption("--require-kimi-ep"):
        passed = {r.nodeid for r in reports if r.when == "call" and r.passed and not hasattr(r, "wasxfail")}
        missing = sorted(KIMI_EP_TESTS - passed)
        unlisted = sorted({r.nodeid for r in reports if "kimi_ep" in r.keywords} - KIMI_EP_TESTS)
        for problem, nodes in (("did not pass", missing), ("is not in tests/kimi_ep_tests.txt", unlisted)):
            if nodes:
                reporter.ensure_newline()
                reporter.write_line(f"ERROR: --require-kimi-ep: {len(nodes)} test(s) {problem}:", red=True)
                for node in nodes:
                    reporter.write_line(f"  {node}", red=True)
                session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    skipped = [r for r in terminalreporter.stats.get("skipped", []) if REASON in str(r.longrepr)]
    terminalreporter.write_line(f"receiver snapshots: {len(skipped)} test(s) skipped ({REASON})")


@pytest.hookimpl(trylast=True)  # after pytest's own -m / -k deselection: these are the selected tests
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    off = [m for m in _OPT_IN if not _opted_in(config, m)]
    gated = [item for item in items if any(m in item.keywords for m in off)]
    if items and len(gated) == len(items) and not config.option.collectonly:
        needed = sorted({m for item in gated for m in off if m in item.keywords})
        raise pytest.UsageError(
            f"every selected test is an opt-in test without its flag ({', '.join('--' + m for m in needed)}):"
            " the run would skip them all"
        )
    for item in gated:
        marker = next(m for m in off if m in item.keywords)
        item.add_marker(pytest.mark.skip(reason=_OPT_IN[marker]))


@pytest.fixture
def repo() -> Path:
    return REPO


# ---- the receiver tests' tiny Kimi K2.6 (``tests/kimi_tiny.py``) and hf on it ----------------------------------------
# Here, not in tests/unit/conftest.py: pytest 9.1 binds a conftest's fixtures to the first Directory node of its folder,
# and an argument list that names a file of tests/ between two of tests/unit (``tests/unit/a.py tests/test_docs.py
# tests/unit/b.py``) collects tests/unit twice, so b.py did not see them ("fixture 'tiny' not found"). tests/ itself is
# collected once whatever the arguments. They need the pinned Kimi snapshot's code in the Hugging Face cache; without it
# the tests that use them skip (``--require-snapshots`` makes that a failure).


@pytest.fixture(scope="module")
def hub(tmp_path_factory):
    """A Hugging Face hub cache holding the fixtures; the backends resolve pinned snapshots in it."""
    # imported here, not at module load: tests.kimi_tiny reaches torch.compile, which queries CUDA, and this conftest
    # loads before pytest_configure hides the devices; and before the patch, which its REAL_HUB must not see
    import tests.kimi_tiny  # noqa: F401

    root = tmp_path_factory.mktemp("hub")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(constants, "HF_HUB_CACHE", str(root))
        yield root


@pytest.fixture(scope="module")
def tiny(hub):
    from tests.kimi_tiny import TINY_LAYERS, build_tiny_kimi

    return build_tiny_kimi(hub, "kimi-tiny", layers=TINY_LAYERS)


@pytest.fixture
def hf_on_tiny(tiny):
    """hf on its own copy of the tiny Kimi model, built and used under the float64 default dtype.

    The remote code's mask helper fills in the default dtype, and hf's construction probes one forward (its head
    check), so the default is float64 from before the build until the test ends; the shared ``tiny`` model is never
    converted or hooked.
    """
    from latent_interconnect.receivers.hf import HFReceiver  # at use: this conftest loads before the device policy

    spec, model = tiny
    hf_spec = spec.model_copy(update={"backends": {"score": "hf", "capture": "hf"}, "plugins": ()})
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    hf = None
    try:
        hf = HFReceiver(hf_spec, model=copy.deepcopy(model).to(torch.float64))
        yield hf
    finally:
        if hf is not None:
            hf.close()
        torch.set_default_dtype(default)
