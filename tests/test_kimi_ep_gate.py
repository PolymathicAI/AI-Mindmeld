"""``--require-kimi-ep`` (``tests/conftest.py``) passes only when every listed ``kimi_ep`` test passed.

Each case runs pytest in a subprocess on a copy of this suite's conftest, with stand-in ``kimi_ep`` modules whose tests
carry the listed node ids (``tests/kimi_ep_tests.txt``) and do nothing, so no optional setup is needed.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

import pytest

pytest_plugins = ("pytester",)

TESTS = Path(__file__).resolve().parent
LISTED = sorted((TESTS / "kimi_ep_tests.txt").read_text().split())


def _modules(change: str | None = None) -> dict[str, str]:
    """Stand-in module sources by path; ``change`` alters the first listed test: renamed, skipped, xfail or absent."""
    params: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for node in LISTED:
        path, name = node.split("::")
        func, _, param = name.partition("[")
        params[path][func].append(param.removesuffix("]"))
    first_path, first_func = LISTED[0].split("::")[0], LISTED[0].split("::")[1].partition("[")[0]
    out = {}
    for path, funcs in params.items():
        lines = ["import pytest", "", "pytestmark = pytest.mark.kimi_ep", ""]
        if change == "module-skipped" and path == first_path:
            lines.insert(1, 'pytest.importorskip("not_an_installed_module")')
        for func, values in funcs.items():
            first = path == first_path and func == first_func
            name = func + "_renamed" if first and change == "renamed" else func
            if first and change == "skipped":
                lines.append('@pytest.mark.skip(reason="absent setup")')
            if first and change == "xfail":
                lines.append("@pytest.mark.xfail(run=False)")
            if values != [""]:
                lines.append(f"@pytest.mark.parametrize('p', {values!r})")
            lines += [f"def {name}({'p' if values != [''] else ''}):", "    pass", ""]
        out[path] = "\n".join(lines)
    return out


@pytest.fixture
def suite(pytester: pytest.Pytester):
    for name in ("conftest.py", "snapshots.py", "kimi_ep_tests.txt"):
        (pytester.path / "tests").mkdir(exist_ok=True)
        shutil.copyfile(TESTS / name, pytester.path / "tests" / name)
    pytester.makepyprojecttoml('[tool.pytest.ini_options]\npythonpath = ["."]\naddopts = "--strict-markers"\n')

    def run(change: str | None = None, *args: str) -> pytest.RunResult:
        for path, source in _modules(change).items():
            (pytester.path / path).parent.mkdir(parents=True, exist_ok=True)
            (pytester.path / path).write_text(source)
        return pytester.runpytest_subprocess("-p", "no:cacheprovider", "-m", "kimi_ep", "--require-kimi-ep", *args)

    return run


def test_every_listed_test_passing_passes(suite):
    result = suite()
    result.assert_outcomes(passed=len(LISTED))
    assert result.ret == 0


@pytest.mark.parametrize("args", [("-k", "options"), ("--deselect", LISTED[0])])
def test_a_narrowed_selection_is_refused(suite, args):
    result = suite(None, *args)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--require-kimi-ep runs every kimi_ep test: drop -k and --deselect*"])


@pytest.mark.parametrize(
    ("change", "shown"),
    [
        ("renamed", ["*1 test(s) did not pass:*", "*1 test(s) is not in tests/kimi_ep_tests.txt:*"]),
        ("skipped", ["*--require-kimi-ep: skipped*", "*1 test(s) did not pass:*"]),
        ("xfail", ["*--require-kimi-ep: xfailed or xpassed*", "*1 test(s) did not pass:*"]),
        ("module-skipped", ["*skipped at collection under --require-kimi-ep*"]),
    ],
)
def test_a_listed_test_that_did_not_pass_fails_the_run(suite, change, shown):
    result = suite(change)
    assert result.ret != 0
    for line in shown:  # each on its own: the gate's lines print before the failure details
        result.stdout.fnmatch_lines([line])
