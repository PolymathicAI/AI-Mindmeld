# Contributing

## List your connector

Host your code, weights, card, and support in your own repository or model hub. The
[LIP library](library/README.md#community-connectors) keeps a description and links.

For implementation and training, see [Extending the package](examples/guides/extending.md).
For bundling, checking, and hosting your checkpoint, see
[Packaging and sharing a connector](examples/guides/contributing-a-connector.md).

1. Publish versioned downloads and a working inference example. Your card should describe supported
   models, training data, injection layout, tested engine version, and licenses.
   Use the [card template](library/CONNECTOR_CARD_TEMPLATE.md) if helpful.
2. Fork this repository and create a branch. Add an entry under **Community connectors** in
   `library/README.md`: name, one-sentence description, supported LLMs, and card/download links.
3. Push your branch and open a PR against `main`, titled `Library: <name>`. Link the inference example.

We check completeness and links. You maintain compatibility, hosting, and support; send another PR to
update the listing. Keep connector code and weights in your project.

## Improve AI Mindmeld

Fork, create a branch, and open a focused PR against `main`. Explain the problem, your change, and how
you checked it; include relevant test results for code changes. Open an issue before a larger change.

For code changes, run the [developer checks](#developer-checks) and the relevant tests.

## Developer checks

Run these checks from the checkout in the [Quickstart environment](examples/README.md#1-install)
(release lock and `.[dev,hf,chat]`). The temporary data root keeps tests separate from your downloads:

```bash
ruff check .
ruff format --check .
export AIMM_TEST_ROOT=~/aimm-tests
mkdir -p "$AIMM_TEST_ROOT"
LIP_DATA=$(mktemp -d "$AIMM_TEST_ROOT/run.XXXXXX") python -m pytest -n 8 --require-e2e tests
```

The optional gates below require their model-specific setup and treat skipped tests as failures.

### LLM snapshot tests (`--require-snapshots`)

Tests that read an LLM's real tokenizer (`tests/snapshots.py`) skip without its local snapshot, and
`--require-snapshots` turns those skips into failures. The gate needs `google/gemma-3-4b-it` at revision
`093f9f388b31de276ce2de164bdc2081324b9767` under `$HF_HOME/hub`: the first `ask.py` call of
[Quickstart step 4](examples/README.md#4-ask-a-question) provides it, or, after `hf auth login`, a download
without weights:

```bash
hf download google/gemma-3-4b-it --revision 093f9f388b31de276ce2de164bdc2081324b9767 \
  config.json tokenizer_config.json tokenizer.json
```

Then run the gate from the checkout:

```bash
python -m pytest -n 8 --require-snapshots tests
```

### Kimi K2.6 CPU tests (`kimi_ep`)

`tests/unit/test_kimi_ep.py` checks the `kimi-ep` training backend on a tiny random Kimi K2.6 model in float64 on
CPU: its loss, write gradients and one Adam update equal the prepared `hf` reference's (padded and packed-eager layouts,
one and two expert blocks, activation checkpointing on and off), its in-job scores equal `kimi-stream`'s, and the
INT4 expert path equals its dequantized twin. `tests/unit/test_kimi_finite_differences.py` checks that reference
against central finite differences, and that the reference without Kimi's training preparation (right loss, non-zero
but wrong gradients) is refused.

The tests build the tiny model from the pinned snapshot's code and config; no Kimi weights are downloaded.

From the same checkout and Quickstart environment, install the optional dependencies and download the model code:

```bash
uv pip install --no-deps compressed-tensors==0.12.2 loguru==0.7.3 tiktoken==0.14.0  # dependencies: in the lock
export AIMM_KIMI_HUB=~/aimm-kimi-hub   # a Hugging Face hub cache holding only these files
hf download moonshotai/Kimi-K2.6 --revision 7eb5002f6aadc958aed6a9177b7ed26bb94011bb --cache-dir "$AIMM_KIMI_HUB" \
  --include '*.py' config.json generation_config.json tokenizer_config.json tiktoken.model chat_template.jinja
```

The revision is the one pinned in `configs/receivers/kimi-k2.6.yaml`; the download is about 3 MB. Run:

```bash
python -m pytest -n 8 -m kimi_ep --require-kimi-ep tests
```

`--require-kimi-ep` passes only if every test listed in `tests/kimi_ep_tests.txt` (77) passed: it refuses `-k` and
`--deselect`, and fails the run if a `kimi_ep` test skips, xfails or xpasses, a module is skipped at collection, a
listed test did not pass, or an unlisted `kimi_ep` test reported.
