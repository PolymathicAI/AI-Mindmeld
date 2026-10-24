# Quickstart

Install AI Mindmeld, download two connectors, and use Gemma 3 4B to answer questions and run a small
evaluation on CPU. After steps 1–5, optional steps 6–9 add your own protein sequences, two DEMO non-linear
connectors and a short training run. To build a connector of your own, see
[Build your own connector](#build-your-own-connector). Browse [Examples and guides](docs/README.md)
for chat, demos, and development workflows.

## Before you start

You need Git and a Linux shell. Download the [connectors](https://huggingface.co/sgolkar/ai-mindmeld-connectors)
and [data](https://huggingface.co/datasets/sgolkar/ai-mindmeld-data) as shown below. To load Gemma 3,
accept [its model terms](https://huggingface.co/google/gemma-3-4b-it) and authenticate with Hugging Face.

| Path through this guide | Disk space | Observed peak RAM |
|---|---:|---:|
| Install, infer, evaluate (steps 1–5) | About 26 GB | About 12 GB |
| Add ESM-C and MIMIC (steps 6–7) | Another 22 GB | About 14 GB |
| Try two DEMO non-linear connectors (step 8) | Another 0.15 GB | About 12 GB |
| Train for three steps (step 9) | Another 7 GB | About 21 GB |

Disk estimates include environments and caches; sharing files through uv can reduce them. Downloads
are kept in the Hugging Face cache and copied to the data directory. Choose paths with enough space:

```bash
export AIMM=~/ai-mindmeld
export LIP_DATA=~/aimm-data
export AIMM_WORK=~/aimm-work
export HF_HOME=~/.cache/huggingface
export UV_CACHE_DIR=~/.cache/uv
export UV_PYTHON_INSTALL_DIR=~/.local/share/uv/python
export CUDA_VISIBLE_DEVICES=""
```

Keep the same shell throughout. After reopening it, set these variables again, activate
`$AIMM/.venv/bin/activate`, and return to `$AIMM_WORK` for steps 4 onward.
The empty `CUDA_VISIBLE_DEVICES` keeps every step here on CPU; to serve chat on a GPU, see
[the chat guide](guides/chat.md#gemma-setup-and-launch).

## 1. Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed. This command leaves shell
profiles unchanged; add its install directory to `PATH` in this shell and whenever you reopen it.
The default is `~/.local/bin`; an existing `UV_INSTALL_DIR` overrides it. uv installs Python 3.12 if needed.

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
export PATH="${UV_INSTALL_DIR:-$HOME/.local/bin}:$PATH"
```

Clone release v1.1.0 of the repository:

```bash
git clone --branch v1.1.0 https://github.com/PolymathicAI/AI-Mindmeld.git "$AIMM" && cd "$AIMM"
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match -r env/locks/release.txt
uv pip install --no-deps -e '.[dev,hf,chat]'
python -c "import latent_interconnect; print(latent_interconnect.__file__)"   # .../src/latent_interconnect/__init__.py
```

The import checks the installation; its path should end in `src/latent_interconnect/__init__.py`.
Continue through step 4 to ask your first question. The pinned environment includes PyTorch with CUDA
libraries, but the commands here use CPU.

## 2. Log in to Hugging Face

```bash
hf auth login                  # paste a token with read access
```

For an unattended run, use `export HF_TOKEN_PATH=/path/to/token` instead. The token needs read access
to Gemma. An authentication error usually means access or accepted model terms are missing.

## 3. Download connectors and evaluation data

The LLM that receives specialist representations is the receiving LLM, or receiver. Its configuration is
called a receiver config in the download report.

```bash
python -m latent_interconnect.release_download \
  --repo sgolkar/ai-mindmeld-connectors --revision f69ffac6715789f730ef0985ef54ae82c7051954 \
  --connector gemma-3-4b-it__esmc-600m --connector gemma-3-4b-it__chemfm-1b \
  --data protein-esmc-600m-mini --data molecule-chemfm-1b-mini \
  --dest $LIP_DATA
```

This downloads about 400 MB: two connectors and two eight-object evaluation sets. The report lists eight pins,
each connector's kind, input and write layers, checkpoint directory, plugins, license, and receiver config,
then confirms `verified 31 installed files (sha256)`. `ask.py` takes a connector by its name and finds its
checkpoint directory under `$LIP_DATA`.
A checksum failure identifies the damaged file and cache entry; follow its removal instruction before retrying.

Use `python -m latent_interconnect.release_download --list --kinds` to see bundle names, sizes, kinds, and DEMO
markers. Download more with repeated `--connector NAME` or `--data NAME` arguments. Give `ask.py` the receiver
config named in the report: either `configs/receivers/<receiver>.yaml` in the checkout or the bundle's
installed config in `$LIP_DATA/lip/receivers/`.

For Gemma 3 12B, accept [the 12B model's terms](https://huggingface.co/google/gemma-3-12b-it) separately and use
`configs/receivers/gemma-3-12b-it.yaml`. Allow about 29 GB RAM for CPU inference; model weights are about 24 GB. See the
[library](../library/README.md) for compatibility and the [data inventory](../release/data_full.md) for contents.

## 4. Ask a question

Keep generated files outside the checkout: evaluation and training require a clean, committed tree.

```bash
mkdir -p "$AIMM_WORK" && cd "$AIMM_WORK"
```

Extract one protein and one molecule embedding from the downloaded sets:

```bash
python - <<'EOF'
import numpy as np
from latent_interconnect.encoders.store import EmbeddingReader
from latent_interconnect.workspace import Workspace
ws = Workspace.from_env()
for pin, out in [("release/protein/esmc-600m/embeddings-mini", "protein.npy"),
                 ("release/molecule/chemfm-1b/embeddings-mini", "molecule.npy")]:
    r = EmbeddingReader(ws.artifacts, ws.artifacts.ref(ws.pins()[pin].sha256))
    oid = r.object_ids[0]
    np.save(out, np.concatenate([r.pooled([oid], layer)[0] for layer in r.layers]))
    print(oid)
EOF
```

This prints `A0A024FA41` and `CHEBI:101085`. Ask about each object:

```bash
python $AIMM/examples/ask.py $AIMM/configs/receivers/gemma-3-4b-it.yaml \
  gemma-3-4b-it__esmc-600m protein.npy \
  "What does the protein <SLOT:x> do? Answer in one sentence." --max-new-tokens 64
python $AIMM/examples/ask.py $AIMM/configs/receivers/gemma-3-4b-it.yaml \
  gemma-3-4b-it__chemfm-1b molecule.npy \
  "What kind of compound is <SLOT:x>? Answer in one sentence."
```

The first call downloads Gemma 3 4B (about 8.6 GB). With cached weights, each of these two `ask.py`
calls took 20 seconds to two minutes on an otherwise idle CPU workstation.

Generation is greedy and stops after at most 512 tokens. Add `--max-new-tokens 64` after the question
to shorten the output.

**Yes/no probabilities.** `--yes-no` reads the answer probabilities instead of generating, as the evaluation suites
score a yes/no question. Its argument is the LLM's answer convention: `configs/answers/gemma-3.yaml` for
Gemma 3 4B and 12B, `configs/answers/kimi-k2.6.yaml` for Kimi K2.6. Write the question as the suites word it, without
an answer instruction; the convention appends the instruction, starts the answer, and turns thinking off.
For Kimi K2.6 the LLM's `score` backend (`kimi-stream`) needs the full Kimi weights and a GPU, and its tokenizer
needs `tiktoken` ([Kimi K2.6 CPU tests](../CONTRIBUTING.md#kimi-k26-cpu-tests-kimi_ep)).


```bash
python $AIMM/examples/ask.py $AIMM/configs/receivers/gemma-3-4b-it.yaml \
  gemma-3-4b-it__esmc-600m protein.npy \
  "Here is an unknown protein, given to you as an embedding:<SLOT:x>. Is this protein located in a membrane?" \
  --yes-no $AIMM/configs/answers/gemma-3.yaml
```

The output gives P(Yes) and P(No) of the next token under the full softmax, each with its token string and id, then
the log-odds `log P(Yes) − log P(No)`. The log-odds is the score the suites record and their AUROCs rank. The last line
is the pair-normalized `P(Yes) / (P(Yes) + P(No))`, a different quantity from the full-softmax P(Yes):

```text
P(Yes) = 0.90054  (full softmax; token ' Yes', id 8438)
P(No) = 0.0980971  (full softmax; token ' No', id 2301)
log P(Yes) - log P(No) = 2.21704  (the suites' score)
pair-normalized P(Yes) / (P(Yes) + P(No)) = 0.901769
```

Scores shift slightly with batching and hardware (about 0.05–0.07 in log-odds); keep both the same when comparing.

**ChemFM-3B from a stored vector.** `gemma-3-4b-it__chemfm-3b` (0.38 GB) has no mini set, but the data bundle
`embeddings-chemfm-3b` (0.89 GB) holds its molecules' vectors, so these examples do not need to load
the ChemFM-3B encoder:

```bash
python -m latent_interconnect.release_download --connector gemma-3-4b-it__chemfm-3b --data embeddings-chemfm-3b \
  --dest $LIP_DATA
python - <<'EOF'
import numpy as np
from latent_interconnect.encoders.store import EmbeddingReader
from latent_interconnect.workspace import Workspace
ws = Workspace.from_env()
r = EmbeddingReader(ws.artifacts, ws.artifacts.ref(ws.pins()["embeddings/chemfm-3b/molecule-eval-novel-scaffold"].sha256))
oid = r.object_ids[0]
np.save("molecule-3b.npy", np.concatenate([r.pooled([oid], layer)[0] for layer in r.layers]))
print(oid)
EOF
python $AIMM/examples/ask.py $LIP_DATA/lip/receivers/gemma-3-4b-it.yaml \
  gemma-3-4b-it__chemfm-3b molecule-3b.npy \
  "What kind of compound is <SLOT:x>? Answer in one sentence."
```

The selected molecule is `CHEBI:101085` (diclofenamide).
These examples read stored embeddings; encoding a new SMILES requires ChemFM-3B itself and its
matching extraction recipe (`configs/encoders/chemfm-3b.yaml`).

To chat with an LLM in a browser, see [Interactive chat (web GUI)](guides/chat.md).

## 5. Run a small evaluation

Score yes/no questions about protein properties and molecule classes. Each command runs the suite with the config
from its YAML file and saves its output to the given destination:

```bash
export CUDA_VISIBLE_DEVICES=""
python -m latent_interconnect.run.suite \
  --config $AIMM/configs/release/suites/gemma-3-4b-it-esmc-600m-protein-access-mini-cpu.yaml \
  --destination quickstart/protein-access-mini
python -m latent_interconnect.run.suite \
  --config $AIMM/configs/release/suites/gemma-3-4b-it-chemfm-1b-molecule-access-novel-scaffold-mini-cpu.yaml \
  --destination quickstart/molecule-access-novel-scaffold-mini
```

Each of these mini suites needs about 12 GB RAM and takes roughly 20–70 seconds on an otherwise idle CPU
workstation. These eight-object sets demonstrate the evaluation workflow; for benchmark comparisons, use the full
evaluation data and the paper's task definitions.

Save this small summary script, which evaluates the scores from those outputs; step 9 reuses it for the trained
checkpoint:

```bash
cat > summarize_scores.py <<'EOF'
import sys
import numpy as np, pandas as pd
from latent_interconnect.stats.auroc import auroc
from latent_interconnect.workspace import Workspace
ws = Workspace.from_env()
for name in sys.argv[1:]:
    t = pd.read_parquet(ws.artifacts.path(ws.artifacts.ref(ws.pins()[name].sha256)) / "scores.parquet")
    y, s = t["label"].to_numpy(), t["score"].to_numpy()
    labelled = (y == 0) | (y == 1)
    auc = auroc(s[labelled], y[labelled] == 1)
    print(f"{name}: {len(t)} questions, accuracy {np.mean((s > 0) == (y == 1)):.3f}, AUROC {auc:.3f}")
EOF
python summarize_scores.py \
  quickstart/protein-access-mini/scores \
  quickstart/molecule-access-novel-scaffold-mini/scores
```

## 6. Embed your own proteins (ESM-C 600M)

Keep the release environment active. The following creates a separate ESM-C environment and embeds
human ubiquitin. Replace `my.fasta` with your own sequences, one record per protein using the 20 standard
amino-acid letters.

```bash
cd "$AIMM_WORK"
uv venv --python 3.12 .venv-esmc
uv pip install --python .venv-esmc/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 -c $AIMM/env/constraints/esmc.txt esm -e $AIMM
printf '>ubiquitin\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n' > my.fasta
.venv-esmc/bin/python $AIMM/examples/embed.py \
  $LIP_DATA/lip/bundles/gemma-3-4b-it__esmc-600m.json my.fasta esmc
```

The ESM-C weights download is about 2.3 GB; allow about 5 GB RAM for extraction.
The script reads the model revision, layers 33 and 36, and mean-pooling recipe from the connector's
manifest. It writes 2,304 values per sequence to `esmc/<id>.npy`. `examples/embed.py` uses the recorded encoder recipe,
including recipes for registered [third-party extractors](guides/extending.md). Other specialists require their own
matching extraction recipe. Use the resulting embedding:

```bash
python $AIMM/examples/ask.py $AIMM/configs/receivers/gemma-3-4b-it.yaml \
  gemma-3-4b-it__esmc-600m esmc/ubiquitin.npy \
  "Is the protein <SLOT:x> an enzyme? Answer yes or no, then one sentence."
```

## 7. MIMIC from raw inputs

Use the public MIMIC 1.0 package to extract mean residue states at layers 16 and 20 for the existing
MIMIC connector. Keep the release environment active; the model runs in its own environment:

```bash
cd "$AIMM_WORK"
uv venv --python 3.12 .venv-mimic
uv pip install --python .venv-mimic/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 -c $AIMM/env/constraints/mimic.txt \
  torch transformers huggingface-hub \
  "mimic @ git+https://github.com/PolymathicAI/MIMIC.git@9e652f16491e6c2e3881111e24b285c288554275" -e $AIMM
python -m latent_interconnect.release_download \
  --repo sgolkar/ai-mindmeld-connectors --revision f69ffac6715789f730ef0985ef54ae82c7051954 \
  --connector gemma-3-4b-it__mimic --dest $LIP_DATA
printf '>ubiquitin\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n' > my.fasta
.venv-mimic/bin/python $AIMM/examples/embed_mimic.py \
  $LIP_DATA/lip/bundles/gemma-3-4b-it__mimic.json my.fasta mimic
python $AIMM/examples/ask.py $AIMM/configs/receivers/gemma-3-4b-it.yaml \
  gemma-3-4b-it__mimic mimic/ubiquitin.npy \
  "What does the protein <SLOT:x> do? Answer in one sentence." --max-new-tokens 64
```

This downloads about 2.8 GB of public MIMIC weights and writes 3,072 values per sequence to
`mimic/<id>.npy`. Allow about 14 GB RAM for extraction.

## 8. Try two DEMO non-linear connectors

**These are software DEMOs, not research results.** Each trained for 100 CPU steps; answers are often wrong.
Their weights are covered by [our CC BY 4.0 notice](../LICENSE-ARTIFACTS.md); Gemma 3 and ESM-C 600M terms also apply.

| Bundle | Kind | Input to `ask.py` | Tokens (write layers) |
|---|---|---|---|
| `gemma-3-4b-it__esmc-600m-mlp-demo` | `mlp` | pooled vector, `<id>.npy` | 1 (2, 7, 11, 17, 26, 33) |
| `gemma-3-4b-it__esmc-600m-xattn3-demo` | `xattn` (cross-attention) | per-residue states, `<id>.residues.npy` | 3 ([2, 17], [7, 26], [11, 33]) |

Both read ESM-C 600M layers 33 and 36 and start from the released ridge. Use the ESM-C environment and
`my.fasta` of step 6:

```bash
cd "$AIMM_WORK"
python -m latent_interconnect.release_download \
  --connector gemma-3-4b-it__esmc-600m-mlp-demo --connector gemma-3-4b-it__esmc-600m-xattn3-demo \
  --dest $LIP_DATA
.venv-esmc/bin/python $AIMM/examples/embed.py \
  $LIP_DATA/lip/bundles/gemma-3-4b-it__esmc-600m-mlp-demo.json my.fasta demo-pooled
.venv-esmc/bin/python $AIMM/examples/embed.py \
  $LIP_DATA/lip/bundles/gemma-3-4b-it__esmc-600m-xattn3-demo.json my.fasta demo-residues
python $AIMM/examples/ask.py $AIMM/configs/receivers/gemma-3-4b-it.yaml \
  gemma-3-4b-it__esmc-600m-mlp-demo demo-pooled/ubiquitin.npy \
  "What does the protein <SLOT:x> do? Answer in one sentence." --max-new-tokens 64
python $AIMM/examples/ask.py $AIMM/configs/receivers/gemma-3-4b-it.yaml \
  gemma-3-4b-it__esmc-600m-xattn3-demo demo-residues/ubiquitin.residues.npy \
  "What does the protein <SLOT:x> do? Answer in one sentence." --max-new-tokens 64
```

The xattn DEMO's recipe sets `residues: true`, so `embed.py` also writes `<id>.residues.npy`, one row of 2,304
values per residue. Given only `<id>.npy`, the xattn DEMO refuses and names the file it needs. A pooled kind given
`<id>.residues.npy` uses the mean of the rows, which is the pooled vector for `pooling: mean`; `ask.py` says so on
stderr before it answers.


## 9. Try a short training run

This example trains a Gemma 3 4B × ESM-C 600M connector for three steps of two examples on CPU. It uses
the released training data and ridge initialization; it checks the training path, not model quality.
Keep the checkout clean and the release environment active.

```bash
cd "$AIMM_WORK"
python -m latent_interconnect.release_download \
  --data protein-domain --data protein-training-set --data embeddings-esmc-600m \
  --data ridge-gemma-3-4b-it__esmc-600m --dest $LIP_DATA
python -m latent_interconnect.run.train \
  --config $AIMM/examples/configs/train-from-release/gemma-3-4b-it-esmc-600m-cpu-smoke.yaml \
  --destination train-from-release/gemma-3-4b-it/esmc-600m/cpu-smoke
```

The input download is 3.44 GB. Allow about 21 GB RAM and roughly four minutes on a CPU workstation.
Each step prints `train: step N/3 loss L loss_w W`. `loss` is the mean negative log-likelihood per answer
token; `loss_w` is the weighted objective, which can emphasize initial tokens, answer segments, or tasks.

Evaluate the resulting checkpoint:

```bash
python -m latent_interconnect.release_download --data protein-esmc-600m-mini --dest $LIP_DATA
python -m latent_interconnect.run.suite \
  --config $AIMM/examples/configs/train-from-release/protein-access-mini-cpu-smoke.yaml \
  --destination quickstart/protein-access-mini-cpu-smoke
python summarize_scores.py quickstart/protein-access-mini-cpu-smoke/scores
```

## Build your own connector

[Train a connector](guides/extending.md#6-train-a-connector-on-cpu) with generated configs for your LLM,
encoder, and `affine`, `mlp`, or `xattn3` kind. Then [package and share it](guides/contributing-a-connector.md)
from your own repository and submit a link to the LIP library.

## Data and licenses

Choose bundles from the [data inventory](../release/data_full.md) and
[Hugging Face data repository](https://huggingface.co/datasets/sgolkar/ai-mindmeld-data).
Download selected bundles with `--data NAME`. To download all full-data and mini bundles, preview the
sizes first; `--all-data` excludes software DEMOs:

```bash
python -m latent_interconnect.release_download --all-data --dry-run   # bundles and total size; downloads nothing
python -m latent_interconnect.release_download --all-data --dest $LIP_DATA
```

The cached and installed copies together require about 62 GB, before model weights and environments.
See [Models and licensing](../README.md#models-and-licensing) for code, artifact, and upstream terms.
