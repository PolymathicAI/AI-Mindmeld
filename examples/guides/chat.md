# Interactive chat (web GUI)

Ask about proteins and molecules in the browser or through the HTTP API. Placeholders such as `[[P1]]`
connect objects to specialist representations. Ask focused questions and request direct answers;
see [Limitations](../../README.md#limitations). The library's AUROC scores come from direct Yes/No
evaluation, rather than conversational responses.

## Gemma setup and launch

Complete [Quickstart installation and authentication](../README.md#1-install), including its shell variables
and `chat` extra. Run commands from the checkout, `cd "$AIMM"`; `aimm-chat --help` lists `check` and `serve`.
For GPU serving, undo the Quickstart's empty `CUDA_VISIBLE_DEVICES`: use `unset CUDA_VISIBLE_DEVICES`,
or restore the GPU selection supplied by your scheduler. An empty value forces CPU execution.
The installed command serves one LLM in the foreground using [release.yaml](../../configs/chat/release.yaml).
`check` inspects the configuration, installed connectors, visible GPUs and free memory, Hugging Face
credentials, and encoder environments without loading a model. A failed required setup check returns a
nonzero exit status. Choose the same model and execution options for checking and serving.

Accept the terms for [Gemma 3 4B](https://huggingface.co/google/gemma-3-4b-it) or
[Gemma 3 12B](https://huggingface.co/google/gemma-3-12b-it) separately, then run `hf auth login`.
Authentication uses `HF_TOKEN` first, otherwise `HF_TOKEN_PATH` (default `$HF_HOME/token`).
Weights download into the Hugging Face cache at the revision pinned in the receiving LLM config:
about **8.6 GB for 4B** or **24 GB for 12B**. To prefetch:

```bash
hf download google/gemma-3-4b-it --revision 093f9f388b31de276ce2de164bdc2081324b9767
hf download google/gemma-3-12b-it --revision 96b6f1eccf38110c56df3a15bffe176da04bfd80
```

Download the ESM-C 600M connector and its domain, training text, monitor items, embeddings, and ridge
start (about **3.5 GB**, separate from LLM weights):

```bash
python -m latent_interconnect.release_download --connector gemma-3-4b-it__esmc-600m \
  --data protein-domain --data protein-training-set --data protein-items \
  --data embeddings-esmc-600m --data ridge-gemma-3-4b-it__esmc-600m --dest $LIP_DATA
```

Defaults select the pinned [release downloads](../../release/data_full.md#downloads-and-verification).
For 12B, substitute `gemma-3-12b-it__esmc-600m` and `ridge-gemma-3-12b-it__esmc-600m`.
`check` lists installed connectors and download commands for missing ones:

```bash
aimm-chat check --model gemma-3-4b-it
aimm-chat serve --model gemma-3-4b-it
```

These commands serve cached objects without an encoder. For raw proteins, create the
[ESM-C environment](../README.md#6-embed-your-own-proteins-esm-c-600m), then add
`--encoder-python esmc=$AIMM_WORK/.venv-esmc/bin/python` to **both** commands. Live encoders use the first
visible GPU alongside the LLM; add `--encoder-device cpu` to keep them on CPU. Use `--model gemma-3-12b-it`
for 12B, or `--port N` if port 8811 is occupied. For example, the raw-protein variant is:

```bash
aimm-chat check --model gemma-3-4b-it --encoder-python esmc=$AIMM_WORK/.venv-esmc/bin/python
aimm-chat serve --model gemma-3-4b-it --encoder-python esmc=$AIMM_WORK/.venv-esmc/bin/python
```

Keep the release environment active for these commands; the separate interpreter runs the encoder.
The 4B and 12B connectors share domain tables, cached embeddings, training text, and monitor items,
so those data bundles can be installed once for both models.

Open `http://127.0.0.1:8811/`. Enter the first line of `~/.config/aimm/chat-password`; `serve` creates
this file with a random password and mode 600 on first start. For shared/headless use, select
`--password-file /path/to/password` in a directory only you can read. The UI shows loading phases;
`/api/health` reports `ready: true` or `startup.failed`. Press Ctrl-C or send SIGTERM to stop;
restart with another `--model` to switch LLMs. On first start, model downloading precedes loading:
the connectors and cached embeddings load first, followed by the LLM. Wait for readiness before sending
your first question. For remote access, use HTTPS with `--secure-cookie`;
`--no-auth` is restricted to loopback.

Default caps are **2,056 prompt tokens** and **256 answer tokens**. Adjust with `--max-prompt-tokens N`
and `--max-tokens-cap N` (up to 2,048); `--max-prompt-tokens 0` uses Gemma's 8,192-token context.
See `aimm-chat serve --help` for other options.

## Use the browser

Choose **Add placeholder**, select a deployed connector and input type, and enter an object.
For ESM-C 600M, select `uniprot`, enter cached accession `A0A011QK89`, and insert `[[P1]]` into
“What is the likely function of [[P1]]? Answer directly.” A molecule example is `CHEBI:10002`,
with a molecule connector and `chebi`. Proteins also accept `sequence`; molecules accept `smiles`.
A cached ID uses stored vectors; an uncached domain ID or raw value requires the matching live encoder.

Enter sends; Shift+Enter inserts a newline. Enter does not send during input-method composition.
Alt+P opens Add placeholder; Escape closes side drawers and dismisses dialogs.
History and settings use browser storage. Each generation request sends the retained conversation and
resolves its referenced placeholders again. Speed depends on engine, hardware, conversation length, and reasoning.

## Yes/No probability

For Gemma, add `--yes-no $AIMM/configs/answers/gemma-3.yaml` to `serve` to enable the **Yes/No probability** switch.
Write one question without an answer instruction: scoring appends the configured instruction and prefix
(`Answer:` for Gemma), disables thinking, and uses only that user turn, without history or a system prompt.
The card shows full-softmax P(Yes), P(No), the suite score `log P(Yes) − log P(No)`, and the distinct
pair-normalized value `P(Yes) / (P(Yes) + P(No))`. Both the scored message and its card are excluded from
later generation conversations. To suppress future notices in this browser, select the notice's checkbox
and press **OK**. Kimi scores with `kimi-stream` but generates with `kimi-vllm`, so its chat has no
Yes/No switch and ignores `--yes-no`.

## GPU memory and live inputs

Observed whole-device peaks on one **H100 (95,830 MiB)**, in **bf16**, with the default **2,056/256-token**
caps and sampling once a second:

| Phase | Gemma 3 4B | Gemma 3 12B |
|---|---|---|
| Loading | 11.2 GiB | 27.2 GiB |
| Answering with a cached id | 11.3 GiB | 31.1 GiB |
| Answering with a raw sequence (ESM-C 600M encoding live) | 13.2 GiB | 33.0 GiB |

Only the last row includes ESM-C 600M on the same GPU. These are observed peaks, not minimum GPU sizes.
Longer prompts, higher caps or another encoder use more memory; leave a few GiB of headroom.

ESM-C 600M needs its separate environment; ESM-C 6B also needs a GPU. ChemFM, MoLFormer, and Uni-Mol2
need their matching encoder environments; see [representation compatibility](../../release/data_full.md#compatibility).
The matching MIMIC checkpoint is not public, so release chat cannot encode raw MIMIC inputs;
[public MIMIC 1.0](../README.md#7-mimic-from-raw-inputs) provides an offline embedding route.
The release does not set up a live PGLM environment.

`--max-resident-encoders 1` keeps one live specialist resident. Live inputs, embeddings, and write banks
use the separate disposable `--store` (default `$LIP_DATA/chat/store`), outside the released artifact store.
Deleting this chat store requires live inputs to be encoded again. Switching to a different specialist
can also require loading its encoder when the resident-encoder limit is reached.

To add a connector, adapt the [deployment configuration](../../configs/chat/all-connectors.yaml) and supply
it with `--config`. Its receiving LLM, checkpoint, specialist recipe, and referenced data must match;
install every domain, embedding, ridge, training-set, and monitor-item pin named by its run. The
[configuration source](../../src/latent_interconnect/chat/config.py) describes those fields.

## Kimi K2.6

Use **eight whole H200 GPUs on one node**, all visible, without MIG. Kimi needs a separate vLLM environment:
vLLM 0.28 uses transformers 5, while the release environment uses transformers 4. The driver must support
CUDA 13. From the checkout, with the Quickstart's `AIMM_WORK` set:

```bash
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python --no-deps -r env/locks/vllm.txt
export CUDA_HOME=$PWD/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git "$AIMM_WORK/DeepGEMM"
git -C "$AIMM_WORK/DeepGEMM" checkout 8b1392b978f5a03c828dd1711090d7fb50958b8a
git -C "$AIMM_WORK/DeepGEMM" submodule update --init --recursive
mkdir -p "$AIMM_WORK/cuda-lib"
ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$AIMM_WORK/cuda-lib/libcudart.so"
ln -sf "$CUDA_HOME/lib/libnvrtc.so.13" "$AIMM_WORK/cuda-lib/libnvrtc.so"
LIBRARY_PATH="$AIMM_WORK/cuda-lib" uv pip install --python .venv-vllm/bin/python --no-deps --no-build-isolation \
  "$AIMM_WORK/DeepGEMM"
uv pip install --python .venv-vllm/bin/python --no-deps -e .
.venv-vllm/bin/python -c "import vllm, deep_gemm, latent_interconnect; print(vllm.__version__)"
```

Read and accept the [Kimi model terms](https://huggingface.co/moonshotai/Kimi-K2.6).
Predownload the pinned weights into the Hugging Face cache (**about 595 GB**):

```bash
hf download moonshotai/Kimi-K2.6 --revision 7eb5002f6aadc958aed6a9177b7ed26bb94011bb
```

For a copy on another disk, add `--option weights=DIR`; keep the pinned cache snapshot because the
backend checks the copy against it. The default deployment offers the ESM-C 6B four-token MLP
connector (**8.06 GB**) and the two Kimi connectors trained on all tasks. Download the first and its inputs from the
release environment:

```bash
python -m latent_interconnect.release_download --connector kimi-k2.6__esmc-6b-mlp4-staggered \
  --data protein-domain --data protein-training-set --data protein-items \
  --data embeddings-esmc-6b --data ridge-kimi-k2.6__esmc-6b --dest $LIP_DATA
```

On the GPU node, use the vLLM environment's commands:

```bash
.venv-vllm/bin/aimm-chat check --model kimi-k2.6
.venv-vllm/bin/aimm-chat serve --model kimi-k2.6
```

Raw sequences require the ESM-C interpreter option on both commands, as for Gemma. ESM-C 6B shares the
first GPU with Kimi and needs extra memory; `--option gpu_memory_utilization=0.8` reduces vLLM's reserved
share from 0.9 to leave encoder headroom. Loading can take tens of minutes; use the same readiness,
password, and stop flow as Gemma. The same prompt/answer caps apply; `--max-prompt-tokens 0` uses Kimi's
16,384-token context. The bundled Kimi adapter disables prefix caching, so each generation turn prefills
the retained conversation again.
The node must expose all eight whole GPUs when `check` and `serve` run. Keep the pinned model snapshot
available to that node as well as the installed connector data. The DeepGEMM build itself does not
require a GPU. Use `.venv-vllm/bin/aimm-chat` explicitly so the server loads the vLLM installation rather
than the release environment's command.

## HTTP API

`GET /api/health` is unauthenticated. Other `/api/*` routes except `login` need a Bearer password or
session cookie; `POST /api/login {token}` creates the cookie. In another shell, select the password file
used by your running server and a total timeout in seconds:

```bash
CHAT_PASSWORD_FILE=$HOME/.config/aimm/chat-password   # match serve --password-file if changed
read -r CHAT_PASSWORD < "$CHAT_PASSWORD_FILE"
read -r -p 'Total request timeout in seconds: ' CHAT_TIMEOUT
curl --fail --max-time "$CHAT_TIMEOUT" -H "Authorization: Bearer $CHAT_PASSWORD" \
  http://127.0.0.1:8811/api/config
curl --fail -N --max-time "$CHAT_TIMEOUT" \
  -H "Authorization: Bearer $CHAT_PASSWORD" -H 'Content-Type: application/json' \
  http://127.0.0.1:8811/api/chat --data-binary @- <<'JSON'
{"messages":[{"role":"user","content":"What is the likely function of [[P1]]? Answer directly."}],"placeholders":{"P1":{"connector":"esmc-600m","input_type":"uniprot","value":"A0A011QK89"}},"thinking":false,"sampling":{"temperature":0,"max_tokens":256}}
JSON
```

The example uses `esmc-600m` from the Gemma setup. `/api/config` lists deployed connector IDs and input types;
use those IDs, not download bundle names. `POST /api/placeholder/check` accepts `{connector, input_type, value}`
and reports `cache`, `live`, or an error. It resolves the object without generating an answer, which helps
check that the selected connector can use the input. A domain ID missing from the cache is encoded
from the domain table's sequence or SMILES, so checking that ID also needs the live encoder.
`readout: "yes_no"` follows the browser scoring rules above.

Chat returns SSE events: `status` with a request ID, `content`/`reasoning` deltas, then terminal `done` or
`error`. Handle that terminal event; an HTTP success alone does not establish completion. Your timeout is a
chosen total limit, not a completion estimate; 15-second keepalives do not establish progress.
`POST /api/stop {request_id}` or disconnecting stops a request; during live encoding, stopping takes effect
when encoding returns, and the completed encoding remains in the chat store for reuse.

See the [Chat CLI demo](../demos/README.md) for requests and transcripts. Existing references cover
[request/event models](../../src/latent_interconnect/chat/service.py),
[HTTP handling](../../src/latent_interconnect/chat/server.py), [configuration](../../src/latent_interconnect/chat/config.py),
[deployment checks](../../src/latent_interconnect/chat/smoke.py), and the
[default](../../configs/chat/release.yaml) and [all-connectors](../../configs/chat/all-connectors.yaml) configurations.
