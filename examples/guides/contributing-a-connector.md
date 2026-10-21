# Packaging and sharing a connector

Host your connector's code, weights, card, and support in your own repository. The
[LIP library](../../library/README.md#community-connectors) lists links.
To submit a listing or improve the main repository, follow [CONTRIBUTING.md](../../CONTRIBUTING.md).

## 1. Package

The receiving LLM is called the receiver in configuration files and download reports.

You need the release environment, a committed training checkpoint in `$LIP_DATA`, and the run config
that produced it. The [generated pipeline](extending.md#6-train-a-connector-on-cpu) already packages its
checkpoint into `hub/connectors/`. The commands below package a checkpoint from your own training run.
Set the variables for your own run before continuing:

```bash
export SHARE_WORK=$HOME/aimm-share
export CONNECTOR_REPO=your-name/your-connectors       # your Hugging Face model repository
export CHECKPOINT_PIN=your-name/my-run/train/step_002000
export RUN_CONFIG=/absolute/path/to/train.yaml
export CONNECTOR_NAME=my-connector
export CONNECTOR_LICENSE=MIT                        # replace with your connector's actual license
python -m latent_interconnect.bundle_connector \
  --pin "$CHECKPOINT_PIN" --run-config "$RUN_CONFIG" \
  --name "$CONNECTOR_NAME" --out "$SHARE_WORK/hub/connectors" \
  --license "$CONNECTOR_LICENSE" --card-url "https://huggingface.co/$CONNECTOR_REPO"
```

For a bundle made by the generated pipeline, which is already packaged, skip this command and set
`SHARE_WORK=$PIPELINE_WORK` and `CONNECTOR_NAME=<receiver>__<recipe>-<kind>` (its bundle name) before §2.
`--name` names the connector in `configure_pipeline.py` but the bundle in `bundle_connector`.

Use your own namespace for the checkpoint pin: installation refuses to overwrite an existing pin that
points elsewhere. `--install-pin` can set this name at packaging time. If you have a checkpoint directory
instead of a pin, use `--checkpoint DIR` (containing `MANIFEST.json`, `connector.json`, and
`weights.safetensors`).

The packager checks file hashes, connector/config identities, and pinned model revisions before writing
the bundle and its `connectors.json` index. The bundle records the models, injection layout, input kind
(pooled or residues), plugins, receiver config, and license. With `--pin`, it also records training provenance
from the pin history; `--source-commit SHA` supplies an explicit code revision when needed.
Repeat with the same `--out` to add another connector. Use `--demo` for software demonstrations and
`--note TEXT` for a short qualification.

The project's Apache-2.0 license covers its code. Choose your connector's license consistently with its
model and data terms.

## 2. Check and publish

Install into a fresh data directory, then run your inference example against that installation:

```bash
export CHECK_DATA=$(mktemp -d "$SHARE_WORK/check-data.XXXXXX")
python -m latent_interconnect.release_download --local-source "$SHARE_WORK/hub" \
  --connector "$CONNECTOR_NAME" --dest "$CHECK_DATA"
```

The report gives the checkpoint directory, input kind, required plugins, and receiver config to use with
`examples/ask.py`. The packaged bundle installs the config at `$CHECK_DATA/lip/receivers/<receiver>.yaml`.
Set `LIP_DATA=$CHECK_DATA` when checking inference. Embed from the installed bundle and ask with the installed
receiver config (`ENCODER_PY` is the encoder environment's interpreter, `python` for `environment: lip`; use
`emb/<id>.residues.npy` for a residue kind such as `xattn3`):

```bash
export LIP_DATA=$CHECK_DATA
"${ENCODER_PY:-python}" $AIMM/examples/embed.py "$CHECK_DATA/lip/bundles/$CONNECTOR_NAME.json" my.fasta emb
python $AIMM/examples/ask.py "$CHECK_DATA/lip/receivers/<receiver>.yaml" "$CONNECTOR_NAME" emb/<id>.npy \
  "Where in the cell is the protein <SLOT:x> found? Answer in one sentence." --max-new-tokens 48
```

Document how to install any plugin code; the bundle names plugins but does not distribute them.

Add a [connector card](../../library/CONNECTOR_CARD_TEMPLATE.md) as
`$SHARE_WORK/hub/connectors/README.md`. Include the training data, specialist read layers, pooling,
receiving LLM write layers, token placement, tested AI Mindmeld version, and a runnable inference example.

Upload to your own Hugging Face model repository and record its full commit:

```bash
hf auth login
hf upload "$CONNECTOR_REPO" "$SHARE_WORK/hub/connectors" . --repo-type model
python -c 'import os; from huggingface_hub import HfApi; print(HfApi().model_info(os.environ["CONNECTOR_REPO"]).sha)'
```

Use the printed 40-character commit in the installation command you publish:

```bash
export CONNECTOR_REVISION=REPLACE_WITH_FULL_40_CHARACTER_COMMIT
python -m latent_interconnect.release_download \
  --repo "$CONNECTOR_REPO" --revision "$CONNECTOR_REVISION" \
  --connector "$CONNECTOR_NAME" --dest "$LIP_DATA"
```

Custom repositories require a full commit; branches, tags, and short hashes are refused.

## 3. Submit a listing PR

Fork AI Mindmeld, create a branch, and add an entry under
[Community connectors](../../library/README.md#community-connectors): name, one-sentence description,
supported LLMs, and links to your card and versioned download. Include the install command and
inference example in the PR. Open it against `main`, titled `Library: <name>`.

For maintenance responsibilities and changes to AI Mindmeld, see [CONTRIBUTING.md](../../CONTRIBUTING.md).
