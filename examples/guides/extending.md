# Extending the package

A connector maps a specialist model's representations into a receiving LLM, called the receiver in
configuration files. The LLM's configuration is called a receiver config.

To train with an existing connector kind, start with [the CPU pipeline](#6-train-a-connector-on-cpu).
It generates consistent configs for your LLM and specialist, runs training and evaluation, and packages
the checkpoint. To add an architecture, domain, encoder, text stream, or LLM backend, use the interfaces
below. Each extension implements an interface, registers a name, and supplies a config.

Third-party modules go in the config's `plugins` list. They are imported before names are resolved.
Host your code and weights yourself, then
[submit a library listing](contributing-a-connector.md).

## 1. A new connector architecture

Implement `latent_interconnect.writes.kinds.EntryMap` and register it in `MAP_KINDS` in the same module.
Provide a `Params` config class, `create` (initialize from a ridge map), `empty` (shapes for loading), and
`forward`. Optional hooks are `init_random`, `noise_targets`, `lr_reference`, and `decayed` (weight decay).
Set `kind`, `kind_params`, and `plugins` in the connector config.

Run `latent_interconnect.writes.conformance.check_kind`: it checks output shape/dtype, checkpoint round trip,
and a learning-rate reference for every parameter. A zero-initialized parameter needs an explicit `lr_reference`.
See [gated_kind.py](../gated_kind.py) and its [config](../configs/gated-connector.yaml).

## 2. A new object domain and encoder

**Domain.** Implement `Domain`, register it in `DOMAINS` (both in `latent_interconnect.domains.base`), and
pass `latent_interconnect.domains.conformance.check_domain`. A `DomainSpec` identifies the domain by name and
its data digest. The `domain` stage builds its data from a `DomainDataSpec` subclass registered in
`latent_interconnect.domains.tables.DOMAIN_DATA`; sources are cited by path and sha256.

Every domain needs ridge targets: names or descriptions the LLM knows. `Domain.capture_recipes`
declares each `CaptureRecipe`'s prompt, placeholder, and screened context field with its token cap.
The connector's token `target` selects one. See [rna_domain.py](../rna_domain.py).

**Encoder.** Implement `Extractor` and register it in `EXTRACTORS` (both in
`latent_interconnect.encoders.base`), then write an `EncoderRecipe`. Keep the extractor in its own module:
module-level imports may use the standard library, numpy, and worker-side package modules; import model
libraries inside `load` so another environment can read the recipe.

For protein models, subclass `latent_interconnect.encoders.extractors.WindowedExtractor` and implement
`load` and `_forward`; it handles windows, padded-token batches, pooling, residue storage, and fingerprints.
[esm2_extractor.py](../esm2_extractor.py) and [esm2-8m.yaml](../configs/esm2-8m.yaml)
show a Hugging Face extractor plugin. [kmer_extractor.py](../kmer_extractor.py) and
[rna-kmer.yaml](../configs/rna-kmer.yaml) show a smaller example. For a non-ESM protein model, start
from a copy of `esm2_extractor.py` (§6, "For a different protein specialist").

A recipe pins the model revision and its `read` layers: hidden-state index 0 is the embedding output;
`num_hidden_layers` is the final layer. `configure_pipeline.py --inspect MODEL_ID` prints the revision
and depth. `environment: lip` runs extraction in the release environment; another environment name
requires `--option python=/path/to/encoder-env/bin/python` when running the extract stage.

### Make plugins importable

Put the plugin directory on `PYTHONPATH`, or install its package in every environment that runs it:
stages, encoder workers, embedding, and inference. For the shipped examples:

```bash
export PYTHONPATH="$AIMM${PYTHONPATH:+:$PYTHONPATH}"
```

Then `plugins: [examples.esm2_extractor]` resolves from any directory. Generated commands do this for
`examples.*` recipes; other plugin packages must already be importable. A recipe's plugins register extractors;
a connector's register map kinds; a receiver's register backends.

Stage provenance records this package's commit, not the plugin's. Keep plugin code under version control and
state its required version in the connector card. Bundles list plugin names but do not include their code.

## 3. New training text

For new tasks, create a new version of the domain data (task catalog, labels, evaluation frames). Runs and suites
cite its digest. Training wording belongs in the stream config under `configs/streams/` and is part of the run's
identity and zero-shot screening. Evaluation wording alone can use `SuiteSpec.frame` without new domain data.

For a new text format, implement `latent_interconnect.data.streams.Stream` with a `Params` class and register
it in `STREAMS`. Its `declare` method states `Supervision` from a values-free `CatalogView`; the resulting
`DomainView` and `TrainingText` restrict reads to that declaration. Give the stream a share in the run's mixture
(`latent_interconnect.data.sampler.MixtureEntry`). See [corpus_stream.py](../corpus_stream.py).

## 4. A different language model

The generic `hf` backend uses a `ReceiverSpec` to load a Hugging Face causal LM whose `get_decoder()` exposes
a list of decoder layers, including multimodal wrappers such as Gemma 3. At the pinned transformers 4.57.6,
GPT-2, GPT-NeoX, Falcon, and BLOOM are refused because their decoder is not found under `decoder` or `model`.

Generate a starting receiver config without downloading weights:

```bash
python "$AIMM/examples/configure_pipeline.py" --inspect MODEL_ID > my-receiver.yaml
```

Replace `MODEL_ID` with the Hugging Face model ID. Inspection reads config/tokenizer files, resolves a full
revision, and reports depth, width, decoder-layer path, that the placeholder is one token, and suggested
write layers. Use `--revision` to select another revision; gated models require access and authentication.
The placeholder is the slot token's text, `--placeholder` (default `" X"`); it must encode to exactly one token.
When it does not (TinyLlama's tokenizer encodes `" X"` as two), when no decoder-layer list is found, or when the
model's `architectures` name a causal LM that `AutoModelForCausalLM` cannot build (such as Llava or T5), nothing is
printed on stdout: the reason and, for the placeholder, a one-token candidate go to stderr, and the command exits 3. Rerun
with the suggested text, for example `--placeholder X`, and check the new `my-receiver.yaml`. The placeholder is
encoded on its own and its one token id is inserted for each slot token, so a leading space only selects which id that
is. Only an encoder-only model (such as `Rostlab/prot_bert`) prints its revision and final layer as an encoder recipe
and no receiver config (exit 0).
Check the generated name and fields before training. [llama-receiver.yaml](../configs/llama-receiver.yaml)
explains the fields.

The CPU pipeline uses no thinking mode. For reasoning prompts, configure the model's thinking switch and
closing text (`template_kwargs`, `thinking_close`), or its `separator` and `thinking_request`; the shipped
[Kimi](../../configs/receivers/kimi-k2.6.yaml) and [Gemma](../../configs/receivers/gemma-3-4b-it.yaml) configs give examples.
For a usable connector, capture states, scan candidate write layers, [choose layers](choosing_layers.md), fit
ridges, and train. Automatic layers below are a starting point for checking the pipeline.

### A new LLM backend

Register backend classes in `latent_interconnect.receivers.base.RECEIVER_BACKENDS`.
The receiver openers call `cls(spec, plan=..., **kwargs)`: `plan` comes from the stage config;
execution options are annotated constructor parameters bound by `latent_interconnect.config.bind_options`;
constructed objects such as loaded models arrive through the opener's `**construction`.
See `latent_interconnect.receivers.kimi_k26` for an implementation.

A training backend implements `TrainingBackend`. `open_training` checks for finite loss and nonzero, finite
gradients into every write, but cannot detect a partially disconnected gradient. Also test parity against the
`hf` backend on the same model in float64 on CPU: `receivers.conformance.check_training_parity` compares loss,
every write gradient, and an optimizer update; `check_finite_differences` checks the reference.

Models that disable gradients in part of their forward pass need preparation. Register its check in
`receivers.base.PREPARATIONS` and name it in the receiver's `training_preparation`; the backend refuses
training until it passes. Kimi's `enable_moe_gradients` is an example.

## 5. Running the stages

Each stage uses `python -m latent_interconnect.run.<stage> --config FILE --pin NAME`.
`--option NAME=VALUE` controls execution (devices, memory, interpreter), while anything that changes results
belongs in the config. Options are recorded in provenance. Configs refer to upstream outputs with `{$pin: name}`.

A config's `destination` must match its pin. Replacing a pin or declaring a rerun canonical requires
`--repin REASON`, and history retains the change. Pin the first execution: rerunning an unpinned computation
and then trying to pin it creates ambiguity. To adopt a recorded execution without recomputing, use
`python -m latent_interconnect.run.pin NAME --record RECORD_ID`.

Stages require a clean, committed checkout and record its commit. Keep generated configs and data outside it.

### Extending a training run

To continue a finished run to more steps, raise `steps` in a copy of its config and pass the finished run's steps as
`extend_from`:

```bash
python -m latent_interconnect.run.train --config run-6000-steps.yaml --option extend_from=5000 --pin NAME
```

The stage refuses if the new config differs from the finished one in anything but `steps`, or if the learning-rate
schedule depends on the total steps (`cosine`). A completed run keeps its final state, `resume.pt`, for this: about
three times the connector's weights (22–25 GB for a large multi-token MLP connector). Deleting it frees the space,
but the run can then no longer be extended.

## 6. Train a connector on CPU

This small example uses Qwen2.5-0.5B-Instruct with ESM-C 300M to train a connector on 32 proteins
(24 train, 8 evaluation) for 10 steps. It checks the pipeline end to end; answers from this tiny run are often wrong.
The models are public and download about 1 GB and 1.2 GB respectively.

### Configure and run

Complete [Quickstart installation](../README.md#1-install). Only if your recipe uses
`environment: esmc`, as this example's ESM-C 300M recipe does, also create the
[ESM-C environment](../README.md#6-embed-your-own-proteins-esm-c-600m): run only that step's
`uv venv` and `uv pip install` lines (in `$AIMM_WORK`, so `ENCODER_PY` below finds it), not its embedding and ask.
Keep the release environment active and `AIMM` pointing to the clean checkout. Choose a work root with space,
then create a fresh run directory and data root. The `--receiver` argument selects the LLM through its
receiver config, and `--encoder` supplies the specialist recipe:

```bash
export AIMM_WORK=${AIMM_WORK:-$HOME/aimm-work}
mkdir -p "$AIMM_WORK"
export PIPELINE_WORK=$(mktemp -d "$AIMM_WORK/pipeline.XXXXXX")
export ENCODER_PY=$AIMM_WORK/.venv-esmc/bin/python   # change if your ESM-C environment is elsewhere
cd "$PIPELINE_WORK"
export LIP_DATA=$PIPELINE_WORK/data CUDA_VISIBLE_DEVICES=""
python "$AIMM/examples/configure_pipeline.py" \
  --receiver "$AIMM/examples/configs/pipeline/qwen2.5-0.5b-instruct.yaml" \
  --encoder "$AIMM/examples/configs/pipeline/esmc-300m.yaml" \
  --kind affine --pin-prefix my-name/qwen --out configs
printf '>P05204\nMPKRKAEGDAKGDKAKVKDEPQRRSARLSAKPAPPKPEPKPKKAPAKKGEKVPKGKKGKADAGKEGNNPAENGDAKTDQAQKAEGAGDAK\n' > my.fasta
LICENSE='<your license>' DEMO=1 bash configs/commands.sh
```

Replace `<your license>` with your connector's own license, chosen consistently with the LLM's and encoder's
terms; packaging (`latent_interconnect.bundle_connector --license`) records it in `bundle.json` as given.
`DEMO=1` marks the bundle as a software demonstration.
The default bundle name is `<receiver>__<recipe>-<kind>`; `--name` overrides the connector name.
`OUT` sets packaging's output directory (default `hub/connectors`, relative to the run directory): the bundle is
written to `$OUT/<receiver>__<connector name>`.

The generator writes the configs and `commands.sh`. Before running stages, the script checks the recipe's
plugins and the inference FASTA. It then runs these steps, printing stage times and a score summary (one line
per suite, condition and task; condition `name` gives the protein's name in the prompt text only, with no connector
read, and `own` has the connector read the protein's own embedding):

| Steps | Work | Configs |
|---|---|---|
| 1–3 | Build the domain and train/eval/all item sets | `domain.yaml`, `items-*.yaml` |
| 4 | Extract specialist states | `extract.yaml` |
| 5–7 | Capture LLM states, fit ridges, initialize | `capture.yaml`, `ridge.yaml`, `init.yaml` |
| 8–11 | Evaluate, build training text, train, evaluate again | `suite-init.yaml`, `training_set.yaml`, `train.yaml`, `suite-trained.yaml` |
| 12 | Embed `my.fasta` and ask about `P05204` | copied encoder/LLM configs, trained checkpoint |
| Package | Write checkpoint, metadata, and index | `hub/connectors/` |

**Optional: layer-selection diagnostic.** For this Qwen/ESM-C setup, follow the
[eight diagnostic commands](choosing_layers.md#run-the-example-diagnostic) in the same data root. They use
the shipped `examples/configs/pipeline/capture.yaml` to capture `[0, 4, 8, 12, 16, 20]` at
`mine/qwen/capture`. The generated capture above is `[0, 4, 8, 12, 16]` at `my-name/qwen/capture`;
it lacks layer 20 and cannot replace the diagnostic capture.

**Use a fresh `LIP_DATA` for changed data, recipes, or settings.** `commands.sh` skips a step only when its pin
holds this config's own record: before each step, `configure_pipeline.py --verify-pin` compares the step's config
digest, loaded with the current pins, with the `config_sha` of the record pinned at `<pin>@record`. A pin from other
inputs or settings, or one it cannot verify, stops the script with a refusal line followed by the recovery routes
below, and exit 1. A pinned step whose config does not load (for example a validation error after
a hand edit) is not a reuse conflict: the script prints the error and "fix the config" and exits 1. The domain, item, and training-set pins are fixed names, so `--pin-prefix` alone does not isolate them. Share a root
only between pipelines whose upstream data inputs are identical; the shared steps are then skipped, and kinds or
prefixes can share it. Do not delete old pins to force a rerun; keep the old root and create a fresh one,
together with a fresh run directory: packaging refuses to write another checkpoint's bundle over
`hub/connectors/<bundle>` of the old run. Step 12 and packaging have no pin check, so they always run again.

After the refusal line, `commands.sh` prints two recovery routes with this run's values, every path quoted so the
printed lines paste as they are:

- **A new run** (keeps the old run and root): `cd "$(mktemp -d <parent>/pipeline.XXXXXX)" && export
  LIP_DATA=$PWD/data`, where `<parent>` is the parent of the current run directory (`$AIMM_WORK` above), then the
  same `configure_pipeline.py` command with `--out configs` (printed with its flags). Hand edits to the run
  directory's configs do not carry over by themselves: configure writes them again from the templates. The next
  printed line, `cp -- <old configs>/{domain.yaml,...,train.yaml,...} configs/`, copies every config you may edit
  (`domain.yaml`, `items-*.yaml`, the stage configs, `connector.yaml`, `train.yaml` and the receiver and encoder
  copies) over the regenerated ones. It leaves out `domain-filled.yaml` and `domain-check/` (filled by `fill()`)
  and `commands.sh` (written for its own run directory): redo a hand edit of `commands.sh`, such as of `fill()`, in
  the new one, or give the training data as `--data-fasta` and `--data-tsv`, which carry over. The last line is
  `export FASTA=<the old FASTA's full path> ID=<ID> && bash configs/commands.sh`, with `LICENSE` and `DEMO` when they
  are set; keep your other variables (`ENCODER_PY`).
- **The same run directory with a fresh data root and bundle directory:** `LIP_DATA=<new dir> OUT=<new hub dir>
  FASTA=<the old FASTA's full path> ID=<ID> bash configs/commands.sh`, with `LICENSE` and `DEMO` when they are set,
  as in the new-run route. With the fresh root, `configs/domain-filled.yaml` is filled again from the current data.

Resuming is supported only as described here: an unchanged rerun skips the
steps whose pins hold their own records.

### Your LLM and encoder

Repeat the setup above in a new run directory and data root. Set `--receiver` to the receiver config for
your LLM and `--encoder` to your specialist's recipe. Then choose `--kind affine`, `mlp`, or `xattn3`.
For Gemma 3 4B, use `$AIMM/configs/receivers/gemma-3-4b-it.yaml` after accepting that model's terms and
logging in. A CPU smoke run observed about 12 GB RAM for inference and 21 GB for training.

For a different protein specialist, [esm2-8m.yaml](../configs/esm2-8m.yaml) supplies a plugin recipe
that runs in the release environment (`environment: lip`), so no `ENCODER_PY` is needed. For another ESM-2
size, copy it and change `name`, `model_id`, `revision`, and `read`. Inspection prints the full revision and final
hidden-state index; for example, `facebook/esm2_t30_150M_UR50D` uses `read: [30]`. Keep the plugin declaration.
For a non-ESM model, copy [esm2_extractor.py](../esm2_extractor.py), change `load` and `_forward` and any
tokenizer quirks (ProtBert, for example, wants its residues separated by spaces), and register and import it under
a new name (the `EXTRACTORS.register` name and the recipe's `extractor` and `plugins`). Put the plugin's folder on
`PYTHONPATH` yourself: `commands.sh` adds only the checkout, for `examples.*` plugins.

`commands.sh` embeds the file named by `FASTA` (default `my.fasta`) and asks about `ID` (default `P05204`).
Set these when using another sequence. These variables select the inference example, not the training data.

### Your training data

Give `configure_pipeline.py` your FASTA and annotation TSV as `--data-fasta` and `--data-tsv` (or edit the two
paths in `commands.sh`'s `fill()` line). When the domain is already pinned, its config and a copy of the sources are
filled under `$P/domain-check`. Follow
[proteins.tsv](../data/proteins.tsv): columns `accession`, `split` (`train`, `val`, or `test`),
semicolon-separated `keywords`, comma-and-space-separated `lineage`, and `caption`.
An optional `cluster` groups homologs; item sets take whole clusters. Assign splits by homology where needed.
Edit `configs/domain.yaml`'s label rules, tasks, frames, and capture targets to match your annotations.

`make_domain.py` builds the four source tables under `LIP_DATA` and writes `domain-filled.yaml` with their hashes.
The supplied examples are [32 UniProtKB/Swiss-Prot entries](../data/README.md).
To increase training steps, edit `train.yaml`, the checkpoint referenced in `suite-trained.yaml`, and the
checkpoint used for inference and packaging in `commands.sh` together.

### Choosing a connector kind

| Kind | Input | Write operation | Initialization |
|---|---|---|---|
| `affine` | Pooled vector | Affine map | Ridge map |
| `mlp` | Pooled vector | Residual feed-forward block in a `d_s`-wide stream | Rank-`d_s` ridge approximation |
| `xattn` | Per-residue states | Learned-query attention pooling, then projection | Near the rank-`d_s` ridge of the residue mean; exact at `q_init: 0` |
| `xblock` | Per-residue states | Attention pooling plus a feed-forward block | As `xattn` |

The generator supports `affine`, `mlp`, and `xattn3` (three staggered `xattn` tokens). For `mlp` it sets
`ff_mult: 2`; for `xattn3`, `heads: 4` and `d_k: 32`. The default `d_s` is `min(256, receiver width)`, rounded
down to a multiple of four for `xattn3`. It must also fit the specialist input width, which the generator does
not infer: use `--d-s` for a narrower encoder. Residue kinds require `d_s` divisible by `heads`.
For custom configs, state `kind_params` explicitly; `xblock` also takes `ff_mult`.

`xattn3` creates a residue recipe when needed (`residues: true`, a new name, float32 rows).
`embed.py` then writes `<id>.npy` and `<id>.residues.npy`; `ask.py` selects the required input from the checkpoint
kind and refuses a pooled-only file for a residue kind. Residue storage and attention costs grow with sequence
length. More slot tokens require matching entries in `tokens` and `layers.per_token`.

### Write layers

`--layers auto` writes explicit starting layers into the config; it does not perform a layer study.
For depth `d`, set `s = max(1, round(d/6))` (halves up) and `t = floor(2d/3)`.
One token writes at `s, 2s, ...` through `t`. For `k` staggered tokens, use `o = max(1, floor(s/2))`:
token 0 writes at `s, s+ko, ...` through `t`, and token `i` offsets these by `io`, dropping layers at or beyond `d`.
Capture and ridge configs include layer 0 and every write layer.

| LLM depth | One token | Three staggered tokens |
|---|---|---|
| 16 (OLMo-2 1B) | `[3, 6, 9]` | `[[3,6,9], [4,7,10], [5,8,11]]` |
| 24 (Qwen2.5-0.5B) | `[4, 8, 12, 16]` | `[[4,10,16], [6,12,18], [8,14,20]]` |
| 34 (Gemma 3 4B) | `[6, 12, 18]` | `[[6,15], [9,18], [12,21]]` |

Pass your own layers with `--layers 4,8,12,16`, or one list per token with
`--layers 4,10,16/6,12,18/8,14,20`. For a production connector, use the
[layer-selection guide](choosing_layers.md).

### Checks and sharing

To publish the trained checkpoint, follow [Packaging and sharing a connector](contributing-a-connector.md).
Keep its code, weights, card, and support in your repository; submit only the library entry here.

## Adding a field to a stored model

Stored records, checkpoint configs, and stage configs are identified by `Config.identity` digests.
Declare new fields with `latent_interconnect.config.added_field` and the default represented by old objects.
While the field holds that default, it stays out of the identity and old digests remain valid. Check this when
adding fields: old serialized objects should retain their identities when loaded with the default.
