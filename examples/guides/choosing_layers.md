# Choosing layers

- **Scope:** how to read the layer-selection scans and write a connector's layers into its config. The tools are in
  `latent_interconnect.evaluation.layer_selection` (the read-layer scan in
  `latent_interconnect.evaluation.read_scan`, the set comparison in `latent_interconnect.evaluation.layer_sets`; the
  LLM scans' shared parts in `latent_interconnect.evaluation.scan_states`). Where layers live in a config:
  `latent_interconnect.writes.layers` (write layers) and `latent_interconnect.writes.inputs.InputSpec` (read layers).
  Why they were chosen goes in `latent_interconnect.writes.connector.LayerChoice`.
- **Choose layers from the scans.** Read the diagnostics, write explicit layer lists into the config, and cite
  the scan records in `LayerChoice.records`. The paper's layers were picked by hand from these diagnostics for each
  run. This guide explains what to look for and the rule of thumb the paper used (§2).
- **The process**, in stages: the `read_scan` stage (property and variant probes, CPU), from which a person picks
  the read layers (§3); then the `ridge` stage's held-out scan ridge from them (fit on the selection objects minus
  the scan items) and the `study` stage's injection scan of its predictions, read next to the domain's own-state
  reference (`scan` stage), from which a person picks the write layers (§2).
- **A new LLM without a study.** For a first check of a setup (as in `examples/guides/extending.md` §6), spread 4 to 6
  write layers over the first two thirds of the LLM's depth, and say so in `LayerChoice.note`. That checks that
  the pieces fit; it is not a layer choice to report results from.

Layer numbering on both sides: layer L is `hidden_states[L]`. For the LLM, a write at L is applied at the input of
decoder layer L, and layer 0 is the token-embedding output. For an encoder, layer 0 is the embedding output, layer i is
the output of block i, and the last index is the final output after the final norm.

## 1. The scans

| scan | function | answers |
|---|---|---|
| own-state reference (oracle source) | `injection_scan` | where the LLM can use its own state written at one layer (no encoder); a reference, not a ceiling: a mapped state can beat it |
| LLM injection, ridge source | `injection_scan` | how much an encoder's ridge prediction reaches, per layer; a study scans the held-out scan ridge, fit without the scanned objects (`build.ridge`, `exclude`) |
| encoder read layers | `read_layer_scan` | which encoder layers carry the domain's properties: cross-validated ridge probes of every Yes/No task and, for proteins, of the variant labels |
| layer sets | `compare_layer_sets` | how whole candidate write sets compare at initialization, paired |

For one (LLM, encoder) setup the diagnostic is a `LayerSelectionStudy` (the held-out ridge's injection scan, the
`study` stage, a GPU job for a large LLM), the LLM's own-state reference and the encoder's read-layer scan
(the `read_scan` stage, CPU). The own-state reference depends on no encoder, so it is one `scan` record (a
`ScanRunConfig`) per LLM-side spec, cited by every setup of that domain; the analysis refuses it unless it is
the study's scan in every field but the written states, run with the study's device class and batch plan, and writes
the capture state the held-out ridge fits (the same capture and `pool`: the capture's stored pooled state, or with
`pool: mean` its read-token means). A fixed-source own-state curve writes one layer S's state at every layer above 0
(`fixed_source`, `evaluation.scan_states.WrittenScan`); it shows whether a state formed at S helps when written
elsewhere, and it is never a study's reference.

Once candidate sets have been read off them, the `scan` stage compares the candidates (a `LayerSetComparisonSpec`),
without re-running the diagnostic. After the write layers are chosen, the same comparison can explore the connector's
step-0 α schedules; `layer_sets.connector_start` writes a scored start into the connector config (diagnostics
refused; the module docstring). One schedule is computed from the single-layer scores themselves:
`layer_sets.auroc_weighted_alphas` starts each layer in proportion to its mean AUROC above 0.5 and leaves out a layer
at or below it; `layer_sets.clipped_auroc_weighted_alphas` clips those starts to [0.1, 0.9] without rescaling, so
every layer is written. Such a candidate is selected on the data it is then scored on: report it as data-selected
and exploratory, never as independent evidence.

A layer-selection study runs inside the `train` split: the
probes are cross-validated over cluster folds, the held-out ridge never sees the scanned objects, it never sees
`val` or `test`, and the own-state reference scores the study's items. The setup's `layer_selection` analysis
(`latent_interconnect.evaluation.layer_selection_analysis`) spans the study, own-state reference, held-out ridge,
read-scan and set-comparison records, and refuses records that are not one setup's.

Layer scans require the LLM, specialist embeddings, and enough memory for the selected batch plan.
The small example below runs on CPU; large LLMs require GPU resources.

### Run the example diagnostic

First run the unchanged Qwen2.5-0.5B/ESM-C 300M [worked pipeline](extending.md#6-train-a-connector-on-cpu).
Keep its release environment, ESM-C interpreter (`ENCODER_PY`), `AIMM`, `PIPELINE_WORK`, and `LIP_DATA`.
The diagnostic reuses `mine/protein-domain` and `mine/items-train` from that example. Keep the checkout clean
and committed, and copy the [templates](../configs/pipeline/) to a new directory outside it:

```bash
cd "$PIPELINE_WORK"
export CUDA_VISIBLE_DEVICES=""
cp -r "$AIMM/examples/configs/pipeline" layer-selection
```

Run the eight stages below in order. The separate [capture.yaml](../configs/pipeline/capture.yaml) captures
`[0, 4, 8, 12, 16, 20]` under `mine/qwen/capture`. The worked pipeline's generated capture at
`my-name/qwen/capture` contains only `[0, 4, 8, 12, 16]`; it cannot supply the layer-20 diagnostic.
Each `--pin` below matches its template's `destination`.

```bash
python -m latent_interconnect.run.capture --config layer-selection/capture.yaml \
  --pin mine/qwen/capture
python -m latent_interconnect.run.items --config layer-selection/items-scan.yaml \
  --pin mine/items-scan
python -m latent_interconnect.run.extract --config layer-selection/extract-layers.yaml \
  --pin mine/esmc-300m-layers/embeddings --option python="$ENCODER_PY"
python -m latent_interconnect.run.read_scan --config layer-selection/read-scan.yaml \
  --pin mine/esmc-300m-layers/read-scan
python -m latent_interconnect.run.ridge --config layer-selection/ridge-heldout.yaml \
  --pin mine/qwen/ridge-heldout
python -m latent_interconnect.run.scan --config layer-selection/scan-reference.yaml \
  --pin mine/qwen/scan-reference
python -m latent_interconnect.run.study --config layer-selection/study.yaml \
  --pin mine/qwen/study
python -m latent_interconnect.run.analysis --config layer-selection/analysis-layer-selection.yaml \
  --pin mine/qwen/layer-selection
```

Extraction uses [esmc-300m-layers.yaml](../configs/pipeline/esmc-300m-layers.yaml) to read six specialist
layers. The read scan compares them on training objects; the held-out ridge excludes `mine/items-scan`,
and the LLM scan and study score only those excluded objects. The final analysis links the study,
own-state reference, held-out ridge, and read-scan records at `mine/qwen/layer-selection`. Read the diagnostics using sections 2–3 below.
For another LLM or specialist, adapt the model references, layer lists, answer words, and dependent pins
together in a fresh data root, following the [extension guide](extending.md#6-train-a-connector-on-cpu).

## 2. Reading the LLM scans

From the injection curves, read off:

- **onset:** the first layer where a single write becomes clearly usable;
- **peak:** where a single write works best, and the layers close to it;
- **fade:** the depth beyond which a single write is near chance.

The own-state curve describes the LLM. A ridge curve describes the LLM plus one encoder, and its onset can
differ from the own-state curve's. **The paper's rule of thumb:** 5 or 6 write layers spanning the trunk, one per
block of about 10-12 layers, each picked from the scan for coverage of the question types (per-task curves), not
only the peak.

**Noise floor.** A scan's scores depend on how its layers are grouped into calls (bf16 batching, `examples/guides/batching.md`;
`layers_per_call` is part of the identity, so the oracle and every study share it). Measured on 300 protein scan
items, layers 0 and 15 scored in one 16-layer call vs alone: per-task AUROC moved by up to 0.074 and the five-task
mean by up to 0.028, 10-14% of a typical above-chance gap. Treat a layer that keeps 0.6-0.8 of the peak's
above-chance AUROC as a judgment call.

**Considerations:**

- Layer 0 can carry a write (a constant gives the first layers something to attend to).
- The first object-dependent write near the onset, and writes in the onset-to-peak band, are what the single-layer
  scans support.
- **Deep writes.** Single-layer and joint scans at initialization do *not* show a benefit from writes beyond the
  fade. Adding deep layers to an in-band set gave slightly negative point estimates, within noise. The case for
  writing across the whole depth rests on trained runs that wrote deep, which also differed in other ways, so the
  evidence is confounded. Use the layer-set comparison, and say in `LayerChoice.note` which consideration decided.
- **Gemma 3 (sliding-window layers).** Five of every six Gemma 3 layers attend over a 1,024-token window only (full
  attention: 4B 5, 11, 17, 23, 29; 12B every sixth, 5 to 47; `configs/receivers/gemma-3-<size>.yaml`). A write that
  must reach an answer more than 1,024 tokens after the slot (an answer after a reasoning trace) needs a
  full-attention layer at or after it, and the scans' short prompts cannot show this.
- With several slot tokens, a staggered assignment (tokens on different layers) and a shared one (all tokens on the
  same layers) are different models. Choose explicitly.

## 3. Reading the encoder scan

- Compare, per encoder layer, the probe AUROC of the properties (function, localization, taxonomy, size, variant
  effect). The scan has no LLM targets.
- Reading where most properties peak, plus a late layer for what peaks at the end of the encoder (structure, taxonomy,
  variant effects), is one reasonable choice. Pair scores that combine many metrics can prefer other pairs. Which
  metrics matter is a choice; state it in `LayerChoice.note`.
- Encoders with a defined output (register tokens, a summary vector) can read that output instead.
- **Three diagnostics per encoder:** (1) a layer scan of the mean-pooled tokens, specials excluded; (2) the special
  tokens (CLS, BOS, EOS or registers) at the final layer only; (3) the representation the encoder's own paper
  recommends.
- **Joint reads.** When the families peak at different layers, a second read scan declares hand-chosen
  `combinations`, each probed like a layer on the concatenated pooled vectors, with the layers they use. Keep them
  few (the paper used at most 8, of 2 layers, rarely 3). `python -m latent_interconnect.run.read_scan_report NAME
  [--drop PROBE ...]` prints each combination's probe mean (unstable probes dropped), variant AUROCs and within-fold
  AUROCs next to the best single layer (`read_scan.compare_reads`).
- **Two extractions jointly.** A combination joins layers of one extraction; a read of another extraction is appended
  by the spec's `append` (its recipe, artifacts and one layer), concatenated after every read per object. A chosen
  joint read is trained the same way: an `InputSpec`'s `append` (the other recipe and one layer; its artifacts listed
  beside the input's in the citing `embeddings`) feeds the ridge, the injection scan and the connector through the
  one loader (`encoders.loader.InputLoader`); e.g. `configs/shared/read-molformer-c3.yaml`.

## 4. Writing the choice down

Explicit lists in the connector config, with the evidence cited by record key. The `layer_choice` field is rationale:
editing it does not change the connector's identity. The example connectors `examples/configs/gated-connector.yaml`
and `examples/configs/pipeline/connector.yaml` show the shape.
