# LIP library

Find a connector by specialist and receiving LLM (receiver), then download its versioned bundle. Each connector depends
on the exact specialist revision, extraction recipe, and LLM recorded in its manifest.

[Quickstart](../examples/README.md) · [Submit a connector](../CONTRIBUTING.md#list-your-connector)

## Uploaded connectors

**25 release connectors**, available on
[Hugging Face](https://huggingface.co/sgolkar/ai-mindmeld-connectors).
Two additional `*-demo` bundles in the index are software examples
([Quickstart step 8](../examples/README.md#8-try-two-demo-non-linear-connectors)), excluded from this count.

The tables link revision `72a79954ec2780f49f96ccf3a4bdaf0d19642f6b`. Sizes are decimal MB for connector
files only; specialist and LLM weights are separate downloads. Copy the bundle name into
`--connector`; repeat the option to select more than one.

```bash
python -m latent_interconnect.release_download --list
python -m latent_interconnect.release_download \
  --revision 72a79954ec2780f49f96ccf3a4bdaf0d19642f6b \
  --connector gemma-3-4b-it__chemfm-3b --dest ~/aimm-data
```

Choose an LLM to browse its protein and molecule connectors, ordered by decreasing **total AUROC**
(the equally weighted mean of trained-task and unseen-group AUROC, the paper's weighting).
Protein totals use the five unseen groups.
[Scores and ranking details](../release/library-ranking.json) use each released checkpoint.
ChemFM-3B scores use a different molecule evaluation set from the other molecule specialists.

**Interpreting the scores.** These AUROCs describe the released connectors, not the underlying specialists'
capabilities: we did not optimize connector training separately for each specialist, and affine connectors rely on
the linear compatibility of the specialist embedding and the receiving LLM's representations. We also deliberately
held out tasks during training to test zero-shot transfer. Training on all tasks would be expected to
produce better connectors for those tasks; the released checkpoints preserve the paper's evaluation setup.

<details open>
<summary><strong>Kimi K2.6</strong></summary>

### Protein

| Specialist | Injection | Total AUROC | Bundle name / download | MB | Manifest |
|---|---|---:|---|---:|---|
| ESM-C 6B | MLP, four staggered tokens | 0.92 | [kimi-k2.6__esmc-6b-mlp4-staggered](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__esmc-6b-mlp4-staggered) | 8,055.9 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__esmc-6b-mlp4-staggered/bundle.json) |
| ESM-C 6B | Affine, one token | 0.89 | [kimi-k2.6__esmc-6b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__esmc-6b) | 881.0 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__esmc-6b/bundle.json) |
| PGLM-1B | Affine, one token | 0.89 | [kimi-k2.6__pglm-1b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__pglm-1b) | 352.5 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__pglm-1b/bundle.json) |
| ESM-C 600M | Affine, one token | 0.89 | [kimi-k2.6__esmc-600m](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__esmc-600m) | 396.6 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__esmc-600m/bundle.json) |
| MIMIC | Affine, one token | 0.88 | [kimi-k2.6__mimic](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__mimic) | 528.7 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__mimic/bundle.json) |

### Molecule

| Specialist | Injection | Total AUROC | Bundle name / download | MB | Manifest |
|---|---|---:|---|---:|---|
| ChemFM-3B | MLP, four tokens | 0.88 | [kimi-k2.6__chemfm-3b-mlp4](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__chemfm-3b-mlp4) | 8,307.6 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__chemfm-3b-mlp4/bundle.json) |
| ChemFM-1B | Affine, one token | 0.84 | [kimi-k2.6__chemfm-1b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__chemfm-1b) | 704.9 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__chemfm-1b/bundle.json) |
| MoLFormer-c3 | Affine, one token | 0.80 | [kimi-k2.6__molformer-c3](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__molformer-c3) | 264.5 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__molformer-c3/bundle.json) |
| Uni-Mol2 1.1B | Affine, one token | 0.78 | [kimi-k2.6__unimol2-1.1b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__unimol2-1.1b) | 528.7 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/kimi-k2.6__unimol2-1.1b/bundle.json) |

</details>

<details>
<summary><strong>Gemma 3 4B</strong></summary>

### Protein

| Specialist | Injection | Total AUROC | Bundle name / download | MB | Manifest |
|---|---|---:|---|---:|---|
| ESM-C 6B | Affine, one token | 0.77 | [gemma-3-4b-it__esmc-6b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__esmc-6b) | 314.7 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__esmc-6b/bundle.json) |
| ESM-C 600M | Affine, one token | 0.76 | [gemma-3-4b-it__esmc-600m](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__esmc-600m) | 141.7 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__esmc-600m/bundle.json) |
| MIMIC | Affine, one token | 0.76 | [gemma-3-4b-it__mimic](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__mimic) | 188.8 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__mimic/bundle.json) |
| PGLM-1B | Affine, one token | 0.74 | [gemma-3-4b-it__pglm-1b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__pglm-1b) | 125.9 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__pglm-1b/bundle.json) |

### Molecule

| Specialist | Injection | Total AUROC | Bundle name / download | MB | Manifest |
|---|---|---:|---|---:|---|
| ChemFM-1B | Affine, one token | 0.80 | [gemma-3-4b-it__chemfm-1b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__chemfm-1b) | 251.8 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__chemfm-1b/bundle.json) |
| ChemFM-3B | Affine, one token | 0.76 | [gemma-3-4b-it__chemfm-3b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__chemfm-3b) | 377.6 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__chemfm-3b/bundle.json) |
| Uni-Mol2 1.1B | Affine, one token | 0.72 | [gemma-3-4b-it__unimol2-1.1b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__unimol2-1.1b) | 188.8 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__unimol2-1.1b/bundle.json) |
| MoLFormer-c3 | Affine, one token | 0.68 | [gemma-3-4b-it__molformer-c3](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__molformer-c3) | 94.5 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-4b-it__molformer-c3/bundle.json) |

</details>

<details>
<summary><strong>Gemma 3 12B</strong></summary>

### Protein

| Specialist | Injection | Total AUROC | Bundle name / download | MB | Manifest |
|---|---|---:|---|---:|---|
| ESM-C 600M | Affine, one token | 0.82 | [gemma-3-12b-it__esmc-600m](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__esmc-600m) | 212.5 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__esmc-600m/bundle.json) |
| ESM-C 6B | Affine, one token | 0.82 | [gemma-3-12b-it__esmc-6b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__esmc-6b) | 472.0 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__esmc-6b/bundle.json) |
| MIMIC | Affine, one token | 0.81 | [gemma-3-12b-it__mimic](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__mimic) | 283.3 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__mimic/bundle.json) |
| PGLM-1B | Affine, one token | 0.81 | [gemma-3-12b-it__pglm-1b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__pglm-1b) | 188.9 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__pglm-1b/bundle.json) |

### Molecule

| Specialist | Injection | Total AUROC | Bundle name / download | MB | Manifest |
|---|---|---:|---|---:|---|
| ChemFM-3B | Affine, one token | 0.82 | [gemma-3-12b-it__chemfm-3b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__chemfm-3b) | 566.4 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__chemfm-3b/bundle.json) |
| ChemFM-1B | Affine, one token | 0.81 | [gemma-3-12b-it__chemfm-1b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__chemfm-1b) | 377.6 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__chemfm-1b/bundle.json) |
| MoLFormer-c3 | Affine, one token | 0.76 | [gemma-3-12b-it__molformer-c3](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__molformer-c3) | 141.7 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__molformer-c3/bundle.json) |
| Uni-Mol2 1.1B | Affine, one token | 0.76 | [gemma-3-12b-it__unimol2-1.1b](https://huggingface.co/sgolkar/ai-mindmeld-connectors/tree/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__unimol2-1.1b) | 283.3 | [Manifest](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/gemma-3-12b-it__unimol2-1.1b/bundle.json) |

</details>

The [index](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/connectors.json)
records checkpoint identities and manifest checksums. Manifests identify model revisions, configurations,
and file checksums.

For training details, see the [paper's methods](https://polymathic-ai.org/papers/aimm.pdf). Each connector's
manifest in the [tables above](#uploaded-connectors), [specialist read configs](../configs/shared/), and
[receiving LLM write configs](../configs/connectors/) describe its representations and injection layout.
For receiving LLM writes: Layer `L` is the input to decoder block `L`, counted from zero; layer 0 is the token embedding.
Use `python -m latent_interconnect.release_download --list --kinds` to list kinds, inputs, and token counts.
Evaluation data are in the [data repository](https://huggingface.co/datasets/sgolkar/ai-mindmeld-data) and its
[pinned index](https://huggingface.co/datasets/sgolkar/ai-mindmeld-data/resolve/a27d7808625794b31e44ad9b909816c17748b8f0/data.json),
with mini sets of [eight proteins](https://huggingface.co/datasets/sgolkar/ai-mindmeld-data/tree/a27d7808625794b31e44ad9b909816c17748b8f0/protein-esmc-600m-mini)
and [eight molecules](https://huggingface.co/datasets/sgolkar/ai-mindmeld-data/tree/a27d7808625794b31e44ad9b909816c17748b8f0/molecule-chemfm-1b-mini);
the [full inventory](../release/data_full.md) covers reusable training inputs, shared ridge bundles, verification, and compatibility.
For your own inputs, follow the [ESM-C](../examples/README.md#6-embed-your-own-proteins-esm-c-600m) or
[MIMIC](../examples/README.md#7-mimic-from-raw-inputs) examples, using the specialist revision and extraction
recipe recorded in your connector's manifest.

## Licenses

[Apache-2.0](../LICENSE) covers our code. Our connector weights, the two minimal evaluation sets, and ridge
starts are [CC BY 4.0](../LICENSE-ARTIFACTS.md). Credit the [paper first and repository second](../README.md#citation),
and retain upstream attribution and terms. Specialist and receiver models, separately released embedding
bundles, other full datasets, and community connectors retain their own terms.
See [non-commercial items](../LICENSE-ARTIFACTS.md#non-commercial-items) for applicable restrictions.

See the
[data inventory](../release/data_full.md#licenses-and-provenance)
for source terms and grant scope.

## Community connectors

Authors host and maintain their own code, cards, downloads, and support. We maintain the links.
[Send a PR](../CONTRIBUTING.md#list-your-connector) with a name, short description, supported LLMs,
and links to your project and versioned downloads. To build and package one, see
[Extending the package](../examples/guides/extending.md) and
[Packaging and sharing a connector](../examples/guides/contributing-a-connector.md).
