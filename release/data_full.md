# Data for training, evaluation, and inference

Use these bundles to train connectors, evaluate them on labeled objects, or ask questions about stored
specialist representations. The [Quickstart](../examples/README.md) covers installation, model access,
and small examples. The [LIP library](../library/README.md) lists compatible pretrained connectors.

## Contents

The data index contains **41 bundles: 38 full-data bundles, two minimal evaluation sets, and one software
DEMO bundle**. The connector index contains **27 bundles: 25 pretrained connectors and two software DEMOs**.

| Data | Contents | Bundle names |
|---|---|---|
| Domains | Object tables, splits, clusters, labels, task catalogs, and text fields; protein data also include variants. | `protein-domain`, `molecule-domain` (includes `molecule-text`) |
| Training sets | Text and annotations consumed by the training streams. | `protein-training-set`, `molecule-training-set` |
| Item sets | Objects selected for evaluation and training monitoring. | `protein-items`, `molecule-items` |
| Embeddings | Specialist representations for training, evaluation, and stored-vector inference, at the read layers each connector uses. | `embeddings-<specialist>` |
| Ridge starts | Initial mappings from specialist features to receiving LLM states, at the connector's write layers. | `ridge-<receiver>__<specialist>` |
| Minimal evaluation sets | Eight proteins with ESM-C 600M vectors or eight molecules with ChemFM-1B vectors, together with their domain tables and evaluation items. | `protein-esmc-600m-mini`, `molecule-chemfm-1b-mini` |
| Residue DEMO | Pooled and per-residue ESM-C 600M states for the protein mini set, for the software examples. | `protein-esmc-600m-residues-mini-demo` |

## Downloads and verification

Set up the release environment and `LIP_DATA` as in [Quickstart steps 1–2](../examples/README.md#1-install).
Use the indexes to choose bundles and preview storage requirements:

```bash
python -m latent_interconnect.release_download --list
python -m latent_interconnect.release_download --list --kinds
python -m latent_interconnect.release_download --all-data --dry-run
```

`--all-data` selects the **38 full-data bundles and two mini sets**; the DEMO bundle is separate.
`--dry-run` lists sizes without downloading. Downloads use both a Hugging Face cache and an installed
copy, so allow space for both, plus model weights and environments. Select only the inputs you need
with repeated `--data NAME` or `--connector NAME` options.

For example, download the molecule tables and text without loading a specialist or LLM:

```bash
python -m latent_interconnect.release_download \
  --data molecule-domain --data molecule-training-set --data molecule-items --dest $LIP_DATA
```

The defaults pin data revision `a27d7808625794b31e44ad9b909816c17748b8f0` and connector revision
`72a79954ec2780f49f96ccf3a4bdaf0d19642f6b`. The
[data index](https://huggingface.co/datasets/sgolkar/ai-mindmeld-data/resolve/a27d7808625794b31e44ad9b909816c17748b8f0/data.json)
and [connector index](https://huggingface.co/sgolkar/ai-mindmeld-connectors/resolve/72a79954ec2780f49f96ccf3a4bdaf0d19642f6b/connectors.json)
record bundle paths, sizes, and manifest checksums. Use `--data-revision` and `--revision` to select other
full commit revisions explicitly.

Each `aimm-index/1` index records the sha256 of an `aimm-bundle/1` bundle manifest. The downloader checks
that manifest, each payload's size and sha256, and each artifact's manifest digest against its ID before
committing or pinning that artifact. Successful installation prints `verified N installed files (sha256)`.
A failed check stops the download and identifies the file and cache entry; follow that removal instruction
before retrying. Keep verification enabled and use the reported checkpoint directory rather than guessing
an artifact ID.

Installed artifacts live under `$LIP_DATA/lip/artifacts/`, their names are recorded in
`$LIP_DATA/lip/pins.yaml`, and bundle manifests are kept in `$LIP_DATA/lip/bundles/`. The manifests identify
model revisions, extraction recipes, files, and applicable terms. Use the named pins in configurations;
content IDs identify the exact installed artifacts.

## Inputs for training, evaluation, and inference

**Inference** needs a compatible pretrained connector, its receiving LLM, and an embedding made with the
specialist revision and extraction recipe in the connector's manifest. Stored embeddings avoid loading
the specialist. Raw inputs need its separate dependencies and model weights; model access terms still apply.

**Evaluation** adds the matching domain table and labeled item set. Start with a mini set and the
[Quickstart evaluation commands](../examples/README.md#5-run-a-small-evaluation). Full suite configs
in [configs/suites](../configs/suites/) declare their domain, objects, tasks, embeddings, and connector.

**Training** adds a training set and ridge start. The [run configs](../configs/runs/) name their inputs;
training-monitor items are separate from evaluation items. Training also needs the LLM weights and the
resources required by that model. The short [CPU training example](../examples/README.md#9-try-a-short-training-run)
uses Gemma 3 4B with ESM-C 600M.

The table maps connector families to reusable inputs. `<receiver>` is the LLM's identifier:
`gemma-3-4b-it`, `gemma-3-12b-it`, or `kimi-k2.6`. Downloading data does not install the specialist's environment.

| Connectors | Data bundles | New embeddings from your own inputs |
|---|---|---|
| `<receiver>__esmc-600m` (3) | `protein-domain`, `protein-training-set`, `protein-items`, `embeddings-esmc-600m`, `ridge-<receiver>__esmc-600m` | Use `examples/embed.py`; see [ESM-C setup](../examples/README.md#6-embed-your-own-proteins-esm-c-600m). |
| `<receiver>__mimic` (3) | the protein bundles above, `embeddings-mimic`, `ridge-<receiver>__mimic` | Use the public MIMIC 1.0 model with `examples/embed_mimic.py`; see [MIMIC setup](../examples/README.md#7-mimic-from-raw-inputs). |
| `<receiver>__pglm-1b` (3) | the protein bundles above, `embeddings-pglm-1b`, `ridge-<receiver>__pglm-1b` | Use `configs/encoders/pglm-1b.yaml`; supply an interpreter with PGLM's dependencies for its `esmc` environment entry ([encoder setup](../examples/guides/extending.md#2-a-new-object-domain-and-encoder)). |
| `gemma-3-4b-it__esmc-6b`, `gemma-3-12b-it__esmc-6b`, `kimi-k2.6__esmc-6b`, `kimi-k2.6__esmc-6b-mlp4-staggered` (4) | the protein bundles above, `embeddings-esmc-6b`, `ridge-gemma-3-4b-it__esmc-6b`, `ridge-gemma-3-12b-it__esmc-6b` or `ridge-kimi-k2.6__esmc-6b` | Use `configs/encoders/esmc-6b-l51l80.yaml`, an `esmc` environment and a GPU. |
| `<receiver>__chemfm-1b` (3) | `molecule-domain`, `molecule-training-set`, `molecule-items`, `embeddings-chemfm-1b`, `ridge-<receiver>__chemfm-1b` | Use `configs/encoders/chemfm-1b.yaml` in a matching specialist environment. |
| `<receiver>__molformer-c3` (3) | the molecule bundles above, `embeddings-molformer-c3`, `ridge-<receiver>__molformer-c3` | Use `configs/encoders/molformer-c3-mean-l9.yaml` and `configs/encoders/molformer-c3-eos-l12.yaml` for the two representation reads. |
| `<receiver>__unimol2-1.1b` (3) | the molecule bundles above, `embeddings-unimol2-1.1b`, `ridge-<receiver>__unimol2-1.1b` | Use `configs/encoders/unimol2-1.1b.yaml`. |
| `gemma-3-4b-it__chemfm-3b`, `gemma-3-12b-it__chemfm-3b`, `kimi-k2.6__chemfm-3b-mlp4` (3) | `molecule-domain`, `molecule-training-set`, `molecule-items`, `embeddings-chemfm-3b`, `ridge-gemma-3-{4b,12b}-it__chemfm-3b` or `ridge-kimi-k2.6__chemfm-3b` | Use `configs/encoders/chemfm-3b.yaml`; stored-vector inference is shown in [Quickstart step 4](../examples/README.md#4-ask-a-question). |

The Kimi K2.6 ESM-C 6B affine and four-token MLP connectors share a ridge bundle containing the union of
their write layers. ESM-C 6B inputs are pooled layers 51 and 80; ChemFM-3B inputs use layer 27.

Molecule training and its construction check also read the ChEBI release 255 ontology, which the molecule
training set's screen cites by sha256 and no bundle supplies. Download it from the EBI archive and verify it:

```bash
mkdir -p $LIP_DATA/datasets/smallmol
curl -L -o $LIP_DATA/datasets/smallmol/chebi.obo.gz \
  https://ftp.ebi.ac.uk/pub/databases/chebi/archive/rel255/ontology/chebi.obo.gz
echo "c2765eb4599fbcbb265385f5280a241c83ef3cecb302f92afdc2879f9c7e7d6c  $LIP_DATA/datasets/smallmol/chebi.obo.gz" \
  | sha256sum -c -
```

The current ChEBI download URL serves a newer release, whose file does not match this sha256.

## Reusable examples

- **Ask and evaluate with stored vectors:** follow [Quickstart steps 3–5](../examples/README.md#3-download-connectors-and-evaluation-data).
  The mini sets include their embeddings; the ChemFM-3B example uses its full embedding bundle.
- **Train from released inputs:** follow [Quickstart step 9](../examples/README.md#9-try-a-short-training-run)
  and its [training config](../examples/configs/train-from-release/gemma-3-4b-it-esmc-600m-cpu-smoke.yaml).
- **Train on your own data:** the [extension guide](../examples/guides/extending.md#6-train-a-connector-on-cpu) uses
  [configure_pipeline.py](../examples/configure_pipeline.py) and [pipeline templates](../examples/configs/pipeline/)
  to create domain, extraction, capture, ridge, training, and evaluation configurations from your FASTA and annotations.
- **Embed new proteins:** use [embed.py](../examples/embed.py) with the connector manifest or an encoder recipe;
  see the [ESM-C](../examples/README.md#6-embed-your-own-proteins-esm-c-600m) and
  [public MIMIC](../examples/README.md#7-mimic-from-raw-inputs) examples. Keep the recipe's model revision,
  read-layer order, pooling, and environment. The extension guide explains [third-party extractors](../examples/guides/extending.md#2-a-new-object-domain-and-encoder).

## Compatibility

A connector's specialist representation and receiving LLM are part of its identity. Match the specialist
revision, read layers, pooling, vector width, and LLM configuration in the bundle. A shared model name alone
is not sufficient. For Uni-Mol2, live single-molecule embeddings can differ from stored batch-computed
representations; use stored vectors when comparing against the corresponding benchmark inputs.

The full embedding bundles contain the pooled representations used by the released affine and MLP connectors.
Connectors that read per-residue states require residue embeddings; the
[nonlinear software examples](../examples/README.md#8-try-two-demo-non-linear-connectors) show that input format.
To extract different layers or representations, use your own objects and a matching encoder recipe through
the [extension workflow](../examples/guides/extending.md#6-train-a-connector-on-cpu).

Ridge starts initialize new training runs.

## Licenses and provenance

| Data | Source, release | sha256 (source file) | License as stated |
|---|---|---|---|
| protein domain, training set, items | UniProtKB/Swiss-Prot 2026_03 (02-Sep-2026): `uniprot_sprot.dat.gz`, the REST TSV export (`swissprot.parquet`), `keywlist.txt`, mutagenesis/variant features + `humsavar.txt` (variant table) | dat.gz `c87208822a56…`, swissprot.parquet `1f6eaf3dde66…`, fields `352f9cc38a3f…`, keywlist `0a03fc48ffe5…`, uniprot_features `679a3c078198…`, humsavar `68ce06dc07c5…`, variants `d9a10dcd36b9…` | **CC BY 4.0**: release README "LICENSE" section ("Creative Commons Attribution 4.0 International (CC BY 4.0) License ... to all copyrightable parts of our databases", (c) UniProt Consortium); humsavar footer "Distributed under the Creative Commons Attribution (CC BY 4.0) License" |
| protein split / clusters | ours: MMseqs2 id30/c80 clustering of Swiss-Prot | split `44eb9b597d64…`, clusters `35e7397879c9…` | derived from UniProt (CC BY 4.0); no separate grant stated for our derived split |
| molecule domain (`molecule-domain`), text, training sets, items | ChEBI release 255 (`chebi.obo` data-version 255, 2026-09-09) and its flat-file SMILES; UniProt catalytic table | chebi.obo.gz `c2765eb4599f…`, restored entries `cbfb87c134e5…`, labels `9f537ab2c1cc…`, uniprot catalytic `fef8714a279e…`, split `94451bfd0b19…` | **CC BY 4.0**: ChEBI about page ("The data on this website is available under the Creative Commons License (CC BY 4.0)"; "All data in the ChEBI database is non-proprietary or is derived from a non-proprietary source"), EMBL-EBI terms of use |
| embeddings ESM-C 600M / 6B | `biohub/ESMC-600M@fcc9d36f`, `biohub/ESMC-6B@af1602ba` | | model: HF tags `mit` + `other`; weights under the **MIT License**, as the official repository states ("These models are available under the MIT license", [Biohub/esm LICENSE.md](https://github.com/Biohub/esm/blob/main/LICENSE.md)); use per the [Biohub acceptable use policy](https://biohub.org/acceptable-use-policy/) |
| embeddings PGLM-1B | `biomap-research/proteinglm-1b-mlm@676c1bcb` | | **CC BY-NC 4.0** (non-commercial): model card "open source under the Creative Commons Attribution-NonCommercial 4.0 International License"; the model repository's [LICENSE](https://huggingface.co/biomap-research/proteinglm-1b-mlm/blob/676c1bcb9f737c166a25587f83fd580610acb1ad/LICENSE) is that license's text. PGLM-1B's non-commercial terms apply to these embeddings |
| embeddings MIMIC | Polymathic AI checkpoint `5.7_main/E20-D12-H1536-pt7-synced` (sha256 `1113846c2218…`) | | the source checkpoint listed here is not publicly released and no public license is stated for it; contact the authors before use. The public MIMIC 1.0, `polymathic-ai/MIMIC`, is tagged `mit` |
| embeddings ChemFM-1B | `ChemFM/ChemFM-1B@f99dc2e8` | | **CC BY-NC 4.0** (non-commercial): the official repository's [LICENSE](https://github.com/TheLuoFengLab/ChemFM/blob/master/LICENSE), where the paper (arXiv:2410.21422, "Code availability") places the model checkpoints; the Hugging Face card's `mit` tag is metadata without a license text |
| embeddings ChemFM-3B | `ChemFM/ChemFM-3B@c1464dd3` | | **CC BY-NC 4.0** (non-commercial), as ChemFM-1B: the official repository's [LICENSE](https://github.com/TheLuoFengLab/ChemFM/blob/master/LICENSE); the Hugging Face card's `mit` tag is metadata without a license text |
| embeddings MoLFormer-c3 | `DeepChem/MoLFormer-c3-1.1B@9f1b9ea3` (code `ibm-research/MoLFormer-XL-both-10pct`) | | HF tags `mit` (code `apache-2.0`) |
| embeddings Uni-Mol2 1.1B | unimol_tools 0.1.6 `unimolv2/1.1B` (checkpoint sha256 `48e1ba230119…`) | | `dptech/Uni-Mol2` HF tag `mit`; the checkpoint is that repository's `modelzoo/1.1B/checkpoint.pt` at revision `bf37d253` (same sha256) |
| ridge starts (incl. ChemFM-3B and the ESM-C 6B ones) | fits to Gemma 3 4B/12B and Kimi K2.6 hidden states | | Our contributions: [CC BY 4.0](../LICENSE-ARTIFACTS.md). Each bundle's `license` field states the grant, the credit and the LLM's and specialist's terms, which also apply |
| DEMO data bundle `protein-esmc-600m-residues-mini-demo` (index group `demo`) | pooled and per-residue ESM-C 600M states (layers 33 and 36) of the protein mini set's eight objects, with its domain and items; a software demo | | not covered by our CC BY 4.0 grant; no license is stated for this bundle. The UniProtKB/Swiss-Prot (CC BY 4.0) and ESM-C 600M terms above apply |
| minimal eval sets (`protein-esmc-600m-mini`, `molecule-chemfm-1b-mini`) | cut from the protein / molecule domain and items and the ESM-C 600M / ChemFM-1B embeddings above | | Our contributions: [CC BY 4.0](../LICENSE-ARTIFACTS.md), covering each set as packaged, its embeddings included; the specialist's terms on those embedded vectors apply on top (ChemFM-1B: non-commercial), with source terms retained; each bundle's `license` field states the grant, the credit and those terms |

Our connector weights, the two named minimal evaluation sets, and ridge starts are licensed under
[CC BY 4.0](../LICENSE-ARTIFACTS.md), with [paper-first, repository-second credit](../README.md#citation).
The other full-data bundles, including the separately released embedding bundles, and the residue DEMO
bundle are outside this grant and retain their own source-specific terms.
Upstream source, LLM, and specialist terms remain applicable, including the non-commercial terms of ChemFM
and PGLM-1B ([items](../LICENSE-ARTIFACTS.md#non-commercial-items)).

BindingDB, Rhea, PFMbench, ProteinGym and the Neuronpedia SAE explanations are not redistributed; obtain them from their sources.
