---
license: other
license_name: ai-mindmeld-artifacts
license_link: https://github.com/PolymathicAI/AI-Mindmeld/blob/main/LICENSE-ARTIFACTS.md
viewer: false
tags: [ai-mindmeld, latent-interconnect-protocol, protein, chemistry, embeddings]
---

# AI Mindmeld data

[AI Mindmeld](https://github.com/PolymathicAI/AI-Mindmeld) connects scientific specialist models to a frozen large language model through the Latent
Interconnect Protocol (LIP), whose small, independently trained connectors bring specialist representations
directly into the LLM's internal states. This repository supplies the inputs for training and evaluating all
29 release connectors in [sgolkar/ai-mindmeld-connectors](https://huggingface.co/sgolkar/ai-mindmeld-connectors).

It holds 43 full-data bundles and two minimal eight-object evaluation sets (45 bundles, 30.93 GB), plus one small
DEMO bundle for software examples: protein (UniProtKB/Swiss-Prot) and molecule (ChEBI) domain tables, training
text, evaluation items, pooled embeddings at the layers the connectors read, ridge starts, and the Rhea data for the
paper's Rhea results (`rhea-items`, `embeddings-esmc-6b-rhea`, `embeddings-chemfm-3b-rhea`). Not included:
per-residue arrays outside the DEMO bundle, all-layer scans, step-0 checkpoints and one unused ChemFM-3B evaluation embedding set. BindingDB, PFMbench,
ProteinGym and the Neuronpedia SAE explanations are not redistributed; obtain them from their sources.

## Download

Install the release and set `LIP_DATA` as shown in the [Quickstart](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/examples/README.md). These commands use the checkout’s pinned release revisions by default.

```bash
python -m latent_interconnect.release_download --all-data --dry-run   # lists bundles and total size; installs no artifacts
python -m latent_interconnect.release_download --all-data --dest $LIP_DATA
```

Download only what you need with `--data NAME`. The
[inventory](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/release/data_full.md) lists every bundle and
the bundles each connector needs.

## Links

- Code and documentation: https://github.com/PolymathicAI/AI-Mindmeld
- Paper: https://arxiv.org/abs/2610.30215
- [LIP library](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/library/README.md) and
  [ranking details](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/release/library-ranking.json)

## License

The AI Mindmeld authors license the ridge starts and the two minimal evaluation sets under
[CC BY 4.0](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/LICENSE-ARTIFACTS.md). Please credit:

> Siavash Golkar, Jake Kovalic, and Shirley Ho. *AI Mindmeld: A Latent Interconnect Protocol.* arXiv:2610.30215, 2026.
>
> Software: Siavash Golkar, Dennis Bollweg, and the AI Mindmeld contributors, *AI Mindmeld*, https://github.com/PolymathicAI/AI-Mindmeld.

The other bundles, including the embedding bundles and the DEMO bundle, are outside this grant and keep their
source terms. Upstream source, receiver and specialist terms apply throughout. For ChemFM-1B and ChemFM-3B, this release follows the [ChemFM project license](https://github.com/TheLuoFengLab/ChemFM/blob/master/LICENSE), CC BY-NC 4.0; their Hugging Face metadata instead lists MIT. The [PGLM-1B model card](https://huggingface.co/biomap-research/proteinglm-1b-mlm) specifies CC BY-NC 4.0. The supplied MIMIC embeddings support the released training and evaluation paths. Encoding new inputs requires the checkpoint and extraction recipe identified in the bundle. The bundle does not state a public license for that checkpoint; contact the authors for the applicable terms. See the inventory's
[license table](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/release/data_full.md#licenses-and-provenance)
and the [non-commercial items](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/LICENSE-ARTIFACTS.md#non-commercial-items).

## Citation

Use the credit line above: cite the paper first, then the repository, with the software revision you used ([CITATION.cff](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/CITATION.cff)).
