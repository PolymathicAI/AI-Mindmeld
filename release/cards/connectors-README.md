---
license: other
license_name: ai-mindmeld-artifacts
license_link: https://github.com/PolymathicAI/AI-Mindmeld/blob/main/LICENSE-ARTIFACTS.md
tags: [ai-mindmeld, latent-interconnect-protocol, protein, chemistry, multimodal]
---

# AI Mindmeld connectors

[AI Mindmeld](https://github.com/PolymathicAI/AI-Mindmeld) connects scientific specialist models to a large language model through the Latent Interconnect
Protocol (LIP). Its connectors are learned mappings that bring a specialist's representation of an object
directly into the LLM's hidden states at a slot in the prompt, while both models stay frozen; independently
trained connectors can be combined in one prompt without joint training.

This repository holds the 29 trained release connectors, plus two DEMO connectors for software examples.
They connect three protein specialist families (ESM-C 600M and 6B, PGLM-1B and MIMIC) and three molecule specialist families
(ChemFM-1B and 3B, MoLFormer-c3 and Uni-Mol2 1.1B) to Gemma 3 4B (8 connectors), Gemma 3 12B (10) and
Kimi K2.6 (11). Each connector is specific to its specialist, extraction recipe and LLM; specialist and LLM
weights are separate downloads.

Four of them (`*-all-tasks`) are trained on all tasks, so they have no zero-shot evaluation.

## Download

Install the release and set `LIP_DATA` as shown in the [Quickstart](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/examples/README.md). These commands use the checkout’s pinned release revisions by default.

```bash
python -m latent_interconnect.release_download \
  --connector gemma-3-4b-it__esmc-600m --connector gemma-3-4b-it__chemfm-1b \
  --data protein-esmc-600m-mini --data molecule-chemfm-1b-mini \
  --dest $LIP_DATA
```

Use `python -m latent_interconnect.release_download --list --kinds` to see bundle names, sizes, kinds, and DEMO
markers. The [data inventory](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/release/data_full.md) lists the data each connector needs.

## Links

- Code and documentation: https://github.com/PolymathicAI/AI-Mindmeld
- Paper: https://arxiv.org/abs/2610.30215
- [LIP library](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/library/README.md) (connector descriptions
  and scores) and [ranking details](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/release/library-ranking.json)
- Training data: [sgolkar/ai-mindmeld-data](https://huggingface.co/datasets/sgolkar/ai-mindmeld-data)

## License

The AI Mindmeld authors license the connector weights, including the DEMO connectors, under
[CC BY 4.0](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/LICENSE-ARTIFACTS.md). Please credit:

> Siavash Golkar, Jake Kovalic, and Shirley Ho. *AI Mindmeld: A Latent Interconnect Protocol.* arXiv:2610.30215, 2026.
>
> Software: Siavash Golkar, Dennis Bollweg, and the AI Mindmeld contributors, *AI Mindmeld*, https://github.com/PolymathicAI/AI-Mindmeld.

The grant covers only rights the authors can license. Upstream terms of the receiver LLMs and specialist models
apply on top. For ChemFM-1B and ChemFM-3B, this release follows the [ChemFM project license](https://github.com/TheLuoFengLab/ChemFM/blob/master/LICENSE), CC BY-NC 4.0; their Hugging Face metadata instead lists MIT. The [PGLM-1B model card](https://huggingface.co/biomap-research/proteinglm-1b-mlm) specifies CC BY-NC 4.0. See the
[list of non-commercial items](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/LICENSE-ARTIFACTS.md#non-commercial-items)
and the inventory's [license notes](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/release/data_full.md#licenses-and-provenance). The code is Apache-2.0.

## Citation

Use the credit line above: cite the paper first, then the repository, with the software revision you used ([CITATION.cff](https://github.com/PolymathicAI/AI-Mindmeld/blob/main/CITATION.cff)).
