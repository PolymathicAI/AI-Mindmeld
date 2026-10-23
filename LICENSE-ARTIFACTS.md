# Connector weights and selected data

The AI Mindmeld authors license their contributions to the following released artifacts under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):

- AI Mindmeld connector weights, including the software DEMO connectors;
- the two minimal evaluation sets, `protein-esmc-600m-mini` and `molecule-chemfm-1b-mini`;
- the ridge starts listed in the [release data inventory](release/data_full.md#inputs-for-training-evaluation-and-inference).

The [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en) governs this grant,
including its attribution requirements and disclaimer of warranties. When sharing these artifacts or
adaptations, retain the supplied attribution and notices, link the license, and indicate changes.

## Attribution

Please use the following credit, with the paper first and repository second:

> Siavash Golkar, Jake Kovalic, and Shirley Ho. *AI Mindmeld: A Latent Interconnect Protocol.*
> arXiv:2610.30215, 2026. https://arxiv.org/abs/2610.30215
>
> Software: Siavash Golkar, Dennis Bollweg, and the AI Mindmeld contributors, *AI Mindmeld*,
> https://github.com/PolymathicAI/AI-Mindmeld.

Include the artifact name and version or download URL, and identify any modifications. See
[CITATION.cff](CITATION.cff) for machine-readable paper and software citations.

## Scope and upstream terms

This grant applies only to rights the AI Mindmeld authors can license. It does not replace the terms of
upstream LLMs, specialist models, or source data, including any non-commercial restrictions.
Retain their attribution and license notices and consult the model revisions and source provenance in each
bundle.

The grant covers the two minimal evaluation sets as packaged, including the ESM-C 600M and ChemFM-1B embeddings
they contain; the specialist's terms on those embedded vectors still apply on top. Other full datasets, including
the separately released embedding data bundles (`embeddings-*`), the separate residue DEMO data bundle
`protein-esmc-600m-residues-mini-demo`, and third-party community connectors are outside this grant and keep
their own terms. Those terms are described in the
[data inventory](release/data_full.md#licenses-and-provenance) and their own cards. Project code remains
[Apache-2.0](LICENSE); see [NOTICE](NOTICE) for attribution.

## Non-commercial items

These items are built on specialists with non-commercial terms. Those terms apply on top of our CC BY 4.0 grant
for the connectors, ridge starts and minimal evaluation set below, and they govern the embedding data bundles
below, which are outside that grant:

- **ChemFM** (CC BY-NC 4.0, per the [LICENSE](https://github.com/TheLuoFengLab/ChemFM/blob/master/LICENSE) of
  github.com/TheLuoFengLab/ChemFM): the connectors `*__chemfm-1b` and `*__chemfm-3b*` (including
  `kimi-k2.6__chemfm-3b-mlp4`), their ridge starts `ridge-*__chemfm-1b` and `ridge-*__chemfm-3b`, the minimal
  evaluation set `molecule-chemfm-1b-mini`, and the embedding data bundles `embeddings-chemfm-1b` and
  `embeddings-chemfm-3b`.
- **PGLM-1B** (Creative Commons Attribution-NonCommercial 4.0, per its model card): the connectors `*__pglm-1b`,
  their ridge starts `ridge-*__pglm-1b`, and the embedding data bundle `embeddings-pglm-1b`.
