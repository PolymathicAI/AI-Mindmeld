"""Latent Interconnect Protocol (LIP): connectors that let a frozen language model read specialist embeddings.

A *connector* maps an object's specialist embedding (a protein from ESM-C, a compound from MoLFormer, ...) to writes
into a frozen *receiver* language model's residual stream, at the positions of a slot in the prompt. The receiver is
then asked about the object in natural language.

Subpackages, in dependency order (a package imports only from packages above it; a convention
the shipped tests do not check):

- ``canonical``, ``config``, ``plugins``, ``paths``, ``provenance``, ``artifacts``: kernels shared by everything.
- ``stats``: AUROC, the one bootstrap scheme and its intervals (estimates, never judgements).
- ``writes``: the write rule, write layers, map kinds, connectors, write sources and checkpoints.
- ``receivers``: the receiver protocols, the renderer and the generic Hugging Face receiver. Model-specific efficient
  backends (``receivers.kimi_k26``) are plugins and are never imported by core code.
- ``encoders``: specialist embedding extraction and the content-addressed embedding store.
- ``domains``: object domains under one contract.
- ``data``: item sets, streams, the sampler, suite specs, the screened training text.
- ``build``: construction stages (receiver-state capture, ridge fits, the zero-shot-screened training set).
- ``training``: the run config and the one trainer.
- ``evaluation``: suite runners, prompt vetting, analyses (their records end the pipeline), layer-selection
  tools.

How to extend each part and run the stages: ``examples/guides/extending.md`` in the repository.
"""

__version__ = "1.0.0"
