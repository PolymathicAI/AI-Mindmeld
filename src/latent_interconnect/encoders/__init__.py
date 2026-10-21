"""Specialist embeddings as a first-class stage: extraction, storage and loading.

Worker side (imported inside encoder environments; standard library + numpy only):

- ``base``: :class:`RawObject`, :class:`ExtractedBatch`, the :class:`Extractor` protocol and the ``EXTRACTORS``
  registry.
- ``store``: the embedding artifact format (resumable parts, memory-mapped reads).
- ``jobs``: :class:`ExtractionJob`, the JSON wire format, and the worker loop (statically sharded, resumable);
  ``worker``: its command-line entry point.
- ``extractors/``: one module per encoder: ``mimic`` and ``esmc`` (600M and 6B, multi-layer, per-residue)
  now; ``pglm``, ``molformer``, ``unimol2``, ``chemfm`` in the expansion. The receiver's own states are captured by
  ``build.capture``.

Orchestrator side (pydantic, torch):

- ``recipes``: :class:`EncoderRecipe` configs.
- ``loader``: :class:`InputLoader`, store → :class:`~latent_interconnect.writes.inputs.ObjectBatch`.

This ``__init__`` re-exports nothing, so that importing a worker-side module never pulls in pydantic or torch.
"""
