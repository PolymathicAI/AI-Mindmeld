"""Token provenance of a committed capture, derived again on the CPU: which token each stored state was read at.

A capture artifact holds states, not text: no rendered prompt, token id or read span. :func:`token_provenance` renders
the capture's prompts again with the one renderer and the receiver's tokenizer (``build.capture.capture_prompts``; no
receiver weights are loaded, no forward pass runs) and reports, for the objects asked about, the prompt length, the
read span, the read position and the token there. It is a **new derived output**, not evidence stored by the capture:
it stands only as far as the rendering reproduces the capture, which is reported, never assumed, by three comparisons
with the capture's own metadata: the capture spec's digest, the objects it kept (and the count it left out), and the
digest of its micro-batch partition (a function of every prompt's token count). A mismatch is reported in the result
(the ``recorded_*`` fields beside the derived ones); nothing is substituted.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.build.capture import CaptureSpec, capture_prompts, context_cuts
from latent_interconnect.config import Config
from latent_interconnect.encoders.store import EmbeddingReader
from latent_interconnect.receivers.kernel import partition_digest, plan_batches, planned_lengths
from latent_interconnect.receivers.render import Renderer

__all__ = ["TokenProvenance", "token_provenance"]


class TokenProvenance(Config):
    """Where a capture's states were read, re-derived (module docstring); per-object columns are in the asked order.

    Attributes:
        spec: the digest of the capture spec that was rendered again.
        recorded_spec: the spec digest in the capture artifact's metadata.
        objects_match: whether the objects the rendering keeps are exactly the objects the artifact holds.
        left_out: the objects the rendering leaves out (no context text).
        recorded_left_out: the count in the artifact's metadata.
        partition: the digest of the rendered prompts' micro-batch partition under the capture's batch plan.
        recorded_partition: the partition digest in the artifact's metadata.
        prompt_tokens: each object's prompt length in tokens.
        read_start: the first token position of its read span (the read text's tokens).
        read_end: the last token position of its read span.
        position: the position whose state the capture's ``last`` target is.
        token_id: the token id at that position.
        token: that token, decoded.
        context_cut: whether the context was cut to fit its token cap (empty for a recipe without a context).
    """

    spec: str
    recorded_spec: str
    objects_match: bool
    left_out: int = Field(ge=0)
    recorded_left_out: int = Field(ge=0)
    partition: str
    recorded_partition: str
    prompt_tokens: tuple[int, ...]
    read_start: tuple[int, ...]
    read_end: tuple[int, ...]
    position: tuple[int, ...]
    token_id: tuple[int, ...]
    token: tuple[str, ...]
    context_cut: tuple[bool, ...]


def token_provenance(
    spec: CaptureSpec, store: ArtifactStore, renderer: Renderer, reader: EmbeddingReader, objects: Sequence[str]
) -> TokenProvenance:
    """Render the capture's prompts again and return the token provenance of ``objects``.

    Args:
        spec: the capture's spec (its record's config).
        store: the artifact store (the domain, the object list, the training text).
        renderer: the receiver's renderer (``receivers.open_renderer``).
        reader: the capture artifact.
        objects: the objects to report, each held by the rendering.

    Raises:
        KeyError: if an object is not among the rendered ones.
        ValueError: as ``build.capture.capture_prompts``, or if a prompt has no read position.
    """
    ids, prompts, left_out = capture_prompts(spec, store, renderer)
    partition = plan_batches(planned_lengths(prompts, "capture"), spec.plan)
    row = {oid: i for i, oid in enumerate(ids)}
    chosen = [prompts[row[oid]] for oid in objects]
    if any(p.capture_position is None or not p.capture_span for p in chosen):
        raise ValueError(f"capture recipe {spec.recipe!r}: a rendered prompt has no read position")
    positions = [int(p.capture_position or 0) for p in chosen]
    token_ids = [p.input_ids[at] for p, at in zip(chosen, positions, strict=True)]
    cuts = context_cuts(spec, store, renderer, objects)
    plan = reader.meta.get("batch_plan") or {}
    return TokenProvenance(
        spec=spec.digest(),
        recorded_spec=str(reader.meta.get("spec", "")),
        objects_match=set(ids) == set(reader.object_ids) and len(ids) == len(reader),
        left_out=left_out,
        recorded_left_out=int(reader.meta.get("left_out", 0)),
        partition=partition_digest(partition),
        recorded_partition=str(plan.get("partition", "")),
        prompt_tokens=tuple(len(p.input_ids) for p in chosen),
        read_start=tuple(p.capture_span[0] for p in chosen),
        read_end=tuple(p.capture_span[-1] for p in chosen),
        position=tuple(positions),
        token_id=tuple(token_ids),
        token=tuple(renderer.tokenizer.decode([t]) for t in token_ids),
        context_cut=tuple(cuts[oid] for oid in objects) if cuts else (),
    )
