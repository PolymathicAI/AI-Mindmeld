"""The training-data sampler: mixture of streams → rendered prompts, deterministic per step.

It lives below both ``training`` (which consumes its batches) and ``evaluation`` (which may replay it), so neither
depends on the other. Streams draw their text from the screened training set (:class:`TrainingText`), so the sampler
needs no screen of its own. The batch size is a fixed number in the run config; measuring what fits on a GPU is a
separate tool whose result is written into the config.

**Step-indexed draws.** Every draw of step ``s`` comes from generators seeded by ``(seed, s)`` alone, so step ``s``'s
examples are the same in an uninterrupted run, a resumed one and a CPU replay, with no generator state carried across
steps. Within a step, three independent generators:

1. ``(seed, s, 0)``: each example's stream, drawn by share (the original sampler's per-example stream draw);
2. ``(seed, s, 1, j)``: stream ``j``'s own examples (``Stream.sample``), so one stream's draws never move another's;
3. ``(seed, s, 2)``: the reasoning scaffold, attempted on each eligible row of a stream that carries one
   (``data.scaffold``) with probability ``ScaffoldSpec.rate``, so scaffolding never moves a stream's draws.

4. ``(seed, s, 3)``: the in-context examples (``data.context``), given to each eligible row of a stream that carries
   them with probability ``ContextSpec.rate``, before the scaffold.

A stream opts into the row transforms through attributes: ``scaffold`` (a ``ScaffoldSpec`` or None), ``context`` (a
``ContextSpec`` or None), ``context_yes_no`` (a Yes/No stream's questions per task and answer words, or None) and
``context_cue`` (its answer convention's prefix, ``data.context.Context`` ``cue``), ``answer_instruction`` (its answer
convention's instruction, which a scaffolded row drops: ``data.scaffold.Scaffold``); a stream with a ``bind`` method is
given the renderer's token offsets before its first draw. A row is eligible when it names its subject slot
(``Example.subject``). Examples keep the order of step 1. Objects are drawn by the streams independently per example
(with replacement across a step and across steps), not as epochs of a permutation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import Field

from latent_interconnect.config import Config
from latent_interconnect.data.context import Context, ContextSpec
from latent_interconnect.data.scaffold import Scaffold, ScaffoldSpec
from latent_interconnect.data.streams import Example, Stream, boosted, build_stream
from latent_interconnect.data.training_text import TrainingText
from latent_interconnect.domains.base import Domain
from latent_interconnect.receivers.render import RenderedPrompt, Renderer, SlotBinding, WeightedSpan

__all__ = ["MixtureEntry", "Sampler"]

_STREAM, _EXAMPLES, _SCAFFOLD, _CONTEXT = 0, 1, 2, 3
"""The sub-seeds of a step's three generators (module docstring)."""


class MixtureEntry(Config):
    """One stream in the training mixture.

    Attributes:
        stream: registered stream name.
        share: fraction of examples drawn from it (shares sum to 1).
        params: stream parameters, validated by the stream's ``Params``.
    """

    stream: str
    share: float = Field(gt=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


class Sampler:
    """Draws and renders the examples of each step.

    Args:
        mixture: the stream mixture.
        domain: the domain at the run's data version (streams are built on it).
        renderer: the receiver's renderer.
        batch: examples per step.
        seed: data seed.
        text: the screened training set streams draw from.
        connector: the connector argument every slot is bound to (the run trains one connector).
        k: that connector's slot-token count.
        thinking_close: the receiver's end-of-thinking text (``ReceiverSpec.thinking_close``), or None for a receiver
            without a thinking mode: it closes a reasoning scaffold (``data.scaffold.Scaffold``).
        separator: the receiver's separator (``ReceiverSpec.separator``): the scaffold's close without a thinking close.

    Raises:
        ValueError: if a stream has a scaffold and the receiver neither a thinking close nor a separator
            (``data.scaffold.Scaffold``).
    """

    def __init__(
        self,
        *,
        mixture: Sequence[MixtureEntry],
        domain: Domain,
        renderer: Renderer,
        batch: int,
        seed: int,
        text: TrainingText,
        connector: str,
        k: int,
        thinking_close: str | None,
        separator: str | None,
    ) -> None:
        self.mixture, self.domain, self.renderer = tuple(mixture), domain, renderer
        self.batch, self.seed = batch, seed
        self.text = text
        self.connector, self.k = connector, k
        self.streams: list[Stream] = []
        self.scaffolds: list[Scaffold | None] = []
        self.contexts: list[Context | None] = []
        for m in self.mixture:
            built = build_stream(m.stream, m.params, domain, text)
            stream, view = built.stream, built.text
            bind = getattr(stream, "bind", None)
            if callable(bind):  # a stream that sizes its answers in receiver tokens (data.relational_stream)
                bind(renderer.target_token_offsets)
            spec = getattr(stream, "scaffold", None)
            scaffold = None
            if isinstance(spec, ScaffoldSpec):
                scaffold = Scaffold(
                    spec,
                    built.domain,
                    view,
                    renderer.target_token_count,
                    thinking_close,
                    separator=separator,
                    instruction=getattr(stream, "answer_instruction", ""),
                )
            context = getattr(stream, "context", None)
            self.contexts.append(
                Context(
                    context,
                    built.domain,
                    view,
                    renderer.target_token_count,
                    getattr(stream, "context_yes_no", None),
                    grammar=getattr(stream, "grammar", None),
                    cue=getattr(stream, "context_cue", ""),
                )
                if isinstance(context, ContextSpec)
                else None
            )
            self.streams.append(stream)
            self.scaffolds.append(scaffold)
        shares = np.array([m.share for m in self.mixture], dtype=np.float64)
        self._shares = shares / shares.sum()

    def objects(self) -> frozenset[str]:
        """Return every object an example can bind.

        The training text's objects and, from a stream with a ``bound_objects`` method, the objects it binds beyond
        them (the variant stream's wild-type parents).
        """
        out = set(self.text.object_ids)
        for stream in self.streams:
            bound = getattr(stream, "bound_objects", None)
            if callable(bound):
                out |= bound()
        return frozenset(out)

    def derived(self) -> dict[str, dict[str, object]]:
        """Return what each mixture entry derived from the training text (grammar formats, give-aways).

        Keyed ``"<index>:<stream>"``, from each stream's ``derived`` method (part of ``Stream``) and its scaffold's. A
        deterministic function of the run config, its pinned training set and the code: the trainer writes it into the
        run directory when the run starts, refuses to continue a run that would now derive anything else, and the train
        stage commits that file.
        """
        out = {}
        for i, (m, stream, scaffold) in enumerate(zip(self.mixture, self.streams, self.scaffolds, strict=True)):
            found = dict(stream.derived())
            if scaffold is not None:
                found |= scaffold.derived()
            if found:
                out[f"{i}:{m.stream}"] = found
        return out

    def stream_choice(self, step: int) -> np.ndarray:
        """Return each row's mixture entry (index into the mixture) at ``step``: the draw ``examples`` uses."""
        return np.random.default_rng([self.seed, step, _STREAM]).choice(len(self.streams), self.batch, p=self._shares)

    def examples(self, step: int) -> list[Example]:
        """Return the examples of ``step`` (before rendering)."""
        which = self.stream_choice(step)
        drawn = {
            j: iter(stream.sample(np.random.default_rng([self.seed, step, _EXAMPLES, j]), int((which == j).sum())))
            for j, stream in enumerate(self.streams)
        }
        srng = np.random.default_rng([self.seed, step, _SCAFFOLD])
        crng = np.random.default_rng([self.seed, step, _CONTEXT])
        out = []
        for j in which:
            example = next(drawn[int(j)])
            context = self.contexts[int(j)]
            if context is not None and example.subject is not None and crng.random() < context.spec.rate:
                example = context.apply(example, crng)
            scaffold = self.scaffolds[int(j)]
            if scaffold is not None and scaffold.eligible(example) and srng.random() < scaffold.spec.rate:
                example = scaffold.apply(example, srng) or example
            out.append(example)
        return out

    def prompts(self, step: int) -> list[RenderedPrompt]:
        """Return the rendered prompts of ``step``."""
        return [self.render(e) for e in self.examples(step)]

    def render(self, example: Example) -> RenderedPrompt:
        """Render one example: every slot bound to the run's connector, the target in its parts.

        An answer with a segment boost gets its per-token weights here, from the renderer's token offsets
        (``data.streams.boosted``): its spans' weights, the heads raised, as one span with a per-token curve.
        """
        bindings = {slot: SlotBinding(oid, self.connector, self.k) for slot, oid in example.objects.items()}
        if example.boost is not None:
            example = self._boost(example)
        return self.renderer.render(example.conversation, bindings, target=list(example.parts()))

    def _boost(self, example: Example) -> Example:
        if len(example.spans) != 1 or example.spans[0].start != 0 or example.spans[0].end != len(example.target):
            raise ValueError("a boosted answer has one span over all of it (a text stream's answer)")
        offsets = self.renderer.target_token_offsets(example.target)[: example.max_target_tokens]
        span = example.spans[0]
        curve = span.curve or (span.weight,)
        weights = [curve[min(i, len(curve) - 1)] for i in range(len(offsets))]
        boost = example.boost
        if boost is None or not weights:
            return example
        new = boosted(example.target, offsets, weights, boost)
        return dataclasses.replace(example, spans=(WeightedSpan(0, len(example.target), new[0], new),), boost=None)
