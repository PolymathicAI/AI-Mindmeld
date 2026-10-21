"""Extension path 3: new training text, as a registered stream (``example-corpus``)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from latent_interconnect.config import Config
from latent_interconnect.data.streams import STREAMS, Example, Supervision
from latent_interconnect.data.training_text import TrainingText
from latent_interconnect.domains.views import CatalogView, DomainView
from latent_interconnect.receivers.render import Conversation, Message, WeightedSpan


class CorpusParams(Config):
    """Parameters of the corpus stream.

    Attributes:
        question: the question asked about each object (with a ``<SLOT:x>`` marker).
        field: the screened text field used as the answer.
    """

    question: str
    field: str


@STREAMS.register("example-corpus")
class CorpusStream:
    """Asks one question per object and trains on a screened text field, every answer token at weight 1."""

    Params: ClassVar[type[Config]] = CorpusParams

    def __init__(self, domain: DomainView, params: Config, text: TrainingText) -> None:
        if not isinstance(params, CorpusParams):
            raise TypeError(f"expected CorpusParams, got {type(params).__name__}")
        self.params = params
        self._supervision = self.declare(domain, params)
        self.answers = {o: f[params.field] for o, f in text.fields.items() if params.field in f}
        self.ids = tuple(o for o in text.object_ids if o in self.answers)

    def derived(self) -> dict[str, object]:
        """Return what this stream derived from its training text: nothing."""
        return {}

    def sample(self, rng: np.random.Generator, n: int) -> list[Example]:
        """Draw ``n`` examples."""
        out = []
        for i in rng.integers(0, len(self.ids), size=n):
            oid = self.ids[int(i)]
            answer = self.answers[oid]
            conv = Conversation(messages=(Message(role="user", content=self.params.question),))
            out.append(Example(conv, {"x": oid}, answer, (WeightedSpan(0, len(answer), 1.0),), "example-corpus"))
        return out

    @classmethod
    def declare(cls, catalog: CatalogView, params: Config) -> Supervision:
        """Declare the one field read and the question written (the zero-shot construction check reads it)."""
        del catalog
        if not isinstance(params, CorpusParams):
            raise TypeError(f"expected CorpusParams, got {type(params).__name__}")
        return Supervision(frozenset(), frozenset(), frozenset({params.field}), (params.question,))

    def supervision(self) -> Supervision:
        """Return :meth:`declare`'s declaration."""
        return self._supervision
