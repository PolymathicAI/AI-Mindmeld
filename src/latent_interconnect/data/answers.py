"""The answer convention: how one LLM is asked for, and answers, a Yes/No question.

A convention is one small config (``configs/answers/``) that every config producing or scoring a binary answer cites
explicitly (``answer``): suites (``data.suites.SuiteSpec``), receiver scans
(``evaluation.scan_states.WrittenScan``) and the Yes/No training streams (``yes_no``, ``variant``'s binary kinds, the
``relational`` stream's Yes/No-led families, whose instruction is the stream config's own ``instruction``). The
wording around it (the frame, the template, the grammar row) carries only the task and the object; the convention adds,
through the renderer's ordinary interfaces:

- the **instruction**, appended after one space to the user message (:meth:`AnswerConvention.user`);
- the **prefix**, the assistant turn's text before the answer (``Conversation.prefill``): context, never a weighted
  target;
- the **surfaces**, the answer texts read by a scoring readout and supervised by a training row, each one token
  (``Renderer.token_id``, checked by :meth:`Answering.ids`).

So a training row and a scored prompt of one receiver end on the same tokens: the user turn's instruction, the
generation prompt, the prefix, then the answer token. The instruction asks for a direct answer, so the reasoning route
omits it: a scaffold row (``data.scaffold``), a trace's prompt (a ``generation`` suite takes no convention) and
its post-reasoning readout keep only their own reason-then-answer instruction, and after the reasoning its close (the
receiver's thinking close or separator, ``ReceiverSpec``) comes first, then the same prefix and answer.

A config without ``answer`` keeps its inline policy (the instruction inside its frame or template, the answer words in
its own fields), as Kimi K2.6's other configs do, each route with its own wording (they are not
migrated); configs refuse to state both. Kimi's one convention file (``configs/answers/kimi-k2.6.yaml``,
whose header names the record of its inline routes) is cited by its relational streams and the mutation
suites but the binary seen ones, which ask in the variant training stream's own wording.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import Field, model_validator

from latent_interconnect.config import Config

__all__ = ["AnswerConvention", "Answering", "answering"]


class AnswerConvention(Config):
    """One LLM's Yes/No answer format (the module docstring; :class:`Answering` applies it).

    Attributes:
        instruction: the answer instruction, appended after one space to the user message (Gemma: ``"Answer
            with only Yes or No."``).
        prefix: the assistant turn's text before the answer (Gemma: ``"Answer:"``); ``""`` for none, the answer
            then opening the turn.
        yes: the positive answer's text, one token (after ``"Answer:"``: ``" Yes"``).
        no: the negative answer's text, one token.
    """

    instruction: str = Field(min_length=1)
    prefix: str = ""
    yes: str = Field(min_length=1)
    no: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> AnswerConvention:
        if self.yes == self.no:
            raise ValueError(f"the yes and no answers must differ, both are {self.yes!r}")
        return self

    def policy(self) -> Answering:
        """Return the policy that applies this convention."""
        return Answering(self.instruction, self.prefix, self.yes, self.no, convention=True)


@dataclass(frozen=True)
class Answering:
    """The answer policy one consumer applies: its convention's, or its config's inline one (:func:`answering`).

    Attributes:
        instruction: the instruction appended to the user message; ``""`` when the wording carries its own (inline).
        prefix: the assistant turn's text before the answer (``""`` for none).
        yes: the positive answer's text.
        no: the negative answer's text.
        convention: whether it comes from an :class:`AnswerConvention`.
    """

    instruction: str
    prefix: str
    yes: str
    no: str
    convention: bool

    def user(self, wording: str) -> str:
        """Return the user message: the task and object ``wording``, then the instruction after one space."""
        return f"{wording} {self.instruction}" if self.instruction else wording

    def prefill(self) -> str | None:
        """Return the assistant turn's text before the answer, or None for none."""
        return self.prefix or None

    def word(self, label: bool) -> str:
        """Return the answer text of ``label``."""
        return self.yes if label else self.no

    def ids(self, token_id: Callable[[str], int]) -> tuple[int, int]:
        """Return the (yes, no) token ids through ``token_id`` (``Renderer.token_id``: each exactly one token).

        Raises:
            ValueError: if an answer is not exactly one token, or both are the same token.
        """
        ids = (token_id(self.yes), token_id(self.no))
        if ids[0] == ids[1]:
            raise ValueError(f"answers {self.yes!r} and {self.no!r} are the same token {ids[0]}")
        return ids


def answering(
    convention: AnswerConvention | None, *, yes: str | None, no: str | None, prefix: str = "", where: str
) -> Answering:
    """Return the policy of a config that cites ``convention`` or states inline words, exactly one of the two.

    Args:
        convention: the config's ``answer``, or None.
        yes: its inline positive answer, or None.
        no: its inline negative answer, or None.
        prefix: its inline prefill ("" for none).
        where: the config, named in a refusal.

    Raises:
        ValueError: if the config gives both a convention and inline words or prefill, or neither.
    """
    if convention is not None:
        if yes is not None or no is not None or prefix:
            raise ValueError(f"{where}: cites an answer convention and states inline answers {yes!r}/{no!r}/{prefix!r}")
        return convention.policy()
    if yes is None or no is None:
        raise ValueError(f"{where}: neither an answer convention nor inline yes and no answers")
    return Answering("", prefix, yes, no, convention=False)
