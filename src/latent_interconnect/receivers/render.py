"""The one renderer: conversation + slot bindings → token ids, slot positions, targets (``lip-render/1``).

Every path that builds a prompt (training streams, every suite, generation, state capture) calls :class:`Renderer`.
Lesson: the original code rendered prompts in seven places.

Slot markers in message text are ``<SLOT:name>``. Each becomes k copies of the receiver's placeholder token (k = the
bound connector's token count), and one space directly before a marker is removed, so ``"the protein <SLOT:p>"`` and
``"the protein<SLOT:p>"`` render alike. A marker may instead be bound to a supplied text (an object's name in an
evaluation's name condition, a capture recipe's ``{name}``): the text is then its own tokenization piece, the
counterpart of the slot token. The piece is ``" " + text`` when text of its message precedes it, as
the original code encoded ``"...embedding:" + " " + name``, and the bare text when it opens its message or the prefill.
The renderer adds no words but the receiver's own: its chat template (the
tokenizer's own at the pinned revision, called with ``ReceiverSpec.template_kwargs`` for the conversation's flags, e.g.
thinking) and, in thinking mode, its thinking request (``ReceiverSpec.thinking_request``). Target weights are given as
character spans of the target text with a weight per span; the renderer maps them onto tokens by decoding token prefixes
(not every tokenizer has offset maps; Kimi's has none), so no other module tokenizes.

The algorithm (``lip-render/1``):

0. in thinking mode, append the receiver's thinking request, after one space, to the last user message (of a copy:
   the conversation is unchanged);
1. apply the chat template to the messages as text, with ``add_generation_prompt=True`` and the receiver's arguments
   for each conversation flag in its stated state (``ReceiverSpec.template_kwargs``), then append the prefill's
   leading text (its text parts before the first token-id part);
2. split the text at the markers (removing one space before each);
3. encode each piece separately, without special tokens, and insert k placeholder ids per bound marker, or the
   separately encoded text of a text marker (with one leading space unless it opens its message or the prefill);
4. append the rest of the prefill part by part: a token-id part (:class:`TokenIds`, e.g. a generated trace) as its ids,
   verbatim, never re-tokenized; a text part encoded on its own, without markers;
5. encode the target separately, part by part (each cut at its own cap), and append it after the prompt. A target
   never includes an end-of-turn token.

**Token-id prefill.** A readout that continues a generated trace (a scoring suite reading a generation record,
``evaluation.suites``) puts the trace's token ids in the prefill as a :class:`TokenIds` part, so what is scored is
exactly what was generated: decoding and re-encoding a trace need not give back its ids.

A capture prompt (``capture`` or ``read``) has no target: its positions index the rendered ids. A capture recipe
(``domains.base.CaptureRecipe``) is rendered by :meth:`Renderer.render_recipe`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, Protocol

from latent_interconnect.receivers._spans import (
    SLOT_PATTERN,
    Boundaries,
    check_markers,
    locate,
    opening_markers,
    span_weights,
    split_markers,
)

__all__ = [
    "FLAGS",
    "SLOT_PATTERN",
    "Conversation",
    "Message",
    "RecipeText",
    "RenderedPrompt",
    "Renderer",
    "SlotBinding",
    "SlotOccurrence",
    "Target",
    "TargetPart",
    "TokenIds",
    "Tokenizer",
    "WeightedSpan",
]

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_NO_TEXTS: Mapping[str, str] = MappingProxyType({})
FLAGS = ("thinking",)
"""The conversation flags (:class:`Conversation` fields) that select chat-template arguments."""


class RecipeText(Protocol):
    """One recipe message, structurally (``domains.base.RecipeMessage``; ``receivers`` sits below ``domains``)."""

    @property
    def role(self) -> Literal["system", "user", "assistant"]:
        """The message's role."""
        ...

    @property
    def text(self) -> str:
        """Its text, with ``{key}`` placeholders."""
        ...


class Tokenizer(Protocol):
    """The part of a Hugging Face tokenizer the renderer uses."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        """Encode text to ids."""
        ...

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: object) -> str | list[int]:
        """Render a conversation with the model's chat template."""
        ...

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = ..., spaces_between_special_tokens: bool = ...
    ) -> str:
        """Decode ids to text (spans are mapped by prefix decoding, since not every tokenizer has offset maps)."""
        ...


@dataclass(frozen=True)
class Message:
    """One chat message.

    Attributes:
        role: ``system``, ``user`` or ``assistant``.
        content: the text; it may contain ``<SLOT:name>`` markers.
    """

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class TokenIds:
    """Token ids that enter a prefill verbatim (e.g. a generated trace), never decoded and re-encoded.

    The part may be empty (a trace that closed its thinking at once): it still ends the prefill's leading text, so the
    parts after it are encoded as their own pieces whatever the trace's length.

    Attributes:
        ids: the ids, each a non-negative integer.

    Raises:
        ValueError: if ``ids`` holds anything but non-negative ``int`` values.
    """

    ids: tuple[int, ...]

    def __post_init__(self) -> None:
        bad = [i for i in self.ids if type(i) is not int or i < 0]
        if bad:
            raise ValueError(f"token ids must be non-negative ints, got {bad[:5]}")


@dataclass(frozen=True)
class Conversation:
    """A prompt before rendering.

    Attributes:
        messages: the chat messages.
        prefill: what the assistant turn starts with, or None: a text (e.g. a reasoning prefix), or parts, each a text
            or :class:`TokenIds`. Text parts before the first token-id part are the prefill's leading text, appended
            to the template text (they may hold markers); every later part is its own piece (a token-id part verbatim,
            a text part encoded alone, without markers).
        thinking: whether the receiver's thinking mode is on (receiver-specific template flag).
    """

    messages: tuple[Message, ...]
    prefill: str | tuple[str | TokenIds, ...] | None = None
    thinking: bool = False

    def prefill_parts(self) -> tuple[str, tuple[str | TokenIds, ...]]:
        """Return the prefill's leading text and its parts from the first token-id part on (both empty if none)."""
        if self.prefill is None or isinstance(self.prefill, str):
            return self.prefill or "", ()
        first = next((i for i, part in enumerate(self.prefill) if isinstance(part, TokenIds)), len(self.prefill))
        lead = self.prefill[:first]
        return "".join(part for part in lead if isinstance(part, str)), self.prefill[first:]


@dataclass(frozen=True)
class SlotBinding:
    """What fills one slot: an object, through a connector with k tokens.

    Attributes:
        object_id: the object.
        connector: the connector argument that writes it (a key of the write sources).
        k: the connector's slot-token count (the slot renders as k placeholder tokens).
    """

    object_id: str
    connector: str
    k: int


@dataclass(frozen=True)
class SlotOccurrence:
    """Where a bound slot landed in the rendered ids: k consecutive positions.

    Attributes:
        slot: the marker name.
        binding: what fills it.
        positions: its k token positions.
    """

    slot: str
    binding: SlotBinding
    positions: tuple[int, ...]


@dataclass(frozen=True)
class Target:
    """A supervised continuation: token ids after the prompt, with a loss weight per token.

    A target never includes an end-of-turn token.

    Attributes:
        token_ids: the continuation's ids.
        weights: one loss weight per id.
    """

    token_ids: tuple[int, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class RenderedPrompt:
    """A rendered prompt.

    Attributes:
        input_ids: prompt ids (targets excluded).
        slots: every slot occurrence, in order of position.
        target: the supervised continuation, if any (training, NLL readouts).
        capture_position: position whose hidden state a capture pass reads (e.g. the last name token), if any. A
            capture prompt has no target, so it indexes ``input_ids``.
        capture_span: every token position of the read text, in order, read by a span capture (e.g. the mean over a
            name's tokens, the ridge bias); empty without one.
    """

    input_ids: tuple[int, ...]
    slots: tuple[SlotOccurrence, ...]
    target: Target | None = None
    capture_position: int | None = None
    capture_span: tuple[int, ...] = ()


@dataclass(frozen=True)
class TargetPart:
    """One part of a target, encoded on its own with its own loss weights and token cap.

    E.g. a reasoning scaffold, the receiver's thinking close (``ReceiverSpec.thinking_close``), then the answer.

    Attributes:
        text: the part's text.
        spans: loss weights over ``text`` (tokens outside every span get weight 0).
        max_tokens: cap on this part's tokens: a longer part is cut from its end, the other parts untouched (no
            end-of-turn token either way); None for no cap.
    """

    text: str
    spans: tuple[WeightedSpan, ...] = ()
    max_tokens: int | None = None


@dataclass(frozen=True)
class WeightedSpan:
    """A character span ``[start, end)`` of a target text and the loss weights of its tokens.

    Attributes:
        start: first character.
        end: one past the last character.
        weight: the weight of every token in the span when ``curve`` is empty.
        curve: per-token weights along the span (positional decay): token i of the span gets ``curve[i]``, and
            tokens past the curve's end get its last value.
    """

    start: int
    end: int
    weight: float
    curve: tuple[float, ...] = ()


class Renderer:
    """Renders conversations for one receiver.

    Args:
        tokenizer: the receiver's tokenizer.
        placeholder: the receiver's slot-token text (``ReceiverSpec.placeholder``).
        template_kwargs: ``ReceiverSpec.template_kwargs`` (flag → on / off → arguments).
        thinking_request: ``ReceiverSpec.thinking_request``, appended to the last user message in thinking mode.

    Raises:
        ValueError: if the placeholder does not encode to exactly one token id, or ``template_kwargs`` names a flag
            :class:`Conversation` does not have (:data:`FLAGS`).
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        placeholder: str,
        template_kwargs: Mapping[str, Mapping[Literal["on", "off"], Mapping[str, object]]],
        thinking_request: str | None = None,
    ) -> None:
        unknown = sorted(set(template_kwargs) - set(FLAGS))
        if unknown:
            raise ValueError(f"template arguments for unknown conversation flags {unknown}; flags are {FLAGS}")
        self.tokenizer = tokenizer
        self.placeholder = placeholder
        self.template_kwargs = template_kwargs
        self.thinking_request = thinking_request
        self.placeholder_id: int = self.token_id(placeholder)

    def token_id(self, text: str) -> int:
        """Return the one token id ``text`` encodes to (e.g. the ``"Yes"`` / ``"No"`` answer ids).

        Raises:
            ValueError: if ``text`` does not encode to exactly one token.
        """
        ids = self._encode_part(text)
        if len(ids) != 1:
            raise ValueError(f"{text!r} encodes to {len(ids)} tokens {ids}, not one")
        return ids[0]

    def render(
        self,
        conversation: Conversation,
        bindings: Mapping[str, SlotBinding],
        *,
        target: str | Sequence[TargetPart] | None = None,
        spans: Sequence[WeightedSpan] = (),
        max_target_tokens: int | None = None,
        capture: tuple[str, int] | None = None,
        texts: Mapping[str, str] = _NO_TEXTS,
        read: str | None = None,
    ) -> RenderedPrompt:
        """Render one prompt.

        Args:
            conversation: the messages.
            bindings: slot name → binding, for every marker not in ``texts``.
            target: the supervised continuation, if any: a string, or ordered target parts (:class:`TargetPart`), each
                encoded on its own with its own spans and cap, then concatenated. A string is the one-part case
                ``TargetPart(target, spans, max_target_tokens)``; with parts, ``spans`` and ``max_target_tokens`` stay
                unset.
            spans: loss weights over a string ``target`` (tokens outside every span get weight 0). A token belongs to
                every span it overlaps by at least one character and takes the weight of the first; spans may not
                overlap.
            max_target_tokens: cap on the target's tokens: a longer target is cut to it (no end-of-turn token
                either way).
            capture: ``(message text fragment, offset)`` locating the capture position: the fragment must occur
                exactly once in the rendered text, and the position is the token holding its character at
                ``offset`` (e.g. the last character of the object's name; negative offsets count from its end).
            texts: marker name → supplied non-empty text, rendered as its own piece (no placeholder, no write):
                ``" " + text`` after any text of its message (a newline included), the bare text when it opens its
                message or the prefill. A content that starts with one space and the marker counts as opening: the
                marker removes that space.
            read: a text marker that occurs once; the result's ``capture_span`` is every token of its piece and its
                ``capture_position`` the last of them (the target state; the span's mean is the ridge bias).

        Raises:
            KeyError: if a marker has neither a binding nor a text, a binding or text is unused, or a name is in both.
            ValueError: if a flag has no template arguments for its state, a thinking request has no user message,
                ``spans`` or ``max_target_tokens`` come
                without a target, a capture comes with one, ``capture`` and ``read`` come together, ``read`` is not
                a text marker occurring once, a supplied text is empty, the chat template moves or alters a marker, a
                span is out of range or overlaps another, a capture fragment is not unique or crosses a marker (or
                lies after a token-id prefill part), a prefill part after a token-id part holds a marker, or the
                tokenizer does not decode its own ids back to the text.
        """
        if target is not None and (capture is not None or read is not None):
            raise ValueError("a capture prompt has no target")
        if capture is not None and read is not None:
            raise ValueError("capture and read both set the capture position: give one")
        conversation = self._requested(conversation)
        kwargs = self._template_kwargs(conversation)
        messages = [{"role": m.role, "content": m.content} for m in conversation.messages]
        text = _template_text(self.tokenizer, messages, add_generation_prompt=True, **kwargs)
        lead, tail = conversation.prefill_parts()
        text += lead
        pieces, names = split_markers(text)
        check_markers(names, set(bindings), texts)
        empty = sorted(k for k, v in texts.items() if not v)
        if empty:
            raise ValueError(f"supplied texts {empty} are empty")
        contents = [*(m.content for m in conversation.messages), lead]
        opens = opening_markers(contents, names)
        if read is not None and (read not in texts or names.count(read) != 1):
            raise ValueError(
                f"read {read!r} must be a text marker occurring once; markers {names}, texts {sorted(texts)}"
            )
        ids: list[int] = []
        slots: list[SlotOccurrence] = []
        span: tuple[int, ...] = ()
        for i, piece in enumerate(pieces):
            piece.offset = len(ids)
            piece.ids = self._encode_part(piece.text)
            ids += piece.ids
            if i == len(names):
                break
            if names[i] in texts:
                lead = "" if opens[i] else " "
                fill = self._encode_part(lead + texts[names[i]])
                if names[i] == read:
                    span = tuple(range(len(ids), len(ids) + len(fill)))
                ids += fill
                continue
            binding = bindings[names[i]]
            if binding.k < 1:
                raise ValueError(f"slot {names[i]!r}: k = {binding.k}, at least 1 placeholder is needed")
            slots.append(SlotOccurrence(names[i], binding, tuple(range(len(ids), len(ids) + binding.k))))
            ids += [self.placeholder_id] * binding.k
        ids += self._tail_ids(tail)
        position = span[-1] if span else None
        if capture is not None:
            fragment, offset = capture
            if not -len(fragment) <= offset < len(fragment):
                raise ValueError(f"capture offset {offset} is outside the fragment {fragment!r}")
            piece, start = locate(text, pieces, fragment)
            position = piece.offset + piece.bounds(self.tokenizer).token_at(start + offset % len(fragment))
        return RenderedPrompt(
            input_ids=tuple(ids),
            slots=tuple(slots),
            target=self._target(target, spans, max_target_tokens),
            capture_position=position,
            capture_span=span,
        )

    def render_recipe(
        self, messages: Sequence[RecipeText], values: Mapping[str, str], read: str, *, thinking: bool = False
    ) -> RenderedPrompt:
        """Render a capture recipe's messages (``domains.base.CaptureRecipe``) for one object.

        Each ``{key}`` placeholder takes ``values[key]``. ``{name}`` and the read placeholder become text markers, each
        its own tokenization piece (as :meth:`render` ``texts``: one leading space after text of its message, none when
        it opens its message or the prefill); every other placeholder (``{noun}``, an unread ``{context}``) is plain
        text. A trailing assistant message is the prefill: the answer the receiver reads after the generation prompt,
        with no end of turn (the original in-chat caption state).

        Args:
            messages: the recipe's messages (``CaptureRecipe.messages``: ``role`` and ``text`` with placeholders).
            values: placeholder → value (``noun``, ``name``, ``context``), exactly the placeholders used.
            read: the placeholder whose tokens are read; it occurs exactly once.
            thinking: the receiver's thinking mode for this prompt.

        Returns:
            The prompt, with ``capture_span`` the read placeholder's tokens and ``capture_position`` the last of them.

        Raises:
            KeyError: if a placeholder has no value or a value no placeholder.
            ValueError: if the recipe has no message besides the prefill, a message already holds a slot marker, or
                as :meth:`render`.
        """
        pieces = {"name", read}
        used: set[str] = set()

        def fill(m: re.Match[str]) -> str:
            key = m.group(1)
            if key not in values:
                raise KeyError(f"recipe placeholder {{{key}}} has no value; values {sorted(values)}")
            used.add(key)
            return f"<SLOT:{key}>" if key in pieces else values[key]

        filled = []
        for message in messages:
            if re.search(SLOT_PATTERN, message.text):
                raise ValueError(f"recipe message {message.text!r} holds a slot marker")
            filled.append(Message(message.role, _PLACEHOLDER.sub(fill, message.text)))
        unused = sorted(set(values) - used)
        if unused:
            raise KeyError(f"values {unused} fill no placeholder")
        prefill = filled.pop().content if filled and filled[-1].role == "assistant" else None
        if not filled:
            raise ValueError("a recipe needs a message before its prefilled answer")
        texts = {key: values[key] for key in pieces & used}
        conversation = Conversation(tuple(filled), prefill=prefill, thinking=thinking)
        return self.render(conversation, {}, texts=texts, read=read)

    def _tail_ids(self, tail: Sequence[str | TokenIds]) -> list[int]:
        """Return the ids of the prefill's parts from its first token-id part on: ids verbatim, texts on their own."""
        ids: list[int] = []
        for part in tail:
            if isinstance(part, TokenIds):
                ids += part.ids
            elif re.search(SLOT_PATTERN, part):
                raise ValueError(f"prefill part {part!r} follows a token-id part, so it cannot hold a marker")
            else:
                ids += self._encode_part(part)
        return ids

    def _requested(self, conversation: Conversation) -> Conversation:
        """Return ``conversation``, in thinking mode a copy whose last user message ends with the thinking request."""
        if not conversation.thinking or self.thinking_request is None:
            return conversation
        users = [i for i, m in enumerate(conversation.messages) if m.role == "user"]
        if not users:
            raise ValueError("the thinking request goes into the last user message; the conversation has none")
        turns = list(conversation.messages)
        turns[users[-1]] = Message("user", f"{turns[users[-1]].content} {self.thinking_request}")
        return replace(conversation, messages=tuple(turns))

    def _template_kwargs(self, conversation: Conversation) -> dict[str, object]:
        out: dict[str, object] = {}
        for flag in FLAGS:
            on = bool(getattr(conversation, flag))
            state: Literal["on", "off"] = "on" if on else "off"
            if flag not in self.template_kwargs:
                if on:
                    raise ValueError(f"the receiver states no template arguments for {flag!r}, so it cannot be on")
                continue
            if state not in self.template_kwargs[flag]:
                raise ValueError(f"the receiver states no template arguments for {flag!r} {state!r}")
            out |= self.template_kwargs[flag][state]
        return out

    def target_token_count(self, text: str) -> int:
        """Return how many tokens ``text`` takes as a target part (:class:`TargetPart`), before any cap.

        The renderer stays the one user of the tokenizer; callers that fit a target part under a cap ask it (e.g. a
        scaffold that drops facts until its part fits). Not for prompt text, which is split at slot markers.
        """
        return len(self._encode_part(text))

    def target_token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        """Return the ``[start, end)`` character range in ``text`` of each token it takes as a target part.

        The same encoding as :meth:`target_token_count` (so there is one range per token) and the same prefix-decoding
        boundaries as span weights. A token that completes a multi-byte character covers that character; a token
        covering no complete character gets an empty range at its position. The ranges tile ``text`` in order. Lets a
        caller place per-token weights by character position (e.g. a boost on the token after each ``;``) and still
        hand the renderer only :class:`WeightedSpan` s.

        Raises:
            ValueError: if the tokenizer does not decode its own ids back to ``text``.
        """
        bounds = Boundaries(self.tokenizer, self._encode_part(text), text)
        return tuple((bounds.b(i), bounds.b(i + 1)) for i in range(len(bounds.ids)))

    def target_prefix(self, text: str, tokens: int) -> str:
        """Return the text the first ``tokens`` tokens of ``text`` take as a target part (all of it if it is shorter).

        The same encoding as :meth:`target_token_count`; a token that ends inside a multi-byte character adds nothing
        (as in :meth:`target_token_offsets`). Lets a caller cut a text at a boundary of its own within a token cap
        (e.g. a sentence end, ``domains.molecule_text.sentences.trim``) with one decode, not one per token.

        Raises:
            ValueError: if ``tokens`` is negative, or the tokenizer does not decode its own ids back to ``text``.
        """
        if tokens < 0:
            raise ValueError(f"tokens must be >= 0, got {tokens}")
        ids = self._encode_part(text)
        if tokens >= len(ids):
            return text
        return text[: Boundaries(self.tokenizer, ids, text).b(tokens)]

    def _encode_part(self, text: str) -> list[int]:
        """Encode text on its own, without special tokens: the one encoding of every piece, part and lookup."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _target(
        self, target: str | Sequence[TargetPart] | None, spans: Sequence[WeightedSpan], cap: int | None
    ) -> Target | None:
        if target is None:
            if spans or cap is not None:
                raise ValueError("spans and max_target_tokens need a target")
            return None
        if isinstance(target, str):
            parts: Sequence[TargetPart] = (TargetPart(target, tuple(spans), cap),)
        elif spans or cap is not None:
            raise ValueError("a target given as parts carries its spans and caps in its parts")
        else:
            parts = target
        if not parts:
            raise ValueError("a target needs at least one part")
        ids: list[int] = []
        weights: list[float] = []
        for part in parts:
            if part.max_tokens is not None and part.max_tokens < 1:
                raise ValueError(f"a target part's cap must be at least 1, got {part.max_tokens}")
            part_ids = self._encode_part(part.text)
            part_weights = span_weights(Boundaries(self.tokenizer, part_ids, part.text), part.spans, len(part.text))
            ids += part_ids[: part.max_tokens]
            weights += part_weights[: part.max_tokens]
        return Target(token_ids=tuple(ids), weights=tuple(weights))


def _template_text(tokenizer: Tokenizer, messages: list[dict[str, str]], **kwargs: object) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template(tokenize=False) returned {type(text).__name__}, not str")
    return text
