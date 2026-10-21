"""Character ↔ token mapping for the renderer, by decoding token prefixes (no tokenizer offset maps needed).

Private to ``receivers.render``: not every tokenizer has offset maps (Kimi's has none), so the renderer maps character
ranges of a text onto its token ids by decoding prefixes, lazily and with a binary search. It also holds the text
pieces between slot markers (splitting, binding checks, which markers open their message), fragment location and
the span → token weight mapping.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from typing import Protocol

__all__ = [
    "SLOT_PATTERN",
    "Boundaries",
    "Piece",
    "check_markers",
    "locate",
    "opening_markers",
    "span_weights",
    "split_markers",
]

SLOT_PATTERN = r" ?<SLOT:([A-Za-z0-9_]+)>"
"""A slot marker ``<SLOT:name>`` and the one space before it, which the marker removes."""


class _Decoder(Protocol):
    def decode(self, token_ids: Sequence[int]) -> str: ...


class _Span(Protocol):
    """``render.WeightedSpan``, structurally."""

    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...
    @property
    def weight(self) -> float: ...
    @property
    def curve(self) -> tuple[float, ...]: ...


class Boundaries:
    """Character boundaries of a token sequence in its text.

    Token i covers characters ``[b(i), b(i + 1))``. A token that ends inside a multi-byte character decodes to a
    replacement character; it covers nothing, and the token that completes the character covers it.

    Args:
        tokenizer: decodes ids to text.
        ids: the text's token ids.
        text: the text.

    Raises:
        ValueError: if the ids do not decode back to the text.
    """

    def __init__(self, tokenizer: _Decoder, ids: Sequence[int], text: str) -> None:
        self.tokenizer, self.ids, self.text = tokenizer, ids, text
        self._b: dict[int, int] = {0: 0}
        decoded = tokenizer.decode(list(ids))
        if decoded != text:
            raise ValueError(f"the tokenizer does not decode its own ids back to the text: {decoded!r} != {text!r}")
        self._b[len(ids)] = len(text)

    def b(self, i: int) -> int:
        """Return the number of characters the first ``i`` tokens decode to.

        Raises:
            ValueError: if that prefix does not decode to a prefix of the text.
        """
        if i not in self._b:
            prefix = self.tokenizer.decode(list(self.ids[:i])).rstrip("�")
            if not self.text.startswith(prefix):
                raise ValueError(f"token prefix {i} decodes to {prefix!r}, which does not start the text")
            self._b[i] = len(prefix)
        return self._b[i]

    def _first(self, pred: Callable[[int], bool]) -> int:
        """The smallest i in [0, n] with ``pred(i)`` (``pred`` monotone and true at n)."""
        lo, hi = 0, len(self.ids)
        while lo < hi:
            mid = (lo + hi) // 2
            if pred(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def overlapping(self, start: int, end: int) -> range:
        """Return the tokens that overlap characters ``[start, end)`` by at least one character."""
        n = len(self.ids)
        lo = self._first(lambda i: i == n or self.b(i + 1) > start)
        hi = self._first(lambda i: i == n or self.b(i) >= end)
        return range(lo, hi)

    def token_at(self, char: int) -> int:
        """Return the token holding character ``char``.

        Raises:
            ValueError: if ``char`` is outside the text.
        """
        tokens = self.overlapping(char, char + 1)
        if not 0 <= char < len(self.text) or not tokens:
            raise ValueError(f"character {char} is outside the {len(self.text)}-char text")
        return tokens[0]


@dataclass
class Piece:
    """Text between slot markers.

    Attributes:
        text: the text.
        start: its first character in the rendered text.
        end: one past its last character.
        offset: the prompt position of its first token.
        ids: its token ids, encoded on their own.
    """

    text: str
    start: int
    end: int
    offset: int = 0
    ids: list[int] = field(default_factory=list)

    def bounds(self, tokenizer: _Decoder) -> Boundaries:
        """Return the piece's character boundaries."""
        return Boundaries(tokenizer, self.ids, self.text)


def locate(text: str, pieces: Sequence[Piece], fragment: str) -> tuple[Piece, int]:
    """Return the piece holding the one occurrence of ``fragment`` and the fragment's start within it."""
    found = [m.start() for m in re.finditer(re.escape(fragment), text)] if fragment else []
    if len(found) != 1:
        raise ValueError(f"capture fragment {fragment!r} occurs {len(found)} times in the rendered text, not once")
    for piece in pieces:
        if piece.start <= found[0] and found[0] + len(fragment) <= piece.end:
            return piece, found[0] - piece.start
    raise ValueError(f"capture fragment {fragment!r} crosses a slot marker")


def span_weights(bounds: Boundaries, spans: Sequence[_Span], n_chars: int) -> list[float]:
    """Return one weight per token: the first span it overlaps (its curve along the span), 0 outside every span.

    Raises:
        ValueError: if a span is out of range or overlaps another.
    """
    weights = [0.0] * len(bounds.ids)
    taken = [False] * len(bounds.ids)
    previous_end = 0
    for span in sorted(spans, key=lambda s: s.start):
        if not 0 <= span.start < span.end <= n_chars:
            raise ValueError(f"span [{span.start}, {span.end}) is not a non-empty range of the {n_chars}-char target")
        if span.start < previous_end:
            raise ValueError(f"span [{span.start}, {span.end}) overlaps the span before it")
        previous_end = span.end
        j = 0
        for i in bounds.overlapping(span.start, span.end):
            if not taken[i]:
                weights[i] = span.curve[min(j, len(span.curve) - 1)] if span.curve else span.weight
                taken[i] = True
                j += 1
    return weights


def split_markers(text: str) -> tuple[list[Piece], list[str]]:
    """Split rendered text at the markers: the pieces around them (one more than markers) and the marker names."""
    pieces, names, at = [], [], 0
    for m in re.finditer(SLOT_PATTERN, text):
        pieces.append(Piece(text[at : m.start()], at, m.start()))
        names.append(m.group(1))
        at = m.end()
    pieces.append(Piece(text[at:], at, len(text)))
    return pieces, names


def opening_markers(contents: Sequence[str], names: Sequence[str]) -> list[bool]:
    """Per marker of the rendered text, whether it opens its content (a message or the prefill, in order).

    Raises:
        ValueError: if the markers of the messages and prefill, in order, are not those of the rendered text.
    """
    found: list[tuple[str, bool]] = []
    for content in contents:
        found += [(m.group(1), m.start() == 0) for m in re.finditer(SLOT_PATTERN, content)]
    if [name for name, _ in found] != list(names):
        raise ValueError(f"the chat template moved or altered slot markers: {[n for n, _ in found]} became {names}")
    return [opens for _, opens in found]


def check_markers(names: Sequence[str], bindings: Set[str], texts: Mapping[str, str]) -> None:
    """Check that every marker has exactly one of a binding or a text, and that each is used.

    Raises:
        KeyError: for a marker with neither, a name with both, or a binding or text without a marker.
    """
    both = sorted(bindings & set(texts))
    if both:
        raise KeyError(f"markers both bound and given a text: {both}")
    missing = sorted(set(names) - bindings - set(texts))
    if missing:
        raise KeyError(f"slot markers without a binding: {missing}")
    unused = sorted((bindings | set(texts)) - set(names))
    if unused:
        raise KeyError(f"bindings without a slot marker: {unused}")
