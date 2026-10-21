"""Where a reasoning trace is cut before the forced answer: capped, at its last sentence; finished, before its answer.

A finished trace is cut so only on a receiver without a thinking close (:func:`answer_line_cut`).

A capped trace that never closed its thinking is cut off at its most recent complete sentence, its reasoning is closed
there, and an answer is forced.

The one definition of a sentence end (:func:`sentence_ends`): a ``.``, ``!`` or ``?``, optionally followed by closing
quotes, brackets or Markdown emphasis (``*``, ``_``: ``**cytoplasmic.**``), then whitespace (a space or a line break).
Not a sentence end:

- a line break alone (a half-finished bullet or fragment ending in a newline is not a complete sentence);
- a decimal point (``3.5``: no whitespace follows the ``.``);
- the period of a bare number at a line's start (a list marker: a line "2. Binds ATP" after a line break);
- the end of one of :data:`ABBREVIATIONS` (``e.g.``, ``i.e.``, ``etc.``, ``vs.``, ``cf.``, ``approx.``), whatever
  follows it;
- a ``.`` at the very end of the text (nothing shows it is not cut mid-token, as ``3.`` of ``3.5``).

The cut (:func:`last_sentence_cut`) keeps the fewest leading token ids whose decoded text reaches the last sentence
end: the first token boundary at or after it, so a token that carries the full stop and its space or line break is
kept whole. A trace with no sentence end is kept whole and the fallback is reported, never silent.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Literal

__all__ = ["ABBREVIATIONS", "CutRule", "answer_line_cut", "last_sentence_cut", "sentence_ends"]

ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "cf.", "approx.")
"""Abbreviations whose final period is not a sentence end (matched case-insensitively at a word start)."""
CutRule = Literal["none", "last_sentence", "fallback", "answer_line"]
"""How a trace was cut before the forced answer: not cut (a closed trace, or a finished one that states no answer), at
its last sentence end, kept whole because it has none (the fallback), or before its own answer
(:func:`answer_line_cut`)."""
_END = re.compile(r"[.!?][\"'”’)\]*_]*(?=\s)")
_LIST_NUMBER = re.compile(r"(?:^|\n)[ \t]*\d+\.$")
"""A bare number at a line's start before its period: a list marker ("2." opening a line), not a sentence end."""
_ABBREVIATION = re.compile(r"(?:^|(?<=[\s(\[\"']))(?:" + "|".join(re.escape(a) for a in ABBREVIATIONS) + r")$", re.I)


def sentence_ends(text: str) -> list[int]:
    """Return the character offsets just past every sentence end of ``text`` (module docstring), in order."""
    ends = []
    for match in _END.finditer(text):
        stop = match.start() + 1  # just past the terminal mark; closers count as part of the sentence below
        if match.group()[0] == "." and (_ABBREVIATION.search(text[:stop]) or _LIST_NUMBER.search(text[:stop])):
            continue
        ends.append(match.end())
    return ends


def last_sentence_cut(token_ids: Sequence[int], decode: Callable[[Sequence[int]], str]) -> tuple[int, CutRule]:
    """Return how many leading ids to keep, and the rule that decided it, for a trace cut at its last sentence.

    Args:
        token_ids: the trace's ids (a capped trace's, without any thinking close).
        decode: the receiver's continuation decoder (``receivers.base.generated_text`` with its tokenizer).

    Returns:
        ``(k, "last_sentence")`` with ``k`` the first token boundary at or after the last sentence end, or
        ``(len(token_ids), "fallback")`` when the decoded trace has no sentence end.
    """
    ends = sentence_ends(decode(token_ids))
    if not ends:
        return len(token_ids), "fallback"
    return next(k for k in range(1, len(token_ids) + 1) if len(decode(token_ids[:k])) >= ends[-1]), "last_sentence"


def answer_line_cut(
    token_ids: Sequence[int], decode: Callable[[Sequence[int]], str], label: str
) -> tuple[int, CutRule]:
    """Return how many leading ids to keep, and the rule, for a finished trace cut before its own answer.

    On a receiver without a thinking close a finished trace ends on its own
    ``Answer: Yes`` / ``Answer: No``, which the forced answer would otherwise copy. The cut keeps the fewest leading ids
    whose decoded text reaches the text before the last ``label`` (at a line's start or not), less trailing whitespace,
    so the receiver's separator and the readout's prefix follow the reasoning itself.

    Args:
        token_ids: the trace's ids, without its end id.
        decode: the receiver's continuation decoder (``receivers.base.generated_text`` with its tokenizer).
        label: the readout's answer prefix (``"Answer:"``).

    Returns:
        ``(k, "answer_line")``, or ``(len(token_ids), "none")`` when the decoded trace has no ``label``.
    """
    text = decode(token_ids)
    at = text.rfind(label)
    if at < 0:
        return len(token_ids), "none"
    stop, k = len(text[:at].rstrip()), len(token_ids)
    while k > 0 and len(decode(token_ids[: k - 1])) >= stop:  # the answer sits at the end: walk back to it
        k -= 1
    return k, "answer_line"
