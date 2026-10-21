"""Progress lines for long backend calls: logging only, never read by anything computed, recorded or digested.

A long suite (reasoning-trace generation above all) spends minutes to hours in one backend call. :class:`Progress`
writes one line to stderr at most every :data:`PROGRESS_EVERY_S` seconds, and a closing line at the end of a call that
wrote one, so a short call stays silent; a rendered job's ``lip_run`` appends stderr to the job's ``log.txt``::

    progress hf generate: 120/2000 items (6.0%), elapsed 0:10:03, 0.199 items/s, ~2:37:30 left
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from datetime import timedelta
from typing import TextIO

__all__ = ["PROGRESS_EVERY_S", "Progress"]

PROGRESS_EVERY_S = 60.0
"""Seconds between two progress lines of one call."""


class Progress:
    """Count the items of one call and write a progress line once ``every_s`` seconds passed since the last.

    Args:
        label: what the line names (e.g. ``"hf generate"``).
        total: the call's items.
        every_s: seconds between two lines.
        clock: the monotonic clock (tests pass a fake one).
        stream: where lines go; None: ``sys.stderr`` at the time of writing.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        every_s: float = PROGRESS_EVERY_S,
        clock: Callable[[], float] = time.monotonic,
        stream: TextIO | None = None,
    ) -> None:
        self.label, self.total, self.every_s = label, total, every_s
        self.done = 0
        self._clock, self._stream = clock, stream
        self._start = self._last = clock()
        self._wrote = False

    def advance(self, n: int = 1) -> None:
        """Count ``n`` more items done; write a line when due, or the closing one when the call ends."""
        self.done += n
        now = self._clock()
        if now - self._last >= self.every_s or (self._wrote and self.done >= self.total):
            self._write(now)

    def _write(self, now: float) -> None:
        elapsed = now - self._start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        left = f"~{timedelta(seconds=round((self.total - self.done) / rate))} left" if rate > 0 else "time left unknown"
        pct = 100.0 * self.done / self.total if self.total else 100.0
        stream = self._stream if self._stream is not None else sys.stderr
        stream.write(
            f"progress {self.label}: {self.done}/{self.total} items ({pct:.1f}%), "
            f"elapsed {timedelta(seconds=round(elapsed))}, {rate:.3g} items/s, {left}\n"
        )
        stream.flush()
        self._last, self._wrote = now, True
