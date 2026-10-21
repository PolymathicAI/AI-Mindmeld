"""The batch measurement tool: how many examples one ``loss`` call can take on the training hardware.

Run it with ``python -m latent_interconnect.run.measure --config <measure.yaml>``.

The run's batch is a fixed config value (``RunConfig.batch`` and ``micro_batch``), so what fits is measured
beforehand and written into the run config by a person. For each candidate size, largest first, the tool builds the
worst case the run will meet within its first ``probe_steps`` steps: the candidate's number of longest rendered
prompts (prompt and target tokens) among those steps' examples, drawn at the run's own rates (the original autotune
probe forced in-context examples on every row, a stricter worst case). It runs one training update on it with the
trainer's own parts: the maps and their step-0 drift references placed and the run's AdamW
(``Trainer.optimizer``), whose moments exist before the first candidate (as from the run's first update on), then the
writes, one forward and backward through the training backend (``TrainingBackend.loss``) and the update's tail
(``Trainer.apply``: penalty, clip, optimizer step) at learning rate 0, so the step-0 connector never moves. It reports
each visible device's peak allocated memory as a fraction of its total, the seconds and the tokens. The largest
candidate under ``memory_rule`` is the recommendation (below that fraction of every device's memory; the original
code's rule was 0.90, the paper's production measurements take 0.92). The
``measure`` stage commits the result; a person writes the size into the run config. Memory is NaN on CPU, where no
candidate is recommended.

A candidate that runs out of device memory (``torch.cuda.OutOfMemoryError`` only, in the writes, the loss, the
backward or the step) is recorded as ``oom``, with no memory or seconds, and never fits; its gradients and graph are
released and every device's cache emptied before the next, smaller candidate. Any other error ends the measurement.
The execution options (``kimi-ep``'s ``checkpointing``) are the stage's ``--option``s, plus the run's ``checkpointing``.
"""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from pydantic import Field

from latent_interconnect.config import StageConfig
from latent_interconnect.receivers.kernel import resolve_sources
from latent_interconnect.receivers.render import RenderedPrompt
from latent_interconnect.training.config import RunConfig
from latent_interconnect.training.trainer import Trainer
from latent_interconnect.writes.sources import LiveSource, WriteSource

__all__ = ["MeasureSpec", "Measurement", "measure_batches", "worst_case"]


class MeasureSpec(StageConfig):
    """A batch measurement.

    Attributes:
        run: the run config (embedded) whose data, connector and receiver are measured.
        candidates: the examples per ``loss`` call to try, largest first.
        probe_steps: the steps whose examples the worst case is drawn from.
        memory_rule: the fraction of a device's memory a candidate's peak must stay below (the original code: 0.90;
            the paper's production measurements: 0.92).
    """

    run: RunConfig
    candidates: tuple[int, ...] = Field(min_length=1)
    probe_steps: int = Field(ge=1)
    memory_rule: float = Field(gt=0.0, le=1.0)


@dataclass(frozen=True)
class Measurement:
    """One candidate's measurement.

    Attributes:
        size: examples in the call.
        oom: the candidate ran out of device memory: it does not fit, and its memory and seconds are NaN.
        peak_memory: the fullest device's peak allocated memory as a fraction of its total (NaN on CPU or ``oom``).
        device_peaks: each visible device's peak fraction, in device order (empty on CPU or ``oom``).
        start_memory: each visible device's allocated fraction when the candidate started (after the previous one's
            release; empty on CPU).
        seconds: wall time of the update (writes, forward, backward, optimizer step).
        tokens: prompt and target tokens processed.
    """

    size: int
    oom: bool
    peak_memory: float
    device_peaks: tuple[float, ...]
    start_memory: tuple[float, ...]
    seconds: float
    tokens: float

    def fits(self, rule: float) -> bool:
        """Return whether the peak memory is measured and below ``rule``."""
        return not math.isnan(self.peak_memory) and self.peak_memory < rule


def worst_case(prompts: Sequence[RenderedPrompt], size: int) -> list[RenderedPrompt]:
    """Return the ``size`` longest prompts (prompt and target tokens), longest first; ties keep their order."""

    def length(p: RenderedPrompt) -> int:
        return len(p.input_ids) + (len(p.target.token_ids) if p.target else 0)

    order = sorted(range(len(prompts)), key=lambda i: -length(prompts[i]))
    return [prompts[i] for i in order[:size]]


def measure_batches(trainer: Trainer, spec: MeasureSpec) -> list[Measurement]:
    """Measure every candidate on ``trainer``'s backend and step-0 connector (see the module docstring).

    Raises:
        ValueError: if a candidate exceeds the examples the probe steps hold.
    """
    prompts = [p for step in range(spec.probe_steps) for p in trainer.sampler.prompts(step)]
    too_big = [c for c in spec.candidates if c > len(prompts)]
    if too_big:
        raise ValueError(f"candidates {too_big} exceed the {len(prompts)} examples of {spec.probe_steps} probe steps")
    optimizer = trainer.optimizer()
    for group in optimizer.param_groups:
        group["lr"] = 0.0  # the probe never moves the step-0 connector
    for p in trainer.connector.parameters():
        p.grad = torch.zeros_like(p)
    trainer.apply(optimizer)  # AdamW's moments now exist, as during the run
    source = {trainer.config.connector.name: LiveSource(trainer.connector, trainer.load)}
    out = []
    for size in sorted(spec.candidates, reverse=True):
        batch = worst_case(prompts, size)
        start = _fractions(torch.cuda.memory_allocated)
        try:
            out.append(_update(trainer, optimizer, batch, source, start))
        except torch.cuda.OutOfMemoryError:
            out.append(Measurement(size, True, math.nan, (), start, math.nan, _tokens(batch)))
        # here, after the handler: the exception's traceback, and the graph and writes its frames held, are gone
        trainer.connector.zero_grad(set_to_none=True)
        gc.collect()  # frames caught in reference cycles
        torch.cuda.empty_cache()  # the caching allocator's free blocks, on every device
    return out


def _update(
    trainer: Trainer,
    optimizer: torch.optim.Optimizer,
    batch: Sequence[RenderedPrompt],
    source: Mapping[str, WriteSource],
    start: tuple[float, ...],
) -> Measurement:
    """One training update of ``batch`` as one ``loss`` call, measured on every visible device."""
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)
    began = time.perf_counter()
    terms = trainer.backend.loss(batch, resolve_sources(batch, source))
    torch.autograd.backward(terms.weighted_sum / terms.weight_sum)
    trainer.apply(optimizer)
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
    seconds = time.perf_counter() - began
    peaks = _fractions(torch.cuda.max_memory_allocated)
    return Measurement(len(batch), False, max(peaks, default=math.nan), peaks, start, seconds, _tokens(batch))


def _fractions(allocated: Callable[[int], int]) -> tuple[float, ...]:
    """Each visible device's ``allocated(device)`` as a fraction of its total memory (empty on CPU)."""
    n = torch.cuda.device_count()
    return tuple(allocated(d) / torch.cuda.get_device_properties(d).total_memory for d in range(n))


def _tokens(batch: Sequence[RenderedPrompt]) -> float:
    return float(sum(len(p.input_ids) + len(p.target.token_ids if p.target else ()) for p in batch))
