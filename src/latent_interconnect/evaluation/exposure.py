"""The ``description_exposure`` analysis kind: expected description-target exposure per zero-shot task.

The share of all training examples that are text-stream rows asking about a **positive** object of a zero-shot task
and answered with a description whose capped text matches the task's cue. It is an **expectation** under the run's own
sampler, not a replay of executed batches:

``fraction(task, tier) = p_text · Σ_a P(a) · pos(a) · Σ_f P(f|a) · 1[tier matches the capped answer of field f of a]``

- ``p_text`` is the ``text`` entry's share of the mixture (``MixtureEntry.share`` over the shares' sum): every example
  draws its stream by share (``data.sampler``), so the denominator is all examples and the fraction does not depend on
  the run length or batch;
- ``P(a)`` is the built ``TextStream``'s own draw probability (``ObjectPool.probabilities``: ``cluster_sqrt`` weights
  counted over the stream's population, renormalized over the objects that answer a template), so the weighting is
  the sampler's by construction;
- ``P(f|a)`` is the share of the templates the object answers (``TextStream.answerable``) whose task's field is ``f``:
  a template is uniform among those it answers, so a field is weighted by its template count over the object's
  answered fields;
- the answer is the field's training text as the stream words it (``data.streams.worded``: an item-list field's
  answer line), and ``pos(a)`` is the task's domain label (``TaskSpec.label`` of ``Domain.labels``);
- the cap is the declared one (:class:`CharCap`: the first characters; :class:`TokenCap`: when longer, the decoded
  first tokens of a receiver's tokenizer), and each tier is also reported uncapped (the cap's sensitivity).

It is the quantity of the paper's description-exposure check. Left out, as in that check: grammar-worded rows
(treated as template rows), the template redraw limit (``qa_streams.TEMPLATE_TRIES``), in-context examples, the
reasoning scaffold, and the relational, Yes/No and variant streams. The cues (``ExposureCues``) are versioned data
(``configs/analyses/exposure/``); a task's tiers are reported side by side, with the one that decides the threshold
named (``TaskCues.headline``).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from typing import ClassVar

from pydantic import Field, model_validator

from latent_interconnect.artifacts.records import Record, record_output
from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.config import Config
from latent_interconnect.data.items import ItemSet
from latent_interconnect.data.qa_streams import TextStream
from latent_interconnect.data.sampler import MixtureEntry
from latent_interconnect.data.streams import build_stream, worded
from latent_interconnect.data.training_text import TRAINING_SET_KIND, load_training_text
from latent_interconnect.domains.base import DomainSpec, open_domain
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS
from latent_interconnect.evaluation.results import FINITE, AnalysisResult
from latent_interconnect.receivers.base import ReceiverSpec
from latent_interconnect.receivers.hf import load_tokenizer
from latent_interconnect.stats.bootstrap import ResamplingUnits

__all__ = [
    "Alternative",
    "CharCap",
    "DescriptionExposure",
    "DescriptionExposureKind",
    "DescriptionExposureResult",
    "ExposureCues",
    "Strip",
    "TaskCues",
    "TaskExposure",
    "TierExposure",
    "TokenCap",
]


class Strip(Config):
    """A rewrite applied to an answer before an alternative's patterns are searched (``re.sub``, every match).

    Attributes:
        pattern: the regular expression (flags inline, e.g. ``(?i)``).
        replace: the replacement text.
    """

    pattern: str = Field(min_length=1)
    replace: str


class Alternative(Config):
    """One way a tier matches: after the named strips, in order, any of the named patterns is found.

    Attributes:
        match: names of the task's patterns (``TaskCues.patterns``); any one found is a match.
        strip: names of the cue file's strips (``ExposureCues.strip``), applied in order first; none: the answer as is.
    """

    match: tuple[str, ...] = Field(min_length=1)
    strip: tuple[str, ...] = ()


class TaskCues(Config):
    """One zero-shot task's cues.

    Attributes:
        patterns: name → regular expression (flags inline), searched with ``re.search``.
        tiers: tier name → its alternatives; the tier matches an answer when any alternative does.
        headline: the tier that decides the task's threshold (every tier is reported).
        extension: whether the task extends the paper's criterion (its cue is an added one,
            not one the paper's < 0.1 % claim used); reported with the result.
    """

    patterns: dict[str, str] = Field(min_length=1)
    tiers: dict[str, tuple[Alternative, ...]] = Field(min_length=1)
    headline: str
    extension: bool = False

    @model_validator(mode="after")
    def _check(self) -> TaskCues:
        if self.headline not in self.tiers:
            raise ValueError(f"headline tier {self.headline!r} is not among the tiers {sorted(self.tiers)}")
        unknown = sorted({p for alts in self.tiers.values() for a in alts for p in a.match} - set(self.patterns))
        if unknown:
            raise ValueError(f"tiers name patterns {unknown} the task does not define")
        empty = sorted(name for name, alts in self.tiers.items() if not alts)
        if empty:
            raise ValueError(f"tiers {empty} have no alternative")
        for name, pattern in self.patterns.items():
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"pattern {name!r} does not compile: {e}") from e
        return self


class ExposureCues(Config):
    """The cue data of one domain's zero-shot tasks (a versioned file under ``configs/analyses/exposure/``).

    Attributes:
        tasks: zero-shot task id (of the domain catalog) → its cues.
        strip: name → a rewrite the tasks' alternatives may apply before matching.
    """

    tasks: dict[str, TaskCues] = Field(min_length=1)
    strip: dict[str, Strip] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> ExposureCues:
        named = {s for t in self.tasks.values() for alts in t.tiers.values() for a in alts for s in a.strip}
        unknown = sorted(named - set(self.strip))
        if unknown:
            raise ValueError(f"alternatives name strips {unknown} the cue file does not define")
        for name, s in self.strip.items():
            try:
                re.compile(s.pattern)
            except re.error as e:
                raise ValueError(f"strip {name!r} does not compile: {e}") from e
        return self


class CharCap(Config):
    """The answer cut to its first characters (the protein check's proxy for the token cap).

    Attributes:
        chars: characters kept.
    """

    chars: int = Field(ge=1)


class TokenCap(Config):
    """The answer cut to its first tokens of a receiver's tokenizer (the molecule check's cap).

    An answer of at most ``tokens`` tokens (``encode`` without special tokens) is kept whole; a longer one is the
    ``decode`` of its first ``tokens`` ids.

    Attributes:
        tokens: tokens kept.
        receiver: the receiver whose tokenizer counts them (``receivers.hf.load_tokenizer``, on CPU).
    """

    tokens: int = Field(ge=1)
    receiver: ReceiverSpec


class DescriptionExposure(Config):
    """Parameters of the ``description_exposure`` kind (one input: the ``training_set`` stage's record).

    Attributes:
        mixture: the training runs' mixture; its one ``text`` entry is the stream measured and its share ``p_text``.
        cap: the answer cap the cues are searched within.
        cues: the zero-shot tasks' cues.
    """

    mixture: tuple[MixtureEntry, ...] = Field(min_length=1)
    cap: CharCap | TokenCap
    cues: ExposureCues

    @model_validator(mode="after")
    def _one_text(self) -> DescriptionExposure:
        n = sum(m.stream == "text" for m in self.mixture)
        if n != 1:
            raise ValueError(f"the mixture needs exactly one text stream, not {n}")
        return self


class TierExposure(Config):
    """One tier's exposure, as fractions of all training examples.

    Attributes:
        capped: within the declared cap (the reported value).
        uncapped: on the whole answer (the cap's sensitivity).
    """

    model_config = FINITE

    capped: float = Field(ge=0.0, le=1.0)
    uncapped: float = Field(ge=0.0, le=1.0)


class TaskExposure(Config):
    """One zero-shot task's exposure.

    Attributes:
        label: the domain label column whose true values are the positives.
        positives: positive objects among the text stream's population (its objects that are not, e.g., variants).
        headline: the tier that decides the threshold (``TaskCues.headline``).
        extension: whether the task extends the paper's criterion (``TaskCues.extension``).
        tiers: tier → its exposure.
    """

    model_config = FINITE

    label: str
    positives: int = Field(ge=0)
    headline: str
    extension: bool
    tiers: dict[str, TierExposure]


class DescriptionExposureResult(AnalysisResult):
    """Result of ``description_exposure``.

    Attributes:
        training_set: the digest of the ``training_set`` artifact the stream drew from.
        p_text: the text stream's share of all examples.
        objects: the objects the text stream draws (those answering at least one template).
        tasks: zero-shot task → its exposure.
    """

    model_config = FINITE

    training_set: str
    p_text: float = Field(gt=0.0, le=1.0)
    objects: int = Field(ge=1)
    tasks: dict[str, TaskExposure]


def _cutter(cap: CharCap | TokenCap) -> Callable[[str], str]:
    """The answer as the cap leaves it."""
    if isinstance(cap, CharCap):
        return lambda text: text[: cap.chars]
    tokenizer = load_tokenizer(cap.receiver)
    cut: dict[str, str] = {}  # an answer repeats across objects; tokenize it once

    def tokens(text: str) -> str:
        if text not in cut:
            ids = tokenizer.encode(text, add_special_tokens=False)
            cut[text] = text if len(ids) <= cap.tokens else tokenizer.decode(ids[: cap.tokens])
        return cut[text]

    return tokens


def _matcher(
    cues: TaskCues, alternatives: tuple[Alternative, ...], strip: Mapping[str, Strip]
) -> Callable[[str], bool]:
    """Whether an answer matches a tier: any alternative's patterns found after its strips."""
    compiled = [
        (
            [(re.compile(strip[s].pattern), strip[s].replace) for s in a.strip],
            [re.compile(cues.patterns[p]) for p in a.match],
        )
        for a in alternatives
    ]

    def match(text: str) -> bool:
        for rewrites, patterns in compiled:
            stripped = text
            for pattern, replace in rewrites:
                stripped = pattern.sub(replace, stripped)
            if any(p.search(stripped) for p in patterns):
                return True
        return False

    return match


@ANALYSIS_KINDS.register("description_exposure")
class DescriptionExposureKind:
    """Expected description-target exposure per zero-shot task (params :class:`DescriptionExposure`)."""

    Params: ClassVar[type[Config]] = DescriptionExposure
    Result: ClassVar[type[AnalysisResult]] = DescriptionExposureResult

    def run(
        self,
        params: DescriptionExposure,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> DescriptionExposureResult:
        """Return every cued task's exposure (module docstring).

        Raises:
            ValueError: if there is not exactly one input, or it has no ``training_set`` output.
            KeyError: if a cued task is not in the domain catalog.
            TypeError: if the registered ``text`` stream is not ``qa_streams.TextStream``.
        """
        del items  # the stream reads the training set and its domain
        if len(inputs) != 1:
            raise ValueError(f"description_exposure reads one training_set record, not {sorted(inputs)}")
        (record,) = inputs.values()
        ref = record_output(record, TRAINING_SET_KIND)
        domain = open_domain(DomainSpec.model_validate(record.config["domain"]), artifacts)
        catalog = domain.tasks()
        unknown = sorted(set(params.cues.tasks) - set(catalog))
        if unknown:
            raise KeyError(f"cued tasks {unknown} are not in the domain catalog")
        (entry,) = (m for m in params.mixture if m.stream == "text")
        stream = build_stream("text", entry.params, domain, load_training_text(artifacts, ref)).stream
        if not isinstance(stream, TextStream):
            raise TypeError(f"the text stream built a {type(stream).__name__}")
        p_text = entry.share / math.fsum(m.share for m in params.mixture)
        # One row per (object, answered field): P(a)·P(f|a) and the worded answer, in the pool's order.
        rows: list[tuple[str, float, str]] = []
        for oid, p in zip(stream.pool.ids, stream.pool.probabilities(), strict=True):
            templates = stream.answerable[oid]
            per_field = Counter(stream.tasks[t.task or ""].label for t in templates)
            for field, n in sorted(per_field.items()):
                answer = worded(stream.params.answer_lines, field, stream.text.fields[oid][field])
                rows.append((oid, float(p) * n / len(templates), answer))
        cut, labels, population = _cutter(params.cap), domain.labels(), list(stream.answerable)
        tasks = {}
        for task, cues in sorted(params.cues.tasks.items()):
            label = catalog[task].label
            positive = set(labels.index[labels[label].fillna(False).astype(bool)])
            mine = [(mass, answer) for oid, mass, answer in rows if oid in positive]
            tiers = {}
            for tier, alternatives in sorted(cues.tiers.items()):
                match = _matcher(cues, alternatives, params.cues.strip)
                tiers[tier] = TierExposure(
                    capped=p_text * math.fsum(m for m, a in mine if match(cut(a))),
                    uncapped=p_text * math.fsum(m for m, a in mine if match(a)),
                )
            tasks[task] = TaskExposure(
                label=label,
                positives=sum(o in positive for o in population),
                headline=cues.headline,
                extension=cues.extension,
                tiers=tiers,
            )
        return DescriptionExposureResult(
            units=units, training_set=ref.sha256, p_text=p_text, objects=len(stream.pool.ids), tasks=tasks
        )
