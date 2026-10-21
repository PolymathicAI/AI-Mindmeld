"""The run config: everything a training run reads, in one validated file (``configs/runs/*.yaml``).

There are no flags beyond ``--config``. The connector and receiver configs are embedded (``$ref`` in YAML), the
domain is cited by data digest, and inputs by artifact digest, so the run's digest covers everything that changes its
numbers.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, model_validator

from latent_interconnect.build.init_state import InitSpec, SourceFit
from latent_interconnect.config import ArtifactDigest, Config, StageConfig, added_field
from latent_interconnect.data.sampler import MixtureEntry
from latent_interconnect.data.suites import SuiteSpec
from latent_interconnect.domains.base import DomainSpec
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.provenance import DeviceClass
from latent_interconnect.receivers.base import ReceiverSpec
from latent_interconnect.receivers.kernel import BatchPlan
from latent_interconnect.writes.connector import ConnectorSpec

__all__ = ["AchievedSpec", "CheckpointSpec", "OptimizerSpec", "RunConfig"]


class OptimizerSpec(Config):
    """AdamW with per-tensor learning rates ``lr_rel × reference`` (references from each map kind).

    Every group's rate is its base rate times :meth:`factor`: a linear warmup, then constant or a cosine. The schedule
    is keyed to the global step (origin 0), so a resumed run continues the same curve (the original
    ``--lr-anchor start``).

    Weight decay is relative and independent of the learning rate: at the update of step ``s`` each tensor a map
    kind declares in ``decayed()`` (``affine``'s W, a stream kind's output map) is scaled by
    ``1 − factor(s)·weight_decay`` before the Adam step, whatever its learning rate; biases, the layer-0 constant,
    every other parameter and the α logits are not decayed. Over a run an unopposed weight shrinks by
    ``exp(−weight_decay · Σ factor)``. (PyTorch's AdamW decay is decoupled from the gradient moments but scaled by
    each group's rate, ``lr_rel × reference``, so a small tensor would barely decay; the trainer sets each group's
    ``weight_decay`` to ``weight_decay / base rate`` to undo that.)

    Attributes:
        lr_rel: relative step size.
        lr_alpha: learning rate of the α logits.
        warmup_steps: linear warmup.
        schedule: ``constant`` after warmup, or ``cosine``.
        lr_floor: the cosine's final factor (originally ``--lr-min-frac``); required with ``cosine``, absent otherwise.
        weight_decay: the per-step relative decay λ of the decayed tensors (0: none; see above).
        clip: global gradient-norm clip.
    """

    lr_rel: float = Field(gt=0.0)
    lr_alpha: float = Field(gt=0.0)
    warmup_steps: int = Field(ge=0)
    schedule: Literal["constant", "cosine"]
    lr_floor: float | None = Field(default=None, ge=0.0, le=1.0)
    weight_decay: float = Field(ge=0.0, lt=1.0)
    clip: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> OptimizerSpec:
        if (self.schedule == "cosine") != (self.lr_floor is not None):
            raise ValueError("lr_floor goes with the cosine schedule, and only with it")
        return self

    def factor(self, step: int, steps: int) -> float:
        """Return the rate factor of the update at 0-based ``step`` of a ``steps``-step run (originally ``lr_mult``).

        Warmup ``min(1, (step + 1) / warmup_steps)``; the cosine runs from 1 at step 0 to ``lr_floor`` at step
        ``steps - 1``.
        """
        warm = min(1.0, (step + 1) / max(1, self.warmup_steps))
        if self.lr_floor is None:
            return warm
        t = min(1.0, max(0.0, step / max(1, steps - 1)))
        return warm * (self.lr_floor + (1.0 - self.lr_floor) * 0.5 * (1.0 + math.cos(math.pi * t)))


class CheckpointSpec(Config):
    """Resume-state policy (``writes.checkpoint``); committed checkpoints follow ``RunConfig.checkpoint_every``.

    Attributes:
        resume_every: steps between resume-state saves.
        on_signal: also save resume state on SIGUSR1, then exit.
    """

    resume_every: int = Field(ge=1)
    on_signal: bool = True


class RunConfig(StageConfig):
    """A training run.

    Attributes:
        name: run name (records are keyed by the digest, not the name).
        connector: the connector config (embedded).
        receiver: the receiver config (embedded).
        encoder: the encoder recipe the connector reads (embedded; the step-0 inputs check its input against it).
        domain: the domain at a fixed data version.
        ridge: target name → ridge-init artifact digest, one per ridge target of the connector's tokens.
        source_fit: the recipe the ridges were fit on, when the run's ``encoder`` adds one declared option to it
            (``build.init_state.SourceFit``: the original start kept for a run on corrected inputs; added field).
        embeddings: digests of every embedding artifact the run reads (train pools, eval sets, variants).
        training_set: digest of the screened training set (``build.zero_shot``): objects and the text fields
            streams may use.
        mixture: the stream mixture.
        steps: optimizer steps (an upper bound for walltime-bound runs).
        batch: examples per step, fixed (measured beforehand with the batch measurement tool).
        micro_batch: examples per ``TrainingBackend.loss`` call (the batch is split in order; the gradient is the
            whole batch's ``Σ w·ℓ / Σ w``); None runs the batch as one call. Part of the identity: bf16 rounding and
            expert routing make the split change the numbers.
        seed: data-order seed (the connector's own seed governs initialization).
        device_class: the declared training hardware (e.g. ``8xH200``); the run refuses another kind or topology of
            hardware, not another accelerator model (:meth:`DeviceClass.require`), and records the one it used.
        optimizer: optimizer settings.
        checkpoints: checkpoint policy.
        eval_suite: the in-job suite (embedded), or None; a scoring suite (capability ``score``) that reads no
            upstream run, since the training backend scores and no generation record exists mid-run. Where it cites
            the run's domain, it cites the run's data version of it.
        eval_every: steps between in-job evaluations (each also at step 0 and the end).
        checkpoint_every: steps between committed checkpoints (each also at step 0 and the end); ``eval_every`` must be
            a multiple of it, so every evaluation scores a committed checkpoint. None: checkpoints follow
            ``eval_every`` (added field).
        plan: the batch plan the training backend scores in-job evaluations with (part of the identity).
        plugins: extra modules to import (third-party streams, kinds, backends).
        checkpointing: the training backend's activation checkpointing (``kimi-ep``: recompute each layer in backward),
            handed to it as its ``checkpointing`` option; None leaves the backend's default and its ``--option``
            (added field; the paper runs set false).
    """

    name: str
    connector: ConnectorSpec
    receiver: ReceiverSpec
    encoder: EncoderRecipe
    domain: DomainSpec
    ridge: dict[str, ArtifactDigest] = Field(default_factory=dict)
    source_fit: SourceFit | None = added_field(default=None)
    embeddings: tuple[ArtifactDigest, ...] = Field(min_length=1)
    training_set: ArtifactDigest
    mixture: tuple[MixtureEntry, ...] = Field(min_length=1)
    steps: int = Field(ge=1)
    batch: int = Field(ge=1)
    micro_batch: int | None = Field(default=None, ge=1)
    seed: int = 0
    device_class: DeviceClass
    optimizer: OptimizerSpec
    checkpoints: CheckpointSpec
    eval_suite: SuiteSpec | None = None
    eval_every: int | None = Field(default=None, ge=1)
    plan: BatchPlan = BatchPlan()
    plugins: tuple[str, ...] = ()
    checkpointing: bool | None = added_field(default=None)
    checkpoint_every: int | None = added_field(default=None, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _one_call(cls, data: Any) -> Any:
        """``micro_batch == batch`` is the one-call run: stored as None, so both spellings have one identity."""
        if (
            isinstance(data, dict)
            and data.get("micro_batch") is not None
            and data.get("micro_batch") == data.get("batch")
        ):
            return {**data, "micro_batch": None}
        return data

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        total = sum(m.share for m in self.mixture)
        if not math.isclose(total, 1.0):
            raise ValueError(f"run {self.name!r}: mixture shares sum to {total}, not 1")
        if self.micro_batch is not None and self.micro_batch > self.batch:
            raise ValueError(f"run {self.name!r}: micro_batch {self.micro_batch} exceeds the batch {self.batch}")
        if (self.eval_suite is None) != (self.eval_every is None):
            raise ValueError(f"run {self.name!r}: eval_suite and eval_every go together")
        if self.checkpoint_every and self.eval_every and self.eval_every % self.checkpoint_every:
            raise ValueError(
                f"run {self.name!r}: eval_every {self.eval_every} is not a multiple of checkpoint_every "
                f"{self.checkpoint_every}"
            )
        suite = self.eval_suite
        if suite is not None and (suite.capability != "score" or suite.upstream_kind is not None):
            raise ValueError(
                f"run {self.name!r}: eval_suite {suite.id!r} must score on the training backend and read no upstream "
                f"run (capability {suite.capability!r}, upstream_kind {suite.upstream_kind!r})"
            )
        if suite is not None:
            # the in-job evaluation reads the run's embeddings, which the loader compares with the run's domain
            other = [d.data for d in suite.domains if d.name == self.domain.name and d.data != self.domain.data]
            if other:
                raise ValueError(
                    f"run {self.name!r}: eval_suite {suite.id!r} cites {self.domain.name} data {other[0][:12]}, "
                    f"the run {self.domain.data[:12]}: an in-job suite scores on the run's version of its domain"
                )
        # the step-0 inputs are consistent (ridge targets, receiver, encoder, read layers, chosen write tokens and
        # layers): a run with pending or wrong read layers, or pending writes, is refused here, at load, not at step 0
        # in a GPU job
        self.init_spec()
        return self

    def init_spec(self) -> InitSpec:
        """Return the initial-state inputs of this run: the trainer's step 0 is ``build.init_state`` on them."""
        return InitSpec(
            connector=self.connector,
            receiver=self.receiver,
            encoder=self.encoder,
            ridge=self.ridge,
            source_fit=self.source_fit,
        )


class AchievedSpec(StageConfig):
    """The publication of a stopped run's achieved checkpoints (the ``achieved`` stage, ``run.achieved``).

    Its own computation, never the run's: the record kind is ``achieved`` and the config is this one, so a later
    completion of the run is the first execution of the run's computation and pins under the run's destination, while
    this config's destination (e.g. ``connectors/kimi-k2.6/esmc-6b-mlp4-staggered``) names what the publication made.

    Attributes:
        run: the run config (embedded) whose run directory is published; its digest names the directory.
        device_class: ``cpu``: a publication opens no backend and trains nothing.
    """

    run: RunConfig
    device_class: DeviceClass

    @model_validator(mode="after")
    def _on_cpu(self) -> AchievedSpec:
        if self.device_class is not DeviceClass.CPU:
            raise ValueError(f"publication of run {self.run.name!r}: device_class must be cpu (it trains nothing)")
        return self
