"""Entry point: score connector checkpoints on a suite.

Per connector argument, the stage loads the checkpoint, reads the cited embedding artifacts through the connector's
own ``InputSpec``, checked against the encoder recipe the checkpoint records (``writes.checkpoint.checkpoint_encoder``:
its one source, never the embedding artifact's own metadata, which would check the artifact against itself;
and so for an appended extraction's recipe when the checkpoint records it),
and serves its writes as the receiver's backend for the suite's capability takes them
(its class's ``reads_banks``, ``receivers.base``): live (``LiveSource``) to a backend in this process, or from a write
bank (``BankSource``) to a backend that reads banks by reference. A bank holds the connector's writes for the objects
the suite scores through that argument (``scored_objects``: the item set's objects of every slot role the argument
fills, and the partners its variant conditions or a wild-type slot write; a prompt binds no other), less the objects
its inputs recorded as skipped (``unavailable_objects``), not every object its embedding artifacts hold
(``writes.sources.build_bank``); it is an output of the run, recorded as ``bank/<argument>``. The suite kind's
registered runner then scores in the context the workspace builds (``Workspace.suite_context``), which carries those
skipped objects with their recorded reasons; a kind that cannot leave their rows out is refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.data.items import ItemSet
from latent_interconnect.data.suites import FIELD_CONDITIONS, Condition
from latent_interconnect.encoders.loader import Consumed, InputLoader
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import EmbeddingReader
from latent_interconnect.evaluation.suites import SUITE_KINDS, SUITE_RECORD_KIND, SuiteRunConfig
from latent_interconnect.provenance import loaded_packages
from latent_interconnect.receivers import backend_class
from latent_interconnect.receivers.base import Capability, ReceiverSpec
from latent_interconnect.run._main import (
    SCORING_PACKAGES,
    ExecutionTarget,
    Stage,
    StageResult,
    require_hardware,
    stage_main,
    visible_devices,
)
from latent_interconnect.workspace import Workspace
from latent_interconnect.writes.checkpoint import checkpoint_encoder, load_connector
from latent_interconnect.writes.connector import Connector
from latent_interconnect.writes.sources import BankSource, LiveSource, WriteSource, build_bank

__all__ = [
    "BANK_OBJECTS",
    "STAGE",
    "CheckedConnector",
    "checked_connector",
    "main",
    "require_stated_padding",
    "scored_objects",
    "unavailable_objects",
]

BANK_OBJECTS = 256
"""Objects per connector call while a bank is built: it bounds host memory for per-residue inputs. Fixed, so a bank
is reproducible bit for bit; another split could move a float32 row in its last bits (the matrix kernels' blocking),
far below the bf16 residual stream's resolution."""


@dataclass(frozen=True)
class CheckedConnector:
    """A checkpoint's connector, checked against its use, with the encoder recipes the checkpoint records.

    Made by :func:`checked_connector`, the one assembly of a connector for inference: the ``suite`` stage and
    ``latent_interconnect.chat`` build every write source through it.

    Attributes:
        connector: the connector (on the host).
        checkpoint: the checkpoint artifact it was loaded from.
        encoder: the encoder recipe the checkpoint records (``writes.checkpoint.checkpoint_encoder``).
        append_encoder: the recipe of its appended extraction, recorded likewise, or None.
    """

    connector: Connector
    checkpoint: ArtifactRef
    encoder: EncoderRecipe
    append_encoder: EncoderRecipe | None

    def loader(self, readers: Sequence[EmbeddingReader], consumed: Consumed | None) -> InputLoader:
        """Return the loader of these embedding artifacts through the connector's input, checked against its recipes."""
        return InputLoader(
            self.connector.spec.input,
            readers,
            self.connector.input_kinds,
            encoder=self.encoder,
            append_encoder=self.append_encoder,
            consumed=consumed,
        )

    def source(
        self, loader: InputLoader, objects: Sequence[str], store: ArtifactStore, *, banked: bool
    ) -> tuple[WriteSource, ArtifactRef | None]:
        """Return the write source a backend takes and the bank built for it, if any.

        ``banked`` (the backend's ``reads_banks``): a write bank of ``objects`` (in :data:`BANK_OBJECTS` per connector
        call), committed to ``store``; else the live connector over the loader (``objects`` unused).
        """
        if not banked:
            return LiveSource(self.connector, loader.batch), None
        ids = list(objects)
        batches = [loader.batch(ids[i : i + BANK_OBJECTS]) for i in range(0, len(ids), BANK_OBJECTS)]
        bank = build_bank(self.connector, self.checkpoint, batches, store)
        return BankSource(store, bank), bank


def checked_connector(
    store: ArtifactStore, checkpoint: ArtifactRef, *, receiver: str, step: int | None, what: str
) -> CheckedConnector:
    """Load a checkpoint for inference on ``receiver`` at ``step``: the one rule (suite stage, chat).

    Args:
        store: the store holding the checkpoint.
        checkpoint: the checkpoint artifact.
        receiver: the receiver's name the connector must have been built for.
        step: the training step the use declares.
        what: the checkpoint, as refusals name it.

    Raises:
        ValueError: if the checkpoint is at another step, or its connector was built for another receiver (no
            checkpoint is used across receivers).
        MissingEncoderError: if the checkpoint records no encoder recipe, or none for the extraction its connector
            appends.
    """
    connector, at = load_connector(store, checkpoint)
    if at != step:
        raise ValueError(f"{what} is at step {at}, the run declares step {step}")
    if connector.spec.receiver != receiver:
        raise ValueError(f"{what} is a connector for {connector.spec.receiver!r}, the run uses {receiver!r}")
    encoder = EncoderRecipe.model_validate(checkpoint_encoder(store, checkpoint))
    append_encoder = None
    if connector.spec.input.append is not None:  # never the appended artifacts' own recipe
        append_encoder = EncoderRecipe.model_validate(checkpoint_encoder(store, checkpoint, "append_encoder"))
    return CheckedConnector(connector, checkpoint, encoder, append_encoder)


def require_stated_padding(receiver: ReceiverSpec, capability: Capability, stated: bool, what: str) -> None:
    """Refuse a fresh scoring run on a backend that pads whose plan does not state ``pad_every_batch`` (``stated``).

    The padded route is the intended default of every scoring suite on a backend that can pad (its class's
    ``pads_every_batch``: ``hf``). The stored field keeps its historical default (``BatchPlan.pad_every_batch`` is an
    added field, False, so every pinned config hashes as it always has); so an execution refuses a plan on such a
    backend that leaves the field out, rather than running it unpadded by omission. A config states it
    (``pad_every_batch: true``, which the ports write through their profile's ``plan``), or states ``false`` for a
    deliberately unpadded run. Reading a record never comes here: a historical config still loads and hashes unchanged.
    A backend that cannot pad (``kimi-stream``: no ``pads_every_batch``) takes no statement; ``receivers._open``
    refuses ``true`` for it. The one rule: the suite stage applies it first, and the job renderer before a job is
    written, so an omission never reaches a node.

    Args:
        receiver: the run's receiver.
        capability: the suite's capability (only a scoring plan has the field).
        stated: whether the run's plan states ``pad_every_batch`` (in its config, true or false).
        what: the run, for the message.

    Raises:
        ValueError: if the run scores on a backend that pads and its plan does not state ``pad_every_batch``.
    """
    if not stated and getattr(backend_class(receiver, capability), "pads_every_batch", False):
        raise ValueError(
            f"{what}: backend {receiver.backends[capability]!r} pads; its plan must state pad_every_batch (true: "
            "the padded route; false: a declared unpadded run)"
        )


def scored_objects(config: SuiteRunConfig, items: ItemSet) -> dict[str, list[str]]:
    """Return connector argument → the objects its writes are asked for, sorted.

    Every slot's role column it fills, and that role's partner fields (``<condition>.<role>``) of the variant
    conditions the suite scores and, for a wild-type slot, ``wt.<role>``. The runners bind only these (own, mismatched
    by derangement within the column, a partner, or none for text conditions), so a bank of them serves every prompt
    of the suite.
    """
    conditions = {c for combo in config.suite.conditions for c in combo.values()}
    out = {}
    for arg in config.checkpoints:
        ids: set[str] = set()
        for slot in (s for s in config.suite.slots if s.connector == arg):
            prefixes = {str(c) for c in conditions & FIELD_CONDITIONS if c != Condition.NAME_SUBSTITUTION}
            prefixes |= {"wt"} if slot.fill == "wt" else set()
            ids |= set(items.objects[slot.role])
            ids |= {str(o) for p in prefixes for o in items.fields[f"{p}.{slot.role}"]}
        out[arg] = sorted(ids)
    return out


def unavailable_objects(arg: str, missing: Mapping[str, str], readers: Sequence[EmbeddingReader]) -> dict[str, str]:
    """Return the scored objects argument ``arg``'s inputs lack → the reason its extractions recorded for skipping them.

    An object an extraction skipped (its ``skipped`` metadata, ``encoders.store``; MIMIC: ``length>10000``) is
    unavailable, with that reason: the runner leaves out the rows that write it (``evaluation.runners``). Any other
    missing object is a fault, never disclosed as unavailable.

    Args:
        arg: the connector argument (for the message).
        missing: object → why the loader cannot serve it (``InputLoader.missing`` of the scored objects).
        readers: the argument's embedding artifacts.

    Raises:
        ValueError: if a missing object has no recorded skip.
    """
    recorded = {o: str(r) for reader in readers for o, r in reader.meta.get("skipped", {}).items()}
    if unrecorded := sorted(set(missing) - set(recorded)):
        examples = {o: missing[o] for o in unrecorded[:3]}
        raise ValueError(
            f"argument {arg!r}: {len(unrecorded)} scored object(s) have no input and no recorded skip, e.g. {examples}"
        )
    return {o: recorded[o] for o in sorted(missing)}


def _consumed(config: SuiteRunConfig, ws: Workspace, items: ItemSet, arg: str) -> Consumed | None:
    """Return the values argument ``arg``'s inputs must be extracted from: its slots' domain at the suite's version.

    None for an argument no slot is bound to (it loads nothing).

    Raises:
        ValueError: if the argument's slots are not of exactly one of the suite's domains.
    """
    names = {items.roles[s.role] for s in config.suite.slots if s.connector == arg}
    if not names:
        return None
    specs = [d for d in config.suite.domains if d.name in names]
    if len(specs) != 1:
        raise ValueError(f"argument {arg!r}: its slots are of domains {sorted(names)}, {len(specs)} cited; one needed")
    (spec,) = specs
    values = ws.domain(spec).objects()["value"].to_dict()
    return Consumed(f"{spec.name} {spec.data[:12]}", values, ws.artifacts.root)


def _run(config: SuiteRunConfig, ws: Workspace, **backend_options: str) -> StageResult:
    """Output ``scores`` or ``generations`` (named by the artifact's kind) and a ``bank/<argument>`` per bank built.

    The backend's ``execution()`` is recorded, and the versions of the scoring libraries the process loaded,
    read after the run (``SCORING_PACKAGES``).

    Inputs ``Workspace.suite_inputs``, records ``Workspace.suite_records`` (the upstream generation run, if any).

    ``backend_options`` (execution options as text, e.g. a weights directory) reach the scoring backend's constructor
    through ``Workspace.suite_context``; the batch plan is the config's.

    The declared hardware is checked first, before any checkpoint or model loads; the runner checks it again for
    in-job contexts.

    Raises:
        ValueError: first, for a plan that does not state padding on a backend that pads
            (:func:`require_stated_padding`).
        WrongHardwareError: if the visible devices are not the config's ``device_class``.
        ValueError: if a checkpoint's step is not the config's ``step``, or its connector was built for another
            receiver than the suite's.
        MissingEncoderError: if a checkpoint records no encoder recipe, or none for the extraction its connector
            appends.
        ValueError: also if a banked argument's inputs lack an object the suite scores through it, or hold it as
            extracted from another value than the suite's domain has (``encoders.loader.Consumed``); or if an
            object its inputs recorded as skipped is scored by a kind that cannot leave its rows out.
    """
    if config.plan is not None:
        stated = "pad_every_batch" in config.plan.model_fields_set
        require_stated_padding(config.receiver, config.suite.capability, stated, f"suite {config.suite.id!r}")
    devices = visible_devices()
    require_hardware(devices, config.device_class)
    inputs = ws.suite_inputs(config)
    banked = backend_class(config.receiver, config.suite.capability).reads_banks
    items = ws.item_set(config.suite.item_set)
    scored = scored_objects(config, items)
    unavailable: dict[str, str] = {}
    sources: dict[str, WriteSource] = {}
    banks: dict[str, ArtifactRef] = {}
    for arg in sorted(config.checkpoints):
        checkpoint = inputs[f"checkpoint/{arg}"]
        what = f"checkpoint {arg!r}"
        checked = checked_connector(
            ws.artifacts, checkpoint, receiver=config.receiver.name, step=config.step, what=what
        )
        readers = [EmbeddingReader(ws.artifacts, inputs[f"inputs/{arg}/{i}"]) for i in range(len(config.inputs[arg]))]
        loader = checked.loader(readers, _consumed(config, ws, items, arg))
        lost = unavailable_objects(arg, loader.missing(scored[arg]), readers)
        unavailable |= lost
        ids = [o for o in scored[arg] if o not in lost]
        sources[arg], bank = checked.source(loader, ids, ws.artifacts, banked=banked)
        if bank is not None:
            banks[f"bank/{arg}"] = bank
    runner = SUITE_KINDS.get(config.suite.kind)
    if unavailable and not getattr(runner, "leaves_out_unavailable", False):
        raise ValueError(
            f"suite {config.suite.id!r}: kind {config.suite.kind!r} cannot leave out the rows of objects its inputs "
            f"skipped, e.g. {dict(list(unavailable.items())[:3])}"
        )
    ctx = replace(ws.suite_context(config, backend_options), unavailable=unavailable)
    output = runner(config.suite).run(ctx, sources, config.step)
    return StageResult(
        outputs={output.kind: output, **banks},
        inputs=inputs,
        records=ws.suite_records(config),
        packages=loaded_packages(SCORING_PACKAGES),
        devices=devices,
        execution=ctx.backend.execution(),
        purpose=ctx.purpose,
    )


STAGE = Stage(
    SUITE_RECORD_KIND,
    SuiteRunConfig,
    _run,
    lambda c: ExecutionTarget(c.device_class, c.receiver, c.suite.capability),
    outputs="scores|generations|bank/[^/]+",
)
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.suite --config <yaml> [--pin NAME [--repin REASON]] [--option K=V]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
