"""The shared ``main`` of every entry point: parse, build the workspace, load, run, record, report, pin.

**Stage options.** ``--option NAME=VALUE`` (repeatable) sets an execution option of the stage: where and how fast it
runs (devices, memory, pass size, a weights directory, an interpreter, kernels). An execution option may change
floating-point precision or rounding, which is why every one is recorded and a training run refuses to resume under
incompatible ones (``training.trainer.Trainer._check_execution``); it never changes the objective, the data, the
trainable parameters or the write rule, which are the config's. This paragraph is that rule's home. Options are
bound to parameters by ``config.bind_options``, applied twice: here, to the stage function's own keyword parameters, and
in ``receivers`` (``_open``, where every backend is built), to the backend constructor's parameters for the options
the stage hands on through its ``**backend_options``. Each value is validated against its parameter's annotation;
refused, before any config loads, are a parameter the caller supplies (``config``, ``ws``), a field of the stage's
config, and a backend parameter that is part of a result's identity (``receivers.base.IDENTITY_PARAMETERS``: the plans,
on every stage). :func:`stage_main` records the options' text in the provenance's ``execution`` as
``option.<NAME>``, next to the stage's own static facts (``StageResult.execution``).

**Destination.** ``--destination NAME`` overrides the config's ``destination`` (execution metadata, outside the
identity) for the run and names the outputs and the record ``NAME`` exactly as ``--pin NAME`` would with a config
declaring that destination, so ``--pin`` is then not needed (:func:`check_pin`). The record keeps the override in the
provenance's ``execution``: ``destination.used`` (``NAME``) and ``destination.declared`` (the config's destination,
absent when it declares none).
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from latent_interconnect.artifacts.records import CanonicalChoice, Purpose, Record, ResolvedRecords
from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.config import DESTINATION, Config, OptionError, Pin, StageConfig, bind_options
from latent_interconnect.provenance import DeviceClass, Extension, Segment, visible_devices
from latent_interconnect.receivers.base import IDENTITY_PARAMETERS, Capability, ReceiverSpec
from latent_interconnect.workspace import PinConflictError, Workspace

__all__ = [
    "ADOPT_AMBIGUOUS",
    "PIN_REFUSED",
    "SCORING_PACKAGES",
    "AmbiguousExecutionError",
    "ExecutionTarget",
    "PinDestinationError",
    "Stage",
    "StageResult",
    "adopt",
    "check_pin",
    "require_hardware",
    "stage_main",
    "stage_options",
    "stage_pins",
    "visible_devices",
]

C = TypeVar("C", bound=StageConfig)
SCORING_PACKAGES = (
    "torch",
    "numpy",
    "scipy",
    "transformers",
    "tokenizers",
    "safetensors",
    "accelerate",
    "compressed_tensors",
    "vllm",
)
"""The libraries behind a receiver readout and its statistics, by module name: the ``scan`` and ``suite`` stages record
the versions of those their process loaded (``provenance.loaded_packages``), which depend on the backend (``hf``:
transformers, tokenizers, safetensors, accelerate; Kimi's: compressed_tensors or vllm)."""
PIN_REFUSED = 3
"""Exit code of a stage run whose record and outputs are committed but whose ``--pin`` names already name other
targets (pins are never overwritten; ``--repin REASON`` declares a re-pin)."""


@dataclass(frozen=True)
class StageResult:
    """What a stage function returns.

    Attributes:
        outputs: output name → committed artifact. Multi-output stages name them by what they are: ``train``
            ``step_<step, 6 digits>`` per committed checkpoint, ``study`` ``injection/numbers`` and
            ``injection/scores``.
        inputs: input name → the artifacts it read (recorded for lineage and purpose inheritance).
        records: input name → the exact executions it read (e.g. an analysis's inputs), resolved once by the stage
            (``artifacts.records.ResolvedRecords``): the record lists their computation keys (``input_records``) and
            their ``record_id`` (``Provenance.input_executions``), and inherits their purpose from these same
            executions, never from a canonical lookup made after the run.
        packages: package versions that affect the stage's numerics, each stage's own choice: a fixed dict of the
            modules it imports, or, where they depend on the backend (``scan``, ``suite``: :data:`SCORING_PACKAGES`;
            ``analysis``), ``provenance.loaded_packages`` of its list. Empty for a stage that reports none.
        devices: accelerator names used.
        execution: how the stage executed, recorded in the provenance and never in an identity: static facts
            only, namely the backend's ``execution()`` (the key list in ``receivers.base``) or the extractor's
            fingerprint, and the values a stage resolved for its options (e.g. a default memory budget and its
            source). ``stage_main`` adds the options as given (``option.<NAME>``). The configured batch plan is not
            here: it is in the config, so in the identity.
        purpose: the declared purpose (``result``, ``monitor`` or ``anchor``); the record's purpose is the stricter of
            this and its inputs' (``Workspace.purpose``).
        resumed: the earlier jobs of a resumed execution (``Provenance.resumed``; the ``train`` stage's, from its
            segment ledger), recorded so the execution's cost covers every job.
        extended_from: the source of an extended training run (``Provenance.extended_from``; the ``train`` stage's
            ``extend_from`` option), or None. The source record is also listed in :attr:`records`.
    """

    outputs: Mapping[str, ArtifactRef]
    inputs: Mapping[str, ArtifactRef] = field(default_factory=dict)
    records: Mapping[str, Record] = field(default_factory=dict)
    packages: Mapping[str, str] = field(default_factory=dict)
    devices: tuple[str, ...] = ()
    execution: Mapping[str, str] = field(default_factory=dict)
    purpose: Purpose = "result"
    resumed: tuple[Segment, ...] = ()
    extended_from: Extension | None = None


@dataclass(frozen=True)
class ExecutionTarget:
    """Where one stage run executes, read from its config: what a job checks before it claims anything.

    Attributes:
        device_class: the run's declared hardware (its config's ``device_class``), or None if it declares none.
        receiver: the receiver whose backend the run opens, or None if it opens none.
        capability: the backend it opens (``score``, ``generate``, ``capture``, ``train``); set with ``receiver``.
    """

    device_class: DeviceClass | None = None
    receiver: ReceiverSpec | None = None
    capability: Capability | None = None


@dataclass(frozen=True)
class Stage:
    """A stage as its entry point declares it (``latent_interconnect.run.<stage>.STAGE``).

    Site tooling reads it to check a job before it runs: the config class loads the job's config, ``target`` reads
    where the run executes from it, and the function's signature validates the job's options
    (:func:`stage_options`).

    Attributes:
        name: the stage name (the record kind).
        config: the stage's config class.
        run: the stage function, ``run(config, ws, **options)``.
        target: config → where the run executes; None for a stage that opens no backend and declares no hardware.
        outputs: the stage's output naming, a regular expression every output name matches in full (``train``: a
            ``step_`` and six digits, or ``log``); ``--pin D`` pins ``D/<output>`` (:func:`stage_pins`), so ``D/<rest>``
            names an output of the stage declared at ``D`` only if ``<rest>`` matches
            (``latent_interconnect.config_index``).
    """

    name: str
    config: type[StageConfig]
    run: Callable[..., StageResult | None]
    target: Callable[[Any], ExecutionTarget] | None = None
    outputs: str = field(kw_only=True)


def require_hardware(devices: tuple[str, ...], declared: DeviceClass | None) -> None:
    """The hardware gate of a stage that runs on devices, passed before anything loads.

    ``devices`` must be the ``declared`` class if the config declares one: its kind and topology, any accelerator
    model (``DeviceClass.matches``). Earlier executions of the config on other hardware no longer refuse it
    (an earlier refusal is lifted): the run is another execution of the computation, its record holds its own devices,
    and the first execution stays canonical (``artifacts.records``).

    Raises:
        WrongHardwareError: if ``devices`` are not the declared class.
    """
    if declared is not None:
        declared.require(devices)


def stage_options(
    raw: Mapping[str, str], run: Callable[..., StageResult | None], model: type[Config]
) -> dict[str, Any]:
    """Bind option texts to the stage function (the rule in the module docstring).

    Returns:
        The stage's own options, typed, and the texts it hands to its backend through ``**backend_options``.

    Raises:
        OptionError: as ``config.bind_options``, with the config's fields and ``IDENTITY_PARAMETERS`` refused.
    """
    refused = dict.fromkeys(model.model_fields, f"a field of {model.__name__}: identity, set it in the config")
    typed, rest = bind_options(run, raw, refused | IDENTITY_PARAMETERS)
    return {**typed, **rest}


def _pairs(pairs: Sequence[str]) -> dict[str, str]:
    raw: dict[str, str] = {}
    for pair in pairs:
        name, sep, text = pair.partition("=")
        if not sep or not name:
            raise OptionError(f"option {pair!r}: expected NAME=VALUE")
        if name in raw:
            raise OptionError(f"option {name!r} given twice")
        raw[name] = text
    return raw


class PinDestinationError(ValueError):
    """A ``--pin`` other than its run's destination (``--destination``, else ``StageConfig.destination``)."""


def check_pin(destination: str | None, pin: str | None, override: str | None = None) -> str | None:
    """Return the name a stage run pins under; refuse a ``--pin`` other than its destination, before anything runs.

    ``override`` is ``--destination``: it replaces the config's ``destination`` for the run. The run's destination is
    ``override`` if given, else ``destination``. The cases: neither ``--pin`` nor ``--destination`` pins nothing (there
    is no default); ``--destination`` alone pins under it, whatever the config declares; ``--pin`` must equal the run's
    destination when there is one, and a config that declares none (and no ``--destination``) takes any ``--pin``. A
    re-pin (``--repin``) needs ``--pin`` or ``--destination``, so it re-declares the destination's names and never
    renames. ``stage_main`` checks this before the stage runs, and a job renderer should check it, with the job's
    ``--destination`` as ``override``, when it renders a job.

    Returns:
        The pin name: ``override`` if given, else ``pin`` (None: nothing is pinned).

    Raises:
        PinDestinationError: if ``pin`` differs from ``override``, or, without ``override``, from ``destination``.
    """
    if override is not None:
        if pin is not None and pin != override:
            raise PinDestinationError(f"--pin {pin!r} is not --destination {override!r}")
        return override
    if pin is not None and destination is not None and pin != destination:
        raise PinDestinationError(f"--pin {pin!r} is not the config's destination {destination!r}")
    return pin


def _destination(text: str) -> str:
    """Parse ``--destination``: a destination by ``StageConfig``'s own rule (:data:`config.DESTINATION`)."""
    if not re.fullmatch(DESTINATION, text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a destination (pattern {DESTINATION})")
    return text


def _override(config: C, destination: str | None) -> tuple[C, dict[str, str]]:
    """Return the config a run uses with ``--destination`` (``None``: none given) and what its record notes of it.

    ``destination`` replaces the config's own (execution metadata: the identity and the record's config are unchanged).
    Noted, in the provenance's ``execution``: ``destination.used`` and, if the config declares one,
    ``destination.declared``.
    """
    if destination is None:
        return config, {}
    noted = {"destination.declared": config.destination} if config.destination is not None else {}
    noted["destination.used"] = destination
    return config.model_copy(update={"destination": destination}), noted


def stage_pins(name: str, record: Record) -> dict[str, Pin]:
    """Return the pins ``--pin NAME`` makes for a stage record: the one naming rule (the stage and ``run.pin``).

    ``NAME/<output>`` → each output (artifact pins), ``NAME`` too when there is exactly one output, and
    ``NAME@record`` → the record's computation key (a record pin).
    """
    pins = {f"{name}/{out}": Pin(target="artifact", sha256=ref.sha256) for out, ref in record.outputs.items()}
    if len(record.outputs) == 1:
        pins[name] = Pin(target="artifact", sha256=next(iter(record.outputs.values())).sha256)
    pins[f"{name}@record"] = Pin(target="record", sha256=record.computation_key())
    return pins


class AmbiguousExecutionError(ValueError):
    """A computation has several executions, and the one to adopt is neither declared canonical nor re-pinned."""


ADOPT_AMBIGUOUS = 2
"""Exit code of a stage run (or ``run.pin``) whose computation has several executions and no declaration of which one
its pins mean: the record and outputs stand, nothing is pinned, ``--repin REASON`` declares it."""


def adopt(ws: Workspace, name: str, record: Record, repin: str | None) -> None:
    """Pin a record under ``name`` as ``--pin`` does, and make ``NAME@record`` resolve to that execution.

    The names are :func:`stage_pins`', written by ``Workspace.pin_all`` (never overwritten unless ``repin`` declares
    the re-pin). A declared re-pin of a computation executed more than once declares this execution canonical with
    the same reason (``RecordStore.declare_canonical``), so ``NAME@record`` and ``NAME/<output>`` name the same
    execution. A re-pin keeps the stage: moving ``NAME@record`` to a record of another kind is refused. Without
    ``repin``, a computation executed more than once is adopted only if this execution is already its declared
    canonical one; otherwise ``NAME@record`` would not resolve, and nothing is pinned.

    Raises:
        PinConflictError: for a conflicting name without ``repin``, or a re-pin to a record of another kind.
        AmbiguousExecutionError: for a computation with several executions, none declared canonical as this one, and
            no ``repin``.
    """
    key = record.computation_key()
    executions = ws.records.executions(key)
    if repin is None and len(executions) > 1:
        try:
            declared = ws.records.canonical(key).record_id()
        except LookupError:
            declared = None
        if declared != record.record_id():
            raise AmbiguousExecutionError(
                f"computation {key[:12]} has {len(executions)} executions: NAME@record would not resolve to this one; "
                "--repin REASON declares it canonical"
            )
    prior = ws.pins().get(f"{name}@record")
    if repin is not None and prior is not None and prior.sha256 != key:
        before = ws.records.executions(prior.sha256)
        if not before or before[0].kind != record.kind:
            kind = before[0].kind if before else "unknown (no record)"
            raise PinConflictError(f"{name}@record names a {kind} record; a re-pin keeps the stage, not {record.kind}")
    ws.pin_all(stage_pins(name, record), repin=repin, record=record.record_id())
    if repin is not None and len(executions) > 1:
        ws.records.declare_canonical(CanonicalChoice(computation_key=key, record_id=record.record_id(), reason=repin))


def stage_main(argv: Sequence[str] | None, stage: str, model: type[C], run: Callable[..., StageResult | None]) -> int:
    """Run one stage from the command line; return the process exit code.

    The stage runs on the workspace with its record store wrapped in one ``ResolvedRecords``, so each record it reads by
    key is resolved once; the record's ``input_records``, ``Provenance.input_executions`` (their ``record_id``) and
    purpose come from those executions (``StageResult.records``), which must list every record it read (else
    RuntimeError, before any record is written).
    The stage's record is written by :meth:`Workspace.record` (a stage that returns None, a read-scan shard, writes
    none and exits 0). Printed: the record's computation key and each output
    digest. With ``--pin NAME``: ``NAME/<output>`` → each output (artifact pins), ``NAME`` too when there is exactly one
    output, and ``NAME@record`` → the computation key (a record pin), all or none; a name that already names another
    target is refused (:data:`PIN_REFUSED`) unless ``--repin REASON`` declares the re-pin, and so are the names of a
    computation run more than once whose execution is not declared (:data:`ADOPT_AMBIGUOUS`; :func:`adopt`). A
    ``--pin`` other than the config's declared destination is refused before the stage runs (:func:`check_pin`; exit 2).
    ``--destination NAME`` overrides the config's destination: the stage runs with it, the outputs and the record are
    pinned under ``NAME`` as with ``--pin NAME``, a ``--pin`` other than ``NAME`` is refused before the stage runs, and
    the provenance's ``execution`` records ``destination.used`` and ``destination.declared`` (the module docstring).
    ``--option NAME=VALUE`` passes an execution option to the stage (:func:`stage_options`). ``--purpose anchor``
    declares the record ``anchor``, of unknown lineage, whatever its inputs: every record built on its outputs inherits
    it (``Workspace.purpose``), so none is eligible as a result. ``--purpose result`` (the default) declares nothing
    stricter than the inputs give.

    Args:
        argv: arguments (``sys.argv[1:]`` when None).
        stage: the stage name (also the record kind).
        model: the stage's config class.
        run: the stage function, ``run(config, ws, **options)``.
    """
    parser = argparse.ArgumentParser(prog=f"python -m latent_interconnect.run.{stage}")
    parser.add_argument("--config", type=Path, required=True, help="the stage's YAML config")
    parser.add_argument("--pin", help="pin the outputs and the record under this name")
    parser.add_argument(
        "--destination",
        type=_destination,
        metavar="NAME",
        help="store the outputs and name them (and the record) NAME, in place of the config's destination; "
        "without this or --pin, the outputs are stored but not named",
    )
    parser.add_argument(
        "--repin",
        metavar="REASON",
        help="with --pin or --destination: re-pin names that already name another target",
    )
    parser.add_argument(
        "--option", action="append", default=[], metavar="NAME=VALUE", help="an execution option of the stage"
    )
    parser.add_argument(
        "--purpose",
        choices=["result", "anchor"],
        default="result",
        help="the record's lineage class: result (eligible as a result if every input is) or anchor (of unknown "
        "lineage, never a result, and inherited by every record built on its outputs)",
    )
    args = parser.parse_args(argv)
    if args.repin is not None and not args.pin and args.destination is None:
        parser.error("--repin re-declares the names of --pin or --destination; give one of them too")
    if args.repin is not None and not args.repin.strip():
        parser.error("--repin states the reason for the re-pin")
    try:
        raw = _pairs(args.option)
        options = stage_options(raw, run, model)
    except OptionError as err:
        parser.error(str(err))
    ws = Workspace.from_env()
    config, cited = ws.load_cited(args.config, model)
    try:
        pin_name = check_pin(config.destination, args.pin, args.destination)
    except PinDestinationError as err:
        parser.error(str(err))
    config, overridden = _override(config, args.destination)
    started = ws.provenance({}, ())
    # one ResolvedRecords per stage run: every record the stage reads by key is resolved once and listed
    resolved = ResolvedRecords.of(ws.records)
    result = run(config, dataclasses.replace(ws, records=resolved), **options)
    if result is None:
        return 0
    unlisted = set(resolved.resolved) - {rec.computation_key() for rec in result.records.values()}
    if unlisted:  # an unlisted record would escape the purpose inheritance (Workspace.purpose)
        raise RuntimeError(f"{stage}: read records it does not list in StageResult.records: {sorted(unlisted)}")
    execution = {**result.execution, **{f"option.{k}": v for k, v in sorted(raw.items())}, **overridden}
    provenance = started.model_copy(
        update={
            "packages": dict(result.packages),
            "devices": result.devices,
            "execution": execution,
            "pins": {name: pin.sha256 for name, pin in cited.items()},
            "resumed": result.resumed,
            "input_executions": {role: rec.record_id() for role, rec in sorted(result.records.items())},
            "extended_from": result.extended_from,
        }
    )
    declared: Purpose = "anchor" if args.purpose == "anchor" else result.purpose
    purpose = ws.purpose(declared, result.inputs, result.records)
    keys = {role: rec.computation_key() for role, rec in result.records.items()}
    record = ws.record(stage, config, result.inputs, result.outputs, provenance, records=keys, purpose=purpose)
    key = record.computation_key()
    sys.stdout.write(f"{stage}: record {key}\n")
    for out, ref in result.outputs.items():
        sys.stdout.write(f"  {out}: {ref.sha256}\n")
    if pin_name:
        try:
            adopt(ws, pin_name, record, args.repin)
        except PinConflictError as err:  # the record and outputs stand; only the names are refused
            sys.stderr.write(f"{stage}: not pinned: {err} (--repin REASON)\n")
            return PIN_REFUSED
        except AmbiguousExecutionError as err:
            sys.stderr.write(f"{stage}: not pinned: {err}\n")
            return ADOPT_AMBIGUOUS
    return 0
