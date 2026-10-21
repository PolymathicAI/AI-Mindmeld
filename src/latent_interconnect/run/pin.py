"""Pin an existing record's outputs: adopt a computation deliberately, without recomputing it (not a stage).

``python -m latent_interconnect.run.pin NAME --record <record id> [--output OUT] [--repin REASON]``

A stage's ``--pin`` refuses names that already name other targets, so a rerun whose outputs differ (non-deterministic
GPU execution) is recorded but not pinned. This command adopts such an execution afterwards, by its record id:

- without ``--output`` it makes the pins ``--pin NAME`` would have made (``run._main.stage_pins``); with
  ``--output OUT`` it pins ``NAME`` to that one output;
- the pins go through ``Workspace.pin_all``: never overwritten unless ``--repin REASON`` declares the re-pin, and
  every change is in the pins history with the record id;
- without ``--output`` it is ``run._main.adopt``, as a stage's ``--pin``: with several executions of the computation,
  the adopted one is declared canonical with the re-pin's reason, so a config citing ``NAME@record`` resolves to it;
  a re-pin keeps the stage (``NAME@record`` never moves to a record of another kind).

It is the deliberate adoption and restore route, so it does not check ``NAME`` against the configs' declared
destinations (``config.StageConfig.destination``): restoring a missing alias from its record is what the site's
existence check recommends, and that check reports a name pinned to a record its
declared producer did not make as not produced.

``python -m latent_interconnect.run.pin --retire REASON NAME... [--configs DIR]`` retires dead names (:func:`retire`):
it refuses any name a config under ``DIR`` (default ``configs``, every file, preserved fixtures included) cites, and
otherwise removes the names through ``Workspace.unpin_all`` (a history line each, ``new`` null, with the reason).

Exit codes: 0, 2 (usage: an unknown record or output, an empty reason, several executions without a reason, a name to
retire that is not pinned), 3 (a pin conflict without ``--repin``, a re-pin to another stage's record, or a name to
retire that a config cites).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Collection, Sequence
from pathlib import Path

from latent_interconnect.config import Pin
from latent_interconnect.config_index import cited_pins
from latent_interconnect.run._main import PIN_REFUSED, AmbiguousExecutionError, adopt, stage_pins
from latent_interconnect.workspace import PinConflictError, Workspace

__all__ = ["CitedPinError", "main", "retire"]


class CitedPinError(ValueError):
    """A name to retire is still cited by a config."""


def retire(ws: Workspace, names: Collection[str], reason: str, configs: Path) -> None:
    """Retire pinned ``names`` (``Workspace.unpin_all``) unless a config file under ``configs`` cites one.

    Raises:
        CitedPinError: naming each cited name and the files citing it; nothing is retired.
        ValueError: if ``configs`` is not a directory of config files that cite pins (a missing or wrong root would
            check nothing), or as ``Workspace.unpin_all``; nothing is retired.
    """
    citing = {}
    if Path(configs).is_dir():
        citing = {path: set(cited_pins(path)) for path in sorted(Path(configs).rglob("*.yaml"))}
    if not any(citing.values()):
        raise ValueError(f"--configs {configs}: no config there cites a pin, so no citation could be checked")
    cited: dict[str, list[str]] = {}
    for path, pins in citing.items():
        for pin in pins & set(names):
            cited.setdefault(pin, []).append(str(path))
    if cited:
        raise CitedPinError(f"still cited, not retired: {cited}")
    ws.unpin_all(names, reason)


def main(argv: Sequence[str] | None = None) -> int:
    """``... NAME --record <record id> [--output OUT] [--repin REASON]`` or ``... --retire REASON NAME...``."""
    parser = argparse.ArgumentParser(prog="python -m latent_interconnect.run.pin")
    parser.add_argument("name", nargs="+", help="the pin name (as a stage's --pin NAME); with --retire, the names")
    parser.add_argument("--record", help="the record id of the execution to adopt")
    parser.add_argument("--output", help="pin NAME to this one output instead of making --pin's names")
    parser.add_argument("--repin", metavar="REASON", help="re-pin names that already name other targets")
    parser.add_argument("--retire", metavar="REASON", help="retire the names (refused while a config cites one)")
    parser.add_argument("--configs", type=Path, default=Path("configs"), help="the configs --retire checks")
    args = parser.parse_args(argv)
    if args.retire is not None:
        return _retire_main(parser, args)
    if args.record is None or len(args.name) != 1:
        parser.error("adopting takes one NAME and --record")
    args.name = args.name[0]
    if args.repin is not None and not args.repin.strip():
        parser.error("--repin states the reason for the re-pin")
    ws = Workspace.from_env()
    try:
        record = ws.records.get(args.record)
    except (LookupError, ValueError) as err:
        parser.error(f"--record {args.record!r}: {err}")
    if args.output is not None and args.output not in record.outputs:
        parser.error(f"record {args.record[:12]} has no output {args.output!r}; it has {sorted(record.outputs)}")
    try:
        if args.output is None:
            adopt(ws, args.name, record, args.repin)
            pins = stage_pins(args.name, record)
        else:
            pins = {args.name: Pin(target="artifact", sha256=record.outputs[args.output].sha256)}
            ws.pin_all(pins, repin=args.repin, record=args.record)
    except PinConflictError as err:
        sys.stderr.write(f"pin: not pinned: {err} (--repin REASON)\n")
        return PIN_REFUSED
    except AmbiguousExecutionError as err:  # adopt's rule: several executions need a declaration
        parser.error(str(err))
    for name, pin in sorted(pins.items()):
        sys.stdout.write(f"  {name}: {pin.target} {pin.sha256}\n")
    return 0


def _retire_main(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.record or args.output or args.repin:
        parser.error("--retire takes names only")
    try:
        retire(Workspace.from_env(), args.name, args.retire, args.configs)
    except CitedPinError as err:
        sys.stderr.write(f"pin: {err}\n")
        return PIN_REFUSED
    except ValueError as err:
        parser.error(str(err))
    for name in sorted(args.name):
        sys.stdout.write(f"  retired {name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
