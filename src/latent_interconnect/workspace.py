"""The workspace: the one place that builds stores, pins and contexts for a stage run.

Every entry point (``latent_interconnect.run``) and the tiny end-to-end test build their collaborators here, so
there is one wiring path:

- ``paths``: data roots from the environment;
- ``artifacts``, ``records``: the stores;
- ``pins``: the name → :class:`~latent_interconnect.config.Pin` map (an artifact digest or a record computation key)
  that configs cite with ``{$pin: name}``, stored as one YAML file in the data root, updated under a lock when a stage
  commits a named output, and checked against the stores. A pin is never overwritten: pinning a name to another
  target is refused unless it is a declared re-pin (a reason), and every change is appended to the pins history
  (:data:`PINS_HISTORY_NAME`: time, who, name, old and new target, reason); pinning a name to its own target is a
  no-op;
- :meth:`Workspace.record`: the one place a stage run's record is written (provenance captured from the code
  checkout that holds this package; a dirty tree is refused by the record store, so entry-point runs need a committed
  tree). A worktree's subprocess runs import the worktree's own code only if its ``src`` comes first on the path
  (an editable install of that worktree, or ``PYTHONPATH=<worktree>/src``); the e2e tests call ``main(argv)``
  in-process;
- :meth:`Workspace.purpose`: a record's purpose follows its inputs (``anchor`` and ``monitor`` are contagious, and an
  input with no producing record counts as ``anchor``: every ``result`` input enters through a stage);
- :meth:`Workspace.domain` and :meth:`Workspace.item_set`: a cited domain or item set → the object, through the owning
  module's one loader (``open_domain``, ``load_item_set``); cited digests become references with ``ArtifactStore.ref``;
- :meth:`Workspace.scoring_context`: the :class:`~latent_interconnect.evaluation.suites.SuiteContext` a runner or a scan
  scores in; :meth:`Workspace.suite_context` builds a suite run's through it (its upstream generation run resolved and
  checked), and the ``scan`` stage a scan's;
- :meth:`Workspace.suite_inputs`, :meth:`Workspace.scan_inputs`, :meth:`Workspace.study_inputs` and
  :meth:`Workspace.read_scan_inputs`: the artifacts a suite run, a scan run, a layer-selection study or a read-layer
  scan reads (its record's inputs and its purpose's sources), and :meth:`Workspace.suite_records`, the records a suite
  run reads;
"""

from __future__ import annotations

import getpass
import os
import platform
import time
import uuid
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar

import yaml

from latent_interconnect.artifacts.records import Purpose, Record, RecordRef, RecordStore, strictest_purpose
from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore, IntegrityError
from latent_interconnect.config import (
    Config,
    Pin,
    PinTarget,
    RecordKey,
    canonical_json,
    is_sha256,
    load_config,
    load_yaml,
)
from latent_interconnect.data.items import ItemSet, load_item_set
from latent_interconnect.domains.base import Domain, DomainSpec, open_domain
from latent_interconnect.evaluation.generations import GENERATIONS_KIND, read_generations
from latent_interconnect.evaluation.layer_selection import LayerSelectionStudy, ScanRunConfig
from latent_interconnect.evaluation.read_scan import ReadLayerScanSpec
from latent_interconnect.evaluation.scan_states import WrittenScan
from latent_interconnect.evaluation.suites import SuiteContext, SuiteRunConfig, Upstream, UpstreamError, check_upstream
from latent_interconnect.paths import Paths, code_checkout
from latent_interconnect.provenance import DeviceClass, Provenance, code_state
from latent_interconnect.receivers import open_generation, open_renderer, open_scoring
from latent_interconnect.receivers.base import GenerationBackend, GenerationPlan, ReceiverSpec, ScoringBackend
from latent_interconnect.receivers.dictionary import StateEdit
from latent_interconnect.receivers.kernel import BatchPlan

__all__ = ["PINS_HISTORY_NAME", "PinConflictError", "Workspace"]

PINS_HISTORY_NAME = "pins.history.jsonl"
"""The append-only history of every pin change, next to the pins file (one JSON object per line)."""


class PinConflictError(ValueError):
    """A pin name already names another target, and the new pin is not a declared re-pin."""


C = TypeVar("C", bound=Config)


class _CitedPins(Mapping[str, Pin]):
    """The pins map, remembering every name a config looked up."""

    def __init__(self, pins: Mapping[str, Pin]) -> None:
        self.pins = pins
        self.used: dict[str, Pin] = {}

    def __getitem__(self, name: str) -> Pin:
        pin = self.pins[name]
        self.used[name] = pin
        return pin

    def __iter__(self) -> Iterator[str]:
        return iter(self.pins)

    def __len__(self) -> int:
        return len(self.pins)

    def __contains__(self, name: object) -> bool:
        return name in self.pins


@dataclass(frozen=True)
class Workspace:
    """Stores, pins and context factories over one data root.

    Attributes:
        paths: the data roots.
        artifacts: the artifact store.
        records: the record store.
        pins_file: the YAML file holding the pins map.
        repo: the code checkout that holds this package (its commit is what provenance records); None for an
            installed package.
    """

    paths: Paths
    artifacts: ArtifactStore
    records: RecordStore
    pins_file: Path
    repo: Path | None

    @classmethod
    def from_env(cls) -> Workspace:
        """Build the workspace from ``LIP_*`` environment variables (``paths.Paths.from_env``)."""
        paths = Paths.from_env()
        return cls(
            paths,
            ArtifactStore(paths.artifacts),
            RecordStore(paths.records),
            paths.data / "lip" / "pins.yaml",
            code_checkout(),
        )

    def pins(self) -> dict[str, Pin]:
        """Return the pins map (empty if no pin was written yet)."""
        if not self.pins_file.is_file():
            return {}
        data = load_yaml(self.pins_file.read_text()) or {}
        return {str(k): Pin.model_validate(v) for k, v in data.items()}

    def pin(
        self, name: str, digest: str, target: PinTarget, *, repin: str | None = None, record: str | None = None
    ) -> None:
        """Pin ``name`` to (``target``, ``digest``): :meth:`pin_all` with one pin."""
        self.pin_all({name: Pin(target=target, sha256=digest)}, repin=repin, record=record)

    def pin_all(self, pins: Mapping[str, Pin], *, repin: str | None = None, record: str | None = None) -> None:
        """Pin several names at once, all or none, after checking that each names an existing artifact or record.

        A name already pinned to the same target is left as it is. A name pinned to another target is refused unless
        ``repin`` gives the reason for a declared re-pin. Every change is appended to the pins history
        (``pins.history.jsonl`` next to the pins file): an id, time, who (user and host), name, old and new target,
        reason, and ``record``, the id of the record that produced the pinned target (the pinning record).

        Write order: the history line is appended first, then the pins file is replaced (an atomic rename). A history
        line is therefore a change that was decided; if the replacement fails, a line ``{"id": ..., "applied":
        false}`` follows it, so a reader takes a change as applied unless a later line with its id says otherwise. A
        process killed between the two leaves a decided change unapplied without that marker; the pins file is then
        the truth, and the pins file and history disagree only in that case.

        The pins file is read, updated and atomically replaced under a lock (an atomic ``mkdir``), so concurrent stages
        do not lose each other's pins. A process killed while holding the lock leaves the directory behind; once no
        stage is pinning, remove it by hand (the timeout error names it).

        Raises:
            ValueError: if a digest is not a sha256 hex digest or names nothing in the stores, or ``repin`` is empty.
            PinConflictError: if a name already names another target and ``repin`` is None.
            TimeoutError: if the lock is not acquired within a minute.
        """
        if repin is not None and not repin.strip():
            raise ValueError("a re-pin states its reason")
        for name, pin in pins.items():
            if not is_sha256(pin.sha256):
                raise ValueError(f"pin {name!r}: not a sha256 hex digest: {pin.sha256!r}")
            store = self.artifacts if pin.target == "artifact" else self.records
            if not store.exists(pin.sha256):
                raise ValueError(f"pin {name!r}: no {pin.target} {pin.sha256[:12]} in the stores")
        with self._pins_lock():
            current = self.pins()
            changed = {name: pin for name, pin in sorted(pins.items()) if current.get(name) != pin}
            conflicts = {name: current[name] for name in changed if name in current}
            if conflicts and repin is None:
                named = ", ".join(f"{n!r} names {p.target} {p.sha256[:12]}" for n, p in conflicts.items())
                raise PinConflictError(f"pins are never overwritten: {named}; a re-pin is declared with its reason")
            if changed:
                self._replace_pins(current, current | changed, changed, repin, record)

    def unpin_all(self, names: Collection[str], reason: str) -> None:
        """Retire pinned names at once, all or none (``run.pin --retire``, which refuses a name a config cites).

        Under :meth:`pin_all`'s lock and write order, with a history line per name whose ``new`` is null and whose
        ``reason`` is ``reason``. The artifacts and records stay; only the names go.

        Raises:
            ValueError: if ``reason`` is empty, ``names`` is empty, or a name is not pinned.
            TimeoutError: if the lock is not acquired within a minute.
        """
        if not reason.strip():
            raise ValueError("a retirement states its reason")
        if not names:
            raise ValueError("no names to retire")
        with self._pins_lock():
            current = self.pins()
            missing = sorted(set(names) - set(current))
            if missing:
                raise ValueError(f"not pinned: {missing}")
            gone: dict[str, Pin | None] = dict.fromkeys(sorted(names))
            kept = {k: v for k, v in current.items() if k not in gone}
            self._replace_pins(current, kept, gone, reason, None)

    @contextmanager
    def _pins_lock(self) -> Iterator[None]:
        self.pins_file.parent.mkdir(parents=True, exist_ok=True)
        lock = self.pins_file.with_name(self.pins_file.name + ".lock")
        deadline = time.monotonic() + 60
        while True:
            try:
                lock.mkdir()
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"{lock} is held; if no stage is pinning, a killed one left it: rmdir it"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            lock.rmdir()

    def _replace_pins(
        self,
        current: Mapping[str, Pin],
        merged: Mapping[str, Pin],
        changed: Mapping[str, Pin | None],
        reason: str | None,
        record: str | None,
    ) -> None:
        """Log ``changed`` (None: retired), then atomically replace the pins file with ``merged`` (:meth:`pin_all`)."""
        tmp = self.pins_file.with_name(f"{self.pins_file.name}.{uuid.uuid4().hex}")
        tmp.write_text(yaml.safe_dump({k: v.model_dump() for k, v in sorted(merged.items())}, sort_keys=False))
        ids = self._log_pins(current, changed, reason, record)
        try:
            tmp.rename(self.pins_file)
        except OSError:
            self._append_history([{"id": i, "applied": False} for i in ids])
            tmp.unlink(missing_ok=True)
            raise

    def _log_pins(
        self, current: Mapping[str, Pin], changed: Mapping[str, Pin | None], repin: str | None, record: str | None
    ) -> list[str]:
        when = datetime.now(UTC).isoformat(timespec="seconds")
        who = f"{getpass.getuser()}@{platform.node()}"
        lines = [
            {
                "id": uuid.uuid4().hex,
                "at": when,
                "by": who,
                "name": name,
                "old": current[name].model_dump() if name in current else None,
                "new": None if pin is None else pin.model_dump(),
                "reason": repin if name in current else None,
                "record": record,
            }
            for name, pin in changed.items()
        ]
        self._append_history(lines)
        return [str(line["id"]) for line in lines]

    def _append_history(self, lines: Sequence[Mapping[str, object]]) -> None:
        text = "".join(canonical_json(line) + "\n" for line in lines)
        fd = os.open(self.pins_file.with_name(PINS_HISTORY_NAME), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, text.encode())
        finally:
            os.close(fd)

    def provenance(self, packages: Mapping[str, str], devices: tuple[str, ...]) -> Provenance:
        """Capture the provenance of the current process against this workspace's code checkout."""
        return Provenance.capture(self.repo, dict(packages), devices)

    def record(
        self,
        kind: str,
        config: Config,
        inputs: Mapping[str, ArtifactRef],
        outputs: Mapping[str, ArtifactRef],
        provenance: Provenance,
        *,
        records: Mapping[str, RecordKey] | None = None,
        purpose: Purpose = "result",
    ) -> Record:
        """Write the record of one stage run (provenance finished now) and return it; ``purpose`` is written as is."""
        rec = Record(
            kind=kind,
            purpose=purpose,
            config_sha=config.digest(),
            config=config.model_dump(mode="json"),
            inputs={k: RecordRef(sha256=v.sha256, kind=v.kind) for k, v in inputs.items()},
            input_records=dict(records or {}),
            outputs={k: RecordRef(sha256=v.sha256, kind=v.kind) for k, v in outputs.items()},
            provenance=provenance.finish(),
        )
        self.records.put(rec)
        return rec

    def load(self, path: Path, model: type[C]) -> C:
        """Load a config with this workspace's pins."""
        return self.load_cited(path, model)[0]

    def load_cited(self, path: Path, model: type[C]) -> tuple[C, dict[str, Pin]]:
        """Load a config with this workspace's pins; also return the pins it cited, by name (a stage records them)."""
        cited = _CitedPins(self.pins())
        return load_config(path, model, cited), dict(sorted(cited.used.items()))

    def purpose(self, declared: Purpose, inputs: Mapping[str, ArtifactRef], records: Mapping[str, Record]) -> Purpose:
        """Return a record's purpose: the strictest class of the declared one and its inputs'.

        The order is anchor > monitor > result. Input records count by their class (``Record.purpose_class``: a
        legacy stored value is classified, never rewritten): the executions the stage read (``StageResult.records``),
        not a fresh canonical lookup, so a canonical choice declared during the run cannot change it. Input artifacts
        count by the records that produced them (``RecordStore.produced_by``). An artifact no record produced counts
        as anchor-made: unknown lineage is never a result, so every ``result`` input enters through a stage.

        Raises:
            ValueError: if a purpose is unknown (``artifacts.records.purpose_class``).
        """
        found: list[str] = [declared]
        found += [rec.purpose for rec in records.values()]
        for ref in inputs.values():
            made_by = self.records.produced_by(ref.sha256)
            found += [r.purpose for r in made_by] if made_by else ["anchor"]
        return strictest_purpose(found)

    def input_problems(self, path: Path, model: type[Config]) -> list[str]:
        """Return what fails the provenance check for the inputs a config cites by pin; empty when nothing does.

        Each cited artifact must verify against its manifest (``ArtifactStore.verify``) and have been produced by a
        stage record of class ``result`` (a stored ``paper`` included, ``Record.purpose_class``): made by the stages
        from stage-made inputs (:meth:`purpose`). Each cited
        record must exist. No comparison with the old output is required. Digests a config writes out
        instead of citing a pin are not seen (configs cite upstream outputs by pin).
        """
        _, cited = self.load_cited(path, model)
        problems = []
        for name, pin in cited.items():
            if pin.target == "record":
                if not self.records.exists(pin.sha256):
                    problems.append(f"{name}: no record {pin.sha256[:12]}")
            else:
                try:
                    self.artifacts.verify(self.artifacts.ref(pin.sha256))
                except (IntegrityError, FileNotFoundError) as err:
                    problems.append(f"{name}: does not verify: {err}")
                    continue
                made_by = self.records.produced_by(pin.sha256)
                if not any(r.purpose_class() == "result" for r in made_by):
                    found = sorted({f"{r.kind}/{r.purpose}" for r in made_by}) or ["no record"]
                    problems.append(f"{name}: not produced by the new code from new inputs ({', '.join(found)})")
        return problems

    def item_set(self, digest: str) -> ItemSet:
        """Return the item set a config cites by digest (``data.items.load_item_set``; its kind is checked)."""
        return load_item_set(self.artifacts, self.artifacts.ref(digest))

    def domain(self, spec: DomainSpec) -> Domain:
        """Return the domain a config cites (``domains.base.open_domain`` on this workspace's store)."""
        return open_domain(spec, self.artifacts)

    def suite_inputs(self, run: SuiteRunConfig) -> dict[str, ArtifactRef]:
        """Return the artifacts a suite run reads, by input name.

        Names: ``item_set``, ``domain/<name>`` per domain, ``checkpoint/<arg>`` and ``inputs/<arg>/<i>`` per connector
        argument, and ``upstream/generations``, the upstream run's generations, when the run cites one. The ``suite``
        stage records them, and they set the run's purpose (:meth:`purpose`).

        Raises:
            UpstreamError: if the suite's upstream record has no canonical execution.
        """
        out = {"item_set": self.artifacts.ref(run.suite.item_set)}
        out |= {f"domain/{d.name}": self.artifacts.ref(d.data) for d in run.suite.domains}
        for arg, digest in sorted(run.checkpoints.items()):
            out[f"checkpoint/{arg}"] = self.artifacts.ref(digest)
            out |= {f"inputs/{arg}/{i}": self.artifacts.ref(d) for i, d in enumerate(run.inputs[arg])}
        record = self._upstream_record(run)
        if record is not None:
            generations = record.outputs.get(GENERATIONS_KIND)
            if generations is None:
                raise UpstreamError(
                    f"suite {run.suite.id!r}: upstream record {record.kind!r} has no generations output"
                )
            out["upstream/generations"] = self.artifacts.ref(generations.sha256)
        return out

    def scan_inputs(self, run: ScanRunConfig) -> dict[str, ArtifactRef]:
        """Return the artifacts a scan run reads, by input name.

        Names: ``item_set``, ``domain/<name>``, ``states`` and ``embeddings/<i>`` (a ridge source's). The ``scan``
        stage records them, and they set the run's purpose (:meth:`purpose`).
        """
        return {name: self.artifacts.ref(digest) for name, digest in _written_inputs(run.scan).items()}

    def study_inputs(self, study: LayerSelectionStudy) -> dict[str, ArtifactRef]:
        """Return the artifacts a layer-selection study reads, by input name.

        Names: its injection scan's (as :meth:`scan_inputs`) under ``injection/``. The ``study`` stage records them,
        and they set the record's purpose.
        """
        return {f"injection/{n}": self.artifacts.ref(d) for n, d in _written_inputs(study.injection).items()}

    def read_scan_inputs(self, spec: ReadLayerScanSpec) -> dict[str, ArtifactRef]:
        """Return the artifacts a read-layer scan reads, by input name.

        Names: ``domain/<name>``, ``objects``, ``embeddings/<i>`` and, with variant probes, ``variants/objects`` and
        ``variants/embeddings/<i>``, and with an appended extraction ``append/embeddings/<i>``. The ``read_scan`` stage
        records them.
        """
        out = {f"domain/{spec.domain.name}": spec.domain.data, "objects": spec.objects}
        out |= {f"embeddings/{i}": d for i, d in enumerate(spec.embeddings)}
        if spec.variants is not None:
            out["variants/objects"] = spec.variants.objects
            out |= {f"variants/embeddings/{i}": d for i, d in enumerate(spec.variants.embeddings)}
        if spec.append is not None:
            out |= {f"append/embeddings/{i}": d for i, d in enumerate(spec.append.embeddings)}
        return {name: self.artifacts.ref(digest) for name, digest in out.items()}

    def suite_context(self, run: SuiteRunConfig, options: Mapping[str, str] | None = None) -> SuiteContext:
        """Build a suite run's context (:meth:`scoring_context` for the suite's capability, receiver and plans).

        The purpose is the strictest of the run's declared one and its inputs' (:meth:`suite_inputs`,
        :meth:`purpose`), so a run on inputs of unknown lineage is an anchor run. Hardware is checked
        by the stage before anything loads, and by the runner. A run citing an upstream record gets it resolved to
        its canonical execution, checked (``evaluation.suites.check_upstream``) and its generations table read, before
        any backend is built.

        Raises:
            ValueError: if the receiver names no backend for the suite's capability.
            UpstreamError: if the upstream record is missing or does not match the run.
        """
        upstream = self._upstream(run)
        purpose = self.purpose("result", self.suite_inputs(run), self.suite_records(run))
        return self.scoring_context(
            run.receiver,
            run.suite.capability,
            plan=run.plan,
            generation_plan=run.generation_plan,
            device_class=run.device_class,
            purpose=purpose,
            options=options,
            upstream=upstream,
            edit=run.edit,
        )

    def scoring_context(
        self,
        receiver: ReceiverSpec,
        capability: Literal["score", "generate"],
        *,
        plan: BatchPlan | None,
        device_class: DeviceClass,
        purpose: Purpose,
        generation_plan: GenerationPlan | None = None,
        options: Mapping[str, str] | None = None,
        upstream: Upstream | None = None,
        edit: StateEdit | None = None,
    ) -> SuiteContext:
        """Build the context a runner or scan scores in: backend, renderer, resolvers, plan, device class and purpose.

        The backend is the receiver's for ``capability`` (``score`` or ``generate``), constructed with the configured
        ``plan`` or ``generation_plan`` (identity; the other capability's is None) and ``options``, execution options
        as text (e.g. a weights directory) that the backend's constructor binds (``config.bind_options``). The
        context carries both plans, which the runners check against the backend's, and
        ``upstream``, a suite run's resolved upstream generation run. ``purpose`` is the caller's resolved one
        (:meth:`purpose`). ``edit`` is a scoring run's state edit (``SuiteRunConfig.edit``; identity), given to
        the scoring backend only when set.

        Raises:
            ValueError: if the receiver names no backend for ``capability``, or ``edit`` is given for generation or
                to a backend that applies none.
        """
        if capability not in receiver.backends:
            raise ValueError(f"receiver {receiver.name!r} has no {capability!r} backend")
        backend: ScoringBackend | GenerationBackend
        if capability == "score":
            backend = open_scoring(receiver, plan=plan, edit=edit, options=options)
        elif edit is not None:
            raise ValueError("a state edit applies to scoring only; a generation context takes none")
        else:
            backend = open_generation(receiver, generation_plan=generation_plan, options=options)
        return SuiteContext(
            artifacts=self.artifacts,
            records=self.records,
            receiver=receiver,
            backend_name=receiver.backends[capability],
            backend=backend,
            renderer=open_renderer(receiver),
            domain=self.domain,
            device_class=device_class,
            purpose=purpose,
            items=self.item_set,
            plan=plan,
            code=code_state(self.repo),
            generation_plan=generation_plan,
            upstream=upstream,
        )

    def _upstream_record(self, run: SuiteRunConfig) -> Record | None:
        key = run.upstream
        if key is None:
            return None
        try:
            return self.records.canonical(key)
        except LookupError as err:
            raise UpstreamError(
                f"suite {run.suite.id!r}: upstream record {key[:12]} has no canonical execution"
            ) from err

    def _upstream(self, run: SuiteRunConfig) -> Upstream | None:
        record = self._upstream_record(run)
        if record is None or run.upstream is None:
            return None
        config = check_upstream(run, record)
        out = record.outputs[GENERATIONS_KIND]
        table = read_generations(self.artifacts, ArtifactRef(out.sha256, out.kind))
        return Upstream(key=run.upstream, run=config, generations=table)

    def suite_records(self, run: SuiteRunConfig) -> dict[str, Record]:
        """Return the executions a suite run reads, by input name: ``upstream`` (canonical) when it cites one.

        Raises:
            UpstreamError: if the upstream record has no canonical execution.
        """
        record = self._upstream_record(run)
        return {} if record is None else {"upstream": record}


def _written_inputs(spec: WrittenScan) -> dict[str, str]:
    """Return the artifacts a receiver scan reads, by name.

    Names: ``item_set``, ``domain/<name>``, ``states`` and ``embeddings/<i>``.
    """
    out = {"item_set": spec.item_set, f"domain/{spec.domain.name}": spec.domain.data, "states": spec.states}
    return out | {f"embeddings/{i}": d for i, d in enumerate(spec.embeddings)}
