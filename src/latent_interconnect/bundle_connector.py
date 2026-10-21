"""Package a trained connector checkpoint as a release bundle, the format ``release_download`` installs.

    python -m latent_interconnect.bundle_connector \
        --checkpoint <checkpoint dir> --run-config <run config> --license <id> --out <repo dir> \
        [--name NAME] [--label LABEL] [--note TEXT] [--demo] [--card-url URL] [--source-commit SHA] [--install-pin PIN]

``--checkpoint`` is a committed checkpoint artifact (``MANIFEST.json``, ``connector.json``, ``weights.safetensors``),
given as its directory or, with ``--pin NAME`` instead, by a pin of the data root ``LIP_DATA``. ``--run-config``
is the run config the connector was trained with (:class:`~latent_interconnect.training.RunConfig`); it names the
receiver, the specialist's encoder recipe and the domain. Its ``$pin`` references are not resolved: only the
identities it embeds are read (:func:`run_stand_ins`).

``bundle.json``'s ``source`` says where the checkpoint came from. Given ``--pin``, the record that pinned it
(:func:`pinning_record`: the pins history's latest line that set the pin to this checkpoint, a record listing it as an
output) fills ``pinning_record`` (its id), ``run_record_key`` (its computation key) and, unless ``--source-commit``
gives one, ``research_commit`` (the code commit its provenance cites). Without a pin, or when the data root holds no
such record, they stay null unless ``--source-commit`` sets the commit.

``--out`` is laid out as a model repository of bundles, as the released one is: ``<out>/<name>/`` holds the checkpoint
as stored and its ``bundle.json`` (format ``aimm-bundle/1``: receiver and specialist identities, the read and write
layout, the connector digest, each file's sha256 and size, the pin it installs and the license), and
``<out>/connectors.json`` is the index (format ``aimm-index/1``) with each bundle's sha256. Packaging a second
connector into the same ``--out`` adds it to the index.

A bundle packaged here also states, at its top level, what a user needs to run it without the run config:

- ``connector_kind`` (:func:`connector_layout`): the map ``kind`` (``affine``, ``mlp``, ``xattn``, ``xblock`` or a
  plugin's), its ``kind_params``, the ``input`` it reads (``pooled`` or ``residues``), ``layer0``, the number of slot
  ``tokens`` and each token's ``write_layers``;
- ``plugins``: the modules the connector, its encoder recipe and its receiver import (empty for core code only);
- ``receiver_config``: the receiver config the connector was trained with (a
  :class:`~latent_interconnect.receivers.base.ReceiverSpec`), which ``release_download`` installs as
  ``<dest>/lip/receivers/<receiver>.yaml`` for ``examples/ask.py``.

``--note`` adds a free-text ``note`` and ``--demo`` the field ``demo: true``, marking the bundle as a software demo.

``--license`` is required and is this connector's own license, which ``bundle.json`` records and ``release_download``
prints at install. The project's Apache-2.0 license covers the project's code only, not a connector's weights: set the
license your weights may be released under, given the licenses of the receiver, the specialist model and the data
they were trained from. Users check and accept each connector's license before use.

Upload ``<out>`` as it is to a Hugging Face model repository;
``python -m latent_interconnect.release_download --repo <user/repo> --revision <commit> --connector <name>`` installs
from it.

Before writing, the checkpoint is checked: every file against its manifest, the manifest kind, the stored spec's digest
against the manifest's and the run config's connector, the receiver name, depth and width against the run config's
receiver, and the stored encoder recipe against the run config's. Any missing field or mismatch stops packaging with
:class:`BundleError` before anything is written.

``release/stage_bundles.py`` builds the released bundles with the same functions (:func:`copy_artifact_dir`,
:func:`connector_bundle`, :func:`write_bundle`, :func:`write_index`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from latent_interconnect.artifacts.records import Record
from latent_interconnect.artifacts.store import MANIFEST_NAME, Manifest, sha256_file
from latent_interconnect.canonical import is_sha256
from latent_interconnect.config import PIN_KEY, Pin, load_config, load_resolved
from latent_interconnect.config_index import cited_pins, stand_in_pins
from latent_interconnect.plugins import load_plugins
from latent_interconnect.training import RunConfig
from latent_interconnect.writes.checkpoint import CHECKPOINT_KIND
from latent_interconnect.writes.connector import ConnectorSpec
from latent_interconnect.writes.kinds import MAP_KINDS
from latent_interconnect.writes.layers import entries

if TYPE_CHECKING:
    from latent_interconnect.workspace import Workspace

__all__ = [
    "BUNDLE_FORMAT",
    "INDEX_FORMAT",
    "BundleError",
    "check_identity",
    "connector_bundle",
    "connector_layout",
    "copy_artifact_dir",
    "main",
    "package",
    "pinning_record",
    "run_stand_ins",
    "write_bundle",
    "write_index",
]

BUNDLE_FORMAT = "aimm-bundle/1"
INDEX_FORMAT = "aimm-index/1"
_BUNDLE_FILE = "bundle.json"
_SPEC_FILE = "connector.json"
_HEADER_KEYS = ("spec", "encoder", "depth", "d_in", "d_model", "step")


class BundleError(ValueError):
    """A checkpoint or its run config lacks what a bundle records, or they disagree."""


def copy_artifact_dir(src: Path, dest: Path) -> tuple[Manifest, dict[str, Any]]:
    """Check an artifact directory against its manifest, then copy its files as stored; return it and the entry.

    Every source file is checked (present, size and sha256) before ``dest`` is created, so a missing or changed file
    writes nothing. Each copy is checked again. The entry is the bundle's record of the artifact (id, kind, per-file
    sha256 and size, total bytes); the caller adds its ``path`` in the bundle. A target already holding the right bytes
    is not copied again.

    Raises:
        BundleError: if ``src`` has no manifest, lacks a file it lists, a file differs from it, or a copy does.
    """
    if not (src / MANIFEST_NAME).is_file():
        raise BundleError(f"{src} has no {MANIFEST_NAME}: give a committed artifact's directory")
    manifest = Manifest.load(src / MANIFEST_NAME)
    for rel, (sha, size) in sorted(manifest.files.items()):
        if not (src / rel).is_file():
            raise BundleError(f"{src}/{rel} is missing; {MANIFEST_NAME} lists it")
        if (src / rel).stat().st_size != size or sha256_file(src / rel) != sha:
            raise BundleError(f"{src.name}/{rel} differs from {MANIFEST_NAME} (size or sha256): a changed checkpoint")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / MANIFEST_NAME, dest / MANIFEST_NAME)
    for rel, (sha, size) in sorted(manifest.files.items()):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not (target.is_file() and target.stat().st_size == size and sha256_file(target) == sha):
            shutil.copyfile(src / rel, target)
        got = sha256_file(target)
        if got != sha:
            raise BundleError(f"{src.name}/{rel}: copy has sha256 {got}, manifest {sha}")
    entry = {
        "sha256": manifest.sha256(),
        "kind": manifest.kind,
        "files": {rel: {"sha256": sha, "bytes": size} for rel, (sha, size) in sorted(manifest.files.items())},
        "bytes": sum(size for _, size in manifest.files.values()),
    }
    return manifest, entry


def check_identity(name: str, header: Mapping[str, Any], manifest: Manifest, run: RunConfig, receiver: str) -> None:
    """Check that a stored checkpoint is the run config's connector, built for ``receiver``.

    Raises:
        BundleError: if the stored spec's digest, the manifest's and the run config's connector differ, or the
            checkpoint was built for another receiver.
    """
    spec = header["spec"]
    digest = ConnectorSpec.model_validate(spec).digest()
    if not (digest == manifest.meta.get("connector_sha") == run.connector.digest()):
        raise BundleError(f"{name}: the stored connector, its manifest and the run config's connector differ")
    if spec["receiver"] != receiver:
        raise BundleError(f"{name}: built for receiver {spec['receiver']!r}, not {receiver!r}")


def connector_layout(spec: Mapping[str, Any], depth: int) -> dict[str, Any]:
    """Return a stored connector spec's kind and write layout, the bundle's ``connector_kind`` (module docstring).

    ``spec`` is a checkpoint's ``connector.json`` ``spec``, ``depth`` the receiver's. The spec's plugins are imported
    so that a plugin kind's input is known.
    """
    cs = ConnectorSpec.model_validate(spec)
    load_plugins(cs.plugins)
    per_token: list[list[int]] = [[] for _ in cs.tokens]
    for e in entries(cs.layers, cs.k, depth):
        per_token[e.token].append(e.layer)
    return {
        "kind": cs.kind,
        "kind_params": dict(cs.kind_params),
        "input": MAP_KINDS.get(cs.kind).input_kind.value,
        "layer0": cs.layer0,
        "tokens": cs.k,
        "write_layers": per_token,
    }


def connector_bundle(
    *,
    name: str,
    label: str,
    header: Mapping[str, Any],
    manifest: Manifest,
    artifact: Mapping[str, Any],
    run: RunConfig,
    receiver: Mapping[str, str],
    pin: str,
    source: Mapping[str, Any],
    operational_only: bool,
    license: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a connector's ``bundle.json`` body from its stored checkpoint and the run config that trained it.

    ``header`` is the checkpoint's ``connector.json``, ``manifest`` its manifest, ``artifact`` its entry
    (:func:`copy_artifact_dir`, with ``path``), ``run`` the run config, ``receiver`` the receiver's name, model id and
    revision, and ``pin`` the pin name the bundle installs. ``extra`` adds top-level fields (e.g. a card URL).

    Raises:
        BundleError: as :func:`check_identity`.
    """
    check_identity(name, header, manifest, run, receiver["name"])
    spec = header["spec"]
    sha = manifest.sha256()
    bundle = {
        "format": BUNDLE_FORMAT,
        "type": "connector",
        "name": name,
        "label": label,
        "connector": spec["name"],
        "step": header["step"],
        "receiver": dict(receiver),
        "specialist": {
            "name": run.encoder.name,
            "model_id": run.encoder.model_id,
            "revision": run.encoder.revision,
            "domain": run.domain.name,
        },
        "extraction": {
            "recipe": header["encoder"],
            "read_layers": spec["input"]["layers"],
            "combine": spec["input"]["combine"],
            "pooling": header["encoder"]["pooling"],
            "d_in": header["d_in"],
        },
        "write": {
            "tokens": spec["tokens"],
            "layers": spec["layers"],
            "alpha": spec["alpha"],
            "kind": spec["kind"],
            "receiver_depth": header["depth"],
            "d_model": header["d_model"],
        },
        "connector_sha": manifest.meta["connector_sha"],
        "checkpoint_manifest_sha256": sha,
        "artifacts": [dict(artifact)],
        "pins": {pin: {"target": "artifact", "sha256": sha}},
        "source": dict(source),
        "operational_only": operational_only,
        "license": license,
    }
    return bundle | dict(extra or {})


def write_bundle(bundle_dir: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``bundle_dir/bundle.json`` and return the bundle's index entry (its sha256 and size)."""
    path = bundle_dir / _BUNDLE_FILE
    path.write_text(json.dumps(bundle, indent=1, sort_keys=True) + "\n")
    entry = {
        "name": bundle["name"],
        "path": bundle_dir.name,
        "bundle_sha256": sha256_file(path),
        "bytes": sum(a["bytes"] for a in bundle["artifacts"]),
    }
    if bundle["type"] == "connector":
        entry["checkpoint"] = bundle["checkpoint_manifest_sha256"]
    return entry


def write_index(path: Path, key: str, entries: Sequence[Mapping[str, Any]], license: str) -> None:
    """Write an index (``connectors.json``, ``data.json``) listing ``entries`` under ``key``."""
    body = {"format": INDEX_FORMAT, "license": license, key: list(entries)}
    path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")


def run_stand_ins(run_config: Path) -> dict[str, Pin]:
    """Return the pins to load a run config with outside a store: a stand-in for each pin it cites (``stand_in_pins``).

    One exception keeps the config valid: when its ``source_fit.ridges`` lists ridges by their digests while its
    ``ridge`` cites them by pin, each such ridge pin stands in as one of those listed digests (in sorted order, the
    same number of each), since :class:`~latent_interconnect.build.init_state.SourceFit` checks that the two name the
    same ridges and a stand-in digest would match none. The packager reads no ridge digest, and the config's text,
    hence the run's identity under the store's pins, is unchanged. Unequal numbers are left as they are, so the
    config's own check refuses them.
    """
    pins = stand_in_pins(cited_pins(run_config))
    tree = load_resolved(run_config, unknown_pins="keep")
    fit = tree.get("source_fit") if isinstance(tree, dict) else None
    ridge = tree.get("ridge") if isinstance(tree, dict) else None
    if not (isinstance(fit, dict) and isinstance(ridge, dict) and isinstance(fit.get("ridges"), list)):
        return pins

    def pin_of(node: Any) -> str | None:
        return str(node[PIN_KEY]) if isinstance(node, dict) and set(node) == {PIN_KEY} else None

    listed = fit["ridges"]
    listed_pins = {p for n in listed if (p := pin_of(n)) is not None}
    literal = sorted(
        {n for n in listed if isinstance(n, str) and is_sha256(n)} - {v for v in ridge.values() if isinstance(v, str)}
    )
    unlisted = sorted({p for v in ridge.values() if (p := pin_of(v)) is not None} - listed_pins)
    if literal and len(literal) == len(unlisted):
        pins |= {p: Pin(target="artifact", sha256=d) for p, d in zip(unlisted, literal, strict=True)}
    return pins


def pinning_record(ws: Workspace, pin: str, sha256: str) -> Record | None:
    """Return the record that pinned ``pin`` to the artifact ``sha256`` in workspace ``ws``, or None if none is found.

    The pins history's latest line that set ``pin`` to ``sha256`` names the record; it counts only if it is in the
    record store and lists ``sha256`` among its outputs.
    """
    from latent_interconnect.workspace import PINS_HISTORY_NAME  # noqa: PLC0415  (as main's Workspace import)

    history = ws.pins_file.parent / PINS_HISTORY_NAME
    if not history.is_file():
        return None
    record_id = None
    for line in history.read_text().splitlines():
        entry = json.loads(line)
        new = entry.get("new")
        if entry.get("name") == pin and isinstance(new, dict) and new.get("sha256") == sha256 and entry.get("record"):
            record_id = str(entry["record"])
    if record_id is None:
        return None
    try:
        record: Record = ws.records.get(record_id)
    except LookupError:
        return None
    return record if any(o.sha256 == sha256 for o in record.outputs.values()) else None


def _require(mapping: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    missing = [k for k in keys if mapping.get(k) is None]
    if missing:
        raise BundleError(f"{where} lacks {', '.join(missing)}")


def _read_stored(checkpoint: Path) -> tuple[Manifest, dict[str, Any]]:
    """Return a stored checkpoint's manifest and ``connector.json``, each checked for the fields a bundle records."""
    for f in (MANIFEST_NAME, _SPEC_FILE):
        if not (checkpoint / f).is_file():
            raise BundleError(f"{checkpoint} has no {f}: give a committed checkpoint's directory")
    manifest = Manifest.load(checkpoint / MANIFEST_NAME)
    if manifest.kind != CHECKPOINT_KIND:
        raise BundleError(f"{checkpoint} is a {manifest.kind!r} artifact, not a checkpoint")
    _require(manifest.meta, ("connector", "connector_sha", "step"), f"{checkpoint}/{MANIFEST_NAME} meta")
    header: dict[str, Any] = json.loads((checkpoint / _SPEC_FILE).read_text())
    _require(header, _HEADER_KEYS, f"{checkpoint}/{_SPEC_FILE}")
    if header["step"] != manifest.meta["step"]:
        raise BundleError(f"{checkpoint}: {_SPEC_FILE} step {header['step']}, manifest step {manifest.meta['step']}")
    return manifest, header


def package(
    checkpoint: Path,
    run_config: Path,
    out: Path,
    *,
    license: str,
    name: str | None = None,
    label: str | None = None,
    note: str | None = None,
    demo: bool = False,
    card_url: str | None = None,
    source_commit: str | None = None,
    pin: str | None = None,
    record: Record | None = None,
) -> Path:
    """Package one checkpoint as ``out/<name>/`` and add it to ``out/connectors.json``; return the bundle directory.

    The bundle installs the checkpoint as the pin ``pin``, by default ``<run destination>/step_NNNNNN`` (the name a
    stage's run gives it). ``note`` and ``demo`` add the bundle's ``note`` and ``demo: true`` (module docstring).
    ``record``, the record that produced the checkpoint (:func:`pinning_record`), fills ``source``'s record fields and,
    when ``source_commit`` is None, its commit.

    Raises:
        BundleError: for missing metadata, a checkpoint file missing or differing from its manifest, a mismatch
            between the checkpoint and the run config, or a ``name`` already in ``out`` for another checkpoint.
            These are all checked before ``out`` is written to, so none of them leaves a partial bundle; a failure
            while copying (a disk error, or a copy that does not hash to its manifest) can leave files in
            ``out/<name>/`` but no ``bundle.json`` and no index entry for them.
    """
    if not license.strip():
        raise BundleError("--license is empty: name the license the weights are released under")
    checkpoint, out = Path(checkpoint), Path(out)
    manifest, header = _read_stored(checkpoint)
    if record is not None and not any(o.sha256 == manifest.sha256() for o in record.outputs.values()):
        raise BundleError(f"record {record.record_id()[:12]} did not produce the checkpoint {manifest.sha256()[:12]}")
    run = load_config(run_config, RunConfig, run_stand_ins(run_config))
    rec = run.receiver
    if (header["depth"], header["d_model"]) != (rec.depth, rec.d_model):
        raise BundleError(
            f"{checkpoint}: built for depth {header['depth']} x {header['d_model']}, "
            f"receiver {rec.name!r} is {rec.depth} x {rec.d_model}"
        )
    if header["encoder"] != run.encoder.identity():
        raise BundleError(f"{checkpoint}: its stored encoder recipe is not the run config's {run.encoder.name!r}")
    for what, value in (("receiver", rec.revision), ("encoder", run.encoder.revision)):
        if value in ("", "pending"):
            raise BundleError(f"{run_config}: the {what} revision is {value!r}; pin the exact revision")
    spec_name = str(header["spec"]["name"])
    name = name or f"{rec.name}__{spec_name}"
    if "/" in name or name in ("", ".", ".."):
        raise BundleError(f"bundle name {name!r} must be one path component")
    if pin is None:
        if run.destination is None:
            raise BundleError(f"{run_config} names no destination: give the pin to install it as (--install-pin)")
        pin = f"{run.destination}/step_{int(header['step']):06d}"
    sha = manifest.sha256()
    index_path = out / "connectors.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {"connectors": []}
    others = [e for e in index["connectors"] if e["name"] != name]
    old = next((e for e in index["connectors"] if e["name"] == name), None)
    if (old is not None and old.get("checkpoint") != sha) or (
        old is None and (out / name).exists()
    ):  # never mix two checkpoints' files in one directory
        raise BundleError(f"{out / name} already holds another bundle: choose another --name or --out")
    check_identity(str(checkpoint), header, manifest, run, rec.name)
    layout = connector_layout(header["spec"], int(header["depth"]))
    bundle_dir = out / name
    _, art = copy_artifact_dir(checkpoint, bundle_dir)
    art["path"] = "."
    bundle = connector_bundle(
        name=name,
        label=label or f"{rec.name} x {spec_name}",
        header=header,
        manifest=manifest,
        artifact=art,
        run=run,
        receiver={"name": rec.name, "model_id": rec.model_id, "revision": rec.revision},
        pin=pin,
        source={
            "research_commit": source_commit or (record.provenance.commit if record is not None else None),
            "run_config": run.name,
            "run_destination": run.destination,
            "checkpoint_pin": pin,
            "pinning_record": record.record_id() if record is not None else None,
            "run_record_key": record.computation_key() if record is not None else None,
        },
        operational_only=False,
        license=license,
        extra={
            "connector_kind": layout,
            "plugins": sorted({*header["spec"].get("plugins", ()), *run.encoder.plugins, *rec.plugins}),
            "receiver_config": rec.model_dump(mode="json"),
            **({"card_url": card_url} if card_url else {}),
            **({"note": note} if note else {}),
            **({"demo": True} if demo else {}),
        },
    )
    entries = sorted([*others, write_bundle(bundle_dir, bundle)], key=lambda e: e["name"])
    licenses = {json.loads((out / e["path"] / _BUNDLE_FILE).read_text())["license"] for e in entries}
    write_index(index_path, "connectors", entries, licenses.pop() if len(licenses) == 1 else "per bundle")
    return bundle_dir


def main(argv: Sequence[str] | None = None) -> int:
    """The command line (module docstring)."""
    parser = argparse.ArgumentParser(
        prog="python -m latent_interconnect.bundle_connector",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="the committed checkpoint's directory")
    src.add_argument("--pin", help="or the checkpoint's pin in the data root LIP_DATA (e.g. <destination>/step_002000)")
    parser.add_argument(
        "--run-config",
        dest="run_config",
        type=Path,
        required=True,
        help="the run config it was trained with",
    )
    parser.add_argument("--out", type=Path, required=True, help="the repository tree to write (index + bundles)")
    parser.add_argument(
        "--license",
        required=True,
        help="this connector's own license (e.g. an SPDX id); the project's Apache-2.0 covers its code only",
    )
    parser.add_argument("--name", help="the bundle's name (default <receiver>__<connector>)")
    parser.add_argument("--label", help="a human-readable label (default '<receiver> x <connector>')")
    parser.add_argument("--note", help="a free-text note recorded in bundle.json (e.g. what the connector is for)")
    parser.add_argument("--demo", action="store_true", help="mark the bundle as a software demo")
    parser.add_argument("--card-url", help="the URL of the connector's model card")
    parser.add_argument(
        "--source-commit",
        help="the commit of the code that trained it (default with --pin: the one the pinning record cites)",
    )
    parser.add_argument(
        "--install-pin", help="the pin the bundle installs (default: --pin, else <run destination>/step_NNNNNN)"
    )
    args = parser.parse_args(argv)
    checkpoint, record = args.checkpoint, None
    if args.pin is not None:
        from latent_interconnect.workspace import Workspace  # noqa: PLC0415

        ws = Workspace.from_env()
        pins = ws.pins()
        if args.pin not in pins:
            parser.error(f"no pin {args.pin!r} in {ws.pins_file}")
        checkpoint = ws.artifacts.path(ws.artifacts.ref(pins[args.pin].sha256))
        record = pinning_record(ws, args.pin, pins[args.pin].sha256)
    try:
        bundle_dir = package(
            checkpoint,
            args.run_config,
            args.out,
            license=args.license,
            name=args.name,
            label=args.label,
            note=args.note,
            demo=args.demo,
            card_url=args.card_url,
            source_commit=args.source_commit,
            pin=args.install_pin or args.pin,
            record=record,
        )
    except BundleError as e:
        sys.stderr.write(f"error: {e}\n")
        return 2
    bundle = json.loads((bundle_dir / _BUNDLE_FILE).read_text())
    sys.stdout.write(f"bundle {bundle['name']} -> {bundle_dir} ({bundle['checkpoint_manifest_sha256'][:12]})\n")
    sys.stdout.write(f"index {args.out / 'connectors.json'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
