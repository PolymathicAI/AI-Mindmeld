"""Check the 19 staged connector bundles against this checkout's chat config and run configs (read only).

    python release/check_connector_identity.py STAGED_CONNECTORS_DIR [--no-hash]

``STAGED_CONNECTORS_DIR`` is the connectors tree :mod:`stage_bundles` writes (``OUT_DIR/connectors``), or a download
of the model repository at a revision. For each connector of ``configs/chat/all-connectors.yaml`` it finds the bundle
whose pins name the connector's checkpoint pin (the run's ``<destination>/step_NNNNNN``) and checks:

- the index's sha256 of ``bundle.json``;
- the manifest digest equals the artifact id, the bundle's ``checkpoint_manifest_sha256`` and the pin's sha256;
- the run config's connector digest equals ``MANIFEST.meta.connector_sha`` and the bundle's ``connector_sha``;
- the stored ``connector.json`` spec has the same digest;
- the spec's receiver and the checkpoint's step are the ones the chat config names;
- unless ``--no-hash``, every file's sha256 and size equal the manifest's.

It prints one line per connector and exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from latent_interconnect.artifacts.store import MANIFEST_NAME, Manifest, sha256_file
from latent_interconnect.chat.config import ChatConfig
from latent_interconnect.config import load_config
from latent_interconnect.config_index import cited_pins, stand_in_pins
from latent_interconnect.writes.connector import ConnectorSpec

ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "configs/chat/all-connectors.yaml"


def check_bundle(
    bundle_dir: Path, bundle: dict[str, Any], entry: Any, receiver: str, hash_files: bool
) -> dict[str, bool]:
    """Return each named check's result for one connector's bundle."""
    pin = entry.checkpoint_pin
    manifest = Manifest.load(bundle_dir / MANIFEST_NAME)
    header = json.loads((bundle_dir / "connector.json").read_text())
    want = entry.run.connector.digest()
    checks = {
        "manifest=id=pin": manifest.sha256() == bundle["checkpoint_manifest_sha256"] == bundle["pins"][pin]["sha256"],
        "kind": manifest.kind == "checkpoint",
        "run.connector=meta": want == manifest.meta["connector_sha"] == bundle["connector_sha"],
        "stored spec=meta": ConnectorSpec.model_validate(header["spec"]).digest() == manifest.meta["connector_sha"],
        "receiver": header["spec"]["receiver"] == receiver,
        "step": header["step"] == manifest.meta["step"] == entry.step,
    }
    if hash_files:
        checks["files"] = all(
            (bundle_dir / rel).stat().st_size == size and sha256_file(bundle_dir / rel) == sha
            for rel, (sha, size) in manifest.files.items()
        )
    return checks


def main(argv: list[str] | None = None) -> int:
    """The command line (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("connectors", type=Path, help="the staged connectors tree (with connectors.json)")
    parser.add_argument("--no-hash", action="store_true", help="skip re-hashing the weight files")
    args = parser.parse_args(argv)
    index = json.loads((args.connectors / "connectors.json").read_text())
    by_pin: dict[str, tuple[Path, dict[str, Any]]] = {}
    index_ok: dict[str, bool] = {}
    for e in index["connectors"]:
        bundle_dir = args.connectors / e["path"]
        index_ok[e["name"]] = sha256_file(bundle_dir / "bundle.json") == e["bundle_sha256"]
        bundle = json.loads((bundle_dir / "bundle.json").read_text())
        for pin in bundle["pins"]:
            by_pin[pin] = (bundle_dir, bundle)
    chat = load_config(CHAT, ChatConfig, stand_in_pins(cited_pins(CHAT)))
    n = ok = 0
    for rname, receiver in chat.receivers.items():
        for cid, entry in receiver.connectors.items():
            n += 1
            name = f"{rname}__{cid}"
            if entry.checkpoint_pin not in by_pin:
                sys.stdout.write(f"{name}: MISSING (no bundle pins {entry.checkpoint_pin})\n")
                continue
            bundle_dir, bundle = by_pin[entry.checkpoint_pin]
            checks = {"index": index_ok.get(bundle["name"], False), "name": bundle["name"] == name}
            checks |= check_bundle(bundle_dir, bundle, entry, rname, not args.no_hash)
            good = all(checks.values())
            ok += good
            failed = [k for k, v in checks.items() if not v]
            sha = bundle["checkpoint_manifest_sha256"][:12]
            sys.stdout.write(f"{name} {sha} {'OK' if good else 'FAILED ' + ', '.join(failed)}\n")
    sys.stdout.write(f"{ok}/{n} match\n")
    return 0 if ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
