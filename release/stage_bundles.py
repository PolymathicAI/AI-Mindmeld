"""Stage the release bundles from a data root's store into a directory laid out as the hub repository will be.

    LIP_DATA=<data root> python release/stage_bundles.py OUT_DIR

Reads the source store only. ``OUT_DIR/connectors`` is the model repository's tree
(``sgolkar/ai-mindmeld-connectors``): for each of the 19 chat connectors (``configs/chat/all-connectors.yaml``) the
checkpoint artifact as stored in ``<receiver>__<connector>/`` with its ``bundle.json``, and the index
``connectors.json``. ``OUT_DIR/data`` is the dataset repository's tree (``sgolkar/ai-mindmeld-data``): the
minimal evaluation sets of the Quickstart (``examples/README.md``), built in a store under
``OUT_DIR/build`` from the paper suites' inputs, in ``<bundle>/<artifact id>/``, and the index ``data.json``. Every
copied file's sha256 is checked against the artifact's manifest. ``latent_interconnect.release_download`` reads it. The
bundle and index bodies come from :mod:`latent_interconnect.bundle_connector`, which packages a contributor's
checkpoint in the same format.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from latent_interconnect.artifacts.records import RecordStore
from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.bundle_connector import connector_bundle, copy_artifact_dir, write_bundle, write_index
from latent_interconnect.chat.config import ChatConfig
from latent_interconnect.config import Pin
from latent_interconnect.data.items import ItemSet, commit_item_set, load_item_set
from latent_interconnect.domains.base import DOMAINS
from latent_interconnect.encoders.base import RawObject
from latent_interconnect.encoders.jobs import objects_sha256
from latent_interconnect.encoders.store import EmbeddingReader
from latent_interconnect.evaluation.suites import SuiteRunConfig
from latent_interconnect.paths import Paths
from latent_interconnect.workspace import Workspace
from latent_interconnect.writes.checkpoint import read_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CREDIT = (
    "credit: Siavash Golkar, Jake Kovalic, and Shirley Ho, AI Mindmeld: A Latent Interconnect Protocol, 2026, "
    "Manuscript; then Siavash Golkar and the AI Mindmeld contributors, AI Mindmeld, "
    "https://github.com/PolymathicAI/AI-Mindmeld."
)
"""The attribution every CC BY 4.0 grant of this project carries."""
OWN = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) for {what}; " + CREDIT
"""Our own grant, for ``what`` the bundle holds."""
INDEX_LICENSE = "per bundle (bundle.json 'license')"
"""An index's ``license``: each bundle states its own terms."""
ESMC_TERMS = (
    "ESM-C terms (Hugging Face license tags 'mit' and 'other', Biohub acceptable use policy; weights under the MIT "
    'License, as the official repository states ("These models are available under the MIT license", '
    "https://github.com/Biohub/esm/blob/main/LICENSE.md); use per https://biohub.org/acceptable-use-policy/)"
)
CHEMFM_TERMS = (
    "CC BY-NC 4.0, non-commercial (the LICENSE of the official repository github.com/TheLuoFengLab/ChemFM, where the "
    "paper places the checkpoints; the Hugging Face card's 'mit' tag is metadata only)"
)
RECEIVER_TERMS = {
    "moonshotai/Kimi-K2.6": "the Kimi K2.6 license (its model card names a Modified MIT License, Hugging Face "
    "license_name 'modified-mit')",
    "google/gemma-3-4b-it": "Gemma Terms of Use (Hugging Face license tag 'gemma'; gated)",
    "google/gemma-3-12b-it": "Gemma Terms of Use (Hugging Face license tag 'gemma'; gated)",
}
"""Receiver model id -> its upstream terms, stated after ``<model id>@<revision[:8]>``."""
SPECIALIST_TERMS = {
    "esmc-600m": f"biohub/ESMC-600M@fcc9d36f: {ESMC_TERMS}",
    "esmc-6b": f"biohub/ESMC-6B@af1602ba: {ESMC_TERMS}",
    "mimic": "MIMIC (Polymathic AI, internal checkpoint 5.7_main, sha256 1113846c...; not public; the public "
    "MIMIC 1.0 is polymathic-ai/MIMIC, Hugging Face license tag 'mit'): the MIMIC weights are not publicly released "
    "and no public license is stated for them; contact the authors before use",
    "pglm-1b": "biomap-research/proteinglm-1b-mlm@676c1bcb: Creative Commons Attribution-NonCommercial 4.0 "
    "International (model card), non-commercial",
    "chemfm-1b": f"ChemFM/ChemFM-1B@f99dc2e8: {CHEMFM_TERMS}",
    "molformer-c3": "DeepChem/MoLFormer-c3-1.1B@9f1b9ea3: Hugging Face license tag 'mit' (code "
    "ibm-research/MoLFormer-XL-both-10pct: 'apache-2.0')",
    "unimol2-1.1b": "Uni-Mol2 1.1B via unimol_tools 0.1.6 (checkpoint sha256 48e1ba23...); dptech/Uni-Mol2 Hugging "
    "Face license tag 'mit'; the checkpoint is that repository's modelzoo/1.1B/checkpoint.pt at revision bf37d253 "
    "(same sha256)",
}
"""Chat connector id -> its specialist's identity and upstream terms."""
SOURCE_COMMIT = "7740f56f433eed87e60f1f4f13b23d6aaef8f52e"
SUITES = "configs/suites/gemma-3-4b-it"
DATA = {  # bundle -> (the paper suite whose inputs it is cut from, its pins' prefix)
    "protein-esmc-600m-mini": (f"{SUITES}/esmc-600m-protein-access-trained.yaml", "protein/esmc-600m"),
    "molecule-chemfm-1b-mini": (
        f"{SUITES}/chemfm-1b-molecule-access-novel-scaffold-trained.yaml",
        "molecule/chemfm-1b",
    ),
}
DATA_SOURCES = {  # bundle -> (the source data and its terms, the connector id of the specialist that embedded it)
    "protein-esmc-600m-mini": ("UniProtKB/Swiss-Prot release 2026_03 (CC BY 4.0, (c) UniProt Consortium)", "esmc-600m"),
    "molecule-chemfm-1b-mini": (
        "ChEBI release 255 (CC BY 4.0, EMBL-EBI terms of use) and the UniProtKB/Swiss-Prot 2026_03 catalytic-activity "
        "table (CC BY 4.0)",
        "chemfm-1b",
    ),
}
N_OBJECTS = 8
"""Objects in each minimal eval set (the minimal usable set)."""


def connector_license(model_id: str, revision: str, connector: str) -> str:
    """The license string of a chat connector's bundle: our grant, then the receiver's and specialist's terms."""
    receiver = f"receiver {model_id}@{revision[:8]}: {RECEIVER_TERMS[model_id]}"
    return (
        f"{OWN.format(what='these connector weights')} Upstream terms also apply and are not overridden: "
        f"{receiver}; specialist {SPECIALIST_TERMS[connector]}"
    )


def data_license(bundle: str) -> str:
    """The license string of a minimal eval set's bundle: our grant, its source data, its embedding model's terms."""
    source, specialist = DATA_SOURCES[bundle]
    return (
        f"{OWN.format(what='this minimal eval set')} Derived from {source}, with embeddings made with "
        f"{SPECIALIST_TERMS[specialist]}; those terms also apply and are not overridden"
    )


def history_records(ws: Workspace) -> dict[str, str]:
    """Pin name -> the record id of the latest history line that set it (the pinning record)."""
    out: dict[str, str] = {}
    for line in (ws.pins_file.parent / "pins.history.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry.get("new") and entry.get("record"):
            out[entry["name"]] = entry["record"]
    return out


def copy_artifact(ws: Workspace, ref: ArtifactRef, dest: Path) -> dict[str, Any]:
    """Copy one artifact's files as stored, check each copy against the manifest, return its bundle entry."""
    manifest, entry = copy_artifact_dir(ws.artifacts.path(ref), dest)
    if manifest.sha256() != ref.sha256:
        raise RuntimeError(f"{ref.short}: its manifest hashes to {manifest.sha256()[:12]}")
    return entry


def stage_connectors(ws: Workspace, out: Path) -> list[dict[str, Any]]:
    """Stage the 19 chat connectors and return their index entries."""
    cfg = ws.load(ROOT / "configs/chat/all-connectors.yaml", ChatConfig)
    pins, recs = ws.pins(), history_records(ws)
    index = []
    for rname, receiver in cfg.receivers.items():
        for cid, entry in receiver.connectors.items():
            name = f"{rname}__{cid}"
            run = entry.run
            pin = pins[entry.checkpoint_pin]
            ref = ws.artifacts.ref(pin.sha256)
            bundle_dir = out / "connectors" / name
            art = copy_artifact(ws, ref, bundle_dir)
            art["path"] = "."
            _, step = read_checkpoint(bundle_dir)
            if step != entry.step:
                raise RuntimeError(f"{name}: checkpoint step {step}, the chat config names {entry.step}")
            record_pin = pins.get(f"{run.destination}@record")
            bundle = connector_bundle(
                name=name,
                label=f"{receiver.label} x {entry.label}",
                header=json.loads((bundle_dir / "connector.json").read_text()),
                manifest=ws.artifacts.manifest(ref),
                artifact=art,
                run=run,
                receiver={
                    "name": receiver.receiver.name,
                    "model_id": receiver.receiver.model_id,
                    "revision": receiver.receiver.revision,
                },
                pin=entry.checkpoint_pin,
                source={
                    "research_commit": SOURCE_COMMIT,
                    "run_config": run.name,
                    "run_destination": run.destination,
                    "checkpoint_pin": entry.checkpoint_pin,
                    "pinning_record": recs.get(entry.checkpoint_pin),
                    "run_record_key": record_pin.sha256 if record_pin else None,
                },
                operational_only=entry.operational_only,
                license=connector_license(receiver.receiver.model_id, receiver.receiver.revision, cid),
            )
            index.append(write_bundle(bundle_dir, bundle))
            print(name, ref.short, art["bytes"], flush=True)  # noqa: T201
    return index


def _minimal_artifacts(src: Workspace, build: Workspace, suite: Path, prefix: str) -> dict[str, ArtifactRef]:
    """Build the minimal eval inputs of one suite in ``build``: the first :data:`N_OBJECTS` objects of its item set.

    The item set, the domain tables and the embeddings are cut to those objects (the embeddings hold only the layers
    the connector reads, as the source does). Each is committed through the store, so its id is its own; it records
    the research artifact it was cut from (``subset_of``). Returns pin name -> artifact.
    """
    config, cited = src.load_cited(ROOT / suite, SuiteRunConfig)
    by_kind = {src.artifacts.ref(p.sha256).kind: (n, src.artifacts.ref(p.sha256)) for n, p in cited.items()}
    _, items_ref = by_kind["item_set"]
    _, domain_ref = by_kind["domain_data"]
    _, emb_ref = by_kind["embeddings"]
    items = load_item_set(src.artifacts, items_ref)
    keep = slice(0, N_OBJECTS)
    small = ItemSet(
        f"{items.name}-mini{N_OBJECTS}",
        items.roles,
        {r: v[keep] for r, v in items.objects.items()},
        {f: v[keep] for f, v in items.fields.items()},
    )
    ids = sorted({o for v in small.objects.values() for o in v})
    out: dict[str, ArtifactRef] = {}
    # the domain: every table cut to the objects (each is keyed by object_id), the catalog as it is
    staged = build.artifacts.staging(f"mini-domain-{domain_ref.sha256}-{N_OBJECTS}")
    droot = src.artifacts.path(domain_ref)
    for rel in src.artifacts.manifest(domain_ref).files:
        if rel.endswith(".parquet"):
            table = pd.read_parquet(droot / rel)
            table[table["object_id"].isin(ids)].reset_index(drop=True).to_parquet(staged / rel, index=False)
        else:
            shutil.copyfile(droot / rel, staged / rel)
    dmeta = dict(src.artifacts.manifest(domain_ref).meta) | {"subset_of": domain_ref.sha256, "objects_kept": len(ids)}
    small_domain = build.artifacts.commit(staged, domain_ref.kind, dmeta)
    out[f"release/{prefix}/domain-mini"] = small_domain
    out[f"release/{prefix}/items-mini"] = commit_item_set(
        build.artifacts, small, {"subset_of": items_ref.sha256, "selection": f"the first {N_OBJECTS} items"}
    )
    # the embeddings: one part, the read layers only, the object list's digest over the kept objects
    reader = EmbeddingReader(src.artifacts, emb_ref)
    staged = build.artifacts.staging(f"mini-emb-{emb_ref.sha256}-{N_OBJECTS}")
    (staged / "part-00000").mkdir(exist_ok=True)
    (staged / "part-00000" / "ids.json").write_text(json.dumps(ids))
    for layer in reader.layers:
        np.save(staged / "part-00000" / f"pooled.L{layer}.npy", np.ascontiguousarray(reader.pooled(ids, layer)))
    values = DOMAINS.get(config.suite.domains[0].name)(build.artifacts.path(small_domain)).objects()["value"]
    emeta = dict(reader.meta) | {
        "objects": objects_sha256(RawObject(o, str(values[o])) for o in ids),
        "n_skipped": 0,
        "skipped": {},
        "subset_of": emb_ref.sha256,
    }
    out[f"release/{prefix}/embeddings-mini"] = build.artifacts.commit(staged, emb_ref.kind, emeta)
    return out


def stage_data(ws: Workspace, out: Path) -> list[dict[str, Any]]:
    """Build and stage the minimal eval inputs of the declared path (``DATA``); return their index entries."""
    build = _workspace(out / "build")
    index = []
    for name, (suite, prefix) in DATA.items():
        made = _minimal_artifacts(ws, build, Path(suite), prefix)
        build.pin_all({k: Pin(target="artifact", sha256=v.sha256) for k, v in made.items()}, repin="rebuilt")
        bundle_dir = out / "data" / name
        arts, pins = [], {}
        for pin_name, ref in made.items():
            art = copy_artifact(build, ref, bundle_dir / ref.sha256)
            art["path"] = ref.sha256
            art["pin"] = pin_name
            arts.append(art)
            pins[pin_name] = {"target": "artifact", "sha256": ref.sha256}
            print(name, pin_name, ref.short, art["bytes"], flush=True)  # noqa: T201
        bundle = {
            "format": "aimm-bundle/1",
            "type": "data",
            "name": name,
            "cut_from_suite": suite,
            "objects": N_OBJECTS,
            "artifacts": arts,
            "pins": pins,
            "source": {"research_commit": SOURCE_COMMIT},
            "license": data_license(name),
        }
        index.append(write_bundle(bundle_dir, bundle))
    return index


def _workspace(root: Path) -> Workspace:
    paths = Paths.from_env({"LIP_DATA": str(root)})
    return Workspace(
        paths, ArtifactStore(paths.artifacts), RecordStore(paths.records), root / "lip" / "pins.yaml", None
    )


def main() -> None:
    """Stage everything into ``sys.argv[1]``."""
    out = Path(sys.argv[1])
    ws = Workspace.from_env()
    for kind, index in (("connectors", stage_connectors(ws, out)), ("data", stage_data(ws, out))):
        write_index(out / kind / f"{kind}.json", kind, index, INDEX_LICENSE)


if __name__ == "__main__":
    main()
