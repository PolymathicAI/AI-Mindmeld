"""Approved release exports of training sets: one strict reader of their construction definition and the registry.

A release export of a training set is a new artifact that holds the original training set's objects and text
unchanged, but whose metadata names the exported domain (an artifact whose names were mapped, for instance a task id).
Its training text was not produced on that domain, so it carries no producing spec: ``meta.spec`` and
``meta.spec_digest`` are absent and :func:`latent_interconnect.build.zero_shot.training_set_spec` refuses it. Instead
its manifest metadata holds :data:`MARKER`, a :class:`ReleaseExport` descriptor: the original training set (its id, its
domain and the two digests of its producing spec), the target domain, the **construction definition** (a
``TrainingSetSpec``-shaped mapping derived mechanically from the original spec) and the digests of the export receipt
(:data:`RECEIPT_FILE`, a file of the artifact) and of its transformation inventory.

A definition never states what produced the training set; it states what the consumers check the training set
against (``training/trainer.py``, ``build.zero_shot.check_construction``, ``evaluation.logistic_baseline``), all through
``build.zero_shot.training_set_construction``. A descriptor is trusted only through the registry packaged with this
code, :data:`REGISTRY_FILE` (never through data an artifact supplies): an entry, keyed by the export's final artifact
id, pins the descriptor's digest, the receipt's, the transformation inventory's, the exact target domain and the
original training set. The receipt's bytes were frozen before the final id was computed and the registry binds that
id afterwards, so neither the receipt nor the descriptor cites the id (no hash cycle).

:func:`resolve_release_export` fails closed: an export marker with a missing, malformed or contradictory descriptor,
receipt, registry entry or file refuses, and nothing falls back to ``meta.spec``. Ordinary training sets (historical
or newly built) never read the registry. The receipt attests that the original published artifacts' bytes were
verified and the enumerated transformation was applied; it is not a reconstruction from upstream sources, it grants no
scientific eligibility, and it does not relate any other artifact to the original.

Digest rules: ``spec_digest`` is ``TrainingSetSpec.digest()`` as the original recorded it (the canonical identity,
rationale fields excluded), so equal digests need not mean equal spec bytes; ``spec_sha256`` and every ``*_sha256``
here are sha256 of ``canonical_json`` of the whole value (every field) or of a file's bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

from pydantic import ValidationError

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore, Manifest
from latent_interconnect.canonical import canonical_sha256, is_sha256
from latent_interconnect.config import ArtifactDigest, Config
from latent_interconnect.domains.base import DOMAIN_DATA_KIND, DomainSpec

__all__ = [
    "EXPORT_FORMAT",
    "MARKER",
    "RECEIPT_FILE",
    "RECEIPT_FORMAT",
    "REGISTRY_FILE",
    "REGISTRY_FORMAT",
    "ExportedFrom",
    "RegistryEntry",
    "ReleaseExport",
    "release_registry",
    "resolve_release_export",
]

EXPORT_FORMAT = "aimm-release-export/1"
"""The one supported descriptor schema."""
RECEIPT_FORMAT = "aimm-release-export-receipt/1"
"""The one supported receipt schema."""
REGISTRY_FORMAT = "aimm-release-export-registry/1"
"""The one supported registry schema."""
MARKER = "release_export"
"""The manifest metadata key of an export's descriptor: present means the strict export path, never ``meta.spec``."""
RECEIPT_FILE = "release-export-receipt.json"
"""The export receipt, a file of the exported training set (its bytes are pinned by the descriptor and the registry)."""
REGISTRY_FILE = "release_metadata_exports.json"
"""The registry, package data of ``latent_interconnect`` (``pyproject.toml`` ``[tool.setuptools.package-data]``)."""


class ExportedFrom(Config):
    """The original training set an export is made from.

    Attributes:
        training_set: the original artifact id.
        domain: the domain its producing spec names.
        spec_digest: its producing spec's recorded ``meta.spec_digest`` (``TrainingSetSpec.digest()``: rationale
            fields excluded).
        spec_sha256: sha256 of ``canonical_json`` of its whole producing spec (``meta.spec``, every field).
    """

    training_set: ArtifactDigest
    domain: DomainSpec
    spec_digest: ArtifactDigest
    spec_sha256: ArtifactDigest


class ReleaseExport(Config):
    """The descriptor (:data:`MARKER`) of an exported training set.

    Attributes:
        format: :data:`EXPORT_FORMAT`.
        exported_from: the original training set.
        target_domain: the domain the definition names (the exported domain).
        definition: the construction definition, ``TrainingSetSpec``-shaped (validated by the caller).
        definition_sha256: sha256 of ``canonical_json(definition)``.
        transformations_sha256: sha256 of ``canonical_json`` of the receipt's ``transformations``.
        receipt_sha256: sha256 of the bytes of :data:`RECEIPT_FILE`.
    """

    format: Literal["aimm-release-export/1"]
    exported_from: ExportedFrom
    target_domain: DomainSpec
    definition: dict[str, Any]
    definition_sha256: ArtifactDigest
    transformations_sha256: ArtifactDigest
    receipt_sha256: ArtifactDigest


class RegistryEntry(Config):
    """One approved export in the registry, keyed by its final artifact id.

    Attributes:
        exported_from: the original training set's artifact id.
        target_domain: the exact domain the export's definition must name.
        descriptor_sha256: sha256 of ``canonical_json`` of the descriptor (:data:`MARKER`'s value).
        receipt_sha256: sha256 of the receipt file's bytes.
        transformations_sha256: sha256 of ``canonical_json`` of the receipt's transformation inventory.
    """

    exported_from: ArtifactDigest
    target_domain: DomainSpec
    descriptor_sha256: ArtifactDigest
    receipt_sha256: ArtifactDigest
    transformations_sha256: ArtifactDigest


class _Registry(Config):
    format: Literal["aimm-release-export-registry/1"]
    exports: dict[str, RegistryEntry]


def _registry_text() -> str:
    """Return the packaged registry's text (installed with the code; never read from an artifact or a data root)."""
    return resources.files("latent_interconnect").joinpath(REGISTRY_FILE).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def release_registry() -> Mapping[str, RegistryEntry]:
    """Return the packaged registry, export artifact id → its :class:`RegistryEntry` (read once).

    Raises:
        ValueError: if the packaged registry is missing or malformed (then every export refuses).
    """
    try:
        text = _registry_text()
        registry = _Registry.model_validate(json.loads(text))
    except (OSError, ValueError, ValidationError) as e:
        raise ValueError(f"the release export registry {REGISTRY_FILE} is missing or malformed: {e}") from e
    bad = [k for k in registry.exports if not is_sha256(k)]
    if bad:
        raise ValueError(f"the release export registry {REGISTRY_FILE} keys are not artifact ids: {bad[:3]}")
    return dict(registry.exports)


def resolve_release_export(store: ArtifactStore, ref: ArtifactRef, manifest: Manifest) -> dict[str, Any]:
    """Return an approved export's construction definition, after every binding check; refuse anything else.

    ``manifest`` is ``store.manifest(ref)`` (already checked against ``ref``) and holds :data:`MARKER`. Checked, all
    against the packaged registry: the export's id has an entry; the metadata holds no producing spec; the descriptor
    has the supported schema and the entry's digest; the export's files (the receipt among them) match the manifest;
    the receipt has the supported schema, the pinned bytes, and states the same original, target domain, definition
    digest and transformation inventory; the definition has its digest and names exactly the entry's target domain;
    the entry's original is the descriptor's; and the registered target domain's files match its manifest.

    Raises:
        ValueError: naming the first binding that fails (never a fallback to ``meta.spec``).
        IntegrityError: if a file of the export or of its registered target domain differs from its manifest.
    """
    where = f"release export {ref.short}"
    meta = manifest.meta
    entry = release_registry().get(ref.sha256)
    if entry is None:
        raise ValueError(f"{where}: not an approved release export (no registry entry for its id)")
    if "spec" in meta or "spec_digest" in meta:
        raise ValueError(f"{where}: an export marker and a producing spec contradict each other")
    raw = meta.get(MARKER)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: its {MARKER!r} descriptor is not a mapping")
    try:
        descriptor = ReleaseExport.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"{where}: malformed or unsupported descriptor: {e}") from e
    if canonical_sha256(raw) != entry.descriptor_sha256:
        raise ValueError(f"{where}: its descriptor is not the registered one")
    if descriptor.exported_from.training_set != entry.exported_from:
        raise ValueError(f"{where}: its original training set is not the registered one")
    receipt = _registered_receipt(where, store, ref, manifest=manifest, descriptor=descriptor, entry=entry)
    definition = _registered_definition(where, receipt, descriptor=descriptor, entry=entry)
    store.verify(ArtifactRef(entry.target_domain.data, DOMAIN_DATA_KIND))  # its files, not only its manifest identity
    return definition


def _registered_receipt(
    where: str,
    store: ArtifactStore,
    ref: ArtifactRef,
    *,
    manifest: Manifest,
    descriptor: ReleaseExport,
    entry: RegistryEntry,
) -> dict[str, Any]:
    """Return the export's receipt once the manifest, the descriptor, the registry and its bytes all pin it."""
    if RECEIPT_FILE not in manifest.files:
        raise ValueError(f"{where}: no {RECEIPT_FILE} in its manifest")
    if not (manifest.files[RECEIPT_FILE][0] == descriptor.receipt_sha256 == entry.receipt_sha256):
        raise ValueError(f"{where}: its receipt is not the registered one")
    store.verify(ref)  # every file, the receipt included, against the manifest
    data = (store.path(ref) / RECEIPT_FILE).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.receipt_sha256:
        raise ValueError(f"{where}: the receipt's bytes are not the registered ones")
    try:
        receipt = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ValueError(f"{where}: the receipt is not JSON: {e}") from e
    keys = {"format", "exported_from", "target_domain", "definition_sha256", "transformations", "verification"}
    if not isinstance(receipt, dict) or set(receipt) != keys or receipt["format"] != RECEIPT_FORMAT:
        raise ValueError(f"{where}: the receipt does not have the supported schema {RECEIPT_FORMAT}")
    return receipt


def _registered_definition(
    where: str, receipt: Mapping[str, Any], *, descriptor: ReleaseExport, entry: RegistryEntry
) -> dict[str, Any]:
    """Return the definition once the receipt, the descriptor and the registry agree on it and on its domain."""
    if canonical_sha256(receipt["transformations"]) != descriptor.transformations_sha256:
        raise ValueError(f"{where}: the receipt's transformation inventory is not the descriptor's")
    if descriptor.transformations_sha256 != entry.transformations_sha256:
        raise ValueError(f"{where}: its transformation inventory is not the registered one")
    if receipt["exported_from"] != descriptor.exported_from.model_dump(mode="json"):
        raise ValueError(f"{where}: the receipt and the descriptor name different originals")
    target = descriptor.target_domain
    if receipt["target_domain"] != target.model_dump(mode="json") or target != entry.target_domain:
        raise ValueError(f"{where}: the target domain is not the registered one")
    if receipt["definition_sha256"] != descriptor.definition_sha256:
        raise ValueError(f"{where}: the receipt and the descriptor name different definitions")
    if canonical_sha256(descriptor.definition) != descriptor.definition_sha256:
        raise ValueError(f"{where}: its definition does not have its digest")
    try:
        named = DomainSpec.model_validate(descriptor.definition.get("domain"))
    except ValidationError as e:
        raise ValueError(f"{where}: its definition names no valid domain: {e}") from e
    if named != target:
        raise ValueError(f"{where}: its definition names domain {named.data[:12]}, not the target {target.data[:12]}")
    return dict(descriptor.definition)
