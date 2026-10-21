"""Published written descriptions of a sparse dictionary's features, joined only to the exact dictionary they name.

A description source (:class:`DescriptionSource`) is one frozen bulk export of machine-written feature explanations
(Neuronpedia's, for Gemma Scope 2), held as one tar under the data root and cited by path and sha256, as every outside
source is (``domains.tables.SourceFile``). The source states which dictionary it describes (``repository`` and
``folder``, as Neuronpedia's API metadata named them when the export was fetched); that mapping is **metadata, not a
check of weights**. :func:`require_exact` refuses to attach a source to any other dictionary: never another layer,
family, width, sparsity setting or model.

What a description is: a machine-written hypothesis from top-activating text examples, not a validated meaning. A
feature without one stays uninterpreted; nothing is generated, borrowed or inferred here. Every description the export
holds for a feature is kept (none is chosen over another), with its explainer, type, author and date. Every join
carries that limit in the output itself, as one sentence (:func:`link_caveat`, ``Descriptions.caveat``).

The export is third-party data: each member is read as gzip-compressed JSON lines and nothing else, and a record that
is not of the source's model and source id, or whose feature index is not in the dictionary's range, is an error.
"""

from __future__ import annotations

import gzip
import json
import tarfile
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field

from latent_interconnect.config import Config
from latent_interconnect.domains.tables import SourceFile
from latent_interconnect.receivers.dictionary import DictionarySpec

__all__ = [
    "DescriptionSource",
    "Descriptions",
    "FeatureDescription",
    "link_caveat",
    "read_descriptions",
    "require_exact",
]

_MEMBER = ".jsonl.gz"


class DescriptionSource(Config):
    """One bulk export of feature descriptions and the dictionary its provider's metadata names.

    Attributes:
        provider: who published the export.
        model: the provider's model id (each record's ``modelId``).
        source: the provider's source id (each record's ``layer``), e.g. ``9-gemmascope-2-res-16k``.
        repository: the dictionary's Hugging Face repository, per the provider's API metadata.
        folder: the dictionary's folder in it, per the same metadata (the sparsity setting included).
        file: the export's tar under the data root, with its sha256 (members ``*.jsonl.gz``).
    """

    provider: Literal["neuronpedia"]
    model: str = Field(min_length=1)
    source: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    folder: str = Field(min_length=1)
    file: SourceFile


class FeatureDescription(Config):
    """One published description of one feature, as the export gives it.

    Attributes:
        id: the provider's explanation id.
        description: the text.
        type_name: the explanation type (``typeName``).
        explainer: the model that wrote it (``explanationModelName``).
        author: the provider's author id (``authorId``).
        created_at: its creation time (``createdAt``).
    """

    id: str
    description: str
    type_name: str
    explainer: str
    author: str
    created_at: str


class Descriptions(Config):
    """The descriptions joined to one dictionary's listed features.

    Attributes:
        source: the source (its mapping to the dictionary is metadata, not weights).
        snapshot: where the text came from: the frozen bulk export (no live record is read).
        records: the description records in the export.
        described: the distinct features the export describes, of the dictionary's ``width``.
        width: the dictionary's width.
        selected: listed feature → every description the export holds for it.
        uninterpreted: the listed features the export has no description for.
        explainers: the distinct explanation models the export's records name (``explanationModelName``), sorted.
        caveat: what the join rests on, as one sentence (:func:`link_caveat`); a reader of the record or of its text
            tables meets it wherever the descriptions are.
    """

    source: DescriptionSource
    snapshot: Literal["bulk_export"] = "bulk_export"
    records: int = Field(ge=0)
    described: int = Field(ge=0)
    width: int = Field(ge=1)
    selected: dict[int, tuple[FeatureDescription, ...]]
    uninterpreted: tuple[int, ...]
    explainers: tuple[str, ...]
    caveat: str = Field(min_length=1)


def link_caveat(explainers: Sequence[str]) -> str:
    """Return the one sentence every joined description carries: who wrote it, and what links it to the dictionary.

    Args:
        explainers: the explanation models the export's records name.
    """
    model = ", ".join(explainers) if explainers else "not named in the export"
    return (
        f"The feature descriptions are machine-written (explanation model: {model}) and are linked to this "
        "dictionary by Neuronpedia's API metadata only: the link is NOT verified against the dictionary weights."
    )


def require_exact(source: DescriptionSource, dictionary: DictionarySpec) -> None:
    """Refuse a description source that does not name exactly this dictionary.

    Raises:
        ValueError: if the source's repository or folder is not the dictionary's.
    """
    if (source.repository, source.folder) != (dictionary.repository, dictionary.folder):
        raise ValueError(
            f"descriptions of {source.repository}/{source.folder} ({source.provider} {source.model}/{source.source}) "
            f"cannot be attached to dictionary {dictionary.repository}/{dictionary.folder}: a description belongs to "
            "the exact dictionary its source names, never to another layer, family, width or sparsity setting"
        )


def read_descriptions(path: Path, source: DescriptionSource, width: int, selected: Collection[int]) -> Descriptions:
    """Read the export at ``path`` and return the descriptions of the ``selected`` features.

    Args:
        path: the export's tar (its sha256 already checked, ``domains.tables.read_source``).
        source: the source the tar is.
        width: the dictionary's width.
        selected: the features to join (those in a compact list).

    Raises:
        ValueError: if a line is not a JSON object with a text ``description`` for a feature ``index`` in
            ``0 .. width - 1`` of this source's model and source id.
    """
    wanted = set(selected)
    found: dict[int, list[FeatureDescription]] = {f: [] for f in sorted(wanted)}
    described: set[int] = set()
    explainers: set[str] = set()
    records = 0
    with tarfile.open(path, mode="r:") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(_MEMBER):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"{path.name}: member {member.name!r} cannot be read")
            for number, line in enumerate(gzip.decompress(handle.read()).splitlines(), start=1):
                where = f"{path.name}: {member.name} line {number}"
                record = json.loads(line)
                if not isinstance(record, dict) or (record.get("modelId"), record.get("layer")) != (
                    source.model,
                    source.source,
                ):
                    raise ValueError(f"{where} is not a record of {source.model}/{source.source}")
                text = record.get("description")
                try:
                    feature = int(record["index"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{where} has no integer feature index") from error
                if not 0 <= feature < width or not isinstance(text, str) or not text:
                    raise ValueError(f"{where}: feature {feature} of width {width}, or no description text")
                records += 1
                described.add(feature)
                if record.get("explanationModelName"):
                    explainers.add(str(record["explanationModelName"]))
                if feature in wanted:
                    found[feature].append(
                        FeatureDescription(
                            id=str(record.get("id", "")),
                            description=text,
                            type_name=str(record.get("typeName", "")),
                            explainer=str(record.get("explanationModelName", "")),
                            author=str(record.get("authorId", "")),
                            created_at=str(record.get("createdAt", "")),
                        )
                    )
    return Descriptions(
        source=source,
        records=records,
        described=len(described),
        width=width,
        selected={f: tuple(sorted(d, key=lambda x: x.id)) for f, d in found.items() if d},
        uninterpreted=tuple(f for f, d in found.items() if not d),
        explainers=tuple(sorted(explainers)),
        caveat=link_caveat(sorted(explainers)),
    )
