"""From the embedding store to :class:`~latent_interconnect.writes.inputs.ObjectBatch` (orchestrator side, torch).

The loader composes a connector's input from one or more embedding artifacts per its :class:`InputSpec`: the spec's
layers (or :data:`~latent_interconnect.encoders.store.OUTPUT_LAYER` for an encoder's defined output) are read, object
by object, from the one artifact that holds that layer for that object (artifacts may split the layers, the objects,
or both), and with ``combine = "concat"`` concatenated along features in the spec's order. Pooled vectors and/or
residues are loaded as the connector's map kinds need
(:attr:`~latent_interconnect.writes.connector.Connector.input_kinds`); residues are padded into ``(B, T, d)`` with a
validity mask. Values are raw: standardization belongs to the connector.

Every load is checked twice, by two rules with one home each: the spec against the encoder recipe by the one
read-layer rule (:func:`~latent_interconnect.writes.inputs.check_read_layers`, also run by ``InitSpec`` validation),
and every artifact against the recipe it must come from: the recipe in its metadata (the extraction's recipe) must
have the encoder recipe's digest, its whole content, not only its name, since two encoders of one width would otherwise
be swapped silently (MIMIC 5.5 and 5.7 are both 7,680-wide; the old code's encoder guard).

A spec's ``append`` reads a second extraction jointly: the readers recording its recipe's name are its own, and
its pooled vector at its one layer is concatenated after the input's, per object, by a loader of its own (its
recipe, given as ``append_encoder`` or else the one its artifacts record, one for all of them, takes the encoder's place
in both checks, so a given recipe is checked by digest like the main one).

A third check ties the vectors to the inputs a run consumes (:class:`Consumed`): an artifact is found by object id, and
an id does not say which string or sequence a vector is of. A run on a corrected domain version that still cites an
extraction of the earlier strings is refused, object by object, against the object list the extraction was keyed on.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pydantic import ValidationError

from latent_interconnect.encoders.base import RawObject
from latent_interconnect.encoders.jobs import objects_sha256, staged_values
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import _SKIPPED_FILE, OUTPUT_LAYER, EmbeddingReader
from latent_interconnect.writes.inputs import InputKind, InputSpec, ObjectBatch, check_read_layers

__all__ = ["Consumed", "InputLoader"]


@dataclass(frozen=True)
class Consumed:
    """The raw inputs a run consumes, which the vectors it loads must have been extracted from.

    Attributes:
        domain: the domain at its data version, as refusals name it.
        values: object id → the domain's raw encoder input (its object table's ``value``).
        root: the artifact store root, whose staging area keeps each extraction's object list
            (``encoders.jobs.staged_values``). A store that holds an artifact without its list (a copied store) is
            compared on the whole list instead: the list rebuilt from ``values`` must have the sha256 the artifact
            records.
    """

    domain: str
    values: Mapping[str, object]
    root: Path


def _recipe_digest(reader: EmbeddingReader) -> str | None:
    """Return the digest of the encoder recipe an artifact records, or None if it records none in full."""
    content = reader.meta.get("recipe")
    if not isinstance(content, Mapping):
        return None
    try:
        return EncoderRecipe.model_validate(dict(content)).digest()
    except ValidationError:  # a name-only or partial record names no recipe content to compare
        return None


def _recipe_name(reader: EmbeddingReader) -> object:
    content = reader.meta.get("recipe")
    return content.get("name") if isinstance(content, Mapping) else content


def _refusal(oid: str, layer: int, holders: int) -> str:
    return f"{oid}: held by {holders or 'no'} embedding artifact{'s' if holders > 1 else ''} at layer {layer}"


def _stale_refusal(reader: EmbeddingReader, domain: str, stale: Sequence[str]) -> str:
    return (
        f"embedding artifact {reader.root.name[:12]} ({reader.recipe}) was extracted from other values than "
        f"{domain} holds for {len(stale)} object(s), e.g. {list(stale[:5])}"
    )


def _rebuilt(reader: EmbeddingReader, listed: str, consumed: Consumed) -> dict[str, str]:
    """The values ``reader``'s artifact was extracted from, when its list is not staged: the consumed ones.

    Shown on the whole list: the artifact's objects, part by part in row order, with the consumed values hash to
    ``listed``. A part's skipped objects are in the list but in no row; they are placed by sorting their part, which
    is their place in the lists the ``extract`` stage has been given, and the sha256 decides.

    Raises:
        ValueError: if the rebuilt list has another sha256 (some value differs, which one is not known).
    """
    order: list[str] = []
    for part in sorted(p for p in reader.root.iterdir() if p.name.startswith("part-")):
        ids = json.loads((part / "ids.json").read_text())
        skipped = part / _SKIPPED_FILE
        order += sorted([*ids, *json.loads(skipped.read_text())]) if skipped.is_file() else ids
    values = {o: str(consumed.values[o]) for o in order if o in consumed.values}
    if len(values) != len(order) or objects_sha256(RawObject(o, v) for o, v in values.items()) != listed:
        raise ValueError(
            f"embedding artifact {reader.root.name[:12]} ({reader.recipe}): the object list it was extracted from "
            f"(objects-{listed[:12]}) is not in the staging area of {consumed.root}, and its {len(order)} objects "
            f"with the values of {consumed.domain} are another list: it cannot be compared object by object"
        )
    return values


class InputLoader:
    """Loads connector inputs for batches of object ids.

    Args:
        spec: what to read.
        readers: one reader per embedding artifact the spec needs.
        kinds: the input kinds to load (``Connector.input_kinds``); ``InputKind.NONE`` needs no data and is ignored.
        encoder: the encoder recipe the spec names.
        append_encoder: the recipe of the spec's appended extraction, whose digest its readers must record; None takes
            the recipe they record (still one for all of them).
        consumed: the values the run consumes for the objects it loads. With it, an object is refused whose value
            differs from the one its artifact was extracted from (the domain's strings were corrected after the
            extraction, or the artifact is of another data version): an object id alone does not say which input a
            vector is of. An artifact whose list is not in the staging area is loaded only if its whole list, rebuilt
            from these values, is the one it records. An artifact that records no object list (not made by the
            ``extract`` stage) is not compared. None compares nothing: the fits and scans that are kept as made read
            their inputs this way.

    Raises:
        ValueError: as ``writes.inputs.check_read_layers`` (another recipe, pending, repeated or out-of-read layers,
            layers against an output recipe or none against another), or if ``single`` names more than one layer,
            ``kinds`` asks for nothing loadable, a reader's artifact does not record ``encoder`` (by digest), or a
            layer is held by no reader; with ``append``, if no reader is of its recipe, its readers record more than
            one recipe or not ``append_encoder``, or ``kinds`` asks for residues.
    """

    def __init__(
        self,
        spec: InputSpec,
        readers: Sequence[EmbeddingReader],
        kinds: Collection[InputKind],
        *,
        encoder: EncoderRecipe,
        append_encoder: EncoderRecipe | None = None,
        consumed: Consumed | None = None,
    ) -> None:
        self.encoder = encoder
        self.consumed = consumed
        self._extracted: dict[EmbeddingReader, dict[str, str] | None] = {}
        self.append: InputLoader | None = None
        if (a := spec.append) is not None:
            theirs = [r for r in readers if _recipe_name(r) == a.recipe]
            if not theirs:
                raise ValueError(f"input of {spec.recipe!r}: none of the readers is of the appended {a.recipe!r}")
            if frozenset(kinds) - {InputKind.NONE} != {InputKind.POOLED}:
                raise ValueError(f"input of {spec.recipe!r}: an appended extraction is read as pooled vectors only")
            own = append_encoder or EncoderRecipe.model_validate(dict(theirs[0].meta["recipe"]))
            self.append = InputLoader(
                InputSpec(recipe=a.recipe, layers=(a.layer,)), theirs, kinds, encoder=own, consumed=consumed
            )
            readers = [r for r in readers if _recipe_name(r) != a.recipe]
        layers = check_read_layers(spec, encoder) or (OUTPUT_LAYER,)
        if spec.combine == "single" and len(layers) != 1:
            raise ValueError(f"input of {spec.recipe!r}: combine 'single' takes one layer, got {layers}")
        foreign = sum(_recipe_digest(r) != encoder.digest() for r in readers)
        if foreign:
            raise ValueError(
                f"input of {spec.recipe!r}: recipe content (digest) differs in {foreign} of {len(readers)} artifacts "
                f"(expected {encoder.digest()[:12]})"
            )
        self.spec = spec
        self.readers = tuple(readers)
        self.kinds = frozenset(kinds) - {InputKind.NONE}
        if not self.kinds:
            raise ValueError("nothing to load: kinds holds neither pooled nor residues")
        self.layers: tuple[int, ...] = tuple(layers)
        self._holders: dict[int, tuple[EmbeddingReader, ...]] = {}
        for layer in self.layers:
            self._holders[layer] = tuple(r for r in self.readers if layer in r.layers)
            if not self._holders[layer]:
                raise ValueError(f"layer {layer} of {spec.recipe!r} is held by none of the readers")

    def batch(self, object_ids: Sequence[str]) -> ObjectBatch:
        """Return the inputs of these objects, in this order.

        Raises:
            KeyError: for an object no reader holds at some layer.
            ValueError: for an empty batch, an object more than one reader holds at a layer, an object whose
                consumed value is not the one its artifact was extracted from (``consumed``; also if that artifact's
                object list is not in the staging area and differs from the one rebuilt from the consumed values), or
                residues whose lengths or stored dtypes differ across layers for an object.
        """
        ids = tuple(object_ids)
        if not ids:
            raise ValueError("an empty batch")
        pooled = residues = mask = None
        if InputKind.POOLED in self.kinds:
            parts = [self._pooled(ids, L) for L in self.layers]
            if self.append is not None:
                parts.append(self.append._pooled(ids, self.append.layers[0]))
            pooled = torch.from_numpy(np.concatenate(parts, axis=1))
        if InputKind.RESIDUES in self.kinds:
            residues, mask = self._residues(ids)
        return ObjectBatch(object_ids=ids, pooled=pooled, residues=residues, mask=mask)

    def missing(self, object_ids: Iterable[str]) -> dict[str, str]:
        """Return object → why :meth:`batch` would refuse it, for each object it would refuse on these grounds.

        The batch's own holder rule: at every read layer, exactly one of the readers holding that layer holds the
        object; and, with ``consumed``, the object's consumed value is the one that holder's artifact was extracted
        from. Refusals that depend on the stored data (residue lengths or dtypes differing across layers) are not
        seen here; ``batch`` still raises those when it reads the object.

        Raises:
            ValueError: with ``consumed``, if a holder's object list is not in the staging area and differs from the
                one rebuilt from the consumed values.
        """
        ids = tuple(dict.fromkeys(object_ids))
        out: dict[str, str] = {}
        for oid in ids:
            for layer in self.layers:
                found = self._found(oid, layer)
                if len(found) != 1:
                    out[oid] = _refusal(oid, layer, len(found))
                    break
                holder = self._holders[layer][found[0]]
                if self.consumed is not None and self._stale(holder, (oid,)):
                    out[oid] = f"{oid}: {_stale_refusal(holder, self.consumed.domain, (oid,))}"
                    break
        if self.append is not None:
            out = self.append.missing(ids) | out
        return out

    def _found(self, oid: str, layer: int) -> list[int]:
        """The indices of the readers holding ``layer`` that hold ``oid`` there."""
        return [j for j, r in enumerate(self._holders[layer]) if oid in r]

    def _groups(self, ids: tuple[str, ...], layer: int) -> list[tuple[EmbeddingReader, list[int]]]:
        """Partition the batch positions by the one reader that holds each object at ``layer``."""
        groups: dict[int, list[int]] = {}
        holders = self._holders[layer]
        for i, oid in enumerate(ids):
            found = self._found(oid, layer)
            if not found:
                raise KeyError(_refusal(oid, layer, 0))
            if len(found) > 1:
                raise ValueError(_refusal(oid, layer, len(found)))
            groups.setdefault(found[0], []).append(i)
        if self.consumed is not None:
            for j, rows in groups.items():
                stale = self._stale(holders[j], [ids[i] for i in rows])
                if stale:
                    raise ValueError(_stale_refusal(holders[j], self.consumed.domain, stale))
        return [(holders[j], rows) for j, rows in groups.items()]

    def _stale(self, reader: EmbeddingReader, ids: Iterable[str]) -> list[str]:
        """The objects of ``ids``, held by ``reader``, whose consumed value is not the one it was extracted from."""
        consumed = self.consumed
        if consumed is None:
            return []
        if reader not in self._extracted:
            listed = reader.meta.get("objects")
            values = None
            if isinstance(listed, str):  # the extract stage's record of the list (encoders.jobs.finalize)
                values = staged_values(consumed.root, listed)
                if values is None:  # a store that holds the artifact without its list: the whole list instead
                    values = _rebuilt(reader, listed, consumed)
            self._extracted[reader] = values
        extracted = self._extracted[reader]
        if extracted is None:  # an artifact that records no object list
            return []
        now = consumed.values
        return [o for o in ids if o not in now or str(now[o]) != extracted.get(o)]

    def _pooled(self, ids: tuple[str, ...], layer: int) -> np.ndarray:
        parts = [(rows, r.pooled([ids[i] for i in rows], layer)) for r, rows in self._groups(ids, layer)]
        out = np.empty((len(ids), parts[0][1].shape[1]), dtype=np.float32)
        for rows, values in parts:
            out[rows] = values
        return out

    def _residues(self, ids: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        per_layer: list[list[np.ndarray]] = []
        for L in self.layers:
            by_position: list[np.ndarray] = [np.empty(0)] * len(ids)
            for reader, positions in self._groups(ids, L):
                found = reader.residues([ids[i] for i in positions], L)
                for i, values in zip(positions, found, strict=True):
                    by_position[i] = values
            per_layer.append(by_position)
        objects: list[np.ndarray] = []
        for i, oid in enumerate(ids):
            lengths = {len(layer_rows[i]) for layer_rows in per_layer}
            if len(lengths) != 1:
                raise ValueError(f"{oid}: residue lengths {sorted(lengths)} differ across layers {self.layers}")
            dtypes = {layer_rows[i].dtype for layer_rows in per_layer}
            if len(dtypes) != 1:  # bf16 bit patterns would be silently upcast as integers
                raise ValueError(f"{oid}: residues stored as {sorted(map(str, dtypes))} across layers {self.layers}")
            objects.append(np.concatenate([layer_rows[i] for layer_rows in per_layer], axis=1))
        width, longest = objects[0].shape[1], max(len(o) for o in objects)
        out = np.zeros((len(objects), longest, width), dtype=objects[0].dtype)
        valid = np.zeros((len(objects), longest), dtype=bool)
        for i, o in enumerate(objects):
            out[i, : len(o)] = o
            valid[i, : len(o)] = True
        tensor = torch.from_numpy(out)
        # bf16 residues are stored as uint16 bit patterns (encoders.store); reinterpret, do not convert.
        if out.dtype == np.uint16:
            tensor = tensor.view(torch.bfloat16)
        return tensor, torch.from_numpy(valid)
