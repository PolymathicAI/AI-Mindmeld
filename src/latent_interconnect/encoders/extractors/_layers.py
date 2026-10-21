"""The one read-layer rule of the extractors: which ``hidden_states`` indices a recipe reads, stored where.

Layer ``i`` is ``hidden_states[i]``: 0 the embedding output (or the first block's input), ``i`` the output of block
``i``, and ``D`` the final output, after the final norm when the model applies one. ``read`` is a tuple of indices,
``"all"`` (``0..D``, the read-layer scans) or, for an encoder whose paper input is its final output, ``"output"``:
layer ``D`` stored as :data:`~latent_interconnect.encoders.store.OUTPUT_LAYER` (the original registers' layer), with
``D`` recorded in the fingerprint's ``output_layer``. The protein models refuse ``"output"`` (name the final index:
their original embeddings are stored at it); the molecule models accept it. ``"pending"`` never runs.
"""

from __future__ import annotations

from collections.abc import Sequence

from latent_interconnect.encoders.store import OUTPUT_LAYER

__all__ = ["check_read", "read_layers"]


def check_read(read: str | Sequence[int], name: str, *, output: bool) -> None:
    """Refuse a ``read`` that cannot run (before the model is loaded and its depth known).

    Raises:
        ValueError: for ``"pending"``, or ``"output"`` where the extractor does not accept it.
    """
    if read == "pending":
        raise ValueError(f"{name}: read 'pending' cannot run")
    if read == "output" and not output:
        raise ValueError(f"{name}: read 'output'; name the layers (the final output is the last index)")


def read_layers(
    read: str | Sequence[int], depth: int, name: str, *, output: bool
) -> tuple[tuple[int, ...], dict[int, int]]:
    """Return the indices to compute and the key each is stored under (``OUTPUT_LAYER`` for ``"output"``).

    Raises:
        ValueError: as :func:`check_read`, or for an index outside ``0..depth``.
    """
    check_read(read, name, output=output)
    if read == "output":
        return (depth,), {depth: OUTPUT_LAYER}
    layers = tuple(range(depth + 1)) if read == "all" else tuple(int(L) for L in read)
    bad = [L for L in layers if not 0 <= L <= depth]
    if bad:
        raise ValueError(f"{name}: layers {bad} outside 0..{depth}")
    return layers, {L: L for L in layers}
