"""Regional compilation for ``kimi-ep`` (``compile="regional"``), per backend, switched on and off on the loaded model.

``torch.compile`` of the small-kernel regions around the expert blocks, on CUDA devices only (``ep.KimiEP`` refuses the
choice on any other device). The regions:

- the packed attention, ``packed.mla_packed`` (the packed layout only; the padded layout runs the remote layers as
  they are);
- the routed combine, ``experts._combine``, and the expert-side combine's ``experts._partial`` and
  ``experts._sum_partials``;
- per MoE layer, the gate's and the shared experts' ``forward``.

The expert blocks (``experts._Experts``) and everything that holds INT4 weights stay eager by design, and so do the
operations between a region's graphs where dynamo breaks the graph (``fullgraph=False``): those are the intentional
eager parts, counted as graph breaks by :func:`dynamo_counts`. The choice is the backend's: a backend passes the
compiled attention to ``packed.packed_layer``, its MoE modules read their options and pick the compiled combine, and
its gates and shared experts get a compiled ``forward`` as an instance attribute, deleted again when the choice is
undone (:func:`compile_moe_parts`). Nothing module-level is swapped, so other backends in the process run eagerly.

**Recompiles.** ``torch.compile(dynamic=True)`` traces the batch and length dimensions symbolically: the regions read
the packing's lengths as tensors' shapes or plain ints, never a list's length, so another batch reuses the graph. A
code object can still gain cache entries per layer (each layer's modules are its own inputs) and per device, so each
region's limit is derived from the model: :func:`recompile_limit` entries per code object, against dynamo's default
of 8, with the accumulated limit at least its default of 256. The limits are set for each region's own calls only
(``torch._dynamo.config.patch`` around the call: :class:`Region`) with ``fail_on_recompile_limit_hit``, so a region that
exhausts its limit raises instead of running eagerly while counted as compiled. torch is pinned (2.11 in
``env/constraints/lip.txt``); the config names are that version's.

The compiled regions are made once per process and limit (:func:`compiled`) and shared by the backends that choose
them; a module's compiled ``forward`` is kept on the module between switches. They hold compiled code and guards, no
device memory of their own (no CUDA graphs: ``torch.compile``'s default mode). Compiled arithmetic is not the eager
arithmetic: the same function, to rounding (fused and reordered operations), so not bit-identical.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import torch
from torch import nn

__all__ = ["Compile", "Region", "compile_moe_parts", "compiled", "dynamo_counts", "recompile_limit"]

Compile = Literal["none", "regional"]
"""``kimi-ep``'s ``compile`` option: ``"none"`` (the original: eager) or ``"regional"`` (this module; CUDA)."""
_ACCUMULATED_LIMIT = 256  # dynamo's default accumulated recompile limit (torch 2.11), kept as a floor


def recompile_limit(depth: int, devices: int) -> int:
    """A region's limit of dynamo cache entries per code object: twice the larger of the layers and the devices.

    One entry per layer (a layer's modules are a region's inputs) or per device (the combine), doubled for dynamo's
    specialization of a size of 0 or 1 (a block that receives one token).
    """
    return 2 * max(depth, devices)


class Region:
    """``torch.compile(fn, dynamic=True, fullgraph=False)``, called with dynamo's limits scoped to the call.

    Args:
        fn: the region's function (or bound method).
        limit: the cache entries one code object may hold (:func:`recompile_limit`); exceeding it raises.
    """

    def __init__(self, fn: Callable[..., Any], limit: int) -> None:
        self.limit = limit
        self._compiled = torch.compile(fn, dynamic=True, fullgraph=False)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the compiled region under its limits; a region that exhausts them raises (module docstring)."""
        config = cast(Any, torch._dynamo.config)
        with config.patch(
            recompile_limit=self.limit,
            accumulated_recompile_limit=max(_ACCUMULATED_LIMIT, self.limit),
            fail_on_recompile_limit_hit=True,
        ):
            return self._compiled(*args, **kwargs)


_REGIONS: dict[tuple[Callable[..., Any], int], Region] = {}


def compiled(fn: Callable[..., Any], limit: int) -> Region:
    """``fn``'s :class:`Region` with ``limit``, made on the first request and kept for the process."""
    key = (fn, limit)
    if key not in _REGIONS:
        _REGIONS[key] = Region(fn, limit)
    return _REGIONS[key]


def _moe_parts(layers: Sequence[nn.Module]) -> list[nn.Module]:
    """The gate and the shared experts of every converted MoE layer (none of them holds INT4 weights)."""
    parts: list[nn.Module] = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "_ep_blocks"):
            continue
        for name in ("gate", "shared_experts"):
            part = getattr(mlp, name, None)
            if isinstance(part, nn.Module):
                parts.append(part)
    return parts


def compile_moe_parts(layers: Sequence[nn.Module], on: bool, limit: int) -> None:
    """Give every converted MoE layer's gate and shared experts a compiled ``forward`` (on), or remove it (off).

    The compiled ``forward`` (a :class:`Region` of the module's bound ``forward``) is an instance attribute over the
    class's method, so removing it restores the module exactly; each module keeps its region between switches.
    """
    for part in _moe_parts(layers):
        state = vars(part)
        if on and "forward" not in state:
            kept = state.get("_ep_compiled_forward")
            if kept is None:
                kept = Region(type(part).forward.__get__(part), limit)
                state["_ep_compiled_forward"] = kept
            state["forward"] = kept
        elif not on and "forward" in state:
            del state["forward"]


def dynamo_counts() -> dict[str, int]:
    """The process's dynamo counters so far, for a driver to record: frames compiled, graph breaks, distinct graphs.

    A frame is a function or its resumption after a graph break; first compiles and recompiles both count.
    """
    counters = cast(Any, torch._dynamo.utils).counters
    return {
        "frames_compiled": int(counters["frames"].get("ok", 0)),
        "graph_breaks": int(sum(counters["graph_break"].values())),
        "unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
    }
