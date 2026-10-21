"""Expert-parallel routed experts for ``kimi-ep``: each MoE layer's experts split into device blocks, applied grouped.

Kimi's own ``moe_infer`` loops over up to 384 experts in Python, each with three INT4 linears decompressed by many
small kernels per call. Here each MoE layer's experts are split into contiguous blocks, one per device (Kimi's own
expert-parallel slicing: ``n_experts / n_blocks`` each), each block's weights are kept packed on its device,
dequantized in one vectorized pass per call, and applied with one grouped GEMM. The weights are frozen: backward gives
only the gradient into the tokens, and dequantizes again rather than letting autograd keep the dense weights.

Ported from the original research trainer's expert-parallel module, on principles:

- the routing, the sort and the combine are the remote ``moe_infer``'s, so the routed experts see the same tokens with
  the same weights;
- deliberate deviations: blocks also accept dense experts (``nn.Linear``), so the float64 test model runs the same
  code; the GEMM mode is the backend's configured ``expert_gemm`` (``"grouped"`` or ``"loop"``), never discovered by
  trying and falling back; the expert activation is checked to be the config's ``silu`` instead of assumed.

The per-expert ``"loop"`` mode makes exactly the reference's calls (``x Wᵀ`` forward, ``g W`` backward per expert), so
it is the reference configuration the grouped mode is compared with on the real model.

How the blocks are dispatched is the backend's ``expert_dispatch`` option (:data:`ExpertDispatch`). It changes when
kernels and copies are issued, never which kernels run on which inputs on each device. How the sorted rows are cut into
the blocks' parts is its ``expert_rows`` option (:data:`ExpertRows`). How a compressed block is unpacked is its
``dequant`` option (:data:`Dequant`). Where the routed outputs are weighted and summed per token is its ``combine``
option (:data:`ExpertCombine`). With the backend's ``compile="regional"`` the combine runs compiled (``regional``).
"""

from __future__ import annotations

import contextlib
import types
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from latent_interconnect.receivers.kimi_k26.regional import compiled

__all__ = [
    "Dequant",
    "ExpertBlock",
    "ExpertCombine",
    "ExpertDispatch",
    "ExpertGemm",
    "ExpertRows",
    "block_bytes",
    "convert_moe",
    "dequantize_int4",
]

ExpertGemm = Literal["grouped", "loop"]
"""How a block applies its experts: one grouped GEMM (``F.grouped_mm``, CUDA) or one GEMM per expert (any device)."""
ExpertDispatch = Literal["serial", "concurrent", "streams"]
"""How a MoE layer issues its blocks (:func:`_moe_infer`):

- ``"serial"`` (the original order): each block's rows are copied to its device, its experts launched and its output
  copied back before the next block's rows are sent. A device-to-device copy waits on both devices' current streams,
  so each block waits for the one before it: the blocks run one after another.
- ``"concurrent"``: every block's experts are launched before any output is copied back, so the blocks run on their
  devices at the same time.
- ``"streams"`` (CUDA only; the backend's GPU default): ``"concurrent"``, and every cross-device copy of the dispatch
  and the gather runs on a side stream of its device pair (:class:`_PeerCopy`), in forward and in backward, joined to
  the current streams by stream waits; all the dispatch copies are issued before any block starts. A plain copy runs
  on the current streams, so the copies queue behind each other and behind the home device's own block.

Each device runs the same kernels on the same inputs in the same order in all three; only the launch and copy order
across devices changes, and a copy does not change a value."""
ExpertRows = Literal["slice", "split"]
"""How a MoE layer cuts its sorted rows into the blocks' parts (:func:`_parts`):

- ``"slice"`` (the original): one slice per block. In backward each slice's gradient is added into a
  zero-filled tensor of all the rows: one full-size fill and addition per block.
- ``"split"``: one ``torch.split``, whose backward concatenates the parts' gradients.

Every element of the rows' gradient is the gradient of the one part that holds it, added to zero (exact) or copied,
so both give the same values."""

Dequant = Literal["eager", "triton"]
"""How a compressed block's INT4 weights are unpacked on each call (:func:`dequantize_int4`):

- ``"eager"`` (the original): vectorized PyTorch, any device; five elementwise passes over the block, each writing a
  full-size temporary.
- ``"triton"`` (CUDA tensors only; the backend's GPU default): one Triton pass that reads each packed word and its
  group's scale once (``int4_triton``), multiplying in fp32 and rounding once, as PyTorch's bf16 multiply does: the same
  bits. A CPU tensor or a missing Triton is refused, never unpacked eagerly instead."""

ExpertCombine = Literal["home", "expert"]
"""Where a MoE layer weights its routed outputs by the gate and sums them per token (:func:`_moe_infer`):

- ``"home"`` (the original, the remote ``moe_infer``'s): every block's output rows are gathered on the layer's device,
  unsorted to ``(T, k, H)``, weighted in the gate weights' dtype (float32 on the real model) and summed over the k
  experts, then cast to the hidden dtype.
- ``"expert"``: each block weights its own rows on its own device and sums them per token
  into a ``(T, H)`` partial in the gate weights' dtype (:func:`_partial`); the layer's device adds the blocks' partials
  and casts once (:func:`_sum_partials`). The combine work runs on every block at once; the gather moves ``T × H``
  gate-dtype values per block instead of the block's rows in the hidden dtype (about twice the bytes on the full 8-GPU
  model, k = 8).

The same sum taken in another order (per block, then over blocks), so not bit-identical: the same function, to
rounding (``tests/unit/test_kimi_ep_options.py`` checks it in float64 and float32). Its ``index_add`` accumulates with
atomic adds on CUDA, nondeterministic in order like the backward of the original's ``x[token_of]`` gather; under
``torch.use_deterministic_algorithms(True)`` both take torch's deterministic implementation or raise."""

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_DEQUANT_CHUNK = 8  # experts dequantized at once: bounds the int32 temporaries (the original code's value)


def dequantize_int4(packed: Tensor, scale: Tensor, group_size: int, kernel: Dequant = "eager") -> Tensor:
    """Dequantize a stack of compressed-tensors ``pack-quantized`` INT4 weights.

    Args:
        packed: ``(E, out, in / 8)`` int32, eight 4-bit values per word along the input dimension.
        scale: ``(E, out, in / group_size)``.
        group_size: inputs per scale.
        kernel: the unpack (:data:`Dequant`).

    Returns:
        ``(E, out, in)`` in ``scale``'s dtype: ``(nibble - 8) · scale``, the arithmetic of compressed-tensors'
        ``unpack_from_int32`` and symmetric dequantization, vectorized over the eight nibbles and the experts.

    Raises:
        ValueError: for ``"triton"``, what ``int4_triton.dequantize_int4_triton`` refuses (a tensor that is not on a
            CUDA device, a group size that is not a positive multiple of 8, shapes that disagree).
        ImportError: for ``"triton"``, if Triton cannot be imported.
    """
    if kernel == "triton":
        try:
            from latent_interconnect.receivers.kimi_k26.int4_triton import dequantize_int4_triton  # noqa: PLC0415
        except ImportError as e:  # only this unpack needs Triton, so the eager paths import without it
            raise ImportError(
                f"dequant 'triton' on {packed.device} needs Triton (shipped with torch's CUDA wheels): {e}"
            ) from e
        return dequantize_int4_triton(packed, scale, group_size)
    n, out, words = packed.shape
    result = torch.empty(n, out, words * 8, dtype=scale.dtype, device=packed.device)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    for e0 in range(0, n, _DEQUANT_CHUNK):
        nibbles = ((packed[e0 : e0 + _DEQUANT_CHUNK].unsqueeze(-1) >> shifts) & 15).flatten(-2) - 8
        grouped = nibbles.to(scale.dtype).view(*nibbles.shape[:-1], -1, group_size)
        weights = grouped * scale[e0 : e0 + _DEQUANT_CHUNK].unsqueeze(-1)
        result[e0 : e0 + _DEQUANT_CHUNK] = weights.flatten(-2)
    return result


class ExpertBlock:
    """One device's contiguous block of one MoE layer's routed experts, its weights on that device.

    The block takes the experts' weights, one expert at a time: each expert linear's tensors are copied into the
    block's ``(n, ...)`` tensors on ``device``, allocated once per projection, and the linear is released right after
    (its tensors move to the meta device). The conversion's extra memory is therefore at most one projection of one
    block (no stacked temporary on the source device), and the source is freed without relying on Python's cycle
    collector. That reliance caused a GPU parity job's out-of-memory: compressed-tensors binds a closure over each
    ``CompressedLinear`` to the module itself, so dropping the expert modules frees nothing until a full collection
    runs, and every device came to hold its own layers' experts beside its blocks of every layer (measured on CPU with
    the real compressed-tensors; the budget is in ``ep.plan_memory``).

    Args:
        experts: the block's expert modules (``gate_proj``, ``up_proj``, ``down_proj``, bias-free), all compressed
            (INT4) or all dense, of one shape. They are emptied: their tensors are on the meta device afterwards.
        device: the block's device.
        options: the backend's speed options (``ep.resolve_options``), shared by reference: a compressed block unpacks
            its weights as their ``"dequant"`` says on each call (:data:`Dequant`).

    Raises:
        ValueError: if the experts mix compressed and dense projections, carry a bias, or disagree in group size or in
            a tensor's shape or dtype. Raised before any weight is taken: a refused block leaves its experts intact.
    """

    def __init__(self, experts: Sequence[nn.Module], device: torch.device, options: Mapping[str, str]) -> None:
        self.device = device
        self.n = len(experts)
        self.options = options
        self._packed: dict[str, tuple[Tensor, Tensor, int]] = {}
        self._dense: dict[str, Tensor] = {}
        plan = {name: _check_projection(name, [cast(Any, getattr(e, name)) for e in experts]) for name in _PROJECTIONS}
        for name, (attrs, group) in plan.items():
            taken = _take([getattr(e, name) for e in experts], attrs, device)
            if group is None:
                self._dense[name] = taken[0]
            else:
                self._packed[name] = (taken[0], taken[1], group)

    def weight(self, name: str) -> Tensor:
        """Return the block's ``(n, out, in)`` weight of one projection (dequantized afresh for a compressed block)."""
        if name in self._dense:
            return self._dense[name]
        packed, scale, group = self._packed[name]
        return dequantize_int4(packed, scale, group, cast(Dequant, self.options["dequant"]))


def _check_projection(name: str, linears: Sequence[Any]) -> tuple[tuple[str, ...], int | None]:
    """Check one projection across a block's experts; return the tensors to take and the group size (None: dense).

    Raises:
        ValueError: a bias, a mix of compressed and dense experts, differing group sizes, or a tensor whose shape or
            dtype differs between experts.
    """
    if any(getattr(m, "bias", None) is not None for m in linears):
        raise ValueError(f"expert {name} has a bias; routed experts are bias-free")
    compressed = [hasattr(m, "weight_packed") for m in linears]
    if all(compressed):
        groups = {int(m.quantization_scheme.weights.group_size) for m in linears}
        if len(groups) != 1:
            raise ValueError(f"expert {name}: group sizes {sorted(groups)} differ within a block")
        attrs: tuple[str, ...] = ("weight_packed", "weight_scale")
        group: int | None = groups.pop()
    elif not any(compressed):
        attrs, group = ("weight",), None
    else:
        raise ValueError(f"expert {name}: a block mixes compressed and dense experts")
    for attr in attrs:
        kinds = {(tuple(getattr(m, attr).shape), getattr(m, attr).dtype) for m in linears}
        if len(kinds) != 1:
            raise ValueError(f"expert {name}.{attr}: shapes/dtypes {sorted(map(str, kinds))} differ within a block")
    return attrs, group


def block_bytes(experts: Sequence[nn.Module]) -> int:
    """The bytes a block of these experts keeps (what :class:`ExpertBlock` takes; e.g. not ``weight_shape``).

    Raises:
        ValueError: as :class:`ExpertBlock` would refuse the experts.
    """
    total = 0
    for name in _PROJECTIONS:
        linears = [cast(Any, getattr(e, name)) for e in experts]
        attrs, _ = _check_projection(name, linears)
        total += sum(getattr(m, a).numel() * getattr(m, a).element_size() for m in linears for a in attrs)
    return total


def _take(linears: Sequence[nn.Module], attrs: Sequence[str], device: torch.device) -> list[Tensor]:
    """Copy each linear's ``attrs`` into stacked tensors on ``device``, releasing each linear as soon as it is copied.

    A cross-device copy runs on the source device's stream, so the source block freed after it is reused only after
    the copy (the caching allocator is stream-ordered).
    """
    first = cast(Any, linears[0])
    out = [
        torch.empty((len(linears), *getattr(first, a).shape), dtype=getattr(first, a).dtype, device=device)
        for a in attrs
    ]
    with torch.no_grad():
        for i, m in enumerate(linears):
            for dest, a in zip(out, attrs, strict=True):
                dest[i].copy_(getattr(m, a))
            m.to("meta")  # every tensor of the linear, the copied ones and the rest (e.g. weight_shape), released here
    return out


def _gemm(x: Tensor, w: Tensor, offsets: Tensor, ends: Sequence[int], *, transpose: bool, mode: ExpertGemm) -> Tensor:
    """Per-expert products of the rows of ``x`` (grouped by ``offsets``): ``x W_eᵀ`` if ``transpose`` else ``x W_e``.

    ``offsets`` are the cumulative row ends (int32, on the device) and ``ends`` the same on the host (the loop needs
    no device synchronization).
    """
    if mode == "grouped":
        b = w.transpose(-2, -1) if transpose else w
        out: Tensor = cast(Any, F).grouped_mm(x, b.to(x.dtype), offs=offsets)
        return out
    parts, start = [], 0
    for e, end in enumerate(ends):
        we = w[e].to(x.dtype)
        parts.append(F.linear(x[start:end], we) if transpose else x[start:end] @ we)
        start = end
    return torch.cat(parts) if parts else x.new_empty(0, w.shape[-2] if transpose else w.shape[-1])


def _on(device: torch.device) -> contextlib.AbstractContextManager[Any]:
    """Make ``device`` the current CUDA device for the block's kernels (no-op on CPU)."""
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


class _Experts(torch.autograd.Function):
    """``y = down(silu(gate x) · up x)`` per expert, for one block. Saves the two hidden activations only."""

    @staticmethod
    def forward(  # noqa: PLR0917  (autograd passes forward's inputs positionally)
        ctx: Any, x: Tensor, offsets: Tensor, block: ExpertBlock, ends: list[int], mode: ExpertGemm
    ) -> Tensor:
        with _on(block.device):
            h1 = _gemm(x, block.weight("gate_proj"), offsets, ends, transpose=True, mode=mode)
            h3 = _gemm(x, block.weight("up_proj"), offsets, ends, transpose=True, mode=mode)
            y = _gemm(F.silu(h1) * h3, block.weight("down_proj"), offsets, ends, transpose=True, mode=mode)
        ctx.save_for_backward(h1, h3, offsets)
        ctx.block, ctx.ends, ctx.mode = block, ends, mode
        return y

    @staticmethod
    def backward(ctx: Any, *grads: Tensor) -> tuple[Tensor, None, None, None, None]:
        (gy,) = grads
        h1, h3, offsets = ctx.saved_tensors
        block, ends, mode = ctx.block, ctx.ends, ctx.mode
        with _on(block.device):
            ga = _gemm(gy.contiguous(), block.weight("down_proj"), offsets, ends, transpose=False, mode=mode)
            with torch.enable_grad():  # the local derivative of silu(h1) · h3, as autograd takes it
                a1, a3 = h1.detach().requires_grad_(True), h3.detach().requires_grad_(True)
                g1, g3 = torch.autograd.grad(F.silu(a1) * a3, (a1, a3), ga)
            gx = _gemm(g1, block.weight("gate_proj"), offsets, ends, transpose=False, mode=mode)
            gx = gx + _gemm(g3, block.weight("up_proj"), offsets, ends, transpose=False, mode=mode)
        return gx, None, None, None, None


_STREAMS: dict[tuple[torch.device, torch.device], tuple[Any, Any]] = {}
"""The side streams of each (source, destination) CUDA device pair a copy used: made on the pair's first copy and kept
for the process's lifetime (two streams per ordered pair, at most 2 x 8 x 7 on one node; a stream holds no memory)."""


def _side_streams(src: torch.device, dst: torch.device) -> tuple[Any, Any]:
    """The side streams of one (source, destination) CUDA device pair: one on each device.

    Raises:
        ValueError: for a device without an index (``"cuda"``): the pair must name its devices.
    """
    if src.index is None or dst.index is None:
        raise ValueError(f"side-stream copies need indexed CUDA devices; got {src} and {dst}")
    key = (src, dst)
    if key not in _STREAMS:
        stream: Any = torch.cuda.Stream  # untyped in torch
        _STREAMS[key] = (stream(src), stream(dst))
    return _STREAMS[key]


def _side_copy(t: Tensor, dst: torch.device) -> Tensor:
    """Copy ``t`` to ``dst`` on the device pair's side streams; the destination's current stream waits for the copy.

    The source side stream first waits for the source's current stream (the input is ready); the copy runs with the
    side streams current on both devices, so it neither waits for nor holds the two current streams; the destination's
    current stream then waits for the destination side stream. ``record_stream`` keeps each tensor's memory from reuse
    until the stream using it is done. A tensor already on ``dst`` is returned as it is.

    Raises:
        ValueError: if either device is not CUDA (``expert_dispatch="streams"`` is CUDA-only; ``ep.KimiEP`` refuses it
            on other devices).
    """
    src = t.device
    if src == dst:
        return t
    if not src.type == dst.type == "cuda":
        raise ValueError(f"side-stream copies are between CUDA devices; got {src} -> {dst}")
    s_src, s_dst = _side_streams(src, dst)
    s_src.wait_stream(torch.cuda.current_stream(src))
    with torch.cuda.stream(s_src), torch.cuda.stream(s_dst):
        out = t.to(dst, non_blocking=True)
    t.record_stream(s_src)
    main = torch.cuda.current_stream(dst)
    main.wait_stream(s_dst)
    out.record_stream(main)
    return out


class _PeerCopy(torch.autograd.Function):
    """``t.to(device)`` on side streams (:func:`_side_copy`), and its gradient copied back the same way."""

    @staticmethod
    def forward(ctx: Any, t: Tensor, device: torch.device) -> Tensor:
        ctx.source = t.device
        return _side_copy(t, device)

    @staticmethod
    def backward(ctx: Any, *grads: Tensor) -> tuple[Tensor, None]:
        (g,) = grads
        return _side_copy(g.contiguous(), ctx.source), None


def _move(t: Tensor, device: torch.device, side: bool) -> Tensor:
    """Move ``t`` to ``device``: on side streams when ``side`` (:class:`_PeerCopy`), else the plain copy.

    The plain copy is the original one: non-blocking between CUDA devices (device-to-device copies are stream-ordered).
    """
    if t.device == device:
        return t
    if side:
        return cast(Tensor, cast(Any, _PeerCopy).apply(t, device))
    return t.to(device, non_blocking=t.device.type == device.type == "cuda")


def _parts(t: Tensor, sizes: list[int], rows: ExpertRows) -> list[Tensor]:
    """Cut ``t`` (rows grouped by block, ``sizes`` rows each) into the blocks' parts (:data:`ExpertRows`)."""
    if rows == "split":
        return list(torch.split(t, sizes))
    starts = [sum(sizes[:b]) for b in range(len(sizes))]
    return [t[start : start + n] for start, n in zip(starts, sizes, strict=True)]


def _combine(outs: list[Tensor], tokens: Tensor, order: Tensor, topk_ids: Tensor, topk_weight: Tensor) -> Tensor:
    """The remote ``moe_infer``'s combine (``combine="home"``): unsort the gathered rows, weight them, sum per token."""
    combined = torch.cat(outs, dim=0) if outs else tokens.new_empty(0, tokens.shape[-1])
    unsorted = torch.empty_like(combined)
    unsorted[order] = combined
    weighted = unsorted.view(*topk_ids.shape, -1).type(topk_weight.dtype) * topk_weight.unsqueeze(-1)
    return weighted.sum(dim=1).type(unsorted.dtype)


def _partial(y: Tensor, weight: Tensor, index: Tensor, n_tokens: int) -> Tensor:
    """One block's share of the combine, on the block's device: ``Σ_rows w · y`` per token.

    The partial is ``(T, H)`` in ``weight``'s dtype (the combine's upcast); a token none of the block's rows belong to
    gets zeros.
    """
    weighted = y.type(weight.dtype) * weight.unsqueeze(-1)
    return weighted.new_zeros(n_tokens, y.shape[-1]).index_add(0, index, weighted)


def _sum_partials(partials: list[Tensor], x: Tensor, dtype: torch.dtype) -> Tensor:
    """Add the blocks' partials (already on ``x``'s device) in ``dtype`` and cast once to ``x``'s dtype."""
    total = x.new_zeros(x.shape[0], x.shape[-1], dtype=dtype)
    for p in partials:
        total = total + p
    return total.type(x.dtype)


def _moe_infer(self: Any, x: Tensor, topk_ids: Tensor, topk_weight: Tensor) -> Tensor:
    """The remote ``DeepseekV3MoE.moe_infer`` (sort, apply, combine), with the experts applied per device block.

    Differentiable (the remote one is ``@torch.no_grad``); one host synchronization per layer (the expert token
    counts), as in the remote code. The module's ``_ep_options`` (the backend's speed options, read on every call) say
    how the blocks are issued (:data:`ExpertDispatch`), how their rows are cut (:data:`ExpertRows`), where the outputs
    are combined (:data:`ExpertCombine`) and whether the combine runs compiled (``regional``).
    """
    n_experts = self._ep_n_experts
    options: Mapping[str, str] = self._ep_options
    dispatch, rows_cut = options["expert_dispatch"], cast(ExpertRows, options["expert_rows"])
    side = dispatch == "streams"
    expert_side = options["combine"] == "expert"
    regional = options["compile"] == "regional"

    def region(fn: Any) -> Any:  # compile="regional": the compiled region
        return compiled(fn, self._ep_compile_limit) if regional else fn

    counts = topk_ids.new_zeros((topk_ids.shape[0], n_experts)).scatter_(1, topk_ids, 1).sum(dim=0)
    order = topk_ids.view(-1).argsort()
    token_of = order // topk_ids.shape[1]  # each sorted row's token
    tokens = x[token_of]
    host = counts.cpu()
    offsets_all = torch.cumsum(counts.view(len(self._ep_blocks), -1), 1).to(torch.int32)
    blocks: list[ExpertBlock] = self._ep_blocks
    block_counts = [host[b * block.n : (b + 1) * block.n] for b, block in enumerate(blocks)]
    sizes = [int(c.sum()) for c in block_counts]
    parts = _parts(tokens, sizes, rows_cut)
    active = [b for b, n in enumerate(sizes) if n]
    if expert_side:  # each sorted row's gate weight and token, cut as the rows are
        weight_parts = _parts(topk_weight.view(-1)[order], sizes, rows_cut)
        index_parts = _parts(token_of, sizes, rows_cut)

    def send(b: int) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        device = blocks[b].device
        rows = _move(parts[b], device, side)
        offsets = _side_copy(offsets_all[b], device) if side else _move(offsets_all[b], device, False)
        if not expert_side:
            return rows, offsets, None, None
        index = _side_copy(index_parts[b], device) if side else _move(index_parts[b], device, False)
        return rows, offsets, _move(weight_parts[b], device, side), index

    staged = {b: send(b) for b in active} if side else {}  # streams: every dispatch copy before any block starts
    outs: list[Tensor] = []
    launched: list[Tensor] = []
    for b in active:
        rows, offsets, weight, index = staged[b] if side else send(b)
        ends = torch.cumsum(block_counts[b], 0).tolist()
        y = cast(Any, _Experts).apply(rows, offsets, blocks[b], ends, self._ep_gemm)
        if weight is not None and index is not None:
            y = region(_partial)(y, weight, index, x.shape[0])
        if dispatch == "serial":
            outs.append(_move(y, x.device, False))
        else:
            launched.append(y)
    if dispatch != "serial":  # copied back only once every block is launched
        outs = [_move(y, x.device, side) for y in launched]
    if expert_side:
        return cast(Tensor, region(_sum_partials)(outs, x, topk_weight.dtype))
    return cast(Tensor, region(_combine)(outs, tokens, order, topk_ids, topk_weight))


def convert_moe(
    mlp: nn.Module, devices: Sequence[torch.device], gemm: ExpertGemm, activation: str, *, options: Mapping[str, str]
) -> None:
    """Convert one MoE module in place: its routed experts become ``len(devices)`` blocks, one per device.

    The module and its blocks keep ``options`` (the backend's speed options, ``ep.resolve_options``) by reference, so
    the backend's ``configure`` reaches them by updating that one mapping.

    A device may repeat (several blocks on one device: the CPU tests of expert parallelism). The blocks take the
    experts' weights one expert at a time (:class:`ExpertBlock`), so the experts' memory is released during the call,
    whatever references to the emptied expert modules remain; the shared experts and the gate stay as they are.

    Raises:
        ValueError: if the experts do not split evenly over the devices, or the activation is not ``silu``.
    """
    if activation != "silu":
        raise ValueError(f"the expert blocks implement silu experts; the config's activation is {activation!r}")
    module = cast(Any, mlp)
    experts = list(module.experts)
    if len(experts) % len(devices):
        raise ValueError(f"{len(experts)} experts do not split evenly over {len(devices)} expert-parallel devices")
    per = len(experts) // len(devices)
    module._ep_n_experts = len(experts)
    module._ep_blocks = [ExpertBlock(experts[i * per : (i + 1) * per], d, options) for i, d in enumerate(devices)]
    module._ep_gemm = gemm
    module._ep_options = options
    module.experts = nn.ModuleList()
    module.moe_infer = types.MethodType(_moe_infer, module)


def moe_modules(layers: Sequence[nn.Module], *, converted: bool = False) -> Iterator[nn.Module]:
    """Yield the MoE modules (an ``mlp`` with routed ``experts``) of the decoder layers, in layer order.

    With ``converted``, the MoE modules :func:`convert_moe` converted (their experts are blocks) instead.
    """
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        if hasattr(mlp, "_ep_blocks") if converted else len(getattr(mlp, "experts", ())) > 0:
            yield mlp
