"""The ``kimi-ep`` backend: Kimi K2.6 resident on one node, for training (loss, gradients into the writes) and scoring.

Every decoder layer stays resident, the layers spread over the node's devices in contiguous ranges (``placement``),
and every MoE layer's routed experts are split over all the devices (expert parallelism, ``experts.convert_moe``):
one process, no ``torch.distributed`` (the connector is small and nothing else trains). Packed sequences
(``packed.Packing``) keep padding out of every layer; activation checkpointing bounds the activations.

The semantics are the shared rules, never a second home: the writes enter through ``kernel.apply_writes`` at the
input of decoder layer L, a request is checked before any model work (``receivers.base.check_score_request``; the
loss checks its writes then its targets in the same order), scores are partitioned by ``kernel.plan_batches``, every
logit comes from ``kernel.readout_logits``, and the loss is the receivers' ``LossTerms`` over the trainer's
micro-batch (not re-planned). They are checked against ``hf`` on the tiny Kimi model in float64 (loss, connector
gradients, one optimizer update; ``tests/unit/test_kimi_ep.py``).

Execution options (recorded in :meth:`KimiEP.execution`; what an execution option may change is the rule of
``run._main``, "Stage options"):

- ``layout``: ``"packed"`` (training) or ``"padded"`` (the remote layers' own eager forward: the reference);
- ``attention``: for the packed layout, ``"varlen"`` (FlashAttention-2, GPU) or ``"eager"`` (any device);
- ``expert_gemm``: ``"grouped"`` (one grouped GEMM per block, CUDA) or ``"loop"`` (one GEMM per expert: the
  reference's calls);
- ``checkpointing``: recompute each layer in backward instead of keeping its activations (on by default on GPUs;
  ``RunConfig.checkpointing: false``, or ``--option checkpointing=false`` of ``train`` or ``measure`` when the run
  config leaves ``checkpointing`` unset (refused when it sets it), turns it off;
  ``execution()`` reports it as ``"every layer"`` or ``"none"``).

Speed options, each set on the constructor, by :meth:`KimiEP.configure` or by ``--option NAME=VALUE`` of ``train`` or
``measure``. Like ``expert_gemm`` and ``checkpointing``, None (the default) is device-dependent: on GPUs
:data:`CUDA_DEFAULTS` (the fast choice, but the original for ``combine`` and ``compile``), otherwise the original
behaviour (:data:`REFERENCE_OPTIONS`); the original behaviour is always available by naming its value:

- ``expert_dispatch``: ``"serial"`` (the original), ``"concurrent"`` or ``"streams"`` (CUDA; the GPU default): how a
  MoE layer issues its expert blocks and their copies (``experts.ExpertDispatch``);
- ``expert_rows``: ``"slice"`` (the original) or ``"split"`` (the GPU default): how the sorted rows are cut into the
  blocks' parts (``experts.ExpertRows``);
- ``dequant``: ``"eager"`` (the original) or ``"triton"`` (CUDA; the GPU default): how the INT4 expert blocks unpack
  their weights (``experts.Dequant``);
- ``head``: ``"fp32"`` (the original) or ``"bf16"`` (the GPU default): the training loss's head (:data:`LossHead`).
  Changes the loss's rounding by design;
- ``combine``: ``"home"`` (the original; the default) or ``"expert"``: where the routed outputs are weighted and
  summed (``experts.ExpertCombine``). The same function in another summation order: not bit-identical;
- ``compile``: ``"none"`` (the original; the default) or ``"regional"`` (CUDA): ``torch.compile`` of the layers'
  small-kernel regions around the expert blocks (``regional``). Compiled arithmetic: not bit-identical.

The options are one mapping (:func:`resolve_options`, every value checked against :data:`SPEED_OPTIONS`), held by the
backend and shared by reference with its MoE modules and expert blocks; :meth:`KimiEP.configure` updates it in place.

``execution()`` records every option's effective value (its own key, named as the option; the other keys keep their
text) whenever one of them is not the original, and none of them when all are: a run on the original behaviour keeps the
original execution facts, so its resume states and completed runs continue unchanged, and every other run records each
choice. The trainer compares the whole dict (``training.trainer.Trainer._check_execution``): a record without option
keys resumes only when every effective value is the original one (on GPUs: ``--option`` with each value of
:data:`REFERENCE_OPTIONS`, which the refusal names), and a run on other choices resumes only under the same choices.

The GPU parity job compares the training configuration (packed, varlen, grouped, eight devices, checkpointed) with the
reference one (padded, eager, loop, not checkpointed) and with ``kimi-stream``.

Ported from the original research trainer and its expert-parallel module.
Deliberate deviations:

- the writes are the receivers' ``interp_v1`` through the one kernel (the original applied its own injection modes);
- the head is the float32 ``readout_logits`` (the original cast bf16 logits); ``head="bf16"`` is Kimi's own, for the
  training loss only;
- options replace the original's module-level switches and automatic fallbacks;
- the loss returns un-normalized terms, so the trainer weighs micro-batches.

Memory (a GPU parity job ran out of memory converting the experts, 39 minutes in): before anything loads, the device
memory the load needs is planned from the snapshot (:func:`plan_memory`) and checked against each GPU's free memory
(:class:`MemoryPlan.check`), so a plan that cannot fit fails in seconds. The conversion then checks, layer by layer,
that it conserves memory (the blocks take the experts' weights, never copy them).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal, cast, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from latent_interconnect.receivers.base import (
    RECEIVER_BACKENDS,
    LossTerms,
    Readout,
    ReceiverSpec,
    ScoringBackend,
    TrainingBackend,
    check_score_request,
    write_layers,
)
from latent_interconnect.receivers.kernel import (
    BatchPlan,
    LayerWrites,
    apply_writes,
    plan_batches,
    readout_logits,
    require_targets,
    select_writes,
    state_dtype,
)
from latent_interconnect.receivers.kimi_k26.common import attention_mask, padded_ids, padding_mask, read_ahead, read_out
from latent_interconnect.receivers.kimi_k26.experts import (
    Dequant,
    ExpertCombine,
    ExpertDispatch,
    ExpertGemm,
    ExpertRows,
    block_bytes,
    convert_moe,
    moe_modules,
)
from latent_interconnect.receivers.kimi_k26.packed import Attention, Packing, mla_packed, packed_layer
from latent_interconnect.receivers.kimi_k26.regional import Compile, compile_moe_parts, compiled, recompile_limit
from latent_interconnect.receivers.kimi_k26.weights import EMBED_PREFIX, HEAD_PREFIX, KimiSnapshot, open_snapshot
from latent_interconnect.receivers.render import RenderedPrompt
from latent_interconnect.writes.sources import WriteSource

__all__ = [
    "CUDA_DEFAULTS",
    "REFERENCE_OPTIONS",
    "RESERVE_BYTES",
    "KimiEP",
    "Layout",
    "LossHead",
    "MemoryBudgetError",
    "MemoryPlan",
    "available_bytes",
    "convert_conserving",
    "plan_memory",
    "recorded_execution",
    "resolve_options",
]

Layout = Literal["packed", "padded"]
"""How a forward pass lays out its sequences: one packed stream of real tokens, or a right-padded batch."""

LossHead = Literal["fp32", "bf16"]
"""The training loss's head (:meth:`KimiEP.loss`):

- ``"fp32"`` (the original): ``kernel.readout_logits``, the head in the state dtype (float32 for a bf16 body).
- ``"bf16"`` (the GPU default): Kimi's own head: the final states and the head in bf16, one bf16 matmul (tensor cores
  on GPUs), the logits upcast to float32 for the cross entropy, the token losses in the state dtype.

For the training loss only: ``score`` (in-job evaluation) and every readout keep ``readout_logits`` in the state dtype;
they call it through ``common.read_out``, never through the loss, so no score shares the bf16 path. The bf16 head is a
copy made when the choice is made (vocabulary x hidden in bf16: 2.19 GiB on K2.6, on the head's device, planned by
:func:`plan_memory`) and released when it is undone; the real head is bf16 promoted to float32, so its bf16 copy holds
the snapshot's own bits. A precision change: not exact."""

GiB = 1 << 30
RESERVE_BYTES = 8 * GiB
"""Free memory each GPU must keep beyond the load's planned peak: the CUDA libraries' workspaces, the peer mappings of
the cross-device copies and the caching allocator's rounding. Not a measurement: the load is planned exactly, this is
a floor for what the plan leaves out. A step's activations are not planned here: they come after the load, in each
device's capacity minus its planned ``final``, bounded by the batch and checkpointing (the GPU parity job's reference
pass runs without checkpointing).

The speed options add one persistent buffer, which :func:`plan_memory` plans: ``head="bf16"``'s bf16 head copy
(vocabulary x hidden x 2 bytes on the last layer's device: 2.19 GiB on K2.6). Their other allocations live within a
step, among the activations the plan leaves to each device's capacity minus its ``final``; per MoE layer, with T routed
tokens in the micro-batch, k experts per token, H hidden and B blocks (derived from the code; K2.6:
k = 8, H = 7168):

- ``combine="expert"``: a ``T x H`` float32 partial per block on the block's device, the B partials gathered on the
  layer's device (``4 B H`` bytes per token there: 224 KiB with 8 blocks, 112 KiB with 4) in place of the original
  combine's gathered rows, their unsorted copy and their float32 weighting (``8 k H`` bytes per token: 448 KiB);
  without checkpointing, the float32 rows saved for backward (``4 k H`` per token: 224 KiB) sit on the block devices
  instead of the layer's.
- ``expert_dispatch="streams"``: no new tensor; the copied rows, offsets and outputs (the original's copies) are kept
  from reuse until their side stream is done (``record_stream``), at most one layer's copies longer (``2 k H`` bytes
  per token each way: 112 KiB). The side streams hold no memory.
- ``expert_rows="split"``: less than ``"slice"`` in backward (one concatenation instead of a zero-filled full-size row
  gradient per block).
- ``dequant="triton"``: the same ``(E, out, in)`` result as the eager unpack, without the eager unpack's int32 and
  float temporaries (for 8 experts at once, at least ``8 x out x in x 4`` bytes: 448 MiB for a K2.6 projection).
- ``compile="regional"``: no CUDA graphs, so no graph memory pool; inductor's temporaries replace the eager
  intermediates of the same operations.

A batch measurement (``training.measure``, ``RunConfig.batch``) records its execution: one whose execution has no
option keys ran the reference options, and under other options the batch is measured again (the measurement is keyed
by the execution).
"""
SPEED_OPTIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "expert_dispatch": get_args(ExpertDispatch),
        "expert_rows": get_args(ExpertRows),
        "dequant": get_args(Dequant),
        "head": get_args(LossHead),
        "combine": get_args(ExpertCombine),
        "compile": get_args(Compile),
    }
)
"""The speed options (module docstring) and each one's values, the original first: the one definition."""
REFERENCE_OPTIONS: Mapping[str, str] = MappingProxyType({name: values[0] for name, values in SPEED_OPTIONS.items()})
"""The speed options' original values: the defaults without CUDA devices; ``execution()`` omits the options when every
effective value is one of these."""
CUDA_DEFAULTS: Mapping[str, str] = MappingProxyType(
    {
        "expert_dispatch": "streams",
        "expert_rows": "split",
        "dequant": "triton",
        "head": "bf16",
        "combine": "home",
        "compile": "none",
    }
)
"""The speed options' defaults on CUDA devices (module docstring)."""
NEEDS_CUDA: frozenset[tuple[str, str]] = frozenset(
    {("expert_dispatch", "streams"), ("dequant", "triton"), ("compile", "regional")}
)
"""The choices that run on CUDA devices only: :class:`KimiEP` refuses them on any other (the one check of the rule)."""
_CONSERVATION_TOLERANCE = 0.01
"""Allowed growth of the total allocated memory over one layer's conversion, as a fraction of that layer's expert
bytes. The blocks take exactly the experts' bytes; a conversion that kept its source grows by 100 %."""


class MemoryBudgetError(RuntimeError):
    """The planned load does not fit the devices' free memory, or a conversion did not conserve memory."""


@dataclass(frozen=True)
class MemoryPlan:
    """The device memory a ``kimi-ep`` load needs, per device, in bytes (the model's tensors only).

    Attributes:
        devices: the distinct devices, in first-use order.
        build: each device's peak while the layers are built (each layer is loaded beside its built copy).
        convert: each device's peak while the experts are split into blocks: an upper bound, every block of a layer
            counted on its device before the layer's own experts are released.
        final: each device's resident model once loaded (the head promoted to the state dtype).
    """

    devices: tuple[torch.device, ...]
    build: tuple[int, ...]
    convert: tuple[int, ...]
    final: tuple[int, ...]

    @property
    def peak(self) -> tuple[int, ...]:
        """Each device's peak over the load."""
        return tuple(max(b, c, f) for b, c, f in zip(self.build, self.convert, self.final, strict=True))

    def table(self) -> dict[str, dict[str, float]]:
        """The plan in GiB per device (for logs and reports)."""
        return {
            str(d): {"build": b / GiB, "convert": c / GiB, "final": f / GiB}
            for d, b, c, f in zip(self.devices, self.build, self.convert, self.final, strict=True)
        }

    def check(self, free: Sequence[int], reserve: int = RESERVE_BYTES) -> None:
        """Refuse a load whose peak plus ``reserve`` exceeds a device's ``free`` bytes (one entry per device).

        Raises:
            MemoryBudgetError: naming every device that does not fit, with its planned phases and its free memory.
        """
        short = [
            f"{d}: peak {p / GiB:.1f} GiB (build {b / GiB:.1f}, convert {c / GiB:.1f}, final {f / GiB:.1f})"
            f" + reserve {reserve / GiB:.1f} GiB > free {room / GiB:.1f} GiB"
            for d, p, b, c, f, room in zip(
                self.devices, self.peak, self.build, self.convert, self.final, free, strict=True
            )
            if p + reserve > room
        ]
        if short:
            raise MemoryBudgetError("kimi-ep's load does not fit: " + "; ".join(short))


def available_bytes(device: torch.device) -> tuple[int, int]:
    """An estimated budget for a new allocation on a CUDA ``device``: driver-free memory and the reusable cache.

    The caching allocator serves a new allocation from its reserved but unallocated blocks first, so after training
    steps the driver's free memory alone understates the budget. The sum is an estimate: it does not
    ensure a contiguous block of the size asked for; the actual allocation (or its refusal) decides. The one
    definition the load plan's check and ``configure`` use; it frees nothing (no ``empty_cache``), so later steps
    allocate as before.
    """
    free = int(torch.cuda.mem_get_info(device)[0])
    cached = int(torch.cuda.memory_reserved(device)) - int(torch.cuda.memory_allocated(device))
    return free, max(cached, 0)


def resolve_options(cuda: bool, **given: str | None) -> dict[str, str]:
    """The speed options' effective values: each one given, else its default, keyed and ordered as the reference.

    None takes the default: :data:`CUDA_DEFAULTS` on CUDA devices and :data:`REFERENCE_OPTIONS` otherwise. Every caller
    (the constructor, :meth:`KimiEP.configure`, a driver) goes through this check.

    Raises:
        ValueError: for a name that is not a speed option, or a value that is not one of its :data:`SPEED_OPTIONS`.
    """
    unknown = sorted(set(given) - set(SPEED_OPTIONS))
    if unknown:
        raise ValueError(f"not kimi-ep speed options: {unknown}; the options are {list(SPEED_OPTIONS)}")
    defaults = CUDA_DEFAULTS if cuda else REFERENCE_OPTIONS
    chosen = {}
    for name, allowed in SPEED_OPTIONS.items():
        value = given.get(name)
        if value is not None and value not in allowed:
            raise ValueError(f"kimi-ep option {name}={value!r}: one of {list(allowed)}")
        chosen[name] = defaults[name] if value is None else value
    return chosen


def recorded_execution(facts: Mapping[str, str], options: Mapping[str, str]) -> dict[str, str]:
    """The execution facts with the speed options as :meth:`KimiEP.execution` records them.

    Every effective value (``options``, keyed as :data:`REFERENCE_OPTIONS`) whenever one is not the reference value,
    none of them when all are: the one home of that rule.
    """
    return {**facts, **options} if dict(options) != dict(REFERENCE_OPTIONS) else dict(facts)


def _nbytes(module: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in [*module.parameters(), *module.buffers()])


def _layer_bytes(snapshot: KimiSnapshot, index: int, dtype: torch.dtype) -> tuple[int, int, int]:
    """Decoder layer ``index``'s bytes on a device: all, its routed experts', and what its expert blocks keep.

    From the layer's skeleton on the meta device (``KimiSnapshot.skeleton``, the construction ``build_layer`` loads
    into), so buffers that are not in the checkpoint (the rotary tables: 67 MB per layer on K2.6) and the compressed
    tensors' dtypes are counted exactly, without reading a weight.
    """
    layer = snapshot.skeleton(index, dtype, torch.device("meta"))
    moe = [list(cast(Any, m).experts) for m in moe_modules([layer])]
    return _nbytes(layer), sum(_nbytes(e) for ex in moe for e in ex), sum(block_bytes(ex) for ex in moe)


def _conversion_order(homes: Sequence[torch.device], moe: Sequence[int]) -> list[int]:
    """The MoE layers in conversion order: round-robin over the layers' home devices.

    In layer order, the last devices would receive a block of every earlier layer before giving away their own (about
    +60 GB on GPUs 6 and 7 of an 8×H200 node in the original trainer); interleaving the home devices keeps each device's
    give and take balanced.
    """
    by_home: dict[torch.device, list[int]] = {}
    for li in moe:
        by_home.setdefault(homes[li], []).append(li)
    lists = list(by_home.values())
    return [lst[r] for r in range(max(map(len, lists), default=0)) for lst in lists if r < len(lst)]


def _layer_homes(devices: Sequence[torch.device], depth: int) -> list[torch.device]:
    """Contiguous layer ranges per device, as the original trainer placed them."""
    n = len(devices)
    return [devices[min(n - 1, (li * n) // depth)] for li in range(depth)]


def plan_memory(
    snapshot: KimiSnapshot, devices: Sequence[torch.device], dtype: torch.dtype, *, head: LossHead = "fp32"
) -> MemoryPlan:
    """Plan the device memory of a ``kimi-ep`` load, phase by phase, as :class:`KimiEP` loads (no weight is read).

    The embedding (first device), then each decoder layer on its home device, loaded beside its built copy; then the
    experts split over ``devices``, in the conversion order; then the final norm and the head on the last layer's
    device, the head promoted to the state dtype beside its body-dtype copy; with ``head="bf16"``, the head's bf16 copy
    beside the promoted head (:data:`LossHead`; the speed options' only persistent buffer, :data:`RESERVE_BYTES`).
    """
    distinct = tuple(dict.fromkeys(devices))
    at = {d: i for i, d in enumerate(distinct)}
    homes = _layer_homes(devices, snapshot.depth)
    sizes = [_layer_bytes(snapshot, li, dtype) for li in range(snapshot.depth)]
    matrix = int(snapshot.config.vocab_size) * int(snapshot.config.hidden_size)
    held = [0] * len(distinct)
    build = [0] * len(distinct)
    held[at[homes[0]]] += matrix * dtype.itemsize
    for li, (total, _, _) in enumerate(sizes):
        k = at[homes[li]]
        build[k] = max(build[k], held[k] + 2 * total)
        held[k] += total
    convert = held.copy()
    for li in _conversion_order(homes, [li for li, (_, e, _) in enumerate(sizes) if e]):
        _, experts, kept = sizes[li]
        for d in devices:
            held[at[d]] += kept // len(devices)
            convert[at[d]] = max(convert[at[d]], held[at[d]])
        held[at[homes[li]]] -= experts
    last = at[homes[-1]]
    held[last] += int(snapshot.config.hidden_size) * dtype.itemsize  # the final norm
    body, promoted = matrix * dtype.itemsize, matrix * state_dtype(dtype).itemsize
    convert[last] = max(convert[last], held[last] + body + (promoted if promoted != body else 0))
    held[last] += promoted
    if head == "bf16":
        held[last] += matrix * torch.bfloat16.itemsize
    return MemoryPlan(distinct, tuple(build), tuple(convert), tuple(held))


def convert_conserving(
    module: nn.Module,
    index: int,
    devices: Sequence[torch.device],
    gemm: ExpertGemm,
    activation: str,
    *,
    options: Mapping[str, str],
) -> None:
    """Convert MoE layer ``index``'s ``module`` (``experts.convert_moe``), checking on GPUs that it conserves memory.

    The total allocated memory over the GPUs among ``devices`` may grow by at most :data:`_CONSERVATION_TOLERANCE` of
    the layer's expert bytes: a conversion that kept its source fails at the first layer, not when the devices fill.
    The GPU parity job's step 0 runs it on one real layer, so a release regression fails in minutes, before the stream
    pass.

    Raises:
        MemoryBudgetError: if the conversion grew the total allocated memory.
    """
    cuda = sorted({d for d in devices if d.type == "cuda"}, key=str)
    experts = sum(_nbytes(m) for m in cast(Any, module).experts)
    before = sum(torch.cuda.memory_allocated(d) for d in cuda)
    convert_moe(module, devices, gemm, activation, options=options)
    if cuda:
        grown = sum(torch.cuda.memory_allocated(d) for d in cuda) - before
        if grown > _CONSERVATION_TOLERANCE * experts:
            raise MemoryBudgetError(
                f"converting layer {index}'s experts ({experts / GiB:.2f} GiB) grew the allocated memory by"
                f" {grown / GiB:.2f} GiB: the blocks must take the experts' weights, not copy them"
            )
        torch.cuda.empty_cache()


@RECEIVER_BACKENDS.register("kimi-ep")
class KimiEP:
    """Kimi K2.6 ``train`` and ``score``, resident with expert parallelism.

    Implements :class:`~latent_interconnect.receivers.base.TrainingBackend` (which includes scoring, for in-job
    evaluation on the loaded model).

    Args:
        spec: the receiver config (``configs/receivers/kimi-k2.6.yaml``).
        weights: the weights directory (``kimi_k26.weights.open_snapshot``; None: the pinned snapshot).
        devices: the devices the layers and the expert blocks use (None: every visible GPU). A device may repeat
            (several expert blocks on one device, as the CPU tests of expert parallelism do).
        dtype: the body's precision (bf16, Kimi's native; the CPU parity tests use float64).
        plan: the batch plan of ``score`` (None: ``BatchPlan()``); ``loss`` runs the trainer's micro-batch as given.
        layout: ``"packed"`` or ``"padded"``.
        attention: the packed attention (None: ``"varlen"`` on GPUs, ``"eager"`` otherwise); the padded layout runs
            the remote layers' eager attention.
        expert_gemm: ``"grouped"`` or ``"loop"`` (None: ``"grouped"`` on GPUs, ``"loop"`` otherwise).
        checkpointing: recompute layers in backward (None: on for GPUs, off otherwise).
        expert_dispatch: how the MoE layers issue their expert blocks (``experts.ExpertDispatch``; None: ``"streams"``
            on GPUs, ``"serial"`` otherwise).
        expert_rows: how the MoE layers cut their rows into the blocks' parts (``experts.ExpertRows``; None:
            ``"split"`` on GPUs, ``"slice"`` otherwise).
        dequant: how the INT4 expert blocks unpack their weights (``experts.Dequant``; None: ``"triton"`` on GPUs,
            ``"eager"`` otherwise).
        head: the training loss's head (:data:`LossHead`; None: ``"bf16"`` on GPUs, ``"fp32"`` otherwise).
        combine: where the MoE layers weight and sum their routed outputs (``experts.ExpertCombine``; None:
            ``"home"``).
        compile: ``"regional"`` compiles the small-kernel regions around the expert blocks (``regional``; CUDA; None:
            ``"none"``). The parameter is named as the option it sets (``--option
            compile=...``, ``execution()["compile"]``), so it shadows the builtin within the constructor and
            ``configure`` only.

    Raises:
        ValueError: if no device is given or visible, ``"varlen"`` is asked for the padded layout, an option needs
            CUDA on another device, a speed option's value is not one of its :data:`SPEED_OPTIONS`, or the experts do
            not split evenly over the devices.
    """

    reads_banks: ClassVar[bool] = False  # takes live connectors and banks alike (in-process)
    reference_options: ClassVar[Mapping[str, str]] = REFERENCE_OPTIONS  # receivers.base.TrainingBackend

    def __init__(
        self,
        spec: ReceiverSpec,
        *,
        weights: Path | None = None,
        devices: Sequence[str] | None = None,
        dtype: torch.dtype = torch.bfloat16,
        plan: BatchPlan | None = None,
        layout: Layout = "packed",
        attention: Attention | None = None,
        expert_gemm: ExpertGemm | None = None,
        checkpointing: bool | None = None,
        expert_dispatch: ExpertDispatch | None = None,
        expert_rows: ExpertRows | None = None,
        dequant: Dequant | None = None,
        head: LossHead | None = None,
        combine: ExpertCombine | None = None,
        compile: Compile | None = None,
    ) -> None:
        names = list(devices) if devices is not None else [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not names:
            raise ValueError("kimi-ep needs devices: none were given and no GPU is visible")
        self.devices = [torch.device(d) for d in names]
        cuda = all(d.type == "cuda" for d in self.devices)
        self.layout: Layout = layout
        self.attention: Attention = attention or ("varlen" if cuda else "eager")
        self.expert_gemm: ExpertGemm = expert_gemm or ("grouped" if cuda else "loop")
        self.checkpointing = cuda if checkpointing is None else checkpointing
        self._options = resolve_options(
            cuda,
            expert_dispatch=expert_dispatch,
            expert_rows=expert_rows,
            dequant=dequant,
            head=head,
            combine=combine,
            compile=compile,
        )
        self._head_bf16: Tensor | None = None
        self._check_options(self._options)
        self.spec = spec
        self.dtype = dtype
        self.plan = BatchPlan() if plan is None else plan
        self.snapshot = open_snapshot(spec, weights)
        self.weights = self.snapshot.path
        self.device_of = _layer_homes(self.devices, self.snapshot.depth)
        self.memory = plan_memory(self.snapshot, self.devices, dtype, head=cast(LossHead, self._options["head"]))
        if cuda:  # before anything loads: a load that cannot fit fails here, in seconds
            self.memory.check([sum(available_bytes(d)) for d in self.memory.devices])
        self.embedding = self.snapshot.matrix(EMBED_PREFIX, self.device_of[0], dtype)
        self.layers = self._build_layers()
        self._convert_experts()
        last = self.device_of[-1]
        self.norm = self.snapshot.build_norm(last, dtype)
        # the head is a body weight (dtype), promoted once to the state dtype the one head computes in
        self.head = self.snapshot.matrix(HEAD_PREFIX, last, dtype).to(state_dtype(dtype))
        self._apply_options()

    def configure(
        self,
        *,
        layout: Layout | None = None,
        attention: Attention | None = None,
        expert_gemm: ExpertGemm | None = None,
        checkpointing: bool | None = None,
        expert_dispatch: ExpertDispatch | None = None,
        expert_rows: ExpertRows | None = None,
        dequant: Dequant | None = None,
        head: LossHead | None = None,
        combine: ExpertCombine | None = None,
        compile: Compile | None = None,
    ) -> None:
        """Change execution options on the loaded model (None keeps an option); every option switches both ways.

        The numerics follow the options (``execution()`` reports them). This lets a parity check compare the training
        configuration with the reference one on one resident copy of the model: two do not fit on one node.

        The memory plan (``memory``, ``memory_report``) follows the options. Switching to ``head="bf16"`` allocates the
        head's bf16 copy on the loaded model, so on a GPU it is checked against that device's free memory first.

        If applying the options fails in any way, the previous options are restored and applied again before the
        failure is raised.

        Raises:
            ValueError: for an option the devices or the layout cannot run; the backend keeps its options.
            MemoryBudgetError: if the bf16 head copy exceeds the head's GPU's estimated budget now
                (:func:`available_bytes`); the backend keeps its options.
        """
        given = {"layout": layout, "attention": attention, "expert_gemm": expert_gemm, "checkpointing": checkpointing}
        speed = {
            "expert_dispatch": expert_dispatch,
            "expert_rows": expert_rows,
            "dequant": dequant,
            "head": head,
            "combine": combine,
            "compile": compile,
        }
        cuda = all(d.type == "cuda" for d in self.devices)
        options = resolve_options(cuda, **{**self._options, **{k: v for k, v in speed.items() if v is not None}})
        before = {name: getattr(self, name) for name in given}
        for name, value in given.items():
            if value is not None:
                setattr(self, name, value)
        try:
            self._check_options(options)
            self._check_head_copy(options)
        except (ValueError, MemoryBudgetError):
            for name, value in before.items():
                setattr(self, name, value)
            raise
        head_changed = options["head"] != self._options["head"]
        previous = dict(self._options)
        self._options.update(options)  # in place: the MoE modules and blocks hold this mapping
        try:
            self._apply_options()
        except BaseException:  # any failure: the options as they were, applied again, then the failure
            self._options.update(previous)
            for name, value in before.items():
                setattr(self, name, value)
            self._apply_options()
            raise
        if head_changed:
            self.memory = plan_memory(self.snapshot, self.devices, self.dtype, head=cast(LossHead, options["head"]))

    def _check_head_copy(self, options: Mapping[str, str]) -> None:
        """Refuse a switch to the bf16 head whose copy exceeds its GPU's estimated budget (:func:`available_bytes`)."""
        dev = self.head.device
        if options["head"] != "bf16" or self._head_bf16 is not None or dev.type != "cuda":
            return
        need = self.head.numel() * torch.bfloat16.itemsize
        free, cached = available_bytes(dev)
        if need > free + cached:
            raise MemoryBudgetError(
                f"head='bf16': its copy ({need / GiB:.2f} GiB) exceeds the estimated budget of {dev}:"
                f" {free / GiB:.2f} GiB free on the device and {cached / GiB:.2f} GiB reusable in the caching allocator"
            )

    def _check_options(self, options: Mapping[str, str]) -> None:
        cuda = all(d.type == "cuda" for d in self.devices)
        if self.layout == "padded" and self.attention != "eager":
            raise ValueError("the padded layout runs the remote eager attention; 'varlen' needs the packed layout")
        if not cuda and (self.attention == "varlen" or self.expert_gemm == "grouped"):
            raise ValueError(f"attention {self.attention!r} / expert_gemm {self.expert_gemm!r} need CUDA devices")
        for name, value in options.items():
            if not cuda and (name, value) in NEEDS_CUDA:
                raise ValueError(
                    f"{name} {value!r} needs CUDA devices; {name}={REFERENCE_OPTIONS[name]!r} runs anywhere"
                )

    def _apply_options(self) -> None:
        """Make or release the bf16 head copy, (un)compile the MoE regions and set the GEMM mode on the MoE modules.

        The other options reach the MoE modules and blocks through the mapping they share with the backend.
        """
        if self._options["head"] == "bf16":
            self._head_bf16 = self.head.to(torch.bfloat16) if self._head_bf16 is None else self._head_bf16
        else:
            self._head_bf16 = None
        limit = recompile_limit(len(self.layers), len(set(self.devices)))
        compile_moe_parts(self.layers, self._options["compile"] == "regional", limit)
        for mlp in moe_modules(self.layers, converted=True):
            cast(Any, mlp)._ep_gemm, cast(Any, mlp)._ep_compile_limit = self.expert_gemm, limit

    def options(self) -> dict[str, str]:
        """The speed options' effective values (a copy; module docstring), keyed as :data:`REFERENCE_OPTIONS`."""
        return dict(self._options)

    def _build_layers(self) -> list[nn.Module]:
        """Build every decoder layer on its device, reading the next layer's shards ahead in the background."""
        snapshot, depth = self.snapshot, self.snapshot.depth
        prefixes = [snapshot.layer_prefix(i) for i in range(depth)]
        return [snapshot.build_layer(i, self.device_of[i], self.dtype) for i in read_ahead(snapshot, prefixes, depth)]

    def _convert_experts(self) -> None:
        """Split every MoE layer's experts over the devices, in :func:`_conversion_order` (:func:`convert_conserving`).

        Raises:
            MemoryBudgetError: if a layer's conversion grew the total allocated memory.
        """
        moe = {li: m for li, layer in enumerate(self.layers) for m in moe_modules([layer])}
        activation = str(self.snapshot.config.hidden_act)
        for li in _conversion_order(self.device_of, list(moe)):
            convert_conserving(moe[li], li, self.devices, self.expert_gemm, activation, options=self._options)

    # -- capabilities ------------------------------------------------------------------------------------------------

    def execution(self) -> dict[str, str]:
        """Return the static execution facts of :mod:`latent_interconnect.receivers.base`: no history.

        The speed options are added, all of them, unless every effective value is the original (module docstring).
        """
        quantization = self.snapshot.quantization
        names = sorted({torch.cuda.get_device_name(d) if d.type == "cuda" else d.type for d in self.devices})
        facts = {
            "backend": "kimi-ep",
            "dtype": str(self.dtype).removeprefix("torch."),
            "quantization": "none" if quantization is None else f"compressed-tensors {quantization.format}",
            "device_name": ",".join(names),
            "device_count": str(len(set(self.devices))),
            "attention": "padded-eager" if self.layout == "padded" else f"packed-{self.attention}",
            "experts": f"{self.expert_gemm}, {len(self.devices)} blocks",
            "checkpointing": "every layer" if self.checkpointing else "none",
            "revision": self.spec.revision,
            "weights": str(self.weights),
        }
        return recorded_execution(facts, self._options)

    def memory_report(self) -> dict[str, dict[str, float]]:
        """The planned memory per device (GiB) and, on GPUs, the allocated memory now (``allocated``, GiB)."""
        table = self.memory.table()
        for d in self.memory.devices:
            if d.type == "cuda":
                table[str(d)]["allocated"] = torch.cuda.memory_allocated(d) / GiB
        return table

    def placement(self) -> Mapping[int, torch.device]:
        """Return layer → the device of decoder layer L's input (layer 0's is the embedding's)."""
        return dict(enumerate(self.device_of))

    def loss(self, prompts: Sequence[RenderedPrompt], writes: Mapping[int, LayerWrites]) -> LossTerms:
        """Weighted LM loss terms ``Σ w·ℓ`` and ``Σ w`` over the target tokens, with gradients into the writes.

        Also each prompt's detached token NLLs (``LossTerms.token_nll``, for the trainer's logs).

        The prompts are the trainer's micro-batch, run as one forward pass. The writes are checked first
        (``receivers.base.write_layers``), then the targets (``kernel.require_targets``): ``check_score_request``'s
        order.

        Raises:
            ValueError: a write at a layer the model does not have, or a prompt without target tokens.
        """
        write_layers(writes, self.spec.depth)
        targets = require_targets(prompts, "loss")
        sequences = [p.input_ids + t.token_ids for p, t in zip(prompts, targets, strict=True)]
        final = self._final(sequences, writes)
        rows: list[int] = []
        positions: list[int] = []
        labels: list[int] = []
        weights: list[float] = []
        for i, (p, t) in enumerate(zip(prompts, targets, strict=True)):
            n = len(p.input_ids)
            rows += [i] * len(t.token_ids)
            positions += range(n - 1, n - 1 + len(t.token_ids))  # position t - 1 predicts token t
            labels += t.token_ids
            weights += t.weights
        dev = final.device
        states = final[torch.tensor(rows, device=dev), torch.tensor(positions, device=dev)]
        target = torch.tensor(labels, device=dev)
        if self._head_bf16 is None:
            ell = F.cross_entropy(readout_logits(states, self.head), target, reduction="none")
        else:  # head="bf16" (LossHead)
            logits = (states.to(torch.bfloat16) @ self._head_bf16.T).float()
            ell = F.cross_entropy(logits, target, reduction="none").to(state_dtype(self.dtype))
        w = torch.tensor(weights, dtype=ell.dtype, device=dev)
        per_prompt = torch.split(ell.detach(), [len(t.token_ids) for t in targets])  # for the trainer's logs only
        return LossTerms(weighted_sum=(w * ell).sum(), weight_sum=w.sum(), token_nll=tuple(per_prompt))

    def score(self, prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource], readout: Readout) -> Tensor:
        """Teacher-forced readout with writes (in-job evaluation on the loaded model).

        Raises:
            ValueError: by the one score-request rule (``receivers.base.check_score_request``), checked before any
                model work.
        """
        with torch.no_grad():
            request = check_score_request(prompts, sources, readout, self.spec.depth, self.plan)
            if not prompts:
                width = readout.width
                return torch.empty((0,) if width is None else (0, width), dtype=state_dtype(self.dtype))
            if request.targets is None:
                sequences = [p.input_ids for p in prompts]
            else:
                sequences = [p.input_ids + t.token_ids for p, t in zip(prompts, request.targets, strict=True)]
            partition = plan_batches(request.lengths, self.plan)
            finals = [
                self._final([sequences[i] for i in batch], select_writes(request.writes, batch)) for batch in partition
            ]
            return read_out(prompts, sequences, partition, finals, readout=readout, head=self.head)

    # -- the body ----------------------------------------------------------------------------------------------------

    def _final(self, sequences: Sequence[Sequence[int]], writes: Mapping[int, LayerWrites]) -> Tensor:
        """Run the body on the sequences with the writes; return the ``(B, T, d)`` final normed states, right-padded.

        Every layer's rotary table first covers the longest sequence (:meth:`_cover_positions`), outside any layer.
        """
        self._cover_positions(max(len(s) for s in sequences))
        if self.layout == "padded":
            return self._final_padded(sequences, writes)
        return self._final_packed(sequences, writes)

    def _cover_positions(self, longest: int) -> None:
        """Make every layer's rotary table cover positions ``[0, longest)``, before any layer runs.

        Kimi's rotary module builds its cos/sin table on its first call and again whenever a call's length exceeds the
        cached one (the remote ``DeepseekV3RotaryEmbedding.forward``). Inside a layer that rebuild happens in the
        checkpointed forward and not in its recompute (the table then covers the length), so the two run different
        code, and a compiled region traces different graphs for them: the recompute's saved tensors then differ from
        the forward's (``torch.utils.checkpoint``'s ``CheckpointError``). Called here, with the dtype and device the
        layer's call would use, the rebuild is the one that call would make, made earlier: a position's cos/sin
        depends on the position alone, not on the cached length, so the tables and every output are the same values,
        and no rebuild happens inside a layer.
        """
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                rotary = cast(Any, layer).self_attn.rotary_emb
                rotary(torch.empty(0, dtype=self.dtype, device=self.device_of[index]), seq_len=longest)

    def _run(self, fn: Any, h: Tensor) -> Tensor:
        """Run one decoder layer's function on ``h``, recomputed in backward when checkpointing."""
        if self.checkpointing and torch.is_grad_enabled():
            return cast(Tensor, checkpoint(fn, h, use_reentrant=False))
        return cast(Tensor, fn(h))

    def _final_padded(self, sequences: Sequence[Sequence[int]], writes: Mapping[int, LayerWrites]) -> Tensor:
        ids = padded_ids(sequences, int(self.snapshot.config.pad_token_id))
        mask = padding_mask([len(s) for s in sequences])
        t = ids.shape[1]
        h = F.embedding(ids.to(self.device_of[0]), self.embedding)
        for index, layer in enumerate(self.layers):
            dev = self.device_of[index]
            h = h.to(dev)
            if index in writes:
                h = apply_writes(h, writes[index])
            additive = attention_mask(mask, self.dtype, dev)
            position = torch.arange(t, device=dev).unsqueeze(0)

            def run(x: Tensor, layer: Any = layer, additive: Tensor = additive, position: Tensor = position) -> Tensor:
                return cast(Tensor, layer(x, attention_mask=additive, position_ids=position, use_cache=False)[0])

            h = self._run(run, h)
        return cast(Tensor, self.norm(h.to(self.device_of[-1])))

    def _final_packed(self, sequences: Sequence[Sequence[int]], writes: Mapping[int, LayerWrites]) -> Tensor:
        lengths = [len(s) for s in sequences]
        pack = Packing(lengths)
        flat = torch.tensor([token for s in sequences for token in s], dtype=torch.int64)
        h = F.embedding(flat.to(self.device_of[0]), self.embedding).unsqueeze(0)
        mask = padding_mask(lengths) if self.attention == "eager" else None  # eager: scattered to the padded layout
        limit = recompile_limit(len(self.layers), len(set(self.devices)))
        attend = compiled(mla_packed, limit) if self._options["compile"] == "regional" else mla_packed
        for index, layer in enumerate(self.layers):
            dev = self.device_of[index]
            h = h.to(dev)
            if index in writes:  # sites move from (row, position) to the stream: one row, the packed index
                w = writes[index]
                at = pack.index(w.rows, w.positions)
                h = apply_writes(h, LayerWrites(torch.zeros_like(at), at, w.deltas, w.alphas))
            additive = None if mask is None else attention_mask(mask, self.dtype, dev)

            def run(x: Tensor, layer: nn.Module = layer, additive: Tensor | None = additive) -> Tensor:
                return packed_layer(layer, x, pack, self.attention, additive, attend=attend)

            h = self._run(run, h)
        stream = cast(Tensor, self.norm(h.to(self.device_of[-1])))[0]
        padded = stream.new_zeros(len(sequences) * pack.longest, stream.shape[-1])
        padded = padded.index_copy(0, pack.keep.to(stream.device), stream)
        return padded.view(len(sequences), pack.longest, -1)


# Static protocol conformance, checked by mypy: the registry hands backends out through a cast, which would hide a
# signature that drifted from the protocol.
_SCORING: type[ScoringBackend] = KimiEP
_TRAINING: type[TrainingBackend] = KimiEP
