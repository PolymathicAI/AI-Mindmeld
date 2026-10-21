"""Kimi K2.6 weights on disk: the snapshot layout, page-cache warming, and building one decoder layer at a time.

A Kimi K2.6 snapshot is the multimodal ``KimiK25ForConditionalGeneration`` checkpoint. Its language model is a
DeepSeek-V3 decoder stored under the key prefix :data:`LANGUAGE_MODEL_PREFIX`, with its config at
``config.text_config`` and its code in the snapshot's own ``modeling_deepseek`` module (``trust_remote_code``; the
revision is pinned in the receiver config). The routed experts are INT4 (compressed-tensors ``pack-quantized``,
group 32); attention, the shared experts, the dense layer-0 MLP and the head are bf16. A snapshot whose text config
has no ``quantization_config`` (the shrunk float64 test model) is loaded densely through the same path.

Pitfalls this module handles (measured; from the original runbook):

- ``compressed-tensors`` 0.12.2 (``transformers`` < 5; newer releases fail on packed INT4) replaces a compressed
  weight by its decompressed bf16 copy on first use, which would keep a whole layer's experts dense in memory.
  :func:`install_per_call_decompression` makes every call decompress afresh and keep nothing;
- a single reader on ceph reaches about 85 MB/s and eight parallel readers about 750 MB/s, so
  :meth:`KimiSnapshot.prefetch` warms the page cache with :data:`PREFETCH_STREAMS` readers.

The caller frees each layer and runs ``gc.collect()`` after it (``stream.py``): the remote layers hold reference
cycles that otherwise leak about 9.5 GB per layer.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.linear.compressed_linear import CompressedLinear
from compressed_tensors.quantization import QuantizationStatus, apply_quantization_config
from huggingface_hub import constants
from safetensors import safe_open
from torch import Tensor, nn
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from latent_interconnect.receivers.base import PREPARATIONS, ReceiverSpec

__all__ = [
    "EMBED_PREFIX",
    "HEAD_PREFIX",
    "INDEX_FILE",
    "LANGUAGE_MODEL_PREFIX",
    "NORM_PREFIX",
    "PREFETCH_STREAMS",
    "SERVING_FILES",
    "KimiSnapshot",
    "SnapshotMismatchError",
    "TransientLinear",
    "check_weights",
    "enable_moe_gradients",
    "install_per_call_decompression",
    "model_files",
    "moe_gradients",
    "open_snapshot",
    "require_moe_gradients",
    "snapshot_path",
]

LANGUAGE_MODEL_PREFIX = "language_model."
EMBED_PREFIX = f"{LANGUAGE_MODEL_PREFIX}model.embed_tokens."
NORM_PREFIX = f"{LANGUAGE_MODEL_PREFIX}model.norm."
HEAD_PREFIX = f"{LANGUAGE_MODEL_PREFIX}lm_head."
INDEX_FILE = "model.safetensors.index.json"
SERVING_FILES = ("generation_config.json", "tokenizer_config.json", "tiktoken.model", "chat_template.jinja")
"""Read from the weights directory by vLLM (tokenizer, chat template, end-of-sequence ids) and by the tokenizer."""
PREFETCH_STREAMS = 8
"""Parallel readers per shard file: one ceph stream reads ~85 MB/s, eight ~750 MB/s (the original runbook)."""
_SHOWN = 5  # differing files named in a mismatch message (the count is always given)
_PREFETCH_CHUNK = 64 << 20  # bytes per read call: large sequential reads, as in the measured original prefetch


class SnapshotMismatchError(ValueError):
    """A weights directory differs from the pinned snapshot it stands in for."""


def model_files(path: Path) -> dict[str, int]:
    """Return name → size (bytes) of the files a Kimi snapshot is read from.

    These are ``config.json``, the remote code (``*.py``), :data:`SERVING_FILES`, the safetensors index and every shard
    the index names (size -1: absent). Documentation and evaluation files are never read, and a copy may leave them out.

    Raises:
        FileNotFoundError: if the directory has no safetensors index.
    """
    index = path / INDEX_FILE
    if not index.is_file():
        raise FileNotFoundError(f"no {INDEX_FILE} in the Kimi snapshot {path}")
    shards = set(json.loads(index.read_text())["weight_map"].values())
    names = {"config.json", INDEX_FILE, *SERVING_FILES, *shards} | {p.name for p in path.glob("*.py")}
    return {name: (path / name).stat().st_size if (path / name).is_file() else -1 for name in sorted(names)}


def check_weights(weights: Path, pinned: Path) -> None:
    """Refuse a weights directory whose model files differ from the pinned snapshot's in name or size.

    Args:
        weights: the directory the backend will read (e.g. a node-local copy).
        pinned: the snapshot at the receiver config's revision (``snapshots/<revision>`` in the Hugging Face cache).

    Raises:
        SnapshotMismatchError: with the number of differing files (missing, extra or of another size) and the first
            few of them.
    """
    have, want = model_files(weights), model_files(pinned)

    def size(files: dict[str, int], name: str) -> str:
        return str(files[name]) if files.get(name, -1) >= 0 else "absent"

    bad = sorted(
        f"{name}: {size(have, name)} bytes in {weights}, {size(want, name)} in the pinned snapshot"
        for name in set(have) | set(want)
        if have.get(name, -1) != want.get(name, -1) or have.get(name, -1) < 0
    )
    if bad:
        shown = "; ".join(bad[:_SHOWN]) + (f"; and {len(bad) - _SHOWN} more" if len(bad) > _SHOWN else "")
        raise SnapshotMismatchError(f"{weights} differs from {pinned} ({len(bad)} differing): {shown}")


class TransientLinear(torch.autograd.Function):
    """``y = x Wᵀ + b`` with a compressed weight decompressed on the fly and never kept, in forward and in backward.

    The weight and the bias are frozen, so backward returns only the gradient into ``x``, decompressing the weight again
    rather than letting autograd save it (a saved dense copy of every routed expert would cost about 19 GB per MoE
    layer at batch 1, measured with the original trainer). Inference and training use the same function (one patch).
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, module: Any) -> Tensor:
        """Decompress the module's weight and apply it (and the module's bias, if any)."""
        ctx.module = module
        bias = getattr(module, "bias", None)  # a Linear has the attribute (maybe None); a bare stand-in may not
        bias = None if bias is None else bias.to(x.dtype)
        return F.linear(x, module.compressor.decompress_module(module).to(x.dtype), bias)

    @staticmethod
    def backward(ctx: Any, *grads: Tensor) -> tuple[Tensor, None]:
        """Return the gradient into ``x`` (the weight is decompressed again); none into the frozen module."""
        (g,) = grads
        return g @ ctx.module.compressor.decompress_module(ctx.module).to(g.dtype), None


def install_per_call_decompression() -> None:
    """Make ``CompressedLinear`` run :class:`TransientLinear`: decompress on every call, keep no dense copy.

    Idempotent.

    The library's own forward decompresses once and stores the bf16 weight on the module, which would keep a dense
    copy of all 384 experts per layer. compressed-tensors wraps each module's forward when the layer is built and
    keeps the function current at that moment, so the patch must be in place before any layer is built;
    :class:`KimiSnapshot` installs it when it opens a quantized snapshot. It is process-global and serves both the
    streaming scorer and the trainer (gradients into the input flow through it).
    """

    def forward(self: Any, input: Tensor) -> Tensor:
        if self.quantization_status != QuantizationStatus.COMPRESSED:
            raise RuntimeError(f"expected a compressed module, found status {self.quantization_status!r}")
        out: Tensor = cast(Any, TransientLinear).apply(input, self)
        return out

    CompressedLinear.forward = forward


def enable_moe_gradients(path: Path) -> None:
    """Let gradients flow through the snapshot's own routed experts (process-global, idempotent).

    Kimi's remote ``DeepseekV3MoE.moe_infer`` is decorated ``@torch.no_grad()`` (its MoE has no training branch), so
    a model run through it in eval mode silently drops the routed experts' contribution to every gradient. This
    unwraps the decorator, keeping routing identical. ``kimi-ep`` does not need it (its expert-parallel ``moe_infer``
    replaces the remote one per module); any backend that trains through the remote module does (the generic ``hf``
    backend on Kimi, the float64 reference of the CPU parity tests).
    """
    moe: Any = get_class_from_dynamic_module("modeling_deepseek.DeepseekV3MoE", str(path))
    wrapped = getattr(moe.moe_infer, "__wrapped__", None)
    if wrapped is not None:
        moe.moe_infer = wrapped


@PREPARATIONS.register("kimi-moe-gradients")
def require_moe_gradients(model: nn.Module) -> None:
    """Refuse a Kimi model that still runs its remote ``moe_infer`` under ``@torch.no_grad`` (the declared preparation).

    Kimi K2.6's receiver config declares ``training_preparation: kimi-moe-gradients``; ``hf`` calls this before every
    loss. The model is prepared when no MoE module's class still wraps ``moe_infer`` (``enable_moe_gradients``).

    Raises:
        ValueError: if a MoE class still wraps it, or the model has no MoE module (this check would not apply).
    """
    classes: set[Any] = {type(m) for m in model.modules() if hasattr(type(m), "moe_infer")}  # remote classes
    if not classes:
        raise ValueError("the kimi-moe-gradients preparation found no MoE module (moe_infer) in the model")
    blocked = sorted(c.__qualname__ for c in classes if hasattr(c.moe_infer, "__wrapped__"))
    if blocked:
        raise ValueError(
            f"{blocked} run moe_infer under @torch.no_grad, so training through them would silently drop the routed"
            " experts from every gradient: apply latent_interconnect.receivers.kimi_k26.weights.enable_moe_gradients"
            " first (the receiver's declared training_preparation, kimi-moe-gradients)"
        )


@contextmanager
def moe_gradients(path: Path) -> Iterator[None]:
    """:func:`enable_moe_gradients` for a scope: the remote ``moe_infer`` is restored on exit, as it was on entry.

    For tests: each one that needs gradients through the remote experts prepares them itself, so no test passes
    because an earlier one left the process prepared.
    """
    moe: Any = get_class_from_dynamic_module("modeling_deepseek.DeepseekV3MoE", str(path))
    original = moe.__dict__["moe_infer"]
    enable_moe_gradients(path)
    try:
        yield
    finally:
        moe.moe_infer = original


def snapshot_path(spec: ReceiverSpec, weights: Path | None) -> Path:
    """Return the directory to read the receiver's weights from: the pinned snapshot, or a checked copy of it.

    The pinned snapshot is ``snapshots/<spec.revision>`` of ``spec.model_id`` in the Hugging Face cache (never
    downloaded), resolved from the cache layout and required to hold every file of :func:`model_files`. Files the
    backends never read (documentation, ``.eval_results/``) may be absent: huggingface_hub's own completeness check
    (``snapshot_download``, ``IncompleteSnapshotError`` from 1.x) refuses such a snapshot, so it is not used, and every
    environment resolves the snapshot alike. A given ``weights`` directory (e.g. a node-local copy) must hold the same
    model files, by name and size (:func:`check_weights`). Every Kimi backend finds its weights here (``kimi-vllm``
    hands the directory to vLLM).

    Raises:
        FileNotFoundError: if the pinned snapshot is not in the cache, has no safetensors index or lacks a model file.
        SnapshotMismatchError: if ``weights`` differs from the pinned snapshot.
    """
    pinned = Path(constants.HF_HUB_CACHE) / f"models--{spec.model_id.replace('/', '--')}" / "snapshots" / spec.revision
    if not pinned.is_dir():
        raise FileNotFoundError(
            f"{spec.model_id} at revision {spec.revision} is not in the Hugging Face cache ({pinned})"
        )
    missing = [name for name, size in model_files(pinned).items() if size < 0]
    if missing:
        shown = ", ".join(missing[:_SHOWN]) + (f" and {len(missing) - _SHOWN} more" if len(missing) > _SHOWN else "")
        raise FileNotFoundError(f"the pinned snapshot {pinned} lacks {len(missing)} file(s) Kimi reads: {shown}")
    if weights is None:
        return pinned
    check_weights(weights, pinned)
    return weights


def open_snapshot(spec: ReceiverSpec, weights: Path | None) -> KimiSnapshot:
    """Open the receiver's weights from :func:`snapshot_path`.

    Raises:
        FileNotFoundError: if a directory has no safetensors index.
        SnapshotMismatchError: if ``weights`` differs from the pinned snapshot.
        ValueError: if the snapshot's shape differs from the receiver config's.
    """
    return KimiSnapshot(snapshot_path(spec, weights), depth=spec.depth, d_model=spec.d_model)


class KimiSnapshot:
    """A local Kimi K2.6 snapshot directory, read one module at a time.

    Args:
        path: the snapshot directory (``config.json``, the remote code, ``model.safetensors.index.json`` and the
            shards).
        depth: the receiver config's decoder-layer count, asserted against the snapshot.
        d_model: the receiver config's hidden width, asserted against the snapshot.

    Attributes:
        path: the snapshot directory.
        config: the language model's config (``text_config``), with eager attention set.
        depth: the decoder-layer count (checked equal to the receiver config's): the one source of the layer range
            that loading and planning a load iterate.
        quantization: the compressed-tensors quantization config, or None for a dense snapshot.

    Raises:
        FileNotFoundError: if the directory has no safetensors index.
        ValueError: if the snapshot's depth or width differs from the receiver config's.
    """

    def __init__(self, path: Path, *, depth: int, d_model: int) -> None:

        index = path / INDEX_FILE
        if not index.is_file():
            raise FileNotFoundError(f"no {INDEX_FILE} in the Kimi snapshot {path}")
        self.path = path
        self._weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
        config = AutoConfig.from_pretrained(path, trust_remote_code=True).text_config
        if (config.num_hidden_layers, config.hidden_size) != (depth, d_model):
            raise ValueError(
                f"snapshot {path} has {config.num_hidden_layers} layers of width {config.hidden_size}; "
                f"the receiver config says {depth} of width {d_model}"
            )
        # The streamed pass calls one decoder layer at a time with a 4-d padding mask: eager attention is the
        # implementation that takes it, and the one the original step-0 scores ran.
        config._attn_implementation = "eager"
        self.config: Any = config
        self.depth = depth
        self._layer_class: Any = get_class_from_dynamic_module("modeling_deepseek.DeepseekV3DecoderLayer", str(path))
        self._norm_class: Any = get_class_from_dynamic_module("modeling_deepseek.DeepseekV3RMSNorm", str(path))
        self.quantization: Any = None
        compression = getattr(config, "quantization_config", None)
        if compression is not None:
            self.quantization = ModelCompressor.from_compression_config(compression).quantization_config
            install_per_call_decompression()

    @staticmethod
    def layer_prefix(index: int) -> str:
        """Return the key prefix of decoder layer ``index``."""
        return f"{LANGUAGE_MODEL_PREFIX}model.layers.{index}."

    def files(self, prefix: str) -> list[str]:
        """Return the shard files holding any key under ``prefix``, sorted."""
        return sorted({f for k, f in self._weight_map.items() if k.startswith(prefix)})

    def prefetch(self, prefix: str) -> None:
        """Read the shard files under ``prefix`` into the page cache with :data:`PREFETCH_STREAMS` parallel readers."""
        for name in self.files(prefix):
            path = self.path / name
            size = path.stat().st_size
            step = -(-size // PREFETCH_STREAMS)
            ranges = [(i * step, min(size, (i + 1) * step)) for i in range(PREFETCH_STREAMS)]
            with ThreadPoolExecutor(PREFETCH_STREAMS) as pool:
                for done in [pool.submit(_read_range, path, start, end) for start, end in ranges]:
                    done.result()

    def tensors(self, prefix: str, device: torch.device) -> dict[str, Tensor]:
        """Return every tensor under ``prefix``, keyed by the rest of its name, on ``device``.

        Raises:
            KeyError: if no key starts with ``prefix``.
        """
        by_file: dict[str, list[str]] = {}
        for key, name in self._weight_map.items():
            if key.startswith(prefix):
                by_file.setdefault(name, []).append(key)
        if not by_file:
            raise KeyError(f"no tensor under {prefix!r} in {self.path}")
        out: dict[str, Tensor] = {}
        for name, keys in sorted(by_file.items()):
            with safe_open(str(self.path / name), framework="pt", device=str(device)) as handle:
                for key in keys:
                    out[key[len(prefix) :]] = handle.get_tensor(key)
        return out

    def matrix(self, prefix: str, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return the ``weight`` under ``prefix`` (the embedding table or the head) as ``dtype`` on ``device``."""
        return self.tensors(prefix, device)["weight"].to(dtype)

    def skeleton(self, index: int, dtype: torch.dtype, device: torch.device) -> nn.Module:
        """Construct decoder layer ``index`` on ``device`` without its weights: the class, ``dtype``, the compression.

        The one construction :meth:`build_layer` loads into, and the one a memory plan measures on the meta device
        (``ep.plan_memory``): buffers that are not in the checkpoint (the rotary tables) and the compressed tensors'
        dtypes are those of the built layer. The parameters are uninitialized.
        """
        # imported here: it moved in transformers 5, the vllm environment's, which imports this plugin but never
        # builds a layer (kimi-vllm runs vLLM's own model); kimi-stream and kimi-ep run in the project venv
        from transformers.modeling_utils import no_init_weights  # noqa: PLC0415

        with no_init_weights(), device:
            layer: nn.Module = self._layer_class(self.config, index).to(dtype)
        if self.quantization is not None:
            apply_quantization_config(layer, self.quantization, run_compressed=True)
        return layer

    def build_layer(self, index: int, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """Build decoder layer ``index`` with its weights loaded, in eval mode, frozen.

        The skeleton is constructed on the host, then moved to ``device`` and loaded there (``ep.plan_memory``'s build
        phase counts the moved skeleton beside the loaded tensors).

        Raises:
            KeyError: if the snapshot lacks a parameter of the layer or holds one the layer does not have.
        """
        layer = self.skeleton(index, dtype, torch.device("cpu")).to(device).eval()
        missing, unexpected = layer.load_state_dict(self.tensors(self.layer_prefix(index), device), strict=False)
        if missing or unexpected:
            raise KeyError(f"layer {index}: missing {missing[:5]}, unexpected {unexpected[:5]} in {self.path}")
        layer.requires_grad_(False)
        built: nn.Module = layer
        return built

    def build_norm(self, device: torch.device, dtype: torch.dtype) -> nn.Module:
        """Build the final RMSNorm with its weight loaded."""
        norm: nn.Module = self._norm_class(self.config.hidden_size, eps=self.config.rms_norm_eps).to(device, dtype)
        norm.load_state_dict(self.tensors(NORM_PREFIX, device))
        norm.requires_grad_(False)
        return norm


def _read_range(path: Path, start: int, end: int) -> None:
    with path.open("rb", buffering=0) as handle:
        pos = start
        while pos < end:
            n = len(os.pread(handle.fileno(), min(_PREFETCH_CHUNK, end - pos), pos))
            if n == 0:
                return
            pos += n
