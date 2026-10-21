"""A shrunk Kimi K2.6 for the CPU parity tests: the snapshot's own remote-code DeepSeek-V3 decoder (MLA attention with
YaRN rotary, a dense layer 0, routed-MoE layers with sigmoid noaux_tc gating), shrunk dimensions, random float64
weights, written in the real snapshot's layout (``language_model.`` keys, ``text_config``, sharded safetensors with an
index) into a Hugging Face hub cache directory (``models--lip-test--<name>/snapshots/<revision>``), so ``kimi-stream``
finds and checks it exactly as it does the real model. The tests point ``huggingface_hub.constants.HF_HUB_CACHE`` at
that directory.

With ``quantized=True`` the routed experts are stored INT4 as in the real snapshot (compressed-tensors
``pack-quantized``, group 32, the real quantization config), and the returned model holds their dequantized weights;
:func:`write_dense` writes that model as a dense twin snapshot.

The remote code, config and serving files (no weights) come from a Hugging Face hub cache directory named by
:data:`HUB_VARIABLE`, holding the pinned revision of ``configs/receivers/kimi-k2.6.yaml`` (``CONTRIBUTING.md``); without
it the tests that need this module skip.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from latent_interconnect.config import load_config
from latent_interconnect.receivers.base import ReceiverSpec
from latent_interconnect.receivers.kimi_k26.weights import SERVING_FILES

REPO = Path(__file__).resolve().parents[1]
TINY_LAYERS = 3
"""Decoder layers of the shared tiny Kimi (``tests/conftest.py``): a dense layer 0 and two MoE layers."""
VOCAB = 300
SHRINK = {
    "hidden_size": 64,
    "intermediate_size": 96,
    "moe_intermediate_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": 32,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "n_routed_experts": 16,
    "num_experts_per_tok": 4,
    "n_shared_experts": 1,
    "vocab_size": VOCAB,
    "pad_token_id": VOCAB - 1,
    "bos_token_id": VOCAB - 3,
    "eos_token_id": VOCAB - 2,
    "max_position_embeddings": 256,  # only the length of the cached rotary table; YaRN's scaling is unchanged
}
HUB_VARIABLE = "AIMM_KIMI_HUB"
"""The environment variable naming the hub cache that holds the pinned Kimi snapshot's non-weight files."""
REAL_HUB = os.environ.get(HUB_VARIABLE)  # read before any test points the hub elsewhere
SKIP_REASON = f"needs the Kimi K2.6 non-weight snapshot in the hub cache {HUB_VARIABLE} (CONTRIBUTING.md)"
"""Not ``tests.snapshots.REASON``: ``--require-kimi-ep``, not ``--require-snapshots``, makes these skips failures."""
EXPERT = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj$")


def kimi_spec() -> ReceiverSpec:
    return load_config(REPO / "configs" / "receivers" / "kimi-k2.6.yaml", ReceiverSpec)


def kimi_source() -> Path:
    """The cached Kimi K2.6 snapshot at the pinned revision (config and remote code are all that is read)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    spec = kimi_spec()
    if not REAL_HUB:
        pytest.skip(f"{SKIP_REASON}: {HUB_VARIABLE} is not set")
    try:
        return Path(snapshot_download(spec.model_id, revision=spec.revision, local_files_only=True, cache_dir=REAL_HUB))
    except LocalEntryNotFoundError:
        pytest.skip(f"{SKIP_REASON}: not in {HUB_VARIABLE}={REAL_HUB}")


def _write_config(dest: Path, layers: int, quantized: bool) -> dict[str, Any]:
    """Copy the remote code and the serving files and write the shrunk config; return the real quantization config."""
    source = kimi_source()
    dest.mkdir(parents=True, exist_ok=True)
    for code in [*sorted(source.glob("*.py")), *(source / name for name in SERVING_FILES)]:
        shutil.copyfile(code, dest / code.name)
    config = json.loads((source / "config.json").read_text())
    text = config["text_config"]
    compression: dict[str, Any] = text.pop("quantization_config")
    text.update(SHRINK, num_hidden_layers=layers)
    if quantized:
        text["quantization_config"] = compression
    config["pad_token_id"] = text["pad_token_id"]
    (dest / "config.json").write_text(json.dumps(config, indent=1))
    return compression


def _write_weights(dest: Path, state: dict[str, torch.Tensor], layers: int) -> None:
    """Save ``state`` under ``language_model.`` keys over two shard files, with the index."""
    shards: dict[str, dict[str, torch.Tensor]] = {"model-1.safetensors": {}, "model-2.safetensors": {}}
    for key, value in state.items():
        late = any(f"layers.{i}." in key for i in range(layers // 2 + 1, layers)) or "lm_head" in key
        shards["model-2.safetensors" if late else "model-1.safetensors"][f"language_model.{key}"] = value.contiguous()
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, str(dest / name))
        weight_map.update(dict.fromkeys(tensors, name))
    (dest / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def snapshot_dir(hub: Path, name: str, layers: int, seed: int, quantized: bool) -> tuple[ReceiverSpec, Path]:
    """The receiver spec of a fixture and its ``snapshots/<revision>`` directory in ``hub``."""
    revision = hashlib.sha1(f"{name}:{seed}:{quantized}".encode()).hexdigest()  # a commit-shaped revision
    model_id = f"lip-test/{name}"
    update = {"model_id": model_id, "revision": revision, "depth": layers, "d_model": SHRINK["hidden_size"]}
    dest = hub / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    return kimi_spec().model_copy(update=update), dest


def _dense_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    # the rotary tables are non-persistent buffers, so the state dict holds exactly what the real snapshot stores
    return dict(model.state_dict())


def _randomize(model: torch.nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if "norm" in name:
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g, dtype=torch.float64))
            elif "e_score_correction_bias" in name:
                p.copy_(0.02 * torch.randn(p.shape, generator=g, dtype=torch.float64))
            else:
                p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64) / p.shape[-1] ** 0.5)


def _quantize_experts(model: torch.nn.Module, compression: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Compress every routed expert's weight; return the compressed tensors by key.

    The model's expert weights are replaced by their dequantized values, which is what a float64 stream computes from
    the compressed tensors (q · scale with bf16 scales is exact in float64).
    """
    from compressed_tensors.compressors import BaseCompressor, ModelCompressor

    config = ModelCompressor.from_compression_config(compression).quantization_config
    args = config.config_groups["group_0"].weights
    compressor = BaseCompressor.load_from_registry(config.format)
    out = {}
    for name, module in sorted(model.named_modules()):
        if not EXPERT.search(name):
            continue
        w = module.weight.detach()
        scale = (w.abs().reshape(w.shape[0], -1, args.group_size).amax(-1) / 7).to(torch.bfloat16)
        packed = compressor.compress_weight(w.to(torch.float32), scale.to(torch.float32), args)
        dense = compressor.decompress_weight({**packed, "weight_scale": scale.to(torch.float64)}, args)
        with torch.no_grad():
            module.weight.copy_(dense)
        out.update({f"{name}.{k}": v for k, v in {**packed, "weight_scale": scale}.items()})
    return out


def build_tiny_kimi(
    hub: Path, name: str, *, layers: int = 3, seed: int = 0, quantized: bool = False
) -> tuple[ReceiverSpec, torch.nn.Module]:
    """Write a shrunk float64 Kimi snapshot into the hub cache ``hub``; return its receiver spec and the model.

    The model is the remote ``DeepseekV3ForCausalLM`` with the snapshot's weights (dequantized when ``quantized``):
    the reference the backends are compared with.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    spec, dest = snapshot_dir(hub, name, layers, seed, quantized)
    compression = _write_config(dest, layers, quantized=False)
    cfg = AutoConfig.from_pretrained(dest, trust_remote_code=True).text_config
    cfg._attn_implementation = "eager"
    model = get_class_from_dynamic_module("modeling_deepseek.DeepseekV3ForCausalLM", str(dest))(cfg)
    model = model.to(torch.float64).eval()
    model.requires_grad_(False)
    _randomize(model, seed)
    state = _dense_state(model)
    if quantized:
        compressed = _quantize_experts(model, compression)
        state = {k: v for k, v in state.items() if not EXPERT.search(k.removesuffix(".weight"))} | compressed
        _write_config(dest, layers, quantized=True)
    _write_weights(dest, state, layers)
    return spec, model


def write_dense(model: torch.nn.Module, hub: Path, name: str, *, layers: int, seed: int) -> ReceiverSpec:
    """Write ``model`` as a dense snapshot into ``hub`` (the dense twin of a quantized fixture)."""
    spec, dest = snapshot_dir(hub, name, layers, seed, quantized=False)
    _write_config(dest, layers, quantized=False)
    _write_weights(dest, _dense_state(model), layers)
    return spec
