"""Kimi K2.6 receiver plugin: efficient backends for one model, registered by name, never imported by core code.

Importing this package registers its backends (chosen per capability in ``configs/receivers/kimi-k2.6.yaml``):

- ``kimi-stream`` (``stream.py``; ``score``, ``capture``): one GPU, the model streamed one decoder layer at a time
  from local disk. Memory ≈ one layer; used for scoring suites, state capture and the layer scans.
- ``kimi-ep`` (``ep.py``; ``train``, and ``score`` for in-job evaluation): the resident 8-GPU path. INT4 routed
  experts are dequantized into expert-parallel blocks (``experts.py``), MLA attention is packed and variable-length
  (``packed.py``; no padding), with activation checkpointing (an execution option). The trainer,
  data, connector and optimizer are the generic ones (save-on-signal is the trainer's).
- ``kimi-vllm`` (``vllm.py``; ``generate``, and ``score`` for teacher-forced log-likelihood): a resident vLLM
  engine on one 8×H200 node, in the ``vllm`` environment; its worker extension (``vllm_worker.py``) applies the writes
  during prefill in vLLM's fused-residual form, reads write banks (``writes.sources``) by reference and computes every
  logit with the one head.

Shared by the backends: ``weights.py`` (opening the checked snapshot, page-cache warming, ``TransientLinear``, the
per-call INT4 decompression that serves inference and training, and building one decoder layer) and ``common.py``
(the padded attention mask and the readout from final states).

Everything Kimi-specific lives here: the INT4 / compressed-tensors pins, the MoE expert layout, MLA, the model shapes
(read from the receiver config and asserted at load) and the thinking-mode template flag. Site-specific weight
staging is not part of this package.

Testing: the semantics (writes, rendering, eager padded attention, loss weighting) are checked against
``hf`` on a shrunk, unquantized DeepSeek-V3-architecture model in float64 on CPU: scores, and for training the
same-input loss, the connector gradients and one optimizer update (``tests/unit/test_kimi_ep.py``,
through ``receivers.conformance``). Training through Kimi's remote MoE needs one
preparation: its ``moe_infer`` is ``@torch.no_grad``, and ``weights.enable_moe_gradients`` unwraps it; without it
``hf`` returns the right loss and silently drops the routed experts from the gradient, which the finite-difference
test catches and the runtime check at ``open_training`` does not. ``kimi-ep`` replaces that code and needs none.
The variable-length attention, the grouped expert GEMM and the vLLM path have no CPU float64 kernels, so they are
checked by GPU parity jobs on the real model (``kimi-ep`` against its reference configuration, ``kimi-vllm``
against ``kimi-stream``).

Why separate modules behind generic protocols: the speed comes from model-specific kernels and
memory layouts, while the semantics are generic and must be identical across backends. The generic path is the
reference every fast path is tested against.
"""

from latent_interconnect.receivers.kimi_k26 import ep as _ep  # noqa: F401  (registers "kimi-ep")
from latent_interconnect.receivers.kimi_k26 import stream as _stream  # noqa: F401  (registers "kimi-stream")
from latent_interconnect.receivers.kimi_k26 import vllm as _vllm  # noqa: F401  (registers "kimi-vllm")
