"""Kimi execution the backends share: padded batches and their mask, the read-ahead of shards, the readout.

Right-padded batches and their attention mask, the background read-ahead of layer shards, and the readout from final
states.

The core rules (writes, targets, lengths, the planner, the one head) live in ``receivers.base`` and
``receivers.kernel``; this module holds the Kimi-specific execution that ``kimi-stream`` and ``kimi-ep`` would
otherwise each repeat.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from torch import Tensor

from latent_interconnect.receivers.base import Readout
from latent_interconnect.receivers.kernel import readout_logits, state_dtype
from latent_interconnect.receivers.kimi_k26.weights import KimiSnapshot
from latent_interconnect.receivers.render import RenderedPrompt

__all__ = ["attention_mask", "padded_ids", "padding_mask", "read_ahead", "read_out"]


def padding_mask(lengths: Sequence[int]) -> Tensor:
    """``(n, max(lengths))`` int64: 1 on each row's real tokens, 0 on its right padding."""
    mask = torch.zeros(len(lengths), max(lengths), dtype=torch.int64)
    for r, n in enumerate(lengths):
        mask[r, :n] = 1
    return mask


def padded_ids(sequences: Sequence[Sequence[int]], pad_id: int) -> Tensor:
    """``(n, longest)`` int64 token ids, each row right-padded with ``pad_id`` (masked, so its value never matters)."""
    ids = torch.full((len(sequences), max(len(s) for s in sequences)), pad_id, dtype=torch.int64)
    for r, s in enumerate(sequences):
        ids[r, : len(s)] = torch.tensor(s, dtype=torch.int64)
    return ids


def read_ahead(snapshot: KimiSnapshot, prefixes: Sequence[str], steps: int) -> Iterator[int]:
    """Yield ``0 .. steps - 1``; during step i the shards of ``prefixes[i + 1]`` are read into the page cache.

    ``prefixes[0]`` is read before step 0. ``prefixes`` may hold one more entry than ``steps`` (e.g. the final norm,
    read during the last layer).
    """
    with ThreadPoolExecutor(1) as pool:
        pending: Future[None] | None = pool.submit(snapshot.prefetch, prefixes[0]) if prefixes else None
        for index in range(steps):
            if pending is not None:
                pending.result()
            ahead = index + 1 < len(prefixes)
            pending = pool.submit(snapshot.prefetch, prefixes[index + 1]) if ahead else None
            yield index
        if pending is not None:
            pending.result()


def attention_mask(mask: Tensor, dtype: torch.dtype, device: torch.device) -> Tensor:
    """The additive 4-d mask of Kimi's eager attention: 0 where a query sees a key, the dtype's minimum elsewhere.

    A query sees the keys at or before it that are not padding (right padding): the values transformers'
    ``_prepare_4d_causal_attention_mask`` gives, built here because that helper fills in the default dtype and
    overflows for float64.

    Args:
        mask: ``(n, t)`` 1 on real tokens, 0 on padding.
        dtype: the attention's dtype.
        device: where the mask is used.
    """
    n, t = mask.shape
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    visible = causal.unsqueeze(0) & mask.bool().unsqueeze(1)
    additive = torch.zeros(n, 1, t, t, dtype=dtype)
    additive.masked_fill_(~visible.unsqueeze(1), torch.finfo(dtype).min)
    return additive.to(device)


def read_out(
    prompts: Sequence[RenderedPrompt],
    sequences: Sequence[tuple[int, ...]],
    rows: Sequence[Sequence[int]],
    finals: Sequence[Tensor],
    *,
    readout: Readout,
    head: Tensor,
) -> Tensor:
    """Apply the one head at the readout positions and return the readout in prompt order, on the host.

    Args:
        prompts: the call's prompts.
        sequences: what each prompt fed (a ``target_nll`` sequence is its prompt followed by its target, so the target
            tokens are the sequence's tail).
        rows: per micro-batch, the prompt index of each of its rows.
        finals: per micro-batch, the ``(rows, T, d)`` final (normed) states, right-padded.
        readout: what to read.
        head: the output projection, already in the state dtype (``kernel.state_dtype``): the caller promotes it
            once, so ``kernel.readout_logits`` does not cast the vocabulary-sized matrix on every call.

    Returns:
        ``(N, width)`` next-token log-probs, or ``(N,)`` unweighted mean target NLL.
    """
    dev = head.device
    out_dtype = state_dtype(finals[0].dtype if finals else head.dtype)
    if readout.kind == "next_token":
        out = torch.empty(len(prompts), len(readout.token_ids), dtype=out_dtype)
        ids = torch.tensor(readout.token_ids, device=dev)
        for batch, h in zip(rows, finals, strict=True):
            last = torch.tensor([len(prompts[i].input_ids) - 1 for i in batch], device=h.device)
            states = h[torch.arange(len(batch), device=h.device), last].to(dev)
            out[list(batch)] = readout_logits(states, head).log_softmax(-1)[:, ids].to("cpu", out_dtype)
        return out
    nll = torch.empty(len(prompts), dtype=out_dtype)
    for batch, h in zip(rows, finals, strict=True):
        for r, i in enumerate(batch):
            start, end = len(prompts[i].input_ids), len(sequences[i])
            logp = readout_logits(h[r, start - 1 : end - 1].to(dev), head).log_softmax(-1)
            tokens = torch.tensor(sequences[i][start:end], device=dev).unsqueeze(1)
            nll[i] = -logp.gather(1, tokens).mean().to("cpu", out_dtype)
    return nll
