"""A tiny random receiver built in-process (no download): byte-level BPE tokenizer, chat template, 4-layer Llama.

:func:`build_tiny_receiver` writes the model and
tokenizer into a directory and returns ``configs/receivers/tiny-llama.yaml`` pointed at it, so the renderer, the
``hf`` backend and the entry points load it like any model; :class:`FixedSource` serves fixed writes.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from latent_interconnect.artifacts.store import ArtifactRef
from latent_interconnect.config import load_config
from latent_interconnect.receivers.base import ReceiverSpec
from latent_interconnect.writes.connector import SlotWrites
from latent_interconnect.writes.layers import Entry

ROOT = Path(__file__).resolve().parents[2]
SPECIALS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<think>", "</think>"]
# Kimi-like: the generation prompt opens the assistant turn with an empty (thinking off) or open (on) think block, and
# a past assistant turn is written as the generation prompt (thinking off) + content + end of turn.
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'assistant' %}<|assistant|><think></think>{{ m['content'] }}<|end|>"
    "{% else %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% if thinking %}<think>{% else %}<think></think>{% endif %}{% endif %}"
)
CORPUS = [
    "Here is an unknown protein, given to you as an embedding: X. Is it an enzyme? Answer Yes or No.",
    "Describe what this protein is and what it does. It binds DNA and RNA in the nucleus.",
    "The protein X is a kinase; the protein X is a membrane transporter. Yes. No.",
    "Insulin is a hormone. Hemoglobin carries oxygen. α-helix and β-sheet, café.",
] * 20 + [" X"] * 200


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """A byte-level BPE tokenizer trained on a fixed corpus (deterministic), with the chat template above."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        chat_template=CHAT_TEMPLATE,
        eos_token="<|end|>",
        additional_special_tokens=[s for s in SPECIALS if s != "<|end|>"],
        clean_up_tokenization_spaces=False,
    )


def tiny_model(tokenizer: PreTrainedTokenizerFast, seed: int = 0) -> LlamaForCausalLM:
    """A random 4-layer, width-64 Llama in float32 (eager attention), matching ``tiny-llama.yaml``."""
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|end|>"),
        attn_implementation="eager",
        dtype="float32",
    )
    return LlamaForCausalLM(config).eval()


def build_tiny_receiver(directory: Path) -> ReceiverSpec:
    """Save the tiny model and tokenizer into ``directory``; return the tiny-llama receiver config pointing at it."""
    tok = tiny_tokenizer()
    tiny_model(tok).save_pretrained(directory)
    tok.save_pretrained(directory)
    spec = load_config(ROOT / "configs" / "receivers" / "tiny-llama.yaml", ReceiverSpec)
    return spec.model_copy(update={"model_id": str(directory)})


class FixedSource:
    """A write source serving fixed writes (rows selected by object id)."""

    def __init__(self, writes: SlotWrites) -> None:
        self.writes = writes
        self.calls: list[tuple[str, ...]] = []  # the object lists writes_for was asked for, in order

    @property
    def bank(self) -> ArtifactRef | None:
        return None

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(self.writes.deltas)

    def writes_for(self, object_ids: Sequence[str]) -> SlotWrites:
        self.calls.append(tuple(object_ids))
        rows = [self.writes.object_ids.index(o) for o in object_ids]
        return SlotWrites(
            tuple(object_ids), {e: d[rows] for e, d in self.writes.deltas.items()}, dict(self.writes.alphas)
        )
