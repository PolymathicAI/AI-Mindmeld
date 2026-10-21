"""A tiny random Gemma 3 receiver built in-process (no download: the real ``google/gemma-3-*-it`` weights are gated).

:func:`build_tiny_gemma` builds the model through ``AutoModelForCausalLM.from_config`` on a ``Gemma3Config``, which at
transformers 4.57.6 (the project venv's version, where these tests were measured) gives
``Gemma3ForConditionalGeneration``: the multimodal class the real 4B and 12B instruct models load as (a SigLIP tower,
a projector and the text decoder ``model.language_model``, a ``Gemma3TextModel``), not the text-only
``Gemma3ForCausalLM``. Its text decoder keeps the features the hooks meet in the real models: 5 sliding-window layers
to 1 global, sandwich RMSNorms with a (1 + w) scale computed in float32, QK-norm, the √d embedding scale, a tied head
and no logit softcapping. The tokenizer is a character tokenizer whose special tokens sit at Gemma's ids (``<pad>`` 0,
``<eos>`` 1, ``<bos>`` 2, ``<start_of_turn>`` 105, ``<end_of_turn>`` 106), with Gemma 3's own chat template; the
generation config is the instruct models' (end ids 1 and 106). Both are saved into a directory with the model, and the
returned receiver config points at it, so ``hf`` loads it like any model.
"""

from __future__ import annotations

import string
from pathlib import Path

import torch
from tokenizers import Regex, decoders, models, pre_tokenizers
from tokenizers import Tokenizer as RawTokenizer
from transformers import AutoModelForCausalLM, Gemma3Config, GenerationConfig, PreTrainedTokenizerFast

from latent_interconnect.receivers.base import ReceiverSpec

DEPTH = 7
"""Decoder layers: 0-4 and 6 sliding-window, 5 global (Gemma 3's pattern of 6)."""
D = 64
END_OF_TURN = 106
"""``<end_of_turn>``, as in the real Gemma 3 vocabulary: an end id of the instruct models' generation config."""
EOS = 1
SLIDING_WINDOW = 8
"""Shorter than every rendered test prompt, so the local layers' window is exercised, and as long as the runtime
gradient probe (``receivers.gradients``: 8 positions), which the real models' window of 1,024 holds whole."""

# Gemma 3's chat template, verbatim (the 4B-it and 12B-it tokenizer configs, the same text; gemma/FEASIBILITY.md):
# the system message is folded into the first user turn, the assistant role is "model".
GEMMA3_CHAT_TEMPLATE = """\
{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '

' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '

' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '
' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>
' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model
'}}
{%- endif -%}
"""

_SPECIAL = {0: "<pad>", 1: "<eos>", 2: "<bos>", 3: "<unk>", 105: "<start_of_turn>", END_OF_TURN: "<end_of_turn>"}
_IMAGE = ("<start_of_image>", "<end_of_image>", "<image_soft_token>")


def tiny_gemma_tokenizer() -> PreTrainedTokenizerFast:
    """A character tokenizer at Gemma's special ids (``" c"`` one token, so the placeholder ``" X"`` is one id)."""
    chars = [c for c in string.printable if not c.isspace()]
    regular = iter([" ", "\n", *chars, *(" " + c for c in chars)])
    vocab: list[str] = []
    while len(vocab) <= max(_SPECIAL):
        vocab.append(_SPECIAL.get(len(vocab)) or next(regular))
    vocab += [*regular, *_IMAGE]
    raw = RawTokenizer(models.WordLevel({t: i for i, t in enumerate(vocab)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Split(Regex(r" ?[^\s]|\s"), behavior="isolated")
    raw.decoder = decoders.Fuse()
    raw.add_special_tokens([*_SPECIAL.values(), *_IMAGE])
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=["<start_of_turn>", "<end_of_turn>", *_IMAGE],
    )
    tok.chat_template = GEMMA3_CHAT_TEMPLATE
    return tok


def tiny_gemma_config(tokenizer: PreTrainedTokenizerFast) -> Gemma3Config:
    """The tiny Gemma 3 config: a 7-layer, width-64 text decoder and a 2-layer vision tower."""
    image = [tokenizer.convert_tokens_to_ids(t) for t in _IMAGE]
    return Gemma3Config(
        text_config={
            "vocab_size": len(tokenizer),
            "hidden_size": D,
            "intermediate_size": 128,
            "num_hidden_layers": DEPTH,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "query_pre_attn_scalar": 32,
            "sliding_window": SLIDING_WINDOW,
            "max_position_embeddings": 512,
            "pad_token_id": 0,
            "eos_token_id": EOS,
            "bos_token_id": 2,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 28,
            "patch_size": 14,
        },
        mm_tokens_per_image=4,
        boi_token_index=image[0],
        eoi_token_index=image[1],
        image_token_index=image[2],
    )


def build_tiny_gemma(directory: Path, seed: int = 0) -> ReceiverSpec:
    """Build the tiny Gemma 3 through ``AutoModelForCausalLM`` (float32), save it and its tokenizer into ``directory``.

    Returns:
        A receiver config pointing at ``directory``: every capability ``hf``, decoder layers at
        ``model.language_model.layers``.

    Raises:
        TypeError: if ``AutoModelForCausalLM`` does not map the config to ``Gemma3ForConditionalGeneration`` (another
            transformers version: the fixture would no longer be the class the real models load as).
    """
    tok = tiny_gemma_tokenizer()
    torch.manual_seed(seed)
    config = tiny_gemma_config(tok)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    if type(model).__name__ != "Gemma3ForConditionalGeneration":
        raise TypeError(f"AutoModelForCausalLM built {type(model).__name__}, not Gemma3ForConditionalGeneration")
    with torch.no_grad():
        # the random init zeroes the padding row (<pad>, id 0), which no bias ever moves, so a <pad> state would stay
        # zero through every layer; the trained rows are not zero (norm 1.02 in the 4B, 0.96 in the 12B)
        model.get_input_embeddings().weight[0].normal_(0.0, config.text_config.initializer_range)
    model.generation_config = GenerationConfig(bos_token_id=2, eos_token_id=[EOS, END_OF_TURN], pad_token_id=0)
    model.save_pretrained(directory)
    tok.save_pretrained(directory)
    return ReceiverSpec(
        name="tiny-gemma",
        model_id=str(directory),
        revision=f"random-seed-{seed}",
        depth=DEPTH,
        d_model=D,
        placeholder=" X",
        layers_path="model.language_model.layers",
        backends={"score": "hf", "train": "hf", "generate": "hf", "capture": "hf"},
    )
