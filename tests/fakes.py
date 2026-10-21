"""Shared fakes for the unit tests: small, deterministic, and implementing the real interfaces.

Import from here instead of writing a test-local fake. The slice itself runs on the real domains and Kimi; these are
for unit tests.

- :func:`tiny_tokenizer`: a real Hugging Face fast tokenizer built in memory (one token per character, a leading space
  joined to the next character, so the slot placeholder ``" X"`` is one token), with a chat template;
- :func:`build_tiny_llama`: saves a random 4-layer Llama and that tokenizer into a directory and returns
  ``configs/receivers/tiny-llama.yaml`` pointing at it, so backends, the renderer and ``python -m`` entry points load
  it like any model;
- :class:`ToyExtractor` (extractor ``fake-toy``) and :func:`commit_toy_embeddings`: deterministic embeddings in the
  real embedding-store format;
- :class:`ToyDomain` (domain ``fake-toy``) and :func:`commit_toy_domain`: twelve protein-like objects in a real
  ``domain_data`` artifact;
- :class:`FakeScoringBackend` (receiver backend ``fake-score``): scores that depend on the writes through the one
  resolver, so a suite can tell a connector's writes from none;
- :class:`FakeTrainingBackend` (receiver backend ``fake-train``): a differentiable loss of the writes, with the
  ``receivers.base.LossTerms`` contract (``token_nll`` included), and the fake's scores;
- :class:`FakeGenerationBackend` (receiver backend ``fake-generate``): continuations chosen by the prompt and its
  writes, in :func:`tiny_tokenizer` ids, with the cap and completion of ``receivers.base.Generation``;
- :func:`make_workspace`: a :class:`~latent_interconnect.workspace.Workspace` with real stores under a directory
  (there is no in-memory record store, ``artifacts.records``).
"""

from __future__ import annotations

import hashlib
import json
import string
from collections.abc import Generator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from latent_interconnect.artifacts.records import RecordStore
from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import load_config
from latent_interconnect.domains.base import (
    DOMAIN_DATA_KIND,
    DOMAINS,
    CaptureRecipe,
    Domain,
    Frame,
    RecipeMessage,
    TaskKind,
    TaskScope,
    TaskSpec,
)
from latent_interconnect.encoders.base import EXTRACTORS, ExtractedBatch, RawObject
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.encoders.store import commit_embeddings
from latent_interconnect.paths import Paths
from latent_interconnect.receivers.base import (
    RECEIVER_BACKENDS,
    Generation,
    GenerationParams,
    GenerationPlan,
    LossTerms,
    Readout,
    ReceiverSpec,
    check_score_request,
    checked_write_layers,
    end_of_generation,
    generated_text,
    write_layers,
)
from latent_interconnect.receivers.kernel import BatchPlan, LayerWrites, require_targets, resolve_sources
from latent_interconnect.receivers.render import RenderedPrompt
from latent_interconnect.workspace import Workspace
from latent_interconnect.writes.sources import WriteSource

ROOT = Path(__file__).resolve().parents[1]
TINY_LLAMA = ROOT / "configs" / "receivers" / "tiny-llama.yaml"

SPECIAL_TOKENS = ("<unk>", "<pad>", "<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<think>")
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% if thinking %}<think>{% endif %}{% endif %}"
)
"""Role markers around each message; ``thinking=True`` (a template kwarg, as Kimi's) opens a reasoning block."""


def tiny_tokenizer() -> object:
    """Return a ``PreTrainedTokenizerFast``: one token per character, ``" c"`` one token, unknown characters ``<unk>``.

    Offsets, special tokens, ``decode`` and ``apply_chat_template`` behave as in any Hugging Face tokenizer.
    """
    from tokenizers import Regex, decoders, models, pre_tokenizers
    from tokenizers import Tokenizer as RawTokenizer
    from transformers import PreTrainedTokenizerFast

    chars = [c for c in string.printable if not c.isspace()]
    vocab = [*SPECIAL_TOKENS, " ", "\n", *chars, *(" " + c for c in chars)]
    raw = RawTokenizer(models.WordLevel({t: i for i, t in enumerate(vocab)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Split(Regex(r" ?[^\s]|\s"), behavior="isolated")
    raw.decoder = decoders.Fuse()
    raw.add_special_tokens(list(SPECIAL_TOKENS))
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>"
    )
    tok.chat_template = CHAT_TEMPLATE
    return tok


def build_tiny_llama(directory: Path, seed: int = 0, depth: int | None = None) -> ReceiverSpec:
    """Save a random 4-layer Llama (sized by ``tiny-llama.yaml``) and :func:`tiny_tokenizer` into ``directory``.

    ``depth`` gives it that many decoder layers instead (the configs' layers of a deeper receiver run unchanged).

    Returns:
        ``configs/receivers/tiny-llama.yaml`` with ``model_id`` set to ``directory`` (and ``depth``, if given).
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    spec = load_config(TINY_LLAMA, ReceiverSpec)
    spec = spec.model_copy(update={"model_id": str(directory), "depth": depth or spec.depth})
    tok = tiny_tokenizer()
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=spec.d_model,
        intermediate_size=2 * spec.d_model,
        num_hidden_layers=spec.depth,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).save_pretrained(directory)
    tok.save_pretrained(directory)
    return spec


TOY_DIM = 8


@EXTRACTORS.register("fake-toy")
class ToyExtractor:
    """A deterministic encoder: each residue is a fixed random vector of its letter; pooled = mean of residues.

    The recipe's ``read`` layers are honored (layer ``i`` scales the residue vectors by ``1 + i``; ``"output"`` is
    layer 0); ``residues`` adds per-residue states; ``options.dim`` sets the width (default :data:`TOY_DIM`); an object
    whose value contains ``!`` is skipped (an input the encoder cannot read), and one whose value contains ``#`` makes
    the extraction raise (an encoder that fails, e.g. out of memory).
    """

    def __init__(self, recipe: Mapping[str, object], device: str) -> None:
        self.recipe, self.device = recipe, device
        options = recipe.get("options") or {}
        self.dim = int(options.get("dim", TOY_DIM)) if isinstance(options, Mapping) else TOY_DIM
        read = recipe.get("read", "output")
        self.layers = (0,) if read == "output" else tuple(int(x) for x in read)
        self.table = np.random.default_rng(0).normal(size=(128, self.dim)).astype(np.float32)

    def load(self) -> None:
        """Nothing to load."""

    def fingerprint(self) -> Mapping[str, str]:
        """Identify the computation."""
        return {"extractor": "fake-toy", "dim": str(self.dim), "dtype": "float32"}

    def extract(self, objects: Sequence[RawObject]) -> ExtractedBatch:
        """Return pooled vectors (and residues if the recipe asks) at the recipe's layers."""
        if failing := [o.object_id for o in objects if "#" in o.value]:
            raise RuntimeError(f"toy encoder failure on {failing}")
        skipped = {o.object_id: "unreadable: '!'" for o in objects if "!" in o.value}
        objects = [o for o in objects if o.object_id not in skipped]
        residues = {
            layer: [self.table[[ord(c) % 128 for c in o.value]] * (1 + layer) for o in objects] for layer in self.layers
        }
        pooled = {
            layer: np.stack([r.mean(axis=0) for r in rows]) if rows else np.zeros((0, self.dim), np.float32)
            for layer, rows in residues.items()
        }
        keep = residues if self.recipe.get("residues") else {}
        return ExtractedBatch(tuple(o.object_id for o in objects), pooled, keep, skipped=skipped)


def toy_recipe(layers: Sequence[int] = (0,), residues: bool = False) -> EncoderRecipe:
    """Return the encoder recipe of :class:`ToyExtractor` (domain ``fake-toy``, run in this process's ``lip``)."""
    return EncoderRecipe(
        name="fake-toy",
        extractor="fake-toy",
        domain="fake-toy",
        model_id="none",
        revision="v1",
        read=tuple(layers),
        pooling="mean",
        residues=residues,
        environment="lip",
        batch_tokens=4096,
    )


def commit_toy_embeddings(
    store: ArtifactStore, objects: Mapping[str, str], layers: Sequence[int] = (0,), residues: bool = False
) -> ArtifactRef:
    """Extract ``objects`` (id → sequence) with :class:`ToyExtractor` and commit one ``embeddings`` artifact.

    It is written with the store's own writer (``encoders.store.commit_embeddings``) and records the recipe and the
    fingerprint as an extraction does, so loaders that check the recipe accept it.
    """
    recipe = toy_recipe(layers, residues)
    extractor = ToyExtractor(recipe.model_dump(mode="json"), "cpu")
    batch = extractor.extract([RawObject(i, v) for i, v in sorted(objects.items())])
    key = hashlib.sha256(json.dumps([sorted(objects.items()), list(layers), residues]).encode()).hexdigest()
    meta = {"recipe": recipe.model_dump(mode="json"), "fingerprint": dict(extractor.fingerprint())}
    return commit_embeddings(store, f"toy-embeddings-{key[:16]}", [batch], meta)


TOY_OBJECTS = {
    "P01": ("MKCLLAVA", "alpha kinase"),
    "P02": ("MSTNPKPQ", "beta ligand"),
    "P03": ("MCCGHWKE", "gamma cyclin"),
    "P04": ("MAALLKGV", "delta synthol"),
    "P05": ("MKKCWYPR", "epsilon redox"),
    "P06": ("MTTSSAAG", "zeta oxidant"),
    "P07": ("MCWAKLLR", "eta transfer"),
    "P08": ("MNNQQEED", "theta lyasic"),
    "P09": ("MRCKLHHA", "iota mutant"),
    "P10": ("MGGPPSTV", "kappa hydrol"),
    "P11": ("MCAGKCLW", "lambda isomer"),
    "P12": ("MDEDEKKA", "mu peptidy"),
}
"""object id → (sequence, name). Splits: P01-P06 train, P07-P09 val, P10-P12 test; two clusters by parity.

Names end in different letters, so the character tokenizer gives each a different last name token (distinct capture
targets; constant targets would leave the ridge nothing to fit)."""


def commit_toy_domain(store: ArtifactStore) -> ArtifactRef:
    """Commit the :class:`ToyDomain` data as a ``domain_data`` artifact; cite it as ``DomainSpec(name="fake-toy")``."""
    ids = sorted(TOY_OBJECTS)
    data = {
        "objects": {i: {"value": TOY_OBJECTS[i][0], "name": TOY_OBJECTS[i][1]} for i in ids},
        "split": {i: "train" if n < 6 else "val" if n < 9 else "test" for n, i in enumerate(ids)},
    }
    staged = store.staging("toy-domain-v1")
    (staged / "domain.json").write_text(json.dumps(data, sort_keys=True))
    return store.commit(staged, DOMAIN_DATA_KIND, {"domain": "fake-toy", "version": 1})


@DOMAINS.register("fake-toy")
class ToyDomain(Domain):
    """Twelve protein-like objects read from a ``domain_data`` artifact (:func:`commit_toy_domain`).

    Labels: ``has_c`` (contains a cysteine; trained), ``long_name`` (name longer than 11 characters; held out) and
    ``length`` (numeric). Field ``description``. Frames ``carrier`` and ``yes_no``; capture recipe ``name``.
    """

    name, noun = "fake-toy", "protein"
    object_kind = "fake-toy"

    def __init__(self, data_root: Path) -> None:
        super().__init__(data_root)
        data = json.loads((Path(data_root) / "domain.json").read_text())
        index = pd.Index(sorted(data["objects"]), name="object_id")
        self._objects = pd.DataFrame([data["objects"][i] for i in index], index=index)
        self._split = pd.Series(data["split"]).reindex(index)

    def objects(self) -> pd.DataFrame:
        """Return the object table."""
        return self._objects

    def split(self) -> pd.Series:
        """Return the split."""
        return self._split

    def clusters(self) -> pd.Series:
        """Return two clusters, alternating."""
        return pd.Series([n % 2 for n in range(len(self._objects))], index=self._objects.index)

    def labels(self) -> pd.DataFrame:
        """Return ``has_c``, ``long_name`` and ``length``."""
        value, name = self._objects["value"], self._objects["name"]
        return pd.DataFrame(
            {
                "has_c": value.str.contains("C"),
                "long_name": name.str.len() > 11,
                "length": value.str.len().astype(float),
            }
        )

    def fields(self) -> pd.DataFrame:
        """Return a one-sentence description per object."""
        return pd.DataFrame({"description": "A protein called " + self._objects["name"] + "."})

    def tasks(self) -> Mapping[str, TaskSpec]:
        """Return two Yes/No tasks (trained, held out) and the description task; questions are bare."""
        yn = TaskKind.YES_NO
        return {
            "has_c": TaskSpec(
                id="has_c", kind=yn, question="Does it contain cysteine?", label="has_c", scope=TaskScope.TRAINED
            ),
            "long_name": TaskSpec(
                id="long_name", kind=yn, question="Is it long-named?", label="long_name", scope=TaskScope.HELD_OUT
            ),
            "describe": TaskSpec(
                id="describe",
                kind=TaskKind.TEXT,
                question="Describe it.",
                label="description",
                scope=TaskScope.TRAINED,
            ),
        }

    def frames(self) -> Mapping[str, Frame]:
        """Return the carrier frame and the Yes/No frame (which places a task's bare question)."""
        return {
            "carrier": Frame(id="carrier", user="Consider the protein<SLOT:x>."),
            "yes_no": Frame(id="yes_no", user="Here is a protein:<SLOT:x>. {question} Answer Yes or No."),
        }

    def capture_recipes(self) -> Mapping[str, CaptureRecipe]:
        """Return the original-style name recipe: the name as the assistant's prefilled answer."""
        recipe = CaptureRecipe(
            name="name",
            messages=(
                RecipeMessage(role="user", text="Name a protein."),
                RecipeMessage(role="assistant", text=" {name}"),
            ),
            read="name",
        )
        return {recipe.name: recipe}

    def scope_mask(self, task: TaskSpec, object_ids: pd.Index) -> pd.Series:
        """Every object is in scope."""
        del task
        return pd.Series(True, index=object_ids)


@RECEIVER_BACKENDS.register("fake-score")
class FakeScoringBackend:
    """A scoring backend without a model: a prompt's score is a fixed function of its ids plus its writes.

    For prompt n, ``s_n = h(input_ids) + Σ α·Σδ`` over the sites written in it, resolved and checked by
    ``receivers.base.check_score_request`` (the request check every backend applies first), with ``h`` a hash in
    [0, 1). ``next_token`` returns log-softmax over the requested ids of ``s_n · (+1, −1, +1, …)``, so the first id
    gains with positive writes; ``target_nll`` returns ``softplus(−s_n)``. Float32 on CPU.

    Args:
        spec: the receiver config.
        plan: the batch plan (``receivers.base`` contract; default ``BatchPlan()``); exposed as ``.plan``.
    """

    reads_banks: ClassVar[bool] = False

    def __init__(self, spec: ReceiverSpec, plan: BatchPlan | None = None) -> None:
        self.spec = spec
        self._plan = BatchPlan() if plan is None else plan

    @property
    def plan(self) -> BatchPlan:
        """The configured batch plan (the fake scores each prompt alone, so it changes nothing)."""
        return self._plan

    def execution(self) -> dict[str, str]:
        """Return the fake's execution with every key ``receivers.base`` requires (on the CPU, no devices)."""
        return {
            "backend": "fake-score",
            "dtype": "float32",
            "quantization": "none",
            "device_name": "cpu",
            "device_count": "0",
            "attention": "none",
            "revision": self.spec.revision,
            "weights": self.spec.model_id,
        }

    def score(self, prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource], readout: Readout) -> Tensor:
        """Return ``(N, len(token_ids))`` log-probs (``next_token``) or ``(N,)`` NLL (``target_nll``)."""
        request = check_score_request(
            prompts, sources, readout, self.spec.depth, self.plan
        )  # every backend's first check
        s = torch.tensor([_unit_hash(p.input_ids) for p in prompts], dtype=torch.float32)
        for writes in request.writes.values():
            contribution = writes.alphas.float() * writes.deltas.float().sum(dim=-1)
            s = s.index_add(0, writes.rows, contribution.detach())
        if readout.kind == "target_nll":
            return torch.nn.functional.softplus(-s)
        signs = torch.tensor([1.0 if j % 2 == 0 else -1.0 for j in range(len(readout.token_ids))])
        return torch.log_softmax(s[:, None] * signs[None, :], dim=-1)


@RECEIVER_BACKENDS.register("fake-train")
class FakeTrainingBackend(FakeScoringBackend):
    """A training backend without a model: target token t of prompt n has NLL ``softplus(h(n, t) − s_n)``.

    ``s_n = Σ α·mean(δ)`` over the sites written in prompt n (the mean over the width, so the loss does not saturate
    and every map tensor gets a gradient) and ``h`` a hash of the prompt's ids and the token, in [0, 1). The loss terms
    follow ``receivers.base.LossTerms`` (``Σ w·ℓ`` with gradients into the writes, ``Σ w``, detached per-prompt
    ``token_nll``), after the backends' checks (write layers, then targets). Every layer is on the CPU.
    """

    reference_options: ClassVar[Mapping[str, str]] = {}

    def placement(self) -> Mapping[int, torch.device]:
        """Every layer's input is on the CPU."""
        return dict.fromkeys(range(self.spec.depth), torch.device("cpu"))

    def loss(self, prompts: Sequence[RenderedPrompt], writes: Mapping[int, LayerWrites]) -> LossTerms:
        """Return the loss terms of the prompts' target tokens."""
        write_layers(writes, self.spec.depth)
        targets = require_targets(prompts, "loss")
        s = torch.zeros(len(prompts))
        for w in writes.values():
            s = s.index_add(0, w.rows, w.alphas.float() * w.deltas.float().mean(dim=-1))
        nll = []
        for i, (p, t) in enumerate(zip(prompts, targets, strict=True)):
            h = torch.tensor([_unit_hash((*p.input_ids, *t.token_ids[: j + 1])) for j in range(len(t.token_ids))])
            nll.append(torch.nn.functional.softplus(h - s[i]))
        weights = [torch.tensor(t.weights, dtype=torch.float32) for t in targets]
        weighted = sum((w * n).sum() for w, n in zip(weights, nll, strict=True))
        return LossTerms(
            weighted_sum=torch.as_tensor(weighted),
            weight_sum=torch.stack([w.sum() for w in weights]).sum(),
            token_nll=tuple(n.detach() for n in nll),
        )


@RECEIVER_BACKENDS.register("fake-generate")
class FakeGenerationBackend:
    """A generation backend without a model: each prompt's continuation is one of :attr:`TEXTS`.

    For prompt n the text is ``TEXTS[floor(len(TEXTS) · frac(h(input_ids) + Σ α·Σδ))]`` (``h`` as in
    :class:`FakeScoringBackend`, the writes resolved per prompt), so a connector's writes change the choice. The ids are
    the text's :func:`tiny_tokenizer` ids followed by ``</s>``; a continuation longer than ``max_new_tokens`` is cut
    there and unfinished (``text`` the decoded cut ids), as ``receivers.base.Generation`` states. Every call is
    recorded in :attr:`calls` (the prompts and parameters), so tests can inspect what was asked.

    Args:
        spec: the receiver config.
        generation_plan: the generation plan (``receivers.base`` contract); exposed as ``.generation_plan``.
    """

    TEXTS: tuple[str, ...] = ("Yes, it does.", "No.", "It is alpha kinase.", "Perhaps.")
    """The continuations to choose from (tests may set another tuple on the class)."""

    reads_banks: ClassVar[bool] = False

    def __init__(self, spec: ReceiverSpec, generation_plan: GenerationPlan | None = None) -> None:
        self.spec = spec
        self._plan = GenerationPlan() if generation_plan is None else generation_plan
        self.tokenizer = tiny_tokenizer()
        self.calls: list[tuple[tuple[RenderedPrompt, ...], GenerationParams]] = []

    @property
    def generation_plan(self) -> GenerationPlan:
        """The configured generation plan (the fake generates each prompt alone, so it changes nothing)."""
        return self._plan

    def execution(self) -> dict[str, str]:
        """Return the fake's execution (no accelerator)."""
        return {"backend": "fake-generate", "dtype": "float32", "device_name": "cpu", "device_count": "0"}

    def generate(
        self, prompts: Sequence[RenderedPrompt], sources: Mapping[str, WriteSource], params: GenerationParams
    ) -> list[Generation]:
        """Return one :class:`Generation` per prompt (write layers checked first, as every backend does).

        The chosen text's ids, then ``</s>``, are emitted token by token under the shared stopping rule
        (``receivers.base.end_of_generation``: end id or stop string) up to ``max_new_tokens``.
        """
        checked_write_layers(prompts, sources, self.spec.depth)
        self.calls.append((tuple(prompts), params))
        tok = self.tokenizer
        eos = {tok.eos_token_id}  # type: ignore[attr-defined]
        out = []
        for prompt in prompts:
            s = _unit_hash(prompt.input_ids)
            for writes in resolve_sources([prompt], sources).values():
                s += float((writes.alphas.float() * writes.deltas.float().sum(dim=-1)).sum())
            text = self.TEXTS[int(len(self.TEXTS) * (s % 1.0))]
            ids = [*tok.encode(text, add_special_tokens=False), *eos]  # type: ignore[attr-defined]
            generation = None
            for n in range(1, min(len(ids), params.max_new_tokens) + 1):
                ended = end_of_generation(ids[:n], eos=eos, stop=params.stop, decode=partial(generated_text, tok))  # type: ignore[arg-type]
                if ended is not None:
                    generation = Generation(text=ended, token_ids=tuple(ids[:n]), finished=True)
                    break
            if generation is None:
                cut = ids[: params.max_new_tokens]
                generation = Generation(text=generated_text(tok, cut), token_ids=tuple(cut), finished=False)  # type: ignore[arg-type]
            out.append(generation)
        return out

    def stream(
        self, prompt: RenderedPrompt, sources: Mapping[str, WriteSource], params: GenerationParams
    ) -> Generator[tuple[int, ...], None, None]:
        """Yield :meth:`generate`'s ids for the prompt alone, one per step (the protocol's stream)."""
        (generation,) = self.generate([prompt], sources, params)
        yield from ((t,) for t in generation.token_ids)


def _unit_hash(ids: Sequence[int]) -> float:
    digest = hashlib.sha256(json.dumps(list(ids)).encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def make_workspace(root: Path) -> Workspace:
    """Return a workspace whose data root is ``root`` (real artifact and record stores; no code checkout recorded)."""
    paths = Paths.from_env({"LIP_DATA": str(root)})
    return Workspace(
        paths, ArtifactStore(paths.artifacts), RecordStore(paths.records), paths.data / "lip" / "pins.yaml", None
    )


def fake_receiver(base: ReceiverSpec | None = None) -> ReceiverSpec:
    """Return ``base`` (default: ``tiny-llama.yaml``) with scoring served by ``fake-score``, generation by
    ``fake-generate``."""
    spec = base or load_config(TINY_LLAMA, ReceiverSpec)
    return spec.model_copy(update={"backends": {"score": "fake-score", "generate": "fake-generate"}})
