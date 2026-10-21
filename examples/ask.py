"""Ask a receiver about an object through a trained connector: the shortest use of the package.

    python examples/ask.py RECEIVER.yaml CONNECTOR EMBEDDING.npy "What does <SLOT:x> do?" [--max-new-tokens N]
    python examples/ask.py RECEIVER.yaml CONNECTOR EMBEDDING.npy \
        "Here is an unknown protein, given to you as an embedding:<SLOT:x>. Is it a kinase?" --yes-no ANSWER.yaml

Inputs are plain files: a receiver config, the connector and the object's embedding from the encoder the
connector was trained on (``examples/embed.py``). No pins or records are involved.

``CONNECTOR`` is a checkpoint directory or the name of a connector installed by ``release_download`` under
``$LIP_DATA``, e.g. ``gemma-3-4b-it__esmc-600m`` (:func:`connector_checkpoint`). An argument that exists on disk is
always taken as the directory, even when an installed connector has the same name; any other argument without a ``/``
is looked up as a name. A name that is not installed stops it with one line naming the connectors that are, and the
download command.

The embedding is a ``.npy`` file: the pooled vector ``(d_in,)`` (``<id>.npy``), or the per-residue states
``(n_residues, d_in)`` (``<id>.residues.npy``, from a recipe with ``residues: true``). The connector's map kinds say
which it reads (``Connector.input_kinds``): a pooled kind (``affine``, ``mlp``) the vector, a residue kind (``xattn``,
``xblock``) the residues. Given residues, the pooled vector is their mean (the ``mean`` pooling), so one residue file
serves every kind (a pooled kind given ``<id>.residues.npy`` reads their mean, and the stderr line below says so);
given only the vector, a residue kind is refused with a message naming the file it needs
(``<dir>/<id>.residues.npy`` next to the vector given). No flag depends on the kind.

Before answering, one line on stderr names the connector (its spec name and, for a checkpoint installed by
``release_download``, the bundle it came from), its kind and what it read. A file that does not fit the
connector (a wrong shape or width, a residue kind given the vector, a checkpoint that does not load) stops it with
one ``error: ...`` line on stderr and exit status 1.

``--yes-no ANSWER.yaml`` reads the answer probabilities instead of generating, as the ``yes_no`` suites score
(``evaluation.runners.YesNo``). ``ANSWER.yaml`` is the receiver's answer convention (``configs/answers/``,
``data.answers``), the one its suites cite: its instruction follows the question after one space, its prefix opens
the answer turn, thinking is off, and the next token's log-probabilities of its two answer texts (each one token) are
read under the full softmax. The question is the suite's wording without an answer instruction, e.g. the protein
frame ``"Here is an unknown protein, given to you as an embedding:<SLOT:x>. Is it a kinase?"``. It prints
P(Yes) and P(No) under the full softmax with each answer's token string and id, the log-odds
log P(Yes) - log P(No) (the suites' score, a score table's ``score``), and, labeled separately, the pair-normalized
P(Yes) / (P(Yes) + P(No)).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.config import load_config
from latent_interconnect.data.answers import AnswerConvention
from latent_interconnect.evaluation.runners import answer_log_probs, log_odds
from latent_interconnect.evaluation.runners import yes_no_prompt as _yes_no_prompt
from latent_interconnect.paths import Paths
from latent_interconnect.receivers import backend_class, open_generation, open_renderer, open_scoring
from latent_interconnect.receivers.base import GenerationParams, Readout, ReceiverSpec
from latent_interconnect.receivers.kernel import BatchPlan
from latent_interconnect.receivers.render import Conversation, Message, RenderedPrompt, Renderer, SlotBinding
from latent_interconnect.writes.checkpoint import CHECKPOINT_KIND, read_checkpoint
from latent_interconnect.writes.inputs import InputKind, ObjectBatch
from latent_interconnect.writes.sources import LiveSource


def object_batch(
    embedding: np.ndarray, kinds: frozenset[InputKind], pooling: str = "mean", *, source: str | Path | None = None
) -> ObjectBatch:
    """Return the one-object batch a connector reading ``kinds`` needs from a pooled vector or residue states.

    ``source``, the embedding's file if given, lets the residue refusal name the file it needs:
    ``<dir>/<id>.residues.npy`` next to ``<dir>/<id>.npy``.

    Raises:
        ValueError: for an array that is neither ``(d_in,)`` nor ``(n_residues, d_in)``, a residue kind given only
            the pooled vector, or a pooled kind given residues of a recipe whose ``pooling`` is not ``mean``.
    """
    x = torch.as_tensor(np.asarray(embedding), dtype=torch.float32)
    if x.ndim == 1:
        if InputKind.RESIDUES in kinds:
            wanted = "<id>.residues.npy"
            if source is not None:
                path = Path(source)
                stem = path.name.removesuffix(".npy")
                wanted = str(path.with_name(f"{stem}.residues.npy"))
            raise ValueError(
                f"this connector's map kind reads per-residue states: give {wanted}, which examples/embed.py writes "
                "with the connector's residue recipe (the recipe it was trained on, residues: true)"
            )
        return ObjectBatch(("x",), pooled=x[None])
    if x.ndim != 2 or len(x) == 0:
        raise ValueError(f"an embedding is (d_in,) or (n_residues, d_in), not {tuple(x.shape)}")
    if InputKind.POOLED in kinds and pooling != "mean":
        raise ValueError(f"the recipe pools by {pooling!r}, not the residue mean: give the pooled <id>.npy")
    mask = torch.ones(1, len(x), dtype=torch.bool)
    return ObjectBatch(("x",), pooled=x.mean(0)[None], residues=x[None], mask=mask)


def installed_bundle(checkpoint: Path) -> str | None:
    """Return the name of the installed bundle holding ``checkpoint``, or None.

    A bundle install keeps each ``bundle.json`` as ``<data root>/lip/bundles/<name>.json`` and the checkpoint as
    ``<data root>/lip/artifacts/<xx>/<sha256>``; any other directory names no bundle.
    """
    path = Path(checkpoint).resolve()
    if path.parent.parent.name != "artifacts" or path.parent.parent.parent.name != "lip":
        return None
    for kept in sorted((path.parent.parent.parent / "bundles").glob("*.json")):
        try:
            bundle = json.loads(kept.read_text())
        except (OSError, ValueError):
            continue
        if any(a.get("sha256") == path.name for a in bundle.get("artifacts", ())):
            return str(bundle.get("name", kept.stem))
    return None


def installed_connectors(data_root: str | Path) -> list[str]:
    """Return the names of the connectors installed under ``data_root``, sorted.

    A connector is a kept ``<data root>/lip/bundles/<name>.json`` that lists a checkpoint artifact (data bundles,
    kept there too, list none).
    """
    names = []
    for kept in sorted((Path(data_root) / "lip" / "bundles").glob("*.json")):
        try:
            bundle = json.loads(kept.read_text())
        except (OSError, ValueError):
            continue
        if any(a.get("kind") == CHECKPOINT_KIND for a in bundle.get("artifacts", ())):
            names.append(kept.stem)
    return names


def connector_checkpoint(connector: str | Path, data_root: str | Path | None = None) -> Path:
    """Return the checkpoint directory ``connector`` names: a directory as given, or an installed connector's.

    An argument that exists on disk, or that contains a ``/``, is a directory and is returned unchanged. Any other is
    the name of a connector installed by ``release_download`` under ``data_root`` (default ``$LIP_DATA``): its kept
    ``<data root>/lip/bundles/<name>.json`` lists the checkpoint artifact, installed as
    ``<data root>/lip/artifacts/<xx>/<sha256>``, the directory the download report prints.

    Raises:
        ValueError: for a name that is not installed (the message names those that are, and the download command),
            or a bundle that lists no single checkpoint.
    """
    given = Path(connector)
    if given.exists() or "/" in str(connector):
        return given
    name = str(connector)
    root = data_root if data_root is not None else os.environ.get("LIP_DATA")
    if not root:
        raise ValueError(f"no directory {name!r}; to use an installed connector of that name, set LIP_DATA")
    paths = Paths.from_env({"LIP_DATA": str(root)})  # the layout release_download installs into
    kept = paths.data / "lip" / "bundles" / f"{name}.json"
    if name not in installed_connectors(paths.data):
        have = ", ".join(installed_connectors(paths.data)) or "none"
        raise ValueError(
            f"no directory {name!r} and no connector {name!r} installed under {paths.data} (installed: {have}); "
            f"download it with: python -m latent_interconnect.release_download --connector {name} --dest $LIP_DATA"
        )
    bundle = json.loads(kept.read_text())
    store = ArtifactStore(paths.artifacts)
    checkpoints = [store.path(store.ref(a["sha256"])) for a in bundle["artifacts"] if a.get("kind") == CHECKPOINT_KIND]
    if len(checkpoints) != 1:
        raise ValueError(f"connector {name!r} lists {len(checkpoints)} checkpoints; give the checkpoint directory")
    return checkpoints[0]


def _prepared(
    receiver: Path,
    checkpoint: str | Path,
    embedding: np.ndarray,
    *,
    log: Callable[[str], object] | None,
    embedding_file: str | Path | None,
) -> tuple[ReceiverSpec, LiveSource, int]:
    """Return the receiver config, the connector's write source for the one object and its slot-token count k.

    ``checkpoint`` is a checkpoint directory or an installed connector's name (:func:`connector_checkpoint`).

    ``log`` and ``embedding_file`` as :func:`ask`.
    """
    spec = load_config(receiver, ReceiverSpec)
    checkpoint = connector_checkpoint(checkpoint)
    connector, _ = read_checkpoint(checkpoint)
    pooling = json.loads((Path(checkpoint) / "connector.json").read_text())["encoder"]["pooling"]
    batch = object_batch(embedding, connector.input_kinds, pooling, source=embedding_file)
    if log is not None:
        if InputKind.RESIDUES in connector.input_kinds:
            read = f"residues ({batch.residues.shape[1] if batch.residues is not None else 0} rows)"
        elif batch.residues is not None:
            read = f"pooled (the mean of the {batch.residues.shape[1]} residue rows given)"
        else:
            read = "pooled"
        bundle = installed_bundle(checkpoint)
        named = f"{connector.spec.name} (bundle {bundle})" if bundle else connector.spec.name
        log(f"connector {named}: kind {connector.spec.kind}, reads {read}")
    return spec, LiveSource(connector, lambda _ids: batch), connector.spec.k


def ask(
    receiver: Path,
    checkpoint: str | Path,
    embedding: np.ndarray,
    question: str,
    max_new_tokens: int = 512,
    *,
    log: Callable[[str], object] | None = None,
    embedding_file: str | Path | None = None,
) -> str:
    """Return the receiver's answer to ``question``, whose ``<SLOT:x>`` marker carries the object.

    ``checkpoint`` is a checkpoint directory or an installed connector's name (:func:`connector_checkpoint`).

    ``log``, if given, receives one line before generation: the connector's name (and its installed bundle's,
    :func:`installed_bundle`), its kind and what it read. ``embedding_file`` is the embedding's file
    (:func:`object_batch`'s ``source``).
    """
    spec, source, k = _prepared(receiver, checkpoint, embedding, log=log, embedding_file=embedding_file)
    prompt = open_renderer(spec).render(Conversation((Message("user", question),)), {"x": SlotBinding("x", "x", k)})
    params = GenerationParams(max_new_tokens=max_new_tokens)
    return open_generation(spec).generate([prompt], {"x": source}, params)[0].text


@dataclass(frozen=True)
class YesNoAnswer:
    """The answer probabilities of one Yes/No question, as a ``yes_no`` suite reads them (:func:`ask_yes_no`).

    Attributes:
        yes: the positive answer's text (the convention's, e.g. ``" Yes"``), read as one token.
        no: the negative answer's text.
        yes_id: the token id of ``yes``.
        no_id: the token id of ``no``.
        logp_yes: log P(yes) of the next token under the full softmax (a score table's ``logp_yes``).
        logp_no: log P(no), likewise (``logp_no``).
        score: ``logp_yes - logp_no``, the suites' score (``evaluation.runners.log_odds``).
    """

    yes: str
    no: str
    yes_id: int
    no_id: int
    logp_yes: float
    logp_no: float
    score: float

    @property
    def p_yes(self) -> float:
        """P(yes) of the next token."""
        return math.exp(self.logp_yes)

    @property
    def p_no(self) -> float:
        """P(no) of the next token."""
        return math.exp(self.logp_no)

    @property
    def pair_normalized(self) -> float:
        """The pair-normalized P(yes) / (P(yes) + P(no)) = 1 / (1 + exp(-score)); not a full-softmax probability."""
        return 1.0 / (1.0 + math.exp(-self.score))

    def lines(self) -> list[str]:
        """Return the printed report: the full-softmax probabilities, the log-odds and the pair-normalized value."""
        return [
            f"P(Yes) = {self.p_yes:.6g}  (full softmax; token {self.yes!r}, id {self.yes_id})",
            f"P(No) = {self.p_no:.6g}  (full softmax; token {self.no!r}, id {self.no_id})",
            f"log P(Yes) - log P(No) = {self.score:.6g}  (the suites' score)",
            f"pair-normalized P(Yes) / (P(Yes) + P(No)) = {self.pair_normalized:.6g}",
        ]


def yes_no_prompt(
    renderer: Renderer, convention: AnswerConvention, question: str, k: int
) -> tuple[RenderedPrompt, Readout]:
    """Return the prompt of the Yes/No ``question`` and the readout of its two answers, as a ``yes_no`` suite has them.

    The user message is ``question`` with the convention's instruction after one space, the answer turn opens with
    its prefix, thinking is off, and the readout is the next token's (yes, no) ids
    (``evaluation.runners.yes_no_prompt``, which the chat's Yes/No readout renders too). The slot ``x`` takes ``k``
    tokens of connector ``x``.

    Raises:
        ValueError: if an answer text is not exactly one token of the receiver's tokenizer.
    """
    return _yes_no_prompt(renderer, convention.policy(), question, {"x": SlotBinding("x", "x", k)})


def ask_yes_no(
    receiver: Path,
    checkpoint: str | Path,
    embedding: np.ndarray,
    question: str,
    answer: Path | AnswerConvention,
    *,
    log: Callable[[str], object] | None = None,
    embedding_file: str | Path | None = None,
) -> YesNoAnswer:
    """Return the answer probabilities of the Yes/No ``question``, rendered and read as a ``yes_no`` suite does.

    ``answer`` is the receiver's answer convention (a ``configs/answers/`` file, or its config): its instruction
    follows ``question`` after one space (``Answering.user``), its prefix opens the answer turn, thinking is off, and
    the readout is the next token's log-probabilities of its two answers (``evaluation.runners.answer_readout``, each
    one token). On a backend that pads (``hf``), every batch is right-padded by one masked column
    (``BatchPlan.pad_every_batch``), as the release's suite configs state. ``log`` and ``embedding_file`` as
    :func:`ask`; ``log`` also receives a line naming the instruction and prefix.

    Raises:
        ValueError: if an answer text is not exactly one token of the receiver's tokenizer, or as :func:`ask`.
    """
    convention = answer if isinstance(answer, AnswerConvention) else load_config(answer, AnswerConvention)
    spec, source, k = _prepared(receiver, checkpoint, embedding, log=log, embedding_file=embedding_file)
    if log is not None:
        log(
            f"yes/no readout: instruction {convention.instruction!r}, answer prefix {convention.prefix!r}, thinking off"
        )
    prompt, readout = yes_no_prompt(open_renderer(spec), convention, question, k)
    pads = bool(getattr(backend_class(spec, "score"), "pads_every_batch", False))
    backend = open_scoring(spec, plan=BatchPlan(pad_every_batch=pads))
    read = backend.score([prompt], {"x": source}, readout)
    (logp_yes, logp_no), score = answer_log_probs(read)[0], log_odds(read)[0]
    yes_id, no_id = readout.token_ids
    return YesNoAnswer(convention.yes, convention.no, yes_id, no_id, float(logp_yes), float(logp_no), float(score))


def _stderr_line(line: str) -> None:
    sys.stderr.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "Yes/no example (Gemma 3): ask.py RECEIVER.yaml gemma-3-4b-it__esmc-600m protein.npy "
            '"Here is an unknown protein, given to you as an embedding:<SLOT:x>. Is it a kinase?" '
            "--yes-no configs/answers/gemma-3.yaml. Write the question without an answer instruction; the "
            "convention adds it. The output is P(Yes) and P(No) under the full softmax, the log-odds "
            "log P(Yes) - log P(No), which is the score the evaluation suites record and their AUROCs rank, and "
            "the pair-normalized P(Yes) / (P(Yes) + P(No))."
        ),
    )
    parser.add_argument("receiver", type=Path, help="the receiver config (YAML)")
    parser.add_argument(
        "checkpoint",
        metavar="CONNECTOR",
        help=(
            "the connector: its checkpoint directory, or the name of a connector installed under $LIP_DATA by "
            "release_download (e.g. gemma-3-4b-it__esmc-600m). A path that exists is always taken as the directory."
        ),
    )
    parser.add_argument("embedding", help="the object's embedding: <id>.npy (pooled) or <id>.residues.npy")
    parser.add_argument("question", help="the question; its <SLOT:x> marker carries the object")
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="cap on the answer's length in tokens (default 512)"
    )
    parser.add_argument(
        "--yes-no",
        type=Path,
        metavar="ANSWER.yaml",
        help=(
            "score a yes/no question as the evaluation suites do instead of generating: the receiver's answer "
            "convention (configs/answers/gemma-3.yaml for Gemma 3, configs/answers/kimi-k2.6.yaml for Kimi "
            "K2.6) adds its instruction and answer prefix, with thinking off. For Kimi K2.6 the receiver's score "
            "backend (kimi-stream) needs the full Kimi weights and a GPU, and its tokenizer needs tiktoken "
            "(CONTRIBUTING.md)."
        ),
    )
    args = parser.parse_args()
    try:
        embedding = np.load(args.embedding)
        if args.yes_no is None:
            out = (
                ask(
                    args.receiver,
                    args.checkpoint,
                    embedding,
                    args.question,
                    args.max_new_tokens,
                    log=_stderr_line,
                    embedding_file=args.embedding,
                )
                + "\n"
            )
        else:
            read = ask_yes_no(
                args.receiver,
                args.checkpoint,
                embedding,
                args.question,
                args.yes_no,
                log=_stderr_line,
                embedding_file=args.embedding,
            )
            out = "".join(line + "\n" for line in read.lines())
    except (
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        ImportError,
    ) as e:  # incl. pydantic's ValidationError, IntegrityError
        sys.stderr.write(f"error: {e}\n")
        raise SystemExit(1) from None
    sys.stdout.write(out)
