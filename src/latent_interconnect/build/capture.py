"""Receiver-state capture: the receiver's own hidden states for objects, the targets of ridge initialization.

A capture runs one declared *capture recipe* (``domains.base.CaptureRecipe``) of the object's domain:

- **Prompt.** The one renderer renders the recipe (:meth:`~latent_interconnect.receivers.render.Renderer.render_recipe`)
  with ``{noun}`` (the domain's noun), ``{name}`` (the object's name) and ``{context}`` (the recipe's
  ``context_field`` from the zero-shot-screened training text, ``data.training_text``, fitted by
  :func:`fit_context`). The name and the read placeholder are their own tokenization pieces, with one leading space
  after text of their message (the original capture's ``" " + name`` piece); a trailing assistant message is
  the prefill, the answer the receiver reads with no end of turn. Objects without that field in the training text are
  left out; their count is recorded in the artifact (``left_out``).
- **Read.** The read span is every token of the read placeholder's piece, its leading space included, as the original
  name-token mean was.
- **States.** :meth:`~latent_interconnect.receivers.base.CaptureBackend.capture_span` reduces the span on the
  device and returns, per layer, :class:`~latent_interconnect.receivers.base.SpanStates`: the last read token's state
  and the mean over the read tokens (no per-token state reaches the host). The artifact holds both: the state the
  recipe's ``target`` names (``last``, or ``mean``) as the pooled vector, and the mean as ``means`` (the ridge bias,
  ``build.ridge``, always the mean). A consumer maps to the pooled state by default or selects the means
  (``RidgeSpec.pool``, ``WrittenScan.pool``), so either target comes from one capture.

**Batching** (``examples/guides/batching.md``). The capture's prompts are partitioned once, by the one planner
(``kernel.plan_batches`` over ``kernel.planned_lengths(prompts, "capture")``) under the capture's ``plan``, which the
backend must share; the partition's digest is recorded in the artifact (``batch_plan``), so "the same micro-batches"
is checkable. Parts, the storage and resume unit, are runs of whole micro-batches of at least :data:`PART_OBJECTS`
objects, and a pass streams a run of whole parts, so every backend call re-plans exactly the micro-batches it was
given. An interrupted capture resumes at its first unwritten part, and a committed one is returned, not recomputed;
either only once the store accepts the capture's origin (:func:`capture_origin`, ``ArtifactStore.require_origin``).

The stage opens the renderer and the capture backend (with the capture's plan) and passes them in
(``run.capture``); it records the backend's ``execution()``, the pass size and the memory budget. The pass
size (how many objects stream through the receiver's layers together; by default the whole job, as the old capture)
and the memory budget (:func:`pass_bytes`; passes are split to fit it) are execution options, not identity.
Runs in the receiver's environment.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pydantic import Field

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.config import ArtifactDigest, StageConfig
from latent_interconnect.data.items import ITEM_SET_KIND, load_item_set
from latent_interconnect.data.training_text import TRAINING_SET_KIND, load_training_text
from latent_interconnect.domains.base import CaptureRecipe, Domain, DomainSpec, open_domain
from latent_interconnect.encoders.base import ExtractedBatch
from latent_interconnect.encoders.store import PartWriter, finalize_embeddings, writer_lock
from latent_interconnect.paths import running_code
from latent_interconnect.provenance import DeviceClass, hardware_of
from latent_interconnect.receivers.base import CaptureBackend, ReceiverSpec
from latent_interconnect.receivers.kernel import BatchPlan, Partition, partition_digest, plan_batches, planned_lengths
from latent_interconnect.receivers.render import RenderedPrompt, Renderer

__all__ = [
    "PART_OBJECTS",
    "CaptureSpec",
    "JobMemory",
    "capture_key",
    "capture_origin",
    "capture_prompts",
    "capture_states",
    "context_cuts",
    "default_memory_budget",
    "fit_context",
    "host_memory_bytes",
    "job_memory_limit",
    "pass_bytes",
]

PART_OBJECTS = 256
"""The least objects per stored part: a part, the storage and resume unit, is a run of whole micro-batches of the
capture's one partition holding at least this many objects (the last may hold fewer), so a part never splits a
micro-batch. It shapes the artifact's files, so it is a fixed constant, not tuned per run and not part of the
capture's key. How many objects stream through the receiver together is the separate ``pass_objects`` execution
option of :func:`capture_states`."""
_PLACEHOLDERS = ("noun", "name", "context")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class CaptureSpec(StageConfig):
    """One capture.

    Attributes:
        recipe: the capture recipe's name in the domain (``Domain.capture_recipes``); also the target name.
        receiver: the receiver (embedded, so the capture's identity covers it).
        domain: the domain at a fixed data version (its recipes, names and objects).
        training_set: the screened training set whose text fills ``{context}`` (``build.zero_shot``); required by a
            recipe with a context, refused otherwise. Objects without that field are left out, and their count is
            recorded.
        layers: receiver layers to keep.
        objects: digest of the object list (``data.items``).
        device_class: the declared hardware (``A100`` with ``kimi-stream``, as scoring); captures feed every
            ridge, so their precision and hardware are part of the step-0 match.
        plan: the batch plan (``receivers.kernel.BatchPlan``): the capture's prompts are partitioned once by
            ``kernel.plan_batches``, and the backend must be configured with the same plan.
    """

    recipe: str
    receiver: ReceiverSpec
    domain: DomainSpec
    training_set: ArtifactDigest | None = None
    layers: tuple[int, ...] = Field(min_length=1)
    objects: ArtifactDigest
    device_class: DeviceClass
    plan: BatchPlan = BatchPlan()


def capture_key(spec: CaptureSpec) -> str:
    """Return the capture's computation key (its staging directory name)."""
    return f"capture-{spec.digest()[:32]}"


def capture_origin(backend: CaptureBackend, devices: Sequence[str]) -> dict[str, object]:
    """Return a capture's origin as it runs (``ArtifactStore.require_origin``).

    It holds the code (commit, dirty), the hardware, the backend's ``execution()`` (dtype, quantization, revision)
    and the torch and transformers versions. The weights are identified by ``revision``: every backend loads or checks
    its weights at the receiver's pinned revision (``hf`` loads ``model_id`` at it; the Kimi backends refuse a copy
    whose model files differ from the pinned snapshot's, ``kimi_k26.weights.check_weights``). The directory they were
    read from, ``execution()["weights"]``, is left out: node-local copies differ by node and job.
    """
    commit, dirty = running_code()
    libraries = {name: importlib.metadata.version(name) for name in ("torch", "transformers")}
    hardware = list(hardware_of(devices))
    execution = {k: v for k, v in backend.execution().items() if k != "weights"}
    return {"commit": commit, "dirty": dirty, "hardware": hardware, "execution": execution} | libraries


def fit_context(text: str, cap: int, count: Callable[[str], int]) -> str:
    """Return the whole sentences of ``text`` that fit in ``cap`` tokens.

    A first sentence longer than the cap is cut at its last word boundary within the cap.

    Args:
        text: the context text.
        cap: the token cap.
        count: the receiver's token count of a text.

    Raises:
        ValueError: if not even the first word fits.
    """
    sentences = [s for s in _SENTENCE_END.split(text.strip()) if s]
    kept: list[str] = []
    for sentence in sentences:
        if count(" ".join([*kept, sentence])) > cap:
            break
        kept.append(sentence)
    if kept:
        return " ".join(kept)
    words = sentences[0].split(" ") if sentences else []
    lo, hi = 0, len(words)  # the largest k with the first k words within the cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(" ".join(words[:mid])) <= cap:
            lo = mid
        else:
            hi = mid - 1
    if lo == 0:
        raise ValueError(f"not even the first word of {text[:40]!r}... fits in {cap} tokens")
    return " ".join(words[:lo])


_PROC_CGROUP = Path("/proc/self/cgroup")
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_CGROUP_FIELDS = 3
_V1_UNLIMITED = 2**62
"""cgroup v1 states "no limit" as a page-rounded maximum signed 64-bit value; anything this large is no limit."""
_STATE_BYTES = 4
"""Bytes per stored value of a captured or streamed state (float32, the widest the backends return on GPU)."""


def pass_bytes(prompts: Sequence[RenderedPrompt], n_layers: int, width: int) -> int:
    """Estimate a pass's host memory: the reduced states and the residual stream.

    Per captured layer, ``SpanStates`` holds two ``(N, width)`` arrays (last and mean); the receiver's residual
    stream holds one state per prompt token while it streams the layers. Both at :data:`_STATE_BYTES` per value.
    """
    tokens = sum(len(p.input_ids) for p in prompts)
    return _STATE_BYTES * width * (2 * len(prompts) * n_layers + tokens)


def capture_prompts(
    spec: CaptureSpec, store: ArtifactStore, renderer: Renderer
) -> tuple[tuple[str, ...], list[RenderedPrompt], int]:
    """Return the captured objects (object-list order), their rendered prompts, and how many were left out.

    Raises:
        KeyError: if the recipe is not in the domain.
        ValueError: if the training set is missing for a recipe with a context (or given without one), an object is
            not in the domain, or a name to read is empty.
    """
    _, ids, prompts, left_out = _plan(spec, store, renderer)
    return ids, prompts, left_out


def _plan(
    spec: CaptureSpec, store: ArtifactStore, renderer: Renderer
) -> tuple[CaptureRecipe, tuple[str, ...], list[RenderedPrompt], int]:
    domain = open_domain(spec.domain, store)
    recipe = domain.capture_recipes()[spec.recipe]
    if (recipe.context_field is None) != (spec.training_set is None):
        raise ValueError(f"recipe {recipe.name!r}: a training set goes with a context, and only with one")
    ids = load_item_set(store, ArtifactRef(spec.objects, ITEM_SET_KIND)).object_ids
    names = _names(domain, ids)
    contexts = _contexts(spec, store, recipe)
    count = _counter(renderer)
    used = {key for m in recipe.messages for key in _PLACEHOLDERS if "{" + key + "}" in m.text}
    kept, prompts = [], []
    for oid in ids:
        values = {"noun": domain.noun, "name": names[oid]}
        if recipe.context_field is not None:
            text = contexts.get(oid, {}).get(recipe.context_field)
            if not text:
                continue
            values["context"] = fit_context(text, int(recipe.context_tokens or 0), count)
        if recipe.read == "name" and not values["name"]:
            raise ValueError(f"{oid} has no name to read")
        kept.append(oid)
        filled = {k: v for k, v in values.items() if k in used}
        prompts.append(renderer.render_recipe(recipe.messages, filled, recipe.read, thinking=recipe.thinking))
    return recipe, tuple(kept), prompts, len(ids) - len(kept)


def _contexts(spec: CaptureSpec, store: ArtifactStore, recipe: CaptureRecipe) -> Mapping[str, Mapping[str, str]]:
    """Return object → the recipe's context field from the capture's training set (empty without one)."""
    if spec.training_set is None:
        return {}
    ref = ArtifactRef(spec.training_set, TRAINING_SET_KIND)
    return load_training_text(store, ref, fields=[recipe.context_field or ""]).fields


def _counter(renderer: Renderer) -> Callable[[str], int]:
    """Return the receiver's token count of a text (no special tokens), as :func:`fit_context` takes it."""

    def count(text: str) -> int:
        return len(renderer.tokenizer.encode(text, add_special_tokens=False))

    return count


def context_cuts(spec: CaptureSpec, store: ArtifactStore, renderer: Renderer, ids: Sequence[str]) -> dict[str, bool]:
    """Return, for each of ``ids`` the capture gives a context, whether :func:`fit_context` cut that context.

    True: the prompt holds fewer sentences (or words) than the training text's field; False: the whole field. An
    object without the field (left out of the capture) and every object of a recipe without a context are absent.
    Derived by rendering again on the CPU, as :func:`capture_prompts`; the capture stored no text.

    Raises:
        KeyError: if the recipe is not in the domain.
    """
    recipe = open_domain(spec.domain, store).capture_recipes()[spec.recipe]
    if recipe.context_field is None:
        return {}
    contexts, count = _contexts(spec, store, recipe), _counter(renderer)
    out = {}
    for oid in ids:
        text = contexts.get(oid, {}).get(recipe.context_field)
        if text:
            whole = " ".join(part for part in _SENTENCE_END.split(text.strip()) if part)
            out[oid] = fit_context(text, int(recipe.context_tokens or 0), count) != whole
    return out


def _names(domain: Domain, ids: Sequence[str]) -> Mapping[str, str]:
    table = domain.objects()
    missing = [o for o in ids if o not in table.index]
    if missing:
        raise ValueError(f"{len(missing)} objects are not in domain {domain.name!r}, e.g. {missing[:3]}")
    return {o: str(table.at[o, "name"]) for o in ids}


def capture_states(
    spec: CaptureSpec,
    store: ArtifactStore,
    renderer: Renderer,
    backend: CaptureBackend,
    devices: Sequence[str],
    *,
    pass_objects: int | None = None,
    memory_bytes: int | None = None,
    handoff: Mapping[str, object] | None = None,
) -> ArtifactRef:
    """Render, capture and commit an ``embeddings``-format artifact of receiver states (target and bias per layer).

    Args:
        spec: the capture.
        store: the artifact store.
        renderer: the receiver's renderer (``receivers.open_renderer``), opened by the stage.
        backend: the receiver's capture backend, opened by the stage (which records its ``execution()``).
        devices: the visible accelerators' names, checked against ``spec.device_class``.
        pass_objects: an execution option, not identity: at most this many objects go through the backend in one
            pass (one streaming of the receiver's layers); None (the default) streams the whole job in one pass, as
            the old capture did. A pass always holds whole parts, at least one.
        memory_bytes: an execution option: the host-memory budget of one pass (:func:`pass_bytes`); a pass that
            would exceed it is split at a part boundary. None (the default) is :func:`default_memory_budget`: half
            the job's memory limit (cgroup ``memory.max``), or half the host's memory without one.
        handoff: a declared origin handoff (``ArtifactStore.require_origin``), for work of other code.

    Raises:
        WrongHardwareError: if ``devices`` are not ``spec.device_class``.
        MemoryError: if a single part's estimate exceeds ``memory_bytes`` (no split can fit it).
        ValueError: as :func:`capture_prompts`, if no object is left, ``pass_objects`` or ``memory_bytes`` is below
            1, a part written earlier holds other objects than this capture's part (another part layout), or the
            backend returns other layers or shapes.
        OriginError: if the capture's work comes from another origin and no valid handoff covers it.
    """
    spec.device_class.require(devices)
    if pass_objects is not None and pass_objects < 1:
        raise ValueError(f"pass_objects {pass_objects}: at least one object per pass")
    if memory_bytes is not None and memory_bytes < 1:
        raise ValueError(f"memory_bytes {memory_bytes}: a budget is positive")
    budget_bytes = memory_bytes if memory_bytes is not None else default_memory_budget()[0]
    key = capture_key(spec)
    store.require_origin(key, capture_origin(backend, devices), handoff)
    done = store.committed(key)
    if done is not None:
        return done
    recipe, ids, prompts, left_out = _plan(spec, store, renderer)
    if not ids:
        raise ValueError(f"capture {spec.recipe!r}: every object was left out")
    staging = store.resume(key)
    if isinstance(staging, ArtifactRef):  # another capture of the key committed while this one planned
        return staging
    with writer_lock(staging) as token:  # a killed capture's temporary parts are swept by the next writer
        if backend.plan != spec.plan:
            raise ValueError(f"the backend batches with {backend.plan}, the capture declares {spec.plan}")
        partition = plan_batches(planned_lengths(prompts, "capture"), spec.plan)
        parts = _parts_of(partition)
        pending = _pending_parts(staging, parts, ids, token)
        width, depth = spec.receiver.d_model, len(spec.layers)

        def cost(chosen: Sequence[int]) -> int:
            return pass_bytes([prompts[r] for i in chosen for r in parts[i]], depth, width)

        objects_cap = pass_objects if pass_objects is not None else len(ids)
        for batch in _passes(pending, parts, objects_cap, budget_bytes, cost):
            rows = [r for i, _ in batch for r in parts[i]]
            states = backend.capture_span([prompts[r] for r in rows], spec.layers)
            if set(states) != set(spec.layers):
                raise ValueError(f"backend returned layers {sorted(states)}, asked for {sorted(spec.layers)}")
            lasts = {L: _rows(states[L].last, len(rows), width) for L in spec.layers}
            means_all = {L: _rows(states[L].mean, len(rows), width) for L in spec.layers}
            at = {r: j for j, r in enumerate(rows)}
            for i, writer in batch:
                sel = [at[r] for r in parts[i]]
                pooled, means = {}, {}
                for L in spec.layers:
                    means[L] = means_all[L][sel]
                    pooled[L] = means[L] if recipe.target == "mean" else lasts[L][sel]
                writer.write(ExtractedBatch(tuple(ids[r] for r in parts[i]), pooled, {}, means))
    meta = {
        "spec": spec.digest(),
        "recipe": spec.recipe,
        "target": recipe.target,
        "left_out": left_out,
        "batch_plan": {"plan": spec.plan.identity(), "partition": partition_digest(partition)},
    }
    return finalize_embeddings(store, staging, len(parts), meta)


def _parts_of(partition: Partition) -> list[tuple[int, ...]]:
    """Cut the micro-batches, in partition order, into parts of whole micro-batches.

    A part closes once it holds at least :data:`PART_OBJECTS` objects (the last part may hold fewer), so every pass,
    a run of whole parts, streams consecutive micro-batches of the one partition.
    """
    parts: list[tuple[int, ...]] = []
    current: list[int] = []
    for micro in partition:
        current.extend(micro)
        if len(current) >= PART_OBJECTS:
            parts.append(tuple(current))
            current = []
    if current:
        parts.append(tuple(current))
    return parts


def _pending_parts(
    staging: Path, parts: Sequence[Sequence[int]], ids: Sequence[str], token: str
) -> list[tuple[int, PartWriter]]:
    """Return the parts left to write; a part already written must hold this capture's objects for it."""
    pending = []
    for i, rows in enumerate(parts):
        writer = PartWriter(staging, i, token)
        if not writer.done():
            pending.append((i, writer))
        elif json.loads((writer.final / "ids.json").read_text()) != [ids[r] for r in rows]:
            raise ValueError(f"{writer.final}: written for another part layout; this capture's part {i} differs")
    return pending


def _passes(
    pending: list[tuple[int, PartWriter]],
    parts: Sequence[Sequence[int]],
    objects_cap: int,
    budget_bytes: int,
    cost: Callable[[Sequence[int]], int],
) -> Iterator[list[tuple[int, PartWriter]]]:
    """Group the pending parts, in order, into passes within both caps; a pass holds whole parts, at least one.

    Raises:
        MemoryError: if one part alone is over the memory budget.
    """
    while pending:
        batch = [pending.pop(0)]
        if cost([batch[0][0]]) > budget_bytes:
            raise MemoryError(f"part {batch[0][0]} needs {cost([batch[0][0]])} bytes, over the {budget_bytes} budget")
        while (
            pending
            and sum(len(parts[i]) for i, _ in batch) + len(parts[pending[0][0]]) <= objects_cap
            and cost([i for i, _ in batch] + [pending[0][0]]) <= budget_bytes
        ):
            batch.append(pending.pop(0))
        yield batch


def host_memory_bytes() -> int:
    """Return this host's physical memory in bytes."""
    return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))


@dataclass(frozen=True)
class JobMemory:
    """What limits this process's memory, as far as the cgroups say.

    Attributes:
        limit: the job's memory limit in bytes, or None when there is none or it is unknown.
        source: where it was read, or why there is none (recorded with every capture's budget).
    """

    limit: int | None
    source: str


def _v2_unlimited(text: str) -> bool:
    return text == "max"


def _v1_unlimited(text: str) -> bool:
    return int(text) >= _V1_UNLIMITED


def job_memory_limit(proc_cgroup: Path = _PROC_CGROUP, cgroup_root: Path = _CGROUP_ROOT) -> JobMemory:
    """Return this process's job memory limit: the smallest limit on its cgroup or an ancestor.

    A Slurm allocation places the job's processes in a limited cgroup, so it counts. cgroup v2 (the unified ``0::``
    entry) is read as ``memory.max`` (``max`` is no limit); cgroup v1 (a ``memory`` controller entry, mounted at
    ``<root>/memory``) as ``memory.limit_in_bytes`` (values from 2**62 up are its "no limit"). When neither hierarchy
    exposes a memory limit file for the process, or a file cannot be read or parsed, the limit is unknown, and that is
    stated, never assumed absent.

    Args:
        proc_cgroup: the process's cgroup membership file.
        cgroup_root: the cgroup mount.
    """
    try:
        return _job_memory_limit(proc_cgroup, cgroup_root)
    except (OSError, ValueError) as err:
        return JobMemory(None, f"unknown: {type(err).__name__} reading the cgroup limits: {err}")


def _job_memory_limit(proc_cgroup: Path, cgroup_root: Path) -> JobMemory:
    if not proc_cgroup.is_file():
        return JobMemory(None, f"unknown: no {proc_cgroup}")
    parsed = [line.split(":", 2) for line in proc_cgroup.read_text().splitlines()]
    entries = [e for e in parsed if len(e) == _CGROUP_FIELDS]  # hierarchy-id:controllers:path
    v2 = [path for hierarchy, _, path in entries if hierarchy == "0"]
    v1 = [path for _, controllers, path in entries if "memory" in controllers.split(",")]
    for base, path, name, unlimited in (
        *((cgroup_root, p, "memory.max", _v2_unlimited) for p in v2),
        *((cgroup_root / "memory", p, "memory.limit_in_bytes", _v1_unlimited) for p in v1),
    ):
        found, limits = False, []
        node = base / path.lstrip("/")
        while True:
            limit_file = node / name
            if limit_file.is_file():
                found = True
                text = limit_file.read_text().strip()
                if not unlimited(text):
                    limits.append(int(text))
            if node == base or base not in node.parents:
                break
            node = node.parent
        if limits:
            return JobMemory(min(limits), f"job limit ({name})")
        if found:
            return JobMemory(None, f"no job limit ({name})")
    return JobMemory(None, "unknown: no cgroup memory limit file for this process")


def default_memory_budget() -> tuple[int, str]:
    """Return the default pass budget and where it came from: half the job's limit, or half the host's memory."""
    job = job_memory_limit()
    if job.limit is not None:
        return job.limit // 2, f"half the {job.source}"
    return host_memory_bytes() // 2, f"half the host memory ({job.source})"


def _rows(states: torch.Tensor, n: int, width: int) -> np.ndarray:
    """One layer's reduced states (``SpanStates.last`` or ``.mean``) as float32, checked for shape."""
    if tuple(states.shape) != (n, width):
        raise ValueError(f"backend returned states of shape {tuple(states.shape)}, expected {(n, width)}")
    out: np.ndarray = states.detach().to(torch.float32).cpu().numpy()
    return out
