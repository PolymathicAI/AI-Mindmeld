"""Receivers: the frozen language model side.

- ``base``: :class:`ReceiverSpec`, the capability protocols and the backend registry.
- ``render``: the one renderer.
- ``kernel``: per-layer writes and the one kernel that applies them.
- ``dictionary``: a frozen sparse dictionary and the one opt-in state edit after a write.
- ``gradients``: the runtime gradient check ``open_training`` runs on every training backend.
- ``conformance``: what a training backend must satisfy in its tests (gradient and update parity, finite
  differences).
- ``hf``: the generic Hugging Face backend (registers ``"hf"``).
- ``progress``: periodic progress lines on stderr during long backend calls (logging only).
- ``kimi_k26``: Kimi K2.6 plugin backends (imported only through ``ReceiverSpec.plugins``).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, cast

from latent_interconnect.config import OptionError, bind_options
from latent_interconnect.plugins import load_plugins
from latent_interconnect.receivers.base import (
    IDENTITY_PARAMETERS,
    PREPARATIONS,
    RECEIVER_BACKENDS,
    Capability,
    CaptureBackend,
    GenerationBackend,
    GenerationPlan,
    ReceiverSpec,
    ScoringBackend,
    TrainingBackend,
)
from latent_interconnect.receivers.dictionary import StateEdit
from latent_interconnect.receivers.gradients import check_gradients
from latent_interconnect.receivers.hf import load_tokenizer  # importing hf registers the generic backend
from latent_interconnect.receivers.kernel import BatchPlan
from latent_interconnect.receivers.render import Renderer, Tokenizer

__all__ = ["backend_class", "open_capture", "open_generation", "open_renderer", "open_scoring", "open_training"]


def _open(
    spec: ReceiverSpec,
    capability: Capability,
    protocol: type[Any],
    *,
    identity: Mapping[str, Any],
    options: Mapping[str, str] | None,
    construction: dict[str, Any],
) -> Any:
    """Construct a backend: the one construction point, where command-line options are bound to its constructor.

    ``identity`` holds the identity parameters given (``base.IDENTITY_PARAMETERS``: ``plan`` or ``generation_plan``,
    and a scoring backend's ``edit``); a None value leaves the backend's default. A state edit is given only to a
    backend whose constructor takes one: any other refuses it here, before anything loads, and so does every opener
    but ``open_scoring`` (an ``edit`` among ``construction``: only a scoring run declares one). The spec's declared
    training preparation is resolved here, for every capability, once its plugins are loaded: a name no plugin
    registers is refused at open (``UnknownNameError``), not first noticed by a backend that applies it.

    Raises:
        ValueError: if a state edit is given to a backend that applies none, or to an opener other than
            ``open_scoring``; if a plan with ``pad_every_batch`` is given to a backend that does not right-pad its
            batches (only ``hf`` does).
    """
    if "edit" in construction:
        raise ValueError(
            f"a {capability!r} backend takes no state edit (receivers.dictionary.StateEdit): only a scoring run "
            "declares one (SuiteRunConfig.edit, given through open_scoring)"
        )
    cls = backend_class(spec, capability)
    if identity.get("edit") is not None and "edit" not in inspect.signature(cls).parameters:
        raise ValueError(
            f"backend {spec.backends[capability]!r} applies no state edit (receivers.dictionary.StateEdit); only a "
            "backend whose constructor takes `edit` does (hf)"
        )
    plan = identity.get("plan")
    if plan is not None and plan.pad_every_batch and not getattr(cls, "pads_every_batch", False):
        raise ValueError(
            f"backend {spec.backends[capability]!r} does not serve a plan with pad_every_batch (BatchPlan); only a "
            "backend that right-pads its batches does (hf)"
        )
    if spec.training_preparation is not None:
        PREPARATIONS.get(spec.training_preparation)
    typed, rest = bind_options(cls, options or {}, refused=IDENTITY_PARAMETERS)
    if rest:  # a backend takes no open-ended options
        raise OptionError(f"backend {spec.backends[capability]!r} has no parameters {sorted(rest)}")
    both = sorted(set(typed) & set(construction))
    if both:
        raise OptionError(f"backend options {both} are given both as text and as values")
    kwargs = {**construction, **typed} | {k: v for k, v in identity.items() if v is not None}
    backend = cls(spec, **kwargs)
    if not isinstance(backend, protocol):
        raise TypeError(f"backend {spec.backends[capability]!r} does not implement {protocol.__name__}")
    return backend


def backend_class(spec: ReceiverSpec, capability: Capability) -> type[Any]:
    """Return the registered backend class that serves ``capability`` for the receiver, without constructing it.

    The receiver's plugins are loaded first (``ReceiverSpec.plugins``). A stage reads class-level facts from it before
    anything loads, e.g. whether the backend reads write banks (its ``reads_banks``, ``base.ScoringBackend``).

    Raises:
        KeyError: if the receiver names no backend for the capability.
        UnknownNameError: if the named backend is not registered.
    """
    load_plugins(spec.plugins)
    return RECEIVER_BACKENDS.get(spec.backends[capability])


def open_scoring(
    spec: ReceiverSpec,
    *,
    plan: BatchPlan | None = None,
    edit: StateEdit | None = None,
    options: Mapping[str, str] | None = None,
    **construction: Any,
) -> ScoringBackend:
    """Construct the receiver's ``score`` backend.

    Args:
        spec: the receiver config.
        plan: the batch plan (identity: the config's, passed explicitly; ``base.IDENTITY_PARAMETERS``); the backend's
            default when None.
        edit: a state edit (identity: the config's, ``SuiteRunConfig.edit``), or None for none, in which case
            the backend is constructed exactly as without this parameter.
        options: execution options as command-line text, bound to the backend constructor's parameters
            (``config.bind_options``: typed, identity parameters refused).
        construction: constructor arguments as values (a built model, a dtype), from code.

    Raises:
        ValueError: if ``edit`` is given and the backend applies no state edit.
    """
    identity = {"plan": plan, "edit": edit}
    return cast(
        ScoringBackend,
        _open(spec, "score", ScoringBackend, identity=identity, options=options, construction=construction),
    )


def open_training(
    spec: ReceiverSpec, *, plan: BatchPlan | None = None, options: Mapping[str, str] | None = None, **construction: Any
) -> TrainingBackend:
    """Construct the receiver's ``train`` backend (which also scores, for in-job evaluation) and check its gradients.

    Arguments as :func:`open_scoring`. The one place the runtime gradient check runs
    (:func:`~latent_interconnect.receivers.gradients.check_gradients`, once, before any training step): the probe
    loss must be finite and give every write, at every layer, a finite non-zero gradient. That the gradient is also
    correct is established by the backend's preparation and its parity tests (``receivers.conformance``), not here.
    A training backend constructed directly, not through this function, is never checked.

    Raises:
        ValueError: if the runtime gradient check fails.
    """
    backend = cast(
        TrainingBackend,
        _open(spec, "train", TrainingBackend, identity={"plan": plan}, options=options, construction=construction),
    )
    check_gradients(backend, spec)
    return backend


def open_generation(
    spec: ReceiverSpec,
    *,
    generation_plan: GenerationPlan | None = None,
    options: Mapping[str, str] | None = None,
    **construction: Any,
) -> GenerationBackend:
    """Construct the receiver's ``generate`` backend.

    Arguments as :func:`open_scoring`, with ``generation_plan`` (identity: the config's, passed explicitly;
    ``base.IDENTITY_PARAMETERS``; the backend's default when None) in place of ``plan``.
    """
    return cast(
        GenerationBackend,
        _open(
            spec,
            "generate",
            GenerationBackend,
            identity={"generation_plan": generation_plan},
            options=options,
            construction=construction,
        ),
    )


def open_capture(
    spec: ReceiverSpec, *, plan: BatchPlan | None = None, options: Mapping[str, str] | None = None, **construction: Any
) -> CaptureBackend:
    """Construct the receiver's ``capture`` backend (arguments as :func:`open_scoring`)."""
    return cast(
        CaptureBackend,
        _open(spec, "capture", CaptureBackend, identity={"plan": plan}, options=options, construction=construction),
    )


def open_renderer(spec: ReceiverSpec, tokenizer: Tokenizer | None = None) -> Renderer:
    """Return the receiver's renderer, loading its tokenizer at ``spec.revision`` unless one is given (fakes do).

    Every caller (sampler, capture, suites, backends) gets the renderer this way, so they render identically.

    Raises:
        ValueError: if the placeholder does not encode to exactly one token id.
    """
    tok = load_tokenizer(spec) if tokenizer is None else tokenizer
    return Renderer(tok, spec.placeholder, spec.template_kwargs, spec.thinking_request)
