"""Entry point: capture the receiver's states for one capture recipe and commit them."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from latent_interconnect.artifacts.store import ArtifactRef, read_handoff
from latent_interconnect.build.capture import CaptureSpec, capture_key, capture_states, default_memory_budget
from latent_interconnect.canonical import canonical_json
from latent_interconnect.data.items import ITEM_SET_KIND
from latent_interconnect.data.training_text import TRAINING_SET_KIND
from latent_interconnect.domains.base import DOMAIN_DATA_KIND
from latent_interconnect.receivers import open_capture, open_renderer
from latent_interconnect.run._main import (
    ExecutionTarget,
    Stage,
    StageResult,
    require_hardware,
    stage_main,
    visible_devices,
)
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "main"]


def _run(
    config: CaptureSpec,
    ws: Workspace,
    pass_objects: int | None = None,
    memory_bytes: int | None = None,
    origin_handoff: Path | None = None,
    **backend_options: str,
) -> StageResult:
    """Output: ``states`` (embeddings). Inputs: ``domain``, ``objects`` and a context recipe's ``training_set``.

    The receiver's ``capture`` backend is opened with ``receivers.open_capture``, the capture's batch plan (identity)
    and ``backend_options`` (execution options as text, e.g. the weights directory, bound by the backend's
    constructor; ``run._main`` refuses the plan as an option). ``pass_objects`` and ``memory_bytes`` are the capture's
    pass size and pass memory budget (``build.capture.capture_states``). All are execution, not identity: the record's
    ``execution`` is the backend's ``execution()`` plus ``pass_objects`` (``"all"`` for one pass) and the
    ``memory_bytes`` budget the capture ran with and its ``memory_source`` (``explicit``, or the resolved default
    of ``build.capture.default_memory_budget``: the job limit, or the host with the reason). ``origin_handoff`` is a
    JSON file declaring an origin handoff (``ArtifactStore.require_origin``), for work of other code; the record's
    ``execution`` holds the capture's origin record (``origin``: the origin its work was accepted under and every
    handoff).

    Raises:
        WrongHardwareError: if the visible devices are not the config's ``device_class`` (before the backend opens).
        OriginError: if the capture's work comes from another origin and no valid handoff covers it.
    """
    devices = visible_devices()
    require_hardware(devices, config.device_class)
    backend = open_capture(config.receiver, plan=config.plan, options=backend_options)
    renderer = open_renderer(config.receiver)
    budget, source = (memory_bytes, "explicit") if memory_bytes is not None else default_memory_budget()
    ref = capture_states(
        config,
        ws.artifacts,
        renderer,
        backend,
        devices,
        pass_objects=pass_objects,
        memory_bytes=budget,
        handoff=read_handoff(origin_handoff),
    )
    inputs = {
        "domain": ArtifactRef(config.domain.data, DOMAIN_DATA_KIND),
        "objects": ArtifactRef(config.objects, ITEM_SET_KIND),
    }
    if config.training_set is not None:
        inputs["training_set"] = ArtifactRef(config.training_set, TRAINING_SET_KIND)
    execution = {
        **backend.execution(),
        "pass_objects": "all" if pass_objects is None else str(pass_objects),
        "memory_bytes": str(budget),
        "memory_source": source,
        "origin": canonical_json(ws.artifacts.origin(capture_key(config))),
    }
    return StageResult(outputs={"states": ref}, inputs=inputs, devices=devices, execution=execution)


STAGE = Stage(
    "capture",
    CaptureSpec,
    _run,
    lambda c: ExecutionTarget(c.device_class, c.receiver, "capture"),
    outputs="states",
)
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.capture --config <yaml> [--pin NAME [--repin REASON]] [--option K=V]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
