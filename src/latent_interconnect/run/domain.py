"""Entry point: build a domain's data from its build spec and commit the ``domain_data`` artifact.

Domain data enters the pipeline only through this stage, so every ``result`` input that cites it has a producing record
(``Workspace.purpose``: an artifact without one counts as ``anchor``). The stage builds any domain whose data spec is
registered in ``latent_interconnect.domains.DOMAIN_DATA`` (``protein``, ``molecule``, or a third party's, loaded
through ``plugins``); it has no domain-specific code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import pyarrow
from pydantic import SerializeAsAny, model_validator

from latent_interconnect.config import StageConfig
from latent_interconnect.domains import DOMAIN_DATA, DomainDataSpec
from latent_interconnect.plugins import load_plugins
from latent_interconnect.run._main import Stage, StageResult, stage_main
from latent_interconnect.workspace import Workspace

__all__ = ["STAGE", "DomainBuild", "main"]


class DomainBuild(StageConfig):
    """One domain-data build: the domain and its data spec.

    Attributes:
        domain: the domain name; its spec class is ``DOMAIN_DATA``'s entry under this name.
        plugins: modules to import before resolving ``domain`` (third-party domains).
        spec: the data spec, validated as the domain's spec class (its identity is that class's identity), and
            serialized as that class, so a record's config holds the whole spec and reloads.
    """

    domain: str
    plugins: tuple[str, ...] = ()
    spec: SerializeAsAny[DomainDataSpec]

    @model_validator(mode="before")
    @classmethod
    def _typed_spec(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("spec"), dict):
            load_plugins(tuple(data.get("plugins", ())))
            return {**data, "spec": DOMAIN_DATA.get(str(data.get("domain"))).model_validate(data["spec"])}
        return data

    @model_validator(mode="after")
    def _check(self) -> DomainBuild:
        expected = DOMAIN_DATA.get(self.domain)
        if type(self.spec) is not expected:
            raise ValueError(f"domain {self.domain!r} takes a {expected.__name__}, not a {type(self.spec).__name__}")
        return self


def _run(config: DomainBuild, ws: Workspace) -> StageResult:
    """Output: ``domain_data`` (domain_data); inputs: the artifacts the spec reads (``DomainDataSpec.artifact_inputs``).

    Its other sources are files under the data root, hash-checked by the build.
    """
    ref = config.spec.build(ws.artifacts, ws.paths.data)
    return StageResult(
        outputs={"domain_data": ref},
        inputs=config.spec.artifact_inputs(ws.artifacts),
        packages={"pandas": pd.__version__, "pyarrow": pyarrow.__version__},
    )


STAGE = Stage("domain", DomainBuild, _run, outputs="domain_data")
"""This stage's name, config class and function."""


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m latent_interconnect.run.domain --config <yaml> [--pin NAME [--repin REASON]] [--option K=V]``."""
    return stage_main(argv, STAGE.name, STAGE.config, STAGE.run)


if __name__ == "__main__":
    raise SystemExit(main())
