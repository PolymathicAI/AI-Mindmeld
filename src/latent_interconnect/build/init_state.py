"""The initial state: a connector config plus its ridge solutions → the committed step-0 checkpoint.

This is the one initial-state procedure: the ``init`` stage runs it on its own (train-free connectors, the
vertical slice's stage 1), and the trainer calls :func:`initial_connector` and commits the result as its step 0: the
connector is :meth:`~latent_interconnect.writes.connector.Connector.from_ridge` on the cited ridge artifacts.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from latent_interconnect.artifacts.store import ArtifactRef, ArtifactStore
from latent_interconnect.build.ridge import load_ridge
from latent_interconnect.config import ArtifactDigest, Config, StageConfig, added_field
from latent_interconnect.encoders.recipes import EncoderRecipe
from latent_interconnect.receivers.base import ReceiverSpec
from latent_interconnect.writes.checkpoint import commit_checkpoint
from latent_interconnect.writes.connector import Connector, ConnectorSpec, expected_entries
from latent_interconnect.writes.inputs import InputSpec, check_read_layers

__all__ = ["InitSpec", "SourceFit", "commit_initial_state", "initial_connector"]

RESIDUES = "residues"
"""The recipe field a :class:`SourceFit` may name in place of an ``options`` key: per-residue storage turned on."""


class SourceFit(Config):
    """A start kept on a recipe that differs from the one its ridges were fit on by one declared option.

    A ridge records the digest of the recipe its fit read, and :meth:`RidgeSolution.require_fit` refuses another.
    A start kept without a refit for inputs whose recipe adds one option (the Uni-Mol2 conformer
    retry, ``extractors._conformer_retry``) declares here what its ridges were really fit on: the ridges are checked
    against :attr:`recipe` instead, every other check stays, and the runtime recipe must be :attr:`recipe` plus
    :attr:`option` and nothing else. Only the ridges named in :attr:`ridges` are covered; any other mismatch is
    refused as before. The ridges, their arrays and so the step-0 numbers are those of a start on :attr:`recipe`; the
    step-0 checkpoint records the runtime recipe, which its inputs are checked against.

    :attr:`option` ``residues`` names the recipe field instead of an ``options`` key: the runtime recipe turns on
    per-residue storage (``residues: true``) beside the same pooled vectors, as the molecule tr arm's per-token ChemFM
    rows (``extractors._molecules.token_rows``, whose mean is the pooled vector the ridges were fit on).

    Attributes:
        recipe: the encoder recipe the ridges were fit on (its digest is the one each ridge records).
        option: the one ``options`` key the runtime recipe adds to :attr:`recipe`, or ``residues`` (the field turned
            on).
        ridges: the exact ridge-init artifacts this declaration covers (sha256), each one the spec cites.
    """

    recipe: EncoderRecipe
    option: str = Field(min_length=1)
    ridges: tuple[ArtifactDigest, ...] = Field(min_length=1)

    def require_covers(self, ridges: dict[str, ArtifactDigest], runtime: EncoderRecipe) -> None:
        """Refuse a spec this declaration does not cover.

        Raises:
            ValueError: if the spec cites a ridge not named here, names a ridge it does not cite, or the runtime
                recipe is not :attr:`recipe` plus :attr:`option` alone (another option, model, revision, read layers,
                pooling or anything else differs, or the option is already in :attr:`recipe`).
        """
        cited = set(ridges.values())
        if cited != set(self.ridges):
            raise ValueError(
                f"source_fit covers ridges {sorted(d[:12] for d in self.ridges)}, the spec cites "
                f"{sorted(d[:12] for d in cited)}"
            )
        source, used = self.recipe.model_dump(mode="json"), runtime.model_dump(mode="json")
        if self.option == RESIDUES:
            if source[RESIDUES] or not used[RESIDUES]:
                raise ValueError("source_fit: the runtime recipe must turn on residues; the source recipe has them off")
            used[RESIDUES] = False
        elif self.option in source["options"] or self.option not in used["options"]:
            raise ValueError(f"source_fit: the runtime recipe must add option {self.option!r} to the source recipe")
        else:
            used["options"] = {k: v for k, v in used["options"].items() if k != self.option}
        if used != source:
            differ = sorted(k for k in source if source[k] != used.get(k))
            raise ValueError(
                f"source_fit: the runtime recipe differs from the source recipe beyond {self.option!r}: {differ}"
            )


class InitSpec(StageConfig):
    """What the initial state depends on.

    Attributes:
        connector: the connector config (embedded).
        receiver: the receiver config (embedded; its depth and width size the maps).
        encoder: the encoder recipe the connector reads (embedded; the connector's input is checked against it at
            load, ``writes.inputs.check_read_layers``).
        ridge: target name → ridge-init artifact digest, one per ridge target of the tokens.
        embeddings: always empty (the standardization comes from the ridge); the field is part of every stored init
            identity.
        append_encoder: the recipe of the connector's appended extraction (``InputSpec.append``), checked
            against every ridge's recorded digest and recorded in the step-0 checkpoint as the trainer's is; required
            by :func:`commit_initial_state` when the connector appends (None in ``RunConfig.init_spec``, where the
            trainer takes it from the run's embeddings; added field).
        source_fit: the recipe the ridges were fit on, when it is not ``encoder`` (:class:`SourceFit`; added field,
            None: the ridges were fit on ``encoder``).
    """

    connector: ConnectorSpec
    receiver: ReceiverSpec
    encoder: EncoderRecipe
    ridge: dict[str, ArtifactDigest] = Field(default_factory=dict)
    embeddings: tuple[()] = ()
    append_encoder: EncoderRecipe | None = added_field(default=None)
    source_fit: SourceFit | None = added_field(default=None)

    @model_validator(mode="after")
    def _check(self) -> InitSpec:
        if set(self.ridge) != self.connector.ridge_targets():
            raise ValueError(
                f"ridge covers {sorted(self.ridge)}, the tokens cite {sorted(self.connector.ridge_targets())}"
            )
        if self.connector.receiver != self.receiver.name:
            raise ValueError(f"connector is for {self.connector.receiver!r}, not {self.receiver.name!r}")
        # the one place a connector meets its recipe for step 0 (init configs and, through init_spec, run configs): an
        # init with pending or wrong read layers, or pending write tokens or layers, is refused at load, not at step 0
        # in a GPU job; a connector config alone may stay pending
        check_read_layers(self.connector.input, self.encoder)
        if (appended := self.append_encoder) is not None:
            if (a := self.connector.input.append) is None:
                raise ValueError(f"connector {self.connector.name!r} appends no extraction; drop append_encoder")
            check_read_layers(InputSpec(recipe=a.recipe, layers=(a.layer,)), appended)
        self.connector.ready_tokens()
        if self.source_fit is not None:
            self.source_fit.require_covers(self.ridge, self.encoder)
        return self


def initial_connector(spec: InitSpec, store: ArtifactStore) -> Connector:
    """Build the step-0 connector (on CPU).

    The spec's validation already applied the read-layer rule (``writes.inputs.check_read_layers``) and refused
    pending write tokens and layers (``ConnectorSpec.ready_tokens``).

    Raises:
        ValueError: if a ridge fit read another recipe than the connector's input
            or another appended extraction (its recipe's content is checked where its artifacts are loaded:
            ``Trainer.from_config``, ``evaluation.scan_states``),
            lacks a layer the connector writes for that target, or its width is not the receiver's.
    """
    c = spec.connector
    ridge = {t: load_ridge(store, ArtifactRef(d, "ridge_init")) for t, d in sorted(spec.ridge.items())}
    appended = None if spec.append_encoder is None else spec.append_encoder.digest()
    fit_on = spec.encoder if spec.source_fit is None else spec.source_fit.recipe  # SourceFit: declared and covered
    for t, r in ridge.items():
        if r.target != t:
            raise ValueError(f"the ridge given for target {t!r} was fit to target {r.target!r}")
        r.require_fit(
            "connector",
            input=c.input,
            recipe_digest=fit_on.digest(),
            receiver={"model_id": spec.receiver.model_id, "revision": spec.receiver.revision},
            append_digest=appended,
        )
    tokens = c.ready_tokens()
    for e in expected_entries(c, spec.receiver.depth):
        target = tokens[e.token].target
        if e.layer not in ridge[target].layers:
            raise ValueError(
                f"ridge {target!r} has no layer {e.layer} (entry {e.key()}); it has {sorted(ridge[target].layers)}"
            )
    connector = Connector.from_ridge(c, spec.receiver.depth, ridge)
    if connector.d_model != spec.receiver.d_model:
        raise ValueError(f"the ridge targets are {connector.d_model}-wide, the receiver is {spec.receiver.d_model}")
    return connector


def commit_initial_state(spec: InitSpec, store: ArtifactStore) -> ArtifactRef:
    """Build the step-0 connector and commit it at step 0 (``writes.checkpoint``) with its encoder recipes.

    The appended extraction's recipe is recorded as the trainer's step 0 records it, so a step-0 suite checks the
    appended inputs against it.

    Raises:
        ValueError: as :func:`initial_connector`, or if the connector appends an extraction the spec gives no
            ``append_encoder`` for.
    """
    if spec.connector.input.append is not None and spec.append_encoder is None:
        raise ValueError(f"connector {spec.connector.name!r} appends an extraction: give its append_encoder")
    connector = initial_connector(spec, store)
    return commit_checkpoint(connector, 0, store, encoder=spec.encoder, append_encoder=spec.append_encoder)
