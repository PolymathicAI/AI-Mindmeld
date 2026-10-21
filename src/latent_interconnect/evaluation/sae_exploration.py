"""The ``sae_exploration`` analysis kind: one frozen sparse dictionary applied to a capture's own text states.

Exploratory and diagnostic, never paper: it asks the receiver nothing and trains nothing. It reads the
receiver's **own** states of a committed capture (the stored ``last`` read-token state of each object, e.g. the last
name token of ``name_described``; no new capture, no forward pass, no write), applies one published dictionary
(``receivers.dictionary``) and reports numbers only: no verdict, threshold or test.

One input record (role ``capture``): a ``capture`` stage record. Its config gives the receiver, the domain (labels,
scope masks) and the recipe; its ``embeddings`` output holds the states in ``hidden_states`` numbering, so *receiver
layer L* is the input of decoder layer L, which a dictionary published for the output of layer L − 1 reads
(``receivers.dictionary.load_dictionary`` checks that from the dictionary's own config; nothing is remapped here).

What it computes (params :class:`SaeExploration`):

- **Reconstruction**, at every layer of ``reconstruction_layers`` (:class:`LayerReconstruction`): the same dictionary,
  loaded for its declared site ``layer``, is applied unchanged to the capture's states of each listed layer. A list
  with the site's neighbours is the off-by-one diagnostic: the record shows the three errors side by side and the
  dictionary's identity apart from the *input* layer; a better neighbour is simply visible, nothing selects a layer.
  Per layer: the headline fraction unexplained ``Σ|h − ĥ|² / Σ|h − mean(h)|²`` with ``ĥ = decode(encode(h))`` and the
  mean over the same objects at that layer (``receivers.dictionary.ReconstructionCheck``; the mean state alone scores
  1, an exact reconstruction 0); the secondary norm-normalized aggregate ``Σ|h − ĥ|² / Σ|h|²`` beside what the mean
  state alone scores under it; the per-object relative error ``|h − ĥ| / |h|`` and active-feature count with their
  distributions. The three summaries carry the package's bootstrap interval over the analysis' resampling units
  (``stats.state_error``: mean and denominator recomputed in every draw).
- **Feature tables** at the site ``layer`` only, when ``tasks`` are given (``evaluation.sae_tables``): support per
  Yes/No task before any ranking, the sparse activations, strength and separation per feature, compact lists for the
  tasks with enough support.
- **Descriptions**, when the params name a source (``evaluation.sae_descriptions``): refused unless the source names
  exactly this dictionary; joined to the listed features only. Without a source the result is ``uninterpreted``.
  A joined result states, in the record (``Descriptions.caveat``) and in its text tables, that the descriptions are
  machine-written and that their link to the dictionary is the provider's metadata, not verified against the weights.
- **Token provenance**, when asked (``evaluation.capture_tokens``): the read positions re-derived on the CPU.

What it does not establish: the objects are training objects described by the training set's own text (in-sample), the
last name token also carries spelling, every list is selected on the objects it is computed on, a description is a
machine-written hypothesis linked by metadata, and a strong or separating feature is not shown to be used by the
receiver for anything. A dictionary's feature indices and raw scales are not comparable with another dictionary's.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import ClassVar, Literal

import numpy as np
import torch
from pydantic import Field, model_validator

from latent_interconnect.artifacts.records import Record, record_output
from latent_interconnect.artifacts.store import ArtifactStore
from latent_interconnect.build.capture import CaptureSpec
from latent_interconnect.config import ArtifactDigest, Config
from latent_interconnect.data.items import ItemSet
from latent_interconnect.domains.base import open_domain
from latent_interconnect.domains.tables import read_source
from latent_interconnect.encoders.store import EMBEDDINGS_KIND, EmbeddingReader
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS
from latent_interconnect.evaluation.capture_tokens import TokenProvenance, token_provenance
from latent_interconnect.evaluation.contrasts import unit_columns
from latent_interconnect.evaluation.results import FINITE, AnalysisResult, Estimate
from latent_interconnect.evaluation.sae_descriptions import (
    Descriptions,
    DescriptionSource,
    read_descriptions,
    require_exact,
)
from latent_interconnect.evaluation.sae_tables import (
    FeatureTables,
    Reporting,
    feature_tables,
    listed_features,
    task_cohorts,
)
from latent_interconnect.paths import Paths
from latent_interconnect.receivers import open_renderer
from latent_interconnect.receivers.dictionary import (
    Dictionary,
    DictionarySpec,
    ReconstructionCheck,
    StateCodes,
    encode_states,
    load_dictionary,
    read_config,
)
from latent_interconnect.stats.bootstrap import Bootstrap, ResamplingUnits, unit_index
from latent_interconnect.stats.state_error import StateErrors

__all__ = [
    "DESCRIPTION_COLUMN",
    "NORM_ROWS",
    "QUANTILES",
    "TABLES_FILE",
    "Distribution",
    "LayerReconstruction",
    "SaeExploration",
    "SaeExplorationKind",
    "SaeExplorationResult",
]

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
"""The quantiles of a per-object distribution a result reports, beside its least, greatest and mean values."""
NORM_ROWS = 4096
"""The decoder rows whose norms are computed at once (a bounded copy, whatever the number of active features)."""
TABLES_FILE = "tables.md"
"""The readable companion of a result: its reconstruction, support and compact lists as text tables."""
DESCRIPTION_COLUMN = "description (machine-written; link to the dictionary not verified against its weights)"
"""The heading of a joined description column, so that each table says what ``Descriptions.caveat`` says once."""


class SaeExploration(Config):
    """Parameters of the ``sae_exploration`` kind.

    Attributes:
        item_set: the objects (a single-role item set, each object held by the capture); its resampling-unit column
            (``AnalysisSpec.units``, one factor) is the objects' groups.
        layer: the dictionary's site: the receiver layer L whose input it was published for (its own config must name
            the output of decoder layer L − 1).
        dictionary: the dictionary.
        reconstruction_layers: the capture layers whose states the dictionary reconstructs; it holds ``layer``. With
            the neighbours L − 1 and L + 1 it is the off-by-one diagnostic.
        tasks: the Yes/No tasks of the feature tables (at ``layer``); empty: reconstruction only.
        reporting: the tables' recipe; given exactly when ``tasks`` are.
        descriptions: the published descriptions of exactly this dictionary's features, or None (uninterpreted).
            Refused for any other dictionary, and without ``tasks``.
        token_provenance: whether the capture's read positions are derived again on the CPU and reported.
    """

    item_set: ArtifactDigest
    layer: int = Field(ge=1)
    dictionary: DictionarySpec
    reconstruction_layers: tuple[int, ...] = Field(min_length=1)
    tasks: tuple[str, ...]
    reporting: Reporting | None = None
    descriptions: DescriptionSource | None = None
    token_provenance: bool

    @model_validator(mode="after")
    def _check(self) -> SaeExploration:
        layers = self.reconstruction_layers
        if len(set(layers)) != len(layers) or self.layer not in layers:
            raise ValueError(f"reconstruction_layers {list(layers)}: distinct layers that hold the site {self.layer}")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("the tasks repeat")
        if bool(self.tasks) != (self.reporting is not None):
            raise ValueError("reporting (the tables' recipe) is given exactly when tasks are")
        if self.descriptions is not None:
            if not self.tasks:
                raise ValueError("descriptions are joined to listed features: without tasks there is no list")
            require_exact(self.descriptions, self.dictionary)
        return self


class Distribution(Config):
    """A per-object quantity's distribution over the objects.

    Attributes:
        least: its least value.
        quantiles: its values at :data:`QUANTILES`, in that order.
        greatest: its greatest value.
        mean: its mean.
    """

    model_config = FINITE

    least: float
    quantiles: tuple[float, ...]
    greatest: float
    mean: float

    @classmethod
    def of(cls, values: np.ndarray) -> Distribution:
        """Return the distribution of ``values``."""
        v = np.asarray(values, dtype=np.float64)
        at = np.quantile(v, QUANTILES)
        return cls(
            least=float(v.min()), quantiles=tuple(float(q) for q in at), greatest=float(v.max()), mean=float(v.mean())
        )


class LayerReconstruction(Config):
    """One dictionary's reconstruction of the objects' states at one capture layer (the *input* layer).

    Attributes:
        n: the objects.
        unexplained: the fraction unexplained, with its interval (the headline; module docstring).
        norm_normalized: ``Σ|h − ĥ|² / Σ|h|²``, with its interval (secondary; not a fraction of variance).
        relative_error_mean: the mean per-object relative error, with its interval (secondary).
        mean_state_norm_normalized: what the mean state alone scores under ``norm_normalized``,
            ``Σ|h − mean(h)|² / Σ|h|²`` (under ``unexplained`` it scores 1 by definition).
        centered_total: ``Σ|h − mean(h)|²``, the headline's denominator.
        norm_total: ``Σ|h|²``, the secondary aggregate's denominator.
        distinct_states: the number of distinct state vectors among the objects.
        inactive_states: the objects on which no feature is active (their reconstruction is the decoder bias).
        relative_error: the distribution of the per-object relative error.
        active: the distribution of the per-object number of active features (to set beside the declared ``l0``).
        object_relative_error: each object's relative error, in the result's object order.
        object_active: each object's number of active features.
    """

    model_config = FINITE

    n: int = Field(ge=2)
    unexplained: Estimate
    norm_normalized: Estimate
    relative_error_mean: Estimate
    mean_state_norm_normalized: float
    centered_total: float
    norm_total: float
    distinct_states: int = Field(ge=1)
    inactive_states: int = Field(ge=0)
    relative_error: Distribution
    active: Distribution
    object_relative_error: tuple[float, ...]
    object_active: tuple[int, ...]


class SaeExplorationResult(AnalysisResult):
    """Result of ``sae_exploration`` (exploratory; module docstring). ``units`` is the one group factor.

    Attributes:
        receiver: the receiver's model id (the capture's).
        receiver_revision: its revision.
        recipe: the capture recipe whose read token the states are.
        target: which read-token state the capture stored as its pooled state (``last``).
        states: the digest of the capture's states artifact.
        item_set: the digest of the item set.
        objects: the objects, in the order of every per-object column.
        groups: each object's group (the resampling unit).
        layer: the dictionary's declared site (receiver layer).
        dictionary: the dictionary's identity (repository, revision, folder, parameter and config sha256).
        hook: the hook its config names.
        width: its number of features.
        declared_l0: the ``l0`` its config declares (a published average, distinct from the observed counts), or None.
        reconstruction: input layer → the reconstruction of that layer's states.
        tables: the feature tables at ``layer``, or None (reconstruction only).
        interpretation: ``exact_descriptions`` when a source naming exactly this dictionary was joined, else
            ``uninterpreted`` (numbers only).
        descriptions: the joined descriptions, or None; they hold the sentence on what the join rests on (``caveat``).
        tokens: the re-derived token provenance, or None.
    """

    model_config = FINITE

    receiver: str
    receiver_revision: str
    recipe: str
    target: str
    states: str
    item_set: str
    objects: tuple[str, ...]
    groups: tuple[str, ...]
    layer: int
    dictionary: DictionarySpec
    hook: str
    width: int
    declared_l0: float | None
    reconstruction: dict[int, LayerReconstruction]
    tables: FeatureTables | None = None
    interpretation: Literal["exact_descriptions", "uninterpreted"]
    descriptions: Descriptions | None = None
    tokens: TokenProvenance | None = None

    def companion_files(self) -> dict[str, str]:
        """Return :data:`TABLES_FILE`: the result's tables as text, for a person (nothing reads it back)."""
        return {TABLES_FILE: _tables_text(self)}


@ANALYSIS_KINDS.register("sae_exploration")
class SaeExplorationKind:
    """One dictionary on one capture's own states (params :class:`SaeExploration`)."""

    Params: ClassVar[type[Config]] = SaeExploration
    Result: ClassVar[type[AnalysisResult]] = SaeExplorationResult
    packages: ClassVar[tuple[str, ...]] = ("torch", "safetensors", "huggingface_hub", "transformers", "tokenizers")
    """The libraries it computes with beyond the analysis stage's own; their loaded versions are recorded."""

    def run(
        self,
        params: SaeExploration,
        units: ResamplingUnits,
        inputs: Mapping[str, Record],
        artifacts: ArtifactStore,
        items: Callable[[str], ItemSet],
    ) -> SaeExplorationResult:
        """Return the reconstruction of every listed layer and, with tasks, the site's feature tables.

        Raises:
            ValueError: if the inputs are not one ``capture`` record under the role ``capture``, the capture did not
                store the last read-token state, the units are not one factor, an object is not in the capture or
                has a zero state, or as ``receivers.dictionary.load_dictionary`` and ``sae_tables.task_cohorts``.
            KeyError: if a task is not in the domain, or the unit factor is not a column of the item set.
        """
        if set(inputs) != {"capture"} or inputs["capture"].kind != "capture":
            raise ValueError(f"sae_exploration takes one input, 'capture' (a capture record), not {sorted(inputs)}")
        if len(units.factors) != 1:
            raise ValueError(f"sae_exploration resamples one group factor, not {list(units.factors)}")
        spec = CaptureSpec.model_validate(inputs["capture"].config)
        states_ref = record_output(inputs["capture"], EMBEDDINGS_KIND)
        reader = EmbeddingReader(artifacts, states_ref)
        if reader.meta.get("target") != "last":
            raise ValueError(f"the capture stored its {reader.meta.get('target')!r} state, not the last read token's")
        population = items(params.item_set)
        objects = population.object_ids
        missing = [o for o in objects if o not in reader]
        if missing:
            raise ValueError(f"{len(missing)} of {len(objects)} objects are not in the capture, e.g. {missing[0]!r}")
        factor = units.factors[0]
        groups = unit_columns(population, units.factors)[factor]
        index = unit_index(groups)
        unit = np.array([index[g] for g in groups], dtype=np.int64)
        receiver = spec.receiver
        config = read_config(params.dictionary)
        dictionary = load_dictionary(
            params.dictionary, model_id=receiver.model_id, layer=params.layer, d_model=receiver.d_model
        )
        bootstrap = Bootstrap({factor: len(index)})
        reconstruction, site = {}, None
        for layer in params.reconstruction_layers:
            states = reader.pooled(objects, layer)
            codes = encode_states(dictionary, torch.from_numpy(states))
            reconstruction[layer] = _reconstruction(
                states, codes, StateErrors(states, codes.error_sq.numpy(), unit, len(index), factor), bootstrap
            )
            if layer == params.layer:
                site = codes
        tables = descriptions = None
        norms = _decoder_norms(dictionary, site) if params.reporting is not None and site is not None else {}
        del dictionary  # the parameters are the job's memory: the tables below are made without them
        if params.reporting is not None and site is not None:
            support, cohorts = task_cohorts(
                open_domain(spec.domain, artifacts), objects, unit, params.tasks, params.reporting
            )
            tables = feature_tables(
                site.rows.numpy(),
                site.features.numpy(),
                site.values.numpy(),
                decoder_norms=norms,
                groups=unit,
                support=support,
                cohorts=cohorts,
                reporting=params.reporting,
            )
        if tables is not None and params.descriptions is not None:
            path = read_source(Paths.from_env().data, params.descriptions.file)
            descriptions = read_descriptions(path, params.descriptions, int(config["width"]), listed_features(tables))
        tokens = None
        if params.token_provenance:
            tokens = token_provenance(spec, artifacts, open_renderer(receiver), reader, objects)
        l0 = config.get("l0")
        return SaeExplorationResult(
            units=units,
            receiver=receiver.model_id,
            receiver_revision=receiver.revision,
            recipe=spec.recipe,
            target="last",
            states=states_ref.sha256,
            item_set=params.item_set,
            objects=objects,
            groups=groups,
            layer=params.layer,
            dictionary=params.dictionary,
            hook=str(config["hf_hook_point_in"]),
            width=int(config["width"]),
            declared_l0=None if l0 is None else float(l0),
            reconstruction=reconstruction,
            tables=tables,
            interpretation="uninterpreted" if descriptions is None else "exact_descriptions",
            descriptions=descriptions,
            tokens=tokens,
        )


def _reconstruction(
    states: np.ndarray, codes: StateCodes, errors: StateErrors, bootstrap: Bootstrap
) -> LayerReconstruction:
    """Return one layer's reconstruction summary: the headline from the one check, the intervals from the bootstrap."""
    check = ReconstructionCheck.of(torch.from_numpy(states), codes)
    whole = errors.all_units()
    relative = (codes.error_sq / codes.norm_sq).sqrt().numpy()
    active = codes.active.numpy()
    return LayerReconstruction(
        n=check.n,
        unexplained=Estimate.of(bootstrap.interval(errors.unexplained, check.unexplained)),
        norm_normalized=Estimate.of(bootstrap.interval(errors.norm_normalized, errors.norm_normalized(whole))),
        relative_error_mean=Estimate.of(bootstrap.interval(errors.relative, check.relative_error)),
        mean_state_norm_normalized=errors.mean_baseline(whole),
        centered_total=errors.centered_total(whole),
        norm_total=float(codes.norm_sq.sum()),
        distinct_states=int(np.unique(states, axis=0).shape[0]),
        inactive_states=int((active == 0).sum()),
        relative_error=Distribution.of(relative),
        active=Distribution.of(active),
        object_relative_error=tuple(float(v) for v in relative),
        object_active=tuple(int(v) for v in active),
    )


def _decoder_norms(dictionary: Dictionary, codes: StateCodes) -> dict[int, float]:
    """Return active feature → the norm of its decoder row, read :data:`NORM_ROWS` rows at a time."""
    active = torch.unique(codes.features)
    norms: dict[int, float] = {}
    for start in range(0, int(active.shape[0]), NORM_ROWS):
        ids = active[start : start + NORM_ROWS]
        values = dictionary.w_dec[ids].double().norm(dim=1)
        norms.update((int(f), float(v)) for f, v in zip(ids, values, strict=True))
    return norms


def _row(cells: list[object]) -> str:
    return "| " + " | ".join(f"{c:.4g}" if isinstance(c, float) else str(c) for c in cells) + " |"


def _table(header: list[str], rows: list[list[object]]) -> list[str]:
    return [_row(list(header)), _row(["---"] * len(header)), *(_row(r) for r in rows), ""]


def _tables_text(result: SaeExplorationResult) -> str:
    """Return the result as text tables: the reconstruction per input layer, then support and the compact lists."""
    spec = result.dictionary
    lines = [
        f"# sae_exploration (exploratory, diagnostic; in-sample objects): {spec.repository} {spec.folder}",
        "",
        f"Receiver {result.receiver}; declared site: receiver layer {result.layer} (hook {result.hook}); width "
        f"{result.width}; declared l0 {result.declared_l0}; {len(result.objects)} objects; recipe {result.recipe} "
        f"({result.target} read token); interpretation: {result.interpretation}.",
        "",
    ]
    if result.descriptions is not None:
        lines += [result.descriptions.caveat, ""]
    lines += ["## Reconstruction per input layer (the same dictionary; intervals: bootstrap over groups)", ""]
    lines += _table(
        ["input layer", "n", "fraction unexplained", "lo", "hi", "norm-normalized", "mean state alone",
         "relative error (mean)", "relative error (median)", "active (mean)", "active (median)", "no active feature"],
        [
            [layer, r.n, r.unexplained.estimate, r.unexplained.lo, r.unexplained.hi, r.norm_normalized.estimate,
             r.mean_state_norm_normalized, r.relative_error.mean, r.relative_error.quantiles[QUANTILES.index(0.5)],
             r.active.mean, r.active.quantiles[QUANTILES.index(0.5)], r.inactive_states]
            for layer, r in sorted(result.reconstruction.items())
        ],
    )  # fmt: skip
    tables = result.tables
    if tables is None:
        return "\n".join(lines)
    described = {} if result.descriptions is None else result.descriptions.selected
    column = "description" if result.descriptions is None else DESCRIPTION_COLUMN

    def text(feature: int) -> str:
        return " / ".join(d.description for d in described.get(feature, ())) or "(uninterpreted)"

    lines += ["## Support per task (counted before any ranking)", ""]
    lines += _table(
        [
            "task",
            "scope",
            "raw yes",
            "raw no",
            "unknown",
            "out of scope",
            "yes",
            "no",
            "yes groups",
            "no groups",
            "tier",
        ],
        [
            [t, s.scope, s.raw_yes, s.raw_no, s.unknown, s.out_of_scope, s.yes, s.no, s.yes_groups, s.no_groups, s.tier]
            for t, s in tables.support.items()
        ],
    )
    strength = ["feature", "mean strength", "mean activation", "prevalence", "active groups", column]
    lines += ["## Strongest features over all objects", ""]
    lines += _table(
        strength,
        [
            [f.feature, f.mean_strength, f.mean_activation, f.prevalence, f.active_groups, text(f.feature)]
            for f in tables.strongest
        ],
    )
    for task, lists in tables.lists.items():
        for title, ranked in (("yes", lists.strongest_yes), ("no", lists.strongest_no)):
            lines += [f"## {task}: strongest features of the eligible {title} cohort", ""]
            lines += _table(
                strength,
                [
                    [f.feature, f.mean_strength, f.mean_activation, f.prevalence, f.active_groups, text(f.feature)]
                    for f in ranked
                ],
            )
        for title, separators in (("yes", lists.higher_in_yes), ("no", lists.higher_in_no)):
            lines += [f"## {task}: unvalidated discovery ranking, higher in the {title} cohort", ""]
            lines += _table(
                ["feature", "effect", "prevalence yes", "prevalence no", "difference", "active groups", column],
                [
                    [
                        f.feature,
                        f.effect,
                        f.prevalence_yes,
                        f.prevalence_no,
                        f.prevalence_difference,
                        f.active_groups,
                        text(f.feature),
                    ]
                    for f in separators
                ],
            )
    return "\n".join(lines)
