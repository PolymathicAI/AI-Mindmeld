"""Evaluation: suite runners, the score schema, analyses and the layer-selection tools.

Importing the package registers the built kinds: the suite runners (``runners``, ``cross_domain``, ``generating``)
and the analysis kinds (``contrasts``, ``counts``, ``cost``, ``cross_domain_quads``, ``cross_domain_ranks``, ``drift``,
``excerpts``, ``explicit_pairs``, ``exposure``, ``generation_analyses``, ``layer_selection_analysis``,
``sae_exploration``, ``stratified``).
"""

from latent_interconnect.evaluation import contrasts as _contrasts  # noqa: F401  (registers analysis kinds)
from latent_interconnect.evaluation import cost as _cost  # noqa: F401  (registers cost)
from latent_interconnect.evaluation import counts as _counts  # noqa: F401  (registers counts, removal_counts)
from latent_interconnect.evaluation import cross_domain as _cross_domain  # noqa: F401  (registers a suite kind)
from latent_interconnect.evaluation import cross_domain_quads as _cross_domain_quads  # noqa: F401  (an analysis kind)
from latent_interconnect.evaluation import cross_domain_ranks as _cross_domain_ranks  # noqa: F401  (an analysis kind)
from latent_interconnect.evaluation import drift as _drift  # noqa: F401  (registers execution_drift)
from latent_interconnect.evaluation import excerpts as _excerpts  # noqa: F401  (registers excerpts)
from latent_interconnect.evaluation import explicit_pairs as _explicit_pairs  # noqa: F401  (registers explicit_pairs)
from latent_interconnect.evaluation import exposure as _exposure  # noqa: F401  (registers description_exposure)
from latent_interconnect.evaluation import generating as _generating  # noqa: F401  (registers generation suite kinds)
from latent_interconnect.evaluation import generation_analyses as _generation_analyses  # noqa: F401  (analysis kinds)
from latent_interconnect.evaluation import layer_selection_analysis as _layer_selection  # noqa: F401  (registers one)
from latent_interconnect.evaluation import runners as _runners  # noqa: F401  (registers suite kinds)
from latent_interconnect.evaluation import sae_exploration as _sae_exploration  # noqa: F401  (registers one)
from latent_interconnect.evaluation import stratified as _stratified  # noqa: F401  (registers stratified_contrast)
from latent_interconnect.evaluation.analysis import ANALYSIS_KINDS, AnalysisSpec, Contrast, Group, GroupedContrasts
from latent_interconnect.evaluation.scores import GENERATION_COLUMNS, ROLE_COLUMNS, SCORE_COLUMNS
from latent_interconnect.evaluation.suites import SUITE_KINDS, SuiteContext, SuiteRunConfig, SuiteRunner

__all__ = [
    "ANALYSIS_KINDS",
    "GENERATION_COLUMNS",
    "ROLE_COLUMNS",
    "SCORE_COLUMNS",
    "SUITE_KINDS",
    "AnalysisSpec",
    "Contrast",
    "Group",
    "GroupedContrasts",
    "SuiteContext",
    "SuiteRunConfig",
    "SuiteRunner",
]
