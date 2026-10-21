"""Statistics: AUROC, the bootstrap, the shared-unit interval of a crossed design, a reconstruction's error summaries.

Nothing outside this package computes an interval: the bootstrap (``stats.bootstrap``) everywhere by default, and the
shared-unit interval (``stats.shared_units``) only where a config selects it. Estimates and intervals only: no
grade, margin or p-value.
"""

from latent_interconnect.stats.auroc import (
    auroc,
    auroc_columns,
    auroc_counts,
    auroc_influence,
    class_strata,
    has_class_units,
)
from latent_interconnect.stats.bootstrap import (
    DRAWS,
    LEVEL,
    SEED,
    Bootstrap,
    Interval,
    IntervalFlag,
    ResamplingUnits,
    Statistic,
    defined_draws,
    summarize,
    unit_index,
)
from latent_interconnect.stats.shared_units import (
    FULL_RANGE,
    INDEPENDENT_VARIANCE,
    SHARED_UNIT,
    SharedUnitVariance,
    shared_unit_interval,
    shared_unit_variance,
)
from latent_interconnect.stats.state_error import StateErrors

__all__ = [
    "DRAWS",
    "FULL_RANGE",
    "INDEPENDENT_VARIANCE",
    "LEVEL",
    "SEED",
    "SHARED_UNIT",
    "Bootstrap",
    "Interval",
    "IntervalFlag",
    "ResamplingUnits",
    "SharedUnitVariance",
    "StateErrors",
    "Statistic",
    "auroc",
    "auroc_columns",
    "auroc_counts",
    "auroc_influence",
    "class_strata",
    "defined_draws",
    "has_class_units",
    "shared_unit_interval",
    "shared_unit_variance",
    "summarize",
    "unit_index",
]
