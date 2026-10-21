"""Training: the run config and the one trainer. Streams and the sampler are in ``data``."""

from latent_interconnect.training.config import RunConfig
from latent_interconnect.training.trainer import InJobEvaluator, Trainer

__all__ = ["InJobEvaluator", "RunConfig", "Trainer"]
