"""Data below training and evaluation: item sets, streams, the sampler and suite specs.

Importing the package registers the built-in streams (``data.qa_streams``, ``data.relational_stream``,
``data.variant_stream``) and, explicitly below, the relational answers of each domain (``protein``, ``molecule``).
"""

from latent_interconnect.data import qa_streams as _qa_streams  # registers "text" and "yes_no"
from latent_interconnect.data import relational_stream as _relational_stream  # registers "relational"
from latent_interconnect.data import variant_stream as _variant_stream  # registers "variant"
from latent_interconnect.data.items import ItemSet, derangement
from latent_interconnect.data.relational_answers_molecule import MOLECULE
from latent_interconnect.data.relational_base import RELATIONAL_ANSWERS
from latent_interconnect.data.relational_protein import PROTEIN
from latent_interconnect.data.sampler import MixtureEntry, Sampler
from latent_interconnect.data.scaffold import FactSpec, ScaffoldSpec
from latent_interconnect.data.streams import STREAMS, Example, LossCurve, Stream
from latent_interconnect.data.suites import Condition, SlotSpec, SuiteSpec
from latent_interconnect.data.training_text import TrainingText

del _qa_streams, _relational_stream, _variant_stream
RELATIONAL_ANSWERS.add("protein", PROTEIN)
RELATIONAL_ANSWERS.add("molecule", MOLECULE)

__all__ = [
    "STREAMS",
    "Condition",
    "Example",
    "FactSpec",
    "ItemSet",
    "LossCurve",
    "MixtureEntry",
    "Sampler",
    "ScaffoldSpec",
    "SlotSpec",
    "Stream",
    "SuiteSpec",
    "TrainingText",
    "derangement",
]
