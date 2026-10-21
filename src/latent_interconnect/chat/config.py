"""The chat deployment config (``configs/chat/<site>.yaml``): receivers, their connectors, inputs and limits, as data.

A chat server serves one receiver of the config at a time (``--receiver``), so the config may name every receiver a
site can deploy; switching is a restart with another ``--receiver`` (its load cost: ``examples/guides/chat.md``). Each
connector is a training run's config (``$ref``) at a declared step: the run gives the receiver, the encoder recipe, the
domain and the cached embeddings (its ``embeddings`` pins), and its checkpoint is the run's output
``<destination>/step_N`` (``N`` six digits; its pin, read from the workspace's pins: a missing pin is an error, never a
fallback). A run of another receiver is
refused here, and its checkpoint's own receiver is checked again when the server loads it (``run.suite``'s
``checked_connector``).
"""

from __future__ import annotations

from pydantic import Field, model_validator

from latent_interconnect.config import Config
from latent_interconnect.receivers.base import GenerationPlan, ReceiverSpec
from latent_interconnect.training.config import RunConfig

__all__ = ["ChatConfig", "ChatConnector", "ChatDefaults", "ChatInputs", "ChatReceiver"]


class ChatInputs(Config):
    """How the objects of one domain are given in a chat placeholder.

    Attributes:
        id: the input type of an object id of the domain's object table (e.g. ``uniprot``): served from a run's
            cached embeddings when they hold it, else encoded live from the table's value.
        value: the input type of a raw value (e.g. ``sequence``, ``smiles``): always encoded live.
        max_chars: the longest raw value accepted, in characters (checked before any encoding).
    """

    id: str = Field(min_length=1)
    value: str = Field(min_length=1)
    max_chars: int = Field(ge=1)


class ChatConnector(Config):
    """One connector a receiver offers in chat.

    Attributes:
        label: its name in the UI.
        run: the training run that made it (``$ref`` to its config).
        step: the training step whose checkpoint is served.
        operational_only: a legacy exploratory-chat annotation, independent of the checkpoint's classification
            evaluation; ``/api/config`` reports it.
    """

    label: str
    run: RunConfig
    step: int = Field(ge=0)
    operational_only: bool = False

    @property
    def checkpoint_pin(self) -> str:
        """The pin of the served checkpoint: the run's ``<destination>/step_NNNNNN`` output."""
        if self.run.destination is None:
            raise ValueError(f"run {self.run.name!r} declares no destination, so its checkpoints have no pin")
        return f"{self.run.destination}/step_{self.step:06d}"


class ChatReceiver(Config):
    """One receiver a chat server can serve, with its connectors.

    Attributes:
        label: its name in the UI.
        receiver: the receiver config (``$ref``).
        generation_plan: the generation plan the backend opens with (None: the backend's default).
        context: the most tokens a prompt plus its continuation may hold (a longer conversation is refused, a long
            answer's cap is lowered to fit).
        expected_load_s: the load time the UI shows while the receiver loads (a measured value,
            ``examples/guides/chat.md``).
        connectors: connector id → connector; every run's receiver must be this receiver (no checkpoint is reused
            across receivers).
    """

    label: str
    receiver: ReceiverSpec
    generation_plan: GenerationPlan | None = None
    context: int = Field(ge=2)
    expected_load_s: float = Field(default=60.0, gt=0)
    connectors: dict[str, ChatConnector] = Field(min_length=1)

    @model_validator(mode="after")
    def _same_receiver(self) -> ChatReceiver:
        other = {c: e.run.receiver.name for c, e in self.connectors.items() if e.run.receiver != self.receiver}
        if other:
            raise ValueError(f"receiver {self.receiver.name!r}: connectors of runs on other receivers {other}")
        return self


class ChatDefaults(Config):
    """The settings a new conversation starts with.

    Attributes:
        thinking: thinking mode.
        temperature: sampling temperature (0 = greedy).
        max_tokens: the answer's token cap.
        system: the system message ("" for none).
    """

    thinking: bool = False
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    system: str = ""


class ChatConfig(Config):
    """A chat deployment.

    Attributes:
        receivers: receiver id → receiver; the server serves one, chosen at launch.
        inputs: domain name → how its objects are given (every connector's run domain needs an entry).
        defaults: a new conversation's settings.
        max_tokens_cap: the largest answer cap a request may ask for.
        max_placeholders: the most placeholders one conversation may use.
    """

    receivers: dict[str, ChatReceiver] = Field(min_length=1)
    inputs: dict[str, ChatInputs]
    defaults: ChatDefaults = ChatDefaults()
    max_tokens_cap: int = Field(default=16_384, ge=1)
    max_placeholders: int = Field(default=32, ge=1)

    @model_validator(mode="after")
    def _inputs_cover_domains(self) -> ChatConfig:
        domains = {e.run.domain.name for r in self.receivers.values() for e in r.connectors.values()}
        if missing := sorted(domains - set(self.inputs)):
            raise ValueError(f"no inputs declared for the domains {missing}")
        if self.defaults.max_tokens > self.max_tokens_cap:
            raise ValueError(f"default max_tokens {self.defaults.max_tokens} is over the cap {self.max_tokens_cap}")
        return self
