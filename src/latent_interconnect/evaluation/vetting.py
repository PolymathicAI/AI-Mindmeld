"""Prompt vetting: evaluation prompts cannot reveal the answer (replacing the shuffled control).

Evaluation prompts have no in-context examples, so one check remains, run by every suite
runner on the prompts it renders, before scoring:

- **Object-blind text:** within a task and condition, every item's prompt has the same token ids. The object enters
  only through the slot writes, so the text cannot carry an object's label. The name condition is exempt, since it
  replaces the slot with the name by design.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["PromptRecord", "vet_prompts"]


@dataclass(frozen=True)
class PromptRecord:
    """One rendered evaluation prompt, as the vetting sees it.

    Attributes:
        task: the task id.
        condition: the condition combination, e.g. ``"object=own"``.
        item: the item index.
        input_ids: the rendered token ids.
    """

    task: str
    condition: str
    item: int
    input_ids: tuple[int, ...]


def vet_prompts(prompts: Sequence[PromptRecord], name_conditions: Sequence[str] = ()) -> list[str]:
    """Return one finding per (task, condition) whose prompt text depends on the item (empty when clean).

    Args:
        prompts: every prompt a suite renders.
        name_conditions: condition labels that put the object's name in the text (exempt).
    """
    groups: dict[tuple[str, str], set[tuple[int, ...]]] = defaultdict(set)
    for p in prompts:
        if p.condition not in name_conditions:
            groups[(p.task, p.condition)].add(p.input_ids)
    return [
        f"{task} [{condition}]: {len(variants)} different prompt texts; the text depends on the item"
        for (task, condition), variants in sorted(groups.items())
        if len(variants) > 1
    ]
