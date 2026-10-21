"""The chat UI's history (``chat/static/history.js``), run by Node.js: the turns each request sends.

A Yes/No turn is scored alone, and it leaves later history together with its result card: dropping only the card would
leave two user turns in a row, which an alternating chat template (Gemma's) refuses. A failed reply still leaves with
its user turn. Skips without ``node`` on the PATH. On Gemma 3 4B's tokenizer (its local snapshot), the template
renders the paired history and refuses the unpaired one.
"""

import itertools
import json
import shutil
import subprocess

import pytest
from tests import snapshots
from tests.unit.tiny_receiver import ROOT

from latent_interconnect.config import load_config
from latent_interconnect.receivers import open_renderer
from latent_interconnect.receivers.base import ReceiverSpec
from latent_interconnect.receivers.render import Conversation, Message

HISTORY = ROOT / "src" / "latent_interconnect" / "chat" / "static" / "history.js"
NODE = shutil.which("node")
SCORED = [
    {"role": "user", "content": "Here is a protein:[[P1]]. Describe it."},
    {"role": "assistant", "content": "A kinase.", "reasoning": "It phosphorylates.", "status": "done"},
    {"role": "user", "content": "Is [[P1]] an enzyme?", "readout": "yes_no"},
    {"role": "assistant", "content": "", "readout": "yes_no", "status": "done", "yesNo": {"score": 2.2}},
]


def request(messages, composer=None):
    """``AimmHistory.request(messages, composer)``, computed by the shipped script."""
    script = (
        "const h = require(process.argv[1]);"
        "process.stdout.write(JSON.stringify(h.request(JSON.parse(process.argv[2]), JSON.parse(process.argv[3]))));"
    )
    args = [str(HISTORY), json.dumps(messages), json.dumps(composer or {})]
    out = subprocess.run([NODE, "-e", script, *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="needs Node.js (node) to run the UI's history.js")
def test_a_yes_no_turn_is_sent_alone_and_leaves_later_history_with_its_card():
    assert request(SCORED[:3]) == {
        "messages": [{"role": "user", "content": "Is [[P1]] an enzyme?"}],
        "readout": "yes_no",
    }
    later = request([*SCORED, {"role": "user", "content": "Where is [[P1]] found?"}])
    assert later == {
        "messages": [
            {"role": "user", "content": "Here is a protein:[[P1]]. Describe it."},
            {"role": "assistant", "content": "A kinase.", "reasoning": "It phosphorylates."},
            {"role": "user", "content": "Where is [[P1]] found?"},
        ],
        "readout": "generate",
        "thinking": False,
    }


@pytest.mark.skipif(NODE is None, reason="needs Node.js (node) to run the UI's history.js")
def test_a_generation_asks_for_thinking_only_as_the_composer_shows_it():
    earlier = SCORED[:2]  # regenerating an ordinary reply: its user turn is the last
    assert request(earlier[:1], {"thinking": True})["thinking"] is True
    # Thinking stored on, but the Yes/No switch on shows it off and disabled: the request says off too
    assert request(earlier[:1], {"thinking": True, "yesNo": True})["thinking"] is False
    assert "thinking" not in request(SCORED[:3], {"thinking": True})  # a score sends no thinking mode


@pytest.mark.skipif(NODE is None, reason="needs Node.js (node) to run the UI's history.js")
def test_a_failed_card_or_reply_leaves_with_its_turn():
    failed_card = [*SCORED[:3], {"role": "assistant", "content": "", "readout": "yes_no", "status": "error"}]
    failed_reply = [
        {"role": "user", "content": "Describe [[P1]]."},
        {"role": "assistant", "content": "", "status": "stopped"},
    ]
    for earlier in (failed_card, failed_reply):
        sent = request([*earlier, {"role": "user", "content": "Again?"}])
        assert sent["messages"][-1] == {"role": "user", "content": "Again?"}
        roles = [m["role"] for m in sent["messages"]]
        assert all(a != b for a, b in itertools.pairwise(roles))  # alternating
    assert request([*failed_reply, {"role": "user", "content": "Again?"}])["messages"] == [
        {"role": "user", "content": "Again?"}
    ]


@snapshots.requires("gemma-3-4b-it")
def test_gemma_renders_the_paired_history_and_refuses_two_user_turns_in_a_row():
    spec = load_config(ROOT / "configs" / "receivers" / "gemma-3-4b-it.yaml", ReceiverSpec)
    renderer = open_renderer(spec.model_copy(update={"model_id": str(snapshots.snapshot("gemma-3-4b-it"))}))
    paired = (Message("user", "Describe it."), Message("assistant", "A kinase."), Message("user", "Where is it found?"))
    assert renderer.render(Conversation(paired), {}).input_ids
    with pytest.raises(Exception, match="alternate"):  # the card dropped alone: the scored turn stays
        renderer.render(Conversation((*paired[:2], Message("user", "Is it an enzyme?"), paired[2])), {})
