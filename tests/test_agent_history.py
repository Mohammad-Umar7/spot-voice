"""Brain internals: history trimming and tool_result construction.

Trimming a conversation that contains tool calls is easy to get subtly wrong -- an
orphaned ``tool_result`` at the head of the window is a hard 400 from the API,
mid-demo. These tests pin the invariant.
"""

from __future__ import annotations

import base64
import json

from spot_voice.brain.agent import (
    MAX_HISTORY_MESSAGES,
    _collect_text,
    _is_clean_start,
    _tool_result_block,
)


class FakeBlock:
    """Stand-in for an SDK content block."""

    def __init__(self, type_: str, **fields) -> None:
        self.type = type_
        for key, value in fields.items():
            setattr(self, key, value)


# ----------------------------------------------------------------------
# tool_result blocks


def test_tool_result_carries_the_payload_as_json_text():
    block = _tool_result_block("toolu_1", {"ok": True, "message": "Standing."})
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_1"
    assert json.loads(block["content"][0]["text"]) == {"ok": True, "message": "Standing."}


def test_tool_result_attaches_an_image_block_when_a_photo_was_taken():
    jpeg = b"\xff\xd8\xff\xe0 fake jpeg bytes"
    block = _tool_result_block("toolu_2", {"ok": True, "message": "Here."}, jpeg)

    assert len(block["content"]) == 2
    image = block["content"][1]
    assert image["type"] == "image"
    assert image["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(image["source"]["data"]) == jpeg


def test_failures_are_ordinary_results_not_is_error():
    # The system prompt tells Claude to speak the message from a failed tool, so
    # the result must arrive as data rather than as a broken-tool signal.
    block = _tool_result_block("toolu_3", {"ok": False, "message": "I lost connection."})
    assert "is_error" not in block
    assert json.loads(block["content"][0]["text"])["ok"] is False


# ----------------------------------------------------------------------
# Window repair


def test_a_plain_user_turn_is_a_valid_window_start():
    assert _is_clean_start({"role": "user", "content": "stand up"})
    assert _is_clean_start(
        {"role": "user", "content": [{"type": "text", "text": "stand up"}]}
    )


def test_an_assistant_turn_is_not_a_valid_window_start():
    assert not _is_clean_start({"role": "assistant", "content": "ok"})


def test_a_tool_result_turn_is_not_a_valid_window_start():
    orphan = {
        "role": "user",
        "content": [_tool_result_block("toolu_9", {"ok": True, "message": "done"})],
    }
    assert not _is_clean_start(orphan)


def test_trimming_never_leaves_an_orphaned_tool_result_at_the_head():
    from spot_voice.brain.agent import Brain

    brain = Brain.__new__(Brain)  # no API client needed for the pure logic
    brain._messages = []
    for index in range(MAX_HISTORY_MESSAGES + 10):
        brain._messages.append({"role": "user", "content": f"turn {index}"})
        brain._messages.append(
            {"role": "assistant", "content": [FakeBlock("tool_use", id="t", name="sit")]}
        )
        brain._messages.append(
            {
                "role": "user",
                "content": [_tool_result_block("t", {"ok": True, "message": "Sitting."})],
            }
        )

    brain._trim()

    assert len(brain._messages) <= MAX_HISTORY_MESSAGES
    assert _is_clean_start(brain._messages[0])


def test_trimming_an_empty_history_is_safe():
    from spot_voice.brain.agent import Brain

    brain = Brain.__new__(Brain)
    brain._messages = []
    brain._trim()
    assert brain._messages == []


# ----------------------------------------------------------------------
# Reply text


def test_text_blocks_are_joined_and_stripped():
    content = [
        FakeBlock("text", text="  Walking to the loading bay. "),
        FakeBlock("tool_use", id="t", name="navigate_to", input={}),
        FakeBlock("text", text="I'll tell you what I see."),
    ]
    assert _collect_text(content) == (
        "Walking to the loading bay. I'll tell you what I see."
    )


def test_collect_text_handles_dicts_and_emptiness():
    assert _collect_text([{"type": "text", "text": "hi"}]) == "hi"
    assert _collect_text([]) == ""
    assert _collect_text(None) == ""
