"""Tool schemas: valid JSON Schema, complete, and no unsafe tool slipped in."""

from __future__ import annotations

import json

import pytest

from spot_voice.brain.prompts import SYSTEM_PROMPT, system_blocks
from spot_voice.brain.tools import (
    CAMERAS,
    MOVE_DIRECTIONS,
    TOOL_NAMES,
    TOOLS,
    tools_with_cache_breakpoint,
)

EXPECTED_TOOLS = {
    "power_on",
    "stand",
    "sit",
    "move",
    "go_where_pointed",
    "come_here",
    "navigate_to",
    "list_waypoints",
    "start_follow",
    "stop_follow",
    "capture_image",
    "scan_room",
    "get_status",
    "dock",
    "undock",
    "speak",
    "emote",
    "stop_all",
}

#: Anything that would weaken or bypass a robot safety system must never appear
#: as a tool. This is the guard against one being added by accident.
FORBIDDEN_SUBSTRINGS = (
    "estop_allow",
    "release_estop",
    "allow_estop",
    "obstacle",
    "padding",
    "disable_safety",
    "override",
)


def test_tool_set_is_exactly_the_agreed_surface():
    assert set(TOOL_NAMES) == EXPECTED_TOOLS


def test_tool_names_are_unique():
    names = [tool["name"] for tool in TOOLS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool["name"])
def test_each_tool_is_well_formed(tool):
    assert set(tool) >= {"name", "description", "input_schema"}
    assert tool["name"].islower()
    # Descriptions are the main signal the model uses to pick a tool; a one-liner
    # is not enough for a robot that can physically move.
    assert len(tool["description"]) > 40

    schema = tool["input_schema"]
    assert schema["type"] == "object"
    assert isinstance(schema["properties"], dict)
    assert isinstance(schema.get("required", []), list)

    for name, prop in schema["properties"].items():
        assert prop["type"] in {"string", "number", "integer", "boolean", "array", "object"}
        assert prop.get("description"), f"{tool['name']}.{name} needs a description"

    for required in schema.get("required", []):
        assert required in schema["properties"], f"{tool['name']}: {required} not declared"


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool["name"])
def test_schemas_are_json_serialisable(tool):
    json.dumps(tool)


def test_move_enumerates_its_directions():
    move = next(tool for tool in TOOLS if tool["name"] == "move")
    assert move["input_schema"]["properties"]["direction"]["enum"] == MOVE_DIRECTIONS
    assert move["input_schema"]["required"] == ["direction"]


def test_capture_image_enumerates_its_cameras_and_defaults_to_front():
    capture = next(tool for tool in TOOLS if tool["name"] == "capture_image")
    assert capture["input_schema"]["properties"]["camera"]["enum"] == CAMERAS
    # camera is optional so a bare "what do you see" needs no argument.
    assert capture["input_schema"]["required"] == []


def test_no_tool_exposes_a_safety_override():
    blob = json.dumps(TOOLS).lower()
    for forbidden in FORBIDDEN_SUBSTRINGS:
        if forbidden == "obstacle":
            # The word may appear in prose ("obstacle avoidance stays on") but
            # never as a tool or parameter name.
            assert not any(forbidden in tool["name"] for tool in TOOLS)
            assert not any(
                forbidden in name
                for tool in TOOLS
                for name in tool["input_schema"]["properties"]
            )
            continue
        assert forbidden not in blob, forbidden


def test_cache_breakpoint_sits_on_the_last_tool_only():
    cached = tools_with_cache_breakpoint()
    assert cached[-1]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in tool for tool in cached[:-1])
    # The originals must not be mutated -- the module-level list is reused.
    assert all("cache_control" not in tool for tool in TOOLS)


# ----------------------------------------------------------------------
# System prompt


def test_system_prompt_covers_the_voice_and_safety_rules():
    lowered = SYSTEM_PROMPT.lower()
    assert "one or two short sentences" in lowered
    assert "spot" in lowered
    assert "obstacle avoidance" in lowered


def test_system_blocks_carry_one_cache_breakpoint():
    blocks = system_blocks()
    assert len(blocks) == 1
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    with_context = system_blocks("Places on this facility map: entrance.")
    assert len(with_context) == 2
    assert "cache_control" not in with_context[0]
    assert with_context[-1]["cache_control"] == {"type": "ephemeral"}


def test_system_prompt_has_no_per_request_content():
    # Anything that changes between requests would silently break prompt caching.
    for marker in ("{", "}", "%s"):
        assert marker not in SYSTEM_PROMPT
