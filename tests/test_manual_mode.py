"""Manual mode: the bring-up path that needs nothing but the robot.

"Stand" is not a reflex word, so without this you would need a working Anthropic
connection to stand a real robot -- which is the one thing you cannot count on at
stage 2 of the rollout, standing the robot for the first time over a tether.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from spot_voice.brain.dispatcher import ToolDispatcher
from spot_voice.brain.tools import TOOL_NAMES
from spot_voice.main import parse_manual_command
from spot_voice.robot.mock import MockSpot

QUIET = Console(quiet=True)


@pytest.fixture()
def dispatcher():
    spot = MockSpot(dock_id=None, console=QUIET)
    spot.connect()
    return ToolDispatcher(robot=spot, console=QUIET)


# ----------------------------------------------------------------------
# Parsing


@pytest.mark.parametrize(
    "line,expected",
    [
        ("stand", ("stand", {})),
        ("up", ("stand", {})),
        ("sit", ("sit", {})),
        ("down", ("sit", {})),
        ("stop", ("stop_all", {})),
        ("halt", ("stop_all", {})),
        ("status", ("get_status", {})),
        ("waypoints", ("list_waypoints", {})),
        ("undock", ("undock", {})),
        ("dock", ("dock", {})),
        ("follow", ("start_follow", {})),
        ("unfollow", ("stop_follow", {})),
    ],
)
def test_bare_commands(line, expected):
    assert parse_manual_command(line) == expected


def test_commands_are_case_and_whitespace_insensitive():
    assert parse_manual_command("  STAND  ") == ("stand", {})


def test_move_takes_metres_for_a_straight_line():
    assert parse_manual_command("move forward 1.5") == (
        "move",
        {"direction": "forward", "distance_m": 1.5},
    )


def test_move_takes_degrees_for_a_turn():
    assert parse_manual_command("move turn_left 90") == (
        "move",
        {"direction": "turn_left", "degrees": 90.0},
    )


def test_move_without_an_amount_uses_the_tool_default():
    assert parse_manual_command("move forward") == ("move", {"direction": "forward"})


def test_move_with_a_non_numeric_amount_is_rejected():
    assert parse_manual_command("move forward abit") is None


def test_look_defaults_to_the_front_camera():
    assert parse_manual_command("look") == ("capture_image", {"camera": "front"})
    assert parse_manual_command("look left") == ("capture_image", {"camera": "left"})


def test_go_keeps_multi_word_waypoint_names():
    assert parse_manual_command("go loading bay") == (
        "navigate_to",
        {"waypoint_name": "loading bay"},
    )


def test_say_keeps_the_whole_sentence():
    assert parse_manual_command("say checking the panel now") == (
        "speak",
        {"text": "checking the panel now"},
    )


@pytest.mark.parametrize("line", ["", "   ", "fly", "teleport home", "move"])
def test_unknown_commands_are_rejected_rather_than_guessed(line):
    assert parse_manual_command(line) is None


def test_every_command_maps_to_a_real_tool():
    lines = [
        "stand", "sit", "stop", "status", "waypoints", "dock", "undock",
        "follow", "unfollow", "look", "say hi", "go entrance", "move forward 1",
    ]
    for line in lines:
        parsed = parse_manual_command(line)
        assert parsed is not None, line
        assert parsed[0] in TOOL_NAMES, line


# ----------------------------------------------------------------------
# Execution -- the bring-up sequence itself


def test_the_bring_up_sequence_works_with_no_brain(dispatcher):
    # This is the scenario: robot sitting on the floor, motors off, no
    # Anthropic connection. Nothing here touches the brain module.
    before = dispatcher.dispatch("get_status", {}).payload
    assert before["motor_power"] == "off"
    assert before["posture"] == "sitting"

    name, arguments = parse_manual_command("stand")
    assert dispatcher.dispatch(name, arguments).ok

    after = dispatcher.dispatch("get_status", {}).payload
    assert after["motor_power"] == "on"
    assert after["posture"] == "standing"


def test_manual_mode_imports_nothing_from_the_brain_agent(monkeypatch):
    # Guard the property that makes this path trustworthy: if importing
    # spot_voice.main pulled in the Anthropic client, manual mode would fail on
    # a machine with no SDK installed.
    import sys

    monkeypatch.setitem(sys.modules, "anthropic", None)
    import importlib

    import spot_voice.main as main_module

    importlib.reload(main_module)
    assert main_module.parse_manual_command("stand") == ("stand", {})
