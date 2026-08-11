"""Dock state gates motion.

Standing up while sitting on the dock is not something to attempt. Every command
that would put weight on the legs has to refuse until the robot is off the dock,
and the refusal has to name the fix so Claude can chain `undock` itself.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from spot_voice.brain.dispatcher import ToolDispatcher
from spot_voice.robot.mock import MockSpot

QUIET = Console(quiet=True)

DOCK_ID = 520

#: Tools that put weight on the legs and must refuse while docked.
MOTION_TOOLS = [
    ("stand", {}),
    ("move", {"direction": "forward", "distance_m": 0.5}),
    ("navigate_to", {"waypoint_name": "loading-bay"}),
]


@pytest.fixture()
def docked():
    """A mock robot that starts life sitting on its dock."""
    spot = MockSpot(dock_id=DOCK_ID, console=QUIET)
    spot.connect()
    assert spot.is_docked()
    return spot


@pytest.fixture()
def dispatcher(docked: MockSpot):
    return ToolDispatcher(robot=docked, console=QUIET)


# ----------------------------------------------------------------------
# Refusals while docked


@pytest.mark.parametrize("name,arguments", MOTION_TOOLS, ids=lambda v: str(v))
def test_motion_refuses_while_docked(dispatcher, name, arguments):
    if name == "navigate_to":
        pytest.skip("navigate_to deliberately undocks itself; covered separately")
    result = dispatcher.dispatch(name, arguments)
    assert result.ok is False
    assert "dock" in result.payload["message"].lower()
    assert "undock" in result.payload["message"].lower()


def test_the_refusal_tells_claude_what_to_do_next(dispatcher):
    # The message has to name the tool that fixes it, or the model has nothing
    # to chain to and the operator just hears a dead end.
    message = dispatcher.dispatch("move", {"direction": "forward"}).payload["message"]
    assert "undock" in message.lower()


def test_docked_robot_still_answers_questions(dispatcher):
    # Sensing must not be gated: status and cameras work on the dock.
    for name in ("get_status", "capture_image", "list_waypoints"):
        assert dispatcher.dispatch(name, {}).ok is True


def test_status_reports_the_dock_state(dispatcher):
    assert dispatcher.dispatch("get_status", {}).payload["docked"] is True


# ----------------------------------------------------------------------
# Undock


def test_undock_powers_on_and_stands(docked: MockSpot):
    result = docked.undock()
    assert result.ok
    assert not docked.is_docked()
    status = docked.get_status()
    assert status.data["motor_power"] == "on"
    assert status.data["posture"] == "standing"


def test_undock_when_already_off_the_dock_is_not_an_error(docked: MockSpot):
    docked.undock()
    again = docked.undock()
    assert again.ok is True
    assert "not on the dock" in again.message.lower()


def test_motion_works_once_undocked(dispatcher, docked: MockSpot):
    assert dispatcher.dispatch("undock", {}).ok
    assert dispatcher.dispatch("stand", {}).ok
    assert dispatcher.dispatch("move", {"direction": "forward", "distance_m": 0.1}).ok


def test_navigate_to_undocks_itself(dispatcher, docked: MockSpot):
    # "Go to the loading bay" is unambiguous about wanting to leave the dock, so
    # this one does not make the operator ask twice.
    result = dispatcher.dispatch("navigate_to", {"waypoint_name": "loading-bay"})
    assert result.ok is True
    assert not docked.is_docked()


# ----------------------------------------------------------------------
# Dock


def test_dock_returns_to_the_dock(dispatcher, docked: MockSpot):
    dispatcher.dispatch("undock", {})
    assert not docked.is_docked()

    result = dispatcher.dispatch("dock", {})
    assert result.ok is True
    assert docked.is_docked()


def test_docking_twice_is_harmless(dispatcher, docked: MockSpot):
    result = dispatcher.dispatch("dock", {})
    assert result.ok is True
    assert "already docked" in result.payload["message"].lower()


def test_dock_without_a_configured_dock_id_says_so():
    spot = MockSpot(dock_id=None, console=QUIET)
    spot.connect()
    result = ToolDispatcher(robot=spot, console=QUIET).dispatch("dock", {})
    assert result.ok is False
    assert "no dock" in result.payload["message"].lower()


# ----------------------------------------------------------------------
# Safety words still work on the dock


def test_stop_and_sit_are_never_gated_by_dock_state(docked: MockSpot):
    from spot_voice.safety.reflex import ReflexEngine

    engine = ReflexEngine(docked)
    for phrase in ("stop", "sit down"):
        outcome = engine.handle(phrase)
        assert outcome is not None and outcome.ok, phrase
