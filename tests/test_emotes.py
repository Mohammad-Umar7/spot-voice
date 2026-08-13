"""Body-language gestures, and the pose clamping that keeps them safe.

Spot has no arm, so "say hi to these people" is a body movement. The model picks
the gesture; this module decides how far the robot is allowed to lean doing it.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from spot_voice.brain.dispatcher import ToolDispatcher
from spot_voice.robot.emotes import (
    GESTURES,
    MAX_HEIGHT_OFFSET,
    MAX_PITCH,
    MAX_ROLL,
    MAX_YAW,
    NEUTRAL,
    Pose,
    available,
    clamp_pose,
    resolve,
    sequence,
)
from spot_voice.robot.mock import MockSpot

QUIET = Console(quiet=True)


@pytest.fixture()
def dispatcher():
    spot = MockSpot(dock_id=None, console=QUIET)
    spot.connect()
    return ToolDispatcher(robot=spot, console=QUIET)


# ----------------------------------------------------------------------
# Clamping -- the safety layer


def test_a_reasonable_pose_passes_through():
    pose = Pose(yaw=0.2, roll=0.1, pitch=0.15, height=0.05, hold=0.4)
    assert clamp_pose(pose) == pose


@pytest.mark.parametrize(
    "field,limit",
    [("yaw", MAX_YAW), ("roll", MAX_ROLL), ("pitch", MAX_PITCH), ("height", MAX_HEIGHT_OFFSET)],
)
def test_every_axis_is_clamped_both_ways(field, limit):
    assert getattr(clamp_pose(Pose(**{field: 99.0})), field) == limit
    assert getattr(clamp_pose(Pose(**{field: -99.0})), field) == -limit


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "over", None])
def test_nonsense_angles_collapse_to_standing_straight(bad):
    pose = clamp_pose(Pose(yaw=bad, roll=bad, pitch=bad, height=bad))
    assert (pose.yaw, pose.roll, pose.pitch, pose.height) == (0.0, 0.0, 0.0, 0.0)


def test_hold_times_are_bounded_and_positive():
    assert clamp_pose(Pose(hold=99.0)).hold <= 1.5
    assert clamp_pose(Pose(hold=-3.0)).hold > 0
    assert clamp_pose(Pose(hold=0.0)).hold > 0


# ----------------------------------------------------------------------
# The gesture library


def test_every_gesture_is_clamped_and_ends_standing_straight():
    for name in available():
        summary, poses = sequence(name)
        assert summary and poses
        assert poses[-1] == NEUTRAL, f"{name} must finish neutral"
        for pose in poses:
            assert abs(pose.pitch) <= MAX_PITCH
            assert abs(pose.roll) <= MAX_ROLL
            assert abs(pose.yaw) <= MAX_YAW
            assert abs(pose.height) <= MAX_HEIGHT_OFFSET


def test_no_gesture_runs_longer_than_the_cap():
    from spot_voice.robot.emotes import MAX_GESTURE_SEC

    for name in available():
        _summary, poses = sequence(name)
        assert sum(pose.hold for pose in poses) <= MAX_GESTURE_SEC + 1.5


def test_gestures_are_short_enough_to_read_as_a_gesture():
    # A long animation in front of people reads as a malfunction.
    for name in available():
        _summary, poses = sequence(name)
        assert sum(pose.hold for pose in poses) < 4.0, name


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("hi", "greet"),
        ("hello", "greet"),
        ("wave", "greet"),  # no arm, so a greeting bow stands in
        ("say hi", "greet"),
        ("Say-Hello", "greet"),
        ("dance", "wiggle"),
        ("agree", "yes"),
        ("shake head", "shake"),
        ("BOW", "bow"),
        ("look around", "look_around"),
    ],
)
def test_spoken_names_resolve_to_real_gestures(spoken, expected):
    assert resolve(spoken) == expected


def test_unknown_gestures_resolve_to_nothing():
    for name in ("backflip", "", "somersault", None):
        assert resolve(name) is None
        assert sequence(name) is None


def test_greet_is_the_one_for_saying_hello():
    assert "greet" in GESTURES
    summary, poses = sequence("greet")
    assert "hello" in summary.lower()
    assert len(poses) > 1  # it is a sequence, not a single lean


# ----------------------------------------------------------------------
# Through the dispatcher


def test_saying_hi_produces_a_gesture(dispatcher):
    result = dispatcher.dispatch("emote", {"gesture": "greet"})
    assert result.ok is True
    assert result.payload["message"]


def test_an_unknown_gesture_lists_the_real_ones(dispatcher):
    result = dispatcher.dispatch("emote", {"gesture": "backflip"})
    assert result.ok is False
    assert "greet" in result.payload["message"]


def test_a_missing_gesture_argument_is_handled(dispatcher):
    assert dispatcher.dispatch("emote", {}).ok is False


def test_gestures_are_refused_on_the_dock():
    spot = MockSpot(dock_id=520, console=QUIET)
    spot.connect()
    result = ToolDispatcher(robot=spot, console=QUIET).dispatch(
        "emote", {"gesture": "greet"}
    )
    assert result.ok is False
    assert "dock" in result.payload["message"].lower()


def test_a_gesture_stands_the_robot_up_first(dispatcher):
    dispatcher.dispatch("sit", {})
    assert dispatcher.dispatch("get_status", {}).payload["posture"] == "sitting"

    assert dispatcher.dispatch("emote", {"gesture": "nod"}).ok
    assert dispatcher.dispatch("get_status", {}).payload["posture"] == "standing"


def test_the_emote_tool_offers_every_gesture():
    from spot_voice.brain.tools import GESTURES as SCHEMA_GESTURES, TOOLS

    emote = next(tool for tool in TOOLS if tool["name"] == "emote")
    assert set(SCHEMA_GESTURES) == set(available())
    assert emote["input_schema"]["properties"]["gesture"]["enum"] == SCHEMA_GESTURES
