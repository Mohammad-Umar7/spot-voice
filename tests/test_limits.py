"""Velocity capping: no path to the SDK may exceed the hard limits."""

from __future__ import annotations

import math

import pytest

from spot_voice.robot.limits import (
    MAX_MOVE_DEGREES,
    MAX_MOVE_DISTANCE_M,
    MAX_MOVE_DURATION_S,
    MAX_VROT,
    MAX_VX,
    MAX_VY,
    VELOCITY_CMD_DURATION,
    VELOCITY_CMD_PERIOD,
    clamp_degrees,
    clamp_distance,
    clamp_duration,
    clamp_velocity,
)
from spot_voice.robot.motion import plan_move


def test_limits_are_the_documented_values():
    assert (MAX_VX, MAX_VY, MAX_VROT) == (0.6, 0.4, 0.8)


def test_command_expiry_outlasts_the_reissue_period():
    # The dead-man's switch only works if commands overlap rather than gap.
    assert VELOCITY_CMD_PERIOD < VELOCITY_CMD_DURATION


def test_values_within_limits_pass_through_untouched():
    velocity = clamp_velocity(0.3, -0.2, 0.5)
    assert velocity.as_tuple() == (0.3, -0.2, 0.5)
    assert velocity.clamped is False


@pytest.mark.parametrize(
    "requested,expected",
    [
        ((10.0, 10.0, 10.0), (MAX_VX, MAX_VY, MAX_VROT)),
        ((-10.0, -10.0, -10.0), (-MAX_VX, -MAX_VY, -MAX_VROT)),
        ((1e9, 0.0, 0.0), (MAX_VX, 0.0, 0.0)),
    ],
)
def test_excessive_values_are_clamped(requested, expected):
    velocity = clamp_velocity(*requested)
    assert velocity.as_tuple() == expected
    assert velocity.clamped is True


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf"), "fast", None, object()]
)
def test_non_numeric_and_non_finite_collapse_to_zero(bad):
    velocity = clamp_velocity(bad, bad, bad)
    assert velocity.as_tuple() == (0.0, 0.0, 0.0)
    assert velocity.clamped is True


def test_boundary_values_are_not_flagged_as_clamped():
    velocity = clamp_velocity(MAX_VX, -MAX_VY, MAX_VROT)
    assert velocity.as_tuple() == (MAX_VX, -MAX_VY, MAX_VROT)
    assert velocity.clamped is False


def test_distance_degrees_and_duration_are_bounded():
    assert clamp_distance(500.0) == (MAX_MOVE_DISTANCE_M, True)
    assert clamp_distance(-2.0) == (2.0, False)  # magnitude; direction is separate
    assert clamp_distance("far") == (0.0, True)
    assert clamp_degrees(10_000) == (MAX_MOVE_DEGREES, True)
    assert clamp_duration(10_000) == (MAX_MOVE_DURATION_S, True)
    assert clamp_duration(-1) == (0.0, True)


# ----------------------------------------------------------------------
# move() planning, shared by the mock and the real client


def test_move_plan_respects_caps_even_when_a_huge_speed_is_requested():
    plan = plan_move("forward", distance_m=100.0, degrees=None, speed=99.0)
    assert plan is not None
    velocity, duration, message = plan
    assert velocity.v_x == MAX_VX
    assert velocity.clamped is True
    assert duration <= MAX_MOVE_DURATION_S
    # Distance is capped too, so the spoken summary must not promise 100 m.
    assert "5.0 metres" in message


def test_move_plan_turn_durations_are_consistent():
    plan = plan_move("turn_left", distance_m=None, degrees=90.0, speed=None)
    assert plan is not None
    velocity, duration, message = plan
    assert velocity.v_rot > 0  # counter-clockwise
    assert duration == pytest.approx(math.radians(90.0) / velocity.v_rot, rel=1e-6)
    assert "left" in message


def test_move_plan_turn_right_is_clockwise():
    plan = plan_move("turn_right", None, 45.0, None)
    assert plan is not None
    assert plan[0].v_rot < 0
    assert "right" in plan[2]


@pytest.mark.parametrize("direction", ["forward", "back", "left", "right"])
def test_move_plan_never_mixes_rotation_into_a_straight_move(direction):
    plan = plan_move(direction, 1.0, None, None)
    assert plan is not None
    assert plan[0].v_rot == 0.0


def test_move_plan_rejects_unknown_directions():
    assert plan_move("sideways-ish", 1.0, None, None) is None
    assert plan_move("", None, None, None) is None
    assert plan_move(None, None, None, None) is None  # type: ignore[arg-type]


def test_move_plan_tolerates_spoken_variants():
    for direction in ("Forward", "forwards", "turn left", "turn-left", "backwards"):
        assert plan_move(direction, 1.0, 90.0, None) is not None, direction
