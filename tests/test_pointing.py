"""Reading "stand over there" off a pointing arm.

The design rests on one honest split: the gesture gives a reliable **bearing**
and no distance at all. These tests pin both halves -- that the bearing tracks
the arm, and that the distance only ever comes from a real measurement or a
deliberately modest default.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from spot_voice.brain.dispatcher import ToolDispatcher
from spot_voice.robot.mock import MockSpot
from spot_voice.vision.pointing import (
    DEFAULT_DISTANCE_M,
    STANDOFF_M,
    PointingResult,
    apply_measured_distance,
    detect_pointing,
    pick_pointer,
)

QUIET = Console(quiet=True)
FRAME_W = 640

# COCO keypoint indices.
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10


def pose(*, left_wrist=None, right_wrist=None, shoulder_y=200.0, confidence=0.9):
    """Build 17 COCO keypoints for someone facing the robot.

    Shoulders sit either side of frame centre; wrists are placed where the test
    wants them. Unset joints get low confidence so they are ignored.
    """
    points = [[0.0, 0.0, 0.0] for _ in range(17)]
    points[L_SHOULDER] = [280.0, shoulder_y, confidence]
    points[R_SHOULDER] = [360.0, shoulder_y, confidence]
    if left_wrist is not None:
        points[L_WRIST] = [left_wrist[0], left_wrist[1], confidence]
    if right_wrist is not None:
        points[R_WRIST] = [right_wrist[0], right_wrist[1], confidence]
    return points


# ----------------------------------------------------------------------
# Reading the gesture


def test_arms_down_is_not_a_pointing_gesture():
    # Wrists just below the shoulders: standing normally, not pointing.
    resting = pose(left_wrist=(285.0, 260.0), right_wrist=(355.0, 260.0))
    assert detect_pointing(resting, FRAME_W) is None


def test_an_arm_extended_left_reads_as_a_left_bearing():
    # Positive bearing is to the robot's left, which is the low-x side of the
    # image, so the arm reaching towards x=60 must give a positive bearing.
    result = detect_pointing(pose(left_wrist=(60.0, 200.0)), FRAME_W)
    assert result is not None
    assert result.bearing_deg > 0
    assert result.arm == "left"


def test_an_arm_extended_right_reads_as_a_right_bearing():
    result = detect_pointing(pose(right_wrist=(600.0, 200.0)), FRAME_W)
    assert result is not None
    assert result.bearing_deg < 0
    assert result.arm == "right"


def test_pointing_further_out_gives_a_wider_bearing():
    # Both must clear the extension threshold, or they are not gestures at all.
    slight = detect_pointing(pose(left_wrist=(175.0, 200.0)), FRAME_W)
    wide = detect_pointing(pose(left_wrist=(60.0, 200.0)), FRAME_W)
    assert slight is not None and wide is not None
    assert wide.bearing_deg > slight.bearing_deg


def test_the_bearing_does_not_saturate_across_normal_gestures():
    # The failure this guards against: clamping the projected pixel to the frame
    # made every wide gesture return the same bearing, so "slightly left" and
    # "hard left" were indistinguishable.
    bearings = [
        detect_pointing(pose(left_wrist=(x, 200.0)), FRAME_W).bearing_deg
        for x in (180.0, 150.0, 120.0)
    ]
    assert len(set(bearings)) == 3, f"bearings collapsed: {bearings}"
    assert bearings == sorted(bearings)


def test_the_more_extended_arm_is_the_one_that_counts():
    # One arm half-raised, the other fully extended: read the extended one.
    points = pose(left_wrist=(250.0, 240.0), right_wrist=(620.0, 195.0))
    result = detect_pointing(points, FRAME_W)
    assert result is not None
    assert result.arm == "right"


def test_the_bearing_never_exceeds_half_the_field_of_view():
    # Pointing past the edge of what the camera can see saturates rather than
    # producing a bearing the robot cannot have observed.
    for wrist_x in (-500.0, 0.0, 640.0, 2000.0):
        result = detect_pointing(pose(left_wrist=(wrist_x, 200.0)), FRAME_W)
        if result is not None:
            assert abs(result.bearing_deg) <= 50.0 + 1e-6


def test_low_confidence_joints_are_ignored():
    faint = pose(left_wrist=(60.0, 200.0), confidence=0.1)
    assert detect_pointing(faint, FRAME_W) is None


def test_malformed_keypoints_do_not_raise():
    for bad in (None, [], [[0, 0, 0]] * 3, "not keypoints"):
        assert detect_pointing(bad, FRAME_W) is None
    assert detect_pointing(pose(left_wrist=(60.0, 200.0)), 0) is None


def test_a_gesture_read_carries_a_default_distance_not_a_guess():
    # The gesture itself contains no distance information, so before any
    # measurement it must fall back to the modest default.
    result = detect_pointing(pose(left_wrist=(60.0, 200.0)), FRAME_W)
    assert result.distance_m == DEFAULT_DISTANCE_M
    assert result.distance_measured is False


# ----------------------------------------------------------------------
# Distance comes from the sensor, not the gesture


def test_a_measurement_replaces_the_default_and_keeps_a_standoff():
    reading = PointingResult(bearing_deg=20.0, distance_m=DEFAULT_DISTANCE_M, arm="left", confidence=0.8)
    updated = apply_measured_distance(reading, measured_m=4.0)

    assert updated.distance_measured is True
    assert updated.distance_m == pytest.approx(4.0 - STANDOFF_M)
    assert updated.bearing_deg == 20.0  # bearing is untouched


def test_a_wall_closer_than_the_standoff_means_do_not_move():
    reading = PointingResult(bearing_deg=0.0, distance_m=2.0, arm="left", confidence=0.8)
    assert apply_measured_distance(reading, measured_m=0.6).distance_m == 0.0


def test_a_missing_measurement_leaves_the_modest_default():
    reading = PointingResult(bearing_deg=0.0, distance_m=DEFAULT_DISTANCE_M, arm="left", confidence=0.8)
    for missing in (None, float("nan"), 0.0, -3.0):
        result = apply_measured_distance(reading, measured_m=missing)
        assert result.distance_m == DEFAULT_DISTANCE_M
        assert result.distance_measured is False


def test_a_measurement_is_capped_by_the_move_limit():
    from spot_voice.robot.limits import MAX_MOVE_DISTANCE_M

    reading = PointingResult(bearing_deg=0.0, distance_m=2.0, arm="left", confidence=0.8)
    updated = apply_measured_distance(reading, measured_m=100.0, max_distance_m=MAX_MOVE_DISTANCE_M)
    assert updated.distance_m <= MAX_MOVE_DISTANCE_M


# ----------------------------------------------------------------------
# What the robot says before it moves


def test_the_spoken_summary_names_the_side_and_the_distance():
    left = PointingResult(bearing_deg=30.0, distance_m=3.0, arm="left", confidence=0.9, distance_measured=True)
    assert "left" in left.describe() and "3.0" in left.describe()

    right = PointingResult(bearing_deg=-25.0, distance_m=2.0, arm="right", confidence=0.9)
    assert "right" in right.describe()


def test_a_near_centre_bearing_reads_as_straight_ahead():
    result = PointingResult(bearing_deg=2.0, distance_m=2.0, arm="left", confidence=0.9)
    assert "straight ahead" in result.describe()


def test_an_estimated_distance_is_hedged_and_a_measured_one_is_not():
    estimated = PointingResult(bearing_deg=0.0, distance_m=2.0, arm="left", confidence=0.9)
    measured = PointingResult(bearing_deg=0.0, distance_m=2.0, arm="left", confidence=0.9, distance_measured=True)
    assert "roughly" in estimated.describe()
    assert "roughly" not in measured.describe()


# ----------------------------------------------------------------------
# Choosing whose gesture to read


def test_the_nearest_person_is_the_one_whose_gesture_counts():
    near = ([], (300, 100, 380, 460))  # tall box = close
    far = ([], (100, 220, 130, 300))
    assert pick_pointer([far, near])[1] == near[1]
    assert pick_pointer([]) is None
    assert pick_pointer(None) is None


# ----------------------------------------------------------------------
# Through the dispatcher


@pytest.fixture()
def robot():
    spot = MockSpot(dock_id=None, console=QUIET)
    spot.connect()
    return spot


def test_nobody_pointing_asks_them_to_try_again(robot):
    dispatcher = ToolDispatcher(
        robot=robot, console=QUIET, pose_reader=lambda _jpeg: None
    )
    result = dispatcher.dispatch("go_where_pointed", {})
    assert result.ok is False
    assert "arm out" in result.payload["message"]


def test_no_pose_model_is_reported_not_crashed(robot):
    dispatcher = ToolDispatcher(robot=robot, console=QUIET, pose_reader=None)
    result = dispatcher.dispatch("go_where_pointed", {})
    assert result.ok is False
    assert "pose model" in result.payload["message"]


def test_a_broken_pose_reader_never_raises_into_the_model(robot):
    def explode(_jpeg):
        raise RuntimeError("weights missing")

    dispatcher = ToolDispatcher(robot=robot, console=QUIET, pose_reader=explode)
    result = dispatcher.dispatch("go_where_pointed", {})
    assert result.ok is False
    assert result.payload["message"]


def test_a_successful_read_turns_then_walks_and_reports_both(robot):
    reading = PointingResult(bearing_deg=25.0, distance_m=2.0, arm="left", confidence=0.9)
    spoken: list[str] = []
    dispatcher = ToolDispatcher(
        robot=robot,
        console=QUIET,
        speak=spoken.append,
        pose_reader=lambda _jpeg: reading,
    )

    result = dispatcher.dispatch("go_where_pointed", {})

    assert result.ok is True
    assert result.payload["bearing_deg"] == 25.0
    # The mock reports depth, so the distance must be a measurement.
    assert result.payload["distance_measured"] is True
    # And it announced its interpretation before moving.
    assert spoken and "pointing" in spoken[0]


def test_it_says_what_it_understood_before_it_moves(robot):
    # The whole point of announcing: a misread bearing is correctable by the
    # operator rather than acted on silently.
    order: list[str] = []
    reading = PointingResult(bearing_deg=40.0, distance_m=2.0, arm="left", confidence=0.9)

    original_move = robot.move

    def record_move(*args, **kwargs):
        order.append("move")
        return original_move(*args, **kwargs)

    robot.move = record_move
    dispatcher = ToolDispatcher(
        robot=robot,
        console=QUIET,
        speak=lambda text: order.append("speak"),
        pose_reader=lambda _jpeg: reading,
    )

    dispatcher.dispatch("go_where_pointed", {})

    assert order and order[0] == "speak"


def test_something_directly_in_the_way_means_stay_put(robot):
    reading = PointingResult(bearing_deg=0.0, distance_m=2.0, arm="left", confidence=0.9)
    robot.measure_distance = lambda bearing_deg=0.0: 0.5  # wall right there

    dispatcher = ToolDispatcher(
        robot=robot, console=QUIET, pose_reader=lambda _jpeg: reading
    )
    result = dispatcher.dispatch("go_where_pointed", {})

    assert result.ok is False
    assert "in the way" in result.payload["message"]
