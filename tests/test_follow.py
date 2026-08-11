"""Follow-me: target selection and the P-controller, plus thread lifecycle."""

from __future__ import annotations

import time

from spot_voice.robot.follow import (
    FORWARD_DEADBAND,
    TARGET_BBOX_HEIGHT_FRACTION,
    YAW_DEADBAND,
    FollowController,
    compute_velocity,
    pick_target,
)
from spot_voice.robot.limits import MAX_VROT, MAX_VX

FRAME = (640, 480)


def _box_at(centre_x: float, height_fraction: float) -> tuple[int, int, int, int, float]:
    """Build a bounding box centred at ``centre_x`` with the given height."""
    height = height_fraction * FRAME[1]
    width = height * 0.4
    return (
        int(centre_x - width / 2),
        int(FRAME[1] / 2 - height / 2),
        int(centre_x + width / 2),
        int(FRAME[1] / 2 + height / 2),
        0.9,
    )


# ----------------------------------------------------------------------
# Target selection


def test_no_boxes_means_no_target():
    assert pick_target([], FRAME[0]) is None


def test_prefers_the_larger_person():
    small = _box_at(320, 0.2)
    large = _box_at(320, 0.6)
    assert pick_target([small, large], FRAME[0]) == large


def test_prefers_the_more_centred_person_at_similar_size():
    centred = _box_at(320, 0.5)
    edge = _box_at(40, 0.5)
    assert pick_target([edge, centred], FRAME[0]) == centred


# ----------------------------------------------------------------------
# Control law


def test_centred_at_standoff_holds_still():
    box = _box_at(320, TARGET_BBOX_HEIGHT_FRACTION)
    velocity = compute_velocity(box, FRAME)
    assert velocity.as_tuple() == (0.0, 0.0, 0.0)


def test_person_to_the_right_turns_clockwise():
    velocity = compute_velocity(_box_at(560, TARGET_BBOX_HEIGHT_FRACTION), FRAME)
    assert velocity.v_rot < 0


def test_person_to_the_left_turns_counter_clockwise():
    velocity = compute_velocity(_box_at(80, TARGET_BBOX_HEIGHT_FRACTION), FRAME)
    assert velocity.v_rot > 0


def test_small_offsets_inside_the_deadband_do_not_twitch():
    offset_px = (YAW_DEADBAND * 0.5) * (FRAME[0] / 2)
    velocity = compute_velocity(
        _box_at(320 + offset_px, TARGET_BBOX_HEIGHT_FRACTION), FRAME
    )
    assert velocity.v_rot == 0.0


def test_distant_person_is_approached():
    velocity = compute_velocity(_box_at(320, TARGET_BBOX_HEIGHT_FRACTION * 0.4), FRAME)
    assert velocity.v_x > 0


def test_close_person_stops_rather_than_reversing():
    velocity = compute_velocity(_box_at(320, TARGET_BBOX_HEIGHT_FRACTION * 1.6), FRAME)
    assert velocity.v_x == 0.0


def test_size_errors_inside_the_deadband_do_not_creep():
    fraction = TARGET_BBOX_HEIGHT_FRACTION * (1 - FORWARD_DEADBAND * 0.5)
    assert compute_velocity(_box_at(320, fraction), FRAME).v_x == 0.0


def test_extreme_error_still_respects_the_hard_caps():
    velocity = compute_velocity((0, 0, 5, 5, 0.9), FRAME)  # tiny box, far left
    assert abs(velocity.v_x) <= MAX_VX
    assert abs(velocity.v_rot) <= MAX_VROT


def test_degenerate_frame_size_does_not_divide_by_zero():
    velocity = compute_velocity(_box_at(320, 0.5), (0, 0))
    assert abs(velocity.v_x) <= MAX_VX
    assert abs(velocity.v_rot) <= MAX_VROT


# ----------------------------------------------------------------------
# Thread lifecycle


class RecordingRobot:
    """Minimal robot stub that records velocity commands."""

    def __init__(self) -> None:
        self.commands: list[tuple[float, float, float]] = []

    def capture_image(self, camera: str = "front"):
        from spot_voice.robot.base import ActionResult

        return ActionResult(True, "frame", None, image_jpeg=b"\xff\xd8fake")

    def drive(self, v_x: float, v_y: float, v_rot: float) -> None:
        self.commands.append((v_x, v_y, v_rot))


class AlwaysSeesSomeone:
    frame_size = (640, 480)

    def detect(self, _frame: bytes):
        return [_box_at(500, 0.3)]


class NeverSeesAnyone:
    frame_size = (640, 480)

    def detect(self, _frame: bytes):
        return []


def test_controller_drives_while_following_and_zeroes_on_stop():
    robot = RecordingRobot()
    controller = FollowController(robot, lambda: AlwaysSeesSomeone())

    ok, message = controller.start()
    assert ok and message
    time.sleep(0.5)
    assert controller.active
    assert any(command != (0.0, 0.0, 0.0) for command in robot.commands)

    controller.stop()
    assert not controller.active
    assert robot.commands[-1] == (0.0, 0.0, 0.0)


def test_losing_the_person_announces_it_once_and_holds_still():
    robot = RecordingRobot()
    spoken: list[str] = []
    controller = FollowController(robot, lambda: NeverSeesAnyone(), say=spoken.append)

    controller.start()
    time.sleep(2.6)  # LOST_AFTER_SEC is 2.0
    controller.stop()

    assert spoken.count("I lost you.") == 1
    assert all(command == (0.0, 0.0, 0.0) for command in robot.commands)


def test_starting_twice_is_idempotent():
    controller = FollowController(RecordingRobot(), lambda: AlwaysSeesSomeone())
    controller.start()
    ok, message = controller.start()
    controller.stop()
    assert ok and "already" in message


def test_stopping_when_idle_is_safe():
    controller = FollowController(RecordingRobot(), lambda: AlwaysSeesSomeone())
    ok, message = controller.stop()
    assert ok and message


def test_a_broken_detector_is_reported_not_raised():
    def explode():
        raise ImportError("ultralytics is not installed")

    controller = FollowController(RecordingRobot(), explode)
    ok, message = controller.start()
    assert ok is False
    assert "detector" in message.lower()
