"""Follow-me: target selection and the P-controller, plus thread lifecycle."""

from __future__ import annotations

import time

from spot_voice.robot.follow import (
    LOST_AFTER_SEC,
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


def test_losing_the_person_announces_it_once_and_sweeps_to_look():
    robot = RecordingRobot()
    spoken: list[str] = []
    controller = FollowController(robot, lambda: NeverSeesAnyone(), say=spoken.append)

    controller.start()
    time.sleep(LOST_AFTER_SEC + 0.8)
    controller.stop()

    assert spoken.count("I lost you.") == 1
    # It looks for you rather than giving up where it stands.
    assert any(command[2] != 0.0 for command in robot.commands)


def test_the_sweep_only_ever_turns_on_the_spot():
    # A robot that has lost its operator must not wander off looking. Rotation
    # is safe -- it stays where you left it; translation is not.
    robot = RecordingRobot()
    controller = FollowController(robot, lambda: NeverSeesAnyone())

    controller.start()
    time.sleep(1.2)
    controller.stop()

    assert robot.commands, "expected some commands"
    for v_x, v_y, _v_rot in robot.commands:
        assert v_x == 0.0 and v_y == 0.0


def test_the_sweep_respects_the_yaw_cap():
    from spot_voice.robot.follow import SEARCH_YAW_RATE
    from spot_voice.robot.limits import MAX_VROT

    assert 0 < SEARCH_YAW_RATE <= MAX_VROT
    robot = RecordingRobot()
    controller = FollowController(robot, lambda: NeverSeesAnyone())
    controller.start()
    time.sleep(0.6)
    controller.stop()

    assert all(abs(command[2]) <= MAX_VROT for command in robot.commands)


def test_the_sweep_gives_up_and_says_so_rather_than_spinning_forever():
    import spot_voice.robot.follow as follow_module

    robot = RecordingRobot()
    spoken: list[str] = []
    controller = FollowController(robot, lambda: NeverSeesAnyone(), say=spoken.append)

    original = follow_module.SEARCH_TIMEOUT_SEC
    follow_module.SEARCH_TIMEOUT_SEC = 0.5
    try:
        controller.start()
        time.sleep(1.5)
        controller.stop()
    finally:
        follow_module.SEARCH_TIMEOUT_SEC = original

    assert any("can't find you" in line for line in spoken)
    # And it stops turning once it has given up.
    assert robot.commands[-1] == (0.0, 0.0, 0.0)


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


# ----------------------------------------------------------------------
# Sticky tracking: once locked, the target must not silently swap


def test_acquisition_without_a_lock_picks_the_prominent_person():
    # The "stand in front of it when you say follow me" behaviour.
    you = _box_at(320, 0.5)
    someone_far = _box_at(100, 0.25)
    assert pick_target([someone_far, you], FRAME[0], previous=None) == you


def test_a_crosser_cannot_steal_the_lock():
    # The scenario the lock exists for: someone steps between Spot and its
    # target. Their box overlaps the target's and is much bigger (closer to the
    # camera). Without stickiness they would win on size alone.
    you_before = _box_at(320, 0.5)
    you_now = _box_at(330, 0.5)  # you, one frame later
    crosser = _box_at(325, 0.9)  # overlapping, far taller

    assert pick_target([crosser, you_now], FRAME[0], previous=you_before) == you_now


def test_full_occlusion_holds_still_rather_than_following_the_wrong_person():
    # You are fully hidden behind the crosser: only their (much taller) box is
    # visible. The controller must treat you as not-seen-this-frame, not
    # transfer the lock.
    you_before = _box_at(320, 0.5)
    crosser = _box_at(322, 0.9)

    assert pick_target([crosser], FRAME[0], previous=you_before) is None


def test_normal_walking_motion_keeps_the_lock():
    you_before = _box_at(320, 0.5)
    you_now = _box_at(360, 0.55)  # stepped sideways, slightly closer
    assert pick_target([you_now], FRAME[0], previous=you_before) == you_now


def test_a_distant_person_across_the_room_never_matches_the_lock():
    you_before = _box_at(320, 0.5)
    unrelated = _box_at(600, 0.5)  # same size, no overlap at all
    assert pick_target([unrelated], FRAME[0], previous=you_before) is None


def test_among_overlapping_matches_the_larger_overlap_wins():
    you_before = _box_at(320, 0.5)
    near_match = _box_at(324, 0.52)
    weak_match = _box_at(380, 0.5)
    picked = pick_target([weak_match, near_match], FRAME[0], previous=you_before)
    assert picked == near_match


def test_controller_reacquires_after_announcing_lost():
    # End to end: track someone, they vanish long enough for "I lost you",
    # then a person appears again -- the controller must lock on fresh.
    from spot_voice.robot.follow import LOST_AFTER_SEC

    class Vanishing:
        frame_size = (640, 480)

        def __init__(self) -> None:
            self.phase = "visible"

        def detect(self, _frame: bytes):
            if self.phase == "gone":
                return []
            return [_box_at(500, 0.3)]

    robot = RecordingRobot()
    spoken: list[str] = []
    detector = Vanishing()
    controller = FollowController(robot, lambda: detector, say=spoken.append)

    controller.start()
    time.sleep(0.4)  # tracking
    detector.phase = "gone"
    time.sleep(LOST_AFTER_SEC + 0.8)  # long enough to announce
    detector.phase = "visible"
    time.sleep(0.5)  # must re-acquire and drive again
    driving_after_reacquire = any(
        command != (0.0, 0.0, 0.0) for command in robot.commands[-3:]
    )
    controller.stop()

    assert spoken == ["I lost you."]
    assert driving_after_reacquire


# ----------------------------------------------------------------------
# Control telemetry
#
# TARGET_BBOX_HEIGHT_FRACTION and the achieved loop rate cannot be settled
# without the robot, and a P-controller tuned against guesses either hunts or
# lags with no way to tell which from the outside. These tests pin the log line
# that turns both into measurements.


def test_the_telemetry_line_reports_the_two_numbers_that_need_calibrating(caplog):
    import logging

    from spot_voice.robot.follow import TELEMETRY_PERIOD_SEC, _Telemetry, tracking_errors

    telemetry = _Telemetry(started=0.0)
    box = _box_at(FRAME[0] / 2, 0.40)  # a person standing too far away
    telemetry.record(*tracking_errors(box, FRAME), compute_velocity(box, FRAME))
    telemetry.tick(0.05)

    with caplog.at_level(logging.INFO, logger="spot_voice.robot.follow"):
        telemetry.maybe_log(TELEMETRY_PERIOD_SEC)

    line = caplog.text
    assert "bbox_frac=0.40" in line  # what to set the target to
    assert "Hz" in line  # whether the loop keeps up
    assert "worst cycle 50 ms" in line


def test_telemetry_stays_quiet_until_the_window_has_passed(caplog):
    import logging

    from spot_voice.robot.follow import TELEMETRY_PERIOD_SEC, _Telemetry

    telemetry = _Telemetry(started=0.0)
    telemetry.tick(0.05)

    with caplog.at_level(logging.INFO, logger="spot_voice.robot.follow"):
        telemetry.maybe_log(TELEMETRY_PERIOD_SEC * 0.5)

    assert caplog.text == ""


def test_telemetry_says_so_rather_than_dividing_by_zero_with_no_target(caplog):
    import logging

    from spot_voice.robot.follow import TELEMETRY_PERIOD_SEC, _Telemetry

    telemetry = _Telemetry(started=0.0)
    telemetry.tick(0.05)  # loop ran, but nobody was detected

    with caplog.at_level(logging.INFO, logger="spot_voice.robot.follow"):
        telemetry.maybe_log(TELEMETRY_PERIOD_SEC)

    assert "no target this second" in caplog.text


def test_the_errors_the_telemetry_reports_are_the_ones_the_controller_used():
    """A second copy of the formula would drift; there must be only one."""
    from spot_voice.robot.follow import KP_YAW, tracking_errors

    box = _box_at(FRAME[0] * 0.75, 0.40)  # well right of centre
    error_x, box_fraction, _error_size = tracking_errors(box, FRAME)
    velocity = compute_velocity(box, FRAME)

    assert abs(box_fraction - 0.40) < 0.01
    assert error_x > YAW_DEADBAND
    # The logged error is exactly what produced the logged command.
    assert abs(velocity.v_rot - (-KP_YAW * error_x)) < 1e-6
