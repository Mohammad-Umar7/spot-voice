"""Enrolment: get the robot upright before photographing anyone.

These exist because of a real failed run. Spot was sitting, so its front
cameras were at knee height pointing at the floor, every sample came back "no
face found", and the advice printed -- stand closer, better light -- sent the
operator looking in completely the wrong place.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from spot_voice.enroll import enroll
from spot_voice.robot.base import ActionResult, fail, ok

QUIET = Console(quiet=True)


class FakeConfig:
    def __init__(self, tmp_path: Path) -> None:
        self.face_store_path = tmp_path / "faces.json"


class FakeRobot:
    """Records the order things happened in."""

    def __init__(self, docked: bool = False, can_stand: bool = True) -> None:
        self._docked = docked
        self._can_stand = can_stand
        self.calls: list[str] = []

    def is_docked(self) -> bool:
        return self._docked

    def stand(self) -> ActionResult:
        self.calls.append("stand")
        return ok("Standing.") if self._can_stand else fail("Motors are off.")

    def capture_image(self, camera: str = "front") -> ActionResult:
        self.calls.append("capture")
        return ActionResult(True, "frame", None, image_jpeg=b"\xff\xd8fake")


@pytest.fixture()
def config(tmp_path: Path) -> FakeConfig:
    return FakeConfig(tmp_path)


def test_a_docked_robot_is_refused_rather_than_photographed(config, monkeypatch):
    """On the dock the cameras are low and aimed wrong, and undocking is a
    bigger action than this command should take on its own."""
    robot = FakeRobot(docked=True)

    code = enroll("awaiz", config, QUIET, robot, samples=1, gap=0.0)

    assert code == 2
    assert "capture" not in robot.calls, "must not sample from the dock"
    assert "stand" not in robot.calls, "must not stand up on a charger"


def test_a_sitting_robot_is_stood_up_before_any_frame_is_taken(config, monkeypatch):
    robot = FakeRobot(docked=False)
    monkeypatch.setattr("spot_voice.enroll.STAND_SETTLE_SEC", 0.0)

    enroll("awaiz", config, QUIET, robot, samples=1, gap=0.0)

    assert "stand" in robot.calls
    assert robot.calls.index("stand") < robot.calls.index("capture"), (
        "frames were taken before the robot was upright"
    )


def test_a_robot_that_cannot_stand_says_so_instead_of_sampling_blind(config):
    robot = FakeRobot(docked=False, can_stand=False)

    code = enroll("awaiz", config, QUIET, robot, samples=1, gap=0.0)

    assert code == 1
    assert "capture" not in robot.calls


def test_a_nameless_enrolment_is_refused_before_touching_the_robot(config):
    robot = FakeRobot()

    assert enroll("  ", config, QUIET, robot) == 2
    assert robot.calls == []


def test_enrolment_without_a_robot_refuses_rather_than_using_a_webcam(config):
    """A webcam sample would enrol fine and then fail to match Spot's fisheye."""
    assert enroll("awaiz", config, QUIET, None) == 2


# ----------------------------------------------------------------------
# Choosing which face in the frame is the subject
#
# Refusing whenever a second face appeared was the original rule and it did not
# survive a real workplace: colleagues walked through shot, four of five samples
# were discarded, and the identity ended up built from one frame -- exactly the
# brittleness the five prompts exist to prevent.


def _face(box, embedding):
    return (box, embedding)


ME = [1.0, 0.0, 0.0]
SOMEONE_ELSE = [0.0, 1.0, 0.0]


def test_a_lone_face_is_taken():
    from spot_voice.enroll import pick_subject

    chosen, _ = pick_subject([_face((0, 0, 100, 100), ME)])
    assert chosen is not None


def test_the_near_face_wins_because_the_subject_stands_closest():
    from spot_voice.enroll import pick_subject

    near = _face((0, 0, 200, 200), ME)          # a metre away
    far = _face((400, 0, 450, 50), SOMEONE_ELSE)  # across the room

    chosen, _ = pick_subject([far, near])
    assert chosen == near


def test_two_faces_at_the_same_distance_are_refused_not_guessed():
    """Genuinely ambiguous, and enrolling the wrong person is unrecoverable."""
    from spot_voice.enroll import pick_subject

    chosen, reason = pick_subject(
        [_face((0, 0, 200, 200), ME), _face((300, 0, 495, 195), SOMEONE_ELSE)]
    )
    assert chosen is None
    assert "same distance" in reason


def test_once_one_sample_is_accepted_later_frames_must_match_it():
    """The guard that makes picking-by-size safe: a wrong pick cannot stick."""
    from spot_voice.enroll import pick_subject

    # A closer stranger would otherwise win on size alone.
    stranger_up_close = _face((0, 0, 400, 400), SOMEONE_ELSE)
    me_further_back = _face((400, 0, 500, 100), ME)

    chosen, _ = pick_subject([stranger_up_close, me_further_back], anchor=ME)
    assert chosen == me_further_back


def test_a_frame_without_the_anchored_person_is_skipped():
    from spot_voice.enroll import pick_subject

    chosen, reason = pick_subject([_face((0, 0, 200, 200), SOMEONE_ELSE)], anchor=ME)
    assert chosen is None
    assert "same person" in reason


def test_two_faces_both_matching_the_anchor_are_refused():
    from spot_voice.enroll import pick_subject

    chosen, reason = pick_subject(
        [_face((0, 0, 200, 200), ME), _face((300, 0, 400, 100), ME)], anchor=ME
    )
    assert chosen is None
    assert "both look like you" in reason


# ----------------------------------------------------------------------
# Enrolling from photos


def test_a_missing_photo_folder_is_reported(config):
    from spot_voice.enroll import enroll_from_photos

    assert enroll_from_photos("awaiz", config, QUIET, "/nope/not/here") == 2


def test_a_folder_with_no_images_is_reported(config, tmp_path):
    from spot_voice.enroll import enroll_from_photos

    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    assert enroll_from_photos("awaiz", config, QUIET, tmp_path) == 2


def test_photo_enrolment_takes_no_robot_argument_at_all(config):
    """The whole point: no connection, no lease, no robot in the room."""
    import inspect

    from spot_voice.enroll import enroll_from_photos

    parameters = inspect.signature(enroll_from_photos).parameters
    assert "robot" not in parameters
    assert list(parameters) == ["name", "config", "console", "folder"]
