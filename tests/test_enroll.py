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
