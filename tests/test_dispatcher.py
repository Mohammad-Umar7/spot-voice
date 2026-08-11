"""Tool dispatcher: every call returns {ok, message}, and nothing ever raises."""

from __future__ import annotations

import pytest
from rich.console import Console

from spot_voice.brain.dispatcher import ToolDispatcher
from spot_voice.brain.tools import TOOL_NAMES
from spot_voice.robot.mock import MockSpot

QUIET = Console(quiet=True)


@pytest.fixture()
def robot() -> MockSpot:
    spot = MockSpot(dock_id=None, console=QUIET)
    spot.connect()
    return spot


class FakeFollow:
    def __init__(self) -> None:
        self._active = False
        self.started = 0
        self.stopped = 0

    @property
    def active(self) -> bool:
        return self._active

    def start(self):
        self.started += 1
        self._active = True
        return True, "Following you."

    def stop(self, timeout: float = 2.0):
        self.stopped += 1
        was = self._active
        self._active = False
        return True, "Stopped following." if was else "I wasn't following anyone."


@pytest.fixture()
def dispatcher(robot: MockSpot):
    spoken: list[str] = []
    follow = FakeFollow()
    instance = ToolDispatcher(
        robot=robot, follow=follow, speak=spoken.append, console=QUIET
    )
    instance.spoken = spoken  # type: ignore[attr-defined]
    instance.follow = follow  # type: ignore[attr-defined]
    return instance


# ----------------------------------------------------------------------
# Coverage of the declared tool surface


def test_dispatcher_handles_exactly_the_declared_tools(dispatcher):
    assert dispatcher.handled_tools == TOOL_NAMES


def test_unknown_tool_is_reported_not_raised(dispatcher):
    result = dispatcher.dispatch("self_destruct", {})
    assert result.ok is False
    assert "don't have a tool" in result.payload["message"]


# ----------------------------------------------------------------------
# Result shape


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("stand", {}),
        ("sit", {}),
        ("stop_all", {}),
        ("get_status", {}),
        ("list_waypoints", {}),
        ("capture_image", {"camera": "front"}),
        ("move", {"direction": "forward", "distance_m": 0.2}),
        ("navigate_to", {"waypoint_name": "entrance"}),
        ("start_follow", {}),
        ("stop_follow", {}),
        ("speak", {"text": "hello"}),
        ("dock", {}),
        ("undock", {}),
    ],
)
def test_every_tool_returns_ok_and_message(dispatcher, name, arguments):
    result = dispatcher.dispatch(name, arguments)
    assert set(result.payload) >= {"ok", "message"}
    assert isinstance(result.payload["ok"], bool)
    assert isinstance(result.payload["message"], str)
    assert result.payload["message"]
    assert result.duration_ms >= 0


def test_payload_is_json_serialisable(dispatcher):
    import json

    for name in ("get_status", "list_waypoints", "capture_image"):
        result = dispatcher.dispatch(name, {})
        json.dumps(result.payload)  # must not raise


# ----------------------------------------------------------------------
# Malformed model output


@pytest.mark.parametrize("bad_input", [None, "not a dict", 42, [], {"unexpected": True}])
def test_malformed_tool_input_never_raises(dispatcher, bad_input):
    for name in sorted(TOOL_NAMES):
        result = dispatcher.dispatch(name, bad_input)
        assert isinstance(result.ok, bool)
        assert result.payload["message"]


def test_move_without_direction_asks_for_one(dispatcher):
    result = dispatcher.dispatch("move", {})
    assert result.ok is False
    assert "direction" in result.payload["message"]


def test_move_with_nonsense_numbers_is_handled(dispatcher):
    result = dispatcher.dispatch(
        "move", {"direction": "forward", "distance_m": "a couple", "speed": None}
    )
    assert isinstance(result.ok, bool)
    assert result.payload["message"]


def test_navigate_to_unknown_place_lists_what_it_knows(dispatcher):
    result = dispatcher.dispatch("navigate_to", {"waypoint_name": "the moon"})
    assert result.ok is False
    assert "I know" in result.payload["message"]


def test_capture_image_rejects_an_unknown_camera(dispatcher):
    result = dispatcher.dispatch("capture_image", {"camera": "rear"})
    assert result.ok is False
    assert result.image_jpeg is None


# ----------------------------------------------------------------------
# Behaviour


def test_capture_image_returns_jpeg_bytes_but_keeps_them_out_of_the_payload(dispatcher):
    result = dispatcher.dispatch("capture_image", {"camera": "front"})
    assert result.ok
    assert result.image_jpeg and result.image_jpeg[:2] == b"\xff\xd8"  # JPEG SOI
    assert "image_jpeg" not in result.payload


def test_speak_tool_routes_to_the_speaker(dispatcher):
    result = dispatcher.dispatch("speak", {"text": "checking the panel"})
    assert result.ok
    assert dispatcher.spoken == ["checking the panel"]


def test_speak_tool_rejects_empty_text(dispatcher):
    assert dispatcher.dispatch("speak", {"text": "   "}).ok is False


def test_motion_tools_stop_follow_first(dispatcher):
    dispatcher.dispatch("start_follow", {})
    assert dispatcher.follow.active
    dispatcher.dispatch("move", {"direction": "forward", "distance_m": 0.1})
    assert dispatcher.follow.stopped >= 1
    assert not dispatcher.follow.active


def test_get_status_reports_follow_state(dispatcher):
    dispatcher.dispatch("start_follow", {})
    result = dispatcher.dispatch("get_status", {})
    assert result.payload["following"] is True


def test_a_raising_robot_becomes_a_failed_result(dispatcher):
    class Exploding:
        def stop_all(self):
            raise ConnectionError("wifi dropped")

    broken = ToolDispatcher(robot=Exploding(), console=QUIET)  # type: ignore[arg-type]
    result = broken.dispatch("stop_all", {})
    assert result.ok is False
    assert "connection" in result.payload["message"].lower()
