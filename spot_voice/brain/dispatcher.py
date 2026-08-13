"""The single place tool calls turn into robot actions.

Everything Claude asks for funnels through :meth:`ToolDispatcher.dispatch`. That
gives one choke point for logging, argument validation and the hard rule that
**no exception ever reaches the model**: a failure comes back as
``{"ok": false, "message": "..."}`` in exactly the shape a success does.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console

from ..robot.base import ActionResult, RobotInterface, fail
from ..robot.errors import to_speakable
from .tools import TOOL_NAMES

LOGGER = logging.getLogger(__name__)


@dataclass
class DispatchResult:
    """Outcome of one tool call."""

    name: str
    ok: bool
    payload: dict[str, Any]
    duration_ms: float
    image_jpeg: bytes | None = field(default=None, repr=False)


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce a tool input into a dict, tolerating malformed model output."""
    return value if isinstance(value, dict) else {}


def _as_number(value: Any) -> float | None:
    """Coerce a JSON value into a float, or ``None`` when it isn't numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _as_text(value: Any) -> str:
    """Coerce a JSON value into a stripped string."""
    if value is None or isinstance(value, (dict, list)):
        return ""
    return str(value).strip()


class ToolDispatcher:
    """Executes Claude's tool calls against the robot layer.

    Args:
        robot: The robot to command (mock or real -- identical interface).
        follow: The follow-me controller.
        speak: Callback used by the ``speak`` tool. Blocking is fine.
        console: Rich console for the human-readable tool log.
    """

    def __init__(
        self,
        robot: RobotInterface,
        follow=None,
        speak: Callable[[str], None] | None = None,
        console: Console | None = None,
        pose_reader: Callable[[bytes], Any] | None = None,
    ) -> None:
        self._robot = robot
        self._follow = follow
        self._speak = speak
        self._console = console or Console()
        # Reads a pointing gesture from a JPEG. Injected so the pose model is
        # only loaded when something actually needs it.
        self._pose_reader = pose_reader
        self._handlers: dict[str, Callable[[dict[str, Any]], ActionResult]] = {
            "power_on": self._power_on,
            "stand": self._stand,
            "sit": self._sit,
            "emote": self._emote,
            "move": self._move,
            "go_where_pointed": self._go_where_pointed,
            "navigate_to": self._navigate_to,
            "list_waypoints": self._list_waypoints,
            "start_follow": self._start_follow,
            "stop_follow": self._stop_follow,
            "capture_image": self._capture_image,
            "get_status": self._get_status,
            "dock": self._dock,
            "undock": self._undock,
            "speak": self._speak_tool,
            "stop_all": self._stop_all,
        }

    # ------------------------------------------------------------------

    @property
    def handled_tools(self) -> frozenset[str]:
        """Names this dispatcher can execute."""
        return frozenset(self._handlers)

    def dispatch(self, name: str, tool_input: Any) -> DispatchResult:
        """Execute one tool call. Never raises.

        Args:
            name: Tool name from the ``tool_use`` block.
            tool_input: Tool input from the ``tool_use`` block.

        Returns:
            A :class:`DispatchResult` whose ``payload`` is safe to hand back to
            the model as a ``tool_result``.
        """
        started = time.perf_counter()
        arguments = _as_dict(tool_input)
        self._console.print(
            f"[magenta]tool >[/magenta] [bold]{name}[/bold]"
            + (f" [dim]{arguments}[/dim]" if arguments else "")
        )

        handler = self._handlers.get(name)
        if handler is None:
            known = ", ".join(sorted(TOOL_NAMES))
            result = fail(f"I don't have a tool called {name}. I have: {known}.")
        else:
            try:
                result = handler(arguments)
            except Exception as exc:  # the model must never see a traceback
                LOGGER.exception("Tool %s raised", name)
                result = fail(to_speakable(exc))

        duration_ms = (time.perf_counter() - started) * 1000.0
        marker = "[green]ok[/green]" if result.ok else "[red]failed[/red]"
        self._console.print(
            f"[magenta]tool <[/magenta] [bold]{name}[/bold] {marker} "
            f"[dim]{duration_ms:.0f} ms[/dim] {result.message}"
        )
        LOGGER.info(
            "tool=%s ok=%s ms=%.0f message=%s", name, result.ok, duration_ms, result.message
        )
        return DispatchResult(
            name=name,
            ok=result.ok,
            payload=result.to_model_payload(),
            duration_ms=duration_ms,
            image_jpeg=result.image_jpeg,
        )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _power_on(self, _arguments: dict[str, Any]) -> ActionResult:
        return self._robot.power_on()

    def _stand(self, _arguments: dict[str, Any]) -> ActionResult:
        if self._follow is not None and self._follow.active:
            self._follow.stop()
        return self._robot.stand()

    def _sit(self, _arguments: dict[str, Any]) -> ActionResult:
        if self._follow is not None and self._follow.active:
            self._follow.stop()
        return self._robot.sit()

    def _emote(self, arguments: dict[str, Any]) -> ActionResult:
        gesture = _as_text(arguments.get("gesture")).lower()
        if not gesture:
            return fail("I need to know which gesture to do.")
        # Gestures are stand-only, so stop walking first.
        if self._follow is not None and self._follow.active:
            self._follow.stop()
        return self._robot.emote(gesture)

    def _move(self, arguments: dict[str, Any]) -> ActionResult:
        direction = _as_text(arguments.get("direction")).lower()
        if not direction:
            return fail(
                "I need a direction to move: forward, back, left, right, "
                "turn left or turn right."
            )
        if self._follow is not None and self._follow.active:
            self._follow.stop()
        return self._robot.move(
            direction=direction,
            distance_m=_as_number(arguments.get("distance_m")),
            degrees=_as_number(arguments.get("degrees")),
            speed=_as_number(arguments.get("speed")),
        )

    def _go_where_pointed(self, _arguments: dict[str, Any]) -> ActionResult:
        """Read a pointing gesture, measure the distance, and walk there.

        The gesture supplies the bearing, which it does reliably. It supplies no
        distance at all -- so that comes from the depth sensor, stopping short of
        whatever is in the way. Spot says what it understood before moving, so a
        misread is corrected rather than acted upon.
        """
        from ..vision.pointing import apply_measured_distance

        if self._pose_reader is None:
            return fail("I can't see gestures right now -- my pose model isn't loaded.")
        if self._follow is not None and self._follow.active:
            self._follow.stop()

        capture = self._robot.capture_image("front")
        if not capture.ok or not capture.image_jpeg:
            return fail(f"I couldn't get a camera frame. {capture.message}")

        try:
            reading = self._pose_reader(capture.image_jpeg)
        except Exception as exc:
            LOGGER.warning("pose read failed", exc_info=True)
            return fail(f"I couldn't read the gesture. {type(exc).__name__}.")

        if reading is None:
            return fail(
                "I can't see anyone pointing. Hold your arm out towards where "
                "you want me and ask again."
            )

        measured = None
        try:
            measured = self._robot.measure_distance(reading.bearing_deg)
        except Exception:
            LOGGER.debug("depth measurement failed", exc_info=True)
        reading = apply_measured_distance(reading, measured)

        if reading.distance_m < 0.3:
            return fail(
                f"You're pointing {reading.describe()}, but there's something in "
                "the way. I'll stay here."
            )

        # Say it before doing it: the bearing can be misread, and hearing the
        # interpretation is what lets the operator stop it.
        summary = f"You're pointing {reading.describe()}. Going there."
        if self._speak is not None:
            try:
                self._speak(summary)
            except Exception:
                LOGGER.debug("could not announce the gesture", exc_info=True)

        if abs(reading.bearing_deg) >= 8:
            turn = self._robot.move(
                direction="turn_left" if reading.bearing_deg > 0 else "turn_right",
                degrees=abs(reading.bearing_deg),
            )
            if not turn.ok:
                return turn

        walk = self._robot.move(direction="forward", distance_m=reading.distance_m)
        if not walk.ok:
            return walk
        return ActionResult(
            True,
            f"I'm where you pointed, {reading.describe()}.",
            {
                "bearing_deg": round(reading.bearing_deg, 1),
                "distance_m": round(reading.distance_m, 2),
                "distance_measured": reading.distance_measured,
                "arm": reading.arm,
                "confidence": round(reading.confidence, 2),
            },
        )

    def _navigate_to(self, arguments: dict[str, Any]) -> ActionResult:
        name = _as_text(arguments.get("waypoint_name"))
        if not name:
            return fail("I need the name of a place to walk to.")
        if self._follow is not None and self._follow.active:
            self._follow.stop()
        return self._robot.navigate_to(name)

    def _list_waypoints(self, _arguments: dict[str, Any]) -> ActionResult:
        return self._robot.list_waypoints()

    def _start_follow(self, _arguments: dict[str, Any]) -> ActionResult:
        if self._follow is None:
            return fail("Follow-me isn't available right now.")
        stand = self._robot.stand()
        if not stand.ok:
            return stand
        ok_flag, message = self._follow.start()
        return ActionResult(ok_flag, message)

    def _stop_follow(self, _arguments: dict[str, Any]) -> ActionResult:
        if self._follow is None:
            return fail("Follow-me isn't available right now.")
        ok_flag, message = self._follow.stop()
        # The last velocity command can still have time left on it; settle.
        self._robot.stop_all()
        return ActionResult(ok_flag, message)

    def _capture_image(self, arguments: dict[str, Any]) -> ActionResult:
        camera = _as_text(arguments.get("camera")).lower() or "front"
        return self._robot.capture_image(camera)

    def _get_status(self, _arguments: dict[str, Any]) -> ActionResult:
        result = self._robot.get_status()
        if result.data is not None and self._follow is not None:
            result.data["following"] = self._follow.active
        return result

    def _dock(self, _arguments: dict[str, Any]) -> ActionResult:
        if self._follow is not None and self._follow.active:
            self._follow.stop()
        return self._robot.dock()

    def _undock(self, _arguments: dict[str, Any]) -> ActionResult:
        return self._robot.undock()

    def _speak_tool(self, arguments: dict[str, Any]) -> ActionResult:
        text = _as_text(arguments.get("text"))
        if not text:
            return fail("There was nothing to say.")
        if self._speak is None:
            return fail("My speaker isn't available.")
        # Fire and forget. Synthesis plus playback is 4-6 seconds; blocking the
        # tool loop on it meant a single "stand up" spent longer talking about
        # itself than acting. The Speaker serialises internally, so queued
        # utterances still play in order.
        threading.Thread(
            target=self._speak_off_thread, args=(text,), daemon=True
        ).start()
        return ActionResult(True, "Speaking now.", {"spoken": text})

    def _speak_off_thread(self, text: str) -> None:
        """Speak without holding up the tool loop. Never raises."""
        try:
            self._speak(text)  # type: ignore[misc]
        except Exception:
            LOGGER.warning("speak tool failed", exc_info=True)

    def _stop_all(self, _arguments: dict[str, Any]) -> ActionResult:
        if self._follow is not None and self._follow.active:
            self._follow.stop()
        return self._robot.stop_all()
