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

from ..robot.base import ActionResult, RobotInterface, fail, ok
from ..robot.errors import to_speakable
from .tools import TOOL_NAMES

LOGGER = logging.getLogger(__name__)

#: How many headings a 360 scan looks from. One front frame covers roughly a
#: 100 degree wedge, so four gives full cover with a little overlap -- and each
#: extra heading costs a turn plus a vision call, which the operator waits out.
SCAN_HEADINGS = 4

#: Pause after each turn so the frame is taken from a settled body rather than
#: one still swaying, which blurs it and loses small detections.
SCAN_SETTLE_SEC = 0.8



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
        describe_image: Callable[[bytes], str] | None = None,
    ) -> None:
        self._robot = robot
        self._follow = follow
        self._speak = speak
        self._console = console or Console()
        # Turns one frame into words. Needed by scan_room, which produces
        # several frames in a single call and so cannot use the agent's
        # one-image-per-tool-result path.
        self._describe_image = describe_image
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
            "come_here": self._come_here,
            "navigate_to": self._navigate_to,
            "list_waypoints": self._list_waypoints,
            "capture_image": self._capture_image,
            "scan_room": self._scan_room,
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

    def _scan_room(self, arguments: dict[str, Any]) -> ActionResult:
        """Turn on the spot, look at each heading, and report what is there.

        One camera frame covers roughly a 100 degree wedge, so "what is in this
        room" was previously unanswerable -- there was no tool for it, and the
        model answered anyway. "Zero people in the room" was invented, not seen.

        Two deliberate choices about counting. People are counted by the local
        person detector rather than by asking the vision model how many it can
        see: detectors are built for exactly that and vision models are famously
        unreliable at counting. And the total is reported as a floor, "at least
        N", because the headings overlap slightly and a person standing in the
        seam appears twice -- claiming an exact count would be a false
        precision that someone might act on.
        """
        if self._follow is not None and getattr(self._follow, "active", False):
            self._follow.stop()

        headings = SCAN_HEADINGS
        seen: list[dict[str, Any]] = []
        best_count = 0

        for index in range(headings):
            if index > 0:
                turn = self._robot.move(
                    direction="turn_left", degrees=360.0 / headings
                )
                if not turn.ok:
                    return fail(f"I couldn't turn to look around. {turn.message}")
                time.sleep(SCAN_SETTLE_SEC)  # let the body stop swaying

            capture = self._robot.capture_image("front")
            if not capture.ok or not capture.image_jpeg:
                LOGGER.debug("scan frame %d failed: %s", index, capture.message)
                continue

            people = self._count_people(capture.image_jpeg)
            description = ""
            if self._describe_image is not None:
                try:
                    description = self._describe_image(capture.image_jpeg)
                except Exception as exc:
                    LOGGER.warning("scan description failed", exc_info=True)
                    description = f"(could not describe: {type(exc).__name__})"

            best_count = max(best_count, people if people is not None else 0)
            entry: dict[str, Any] = {"heading_deg": round(index * 360.0 / headings)}
            if people is not None:
                entry["people_detected"] = people
            if description:
                entry["view"] = description
            seen.append(entry)
            self._console.print(
                f"[blue]scan[/blue] [dim]{entry['heading_deg']}deg "
                f"people={people} {description[:60]}[/dim]"
            )

        if not seen:
            return fail("I turned all the way round but couldn't get any camera frames.")

        total = sum(entry.get("people_detected", 0) for entry in seen)
        return ok(
            f"Looked all the way round from {len(seen)} headings.",
            headings=seen,
            people_seen_total=total,
            people_note=(
                "Counts come from a person detector, one frame per heading. "
                "Headings overlap slightly, so report this as 'at least' rather "
                "than an exact number, and say it is what you could see rather "
                "than what is in the room."
            ),
        )

    def _count_people(self, frame_jpeg: bytes) -> int | None:
        """How many people the local detector finds. None if it cannot run."""
        if self._follow is None:
            return None
        detector = getattr(self._follow, "person_detector", None)
        if detector is None:
            return None
        try:
            return len(detector.detect(frame_jpeg))
        except Exception:
            LOGGER.debug("person count failed", exc_info=True)
            return None

    def _come_here(self, arguments: dict[str, Any]) -> ActionResult:
        """Walk to the operator and settle beside or in front of them.

        The counterpart to :meth:`_go_where_pointed`. That one needs a gesture
        because it targets somewhere away from the person; this one needs none,
        because the person *is* the target and their own body is the reference.

        Walking toward a person is the one direction worth being careful in, so
        there are four guards: the approach stops a conversational distance
        short rather than at them, an implausible distance estimate is refused
        outright instead of walked, Spot says what it understood before moving,
        and Spot's own obstacle avoidance treats the person as an obstacle
        regardless of any of the above.
        """
        from ..vision.proxemics import MAX_APPROACH_M, approach_person

        position = str(arguments.get("position") or "in_front").strip().lower()
        posture = str(arguments.get("posture") or "stand").strip().lower()
        if position not in {"in_front", "beside"}:
            position = "in_front"
        if posture not in {"stand", "sit"}:
            posture = "stand"

        if self._follow is not None and self._follow.active:
            self._follow.stop()

        if self._follow is None:
            return fail("I can't look for you right now -- my person detector isn't set up.")
        try:
            located = self._follow.locate_operator()
        except Exception as exc:
            LOGGER.warning("locate_operator failed", exc_info=True)
            return fail(f"I couldn't look for you. {type(exc).__name__}.")

        if located is None:
            return fail(
                "I can't see you. Stand where I'm facing and ask again, or tell "
                "me where to go instead."
            )

        bearing_deg, distance_m = located
        plan = approach_person(bearing_deg, distance_m, position=position)
        if plan is None:
            return fail(
                f"I think you're more than {MAX_APPROACH_M:.0f} metres away, which "
                "is too far for me to judge accurately. Come closer or point "
                "where you want me."
            )

        # Say it before doing it. This walks toward a person, so an operator who
        # hears the wrong interpretation needs the chance to say stop first.
        if self._speak is not None:
            try:
                self._speak(plan.summary)
            except Exception:
                LOGGER.debug("could not announce the approach", exc_info=True)

        if not plan.already_there:
            moved = self._walk_to(plan)
            if not moved.ok:
                return moved

        if posture == "sit":
            settled = self._robot.sit()
            return settled if not settled.ok else ok(f"{plan.summary} Sitting.")
        return ok(plan.summary)

    def _walk_to(self, plan) -> ActionResult:
        """Drive one approach, preferring a trajectory goal over a blind walk.

        ``walk_toward`` hands the goal to Spot's own planner, which routes round
        anything in the way. The turn-then-walk fallback cannot do that, so it
        is only used when the robot has no trajectory support.
        """
        try:
            if self._robot.walk_toward(
                plan.bearing_deg, plan.distance_m, plan.standoff_m
            ):
                return ok("On my way.")
        except Exception:
            LOGGER.debug("walk_toward failed; falling back", exc_info=True)

        if abs(plan.bearing_deg) >= 8:
            turn = self._robot.move(
                direction="turn_left" if plan.bearing_deg > 0 else "turn_right",
                degrees=abs(plan.bearing_deg),
            )
            if not turn.ok:
                return turn
        remaining = max(0.0, plan.distance_m - plan.standoff_m)
        if remaining < 0.1:
            return ok("Close enough.")
        return self._robot.move(direction="forward", distance_m=remaining)

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
