"""A simulated Spot, good enough to rehearse the whole experience at a desk.

Mock mode is first-class and stays that way: every tool, the reflex lane, the
Claude loop, follow-me and TTS all run for real. Only the SDK calls are replaced
with logging plus a small state machine. Nothing here imports ``bosdyn``, so the
whole program runs on a laptop with no Spot SDK installed.
"""

from __future__ import annotations

import math
import random
import threading
import time
from pathlib import Path

from rich.console import Console

from .base import CAMERA_SOURCES, ActionResult, RobotInterface, fail, ok
from .limits import VELOCITY_CMD_DURATION, clamp_velocity
from .motion import match_waypoint, plan_move

_CONSOLE = Console()

#: Waypoints the mock pretends to know when no real map is available.
DEFAULT_MOCK_WAYPOINTS = [
    "entrance",
    "assembly-line",
    "control-panel",
    "compressor-room",
    "loading-bay",
    "dock",
]

#: Where the fallback camera frame lives, relative to the package.
_ASSET_IMAGE = Path(__file__).resolve().parent.parent / "assets" / "test_scene.jpg"

#: Minimum gap between repeats of a throttled log line.
LOG_THROTTLE_SEC = 3.0


class MockSpot(RobotInterface):
    """In-memory stand-in for a real Spot.

    Args:
        graph_path: Optional path to a ``downloaded_graph`` folder. If it exists
            the waypoint names are read from disk when possible, otherwise the
            built-in list is used.
        dock_id: The dock id the real robot would use.
        image_path: Optional override for the JPEG returned by ``capture_image``.
        console: Rich console for the action log.
    """

    def __init__(
        self,
        graph_path: Path | None = None,
        dock_id: int | None = None,
        image_path: Path | None = None,
        console: Console | None = None,
    ) -> None:
        self._console = console or _CONSOLE
        self._graph_path = graph_path
        self._dock_id = dock_id
        self._image_path = image_path or _ASSET_IMAGE
        self._lock = threading.RLock()
        self._log_times: dict[str, float] = {}
        self._log_counts: dict[str, int] = {}

        self._connected = False
        self._standing = False
        self._powered = False
        self._docked = dock_id is not None
        self._localized = False
        self._battery = 87.0
        self._location = "dock" if self._docked else "unknown"
        self._last_drive: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_drive_at = 0.0
        self._started = time.monotonic()
        self._waypoints = DEFAULT_MOCK_WAYPOINTS[:]

    # ------------------------------------------------------------------
    def _log(self, message: str, throttle_key: str | None = None) -> None:
        """Print a simulated SDK call.

        ``throttle_key`` rate-limits repeated messages: follow-me grabs a frame
        eight times a second, and an unthrottled log would bury everything else.
        """
        if throttle_key is not None:
            now = time.monotonic()
            last = self._log_times.get(throttle_key, 0.0)
            if now - last < LOG_THROTTLE_SEC:
                self._log_counts[throttle_key] = self._log_counts.get(throttle_key, 0) + 1
                return
            self._log_times[throttle_key] = now
            repeats = self._log_counts.pop(throttle_key, 0)
            if repeats:
                message = f"{message}  (+{repeats} more since last line)"
        self._console.print(f"[dim]\\[mock][/dim] [cyan]{message}[/cyan]")

    def _drain_battery(self, amount: float) -> None:
        self._battery = max(0.0, self._battery - amount)

    # --- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        self._log("connecting... authenticated, time synced, lease taken, e-stop armed")
        time.sleep(0.2)
        self._connected = True
        if self._graph_path is not None and self._graph_path.exists():
            names = _read_waypoint_names(self._graph_path)
            if names:
                self._waypoints = names
                self._log(f"loaded {len(names)} waypoint names from {self._graph_path}")

    def shutdown(self) -> None:
        if self._connected:
            self._log("lease released, e-stop endpoint shut down")
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    # --- posture -----------------------------------------------------------

    def stand(self) -> ActionResult:
        with self._lock:
            if self._docked:
                return fail("I'm on the dock. Ask me to undock first.")
            self._powered = True
            self._standing = True
            self._drain_battery(0.3)
            self._log("power on -> blocking_stand")
        time.sleep(0.4)
        return ok("Standing.")

    def sit(self) -> ActionResult:
        with self._lock:
            self._standing = False
            self._last_drive = (0.0, 0.0, 0.0)
            self._log("synchro_sit_command")
        # The real synchro_sit_command is a non-blocking RPC that returns as soon
        # as the robot accepts it, so keep the simulated cost in that ballpark --
        # the reflex lane's latency budget is measured against this path.
        time.sleep(0.05)
        return ok("Sitting.")

    # --- motion ------------------------------------------------------------

    def drive(self, v_x: float, v_y: float, v_rot: float) -> None:
        velocity = clamp_velocity(v_x, v_y, v_rot)
        with self._lock:
            self._last_drive = velocity.as_tuple()
            self._last_drive_at = time.monotonic()
        # Only log meaningful, non-zero commands so the follow-me loop at 8 Hz
        # does not flood the console.
        if any(abs(component) > 0.01 for component in velocity.as_tuple()):
            self._drain_battery(0.002)

    def move(
        self,
        direction: str,
        distance_m: float | None = None,
        degrees: float | None = None,
        speed: float | None = None,
    ) -> ActionResult:
        plan = plan_move(direction, distance_m, degrees, speed)
        if plan is None:
            return fail(
                f"I don't know how to move {direction}. "
                "I can go forward, back, left, right, turn left or turn right."
            )
        velocity, duration, human = plan

        if self._docked:
            return fail("I'm on the dock. Ask me to undock first.")

        with self._lock:
            if not self._standing:
                self._log("not standing yet -> standing first")
                self._standing = True
                self._powered = True
            self._log(
                f"velocity segment {velocity.as_tuple()} for {duration:.1f}s "
                f"(end_time = now + {VELOCITY_CMD_DURATION}s, re-issued)"
            )

        # Simulate the re-issue loop so the timing feels like the real thing.
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.drive(*velocity.as_tuple())
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        self.drive(0.0, 0.0, 0.0)
        self._drain_battery(0.1)
        self._localized = False if velocity.v_x else self._localized
        return ok(human)

    def stop_all(self) -> ActionResult:
        with self._lock:
            self._last_drive = (0.0, 0.0, 0.0)
            self._log("cancel active command/mission -> safe stop (settle)")
        return ok("Stopped.")

    # --- navigation --------------------------------------------------------

    def list_waypoints(self) -> ActionResult:
        with self._lock:
            names = list(self._waypoints)
        if not names:
            return fail("I don't have a map loaded.")
        return ok(f"I know {len(names)} places.", waypoints=names)

    def navigate_to(self, waypoint_name: str) -> ActionResult:
        with self._lock:
            names = list(self._waypoints)
        match = match_waypoint(waypoint_name, names)
        if match is None:
            return fail(
                f"I don't know a place called {waypoint_name}. I know: " + ", ".join(names) + "."
            )
        # "Go to X" is unambiguous about wanting to leave the dock, so this one
        # undocks itself rather than making the operator ask twice. Must match
        # SpotClient.navigate_to, or mock mode mispredicts the real robot.
        if self._docked:
            undocked = self.undock()
            if not undocked.ok:
                return undocked

        self._log(f"upload graph -> localize on fiducial -> navigate_to({match})")
        self._standing = True
        self._powered = True
        self._localized = True
        # Pretend the walk takes a moment so speech and interruption behave
        # realistically without making the demo tedious.
        for _ in range(6):
            time.sleep(0.25)
            self.drive(0.4, 0.0, 0.0)
        self.drive(0.0, 0.0, 0.0)
        with self._lock:
            self._location = match
        self._drain_battery(0.5)
        return ok(f"I'm at {match}.", waypoint=match)

    # --- docking -----------------------------------------------------------

    def dock(self) -> ActionResult:
        if self._dock_id is None:
            return fail("No dock is configured, so I can't dock.")
        if self._docked:
            return ok("I'm already docked.")
        self._log(f"blocking_stand -> blocking_dock_robot(dock_id={self._dock_id})")
        time.sleep(0.8)
        with self._lock:
            self._docked = True
            self._standing = False
            self._location = "dock"
        return ok("Docked.")

    def undock(self) -> ActionResult:
        if not self._docked:
            return ok("I'm not on the dock.")
        self._log("power on -> blocking_go_to_prep_pose")
        time.sleep(0.8)
        with self._lock:
            self._docked = False
            self._standing = True
            self._powered = True
            self._location = "near dock"
        return ok("Off the dock and standing.")

    def is_docked(self) -> bool:
        return self._docked

    # --- sensing -----------------------------------------------------------

    def capture_image(self, camera: str = "front") -> ActionResult:
        if camera not in CAMERA_SOURCES:
            return fail(
                f"I don't have a {camera} camera. I can use front, left or right."
            )
        source, rotation = CAMERA_SOURCES[camera]
        self._log(
            f"image_client.get_image({source}) rotate {rotation}deg",
            throttle_key=f"image:{camera}",
        )
        try:
            data = self._image_path.read_bytes()
        except OSError:
            return fail("My camera isn't returning an image right now.")
        return ActionResult(
            True,
            f"Here's what my {camera} camera sees.",
            {"camera": camera, "source": source},
            image_jpeg=data,
        )

    def get_status(self) -> ActionResult:
        with self._lock:
            uptime = time.monotonic() - self._started
            status = {
                "battery_percent": round(self._battery, 1),
                "motor_power": "on" if self._powered else "off",
                "posture": "standing" if self._standing else "sitting",
                "lease": "held" if self._connected else "not held",
                "estop": "released (software e-stop armed)",
                "localized": self._localized,
                "location": self._location,
                "docked": self._docked,
                "uptime_seconds": round(uptime, 1),
                "mock": True,
            }
        return ok(
            f"Battery {status['battery_percent']:.0f} percent, "
            f"{status['posture']}, motors {status['motor_power']}.",
            **status,
        )

    # --- audio -------------------------------------------------------------

    def play_wav(self, path: str, blocking: bool = True) -> None:
        raise RuntimeError(
            "Mock robot has no speaker. Set AUDIO_OUT=laptop when MOCK_ROBOT=true."
        )


class SimulatedPerson:
    """A believable moving person for follow-me rehearsal in mock mode.

    Produces bounding boxes in a 640x480 frame that drift left and right and
    change apparent size, so the P-controller in
    :mod:`spot_voice.robot.follow` genuinely has something to track. It
    occasionally "disappears" so the lost-target path gets exercised too.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._t0 = time.monotonic()
        self._width = 640
        self._height = 480

    def detect(self, frame_jpeg: bytes) -> tuple[int, int, int, int] | None:
        """Return ``(x1, y1, x2, y2)`` for the simulated person, or ``None``."""
        elapsed = time.monotonic() - self._t0
        # Vanish for a couple of seconds every ~25s to exercise "I lost you".
        if 20.0 <= (elapsed % 25.0) <= 22.5:
            return None
        centre_x = self._width / 2 + math.sin(elapsed * 0.6) * 140
        box_h = 260 + math.sin(elapsed * 0.35) * 70
        box_w = box_h * 0.4
        centre_y = self._height / 2 + 20
        x1 = int(centre_x - box_w / 2)
        x2 = int(centre_x + box_w / 2)
        y1 = int(centre_y - box_h / 2)
        y2 = int(centre_y + box_h / 2)
        return (x1, y1, x2, y2)

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._width, self._height)


def _read_waypoint_names(graph_path: Path) -> list[str]:
    """Best-effort read of waypoint annotation names from a graph file on disk.

    Uses the Spot SDK's protobuf definitions when available. In mock mode on a
    machine without the SDK this simply returns an empty list and the built-in
    waypoint names are used instead.
    """
    graph_file = graph_path / "graph"
    if not graph_file.exists():
        return []
    try:
        from bosdyn.api.graph_nav import map_pb2  # type: ignore[import-not-found]
    except Exception:
        return []
    try:
        graph = map_pb2.Graph()
        graph.ParseFromString(graph_file.read_bytes())
        return sorted({wp.annotations.name for wp in graph.waypoints if wp.annotations.name})
    except Exception:
        return []
