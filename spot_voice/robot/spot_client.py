"""The real robot layer, built on the official Boston Dynamics SDK (bosdyn-client 4.x).

Only high-level commands go through here. Spot's own safety systems -- obstacle
avoidance, self-righting, stair handling -- stay at their factory defaults; this
module never sets an obstacle-padding parameter or any other safety override.

All ``bosdyn`` imports are local to the functions that need them so that mock
mode runs on a machine with no Spot SDK installed.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from .base import CAMERA_SOURCES, ActionResult, RobotInterface, fail, ok
from .errors import (
    RobotBusy,
    RobotUnreachable,
    SpotVoiceError,
    is_connection_error,
    to_speakable,
)
from .estop import SoftwareEstop
from .graphnav import GraphNav
from .limits import VELOCITY_CMD_DURATION, VELOCITY_CMD_PERIOD, clamp_velocity
from .motion import plan_move

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

#: SDK client name registered with the robot's directory.
SDK_NAME = "SpotVoiceClient"

#: Reconnect backoff schedule in seconds.
_BACKOFF = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0)


class SpotClient(RobotInterface):
    """Talks to a real Spot.

    Args:
        ip: Robot hostname or IP (``SPOT_IP``).
        graph_path: ``downloaded_graph`` folder for GraphNav, or ``None``.
        dock_id: Dock fiducial id used by dock/undock, or ``None``.
        on_connection_lost: Optional callback invoked once when the link drops.
        on_reconnected: Optional callback invoked once when it comes back.
    """

    def __init__(
        self,
        ip: str,
        graph_path: Path | None = None,
        dock_id: int | None = None,
        on_connection_lost: Callable[[str], None] | None = None,
        on_reconnected: Callable[[], None] | None = None,
    ) -> None:
        self._ip = ip
        self._graph_path = graph_path
        self._dock_id = dock_id
        self._on_connection_lost = on_connection_lost
        self._on_reconnected = on_reconnected

        self._robot: Any = None
        self._lease_client: Any = None
        self._lease_keepalive: Any = None
        self._command_client: Any = None
        self._state_client: Any = None
        self._image_client: Any = None
        self._power_client: Any = None
        self._estop: SoftwareEstop | None = None
        self._graphnav: GraphNav | None = None
        self._sounds: dict[str, Any] = {}

        self._lock = threading.RLock()
        self._connected = False
        # Two separate flags, deliberately. `_abort` means "the operator said
        # stop" and must survive a reconnect -- clearing it there would silently
        # drop a safety request. `_closing` means "we are shutting down" and is
        # what stops the reconnect loop.
        self._abort = threading.Event()
        self._closing = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Authenticate, sync time, take the lease and arm the software e-stop."""
        import bosdyn.client
        import bosdyn.client.util
        from bosdyn.client import spot_cam
        from bosdyn.client.estop import EstopClient
        from bosdyn.client.image import ImageClient
        from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
        from bosdyn.client.power import PowerClient
        from bosdyn.client.robot_command import RobotCommandClient
        from bosdyn.client.robot_state import RobotStateClient

        with self._lock:
            self._closing.clear()
            sdk = bosdyn.client.create_standard_sdk(SDK_NAME)
            # Spot CAM services (audio, among others) are not in the standard
            # SDK; register them or the speaker is unreachable.
            spot_cam.register_all_service_clients(sdk)

            try:
                self._robot = sdk.create_robot(self._ip)
                # Reads BOSDYN_CLIENT_USERNAME / BOSDYN_CLIENT_PASSWORD from the
                # environment. Credentials never pass through our code.
                bosdyn.client.util.authenticate(self._robot)
                self._robot.time_sync.wait_for_sync(timeout_sec=10)
            except Exception as exc:
                raise RobotUnreachable(to_speakable(exc)) from exc

            self._command_client = self._robot.ensure_client(
                RobotCommandClient.default_service_name
            )
            self._state_client = self._robot.ensure_client(RobotStateClient.default_service_name)
            self._image_client = self._robot.ensure_client(ImageClient.default_service_name)
            self._power_client = self._robot.ensure_client(PowerClient.default_service_name)

            # Lease: deliberately hijack a stale lease left by a crashed session,
            # then hold it with a keep-alive that returns it on exit.
            self._lease_client = self._robot.ensure_client(LeaseClient.default_service_name)
            try:
                self._lease_client.take()
                self._lease_keepalive = LeaseKeepAlive(
                    self._lease_client, must_acquire=True, return_at_exit=True
                )
            except ResourceAlreadyClaimedError as exc:
                raise RobotBusy("Spot is claimed by another controller.") from exc
            except Exception as exc:
                raise RobotUnreachable(to_speakable(exc)) from exc

            try:
                estop_client = self._robot.ensure_client(EstopClient.default_service_name)
                self._estop = SoftwareEstop(estop_client)
            except Exception as exc:
                raise SpotVoiceError(to_speakable(exc)) from exc

            if self._graph_path is not None:
                try:
                    self._graphnav = GraphNav(self._robot, self._graph_path)
                except Exception:
                    LOGGER.warning("GraphNav unavailable on this robot", exc_info=True)
                    self._graphnav = None

            self._connected = True
            LOGGER.info("Connected to Spot at %s", self._ip)

    def shutdown(self) -> None:
        """Release everything and stop reconnecting. Safe to call more than once."""
        self._closing.set()
        self._abort.set()
        self._teardown()

    def _teardown(self) -> None:
        """Drop the lease and e-stop endpoint without ending the session.

        Used both by :meth:`shutdown` and by the reconnect loop, which must be
        able to discard stale clients without declaring the session over.
        """
        with self._lock:
            if self._estop is not None:
                self._estop.shutdown()
                self._estop = None
            if self._lease_keepalive is not None:
                try:
                    self._lease_keepalive.shutdown()
                except Exception:  # pragma: no cover - best-effort teardown
                    LOGGER.debug("lease keep-alive shutdown raised", exc_info=True)
                self._lease_keepalive = None
            self._connected = False
            LOGGER.info("Disconnected from Spot")

    @property
    def connected(self) -> bool:
        return self._connected and self._lease_keepalive is not None

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    def _call(self, description: str, func: Callable[[], T]) -> T:
        """Run an SDK call, reconnecting with exponential backoff on transport loss.

        The voice loop stays alive across a connection drop: the caller gets a
        speakable error, the operator hears it, and the next command retries.
        """
        try:
            return func()
        except Exception as exc:
            if not is_connection_error(exc):
                raise
            LOGGER.warning("%s failed: %s -- attempting reconnect", description, exc)
            if self._on_connection_lost is not None:
                self._on_connection_lost(to_speakable(exc))
            self._connected = False
            if not self._reconnect():
                raise RobotUnreachable("I lost connection to Spot.") from exc
            if self._on_reconnected is not None:
                self._on_reconnected()
            return func()

    def _reconnect(self) -> bool:
        """Try to re-establish the session with exponential backoff.

        Never clears ``_abort``: if the operator said "stop" while the link was
        down, that request must still be honoured once it is back.
        """
        for delay in _BACKOFF:
            if self._closing.is_set():
                return False
            time.sleep(delay)
            try:
                self._teardown()
            except Exception:  # pragma: no cover - stale clients may throw
                LOGGER.debug("teardown before reconnect raised", exc_info=True)
            try:
                self.connect()
                LOGGER.info("Reconnected to Spot")
                return True
            except Exception as exc:
                LOGGER.warning("Reconnect attempt failed: %s", exc)
        return False

    # ------------------------------------------------------------------
    # Posture
    # ------------------------------------------------------------------

    def _power_on_if_needed(self) -> None:
        from bosdyn.api import robot_state_pb2
        from bosdyn.client.power import power_on

        state = self._state_client.get_robot_state()
        if state.power_state.motor_power_state != robot_state_pb2.PowerState.STATE_ON:
            LOGGER.info("Powering motors on")
            power_on(self._power_client)
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                state = self._state_client.get_robot_state()
                if state.power_state.motor_power_state == robot_state_pb2.PowerState.STATE_ON:
                    return
                time.sleep(0.25)
            raise SpotVoiceError("Spot didn't power on. Check the battery and the tablet.")

    def power_on(self) -> ActionResult:
        """Turn motor power on and leave the posture alone.

        Standing is a separate step. Powering on is what fails when the e-stop
        is asserted or the battery is flat, so it is worth being able to do it
        -- and hear about it -- on its own.
        """
        try:
            self._call("power on", self._power_on_if_needed)
        except Exception as exc:
            return fail(to_speakable(exc))
        return ok("Motors are on.")

    def stand(self) -> ActionResult:
        from bosdyn.client.robot_command import blocking_stand

        if self.is_docked():
            return fail("I'm on the dock. Ask me to undock first.")
        try:
            self._call("power on", self._power_on_if_needed)
            self._call("stand", lambda: blocking_stand(self._command_client, timeout_sec=10))
        except Exception as exc:
            return fail(to_speakable(exc))
        return ok("Standing.")

    def sit(self) -> ActionResult:
        from bosdyn.client.robot_command import RobotCommandBuilder

        try:
            self._call(
                "sit",
                lambda: self._command_client.robot_command(
                    RobotCommandBuilder.synchro_sit_command()
                ),
            )
        except Exception as exc:
            return fail(to_speakable(exc))
        return ok("Sitting.")

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def drive(self, v_x: float, v_y: float, v_rot: float) -> None:
        """Issue one capped velocity command valid for ``VELOCITY_CMD_DURATION``.

        The short expiry is the dead-man's switch: if this process dies or wifi
        drops mid-walk, the command lapses and Spot stops itself.
        """
        from bosdyn.client.robot_command import RobotCommandBuilder

        velocity = clamp_velocity(v_x, v_y, v_rot)
        command = RobotCommandBuilder.synchro_velocity_command(
            v_x=velocity.v_x, v_y=velocity.v_y, v_rot=velocity.v_rot
        )
        self._command_client.robot_command(
            command, end_time_secs=time.time() + VELOCITY_CMD_DURATION
        )

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

        # Same guard as stand(): standing up while sitting on the dock is not a
        # thing to attempt. Claude has the undock tool and this message tells it
        # what to do, so "walk forward" while docked becomes undock then walk.
        if self.is_docked():
            return fail("I'm on the dock. Ask me to undock first.")

        # Stand before walking. A velocity command issued from a sit is not
        # reliably honoured, and standing first is what the operator expects.
        from bosdyn.client.robot_command import blocking_stand

        try:
            self._call("power on", self._power_on_if_needed)
            self._call("stand", lambda: blocking_stand(self._command_client, timeout_sec=10))
        except Exception as exc:
            return fail(to_speakable(exc))

        self._abort.clear()
        deadline = time.monotonic() + duration
        try:
            while time.monotonic() < deadline and not self._abort.is_set():
                self._call("drive", lambda: self.drive(*velocity.as_tuple()))
                time.sleep(min(VELOCITY_CMD_PERIOD, max(0.0, deadline - time.monotonic())))
            self._call("stop", lambda: self.drive(0.0, 0.0, 0.0))
        except Exception as exc:
            return fail(to_speakable(exc))

        if self._abort.is_set():
            return ok("Stopped.")
        return ok(human)

    def stop_all(self) -> ActionResult:
        """Cancel whatever is running and settle into a safe stop.

        This is the reflex "stop" action. It does not cut motor power -- the
        robot holds a stable stand rather than dropping. ``settle_then_cut`` on
        the e-stop is reserved for a genuine emergency power cut, and the
        physical tablet e-stop remains the ultimate authority.
        """
        from bosdyn.client.robot_command import RobotCommandBuilder

        self._abort.set()
        errors: list[str] = []

        if self._graphnav is not None:
            try:
                self._graphnav.cancel()
            except Exception as exc:  # pragma: no cover - best effort
                LOGGER.debug("graphnav cancel failed: %s", exc)

        try:
            self._command_client.robot_command(RobotCommandBuilder.stop_command())
        except Exception as exc:
            errors.append(to_speakable(exc))

        try:
            self.drive(0.0, 0.0, 0.0)
        except Exception as exc:
            errors.append(to_speakable(exc))

        if errors:
            return fail(errors[0])
        return ok("Stopped.")

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def list_waypoints(self) -> ActionResult:
        if self._graphnav is None:
            return fail("I don't have a map loaded, so I don't know any places.")
        try:
            self._call("upload graph", self._graphnav.upload_graph)
            names = self._graphnav.waypoint_names()
        except Exception as exc:
            return fail(to_speakable(exc))
        if not names:
            return fail("My map has no named places in it.")
        return ok(f"I know {len(names)} places.", waypoints=names)

    def navigate_to(self, waypoint_name: str) -> ActionResult:
        if self._graphnav is None:
            return fail("I don't have a map loaded, so I can't navigate.")
        if self.is_docked():
            undock = self.undock()
            if not undock.ok:
                return undock

        try:
            self._call("upload graph", self._graphnav.upload_graph)
        except Exception as exc:
            return fail(to_speakable(exc))

        waypoint_id = self._graphnav.resolve(waypoint_name)
        if waypoint_id is None:
            names = self._graphnav.waypoint_names()
            known = ", ".join(names) if names else "nothing yet"
            return fail(f"I don't know a place called {waypoint_name}. I know: {known}.")

        from bosdyn.client.robot_command import blocking_stand

        try:
            self._call("power on", self._power_on_if_needed)
            # Stand before localizing. The reference platform does this too, and
            # the reason is physical: fiducial localization works off what the
            # body cameras can see, and a sitting robot's cameras sit low and
            # angled at the floor. Standing first makes the fix far more likely
            # to succeed on the first try.
            self._call("stand", lambda: blocking_stand(self._command_client, timeout_sec=10))
            if not self._graphnav.localized:
                self._call("localize", self._graphnav.localize_on_fiducial)
        except Exception as exc:
            return fail(to_speakable(exc))

        self._abort.clear()
        try:
            self._graphnav.navigate_to(waypoint_id, should_stop=self._abort.is_set)
        except Exception as exc:
            return fail(to_speakable(exc))
        return ok(f"I'm at {waypoint_name}.", waypoint=waypoint_name)

    # ------------------------------------------------------------------
    # Docking
    # ------------------------------------------------------------------

    def dock(self) -> ActionResult:
        from bosdyn.client.docking import blocking_dock_robot
        from bosdyn.client.robot_command import blocking_stand

        if self._dock_id is None:
            return fail("No dock is configured, so I can't dock.")
        if self.is_docked():
            return ok("I'm already docked.")
        try:
            self._call("power on", self._power_on_if_needed)
            self._call("stand", lambda: blocking_stand(self._command_client, timeout_sec=10))
            self._call("dock", lambda: blocking_dock_robot(self._robot, self._dock_id))
        except Exception as exc:
            return fail(to_speakable(exc))
        return ok("Docked.")

    def undock(self) -> ActionResult:
        from bosdyn.client.docking import blocking_go_to_prep_pose, get_dock_id

        try:
            dock_id = self._call("dock id", lambda: get_dock_id(self._robot))
        except Exception as exc:
            return fail(to_speakable(exc))
        if dock_id is None:
            return ok("I'm not on the dock. Say stand if you want me up.")
        try:
            self._call("power on", self._power_on_if_needed)
            self._call("undock", lambda: blocking_go_to_prep_pose(self._robot, dock_id))
        except Exception as exc:
            return fail(to_speakable(exc))
        return ok("Off the dock and standing.")

    def is_docked(self) -> bool:
        from bosdyn.client.docking import get_dock_id

        try:
            return get_dock_id(self._robot) is not None
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def capture_image(self, camera: str = "front") -> ActionResult:
        if camera not in CAMERA_SOURCES:
            return fail(f"I don't have a {camera} camera. I can use front, left or right.")
        source, rotation = CAMERA_SOURCES[camera]
        try:
            responses = self._call(
                "image", lambda: self._image_client.get_image_from_sources([source])
            )
        except Exception as exc:
            return fail(to_speakable(exc))
        if not responses:
            return fail("My camera isn't returning an image right now.")

        try:
            jpeg = _to_upright_jpeg(responses[0], rotation)
        except Exception as exc:
            LOGGER.warning("Image decode failed", exc_info=True)
            return fail(f"I couldn't process the camera image. {to_speakable(exc)}")

        return ActionResult(
            True,
            f"Here's what my {camera} camera sees.",
            {"camera": camera, "source": source},
            image_jpeg=jpeg,
        )

    def get_status(self) -> ActionResult:
        from bosdyn.api import robot_state_pb2

        try:
            state = self._call("state", self._state_client.get_robot_state)
        except Exception as exc:
            return fail(to_speakable(exc))

        battery = None
        if state.battery_states:
            battery = state.battery_states[0].charge_percentage.value
        motors_on = (
            state.power_state.motor_power_state == robot_state_pb2.PowerState.STATE_ON
        )
        estop_state = "unknown"
        if self._estop is not None:
            estop_state = "asserted" if self._estop.is_stopped() else "released"

        localization = "no map"
        if self._graphnav is not None:
            try:
                localization = self._graphnav.localization_summary()
            except Exception:
                localization = "unknown"

        data = {
            "battery_percent": round(battery, 1) if battery is not None else None,
            "motor_power": "on" if motors_on else "off",
            "lease": "held" if self.connected else "not held",
            "estop": estop_state,
            "localization": localization,
            "docked": self.is_docked(),
        }
        spoken = (
            f"Battery {data['battery_percent']:.0f} percent, motors {data['motor_power']}, "
            f"{localization}."
            if battery is not None
            else f"Motors {data['motor_power']}, {localization}."
        )
        return ok(spoken, **data)

    # ------------------------------------------------------------------
    # Audio (Spot CAM speaker)
    # ------------------------------------------------------------------

    def play_wav(self, path: str, blocking: bool = True) -> None:
        """Load and play a WAV file through Spot CAM's speaker."""
        import wave

        from bosdyn.api.spot_cam import audio_pb2
        from bosdyn.client.spot_cam.audio import AudioClient

        audio_client = self._robot.ensure_client(AudioClient.default_service_name)
        name = Path(path).stem

        if name not in self._sounds:
            sound = audio_pb2.Sound(name=name)
            audio_client.load_sound(sound, Path(path).read_bytes())
            audio_client.set_volume(100.0)
            self._sounds[name] = sound
        else:
            # Re-upload: the file behind a given utterance changes every time.
            audio_client.load_sound(self._sounds[name], Path(path).read_bytes())

        audio_client.play_sound(self._sounds[name], 1.0)

        if blocking:
            with wave.open(path, "rb") as handle:
                duration = handle.getnframes() / float(handle.getframerate())
            # Slight padding: the robot link adds latency before playback starts.
            time.sleep(duration + 0.5)


# ----------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------


def _to_upright_jpeg(image_response: Any, rotation_degrees: int) -> bytes:
    """Decode a Spot image response, rotate it upright and re-encode as JPEG.

    Spot's front fisheye cameras are mounted sideways, so their frames arrive
    rotated. The angles come from the official SDK image example.
    """
    import cv2
    from bosdyn.api import image_pb2

    shot = image_response.shot.image
    buffer = np.frombuffer(shot.data, dtype=np.uint8)

    if shot.format == image_pb2.Image.FORMAT_RAW:
        channels = {
            image_pb2.Image.PIXEL_FORMAT_RGB_U8: 3,
            image_pb2.Image.PIXEL_FORMAT_RGBA_U8: 4,
            image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8: 1,
        }.get(shot.pixel_format, 1)
        try:
            frame = buffer.reshape((shot.rows, shot.cols, channels))
        except ValueError:
            frame = cv2.imdecode(buffer, -1)
    else:
        frame = cv2.imdecode(buffer, -1)

    if frame is None:
        raise ValueError("could not decode camera image")

    frame = rotate_image(frame, rotation_degrees)
    success, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        raise ValueError("could not encode camera image as JPEG")
    return encoded.tobytes()


def rotate_image(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate an image by an arbitrary angle, expanding the canvas to fit."""
    import cv2

    if degrees % 360 == 0:
        return frame
    if degrees % 90 == 0:
        turns = (degrees // 90) % 4
        codes = {
            1: cv2.ROTATE_90_COUNTERCLOCKWISE,
            2: cv2.ROTATE_180,
            3: cv2.ROTATE_90_CLOCKWISE,
        }
        return cv2.rotate(frame, codes[turns])

    height, width = frame.shape[:2]
    centre = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += new_width / 2.0 - centre[0]
    matrix[1, 2] += new_height / 2.0 - centre[1]
    return cv2.warpAffine(frame, matrix, (new_width, new_height))
