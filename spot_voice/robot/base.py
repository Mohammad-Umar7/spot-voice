"""The robot interface every lane in the program talks to.

Both :class:`~spot_voice.robot.mock.MockSpot` and
:class:`~spot_voice.robot.spot_client.SpotClient` implement this, so the voice
loop, the reflex lane, the tool dispatcher and the follow-me thread are all
identical at a desk and on a real robot.

Implementations raise; the dispatcher is the single place that converts raised
exceptions into ``{ok, message}`` for Claude.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

#: Camera names exposed to Claude, and the Spot image sources behind them.
#:
#: The two front fisheye cameras are physically mounted sideways on Spot, so
#: their frames must be rotated to be upright. The angles below match the
#: official SDK image examples.
CAMERA_SOURCES: dict[str, tuple[str, int]] = {
    "front": ("frontleft_fisheye_image", -78),
    "left": ("left_fisheye_image", 0),
    "right": ("right_fisheye_image", 180),
}


@dataclass
class ActionResult:
    """Outcome of a robot action.

    Attributes:
        ok: Whether the action succeeded.
        message: A short sentence written to be spoken aloud.
        data: Optional structured payload (status dicts, waypoint lists, ...).
        image_jpeg: Optional JPEG bytes, set only by camera captures.
    """

    ok: bool
    message: str
    data: dict[str, Any] | None = None
    image_jpeg: bytes | None = field(default=None, repr=False)

    def to_model_payload(self) -> dict[str, Any]:
        """Return the JSON-serialisable dict handed back to Claude.

        Image bytes are deliberately excluded -- they travel as a separate image
        content block, not as base64 inside the JSON.
        """
        payload: dict[str, Any] = {"ok": self.ok, "message": self.message}
        if self.data:
            payload.update(self.data)
        return payload


def ok(message: str, **data: Any) -> ActionResult:
    """Shorthand for a successful :class:`ActionResult`."""
    return ActionResult(True, message, data or None)


def fail(message: str, **data: Any) -> ActionResult:
    """Shorthand for a failed :class:`ActionResult`."""
    return ActionResult(False, message, data or None)


class RobotInterface(abc.ABC):
    """High-level commands issued through the official Spot SDK.

    Deliberately narrow: only whole-robot, high-level actions are exposed. There
    is no method here that disables, bypasses or reconfigures Spot's own safety
    systems, and none should ever be added.
    """

    # --- lifecycle ---------------------------------------------------------

    @abc.abstractmethod
    def connect(self) -> None:
        """Authenticate, sync time, take the lease and arm the software e-stop."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release the lease and e-stop endpoint. Safe to call more than once."""

    @property
    @abc.abstractmethod
    def connected(self) -> bool:
        """True when the robot is reachable and we hold the lease."""

    # --- posture -----------------------------------------------------------

    @abc.abstractmethod
    def power_on(self) -> ActionResult:
        """Turn motor power on without changing posture."""

    @abc.abstractmethod
    def stand(self) -> ActionResult:
        """Power on if needed and stand up."""

    @abc.abstractmethod
    def sit(self) -> ActionResult:
        """Sit down."""

    @abc.abstractmethod
    def emote(self, gesture: str) -> ActionResult:
        """Perform a short body-language gesture while standing."""

    # --- motion ------------------------------------------------------------

    @abc.abstractmethod
    def drive(self, v_x: float, v_y: float, v_rot: float) -> None:
        """Issue one capped velocity command with a short expiry.

        Callers must re-issue this faster than the command expires; that expiry
        is the dead-man's switch.
        """

    @abc.abstractmethod
    def move(
        self,
        direction: str,
        distance_m: float | None = None,
        degrees: float | None = None,
        speed: float | None = None,
    ) -> ActionResult:
        """Walk a bounded, timed segment in one direction and then stop."""

    @abc.abstractmethod
    def stop_all(self) -> ActionResult:
        """Cancel any active command or mission and settle into a safe stop."""

    # --- navigation --------------------------------------------------------

    @abc.abstractmethod
    def list_waypoints(self) -> ActionResult:
        """Return the named waypoints from the uploaded GraphNav map."""

    @abc.abstractmethod
    def navigate_to(self, waypoint_name: str) -> ActionResult:
        """Autonomously walk to a named waypoint using GraphNav."""

    # --- docking -----------------------------------------------------------

    @abc.abstractmethod
    def dock(self) -> ActionResult:
        """Stand, then dock at the configured dock id."""

    @abc.abstractmethod
    def undock(self) -> ActionResult:
        """Power on and walk off the dock into the prep pose."""

    @abc.abstractmethod
    def is_docked(self) -> bool:
        """True when the robot reports it is sitting on a dock."""

    # --- sensing -----------------------------------------------------------

    @abc.abstractmethod
    def capture_image(self, camera: str = "front") -> ActionResult:
        """Grab one upright JPEG frame from the named camera."""

    @abc.abstractmethod
    def get_status(self) -> ActionResult:
        """Return battery, motor power, lease/e-stop state and localization."""

    # --- audio -------------------------------------------------------------

    @abc.abstractmethod
    def play_wav(self, path: str, blocking: bool = True) -> None:
        """Play a WAV file through the robot's own speaker (Spot CAM audio)."""
