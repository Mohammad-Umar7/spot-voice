"""Hard motion limits.

These are **deliberately constants in code, not configuration**. Nothing in
``.env`` can raise them, and no Claude tool call can exceed them: every velocity
that reaches the Spot SDK passes through :func:`clamp_velocity` first.

These limits sit *on top of* Spot's own safety systems (obstacle avoidance,
self-righting, stair handling), which stay at their factory defaults. We never
touch obstacle-padding parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Maximum forward/backward speed in metres per second.
MAX_VX = 0.6
#: Maximum lateral (strafe) speed in metres per second.
MAX_VY = 0.4
#: Maximum yaw rate in radians per second.
MAX_VROT = 0.8

#: How far into the future each velocity command is valid, in seconds.
#:
#: This is the dead-man's switch. Velocity commands carry an ``end_time_secs`` of
#: ``now + VELOCITY_CMD_DURATION``; an active loop re-issues them at a faster
#: cadence. If this program dies, the laptop sleeps, or wifi drops, no further
#: commands arrive, the last one expires, and the robot stops itself.
VELOCITY_CMD_DURATION = 0.6

#: Re-issue period for the velocity loop. Must be comfortably below
#: ``VELOCITY_CMD_DURATION`` so commands overlap rather than gap.
VELOCITY_CMD_PERIOD = 0.25

#: Sanity bounds on a single ``move`` tool call, so a mis-heard "twenty metres"
#: cannot turn into a two-minute walk.
MAX_MOVE_DISTANCE_M = 5.0
MAX_MOVE_DEGREES = 360.0
MAX_MOVE_DURATION_S = 30.0

#: Minimum useful speeds -- below these Spot effectively will not move.
MIN_VX = 0.1
MIN_VROT = 0.15


@dataclass(frozen=True)
class Velocity:
    """A capped body-frame velocity command."""

    v_x: float
    v_y: float
    v_rot: float
    clamped: bool

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.v_x, self.v_y, self.v_rot)


def _clamp(value: float, limit: float) -> tuple[float, bool]:
    """Clamp ``value`` into ``[-limit, limit]`` and report whether it was changed."""
    if value > limit:
        return limit, True
    if value < -limit:
        return -limit, True
    return value, False


def clamp_velocity(v_x: float, v_y: float, v_rot: float) -> Velocity:
    """Clamp a requested velocity to the hard caps.

    Non-finite inputs (``nan``/``inf``) collapse to zero -- a NaN reaching the SDK
    would be an undefined command, and stopping is the safe interpretation.

    Args:
        v_x: Forward velocity in m/s (positive = forward).
        v_y: Lateral velocity in m/s (positive = left).
        v_rot: Yaw rate in rad/s (positive = counter-clockwise).

    Returns:
        A :class:`Velocity` whose components are within the hard caps.
    """
    clamped = False
    out: list[float] = []
    for value, limit in ((v_x, MAX_VX), (v_y, MAX_VY), (v_rot, MAX_VROT)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            number, clamped = 0.0, True
        else:
            if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
                number, clamped = 0.0, True
        number, hit = _clamp(number, limit)
        clamped = clamped or hit
        out.append(number)
    return Velocity(out[0], out[1], out[2], clamped)


def clamp_distance(distance_m: float) -> tuple[float, bool]:
    """Clamp a linear move distance. Returns ``(distance, was_clamped)``."""
    try:
        value = abs(float(distance_m))
    except (TypeError, ValueError):
        return 0.0, True
    if value != value:  # NaN
        return 0.0, True
    if value > MAX_MOVE_DISTANCE_M:
        return MAX_MOVE_DISTANCE_M, True
    return value, False


def clamp_degrees(degrees: float) -> tuple[float, bool]:
    """Clamp a turn magnitude in degrees. Returns ``(degrees, was_clamped)``."""
    try:
        value = abs(float(degrees))
    except (TypeError, ValueError):
        return 0.0, True
    if value != value:  # NaN
        return 0.0, True
    if value > MAX_MOVE_DEGREES:
        return MAX_MOVE_DEGREES, True
    return value, False


def clamp_duration(seconds: float) -> tuple[float, bool]:
    """Clamp a timed-motion duration. Returns ``(seconds, was_clamped)``."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return 0.0, True
    if value != value or value < 0:
        return 0.0, True
    if value > MAX_MOVE_DURATION_S:
        return MAX_MOVE_DURATION_S, True
    return value, False
