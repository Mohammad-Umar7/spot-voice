"""Turning a spoken ``move`` request into a capped velocity plus a duration.

Shared by the real client and the mock so both interpret "forward two metres" and
"turn left ninety degrees" identically. Pure arithmetic, no SDK, no I/O — which
is what makes it straightforward to test.
"""

from __future__ import annotations

import math

from .limits import (
    MIN_VROT,
    MIN_VX,
    Velocity,
    clamp_degrees,
    clamp_distance,
    clamp_duration,
    clamp_velocity,
)

#: Direction word -> (forward sign, left sign). Spoken variants included because
#: speech-to-text produces all of them.
_LINEAR_AXES: dict[str, tuple[float, float]] = {
    "forward": (1.0, 0.0),
    "forwards": (1.0, 0.0),
    "back": (-1.0, 0.0),
    "backward": (-1.0, 0.0),
    "backwards": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
}

#: Word used in the spoken confirmation for each linear direction.
_LINEAR_WORDS: dict[str, str] = {
    "forward": "forward",
    "forwards": "forward",
    "back": "back",
    "backward": "back",
    "backwards": "back",
    "left": "left",
    "right": "right",
}

_TURN_LEFT = {"turn_left", "left_turn"}
_TURN_RIGHT = {"turn_right", "right_turn"}

#: Defaults when the operator does not say how far or how fast.
DEFAULT_DISTANCE_M = 1.0
DEFAULT_DEGREES = 90.0
DEFAULT_LINEAR_SPEED = 0.4
DEFAULT_TURN_RATE = 0.5


def normalise_direction(direction: str | None) -> str:
    """Fold spoken variants ("turn left", "Turn-Left") into one canonical token."""
    if not direction:
        return ""
    return direction.strip().lower().replace("-", "_").replace(" ", "_")


def plan_move(
    direction: str,
    distance_m: float | None = None,
    degrees: float | None = None,
    speed: float | None = None,
) -> tuple[Velocity, float, str] | None:
    """Plan one bounded move segment.

    Args:
        direction: forward / back / left / right / turn_left / turn_right.
        distance_m: Metres for a linear move. Defaults to 1.0, capped.
        degrees: Degrees for a turn. Defaults to 90, capped.
        speed: m/s or rad/s. Capped by :func:`~spot_voice.robot.limits.clamp_velocity`.

    Returns:
        ``(velocity, duration_seconds, spoken_summary)``, or ``None`` when the
        direction is not one this robot understands.
    """
    direction = normalise_direction(direction)

    if direction in _TURN_LEFT or direction in _TURN_RIGHT:
        magnitude, _ = clamp_degrees(degrees if degrees is not None else DEFAULT_DEGREES)
        if magnitude < 1.0:
            magnitude = DEFAULT_DEGREES
        rate = max(MIN_VROT, abs(speed) if speed else DEFAULT_TURN_RATE)
        sign = 1.0 if direction in _TURN_LEFT else -1.0
        velocity = clamp_velocity(0.0, 0.0, sign * rate)
        duration, _ = clamp_duration(
            math.radians(magnitude) / max(abs(velocity.v_rot), 1e-3)
        )
        word = "left" if sign > 0 else "right"
        return velocity, duration, f"Turned {magnitude:.0f} degrees {word}."

    axis = _LINEAR_AXES.get(direction)
    if axis is None:
        return None

    magnitude, _ = clamp_distance(
        distance_m if distance_m is not None else DEFAULT_DISTANCE_M
    )
    if magnitude < 0.05:
        magnitude = DEFAULT_DISTANCE_M
    rate = max(MIN_VX, abs(speed) if speed else DEFAULT_LINEAR_SPEED)
    velocity = clamp_velocity(axis[0] * rate, axis[1] * rate, 0.0)
    effective = max(abs(velocity.v_x), abs(velocity.v_y), 1e-3)
    duration, _ = clamp_duration(magnitude / effective)
    return velocity, duration, f"Moved {magnitude:.1f} metres {_LINEAR_WORDS[direction]}."


def match_waypoint(requested: str, known: list[str]) -> str | None:
    """Resolve a spoken place name against the map, tolerating small slips.

    Exact match, then containment either way, then a closest-match fallback with
    a similarity floor so a wildly different name is reported as unknown rather
    than silently walking somewhere else.
    """
    if not requested or not known:
        return None
    needle = requested.strip().lower().replace("_", " ").replace("-", " ")
    normalised = {name: name.lower().replace("_", " ").replace("-", " ") for name in known}

    for name, value in normalised.items():
        if value == needle:
            return name
    for name, value in normalised.items():
        if needle in value or value in needle:
            return name

    from difflib import SequenceMatcher

    best_name, best_score = None, 0.0
    for name, value in normalised.items():
        score = SequenceMatcher(None, needle, value).ratio()
        if score > best_score:
            best_name, best_score = name, score
    return best_name if best_score >= 0.75 else None
