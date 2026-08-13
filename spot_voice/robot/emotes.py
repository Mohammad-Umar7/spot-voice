"""Body-language gestures.

Spot has no arm here, so "say hi to these people" cannot be a wave. What it does
have is a body it can pitch, roll, yaw and raise while standing -- which is
enough for a nod, a bow, a head-tilt and a wiggle, and those read clearly to a
room of people.

Every gesture is a short sequence of body poses. The angles are clamped here,
the same way velocities are clamped in :mod:`spot_voice.robot.limits`: nothing a
model asks for can tip the robot past a safe pose. Gestures only run while
standing and never during a walk -- the dispatcher enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Hard caps on body pose, in radians. Spot can physically exceed these; these
#: are the limits this program will ask for. Generous enough to read as a
#: gesture, small enough to stay stable.
MAX_PITCH = 0.35
MAX_ROLL = 0.25
MAX_YAW = 0.40

#: Body height offset in metres, relative to nominal stand.
MAX_HEIGHT_OFFSET = 0.12

#: Longest a single pose may be held, and the longest a whole gesture may run.
MAX_POSE_HOLD_SEC = 1.5
MAX_GESTURE_SEC = 8.0


@dataclass(frozen=True)
class Pose:
    """One body pose in a gesture.

    Angles are in radians, in the robot's body frame. Positive pitch tips the
    nose down; positive roll leans right; positive yaw turns the body left
    without moving the feet.
    """

    yaw: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    height: float = 0.0
    hold: float = 0.4


def clamp_pose(pose: Pose) -> Pose:
    """Clamp a pose into the safe envelope.

    Non-finite values collapse to neutral: a NaN reaching the SDK would be an
    undefined body pose, and standing straight is the safe reading.
    """

    def bound(value: float, limit: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if number != number or number in (float("inf"), float("-inf")):
            return 0.0
        return max(-limit, min(limit, number))

    hold = bound(pose.hold, MAX_POSE_HOLD_SEC)
    return Pose(
        yaw=bound(pose.yaw, MAX_YAW),
        roll=bound(pose.roll, MAX_ROLL),
        pitch=bound(pose.pitch, MAX_PITCH),
        height=bound(pose.height, MAX_HEIGHT_OFFSET),
        hold=max(0.05, abs(hold)),
    )


#: Neutral: what every gesture returns to when it finishes.
NEUTRAL = Pose()

#: The gesture library. Keep these short -- they run in front of people, and a
#: long animation reads as a malfunction rather than a greeting.
GESTURES: dict[str, tuple[str, tuple[Pose, ...]]] = {
    "nod": (
        "Nodded.",
        (Pose(pitch=0.25, hold=0.3), NEUTRAL, Pose(pitch=0.25, hold=0.3), NEUTRAL),
    ),
    "shake": (
        "Shook my head.",
        (Pose(yaw=0.3, hold=0.3), Pose(yaw=-0.3, hold=0.3), Pose(yaw=0.3, hold=0.3), NEUTRAL),
    ),
    "bow": (
        "Took a bow.",
        (Pose(pitch=0.35, height=-0.1, hold=0.9), NEUTRAL),
    ),
    "greet": (
        "Said hello.",
        (
            Pose(pitch=0.3, height=-0.08, hold=0.6),  # a bow
            Pose(roll=0.2, hold=0.25),  # then a friendly tilt each way
            Pose(roll=-0.2, hold=0.25),
            NEUTRAL,
        ),
    ),
    "wiggle": (
        "Had a wiggle.",
        (
            Pose(roll=0.22, hold=0.25),
            Pose(roll=-0.22, hold=0.25),
            Pose(roll=0.22, hold=0.25),
            Pose(roll=-0.22, hold=0.25),
            NEUTRAL,
        ),
    ),
    "look_around": (
        "Had a look around.",
        (Pose(yaw=0.4, hold=0.8), Pose(yaw=-0.4, hold=0.8), NEUTRAL),
    ),
    "stretch": (
        "Stretched.",
        (Pose(height=0.12, hold=0.7), Pose(height=-0.1, pitch=0.15, hold=0.6), NEUTRAL),
    ),
    "yes": ("Yes.", (Pose(pitch=0.22, hold=0.25), NEUTRAL)),
    "no": ("No.", (Pose(yaw=0.28, hold=0.25), Pose(yaw=-0.28, hold=0.25), NEUTRAL)),
}

#: Spoken names the model or operator might use, mapped onto the library.
ALIASES: dict[str, str] = {
    "hello": "greet",
    "hi": "greet",
    "hey": "greet",
    "wave": "greet",  # no arm, so a greeting bow stands in for a wave
    "say_hi": "greet",
    "say_hello": "greet",
    "greeting": "greet",
    "welcome": "greet",
    "bow_down": "bow",
    "curtsy": "bow",
    "nod_yes": "yes",
    "agree": "yes",
    "disagree": "no",
    "shake_head": "shake",
    "dance": "wiggle",
    "wag": "wiggle",
    "scan": "look_around",
    "lookaround": "look_around",
}


def resolve(name: str) -> str | None:
    """Map a spoken gesture name onto one in the library, or ``None``."""
    if not name:
        return None
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in GESTURES:
        return key
    return ALIASES.get(key)


def sequence(name: str) -> tuple[str, tuple[Pose, ...]] | None:
    """Return ``(spoken_summary, clamped_poses)`` for a gesture, or ``None``.

    The returned poses are already clamped and the whole sequence is bounded to
    :data:`MAX_GESTURE_SEC`, so callers can execute them without re-checking.
    """
    resolved = resolve(name)
    if resolved is None:
        return None
    summary, poses = GESTURES[resolved]

    clamped: list[Pose] = []
    total = 0.0
    for pose in poses:
        safe = clamp_pose(pose)
        if total + safe.hold > MAX_GESTURE_SEC:
            break
        clamped.append(safe)
        total += safe.hold
    # Always finish standing straight, whatever got truncated.
    if not clamped or clamped[-1] != NEUTRAL:
        clamped.append(NEUTRAL)
    return summary, tuple(clamped)


def available() -> list[str]:
    """Gesture names, for tool schemas and error messages."""
    return sorted(GESTURES)
