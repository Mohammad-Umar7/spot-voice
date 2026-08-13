"""Where "here" and "beside me" actually are.

"Stand over there" and "sit beside me" sound like the same kind of instruction
and are not. The difference is not direction -- it is whose body the target is
measured from.

Research on spatial deixis in human-robot interaction models this as
*peri-personal space*: the bubble immediately around a person, roughly what they
can reach. "There" points outside that bubble and needs a gesture to say which
way. "Here" and "beside me" name a spot *inside* it, and need no gesture at all,
because the operator's own body is the reference. Asking someone to point at
themselves would be absurd, which is the giveaway that these are different
operations.

So this module converts "where the person is" into "where the robot should
stand", and the pointing code is left to handle only the outward case.

The honest limitation, from the same literature: humans agree with each other
about deixis only about two thirds of the time. Where "here" stops and "there"
starts is genuinely fuzzy, and people disambiguate with gaze, which this robot
does not track. That is why every caller of this module announces what it
understood before it moves -- the ambiguity cannot be engineered away, only
surfaced in time for a person to correct it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: How close the robot ends up when told to come to someone, in metres.
#: A comfortable conversational distance -- close enough to be "here", far
#: enough not to loom over them.
PERSONAL_STANDOFF_M = 1.2

#: Sideways offset when asked to be *beside* someone rather than in front.
#: About one body width, so it ends up alongside rather than pressed against.
BESIDE_OFFSET_M = 0.9

#: Standoff used for "beside": near zero, because the offset itself provides the
#: separation. Stopping short as well would leave it diagonally behind them.
BESIDE_STANDOFF_M = 0.15

#: Beyond this the apparent-size distance estimate is too coarse to act on, and
#: "come here" would become a long unsupervised walk toward a guess.
MAX_APPROACH_M = 8.0

#: Inside this the robot is already there; walking further just crowds them.
ALREADY_HERE_M = 0.8


@dataclass(frozen=True)
class Approach:
    """Where to walk, expressed the way :meth:`RobotInterface.walk_toward` wants."""

    bearing_deg: float
    distance_m: float
    standoff_m: float
    #: Spoken before moving, because a misread here walks at a person.
    summary: str

    @property
    def already_there(self) -> bool:
        return self.distance_m <= 0.0


def approach_person(
    bearing_deg: float,
    distance_m: float,
    position: str = "in_front",
) -> Approach | None:
    """Turn a person's location into somewhere to stand near them.

    Args:
        bearing_deg: Degrees to the person, positive to the robot's left.
        distance_m: Rough metres to the person.
        position: ``"in_front"`` for "come here", ``"beside"`` for "beside me".

    Returns:
        An :class:`Approach`, or ``None`` when the person is too far away for
        the estimate to be worth acting on.
    """
    if distance_m <= 0.0 or distance_m > MAX_APPROACH_M:
        return None

    if position == "beside":
        return _beside(bearing_deg, distance_m)
    return _in_front(bearing_deg, distance_m)


def _in_front(bearing_deg: float, distance_m: float) -> Approach:
    """Straight at them, stopping a conversational distance short."""
    if distance_m <= ALREADY_HERE_M:
        return Approach(
            bearing_deg=bearing_deg,
            distance_m=0.0,
            standoff_m=0.0,
            summary="I'm already next to you.",
        )
    return Approach(
        bearing_deg=bearing_deg,
        distance_m=distance_m,
        standoff_m=PERSONAL_STANDOFF_M,
        summary=(
            f"Coming to you -- about {max(0.0, distance_m - PERSONAL_STANDOFF_M):.1f} "
            "metres."
        ),
    )


def _beside(bearing_deg: float, distance_m: float) -> Approach:
    """Alongside them rather than facing them.

    The target is the person's position shifted sideways by one body width. The
    side is chosen as whichever needs less turning, which is also the side that
    avoids walking across their front to get there.
    """
    bearing = math.radians(bearing_deg)
    person_x = distance_m * math.cos(bearing)
    person_y = distance_m * math.sin(bearing)

    options = []
    for side, offset in (("left", BESIDE_OFFSET_M), ("right", -BESIDE_OFFSET_M)):
        goal_x, goal_y = person_x, person_y + offset
        options.append(
            (
                abs(math.degrees(math.atan2(goal_y, goal_x))),
                side,
                math.degrees(math.atan2(goal_y, goal_x)),
                math.hypot(goal_x, goal_y),
            )
        )
    _score, side, goal_bearing, goal_distance = min(options)

    # "left"/"right" here are the robot's, and saying so avoids the classic
    # confusion where the robot means its left and the person hears theirs.
    return Approach(
        bearing_deg=goal_bearing,
        distance_m=goal_distance,
        standoff_m=BESIDE_STANDOFF_M,
        summary=f"Coming to stand beside you, on my {side}.",
    )
