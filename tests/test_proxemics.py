""""Come here" and "sit beside me": targets measured from the person's body.

The distinction these pin down is that "over there" is measured from the
gesture and "here" is measured from the operator, so the second needs no
gesture at all. Getting that wrong is not a small error -- routing "sit beside
me" through the pointing code asks someone to point at themselves.
"""

from __future__ import annotations

import math

from spot_voice.vision.proxemics import (
    ALREADY_HERE_M,
    BESIDE_OFFSET_M,
    MAX_APPROACH_M,
    PERSONAL_STANDOFF_M,
    approach_person,
)


# ----------------------------------------------------------------------
# Coming to the person


def test_coming_to_someone_stops_short_of_them_rather_than_at_them():
    plan = approach_person(bearing_deg=0.0, distance_m=4.0)

    assert plan is not None
    assert plan.standoff_m == PERSONAL_STANDOFF_M
    # The robot ends a conversational distance away, not on their feet.
    assert plan.distance_m - plan.standoff_m == 4.0 - PERSONAL_STANDOFF_M


def test_coming_to_someone_heads_straight_at_them():
    plan = approach_person(bearing_deg=25.0, distance_m=3.0)
    assert plan.bearing_deg == 25.0


def test_being_told_to_come_when_already_there_does_not_shuffle_closer():
    plan = approach_person(bearing_deg=0.0, distance_m=ALREADY_HERE_M - 0.1)

    assert plan.already_there
    assert plan.distance_m == 0.0
    assert "already" in plan.summary.lower()


# ----------------------------------------------------------------------
# Standing beside them


def test_beside_ends_up_alongside_not_in_front():
    person_bearing, person_distance = 0.0, 3.0
    plan = approach_person(person_bearing, person_distance, position="beside")

    # Convert the goal back to x (forward) and y (left) to check the geometry.
    goal_x = plan.distance_m * math.cos(math.radians(plan.bearing_deg))
    goal_y = plan.distance_m * math.sin(math.radians(plan.bearing_deg))

    # Same depth as the person...
    assert abs(goal_x - person_distance) < 0.01
    # ...but a body width to one side of them.
    assert abs(abs(goal_y) - BESIDE_OFFSET_M) < 0.01


def test_beside_barely_stops_short_because_the_offset_is_the_separation():
    plan = approach_person(0.0, 3.0, position="beside")
    # Stopping a full personal-standoff short as well would leave it diagonally
    # behind the person rather than level with them.
    assert plan.standoff_m < PERSONAL_STANDOFF_M / 2


def test_beside_picks_the_side_that_needs_less_turning():
    """Which also avoids walking across the person's front to get there."""
    # Person off to the robot's left: going round their left is a bigger turn
    # than tucking in on the side nearer the robot's centre line.
    plan = approach_person(bearing_deg=40.0, distance_m=3.0)
    beside = approach_person(bearing_deg=40.0, distance_m=3.0, position="beside")

    assert abs(beside.bearing_deg) < abs(plan.bearing_deg)


def test_beside_says_whose_left_it_means():
    """"My left" is unambiguous; "the left" is not."""
    plan = approach_person(0.0, 3.0, position="beside")
    assert "my left" in plan.summary or "my right" in plan.summary


# ----------------------------------------------------------------------
# Refusing rather than guessing


def test_an_implausible_distance_is_refused_rather_than_walked():
    """Distance comes from apparent size, which is coarse at range.

    Beyond the limit the robot would be setting off on a long unsupervised walk
    toward a number it does not really know, which is the wrong way to be wrong.
    """
    assert approach_person(0.0, MAX_APPROACH_M + 1) is None


def test_a_nonsense_distance_is_refused():
    assert approach_person(0.0, 0.0) is None
    assert approach_person(0.0, -2.0) is None


def test_every_plan_carries_something_to_say_before_moving():
    """Deixis is ambiguous enough that announcing is part of the design.

    Humans agree with each other on "here" only about two thirds of the time,
    so the interpretation has to be voiced while there is still time to object.
    """
    for position in ("in_front", "beside"):
        for distance in (1.0, 3.0, 7.0):
            plan = approach_person(10.0, distance, position=position)
            assert plan is not None
            assert plan.summary.strip()


def test_an_unknown_position_falls_back_to_coming_in_front():
    plan = approach_person(0.0, 3.0, position="somewhere_odd")
    assert plan.standoff_m == PERSONAL_STANDOFF_M
