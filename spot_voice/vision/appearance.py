"""Appearance signatures: recognising a person from behind.

This module exists because of a specific problem. Face recognition can confirm
who you are only while you are facing the robot -- and the entire point of
follow-me is that you turn around and walk away. From that moment on Spot sees
your back, and no face recogniser on earth will help it.

So the face is used once, as an introduction. At the instant it confirms you, we
capture a signature of your *whole body* -- the colours of what you are wearing,
split into torso and legs. That signature works from any angle, including from
directly behind, and it is what actually follows you.

The signature is a coarse hue/saturation histogram per band. Deliberately
coarse: it should say "the person in the blue jacket and dark trousers", not
attempt to be a fingerprint. Cheap enough to run every frame (well under a
millisecond) and it degrades gracefully as lighting changes.

Known limit, worth stating plainly: in a facility where everyone wears the same
hi-vis and the same overalls, this cannot tell people apart. The geometric lock
and the "I lost you" recovery are the safety net for that case.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)

#: Histogram bins. Coarse on purpose -- see the module docstring.
HUE_BINS = 12
SATURATION_BINS = 4

#: Fraction of the person's box treated as torso and as legs. The top slice is
#: dropped because it is mostly head and background, which tells us little about
#: clothing and changes a lot as the person turns.
HEAD_SKIP = 0.15
TORSO_END = 0.55

#: Similarity below which two signatures are considered different people.
#: Tuned to be forgiving: a false "that isn't you" costs a re-acquisition, while
#: a false "that is you" means following a stranger.
MATCH_THRESHOLD = 0.55

#: How much of a confirmed sighting to blend into the stored signature. Low, so
#: the memory tracks gradual lighting change without drifting onto whoever
#: happens to be nearby.
UPDATE_RATE = 0.15


@dataclass
class AppearanceSignature:
    """A coarse colour description of a person's torso and legs."""

    torso: list[float] = field(default_factory=list)
    legs: list[float] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return bool(self.torso) or bool(self.legs)


def _normalise(values: list[float]) -> list[float]:
    """Scale a histogram so its entries sum to one."""
    total = sum(values)
    if total <= 0:
        return [0.0] * len(values)
    return [value / total for value in values]


def _histogram(patch) -> list[float]:
    """Build a coarse hue/saturation histogram from a BGR image patch."""
    import cv2
    import numpy as np

    if patch is None or patch.size == 0:
        return []

    # Greyscale frames (Spot's body fisheyes) have no colour to work with, so
    # fall back to a brightness histogram. Weaker, but it still separates a
    # light coat from a dark one.
    if patch.ndim == 2 or patch.shape[2] == 1:
        grey = patch if patch.ndim == 2 else patch[:, :, 0]
        counts = cv2.calcHist([grey], [0], None, [HUE_BINS * SATURATION_BINS], [0, 256])
        return _normalise([float(value) for value in counts.flatten()])

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # Ignore very dark pixels: their hue is noise, and shadow would otherwise
    # dominate the signature.
    mask = (hsv[:, :, 2] > 40).astype("uint8") * 255
    if not int(np.count_nonzero(mask)):
        mask = None
    counts = cv2.calcHist(
        [hsv], [0, 1], mask, [HUE_BINS, SATURATION_BINS], [0, 180, 0, 256]
    )
    return _normalise([float(value) for value in counts.flatten()])


def describe(frame, box: tuple[int, int, int, int]) -> AppearanceSignature:
    """Build a signature for the person inside ``box``.

    Args:
        frame: Decoded image (BGR or greyscale) the box came from.
        box: ``(x1, y1, x2, y2)`` pixel bounds of the person.

    Returns:
        The signature. Empty when the box is degenerate or off-frame.
    """
    if frame is None:
        return AppearanceSignature()
    height, width = frame.shape[:2]

    x1 = max(0, min(int(box[0]), width))
    x2 = max(0, min(int(box[2]), width))
    y1 = max(0, min(int(box[1]), height))
    y2 = max(0, min(int(box[3]), height))
    if x2 - x1 < 4 or y2 - y1 < 8:
        return AppearanceSignature()

    box_height = y2 - y1
    torso_top = y1 + int(box_height * HEAD_SKIP)
    torso_bottom = y1 + int(box_height * TORSO_END)

    return AppearanceSignature(
        torso=_histogram(frame[torso_top:torso_bottom, x1:x2]),
        legs=_histogram(frame[torso_bottom:y2, x1:x2]),
    )


def _intersection(a: list[float], b: list[float]) -> float:
    """Histogram intersection: 1.0 identical, 0.0 nothing in common."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(min(left, right) for left, right in zip(a, b))


def similarity(a: AppearanceSignature, b: AppearanceSignature) -> float:
    """How alike two signatures are, from 0.0 to 1.0.

    Torso is weighted above legs: it is the larger, more distinctive region, and
    legs are more often clipped by the bottom of the frame when someone is close.
    """
    if not a.valid or not b.valid:
        return 0.0
    torso = _intersection(a.torso, b.torso)
    legs = _intersection(a.legs, b.legs)
    if a.torso and b.torso and a.legs and b.legs:
        return 0.65 * torso + 0.35 * legs
    return torso if (a.torso and b.torso) else legs


def matches(a: AppearanceSignature, b: AppearanceSignature, threshold: float = MATCH_THRESHOLD) -> bool:
    """True when two signatures are similar enough to be the same person."""
    return similarity(a, b) >= threshold


def blend(stored: AppearanceSignature, fresh: AppearanceSignature, rate: float = UPDATE_RATE) -> AppearanceSignature:
    """Fold a confirmed sighting into the stored signature.

    Lets the memory follow gradual change -- walking from a lit aisle into a
    shaded one -- without letting it drift onto a different person. Only ever
    called with a sighting the tracker already accepted.
    """
    if not stored.valid:
        return fresh
    if not fresh.valid:
        return stored

    def mix(old: list[float], new: list[float]) -> list[float]:
        if not old or not new or len(old) != len(new):
            return old or new
        return _normalise(
            [(1 - rate) * o + rate * n for o, n in zip(old, new)]
        )

    return AppearanceSignature(
        torso=mix(stored.torso, fresh.torso), legs=mix(stored.legs, fresh.legs)
    )
