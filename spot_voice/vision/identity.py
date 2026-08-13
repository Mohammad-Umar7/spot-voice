"""Deciding, frame by frame, whether the person in view is still the operator.

The problem this solves: face recognition works only while you are facing the
robot, and follow-me means you turn around and walk away. Ten seconds in, Spot
has nothing but your back.

The answer is to use three layers, each doing the job it is actually good at:

===============  ==================  ==================  ====================
Layer            Runs                Works from behind?  Cost
===============  ==================  ==================  ====================
Geometry         every frame, 8 Hz   yes                 free
Appearance       ~2 Hz, and on a     **yes**             under a millisecond
                 broken lock
Face             acquisition, then   no                  50-100 ms
                 opportunistically
===============  ==================  ==================  ====================

* **Geometry** (overlap and size versus the previous sighting) carries normal
  walking. It is what stops a passer-by stealing the follow.
* **Appearance** (the colours of what you are wearing) is the layer that makes
  following-from-behind work at all, and the one that re-finds you after you
  round a corner.
* **Face** introduces you once at the start, and re-confirms for free on the
  occasions you glance back at the robot -- which also refreshes the appearance
  memory, so it tracks changing light.

Nothing here moves the robot. It answers "which box is the operator", and the
follow controller does the driving.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from . import appearance as appearance_module
from .appearance import AppearanceSignature

LOGGER = logging.getLogger(__name__)

#: Minimum box overlap for the geometric layer to accept a detection.
STICKY_IOU = 0.10
#: Allowed change in apparent height between consecutive sightings. This is what
#: rejects someone stepping between you and the robot: being nearer the camera
#: makes their box far taller than yours.
STICKY_HEIGHT_RATIO = (0.6, 1.6)

#: How often to re-check appearance while the geometric lock is holding. The
#: geometry is reliable frame to frame; appearance is the periodic audit.
APPEARANCE_CHECK_PERIOD = 0.5

#: How long the target may be unseen before the lock is dropped entirely.
LOST_AFTER_SEC = 2.0


class LockState(str, Enum):
    """What the tracker currently believes."""

    #: Nobody locked. Waiting for a face, or for a fallback acquisition.
    SEARCHING = "searching"
    #: Locked and seeing the target.
    TRACKING = "tracking"
    #: Locked but not seeing them this frame. Hold position, do not re-target.
    OCCLUDED = "occluded"


@dataclass
class Target:
    """The person being followed."""

    box: tuple[int, int, int, int, float]
    signature: AppearanceSignature
    name: str | None = None
    #: True when the lock was established by recognising a face, rather than by
    #: falling back to "nearest and most centred".
    confirmed_by_face: bool = False
    last_seen: float = field(default_factory=time.monotonic)
    last_face_check: float = 0.0


def iou(a, b) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2, ...)`` boxes."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def most_prominent(boxes, frame_width: int):
    """Nearest and most centred person -- the acquisition fallback."""
    if not boxes:
        return None
    centre = frame_width / 2.0 or 1.0
    best, best_score = None, float("-inf")
    for box in boxes:
        area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        offset = abs((box[0] + box[2]) / 2.0 - centre) / centre
        score = area * (1.0 - 0.5 * min(offset, 1.0)) * box[4]
        if score > best_score:
            best, best_score = box, score
    return best


class IdentityTracker:
    """Holds the lock on one person across frames.

    Args:
        operator: Enrolled name to follow. ``None`` means follow whoever is
            acquired, without requiring a face match.
        appearance_threshold: Similarity floor for the appearance layer.
    """

    def __init__(
        self,
        operator: str | None = None,
        appearance_threshold: float = appearance_module.MATCH_THRESHOLD,
    ) -> None:
        self.operator = operator
        self._threshold = appearance_threshold
        self._target: Target | None = None
        self._state = LockState.SEARCHING

    # ------------------------------------------------------------------

    @property
    def state(self) -> LockState:
        return self._state

    @property
    def locked(self) -> bool:
        return self._target is not None

    @property
    def target_name(self) -> str | None:
        return self._target.name if self._target else None

    @property
    def confirmed_by_face(self) -> bool:
        return bool(self._target and self._target.confirmed_by_face)

    def release(self) -> None:
        """Drop the lock and go back to searching."""
        if self._target is not None:
            LOGGER.info("Identity lock released")
        self._target = None
        self._state = LockState.SEARCHING

    # ------------------------------------------------------------------

    def acquire(
        self,
        frame,
        box: tuple[int, int, int, int, float],
        name: str | None = None,
        by_face: bool = False,
    ) -> None:
        """Lock onto a person and memorise how they look.

        This is the hand-off: the face said *who*, and from here on the stored
        appearance signature is what recognises them -- including from behind.
        """
        signature = appearance_module.describe(frame, box[:4])
        now = time.monotonic()
        self._target = Target(
            box=box,
            signature=signature,
            name=name,
            confirmed_by_face=by_face,
            last_seen=now,
            last_face_check=now if by_face else 0.0,
        )
        self._state = LockState.TRACKING
        LOGGER.info(
            "Locked onto %s (%s, appearance %s)",
            name or "operator",
            "face confirmed" if by_face else "position",
            "captured" if signature.valid else "unavailable",
        )

    # ------------------------------------------------------------------

    def update(self, frame, boxes, faces=None, face_store=None):
        """Pick the target out of this frame's detections.

        Args:
            frame: Decoded image the boxes came from.
            boxes: Person boxes, ``(x1, y1, x2, y2, confidence)``.
            faces: Optional ``[(face_box, embedding), ...]`` for this frame.
            face_store: Optional :class:`~spot_voice.vision.faces.FaceStore`.

        Returns:
            The target's box this frame, or ``None`` when it was not seen --
            in which case the caller should hold position rather than re-target.
        """
        now = time.monotonic()

        # A visible, recognised face always wins. Cheap accuracy on the
        # occasions the operator turns round to look at the robot.
        face_hit = self._match_face(boxes, faces, face_store)
        if face_hit is not None:
            box, name = face_hit
            if self._target is None:
                self.acquire(frame, box, name=name, by_face=True)
            else:
                self._confirm(frame, box, now, refresh_appearance=True)
                self._target.name = name or self._target.name
                self._target.confirmed_by_face = True
                self._target.last_face_check = now
            return box

        if self._target is None:
            # Not locked, and no face to introduce anyone. The caller decides
            # whether to fall back to most_prominent (see acquire_fallback).
            self._state = LockState.SEARCHING
            return None

        candidate = self._match_geometry(boxes)

        # Periodically audit the geometric lock against the appearance memory.
        # This is what catches a slow hand-off onto someone walking alongside.
        if candidate is not None and now - self._target.last_face_check > APPEARANCE_CHECK_PERIOD:
            if not self._appearance_agrees(frame, candidate):
                LOGGER.info("Appearance check rejected the geometric candidate")
                candidate = None

        # Geometry lost them -- try to re-find by appearance alone. This is the
        # path that works from behind, and the one that recovers them after a
        # brief occlusion.
        if candidate is None:
            candidate = self._match_appearance(frame, boxes)

        if candidate is None:
            self._state = LockState.OCCLUDED
            if now - self._target.last_seen > LOST_AFTER_SEC:
                self.release()
            return None

        self._confirm(frame, candidate, now, refresh_appearance=False)
        return candidate

    def acquire_fallback(self, frame, boxes, frame_width: int):
        """Lock onto the most prominent person, with no face involved.

        Used when nobody is enrolled, or when the operator asked to follow and
        the robot cannot see a face -- following the person standing in front of
        it is better than refusing.
        """
        box = most_prominent(boxes, frame_width)
        if box is None:
            return None
        self.acquire(frame, box, name=None, by_face=False)
        return box

    # ------------------------------------------------------------------

    def _confirm(self, frame, box, now: float, refresh_appearance: bool) -> None:
        """Record a sighting and, optionally, update the appearance memory."""
        assert self._target is not None
        self._target.box = box
        self._target.last_seen = now
        self._state = LockState.TRACKING
        if refresh_appearance:
            fresh = appearance_module.describe(frame, box[:4])
            if fresh.valid:
                self._target.signature = appearance_module.blend(
                    self._target.signature, fresh
                )

    def _match_face(self, boxes, faces, face_store):
        """Find a person box whose face is recognised as the operator."""
        if not faces or face_store is None or not boxes:
            return None
        from .faces import face_inside

        for face_box, embedding in faces:
            name, score = face_store.identify(embedding)
            if name is None:
                continue
            if self.operator is not None and name != self.operator:
                continue
            for person in boxes:
                if face_inside(face_box, person):
                    LOGGER.debug("Face matched %s (%.2f)", name, score)
                    return person, name
        return None

    def _match_geometry(self, boxes):
        """Best detection consistent with the previous sighting."""
        if self._target is None or not boxes:
            return None
        previous = self._target.box
        previous_height = max(1.0, float(previous[3] - previous[1]))
        best, best_overlap = None, 0.0
        for box in boxes:
            overlap = iou(box, previous)
            if overlap < STICKY_IOU:
                continue
            ratio = max(0.0, float(box[3] - box[1])) / previous_height
            if not (STICKY_HEIGHT_RATIO[0] <= ratio <= STICKY_HEIGHT_RATIO[1]):
                continue
            if overlap > best_overlap:
                best, best_overlap = box, overlap
        return best

    def _match_appearance(self, frame, boxes):
        """Best detection matching the stored appearance signature."""
        if self._target is None or not boxes or not self._target.signature.valid:
            return None
        best, best_score = None, 0.0
        for box in boxes:
            candidate = appearance_module.describe(frame, box[:4])
            score = appearance_module.similarity(self._target.signature, candidate)
            if score > best_score:
                best, best_score = box, score
        if best_score >= self._threshold:
            LOGGER.debug("Re-found by appearance (%.2f)", best_score)
            return best
        return None

    def _appearance_agrees(self, frame, box) -> bool:
        """True when a box still looks like the person we memorised."""
        if self._target is None or not self._target.signature.valid:
            return True  # nothing to check against; trust the geometry
        candidate = appearance_module.describe(frame, box[:4])
        if not candidate.valid:
            return True
        return appearance_module.similarity(self._target.signature, candidate) >= self._threshold
