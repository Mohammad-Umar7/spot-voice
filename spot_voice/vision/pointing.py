"""Working out where someone is pointing.

"Stand over there" plus an outstretched arm. The honest version of what this can
and cannot do, because it decides the design:

* **Bearing is reliable.** The direction of a pointing arm in the image maps
  cleanly onto an angle left or right of the robot. This is the part that works.
* **Distance is not.** A pointing gesture carries no distance information at
  all -- the same arm angle means "that bench" and "the far wall". No amount of
  cleverness recovers a number the gesture never contained.

So distance comes from somewhere else: Spot's stereo depth cameras report how
far away things actually are along that bearing, and the robot walks to a
standoff short of the first thing in the way. If depth is unavailable it falls
back to a modest default rather than guessing big.

Because the reading can be wrong, the flow always *says what it understood*
before moving -- "about thirty degrees to my left, roughly three metres" -- so a
misread is something the operator corrects, not something the robot surprises
them with.

Keypoints come from YOLOv8-pose, which returns the 17 COCO body points. Only
four matter here: the shoulders and wrists.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

#: COCO keypoint indices used for the pointing ray.
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10

#: Keypoint confidence below which a joint is treated as not seen.
MIN_KEYPOINT_CONFIDENCE = 0.4

#: An arm counts as "pointing" only when it is extended this far, measured as
#: shoulder-to-wrist distance over shoulder width. A resting arm scores near 1;
#: a raised, extended one scores well above it.
MIN_EXTENSION_RATIO = 1.25

#: Horizontal field of view of Spot's front fisheye cameras, in degrees.
#: Used to turn a pixel offset into a bearing. Approximate -- it is the number
#: to adjust if pointing reads consistently short or wide on the real robot.
CAMERA_HFOV_DEG = 100.0

#: How far past the wrist to project the pointing ray, as a multiple of the
#: shoulder-to-wrist span. People indicate somewhere beyond their hand, not at
#: it; 1.0 reads real gestures well without saturating at the frame edge.
RAY_EXTENSION = 1.0

#: Distance used when depth is unavailable. Deliberately modest: walking too far
#: on a guess is worse than walking too short and being asked to continue.
DEFAULT_DISTANCE_M = 2.0

#: Never walk closer than this to whatever the depth camera found.
STANDOFF_M = 1.0


@dataclass
class PointingResult:
    """Where the operator appears to be pointing."""

    #: Degrees from straight ahead. Positive is to the robot's left.
    bearing_deg: float
    #: Metres to travel. Never more than the move cap.
    distance_m: float
    #: Which arm was used, for the spoken confirmation.
    arm: str
    #: How confident the gesture read is, 0-1.
    confidence: float
    #: True when ``distance_m`` came from depth rather than the default.
    distance_measured: bool = False

    def describe(self) -> str:
        """A sentence the robot can say before it moves."""
        side = "left" if self.bearing_deg > 0 else "right"
        if abs(self.bearing_deg) < 8:
            where = "straight ahead"
        else:
            where = f"about {abs(self.bearing_deg):.0f} degrees to my {side}"
        how_far = (
            f"{self.distance_m:.1f} metres"
            if self.distance_measured
            else f"roughly {self.distance_m:.1f} metres"
        )
        return f"{where}, {how_far}"


def _visible(keypoints, index: int) -> bool:
    """True when a keypoint was detected confidently enough to use."""
    if index >= len(keypoints):
        return False
    point = keypoints[index]
    return len(point) > 2 and point[2] >= MIN_KEYPOINT_CONFIDENCE


def _extension(keypoints, shoulder: int, wrist: int, shoulder_width: float) -> float:
    """How extended an arm is, relative to shoulder width."""
    if not (_visible(keypoints, shoulder) and _visible(keypoints, wrist)):
        return 0.0
    dx = keypoints[wrist][0] - keypoints[shoulder][0]
    dy = keypoints[wrist][1] - keypoints[shoulder][1]
    return math.hypot(dx, dy) / max(shoulder_width, 1.0)


def detect_pointing(keypoints, frame_width: int, hfov_deg: float = CAMERA_HFOV_DEG):
    """Read a pointing gesture from one person's pose keypoints.

    Args:
        keypoints: 17 COCO keypoints as ``(x, y, confidence)``.
        frame_width: Width of the frame the keypoints came from.
        hfov_deg: Horizontal field of view of the camera.

    Returns:
        A :class:`PointingResult` with bearing and a default distance, or
        ``None`` when neither arm is extended enough to count as pointing.
    """
    if keypoints is None or len(keypoints) < 11 or frame_width <= 0:
        return None
    if not (_visible(keypoints, LEFT_SHOULDER) and _visible(keypoints, RIGHT_SHOULDER)):
        return None

    shoulder_width = abs(keypoints[LEFT_SHOULDER][0] - keypoints[RIGHT_SHOULDER][0])
    if shoulder_width < 4:
        return None

    left = _extension(keypoints, LEFT_SHOULDER, LEFT_WRIST, shoulder_width)
    right = _extension(keypoints, RIGHT_SHOULDER, RIGHT_WRIST, shoulder_width)

    if max(left, right) < MIN_EXTENSION_RATIO:
        return None  # both arms down: not a pointing gesture

    if left >= right:
        shoulder, wrist, arm, extension = LEFT_SHOULDER, LEFT_WRIST, "left", left
    else:
        shoulder, wrist, arm, extension = RIGHT_SHOULDER, RIGHT_WRIST, "right", right

    # The ray runs shoulder -> wrist. A person pointing means somewhere past
    # their hand, not at it, so the ray is extended beyond the wrist.
    shoulder_x = keypoints[shoulder][0]
    wrist_x = keypoints[wrist][0]
    direction = wrist_x - shoulder_x
    projected_x = wrist_x + direction * RAY_EXTENSION

    # Pixel offset from centre -> bearing. Positive x is to the right in image
    # space, which is the robot's right, so the sign flips for a left-positive
    # bearing convention.
    #
    # The *bearing* is clamped, not the pixel. Clamping the pixel to the frame
    # first would collapse every wide gesture onto the frame edge and give the
    # same answer for "just left of you" and "right over there"; clamping at the
    # end keeps the reading monotonic until it genuinely saturates at the limit
    # of what the camera could have seen.
    offset = (projected_x - frame_width / 2.0) / (frame_width / 2.0)
    limit = hfov_deg / 2.0
    bearing = max(-limit, min(limit, -offset * limit))

    # Confidence blends how extended the arm is with how sure the joints were.
    joint_confidence = min(keypoints[shoulder][2], keypoints[wrist][2])
    confidence = max(0.0, min(1.0, (extension / 2.0) * joint_confidence))

    return PointingResult(
        bearing_deg=bearing,
        distance_m=DEFAULT_DISTANCE_M,
        arm=arm,
        confidence=confidence,
    )


def apply_measured_distance(
    result: PointingResult,
    measured_m: float | None,
    standoff_m: float = STANDOFF_M,
    max_distance_m: float = 5.0,
) -> PointingResult:
    """Replace the default distance with a real depth reading.

    Stops short of whatever the depth camera found, so "over there" never means
    "into that". A missing reading leaves the modest default in place rather
    than inventing a number.
    """
    if measured_m is None or measured_m != measured_m or measured_m <= 0:
        return result

    travel = max(0.0, measured_m - standoff_m)
    travel = min(travel, max_distance_m)
    return PointingResult(
        bearing_deg=result.bearing_deg,
        distance_m=travel,
        arm=result.arm,
        confidence=result.confidence,
        distance_measured=True,
    )


def pick_pointer(people):
    """Choose whose pointing gesture to read when several people are in frame.

    The nearest person wins -- if two people are gesturing, the operator is the
    one standing closest to the robot, and following a bystander's gesture would
    be worse than doing nothing.

    Args:
        people: ``[(keypoints, box), ...]``.

    Returns:
        The chosen ``(keypoints, box)``, or ``None``.
    """
    best, best_height = None, 0.0
    for keypoints, box in people or []:
        height = abs(box[3] - box[1])
        if height > best_height:
            best, best_height = (keypoints, box), height
    return best


class YoloPoseReader:
    """Reads pointing gestures from a JPEG using YOLOv8-pose.

    Separate from the person detector used by follow-me: that one only needs
    boxes and runs at 8 Hz, while this runs once per request and needs joints.
    Loaded lazily, because the weights download on first use.
    """

    def __init__(self, weights: str = "yolov8n-pose.pt") -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)

    def __call__(self, frame_jpeg: bytes):
        """Return a :class:`PointingResult` for the nearest pointer, or ``None``."""
        import cv2
        import numpy as np

        frame = cv2.imdecode(np.frombuffer(frame_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        height, width = frame.shape[:2]

        results = self._model.predict(frame, classes=[0], verbose=False)
        people = []
        for result in results:
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None or result.boxes is None:
                continue
            for person_index, box in enumerate(result.boxes):
                try:
                    points = keypoints.data[person_index].tolist()
                    bounds = [float(v) for v in box.xyxy[0].tolist()]
                except (IndexError, AttributeError):
                    continue
                people.append((points, bounds))

        chosen = pick_pointer(people)
        if chosen is None:
            return None
        return detect_pointing(chosen[0], width)
