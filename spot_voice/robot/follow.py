"""Follow-me: a background thread that walks Spot along behind a person.

Shape of the loop, 5-10 Hz:

    front camera frame -> person detection -> pick the best candidate
      -> P-control (yaw from horizontal offset, forward speed from apparent size)
      -> capped velocity command with the 0.6 s dead-man expiry

Claude only ever starts and stops this; it never steers. Spot's own obstacle
avoidance stays on at factory defaults and is the safety net -- this controller
deliberately never touches obstacle padding, and every velocity it produces goes
through :func:`~spot_voice.robot.limits.clamp_velocity`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Protocol, Sequence

from ..vision.identity import IdentityTracker, LockState
from .base import RobotInterface
from .limits import MAX_VROT, MAX_VX, clamp_velocity

LOGGER = logging.getLogger(__name__)

#: Control loop rate.
LOOP_HZ = 8.0

#: Fraction of frame height a person occupies at the desired standoff (~1.5 m).
#:
#: This is the one number that needs calibrating on the real robot: stand at
#: 1.5 m in front of Spot, run follow-me, and read the logged ``bbox_frac``.
TARGET_BBOX_HEIGHT_FRACTION = 0.55

#: Proportional gains.
KP_YAW = 1.1  # rad/s per unit of normalised horizontal error
KP_FORWARD = 0.9  # m/s per unit of normalised size error

#: Deadbands, in normalised error units. Inside these the robot holds still,
#: which stops it twitching when the operator is basically where it wants them.
YAW_DEADBAND = 0.08
FORWARD_DEADBAND = 0.10

#: Stop closing in when the person is nearer than this (negative size error).
TOO_CLOSE_ERROR = -0.15

#: Declare the target lost after this many seconds with no detection.
#:
#: Raised from 2.0 after watching it on the robot. Two seconds sounds generous
#: and is not: a person turning a corner, glancing back, or briefly passing
#: behind someone else vanishes for about that long. Every drop then triggers a
#: fresh acquisition, and a fresh acquisition takes *whoever is in front* --
#: which is how it ended up following the wrong person. Holding the lock through
#: short gaps is what keeps it on the right one.
LOST_AFTER_SEC = 4.0

#: Detection confidence floor for the YOLO detector.
MIN_CONFIDENCE = 0.4

#: How long ``start()`` waits for the detector to load before replying anyway.
#: Generous, because the first YOLO run may be downloading model weights.
DETECTOR_LOAD_TIMEOUT_SEC = 20.0

#: Minimum box overlap (intersection over union) with the last sighting for a
#: detection to count as the same person. Lenient: at 8 Hz a walking person
#: moves only a few dozen pixels between frames.
STICKY_IOU = 0.10

#: Allowed change in apparent height between consecutive sightings of the
#: tracked person. A person stepping between Spot and its target is much closer
#: to the camera, so their box is far taller -- this is what stops the lock
#: jumping to them even though their box overlaps the target's.
STICKY_HEIGHT_RATIO = (0.6, 1.6)

#: Yaw rate used when sweeping to look for the operator. Deliberately slow: a
#: robot spinning quickly near people is alarming, and a slow sweep gives the
#: face recogniser several frames per bearing rather than a smear.
SEARCH_YAW_RATE = 0.3

#: Give up sweeping after this long and say so, rather than turning forever.
SEARCH_TIMEOUT_SEC = 20.0

#: How often to run face recognition during a follow. Faces are only visible on
#: the occasions the operator glances back, and recognition costs ~100 ms, so
#: this is an opportunistic top-up rather than part of the tracking loop.
FACE_CHECK_PERIOD_SEC = 1.5


class PersonDetector(Protocol):
    """Anything that can find people in a JPEG frame."""

    def detect(self, frame_jpeg: bytes) -> Sequence[tuple[int, int, int, int, float]]:
        """Return ``(x1, y1, x2, y2, confidence)`` boxes, one per person."""

    @property
    def frame_size(self) -> tuple[int, int]:
        """Return ``(width, height)`` of the last processed frame."""


class YoloPersonDetector:
    """Person detection with ultralytics YOLOv8-nano. CPU is fine at this rate.

    The model file is downloaded by ultralytics on first use, so the first run
    needs internet. After that it is cached locally.
    """

    def __init__(self, weights: str = "yolov8n.pt") -> None:
        from ultralytics import YOLO  # imported lazily: heavy, and optional

        self._model = YOLO(weights)
        self._frame_size = (640, 480)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    def detect(self, frame_jpeg: bytes) -> list[tuple[int, int, int, int, float]]:
        import cv2
        import numpy as np

        buffer = np.frombuffer(frame_jpeg, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            return []
        height, width = frame.shape[:2]
        self._frame_size = (width, height)

        # classes=[0] restricts inference to the COCO "person" class.
        results = self._model.predict(
            frame, classes=[0], conf=MIN_CONFIDENCE, verbose=False
        )
        boxes: list[tuple[int, int, int, int, float]] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                boxes.append((int(x1), int(y1), int(x2), int(y2), float(box.conf[0])))
        return boxes


class SimulatedPersonDetector:
    """Mock-mode detector so follow-me is exercisable at a desk.

    Wraps :class:`~spot_voice.robot.mock.SimulatedPerson`, which walks a
    plausible bounding box left and right and occasionally disappears, so both
    the tracking path and the "I lost you" path get rehearsed.
    """

    def __init__(self, seed: int | None = None) -> None:
        from .mock import SimulatedPerson

        self._person = SimulatedPerson(seed=seed)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._person.frame_size

    def detect(self, frame_jpeg: bytes) -> list[tuple[int, int, int, int, float]]:
        box = self._person.detect(frame_jpeg)
        if box is None:
            return []
        return [(box[0], box[1], box[2], box[3], 0.9)]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
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


def pick_target(
    boxes: Sequence[tuple[int, int, int, int, float]],
    frame_width: int,
    previous: tuple[int, int, int, int, float] | None = None,
) -> tuple[int, int, int, int, float] | None:
    """Choose which detected person to follow.

    Two modes:

    * **Acquiring** (``previous is None``): score each box by area (nearer
      people are bigger) and by how centred it is -- so saying "follow me" locks
      onto whoever is standing front-and-centre of the robot.
    * **Tracking** (``previous`` given): stick to the person already being
      followed. A detection only counts as them if it overlaps the last
      sighting *and* is a similar apparent size; among matches, the largest
      overlap wins. If nobody matches, return ``None`` -- the controller holds
      still rather than transferring the lock to whoever happens to look
      biggest, which is exactly what a person walking between Spot and its
      target would otherwise cause.
    """
    if not boxes:
        return None

    if previous is not None:
        previous_height = max(1.0, float(previous[3] - previous[1]))
        best_match, best_overlap = None, 0.0
        for box in boxes:
            overlap = _iou(box, previous)
            if overlap < STICKY_IOU:
                continue
            ratio = max(0.0, float(box[3] - box[1])) / previous_height
            if not (STICKY_HEIGHT_RATIO[0] <= ratio <= STICKY_HEIGHT_RATIO[1]):
                continue
            if overlap > best_overlap:
                best_match, best_overlap = box, overlap
        return best_match
    centre = frame_width / 2.0 or 1.0
    best, best_score = None, float("-inf")
    for box in boxes:
        x1, y1, x2, y2, conf = box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        box_centre = (x1 + x2) / 2.0
        offset = abs(box_centre - centre) / centre
        score = area * (1.0 - 0.5 * min(offset, 1.0)) * conf
        if score > best_score:
            best, best_score = box, score
    return best


def compute_velocity(
    box: tuple[int, int, int, int, float],
    frame_size: tuple[int, int],
    target_fraction: float = TARGET_BBOX_HEIGHT_FRACTION,
):
    """Turn one bounding box into a capped velocity command.

    Args:
        box: ``(x1, y1, x2, y2, confidence)``.
        frame_size: ``(width, height)`` of the frame the box came from.
        target_fraction: Desired box height as a fraction of frame height.

    Returns:
        A :class:`~spot_voice.robot.limits.Velocity` within the hard caps.
    """
    width, height = frame_size
    width = width or 1
    height = height or 1
    x1, y1, x2, y2, _conf = box

    box_centre = (x1 + x2) / 2.0
    error_x = (box_centre - width / 2.0) / (width / 2.0)  # -1 (left) .. +1 (right)
    v_rot = 0.0
    if abs(error_x) > YAW_DEADBAND:
        # Person to the right of centre => negative yaw (clockwise) to face them.
        v_rot = max(-MAX_VROT, min(MAX_VROT, -KP_YAW * error_x))

    box_fraction = max(0.0, (y2 - y1)) / height
    error_size = (target_fraction - box_fraction) / target_fraction
    v_x = 0.0
    if error_size > FORWARD_DEADBAND:
        v_x = max(0.0, min(MAX_VX, KP_FORWARD * error_size))
    elif error_size < TOO_CLOSE_ERROR:
        # Too close: hold position rather than backing up blind.
        v_x = 0.0

    return clamp_velocity(v_x, 0.0, v_rot)


class FollowController:
    """Owns the follow-me thread.

    Args:
        robot: The robot to drive.
        detector_factory: Builds the detector on the worker thread, so a slow
            model load never blocks the voice loop.
        say: Callback used for spoken notifications ("I lost you.").
    """

    def __init__(
        self,
        robot: RobotInterface,
        detector_factory: Callable[[], PersonDetector],
        say: Callable[[str], None] | None = None,
        tracker_factory: Callable[[], "IdentityTracker"] | None = None,
        face_factory: Callable[[], tuple] | None = None,
    ) -> None:
        self._robot = robot
        self._detector_factory = detector_factory
        self._say = say or (lambda _text: None)
        self._tracker_factory = tracker_factory
        # Returns (recogniser, store). Built on the worker thread because
        # loading a face model takes seconds and must not block the voice loop.
        self._face_factory = face_factory
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._error: str | None = None

    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True while the follow thread is running."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> tuple[bool, str]:
        """Start following. Returns ``(ok, spoken_message)``."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True, "I'm already following you."
            self._error = None
            self._stop_event.clear()
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._run, name="follow-me", daemon=True
            )
            self._thread.start()

        # Wait until the detector has actually loaded, so a missing dependency is
        # reported now rather than silently a second later. The model load is
        # what the timeout is really for; in mock mode this returns immediately.
        self._ready.wait(timeout=DETECTOR_LOAD_TIMEOUT_SEC)
        if self._error:
            return False, self._error
        if not self._ready.is_set():
            LOGGER.info("Detector still loading; following will begin shortly")
        return True, "Following you. Say stop when you want me to hold."

    def stop(self, timeout: float = 2.0) -> tuple[bool, str]:
        """Kill the follow thread immediately and bring the robot to a halt."""
        with self._lock:
            thread = self._thread
            self._thread = None
        self._stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        try:
            self._robot.drive(0.0, 0.0, 0.0)
        except Exception:  # pragma: no cover - stopping must never raise
            LOGGER.debug("zero-velocity command failed during stop", exc_info=True)
        if thread is None:
            return True, "I wasn't following anyone."
        return True, "Stopped following."

    # ------------------------------------------------------------------

    def _sweep(self, elapsed: float) -> None:
        """Turn slowly in place to look for the operator.

        This is what "look for me first" means physically. It only ever rotates
        -- never translates -- so the robot stays where you left it, and it goes
        through the same capped velocity path as everything else. After
        ``SEARCH_TIMEOUT_SEC`` it stops and says so rather than spinning
        indefinitely.
        """
        if elapsed > SEARCH_TIMEOUT_SEC:
            self._robot.drive(0.0, 0.0, 0.0)
            if not self._gave_up_searching:
                self._gave_up_searching = True
                LOGGER.info("Gave up sweeping for the operator")
                self._say("I can't find you. Step in front of me and say follow me again.")
            return
        self._robot.drive(0.0, 0.0, SEARCH_YAW_RATE)

    def _build_faces(self):
        """Construct the face recogniser and store, or ``(None, None)``.

        Never fatal: without faces, follow-me still works by locking onto
        whoever is in front of the robot, which is how it behaved before face
        recognition existed.
        """
        if self._face_factory is None:
            return None, None
        try:
            return self._face_factory()
        except Exception as exc:
            LOGGER.warning("Face recognition unavailable (%s)", exc, exc_info=True)
            return None, None

    def _run(self) -> None:
        try:
            detector = self._detector_factory()
        except Exception as exc:
            LOGGER.warning("Follow-me detector unavailable", exc_info=True)
            self._error = (
                "I can't start following -- my person detector isn't available. "
                f"{type(exc).__name__}."
            )
            self._ready.set()
            return
        self._ready.set()

        tracker = self._tracker_factory() if self._tracker_factory else IdentityTracker()
        recogniser, face_store = self._build_faces()

        period = 1.0 / LOOP_HZ
        last_seen = time.monotonic()
        last_face_check = 0.0
        announced_lost = False
        # When set, the moment the current sweep began. None while tracking.
        searching_since: float | None = None
        self._gave_up_searching = False
        LOGGER.info(
            "Follow-me started (operator=%s, faces=%s)",
            tracker.operator or "whoever is in front",
            "on" if recogniser else "off",
        )

        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            now = cycle_start
            try:
                capture = self._robot.capture_image("front")
                boxes: Sequence[tuple[int, int, int, int, float]] = []
                image = None
                if capture.ok and capture.image_jpeg:
                    boxes = detector.detect(capture.image_jpeg)
                    image = decode_jpeg(capture.image_jpeg)

                # Faces are expensive and only visible when the operator turns
                # round, so this is an occasional top-up, not part of the loop.
                faces = None
                if recogniser is not None and now - last_face_check > FACE_CHECK_PERIOD_SEC:
                    last_face_check = now
                    try:
                        faces = recogniser.detect(image)
                    except Exception:
                        LOGGER.debug("face detection failed this frame", exc_info=True)

                target = tracker.update(image, boxes, faces=faces, face_store=face_store)

                # Nothing locked, and no recognised face to introduce anyone.
                # If we are not looking for a specific person, following whoever
                # stands in front of the robot is the right answer.
                if target is None and not tracker.locked and boxes:
                    if tracker.operator is None or recogniser is None:
                        target = tracker.acquire_fallback(
                            image, boxes, detector.frame_size[0]
                        )

                if target is None:
                    if searching_since is None:
                        searching_since = now
                    self._sweep(now - searching_since)
                    if not announced_lost and now - last_seen > LOST_AFTER_SEC:
                        announced_lost = True
                        tracker.release()
                        LOGGER.info("Follow-me lost the target")
                        self._say("I lost you.")
                else:
                    if searching_since is not None:
                        LOGGER.info("Reacquired after sweeping")
                    searching_since = None
                    last_seen = now
                    announced_lost = False
                    velocity = compute_velocity(target, detector.frame_size)
                    self._robot.drive(*velocity.as_tuple())
            except Exception:
                LOGGER.warning("Follow-me iteration failed", exc_info=True)
                try:
                    self._robot.drive(0.0, 0.0, 0.0)
                except Exception:
                    pass
                # Back off briefly rather than hammering a broken link.
                self._stop_event.wait(0.5)

            elapsed = time.monotonic() - cycle_start
            self._stop_event.wait(max(0.0, period - elapsed))

        try:
            self._robot.drive(0.0, 0.0, 0.0)
        except Exception:  # pragma: no cover
            pass
        LOGGER.info("Follow-me stopped")


def decode_jpeg(data: bytes):
    """Decode JPEG bytes to a BGR image, or ``None``.

    The appearance layer needs pixels, not just boxes -- it is the colours of
    what the person is wearing that make following-from-behind work.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover - opencv is a hard dependency
        return None
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def make_detector_factory(mock: bool) -> Callable[[], PersonDetector]:
    """Return the detector factory appropriate to the current mode."""
    if mock:
        return lambda: SimulatedPersonDetector()
    return lambda: YoloPersonDetector()
