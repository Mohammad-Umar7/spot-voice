"""Face recognition: working out which of the people in frame is the operator.

Used at two moments only, never for continuous tracking:

* **Enrollment** -- once, offline, via ``python -m spot_voice --enroll <name>``.
* **Acquisition** -- when you say "follow me", to decide which detected person
  is you. After that, :mod:`spot_voice.vision.appearance` takes over, because
  you will be walking away and your face will not be visible.

It is also used *opportunistically* during a follow: if you happen to glance
back at the robot and your face is recognised, the tracker re-confirms the lock
and refreshes the appearance memory. Free accuracy whenever you look round.

Privacy: enrollment stores a numeric embedding, not a photograph, in a local
file. Nothing is uploaded. Faces of people who have not been enrolled are
compared in memory and discarded -- they are never written to disk.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

#: Cosine similarity above which two embeddings are the same person. ArcFace
#: embeddings typically separate identities well above 0.5; 0.42 is deliberately
#: a little forgiving, since the cost of a miss is a re-scan, and the geometric
#: and appearance layers both have to agree before anything moves.
MATCH_THRESHOLD = 0.42

#: Filename for the enrollment store inside the work directory.
STORE_FILENAME = "faces.json"


@dataclass
class FaceMatch:
    """A recognised face."""

    name: str
    score: float
    box: tuple[int, int, int, int]


class FaceStore:
    """Enrolled identities, persisted as embeddings in a JSON file.

    Args:
        path: Where the store lives.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._people: dict[str, list[list[float]]] = {}
        self.load()

    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def names(self) -> list[str]:
        return sorted(self._people)

    @property
    def is_empty(self) -> bool:
        return not self._people

    def load(self) -> None:
        """Read the store from disk. A missing or corrupt file is not fatal."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._people = {
                str(name): [[float(v) for v in vector] for vector in vectors]
                for name, vectors in raw.items()
            }
        except Exception:
            LOGGER.warning("Could not read %s; starting empty", self._path, exc_info=True)
            self._people = {}

    def save(self) -> None:
        """Write the store to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._people), encoding="utf-8")

    def add(self, name: str, embedding: list[float]) -> None:
        """Record another sample for ``name``.

        Several samples per person is the point: different angles and lighting
        make acquisition far more reliable than a single reference shot.
        """
        self._people.setdefault(name, []).append([float(value) for value in embedding])

    def forget(self, name: str) -> bool:
        """Remove someone. Returns True if they were enrolled."""
        return self._people.pop(name, None) is not None

    def identify(self, embedding: list[float], threshold: float = MATCH_THRESHOLD):
        """Find the closest enrolled identity.

        Returns:
            ``(name, score)`` for the best match above ``threshold``, else
            ``(None, best_score)`` so callers can log how close it got.
        """
        best_name, best_score = None, 0.0
        for name, vectors in self._people.items():
            for stored in vectors:
                score = cosine_similarity(embedding, stored)
                if score > best_score:
                    best_name, best_score = name, score
        if best_score >= threshold:
            return best_name, best_score
        return None, best_score


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two embeddings, clamped to ``[-1, 1]``."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class FaceRecogniser:
    """Detects faces in a frame and embeds them.

    Wraps InsightFace, which ships CPU wheels for Windows and does detection and
    embedding in one pass. Imported lazily so that everything else -- mock mode,
    the reflex lane, the tests -- runs on a machine without it.
    """

    def __init__(self, model_name: str = "buffalo_l", det_size: int = 640) -> None:
        import warnings

        from insightface.app import FaceAnalysis

        # insightface calls into numpy and scikit-image in ways both now warn
        # about. Nothing here can act on them -- they are about the internals of
        # a dependency -- and they print on every single detection, which buries
        # the sample-by-sample feedback the operator is standing there reading.
        warnings.filterwarnings("ignore", category=FutureWarning, module="insightface.*")

        self._app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        LOGGER.info("Face recogniser ready (%s, CPU)", model_name)

    def detect(self, frame) -> list[tuple[tuple[int, int, int, int], list[float]]]:
        """Return ``(box, embedding)`` for every face found in a BGR frame."""
        if frame is None:
            return []
        results = []
        for face in self._app.get(frame):
            x1, y1, x2, y2 = (int(value) for value in face.bbox)
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            results.append(((x1, y1, x2, y2), [float(value) for value in embedding]))
        return results


def face_inside(
    face_box: tuple[int, int, int, int], person_box: tuple[int, int, int, int, float]
) -> bool:
    """True when a face box sits within a person box.

    This is how a recognised face is attributed to a detected body: the face
    tells us *who*, the body box is what the follow controller actually tracks.
    """
    face_cx = (face_box[0] + face_box[2]) / 2.0
    face_cy = (face_box[1] + face_box[3]) / 2.0
    return (
        person_box[0] <= face_cx <= person_box[2]
        and person_box[1] <= face_cy <= person_box[3]
    )
