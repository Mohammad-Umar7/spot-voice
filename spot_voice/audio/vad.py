"""Voice activity detection and utterance segmentation.

Audio arrives as a continuous 16 kHz mono stream. This module cuts it into
utterances: it waits for speech to start, keeps a little of the audio from
*before* the trigger so the first word is not clipped, and ends the utterance
after a stretch of silence.

Two detectors, chosen automatically:

* ``webrtcvad`` -- the standard WebRTC VAD. Fast and robust; preferred.
* an RMS energy gate -- no dependencies, used when webrtcvad is not installed so
  the program still runs.
"""

from __future__ import annotations

import collections
import logging
import math
from typing import Iterator, Protocol

LOGGER = logging.getLogger(__name__)

#: Capture format. webrtcvad only accepts 8/16/32 kHz mono 16-bit.
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2  # int16

#: Audio kept from before speech was detected, so the first phoneme survives.
#: Audio kept from *before* speech is detected, and prepended to the utterance.
#: Generous on purpose: it costs a fraction of a second of extra audio to
#: transcribe and it is what stops the first word being clipped.
PRE_SPEECH_MS = 700

#: Window the "has speech started" decision is made over. Deliberately shorter
#: than the pre-roll -- it controls how quickly the segmenter reacts, while
#: PRE_SPEECH_MS controls how much lead-in survives. Tying them together is
#: what ate the start of words.
TRIGGER_WINDOW_MS = 300
#: Silence needed to close an utterance.
POST_SPEECH_MS = 700
#: Fraction of a window that must be speech (or silence) to flip state.
TRIGGER_RATIO = 0.6
#: Utterances longer than this are cut, so a noisy room cannot stall the loop.
MAX_UTTERANCE_MS = 15000
#: Utterances shorter than this are dropped as coughs and door clicks.
MIN_UTTERANCE_MS = 300


class VoiceDetector(Protocol):
    """Decides whether one frame contains speech."""

    def is_speech(self, frame: bytes) -> bool: ...

    @property
    def name(self) -> str: ...


class WebRtcDetector:
    """WebRTC VAD wrapper.

    Args:
        aggressiveness: 0 (permissive) to 3 (aggressive). 2 is a good default
            for a lav mic in a workshop.
    """

    def __init__(self, aggressiveness: int = 2) -> None:
        import webrtcvad  # provided by 'webrtcvad' or 'webrtcvad-wheels'

        self._vad = webrtcvad.Vad(aggressiveness)

    @property
    def name(self) -> str:
        return "webrtcvad"

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != FRAME_BYTES:
            return False
        try:
            return bool(self._vad.is_speech(frame, SAMPLE_RATE))
        except Exception:  # pragma: no cover - malformed frame
            return False


class EnergyDetector:
    """RMS energy gate with a slowly adapting noise floor.

    A dependency-free fallback. Less precise than WebRTC's model, but it keeps
    the program usable when ``webrtcvad`` will not install.
    """

    def __init__(self, start_threshold: float = 500.0, margin: float = 2.5) -> None:
        self._noise_floor = start_threshold
        self._margin = margin

    @property
    def name(self) -> str:
        return "energy"

    def is_speech(self, frame: bytes) -> bool:
        rms = _frame_rms(frame)
        speech = rms > self._noise_floor * self._margin
        if not speech:
            # Adapt slowly towards the ambient level.
            self._noise_floor = 0.98 * self._noise_floor + 0.02 * max(rms, 1.0)
        return speech


def _frame_rms(frame: bytes) -> float:
    """Root-mean-square amplitude of a 16-bit PCM frame."""
    if not frame:
        return 0.0
    import array

    samples = array.array("h")
    samples.frombytes(frame[: len(frame) - (len(frame) % 2)])
    if not samples:
        return 0.0
    total = sum(float(sample) * sample for sample in samples)
    return math.sqrt(total / len(samples))


def make_detector(aggressiveness: int = 2) -> VoiceDetector:
    """Return the best voice detector available on this machine."""
    try:
        detector = WebRtcDetector(aggressiveness)
        LOGGER.info("Using webrtcvad (aggressiveness %d)", aggressiveness)
        return detector
    except Exception as exc:
        LOGGER.warning(
            "webrtcvad unavailable (%s) -- falling back to the energy detector", exc
        )
        return EnergyDetector()


class UtteranceSegmenter:
    """Turns a stream of 20 ms frames into complete utterances.

    Feed frames with :meth:`push`; it yields each finished utterance as raw
    16-bit PCM bytes.
    """

    def __init__(
        self,
        detector: VoiceDetector | None = None,
        pre_speech_ms: int = PRE_SPEECH_MS,
        post_speech_ms: int = POST_SPEECH_MS,
        max_utterance_ms: int = MAX_UTTERANCE_MS,
        min_utterance_ms: int = MIN_UTTERANCE_MS,
    ) -> None:
        self._detector = detector or make_detector()
        self._pre_frames = max(1, pre_speech_ms // FRAME_MS)
        self._post_frames = max(1, post_speech_ms // FRAME_MS)
        self._max_frames = max(1, max_utterance_ms // FRAME_MS)
        self._min_frames = max(1, min_utterance_ms // FRAME_MS)
        # Decision window, kept separate from the pre-roll. See push().
        self._trigger_frames = min(
            self._pre_frames, max(1, TRIGGER_WINDOW_MS // FRAME_MS)
        )

        self._ring: collections.deque[tuple[bytes, bool]] = collections.deque(
            maxlen=self._pre_frames
        )
        self._voiced: list[bytes] = []
        self._triggered = False
        self._silence_run = 0

    @property
    def detector_name(self) -> str:
        return self._detector.name

    @property
    def in_speech(self) -> bool:
        return self._triggered

    def reset(self) -> None:
        """Drop any partial utterance, e.g. after the robot spoke."""
        self._ring.clear()
        self._voiced.clear()
        self._triggered = False
        self._silence_run = 0

    def push(self, frame: bytes) -> Iterator[bytes]:
        """Feed one 20 ms frame; yields a complete utterance when one ends."""
        if len(frame) != FRAME_BYTES:
            return
        speech = self._detector.is_speech(frame)

        if not self._triggered:
            self._ring.append((frame, speech))
            # The decision is made on the most recent frames only, while the
            # ring keeps a longer tail to prepend once it fires.
            #
            # These were one and the same buffer, and it lost the start of
            # words. Requiring 60% of a 320 ms window to be speech means about
            # 200 ms of it is already the utterance, leaving barely 120 ms of
            # lead-in -- and an unvoiced fricative lasts about that long and is
            # exactly what webrtcvad is worst at hearing. The result on the
            # robot was "Spot, stand up" transcribed as "or stand up": the word
            # was not mistranscribed, it was never recorded.
            #
            # Enlarging the single buffer could not fix it, because the trigger
            # threshold scaled with it: a longer window needed proportionally
            # more speech before firing, and the lead-in stayed just as thin.
            window = list(self._ring)[-self._trigger_frames :]
            voiced_count = sum(1 for _, flag in window if flag)
            if (
                len(window) == self._trigger_frames
                and voiced_count > TRIGGER_RATIO * self._trigger_frames
            ):
                self._triggered = True
                self._silence_run = 0
                self._voiced = [buffered for buffered, _ in self._ring]
                self._ring.clear()
            return

        self._voiced.append(frame)
        self._silence_run = 0 if speech else self._silence_run + 1

        too_long = len(self._voiced) >= self._max_frames
        if self._silence_run >= self._post_frames or too_long:
            audio = b"".join(self._voiced)
            long_enough = len(self._voiced) >= self._min_frames
            self.reset()
            if long_enough:
                if too_long:
                    LOGGER.info("Utterance hit the length cap; cutting it here")
                yield audio


def pcm_to_float32(pcm: bytes):
    """Convert 16-bit PCM bytes to the float32 array faster-whisper expects."""
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16)
    return samples.astype("float32") / 32768.0
