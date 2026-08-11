"""Local speech-to-text with faster-whisper.

Everything stays on the laptop: no audio ever leaves the machine, which matters
because the internet link on a demo is a phone tether and because a facility
walkthrough is not something to stream to a cloud transcriber.

``base`` (int8) is the default -- roughly real time on a laptop CPU and accurate
enough for short commands. ``small`` is noticeably better on accented speech and
still workable; set ``WHISPER_MODEL=small`` if transcripts are shaky.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

#: Whisper likes to fill silence with these. Dropping them prevents phantom turns.
_HALLUCINATION_PATTERNS = (
    re.compile(r"^\W*thanks? for watching\W*$", re.I),
    re.compile(r"^\W*thank you\W*$", re.I),
    re.compile(r"^\W*you\W*$", re.I),
    re.compile(r"^\W*bye\W*$", re.I),
    re.compile(r"^\W*\[?(music|applause|silence|blank_audio)\]?\W*$", re.I),
    re.compile(r"^\W*subtitles? by.*$", re.I),
)


@dataclass
class Transcript:
    """One decoded utterance."""

    text: str
    duration_ms: float
    audio_ms: float
    language: str = "en"

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class Transcriber:
    """Wraps a faster-whisper model.

    Args:
        model_size: ``tiny``, ``base``, ``small`` ... Larger is slower.
        language: Forced decode language; ``en`` avoids costly detection.
        compute_type: ``int8`` is the right choice on a CPU-only laptop.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "en",
        compute_type: str = "int8",
        device: str = "cpu",
    ) -> None:
        from faster_whisper import WhisperModel

        LOGGER.info("Loading faster-whisper %r (%s, %s)...", model_size, device, compute_type)
        started = time.perf_counter()
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language
        LOGGER.info("Model ready in %.1fs", time.perf_counter() - started)

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = 16000) -> Transcript:
        """Decode 16-bit PCM audio into text."""
        from .vad import pcm_to_float32

        audio = pcm_to_float32(pcm)
        audio_ms = len(audio) / sample_rate * 1000.0

        started = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=1,  # greedy: short commands, latency matters more than nuance
            vad_filter=False,  # segmentation already happened upstream
            condition_on_previous_text=False,  # stop one bad decode poisoning the next
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        duration_ms = (time.perf_counter() - started) * 1000.0

        if _looks_like_hallucination(text):
            LOGGER.debug("Dropping likely hallucination: %r", text)
            text = ""

        return Transcript(
            text=text,
            duration_ms=duration_ms,
            audio_ms=audio_ms,
            language=getattr(info, "language", self._language),
        )


def _looks_like_hallucination(text: str) -> bool:
    """True for the stock phrases Whisper emits when handed near-silence."""
    stripped = text.strip()
    if not stripped:
        return True
    return any(pattern.match(stripped) for pattern in _HALLUCINATION_PATTERNS)
