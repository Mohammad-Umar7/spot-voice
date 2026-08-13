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
    #: Whisper's confidence signals, kept for logging and tuning.
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0

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
        collected, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=1,  # greedy: short commands, latency matters more than nuance
            vad_filter=False,  # segmentation already happened upstream
            condition_on_previous_text=False,  # stop one bad decode poisoning the next
        )
        segments = list(collected)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        duration_ms = (time.perf_counter() - started) * 1000.0

        # Worst-case confidence across the segments: one bad patch is enough to
        # make the whole utterance untrustworthy as a robot command.
        no_speech = max(
            (getattr(s, "no_speech_prob", 0.0) or 0.0 for s in segments), default=0.0
        )
        logprob = min(
            (getattr(s, "avg_logprob", 0.0) or 0.0 for s in segments), default=0.0
        )

        if _looks_like_hallucination(text):
            LOGGER.debug("Dropping likely hallucination: %r", text)
            text = ""

        return Transcript(
            text=text,
            duration_ms=duration_ms,
            audio_ms=audio_ms,
            language=getattr(info, "language", self._language),
            no_speech_prob=no_speech,
            avg_logprob=logprob,
        )


#: Below this, a "transcript" is a cough, a chair or a footstep the VAD let
#: through. Real commands are longer.
MIN_TRANSCRIPT_CHARS = 4

#: Whisper's own estimate that a segment contains no speech at all. Above this,
#: whatever words came out were invented over room noise.
MAX_NO_SPEECH_PROB = 0.6

#: Average token log-probability below which the decoder was guessing. Real
#: speech on a decent mic sits around -0.3; noise-driven output goes well below.
MIN_AVG_LOGPROB = -1.0


def _looks_like_noise(text: str, no_speech_prob: float, avg_logprob: float) -> bool:
    """True when the decoder was not confident this was speech.

    An earlier attempt filtered on characters-per-second, on the theory that
    invented text is sparse. It is not: "holo mate" and "This is a monogram" --
    both real misfires on the robot -- have exactly the density of a genuine
    command. Text statistics cannot separate a misheard sentence from a real
    one, because the mistake is upstream of the text.

    Whisper's own confidence can. ``no_speech_prob`` is its estimate that the
    audio contained no speech, and ``avg_logprob`` is how sure it was of the
    tokens it chose. Noise-driven output scores badly on both while a real
    command does not.
    """
    if len(text.strip()) < MIN_TRANSCRIPT_CHARS:
        return True
    return no_speech_prob > MAX_NO_SPEECH_PROB or avg_logprob < MIN_AVG_LOGPROB


def _looks_like_hallucination(text: str) -> bool:
    """True for the stock phrases Whisper emits when handed near-silence."""
    stripped = text.strip()
    if not stripped:
        return True
    return any(pattern.match(stripped) for pattern in _HALLUCINATION_PATTERNS)
