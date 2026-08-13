"""The audio thread must never wait for transcription.

A real session logged "Dropped 2150 audio frames while the queue was full" --
43 seconds of speech that never reached the transcriber, because segmentation
and transcription shared one thread and Whisper took 5-14 seconds per
utterance. From outside that looked like the robot ignoring people.
"""

from __future__ import annotations

import threading
import time

from spot_voice.audio.listener import (
    MAX_UTTERANCE_AGE_SEC,
    UTTERANCE_QUEUE_MAX,
    Listener,
)
from spot_voice.audio.stt import Transcript


class SlowTranscriber:
    """Stands in for Whisper taking seconds per utterance."""

    def __init__(self, delay: float = 0.4) -> None:
        self.delay = delay
        self.seen: list[bytes] = []
        self.busy = threading.Event()

    def transcribe_pcm(self, pcm: bytes, sample_rate: int = 16000) -> Transcript:
        self.busy.set()
        time.sleep(self.delay)
        self.seen.append(pcm)
        return Transcript(text="stand up", duration_ms=1.0, audio_ms=1000.0)


def _listener(transcriber, on_transcript=lambda _t: None) -> Listener:
    return Listener(transcriber=transcriber, on_transcript=on_transcript)


def test_segmenting_does_not_wait_for_transcription():
    """The whole point: handing over an utterance must return immediately."""
    listener = _listener(SlowTranscriber(delay=5.0))
    listener._stop.clear()
    worker = threading.Thread(target=listener._transcribe_loop, daemon=True)
    worker.start()
    try:
        started = time.perf_counter()
        for _ in range(3):
            listener._enqueue_utterance(b"\x00" * 320)
        elapsed = time.perf_counter() - started
    finally:
        listener._stop.set()
        worker.join(timeout=1.0)

    assert elapsed < 0.5, f"enqueueing blocked for {elapsed:.2f}s"


def test_a_backlog_drops_the_oldest_not_the_newest():
    """A stale robot command is worse than a missed one."""
    listener = _listener(SlowTranscriber())

    for index in range(UTTERANCE_QUEUE_MAX + 3):
        listener._enqueue_utterance(bytes([index]) * 320)

    assert listener._utterances.qsize() == UTTERANCE_QUEUE_MAX
    assert listener._stale == 3
    # What survived is the most recent, not the first.
    _queued_at, newest = list(listener._utterances.queue)[-1]
    assert newest[0] == UTTERANCE_QUEUE_MAX + 2


def test_an_utterance_that_waited_too_long_is_never_transcribed():
    transcriber = SlowTranscriber(delay=0.0)
    listener = _listener(transcriber)
    listener._stop.clear()

    stale_time = time.monotonic() - (MAX_UTTERANCE_AGE_SEC + 1)
    listener._utterances.put((stale_time, b"\x00" * 320))
    listener._utterances.put((time.monotonic(), b"\x01" * 320))

    worker = threading.Thread(target=listener._transcribe_loop, daemon=True)
    worker.start()
    time.sleep(0.4)
    listener._stop.set()
    worker.join(timeout=1.0)

    assert len(transcriber.seen) == 1, "the stale utterance should not be transcribed"
    assert transcriber.seen[0][0] == 1
    assert listener._stale == 1


def test_transcripts_still_reach_the_handler():
    delivered: list[str] = []
    transcriber = SlowTranscriber(delay=0.0)
    listener = _listener(transcriber, on_transcript=lambda t: delivered.append(t.text))
    listener._stop.clear()

    listener._enqueue_utterance(b"\x00" * 320)
    worker = threading.Thread(target=listener._transcribe_loop, daemon=True)
    worker.start()
    time.sleep(0.4)
    listener._stop.set()
    worker.join(timeout=1.0)

    assert delivered == ["stand up"]


# ----------------------------------------------------------------------
# Device selection
#
# On this project's laptop `small` takes 4.58s on the CPU and 0.19s on the GPU.
# At CPU speed the transcriber cannot keep up with someone talking, which is
# what made the robot feel deaf -- so using an available GPU is not a tuning
# preference, it is the difference between working and not.


def test_auto_prefers_the_gpu_when_one_is_visible(monkeypatch):
    import spot_voice.audio.stt as stt

    monkeypatch.setattr(stt, "cuda_is_available", lambda: True)
    assert stt.resolve_device("auto") == ("cuda", "float16")


def test_auto_falls_back_to_cpu_with_no_gpu(monkeypatch):
    import spot_voice.audio.stt as stt

    monkeypatch.setattr(stt, "cuda_is_available", lambda: False)
    assert stt.resolve_device("auto") == ("cpu", "int8")


def test_an_explicit_device_is_honoured(monkeypatch):
    import spot_voice.audio.stt as stt

    monkeypatch.setattr(stt, "cuda_is_available", lambda: True)
    assert stt.resolve_device("cpu") == ("cpu", "int8")


def test_an_explicit_compute_type_is_never_overridden():
    from spot_voice.audio.stt import resolve_device

    assert resolve_device("cuda", "int8_float16") == ("cuda", "int8_float16")


def test_a_missing_ctranslate2_is_not_fatal(monkeypatch):
    """Asking the GPU question must never stop the program starting."""
    import builtins

    import spot_voice.audio.stt as stt

    real_import = builtins.__import__

    def explode(name, *args, **kwargs):
        if name == "ctranslate2":
            raise ImportError("no ctranslate2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", explode)
    assert stt.cuda_is_available() is False


# ----------------------------------------------------------------------
# Keeping the start of the first word
#
# On the robot, "Spot, stand up" was transcribed as "or stand up" and "Spot,
# say hi" as "or say hi" -- while the same sentences, said again, came through
# perfectly. The word was not mistranscribed; it was never recorded. The
# segmenter's decision window and its pre-roll buffer were the same deque, so
# by the time enough speech had accumulated to trigger, the quiet fricative at
# the start had already scrolled out of it.


class ScriptedDetector:
    """Flags speech per frame from a fixed pattern."""

    name = "scripted"

    def __init__(self, pattern) -> None:
        self.pattern = list(pattern)
        self.index = -1

    def is_speech(self, _frame: bytes) -> bool:
        self.index += 1
        if self.index < len(self.pattern):
            return self.pattern[self.index]
        return False


def _frames(segmenter, count, marker=0):
    from spot_voice.audio.vad import FRAME_BYTES

    out = []
    for _ in range(count):
        out.extend(segmenter.push(bytes([marker]) * FRAME_BYTES))
    return out


def test_the_pre_roll_is_longer_than_the_trigger_window():
    """The whole fix: deciding quickly must not mean keeping little."""
    from spot_voice.audio.vad import PRE_SPEECH_MS, TRIGGER_WINDOW_MS

    assert PRE_SPEECH_MS > TRIGGER_WINDOW_MS


def test_audio_from_before_the_trigger_is_kept():
    from spot_voice.audio.vad import FRAME_MS, PRE_SPEECH_MS, UtteranceSegmenter

    # Silence, then continuous speech. The quiet run stands in for the "s" of
    # "Spot" -- audio the detector does not flag but the transcriber needs.
    lead_in = 10
    detector = ScriptedDetector(
        [False] * lead_in + [True] * 60 + [False] * 200
    )
    segmenter = UtteranceSegmenter(detector=detector)

    _frames(segmenter, lead_in, marker=1)   # the lead-in
    _frames(segmenter, 60, marker=2)        # the rest of the word
    utterances = _frames(segmenter, 60, marker=0)  # trailing silence ends it

    assert utterances, "no utterance was produced"
    audio = utterances[0]

    from spot_voice.audio.vad import FRAME_BYTES

    kept = 0
    while audio[kept * FRAME_BYTES : (kept + 1) * FRAME_BYTES] == b"\x01" * FRAME_BYTES:
        kept += 1

    # All of it, not merely some. The old design retained 120 ms however much
    # there was, because the pre-roll was also the decision window; an unvoiced
    # fricative runs 100-150 ms, which is why the word survived on some takes
    # and vanished on others.
    assert kept == lead_in, (
        f"kept only {kept * FRAME_MS} ms of a {lead_in * FRAME_MS} ms lead-in"
    )
    assert PRE_SPEECH_MS // FRAME_MS >= lead_in
