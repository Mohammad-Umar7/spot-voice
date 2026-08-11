"""The always-listening loop: microphone -> VAD -> local Whisper -> callback.

Two threads:

* PortAudio's own callback thread pushes raw 20 ms frames onto a queue. It does
  no work beyond that -- blocking there would drop audio.
* A worker thread segments the frames into utterances and transcribes them.

Transcription is deliberately *not* done on the audio thread, and the callback
the worker invokes runs on the worker thread, so the caller must not assume it
is on the main thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable

from rich.console import Console

from .devices import select_input_device
from .stt import Transcriber, Transcript
from .vad import FRAME_BYTES, FRAME_SAMPLES, SAMPLE_RATE, UtteranceSegmenter

LOGGER = logging.getLogger(__name__)

#: Frames buffered between the audio callback and the worker (~10 s).
_QUEUE_MAX = 500


class Listener:
    """Captures speech and hands finished transcripts to a callback.

    Args:
        transcriber: Local speech-to-text engine.
        on_transcript: Called with each non-empty :class:`Transcript`.
        mic_name: ``MIC_DEVICE_NAME`` substring; empty means system default.
        console: Rich console for the live transcript log.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        on_transcript: Callable[[Transcript], None],
        mic_name: str = "",
        console: Console | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._on_transcript = on_transcript
        self._mic_name = mic_name
        self._console = console or Console()

        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=_QUEUE_MAX)
        self._segmenter = UtteranceSegmenter()
        self._stop = threading.Event()
        self._muted = threading.Event()
        self._worker: threading.Thread | None = None
        self._stream = None
        self._dropped = 0

    # ------------------------------------------------------------------

    @property
    def muted(self) -> bool:
        return self._muted.is_set()

    def mute(self) -> None:
        """Stop accepting audio (called while Spot is speaking)."""
        self._muted.set()
        self._segmenter.reset()

    def unmute(self) -> None:
        """Resume listening, discarding whatever arrived while muted."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._segmenter.reset()
        self._muted.clear()

    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the microphone and start transcribing."""
        import sounddevice

        device_index = select_input_device(self._mic_name)
        device_label = (
            sounddevice.query_devices(device_index)["name"]
            if device_index is not None
            else "system default"
        )
        self._console.print(
            f"[bold]Microphone:[/bold] {device_label} "
            f"[dim](VAD: {self._segmenter.detector_name})[/dim]"
        )

        self._stop.clear()
        self._stream = sounddevice.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            device=device_index,
            dtype="int16",
            channels=1,
            callback=self._audio_callback,
        )
        self._stream.start()

        self._worker = threading.Thread(target=self._run, name="stt-worker", daemon=True)
        self._worker.start()
        LOGGER.info("Listening on %s", device_label)

    def stop(self) -> None:
        """Close the microphone and stop the worker. Safe to call twice."""
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # pragma: no cover - teardown best effort
                LOGGER.debug("closing audio stream raised", exc_info=True)
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            self._worker = None
        if self._dropped:
            LOGGER.info("Dropped %d audio frames while the queue was full", self._dropped)

    # ------------------------------------------------------------------

    def _audio_callback(self, indata, _frames, _time_info, status) -> None:
        """PortAudio callback. Must return fast -- it only enqueues."""
        if status:
            LOGGER.debug("audio status: %s", status)
        if self._muted.is_set():
            return
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            self._dropped += 1

    def _run(self) -> None:
        """Segment frames into utterances and transcribe them."""
        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if len(frame) != FRAME_BYTES:
                continue

            for utterance in self._segmenter.push(frame):
                if self._stop.is_set() or self._muted.is_set():
                    continue
                self._handle_utterance(utterance)

    def _handle_utterance(self, pcm: bytes) -> None:
        """Transcribe one utterance and forward it."""
        started = time.perf_counter()
        try:
            transcript = self._transcriber.transcribe_pcm(pcm)
        except Exception:
            LOGGER.exception("Transcription failed")
            return

        if transcript.is_empty:
            LOGGER.debug(
                "Empty transcript for %.0f ms of audio", transcript.audio_ms
            )
            return

        total_ms = (time.perf_counter() - started) * 1000.0
        self._console.print(
            f"[bold cyan]You:[/bold cyan] {transcript.text} "
            f"[dim]({transcript.audio_ms:.0f} ms audio, {total_ms:.0f} ms stt)[/dim]"
        )
        try:
            self._on_transcript(transcript)
        except Exception:
            LOGGER.exception("Transcript handler raised")
