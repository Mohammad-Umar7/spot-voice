"""Speaking: synthesise a reply and play it out of the right speaker.

``AUDIO_OUT=robot`` plays through Spot CAM's own speaker (the robot talks).
``AUDIO_OUT=laptop`` plays locally, which is what mock mode uses.

While audio is playing the microphone is gated (see ``MUTE_WHILE_SPEAKING``) so
Spot does not transcribe its own voice and talk to itself. That is a real
tradeoff: for the two or three seconds a reply lasts, spoken reflex words are not
heard. The physical e-stop on the tablet is always live and is the answer to
that gap.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from rich.console import Console

from .engines import (
    TtsUnavailable,
    synthesize_edge,
    synthesize_offline,
    wav_duration_seconds,
)

LOGGER = logging.getLogger(__name__)

#: Extra time the mic stays gated after playback, to swallow room echo.
MUTE_TAIL_SEC = 0.35


class Speaker:
    """Turns text into sound.

    Args:
        engine: ``"edge"`` or ``"offline"``.
        audio_out: ``"robot"`` or ``"laptop"``.
        voice: edge-tts voice name.
        work_dir: Where WAV files are written.
        robot: Robot used when ``audio_out == "robot"``.
        on_speech_start: Called just before playback (used to gate the mic).
        on_speech_end: Called just after playback finishes.
        console: Rich console for the transcript of what Spot says.
    """

    def __init__(
        self,
        engine: str = "edge",
        audio_out: str = "laptop",
        voice: str = "en-US-GuyNeural",
        work_dir: Path | None = None,
        robot=None,
        on_speech_start: Callable[[], None] | None = None,
        on_speech_end: Callable[[], None] | None = None,
        console: Console | None = None,
    ) -> None:
        self._engine = engine
        self._audio_out = audio_out
        self._voice = voice
        self._work_dir = work_dir or (Path.home() / ".spot_voice")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._robot = robot
        self._on_start = on_speech_start
        self._on_end = on_speech_end
        self._console = console or Console()

        self._lock = threading.Lock()
        self._counter = 0
        self._edge_broken = False

    # ------------------------------------------------------------------

    def speak(self, text: str, blocking: bool = True) -> bool:
        """Say ``text`` out loud. Returns ``True`` when audio actually played.

        Never raises: if synthesis or playback fails the text is still printed,
        so the operator can read what Spot meant to say.
        """
        text = (text or "").strip()
        if not text:
            return False

        self._console.print(f"[bold green]Spot:[/bold green] {text}")

        with self._lock:
            try:
                wav_path = self._synthesize(text)
            except TtsUnavailable as exc:
                LOGGER.warning("Speech synthesis unavailable: %s", exc)
                self._console.print(f"[yellow]  (no audio: {exc})[/yellow]")
                return False

            if self._on_start is not None:
                self._safe(self._on_start)
            try:
                self._play(wav_path, blocking=blocking)
                return True
            except Exception as exc:
                LOGGER.warning("Playback failed", exc_info=True)
                self._console.print(f"[yellow]  (playback failed: {exc})[/yellow]")
                return False
            finally:
                if self._on_end is not None:
                    # Let the tail of the audio clear the room before listening.
                    threading.Timer(MUTE_TAIL_SEC, lambda: self._safe(self._on_end)).start()

    def speak_async(self, text: str) -> threading.Thread:
        """Speak on a background thread and return it, for non-blocking replies."""
        thread = threading.Thread(target=self.speak, args=(text,), daemon=True)
        thread.start()
        return thread

    # ------------------------------------------------------------------

    def _synthesize(self, text: str) -> Path:
        """Produce a WAV for ``text``, falling back to the offline engine."""
        self._counter += 1
        # Two alternating filenames: Spot CAM keys loaded sounds by name, and
        # rotating avoids re-uploading over a sound that is still playing.
        out = self._work_dir / f"speech_{self._counter % 2}.wav"
        out.unlink(missing_ok=True)

        if self._engine == "edge" and not self._edge_broken:
            try:
                return synthesize_edge(text, self._voice, out)
            except TtsUnavailable as exc:
                LOGGER.warning("edge-tts unavailable (%s) -- falling back to offline", exc)
                self._console.print(
                    "[yellow]edge-tts unavailable; using the offline voice from now on.[/yellow]"
                )
                self._edge_broken = True

        return synthesize_offline(text, out)

    def _play(self, wav_path: Path, blocking: bool) -> None:
        """Route the WAV to the configured output."""
        if self._audio_out == "robot":
            if self._robot is None:
                raise RuntimeError("AUDIO_OUT=robot but no robot is connected")
            self._robot.play_wav(str(wav_path), blocking=blocking)
            return
        _play_locally(wav_path, blocking=blocking)

    @staticmethod
    def _safe(callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:  # pragma: no cover - a mic gate must never break speech
            LOGGER.debug("speech callback failed", exc_info=True)


def _play_locally(wav_path: Path, blocking: bool = True) -> None:
    """Play a WAV on the laptop, preferring simpleaudio and falling back to winsound."""
    try:
        import simpleaudio
    except ImportError:
        simpleaudio = None  # type: ignore[assignment]

    if simpleaudio is not None:
        wave_obj = simpleaudio.WaveObject.from_wave_file(str(wav_path))
        play_obj = wave_obj.play()
        if blocking:
            play_obj.wait_done()
        return

    try:
        import winsound
    except ImportError as exc:  # pragma: no cover - non-Windows without simpleaudio
        raise RuntimeError(
            "No audio playback backend. Install 'simpleaudio'."
        ) from exc

    flags = winsound.SND_FILENAME
    if not blocking:
        flags |= winsound.SND_ASYNC
    winsound.PlaySound(str(wav_path), flags)
    if blocking:
        # winsound's synchronous mode already blocks; this is belt and braces
        # for the async path callers may rely on.
        time.sleep(0)


__all__ = ["Speaker", "wav_duration_seconds"]
