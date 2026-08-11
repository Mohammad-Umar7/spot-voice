"""Speech synthesis. Two engines, one output format.

Both engines produce a **16 kHz mono 16-bit PCM WAV**, because that is what Spot
CAM's audio service can load and what ``simpleaudio`` can play. edge-tts returns
MP3, so it needs a decoder; the chain tries, in order, ``miniaudio`` (pip wheel,
no external binary), the ffmpeg bundled by ``imageio-ffmpeg``, and finally
``ffmpeg`` on ``PATH``. If none is available the edge engine reports that and
:class:`~spot_voice.tts.speaker.Speaker` falls back to the offline engine.

Privacy: ``TTS_ENGINE=edge`` sends the text of every reply to Microsoft's Edge
TTS servers. ``TTS_ENGINE=offline`` uses the local Windows voice and sends
nothing anywhere.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import threading
import wave
from pathlib import Path

LOGGER = logging.getLogger(__name__)

#: Output format, chosen to match Spot CAM's audio service.
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes, i.e. 16-bit PCM

_PYTTSX3_LOCK = threading.Lock()


class TtsUnavailable(RuntimeError):
    """Raised when an engine cannot produce audio at all."""


# ----------------------------------------------------------------------
# edge-tts
# ----------------------------------------------------------------------


def synthesize_edge(text: str, voice: str, out_wav: Path) -> Path:
    """Synthesise ``text`` with edge-tts and write a WAV to ``out_wav``.

    Raises:
        TtsUnavailable: If edge-tts or an MP3 decoder is missing, or the network
            call fails.
    """
    try:
        import edge_tts
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise TtsUnavailable("edge-tts is not installed") from exc

    mp3_path = out_wav.with_suffix(".mp3")

    async def _run() -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(mp3_path))

    try:
        asyncio.run(_run())
    except Exception as exc:
        raise TtsUnavailable(f"edge-tts failed: {exc}") from exc

    try:
        mp3_to_wav(mp3_path, out_wav)
    finally:
        mp3_path.unlink(missing_ok=True)
    return out_wav


def mp3_to_wav(mp3_path: Path, out_wav: Path) -> Path:
    """Decode an MP3 to 16 kHz mono 16-bit PCM WAV.

    Tries every available decoder in turn so a missing optional dependency
    degrades rather than breaks.

    Raises:
        TtsUnavailable: When no decoder is available.
    """
    for decoder in (_decode_with_miniaudio, _decode_with_bundled_ffmpeg, _decode_with_path_ffmpeg):
        try:
            if decoder(mp3_path, out_wav):
                return out_wav
        except Exception as exc:
            LOGGER.debug("%s failed: %s", decoder.__name__, exc)
    raise TtsUnavailable(
        "No MP3 decoder available. Install 'miniaudio' or 'imageio-ffmpeg', "
        "or set TTS_ENGINE=offline."
    )


def _decode_with_miniaudio(mp3_path: Path, out_wav: Path) -> bool:
    """Decode using the ``miniaudio`` wheel (no external binary needed)."""
    try:
        import miniaudio
    except ImportError:
        return False

    decoded = miniaudio.decode_file(
        str(mp3_path),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=CHANNELS,
        sample_rate=SAMPLE_RATE,
    )
    with wave.open(str(out_wav), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(decoded.samples.tobytes())
    return True


def _run_ffmpeg(exe: str, mp3_path: Path, out_wav: Path) -> bool:
    """Run ffmpeg to transcode, returning True on success."""
    command = [
        exe,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(mp3_path),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        str(CHANNELS),
        "-sample_fmt",
        "s16",
        str(out_wav),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        LOGGER.debug("ffmpeg failed: %s", completed.stderr.decode(errors="replace"))
        return False
    return out_wav.exists()


def _decode_with_bundled_ffmpeg(mp3_path: Path, out_wav: Path) -> bool:
    """Decode using the ffmpeg binary shipped by ``imageio-ffmpeg``."""
    try:
        import imageio_ffmpeg
    except ImportError:
        return False
    return _run_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe(), mp3_path, out_wav)


def _decode_with_path_ffmpeg(mp3_path: Path, out_wav: Path) -> bool:
    """Decode using an ``ffmpeg`` already on PATH."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    return _run_ffmpeg(exe, mp3_path, out_wav)


# ----------------------------------------------------------------------
# pyttsx3 (offline)
# ----------------------------------------------------------------------


def synthesize_offline(text: str, out_wav: Path, rate: int | None = None) -> Path:
    """Synthesise ``text`` with the local system voice (SAPI5 on Windows).

    A fresh engine is created per call: pyttsx3 is unreliable when
    ``runAndWait`` is used repeatedly on one instance.

    Raises:
        TtsUnavailable: If pyttsx3 is missing or produced no audio.
    """
    try:
        import pyttsx3
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise TtsUnavailable("pyttsx3 is not installed") from exc

    with _PYTTSX3_LOCK:
        engine = pyttsx3.init()
        try:
            if rate is not None:
                engine.setProperty("rate", rate)
            engine.save_to_file(text, str(out_wav))
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:  # pragma: no cover
                pass

    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise TtsUnavailable("pyttsx3 produced no audio")
    return _normalise_wav(out_wav)


def _normalise_wav(path: Path) -> Path:
    """Resample/downmix a WAV in place to the format Spot CAM expects.

    Uses only the standard library. If the file is already in the target format
    it is left untouched.

    ``audioop`` was removed in Python 3.13; on a runtime without it the file is
    returned unchanged rather than failing, since most system voices already
    emit mono 16-bit PCM and Spot CAM generally accepts it.
    """
    try:
        import audioop
    except ImportError:  # pragma: no cover - Python 3.13+
        LOGGER.debug("audioop unavailable; leaving WAV format untouched")
        return path

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if (channels, width, rate) == (CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE):
        return path

    if width != SAMPLE_WIDTH:
        frames = audioop.lin2lin(frames, width, SAMPLE_WIDTH)
        width = SAMPLE_WIDTH
    if channels > CHANNELS:
        frames = audioop.tomono(frames, width, 0.5, 0.5)
        channels = CHANNELS
    if rate != SAMPLE_RATE:
        frames, _ = audioop.ratecv(frames, width, channels, rate, SAMPLE_RATE, None)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(frames)
    return path


def wav_duration_seconds(path: Path) -> float:
    """Length of a WAV file in seconds."""
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate() or SAMPLE_RATE)
