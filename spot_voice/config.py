"""Configuration loaded from the environment (``.env`` only -- nothing hardcoded).

Every secret and site-specific value lives in ``.env``. See ``.env.example`` for the
full list. This module reads it once at startup and hands back a frozen
:class:`Config`; nothing else in the project touches ``os.environ`` for settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

#: Environment variables the Boston Dynamics SDK reads directly via
#: ``bosdyn.client.util.authenticate``. We never read the password ourselves.
BOSDYN_USERNAME_ENV = "BOSDYN_CLIENT_USERNAME"
BOSDYN_PASSWORD_ENV = "BOSDYN_CLIENT_PASSWORD"


class ConfigError(RuntimeError):
    """Raised when the environment is missing something the program cannot run without."""


def _as_bool(raw: str | None, default: bool) -> bool:
    """Parse a permissive boolean from the environment."""
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(raw: str | None, default: int | None) -> int | None:
    """Parse an optional integer, returning ``default`` when unset or malformed."""
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    """Immutable snapshot of the runtime configuration."""

    # --- Robot -------------------------------------------------------------
    spot_ip: str
    dock_id: int | None
    graph_path: Path | None
    mock_robot: bool

    # --- Anthropic ---------------------------------------------------------
    anthropic_api_key: str
    anthropic_model: str

    # --- Audio in ----------------------------------------------------------
    mic_device_name: str
    whisper_model: str
    stt_language: str

    # --- Audio out ---------------------------------------------------------
    tts_engine: str  # "edge" | "offline"
    audio_out: str  # "robot" | "laptop"
    tts_voice: str
    mute_while_speaking: bool

    # --- Misc --------------------------------------------------------------
    log_level: str
    work_dir: Path = field(default_factory=lambda: Path.home() / ".spot_voice")

    # ------------------------------------------------------------------
    @property
    def brain_enabled(self) -> bool:
        """True when an Anthropic key is present, so the LLM lane can run.

        The reflex lane never depends on this -- safety words work with the API
        completely unreachable.
        """
        return bool(self.anthropic_api_key)

    def describe(self) -> list[tuple[str, str]]:
        """Return a redacted (key, value) list suitable for printing at startup."""
        key = self.anthropic_api_key
        redacted = f"{key[:8]}...{key[-4:]}" if len(key) > 14 else ("set" if key else "MISSING")
        return [
            ("mode", "MOCK (no robot)" if self.mock_robot else f"REAL ROBOT @ {self.spot_ip}"),
            ("anthropic model", self.anthropic_model),
            ("anthropic key", redacted),
            ("mic", self.mic_device_name or "<system default>"),
            ("whisper model", self.whisper_model),
            ("tts", f"{self.tts_engine} -> {self.audio_out}"),
            ("graph", str(self.graph_path) if self.graph_path else "<none>"),
            ("dock id", str(self.dock_id) if self.dock_id is not None else "<none>"),
        ]


def load_config(env_file: str | os.PathLike[str] | None = None) -> Config:
    """Read ``.env`` (if present) plus the process environment into a :class:`Config`.

    Args:
        env_file: Optional explicit path to a ``.env`` file. When ``None`` the
            usual discovery (current directory upwards) is used.

    Raises:
        ConfigError: If a value required for the selected mode is missing.
    """
    if env_file is not None:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)

    mock = _as_bool(os.getenv("MOCK_ROBOT"), default=True)
    spot_ip = (os.getenv("SPOT_IP") or "").strip()
    graph_raw = (os.getenv("GRAPH_PATH") or "").strip()

    cfg = Config(
        spot_ip=spot_ip,
        dock_id=_as_int(os.getenv("DOCK_ID"), None),
        graph_path=Path(graph_raw).expanduser() if graph_raw else None,
        mock_robot=mock,
        anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY") or "").strip(),
        anthropic_model=(os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-6").strip(),
        mic_device_name=(os.getenv("MIC_DEVICE_NAME") or "").strip(),
        whisper_model=(os.getenv("WHISPER_MODEL") or "base").strip(),
        stt_language=(os.getenv("STT_LANGUAGE") or "en").strip(),
        tts_engine=(os.getenv("TTS_ENGINE") or "edge").strip().lower(),
        audio_out=(os.getenv("AUDIO_OUT") or "laptop").strip().lower(),
        tts_voice=(os.getenv("TTS_VOICE") or "en-US-GuyNeural").strip(),
        mute_while_speaking=_as_bool(os.getenv("MUTE_WHILE_SPEAKING"), default=True),
        log_level=(os.getenv("LOG_LEVEL") or "INFO").strip().upper(),
    )

    _validate(cfg)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _validate(cfg: Config) -> None:
    """Fail fast on configuration that cannot possibly work."""
    if cfg.tts_engine not in {"edge", "offline"}:
        raise ConfigError(f"TTS_ENGINE must be 'edge' or 'offline', got {cfg.tts_engine!r}")
    if cfg.audio_out not in {"robot", "laptop"}:
        raise ConfigError(f"AUDIO_OUT must be 'robot' or 'laptop', got {cfg.audio_out!r}")

    if cfg.mock_robot:
        if cfg.audio_out == "robot":
            raise ConfigError(
                "AUDIO_OUT=robot is meaningless with MOCK_ROBOT=true -- set AUDIO_OUT=laptop."
            )
        return

    # Real-robot mode: everything the SDK needs must be present.
    missing: list[str] = []
    if not cfg.spot_ip:
        missing.append("SPOT_IP")
    if not os.getenv(BOSDYN_USERNAME_ENV):
        missing.append(BOSDYN_USERNAME_ENV)
    if not os.getenv(BOSDYN_PASSWORD_ENV):
        missing.append(BOSDYN_PASSWORD_ENV)
    if missing:
        raise ConfigError(
            "MOCK_ROBOT=false but these are missing from .env: " + ", ".join(missing)
        )

    if cfg.graph_path is not None and not cfg.graph_path.exists():
        raise ConfigError(
            f"GRAPH_PATH does not exist: {cfg.graph_path}. It should point at the "
            "'downloaded_graph' folder containing 'graph', 'waypoint_snapshots/' and "
            "'edge_snapshots/'."
        )
