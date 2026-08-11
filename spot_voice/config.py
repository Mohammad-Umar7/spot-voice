"""Configuration loaded from the environment (``.env`` only -- nothing hardcoded).

Every secret and site-specific value lives in ``.env``. See ``.env.example`` for the
full list. This module reads it once at startup and hands back a frozen
:class:`Config`; nothing else in the project touches ``os.environ`` for settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
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

    # --- Models ------------------------------------------------------------
    # Which provider does tool calling, and which one reads camera frames.
    # Anthropic is the production target; the others exist so the whole system
    # can be wired up and rehearsed before the production key is in play.
    llm_provider: str  # "anthropic" | "groq"
    vision_provider: str  # "anthropic" | "gemini" | "none"

    anthropic_api_key: str
    anthropic_model: str
    groq_api_key: str
    groq_model: str
    gemini_api_key: str
    gemini_model: str

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
    def llm_api_key(self) -> str:
        """The key for whichever provider is doing tool calling."""
        return {
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
        }.get(self.llm_provider, "")

    @property
    def llm_model(self) -> str:
        """The model id for whichever provider is doing tool calling."""
        return {
            "anthropic": self.anthropic_model,
            "groq": self.groq_model,
        }.get(self.llm_provider, "")

    @property
    def brain_enabled(self) -> bool:
        """True when the selected provider has a key, so the LLM lane can run.

        The reflex lane never depends on this -- safety words work with every
        API completely unreachable.
        """
        return bool(self.llm_api_key)

    def describe(self) -> list[tuple[str, str]]:
        """Return a redacted (key, value) list suitable for printing at startup."""
        key = self.llm_api_key
        redacted = f"{key[:8]}...{key[-4:]}" if len(key) > 14 else ("set" if key else "MISSING")
        vision = self.vision_provider or "none"
        if vision == "anthropic":
            vision = "anthropic (native, in-model)"
        elif vision == "gemini":
            vision = f"gemini ({self.gemini_model})"
        return [
            ("mode", "MOCK (no robot)" if self.mock_robot else f"REAL ROBOT @ {self.spot_ip}"),
            ("tool calling", f"{self.llm_provider} ({self.llm_model})"),
            ("vision", vision),
            ("api key", redacted),
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
        llm_provider=(os.getenv("LLM_PROVIDER") or "anthropic").strip().lower(),
        vision_provider=(os.getenv("VISION_PROVIDER") or "").strip().lower(),
        anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY") or "").strip(),
        anthropic_model=(os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-6").strip(),
        groq_api_key=(os.getenv("GROQ_API_KEY") or "").strip(),
        groq_model=(os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip(),
        gemini_api_key=(os.getenv("GEMINI_API_KEY") or "").strip(),
        gemini_model=(os.getenv("GEMINI_MODEL") or "gemini-2.0-flash").strip(),
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
    if not cfg.vision_provider:
        cfg = replace(cfg, vision_provider=_default_vision_provider(cfg))
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _default_vision_provider(cfg: Config) -> str:
    """Pick a sensible vision provider when the operator did not name one.

    On Anthropic the model reads images itself, so there is nothing to add. On a
    text-only provider, prefer Gemini if a key is present -- otherwise the robot
    can take photos it cannot interpret, which is a confusing default.
    """
    if cfg.llm_provider == "anthropic":
        return "anthropic"
    return "gemini" if cfg.gemini_api_key else "none"


def _validate(cfg: Config) -> None:
    """Fail fast on configuration that cannot possibly work."""
    if cfg.llm_provider not in {"anthropic", "groq"}:
        raise ConfigError(
            f"LLM_PROVIDER must be 'anthropic' or 'groq', got {cfg.llm_provider!r}"
        )
    if cfg.vision_provider not in {"", "anthropic", "gemini", "none"}:
        raise ConfigError(
            "VISION_PROVIDER must be 'anthropic', 'gemini' or 'none', "
            f"got {cfg.vision_provider!r}"
        )
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
