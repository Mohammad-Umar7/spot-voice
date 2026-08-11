"""Model providers, selected by ``LLM_PROVIDER`` and ``VISION_PROVIDER``.

The split exists so the project can be wired up and rehearsed on cheap, fast
providers and then moved to Anthropic for production by changing two lines of
``.env`` -- no code change, and nothing above this package knows the difference.

Provider matrix:

===============  =============  ==============  =========================
LLM_PROVIDER     tool calling   sees images     prompt caching
===============  =============  ==============  =========================
``anthropic``    native         yes, natively   yes
``groq``         OpenAI shape   no, via Gemini  no
===============  =============  ==============  =========================
"""

from __future__ import annotations

import logging

from .base import (
    LLMProvider,
    ProviderResponse,
    TextBlock,
    ToolCall,
    ToolUseBlock,
    VisionProvider,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "LLMProvider",
    "ProviderResponse",
    "TextBlock",
    "ToolCall",
    "ToolUseBlock",
    "VisionProvider",
    "build_llm_provider",
    "build_vision_provider",
    "ProviderUnavailable",
]


class ProviderUnavailable(RuntimeError):
    """Raised when a provider is selected but cannot be constructed."""


def build_llm_provider(config) -> LLMProvider:
    """Construct the tool-calling provider named by ``LLM_PROVIDER``.

    Raises:
        ProviderUnavailable: If the key is missing or the SDK is not installed.
    """
    choice = (config.llm_provider or "anthropic").lower()

    if choice == "anthropic":
        if not config.anthropic_api_key:
            raise ProviderUnavailable("ANTHROPIC_API_KEY is not set")
        from .anthropic_provider import AnthropicProvider

        try:
            return AnthropicProvider(config.anthropic_api_key, config.anthropic_model)
        except ImportError as exc:
            raise ProviderUnavailable(f"pip install anthropic ({exc})") from exc

    if choice == "groq":
        if not config.groq_api_key:
            raise ProviderUnavailable("GROQ_API_KEY is not set")
        from .groq_provider import GroqProvider

        try:
            return GroqProvider(config.groq_api_key, config.groq_model)
        except ImportError as exc:
            raise ProviderUnavailable(f"pip install groq ({exc})") from exc

    raise ProviderUnavailable(
        f"LLM_PROVIDER must be 'anthropic' or 'groq', got {config.llm_provider!r}"
    )


def build_vision_provider(config) -> VisionProvider | None:
    """Construct the image provider, or ``None`` when the LLM sees images itself.

    Never raises: a missing vision provider degrades to a robot that can take a
    photo and says it cannot interpret it, which is better than refusing to
    start.
    """
    choice = (config.vision_provider or "").lower()

    if choice in {"", "none"}:
        from .gemini_vision import NullVisionProvider

        return NullVisionProvider()

    if choice == "gemini":
        if not config.gemini_api_key:
            LOGGER.warning("VISION_PROVIDER=gemini but GEMINI_API_KEY is not set")
            from .gemini_vision import NullVisionProvider

            return NullVisionProvider()
        from .gemini_vision import GeminiVisionProvider

        try:
            return GeminiVisionProvider(config.gemini_api_key, config.gemini_model)
        except ImportError as exc:
            LOGGER.warning("Gemini unavailable (%s) -- pip install google-generativeai", exc)
            from .gemini_vision import NullVisionProvider

            return NullVisionProvider()

    if choice == "anthropic":
        # The LLM provider handles images natively; nothing extra to build.
        return None

    LOGGER.warning("Unknown VISION_PROVIDER %r -- images will not be interpreted", choice)
    from .gemini_vision import NullVisionProvider

    return NullVisionProvider()
