"""Anthropic Messages API provider -- the production target.

This one is a passthrough: the canonical format *is* Anthropic's, so there is no
translation, images ride natively as image content blocks, and the prompt-cache
breakpoints on the system prompt and tool definitions do their job.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import LLMProvider, ProviderResponse, ToolCall

LOGGER = logging.getLogger(__name__)

#: Request timeout. Generous because this call may be going out over a phone
#: tether while the robot link uses the other interface.
REQUEST_TIMEOUT_SEC = 60.0


class AnthropicProvider(LLMProvider):
    """Direct Anthropic Messages API tool use. No MCP, no framework."""

    name = "anthropic"
    supports_images = True
    supports_prompt_caching = True

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=REQUEST_TIMEOUT_SEC, max_retries=2
        )
        self._model = model

    def complete(
        self,
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> ProviderResponse:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )

        # A refusal carries no usable content; guard before reading it.
        if response.stop_reason == "refusal":
            return ProviderResponse(content=[], stop_reason="refusal")

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", "") == "tool_use":
                tool_calls.append(ToolCall(block.id, block.name, dict(block.input or {})))

        usage = getattr(response, "usage", None)
        return ProviderResponse(
            content=list(response.content),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            text=" ".join(part.strip() for part in text_parts if part.strip()).strip(),
            usage={
                "input": getattr(usage, "input_tokens", None),
                "output": getattr(usage, "output_tokens", None),
                "cache_write": getattr(usage, "cache_creation_input_tokens", 0),
                "cache_read": getattr(usage, "cache_read_input_tokens", 0),
            },
        )

    def describe_error(self, exc: BaseException) -> str | None:
        anthropic = self._anthropic
        if isinstance(exc, anthropic.APIConnectionError):
            LOGGER.warning("Anthropic connection error: %s", exc)
            return (
                "I can't reach my language service right now, but safety "
                "commands still work."
            )
        if isinstance(exc, anthropic.RateLimitError):
            LOGGER.warning("Anthropic rate limited")
            return "I'm being rate limited. Give me a few seconds and ask again."
        if isinstance(exc, anthropic.AuthenticationError):
            LOGGER.error("Anthropic auth failed")
            return "My language service rejected my key. Check the configuration."
        if isinstance(exc, anthropic.APIStatusError):
            LOGGER.error("Anthropic API error %s: %s", exc.status_code, exc)
            return "My language service returned an error. Try again in a moment."
        return None
