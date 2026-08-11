"""The provider interface, and the canonical message format behind it.

**The internal format is Anthropic's.** Messages, content blocks and tool
schemas are all held in Anthropic shape, and each provider translates to and
from its own wire format at the edge. That choice is deliberate: Anthropic is
where this project ends up, so the final switch is a config change with no
translation left in the path, and the Groq adapter is the piece that gets
deleted rather than the core.

A provider's job is narrow: take (system, tools, messages), return text plus any
tool calls. The agent loop, the dispatcher, the caps and the reflex lane are all
provider-independent.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """One tool the model wants executed, in canonical form."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class TextBlock:
    """A canonical assistant text block."""

    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    """A canonical assistant tool_use block."""

    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class ProviderResponse:
    """One model turn, normalised across providers.

    Attributes:
        content: Assistant content blocks to append to history verbatim.
        tool_calls: Tool calls to execute, in order.
        stop_reason: ``end_turn`` | ``tool_use`` | ``max_tokens`` | ``refusal``.
        text: All text blocks joined, ready to speak.
        usage: Token accounting, best effort and provider-shaped.
    """

    content: list[Any]
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


class LLMProvider(abc.ABC):
    """A model that can hold a conversation and call tools."""

    #: Short name for logs and the startup banner.
    name: str = "provider"

    #: Whether the model can read an image content block directly. When False,
    #: the agent routes camera frames through a vision provider first and hands
    #: back a text description instead.
    supports_images: bool = False

    #: Whether the provider bills for and honours prompt-cache breakpoints.
    supports_prompt_caching: bool = False

    @abc.abstractmethod
    def complete(
        self,
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> ProviderResponse:
        """Run one turn. Raises on transport failure; the agent translates that."""

    @abc.abstractmethod
    def describe_error(self, exc: BaseException) -> str | None:
        """Return a speakable sentence for a provider-specific error, or None."""


class VisionProvider(abc.ABC):
    """A model that can look at a JPEG and say what is in it."""

    name: str = "vision"

    @abc.abstractmethod
    def describe(self, image_jpeg: bytes, prompt: str) -> str:
        """Return a short description of the image. Raises on failure."""


def strip_cache_control(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop Anthropic ``cache_control`` keys, which other providers reject."""
    cleaned: list[dict[str, Any]] = []
    for item in items:
        copy = {key: value for key, value in item.items() if key != "cache_control"}
        cleaned.append(copy)
    return cleaned


def block_type(block: Any) -> str:
    """Read a block's type whether it is a dict or an SDK/canonical object."""
    if isinstance(block, dict):
        return str(block.get("type", ""))
    return str(getattr(block, "type", ""))


def block_field(block: Any, name: str, default: Any = None) -> Any:
    """Read a field off a block whether it is a dict or an object."""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)
