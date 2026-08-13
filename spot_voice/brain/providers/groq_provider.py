"""Groq provider -- fast, cheap tool calling for the testing phase.

Groq speaks the OpenAI chat-completions dialect, so this module is a translator:
Anthropic-shaped messages and tool schemas go in, OpenAI shapes go out to Groq,
and the reply comes back as canonical blocks the agent loop already understands.

Two things to know when running on this provider:

* **The models are text-only.** A camera frame cannot be handed to Groq. The
  agent detects that via ``supports_images = False`` and routes the JPEG through
  the vision provider first, feeding back a written description instead. So the
  inspection flow becomes "vision model looks, text model reasons" rather than
  "one model sees the photo".
* **Tool-calling accuracy is lower than the production model.** That is fine for
  wiring and rehearsal, and the parts that actually keep the robot safe do not
  depend on it: safety words never reach any model, and the velocity caps are
  enforced in the robot layer regardless of what gets called.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .base import (
    LLMProvider,
    ProviderResponse,
    TextBlock,
    ToolCall,
    ToolUseBlock,
    block_field,
    block_type,
    strip_cache_control,
)

LOGGER = logging.getLogger(__name__)

REQUEST_TIMEOUT_SEC = 60.0


class GroqProvider(LLMProvider):
    """Tool calling on Groq, translated to and from the canonical format."""

    name = "groq"
    supports_images = False
    supports_prompt_caching = False

    def __init__(self, api_key: str, model: str) -> None:
        import groq

        self._groq = groq
        # max_retries=0: the failure this provider actually hits is
        # `tool_use_failed`, a deterministic 400 from the model emitting
        # malformed function-call syntax. Retrying it changes nothing and cost
        # 44 seconds of dead air on the robot. Handled explicitly in complete()
        # instead; genuine rate limits surface as a spoken message.
        self._client = groq.Groq(api_key=api_key, timeout=REQUEST_TIMEOUT_SEC, max_retries=0)
        self._model = model

    # ------------------------------------------------------------------

    def complete(
        self,
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> ProviderResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=to_openai_messages(system, messages),
                tools=to_openai_tools(tools),
                tool_choice="auto",
            )
        except Exception as exc:
            salvaged = salvage_failed_tool_call(exc)
            if salvaged is None:
                raise
            LOGGER.warning("Groq emitted a malformed tool call; salvaged the text")
            return salvaged

        return from_openai_response(response)

    def describe_error(self, exc: BaseException) -> str | None:
        groq = self._groq
        for attribute, message in (
            ("APIConnectionError", "I can't reach my language service right now, "
                                   "but safety commands still work."),
            ("RateLimitError", "I'm being rate limited. Give me a few seconds and ask again."),
            ("AuthenticationError", "My language service rejected my key. "
                                    "Check the configuration."),
            ("APIStatusError", "My language service returned an error. "
                               "Try again in a moment."),
        ):
            klass = getattr(groq, attribute, None)
            if klass is not None and isinstance(exc, klass):
                LOGGER.warning("Groq %s: %s", attribute, exc)
                return message
        return None


# ----------------------------------------------------------------------
# Translation. Kept as free functions so they are testable without a key.
# ----------------------------------------------------------------------


#: Groq returns this code when the model writes function-call syntax the API
#: cannot parse. It is a property of the weaker model, not of the request.
TOOL_USE_FAILED = "tool_use_failed"


def salvage_failed_tool_call(exc: BaseException) -> ProviderResponse | None:
    """Recover a usable turn from Groq's ``tool_use_failed`` 400.

    The model wanted to act but wrote the call wrongly, e.g.::

        <function=speak":{"text": "I'm standing and my motors are on."}</function>

    The intent is right there in the payload, so rather than surface an error
    after a retry storm, pull the text out and end the turn with it. Returns
    ``None`` for any other failure, which then propagates normally.
    """
    body = getattr(exc, "body", None) or {}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    if error.get("code") != TOOL_USE_FAILED and TOOL_USE_FAILED not in str(exc):
        return None

    generation = str(error.get("failed_generation") or "")
    # The text the model meant to speak, if it got that far.
    match = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', generation)
    text = ""
    if match:
        try:
            text = json.loads(f'"{match.group(1)}"')
        except ValueError:
            text = match.group(1)

    return ProviderResponse(
        content=[TextBlock(text=text)] if text else [],
        tool_calls=[],
        stop_reason="end_turn",
        text=text,
    )


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic tool schemas -> OpenAI function schemas."""
    converted: list[dict[str, Any]] = []
    for tool in strip_cache_control(tools):
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
        )
    return converted


def _text_of(content: Any) -> str:
    """Flatten a message content value to plain text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if block_type(block) == "text":
            parts.append(str(block_field(block, "text", "")))
    return "\n".join(part for part in parts if part).strip()


def to_openai_messages(
    system: list[dict[str, Any]], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Anthropic messages -> OpenAI chat messages.

    The interesting part is tool plumbing: Anthropic carries tool results as
    blocks inside a *user* message, while OpenAI wants one message per result
    with ``role: "tool"``.
    """
    converted: list[dict[str, Any]] = []

    system_text = "\n\n".join(
        str(block.get("text", "")) for block in system if block.get("text")
    ).strip()
    if system_text:
        converted.append({"role": "system", "content": system_text})

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            results = [
                block for block in (content or []) if block_type(block) == "tool_result"
            ] if not isinstance(content, str) else []
            if results:
                for block in results:
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": block_field(block, "tool_use_id", ""),
                            "content": _tool_result_text(block_field(block, "content")),
                        }
                    )
                # Any plain text alongside the results still belongs to the user.
                extra = _text_of(
                    [b for b in content if block_type(b) == "text"]
                )
                if extra:
                    converted.append({"role": "user", "content": extra})
            else:
                converted.append({"role": "user", "content": _text_of(content)})
            continue

        if role == "assistant":
            text = _text_of(content)
            tool_calls = []
            for block in content or []:
                if block_type(block) != "tool_use":
                    continue
                tool_calls.append(
                    {
                        "id": block_field(block, "id", ""),
                        "type": "function",
                        "function": {
                            "name": block_field(block, "name", ""),
                            "arguments": json.dumps(block_field(block, "input") or {}),
                        },
                    }
                )
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            converted.append(entry)
            continue

    return converted


def _tool_result_text(content: Any) -> str:
    """Flatten a tool_result's content to text.

    Image blocks are dropped here by design -- if a frame reached this point the
    provider is text-only, and the agent has already replaced it with a written
    description. This is the belt to that braces.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if block_type(block) == "text":
            parts.append(str(block_field(block, "text", "")))
    return "\n".join(parts) if parts else "{}"


def from_openai_response(response: Any) -> ProviderResponse:
    """OpenAI-shaped completion -> canonical :class:`ProviderResponse`."""
    choice = response.choices[0]
    message = choice.message

    content: list[Any] = []
    text = (getattr(message, "content", None) or "").strip()
    if text:
        content.append(TextBlock(text=text))

    tool_calls: list[ToolCall] = []
    for call in getattr(message, "tool_calls", None) or []:
        raw = call.function.arguments or "{}"
        try:
            arguments = json.loads(raw)
        except (TypeError, ValueError):
            # A malformed argument blob must not raise -- the dispatcher will
            # reject an empty input with a speakable message instead.
            LOGGER.warning("Could not parse tool arguments: %r", raw)
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        tool_calls.append(ToolCall(call.id, call.function.name, arguments))
        content.append(ToolUseBlock(id=call.id, name=call.function.name, input=arguments))

    finish = getattr(choice, "finish_reason", "stop")
    stop_reason = {
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "length": "max_tokens",
        "stop": "end_turn",
        "content_filter": "refusal",
    }.get(finish, "end_turn")
    if tool_calls and stop_reason == "end_turn":
        # Some models return finish_reason "stop" alongside tool calls.
        stop_reason = "tool_use"

    usage = getattr(response, "usage", None)
    return ProviderResponse(
        content=content,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        text=text,
        usage={
            "input": getattr(usage, "prompt_tokens", None),
            "output": getattr(usage, "completion_tokens", None),
        },
    )
