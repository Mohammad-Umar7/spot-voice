"""The language lane: a tool-use loop over a swappable model provider.

The loop itself is provider-independent. It sends a turn, executes whatever
tools come back through the single dispatcher, feeds the results in, and repeats
until the model stops calling tools. Which model is on the other end -- Anthropic
in production, Groq during bring-up -- is a config decision made in
:mod:`spot_voice.brain.providers`.

Design notes:

* **Canonical format is Anthropic's.** History, content blocks and tool schemas
  are held in Anthropic shape; adapters translate at the edge. Switching to
  Anthropic at the end therefore removes translation rather than adding it.
* **Images depend on the provider.** When the model can see (Anthropic), a
  camera frame is attached as an image block. When it cannot (Groq), the frame
  goes to the vision provider first and its written description takes the
  image's place -- so the tool-calling model reads prose rather than pixels.
* **Rolling history never orphans a tool_result.** Trimming repairs the window
  so it always starts on a clean user turn.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from .dispatcher import DispatchResult, ToolDispatcher
from .prompts import system_blocks
from .providers import LLMProvider, VisionProvider
from .tools import tools_with_cache_breakpoint

LOGGER = logging.getLogger(__name__)

#: Roughly ten turns of conversation, counting tool round-trips.
MAX_HISTORY_MESSAGES = 24
#: Ceiling on tool round-trips within a single operator utterance.
MAX_TOOL_ITERATIONS = 8
#: Spoken replies are short; this is generous headroom, not a target.
MAX_TOKENS = 1024


@dataclass
class BrainReply:
    """The result of handling one operator utterance."""

    text: str
    tool_calls: list[DispatchResult] = field(default_factory=list)
    error: str | None = None
    aborted: bool = False


class Brain:
    """Runs the tool-use loop for one voice session.

    Args:
        provider: The tool-calling model.
        dispatcher: Executes tool calls against the robot.
        vision: Used to describe camera frames when ``provider`` cannot see
            images. Ignored when it can.
        extra_context: Stable, site-specific text appended to the system prompt
            (for example the waypoint list on this map).
        console: Rich console for the conversation log.
    """

    def __init__(
        self,
        provider: LLMProvider,
        dispatcher: ToolDispatcher,
        vision: VisionProvider | None = None,
        extra_context: str | None = None,
        console: Console | None = None,
    ) -> None:
        self._provider = provider
        self._dispatcher = dispatcher
        self._vision = vision
        self._console = console or Console()
        self._system = system_blocks(extra_context)
        self._tools = tools_with_cache_breakpoint()
        self._messages: list[dict[str, Any]] = []
        self._abort = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def abort(self) -> None:
        """Abandon the current tool loop.

        Called by the reflex lane so a spoken "stop" ends an in-flight sequence
        of tool calls rather than letting the rest of the plan play out.
        """
        self._abort.set()

    def reset(self) -> None:
        """Forget the conversation history."""
        with self._lock:
            self._messages.clear()

    # ------------------------------------------------------------------

    def handle(self, transcript: str) -> BrainReply:
        """Send one utterance through the tool-use loop and return the reply."""
        with self._lock:
            self._abort.clear()
            self._messages.append({"role": "user", "content": transcript})
            self._trim()

            reply = BrainReply(text="")
            for _iteration in range(MAX_TOOL_ITERATIONS):
                if self._abort.is_set():
                    reply.aborted = True
                    reply.text = ""
                    break

                try:
                    response = self._provider.complete(
                        system=self._system,
                        tools=self._tools,
                        messages=self._messages,
                        max_tokens=MAX_TOKENS,
                    )
                except Exception as exc:
                    reply.error = self._describe_api_error(exc)
                    reply.text = reply.error
                    # Drop the unanswered user turn so the next utterance starts
                    # from a consistent history.
                    if self._messages and self._messages[-1]["role"] == "user":
                        self._messages.pop()
                    return reply

                self._log_usage(response)

                if response.stop_reason == "refusal":
                    reply.text = "I can't help with that one."
                    self._messages.append({"role": "assistant", "content": reply.text})
                    return reply

                self._messages.append({"role": "assistant", "content": response.content})

                if not response.tool_calls:
                    reply.text = response.text
                    if response.stop_reason == "max_tokens" and not reply.text:
                        reply.text = "I ran long there. Ask me again more specifically."
                    self._trim()
                    return reply

                results_content: list[dict[str, Any]] = []
                for call in response.tool_calls:
                    if self._abort.is_set():
                        results_content.append(
                            self._tool_result(
                                call.id,
                                {"ok": False, "message": "Cancelled: the operator said stop."},
                            )
                        )
                        continue
                    outcome = self._dispatcher.dispatch(call.name, call.input)
                    reply.tool_calls.append(outcome)
                    results_content.append(
                        self._tool_result(call.id, outcome.payload, outcome.image_jpeg)
                    )

                # All tool results for one assistant turn go back in ONE user
                # message -- splitting them teaches the model to stop calling
                # tools in parallel.
                self._messages.append({"role": "user", "content": results_content})
                self._trim()

                if self._abort.is_set():
                    reply.aborted = True
                    reply.text = ""
                    break
            else:
                reply.text = (
                    "That's taking more steps than I expected. "
                    "What would you like me to do?"
                )
                LOGGER.warning("Tool loop hit the %d iteration cap", MAX_TOOL_ITERATIONS)

            return reply

    # ------------------------------------------------------------------

    def _tool_result(
        self, tool_use_id: str, payload: dict[str, Any], image_jpeg: bytes | None = None
    ) -> dict[str, Any]:
        """Build a ``tool_result`` block, handling the image however this
        provider needs it.

        Failures come back as ordinary results with ``ok: false`` rather than
        ``is_error``: the message is written to be spoken, and the model should
        read it as data about the robot, not as a broken tool.
        """
        payload = dict(payload)

        if image_jpeg and not self._provider.supports_images:
            # Text-only model: the frame cannot go in, so a description does.
            payload["image_description"] = self._describe_image(image_jpeg)
            image_jpeg = None

        content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload)}]
        if image_jpeg:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(image_jpeg).decode("ascii"),
                    },
                }
            )
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}

    def _describe_image(self, image_jpeg: bytes) -> str:
        """Turn a camera frame into words. Never raises."""
        if self._vision is None:
            return (
                "A photo was captured but there is no way to interpret it. "
                "Tell the operator you cannot see it."
            )
        try:
            description = self._vision.describe(image_jpeg)
        except Exception as exc:
            LOGGER.warning("Vision provider failed", exc_info=True)
            return (
                "A photo was captured but the vision service failed "
                f"({type(exc).__name__}). Tell the operator you could not see it."
            )
        self._console.print(f"[blue]vision[/blue] [dim]{description}[/dim]")
        return description

    def _describe_api_error(self, exc: BaseException) -> str:
        """Turn a provider exception into one speakable sentence."""
        described = self._provider.describe_error(exc)
        if described is not None:
            return described
        LOGGER.exception("Unexpected error talking to %s", self._provider.name)
        return "Something went wrong on my side. Safety commands still work."

    def _log_usage(self, response: Any) -> None:
        """Log token usage, including whether the prompt cache is being hit."""
        usage = response.usage or {}
        LOGGER.info(
            "%s in=%s out=%s cache_write=%s cache_read=%s stop=%s",
            self._provider.name,
            usage.get("input", "?"),
            usage.get("output", "?"),
            usage.get("cache_write", "-"),
            usage.get("cache_read", "-"),
            response.stop_reason,
        )

    def _trim(self) -> None:
        """Keep the rolling window bounded without orphaning tool results.

        A ``tool_result`` block is only valid directly after the ``tool_use``
        that produced it, so after dropping old messages the window is repaired
        until it starts on a user turn that contains no tool results.
        """
        while len(self._messages) > MAX_HISTORY_MESSAGES:
            self._messages.pop(0)
        while self._messages and not _is_clean_start(self._messages[0]):
            self._messages.pop(0)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _collect_text(content: Any) -> str:
    """Join every text block of an assistant response into one spoken string."""
    parts: list[str] = []
    for block in content or []:
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return " ".join(part.strip() for part in parts if part and part.strip()).strip()


def _tool_result_block(
    tool_use_id: str, payload: dict[str, Any], image_jpeg: bytes | None = None
) -> dict[str, Any]:
    """Build a ``tool_result`` block with a native image attachment.

    Kept as a free function for the provider-independent tests; the Brain uses
    :meth:`Brain._tool_result`, which additionally routes images through the
    vision provider when the model cannot see them.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(payload)}]
    if image_jpeg:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(image_jpeg).decode("ascii"),
                },
            }
        )
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def _is_clean_start(message: dict[str, Any]) -> bool:
    """True when a message can legally be the first in the history window."""
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return not any(
            (isinstance(block, dict) and block.get("type") == "tool_result")
            or getattr(block, "type", "") == "tool_result"
            for block in content
        )
    return False
