"""The Claude lane: a direct Anthropic Messages API tool-use loop.

No MCP, no tool servers, no framework -- just ``anthropic.Anthropic`` and a loop
over ``tool_use`` blocks. Anything the reflex lane did not catch arrives here.

Design notes:

* **Prompt caching.** Tools render before ``system``, and both are frozen for the
  whole session, so a single cache breakpoint on each keeps the fixed prefix warm
  across the many short turns a voice session produces.
* **No thinking parameter.** Voice replies are one or two sentences and latency
  is what the operator feels. Omitting ``thinking`` gives the fastest first token
  on the default model and is valid on every current model, so the model id in
  ``.env`` can be swapped without touching code.
* **Rolling history.** Trimming never orphans a ``tool_result``: the window is
  repaired so it always starts on a clean user turn.
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
from .tools import tools_with_cache_breakpoint

LOGGER = logging.getLogger(__name__)

#: Roughly ten turns of conversation, counting tool round-trips.
MAX_HISTORY_MESSAGES = 24
#: Ceiling on tool round-trips within a single operator utterance.
MAX_TOOL_ITERATIONS = 8
#: Spoken replies are short; this is generous headroom, not a target.
MAX_TOKENS = 1024
#: Request timeout, generous because the Anthropic call may be going out over a
#: phone tether while the robot link uses the other interface.
REQUEST_TIMEOUT_SEC = 60.0


@dataclass
class BrainReply:
    """The result of handling one operator utterance."""

    text: str
    tool_calls: list[DispatchResult] = field(default_factory=list)
    error: str | None = None
    aborted: bool = False


class Brain:
    """Runs the Anthropic tool-use loop for one voice session.

    Args:
        api_key: Anthropic API key (from ``.env``; never hardcoded).
        model: Model id, e.g. ``claude-sonnet-4-6``.
        dispatcher: Executes tool calls against the robot.
        extra_context: Optional stable, site-specific text appended to the system
            prompt (for example the waypoint list on this map).
        console: Rich console for the conversation log.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        dispatcher: ToolDispatcher,
        extra_context: str | None = None,
        console: Console | None = None,
    ) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=REQUEST_TIMEOUT_SEC,
            max_retries=2,
        )
        self._model = model
        self._dispatcher = dispatcher
        self._console = console or Console()
        self._system = system_blocks(extra_context)
        self._tools = tools_with_cache_breakpoint()
        self._messages: list[dict[str, Any]] = []
        self._abort = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

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
            for iteration in range(MAX_TOOL_ITERATIONS):
                if self._abort.is_set():
                    reply.aborted = True
                    reply.text = ""
                    break

                try:
                    response = self._client.messages.create(
                        model=self._model,
                        max_tokens=MAX_TOKENS,
                        system=self._system,
                        tools=self._tools,
                        messages=self._messages,
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

                # Guard before touching content: a refusal carries no usable text.
                if response.stop_reason == "refusal":
                    reply.text = "I can't help with that one."
                    self._messages.append(
                        {"role": "assistant", "content": reply.text}
                    )
                    return reply

                self._messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "pause_turn":
                    # A server-side tool paused; re-send to let it resume.
                    continue

                if response.stop_reason != "tool_use":
                    reply.text = _collect_text(response.content)
                    if response.stop_reason == "max_tokens" and not reply.text:
                        reply.text = "I ran long there. Ask me again more specifically."
                    self._trim()
                    return reply

                tool_blocks = [
                    block for block in response.content if getattr(block, "type", "") == "tool_use"
                ]
                results_content: list[dict[str, Any]] = []
                for block in tool_blocks:
                    if self._abort.is_set():
                        results_content.append(
                            _tool_result_block(
                                block.id,
                                {"ok": False, "message": "Cancelled: the operator said stop."},
                            )
                        )
                        continue
                    outcome = self._dispatcher.dispatch(block.name, block.input)
                    reply.tool_calls.append(outcome)
                    results_content.append(
                        _tool_result_block(block.id, outcome.payload, outcome.image_jpeg)
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
                reply.text = "That's taking more steps than I expected. What would you like me to do?"
                LOGGER.warning("Tool loop hit the %d iteration cap", MAX_TOOL_ITERATIONS)

            return reply

    # ------------------------------------------------------------------

    def _describe_api_error(self, exc: BaseException) -> str:
        """Turn an SDK exception into one speakable sentence."""
        anthropic = self._anthropic
        if isinstance(exc, anthropic.APIConnectionError):
            LOGGER.warning("Anthropic connection error: %s", exc)
            return "I can't reach my language service right now, but safety commands still work."
        if isinstance(exc, anthropic.RateLimitError):
            LOGGER.warning("Anthropic rate limited")
            return "I'm being rate limited. Give me a few seconds and ask again."
        if isinstance(exc, anthropic.AuthenticationError):
            LOGGER.error("Anthropic auth failed")
            return "My language service rejected my key. Check the configuration."
        if isinstance(exc, anthropic.APIStatusError):
            LOGGER.error("Anthropic API error %s: %s", exc.status_code, exc)
            return "My language service returned an error. Try again in a moment."
        LOGGER.exception("Unexpected error talking to Anthropic")
        return "Something went wrong on my side. Safety commands still work."

    def _log_usage(self, response: Any) -> None:
        """Log token usage, including whether the prompt cache is being hit."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        LOGGER.info(
            "anthropic in=%s out=%s cache_write=%s cache_read=%s stop=%s",
            getattr(usage, "input_tokens", "?"),
            getattr(usage, "output_tokens", "?"),
            getattr(usage, "cache_creation_input_tokens", 0),
            getattr(usage, "cache_read_input_tokens", 0),
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
    """Build a ``tool_result`` block, attaching an image when the tool produced one.

    Failures come back as ordinary results with ``ok: false`` rather than
    ``is_error``: the message is written to be spoken, and the model should read
    it as data about the robot, not as a broken tool.
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
