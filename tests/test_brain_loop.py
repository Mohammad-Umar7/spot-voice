"""End-to-end shape of the tool-use loop, against a scripted fake provider.

No API key, no network, no provider SDK. Because the loop now sits on the
provider interface, this exercises the real loop rather than a mocked SDK.
"""

from __future__ import annotations

import json

import pytest
from rich.console import Console

from spot_voice.brain.agent import Brain
from spot_voice.brain.dispatcher import ToolDispatcher
from spot_voice.brain.providers import LLMProvider, ProviderResponse, TextBlock, ToolCall
from spot_voice.brain.providers.base import ToolUseBlock, VisionProvider
from spot_voice.robot.mock import MockSpot

QUIET = Console(quiet=True)


# ----------------------------------------------------------------------
# Fakes


class FakeError(Exception):
    """A provider-specific error the fake knows how to describe."""


class FakeProvider(LLMProvider):
    """Replays a scripted list of responses and records every request."""

    name = "fake"

    def __init__(
        self,
        script,
        supports_images: bool = True,
        supports_prompt_caching: bool = True,
    ) -> None:
        self.script = list(script)
        self.supports_images = supports_images
        # Decides whether the Brain sends the detailed prompt or the compact
        # one, so tests have to be explicit about it.
        self.supports_prompt_caching = supports_prompt_caching
        self.calls: list[dict] = []

    def complete(self, system, tools, messages, max_tokens):
        # Snapshot: the Brain passes its live history and keeps appending.
        self.calls.append(
            {
                "system": system,
                "tools": tools,
                "messages": list(messages),
                "max_tokens": max_tokens,
            }
        )
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def describe_error(self, exc):
        if isinstance(exc, FakeError):
            return "I can't reach my language service right now, but safety commands still work."
        return None


class FakeVision(VisionProvider):
    name = "fake-vision"

    def __init__(self, text: str = "Six gauges and a hazard sign.") -> None:
        self.text = text
        self.seen: list[bytes] = []

    def describe(self, image_jpeg: bytes, prompt: str = "") -> str:
        self.seen.append(image_jpeg)
        return self.text


class ExplodingVision(VisionProvider):
    name = "broken-vision"

    def describe(self, image_jpeg: bytes, prompt: str = "") -> str:
        raise RuntimeError("quota exceeded")


def reply(text: str) -> ProviderResponse:
    return ProviderResponse(
        content=[TextBlock(text=text)], stop_reason="end_turn", text=text
    )


def calls(*specs) -> ProviderResponse:
    """Build a tool_use turn from ``(id, name, input)`` triples."""
    tool_calls = [ToolCall(i, n, a) for i, n, a in specs]
    content = [ToolUseBlock(id=i, name=n, input=a) for i, n, a in specs]
    return ProviderResponse(content=content, tool_calls=tool_calls, stop_reason="tool_use")


@pytest.fixture()
def dispatcher():
    robot = MockSpot(console=QUIET)
    robot.connect()
    return ToolDispatcher(robot=robot, console=QUIET)


def make_brain(
    dispatcher, script, supports_images=True, vision=None, caching=True
):
    provider = FakeProvider(
        script, supports_images=supports_images, supports_prompt_caching=caching
    )
    brain = Brain(provider=provider, dispatcher=dispatcher, vision=vision, console=QUIET)
    return brain, provider


# ----------------------------------------------------------------------
# Happy paths


def test_plain_answer_needs_no_tools(dispatcher):
    brain, provider = make_brain(dispatcher, [reply("I'm ready.")])

    result = brain.handle("are you there")

    assert result.text == "I'm ready."
    assert result.tool_calls == []
    assert len(provider.calls) == 1


def test_tool_call_round_trip(dispatcher):
    brain, provider = make_brain(
        dispatcher, [calls(("t1", "stand", {})), reply("Standing.")]
    )

    result = brain.handle("stand up please")

    assert result.text == "Standing."
    assert [call.name for call in result.tool_calls] == ["stand"]

    tool_turn = provider.calls[1]["messages"][-1]
    assert tool_turn["role"] == "user"
    assert tool_turn["content"][0]["type"] == "tool_result"
    assert json.loads(tool_turn["content"][0]["content"][0]["text"])["ok"] is True


def test_parallel_tool_calls_come_back_in_one_user_message(dispatcher):
    brain, provider = make_brain(
        dispatcher,
        [calls(("a", "get_status", {}), ("b", "list_waypoints", {})), reply("All good.")],
    )

    brain.handle("status and places")

    tool_turn = provider.calls[1]["messages"][-1]
    assert len(tool_turn["content"]) == 2
    assert {block["tool_use_id"] for block in tool_turn["content"]} == {"a", "b"}


# ----------------------------------------------------------------------
# Images: the behaviour that differs by provider


def test_a_seeing_provider_gets_the_photo_itself(dispatcher):
    brain, provider = make_brain(
        dispatcher,
        [calls(("t1", "capture_image", {})), reply("I see a control panel.")],
        supports_images=True,
    )

    brain.handle("what do you see")

    blocks = provider.calls[1]["messages"][-1]["content"][0]["content"]
    image = next(block for block in blocks if block["type"] == "image")
    assert image["source"]["media_type"] == "image/jpeg"


def test_a_text_only_provider_gets_a_written_description_instead(dispatcher):
    vision = FakeVision("Six gauges, a hazard sign, and a pallet.")
    brain, provider = make_brain(
        dispatcher,
        [calls(("t1", "capture_image", {})), reply("I see gauges and a hazard sign.")],
        supports_images=False,
        vision=vision,
    )

    brain.handle("what do you see")

    blocks = provider.calls[1]["messages"][-1]["content"][0]["content"]
    # No image block at all -- the model could not read one.
    assert all(block["type"] != "image" for block in blocks)
    payload = json.loads(blocks[0]["text"])
    assert payload["image_description"] == "Six gauges, a hazard sign, and a pallet."
    assert len(vision.seen) == 1
    assert vision.seen[0][:2] == b"\xff\xd8"  # a real JPEG reached the vision model


def test_a_broken_vision_provider_says_so_rather_than_inventing(dispatcher):
    brain, provider = make_brain(
        dispatcher,
        [calls(("t1", "capture_image", {})), reply("I couldn't see it.")],
        supports_images=False,
        vision=ExplodingVision(),
    )

    brain.handle("what do you see")

    payload = json.loads(
        provider.calls[1]["messages"][-1]["content"][0]["content"][0]["text"]
    )
    assert "could not see" in payload["image_description"]
    assert "RuntimeError" in payload["image_description"]


def test_no_vision_provider_at_all_is_handled(dispatcher):
    brain, provider = make_brain(
        dispatcher,
        [calls(("t1", "capture_image", {})), reply("I can't see it.")],
        supports_images=False,
        vision=None,
    )

    brain.handle("what do you see")

    payload = json.loads(
        provider.calls[1]["messages"][-1]["content"][0]["content"][0]["text"]
    )
    assert "cannot see" in payload["image_description"]


# ----------------------------------------------------------------------
# Prompt shape


def test_system_and_tools_carry_cache_breakpoints(dispatcher):
    brain, provider = make_brain(dispatcher, [reply("hi")])

    brain.handle("hello")

    call = provider.calls[0]
    assert call["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert call["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_system_and_tools_are_byte_identical_across_turns(dispatcher):
    brain, provider = make_brain(dispatcher, [reply("one"), reply("two")])

    brain.handle("first")
    brain.handle("second")

    # Any drift in the prefix would silently destroy the cache hit rate.
    assert provider.calls[0]["system"] == provider.calls[1]["system"]
    assert provider.calls[0]["tools"] == provider.calls[1]["tools"]


# ----------------------------------------------------------------------
# Failure paths


def test_provider_failure_yields_a_speakable_sentence(dispatcher):
    brain, _provider = make_brain(dispatcher, [FakeError("no route to host")])

    result = brain.handle("what do you see")

    assert result.error
    assert "safety commands still work" in result.text
    assert brain._messages == []  # the unanswered turn must not linger


def test_an_undescribed_error_still_produces_something_speakable(dispatcher):
    brain, _provider = make_brain(dispatcher, [ValueError("something odd")])

    result = brain.handle("hello")

    assert result.text
    assert "Safety commands still work" in result.text


def test_refusal_is_handled_without_touching_content(dispatcher):
    brain, _provider = make_brain(
        dispatcher, [ProviderResponse(content=[], stop_reason="refusal")]
    )

    assert brain.handle("do something unsafe").text == "I can't help with that one."


def test_abort_cancels_an_in_flight_tool_loop(dispatcher, monkeypatch):
    brain, _provider = make_brain(
        dispatcher, [calls(("t1", "get_status", {})), reply("never reached")]
    )
    original = dispatcher.dispatch

    def dispatch_then_abort(name, tool_input):
        result = original(name, tool_input)
        brain.abort()  # the operator said "stop" mid-sequence
        return result

    monkeypatch.setattr(dispatcher, "dispatch", dispatch_then_abort)

    result = brain.handle("what's your status")

    assert result.aborted is True
    assert result.text == ""


def test_the_loop_cannot_run_away(dispatcher):
    from spot_voice.brain.agent import MAX_TOOL_ITERATIONS

    script = [calls((f"t{i}", "get_status", {})) for i in range(MAX_TOOL_ITERATIONS + 4)]
    brain, provider = make_brain(dispatcher, script)

    result = brain.handle("loop forever")

    assert len(provider.calls) == MAX_TOOL_ITERATIONS
    assert result.text


# ----------------------------------------------------------------------
# Prompt size is chosen per provider
#
# Groq has no prompt caching and a 12,000 token/minute free tier, while the full
# prompt is ~2300 tokens -- four requests a minute, which one spoken sentence
# can exhaust. The detailed prompt is for providers that cache it.


def test_a_non_caching_provider_gets_a_much_smaller_prompt(dispatcher):
    import json

    caching, cached_provider = make_brain(dispatcher, [reply("hi")], caching=True)
    caching.handle("hello")

    lean, lean_provider = make_brain(dispatcher, [reply("hi")], caching=False)
    lean.handle("hello")

    def size(call):
        return len(json.dumps(call["system"])) + len(json.dumps(call["tools"]))

    big = size(cached_provider.calls[0])
    small = size(lean_provider.calls[0])
    assert small < big * 0.65, f"compact prompt is {small} vs {big}"


def test_the_compact_prompt_keeps_every_tool_and_its_enums(dispatcher):
    from spot_voice.brain.tools import TOOL_NAMES

    brain, provider = make_brain(dispatcher, [reply("hi")], caching=False)
    brain.handle("hello")

    tools = provider.calls[0]["tools"]
    assert {tool["name"] for tool in tools} == set(TOOL_NAMES)
    # The enums are what actually constrain the model; only prose is dropped.
    move = next(tool for tool in tools if tool["name"] == "move")
    assert move["input_schema"]["properties"]["direction"]["enum"]
    assert move["input_schema"]["required"] == ["direction"]


def test_a_non_caching_provider_keeps_a_shorter_history(dispatcher):
    from spot_voice.brain.agent import COMPACT_HISTORY_MESSAGES, MAX_HISTORY_MESSAGES

    caching, _ = make_brain(dispatcher, [reply("x")], caching=True)
    lean, _ = make_brain(dispatcher, [reply("x")], caching=False)

    assert caching._max_history == MAX_HISTORY_MESSAGES
    assert lean._max_history == COMPACT_HISTORY_MESSAGES
    assert COMPACT_HISTORY_MESSAGES < MAX_HISTORY_MESSAGES
