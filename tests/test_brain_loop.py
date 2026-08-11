"""End-to-end shape of the Anthropic tool-use loop, with a scripted fake client.

No API key and no network: a stand-in ``anthropic`` module is injected so the
loop, the tool round-trip, the image attachment and the error paths can all be
exercised deterministically.
"""

from __future__ import annotations

import json
import sys
import types

import pytest
from rich.console import Console

from spot_voice.brain.dispatcher import ToolDispatcher
from spot_voice.robot.mock import MockSpot

QUIET = Console(quiet=True)


# ----------------------------------------------------------------------
# A minimal stand-in for the anthropic SDK.


class _Block:
    def __init__(self, type_: str, **fields) -> None:
        self.type = type_
        for key, value in fields.items():
            setattr(self, key, value)


class _Usage:
    input_tokens = 100
    output_tokens = 20
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 90


class _Response:
    def __init__(self, content, stop_reason: str) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _Usage()


class _Messages:
    def __init__(self, script) -> None:
        self._script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        # Snapshot the message list: the Brain passes its live history and keeps
        # appending to it, so keeping the reference would show later state.
        snapshot = dict(kwargs)
        snapshot["messages"] = list(kwargs.get("messages", []))
        self.calls.append(snapshot)
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class _FakeClient:
    def __init__(self, script, **_kwargs) -> None:
        self.messages = _Messages(script)


def _install_fake_anthropic(monkeypatch, script):
    """Put a fake ``anthropic`` module in ``sys.modules`` and return the client."""
    holder: dict[str, _FakeClient] = {}

    module = types.ModuleType("anthropic")

    class APIError(Exception):
        pass

    class APIStatusError(APIError):
        def __init__(self, message="", status_code=500):
            super().__init__(message)
            self.status_code = status_code

    class APIConnectionError(APIError):
        pass

    class RateLimitError(APIStatusError):
        pass

    class AuthenticationError(APIStatusError):
        pass

    def _factory(**kwargs):
        client = _FakeClient(script, **kwargs)
        holder["client"] = client
        return client

    module.Anthropic = _factory  # type: ignore[attr-defined]
    module.APIError = APIError  # type: ignore[attr-defined]
    module.APIStatusError = APIStatusError  # type: ignore[attr-defined]
    module.APIConnectionError = APIConnectionError  # type: ignore[attr-defined]
    module.RateLimitError = RateLimitError  # type: ignore[attr-defined]
    module.AuthenticationError = AuthenticationError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return module, holder


@pytest.fixture()
def dispatcher():
    robot = MockSpot(console=QUIET)
    robot.connect()
    return ToolDispatcher(robot=robot, console=QUIET)


def _make_brain(monkeypatch, script, dispatcher):
    module, holder = _install_fake_anthropic(monkeypatch, script)
    from spot_voice.brain.agent import Brain

    brain = Brain(
        api_key="test-key",
        model="claude-sonnet-4-6",
        dispatcher=dispatcher,
        console=QUIET,
    )
    return brain, module, holder


# ----------------------------------------------------------------------
# Happy paths


def test_plain_answer_needs_no_tools(monkeypatch, dispatcher):
    script = [_Response([_Block("text", text="I'm ready.")], "end_turn")]
    brain, _module, holder = _make_brain(monkeypatch, script, dispatcher)

    reply = brain.handle("are you there")

    assert reply.text == "I'm ready."
    assert reply.tool_calls == []
    assert len(holder["client"].messages.calls) == 1


def test_tool_call_round_trip(monkeypatch, dispatcher):
    script = [
        _Response(
            [_Block("tool_use", id="toolu_1", name="stand", input={})],
            "tool_use",
        ),
        _Response([_Block("text", text="Standing.")], "end_turn"),
    ]
    brain, _module, holder = _make_brain(monkeypatch, script, dispatcher)

    reply = brain.handle("stand up please")

    assert reply.text == "Standing."
    assert [call.name for call in reply.tool_calls] == ["stand"]

    # The second request must carry the tool_result back in a single user turn.
    second = holder["client"].messages.calls[1]["messages"]
    tool_turn = second[-1]
    assert tool_turn["role"] == "user"
    assert tool_turn["content"][0]["type"] == "tool_result"
    payload = json.loads(tool_turn["content"][0]["content"][0]["text"])
    assert payload["ok"] is True


def test_inspection_flow_attaches_the_photo_for_the_model_to_see(
    monkeypatch, dispatcher
):
    script = [
        _Response(
            [
                _Block(
                    "tool_use",
                    id="toolu_1",
                    name="navigate_to",
                    input={"waypoint_name": "control-panel"},
                )
            ],
            "tool_use",
        ),
        _Response(
            [_Block("tool_use", id="toolu_2", name="capture_image", input={})],
            "tool_use",
        ),
        _Response(
            [_Block("text", text="I see a control panel with six gauges.")], "end_turn"
        ),
    ]
    brain, _module, holder = _make_brain(monkeypatch, script, dispatcher)

    reply = brain.handle("go to the control panel and tell me what you see")

    assert [call.name for call in reply.tool_calls] == ["navigate_to", "capture_image"]
    assert "control panel" in reply.text

    last_turn = holder["client"].messages.calls[2]["messages"][-1]
    blocks = last_turn["content"][0]["content"]
    assert any(block["type"] == "image" for block in blocks)
    image = next(block for block in blocks if block["type"] == "image")
    assert image["source"]["media_type"] == "image/jpeg"


def test_parallel_tool_calls_come_back_in_one_user_message(monkeypatch, dispatcher):
    script = [
        _Response(
            [
                _Block("tool_use", id="a", name="get_status", input={}),
                _Block("tool_use", id="b", name="list_waypoints", input={}),
            ],
            "tool_use",
        ),
        _Response([_Block("text", text="All good.")], "end_turn"),
    ]
    brain, _module, holder = _make_brain(monkeypatch, script, dispatcher)

    brain.handle("status and places")

    tool_turn = holder["client"].messages.calls[1]["messages"][-1]
    assert len(tool_turn["content"]) == 2
    assert {block["tool_use_id"] for block in tool_turn["content"]} == {"a", "b"}


# ----------------------------------------------------------------------
# Prompt caching


def test_system_and_tools_carry_cache_breakpoints(monkeypatch, dispatcher):
    script = [_Response([_Block("text", text="hi")], "end_turn")]
    brain, _module, holder = _make_brain(monkeypatch, script, dispatcher)

    brain.handle("hello")

    call = holder["client"].messages.calls[0]
    assert call["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert call["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_system_and_tools_are_byte_identical_across_turns(monkeypatch, dispatcher):
    script = [
        _Response([_Block("text", text="one")], "end_turn"),
        _Response([_Block("text", text="two")], "end_turn"),
    ]
    brain, _module, holder = _make_brain(monkeypatch, script, dispatcher)

    brain.handle("first")
    brain.handle("second")

    calls = holder["client"].messages.calls
    # Any drift in the prefix would silently destroy the cache hit rate.
    assert calls[0]["system"] == calls[1]["system"]
    assert calls[0]["tools"] == calls[1]["tools"]


# ----------------------------------------------------------------------
# Failure paths


def test_connection_failure_yields_a_speakable_sentence(monkeypatch, dispatcher):
    module, _holder = _install_fake_anthropic(monkeypatch, [])
    from spot_voice.brain.agent import Brain

    brain = Brain("k", "claude-sonnet-4-6", dispatcher, console=QUIET)
    brain._client.messages._script = [module.APIConnectionError("no route to host")]

    reply = brain.handle("what do you see")

    assert reply.error
    assert "safety commands still work" in reply.text
    # The unanswered user turn must not linger in the history.
    assert brain._messages == []


def test_refusal_is_handled_without_touching_content(monkeypatch, dispatcher):
    script = [_Response([], "refusal")]
    brain, _module, _holder = _make_brain(monkeypatch, script, dispatcher)

    reply = brain.handle("do something unsafe")

    assert reply.text == "I can't help with that one."


def test_abort_cancels_an_in_flight_tool_loop(monkeypatch, dispatcher):
    script = [
        _Response(
            [_Block("tool_use", id="toolu_1", name="get_status", input={})], "tool_use"
        ),
        _Response([_Block("text", text="never reached")], "end_turn"),
    ]
    brain, _module, _holder = _make_brain(monkeypatch, script, dispatcher)

    original = dispatcher.dispatch

    def dispatch_then_abort(name, tool_input):
        result = original(name, tool_input)
        brain.abort()  # simulate the operator saying "stop" mid-sequence
        return result

    monkeypatch.setattr(dispatcher, "dispatch", dispatch_then_abort)

    reply = brain.handle("what's your status")

    assert reply.aborted is True
    assert reply.text == ""


def test_the_loop_cannot_run_away(monkeypatch, dispatcher):
    from spot_voice.brain.agent import MAX_TOOL_ITERATIONS

    script = [
        _Response(
            [_Block("tool_use", id=f"t{index}", name="get_status", input={})],
            "tool_use",
        )
        for index in range(MAX_TOOL_ITERATIONS + 4)
    ]
    brain, _module, holder = _make_brain(monkeypatch, script, dispatcher)

    reply = brain.handle("loop forever")

    assert len(holder["client"].messages.calls) == MAX_TOOL_ITERATIONS
    assert reply.text
