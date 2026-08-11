"""The provider translation layer.

The Groq adapter converts between Anthropic-shaped messages and the OpenAI chat
dialect. That translation is where a swap between providers silently breaks --
particularly tool plumbing, which Anthropic carries as blocks inside a user
message and OpenAI carries as separate ``role: "tool"`` messages. These are pure
functions, so they are testable without a key.
"""

from __future__ import annotations

import json
import types

import pytest

from spot_voice.brain.providers.base import (
    TextBlock,
    ToolUseBlock,
    block_field,
    block_type,
    strip_cache_control,
)
from spot_voice.brain.providers.groq_provider import (
    from_openai_response,
    to_openai_messages,
    to_openai_tools,
)
from spot_voice.brain.tools import TOOLS, tools_with_cache_breakpoint


# ----------------------------------------------------------------------
# Block helpers


def test_block_helpers_read_dicts_and_objects_alike():
    assert block_type({"type": "text"}) == "text"
    assert block_type(TextBlock(text="hi")) == "text"
    assert block_field({"text": "hi"}, "text") == "hi"
    assert block_field(TextBlock(text="hi"), "text") == "hi"
    assert block_field({}, "missing", "fallback") == "fallback"


def test_cache_control_is_stripped_without_mutating_the_original():
    tools = tools_with_cache_breakpoint()
    cleaned = strip_cache_control(tools)
    assert all("cache_control" not in tool for tool in cleaned)
    # Providers that do support caching must still see the breakpoint.
    assert "cache_control" in tools[-1]


# ----------------------------------------------------------------------
# Tool schemas


def test_every_tool_survives_translation_to_openai_shape():
    converted = to_openai_tools(tools_with_cache_breakpoint())
    assert len(converted) == len(TOOLS)
    for tool, original in zip(converted, TOOLS):
        assert tool["type"] == "function"
        assert tool["function"]["name"] == original["name"]
        assert tool["function"]["description"] == original["description"]
        assert tool["function"]["parameters"] == original["input_schema"]
        json.dumps(tool)  # must be serialisable for the wire


def test_translated_tools_carry_no_anthropic_only_keys():
    blob = json.dumps(to_openai_tools(tools_with_cache_breakpoint()))
    assert "cache_control" not in blob
    assert "input_schema" not in blob


# ----------------------------------------------------------------------
# Messages


def test_system_blocks_collapse_into_one_system_message():
    messages = to_openai_messages(
        [{"type": "text", "text": "You are Spot."}, {"type": "text", "text": "Places: bay."}],
        [{"role": "user", "content": "hello"}],
    )
    assert messages[0]["role"] == "system"
    assert "You are Spot." in messages[0]["content"]
    assert "Places: bay." in messages[0]["content"]


def test_a_plain_user_turn_survives():
    messages = to_openai_messages([], [{"role": "user", "content": "stand up"}])
    assert messages == [{"role": "user", "content": "stand up"}]


def test_assistant_tool_calls_become_openai_tool_calls():
    history = [
        {"role": "user", "content": "stand up"},
        {
            "role": "assistant",
            "content": [
                TextBlock(text="Standing."),
                ToolUseBlock(id="t1", name="stand", input={}),
            ],
        },
    ]
    messages = to_openai_messages([], history)

    assistant = messages[-1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Standing."
    assert assistant["tool_calls"][0]["id"] == "t1"
    assert assistant["tool_calls"][0]["function"]["name"] == "stand"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {}


def test_tool_results_become_separate_tool_role_messages():
    # This is the shape difference that breaks a naive port: Anthropic nests
    # results inside a user message, OpenAI wants one message each.
    history = [
        {"role": "user", "content": "status"},
        {
            "role": "assistant",
            "content": [ToolUseBlock(id="t1", name="get_status", input={})],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": '{"ok": true}'}],
                }
            ],
        },
    ]
    messages = to_openai_messages([], history)

    tool_message = messages[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "t1"
    assert json.loads(tool_message["content"])["ok"] is True


def test_parallel_tool_results_become_one_message_each():
    history = [
        {"role": "user", "content": "status and places"},
        {
            "role": "assistant",
            "content": [
                ToolUseBlock(id="a", name="get_status", input={}),
                ToolUseBlock(id="b", name="list_waypoints", input={}),
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": "1"}]},
                {"type": "tool_result", "tool_use_id": "b", "content": [{"type": "text", "text": "2"}]},
            ],
        },
    ]
    messages = to_openai_messages([], history)
    tool_messages = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]


def test_an_image_block_never_leaks_to_a_text_only_provider():
    # The agent should have replaced it already; this is the second line of
    # defence, because sending an image block to Groq is a hard API error.
    history = [
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": [ToolUseBlock(id="t1", name="capture_image", input={})]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [
                        {"type": "text", "text": '{"ok": true}'},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"},
                        },
                    ],
                }
            ],
        },
    ]
    messages = to_openai_messages([], history)
    blob = json.dumps(messages)
    assert "AAAA" not in blob  # the base64 payload is gone
    assert '"type": "image"' not in blob  # and so is the block itself
    assert "base64" not in blob
    # The text half of the same tool result must survive.
    assert json.loads(messages[-1]["content"])["ok"] is True


def test_the_whole_translation_is_json_serialisable():
    history = [
        {"role": "user", "content": "go to the bay and look"},
        {
            "role": "assistant",
            "content": [
                TextBlock(text="On my way."),
                ToolUseBlock(id="t1", name="navigate_to", input={"waypoint_name": "bay"}),
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "{}"}]}
            ],
        },
    ]
    json.dumps(to_openai_messages([{"type": "text", "text": "sys"}], history))


# ----------------------------------------------------------------------
# Responses


def _fake_completion(content=None, tool_calls=None, finish_reason="stop"):
    """Build an object shaped like an OpenAI chat completion."""
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = types.SimpleNamespace(prompt_tokens=100, completion_tokens=20)
    return types.SimpleNamespace(choices=[choice], usage=usage)


def _fake_tool_call(call_id, name, arguments):
    return types.SimpleNamespace(
        id=call_id, function=types.SimpleNamespace(name=name, arguments=arguments)
    )


def test_a_plain_answer_comes_back_as_text():
    response = from_openai_response(_fake_completion(content="I'm ready."))
    assert response.text == "I'm ready."
    assert response.stop_reason == "end_turn"
    assert response.tool_calls == []


def test_tool_calls_are_parsed_into_canonical_form():
    completion = _fake_completion(
        tool_calls=[_fake_tool_call("t1", "move", '{"direction": "forward"}')],
        finish_reason="tool_calls",
    )
    response = from_openai_response(completion)

    assert response.stop_reason == "tool_use"
    assert response.tool_calls[0].name == "move"
    assert response.tool_calls[0].input == {"direction": "forward"}
    # The same call must also appear in content, so history replays correctly.
    assert any(block_type(b) == "tool_use" for b in response.content)


def test_malformed_tool_arguments_do_not_raise():
    # A weaker model emitting broken JSON must not take the robot session down;
    # an empty input reaches the dispatcher, which refuses it speakably.
    completion = _fake_completion(
        tool_calls=[_fake_tool_call("t1", "move", "{not json")], finish_reason="tool_calls"
    )
    response = from_openai_response(completion)
    assert response.tool_calls[0].input == {}


def test_non_object_tool_arguments_are_coerced():
    completion = _fake_completion(
        tool_calls=[_fake_tool_call("t1", "stand", "[1, 2]")], finish_reason="tool_calls"
    )
    assert from_openai_response(completion).tool_calls[0].input == {}


def test_tool_calls_with_a_stop_finish_reason_are_still_tool_use():
    # Some models report finish_reason "stop" alongside tool calls; treating
    # that as end_turn would silently drop the call.
    completion = _fake_completion(
        tool_calls=[_fake_tool_call("t1", "sit", "{}")], finish_reason="stop"
    )
    assert from_openai_response(completion).stop_reason == "tool_use"


@pytest.mark.parametrize(
    "finish,expected",
    [("length", "max_tokens"), ("content_filter", "refusal"), ("stop", "end_turn")],
)
def test_finish_reasons_map_to_canonical_stop_reasons(finish, expected):
    assert from_openai_response(_fake_completion(content="x", finish_reason=finish)).stop_reason == expected


# ----------------------------------------------------------------------
# Factory


def test_the_factory_reports_a_missing_key_rather_than_crashing():
    from spot_voice.brain.providers import ProviderUnavailable, build_llm_provider

    config = types.SimpleNamespace(
        llm_provider="groq", groq_api_key="", groq_model="m",
        anthropic_api_key="", anthropic_model="m",
    )
    with pytest.raises(ProviderUnavailable, match="GROQ_API_KEY"):
        build_llm_provider(config)


def test_an_unknown_provider_is_rejected():
    from spot_voice.brain.providers import ProviderUnavailable, build_llm_provider

    config = types.SimpleNamespace(
        llm_provider="openai", groq_api_key="k", groq_model="m",
        anthropic_api_key="k", anthropic_model="m",
    )
    with pytest.raises(ProviderUnavailable):
        build_llm_provider(config)


def test_vision_falls_back_to_a_null_provider_rather_than_failing():
    from spot_voice.brain.providers import build_vision_provider

    config = types.SimpleNamespace(
        vision_provider="gemini", gemini_api_key="", gemini_model="m"
    )
    vision = build_vision_provider(config)
    assert vision is not None
    described = vision.describe(b"\xff\xd8", "")
    assert "no vision model" in described


def test_anthropic_vision_means_nothing_extra_to_build():
    from spot_voice.brain.providers import build_vision_provider

    config = types.SimpleNamespace(
        vision_provider="anthropic", gemini_api_key="", gemini_model="m"
    )
    assert build_vision_provider(config) is None
