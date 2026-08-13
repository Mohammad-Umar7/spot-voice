"""Tool definitions handed to Claude (Anthropic Messages API tool use).

This is the complete surface Claude can act through. There is deliberately no
tool for releasing the e-stop, changing obstacle-avoidance parameters, or
otherwise weakening a robot safety system -- those are not exposed, by design.

Every tool returns ``{"ok": bool, "message": str}`` (plus extra fields where
useful). Tools never raise into the model; see
:mod:`spot_voice.brain.dispatcher`.
"""

from __future__ import annotations

from typing import Any

from ..robot.emotes import available as available_gestures
from ..robot.limits import (
    MAX_MOVE_DEGREES,
    MAX_MOVE_DISTANCE_M,
    MAX_VROT,
    MAX_VX,
)

#: Directions accepted by the ``move`` tool.
MOVE_DIRECTIONS = ["forward", "back", "left", "right", "turn_left", "turn_right"]

#: Cameras accepted by the ``capture_image`` tool.
CAMERAS = ["front", "left", "right"]

#: Body-language gestures the ``emote`` tool can perform.
GESTURES = available_gestures()


TOOLS: list[dict[str, Any]] = [
    {
        "name": "power_on",
        "description": (
            "Turn the motors on without changing posture. You rarely need this "
            "on its own -- stand, move, navigate_to, dock and undock all power "
            "the motors themselves. Use it when the operator explicitly asks to "
            "power on, or to check whether powering on is what is failing."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "stand",
        "description": (
            "Power on the motors if needed and stand up. Use before any movement. "
            "Fails if Spot is on its dock -- undock first."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "sit",
        "description": "Sit down. Use when the operator wants Spot to rest or park in place.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "move",
        "description": (
            "Walk a short, bounded distance in one direction, or turn in place. "
            "Use for small local adjustments the operator describes directly "
            "('come forward a metre', 'turn left'). For travelling somewhere in "
            "the facility, use navigate_to instead. "
            f"Distance is capped at {MAX_MOVE_DISTANCE_M} m and turns at "
            f"{MAX_MOVE_DEGREES:.0f} degrees; speed is capped in the robot layer "
            "regardless of what you request."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": MOVE_DIRECTIONS,
                    "description": (
                        "Direction of travel. 'left'/'right' strafe sideways; "
                        "'turn_left'/'turn_right' rotate in place."
                    ),
                },
                "distance_m": {
                    "type": "number",
                    "description": (
                        "Distance in metres for forward/back/left/right. "
                        f"Defaults to 1.0 and is capped at {MAX_MOVE_DISTANCE_M}."
                    ),
                },
                "degrees": {
                    "type": "number",
                    "description": (
                        "Rotation in degrees for turn_left/turn_right. Defaults to 90."
                    ),
                },
                "speed": {
                    "type": "number",
                    "description": (
                        "Optional speed: m/s for linear moves "
                        f"(hard cap {MAX_VX}), rad/s for turns (hard cap {MAX_VROT})."
                    ),
                },
            },
            "required": ["direction"],
        },
    },
    {
        "name": "go_where_pointed",
        "description": (
            "Look for the operator pointing, work out the direction, measure how "
            "far the floor is clear that way, and walk there. Use this only when "
            "they mean somewhere AWAY from themselves and are gesturing at it: "
            "'stand over there', 'go that way', 'wait by the door'. Requires an "
            "outstretched arm; if nobody is pointing it says so. For anywhere "
            "near the operator themselves -- 'come here', 'sit beside me' -- use "
            "come_here instead, which needs no gesture."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "come_here",
        "description": (
            "Walk to the operator and settle next to them. Use for 'come here', "
            "'sit here', 'stand beside me', 'come to me' -- anything naming a "
            "spot at the person rather than away from them. No pointing gesture "
            "is needed: Spot finds the operator with its camera, so their arms "
            "can be full. Spot says what it understood before it moves."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "string",
                    "enum": ["in_front", "beside"],
                    "description": (
                        "'beside' for 'beside me' or 'next to me' -- ends up "
                        "alongside them. 'in_front' for a plain 'come here' -- "
                        "ends up facing them. Defaults to in_front."
                    ),
                },
                "posture": {
                    "type": "string",
                    "enum": ["stand", "sit"],
                    "description": (
                        "What to do on arrival. 'sit' for 'sit here' or 'sit "
                        "beside me'. Defaults to stand."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "navigate_to",
        "description": (
            "Autonomously walk to a named waypoint on the facility map using "
            "GraphNav. Spot must be able to see a fiducial marker to localize "
            "the first time. Call list_waypoints if you are unsure of the name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "waypoint_name": {
                    "type": "string",
                    "description": "Name of the waypoint as recorded on the map.",
                }
            },
            "required": ["waypoint_name"],
        },
    },
    {
        "name": "list_waypoints",
        "description": (
            "List the named places on the facility map. Call this when the "
            "operator asks where Spot can go, or when a requested place name "
            "was not recognised."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "start_follow",
        "description": (
            "Start follow-me. If the operator's face is enrolled, Spot turns "
            "slowly on the spot to look for them, locks on once it recognises "
            "them, and then follows that same person at roughly 1.5 m using how "
            "they look from behind -- so it keeps them even when it can only see "
            "their back, and will not switch to someone who walks past. Without "
            "an enrolled face it locks onto whoever is standing in front of it, "
            "so it is worth telling the operator to stand in front of you. "
            "Spot's own obstacle avoidance remains active. Stops on stop_follow, "
            "the spoken word 'stop', or after the person is out of sight for two "
            "seconds ('I lost you'), after which it looks for them again."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "stop_follow",
        "description": (
            "Stop follow-me and hold position. Call this when the operator says "
            "they are done walking, or asks you to wait somewhere while they "
            "carry on without you."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "capture_image",
        "description": (
            "Take one photo from a body camera and return it so you can see it. "
            "Use this for inspection requests -- after navigating somewhere, "
            "capture an image and describe what is actually visible. Do not "
            "guess at contents you have not seen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "enum": CAMERAS,
                    "description": "Which camera to use. Defaults to front.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "scan_room",
        "description": (
            "Turn all the way round, look at each heading, and report what is "
            "there. Use for 'scan the room', 'how many people are here', "
            "'what's around you', 'look around'. One camera frame only covers "
            "about a hundred degrees, so this is the ONLY way to answer a "
            "question about the whole room -- a single capture_image cannot, "
            "and you must never answer such a question without calling this. "
            "Takes around twenty seconds because the robot physically turns. "
            "People counts come from a detector, and headings overlap, so "
            "report the number as 'at least N that I could see', never as an "
            "exact count of the room."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_status",
        "description": (
            "Report battery percentage, motor power, lease and e-stop state, "
            "localization and dock state."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "dock",
        "description": "Walk onto the charging dock and power down onto it.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "undock",
        "description": (
            "Power on and step off the charging dock into a standing pose. "
            "Needed before any movement when Spot starts the session docked. "
            "If Spot is not on the dock this reports that and does nothing -- "
            "call stand instead to get it on its feet."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "emote",
        "description": (
            "Perform a short body-language gesture. Spot has no arm, so these "
            "are body movements: a bow, a nod, a head tilt, a wiggle. Use one "
            "when the operator asks you to greet people, acknowledge something, "
            "agree or disagree, or when a gesture would read better than words. "
            "'greet' is the one for saying hello to a room. Gestures only work "
            "while standing and take a couple of seconds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gesture": {
                    "type": "string",
                    "enum": GESTURES,
                    "description": (
                        "Which gesture to perform. Pick the closest fit rather "
                        "than refusing."
                    ),
                }
            },
            "required": ["gesture"],
        },
    },
    {
        "name": "speak",
        "description": (
            "Almost never needed. Everything you write as your reply is "
            "already spoken aloud automatically, so calling this to say "
            "something is saying it twice and makes the robot slow and "
            "repetitive. Do not call it to acknowledge a command, report a "
            "result, or confirm a status -- just write that in your reply "
            "instead. The single valid use is a progress note partway through a "
            "long walk that has not finished yet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "One or two sentences to speak aloud.",
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "stop_all",
        "description": (
            "Cancel everything in progress -- follow-me, navigation, any motion -- "
            "and settle into a safe stop. Use for any request to stop, wait or "
            "hold. This does not cut motor power; the physical e-stop on the "
            "tablet remains the ultimate authority."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


#: Names of every tool Claude may call.
TOOL_NAMES: frozenset[str] = frozenset(tool["name"] for tool in TOOLS)


def tools_with_cache_breakpoint() -> list[dict[str, Any]]:
    """Return the tool list with a prompt-cache breakpoint on the last entry.

    Tools render before ``system`` in the prompt, so caching here plus a
    breakpoint on the system prompt keeps the whole fixed prefix warm across the
    many short turns a voice session produces.
    """
    cached = [dict(tool) for tool in TOOLS]
    cached[-1]["cache_control"] = {"type": "ephemeral"}
    return cached


# ----------------------------------------------------------------------
# Compact tool schemas, for providers without prompt caching
# ----------------------------------------------------------------------
#
# The descriptions above are written for Anthropic, where the whole prefix is
# cached and costs almost nothing after the first turn -- so detail is free and
# improves tool choice. On Groq there is no caching: all 16 schemas are re-sent
# every request, ~1850 tokens against a 12,000/minute limit, which is four
# requests a minute. One spoken sentence can take three. That is what made the
# robot feel broken rather than slow.
#
# These say the same things in a fraction of the words: when to call it, and the
# one constraint that matters. Nothing safety-relevant lives here anyway -- the
# caps are enforced in the robot layer whatever the model asks for.

COMPACT_DESCRIPTIONS: dict[str, str] = {
    "power_on": "Motors on, posture unchanged. Rarely needed alone.",
    "stand": "Stand up, powering motors if needed. Needed before moving.",
    "sit": "Sit down.",
    "emote": "Body-language gesture: " + " | ".join(GESTURES) + ". Use for greetings and reactions.",
    "move": (
        f"Walk a short distance or turn in place. direction is one of "
        f"{'/'.join(MOVE_DIRECTIONS)}. distance_m for straight moves (max "
        f"{MAX_MOVE_DISTANCE_M}), degrees for turns. For travelling somewhere "
        "named, use navigate_to."
    ),
    "go_where_pointed": (
        "Walk where the operator points. Only for somewhere away from them: "
        "'over there', 'that way'. Needs an outstretched arm."
    ),
    "come_here": (
        "Walk to the operator and stop by them. For 'come here', 'sit here', "
        "'stand beside me'. No gesture needed."
    ),
    "navigate_to": "Walk to a named map waypoint. Use list_waypoints if unsure of the name.",
    "list_waypoints": "Names of places on the map.",
    "start_follow": "Start following the operator. Tell them to stand in front of you.",
    "stop_follow": "Stop following and hold position.",
    "capture_image": "Take a photo and see it. Use for 'what do you see'. Describe only what is there.",
    "get_status": "Battery, motor power, e-stop, dock and localization.",
    "dock": "Return to the charging dock.",
    "undock": "Step off the dock. If not docked, use stand instead.",
    "speak": "Almost never needed -- your reply is already spoken aloud. Only for a progress note mid-walk.",
    "stop_all": "Cancel everything and stop safely.",
}


def compact_tools() -> list[dict[str, Any]]:
    """The same tools with terse descriptions, for rate-limited providers."""
    trimmed: list[dict[str, Any]] = []
    for tool in TOOLS:
        copy = dict(tool)
        description = COMPACT_DESCRIPTIONS.get(tool["name"])
        if description:
            copy["description"] = description
        copy["input_schema"] = _compact_schema(tool["input_schema"])
        trimmed.append(copy)
    return trimmed


def _compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop per-parameter prose, keeping types, enums and what is required.

    The enum values are what actually constrain the model; the sentences
    explaining them are what cost the tokens.
    """
    properties = {}
    for name, prop in schema.get("properties", {}).items():
        slim = {key: value for key, value in prop.items() if key != "description"}
        properties[name] = slim
    return {
        "type": schema.get("type", "object"),
        "properties": properties,
        "required": schema.get("required", []),
    }


def tools_for(provider) -> list[dict[str, Any]]:
    """Pick the right tool schemas for a provider.

    Caching providers get the detailed descriptions, because they are cached and
    better descriptions mean better tool choice. Everyone else gets the compact
    ones, because there the same detail is a per-request tax.
    """
    if getattr(provider, "supports_prompt_caching", False):
        return tools_with_cache_breakpoint()
    return compact_tools()
