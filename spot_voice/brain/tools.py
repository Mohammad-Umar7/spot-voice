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


TOOLS: list[dict[str, Any]] = [
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
            "Start follow-me. Spot tracks the nearest, most centred person with "
            "its front camera and walks along behind them at roughly 1.5 m. "
            "Spot's own obstacle avoidance remains active. Stops on stop_follow, "
            "on the spoken word 'stop', or when the person is out of sight for "
            "two seconds."
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
            "Needed before any movement when Spot starts the session docked."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "speak",
        "description": (
            "Say something out loud immediately, mid-task. Use only when the "
            "operator needs to hear something before the task finishes, such as "
            "a progress note during a long walk. Your normal reply is already "
            "spoken, so do not use this to repeat it."
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
