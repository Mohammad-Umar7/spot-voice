"""The mock and the real client must stay interchangeable.

Everything above the robot layer -- reflex lane, dispatcher, follow-me -- is
written against one interface. If the two implementations drift, mock mode stops
predicting what the real robot will do, which is the whole point of it.

The SDK-symbol checks are skipped when ``bosdyn`` is not installed, so this file
still runs on a laptop set up for mock mode only.
"""

from __future__ import annotations

import inspect

import pytest

from spot_voice.robot.base import CAMERA_SOURCES, RobotInterface
from spot_voice.robot.mock import MockSpot

#: Methods every implementation must provide.
INTERFACE_METHODS = [
    name
    for name, value in vars(RobotInterface).items()
    if getattr(value, "__isabstractmethod__", False)
]


def test_the_interface_is_not_empty():
    assert len(INTERFACE_METHODS) >= 12


def test_mock_implements_the_whole_interface():
    spot = MockSpot()  # would raise TypeError if anything were unimplemented
    for name in INTERFACE_METHODS:
        assert hasattr(spot, name), name


def test_mock_exposes_no_safety_override():
    forbidden = ("allow", "release_estop", "estop_allow", "obstacle", "padding")
    public = [name for name in dir(MockSpot) if not name.startswith("_")]
    for name in public:
        assert not any(word in name.lower() for word in forbidden), name


def test_camera_names_are_the_ones_the_tool_schema_offers():
    from spot_voice.brain.tools import CAMERAS

    assert set(CAMERAS) == set(CAMERA_SOURCES)


def test_front_cameras_are_rotated_upright():
    # Spot's front fisheye cameras are mounted sideways; a zero rotation here
    # would mean Claude describes a sideways scene.
    assert CAMERA_SOURCES["front"][1] != 0


# ----------------------------------------------------------------------
# Real client -- only runs where the Spot SDK is installed.

bosdyn = pytest.importorskip("bosdyn.client", reason="Spot SDK not installed")


@pytest.fixture(scope="module")
def spot_client_class():
    from spot_voice.robot.spot_client import SpotClient

    return SpotClient


def test_real_client_implements_the_whole_interface(spot_client_class):
    # Instantiating does not talk to a robot; connect() does.
    client = spot_client_class(ip="192.0.2.1")
    assert isinstance(client, RobotInterface)
    for name in INTERFACE_METHODS:
        assert hasattr(client, name), name


def _callable_for(klass, name):
    """Return the underlying function, unwrapping properties."""
    attribute = inspect.getattr_static(klass, name)
    return attribute.fget if isinstance(attribute, property) else attribute


@pytest.mark.parametrize("name", INTERFACE_METHODS)
def test_mock_and_real_signatures_match(spot_client_class, name):
    mock_signature = inspect.signature(_callable_for(MockSpot, name))
    real_signature = inspect.signature(_callable_for(spot_client_class, name))
    assert mock_signature == real_signature, name


@pytest.mark.parametrize("name", INTERFACE_METHODS)
def test_properties_stay_properties_in_both(spot_client_class, name):
    is_property_on_interface = isinstance(
        inspect.getattr_static(RobotInterface, name), property
    )
    for klass in (MockSpot, spot_client_class):
        assert (
            isinstance(inspect.getattr_static(klass, name), property)
            is is_property_on_interface
        ), f"{klass.__name__}.{name}"


def test_every_sdk_symbol_the_real_client_uses_resolves():
    """Catch a rename in the SDK before it turns into a failure on the robot."""
    import importlib

    expected = {
        "bosdyn.client": ["create_standard_sdk"],
        "bosdyn.client.util": ["authenticate"],
        "bosdyn.client.spot_cam": ["register_all_service_clients"],
        "bosdyn.client.estop": [
            "EstopClient",
            "EstopEndpoint",
            "EstopKeepAlive",
            "is_estopped",
        ],
        "bosdyn.client.image": ["ImageClient"],
        "bosdyn.client.lease": [
            "LeaseClient",
            "LeaseKeepAlive",
            "ResourceAlreadyClaimedError",
        ],
        "bosdyn.client.power": ["PowerClient", "power_on"],
        "bosdyn.client.robot_command": [
            "RobotCommandClient",
            "RobotCommandBuilder",
            "blocking_stand",
        ],
        "bosdyn.client.robot_state": ["RobotStateClient"],
        "bosdyn.client.docking": [
            "blocking_dock_robot",
            "blocking_go_to_prep_pose",
            "get_dock_id",
        ],
        "bosdyn.client.graph_nav": ["GraphNavClient"],
        "bosdyn.client.frame_helpers": ["get_odom_tform_body"],
        "bosdyn.client.spot_cam.audio": ["AudioClient"],
        "bosdyn.api": ["robot_state_pb2", "image_pb2"],
        "bosdyn.api.spot_cam": ["audio_pb2"],
        "bosdyn.api.graph_nav": ["map_pb2", "nav_pb2", "graph_nav_pb2"],
    }
    missing = []
    for module_name, names in expected.items():
        module = importlib.import_module(module_name)
        missing += [f"{module_name}.{n}" for n in names if not hasattr(module, n)]
    assert not missing, missing


def test_command_builder_has_the_three_commands_we_issue():
    from bosdyn.client.robot_command import RobotCommandBuilder

    for name in ("synchro_velocity_command", "synchro_sit_command", "stop_command"):
        assert hasattr(RobotCommandBuilder, name), name


def test_navigation_feedback_statuses_we_branch_on_exist():
    from bosdyn.api.graph_nav import graph_nav_pb2

    response = graph_nav_pb2.NavigationFeedbackResponse
    for name in (
        "STATUS_REACHED_GOAL",
        "STATUS_LOST",
        "STATUS_STUCK",
        "STATUS_ROBOT_IMPAIRED",
    ):
        assert hasattr(response, name), name


def test_image_rotation_expands_the_canvas():
    import numpy as np

    from spot_voice.robot.spot_client import rotate_image

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert rotate_image(frame, 0).shape == frame.shape
    assert rotate_image(frame, 180).shape == frame.shape
    assert rotate_image(frame, 90).shape[:2] == (640, 480)
    # An arbitrary angle must not crop the image.
    rotated = rotate_image(frame, -78)
    assert rotated.shape[0] >= 480 and rotated.shape[1] >= 480


def test_graph_name_index_drops_ambiguous_names():
    from bosdyn.api.graph_nav import map_pb2

    from spot_voice.robot.graphnav import build_name_index

    graph = map_pb2.Graph()
    for waypoint_id, name in (
        ("id-1", "entrance"),
        ("id-2", "loading bay"),
        ("id-3", "loading bay"),  # duplicate name: unusable for navigation
        ("id-4", ""),  # unnamed
    ):
        waypoint = graph.waypoints.add()
        waypoint.id = waypoint_id
        waypoint.annotations.name = name

    index = build_name_index(graph)
    assert index == {"entrance": "id-1"}
