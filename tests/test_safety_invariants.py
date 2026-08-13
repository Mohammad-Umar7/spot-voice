"""Safety invariants checked against the source itself.

These are not behaviour tests. They read the project's own code and fail the
build if a guarantee that was promised in the README stops being true. That
matters for a robot: the promises are easy to make and easy to erode six commits
later, and nobody notices until it is standing in a facility.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "spot_voice"

#: Every field of ``bosdyn.api.spot.MobilityParams.ObstacleParams``. Setting any
#: of these weakens or disables Spot's own obstacle avoidance. This project must
#: never reference one.
OBSTACLE_PARAM_FIELDS = (
    "disable_vision_foot_obstacle_avoidance",
    "disable_vision_foot_constraint_avoidance",
    "disable_vision_body_obstacle_avoidance",
    "obstacle_avoidance_padding",
    "disable_vision_foot_obstacle_body_assist",
    "disable_vision_negative_obstacles",
    "obstacle_params",
)

#: Other mobility settings that trade away a safety behaviour.
OTHER_SAFETY_FIELDS = (
    "allow_degraded_perception",
    "disable_nearmap_cliff_avoidance",
    "disable_missing_data_cliffs",
    "disallow_stair_tracker",
    "disable_stair_error_auto_descent",
)


def source_files() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def test_there_is_source_to_check():
    assert len(source_files()) > 15


@pytest.mark.parametrize("field", OBSTACLE_PARAM_FIELDS)
def test_obstacle_avoidance_is_never_touched(field):
    """Spot's obstacle avoidance stays at factory defaults, permanently.

    The way to weaken it is to set a field on MobilityParams.obstacle_params.
    The way to guarantee we never do is to never name one.
    """
    offenders = [
        f"{path.relative_to(PACKAGE)}:{number}"
        for path in source_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        # Prose in docstrings and comments is fine -- code is not.
        if field in line and not line.lstrip().startswith(("#", '"', "'"))
    ]
    assert not offenders, f"{field} referenced at: {offenders}"


@pytest.mark.parametrize("field", OTHER_SAFETY_FIELDS)
def test_no_other_perception_safety_is_traded_away(field):
    offenders = [
        f"{path.relative_to(PACKAGE)}:{number}"
        for path in source_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if field in line and not line.lstrip().startswith(("#", '"', "'"))
    ]
    assert not offenders, f"{field} referenced at: {offenders}"


def test_mobility_params_are_never_constructed_directly():
    """Building the protobuf by hand would bypass every check below it."""
    offenders = [
        f"{path.relative_to(PACKAGE)}:{number}"
        for path in source_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "MobilityParams(" in line
    ]
    assert not offenders, f"MobilityParams constructed at: {offenders}"


def test_mobility_params_are_built_in_exactly_one_place():
    """Params may be built once, for speed capping, and nowhere else.

    This rule used to be "never build params at all", which was simpler and was
    true while every command was a velocity command -- those carry their own
    speed, so clamping the velocity enforced the caps.

    A trajectory command does not carry a speed. Spot chooses its own to reach
    the goal, and its default is faster than MAX_VX, so a trajectory issued with
    default params would silently exceed the velocity ceiling. The ceiling has
    to be handed to the planner, and that means a params object.

    So the guarantee moves rather than weakens: exactly one construction site,
    which the next test pins to speed limits only.
    """
    sites = [
        f"{path.relative_to(PACKAGE)}:{number}"
        for path in source_files()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "mobility_params(" in line and not line.lstrip().startswith(("#", '"', "'"))
    ]
    assert len(sites) == 1, f"expected one mobility_params call site, found: {sites}"
    assert sites[0].startswith("robot\\spot_client.py") or sites[0].startswith(
        "robot/spot_client.py"
    ), f"mobility params built outside the robot layer: {sites[0]}"


def test_the_only_mobility_params_set_speed_and_nothing_else():
    """Defaults come from the SDK; we override exactly one field.

    Every setting that trades away a safety behaviour is a field on this same
    message, so the thing to pin is which fields the code touches at all. The
    builder must be called with no arguments -- so obstacle avoidance, stair
    handling and the rest keep whatever Boston Dynamics ships -- and ``vel_limit``
    must be the only field written afterwards.
    """
    import ast
    import inspect

    from spot_voice.robot.spot_client import SpotClient

    tree = ast.parse(inspect.getsource(SpotClient._speed_capped_params).strip())

    builder_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "mobility_params"
    ]
    assert len(builder_calls) == 1
    assert not builder_calls[0].args and not builder_calls[0].keywords, (
        "mobility_params must be called with no arguments, so every safety "
        "setting keeps its SDK default"
    )

    # Any field written on the params object, however it is written.
    touched = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "params"
    }
    assert touched == {"vel_limit"}, f"params fields touched: {sorted(touched)}"


def test_the_speed_cap_handed_to_the_planner_is_the_hard_cap():
    """A trajectory must not be allowed to outrun the velocity ceiling.

    Reads the numbers actually placed in the limit rather than trusting that the
    constants were referenced, because referencing MAX_VX and then halving it --
    or doubling it -- would pass any check that only looked for the name.
    """
    pytest.importorskip("bosdyn.api")
    from spot_voice.robot.limits import MAX_VROT, MAX_VX, MAX_VY
    from spot_voice.robot.spot_client import SpotClient

    params = SpotClient._speed_capped_params(object.__new__(SpotClient))
    fastest = params.vel_limit.max_vel
    slowest = params.vel_limit.min_vel

    assert fastest.linear.x == MAX_VX and slowest.linear.x == -MAX_VX
    assert fastest.linear.y == MAX_VY and slowest.linear.y == -MAX_VY
    assert fastest.angular == MAX_VROT and slowest.angular == -MAX_VROT


def test_a_lapsed_trajectory_goal_cannot_carry_the_robot_far():
    """The trajectory dead-man is looser than the velocity one; bound it anyway.

    A goal outlives a velocity command because a planner needs room to work, so
    the guarantee is stated as a distance rather than a time: if this process
    dies mid-follow, the robot coasts less than a metre before stopping.
    """
    from spot_voice.robot.limits import MAX_VX, TRAJECTORY_CMD_DURATION

    assert TRAJECTORY_CMD_DURATION * MAX_VX < 1.0


def test_estop_release_is_never_reachable_from_a_tool():
    """Releasing the e-stop is CLI-only, never something the model can call."""
    from spot_voice.brain.dispatcher import ToolDispatcher
    from spot_voice.robot.mock import MockSpot

    dispatcher = ToolDispatcher(robot=MockSpot())
    for name in dispatcher.handled_tools:
        assert "estop" not in name
        assert "allow" not in name

    # And the robot interface itself exposes no release method.
    from spot_voice.robot.base import RobotInterface

    for attribute in dir(RobotInterface):
        assert attribute not in {"allow", "estop_allow", "release_estop"}


def test_actual_face_recognition_is_confined_to_follow_me():
    """Only follow-me ever *recognises* a face.

    Undocking, standing, walking and everything else must work with nobody
    enrolled and no camera view of anyone's face. The guarantee is about
    ``FaceRecogniser`` -- the thing that runs a model over a frame. Merely
    reading the enrollment list (``FaceStore``) is reporting, not recognising,
    and is covered separately below.
    """
    allowed = {
        "robot/follow.py",  # the follow thread, which is the point
        "vision/faces.py",  # the implementation
        "enroll.py",  # the one-time enrollment command
        "main.py",  # wiring only
    }
    for path in source_files():
        relative = path.relative_to(PACKAGE).as_posix()
        if relative in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        assert "FaceRecogniser" not in text, f"{relative} recognises faces"


def test_reading_the_enrollment_list_stays_read_only():
    """``FaceStore`` may be read for reporting, but never outside these files."""
    allowed = {
        "robot/follow.py",
        "vision/faces.py",
        "vision/identity.py",
        "enroll.py",
        "main.py",
        "config.py",
        "preflight.py",  # --check reports who is enrolled
    }
    for path in source_files():
        relative = path.relative_to(PACKAGE).as_posix()
        if relative in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        assert "FaceStore" not in text, f"{relative} touches the face store"


def test_the_dispatcher_never_touches_faces():
    """No tool but start_follow can trigger face recognition."""
    dispatcher_source = (PACKAGE / "brain" / "dispatcher.py").read_text(encoding="utf-8")
    assert "face" not in dispatcher_source.lower().replace("interface", "")


def test_velocity_commands_always_go_through_the_clamp():
    """Every synchro_velocity_command in the project uses clamped values."""
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        if "synchro_velocity_command" not in text:
            continue
        # The only place that builds one must clamp immediately beforehand.
        assert "clamp_velocity" in text, f"{path.name} builds velocity without clamping"


def test_every_velocity_command_carries_an_expiry():
    """The dead-man's switch: no velocity command without an end time."""
    client = (PACKAGE / "robot" / "spot_client.py").read_text(encoding="utf-8")
    velocity_calls = client.count("synchro_velocity_command")
    expiry_uses = client.count("end_time_secs")
    assert velocity_calls > 0
    assert expiry_uses >= velocity_calls, "a velocity command is missing end_time_secs"


# ----------------------------------------------------------------------
# The lease keep-alive callback
#
# Learned the hard way on the robot: this callback runs *inside* the keep-alive
# thread. A wrong signature raised TypeError there, killed the thread, and the
# lease was never retained again -- turning a two-second blip into a dead
# session where every command answered "I lost my control lease".


def test_the_lease_failure_callback_matches_what_the_sdk_passes():
    import inspect

    pytest.importorskip("bosdyn.client", reason="Spot SDK not installed")
    from spot_voice.robot.spot_client import SpotClient

    signature = inspect.signature(SpotClient._on_lease_lost)
    # self + the exception the SDK passes as retain_lease_failed_cb(exc)
    assert len(signature.parameters) == 2, signature


def test_the_lease_failure_callback_never_raises():
    pytest.importorskip("bosdyn.client", reason="Spot SDK not installed")
    from spot_voice.robot.spot_client import SpotClient

    client = SpotClient(ip="192.0.2.1")

    # A notifier that explodes must not propagate into the keep-alive thread.
    def explode(_message):
        raise RuntimeError("speaker on fire")

    client._on_lease_conflict = explode
    client._on_lease_lost(RuntimeError("LeaseUseError"))  # must not raise
    assert client.lease_lost is True


def test_repeated_lease_failures_notify_once():
    pytest.importorskip("bosdyn.client", reason="Spot SDK not installed")
    from spot_voice.robot.spot_client import SpotClient

    client = SpotClient(ip="192.0.2.1")
    said: list[str] = []
    client._on_lease_conflict = said.append

    for _ in range(5):  # the keep-alive retries every 2 seconds
        client._on_lease_lost(RuntimeError("LeaseUseError"))

    assert len(said) == 1
