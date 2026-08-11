"""Error taxonomy: turn SDK exceptions into short sentences a robot can say.

Every message here is written to be *spoken*, not read: one sentence, no jargon,
and where possible it tells the operator what to do next.

Matching is done on the exception's **class name** rather than by importing
``bosdyn`` types, so this module works unchanged in mock mode on a machine where
the Spot SDK is not installed.
"""

from __future__ import annotations

from typing import Iterable


class SpotVoiceError(RuntimeError):
    """Base class for errors that already carry a speakable message."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.speakable = message


class RobotUnreachable(SpotVoiceError):
    """Network/RPC level failure talking to the robot."""


class RobotBusy(SpotVoiceError):
    """Another controller holds the lease, or the robot is estopped."""


class NotLocalized(SpotVoiceError):
    """GraphNav has no fix on where the robot is."""


# Exception class name -> spoken message. Ordered most-specific first.
_SPEAKABLE: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("ResourceAlreadyClaimedError",),
        "Spot is claimed by another controller.",
    ),
    (
        ("NoSuchLease", "LeaseUseError"),
        "I lost my control lease. Please try again.",
    ),
    (
        ("EstoppedError", "MotorsOnError"),
        "Spot is emergency stopped. Release it on the tablet before I can move.",
    ),
    (
        ("UnregisteredServiceNameError",),
        "Spot is still booting. Give it a moment and ask again.",
    ),
    (
        ("FaultedError",),
        "Spot hit a fault powering on. Check the battery and the tablet.",
    ),
    (
        ("RobotNotLocalizedToRouteError", "RobotLostError"),
        "I can't localize. I need to see a fiducial marker.",
    ),
    (
        ("RobotImpairedError",),
        "Spot reports it is impaired and can't move right now.",
    ),
    (
        ("CommandFailedError", "CommandTimedOutError"),
        "Spot didn't complete that command. I'll stay put.",
    ),
    (
        ("UnableToConnectToRobotError", "ProxyConnectionError", "TimedOutError", "RpcError"),
        "I lost connection to Spot.",
    ),
    (
        ("UnimplementedError", "EndpointUnknownError"),
        "That isn't available on this robot.",
    ),
    (
        ("PermissionDeniedError", "InvalidLoginError", "InvalidTokenError"),
        "My credentials for Spot were rejected.",
    ),
    (
        ("FileNotFoundError", "NotADirectoryError"),
        "I couldn't load my map files from disk.",
    ),
    (
        ("ConnectionError", "OSError"),
        "I lost connection to Spot.",
    ),
)


def _class_names(exc: BaseException) -> Iterable[str]:
    """Yield the exception's class name and those of all its base classes."""
    for klass in type(exc).__mro__:
        yield klass.__name__


def to_speakable(exc: BaseException, fallback: str = "Something went wrong on the robot.") -> str:
    """Map an exception to a short sentence suitable for text-to-speech.

    Args:
        exc: The exception raised by the SDK (or anything else).
        fallback: What to say when nothing matches.

    Returns:
        A one-sentence, speakable description of the failure.
    """
    if isinstance(exc, SpotVoiceError):
        return exc.speakable

    names = set(_class_names(exc))
    for candidates, message in _SPEAKABLE:
        if names.intersection(candidates):
            return message

    # Several Spot errors are ResponseError subclasses whose useful detail lives
    # in the message text rather than the type.
    text = str(exc)
    if "STATUS_NO_MATCHING_FIDUCIAL" in text:
        return "I can't localize. I need to see a fiducial marker."
    if "STATUS_ERROR_DOCK_NOT_FOUND" in text:
        return "I can't find the dock. Make sure its marker is in view."
    if "STATUS_STUCK" in text:
        return "I got stuck on the way there. I've stopped."
    if "STATUS_LOST" in text:
        return "I got lost on the route. I've stopped."
    return fallback


def is_connection_error(exc: BaseException) -> bool:
    """True when the exception looks like a transport problem worth retrying."""
    names = set(_class_names(exc))
    return bool(
        names.intersection(
            {
                "UnableToConnectToRobotError",
                "ProxyConnectionError",
                "TimedOutError",
                "RpcError",
                "ConnectionError",
                "ConnectionResetError",
                "OSError",
            }
        )
    )
