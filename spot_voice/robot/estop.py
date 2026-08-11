"""Software e-stop endpoint, no GUI.

Spot requires a registered e-stop endpoint that checks in periodically. If the
check-ins stop -- program crash, wifi drop, laptop asleep -- the robot cuts motor
power on its own. That behaviour is a feature and we rely on it.

Scope note: ``allow()`` (releasing the e-stop) is deliberately **not** exposed as
a Claude tool. It is called once, by this program, at startup. The physical
tablet e-stop remains the ultimate authority at all times.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

#: Seconds the robot will wait for a check-in before cutting motor power itself.
ESTOP_TIMEOUT_SEC = 3.0
#: Name this endpoint registers under, visible on the tablet.
ESTOP_NAME = "SpotVoice E-Stop"


class SoftwareEstop:
    """Wrapper around an ``EstopEndpoint`` plus its keep-alive.

    Args:
        estop_client: A ``bosdyn.client.estop.EstopClient``.
        timeout_sec: Check-in timeout registered with the robot.
        name: Endpoint name shown on the tablet.
    """

    def __init__(
        self,
        estop_client,
        timeout_sec: float = ESTOP_TIMEOUT_SEC,
        name: str = ESTOP_NAME,
    ) -> None:
        from bosdyn.client.estop import EstopEndpoint, EstopKeepAlive

        endpoint = EstopEndpoint(estop_client, name, timeout_sec)
        # Take sole ownership of the e-stop system for this endpoint.
        endpoint.force_simple_setup()

        self._client = estop_client
        self._keep_alive = EstopKeepAlive(endpoint)
        # Start in the released state so the robot can move; the keep-alive is
        # what keeps it that way.
        self._keep_alive.allow()
        LOGGER.info("Software e-stop registered as %r (timeout %.1fs)", name, timeout_sec)

    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Assert the e-stop: motor power is cut immediately."""
        self._keep_alive.stop()

    def allow(self) -> None:
        """Release the e-stop.

        CLI/startup only -- never reachable from a Claude tool call.
        """
        self._keep_alive.allow()

    def settle_then_cut(self) -> None:
        """Let the robot settle to the ground, then cut motor power.

        Reserved for a genuine emergency power cut. The reflex "stop" word uses
        :meth:`~spot_voice.robot.spot_client.SpotClient.stop_all` instead, which
        cancels motion and holds a safe stand.
        """
        self._keep_alive.settle_then_cut()

    def is_stopped(self) -> bool:
        """True when the robot's e-stop system is currently asserted."""
        from bosdyn.client.estop import is_estopped

        try:
            return bool(is_estopped(self._client))
        except Exception:  # transport failure -- report unknown as "not stopped"
            return False

    def shutdown(self) -> None:
        """Stop checking in and deregister. Safe to call twice."""
        try:
            self._keep_alive.shutdown()
        except Exception:  # pragma: no cover - best-effort teardown
            LOGGER.debug("e-stop keep-alive shutdown raised", exc_info=True)

    def __enter__(self) -> "SoftwareEstop":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
