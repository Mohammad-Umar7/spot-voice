"""GraphNav: upload the facility map, localize on a fiducial, walk to a waypoint.

Expects ``GRAPH_PATH`` to point at a ``downloaded_graph`` folder laid out the way
Autowalk / the SDK's ``graph_nav_command_line`` example writes it::

    downloaded_graph/
        graph
        waypoint_snapshots/<snapshot ids>
        edge_snapshots/<snapshot ids>

All ``bosdyn`` imports are function- or method-local so this module can be
imported (and its pure helpers tested) without the SDK installed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .errors import NotLocalized, SpotVoiceError

LOGGER = logging.getLogger(__name__)

#: Seconds between navigation feedback polls while walking a route.
_FEEDBACK_PERIOD = 0.3
#: Travel-speed limit handed to ``navigate_to`` (seconds of command validity).
_NAV_COMMAND_DURATION = 1.0
#: Give up on a single navigation command after this long.
_NAV_TIMEOUT_SEC = 300.0


def build_name_index(graph) -> dict[str, str]:
    """Map waypoint annotation names to unique waypoint ids.

    Names are not guaranteed unique in a GraphNav map. When two waypoints share a
    name we keep neither -- navigating to an ambiguous name would be a coin flip.

    Args:
        graph: A ``bosdyn.api.graph_nav.map_pb2.Graph``.

    Returns:
        ``{annotation_name: waypoint_id}`` for every unambiguously named waypoint.
    """
    seen: dict[str, str | None] = {}
    for waypoint in graph.waypoints:
        name = waypoint.annotations.name
        if not name:
            continue
        seen[name] = None if name in seen else waypoint.id
    return {name: wp_id for name, wp_id in seen.items() if wp_id is not None}


class GraphNav:
    """Thin, purpose-built wrapper over ``GraphNavClient``.

    Args:
        robot: A connected ``bosdyn.client.robot.Robot``.
        graph_path: Path to the ``downloaded_graph`` folder.
    """

    def __init__(self, robot: Any, graph_path: Path | None) -> None:
        from bosdyn.client.graph_nav import GraphNavClient

        self._robot = robot
        self._graph_path = graph_path
        self._client = robot.ensure_client(GraphNavClient.default_service_name)
        self._uploaded = False
        self._localized = False
        self._cancelled = False
        self._names: dict[str, str] = {}

    # ------------------------------------------------------------------

    @property
    def localized(self) -> bool:
        return self._localized

    def waypoint_names(self) -> list[str]:
        """Names known to the currently loaded map, sorted for stable speech."""
        if not self._names:
            self._refresh_names_from_robot()
        return sorted(self._names)

    # ------------------------------------------------------------------

    def upload_graph(self, force: bool = False) -> int:
        """Read the map from disk and upload it plus its snapshots to the robot.

        Returns:
            The number of waypoints in the uploaded graph.

        Raises:
            SpotVoiceError: If ``GRAPH_PATH`` is unset or the files are missing.
        """
        from bosdyn.api.graph_nav import map_pb2

        if self._uploaded and not force:
            return len(self._names)
        if self._graph_path is None:
            raise SpotVoiceError("I don't have a map configured, so I can't navigate.")

        graph_file = self._graph_path / "graph"
        if not graph_file.exists():
            raise SpotVoiceError("I couldn't load my map files from disk.")

        graph = map_pb2.Graph()
        graph.ParseFromString(graph_file.read_bytes())
        LOGGER.info(
            "Loaded map: %d waypoints, %d edges", len(graph.waypoints), len(graph.edges)
        )

        waypoint_snapshots: dict[str, Any] = {}
        for waypoint in graph.waypoints:
            if not waypoint.snapshot_id:
                continue
            path = self._graph_path / "waypoint_snapshots" / waypoint.snapshot_id
            if not path.exists():
                continue
            snapshot = map_pb2.WaypointSnapshot()
            snapshot.ParseFromString(path.read_bytes())
            waypoint_snapshots[snapshot.id] = snapshot

        edge_snapshots: dict[str, Any] = {}
        for edge in graph.edges:
            if not edge.snapshot_id:
                continue
            path = self._graph_path / "edge_snapshots" / edge.snapshot_id
            if not path.exists():
                continue
            snapshot = map_pb2.EdgeSnapshot()
            snapshot.ParseFromString(path.read_bytes())
            edge_snapshots[snapshot.id] = snapshot

        # Ask the robot to generate an anchoring only when the map has none.
        generate_anchoring = not len(graph.anchoring.anchors)
        response = self._client.upload_graph(
            graph=graph, generate_new_anchoring=generate_anchoring
        )
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            if snapshot_id in waypoint_snapshots:
                self._client.upload_waypoint_snapshot(waypoint_snapshots[snapshot_id])
        for snapshot_id in response.unknown_edge_snapshot_ids:
            if snapshot_id in edge_snapshots:
                self._client.upload_edge_snapshot(edge_snapshots[snapshot_id])

        self._names = build_name_index(graph)
        self._uploaded = True
        LOGGER.info("Map uploaded. Named waypoints: %s", ", ".join(sorted(self._names)))
        return len(graph.waypoints)

    # ------------------------------------------------------------------

    def localize_on_fiducial(self) -> None:
        """Set the initial localization from the nearest visible fiducial.

        Raises:
            NotLocalized: If the robot cannot see a fiducial it recognises.
        """
        from bosdyn.api.graph_nav import nav_pb2
        from bosdyn.client.frame_helpers import get_odom_tform_body
        from bosdyn.client.robot_state import RobotStateClient

        state_client = self._robot.ensure_client(RobotStateClient.default_service_name)
        state = state_client.get_robot_state()
        odom_tform_body = get_odom_tform_body(
            state.kinematic_state.transforms_snapshot
        ).to_proto()

        try:
            self._client.set_localization(
                initial_guess_localization=nav_pb2.Localization(),
                ko_tform_body=odom_tform_body,
            )
        except Exception as exc:
            text = str(exc)
            if "STATUS_NO_MATCHING_FIDUCIAL" in text or "NoMatchingFiducial" in type(exc).__name__:
                raise NotLocalized(
                    "I can't localize. I need to see a fiducial marker."
                ) from exc
            raise
        self._localized = True
        LOGGER.info("Localized from fiducial.")

    def ensure_ready(self) -> None:
        """Upload the map and localize if that has not happened yet."""
        self.upload_graph()
        if not self._localized:
            self.localize_on_fiducial()

    # ------------------------------------------------------------------

    def resolve(self, name: str) -> str | None:
        """Resolve a spoken place name to a waypoint id, or ``None``."""
        if not self._names:
            self._refresh_names_from_robot()
        if not name:
            return None
        needle = name.strip().lower().replace("_", " ").replace("-", " ")
        table = {key: key.lower().replace("_", " ").replace("-", " ") for key in self._names}
        for key, value in table.items():
            if value == needle:
                return self._names[key]
        for key, value in table.items():
            if needle in value or value in needle:
                return self._names[key]
        from difflib import SequenceMatcher

        best_key, best_score = None, 0.0
        for key, value in table.items():
            score = SequenceMatcher(None, needle, value).ratio()
            if score > best_score:
                best_key, best_score = key, score
        return self._names[best_key] if best_key and best_score >= 0.75 else None

    # ------------------------------------------------------------------

    def navigate_to(self, waypoint_id: str, should_stop=None) -> None:
        """Walk to ``waypoint_id``, polling feedback until it arrives or fails.

        The command is re-issued roughly three times a second so that killing
        this program, or asserting the e-stop, stops the robot promptly.

        Args:
            waypoint_id: Destination waypoint id.
            should_stop: Optional zero-arg callable; when it returns ``True`` the
                walk is abandoned (used by the reflex "stop" lane).

        Raises:
            SpotVoiceError: With a speakable message on any navigation failure.
        """
        from bosdyn.api.graph_nav import graph_nav_pb2

        command_id: int | None = None
        deadline = time.monotonic() + _NAV_TIMEOUT_SEC
        self._cancelled = False

        while True:
            if self._cancelled or (should_stop is not None and should_stop()):
                # Stop re-issuing: the command already on the robot expires
                # within a second and Spot brings itself to a halt.
                raise SpotVoiceError("I stopped on the way there.")
            if time.monotonic() > deadline:
                raise SpotVoiceError("That walk took too long, so I stopped.")

            command_id = self._client.navigate_to(
                waypoint_id, _NAV_COMMAND_DURATION, command_id=command_id
            )
            time.sleep(_FEEDBACK_PERIOD)

            status = self._client.navigation_feedback(command_id).status
            if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                return
            if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
                raise NotLocalized("I got lost on the route. I've stopped.")
            if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
                raise SpotVoiceError("I got stuck on the way there. I've stopped.")
            if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
                raise SpotVoiceError("Spot reports it is impaired and can't move right now.")

    def cancel(self) -> None:
        """Ask any in-progress route to end.

        GraphNav has no explicit cancel RPC. What actually stops the robot is
        two things, both of which happen elsewhere:

        * :meth:`navigate_to` stops re-issuing its command, so the one on the
          robot expires after ``_NAV_COMMAND_DURATION`` and Spot halts itself;
        * :meth:`SpotClient.stop_all` issues a stop command through the
          RobotCommandClient, which pre-empts the route immediately.

        This method just raises the flag the navigation loop watches, so it also
        works when ``should_stop`` was not supplied by the caller.
        """
        self._cancelled = True

    def localization_summary(self) -> str:
        """Human-readable localization state for ``get_status``."""
        if not self._localized:
            return "not localized"
        try:
            state = self._client.get_localization_state()
            waypoint_id = state.localization.waypoint_id
        except Exception:
            return "localized"
        for name, wp_id in self._names.items():
            if wp_id == waypoint_id:
                return f"localized near {name}"
        return "localized"

    # ------------------------------------------------------------------

    def _refresh_names_from_robot(self) -> None:
        """Pull the graph currently loaded on the robot and index its names."""
        try:
            graph = self._client.download_graph()
        except Exception:
            LOGGER.debug("download_graph failed", exc_info=True)
            return
        if graph is not None:
            self._names = build_name_index(graph)
