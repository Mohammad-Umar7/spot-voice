"""Pre-flight checks: is everything this needs actually reachable, right now?

Run ``python -m spot_voice --check`` before a demo. It answers the questions
that otherwise get answered by something failing in front of an audience: can
this laptop reach the robot, can it reach the model API, are the credentials
present, is the microphone there, is the map where it says it is.

Uses plain TCP connects rather than the SDKs, so it is fast, needs no
credentials, and works even when a package is missing.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from urllib.parse import urlparse

#: Spot's gRPC/HTTPS API port.
SPOT_API_PORT = 443

#: How long to wait on each connect. Short: a demo-floor check should be quick,
#: and anything slower than this will feel broken in use anyway.
CONNECT_TIMEOUT_SEC = 3.0

#: Where each model provider lives, for the reachability check.
PROVIDER_HOSTS = {
    "anthropic": "api.anthropic.com",
    "groq": "api.groq.com",
}
GEMINI_HOST = "generativelanguage.googleapis.com"


@dataclass
class Check:
    """One pre-flight result."""

    name: str
    ok: bool
    detail: str
    fix: str = ""

    @property
    def symbol(self) -> str:
        return "PASS" if self.ok else "FAIL"


def can_reach(host: str, port: int, timeout: float = CONNECT_TIMEOUT_SEC) -> tuple[bool, str]:
    """Try a TCP connect. Returns ``(reachable, detail)``.

    A refused connection still proves the host is *there*, which is the useful
    distinction: "wrong port" is a very different problem from "wrong network".
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{host}:{port} reachable"
    except socket.timeout:
        return False, f"{host}:{port} timed out after {timeout:.0f}s"
    except ConnectionRefusedError:
        return False, f"{host}:{port} refused (host is up, nothing listening)"
    except socket.gaierror as exc:
        return False, f"{host} did not resolve ({exc.strerror or exc})"
    except OSError as exc:
        return False, f"{host}:{port} unreachable ({exc.strerror or exc})"


def _host_of(value: str) -> str:
    """Accept a bare host, an IP, or a URL and return just the host."""
    value = (value or "").strip()
    if "://" in value:
        return urlparse(value).hostname or value
    return value.split("/")[0].split(":")[0]


# ----------------------------------------------------------------------
# Individual checks
# ----------------------------------------------------------------------


def check_robot(config) -> Check:
    """Can this laptop reach Spot's API?"""
    if config.mock_robot:
        return Check("robot", True, "MOCK_ROBOT=true, no robot needed")
    host = _host_of(config.spot_ip)
    if not host:
        return Check("robot", False, "SPOT_IP is not set", "Set SPOT_IP in .env")
    ok, detail = can_reach(host, SPOT_API_PORT)
    return Check(
        "robot",
        ok,
        detail,
        "" if ok else (
            "Check you are on the robot's wifi and that SPOT_IP is current -- "
            "a DHCP address can move when the robot reboots."
        ),
    )


def check_robot_credentials(config) -> Check:
    """Are the Spot credentials the SDK reads actually present?"""
    import os

    from .config import BOSDYN_PASSWORD_ENV, BOSDYN_USERNAME_ENV

    if config.mock_robot:
        return Check("robot login", True, "MOCK_ROBOT=true, not needed")
    missing = [
        name for name in (BOSDYN_USERNAME_ENV, BOSDYN_PASSWORD_ENV) if not os.getenv(name)
    ]
    if missing:
        return Check(
            "robot login", False, f"missing: {', '.join(missing)}", "Set them in .env"
        )
    return Check("robot login", True, "credentials present")


def check_spot_sdk(config) -> Check:
    """Is the Boston Dynamics SDK importable?"""
    if config.mock_robot:
        return Check("spot sdk", True, "MOCK_ROBOT=true, not imported")
    try:
        import bosdyn.client  # noqa: F401
    except ImportError as exc:
        return Check("spot sdk", False, str(exc), "pip install bosdyn-client")
    return Check("spot sdk", True, "bosdyn-client importable")


def check_model_provider(config) -> Check:
    """Can this laptop reach the tool-calling provider, and is there a key?"""
    provider = config.llm_provider
    host = PROVIDER_HOSTS.get(provider)
    if host is None:
        return Check("model api", False, f"unknown provider {provider!r}", "Fix LLM_PROVIDER")
    if not config.llm_api_key:
        return Check(
            "model api",
            False,
            f"{provider}: no API key set",
            f"Set the {provider.upper()}_API_KEY in .env",
        )
    ok, detail = can_reach(host, 443)
    return Check(
        "model api",
        ok,
        f"{provider} ({config.llm_model}): {detail}",
        "" if ok else (
            "No internet path. On the robot's own access point you need a second "
            "interface -- phone tether or ethernet. See the README."
        ),
    )


def check_vision_provider(config) -> Check:
    """Can this laptop reach the vision provider, if one is configured?"""
    choice = config.vision_provider
    if choice in {"", "none"}:
        return Check("vision api", True, "no vision provider; photos will not be described")
    if choice == "anthropic":
        return Check("vision api", True, "images handled by the model itself")
    if choice == "gemini":
        if not config.gemini_api_key:
            return Check(
                "vision api", False, "GEMINI_API_KEY is not set", "Set it, or VISION_PROVIDER=none"
            )
        ok, detail = can_reach(GEMINI_HOST, 443)
        return Check("vision api", ok, f"gemini ({config.gemini_model}): {detail}")
    return Check("vision api", False, f"unknown provider {choice!r}", "Fix VISION_PROVIDER")


def check_depth_sensing(config) -> Check:
    """Report which range sensor this robot actually has.

    Base Spot has stereo depth on every camera pair. The Enhanced Autonomy
    Payload adds a Velodyne LiDAR, which registers as a point-cloud service.
    Asked at runtime rather than assumed either way -- it decides how accurate
    "go where I'm pointing" will be.
    """
    if config.mock_robot:
        return Check("range sensing", True, "MOCK_ROBOT=true, simulated depth")
    try:
        import bosdyn.client  # noqa: F401
    except ImportError:
        return Check("range sensing", True, "unknown (Spot SDK not installed)")
    return Check(
        "range sensing",
        True,
        "checked when connected: LiDAR if a point-cloud service is registered, "
        "otherwise the stereo depth cameras",
    )


def check_faces(config) -> Check:
    """Is anyone enrolled, and can the recogniser load?"""
    if not config.face_recognition:
        return Check(
            "face recognition",
            True,
            "FACE_RECOGNITION=false; follow-me uses position only",
        )

    from .vision.faces import FaceStore

    store = FaceStore(config.face_store_path)
    if store.is_empty:
        return Check(
            "face recognition",
            True,
            "nobody enrolled; follow-me locks onto whoever is in front",
            "Run: python -m spot_voice --enroll <name>",
        )
    try:
        import insightface  # noqa: F401
    except ImportError:
        return Check(
            "face recognition",
            False,
            f"enrolled: {', '.join(store.names)} -- but insightface is not installed",
            "pip install insightface onnxruntime, or set FACE_RECOGNITION=false",
        )
    following = config.operator_name or "(OPERATOR_NAME not set)"
    return Check("face recognition", True, f"enrolled: {', '.join(store.names)}; following {following}")


def check_microphone(config) -> Check:
    """Does the configured microphone exist?"""
    from .audio.devices import MicrophoneNotFound, list_input_devices, select_input_device

    devices = list_input_devices()
    if not devices:
        return Check(
            "microphone", False, "no input devices found", "Check the audio stack / drivers"
        )
    try:
        index = select_input_device(config.mic_device_name, devices)
    except MicrophoneNotFound:
        names = ", ".join(device.name for device in devices[:4])
        return Check(
            "microphone",
            False,
            f"nothing matches {config.mic_device_name!r}",
            f"Available: {names}. Run --list-devices for the full list.",
        )
    if index is None:
        default = next((d for d in devices if d.is_default), devices[0])
        return Check("microphone", True, f"system default: {default.name}")
    chosen = next(device for device in devices if device.index == index)
    return Check("microphone", True, chosen.name)


def check_map(config) -> Check:
    """Is the GraphNav map where GRAPH_PATH says it is?"""
    if config.graph_path is None:
        return Check("map", True, "no GRAPH_PATH set; navigate_to will be unavailable")
    if not config.graph_path.exists():
        return Check("map", False, f"{config.graph_path} does not exist", "Fix GRAPH_PATH")
    if not (config.graph_path / "graph").exists():
        return Check(
            "map",
            False,
            f"no 'graph' file inside {config.graph_path}",
            "GRAPH_PATH should point at the 'downloaded_graph' folder itself",
        )
    waypoints = config.graph_path / "waypoint_snapshots"
    count = len(list(waypoints.iterdir())) if waypoints.is_dir() else 0
    return Check("map", True, f"graph found, {count} waypoint snapshots")


# ----------------------------------------------------------------------
# Finding the robot
# ----------------------------------------------------------------------

#: Timeout per host when sweeping a subnet. Short, because 254 of them run.
SCAN_TIMEOUT_SEC = 0.6
#: Concurrent probes. High enough to finish in seconds, low enough to be polite.
SCAN_WORKERS = 64


def subnet_of(address: str) -> str | None:
    """Return the ``a.b.c`` prefix of an IPv4 address, or ``None``.

    Args:
        address: Something like ``192.168.33.137``.
    """
    parts = _host_of(address).split(".")
    if len(parts) != 4:
        return None
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None
    if not all(0 <= number <= 255 for number in numbers):
        return None
    return ".".join(parts[:3])


def find_robots(
    subnet: str,
    port: int = SPOT_API_PORT,
    timeout: float = SCAN_TIMEOUT_SEC,
    workers: int = SCAN_WORKERS,
) -> list[str]:
    """Sweep a /24 for hosts answering on Spot's API port.

    Spot's address is handed out by DHCP on the facility wifi, so it moves. This
    turns "which address is it today" into a few seconds of waiting instead of a
    debugging session.

    Args:
        subnet: The ``a.b.c`` prefix to sweep.
        port: Port to probe. Defaults to Spot's API port.
        timeout: Per-host connect timeout.
        workers: How many probes to run at once.

    Returns:
        Addresses that accepted a connection, in numeric order.
    """
    from concurrent.futures import ThreadPoolExecutor

    candidates = [f"{subnet}.{host}" for host in range(1, 255)]

    def probe(address: str) -> str | None:
        reachable, _detail = can_reach(address, port, timeout=timeout)
        return address if reachable else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        found = [result for result in pool.map(probe, candidates) if result]
    return sorted(found, key=lambda address: int(address.rsplit(".", 1)[1]))


#: Strings in a TLS certificate that suggest the host really is a Spot rather
#: than a router or a printer that also happens to answer on 443.
SPOT_CERT_HINTS = ("boston", "bosdyn", "spot")


def tls_identity(host: str, port: int = SPOT_API_PORT, timeout: float = 2.0) -> str | None:
    """Read identifying text out of a host's TLS certificate.

    Spot presents a self-signed certificate, so this deliberately does not
    verify it -- the goal is identification, not trust. Nothing is sent and no
    credentials are involved; it is the handshake and then a disconnect.

    Returns:
        A short printable summary of the certificate's names, or ``None``.
    """
    import ssl

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((host, port), timeout=timeout) as raw:
            with context.wrap_socket(raw, server_hostname=host) as tls:
                der = tls.getpeercert(binary_form=True)
    except Exception:
        return None
    if not der:
        return None

    # The certificate is DER, so rather than pull in an X.509 parser just
    # recover the printable runs -- which is where the subject and issuer names
    # live -- and show them. Crude, but it needs no dependency and the operator
    # only has to recognise the name.
    text = "".join(chr(byte) if 32 <= byte < 127 else "\n" for byte in der)
    tokens = [token.strip() for token in text.split("\n") if len(token.strip()) >= 4]
    unique = list(dict.fromkeys(tokens))
    return ", ".join(unique)[:120] or None


def looks_like_spot(identity: str | None) -> bool:
    """True when a certificate summary mentions Boston Dynamics or Spot."""
    if not identity:
        return False
    lowered = identity.lower()
    return any(hint in lowered for hint in SPOT_CERT_HINTS)


#: Every check, in the order they are worth reading.
ALL_CHECKS = (
    check_robot,
    check_robot_credentials,
    check_spot_sdk,
    check_model_provider,
    check_vision_provider,
    check_depth_sensing,
    check_faces,
    check_microphone,
    check_map,
)


def run_all(config) -> list[Check]:
    """Run every pre-flight check. Never raises."""
    results: list[Check] = []
    for check in ALL_CHECKS:
        try:
            results.append(check(config))
        except Exception as exc:  # a broken check must not hide the others
            results.append(
                Check(getattr(check, "__name__", "check"), False, f"check failed: {exc}")
            )
    return results
