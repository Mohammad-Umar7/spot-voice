"""Entry point: wire the lanes together and run the voice loop.

    your voice
      -> microphone capture (16 kHz mono)
      -> VAD segmentation
      -> local faster-whisper
      -> reflex check
           |  match    -> hardcoded robot action, immediately
           |  no match -> Claude tool-use loop -> robot layer -> Spot
      -> reply text -> TTS -> Spot's speaker

Run ``python -m spot_voice --help`` for the flags.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from typing import Any, Callable

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .audio.devices import list_input_devices
from .brain.dispatcher import ToolDispatcher
from .config import Config, ConfigError, load_config
from .robot.base import RobotInterface
from .robot.follow import FollowController, make_detector_factory
from .safety.reflex import ReflexEngine
from .wake import WakeGate
from .tts.speaker import Speaker

LOGGER = logging.getLogger("spot_voice")
CONSOLE = Console()


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------


def build_robot(
    config: Config,
    console: Console,
    on_lease_conflict: Callable[[str], None] | None = None,
) -> RobotInterface:
    """Construct the mock or the real robot layer, per ``MOCK_ROBOT``."""
    if config.mock_robot:
        from .robot.mock import MockSpot

        return MockSpot(
            graph_path=config.graph_path,
            dock_id=config.dock_id,
            console=console,
        )

    from .robot.spot_client import SpotClient

    return SpotClient(
        ip=config.spot_ip,
        graph_path=config.graph_path,
        dock_id=config.dock_id,
        on_lease_conflict=on_lease_conflict,
    )


class VoiceApp:
    """Owns every component and routes one transcript at a time."""

    def __init__(self, config: Config, console: Console | None = None) -> None:
        self.config = config
        self.console = console or CONSOLE

        self.robot = build_robot(config, self.console, self._on_lease_conflict)
        self.follow = FollowController(
            self.robot,
            detector_factory=make_detector_factory(config.mock_robot),
            say=lambda text: self.speaker.speak(text),
            tracker_factory=self._build_tracker,
            face_factory=self._build_face_pipeline,
        )
        self.speaker = Speaker(
            engine=config.tts_engine,
            audio_out=config.audio_out,
            voice=config.tts_voice,
            work_dir=config.work_dir,
            robot=self.robot,
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
            console=self.console,
        )
        self.dispatcher = ToolDispatcher(
            robot=self.robot,
            follow=self.follow,
            speak=lambda text: self.speaker.speak(text),
            console=self.console,
            pose_reader=self._read_pointing,
        )
        self._pose_model: Any = None
        self.brain: Any = None
        self.reflex = ReflexEngine(
            robot=self.robot,
            follow=self.follow,
            say=lambda text: self.speaker.speak(text),
            on_abort=self._abort_brain,
        )
        self.wake = WakeGate(
            wake_word=config.wake_word,
            follow_up_sec=config.wake_follow_up_sec,
            enabled=config.require_wake_word,
        )

        self.listener: Any = None
        self._busy = threading.Lock()
        self._shutdown = threading.Event()

    # ------------------------------------------------------------------

    def _read_pointing(self, frame_jpeg: bytes):
        """Read a pointing gesture from a frame, loading the model on first use.

        Lazy because the pose weights download on first run and most sessions
        never ask anyone to point at anything.
        """
        if self._pose_model is None:
            from .vision.pointing import YoloPoseReader

            self._pose_model = YoloPoseReader()
        return self._pose_model(frame_jpeg)

    def _build_tracker(self):
        """The identity tracker follow-me uses to stay locked on one person."""
        from .vision.identity import IdentityTracker

        return IdentityTracker(operator=self.config.operator_name or None)

    def _build_face_pipeline(self):
        """Return ``(recogniser, store)``, or ``(None, None)``.

        Called on the follow thread, because loading a face model takes seconds.
        Missing model or empty enrollment is not an error: follow-me falls back
        to locking onto whoever is in front of the robot.
        """
        if not self.config.face_recognition:
            return None, None

        from .vision.faces import FaceRecogniser, FaceStore

        store = FaceStore(self.config.face_store_path)
        if store.is_empty:
            LOGGER.info("No faces enrolled; follow-me will use position only")
            return None, None
        try:
            return FaceRecogniser(), store
        except Exception as exc:
            LOGGER.warning("Face recogniser unavailable: %s", exc)
            return None, None

    def _on_lease_conflict(self, message: str) -> None:
        """Say, loudly and out loud, that something else took control.

        Without this the failure is invisible: commands still return success
        while the robot ignores them, so it looks like the software is broken
        rather than out-ranked.
        """
        self.console.print(f"\n[bold red]LEASE LOST[/bold red] {message}\n")
        try:
            self.speaker.speak("I've lost control of Spot. Something else has taken the lease.")
        except Exception:
            LOGGER.debug("could not announce lease loss", exc_info=True)

    def _on_speech_start(self) -> None:
        if self.listener is not None and self.config.mute_while_speaking:
            self.listener.mute()

    def _on_speech_end(self) -> None:
        if self.listener is not None and self.config.mute_while_speaking:
            self.listener.unmute()

    def _abort_brain(self) -> None:
        if self.brain is not None:
            self.brain.abort()

    # ------------------------------------------------------------------

    def start(self, with_microphone: bool = True, with_brain: bool = True) -> None:
        """Connect to the robot, build the brain and (optionally) open the mic.

        Args:
            with_microphone: Open the mic and load the speech model.
            with_brain: Build the Anthropic client. Manual mode passes ``False``
                so bring-up needs nothing but the robot.
        """
        self.robot.connect()
        self.report_robot_state()

        if not with_brain:
            # Manual mode. Deliberately no brain; run_manual_mode says so.
            pass
        else:
            self._build_brain()

        if with_microphone:
            from .audio.listener import Listener
            from .audio.stt import Transcriber

            transcriber = Transcriber(
                model_size=self.config.whisper_model,
                language=self.config.stt_language,
            )
            self.listener = Listener(
                transcriber=transcriber,
                on_transcript=lambda transcript: self.handle(transcript.text),
                mic_name=self.config.mic_device_name,
                console=self.console,
            )
            self.listener.start()

    def _build_brain(self) -> None:
        """Construct the model providers named in .env.

        Degrades rather than refusing to start: a missing key or SDK leaves the
        reflex lane running, which is the part that matters most and has no
        dependency on any of this.
        """
        from .brain.agent import Brain
        from .brain.providers import (
            ProviderUnavailable,
            build_llm_provider,
            build_vision_provider,
        )

        try:
            provider = build_llm_provider(self.config)
        except ProviderUnavailable as exc:
            self.console.print(
                f"[yellow]No language model ({exc}) -- running reflex-only. "
                "Safety words work; conversation does not.[/yellow]"
            )
            return

        vision = build_vision_provider(self.config)
        if vision is not None and not provider.supports_images:
            self.console.print(
                f"[dim]{provider.name} cannot see images, so camera frames go "
                f"through {vision.name} and come back as text.[/dim]"
            )

        self.brain = Brain(
            provider=provider,
            dispatcher=self.dispatcher,
            vision=vision,
            extra_context=self._map_context(),
            console=self.console,
        )

    def report_robot_state(self) -> None:
        """Print what the robot is doing right now, and anything blocking motion.

        Printed once after connecting. Without it you are talking to a robot
        whose posture, power and e-stop state you cannot see -- which is the
        difference between "say stand and it works" and "say stand and wonder".
        """
        result = self.robot.get_status()
        if not result.ok or not result.data:
            self.console.print(f"[red]Could not read robot state: {result.message}[/red]")
            return

        state = result.data
        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim")
        table.add_column()
        for key in (
            "battery_percent",
            "motor_power",
            "posture",
            "docked",
            "estop",
            "lease",
            "localization",
            "location",
        ):
            if key in state and state[key] is not None:
                table.add_row(key.replace("_", " "), str(state[key]))
        self.console.print(Panel(table, title="robot state", border_style="green"))

        # Anything that would make "stand" fail, said plainly and in the order
        # you would have to fix it.
        blockers: list[str] = []
        if str(state.get("estop", "")).startswith("asserted"):
            blockers.append(
                "E-stop is asserted. Release it on the tablet before anything can move."
            )
        if state.get("lease") == "not held":
            blockers.append("I don't hold the lease. Close the tablet app and restart me.")
        if state.get("docked"):
            blockers.append("Spot is on the dock. Say 'undock' before asking it to stand.")
        battery = state.get("battery_percent")
        if isinstance(battery, (int, float)) and battery < 20:
            blockers.append(f"Battery is at {battery:.0f} percent. Dock it before a walk.")

        if blockers:
            for line in blockers:
                self.console.print(f"[yellow]![/yellow] {line}")
        elif state.get("posture") == "sitting" or state.get("motor_power") == "off":
            self.console.print(
                "[green]Ready.[/green] Spot is sitting with motors off. "
                "Say [bold]\"stand up\"[/bold] -- that powers the motors and stands it "
                "in one step."
            )
        else:
            self.console.print("[green]Ready.[/green] Spot is standing.")
        self.console.print()

    def _map_context(self) -> str | None:
        """Stable, site-specific text appended to the system prompt.

        Fetched once at startup so it stays byte-identical for the session and
        does not invalidate the prompt cache.
        """
        try:
            result = self.robot.list_waypoints()
        except Exception:
            return None
        if not result.ok or not result.data:
            return None
        names = result.data.get("waypoints") or []
        if not names:
            return None
        return "Places on this facility map: " + ", ".join(names) + "."

    # ------------------------------------------------------------------

    def handle(self, text: str) -> None:
        """Route one utterance: reflex lane first, then the Claude lane."""
        text = (text or "").strip()
        if not text:
            return

        # Reflex lane. Runs on whatever thread delivered the transcript, ahead
        # of any lock, so a spoken "stop" is never queued behind a Claude call.
        outcome = self.reflex.handle(text)
        if outcome is not None:
            self.console.print(
                f"[bold red]REFLEX[/bold red] {outcome.match.action.value} "
                f"[dim]({outcome.latency_ms:.0f} ms)[/dim] {outcome.message}"
            )
            return

        # Addressing. Below the reflex lane on purpose: safety words are already
        # handled above and must never need the robot's name, but a *command*
        # does, otherwise every sentence spoken near the microphone is one.
        address = self.wake.check(text)
        if not address.addressed:
            self.console.print(
                f"[dim]not for me: {text}[/dim]"
                if self.wake.enabled
                else f"[dim]{text}[/dim]"
            )
            return
        if address.is_bare_name:
            self.speaker.speak("Yes?")
            self.wake.note_reply()
            return
        text = address.command

        if self.brain is None:
            self.speaker.speak("I don't have my language service, but stop and sit still work.")
            return

        if not self._busy.acquire(blocking=False):
            self.console.print("[yellow](still working on the last one -- ignored)[/yellow]")
            return
        try:
            started = time.perf_counter()
            reply = self.brain.handle(text)
            elapsed = (time.perf_counter() - started) * 1000.0
            LOGGER.info(
                "brain turn: %d tool calls in %.0f ms", len(reply.tool_calls), elapsed
            )
            if reply.aborted:
                self.console.print("[yellow](cancelled by a safety word)[/yellow]")
                return
            if reply.text:
                self.speaker.speak(reply.text)
                self.wake.note_reply()
        finally:
            self._busy.release()

    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop everything in a safe order. Safe to call twice."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self.console.print("\n[bold]Shutting down...[/bold]")
        try:
            self.follow.stop()
        except Exception:
            LOGGER.debug("follow stop failed", exc_info=True)
        if self.listener is not None:
            try:
                self.listener.stop()
            except Exception:
                LOGGER.debug("listener stop failed", exc_info=True)
        try:
            self.robot.stop_all()
        except Exception:
            LOGGER.debug("stop_all failed", exc_info=True)
        try:
            self.robot.shutdown()
        except Exception:
            LOGGER.debug("robot shutdown failed", exc_info=True)
        self.console.print("[bold]Done.[/bold]")


# ----------------------------------------------------------------------
# Manual mode
# ----------------------------------------------------------------------

#: Bare commands with no arguments.
_MANUAL_ALIASES: dict[str, str] = {
    "power": "power_on",
    "poweron": "power_on",
    "power_on": "power_on",
    "stand": "stand",
    "up": "stand",
    "sit": "sit",
    "down": "sit",
    "stop": "stop_all",
    "halt": "stop_all",
    "status": "get_status",
    "state": "get_status",
    "waypoints": "list_waypoints",
    "places": "list_waypoints",
    "dock": "dock",
    "undock": "undock",
    "follow": "start_follow",
    "unfollow": "stop_follow",
}

MANUAL_HELP = """\
  stand | up            power the motors and stand up
  sit | down            sit down
  stop | halt           cancel everything, safe stop
  status                battery, motor power, e-stop, dock, localization
  move <dir> [amount]   move forward 1.5 | move turn_left 90
  go <waypoint>         navigate to a named place
  look [front|left|right]   take a photo
  waypoints             list the places on the map
  undock | dock         leave or return to the charger
  follow | unfollow     follow-me
  point                 look for a pointing gesture and walk that way
  emote <gesture>       greet | bow | nod | shake | wiggle | look_around
  say <text>            speak through the speaker
  help                  this list
  quit                  exit\
"""


def parse_manual_command(line: str) -> tuple[str, dict[str, Any]] | None:
    """Map a typed operator command to a ``(tool_name, arguments)`` pair.

    Returns ``None`` when the line is not a command this mode understands.
    """
    tokens = (line or "").strip().split()
    if not tokens:
        return None
    head = tokens[0].lower()
    rest = tokens[1:]

    if head in _MANUAL_ALIASES:
        return _MANUAL_ALIASES[head], {}

    if head in {"look", "photo", "picture"}:
        camera = rest[0].lower() if rest else "front"
        return "capture_image", {"camera": camera}

    if head == "say":
        return "speak", {"text": " ".join(rest)}

    if head in {"point", "there", "gothere"}:
        return "go_where_pointed", {}

    if head in {"emote", "gesture", "do"}:
        return ("emote", {"gesture": rest[0]}) if rest else None

    if head in {"go", "goto", "navigate"}:
        return "navigate_to", {"waypoint_name": " ".join(rest)}

    if head == "move":
        if not rest:
            return None
        direction = rest[0].lower()
        arguments: dict[str, Any] = {"direction": direction}
        if len(rest) > 1:
            try:
                amount = float(rest[1])
            except ValueError:
                return None
            # A turn is measured in degrees, everything else in metres.
            key = "degrees" if direction.startswith("turn") else "distance_m"
            arguments[key] = amount
        return "move", arguments

    return None


def run_manual_mode(app: "VoiceApp", console: Console) -> None:
    """Drive the robot by typing tool commands, with Claude out of the loop.

    This exists for robot bring-up. Stages 2 and 3 of the rollout are exactly
    when you are standing a real robot for the first time over a tether that may
    not be up yet, and "stand" is not a reflex word -- so without this you would
    need a working Anthropic connection to stand the robot. This path needs
    nothing but the robot.
    """
    console.print(
        "[bold]Manual mode.[/bold] Commands go straight to the robot -- "
        "no speech, no Claude, no internet needed.\n"
    )
    console.print(f"[dim]{MANUAL_HELP}[/dim]\n")

    while True:
        try:
            line = input("robot> ").strip()
        except EOFError:
            return
        if not line:
            continue
        if line.lower() in {"quit", "exit", "q"}:
            return
        if line.lower() in {"help", "?"}:
            console.print(f"[dim]{MANUAL_HELP}[/dim]")
            continue

        parsed = parse_manual_command(line)
        if parsed is None:
            console.print(
                f"[yellow]Don't know '{line}'. Type 'help' for the list.[/yellow]"
            )
            continue

        name, arguments = parsed
        result = app.dispatcher.dispatch(name, arguments)
        if result.payload.get("waypoints"):
            console.print("  " + ", ".join(result.payload["waypoints"]))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def print_devices(console: Console) -> None:
    """Print every audio input, so ``MIC_DEVICE_NAME`` is easy to choose."""
    devices = list_input_devices()
    if not devices:
        console.print("[red]No audio input devices found.[/red]")
        return
    table = Table(title="Audio input devices", header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Name")
    table.add_column("Ch", justify="right")
    table.add_column("Rate", justify="right")
    for device in devices:
        name = device.name + (" (default)" if device.is_default else "")
        table.add_row(
            str(device.index), name, str(device.channels), f"{device.default_samplerate:.0f}"
        )
    console.print(table)
    console.print(
        "\nSet [bold]MIC_DEVICE_NAME[/bold] in .env to any distinctive part of the "
        "name, e.g. [bold]USB Audio[/bold]."
    )


def print_preflight(config: Config, console: Console) -> int:
    """Run the pre-flight checks and print them. Returns a process exit code."""
    from .preflight import run_all

    console.print("[bold]Pre-flight[/bold]\n")
    results = run_all(config)

    table = Table(show_header=True, header_style="bold")
    table.add_column("", width=4)
    table.add_column("check")
    table.add_column("detail")
    for result in results:
        colour = "green" if result.ok else "red"
        table.add_row(
            f"[{colour}]{result.symbol}[/{colour}]", result.name, result.detail
        )
    console.print(table)

    failures = [result for result in results if not result.ok]
    if not failures:
        console.print("\n[bold green]All clear.[/bold green]")
        return 0
    console.print()
    for result in failures:
        if result.fix:
            console.print(f"[yellow]{result.name}:[/yellow] {result.fix}")
    console.print(f"\n[bold red]{len(failures)} check(s) failed.[/bold red]")
    return 1


def run_vision_test(config: Config, console: Console) -> int:
    """Send one image through the vision provider and print the description.

    Isolates the image path from everything else. When "what do you see" gives a
    poor answer, this says whether the problem is the vision model, the robot's
    camera, or the tool-calling model -- which is three different fixes.
    """
    from pathlib import Path

    from .brain.providers import build_vision_provider

    image_path = Path(__file__).resolve().parent / "assets" / "test_scene.jpg"
    if not image_path.exists():
        console.print(f"[red]Test image missing: {image_path}[/red]")
        return 1

    console.print(
        f"Vision provider: [bold]{config.vision_provider or 'none'}[/bold]"
        + (f" ({config.gemini_model})" if config.vision_provider == "gemini" else "")
    )

    vision = build_vision_provider(config)
    if vision is None:
        console.print(
            "[green]The language model reads images itself; no separate vision "
            "provider needed.[/green]"
        )
        return 0

    console.print(f"Sending {image_path.name} ({image_path.stat().st_size} bytes)...\n")
    try:
        description = vision.describe(image_path.read_bytes())
    except Exception as exc:
        console.print(f"[bold red]Vision failed:[/bold red] {exc}")
        console.print(
            "\n[dim]Fallback: set VISION_PROVIDER=none and Spot will say it "
            "cannot see the photo rather than inventing one.[/dim]"
        )
        return 1

    console.print(f"[bold green]It says:[/bold green] {description}\n")
    if "no vision model is configured" in description:
        console.print(
            "[yellow]That is the placeholder, not a real description -- no vision "
            "provider is actually wired up.[/yellow]"
        )
        return 1
    console.print(
        "[dim]The image is a synthetic facility scene: gauges, a hazard sign, a "
        "cabinet, a pallet, a spill. If the description matches, the image path "
        "works end to end.[/dim]"
    )
    return 0


def run_enrollment(config: Config, name: str, console: Console) -> int:
    """Enroll a face from Spot's own camera.

    Requires a real robot. There is deliberately no webcam fallback: the
    embedding has to come through the same lens that will do the recognising,
    or it will match here and fail on the day.
    """
    from .enroll import enroll

    if config.mock_robot:
        return enroll(name, config, console, robot=None)  # prints why, exits 2

    robot = build_robot(config, console)
    try:
        robot.connect()
    except Exception as exc:
        console.print(
            f"[red]Could not reach the robot:[/red] {exc}\n"
            "Enrollment needs Spot's camera. Check the path with: "
            "python -m spot_voice --check"
        )
        return 1
    try:
        return enroll(name, config, console, robot=robot)
    finally:
        robot.shutdown()


def print_find_robot(config: Config, console: Console) -> int:
    """Sweep the configured subnet for anything answering on Spot's API port."""
    from .preflight import (
        SPOT_API_PORT,
        candidate_addresses,
        find_robots,
        subnet_of,
    )

    subnet = subnet_of(config.spot_ip)
    if subnet is None:
        console.print(
            f"[red]Can't work out a subnet from SPOT_IP={config.spot_ip!r}.[/red] "
            "Set it to any address on the robot's network first."
        )
        return 2

    console.print(
        f"Sweeping [bold]{subnet}.1-254[/bold] for anything listening on port "
        f"{SPOT_API_PORT}...\n"
    )
    found = find_robots(subnet)

    if not found:
        console.print(
            "[yellow]Nothing found.[/yellow] Check you are on the same wifi as the "
            "robot and that it is powered on."
        )
        return 1

    from .preflight import looks_like_spot, tls_identity

    likely: list[str] = []
    for address in found:
        identity = tls_identity(address)
        is_spot = looks_like_spot(identity)
        if is_spot:
            likely.append(address)
        label = "[green]looks like a Spot[/green]" if is_spot else "[dim]not a Spot[/dim]"
        current = " [green](SPOT_IP in .env)[/green]" if address == config.spot_ip else ""
        console.print(f"  [bold]{address}[/bold]{current}  {label}")
        if identity:
            console.print(f"      [dim]cert: {identity}[/dim]")

    if likely:
        console.print(
            f"\nSet [bold]SPOT_IP={likely[0]}[/bold] in .env if that matches the "
            "tablet."
        )
    else:
        console.print(
            "\n[yellow]None of these presented a Boston Dynamics certificate.[/yellow] "
            "The robot may be powered off, or on a different network than this "
            "laptop."
        )
    console.print(
        "[dim]The robot's tablet, under MY ROBOTS, is the authoritative source -- "
        "match against that.[/dim]"
    )
    if config.spot_ip not in found:
        console.print(
            f"\n[yellow]SPOT_IP is currently {config.spot_ip}, which did not "
            "answer.[/yellow] Update it in .env to whichever address above the "
            "tablet shows."
        )
    return 0


def print_banner(config: Config, console: Console) -> None:
    """Startup summary, including the safety posture."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    for key, value in config.describe():
        table.add_row(key, str(value))
    console.print(Panel(table, title="spot-voice", border_style="cyan"))
    console.print(
        "[dim]Spot's own obstacle avoidance, self-righting and stair handling stay "
        "on at factory defaults. The tablet e-stop is always the ultimate "
        "authority. Safety words: stop / freeze / halt / sit / stop following."
        "[/dim]\n"
    )


def harden_console_encoding() -> None:
    """Stop an odd character in a reply from killing the process.

    The default Windows console codepage is cp1252, which cannot encode emoji or
    arrows. Everything this program prints is ASCII, but Claude's replies are
    not under our control, and a UnicodeEncodeError mid-demo would take the
    robot session down with it. Replacing unencodable characters is strictly
    better than crashing.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # pragma: no cover - depends on the host
            pass


def setup_logging(level: str) -> None:
    """Rich logging to the console at the configured level."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=CONSOLE, rich_tracebacks=True, show_path=False)],
    )
    # These are chatty and rarely what you want to read.
    for noisy in ("urllib3", "httpx", "httpcore", "bosdyn", "ultralytics"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="spot-voice",
        description="Voice-commanded control of a Boston Dynamics Spot.",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="list audio inputs and exit"
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="type commands instead of speaking them (no microphone needed)",
    )
    parser.add_argument(
        "--say", metavar="TEXT", help="handle one command and exit (useful for testing)"
    )
    parser.add_argument(
        "--enroll",
        metavar="NAME",
        help="record your face so follow-me looks for you specifically, then exit",
    )
    parser.add_argument(
        "--forget", metavar="NAME", help="remove an enrolled face and exit"
    )
    parser.add_argument(
        "--test-vision",
        action="store_true",
        help="send one test image through the configured vision provider and "
        "print what it says. Verifies the whole image path in isolation.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="pre-flight: can this laptop reach the robot, the model API and the "
        "microphone? Run this before a demo.",
    )
    parser.add_argument(
        "--find-robot",
        action="store_true",
        help="sweep SPOT_IP's subnet for the robot. Use when its DHCP address "
        "has moved and you need to find it again.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="type tool commands straight at the robot; no speech, no Claude, "
        "no internet needed (use this for robot bring-up)",
    )
    parser.add_argument("--env", metavar="PATH", help="path to a .env file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the application. Returns a process exit code."""
    args = parse_args(argv)
    harden_console_encoding()

    if args.list_devices:
        print_devices(CONSOLE)
        return 0

    try:
        config = load_config(args.env)
    except ConfigError as exc:
        CONSOLE.print(f"[bold red]Configuration problem:[/bold red] {exc}")
        CONSOLE.print("[dim]Copy .env.example to .env and fill it in.[/dim]")
        return 2

    setup_logging(config.log_level)

    if args.forget:
        from .enroll import forget

        return forget(args.forget, config, CONSOLE)

    if args.enroll:
        return run_enrollment(config, args.enroll, CONSOLE)

    if args.test_vision:
        return run_vision_test(config, CONSOLE)

    if args.find_robot:
        return print_find_robot(config, CONSOLE)

    if args.check:
        return print_preflight(config, CONSOLE)

    print_banner(config, CONSOLE)

    app = VoiceApp(config, CONSOLE)

    def _handle_signal(_signum, _frame) -> None:
        app.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)

    needs_microphone = not (args.text or args.say or args.manual)
    try:
        # Manual mode is the bring-up path, so it must not depend on the
        # Anthropic SDK, a key, or a working internet connection.
        app.start(with_microphone=needs_microphone, with_brain=not args.manual)
    except Exception as exc:
        CONSOLE.print(f"[bold red]Startup failed:[/bold red] {exc}")
        LOGGER.debug("startup failure", exc_info=True)
        app.shutdown()
        return 1

    try:
        if args.manual:
            run_manual_mode(app, CONSOLE)
        elif args.say:
            app.handle(args.say)
        elif args.text:
            CONSOLE.print("[dim]Type a command and press enter. Ctrl-C to quit.[/dim]\n")
            while True:
                try:
                    line = input("you> ").strip()
                except EOFError:
                    break
                if line.lower() in {"quit", "exit"}:
                    break
                app.handle(line)
        else:
            CONSOLE.print("[bold]Listening. Ctrl-C to quit.[/bold]\n")
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
