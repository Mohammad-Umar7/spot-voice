"""Face enrollment: teaching Spot who you are.

Run once, before a demo::

    python -m spot_voice --enroll umar

Stand in front of the camera and it takes several samples a second apart, from
slightly different angles. Several beats one: acquisition on the robot happens
in whatever light the facility has, at whatever angle you happen to be standing.

**Enrollment always uses Spot's own camera, and there is no fallback.** That is
deliberate rather than fussy: recognition happens on greyscale fisheye frames
through Spot's lens, and an embedding taken from a crisp colour webcam is a poor
match for those. A webcam fallback would enroll you successfully and then fail to
recognise you on the day, which is worse than refusing. So this needs a real
robot, connected -- it is a one-time setup step, not something to rehearse at a
desk.

Privacy: what is written to disk is a numeric embedding, not a photograph. It
stays in your local work directory and is never uploaded. Enroll only people who
have agreed to it.
"""

from __future__ import annotations

import logging
import time

from rich.console import Console

from .vision.faces import FaceStore

LOGGER = logging.getLogger(__name__)

#: How many samples to take. Enough for angle and lighting variation without
#: making the operator stand still for a minute.
SAMPLE_COUNT = 5
#: Gap between samples, so you have time to shift position slightly.
SAMPLE_GAP_SEC = 1.2

#: Pause after standing, so frames are taken from a settled robot rather than
#: one still rising -- mid-rise frames are tilted and motion-blurred.
STAND_SETTLE_SEC = 2.0


#: What to do between samples, so the samples actually differ. Variety in angle
#: is what makes acquisition work in whatever pose you happen to be standing.
POSE_PROMPTS = (
    "look straight at me",
    "turn your head slightly left",
    "turn your head slightly right",
    "tilt your chin up a little",
    "look straight at me again",
)


def _robot_frames(robot, count: int, gap: float, console: Console):
    """Yield frames from Spot's front camera, telling the operator what to do.

    You do this standing in front of a robot, not watching a terminal, so each
    sample is announced before it is taken rather than reported after. Silent
    capture would mean holding still at the wrong moments.
    """
    from .robot.follow import decode_jpeg

    console.print("[bold]Starting in 3 seconds -- get in front of the robot.[/bold]")
    time.sleep(3.0)

    for index in range(count):
        prompt = POSE_PROMPTS[index % len(POSE_PROMPTS)]
        console.print(f"\n[bold cyan]{index + 1}/{count}:[/bold cyan] {prompt}...")
        time.sleep(gap)

        capture = robot.capture_image("front")
        if not capture.ok or not capture.image_jpeg:
            console.print(f"[yellow]  camera returned nothing: {capture.message}[/yellow]")
            continue
        frame = decode_jpeg(capture.image_jpeg)
        if frame is None:
            console.print("[yellow]  could not decode that frame[/yellow]")
            continue
        yield frame


def _ensure_standing(robot, console: Console) -> int | None:
    """Get Spot upright before sampling. Returns an exit code on failure.

    A sitting Spot has its front cameras at about knee height, aimed down. A
    person standing a metre away is then simply not in the picture, and every
    sample comes back "no face found" while the operator is doing everything
    right. The first run of this hit exactly that, and the advice it printed --
    stand closer, better light -- sent them looking in the wrong place entirely.

    Docked is left alone rather than auto-undocked: driving off a charger is a
    bigger action than standing up, and it should be an explicit request.
    """
    try:
        if robot.is_docked():
            console.print(
                "[red]Spot is on the dock.[/red] Undock it first -- on the dock "
                "its cameras are low and pointing the wrong way.\n"
                "Say 'undock' in the voice loop, or use the tablet."
            )
            return 2
    except Exception:
        LOGGER.debug("dock state unavailable", exc_info=True)

    console.print("Standing Spot up so its cameras are at head height...")
    try:
        result = robot.stand()
    except Exception as exc:
        console.print(f"[red]Could not stand Spot up: {exc}[/red]")
        return 1
    if not result.ok:
        console.print(f"[red]Could not stand Spot up: {result.message}[/red]")
        return 1

    # Settling takes a moment; sampling mid-rise gets a tilted, blurred frame.
    time.sleep(STAND_SETTLE_SEC)
    return None


def enroll(
    name: str,
    config,
    console: Console,
    robot,
    samples: int = SAMPLE_COUNT,
    gap: float = SAMPLE_GAP_SEC,
) -> int:
    """Enroll ``name`` from Spot's front camera. Returns a process exit code.

    Args:
        name: The identity to record. Reuse the same name to add more samples.
        config: Loaded :class:`~spot_voice.config.Config`.
        console: Rich console for progress.
        robot: A connected robot. Required -- see the module docstring for why
            there is no webcam fallback.
        samples: How many frames to try to capture.
        gap: Seconds between captures.
    """
    name = (name or "").strip()
    if not name:
        console.print("[red]Give a name to enroll, e.g. --enroll umar[/red]")
        return 2

    if robot is None:
        console.print(
            "[red]Enrollment needs Spot's own camera.[/red]\n"
            "Recognition runs on Spot's greyscale fisheye frames, so the sample "
            "has to come through the same lens -- a webcam enrollment would look "
            "fine here and then fail to recognise you on the day.\n"
            "Set MOCK_ROBOT=false, get on the robot's wifi, and try again. "
            "Check the path first with: python -m spot_voice --check"
        )
        return 2

    try:
        from .vision.faces import FaceRecogniser

        recogniser = FaceRecogniser()
    except ImportError as exc:
        console.print(
            f"[red]Face recognition is not installed ({exc}).[/red]\n"
            "Run: pip install insightface onnxruntime"
        )
        return 1
    except Exception as exc:
        console.print(f"[red]Could not start the face recogniser: {exc}[/red]")
        return 1

    posture = _ensure_standing(robot, console)
    if posture is not None:
        return posture

    store = FaceStore(config.face_store_path)
    console.print(
        f"Enrolling [bold]{name}[/bold] from Spot's front camera.\n"
        "Stand about a metre in front of the robot, facing it, well lit, and "
        "move your head slightly between samples.\n"
    )

    frames = _robot_frames(robot, samples, gap, console)

    captured = 0
    for frame in frames:
        try:
            faces = recogniser.detect(frame)
        except Exception as exc:
            console.print(f"[yellow]  detection failed: {exc}[/yellow]")
            continue
        if not faces:
            console.print(
                "[yellow]  no face found -- get closer and face the robot[/yellow]"
            )
            continue
        if len(faces) > 1:
            # Ambiguous: enrolling the wrong face would be worse than skipping.
            console.print(
                "[yellow]  more than one face in frame -- skipped. "
                "Enroll with nobody else in shot.[/yellow]"
            )
            continue
        _box, embedding = faces[0]
        store.add(name, embedding)
        captured += 1
        console.print(f"[green]  got it ({captured} so far)[/green]")

    if captured == 0:
        console.print(
            "\n[red]No usable samples.[/red] In rough order of likelihood:\n"
            "  1. Spot was not upright, so its cameras were aimed at the floor. "
            "Check it is standing, not sitting or lying down.\n"
            "  2. You were too far away. Spot's fisheye lenses are very "
            "wide-angle, so a face shrinks fast with distance -- stand about a "
            "metre away, square on to the front of the robot.\n"
            "  3. Too little light, or you were lit from behind.\n"
            "  4. Somebody else was in shot, which is skipped deliberately "
            "rather than risk enrolling the wrong face."
        )
        return 1

    store.save()
    console.print(
        f"\n[green]Enrolled {name} with {captured} sample(s).[/green]\n"
        f"Stored as embeddings in {store.path} -- no photographs, nothing uploaded."
    )
    console.print(f"Known faces: {', '.join(store.names)}")
    console.print(
        f"\nSet [bold]OPERATOR_NAME={name}[/bold] in .env so follow-me looks for you."
    )
    return 0


def forget(name: str, config, console: Console) -> int:
    """Remove an enrolled identity. Returns a process exit code."""
    store = FaceStore(config.face_store_path)
    if store.forget(name.strip()):
        store.save()
        console.print(f"[green]Removed {name}.[/green]")
        return 0
    console.print(
        f"[yellow]{name} is not enrolled.[/yellow] Known: "
        + (", ".join(store.names) or "nobody")
    )
    return 1
