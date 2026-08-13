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


#: How much bigger the subject's face must be than the next one in frame before
#: it is safe to assume the big one is the person enrolling.
#:
#: Apparent area falls off with the square of distance, so someone at a metre
#: is four times the area of someone at two metres. Requiring double means the
#: runner-up has to be within about 1.4x the subject's distance to cause a
#: skip -- which is to say, standing more or less alongside them, where the
#: frame genuinely is ambiguous and refusing is the right answer.
DOMINANT_FACE_RATIO = 2.0


def _area(box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box[:4]
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def pick_subject(faces, anchor=None):
    """Choose which face in the frame is the person being enrolled.

    Args:
        faces: ``[(box, embedding), ...]`` as returned by the recogniser.
        anchor: Embedding of the first accepted sample, when there is one.

    Returns:
        ``(chosen_face, "")`` or ``(None, reason)``.

    Refusing outright whenever a second face appeared was the original rule,
    and it is too strict to survive a real workplace: colleagues walk through
    shot and four of five samples get thrown away, leaving an identity built
    from a single frame. A single-sample enrolment is exactly the brittle
    result the five prompts exist to avoid.

    Two things make picking safe instead. The subject is deliberately close, so
    they are much the largest face and a clear size margin identifies them. And
    once one sample is accepted, every later one has to look like it -- so even
    a wrong pick cannot mix a second person into the same identity.
    """
    from .vision.faces import MATCH_THRESHOLD, cosine_similarity

    if not faces:
        return None, "no face found"
    if len(faces) == 1:
        only = faces[0]
        if anchor is not None and cosine_similarity(anchor, only[1]) < MATCH_THRESHOLD:
            return None, "that doesn't look like the same person -- skipped"
        return only, ""

    if anchor is not None:
        # Already know who we are enrolling: take whoever matches, ignore the rest.
        matches = [
            face for face in faces if cosine_similarity(anchor, face[1]) >= MATCH_THRESHOLD
        ]
        if len(matches) == 1:
            return matches[0], ""
        if not matches:
            return None, "you're not in this frame -- skipped"
        return None, "two faces both look like you -- skipped"

    ranked = sorted(faces, key=lambda face: _area(face[0]), reverse=True)
    biggest, runner_up = _area(ranked[0][0]), _area(ranked[1][0])
    if runner_up <= 0 or biggest / runner_up >= DOMINANT_FACE_RATIO:
        return ranked[0], ""
    return (
        None,
        "two faces at about the same distance -- skipped, "
        "step closer than anyone else in shot",
    )


#: Image types accepted when enrolling from a folder of photos.
PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def enroll_from_photos(name: str, config, console: Console, folder) -> int:
    """Enroll from a folder of ordinary photographs.

    The alternative to standing in front of the robot, and often the better
    one: five varied photos off a phone take thirty seconds, where five robot
    samples need a clear room and a colleague to not walk through shot.

    Robot-camera samples still match slightly better, because they come through
    the lens that will do the recognising. The gap is smaller than it sounds --
    insightface aligns every face by its landmarks before embedding it, which
    normalises most of the fisheye geometry and the loss of colour -- but it is
    real, so mixing a couple of robot samples in with the photos beats either
    alone. Reusing the same name adds to an identity rather than replacing it,
    so both routes compose.

    Photos should be of one person: crops of a group shot are fine, a group
    shot is not, because there is no anchor to say which face is yours.
    """
    from pathlib import Path

    folder = Path(folder).expanduser()
    if not folder.exists():
        console.print(f"[red]No such folder: {folder}[/red]")
        return 2
    if folder.is_file():
        photos = [folder]
    else:
        photos = sorted(
            path
            for path in folder.iterdir()
            if path.suffix.lower() in PHOTO_SUFFIXES
        )
    if not photos:
        console.print(
            f"[red]No images in {folder}.[/red] "
            f"Looking for: {', '.join(PHOTO_SUFFIXES)}"
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

    import cv2

    store = FaceStore(config.face_store_path)
    console.print(f"Enrolling [bold]{name}[/bold] from {len(photos)} photo(s).\n")

    captured = 0
    anchor: list[float] | None = None
    for photo in photos:
        frame = cv2.imread(str(photo))
        if frame is None:
            console.print(f"[yellow]{photo.name}: could not read[/yellow]")
            continue
        try:
            faces = recogniser.detect(frame)
        except Exception as exc:
            console.print(f"[yellow]{photo.name}: detection failed ({exc})[/yellow]")
            continue

        chosen, reason = pick_subject(faces, anchor)
        if chosen is None:
            console.print(f"[yellow]{photo.name}: {reason}[/yellow]")
            continue
        _box, embedding = chosen
        if anchor is None:
            anchor = embedding
        store.add(name, embedding)
        captured += 1
        console.print(f"[green]{photo.name}: got it ({captured} so far)[/green]")

    if captured == 0:
        console.print(
            "\n[red]No usable faces.[/red] Photos need one clear, front-facing "
            "face each, reasonably close and well lit."
        )
        return 1

    store.save()
    console.print(
        f"\n[green]Enrolled {name} with {captured} sample(s).[/green]\n"
        f"Stored as embeddings in {store.path} -- no photographs, nothing uploaded.\n"
        f"Known faces: {', '.join(store.names)}\n\n"
        f"Set OPERATOR_NAME={name} in .env so follow-me looks for you."
    )
    return 0


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
    #: The first accepted face. Every later sample must match it, so a busy
    #: room cannot quietly mix two people into one identity.
    anchor: list[float] | None = None
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

        chosen, reason = pick_subject(faces, anchor)
        if chosen is None:
            console.print(f"[yellow]  {reason}[/yellow]")
            continue

        _box, embedding = chosen
        if anchor is None:
            anchor = embedding
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
