"""Reflex matcher: the safety lane must catch stop words and not fire on prose."""

from __future__ import annotations

import time

import pytest

from spot_voice.safety.reflex import (
    ReflexAction,
    ReflexEngine,
    match_reflex,
    normalise,
)


class FakeResult:
    def __init__(self, ok: bool = True, message: str = "Stopped.") -> None:
        self.ok = ok
        self.message = message


class FakeRobot:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def stop_all(self) -> FakeResult:
        self.calls.append("stop_all")
        return FakeResult(True, "Stopped.")

    def sit(self) -> FakeResult:
        self.calls.append("sit")
        return FakeResult(True, "Sitting.")


class FakeFollow:
    def __init__(self, active: bool = False) -> None:
        self._active = active
        self.stopped = False

    @property
    def active(self) -> bool:
        return self._active

    def stop(self):
        self.stopped = True
        self._active = False
        return True, "Stopped following."


# ----------------------------------------------------------------------
# Normalisation


def test_normalise_strips_punctuation_and_case():
    assert normalise("  STOP!! ") == ["stop"]
    assert normalise("Sit, down.") == ["sit", "down"]
    assert normalise("") == []
    assert normalise(None) == []  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Positive matches


@pytest.mark.parametrize(
    "transcript",
    [
        "stop",
        "Stop!",
        "STOP",
        "okay stop",
        "spot stop",
        "freeze",
        "halt",
        "emergency stop",
        "whoa",
        "hold up",
        "stop now",
    ],
)
def test_stop_words_match(transcript):
    found = match_reflex(transcript)
    assert found is not None, transcript
    assert found.action is ReflexAction.STOP


@pytest.mark.parametrize("transcript", ["sit", "sit down", "Sit down please", "lie down"])
def test_sit_words_match(transcript):
    found = match_reflex(transcript)
    assert found is not None, transcript
    assert found.action is ReflexAction.SIT


@pytest.mark.parametrize(
    "transcript",
    ["stop following", "stop following me", "quit following", "stay there"],
)
def test_stop_follow_matches_and_wins_over_plain_stop(transcript):
    found = match_reflex(transcript)
    assert found is not None, transcript
    assert found.action is ReflexAction.STOP_FOLLOW


def test_transcription_slips_still_match():
    # faster-whisper occasionally lands one letter off; the safety lane must
    # tolerate that rather than silently ignore a stop word.
    for slip in ("stopp", "stoppp", "stopped", "sitt down", "freezing"):
        assert match_reflex(slip) is not None, slip


# ----------------------------------------------------------------------
# Negative matches


@pytest.mark.parametrize(
    "transcript",
    [
        "",
        "   ",
        "go to the loading bay",
        "what do you see",
        "take a photo of the control panel",
        "how much battery do you have",
        "walk forward two metres",
        "shop floor",  # 'shop' must not read as 'stop'
        "top of the stairs",  # 'top' must not read as 'stop'
        "hello there",
        "start following me",
    ],
)
def test_ordinary_speech_does_not_trigger(transcript):
    assert match_reflex(transcript) is None, transcript


def test_matcher_biases_towards_stopping():
    # Documented, deliberate: a negated stop still stops. A spurious stop is an
    # inconvenience; a missed stop is a safety incident.
    assert match_reflex("don't stop").action is ReflexAction.STOP  # type: ignore[union-attr]


def test_matching_is_fast_enough_for_the_300ms_budget():
    # Worst case is a no-match: every phrase is fuzzy-scored against every window.
    iterations = 200
    started = time.perf_counter()
    for _ in range(iterations):
        match_reflex("go to the compressor room and tell me what you see")
    per_call_ms = (time.perf_counter() - started) * 1000.0 / iterations
    # Matching must be a rounding error against the 300 ms transcript-to-action
    # budget; the robot command itself is what should own that time.
    assert per_call_ms < 10.0, f"{per_call_ms:.2f} ms per match"


# ----------------------------------------------------------------------
# Engine


def test_engine_stop_cancels_follow_and_stops_robot():
    robot, follow = FakeRobot(), FakeFollow(active=True)
    aborted: list[bool] = []
    engine = ReflexEngine(robot, follow, say=None, on_abort=lambda: aborted.append(True))

    outcome = engine.handle("stop")

    assert outcome is not None
    assert outcome.ok
    assert follow.stopped
    assert "stop_all" in robot.calls
    assert aborted == [True]
    assert outcome.latency_ms < 300.0


def test_engine_sit_stops_follow_first():
    robot, follow = FakeRobot(), FakeFollow(active=True)
    engine = ReflexEngine(robot, follow)

    outcome = engine.handle("sit down")

    assert outcome is not None and outcome.ok
    assert follow.stopped
    assert robot.calls == ["sit"]


def test_engine_stop_follow_when_not_following():
    robot, follow = FakeRobot(), FakeFollow(active=False)
    engine = ReflexEngine(robot, follow)

    outcome = engine.handle("stop following")

    assert outcome is not None and outcome.ok
    assert "wasn't following" in outcome.message
    assert "stop_all" in robot.calls


def test_engine_returns_none_for_ordinary_speech():
    engine = ReflexEngine(FakeRobot(), FakeFollow())
    assert engine.handle("go to the loading bay") is None


def test_engine_never_raises_when_the_robot_fails():
    class BrokenRobot:
        def stop_all(self):
            raise RuntimeError("radio died")

    engine = ReflexEngine(BrokenRobot(), None)
    outcome = engine.handle("stop")

    assert outcome is not None
    assert outcome.ok is False
    assert outcome.message


def test_engine_works_without_any_network_dependency():
    # The reflex lane must not touch the Anthropic client at all: constructing
    # it with no brain, no speaker and no follow controller still works.
    engine = ReflexEngine(FakeRobot())
    assert engine.handle("freeze") is not None
