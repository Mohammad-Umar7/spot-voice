"""Addressing: the robot acts on "Spot, do X" and ignores everything else.

The safety half of this is the important half. A wake word that could swallow a
spoken "stop" would be worse than having no wake word at all, so several of
these tests exist purely to prove it cannot.
"""

from __future__ import annotations

import time

from spot_voice.wake import (
    DEFAULT_WAKE_WORD,
    Address,
    WakeGate,
    sounds_like_name,
)


def gate(**kwargs) -> WakeGate:
    return WakeGate(**kwargs)


# ----------------------------------------------------------------------
# Being addressed


def test_a_command_starting_with_the_name_is_accepted_and_the_name_removed():
    result = gate().check("Spot, stand up")

    assert result.addressed
    assert result.command == "stand up"
    assert result.via == "name"


def test_a_lead_in_before_the_name_still_counts():
    for phrase in ("hey Spot stand up", "ok Spot, stand up", "um Spot stand up"):
        result = gate().check(phrase)
        assert result.addressed, phrase
        assert result.command == "stand up", phrase


def test_speech_to_text_mangling_the_name_still_counts():
    """The name is one short word, which is exactly what Whisper fumbles."""
    for heard in ("Spot. Stand up", "spott stand up", "Sport, stand up", "pot stand up"):
        assert gate().check(heard).addressed, heard


def test_mishearings_too_close_to_a_safety_word_are_refused():
    """"spa" is a real mishearing of the name and is rejected anyway.

    It scores 0.57 against "spot" while "stop" scores 0.50 -- seven points
    apart. Accepting "spa" would mean a threshold low enough to be one bad
    frame away from accepting "stop", and a missed command costs a repeat where
    a missed stop costs an incident. So this one is deliberately not supported.
    """
    result = gate().check("Spa, stand up")
    # It is still let through -- but as an instruction ("stand"), never as the
    # name. The name route stays strict so it cannot drift toward "stop".
    assert result.via != "name"
    assert gate()._strip_name("Spa, stand up") is None


def test_the_name_on_its_own_is_addressed_but_carries_no_command():
    result = gate().check("Spot?")

    assert result.addressed
    assert result.is_bare_name
    assert result.command == ""


def test_the_command_keeps_its_original_wording():
    """Only the name is stripped -- the rest reaches the model untouched."""
    result = gate().check("Spot, go to the loading bay and wait")
    assert result.command == "go to the loading bay and wait"


# ----------------------------------------------------------------------
# Not being addressed


def test_talking_to_a_colleague_is_ignored():
    for phrase in (
        "can you pass me the tablet",
        "yeah",
        "no I don't think so",
        "give me a minute",
    ):
        result = gate().check(phrase)
        assert not result.addressed, phrase
        assert result.command == ""


def test_the_name_in_the_middle_of_a_sentence_does_not_summon_it():
    """Talking *about* the robot is not talking *to* it."""
    assert not gate().check("I was telling him the Spot can climb stairs").addressed


def test_empty_and_whitespace_are_not_addressed():
    assert not gate().check("").addressed
    assert not gate().check("   ").addressed
    assert not gate().check(None).addressed


# ----------------------------------------------------------------------
# Safety: the wake word must never intercept a safety word
#
# "stop" and "spot" differ by one transposition, which is uncomfortably close
# for two words with opposite consequences.


def test_stop_is_not_mistaken_for_the_robots_name():
    assert not sounds_like_name("stop")
    assert not sounds_like_name("stopped")
    assert not sounds_like_name("stop!")


def test_a_bare_safety_word_is_not_addressed_so_it_can_never_be_consumed_here():
    """The reflex lane runs first; this proves the gate would decline it anyway.

    Belt and braces. If someone ever reorders ``VoiceApp.handle`` and the gate
    ends up ahead of the reflex lane, "stop" must still not be eaten silently.
    """
    for word in ("stop", "freeze", "halt", "stop now", "emergency stop"):
        assert not gate().check(word).addressed, word


def test_the_reflex_lane_is_reached_before_the_wake_gate():
    """Ordering inside VoiceApp.handle is a safety property, so pin it.

    A safety word must be matched before anything can decide the utterance was
    not addressed to the robot.
    """
    import inspect

    from spot_voice.main import VoiceApp

    source = inspect.getsource(VoiceApp.handle)
    assert "self.reflex.handle" in source
    assert "self.wake.check" in source
    assert source.index("self.reflex.handle") < source.index("self.wake.check")


# ----------------------------------------------------------------------
# Configuration


def test_the_gate_can_be_turned_off_entirely():
    result = gate(enabled=False).check("stand up")

    assert result.addressed
    assert result.command == "stand up"
    assert result.via == "gate-disabled"


def test_a_follow_up_needs_the_name_again_by_default():
    """Default is strict, because the window reopens the very problem the name
    is here to close: the sentence you say to a colleague straight after the
    robot answers falls inside it."""
    engine = gate()
    engine.note_reply()

    assert not engine.check("yeah do that one").addressed


def test_a_follow_up_window_can_be_opened_when_wanted():
    engine = gate(follow_up_sec=5.0)
    engine.note_reply()

    result = engine.check("yeah do that one")
    assert result.addressed
    assert result.via == "follow-up"
    assert result.command == "yeah do that one"


def test_the_follow_up_window_closes():
    engine = gate(follow_up_sec=0.2)
    engine.note_reply()
    time.sleep(0.35)

    assert not engine.check("yeah do that one").addressed


def test_a_custom_name_can_be_configured():
    engine = gate(wake_word="rex")

    assert engine.check("Rex, sit").via == "name"
    assert engine._strip_name("Spot, sit") is None


def test_the_default_name_is_the_one_documented_in_the_example_env():
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / ".env.example"
    if not example.exists():  # pragma: no cover - example is optional
        return
    text = example.read_text(encoding="utf-8")
    if "WAKE_WORD" in text:
        line = next(l for l in text.splitlines() if l.startswith("WAKE_WORD"))
        assert line.split("=", 1)[1].strip().lower() == DEFAULT_WAKE_WORD


def test_address_is_a_plain_value_with_no_surprises():
    empty = Address(addressed=False, command="")
    assert not empty.is_bare_name
