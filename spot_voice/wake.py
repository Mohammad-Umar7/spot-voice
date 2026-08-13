"""Addressing: deciding whether an utterance was aimed at the robot.

A microphone in a facility hears everything, and until now everything it heard
that was not a safety word went to the language model as a command. On a real
walkthrough that is wrong most of the time -- you are talking to the people
around you far more than you are talking to the robot.

So a command has to be addressed to it by name: "Spot, stand up". Anything else
is treated as conversation and ignored.

**Safety words are deliberately not gated by this.** They are matched earlier, in
:mod:`spot_voice.safety.reflex`, and never reach this module. If someone shouts
"stop" because a leg is about to come down on a cable, requiring them to
remember the robot's name first would be indefensible. The name exists to stop
false *commands*, not to stand between a person and an emergency stop.

The matching is fuzzy for the same reason the reflex matcher is: speech-to-text
mishears short words constantly, and "Spa, stand up" should still work. It
reuses the reflex tokeniser and similarity so a name is judged by exactly the
same rules a safety word is.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from .safety.reflex import REFLEX_RULES, _token_similarity, normalise

LOGGER = logging.getLogger(__name__)

#: What the robot answers to when nothing else is configured.
DEFAULT_WAKE_WORD = "spot"

#: How far into an utterance the name may appear and still count as addressing.
#: Three allows "hey Spot", "ok Spot" and "um, Spot" without allowing a name
#: mentioned in the middle of a sentence about the robot to trigger it.
LEAD_TOKENS = 3

#: How close a word must sound to the name. Higher than the reflex threshold on
#: purpose: a missed command is a repeat, whereas a missed "stop" is an
#: incident, so the two errors are not worth trading off the same way.
NAME_SIMILARITY = 0.82

#: Words that may precede the name without breaking the addressing.
LEAD_INS = frozenset({"hey", "hi", "ok", "okay", "yo", "hello", "um", "uh", "so"})

#: Every word that appears in a reflex phrase. None of these may ever be taken
#: for the robot's name, whatever the similarity score says.
#:
#: This is a hard floor rather than a tuning choice. "stop" and "spot" differ by
#: one transposition and score 0.50 against each other, while "spa" -- a real
#: mishearing of the name -- scores 0.57. Those are seven points apart, which is
#: far too thin a margin to defend by threshold alone. So the threshold stays
#: high enough to reject "stop" comfortably, and this set makes it impossible
#: even if someone later lowers it, or configures the name *to* a safety word.
SAFETY_WORDS = frozenset(
    token
    for rule in REFLEX_RULES
    for phrase in rule.phrases
    for token in phrase.split()
)

_NON_WORD = re.compile(r"[^\w]+")


@dataclass
class Address:
    """The verdict on one utterance."""

    #: True when this was aimed at the robot and should be acted on.
    addressed: bool
    #: The utterance with the name and any lead-in removed, ready for the model.
    command: str
    #: Why it was accepted, for the log. Empty when it was not.
    via: str = ""

    @property
    def is_bare_name(self) -> bool:
        """True for "Spot" on its own -- addressed, but no instruction yet."""
        return self.addressed and not self.command.strip()


class WakeGate:
    """Decides whether an utterance was addressed to the robot.

    Args:
        wake_word: The name to answer to.
        follow_up_sec: How long after the robot replies a follow-up is accepted
            without repeating the name. Zero -- the default -- means the name is
            always required, which is the conservative choice: a follow-up
            window reopens exactly the false-trigger problem the name is here to
            close, because the sentence you say to a colleague right after the
            robot answers is inside the window.
        enabled: When False every utterance is treated as addressed, which is
            the behaviour this class replaced.
    """

    def __init__(
        self,
        wake_word: str = DEFAULT_WAKE_WORD,
        follow_up_sec: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self._name = (wake_word or DEFAULT_WAKE_WORD).strip().lower()
        self._follow_up_sec = max(0.0, follow_up_sec)
        self._enabled = enabled
        self._replied_at = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    def note_reply(self) -> None:
        """Record that the robot just answered, opening the follow-up window."""
        self._replied_at = time.monotonic()

    def check(self, text: str) -> Address:
        """Decide whether ``text`` was addressed to the robot."""
        if not self._enabled:
            return Address(addressed=True, command=text, via="gate-disabled")

        stripped = self._strip_name(text)
        if stripped is not None:
            return Address(addressed=True, command=stripped, via="name")

        if self._follow_up_sec and (
            time.monotonic() - self._replied_at <= self._follow_up_sec
        ):
            return Address(addressed=True, command=text, via="follow-up")

        # The name did not survive transcription. This is not an edge case:
        # "Spot, stand up" came back from Whisper as "Peaceful stand up", which
        # scores 0.17 against the name -- no threshold reaches that, and one
        # low enough would swallow "sit" and "stop", which score 0.57 and 0.50.
        #
        # So a plain instruction is accepted on its own. It is a weaker signal
        # than the name and deliberately so: losing every command to a
        # mis-heard name is a worse failure than occasionally acting on a
        # sentence that was not meant for the robot. Conversation that contains
        # no instruction -- "yeah", "keep", "I think the battery is low" --
        # still goes nowhere.
        if _is_instruction(text):
            return Address(addressed=True, command=text, via="instruction")

        return Address(addressed=False, command="")

    # ------------------------------------------------------------------

    def _strip_name(self, text: str) -> str | None:
        """The utterance minus its leading name, or None if it has none.

        Works on the original words rather than the normalised tokens so the
        command handed onward keeps its original spelling and punctuation.
        """
        words = (text or "").split()
        if not words:
            return None

        for index, word in enumerate(words[:LEAD_TOKENS]):
            bare = _NON_WORD.sub("", word.lower())
            if not bare or bare in LEAD_INS:
                continue
            if bare in SAFETY_WORDS:
                # Never claim a safety word as the name, however close it
                # sounds. It belongs to the reflex lane, which has already had
                # its chance at it, and swallowing it here would be silent.
                continue
            if _token_similarity(bare, self._name) >= NAME_SIMILARITY:
                return " ".join(words[index + 1 :]).lstrip(",. ").strip()
            # Keep scanning the rest of the lead window rather than giving up
            # here. Bailing on the first unrecognised word meant only the very
            # first word was ever really checked, so "let's Spot undock" was
            # rejected on the word "let's".
        return None


#: Words that only really appear when somebody is instructing a robot.
#:
#: Chosen to be things you say *to* a machine rather than about one. "Battery"
#: and "status" are here; "he", "it" and "think" are not. The test is whether
#: the word would show up in a sentence aimed at a colleague, because that is
#: the traffic this gate exists to ignore.
COMMAND_WORDS = frozenset(
    {
        "stand", "standup", "sit", "undock", "dock",
        "walk", "come", "go", "move", "forward", "backward", "backwards",
        "turn", "left", "right", "power", "powers", "motors",
        "bow", "nod", "wave", "spin", "dance", "rotate",
        "scan", "around", "count",
        "photo", "picture", "look", "see", "camera", "status", "battery",
        "waypoint", "navigate", "wait", "hold", "settle", "rest",
    }
)

#: An utterance shorter than this is noise or a filler word, never an order.
MIN_INSTRUCTION_TOKENS = 1

#: Longest utterance that may be admitted on a command word alone.
#:
#: Orders are short: "stand up", "sit down", "scan the room". Explaining the
#: robot to a room is not, and during a live demo the sentence "and yeah, there
#: are multiple capabilities, so it follows us as well" was admitted on the word
#: "follows" and answered with stop_all. Length separates the two far better
#: than vocabulary does, because the giveaway is not which words were used but
#: how many. Anything longer still works -- it just has to be addressed by name.
MAX_INSTRUCTION_TOKENS = 8


#: How close a word must be to a command word. Matched fuzzily for the same
#: reason the name is: "Spot, undock" came back as "Let's put andock", and an
#: exact-match list would have thrown that away too.
COMMAND_SIMILARITY = 0.8

#: Shortest word worth fuzzy-matching against the command list.
MIN_FUZZY_LEN = 4


def _is_instruction(text: str) -> bool:
    """True when the utterance reads as an order rather than conversation."""
    tokens = normalise(text)
    if not (MIN_INSTRUCTION_TOKENS <= len(tokens) <= MAX_INSTRUCTION_TOKENS):
        return False
    for token in tokens:
        if token in COMMAND_WORDS:
            return True
        # Fuzzy matching only on longer words. Short ones are far too easy to
        # collide with ordinary speech -- "no" scores 0.95 against "nod" on the
        # prefix rule, which turned "no I don't think so" into an order.
        if len(token) < MIN_FUZZY_LEN:
            continue
        if any(
            len(command) >= MIN_FUZZY_LEN
            and _token_similarity(token, command) >= COMMAND_SIMILARITY
            for command in COMMAND_WORDS
        ):
            return True
    return False


def sounds_like_name(word: str, wake_word: str = DEFAULT_WAKE_WORD) -> bool:
    """Whether one word would be accepted as the robot's name.

    Exposed for the tests that check the name cannot collide with a safety word.
    """
    tokens = normalise(word)
    if not tokens:
        return False
    if tokens[0] in SAFETY_WORDS:
        return False
    return _token_similarity(tokens[0], wake_word.lower()) >= NAME_SIMILARITY
