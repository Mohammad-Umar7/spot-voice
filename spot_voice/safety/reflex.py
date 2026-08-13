"""The reflex lane: safety words that never reach the language model.

Every transcript is matched here **before** any Anthropic call. A hit executes a
hardcoded robot action immediately -- target is well under 300 ms from transcript
to command -- and the utterance is not forwarded to Claude.

This lane has no dependency on the Anthropic API, the network, or the brain
module. With the internet unplugged, "stop" still stops the robot.

Bias note: the matcher is deliberately tuned to over-trigger rather than
under-trigger. A spurious stop is an inconvenience; a missed stop is a safety
incident. So "don't stop" and "non-stop" will both stop the robot, and that is
the intended behaviour.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import Callable, Sequence

LOGGER = logging.getLogger(__name__)

_NON_WORD = re.compile(r"[^a-z0-9]+")

#: Similarity a single-word phrase must reach to count as a fuzzy hit.
SINGLE_WORD_THRESHOLD = 0.88
#: Average similarity a multi-word phrase must reach.
MULTI_WORD_THRESHOLD = 0.85
#: No single word in a multi-word phrase may score below this, so
#: "start following me" cannot average its way into "stop following me".
MULTI_WORD_MIN_TOKEN = 0.6
#: Two tokens sharing a prefix this long are treated as the same word
#: ("stop" / "stopped", "freeze" / "freezing").
PREFIX_MIN_LEN = 4


class ReflexAction(str, Enum):
    """What a matched safety word does."""

    STOP = "stop"
    SIT = "sit"
    STOP_FOLLOW = "stop_follow"


@dataclass(frozen=True)
class ReflexRule:
    """A set of spoken phrases mapped to one action."""

    action: ReflexAction
    phrases: tuple[str, ...]


#: Rules in priority order. Longer, more specific phrases are checked first so
#: "stop following" resolves to STOP_FOLLOW rather than the broader STOP.
REFLEX_RULES: tuple[ReflexRule, ...] = (
    ReflexRule(
        ReflexAction.STOP,
        (
            # "stay there" and "stop following" used to route to a dedicated
            # STOP_FOLLOW action. Following has been removed, so rather than
            # leave them as no-ops they now do the safe thing and stop outright.
            "stop following",
            "stop following me",
            "stop follow",
            "quit following",
            "don t follow me",
            "stay there",
            "stop",
            "freeze",
            "halt",
            "hold up",
            "stop it",
            "stop now",
            "emergency stop",
            "whoa",
        ),
    ),
    ReflexRule(
        ReflexAction.SIT,
        (
            "sit",
            "sit down",
            "lie down",
            "take a seat",
        ),
    ),
)


@dataclass(frozen=True)
class ReflexMatch:
    """A matched safety word."""

    action: ReflexAction
    phrase: str
    score: float
    exact: bool


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------


def normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation and split a transcript into word tokens."""
    return [token for token in _NON_WORD.sub(" ", (text or "").lower()).split() if token]


def _common_prefix_len(a: str, b: str) -> int:
    """Number of leading characters two words share."""
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def _token_similarity(a: str, b: str) -> float:
    """Similarity of two words, with a prefix rule for inflected forms.

    The prefix rule is what makes "stopped", "stopp" and "freezing" register as
    their base words, which is exactly the shape of error speech-to-text makes.
    """
    if a == b:
        return 1.0
    if _common_prefix_len(a, b) >= PREFIX_MIN_LEN:
        return 0.95
    return SequenceMatcher(None, a, b).ratio()


def _contains_sequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """True when ``needle`` appears as a contiguous run inside ``haystack``."""
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    for index, token in enumerate(haystack):
        if token != first:
            continue
        if list(haystack[index : index + len(needle)]) == list(needle):
            return True
    return False


def _best_window_score(
    tokens: Sequence[str], phrase_tokens: Sequence[str]
) -> tuple[float, float]:
    """Score every same-length window of the transcript against a phrase.

    Returns:
        ``(best_average_similarity, min_token_similarity_of_that_window)``. The
        second value is the anchor check: a window only counts if every word in
        it is at least plausibly the corresponding phrase word.
    """
    size = len(phrase_tokens)
    if size == 0 or not tokens:
        return 0.0, 0.0
    best_average, best_minimum = 0.0, 0.0
    for start in range(0, max(1, len(tokens) - size + 1)):
        window = tokens[start : start + size]
        if len(window) != size:
            continue
        scores = [_token_similarity(a, b) for a, b in zip(window, phrase_tokens)]
        average = sum(scores) / size
        if average > best_average:
            best_average, best_minimum = average, min(scores)
    return best_average, best_minimum


#: Words that turn "sit" from a posture into a destination.
#:
#: "Sit" means settle where you are. "Sit beside me" means walk over here and
#: then settle -- a different action with a different tool behind it. The reflex
#: lane matches by containment, so without this it sees the word "sit", fires
#: instantly, and the robot sits exactly where it was standing.
#:
#: Applied to the sit rule **only**. "Stop here" and "stop over there" are still
#: stops, and must never be deferred to a model round-trip for any reason.
PLACEMENT_WORDS = frozenset(
    {"here", "there", "beside", "next", "near", "by", "with", "over", "alongside"}
)


def _mentions_a_place(tokens: Sequence[str]) -> bool:
    """True when the utterance names somewhere to go, not just a posture."""
    return any(token in PLACEMENT_WORDS for token in tokens)


def match_reflex(
    transcript: str, rules: Sequence[ReflexRule] = REFLEX_RULES
) -> ReflexMatch | None:
    """Match a transcript against the reflex phrases.

    Args:
        transcript: Raw text straight out of the speech-to-text stage.
        rules: Rule set to match against; defaults to :data:`REFLEX_RULES`.

    Returns:
        The matching :class:`ReflexMatch`, or ``None`` when nothing matched.
    """
    tokens = normalise(transcript)
    if not tokens:
        return None

    # "Sit beside me" is a walking instruction wearing the word "sit". Hand it
    # to the model, which has come_here. Stopping is never reinterpreted this
    # way -- see PLACEMENT_WORDS.
    if _mentions_a_place(tokens):
        rules = tuple(rule for rule in rules if rule.action is not ReflexAction.SIT)

    # Pass 1: exact containment. Cheap, and covers the overwhelming majority.
    for rule in rules:
        for phrase in rule.phrases:
            phrase_tokens = normalise(phrase)
            if _contains_sequence(tokens, phrase_tokens):
                return ReflexMatch(rule.action, phrase, 1.0, exact=True)

    # Pass 2: fuzzy, to absorb speech-to-text slips ("stopp", "sitt down").
    for rule in rules:
        for phrase in rule.phrases:
            phrase_tokens = normalise(phrase)
            if not phrase_tokens:
                continue
            single = len(phrase_tokens) == 1
            threshold = SINGLE_WORD_THRESHOLD if single else MULTI_WORD_THRESHOLD
            average, minimum = _best_window_score(tokens, phrase_tokens)
            if average < threshold:
                continue
            if not single and minimum < MULTI_WORD_MIN_TOKEN:
                continue
            return ReflexMatch(rule.action, phrase, average, exact=False)
    return None


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------


@dataclass
class ReflexOutcome:
    """What the reflex lane did, for logging and for the caller."""

    match: ReflexMatch
    ok: bool
    message: str
    latency_ms: float


class ReflexEngine:
    """Matches safety words and executes the mapped action immediately.

    The spoken confirmation is dispatched on a separate thread so that
    synthesising and playing speech never delays the robot command.

    Args:
        robot: The robot to command.
        follow: The follow-me controller, so "stop" can kill it instantly.
        say: Callback that speaks a sentence (may block; it is called off-thread).
        on_abort: Optional callback fired on STOP, used to abandon an in-flight
            Claude tool loop.
    """

    def __init__(
        self,
        robot,
        follow=None,
        say: Callable[[str], None] | None = None,
        on_abort: Callable[[], None] | None = None,
    ) -> None:
        self._robot = robot
        self._follow = follow
        self._say = say
        self._on_abort = on_abort

    def handle(self, transcript: str) -> ReflexOutcome | None:
        """Check a transcript and, on a hit, run the action. Returns the outcome."""
        found = match_reflex(transcript)
        if found is None:
            return None

        started = time.perf_counter()
        try:
            ok_flag, message = self._execute(found.action)
        except Exception as exc:  # the reflex lane must never raise
            LOGGER.exception("Reflex action failed")
            ok_flag, message = False, f"I couldn't stop cleanly: {type(exc).__name__}."
        latency_ms = (time.perf_counter() - started) * 1000.0

        LOGGER.info(
            "REFLEX %s (%r, score %.2f) -> %s in %.0f ms",
            found.action.value,
            found.phrase,
            found.score,
            "ok" if ok_flag else "FAILED",
            latency_ms,
        )

        if self._say is not None and message:
            threading.Thread(
                target=self._speak_safely, args=(message,), daemon=True
            ).start()

        return ReflexOutcome(found, ok_flag, message, latency_ms)

    # ------------------------------------------------------------------

    def _speak_safely(self, message: str) -> None:
        try:
            self._say(message)  # type: ignore[misc]
        except Exception:  # pragma: no cover - speech must never break safety
            LOGGER.debug("reflex speech failed", exc_info=True)

    def _execute(self, action: ReflexAction) -> tuple[bool, str]:
        """Run the hardcoded action for a matched reflex."""
        if action is ReflexAction.STOP:
            if self._on_abort is not None:
                self._on_abort()
            if self._follow is not None and self._follow.active:
                self._follow.stop()
            result = self._robot.stop_all()
            return result.ok, result.message

        if action is ReflexAction.STOP_FOLLOW:
            if self._on_abort is not None:
                self._on_abort()
            if self._follow is not None and self._follow.active:
                self._follow.stop()
                # Still settle the body: the last velocity command may have up to
                # VELOCITY_CMD_DURATION left on it.
                self._robot.stop_all()
                return True, "Stopped following."
            self._robot.stop_all()
            return True, "I wasn't following anyone."

        if action is ReflexAction.SIT:
            if self._follow is not None and self._follow.active:
                self._follow.stop()
            result = self._robot.sit()
            return result.ok, result.message

        return False, "I didn't understand that safety word."  # pragma: no cover
