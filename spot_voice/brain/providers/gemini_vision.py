"""Gemini vision provider -- looks at a camera frame and says what is there.

Used when the tool-calling provider is text-only (Groq). The agent hands the
JPEG here, gets a written description back, and puts that in the ``tool_result``
where the image would otherwise have gone. From the tool-calling model's point
of view, ``capture_image`` simply returns prose.

That is a genuine behaviour difference from running everything on Anthropic,
where the model looks at the photo itself: here, whatever the vision model fails
to mention is invisible downstream. Worth remembering when an inspection answer
seems thin.
"""

from __future__ import annotations

import logging

from .base import VisionProvider

LOGGER = logging.getLogger(__name__)

#: What the vision model is asked when the operator gave no particular question.
DEFAULT_PROMPT = (
    "You are the eyes of an inspection robot in an industrial facility. "
    "Describe what is visible in this camera frame in two or three sentences. "
    "Lead with anything that matters for an inspection: equipment, gauges and "
    "their readings, signage, obstructions, spills, damage, or people. "
    "Describe only what you can actually see. Do not speculate."
)


class GeminiVisionProvider(VisionProvider):
    """Image understanding via Google's Gemini API."""

    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        # Note: keys come in more than one shape. An "AQ."-prefixed key
        # authenticates just as well as the "AIza" ones AI Studio hands out, so
        # do not warn on the prefix -- it only produces false alarms.
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model
        self._model = genai.GenerativeModel(model)

    def describe(self, image_jpeg: bytes, prompt: str = DEFAULT_PROMPT) -> str:
        """Return a short description of the frame.

        Raises:
            RuntimeError: If the model returns nothing usable.
        """
        try:
            response = self._model.generate_content(
                [prompt, {"mime_type": "image/jpeg", "data": image_jpeg}]
            )
        except Exception as exc:
            raise RuntimeError(explain_gemini_error(exc)) from exc

        text = (getattr(response, "text", "") or "").strip()
        if not text:
            raise RuntimeError("Gemini returned no description")
        LOGGER.info("Gemini described a frame in %d characters", len(text))
        return text


def list_available_models() -> list[str]:
    """Model ids this key can actually use, newest-looking first.

    Asking beats guessing: the ids move faster than any list hardcoded here,
    and a wrong id and a bad key produce near-identical errors.
    """
    try:
        import google.generativeai as genai

        names = [
            model.name.removeprefix("models/")
            for model in genai.list_models()
            if "generateContent" in getattr(model, "supported_generation_methods", [])
        ]
    except Exception:
        LOGGER.debug("could not list Gemini models", exc_info=True)
        return []
    # Prefer flash variants: this runs on a robot, so latency matters more than
    # the last few points of quality on a "what do you see".
    return sorted(names, key=lambda name: (0 if "flash" in name else 1, name))


def explain_gemini_error(exc: BaseException) -> str:
    """Turn a Gemini failure into something that names the actual fix.

    Gemini's errors are famously opaque -- a bad key, a wrong model id and an
    unenabled API all surface as similar-looking exceptions.
    """
    text = str(exc).lower()
    if "api key not valid" in text or "api_key_invalid" in text:
        return (
            "Gemini rejected the API key. Get one from "
            "https://aistudio.google.com/apikey -- it should start with 'AIza'."
        )
    if "permission" in text or "403" in text:
        return (
            "Gemini refused the request. The key may be for a different project, "
            "or the Generative Language API may not be enabled on it."
        )
    if "not found" in text or "404" in text:
        available = list_available_models()
        if available:
            return (
                "Gemini does not have that model. Set GEMINI_MODEL to one of: "
                + ", ".join(available[:6])
            )
        return (
            "Gemini does not recognise that model id. Check GEMINI_MODEL against "
            "https://ai.google.dev/gemini-api/docs/models"
        )
    if "quota" in text or "429" in text or "resource_exhausted" in text:
        return "Gemini is rate limiting or the free quota is used up. Wait and retry."
    return f"Gemini failed: {exc}"


class NullVisionProvider(VisionProvider):
    """Stand-in for when no vision provider is configured.

    Rather than pretend, it says plainly that the robot took a photo but has no
    way to interpret it -- which is a true and speakable thing for the robot to
    report, and much better than a hallucinated description.
    """

    name = "none"

    def describe(self, image_jpeg: bytes, prompt: str = DEFAULT_PROMPT) -> str:
        return (
            "A photo was captured, but no vision model is configured, so its "
            "contents are unknown. Tell the operator you can take pictures but "
            "cannot currently see them."
        )
