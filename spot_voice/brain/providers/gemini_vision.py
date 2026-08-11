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
        response = self._model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": image_jpeg}]
        )
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            raise RuntimeError("Gemini returned no description")
        LOGGER.info("Gemini described a frame in %d characters", len(text))
        return text


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
