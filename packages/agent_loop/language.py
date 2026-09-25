"""Keep the conversation language anchored to the current user question."""

from __future__ import annotations

import re
from typing import Literal


Language = Literal["en", "zh"]
_HAN = re.compile(r"[\u3400-\u9fff]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")


def preferred_language(text: str) -> Language:
    """Use the dominant writing system; technical English terms can occur in Chinese."""
    han = len(_HAN.findall(text))
    latin_words = len(_LATIN_WORD.findall(text))
    return "zh" if han >= max(2, latin_words / 2) else "en"


def language_instruction(language: Language) -> str:
    name = "Chinese" if language == "zh" else "English"
    return (
        f"\nThe current user question is in {name}. Write the final answer and any "
        f"clarification question in {name}, regardless of the languages in tools, "
        "sources, code, or previous turns. Preserve source titles and quotations "
        "where needed. Analysis stage titles remain in English for the workspace UI."
    )
