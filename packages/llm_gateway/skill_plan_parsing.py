"""Response parsing helpers for skill planning."""

from __future__ import annotations

from typing import Any, Dict, Optional

from packages.llm_gateway.openai_compatible_client import OpenAICompatibleClientAdapter


def extract_response_text(response: Any) -> str:
    return OpenAICompatibleClientAdapter.extract_response_text(response)


def parse_json_response(text: str) -> Dict[str, Any]:
    return OpenAICompatibleClientAdapter.parse_json_response(text)


def looks_like_json_parse_error(error: Optional[str]) -> bool:
    if not error:
        return False
    lowered = str(error).lower()
    return (
        "not valid json" in lowered
        or "valid json" in lowered
        or "expecting" in lowered
        or "unterminated" in lowered
        or "jsondecodeerror" in lowered
    )
