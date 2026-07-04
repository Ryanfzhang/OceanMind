from __future__ import annotations

import json
from typing import Any, List

from packages.llm_gateway.openai_compatible_client import OpenAICompatibleClientAdapter


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def needs_english_translation(text: Any) -> bool:
    return isinstance(text, str) and bool(text.strip()) and contains_cjk(text)


def translate_strings_to_english(
    *,
    adapter: OpenAICompatibleClientAdapter,
    client: Any,
    texts: List[str],
    request_name: str,
    max_tokens: int = 1800,
    temperature: float = 0.0,
) -> List[str]:
    if not texts:
        return []

    payload = {
        "texts": texts,
        "required_output_schema": {
            "translations": ["string"],
        },
    }
    system = (
        "Translate every input string into English only.\n"
        "Do not summarize, omit, or reinterpret.\n"
        "Preserve all numbers, dates, years, coordinates, p-values, result IDs, tool names, units, acronyms, "
        "variable names, and code-like tokens exactly.\n"
        "Return JSON only."
    )

    response = adapter.create_message(
        client=client,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}],
        request_name=request_name,
        json_response=True,
    )
    parsed = adapter.parse_json_response(adapter.extract_response_text(response))
    translations = parsed.get("translations")
    if not isinstance(translations, list) or len(translations) != len(texts):
        raise ValueError("English translation response returned an invalid translations array.")
    normalized: List[str] = []
    for item in translations:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("English translation response contained an empty translated string.")
        normalized.append(item.strip())
    return normalized
