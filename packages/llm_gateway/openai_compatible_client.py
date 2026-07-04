from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

import httpx

from packages.llm_gateway.config import (
    load_llm_api_key,
    load_llm_base_url,
    should_use_legacy_anthropic_config,
)


class OpenAICompatibleClientAdapter:
    """Provider-neutral LLM adapter with OpenAI-compatible as the primary path."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str,
        client: Optional[Any] = None,
        trust_env: bool = False,
        request_retries: int = 2,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = client
        self.trust_env = trust_env
        self.request_retries = request_retries

    def get_client(self) -> Any:
        if self.client is not None:
            return self.client

        if self._should_use_legacy_anthropic():
            self.client = self._build_anthropic_client()
            return self.client

        self.client = self._build_openai_client()
        return self.client

    def create_message(
        self,
        *,
        client: Any,
        max_tokens: int,
        temperature: float,
        system: str,
        messages: List[Dict[str, Any]],
        request_name: str,
        json_response: bool = False,
    ) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self.request_retries + 1):
            try:
                if self._supports_chat_completions(client):
                    kwargs: Dict[str, Any] = {}
                    if json_response:
                        kwargs["response_format"] = {"type": "json_object"}
                    return client.chat.completions.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=self._build_openai_messages(system, messages),
                        **kwargs,
                    )
                if self._supports_responses_api(client):
                    kwargs = {}
                    if json_response:
                        kwargs["text"] = {"format": {"type": "json_object"}}
                    return client.responses.create(
                        model=self.model,
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        input=self._build_openai_messages(system, messages),
                        **kwargs,
                    )
                if self._supports_anthropic_messages(client):
                    system_payload: Any = system
                    if isinstance(system, str):
                        system_payload = [
                            {
                                "type": "text",
                                "text": system,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                    return client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_payload,
                        messages=messages,
                    )
                raise TypeError("Unsupported LLM client interface.")
            except Exception as exc:
                last_exc = exc
                if attempt < self.request_retries:
                    time.sleep(min(2 ** attempt, 4))
                    continue

        base_url = self.base_url or load_llm_base_url()
        raise RuntimeError(
            f"LLM request failed for {request_name} after {self.request_retries + 1} attempts "
            f"(model={self.model}, base_url={base_url}, error_type={type(last_exc).__name__}, error={last_exc})"
        ) from last_exc

    @staticmethod
    def extract_response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            parts: List[str] = []
            for choice in choices:
                message = getattr(choice, "message", None)
                if message is None and isinstance(choice, dict):
                    message = choice.get("message")
                if message is None:
                    continue
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                parts.extend(OpenAICompatibleClientAdapter._coerce_text_parts(content))
            text = "\n".join(part for part in parts if part).strip()
            if text:
                return text
            detail = OpenAICompatibleClientAdapter._empty_chat_completion_detail(response)
            if detail:
                raise ValueError(detail)

        content = getattr(response, "content", None)
        if content is None and isinstance(response, dict):
            content = response.get("content")
        if content is not None:
            parts = []
            for block in content:
                block_type = getattr(block, "type", None)
                block_text = getattr(block, "text", None)
                if block_type is None and isinstance(block, dict):
                    block_type = block.get("type")
                    block_text = block.get("text")
                if block_type == "text" and block_text is not None:
                    parts.append(str(block_text))
            text = "\n".join(parts).strip()
            if text:
                return text

        raise ValueError("LLM response does not contain text content.")

    @staticmethod
    def _empty_chat_completion_detail(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            return ""

        finish_reasons: List[str] = []
        reasoning_seen = False
        for choice in choices:
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is None and isinstance(choice, dict):
                finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(str(finish_reason))

            message = getattr(choice, "message", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message")
            reasoning_content = None
            if message is not None:
                reasoning_content = getattr(message, "reasoning_content", None)
                if reasoning_content is None and isinstance(message, dict):
                    reasoning_content = message.get("reasoning_content")
            if isinstance(reasoning_content, str) and reasoning_content.strip():
                reasoning_seen = True

        model = getattr(response, "model", None)
        if model is None and isinstance(response, dict):
            model = response.get("model")

        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        reasoning_tokens = OpenAICompatibleClientAdapter._nested_usage_value(
            usage,
            ["completion_tokens_details", "reasoning_tokens"],
        )
        completion_tokens = OpenAICompatibleClientAdapter._nested_usage_value(usage, ["completion_tokens"])

        parts = ["LLM response does not contain final text content"]
        if reasoning_seen:
            parts.append("it contained reasoning_content only")
        if finish_reasons:
            parts.append(f"finish_reason={','.join(sorted(set(finish_reasons)))}")
        if model:
            parts.append(f"model={model}")
        if reasoning_tokens is not None:
            parts.append(f"reasoning_tokens={reasoning_tokens}")
        if completion_tokens is not None:
            parts.append(f"completion_tokens={completion_tokens}")
        return "; ".join(parts) + "."

    @staticmethod
    def _nested_usage_value(value: Any, path: List[str]) -> Any:
        current = value
        for key in path:
            if current is None:
                return None
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
        return current

    @staticmethod
    def parse_json_response(text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned)

        try:
            return OpenAICompatibleClientAdapter._loads_json_with_common_repairs(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("LLM response does not contain valid JSON.")
            try:
                return OpenAICompatibleClientAdapter._loads_json_with_common_repairs(cleaned[start:end + 1])
            except json.JSONDecodeError as extracted_error:
                raise ValueError(
                    "LLM response is not valid JSON: "
                    f"{extracted_error.msg} at line {extracted_error.lineno}, "
                    f"column {extracted_error.colno}."
                ) from extracted_error

    @staticmethod
    def _loads_json_with_common_repairs(candidate: str) -> Any:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as original_error:
            repaired = OpenAICompatibleClientAdapter._repair_common_json_commas(candidate)
            repaired = OpenAICompatibleClientAdapter._repair_common_json_trailing_commas(repaired)
            if repaired != candidate:
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
            raise original_error

    @staticmethod
    def _repair_common_json_commas(candidate: str) -> str:
        """Repair common LLM JSON slips such as missing commas between lines."""
        lines = candidate.splitlines()
        if len(lines) <= 1:
            return candidate

        repaired_lines = list(lines)
        previous_index: Optional[int] = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if previous_index is not None and OpenAICompatibleClientAdapter._needs_json_comma(
                repaired_lines[previous_index].strip(),
                stripped,
            ):
                repaired_lines[previous_index] = OpenAICompatibleClientAdapter._append_json_comma(
                    repaired_lines[previous_index]
                )
            previous_index = index

        repaired = "\n".join(repaired_lines)
        return re.sub(
            r'([}\]"0-9]|true|false|null)(\s+)(?=(["{\[]))',
            r"\1, ",
            repaired,
        )

    @staticmethod
    def _needs_json_comma(previous: str, current: str) -> bool:
        if not previous or not current:
            return False
        if previous.endswith((",", "{", "[", ":")):
            return False
        if current[0] in "}]":
            return False
        return bool(re.match(r'(?:"|[{\[]|-?\d|true\b|false\b|null\b)', current))

    @staticmethod
    def _append_json_comma(line: str) -> str:
        stripped = line.rstrip()
        return stripped + "," + line[len(stripped):]

    @staticmethod
    def _repair_common_json_trailing_commas(candidate: str) -> str:
        """Remove trailing commas before closing braces/brackets."""
        return re.sub(r",(\s*[}\]])", r"\1", candidate)

    def _build_openai_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required to use the OpenAI-compatible LLM gateway.") from exc

        return OpenAI(
            api_key=self.api_key or load_llm_api_key(),
            base_url=self.base_url or load_llm_base_url(),
            http_client=httpx.Client(
                trust_env=self.trust_env,
                timeout=180.0,
                follow_redirects=True,
            ),
        )

    def _build_anthropic_client(self) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("anthropic package is required for deprecated Anthropic fallback mode.") from exc

        return Anthropic(
            api_key=self.api_key or load_llm_api_key(),
            base_url=self.base_url or load_llm_base_url(),
            http_client=httpx.Client(
                trust_env=self.trust_env,
                timeout=180.0,
                follow_redirects=True,
            ),
        )

    def _should_use_legacy_anthropic(self) -> bool:
        if isinstance(self.base_url, str) and self.base_url.strip():
            lowered = self.base_url.lower()
            return "anthropic" in lowered or "claudecode" in lowered
        return should_use_legacy_anthropic_config()

    @staticmethod
    def _build_openai_messages(system: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = [{"role": "system", "content": system}]
        for message in messages:
            normalized.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": OpenAICompatibleClientAdapter._normalize_message_content(message.get("content", "")),
                }
            )
        return normalized

    @staticmethod
    def _normalize_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            pieces: List[str] = []
            for item in content:
                if isinstance(item, str):
                    pieces.append(item)
                    continue
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
            return "\n".join(piece for piece in pieces if piece)
        return str(content)

    @staticmethod
    def _coerce_text_parts(content: Any) -> List[str]:
        if isinstance(content, str):
            return [content]
        if not isinstance(content, list):
            return [str(content)] if content is not None else []

        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            text = getattr(item, "text", None)
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"text", "output_text"} and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                    continue
                if item_type == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
                    continue
            if isinstance(text, str):
                parts.append(text)
        return parts

    @staticmethod
    def _supports_chat_completions(client: Any) -> bool:
        chat = getattr(client, "chat", None)
        completions = getattr(chat, "completions", None)
        return callable(getattr(completions, "create", None))

    @staticmethod
    def _supports_responses_api(client: Any) -> bool:
        responses = getattr(client, "responses", None)
        return callable(getattr(responses, "create", None))

    @staticmethod
    def _supports_anthropic_messages(client: Any) -> bool:
        messages = getattr(client, "messages", None)
        return callable(getattr(messages, "create", None))
