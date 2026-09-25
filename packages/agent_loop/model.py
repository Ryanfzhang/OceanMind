"""One OpenAI-compatible chat completion, without agent routing."""

from __future__ import annotations

from typing import Any


def _get(value: Any, key: str) -> Any:
    """Accept SDK response objects and simple dictionaries used by test clients."""
    return value[key] if isinstance(value, dict) else getattr(value, key)


class OpenAIChatModel:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        client: Any = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Return an assistant message ready to append to graph state."""
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if timeout is not None:
            kwargs["timeout"] = timeout
        response = self.client.chat.completions.create(**kwargs)
        message = _get(_get(response, "choices")[0], "message")
        result: dict[str, Any] = {
            "role": "assistant",
            "content": _get(message, "content"),
        }
        reasoning = (
            message.get("reasoning_content") if isinstance(message, dict)
            else getattr(message, "reasoning_content", None)
        )
        if reasoning is not None:
            # DeepSeek thinking mode requires this field on subsequent tool requests.
            result["reasoning_content"] = reasoning
        calls = (
            message.get("tool_calls")
            if isinstance(message, dict)
            else getattr(message, "tool_calls", None)
        )
        if calls:
            result["tool_calls"] = [
                {
                    "id": _get(call, "id"),
                    "type": _get(call, "type"),
                    "function": {
                        "name": _get(_get(call, "function"), "name"),
                        "arguments": _get(_get(call, "function"), "arguments"),
                    },
                }
                for call in calls
            ]
        return result
