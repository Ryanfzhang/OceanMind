"""XML input/output contracts for LLM-assisted harness planning."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html import unescape
from typing import Any, Iterable, Mapping, Optional, Sequence
from xml.sax.saxutils import escape


class LLMContractError(ValueError):
    """Raised when an LLM response does not satisfy the XML/JSON contract."""


@dataclass(frozen=True)
class XMLIOContract:
    task: str
    input_payload: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    rules: Sequence[str] = field(default_factory=tuple)


def render_xml_io_contract(contract: XMLIOContract) -> str:
    """Render a prompt section with explicit XML input and output contracts."""

    input_json = json.dumps(contract.input_payload, ensure_ascii=False, indent=2, sort_keys=True)
    output_json = json.dumps(contract.output_schema, ensure_ascii=False, indent=2, sort_keys=True)
    rules = list(contract.rules) or [
        "Read only the JSON inside <input>.",
        "Return only one JSON object inside <output>.",
        "Do not put explanations, markdown, or code fences outside <output>.",
        "Use null for unknown optional values and do not invent unavailable data.",
    ]
    rule_lines = "\n".join(f"  <rule>{escape(rule)}</rule>" for rule in rules)
    return (
        "<planner_io_contract>\n"
        f"  <task>{escape(contract.task)}</task>\n"
        "  <input format=\"json\">\n"
        f"{_indent_xml_text(escape(input_json), spaces=4)}\n"
        "  </input>\n"
        "  <output format=\"json\">\n"
        f"{_indent_xml_text(escape(output_json), spaces=4)}\n"
        "  </output>\n"
        "  <rules>\n"
        f"{rule_lines}\n"
        "  </rules>\n"
        "</planner_io_contract>"
    )


def parse_json_from_xml_output(
    text: str,
    *,
    required_keys: Optional[Iterable[str]] = None,
    expect_object: bool = True,
) -> Any:
    """Extract and repair JSON from the last ``<output>...</output>`` block."""

    output_body = extract_output_body(text)
    candidate = extract_json_candidate(output_body)
    parsed = loads_json_with_repairs(candidate)
    if expect_object and not isinstance(parsed, dict):
        raise LLMContractError("LLM <output> JSON must be an object.")
    if required_keys and isinstance(parsed, dict):
        missing = [key for key in required_keys if key not in parsed]
        if missing:
            raise LLMContractError(f"LLM <output> JSON is missing required keys: {', '.join(missing)}")
    return parsed


def extract_output_body(text: str) -> str:
    matches = re.findall(r"<output(?:\s+[^>]*)?>(.*?)</output>", text, flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        raise LLMContractError("LLM response must include an <output>...</output> block.")
    return unescape(matches[-1]).strip()


def extract_json_candidate(text: str) -> str:
    cleaned = _strip_code_fence(text.strip())
    if cleaned.startswith(("{", "[")):
        return cleaned

    starts = [index for index, char in enumerate(cleaned) if char in "{["]
    for start in starts:
        end = _find_matching_json_end(cleaned, start)
        if end is not None:
            return cleaned[start : end + 1]
    raise LLMContractError("LLM <output> block does not contain a JSON object or array.")


def loads_json_with_repairs(candidate: str) -> Any:
    cleaned = _strip_code_fence(candidate.strip())
    attempts = [cleaned]
    repaired = repair_json_candidate(cleaned)
    if repaired != cleaned:
        attempts.append(repaired)

    last_error: Optional[json.JSONDecodeError] = None
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is None:
        raise LLMContractError("LLM <output> JSON could not be parsed.")
    raise LLMContractError(
        "LLM <output> JSON is invalid: "
        f"{last_error.msg} at line {last_error.lineno}, column {last_error.colno}."
    ) from last_error


def repair_json_candidate(candidate: str) -> str:
    repaired = _strip_json_comments(candidate)
    repaired = _repair_jsonish_literals(repaired)
    repaired = _repair_missing_commas(repaired)
    repaired = _repair_trailing_commas(repaired)
    return repaired


def _indent_xml_text(text: str, *, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n\s*```$", "", cleaned)
    return cleaned.strip()


def _find_matching_json_end(text: str, start: int) -> Optional[int]:
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    stack = [closer]
    in_string = False
    escaped = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return index
        elif char == closer and not stack:
            return index
    return None


def _strip_json_comments(candidate: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(candidate):
        char = candidate[index]
        next_char = candidate[index + 1] if index + 1 < len(candidate) else ""
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and next_char == "/":
            index += 2
            while index < len(candidate) and candidate[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(candidate) and not (candidate[index] == "*" and candidate[index + 1] == "/"):
                index += 1
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _repair_jsonish_literals(candidate: str) -> str:
    repaired = re.sub(r"\bTrue\b", "true", candidate)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    return repaired


def _repair_missing_commas(candidate: str) -> str:
    lines = candidate.splitlines()
    if len(lines) <= 1:
        return candidate
    repaired = list(lines)
    previous_index: Optional[int] = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if previous_index is not None and _needs_json_comma(repaired[previous_index].strip(), stripped):
            repaired[previous_index] = _append_json_comma(repaired[previous_index])
        previous_index = index
    joined = "\n".join(repaired)
    return re.sub(r'([}\]"0-9]|true|false|null)(\s+)(?=(["{\[]))', r"\1, ", joined)


def _needs_json_comma(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if previous.endswith((",", "{", "[", ":")):
        return False
    if current[0] in "}]":
        return False
    return bool(re.match(r'(?:"|[{\[]|-?\d|true\b|false\b|null\b|True\b|False\b|None\b)', current))


def _append_json_comma(line: str) -> str:
    stripped = line.rstrip()
    return stripped + "," + line[len(stripped):]


def _repair_trailing_commas(candidate: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", candidate)
