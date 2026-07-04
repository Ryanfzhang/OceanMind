from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from packages.skill_system.loader import list_skill_ids, load_skill_markdown
from packages.tool_loader.registry import get_tool_output_type


TEST_QUERIES_PATH = Path(__file__).resolve().parents[2] / "tests" / "skills" / "test_queries.md"


def build_planner_hint(skill_id: str, skill_markdown: str) -> Dict[str, Any]:
    sections = _parse_sections(skill_markdown)
    positive_examples = _extract_bullet_items(sections.get("applicable scenarios", []))
    positive_examples.extend(_extract_example_queries(skill_markdown))
    negative_examples = _extract_bullet_items(sections.get("do not use when", []))
    test_query = _load_test_queries().get(skill_id)
    if test_query:
        positive_examples.append(test_query)
    if not positive_examples:
        positive_examples.append(skill_id.replace("_", " "))

    required_entities, optional_entities = _extract_input_parameters(skill_markdown)
    tool_names = _extract_tool_names(skill_markdown)
    result_types = [get_tool_output_type(tool_name) for tool_name in tool_names if get_tool_output_type(tool_name)]

    description_lines = sections.get("description", [])
    intent_summary = " ".join(line.strip() for line in description_lines[:3]).strip()
    if not intent_summary:
        intent_summary = skill_id.replace("_", " ")

    return {
        "intent_summary": intent_summary,
        "positive_query_examples": _dedupe_preserve([item for item in positive_examples if item])[:6],
        "negative_query_examples": _dedupe_preserve([item for item in negative_examples if item])[:4],
        "required_entities": required_entities,
        "inferable_entities": optional_entities,
        "result_types": _dedupe_preserve([item for item in result_types if item]),
    }


@lru_cache(maxsize=1)
def load_all_planner_hints(skills_root: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    return {
        skill_id: build_planner_hint(skill_id, load_skill_markdown(skill_id, skills_root))
        for skill_id in list_skill_ids(skills_root)
    }


def _load_test_queries() -> Dict[str, str]:
    if not TEST_QUERIES_PATH.exists():
        return {}
    text = TEST_QUERIES_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        r"##\s+\d+\.\s+([A-Za-z0-9_]+).*?\n\*\*Query:\*\*\s+\"([^\"]+)\"",
        flags=re.DOTALL,
    )
    return {skill_id: query for skill_id, query in pattern.findall(text)}


def _parse_sections(skill_markdown: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current_section: Optional[str] = None
    in_code_block = False

    for raw_line in skill_markdown.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped.startswith("## "):
            current_section = stripped[3:].strip().lower()
            sections.setdefault(current_section, [])
            continue
        if current_section is not None:
            sections.setdefault(current_section, []).append(raw_line.rstrip())
    return sections


def _extract_bullet_items(lines: List[str]) -> List[str]:
    items: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip().strip('"'))
    return items


def _extract_example_queries(skill_markdown: str) -> List[str]:
    pattern = re.compile(r"\*\*User Query:\*\*.*?```text\s*(.*?)\s*```", flags=re.DOTALL)
    return [match.strip() for match in pattern.findall(skill_markdown)]


def _extract_input_parameters(skill_markdown: str) -> tuple[List[str], List[str]]:
    required_pattern = re.compile(r"-\s+Required:\s*(.*?)(?:-\s+Optional:|##|\Z)", flags=re.DOTALL)
    optional_pattern = re.compile(r"-\s+Optional:\s*(.*?)(?:##|\Z)", flags=re.DOTALL)

    def parse_block(pattern: re.Pattern[str]) -> List[str]:
        match = pattern.search(skill_markdown)
        if not match:
            return []
        block = match.group(1)
        return [item.strip("` ").strip() for item in re.findall(r"-\s+`?([A-Za-z0-9_.]+)`?", block)]

    return parse_block(required_pattern), parse_block(optional_pattern)


def _extract_tool_names(skill_markdown: str) -> List[str]:
    explicit_pattern = re.compile(r"\*\*Tool\*\*:\s*`([^`]+)`")
    inline_pattern = re.compile(r"`([^`]+)`")
    seen: List[str] = []
    for match in explicit_pattern.findall(skill_markdown):
        if match not in seen:
            seen.append(match)
    for match in inline_pattern.findall(skill_markdown):
        candidate = match[:-2] if match.endswith("()") else match
        if candidate not in seen:
            seen.append(candidate)
    return seen


def _dedupe_preserve(values: List[Any]) -> List[Any]:
    seen: List[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.append(value)
    return seen
