"""Load SKILL.md files as optional knowledge context, not workflow authority."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


def load_skill_contexts(skills_root: str | Path = "skills") -> Dict[str, str]:
    root = Path(skills_root)
    if not root.exists():
        return {}
    contexts: Dict[str, str] = {}
    for skill_file in root.glob("*/SKILL.md"):
        try:
            contexts[skill_file.parent.name] = skill_file.read_text(encoding="utf-8")
        except OSError:
            continue
    return contexts


def retrieve_skill_context(
    query: str,
    *,
    skills_root: str | Path = "skills",
    limit: int = 4,
) -> Dict[str, str]:
    """Return a small set of relevant skill markdowns by simple token overlap."""

    contexts = load_skill_contexts(skills_root)
    if not contexts:
        return {}
    query_tokens = _tokens(query)
    scored: List[Tuple[int, str]] = []
    for skill_id, markdown in contexts.items():
        haystack = _tokens(skill_id.replace("_", " ") + " " + markdown[:4000])
        overlap = len(query_tokens & haystack)
        if overlap:
            scored.append((overlap, skill_id))
    scored.sort(reverse=True)
    return {skill_id: contexts[skill_id] for _, skill_id in scored[: max(1, limit)]}


def _tokens(text: str) -> set[str]:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return {token for token in normalized.split() if len(token) >= 2}

