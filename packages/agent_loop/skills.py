"""List skill summaries and read one skill as plain text on demand."""

from __future__ import annotations

import re
from pathlib import Path


DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"
MAX_SKILL_BYTES = 64 * 1024
MAX_FRONTMATTER_BYTES = 4 * 1024
_SKILL_ID = re.compile(r"[a-z][a-z0-9_]*\Z")

READ_SKILL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_skill",
        "description": "Read one ocean skill's guidance as plain text. Use the skill index to choose an ID.",
        "parameters": {
            "type": "object",
            "properties": {"skill_id": {"type": "string", "description": "Exact skill ID from the index."}},
            "required": ["skill_id"],
            "additionalProperties": False,
        },
    },
}


def _skill_file(skill_id: str, root: Path | str | None) -> Path:
    if not isinstance(skill_id, str) or not _SKILL_ID.fullmatch(skill_id):
        raise ValueError("Invalid skill ID")
    base = Path(root or DEFAULT_SKILLS_ROOT).resolve()
    path = (base / skill_id / "SKILL.md").resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"Unknown or unsafe skill ID: {skill_id}")
    if path.stat().st_size > MAX_SKILL_BYTES:
        raise ValueError(f"Skill is too large: {skill_id}")
    return path


def _metadata(path: Path) -> dict[str, str]:
    with path.open("rb") as source:
        front = source.read(MAX_FRONTMATTER_BYTES + 1)
    prefix_length = 6 if front.startswith(b"\xef\xbb\xbf---") else 3
    closing = re.search(rb"\r?\n---\r?\n", front[prefix_length:])
    end = prefix_length + closing.end() if closing else -1
    if end < 0 or end > MAX_FRONTMATTER_BYTES:
        raise ValueError(f"Skill frontmatter is missing or too long: {path.parent.name}")
    lines = front[:end].decode("utf-8-sig").splitlines()
    if not lines or lines[0] != "---" or "---" not in lines[1:]:
        raise ValueError(f"Missing skill frontmatter: {path.parent.name}")
    fields = dict(
        line.split(":", 1)
        for line in lines[1:lines.index("---", 1)]
        if ":" in line and not line.startswith(" ")
    )
    skill_id = fields.get("skill_id", "").strip()
    description = fields.get("description", "").strip()
    if skill_id != path.parent.name or not description:
        raise ValueError(f"Invalid skill summary: {path.parent.name}")
    return {"skill_id": skill_id, "description": description[:240]}


def list_skills(root: Path | str | None = None) -> list[dict[str, str]]:
    """Return the complete, lightweight index without loading skill bodies."""
    base = Path(root or DEFAULT_SKILLS_ROOT).resolve()
    if not base.is_dir():
        raise ValueError(f"Skills directory does not exist: {base}")
    summaries = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and _SKILL_ID.fullmatch(child.name):
            try:
                summaries.append(_metadata(_skill_file(child.name, base)))
            except (OSError, ValueError):
                # Skills are optional guidance; one unreadable entry must not
                # prevent the model from answering or requesting user input.
                continue
    return summaries


def read_skill(skill_id: str, root: Path | str | None = None) -> str:
    """Return the selected Markdown; never compile or execute its examples."""
    path = _skill_file(skill_id, root)
    with path.open("rb") as source:
        content = source.read(MAX_SKILL_BYTES + 1)
    if len(content) > MAX_SKILL_BYTES:
        raise ValueError(f"Skill is too large: {skill_id}")
    return content.decode("utf-8")
