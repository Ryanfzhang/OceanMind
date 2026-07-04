"""
Skill loader utilities.

These helpers load raw `SKILL.md` documents from the repository without
requiring a structured parser.
"""

from pathlib import Path
from typing import Dict, List, Optional


def get_skills_root(skills_root: Optional[str] = None) -> Path:
    """Return the skills root directory."""
    if skills_root is not None:
        return Path(skills_root)
    return Path(__file__).resolve().parents[2] / "skills"


def list_skill_ids(skills_root: Optional[str] = None) -> List[str]:
    """List all skill ids that contain a `SKILL.md` file."""
    root = get_skills_root(skills_root)
    if not root.exists():
        return []

    skill_ids = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "SKILL.md").exists():
            skill_ids.append(child.name)
    return skill_ids


def get_skill_path(skill_id: str, skills_root: Optional[str] = None) -> Path:
    """Return the filesystem path to a skill markdown file."""
    root = get_skills_root(skills_root)
    return root / skill_id / "SKILL.md"


def load_skill_markdown(skill_id: str, skills_root: Optional[str] = None) -> str:
    """Load a single skill markdown document."""
    path = get_skill_path(skill_id, skills_root)
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")
    return path.read_text(encoding="utf-8")


def load_all_skill_markdowns(skills_root: Optional[str] = None) -> Dict[str, str]:
    """Load all available skill markdown documents."""
    return {
        skill_id: load_skill_markdown(skill_id, skills_root)
        for skill_id in list_skill_ids(skills_root)
    }
