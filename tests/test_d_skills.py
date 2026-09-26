"""Skill discovery stays lightweight; reading stays scoped to one file."""

import pytest

from packages.agent_loop.skills import (
    DEFAULT_SKILLS_ROOT,
    MAX_SKILL_BYTES,
    READ_SKILL_SCHEMA,
    list_skills,
    read_skill,
)


def test_complete_index_only_contains_summaries():
    index = list_skills()
    assert len(index) == 63
    assert len({entry["skill_id"] for entry in index}) == len(index)
    assert all(set(entry) == {"skill_id", "description"} for entry in index)
    assert all("## Workflow" not in str(entry) for entry in index)
    assert READ_SKILL_SCHEMA["function"]["name"] == "read_skill"


def test_read_one_skill_preserves_plain_markdown():
    body = read_skill("ocean_event_statistics")
    assert body.startswith("---\nskill_id: ocean_event_statistics")
    assert "## Workflow" in body
    assert "compute_event_statistics(" in body


def test_windows_crlf_skill_frontmatter_is_indexed(tmp_path):
    original_index = list_skills()
    for item in original_index:
        skill_id = item["skill_id"]
        folder = tmp_path / skill_id
        folder.mkdir()
        body = (DEFAULT_SKILLS_ROOT / skill_id / "SKILL.md").read_bytes()
        (folder / "SKILL.md").write_bytes(body.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert list_skills(tmp_path) == original_index
    assert "include_climatology" in read_skill("ocean_transport_analysis", tmp_path)


def test_invalid_optional_skill_does_not_block_index(tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "SKILL.md").write_text("No frontmatter")
    valid = tmp_path / "valid"
    valid.mkdir()
    (valid / "SKILL.md").write_text(
        "---\nskill_id: valid\ndescription: Useful guidance.\n---\nBody\n"
    )
    assert list_skills(tmp_path) == [
        {"skill_id": "valid", "description": "Useful guidance."}
    ]


def test_index_does_not_decode_skill_body(tmp_path):
    folder = tmp_path / "sample"
    folder.mkdir()
    (folder / "SKILL.md").write_bytes(
        b"---\nskill_id: sample\ndescription: Example\n---\n\xff"
    )
    assert list_skills(tmp_path) == [{"skill_id": "sample", "description": "Example"}]
    with pytest.raises(UnicodeDecodeError):
        read_skill("sample", tmp_path)


@pytest.mark.parametrize("skill_id", ["../other", "ocean_eddy_detection/SKILL.md", ".", "", "Ocean_Eddy", "does_not_exist"])
def test_invalid_or_unknown_id_rejected(skill_id):
    with pytest.raises(ValueError):
        read_skill(skill_id)


def test_symlink_escape_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("---\nskill_id: leak\ndescription: leak\n---\nsecret")
    root = tmp_path / "skills"
    root.mkdir()
    (root / "leak").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        read_skill("leak", root)


def test_oversized_skill_rejected(tmp_path):
    root = tmp_path / "skills"
    folder = root / "big"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_bytes(b"x" * (MAX_SKILL_BYTES + 1))
    with pytest.raises(ValueError, match="too large"):
        read_skill("big", root)
