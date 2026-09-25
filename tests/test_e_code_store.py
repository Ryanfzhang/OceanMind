import hashlib
from pathlib import Path

import pytest

from packages.analysis_runtime.code_store import CodeStore, MAX_CODE_BYTES


def test_code_versions_reopen_exactly_and_show_parent_diff(tmp_path):
    store = CodeStore(tmp_path, "run_abc")
    first_source = "# 温度\nvalue = 1\n"
    first = store.write_analysis(first_source)
    second_source = "# 温度\nvalue = 2\n"
    second = store.write_analysis(second_source, previous_version=first["code_id"])

    assert first["previous_version"] is None
    assert second["previous_version"] == first["code_id"]
    assert second["code_id"] != first["code_id"]
    assert Path(second["path"]).is_relative_to(tmp_path)
    assert Path(second["path"]).suffix == ".py"
    assert second["sha256"] == hashlib.sha256(second_source.encode()).hexdigest()
    reopened = CodeStore(tmp_path, "run_abc")
    assert reopened.get_path(first["code_id"]).read_text() == first_source
    assert reopened.read_artifact(second["code_id"])["content"] == second_source
    assert reopened.read_artifact(first["code_id"])["content"] == first_source
    diff = reopened.read_artifact(second["code_id"], view="diff")["content"]
    assert f"--- {first['code_id']}" in diff
    assert f"+++ {second['code_id']}" in diff
    assert "-value = 1" in diff and "+value = 2" in diff
    assert second["diff"] == diff


def test_bounded_code_and_diff_reads_use_character_offsets(tmp_path):
    store = CodeStore(tmp_path, "run_abc")
    first = store.write_analysis("a = '海洋'\n" + "x = 0\n" * 1000)
    second = store.write_analysis("a = '海洋'\n" + "x = 1\n" * 1000,
                                  previous_version=first["code_id"])
    assert second["diff_truncated"]

    chunks = []
    offset = 0
    while True:
        part = store.read_artifact(first["code_id"], offset=offset, max_chars=997)
        chunks.append(part["content"])
        offset = part["next_offset"]
        if not part["truncated"]:
            break
    assert "".join(chunks) == store.get_path(first["code_id"]).read_text()

    diff = store.read_artifact(second["code_id"], view="diff", max_chars=100)
    assert len(diff["content"]) == 100 and diff["truncated"]
    assert store.read_artifact(second["code_id"], view="diff",
                               offset=diff["next_offset"], max_chars=100)["content"]


def test_rejects_traversal_cross_run_mutation_and_oversized_code(tmp_path):
    store = CodeStore(tmp_path, "run_a")
    first = store.write_analysis("print(1)\n")
    code_id = first["code_id"]
    for bad_id in ("../outside", f"{code_id}/../x", "code_" + "a" * 31 + "/"):
        with pytest.raises(ValueError, match="code ID"):
            store.read_artifact(bad_id)
    with pytest.raises(ValueError, match="run ID"):
        CodeStore(tmp_path, "../other")
    with pytest.raises(FileNotFoundError):
        CodeStore(tmp_path, "run_b").get_path(code_id)
    with pytest.raises(FileNotFoundError):
        CodeStore(tmp_path, "run_b").write_analysis("print(2)\n", previous_version=code_id)
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        store.write_analysis("x" * (MAX_CODE_BYTES + 1))
    with pytest.raises(ValueError, match="max_chars"):
        store.read_artifact(code_id, max_chars=10_001)

    store.get_path(code_id).write_text("print(2)\n")
    with pytest.raises(ValueError, match="changed"):
        store.get_path(code_id)


def test_run_code_directory_cannot_be_redirected_after_store_creation(tmp_path):
    store = CodeStore(tmp_path, "run_a")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    store.directory.rmdir()
    store.directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        store.write_analysis("print(1)\n")
    assert list(outside.iterdir()) == []
