from __future__ import annotations

from packages import conversation_memory as memory


def test_memory_index_is_scope_aware_and_compact() -> None:
    state: dict = {}
    entry = memory.remember_memory_entry(
        state,
        scope="conversation",
        kind="region",
        match_key="region:pearl_river_plume",
        text="Pearl River plume box for chlorophyll bloom analysis with a longer description than the index needs.",
        data={
            "lon_range": [112, 116],
            "lat_range": [20, 23],
            "variables": ["chlorophyll"],
            "polygon_points": [[112, 20], [116, 20], [116, 23], [112, 23]],
        },
        salience=0.9,
    )

    assert entry is not None
    assert entry["id"] == "conversation:mem_0001"
    assert entry["scope"] == "conversation"

    index = memory.build_memory_index(memory.list_memory_entries(state))

    assert index[0]["id"] == "conversation:mem_0001"
    assert index[0]["scope"] == "conversation"
    assert index[0]["kind"] == "region"
    assert index[0]["variables"] == ["chlorophyll"]
    assert index[0]["region"]["lon_range"] == [112, 116]
    assert "data" not in index[0]


def test_retrieve_uses_index_candidates_and_hydrates_selected_entries(monkeypatch) -> None:
    state: dict = {}
    memory.remember_memory_entry(
        state,
        kind="region",
        match_key="region:pearl_river_plume",
        text="Pearl River plume named region for chlorophyll bloom queries.",
        data={"lon_range": [112, 116], "lat_range": [20, 23], "variables": ["chlorophyll"]},
        salience=0.9,
    )
    memory.remember_memory_entry(
        state,
        kind="artifact",
        match_key="artifact:old_profile",
        text="Old temperature profile near Luzon Strait.",
        data={"result_id": "profile_1", "variables": ["temp"]},
        salience=0.4,
    )

    seen: dict = {}

    def fake_select_memory_with_llm(*, planner, query, entries, max_entries_per_role):
        seen["entries"] = entries
        assert entries
        assert all("index_text" in entry for entry in entries)
        assert all("data" not in entry for entry in entries)
        selected_id = entries[0]["id"]
        return {
            "router": [{"id": selected_id, "reason": "resolves the named region"}],
            "planner": [{"id": selected_id, "reason": "provides analysis bounds"}],
            "synthesizer": [],
            "general_answer": [],
        }

    monkeypatch.setattr(memory, "_select_memory_with_llm", fake_select_memory_with_llm)

    packets = memory.retrieve_memory_packets(
        conversation_state=state,
        query="Use the Pearl River plume region for chlorophyll",
        planner=object(),
    )

    assert seen["entries"][0]["id"] == "conversation:mem_0001"
    router_entry = packets["router"]["entries"][0]
    assert router_entry["data"]["lon_range"] == [112, 116]
    assert router_entry["data"]["lat_range"] == [20, 23]


def test_project_memory_can_be_recalled_without_user_scope(monkeypatch) -> None:
    conversation_state: dict = {}
    project_state: dict = {}
    memory.remember_memory_entry(
        project_state,
        scope="project",
        kind="workflow_pattern",
        match_key="workflow:bloom_frequency",
        text="For bloom frequency requests, run detection first and then compute_event_frequency_map.",
        data={"tags": ["bloom", "frequency"], "variables": ["chlorophyll"]},
        salience=0.85,
    )

    def fake_select_memory_with_llm(*, planner, query, entries, max_entries_per_role):
        project_ids = [entry["id"] for entry in entries if entry.get("scope") == "project"]
        assert project_ids == ["project:mem_0001"]
        return {
            "router": [],
            "planner": [{"id": project_ids[0], "reason": "project workflow convention"}],
            "synthesizer": [],
            "general_answer": [],
        }

    monkeypatch.setattr(memory, "_select_memory_with_llm", fake_select_memory_with_llm)

    packets = memory.retrieve_memory_packets(
        conversation_state=conversation_state,
        project_memory_state=project_state,
        query="Make a bloom frequency map",
        planner=object(),
    )

    planner_entry = packets["planner"]["entries"][0]
    assert planner_entry["scope"] == "project"
    assert planner_entry["kind"] == "workflow_pattern"
    assert "user" not in {entry["scope"] for entry in packets["planner"]["entries"]}
