from __future__ import annotations

from apps.api.main import _merge_memory_resolved_entities, _merge_named_region_resolved_entities
from packages.conversation_memory import (
    MemoryAgent,
    list_memory_entries,
    parse_named_region_definition,
    remember_memory_entry,
)


def test_parse_chinese_named_region_definition_in_memory_agent() -> None:
    parsed = parse_named_region_definition("把 112E-116E, 20N-23N 叫做珠江口羽流区。")

    assert parsed == {
        "name": "珠江口羽流区",
        "lon_range": [112.0, 116.0],
        "lat_range": [20.0, 23.0],
    }


def test_memory_agent_records_named_region_without_command_preprocessor() -> None:
    conversation_state: dict = {}
    operations = MemoryAgent().record(
        conversation_state=conversation_state,
        planner=object(),
        turn_packet={
            "turn_id": 1,
            "user_query": "把 112E-116E, 20N-23N 叫做珠江口羽流区。",
            "effective_query": "把 112E-116E, 20N-23N 叫做珠江口羽流区。",
            "routing_mode": "general_answer",
            "status": "completed",
        },
    )

    entries = list_memory_entries(conversation_state)

    assert operations[0]["kind"] == "region"
    assert entries[0]["match_key"] == "region:珠江口羽流区"
    assert entries[0]["data"]["lon_range"] == [112.0, 116.0]
    assert entries[0]["data"]["lat_range"] == [20.0, 23.0]


def test_memory_agent_recall_returns_resolved_region_entities(monkeypatch) -> None:
    conversation_state: dict = {}
    remember_memory_entry(
        conversation_state,
        scope="conversation",
        kind="region",
        match_key="region:珠江口羽流区",
        text="User defined named region '珠江口羽流区' as lon_range=[112, 116] and lat_range=[20, 23].",
        data={
            "name": "珠江口羽流区",
            "aliases": ["珠江口羽流区"],
            "lon_range": [112, 116],
            "lat_range": [20, 23],
        },
        confidence=1.0,
        salience=0.95,
    )

    def fake_select_memory_with_llm(*, planner, query, entries, max_entries_per_role):
        selected_id = entries[0]["id"]
        return {
            "router": [{"id": selected_id, "reason": "resolves named region"}],
            "planner": [{"id": selected_id, "reason": "provides spatial bounds"}],
            "synthesizer": [],
            "general_answer": [],
        }

    monkeypatch.setattr("packages.conversation_memory._select_memory_with_llm", fake_select_memory_with_llm)

    planner_packet = MemoryAgent().recall(
        conversation_state=conversation_state,
        query="用珠江口羽流区分析 chlorophyll bloom。",
        planner=object(),
        role="planner",
    )

    assert planner_packet["resolved_entities"]["region_source"] == "conversation_memory"
    assert planner_packet["resolved_entities"]["lon_range"] == [112.0, 116.0]
    assert planner_packet["resolved_entities"]["lat_range"] == [20.0, 23.0]
    assert planner_packet["resolved_entities"]["region"]["name"] == "珠江口羽流区"


def test_memory_resolved_entity_merge_does_not_override_explicit_bounds() -> None:
    merged = _merge_memory_resolved_entities(
        {"lon_range": [1, 2], "lat_range": [3, 4]},
        {
            "lon_range": [112, 116],
            "lat_range": [20, 23],
            "region": {"name": "珠江口羽流区"},
            "named_regions": ["珠江口羽流区"],
        },
    )

    assert merged["lon_range"] == [1, 2]
    assert merged["lat_range"] == [3, 4]
    assert merged["named_regions"] == ["珠江口羽流区"]
    assert "region" not in merged


def test_query_named_region_is_injected_before_harness_planning() -> None:
    class FakePlanner:
        @staticmethod
        def resolve_named_region_entities(query: str) -> dict:
            assert "south china sea" in query
            return {
                "region_name": "south china sea",
                "region_source": "router_named_region",
                "named_regions": ["south china sea"],
                "lon_range": [105.0, 122.0],
                "lat_range": [5.0, 23.0],
            }

    merged = _merge_named_region_resolved_entities(
        {},
        "show me the sst mean from 2014 to 2022 south china sea",
        FakePlanner(),
    )

    assert merged["lon_range"] == [105.0, 122.0]
    assert merged["lat_range"] == [5.0, 23.0]
    assert merged["named_regions"] == ["south china sea"]


def test_query_named_region_does_not_override_explicit_frontend_bounds() -> None:
    class FakePlanner:
        @staticmethod
        def resolve_named_region_entities(query: str) -> dict:
            return {
                "named_regions": ["south china sea"],
                "lon_range": [105.0, 122.0],
                "lat_range": [5.0, 23.0],
            }

    merged = _merge_named_region_resolved_entities(
        {"lon_range": [110.0, 111.0], "lat_range": [20.0, 21.0]},
        "show me the sst mean from 2014 to 2022 south china sea",
        FakePlanner(),
    )

    assert merged["lon_range"] == [110.0, 111.0]
    assert merged["lat_range"] == [20.0, 21.0]
    assert merged["named_regions"] == ["south china sea"]
