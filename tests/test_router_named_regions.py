from packages.harness.planner import OceanHarnessPlanner
from packages.llm_gateway.unified_processor import UnifiedQueryProcessor


def test_unified_router_strips_parameter_entities():
    processor = UnifiedQueryProcessor(client=object())
    decision = processor._normalize_decision(
        {
            "routing_mode": "dataset_analysis",
            "needs_dataset": True,
            "confidence": 0.8,
            "reason": "dataset analysis",
            "extracted_entities": {
                "region_name": "south china sea",
                "lon_range": [105.0, 122.0],
                "lat_range": [5.0, 23.0],
                "time_range": ["2011-01-01", "2014-12-31"],
                "analysis_hint": "hypoxia",
            },
        },
        query="能否查看2011到2014年南海的缺氧区域",
    )

    entities = decision["extracted_entities"]
    assert entities == {"analysis_hint": "hypoxia"}


def test_harness_consumes_explicit_region_for_hypoxia_manual():
    query = "能否查看2011到2014年南海的缺氧区域"
    plan = OceanHarnessPlanner().generate_plan_for_query(
        query,
        extracted_params={
            "region": {
                "lon_range": [105.0, 122.0],
                "lat_range": [5.0, 23.0],
            },
        },
        additional_context={},
    )

    assert plan["status"] == "ready"
    assert "ocean_hypoxia_detection" in plan["skills_used"]
    assert plan["data_requirements"]["lon_range"] == [105.0, 122.0]
    assert plan["data_requirements"]["lat_range"] == [5.0, 23.0]
    assert plan["data_requirements"]["time_range"] == ["2011-01-01", "2014-12-31"]
    assert [step["tool"] for step in plan["steps"]] == [
        "load_dataset",
        "detect_hypoxia",
        "compute_event_summary_map",
        "compute_event_summary_map",
    ]
