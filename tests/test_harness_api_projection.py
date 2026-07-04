from apps.api.main import (
    QueryRequest,
    _build_policy_context_packet,
    _build_result_synthesis_fallback,
    _hydrate_workspace_geometry_extracted_params,
    _public_failure_message,
    _requires_llm_result_synthesis,
    _should_suppress_default_policy_report,
    _translate_runtime_event,
)
from packages.harness.planner import OceanHarnessPlanner
from packages.llm_gateway.result_synthesizer import ResultSynthesizer


def test_workspace_polygon_context_hydrates_extracted_mask_polygon():
    polygon = [[113, 18], [114, 18], [114, 19], [113, 19]]
    extracted = _hydrate_workspace_geometry_extracted_params(
        {"selection_mode": "none"},
        {
            "workspace_context": {
                "workspace_selection": {
                    "selected_region": {
                        "type": "polygon",
                        "points": polygon,
                    }
                }
            }
        },
    )

    assert extracted["mask_polygon"] == [[113.0, 18.0], [114.0, 18.0], [114.0, 19.0], [113.0, 19.0]]
    assert extracted["drawn_polygon_points"] == extracted["mask_polygon"]
    assert extracted["region_selection_type"] == "polygon"
    assert extracted["region"] == {"lon_range": [113.0, 114.0], "lat_range": [18.0, 19.0]}


def test_workspace_transect_context_hydrates_extracted_transect_points():
    transect = [[113, 18], [114, 19]]
    extracted = _hydrate_workspace_geometry_extracted_params(
        {},
        {
            "workspace_context": {
                "workspace_selection": {
                    "selected_transect": {
                        "type": "transect",
                        "points": transect,
                    }
                }
            }
        },
    )

    assert extracted["transect_points"] == [[113.0, 18.0], [114.0, 19.0]]
    assert extracted["drawn_transect_points"] == extracted["transect_points"]
    assert extracted["transect_selection_type"] == "transect"


def test_harness_events_project_to_existing_frontend_protocol():
    plan = OceanHarnessPlanner().generate_plan_for_query(
        "Analyze temperature trend in this region from 2010 to 2012",
        extracted_params={"lon_range": [110, 120], "lat_range": [18, 23]},
        additional_context={},
    )
    step_cards = {}
    step_order = []
    result_cards = []
    request = QueryRequest(query="Analyze temperature trend in this region from 2010 to 2012")

    plan_events = _translate_runtime_event(
        raw_event={
            "type": "plan_generated",
            "skill_id": plan["skill_id"],
            "skills_used": plan["skills_used"],
            "plan": plan,
        },
        generated_plan=None,
        step_cards_by_id=step_cards,
        step_order=step_order,
        executor=None,
        visible_result_cards=result_cards,
        request=request,
    )
    assert plan_events[0]["type"] == "plan_ready"
    assert plan_events[0]["plan_steps"]

    first_step = plan["steps"][0]
    start_events = _translate_runtime_event(
        raw_event={
            "type": "step_start",
            "step_id": first_step["step_id"],
            "tool": first_step["tool"],
        },
        generated_plan=plan,
        step_cards_by_id=step_cards,
        step_order=step_order,
        executor=None,
        visible_result_cards=result_cards,
        request=request,
    )
    assert start_events[0]["type"] == "step_started"

    complete_events = _translate_runtime_event(
        raw_event={
            "type": "step_complete",
            "step_id": first_step["step_id"],
            "tool": first_step["tool"],
            "result_id": first_step["save_as"],
            "output_type": "data_container_result",
            "result_summary": {
                "type": "data_container_result",
                "dims": ["time", "depth", "lat", "lon"],
                "shape": [1, 1, 2, 2],
                "variable": "temp",
            },
        },
        generated_plan=plan,
        step_cards_by_id=step_cards,
        step_order=step_order,
        executor=None,
        visible_result_cards=result_cards,
        request=request,
    )
    assert complete_events[0]["type"] == "step_completed"
    assert result_cards
    assert result_cards[0]["id"] == first_step["save_as"]


def test_manual_heatwave_plan_projects_to_frontend_steps():
    plan = OceanHarnessPlanner().generate_plan_for_query(
        "Detect marine heatwave events in the region (113 ° E-117 ° E, 19 °N-22 °N) from January to March 2015",
        extracted_params={},
        additional_context={},
    )
    request = QueryRequest(
        query="Detect marine heatwave events in the region (113 ° E-117 ° E, 19 °N-22 °N) from January to March 2015"
    )

    plan_events = _translate_runtime_event(
        raw_event={
            "type": "plan_generated",
            "skill_id": plan["skill_id"],
            "skills_used": plan["skills_used"],
            "plan": plan,
        },
        generated_plan=None,
        step_cards_by_id={},
        step_order=[],
        executor=None,
        visible_result_cards=[],
        request=request,
    )

    frontend_steps = plan_events[0]["plan_steps"]
    assert [step["tool"] for step in frontend_steps] == [
        "load_dataset",
        "detect_heatwaves",
        "compute_event_summary_map",
        "compute_event_summary_map",
    ]
    assert frontend_steps[1]["save_as"] == "heatwave_detection"


def test_replanned_harness_plan_projects_to_frontend_protocol():
    plan = OceanHarnessPlanner().generate_plan_for_query(
        "Plot oxygen in this region from 2010 to 2012",
        extracted_params={"lon_range": [110, 120], "lat_range": [18, 23]},
        additional_context={},
    )
    request = QueryRequest(query="Plot oxygen in this region from 2010 to 2012")

    replanned_events = _translate_runtime_event(
        raw_event={
            "type": "plan_replanned",
            "skill_id": plan["skill_id"],
            "skills_used": plan["skills_used"],
            "plan": plan,
            "reason": "generated_code_error",
        },
        generated_plan=None,
        step_cards_by_id={},
        step_order=[],
        executor=None,
        visible_result_cards=[],
        request=request,
    )

    assert replanned_events[0]["type"] == "plan_replanned"
    assert replanned_events[0]["plan_steps"]
    assert replanned_events[0]["reason"] == "generated_code_error"


def test_step_reflection_projects_as_running_card_without_error():
    plan = OceanHarnessPlanner().generate_plan_for_query(
        "Plot oxygen in this region from 2010 to 2012",
        extracted_params={"lon_range": [110, 120], "lat_range": [18, 23]},
        additional_context={},
    )
    request = QueryRequest(query="Plot oxygen in this region from 2010 to 2012")
    step = plan["steps"][-1]

    events = _translate_runtime_event(
        raw_event={
            "type": "step_reflection_started",
            "step_id": step["step_id"],
            "tool": step["tool"],
            "reason": "generated_code_error",
            "replans_used": 0,
            "max_replans": 1,
        },
        generated_plan=plan,
        step_cards_by_id={},
        step_order=[],
        executor=None,
        visible_result_cards=[],
        request=request,
    )

    assert events[0]["type"] == "step_reflection_started"
    assert events[0]["reason"] == "generated_code_error"
    assert events[0]["step_card"]["status"] == "running"
    assert "error" not in events[0]["step_card"]
    assert events[0]["step_card"]["progress"]["phase"] == "reflection"


def test_code_agent_format_error_message_does_not_blame_planner():
    message = _public_failure_message(
        "llm_format",
        "CodeAgent design output was not a valid JSON analysis design: Expecting value",
        query="Compute a custom oxygen variability index",
    )

    assert "CodeAgent" in message
    assert "planner returned" not in message.lower()


def test_code_agent_error_message_does_not_parse_negated_missing_variable():
    message = _public_failure_message(
        "llm_format",
        "CodeAgent could not produce a valid analysis design or Python entrypoint for the generated-code step. "
        "This is an internal code-generation format issue, not a missing variable, time range, or region in your request.",
        query="Show regions where oxygen < 60 and temp > 28",
    )

    assert "CodeAgent" in message
    assert "variable or variables" not in message
    assert "missing fields" not in message.lower()


def test_code_agent_reasoning_only_error_message_does_not_blame_planner():
    message = _public_failure_message(
        "llm_format",
        "CodeAgent design output was not a valid JSON analysis design after retry: "
        "LLM response does not contain final text content; it contained reasoning_content only; "
        "finish_reason=length",
        query="Compute a custom oxygen variability index",
    )

    assert "CodeAgent" in message
    assert "Planner" not in message


def test_reasoning_only_selector_failure_message_is_specific():
    message = _public_failure_message(
        "llm_format",
        "Planner skill selector failed to return a final routing contract: "
        "(LLM response does not contain final text content; it contained reasoning_content only; "
        "finish_reason=length; model=deepseek-v4-flash; reasoning_tokens=2400; completion_tokens=2400).",
        query="plot the winter mean relative vorticity",
    )

    assert "Planner skill selector" in message
    assert "reasoning" in message
    assert "No variable" in message
    assert "incomplete structured workflow contract" not in message


def test_generic_planning_failure_message_does_not_infer_missing_variable():
    message = _public_failure_message(
        "planning",
        "The planner could not generate an executable OceanMind workflow, and it did not report any specific "
        "missing fields. Please try again; if it repeats, include the exact variable, time range, region, and output type.",
        query="Plot January 2018 bottom salinity over the South China Sea",
    )

    assert "missing the fields" not in message
    assert "variable or variables" not in message
    assert "could not generate an executable" in message


def test_invalid_workflow_syntax_message_does_not_infer_missing_variable():
    message = _public_failure_message(
        "planning",
        "Invalid workflow Python block in ocean_spatial_field_analysis: invalid syntax (<unknown>, line 24)",
        query="Plot January 2018 bottom salinity over the South China Sea",
    )

    assert "missing the fields" not in message
    assert "variable or variables" not in message


def test_selector_policy_flag_blocks_policy_cards_for_scientific_control_query():
    query = (
        "For the Yellow Sea from 2018 to 2020, first diagnose whether stratification strengthened "
        "and whether bottom oxygen was consistent with stratification control, then test whether the "
        "stratification stability time series leads or lags the oxygen time series, and explain what "
        "in the available variables supports this interpretation."
    )
    active_plan = {
        "skill_id": "ocean_stratification_diagnostics",
        "skills_used": ["ocean_stratification_diagnostics", "ocean_lag_correlation"],
        "planner_llm_decision": {
            "policy_making_intent": False,
            "policy_making_reason": "Scientific mechanism/control diagnosis, not policymaking.",
        },
    }

    assert _requires_llm_result_synthesis(
        active_plan=active_plan,
        synthesis_profile_id=None,
        user_request=query,
    ) is False
    assert _build_policy_context_packet(
        user_request=query,
        evidence_packets=[
            {
                "result_id": "stability_oxygen_lag_correlation",
                "title": "Stratification stability versus bottom oxygen lag correlation",
            }
        ],
        result_summaries={},
        synthesis_profile_id=None,
        policy_making_intent=False,
    ) is None
    assert _should_suppress_default_policy_report(query, active_plan) is True
    assert ResultSynthesizer()._policy_guidance_requested(query) is False


def test_result_synthesis_fallback_preserves_completed_result_metadata():
    payload = _build_result_synthesis_fallback(
        user_request="Compare bloom hotspots with water masses",
        active_plan={"skill_id": "ocean_watermass_event_association"},
        completed_steps=[
            {
                "step_id": "association",
                "tool": "compute_watermass_event_association",
                "result_id": "watermass_association",
            }
        ],
        result_summaries={
            "watermass_association": {
                "type": "watermass_event_association_result",
                "title": "Water-mass event association",
                "top_associated_watermass_name": "Shelf Mixed Water",
                "association_score": 0.81,
                "hotspot_tile_count": 42,
                "valid_tile_count": 900,
            }
        },
        synthesis_error="LLM synthesis returned invalid JSON",
    )

    assert "workflow completed" in payload["summary"]
    assert payload["scientific_findings"]
    assert payload["scientific_findings"][0]["result_id"] == "watermass_association"
    assert any("Shelf Mixed Water" in item for item in payload["scientific_findings"][0]["evidence"])
    assert payload["synthesis_warnings"]
