from benchmarks.evaluation_metrics import (
    FROZEN_REACT_AGENT_TYPES,
    build_run_dataframe,
)
from benchmarks.gold_tool_specs import load_gold_tool_specs
from benchmarks.query_suite import load_query_suite
from benchmarks.trace import AgentTrace


def test_gold_tool_specs_cover_every_query_and_preserve_reference_chain():
    queries = load_query_suite()
    specs = load_gold_tool_specs(queries)

    assert len(specs) == len(queries)
    assert specs[1].reference_tools == ["load_dataset", "extract_regional_mean"]
    assert specs[1].setup_tools == ["load_dataset"]
    assert specs[1].required_tools == ["extract_regional_mean"]
    assert specs[1].scope_requirements["variables"] == ["temp"]

    lag_spec = specs[57]
    assert lag_spec.reference_tools.count("load_dataset") == 2
    assert lag_spec.reference_tools.count("extract_regional_mean") == 2
    assert "compute_lag_correlation" in lag_spec.required_tools


def test_interpretation_gold_spec_supports_multi_skill_queries():
    specs = load_gold_tool_specs()
    synthesis_spec = specs[233]

    assert synthesis_spec.expected_skill == "ocean_evidence_synthesis"
    assert "ocean_stratification_diagnostics" in synthesis_spec.expected_skills
    assert "ocean_horizontal_advection_attribution" in synthesis_spec.expected_skills
    assert "ocean_proxy_counterfactual_experiment" in synthesis_spec.expected_skills
    assert "assemble_mechanism_evidence_report" in synthesis_spec.required_tools


def test_current_speed_gold_chain_uses_speed_tool():
    queries = {query.id: query for query in load_query_suite()}
    specs = load_gold_tool_specs()

    assert [step.tool for step in queries[97].gold_chain] == [
        "load_dataset",
        "load_dataset",
        "compute_speed_from_uv",
        "compute_hovmoller",
    ]
    assert specs[97].required_tools == ["compute_speed_from_uv", "compute_hovmoller"]


def test_vertical_structure_gold_spec_is_query_aware():
    specs = load_gold_tool_specs()

    assert specs[125].required_tools == [
        "compute_density",
        "identify_mixed_layer_depth",
        "identify_thermocline_depth",
        "identify_pycnocline_depth",
    ]
    assert specs[126].required_tools == ["compute_density", "identify_mixed_layer_depth"]
    assert specs[127].required_tools == ["identify_thermocline_depth"]
    assert specs[128].required_tools == ["compute_density", "identify_pycnocline_depth"]


def test_unified_run_dataframe_uses_gold_tool_spec_metrics():
    specs = load_gold_tool_specs()
    trace = AgentTrace(
        query_id=1,
        query="Calculate the daily SST time series in the region from January to March 2015",
        expected_skill="ocean_timeseries",
        complexity_level="L1",
        result_type="timeseries",
        agent_type="react_qwen",
        run_index=999,
        success=True,
        optimal_steps=2,
        gold_chain_length=2,
        variant_type="base",
        region="SCS",
        time_scale="seasonal",
    )
    trace.record_tool_call("load_dataset", {"variable": "temp"}, True)
    trace.record_tool_call("extract_regional_mean", {}, True)

    frame = build_run_dataframe([trace], gold_specs=specs)
    row = frame.iloc[0]

    assert bool(row["pass"]) is True
    assert bool(row["tool_hit"]) is True
    assert row["gold_required_tool_coverage"] == 1.0
    assert row["gold_required_tools"] == ("extract_regional_mean",)
    assert row["gold_required_tool_count"] == 1
    assert row["matched_gold_required_tool_count"] == 1
    assert row["tool_call_count"] == 2
    assert row["successful_tool_count"] == 2
    assert row["failed_tool_count"] == 0
    assert row["core_steps"] == 2
    assert row["reference_steps"] == 2


def test_tool_hit_requires_all_required_gold_tools():
    specs = load_gold_tool_specs()
    trace = AgentTrace(
        query_id=57,
        query="Compute lag correlation between two regional mean time series",
        expected_skill="ocean_lag_correlation",
        complexity_level="L3",
        result_type="timeseries",
        agent_type="react_qwen",
        run_index=999,
        success=True,
        optimal_steps=7,
        gold_chain_length=7,
        variant_type="base",
        region="SCS",
        time_scale="seasonal",
    )
    trace.record_tool_call("extract_regional_mean", {}, True)

    frame = build_run_dataframe([trace], gold_specs=specs)
    row = frame.iloc[0]

    assert specs[57].required_tools == ["extract_regional_mean", "compute_lag_correlation"]
    assert bool(row["tool_hit"]) is False
    assert row["gold_required_tool_coverage"] == 0.5
    assert row["matched_gold_required_tools"] == ("extract_regional_mean",)
    assert row["gold_required_tool_count"] == 2
    assert row["matched_gold_required_tool_count"] == 1


def test_core_step_match_penalizes_zero_tool_runs():
    specs = load_gold_tool_specs()
    trace = AgentTrace(
        query_id=1,
        query="Calculate the daily SST time series in the region from January to March 2015",
        expected_skill="ocean_timeseries",
        complexity_level="L1",
        result_type="timeseries",
        agent_type="react_qwen",
        run_index=999,
        success=False,
        optimal_steps=2,
        gold_chain_length=2,
        variant_type="base",
        region="SCS",
        time_scale="seasonal",
    )

    frame = build_run_dataframe([trace], gold_specs=specs)
    row = frame.iloc[0]

    assert row["core_steps"] == 0
    assert row["reference_steps"] == 2
    assert row["core_step_match"] == 0.0
    assert row["gold_step_alignment"] == 0.0


def test_react_baseline_agent_ids_stay_frozen():
    assert FROZEN_REACT_AGENT_TYPES == ("react_qwen", "react_claude")
