from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.harness.code_agent import CodeAgent
from packages.harness.data_scope import DataScopeResolver, vertical_spec_from_text
from packages.harness.planner import (
    OceanHarnessPlanner,
    _llm_skill_contract_briefs,
    _llm_skill_header_briefs,
    _llm_skill_workflow_briefs,
)
from packages.harness.planner_agent import PlannerAgent, _default_selector_model_for
from packages.harness.manual_loader import load_skill_specs, parse_workflow_planner_parameters, parse_workflow_steps, select_skill_workflow
from packages.tool_loader.registry import TOOL_CONTRACTS


class FakeHarnessLLMPlanner:
    def __init__(self, payload):
        self.payload = payload
        self.last_kwargs = {}

    def plan_harness_task_graph(self, **kwargs):
        self.last_kwargs = kwargs
        return self.payload


class ExplodingHarnessLLMPlanner:
    def plan_harness_task_graph(self, **kwargs):
        raise AssertionError("condition-mask spatial map requests should not enter the LLM planner")


class FakeChatClient:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.models = []
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.models.append(kwargs["model"])
        self.requests.append(kwargs)
        text = next(self._responses)
        if not isinstance(text, str):
            return text
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text),
                )
            ]
        )


def fake_generated_code_writer(payload):
    return """
def run(inputs, params):
    return {
        "output_type": "generic_result",
        "value": {"generated": True},
        "metadata": {"analysis_design": params.get("analysis_design", {})},
    }
"""


def fake_code_agent():
    return CodeAgent(code_writer=fake_generated_code_writer)


def test_sst_text_resolves_to_surface_depth_range():
    vertical = vertical_spec_from_text(
        "Calculate the daily SST time series in the region (113°E-117°E, 19°N-22°N) from January to March 2015"
    )

    assert vertical.mode == "surface"
    assert vertical.depth_range == (0.0, 0.0)


def test_sst_timeseries_plan_preserves_surface_selection():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "Calculate the daily SST time series in the region (113°E-117°E, 19°N-22°N) from January to March 2015",
        extracted_params={},
        additional_context={},
    )

    surface_steps = [
        step for step in plan["steps"]
        if step["params"].get("vertical_mode") == "surface"
        or step["params"].get("mode") == "surface"
        or step["params"].get("depth_range") == [0.0, 0.0]
    ]
    assert surface_steps


def test_all_skill_workflow_tool_params_match_registry_contracts():
    issues = []
    for skill_id, spec in sorted(load_skill_specs().items()):
        for step in spec.workflow.steps:
            contract = TOOL_CONTRACTS.get(step.tool)
            if contract is None:
                issues.append(f"{skill_id}:{step.save_as} uses unknown tool {step.tool}")
                continue
            allowed = set((contract.get("inputs") or {}).keys())
            extra = sorted(set(step.params_template) - allowed)
            if extra:
                issues.append(f"{skill_id}:{step.save_as}:{step.tool} has unsupported params {extra}")
    assert issues == []


def test_field_producing_skill_steps_are_typed_as_fields():
    skills = load_skill_specs()
    mesoscale_steps = {step.save_as: step for step in skills["ocean_mesoscale_organization_analysis"].workflow.steps}
    layer_steps = {step.save_as: step for step in skills["ocean_layer_mean_analysis"].workflow.steps}
    strat_steps = {step.save_as: step for step in skills["ocean_stratification_diagnostics"].workflow.steps}

    assert mesoscale_steps["front_proximity"].output_artifact["kind"] == "field"
    assert layer_steps["layer_mean_field"].output_artifact["kind"] == "field"
    assert strat_steps["thermo_dataset"].output_artifact["kind"] == "field"


def test_stratification_skill_defaults_to_stability_timeseries():
    skill = load_skill_specs()["ocean_stratification_diagnostics"]
    steps = {step.save_as: step for step in skill.workflow.steps}

    assert steps["temp_field"].params_template["variable"] == "temp"
    assert steps["salt_field"].params_template["variable"] == "salt"
    assert "density_gradient_profile" not in steps
    assert steps["stability_timeseries"].tool == "compute_vertical_stability_timeseries"
    assert steps["stability_timeseries"].output_artifact["frontend_type"] == "timeseries_result"


def test_watermass_event_skill_has_executable_tile_workflow():
    skill = load_skill_specs()["ocean_watermass_event_association"]
    steps = {step.save_as: step for step in skill.workflow.steps}

    assert steps["chlorophyll_field"].params_template["variable"] == "chlorophyll"
    assert steps["temp_field"].params_template["variable"] == "temp"
    assert steps["salt_field"].params_template["variable"] == "salt"
    assert steps["watermass_association"].tool == "compute_watermass_event_association"
    assert steps["watermass_association"].params_template["subregion_grid"] == [30, 30]
    assert steps["hotspot_tile_map"].tool == "build_watermass_tile_map"
    assert steps["dominant_watermass_tile_map"].tool == "build_watermass_tile_map"
    assert steps["watermass_ts_diagram"].tool == "build_watermass_ts_diagram"


def test_watermass_hotspot_query_prefers_watermass_association_workflow():
    workflow = select_skill_workflow(
        "In the East China Sea, divide the region into a 30x30 tile grid and ask whether "
        "surface chlorophyll bloom hotspots are associated with particular water masses."
    )

    assert workflow is not None
    assert workflow.skill_id == "ocean_watermass_event_association"


def test_watermass_association_plan_exposes_classification_step_label():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "In the East China Sea (118E-126E, 24N-33N), divide the region into a 30x30 tile grid "
        "and ask whether surface chlorophyll bloom hotspots from April to June 2015 are associated "
        "with particular water masses. Use temperature and salinity to identify the dominant water "
        "mass in each tile, compare hotspot tiles with the regional background, and explain how "
        "strong the evidence is.",
        extracted_params={},
        additional_context={},
    )

    association_step = next(step for step in plan["steps"] if step["tool"] == "compute_watermass_event_association")
    assert association_step["human_label"] == "Classify water masses"
    assert "Classify dominant water masses" in association_step["technical_label"]


def test_transport_skill_exposes_streamfunction_map_tool_as_executable_workflow():
    skill = load_skill_specs()["ocean_transport_analysis"]
    steps = {step.save_as: step for step in skill.workflow.steps}

    assert steps["transport_streamfunction_map"].tool == "compute_transport_streamfunction_map"
    assert steps["transport_streamfunction_map"].output_artifact["kind"] == "map"


def test_east_china_sea_named_region_counts_as_explicit_region():
    scope = DataScopeResolver().resolve(
        user_request="Analyze surface chlorophyll bloom hotspots in the East China Sea from April to June 2015.",
        extracted_params={},
        additional_context={},
    )

    assert scope.explicit_region is True
    assert scope.lon_range == (118.0, 126.0)
    assert scope.lat_range == (24.0, 33.0)
    assert scope.time_range == ("2015-04-01", "2015-06-30")


def test_explicit_degree_bounds_override_named_region_and_upper_layer_parses():
    scope = DataScopeResolver().resolve(
        user_request=(
            "For the northern South China Sea from January to June 2015, determine whether upper-50 m "
            "salinity changes are controlled more by horizontal advection or local tendency. "
            "Use 110°E–120°E and 18°N–23°N."
        ),
        extracted_params={},
        additional_context={},
    )

    assert scope.lon_range == (110.0, 120.0)
    assert scope.lat_range == (18.0, 23.0)
    assert scope.vertical.depth_range == (0.0, 50.0)


def hypoxia_workflow_code(
    *,
    lon_range="[105.0, 122.0]",
    lat_range="[5.0, 23.0]",
    time_range="['2011-01-01', '2014-12-31']",
    oxygen_threshold=60,
    min_duration_days=3,
):
    return f"""
variables = ['oxygen']
lon_range = {lon_range}
lat_range = {lat_range}
time_range = {time_range}
vertical_mode = 'bottom'
depth_value = None
depth_range = None
depth_aggregation = 'mean'
oxygen_threshold = {oxygen_threshold}
severe_threshold = 20
min_area_km2 = 100
min_duration_days = {min_duration_days}

oxygen_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
hypoxia_detection = detect_hypoxia(
    oxygen=oxygen_field.data,
    oxygen_threshold=oxygen_threshold,
    severe_threshold=severe_threshold,
    min_area_km2=min_area_km2,
    min_duration_days=min_duration_days,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
hypoxic_days = compute_event_summary_map(
    event_detection=hypoxia_detection,
    data=oxygen_field.data,
    summary_mode='event_days',
)
"""


def trend_workflow_code():
    return """
variables = ['temp']
lon_range = [110.0, 120.0]
lat_range = [18.0, 23.0]
time_range = ['2011-01-01', '2012-12-31']
depth_range = None
depth_aggregation = 'mean'
confidence_level = 0.95

raw_data = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
timeseries = extract_regional_mean(
    data=raw_data.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_aggregation=depth_aggregation,
)
trend_result = compute_trend(
    timeseries=timeseries,
    confidence_level=confidence_level,
)
"""


def bloom_workflow_code(*, threshold_line="threshold = 1.0", percentile_line="percentile_threshold = None"):
    return f"""
variables = ['chlorophyll']
lon_range = [110.0, 120.0]
lat_range = [18.0, 23.0]
time_range = ['2011-01-01', '2012-12-31']
vertical_mode = 'surface'
depth_value = None
depth_range = None
depth_aggregation = 'mean'
{threshold_line}
{percentile_line}
min_duration_days = 5
min_area_km2 = 500

chlorophyll_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
bloom_detection = detect_algal_blooms(
    chlorophyll=chlorophyll_field.data,
    threshold=threshold,
    percentile_threshold=percentile_threshold,
    min_duration_days=min_duration_days,
    min_area_km2=min_area_km2,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
"""


def heatwave_workflow_code():
    return """
variables = ['temp']
lon_range = [110.0, 120.0]
lat_range = [18.0, 23.0]
time_range = ['2011-01-01', '2012-12-31']
vertical_mode = 'surface'
depth_value = None
depth_range = None
depth_aggregation = 'mean'
percentile_threshold = 90
min_duration_days = 5
min_area_km2 = 1000

temperature_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
heatwave_detection = detect_heatwaves(
    temp=temperature_field.data,
    percentile_threshold=percentile_threshold,
    min_duration_days=min_duration_days,
    min_area_km2=min_area_km2,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
"""


def spatial_bottom_workflow_code():
    return """
variables = ['salt']
lon_range = [113.0, 124.0]
lat_range = [13.5, 24.5]
time_range = ['2018-01-01', '2018-01-31']
vertical_mode = 'bottom'
depth_value = None
depth_range = None
depth_aggregation = 'mean'
time_aggregation = 'mean'

salinity_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    vertical_mode=vertical_mode,
    depth_value=depth_value,
    depth_range=depth_range,
)
salinity_map = compute_spatial_field(
    data=salinity_field.data,
    time_range=time_range,
    time_aggregation=time_aggregation,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
"""


def budget_workflow_code():
    return """
variables = ['salt']
lon_range = [110.0, 120.0]
lat_range = [18.0, 23.0]
time_range = ['2015-01-01', '2015-06-30']
depth_range = [0.0, 50.0]
depth_aggregation = 'mean'
weighting = 'area_weighted'

tracer_field = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
u_field = load_dataset(
    variable='u',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
v_field = load_dataset(
    variable='v',
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
local_tendency_field = compute_local_tendency(
    data=tracer_field.data,
)
local_tendency_ts = extract_regional_mean(
    data=local_tendency_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
)
horizontal_advection_ts = compute_tracer_horizontal_advection_timeseries(
    data=tracer_field.data,
    u_data=u_field.data,
    v_data=v_field.data,
    lon_range=lon_range,
    lat_range=lat_range,
    depth_range=depth_range,
    depth_aggregation=depth_aggregation,
    weighting=weighting,
)
residual_ts = compute_budget_residual(
    local_tendency=local_tendency_ts,
    horizontal_advection=horizontal_advection_ts,
)
comparison_result = compare_budget_term_magnitudes(
    local_tendency=local_tendency_ts,
    horizontal_advection=horizontal_advection_ts,
    residual=residual_ts,
)
"""


def test_skill_markdown_loads_as_analysis_manual():
    skills = load_skill_specs()
    skill_files = list(Path("skills").glob("*/SKILL.md"))
    known_produce_types = {
        "analysis_result",
        "field_result",
        "mask_result",
        "event_detection_result",
        "spatial_field_result",
        "timeseries_result",
        "trend_result",
        "spectrum_result",
        "profile_result",
        "section_result",
        "hovmoller_result",
        "evidence_report_result",
    }
    known_mask_builders = {"polygon", "isobath", "threshold", "condition", "combined"}

    assert len(skills) == len(skill_files)
    assert all("## Analysis Manual" not in skill_file.read_text(encoding="utf-8") for skill_file in skill_files)
    assert all(skill.header is not None for skill in skills.values())
    assert all(skill.header.skill_id == Path(skill.path).parent.name for skill in skills.values())
    assert all(skill.workflow for skill in skills.values())
    for skill in skills.values():
        header = skill.header
        assert all(item.get("type") in known_produce_types for item in header.produces)
        mask_support = header.mask_support
        assert isinstance(mask_support.get("accepts_analysis_mask"), bool)
        assert set(mask_support.get("can_build_masks", [])) <= known_mask_builders

    heatwave = skills["ocean_heatwave_detection"]
    workflow = heatwave.workflow

    assert workflow.skill_id == "ocean_heatwave_detection"
    assert [step.tool for step in workflow.steps] == [
        "load_dataset",
        "detect_heatwaves",
        "compute_event_summary_map",
        "compute_event_summary_map",
    ]
    assert workflow.steps[0].save_as == "temperature_field"
    assert workflow.required_inputs["variables"] == ["temp"]
    assert workflow.steps[0].output_artifact["dims"] == ["time", "depth", "lat", "lon"]
    assert "marine_heatwave_burden" in workflow.final_artifacts


def test_workflow_parser_converts_artifact_access_and_rejects_arbitrary_python():
    workflow = """
```python
# IMPORTANT: Planner fills shared scope here before instantiating tool calls.
variables = [variable]  # <-- MODIFY: requested variable list in load order
lon_range = None  # <-- MODIFY: west/east bounds from query
lat_range = None  # <-- MODIFY: south/north bounds from query
time_range = None  # <-- MODIFY: analysis window from query
depth_range = None  # <-- MODIFY: optional vertical range from query
raw = load_dataset(
    variable=variables[0],
    lon_range=lon_range,
    lat_range=lat_range,
    time_range=time_range,
    depth_range=depth_range,
)
condition = build_condition_mask(
    fields={"oxygen": raw.data},
    expression="oxygen < 60",  # threshold logic from query
)
masked = apply_mask(data=raw.data, mask=condition.data)
```
"""
    steps = parse_workflow_steps(workflow, manual_id="test")
    parameters = parse_workflow_planner_parameters(workflow, manual_id="test")

    assert [step.tool for step in steps] == ["load_dataset", "build_condition_mask", "apply_mask"]
    assert parameters["variables"]["template"] == ["{variable}"]
    assert parameters["variables"]["doc"] == "<-- MODIFY: requested variable list in load order"
    assert parameters["time_range"]["template"] == "{time_range}"
    assert steps[0].params_template["lon_range"] == "{region.lon_range}"
    assert steps[0].params_template["variable"] == "{variable}"
    assert steps[0].params_template["depth_range"] == "{depth_range}"
    assert steps[1].params_template["fields"] == {"oxygen": "$ref:raw.data"}
    assert steps[1].param_docs["expression"] == "threshold logic from query"
    assert steps[2].params_template["mask"] == "$ref:condition.data"

    bad_workflow = """
```python
import os
masked = apply_mask(data=raw.data, mask=condition.data)
```
"""
    try:
        parse_workflow_steps(bad_workflow, manual_id="bad")
    except ValueError as exc:
        assert "assignment-style tool calls" in str(exc)
    else:
        raise AssertionError("arbitrary Python workflow block was accepted")


def test_planner_agent_parses_markdown_yaml_contract():
    markdown = """
The selected workflow is hypoxia detection.

```yaml
status: ready
route: skill_workflow
selected_skill_id: ocean_hypoxia_detection
resolved_scope:
  variable: oxygen
  lon_range: [105, 122]
  lat_range: [5, 23]
  time_range: ['2011-01-01', '2014-12-31']
parameters:
  oxygen_threshold: 60
task_graph:
  nodes:
    - id: oxygen_field
      tool: load_dataset
      params:
        variable: oxygen
        lon_range: [105, 122]
        lat_range: [5, 23]
      save_as: oxygen_field
missing_fields: []
reason: LLM resolved scope from query.
```
"""
    payload = PlannerAgent.parse_markdown_contract(markdown)

    assert payload["selected_skill_id"] == "ocean_hypoxia_detection"
    assert payload["resolved_scope"]["lon_range"] == [105, 122]
    assert payload["task_graph"]["nodes"][0]["params"]["lat_range"] == [5, 23]


def test_planner_agent_repairs_minor_contract_format_errors():
    unfenced_truncated_json = """
Here is the planner contract:
{
  "status": "ready",
  "route": "skill_workflow",
  "selected_skill_id": "ocean_event_frequency_map",
  "selected_skill_ids": ["ocean_event_frequency_map"],
  "reason": "selected frequency-map workflow",
"""
    payload = PlannerAgent.parse_markdown_contract(unfenced_truncated_json)

    assert payload["status"] == "ready"
    assert payload["selected_skill_ids"] == ["ocean_event_frequency_map"]

    unterminated_fence = """
```yaml
status: ready
route: skill_workflow
selected_skill_id: ocean_heatwave_detection
selected_skill_ids: [ocean_heatwave_detection]
reason: selected heatwave workflow
"""
    payload = PlannerAgent.parse_markdown_contract(unterminated_fence)

    assert payload["route"] == "skill_workflow"
    assert payload["selected_skill_id"] == "ocean_heatwave_detection"


def test_planner_agent_format_repair_does_not_invent_missing_scope():
    incomplete_but_parseable = """
```json
{
  "status": "ready",
  "route": "skill_workflow",
  "selected_skill_id": "ocean_hypoxia_detection",
  "resolved_scope": {
    "variable": "oxygen"
  },
  "missing_fields": []
}
```
"""
    payload = PlannerAgent.parse_markdown_contract(incomplete_but_parseable)

    assert payload["resolved_scope"] == {"variable": "oxygen"}
    assert "lon_range" not in payload["resolved_scope"]
    assert "time_range" not in payload["resolved_scope"]


def test_llm_skill_headers_are_slim_selection_heads():
    headers = _llm_skill_header_briefs()
    assert headers
    expected_keys = {"skill_id", "description", "input_intent", "output_intent", "avoid_when", "variables", "composes_with"}
    assert all(set(item) == expected_keys for item in headers)
    assert all(item["skill_id"] and item["description"] for item in headers)
    assert all(isinstance(item["input_intent"], str) for item in headers)
    assert all(isinstance(item["output_intent"], str) for item in headers)
    assert all(isinstance(item["avoid_when"], list) for item in headers)
    assert all(isinstance(item["composes_with"], list) for item in headers)
    assert all("workflow" not in item for item in headers)


def test_llm_skill_contract_briefs_exclude_workflow_steps():
    details = _llm_skill_contract_briefs(["ocean_hypoxia_detection", "ocean_stratification_diagnostics"])

    assert set(details) == {"ocean_hypoxia_detection", "ocean_stratification_diagnostics"}
    for payload in details.values():
        assert set(payload) == {"contract"}
        contract = payload["contract"]
        assert "steps" not in contract
        assert "workflow" not in contract
        assert "final_artifacts" not in contract
        assert "required_slots" in contract
        assert "optional_slots" in contract
        assert "defaults" in contract
        assert "outputs" in contract

    hypoxia = details["ocean_hypoxia_detection"]["contract"]
    assert hypoxia["event_type"] == "hypoxia"
    assert "variables" in hypoxia["required_slots"]
    assert hypoxia["defaults"]["vertical_mode"] == "bottom"


def test_llm_selected_workflow_details_mark_unbound_refs_as_symbolic():
    details = _llm_skill_workflow_briefs(["ocean_trend_analysis"])
    assert set(details["ocean_trend_analysis"]) == {"workflow"}
    trend = details["ocean_trend_analysis"]
    steps = trend["workflow"]["steps"]
    extract_mean = next(step for step in steps if step["tool"] == "extract_regional_mean")

    assert all("param_docs" not in step for step in steps)
    assert all("planner_note" not in step for step in steps)
    assert all(set(step) <= {"tool", "save_as", "params", "unbound_template_refs"} for step in steps)
    assert all(step["tool"] != "combine_masks" for step in steps)
    assert all("unbound_template_refs" not in step for step in steps)
    assert "unbound_template_refs" not in extract_mean
    assert extract_mean["params"]["data"] == "$ref:raw_data.data"
    assert "reference_rule" in trend["workflow"]["graph_contract"]

    frequency = _llm_skill_workflow_briefs(["ocean_event_frequency_map"])["ocean_event_frequency_map"]
    planner_parameters = frequency["workflow"]["planner_parameters"]
    assert planner_parameters["lon_range"]["template"] == "{region.lon_range}"
    assert "west/east bounds" in planner_parameters["lon_range"]["doc"]
    assert planner_parameters["variables"]["template"] == ["{variable}"]
    assert planner_parameters["variables"]["default"] == ["temp"]


def test_dynamics_vorticity_skill_exposes_map_tool_to_llm():
    headers = {item["skill_id"]: item for item in _llm_skill_header_briefs()}
    dynamics_header = headers["ocean_dynamics_diagnostics"]
    assert "relative vorticity" in dynamics_header["description"].lower()
    assert "2d" in dynamics_header["output_intent"].lower()
    assert "map" in dynamics_header["output_intent"].lower()
    assert any("time-depth" in item.lower() or "hovmoller" in item.lower() for item in dynamics_header["avoid_when"])

    details = _llm_skill_workflow_briefs(["ocean_dynamics_diagnostics"])
    workflow = details["ocean_dynamics_diagnostics"]["workflow"]
    steps = workflow["steps"]

    assert [step["tool"] for step in steps] == [
        "load_dataset",
        "load_dataset",
        "compute_spatial_vorticity_map",
    ]
    assert steps[-1]["params"]["u"] == "$ref:u_field.data"
    assert steps[-1]["params"]["v"] == "$ref:v_field.data"
    assert workflow["planner_parameters"]["season_filter"]["template"] == "{season_filter}"
    assert workflow["planner_parameters"]["time_aggregation"]["default"] == "mean"

    skill = load_skill_specs()["ocean_dynamics_diagnostics"]
    vorticity_step = next(step for step in skill.workflow.steps if step.tool == "compute_spatial_vorticity_map")
    assert vorticity_step.output_artifact["kind"] == "map"
    assert vorticity_step.output_artifact["frontend_type"] == "spatial_field_result"


def test_derived_hovmoller_skill_exposes_masked_vorticity_time_depth_workflow():
    headers = {item["skill_id"]: item for item in _llm_skill_header_briefs()}
    header = headers["ocean_derived_hovmoller_analysis"]
    assert "relative vorticity" in header["description"].lower()
    assert "time-depth" in header["description"].lower()
    assert "relative vorticity" in header["input_intent"].lower()
    assert "time-depth" in header["output_intent"].lower()
    assert "hovmoller" in header["output_intent"].lower()
    assert "polygon" in header["input_intent"].lower()
    assert "isobath" in header["input_intent"].lower()

    details = _llm_skill_workflow_briefs(["ocean_derived_hovmoller_analysis"])
    workflow = details["ocean_derived_hovmoller_analysis"]["workflow"]
    steps = workflow["steps"]

    assert [step["tool"] for step in steps] == [
        "load_dataset",
        "load_dataset",
        "build_polygon_mask",
        "build_isobath_mask",
        "combine_masks",
        "compute_derived_field",
        "apply_mask",
        "compute_hovmoller",
    ]
    assert "compute_spatial_vorticity_map" not in [step["tool"] for step in steps]
    assert steps[2]["params"]["polygon_points"] == "{mask_polygon}"
    assert steps[3]["params"]["isobath_depth"] == "{mask_isobath_depth}"
    assert steps[4]["params"]["masks"] == ["$ref:polygon_mask.data", "$ref:isobath_mask.data"]
    assert steps[5]["params"]["field_type"] == "{field_type}"
    assert steps[6]["params"]["data"] == "$ref:derived_field.data"
    assert steps[6]["params"]["mask"] == "$ref:analysis_mask.data"
    assert steps[7]["params"]["data"] == "$ref:masked_derived_field.data"
    assert workflow["planner_parameters"]["diagram_type"]["default"] == "time_depth"
    assert workflow["planner_parameters"]["mask_isobath_depth"]["default"] is None
    assert workflow["planner_parameters"]["mask_isobath_comparison"]["default"] == "deeper_or_equal"


def test_optional_mask_branch_is_skipped_when_hovmoller_has_no_mask_inputs():
    planner = OceanHarnessPlanner()

    plan = planner.generate_plan_for_query(
        "Use the ocean_derived_hovmoller_analysis workflow to make a time-depth Hovmoller diagram of relative vorticity averaged over 113E-114E and 18N-19N from 2015-01-01 to 2015-01-21.",
        extracted_params={"lon_range": [113.0, 114.0], "lat_range": [18.0, 19.0]},
        additional_context={},
    )

    steps = {step["save_as"]: step for step in plan["steps"]}
    assert "polygon_mask" not in steps
    assert "isobath_mask" not in steps
    assert "analysis_mask" not in steps
    assert "masked_derived_field" not in steps
    assert steps["hovmoller_result"]["params"]["data"] == "$ref:derived_field.data"


def test_optional_mask_branch_runs_when_hovmoller_mask_inputs_are_bound():
    planner = OceanHarnessPlanner()

    plan = planner.generate_plan_for_query(
        "Use the ocean_derived_hovmoller_analysis workflow to make a masked time-depth Hovmoller diagram of relative vorticity.",
        extracted_params={
            "lon_range": [113.0, 114.0],
            "lat_range": [18.0, 19.0],
            "time_range": ["2015-01-01", "2015-01-21"],
            "mask_polygon": [[113.0, 18.0], [114.0, 18.0], [114.0, 19.0], [113.0, 19.0]],
            "mask_isobath_depth": 100,
            "mask_isobath_comparison": "deeper_or_equal",
        },
        additional_context={},
    )

    steps = {step["save_as"]: step for step in plan["steps"]}
    assert steps["polygon_mask"]["params"]["polygon_points"] == [
        [113.0, 18.0],
        [114.0, 18.0],
        [114.0, 19.0],
        [113.0, 19.0],
    ]
    assert steps["isobath_mask"]["params"]["isobath_depth"] == 100
    assert steps["analysis_mask"]["params"]["masks"] == [
        "$ref:polygon_mask.data",
        "$ref:isobath_mask.data",
    ]
    assert steps["masked_derived_field"]["params"]["mask"] == "$ref:analysis_mask.data"
    assert steps["hovmoller_result"]["params"]["data"] == "$ref:masked_derived_field.data"


def test_derived_hovmoller_skill_declares_fixed_source_variables():
    headers = {item["skill_id"]: item for item in _llm_skill_header_briefs()}
    header = headers["ocean_derived_hovmoller_analysis"]
    assert header["variables"] == ["u", "v"]

    workflow = _llm_skill_workflow_briefs(["ocean_derived_hovmoller_analysis"])["ocean_derived_hovmoller_analysis"]["workflow"]
    assert workflow["required_inputs"]["variables"] == ["u", "v"]
    assert workflow["planner_parameters"]["variables"]["template"] == ["u", "v"]
    assert workflow["planner_parameters"]["variables"]["default"] == ["u", "v"]
    assert [step["params"]["variable"] for step in workflow["steps"][:2]] == ["u", "v"]


def test_budget_skill_reduces_terms_to_timeseries_before_comparison():
    workflow = load_skill_specs()["ocean_budget_analysis"].workflow
    steps = {step.save_as: step for step in workflow.steps}

    assert [step.tool for step in workflow.steps] == [
        "load_dataset",
        "load_dataset",
        "load_dataset",
        "compute_local_tendency",
        "extract_regional_mean",
        "compute_tracer_horizontal_advection_timeseries",
        "compute_budget_residual",
        "compare_budget_term_magnitudes",
    ]
    assert steps["local_tendency_ts"].params_template["data"] == "$ref:local_tendency_field.data"
    assert steps["horizontal_advection_ts"].params_template["data"] == "$ref:tracer_field.data"
    comparison_params = steps["comparison_result"].params_template
    assert comparison_params["local_tendency"] == "$ref:local_tendency_ts"
    assert comparison_params["horizontal_advection"] == "$ref:horizontal_advection_ts"
    assert comparison_params["residual"] == "$ref:residual_ts"


def test_llm_budget_workflow_compiles_timeseries_budget_terms():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_budget_analysis",
                "resolved_scope": {
                    "dataset": "current",
                    "variable": "salt",
                    "variables": ["salt", "u", "v"],
                    "lon_range": [110.0, 120.0],
                    "lat_range": [18.0, 23.0],
                    "time_range": ["2015-01-01", "2015-06-30"],
                    "vertical_mode": "depth_range",
                    "depth_range": [0.0, 50.0],
                    "depth_aggregation": "mean",
                },
                "parameters": {},
                "workflow_code": budget_workflow_code(),
                "final_artifacts": ["comparison_result"],
                "missing_fields": [],
                "reason": "upper-50 m salinity budget comparison",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "For the northern South China Sea from January to June 2015, determine whether upper-50 m salinity changes are controlled more by horizontal advection or local tendency. Use 110E-120E and 18N-23N, compare the budget terms in consistent units, and explain the residual uncertainty.",
        extracted_params={},
        additional_context={},
    )

    assert plan["status"] == "ready"
    steps = {step["save_as"]: step for step in plan["steps"]}
    assert steps["local_tendency_field"]["tool"] == "compute_local_tendency"
    assert steps["local_tendency_ts"]["tool"] == "extract_regional_mean"
    assert steps["horizontal_advection_ts"]["tool"] == "compute_tracer_horizontal_advection_timeseries"
    assert steps["residual_ts"]["tool"] == "compute_budget_residual"
    comparison_params = steps["comparison_result"]["params"]
    assert comparison_params["local_tendency"] == "$ref:local_tendency_ts"
    assert comparison_params["horizontal_advection"] == "$ref:horizontal_advection_ts"
    assert comparison_params["residual"] == "$ref:residual_ts"


def test_llm_budget_workflow_todo_uses_python_template_parameter_contract():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_budget_analysis",
                "selected_skill_ids": ["ocean_budget_analysis"],
                "resolved_scope": {
                    "dataset": "current",
                    "variable": "u",
                    "variables": ["u", "v"],
                    "lon_range": [110.0, 120.0],
                    "lat_range": [18.0, 23.0],
                    "time_range": ["2015-01-01", "2015-06-30"],
                    "vertical_mode": "depth_range",
                    "depth_range": [0.0, 50.0],
                    "depth_aggregation": "mean",
                },
                "workflow_todo": {
                    "shared_scope": {
                        "lon_range": [110.0, 120.0],
                        "lat_range": [18.0, 23.0],
                        "time_range": ["2015-01-01", "2015-06-30"],
                        "depth_range": [0.0, 50.0],
                        "depth_aggregation": "mean",
                        "weighting": "area_weighted",
                    },
                    "todo": [
                        {"id": "tracer_field", "tool": "load_dataset", "params": {"variable": "salt"}},
                        {"id": "u_field", "tool": "load_dataset", "params": {"variable": "u"}},
                        {"id": "v_field", "tool": "load_dataset", "params": {"variable": "v"}},
                        {
                            "id": "local_tendency_field",
                            "tool": "compute_local_tendency",
                            "using": {"data": "tracer_field.data"},
                            "params": {},
                        },
                        {
                            "id": "local_tendency_ts",
                            "tool": "extract_regional_mean",
                            "using": {"data": "local_tendency_field.data"},
                            "params": {},
                        },
                        {
                            "id": "horizontal_advection_ts",
                            "tool": "compute_tracer_horizontal_advection_timeseries",
                            "using": {
                                "data": "tracer_field.data",
                                "u_data": "u_field.data",
                                "v_data": "v_field.data",
                            },
                            "params": {},
                        },
                        {
                            "id": "residual_ts",
                            "tool": "compute_budget_residual",
                            "using": {
                                "local_tendency": "local_tendency_ts",
                                "horizontal_advection": "horizontal_advection_ts",
                            },
                            "params": {},
                        },
                        {
                            "id": "comparison_result",
                            "tool": "compare_budget_term_magnitudes",
                            "using": {
                                "local_tendency": "local_tendency_ts",
                                "horizontal_advection": "horizontal_advection_ts",
                                "residual": "residual_ts",
                            },
                            "params": {},
                        },
                    ],
                },
                "final_artifacts": ["horizontal_advection_ts", "residual_ts", "comparison_result"],
                "missing_fields": [],
                "reason": "salinity budget comparison",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "For the northern South China Sea from January to June 2015, determine whether upper-50 m salinity changes are controlled more by horizontal advection or local tendency. Use 110E-120E and 18N-23N.",
        extracted_params={},
        additional_context={},
    )

    steps = {step["save_as"]: step for step in plan["steps"]}
    tracer_load = steps["tracer_field"]["params"]
    assert tracer_load["variable"] == "salt"
    assert tracer_load["depth_range"] == [0.0, 50.0]
    assert "depth_aggregation" not in tracer_load
    assert steps["local_tendency_ts"]["params"]["depth_aggregation"] == "mean"
    assert steps["horizontal_advection_ts"]["params"]["depth_aggregation"] == "mean"
    assert steps["horizontal_advection_ts"]["params"]["weighting"] == "area_weighted"


def test_backend_template_skill_compiles_hypoxia_workflow_and_reuses_load():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_hypoxia_detection",
                "selected_skill_ids": ["ocean_hypoxia_detection"],
                "resolved_scope": {
                    "variable": "oxygen",
                    "variables": ["oxygen"],
                    "lon_range": [119.0, 126.0],
                    "lat_range": [32.0, 39.5],
                    "time_range": ["2020-01-01", "2022-12-31"],
                    "vertical_mode": "bottom",
                    "depth_aggregation": "mean",
                },
                "final_artifacts": ["hypoxic_days", "hypoxia_oxygen_deficit_burden"],
                "missing_fields": [],
                "reason": "bottom hypoxia summary maps",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Use year-by-year evidence on bottom hypoxic days and oxygen-deficit burden in the Yellow Sea from 2020-2022.",
        extracted_params={},
        additional_context={},
    )

    assert plan["status"] == "ready"
    steps = plan["steps"]
    assert [step["tool"] for step in steps].count("load_dataset") == 1
    by_id = {step["save_as"]: step for step in steps}
    assert by_id["hypoxic_days"]["params"]["data"] == "$ref:oxygen_field.data"
    assert by_id["hypoxia_oxygen_deficit_burden"]["params"]["data"] == "$ref:oxygen_field.data"
    assert by_id["hypoxic_days"]["params"]["summary_mode"] == "event_days"
    assert plan["task_graph"]["metadata"]["planner_agent_contract"] == "backend_template.dsl"


def test_semantic_plan_slots_compile_through_backend_template():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_hypoxia_detection",
                "selected_skill_ids": ["ocean_hypoxia_detection"],
                "semantic_plan": {
                    "shared_scope": {
                        "lon_range": [119.0, 126.0],
                        "lat_range": [32.0, 39.5],
                        "time_range": ["2020-01-01", "2022-12-31"],
                    },
                    "skills": [
                        {
                            "skill_id": "ocean_hypoxia_detection",
                            "slots": {
                                "variable": "oxygen",
                                "vertical_mode": "bottom",
                                "depth_aggregation": "mean",
                            },
                        }
                    ],
                    "composition": {"type": "single", "purpose": "bottom hypoxia detection"},
                },
                "missing_fields": [],
                "reason": "semantic hypoxia plan",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Detect low oxygen in the selected shelf region.",
        extracted_params={},
        additional_context={},
    )

    steps = {step["save_as"]: step for step in plan["steps"]}
    assert steps["oxygen_field"]["params"]["variable"] == "oxygen"
    assert steps["oxygen_field"]["params"]["vertical_mode"] == "bottom"
    assert steps["oxygen_field"]["params"]["lon_range"] == [119.0, 126.0]
    assert steps["oxygen_field"]["params"]["time_range"] == ["2020-01-01", "2022-12-31"]
    assert plan["planner_llm_decision"]["resolved_scope"]["variables"] == ["oxygen"]
    assert plan["planner_llm_decision"]["parameters"]["skill_slots"]["ocean_hypoxia_detection"]["variable"] == "oxygen"


def test_backend_template_skill_compiles_multivariable_stratification_workflow():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_stratification_diagnostics",
                "selected_skill_ids": ["ocean_stratification_diagnostics"],
                "resolved_scope": {
                    "variables": ["temp", "salt"],
                    "lon_range": [119.0, 126.0],
                    "lat_range": [32.0, 39.5],
                    "time_range": ["2020-01-01", "2022-12-31"],
                    "vertical_mode": "depth_range",
                    "depth_range": [0.0, 200.0],
                },
                "final_artifacts": ["stability_timeseries"],
                "missing_fields": [],
                "reason": "upper-water stratification evidence",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Compute Yellow Sea stratification stability from 2020 to 2022 using temperature and salinity.",
        extracted_params={},
        additional_context={},
    )

    steps = {step["save_as"]: step for step in plan["steps"]}
    assert steps["temp_field"]["tool"] == "load_dataset"
    assert steps["salt_field"]["tool"] == "load_dataset"
    assert steps["thermo_dataset"]["tool"] == "assemble_dataset"
    assert steps["thermo_dataset"]["params"]["variables"] == {
        "temp": "$ref:temp_field.data",
        "salt": "$ref:salt_field.data",
    }
    assert steps["density_field"]["params"]["data"] == "$ref:thermo_dataset.data"
    assert steps["stability_timeseries"]["params"]["density"] == "$ref:density_field.data"


def test_backend_template_composes_stratification_with_bottom_oxygen_lag_workflow():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_stratification_diagnostics",
                "selected_skill_ids": [
                    "ocean_stratification_diagnostics",
                    "ocean_lag_correlation",
                ],
                "resolved_scope": {
                    "variables": ["temp", "salt", "oxygen"],
                    "lon_range": [119.0, 126.0],
                    "lat_range": [32.0, 39.5],
                    "time_range": ["2018-01-01", "2020-12-31"],
                },
                "final_artifacts": [],
                "missing_fields": [],
                "reason": "Diagnose stratification, then test whether it leads or lags bottom oxygen.",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        (
            "For the Yellow Sea from 2018 to 2020, first diagnose whether stratification strengthened "
            "and whether bottom oxygen was consistent with stratification control, then test whether the "
            "stratification stability time series leads or lags the oxygen time series."
        ),
        extracted_params={},
        additional_context={},
    )

    assert "ocean_stratification_diagnostics" in plan["skills_used"]
    assert "ocean_lag_correlation" in plan["skills_used"]
    steps = {step["save_as"]: step for step in plan["steps"]}
    assert steps["bottom_oxygen_field"]["tool"] == "load_dataset"
    assert steps["bottom_oxygen_field"]["params"]["variable"] == "oxygen"
    assert steps["bottom_oxygen_field"]["params"]["vertical_mode"] == "bottom"
    assert steps["bottom_oxygen_timeseries"]["tool"] == "extract_timeseries"
    assert steps["bottom_oxygen_timeseries"]["params"]["data"] == "$ref:bottom_oxygen_field.data"
    assert steps["stability_oxygen_lag_correlation"]["tool"] == "compute_lag_correlation"
    assert steps["stability_oxygen_lag_correlation"]["params"]["timeseries1"] == "$ref:stability_timeseries"
    assert steps["stability_oxygen_lag_correlation"]["params"]["timeseries2"] == "$ref:bottom_oxygen_timeseries"
    assert plan["task_graph"]["final_artifacts"] == [
        "stability_timeseries",
        "bottom_oxygen_timeseries",
        "stability_oxygen_lag_correlation",
    ]
    assert plan["task_graph"]["metadata"]["planner_agent_contract"] == "backend_template_composition.dsl"
    assert plan["task_graph"]["metadata"]["composition"] == "stratification_oxygen_lag"


def test_backend_template_skill_compiles_mask_branch_from_context_inputs():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_derived_hovmoller_analysis",
                "selected_skill_ids": ["ocean_derived_hovmoller_analysis"],
                "resolved_scope": {
                    "variables": ["u", "v"],
                    "lon_range": [110.0, 121.5],
                    "lat_range": [5.0, 23.5],
                    "time_range": ["2010-01-01", "2022-12-31"],
                    "depth_range": [0.0, 750.0],
                },
                "final_artifacts": ["analysis_mask"],
                "missing_fields": [],
                "reason": "combine drawn polygon and isobath masks",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Use a drawn polygon and keep only areas deeper than or equal to 100 m.",
        extracted_params={
            "mask_polygon": [[110, 5], [121.5, 5], [121.5, 23.5], [110, 23.5]],
            "mask_isobath_depth": 100,
            "mask_isobath_comparison": ">=",
        },
        additional_context={},
    )

    steps = {step["save_as"]: step for step in plan["steps"]}
    assert steps["polygon_mask"]["params"]["polygon_points"] == [
        [110, 5],
        [121.5, 5],
        [121.5, 23.5],
        [110, 23.5],
    ]
    assert steps["analysis_mask"]["tool"] == "combine_masks"
    assert steps["analysis_mask"]["params"]["masks"] == [
        "$ref:polygon_mask.data",
        "$ref:isobath_mask.data",
    ]
    assert "data" not in steps["analysis_mask"]["params"]
    assert "input_2" not in steps["analysis_mask"]["params"]


def test_planner_agent_skips_task_graph_llm_for_skill_workflow_routes():
    client = FakeChatClient(
        [
            '{"status":"ready","route":"skill_workflow","selected_skill_id":"ocean_transport_analysis",'
            '"selected_skill_ids":["ocean_transport_analysis"],"reason":"transport skill"}',
            '{"status":"ready","route":"skill_workflow","selected_skill_id":"ocean_transport_analysis",'
            '"selected_skill_ids":["ocean_transport_analysis"],'
            '"semantic_plan":{"shared_scope":{},'
            '"skills":[{"skill_id":"ocean_transport_analysis","slots":{"variable":"u"}}],'
            '"composition":{"type":"single","purpose":"transport streamfunction"}},"missing_fields":[],"reason":"semantic slots"}',
        ]
    )
    agent = PlannerAgent(
        api_key="test",
        base_url="https://example.test",
        model="planner-main",
        selector_model="selector-fast",
        client=client,
        request_retries=0,
    )

    decision = agent.plan_harness_task_graph(
        user_request="Make a transport streamfunction map.",
        dataset={},
        frontend_extracted_params={},
        workspace_context={},
        conversation_memory={},
        skill_headers=_llm_skill_header_briefs(),
        skill_workflow_loader=_llm_skill_workflow_briefs,
    )

    assert len(client.requests) == 2
    assert decision["planning_mode"] == "semantic_slots"
    assert decision["planner_agent_timings"]["planner.semantic_plan"] >= 0.0
    assert decision["task_graph"]["nodes"] == []
    semantic_payload = client.requests[1]["messages"][-1]["content"]
    assert "selected_skill_contracts" in semantic_payload
    assert '"steps"' not in semantic_payload
    assert "workflow_todo" not in semantic_payload
    assert decision["semantic_plan"]["skills"][0]["slots"]["variable"] == "u"


def test_planner_agent_propagates_policy_making_intent_from_selector():
    client = FakeChatClient(
        [
            '{"status":"ready","route":"skill_workflow","selected_skill_id":"ocean_stratification_diagnostics",'
            '"selected_skill_ids":["ocean_stratification_diagnostics","ocean_lag_correlation"],'
            '"policy_making_intent":false,'
            '"policy_making_reason":"The request asks for scientific mechanism diagnosis, not policy advice.",'
            '"reason":"diagnose stratification and lag relationship"}',
            '{"status":"ready","route":"skill_workflow","selected_skill_id":"ocean_stratification_diagnostics",'
            '"selected_skill_ids":["ocean_stratification_diagnostics","ocean_lag_correlation"],'
            '"semantic_plan":{"shared_scope":{},'
            '"skills":[{"skill_id":"ocean_stratification_diagnostics","slots":{"variables":["temp","salt"],"depth_range":[0,200]}},'
            '{"skill_id":"ocean_lag_correlation","slots":{"max_lag":12}}],'
            '"composition":{"type":"pipeline","purpose":"stratification oxygen lag"}},"missing_fields":[],"reason":"semantic slots"}',
        ]
    )
    agent = PlannerAgent(
        api_key="test",
        base_url="https://example.test",
        model="planner-main",
        selector_model="selector-fast",
        client=client,
        request_retries=0,
    )

    decision = agent.plan_harness_task_graph(
        user_request=(
            "For the Yellow Sea from 2018 to 2020, first diagnose whether stratification strengthened "
            "and whether bottom oxygen was consistent with stratification control, then test whether the "
            "stratification stability time series leads or lags the oxygen time series."
        ),
        dataset={},
        frontend_extracted_params={},
        workspace_context={},
        conversation_memory={},
        skill_headers=_llm_skill_header_briefs(),
        skill_workflow_loader=_llm_skill_workflow_briefs,
    )

    assert len(client.requests) == 2
    assert decision["planning_mode"] == "semantic_slots"
    assert decision["policy_making_intent"] is False
    assert decision["planner_skill_selection"]["policy_making_intent"] is False
    assert decision["semantic_plan"]["composition"]["type"] == "pipeline"


def test_planner_agent_policy_making_intent_is_selector_authoritative():
    client = FakeChatClient(
        [
            '{"status":"ready","route":"generated_code","selected_skill_id":null,'
            '"selected_skill_ids":[],"policy_making_intent":false,'
            '"policy_making_reason":"Scientific mechanism diagnosis, not policymaking.",'
            '"reason":"custom mechanism diagnostic"}',
            '{"status":"ready","route":"generated_code","selected_skill_id":null,'
            '"selected_skill_ids":[],"policy_making_intent":true,'
            '"resolved_scope":{"variables":["temp","salt","oxygen"]},'
            '"parameters":{},"task_graph":{"final_artifacts":[],"nodes":[]},'
            '"missing_fields":[],"clarification_question":"","reason":"planned"}',
        ]
    )
    agent = PlannerAgent(
        api_key="test",
        base_url="https://example.test",
        model="planner-main",
        selector_model="selector-fast",
        client=client,
        request_retries=0,
    )

    decision = agent.plan_harness_task_graph(
        user_request="Explain whether oxygen variability is controlled by stratification.",
        dataset={},
        frontend_extracted_params={},
        workspace_context={},
        conversation_memory={},
        skill_headers=[],
        skill_workflow_loader=lambda skill_ids: {},
    )

    assert len(client.requests) == 2
    assert decision["policy_making_intent"] is False
    assert decision["policy_making_reason"] == "Scientific mechanism diagnosis, not policymaking."


def test_backend_template_skill_uses_query_season_filter_for_transport_loads():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_transport_analysis",
                "selected_skill_ids": ["ocean_transport_analysis"],
                "resolved_scope": {
                    "variables": ["u"],
                    "lon_range": [113.0, 114.0],
                    "lat_range": [18.0, 19.0],
                    "time_range": ["2015-01-01", "2015-12-31"],
                },
                "final_artifacts": ["transport_streamfunction_map"],
                "missing_fields": [],
                "reason": "seasonal load comparison",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Plot the summer mean transport streamfunction map for 2015.",
        extracted_params={},
        additional_context={},
    )

    load_steps = [step for step in plan["steps"] if step["tool"] == "load_dataset"]
    assert [step["save_as"] for step in load_steps] == ["u_field", "v_field"]
    assert load_steps[0]["params"]["season_filter"] == "summer"
    assert load_steps[1]["params"]["season_filter"] == "summer"


def test_backend_template_skill_compiles_transport_streamfunction_exact_tool():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_transport_analysis",
                "selected_skill_ids": ["ocean_transport_analysis"],
                "resolved_scope": {
                    "variables": ["u", "v"],
                    "lon_range": [99.0, 145.0],
                    "lat_range": [0.95, 50.0],
                    "time_range": ["2011-01-01", "2022-12-31"],
                    "depth_range": [0.0, 750.0],
                },
                "workflow_todo": {
                    "shared_scope": {
                        "lon_range": [99.0, 145.0],
                        "lat_range": [0.95, 50.0],
                        "time_range": ["2011-01-01", "2022-12-31"],
                        "depth_range": [0.0, 750.0],
                        "season_filter": "summer",
                        "time_aggregation": "mean",
                    },
                    "todo": [
                        {"id": "u_field", "tool": "load_dataset", "params": {"variable": "u"}},
                        {"id": "v_field", "tool": "load_dataset", "params": {"variable": "v"}},
                        {
                            "id": "transport_streamfunction_map",
                            "tool": "compute_transport_streamfunction_map",
                            "using": ["u_field.data", "v_field.data"],
                            "params": {"regional_gauge": "gan_fig10_china_seas"},
                        },
                    ],
                },
                "final_artifacts": ["transport_streamfunction_map"],
                "missing_fields": [],
                "reason": "summer mean volume transport streamfunction",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "plot the summer mean volume transport streamfunction in the upper 0-750 m over the western Pacific and China Seas for 2011-2022. Use 99E-145E and 0.95N-50N.",
        extracted_params={},
        additional_context={},
    )

    steps = {step["save_as"]: step for step in plan["steps"]}
    streamfunction = steps["transport_streamfunction_map"]
    assert streamfunction["tool"] == "compute_transport_streamfunction_map"
    assert streamfunction["params"]["u"] == "$ref:u_field.data"
    assert streamfunction["params"]["v"] == "$ref:v_field.data"
    assert streamfunction["params"]["depth_range"] == [0.0, 750.0]
    assert streamfunction["params"]["time_aggregation"] == "mean"
    assert "lon_range" not in streamfunction["params"]
    assert "lat_range" not in streamfunction["params"]
    assert steps["u_field"]["params"]["season_filter"] == "summer"
    assert steps["v_field"]["params"]["season_filter"] == "summer"


def test_backend_template_skill_compiles_transport_flux_hovmoller_exact_tool():
    transect = [[119.5, 23.5], [121.5, 25.5]]
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_transport_analysis",
                "selected_skill_ids": ["ocean_transport_analysis"],
                "resolved_scope": {
                    "variables": ["u", "v"],
                    "lon_range": [118.0, 123.0],
                    "lat_range": [22.0, 27.0],
                    "time_range": ["2011-01-01", "2022-12-31"],
                    "depth_range": [0.0, 60.0],
                },
                "workflow_todo": {
                    "shared_scope": {
                        "lon_range": [118.0, 123.0],
                        "lat_range": [22.0, 27.0],
                        "time_range": ["2011-01-01", "2022-12-31"],
                        "depth_range": [0.0, 60.0],
                        "transect_points": transect,
                        "n_samples": 100,
                        "method": "linear",
                    },
                    "todo": [
                        {"id": "u_field", "tool": "load_dataset", "params": {"variable": "u"}},
                        {"id": "v_field", "tool": "load_dataset", "params": {"variable": "v"}},
                        {
                            "id": "transport_flux_hovmoller",
                            "tool": "compute_transect_normal_flux_hovmoller",
                            "using": ["u_field.data", "v_field.data"],
                        },
                    ],
                },
                "final_artifacts": ["transport_flux_hovmoller"],
                "missing_fields": [],
                "reason": "normal volume flux Hovmoller",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Make the 0-60 m time-depth diagram of normal volume flux across the drawn Taiwan Strait transect for 2011-2022.",
        extracted_params={"transect_points": transect},
        additional_context={},
    )

    steps = {step["save_as"]: step for step in plan["steps"]}
    hovmoller = steps["transport_flux_hovmoller"]
    assert hovmoller["tool"] == "compute_transect_normal_flux_hovmoller"
    assert hovmoller["params"]["u"] == "$ref:u_field.data"
    assert hovmoller["params"]["v"] == "$ref:v_field.data"
    assert hovmoller["params"]["depth_range"] == [0.0, 60.0]
    assert hovmoller["params"]["transect_points"] == transect
    assert hovmoller["params"]["n_samples"] == 100
    assert hovmoller["harness_node"]["output"]["kind"] == "hovmoller"
    assert hovmoller["harness_node"]["output"]["frontend_type"] == "hovmoller_result"


def test_backend_template_skill_ignores_generated_python_analysis_todo_branch():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_transport_analysis",
                "selected_skill_ids": ["ocean_transport_analysis"],
                "resolved_scope": {
                    "variables": ["u", "v"],
                    "lon_range": [118.0, 123.0],
                    "lat_range": [22.0, 27.0],
                    "time_range": ["2011-01-01", "2022-12-31"],
                    "depth_range": [0.0, 60.0],
                },
                "workflow_todo": {
                    "todo": [
                        {"id": "u_field", "tool": "load_dataset", "params": {"variable": "u"}},
                        {"id": "v_field", "tool": "load_dataset", "params": {"variable": "v"}},
                        {
                            "id": "transport_flux_hovmoller",
                            "tool": "generated_python_analysis",
                            "using": ["u_field.data", "v_field.data"],
                        },
                    ],
                },
                "final_artifacts": ["transport_flux_hovmoller"],
                "missing_fields": [],
                "reason": "invalid generated-code branch inside transport workflow",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Make the 0-60 m time-depth diagram of normal volume flux across the drawn Taiwan Strait transect for 2011-2022.",
        extracted_params={"transect_points": [[119.5, 23.5], [121.5, 25.5]]},
        additional_context={},
    )

    assert all(step["tool"] != "generated_python_analysis" for step in plan["steps"])
    assert plan["task_graph"]["metadata"]["planner_agent_contract"] == "backend_template.dsl"
    assert plan["task_graph"]["final_artifacts"] == ["transport_flux_hovmoller"]


def test_planner_agent_selects_derived_hovmoller_for_time_depth_vorticity_fixture():
    client = FakeChatClient(
        [
            '{"status":"ready","route":"skill_workflow",'
            '"selected_skill_id":"ocean_derived_hovmoller_analysis",'
            '"selected_skill_ids":["ocean_derived_hovmoller_analysis"],'
            '"reason":"time-depth relative vorticity Hovmoller with polygon and isobath mask"}',
            '{"status":"ready","route":"skill_workflow",'
            '"selected_skill_id":"ocean_derived_hovmoller_analysis",'
            '"selected_skill_ids":["ocean_derived_hovmoller_analysis"],'
            '"semantic_plan":{"shared_scope":{"time_range":["2011-01-01","2022-12-31"]},'
            '"skills":[{"skill_id":"ocean_derived_hovmoller_analysis",'
            '"slots":{"variables":["u","v"],"field_type":"vorticity","diagram_type":"time_depth","depth_range":[0,100]}}],'
            '"composition":{"type":"single","purpose":"time-depth vorticity Hovmoller"}},'
            '"missing_fields":[],"reason":"planned"}',
        ]
    )
    agent = PlannerAgent(
        model="deepseek-v4-pro",
        selector_model="deepseek-v4-flash",
        client=client,
        request_retries=0,
    )

    decision = agent.plan_harness_task_graph(
        user_request=(
            "make a time-depth diagram of area-averaged relative vorticity in the deep South China Sea basin "
            "for 2010-2022. Use the drawn polygon as the analysis mask, and keep only areas deeper than or "
            "equal to 100 m."
        ),
        dataset={},
        frontend_extracted_params={},
        workspace_context={},
        conversation_memory={},
        skill_headers=_llm_skill_header_briefs(),
        skill_workflow_loader=_llm_skill_workflow_briefs,
    )

    assert decision["selected_skill_id"] == "ocean_derived_hovmoller_analysis"
    assert decision["planner_skill_selection"]["selected_skill_ids"] == ["ocean_derived_hovmoller_analysis"]
    selector_payload = client.requests[0]["messages"][-1]["content"]
    assert '"input_intent"' in selector_payload
    assert '"output_intent"' in selector_payload
    assert '"avoid_when"' in selector_payload
    assert '"variables"' in selector_payload
    assert "ocean_dynamics_diagnostics" in selector_payload
    assert "Use ocean_derived_hovmoller_analysis for time-depth or Hovmoller output." in selector_payload
    assert len(client.requests) == 2
    assert decision["planning_mode"] == "semantic_slots"
    semantic_payload = client.requests[1]["messages"][-1]["content"]
    assert "selected_skill_contracts" in semantic_payload
    assert '"steps"' not in semantic_payload
    assert "workflow_todo" not in semantic_payload
    assert decision["resolved_scope"]["variables"] == ["u", "v"]
    assert decision["parameters"]["field_type"] == "vorticity"


def test_planner_agent_uses_small_selector_model_before_main_planner_model():
    client = FakeChatClient(
        [
            """
```yaml
status: ready
route: skill_workflow
selected_skill_id: ocean_heatwave_detection
selected_skill_ids: [ocean_heatwave_detection]
reason: selected heatwave workflow
```
""",
            """
```yaml
status: ready
route: skill_workflow
selected_skill_id: ocean_heatwave_detection
selected_skill_ids: [ocean_heatwave_detection]
semantic_plan:
  shared_scope: {}
  skills:
    - skill_id: ocean_heatwave_detection
      slots:
        variable: temp
        vertical_mode: surface
  composition:
    type: single
    purpose: heatwave map
missing_fields: []
reason: planned
```
""",
        ]
    )
    agent = PlannerAgent(
        model="deepseek-v4-pro",
        selector_model="deepseek-v4-flash",
        client=client,
        request_retries=0,
    )

    decision = agent.plan_harness_task_graph(
        user_request="Show a heatwave map",
        dataset={},
        frontend_extracted_params={},
        workspace_context={},
        conversation_memory={},
        skill_headers=[
            {
                "skill_id": "ocean_heatwave_detection",
                "description": "Detect marine heatwaves.",
                "input_intent": "Temperature fields and heatwave event thresholds.",
                "output_intent": "Heatwave event summary maps.",
                "avoid_when": [],
                "composes_with": [],
            }
        ],
        skill_workflow_loader=lambda skill_ids: {},
    )

    assert client.models == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert decision["planner_skill_selection"]["selector_model"] == "deepseek-v4-flash"
    assert client.requests[0]["response_format"] == {"type": "json_object"}
    assert client.requests[1]["response_format"] == {"type": "json_object"}
    assert decision["planning_mode"] == "semantic_slots"
    assert decision["semantic_plan"]["skills"][0]["slots"]["variable"] == "temp"


def test_planner_agent_reports_reasoning_only_selector_response():
    reasoning_only_response = SimpleNamespace(
        model="deepseek-v4-flash",
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(
                    content="",
                    reasoning_content="thinking but no final contract",
                ),
            )
        ],
        usage=SimpleNamespace(
            completion_tokens=2400,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2400),
        ),
    )
    client = FakeChatClient(
        [
            reasoning_only_response,
            reasoning_only_response,
        ]
    )
    agent = PlannerAgent(
        model="deepseek-v4-pro",
        selector_model="deepseek-v4-flash",
        client=client,
        request_retries=0,
    )

    with pytest.raises(ValueError) as exc_info:
        agent.plan_harness_task_graph(
            user_request="Plot relative vorticity",
            dataset={},
            frontend_extracted_params={},
            workspace_context={},
            conversation_memory={},
            skill_headers=[],
            skill_workflow_loader=lambda skill_ids: {},
        )

    message = str(exc_info.value)
    assert "Planner skill selector failed" in message
    assert "reasoning_content" in message
    assert "finish_reason=length" in message
    assert client.models == ["deepseek-v4-flash", "deepseek-v4-flash"]


def test_planner_agent_retries_reasoning_only_selector_response():
    client = FakeChatClient(
        [
            SimpleNamespace(
                model="deepseek-v4-flash",
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content="", reasoning_content="thinking"),
                    )
                ],
                usage=SimpleNamespace(
                    completion_tokens=2400,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=2400),
                ),
            ),
            '{"status":"ready","route":"generated_code","selected_skill_id":null,"selected_skill_ids":[],"reason":"custom diagnostic"}',
            """
```yaml
status: ready
route: generated_code
selected_skill_id: null
selected_skill_ids: []
resolved_scope:
  variables: [u, v]
parameters: {}
task_graph:
  final_artifacts: []
  nodes: []
missing_fields: []
reason: planned
```
""",
        ]
    )
    agent = PlannerAgent(
        model="deepseek-v4-pro",
        selector_model="deepseek-v4-flash",
        client=client,
        request_retries=0,
    )

    decision = agent.plan_harness_task_graph(
        user_request="Plot a custom relative vorticity map",
        dataset={},
        frontend_extracted_params={},
        workspace_context={},
        conversation_memory={},
        skill_headers=[],
        skill_workflow_loader=lambda skill_ids: {},
    )

    assert decision["route"] == "generated_code"
    assert client.models == ["deepseek-v4-flash", "deepseek-v4-flash", "deepseek-v4-pro"]


def test_planner_agent_retries_reasoning_only_task_graph_response():
    client = FakeChatClient(
        [
            '{"status":"ready","route":"generated_code","selected_skill_id":null,"selected_skill_ids":[],"reason":"custom diagnostic"}',
            SimpleNamespace(
                model="deepseek-v4-pro",
                choices=[
                    SimpleNamespace(
                        finish_reason="length",
                        message=SimpleNamespace(content="", reasoning_content="thinking"),
                    )
                ],
                usage=SimpleNamespace(
                    completion_tokens=6000,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=6000),
                ),
            ),
            '{"status":"ready","route":"generated_code","selected_skill_id":null,'
            '"selected_skill_ids":[],"resolved_scope":{"variables":["u","v"]},'
            '"parameters":{},"task_graph":{"final_artifacts":[],"nodes":[]},'
            '"missing_fields":[],"reason":"planned"}',
        ]
    )
    agent = PlannerAgent(
        model="deepseek-v4-pro",
        selector_model="deepseek-v4-flash",
        client=client,
        request_retries=0,
    )

    decision = agent.plan_harness_task_graph(
        user_request="Plot a custom relative vorticity map",
        dataset={},
        frontend_extracted_params={},
        workspace_context={},
        conversation_memory={},
        skill_headers=[],
        skill_workflow_loader=lambda skill_ids: {},
    )

    assert decision["route"] == "generated_code"
    assert client.models == ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-pro"]


def test_planner_agent_retries_unparseable_task_graph_response():
    client = FakeChatClient(
        [
            '{"status":"ready","route":"generated_code","selected_skill_id":null,'
            '"selected_skill_ids":[],"reason":"needs generated analysis"}',
            "I will now plan the workflow but forgot to return JSON.",
            '{"status":"ready","route":"generated_code","selected_skill_id":null,'
            '"resolved_scope":{"variables":["u","v"]},"parameters":{"season_filter":"DJF"},'
            '"task_graph":{"final_artifacts":["spatial_diagnostic"],"nodes":[]},'
            '"missing_fields":[],"reason":"planned"}',
        ]
    )
    agent = PlannerAgent(
        model="deepseek-v4-pro",
        selector_model="deepseek-v4-flash",
        client=client,
        request_retries=0,
    )

    decision = agent.plan_harness_task_graph(
        user_request="Plot winter mean relative vorticity",
        dataset={},
        frontend_extracted_params={},
        workspace_context={},
        conversation_memory={},
        skill_headers=[],
        skill_workflow_loader=lambda skill_ids: {},
    )

    assert decision["route"] == "generated_code"
    assert decision["parameters"]["season_filter"] == "DJF"
    assert client.models == ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4-pro"]


def test_planner_agent_selector_model_can_be_overridden_or_inferred_for_qwen():
    assert PlannerAgent(model="deepseek-v4-pro", selector_model="selector-fast", client=object()).selector_model == "selector-fast"
    assert _default_selector_model_for(model="deepseek-v4-pro", base_url=None) == "deepseek-v4-flash"
    assert (
        _default_selector_model_for(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-max",
        )
        == "qwen-max"
    )


def test_llm_harness_decision_fills_scope_and_selects_workflow():
    fake_planner = FakeHarnessLLMPlanner(
        {
            "status": "ready",
            "route": "skill_workflow",
            "selected_skill_id": "ocean_hypoxia_detection",
            "resolved_scope": {
                "dataset": "current",
                "variable": "oxygen",
                "variables": ["oxygen"],
                "lon_range": [105.0, 122.0],
                "lat_range": [5.0, 23.0],
                "time_range": ["2011-01-01", "2014-12-31"],
                "vertical_mode": "bottom",
                "depth_aggregation": "mean",
            },
            "parameters": {
                "oxygen_threshold": 60,
                "min_duration_days": 3,
            },
            "workflow_code": hypoxia_workflow_code(),
            "final_artifacts": ["hypoxic_days"],
            "missing_fields": [],
            "reason": "hypoxia workflow with LLM-resolved South China Sea scope",
        }
    )
    planner = OceanHarnessPlanner(
        llm_planner=fake_planner
    )
    plan = planner.generate_plan_for_query(
        "能否查看2011到2014年南海的缺氧区域",
        extracted_params={},
        additional_context={},
    )

    assert plan["status"] == "ready"
    assert "ocean_hypoxia_detection" in plan["skills_used"]
    load_step = plan["steps"][0]
    assert load_step["tool"] == "load_dataset"
    assert load_step["params"]["lon_range"] == [105.0, 122.0]
    assert load_step["params"]["lat_range"] == [5.0, 23.0]
    assert load_step["params"]["time_range"] == ["2011-01-01", "2014-12-31"]
    assert load_step["params"]["vertical_mode"] == "bottom"
    detect_step = plan["steps"][1]
    assert detect_step["params"]["oxygen_threshold"] == 60
    assert detect_step["params"]["min_duration_days"] == 3
    assert plan["planner_llm_decision"]["selected_skill_id"] == "ocean_hypoxia_detection"
    expected_keys = {"skill_id", "description", "input_intent", "output_intent", "avoid_when", "variables", "composes_with"}
    assert all(set(item) == expected_keys for item in fake_planner.last_kwargs["skill_headers"])
    workflow_loader = fake_planner.last_kwargs["skill_workflow_loader"]
    selected_workflows = workflow_loader(["ocean_hypoxia_detection"])
    assert set(selected_workflows) == {"ocean_hypoxia_detection"}
    assert "workflow" in selected_workflows["ocean_hypoxia_detection"]


def test_llm_spatial_bottom_workflow_uses_bottom_mode_not_fixed_depth():
    fake_planner = FakeHarnessLLMPlanner(
        {
            "status": "ready",
            "route": "skill_workflow",
            "selected_skill_id": "ocean_spatial_field_analysis",
            "resolved_scope": {
                "dataset": "current",
                "variable": "salt",
                "variables": ["salt"],
                "lon_range": [113.0, 124.0],
                "lat_range": [13.5, 24.5],
                "time_range": ["2018-01-01", "2018-01-31"],
                "vertical_mode": "bottom",
                "depth_value": None,
                "depth_range": None,
                "depth_aggregation": "mean",
            },
            "parameters": {},
            "workflow_code": spatial_bottom_workflow_code(),
            "final_artifacts": ["salinity_map"],
            "missing_fields": [],
            "reason": "bottom salinity map over South China Sea",
        }
    )
    planner = OceanHarnessPlanner(llm_planner=fake_planner)

    plan = planner.generate_plan_for_query(
        "Plot January 2018 bottom salinity over the South China Sea, domain 113E-124E and 13.5N-24.5N",
        extracted_params={},
        additional_context={},
    )

    assert plan["status"] == "ready"
    assert plan["planner_llm_decision"]["resolved_scope"]["vertical_mode"] == "bottom"
    assert plan["planner_llm_decision"]["resolved_scope"].get("depth_range") is None
    load_step = next(step for step in plan["steps"] if step["tool"] == "load_dataset")
    assert load_step["params"]["variable"] == "salt"
    assert load_step["params"]["vertical_mode"] == "bottom"
    assert "depth_value" not in load_step["params"]
    assert "depth_range" not in load_step["params"]
    map_step = next(step for step in plan["steps"] if step["tool"] == "compute_spatial_field")
    assert "depth_range" not in map_step["params"]
    assert map_step["params"]["depth_aggregation"] == "mean"


def test_llm_budget_workflow_compiles_timeseries_terms_before_comparison():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_budget_analysis",
                "resolved_scope": {
                    "dataset": "current",
                    "variable": "salt",
                    "variables": ["salt", "u", "v"],
                    "lon_range": [110.0, 120.0],
                    "lat_range": [18.0, 23.0],
                    "time_range": ["2015-01-01", "2015-06-30"],
                    "vertical_mode": "depth_range",
                    "depth_range": [0.0, 50.0],
                    "depth_aggregation": "mean",
                },
                "parameters": {},
                "workflow_code": budget_workflow_code(),
                "final_artifacts": ["comparison_result"],
                "missing_fields": [],
                "reason": "salinity budget comparison",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "For the northern South China Sea from January to June 2015, determine whether upper-50 m salinity changes are controlled more by horizontal advection or local tendency. Use 110E-120E and 18N-23N.",
        extracted_params={},
        additional_context={},
    )

    assert plan["status"] == "ready"
    tools = [step["tool"] for step in plan["steps"]]
    assert tools == [
        "load_dataset",
        "load_dataset",
        "load_dataset",
        "compute_local_tendency",
        "extract_regional_mean",
        "compute_tracer_horizontal_advection_timeseries",
        "compute_budget_residual",
        "compare_budget_term_magnitudes",
    ]
    comparison = next(step for step in plan["steps"] if step["tool"] == "compare_budget_term_magnitudes")
    assert comparison["params"]["local_tendency"] == "$ref:local_tendency_ts"
    assert comparison["params"]["horizontal_advection"] == "$ref:horizontal_advection_ts"
    assert comparison["params"]["residual"] == "$ref:residual_ts"


def test_llm_depth_range_is_normalized_positive_down_before_execution():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "generated_code",
                "selected_skill_id": None,
                "resolved_scope": {
                    "variable": "u",
                    "variables": ["u"],
                    "lon_range": [99.0, 145.0],
                    "lat_range": [0.95, 50.0],
                    "time_range": ["2011-01-01", "2022-12-31"],
                    "vertical_mode": "depth_range",
                    "depth_range": [0, -750],
                    "depth_aggregation": "mean",
                },
                "parameters": {},
                "task_graph": {
                    "final_artifacts": ["u_field"],
                    "nodes": [
                        {
                            "id": "load_u",
                            "tool": "load_dataset",
                            "save_as": "u_field",
                            "params": {
                                "variable": "u",
                                "lon_range": [99.0, 145.0],
                                "lat_range": [0.95, 50.0],
                                "time_range": ["2011-01-01", "2022-12-31"],
                                "depth_range": [0, -750],
                            },
                        }
                    ],
                },
                "missing_fields": [],
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "plot upper 0-750 m relative vorticity",
        extracted_params={},
        additional_context={},
    )

    assert plan["planner_llm_decision"]["resolved_scope"]["depth_range"] == [0.0, 750.0]
    assert plan["steps"][0]["params"]["depth_range"] == [0.0, 750.0]


def test_ready_llm_skill_workflow_decision_can_fall_back_to_selected_skill_template():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_hypoxia_detection",
                "resolved_scope": {
                    "variable": "oxygen",
                    "variables": ["oxygen"],
                    "lon_range": [113.0, 117.0],
                    "lat_range": [19.0, 22.0],
                    "time_range": ["2015-01-01", "2015-03-31"],
                    "vertical_mode": "bottom",
                },
                "parameters": {
                    "oxygen_threshold": 60,
                },
                "missing_fields": [],
                "reason": "Intentionally incomplete scope.",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Detect hypoxia in the region (113 E-117 E, 19 N-22 N) from January to March 2015",
        extracted_params={},
        additional_context={},
    )

    assert plan["status"] == "ready"
    assert [step["tool"] for step in plan["steps"]] == [
        "load_dataset",
        "detect_hypoxia",
        "compute_event_summary_map",
        "compute_event_summary_map",
    ]


def test_backend_skill_template_is_source_of_truth_for_skill_workflow():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_trend_analysis",
                "resolved_scope": {
                    "dataset": "current",
                    "variable": "temp",
                    "lon_range": [110.0, 120.0],
                    "lat_range": [18.0, 23.0],
                    "time_range": ["2011-01-01", "2012-12-31"],
                    "vertical_mode": "surface",
                },
                "parameters": {"confidence_level": 0.95},
                "workflow_code": trend_workflow_code(),
                "final_artifacts": ["trend_result"],
                "missing_fields": [],
                "reason": "Trend query with explicit LLM workflow code.",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Analyze temperature trend in 110-120E, 18-23N from 2011 to 2012",
        extracted_params={},
        additional_context={},
    )

    assert [step["tool"] for step in plan["steps"]] == [
        "load_dataset",
        "extract_regional_mean",
        "compute_trend",
    ]
    assert "combine_masks" not in [step["tool"] for step in plan["steps"]]
    assert plan["task_graph"]["final_artifacts"] == ["trend_result"]
    assert plan["task_graph"]["metadata"]["planner_agent_contract"] == "backend_template.dsl"


def test_llm_generated_python_analysis_node_compiles_to_code_execution():
    planner = OceanHarnessPlanner(
        code_agent=fake_code_agent(),
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "generated_code",
                "resolved_scope": {
                    "dataset": "current",
                    "variable": "oxygen",
                    "variables": ["oxygen"],
                    "lon_range": [110.0, 120.0],
                    "lat_range": [18.0, 23.0],
                    "time_range": ["2011-01-01", "2012-12-31"],
                    "vertical_mode": "surface",
                },
                "parameters": {},
                "task_graph": {
                    "final_artifacts": ["oxygen_variability_index"],
                    "nodes": [
                        {
                            "id": "load_oxygen",
                            "tool": "load_dataset",
                            "params": {
                                "variable": "oxygen",
                                "lon_range": [110.0, 120.0],
                                "lat_range": [18.0, 23.0],
                                "time_range": ["2011-01-01", "2012-12-31"],
                            },
                            "save_as": "oxygen_field",
                        },
                        {
                            "id": "compute_custom_index",
                            "tool": "generated_python_analysis",
                            "params": {
                                "input_refs": {
                                    "field": "$ref:oxygen_field.data",
                                },
                                "analysis_design": {
                                    "data": {
                                        "variable": "oxygen",
                                        "vertical_mode": "surface",
                                    },
                                    "analysis": {
                                        "description": (
                                            "Write Python code to compute a custom oxygen variability index from the "
                                            "loaded surface oxygen field. Because no formula was supplied, choose and "
                                            "record a transparent definition in result metadata."
                                        ),
                                    },
                                    "output": {
                                        "output_type": "spatial_field_result",
                                        "frontend": "spatial_field",
                                        "unit": "percent",
                                    },
                                },
                                "code_steps": [
                                    {"id": "prepare_field"},
                                    {"id": "write_custom_variability_index_code"},
                                    {"id": "return_spatial_field_result"},
                                ],
                            },
                            "save_as": "oxygen_variability_index",
                        },
                    ],
                },
                "missing_fields": [],
                "reason": "The request asks for a custom oxygen variability index.",
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Compute a custom oxygen variability index in 110-120E, 18-23N from 2011 to 2012",
        extracted_params={},
        additional_context={},
    )

    assert "generated_code_agent" in plan["skills_used"]
    assert [step["tool"] for step in plan["steps"]] == ["load_dataset", "generated_python_analysis"]
    code_step = plan["steps"][-1]
    assert code_step["harness_node"]["execution"]["strategy"] == "generated_code"
    assert code_step["params"]["input_refs"]["field"] == "$ref:oxygen_field.data"
    assert "analysis_design" not in code_step["params"]
    assert "method" not in code_step["params"]["planner_analysis_design"]["analysis"]
    assert any(step["id"] == "write_custom_variability_index_code" for step in code_step["params"]["planner_code_steps"])
    assert not code_step["harness_node"]["execution"]["code"]
    assert plan["task_graph"]["metadata"]["binding_report"]["code_nodes"] == ["compute_custom_index"]
    assert plan["task_graph"]["final_artifacts"] == ["oxygen_variability_index"]


def test_lag_correlation_query_uses_lag_manual_not_static_map():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "How are surface SST and upper-50 m oxygen related with time lags in the region (123 ° E-127 ° E, 28 ° N-32 ° N) from 2018 to 2020?",
        extracted_params={},
        additional_context={},
    )

    assert "ocean_lag_correlation" in plan["skills_used"]
    assert plan["planner_pipeline"]["stages"][0]["stage"] == "data_scope_resolver"
    assert plan["planner_pipeline"]["stages"][1]["stage"] == "skill_retriever"
    assert plan["planner_pipeline"]["stages"][1]["selected_skill"] == "ocean_lag_correlation"
    assert plan["planner_pipeline"]["stages"][1]["selected_manual"] == "ocean_lag_correlation"
    assert plan["planner_pipeline"]["stages"][2]["stage"] == "analysis_dag_builder"
    assert plan["planner_pipeline"]["stages"][2]["route"] == "skill_workflow"
    assert "ocean_lag_correlation" in plan["data_requirements"]["manual_context"]
    assert plan["data_requirements"]["lon_range"] == [123.0, 127.0]
    assert plan["data_requirements"]["lat_range"] == [28.0, 32.0]
    assert plan["data_requirements"]["time_range"] == ["2018-01-01", "2020-12-31"]
    tools = [step["tool"] for step in plan["steps"]]
    assert tools == [
        "load_dataset",
        "load_dataset",
        "extract_timeseries",
        "extract_timeseries",
        "compute_lag_correlation",
    ]
    assert "compute_spatial_field" not in tools
    assert plan["steps"][0]["params"]["variable"] == "temp"
    assert plan["steps"][0]["params"]["depth_range"] == [0.0, 0.0]
    assert plan["steps"][1]["params"]["variable"] == "oxygen"
    assert plan["steps"][1]["params"]["depth_range"] == [0.0, 50.0]
    assert plan["steps"][-1]["save_as"] == "lag_correlation_raw"

    semantic = plan["semantic_task_graph"]
    assert semantic["metadata"]["planning_model"] == "artifact_first"
    assert semantic["metadata"]["steps_are_projection"] is True
    assert semantic["metadata"]["manual_usage"]["selected_manual"] == "ocean_lag_correlation"
    assert plan["task_graph"]["metadata"]["compiled_steps_from"] == "semantic_task_graph"

    data_reqs = {item["artifact_id"]: item for item in semantic["data_requirements"]}
    assert data_reqs["raw_data1"]["variables"] == ["temp"]
    assert data_reqs["raw_data1"]["vertical"]["mode"] == "surface"
    assert data_reqs["raw_data1"]["vertical"]["depth_range"] == [0.0, 0.0]
    assert data_reqs["raw_data2"]["variables"] == ["oxygen"]
    assert data_reqs["raw_data2"]["vertical"]["mode"] == "depth_range"
    assert data_reqs["raw_data2"]["vertical"]["depth_range"] == [0.0, 50.0]
    assert [task["operation"] for task in semantic["tasks"]] == tools
    assert plan["synthesis_packet"]["final_artifacts"] == ["timeseries1", "timeseries2", "lag_correlation_raw"]
    assert "timeseries1" in plan["synthesis_packet"]["series_artifacts"]


def test_uncovered_analysis_query_uses_generated_code_agent():
    planner = OceanHarnessPlanner(code_agent=fake_code_agent())
    plan = planner.generate_plan_for_query(
        "Compute a custom oxygen variability index for this region from 2010 to 2012",
        extracted_params={"lon_range": [110, 120], "lat_range": [18, 23]},
        additional_context={},
    )

    assert "generated_code_agent" in plan["skills_used"]
    assert plan["semantic_task_graph"]["metadata"]["pipeline_route"] == "generated_code"
    assert [step["tool"] for step in plan["steps"]] == [
        "load_dataset",
        "generated_python_analysis",
    ]
    assert plan["task_graph"]["metadata"]["binding_report"]["code_nodes"] == ["generated_code_analysis"]
    code_step = plan["steps"][-1]
    assert code_step["harness_node"]["execution"]["strategy"] == "generated_code"
    assert not code_step["harness_node"]["execution"]["code"]
    assert code_step["params"]["input_refs"]["field"] == "$ref:oxygen_raw.data"
    assert code_step["params"]["io_contract"]["repair_loop"]["max_attempts"] == 3

    semantic = plan["semantic_task_graph"]
    code_task = next(task for task in semantic["tasks"] if task["operation"] == "generated_python_analysis")
    assert code_task["implementation"]["strategy"] == "generated_code"
    assert code_task["implementation"]["code_contract"]["entrypoint"] == "run(inputs, params) -> dict"
    assert code_task["outputs"] == ["generated_analysis"]


def test_custom_index_query_without_explicit_region_keeps_time_and_generated_code():
    planner = OceanHarnessPlanner(code_agent=fake_code_agent())
    plan = planner.generate_plan_for_query(
        "Compute a custom oxygen variability index for this region from 2010 to 2012",
        extracted_params={},
        additional_context={},
    )

    assert plan["semantic_task_graph"]["metadata"]["pipeline_route"] == "generated_code"
    assert [step["tool"] for step in plan["steps"]] == [
        "load_dataset",
        "generated_python_analysis",
    ]
    assert plan["data_requirements"]["time_range"] == ["2010-01-01", "2012-12-31"]
    assert plan["steps"][0]["params"]["time_range"] == ["2010-01-01", "2012-12-31"]


def test_heatwave_query_uses_skill_workflow_not_static_map():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "Detect marine heatwave events in the region (113 ° E-117 ° E, 19 °N-22 °N) from January to March 2015",
        extracted_params={},
        additional_context={},
    )

    assert plan["planner_kind"] == "ocean_harness"
    assert "ocean_heatwave_detection" in plan["skills_used"]
    assert plan["data_requirements"]["lon_range"] == [113.0, 117.0]
    assert plan["data_requirements"]["lat_range"] == [19.0, 22.0]
    assert plan["data_requirements"]["time_range"] == ["2015-01-01", "2015-03-31"]

    tools = [step["tool"] for step in plan["steps"]]
    assert tools == [
        "load_dataset",
        "detect_heatwaves",
        "compute_event_summary_map",
        "compute_event_summary_map",
    ]
    assert "compute_spatial_field" not in tools
    assert plan["steps"][0]["save_as"] == "temperature_field"
    assert plan["steps"][1]["params"]["temp"] == "$ref:temperature_field.data"
    assert plan["steps"][1]["params"]["vertical_mode"] == "surface"
    assert plan["steps"][1]["params"]["depth_range"] == [0.0, 0.0]
    assert plan["task_graph"]["metadata"]["manual_id"] == "ocean_heatwave_detection"
    assert plan["semantic_task_graph"]["metadata"]["manual_usage"]["selected_manual"] == "ocean_heatwave_detection"
    assert plan["semantic_task_graph"]["metadata"]["pipeline_route"] == "skill_workflow"
    assert "marine_heatwave_burden" in plan["synthesis_packet"]["image_artifacts"]


def test_manual_event_query_requires_explicit_region_and_time():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "Detect marine heatwave events",
        extracted_params={},
        additional_context={},
    )

    assert plan["status"] == "clarification_needed"
    assert "region.lon_range/lat_range" in plan["missing_fields"]
    assert "time_range" in plan["missing_fields"]
    assert plan["steps"] == []


def test_event_frequency_query_uses_event_frequency_manual():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "Show a heatwave hotspot frequency map for 2011 from 110-120E, 18-23N",
        extracted_params={},
        additional_context={},
    )

    assert "ocean_event_frequency_map" in plan["skills_used"]
    assert [step["tool"] for step in plan["steps"]] == [
        "load_dataset",
        "detect_heatwaves",
        "compute_event_frequency_map",
    ]
    assert plan["steps"][-1]["params"]["event_detection"] == "$ref:event_detection"
    assert "events" not in plan["steps"][-1]["params"]
    assert "resolution_deg" not in plan["steps"][-1]["params"]


def test_bloom_condition_query_inserts_condition_mask_before_detection():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "Detect algal bloom events where chlorophyll > p90 in 110-120E, 18-23N from 2010 to 2012",
        extracted_params={},
        additional_context={},
    )

    assert "ocean_bloom_detection" in plan["skills_used"]
    tools = [step["tool"] for step in plan["steps"]]
    assert tools[:3] == ["load_dataset", "build_condition_mask", "apply_mask"]
    condition_step = next(step for step in plan["steps"] if step["tool"] == "build_condition_mask")
    assert condition_step["params"]["fields"] == {"chlorophyll": "$ref:chlorophyll_field.data"}
    assert condition_step["params"]["expression"] == "chlorophyll > percentile(chlorophyll, 90, dim='time')"
    detect_step = next(step for step in plan["steps"] if step["tool"] == "detect_algal_blooms")
    assert detect_step["params"]["chlorophyll"] == "$ref:chlorophyll_field_masked.data"


def test_llm_bloom_absolute_threshold_uses_tool_default_params():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_bloom_detection",
                "resolved_scope": {
                    "dataset": "current",
                    "variable": "chlorophyll",
                    "variables": ["chlorophyll"],
                    "lon_range": [110.0, 120.0],
                    "lat_range": [18.0, 23.0],
                    "time_range": ["2011-01-01", "2012-12-31"],
                    "vertical_mode": "surface",
                },
                "parameters": {"threshold": 1},
                "workflow_code": bloom_workflow_code(threshold_line="threshold = 1"),
                "final_artifacts": ["bloom_detection"],
                "missing_fields": [],
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Detect algal bloom events where chlorophyll > 1 in 110-120E, 18-23N from 2011 to 2012",
        extracted_params={},
        additional_context={},
    )

    detect_step = next(step for step in plan["steps"] if step["tool"] == "detect_algal_blooms")
    assert detect_step["params"]["threshold"] == 1
    assert "percentile_threshold" not in detect_step["params"]


def test_llm_bloom_defaults_to_absolute_threshold_when_no_percentile_requested():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_bloom_detection",
                "resolved_scope": {
                    "dataset": "current",
                    "variable": "chlorophyll",
                    "variables": ["chlorophyll"],
                    "lon_range": [110.0, 120.0],
                    "lat_range": [18.0, 23.0],
                    "time_range": ["2011-01-01", "2012-12-31"],
                    "vertical_mode": "surface",
                },
                "workflow_code": bloom_workflow_code(),
                "final_artifacts": ["bloom_detection"],
                "missing_fields": [],
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Detect algal bloom events in 110-120E, 18-23N from 2011 to 2012",
        extracted_params={},
        additional_context={},
    )

    detect_step = next(step for step in plan["steps"] if step["tool"] == "detect_algal_blooms")
    assert detect_step["params"]["threshold"] == 1.0
    assert "percentile_threshold" not in detect_step["params"]


def test_llm_heatwave_defaults_to_p90_when_no_threshold_requested():
    planner = OceanHarnessPlanner(
        llm_planner=FakeHarnessLLMPlanner(
            {
                "status": "ready",
                "route": "skill_workflow",
                "selected_skill_id": "ocean_heatwave_detection",
                "resolved_scope": {
                    "dataset": "current",
                    "variable": "temp",
                    "variables": ["temp"],
                    "lon_range": [110.0, 120.0],
                    "lat_range": [18.0, 23.0],
                    "time_range": ["2011-01-01", "2012-12-31"],
                    "vertical_mode": "surface",
                },
                "workflow_code": heatwave_workflow_code(),
                "final_artifacts": ["heatwave_detection"],
                "missing_fields": [],
            }
        )
    )

    plan = planner.generate_plan_for_query(
        "Detect marine heatwave events in 110-120E, 18-23N from 2011 to 2012",
        extracted_params={},
        additional_context={},
    )

    detect_step = next(step for step in plan["steps"] if step["tool"] == "detect_heatwaves")
    assert detect_step["params"]["percentile_threshold"] == 90


def test_polygon_and_condition_mask_are_combined_for_generic_map():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "Show oxygen map inside polygon and oxygen below 60 from 2010 to 2012",
        extracted_params={
            "lon_range": [110, 120],
            "lat_range": [18, 23],
            "mask_polygon": [[110, 18], [120, 18], [120, 23], [110, 23]],
        },
        additional_context={},
    )

    tools = [step["tool"] for step in plan["steps"]]
    assert tools == [
        "load_dataset",
        "build_polygon_mask",
        "build_condition_mask",
        "combine_masks",
        "apply_mask",
        "compute_spatial_field",
    ]
    condition_step = next(step for step in plan["steps"] if step["tool"] == "build_condition_mask")
    assert condition_step["params"]["expression"] == "oxygen < 60.0"
    combine_step = next(step for step in plan["steps"] if step["tool"] == "combine_masks")
    assert combine_step["params"]["masks"] == [
        "$ref:polygon_analysis_mask.data",
        "$ref:condition_analysis_mask.data",
    ]
    map_step = plan["steps"][-1]
    assert map_step["params"]["data"] == "$ref:oxygen_raw_masked.data"


def test_multi_variable_condition_mask_query_uses_mask_workflow_route():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "Show regions where oxygen < 60 and temp > 28 in 110-120E, 18-23N from 2020-06-01 to 2020-08-31",
        extracted_params={},
        additional_context={},
    )

    assert "ocean_masking_workflow" in plan["skills_used"]
    assert [step["tool"] for step in plan["steps"]] == [
        "load_dataset",
        "load_dataset",
        "build_condition_mask",
        "apply_mask",
        "compute_spatial_field",
    ]
    condition_step = next(step for step in plan["steps"] if step["tool"] == "build_condition_mask")
    assert condition_step["params"]["fields"] == {
        "oxygen": "$ref:oxygen_raw.data",
        "temp": "$ref:temp_condition_raw.data",
    }
    assert condition_step["params"]["expression"] == "oxygen < 60.0 and temp > 28.0"
    assert plan["steps"][-1]["params"]["data"] == "$ref:oxygen_raw_masked.data"
    assert plan["planner_pipeline"]["stages"][2]["route"] == "condition_mask_spatial_map"


def test_llm_planner_is_not_used_for_first_class_condition_mask_map_route():
    planner = OceanHarnessPlanner(llm_planner=ExplodingHarnessLLMPlanner())
    plan = planner.generate_plan_for_query(
        "Show regions where oxygen < 60 and temp > 28 in 110-120E, 18-23N from 2020-06-01 to 2020-08-31",
        extracted_params={},
        additional_context={},
    )

    assert "ocean_masking_workflow" in plan["skills_used"]
    assert "generated_code_agent" not in plan["skills_used"]
    assert "generated_python_analysis" not in [step["tool"] for step in plan["steps"]]
    assert [step["tool"] for step in plan["steps"]] == [
        "load_dataset",
        "load_dataset",
        "build_condition_mask",
        "apply_mask",
        "compute_spatial_field",
    ]


def test_hypoxia_driver_query_generates_shape_first_plan():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "得到这个区域底层缺氧地底5m以内的长期趋势、时间序列、功率谱，以及与温度、盐度、流速强度是否有关",
        extracted_params={"lon_range": [110, 120], "lat_range": [18, 23]},
        additional_context={},
    )

    assert plan["skill_id"] == "ocean_harness"
    assert plan["planner_kind"] == "ocean_harness"
    step_ids = [step["step_id"] for step in plan["steps"]]
    assert "select_oxygen_layer" in step_ids
    assert "hypoxia_timeseries" in step_ids
    assert "hypoxia_trend" in step_ids
    assert "hypoxia_spectrum" in step_ids
    assert "temp_lag_correlation" in step_ids
    assert "salt_lag_correlation" in step_ids
    assert "speed_lag_correlation" in step_ids

    task_graph = plan["task_graph"]
    assert task_graph["final_artifacts"] == [
        "hypoxia_timeseries",
        "hypoxia_trend",
        "hypoxia_spectrum",
        "temp_lag_correlation",
        "salt_lag_correlation",
        "speed_lag_correlation",
    ]
    vertical_step = next(step for step in plan["steps"] if step["step_id"] == "select_oxygen_layer")
    assert vertical_step["params"]["mode"] == "bottom_band"
    assert vertical_step["params"]["band_thickness_m"] == 5.0
    semantic = plan["semantic_task_graph"]
    artifacts = {item["artifact_id"]: item for item in semantic["artifacts"]}
    assert artifacts["hypoxia_mask"]["kind"] == "mask"
    assert artifacts["hypoxia_mask"]["role"] == "mask_artifact"
    assert artifacts["oxygen_selected_bottom_depth"]["role"] == "auxiliary_vertical_metadata"
    assert artifacts["oxygen_selected_bottom_depth"]["dims"] == ["lat", "lon"]
    assert artifacts["oxygen_selected_valid_bottom_band_mask"]["kind"] == "mask"
    assert artifacts["oxygen_selected_valid_bottom_band_mask"]["dims"] == ["depth", "lat", "lon"]
    mask_requirement_ids = {item["artifact_id"] for item in semantic["mask_requirements"]}
    assert "hypoxia_mask" in mask_requirement_ids
    assert any(item["role"] == "valid_bottom_band_mask" for item in semantic["mask_requirements"])
    assert semantic["metadata"]["mask_policy"]["depth_selection"] == "vertical_selection_not_roi_mask"
    assert semantic["metadata"]["mask_policy"]["local_bottom"] == "bottom/bottom_band use per-cell deepest valid wet level, not one global depth"


def test_bottom_depth_semantics_are_per_cell_not_fixed_depth():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "Analyze bottom oxygen time series in this region from 2010 to 2012",
        extracted_params={"lon_range": [110, 120], "lat_range": [18, 23]},
        additional_context={},
    )

    assert plan["status"] == "ready"
    assert [step["tool"] for step in plan["steps"]] == ["load_dataset", "select_vertical", "extract_timeseries"]
    data_req = plan["semantic_task_graph"]["data_requirements"][0]
    assert data_req["vertical"]["mode"] == "bottom"
    assert data_req["vertical"]["coordinate_mode"] == "per_cell_local_bottom"
    assert data_req["vertical"]["selector_type"] == "deepest_valid_wet_level"
    assert data_req["vertical"]["is_roi_mask"] is False

    select_task = next(task for task in plan["semantic_task_graph"]["tasks"] if task["operation"] == "select_vertical")
    assert select_task["implementation"]["params"]["mode"] == "bottom"
    artifacts = {item["artifact_id"]: item for item in plan["semantic_task_graph"]["artifacts"]}
    assert artifacts["oxygen_selected_bottom_depth"]["provenance"]["possible_dims"] == [
        ["lat", "lon"],
        ["time", "lat", "lon"],
    ]


def test_generic_trend_query_keeps_router_independent_plan_shape():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "Analyze temperature trend in this region from 2010 to 2012",
        extracted_params={"lon_range": [110, 120], "lat_range": [18, 23]},
        additional_context={},
    )

    assert plan["skill_id"] == "ocean_harness"
    assert [step["step_id"] for step in plan["steps"]][-1] == "compute_trend"
    assert plan["data_requirements"]["time_range"] == ["2010-01-01", "2012-12-31"]


def test_explicit_query_scope_overrides_workspace_defaults():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "Analyze temperature trend in 110-120E, 18-23N from 2010 to 2012",
        extracted_params={},
        additional_context={
            "workspace_context": {
                "time_range": ["2011-01-01", "2011-01-01"],
                "region_bounds": {
                    "lon_min": 98.8,
                    "lon_max": 143.2,
                    "lat_min": 0.9,
                    "lat_max": 48.6,
                },
            },
        },
    )

    assert plan["data_requirements"]["lon_range"] == [110.0, 120.0]
    assert plan["data_requirements"]["lat_range"] == [18.0, 23.0]
    assert plan["data_requirements"]["time_range"] == ["2010-01-01", "2012-12-31"]
    load_step = next(step for step in plan["steps"] if step["tool"] == "load_dataset")
    assert load_step["params"]["lon_range"] == [110.0, 120.0]
    assert load_step["params"]["lat_range"] == [18.0, 23.0]
    assert load_step["params"]["time_range"] == ["2010-01-01", "2012-12-31"]


def test_explicit_query_scope_overrides_frontend_extracted_defaults():
    planner = OceanHarnessPlanner()
    plan = planner.generate_plan_for_query(
        "Analyze temperature trend in 110-120E, 18-23N from 2010 to 2012",
        extracted_params={
            "lon_range": [98.8, 143.2],
            "lat_range": [0.9, 48.6],
            "time_range": ["2011-01-01", "2011-01-01"],
        },
        additional_context={},
    )

    assert plan["data_requirements"]["lon_range"] == [110.0, 120.0]
    assert plan["data_requirements"]["lat_range"] == [18.0, 23.0]
    assert plan["data_requirements"]["time_range"] == ["2010-01-01", "2012-12-31"]
    load_step = next(step for step in plan["steps"] if step["tool"] == "load_dataset")
    assert load_step["params"]["lon_range"] == [110.0, 120.0]
    assert load_step["params"]["lat_range"] == [18.0, 23.0]
    assert load_step["params"]["time_range"] == ["2010-01-01", "2012-12-31"]
