from types import SimpleNamespace

from benchmarks.core_tools import get_skill_core_tools
from benchmarks.gold_types import GoldStep
from benchmarks.oceanmind_runner import OceanMindRunner
from benchmarks.query_suite import QueryCase
from benchmarks.rule_based_review import score_trace
from benchmarks.trace import AgentTrace


class FakeAdapter:
    def __init__(self, model: str, input_tokens: int, output_tokens: int):
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def create_message(self, **kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=self.input_tokens,
                completion_tokens=self.output_tokens,
            )
        )


class FakePlannerAgent:
    DEFAULT_MODEL = "fake-default-model"
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._selector_adapter = FakeAdapter("fake-selector-model", 11, 3)
        self._adapter = FakeAdapter("fake-planner-model", 17, 5)
        self.__class__.instances.append(self)


class FakeHarnessExecutor:
    calls = []

    def __init__(self):
        pass

    async def execute_query(self, user_request, extracted_params, additional_context, planner, planner_kwargs):
        self.__class__.calls.append(
            {
                "user_request": user_request,
                "extracted_params": extracted_params,
                "additional_context": additional_context,
                "planner": planner,
                "planner_kwargs": planner_kwargs,
            }
        )
        planner._selector_adapter.create_message(
            client=object(),
            max_tokens=10,
            temperature=0.0,
            system="",
            messages=[],
            request_name="fake_selector",
            json_response=True,
        )
        planner._adapter.create_message(
            client=object(),
            max_tokens=10,
            temperature=0.0,
            system="",
            messages=[],
            request_name="fake_planner",
            json_response=True,
        )
        yield {"type": "planning_start", "user_request": user_request}
        yield {
            "type": "plan_generated",
            "skill_id": "ocean_harness",
            "skills_used": ["ocean_harness", "ocean_spatial_field_analysis"],
            "plan": {"steps": []},
        }
        yield {"type": "plan_start", "skill_id": "ocean_harness", "n_steps": 2}
        yield {
            "type": "step_start",
            "step_id": "load_temp",
            "tool": "load_dataset",
            "params": {"variable": "temp"},
        }
        yield {
            "type": "step_complete",
            "step_id": "load_temp",
            "tool": "load_dataset",
            "params": {"variable": "temp"},
            "result_id": "temp_field",
            "output_type": "data_container_result",
            "result_summary": {},
        }
        yield {
            "type": "step_start",
            "step_id": "temp_map",
            "tool": "compute_spatial_field",
            "params": {"data": "$ref:temp_field.data"},
        }
        yield {
            "type": "step_complete",
            "step_id": "temp_map",
            "tool": "compute_spatial_field",
            "params": {"data": "$ref:temp_field.data"},
            "result_id": "temp_map",
            "output_type": "spatial_field_result",
            "result_summary": {},
        }
        yield {
            "type": "plan_complete",
            "skill_id": "ocean_harness",
            "skills_used": ["ocean_harness", "ocean_spatial_field_analysis"],
            "results": ["temp_field", "temp_map"],
            "n_steps_completed": 2,
        }


def test_oceanmind_benchmark_runner_uses_current_harness_path(monkeypatch):
    import packages.harness.executor as harness_executor_module
    import packages.harness.planner_agent as planner_agent_module
    import packages.runtime as runtime_module

    FakePlannerAgent.instances = []
    FakeHarnessExecutor.calls = []
    monkeypatch.setattr(planner_agent_module, "PlannerAgent", FakePlannerAgent)
    monkeypatch.setattr(harness_executor_module, "OceanHarnessExecutor", FakeHarnessExecutor)
    monkeypatch.setattr(runtime_module, "get_active_dataset_context", lambda: {"dataset_id": "fake"})

    query_case = QueryCase(
        id=1,
        query="Show the spatial distribution of sea surface temperature in January 2015",
        expected_skill="ocean_spatial_field_analysis",
        complexity_level="L1",
        result_type="spatial_field",
        optimal_steps=2,
        variant_type="base",
        region="SCS",
        time_scale="monthly",
        gold_chain=[
            GoldStep(tool="load_dataset"),
            GoldStep(tool="compute_spatial_field", refs={"data": [0]}),
        ],
    )

    runner = OceanMindRunner(
        model="benchmark-model",
        api_key="benchmark-key",
        base_url="https://benchmark.invalid/v1",
        agent_type="OceanMind_test",
    )

    trace = runner.run(query_case)

    assert trace.success is True
    assert trace.selected_skill == "ocean_spatial_field_analysis"
    assert trace.skills_used == ["ocean_harness", "ocean_spatial_field_analysis"]
    assert [call.tool_name for call in trace.tool_calls] == ["load_dataset", "compute_spatial_field"]
    assert trace.gold_match_success is True
    assert [(call.purpose, call.model) for call in trace.llm_calls] == [
        ("planner_selector", "fake-selector-model"),
        ("planner_task_graph", "fake-planner-model"),
    ]
    assert trace.total_input_tokens == 28
    assert trace.total_output_tokens == 8

    planner = FakePlannerAgent.instances[0]
    assert planner.kwargs["model"] == "benchmark-model"
    assert planner.kwargs["api_key"] == "benchmark-key"
    assert planner.kwargs["base_url"] == "https://benchmark.invalid/v1"
    call = FakeHarnessExecutor.calls[0]
    assert call["planner"] is planner
    assert call["planner_kwargs"] == {"max_replans": 2}
    assert call["additional_context"]["dataset_id"] == "fake"
    assert call["additional_context"]["benchmark_policy"]["disable_clarification"] is True
    assert call["additional_context"]["workspace_context"]["benchmark_policy"]["disable_clarification"] is True


def test_benchmark_core_tools_follow_current_skill_workflows():
    transport_tools = get_skill_core_tools("ocean_transport_analysis")
    assert "compute_transport_streamfunction_map" in transport_tools
    assert "compute_transect_normal_flux_hovmoller" in transport_tools

    dynamics_tools = get_skill_core_tools("ocean_dynamics_diagnostics")
    assert "compute_spatial_vorticity_map" in dynamics_tools


def test_rule_review_requires_oceanmind_selector_and_planner_protocol():
    query_case = _spatial_query_case()
    trace = _successful_spatial_trace(with_llm_calls=False)

    record = score_trace(trace, query_case)

    assert record.pass_decision is False
    assert record.gate_execution is False
    assert "missing required selector+planner" in record.reason


def test_rule_review_rejects_oceanmind_clarification_needed():
    query_case = _spatial_query_case()
    trace = _successful_spatial_trace(with_llm_calls=True)
    trace.error = "clarification_needed"
    trace.error_category = "clarification_needed"

    record = score_trace(trace, query_case)

    assert record.pass_decision is False
    assert record.gate_execution is False
    assert "must not return clarification_needed" in record.reason


def _spatial_query_case() -> QueryCase:
    return QueryCase(
        id=2,
        query=(
            "Show the spatial distribution of sea surface temperature in "
            "(110°E-120°E, 18°N-23°N) from 2015-01-01 to 2015-01-31"
        ),
        expected_skill="ocean_spatial_field_analysis",
        complexity_level="L1",
        result_type="spatial_field",
        optimal_steps=2,
        variant_type="base",
        region="SCS",
        time_scale="monthly",
        gold_chain=[
            GoldStep(
                tool="load_dataset",
                semantic_params={
                    "variable": "temp",
                    "lon_range": [110, 120],
                    "lat_range": [18, 23],
                    "time_range": ["2015-01-01", "2015-01-31"],
                },
            ),
            GoldStep(tool="compute_spatial_field", refs={"data": [0]}),
        ],
    )


def _successful_spatial_trace(*, with_llm_calls: bool) -> AgentTrace:
    trace = AgentTrace(
        query_id=2,
        query=(
            "Show the spatial distribution of sea surface temperature in "
            "(110°E-120°E, 18°N-23°N) from 2015-01-01 to 2015-01-31"
        ),
        expected_skill="ocean_spatial_field_analysis",
        complexity_level="L1",
        result_type="spatial_field",
        agent_type="OceanMind",
        run_index=0,
        success=True,
        selected_skill="ocean_spatial_field_analysis",
        skills_used=["ocean_harness", "ocean_spatial_field_analysis"],
        optimal_steps=2,
        gold_chain_length=2,
    )
    if with_llm_calls:
        trace.record_llm_call("selector-model", 10, 2, 1.0, "planner_selector")
        trace.record_llm_call("planner-model", 20, 4, 1.0, "planner_task_graph")
    trace.record_tool_call(
        "load_dataset",
        {
            "variable": "temp",
            "lon_range": [110, 120],
            "lat_range": [18, 23],
            "time_range": ["2015-01-01", "2015-01-31"],
        },
        success=True,
    )
    trace.record_tool_call(
        "compute_spatial_field",
        {"data": "$ref:temp_field.data"},
        success=True,
    )
    return trace
