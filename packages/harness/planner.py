"""Shape-first planner replacing skill workflow planning for dataset analysis."""

from __future__ import annotations

import calendar
import json
import math
import re
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from packages.harness.ir import ExecutionSpec, ExecutionStrategy, NodeType, TaskNode, task_graph_from_nodes
from packages.harness.llm_contract import XMLIOContract, parse_json_from_xml_output, render_xml_io_contract
from packages.harness.manual_loader import (
    WorkflowTemplate,
    load_skill_specs,
    parse_workflow_planner_parameters,
    parse_workflow_steps,
    retrieve_skill_specs,
)
from packages.harness.data_scope import (
    VARIABLE_ALIASES,
    DataScopeResolver,
    infer_primary_variable,
    vertical_spec_from_text,
)
from packages.harness.code_agent import CodeAgent
from packages.harness.pipeline import HarnessPlanningPipeline
from packages.harness.result_gatherer import ArtifactGatherer, packet_spec_to_dict
from packages.harness.semantic_graph import semantic_graph_from_task_graph, semantic_graph_to_dict
from packages.harness.shapes import ShapeClass, shape_spec_from_dims
from packages.harness.specs import ArtifactKind, ArtifactSpec, FrontendType, ReadSpec, VerticalSpec
from packages.harness.tool_binder import ToolBinder
from packages.harness.contracts import validate_task_graph_contracts
from packages.runtime import get_active_dataset_public_config


_VARIABLE_ALIASES: Dict[str, Tuple[str, ...]] = VARIABLE_ALIASES

_NAMED_REGION_BOUNDS: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "east china sea": ((118.0, 126.0), (24.0, 33.0)),
    "east china sea shelf": ((118.0, 126.0), (24.0, 33.0)),
    "yellow sea": ((119.0, 126.0), (32.0, 39.5)),
    "bohai sea": ((117.0, 122.2), (37.0, 41.3)),
    "south china sea": ((108.0, 121.5), (5.0, 23.5)),
    "pearl river estuary": ((112.0, 116.5), (20.0, 24.0)),
}

_WORKFLOW_TODO_BASE_TOOLS = {"load_dataset", "assemble_dataset"}


class OceanHarnessPlanner:
    """Generate a shape-first task graph and frontend-compatible plan."""

    def __init__(
        self,
        pipeline: Optional[HarnessPlanningPipeline] = None,
        llm_planner: Optional[Any] = None,
        code_agent: Optional[CodeAgent] = None,
    ) -> None:
        self.pipeline = pipeline or HarnessPlanningPipeline()
        self.llm_planner = llm_planner
        self.code_agent = code_agent or CodeAgent()

    @staticmethod
    def render_llm_io_contract(
        *,
        task: str,
        input_payload: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        rules: Iterable[str] = (),
    ) -> str:
        return render_xml_io_contract(
            XMLIOContract(
                task=task,
                input_payload=input_payload,
                output_schema=output_schema,
                rules=tuple(rules),
            )
        )

    @staticmethod
    def parse_llm_output_json(
        text: str,
        *,
        required_keys: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        return parse_json_from_xml_output(text, required_keys=required_keys, expect_object=True)

    def generate_plan_for_query(
        self,
        user_request: str,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        context = PlanningContext.from_inputs(
            user_request=user_request,
            extracted_params=extracted_params or {},
            additional_context=additional_context or {},
        )
        if self.llm_planner is not None:
            if _is_condition_mask_spatial_map_request(context):
                return self._condition_mask_map_plan(context)
            decision = self._generate_llm_harness_decision(
                user_request=user_request,
                extracted_params=extracted_params or {},
                additional_context=additional_context or {},
            )
            return self._plan_from_llm_harness_decision(
                decision,
                user_request=user_request,
                extracted_params=extracted_params or {},
                additional_context=additional_context or {},
            )

        return self.pipeline.run(
            user_request=user_request,
            extracted_params=extracted_params or {},
            additional_context=additional_context or {},
            factory=self,
            context=context,
        )

    def _generate_llm_harness_decision(
        self,
        *,
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        planner_agent = self.llm_planner
        if planner_agent is None:
            raise RuntimeError("LLM harness planner is not configured.")
        if not hasattr(planner_agent, "plan_harness_task_graph"):
            raise TypeError("OceanHarnessPlanner requires a planner agent with plan_harness_task_graph().")
        decision = planner_agent.plan_harness_task_graph(
            user_request=user_request,
            dataset=get_active_dataset_public_config(),
            frontend_extracted_params=extracted_params,
            workspace_context=additional_context.get("workspace_context", {}),
            conversation_memory=additional_context.get("conversation_memory", {}),
            skill_headers=_llm_skill_header_briefs(),
            skill_workflow_loader=_llm_skill_workflow_briefs,
            skill_contract_loader=_llm_skill_contract_briefs,
        )
        if not isinstance(decision, dict):
            raise ValueError("OceanHarness LLM planner returned a non-object payload.")
        return decision

    def _plan_from_llm_harness_decision(
        self,
        decision: Dict[str, Any],
        *,
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = _normalize_llm_harness_decision(decision)
        route = str(normalized.get("route") or "generic_map")
        if normalized.get("status") != "clarification_needed":
            _merge_semantic_plan_into_decision(normalized)
        if normalized.get("status") != "clarification_needed":
            _merge_workflow_todo_scope_into_decision(normalized)
        workflow: Optional[WorkflowTemplate] = None
        planner_agent_contract = "backend_template.dsl"
        if (
            normalized.get("status") != "clarification_needed"
            and route == "skill_workflow"
            and _decision_has_workflow_code(normalized)
        ):
            workflow = _workflow_from_llm_workflow_code(normalized)
            _merge_workflow_code_scope_into_decision(normalized, workflow)
            planner_agent_contract = "planner_workflow_code.dsl"
        else:
            workflow = _workflow_from_llm_decision(normalized)
        extracted = _merge_llm_parameters_into_extracted(extracted_params, normalized)
        resolved_scope = normalized.get("resolved_scope") if isinstance(normalized.get("resolved_scope"), Mapping) else {}
        has_resolved_scope = bool(resolved_scope)
        strict_scope = normalized.get("status") != "clarification_needed" and route != "dataset_info" and has_resolved_scope
        context_payload = dict(additional_context or {})
        if has_resolved_scope:
            context_payload["planner_resolved_scope"] = resolved_scope
            context_payload["planner_resolved_scope_strict"] = strict_scope
            context_payload["planner_scope_required_fields"] = (
                ["lon_range", "lat_range", "time_range", "variables"] if strict_scope else []
            )
        context_payload["planner_llm_decision"] = normalized
        context = PlanningContext.from_inputs(
            user_request=user_request,
            extracted_params=extracted,
            additional_context=context_payload,
        )

        if normalized.get("status") == "clarification_needed":
            missing = normalized.get("missing_fields") or []
            if workflow is not None:
                plan = _clarification_plan(context, workflow, missing)
            else:
                plan = _generic_llm_clarification_plan(context, missing, normalized)
            plan["planner_llm_decision"] = normalized
            return plan

        plan: Optional[Dict[str, Any]] = None
        selected_skill_ids = _decision_selected_skill_ids(normalized)
        if route == "skill_workflow":
            if _decision_has_workflow_code(normalized) and workflow is not None:
                plan = self._skill_workflow_plan(
                    context,
                    workflow,
                    planner_agent_contract=planner_agent_contract,
                    apply_query_masks=False,
                    decision=normalized,
                )
            else:
                composed_plan = self._composed_skill_workflow_plan(
                    context,
                    selected_skill_ids,
                    decision=normalized,
                )
                if composed_plan is not None:
                    plan = composed_plan
                elif workflow is not None:
                    plan = self._skill_workflow_plan(
                        context,
                        workflow,
                        planner_agent_contract=planner_agent_contract,
                        apply_query_masks=False,
                        decision=normalized,
                    )
        if plan is None and (
            llm_nodes := _nodes_from_llm_task_graph(normalized, context, code_agent=self.code_agent)
        ):
            final_artifacts = _llm_final_artifacts(normalized, llm_nodes)
            uses_generated_code = any(
                node.execution is not None and node.execution.strategy == ExecutionStrategy.CODE
                for node in llm_nodes
            )
            plan = _plan_from_nodes(
                context,
                llm_nodes,
                final_artifacts=final_artifacts,
                skill_id="ocean_harness",
                skills_used=[
                    "ocean_harness",
                    *(["generated_code_agent"] if uses_generated_code else []),
                    *(
                        [str(normalized.get("selected_skill_id")).strip()]
                        if str(normalized.get("selected_skill_id") or "").strip()
                        else []
                    ),
                ],
                metadata={
                    "planner": "OceanHarnessPlanner",
                    "planner_agent_contract": "task_graph.nodes",
                    "selected_skill_id": normalized.get("selected_skill_id"),
                    "route": route,
                },
            )
        elif plan is None and route == "dataset_info":
            plan = self._dataset_info_plan(context)
        elif plan is None and route == "pair_lag_relationship":
            plan = self._pair_lag_relationship_plan(context)
        elif plan is None and route == "hypoxia_driver":
            plan = self._hypoxia_driver_plan(context)
        elif plan is None and route == "generic_timeseries":
            plan = self._generic_timeseries_plan(
                context,
                include_trend=bool(normalized.get("include_trend")),
                include_spectrum=bool(normalized.get("include_spectrum")),
            )
        elif plan is None and route == "generated_code":
            plan = self._generic_code_plan(context)
        elif plan is None:
            plan = self._generic_map_plan(context)
        plan["planner_llm_decision"] = normalized
        return plan

    def replan_after_code_error(
        self,
        *,
        previous_plan: Dict[str, Any],
        failed_event: Dict[str, Any],
        user_request: str,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.replan_after_step_error(
            previous_plan=previous_plan,
            failed_event=failed_event,
            user_request=user_request,
            extracted_params=extracted_params,
            additional_context=additional_context,
        )

    def replan_after_step_error(
        self,
        *,
        previous_plan: Dict[str, Any],
        failed_event: Dict[str, Any],
        user_request: str,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        planner_agent = self.llm_planner
        if planner_agent is None or not hasattr(planner_agent, "replan_harness_task_graph"):
            return None
        context_payload = additional_context or {}
        decision = planner_agent.replan_harness_task_graph(
            user_request=user_request,
            previous_plan=previous_plan,
            failed_event=failed_event,
            dataset=get_active_dataset_public_config(),
            frontend_extracted_params=extracted_params or {},
            workspace_context=context_payload.get("workspace_context", {}),
            conversation_memory=context_payload.get("conversation_memory", {}),
            skill_headers=_llm_skill_header_briefs(),
            skill_workflow_loader=_llm_skill_workflow_briefs,
        )
        if not isinstance(decision, dict):
            return None
        return self._plan_from_llm_harness_decision(
            decision,
            user_request=user_request,
            extracted_params=extracted_params or {},
            additional_context=additional_context or {},
        )

    def dataset_info_plan(self, context: "PlanningContext") -> Dict[str, Any]:
        return self._dataset_info_plan(context)

    def skill_workflow_plan(self, context: "PlanningContext", workflow: WorkflowTemplate) -> Dict[str, Any]:
        return self._skill_workflow_plan(context, workflow)

    def manual_recipe_plan(self, context: "PlanningContext", recipe: WorkflowTemplate) -> Dict[str, Any]:
        return self._skill_workflow_plan(context, recipe)

    def pair_lag_relationship_plan(self, context: "PlanningContext") -> Dict[str, Any]:
        return self._pair_lag_relationship_plan(context)

    def hypoxia_driver_plan(self, context: "PlanningContext") -> Dict[str, Any]:
        return self._hypoxia_driver_plan(context)

    def generic_timeseries_plan(
        self,
        context: "PlanningContext",
        *,
        include_trend: bool,
        include_spectrum: bool,
    ) -> Dict[str, Any]:
        return self._generic_timeseries_plan(context, include_trend=include_trend, include_spectrum=include_spectrum)

    def generic_code_plan(self, context: "PlanningContext") -> Dict[str, Any]:
        return self._generic_code_plan(context)

    def generic_map_plan(self, context: "PlanningContext") -> Dict[str, Any]:
        return self._generic_map_plan(context)

    def condition_mask_map_plan(self, context: "PlanningContext") -> Dict[str, Any]:
        return self._condition_mask_map_plan(context)

    def _skill_workflow_plan(
        self,
        context: "PlanningContext",
        workflow: WorkflowTemplate,
        *,
        planner_agent_contract: str = "skill_workflow.dsl",
        apply_query_masks: bool = True,
        decision: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        missing = _missing_workflow_fields(context, workflow)
        if missing:
            return _clarification_plan(context, workflow, missing)
        vertical = _vertical_for_workflow(context, workflow)
        nodes = _workflow_template_nodes(context=context, workflow=workflow, vertical=vertical)
        if apply_query_masks:
            nodes = _insert_skill_query_mask_nodes(context, workflow, nodes, vertical=vertical)
        produced_artifacts = {node.output.artifact_id for node in nodes if node.output is not None}
        final_artifacts = _workflow_final_artifacts_for_request(
            context=context,
            workflow=workflow,
            nodes=nodes,
            decision=decision,
        )
        nodes = _prune_nodes_to_artifacts(nodes, final_artifacts)
        produced_artifacts = {node.output.artifact_id for node in nodes if node.output is not None}
        final_artifacts = tuple(artifact_id for artifact_id in final_artifacts if artifact_id in produced_artifacts)
        if not final_artifacts:
            final_artifacts = tuple(node.output.artifact_id for node in nodes if node.output is not None)
        return _plan_from_nodes(
            context,
            nodes,
            final_artifacts=final_artifacts,
            skill_id="ocean_harness",
            skills_used=["ocean_harness", workflow.skill_id],
            metadata={
                "planner": "OceanHarnessPlanner",
                "planner_agent_contract": planner_agent_contract,
                "skill_id": workflow.skill_id,
                "workflow_id": workflow.workflow_id,
                "manual_id": workflow.skill_id,
                "recipe_id": workflow.workflow_id,
                "event_type": workflow.event_type,
                "output_policy": dict(workflow.output_policy),
                "validation_rules": list(workflow.validation_rules),
            },
        )

    def _composed_skill_workflow_plan(
        self,
        context: "PlanningContext",
        selected_skill_ids: List[str],
        *,
        decision: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        selected = set(selected_skill_ids)
        if {"ocean_stratification_diagnostics", "ocean_lag_correlation"}.issubset(selected):
            return self._stratification_oxygen_lag_plan(
                context,
                selected_skill_ids,
                decision=decision,
            )
        return None

    def _stratification_oxygen_lag_plan(
        self,
        context: "PlanningContext",
        selected_skill_ids: List[str],
        *,
        decision: Mapping[str, Any],
    ) -> Dict[str, Any]:
        workflow = load_skill_specs()["ocean_stratification_diagnostics"].workflow
        missing = _missing_workflow_fields(context, workflow)
        if missing:
            return _clarification_plan(context, workflow, missing)

        vertical = _vertical_for_workflow(context, workflow)
        nodes = _workflow_template_nodes(context=context, workflow=workflow, vertical=vertical)

        oxygen_vertical = VerticalSpec(
            mode="bottom",
            aggregation="mean",
            source_text=context.user_request,
        )
        nodes.append(
            _read_node(
                "read_bottom_oxygen",
                "bottom_oxygen_field",
                "oxygen",
                context,
                vertical=oxygen_vertical,
                load_vertical=True,
            )
        )
        nodes.append(
            _tool_node(
                "bottom_oxygen_timeseries",
                NodeType.REDUCE,
                "Extract bottom oxygen regional mean time series",
                "extract_timeseries",
                {
                    "data": "$ref:bottom_oxygen_field.data",
                    "lon_range": list(context.lon_range),
                    "lat_range": list(context.lat_range),
                    "spatial_aggregation": "mean",
                    "depth_aggregation": "mean",
                },
                "bottom_oxygen_timeseries",
                ArtifactKind.SERIES,
                ("time",),
                FrontendType.TIMESERIES,
                expected_inputs={"data": "bottom_oxygen_field"},
            )
        )
        nodes.append(
            _tool_node(
                "stability_oxygen_lag_correlation",
                NodeType.DIAGNOSE,
                "Diagnose lead-lag relationship between stratification stability and bottom oxygen",
                "compute_lag_correlation",
                {
                    "timeseries1": "$ref:stability_timeseries",
                    "timeseries2": "$ref:bottom_oxygen_timeseries",
                    "max_lag": _extract_max_lag(context.user_request, default=12),
                    "confidence_level": 0.95,
                },
                "stability_oxygen_lag_correlation",
                ArtifactKind.TABLE,
                (),
                FrontendType.LAG_CORRELATION,
                expected_inputs={
                    "timeseries1": "stability_timeseries",
                    "timeseries2": "bottom_oxygen_timeseries",
                },
            )
        )

        skills_used = ["ocean_harness"]
        for skill_id in selected_skill_ids:
            if skill_id not in skills_used:
                skills_used.append(skill_id)
        return _plan_from_nodes(
            context,
            nodes,
            final_artifacts=[
                "stability_timeseries",
                "bottom_oxygen_timeseries",
                "stability_oxygen_lag_correlation",
            ],
            skill_id="ocean_harness",
            skills_used=skills_used,
            metadata={
                "planner": "OceanHarnessPlanner",
                "planner_agent_contract": "backend_template_composition.dsl",
                "composition": "stratification_oxygen_lag",
                "selected_skill_ids": list(selected_skill_ids),
                "primary_skill_id": "ocean_stratification_diagnostics",
                "secondary_skill_id": "ocean_lag_correlation",
                "primary_timeseries": "stability_timeseries",
                "secondary_timeseries": "bottom_oxygen_timeseries",
                "reason": decision.get("reason"),
            },
        )

    def _dataset_info_plan(self, context: "PlanningContext") -> Dict[str, Any]:
        output = _artifact("dataset_info", ArtifactKind.TABLE, (), FrontendType.GENERIC)
        node = TaskNode(
            node_id="get_dataset_info",
            node_type=NodeType.READ,
            intent="Inspect active dataset metadata",
            operation="get_dataset_info",
            execution=ExecutionSpec(
                strategy=ExecutionStrategy.TOOL,
                tool_name="get_dataset_info",
                params={"dataset": context.dataset, "include_runtime_probe": False},
            ),
            output=output,
        )
        return _plan_from_nodes(context, [node], final_artifacts=["dataset_info"])

    def _generic_map_plan(self, context: "PlanningContext") -> Dict[str, Any]:
        variable = context.primary_variable()
        vertical = context.vertical_spec()
        nodes: List[TaskNode] = [
            _read_node("read_field", f"{variable}_raw", variable, context, vertical=vertical, load_vertical=False),
        ]
        current = f"{variable}_raw"
        if vertical.mode != "unspecified":
            nodes.append(_vertical_node("select_field", current, f"{variable}_selected", vertical))
            current = f"{variable}_selected"
        current = _append_query_mask_nodes(context, nodes, source_id=current, variable=variable, vertical=vertical)
        nodes.append(
            _tool_node(
                "compute_map",
                NodeType.REDUCE,
                "Reduce field to a map for frontend display",
                "compute_spatial_field",
                {
                    "data": f"$ref:{current}.data",
                    "time_aggregation": "mean",
                    "depth_aggregation": "mean",
                },
                "map_result",
                ArtifactKind.MAP,
                ("lat", "lon"),
                FrontendType.SPATIAL_FIELD,
                expected_inputs={"data": current},
            )
        )
        return _plan_from_nodes(context, nodes, final_artifacts=["map_result"])

    def _condition_mask_map_plan(self, context: "PlanningContext") -> Dict[str, Any]:
        variable = context.primary_variable()
        vertical = context.vertical_spec()
        nodes: List[TaskNode] = [
            _read_node("read_field", f"{variable}_raw", variable, context, vertical=vertical, load_vertical=False),
        ]
        current = f"{variable}_raw"
        if vertical.mode != "unspecified":
            nodes.append(_vertical_node("select_field", current, f"{variable}_selected", vertical))
            current = f"{variable}_selected"
        current = _append_query_mask_nodes(context, nodes, source_id=current, variable=variable, vertical=vertical)
        nodes.append(
            _tool_node(
                "compute_map",
                NodeType.REDUCE,
                "Reduce condition-masked field to a map for frontend display",
                "compute_spatial_field",
                {
                    "data": f"$ref:{current}.data",
                    "time_aggregation": "mean",
                    "depth_aggregation": "mean",
                },
                "map_result",
                ArtifactKind.MAP,
                ("lat", "lon"),
                FrontendType.SPATIAL_FIELD,
                expected_inputs={"data": current},
            )
        )
        return _plan_from_nodes(
            context,
            nodes,
            final_artifacts=["map_result"],
            skills_used=["ocean_harness", "ocean_masking_workflow"],
            metadata={
                "planner": "OceanHarnessPlanner",
                "planner_agent_contract": "condition_mask_spatial_map",
                "skill_id": "ocean_masking_workflow",
                "workflow_id": "condition_mask_spatial_map",
                "manual_id": "ocean_masking_workflow",
                "recipe_id": "condition_mask_spatial_map",
                "mask_workflow": "condition_mask_spatial_map",
            },
        )

    def _generic_timeseries_plan(
        self,
        context: "PlanningContext",
        *,
        include_trend: bool,
        include_spectrum: bool,
    ) -> Dict[str, Any]:
        variable = context.primary_variable()
        vertical = context.vertical_spec()
        nodes: List[TaskNode] = [
            _read_node("read_field", f"{variable}_raw", variable, context, vertical=vertical, load_vertical=False),
        ]
        current = f"{variable}_raw"
        if vertical.mode != "unspecified":
            nodes.append(_vertical_node("select_field", current, f"{variable}_selected", vertical))
            current = f"{variable}_selected"
        current = _append_query_mask_nodes(context, nodes, source_id=current, variable=variable, vertical=vertical)
        nodes.append(
            _tool_node(
                "extract_timeseries",
                NodeType.REDUCE,
                "Extract region-mean time series",
                "extract_timeseries",
                {
                    "data": f"$ref:{current}.data",
                    "lon_range": list(context.lon_range),
                    "lat_range": list(context.lat_range),
                    "spatial_aggregation": "mean",
                    "depth_aggregation": "mean",
                },
                "timeseries",
                ArtifactKind.SERIES,
                ("time",),
                FrontendType.TIMESERIES,
                expected_inputs={"data": current},
            )
        )
        final_artifacts = ["timeseries"]
        if include_trend:
            nodes.append(_trend_node("compute_trend", "timeseries", "trend"))
            final_artifacts.append("trend")
        if include_spectrum:
            nodes.append(_spectrum_node("compute_spectrum", "timeseries", "spectrum"))
            final_artifacts.append("spectrum")
        return _plan_from_nodes(context, nodes, final_artifacts=final_artifacts)

    def _generic_code_plan(self, context: "PlanningContext") -> Dict[str, Any]:
        variable = context.primary_variable()
        vertical = context.vertical_spec()
        nodes: List[TaskNode] = [
            _read_node("read_code_field", f"{variable}_raw", variable, context, vertical=vertical, load_vertical=False),
        ]
        current = f"{variable}_raw"
        if vertical.mode != "unspecified":
            nodes.append(_vertical_node("select_code_field", current, f"{variable}_selected", vertical))
            current = f"{variable}_selected"
        nodes.append(_generated_code_node(context, input_id=current, output_id="generated_analysis", code_agent=self.code_agent))
        return _plan_from_nodes(
            context,
            nodes,
            final_artifacts=["generated_analysis"],
            skills_used=["ocean_harness", "generated_code_agent"],
            metadata={
                "planner": "OceanHarnessPlanner",
                "manual_id": None,
                "recipe_id": "generated_code_fallback",
                "code_agent": {
                    "input_contract": "run(inputs, params) -> dict",
                    "allowed_libraries": ["numpy", "pandas", "xarray", "scipy"],
                    "blocked_capabilities": ["file", "network", "shell"],
                },
            },
        )

    def _pair_lag_relationship_plan(self, context: "PlanningContext") -> Dict[str, Any]:
        driver_variable, response_variable = _infer_lag_variables(context.user_request)
        if not driver_variable or not response_variable:
            return self._generic_timeseries_plan(context, include_trend=False, include_spectrum=False)
        response_vertical = _variable_vertical_spec(context.user_request, response_variable)
        driver_vertical = _variable_vertical_spec(context.user_request, driver_variable)
        max_lag = _extract_max_lag(context.user_request, default=12)

        nodes: List[TaskNode] = [
            _read_node(
                f"read_{driver_variable}",
                f"{driver_variable}_field",
                driver_variable,
                context,
                vertical=driver_vertical,
                load_vertical=False,
            ),
            _vertical_node(
                f"select_{driver_variable}_layer",
                f"{driver_variable}_field",
                f"{driver_variable}_selected",
                driver_vertical,
            ),
            _tool_node(
                f"{driver_variable}_timeseries",
                NodeType.REDUCE,
                f"Extract regional mean time series for {driver_variable}",
                "extract_timeseries",
                {
                    "data": f"$ref:{driver_variable}_selected.data",
                    "lon_range": list(context.lon_range),
                    "lat_range": list(context.lat_range),
                    "spatial_aggregation": "mean",
                    "depth_aggregation": "mean",
                },
                f"{driver_variable}_timeseries",
                ArtifactKind.SERIES,
                ("time",),
                FrontendType.TIMESERIES,
                expected_inputs={"data": f"{driver_variable}_selected"},
            ),
            _read_node(
                f"read_{response_variable}",
                f"{response_variable}_field",
                response_variable,
                context,
                vertical=response_vertical,
                load_vertical=False,
            ),
            _vertical_node(
                f"select_{response_variable}_layer",
                f"{response_variable}_field",
                f"{response_variable}_selected",
                response_vertical,
            ),
            _tool_node(
                f"{response_variable}_timeseries",
                NodeType.REDUCE,
                f"Extract regional mean time series for {response_variable}",
                "extract_timeseries",
                {
                    "data": f"$ref:{response_variable}_selected.data",
                    "lon_range": list(context.lon_range),
                    "lat_range": list(context.lat_range),
                    "spatial_aggregation": "mean",
                    "depth_aggregation": "mean",
                },
                f"{response_variable}_timeseries",
                ArtifactKind.SERIES,
                ("time",),
                FrontendType.TIMESERIES,
                expected_inputs={"data": f"{response_variable}_selected"},
            ),
            _tool_node(
                "lag_relationship",
                NodeType.DIAGNOSE,
                f"Diagnose lag relationship between {driver_variable} and {response_variable}",
                "compute_lag_correlation",
                {
                    "timeseries1": f"$ref:{driver_variable}_timeseries",
                    "timeseries2": f"$ref:{response_variable}_timeseries",
                    "max_lag": max_lag,
                    "confidence_level": 0.95,
                },
                f"{driver_variable}_{response_variable}_lag_correlation",
                ArtifactKind.TABLE,
                (),
                FrontendType.LAG_CORRELATION,
                expected_inputs={
                    "timeseries1": f"{driver_variable}_timeseries",
                    "timeseries2": f"{response_variable}_timeseries",
                },
            ),
        ]
        return _plan_from_nodes(
            context,
            nodes,
            final_artifacts=[
                f"{driver_variable}_timeseries",
                f"{response_variable}_timeseries",
                f"{driver_variable}_{response_variable}_lag_correlation",
            ],
            skills_used=["ocean_harness", "ocean_variable_lag_relationship"],
            metadata={
                "planner": "OceanHarnessPlanner",
                "skill_id": "ocean_variable_lag_relationship",
                "workflow_id": "pair_lag_relationship",
                "manual_id": "ocean_variable_lag_relationship_manual",
                "recipe_id": "pair_lag_relationship",
                "relationship": {
                    "driver_variable": driver_variable,
                    "response_variable": response_variable,
                    "max_lag": max_lag,
                },
            },
        )

    def _hypoxia_driver_plan(self, context: "PlanningContext") -> Dict[str, Any]:
        vertical = context.vertical_spec(default_bottom=True)
        threshold = _extract_threshold(context.user_request, default=60.0)
        nodes: List[TaskNode] = [
            _read_node("read_oxygen", "oxygen_raw", "oxygen", context, vertical=vertical, load_vertical=False),
            _vertical_node("select_oxygen_layer", "oxygen_raw", "oxygen_selected", vertical),
            _tool_node(
                "hypoxia_mask",
                NodeType.DERIVE,
                "Build hypoxia diagnostic mask from selected oxygen field",
                "build_threshold_mask",
                {
                    "data": "$ref:oxygen_selected.data",
                    "threshold": threshold,
                    "comparison": "lt",
                    "mask_name": "hypoxia_mask",
                },
                "hypoxia_mask",
                ArtifactKind.MASK,
                ("time", "lat", "lon") if not vertical.retain_depth else ("time", "depth", "lat", "lon"),
                FrontendType.DATA_CONTAINER,
                expected_inputs={"data": "oxygen_selected"},
            ),
            _tool_node(
                "hypoxia_timeseries",
                NodeType.REDUCE,
                "Reduce hypoxia mask to regional area-fraction time series",
                "compute_masked_area_fraction_timeseries",
                {"event_mask": "$ref:hypoxia_mask.data"},
                "hypoxia_timeseries",
                ArtifactKind.SERIES,
                ("time",),
                FrontendType.TIMESERIES,
                expected_inputs={"event_mask": "hypoxia_mask"},
            ),
            _trend_node("hypoxia_trend", "hypoxia_timeseries", "hypoxia_trend"),
            _spectrum_node("hypoxia_spectrum", "hypoxia_timeseries", "hypoxia_spectrum"),
        ]

        for variable in ("temp", "salt", "u", "v"):
            nodes.append(_read_node(f"read_{variable}", f"{variable}_raw", variable, context, vertical=vertical, load_vertical=False))
            nodes.append(_vertical_node(f"select_{variable}_layer", f"{variable}_raw", f"{variable}_selected", vertical))

        nodes.append(
            _tool_node(
                "derive_speed",
                NodeType.DERIVE,
                "Compute current speed magnitude from u/v components",
                "compute_speed_from_uv",
                {"u": "$ref:u_selected.data", "v": "$ref:v_selected.data"},
                "speed_selected",
                ArtifactKind.FIELD,
                ("time", "lat", "lon") if not vertical.retain_depth else ("time", "depth", "lat", "lon"),
                FrontendType.DATA_CONTAINER,
                expected_inputs={"u": "u_selected", "v": "v_selected"},
            )
        )

        for driver in ("temp", "salt", "speed"):
            source = f"{driver}_selected"
            nodes.append(
                _tool_node(
                    f"{driver}_timeseries",
                    NodeType.REDUCE,
                    f"Reduce {driver} field to regional mean time series",
                    "compute_masked_mean_timeseries",
                    {"data": f"$ref:{source}.data"},
                    f"{driver}_timeseries",
                    ArtifactKind.SERIES,
                    ("time",),
                    FrontendType.TIMESERIES,
                    expected_inputs={"data": source},
                )
            )
            nodes.append(
                _tool_node(
                    f"{driver}_lag_correlation",
                    NodeType.DIAGNOSE,
                    f"Diagnose lag correlation between hypoxia and {driver}",
                    "compute_lag_correlation",
                    {
                        "timeseries1": "$ref:hypoxia_timeseries",
                        "timeseries2": f"$ref:{driver}_timeseries",
                        "max_lag": 12,
                        "confidence_level": 0.95,
                    },
                    f"{driver}_lag_correlation",
                    ArtifactKind.TABLE,
                    (),
                    FrontendType.LAG_CORRELATION,
                    expected_inputs={
                        "timeseries1": "hypoxia_timeseries",
                        "timeseries2": f"{driver}_timeseries",
                    },
                )
            )

        return _plan_from_nodes(
            context,
            nodes,
            final_artifacts=[
                "hypoxia_timeseries",
                "hypoxia_trend",
                "hypoxia_spectrum",
                "temp_lag_correlation",
                "salt_lag_correlation",
                "speed_lag_correlation",
            ],
        )


class PlanningContext:
    def __init__(
        self,
        *,
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
        dataset: str,
        lon_range: Tuple[float, float],
        lat_range: Tuple[float, float],
        time_range: Optional[Tuple[str, str]],
        vertical: Optional[VerticalSpec] = None,
    ) -> None:
        self.user_request = user_request
        self.extracted_params = extracted_params
        self.additional_context = additional_context
        self.dataset = dataset
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.time_range = time_range
        self.vertical = vertical or VerticalSpec(mode="unspecified")

    def has_explicit_region(self) -> bool:
        scope = DataScopeResolver().resolve(
            user_request=self.user_request,
            extracted_params=self.extracted_params,
            additional_context=self.additional_context,
        )
        return scope.explicit_region

    def has_explicit_time_range(self) -> bool:
        scope = DataScopeResolver().resolve(
            user_request=self.user_request,
            extracted_params=self.extracted_params,
            additional_context=self.additional_context,
        )
        return scope.explicit_time_range

    @classmethod
    def from_inputs(
        cls,
        *,
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> "PlanningContext":
        scope = DataScopeResolver().resolve(
            user_request=user_request,
            extracted_params=extracted_params,
            additional_context=additional_context,
        )
        return cls(
            user_request=user_request,
            extracted_params=extracted_params,
            additional_context=additional_context,
            dataset=scope.dataset,
            lon_range=scope.lon_range,
            lat_range=scope.lat_range,
            time_range=scope.time_range,
            vertical=scope.vertical,
        )

    def primary_variable(self) -> str:
        return infer_primary_variable(self.user_request, self.extracted_params)

    def vertical_spec(self, *, default_bottom: bool = False) -> VerticalSpec:
        if self.vertical.mode != "unspecified":
            return self.vertical
        return vertical_spec_from_text(self.user_request, default_bottom=default_bottom)


def _read_node(
    node_id: str,
    output_id: str,
    variable: str,
    context: PlanningContext,
    *,
    vertical: VerticalSpec,
    load_vertical: bool,
) -> TaskNode:
    params: Dict[str, Any] = {
        "dataset": context.dataset,
        "variable": variable,
        "lon_range": list(context.lon_range),
        "lat_range": list(context.lat_range),
    }
    if context.time_range is not None:
        params["time_range"] = list(context.time_range)
    if load_vertical and vertical.mode in {"bottom", "surface", "fixed_depth", "depth_range"}:
        params["vertical_mode"] = vertical.mode
        if vertical.depth_value is not None:
            params["depth_value"] = vertical.depth_value
        if vertical.depth_range is not None:
            params["depth_range"] = list(vertical.depth_range)
    read_spec = ReadSpec(
        variables=(variable,),
        region={"lon_range": context.lon_range, "lat_range": context.lat_range},
        time_range=context.time_range,
        vertical=vertical,
        expected_shape=ShapeClass.FIELD_4D,
        dataset=context.dataset,
    )
    return TaskNode(
        node_id=node_id,
        node_type=NodeType.READ,
        intent=f"Read {variable} data with user-requested space/time context",
        operation="load_dataset",
        read_spec=read_spec,
        execution=ExecutionSpec(strategy=ExecutionStrategy.TOOL, tool_name="load_dataset", params=params),
        output=_artifact(output_id, ArtifactKind.FIELD, ("time", "depth", "lat", "lon"), FrontendType.DATA_CONTAINER, variable=variable),
    )


def _vertical_node(node_id: str, input_id: str, output_id: str, vertical: VerticalSpec) -> TaskNode:
    params: Dict[str, Any] = {
        "data": f"$ref:{input_id}.data",
        "mode": vertical.mode,
        "aggregation": vertical.aggregation or "mean",
        "retain_depth": vertical.retain_depth,
    }
    if vertical.depth_value is not None:
        params["depth_value"] = vertical.depth_value
    if vertical.depth_range is not None:
        params["depth_range"] = list(vertical.depth_range)
    if vertical.relative_to is not None:
        params["relative_to"] = vertical.relative_to
    if vertical.band_thickness_m is not None:
        params["band_thickness_m"] = vertical.band_thickness_m
    dims = ("time", "depth", "lat", "lon") if vertical.retain_depth else ("time", "lat", "lon")
    if vertical.mode in {"unspecified", "as_is"}:
        dims = ("time", "depth", "lat", "lon")
    return _tool_node(
        node_id,
        NodeType.SELECT,
        "Apply semantic vertical selection",
        "select_vertical",
        params,
        output_id,
        ArtifactKind.FIELD,
        dims,
        FrontendType.DATA_CONTAINER,
        expected_inputs={"data": input_id},
    )


def _trend_node(node_id: str, input_id: str, output_id: str) -> TaskNode:
    return _tool_node(
        node_id,
        NodeType.DIAGNOSE,
        "Compute long-term linear trend",
        "compute_trend",
        {"timeseries": f"$ref:{input_id}", "method": "linear", "confidence_level": 0.95},
        output_id,
        ArtifactKind.TABLE,
        (),
        FrontendType.TREND,
        expected_inputs={"timeseries": input_id},
    )


def _spectrum_node(node_id: str, input_id: str, output_id: str) -> TaskNode:
    return _tool_node(
        node_id,
        NodeType.DIAGNOSE,
        "Compute power spectrum from time series",
        "compute_spectrum",
        {"timeseries": f"$ref:{input_id}", "method": "welch", "detrend": "linear"},
        output_id,
        ArtifactKind.SPECTRUM,
        ("frequency",),
        FrontendType.SPECTRUM,
        expected_inputs={"timeseries": input_id},
    )


def _generated_code_node(
    context: PlanningContext,
    *,
    input_id: str,
    output_id: str,
    code_params: Optional[Mapping[str, Any]] = None,
    code_agent: Optional[CodeAgent] = None,
) -> TaskNode:
    vertical = context.vertical_spec()
    generated_code_params = {
        "user_request": context.user_request,
        "variable": context.primary_variable(),
        "lon_range": list(context.lon_range),
        "lat_range": list(context.lat_range),
        "time_range": list(context.time_range) if context.time_range is not None else None,
        "vertical_mode": vertical.mode,
        "depth_value": vertical.depth_value,
        "depth_range": list(vertical.depth_range) if vertical.depth_range is not None else None,
    }
    planner_analysis_design = None
    planner_code_steps = None
    for key, value in dict(code_params or {}).items():
        if key == "input_refs":
            continue
        if key in {"analysis_design", "planner_analysis_design"}:
            planner_analysis_design = value
            continue
        if key in {"code_steps", "planner_code_steps"}:
            planner_code_steps = value
            continue
        generated_code_params[key] = value
    execution_params: Dict[str, Any] = {
        "input_refs": {"field": f"$ref:{input_id}.data"},
        "code_params": generated_code_params,
        "io_contract": {
            "entrypoint": "run(inputs, params) -> dict",
            "design_status": "deferred_to_code_agent",
            "allowed_libraries": list(CodeAgent.allowed_libraries),
            "blocked_capabilities": list(CodeAgent.blocked_capabilities),
            "repair_loop": {
                "max_attempts": 3,
                "feedback": "traceback + input artifact schema + expected output schema",
                "status": "llm_code_writer",
            },
        },
    }
    if isinstance(planner_analysis_design, Mapping):
        execution_params["planner_analysis_design"] = dict(planner_analysis_design)
    if isinstance(planner_code_steps, list):
        execution_params["planner_code_steps"] = [dict(step) for step in planner_code_steps if isinstance(step, Mapping)]
    return TaskNode(
        node_id="generated_code_analysis",
        node_type=NodeType.DIAGNOSE,
        intent="Run generated Python analysis when no existing skill workflow or bound tool covers the request",
        operation="generated_python_analysis",
        inputs={"field": input_id},
        execution=ExecutionSpec(
            strategy=ExecutionStrategy.CODE,
            code="",
            params=execution_params,
        ),
        output=_artifact(output_id, ArtifactKind.GENERIC, (), FrontendType.GENERIC),
        validation_rules=("code_must_return_dict", "no_file_network_shell"),
    )


def _tool_node(
    node_id: str,
    node_type: NodeType,
    intent: str,
    tool_name: str,
    params: Mapping[str, Any],
    output_id: str,
    kind: ArtifactKind,
    dims: Tuple[str, ...],
    frontend_type: FrontendType,
    *,
    expected_inputs: Optional[Mapping[str, str]] = None,
) -> TaskNode:
    return TaskNode(
        node_id=node_id,
        node_type=node_type,
        intent=intent,
        operation=tool_name,
        inputs=dict(expected_inputs or {}),
        execution=ExecutionSpec(strategy=ExecutionStrategy.TOOL, tool_name=tool_name, params=dict(params)),
        output=_artifact(output_id, kind, dims, frontend_type),
    )


def _tool_call_template_node(
    index: int,
    step: Any,
    *,
    context: PlanningContext,
    workflow: WorkflowTemplate,
    vertical: VerticalSpec,
    resolved_params: Optional[Mapping[str, Any]] = None,
) -> TaskNode:
    params = (
        dict(resolved_params)
        if isinstance(resolved_params, Mapping)
        else _resolve_workflow_params(step.params_template, context=context, workflow=workflow, vertical=vertical)
    )
    tool_name = step.tool
    if tool_name == "load_dataset":
        params.pop("depth_aggregation", None)
        params.setdefault("dataset", context.dataset)
    output_id = step.save_as
    variable = str(params.get("variable") or "") if tool_name == "load_dataset" else None
    node_type = _node_type_for_tool(tool_name)
    kind, dims, frontend_type = _tool_call_output_spec(step, tool_name)
    node_id = _tool_call_node_id(index, tool_name, output_id)
    read_spec = None
    if tool_name == "load_dataset" and variable:
        read_vertical = _vertical_from_read_params(params, fallback=vertical)
        read_spec = ReadSpec(
            variables=(variable,),
            region={"lon_range": context.lon_range, "lat_range": context.lat_range},
            time_range=context.time_range,
            vertical=read_vertical,
            expected_shape=ShapeClass.FIELD_4D,
            dataset=context.dataset,
        )
    return TaskNode(
        node_id=node_id,
        node_type=node_type,
        intent=f"{step.title} using {workflow.skill_id}",
        operation=tool_name,
        inputs=_collect_ref_inputs(params),
        execution=ExecutionSpec(strategy=ExecutionStrategy.TOOL, tool_name=tool_name, params=params),
        read_spec=read_spec,
        output=_artifact(output_id, kind, dims, frontend_type, variable=variable),
        validation_rules=workflow.validation_rules,
    )


def _workflow_template_nodes(
    *,
    context: PlanningContext,
    workflow: WorkflowTemplate,
    vertical: VerticalSpec,
) -> List[TaskNode]:
    nodes: List[TaskNode] = []
    produced: set[str] = set()
    alias_by_artifact: Dict[str, str] = {}
    for index, step in enumerate(workflow.steps, start=1):
        params = _resolve_workflow_params(step.params_template, context=context, workflow=workflow, vertical=vertical)
        params = _rewrite_template_alias_refs(params, alias_by_artifact=alias_by_artifact)
        if not _workflow_template_step_can_bind(step, params, produced):
            alias = _skipped_optional_mask_step_alias(step, params, produced)
            if alias:
                alias_by_artifact[step.save_as] = alias
            continue
        node = _tool_call_template_node(
            index,
            step,
            context=context,
            workflow=workflow,
            vertical=vertical,
            resolved_params=params,
        )
        nodes.append(node)
        produced.add(step.save_as)
        alias_by_artifact[step.save_as] = step.save_as
    return nodes


def _workflow_final_artifacts_for_request(
    *,
    context: PlanningContext,
    workflow: WorkflowTemplate,
    nodes: List[TaskNode],
    decision: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, ...]:
    produced = tuple(node.output.artifact_id for node in nodes if node.output is not None)
    produced_set = set(produced)
    explicit = _explicit_final_artifacts(decision, produced_set=produced_set)
    if explicit:
        return explicit

    workflow_finals = tuple(artifact_id for artifact_id in workflow.final_artifacts if artifact_id in produced_set)
    if not workflow_finals:
        return produced
    if len(workflow_finals) <= 1 or _workflow_final_artifacts_have_dependency(workflow_finals, nodes):
        return workflow_finals

    scored = [
        (_workflow_final_artifact_score(context.user_request, artifact_id, nodes), artifact_id)
        for artifact_id in workflow_finals
    ]
    best_score = max((score for score, _ in scored), default=0)
    if best_score <= 0:
        return workflow_finals
    return tuple(artifact_id for score, artifact_id in scored if score == best_score)


def _explicit_final_artifacts(
    decision: Optional[Mapping[str, Any]],
    *,
    produced_set: set[str],
) -> Tuple[str, ...]:
    if not isinstance(decision, Mapping):
        return ()
    explicit = decision.get("final_artifacts")
    if not isinstance(explicit, list):
        workflow_todo = decision.get("workflow_todo")
        if isinstance(workflow_todo, Mapping) and isinstance(workflow_todo.get("final_artifacts"), list):
            explicit = workflow_todo.get("final_artifacts")
    if not isinstance(explicit, list):
        return ()
    return tuple(str(item).strip() for item in explicit if str(item).strip() in produced_set)


def _workflow_final_artifacts_have_dependency(final_artifacts: Tuple[str, ...], nodes: List[TaskNode]) -> bool:
    final_set = set(final_artifacts)
    dependency_roots = _workflow_dependency_roots_by_artifact(nodes)
    for artifact_id in final_artifacts:
        if dependency_roots.get(artifact_id, set()).intersection(final_set - {artifact_id}):
            return True
    return False


def _workflow_dependency_roots_by_artifact(nodes: List[TaskNode]) -> Dict[str, set[str]]:
    direct_inputs: Dict[str, set[str]] = {}
    for node in nodes:
        if node.output is None:
            continue
        direct_inputs[node.output.artifact_id] = {str(value) for value in node.inputs.values()}

    memo: Dict[str, set[str]] = {}

    def collect(artifact_id: str) -> set[str]:
        if artifact_id in memo:
            return memo[artifact_id]
        roots: set[str] = set()
        for input_id in direct_inputs.get(artifact_id, set()):
            roots.add(input_id)
            roots.update(collect(input_id))
        memo[artifact_id] = roots
        return roots

    return {artifact_id: collect(artifact_id) for artifact_id in direct_inputs}


def _workflow_final_artifact_score(query: str, artifact_id: str, nodes: List[TaskNode]) -> int:
    lowered = query.lower()
    node = next((candidate for candidate in nodes if candidate.output is not None and candidate.output.artifact_id == artifact_id), None)
    if node is None:
        return 0
    operation = str(node.operation or "").lower()
    frontend_type = str(node.output.frontend_type.value if node.output is not None else "").lower()
    haystack = f"{artifact_id.lower()} {operation} {frontend_type}"
    score = 0
    query_tokens = set(re.findall(r"[a-z0-9]+", lowered))
    artifact_tokens = set(re.findall(r"[a-z0-9]+", haystack))
    score += len(query_tokens.intersection(artifact_tokens))
    if re.search(r"\b(time[- ]depth|hovmoller|diagram)\b", lowered) and (
        "hovmoller" in haystack or frontend_type == FrontendType.HOVMOLLER.value
    ):
        score += 8
    if "streamfunction" in lowered and "streamfunction" in haystack:
        score += 8
    if re.search(r"\b(time[- ]series|timeseries|series)\b", lowered) and (
        "timeseries" in haystack or frontend_type == FrontendType.TIMESERIES.value
    ):
        score += 5
    if re.search(r"\bmap|spatial\b", lowered) and (
        "map" in haystack or frontend_type in {FrontendType.SPATIAL_FIELD.value, FrontendType.FIELD_TREND.value}
    ):
        score += 4
    if "residual" in lowered and "residual" in haystack:
        score += 4
    if re.search(r"\b(compare|comparison|dominant|controlled more|magnitude|rank)\b", lowered) and (
        "comparison" in haystack or "compare" in haystack
    ):
        score += 4
    return score


def _prune_nodes_to_artifacts(nodes: List[TaskNode], final_artifacts: Tuple[str, ...]) -> List[TaskNode]:
    if not final_artifacts:
        return nodes
    by_output = {
        node.output.artifact_id: node
        for node in nodes
        if node.output is not None
    }
    required: set[str] = set()

    def include(artifact_id: str) -> None:
        if artifact_id in required:
            return
        node = by_output.get(artifact_id)
        if node is None:
            return
        required.add(artifact_id)
        for input_id in node.inputs.values():
            include(str(input_id))

    for artifact_id in final_artifacts:
        include(artifact_id)
    if not required:
        return nodes
    return [node for node in nodes if node.output is None or node.output.artifact_id in required]


def _workflow_template_step_can_bind(step: Any, params: Mapping[str, Any], produced: set[str]) -> bool:
    for key in _workflow_required_params_for_tool(str(step.tool)):
        if _missing_workflow_param(params.get(key)):
            return False
    return all(ref in produced for ref in _collect_ref_inputs(params))


def _workflow_required_params_for_tool(tool_name: str) -> Tuple[str, ...]:
    if tool_name == "load_dataset":
        return ("variable",)
    if tool_name == "build_polygon_mask":
        return ("data", "polygon_points")
    if tool_name == "build_isobath_mask":
        return ("data", "isobath_depth")
    if tool_name == "combine_masks":
        return ("masks",)
    if tool_name == "apply_mask":
        return ("data", "mask")
    if tool_name == "compute_hovmoller":
        return ("data",)
    return ()


def _missing_workflow_param(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _skipped_optional_mask_step_alias(step: Any, params: Mapping[str, Any], produced: set[str]) -> Optional[str]:
    tool_name = str(step.tool)
    if tool_name == "combine_masks":
        mask_roots = [_ref_root(item) for item in params.get("masks", []) if isinstance(item, str)]
        produced_masks = [root for root in mask_roots if root in produced]
        if len(produced_masks) == 1:
            return produced_masks[0]
    if tool_name == "apply_mask":
        data_root = _ref_root(params.get("data"))
        if data_root in produced:
            return data_root
    return None


def _ref_root(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.startswith("$ref:"):
        return None
    root = value[5:].split(".", 1)[0].strip()
    return root or None


def _rewrite_template_alias_refs(value: Any, *, alias_by_artifact: Mapping[str, str]) -> Any:
    if isinstance(value, str) and value.startswith("$ref:"):
        path = value[5:]
        root, suffix = _split_artifact_ref(path)
        alias = alias_by_artifact.get(root)
        if alias:
            return f"$ref:{alias}{suffix}"
        return value
    if isinstance(value, Mapping):
        return {
            key: _rewrite_template_alias_refs(nested, alias_by_artifact=alias_by_artifact)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_template_alias_refs(nested, alias_by_artifact=alias_by_artifact) for nested in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_template_alias_refs(nested, alias_by_artifact=alias_by_artifact) for nested in value)
    return value


def _append_query_mask_nodes(
    context: PlanningContext,
    nodes: List[TaskNode],
    *,
    source_id: str,
    variable: str,
    vertical: VerticalSpec,
) -> str:
    mask_ids: List[str] = []
    polygon_points = _lookup_context_param("mask_polygon", context) or _lookup_context_param("polygon_points", context)
    if polygon_points is not None:
        nodes.append(
            _tool_node(
                "build_polygon_analysis_mask",
                NodeType.DERIVE,
                "Build polygon analysis mask",
                "build_polygon_mask",
                {
                    "data": f"$ref:{source_id}.data",
                    "polygon_points": polygon_points,
                    "invert": bool(_lookup_context_param("mask_polygon_invert", context) or False),
                },
                "polygon_analysis_mask",
                ArtifactKind.MASK,
                ("lat", "lon"),
                FrontendType.DATA_CONTAINER,
                expected_inputs={"data": source_id},
            )
        )
        mask_ids.append("polygon_analysis_mask")

    condition_expression, condition_variables = _condition_mask_expression(context, default_variable=variable)
    if condition_expression:
        fields: Dict[str, Any] = {}
        for condition_variable in condition_variables:
            if condition_variable == variable:
                fields[condition_variable] = f"$ref:{source_id}.data"
                continue
            condition_source = _append_condition_field_nodes(
                context,
                nodes,
                variable=condition_variable,
                vertical=vertical,
            )
            fields[condition_variable] = f"$ref:{condition_source}.data"
        nodes.append(
            _tool_node(
                "build_condition_analysis_mask",
                NodeType.DERIVE,
                "Build condition analysis mask",
                "build_condition_mask",
                {
                    "fields": fields,
                    "expression": condition_expression,
                    "mask_name": "condition_analysis_mask",
                },
                "condition_analysis_mask",
                ArtifactKind.MASK,
                ("time", "lat", "lon"),
                FrontendType.DATA_CONTAINER,
                expected_inputs={artifact_id.split(".", 1)[0]: artifact_id.split(".", 1)[0] for artifact_id in []},
            )
        )
        mask_ids.append("condition_analysis_mask")

    if not mask_ids:
        return source_id

    final_mask_id = mask_ids[0]
    if len(mask_ids) > 1:
        nodes.append(
            _tool_node(
                "combine_analysis_masks",
                NodeType.DERIVE,
                "Combine analysis masks",
                "combine_masks",
                {
                    "masks": [f"$ref:{mask_id}.data" for mask_id in mask_ids],
                    "operation": "and",
                    "invert": False,
                },
                "analysis_mask",
                ArtifactKind.MASK,
                ("time", "lat", "lon") if any(mask_id == "condition_analysis_mask" for mask_id in mask_ids) else ("lat", "lon"),
                FrontendType.DATA_CONTAINER,
                expected_inputs={mask_id: mask_id for mask_id in mask_ids},
            )
        )
        final_mask_id = "analysis_mask"

    masked_id = f"{source_id}_masked"
    nodes.append(
        _tool_node(
            "apply_analysis_mask",
            NodeType.DERIVE,
            "Apply analysis mask before downstream computation",
            "apply_mask",
            {
                "data": f"$ref:{source_id}.data",
                "mask": f"$ref:{final_mask_id}.data",
            },
            masked_id,
            ArtifactKind.FIELD,
            ("time", "depth", "lat", "lon"),
            FrontendType.DATA_CONTAINER,
            expected_inputs={"data": source_id, "mask": final_mask_id},
        )
    )
    return masked_id


def _append_condition_field_nodes(
    context: PlanningContext,
    nodes: List[TaskNode],
    *,
    variable: str,
    vertical: VerticalSpec,
) -> str:
    raw_id = f"{variable}_condition_raw"
    selected_id = f"{variable}_condition_field"
    existing_ids = {node.output.artifact_id for node in nodes if node.output is not None}
    if selected_id in existing_ids:
        return selected_id
    nodes.append(_read_node(f"read_{variable}_condition_field", raw_id, variable, context, vertical=vertical, load_vertical=False))
    if vertical.mode != "unspecified":
        nodes.append(_vertical_node(f"select_{variable}_condition_field", raw_id, selected_id, vertical))
        return selected_id
    return raw_id


def _insert_skill_query_mask_nodes(
    context: PlanningContext,
    workflow: WorkflowTemplate,
    nodes: List[TaskNode],
    *,
    vertical: VerticalSpec,
) -> List[TaskNode]:
    if not nodes or not _has_query_mask_request(context):
        return nodes
    if workflow.event_type is None and workflow.skill_id not in {"ocean_bloom_detection", "ocean_heatwave_detection", "ocean_hypoxia_detection"}:
        return nodes

    load_index = next(
        (
            index
            for index, node in enumerate(nodes)
            if (node.execution is not None and node.execution.tool_name == "load_dataset" and node.output is not None)
        ),
        None,
    )
    if load_index is None:
        return nodes
    source_node = nodes[load_index]
    source_id = source_node.output.artifact_id if source_node.output is not None else ""
    if not source_id:
        return nodes
    variable = str((source_node.execution.params if source_node.execution else {}).get("variable") or source_node.output.variable or "")
    if not variable:
        return nodes

    prefix = list(nodes[: load_index + 1])
    masked_id = _append_query_mask_nodes(context, prefix, source_id=source_id, variable=variable, vertical=vertical)
    if masked_id == source_id:
        return nodes

    suffix = [_replace_node_ref(node, source_id, masked_id) for node in nodes[load_index + 1 :]]
    return prefix + suffix


def _replace_node_ref(node: TaskNode, old_id: str, new_id: str) -> TaskNode:
    execution = node.execution
    if execution is None:
        return node
    params = _replace_ref_value(execution.params, old_id, new_id)
    updated_execution = replace(execution, params=params)
    return replace(node, inputs=_collect_ref_inputs(params), execution=updated_execution)


def _replace_ref_value(value: Any, old_id: str, new_id: str) -> Any:
    if isinstance(value, str):
        if value == f"$ref:{old_id}":
            return f"$ref:{new_id}"
        if value.startswith(f"$ref:{old_id}."):
            return "$ref:" + new_id + value[len(f"$ref:{old_id}") :]
        return value
    if isinstance(value, Mapping):
        return {key: _replace_ref_value(nested, old_id, new_id) for key, nested in value.items()}
    if isinstance(value, list):
        return [_replace_ref_value(nested, old_id, new_id) for nested in value]
    if isinstance(value, tuple):
        return tuple(_replace_ref_value(nested, old_id, new_id) for nested in value)
    return value


def _has_query_mask_request(context: PlanningContext) -> bool:
    return bool(
        _lookup_context_param("mask_polygon", context)
        or _lookup_context_param("polygon_points", context)
        or _lookup_context_param("mask_condition_expression", context)
        or _condition_mask_expression(context, default_variable=context.primary_variable())[0]
    )


def _is_condition_mask_spatial_map_request(context: PlanningContext) -> bool:
    if not _condition_mask_expression(context, default_variable=context.primary_variable())[0]:
        return False
    lowered = context.user_request.lower()
    if re.search(
        r"\b(trend|time[- ]?series|timeseries|spectrum|correlation|lag|custom|index|detect|detection|event|events|bloom|heatwave|hypoxia|upwelling|eutrophication)\b",
        lowered,
    ):
        return False
    return bool(
        re.search(r"\b(show|map|plot|display|visualize|region|regions|area|areas)\b|显示|区域|地图|画图|可视化", lowered)
    )


def _condition_mask_expression(context: PlanningContext, *, default_variable: str) -> Tuple[Optional[str], Tuple[str, ...]]:
    explicit = _lookup_context_param("mask_condition_expression", context) or _lookup_context_param("condition_expression", context)
    if isinstance(explicit, str) and explicit.strip():
        expression = explicit.strip()
        variables = _condition_variables_from_expression(expression) or (default_variable,)
        return expression, variables

    lowered = context.user_request.lower()
    parts: List[str] = []
    variables: List[str] = []
    pattern = re.compile(
        r"\b(?P<var>oxygen|chlorophyll|chl|temp|temperature|sst|salt|salinity)\b\s*"
        r"(?P<op><=|>=|<|>|below|under|less than|above|over|greater than)\s*"
        r"(?P<value>p\d{1,3}|threshold|[-+]?\d+(?:\.\d+)?)"
    )
    for match in pattern.finditer(lowered):
        variable = _canonical_condition_variable(match.group("var"))
        op = _canonical_condition_operator(match.group("op"))
        raw_value = match.group("value")
        value_expr = _condition_value_expression(raw_value, variable, context)
        if value_expr is None:
            continue
        parts.append(f"{variable} {op} {value_expr}")
        if variable not in variables:
            variables.append(variable)
    if not parts:
        return None, ()
    return " and ".join(parts), tuple(variables or [default_variable])


def _condition_variables_from_expression(expression: str) -> Tuple[str, ...]:
    variables: List[str] = []
    for raw in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression):
        variable = _canonical_condition_variable(raw)
        if variable in {"oxygen", "chlorophyll", "temp", "salt", "u", "v"} and variable not in variables:
            variables.append(variable)
    return tuple(variables)


def _canonical_condition_variable(value: str) -> str:
    lowered = value.lower()
    if lowered in {"chl", "chlorophyll"}:
        return "chlorophyll"
    if lowered in {"temperature", "sst", "temp"}:
        return "temp"
    if lowered in {"salinity", "salt"}:
        return "salt"
    return lowered


def _canonical_condition_operator(value: str) -> str:
    lowered = value.lower()
    if lowered in {"below", "under", "less than", "<"}:
        return "<"
    if lowered in {"above", "over", "greater than", ">"}:
        return ">"
    return lowered


def _condition_value_expression(value: str, variable: str, context: PlanningContext) -> Optional[str]:
    lowered = value.lower()
    if lowered.startswith("p") and lowered[1:].isdigit():
        return f"percentile({variable}, {int(lowered[1:])}, dim='time')"
    if lowered == "threshold":
        threshold = _lookup_context_param("threshold", context) or _lookup_context_param(f"{variable}_threshold", context)
        if threshold is None:
            return None
        try:
            return str(float(threshold))
        except (TypeError, ValueError):
            return str(threshold)
    return str(float(value))


def _vertical_for_workflow(context: PlanningContext, workflow: WorkflowTemplate) -> VerticalSpec:
    requested = context.vertical_spec(default_bottom=workflow.event_type == "hypoxia")
    if requested.mode == "bottom_band":
        return VerticalSpec(
            mode="bottom",
            aggregation=requested.aggregation or workflow.defaults.get("depth_aggregation") or "mean",
            source_text=requested.source_text,
        )
    if requested.mode != "unspecified":
        return requested

    default_mode = workflow.defaults.get("vertical_mode")
    if isinstance(default_mode, str) and default_mode:
        depth_range = _coerce_depth_range(workflow.defaults.get("depth_range"))
        depth_value = workflow.defaults.get("depth_value")
        try:
            depth_value = float(depth_value) if depth_value is not None else None
        except (TypeError, ValueError):
            depth_value = None
        return VerticalSpec(
            mode=default_mode,
            depth_value=depth_value,
            depth_range=depth_range,
            aggregation=str(workflow.defaults.get("depth_aggregation") or "mean"),
            source_text=context.user_request,
        )
    return requested


def _vertical_from_read_params(params: Mapping[str, Any], *, fallback: VerticalSpec) -> VerticalSpec:
    mode = params.get("vertical_mode")
    depth_range = _coerce_depth_range(params.get("depth_range"))
    depth_value: Optional[float] = None
    if params.get("depth_value") is not None:
        try:
            depth_value = float(params.get("depth_value"))
        except (TypeError, ValueError):
            depth_value = None

    if not isinstance(mode, str) or not mode:
        if depth_range is not None and depth_range[0] == 0.0 and depth_range[1] == 0.0:
            mode = "surface"
        elif depth_range is not None:
            mode = "depth_range"
        elif depth_value is not None:
            mode = "fixed_depth"
        else:
            mode = fallback.mode

    return VerticalSpec(
        mode=str(mode or "unspecified"),
        depth_value=depth_value if depth_value is not None else fallback.depth_value,
        depth_range=depth_range if depth_range is not None else fallback.depth_range,
        relative_to=fallback.relative_to,
        band_thickness_m=fallback.band_thickness_m,
        aggregation=fallback.aggregation,
        retain_depth=fallback.retain_depth,
        source_text=fallback.source_text,
    )


def _resolve_workflow_params(
    template: Mapping[str, Any],
    *,
    context: PlanningContext,
    workflow: WorkflowTemplate,
    vertical: VerticalSpec,
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    for key, value in template.items():
        item = _resolve_workflow_value(value, key=key, context=context, workflow=workflow, vertical=vertical)
        if item is None:
            continue
        resolved[key] = item
    return resolved


def _resolve_workflow_value(
    value: Any,
    *,
    key: str,
    context: PlanningContext,
    workflow: WorkflowTemplate,
    vertical: VerticalSpec,
) -> Any:
    if isinstance(value, Mapping):
        nested = {
            nested_key: _resolve_workflow_value(
                nested_value,
                key=str(nested_key),
                context=context,
                workflow=workflow,
                vertical=vertical,
            )
            for nested_key, nested_value in value.items()
        }
        return {nested_key: nested_value for nested_key, nested_value in nested.items() if nested_value is not None}
    if isinstance(value, list):
        return [
            item
            for item in (
                _resolve_workflow_value(item, key=key, context=context, workflow=workflow, vertical=vertical)
                for item in value
            )
            if item is not None
        ]
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        return _workflow_placeholder_value(value[1:-1], key=key, context=context, workflow=workflow, vertical=vertical)
    return value


def _workflow_placeholder_value(
    placeholder: str,
    *,
    key: str,
    context: PlanningContext,
    workflow: WorkflowTemplate,
    vertical: VerticalSpec,
) -> Any:
    if placeholder == "region.lon_range":
        return list(context.lon_range)
    if placeholder == "region.lat_range":
        return list(context.lat_range)
    if placeholder == "time_range":
        return list(context.time_range) if context.time_range is not None else None
    if placeholder == "vertical_mode":
        return None if vertical.mode == "unspecified" else vertical.mode
    if placeholder == "depth_value":
        return vertical.depth_value
    if placeholder == "depth_range":
        if vertical.depth_range is not None:
            return list(vertical.depth_range)
        default = workflow.defaults.get("depth_range")
        if default is not None:
            coerced = _coerce_depth_range(default)
            return list(coerced) if coerced is not None else default
        return None
    if placeholder == "depth_aggregation":
        return vertical.aggregation or workflow.defaults.get("depth_aggregation") or "mean"
    if placeholder == "variable":
        return context.primary_variable()
    if placeholder in {"variable1", "variable2"}:
        variable1, variable2 = _infer_lag_variables(context.user_request)
        return variable1 if placeholder == "variable1" else variable2
    if placeholder in {"depth_range1", "depth_range2"}:
        variable1, variable2 = _infer_lag_variables(context.user_request)
        variable = variable1 if placeholder == "depth_range1" else variable2
        depth_range = _infer_variable_depth_range(context.user_request, variable)
        return list(depth_range) if depth_range is not None else None
    if placeholder == "max_lag":
        return _extract_max_lag(context.user_request, default=int(workflow.defaults.get("max_lag", 12)))

    explicit = _lookup_context_param(placeholder, context)
    if explicit is not None:
        return explicit
    planner_default = _workflow_planner_parameter_default(workflow, placeholder)
    if planner_default is not _NO_WORKFLOW_PARAMETER_DEFAULT:
        return planner_default
    if placeholder in workflow.defaults:
        return workflow.defaults[placeholder]
    return None


_NO_WORKFLOW_PARAMETER_DEFAULT = object()


def _workflow_planner_parameter_default(workflow: WorkflowTemplate, name: str) -> Any:
    parameter = workflow.planner_parameters.get(name)
    if isinstance(parameter, Mapping) and "default" in parameter:
        return parameter.get("default")
    return _NO_WORKFLOW_PARAMETER_DEFAULT


def _lookup_context_param(name: str, context: PlanningContext) -> Any:
    for candidate in (context.extracted_params, context.additional_context.get("workspace_context", {}), context.additional_context):
        if isinstance(candidate, Mapping) and candidate.get(name) is not None:
            return candidate.get(name)
    if name in {"variable1", "variable2"}:
        variable1, variable2 = _infer_lag_variables(context.user_request)
        return variable1 if name == "variable1" else variable2
    if name in {"depth_range1", "depth_range2"}:
        variable1, variable2 = _infer_lag_variables(context.user_request)
        variable = variable1 if name == "depth_range1" else variable2
        depth_range = _infer_variable_depth_range(context.user_request, variable)
        return list(depth_range) if depth_range is not None else None
    if name == "max_lag":
        return _extract_max_lag(context.user_request, default=12)
    if name == "season_filter":
        return _extract_season_filter(context.user_request)
    if name == "time_aggregation":
        return _extract_time_aggregation(context.user_request)
    if name == "regional_gauge":
        return _extract_regional_gauge(context.user_request)
    if name in {"oxygen_threshold", "threshold"}:
        return _extract_threshold(context.user_request, default=None)
    return None


def _extract_season_filter(text: str) -> Optional[str]:
    lowered = text.lower()
    for label in ("DJF", "MAM", "JJA", "SON"):
        if re.search(rf"\b{label.lower()}\b", lowered):
            return label
    aliases = (
        ("winter", "winter"),
        ("spring", "spring"),
        ("summer", "summer"),
        ("fall", "fall"),
        ("autumn", "autumn"),
    )
    for token, value in aliases:
        if re.search(rf"\b{token}\b", lowered):
            return value
    return None


def _extract_time_aggregation(text: str) -> Optional[str]:
    lowered = text.lower()
    if re.search(r"\b(mean|average|averaged|monthly mean|seasonal mean|annual mean)\b", lowered):
        return "mean"
    if re.search(r"\b(sum|total|integrated over time)\b", lowered):
        return "sum"
    if re.search(r"\b(maximum|max)\b", lowered):
        return "max"
    if re.search(r"\b(minimum|min)\b", lowered):
        return "min"
    return None


def _extract_regional_gauge(text: str) -> Optional[str]:
    lowered = text.lower()
    if "gan fig" in lowered or (
        "streamfunction" in lowered
        and "china seas" in lowered
        and ("western pacific" in lowered or "west pacific" in lowered)
    ):
        return "gan_fig10_china_seas"
    return None


def _tool_call_node_id(index: int, tool_name: str, output_id: str) -> str:
    if tool_name == "load_dataset":
        return f"read_{output_id}"
    if tool_name.startswith("detect_"):
        return output_id
    if tool_name == "compute_event_summary_map":
        return f"summarize_{output_id}"
    if tool_name == "compute_event_frequency_map":
        return "compute_event_frequency_map"
    return f"{index}_{output_id}"


def _node_type_for_tool(tool_name: str) -> NodeType:
    if tool_name == "load_dataset":
        return NodeType.READ
    if tool_name.startswith("detect_"):
        return NodeType.DERIVE
    if tool_name.startswith("compute_event_") or tool_name in {
        "extract_timeseries",
        "extract_regional_mean",
        "compute_area_weighted_mean",
        "compute_volume_weighted_mean",
        "compute_tracer_horizontal_advection_timeseries",
        "compute_vertical_stability_timeseries",
        "compute_budget_residual",
    }:
        return NodeType.REDUCE
    return NodeType.DIAGNOSE


def _tool_output_spec(tool_name: str) -> Tuple[ArtifactKind, Tuple[str, ...], FrontendType]:
    if tool_name == "load_dataset":
        return ArtifactKind.FIELD, ("time", "depth", "lat", "lon"), FrontendType.DATA_CONTAINER
    if tool_name == "select_vertical":
        return ArtifactKind.FIELD, ("time", "lat", "lon"), FrontendType.DATA_CONTAINER
    if tool_name in {"build_threshold_mask", "build_condition_mask"}:
        return ArtifactKind.MASK, ("time", "lat", "lon"), FrontendType.DATA_CONTAINER
    if tool_name in {"build_polygon_mask", "build_isobath_mask", "combine_masks"}:
        return ArtifactKind.MASK, ("lat", "lon"), FrontendType.DATA_CONTAINER
    if tool_name in {
        "assemble_dataset",
        "apply_mask",
        "compute_speed_from_uv",
        "compute_density",
        "compute_derived_field",
        "compute_stratification_index",
        "compute_brunt_vaisala_frequency",
        "compute_field_anomaly",
        "compute_field_climatology",
        "compute_layer_mean",
        "compute_mixed_layer_mean",
        "compute_local_tendency",
        "compute_horizontal_advection",
        "compute_vertical_advection",
        "compute_front_proximity_index",
        "compute_eddy_influence_mask",
        "compute_tracer_gradient_alignment",
        "compute_mesoscale_background_separation",
        "compute_flow_structure_context",
        "filter_mesoscale_component",
        "replace_field_with_climatology",
    }:
        return ArtifactKind.FIELD, ("time", "depth", "lat", "lon"), FrontendType.DATA_CONTAINER
    if tool_name.startswith("detect_"):
        return ArtifactKind.TABLE, (), FrontendType.GENERIC
    if tool_name == "compute_watermass_event_association":
        return ArtifactKind.TABLE, (), FrontendType.GENERIC
    if tool_name in {
        "compute_event_summary_map",
        "compute_event_frequency_map",
        "compute_spatial_vorticity_map",
        "compute_transport_streamfunction_map",
        "build_watermass_tile_map",
    }:
        return ArtifactKind.MAP, ("lat", "lon"), FrontendType.SPATIAL_FIELD
    if tool_name in {
        "compute_event_timeseries_count",
        "extract_timeseries",
        "extract_regional_mean",
        "compute_area_weighted_mean",
        "compute_volume_weighted_mean",
        "compute_tracer_horizontal_advection_timeseries",
        "compute_vertical_stability_timeseries",
        "compute_budget_residual",
        "remove_seasonal_cycle",
    }:
        return ArtifactKind.SERIES, ("time",), FrontendType.TIMESERIES
    if tool_name == "compare_budget_term_magnitudes":
        return ArtifactKind.TABLE, (), FrontendType.GENERIC
    if tool_name == "compute_lag_correlation":
        return ArtifactKind.TABLE, (), FrontendType.LAG_CORRELATION
    if tool_name == "compute_trend":
        return ArtifactKind.TABLE, (), FrontendType.TREND
    if tool_name == "compute_field_trend":
        return ArtifactKind.MAP, ("lat", "lon"), FrontendType.FIELD_TREND
    if tool_name == "compute_spectrum":
        return ArtifactKind.SPECTRUM, ("frequency",), FrontendType.SPECTRUM
    if tool_name in {"compute_hovmoller", "compute_transect_normal_flux_hovmoller"}:
        return ArtifactKind.HOVMOLLER, ("time", "depth"), FrontendType.HOVMOLLER
    if tool_name == "build_watermass_ts_diagram":
        return ArtifactKind.GENERIC, (), FrontendType.GENERIC
    return ArtifactKind.GENERIC, (), FrontendType.GENERIC


def _tool_call_output_spec(step: Any, tool_name: str) -> Tuple[ArtifactKind, Tuple[str, ...], FrontendType]:
    output = getattr(step, "output_artifact", None)
    if isinstance(output, Mapping) and output:
        kind = _artifact_kind_from_value(output.get("kind"))
        frontend_type = _frontend_type_from_value(output.get("frontend_type"))
        dims_value = output.get("dims")
        dims = tuple(str(dim) for dim in dims_value) if isinstance(dims_value, (list, tuple)) else None
        default_kind, default_dims, default_frontend = _tool_output_spec(tool_name)
        return kind or default_kind, dims or default_dims, frontend_type or default_frontend
    return _tool_output_spec(tool_name)


def _artifact_kind_from_value(value: Any) -> Optional[ArtifactKind]:
    if isinstance(value, ArtifactKind):
        return value
    if isinstance(value, str) and value in ArtifactKind._value2member_map_:
        return ArtifactKind(value)
    return None


def _frontend_type_from_value(value: Any) -> Optional[FrontendType]:
    if isinstance(value, FrontendType):
        return value
    if isinstance(value, str) and value in FrontendType._value2member_map_:
        return FrontendType(value)
    return None


def _collect_ref_inputs(value: Any) -> Dict[str, str]:
    refs: Dict[str, str] = {}

    def visit(item: Any) -> None:
        if isinstance(item, str) and item.startswith("$ref:"):
            ref = item[5:].split(".", 1)[0]
            refs.setdefault(ref, ref)
        elif isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return refs


def _missing_workflow_fields(context: PlanningContext, workflow: WorkflowTemplate) -> List[str]:
    missing: List[str] = []
    if "requires_region" in workflow.validation_rules and not context.has_explicit_region():
        missing.append("region.lon_range/lat_range")
    if "requires_time_dimension" in workflow.validation_rules and not context.has_explicit_time_range():
        missing.append("time_range")
    if "requires_two_variables" in workflow.validation_rules:
        variable1, variable2 = _infer_lag_variables(context.user_request)
        if not variable1 or not variable2 or variable1 == variable2:
            missing.append("variable1/variable2")
    return missing


def _clarification_plan(context: PlanningContext, workflow: WorkflowTemplate, missing: Iterable[str]) -> Dict[str, Any]:
    missing_list = list(missing)
    empty_graph = {
        "graph_id": "ocean_harness_graph:semantic",
        "metadata": {
            "planning_model": "artifact_first",
            "status": "clarification_needed",
            "skill_usage": {"selected_skill": workflow.skill_id, "selected_workflow": workflow.workflow_id},
            "manual_usage": {"selected_manual": workflow.skill_id, "selected_recipe": workflow.workflow_id},
        },
        "data_requirements": [],
        "mask_requirements": [],
        "artifacts": [],
        "tasks": [],
        "final_artifacts": [],
    }
    return {
        "status": "clarification_needed",
        "skill_id": "ocean_harness",
        "skills_used": ["ocean_harness", workflow.skill_id],
        "planner_kind": "ocean_harness",
        "missing_fields": missing_list,
        "question": _workflow_clarification_question(missing_list),
        "data_requirements": {
            "lon_range": list(context.lon_range),
            "lat_range": list(context.lat_range),
            "time_range": list(context.time_range) if context.time_range is not None else None,
            "dataset": context.dataset,
        },
        "task_graph": {"graph_id": "ocean_harness_graph", "nodes": [], "final_artifacts": [], "metadata": {}},
        "semantic_task_graph": empty_graph,
        "steps": [],
    }


def _workflow_clarification_question(missing: Iterable[str]) -> str:
    missing_list = [str(item) for item in missing if str(item).strip()]
    labels = set(missing_list)
    missing_text = ", ".join(missing_list)
    needs_region = "region.lon_range/lat_range" in labels
    needs_time = "time_range" in labels
    if needs_region and needs_time:
        return f"Please provide the analysis region and time range before I run this event analysis. Missing fields: {missing_text}."
    if needs_region:
        return f"Please provide the analysis region, for example lon/lat bounds or the current selected box. Missing fields: {missing_text}."
    if needs_time:
        return f"Please provide the time range for this event analysis. Missing fields: {missing_text}."
    if missing_text:
        return f"Please provide the missing analysis inputs: {missing_text}."
    return "The planner requested clarification but did not identify any missing fields."


def _artifact(
    artifact_id: str,
    kind: ArtifactKind,
    dims: Tuple[str, ...],
    frontend_type: FrontendType,
    *,
    variable: Optional[str] = None,
) -> ArtifactSpec:
    return ArtifactSpec(
        artifact_id=artifact_id,
        kind=kind,
        shape=shape_spec_from_dims(dims),
        frontend_type=frontend_type,
        variable=variable,
    )


def _llm_skill_header_briefs() -> List[Dict[str, Any]]:
    briefs: List[Dict[str, Any]] = []
    for skill in load_skill_specs().values():
        header = skill.header
        briefs.append(
            {
                "skill_id": skill.skill_id,
                "description": _one_sentence_skill_description(
                    header.description if header else skill.markdown,
                    fallback=skill.title,
                ),
                "input_intent": header.input_intent if header else "",
                "output_intent": header.output_intent if header else "",
                "avoid_when": list(header.avoid_when if header else ()),
                "variables": list(header.variables if header else ()),
                "composes_with": list(header.composes_with if header else ()),
            }
        )
    return briefs


def _llm_skill_workflow_briefs(skill_ids: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    selected = {str(skill_id).strip() for skill_id in skill_ids or [] if str(skill_id).strip()}
    details: Dict[str, Dict[str, Any]] = {}
    for skill in load_skill_specs().values():
        if selected and skill.skill_id not in selected:
            continue
        workflow = skill.workflow
        workflow_payload: Dict[str, Any] = {
            "workflow_id": workflow.workflow_id,
            "steps": _workflow_step_briefs(workflow),
            "final_artifacts": list(workflow.final_artifacts),
        }
        if workflow.event_type:
            workflow_payload["event_type"] = workflow.event_type
        if workflow.planner_parameters:
            workflow_payload["planner_parameters"] = dict(workflow.planner_parameters)
        if workflow.required_inputs:
            workflow_payload["required_inputs"] = dict(workflow.required_inputs)
        if workflow.validation_rules:
            workflow_payload["validation_rules"] = list(workflow.validation_rules)
        workflow_payload["graph_contract"] = {
            "reference_rule": "$ref must target an earlier save_as artifact.",
            "unbound_template_refs": "Bind symbolic refs or skip that optional branch.",
        }
        details[skill.skill_id] = {
            "workflow": workflow_payload,
            "skill_manual": _truncate_text(skill.markdown, 16000),
        }
    return details


def _llm_skill_contract_briefs(skill_ids: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    selected = {str(skill_id).strip() for skill_id in skill_ids or [] if str(skill_id).strip()}
    details: Dict[str, Dict[str, Any]] = {}
    for skill in load_skill_specs().values():
        if selected and skill.skill_id not in selected:
            continue
        header = skill.header
        workflow = skill.workflow
        required_slots: List[str] = []
        if header is not None:
            required_slots.extend(str(item) for item in header.requires if str(item).strip())
        if workflow.required_inputs:
            required_slots.extend(str(item) for item in workflow.required_inputs.keys() if str(item).strip())
        optional_slots = set(str(item) for item in workflow.planner_parameters.keys())
        optional_slots.update(str(item) for item in workflow.defaults.keys())
        contract: Dict[str, Any] = {
            "skill_id": skill.skill_id,
            "description": _one_sentence_skill_description(
                header.description if header else skill.markdown,
                fallback=skill.title,
            ),
            "input_intent": header.input_intent if header else "",
            "output_intent": header.output_intent if header else "",
            "variables": list(header.variables if header else ()),
            "required_slots": sorted(set(required_slots)),
            "optional_slots": sorted(item for item in optional_slots if item),
            "defaults": dict(header.defaults if header else {}) or dict(workflow.defaults),
            "produces": list(header.produces if header else ()),
            "outputs": dict(workflow.output_policy),
            "event_type": workflow.event_type,
            "validation_rules": list(workflow.validation_rules),
            "composes_with": list(header.composes_with if header else ()),
        }
        if workflow.planner_parameters:
            contract["planner_parameters"] = dict(workflow.planner_parameters)
        details[skill.skill_id] = {"contract": contract}
    return details


def _one_sentence_skill_description(text: Any, *, fallback: str = "") -> str:
    source = re.sub(r"\s+", " ", str(text or "").strip())
    if not source:
        source = str(fallback or "").strip()
    if not source:
        return ""
    match = re.search(r"(.+?[.!?。！？])(?:\s|$)", source)
    sentence = match.group(1) if match else source
    return _truncate_text(sentence, 220)


def _workflow_step_briefs(workflow: WorkflowTemplate) -> List[Dict[str, Any]]:
    briefs: List[Dict[str, Any]] = []
    produced: set[str] = set()
    for step in workflow.steps:
        params = dict(step.params_template)
        unbound_refs = _unbound_template_refs(params, produced)
        step_brief: Dict[str, Any] = {
            "tool": step.tool,
            "save_as": step.save_as,
            "params": params,
        }
        if unbound_refs:
            step_brief["unbound_template_refs"] = sorted(unbound_refs)
        briefs.append(step_brief)
        produced.add(step.save_as)
    return briefs


def _unbound_template_refs(value: Any, produced_artifacts: set[str]) -> set[str]:
    refs: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, str):
            if item.startswith("$ref:"):
                path = item.removeprefix("$ref:").strip()
                root = path.split(".", 1)[0].strip()
                if root and root not in produced_artifacts:
                    refs.add(path)
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return refs


def _normalize_llm_harness_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(decision or {})
    status = str(normalized.get("status") or "ready").strip()
    normalized["status"] = "clarification_needed" if status == "clarification_needed" else "ready"
    route = str(normalized.get("route") or "generic_map").strip()
    valid_routes = {
        "dataset_info",
        "skill_workflow",
        "pair_lag_relationship",
        "hypoxia_driver",
        "generic_timeseries",
        "generated_code",
        "generic_map",
    }
    normalized["route"] = route if route in valid_routes else "generic_map"
    if not isinstance(normalized.get("resolved_scope"), dict):
        normalized["resolved_scope"] = {}
    else:
        scope = dict(normalized["resolved_scope"])
        depth_range = _coerce_depth_range(scope.get("depth_range"))
        if depth_range is not None:
            scope["depth_range"] = list(depth_range)
        depth_value = _coerce_depth_value(scope.get("depth_value"))
        if depth_value is not None:
            scope["depth_value"] = depth_value
        normalized["resolved_scope"] = scope
    if not isinstance(normalized.get("parameters"), dict):
        normalized["parameters"] = {}
    missing = normalized.get("missing_fields")
    normalized["missing_fields"] = [
        str(item)
        for item in missing
        if str(item).strip()
    ] if isinstance(missing, list) else []
    return normalized


def _decision_has_workflow_code(decision: Mapping[str, Any]) -> bool:
    workflow_code = decision.get("workflow_code")
    return isinstance(workflow_code, str) and bool(workflow_code.strip())


def _merge_semantic_plan_into_decision(decision: Dict[str, Any]) -> None:
    semantic_plan = decision.get("semantic_plan")
    if not isinstance(semantic_plan, Mapping):
        return
    shared_scope = semantic_plan.get("shared_scope")
    skills = semantic_plan.get("skills")
    composition = semantic_plan.get("composition")

    scope = dict(decision.get("resolved_scope") if isinstance(decision.get("resolved_scope"), Mapping) else {})
    if isinstance(shared_scope, Mapping):
        for key in (
            "dataset",
            "variable",
            "variables",
            "lon_range",
            "lat_range",
            "time_range",
            "vertical_mode",
            "depth_value",
            "depth_range",
        ):
            if scope.get(key) in (None, "", []):
                value = shared_scope.get(key)
                if value is not None and value != "":
                    scope[key] = value

    skill_slots: Dict[str, Dict[str, Any]] = {}
    first_slots: Mapping[str, Any] = {}
    if isinstance(skills, list):
        selected = set(_decision_selected_skill_ids(decision))
        for item in skills:
            if not isinstance(item, Mapping):
                continue
            skill_id = str(item.get("skill_id") or "").strip()
            if selected and skill_id not in selected:
                continue
            slots = item.get("slots") if isinstance(item.get("slots"), Mapping) else {}
            slot_dict = {
                str(key): value
                for key, value in dict(slots).items()
                if str(key).strip() and value is not None and value != ""
            }
            if skill_id:
                skill_slots[skill_id] = slot_dict
                if not first_slots:
                    first_slots = slot_dict

    for key in (
        "variable",
        "variables",
        "vertical_mode",
        "depth_value",
        "depth_range",
    ):
        if scope.get(key) in (None, "", []) and first_slots.get(key) is not None:
            scope[key] = first_slots.get(key)
    if "variable" not in scope and isinstance(scope.get("variables"), list) and scope["variables"]:
        scope["variable"] = scope["variables"][0]
    if "variables" not in scope and isinstance(scope.get("variable"), str) and scope["variable"].strip():
        scope["variables"] = [scope["variable"].strip()]
    decision["resolved_scope"] = scope

    parameters = dict(decision.get("parameters") if isinstance(decision.get("parameters"), Mapping) else {})
    if skill_slots:
        parameters.setdefault("skill_slots", skill_slots)
        for key, value in first_slots.items():
            if key not in {
                "variable",
                "variables",
                "vertical_mode",
                "depth_value",
                "depth_range",
                "depth_aggregation",
            }:
                parameters.setdefault(key, value)
    if isinstance(composition, Mapping):
        parameters.setdefault("composition", dict(composition))
    decision["parameters"] = parameters


def _merge_workflow_todo_scope_into_decision(decision: Dict[str, Any]) -> None:
    workflow_todo = decision.get("workflow_todo")
    if not isinstance(workflow_todo, Mapping):
        return
    shared_scope = workflow_todo.get("shared_scope")
    if not isinstance(shared_scope, Mapping):
        return
    scope = decision.get("resolved_scope") if isinstance(decision.get("resolved_scope"), Mapping) else {}
    merged_scope = dict(scope)
    for key in (
        "dataset",
        "variable",
        "variables",
        "lon_range",
        "lat_range",
        "time_range",
        "vertical_mode",
        "depth_value",
        "depth_range",
    ):
        if key not in merged_scope or merged_scope.get(key) in (None, "", []):
            value = shared_scope.get(key)
            if value is not None and value != "":
                merged_scope[key] = value
    decision["resolved_scope"] = merged_scope


def _merge_llm_parameters_into_extracted(
    extracted_params: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = dict(extracted_params or {})
    scope = decision.get("resolved_scope") if isinstance(decision.get("resolved_scope"), Mapping) else {}
    parameters = decision.get("parameters") if isinstance(decision.get("parameters"), Mapping) else {}
    merged.update(dict(parameters or {}))

    variable = scope.get("variable") if isinstance(scope, Mapping) else None
    if isinstance(variable, str) and variable.strip():
        merged["variable"] = variable.strip()
    variables = scope.get("variables") if isinstance(scope, Mapping) else None
    if isinstance(variables, list):
        normalized_variables = [str(item).strip() for item in variables if str(item).strip()]
        if normalized_variables:
            merged["variables"] = normalized_variables

    lon_range = scope.get("lon_range") if isinstance(scope, Mapping) else None
    lat_range = scope.get("lat_range") if isinstance(scope, Mapping) else None
    coerced_lon = _coerce_range(lon_range)
    coerced_lat = _coerce_range(lat_range)
    if coerced_lon is not None and coerced_lat is not None:
        merged["region"] = {
            "lon_range": list(coerced_lon),
            "lat_range": list(coerced_lat),
        }
    time_range = scope.get("time_range") if isinstance(scope, Mapping) else None
    coerced_time = _coerce_str_range(time_range)
    if coerced_time is not None:
        merged["time_range"] = list(coerced_time)

    if isinstance(scope, Mapping):
        for key in ("vertical_mode",):
            if scope.get(key) is not None:
                merged[key] = scope.get(key)
        depth_value = _coerce_depth_value(scope.get("depth_value"))
        if depth_value is not None:
            merged["depth_value"] = depth_value
        depth_range = _coerce_depth_range(scope.get("depth_range"))
        if depth_range is not None:
            merged["depth_range"] = list(depth_range)
    return merged


def _workflow_from_llm_decision(decision: Mapping[str, Any]) -> Optional[WorkflowTemplate]:
    selected_skill_id = decision.get("selected_skill_id")
    if not isinstance(selected_skill_id, str) or not selected_skill_id.strip():
        return None
    skill = load_skill_specs().get(selected_skill_id.strip())
    return skill.workflow if skill is not None else None


def _workflow_from_llm_workflow_code(decision: Mapping[str, Any]) -> WorkflowTemplate:
    selected_skill_id = decision.get("selected_skill_id")
    if not isinstance(selected_skill_id, str) or not selected_skill_id.strip():
        raise ValueError("LLM workflow_code requires selected_skill_id.")
    workflow_code = decision.get("workflow_code")
    if not isinstance(workflow_code, str) or not workflow_code.strip():
        raise ValueError("LLM skill_workflow decision must include workflow_code.")

    markdown = _workflow_code_markdown(workflow_code)
    steps = tuple(parse_workflow_steps(markdown, skill_id=selected_skill_id.strip()))
    if not steps:
        raise ValueError("LLM workflow_code did not contain any tool-call workflow steps.")
    planner_parameters = dict(parse_workflow_planner_parameters(markdown, skill_id=selected_skill_id.strip()))
    final_artifacts = _llm_workflow_final_artifacts(decision)
    return WorkflowTemplate(
        workflow_id=f"{selected_skill_id.strip()}_llm_workflow",
        skill_id=selected_skill_id.strip(),
        intents=(),
        steps=steps,
        planner_parameters=planner_parameters,
        final_artifacts=tuple(final_artifacts),
    )


def _workflow_code_markdown(workflow_code: str) -> str:
    text = workflow_code.strip()
    return f"```python\n{text}\n```"


def _llm_workflow_final_artifacts(decision: Mapping[str, Any]) -> List[str]:
    explicit = decision.get("final_artifacts")
    if not isinstance(explicit, list):
        raise ValueError("LLM workflow_code decision must include final_artifacts.")
    final_artifacts = [str(item).strip() for item in explicit if str(item).strip()]
    if not final_artifacts:
        raise ValueError("LLM workflow_code decision final_artifacts cannot be empty.")
    return final_artifacts


def _merge_workflow_code_scope_into_decision(decision: Dict[str, Any], workflow: WorkflowTemplate) -> None:
    scope = decision.get("resolved_scope")
    if not isinstance(scope, dict):
        scope = {}
        decision["resolved_scope"] = scope

    load_params = [
        dict(step.params_template)
        for step in workflow.steps
        if step.tool == "load_dataset"
    ]
    variables = [
        str(params.get("variable")).strip()
        for params in load_params
        if isinstance(params.get("variable"), str) and params.get("variable").strip() and not str(params.get("variable")).startswith("{")
    ]
    if variables:
        scope["variables"] = variables
        scope["variable"] = variables[0]

    first_load = load_params[0] if load_params else {}
    lon_range = _coerce_range(first_load.get("lon_range"))
    lat_range = _coerce_range(first_load.get("lat_range"))
    time_range = _coerce_str_range(first_load.get("time_range"))
    depth_range = _coerce_depth_range(first_load.get("depth_range"))
    depth_value = _coerce_depth_value(first_load.get("depth_value"))
    if lon_range is not None:
        scope["lon_range"] = list(lon_range)
    if lat_range is not None:
        scope["lat_range"] = list(lat_range)
    if time_range is not None:
        scope["time_range"] = list(time_range)
    if depth_range is not None:
        scope["depth_range"] = list(depth_range)
    if depth_value is not None:
        scope["depth_value"] = depth_value
    if (
        isinstance(first_load.get("vertical_mode"), str)
        and first_load.get("vertical_mode").strip()
        and not _is_template_placeholder(first_load.get("vertical_mode"))
    ):
        scope["vertical_mode"] = first_load.get("vertical_mode").strip()


def _is_template_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("{") and value.endswith("}")


def _nodes_from_llm_task_graph(
    decision: Mapping[str, Any],
    context: "PlanningContext",
    *,
    code_agent: Optional[CodeAgent] = None,
) -> List[TaskNode]:
    task_graph = decision.get("task_graph") if isinstance(decision.get("task_graph"), Mapping) else {}
    node_payloads = task_graph.get("nodes") if isinstance(task_graph, Mapping) else None
    if not isinstance(node_payloads, list):
        return []

    nodes: List[TaskNode] = []
    used_node_ids: set[str] = set()
    used_outputs: set[str] = set()
    for index, payload in enumerate(node_payloads, start=1):
        if not isinstance(payload, Mapping):
            continue
        tool_name = str(payload.get("tool") or payload.get("operation") or "").strip()
        if not tool_name:
            continue
        output_id = _safe_identifier(payload.get("save_as") or payload.get("output") or payload.get("id") or f"step_{index}")
        output_id = _unique_identifier(output_id, used_outputs)
        used_outputs.add(output_id)
        node_id = _safe_identifier(payload.get("id") or _tool_call_node_id(index, tool_name, output_id))
        node_id = _unique_identifier(node_id, used_node_ids)
        used_node_ids.add(node_id)
        params = _normalize_llm_node_params(payload.get("params"), context=context, tool_name=tool_name)
        if _is_generated_code_tool(tool_name):
            input_id = _generated_code_input_id(params, nodes)
            code_node = _generated_code_node(
                context,
                input_id=input_id,
                output_id=output_id,
                code_params=params,
                code_agent=code_agent,
            )
            nodes.append(
                replace(
                    code_node,
                    node_id=node_id,
                    intent=str(payload.get("description") or payload.get("intent") or code_node.intent),
                )
            )
            continue
        variable = str(params.get("variable") or "") if tool_name == "load_dataset" else None
        kind, dims, frontend_type = _tool_output_spec(tool_name)
        read_spec = None
        if tool_name == "load_dataset" and variable:
            read_vertical = _vertical_from_read_params(params, fallback=context.vertical_spec())
            read_spec = ReadSpec(
                variables=(variable,),
                region={"lon_range": context.lon_range, "lat_range": context.lat_range},
                time_range=context.time_range,
                vertical=read_vertical,
                expected_shape=ShapeClass.FIELD_4D,
                dataset=str(params.get("dataset") or context.dataset),
            )
        nodes.append(
            TaskNode(
                node_id=node_id,
                node_type=_node_type_for_tool(tool_name),
                intent=str(payload.get("description") or payload.get("intent") or f"{tool_name} from planner task graph"),
                operation=tool_name,
                inputs=_collect_ref_inputs(params),
                execution=ExecutionSpec(strategy=ExecutionStrategy.TOOL, tool_name=tool_name, params=params),
                read_spec=read_spec,
                output=_artifact(output_id, kind, dims, frontend_type, variable=variable),
            )
        )
    return nodes


def _nodes_from_llm_workflow_todo(
    decision: Mapping[str, Any],
    context: "PlanningContext",
    *,
    workflow: Optional[WorkflowTemplate] = None,
    code_agent: Optional[CodeAgent] = None,
) -> List[TaskNode]:
    workflow_todo = decision.get("workflow_todo")
    if isinstance(workflow_todo, list):
        shared_scope: Mapping[str, Any] = {}
        todo_items = workflow_todo
    elif isinstance(workflow_todo, Mapping):
        shared_scope = workflow_todo.get("shared_scope") if isinstance(workflow_todo.get("shared_scope"), Mapping) else {}
        todo_items = (
            workflow_todo.get("todo")
            if isinstance(workflow_todo.get("todo"), list)
            else workflow_todo.get("nodes")
            if isinstance(workflow_todo.get("nodes"), list)
            else workflow_todo.get("steps")
            if isinstance(workflow_todo.get("steps"), list)
            else []
        )
    else:
        return []
    if not isinstance(todo_items, list):
        return []

    _validate_workflow_todo_tools(decision, todo_items)
    node_payloads = _workflow_todo_items_to_task_payloads(
        todo_items,
        shared_scope=shared_scope,
        context=context,
        workflow=workflow,
    )
    if not node_payloads:
        return []
    task_graph = {"nodes": node_payloads, "final_artifacts": _workflow_todo_final_artifact_ids(decision)}
    return _nodes_from_llm_task_graph({"task_graph": task_graph}, context, code_agent=code_agent)


def _validate_workflow_todo_tools(decision: Mapping[str, Any], todo_items: Iterable[Any]) -> None:
    selected_skill_ids = _decision_selected_skill_ids(decision)
    allowed_tools = set(_WORKFLOW_TODO_BASE_TOOLS)
    selected_labels: List[str] = []
    skill_specs = load_skill_specs()
    for skill_id in selected_skill_ids:
        spec = skill_specs.get(skill_id)
        if spec is None:
            continue
        selected_labels.append(skill_id)
        allowed_tools.update(step.tool for step in spec.workflow.steps)

    for raw_item in todo_items:
        if not isinstance(raw_item, Mapping):
            continue
        tool_name = _todo_tool_name(raw_item)
        if not tool_name:
            continue
        if _is_generated_code_tool(tool_name):
            selected_text = ", ".join(selected_labels or selected_skill_ids or ["(none)"])
            raise ValueError(
                "Planner contract violation: workflow_todo for skill_workflow cannot contain "
                f"generated-code tool '{tool_name}' while selected_skill_ids={selected_text}. "
                "Use a registered tool exposed by the selected skill workflow, or choose route='generated_code' "
                "only when no selected workflow tool covers the requested computation."
            )
        if tool_name not in allowed_tools:
            selected_text = ", ".join(selected_labels or selected_skill_ids or ["(none)"])
            allowed_text = ", ".join(sorted(allowed_tools))
            raise ValueError(
                "Planner contract violation: workflow_todo used tool "
                f"'{tool_name}', but selected_skill_ids={selected_text} expose only: {allowed_text}."
            )


def _decision_selected_skill_ids(decision: Mapping[str, Any]) -> List[str]:
    values: List[str] = []
    raw_ids = decision.get("selected_skill_ids")
    if isinstance(raw_ids, list):
        values.extend(str(item).strip() for item in raw_ids if str(item).strip())
    primary = str(decision.get("selected_skill_id") or "").strip()
    if primary:
        values.insert(0, primary)
    deduped: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _workflow_todo_items_to_task_payloads(
    todo_items: Iterable[Any],
    *,
    shared_scope: Mapping[str, Any],
    context: "PlanningContext",
    workflow: Optional[WorkflowTemplate] = None,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    alias_by_artifact: Dict[str, str] = {}
    load_output_by_key: Dict[str, str] = {}
    vertical = _vertical_for_workflow(context, workflow) if workflow is not None else None
    template_by_output = {
        step.save_as: step
        for step in workflow.steps
    } if workflow is not None else {}

    for index, raw_item in enumerate(todo_items, start=1):
        if not isinstance(raw_item, Mapping):
            continue
        tool_name = _todo_tool_name(raw_item)
        if not tool_name:
            continue
        output_id = _safe_identifier(
            raw_item.get("as")
            or raw_item.get("as_")
            or raw_item.get("save_as")
            or raw_item.get("output")
            or raw_item.get("id")
            or f"step_{index}"
        )
        template_step = template_by_output.get(output_id)
        template_param_keys: Optional[set[str]] = None
        if template_step is not None and template_step.tool == tool_name and workflow is not None and vertical is not None:
            template_param_keys = set(template_step.params_template.keys())
            params = _workflow_todo_params_from_template_step(
                raw_item,
                template_step=template_step,
                context=context,
                workflow=workflow,
                vertical=vertical,
                alias_by_artifact=alias_by_artifact,
                shared_scope=shared_scope,
            )
        else:
            params = dict(raw_item.get("params") or {}) if isinstance(raw_item.get("params"), Mapping) else {}
            params = _todo_params_with_inline_fields(raw_item, params, tool_name=tool_name)
            params = _apply_workflow_todo_shared_scope(params, shared_scope, tool_name=tool_name)
        params = _apply_workflow_todo_using(
            tool_name,
            params,
            raw_item.get("using") if "using" in raw_item else raw_item.get("inputs"),
            alias_by_artifact=alias_by_artifact,
        )
        params = _apply_workflow_todo_context_params(params, tool_name=tool_name, context=context)
        if template_param_keys is not None:
            _raise_if_todo_params_escape_template(
                output_id=output_id,
                params=params,
                template_param_keys=template_param_keys,
            )

        load_variables = _todo_load_variables(raw_item, params, tool_name=tool_name)
        if tool_name == "load_dataset" and len(load_variables) > 1:
            variable_refs: Dict[str, str] = {}
            for variable in load_variables:
                load_output_id = _safe_identifier(f"{output_id}_{variable}")
                load_params = dict(params)
                load_params.pop("variables", None)
                load_params["variable"] = variable
                resolved_output = _append_or_alias_load_payload(
                    payloads,
                    output_id=load_output_id,
                    params=load_params,
                    context=context,
                    alias_by_artifact=alias_by_artifact,
                    load_output_by_key=load_output_by_key,
                    description=str(raw_item.get("description") or raw_item.get("intent") or ""),
                )
                variable_refs[variable] = f"$ref:{resolved_output}.data"
            payloads.append(
                {
                    "id": _safe_identifier(raw_item.get("id") or f"assemble_{output_id}"),
                    "tool": "assemble_dataset",
                    "params": {"variables": variable_refs},
                    "save_as": output_id,
                    "description": str(raw_item.get("description") or raw_item.get("intent") or "Assemble loaded variables"),
                }
            )
            alias_by_artifact[output_id] = output_id
            continue

        if tool_name == "load_dataset":
            if len(load_variables) == 1:
                params = dict(params)
                params["variable"] = load_variables[0]
                params.pop("variables", None)
            _append_or_alias_load_payload(
                payloads,
                output_id=output_id,
                params=params,
                context=context,
                alias_by_artifact=alias_by_artifact,
                load_output_by_key=load_output_by_key,
                description=str(raw_item.get("description") or raw_item.get("intent") or ""),
            )
            continue

        params = _rewrite_todo_param_refs(params, tool_name=tool_name, alias_by_artifact=alias_by_artifact)
        payloads.append(
            {
                "id": _safe_identifier(raw_item.get("id") or _tool_call_node_id(index, tool_name, output_id)),
                "tool": tool_name,
                "params": params,
                "save_as": output_id,
                "description": str(raw_item.get("description") or raw_item.get("intent") or f"{tool_name} from workflow_todo"),
            }
        )
        alias_by_artifact[output_id] = output_id
    return payloads


def _raise_if_todo_params_escape_template(
    *,
    output_id: str,
    params: Mapping[str, Any],
    template_param_keys: set[str],
) -> None:
    unknown_keys = sorted(key for key in params if key not in template_param_keys)
    if not unknown_keys:
        return
    raise ValueError(
        "Planner contract violation: workflow_todo item "
        f"'{output_id}' introduced parameter(s) outside the selected Python workflow step: "
        f"{', '.join(unknown_keys)}."
    )


def _workflow_todo_params_from_template_step(
    item: Mapping[str, Any],
    *,
    template_step: Any,
    context: "PlanningContext",
    workflow: WorkflowTemplate,
    vertical: VerticalSpec,
    alias_by_artifact: Mapping[str, str],
    shared_scope: Mapping[str, Any],
) -> Dict[str, Any]:
    """Instantiate a workflow_todo item from the parsed Python workflow step.

    The skill's Python block is the parameter contract. LLM todo may fill template
    placeholders (for example variable={variable}) or restate existing refs, but it
    must not add tool parameters absent from that code block.
    """
    params = _resolve_workflow_params(
        template_step.params_template,
        context=context,
        workflow=workflow,
        vertical=vertical,
    )
    params = _rewrite_template_alias_refs(params, alias_by_artifact=alias_by_artifact)
    params = _apply_template_declared_shared_scope(
        params,
        template_step=template_step,
        workflow=workflow,
        shared_scope=shared_scope,
    )

    overrides = dict(item.get("params") or {}) if isinstance(item.get("params"), Mapping) else {}
    overrides = _todo_params_with_inline_fields(item, overrides, tool_name=str(template_step.tool))
    template_keys = set(template_step.params_template.keys())
    unknown_keys = sorted(key for key in overrides if key not in template_keys)
    if unknown_keys:
        raise ValueError(
            "Planner contract violation: workflow_todo item "
            f"'{template_step.save_as}' supplied parameter(s) not present in the selected Python workflow step: "
            f"{', '.join(unknown_keys)}."
        )

    for key, value in overrides.items():
        template_value = template_step.params_template.get(key)
        if key in workflow.planner_parameters or _template_param_accepts_todo_override(template_value, params.get(key), value):
            params[key] = value
    return params


def _apply_template_declared_shared_scope(
    params: Dict[str, Any],
    *,
    template_step: Any,
    workflow: WorkflowTemplate,
    shared_scope: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = dict(params)
    template_keys = set(template_step.params_template.keys())
    for key, value in shared_scope.items():
        if key not in template_keys or value is None:
            continue
        template_value = template_step.params_template.get(key)
        if key in workflow.planner_parameters or _template_value_contains_placeholder(template_value):
            merged[str(key)] = value
    return merged


def _template_param_accepts_todo_override(template_value: Any, resolved_value: Any, override_value: Any) -> bool:
    if _template_value_contains_placeholder(template_value):
        return True
    if _template_value_contains_ref(template_value):
        return True
    return override_value == resolved_value


def _template_value_contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("{") and value.endswith("}")
    if isinstance(value, Mapping):
        return any(_template_value_contains_placeholder(nested) for nested in value.values())
    if isinstance(value, (list, tuple)):
        return any(_template_value_contains_placeholder(nested) for nested in value)
    return False


def _template_value_contains_ref(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("$ref:")
    if isinstance(value, Mapping):
        return any(_template_value_contains_ref(nested) for nested in value.values())
    if isinstance(value, (list, tuple)):
        return any(_template_value_contains_ref(nested) for nested in value)
    return False


def _todo_tool_name(item: Mapping[str, Any]) -> str:
    tool_name = str(item.get("tool") or item.get("operation") or "").strip()
    action = str(item.get("action") or "").strip().lower()
    if not tool_name and action == "load":
        tool_name = "load_dataset"
    if tool_name == "load":
        tool_name = "load_dataset"
    if tool_name == "run":
        tool_name = str(item.get("name") or item.get("tool_name") or "").strip()
    return tool_name


def _todo_params_with_inline_fields(item: Mapping[str, Any], params: Dict[str, Any], *, tool_name: str) -> Dict[str, Any]:
    merged = dict(params)
    for key in (
        "variable",
        "variables",
        "vertical_mode",
        "depth_value",
        "depth_range",
        "depth_aggregation",
        "lon_range",
        "lat_range",
        "time_range",
        "season_filter",
        "summary_mode",
        "mode",
        "time_aggregation",
        "regional_gauge",
    ):
        if key in item and key not in merged:
            merged[key] = item[key]
    if "mode" in merged and "summary_mode" not in merged and tool_name in {"compute_event_summary_map", "compute_event_frequency_map"}:
        merged["summary_mode"] = merged.pop("mode")
    return merged


def _todo_load_variables(item: Mapping[str, Any], params: Mapping[str, Any], *, tool_name: str) -> List[str]:
    if tool_name != "load_dataset":
        return []
    raw = params.get("variables") if params.get("variables") is not None else item.get("variables")
    if raw is None:
        raw = params.get("variable") if params.get("variable") is not None else item.get("variable")
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _apply_workflow_todo_shared_scope(params: Dict[str, Any], shared_scope: Mapping[str, Any], *, tool_name: str) -> Dict[str, Any]:
    merged = dict(params)
    if tool_name == "load_dataset":
        for key in ("dataset", "lon_range", "lat_range", "time_range", "season_filter", "vertical_mode", "depth_value", "depth_range"):
            if key in shared_scope and key not in merged and shared_scope.get(key) is not None:
                merged[key] = shared_scope.get(key)
        return merged
    if tool_name == "compute_transport_streamfunction_map":
        for key in ("depth_range", "time_aggregation", "regional_gauge"):
            if key in shared_scope and key not in merged and shared_scope.get(key) is not None:
                merged[key] = shared_scope.get(key)
        return merged
    if tool_name in {
        "compute_vertical_stability_timeseries",
        "compute_tracer_horizontal_advection_timeseries",
        "compute_spatial_field",
        "compute_event_frequency_map",
    }:
        for key in ("lon_range", "lat_range", "time_range", "depth_range", "depth_aggregation"):
            if key in shared_scope and key not in merged and shared_scope.get(key) is not None:
                merged[key] = shared_scope.get(key)
    if tool_name == "compute_transect_normal_flux_hovmoller":
        for key in ("transect_points", "depth_range", "n_samples", "method"):
            if key in shared_scope and key not in merged and shared_scope.get(key) is not None:
                merged[key] = shared_scope.get(key)
    return merged


def _apply_workflow_todo_context_params(
    params: Dict[str, Any],
    *,
    tool_name: str,
    context: "PlanningContext",
) -> Dict[str, Any]:
    merged = dict(params)
    if tool_name == "build_polygon_mask":
        if "polygon_points" not in merged:
            polygon_points = _lookup_context_param("mask_polygon", context) or _lookup_context_param("polygon_points", context)
            if polygon_points is not None:
                merged["polygon_points"] = polygon_points
        if "invert" not in merged:
            invert = _lookup_context_param("mask_polygon_invert", context)
            if invert is not None:
                merged["invert"] = bool(invert)
    if tool_name == "build_isobath_mask":
        if "isobath_depth" not in merged:
            isobath_depth = _lookup_context_param("mask_isobath_depth", context) or _lookup_context_param("isobath_depth", context)
            if isobath_depth is not None:
                merged["isobath_depth"] = isobath_depth
        if "comparison" not in merged:
            comparison = (
                _lookup_context_param("mask_isobath_comparison", context)
                or _lookup_context_param("isobath_comparison", context)
            )
            if comparison is not None:
                merged["comparison"] = comparison
    return merged


def _apply_workflow_todo_using(
    tool_name: str,
    params: Dict[str, Any],
    using: Any,
    *,
    alias_by_artifact: Mapping[str, str],
) -> Dict[str, Any]:
    merged = dict(params)
    if using is None:
        return merged
    if isinstance(using, Mapping):
        if tool_name == "assemble_dataset" and "variables" not in merged:
            merged["variables"] = _rewrite_todo_refs(dict(using), alias_by_artifact=alias_by_artifact)
            return merged
        for key, value in using.items():
            merged[str(key)] = _todo_ref(value, alias_by_artifact=alias_by_artifact)
        return merged
    values = list(using) if isinstance(using, (list, tuple)) else [using]
    list_param_name = _todo_using_list_param_name(tool_name)
    if list_param_name:
        merged[list_param_name] = [_todo_ref(value, alias_by_artifact=alias_by_artifact) for value in values]
        return merged
    param_order = _todo_using_param_order(tool_name, values)
    for key, value in zip(param_order, values):
        if key not in merged:
            merged[key] = _todo_ref(value, alias_by_artifact=alias_by_artifact)
    return merged


def _todo_using_list_param_name(tool_name: str) -> Optional[str]:
    if tool_name == "combine_masks":
        return "masks"
    if tool_name == "build_condition_mask":
        return "fields"
    return None


def _todo_using_param_order(tool_name: str, values: List[Any]) -> List[str]:
    if tool_name == "detect_hypoxia":
        return ["oxygen"]
    if tool_name == "detect_heatwaves":
        return ["temp"]
    if tool_name == "detect_algal_blooms":
        return ["chlorophyll"]
    if tool_name == "detect_eutrophication":
        return ["chlorophyll", "oxygen"]
    if tool_name == "compute_density":
        return ["data"]
    if tool_name == "compute_vertical_stability_timeseries":
        return ["density"] if len(values) == 1 else ["temp", "salt", "density"][: len(values)]
    if tool_name in {"compute_transport_streamfunction_map", "compute_transect_normal_flux_hovmoller"}:
        return ["u", "v"][: len(values)]
    if tool_name in {"compute_event_summary_map", "compute_event_frequency_map"}:
        return ["event_detection", "data"][: len(values)]
    if tool_name == "assemble_dataset":
        return ["variables"]
    return ["data", *[f"input_{index}" for index in range(2, len(values) + 1)]]


def _append_or_alias_load_payload(
    payloads: List[Dict[str, Any]],
    *,
    output_id: str,
    params: Mapping[str, Any],
    context: "PlanningContext",
    alias_by_artifact: Dict[str, str],
    load_output_by_key: Dict[str, str],
    description: str = "",
) -> str:
    normalized_params = _normalize_llm_node_params(params, context=context, tool_name="load_dataset")
    load_key = _canonical_load_key(normalized_params)
    existing_output = load_output_by_key.get(load_key)
    if existing_output:
        alias_by_artifact[output_id] = existing_output
        return existing_output
    payloads.append(
        {
            "id": _safe_identifier(f"read_{output_id}"),
            "tool": "load_dataset",
            "params": normalized_params,
            "save_as": output_id,
            "description": description or f"Load {normalized_params.get('variable', 'field')} data",
        }
    )
    load_output_by_key[load_key] = output_id
    alias_by_artifact[output_id] = output_id
    return output_id


def _canonical_load_key(params: Mapping[str, Any]) -> str:
    key_payload = {
        key: params.get(key)
        for key in (
            "dataset",
            "variable",
            "lon_range",
            "lat_range",
            "time_range",
            "season_filter",
            "vertical_mode",
            "depth_value",
            "depth_range",
            "depth_aggregation",
        )
        if params.get(key) is not None
    }
    return json.dumps(key_payload, sort_keys=True, ensure_ascii=False, default=str)


def _rewrite_todo_refs(value: Any, *, alias_by_artifact: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        return {key: _rewrite_todo_refs(nested, alias_by_artifact=alias_by_artifact) for key, nested in value.items()}
    if isinstance(value, list):
        return [_rewrite_todo_refs(nested, alias_by_artifact=alias_by_artifact) for nested in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_todo_refs(nested, alias_by_artifact=alias_by_artifact) for nested in value)
    return _todo_ref(value, alias_by_artifact=alias_by_artifact)


def _rewrite_todo_param_refs(value: Any, *, tool_name: str, alias_by_artifact: Mapping[str, str]) -> Any:
    ref_params = _todo_ref_param_names(tool_name)

    def visit(item: Any, *, key: str = "") -> Any:
        if isinstance(item, Mapping):
            if key in ref_params:
                return _rewrite_todo_refs(item, alias_by_artifact=alias_by_artifact)
            return {nested_key: visit(nested, key=str(nested_key)) for nested_key, nested in item.items()}
        if isinstance(item, list):
            if key in ref_params:
                return _rewrite_todo_refs(item, alias_by_artifact=alias_by_artifact)
            return [visit(nested, key=key) for nested in item]
        if isinstance(item, tuple):
            return tuple(visit(nested, key=key) for nested in item)
        if isinstance(item, str) and item.strip().startswith("$ref:"):
            return _todo_ref(item, alias_by_artifact=alias_by_artifact)
        if key in ref_params:
            return _todo_ref(item, alias_by_artifact=alias_by_artifact)
        return item

    return visit(value)


def _todo_ref_param_names(tool_name: str) -> set[str]:
    common = {
        "data",
        "field",
        "temp",
        "salt",
        "density",
        "oxygen",
        "chlorophyll",
        "u",
        "v",
        "u_data",
        "v_data",
        "event_detection",
        "events",
        "mask",
        "analysis_mask",
        "event_mask",
        "bathymetry",
        "masks",
        "fields",
    }
    if tool_name == "assemble_dataset":
        return {"variables"}
    if tool_name == "combine_masks":
        return {"masks"}
    if tool_name == "build_condition_mask":
        return {"fields"}
    return common


def _todo_ref(value: Any, *, alias_by_artifact: Mapping[str, str]) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text.startswith("$ref:"):
        ref = text[5:]
        root, suffix = _split_artifact_ref(ref)
        return "$ref:" + alias_by_artifact.get(root, root) + suffix
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?", text):
        root, suffix = _split_artifact_ref(text)
        return "$ref:" + alias_by_artifact.get(root, root) + suffix
    return value


def _split_artifact_ref(ref: str) -> Tuple[str, str]:
    if "." not in ref:
        return ref, ""
    root, suffix = ref.split(".", 1)
    return root, "." + suffix


def _workflow_todo_final_artifact_ids(decision: Mapping[str, Any]) -> List[str]:
    explicit = decision.get("final_artifacts")
    if isinstance(explicit, list):
        return [str(item).strip() for item in explicit if str(item).strip()]
    workflow_todo = decision.get("workflow_todo")
    if isinstance(workflow_todo, Mapping) and isinstance(workflow_todo.get("final_artifacts"), list):
        return [str(item).strip() for item in workflow_todo["final_artifacts"] if str(item).strip()]
    return []


def _llm_workflow_todo_final_artifacts(decision: Mapping[str, Any], nodes: Iterable[TaskNode]) -> List[str]:
    explicit = _workflow_todo_final_artifact_ids(decision)
    produced = [node.output.artifact_id for node in nodes if node.output is not None]
    selected = [artifact_id for artifact_id in explicit if artifact_id in produced]
    if selected:
        return selected
    consumed: set[str] = set()
    for node in nodes:
        consumed.update(str(value) for value in node.inputs.values())
    leaves = [
        node.output.artifact_id
        for node in nodes
        if node.output is not None and node.output.artifact_id not in consumed
    ]
    visible_leaves = [artifact_id for artifact_id in leaves if not _artifact_id_looks_like_raw_data(artifact_id)]
    return visible_leaves or leaves or produced[-1:]


def _is_generated_code_tool(tool_name: str) -> bool:
    return tool_name in {"generated_python_analysis", "generated_code", "run_generated_code"}


def _generated_code_input_id(params: Mapping[str, Any], nodes: List[TaskNode]) -> str:
    input_refs = params.get("input_refs") if isinstance(params.get("input_refs"), Mapping) else {}
    raw_ref = input_refs.get("field") if isinstance(input_refs, Mapping) else None
    if isinstance(raw_ref, str):
        match = re.match(r"^\$ref:([A-Za-z0-9_]+)(?:\.data)?$", raw_ref.strip())
        if match:
            return match.group(1)
    for node in reversed(nodes):
        if node.output is not None:
            return node.output.artifact_id
    return "raw_data"


def _normalize_llm_node_params(value: Any, *, context: "PlanningContext", tool_name: str) -> Dict[str, Any]:
    params = dict(value) if isinstance(value, Mapping) else {}
    if tool_name == "load_dataset":
        params.pop("depth_aggregation", None)
        params.setdefault("dataset", context.dataset)
        if "lon_range" not in params:
            params["lon_range"] = list(context.lon_range)
        if "lat_range" not in params:
            params["lat_range"] = list(context.lat_range)
        if "time_range" not in params and context.time_range is not None:
            params["time_range"] = list(context.time_range)
    elif tool_name == "detect_algal_blooms":
        threshold = params.get("threshold")
        if threshold is None:
            threshold = _lookup_context_param("threshold", context)
        percentile = params.get("percentile_threshold")
        if threshold is not None:
            params["threshold"] = threshold
            if percentile is None:
                params.pop("percentile_threshold", None)
        elif percentile is None:
            params["threshold"] = 1.0
            params.pop("percentile_threshold", None)
        bloom_type = str(params.get("bloom_type") or "").strip().lower()
        if bloom_type and bloom_type not in {"auto", "spring", "harmful"}:
            params.pop("bloom_type", None)
    elif tool_name == "detect_heatwaves":
        threshold = params.get("threshold")
        if threshold is None and params.get("percentile_threshold") is None:
            params["percentile_threshold"] = 90

    depth_value = _coerce_depth_value(params.get("depth_value"))
    if depth_value is not None:
        params["depth_value"] = depth_value
    depth_range = _coerce_depth_range(params.get("depth_range"))
    if depth_range is not None:
        params["depth_range"] = list(depth_range)

    _drop_null_optional_tool_params(params, tool_name)
    return params


def _drop_null_optional_tool_params(params: Dict[str, Any], tool_name: str) -> None:
    optional_defaults = {
        "detect_algal_blooms": {
            "threshold",
            "percentile_threshold",
            "min_duration_days",
            "min_area_km2",
            "bloom_type",
            "vertical_mode",
            "depth_value",
            "depth_range",
            "depth_aggregation",
            "analysis_mask",
        },
        "detect_heatwaves": {
            "percentile_threshold",
            "min_duration_days",
            "min_area_km2",
            "vertical_mode",
            "depth_value",
            "depth_range",
            "depth_aggregation",
            "analysis_mask",
        },
    }
    for key in optional_defaults.get(tool_name, set()):
        if params.get(key) is None:
            params.pop(key, None)


def _llm_final_artifacts(decision: Mapping[str, Any], nodes: Iterable[TaskNode]) -> List[str]:
    task_graph = decision.get("task_graph") if isinstance(decision.get("task_graph"), Mapping) else {}
    explicit = task_graph.get("final_artifacts") if isinstance(task_graph, Mapping) else None
    produced = [node.output.artifact_id for node in nodes if node.output is not None]
    if isinstance(explicit, list):
        selected = [str(item).strip() for item in explicit if str(item).strip() in produced]
        if selected:
            return selected

    consumed: set[str] = set()
    for node in nodes:
        consumed.update(str(value) for value in node.inputs.values())
    leaves = [
        node.output.artifact_id
        for node in nodes
        if node.output is not None and node.output.artifact_id not in consumed
    ]
    visible_leaves = [
        artifact_id
        for artifact_id in leaves
        if artifact_id and not _artifact_id_looks_like_raw_data(artifact_id)
    ]
    return visible_leaves or leaves or produced[-1:]


def _artifact_id_looks_like_raw_data(artifact_id: str) -> bool:
    lowered = artifact_id.lower()
    return lowered in {"raw", "raw_data"} or lowered.endswith("_raw") or lowered.startswith("raw_")


def _safe_identifier(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "node"


def _unique_identifier(value: str, used: set[str]) -> str:
    candidate = value
    counter = 2
    while candidate in used:
        candidate = f"{value}_{counter}"
        counter += 1
    return candidate


def _generic_llm_clarification_plan(
    context: "PlanningContext",
    missing: Iterable[str],
    decision: Mapping[str, Any],
) -> Dict[str, Any]:
    missing_list = [str(item) for item in missing if str(item).strip()]
    question = str(decision.get("clarification_question") or "").strip() or "Please provide the missing analysis inputs."
    if not missing_list:
        question = "The planner requested clarification but did not identify any missing fields."
    return {
        "status": "clarification_needed",
        "skill_id": "ocean_harness",
        "skills_used": ["ocean_harness"],
        "planner_kind": "ocean_harness",
        "missing_fields": missing_list,
        "question": question,
        "data_requirements": {
            "lon_range": list(context.lon_range),
            "lat_range": list(context.lat_range),
            "time_range": list(context.time_range) if context.time_range is not None else None,
            "dataset": context.dataset,
        },
        "task_graph": {"graph_id": "ocean_harness_graph", "nodes": [], "final_artifacts": [], "metadata": {}},
        "semantic_task_graph": {
            "graph_id": "ocean_harness_graph:semantic",
            "metadata": {
                "planning_model": "llm_harness",
                "status": "clarification_needed",
                "pipeline_route": decision.get("route"),
                "pipeline_reason": decision.get("reason"),
            },
            "data_requirements": [],
            "mask_requirements": [],
            "artifacts": [],
            "tasks": [],
            "final_artifacts": [],
        },
        "steps": [],
    }


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _plan_from_nodes(
    context: PlanningContext,
    nodes: Iterable[TaskNode],
    *,
    final_artifacts: Iterable[str],
    skill_id: str = "ocean_harness",
    skills_used: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    node_list = list(nodes)
    graph_metadata = {"planner": "OceanHarnessPlanner"}
    binding_report = ToolBinder().bind(node_list)
    graph_metadata["binding_report"] = {
        "tool_nodes": list(binding_report.tool_nodes),
        "code_nodes": list(binding_report.code_nodes),
        "missing_bindings": list(binding_report.missing_bindings),
    }
    matched_skills = list(retrieve_skill_specs(context.user_request, limit=6))
    if matched_skills:
        graph_metadata["matched_skills"] = matched_skills
        graph_metadata["matched_manuals"] = matched_skills
    if metadata:
        graph_metadata.update(dict(metadata))
    graph_metadata["compiled_steps_from"] = "semantic_task_graph"
    graph_metadata["step_projection_target"] = "legacy_executor_frontend_protocol"
    graph = task_graph_from_nodes(
        "ocean_harness_graph",
        node_list,
        final_artifacts=final_artifacts,
        metadata=graph_metadata,
    )
    contract_validation = validate_task_graph_contracts(graph)
    graph_metadata["contract_validation"] = [
        {"level": issue.level, "message": issue.message, "path": issue.path}
        for issue in contract_validation.issues
    ]
    contract_validation.raise_for_errors()
    semantic_graph = semantic_graph_from_task_graph(graph, user_request=context.user_request)
    semantic_payload = semantic_graph_to_dict(semantic_graph)
    synthesis_packet = packet_spec_to_dict(ArtifactGatherer().plan_packet(semantic_payload))
    return {
        "status": "ready",
        "skill_id": skill_id,
        "skills_used": list(skills_used or [skill_id]),
        "planner_kind": "ocean_harness",
        "data_requirements": {
            "lon_range": list(context.lon_range),
            "lat_range": list(context.lat_range),
            "time_range": list(context.time_range) if context.time_range is not None else None,
            "dataset": context.dataset,
            "skill_context": matched_skills,
            "manual_context": matched_skills,
            "data_access": semantic_payload["data_requirements"],
            "mask_requirements": semantic_payload["mask_requirements"],
            "artifact_contracts": semantic_payload["artifacts"],
            "mask_policy": semantic_payload["metadata"].get("mask_policy", {}),
            "shape_taxonomy": [
                ShapeClass.FIELD_4D.value,
                ShapeClass.FIELD_3D_TIME_MAP.value,
                ShapeClass.MAP_2D.value,
                ShapeClass.SECTION_2D.value,
                ShapeClass.SERIES_1D.value,
                ShapeClass.SCALAR.value,
            ],
        },
        "semantic_task_graph": semantic_payload,
        "synthesis_packet": synthesis_packet,
        "task_graph": _graph_to_dict(graph),
        "steps": [_node_to_step(node) for node in node_list],
    }


def _node_to_step(node: TaskNode) -> Dict[str, Any]:
    execution = node.execution or ExecutionSpec(strategy=ExecutionStrategy.HARNESS)
    output_id = node.output.artifact_id if node.output is not None else node.node_id
    human_label, technical_label = _step_labels_for_node(node)
    return {
        "step_id": node.node_id,
        "tool": execution.tool_name or node.operation,
        "params": dict(execution.params),
        "save_as": output_id,
        "human_label": human_label,
        "technical_label": technical_label,
        "harness_node": _node_to_dict(node),
    }


def _graph_to_dict(graph: Any) -> Dict[str, Any]:
    return {
        "graph_id": graph.graph_id,
        "final_artifacts": list(graph.final_artifacts),
        "metadata": dict(graph.metadata),
        "nodes": [_node_to_dict(node) for node in graph.nodes],
    }


def _node_to_dict(node: TaskNode) -> Dict[str, Any]:
    return {
        "node_id": node.node_id,
        "node_type": node.node_type.value,
        "intent": node.intent,
        "operation": node.operation,
        "inputs": dict(node.inputs),
        "output": _artifact_to_dict(node.output) if node.output is not None else None,
        "execution": {
            "strategy": node.execution.strategy.value if node.execution else "harness",
            "tool_name": node.execution.tool_name if node.execution else None,
            "code": node.execution.code if node.execution else None,
            "params": dict(node.execution.params) if node.execution else {},
        },
        "validation_rules": list(node.validation_rules),
    }


def _artifact_to_dict(spec: ArtifactSpec) -> Dict[str, Any]:
    return {
        "artifact_id": spec.artifact_id,
        "kind": spec.kind.value,
        "shape_class": spec.shape_class.value,
        "dims": list(spec.dims),
        "frontend_type": spec.frontend_type.value,
        "variable": spec.variable,
    }


def _humanize_node(node: TaskNode) -> str:
    labels = {
        NodeType.READ: "Read data",
        NodeType.SELECT: "Select data",
        NodeType.DERIVE: "Derive field",
        NodeType.REDUCE: "Reduce to series/map",
        NodeType.DIAGNOSE: "Diagnose signal",
        NodeType.PROJECT: "Project result",
        NodeType.SYNTHESIZE: "Synthesize evidence",
    }
    return labels.get(node.node_type, node.operation)


def _step_labels_for_node(node: TaskNode) -> Tuple[str, str]:
    human_label = _humanize_node(node)
    technical_label = node.intent
    execution = node.execution or ExecutionSpec(strategy=ExecutionStrategy.HARNESS)
    tool_name = execution.tool_name or node.operation

    if tool_name == "compute_watermass_event_association":
        return "Classify water masses", "Classify dominant water masses and compare hotspot tiles"

    if tool_name == "build_watermass_tile_map":
        map_kind = str(execution.params.get("map_kind") or "").strip().lower()
        if map_kind == "dominant_watermass":
            return "Project result", "Dominant water-mass tile map"
        return "Project result", "Bloom hotspot tile map"

    if tool_name == "build_watermass_ts_diagram":
        return "Project result", "Water-mass T-S diagram"

    return human_label, technical_label


def _looks_like_dataset_info_request(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\b(dataset|metadata|available variables|data info)\b|数据集|变量有哪些|元数据", lowered))


def _looks_like_hypoxia_driver_request(text: str) -> bool:
    lowered = text.lower()
    has_hypoxia = bool(re.search(r"\b(hypoxia|hypoxic|low oxygen|oxygen)\b|缺氧|低氧|溶解氧", lowered))
    has_complex = bool(re.search(r"\b(trend|spectrum|power spectrum|correlation|relationship|driver|temperature|salinity|speed)\b|趋势|功率谱|谱|相关|关系|温度|盐度|流速", lowered))
    return has_hypoxia and has_complex


def _looks_like_pair_lag_relationship_request(text: str) -> bool:
    lowered = text.lower()
    has_lag = bool(re.search(r"\b(lag|lags|lead|leads|time lag|cross[- ]correlation|delayed)\b|滞后|领先", lowered))
    has_relationship = bool(re.search(r"\b(related|relationship|correlation|coupling|between|with)\b|相关|关系|联系|耦合", lowered))
    variable1, variable2 = _infer_lag_variables(text)
    return has_lag and has_relationship and bool(variable1 and variable2 and variable1 != variable2)


def _looks_like_code_fallback_request(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\b(map|plot|show|visualize|display)\b|画图|地图|显示|可视化", lowered):
        return False
    return bool(
        re.search(
            r"\b(custom|index|metric|diagnose|diagnostic|estimate|calculate|compute|derive|relationship|related|compare|association)\b|"
            r"自定义|指标|诊断|估计|计算|推导|关系|比较|关联",
            lowered,
        )
    )


def _looks_like_trend_request(text: str) -> bool:
    return bool(re.search(r"\btrend|linear trend|long[- ]term\b|趋势|长期变化", text.lower()))


def _looks_like_spectrum_request(text: str) -> bool:
    return bool(re.search(r"\bspectrum|power spectrum|periodic|周期\b|功率谱|谱", text.lower()))


def _looks_like_timeseries_request(text: str) -> bool:
    return bool(re.search(r"\btime series|timeseries|变化序列|时间序列|随时间", text.lower()))


def _resolve_region(
    text: str,
    extracted_params: Mapping[str, Any],
    additional_context: Mapping[str, Any],
    dataset_info: Mapping[str, Any],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    text_region = _extract_region_from_text(text)
    if text_region is not None:
        return text_region

    extracted_region = _region_from_mapping(extracted_params)
    if extracted_region is not None:
        return extracted_region

    for candidate in (additional_context.get("workspace_context", {}), additional_context):
        region = _region_from_mapping(candidate)
        if region is not None:
            return region

    spatial = dataset_info.get("spatial_extent") if isinstance(dataset_info, Mapping) else {}
    lon = _coerce_range(spatial.get("lon") if isinstance(spatial, Mapping) else None) or (0.0, 360.0)
    lat = _coerce_range(spatial.get("lat") if isinstance(spatial, Mapping) else None) or (-90.0, 90.0)
    return lon, lat


def _region_from_mapping(candidate: Any) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if not isinstance(candidate, Mapping):
        return None
    lon_range = _coerce_range(candidate.get("lon_range") or candidate.get("longitude_range"))
    lat_range = _coerce_range(candidate.get("lat_range") or candidate.get("latitude_range"))
    region = candidate.get("region")
    if isinstance(region, Mapping):
        lon_range = lon_range or _coerce_range(region.get("lon_range") or region.get("longitude_range"))
        lat_range = lat_range or _coerce_range(region.get("lat_range") or region.get("latitude_range"))
    region_bounds = candidate.get("region_bounds") or candidate.get("current_region_bounds")
    if isinstance(region_bounds, Mapping):
        lon_range = lon_range or _coerce_range(region_bounds.get("lon") or region_bounds.get("lon_range"))
        lat_range = lat_range or _coerce_range(region_bounds.get("lat") or region_bounds.get("lat_range"))
        if lon_range is None and {"lon_min", "lon_max"}.issubset(region_bounds):
            lon_range = (float(region_bounds["lon_min"]), float(region_bounds["lon_max"]))
        if lat_range is None and {"lat_min", "lat_max"}.issubset(region_bounds):
            lat_range = (float(region_bounds["lat_min"]), float(region_bounds["lat_max"]))
    if lon_range is not None and lat_range is not None:
        return lon_range, lat_range
    return None


def _resolve_time_range(
    text: str,
    extracted_params: Mapping[str, Any],
    additional_context: Mapping[str, Any],
    dataset_info: Mapping[str, Any],
) -> Optional[Tuple[str, str]]:
    parsed_time_range = _extract_time_range_from_text(text)
    if parsed_time_range is not None:
        return parsed_time_range
    years = [int(match) for match in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text)]
    if len(years) >= 2:
        return f"{min(years)}-01-01", f"{max(years)}-12-31"
    if len(years) == 1:
        return f"{years[0]}-01-01", f"{years[0]}-12-31"
    time_range = _coerce_str_range(extracted_params.get("time_range")) if isinstance(extracted_params, Mapping) else None
    if time_range is not None:
        return time_range
    for candidate in (additional_context.get("workspace_context", {}), additional_context):
        if isinstance(candidate, Mapping):
            time_range = _coerce_str_range(candidate.get("time_range"))
            if time_range is not None:
                return time_range
    temporal = dataset_info.get("temporal_extent") if isinstance(dataset_info, Mapping) else {}
    if isinstance(temporal, Mapping) and temporal.get("start") and temporal.get("end"):
        return str(temporal["start"]), str(temporal["end"])
    return None


def _extract_region_from_text(text: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    named_region = _named_region_from_text(text)
    if named_region is not None:
        return named_region

    lon_patterns = (
        r"(?P<a>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[eE]\s*(?:-|–|—|to|到|至)\s*(?P<b>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[eE]",
        r"(?P<a>-?\d+(?:\.\d+)?)\s*(?:-|–|—|to|到|至)\s*(?P<b>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[eE]",
    )
    lat_patterns = (
        r"(?P<a>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[nN]\s*(?:-|–|—|to|到|至)\s*(?P<b>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[nN]",
        r"(?P<a>-?\d+(?:\.\d+)?)\s*(?:-|–|—|to|到|至)\s*(?P<b>-?\d+(?:\.\d+)?)\s*(?:°\s*)?[nN]",
    )
    lon_range = _first_range_match(text, lon_patterns)
    lat_range = _first_range_match(text, lat_patterns)
    if lon_range is not None and lat_range is not None:
        return lon_range, lat_range
    return None


def _named_region_from_text(text: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    lowered = text.lower()
    for name, bounds in _NAMED_REGION_BOUNDS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", lowered):
            return bounds
    return None


def _first_range_match(text: str, patterns: Iterable[str]) -> Optional[Tuple[float, float]]:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            first = float(match.group("a"))
            second = float(match.group("b"))
            return (min(first, second), max(first, second))
    return None


def _extract_time_range_from_text(text: str) -> Optional[Tuple[str, str]]:
    year_range = re.search(
        r"(19\d{2}|20\d{2})\s*(?:年)?\s*(?:-|–|—|to|through|until|到|至)\s*(19\d{2}|20\d{2})\s*(?:年)?",
        text,
        flags=re.IGNORECASE,
    )
    if year_range:
        start_year = int(year_range.group(1))
        end_year = int(year_range.group(2))
        return f"{min(start_year, end_year)}-01-01", f"{max(start_year, end_year)}-12-31"

    iso_range = re.search(
        r"(19\d{2}|20\d{2})[-/](\d{1,2})[-/](\d{1,2}).{0,20}?(19\d{2}|20\d{2})[-/](\d{1,2})[-/](\d{1,2})",
        text,
    )
    if iso_range:
        start = f"{int(iso_range.group(1)):04d}-{int(iso_range.group(2)):02d}-{int(iso_range.group(3)):02d}"
        end = f"{int(iso_range.group(4)):04d}-{int(iso_range.group(5)):02d}-{int(iso_range.group(6)):02d}"
        return start, end

    month_range = _extract_month_range(text)
    if month_range is not None:
        return month_range
    return None


_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _extract_month_range(text: str) -> Optional[Tuple[str, str]]:
    lowered = text.lower()
    month_names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    same_year = re.search(
        rf"\b({month_names})\b\s*(?:to|-|–|—|through|until|到|至)\s*\b({month_names})\b\s*(19\d{{2}}|20\d{{2}})",
        lowered,
    )
    if same_year:
        year = int(same_year.group(3))
        start_month = _MONTHS[same_year.group(1)]
        end_month = _MONTHS[same_year.group(2)]
        return _month_bounds(year, start_month, year, end_month)

    cross_year = re.search(
        rf"\b({month_names})\b\s*(19\d{{2}}|20\d{{2}})\s*(?:to|-|–|—|through|until|到|至)\s*\b({month_names})\b\s*(19\d{{2}}|20\d{{2}})",
        lowered,
    )
    if cross_year:
        return _month_bounds(
            int(cross_year.group(2)),
            _MONTHS[cross_year.group(1)],
            int(cross_year.group(4)),
            _MONTHS[cross_year.group(3)],
        )
    return None


def _month_bounds(start_year: int, start_month: int, end_year: int, end_month: int) -> Tuple[str, str]:
    last_day = calendar.monthrange(end_year, end_month)[1]
    return f"{start_year:04d}-{start_month:02d}-01", f"{end_year:04d}-{end_month:02d}-{last_day:02d}"


def _coerce_range(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, Mapping):
        if {"min", "max"}.issubset(value):
            return float(value["min"]), float(value["max"])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return None


def _coerce_str_range(value: Any) -> Optional[Tuple[str, str]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return str(value[0]), str(value[1])
    return None


def _coerce_depth_range(value: Any) -> Optional[Tuple[float, float]]:
    numeric = _coerce_range(value)
    if numeric is None:
        return None
    return tuple(sorted((abs(float(numeric[0])), abs(float(numeric[1])))))


def _coerce_depth_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return abs(numeric)


def _context_has_region(candidate: Any) -> bool:
    if not isinstance(candidate, Mapping):
        return False
    lon_range = _coerce_range(candidate.get("lon_range") or candidate.get("longitude_range"))
    lat_range = _coerce_range(candidate.get("lat_range") or candidate.get("latitude_range"))
    if lon_range is not None and lat_range is not None:
        return True
    region_bounds = candidate.get("region_bounds") or candidate.get("current_region_bounds")
    if isinstance(region_bounds, Mapping):
        if _coerce_range(region_bounds.get("lon") or region_bounds.get("lon_range")) and _coerce_range(
            region_bounds.get("lat") or region_bounds.get("lat_range")
        ):
            return True
        return {"lon_min", "lon_max", "lat_min", "lat_max"}.issubset(region_bounds)
    return False


def _context_has_time_range(candidate: Any) -> bool:
    return isinstance(candidate, Mapping) and _coerce_str_range(candidate.get("time_range")) is not None


def _contains_alias(text: str, alias: str) -> bool:
    alias_lower = alias.lower()
    if re.fullmatch(r"[a-z0-9_]+", alias_lower):
        return bool(re.search(rf"(?<![a-z0-9_]){re.escape(alias_lower)}(?![a-z0-9_])", text))
    return alias_lower in text


def _infer_lag_variables(text: str) -> Tuple[Optional[str], Optional[str]]:
    lowered = text.lower()
    matches: List[Tuple[int, str]] = []
    for variable, aliases in _VARIABLE_ALIASES.items():
        positions = [
            _alias_position(lowered, alias)
            for alias in aliases
        ]
        positions = [position for position in positions if position is not None]
        if positions:
            matches.append((min(positions), variable))
    matches.sort()
    ordered: List[str] = []
    for _, variable in matches:
        if variable not in ordered:
            ordered.append(variable)
    if len(ordered) >= 2:
        return ordered[0], ordered[1]
    return (ordered[0], None) if ordered else (None, None)


def _alias_position(text: str, alias: str) -> Optional[int]:
    alias_lower = alias.lower()
    if re.fullmatch(r"[a-z0-9_]+", alias_lower):
        match = re.search(rf"(?<![a-z0-9_]){re.escape(alias_lower)}(?![a-z0-9_])", text)
    else:
        match = re.search(re.escape(alias_lower), text)
    return match.start() if match else None


def _infer_variable_depth_range(text: str, variable: Optional[str]) -> Optional[Tuple[float, float]]:
    if not variable:
        return None
    lowered = text.lower()
    aliases = _VARIABLE_ALIASES.get(variable, (variable,))
    alias_pattern = "|".join(re.escape(alias.lower()) for alias in sorted(aliases, key=len, reverse=True))
    if variable == "temp" and re.search(r"\bsst\b|sea[- ]surface temperature|surface sst", lowered):
        return (0.0, 0.0)
    upper = re.search(rf"\bupper[- ]?(\d+(?:\.\d+)?)\s*(?:m|meter|meters|米).{{0,40}}(?:{alias_pattern})", lowered)
    if upper is None:
        upper = re.search(rf"(?:{alias_pattern}).{{0,40}}\bupper[- ]?(\d+(?:\.\d+)?)\s*(?:m|meter|meters|米)", lowered)
    if upper:
        return (0.0, float(upper.group(1)))
    fixed = re.search(rf"\bat\s*(\d+(?:\.\d+)?)\s*(?:m|meter|meters|米).{{0,40}}(?:{alias_pattern})", lowered)
    if fixed is None:
        fixed = re.search(rf"(?:{alias_pattern}).{{0,40}}\bat\s*(\d+(?:\.\d+)?)\s*(?:m|meter|meters|米)", lowered)
    if fixed:
        value = float(fixed.group(1))
        return (value, value)
    if re.search(rf"\b(surface|sea surface)\b.{{0,20}}(?:{alias_pattern})", lowered) or re.search(
        rf"(?:{alias_pattern}).{{0,20}}\b(surface|sea surface)\b",
        lowered,
    ):
        return (0.0, 0.0)
    return None


def _variable_vertical_spec(text: str, variable: str) -> VerticalSpec:
    depth_range = _infer_variable_depth_range(text, variable)
    if depth_range is not None:
        if depth_range[0] == 0.0 and depth_range[1] == 0.0:
            return VerticalSpec(mode="surface", depth_range=depth_range, aggregation="mean", source_text=text)
        if abs(depth_range[0] - depth_range[1]) < 1e-9:
            return VerticalSpec(mode="fixed_depth", depth_value=depth_range[0], depth_range=depth_range, aggregation="mean", source_text=text)
        return VerticalSpec(mode="depth_range", depth_range=depth_range, aggregation="mean", source_text=text)
    return VerticalSpec(mode="unspecified", source_text=text)


def _extract_bottom_band_thickness(text: str) -> Optional[float]:
    patterns = [
        r"(?:bottom|seafloor|near[- ]bottom).{0,20}?(\d+(?:\.\d+)?)\s*m",
        r"底(?:部|层|以上|以内|附近).{0,12}?(\d+(?:\.\d+)?)\s*(?:m|米)",
        r"(\d+(?:\.\d+)?)\s*(?:m|米).{0,12}?(?:bottom|seafloor|底)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _extract_fixed_depth(text: str) -> Optional[float]:
    match = re.search(r"(?:at|固定|深度)\s*(\d+(?:\.\d+)?)\s*(?:m|米)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|米)\s*(?:depth|深处|处)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _extract_depth_range(text: str) -> Optional[Tuple[float, float]]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to|到|至)\s*(\d+(?:\.\d+)?)\s*(?:m|米)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(r"(?:below|deeper than|深于|以下|大于)\s*(\d+(?:\.\d+)?)\s*(?:m|米)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)), 10000.0
    return None


def _extract_threshold(text: str, *, default: Optional[float]) -> Optional[float]:
    patterns = (
        r"(?:threshold|阈值|低于|小于|below|above|greater\s+than|less\s+than)\s*(\d+(?:\.\d+)?)",
        r"(?:chlorophyll|chla|temp|temperature|oxygen|speed|value)\s*(?:>=|>|<=|<)\s*(\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return default


def _extract_max_lag(text: str, *, default: int) -> int:
    patterns = (
        r"max(?:imum)?\s+lag\s*(?:of|=|:)?\s*(\d+)",
        r"(\d+)\s*(?:step|month|day|year|步|月|天|年)s?\s+(?:lag|lags|滞后)",
        r"滞后\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return max(0, int(float(match.group(1))))
            except (TypeError, ValueError):
                continue
    return int(default)
