"""LLM planner agent for OceanHarness task-graph decisions."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import yaml

from packages.llm_gateway.config import DEFAULT_OPENAI_MODEL, load_config_value, load_model_name
from packages.llm_gateway.openai_compatible_client import OpenAICompatibleClientAdapter

PLANNER_SELECTOR_MODEL_ENV_VAR = "PLANNER_SELECTOR_MODEL"


class PlannerAgent:
    """Select skills and resolve task-graph parameters from query/context.

    The agent speaks Markdown for readability, but the executable contract is a
    fenced YAML/JSON block embedded in that Markdown.
    """

    DEFAULT_MODEL = load_model_name("PLANNER_MODEL", default=DEFAULT_OPENAI_MODEL)

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        selector_model: Optional[str] = None,
        client: Optional[Any] = None,
        trust_env: bool = False,
        request_retries: int = 2,
    ) -> None:
        self.model = model
        self.selector_model = selector_model or _load_selector_model(model=model, base_url=base_url)
        self._adapter = OpenAICompatibleClientAdapter(
            api_key=api_key,
            base_url=base_url,
            model=model,
            client=client,
            trust_env=trust_env,
            request_retries=request_retries,
        )
        self._selector_adapter = OpenAICompatibleClientAdapter(
            api_key=api_key,
            base_url=base_url,
            model=self.selector_model,
            client=client,
            trust_env=trust_env,
            request_retries=request_retries,
        )

    def plan_harness_task_graph(
        self,
        *,
        user_request: str,
        dataset: Mapping[str, Any],
        frontend_extracted_params: Mapping[str, Any],
        workspace_context: Mapping[str, Any],
        conversation_memory: Mapping[str, Any],
        skill_headers: list[Mapping[str, Any]],
        skill_workflows: Optional[Mapping[str, Mapping[str, Any]]] = None,
        skill_workflow_loader: Optional[Callable[[Iterable[str]], Mapping[str, Mapping[str, Any]]]] = None,
        skill_contract_loader: Optional[Callable[[Iterable[str]], Mapping[str, Mapping[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        started_at = time.perf_counter()
        selector_started_at = time.perf_counter()
        skill_selection = self.select_harness_skills(
            user_request=user_request,
            skill_headers=skill_headers,
        )
        selector_elapsed_s = round(time.perf_counter() - selector_started_at, 3)
        selected_skill_ids = _selected_skill_ids(skill_selection)
        route = str(skill_selection.get("route") or "").strip()
        if route == "skill_workflow" and selected_skill_ids:
            workflow_load_started_at = time.perf_counter()
            if skill_workflow_loader is not None:
                selected_skill_workflows = dict(skill_workflow_loader(selected_skill_ids))
            else:
                selected_skill_workflows = _select_skill_workflows(skill_workflows or {}, selected_skill_ids)
            workflow_load_elapsed_s = round(time.perf_counter() - workflow_load_started_at, 3)
            task_graph_started_at = time.perf_counter()
            decision = self.plan_selected_harness_task_graph(
                user_request=user_request,
                dataset=dataset,
                frontend_extracted_params=frontend_extracted_params,
                workspace_context=workspace_context,
                conversation_memory=conversation_memory,
                skill_selection=skill_selection,
                selected_skill_workflows=selected_skill_workflows,
            )
            task_graph_elapsed_s = round(time.perf_counter() - task_graph_started_at, 3)
        else:
            workflow_load_started_at = time.perf_counter()
            if skill_workflow_loader is not None:
                selected_skill_workflows = dict(skill_workflow_loader(selected_skill_ids))
            else:
                selected_skill_workflows = _select_skill_workflows(skill_workflows or {}, selected_skill_ids)
            workflow_load_elapsed_s = round(time.perf_counter() - workflow_load_started_at, 3)
            task_graph_started_at = time.perf_counter()
            decision = self.plan_selected_harness_task_graph(
                user_request=user_request,
                dataset=dataset,
                frontend_extracted_params=frontend_extracted_params,
                workspace_context=workspace_context,
                conversation_memory=conversation_memory,
                skill_selection=skill_selection,
                selected_skill_workflows=selected_skill_workflows,
            )
            task_graph_elapsed_s = round(time.perf_counter() - task_graph_started_at, 3)
        if skill_selection.get("route") and not decision.get("route"):
            decision["route"] = skill_selection.get("route")
        if selected_skill_ids and not decision.get("selected_skill_id"):
            decision["selected_skill_id"] = selected_skill_ids[0]
        if selected_skill_ids and not decision.get("selected_skill_ids"):
            decision["selected_skill_ids"] = selected_skill_ids
        _copy_policy_making_intent_from_selection(decision, skill_selection)
        decision["planner_skill_selection"] = skill_selection
        decision["planner_agent_timings"] = {
            "planner.selector": selector_elapsed_s,
            "planner.workflow_load": workflow_load_elapsed_s,
            "planner.task_graph": task_graph_elapsed_s,
            "planner.semantic_plan": task_graph_elapsed_s if route == "skill_workflow" and selected_skill_ids else 0.0,
            "planner.total": round(time.perf_counter() - started_at, 3),
        }
        decision["planner_agent_prompt_sizes"] = {
            "skill_head_count": len(skill_headers),
            "skill_heads_chars": len(json.dumps(skill_headers, ensure_ascii=False)),
            "selected_skill_count": len(selected_skill_ids),
            "selected_workflows_chars": len(json.dumps(selected_skill_workflows, ensure_ascii=False)),
            "selected_contracts_chars": 0,
            "dataset_chars": len(json.dumps(dict(dataset), ensure_ascii=False, default=str)),
            "conversation_memory_chars": len(json.dumps(dict(conversation_memory), ensure_ascii=False, default=str)),
        }
        return decision

    def select_harness_skills(
        self,
        *,
        user_request: str,
        skill_headers: list[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "user_request": user_request,
            "skill_heads": skill_headers,
            "required_contract": {
                "status": "ready",
                "route": (
                    "dataset_info | skill_workflow | pair_lag_relationship | hypoxia_driver | "
                    "generic_timeseries | generated_code | generic_map"
                ),
                "selected_skill_id": "primary skill_id from skill_heads when route is skill_workflow, otherwise null",
                "selected_skill_ids": "at most 3 skill ids needed for this request",
                "policy_making_intent": (
                    "true only when the user asks for policy, management, governance, regulation, "
                    "economic, decision-support, action-plan, or recommendation advice; false for "
                    "scientific mechanism/control/diagnosis/explanation requests"
                ),
                "policy_making_reason": "short reason for the policy_making_intent value",
                "reason": "short reason",
            },
        }
        system = (
            "You are the OceanHarness skill-selection pass.\n"
            "You receive only skill heads. Each head contains skill_id, description, input_intent, "
            "output_intent, avoid_when, and composes_with.\n"
            "Choose the route and the smallest set of skill ids needed for the user request. "
            "Do not use dataset metadata, frontend context, memory, or resolved time/space/depth parameters in this pass. "
            "Do not generate a task graph.\n"
            "Match the user's requested output shape first using output_intent, then match the required source "
            "variables/diagnostic/event inputs using input_intent. Respect avoid_when: do not select a skill when "
            "the request falls under one of its avoid_when cases unless no other head can satisfy the request.\n"
            "Classify policy_making_intent separately from skill choice. Set it true only for requests that ask "
            "for policy, management, governance, regulation, economic, decision-support, action-plan, or "
            "recommendation advice. Set it false for scientific diagnosis, mechanism ranking, evidence grading, "
            "causal interpretation, or phrases like 'stratification control' / 'controlled by advection', where "
            "control means a physical driver rather than policymaking.\n"
            "If a skill head explicitly names the requested diagnostic or analysis product, select that skill before "
            "generic derived-field skills or generated_code.\n"
            "If the request asks for a custom index, custom metric, ad-hoc diagnostic, or formula that is not named by "
            "a skill head, choose route generated_code instead of approximating it with a generic mean or time series. "
            "Return exactly one compact JSON object and no Markdown. "
            "Do not include reasoning, analysis prose, or text outside the JSON object."
        )
        user_content = (
            "Select the skill route. Return only this JSON shape with concrete values:\n"
            "{\"status\":\"ready\",\"route\":\"skill_workflow\",\"selected_skill_id\":null,"
            "\"selected_skill_ids\":[],\"policy_making_intent\":false,"
            "\"policy_making_reason\":\"\",\"reason\":\"\"}\n\n"
            f"Input packet:\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
        )
        client = self._selector_adapter.get_client()

        def create_selector_response(*, max_tokens: int, retry: bool = False) -> Any:
            retry_system = (
                system
                + "\nThis is a retry after an empty, truncated, or unparsable final response. Emit one valid JSON "
                "object immediately; do not spend tokens on reasoning."
                if retry
                else system
            )
            return self._selector_adapter.create_message(
                client=client,
                max_tokens=max_tokens,
                temperature=0.0,
                system=retry_system,
                messages=[{"role": "user", "content": user_content}],
                request_name="ocean_harness_skill_selector_retry" if retry else "ocean_harness_skill_selector",
                json_response=True,
            )

        response = create_selector_response(max_tokens=2400)
        try:
            text = self._selector_adapter.extract_response_text(response)
        except ValueError as exc:
            if not _looks_like_empty_final_text_error(exc):
                raise ValueError(
                    "Planner skill selector failed to return a final routing contract: "
                    f"{exc}"
                ) from exc
            retry_response = create_selector_response(max_tokens=6000, retry=True)
            try:
                text = self._selector_adapter.extract_response_text(retry_response)
            except ValueError as retry_exc:
                raise ValueError(
                    "Planner skill selector failed to return a final routing contract after retry: "
                    f"{retry_exc}"
                ) from retry_exc
        try:
            selection = self.parse_markdown_contract(text)
        except ValueError as exc:
            retry_response = create_selector_response(max_tokens=6000, retry=True)
            try:
                retry_text = self._selector_adapter.extract_response_text(retry_response)
                selection = self.parse_markdown_contract(retry_text)
            except ValueError as retry_exc:
                raise ValueError(
                    "Planner skill selector failed to return a parseable routing contract after retry: "
                    f"{retry_exc}"
                ) from exc
        if not isinstance(selection, dict):
            raise ValueError("Planner skill selector returned a non-object payload.")
        selection["selector_model"] = self.selector_model
        return selection

    def plan_selected_harness_task_graph(
        self,
        *,
        user_request: str,
        dataset: Mapping[str, Any],
        frontend_extracted_params: Mapping[str, Any],
        workspace_context: Mapping[str, Any],
        conversation_memory: Mapping[str, Any],
        skill_selection: Mapping[str, Any],
        selected_skill_workflows: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        clarification_disabled = _benchmark_disables_clarification(
            frontend_extracted_params=frontend_extracted_params,
            workspace_context=workspace_context,
            conversation_memory=conversation_memory,
        )
        payload = {
            "user_request": user_request,
            "dataset": dict(dataset),
            "frontend_extracted_params": dict(frontend_extracted_params),
            "workspace_context": dict(workspace_context),
            "conversation_memory": dict(conversation_memory),
            "skill_selection": dict(skill_selection),
            "selected_skill_workflows": dict(selected_skill_workflows),
            "benchmark_constraints": {
                "disable_clarification": clarification_disabled,
            },
            "required_contract": {
                "status": "ready" if clarification_disabled else "ready | clarification_needed",
                "route": (
                    "dataset_info | skill_workflow | pair_lag_relationship | hypoxia_driver | "
                    "generic_timeseries | generated_code | generic_map"
                ),
                "selected_skill_id": "skill id from skill_headers when route is skill_workflow, otherwise null",
                "resolved_scope": {
                    "dataset": "current dataset id",
                    "variable": "primary variable",
                    "variables": ["variables when multiple fields are needed"],
                    "lon_range": ["west", "east"],
                    "lat_range": ["south", "north"],
                    "time_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
                    "vertical_mode": "surface | bottom | fixed_depth | depth_range | unspecified",
                    "depth_value": None,
                    "depth_range": None,
                },
                "parameters": "extra concrete tool parameters such as thresholds, min_duration_days, max_lag",
                "workflow_code": (
                    "complete assignment-style Python workflow DSL for skill_workflow routes; "
                    "one artifact assignment per registered tool call"
                ),
                "final_artifacts": ["artifact ids produced by workflow_todo or workflow_code that should be treated as the answer"],
                "workflow_todo": {"shared_scope": {}, "todo": []},
                "task_graph": {
                    "final_artifacts": ["generated-code artifact ids when route is generated_code"],
                    "nodes": [
                        {
                            "id": "stable node id",
                            "tool": "generated_python_analysis or supporting load_dataset nodes for generated_code only",
                            "params": "fully resolved concrete params, including ranges",
                            "save_as": "artifact id",
                        }
                    ]
                },
                "missing_fields": ["required inputs that cannot be inferred"],
                "clarification_question": "English question when status is clarification_needed",
                "reason": "short reason",
            },
        }
        system = (
            "You are the OceanHarness planner agent.\n"
            "A previous selection pass already chose the route and selected skill ids from short skill heads.\n"
            "Using the query, selected skill workflow/manual text, dataset metadata, frontend context, and memory, "
            "produce a complete executable workflow_code for skill_workflow routes. workflow_code is a safe Python-like "
            "DSL, not arbitrary Python: it may contain only simple variable assignments and artifact assignments of the "
            "form artifact = tool_name(keyword=value, ...). Do not include imports, functions, classes, loops, if/else, "
            "comprehensions, file I/O, shell commands, comments-only pseudo-code, or calls to tools not exposed or named "
            "by the selected skill workflow/manual text. Use the selected skill contents as the tool-chain source of "
            "truth; do not rely on an external tool registry.\n"
            "For skill_workflow routes, workflow_code is required and workflow_todo should stay empty. Leave "
            "task_graph.nodes empty because workflow_code is the source of truth. final_artifacts must name one or more "
            "artifacts assigned by workflow_code that answer the user request.\n"
            "Every artifact reference such as u_field.data must point to an artifact assigned by an earlier tool call. "
            "Do not repeat equivalent load_dataset calls; load once and reference that artifact from downstream tools. "
            "Tool calls must use only parameters shown in the selected skill workflow/manual examples or clearly implied "
            "by the skill's named tools. "
            "For example, load_dataset receives lon_range/lat_range/vertical_mode/depth_value/depth_range, while "
            "downstream map or reduction tools receive their own reduction parameters; "
            "do not pass lon_range, lat_range, vertical_mode, or depth_value to compute_spatial_field because the "
            "loaded data has already been subset vertically and horizontally.\n"
            "OceanMind depth ranges use non-negative meters positive downward. Interpret 'upper 0-750 m' as "
            "depth_range: [0, 750], 'surface' as [0, 0], and 'at 200 m' as [200, 200]. Do not use negative depth "
            "values unless the dataset metadata explicitly says its depth coordinate is negative.\n"
            "For bottom, near-bottom, seafloor, or benthic requests, write vertical_mode='bottom' with "
            "depth_value=None and depth_range=None in workflow_code. Never encode bottom water as a fixed deepest "
            "coordinate, 8000, -8000, or depth_range=[8000, 8000] / [-8000, -8000].\n"
            "When a selected workflow includes planner_parameters, fill those shared variables directly in "
            "workflow_code before the tool calls. Skill workflow refs listed as unbound or symbolic are template "
            "choices, not executable artifacts; bind them to a concrete earlier artifact or skip that optional branch. "
            "Do not copy optional mask/isobath/condition branches unless the query or frontend context supplies the "
            "mask inputs needed to build them.\n"
            "If the query requests a polygon, isobath, bathymetry, or condition mask, create all mask-producing "
            "nodes before the diagnostic/reduction node that uses the mask. Do not place masks after the step they "
            "are meant to constrain.\n"
            "Instantiate only the branch needed for the user's requested output. For an ambiguous trend over a "
            "region, prefer a regional-mean time-series trend; include a spatial/pixel-wise trend only when the "
            "user asks for a map, spatial trend, pixel-wise trend, or gridded trend.\n"
            "Use compute_spatial_vorticity_map only for 2D map-ready relative-vorticity outputs. For time-depth, "
            "Hovmoller, profile, time series, or any result that must retain time/depth dimensions, compute "
            "relative vorticity with compute_derived_field(field_type='vorticity') and then apply the requested "
            "mask/reduction/diagram tool.\n"
            "When the selected workflow/manual exposes a tool that directly computes the requested diagnostic or output, "
            "instantiate that tool in workflow_code. Use generated_python_analysis only when no selected workflow/tool covers the "
            "requested computation.\n"
            "For ocean_transport_analysis requests asking for a time-depth diagram, Hovmoller, or normal volume flux "
            "across a transect, instantiate compute_transect_normal_flux_hovmoller. Do not use generated_python_analysis "
            "for that branch because the selected transport workflow already exposes the exact tool.\n"
            "If the user asks for a custom index, custom metric, ad-hoc diagnostic, or formula and the selected "
            "workflow does not contain a tool that computes that exact quantity, use route generated_code. Build a "
            "task graph with load_dataset first, then a generated_python_analysis node whose params include "
            "input_refs: {field: $ref:<loaded_artifact>.data}. Do not put executable code in the planner contract. "
            "Do not rename a simple regional mean as the custom index. Do not ask the user to provide Python code "
            "and do not put code in missing_fields. You may include planner_analysis_design or analysis_design as a "
            "non-authoritative hint, but CodeAgent will independently decide the generated step's input/output "
            "contract from the query and resolved input artifact schemas before writing code. For a generic "
            "generated-code analysis with no requested depth or layer, use surface data by default.\n"
            "Return exactly one compact JSON object and no Markdown. Do not include reasoning, analysis prose, "
            "code fences, or text outside the JSON object."
        )
        if clarification_disabled:
            system += (
                "\nBenchmark mode is active and clarification is disabled. Do not return "
                "status='clarification_needed'. Every benchmark query is complete enough to execute; infer missing "
                "scope or optional parameters from the query, dataset context, selected skill defaults, or safe "
                "ocean-domain defaults. If no selected workflow can express the task, choose a ready generated_code "
                "route instead of asking a question."
            )
        user_content = (
            "Produce the planner decision and complete workflow_code. Return only this JSON shape with concrete values:\n"
            "{\"status\":\"ready\",\"route\":\"skill_workflow\",\"selected_skill_id\":null,"
            "\"resolved_scope\":{},\"parameters\":{},"
            "\"workflow_todo\":{\"shared_scope\":{},\"todo\":[]},\"workflow_code\":\"\","
            "\"final_artifacts\":[],\"task_graph\":{\"final_artifacts\":[],\"nodes\":[]},"
            "\"missing_fields\":[],\"clarification_question\":\"\",\"reason\":\"\"}\n\n"
            f"Input packet:\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
        )
        client = self._adapter.get_client()

        def create_planner_response(*, max_tokens: int, retry: bool = False) -> Any:
            retry_system = (
                system
                + "\nThis is a retry after an empty, truncated, or unparsable final response. Emit one valid JSON "
                "object immediately; do not spend tokens on reasoning."
                if retry
                else system
            )
            return self._adapter.create_message(
                client=client,
                max_tokens=max_tokens,
                temperature=0.0,
                system=retry_system,
                messages=[{"role": "user", "content": user_content}],
                request_name="ocean_harness_planner_agent_retry" if retry else "ocean_harness_planner_agent",
                json_response=True,
            )

        response = create_planner_response(max_tokens=6000)
        try:
            text = self._adapter.extract_response_text(response)
        except ValueError as exc:
            if not _looks_like_empty_final_text_error(exc):
                raise ValueError(
                    "Planner task-graph generator failed to return a final workflow contract: "
                    f"{exc}"
                ) from exc
            retry_response = create_planner_response(max_tokens=10000, retry=True)
            try:
                text = self._adapter.extract_response_text(retry_response)
            except ValueError as retry_exc:
                raise ValueError(
                    "Planner task-graph generator failed to return a final workflow contract after retry: "
                    f"{retry_exc}"
                ) from retry_exc
        try:
            return self.parse_markdown_contract(text)
        except ValueError as exc:
            retry_response = create_planner_response(max_tokens=10000, retry=True)
            try:
                retry_text = self._adapter.extract_response_text(retry_response)
                return self.parse_markdown_contract(retry_text)
            except ValueError as retry_exc:
                raise ValueError(
                    "Planner task-graph generator failed to return a parseable workflow contract after retry: "
                    f"{retry_exc}"
                ) from exc

    def plan_selected_skill_semantics(
        self,
        *,
        user_request: str,
        dataset: Mapping[str, Any],
        frontend_extracted_params: Mapping[str, Any],
        workspace_context: Mapping[str, Any],
        conversation_memory: Mapping[str, Any],
        skill_selection: Mapping[str, Any],
        selected_skill_contracts: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        selected_skill_ids = _selected_skill_ids(skill_selection)
        clarification_disabled = _benchmark_disables_clarification(
            frontend_extracted_params=frontend_extracted_params,
            workspace_context=workspace_context,
            conversation_memory=conversation_memory,
        )
        payload = {
            "user_request": user_request,
            "dataset": dict(dataset),
            "frontend_extracted_params": dict(frontend_extracted_params),
            "workspace_context": dict(workspace_context),
            "conversation_memory": dict(conversation_memory),
            "skill_selection": dict(skill_selection),
            "selected_skill_contracts": dict(selected_skill_contracts),
            "benchmark_constraints": {
                "disable_clarification": clarification_disabled,
            },
            "required_contract": {
                "status": "ready | needs_reselect" if clarification_disabled else "ready | clarification_needed | needs_reselect",
                "route": "skill_workflow",
                "selected_skill_id": "primary selected skill id",
                "selected_skill_ids": "exact selected_skill_ids from skill_selection; do not add skills",
                "semantic_plan": {
                    "shared_scope": {
                        "dataset": "current dataset id when known",
                        "lon_range": ["west", "east"],
                        "lat_range": ["south", "north"],
                        "time_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
                    },
                    "skills": [
                        {
                            "skill_id": "one of selected_skill_ids",
                            "slots": "semantic slots for this skill only; no tool names or artifact ids",
                        }
                    ],
                    "composition": {
                        "type": "single | parallel | pipeline | join | rank | comparison",
                        "purpose": "short semantic purpose",
                    },
                },
                "missing_fields": ["semantic slots that cannot be inferred"],
                "clarification_question": "English question when status is clarification_needed",
                "reason": "short reason",
            },
        }
        system = (
            "You are the OceanHarness semantic planner.\n"
            "A previous selector already chose the skills. Your job is only to fill semantic slots and describe "
            "how the selected skills compose. You receive selected skill contracts, not workflow templates.\n"
            "Do not output tool names, task graph nodes, artifact ids, Python code, workflow_todo, workflow_code, "
            "or final_artifacts. Do not add, remove, or replace selected skills. If the selected skills cannot "
            "satisfy the request, return status='needs_reselect' with a reason.\n"
            "Put common region/time/dataset values in semantic_plan.shared_scope. Put skill-specific values in "
            "semantic_plan.skills[].slots. Normalize common ocean aliases when clear: SST means variable temp with "
            "vertical_mode surface and depth_range [0, 0]; bottom oxygen means variable oxygen with vertical_mode "
            "bottom; upper-N m means depth_range [0, N].\n"
            "For one skill use composition.type='single'. For multiple independent evidence branches use 'parallel' "
            "or 'join'. For one skill consuming another skill's output use 'pipeline'. For mechanism ordering use "
            "'rank'. For condition/period/region contrasts use 'comparison'.\n"
            "Return exactly one compact JSON object and no Markdown."
        )
        if clarification_disabled:
            system += (
                "\nBenchmark mode is active and clarification is disabled. Do not return "
                "status='clarification_needed'. Fill semantic slots with concrete inferred values from the query, "
                "dataset context, selected skill defaults, or safe ocean-domain defaults. If the selected skill set "
                "is genuinely unusable, return status='needs_reselect' instead of asking a question."
            )
        user_content = (
            "Produce only this semantic planning contract with concrete values:\n"
            "{\"status\":\"ready\",\"route\":\"skill_workflow\",\"selected_skill_id\":null,"
            "\"selected_skill_ids\":[],\"semantic_plan\":{\"shared_scope\":{},\"skills\":[],"
            "\"composition\":{\"type\":\"single\",\"purpose\":\"\"}},"
            "\"missing_fields\":[],\"clarification_question\":\"\",\"reason\":\"\"}\n\n"
            f"Input packet:\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}\n```"
        )
        client = self._adapter.get_client()

        def create_semantic_response(*, max_tokens: int, retry: bool = False) -> Any:
            retry_system = (
                system
                + "\nThis is a retry after an empty, truncated, or unparsable final response. Emit one valid JSON "
                "object immediately; do not spend tokens on reasoning."
                if retry
                else system
            )
            return self._adapter.create_message(
                client=client,
                max_tokens=max_tokens,
                temperature=0.0,
                system=retry_system,
                messages=[{"role": "user", "content": user_content}],
                request_name="ocean_harness_semantic_planner_retry" if retry else "ocean_harness_semantic_planner",
                json_response=True,
            )

        response = create_semantic_response(max_tokens=3600)
        try:
            text = self._adapter.extract_response_text(response)
        except ValueError as exc:
            if not _looks_like_empty_final_text_error(exc):
                raise ValueError(
                    "Planner semantic generator failed to return a final contract: "
                    f"{exc}"
                ) from exc
            retry_response = create_semantic_response(max_tokens=7200, retry=True)
            try:
                text = self._adapter.extract_response_text(retry_response)
            except ValueError as retry_exc:
                raise ValueError(
                    "Planner semantic generator failed to return a final contract after retry: "
                    f"{retry_exc}"
                ) from retry_exc
        try:
            decision = self.parse_markdown_contract(text)
        except ValueError as exc:
            retry_response = create_semantic_response(max_tokens=7200, retry=True)
            try:
                retry_text = self._adapter.extract_response_text(retry_response)
                decision = self.parse_markdown_contract(retry_text)
            except ValueError as retry_exc:
                raise ValueError(
                    "Planner semantic generator failed to return a parseable contract after retry: "
                    f"{retry_exc}"
                ) from exc
        if not isinstance(decision, dict):
            raise ValueError("Planner semantic generator returned a non-object payload.")
        return _semantic_skill_decision(
            decision,
            skill_selection=skill_selection,
            selected_skill_ids=selected_skill_ids,
        )

    def replan_harness_task_graph(
        self,
        *,
        user_request: str,
        previous_plan: Mapping[str, Any],
        failed_event: Mapping[str, Any],
        dataset: Mapping[str, Any],
        frontend_extracted_params: Mapping[str, Any],
        workspace_context: Mapping[str, Any],
        conversation_memory: Mapping[str, Any],
        skill_headers: list[Mapping[str, Any]],
        skill_workflows: Optional[Mapping[str, Mapping[str, Any]]] = None,
        skill_workflow_loader: Optional[Callable[[Iterable[str]], Mapping[str, Mapping[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        feedback = {
            "previous_plan": dict(previous_plan),
            "failed_event": dict(failed_event),
            "instruction": (
                "Regenerate the planner task graph after a generated-code/runtime failure. "
                "Keep the user's scientific intent unless the error proves it impossible. "
                "Change only the plan/code strategy needed to avoid the failure."
            ),
        }
        return self.plan_harness_task_graph(
            user_request=(
                f"{user_request}\n\n"
                f"Runtime feedback for replanning:\n{json.dumps(feedback, ensure_ascii=False, indent=2)}"
            ),
            dataset=dataset,
            frontend_extracted_params=frontend_extracted_params,
            workspace_context=workspace_context,
            conversation_memory=conversation_memory,
            skill_headers=skill_headers,
            skill_workflows=skill_workflows,
            skill_workflow_loader=skill_workflow_loader,
        )

    @staticmethod
    def parse_markdown_contract(text: str) -> Dict[str, Any]:
        parse_errors: list[str] = []
        for language, content in _iter_contract_candidates(text):
            parsed = _parse_contract_candidate(language, content, parse_errors=parse_errors)
            if parsed is not None:
                return parsed

        detail = f" Last parse error: {parse_errors[-1]}" if parse_errors else ""
        raise ValueError(
            "Planner agent response did not include a parseable YAML/JSON task graph contract."
            f"{detail}"
        )


def _selected_skill_ids(selection: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    raw_ids = selection.get("selected_skill_ids")
    if isinstance(raw_ids, list):
        values.extend(_clean_skill_id(item) for item in raw_ids)
    primary = selection.get("selected_skill_id")
    cleaned_primary = _clean_skill_id(primary)
    if cleaned_primary:
        values.insert(0, cleaned_primary)

    deduped: list[str] = []
    seen: set[str] = set()
    for skill_id in values:
        if not skill_id or skill_id in seen:
            continue
        deduped.append(skill_id)
        seen.add(skill_id)
        if len(deduped) >= 3:
            break
    return deduped


def _backend_template_skill_decision(
    skill_selection: Mapping[str, Any],
    selected_skill_ids: list[str],
) -> Dict[str, Any]:
    decision = {
        "status": "ready",
        "route": "skill_workflow",
        "selected_skill_id": selected_skill_ids[0],
        "selected_skill_ids": selected_skill_ids,
        "resolved_scope": {},
        "parameters": {},
        "workflow_code": "",
        "final_artifacts": [],
        "task_graph": {"final_artifacts": [], "nodes": []},
        "missing_fields": [],
        "clarification_question": "",
        "reason": str(skill_selection.get("reason") or "Selected skill workflow will be compiled from the backend template."),
        "planning_mode": "backend_template",
    }
    _copy_policy_making_intent_from_selection(decision, skill_selection)
    return decision


def _semantic_skill_decision(
    raw_decision: Mapping[str, Any],
    *,
    skill_selection: Mapping[str, Any],
    selected_skill_ids: list[str],
) -> Dict[str, Any]:
    status = str(raw_decision.get("status") or "ready").strip()
    if status not in {"ready", "clarification_needed", "needs_reselect"}:
        status = "ready"
    semantic_plan = _normalize_semantic_plan(
        raw_decision.get("semantic_plan") if isinstance(raw_decision.get("semantic_plan"), Mapping) else raw_decision,
        selected_skill_ids=selected_skill_ids,
    )
    resolved_scope = _semantic_plan_resolved_scope(semantic_plan)
    parameters = _semantic_plan_parameters(semantic_plan)
    if isinstance(raw_decision.get("parameters"), Mapping):
        parameters.update(dict(raw_decision["parameters"]))
    decision = {
        "status": "clarification_needed" if status == "clarification_needed" else "ready",
        "route": "skill_workflow" if status != "needs_reselect" else "generated_code",
        "selected_skill_id": selected_skill_ids[0] if selected_skill_ids else None,
        "selected_skill_ids": list(selected_skill_ids),
        "semantic_plan": semantic_plan,
        "resolved_scope": resolved_scope,
        "parameters": parameters,
        "workflow_code": "",
        "final_artifacts": [],
        "task_graph": {"final_artifacts": [], "nodes": []},
        "missing_fields": _string_list(raw_decision.get("missing_fields")),
        "clarification_question": str(raw_decision.get("clarification_question") or ""),
        "reason": str(raw_decision.get("reason") or skill_selection.get("reason") or "Planned selected skills semantically."),
        "planning_mode": "semantic_slots",
    }
    if status == "needs_reselect":
        decision["needs_reselect"] = True
        decision["route"] = "generated_code"
    _copy_policy_making_intent_from_selection(decision, skill_selection)
    return decision


def _normalize_semantic_plan(value: Any, *, selected_skill_ids: list[str]) -> Dict[str, Any]:
    plan = dict(value) if isinstance(value, Mapping) else {}
    shared_scope = plan.get("shared_scope") if isinstance(plan.get("shared_scope"), Mapping) else {}
    normalized_shared = {
        str(key): val
        for key, val in dict(shared_scope).items()
        if str(key).strip() and val is not None and val != ""
    }
    selected = set(selected_skill_ids)
    skills: list[Dict[str, Any]] = []
    raw_skills = plan.get("skills") if isinstance(plan.get("skills"), list) else []
    for item in raw_skills:
        if not isinstance(item, Mapping):
            continue
        skill_id = _clean_skill_id(item.get("skill_id"))
        if not skill_id or skill_id not in selected:
            continue
        slots = item.get("slots") if isinstance(item.get("slots"), Mapping) else {}
        skills.append(
            {
                "skill_id": skill_id,
                "slots": {
                    str(key): val
                    for key, val in dict(slots).items()
                    if str(key).strip() and val is not None and val != ""
                },
            }
        )
    seen = {item["skill_id"] for item in skills}
    for skill_id in selected_skill_ids:
        if skill_id not in seen:
            skills.append({"skill_id": skill_id, "slots": {}})
    composition = plan.get("composition") if isinstance(plan.get("composition"), Mapping) else {}
    composition_type = str(composition.get("type") or ("single" if len(selected_skill_ids) <= 1 else "join")).strip()
    valid_compositions = {"single", "parallel", "pipeline", "join", "rank", "comparison"}
    if composition_type not in valid_compositions:
        composition_type = "single" if len(selected_skill_ids) <= 1 else "join"
    normalized_composition = {
        str(key): val
        for key, val in dict(composition).items()
        if str(key).strip() and val is not None and val != ""
    }
    normalized_composition["type"] = composition_type
    normalized_composition.setdefault("purpose", "")
    return {
        "shared_scope": normalized_shared,
        "skills": skills,
        "composition": normalized_composition,
    }


def _semantic_plan_resolved_scope(semantic_plan: Mapping[str, Any]) -> Dict[str, Any]:
    shared_scope = semantic_plan.get("shared_scope") if isinstance(semantic_plan.get("shared_scope"), Mapping) else {}
    scope = {
        str(key): val
        for key, val in dict(shared_scope).items()
        if str(key).strip() and val is not None and val != ""
    }
    first_slots: Mapping[str, Any] = {}
    skills = semantic_plan.get("skills")
    if isinstance(skills, list):
        for item in skills:
            if isinstance(item, Mapping) and isinstance(item.get("slots"), Mapping):
                first_slots = item["slots"]
                break
    for key in (
        "variable",
        "variables",
        "vertical_mode",
        "depth_value",
        "depth_range",
        "season_filter",
    ):
        if key not in scope and first_slots.get(key) is not None:
            scope[key] = first_slots.get(key)
    if "variable" not in scope and isinstance(scope.get("variables"), list) and scope["variables"]:
        scope["variable"] = scope["variables"][0]
    if "variables" not in scope and isinstance(scope.get("variable"), str) and scope["variable"].strip():
        scope["variables"] = [scope["variable"].strip()]
    return scope


def _semantic_plan_parameters(semantic_plan: Mapping[str, Any]) -> Dict[str, Any]:
    skill_slots: Dict[str, Dict[str, Any]] = {}
    skills = semantic_plan.get("skills")
    if isinstance(skills, list):
        for item in skills:
            if not isinstance(item, Mapping):
                continue
            skill_id = _clean_skill_id(item.get("skill_id"))
            slots = item.get("slots") if isinstance(item.get("slots"), Mapping) else {}
            if skill_id:
                skill_slots[skill_id] = dict(slots)
    parameters: Dict[str, Any] = {
        "skill_slots": skill_slots,
        "composition": dict(semantic_plan.get("composition") or {})
        if isinstance(semantic_plan.get("composition"), Mapping)
        else {},
    }
    first_slots = next(iter(skill_slots.values()), {})
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
    return parameters


def _skill_contracts_from_workflow_briefs(
    skill_workflows: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    contracts: Dict[str, Mapping[str, Any]] = {}
    for skill_id, detail in skill_workflows.items():
        workflow = detail.get("workflow") if isinstance(detail, Mapping) else None
        if not isinstance(workflow, Mapping):
            continue
        contracts[skill_id] = {
            "skill_id": skill_id,
            "workflow_id": workflow.get("workflow_id"),
            "event_type": workflow.get("event_type"),
            "required_slots": list((workflow.get("required_inputs") or {}).keys())
            if isinstance(workflow.get("required_inputs"), Mapping)
            else [],
            "optional_slots": list((workflow.get("planner_parameters") or {}).keys())
            if isinstance(workflow.get("planner_parameters"), Mapping)
            else [],
            "defaults": {},
            "outputs": workflow.get("output_policy") if isinstance(workflow.get("output_policy"), Mapping) else {},
            "validation_rules": workflow.get("validation_rules") if isinstance(workflow.get("validation_rules"), list) else [],
        }
    return contracts


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _copy_policy_making_intent_from_selection(
    decision: Dict[str, Any],
    skill_selection: Mapping[str, Any],
) -> None:
    value = _coerce_optional_bool(skill_selection.get("policy_making_intent"))
    if value is not None:
        decision["policy_making_intent"] = value
    if isinstance(skill_selection.get("policy_making_reason"), str):
        reason = str(skill_selection.get("policy_making_reason") or "").strip()
        if reason:
            decision["policy_making_reason"] = reason


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    return None


def _looks_like_empty_final_text_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "does not contain final text content" in text
        or "does not contain text content" in text
        or "reasoning_content" in text
    )


def _clean_skill_id(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "none", "null", "n/a", "na"} else text


def _select_skill_workflows(
    skill_workflows: Mapping[str, Mapping[str, Any]],
    selected_skill_ids: list[str],
) -> Dict[str, Mapping[str, Any]]:
    selected: Dict[str, Mapping[str, Any]] = {}
    for skill_id in selected_skill_ids:
        detail = skill_workflows.get(skill_id)
        if isinstance(detail, Mapping):
            selected[skill_id] = detail
    return selected


def _load_selector_model(*, model: str, base_url: Optional[str]) -> str:
    configured = (load_config_value(PLANNER_SELECTOR_MODEL_ENV_VAR) or "").strip()
    if configured and configured.lower() not in {"none", "null", "default"}:
        return configured
    return _default_selector_model_for(model=model, base_url=base_url)


def _default_selector_model_for(*, model: str, base_url: Optional[str]) -> str:
    hint = f"{model or ''} {base_url or ''}".lower()
    if "deepseek" in hint:
        return "deepseek-v4-flash"
    if any(token in hint for token in ("qwen", "dashscope", "aliyun", "alibaba")):
        return model
    if any(token in hint for token in ("openai", "gpt-4", "gpt-5")):
        return DEFAULT_OPENAI_MODEL
    return model


def _benchmark_disables_clarification(
    *,
    frontend_extracted_params: Mapping[str, Any],
    workspace_context: Mapping[str, Any],
    conversation_memory: Mapping[str, Any],
) -> bool:
    for payload in (frontend_extracted_params, workspace_context, conversation_memory):
        if not isinstance(payload, Mapping):
            continue
        benchmark_policy = payload.get("benchmark_policy")
        if isinstance(benchmark_policy, Mapping) and benchmark_policy.get("disable_clarification") is True:
            return True
    return False


_CONTRACT_TOP_LEVEL_KEYS = {
    "status",
    "route",
    "selected_skill_id",
    "selected_skill_ids",
    "resolved_scope",
    "parameters",
    "workflow_code",
    "final_artifacts",
    "task_graph",
    "missing_fields",
    "clarification_question",
    "reason",
}


def _iter_contract_candidates(text: str) -> Iterable[tuple[str, str]]:
    raw_text = (text or "").strip()
    if not raw_text:
        return

    seen: set[tuple[str, str]] = set()

    def emit(language: str, content: str) -> Iterable[tuple[str, str]]:
        normalized_language = "json" if language.lower() == "json" else "yaml"
        normalized_content = content.strip()
        key = (normalized_language, normalized_content)
        if not normalized_content or key in seen:
            return
        seen.add(key)
        yield key

    pattern = re.compile(r"```(?P<lang>yaml|yml|json)?\s*\n(?P<body>.*?)\n```", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(text or ""):
        lang = (match.group("lang") or "yaml").lower()
        body = match.group("body").strip()
        if not body:
            continue
        if _looks_like_contract_text(body):
            yield from emit(lang, body)

    # Common LLM failure: opens a fenced block but truncates before the closing fence.
    open_fence = re.search(r"```(?P<lang>yaml|yml|json)?\s*\n(?P<body>.*)$", raw_text, re.IGNORECASE | re.DOTALL)
    if open_fence and not raw_text.endswith("```"):
        lang = (open_fence.group("lang") or "yaml").lower()
        body = open_fence.group("body").strip()
        if _looks_like_contract_text(body):
            yield from emit(lang, body)

    json_candidate = _extract_json_like_region(raw_text)
    if json_candidate and _looks_like_contract_text(json_candidate):
        yield from emit("json", json_candidate)

    yaml_candidate = _extract_yaml_like_region(raw_text)
    if yaml_candidate and _looks_like_contract_text(yaml_candidate):
        yield from emit("yaml", yaml_candidate)

    if _looks_like_contract_text(raw_text):
        yield from emit("yaml", raw_text)


def _parse_contract_candidate(
    language: str,
    content: str,
    *,
    parse_errors: list[str],
) -> Optional[Dict[str, Any]]:
    for repaired in _contract_text_repairs(content):
        parsers = [json.loads, yaml.safe_load] if language == "json" else [yaml.safe_load, json.loads]
        for parser in parsers:
            try:
                parsed = parser(repaired)
            except Exception as exc:  # Keep trying less strict candidates.
                parse_errors.append(str(exc))
                continue
            if _is_contract_object(parsed):
                return parsed
            if isinstance(parsed, dict):
                parse_errors.append("parsed object did not contain planner contract top-level keys")
            else:
                parse_errors.append("planner contract candidate did not parse to an object")
    return None


def _contract_text_repairs(content: str) -> Iterable[str]:
    cleaned = _strip_surrounding_code_fences(content.strip())
    candidates = [
        cleaned,
        _remove_trailing_commas(cleaned),
    ]

    json_region = _extract_json_like_region(cleaned)
    if json_region:
        candidates.extend([json_region, _remove_trailing_commas(json_region)])

    yaml_region = _extract_yaml_like_region(cleaned)
    if yaml_region:
        candidates.append(yaml_region)

    for candidate in list(candidates):
        balanced = _close_unbalanced_json(candidate)
        if balanced != candidate:
            candidates.append(balanced)
            candidates.append(_remove_trailing_commas(balanced))

    yielded: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in yielded:
            continue
        yielded.add(candidate)
        yield candidate


def _looks_like_contract_text(text: str) -> bool:
    return any(re.search(rf"(^|\n)\s*{re.escape(key)}\s*:", text) for key in _CONTRACT_TOP_LEVEL_KEYS) or any(
        f'"{key}"' in text for key in _CONTRACT_TOP_LEVEL_KEYS
    )


def _is_contract_object(value: Any) -> bool:
    return isinstance(value, dict) and bool(_CONTRACT_TOP_LEVEL_KEYS.intersection(value.keys()))


def _strip_surrounding_code_fences(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:yaml|yml|json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _remove_trailing_commas(text: str) -> str:
    repaired = re.sub(r",(\s*[}\]])", r"\1", text)
    return re.sub(r",\s*$", "", repaired.strip())


def _extract_json_like_region(text: str) -> Optional[str]:
    start = min((idx for idx in (text.find("{"), text.find("[")) if idx >= 0), default=-1)
    if start < 0:
        return None
    snippet = text[start:].strip()
    balanced = _take_balanced_json_prefix(snippet)
    return balanced or snippet


def _take_balanced_json_prefix(text: str) -> Optional[str]:
    stack: list[str] = []
    in_string = False
    escape = False
    quote = ""
    pairs = {"{": "}", "[": "]"}
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char in pairs:
            stack.append(pairs[char])
            continue
        if char in {"}", "]"}:
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[: index + 1].strip()
    return None


def _close_unbalanced_json(text: str) -> str:
    stripped = _remove_trailing_commas(text.strip())
    if not stripped.startswith(("{", "[")):
        return text
    stack: list[str] = []
    in_string = False
    escape = False
    quote = ""
    pairs = {"{": "}", "[": "]"}
    for char in stripped:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif char in {"}", "]"}:
            if not stack or char != stack[-1]:
                return text
            stack.pop()
    if in_string or not stack:
        return stripped
    return f"{stripped}{''.join(reversed(stack))}"


def _extract_yaml_like_region(text: str) -> Optional[str]:
    lines = text.splitlines()
    start_index: Optional[int] = None
    for index, line in enumerate(lines):
        if re.match(rf"^\s*({'|'.join(re.escape(key) for key in _CONTRACT_TOP_LEVEL_KEYS)})\s*:", line):
            start_index = index
            break
    if start_index is None:
        return None
    return "\n".join(lines[start_index:]).strip()
