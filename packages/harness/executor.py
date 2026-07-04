"""Harness executor that emits the same event protocol as the legacy executor."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple

from packages.harness.artifact_store import ArtifactRecord, ArtifactStore
from packages.harness.code_agent import CodeAgent
from packages.harness.code_runner import run_code_node
from packages.harness.contracts import spec_from_xarray, validate_tool_runtime_result
from packages.harness.ir import ExecutionStrategy
from packages.harness.planner import OceanHarnessPlanner
from packages.harness.specs import ArtifactKind, FrontendType
from packages.harness.tool_contracts import get_tool_shape_contract
from packages.tool_loader.introspect import get_tools_cached
from packages.tool_loader.orchestrator import ToolOrchestrator


_HARNESS_TOOL_POOL = ThreadPoolExecutor(max_workers=2)
_HARNESS_PLANNER_POOL = ThreadPoolExecutor(max_workers=2)


def _is_generated_code_step_error(plan: Mapping[str, Any], failed_event: Mapping[str, Any]) -> bool:
    failed_step_id = str(failed_event.get("step_id") or "")
    for step in plan.get("steps", []) or []:
        if not isinstance(step, Mapping):
            continue
        step_id = str(step.get("step_id") or step.get("save_as") or "")
        if step_id != failed_step_id:
            continue
        harness_node = step.get("harness_node") if isinstance(step.get("harness_node"), Mapping) else {}
        execution = harness_node.get("execution") if isinstance(harness_node.get("execution"), Mapping) else {}
        return execution.get("strategy") == ExecutionStrategy.CODE.value
    return False


def _step_error_replan_reason(plan: Mapping[str, Any], failed_event: Mapping[str, Any]) -> str:
    if _is_generated_code_step_error(plan, failed_event):
        return "generated_code_error"
    return "runtime_step_error"


def _step_error_replan_method(planner: Optional[Any]):
    if planner is None:
        return None
    step_method = getattr(planner, "replan_after_step_error", None)
    code_method = getattr(planner, "replan_after_code_error", None)
    planner_type = type(planner)
    base_step_method = getattr(OceanHarnessPlanner, "replan_after_step_error", None)
    base_code_method = getattr(OceanHarnessPlanner, "replan_after_code_error", None)
    type_step_method = getattr(planner_type, "replan_after_step_error", None)
    type_code_method = getattr(planner_type, "replan_after_code_error", None)
    if callable(step_method) and type_step_method is not base_step_method:
        return step_method
    if callable(code_method) and type_code_method is not base_code_method:
        return code_method
    if callable(step_method):
        return step_method
    if callable(code_method):
        return code_method
    return None


def _planner_observability(plan: Mapping[str, Any]) -> Dict[str, Any]:
    decision = plan.get("planner_llm_decision") if isinstance(plan.get("planner_llm_decision"), Mapping) else {}
    payload: Dict[str, Any] = {}
    timings = decision.get("planner_agent_timings") if isinstance(decision, Mapping) else None
    if isinstance(timings, Mapping):
        payload["planner_agent_timings"] = dict(timings)
    prompt_sizes = decision.get("planner_agent_prompt_sizes") if isinstance(decision, Mapping) else None
    if isinstance(prompt_sizes, Mapping):
        payload["planner_agent_prompt_sizes"] = dict(prompt_sizes)
    return payload


def _approved_plan_from_context(additional_context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(additional_context, dict):
        return None
    proposal_context = additional_context.get("analysis_proposal_context")
    if not isinstance(proposal_context, dict):
        return None
    approved_plan = proposal_context.get("approved_plan")
    if not (
        isinstance(approved_plan, dict)
        and isinstance(approved_plan.get("steps"), list)
        and approved_plan.get("steps")
    ):
        return None
    plan = dict(approved_plan)
    plan["steps"] = list(approved_plan.get("steps") or [])
    return plan


class OceanHarnessExecutor:
    def __init__(
        self,
        workspace: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Any]] = None,
        code_agent: Optional[CodeAgent] = None,
    ) -> None:
        self.workspace = workspace or {}
        self.tools = tools or get_tools_cached()
        self.code_agent = code_agent or CodeAgent()
        self.orchestrator = ToolOrchestrator(workspace=self.workspace, tools=self.tools)
        self.results = self.orchestrator.results
        self.store = ArtifactStore()

    async def execute_query(
        self,
        user_request: str,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        planner: Optional[Any] = None,
        planner_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        yield {"type": "planning_start", "user_request": user_request}
        planning_started_at = time.perf_counter()
        planner_options = dict(planner_kwargs or {})
        code_agent = planner_options.pop("code_agent", None) or self.code_agent
        planning_timeout_s = float(planner_options.pop("planning_timeout_s", 0.0))
        active_planner: Optional[Any] = None
        approved_plan = _approved_plan_from_context(additional_context)
        if approved_plan is not None:
            plan = approved_plan
        else:
            active_planner = (
                planner
                if isinstance(planner, OceanHarnessPlanner)
                else OceanHarnessPlanner(llm_planner=planner, code_agent=code_agent)
            )
            local_planner_inline = planner is None or (
                isinstance(active_planner, OceanHarnessPlanner)
                and getattr(active_planner, "llm_planner", None) is None
            )
            if isinstance(active_planner, OceanHarnessPlanner):
                self.code_agent = active_planner.code_agent
            try:
                if local_planner_inline:
                    plan = active_planner.generate_plan_for_query(
                        user_request=user_request,
                        extracted_params=extracted_params,
                        additional_context=additional_context,
                        **planner_options,
                    )
                else:
                    loop = asyncio.get_running_loop()
                    planning_future = loop.run_in_executor(
                        _HARNESS_PLANNER_POOL,
                        lambda: active_planner.generate_plan_for_query(
                            user_request=user_request,
                            extracted_params=extracted_params,
                            additional_context=additional_context,
                            **planner_options,
                        ),
                    )
                    if planning_timeout_s > 0:
                        plan = await asyncio.wait_for(planning_future, timeout=planning_timeout_s)
                    else:
                        plan = await planning_future
            except asyncio.TimeoutError:
                yield {
                    "type": "planning_failed",
                    "error": f"Planning exceeded {planning_timeout_s:.0f}s before producing a plan.",
                    "error_category": "planning_timeout",
                    "planning_elapsed_s": round(time.perf_counter() - planning_started_at, 3),
                }
                return
            except Exception as exc:
                yield {
                    "type": "planning_failed",
                    "error": str(exc),
                    "error_category": "planning_failed",
                    "planning_elapsed_s": round(time.perf_counter() - planning_started_at, 3),
                }
                return

        planning_elapsed_s = round(time.perf_counter() - planning_started_at, 3)
        if not isinstance(plan, dict):
            yield {
                "type": "planning_failed",
                "error": "Planner did not return a valid plan.",
                "error_category": "planning_failed",
                "planning_elapsed_s": planning_elapsed_s,
            }
            return
        if plan.get("status") == "clarification_needed":
            yield {
                "type": "clarification_needed",
                "skill_id": plan.get("skill_id", "ocean_harness"),
                "skills_used": plan.get("skills_used", ["ocean_harness"]),
                "question": plan.get("question") or "Please provide the missing analysis inputs.",
                "missing_fields": plan.get("missing_fields", []),
                "plan": plan,
                "planning_elapsed_s": planning_elapsed_s,
            }
            return

        max_replans = int(planner_options.get("max_replans", 1))
        current_plan = plan
        replans_used = 0
        while True:
            yield {
                "type": "plan_replanned" if replans_used else "plan_generated",
                "skill_id": current_plan.get("skill_id"),
                "skills_used": current_plan.get("skills_used", ["ocean_harness"]),
                "plan": current_plan,
                "replans_used": replans_used,
                "reason": current_plan.get("replan_reason") if replans_used else None,
                "planning_elapsed_s": planning_elapsed_s if replans_used == 0 else None,
                **(_planner_observability(current_plan) if replans_used == 0 else {}),
            }

            failed_event: Optional[Dict[str, Any]] = None
            async for event in self.execute_plan(current_plan):
                if event.get("type") == "step_error":
                    failed_event = dict(event)
                    break
                yield event

            if failed_event is None:
                return
            replan_method = _step_error_replan_method(active_planner)
            can_replan = replans_used < max_replans and replan_method is not None
            if not can_replan:
                yield failed_event
                return

            replan_reason = _step_error_replan_reason(current_plan, failed_event)
            yield {
                "type": "step_reflection_started",
                "step_id": failed_event.get("step_id"),
                "tool": failed_event.get("tool"),
                "params": failed_event.get("params", {}),
                "replans_used": replans_used,
                "max_replans": max_replans,
                "reason": replan_reason,
            }

            try:
                next_plan = replan_method(
                    previous_plan=current_plan,
                    failed_event=failed_event,
                    user_request=user_request,
                    extracted_params=extracted_params,
                    additional_context=additional_context,
                )
            except Exception as exc:
                failed_event["replan_error"] = str(exc)
                yield failed_event
                return
            if not isinstance(next_plan, dict) or next_plan.get("status") == "clarification_needed":
                yield failed_event
                return
            replans_used += 1
            next_plan["replan_reason"] = replan_reason
            current_plan = next_plan

    async def execute_plan(self, plan: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        skill_id = plan.get("skill_id", "ocean_harness")
        steps = list(plan.get("steps", []) or [])
        yield {
            "type": "plan_start",
            "skill_id": skill_id,
            "skills_used": plan.get("skills_used", [skill_id]),
            "n_steps": len(steps),
        }

        completed: List[str] = []
        for step in steps:
            async for event in self._execute_step(step):
                yield event
                if event.get("type") == "step_complete":
                    completed.append(str(event.get("result_id")))
                elif event.get("type") == "step_error":
                    return
        yield {
            "type": "plan_complete",
            "skill_id": skill_id,
            "skills_used": plan.get("skills_used", [skill_id]),
            "results": completed,
            "n_steps_completed": len(completed),
        }

    async def _execute_step(self, step: Mapping[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        step_id = str(step.get("step_id") or step.get("save_as") or "step")
        tool_name = str(step.get("tool") or "")
        params = dict(step.get("params") or {})
        save_as = str(step.get("save_as") or step_id)
        harness_node = step.get("harness_node") if isinstance(step.get("harness_node"), dict) else {}
        execution = harness_node.get("execution") if isinstance(harness_node.get("execution"), dict) else {}
        strategy = str(execution.get("strategy") or "")
        yield {
            "type": "step_start",
            "step_id": step_id,
            "tool": tool_name,
            "description": step.get("technical_label") or step.get("description") or "",
            "params": params,
        }
        progress_queue: Queue[Dict[str, Any]] = Queue()

        def report_progress(payload: Dict[str, Any]) -> None:
            progress_queue.put(payload)

        future = _HARNESS_TOOL_POOL.submit(
            lambda: self._execute_step_sync(
                tool_name,
                params,
                save_as,
                strategy=strategy,
                harness_node=harness_node,
                progress_callback=report_progress,
            )
        )

        while True:
            while True:
                try:
                    progress = progress_queue.get_nowait()
                except Empty:
                    break
                yield {
                    "type": "step_progress",
                    "step_id": step_id,
                    "tool": tool_name,
                    "progress": progress,
                    "params": params,
                }

            if future.done():
                try:
                    execution = future.result()
                except Exception as exc:
                    yield {
                        "type": "step_error",
                        "step_id": step_id,
                        "tool": tool_name,
                        "params": params,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                    return

                while True:
                    try:
                        progress = progress_queue.get_nowait()
                    except Empty:
                        break
                    yield {
                        "type": "step_progress",
                        "step_id": step_id,
                        "tool": tool_name,
                        "progress": progress,
                        "params": params,
                    }

                yield {
                    "type": "step_complete",
                    "step_id": step_id,
                    "tool": tool_name,
                    "result_id": save_as,
                    "ref_id": execution["ref_id"],
                    "output_type": execution["output_type"],
                    "result_summary": execution["summary"],
                    "params": params,
                }
                return

            await asyncio.sleep(0.05)

    def _execute_step_sync(
        self,
        tool_name: str,
        params: Dict[str, Any],
        save_as: str,
        strategy: str = "",
        harness_node: Optional[Mapping[str, Any]] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        if strategy == ExecutionStrategy.CODE.value:
            return self._execute_code_step_sync(
                params=params,
                save_as=save_as,
                harness_node=harness_node or {},
                progress_callback=progress_callback,
            )
        execution = self.orchestrator.execute_tool(
            tool_name=tool_name,
            params=params,
            save_as=save_as,
            progress_callback=progress_callback,
        )
        self._validate_runtime_contract(tool_name, save_as, execution)
        self._record_artifact(save_as, execution, tool_name=tool_name)
        return execution

    def _execute_code_step_sync(
        self,
        *,
        params: Dict[str, Any],
        save_as: str,
        harness_node: Mapping[str, Any],
        progress_callback=None,
    ) -> Dict[str, Any]:
        if progress_callback is not None:
            progress_callback({"phase": "prepare_code_inputs", "message": "Preparing code node inputs", "percent": 0.1})
        execution = harness_node.get("execution") if isinstance(harness_node.get("execution"), Mapping) else {}
        code = str(execution.get("code") or params.get("code") or "")
        input_refs = params.get("input_refs") if isinstance(params.get("input_refs"), Mapping) else {}
        code_inputs = {
            key: self.orchestrator.resolve_references(value)
            for key, value in input_refs.items()
        }
        input_schemas = _code_input_schemas(code_inputs)
        code_params = params.get("code_params") if isinstance(params.get("code_params"), Mapping) else {}
        if not code.strip():
            if progress_callback is not None:
                progress_callback({"phase": "design_generated_code", "message": "Designing generated analysis", "percent": 0.2})
            planner_analysis_design = (
                params.get("planner_analysis_design")
                if isinstance(params.get("planner_analysis_design"), Mapping)
                else params.get("analysis_design")
                if isinstance(params.get("analysis_design"), Mapping)
                else None
            )
            analysis_design = self.code_agent.design_analysis(
                user_request=str(code_params.get("user_request") or ""),
                input_refs=input_refs,
                code_params=code_params,
                input_schemas=input_schemas,
                planner_analysis_design=planner_analysis_design if isinstance(planner_analysis_design, Mapping) else None,
            )
            io_contract = self.code_agent.build_contract(analysis_design)
            code_steps = self.code_agent.write_code_steps(analysis_design, code_params=code_params)
            code_params = dict(code_params)
            code_params["analysis_design"] = analysis_design
            if progress_callback is not None:
                progress_callback({"phase": "write_generated_code", "message": "Writing generated analysis code", "percent": 0.3})
            code = self.code_agent.write_code(
                user_request=str(code_params.get("user_request") or ""),
                input_refs=input_refs,
                code_params=code_params,
                analysis_design=analysis_design,
                code_steps=code_steps,
                contract=io_contract,
                input_schemas=input_schemas,
            )
        else:
            analysis_design = params.get("analysis_design") if isinstance(params.get("analysis_design"), Mapping) else {}
            io_contract = params.get("io_contract") if isinstance(params.get("io_contract"), Mapping) else {}
        if progress_callback is not None:
            progress_callback({"phase": "run_generated_code", "message": "Running generated analysis code", "percent": 0.45})
        raw_result = run_code_node(code, inputs=code_inputs, params=code_params)
        normalized_result = self.orchestrator.normalize_result("generated_code", raw_result)
        _validate_generated_code_contract(normalized_result, io_contract, analysis_design)
        if progress_callback is not None:
            progress_callback({"phase": "wrap_code_result", "message": "Wrapping generated code result", "percent": 0.9})

        with self.orchestrator._lock:
            ref_id = self.orchestrator._next_ref_id()
            self.orchestrator.results[ref_id] = normalized_result
            self.orchestrator.results[save_as] = normalized_result
            if isinstance(self.workspace, dict):
                self.workspace[save_as] = normalized_result

        execution_result = {
            "ref_id": ref_id,
            "result_id": save_as,
            "output_type": normalized_result.get("output_type", "generic_result"),
            "result": normalized_result,
            "summary": self.orchestrator.summarize_result(normalized_result),
        }
        self._record_artifact(save_as, execution_result, tool_name="generated_code")
        return execution_result

    def _validate_runtime_contract(self, tool_name: str, artifact_id: str, execution: Mapping[str, Any]) -> None:
        normalized = execution.get("result")
        if not isinstance(normalized, Mapping):
            return
        validation = validate_tool_runtime_result(
            tool_name,
            artifact_id,
            normalized.get("data"),
            output_type=str(normalized.get("output_type") or execution.get("output_type") or ""),
        )
        validation.raise_for_errors()

    def _record_artifact(self, artifact_id: str, execution: Mapping[str, Any], *, tool_name: str = "") -> None:
        normalized = execution.get("result")
        if not isinstance(normalized, dict):
            return
        output_type = str(normalized.get("output_type") or execution.get("output_type") or "generic_result")
        data = normalized.get("data")
        if data is not None and hasattr(data, "dims"):
            frontend = FrontendType(output_type) if output_type in FrontendType._value2member_map_ else FrontendType.DATA_CONTAINER
            contract = get_tool_shape_contract(tool_name)
            kind = contract.output.kind if contract is not None and contract.output is not None else ArtifactKind.FIELD
            spec = spec_from_xarray(artifact_id, data, kind=kind, frontend_type=frontend)
        else:
            spec = _structured_spec(artifact_id, output_type, normalized)
        self.store.put(
            ArtifactRecord(
                artifact_id=artifact_id,
                spec=spec,
                value=normalized,
                summary=execution.get("summary") or {},
                provenance={"output_type": output_type},
            )
        )

    def get_result(self, result_id: str) -> Dict[str, Any]:
        return self.orchestrator.get_result(result_id)

    def get_result_summaries(self) -> Dict[str, Dict[str, Any]]:
        summaries: Dict[str, Dict[str, Any]] = {}
        seen_ids = set()
        for result_id, result in self.results.items():
            if result_id.startswith("result_"):
                continue
            if id(result) in seen_ids:
                continue
            seen_ids.add(id(result))
            summaries[result_id] = self.orchestrator.summarize_result(result)
        return summaries

    def clear_results(self) -> None:
        self.orchestrator.clear_results()
        self.store.clear()


def _structured_spec(artifact_id: str, output_type: str, normalized: Mapping[str, Any]):
    from packages.harness.shapes import shape_spec_from_dims
    from packages.harness.specs import ArtifactSpec

    frontend = FrontendType(output_type) if output_type in FrontendType._value2member_map_ else FrontendType.GENERIC
    if output_type == FrontendType.TIMESERIES.value:
        kind = ArtifactKind.SERIES
        dims: Tuple[str, ...] = ("time",)
    elif output_type == FrontendType.SPECTRUM.value:
        kind = ArtifactKind.SPECTRUM
        dims = ("frequency",)
    elif output_type == FrontendType.SPATIAL_FIELD.value:
        kind = ArtifactKind.MAP
        dims = ("lat", "lon")
    else:
        kind = ArtifactKind.TABLE
        dims = ()
    metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
    return ArtifactSpec(
        artifact_id=artifact_id,
        kind=kind,
        shape=shape_spec_from_dims(dims),
        units=str(metadata.get("units") or metadata.get("unit") or ""),
        frontend_type=frontend,
        variable=metadata.get("variable") if isinstance(metadata.get("variable"), str) else None,
    )


def _code_input_schemas(inputs: Mapping[str, Any]) -> Dict[str, Any]:
    schemas: Dict[str, Any] = {}
    for key, value in inputs.items():
        schema: Dict[str, Any] = {"python_type": type(value).__name__}
        dims = list(getattr(value, "dims", []) or [])
        if dims:
            schema["dims"] = dims
        shape = list(getattr(value, "shape", []) or [])
        if shape:
            schema["shape"] = [int(dim) for dim in shape]
        name = getattr(value, "name", None)
        if name is not None:
            schema["name"] = str(name)
        attrs = getattr(value, "attrs", None)
        if isinstance(attrs, Mapping):
            schema["attrs"] = {
                str(attr_key): attr_value
                for attr_key, attr_value in attrs.items()
                if isinstance(attr_value, (str, int, float, bool)) or attr_value is None
            }
        coords = getattr(value, "coords", None)
        if coords is not None:
            coord_summary: Dict[str, Any] = {}
            for coord_name in dims:
                if coord_name not in coords:
                    continue
                coord = coords[coord_name]
                coord_values = getattr(coord, "values", None)
                coord_entry: Dict[str, Any] = {}
                if hasattr(coord, "size"):
                    coord_entry["size"] = int(coord.size)
                try:
                    if coord_values is not None and len(coord_values) > 0:
                        coord_entry["start"] = str(coord_values[0])
                        coord_entry["end"] = str(coord_values[-1])
                except Exception:
                    pass
                if coord_entry:
                    coord_summary[str(coord_name)] = coord_entry
            if coord_summary:
                schema["coords"] = coord_summary
        schemas[str(key)] = schema
    return schemas


def _validate_generated_code_contract(
    normalized_result: Mapping[str, Any],
    contract: Mapping[str, Any],
    analysis_design: Mapping[str, Any],
) -> None:
    output = analysis_design.get("output") if isinstance(analysis_design.get("output"), Mapping) else {}
    expected_output_type = str(contract.get("expected_output_type") or output.get("output_type") or "").strip()
    if not expected_output_type or expected_output_type == "generic_result":
        return
    actual_output_type = str(normalized_result.get("output_type") or "").strip()
    if actual_output_type != expected_output_type:
        raise ValueError(
            "Generated code returned output_type "
            f"{actual_output_type or '<missing>'}, expected {expected_output_type} from CodeAgent design."
        )
    required_by_type = contract.get("required_fields_by_output_type")
    required_fields = []
    if isinstance(required_by_type, Mapping):
        candidate = required_by_type.get(expected_output_type)
        if isinstance(candidate, list):
            required_fields = [str(field) for field in candidate]
    missing = [field for field in required_fields if field not in normalized_result]
    if missing:
        raise ValueError(
            "Generated code result is missing required "
            f"{expected_output_type} fields: {', '.join(missing)}."
        )
