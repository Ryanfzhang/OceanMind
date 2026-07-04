"""
Agent Core - Skill执行引擎

负责根据Skill计划执行工具链
"""

import asyncio
import os
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, Queue
from typing import Dict, List, Any, AsyncIterator, Optional, Tuple
from packages.llm_gateway import SkillPlanner
from packages.tool_loader.introspect import get_tools_cached
from packages.tool_loader.orchestrator import ToolOrchestrator
from packages.tool_loader.registry import get_param_contract, get_tool_contract
from packages.tool_loader.validation import normalize_tool_param_value, validate_tool_params


def _worker_count(env_name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(env_name, str(default))))
    except ValueError:
        return default


_PLANNER_POOL = ThreadPoolExecutor(max_workers=_worker_count("OCEAN_PLANNER_WORKERS", 3))
_TOOL_POOL = ThreadPoolExecutor(max_workers=_worker_count("OCEAN_TOOL_WORKERS", 2))


async def _await_thread_future(future: Future, poll_interval: float = 0.02) -> Any:
    """Await a thread future without relying on cross-thread event-loop wakeups."""
    try:
        while True:
            if future.done():
                return future.result()
            await asyncio.sleep(poll_interval)
    except asyncio.CancelledError:
        future.cancel()
        raise


async def _run_in_thread_pool(executor: ThreadPoolExecutor, func) -> Any:
    return await _await_thread_future(executor.submit(func))

_SAFE_TOOL_ALIASES = {
    "detect_bloom_events": "detect_algal_blooms",
    "compute_density_from_temp_salt": "compute_density",
    "compute_spatial_diagnostic": "compute_spatial_field",
}

_SAFE_PARAM_ALIASES = {
    "assemble_dataset": {
        "arrays": "variables",
        "fields": "variables",
    },
    "rank_mechanism_support": {
        "mechanism_scores": "evidence_items",
        "scores": "evidence_items",
    },
    "grade_evidence_strength": {
        "mechanism_scores": "evidence_items",
        "scores": "evidence_items",
    },
}


class PlanNormalizationError(ValueError):
    """Raised when a planner-produced plan cannot be safely normalized."""

    def __init__(self, message: str, *, category: str = "planning_failed"):
        super().__init__(message)
        self.category = category


class SkillExecutor:
    """
    Skill执行器

    负责执行skill计划，调用domain层的工具函数，管理中间结果
    """

    def __init__(self, workspace=None, tools=None):
        """
        初始化执行器

        Args:
            workspace: 工作空间对象（用于保存中间结果）
            tools: 可选的工具映射，传入后跳过自动发现
        """
        self.workspace = workspace or {}
        self.tools = tools or get_tools_cached()
        self.orchestrator = ToolOrchestrator(workspace=self.workspace, tools=self.tools)
        self.results = self.orchestrator.results

    async def execute_plan(self, plan: Dict) -> AsyncIterator[Dict]:
        """
        异步执行skill计划

        Args:
            plan: 执行计划，格式：
                {
                    'skill_id': 'ocean_timeseries',
                    'steps': [
                        {
                            'step_id': 'step_1',
                            'tool': 'load_dataset',
                            'params': {...},
                            'save_as': 'raw_data'
                        },
                        ...
                    ]
                }

        Yields:
            执行事件：
            - {'type': 'step_start', 'step_id': '...', 'tool': '...'}
            - {'type': 'step_complete', 'step_id': '...', 'result_id': '...'}
            - {'type': 'step_error', 'step_id': '...', 'error': '...'}
            - {'type': 'plan_complete', 'results': {...}}

        Example:
            >>> executor = SkillExecutor()
            >>> async for event in executor.execute_plan(plan):
            ...     print(event['type'], event.get('step_id'))
        """
        skill_id = plan.get('skill_id', 'unknown')

        # 发送开始事件
        yield {
            'type': 'plan_start',
            'skill_id': skill_id,
            'n_steps': len(plan.get('steps', []))
        }

        completed_result_ids = []

        for step in plan.get('steps', []):
            async for event in self._execute_step_async(step):
                yield event
                if event['type'] == 'step_complete':
                    completed_result_ids.append(event['result_id'])
                elif event['type'] == 'step_error':
                    return

        # 发送完成事件
        yield {
            'type': 'plan_complete',
            'skill_id': skill_id,
            'results': completed_result_ids,
            'n_steps_completed': len(completed_result_ids)
        }

    def _summarize_result(self, result: Any) -> Dict:
        """
        生成结果摘要（用于事件）

        Args:
            result: 工具执行结果

        Returns:
            结果摘要字典
        """
        return self.orchestrator.summarize_result(result)

    def get_result(self, result_id: str) -> Any:
        """
        获取执行结果

        Args:
            result_id: 结果ID

        Returns:
            结果对象

        Raises:
            KeyError: 如果结果不存在
        """
        return self.orchestrator.get_result(result_id)

    def clear_results(self):
        """清空所有结果"""
        self.orchestrator.clear_results()

    async def execute_query(
        self,
        user_request: str,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        planner: Optional[SkillPlanner] = None,
        planner_kwargs: Optional[Dict[str, Any]] = None,
        supervised: bool = False,
        reviewer: Optional[SkillPlanner] = None,
        review_each_step: bool = False,
        max_replans: int = 2,
    ) -> AsyncIterator[Dict]:
        """
        使用 LLM 自动选择 skill 并生成 plan，然后执行该 plan。

        Yields:
            - {'type': 'planning_start', 'user_request': '...'}
            - {'type': 'plan_generated', 'skill_id': '...', 'plan': {...}}
            - execute_plan() 产生的事件
        """
        yield {
            'type': 'planning_start',
            'user_request': user_request
        }

        if supervised:
            async for event in self.execute_query_supervised(
                user_request=user_request,
                extracted_params=extracted_params,
                additional_context=additional_context,
                planner=planner,
                reviewer=reviewer,
                planner_kwargs=planner_kwargs,
                review_each_step=review_each_step,
                max_replans=max_replans,
            ):
                yield event
            return

        active_planner = planner or SkillPlanner(**(planner_kwargs or {}))
        try:
            plan = await _run_in_thread_pool(
                _PLANNER_POOL,
                lambda: active_planner.generate_plan_for_query(
                    user_request=user_request,
                    extracted_params=extracted_params,
                    additional_context=additional_context,
                )
            )
            plan = self._normalize_generated_plan(
                plan,
                user_request=user_request,
                extracted_params=extracted_params,
                additional_context=additional_context,
            )
        except PlanNormalizationError as exc:
            yield self._planning_failed_event(error=str(exc), category=exc.category)
            return
        except Exception as exc:
            yield self._planning_failed_event(error=str(exc), category="planning_failed")
            return

        if plan.get('status') == 'clarification_needed':
            yield {
                'type': 'clarification_needed',
                'skill_id': plan.get('skill_id'),
                'missing_fields': plan.get('missing_fields', []),
                'question': plan.get('clarification_question', ''),
            }
            return

        yield {
            'type': 'plan_generated',
            'skill_id': plan.get('skill_id'),
            'plan': plan
        }

        async for event in self.execute_plan(plan):
            yield event

    async def execute_query_supervised(
        self,
        user_request: str,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        planner: Optional[SkillPlanner] = None,
        reviewer: Optional[SkillPlanner] = None,
        planner_kwargs: Optional[Dict[str, Any]] = None,
        review_each_step: bool = True,
        max_replans: int = 2,
    ) -> AsyncIterator[Dict]:
        """
        Multi-skill supervisor loop:
        planner generates a plan, deterministic execution runs each step, and
        a reviewer may continue, replan, ask the user, or abort.
        """
        active_planner = planner or SkillPlanner(**(planner_kwargs or {}))
        active_reviewer = reviewer or active_planner

        try:
            plan = await _run_in_thread_pool(
                _PLANNER_POOL,
                lambda: active_planner.generate_plan_for_query(
                    user_request=user_request,
                    extracted_params=extracted_params,
                    additional_context=additional_context,
                    allow_multiple_skills=True,
                )
            )
            plan = self._normalize_generated_plan(
                plan,
                user_request=user_request,
                extracted_params=extracted_params,
                additional_context=additional_context,
            )
        except PlanNormalizationError as exc:
            yield self._planning_failed_event(error=str(exc), category=exc.category)
            return
        except Exception as exc:
            yield self._planning_failed_event(error=str(exc), category="planning_failed")
            return

        if plan.get('status') == 'clarification_needed':
            yield {
                'type': 'clarification_needed',
                'skill_id': plan.get('skill_id'),
                'skills_used': plan.get('skills_used', [plan.get('skill_id')]),
                'missing_fields': plan.get('missing_fields', []),
                'question': plan.get('clarification_question', ''),
            }
            return

        completed_steps: List[Dict[str, Any]] = []
        replans_used = 0
        plan_version = 1
        current_plan = plan

        yield {
            'type': 'plan_generated',
            'skill_id': current_plan.get('skill_id'),
            'skills_used': current_plan.get('skills_used', [current_plan.get('skill_id')]),
            'plan': current_plan
        }

        while True:
            skill_id = current_plan.get('skill_id', 'unknown')
            steps = current_plan.get('steps', [])

            yield {
                'type': 'plan_start',
                'skill_id': skill_id,
                'skills_used': current_plan.get('skills_used', [skill_id]),
                'n_steps': len(steps),
                'plan_version': plan_version,
            }

            replan_triggered = False
            index = 0
            while index < len(steps):
                collected_events: List[Dict[str, Any]] = []
                ok = False
                step_record: Optional[Dict[str, Any]] = None
                step = steps[index]
                async for event in self._execute_step_async(step):
                    yield event
                    collected_events.append(event)
                    if event['type'] == 'step_complete':
                        ok = True
                        step_record = {
                            'step_id': event['step_id'],
                            'tool': event['tool'],
                            'result_id': event['result_id'],
                            'ref_id': event.get('ref_id'),
                            'output_type': event.get('output_type'),
                            'result_summary': event.get('result_summary'),
                        }
                    elif event['type'] == 'step_error':
                        ok = False
                        step_record = None

                last_event = collected_events[-1]
                remaining_steps = steps[index + 1:]

                if ok and step_record is not None:
                    completed_steps.append(step_record)
                    should_review = review_each_step
                else:
                    should_review = True

                if not should_review:
                    index += 1
                    continue

                if not ok and replans_used >= max_replans:
                    yield {
                        'type': 'plan_aborted',
                        'skill_id': skill_id,
                        'reason': 'max_replans_exceeded',
                        'last_error': last_event.get('error', ''),
                    }
                    return

                review_context = {
                    **(additional_context or {}),
                    'available_results': self._build_available_result_summaries(),
                }
                sanitized_last_event = self._sanitize_event_for_review(last_event)
                try:
                    review = await _run_in_thread_pool(
                        _PLANNER_POOL,
                        lambda: active_reviewer.review_execution(
                            user_request=user_request,
                            active_plan=current_plan,
                            completed_steps=completed_steps,
                            remaining_steps=remaining_steps,
                            last_event=sanitized_last_event,
                            extracted_params=extracted_params,
                            additional_context=review_context,
                        ),
                    )
                except PlanNormalizationError as exc:
                    yield self._planning_failed_event(error=str(exc), category=exc.category)
                    return
                except Exception as exc:
                    yield self._planning_failed_event(error=str(exc), category="planning_failed")
                    return

                yield {
                    'type': 'review_decision',
                    'decision': review['decision'],
                    'reason': review['reason'],
                }

                action = review['decision']
                if action == 'continue':
                    if ok:
                        index += 1
                        continue
                    yield {
                        'type': 'plan_aborted',
                        'skill_id': skill_id,
                        'reason': 'review_continue_after_error_is_invalid',
                        'last_error': last_event.get('error', ''),
                    }
                    return

                if action == 'ask_user':
                    yield {
                        'type': 'clarification_needed',
                        'skill_id': current_plan.get('skill_id'),
                        'skills_used': current_plan.get('skills_used', [current_plan.get('skill_id')]),
                        'missing_fields': review.get('missing_fields', []),
                        'question': review.get('clarification_question', ''),
                    }
                    return

                if action == 'abort':
                    yield {
                        'type': 'plan_aborted',
                        'skill_id': skill_id,
                        'reason': review['reason'],
                        'last_error': last_event.get('error', ''),
                    }
                    return

                if action == 'replan':
                    if replans_used >= max_replans:
                        yield {
                            'type': 'plan_aborted',
                            'skill_id': skill_id,
                            'reason': 'max_replans_exceeded',
                            'last_error': last_event.get('error', ''),
                        }
                        return
                    replans_used += 1
                    try:
                        current_plan = self._normalize_generated_plan(
                            review['updated_plan'],
                            user_request=user_request,
                            extracted_params=extracted_params,
                            additional_context=additional_context,
                        )
                    except PlanNormalizationError as exc:
                        yield self._planning_failed_event(error=str(exc), category=exc.category)
                        return
                    plan_version += 1
                    replan_triggered = True
                    yield {
                        'type': 'plan_replanned',
                        'skill_id': current_plan.get('skill_id'),
                        'skills_used': current_plan.get('skills_used', [current_plan.get('skill_id')]),
                        'plan_version': plan_version,
                        'reason': review['reason'],
                        'plan': current_plan,
                    }
                    break

            if replan_triggered:
                continue

            yield {
                'type': 'plan_complete',
                'skill_id': skill_id,
                'skills_used': current_plan.get('skills_used', [skill_id]),
                'results': [step['result_id'] for step in completed_steps],
                'n_steps_completed': len(completed_steps)
            }
            return

    async def _execute_step_async(self, step: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Async step execution that yields step_start BEFORE running the tool,
        allowing the SSE stream to push the event to the frontend immediately.
        The actual tool execution runs in a thread pool to avoid blocking the event loop.
        """
        prepared = self._prepare_step_for_execution(step)
        blocked = False
        for event in self._pre_execution_events(prepared):
            yield event
            if event['type'] == 'step_error':
                blocked = True
        if blocked:
            return

        async for event in self._execute_prepared_step_async(prepared, executor=_TOOL_POOL):
            yield event

    def _prepare_step_for_execution(self, step: Dict[str, Any]) -> Dict[str, Any]:
        step_id = step['step_id']
        tool_name = step['tool']
        save_id = step.get('save_as', step_id)

        sanitized_params, ignored_params = self._sanitize_step_params(tool_name, step.get('params', {}))
        sanitized_step = {**step, 'params': sanitized_params}
        coerced_params = self._coerce_step_params(tool_name, sanitized_step.get('params', {}))
        prepared_step = {**sanitized_step, 'params': coerced_params}
        validation_issues = validate_tool_params(tool_name, prepared_step.get('params', {}))

        return {
            'step_id': step_id,
            'tool_name': tool_name,
            'save_id': save_id,
            'description': step.get('description', ''),
            'coerced_params': coerced_params,
            'prepared_step': prepared_step,
            'ignored_params': ignored_params,
            'validation_issues': validation_issues,
        }

    def _pre_execution_events(self, prepared: Dict[str, Any]) -> List[Dict[str, Any]]:
        step_id = prepared['step_id']
        tool_name = prepared['tool_name']
        events: List[Dict[str, Any]] = [
            {
                'type': 'step_start',
                'step_id': step_id,
                'tool': tool_name,
                'description': prepared.get('description', ''),
                'params': prepared['coerced_params'],
            }
        ]

        for param_name in prepared.get('ignored_params', []):
            events.append({
                'type': 'step_warning',
                'step_id': step_id,
                'tool': tool_name,
                'param': param_name,
                'warning': f"Ignored unsupported parameter '{param_name}' for tool '{tool_name}'."
            })

        for issue in prepared.get('validation_issues', []):
            if issue.get('level') == 'error':
                events.append({
                    'type': 'step_error',
                    'step_id': step_id,
                    'tool': tool_name,
                    'params': prepared['prepared_step'].get('params', {}),
                    'error': issue['message'],
                    'error_type': 'ValidationError'
                })
            else:
                events.append({
                    'type': 'step_warning',
                    'step_id': step_id,
                    'tool': tool_name,
                    'param': issue['param'],
                    'warning': issue['message']
                })
        return events

    def _has_validation_error(self, prepared: Dict[str, Any]) -> bool:
        return any(issue.get('level') == 'error' for issue in prepared.get('validation_issues', []))

    async def _execute_prepared_step_async(self, prepared: Dict[str, Any], executor: ThreadPoolExecutor) -> AsyncIterator[Dict[str, Any]]:
        progress_queue: Queue[Dict[str, Any]] = Queue()

        def report_progress(payload: Dict[str, Any]) -> None:
            progress_queue.put(payload)

        future = executor.submit(
            lambda: self._execute_prepared_step_sync(prepared, progress_callback=report_progress)
        )
        try:
            while True:
                while True:
                    try:
                        progress = progress_queue.get_nowait()
                    except Empty:
                        break
                    yield self._build_step_progress_event(prepared, progress)

                if future.done():
                    try:
                        execution = future.result()
                    except Exception as exc:
                        yield self._build_step_error_event(prepared, exc)
                    else:
                        while True:
                            try:
                                progress = progress_queue.get_nowait()
                            except Empty:
                                break
                            yield self._build_step_progress_event(prepared, progress)
                        yield self._build_step_complete_event(prepared, execution)
                    return

                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            future.cancel()
            raise

    def _execute_prepared_step_sync(
        self,
        prepared: Dict[str, Any],
        progress_callback=None,
    ) -> Dict[str, Any]:
        return self.orchestrator.execute_tool(
            tool_name=prepared['tool_name'],
            params=prepared['prepared_step'].get('params', {}),
            save_as=prepared['save_id'],
            progress_callback=progress_callback,
        )

    def _build_step_progress_event(self, prepared: Dict[str, Any], progress: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'step_progress',
            'step_id': prepared['step_id'],
            'tool': prepared['tool_name'],
            'progress': progress,
            'params': prepared['prepared_step'].get('params', {}),
        }

    def _build_step_complete_event(self, prepared: Dict[str, Any], execution: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'step_complete',
            'step_id': prepared['step_id'],
            'tool': prepared['tool_name'],
            'result_id': prepared['save_id'],
            'ref_id': execution['ref_id'],
            'output_type': execution['output_type'],
            'result_summary': execution['summary'],
            'params': prepared['prepared_step'].get('params', {}),
        }

    def _build_step_error_event(self, prepared: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        return {
            'type': 'step_error',
            'step_id': prepared['step_id'],
            'tool': prepared['tool_name'],
            'params': prepared['prepared_step'].get('params', {}),
            'error': str(exc),
            'error_type': type(exc).__name__
        }

    def _sanitize_step_params(self, tool_name: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        if not isinstance(params, dict):
            return {}, []

        contract = get_tool_contract(tool_name) or {}
        defined_params = set(contract.get('inputs', {}).keys())
        if not defined_params:
            return dict(params), []

        sanitized: Dict[str, Any] = {}
        ignored: List[str] = []
        for param_name, value in params.items():
            if param_name in defined_params:
                sanitized[param_name] = value
            else:
                ignored.append(param_name)
        return sanitized, ignored

    def _coerce_step_params(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coerce LLM-emitted parameter values to match their contract types.

        When the LLM emits a single-element list for a scalar parameter
        (e.g. ``"variable": ["u"]``), extract the scalar so downstream tool
        code never receives an unexpected list.
        """
        coerced: Dict[str, Any] = {}
        for param_name, value in params.items():
            contract = get_param_contract(tool_name, param_name)
            if contract is None:
                coerced[param_name] = value
                continue

            normalized_value = normalize_tool_param_value(
                tool_name,
                param_name,
                value,
                contract,
                use_default=False,
            )

            kind = contract.get('kind')
            param_type = contract.get('type', '')

            if kind == 'literal':
                if param_type in ('string',):
                    if isinstance(normalized_value, (list, tuple)) and normalized_value:
                        first = normalized_value[0]
                        coerced[param_name] = str(first)
                    else:
                        coerced[param_name] = normalized_value
                elif param_type in ('number', 'integer'):
                    if isinstance(normalized_value, (list, tuple)) and normalized_value:
                        first = normalized_value[0]
                        coerced[param_name] = first
                    else:
                        coerced[param_name] = normalized_value
                else:
                    # literal + array or any other type: keep as-is
                    coerced[param_name] = normalized_value
            elif kind == 'literal_or_ref':
                if param_type == 'array':
                    # If a plain string was given instead of a list, wrap it
                    if isinstance(normalized_value, str) and not normalized_value.startswith('$ref:'):
                        coerced[param_name] = [normalized_value]
                    else:
                        coerced[param_name] = normalized_value
                else:
                    coerced[param_name] = normalized_value
            else:
                coerced[param_name] = normalized_value
        return coerced

    def _build_available_result_summaries(self) -> Dict[str, Dict[str, Any]]:
        return self.get_result_summaries()

    def _planning_failed_event(self, *, error: str, category: str) -> Dict[str, Any]:
        return {
            'type': 'planning_failed',
            'error': error,
            'error_category': category,
        }

    def _normalize_generated_plan(
        self,
        plan: Dict[str, Any],
        *,
        user_request: str,
        extracted_params: Optional[Dict[str, Any]],
        additional_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            raise PlanNormalizationError("Planner returned a non-dict plan payload.")

        status = plan.get('status', 'ready')
        if status == 'clarification_needed':
            return plan

        normalized_plan = dict(plan)
        normalized_plan.setdefault('status', 'ready')

        raw_steps = normalized_plan.get('steps', [])
        if not isinstance(raw_steps, list):
            raise PlanNormalizationError("Planner produced a malformed 'steps' field.")

        skill_ids = self._plan_skill_ids(normalized_plan)
        normalized_steps: List[Dict[str, Any]] = []
        for step in raw_steps:
            normalized_steps.append(
                self._normalize_plan_step(
                    step,
                    skill_ids=skill_ids,
                    user_request=user_request,
                    extracted_params=extracted_params or {},
                    additional_context=additional_context or {},
                )
            )

        normalized_plan['steps'] = normalized_steps
        normalized_plan = SkillPlanner._dedupe_plan_save_as_ids(normalized_plan)
        return normalized_plan

    def _normalize_plan_step(
        self,
        step: Dict[str, Any],
        *,
        skill_ids: List[str],
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(step, dict):
            raise PlanNormalizationError("Planner produced a non-dict step.")

        normalized_step = dict(step)
        tool_name = normalized_step.get('tool')
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise PlanNormalizationError("Plan step is missing a valid tool name.")

        normalized_tool = _SAFE_TOOL_ALIASES.get(tool_name, tool_name)
        normalized_step['tool'] = normalized_tool
        if get_tool_contract(normalized_tool) is None:
            raise PlanNormalizationError(
                f"Plan step uses unknown tool '{tool_name}'.",
                category="planning_failed",
            )

        params = normalized_step.get('params', {})
        if not isinstance(params, dict):
            raise PlanNormalizationError(
                f"Plan step for '{normalized_tool}' must contain object params.",
                category="planning_failed",
            )

        params = self._normalize_param_aliases(normalized_tool, params)
        params = self._normalize_step_specific_params(normalized_tool, params)

        rewritten_tool = params.pop('__final_tool__', None)
        if isinstance(rewritten_tool, str):
            normalized_tool = rewritten_tool
            normalized_step['tool'] = rewritten_tool

        validation_issues = validate_tool_params(normalized_tool, params)
        errors = [issue['message'] for issue in validation_issues if issue.get('level') == 'error']
        if errors:
            raise PlanNormalizationError("; ".join(errors), category="validation_error")

        normalized_step['params'] = params
        return normalized_step

    def _normalize_param_aliases(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        alias_map = _SAFE_PARAM_ALIASES.get(tool_name, {})
        if not alias_map:
            return dict(params)

        normalized: Dict[str, Any] = {}
        for param_name, value in params.items():
            normalized[alias_map.get(param_name, param_name)] = value
        return normalized

    def _normalize_step_specific_params(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return dict(params)

    def _plan_skill_ids(self, plan: Dict[str, Any]) -> List[str]:
        skills_used = plan.get('skills_used')
        if isinstance(skills_used, list):
            filtered = [skill_id for skill_id in skills_used if isinstance(skill_id, str)]
            if filtered:
                return filtered
        skill_id = plan.get('skill_id')
        if isinstance(skill_id, str):
            return [skill_id]
        return []

    def get_result_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Return compact summaries for named results stored in the executor."""
        summaries: Dict[str, Dict[str, Any]] = {}
        seen_ids = set()
        for result_id, result in self.results.items():
            if result_id.startswith('result_'):
                continue
            if id(result) in seen_ids:
                continue
            seen_ids.add(id(result))
            summaries[result_id] = self.orchestrator.summarize_result(result)
        return summaries

    def _sanitize_event_for_review(self, event: Dict[str, Any]) -> Dict[str, Any]:
        allowed_keys = {
            'type', 'step_id', 'tool', 'result_id', 'output_type',
            'result_summary', 'error', 'error_type', 'warning', 'param', 'params'
        }
        return {key: value for key, value in event.items() if key in allowed_keys}
