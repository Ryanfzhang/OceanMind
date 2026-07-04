"""High-level planning pipeline for OceanMind harness analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol

from packages.harness.analysis_dag import AnalysisDAGBuilder, AnalysisDAGDecision, PlanRoute
from packages.harness.data_scope import DataScope, DataScopeResolver
from packages.harness.manual_loader import WorkflowTemplate
from packages.harness.manual_retriever import SkillRetrievalResult, SkillRetriever


class HarnessPlanFactory(Protocol):
    """Protocol implemented by the planner's node-building facade."""

    def dataset_info_plan(self, context: Any) -> Dict[str, Any]: ...

    def skill_workflow_plan(self, context: Any, workflow: WorkflowTemplate) -> Dict[str, Any]: ...

    def manual_recipe_plan(self, context: Any, recipe: WorkflowTemplate) -> Dict[str, Any]: ...

    def pair_lag_relationship_plan(self, context: Any) -> Dict[str, Any]: ...

    def hypoxia_driver_plan(self, context: Any) -> Dict[str, Any]: ...

    def condition_mask_map_plan(self, context: Any) -> Dict[str, Any]: ...

    def generic_timeseries_plan(self, context: Any, *, include_trend: bool, include_spectrum: bool) -> Dict[str, Any]: ...

    def generic_code_plan(self, context: Any) -> Dict[str, Any]: ...

    def generic_map_plan(self, context: Any) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class PipelineTrace:
    stages: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"stages": [dict(stage) for stage in self.stages]}


class HarnessPlanningPipeline:
    """Router-adjacent dataset analysis pipeline.

    The API-level router decides whether data access is needed.  This pipeline
    then resolves data scope, retrieves manuals, chooses an analysis DAG family,
    and delegates node construction to a factory.  The factory returns the
    semantic graph, task graph, and frontend-compatible steps.
    """

    def __init__(
        self,
        *,
        data_scope_resolver: Optional[DataScopeResolver] = None,
        skill_retriever: Optional[SkillRetriever] = None,
        manual_retriever: Optional[SkillRetriever] = None,
        dag_builder: Optional[AnalysisDAGBuilder] = None,
    ) -> None:
        self.data_scope_resolver = data_scope_resolver or DataScopeResolver()
        self.skill_retriever = skill_retriever or manual_retriever or SkillRetriever()
        self.dag_builder = dag_builder or AnalysisDAGBuilder()

    def run(
        self,
        *,
        user_request: str,
        extracted_params: Mapping[str, Any],
        additional_context: Mapping[str, Any],
        factory: HarnessPlanFactory,
        context: Any,
    ) -> Dict[str, Any]:
        data_scope = self.data_scope_resolver.resolve(
            user_request=user_request,
            extracted_params=extracted_params,
            additional_context=additional_context,
        )
        skill_result = self.skill_retriever.retrieve(user_request)
        decision = self.dag_builder.decide(user_request, selected_workflow=skill_result.selected_workflow)
        plan = self._build_plan(factory=factory, context=context, decision=decision)
        self._attach_pipeline_metadata(plan, data_scope=data_scope, skill_result=skill_result, decision=decision)
        return plan

    def _build_plan(
        self,
        *,
        factory: HarnessPlanFactory,
        context: Any,
        decision: AnalysisDAGDecision,
    ) -> Dict[str, Any]:
        if decision.route == PlanRoute.DATASET_INFO:
            return factory.dataset_info_plan(context)
        if decision.route in {PlanRoute.SKILL_WORKFLOW, PlanRoute.MANUAL_RECIPE} and decision.selected_workflow is not None:
            if hasattr(factory, "skill_workflow_plan"):
                return factory.skill_workflow_plan(context, decision.selected_workflow)
            return factory.manual_recipe_plan(context, decision.selected_workflow)
        if decision.route == PlanRoute.PAIR_LAG_RELATIONSHIP:
            return factory.pair_lag_relationship_plan(context)
        if decision.route == PlanRoute.HYPOXIA_DRIVER:
            return factory.hypoxia_driver_plan(context)
        if decision.route == PlanRoute.CONDITION_MASK_SPATIAL_MAP:
            return factory.condition_mask_map_plan(context)
        if decision.route == PlanRoute.GENERIC_TIMESERIES:
            return factory.generic_timeseries_plan(
                context,
                include_trend=decision.include_trend,
                include_spectrum=decision.include_spectrum,
            )
        if decision.route == PlanRoute.GENERATED_CODE:
            return factory.generic_code_plan(context)
        return factory.generic_map_plan(context)

    def _attach_pipeline_metadata(
        self,
        plan: Dict[str, Any],
        *,
        data_scope: DataScope,
        skill_result: SkillRetrievalResult,
        decision: AnalysisDAGDecision,
    ) -> None:
        trace = PipelineTrace(
            stages=(
                {
                    "stage": "data_scope_resolver",
                    "output": data_scope.to_requirements_dict(),
                },
                {
                    "stage": "skill_retriever",
                    "matched_skills": list(skill_result.matched_skills),
                    "selected_workflow": skill_result.selected_workflow.workflow_id if skill_result.selected_workflow else None,
                    "selected_skill": skill_result.selected_workflow.skill_id if skill_result.selected_workflow else None,
                    "matched_manuals": list(skill_result.matched_skills),
                    "selected_recipe": skill_result.selected_workflow.workflow_id if skill_result.selected_workflow else None,
                    "selected_manual": skill_result.selected_workflow.skill_id if skill_result.selected_workflow else None,
                },
                {
                    "stage": "analysis_dag_builder",
                    "route": decision.route.value,
                    "include_trend": decision.include_trend,
                    "include_spectrum": decision.include_spectrum,
                    "reason": decision.reason,
                },
                {
                    "stage": "tool_code_binding",
                    "policy": "bind each semantic task to an existing tool, otherwise use generated_code node with run(inputs, params) -> dict",
                },
                {
                    "stage": "execution_projection",
                    "policy": "compile semantic/task graph to legacy steps for current executor and frontend progress",
                },
            )
        )
        plan["planner_pipeline"] = trace.to_dict()
        requirements = plan.setdefault("data_requirements", {})
        if isinstance(requirements, dict):
            requirements["resolved_scope"] = data_scope.to_requirements_dict()
            requirements["skill_context"] = list(skill_result.matched_skills)
            requirements["manual_context"] = list(skill_result.matched_skills)
        semantic = plan.get("semantic_task_graph")
        if isinstance(semantic, dict):
            metadata = semantic.setdefault("metadata", {})
            metadata["pipeline_route"] = decision.route.value
            metadata["pipeline_reason"] = decision.reason
