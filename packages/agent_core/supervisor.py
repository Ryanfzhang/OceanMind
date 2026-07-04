"""
Minimal multi-agent supervisor runtime.

Roles:
- planner / reviewer: SkillPlanner
- executor: deterministic SkillExecutor
- synthesizer: ResultSynthesizer
"""

from typing import Any, AsyncIterator, Dict, Optional

from packages.agent_core.executor import SkillExecutor
from packages.llm_gateway import ResultSynthesizer, SkillPlanner


class MultiAgentSupervisor:
    """
    Coordinate planner/reviewer, deterministic executor, and final synthesizer.
    """

    def __init__(
        self,
        executor: Optional[SkillExecutor] = None,
        planner: Optional[SkillPlanner] = None,
        reviewer: Optional[SkillPlanner] = None,
        synthesizer: Optional[ResultSynthesizer] = None,
        planner_kwargs: Optional[Dict[str, Any]] = None,
        reviewer_kwargs: Optional[Dict[str, Any]] = None,
        synthesizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.executor = executor or SkillExecutor()
        self.planner = planner or SkillPlanner(**(planner_kwargs or {}))
        self.reviewer = reviewer or SkillPlanner(**(reviewer_kwargs or planner_kwargs or {}))
        self.synthesizer = synthesizer or ResultSynthesizer(
            planner=self.planner,
            **(synthesizer_kwargs or {}),
        )

    async def run_query(
        self,
        user_request: str,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        review_each_step: bool = True,
        max_replans: int = 2,
        synthesize: bool = True,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Run the minimal multi-agent workflow.
        """
        last_plan: Optional[Dict[str, Any]] = None
        last_completed_results = []

        async for event in self.executor.execute_query(
            user_request=user_request,
            extracted_params=extracted_params,
            additional_context=additional_context,
            planner=self.planner,
            reviewer=self.reviewer,
            supervised=True,
            review_each_step=review_each_step,
            max_replans=max_replans,
        ):
            if event.get("type") in {"plan_generated", "plan_replanned"}:
                last_plan = event.get("plan")
            if event.get("type") == "plan_complete":
                last_completed_results = list(event.get("results", []))
            yield event

        if not synthesize or not last_plan or not last_completed_results:
            return

        synthesis = self.synthesizer.synthesize(
            user_request=user_request,
            active_plan=last_plan,
            completed_steps=[
                {"result_id": result_id, "summary": self.executor.get_result_summaries().get(result_id, {})}
                for result_id in last_completed_results
            ],
            result_summaries=self.executor.get_result_summaries(),
            additional_context=additional_context,
        )

        yield {
            "type": "final_synthesis",
            "synthesis": synthesis,
        }
