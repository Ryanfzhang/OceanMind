"""Skill retrieval facade for the harness planning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from packages.harness.manual_loader import (
    SkillSpec,
    WorkflowTemplate,
    load_skill_specs,
    retrieve_skill_specs,
    select_skill_workflow,
)


@dataclass(frozen=True)
class SkillRetrievalResult:
    matched_skills: Tuple[str, ...]
    skills: Mapping[str, SkillSpec]
    selected_workflow: Optional[WorkflowTemplate]

    @property
    def matched_manuals(self) -> Tuple[str, ...]:
        return self.matched_skills

    @property
    def manuals(self) -> Mapping[str, SkillSpec]:
        return self.skills

    @property
    def selected_recipe(self) -> Optional[WorkflowTemplate]:
        return self.selected_workflow


class SkillRetriever:
    """Load skill specs as context and choose only strong executable workflows."""

    def retrieve(self, query: str, *, limit: int = 6) -> SkillRetrievalResult:
        all_skills = load_skill_specs()
        skills = retrieve_skill_specs(query, limit=limit)
        workflow = select_skill_workflow(query, skills=all_skills)
        if workflow is not None and workflow.skill_id not in skills:
            if workflow.skill_id in all_skills:
                skills = {workflow.skill_id: all_skills[workflow.skill_id], **skills}
        return SkillRetrievalResult(
            matched_skills=tuple(skills.keys()),
            skills=skills,
            selected_workflow=workflow,
        )


ManualRetrievalResult = SkillRetrievalResult
ManualRetriever = SkillRetriever
