---
skill_id: ocean_policy_recommendation
description: Translates completed ocean evidence into bounded policy and management recommendations.
input_intent: Completed analysis, detection, mechanism, or environment-health evidence plus policy or management question.
output_intent: Policy recommendation report or management guidance card.
avoid_when:
- Do not use as the primary analysis skill when the user asks for raw computation, map, event detection, or diagnostic first.
composes_with:
- ocean_masking_workflow
---
# Ocean Policy Recommendation

## Purpose

This skill translates completed ocean analysis, detection, mechanism, and environment-assessment results into evidence-bounded policy and management recommendations for China Seas and western Pacific datasets. Use it as the final step af...

## Workflow

### Stage 1: Assemble policy recommendations

```python
evidence_items = []
region_scope = 'South China Sea'
policy_context = 'Monitoring and adaptive management'
management_objective = 'Prioritize robust ocean monitoring actions while preserving evidence limits.'
context_note = 'Standalone policy recommendation request without upstream diagnostic artifacts.'

policy_recommendation_report = assemble_policy_recommendation_report(
    evidence_items=evidence_items,
    region_scope=region_scope,
    policy_context=policy_context,
    management_objective=management_objective,
    context_note=context_note,
)
```


## Notes

- Supported mask builders include: threshold, condition, combined.
- Skill files describe retrieval, defaults, composition, and workflow intent; concrete type and shape checks live in the harness contracts.
