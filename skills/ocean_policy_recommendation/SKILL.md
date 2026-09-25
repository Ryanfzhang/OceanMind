---
skill_id: ocean_policy_recommendation
description: Turns completed ocean diagnostics into recommendations whose specificity matches their evidence.
input_intent: A management or policy question with completed analysis, or a request to identify what evidence is missing.
output_intent: Actions, monitoring priorities, and explicit evidence limits for the requested region and objective.
avoid_when:
- Use an analysis skill first when the user requests a new map, event detection, trend, or other computation.
composes_with:
- ocean_environment_health_assessment
- ocean_masking_workflow
---
# Ocean Policy Recommendation

## Start from the question and completed results

Identify the management objective, region, time horizon, and decision scale.
Inspect the actual saved diagnostics before proposing an action. If the user
asks for new low-oxygen, bloom, heat, or stratification calculations, read the
relevant analysis skill and run only those diagnostics first. For a broad
environmental-health question, `ocean_environment_health_assessment` helps
select them. This policy skill is for translating results, not a substitute
for obtaining results.

For each proposed claim, record which result supports it:

| Evidence position | Examples | Appropriate recommendation |
| --- | --- | --- |
| Direct computed endpoint | Bottom oxygen or hypoxic days for oxygen risk; detected heatwaves for heat exposure; bloom events for bloom pressure | Specify monitored zone and season only to the spatial and temporal precision of the result. |
| Proxy or contextual evidence | Density-derived stratification for ventilation vulnerability; chlorophyll for bloom pressure; warming overlap | Recommend targeted monitoring or a testable follow-up, and name the unresolved causal link. Do not state that the proxy proves hypoxia, a bloom, or a pollution source. |
| Missing, failed, or incompatible result | No oxygen data, mismatched periods, empty event output, or an unvalidated source | State that the endpoint is untested. Recommend the minimum missing measurements or analysis; do not issue a location-specific intervention as if risk were established. |

Compare diagnostics only after checking their region, dates, depth, units,
thresholds, and uncertainty. A short record or a nonsignificant trend does not
support a long-term directional claim. If evidence conflicts, show the conflict
and give a reversible monitoring or review action before a strong intervention.
Match each action to an observable trigger for revisiting it, and state who or
what it concerns only when the evidence supports that scope.

If the user asks about a current regulation or policy instrument, retrieve
current authoritative policy sources separately and cite them. A modelled
ocean indicator is evidence about conditions, not evidence that a legal power,
threshold, or obligation exists. Do not invent statutory requirements.

## Produce the answer

Give a short finding for each direct endpoint, then recommendations with their
evidence basis, priority, place/time, and limitation. Put proxy interpretations
and unanswered questions after the direct findings. If no diagnostic result
exists, say so plainly, ask for or suggest the minimum region, period, variable,
and endpoint needed, and limit guidance to evidence gathering or broadly
reversible monitoring. Do not turn `evidence_items=[]` into an apparently
evidence-backed targeted policy conclusion.

The old `assemble_policy_recommendation_report(evidence_items=...,
region_scope=..., policy_context=..., management_objective=...,
context_note=...)` is optional when the user specifically wants that fixed
format. It accepts only compatible normalized dictionaries with a nonempty
`output_type`; raw ordinary Python detector and trend results are often
filtered out. Confirm the accepted evidence count and table before relying on
its recommendations. If the compatible input list is empty, describe the
evidence gap directly instead of presenting the report's default action as a
diagnostic finding. `assemble_environment_health_report` is likewise optional
for an explicitly requested fixed environmental assessment, not a required
policy step.
