"""
LLM-based skill planner.

This module sends a raw `SKILL.md` document plus user inputs to the configured
LLM and expects a strict JSON execution plan in return.
"""

import re
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from packages.llm_gateway.config import DEFAULT_OPENAI_MODEL, LEGACY_ANTHROPIC_MODEL, load_model_name
from packages.llm_gateway.named_regions import (
    KNOWN_NAMED_REGION_BOUNDS as _KNOWN_NAMED_REGION_BOUNDS,
    NAMED_REGION_PLANNING_RULES as _NAMED_REGION_PLANNING_RULES,
    bounds_area as _named_region_bounds_area,
    bounds_are_reasonable_for_known_region as _named_region_bounds_are_reasonable_for_known_region,
    bounds_nearly_equal as _named_region_bounds_nearly_equal,
    bounds_too_broad_for_known_region as _named_region_bounds_too_broad_for_known_region,
    coerce_numeric_range as _coerce_numeric_range_impl,
    extract_named_region_mentions as _extract_named_region_mentions_impl,
    format_region_list as _format_region_list_impl,
    format_spatial_bounds as _format_spatial_bounds_impl,
    named_region_extent_suspicion as _named_region_extent_suspicion_impl,
    ranges_nearly_equal as _named_region_ranges_nearly_equal,
    ranges_overlap as _named_region_ranges_overlap,
    region_alias_matches as _region_alias_matches_impl,
    resolve_named_region_entities as _resolve_named_region_entities_impl,
)
from packages.llm_gateway.openai_compatible_client import OpenAICompatibleClientAdapter
from packages.llm_gateway.skill_plan_parsing import (
    extract_response_text,
    looks_like_json_parse_error,
    parse_json_response,
)
from packages.llm_gateway.skill_planner_client import create_message, get_client
from packages.skill_system import load_all_planner_hints
from packages.skill_system.loader import load_all_skill_markdowns, load_skill_markdown
from packages.tool_loader.registry import get_tool_contract, get_tool_output_type
from packages.tool_loader.validation import normalize_tool_param_value, validate_tool_params

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

_ENVIRONMENT_HEALTH_COMPOSER_TOOLS = {
    "apply_mask",
    "assemble_dataset",
    "assemble_environment_health_report",
    "assemble_policy_recommendation_report",
    "build_condition_mask",
    "build_isobath_mask",
    "build_polygon_mask",
    "build_threshold_mask",
    "combine_masks",
    "compute_area_weighted_mean",
    "compute_density",
    "compute_event_statistics",
    "compute_event_summary_map",
    "compute_trend",
    "compute_vertical_stability_timeseries",
    "detect_algal_blooms",
    "detect_heatwaves",
    "detect_hypoxia",
    "detect_upwelling",
    "load_dataset",
}


def _benchmark_disables_clarification(additional_context: Dict[str, Any]) -> bool:
    if not isinstance(additional_context, dict):
        return False
    policy = additional_context.get("benchmark_policy")
    return isinstance(policy, dict) and bool(policy.get("disable_clarification"))


_POLICY_GUIDANCE_INTENT_RE = re.compile(
    r"\b(policy|policies|management|recommendation|recommendations|economic|economics|"
    r"governance|coastal management|regulation|regulatory|mitigation|decision support|action plan|"
    r"source[- ]?control|nutrient[- ]?control|pollution[- ]?control|discharge[- ]?control)\b|"
    r"政策|管理|建议|经济|治理|管控|监管|排口|减缓|行动",
    re.IGNORECASE,
)

_EXPLICIT_POLICY_REPORT_RE = re.compile(
    r"\b(standalone|separate|dedicated|single)\s+(policy\s+)?(report|card|tool)\b|"
    r"\bpolicy\s+recommendation\s+(report|card|tool)\b|"
    r"单独.*(政策|policy).*?(报告|卡片|工具)|政策建议报告|政策报告",
    re.IGNORECASE,
)

_NEGATED_POLICY_REPORT_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|without|no|not|avoid|exclude|skip|never)\b.{0,80}"
    r"\b(?:policy\s+)?(?:recommendation\s+)?(?:report|card|tool)\b|"
    r"(?:不要|不需要|无需|不生成|别|避免|排除|跳过).{0,50}(政策|policy).*?(报告|卡片|工具)",
    re.IGNORECASE,
)

_EXPLICIT_ENVIRONMENT_REPORT_RE = re.compile(
    r"\b(standalone|separate|dedicated|single|fixed)\s+"
    r"(?:environment(?:al)?[-\s]?health|marine[-\s]?health|health[-\s]?assessment)\s+"
    r"(?:report|card|tool)\b|"
    r"\b(?:environment(?:al)?[-\s]?health|marine[-\s]?health|health[-\s]?assessment)\s+"
    r"(?:report|card|report[-\s]?card|tool)\b|"
    r"环境健康.*?(报告|卡片|报告卡|工具)|海洋健康.*?(报告|卡片|报告卡|工具)|"
    r"固定.*?(环境健康|海洋健康).*?(报告|卡片|报告卡|工具)",
    re.IGNORECASE,
)

_NEGATED_ENVIRONMENT_REPORT_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|without|no|not|avoid|exclude|skip|never)\b.{0,100}"
    r"\b(?:fixed\s+)?(?:environment(?:al)?[-\s]?health|marine[-\s]?health|health[-\s]?assessment)\s+"
    r"(?:report|card|report[-\s]?card|tool)\b|"
    r"(?:不要|不需要|无需|不生成|别|避免|排除|跳过).{0,60}"
    r"(环境健康|海洋健康).*?(报告|卡片|报告卡|工具)",
    re.IGNORECASE,
)

_ENV_HEALTH_BRANCH_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "sst_trend": {
        "name": "sst_trend",
        "indicator_label": "Sea-Surface Temperature Trend",
        "role": "risk_factor",
        "evidence_kind": "trend",
        "worse_when": "increase",
        "result_ref": "$ref:sst_trend",
        "required_results": ["sst_field", "sst_timeseries", "sst_trend"],
    },
    "bloom_frequency_change": {
        "name": "bloom_frequency_change",
        "indicator_label": "Bloom-Event Frequency Change",
        "role": "primary_support",
        "evidence_kind": "event_statistics",
        "worse_when": "increase",
        "result_ref": "$ref:bloom_statistics",
        "required_results": ["bloom_field", "bloom_detection", "bloom_statistics"],
    },
    "bloom_burden": {
        "name": "bloom_burden",
        "indicator_label": "Bloom Chlorophyll Burden",
        "role": "primary_support",
        "evidence_kind": "event_spatial_field",
        "metric": "total",
        "worse_when": "presence",
        "result_ref": "$ref:bloom_chlorophyll_burden",
        "required_results": ["bloom_field", "bloom_detection", "bloom_chlorophyll_burden"],
    },
    "bloom_event_days": {
        "name": "bloom_event_days",
        "indicator_label": "Bloom Event Days",
        "role": "primary_support",
        "evidence_kind": "event_spatial_field",
        "metric": "total",
        "worse_when": "presence",
        "result_ref": "$ref:bloom_event_days",
        "required_results": ["bloom_field", "bloom_detection", "bloom_event_days"],
    },
    "bottom_oxygen_trend": {
        "name": "bottom_oxygen_trend",
        "indicator_label": "Bottom-Oxygen Trend",
        "role": "primary_support",
        "evidence_kind": "trend",
        "worse_when": "decrease",
        "result_ref": "$ref:bottom_oxygen_trend",
        "required_results": ["bottom_oxygen_field", "bottom_oxygen_timeseries", "bottom_oxygen_trend"],
    },
    "bottom_hypoxia_burden": {
        "name": "bottom_hypoxia_burden",
        "indicator_label": "Bottom Hypoxia Oxygen-Deficit Burden",
        "role": "primary_endpoint",
        "evidence_kind": "event_spatial_field",
        "metric": "total",
        "worse_when": "presence",
        "result_ref": "$ref:hypoxia_oxygen_deficit_burden",
        "required_results": ["bottom_oxygen_field", "hypoxia_detection", "hypoxia_oxygen_deficit_burden"],
    },
    "hypoxic_days": {
        "name": "hypoxic_days",
        "indicator_label": "Hypoxic Days",
        "role": "primary_endpoint",
        "evidence_kind": "event_spatial_field",
        "metric": "total",
        "worse_when": "presence",
        "result_ref": "$ref:hypoxic_days",
        "required_results": ["bottom_oxygen_field", "hypoxia_detection", "hypoxic_days"],
    },
    "hypoxia_statistics": {
        "name": "hypoxia_statistics",
        "indicator_label": "Hypoxia Event Statistics",
        "role": "primary_endpoint",
        "evidence_kind": "event_statistics",
        "worse_when": "increase",
        "result_ref": "$ref:hypoxia_statistics",
        "required_results": ["bottom_oxygen_field", "hypoxia_detection", "hypoxia_statistics"],
    },
    "stratification_strength_change": {
        "name": "stratification_strength_change",
        "indicator_label": "Stratification-Strength Change",
        "role": "risk_factor",
        "evidence_kind": "trend",
        "worse_when": "increase",
        "result_ref": "$ref:stratification_trend",
        "required_results": [
            "temp_field",
            "salt_field",
            "thermo_dataset",
            "density_field",
            "stability_timeseries",
            "stratification_trend",
        ],
    },
    "eutrophication_context": {
        "name": "eutrophication_context",
        "indicator_label": "Chlorophyll / Eutrophication Screening Context",
        "role": "auxiliary_context",
        "evidence_kind": "trend",
        "worse_when": "increase",
        "result_ref": "$ref:chlorophyll_context_trend",
        "required_results": ["chlorophyll_context_field", "chlorophyll_context_timeseries", "chlorophyll_context_trend"],
    },
    "heatwave_burden": {
        "name": "heatwave_burden",
        "indicator_label": "Marine Heatwave Burden",
        "role": "primary_endpoint",
        "evidence_kind": "event_spatial_field",
        "metric": "total",
        "worse_when": "presence",
        "result_ref": "$ref:heatwave_burden",
        "required_results": ["heatwave_field", "heatwave_detection", "heatwave_burden"],
    },
    "heatwave_days": {
        "name": "heatwave_days",
        "indicator_label": "Marine Heatwave Days",
        "role": "primary_endpoint",
        "evidence_kind": "event_spatial_field",
        "metric": "total",
        "worse_when": "presence",
        "result_ref": "$ref:heatwave_days",
        "required_results": ["heatwave_field", "heatwave_detection", "heatwave_days"],
    },
    "upwelling_days": {
        "name": "upwelling_days",
        "indicator_label": "Upwelling Days",
        "role": "primary_endpoint",
        "evidence_kind": "event_spatial_field",
        "metric": "total",
        "worse_when": "presence",
        "result_ref": "$ref:upwelling_days",
        "required_results": ["upwelling_field", "upwelling_detection", "upwelling_days"],
    },
}


class SkillPlanner:
    """Generate execution plans from raw `SKILL.md` documents using the configured LLM."""

    LEGACY_DEFAULT_MODEL = LEGACY_ANTHROPIC_MODEL
    PLAN_GENERATION_MIN_TOKENS = 6000
    JSON_RETRY_MIN_TOKENS = 6000
    DEFAULT_MODEL = load_model_name(
        "PLANNER_MODEL",
        default=DEFAULT_OPENAI_MODEL,
        legacy_default=LEGACY_DEFAULT_MODEL,
    )
    _get_client = get_client
    _create_message = create_message
    _extract_response_text = staticmethod(extract_response_text)
    _parse_json_response = staticmethod(parse_json_response)
    _looks_like_json_parse_error = staticmethod(looks_like_json_parse_error)
    _extract_named_region_mentions = staticmethod(_extract_named_region_mentions_impl)
    resolve_named_region_entities = staticmethod(_resolve_named_region_entities_impl)
    _region_alias_matches = staticmethod(_region_alias_matches_impl)
    _coerce_numeric_range = staticmethod(_coerce_numeric_range_impl)
    _named_region_extent_suspicion = staticmethod(_named_region_extent_suspicion_impl)
    _bounds_are_reasonable_for_known_region = staticmethod(_named_region_bounds_are_reasonable_for_known_region)
    _bounds_too_broad_for_known_region = staticmethod(_named_region_bounds_too_broad_for_known_region)
    _bounds_nearly_equal = staticmethod(_named_region_bounds_nearly_equal)
    _ranges_nearly_equal = staticmethod(_named_region_ranges_nearly_equal)
    _ranges_overlap = staticmethod(_named_region_ranges_overlap)
    _bounds_area = staticmethod(_named_region_bounds_area)
    _format_region_list = staticmethod(_format_region_list_impl)
    _format_spatial_bounds = staticmethod(_format_spatial_bounds_impl)

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        skills_root: Optional[str] = None,
        client: Optional[Any] = None,
        trust_env: bool = False,
        base_url: Optional[str] = None,
        request_retries: int = 2,
    ):
        self.api_key = api_key
        self.model = model
        self.skills_root = skills_root
        self.client = client
        self.trust_env = trust_env
        self.base_url = base_url
        self.request_retries = request_retries
        self._adapter = OpenAICompatibleClientAdapter(
            api_key=api_key,
            base_url=base_url,
            model=model,
            client=client,
            trust_env=trust_env,
            request_retries=request_retries,
        )

    def generate_plan(
        self,
        skill_id: str,
        user_request: str,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 6000,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Load a skill markdown file and ask the configured LLM to generate a plan.
        """
        skill_markdown = load_skill_markdown(skill_id, self.skills_root)
        return self.generate_plan_from_markdown(
            skill_id=skill_id,
            skill_markdown=skill_markdown,
            user_request=user_request,
            extracted_params=extracted_params,
            additional_context=additional_context,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def generate_plan_from_markdown(
        self,
        skill_id: str,
        skill_markdown: str,
        user_request: str,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        upstream_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
        max_tokens: int = 6000,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Ask the configured LLM to generate a plan from raw skill markdown.

        Draft plans are normalized and deterministically validated first.  The
        focused reviewer is reserved for validation failures that cannot be
        repaired by the canonical validation path.
        """
        client = self._get_client()
        tool_contracts = self._collect_tool_contracts(skill_markdown, skill_id=skill_id)
        last_error: Optional[str] = None
        last_failed_plan: Optional[Dict[str, Any]] = None

        for attempt in range(1 + max_retries):
            if upstream_outputs:
                messages = self._build_messages_with_upstream(
                    skill_id=skill_id,
                    skill_markdown=skill_markdown,
                    user_request=user_request,
                    extracted_params=extracted_params or {},
                    additional_context=additional_context or {},
                    tool_contracts=tool_contracts,
                    upstream_outputs=upstream_outputs,
                )
            else:
                messages = self._build_messages(
                    skill_id=skill_id,
                    skill_markdown=skill_markdown,
                    user_request=user_request,
                    extracted_params=extracted_params or {},
                    additional_context=additional_context or {},
                    tool_contracts=tool_contracts,
                )

            if last_error is not None:
                messages["messages"].append({
                    "role": "assistant",
                    "content": "[previous attempt produced invalid plan]",
                })
                if self._looks_like_json_parse_error(last_error):
                    retry_content = (
                        "Your previous response was not valid JSON and could not be reviewed. "
                        "Return one complete syntactically valid JSON object only, with no markdown fences, "
                        "comments, trailing commas, or explanatory prose. Keep the plan compact: include only "
                        "status, skill_id, skills_used, steps, missing_fields, and clarification_question. "
                        "Each step must still include step_id, tool, params, and save_as, using the canonical "
                        "save_as ids from planner_contract_packet when present."
                    )
                else:
                    retry_content = (
                        f"Your previous plan was rejected due to the following validation error:\n"
                        f"{last_error}\n\n"
                        f"Rejected reviewed plan:\n"
                        f"{self._truncate_text(json.dumps(last_failed_plan or {}, ensure_ascii=False, indent=2), 5000)}\n\n"
                        f"Please regenerate the plan, fixing the issue above. "
                        f"You remain responsible for returning a complete executable plan with all tool chains, "
                        f"arguments, save_as ids, and $ref links. "
                        f"Use planner_contract_packet.required_environment_health_branch_contracts and "
                        f"planner_contract_packet.canonical_tool_chain_requirements when present. "
                        f"Return syntactically valid JSON with all required commas. "
                        f"Use ONLY tool names from tool_contracts keys."
                    )
                messages["messages"].append({
                    "role": "user",
                    "content": retry_content,
                })

            attempt_max_tokens = max(max_tokens, self.PLAN_GENERATION_MIN_TOKENS)
            if last_error is not None and self._looks_like_json_parse_error(last_error):
                attempt_max_tokens = max(attempt_max_tokens, self.JSON_RETRY_MIN_TOKENS)

            response = self._create_message(
                client=client,
                max_tokens=attempt_max_tokens,
                temperature=temperature,
                system=messages["system"],
                messages=messages["messages"],
                request_name=f"generate_plan:{skill_id}",
                json_response=True,
            )

            response_text = self._extract_response_text(response)
            plan: Optional[Dict[str, Any]] = None
            try:
                plan = self._parse_json_response(response_text)
                plan = self._normalize_plan_shape(plan)
                if plan.get("status", "ready") != "clarification_needed":
                    plan["skill_id"] = skill_id
                self._validate_plan_for_execution(
                    plan,
                    expected_skill_id=skill_id,
                    skill_markdowns={skill_id: skill_markdown},
                    user_request=user_request,
                    extracted_params=extracted_params or {},
                    additional_context=additional_context or {},
                )
                return plan
            except ValueError as exc:
                draft_plan = plan if "plan" in locals() and isinstance(plan, dict) else None
                if draft_plan is not None:
                    try:
                        reviewed_plan = self._review_plan_after_validation_error(
                            user_request=user_request,
                            draft_plan=draft_plan,
                            validation_error=str(exc),
                            skill_markdowns={skill_id: skill_markdown},
                            expected_skill_id=skill_id,
                            extracted_params=extracted_params or {},
                            additional_context=additional_context or {},
                            max_review_retries=max_retries,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        return reviewed_plan
                    except ValueError as review_exc:
                        last_failed_plan = draft_plan
                        last_error = str(review_exc)
                else:
                    last_failed_plan = draft_plan
                    last_error = str(exc)
                if attempt >= max_retries:
                    raise ValueError(self._friendly_plan_validation_error(last_error)) from exc

        raise ValueError("Plan generation failed after retries.")

    def generate_plan_for_query(
        self,
        user_request: str,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 6000,
        temperature: float = 0.0,
        allow_multiple_skills: bool = False,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Ask the configured LLM to choose the best skill and generate a plan from the user query.

        The skill-selection round retries up to *max_retries* times on validation
        failure.  Downstream plan generation uses its own retry budget.
        """
        client = self._get_client()
        skill_markdowns = load_all_skill_markdowns(self.skills_root)
        if not skill_markdowns:
            raise ValueError("No SKILL.md documents were found.")

        approved_skill_ids = self._approved_proposal_skill_ids(additional_context or {}, skill_markdowns)
        if approved_skill_ids:
            primary_skill_id = approved_skill_ids[0]
            if len(approved_skill_ids) == 1:
                plan = self.generate_plan_from_markdown(
                    skill_id=primary_skill_id,
                    skill_markdown=skill_markdowns[primary_skill_id],
                    user_request=user_request,
                    extracted_params=extracted_params,
                    additional_context=additional_context,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                plan["skills_used"] = [primary_skill_id]
                return plan
            selected_markdowns = {
                skill_id: skill_markdowns[skill_id]
                for skill_id in approved_skill_ids
            }
            return self.generate_multi_skill_plan(
                primary_skill_id=primary_skill_id,
                selected_skill_ids=approved_skill_ids,
                skill_markdowns=selected_markdowns,
                user_request=user_request,
                extracted_params=extracted_params,
                additional_context=additional_context,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        skill_hints = load_all_planner_hints(self.skills_root)
        skill_briefs = {
            skill_id: self._build_skill_brief(markdown, skill_hints.get(skill_id))
            for skill_id, markdown in skill_markdowns.items()
        }

        last_error: Optional[str] = None

        for attempt in range(1 + max_retries):
            messages = self._build_query_selection_messages(
                skill_briefs=skill_briefs,
                skill_hints=skill_hints,
                user_request=user_request,
                extracted_params=extracted_params or {},
                additional_context=additional_context or {},
                allow_multiple_skills=allow_multiple_skills,
            )

            if last_error is not None:
                messages["messages"].append({
                    "role": "assistant",
                    "content": "[previous attempt produced invalid selection]",
                })
                messages["messages"].append({
                    "role": "user",
                    "content": (
                        f"Your previous skill selection was rejected due to the following validation error:\n"
                        f"{last_error}\n\n"
                        f"Please regenerate the selection, fixing the issue above. "
                        f"Return syntactically valid JSON with all required commas."
                    ),
                })

            response = self._create_message(
                client=client,
                max_tokens=max_tokens,
                temperature=temperature,
                system=messages["system"],
                messages=messages["messages"],
                request_name="generate_plan_for_query",
                json_response=True,
            )

            response_text = self._extract_response_text(response)
            try:
                selection = self._parse_json_response(response_text)
                selection = self._normalize_plan_shape(selection)
                self._validate_skill_selection_shape(selection)
                if selection.get("status") == "clarification_needed":
                    return selection
                self._validate_known_skills(selection, skill_markdowns)
                selection = self._reroute_policy_selection_for_environment_evidence(
                    selection,
                    user_request=user_request,
                    skill_markdowns=skill_markdowns,
                )
                self._validate_known_skills(selection, skill_markdowns)
                break  # selection is valid
            except ValueError as exc:
                last_error = str(exc)
                if attempt >= max_retries:
                    raise ValueError(last_error) from exc

        selected_skill_ids = self._get_skills_used(selection)
        primary_skill_id = selection["skill_id"]
        selected_markdowns = {
            skill_id: skill_markdowns[skill_id]
            for skill_id in selected_skill_ids
        }

        if len(selected_skill_ids) == 1:
            plan = self.generate_plan_from_markdown(
                skill_id=primary_skill_id,
                skill_markdown=selected_markdowns[primary_skill_id],
                user_request=user_request,
                extracted_params=extracted_params,
                additional_context=additional_context,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            plan["skills_used"] = [primary_skill_id]
            return plan

        return self.generate_multi_skill_plan(
            primary_skill_id=primary_skill_id,
            selected_skill_ids=selected_skill_ids,
            skill_markdowns=selected_markdowns,
            user_request=user_request,
            extracted_params=extracted_params,
            additional_context=additional_context,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _reroute_policy_selection_for_environment_evidence(
        self,
        selection: Dict[str, Any],
        *,
        user_request: str,
        skill_markdowns: Dict[str, str],
    ) -> Dict[str, Any]:
        if selection.get("status", "ready") != "ready":
            return selection
        if "ocean_environment_health_assessment" not in skill_markdowns:
            return selection

        skills_used = self._get_skills_used(selection)
        if set(skills_used) != {"ocean_policy_recommendation"}:
            return selection
        if not self._policy_query_requires_new_environment_evidence(user_request):
            return selection

        rerouted = dict(selection)
        rerouted["skill_id"] = "ocean_environment_health_assessment"
        rerouted["skills_used"] = ["ocean_environment_health_assessment"]
        return rerouted

    @staticmethod
    def _policy_query_requires_new_environment_evidence(user_request: str) -> bool:
        lowered = (user_request or "").lower()
        if not lowered.strip():
            return False

        policy_intent = re.search(
            r"\b(?:policy|policies|management|recommendation|recommendations|monitoring|"
            r"zoning|zone|zones|protection|protect|priority|priorities|governance|"
            r"nutrient[-\s]?control|pollution[-\s]?reduction|pollution[-\s]?control|"
            r"ecological[-\s]?protection|environmental[-\s]?governance|decision support)\b|"
            r"政策|管理|建议|监测|分区|优先区|保护|治理|营养盐控制|污染削减|污染控制|生态保护",
            lowered,
        )
        if not policy_intent:
            return False

        existing_evidence_anchor = re.search(
            r"\b(?:these|those|above|previous|prior|existing|completed|computed|already|detected)\b"
            r".{0,80}\b(?:result|results|evidence|analysis|assessment|event|events|map|maps|trend|trends|diagnostic|diagnostics)\b|"
            r"\b(?:based on|from|after)\b.{0,40}\b(?:existing|completed|computed|detected|above|previous|prior)\b"
            r".{0,80}\b(?:result|results|evidence|analysis|assessment|event|events|map|maps|trend|trends|diagnostic|diagnostics)\b|"
            r"\bturn these\b.{0,80}\b(?:result|results|evidence|analysis|assessment|event|events)\b",
            lowered,
        )
        if existing_evidence_anchor:
            return False

        evidence_metric = re.search(
            r"\b(?:chlorophyll|chla|bloom|bloom[-\s]?event|bloom[-\s]?affected|"
            r"hypoxia|hypoxic|low[-\s]?oxygen|oxygen[-\s]?deficit|heatwave|"
            r"stratification|warming|sst|water[-\s]?quality|hotspot|hotspots|"
            r"event[-\s]?days?|burden|area|trend|trends|year[-\s]?by[-\s]?year)\b|"
            r"叶绿素|藻华|低氧|缺氧|热浪|分层|水质|热点|负担|面积|趋势|逐年",
            lowered,
        )
        if not evidence_metric:
            return False

        new_analysis_intent = re.search(
            r"\b(?:use|using|assess|analy[sz]e|compute|calculate|detect|identify|map|"
            r"show|check|compare|changes?|changed|worsen(?:ed|ing)?|expand(?:ed|ing)?|"
            r"which areas?|where|during|from|in)\b|"
            r"评估|分析|计算|检测|识别|绘制|比较|变化|恶化|扩大|哪里|区域|期间",
            lowered,
        )
        return bool(new_analysis_intent)

    def generate_multi_skill_plan(
        self,
        primary_skill_id: str,
        selected_skill_ids: List[str],
        skill_markdowns: Dict[str, str],
        user_request: str,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 6000,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generate per-skill plans sequentially and compose them into one flat plan.

        If a non-primary sub-skill returns clarification_needed, fall back to
        generating the entire plan from the primary skill alone (single-skill
        path), since meta-skills like mechanism_ranking and evidence_synthesis
        contain complete upstream tool-call chains in their SKILL.md.
        """
        all_steps: List[Dict[str, Any]] = []
        upstream_outputs: Dict[str, Dict[str, Any]] = {}
        save_as_registry: set[str] = set()

        for skill_id in selected_skill_ids:
            sub_plan = self.generate_plan_from_markdown(
                skill_id=skill_id,
                skill_markdown=skill_markdowns[skill_id],
                user_request=user_request,
                extracted_params=extracted_params,
                additional_context=additional_context,
                upstream_outputs=upstream_outputs,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if sub_plan.get("status") == "clarification_needed":
                if skill_id == primary_skill_id:
                    # Primary skill itself needs clarification — propagate.
                    clarification = dict(sub_plan)
                    clarification["skill_id"] = primary_skill_id
                    clarification["skills_used"] = selected_skill_ids
                    clarification["clarification_source_skill"] = skill_id
                    return clarification
                # Non-primary sub-skill returned clarification. Fall back to
                # single-skill plan from the primary skill which contains
                # complete upstream tool chains in its SKILL.md.
                primary_plan = self.generate_plan_from_markdown(
                    skill_id=primary_skill_id,
                    skill_markdown=skill_markdowns[primary_skill_id],
                    user_request=user_request,
                    extracted_params=extracted_params,
                    additional_context=additional_context,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if primary_plan.get("status") == "clarification_needed":
                    clarification = dict(primary_plan)
                    clarification["skill_id"] = primary_skill_id
                    clarification["skills_used"] = selected_skill_ids
                    clarification["clarification_source_skill"] = primary_skill_id
                    return clarification
                primary_plan["skills_used"] = [primary_skill_id]
                self._validate_plan_for_execution(
                    primary_plan,
                    expected_skill_id=primary_skill_id,
                    skill_markdowns={primary_skill_id: skill_markdowns[primary_skill_id]},
                    user_request=user_request,
                    extracted_params=extracted_params or {},
                    additional_context=additional_context or {},
                )
                return primary_plan

            sub_steps = [dict(step) for step in sub_plan.get("steps", [])]
            rename_map = self._dedupe_step_save_as_names(
                skill_id=skill_id,
                steps=sub_steps,
                existing_names=save_as_registry,
            )
            if rename_map:
                sub_steps = self._rewrite_refs_in_steps(sub_steps, rename_map)

            for step in sub_steps:
                save_as = step["save_as"]
                save_as_registry.add(save_as)
                upstream_outputs[save_as] = {
                    "tool": step["tool"],
                    "output_type": get_tool_output_type(step["tool"]) or "generic_result",
                    "skill_origin": skill_id,
                }

            all_steps.extend(sub_steps)

        plan = {
            "status": "ready",
            "skill_id": primary_skill_id,
            "skills_used": selected_skill_ids,
            "steps": all_steps,
        }
        self._validate_plan_for_execution(
            plan,
            expected_skill_id=primary_skill_id,
            skill_markdowns=skill_markdowns,
            user_request=user_request,
            extracted_params=extracted_params or {},
            additional_context=additional_context or {},
        )
        return plan

    def review_execution(
        self,
        user_request: str,
        active_plan: Dict[str, Any],
        completed_steps: List[Dict[str, Any]],
        remaining_steps: List[Dict[str, Any]],
        last_event: Dict[str, Any],
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Review the current execution state and decide whether to continue,
        replan, ask the user, or abort.
        """
        client = self._get_client()
        skill_markdowns = load_all_skill_markdowns(self.skills_root)
        if not skill_markdowns:
            raise ValueError("No SKILL.md documents were found.")

        active_skill_contracts: Dict[str, Dict[str, Any]] = {}
        for skill_id in self._get_skills_used(active_plan):
            markdown = skill_markdowns.get(skill_id)
            if markdown:
                active_skill_contracts.update(self._collect_tool_contracts(markdown, skill_id=skill_id))
        messages = self._build_review_messages(
            user_request=user_request,
            active_plan=active_plan,
            completed_steps=completed_steps,
            remaining_steps=remaining_steps,
            last_event=last_event,
            extracted_params=extracted_params or {},
            additional_context=additional_context or {},
            skill_contracts=active_skill_contracts,
        )

        response = self._create_message(
            client=client,
            max_tokens=max_tokens,
            temperature=temperature,
            system=messages["system"],
            messages=messages["messages"],
            request_name="review_execution",
            json_response=True,
        )

        response_text = self._extract_response_text(response)
        decision = self._parse_json_response(response_text)
        self._validate_review_shape(decision)

        if decision["decision"] == "replan":
            updated_plan = decision.get("updated_plan")
            if not isinstance(updated_plan, dict):
                raise ValueError("Replan decision must include an 'updated_plan' object.")
            updated_plan = self._normalize_plan_shape(updated_plan)
            decision["updated_plan"] = updated_plan
            self._validate_plan_for_execution(
                updated_plan,
                expected_skill_id=None,
                skill_markdowns=skill_markdowns,
                user_request=user_request,
                extracted_params=extracted_params or {},
                additional_context=additional_context or {},
                available_result_types=self._completed_step_result_types(completed_steps),
            )

        return decision

    def _review_draft_plan(
        self,
        *,
        user_request: str,
        draft_plan: Dict[str, Any],
        skill_markdowns: Dict[str, str],
        expected_skill_id: Optional[str],
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Focused plan reviewer retained for explicit or diagnostic use.

        The reviewer receives a compact review packet, not full SKILL.md, and
        returns the complete plan to validate.
        """
        candidate = self._call_focused_plan_reviewer(
            user_request=user_request,
            draft_plan=draft_plan,
            validation_error=None,
            skill_markdowns=skill_markdowns,
            extracted_params=extracted_params,
            additional_context=additional_context,
            max_tokens=max_tokens,
            temperature=temperature,
            request_name="review_draft_plan",
        )
        if candidate.get("status", "ready") != "clarification_needed" and expected_skill_id:
            candidate.setdefault("skill_id", expected_skill_id)
        return self._canonicalize_reviewed_plan_for_validation(
            candidate,
            expected_skill_id=expected_skill_id,
            user_request=user_request,
        )

    def _review_plan_after_validation_error(
        self,
        *,
        user_request: str,
        draft_plan: Dict[str, Any],
        validation_error: str,
        skill_markdowns: Dict[str, str],
        expected_skill_id: Optional[str],
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
        max_review_retries: int,
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Focused reviewer retry used after deterministic validation errors."""
        current_plan = draft_plan
        last_error = validation_error

        for _attempt in range(1 + max(0, max_review_retries)):
            candidate: Optional[Dict[str, Any]] = None
            try:
                candidate = self._call_focused_plan_reviewer(
                    user_request=user_request,
                    draft_plan=current_plan,
                    validation_error=last_error,
                    skill_markdowns=skill_markdowns,
                    extracted_params=extracted_params,
                    additional_context=additional_context,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    request_name="review_plan_after_validation_error",
                )
                if candidate.get("status", "ready") != "clarification_needed" and expected_skill_id:
                    candidate.setdefault("skill_id", expected_skill_id)
                candidate = self._canonicalize_reviewed_plan_for_validation(
                    candidate,
                    expected_skill_id=expected_skill_id,
                    user_request=user_request,
                )
                self._validate_plan_for_execution(
                    candidate,
                    expected_skill_id=expected_skill_id,
                    skill_markdowns=skill_markdowns,
                    user_request=user_request,
                    extracted_params=extracted_params,
                    additional_context=additional_context,
                )
                return candidate
            except ValueError as exc:
                last_error = str(exc)
                if isinstance(candidate, dict):
                    current_plan = candidate

        raise ValueError(last_error)

    def _call_focused_plan_reviewer(
        self,
        *,
        user_request: str,
        draft_plan: Dict[str, Any],
        validation_error: Optional[str],
        skill_markdowns: Dict[str, str],
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
        max_tokens: int,
        temperature: float,
        request_name: str,
    ) -> Dict[str, Any]:
        client = self._get_client()
        messages = self._build_plan_review_messages(
            user_request=user_request,
            draft_plan=draft_plan,
            validation_error=validation_error,
            skill_markdowns=skill_markdowns,
            extracted_params=extracted_params,
            additional_context=additional_context,
        )
        response = self._create_message(
            client=client,
            max_tokens=max_tokens,
            temperature=temperature,
            system=messages["system"],
            messages=messages["messages"],
            request_name=request_name,
            json_response=True,
        )
        payload = self._parse_json_response(self._extract_response_text(response))
        candidate = self._extract_reviewed_plan_payload(payload)
        return self._normalize_plan_shape(candidate)

    @staticmethod
    def _extract_reviewed_plan_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Reviewer correction must be a JSON object.")
        for key in ("updated_plan", "revised_plan", "plan"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                return nested
        return payload

    def _canonicalize_reviewed_plan_for_validation(
        self,
        plan: Dict[str, Any],
        *,
        expected_skill_id: Optional[str],
        user_request: str,
        available_result_ids: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        """Normalize deterministic environment-health refs before strict validation.

        The reviewer still owns scientific scope and step selection. This pass only
        canonicalizes existing standard environment-health step arguments so small
        ref aliases like "$ref:bottom_oxygen_field" do not make an otherwise
        reviewed plan fail before execution.
        """
        if not isinstance(plan, dict) or plan.get("status", "ready") == "clarification_needed":
            return plan
        plan = self._dedupe_plan_save_as_ids(plan, reserved_names=available_result_ids)
        plan = self._repair_point_timeseries_plan(plan, user_request=user_request)
        plan = self._repair_general_bottom_vertical_plan(plan, user_request=user_request)
        skill_id = str(plan.get("skill_id") or expected_skill_id or "").strip()
        steps = plan.get("steps")
        has_hypoxia_step = isinstance(steps, list) and any(
            isinstance(step, dict) and step.get("tool") == "detect_hypoxia"
            for step in steps
        )
        skills_used = {
            str(skill).strip()
            for skill in plan.get("skills_used", [])
            if str(skill).strip()
        } if isinstance(plan.get("skills_used"), list) else set()
        uses_environment_health = (
            skill_id == "ocean_environment_health_assessment"
            or "ocean_environment_health_assessment" in skills_used
        )
        if not uses_environment_health and not has_hypoxia_step:
            return plan
        if self._environment_health_policy_only_request(user_request):
            return plan
        if not uses_environment_health and has_hypoxia_step:
            return self._canonicalize_hypoxia_standard_step_params(
                plan,
                available_result_ids=available_result_ids,
            )
        return self._canonicalize_environment_health_standard_step_params(
            plan,
            user_request=user_request,
            available_result_ids=available_result_ids,
        )

    @staticmethod
    def _dedupe_plan_save_as_ids(
        plan: Dict[str, Any],
        *,
        reserved_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Make step result ids unique while preserving in-order $ref meaning."""
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan

        reserved_save_as = {
            name
            for name in (reserved_names or set())
            if isinstance(name, str) and name
        }
        active_ref_names: Dict[str, str] = {}
        repaired_steps: List[Dict[str, Any]] = []
        changed = False

        for step in steps:
            if not isinstance(step, dict):
                repaired_steps.append(step)
                continue

            repaired_step = dict(step)
            rewritten_params = SkillPlanner._rewrite_refs_in_value_static(
                repaired_step.get("params"),
                active_ref_names,
            )
            if rewritten_params != repaired_step.get("params"):
                repaired_step["params"] = rewritten_params
                changed = True

            save_as = repaired_step.get("save_as")
            if isinstance(save_as, str) and save_as:
                if save_as in reserved_save_as:
                    unique_save_as = SkillPlanner._unique_name(save_as, reserved_save_as)
                    repaired_step["save_as"] = unique_save_as
                    active_ref_names[save_as] = unique_save_as
                    reserved_save_as.add(unique_save_as)
                    changed = True
                else:
                    reserved_save_as.add(save_as)
                    active_ref_names[save_as] = save_as

            repaired_steps.append(repaired_step)

        if not changed:
            return plan
        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    def _canonicalize_plan_in_place_for_validation(
        self,
        plan: Dict[str, Any],
        *,
        expected_skill_id: Optional[str],
        user_request: str,
        available_result_ids: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        canonical = self._canonicalize_reviewed_plan_for_validation(
            plan,
            expected_skill_id=expected_skill_id,
            user_request=user_request,
            available_result_ids=available_result_ids,
        )
        if isinstance(canonical, dict) and canonical is not plan:
            plan.clear()
            plan.update(canonical)
        return plan

    def _build_planner_contract_packet(
        self,
        *,
        skill_id: str,
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compact machine-readable contract for planner-owned executable plans."""
        if skill_id != "ocean_environment_health_assessment":
            return {
                "planner_role": (
                    "Generate the full executable tool plan from the selected skill and tool_contracts. "
                    "No extra canonical environment-health evidence contracts apply to this skill."
                ),
                "skill_id": skill_id,
                "mode": "generic_direct_tool_plan",
            }

        contract_plan = {
            "status": "ready",
            "skill_id": skill_id,
            "skills_used": [skill_id],
            "steps": [],
        }
        tool_chain_requirements = self._build_tool_chain_requirements(
            user_request=user_request,
            draft_plan=contract_plan,
            additional_context=additional_context,
        )
        relevant_tool_names = self._review_relevant_tool_names(
            draft_plan=contract_plan,
            tool_chain_requirements=tool_chain_requirements,
            skill_markdowns={},
        )
        relevant_tool_argument_schemas = {
            tool_name: self._build_minimal_tool_contract(tool_name)
            for tool_name in sorted(relevant_tool_names)
            if get_tool_contract(tool_name)
        }
        branch_contracts: Dict[str, Dict[str, Any]] = {}
        requested_branches: List[str] = []
        suppressed_branches: List[str] = []
        if skill_id == "ocean_environment_health_assessment":
            requested = self._environment_health_requested_branch_keys(user_request)
            suppressed = self._environment_health_suppressed_branch_keys(user_request)
            requested.difference_update(suppressed)
            supported = self._filter_environment_health_branches_by_dataset_support(
                requested,
                additional_context=additional_context,
            )
            requested_branches = self._environment_health_branch_order(supported)
            suppressed_branches = self._environment_health_branch_order(suppressed)
            branch_contracts = {
                branch_key: dict(_ENV_HEALTH_BRANCH_CONTRACTS[branch_key])
                for branch_key in requested_branches
                if branch_key in _ENV_HEALTH_BRANCH_CONTRACTS
            }

        named_regions = self._extract_named_region_mentions(user_request)
        return {
            "planner_role": (
                "Generate the full executable tool plan. Do not return only objectives. "
                "Do not rely on reviewer or backend repair to add canonical chains."
            ),
            "skill_id": skill_id,
            "dataset_capability_summary": self._build_dataset_capability_summary(additional_context),
            "named_regions": named_regions,
            "known_named_region_bounds": {
                region_name: _KNOWN_NAMED_REGION_BOUNDS[region_name]
                for region_name in named_regions
                if region_name in _KNOWN_NAMED_REGION_BOUNDS
            },
            "has_explicit_time_window": self._request_has_explicit_time_window(
                user_request=user_request,
                extracted_params=extracted_params,
                additional_context=additional_context,
            ),
            "time_range_hint": self._review_time_range_hint(
                user_request=user_request,
                extracted_params=extracted_params,
                additional_context=additional_context,
            ),
            "requested_environment_health_branches": requested_branches,
            "suppressed_environment_health_branches": suppressed_branches,
            "required_environment_health_branch_contracts": branch_contracts,
            "canonical_tool_chain_requirements": tool_chain_requirements,
            "relevant_tool_argument_schemas": relevant_tool_argument_schemas,
            "hard_validation_rules": self._focused_review_validation_rules(user_request),
            "explicit_fixed_report_allowed": self._explicit_fixed_report_requested(user_request),
            "result_id_rule": (
                "When a branch contract lists required_results, the executable steps must produce those exact save_as ids "
                "unless an earlier completed result with that id is explicitly available."
            ),
        }

    def _build_plan_review_messages(
        self,
        *,
        user_request: str,
        draft_plan: Dict[str, Any],
        validation_error: Optional[str],
        skill_markdowns: Dict[str, str],
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        review_packet = self._build_focused_plan_review_packet(
            user_request=user_request,
            draft_plan=draft_plan,
            validation_error=validation_error,
            skill_markdowns=skill_markdowns,
            extracted_params=extracted_params,
            additional_context=additional_context,
        )

        system_prompt = (
            "You are a focused plan auditor, not the primary planner.\n"
            "The planner already interpreted the user request and proposed a draft plan. "
            "Review only tool chains, tool arguments, result ids, refs, and deterministic validation rules. "
            "This reviewer is normally called only after deterministic validation failed. "
            "Make the smallest correction needed for executability; do not expand the scientific scope.\n"
            "Rules:\n"
            "1. Use only tools and parameter names present in review_packet.tool_argument_schemas.\n"
            "2. Use review_packet.tool_chain_requirements to add missing glue steps such as assemble_dataset.\n"
            "3. Include complete params for every returned step; do not rely on backend semantic repair.\n"
            "4. Preserve the user's intent from review_packet.user_intent_brief; do not expand the scientific scope.\n"
            "5. Fix review_packet.validation_error directly and preserve unrelated valid steps.\n"
            "6. Also fix obvious missing selected-skill contract steps only when required for validation.\n"
            "7. If a requested support chain is unsupported by dataset capabilities, omit that chain and keep executable evidence.\n"
            "8. Do not add fixed report tools unless explicit_fixed_report_allowed is true.\n"
            "9. Return one complete reviewed plan, not commentary.\n"
            "10. Return JSON only. No markdown fences."
        )

        user_payload = {
            "review_packet": review_packet,
            "required_output_schema": {
                "status": "ready | clarification_needed",
                "skill_id": "string",
                "skills_used": ["string"],
                "steps": [
                    {
                        "step_id": "string",
                        "tool": "string",
                        "params": "object with complete tool args",
                        "save_as": "string",
                    }
                ],
                "missing_fields": ["string"],
                "clarification_question": "string",
            },
        }

        return {
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
                }
            ],
        }

    def _build_focused_plan_review_packet(
        self,
        *,
        user_request: str,
        draft_plan: Dict[str, Any],
        validation_error: Optional[str],
        skill_markdowns: Dict[str, str],
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        tool_chain_requirements = self._build_tool_chain_requirements(
            user_request=user_request,
            draft_plan=draft_plan,
            additional_context=additional_context,
        )
        relevant_tool_names = self._review_relevant_tool_names(
            draft_plan=draft_plan,
            tool_chain_requirements=tool_chain_requirements,
            skill_markdowns=skill_markdowns,
        )
        tool_argument_schemas = {
            tool_name: self._build_minimal_tool_contract(tool_name)
            for tool_name in sorted(relevant_tool_names)
            if get_tool_contract(tool_name)
        }
        return {
            "user_intent_brief": self._build_user_intent_brief(
                user_request=user_request,
                extracted_params=extracted_params,
                additional_context=additional_context,
                draft_plan=draft_plan,
            ),
            "draft_plan": draft_plan,
            "validation_error": validation_error,
            "tool_chain_requirements": tool_chain_requirements,
            "tool_argument_schemas": tool_argument_schemas,
            "dataset_capability_summary": self._build_dataset_capability_summary(additional_context),
            "hard_validation_rules": self._focused_review_validation_rules(user_request),
            "explicit_fixed_report_allowed": self._explicit_fixed_report_requested(user_request),
        }

    def _build_user_intent_brief(
        self,
        *,
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
        draft_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        named_regions = self._extract_named_region_mentions(user_request)
        named_region_bounds = {
            region_name: _KNOWN_NAMED_REGION_BOUNDS[region_name]
            for region_name in named_regions
            if region_name in _KNOWN_NAMED_REGION_BOUNDS
        }
        requested_branches: List[str] = []
        if draft_plan.get("skill_id") == "ocean_environment_health_assessment":
            requested = self._environment_health_requested_branch_keys(user_request)
            requested.difference_update(self._environment_health_suppressed_branch_keys(user_request))
            requested_branches = self._environment_health_branch_order(requested)
        return {
            "request": self._truncate_text(user_request, 800),
            "skill_id": draft_plan.get("skill_id"),
            "skills_used": draft_plan.get("skills_used") or [draft_plan.get("skill_id")],
            "named_regions": named_regions,
            "known_named_region_bounds": named_region_bounds,
            "has_explicit_time_window": self._request_has_explicit_time_window(
                user_request=user_request,
                extracted_params=extracted_params,
                additional_context=additional_context,
            ),
            "time_range_hint": self._review_time_range_hint(
                user_request=user_request,
                extracted_params=extracted_params,
                additional_context=additional_context,
            ),
            "requested_environment_health_branches": requested_branches,
        }

    def _build_dataset_capability_summary(self, additional_context: Dict[str, Any]) -> Dict[str, Any]:
        variables = self._dataset_variable_names(additional_context)
        depth_count = self._dataset_depth_level_count(additional_context)
        dataset = additional_context.get("dataset") if isinstance(additional_context, dict) else None
        temporal_extent = dataset.get("temporal_extent") if isinstance(dataset, dict) else None
        spatial_extent = dataset.get("spatial_extent") if isinstance(dataset, dict) else None
        return {
            "variables": sorted(variables),
            "depth_level_count": depth_count,
            "has_temperature": self._has_any_dataset_variable(variables, {"temp", "temperature", "thetao"}) if variables else None,
            "has_salinity": self._has_any_dataset_variable(variables, {"salt", "salinity", "so"}) if variables else None,
            "has_oxygen": self._has_any_dataset_variable(variables, {"oxygen", "o2"}) if variables else None,
            "has_chlorophyll": self._has_any_dataset_variable(variables, {"chlorophyll", "chl", "chla"}) if variables else None,
            "supports_stratification": (
                self._has_any_dataset_variable(variables, {"temp", "temperature", "thetao"})
                and self._has_any_dataset_variable(variables, {"salt", "salinity", "so"})
                and (depth_count is None or depth_count > 1)
            ) if variables else None,
            "temporal_extent": temporal_extent if isinstance(temporal_extent, dict) else None,
            "spatial_extent": spatial_extent if isinstance(spatial_extent, dict) else None,
        }

    def _build_tool_chain_requirements(
        self,
        *,
        user_request: str,
        draft_plan: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if draft_plan.get("skill_id") != "ocean_environment_health_assessment":
            return self._generic_tool_chain_requirements(draft_plan)

        requested = self._environment_health_requested_branch_keys(user_request)
        requested.difference_update(self._environment_health_suppressed_branch_keys(user_request))
        supported = self._filter_environment_health_branches_by_dataset_support(
            requested,
            additional_context=additional_context,
        )
        requirements: List[Dict[str, Any]] = []
        unsupported = requested - supported
        for branch_key in self._environment_health_branch_order(unsupported):
            requirements.append({
                "objective": branch_key,
                "status": "unsupported_by_dataset_capabilities",
                "instruction": "Do not add this chain; keep executable evidence and let summary report the data gap.",
            })

        hypoxia_branches = {
            "bottom_oxygen_trend",
            "bottom_hypoxia_burden",
            "hypoxic_days",
            "hypoxia_statistics",
        }
        if supported & hypoxia_branches:
            requirements.append(self._environment_health_bottom_oxygen_hypoxia_chain(supported))
        if "sst_trend" in supported:
            requirements.append(self._environment_health_sst_chain())
        if "stratification_strength_change" in supported:
            requirements.append(self._environment_health_stratification_chain())
        if supported & {"bloom_frequency_change", "bloom_burden", "bloom_event_days", "eutrophication_context"}:
            requirements.append(self._environment_health_bloom_chain(supported))
        if supported & {"heatwave_burden", "heatwave_days"}:
            requirements.append(self._environment_health_heatwave_chain(supported))
        return requirements

    def _generic_tool_chain_requirements(self, draft_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        steps = draft_plan.get("steps")
        if not isinstance(steps, list):
            return []
        return [{
            "objective": "draft_plan_ref_integrity",
            "status": "review_only",
            "instruction": "Verify every $ref points to an earlier save_as id and every tool param matches its schema.",
            "tools": sorted({
                str(step.get("tool"))
                for step in steps
                if isinstance(step, dict) and isinstance(step.get("tool"), str)
            }),
        }]

    @staticmethod
    def _environment_health_bottom_oxygen_hypoxia_chain(supported: set[str]) -> Dict[str, Any]:
        steps = [
            {
                "tool": "load_dataset",
                "save_as": "bottom_oxygen_field",
                "params": {"variable": "oxygen", "vertical_mode": "bottom"},
            },
            {
                "tool": "compute_area_weighted_mean",
                "save_as": "bottom_oxygen_timeseries",
                "params": {"data": "$ref:bottom_oxygen_field.data", "depth_aggregation": "mean"},
            },
            {
                "tool": "compute_trend",
                "save_as": "bottom_oxygen_trend",
                "params": {"timeseries": "$ref:bottom_oxygen_timeseries"},
            },
            {
                "tool": "detect_hypoxia",
                "save_as": "hypoxia_detection",
                "params": {"oxygen": "$ref:bottom_oxygen_field.data", "vertical_mode": "bottom"},
            },
        ]
        if "hypoxia_statistics" in supported:
            steps.append({
                "tool": "compute_event_statistics",
                "save_as": "hypoxia_statistics",
                "params": {"events": "$ref:hypoxia_detection.events", "group_by": "year"},
            })
        if "hypoxic_days" in supported:
            steps.append({
                "tool": "compute_event_summary_map",
                "save_as": "hypoxic_days",
                "params": {
                    "event_detection": "$ref:hypoxia_detection",
                    "data": "$ref:bottom_oxygen_field.data",
                    "summary_mode": "event_days",
                },
            })
        if "bottom_hypoxia_burden" in supported:
            steps.append({
                "tool": "compute_event_summary_map",
                "save_as": "hypoxia_oxygen_deficit_burden",
                "params": {
                    "event_detection": "$ref:hypoxia_detection",
                    "data": "$ref:bottom_oxygen_field.data",
                    "summary_mode": "burden",
                },
            })
        return {
            "objective": "bottom_oxygen_and_hypoxia_evidence",
            "status": "required_when_requested",
            "canonical_steps": steps,
        }

    @staticmethod
    def _environment_health_sst_chain() -> Dict[str, Any]:
        return {
            "objective": "sst_trend",
            "status": "required_when_requested",
            "canonical_steps": [
                {"tool": "load_dataset", "save_as": "sst_field", "params": {"variable": "temp", "vertical_mode": "surface"}},
                {"tool": "compute_area_weighted_mean", "save_as": "sst_timeseries", "params": {"data": "$ref:sst_field.data", "depth_aggregation": "mean"}},
                {"tool": "compute_trend", "save_as": "sst_trend", "params": {"timeseries": "$ref:sst_timeseries"}},
            ],
        }

    @staticmethod
    def _environment_health_stratification_chain() -> Dict[str, Any]:
        return {
            "objective": "stratification_strength_change",
            "status": "required_when_requested_and_supported",
            "canonical_steps": [
                {"tool": "load_dataset", "save_as": "temp_field", "params": {"variable": "temp"}},
                {"tool": "load_dataset", "save_as": "salt_field", "params": {"variable": "salt"}},
                {
                    "tool": "assemble_dataset",
                    "save_as": "thermo_dataset",
                    "params": {
                        "variables": {
                            "temp": "$ref:temp_field.data",
                            "salt": "$ref:salt_field.data",
                        }
                    },
                },
                {"tool": "compute_density", "save_as": "density_field", "params": {"data": "$ref:thermo_dataset.data"}},
                {"tool": "compute_vertical_stability_timeseries", "save_as": "stability_timeseries", "params": {"density": "$ref:density_field.data"}},
                {"tool": "compute_trend", "save_as": "stratification_trend", "params": {"timeseries": "$ref:stability_timeseries"}},
            ],
            "note": "thermo_dataset is a canonical save_as result id produced by assemble_dataset, not a tool.",
        }

    @staticmethod
    def _environment_health_bloom_chain(supported: set[str]) -> Dict[str, Any]:
        steps = [
            {"tool": "load_dataset", "save_as": "bloom_field", "params": {"variable": "chlorophyll", "vertical_mode": "surface"}},
            {"tool": "detect_algal_blooms", "save_as": "bloom_detection", "params": {"chlorophyll": "$ref:bloom_field.data"}},
        ]
        if "bloom_frequency_change" in supported:
            steps.append({"tool": "compute_event_statistics", "save_as": "bloom_statistics", "params": {"events": "$ref:bloom_detection.events", "group_by": "year"}})
        if "bloom_event_days" in supported:
            steps.append({"tool": "compute_event_summary_map", "save_as": "bloom_event_days", "params": {"event_detection": "$ref:bloom_detection", "data": "$ref:bloom_field.data", "summary_mode": "event_days"}})
        if "bloom_burden" in supported:
            steps.append({"tool": "compute_event_summary_map", "save_as": "bloom_chlorophyll_burden", "params": {"event_detection": "$ref:bloom_detection", "data": "$ref:bloom_field.data", "summary_mode": "burden"}})
        return {"objective": "bloom_or_chlorophyll_evidence", "status": "required_when_requested", "canonical_steps": steps}

    @staticmethod
    def _environment_health_heatwave_chain(supported: set[str]) -> Dict[str, Any]:
        steps = [
            {"tool": "load_dataset", "save_as": "heatwave_field", "params": {"variable": "temp", "vertical_mode": "surface"}},
            {"tool": "detect_heatwaves", "save_as": "heatwave_detection", "params": {"temp": "$ref:heatwave_field.data"}},
        ]
        if "heatwave_days" in supported:
            steps.append({"tool": "compute_event_summary_map", "save_as": "heatwave_days", "params": {"event_detection": "$ref:heatwave_detection", "data": "$ref:heatwave_field.data", "summary_mode": "event_days"}})
        if "heatwave_burden" in supported:
            steps.append({"tool": "compute_event_summary_map", "save_as": "heatwave_burden", "params": {"event_detection": "$ref:heatwave_detection", "data": "$ref:heatwave_field.data", "summary_mode": "burden"}})
        return {"objective": "heatwave_evidence", "status": "required_when_requested", "canonical_steps": steps}

    def _review_relevant_tool_names(
        self,
        *,
        draft_plan: Dict[str, Any],
        tool_chain_requirements: List[Dict[str, Any]],
        skill_markdowns: Dict[str, str],
    ) -> set[str]:
        allowed_tools: set[str] = set()
        for skill_id, markdown in skill_markdowns.items():
            allowed_tools.update(self._allowed_tool_names_for_skill(skill_id, markdown))

        relevant: set[str] = set()
        for step in draft_plan.get("steps", []):
            if isinstance(step, dict) and isinstance(step.get("tool"), str):
                relevant.add(_SAFE_TOOL_ALIASES.get(step["tool"], step["tool"]))
        for requirement in tool_chain_requirements:
            for step in requirement.get("canonical_steps", []):
                if isinstance(step, dict) and isinstance(step.get("tool"), str):
                    relevant.add(step["tool"])
            tools = requirement.get("tools")
            if isinstance(tools, list):
                relevant.update(str(tool) for tool in tools if str(tool))
        if allowed_tools:
            relevant = {tool for tool in relevant if tool in allowed_tools}
        else:
            relevant = {tool for tool in relevant if get_tool_contract(tool)}
        return relevant

    def _focused_review_validation_rules(self, user_request: str) -> List[str]:
        rules = [
            "Every step must contain step_id, tool, params, and save_as.",
            "Every tool must exist in tool_argument_schemas.",
            "Every param must use the exact name and shape from the tool schema.",
            "Every $ref root must refer to an earlier save_as id or completed result id.",
            "ref_result params must reference whole results; ref_field params must reference the expected field path.",
            "If a named region is in user_intent_brief, every load_dataset step needs explicit lon_range and lat_range.",
            "If user_intent_brief.has_explicit_time_window is true, every load_dataset step needs time_range.",
            "Surface/SST evidence must express surface selection on load_dataset with vertical_mode='surface'; downstream tools that consume that loaded field must not repeat depth_range=[0, 0].",
            "Bottom oxygen evidence must load variable='oxygen' with vertical_mode='bottom' and no depth_range/depth_value.",
            "Hypoxia detection/maps/statistics must reference bottom_oxygen_field and hypoxia_detection exactly.",
            "Do not include assemble_environment_health_report or assemble_policy_recommendation_report unless explicit_fixed_report_allowed is true.",
        ]
        if self._explicit_fixed_report_requested(user_request):
            rules.append("Fixed report tools are allowed only because the user explicitly asked for a report/card/tool.")
        return rules

    def _review_time_range_hint(
        self,
        *,
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Optional[List[str]]:
        for source in self._time_intent_sources(extracted_params, additional_context):
            for key in ("time_range", "date_range", "temporal_range"):
                value = self._coerce_time_range_strings(source.get(key))
                if value is not None:
                    return value
        return self._time_range_from_request_text(user_request)

    @staticmethod
    def _truncate_text(value: Any, max_chars: int) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _friendly_plan_validation_error(error: Any) -> str:
        text = str(error or "Planning failed.")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        deduped: List[str] = []
        seen: set[str] = set()
        for line in lines or [text.strip()]:
            if line in seen:
                continue
            seen.add(line)
            deduped.append(line)
        text = "\n".join(deduped) or "Planning failed."
        match = re.search(r"requested branch '([^']+)' requires result '([^']+)'", text)
        if match:
            branch_key, result_id = match.groups()
            return (
                f"The generated plan missed the required {branch_key.replace('_', ' ')} evidence step "
                f"('{result_id}'). Please regenerate the plan with the complete executable tool chain."
            )
        return text.replace(
            "Reviewer should revise the plan using existing analysis tools, not backend repair.",
            "Regenerate the plan with the missing executable evidence step.",
        )

    def _build_messages(
        self,
        skill_id: str,
        skill_markdown: str,
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
        tool_contracts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a planning assistant. Read the provided SKILL.md and generate "
            "either a strict JSON execution plan or a clarification request.\n"
            "IMPORTANT: When generating clarification questions or human-readable labels, "
            "use ENGLISH only, regardless of the user's query language.\n"
            "Follow the skill exactly.\n"
            "You are the owner of the complete executable plan: every returned step must include the final tool, "
            "complete params, canonical save_as id, and correct $ref links. The reviewer audits your plan; do not "
            "delegate chain construction to the reviewer or backend repair.\n"
            "Use only tools explicitly described by the skill.\n"
            "Use ONLY parameter names defined in tool_contracts — do NOT invent, shorten, or rename parameters.\n"
            "Read planner_contract_packet before writing steps. When it lists required evidence branch contracts or "
            "canonical_tool_chain_requirements, include those requested chains with the exact canonical save_as ids "
            "and refs, while adding the concrete region/time/depth/season arguments you infer from the user request.\n"
            "Extract concrete parameter values from the user request and provided context only.\n"
            "When the user request explicitly specifies a parameter, it overrides any value in additional_context.\n"
            "Use additional_context only as default/current workspace state for parameters the user did not specify.\n"
            f"{_NAMED_REGION_PLANNING_RULES}"
            "IMPORTANT: Use semantic reasoning to determine appropriate parameters. For example:\n"
            "  - If the user asks about 'typical' conditions, 'average' state, 'climatology', 'seasonal patterns', or 'suitable areas', "
            "infer that they need long-term averages (climatology) over multiple years, NOT the specific short time range from workspace.\n"
            "  - If the user asks about a specific date, event, or 'current' conditions, use the workspace time_range.\n"
            "  - If the user asks about a single calendar year such as 'in 2022' or '2022年', treat it as an explicit time window and set every load_dataset.time_range to ['2022-01-01', '2022-12-31'].\n"
            "  - If the user asks a question that requires understanding long-term patterns (e.g., 'where is suitable for X', 'which areas have high Y'), "
            "infer an appropriate multi-year time range for climatology analysis rather than using a short workspace time slice.\n"
            "  - If the user asks for 'all summers', 'every summer', 'summer climatology', 'all winters', 'every winter', or 'winter climatology', "
            "keep the requested multi-year time_range and set load_dataset.season_filter to JJA or DJF rather than approximating with quarter climatology.\n"
            "  - Translate natural-language depth intent into explicit planning parameters whenever the selected skill supports them.\n"
            "    * 'surface', 'SST', 'surface chlorophyll', 'surface salinity', 'surface current' -> set load_dataset.vertical_mode='surface'. Do not repeat depth_range=[0, 0] on downstream tools that consume the loaded field.\n"
            "    * 'upper 50 m', 'upper 100 m', 'upper 200 m' -> use depth_range=[0, 50], [0, 100], [0, 200].\n"
            "    * 'at 50 m', '50 m depth' -> use depth_range=[50, 50] or vertical_mode='fixed_depth' with depth_value=50 when the skill exposes that interface.\n"
            "    * 'bottom', 'bottom-layer', 'near-bottom' -> use vertical_mode='bottom' when the skill supports it; do not fake bottom semantics with a generic depth_range.\n"
            "    * Keep load_dataset and downstream tools aligned to the same effective vertical choice.\n"
            "If additional_context contains conversation_context, treat it as lightweight memory of prior queries, "
            "clarifications, findings, and conclusions. Use it only when it helps resolve the current request or a "
            "pending clarification, and never let it override the current user request.\n"
            "If additional_context contains conversation_memory, it is the role-scoped memory packet selected for "
            "this current query. Use its entries only when their stated selection reasons directly apply. Never use "
            "conversation memory just because it is recent, and never let it override explicit parameters in the "
            "current user request.\n"
            "If additional_context contains planner_prior_context_text, it is a compressed natural-language digest of "
            "prior queries plus reusable prior results. Use it when the current query refers to earlier results such as "
            "'that event above', 'the location identified previously', or 'the core found earlier'. Extract explicit "
            "coordinates, time ranges, thresholds, or representative depths from that text and use them as parameter "
            "values in the new plan. Treat it as compressed hints rather than full result objects; if the needed anchor "
            "is absent, return clarification_needed.\n"
            "If additional_context contains analysis_proposal_context.approved_proposal.skill_plan, it is a user-approved "
            "skill-backed planning skeleton. Follow its planned_steps and planned_tools as the backbone of the execution "
            "plan unless a required parameter is impossible to infer. Do not add unrelated branches just because the "
            "public question is broad.\n"
            "Do not invent missing task inputs.\n"
            "Return clarification_needed ONLY when a required parameter has no value that can be inferred "
            "from the user request, workspace context, or reasonable domain defaults. "
            "If the user names a variable (e.g. SST, chlorophyll, salinity), a region (e.g. South China Sea, Bohai), "
            "or a time period, treat those as sufficient — do not ask for exact bounds or dataset names "
            "if they can be resolved from workspace context.\n"
            "Do not leave placeholder strings like {dataset} or {target_date} in the output.\n"
            "Use $ref:<result_id>.data when a downstream tool expects a DataArray "
            "from load_dataset or another data container result.\n"
            "Use $ref:<result_id> when a downstream tool expects the entire dict "
            "result, such as timeseries or climatology outputs.\n"
            "Output scalar parameters (variable, dataset, field_type, method, aggregation, "
            "depth_mode, etc.) as plain JSON strings — never as single-element lists.\n"
            "\n"
            "TOOL NAMING RULES:\n"
            "- Use ONLY tool names that appear as keys in tool_contracts. Do NOT invent tool names.\n"
            "- For derived fields (vorticity, speed, buoyancy_frequency, etc.), use tool='compute_derived_field' "
            "with the field_type parameter — do NOT create tool names like 'compute_vorticity' or 'compute_speed'.\n"
            "- For spatial diagnostics, use tool='compute_spatial_field' — do NOT use 'compute_spatial_diagnostic'.\n"
            "\n"
            "OUTPUT SCHEMA RULES:\n"
            "- Use snake_case for all field names: 'skill_id' (not 'skillId'), 'skills_used' (not 'selectedSkill').\n"
            "- 'skills_used' must be a JSON array of strings, never a bare string.\n"
            "Return JSON only. Do not wrap the answer in markdown fences."
        )
        if self._skill_is_event_detection_or_analysis(skill_id):
            system_prompt += (
                "\n"
                "EVENT-SPECIFIC DEPTH RULES:\n"
                "Before planning event detection steps, determine one effective vertical selection.\n"
                "If the user explicitly specifies a layer/depth/depth range, use that.\n"
                "Otherwise use additional_context.workspace_context as the current depth default.\n"
                "Keep load_dataset and the downstream detect_* tool consistent with the same vertical choice.\n"
                "When the effective selection is surface, set load_dataset.vertical_mode='surface' and do not repeat depth_range=[0, 0] on downstream detect_* tools.\n"
                "When it is a fixed depth, set load_dataset.depth_range to [depth_value, depth_value].\n"
                "When it is a depth range, pass that same depth_range to load_dataset.\n"
                "When it is bottom, set load_dataset.vertical_mode='bottom' so the data load is already near-bottom; "
                "do not leave load_dataset unconstrained for bottom-oriented event queries.\n"
            )
        if _benchmark_disables_clarification(additional_context):
            system_prompt += (
                "\n"
                "BENCHMARK MODE:\n"
                "- The current request is a complete offline benchmark task.\n"
                "- Do not return status='clarification_needed'.\n"
                "- Infer missing but reasonable details from the user request, active dataset context, skill hints, and ocean-domain defaults.\n"
                "- Return status='ready' with the best executable plan.\n"
            )

        user_payload = {
            "skill_id": skill_id,
            "user_request": user_request,
            "extracted_params": extracted_params,
            "additional_context": additional_context,
            "tool_contracts": tool_contracts,
            "planner_contract_packet": self._build_planner_contract_packet(
                skill_id=skill_id,
                user_request=user_request,
                extracted_params=extracted_params,
                additional_context=additional_context,
            ),
            "skill_markdown": skill_markdown,
            "required_output_schema": {
                "status": "ready | clarification_needed",
                "steps": [
                    {
                        "step_id": "string",
                        "tool": "string",
                        "params": "object",
                        "save_as": "string",
                    }
                ],
                "missing_fields": ["string"],
                "clarification_question": "string",
            },
        }

        return {
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
                }
            ],
        }

    def _approved_proposal_skill_ids(
        self,
        additional_context: Dict[str, Any],
        skill_markdowns: Dict[str, str],
    ) -> List[str]:
        proposal_context = additional_context.get("analysis_proposal_context")
        if not isinstance(proposal_context, dict):
            return []
        proposal = proposal_context.get("approved_proposal")
        if not isinstance(proposal, dict):
            return []

        raw_skill_ids: List[Any] = []
        skill_plan = proposal.get("skill_plan")
        if isinstance(skill_plan, dict):
            primary = skill_plan.get("primary_skill")
            if isinstance(primary, str) and primary.strip():
                raw_skill_ids.append(primary)
            planned_skills = skill_plan.get("skills_used")
            if isinstance(planned_skills, list):
                raw_skill_ids.extend(planned_skills)
        selected_skills = proposal.get("selected_skills")
        if isinstance(selected_skills, list):
            raw_skill_ids.extend(selected_skills)

        seen: set[str] = set()
        skill_ids: List[str] = []
        for raw in raw_skill_ids:
            skill_id = str(raw).strip()
            if not skill_id or skill_id in seen or skill_id not in skill_markdowns:
                continue
            seen.add(skill_id)
            skill_ids.append(skill_id)
        return skill_ids

    def _build_messages_with_upstream(
        self,
        skill_id: str,
        skill_markdown: str,
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
        tool_contracts: Dict[str, Dict[str, Any]],
        upstream_outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        messages = self._build_messages(
            skill_id=skill_id,
            skill_markdown=skill_markdown,
            user_request=user_request,
            extracted_params=extracted_params,
            additional_context=additional_context,
            tool_contracts=tool_contracts,
        )
        payload = json.loads(messages["messages"][0]["content"])
        payload["upstream_outputs"] = upstream_outputs
        messages["messages"][0]["content"] = json.dumps(payload, ensure_ascii=False, indent=2)
        messages["system"] = (
            messages["system"]
            + "\n"
            + "UPSTREAM OUTPUTS FROM PRIOR SKILLS:\n"
            + "The following named results are already available from earlier steps.\n"
            + "Use $ref:<save_as> or $ref:<save_as>.data to reference them instead of re-loading "
            + "or re-computing the same data.\n"
            + "If an upstream output already provides the field your first step would load, "
            + "skip that load step and reference the upstream result directly.\n"
            + self._build_upstream_outputs_manifest(upstream_outputs)
        )
        return messages

    def _build_query_selection_messages(
        self,
        skill_briefs: Dict[str, str],
        skill_hints: Dict[str, Dict[str, Any]],
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
        allow_multiple_skills: bool,
    ) -> Dict[str, Any]:
        if allow_multiple_skills:
            skill_rule = (
                "1. Select the minimal set of one or more existing skill_ids needed to complete the task.\n"
                "2. Set 'skill_id' to the primary skill_id and set 'skills_used' to the full ordered skill list.\n"
                "3. Do not generate execution steps yet.\n"
            )
        else:
            skill_rule = (
                "1. Select exactly one existing skill_id.\n"
                "2. Set 'skills_used' to a single-item list containing that skill_id.\n"
                "3. Do not generate execution steps yet.\n"
            )

        system_prompt = (
            "You are a planning assistant. You will receive multiple skill briefs derived from SKILL.md documents.\n"
            "First select the best skill or skill set. Do not generate execution steps in this stage.\n"
            "Follow these rules:\n"
            f"{skill_rule}"
            "4. Extract concrete parameter values from the user request and provided context only.\n"
            "5. When the user request explicitly specifies a parameter, it overrides any value in additional_context.\n"
            "6. Use additional_context only as default/current workspace state for parameters the user did not specify.\n"
            f"{_NAMED_REGION_PLANNING_RULES}"
            "IMPORTANT: Use semantic reasoning to determine appropriate parameters. For example:\n"
            "  - If the user asks about 'typical' conditions, 'average' state, 'climatology', 'seasonal patterns', or 'suitable areas', "
            "infer that they need long-term averages (climatology) over multiple years, NOT the specific short time range from workspace.\n"
            "  - If the user asks about a specific date, event, or 'current' conditions, use the workspace time_range.\n"
            "  - If the user asks about a single calendar year such as 'in 2022' or '2022年', treat it as an explicit time window and carry ['2022-01-01', '2022-12-31'] into load_dataset.time_range.\n"
            "  - If the user asks a question that requires understanding long-term patterns (e.g., 'where is suitable for X', 'which areas have high Y'), "
            "infer an appropriate multi-year time range for climatology analysis rather than using a short workspace time slice.\n"
            "  - If the user asks for 'all summers', 'every summer', 'summer climatology', 'all winters', 'every winter', or 'winter climatology', "
            "prefer skills that can carry a season_filter on load_dataset instead of quarter-based climatology shortcuts.\n"
            "  - Keep raw tracer analysis separate from derived diagnostics. Raw temperature/SST/salinity/oxygen/chlorophyll requests "
            "should prefer the non-derived skills. Only choose a derived_* skill when the user explicitly asks for a diagnostic such as "
            "speed, vorticity, buoyancy frequency, kinetic energy, local tendency, or advection.\n"
            "7. If additional_context contains conversation_context, treat it as lightweight memory of prior queries, "
            "clarifications, findings, and conclusions. Use it only when it helps resolve the current request or a "
            "pending clarification, and never let it override the current user request.\n"
            "If additional_context contains conversation_memory, it is the role-scoped memory packet selected for "
            "this current query. Use its entries only when their stated selection reasons directly apply. Never use "
            "conversation memory just because it is recent, and never let it override explicit parameters in the "
            "current user request.\n"
            "If additional_context contains planner_prior_context_text, it is a compressed natural-language digest of "
            "prior queries plus reusable prior results. Use it to resolve follow-up references such as 'that event above' "
            "or 'the location identified previously'. Extract explicit coordinates, time ranges, thresholds, or "
            "representative depths from that text when selecting the skill and downstream parameters. If the needed anchor "
            "is not present there, return clarification_needed.\n"
            "8. Do not invent missing task inputs.\n"
            "9. Return status='clarification_needed' ONLY when a required parameter cannot be inferred "
            "from the user request, workspace context, or reasonable domain defaults. "
            "Named variables (SST, chlorophyll), named regions (South China Sea, Bohai), and "
            "stated time periods are sufficient — do not clarify what can be resolved from context.\n"
            "10. Return JSON only. Do not include markdown fences or commentary."
        )
        if _benchmark_disables_clarification(additional_context):
            system_prompt += (
                "\n"
                "BENCHMARK MODE:\n"
                "- The current request is a complete offline benchmark task.\n"
                "- Do not return status='clarification_needed'.\n"
                "- Select the best skill_id or minimal ordered skill set using the query, active dataset context, skill hints, and reasonable ocean-domain defaults.\n"
                "- Return status='ready'.\n"
            )

        user_payload = {
            "user_request": user_request,
            "extracted_params": extracted_params,
            "additional_context": additional_context,
            "available_skill_ids": sorted(skill_briefs.keys()),
            "skill_briefs": skill_briefs,
            "skill_hints": skill_hints,
            "allow_multiple_skills": allow_multiple_skills,
            "required_output_schema": {
                "status": "ready | clarification_needed",
                "skill_id": "string",
                "skills_used": ["string"],
                "missing_fields": ["string"],
                "clarification_question": "string",
            },
        }

        return {
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
                }
            ],
        }

    def _build_review_messages(
        self,
        user_request: str,
        active_plan: Dict[str, Any],
        completed_steps: List[Dict[str, Any]],
        remaining_steps: List[Dict[str, Any]],
        last_event: Dict[str, Any],
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
        skill_contracts: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a planning supervisor reviewing an in-flight execution.\n"
            "Decide whether the workflow should continue, be replanned, ask the user for clarification, or abort.\n"
            "Follow these rules:\n"
            "1. Prefer 'continue' when the current plan remains valid.\n"
            "2. Use 'replan' only when the next steps should change materially.\n"
            "3. If you return 'replan', provide 'updated_plan' containing the revised remaining future steps.\n"
            "4. Any new plan may reference already completed result ids using $ref expressions.\n"
            "5. If required task information is missing, return 'ask_user' with missing_fields and one concise clarification_question.\n"
            "6. Use only tools described by tool_argument_schemas in updated_plan.\n"
            "7. Use ONLY parameter names defined in tool_argument_schemas — do NOT invent, shorten, or rename parameters.\n"
            "8. If additional_context contains planner_prior_context_text, it is a compressed digest of prior queries and reusable prior results. "
            "Use it when replanning follow-up requests that refer to earlier coordinates, times, thresholds, or event centers.\n"
            "9. Read tool argument schema before changing params. The backend will validate but will not repair science semantics.\n"
            "10. Do not add fixed report tools unless the user explicitly asked for a fixed report/card/tool.\n"
            "11. Return JSON only."
        )
        if _benchmark_disables_clarification(additional_context):
            system_prompt += (
                "\n"
                "BENCHMARK MODE:\n"
                "- Do not return decision='ask_user'.\n"
                "- If recovery is possible, return decision='replan' with an executable updated_plan.\n"
                "- If recovery is impossible, return decision='abort'.\n"
            )

        chain_requirements = self._build_tool_chain_requirements(
            user_request=user_request,
            draft_plan=active_plan,
            additional_context=additional_context,
        )
        tool_names = self._review_relevant_tool_names(
            draft_plan=active_plan,
            tool_chain_requirements=chain_requirements,
            skill_markdowns={},
        )
        for tool_name in skill_contracts:
            tool_names.add(tool_name)
        tool_argument_schemas = {
            tool_name: self._build_minimal_tool_contract(tool_name)
            for tool_name in sorted(tool_names)
            if get_tool_contract(tool_name)
        }

        user_payload = {
            "review_packet": {
                "user_intent_brief": self._build_user_intent_brief(
                    user_request=user_request,
                    extracted_params=extracted_params,
                    additional_context=additional_context,
                    draft_plan=active_plan,
                ),
                "active_plan": active_plan,
                "completed_steps": completed_steps,
                "remaining_steps": remaining_steps,
                "last_event": last_event,
                "tool_chain_requirements": chain_requirements,
                "tool_argument_schemas": tool_argument_schemas,
                "dataset_capability_summary": self._build_dataset_capability_summary(additional_context),
                "hard_validation_rules": self._focused_review_validation_rules(user_request),
                "explicit_fixed_report_allowed": self._explicit_fixed_report_requested(user_request),
            },
            "required_output_schema": {
                "decision": "continue | replan | ask_user | abort",
                "reason": "string",
                "updated_plan": {
                    "status": "ready",
                    "skill_id": "string",
                    "skills_used": ["string"],
                    "steps": [
                        {
                            "step_id": "string",
                            "tool": "string",
                            "params": "object",
                            "save_as": "string",
                        }
                    ],
                },
                "missing_fields": ["string"],
                "clarification_question": "string",
            },
        }

        return {
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
                }
            ],
        }

    def _collect_tool_contracts(
        self,
        skill_markdown: str,
        *,
        skill_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        contracts: Dict[str, Dict[str, Any]] = {}
        for tool_name in self._allowed_tool_names_for_skill(skill_id or "", skill_markdown):
            contract = self._build_minimal_tool_contract(tool_name)
            if contract:
                contracts[tool_name] = contract
        return contracts

    def _build_minimal_tool_contract(self, tool_name: str) -> Dict[str, Any]:
        contract = get_tool_contract(tool_name) or {}
        inputs = contract.get("inputs", {})
        minimal_inputs: Dict[str, Dict[str, Any]] = {}

        for param_name, param_contract in inputs.items():
            compact_param: Dict[str, Any] = {}
            for key in (
                "kind",
                "type",
                "expected",
                "minimum",
                "maximum",
                "exclusive_minimum",
                "exclusive_maximum",
                "default",
            ):
                if key in param_contract:
                    compact_param[key] = param_contract[key]
            if compact_param:
                minimal_inputs[param_name] = compact_param

        minimal_contract: Dict[str, Any] = {}
        if "output_type" in contract:
            minimal_contract["output_type"] = contract["output_type"]
        description = self._build_tool_description(tool_name)
        if description:
            minimal_contract["description"] = description
        if minimal_inputs:
            minimal_contract["inputs"] = minimal_inputs
        return minimal_contract

    def _build_tool_description(self, tool_name: str) -> Optional[str]:
        if tool_name == "load_dataset":
            return (
                "Load the source ocean variable from the configured dataset files. "
                "Specify only the variable name — do NOT include a dataset parameter. "
                "The system automatically locates the correct file based on the variable name. "
                "Use variable names like 'temp', 'chlorophyll' (or 'chla'), "
                "'salt', 'oxygen', 'u', 'v'. "
                "When the user names a geographic region, include explicit approximate "
                "lon_range and lat_range for that named region; do not fall back to the "
                "dataset extent or current workspace region unless the user asked for the "
                "current/drawn region. "
                "To constrain vertical loading, use vertical_mode='surface' for surface/SST requests, "
                "depth_range for true depth ranges, and vertical_mode='bottom' for near-bottom loading. "
                "Downstream analysis tools consume the result via $ref:<result_id>.data."
            )
        if tool_name == "assemble_dataset":
            return (
                "Combine multiple named DataArray inputs into one Dataset. "
                "Use the parameter name 'variables', not 'arrays', not 'fields'. "
                "Pass an object mapping variable name to DataArray references, "
                "for example variables={\"u\": \"$ref:u_field.data\", "
                "\"v\": \"$ref:v_field.data\"}."
            )

        default_descriptions = {
            "compute_derived_field": (
                "Compute a derived physical field. Supported field_type values: "
                "'vorticity', 'speed', 'horizontal_gradient', 'vertical_gradient', "
                "'buoyancy_frequency'. Do NOT use this tool for kinetic energy — "
                "use compute_kinetic_energy instead. "
                "Do NOT generate tool names like 'compute_vorticity' or 'compute_speed' — "
                "always use this tool with the correct field_type parameter."
            ),
            "compute_spatial_vorticity_map": (
                "Fast path for map-ready relative vorticity. Use this when the request asks "
                "for a spatial/map plot of time/depth aggregated vorticity, such as a winter "
                "mean upper-layer vorticity map. Pass u and v directly; the tool aggregates "
                "u/v first and then computes the 2D vorticity field."
            ),
            "extract_regional_mean": (
                "Compute a regional mean time series from a loaded data field over "
                "the requested lon/lat box and optional depth selection."
            ),
            "extract_point_timeseries": (
                "Extract a point time series from a loaded data field at lon/lat. "
                "Use method='nearest' when the user asks for the nearest model value "
                "or nearest grid cell. Do NOT replace explicit point requests with "
                "regional or area-weighted means."
            ),
            "extract_timeseries": (
                "Extract either a point or region time series from a loaded data field. "
                "For explicit lon/lat point requests, pass lon, lat, and method='nearest' "
                "instead of lon_range/lat_range."
            ),
            "compute_spatial_field": (
                "Reduce an already loaded field into a map-style spatial result for visualization or "
                "subsequent spatial analysis. Do NOT pass lon_range or lat_range here; "
                "horizontal region selection belongs in load_dataset or extract_4d_subset."
            ),
            "extract_vertical_profile": (
                "Extract a vertical profile at a point or representative location from "
                "a loaded three-dimensional field."
            ),
            "compute_trend": (
                "Fit a trend to a time series and return slope, significance, and "
                "related summary statistics."
            ),
            "compute_kinetic_energy": (
                "Compute kinetic energy from velocity fields. Prefer explicit "
                "'u_data' and 'v_data' DataArray references. A Dataset containing "
                "'u' and 'v' is accepted only as a fallback via 'dataset'/'data'."
            ),
            "compute_vertical_shear": (
                "Compute vertical shear from velocity fields. Prefer explicit "
                "'u_data' and 'v_data' DataArray references. A Dataset containing "
                "'u' and 'v' is accepted only as a fallback via 'dataset'/'data'."
            ),
            "compute_strain_rate": (
                "Compute horizontal strain rate from velocity fields. Prefer explicit "
                "'u_data' and 'v_data' DataArray references. A Dataset containing "
                "'u' and 'v' is accepted only as a fallback via 'dataset'/'data'."
            ),
            "compute_rossby_number": (
                "Compute Rossby number from velocity fields. Prefer explicit "
                "'u_data' and 'v_data' DataArray references. A Dataset containing "
                "'u' and 'v' is accepted only as a fallback via 'dataset'/'data'."
            ),
            "compute_divergence": (
                "Compute horizontal divergence from velocity fields. Prefer explicit "
                "'u_data' and 'v_data' DataArray references. A Dataset containing "
                "'u' and 'v' is accepted only as a fallback via 'dataset'/'data'."
            ),
            "detect_heatwaves": (
                "Detect heatwave events from a temperature field. By default this uses "
                "the surface layer, but you can override depth handling with "
                "vertical_mode plus depth_value or depth_range."
            ),
            "detect_hypoxia": (
                "Detect hypoxia events from an oxygen field. By default this uses the "
                "bottom layer, but you can override depth handling with vertical_mode "
                "plus depth_value or depth_range."
            ),
            "detect_algal_blooms": (
                "Detect bloom events from a chlorophyll field. By default this uses "
                "the surface layer, but you can override depth handling with "
                "vertical_mode plus depth_value or depth_range."
            ),
            "detect_upwelling": (
                "Detect upwelling events from a temperature field. By default this uses "
                "the surface layer, but you can override depth handling with "
                "vertical_mode plus depth_value or depth_range."
            ),
            "detect_eutrophication": (
                "Detect eutrophication-like events from chlorophyll, with optional oxygen. "
                "By default this uses the surface layer, but you can override depth "
                "handling with vertical_mode plus depth_value or depth_range."
            ),
        }
        return default_descriptions.get(tool_name)

    def _extract_tool_names(self, skill_markdown: str) -> List[str]:
        explicit_pattern = re.compile(r"\*\*Tool\*\*:\s*`([^`]+)`")
        inline_pattern = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
        identifier_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
        seen: List[str] = []

        def add_if_tool(candidate: str) -> None:
            if get_tool_contract(candidate) and candidate not in seen:
                seen.append(candidate)

        for match in explicit_pattern.findall(skill_markdown):
            add_if_tool(match)

        for match in inline_pattern.findall(skill_markdown):
            add_if_tool(match)
            for candidate in identifier_pattern.findall(match):
                add_if_tool(candidate)

        if re.search(r"\b[Ll]oad\b", skill_markdown):
            add_if_tool("load_dataset")
        return seen

    def _allowed_tool_names_for_skill(self, skill_id: str, skill_markdown: str) -> List[str]:
        seen = list(self._extract_tool_names(skill_markdown))

        def add_if_tool(candidate: str) -> None:
            if get_tool_contract(candidate) and candidate not in seen:
                seen.append(candidate)

        if skill_id == "ocean_environment_health_assessment":
            for tool_name in sorted(_ENVIRONMENT_HEALTH_COMPOSER_TOOLS):
                add_if_tool(tool_name)
        return seen

    def _build_skill_brief(
        self,
        skill_markdown: str,
        planner_hint: Optional[Dict[str, Any]] = None,
        max_chars: int = 400,
    ) -> str:
        """
        Build a minimal brief for query-time skill selection.

        Selection already receives the available skill_ids separately, so the
        brief only keeps the Description section and excludes metadata,
        scenarios, schemas, and execution details.
        """
        lines = skill_markdown.strip().splitlines()
        if not lines:
            return ""

        sections: Dict[str, List[str]] = {}
        current_section: Optional[str] = None
        in_code_block = False

        for raw_line in lines:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                continue

            if stripped.startswith("## "):
                current_section = stripped[3:].strip().lower()
                sections.setdefault(current_section, [])
                continue

            if current_section is not None:
                sections.setdefault(current_section, []).append(stripped)

        if planner_hint:
            pieces = [str(planner_hint.get("intent_summary", "")).strip()]
            negatives = planner_hint.get("negative_query_examples", [])[:2]
            if negatives:
                pieces.append("Avoid: " + "; ".join(str(example) for example in negatives))
            examples = planner_hint.get("positive_query_examples", [])[:3]
            if examples:
                pieces.append("Examples: " + "; ".join(str(example) for example in examples))
            required = planner_hint.get("required_entities", [])
            if required:
                pieces.append("Needs: " + ", ".join(str(item) for item in required))
            result_types = planner_hint.get("result_types", [])
            if result_types:
                pieces.append("Results: " + ", ".join(str(item) for item in result_types))
            brief = "\n".join(piece for piece in pieces if piece).strip()
        else:
            description_lines = sections.get("description", [])
            brief = "\n".join(description_lines).strip()
        if len(brief) <= max_chars:
            return brief

        cutoff = brief.rfind("\n", 0, max_chars)
        if cutoff == -1:
            cutoff = max_chars
        return brief[:cutoff].rstrip() + "\n..."

    def _build_upstream_outputs_manifest(self, upstream_outputs: Dict[str, Dict[str, Any]]) -> str:
        if not upstream_outputs:
            return "\nAvailable:\n  - none"

        lines = ["\nAvailable:"]
        for save_as, metadata in upstream_outputs.items():
            tool_name = metadata.get("tool", "?")
            output_type = metadata.get("output_type", "generic_result")
            skill_origin = metadata.get("skill_origin", "?")
            lines.append(
                f"\n  - {save_as}: from {tool_name}, type={output_type}, skill={skill_origin}"
            )
        return "".join(lines)

    def _dedupe_step_save_as_names(
        self,
        *,
        skill_id: str,
        steps: List[Dict[str, Any]],
        existing_names: set[str],
    ) -> Dict[str, str]:
        rename_map: Dict[str, str] = {}
        reserved_names = set(existing_names)

        for step in steps:
            save_as = step.get("save_as")
            if not isinstance(save_as, str) or not save_as:
                continue
            if save_as not in reserved_names:
                reserved_names.add(save_as)
                continue

            new_name = self._build_unique_save_as_name(
                skill_id=skill_id,
                original_name=save_as,
                reserved_names=reserved_names,
            )
            rename_map[save_as] = new_name
            step["save_as"] = new_name
            reserved_names.add(new_name)

        return rename_map

    def _build_unique_save_as_name(
        self,
        *,
        skill_id: str,
        original_name: str,
        reserved_names: set[str],
    ) -> str:
        skill_prefix = skill_id.replace("ocean_", "")
        candidate = f"{skill_prefix}_{original_name}"
        suffix = 2
        while candidate in reserved_names:
            candidate = f"{skill_prefix}_{original_name}_{suffix}"
            suffix += 1
        return candidate

    def _rewrite_refs_in_steps(
        self,
        steps: List[Dict[str, Any]],
        rename_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        rewritten_steps: List[Dict[str, Any]] = []
        for step in steps:
            rewritten_step = dict(step)
            rewritten_step["params"] = self._rewrite_refs_in_value(step.get("params", {}), rename_map)
            rewritten_steps.append(rewritten_step)
        return rewritten_steps

    def _rewrite_refs_in_value(self, value: Any, rename_map: Dict[str, str]) -> Any:
        if isinstance(value, dict):
            return {key: self._rewrite_refs_in_value(nested, rename_map) for key, nested in value.items()}
        if isinstance(value, list):
            return [self._rewrite_refs_in_value(nested, rename_map) for nested in value]
        if isinstance(value, tuple):
            return tuple(self._rewrite_refs_in_value(nested, rename_map) for nested in value)
        if not isinstance(value, str) or not value.startswith("$ref:"):
            return value

        ref_expr = value[5:]
        ref_root, separator, ref_path = ref_expr.partition(".")
        if ref_root not in rename_map:
            return value
        rewritten_root = rename_map[ref_root]
        rewritten_expr = rewritten_root if not separator else f"{rewritten_root}.{ref_path}"
        return f"$ref:{rewritten_expr}"

    def _normalize_plan_shape(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            return plan

        normalized = dict(plan)

        if not isinstance(normalized.get("skill_id"), str):
            skills_used = normalized.get("skills_used")
            if isinstance(skills_used, list) and skills_used and isinstance(skills_used[0], str):
                normalized["skill_id"] = skills_used[0]

        steps = normalized.get("steps")
        if isinstance(steps, list):
            normalized["steps"] = [
                self._normalize_step_to_contract(step)
                if isinstance(step, dict)
                else step
                for step in steps
            ]

        return normalized

    def _normalize_step_to_contract(self, step: Dict[str, Any]) -> Dict[str, Any]:
        return self._normalize_step_params_to_contract(step)

    def _normalize_step_params_to_contract(self, step: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = step.get("tool")
        params = step.get("params")
        if not isinstance(tool_name, str) or not isinstance(params, dict):
            return step

        normalized_tool = _SAFE_TOOL_ALIASES.get(tool_name, tool_name)
        alias_map = _SAFE_PARAM_ALIASES.get(normalized_tool, {})
        if alias_map:
            params = {
                alias_map.get(param_name, param_name): value
                for param_name, value in params.items()
            }

        contract = get_tool_contract(normalized_tool) or {}
        defined_params = set(contract.get("inputs", {}).keys())
        normalized_step = dict(step)
        normalized_step["tool"] = normalized_tool
        if not defined_params:
            normalized_step["params"] = params
            return normalized_step

        normalized_params = {}
        for param_name, value in params.items():
            if param_name not in defined_params:
                normalized_params[param_name] = value
                continue
            param_contract = contract.get("inputs", {}).get(param_name, {})
            normalized_params[param_name] = normalize_tool_param_value(
                normalized_tool,
                param_name,
                value,
                param_contract,
                use_default=False,
            )
        if normalized_params == params and normalized_tool == tool_name:
            return step

        normalized_step["params"] = normalized_params
        return normalized_step

    def _repair_skill_specific_plan(
        self,
        plan: Dict[str, Any],
        user_request: str,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compatibility repair path used by tests and legacy call sites.

        The primary planner path now validates deterministic repairs before
        reviewer repair, and these helpers still encode hard contracts for
        vertical selection and legacy report wiring.
        """
        additional_context = additional_context or {}
        repaired = self._normalize_plan_shape(plan)
        skill_id = str(repaired.get("skill_id") or "").strip()

        repaired = self._repair_isobath_mask_reference_plan(repaired)
        repaired = self._repair_bottom_oxygen_vertical_plan(repaired, user_request=user_request)

        if skill_id == "ocean_environment_health_assessment":
            if self._environment_health_policy_only_request(user_request):
                return repaired
            repaired = self._repair_environment_health_plan(repaired, user_request=user_request)
        else:
            repaired = self._suppress_default_environment_assessment_plan(repaired, user_request)
            repaired = self._suppress_default_policy_report_plan(repaired, user_request)

        if skill_id in {
            "ocean_stratification_diagnostics",
            "ocean_environment_health_assessment",
        }:
            repaired = self._repair_stratification_depth_plan(repaired, user_request=user_request)
            repaired = self._repair_stratification_index_method(repaired)
            repaired = self._repair_stratification_lead_lag_plan(repaired, user_request=user_request)

        if skill_id in {"ocean_lag_correlation", "ocean_stratification_diagnostics"}:
            repaired = self._repair_lag_seasonality_plan(repaired, user_request=user_request)

        repaired = self._repair_transport_streamfunction_regional_gauge_plan(
            repaired,
            user_request=user_request,
            additional_context=additional_context,
        )
        repaired = self._repair_step_dependency_order(repaired)
        return repaired

    @staticmethod
    def _canonicalize_environment_health_standard_step_params(
        plan: Dict[str, Any],
        *,
        user_request: str = "",
        available_result_ids: Optional[set[str]] = None,
        requested_branch_keys: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return plan
        repaired_steps = [SkillPlanner._copy_step_with_params(step) for step in steps]
        request_load_context = SkillPlanner._environment_health_request_load_context(user_request)
        load_context = SkillPlanner._environment_health_common_load_context(repaired_steps)
        load_context.update(request_load_context)
        if requested_branch_keys is None:
            requested_branch_keys = SkillPlanner._environment_health_requested_branch_keys(user_request)
            requested_branch_keys.difference_update(
                SkillPlanner._environment_health_suppressed_branch_keys(user_request)
            )
        else:
            requested_branch_keys = set(requested_branch_keys)
        SkillPlanner._ensure_environment_health_required_steps(
            repaired_steps,
            requested_branch_keys=requested_branch_keys,
            available_result_ids=available_result_ids,
        )
        SkillPlanner._ensure_hypoxia_bottom_oxygen_field_precedes_detection(
            repaired_steps,
            available_result_ids=available_result_ids,
        )
        SkillPlanner._apply_environment_health_load_context_to_steps(
            repaired_steps,
            load_context,
            overwrite_keys=set(request_load_context),
        )
        save_as_to_step = {
            step.get("save_as"): step
            for step in repaired_steps
            if isinstance(step, dict) and isinstance(step.get("save_as"), str)
        }
        SkillPlanner._repair_environment_health_standard_steps(save_as_to_step)
        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    @staticmethod
    def _canonicalize_hypoxia_standard_step_params(
        plan: Dict[str, Any],
        *,
        available_result_ids: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return plan
        repaired_steps = [SkillPlanner._copy_step_with_params(step) for step in steps]
        SkillPlanner._ensure_hypoxia_bottom_oxygen_field_precedes_detection(
            repaired_steps,
            available_result_ids=available_result_ids,
        )
        save_as_to_step = {
            step.get("save_as"): step
            for step in repaired_steps
            if isinstance(step, dict) and isinstance(step.get("save_as"), str)
        }
        SkillPlanner._repair_environment_health_standard_steps(save_as_to_step)
        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    @staticmethod
    def _ensure_hypoxia_bottom_oxygen_field_precedes_detection(
        steps: List[Dict[str, Any]],
        *,
        available_result_ids: Optional[set[str]] = None,
    ) -> None:
        first_hypoxia_index = next(
            (
                index
                for index, step in enumerate(steps)
                if isinstance(step, dict) and step.get("tool") == "detect_hypoxia"
            ),
            None,
        )
        if first_hypoxia_index is None:
            return

        bottom_index = next(
            (
                index
                for index, step in enumerate(steps)
                if isinstance(step, dict) and step.get("save_as") == "bottom_oxygen_field"
            ),
            None,
        )
        if bottom_index is None:
            if "bottom_oxygen_field" in (available_result_ids or set()):
                return
            bottom_step = SkillPlanner._environment_health_default_step("bottom_oxygen_field")
            if bottom_step is None:
                return
            SkillPlanner._apply_environment_health_load_context(
                bottom_step,
                SkillPlanner._environment_health_common_load_context(steps),
            )
            steps.insert(first_hypoxia_index, bottom_step)
            return

        if bottom_index < first_hypoxia_index:
            return
        bottom_step = steps.pop(bottom_index)
        steps.insert(first_hypoxia_index, bottom_step)

    def _suppress_default_policy_report_plan(self, plan: Dict[str, Any], user_request: str) -> Dict[str, Any]:
        """Keep fixed policy reports explicit-only; ordinary advice is synthesized later."""
        if self._explicit_policy_report_requested(user_request):
            return plan
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan
        filtered_steps = [
            step
            for step in steps
            if not (isinstance(step, dict) and step.get("tool") == "assemble_policy_recommendation_report")
        ]
        if len(filtered_steps) == len(steps) or not filtered_steps:
            return plan
        repaired = dict(plan)
        repaired["steps"] = filtered_steps
        return repaired

    def _suppress_default_environment_assessment_plan(self, plan: Dict[str, Any], user_request: str) -> Dict[str, Any]:
        """Use analyzed evidence directly unless the user asks for a fixed report/card."""
        if self._explicit_fixed_report_requested(user_request):
            return plan
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan
        filtered_steps = [
            step
            for step in steps
            if not (isinstance(step, dict) and step.get("tool") == "assemble_environment_health_report")
        ]
        if len(filtered_steps) == len(steps) or not filtered_steps:
            return plan
        repaired = dict(plan)
        repaired["steps"] = filtered_steps
        return repaired

    @staticmethod
    def _policy_guidance_requested(user_request: str) -> bool:
        return isinstance(user_request, str) and bool(_POLICY_GUIDANCE_INTENT_RE.search(user_request))

    @staticmethod
    def _environment_health_policy_only_request(user_request: str) -> bool:
        if not SkillPlanner._policy_guidance_requested(user_request):
            return False
        lowered = (user_request or "").lower()
        analysis_markers = (
            r"\b(?:trend|trends|change|changes|assess|assessment|analyze|analysis|"
            r"evaluate|diagnose|detect|map|compute|calculate|suitable|suitability|"
            r"risk|risks|hotspot|hotspots|evidence|which areas|remain)\b|"
            r"趋势|变化|评估|分析|诊断|检测|绘制|计算|适合|适宜|风险|热点|证据|哪里|区域|如何|近几年|近年来"
        )
        return not bool(re.search(analysis_markers, lowered))

    @staticmethod
    def _explicit_policy_report_requested(user_request: str) -> bool:
        return (
            isinstance(user_request, str)
            and bool(_EXPLICIT_POLICY_REPORT_RE.search(user_request))
            and not bool(_NEGATED_POLICY_REPORT_RE.search(user_request))
        )

    @staticmethod
    def _explicit_environment_health_report_requested(user_request: str) -> bool:
        return (
            isinstance(user_request, str)
            and bool(_EXPLICIT_ENVIRONMENT_REPORT_RE.search(user_request))
            and not bool(_NEGATED_ENVIRONMENT_REPORT_RE.search(user_request))
        )

    @staticmethod
    def _explicit_fixed_report_requested(user_request: str) -> bool:
        return (
            SkillPlanner._explicit_policy_report_requested(user_request)
            or SkillPlanner._explicit_environment_health_report_requested(user_request)
        )

    def _repair_isobath_mask_reference_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Use raw velocity fields, not derived vorticity, for wet-depth inference."""
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan

        repaired_steps = [SkillPlanner._copy_step_with_params(step) for step in steps]
        velocity_refs = self._find_velocity_load_refs(repaired_steps)
        fallback_velocity_ref = velocity_refs.get("u") or velocity_refs.get("v")
        if fallback_velocity_ref is None:
            return plan

        derived_to_source_ref: Dict[str, str] = {}
        for step in repaired_steps:
            if step.get("tool") != "compute_derived_field":
                continue
            params = step.get("params")
            save_as = step.get("save_as")
            if not isinstance(params, dict) or not isinstance(save_as, str):
                continue
            field_type = params.get("field_type") or params.get("variable")
            if field_type not in {"vorticity", "relative_vorticity", "relative vorticity"}:
                continue
            source_ref = (
                self._extract_ref_root(params.get("u"))
                or self._extract_ref_root(params.get("v"))
                or fallback_velocity_ref
            )
            if source_ref:
                derived_to_source_ref[save_as] = source_ref

        if not derived_to_source_ref:
            return plan

        changed = False
        for step in repaired_steps:
            if step.get("tool") != "build_isobath_mask":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            data_ref = self._extract_ref_root(params.get("data"))
            source_ref = derived_to_source_ref.get(data_ref or "")
            if source_ref is None:
                continue
            params["data"] = f"$ref:{source_ref}.data"
            changed = True

        if not changed:
            return plan

        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    @staticmethod
    def _repair_stratification_index_method(plan: Dict[str, Any]) -> Dict[str, Any]:
        """Fix hallucinated method values for compute_stratification_index.

        The only valid values are ``"surface_bottom_density_difference"``
        (default) and ``"density_gradient"``.  LLMs frequently hallucinate
        shortened variants such as ``"density_difference"``.
        """
        _VALID_METHODS = {"surface_bottom_density_difference", "density_gradient"}
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return plan
        for step in steps:
            if step.get("tool") != "compute_stratification_index":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            method = params.get("method")
            if method is not None and method not in _VALID_METHODS:
                params["method"] = "surface_bottom_density_difference"
        return plan

    @staticmethod
    def _repair_environment_health_plan(
        plan: Dict[str, Any],
        *,
        user_request: str,
    ) -> Dict[str, Any]:
        """Keep template-composed indicator evidence wired without forcing fixed reports."""
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan
        explicit_environment_report = SkillPlanner._explicit_environment_health_report_requested(user_request)
        explicit_fixed_report = SkillPlanner._explicit_fixed_report_requested(user_request)

        repaired_steps = [SkillPlanner._copy_step_with_params(step) for step in steps]
        save_as_to_step = {
            step.get("save_as"): step
            for step in repaired_steps
            if isinstance(step.get("save_as"), str)
        }

        SkillPlanner._repair_environment_health_standard_steps(save_as_to_step)
        requested_branch_keys = SkillPlanner._repair_environment_health_branch_payloads(
            repaired_steps,
            user_request=user_request,
        )
        requested_branch_keys.update(
            SkillPlanner._environment_health_requested_branch_keys(user_request)
        )
        requested_branch_keys.difference_update(
            SkillPlanner._environment_health_suppressed_branch_keys(user_request)
        )
        SkillPlanner._ensure_environment_health_required_steps(
            repaired_steps,
            requested_branch_keys=requested_branch_keys,
        )
        request_load_context = SkillPlanner._environment_health_request_load_context(user_request)
        load_context = SkillPlanner._environment_health_common_load_context(repaired_steps)
        load_context.update(request_load_context)
        SkillPlanner._apply_environment_health_load_context_to_steps(
            repaired_steps,
            load_context,
            overwrite_keys=set(request_load_context),
        )
        save_as_to_step = {
            step.get("save_as"): step
            for step in repaired_steps
            if isinstance(step.get("save_as"), str)
        }
        SkillPlanner._repair_environment_health_standard_steps(save_as_to_step)

        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        if not explicit_fixed_report:
            repaired_plan["steps"] = [
                step
                for step in repaired_steps
                if not (
                    isinstance(step, dict)
                    and step.get("tool") in {
                        "assemble_environment_health_report",
                        "assemble_policy_recommendation_report",
                    }
                )
            ]
            return repaired_plan

        has_environment_report = any(
            isinstance(step, dict) and step.get("tool") == "assemble_environment_health_report"
            for step in repaired_steps
        )
        SkillPlanner._validate_environment_health_plan(
            repaired_plan,
            requested_branch_keys=requested_branch_keys,
            require_report=explicit_environment_report or has_environment_report,
        )
        return repaired_plan

    @staticmethod
    def _copy_step_with_params(step: Any) -> Dict[str, Any]:
        if not isinstance(step, dict):
            return step
        copied = dict(step)
        params = copied.get("params")
        if isinstance(params, dict):
            copied["params"] = dict(params)
        return copied

    @staticmethod
    def _ensure_environment_health_required_steps(
        steps: List[Dict[str, Any]],
        *,
        requested_branch_keys: set[str],
        available_result_ids: Optional[set[str]] = None,
    ) -> None:
        if not requested_branch_keys:
            return
        available_result_ids = available_result_ids or set()

        save_as_to_step = {
            step.get("save_as"): step
            for step in steps
            if isinstance(step, dict) and isinstance(step.get("save_as"), str)
        }
        required_save_as: List[str] = []
        for branch_key in SkillPlanner._environment_health_branch_order(requested_branch_keys):
            contract = _ENV_HEALTH_BRANCH_CONTRACTS.get(branch_key)
            if not isinstance(contract, dict):
                continue
            for save_as in contract.get("required_results", []):
                if isinstance(save_as, str) and save_as not in required_save_as:
                    required_save_as.append(save_as)

        load_context = SkillPlanner._environment_health_common_load_context(steps)
        missing_steps = []
        for save_as in required_save_as:
            if save_as in save_as_to_step or save_as in available_result_ids:
                continue
            step = SkillPlanner._environment_health_default_step(save_as)
            if step is not None:
                SkillPlanner._apply_environment_health_load_context(step, load_context)
                missing_steps.append(step)
        missing_steps = [step for step in missing_steps if step is not None]
        if not missing_steps:
            return

        required_order = {save_as: index for index, save_as in enumerate(required_save_as)}

        def insertion_index_for(save_as: str) -> int:
            target_order = required_order.get(save_as, len(required_order))
            fallback = next(
                (
                    index
                    for index, step in enumerate(steps)
                    if isinstance(step, dict)
                    and step.get("tool")
                    in {"assemble_environment_health_report", "assemble_policy_recommendation_report"}
                ),
                len(steps),
            )
            for index, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                step_order = required_order.get(step.get("save_as"))
                if step_order is not None and step_order > target_order:
                    return min(index, fallback)
            return fallback

        for step in missing_steps:
            save_as = str(step.get("save_as") or "")
            steps.insert(insertion_index_for(save_as), step)

    @staticmethod
    def _environment_health_common_load_context(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        copy_keys = ("lon_range", "lat_range", "time_range", "season_filter", "dataset", "data_path")
        for step in steps:
            if not isinstance(step, dict) or step.get("tool") != "load_dataset":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            for key in copy_keys:
                if key not in context and key in params:
                    context[key] = params[key]
            if {"lon_range", "lat_range", "time_range"}.issubset(context):
                break
        return context

    @staticmethod
    def _environment_health_request_load_context(user_request: str) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        for region_name in SkillPlanner._extract_named_region_mentions(user_request):
            known_bounds = _KNOWN_NAMED_REGION_BOUNDS.get(region_name)
            if not known_bounds:
                continue
            context["lon_range"] = list(known_bounds["lon_range"])
            context["lat_range"] = list(known_bounds["lat_range"])
            break

        time_range = SkillPlanner._time_range_from_request_text(user_request)
        if time_range is not None:
            context["time_range"] = list(time_range)

        season_filter = SkillPlanner._infer_requested_season_filter(user_request)
        if season_filter is not None:
            context["season_filter"] = season_filter

        return context

    @staticmethod
    def _apply_environment_health_load_context_to_steps(
        steps: List[Dict[str, Any]],
        load_context: Dict[str, Any],
        *,
        overwrite_keys: Optional[set[str]] = None,
    ) -> None:
        for step in steps:
            if isinstance(step, dict):
                SkillPlanner._apply_environment_health_load_context(
                    step,
                    load_context,
                    overwrite_keys=overwrite_keys,
                )

    @staticmethod
    def _apply_environment_health_load_context(
        step: Dict[str, Any],
        load_context: Dict[str, Any],
        *,
        overwrite_keys: Optional[set[str]] = None,
    ) -> None:
        if step.get("tool") != "load_dataset" or not load_context:
            return
        params = dict(step.get("params") or {})
        overwrite_keys = overwrite_keys or set()
        for key, value in load_context.items():
            if key in overwrite_keys:
                params[key] = value
            else:
                params.setdefault(key, value)
        step["params"] = params

    @staticmethod
    def _environment_health_default_step(save_as: str) -> Optional[Dict[str, Any]]:
        defaults: Dict[str, Dict[str, Any]] = {
            "sst_field": {
                "tool": "load_dataset",
                "params": {"variable": "temp", "vertical_mode": "surface"},
            },
            "sst_timeseries": {
                "tool": "compute_area_weighted_mean",
                "params": {"data": "$ref:sst_field.data", "depth_aggregation": "mean"},
            },
            "sst_trend": {
                "tool": "compute_trend",
                "params": {"timeseries": "$ref:sst_timeseries"},
            },
            "bloom_field": {
                "tool": "load_dataset",
                "params": {"variable": "chlorophyll", "vertical_mode": "surface"},
            },
            "bloom_detection": {
                "tool": "detect_algal_blooms",
                "params": {"chlorophyll": "$ref:bloom_field.data"},
            },
            "bloom_statistics": {
                "tool": "compute_event_statistics",
                "params": {"events": "$ref:bloom_detection.events", "group_by": "year"},
            },
            "bloom_event_days": {
                "tool": "compute_event_summary_map",
                "params": {"event_detection": "$ref:bloom_detection", "data": "$ref:bloom_field.data", "summary_mode": "event_days"},
            },
            "bloom_chlorophyll_burden": {
                "tool": "compute_event_summary_map",
                "params": {"event_detection": "$ref:bloom_detection", "data": "$ref:bloom_field.data", "summary_mode": "burden"},
            },
            "bottom_oxygen_field": {
                "tool": "load_dataset",
                "params": {"variable": "oxygen", "vertical_mode": "bottom"},
            },
            "bottom_oxygen_timeseries": {
                "tool": "compute_area_weighted_mean",
                "params": {"data": "$ref:bottom_oxygen_field.data", "depth_aggregation": "mean"},
            },
            "bottom_oxygen_trend": {
                "tool": "compute_trend",
                "params": {"timeseries": "$ref:bottom_oxygen_timeseries"},
            },
            "hypoxia_detection": {
                "tool": "detect_hypoxia",
                "params": {"oxygen": "$ref:bottom_oxygen_field.data", "vertical_mode": "bottom"},
            },
            "hypoxic_days": {
                "tool": "compute_event_summary_map",
                "params": {"event_detection": "$ref:hypoxia_detection", "data": "$ref:bottom_oxygen_field.data", "summary_mode": "event_days"},
            },
            "hypoxia_oxygen_deficit_burden": {
                "tool": "compute_event_summary_map",
                "params": {"event_detection": "$ref:hypoxia_detection", "data": "$ref:bottom_oxygen_field.data", "summary_mode": "burden"},
            },
            "hypoxia_statistics": {
                "tool": "compute_event_statistics",
                "params": {"events": "$ref:hypoxia_detection.events", "group_by": "year"},
            },
            "chlorophyll_context_field": {
                "tool": "load_dataset",
                "params": {"variable": "chlorophyll", "vertical_mode": "surface"},
            },
            "chlorophyll_context_timeseries": {
                "tool": "compute_area_weighted_mean",
                "params": {"data": "$ref:chlorophyll_context_field.data", "depth_aggregation": "mean"},
            },
            "chlorophyll_context_trend": {
                "tool": "compute_trend",
                "params": {"timeseries": "$ref:chlorophyll_context_timeseries"},
            },
            "heatwave_field": {
                "tool": "load_dataset",
                "params": {"variable": "temp", "vertical_mode": "surface"},
            },
            "heatwave_detection": {
                "tool": "detect_heatwaves",
                "params": {"temp": "$ref:heatwave_field.data"},
            },
            "heatwave_days": {
                "tool": "compute_event_summary_map",
                "params": {"event_detection": "$ref:heatwave_detection", "data": "$ref:heatwave_field.data", "summary_mode": "event_days"},
            },
            "heatwave_burden": {
                "tool": "compute_event_summary_map",
                "params": {"event_detection": "$ref:heatwave_detection", "data": "$ref:heatwave_field.data", "summary_mode": "burden"},
            },
            "upwelling_field": {
                "tool": "load_dataset",
                "params": {"variable": "temp", "vertical_mode": "surface"},
            },
            "upwelling_detection": {
                "tool": "detect_upwelling",
                "params": {"temp": "$ref:upwelling_field.data"},
            },
            "upwelling_days": {
                "tool": "compute_event_summary_map",
                "params": {"event_detection": "$ref:upwelling_detection", "data": "$ref:upwelling_field.data", "summary_mode": "event_days"},
            },
            "temp_field": {
                "tool": "load_dataset",
                "params": {"variable": "temp"},
            },
            "salt_field": {
                "tool": "load_dataset",
                "params": {"variable": "salt"},
            },
            "thermo_dataset": {
                "tool": "assemble_dataset",
                "params": {"variables": {"temp": "$ref:temp_field.data", "salt": "$ref:salt_field.data"}},
            },
            "density_field": {
                "tool": "compute_density",
                "params": {"data": "$ref:thermo_dataset.data"},
            },
            "stability_timeseries": {
                "tool": "compute_vertical_stability_timeseries",
                "params": {"density": "$ref:density_field.data"},
            },
            "stratification_trend": {
                "tool": "compute_trend",
                "params": {"timeseries": "$ref:stability_timeseries"},
            },
        }
        template = defaults.get(save_as)
        if template is None:
            return None
        return {
            "step_id": save_as,
            "tool": template["tool"],
            "params": dict(template.get("params") or {}),
            "save_as": save_as,
        }

    @staticmethod
    def _repair_transport_streamfunction_regional_gauge_plan(
        plan: Dict[str, Any],
        *,
        user_request: str,
        additional_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if plan.get("skill_id") != "ocean_transport_analysis":
            return plan

        steps = plan.get("steps")
        if not isinstance(steps, list):
            return plan

        should_apply = (
            SkillPlanner._active_dataset_is_cmoms(additional_context)
            and SkillPlanner._request_mentions_gan_fig10_wpo_china_seas(user_request)
            and SkillPlanner._plan_region_spans_wpo_and_china_seas(plan)
        )

        repaired_steps = [SkillPlanner._copy_step_with_params(step) for step in steps]
        changed = False
        for step in repaired_steps:
            if step.get("tool") != "compute_transport_streamfunction_map":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            if should_apply:
                if params.get("regional_gauge") != "gan_fig10_china_seas":
                    params["regional_gauge"] = "gan_fig10_china_seas"
                    changed = True
            elif "regional_gauge" in params:
                params.pop("regional_gauge", None)
                changed = True

        if not changed:
            return plan
        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    @staticmethod
    def _active_dataset_is_cmoms(additional_context: Dict[str, Any]) -> bool:
        dataset = additional_context.get("dataset")
        if not isinstance(dataset, dict):
            return False
        text = " ".join(
            str(dataset.get(key) or "")
            for key in ("id", "name", "description", "data_path")
        ).lower()
        return "cmoms" in text or "pre_wavyocean" in text

    @staticmethod
    def _request_mentions_gan_fig10_wpo_china_seas(user_request: str) -> bool:
        lowered = (user_request or "").lower()
        mentions_fig10 = bool(re.search(r"\bfig(?:ure)?\s*10\b", lowered))
        mentions_wpo = "western pacific" in lowered or "wpo" in lowered or "西太" in lowered
        mentions_china_seas = "china seas" in lowered or "中国海" in lowered
        mentions_streamfunction = "streamfunction" in lowered or "stream function" in lowered
        return mentions_streamfunction and mentions_china_seas and (mentions_wpo or mentions_fig10)

    @staticmethod
    def _plan_region_spans_wpo_and_china_seas(plan: Dict[str, Any]) -> bool:
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return False

        lon_ranges: List[List[float]] = []
        lat_ranges: List[List[float]] = []
        for step in steps:
            if not isinstance(step, dict) or step.get("tool") != "load_dataset":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            lon_range = SkillPlanner._coerce_numeric_range(params.get("lon_range"))
            lat_range = SkillPlanner._coerce_numeric_range(params.get("lat_range"))
            if lon_range is not None:
                lon_ranges.append(lon_range)
            if lat_range is not None:
                lat_ranges.append(lat_range)

        if not lon_ranges or not lat_ranges:
            return False
        lon_min = min(item[0] for item in lon_ranges)
        lon_max = max(item[1] for item in lon_ranges)
        lat_min = min(item[0] for item in lat_ranges)
        lat_max = max(item[1] for item in lat_ranges)
        covers_china_seas = lon_min <= 112.0 and lat_min <= 10.0 and lat_max >= 22.0
        covers_wpo = lon_max >= 130.0 and lat_min <= 15.0 and lat_max >= 30.0
        return covers_china_seas and covers_wpo

    @staticmethod
    def _repair_bottom_oxygen_vertical_plan(
        plan: Dict[str, Any],
        *,
        user_request: str,
    ) -> Dict[str, Any]:
        """Treat bottom oxygen as a per-cell deepest finite layer, never a fixed depth."""
        if not SkillPlanner._request_mentions_bottom_oxygen(user_request):
            return plan

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan

        repaired_steps = [SkillPlanner._copy_step_with_params(step) for step in steps]
        oxygen_load_refs: set[str] = set()
        repaired_any = False

        for step in repaired_steps:
            if step.get("tool") != "load_dataset":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            variable = SkillPlanner._canonical_variable_name(params.get("variable"))
            if variable != "oxygen":
                continue

            params["vertical_mode"] = "bottom"
            params.pop("depth_range", None)
            params.pop("depth_value", None)
            params.pop("depth_aggregation", None)
            repaired_any = True
            save_as = step.get("save_as")
            if isinstance(save_as, str) and save_as:
                oxygen_load_refs.add(save_as)

        if not oxygen_load_refs and not repaired_any:
            return plan

        downstream_tools = {
            "compute_spatial_field",
            "compute_area_weighted_mean",
            "extract_regional_mean",
            "extract_timeseries",
            "compute_hovmoller",
            "detect_hypoxia",
        }
        for step in repaired_steps:
            if step.get("tool") not in downstream_tools:
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            if not SkillPlanner._value_references_any_result(params, oxygen_load_refs):
                continue
            params.pop("depth_range", None)
            params.pop("depth_value", None)
            params.pop("depth_aggregation", None)
            repaired_any = True

        if not repaired_any:
            return plan
        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    @staticmethod
    def _repair_general_bottom_vertical_plan(
        plan: Dict[str, Any],
        *,
        user_request: str,
    ) -> Dict[str, Any]:
        """Apply deepest-finite bottom semantics for non-oxygen bottom tracer requests."""
        if not SkillPlanner._request_mentions_bottom_layer(user_request):
            return plan

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan

        repaired_steps = [SkillPlanner._copy_step_with_params(step) for step in steps]
        load_steps = [
            step
            for step in repaired_steps
            if isinstance(step, dict) and step.get("tool") == "load_dataset"
        ]
        if not load_steps:
            return plan

        bottom_load_refs: set[str] = set()
        repaired_any = False
        single_load = len(load_steps) == 1

        for step in load_steps:
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            variable = SkillPlanner._canonical_variable_name(params.get("variable"))
            if not single_load and not SkillPlanner._request_mentions_bottom_variable(user_request, variable):
                continue

            params["vertical_mode"] = "bottom"
            params.pop("depth_range", None)
            params.pop("depth_value", None)
            params.pop("depth_aggregation", None)
            repaired_any = True
            save_as = step.get("save_as")
            if isinstance(save_as, str) and save_as:
                bottom_load_refs.add(save_as)

        if not bottom_load_refs:
            return plan

        downstream_tools = {
            "compute_spatial_field",
            "compute_area_weighted_mean",
            "extract_regional_mean",
            "extract_point_timeseries",
            "extract_timeseries",
            "compute_hovmoller",
        }
        for step in repaired_steps:
            if step.get("tool") not in downstream_tools:
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            if not SkillPlanner._value_references_any_result(params, bottom_load_refs):
                continue
            params.pop("depth_range", None)
            params.pop("depth_value", None)
            params.pop("depth_aggregation", None)
            repaired_any = True

        if not repaired_any:
            return plan
        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    @staticmethod
    def _repair_point_timeseries_plan(
        plan: Dict[str, Any],
        *,
        user_request: str,
    ) -> Dict[str, Any]:
        """Convert explicit nearest-point time-series requests away from regional means."""
        skills_used = plan.get("skills_used")
        active_skills = set(skills_used if isinstance(skills_used, list) else [])
        skill_id = plan.get("skill_id")
        if isinstance(skill_id, str):
            active_skills.add(skill_id)
        if "ocean_timeseries" not in active_skills:
            return plan

        point = SkillPlanner._extract_explicit_lon_lat_point(user_request)
        if point is None or not SkillPlanner._request_mentions_point_timeseries(user_request):
            return plan

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan

        repaired_steps = [SkillPlanner._copy_step_with_params(step) for step in steps]
        load_refs: set[str] = set()
        for step in repaired_steps:
            if step.get("tool") != "load_dataset":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            params["lon_range"] = SkillPlanner._point_load_range(point["lon"])
            params["lat_range"] = SkillPlanner._point_load_range(point["lat"])
            save_as = step.get("save_as")
            if isinstance(save_as, str) and save_as:
                load_refs.add(save_as)

        if not load_refs:
            return plan

        repaired_any = False
        for step in repaired_steps:
            tool_name = step.get("tool")
            if tool_name not in {"compute_area_weighted_mean", "extract_regional_mean", "extract_timeseries"}:
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            data_ref = params.get("data")
            if not SkillPlanner._ref_targets_any(data_ref, load_refs):
                continue

            next_params: Dict[str, Any] = {
                "data": data_ref,
                "lon": point["lon"],
                "lat": point["lat"],
                "method": "nearest",
            }
            for key in ("depth_range", "depth_aggregation"):
                if key in params:
                    next_params[key] = params[key]
            step["tool"] = "extract_point_timeseries"
            step["params"] = next_params
            repaired_any = True

        if not repaired_any:
            return plan
        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    @staticmethod
    def _point_load_range(value: float, *, half_width: float = 0.25) -> List[float]:
        center = float(value)
        return [center - half_width, center + half_width]

    @staticmethod
    def _request_mentions_bottom_layer(user_request: str) -> bool:
        lowered = (user_request or "").lower()
        return bool(
            re.search(r"\b(?:bottom|bottom-layer|near-bottom|seafloor|benthic)\b", lowered)
            or "底层" in lowered
            or "近底" in lowered
        )

    @staticmethod
    def _request_mentions_bottom_variable(user_request: str, variable: str) -> bool:
        if not variable:
            return False
        lowered = (user_request or "").lower()
        aliases = {
            "chlorophyll": ("chlorophyll", "chla", "chl", "叶绿素"),
            "oxygen": ("oxygen", "o2", "dissolved oxygen", "氧", "溶解氧"),
            "temp": ("temp", "temperature", "sst", "温度"),
            "salt": ("salt", "salinity", "盐度"),
            "u": ("u", "eastward velocity", "zonal velocity"),
            "v": ("v", "northward velocity", "meridional velocity"),
        }.get(variable, (variable,))
        for alias in aliases:
            if not alias:
                continue
            pattern = (
                rf"\b(?:bottom|bottom-layer|near-bottom|seafloor|benthic)\b.{{0,40}}{re.escape(alias)}"
                rf"|{re.escape(alias)}.{{0,40}}\b(?:bottom|bottom-layer|near-bottom|seafloor|benthic)\b"
            )
            if re.search(pattern, lowered):
                return True
            if any(token in lowered for token in ("底层", "近底")) and alias in lowered:
                return True
        return False

    @staticmethod
    def _request_mentions_point_timeseries(user_request: str) -> bool:
        lowered = (user_request or "").lower()
        has_timeseries_intent = bool(
            re.search(r"\b(?:time[-\s]?series|timeseries|plot|chart|curve)\b", lowered)
            or "时间序列" in lowered
        )
        has_point_intent = bool(
            re.search(r"\b(?:nearest|point|location|longitude|latitude|lon|lat)\b", lowered)
            or "点位" in lowered
            or "最近" in lowered
        )
        return has_timeseries_intent and has_point_intent

    @staticmethod
    def _extract_explicit_lon_lat_point(user_request: str) -> Optional[Dict[str, float]]:
        text = user_request or ""
        lon = SkillPlanner._extract_labeled_coordinate(text, ("longitude", "lon"), "EW")
        lat = SkillPlanner._extract_labeled_coordinate(text, ("latitude", "lat"), "NS")
        if lon is not None and lat is not None:
            return {"lon": lon, "lat": lat}

        pair = re.search(
            r"([-+]?\d+(?:\.\d+)?)\s*[^0-9A-Za-z+-]*([EeWw])"
            r".{0,40}?"
            r"([-+]?\d+(?:\.\d+)?)\s*[^0-9A-Za-z+-]*([NnSs])",
            text,
        )
        if pair:
            return {
                "lon": SkillPlanner._apply_hemisphere(float(pair.group(1)), pair.group(2)),
                "lat": SkillPlanner._apply_hemisphere(float(pair.group(3)), pair.group(4)),
            }
        return None

    @staticmethod
    def _extract_labeled_coordinate(
        text: str,
        labels: Tuple[str, ...],
        hemispheres: str,
    ) -> Optional[float]:
        label_pattern = "|".join(re.escape(label) for label in labels)
        match = re.search(
            rf"\b(?:{label_pattern})\b[^0-9+-]*([-+]?\d+(?:\.\d+)?)"
            rf"[^0-9A-Za-z+-]*([{hemispheres}{hemispheres.lower()}])?",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None
        return SkillPlanner._apply_hemisphere(float(match.group(1)), match.group(2))

    @staticmethod
    def _apply_hemisphere(value: float, hemisphere: Optional[str]) -> float:
        if hemisphere and hemisphere.upper() in {"W", "S"}:
            return -abs(float(value))
        return float(value)

    @staticmethod
    def _repair_environment_health_standard_steps(save_as_to_step: Dict[Any, Dict[str, Any]]) -> None:
        SkillPlanner._patch_load_dataset_step(
            save_as_to_step,
            "sst_field",
            variable="temp",
            vertical_mode="surface",
        )
        SkillPlanner._patch_area_mean_step(
            save_as_to_step,
            "sst_timeseries",
            data_ref="$ref:sst_field.data",
            remove_depth_range=True,
        )
        SkillPlanner._patch_trend_step(
            save_as_to_step,
            "sst_trend",
            timeseries_ref="$ref:sst_timeseries",
        )

        SkillPlanner._patch_load_dataset_step(
            save_as_to_step,
            "bloom_field",
            variable="chlorophyll",
            vertical_mode="surface",
        )
        SkillPlanner._patch_bloom_detection_step(save_as_to_step)
        SkillPlanner._patch_bloom_statistics_step(save_as_to_step)
        SkillPlanner._patch_event_summary_map_step(
            save_as_to_step,
            "bloom_event_days",
            event_detection_ref="$ref:bloom_detection",
            data_ref="$ref:bloom_field.data",
            summary_mode="event_days",
        )
        SkillPlanner._patch_event_summary_map_step(
            save_as_to_step,
            "bloom_chlorophyll_burden",
            event_detection_ref="$ref:bloom_detection",
            data_ref="$ref:bloom_field.data",
            summary_mode="burden",
        )
        SkillPlanner._patch_load_dataset_step(
            save_as_to_step,
            "bottom_oxygen_field",
            variable="oxygen",
            vertical_mode="bottom",
        )
        SkillPlanner._patch_area_mean_step(
            save_as_to_step,
            "bottom_oxygen_timeseries",
            data_ref="$ref:bottom_oxygen_field.data",
            remove_depth_range=True,
        )
        SkillPlanner._patch_trend_step(
            save_as_to_step,
            "bottom_oxygen_trend",
            timeseries_ref="$ref:bottom_oxygen_timeseries",
        )
        SkillPlanner._patch_hypoxia_detection_step(save_as_to_step)
        SkillPlanner._patch_hypoxia_statistics_step(save_as_to_step)
        SkillPlanner._patch_event_summary_map_step(
            save_as_to_step,
            "hypoxic_days",
            event_detection_ref="$ref:hypoxia_detection",
            data_ref="$ref:bottom_oxygen_field.data",
            summary_mode="event_days",
        )
        SkillPlanner._patch_event_summary_map_step(
            save_as_to_step,
            "hypoxia_oxygen_deficit_burden",
            event_detection_ref="$ref:hypoxia_detection",
            data_ref="$ref:bottom_oxygen_field.data",
            summary_mode="burden",
        )
        SkillPlanner._patch_load_dataset_step(
            save_as_to_step,
            "chlorophyll_context_field",
            variable="chlorophyll",
            vertical_mode="surface",
        )
        SkillPlanner._patch_area_mean_step(
            save_as_to_step,
            "chlorophyll_context_timeseries",
            data_ref="$ref:chlorophyll_context_field.data",
            remove_depth_range=True,
        )
        SkillPlanner._patch_trend_step(
            save_as_to_step,
            "chlorophyll_context_trend",
            timeseries_ref="$ref:chlorophyll_context_timeseries",
        )

        SkillPlanner._patch_load_dataset_step(
            save_as_to_step,
            "heatwave_field",
            variable="temp",
            vertical_mode="surface",
        )
        SkillPlanner._patch_event_detection_ref_step(
            save_as_to_step,
            "heatwave_detection",
            tool="detect_heatwaves",
            field_param="temp",
            field_ref="$ref:heatwave_field.data",
            remove_depth_range=True,
        )
        SkillPlanner._patch_event_summary_map_step(
            save_as_to_step,
            "heatwave_days",
            event_detection_ref="$ref:heatwave_detection",
            data_ref="$ref:heatwave_field.data",
            summary_mode="event_days",
        )
        SkillPlanner._patch_event_summary_map_step(
            save_as_to_step,
            "heatwave_burden",
            event_detection_ref="$ref:heatwave_detection",
            data_ref="$ref:heatwave_field.data",
            summary_mode="burden",
        )
        SkillPlanner._patch_load_dataset_step(
            save_as_to_step,
            "upwelling_field",
            variable="temp",
            vertical_mode="surface",
        )
        SkillPlanner._patch_event_detection_ref_step(
            save_as_to_step,
            "upwelling_detection",
            tool="detect_upwelling",
            field_param="temp",
            field_ref="$ref:upwelling_field.data",
            remove_depth_range=True,
        )
        SkillPlanner._patch_event_summary_map_step(
            save_as_to_step,
            "upwelling_days",
            event_detection_ref="$ref:upwelling_detection",
            data_ref="$ref:upwelling_field.data",
            summary_mode="event_days",
        )

        SkillPlanner._patch_load_dataset_step(
            save_as_to_step,
            "temp_field",
            variable="temp",
            remove_single_level_depth=True,
        )
        SkillPlanner._patch_load_dataset_step(
            save_as_to_step,
            "salt_field",
            variable="salt",
            remove_single_level_depth=True,
        )
        SkillPlanner._patch_assemble_dataset_step(
            save_as_to_step,
            "thermo_dataset",
            variables={
                "temp": "$ref:temp_field.data",
                "salt": "$ref:salt_field.data",
            },
        )
        SkillPlanner._patch_data_ref_step(
            save_as_to_step,
            "density_field",
            tool="compute_density",
            param_name="data",
            ref="$ref:thermo_dataset.data",
        )
        SkillPlanner._patch_data_ref_step(
            save_as_to_step,
            "stability_timeseries",
            tool="compute_vertical_stability_timeseries",
            param_name="density",
            ref="$ref:density_field.data",
            remove_single_level_depth=True,
        )
        SkillPlanner._patch_trend_step(
            save_as_to_step,
            "stratification_trend",
            timeseries_ref="$ref:stability_timeseries",
        )

    @staticmethod
    def _patch_load_dataset_step(
        save_as_to_step: Dict[Any, Dict[str, Any]],
        save_as: str,
        *,
        variable: str,
        depth_range: Optional[List[float]] = None,
        vertical_mode: Optional[str] = None,
        remove_single_level_depth: bool = False,
    ) -> None:
        step = save_as_to_step.get(save_as)
        if not isinstance(step, dict) or step.get("tool") != "load_dataset":
            return
        params = dict(step.get("params") or {})
        params["variable"] = variable
        normalized_vertical_mode = vertical_mode.strip().lower() if isinstance(vertical_mode, str) else None
        if normalized_vertical_mode in {"bottom", "surface"}:
            params["vertical_mode"] = normalized_vertical_mode
            params.pop("depth_value", None)
            params.pop("depth_range", None)
        elif depth_range is not None:
            params["depth_range"] = list(depth_range)
            params.pop("vertical_mode", None)
            params.pop("depth_value", None)
        elif remove_single_level_depth:
            SkillPlanner._remove_single_level_vertical_selection(params)
        step["params"] = params

    @staticmethod
    def _patch_area_mean_step(
        save_as_to_step: Dict[Any, Dict[str, Any]],
        save_as: str,
        *,
        data_ref: str,
        depth_range: Optional[List[float]] = None,
        remove_depth_range: bool = False,
    ) -> None:
        step = save_as_to_step.get(save_as)
        if not isinstance(step, dict) or step.get("tool") != "compute_area_weighted_mean":
            return
        params = dict(step.get("params") or {})
        params["data"] = data_ref
        params.setdefault("depth_aggregation", "mean")
        if remove_depth_range:
            params.pop("depth_range", None)
        elif depth_range is not None:
            params["depth_range"] = list(depth_range)
        step["params"] = params

    @staticmethod
    def _patch_trend_step(
        save_as_to_step: Dict[Any, Dict[str, Any]],
        save_as: str,
        *,
        timeseries_ref: str,
    ) -> None:
        step = save_as_to_step.get(save_as)
        if not isinstance(step, dict) or step.get("tool") != "compute_trend":
            return
        params = dict(step.get("params") or {})
        params["timeseries"] = timeseries_ref
        step["params"] = params

    @staticmethod
    def _patch_assemble_dataset_step(
        save_as_to_step: Dict[Any, Dict[str, Any]],
        save_as: str,
        *,
        variables: Dict[str, str],
    ) -> None:
        step = save_as_to_step.get(save_as)
        if not isinstance(step, dict) or step.get("tool") != "assemble_dataset":
            return
        params = dict(step.get("params") or {})
        params["variables"] = dict(variables)
        step["params"] = params

    @staticmethod
    def _patch_bloom_detection_step(save_as_to_step: Dict[Any, Dict[str, Any]]) -> None:
        step = save_as_to_step.get("bloom_detection")
        if not isinstance(step, dict) or step.get("tool") != "detect_algal_blooms":
            return
        params = dict(step.get("params") or {})
        params["chlorophyll"] = "$ref:bloom_field.data"
        params.pop("depth_range", None)
        params.pop("depth_aggregation", None)
        params.pop("vertical_mode", None)
        params.pop("depth_value", None)
        step["params"] = params

    @staticmethod
    def _patch_bloom_statistics_step(save_as_to_step: Dict[Any, Dict[str, Any]]) -> None:
        step = save_as_to_step.get("bloom_statistics")
        if not isinstance(step, dict) or step.get("tool") != "compute_event_statistics":
            return
        params = dict(step.get("params") or {})
        params["events"] = "$ref:bloom_detection.events"
        params["group_by"] = "year"
        step["params"] = params

    @staticmethod
    def _patch_hypoxia_statistics_step(save_as_to_step: Dict[Any, Dict[str, Any]]) -> None:
        step = save_as_to_step.get("hypoxia_statistics")
        if not isinstance(step, dict) or step.get("tool") != "compute_event_statistics":
            return
        params = dict(step.get("params") or {})
        params["events"] = "$ref:hypoxia_detection.events"
        params["group_by"] = "year"
        step["params"] = params

    @staticmethod
    def _patch_hypoxia_detection_step(save_as_to_step: Dict[Any, Dict[str, Any]]) -> None:
        for step in save_as_to_step.values():
            if not isinstance(step, dict) or step.get("tool") != "detect_hypoxia":
                continue
            params = dict(step.get("params") or {})
            params["oxygen"] = "$ref:bottom_oxygen_field.data"
            params["vertical_mode"] = "bottom"
            params.pop("depth_value", None)
            params.pop("depth_range", None)
            step["params"] = params

    @staticmethod
    def _patch_event_detection_ref_step(
        save_as_to_step: Dict[Any, Dict[str, Any]],
        save_as: str,
        *,
        tool: str,
        field_param: str,
        field_ref: str,
        depth_range: Optional[List[float]] = None,
        remove_depth_range: bool = False,
    ) -> None:
        step = save_as_to_step.get(save_as)
        if not isinstance(step, dict) or step.get("tool") != tool:
            return
        params = dict(step.get("params") or {})
        params[field_param] = field_ref
        if remove_depth_range:
            params.pop("depth_range", None)
            params.pop("depth_aggregation", None)
            params.pop("vertical_mode", None)
            params.pop("depth_value", None)
        elif depth_range is not None:
            params["depth_range"] = list(depth_range)
            params["depth_aggregation"] = "mean"
            params.pop("vertical_mode", None)
            params.pop("depth_value", None)
        step["params"] = params

    @staticmethod
    def _patch_event_summary_map_step(
        save_as_to_step: Dict[Any, Dict[str, Any]],
        save_as: str,
        *,
        event_detection_ref: str,
        data_ref: str,
        summary_mode: str,
    ) -> None:
        step = save_as_to_step.get(save_as)
        if not isinstance(step, dict) or step.get("tool") != "compute_event_summary_map":
            return
        params = dict(step.get("params") or {})
        params["event_detection"] = event_detection_ref
        params["data"] = data_ref
        params["summary_mode"] = summary_mode
        step["params"] = params

    @staticmethod
    def _patch_data_ref_step(
        save_as_to_step: Dict[Any, Dict[str, Any]],
        save_as: str,
        *,
        tool: str,
        param_name: str,
        ref: str,
        remove_single_level_depth: bool = False,
    ) -> None:
        step = save_as_to_step.get(save_as)
        if not isinstance(step, dict) or step.get("tool") != tool:
            return
        params = dict(step.get("params") or {})
        params[param_name] = ref
        if remove_single_level_depth:
            depth_range = SkillPlanner._coerce_depth_range_static(params.get("depth_range"))
            if depth_range is not None and SkillPlanner._is_single_depth_range(depth_range):
                params.pop("depth_range", None)
        step["params"] = params

    @staticmethod
    def _repair_environment_health_branch_payloads(
        steps: List[Dict[str, Any]],
        *,
        user_request: str,
    ) -> set[str]:
        requested_branch_keys = SkillPlanner._environment_health_requested_branch_keys(user_request)
        suppressed_branch_keys = SkillPlanner._environment_health_suppressed_branch_keys(user_request)
        requested_branch_keys.difference_update(suppressed_branch_keys)
        assemble_steps = [step for step in steps if step.get("tool") == "assemble_environment_health_report"]
        if not assemble_steps:
            return requested_branch_keys

        for step in assemble_steps:
            params = dict(step.get("params") or {})
            branches = params.get("branches")
            if not isinstance(branches, list):
                branches = []

            repaired_branches: List[Any] = []
            for branch in branches:
                if not isinstance(branch, dict):
                    repaired_branches.append(branch)
                    continue
                repaired_branch = dict(branch)
                branch_key = SkillPlanner._environment_health_branch_key(repaired_branch)
                if branch_key in suppressed_branch_keys:
                    continue
                if branch_key in _ENV_HEALTH_BRANCH_CONTRACTS:
                    contract = _ENV_HEALTH_BRANCH_CONTRACTS[branch_key]
                    requested_branch_keys.add(branch_key)
                    repaired_branch["name"] = contract["name"]
                    repaired_branch["indicator_label"] = contract["indicator_label"]
                    repaired_branch["worse_when"] = contract["worse_when"]
                    repaired_branch["result"] = contract["result_ref"]
                    for key in ("role", "evidence_kind", "metric"):
                        if key in contract:
                            repaired_branch[key] = contract[key]
                repaired_branches.append(repaired_branch)

            present_branch_keys = {
                SkillPlanner._environment_health_branch_key(branch)
                for branch in repaired_branches
                if isinstance(branch, dict)
            }
            for branch_key in SkillPlanner._environment_health_branch_order(requested_branch_keys):
                if branch_key in suppressed_branch_keys or branch_key in present_branch_keys:
                    continue
                contract = _ENV_HEALTH_BRANCH_CONTRACTS.get(branch_key)
                if not isinstance(contract, dict):
                    continue
                missing_branch = {
                    "name": contract["name"],
                    "indicator_label": contract["indicator_label"],
                    "worse_when": contract["worse_when"],
                    "result": contract["result_ref"],
                }
                for key in ("role", "evidence_kind", "metric"):
                    if key in contract:
                        missing_branch[key] = contract[key]
                repaired_branches.append(missing_branch)
                present_branch_keys.add(branch_key)
            params["branches"] = repaired_branches
            step["params"] = params

        return requested_branch_keys

    @staticmethod
    def _environment_health_branch_order(branch_keys: Iterable[str]) -> List[str]:
        requested = set(branch_keys)
        ordered = [key for key in _ENV_HEALTH_BRANCH_CONTRACTS if key in requested]
        ordered.extend(sorted(requested.difference(ordered)))
        return ordered

    @staticmethod
    def _validate_environment_health_plan(
        plan: Dict[str, Any],
        *,
        requested_branch_keys: set[str],
        require_report: bool = False,
    ) -> None:
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return

        assemble_steps = [step for step in steps if step.get("tool") == "assemble_environment_health_report"]
        if not assemble_steps:
            if require_report:
                raise ValueError("Explicit environment health report request must include assemble_environment_health_report.")
            return

        save_as_to_step = {
            step.get("save_as"): step
            for step in steps
            if isinstance(step, dict) and isinstance(step.get("save_as"), str)
        }

        branch_keys_from_payload: List[str] = []
        branch_results_by_key: Dict[str, str] = {}
        configured_branch_refs: List[str] = []
        for step in assemble_steps:
            params = step.get("params")
            branches = params.get("branches") if isinstance(params, dict) else None
            if not isinstance(branches, list) or not branches:
                raise ValueError("Environment health report requires a non-empty branches list.")
            for branch in branches:
                if not isinstance(branch, dict):
                    continue
                branch_key = SkillPlanner._environment_health_branch_key(branch)
                if branch_key not in _ENV_HEALTH_BRANCH_CONTRACTS:
                    if SkillPlanner._is_configured_environment_evidence_branch(branch):
                        result_ref = branch.get("result")
                        if isinstance(result_ref, str) and result_ref.strip():
                            configured_branch_refs.append(result_ref)
                    continue
                if branch_key in branch_results_by_key:
                    raise ValueError(f"Environment health branch '{branch_key}' appears more than once.")
                branch_keys_from_payload.append(branch_key)
                result_ref = branch.get("result")
                branch_results_by_key[branch_key] = result_ref if isinstance(result_ref, str) else ""

        expected_branch_keys = set(branch_keys_from_payload) | set(requested_branch_keys)
        if not expected_branch_keys and not configured_branch_refs:
            raise ValueError("Environment health plan did not identify any supported evidence branches.")

        for branch_key in sorted(expected_branch_keys):
            contract = _ENV_HEALTH_BRANCH_CONTRACTS[branch_key]
            expected_ref = str(contract["result_ref"])
            actual_ref = branch_results_by_key.get(branch_key)
            if actual_ref != expected_ref:
                raise ValueError(
                    f"Environment health branch '{branch_key}' must use result {expected_ref}, "
                    f"not {actual_ref or 'missing'}."
                )
            for save_as in contract["required_results"]:
                if save_as not in save_as_to_step:
                    raise ValueError(
                        f"Environment health branch '{branch_key}' is missing required step result '{save_as}'."
                    )

        recognized_refs = [branch_results_by_key[key] for key in branch_keys_from_payload] + configured_branch_refs
        if len(recognized_refs) != len(set(recognized_refs)):
            raise ValueError("Environment health branches must not reuse the same result reference.")

        SkillPlanner._validate_environment_health_standard_steps(save_as_to_step, expected_branch_keys)

    @staticmethod
    def _validate_environment_health_standard_steps(
        save_as_to_step: Dict[Any, Dict[str, Any]],
        expected_branch_keys: set[str],
    ) -> None:
        if "sst_trend" in expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "sst_field", "load_dataset")
            SkillPlanner._require_surface_load_step(save_as_to_step, "sst_field", "temp")
            SkillPlanner._require_step_tool(save_as_to_step, "sst_timeseries", "compute_area_weighted_mean")
            SkillPlanner._require_step_param(save_as_to_step, "sst_timeseries", "data", "$ref:sst_field.data")
            SkillPlanner._require_step_tool(save_as_to_step, "sst_trend", "compute_trend")
            SkillPlanner._require_step_param(save_as_to_step, "sst_trend", "timeseries", "$ref:sst_timeseries")

        if "bottom_oxygen_trend" in expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "bottom_oxygen_field", "load_dataset")
            oxygen_params = save_as_to_step["bottom_oxygen_field"].get("params")
            if not isinstance(oxygen_params, dict):
                raise ValueError("bottom_oxygen_field must contain params.")
            if SkillPlanner._canonical_variable_name(oxygen_params.get("variable")) != "oxygen":
                raise ValueError("bottom_oxygen_field must load variable='oxygen'.")
            if str(oxygen_params.get("vertical_mode") or "").strip().lower() != "bottom":
                raise ValueError("bottom_oxygen_field must use vertical_mode='bottom'.")
            if "depth_range" in oxygen_params or "depth_value" in oxygen_params:
                raise ValueError("bottom_oxygen_field must not use fixed-depth depth_range/depth_value.")

            SkillPlanner._require_step_tool(save_as_to_step, "bottom_oxygen_timeseries", "compute_area_weighted_mean")
            SkillPlanner._require_step_param(
                save_as_to_step,
                "bottom_oxygen_timeseries",
                "data",
                "$ref:bottom_oxygen_field.data",
            )
            SkillPlanner._require_step_tool(save_as_to_step, "bottom_oxygen_trend", "compute_trend")
            SkillPlanner._require_step_param(
                save_as_to_step,
                "bottom_oxygen_trend",
                "timeseries",
                "$ref:bottom_oxygen_timeseries",
            )

        if {"bottom_hypoxia_burden", "hypoxic_days", "hypoxia_statistics"} & expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "bottom_oxygen_field", "load_dataset")
            SkillPlanner._require_step_tool(save_as_to_step, "hypoxia_detection", "detect_hypoxia")
            SkillPlanner._require_step_param(save_as_to_step, "hypoxia_detection", "oxygen", "$ref:bottom_oxygen_field.data")
            hypoxia_params = save_as_to_step["hypoxia_detection"].get("params")
            if not isinstance(hypoxia_params, dict):
                raise ValueError("hypoxia_detection must contain params.")
            if str(hypoxia_params.get("vertical_mode") or "").strip().lower() != "bottom":
                raise ValueError("hypoxia_detection must use vertical_mode='bottom'.")
        if "bottom_hypoxia_burden" in expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "hypoxia_oxygen_deficit_burden", "compute_event_summary_map")
            SkillPlanner._require_step_param(save_as_to_step, "hypoxia_oxygen_deficit_burden", "event_detection", "$ref:hypoxia_detection")
            SkillPlanner._require_step_param(save_as_to_step, "hypoxia_oxygen_deficit_burden", "data", "$ref:bottom_oxygen_field.data")
            SkillPlanner._require_step_param(save_as_to_step, "hypoxia_oxygen_deficit_burden", "summary_mode", "burden")
        if "hypoxia_statistics" in expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "hypoxia_statistics", "compute_event_statistics")
            SkillPlanner._require_step_param(save_as_to_step, "hypoxia_statistics", "events", "$ref:hypoxia_detection.events")
            SkillPlanner._require_step_param(save_as_to_step, "hypoxia_statistics", "group_by", "year")
        if "hypoxic_days" in expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "hypoxic_days", "compute_event_summary_map")
            SkillPlanner._require_step_param(save_as_to_step, "hypoxic_days", "event_detection", "$ref:hypoxia_detection")
            SkillPlanner._require_step_param(save_as_to_step, "hypoxic_days", "data", "$ref:bottom_oxygen_field.data")
            SkillPlanner._require_step_param(save_as_to_step, "hypoxic_days", "summary_mode", "event_days")

        if "stratification_strength_change" in expected_branch_keys:
            for save_as, variable in (("temp_field", "temp"), ("salt_field", "salt")):
                SkillPlanner._require_step_tool(save_as_to_step, save_as, "load_dataset")
                params = save_as_to_step[save_as].get("params")
                if not isinstance(params, dict):
                    raise ValueError(f"{save_as} must contain params.")
                if SkillPlanner._canonical_variable_name(params.get("variable")) != variable:
                    raise ValueError(f"{save_as} must load variable='{variable}'.")
                depth_range = SkillPlanner._coerce_depth_range_static(params.get("depth_range"))
                if depth_range is not None and SkillPlanner._is_single_depth_range(depth_range):
                    raise ValueError(f"{save_as} must retain a multi-level water-column selection.")
                vertical_mode = str(params.get("vertical_mode") or "").strip().lower()
                if vertical_mode in {"bottom", "surface", "fixed_depth"}:
                    raise ValueError(f"{save_as} must not use single-level vertical_mode='{vertical_mode}'.")

            SkillPlanner._require_step_tool(save_as_to_step, "thermo_dataset", "assemble_dataset")
            SkillPlanner._require_step_param(
                save_as_to_step,
                "thermo_dataset",
                "variables",
                {"temp": "$ref:temp_field.data", "salt": "$ref:salt_field.data"},
            )
            SkillPlanner._require_step_tool(save_as_to_step, "density_field", "compute_density")
            SkillPlanner._require_step_param(save_as_to_step, "density_field", "data", "$ref:thermo_dataset.data")
            SkillPlanner._require_step_tool(
                save_as_to_step,
                "stability_timeseries",
                "compute_vertical_stability_timeseries",
            )
            SkillPlanner._require_step_param(save_as_to_step, "stability_timeseries", "density", "$ref:density_field.data")
            SkillPlanner._require_step_tool(save_as_to_step, "stratification_trend", "compute_trend")
            SkillPlanner._require_step_param(
                save_as_to_step,
                "stratification_trend",
                "timeseries",
                "$ref:stability_timeseries",
            )

        if {"bloom_frequency_change", "bloom_burden", "bloom_event_days"} & expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "bloom_field", "load_dataset")
            bloom_params = save_as_to_step["bloom_field"].get("params")
            if not isinstance(bloom_params, dict):
                raise ValueError("bloom_field must contain params.")
            if SkillPlanner._canonical_variable_name(bloom_params.get("variable")) != "chlorophyll":
                raise ValueError("bloom_field must load variable='chlorophyll'.")
            if str(bloom_params.get("vertical_mode") or "").strip().lower() != "surface":
                raise ValueError("bloom_field must use vertical_mode='surface'.")
            if "depth_range" in bloom_params or "depth_value" in bloom_params:
                raise ValueError("bloom_field must not use fixed surface depth_range/depth_value.")
            SkillPlanner._require_step_tool(save_as_to_step, "bloom_detection", "detect_algal_blooms")
            SkillPlanner._require_step_param(save_as_to_step, "bloom_detection", "chlorophyll", "$ref:bloom_field.data")
        if "bloom_frequency_change" in expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "bloom_statistics", "compute_event_statistics")
            SkillPlanner._require_step_param(save_as_to_step, "bloom_statistics", "events", "$ref:bloom_detection.events")
            SkillPlanner._require_step_param(save_as_to_step, "bloom_statistics", "group_by", "year")

        if "bloom_burden" in expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "bloom_chlorophyll_burden", "compute_event_summary_map")
            SkillPlanner._require_step_param(save_as_to_step, "bloom_chlorophyll_burden", "event_detection", "$ref:bloom_detection")
            SkillPlanner._require_step_param(save_as_to_step, "bloom_chlorophyll_burden", "data", "$ref:bloom_field.data")
            SkillPlanner._require_step_param(save_as_to_step, "bloom_chlorophyll_burden", "summary_mode", "burden")
        if "bloom_event_days" in expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "bloom_event_days", "compute_event_summary_map")
            SkillPlanner._require_step_param(save_as_to_step, "bloom_event_days", "event_detection", "$ref:bloom_detection")
            SkillPlanner._require_step_param(save_as_to_step, "bloom_event_days", "data", "$ref:bloom_field.data")
            SkillPlanner._require_step_param(save_as_to_step, "bloom_event_days", "summary_mode", "event_days")

        if "eutrophication_context" in expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "chlorophyll_context_field", "load_dataset")
            SkillPlanner._require_surface_load_step(save_as_to_step, "chlorophyll_context_field", "chlorophyll")
            SkillPlanner._require_step_tool(save_as_to_step, "chlorophyll_context_timeseries", "compute_area_weighted_mean")
            SkillPlanner._require_step_param(save_as_to_step, "chlorophyll_context_timeseries", "data", "$ref:chlorophyll_context_field.data")
            SkillPlanner._require_step_tool(save_as_to_step, "chlorophyll_context_trend", "compute_trend")
            SkillPlanner._require_step_param(save_as_to_step, "chlorophyll_context_trend", "timeseries", "$ref:chlorophyll_context_timeseries")

        if "heatwave_burden" in expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "heatwave_field", "load_dataset")
            SkillPlanner._require_surface_load_step(save_as_to_step, "heatwave_field", "temp")
            SkillPlanner._require_step_tool(save_as_to_step, "heatwave_detection", "detect_heatwaves")
            SkillPlanner._require_step_param(save_as_to_step, "heatwave_detection", "temp", "$ref:heatwave_field.data")
            SkillPlanner._require_step_tool(save_as_to_step, "heatwave_burden", "compute_event_summary_map")
            SkillPlanner._require_step_param(save_as_to_step, "heatwave_burden", "event_detection", "$ref:heatwave_detection")
            SkillPlanner._require_step_param(save_as_to_step, "heatwave_burden", "data", "$ref:heatwave_field.data")
            SkillPlanner._require_step_param(save_as_to_step, "heatwave_burden", "summary_mode", "burden")
        if "heatwave_days" in expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "heatwave_field", "load_dataset")
            SkillPlanner._require_surface_load_step(save_as_to_step, "heatwave_field", "temp")
            SkillPlanner._require_step_tool(save_as_to_step, "heatwave_detection", "detect_heatwaves")
            SkillPlanner._require_step_param(save_as_to_step, "heatwave_detection", "temp", "$ref:heatwave_field.data")
            SkillPlanner._require_step_tool(save_as_to_step, "heatwave_days", "compute_event_summary_map")
            SkillPlanner._require_step_param(save_as_to_step, "heatwave_days", "event_detection", "$ref:heatwave_detection")
            SkillPlanner._require_step_param(save_as_to_step, "heatwave_days", "data", "$ref:heatwave_field.data")
            SkillPlanner._require_step_param(save_as_to_step, "heatwave_days", "summary_mode", "event_days")
        if "upwelling_days" in expected_branch_keys:
            SkillPlanner._require_step_tool(save_as_to_step, "upwelling_field", "load_dataset")
            SkillPlanner._require_surface_load_step(save_as_to_step, "upwelling_field", "temp")
            SkillPlanner._require_step_tool(save_as_to_step, "upwelling_detection", "detect_upwelling")
            SkillPlanner._require_step_param(save_as_to_step, "upwelling_detection", "temp", "$ref:upwelling_field.data")
            SkillPlanner._require_step_tool(save_as_to_step, "upwelling_days", "compute_event_summary_map")
            SkillPlanner._require_step_param(save_as_to_step, "upwelling_days", "event_detection", "$ref:upwelling_detection")
            SkillPlanner._require_step_param(save_as_to_step, "upwelling_days", "data", "$ref:upwelling_field.data")
            SkillPlanner._require_step_param(save_as_to_step, "upwelling_days", "summary_mode", "event_days")

    @staticmethod
    def _require_step_tool(save_as_to_step: Dict[Any, Dict[str, Any]], save_as: str, expected_tool: str) -> None:
        step = save_as_to_step.get(save_as)
        if not isinstance(step, dict):
            raise ValueError(f"Missing required step result '{save_as}'.")
        if step.get("tool") != expected_tool:
            raise ValueError(f"Step result '{save_as}' must use tool '{expected_tool}', not '{step.get('tool')}'.")

    @staticmethod
    def _require_step_param(
        save_as_to_step: Dict[Any, Dict[str, Any]],
        save_as: str,
        param_name: str,
        expected_value: Any,
    ) -> None:
        params = save_as_to_step[save_as].get("params")
        if not isinstance(params, dict):
            raise ValueError(f"Step result '{save_as}' must contain params.")
        if params.get(param_name) != expected_value:
            raise ValueError(
                f"Step result '{save_as}' must set {param_name}={expected_value!r}, "
                f"not {params.get(param_name)!r}."
            )

    @staticmethod
    def _require_surface_load_step(save_as_to_step: Dict[Any, Dict[str, Any]], save_as: str, variable: str) -> None:
        params = save_as_to_step[save_as].get("params")
        if not isinstance(params, dict):
            raise ValueError(f"Step result '{save_as}' must contain params.")
        if SkillPlanner._canonical_variable_name(params.get("variable")) != variable:
            raise ValueError(f"{save_as} must load variable='{variable}'.")
        if not SkillPlanner._uses_surface_vertical_selection(params):
            raise ValueError(f"{save_as} must use vertical_mode='surface' or depth_range=[0, 0].")

    @staticmethod
    def _uses_surface_vertical_selection(params: Dict[str, Any]) -> bool:
        vertical_mode = str(params.get("vertical_mode") or "").strip().lower()
        if vertical_mode == "surface":
            return True
        if vertical_mode == "bottom":
            return False

        depth_range = SkillPlanner._coerce_depth_range_static(params.get("depth_range"))
        if depth_range is not None:
            return (
                SkillPlanner._is_single_depth_range(depth_range)
                and abs(float(depth_range[0])) <= 1e-6
            )

        depth_value = params.get("depth_value")
        try:
            return abs(float(depth_value)) <= 1e-6
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _environment_health_requested_branch_keys(user_request: str) -> set[str]:
        lowered = (user_request or "").lower()
        requested: set[str] = set()
        suppressed = SkillPlanner._environment_health_suppressed_branch_keys(user_request)
        trend_intent = bool(
            re.search(
                r"\b(?:trend|trends|change|changes|changed|increase|decrease|worsen|"
                r"recent years|over time|time series|year[-\s]?by[-\s]?year|yearly|"
                r"interannual|coming years|remain|future)\b|"
                r"趋势|变化|近几年|近年来|未来|持续|年际|逐年|时间序列",
                lowered,
            )
        )
        spatial_risk_intent = bool(
            re.search(
                r"\b(?:where|area|areas|risk|hotspot|hotspots|map|spatial|burden|"
                r"exposure|suitable|suitability|aquaculture|marine ranching|fishery|fisheries)\b|"
                r"哪里|区域|风险|高风险|热点|空间|分布|负担|暴露|适合|适宜|养殖|海洋牧场|渔业",
                lowered,
            )
        )
        suitability_intent = bool(
            re.search(
                r"\b(?:suitable|suitability|aquaculture|marine ranching|fish farming|fishery|fisheries)\b|"
                r"适合|适宜|养殖|海洋牧场|渔业",
                lowered,
            )
        )
        hypoxia_intent = bool(
            re.search(
                r"\b(?:hypoxia|hypoxic|low[-\s]?oxygen|oxygen[-\s]?deficit|oxygen deficit)\b|"
                r"缺氧|低氧|氧亏缺|氧气亏缺",
                lowered,
            )
        )
        oxygen_intent = bool(
            re.search(
                r"\b(?:bottom[-\s]?oxygen|dissolved oxygen|oxygen)\b|"
                r"底层氧|底层溶解氧|溶解氧|氧含量|氧气",
                lowered,
            )
        )
        if re.search(r"\b(?:sst|sea[-\s]?surface temperature|surface temperature|temperature trend)\b|海表温度|表层温度|升温|增温", lowered):
            requested.add("sst_trend")
        if hypoxia_intent:
            requested.add("bottom_hypoxia_burden")
            if trend_intent or re.search(
                r"\b(?:frequency|statistics|count|counts|duration|by year|"
                r"year[-\s]?by[-\s]?year|yearly|annual|interannual)\b|"
                r"频率|统计|次数|持续时间|逐年|年际",
                lowered,
            ):
                requested.add("hypoxia_statistics")
            if trend_intent or oxygen_intent:
                requested.add("bottom_oxygen_trend")
        if re.search(r"\bhypoxic[-\s]?days?\b|缺氧天数|低氧天数", lowered):
            requested.add("hypoxic_days")
        if (
            re.search(r"\b(?:bloom|chlorophyll|chla)\b|藻华|叶绿素|富营养", lowered)
            and "bloom_frequency_change" not in suppressed
            and not SkillPlanner._environment_health_auxiliary_chlorophyll_requested(lowered)
        ):
            requested.add("bloom_frequency_change")
        if re.search(r"\b(?:chlorophyll|chla|eutrophication|eutroph)\b|叶绿素|富营养", lowered) and SkillPlanner._environment_health_auxiliary_chlorophyll_requested(lowered):
            requested.add("eutrophication_context")
        if re.search(r"\bbloom\b|藻华", lowered) and re.search(r"\b(?:burden|exposure|spatial field|spatial map|summary map)\b|负担|暴露|空间|分布", lowered):
            requested.add("bloom_burden")
        if re.search(
            r"\bbloom[-\s]?(?:event[-\s]?)?days?\b|"
            r"\bbloom\b.{0,50}\b(?:days?|affected area|affected areas|area|areas)\b|"
            r"藻华.{0,20}(天数|影响面积|面积|区域)",
            lowered,
        ):
            requested.add("bloom_event_days")
        if re.search(r"\b(?:heatwave|marine heatwave|heat wave)\b|热浪|海洋热浪", lowered) and re.search(r"\b(?:burden|exposure)\b|负担|暴露", lowered):
            requested.add("heatwave_burden")
        if re.search(r"\b(?:heatwave|marine heatwave|heat wave)\b|热浪|海洋热浪", lowered) and re.search(r"\bdays?\b|天数", lowered):
            requested.add("heatwave_days")
        if re.search(r"\bupwelling\b|上升流", lowered) and re.search(r"\bdays?\b|天数", lowered):
            requested.add("upwelling_days")
        if oxygen_intent:
            requested.add("bottom_oxygen_trend")
        if re.search(r"\bstratification\b|层化|分层|稳定度", lowered):
            requested.add("stratification_strength_change")
        endpoint_evidence_keys = {
            "bottom_oxygen_trend",
            "bottom_hypoxia_burden",
            "hypoxia_statistics",
            "hypoxic_days",
        }
        if suitability_intent and not (endpoint_evidence_keys & requested):
            requested.update({"bottom_oxygen_trend", "bottom_hypoxia_burden"})
            if spatial_risk_intent or trend_intent:
                requested.add("hypoxia_statistics")
        if suitability_intent:
            requested.update({"sst_trend", "stratification_strength_change"})
            if spatial_risk_intent:
                requested.update({
                    "hypoxic_days",
                    "heatwave_days",
                    "heatwave_burden",
                })
                if "bloom_frequency_change" not in suppressed:
                    requested.update({
                        "bloom_event_days",
                        "bloom_burden",
                        "eutrophication_context",
                    })
        if not requested and re.search(r"\b(?:environmental risk|marine risk|risk assessment|high[-\s]?risk areas?)\b|环境风险|高风险区域|高风险区", lowered):
            requested.update({
                "bottom_oxygen_trend",
                "bottom_hypoxia_burden",
                "hypoxia_statistics",
                "hypoxic_days",
                "sst_trend",
                "heatwave_days",
                "heatwave_burden",
                "stratification_strength_change",
            })
            if "bloom_frequency_change" not in suppressed:
                requested.update({
                    "bloom_event_days",
                    "bloom_burden",
                    "eutrophication_context",
                })
        if not requested and re.search(r"\b(?:environment(?:al)? health|marine environment(?:al)? health|overall assessment)\b|环境健康|海洋健康|综合评估|总体评估", lowered):
            requested.update({
                "sst_trend",
                "bloom_frequency_change",
                "bottom_oxygen_trend",
                "stratification_strength_change",
            })
        requested.difference_update(suppressed)
        return requested

    @staticmethod
    def _environment_health_suppressed_branch_keys(user_request: str) -> set[str]:
        lowered = (user_request or "").lower()
        suppressed: set[str] = set()
        bloom_negated = re.search(
            r"\b(?:do not|don't|without|exclude|avoid|not)\b.{0,60}\b(?:bloom|bloom[-\s]?event|bloom[-\s]?frequency)\b",
            lowered,
        ) or re.search(
            r"\b(?:bloom|bloom[-\s]?event|bloom[-\s]?frequency)\b.{0,60}\b(?:not|excluded|auxiliary|context only|not primary)\b",
            lowered,
        ) or re.search(
            r"(不要|不使用|排除|避免).{0,30}(藻华|赤潮|叶绿素)|"
            r"(藻华|赤潮|叶绿素).{0,30}(辅助|背景|不是主要|不作为主要)",
            lowered,
        )
        if bloom_negated:
            suppressed.update({"bloom_frequency_change", "bloom_burden", "bloom_event_days"})
        return suppressed

    @staticmethod
    def _environment_health_auxiliary_chlorophyll_requested(lowered_request: str) -> bool:
        return bool(
            re.search(r"\b(?:chlorophyll|chla|eutrophication|eutroph)\b|叶绿素|富营养", lowered_request)
            and re.search(r"\b(?:auxiliary|context|screening|only as|not primary|supporting)\b|辅助|背景|筛查|仅作为|不是主要|支持", lowered_request)
        )

    @staticmethod
    def _environment_health_branch_key(branch: Dict[str, Any]) -> Optional[str]:
        text = " ".join(
            str(branch.get(key) or "")
            for key in ("name", "indicator_label", "indicator")
        ).strip().lower()
        if not text:
            return None
        normalized = text.replace("_", " ").replace("-", " ")
        if "hypoxic days" in normalized:
            return "hypoxic_days"
        if "hypoxia" in normalized and "days" in normalized:
            return "hypoxic_days"
        if "hypoxia" in normalized and any(token in normalized for token in ("statistics", "frequency", "count", "event")):
            return "hypoxia_statistics"
        if "hypoxia" in normalized and ("burden" in normalized or "deficit" in normalized):
            return "bottom_hypoxia_burden"
        if "bloom" in normalized and "burden" in normalized:
            return "bloom_burden"
        if "bloom" in normalized and "days" in normalized:
            return "bloom_event_days"
        if "eutroph" in normalized or ("chlorophyll" in normalized and "context" in normalized):
            return "eutrophication_context"
        if "heatwave" in normalized and "burden" in normalized:
            return "heatwave_burden"
        if "heatwave" in normalized and "days" in normalized:
            return "heatwave_days"
        if "upwelling" in normalized and "days" in normalized:
            return "upwelling_days"
        if "stratification" in normalized:
            return "stratification_strength_change"
        if "bottom" in normalized and "oxygen" in normalized:
            return "bottom_oxygen_trend"
        if "bloom" in normalized or "chlorophyll" in normalized or "chla" in normalized:
            return "bloom_frequency_change"
        if "sst" in normalized or "sea surface temperature" in normalized or "temperature" in normalized:
            return "sst_trend"
        return None

    @staticmethod
    def _is_configured_environment_evidence_branch(branch: Dict[str, Any]) -> bool:
        evidence_kind = str(branch.get("evidence_kind") or "").strip().lower()
        result_ref = branch.get("result")
        if not evidence_kind or not isinstance(result_ref, str) or not result_ref.strip():
            return False
        if evidence_kind in {"spatial_field", "event_spatial_field"}:
            metric = str(branch.get("metric") or "").strip().lower()
            worse_when = str(branch.get("worse_when") or "").strip().lower()
            return metric in {"mean", "max", "total", "positive_fraction", "nonzero_fraction"} and worse_when in {"increase", "decrease", "presence"}
        return evidence_kind in {"trend", "event_statistics", "event_detection", "event_spatial_distribution"}

    @staticmethod
    def _repair_stratification_depth_plan(
        plan: Dict[str, Any],
        *,
        user_request: str,
    ) -> Dict[str, Any]:
        """Keep stratification temp/salt loads multi-level and bottom semantics response-only."""
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan

        repaired_steps = [dict(step) for step in steps]
        bottom_oxygen_requested = SkillPlanner._request_mentions_bottom_oxygen(user_request)
        oxygen_load_refs: set[str] = set()

        for step in repaired_steps:
            if step.get("tool") != "load_dataset":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            variable = SkillPlanner._canonical_variable_name(params.get("variable"))
            updated_params = dict(params)

            save_as = str(step.get("save_as") or "").strip().lower()
            if (save_as, variable) in {("temp_field", "temp"), ("salt_field", "salt")}:
                SkillPlanner._remove_single_level_vertical_selection(updated_params)
                step["params"] = updated_params
                continue

            if variable == "oxygen" and bottom_oxygen_requested:
                updated_params["vertical_mode"] = "bottom"
                updated_params.pop("depth_value", None)
                updated_params.pop("depth_range", None)
                step["params"] = updated_params
                save_as = step.get("save_as")
                if isinstance(save_as, str):
                    oxygen_load_refs.add(save_as)

        for step in repaired_steps:
            tool_name = step.get("tool")
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            updated_params = dict(params)

            if tool_name in {"compute_stratification_index", "compute_vertical_stability_timeseries"}:
                depth_range = SkillPlanner._coerce_depth_range_static(updated_params.get("depth_range"))
                if depth_range is not None and SkillPlanner._is_single_depth_range(depth_range):
                    updated_params.pop("depth_range", None)
                    step["params"] = updated_params
                continue

            if (
                bottom_oxygen_requested
                and tool_name == "compute_area_weighted_mean"
                and SkillPlanner._ref_targets_any(updated_params.get("data"), oxygen_load_refs)
            ):
                updated_params.pop("depth_range", None)
                step["params"] = updated_params

        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    @staticmethod
    def _repair_stratification_lead_lag_plan(
        plan: Dict[str, Any],
        *,
        user_request: str,
    ) -> Dict[str, Any]:
        """Ensure stratification-response lead/lag requests include lag correlation."""
        if not SkillPlanner._request_mentions_lead_lag(user_request):
            return plan

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan
        if any(step.get("tool") == "compute_lag_correlation" for step in steps):
            return SkillPlanner._move_stratification_lag_steps_after_response_index(plan)

        stability_index = SkillPlanner._find_step_index_for_result(
            steps,
            preferred_save_as="stability_timeseries",
            preferred_tool="compute_vertical_stability_timeseries",
        )
        response_index = SkillPlanner._find_step_index_for_result(
            steps,
            preferred_save_as="response_timeseries",
            preferred_tool="compute_area_weighted_mean",
        )
        if stability_index is None or response_index is None:
            return plan

        stability_save_as = steps[stability_index].get("save_as")
        response_save_as = steps[response_index].get("save_as")
        if not isinstance(stability_save_as, str) or not isinstance(response_save_as, str):
            return plan

        existing_step_ids = {str(step.get("step_id")) for step in steps if step.get("step_id")}
        existing_save_as = {str(step.get("save_as")) for step in steps if step.get("save_as")}
        step_id = SkillPlanner._unique_name("compute_stratification_response_lag", existing_step_ids)
        save_as = SkillPlanner._unique_name("stratification_response_lag", existing_save_as)
        lag_step = {
            "step_id": step_id,
            "tool": "compute_lag_correlation",
            "params": {
                "timeseries1": f"$ref:{stability_save_as}",
                "timeseries2": f"$ref:{response_save_as}",
                "max_lag": 12,
                "confidence_level": 0.95,
            },
            "save_as": save_as,
        }

        response_mechanism_indices = [
            index
            for index, step in enumerate(steps)
            if step.get("tool") == "compute_stratification_response_index"
        ]
        insert_after = max([stability_index, response_index] + response_mechanism_indices) + 1
        repaired_steps = [dict(step) for step in steps]
        repaired_steps.insert(insert_after, lag_step)
        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return SkillPlanner._move_stratification_lag_steps_after_response_index(repaired_plan)

    @staticmethod
    def _move_stratification_lag_steps_after_response_index(plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan

        lag_steps = [dict(step) for step in steps if step.get("tool") == "compute_lag_correlation"]
        if not lag_steps:
            return plan

        non_lag_steps = [dict(step) for step in steps if step.get("tool") != "compute_lag_correlation"]
        response_indices = [
            index
            for index, step in enumerate(non_lag_steps)
            if step.get("tool") == "compute_stratification_response_index"
        ]
        if not response_indices:
            return plan

        insert_at = max(response_indices) + 1
        repaired_steps = non_lag_steps[:insert_at] + lag_steps + non_lag_steps[insert_at:]
        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    @staticmethod
    def _repair_step_dependency_order(plan: Dict[str, Any]) -> Dict[str, Any]:
        """Keep any step that consumes a $ref after the step that produces it."""
        steps = plan.get("steps")
        if not isinstance(steps, list) or len(steps) < 2:
            return plan

        producer_by_result: Dict[str, int] = {}
        for index, step in enumerate(steps):
            save_as = step.get("save_as")
            if isinstance(save_as, str) and save_as:
                producer_by_result[save_as] = index

        if not producer_by_result:
            return plan

        dependencies: Dict[int, set[int]] = {}
        for index, step in enumerate(steps):
            refs = SkillPlanner._collect_ref_roots_static(step.get("params"))
            dependencies[index] = {
                producer_by_result[ref]
                for ref in refs
                if ref in producer_by_result and producer_by_result[ref] != index
            }

        ordered_indices: List[int] = []
        ordered_set: set[int] = set()
        remaining = set(range(len(steps)))
        while remaining:
            ready = [
                index
                for index in sorted(remaining)
                if dependencies.get(index, set()).issubset(ordered_set)
            ]
            if not ready:
                return plan
            next_index = ready[0]
            remaining.remove(next_index)
            ordered_indices.append(next_index)
            ordered_set.add(next_index)

        if ordered_indices == list(range(len(steps))):
            return plan

        repaired_plan = dict(plan)
        repaired_plan["steps"] = [dict(steps[index]) for index in ordered_indices]
        return repaired_plan

    @staticmethod
    def _find_step_index_for_result(
        steps: List[Dict[str, Any]],
        *,
        preferred_save_as: str,
        preferred_tool: str,
    ) -> Optional[int]:
        for index, step in enumerate(steps):
            if step.get("save_as") == preferred_save_as:
                return index
        for index, step in enumerate(steps):
            if step.get("tool") == preferred_tool:
                return index
        return None

    @staticmethod
    def _unique_name(base: str, existing: set[str]) -> str:
        if base not in existing:
            return base
        suffix = 2
        while f"{base}_{suffix}" in existing:
            suffix += 1
        return f"{base}_{suffix}"

    @staticmethod
    def _remove_single_level_vertical_selection(params: Dict[str, Any]) -> None:
        vertical_mode = params.get("vertical_mode")
        if isinstance(vertical_mode, str):
            normalized_mode = vertical_mode.strip().lower()
            if normalized_mode in {"bottom", "surface", "fixed_depth"}:
                params.pop("vertical_mode", None)
            elif normalized_mode == "depth_range":
                depth_range = SkillPlanner._coerce_depth_range_static(params.get("depth_range"))
                if depth_range is None or SkillPlanner._is_single_depth_range(depth_range):
                    params.pop("vertical_mode", None)

        params.pop("depth_value", None)
        depth_range = SkillPlanner._coerce_depth_range_static(params.get("depth_range"))
        if depth_range is not None and SkillPlanner._is_single_depth_range(depth_range):
            params.pop("depth_range", None)

    @staticmethod
    def _repair_lag_seasonality_plan(
        plan: Dict[str, Any],
        *,
        user_request: str,
    ) -> Dict[str, Any]:
        """Use raw lag correlation unless deseasoning was explicitly requested."""
        if SkillPlanner._request_explicitly_requests_deseasoning(user_request):
            return plan

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan

        removed_ref_map: Dict[str, str] = {}
        for step in steps:
            if step.get("tool") != "remove_seasonal_cycle":
                continue
            params = step.get("params")
            save_as = step.get("save_as")
            if not isinstance(params, dict) or not isinstance(save_as, str):
                continue
            source_ref = params.get("timeseries")
            source_root = SkillPlanner._extract_ref_root_static(source_ref)
            if source_root:
                removed_ref_map[save_as] = source_root

        if not removed_ref_map:
            return plan

        raw_lag_exists = any(
            step.get("tool") == "compute_lag_correlation"
            and not SkillPlanner._value_references_any_result(step.get("params"), set(removed_ref_map))
            for step in steps
        )

        repaired_steps: List[Dict[str, Any]] = []
        for step in steps:
            tool_name = step.get("tool")
            if tool_name == "remove_seasonal_cycle":
                continue
            if (
                raw_lag_exists
                and tool_name == "compute_lag_correlation"
                and SkillPlanner._value_references_any_result(step.get("params"), set(removed_ref_map))
            ):
                continue
            repaired_steps.append(SkillPlanner._rewrite_refs_in_step_static(step, removed_ref_map))

        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    @staticmethod
    def _canonical_variable_name(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        aliases = {
            "temperature": "temp",
            "thetao": "temp",
            "salinity": "salt",
            "so": "salt",
            "o2": "oxygen",
            "dissolved_oxygen": "oxygen",
            "oxygen_concentration": "oxygen",
        }
        return aliases.get(normalized, normalized)

    @staticmethod
    def _request_mentions_bottom_oxygen(user_request: str) -> bool:
        lowered = (user_request or "").lower()
        return bool(
            re.search(
                r"\b(?:bottom|near[-\s]?bottom|bottom[-\s]?layer)[-\s]+(?:dissolved[-\s]+)?oxygen\b",
                lowered,
            )
            or re.search(r"\boxygen\b.{0,24}\b(?:bottom|near[-\s]?bottom|bottom[-\s]?layer)\b", lowered)
        )

    @staticmethod
    def _request_mentions_lead_lag(user_request: str) -> bool:
        lowered = (user_request or "").lower()
        return bool(
            re.search(r"\blead[-\s]?lag\b", lowered)
            or re.search(r"\b(?:lead|leads|leading|lag|lags|lagging)\b", lowered)
        )

    @staticmethod
    def _request_explicitly_requests_deseasoning(user_request: str) -> bool:
        lowered = (user_request or "").lower()
        patterns = (
            r"remove\s+(?:the\s+)?seasonal\s+cycle",
            r"removing\s+(?:the\s+)?seasonal\s+cycle",
            r"after\s+removing\s+(?:the\s+)?seasonal",
            r"\bdeseason(?:ed|alize|alized|ing)?\b",
            r"\bde-season(?:ed|alize|alized|ing)?\b",
            r"\banomaly[-\s]?scale\b",
            r"\banomalous\s+coupling\b",
            r"\bbeyond\s+seasonality\b",
            r"\bseasonal-cycle-removed\b",
        )
        return any(re.search(pattern, lowered) for pattern in patterns)

    @staticmethod
    def _coerce_depth_range_static(value: Any) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            return [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_single_depth_range(depth_range: List[float]) -> bool:
        return abs(float(depth_range[0]) - float(depth_range[1])) <= 1e-6

    @staticmethod
    def _ref_targets_any(value: Any, targets: set[str]) -> bool:
        ref_root = SkillPlanner._extract_ref_root_static(value)
        return ref_root in targets if ref_root is not None else False

    @staticmethod
    def _extract_ref_root_static(value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.startswith("$ref:"):
            return None
        return value[5:].split(".", 1)[0]

    @staticmethod
    def _value_references_any_result(value: Any, result_ids: set[str]) -> bool:
        if isinstance(value, str):
            ref_root = SkillPlanner._extract_ref_root_static(value)
            return ref_root in result_ids if ref_root is not None else False
        if isinstance(value, dict):
            return any(SkillPlanner._value_references_any_result(item, result_ids) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(SkillPlanner._value_references_any_result(item, result_ids) for item in value)
        return False

    @staticmethod
    def _collect_ref_roots_static(value: Any) -> set[str]:
        if isinstance(value, str):
            ref_root = SkillPlanner._extract_ref_root_static(value)
            return {ref_root} if ref_root else set()
        if isinstance(value, dict):
            refs: set[str] = set()
            for item in value.values():
                refs.update(SkillPlanner._collect_ref_roots_static(item))
            return refs
        if isinstance(value, (list, tuple)):
            refs: set[str] = set()
            for item in value:
                refs.update(SkillPlanner._collect_ref_roots_static(item))
            return refs
        return set()

    @staticmethod
    def _rewrite_refs_in_step_static(step: Dict[str, Any], rename_map: Dict[str, str]) -> Dict[str, Any]:
        rewritten = dict(step)
        rewritten["params"] = SkillPlanner._rewrite_refs_in_value_static(step.get("params"), rename_map)
        return rewritten

    @staticmethod
    def _rewrite_refs_in_value_static(value: Any, rename_map: Dict[str, str]) -> Any:
        if isinstance(value, dict):
            return {key: SkillPlanner._rewrite_refs_in_value_static(item, rename_map) for key, item in value.items()}
        if isinstance(value, list):
            return [SkillPlanner._rewrite_refs_in_value_static(item, rename_map) for item in value]
        if isinstance(value, tuple):
            return tuple(SkillPlanner._rewrite_refs_in_value_static(item, rename_map) for item in value)
        if not isinstance(value, str) or not value.startswith("$ref:"):
            return value
        ref_expr = value[5:]
        ref_root, separator, ref_path = ref_expr.partition(".")
        if ref_root not in rename_map:
            return value
        rewritten_root = rename_map[ref_root]
        rewritten_expr = rewritten_root if not separator else f"{rewritten_root}.{ref_path}"
        return f"$ref:{rewritten_expr}"

    def _skill_is_event_detection_or_analysis(self, skill_id: str) -> bool:
        return skill_id.startswith("ocean_") and (
            skill_id.endswith("_detection")
            or skill_id in {
                "ocean_event_frequency_map",
                "ocean_event_count_timeseries",
                "ocean_event_statistics",
                "ocean_event_comparison",
            }
        )

    def _repair_event_detection_plan(
        self,
        plan: Dict[str, Any],
        *,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan

        effective_vertical = self._resolve_event_vertical_selection(
            steps,
            extracted_params=extracted_params,
            additional_context=additional_context,
        )
        if effective_vertical is None:
            return plan

        repaired_steps = [dict(step) for step in steps]
        save_as_to_step = {
            step.get("save_as"): step
            for step in repaired_steps
            if isinstance(step.get("save_as"), str)
        }

        for step in repaired_steps:
            tool_name = step.get("tool")
            if tool_name not in {
                "detect_heatwaves",
                "detect_hypoxia",
                "detect_algal_blooms",
                "detect_upwelling",
                "detect_eutrophication",
            }:
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue

            updated_params = dict(params)
            self._apply_event_vertical_params(updated_params, effective_vertical)
            step["params"] = updated_params

            for referenced_save_as in self._extract_data_refs_from_params(updated_params):
                upstream_step = save_as_to_step.get(referenced_save_as)
                if not upstream_step or upstream_step.get("tool") != "load_dataset":
                    continue
                upstream_params = upstream_step.get("params")
                if not isinstance(upstream_params, dict):
                    continue
                patched_upstream_params = dict(upstream_params)
                self._apply_event_load_params(patched_upstream_params, effective_vertical)
                upstream_step["params"] = patched_upstream_params

        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    def _repair_anomaly_plan(
        self,
        plan: Dict[str, Any],
        *,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan

        repaired_steps = [dict(step) for step in steps]
        existing_step_ids = {
            str(step.get("step_id"))
            for step in repaired_steps
            if isinstance(step.get("step_id"), str)
        }
        existing_save_as = {
            str(step.get("save_as"))
            for step in repaired_steps
            if isinstance(step.get("save_as"), str)
        }
        dataset_temporal_range = self._dataset_temporal_range(additional_context)

        index = 0
        changed = False
        while index < len(repaired_steps):
            step = repaired_steps[index]
            if step.get("tool") != "compute_field_anomaly":
                index += 1
                continue

            params = step.get("params")
            if not isinstance(params, dict) or isinstance(params.get("climatology"), str):
                index += 1
                continue

            data_ref = params.get("data")
            if not isinstance(data_ref, str) or not data_ref.startswith("$ref:") or not data_ref.endswith(".data"):
                index += 1
                continue

            source_save_as = data_ref[5:-5]
            save_as_to_step = {
                candidate.get("save_as"): candidate
                for candidate in repaired_steps
                if isinstance(candidate.get("save_as"), str)
            }
            source_step = save_as_to_step.get(source_save_as)
            if not isinstance(source_step, dict):
                index += 1
                continue

            target_time_range = self._anomaly_source_time_range(
                source_step=source_step,
                save_as_to_step=save_as_to_step,
            )
            climatology_time_range = self._resolve_anomaly_climatology_time_range(
                extracted_params=extracted_params,
                dataset_temporal_range=dataset_temporal_range,
                target_time_range=target_time_range,
            )
            if climatology_time_range is None:
                index += 1
                continue

            extra_steps, climatology_ref = self._build_anomaly_climatology_steps(
                source_step=source_step,
                save_as_to_step=save_as_to_step,
                climatology_time_range=climatology_time_range,
                period=params.get("period"),
                existing_step_ids=existing_step_ids,
                existing_save_as=existing_save_as,
            )
            if not extra_steps or not climatology_ref:
                index += 1
                continue

            repaired_steps[index:index] = extra_steps
            index += len(extra_steps)

            updated_params = dict(params)
            updated_params["climatology"] = climatology_ref
            step = dict(repaired_steps[index])
            step["params"] = updated_params
            repaired_steps[index] = step
            changed = True
            index += 1

        if not changed:
            return plan

        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    def _dataset_temporal_range(self, additional_context: Dict[str, Any]) -> Optional[List[str]]:
        dataset = additional_context.get("dataset")
        if not isinstance(dataset, dict):
            return None
        temporal_extent = dataset.get("temporal_extent")
        if not isinstance(temporal_extent, dict):
            return None
        return self._coerce_time_range_strings(
            [temporal_extent.get("start"), temporal_extent.get("end")]
        )

    def _anomaly_source_time_range(
        self,
        *,
        source_step: Dict[str, Any],
        save_as_to_step: Dict[str, Dict[str, Any]],
    ) -> Optional[List[str]]:
        tool_name = source_step.get("tool")
        params = source_step.get("params")
        if not isinstance(params, dict):
            return None
        if tool_name == "load_dataset":
            return self._coerce_time_range_strings(params.get("time_range"))
        if tool_name == "apply_mask":
            data_ref = params.get("data")
            if not isinstance(data_ref, str) or not data_ref.startswith("$ref:") or not data_ref.endswith(".data"):
                return None
            upstream_step = save_as_to_step.get(data_ref[5:-5])
            if not isinstance(upstream_step, dict):
                return None
            upstream_params = upstream_step.get("params")
            if not isinstance(upstream_params, dict):
                return None
            return self._coerce_time_range_strings(upstream_params.get("time_range"))
        return None

    def _resolve_anomaly_climatology_time_range(
        self,
        *,
        extracted_params: Dict[str, Any],
        dataset_temporal_range: Optional[List[str]],
        target_time_range: Optional[List[str]],
    ) -> Optional[List[str]]:
        requested = self._coerce_time_range_strings(extracted_params.get("climatology_time_range"))
        if requested is not None:
            return requested
        if dataset_temporal_range is None:
            return None
        if target_time_range is None or dataset_temporal_range != target_time_range:
            return dataset_temporal_range
        return None

    def _build_anomaly_climatology_steps(
        self,
        *,
        source_step: Dict[str, Any],
        save_as_to_step: Dict[str, Dict[str, Any]],
        climatology_time_range: List[str],
        period: Any,
        existing_step_ids: set[str],
        existing_save_as: set[str],
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        source_tool = source_step.get("tool")
        source_params = source_step.get("params")
        if not isinstance(source_params, dict):
            return [], None

        base_load_step = source_step
        mask_ref = None
        if source_tool == "apply_mask":
            data_ref = source_params.get("data")
            mask_ref = source_params.get("mask")
            if not isinstance(data_ref, str) or not data_ref.startswith("$ref:") or not data_ref.endswith(".data"):
                return [], None
            base_load_step = save_as_to_step.get(data_ref[5:-5], {})

        if base_load_step.get("tool") != "load_dataset":
            return [], None

        base_load_params = base_load_step.get("params")
        if not isinstance(base_load_params, dict):
            return [], None

        load_save_as = self._unique_identifier("climatology_data", existing_save_as)
        load_step_id = self._unique_identifier("load_climatology_data", existing_step_ids)
        climatology_load_params = dict(base_load_params)
        climatology_load_params["time_range"] = list(climatology_time_range)
        extra_steps: List[Dict[str, Any]] = [
            {
                "step_id": load_step_id,
                "tool": "load_dataset",
                "params": climatology_load_params,
                "save_as": load_save_as,
            }
        ]

        climatology_source_ref = f"$ref:{load_save_as}.data"
        if isinstance(mask_ref, str) and mask_ref.startswith("$ref:") and mask_ref.endswith(".data"):
            masked_save_as = self._unique_identifier("masked_climatology_data", existing_save_as)
            masked_step_id = self._unique_identifier("mask_climatology_data", existing_step_ids)
            extra_steps.append(
                {
                    "step_id": masked_step_id,
                    "tool": "apply_mask",
                    "params": {
                        "data": climatology_source_ref,
                        "mask": mask_ref,
                    },
                    "save_as": masked_save_as,
                }
            )
            climatology_source_ref = f"$ref:{masked_save_as}.data"

        climatology_save_as = self._unique_identifier("field_climatology", existing_save_as)
        climatology_step_id = self._unique_identifier("compute_field_climatology", existing_step_ids)
        extra_steps.append(
            {
                "step_id": climatology_step_id,
                "tool": "compute_field_climatology",
                "params": {
                    "data": climatology_source_ref,
                    "period": period if isinstance(period, str) and period.strip() else "monthly",
                },
                "save_as": climatology_save_as,
            }
        )
        return extra_steps, f"$ref:{climatology_save_as}.data"

    def _resolve_event_vertical_selection(
        self,
        steps: List[Dict[str, Any]],
        *,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        for step in steps:
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            selection = self._event_vertical_selection_from_mapping(params)
            if selection is not None:
                return selection

        selection = self._event_vertical_selection_from_mapping(extracted_params)
        if selection is not None:
            return selection

        workspace_context = additional_context.get("workspace_context", {})
        if not isinstance(workspace_context, dict):
            return None

        depth_mode = workspace_context.get("depth_mode")
        current_depth = workspace_context.get("current_depth")
        current_depth_range = workspace_context.get("current_depth_range")
        workspace_depth_range = workspace_context.get("depth_range")

        if depth_mode == "fixed":
            if current_depth is not None:
                depth_value = self._coerce_float(current_depth)
                if depth_value is not None:
                    return {
                        "vertical_mode": "fixed_depth",
                        "depth_value": depth_value,
                        "depth_range": [depth_value, depth_value],
                    }
            range_value = self._coerce_depth_range(current_depth_range) or self._coerce_depth_range(workspace_depth_range)
            if range_value is not None:
                if abs(range_value[0] - range_value[1]) <= 1e-6:
                    return {
                        "vertical_mode": "fixed_depth",
                        "depth_value": range_value[0],
                        "depth_range": range_value,
                    }
                return {
                    "vertical_mode": "depth_range",
                    "depth_range": range_value,
                }

        if depth_mode == "layer_mean":
            range_value = self._coerce_depth_range(current_depth_range) or self._coerce_depth_range(workspace_depth_range)
            if range_value is not None:
                return {
                    "vertical_mode": "depth_range",
                    "depth_range": range_value,
                    "depth_aggregation": "mean",
                }

        return None

    def _event_vertical_selection_from_mapping(self, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        vertical_mode = values.get("vertical_mode")
        depth_value = self._coerce_float(values.get("depth_value"))
        depth_range = self._coerce_depth_range(values.get("depth_range"))
        depth_aggregation = values.get("depth_aggregation")

        if isinstance(vertical_mode, str):
            normalized_mode = vertical_mode.strip().lower()
            if normalized_mode == "surface":
                return {"vertical_mode": "surface"}
            if normalized_mode == "bottom":
                return {"vertical_mode": "bottom"}
            if normalized_mode == "fixed_depth":
                if depth_value is None and depth_range is not None and abs(depth_range[0] - depth_range[1]) <= 1e-6:
                    depth_value = depth_range[0]
                if depth_value is not None:
                    return {
                        "vertical_mode": "fixed_depth",
                        "depth_value": depth_value,
                        "depth_range": [depth_value, depth_value],
                    }
                return None
            if normalized_mode == "depth_range" and depth_range is not None:
                selection: Dict[str, Any] = {
                    "vertical_mode": "depth_range",
                    "depth_range": depth_range,
                }
                if isinstance(depth_aggregation, str) and depth_aggregation.strip():
                    selection["depth_aggregation"] = depth_aggregation.strip().lower()
                return selection

        if depth_range is not None:
            if abs(depth_range[0] - depth_range[1]) <= 1e-6:
                return {
                    "vertical_mode": "fixed_depth",
                    "depth_value": depth_range[0],
                    "depth_range": depth_range,
                }
            selection = {
                "vertical_mode": "depth_range",
                "depth_range": depth_range,
            }
            if isinstance(depth_aggregation, str) and depth_aggregation.strip():
                selection["depth_aggregation"] = depth_aggregation.strip().lower()
            return selection

        if depth_value is not None:
            return {
                "vertical_mode": "fixed_depth",
                "depth_value": depth_value,
                "depth_range": [depth_value, depth_value],
            }

        return None

    def _apply_event_vertical_params(self, params: Dict[str, Any], selection: Dict[str, Any]) -> None:
        vertical_mode = selection.get("vertical_mode")
        if vertical_mode is None:
            return
        params["vertical_mode"] = vertical_mode
        if vertical_mode in {"bottom", "surface"}:
            params.pop("depth_value", None)
            params.pop("depth_range", None)
            params.pop("depth_aggregation", None)
            return
        if "depth_value" in selection:
            params["depth_value"] = selection["depth_value"]
        if "depth_range" in selection:
            params["depth_range"] = selection["depth_range"]
        if "depth_aggregation" in selection:
            params["depth_aggregation"] = selection["depth_aggregation"]

    def _apply_event_load_params(self, params: Dict[str, Any], selection: Dict[str, Any]) -> None:
        vertical_mode = selection.get("vertical_mode")
        if vertical_mode in {"bottom", "surface"}:
            params["vertical_mode"] = vertical_mode
            params.pop("depth_range", None)
            params.pop("depth_value", None)
            return

        params.pop("vertical_mode", None)
        params.pop("depth_value", None)
        load_depth_range = self._event_load_depth_range(selection)
        if load_depth_range is not None:
            params["depth_range"] = load_depth_range

    def _event_load_depth_range(self, selection: Dict[str, Any]) -> Optional[List[float]]:
        vertical_mode = selection.get("vertical_mode")
        if vertical_mode == "surface":
            return None
        if vertical_mode == "fixed_depth":
            depth_value = selection.get("depth_value")
            if isinstance(depth_value, (int, float)):
                depth_float = float(depth_value)
                return [depth_float, depth_float]
        if vertical_mode == "depth_range":
            depth_range = selection.get("depth_range")
            if isinstance(depth_range, list) and len(depth_range) == 2:
                return [float(depth_range[0]), float(depth_range[1])]
        return None

    def _extract_data_refs_from_params(self, params: Dict[str, Any]) -> List[str]:
        refs: List[str] = []
        for value in params.values():
            if not isinstance(value, str) or not value.startswith("$ref:") or not value.endswith(".data"):
                continue
            refs.append(value[5:-5])
        return refs

    def _coerce_float(self, value: Any) -> Optional[float]:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    def _coerce_time_range_strings(self, value: Any) -> Optional[List[str]]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        start, end = value
        if not isinstance(start, str) or not start.strip():
            return None
        if not isinstance(end, str) or not end.strip():
            return None
        return [start.strip(), end.strip()]

    def _unique_identifier(self, base: str, existing: set[str]) -> str:
        candidate = base
        suffix = 2
        while candidate in existing:
            candidate = f"{base}_{suffix}"
            suffix += 1
        existing.add(candidate)
        return candidate

    def _coerce_depth_range(self, value: Any) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        lower = self._coerce_float(value[0])
        upper = self._coerce_float(value[1])
        if lower is None or upper is None:
            return None
        return [lower, upper]

    def _repair_velocity_derived_plan(self, plan: Dict[str, Any], user_request: str) -> Dict[str, Any]:
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return plan

        field_type = self._infer_velocity_derived_field_type(steps, user_request=user_request)
        if field_type not in {"vorticity", "speed"}:
            return plan

        load_refs = self._find_velocity_load_refs(steps)
        u_ref = load_refs.get("u")
        v_ref = load_refs.get("v")
        if not u_ref or not v_ref:
            return plan

        repaired_steps = list(steps)
        if any(step.get("tool") == "compute_spatial_vorticity_map" for step in repaired_steps):
            repaired_plan = dict(plan)
            repaired_plan["steps"] = self._remove_unreferenced_uv_assembly_steps(
                repaired_steps,
                u_ref=u_ref,
                v_ref=v_ref,
            )
            return repaired_plan

        if field_type == "vorticity":
            fast_steps = self._rewrite_spatial_vorticity_plan(
                repaired_steps,
                u_ref=u_ref,
                v_ref=v_ref,
            )
            if fast_steps is not None:
                repaired_plan = dict(plan)
                repaired_plan["steps"] = fast_steps
                return repaired_plan

        derived_index = self._find_derived_field_step_index(repaired_steps, field_type=field_type)
        if derived_index is None:
            derived_index = max(
                index
                for index, step in enumerate(repaired_steps)
                if step.get("save_as") in {u_ref, v_ref}
            ) + 1
            repaired_steps.insert(
                derived_index,
                self._build_derived_field_step(
                    step_id=f"compute_{field_type}",
                    save_as="derived_field",
                    u_ref=u_ref,
                    v_ref=v_ref,
                    field_type=field_type,
                ),
            )
        else:
            existing_step = repaired_steps[derived_index]
            repaired_steps[derived_index] = self._build_derived_field_step(
                step_id=existing_step.get("step_id", f"compute_{field_type}"),
                save_as=existing_step.get("save_as", "derived_field"),
                u_ref=u_ref,
                v_ref=v_ref,
                field_type=field_type,
                description=existing_step.get("description", ""),
            )

        repaired_steps = self._remove_unreferenced_uv_assembly_steps(repaired_steps, u_ref=u_ref, v_ref=v_ref)
        repaired_plan = dict(plan)
        repaired_plan["steps"] = repaired_steps
        return repaired_plan

    def _rewrite_spatial_vorticity_plan(
        self,
        steps: List[Dict[str, Any]],
        *,
        u_ref: str,
        v_ref: str,
    ) -> Optional[List[Dict[str, Any]]]:
        derived_index = self._find_derived_field_step_index(steps, field_type="vorticity")
        if derived_index is None:
            return None

        derived_step = steps[derived_index]
        derived_save_as = derived_step.get("save_as", "derived_field")
        if not isinstance(derived_save_as, str):
            return None

        spatial_index = None
        for index, step in enumerate(steps):
            if step.get("tool") != "compute_spatial_field":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            if self._extract_ref_root(params.get("data")) == derived_save_as:
                spatial_index = index
                break
        if spatial_index is None:
            return None

        for index, step in enumerate(steps):
            if index in {derived_index, spatial_index}:
                continue
            if self._step_references_result(step, derived_save_as):
                return None

        spatial_step = steps[spatial_index]
        spatial_params = spatial_step.get("params") if isinstance(spatial_step.get("params"), dict) else {}
        fast_params: Dict[str, Any] = {
            "u": f"$ref:{u_ref}.data",
            "v": f"$ref:{v_ref}.data",
        }
        for param_name in ("time_range", "time_aggregation", "depth_range", "depth_aggregation", "mask"):
            if param_name in spatial_params:
                fast_params[param_name] = spatial_params[param_name]

        fast_step: Dict[str, Any] = {
            "step_id": spatial_step.get("step_id", "compute_spatial_vorticity_map"),
            "tool": "compute_spatial_vorticity_map",
            "params": fast_params,
            "save_as": spatial_step.get("save_as", "spatial_field"),
        }
        if isinstance(spatial_step.get("description"), str) and spatial_step.get("description"):
            fast_step["description"] = spatial_step["description"]

        rewritten_steps: List[Dict[str, Any]] = []
        for index, step in enumerate(steps):
            if index == derived_index:
                continue
            if index == spatial_index:
                rewritten_steps.append(fast_step)
            else:
                rewritten_steps.append(step)

        return self._remove_unreferenced_uv_assembly_steps(rewritten_steps, u_ref=u_ref, v_ref=v_ref)

    def _infer_velocity_derived_field_type(self, steps: List[Dict[str, Any]], user_request: str) -> Optional[str]:
        normalized_request = user_request.lower()
        if "vorticity" in normalized_request:
            return "vorticity"
        if "speed" in normalized_request or "current speed" in normalized_request:
            return "speed"

        for step in steps:
            params = step.get("params")
            if isinstance(params, dict):
                field_type = params.get("field_type")
                if field_type in {"vorticity", "speed"}:
                    return field_type
                variable = params.get("variable")
                if variable in {"vorticity", "speed"}:
                    return variable
        return None

    def _find_velocity_load_refs(self, steps: List[Dict[str, Any]]) -> Dict[str, str]:
        refs: Dict[str, str] = {}
        for step in steps:
            if step.get("tool") != "load_dataset":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            variable = params.get("variable")
            save_as = step.get("save_as")
            if variable in {"u", "v"} and isinstance(save_as, str):
                refs[variable] = save_as
        return refs

    def _find_or_insert_uv_assembly_step(
        self,
        steps: List[Dict[str, Any]],
        *,
        u_ref: str,
        v_ref: str,
    ) -> tuple[int, str]:
        expected_variables = {
            "u": f"$ref:{u_ref}.data",
            "v": f"$ref:{v_ref}.data",
        }

        for index, step in enumerate(steps):
            if step.get("tool") != "assemble_dataset":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            if params.get("variables") == expected_variables:
                return index, step.get("save_as", "uv_dataset")

        insert_after = max(
            index
            for index, step in enumerate(steps)
            if step.get("save_as") in {u_ref, v_ref}
        )
        assemble_step = {
            "step_id": "assemble_uv",
            "tool": "assemble_dataset",
            "params": {"variables": expected_variables},
            "save_as": "uv_dataset",
        }
        steps.insert(insert_after + 1, assemble_step)
        return insert_after + 1, "uv_dataset"

    def _remove_unreferenced_uv_assembly_steps(
        self,
        steps: List[Dict[str, Any]],
        *,
        u_ref: str,
        v_ref: str,
    ) -> List[Dict[str, Any]]:
        expected_variables = {
            "u": f"$ref:{u_ref}.data",
            "v": f"$ref:{v_ref}.data",
        }
        retained_steps: List[Dict[str, Any]] = []
        for index, step in enumerate(steps):
            if step.get("tool") != "assemble_dataset":
                retained_steps.append(step)
                continue

            params = step.get("params")
            save_as = step.get("save_as")
            if not isinstance(params, dict) or params.get("variables") != expected_variables or not isinstance(save_as, str):
                retained_steps.append(step)
                continue

            if any(
                other_index != index and self._step_references_result(other_step, save_as)
                for other_index, other_step in enumerate(steps)
            ):
                retained_steps.append(step)

        return retained_steps

    def _step_references_result(self, step: Dict[str, Any], result_id: str) -> bool:
        return self._value_references_result(step.get("params"), result_id)

    def _value_references_result(self, value: Any, result_id: str) -> bool:
        if isinstance(value, str):
            return self._extract_ref_root(value) == result_id
        if isinstance(value, dict):
            return any(self._value_references_result(item, result_id) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(self._value_references_result(item, result_id) for item in value)
        return False

    def _extract_ref_root(self, value: Any) -> Optional[str]:
        if not isinstance(value, str) or not value.startswith("$ref:"):
            return None
        return value[5:].split(".", 1)[0]

    def _find_derived_field_step_index(self, steps: List[Dict[str, Any]], field_type: str) -> Optional[int]:
        for index, step in enumerate(steps):
            tool_name = step.get("tool")
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            if tool_name == "compute_derived_field" and params.get("field_type") == field_type:
                return index
            if tool_name == "load_dataset" and params.get("variable") == field_type:
                return index
        return None

    def _build_derived_field_step(
        self,
        *,
        step_id: str,
        save_as: str,
        u_ref: str,
        v_ref: str,
        field_type: str,
        description: str = "",
    ) -> Dict[str, Any]:
        step = {
            "step_id": step_id,
            "tool": "compute_derived_field",
            "params": {
                "u": f"$ref:{u_ref}.data",
                "v": f"$ref:{v_ref}.data",
                "field_type": field_type,
            },
            "save_as": save_as,
        }
        if isinstance(description, str) and description:
            step["description"] = description
        return step

    def _validate_plan_shape(self, plan: Dict[str, Any], expected_skill_id: Optional[str]) -> None:
        if not isinstance(plan, dict):
            raise ValueError("Generated result must be a JSON object.")

        status = plan.get("status", "ready")
        if status not in {"ready", "clarification_needed"}:
            raise ValueError("Generated result status must be 'ready' or 'clarification_needed'.")

        skills_used = plan.get("skills_used")
        if skills_used is not None:
            if not isinstance(skills_used, list) or not skills_used or not all(isinstance(skill, str) for skill in skills_used):
                raise ValueError("Generated result 'skills_used' must be a non-empty list of strings.")

        if status == "clarification_needed":
            missing_fields = plan.get("missing_fields")
            question = plan.get("clarification_question")
            if not isinstance(missing_fields, list) or not missing_fields:
                raise ValueError("Clarification result must contain a non-empty 'missing_fields' list.")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Clarification result must contain a non-empty 'clarification_question'.")
            return

        if "skill_id" not in plan or not isinstance(plan.get("skill_id"), str):
            raise ValueError("Generated result must contain a string 'skill_id'.")
        if expected_skill_id is not None and plan.get("skill_id") != expected_skill_id:
            raise ValueError(
                f"Generated plan skill_id mismatch: expected '{expected_skill_id}', "
                f"got '{plan.get('skill_id')}'."
            )

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("Ready plan must contain a non-empty 'steps' list.")

        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("Each step must be a JSON object.")
            for key in ("step_id", "tool", "params", "save_as"):
                if key not in step:
                    raise ValueError(f"Plan step is missing required key: {key}")
            if not isinstance(step["params"], dict):
                raise ValueError("Plan step 'params' must be an object.")

    def _validate_plan_for_execution(
        self,
        plan: Dict[str, Any],
        *,
        expected_skill_id: Optional[str],
        skill_markdowns: Dict[str, str],
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
        available_result_types: Optional[Dict[str, str]] = None,
    ) -> None:
        self._validate_plan_shape(plan, expected_skill_id=expected_skill_id)
        if plan.get("status", "ready") != "ready":
            return
        available_result_ids = set((available_result_types or {}).keys())
        plan = self._canonicalize_plan_in_place_for_validation(
            plan,
            expected_skill_id=expected_skill_id,
            user_request=user_request,
            available_result_ids=available_result_ids,
        )

        if skill_markdowns:
            self._validate_known_skills(plan, skill_markdowns)
            self._validate_plan_tools(plan, skill_markdowns)

        self._validate_tool_param_contracts(plan)
        self._validate_fixed_reports_explicit_only(plan, user_request=user_request)
        self._validate_plan_references(plan, available_result_types=available_result_types)
        self._validate_named_region_spatial_intent(
            plan,
            user_request=user_request,
            extracted_params=extracted_params,
            additional_context=additional_context,
        )
        self._validate_load_dataset_time_intent(
            plan,
            user_request=user_request,
            extracted_params=extracted_params,
            additional_context=additional_context,
        )
        self._validate_environment_health_execution_semantics(
            plan,
            user_request=user_request,
            additional_context=additional_context,
            available_result_ids=available_result_ids,
        )

    def _validate_tool_param_contracts(self, plan: Dict[str, Any]) -> None:
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return
        for step in steps:
            tool_name = step.get("tool")
            params = step.get("params")
            if not isinstance(tool_name, str) or not isinstance(params, dict):
                continue
            contract = get_tool_contract(tool_name)
            if contract is None:
                raise ValueError(f"Plan step uses unknown tool '{tool_name}'.")
            issues = validate_tool_params(tool_name, params)
            errors = [issue["message"] for issue in issues if issue.get("level") == "error"]
            if errors:
                step_id = step.get("step_id") or step.get("save_as") or tool_name
                raise ValueError(f"Invalid params for step '{step_id}': {'; '.join(errors)}")

    @staticmethod
    def _completed_step_result_types(completed_steps: List[Dict[str, Any]]) -> Dict[str, str]:
        result_types: Dict[str, str] = {}
        for step in completed_steps:
            if not isinstance(step, dict):
                continue
            result_id = step.get("result_id") or step.get("ref_id") or step.get("save_as")
            output_type = step.get("output_type")
            if isinstance(result_id, str) and isinstance(output_type, str):
                result_types[result_id] = output_type
        return result_types

    def _validate_plan_references(
        self,
        plan: Dict[str, Any],
        *,
        available_result_types: Optional[Dict[str, str]] = None,
    ) -> None:
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return

        produced_types: Dict[str, str] = dict(available_result_types or {})
        for step in steps:
            tool_name = step.get("tool")
            params = step.get("params")
            step_id = step.get("step_id") or step.get("save_as") or tool_name
            if not isinstance(tool_name, str) or not isinstance(params, dict):
                continue

            contract = get_tool_contract(tool_name) or {}
            inputs = contract.get("inputs", {})
            for param_name, value in params.items():
                for ref_value in self._iter_ref_values(value):
                    ref_root = self._extract_ref_root(ref_value)
                    if not ref_root or ref_root not in produced_types:
                        raise ValueError(
                            f"Step '{step_id}' parameter '{param_name}' references unknown result "
                            f"{ref_value!r}. References must point to earlier step save_as ids."
                        )
                param_contract = inputs.get(param_name)
                if isinstance(param_contract, dict):
                    self._validate_param_reference_type(
                        step_id=str(step_id),
                        param_name=param_name,
                        value=value,
                        param_contract=param_contract,
                        produced_types=produced_types,
                    )

            save_as = step.get("save_as")
            if isinstance(save_as, str) and save_as:
                if save_as in produced_types:
                    raise ValueError(f"Duplicate step save_as id '{save_as}'.")
                produced_types[save_as] = get_tool_output_type(tool_name) or "generic_result"

    def _validate_param_reference_type(
        self,
        *,
        step_id: str,
        param_name: str,
        value: Any,
        param_contract: Dict[str, Any],
        produced_types: Dict[str, str],
    ) -> None:
        kind = param_contract.get("kind")
        if kind not in {"ref_result", "ref_field"}:
            return
        if not isinstance(value, str) or not value.startswith("$ref:"):
            raise ValueError(
                f"Step '{step_id}' parameter '{param_name}' must be a $ref matching "
                f"{param_contract.get('expected') or kind}."
            )

        ref_expr = value[5:]
        ref_root, separator, ref_path = ref_expr.partition(".")
        actual_type = produced_types.get(ref_root)
        expected = str(param_contract.get("expected") or "")
        expected_type, _, expected_field = expected.partition(".")
        if expected_type and actual_type and actual_type != expected_type:
            raise ValueError(
                f"Step '{step_id}' parameter '{param_name}' expects {expected_type}, "
                f"but {value!r} points to {actual_type}."
            )
        if kind == "ref_result" and separator:
            raise ValueError(
                f"Step '{step_id}' parameter '{param_name}' expects a whole result ref "
                f"({expected or 'ref_result'}), not field path {value!r}."
            )
        if kind == "ref_field" and expected_field and ref_path != expected_field:
            raise ValueError(
                f"Step '{step_id}' parameter '{param_name}' expects field '{expected_field}', "
                f"but received {value!r}."
            )

    @staticmethod
    def _iter_ref_values(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            if value.startswith("$ref:"):
                yield value
            return
        if isinstance(value, dict):
            for nested in value.values():
                yield from SkillPlanner._iter_ref_values(nested)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                yield from SkillPlanner._iter_ref_values(nested)

    def _validate_load_dataset_time_intent(
        self,
        plan: Dict[str, Any],
        *,
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> None:
        if not self._request_has_explicit_time_window(
            user_request=user_request,
            extracted_params=extracted_params,
            additional_context=additional_context,
        ):
            return

        for step in plan.get("steps", []):
            if not isinstance(step, dict) or step.get("tool") != "load_dataset":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            if self._coerce_time_range_strings(params.get("time_range")) is not None:
                continue
            raise ValueError(
                "Time intent was not preserved: the query/proposal specifies a time window, "
                f"but load_dataset step '{step.get('step_id', step.get('save_as', '?'))}' "
                "is missing explicit time_range."
            )

    def _request_has_explicit_time_window(
        self,
        *,
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> bool:
        for source in self._time_intent_sources(extracted_params, additional_context):
            for key in ("time_range", "date_range", "temporal_range"):
                if self._coerce_time_range_strings(source.get(key)) is not None:
                    return True

        request = user_request or ""
        return self._time_range_from_request_text(request) is not None

    @staticmethod
    def _time_range_from_request_text(user_request: str) -> Optional[List[str]]:
        request = user_request or ""
        date_range = re.search(
            r"\b((?:19|20)\d{2}-\d{2}-\d{2})\s*(?:to|through|until|–|—|-|~|至|到)\s*"
            r"((?:19|20)\d{2}-\d{2}-\d{2})\b",
            request,
            re.IGNORECASE,
        )
        if date_range:
            return [date_range.group(1), date_range.group(2)]

        year_range = re.search(
            r"\b((?:19|20)\d{2})\s*(?:to|through|until|–|—|-|~|至|到)\s*((?:19|20)\d{2})\b",
            request,
            re.IGNORECASE,
        )
        if year_range:
            start_year = int(year_range.group(1))
            end_year = int(year_range.group(2))
            return [f"{start_year}-01-01", f"{end_year}-12-31"]

        single_year = re.search(
            r"(?:\b(?:in|during|for|over|throughout|within|calendar\s+year|year)\s+"
            r"((?:19|20)\d{2})\b|\b((?:19|20)\d{2})\s*(?:calendar\s+year|year)\b|"
            r"((?:19|20)\d{2})\s*年)",
            request,
            re.IGNORECASE,
        )
        if single_year:
            year_text = next((group for group in single_year.groups() if group), None)
            if year_text:
                year = int(year_text)
                return [f"{year}-01-01", f"{year}-12-31"]

        years = sorted({int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", request)})
        if len(years) >= 2:
            return [f"{years[0]}-01-01", f"{years[-1]}-12-31"]
        return None

    @staticmethod
    def _time_intent_sources(
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Iterable[Dict[str, Any]]:
        for source in (extracted_params, additional_context):
            if isinstance(source, dict):
                yield source
        if isinstance(additional_context, dict):
            proposal = additional_context.get("analysis_proposal") or additional_context.get("approved_analysis_proposal")
            if isinstance(proposal, dict):
                yield proposal
                skill_plan = proposal.get("skill_plan")
                if isinstance(skill_plan, dict):
                    yield skill_plan

    def _validate_fixed_reports_explicit_only(self, plan: Dict[str, Any], *, user_request: str) -> None:
        for step in plan.get("steps", []):
            if not isinstance(step, dict):
                continue
            tool_name = step.get("tool")
            if tool_name == "assemble_environment_health_report" and not self._explicit_environment_health_report_requested(user_request):
                raise ValueError(
                    "Fixed environment-health report tools are explicit-only. "
                    "Remove assemble_environment_health_report unless the user explicitly asked for a fixed report/card/tool."
                )
            if tool_name == "assemble_policy_recommendation_report" and not self._explicit_policy_report_requested(user_request):
                raise ValueError(
                    "Fixed policy recommendation reports are explicit-only. "
                    "Remove assemble_policy_recommendation_report unless the user explicitly asked for a fixed policy report/card/tool."
                )

    def _validate_environment_health_execution_semantics(
        self,
        plan: Dict[str, Any],
        *,
        user_request: str,
        additional_context: Dict[str, Any],
        available_result_ids: Optional[set[str]] = None,
    ) -> None:
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return
        available_result_ids = available_result_ids or set()

        save_as_to_step = {
            step.get("save_as"): step
            for step in steps
            if isinstance(step, dict) and isinstance(step.get("save_as"), str)
        }
        if any(isinstance(step, dict) and step.get("tool") == "detect_hypoxia" for step in steps):
            self._repair_environment_health_standard_steps(save_as_to_step)
        requested_branch_keys: set[str] = set()
        if "ocean_environment_health_assessment" in self._get_skills_used(plan):
            requested_branch_keys = self._environment_health_requested_branch_keys(user_request)
            requested_branch_keys.difference_update(self._environment_health_suppressed_branch_keys(user_request))
            requested_branch_keys = self._filter_environment_health_branches_by_dataset_support(
                requested_branch_keys,
                additional_context=additional_context,
            )
            self._validate_environment_health_evidence_completeness(
                save_as_to_step,
                requested_branch_keys=requested_branch_keys,
                available_result_ids=available_result_ids,
            )
            self._validate_environment_health_vertical_contracts(
                save_as_to_step,
                requested_branch_keys=requested_branch_keys,
            )

        bottom_oxygen_refs = {
            save_as
            for save_as, step in save_as_to_step.items()
            if self._is_bottom_oxygen_load_step(save_as, step)
        }
        for save_as in sorted(bottom_oxygen_refs):
            self._validate_bottom_oxygen_load_step(save_as_to_step[save_as], save_as=save_as)

        bottom_oxygen_required = (
            self._request_mentions_bottom_oxygen(user_request)
            or bool(
                requested_branch_keys
                & {
                    "bottom_oxygen_trend",
                    "bottom_hypoxia_burden",
                    "hypoxic_days",
                    "hypoxia_statistics",
                }
            )
            or any(isinstance(step, dict) and step.get("tool") == "detect_hypoxia" for step in steps)
        )
        if bottom_oxygen_required:
            for save_as, step in save_as_to_step.items():
                if not isinstance(step, dict) or step.get("tool") != "load_dataset":
                    continue
                params = step.get("params")
                if isinstance(params, dict) and self._canonical_variable_name(params.get("variable")) == "oxygen":
                    self._validate_bottom_oxygen_load_step(step, save_as=str(save_as))

        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("tool") == "detect_hypoxia":
                self._validate_hypoxia_detection_step(step, save_as_to_step, available_result_ids=available_result_ids)
            if step.get("tool") == "compute_event_summary_map":
                self._validate_hypoxia_summary_map_step(step, save_as_to_step, available_result_ids=available_result_ids)
            if step.get("tool") == "compute_event_statistics" and step.get("save_as") == "hypoxia_statistics":
                params = step.get("params") if isinstance(step.get("params"), dict) else {}
                if params.get("events") != "$ref:hypoxia_detection.events":
                    raise ValueError("hypoxia_statistics must use events='$ref:hypoxia_detection.events'.")

    def _validate_environment_health_evidence_completeness(
        self,
        save_as_to_step: Dict[Any, Dict[str, Any]],
        *,
        requested_branch_keys: set[str],
        available_result_ids: set[str],
    ) -> None:
        if not requested_branch_keys:
            return
        for branch_key in self._environment_health_branch_order(requested_branch_keys):
            contract = _ENV_HEALTH_BRANCH_CONTRACTS.get(branch_key)
            if not isinstance(contract, dict):
                continue
            if branch_key == "stratification_strength_change":
                if self._has_stratification_change_evidence(save_as_to_step, available_result_ids):
                    continue
                raise ValueError(
                    "Environment-health evidence plan is incomplete: requested branch "
                    "'stratification_strength_change' requires water-column temperature/salinity "
                    "evidence and a stability/stratification trend. A direct temp+salt -> "
                    "compute_vertical_stability_timeseries -> compute_trend chain is acceptable; "
                    "do not require legacy intermediate result names like 'thermo_dataset'."
                )
            for save_as in contract.get("required_results", []):
                if save_as not in save_as_to_step and save_as not in available_result_ids:
                    raise ValueError(
                        f"Environment-health evidence plan is incomplete: requested branch "
                        f"'{branch_key}' requires result '{save_as}'. Reviewer should revise "
                        "the plan using existing analysis tools, not backend repair."
                    )

    def _validate_environment_health_vertical_contracts(
        self,
        save_as_to_step: Dict[Any, Dict[str, Any]],
        *,
        requested_branch_keys: set[str],
    ) -> None:
        surface_loads: List[Tuple[str, str]] = []
        if "sst_trend" in requested_branch_keys:
            surface_loads.append(("sst_field", "temp"))
        if {"heatwave_burden", "heatwave_days", "upwelling_days"} & requested_branch_keys:
            if "heatwave_burden" in requested_branch_keys or "heatwave_days" in requested_branch_keys:
                surface_loads.append(("heatwave_field", "temp"))
            if "upwelling_days" in requested_branch_keys:
                surface_loads.append(("upwelling_field", "temp"))
        if {"bloom_frequency_change", "bloom_burden", "bloom_event_days"} & requested_branch_keys:
            surface_loads.append(("bloom_field", "chlorophyll"))
        if "eutrophication_context" in requested_branch_keys:
            surface_loads.append(("chlorophyll_context_field", "chlorophyll"))

        for save_as, variable in surface_loads:
            if save_as in save_as_to_step:
                self._require_surface_load_step(save_as_to_step, save_as, variable)

        if "stratification_strength_change" not in requested_branch_keys:
            return
        for save_as, variable in (("temp_field", "temp"), ("salt_field", "salt")):
            step = save_as_to_step.get(save_as)
            if not isinstance(step, dict):
                continue
            if step.get("tool") != "load_dataset":
                raise ValueError(f"{save_as} must use load_dataset for stratification evidence.")
            params = step.get("params")
            if not isinstance(params, dict):
                raise ValueError(f"{save_as} must contain params.")
            if self._canonical_variable_name(params.get("variable")) != variable:
                raise ValueError(f"{save_as} must load variable='{variable}'.")
            depth_range = self._coerce_depth_range_static(params.get("depth_range"))
            if depth_range is not None and self._is_single_depth_range(depth_range):
                raise ValueError(f"{save_as} must retain a multi-level water-column selection.")
            vertical_mode = str(params.get("vertical_mode") or "").strip().lower()
            if vertical_mode in {"bottom", "surface", "fixed_depth"}:
                raise ValueError(f"{save_as} must not use single-level vertical_mode='{vertical_mode}'.")

    def _filter_environment_health_branches_by_dataset_support(
        self,
        requested_branch_keys: set[str],
        *,
        additional_context: Dict[str, Any],
    ) -> set[str]:
        variables = self._dataset_variable_names(additional_context)
        if not variables:
            return set(requested_branch_keys)

        supported = set(requested_branch_keys)
        depth_count = self._dataset_depth_level_count(additional_context)
        if not self._has_any_dataset_variable(variables, {"oxygen", "o2"}):
            supported.difference_update({
                "bottom_oxygen_trend",
                "bottom_hypoxia_burden",
                "hypoxic_days",
                "hypoxia_statistics",
            })
        if not self._has_any_dataset_variable(variables, {"temp", "temperature", "thetao"}):
            supported.difference_update({"sst_trend", "heatwave_burden", "heatwave_days", "upwelling_days"})
        if not self._has_any_dataset_variable(variables, {"chlorophyll", "chl", "chla"}):
            supported.difference_update({
                "bloom_frequency_change",
                "bloom_burden",
                "bloom_event_days",
                "eutrophication_context",
            })
        if (
            not self._has_any_dataset_variable(variables, {"temp", "temperature", "thetao"})
            or not self._has_any_dataset_variable(variables, {"salt", "salinity", "so"})
            or (depth_count is not None and depth_count <= 1)
        ):
            supported.discard("stratification_strength_change")
        return supported

    @staticmethod
    def _dataset_variable_names(additional_context: Dict[str, Any]) -> set[str]:
        dataset = additional_context.get("dataset") if isinstance(additional_context, dict) else None
        if not isinstance(dataset, dict):
            return set()
        raw_variables = dataset.get("variables")
        variables: set[str] = set()
        if isinstance(raw_variables, list):
            variables.update(str(item).strip().lower() for item in raw_variables if str(item).strip())
        variable_names = dataset.get("variable_names")
        if isinstance(variable_names, dict):
            variables.update(str(key).strip().lower() for key in variable_names if str(key).strip())
        return variables

    @staticmethod
    def _has_any_dataset_variable(variables: set[str], aliases: set[str]) -> bool:
        return bool(variables & aliases)

    @staticmethod
    def _dataset_depth_level_count(additional_context: Dict[str, Any]) -> Optional[int]:
        dataset = additional_context.get("dataset") if isinstance(additional_context, dict) else None
        if not isinstance(dataset, dict):
            return None
        depth_levels = dataset.get("depth_levels")
        if isinstance(depth_levels, list):
            return len(depth_levels)
        dimensions = dataset.get("dimensions")
        if isinstance(dimensions, dict):
            depth = dimensions.get("depth")
            if isinstance(depth, int):
                return depth
        return None

    def _has_stratification_change_evidence(
        self,
        save_as_to_step: Dict[Any, Dict[str, Any]],
        available_result_ids: set[str],
    ) -> bool:
        available_ids = set(str(item) for item in available_result_ids)
        produced_ids = {
            save_as
            for save_as in save_as_to_step
            if isinstance(save_as, str)
        }
        if {
            "temp_field",
            "salt_field",
            "stability_timeseries",
            "stratification_trend",
        }.issubset(produced_ids | available_ids):
            return True
        if "stratification_trend" in available_ids or "stability_trend" in available_ids:
            return True

        stability_series_ids: set[str] = set()
        stratification_field_ids: set[str] = set()
        for save_as, step in save_as_to_step.items():
            if not isinstance(save_as, str) or not isinstance(step, dict):
                continue
            tool_name = step.get("tool")
            lowered_id = save_as.lower()
            if tool_name == "compute_vertical_stability_timeseries" or (
                ("stability" in lowered_id or "stratification" in lowered_id)
                and "timeseries" in lowered_id
            ):
                stability_series_ids.add(save_as)
            if tool_name in {"compute_stratification_index", "compute_brunt_vaisala_frequency"} or (
                "stratification" in lowered_id and "trend" not in lowered_id
            ):
                stratification_field_ids.add(save_as)

        for save_as, step in save_as_to_step.items():
            if not isinstance(save_as, str) or not isinstance(step, dict):
                continue
            tool_name = step.get("tool")
            params = step.get("params") if isinstance(step.get("params"), dict) else {}
            lowered_id = save_as.lower()
            if tool_name == "compute_trend":
                timeseries_ref = self._extract_ref_root(params.get("timeseries"))
                if timeseries_ref in stability_series_ids:
                    return True
                if "stratification" in lowered_id or "stability" in lowered_id:
                    return True
            if tool_name == "compute_field_trend":
                data_ref = self._extract_ref_root(params.get("data"))
                if data_ref in stratification_field_ids:
                    return True
                if "stratification" in lowered_id or "stability" in lowered_id:
                    return True
        return False

    def _is_bottom_oxygen_load_step(self, save_as: Any, step: Any) -> bool:
        if not isinstance(save_as, str) or not isinstance(step, dict):
            return False
        if step.get("tool") != "load_dataset":
            return False
        params = step.get("params")
        if not isinstance(params, dict):
            return False
        return "bottom_oxygen" in save_as or self._canonical_variable_name(params.get("variable")) == "oxygen" and str(params.get("vertical_mode") or "").lower() == "bottom"

    def _validate_bottom_oxygen_load_step(self, step: Dict[str, Any], *, save_as: str) -> None:
        params = step.get("params")
        if not isinstance(params, dict):
            raise ValueError(f"{save_as} must contain params.")
        if self._canonical_variable_name(params.get("variable")) != "oxygen":
            raise ValueError(f"{save_as} must load variable='oxygen'.")
        if str(params.get("vertical_mode") or "").strip().lower() != "bottom":
            raise ValueError(f"{save_as} must use vertical_mode='bottom'.")
        if "depth_range" in params or "depth_value" in params:
            raise ValueError(f"{save_as} must not use fixed-depth depth_range/depth_value.")

    def _validate_hypoxia_detection_step(
        self,
        step: Dict[str, Any],
        save_as_to_step: Dict[Any, Dict[str, Any]],
        *,
        available_result_ids: set[str],
    ) -> None:
        params = step.get("params")
        if not isinstance(params, dict):
            raise ValueError("detect_hypoxia step must contain params.")
        if params.get("oxygen") != "$ref:bottom_oxygen_field.data":
            raise ValueError("detect_hypoxia must use oxygen='$ref:bottom_oxygen_field.data'.")
        bottom_step = save_as_to_step.get("bottom_oxygen_field")
        if not isinstance(bottom_step, dict) and "bottom_oxygen_field" not in available_result_ids:
            raise ValueError("detect_hypoxia requires prior bottom_oxygen_field evidence.")
        if isinstance(bottom_step, dict):
            self._validate_bottom_oxygen_load_step(bottom_step, save_as="bottom_oxygen_field")
        if str(params.get("vertical_mode") or "").strip().lower() != "bottom":
            raise ValueError("detect_hypoxia must use vertical_mode='bottom'.")
        if "depth_range" in params or "depth_value" in params:
            raise ValueError("detect_hypoxia must not use fixed-depth depth_range/depth_value for bottom hypoxia.")

    def _validate_hypoxia_summary_map_step(
        self,
        step: Dict[str, Any],
        save_as_to_step: Dict[Any, Dict[str, Any]],
        *,
        available_result_ids: set[str],
    ) -> None:
        save_as = str(step.get("save_as") or "")
        params = step.get("params")
        if not isinstance(params, dict):
            return
        event_ref = params.get("event_detection")
        data_ref = params.get("data")
        is_hypoxia_map = (
            event_ref == "$ref:hypoxia_detection"
            or "hypoxia" in save_as
            or "hypoxic" in save_as
        )
        if not is_hypoxia_map:
            return
        if event_ref != "$ref:hypoxia_detection":
            raise ValueError(f"{save_as or 'hypoxia summary map'} must use event_detection='$ref:hypoxia_detection'.")
        if data_ref != "$ref:bottom_oxygen_field.data":
            raise ValueError(f"{save_as or 'hypoxia summary map'} must use data='$ref:bottom_oxygen_field.data'.")
        if "hypoxia_oxygen_deficit_burden" == save_as and params.get("summary_mode") != "burden":
            raise ValueError("hypoxia_oxygen_deficit_burden must use summary_mode='burden'.")
        if save_as == "hypoxic_days" and params.get("summary_mode") != "event_days":
            raise ValueError("hypoxic_days must use summary_mode='event_days'.")
        bottom_step = save_as_to_step.get("bottom_oxygen_field")
        if not isinstance(bottom_step, dict) and "bottom_oxygen_field" not in available_result_ids:
            raise ValueError("Hypoxia summary maps require prior bottom_oxygen_field evidence.")
        if isinstance(bottom_step, dict):
            self._validate_bottom_oxygen_load_step(bottom_step, save_as="bottom_oxygen_field")

    def _validate_named_region_spatial_intent(
        self,
        plan: Dict[str, Any],
        *,
        user_request: str,
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> None:
        if plan.get("status", "ready") != "ready":
            return
        if self._extracted_params_has_spatial_geometry(extracted_params):
            return

        named_regions = self._extract_named_region_mentions(user_request)
        if not named_regions:
            return

        steps = plan.get("steps")
        if not isinstance(steps, list):
            return

        load_steps = [
            step for step in steps
            if isinstance(step, dict) and step.get("tool") == "load_dataset"
        ]
        if not load_steps:
            return

        context_extents: List[tuple[str, tuple[List[float], List[float]]]] = []
        dataset_extent = self._dataset_spatial_extent(additional_context)
        if dataset_extent is not None:
            context_extents.append(("dataset/full extent", dataset_extent))
        workspace_extent = self._workspace_region_extent(additional_context)
        if workspace_extent is not None:
            context_extents.append(("workspace region", workspace_extent))

        for step in load_steps:
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            lon_range = self._coerce_numeric_range(params.get("lon_range"))
            lat_range = self._coerce_numeric_range(params.get("lat_range"))
            if lon_range is None or lat_range is None:
                raise ValueError(
                    "Named region intent was not preserved: the user named "
                    f"{self._format_region_list(named_regions)}, but load_dataset step "
                    f"'{step.get('step_id', step.get('save_as', '?'))}' is missing explicit "
                    "lon_range/lat_range. Regenerate with approximate bounds for the named "
                    "region, or return clarification_needed if the region cannot be resolved."
                )

            suspicious_reason = self._named_region_extent_suspicion(
                lon_range=lon_range,
                lat_range=lat_range,
                named_regions=named_regions,
                context_extents=context_extents,
            )
            if suspicious_reason is None:
                continue

            raise ValueError(
                "Named region intent was not preserved: the user named "
                f"{self._format_region_list(named_regions)}, but load_dataset step "
                f"'{step.get('step_id', step.get('save_as', '?'))}' uses "
                f"{suspicious_reason} {self._format_spatial_bounds(lon_range, lat_range)}. "
                "Do not silently fall back to dataset or workspace bounds for a named "
                "geographic region. Regenerate the plan with explicit approximate "
                "lon_range/lat_range for the named region, or return clarification_needed "
                "if the region cannot be resolved confidently."
            )

    @staticmethod
    def _extracted_params_has_spatial_geometry(extracted_params: Dict[str, Any]) -> bool:
        for key in ("transect_points", "mask_polygon"):
            value = extracted_params.get(key)
            if isinstance(value, list) and len(value) >= 2:
                return True

        region = extracted_params.get("region")
        if not isinstance(region, dict):
            return False
        lon_range = SkillPlanner._coerce_numeric_range(region.get("lon_range"))
        lat_range = SkillPlanner._coerce_numeric_range(region.get("lat_range"))
        return lon_range is not None and lat_range is not None

    @staticmethod
    def _dataset_spatial_extent(
        additional_context: Dict[str, Any],
    ) -> Optional[tuple[List[float], List[float]]]:
        dataset = additional_context.get("dataset")
        if not isinstance(dataset, dict):
            return None
        spatial_extent = dataset.get("spatial_extent") or dataset.get("spatialExtent")
        if not isinstance(spatial_extent, dict):
            return None
        return SkillPlanner._coerce_spatial_extent_mapping(spatial_extent)

    @staticmethod
    def _workspace_region_extent(
        additional_context: Dict[str, Any],
    ) -> Optional[tuple[List[float], List[float]]]:
        workspace_context = additional_context.get("workspace_context")
        if not isinstance(workspace_context, dict):
            return None

        for key in ("region_bounds", "current_region_bounds"):
            bounds = workspace_context.get(key)
            if isinstance(bounds, dict):
                extent = SkillPlanner._coerce_spatial_extent_mapping(bounds)
                if extent is not None:
                    return extent
        return None

    @staticmethod
    def _coerce_spatial_extent_mapping(
        mapping: Dict[str, Any],
    ) -> Optional[tuple[List[float], List[float]]]:
        lon_range = (
            SkillPlanner._coerce_numeric_range(mapping.get("lon_range"))
            or SkillPlanner._coerce_numeric_range(mapping.get("lon"))
            or SkillPlanner._coerce_numeric_range(mapping.get("longitude"))
        )
        lat_range = (
            SkillPlanner._coerce_numeric_range(mapping.get("lat_range"))
            or SkillPlanner._coerce_numeric_range(mapping.get("lat"))
            or SkillPlanner._coerce_numeric_range(mapping.get("latitude"))
        )

        if lon_range is None:
            lon_range = SkillPlanner._coerce_numeric_range([
                mapping.get("lonMin", mapping.get("lon_min")),
                mapping.get("lonMax", mapping.get("lon_max")),
            ])
        if lat_range is None:
            lat_range = SkillPlanner._coerce_numeric_range([
                mapping.get("latMin", mapping.get("lat_min")),
                mapping.get("latMax", mapping.get("lat_max")),
            ])

        if lon_range is None or lat_range is None:
            return None
        return lon_range, lat_range

    def _validate_skill_selection_shape(self, selection: Dict[str, Any]) -> None:
        if not isinstance(selection, dict):
            raise ValueError("Skill selection result must be a JSON object.")

        status = selection.get("status", "ready")
        if status not in {"ready", "clarification_needed"}:
            raise ValueError("Skill selection status must be 'ready' or 'clarification_needed'.")

        if status == "clarification_needed":
            missing_fields = selection.get("missing_fields")
            question = selection.get("clarification_question")
            if not isinstance(missing_fields, list) or not missing_fields:
                raise ValueError("Clarification result must contain a non-empty 'missing_fields' list.")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Clarification result must contain a non-empty 'clarification_question'.")
            return

        if "skill_id" not in selection or not isinstance(selection.get("skill_id"), str):
            raise ValueError("Skill selection result must contain a string 'skill_id'.")

        skills_used = selection.get("skills_used")
        if skills_used is not None:
            if not isinstance(skills_used, list) or not skills_used or not all(isinstance(skill, str) for skill in skills_used):
                raise ValueError("Skill selection 'skills_used' must be a non-empty list of strings.")

    def _validate_known_skills(self, plan: Dict[str, Any], skill_markdowns: Dict[str, str]) -> None:
        if plan.get("status") == "clarification_needed":
            return
        selected_skill = plan["skill_id"]
        if selected_skill not in skill_markdowns:
            raise ValueError(f"Planner selected an unknown skill_id: {selected_skill}")

        for skill_id in self._get_skills_used(plan):
            if skill_id not in skill_markdowns:
                raise ValueError(f"Planner selected an unknown skill_id in skills_used: {skill_id}")

    def _repair_hallucinated_tool_names(
        self,
        plan: Dict[str, Any],
        skill_markdowns: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Auto-correct hallucinated tool names in a replanned plan.

        When the replanner LLM invents a tool name that is close to a real
        tool (e.g. ``compute_density_from_temp_salt`` vs ``compute_density``),
        replace it with the best matching allowed tool name.
        """
        if plan.get("status", "ready") != "ready":
            return plan

        allowed_tools: set[str] = set()
        for skill_id in self._get_skills_used(plan):
            md = skill_markdowns.get(skill_id)
            if md:
                allowed_tools.update(self._allowed_tool_names_for_skill(skill_id, md))

        if not allowed_tools:
            return plan

        steps = plan.get("steps", [])
        for step in steps:
            tool_name = step.get("tool", "")
            explicit_alias = _SAFE_TOOL_ALIASES.get(tool_name)
            if explicit_alias and explicit_alias in allowed_tools:
                step["tool"] = explicit_alias
                continue
            if tool_name in allowed_tools:
                continue
            # Try to find the best matching allowed tool name.
            best_match = self._find_closest_tool(tool_name, allowed_tools)
            if best_match:
                step["tool"] = best_match

        return plan

    @staticmethod
    def _repair_mesoscale_partition_plan(
        plan: Dict[str, Any],
        *,
        user_request: str,
    ) -> Dict[str, Any]:
        if plan.get("skill_id") != "ocean_mesoscale_organization_analysis":
            return plan

        steps = plan.get("steps")
        if not isinstance(steps, list):
            return plan

        lowered_request = (user_request or "").lower()
        partition_requested = any(
            token in lowered_request
            for token in (
                "subregion",
                "subregions",
                "equal subregions",
                "divide the region",
                "split the region",
                "partition the region",
                "tiles",
                "grids",
                "几等分",
                "等分",
                "分区",
                "子区域",
                "子区",
                "分块",
            )
        )
        if not partition_requested:
            return plan

        grid_match = re.search(r"(\d+)\s*(?:x|×|by)\s*(\d+)", lowered_request)
        if grid_match:
            grid = [int(grid_match.group(1)), int(grid_match.group(2))]
        else:
            grid = [2, 2]

        for step in steps:
            if step.get("tool") != "compute_event_condition_contrast":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            params.setdefault("partition_mode", "lon_lat_grid")
            params.setdefault("subregion_grid", grid)
            params.setdefault("subregion_weighting", "area_weighted")
        return plan

    @staticmethod
    def _repair_season_filter_plan(
        plan: Dict[str, Any],
        *,
        user_request: str,
    ) -> Dict[str, Any]:
        season_filter = SkillPlanner._infer_requested_season_filter(user_request)
        if season_filter is None:
            return plan

        steps = plan.get("steps")
        if not isinstance(steps, list):
            return plan

        for step in steps:
            if step.get("tool") != "load_dataset":
                continue
            params = step.get("params")
            if not isinstance(params, dict):
                continue
            explicit = params.get("season_filter")
            params["season_filter"] = SkillPlanner._normalize_requested_season_filter(explicit) if explicit is not None else season_filter

        return plan

    @staticmethod
    def _infer_requested_season_filter(user_request: str) -> Optional[str]:
        lowered_request = (user_request or "").lower()
        detected = set()

        season_patterns = {
            "DJF": [
                r"\bdjf\b",
                r"\bwinters?\b",
                r"\b(all|every|each)\s+winters?\b",
                r"\bwinter climatolog(?:y|ies)\b",
                r"\bwinter mean\b",
                r"所有冬季",
                r"全部冬季",
                r"每个冬季",
                r"多年冬季",
                r"冬季气候态",
                r"冬季平均",
            ],
            "JJA": [
                r"\bjja\b",
                r"\bsummers?\b",
                r"\b(all|every|each)\s+summers?\b",
                r"\bsummer climatolog(?:y|ies)\b",
                r"\bsummer mean\b",
                r"所有夏季",
                r"全部夏季",
                r"每个夏季",
                r"多年夏季",
                r"夏季气候态",
                r"夏季平均",
            ],
        }

        for canonical, patterns in season_patterns.items():
            if any(re.search(pattern, lowered_request) for pattern in patterns):
                detected.add(canonical)

        if len(detected) != 1:
            return None
        return next(iter(detected))

    @staticmethod
    def _normalize_requested_season_filter(value: Any) -> Optional[str]:
        if value is None:
            return None
        aliases = {
            "djf": "DJF",
            "winter": "DJF",
            "mam": "MAM",
            "spring": "MAM",
            "jja": "JJA",
            "summer": "JJA",
            "son": "SON",
            "fall": "SON",
            "autumn": "SON",
        }
        normalized = aliases.get(str(value).strip().lower())
        return normalized or str(value)

    @staticmethod
    def _find_closest_tool(hallucinated: str, allowed: set) -> str:
        """Return the allowed tool name that shares the longest common
        subsequence with ``hallucinated``, provided the match is strong
        enough (at least 60% of the longer name)."""
        best, best_score = "", 0.0
        hall_lower = hallucinated.lower()
        for candidate in allowed:
            cand_lower = candidate.lower()
            # Check if the hallucinated name contains the candidate or vice versa.
            if cand_lower in hall_lower or hall_lower in cand_lower:
                score = len(cand_lower) / max(len(hall_lower), 1)
                if score > best_score:
                    best, best_score = candidate, score
                continue
            # Simple token-overlap scoring.
            hall_tokens = set(hall_lower.split("_"))
            cand_tokens = set(cand_lower.split("_"))
            overlap = len(hall_tokens & cand_tokens)
            total = max(len(hall_tokens | cand_tokens), 1)
            score = overlap / total
            if score > best_score:
                best, best_score = candidate, score
        # Only accept if the match is strong enough.
        if best_score >= 0.5:
            return best
        return ""

    def _validate_plan_tools(self, plan: Dict[str, Any], skill_markdowns: Dict[str, str]) -> None:
        if plan.get("status", "ready") != "ready":
            return

        allowed_tools = set()
        for skill_id in self._get_skills_used(plan):
            allowed_tools.update(self._allowed_tool_names_for_skill(skill_id, skill_markdowns[skill_id]))

        if not allowed_tools:
            return

        for step in plan.get("steps", []):
            tool_name = step["tool"]
            if tool_name not in allowed_tools:
                allowed_tool_list = ", ".join(sorted(allowed_tools))
                raise ValueError(
                    f"Plan step uses unknown tool '{tool_name}'. "
                    f"Allowed tools for selected skills: {allowed_tool_list}"
                )

    def _get_skills_used(self, plan: Dict[str, Any]) -> List[str]:
        skills_used = plan.get("skills_used")
        if isinstance(skills_used, list) and skills_used:
            return skills_used
        return [plan["skill_id"]]

    def _validate_review_shape(self, decision: Dict[str, Any]) -> None:
        if not isinstance(decision, dict):
            raise ValueError("Review result must be a JSON object.")

        action = decision.get("decision")
        if action not in {"continue", "replan", "ask_user", "abort"}:
            raise ValueError("Review decision must be one of continue, replan, ask_user, abort.")

        reason = decision.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Review result must contain a non-empty 'reason'.")

        if action == "replan" and not isinstance(decision.get("updated_plan"), dict):
            raise ValueError("Replan decision must include an 'updated_plan' object.")

        if action == "ask_user":
            missing_fields = decision.get("missing_fields")
            question = decision.get("clarification_question")
            if not isinstance(missing_fields, list) or not missing_fields:
                raise ValueError("ask_user decision must include a non-empty 'missing_fields' list.")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("ask_user decision must include a non-empty 'clarification_question'.")
