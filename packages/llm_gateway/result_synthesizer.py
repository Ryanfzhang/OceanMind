"""
LLM-based scientific result analyzer.

This module turns a completed execution state into evidence-backed scientific
findings, notable patterns, limitations, and optional UI actions.
"""

import copy
import json
import re
from typing import Any, Dict, List, Optional, Set

from packages.llm_gateway.skill_planner import SkillPlanner
from packages.llm_gateway.config import load_model_name
from packages.llm_gateway.english_translation import needs_english_translation, translate_strings_to_english
from packages.llm_gateway.openai_compatible_client import OpenAICompatibleClientAdapter


class ResultSynthesizer:
    """Interpret multi-step execution results for users or frontends."""

    INTEGRATED_ASSESSMENT_PROFILE_ID = "ocean_integrated_assessment"
    DEFAULT_MODEL = load_model_name("RESULT_SYNTHESIZER_MODEL", default=SkillPlanner.DEFAULT_MODEL)
    POLICY_PRIORITIES: Set[str] = {"high", "medium", "low", "screening"}
    POLICY_ACTION_TYPES: Set[str] = {
        "monitoring",
        "source_control",
        "discharge_outlet",
        "river_estuary",
        "seasonal_management",
        "coastal_planning",
        "economic_assessment",
        "governance",
    }
    POLICY_EVIDENCE_STRENGTHS: Set[str] = {"supported", "limited", "screening", "not_supported"}
    POLICY_SYNTHESIS_CONFIDENCE: Set[str] = {"supported", "limited", "screening", "not_supported"}
    POLICY_RECOMMENDATION_EVIDENCE_STATUS: Set[str] = {"computed", "indirect", "data_gap"}
    ASSESSMENT_THREAD_STATUS: Set[str] = {"computed", "indirect", "data_gap"}
    ASSESSMENT_THREAD_THEMES: Set[str] = {
        "warming/heatwave",
        "bottom_oxygen/hypoxia",
        "stratification",
        "chlorophyll/bloom",
        "data_gap",
    }
    POLICY_SYNTHESIS_ACTION_GROUPS: Set[str] = {
        "spatial_priority",
        "oxygen_response",
        "seasonal_operations",
        "driver_adaptation",
        "source_pathway_screening",
        "economic_data_assessment",
        "validation_gap",
    }
    NUMERIC_PATTERN = r"[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?"
    POLICY_THRESHOLD_CLAIM_RE = re.compile(
        rf"(?:[<>≤≥]\s*{NUMERIC_PATTERN}|"
        r"\b(?:above|over|more than|less than|below|under|at least|at most|greater than|lower than)\s+"
        rf"{NUMERIC_PATTERN}|"
        rf"\b(?:threshold|trigger|cutoff|cut-off)\s*(?:of|at|=|is|:)?\s*{NUMERIC_PATTERN})",
        re.IGNORECASE,
    )
    NUMERIC_TOKEN_RE = re.compile(NUMERIC_PATTERN, re.IGNORECASE)
    POLICY_CUTOFF_CONTEXT_RE = re.compile(
        r"\b(?:avoid|restrict|ban|prohibit|exclude|no new|new siting|licen[cs]e|licensing|permit|"
        r"zoning|zone|buffer|threshold|cutoff|cut-off|trigger|criterion|criteria|limit|cap|"
        r"require|required|suspend|closure|close|reduce|stocking|density|operation|operations)\b|"
        r"禁止|限制|阈值|门槛|触发|分区|缓冲区|养殖|选址|许可|管控|停养|减量",
        re.IGNORECASE,
    )
    POLICY_GENERIC_ROW_RE = re.compile(
        r"\b(?:high[- ]burden zones?|high[- ]burden cells?|top\s+\d*\s*hotspot(?: cells?|s)?|"
        r"top hotspot(?: cells?|s)?|strengthen monitoring|enhanced monitoring|management review|"
        r"risk severity|high[- ]risk zone|oxygen[- ]risk hotspot|computed hotspot|mapped hotspot)\b",
        re.IGNORECASE,
    )
    POLICY_GENERIC_ACTION_RE = re.compile(
        r"\b(?:strengthen|enhance|improve|prioritize|conduct|implement|support)\s+"
        r"(?:monitoring|management|review|protection|planning)\b|"
        r"\b(?:take action|manage risk|support decisions|inform policy)\b",
        re.IGNORECASE,
    )
    POLICY_EVIDENCE_ANCHOR_RE = re.compile(
        rf"\b(?:lon|lat|rank(?:ed)?|coordinate|coordinates|event[_ -]?count|event count|events?|"
        rf"slope|p[-_ ]?value|r[-_ ]?squared|max(?:imum)?|min(?:imum)?|mean|total|centroid|"
        rf"hotspot_region_label|result[_ -]?ids?|[a-z0-9]+_[a-z0-9_]+)\b|{NUMERIC_PATTERN}|°",
        re.IGNORECASE,
    )
    POLICY_OPERATIONAL_DETAIL_RE = re.compile(
        r"\b(?:near[- ]bottom|bottom oxygen|dissolved oxygen|oxygen monitoring|sensor|sensors|ctd|cast|casts|"
        r"profile|profiles|sampling|station|stations|transect|monitoring network|early warning|"
        r"threshold|site suitability|permit|permitting|zoning|marine spatial planning|carrying[- ]capacity|"
        r"stocking|harvest|deployment|fallowing|validate|validation|map review|inspect(?:ing)? the .*map|"
        r"production|cost|revenue|species[- ]tolerance|vertical|water[- ]quality|source investigation)\b",
        re.IGNORECASE,
    )
    POLICY_INTENT_RE = re.compile(
        r"\b(policy|policies|management|recommendation|recommendations|economic|economics|"
        r"governance|coastal management|regulation|regulatory|mitigation|decision support|action plan|"
        r"source[- ]?control|nutrient[- ]?control|pollution[- ]?control|discharge[- ]?control)\b|"
        r"政策|管理|建议|经济|治理|管控|监管|排口|减缓|行动",
        re.IGNORECASE,
    )
    STRUCTURED_POLICY_INTENT_RE = re.compile(
        r"\b(evidence[- ]action matrix|action matrix|policy matrix|management matrix|"
        r"policy report|policy card|report card|fixed report|standalone policy)\b|"
        r"证据.*行动.*矩阵|行动矩阵|政策矩阵|政策报告|报告卡|固定报告",
        re.IGNORECASE,
    )
    NEGATED_STRUCTURED_POLICY_INTENT_RE = re.compile(
        r"\b(?:do\s+not|don't|dont|without|no|not|avoid|skip|exclude)\b.{0,80}?"
        r"\b(?:evidence[- ]action matrix|action matrix|policy matrix|management matrix|"
        r"policy report|policy card|report card|fixed report|standalone policy)\b|"
        r"(?:不要|不需要|无需|不生成|别|避免|排除|跳过).{0,80}?"
        r"(?:证据.*行动.*矩阵|行动矩阵|政策矩阵|政策报告|报告卡|固定报告)",
        re.IGNORECASE,
    )
    INTEGRATED_ASSESSMENT_INTENT_RE = re.compile(
        r"\b(aquaculture|marine ranching|fish farming|fishery|fisheries|"
        r"environmental risk|marine health|environmental health|suitab|remain suitable)\b|"
        r"养鱼|海洋牧场|水产|渔业|适合|风险|环境健康",
        re.IGNORECASE,
    )
    FUTURE_OR_PROJECTION_INTENT_RE = re.compile(
        r"\b(will|future|coming years|remain|forecast|projection|projected|scenario)\b|"
        r"未来|以后|将来|预测|预估",
        re.IGNORECASE,
    )
    FUTURE_EVIDENCE_RE = re.compile(
        r"\b(future_projection|projection|projected|forecast|scenario|climate model|rcp|ssp)\b|"
        r"未来情景|预测|预估",
        re.IGNORECASE,
    )
    OXYGEN_MECHANISM_RE = re.compile(
        r"\b(respiration|breathing|feeding|growth|survival|bottom habitat|habitat suitability|"
        r"oxygen supply|oxygen exposure)\b|呼吸|摄食|生长|存活|栖息|氧暴露",
        re.IGNORECASE,
    )
    THERMAL_MECHANISM_RE = re.compile(
        r"\b(metabolic demand|metabolism|oxygen solubility|thermal stress|heat stress|"
        r"temperature[- ]sensitive|reproduction|disease resistance)\b|代谢|溶解氧|热胁迫|繁殖|抗病",
        re.IGNORECASE,
    )
    BLOOM_MECHANISM_RE = re.compile(
        r"\b(oxygen depletion|toxin|toxins|light limitation|decay|ecological pressure|food[- ]web)\b|"
        r"耗氧|毒素|光限制|分解|生态压力",
        re.IGNORECASE,
    )
    HISTORICAL_INFERENCE_RE = re.compile(
        r"\b(historical|trend[- ]based|not a forecast|not a projection|no future projection|"
        r"recent trend|observed)\b|历史|趋势推断|不是预测|缺少未来",
        re.IGNORECASE,
    )
    ECONOMIC_CLAIM_RE = re.compile(
        r"\b(cost-benefit|net benefit|benefit-cost|(?:economic|financial|commercial|market)\s+viability|"
        r"profit|profits|revenue|revenues|"
        r"dollar|dollars|usd|rmb|yuan|monetary|economic loss|economic losses|"
        r"economic damage|economic damages|avoided cost|costs?|benefits?|damages?|loss(?:es)?)\b",
        re.IGNORECASE,
    )
    ECONOMIC_DATA_GAP_RE = re.compile(
        r"\b(?:no|without|missing|lack(?:ing)?|absent|insufficient|unavailable|"
        r"not supplied|not provided|not available|did not include|does not include|not include)\b.{0,140}"
        r"\b(?:economic|production|stock|cost|benefit|revenue|loss|damage|valuation)\w*\b.{0,80}"
        r"\b(?:data|dataset|evidence|valuation|assessment)\b|"
        r"\b(?:economic|production|stock|cost|benefit|revenue|loss|damage|valuation)\w*\b.{0,80}"
        r"\b(?:data|dataset|evidence|valuation|assessment)\b.{0,140}"
        r"\b(?:no|without|missing|lack(?:ing)?|absent|insufficient|unavailable|"
        r"not supplied|not provided|not available|did not include|does not include|not include|needed|required)\b",
        re.IGNORECASE,
    )
    ECONOMIC_DATA_COLLECTION_RE = re.compile(
        r"\b(?:collect|gather|compile|obtain|add|require|request|build)\b.{0,100}"
        r"\b(?:economic|production|cost|benefit|revenue|loss|damage|valuation)\w*\b.{0,80}"
        r"\b(?:data|dataset|evidence|inventory|assessment)\b|"
        r"\b(?:economic|production|cost|benefit|revenue|loss|damage|valuation)\w*\b.{0,80}"
        r"\b(?:data|dataset|evidence|inventory|assessment)\b.{0,100}"
        r"\b(?:collect|gather|compile|obtain|add|require|request|build)\b",
        re.IGNORECASE,
    )
    ECONOMIC_LIMITATION_RE = re.compile(
        r"\b(?:cannot|can't|can not|could not|unable|not able|unavailable|"
        r"no|without|missing|lack(?:ing)?|absent|not supplied|not provided|not available|"
        r"not quantified|not quantifiable|not estimated|not assessed|not evaluated|"
        r"insufficient|needed|required|did not include|does not include|not include)\b|"
        r"\b(?:do not|don't|should not|must not|may not)\s+"
        r"(?:invent|infer|estimate|quantify|claim|state|assume|convert)\b|"
        r"\b(?:not|no)\b.{0,50}\b(?:estimate|estimates|claim|claims|conclusion|conclusions|quantification)\b|"
        r"\b(?:require|requires|need|needs|needed|must have)\b.{0,120}"
        r"\b(?:economic|production|accounting|cost|benefit|revenue|loss|damage|valuation)\w*\b.{0,80}"
        r"\b(?:data|dataset|evidence|inventory|assessment)\b",
        re.IGNORECASE,
    )
    ECONOMIC_ASSERTION_RE = re.compile(
        r"\b(?:risk of|risks of|exposure to|faces? (?:elevated )?risk of|economic viability)\b.{0,80}"
        r"\b(?:loss|losses|damage|damages|revenue|profit|profits|(?:economic|financial|commercial|market)\s+viability)\b|"
        r"\b(?:will|would)\s+(?:cause|increase|decrease|reduce|improve|generate|save|avoid|damage|raise|lower)\b|"
        r"\b(?:cause|causes|caused|increase|increases|increased|decrease|decreases|decreased|"
        r"reduce|reduces|reduced|improve|improves|improved|generate|generates|generated|"
        r"save|saves|saved|avoid|avoids|avoided)\b.{0,80}"
        r"\b(?:cost|costs|benefit|benefits|revenue|revenues|loss|losses|damage|damages|profit|profits)\b",
        re.IGNORECASE,
    )
    UNSUPPORTED_EXTERNAL_FACT_RE = re.compile(
        r"\b(?:dominant mariculture producer|world'?s dominant|largest mariculture|production|producer|"
        r"sea cucumber mortalit(?:y|ies)|mortality events?|coral bleaching|bleaching risk|"
        r"reported in|studies show|literature shows)\b|"
        r"海参.*死亡|珊瑚.*白化|养殖产量|产量|世界.*水产|研究表明|文献表明",
        re.IGNORECASE,
    )
    DATA_GAP_EVIDENCE_NOTE_RE = re.compile(
        r"\b(?:need|needs|needed|require|requires|required|more|additional|missing|lack(?:ing)?|"
        r"data gap|insufficient|unavailable|monitor(?:ing)?|investigat(?:e|ion)|screen(?:ing)?|"
        r"survey|validate|validation|assess(?:ment)?)\b|"
        r"需要|缺少|不足|数据缺口|监测|调查|筛查|验证|评估",
        re.IGNORECASE,
    )

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = DEFAULT_MODEL,
        planner: Optional[SkillPlanner] = None,
    ):
        self.client = client
        self.model = model
        self.planner = planner
        self._adapter = OpenAICompatibleClientAdapter(
            model=model,
            client=client,
        )

    def synthesize(
        self,
        user_request: str,
        active_plan: Dict[str, Any],
        completed_steps: List[Dict[str, Any]],
        result_summaries: Dict[str, Dict[str, Any]],
        additional_context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2600,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Generate a structured scientific analysis from execution metadata.
        """
        client = self._get_client()
        normalized_context = self._normalize_additional_context_to_english(
            additional_context or {},
            client=client,
        )
        synthesis_profile_id = self._resolve_synthesis_profile_id(
            user_request=user_request,
            active_plan=active_plan,
            additional_context=normalized_context,
        )
        integrated_assessment_requested = (
            synthesis_profile_id == self.INTEGRATED_ASSESSMENT_PROFILE_ID
        )
        structured_policy_requested = self._structured_policy_guidance_requested(user_request)
        planner_policy_making_intent = self._context_policy_making_intent(normalized_context)
        policy_requested = structured_policy_requested or (
            (
                planner_policy_making_intent
                if planner_policy_making_intent is not None
                else self._policy_guidance_requested(user_request)
            )
            and not integrated_assessment_requested
        )
        if integrated_assessment_requested:
            max_tokens = max(int(max_tokens), 4200)
        elif policy_requested:
            max_tokens = max(int(max_tokens), 4200)
        attempts = 2
        validation_error: Optional[str] = None

        for attempt in range(attempts):
            messages = self._build_messages(
                user_request=user_request,
                active_plan=active_plan,
                completed_steps=completed_steps,
                result_summaries=result_summaries,
                additional_context=normalized_context,
                policy_requested=policy_requested,
                structured_policy_requested=structured_policy_requested,
                synthesis_profile_id=synthesis_profile_id,
                validation_error=validation_error,
            )

            response = self._adapter.create_message(
                client=client,
                max_tokens=max_tokens,
                temperature=temperature,
                system=messages["system"],
                messages=messages["messages"],
                request_name="synthesize_result" if attempt == 0 else "synthesize_result_retry",
                json_response=True,
            )

            response_text = self._extract_response_text(response)
            try:
                synthesis = self._parse_json_response(response_text)
                if not policy_requested:
                    synthesis.pop("policy_guidance", None)
                synthesis = self._repair_synthesis_before_validation(
                    synthesis,
                    result_summaries=result_summaries,
                    additional_context=normalized_context,
                )
                self._validate_synthesis_shape(
                    synthesis,
                    policy_requested=policy_requested,
                    structured_policy_requested=structured_policy_requested,
                    synthesis_profile_id=synthesis_profile_id,
                    user_request=user_request,
                    result_summaries=result_summaries,
                    additional_context=normalized_context,
                )
                synthesis = self._compact_synthesis_output(
                    synthesis,
                    policy_requested=policy_requested,
                    synthesis_profile_id=synthesis_profile_id,
                )
                return self._normalize_synthesis_output_to_english(synthesis, client=client)
            except Exception as exc:
                validation_error = str(exc)
                if attempt + 1 < attempts:
                    continue
                raise

        raise ValueError(validation_error or "Synthesis failed.")

    def _build_messages(
        self,
        user_request: str,
        active_plan: Dict[str, Any],
        completed_steps: List[Dict[str, Any]],
        result_summaries: Dict[str, Dict[str, Any]],
        additional_context: Dict[str, Any],
        policy_requested: bool = False,
        structured_policy_requested: Optional[bool] = None,
        synthesis_profile_id: Optional[str] = None,
        validation_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        if structured_policy_requested is None:
            structured_policy_requested = policy_requested
        compact_retry_mode = bool(validation_error and (policy_requested or synthesis_profile_id == self.INTEGRATED_ASSESSMENT_PROFILE_ID))
        policy_context_prompt = ""
        if isinstance(additional_context.get("policy_context_packet"), dict):
            policy_context_prompt = (
                "\nWhen `additional_context.policy_context_packet` is present, use it as the compact policy "
                "vocabulary and evidence-status packet. Prefer its professional policy levers and lexicon, but "
                "do not force every useful policy idea to have direct computed evidence. Keep policy text concrete, "
                "clear about evidence strength, and natural to read. "
                "The action_framework levers are a reference vocabulary, not an exhaustive list; compose levers "
                "or describe a more specific policy lever when the evidence calls for it. Treat "
                "`risk_signals.action_opportunities` as possible policy candidates when present, but write "
                "recommendations as concise briefing-style cards, not field-heavy matrices. Use "
                "`risk_signals.evidence_anchors` and hotspot `evidence_anchor` strings as computed evidence anchors "
                "when evidence_status is computed or indirect. Do not invent "
                "numeric policy thresholds, zone cutoffs, or trigger values. A numeric threshold such as '>1000 days' "
                "is allowed only if the exact threshold appears in the evidence packet or result summary. For "
                "data-gap recommendations, explicitly say more monitoring/data/investigation is needed and do not "
                "present the recommendation as a computed finding.\n"
            )
        policy_prompt = ""
        if policy_requested:
            matrix_instruction = (
                "The user explicitly requested a structured policy matrix/report. The matrix must include "
                "the required action categories when endpoint evidence warrants them.\n"
                if structured_policy_requested
                else "The user requested policy or management advice, but not a fixed report/matrix. "
                "Provide a compact evidence-action matrix, but do not force a complete fixed taxonomy; "
                "include only rows directly supported by the evidence.\n"
            )
            policy_prompt = (
                "\nPolicy / management guidance is requested. Include `policy_guidance` in the JSON. "
                "It must be an evidence-bounded Evidence-Action Matrix based only on `evidence_packets` "
                "and `result_summaries`; "
                "do not introduce uncomputed data, source attribution, or external assumptions.\n"
                f"{matrix_instruction}"
                "Start policy guidance with a concrete `place_based_policy_brief`. If evidence packets "
                "include hotspot coordinates or hotspot_region_label, name that place and connect it to "
                "specific management implications.\n"
                "Policy evidence rules:\n"
                "- Bottom oxygen and hypoxia are endpoint evidence: they can support oxygen monitoring, "
                "hotspot prioritization, seasonal warning, and coastal planning. If multiple mapped "
                "hotspots are provided, recommend a ranked portfolio of places rather than a single-site policy.\n"
                "- SST and stratification are vulnerability or timing evidence only. They must not be used "
                "as pollution-source attribution, but they should still produce concrete adaptation actions "
                "such as vertical-profile monitoring, ventilation screening, and seasonal timing review when present.\n"
                "- Decreasing chlorophyll must not be used as supported evidence for nutrient/source-control "
                "conclusions. It may only weaken or limit a eutrophication pathway claim.\n"
                "- Nutrient/organic loading, river/estuary inputs, and discharge outlets must be framed as "
                "screening, review, low-regret, or hypothesis-testing unless direct source evidence is present.\n"
                "- Economic guidance without economic data may include policy reasoning or motivate economic "
                "assessment, but label it as indirect reasoning or a data gap. Do not quantify costs, benefits, "
                "revenue, damages, or cost-benefit conclusions unless those data are present.\n"
                "Each matrix row must include priority, action_type, target, where_when, evidence_basis, "
                "recommendation, guardrail, and evidence_strength. Avoid generic 'strengthen monitoring' "
                "unless the row says exactly what to monitor, where/when, and why. Evidence basis should quote "
                "available computed anchors such as coordinates, extrema, event counts, trend slope/p-value, "
                "timing windows, and result IDs rather than only naming a broad risk category.\n"
                "Keep the matrix focused: at most 4 rows, each text field usually under 50 words.\n"
            )
        integrated_prompt = ""
        if synthesis_profile_id == self.INTEGRATED_ASSESSMENT_PROFILE_ID:
            integrated_prompt = (
                "\nSYNTHESIS PROFILE: ocean_integrated_assessment.\n"
                "Include a simplified `integrated_assessment` object in the JSON. This is a summary-layer profile only: "
                "do not imply that an integrated-assessment tool or planner skill ran.\n"
                "Use completed tool results, `evidence_packets`, and `result_summaries` to assess marine "
                "environmental suitability for aquaculture or marine ranching, higher-risk areas, environmental "
                "drivers, economic-development implications, environmental-protection implications, policy or "
                "management recommendations, and uncertainty/data gaps.\n"
                "Do not include `policy_guidance` unless the required_output_schema explicitly includes it. "
                "For ordinary policy or management advice in this profile, use `integrated_assessment.management_guidance`.\n"
                "Integrated assessment evidence rules:\n"
                "- High-risk area claims must be grounded in spatial or regional evidence such as bottom oxygen, "
                "hypoxia, oxygen-deficit burden, trend, hotspot, event, SST, heat-stress, or stratification outputs.\n"
                "- Bottom oxygen and hypoxia are endpoint evidence for aquaculture environmental risk. SST and "
                "stratification are vulnerability/timing evidence unless direct endpoint evidence is present.\n"
                "- If bottom oxygen or hypoxia evidence is missing, say the high-risk-area assessment is "
                "insufficient; do not rank specific high-risk areas from SST or stratification alone.\n"
                "- If the user asks about future or coming years but no future projection/scenario result is "
                "provided, frame the answer as historical trend based inference, not as a true forecast.\n"
                "- Economic-development implications without economic data may discuss exposure, operational "
                "stability, screening needs, reasonable policy/industry interpretation, or the need for economic "
                "assessment, but label unsupported parts as indirect reasoning or data gaps. Do not quantify costs, "
                "benefits, revenue, damages, or cost-benefit conclusions unless those data are present.\n"
                "Organize the main integrated answer as a judgment-first narrative, not a field-heavy audit table: "
                "start with `integrated_assessment.direct_answer` as 1-2 sentences, then "
                "`integrated_assessment.assessment_narrative` as 5-8 natural-language paragraphs. The narrative "
                "should be evidence-dense while preserving the same structure: explain the overall suitability "
                "judgment, the main environmental signals, regional or seasonal differences that are actually "
                "supported by completed evidence, stabilizing/adaptation context, and evidence limits. When "
                "available, weave in concrete evidence anchors from result summaries or evidence packets, such as "
                "hotspot names, coordinates, event days, burden values, trend slopes/p-values, extrema, and timing "
                "windows; avoid just repeating the shorter evidence_threads. Use `integrated_assessment.evidence_threads` to group tool results into "
                "`warming/heatwave`, `bottom_oxygen/hypoxia`, `stratification`, `chlorophyll/bloom`, and `data_gap` "
                "threads with status `computed`, `indirect`, or `data_gap`.\n"
                "When the user asks which areas or regions face higher risk, also include "
                "`integrated_assessment.higher_risk_regions` as a compact table with columns for region, major "
                "environmental risks, and evidence. Prefer the smallest supported geographic unit (bay, estuary, "
                "nearshore shelf, or hotspot coordinate) over broad seas; if only a broad sea is supported, list "
                "the severe coordinate(s) in the evidence cell. Use named regions only when supported by completed "
                "hotspot/map labels, coordinates, result summaries, or explicit map-review language. The evidence "
                "cell must cite concrete tool evidence when available, or clearly mark the row as needing more "
                "data/map review.\n"
                "For every substantive narrative paragraph after the opening judgment, use a two-part structure: "
                "first interpret what the data/result indicates, then state why that matters for the user's actual "
                "question. This scientific implication sentence should be general and reusable across integrated "
                "assessment questions, not hard-coded to aquaculture: e.g. explain how warming, oxygen stress, "
                "stratification, circulation, blooms, salinity, or other computed drivers affect suitability, risk, "
                "mechanism support, timing, or uncertainty. Use plain-language mechanism sentences, for example "
                "`This matters because...`: low oxygen can limit respiration, feeding, growth, survival, and bottom "
                "habitat suitability for aquatic animals; warming can raise metabolic demand and reduce oxygen "
                "solubility; blooms can indicate ecological pressure and may amplify oxygen depletion after decay. "
                "These generic mechanisms are allowed, but do not claim observed species mortality, production loss, "
                "or disease outbreaks unless such evidence is present. Domain-level reasoning is allowed only as mechanism-level "
                "interpretation. Clearly separate computed facts from indirect reasoning and data gaps. If you include "
                "reasonable interpretation that is not directly supported by completed tools, keep it but label the "
                "evidence boundary in `synthesis_warnings` or `integrated_assessment.evidence_boundary_notes` rather "
                "than deleting useful context. Do not state external facts such as production dominance, "
                "reported mortality events, coral bleaching, species impacts, or economic outcomes unless those facts "
                "are present in additional_context or completed results.\n"
                "When policy or management guidance is explicitly requested, organize it as a policy briefing after "
                "the main assessment narrative: first `policy_synthesis.one_sentence_judgment` as one direct "
                "suitability/risk judgment, then `policy_synthesis.policy_narrative` as a 200-400 word policy "
                "reasoning narrative, then up to 4 concise `policy_synthesis.policy_recommendations`. The narrative "
                "may combine computed evidence, domain policy reasoning, and explicit data gaps. Recommendations may "
                "be based on direct computed evidence, indirect environmental evidence, or a policy-relevant data gap; "
                "label that honestly with `evidence_status` = computed, indirect, or data_gap. Do not require every "
                "useful policy action to have direct computed evidence, but never present a data-gap recommendation "
                "as a computed finding.\n"
                "Do not include `policy_design_framework`, `decision_rows`, `policy_frame`, or user-visible "
                "`guardrail` fields in `integrated_assessment`.\n"
                "Named regional claims must come from completed evidence such as hotspot labels, lon/lat, "
                "map extrema, event counts, trend values, or map-review statements. Do not invent or require "
                "a fixed region matrix when the tools only produced hotspot or spatial-map evidence.\n"
                "If `assessment_context_packet` is provided, use its `evidence_threads`, `narrative_guidance`, "
                "and `data_gaps` to structure the direct answer and assessment narrative before writing any policy "
                "recommendations.\n"
                "If `policy_context_packet` is provided, include `integrated_assessment.policy_synthesis`: "
                "one direct judgment, one narrative, and up to 4 natural-language recommendations. Each "
                "recommendation should contain a clear action, priority places when available, one concise "
                "evidence_note, an evidence_status, and evidence_result_ids only when the status is computed or "
                "indirect. Source/pathway, nutrient, species-tolerance, economic, acidification, wave, typhoon, "
                "or production actions may be included as evidence_status='data_gap' when policy-relevant, but "
                "the evidence_note must say they need more data, monitoring, or investigation.\n"
            )
        retry_prompt = ""
        if compact_retry_mode:
            retry_prompt = (
                "\nMINIMAL VALID SCHEMA MODE: the previous response failed JSON/schema validation. "
                "Return one shorter valid JSON object. Use at most 2 scientific_findings, at most 4 policy "
                "or policy_synthesis recommendations, no field-heavy matrices, and short strings. Close every "
                "array/object and include commas between fields.\n"
            )
        system_prompt = (
            "You are an ocean-science analysis assistant. Interpret completed analysis results in "
            "concise, evidence-backed JSON.\n"
            "IMPORTANT: Respond in ENGLISH only, regardless of the user's query language.\n"
            "Your job is not to merely restate what tools ran. Extract scientific findings that are "
            "explicitly supported by the provided summaries.\n"
            "Use only the provided execution metadata, evidence packets, and result summaries.\n"
            "Never reveal local server filesystem paths or data_path values. If path-related metadata is redacted, "
            "state that local storage paths are hidden and use safe metadata such as backend, variables, coverage, "
            "resolution, chunks, or store availability instead.\n"
            "When `additional_context.evidence_packets` is present, treat it as the highest-value compact "
            "scientific evidence and use it before the generic `result_summaries`. Evidence packets are "
            "bounded summaries; they intentionally exclude full arrays.\n"
            "Prefer concrete findings tied to numbers, dates, depths, coordinates, counts, p-values, "
            "variance explained, correlations, and extrema.\n"
            "If evidence is insufficient for a stronger claim, say so explicitly in limitations or "
            "uncertainties.\n"
            "Keep the JSON compact and complete: summary under 160 words, findings under 60 words each, "
            "policy/integrated-assessment action fields under about 50 words, and policy rationale fields "
            "under about 80 words unless a numeric evidence detail requires more.\n"
            "Do not invent causal mechanisms or physical explanations that are not supported.\n"
            f"{policy_context_prompt}"
            "When additional_context contains synthesizer_prior_context_text, you may reference earlier "
            "findings and supporting prior results to build cumulative scientific explanations. However: "
            "clearly distinguish current-turn evidence from prior-turn evidence, cite the source turn "
            "query when referencing prior findings, and do not treat prior-turn summaries as current-turn results.\n"
            "Be concise. Limit yourself to 3-5 scientific findings. Keep each evidence list to 2-4 short items.\n"
            "Do not wrap the JSON in markdown fences.\n"
            "Interpret output types as follows when evidence is present:\n"
            "- timeseries_result: discuss extrema timing, anomaly episodes, start/end change, variability, bursts, and temporal clustering.\n"
            "- trend_result: discuss sign, magnitude, p-value, confidence, R-squared, and whether the trend is statistically significant.\n"
            "- field_trend_result: discuss where trends are positive/negative, slope hotspots, significance coverage, and whether the trend pattern is spatially coherent.\n"
            "- lag_correlation_result: discuss optimal lag, sign/magnitude of correlation, and whether the relationship appears meaningful.\n"
            "- profile_result: discuss vertical gradients, strongest transition depth, surface-bottom contrast, and depth-localized anomalies.\n"
            "- section_result: discuss along-transect gradients, distance ranges of extrema, depth-localized transitions, and whether structure is surface-intensified or subsurface.\n"
            "- spatial_field_result: discuss hotspots/coldspots, extrema coordinates, gradients, clustering, and whether anomalies are localized or widespread.\n"
            "- hovmoller_result: discuss when and where extrema occur, coherent bands, depth-localized events, and time windows of strongest anomalies.\n"
            "- eof_result: discuss leading modes, variance explained, dominant times of the principal components, and whether one mode dominates.\n"
            "- ts_diagram_result: discuss T-S spread, clustering, salinity-temperature covariation, and whether the cloud suggests multiple water-mass branches.\n"
            "- watermass_event_association_result: discuss whether hotspot tiles favor specific named water masses, how strongly hotspot composition departs from the background tile distribution, and whether the evidence is strong, moderate, weak, or insufficient.\n"
            "- histogram_result / histogram_2d_result: discuss modal structure, spread, skewness, tails, and concentration regions.\n"
            "- regression_map_result: discuss where relationships are positive/negative, magnitude hotspots, correlation strength, and the fraction of the map that is significant.\n"
            "- composite_result: discuss contrast between positive and negative phases, strongest composite differences, and whether anomalies are spatially coherent.\n"
            "- spectrum_result: discuss dominant periods, secondary peaks, spectral concentration, and whether variability is broad-band or peak-dominated.\n"
            "- layer_transport_result: discuss which depth layers dominate the transport, whether signs reverse with depth, and how strongly the layer contributions differ.\n"
            "- event_detection_result: discuss counts, strongest/largest events, timing windows, durations, and spatial concentration.\n"
            "- event_statistics_result: discuss dominant groups, share of total events, group ranking, contrast between the top and second groups, and whether activity shifts from early to late active groups.\n"
            "- event_spatial_distribution_result / event_comparison_result: discuss hotspots, spatial spread, centroid shifts, and contrasts between groups or periods.\n"
            "When group summaries include ordered periods such as months or seasons, explicitly assess whether activity is concentrated, evenly distributed, or shifted toward later/earlier periods.\n"
            "If event group summaries say `group_label_mode` is `index_proxy`, treat labels like `0`, `1`, `2` as ordered bins only. Do not translate them into calendar months or seasons unless explicit month/season names are provided.\n"
            "- climatology_result: discuss the seasonal or annual mean structure, highlight months or regions "
            "of peak/minimum values, assess spatial coherence, and contrast the mean state across sub-regions.\n"
            "For each lag_correlation_result, also produce a machine-readable lag-selection decision in "
            "`lag_selection_overrides`.\n"
            "Selection rules for lag_correlation_result:\n"
            "- Treat `optimal_lag` and `max_correlation` in the input summary as the raw symmetric best lag.\n"
            "- Use `symmetric_optimal_lag`, `best_positive_lag`, `best_negative_lag`, `zero_lag_correlation`, "
            "and `lag_curve` to decide whether directionality is scientifically meaningful for the user's request.\n"
            "- If directionality is meaningful, choose `selected_mode` as either `positive` or `negative`.\n"
            "- If directionality is not meaningful, choose `selected_mode` as `symmetric` and preserve the raw symmetric best lag.\n"
            "- Do not invent a lag that is not already present in the provided candidates or lag curve.\n"
            "- If you mention an official optimal lag in `summary` or `scientific_findings`, align it with your selected lag decision.\n"
            "- Keep the JSON machine-readable and concise.\n"
            "Return JSON only."
            f"{policy_prompt}"
            f"{integrated_prompt}"
            f"{retry_prompt}"
        )

        required_output_schema: Dict[str, Any] = {
            "summary": "string",
            "synthesis_warnings": [
                "optional warnings for unsupported interpretation retained as policy reasoning, indirect reasoning, or data gaps"
            ],
            "scientific_findings": [
                {
                    "finding": "string",
                    "evidence": ["2-4 short strings"],
                    "result_ids": ["optional list of result IDs backing this finding"],
                }
            ],
            "lag_selection_overrides": [
                {
                    "result_id": "string",
                    "has_clear_directionality": "boolean",
                    "selected_mode": "positive | negative | symmetric",
                    "selected_optimal_lag": "number",
                    "selected_max_correlation": "number",
                    "reason": "string",
                }
            ],
        }
        if policy_requested:
            required_output_schema["policy_guidance"] = {
                "should_include": True,
                "headline": "string",
                "place_based_policy_brief": "string",
                "evidence_action_matrix": [
                    {
                        "priority": "high | medium | low | screening",
                        "action_type": (
                            "monitoring | source_control | discharge_outlet | river_estuary | "
                            "seasonal_management | coastal_planning | economic_assessment | governance"
                        ),
                        "target": "string",
                        "where_when": "string",
                        "evidence_basis": "string",
                        "recommendation": "string",
                        "guardrail": "string",
                        "evidence_strength": "supported | limited | screening | not_supported",
                    }
                ],
                "evidence_limits": ["string"],
            }
        if synthesis_profile_id == self.INTEGRATED_ASSESSMENT_PROFILE_ID:
            required_output_schema["integrated_assessment"] = {
                "profile_id": self.INTEGRATED_ASSESSMENT_PROFILE_ID,
                "direct_answer": (
                    "1-2 sentence overall answer; state whether suitability is broadly retained, fragile, "
                    "hotspot/season dependent, or insufficiently evidenced"
                ),
                "assessment_narrative": (
                    "5-8 natural paragraphs; evidence-dense integrated answer ordered as overall judgment, "
                    "main environmental signals, regional/seasonal differences, stabilizing/adaptation "
                    "context, and evidence limits; include concrete anchors such as coordinates, event days, "
                    "burden values, trend statistics, extrema, and timing windows when available; after each "
                    "major data interpretation, add what it implies for the user's question"
                ),
                "evidence_threads": [
                    {
                        "theme": "warming/heatwave | bottom_oxygen/hypoxia | stratification | chlorophyll/bloom | data_gap",
                        "status": "computed | indirect | data_gap",
                        "evidence_summary": (
                            "one natural sentence summarizing computed evidence, indirect support, or missing data"
                        ),
                        "evidence_result_ids": [
                            "required for computed/indirect; may be empty for data_gap"
                        ],
                    }
                ],
                "suitability": "string",
                "risk_hotspots": ["up to 4 strings"],
                "higher_risk_regions": [
                    {
                        "region": "named coastal region or hotspot area",
                        "major_environmental_risks": (
                            "short natural-language risk summary, e.g. hypoxia, heat stress, bloom pressure"
                        ),
                        "evidence": (
                            "one sentence with computed anchors, result IDs, coordinates, event days, burden values, "
                            "or explicit map-review/data-gap language"
                        ),
                        "evidence_result_ids": [
                            "result IDs supporting this row; may be empty only for map-review/data-gap rows"
                        ],
                    }
                ],
                "environmental_drivers": ["up to 5 strings"],
                "economic_implications": "string",
                "environmental_protection_implications": "string",
                "management_guidance": "1-2 sentence executive summary string, <= 80 words",
                "future_outlook": "string",
                "uncertainty_and_data_gaps": ["string"],
                "evidence_boundary_notes": [
                    "optional notes distinguishing computed evidence from indirect reasoning or data gaps"
                ],
                "evidence_result_ids": ["string"],
                "policy_synthesis": {
                    "one_sentence_judgment": (
                        "one direct sentence judging suitability/risk before listing policies; "
                        "must state whether risk is broad or hotspot/season concentrated"
                    ),
                    "policy_narrative": (
                        "200-400 words; policy briefing narrative explaining risk landscape, management logic, "
                        "evidence strength, indirect reasoning, and data gaps without mechanically repeating cards"
                    ),
                    "policy_recommendations": [
                        {
                            "policy_title": "short natural-language title",
                            "recommended_action": "concise policy recommendation written as natural language",
                            "priority_places": ["optional places, coordinates, seasons, or map-review scopes"],
                            "evidence_status": "computed | indirect | data_gap",
                            "evidence_note": (
                                "one sentence: computed evidence anchor, indirect support, or explicit statement "
                                "that more data/monitoring/investigation is needed"
                            ),
                            "evidence_result_ids": [
                                "required for computed/indirect; may be empty for data_gap"
                            ],
                        }
                    ],
                },
            }
            if not isinstance(additional_context.get("policy_context_packet"), dict):
                required_output_schema["integrated_assessment"].pop("policy_synthesis", None)

        user_payload = {
            "user_request": user_request,
            "active_plan": self._compact_plan(active_plan),
            "completed_steps": self._compact_completed_steps(completed_steps),
            "result_summaries": self._compact_result_summaries(result_summaries),
            "additional_context": additional_context,
            "policy_guidance_required": policy_requested,
            "structured_policy_guidance_required": structured_policy_requested,
            "synthesis_profile_id": synthesis_profile_id,
            "required_output_schema": required_output_schema,
        }
        if validation_error:
            user_payload["previous_validation_error"] = validation_error
            user_payload["retry_instruction"] = (
                "Regenerate one complete, valid JSON object only. The previous response was rejected; "
                "fix the exact JSON/schema issue, include commas between all fields/items, close every "
                "array/object, and do not include markdown or any text outside JSON. Keep it shorter than "
                "the previous response: use at most 3 scientific_findings, at most 4 policy rows if requested, "
                "at most 4 risk_hotspots, and at most 5 uncertainty/data-gap items. Keep scientific claims "
                "evidence-backed and make policy rows comply with the schema and evidence rules."
            )

        return {
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
                }
            ],
        }

    def _translate_texts_to_english(self, texts: List[str], *, client: Any, request_name: str) -> List[str]:
        return translate_strings_to_english(
            adapter=self._adapter,
            client=client,
            texts=texts,
            request_name=request_name,
        )

    def _normalize_text_list_to_english(
        self,
        values: Any,
        *,
        client: Any,
        request_name: str,
    ) -> Optional[List[str]]:
        if not isinstance(values, list):
            return None
        normalized_values: List[Optional[str]] = []
        texts_to_translate: List[str] = []
        positions: List[int] = []
        for item in values:
            if not isinstance(item, str):
                normalized_values.append(None)
                continue
            stripped = item.strip()
            if not stripped:
                normalized_values.append(None)
                continue
            normalized_values.append(stripped)
            if needs_english_translation(stripped):
                positions.append(len(normalized_values) - 1)
                texts_to_translate.append(stripped)
        if texts_to_translate:
            try:
                translated = self._translate_texts_to_english(
                    texts_to_translate,
                    client=client,
                    request_name=request_name,
                )
                for position, translated_text in zip(positions, translated):
                    normalized_values[position] = translated_text
            except Exception:
                for position in positions:
                    normalized_values[position] = None
        return [item for item in normalized_values if isinstance(item, str) and item.strip()]

    def _normalize_optional_text_to_english(
        self,
        value: Any,
        *,
        client: Any,
        request_name: str,
    ) -> Optional[str]:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if not needs_english_translation(stripped):
            return stripped
        try:
            return self._translate_texts_to_english(
                [stripped],
                client=client,
                request_name=request_name,
            )[0]
        except Exception:
            return None

    def _normalize_additional_context_to_english(
        self,
        additional_context: Dict[str, Any],
        *,
        client: Any,
    ) -> Dict[str, Any]:
        if not isinstance(additional_context, dict):
            return {}
        normalized = copy.deepcopy(additional_context)
        conversation_context = normalized.get("conversation_context")
        if isinstance(conversation_context, dict):
            recent_queries = self._normalize_text_list_to_english(
                conversation_context.get("recent_queries"),
                client=client,
                request_name="translate_synthesis_recent_queries",
            )
            if recent_queries is not None:
                conversation_context["recent_queries"] = recent_queries
            conclusions = self._normalize_text_list_to_english(
                conversation_context.get("conclusions"),
                client=client,
                request_name="translate_synthesis_conclusions",
            )
            if conclusions is not None:
                conversation_context["conclusions"] = conclusions

        for field_name in ("prior_queries_text", "synthesizer_prior_context_text"):
            normalized_value = self._normalize_optional_text_to_english(
                normalized.get(field_name),
                client=client,
                request_name=f"translate_{field_name}",
            )
            if normalized_value is None:
                normalized.pop(field_name, None)
            else:
                normalized[field_name] = normalized_value
        return normalized

    def _normalize_synthesis_output_to_english(
        self,
        synthesis: Dict[str, Any],
        *,
        client: Any,
    ) -> Dict[str, Any]:
        if not isinstance(synthesis, dict):
            return synthesis
        normalized = copy.deepcopy(synthesis)

        original_summary = normalized.get("summary")
        summary = self._normalize_optional_text_to_english(
            original_summary,
            client=client,
            request_name="translate_synthesis_summary",
        )
        if summary:
            normalized["summary"] = summary
        elif needs_english_translation(original_summary):
            normalized["summary"] = (
                "Analysis completed, but the synthesized summary could not be translated into English."
            )

        normalized.pop("recommended_followups", None)

        synthesis_warnings = self._normalize_text_list_to_english(
            normalized.get("synthesis_warnings"),
            client=client,
            request_name="translate_synthesis_warnings",
        )
        if synthesis_warnings is not None:
            normalized["synthesis_warnings"] = synthesis_warnings

        findings = normalized.get("scientific_findings")
        if isinstance(findings, list):
            next_findings: List[Dict[str, Any]] = []
            for index, item in enumerate(findings):
                if not isinstance(item, dict):
                    continue
                finding_text = self._normalize_optional_text_to_english(
                    item.get("finding"),
                    client=client,
                    request_name=f"translate_synthesis_finding_{index}",
                )
                if not finding_text:
                    continue
                item_copy = dict(item)
                item_copy["finding"] = finding_text
                evidence = self._normalize_text_list_to_english(
                    item.get("evidence"),
                    client=client,
                    request_name=f"translate_synthesis_finding_evidence_{index}",
                )
                if evidence is not None:
                    item_copy["evidence"] = evidence
                next_findings.append(item_copy)
            normalized["scientific_findings"] = next_findings

        lag_selection_overrides = normalized.get("lag_selection_overrides")
        if isinstance(lag_selection_overrides, list):
            next_overrides: List[Dict[str, Any]] = []
            for index, item in enumerate(lag_selection_overrides):
                if not isinstance(item, dict):
                    continue
                item_copy = dict(item)
                reason = self._normalize_optional_text_to_english(
                    item.get("reason"),
                    client=client,
                    request_name=f"translate_synthesis_lag_selection_reason_{index}",
                )
                if reason:
                    item_copy["reason"] = reason
                next_overrides.append(item_copy)
            normalized["lag_selection_overrides"] = next_overrides

        policy_guidance = normalized.get("policy_guidance")
        if isinstance(policy_guidance, dict):
            normalized_policy = dict(policy_guidance)
            headline = self._normalize_optional_text_to_english(
                normalized_policy.get("headline"),
                client=client,
                request_name="translate_policy_guidance_headline",
            )
            if headline:
                normalized_policy["headline"] = headline

            brief = self._normalize_optional_text_to_english(
                normalized_policy.get("place_based_policy_brief"),
                client=client,
                request_name="translate_policy_guidance_place_brief",
            )
            if brief:
                normalized_policy["place_based_policy_brief"] = brief

            limits = self._normalize_text_list_to_english(
                normalized_policy.get("evidence_limits"),
                client=client,
                request_name="translate_policy_guidance_limits",
            )
            if limits is not None:
                normalized_policy["evidence_limits"] = limits

            matrix = normalized_policy.get("evidence_action_matrix")
            if isinstance(matrix, list):
                translated_matrix: List[Dict[str, Any]] = []
                text_fields = (
                    "target",
                    "where_when",
                    "evidence_basis",
                    "recommendation",
                    "guardrail",
                )
                for index, item in enumerate(matrix):
                    if not isinstance(item, dict):
                        continue
                    item_copy = dict(item)
                    for field_name in text_fields:
                        translated_value = self._normalize_optional_text_to_english(
                            item.get(field_name),
                            client=client,
                            request_name=f"translate_policy_guidance_{field_name}_{index}",
                        )
                        if translated_value:
                            item_copy[field_name] = translated_value
                    translated_matrix.append(item_copy)
                normalized_policy["evidence_action_matrix"] = translated_matrix
            normalized["policy_guidance"] = normalized_policy

        integrated = normalized.get("integrated_assessment")
        if isinstance(integrated, dict):
            normalized_integrated = dict(integrated)
            for field_name in (
                "direct_answer",
                "assessment_narrative",
                "suitability",
                "economic_implications",
                "environmental_protection_implications",
                "future_outlook",
            ):
                translated_value = self._normalize_optional_text_to_english(
                    normalized_integrated.get(field_name),
                    client=client,
                    request_name=f"translate_integrated_assessment_{field_name}",
                )
                if translated_value:
                    normalized_integrated[field_name] = translated_value
            for field_name in (
                "risk_hotspots",
                "environmental_drivers",
                "uncertainty_and_data_gaps",
                "evidence_boundary_notes",
            ):
                translated_list = self._normalize_text_list_to_english(
                    normalized_integrated.get(field_name),
                    client=client,
                    request_name=f"translate_integrated_assessment_{field_name}",
                )
                if translated_list is not None:
                    normalized_integrated[field_name] = translated_list

            management_guidance = normalized_integrated.get("management_guidance")
            if isinstance(management_guidance, list):
                management_guidance = " ".join(
                    item.strip() for item in management_guidance if isinstance(item, str) and item.strip()
                )
            translated_management_guidance = self._normalize_optional_text_to_english(
                management_guidance,
                client=client,
                request_name="translate_integrated_assessment_management_guidance",
            )
            if translated_management_guidance:
                normalized_integrated["management_guidance"] = translated_management_guidance

            evidence_threads = normalized_integrated.get("evidence_threads")
            if isinstance(evidence_threads, list):
                translated_threads: List[Dict[str, Any]] = []
                for index, item in enumerate(evidence_threads):
                    if not isinstance(item, dict):
                        continue
                    item_copy = dict(item)
                    translated_summary = self._normalize_optional_text_to_english(
                        item.get("evidence_summary"),
                        client=client,
                        request_name=f"translate_integrated_evidence_thread_summary_{index}",
                    )
                    if translated_summary:
                        item_copy["evidence_summary"] = translated_summary
                    translated_threads.append(item_copy)
                normalized_integrated["evidence_threads"] = translated_threads

            higher_risk_regions = normalized_integrated.get("higher_risk_regions")
            if isinstance(higher_risk_regions, list):
                translated_regions: List[Dict[str, Any]] = []
                for index, item in enumerate(higher_risk_regions):
                    if not isinstance(item, dict):
                        continue
                    item_copy = dict(item)
                    for field_name in ("region", "major_environmental_risks", "evidence"):
                        translated_value = self._normalize_optional_text_to_english(
                            item.get(field_name),
                            client=client,
                            request_name=f"translate_integrated_higher_risk_region_{field_name}_{index}",
                        )
                        if translated_value:
                            item_copy[field_name] = translated_value
                    translated_regions.append(item_copy)
                normalized_integrated["higher_risk_regions"] = translated_regions

            policy_synthesis = normalized_integrated.get("policy_synthesis")
            if isinstance(policy_synthesis, dict):
                normalized_policy_synthesis = dict(policy_synthesis)
                one_sentence_judgment = self._normalize_optional_text_to_english(
                    normalized_policy_synthesis.get("one_sentence_judgment"),
                    client=client,
                    request_name="translate_integrated_policy_synthesis_judgment",
                )
                if one_sentence_judgment:
                    normalized_policy_synthesis["one_sentence_judgment"] = one_sentence_judgment
                policy_narrative = self._normalize_optional_text_to_english(
                    normalized_policy_synthesis.get("policy_narrative"),
                    client=client,
                    request_name="translate_integrated_policy_synthesis_narrative",
                )
                if policy_narrative:
                    normalized_policy_synthesis["policy_narrative"] = policy_narrative
                recommendations = normalized_policy_synthesis.get("policy_recommendations")
                if isinstance(recommendations, list):
                    translated_recommendations: List[Dict[str, Any]] = []
                    for index, item in enumerate(recommendations):
                        if not isinstance(item, dict):
                            continue
                        item_copy = dict(item)
                        for field_name in (
                            "policy_title",
                            "recommended_action",
                            "evidence_note",
                        ):
                            translated_value = self._normalize_optional_text_to_english(
                                item.get(field_name),
                                client=client,
                                request_name=f"translate_integrated_policy_recommendation_{field_name}_{index}",
                            )
                            if translated_value:
                                item_copy[field_name] = translated_value
                        for field_name in ("priority_places", "supporting_evidence"):
                            translated_list = self._normalize_text_list_to_english(
                                item.get(field_name),
                                client=client,
                                request_name=f"translate_integrated_policy_recommendation_{field_name}_{index}",
                            )
                            if translated_list is not None:
                                item_copy[field_name] = translated_list
                        translated_recommendations.append(item_copy)
                    normalized_policy_synthesis["policy_recommendations"] = translated_recommendations
                normalized_integrated["policy_synthesis"] = normalized_policy_synthesis
            normalized["integrated_assessment"] = normalized_integrated
        return normalized

    def _repair_synthesis_before_validation(
        self,
        synthesis: Dict[str, Any],
        *,
        result_summaries: Dict[str, Dict[str, Any]],
        additional_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Annotate evidence-boundary issues before structural validation.

        Unsupported interpretation should not discard an otherwise useful final
        answer. Keep the generated text, but add warnings/notes that tell the UI
        and reader which claims need more data.
        """
        if not isinstance(synthesis, dict):
            return synthesis
        integrated = synthesis.get("integrated_assessment")
        if not isinstance(integrated, dict):
            return synthesis
        repaired_synthesis = synthesis
        repaired_integrated = integrated

        def ensure_repaired_integrated() -> Dict[str, Any]:
            nonlocal repaired_synthesis, repaired_integrated
            if repaired_synthesis is synthesis:
                repaired_synthesis = dict(synthesis)
            if repaired_integrated is integrated:
                repaired_integrated = dict(integrated)
                repaired_synthesis["integrated_assessment"] = repaired_integrated
            return repaired_integrated

        def warn(message: str) -> None:
            nonlocal repaired_synthesis, repaired_integrated
            if repaired_synthesis is synthesis:
                repaired_synthesis = dict(synthesis)
            self._add_synthesis_warning(repaired_synthesis, message)
            repaired_integrated = ensure_repaired_integrated()
            self._add_integrated_boundary_note(repaired_integrated, message)

        has_economic_data = self._context_contains_economic_data(
            result_summaries=result_summaries,
            additional_context=additional_context,
        )
        has_source_evidence = self._context_contains_source_evidence(
            result_summaries=result_summaries,
            additional_context=additional_context,
        )

        economic_text_parts = []
        for field_name in ("direct_answer", "assessment_narrative", "economic_implications", "management_guidance"):
            value = integrated.get(field_name)
            if isinstance(value, str):
                economic_text_parts.append(value)
        policy_synthesis = integrated.get("policy_synthesis")
        if isinstance(policy_synthesis, dict):
            for field_name in ("one_sentence_judgment", "policy_narrative"):
                value = policy_synthesis.get(field_name)
                if isinstance(value, str):
                    economic_text_parts.append(value)
            recommendations = policy_synthesis.get("policy_recommendations")
            if isinstance(recommendations, list):
                economic_text_parts.extend(
                    self._policy_recommendation_text(item)
                    for item in recommendations
                    if isinstance(item, dict)
                )
        if not has_economic_data and self._text_mentions_economic_claim(" ".join(economic_text_parts)):
            warn(
                "Economic or industry interpretation is not directly supported by production, revenue, cost, "
                "or accounting data and should be treated as contextual reasoning or a data gap."
            )

        if not has_source_evidence:
            policy_synthesis = integrated.get("policy_synthesis")
            if isinstance(policy_synthesis, dict):
                source_policy_found = False
                recommendations = policy_synthesis.get("policy_recommendations")
                if isinstance(recommendations, list):
                    for item in recommendations:
                        if not isinstance(item, dict):
                            continue
                        text = self._policy_recommendation_text(item)
                        if self._text_mentions_source_policy(text) and (
                            self._text_mentions_direct_source_control(text)
                            or not self._source_policy_is_screening_or_investigation(text)
                        ):
                            source_policy_found = True

                rows = policy_synthesis.get("decision_rows")
                if isinstance(rows, list):
                    for item in rows:
                        if not isinstance(item, dict):
                            continue
                        text = self._policy_decision_row_text(item)
                        if (
                            self._text_mentions_source_policy(text)
                            and (
                                str(item.get("confidence") or "").strip() == "supported"
                                or not self._source_policy_is_screening_or_investigation(text)
                            )
                        ):
                            source_policy_found = True
                if source_policy_found:
                    warn(
                        "Source, nutrient, pollution, or land-sea pathway interpretation is not directly supported "
                        "by nutrient, river-input, discharge, or source-inventory data and should be treated as "
                        "screening or a hypothesis needing more data."
                    )

        return repaired_synthesis

    @staticmethod
    def _add_synthesis_warning(synthesis: Dict[str, Any], message: str) -> None:
        warnings = synthesis.get("synthesis_warnings")
        if not isinstance(warnings, list):
            warnings = []
        if message not in warnings:
            warnings.append(message)
        synthesis["synthesis_warnings"] = warnings

    @staticmethod
    def _add_integrated_boundary_note(integrated: Dict[str, Any], message: str) -> None:
        notes = integrated.get("evidence_boundary_notes")
        if not isinstance(notes, list):
            notes = []
        if message not in notes:
            notes.append(message)
        integrated["evidence_boundary_notes"] = notes

    def _warn_synthesis_evidence_boundary(
        self,
        synthesis: Dict[str, Any],
        integrated: Optional[Dict[str, Any]],
        message: str,
    ) -> None:
        if not isinstance(synthesis, dict) or not isinstance(message, str) or not message.strip():
            return
        self._add_synthesis_warning(synthesis, message.strip())
        if isinstance(integrated, dict):
            self._add_integrated_boundary_note(integrated, message.strip())

    @staticmethod
    def _policy_recommendation_text(item: Dict[str, Any]) -> str:
        parts: List[str] = []
        for field_name in (
            "policy_title",
            "recommended_action",
            "evidence_status",
            "evidence_note",
            "why_this_policy",
            "guardrail",
        ):
            value = item.get(field_name)
            if isinstance(value, str):
                parts.append(value)
        for field_name in ("priority_places", "supporting_evidence", "evidence_result_ids"):
            value = item.get(field_name)
            if isinstance(value, list):
                parts.extend(str(entry) for entry in value if isinstance(entry, str))
        return " ".join(parts).lower()

    @staticmethod
    def _policy_decision_row_text(item: Dict[str, Any]) -> str:
        return " ".join(
            str(item.get(field_name, ""))
            for field_name in (
                "decision_unit",
                "action_group",
                "target",
                "policy_lever",
                "where_when",
                "evidence_basis",
                "trigger_evidence",
                "recommended_action",
                "guardrail",
                "rationale",
                "confidence",
            )
        ).lower()

    @staticmethod
    def _repair_source_policy_recommendation(item: Dict[str, Any]) -> Dict[str, Any]:
        repaired = dict(item)
        repaired["policy_title"] = "Source-pathway screening"
        repaired["recommended_action"] = (
            "Conduct source-pathway screening and nutrient/source investigation around the evidence-linked "
            "oxygen hotspot before considering source-control measures."
        )
        repaired["evidence_status"] = "data_gap"
        repaired["evidence_note"] = (
            "Oxygen stress supports source-pathway screening as a data gap, but nutrient, river-input, discharge, "
            "and source-inventory data are needed before attributing causes or prescribing source controls."
        )
        repaired["evidence_result_ids"] = []
        repaired.pop("why_this_policy", None)
        repaired.pop("guardrail", None)
        return repaired

    @staticmethod
    def _repair_source_policy_decision_row(item: Dict[str, Any]) -> Dict[str, Any]:
        repaired = dict(item)
        repaired["action_group"] = "source_pathway_screening"
        repaired["policy_lever"] = "source-pathway screening"
        repaired["recommended_action"] = (
            "Conduct source-pathway screening and nutrient/source investigation around the evidence-linked "
            "oxygen hotspot before considering source-control measures."
        )
        repaired["guardrail"] = (
            "Use endpoint oxygen evidence only to justify screening/investigation; do not attribute causes or "
            "prescribe source controls without direct source-pathway evidence."
        )
        repaired["rationale"] = (
            "The row is downgraded to screening because the completed results identify oxygen stress but do not "
            "include direct nutrient, river-input, discharge, or source-inventory evidence."
        )
        repaired["confidence"] = "screening"
        return repaired

    def _compact_synthesis_output(
        self,
        synthesis: Dict[str, Any],
        *,
        policy_requested: bool,
        synthesis_profile_id: Optional[str],
    ) -> Dict[str, Any]:
        if not isinstance(synthesis, dict):
            return synthesis
        compact = copy.deepcopy(synthesis)
        if isinstance(compact.get("summary"), str):
            compact["summary"] = self._trim_words(compact["summary"], 160)

        synthesis_warnings = compact.get("synthesis_warnings")
        if isinstance(synthesis_warnings, list):
            compact["synthesis_warnings"] = [
                self._trim_words(item, 36)
                for item in synthesis_warnings[:6]
                if isinstance(item, str) and item.strip()
            ]

        findings = compact.get("scientific_findings")
        if isinstance(findings, list):
            next_findings: List[Dict[str, Any]] = []
            for item in findings[:3]:
                if not isinstance(item, dict):
                    continue
                item_copy = dict(item)
                if isinstance(item_copy.get("finding"), str):
                    item_copy["finding"] = self._trim_words(item_copy["finding"], 60)
                evidence = item_copy.get("evidence")
                if isinstance(evidence, list):
                    item_copy["evidence"] = [
                        self._trim_words(entry, 28)
                        for entry in evidence[:3]
                        if isinstance(entry, str) and entry.strip()
                    ]
                result_ids = item_copy.get("result_ids")
                if isinstance(result_ids, list):
                    item_copy["result_ids"] = [
                        entry for entry in result_ids[:4] if isinstance(entry, str) and entry.strip()
                    ]
                next_findings.append(item_copy)
            compact["scientific_findings"] = next_findings

        if policy_requested:
            policy_guidance = compact.get("policy_guidance")
            if isinstance(policy_guidance, dict):
                self._compact_policy_guidance(policy_guidance)

        if synthesis_profile_id == self.INTEGRATED_ASSESSMENT_PROFILE_ID:
            integrated = compact.get("integrated_assessment")
            if isinstance(integrated, dict):
                self._compact_integrated_assessment(integrated)
        return compact

    def _compact_policy_guidance(self, policy_guidance: Dict[str, Any]) -> None:
        for field_name, limit in (
            ("headline", 24),
            ("place_based_policy_brief", 40),
        ):
            if isinstance(policy_guidance.get(field_name), str):
                policy_guidance[field_name] = self._trim_words(policy_guidance[field_name], limit)
        matrix = policy_guidance.get("evidence_action_matrix")
        if isinstance(matrix, list):
            next_rows: List[Dict[str, Any]] = []
            for item in matrix[:4]:
                if not isinstance(item, dict):
                    continue
                item_copy = dict(item)
                for field_name in ("target", "where_when", "evidence_basis", "recommendation", "guardrail"):
                    if isinstance(item_copy.get(field_name), str):
                        item_copy[field_name] = self._trim_words(item_copy[field_name], 32)
                next_rows.append(item_copy)
            policy_guidance["evidence_action_matrix"] = next_rows
        evidence_limits = policy_guidance.get("evidence_limits")
        if isinstance(evidence_limits, list):
            policy_guidance["evidence_limits"] = [
                self._trim_words(item, 28)
                for item in evidence_limits[:4]
                if isinstance(item, str) and item.strip()
            ]

    def _compact_integrated_assessment(self, integrated: Dict[str, Any]) -> None:
        if isinstance(integrated.get("direct_answer"), str):
            integrated["direct_answer"] = self._trim_words(integrated["direct_answer"], 70)
        if isinstance(integrated.get("assessment_narrative"), str):
            integrated["assessment_narrative"] = self._trim_words(integrated["assessment_narrative"], 760)
        for field_name, limit in (
            ("suitability", 45),
            ("economic_implications", 45),
            ("environmental_protection_implications", 45),
            ("future_outlook", 45),
        ):
            if isinstance(integrated.get(field_name), str):
                integrated[field_name] = self._trim_words(integrated[field_name], limit)
        for field_name, max_items, limit in (
            ("risk_hotspots", 4, 32),
            ("environmental_drivers", 5, 32),
            ("uncertainty_and_data_gaps", 5, 32),
            ("evidence_boundary_notes", 6, 36),
            ("evidence_result_ids", 8, 18),
        ):
            values = integrated.get(field_name)
            if isinstance(values, list):
                integrated[field_name] = [
                    self._trim_words(item, limit)
                    for item in values[:max_items]
                    if isinstance(item, str) and item.strip()
                ]
        management_guidance = integrated.get("management_guidance")
        if isinstance(management_guidance, list):
            management_guidance = " ".join(
                item.strip() for item in management_guidance if isinstance(item, str) and item.strip()
            )
        if isinstance(management_guidance, str):
            integrated["management_guidance"] = self._trim_words(management_guidance, 80)
        evidence_threads = integrated.get("evidence_threads")
        if isinstance(evidence_threads, list):
            next_threads: List[Dict[str, Any]] = []
            for item in evidence_threads[:6]:
                if not isinstance(item, dict):
                    continue
                item_copy = dict(item)
                if isinstance(item_copy.get("evidence_summary"), str):
                    item_copy["evidence_summary"] = self._trim_words(item_copy["evidence_summary"], 65)
                evidence_ids = item_copy.get("evidence_result_ids")
                if isinstance(evidence_ids, list):
                    item_copy["evidence_result_ids"] = [
                        entry for entry in evidence_ids[:4] if isinstance(entry, str) and entry.strip()
                    ]
                next_threads.append(item_copy)
            integrated["evidence_threads"] = next_threads
        higher_risk_regions = integrated.get("higher_risk_regions")
        if isinstance(higher_risk_regions, list):
            next_regions: List[Dict[str, Any]] = []
            for item in higher_risk_regions[:8]:
                if not isinstance(item, dict):
                    continue
                item_copy = dict(item)
                for field_name, limit in (
                    ("region", 12),
                    ("major_environmental_risks", 22),
                    ("evidence", 42),
                ):
                    if isinstance(item_copy.get(field_name), str):
                        item_copy[field_name] = self._trim_words(item_copy[field_name], limit)
                evidence_ids = item_copy.get("evidence_result_ids")
                if isinstance(evidence_ids, list):
                    item_copy["evidence_result_ids"] = [
                        self._trim_words(entry, 18)
                        for entry in evidence_ids[:4]
                        if isinstance(entry, str) and entry.strip()
                    ]
                next_regions.append(item_copy)
            integrated["higher_risk_regions"] = next_regions
        integrated.pop("policy_design_framework", None)
        policy_synthesis = integrated.get("policy_synthesis")
        if isinstance(policy_synthesis, dict):
            policy_synthesis.pop("policy_frame", None)
            policy_synthesis.pop("decision_rows", None)
            if isinstance(policy_synthesis.get("one_sentence_judgment"), str):
                policy_synthesis["one_sentence_judgment"] = self._trim_words(
                    policy_synthesis["one_sentence_judgment"],
                    55,
                )
            if isinstance(policy_synthesis.get("policy_narrative"), str):
                policy_synthesis["policy_narrative"] = self._trim_words(
                    policy_synthesis["policy_narrative"],
                    400,
                )
            recommendations = policy_synthesis.get("policy_recommendations")
            if isinstance(recommendations, list):
                next_recommendations: List[Dict[str, Any]] = []
                for item in recommendations[:4]:
                    if not isinstance(item, dict):
                        continue
                    item_copy = dict(item)
                    for field_name, limit in (
                        ("policy_title", 14),
                        ("recommended_action", 70),
                        ("evidence_note", 75),
                    ):
                        if isinstance(item_copy.get(field_name), str):
                            item_copy[field_name] = self._trim_words(item_copy[field_name], limit)
                    values = item_copy.get("priority_places")
                    if isinstance(values, list):
                        item_copy["priority_places"] = [
                            self._trim_words(entry, 35)
                            for entry in values[:4]
                            if isinstance(entry, str) and entry.strip()
                        ]
                    item_copy.pop("supporting_evidence", None)
                    item_copy.pop("why_this_policy", None)
                    item_copy.pop("guardrail", None)
                    evidence_ids = item_copy.get("evidence_result_ids")
                    if isinstance(evidence_ids, list):
                        item_copy["evidence_result_ids"] = [
                            entry for entry in evidence_ids[:3] if isinstance(entry, str) and entry.strip()
                        ]
                    next_recommendations.append(item_copy)
                policy_synthesis["policy_recommendations"] = next_recommendations

    def _select_diverse_policy_synthesis_rows(
        self,
        rows: List[Any],
        *,
        max_items: int,
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        selected_indices: Set[int] = set()
        seen_groups: Set[str] = set()

        for index, item in enumerate(rows):
            if not isinstance(item, dict):
                continue
            action_group = str(item.get("action_group") or "").strip()
            if not action_group or action_group not in self.POLICY_SYNTHESIS_ACTION_GROUPS:
                continue
            if action_group in seen_groups:
                continue
            selected.append(item)
            selected_indices.add(index)
            seen_groups.add(action_group)
            if len(selected) >= max_items:
                return selected

        for index, item in enumerate(rows):
            if index in selected_indices or not isinstance(item, dict):
                continue
            selected.append(item)
            if len(selected) >= max_items:
                break
        return selected

    @staticmethod
    def _trim_words(value: str, max_words: int) -> str:
        if not isinstance(value, str):
            return value
        words = value.strip().split()
        if len(words) <= max_words:
            return value.strip()
        return " ".join(words[:max_words]).rstrip(" ,;:") + "..."

    def _validate_synthesis_shape(
        self,
        synthesis: Dict[str, Any],
        *,
        policy_requested: bool = False,
        structured_policy_requested: Optional[bool] = None,
        synthesis_profile_id: Optional[str] = None,
        user_request: str = "",
        result_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if structured_policy_requested is None:
            structured_policy_requested = policy_requested
        if not isinstance(synthesis, dict):
            raise ValueError("Synthesis result must be a JSON object.")

        if not isinstance(synthesis.get("summary"), str) or not synthesis["summary"].strip():
            raise ValueError("Synthesis result must contain a non-empty 'summary'.")

        scientific_findings = synthesis.get("scientific_findings")
        if scientific_findings is not None:
            if not isinstance(scientific_findings, list):
                raise ValueError("'scientific_findings' must be a list when provided.")
            for item in scientific_findings:
                if not isinstance(item, dict):
                    raise ValueError("Each scientific_finding must be an object.")
                result_ids = item.get("result_ids")
                if result_ids is not None and not (
                    isinstance(result_ids, list) and all(isinstance(entry, str) for entry in result_ids)
                ):
                    raise ValueError("Each scientific_finding.result_ids must be a list of strings when provided.")

        key_findings = synthesis.get("key_findings")
        if key_findings is not None:
            if not (isinstance(key_findings, list) and all(isinstance(item, str) for item in key_findings)):
                raise ValueError("'key_findings' must be a list of strings when provided.")

        notable_patterns = synthesis.get("notable_patterns")
        if notable_patterns is not None and not (
            isinstance(notable_patterns, list) and all(isinstance(item, str) for item in notable_patterns)
        ):
            raise ValueError("'notable_patterns' must be a list of strings when provided.")

        anomalies = synthesis.get("anomalies")
        if anomalies is not None:
            if not isinstance(anomalies, list):
                raise ValueError("'anomalies' must be a list when provided.")
            for item in anomalies:
                if not isinstance(item, dict):
                    raise ValueError("Each anomaly must be an object.")

        significance_assessment = synthesis.get("significance_assessment")
        if significance_assessment is not None:
            if not isinstance(significance_assessment, list):
                raise ValueError("'significance_assessment' must be a list when provided.")
            for item in significance_assessment:
                if not isinstance(item, dict):
                    raise ValueError("Each significance_assessment entry must be an object.")

        uncertainties = synthesis.get("uncertainties")
        if uncertainties is not None and not (
            isinstance(uncertainties, list) and all(isinstance(item, str) for item in uncertainties)
        ):
            raise ValueError("'uncertainties' must be a list of strings when provided.")

        recommended_followups = synthesis.get("recommended_followups")
        if recommended_followups is not None and not (
            isinstance(recommended_followups, list) and all(isinstance(item, str) for item in recommended_followups)
        ):
            raise ValueError("'recommended_followups' must be a list of strings when provided.")

        synthesis_warnings = synthesis.get("synthesis_warnings")
        if synthesis_warnings is not None and not (
            isinstance(synthesis_warnings, list) and all(isinstance(item, str) for item in synthesis_warnings)
        ):
            raise ValueError("'synthesis_warnings' must be a list of strings when provided.")

        lag_selection_overrides = synthesis.get("lag_selection_overrides")
        if lag_selection_overrides is not None:
            if not isinstance(lag_selection_overrides, list):
                raise ValueError("'lag_selection_overrides' must be a list when provided.")
            for item in lag_selection_overrides:
                if not isinstance(item, dict):
                    raise ValueError("Each lag_selection_override must be an object.")
                result_id = item.get("result_id")
                if not isinstance(result_id, str) or not result_id.strip():
                    raise ValueError("Each lag_selection_override must include a non-empty string result_id.")
                if not isinstance(item.get("has_clear_directionality"), bool):
                    raise ValueError("Each lag_selection_override must include boolean has_clear_directionality.")
                selected_mode = item.get("selected_mode")
                if selected_mode not in {"positive", "negative", "symmetric"}:
                    raise ValueError(
                        "Each lag_selection_override.selected_mode must be one of positive, negative, symmetric."
                    )
                if not isinstance(item.get("selected_optimal_lag"), (int, float)):
                    raise ValueError("Each lag_selection_override.selected_optimal_lag must be numeric.")
                if not isinstance(item.get("selected_max_correlation"), (int, float)):
                    raise ValueError("Each lag_selection_override.selected_max_correlation must be numeric.")
                reason = item.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("Each lag_selection_override.reason must be a non-empty string.")

        ui_actions = synthesis.get("ui_actions")
        if ui_actions is not None:
            if not isinstance(ui_actions, list):
                raise ValueError("'ui_actions' must be a list when provided.")
            for action in ui_actions:
                if not isinstance(action, dict):
                    raise ValueError("Each ui_action must be an object.")

        self._validate_policy_guidance(
            synthesis,
            policy_requested=policy_requested,
            structured_policy_requested=structured_policy_requested,
            user_request=user_request,
            result_summaries=result_summaries or {},
            additional_context=additional_context or {},
        )
        self._validate_integrated_assessment(
            synthesis,
            synthesis_profile_id=synthesis_profile_id,
            user_request=user_request,
            result_summaries=result_summaries or {},
            additional_context=additional_context or {},
        )

    def _validate_integrated_assessment(
        self,
        synthesis: Dict[str, Any],
        *,
        synthesis_profile_id: Optional[str],
        user_request: str,
        result_summaries: Dict[str, Dict[str, Any]],
        additional_context: Dict[str, Any],
    ) -> None:
        if synthesis_profile_id != self.INTEGRATED_ASSESSMENT_PROFILE_ID:
            return

        integrated = synthesis.get("integrated_assessment")
        if not isinstance(integrated, dict):
            raise ValueError("Integrated assessment profile requires an 'integrated_assessment' object.")

        profile_id = integrated.get("profile_id")
        if profile_id not in {None, self.INTEGRATED_ASSESSMENT_PROFILE_ID}:
            raise ValueError("'integrated_assessment.profile_id' must be ocean_integrated_assessment when provided.")

        evidence_text = self._context_and_summary_text(
            result_summaries=result_summaries,
            additional_context=additional_context,
        )
        self._validate_integrated_main_answer(
            synthesis,
            integrated,
            user_request=user_request,
            evidence_text=evidence_text,
            result_summaries=result_summaries,
            additional_context=additional_context,
        )

        self._require_non_empty_string(
            integrated.get("suitability"),
            "'integrated_assessment.suitability'",
        )

        risk_hotspots = integrated.get("risk_hotspots")
        self._validate_required_str_list(
            risk_hotspots,
            "'integrated_assessment.risk_hotspots'",
        )

        has_oxygen_or_hypoxia = self._context_contains_oxygen_or_hypoxia_evidence(
            result_summaries=result_summaries,
            additional_context=additional_context,
        )
        if not has_oxygen_or_hypoxia and self._risk_hotspots_make_specific_ranking(risk_hotspots):
            raise ValueError(
                "Integrated risk hotspot ranking requires bottom oxygen or hypoxia evidence; "
                "otherwise mark hotspot evidence as insufficient."
            )
        self._validate_integrated_higher_risk_regions(
            integrated.get("higher_risk_regions"),
        )

        self._validate_required_str_list(
            integrated.get("environmental_drivers"),
            "'integrated_assessment.environmental_drivers'",
        )
        self._require_non_empty_string(
            integrated.get("economic_implications"),
            "'integrated_assessment.economic_implications'",
        )
        self._require_non_empty_string(
            integrated.get("environmental_protection_implications"),
            "'integrated_assessment.environmental_protection_implications'",
        )
        management_guidance_value = integrated.get("management_guidance")
        if isinstance(management_guidance_value, list):
            management_guidance_value = " ".join(
                item.strip() for item in management_guidance_value if isinstance(item, str) and item.strip()
            )
            integrated["management_guidance"] = management_guidance_value
        self._require_non_empty_string(
            management_guidance_value,
            "'integrated_assessment.management_guidance'",
        )
        self._require_non_empty_string(
            integrated.get("future_outlook"),
            "'integrated_assessment.future_outlook'",
        )
        self._validate_required_str_list(
            integrated.get("uncertainty_and_data_gaps"),
            "'integrated_assessment.uncertainty_and_data_gaps'",
        )
        self._validate_required_str_list(
            integrated.get("evidence_result_ids"),
            "'integrated_assessment.evidence_result_ids'",
        )
        evidence_boundary_notes = integrated.get("evidence_boundary_notes")
        if evidence_boundary_notes is not None and not (
            isinstance(evidence_boundary_notes, list)
            and all(isinstance(item, str) for item in evidence_boundary_notes)
        ):
            raise ValueError("'integrated_assessment.evidence_boundary_notes' must be a list of strings when provided.")
        if isinstance(management_guidance_value, str):
            self._validate_no_unsupported_policy_thresholds(
                management_guidance_value,
                evidence_text=evidence_text,
                field_name="'integrated_assessment.management_guidance'",
            )
        self._validate_integrated_policy_synthesis(
            synthesis,
            integrated,
            integrated.get("policy_synthesis"),
            user_request=user_request,
            result_summaries=result_summaries,
            additional_context=additional_context,
        )

        combined_text = self._jsonish_text(integrated)
        if (
            self._future_or_projection_requested(user_request)
            and not self.FUTURE_EVIDENCE_RE.search(evidence_text)
            and not self.HISTORICAL_INFERENCE_RE.search(str(integrated.get("future_outlook") or ""))
        ):
            self._warn_synthesis_evidence_boundary(
                synthesis,
                integrated,
                "Future-oriented interpretation is not directly supported by projection or scenario data and should "
                "be read as historical-trend-based inference."
            )

        if (
            not self._context_contains_economic_data(result_summaries, additional_context)
            and self._text_mentions_economic_claim(str(integrated.get("economic_implications") or ""))
        ):
            self._warn_synthesis_evidence_boundary(
                synthesis,
                integrated,
                "Economic or industry interpretation is not directly supported by production, revenue, cost, "
                "or accounting data and should be treated as contextual reasoning or a data gap."
            )

    def _validate_integrated_main_answer(
        self,
        synthesis: Dict[str, Any],
        integrated: Dict[str, Any],
        *,
        user_request: str,
        evidence_text: str,
        result_summaries: Dict[str, Dict[str, Any]],
        additional_context: Dict[str, Any],
    ) -> None:
        self._require_non_empty_string(
            integrated.get("direct_answer"),
            "'integrated_assessment.direct_answer'",
        )
        self._require_non_empty_string(
            integrated.get("assessment_narrative"),
            "'integrated_assessment.assessment_narrative'",
        )
        main_text = " ".join(
            str(integrated.get(field_name) or "")
            for field_name in ("direct_answer", "assessment_narrative")
        )
        for field_name in ("direct_answer", "assessment_narrative"):
            value = integrated.get(field_name)
            if isinstance(value, str):
                self._validate_no_unsupported_policy_thresholds(
                    value,
                    evidence_text=evidence_text,
                    field_name=f"'integrated_assessment.{field_name}'",
                )
        has_economic_data = self._context_contains_economic_data(result_summaries, additional_context)
        if not has_economic_data and self._text_mentions_economic_claim(main_text):
            self._warn_synthesis_evidence_boundary(
                synthesis,
                integrated,
                "Economic or industry wording in the main assessment is not directly supported by production, "
                "revenue, cost, or accounting data and should be treated as contextual reasoning or a data gap."
            )
        has_source_evidence = self._context_contains_source_evidence(result_summaries, additional_context)
        if not has_source_evidence and self._text_mentions_source_policy(main_text):
            self._warn_synthesis_evidence_boundary(
                synthesis,
                integrated,
                "Source, nutrient, pollution, or land-sea pathway interpretation in the main assessment is not "
                "directly supported by nutrient, river-input, discharge, or source-inventory data and should be "
                "treated as contextual reasoning or a data gap."
            )
        if self._text_mentions_unsupported_external_fact(main_text, evidence_text):
            self._warn_synthesis_evidence_boundary(
                synthesis,
                integrated,
                "External background or literature-style claims in the main assessment were not produced by the "
                "completed tools and should be treated as context needing separate evidence."
            )
        if self.INTEGRATED_ASSESSMENT_INTENT_RE.search(user_request):
            evidence_lower = evidence_text.lower()
            if (
                re.search(r"\b(hypoxia|hypoxic|bottom oxygen|dissolved oxygen|oxygen_deficit)\b", evidence_lower)
                and not self.OXYGEN_MECHANISM_RE.search(main_text)
            ):
                self._warn_synthesis_evidence_boundary(
                    synthesis,
                    integrated,
                    "Oxygen or hypoxia evidence is present, but the narrative gives limited mechanism explanation "
                    "for aquaculture or marine-ranching suitability."
                )
            if (
                re.search(r"\b(heatwave|sst|temperature|thermal|warming)\b", evidence_lower)
                and not self.THERMAL_MECHANISM_RE.search(main_text)
            ):
                self._warn_synthesis_evidence_boundary(
                    synthesis,
                    integrated,
                    "Warming or heatwave evidence is present, but the narrative gives limited thermal-stress "
                    "mechanism explanation."
                )
            if (
                re.search(r"\b(chlorophyll|bloom|algal|eutroph)\b", evidence_lower)
                and not self.BLOOM_MECHANISM_RE.search(main_text)
            ):
                self._warn_synthesis_evidence_boundary(
                    synthesis,
                    integrated,
                    "Chlorophyll or bloom evidence is present, but the narrative gives limited ecological-pressure "
                    "mechanism explanation."
                )
        if (
            self._future_or_projection_requested(user_request)
            and not self.FUTURE_EVIDENCE_RE.search(evidence_text)
            and not self.HISTORICAL_INFERENCE_RE.search(main_text)
        ):
            self._warn_synthesis_evidence_boundary(
                synthesis,
                integrated,
                "Future-oriented wording in the main assessment is not directly supported by projection or scenario "
                "data and should be read as historical-trend-based inference."
            )
        self._validate_integrated_evidence_threads(
            integrated.get("evidence_threads"),
            additional_context=additional_context,
        )

    def _validate_integrated_higher_risk_regions(self, rows: Any) -> None:
        if rows is None:
            return
        if not isinstance(rows, list):
            raise ValueError("'integrated_assessment.higher_risk_regions' must be a list when provided.")
        for index, item in enumerate(rows):
            if not isinstance(item, dict):
                raise ValueError("Each integrated higher-risk region row must be an object.")
            missing = sorted({"region", "major_environmental_risks", "evidence"} - set(item))
            if missing:
                raise ValueError(
                    f"Integrated higher-risk region row {index} is missing required field(s): {', '.join(missing)}."
                )
            for field_name in ("region", "major_environmental_risks", "evidence"):
                value = item.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Integrated higher-risk region row {index}.{field_name} must be a non-empty string."
                    )
            evidence_ids = item.get("evidence_result_ids", [])
            if evidence_ids is None:
                evidence_ids = []
            if not (
                isinstance(evidence_ids, list)
                and all(isinstance(entry, str) and entry.strip() for entry in evidence_ids)
            ):
                raise ValueError(
                    f"Integrated higher-risk region row {index}.evidence_result_ids must be a list of strings."
                )

    def _validate_integrated_evidence_threads(
        self,
        evidence_threads: Any,
        *,
        additional_context: Dict[str, Any],
    ) -> None:
        if not isinstance(evidence_threads, list) or not evidence_threads:
            raise ValueError("'integrated_assessment.evidence_threads' must be a non-empty list.")
        has_computed_thread = False
        for index, item in enumerate(evidence_threads):
            if not isinstance(item, dict):
                raise ValueError("Each integrated evidence thread must be an object.")
            missing = sorted({"theme", "status", "evidence_summary", "evidence_result_ids"} - set(item))
            if missing:
                raise ValueError(
                    f"Integrated evidence thread {index} is missing required field(s): {', '.join(missing)}."
                )
            theme = item.get("theme")
            if theme not in self.ASSESSMENT_THREAD_THEMES:
                raise ValueError(
                    f"Integrated evidence thread {index}.theme must be one of "
                    "warming/heatwave, bottom_oxygen/hypoxia, stratification, chlorophyll/bloom, or data_gap."
                )
            status = item.get("status")
            if status not in self.ASSESSMENT_THREAD_STATUS:
                raise ValueError(
                    f"Integrated evidence thread {index}.status must be one of computed, indirect, or data_gap."
                )
            summary = item.get("evidence_summary")
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError(f"Integrated evidence thread {index}.evidence_summary must be a non-empty string.")
            evidence_ids = item.get("evidence_result_ids")
            if evidence_ids is None:
                evidence_ids = []
            if not (
                isinstance(evidence_ids, list)
                and all(isinstance(entry, str) and entry.strip() for entry in evidence_ids)
            ):
                raise ValueError(f"Integrated evidence thread {index}.evidence_result_ids must be a list of strings.")
            if status in {"computed", "indirect"} and not evidence_ids:
                raise ValueError(
                    f"Integrated evidence thread {index}.evidence_result_ids must be non-empty for status='{status}'."
                )
            if status == "computed":
                has_computed_thread = True
            if status == "data_gap" and not self.DATA_GAP_EVIDENCE_NOTE_RE.search(summary):
                raise ValueError(
                    f"Integrated evidence thread {index}.evidence_summary must explicitly mark missing data, "
                    "monitoring, validation, screening, or investigation needs for data_gap status."
                )
        context_packet = additional_context.get("assessment_context_packet")
        context_threads = context_packet.get("evidence_threads") if isinstance(context_packet, dict) else []
        requires_computed_thread = any(
            isinstance(thread, dict) and thread.get("status") == "computed"
            for thread in context_threads
        )
        if requires_computed_thread and not has_computed_thread:
            raise ValueError(
                "Integrated assessment with assessment_context_packet must include at least one computed evidence thread."
            )

    def _validate_integrated_policy_synthesis(
        self,
        synthesis: Dict[str, Any],
        integrated: Dict[str, Any],
        policy_synthesis: Any,
        *,
        user_request: str,
        result_summaries: Dict[str, Dict[str, Any]],
        additional_context: Dict[str, Any],
    ) -> None:
        requires_judgment_first_policy = isinstance(
            additional_context.get("policy_context_packet"),
            dict,
        )
        if policy_synthesis is None:
            if requires_judgment_first_policy:
                raise ValueError(
                    "Integrated assessment with policy context must include "
                    "'integrated_assessment.policy_synthesis'."
                )
            return
        if not isinstance(policy_synthesis, dict):
            raise ValueError("'integrated_assessment.policy_synthesis' must be an object when provided.")

        judgment = policy_synthesis.get("one_sentence_judgment")
        if requires_judgment_first_policy:
            self._require_non_empty_string(
                judgment,
                "'integrated_assessment.policy_synthesis.one_sentence_judgment'",
            )
        elif judgment is not None and (not isinstance(judgment, str) or not judgment.strip()):
            raise ValueError(
                "'integrated_assessment.policy_synthesis.one_sentence_judgment' must be a non-empty string when provided."
            )

        narrative = policy_synthesis.get("policy_narrative")
        if requires_judgment_first_policy:
            self._require_non_empty_string(
                narrative,
                "'integrated_assessment.policy_synthesis.policy_narrative'",
            )
        elif narrative is not None and (not isinstance(narrative, str) or not narrative.strip()):
            raise ValueError(
                "'integrated_assessment.policy_synthesis.policy_narrative' must be a non-empty string when provided."
            )

        recommendations = policy_synthesis.get("policy_recommendations", [])
        if recommendations is None:
            recommendations = []
        if not isinstance(recommendations, list):
            raise ValueError("'integrated_assessment.policy_synthesis.policy_recommendations' must be a list.")
        if requires_judgment_first_policy and not recommendations:
            raise ValueError(
                "Integrated assessment with policy context must include "
                "'integrated_assessment.policy_synthesis.policy_recommendations'."
            )

        has_economic_data = self._context_contains_economic_data(result_summaries, additional_context)
        has_source_evidence = self._context_contains_source_evidence(result_summaries, additional_context)
        evidence_text = self._context_and_summary_text(
            result_summaries=result_summaries,
            additional_context=additional_context,
        )
        if isinstance(judgment, str):
            self._validate_no_unsupported_policy_thresholds(
                judgment,
                evidence_text=evidence_text,
                field_name="'integrated_assessment.policy_synthesis.one_sentence_judgment'",
            )
            if not has_economic_data and self._text_mentions_economic_claim(judgment):
                self._warn_synthesis_evidence_boundary(
                    synthesis,
                    integrated,
                    "Economic or industry wording in the policy judgment is not directly supported by production, "
                    "revenue, cost, or accounting data and should be treated as policy reasoning or a data gap."
                )
            if not has_source_evidence and self._text_mentions_source_policy(judgment):
                self._warn_synthesis_evidence_boundary(
                    synthesis,
                    integrated,
                    "Source, nutrient, pollution, or land-sea pathway interpretation in the policy judgment is not "
                    "directly supported by nutrient, river-input, discharge, or source-inventory data and should be "
                    "treated as policy reasoning or a data gap."
                )
        if isinstance(narrative, str):
            self._validate_no_unsupported_policy_thresholds(
                narrative,
                evidence_text=evidence_text,
                field_name="'integrated_assessment.policy_synthesis.policy_narrative'",
            )
            if not has_economic_data and self._text_mentions_economic_claim(narrative):
                self._warn_synthesis_evidence_boundary(
                    synthesis,
                    integrated,
                    "Economic or industry wording in the policy narrative is not directly supported by production, "
                    "revenue, cost, or accounting data and should be treated as policy reasoning or a data gap."
                )
            if not has_source_evidence and self._text_mentions_source_policy(narrative):
                self._warn_synthesis_evidence_boundary(
                    synthesis,
                    integrated,
                    "Source, nutrient, pollution, or land-sea pathway interpretation in the policy narrative is not "
                    "directly supported by nutrient, river-input, discharge, or source-inventory data and should be "
                    "treated as policy reasoning or a data gap."
                )
            if (
                self._future_or_projection_requested(user_request)
                and not self.FUTURE_EVIDENCE_RE.search(evidence_text)
                and not self.HISTORICAL_INFERENCE_RE.search(narrative)
            ):
                self._warn_synthesis_evidence_boundary(
                    synthesis,
                    integrated,
                    "Future-oriented wording in the policy narrative is not directly supported by projection or "
                    "scenario data and should be read as historical-trend-based inference."
                )

        self._validate_integrated_policy_recommendations(
            recommendations,
            synthesis=synthesis,
            integrated=integrated,
            has_economic_data=has_economic_data,
            has_source_evidence=has_source_evidence,
            evidence_text=evidence_text,
            additional_context=additional_context,
        )

    def _validate_integrated_policy_recommendations(
        self,
        recommendations: List[Any],
        *,
        synthesis: Dict[str, Any],
        integrated: Dict[str, Any],
        has_economic_data: bool,
        has_source_evidence: bool,
        evidence_text: str,
        additional_context: Dict[str, Any],
    ) -> None:
        required_fields = {
            "policy_title",
            "recommended_action",
            "evidence_status",
            "evidence_note",
            "evidence_result_ids",
        }
        for index, item in enumerate(recommendations):
            if not isinstance(item, dict):
                raise ValueError("Each integrated policy recommendation must be an object.")
            missing = sorted(required_fields - set(item))
            if missing:
                raise ValueError(
                    f"Integrated policy recommendation {index} is missing required field(s): {', '.join(missing)}."
                )
            for field_name in (
                "policy_title",
                "recommended_action",
                "evidence_note",
            ):
                value = item.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Integrated policy recommendation {index}.{field_name} must be a non-empty string."
                    )
                self._validate_no_unsupported_policy_thresholds(
                    value,
                    evidence_text=evidence_text,
                    field_name=f"'integrated_assessment.policy_synthesis.policy_recommendations[{index}].{field_name}'",
                )

            evidence_status = item.get("evidence_status")
            if evidence_status not in self.POLICY_RECOMMENDATION_EVIDENCE_STATUS:
                raise ValueError(
                    f"Integrated policy recommendation {index}.evidence_status must be one of "
                    "computed, indirect, or data_gap."
                )

            priority_places = item.get("priority_places", [])
            if priority_places is None:
                priority_places = []
            if not (
                isinstance(priority_places, list)
                and all(isinstance(entry, str) and entry.strip() for entry in priority_places)
            ):
                raise ValueError(
                    f"Integrated policy recommendation {index}.priority_places must be a list of strings when provided."
                )
            for value_index, value in enumerate(priority_places):
                self._validate_no_unsupported_policy_thresholds(
                    value,
                    evidence_text=evidence_text,
                    field_name=(
                        "'integrated_assessment.policy_synthesis."
                        f"policy_recommendations[{index}].priority_places[{value_index}]'"
                    ),
                )

            evidence_ids = item.get("evidence_result_ids")
            if evidence_ids is None:
                evidence_ids = []
            if not (
                isinstance(evidence_ids, list)
                and all(isinstance(entry, str) and entry.strip() for entry in evidence_ids)
            ):
                raise ValueError(
                    f"Integrated policy recommendation {index}.evidence_result_ids must be a list of strings."
                )
            if evidence_status in {"computed", "indirect"} and not evidence_ids:
                raise ValueError(
                    f"Integrated policy recommendation {index}.evidence_result_ids must be non-empty for "
                    f"evidence_status='{evidence_status}'."
                )
            if evidence_status == "data_gap" and not self.DATA_GAP_EVIDENCE_NOTE_RE.search(
                str(item.get("evidence_note") or "")
            ):
                raise ValueError(
                    f"Integrated policy recommendation {index}.evidence_note must explicitly say more data, "
                    "monitoring, validation, screening, or investigation is needed for data_gap recommendations."
                )

            item_for_specificity = dict(item)
            item_for_specificity["priority_places"] = " ".join(priority_places)
            item_for_specificity["evidence_result_ids"] = " ".join(evidence_ids)
            if evidence_status != "data_gap":
                self._validate_policy_row_specificity(
                    item_for_specificity,
                    row_label=f"Integrated policy recommendation {index}",
                    field_names=(
                        "policy_title",
                        "recommended_action",
                        "priority_places",
                        "evidence_note",
                        "evidence_result_ids",
                    ),
                    additional_context=additional_context,
                )

            item_text = " ".join(
                str(item_for_specificity.get(field_name, ""))
                for field_name in (
                    "policy_title",
                    "recommended_action",
                    "priority_places",
                    "evidence_status",
                    "evidence_note",
                    "evidence_result_ids",
                )
            ).lower()
            if not has_economic_data and self._text_mentions_economic_claim(item_text):
                self._warn_synthesis_evidence_boundary(
                    synthesis,
                    integrated,
                    "Economic or industry wording in policy recommendations is not directly supported by production, "
                    "revenue, cost, or accounting data and should be treated as policy reasoning or a data gap."
                )
            if (
                not has_source_evidence
                and self._text_mentions_source_policy(item_text)
                and (
                    self._text_mentions_direct_source_control(item_text)
                    or not self._source_policy_is_screening_or_investigation(item_text)
                )
            ):
                self._warn_synthesis_evidence_boundary(
                    synthesis,
                    integrated,
                    "Source, nutrient, pollution, or land-sea pathway interpretation in policy recommendations is "
                    "not directly supported by nutrient, river-input, discharge, or source-inventory data and "
                    "should be treated as screening or a hypothesis needing more data."
                )

    @staticmethod
    def _text_mentions_source_policy(text: str) -> bool:
        return bool(
            re.search(
                r"\b(source[- ]?control|nutrient[- ]?(?:load|control|reduction|management)|"
                r"pollution|emission|discharge|outfall|wastewater|river input|river discharge|"
                r"watershed|estuar(?:y|ine) source|land[- ]sea|source[- ]pathway)\b|"
                r"污染|营养盐|排放|排口|污水|流域|河口|陆海|源汇",
                text,
                re.IGNORECASE,
            )
        )

    def _text_mentions_unsupported_external_fact(self, text: str, evidence_text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        if not self.UNSUPPORTED_EXTERNAL_FACT_RE.search(text):
            return False
        evidence_lower = (evidence_text if isinstance(evidence_text, str) else "").lower()
        segments = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?。！？;；])\s+|\n+", text)
            if segment.strip()
        ] or [text]
        for segment in segments:
            if not self.UNSUPPORTED_EXTERNAL_FACT_RE.search(segment):
                continue
            if self._external_fact_match_is_data_gap_context(segment):
                continue
            for match in self.UNSUPPORTED_EXTERNAL_FACT_RE.finditer(segment):
                phrase = match.group(0).strip().lower()
                if phrase and phrase in evidence_lower:
                    continue
                return True
        return False

    def _external_fact_match_is_data_gap_context(self, text: str) -> bool:
        return bool(
            self.DATA_GAP_EVIDENCE_NOTE_RE.search(text)
            and re.search(
                r"\b(data|dataset|evidence|monitoring|projection|scenario|species|tolerance|"
                r"production|economic|outcome|available|supplied|provided|missing|lack|needed|required)\b|"
                r"数据|证据|监测|预测|情景|物种|耐受|产量|经济|缺少|缺乏|需要|未提供",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _source_policy_is_screening_or_investigation(text: str) -> bool:
        return bool(
            re.search(
                r"\b(screen(?:ing)?|investigat(?:e|ion)|audit|review|assess(?:ment)?|"
                r"survey|monitor(?:ing)?|trace|diagnos(?:e|is)|hypothesis|data gap|"
                r"source[- ]pathway screening|nutrient[- ]load screening)\b|"
                r"筛查|调查|审查|评估|监测|核查|溯源|数据缺口",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _text_mentions_direct_source_control(text: str) -> bool:
        return bool(
            re.search(
                r"\b(?:implement|enforce|mandate|require|prescribe|impose|control|reduce|cut|limit)\b.{0,80}"
                r"\b(?:nutrient|pollution|emission|discharge|wastewater|source[- ]?control)\b|"
                r"\b(?:nutrient|pollution|emission|discharge|wastewater|source[- ]?control)\b.{0,80}"
                r"\b(?:control|reduction|reduce|cut|limit|mandate|require|prescribe)\b|"
                r"控排|减排|污染控制|营养盐控制",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _require_non_empty_string(value: Any, field_name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    @staticmethod
    def _validate_optional_str_list(value: Any, field_name: str) -> None:
        if value is None:
            return
        if not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
            raise ValueError(f"{field_name} must be a list of strings when provided.")

    @staticmethod
    def _validate_required_str_list(value: Any, field_name: str) -> None:
        if not (isinstance(value, list) and value and all(isinstance(item, str) and item.strip() for item in value)):
            raise ValueError(f"{field_name} must be a non-empty list of non-empty strings.")

    def _risk_hotspots_make_specific_ranking(self, risk_hotspots: Any) -> bool:
        if not isinstance(risk_hotspots, list):
            return False
        text = " ".join(str(item).lower() for item in risk_hotspots)
        if re.search(r"\b(insufficient|not enough|no bottom oxygen|no hypoxia|data gap|unknown)\b|证据不足|缺少", text):
            return False
        return bool(re.search(r"\b(high|medium|hotspot|higher risk|elevated risk|risk area|priority area)\b|高风险|热点", text))

    def _resolve_synthesis_profile_id(
        self,
        *,
        user_request: str,
        active_plan: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Optional[str]:
        for source in (
            additional_context,
            active_plan,
        ):
            profile_id = self._extract_synthesis_profile_id(source)
            if profile_id:
                return profile_id
        if self._integrated_assessment_requested(user_request):
            return self.INTEGRATED_ASSESSMENT_PROFILE_ID
        return None

    def _extract_synthesis_profile_id(self, value: Any) -> Optional[str]:
        if not isinstance(value, dict):
            return None
        direct = value.get("synthesis_profile_id")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        proposal_context = value.get("analysis_proposal_context")
        if isinstance(proposal_context, dict):
            proposal = proposal_context.get("approved_proposal")
            if isinstance(proposal, dict):
                proposal_profile = proposal.get("synthesis_profile_id")
                if isinstance(proposal_profile, str) and proposal_profile.strip():
                    return proposal_profile.strip()
                skill_plan = proposal.get("skill_plan")
                if isinstance(skill_plan, dict):
                    skill_plan_profile = skill_plan.get("synthesis_profile_id")
                    if isinstance(skill_plan_profile, str) and skill_plan_profile.strip():
                        return skill_plan_profile.strip()
        return None

    def _integrated_assessment_requested(self, user_request: str) -> bool:
        if not isinstance(user_request, str):
            return False
        return bool(self.INTEGRATED_ASSESSMENT_INTENT_RE.search(user_request))

    def _future_or_projection_requested(self, user_request: str) -> bool:
        if not isinstance(user_request, str):
            return False
        return bool(self.FUTURE_OR_PROJECTION_INTENT_RE.search(user_request))

    def _context_contains_oxygen_or_hypoxia_evidence(
        self,
        *,
        result_summaries: Dict[str, Dict[str, Any]],
        additional_context: Dict[str, Any],
    ) -> bool:
        evidence_context = dict(additional_context)
        evidence_context.pop("policy_context_packet", None)
        return self._text_mentions_any(
            self._context_and_summary_text(
                result_summaries=result_summaries,
                additional_context=evidence_context,
            ),
            {"oxygen", "dissolved oxygen", "bottom oxygen", "hypoxia", "hypoxic", "o2", "低氧", "缺氧"},
        )

    def _context_contains_economic_data(
        self,
        result_summaries: Dict[str, Dict[str, Any]],
        additional_context: Dict[str, Any],
    ) -> bool:
        packet_flags = self._policy_context_flags(additional_context)
        if isinstance(packet_flags.get("has_economic_data"), bool):
            return bool(packet_flags["has_economic_data"])
        text = self._context_and_summary_text(
            result_summaries=result_summaries,
            additional_context=additional_context,
        )
        if self.ECONOMIC_DATA_GAP_RE.search(text):
            return False
        return self._text_mentions_any(
            text,
            {
                "economic_data",
                "economic dataset",
                "cost dataset",
                "benefit dataset",
                "revenue dataset",
                "valuation",
                "cost-benefit data",
            },
        )

    def _context_contains_source_evidence(
        self,
        result_summaries: Dict[str, Dict[str, Any]],
        additional_context: Dict[str, Any],
    ) -> bool:
        packet_flags = self._policy_context_flags(additional_context)
        if isinstance(packet_flags.get("has_source_evidence"), bool):
            return bool(packet_flags["has_source_evidence"])
        return self._text_mentions_any(
            self._context_and_summary_text(
                result_summaries=result_summaries,
                additional_context=additional_context,
            ),
            {
                "source_evidence",
                "nutrient_load",
                "nutrient loading",
                "organic loading",
                "river discharge",
                "river input",
                "estuary input",
                "outfall",
                "discharge outlet",
                "source inventory",
                "wastewater",
                "point source",
            },
        )

    @staticmethod
    def _policy_context_flags(additional_context: Dict[str, Any]) -> Dict[str, Any]:
        packet = additional_context.get("policy_context_packet") if isinstance(additional_context, dict) else None
        if not isinstance(packet, dict):
            return {}
        flags = packet.get("data_availability_flags")
        return flags if isinstance(flags, dict) else {}

    def _context_and_summary_text(
        self,
        *,
        result_summaries: Dict[str, Dict[str, Any]],
        additional_context: Dict[str, Any],
    ) -> str:
        return " ".join(
            item
            for item in (
                self._result_summaries_text(result_summaries),
                self._jsonish_text(additional_context),
            )
            if item
        ).lower()

    def _validate_policy_row_specificity(
        self,
        item: Dict[str, Any],
        *,
        row_label: str,
        field_names: tuple[str, ...],
        additional_context: Dict[str, Any],
    ) -> None:
        text = " ".join(str(item.get(field_name, "")) for field_name in field_names).lower()
        action_text = " ".join(
            str(item.get(field_name, ""))
            for field_name in (
                "recommended_action",
                "recommendation",
                "target",
                "where_when",
                "policy_lever",
            )
        ).lower()
        action_group = str(item.get("action_group") or "").strip()
        has_detail_context = self._policy_detail_context_available(additional_context)
        needs_anchor = has_detail_context and (
            bool(self.POLICY_GENERIC_ROW_RE.search(text))
            or action_group in {"spatial_priority", "oxygen_response", "seasonal_operations", "driver_adaptation"}
        )
        if needs_anchor and not self._policy_text_has_concrete_anchor(text, additional_context):
            raise ValueError(
                f"{row_label} must include a concrete evidence anchor such as coordinates, rank, extrema, "
                "event count, trend slope/p-value, timing window, result ID, or a named hotspot from the policy context."
            )
        if self.POLICY_GENERIC_ACTION_RE.search(action_text) and not self.POLICY_OPERATIONAL_DETAIL_RE.search(action_text):
            raise ValueError(
                f"{row_label} must specify operational detail: what to monitor or manage, where/when, and the "
                "decision object affected."
            )

    def _policy_text_has_concrete_anchor(self, text: str, additional_context: Dict[str, Any]) -> bool:
        if self.POLICY_EVIDENCE_ANCHOR_RE.search(text):
            return True
        for term in self._policy_context_anchor_terms(additional_context):
            lowered = term.lower()
            if lowered and lowered in text:
                return True
        return False

    @staticmethod
    def _policy_detail_context_available(additional_context: Dict[str, Any]) -> bool:
        packet = additional_context.get("policy_context_packet") if isinstance(additional_context, dict) else None
        if not isinstance(packet, dict):
            return False
        if packet.get("row_detail_contract"):
            return True
        risk_signals = packet.get("risk_signals")
        if not isinstance(risk_signals, dict):
            return False
        evidence_anchors = risk_signals.get("evidence_anchors")
        if isinstance(evidence_anchors, list) and evidence_anchors:
            return True
        hotspots = risk_signals.get("hotspots")
        return isinstance(hotspots, list) and any(
            isinstance(item, dict) and item.get("evidence_anchor") for item in hotspots
        )

    @staticmethod
    def _policy_context_anchor_terms(additional_context: Dict[str, Any]) -> List[str]:
        packet = additional_context.get("policy_context_packet") if isinstance(additional_context, dict) else None
        if not isinstance(packet, dict):
            return []
        risk_signals = packet.get("risk_signals")
        if not isinstance(risk_signals, dict):
            return []
        terms: List[str] = []
        for anchor in risk_signals.get("evidence_anchors") or []:
            if isinstance(anchor, str) and anchor.strip():
                terms.append(anchor.strip())
        for timing in risk_signals.get("timing") or []:
            if isinstance(timing, str) and timing.strip():
                terms.append(timing.strip())
        generic_labels = {"mapped hotspot", "computed hotspot", "oxygen-risk hotspot", "high-burden zone"}
        for hotspot in risk_signals.get("hotspots") or []:
            if not isinstance(hotspot, dict):
                continue
            for key in ("label", "evidence_anchor"):
                value = hotspot.get(key)
                if not isinstance(value, str):
                    continue
                lowered = value.strip().lower()
                if len(lowered) >= 6 and lowered not in generic_labels:
                    terms.append(value.strip())
        return terms

    def _validate_no_unsupported_policy_thresholds(
        self,
        text: str,
        *,
        evidence_text: str,
        field_name: str,
    ) -> None:
        if not isinstance(text, str) or not text.strip():
            return
        evidence_numbers = self._numeric_token_values(evidence_text)
        unsupported: List[str] = []
        for match in self.POLICY_THRESHOLD_CLAIM_RE.finditer(text):
            context = self._local_policy_threshold_context(text, match.start(), match.end())
            if not self._looks_like_policy_cutoff(context, match.group(0)):
                continue
            for value in self._numeric_token_values(match.group(0)):
                if not self._numeric_value_supported_by_evidence(value, evidence_numbers):
                    unsupported.append(self._format_numeric_token(value))
        if unsupported:
            unique_unsupported = sorted(set(unsupported))
            raise ValueError(
                f"{field_name} contains unsupported numeric policy threshold(s): {', '.join(unique_unsupported)}. "
                "Policy thresholds must appear explicitly in evidence packets or result summaries."
            )

    def _numeric_token_values(self, text: str) -> List[float]:
        values: List[float] = []
        if not isinstance(text, str):
            return values
        for match in self.NUMERIC_TOKEN_RE.finditer(text):
            try:
                value = float(match.group(0).replace(",", ""))
            except ValueError:
                continue
            values.append(value)
        return values

    @staticmethod
    def _local_policy_threshold_context(text: str, start: int, end: int) -> str:
        left = max(0, start - 80)
        right = min(len(text), end + 80)
        return text[left:right].lower()

    def _looks_like_policy_cutoff(self, context: str, matched_text: str) -> bool:
        matched_lower = matched_text.lower()
        if re.search(r"\b(?:threshold|trigger|cutoff|cut-off)\b|阈值|门槛|触发", matched_lower):
            return True
        return bool(self.POLICY_CUTOFF_CONTEXT_RE.search(context))

    @staticmethod
    def _numeric_value_supported_by_evidence(value: float, evidence_values: List[float]) -> bool:
        for evidence_value in evidence_values:
            scale = max(abs(value), abs(evidence_value), 1.0)
            if abs(value - evidence_value) <= max(1e-9, scale * 5e-3):
                return True
        return False

    @staticmethod
    def _format_numeric_token(value: float) -> str:
        return f"{value:.8g}"

    @staticmethod
    def _normalized_policy_reasoning_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value.lower()).strip()

    @staticmethod
    def _jsonish_text(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)

    def _validate_policy_guidance(
        self,
        synthesis: Dict[str, Any],
        *,
        policy_requested: bool,
        structured_policy_requested: Optional[bool] = None,
        user_request: str,
        result_summaries: Dict[str, Dict[str, Any]],
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not policy_requested:
            return
        if structured_policy_requested is None:
            structured_policy_requested = policy_requested
        policy_guidance = synthesis.get("policy_guidance")
        if policy_guidance is None:
            raise ValueError("Policy query must include 'policy_guidance'.")
        if not isinstance(policy_guidance, dict):
            raise ValueError("'policy_guidance' must be an object when provided.")
        self._normalize_policy_guidance_enums(policy_guidance)

        should_include = policy_guidance.get("should_include")
        if not isinstance(should_include, bool):
            raise ValueError("'policy_guidance.should_include' must be boolean.")
        headline = policy_guidance.get("headline")
        if should_include and (not isinstance(headline, str) or not headline.strip()):
            raise ValueError("'policy_guidance.headline' must be a non-empty string when included.")
        place_based_brief = policy_guidance.get("place_based_policy_brief")
        if should_include and (
            not isinstance(place_based_brief, str) or not place_based_brief.strip()
        ):
            raise ValueError(
                "'policy_guidance.place_based_policy_brief' must be a non-empty string when included."
            )

        evidence_limits = policy_guidance.get("evidence_limits", [])
        if evidence_limits is not None and not (
            isinstance(evidence_limits, list) and all(isinstance(item, str) for item in evidence_limits)
        ):
            raise ValueError("'policy_guidance.evidence_limits' must be a list of strings.")

        matrix = policy_guidance.get("evidence_action_matrix", [])
        if matrix is None:
            matrix = []
        if not isinstance(matrix, list):
            raise ValueError("'policy_guidance.evidence_action_matrix' must be a list.")
        if policy_requested and should_include and not matrix:
            raise ValueError("Policy guidance requested but evidence_action_matrix is empty.")

        chlorophyll_decreasing = self._summary_indicates_chlorophyll_decrease(result_summaries)
        has_economic_data = self._summaries_contain_economic_data(result_summaries)
        economic_requested = self._economic_guidance_requested(user_request)
        hypoxia_endpoint_requested = self._hypoxia_endpoint_policy_requested(user_request, result_summaries)
        has_source_evidence = self._summaries_contain_source_evidence(result_summaries)
        evidence_text = self._context_and_summary_text(
            result_summaries=result_summaries,
            additional_context=additional_context or {},
        )

        required_fields = {
            "priority",
            "action_type",
            "target",
            "where_when",
            "evidence_basis",
            "recommendation",
            "guardrail",
            "evidence_strength",
        }
        for index, item in enumerate(matrix):
            if not isinstance(item, dict):
                raise ValueError("Each policy guidance matrix item must be an object.")
            missing = sorted(required_fields - set(item))
            if missing:
                raise ValueError(
                    f"Policy guidance matrix item {index} is missing required field(s): {', '.join(missing)}."
                )
            priority = item.get("priority")
            action_type = item.get("action_type")
            evidence_strength = item.get("evidence_strength")
            if priority not in self.POLICY_PRIORITIES:
                raise ValueError(f"Policy guidance matrix item {index} has invalid priority '{priority}'.")
            if action_type not in self.POLICY_ACTION_TYPES:
                raise ValueError(f"Policy guidance matrix item {index} has invalid action_type '{action_type}'.")
            if evidence_strength not in self.POLICY_EVIDENCE_STRENGTHS:
                raise ValueError(
                    f"Policy guidance matrix item {index} has invalid evidence_strength '{evidence_strength}'."
                )
            for field_name in ("target", "where_when", "evidence_basis", "recommendation", "guardrail"):
                value = item.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Policy guidance matrix item {index}.{field_name} must be a non-empty string."
                    )

            item_text = " ".join(
                str(item.get(field_name, ""))
                for field_name in ("target", "where_when", "evidence_basis", "recommendation", "guardrail")
            ).lower()
            for field_name in ("target", "where_when", "evidence_basis", "recommendation", "guardrail"):
                self._validate_no_unsupported_policy_thresholds(
                    str(item.get(field_name, "")),
                    evidence_text=evidence_text,
                    field_name=f"'policy_guidance.evidence_action_matrix[{index}].{field_name}'",
                )
            self._validate_policy_row_specificity(
                item,
                row_label=f"Policy guidance matrix item {index}",
                field_names=("target", "where_when", "evidence_basis", "recommendation", "guardrail"),
                additional_context=additional_context or {},
            )
            if (
                chlorophyll_decreasing
                and action_type == "source_control"
                and evidence_strength == "supported"
                and self._text_mentions_any(item_text, {"chlorophyll", "chl", "bloom", "eutroph", "nutrient"})
            ):
                self._add_synthesis_warning(
                    synthesis,
                    "Chlorophyll or bloom evidence does not directly prove nutrient or source-control needs; "
                    "treat related policy language as screening or hypothesis generation."
                )

            if (
                action_type in {"source_control", "discharge_outlet", "river_estuary"}
                and evidence_strength == "supported"
                and self._text_mentions_any(item_text, {"sst", "temperature", "temp", "stratification"})
            ):
                self._add_synthesis_warning(
                    synthesis,
                    "SST or stratification evidence supports vulnerability and timing interpretation, not direct "
                    "source or pollution attribution."
                )

            if (
                action_type in {"source_control", "discharge_outlet", "river_estuary"}
                and evidence_strength == "supported"
                and not has_source_evidence
            ):
                self._add_synthesis_warning(
                    synthesis,
                    "Source-control, river/estuary, or discharge-outlet guidance is not directly supported by "
                    "nutrient, river-input, discharge, or source-inventory evidence and should be treated as "
                    "screening or investigation."
                )

            if economic_requested and not has_economic_data and self._text_mentions_economic_claim(item_text):
                self._add_synthesis_warning(
                    synthesis,
                    "Economic guidance is not directly supported by production, revenue, cost, or accounting data "
                    "and should be treated as economic-assessment motivation rather than a quantified conclusion."
                )

        if structured_policy_requested and hypoxia_endpoint_requested and should_include:
            action_types = {
                str(item.get("action_type"))
                for item in matrix
                if isinstance(item, dict) and isinstance(item.get("action_type"), str)
            }
            missing_actions = {"monitoring", "seasonal_management", "coastal_planning"} - action_types
            if missing_actions:
                raise ValueError(
                    "Hypoxia or bottom-oxygen policy guidance must include monitoring, seasonal_management, "
                    f"and coastal_planning rows; missing: {', '.join(sorted(missing_actions))}."
                )

    def _normalize_policy_guidance_enums(self, policy_guidance: Dict[str, Any]) -> None:
        """Normalize common LLM enum aliases before strict validation.

        The matrix should still be schema-bound, but harmless labels like
        "near_term" should not cause an otherwise useful policy section to be
        dropped. Unknown values still fail validation below.
        """
        matrix = policy_guidance.get("evidence_action_matrix")
        if not isinstance(matrix, list):
            return

        priority_aliases = {
            "urgent": "high",
            "critical": "high",
            "immediate": "high",
            "near_term": "medium",
            "near-term": "medium",
            "near term": "medium",
            "medium_term": "medium",
            "mid_term": "medium",
            "moderate": "medium",
            "long_term": "low",
            "long-term": "low",
            "long term": "low",
            "low_regret": "screening",
            "low-regret": "screening",
            "review": "screening",
        }
        action_aliases = {
            "source": "source_control",
            "nutrient_control": "source_control",
            "organic_loading_control": "source_control",
            "outfall": "discharge_outlet",
            "outfall_review": "discharge_outlet",
            "discharge": "discharge_outlet",
            "river": "river_estuary",
            "estuary": "river_estuary",
            "river_inputs": "river_estuary",
            "early_warning": "seasonal_management",
            "seasonal_warning": "seasonal_management",
            "planning": "coastal_planning",
            "coastal_management": "coastal_planning",
            "economic": "economic_assessment",
            "economics": "economic_assessment",
        }
        strength_aliases = {
            "strong": "supported",
            "moderate": "limited",
            "weak": "limited",
            "precautionary": "screening",
            "screen": "screening",
            "unsupported": "not_supported",
            "not supported": "not_supported",
            "not-supported": "not_supported",
        }

        for item in matrix:
            if not isinstance(item, dict):
                continue
            for key, aliases in (
                ("priority", priority_aliases),
                ("action_type", action_aliases),
                ("evidence_strength", strength_aliases),
            ):
                raw_value = item.get(key)
                if not isinstance(raw_value, str):
                    continue
                normalized = " ".join(raw_value.strip().lower().split())
                underscored = normalized.replace(" ", "_")
                item[key] = aliases.get(normalized, aliases.get(underscored, underscored))

    def _policy_guidance_requested(self, user_request: str) -> bool:
        if not isinstance(user_request, str):
            return False
        return bool(self.POLICY_INTENT_RE.search(user_request))

    @staticmethod
    def _context_policy_making_intent(additional_context: Dict[str, Any]) -> Optional[bool]:
        if not isinstance(additional_context, dict):
            return None
        return ResultSynthesizer._coerce_optional_bool(additional_context.get("policy_making_intent"))

    @staticmethod
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

    def _structured_policy_guidance_requested(self, user_request: str) -> bool:
        if not isinstance(user_request, str):
            return False
        stripped_request = self.NEGATED_STRUCTURED_POLICY_INTENT_RE.sub(" ", user_request)
        return bool(self.STRUCTURED_POLICY_INTENT_RE.search(stripped_request))

    def _economic_guidance_requested(self, user_request: str) -> bool:
        if not isinstance(user_request, str):
            return False
        return bool(re.search(r"\b(economic|economics|cost|benefit|revenue|loss|damage)\b|经济|成本|收益|损失", user_request, re.IGNORECASE))

    def _summary_indicates_chlorophyll_decrease(self, result_summaries: Dict[str, Dict[str, Any]]) -> bool:
        text = self._result_summaries_text(result_summaries).lower()
        if "chlorophyll" not in text and '"chl"' not in text and "chl-a" not in text:
            return False
        return bool(
            re.search(r"chlorophyll[^{}]{0,160}(decreas|declin|negative|slope[^{}]{0,40}-)", text)
            or re.search(r"(decreas|declin|negative)[^{}]{0,160}chlorophyll", text)
        )

    def _summaries_contain_economic_data(self, result_summaries: Dict[str, Dict[str, Any]]) -> bool:
        text = self._result_summaries_text(result_summaries).lower()
        if self.ECONOMIC_DATA_GAP_RE.search(text):
            return False
        return self._text_mentions_any(
            text,
            {
                "economic_data",
                "cost",
                "costs",
                "benefit",
                "benefits",
                "revenue",
                "loss",
                "losses",
                "damage",
                "damages",
                "valuation",
                "cost-benefit",
            },
        )

    def _hypoxia_endpoint_policy_requested(
        self,
        user_request: str,
        result_summaries: Dict[str, Dict[str, Any]],
    ) -> bool:
        request_text = user_request.lower() if isinstance(user_request, str) else ""
        if not self._text_mentions_any(request_text, {"hypoxia", "hypoxic", "oxygen", "低氧", "缺氧"}):
            return False
        summary_text = self._result_summaries_text(result_summaries).lower()
        return self._text_mentions_any(summary_text, {"hypoxia", "hypoxic", "oxygen", "dissolved oxygen", "o2"})

    def _summaries_contain_source_evidence(self, result_summaries: Dict[str, Dict[str, Any]]) -> bool:
        text = self._result_summaries_text(result_summaries).lower()
        source_terms = {
            "nutrient_load",
            "nutrient loading",
            "organic loading",
            "river discharge",
            "river input",
            "estuary input",
            "outfall",
            "discharge outlet",
            "source inventory",
            "wastewater",
            "point source",
        }
        return self._text_mentions_any(text, source_terms)

    def _result_summaries_text(self, result_summaries: Dict[str, Dict[str, Any]]) -> str:
        try:
            return json.dumps(result_summaries, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(result_summaries)

    @staticmethod
    def _text_mentions_any(text: str, terms: Set[str]) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in terms)

    @staticmethod
    def _text_mentions_economic_claim(text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        segments = [
            segment.strip()
            for segment in re.split(r"(?<=[.!?。！？;；])\s+|\n+", text)
            if segment.strip()
        ]
        for segment in segments or [text]:
            if not ResultSynthesizer.ECONOMIC_CLAIM_RE.search(segment):
                continue
            if ResultSynthesizer._is_economic_data_gap_statement(segment):
                continue
            if (
                ResultSynthesizer.ECONOMIC_DATA_COLLECTION_RE.search(segment)
                and not ResultSynthesizer.ECONOMIC_ASSERTION_RE.search(segment)
            ):
                continue
            if (
                ResultSynthesizer.ECONOMIC_LIMITATION_RE.search(segment)
                and not ResultSynthesizer.ECONOMIC_ASSERTION_RE.search(segment)
            ):
                continue
            return True
        return False

    @staticmethod
    def _is_economic_data_gap_statement(text: str) -> bool:
        return bool(
            ResultSynthesizer.ECONOMIC_DATA_GAP_RE.search(text)
            and ResultSynthesizer.ECONOMIC_LIMITATION_RE.search(text)
            and not ResultSynthesizer.ECONOMIC_ASSERTION_RE.search(text)
        )

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if self.planner is not None:
            return self.planner._get_client()
        self.client = self._adapter.get_client()
        return self.client

    def _extract_response_text(self, response: Any) -> str:
        return self._adapter.extract_response_text(response)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        return self._adapter.parse_json_response(text)

    def _compact_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = []
        for step in plan.get("steps", []):
            if not isinstance(step, dict):
                continue
            steps.append(
                {
                    "step_id": step.get("step_id"),
                    "tool": step.get("tool"),
                    "save_as": step.get("save_as"),
                }
            )
        compact = {
            "skill_id": plan.get("skill_id"),
            "skills_used": plan.get("skills_used"),
            "status": plan.get("status"),
            "steps": steps,
        }
        return {key: value for key, value in compact.items() if value is not None}

    def _compact_completed_steps(self, completed_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact_steps: List[Dict[str, Any]] = []
        for step in completed_steps:
            if not isinstance(step, dict):
                continue
            result_summary = step.get("result_summary")
            if isinstance(result_summary, dict):
                compact_result_summary = result_summary
            else:
                compact_result_summary = self._compact_object(result_summary, max_depth=4, max_items=10)
            compact_steps.append(
                {
                    "step_id": step.get("step_id"),
                    "tool": step.get("tool"),
                    "result_id": step.get("result_id"),
                    "output_type": step.get("output_type"),
                    "result_summary": compact_result_summary,
                    "summary": self._compact_object(step.get("summary"), max_depth=4, max_items=10),
                }
            )
        return compact_steps

    def _compact_result_summaries(self, result_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {
            result_id: summary if isinstance(summary, dict) else self._compact_object(summary, max_depth=4, max_items=10)
            for result_id, summary in result_summaries.items()
        }

    def _compact_object(self, value: Any, max_depth: int = 3, max_items: int = 6) -> Any:
        if max_depth <= 0:
            return str(type(value).__name__)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            compact = [
                self._compact_object(item, max_depth=max_depth - 1, max_items=max_items)
                for item in value[:max_items]
            ]
            if len(value) > max_items:
                compact.append(f"... ({len(value) - max_items} more)")
            return compact
        if isinstance(value, tuple):
            return self._compact_object(list(value), max_depth=max_depth, max_items=max_items)
        if isinstance(value, dict):
            compact_dict: Dict[str, Any] = {}
            items = list(value.items())
            for index, (key, item) in enumerate(items[:max_items]):
                compact_dict[key] = self._compact_object(
                    item,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                )
            if len(items) > max_items:
                compact_dict["..."] = f"{len(items) - max_items} more keys"
            return compact_dict
        return str(value)
