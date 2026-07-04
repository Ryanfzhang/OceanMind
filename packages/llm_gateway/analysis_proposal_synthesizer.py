from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set

from packages.llm_gateway.skill_planner import SkillPlanner
from packages.llm_gateway.config import load_model_name
from packages.llm_gateway.openai_compatible_client import OpenAICompatibleClientAdapter
from packages.skill_system import list_skill_ids, load_all_planner_hints, load_skill_markdown
from packages.tool_loader.registry import get_tool_contract
from packages.tool_loader.validation import validate_tool_params


class AnalysisProposalSynthesizer:
    """Create a user-visible analysis plan before expensive tool execution."""

    INTEGRATED_ASSESSMENT_PROFILE_ID = "ocean_integrated_assessment"
    PROPOSAL_MIN_TOKENS = 4000
    PLANNER_BACKED_MIN_TOKENS = 6000
    DEFAULT_MODEL = load_model_name("RESULT_SYNTHESIZER_MODEL", default=SkillPlanner.DEFAULT_MODEL)
    REQUIRED_FIELDS = {
        "title",
        "public_question",
        "proposed_query",
        "analysis_steps",
        "expected_outputs",
        "limitations",
        "approval_prompt",
    }

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[Any] = None,
        planner: Optional[SkillPlanner] = None,
        model: str = DEFAULT_MODEL,
        planner_model: Optional[str] = None,
        skills_root: Optional[str] = None,
        trust_env: bool = False,
        request_retries: int = 2,
    ):
        self.skills_root = skills_root
        self._planner = planner
        self._planner_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "model": planner_model or SkillPlanner.DEFAULT_MODEL,
            "skills_root": skills_root,
            "trust_env": trust_env,
            "request_retries": request_retries,
        }
        self._adapter = OpenAICompatibleClientAdapter(
            api_key=api_key,
            base_url=base_url,
            model=model,
            client=client,
            trust_env=trust_env,
            request_retries=request_retries,
        )

    def propose(
        self,
        *,
        user_request: str,
        dataset_context: Dict[str, Any],
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        capabilities = self._dataset_capabilities(dataset_context)
        if self._should_generate_planner_backed_proposal(user_request, capabilities):
            deterministic = self._deterministic_environment_health_proposal(
                user_request=user_request,
                dataset_context=dataset_context,
                capabilities=capabilities,
            )
            if deterministic is not None:
                return deterministic
            try:
                plan = self._generate_planner_backed_plan(
                    user_request=user_request,
                    dataset_context=dataset_context,
                    extracted_params=extracted_params or {},
                    additional_context=additional_context or {},
                    max_tokens=max(max_tokens, self.PROPOSAL_MIN_TOKENS),
                    temperature=temperature,
                )
                proposal = self.build_proposal_from_plan(
                    user_request=user_request,
                    dataset_context=dataset_context,
                    plan=plan,
                    capabilities=capabilities,
                )
                self._validate_proposal(proposal, capabilities)
                return proposal
            except Exception as exc:
                return self._fallback_proposal(
                    user_request,
                    dataset_context,
                    capabilities,
                    f"planner-backed proposal failed: {exc}",
                )

        client = self._adapter.get_client()
        validation_error: Optional[str] = None
        last_proposal: Optional[Dict[str, Any]] = None

        for attempt in range(2):
            messages = self._build_messages(
                user_request=user_request,
                dataset_context=dataset_context,
                extracted_params=extracted_params or {},
                additional_context=additional_context or {},
                capabilities=capabilities,
                validation_error=validation_error,
            )
            response = self._adapter.create_message(
                client=client,
                max_tokens=max(max_tokens, self.PROPOSAL_MIN_TOKENS),
                temperature=temperature,
                system=messages["system"],
                messages=messages["messages"],
                request_name="analysis_proposal" if attempt == 0 else "analysis_proposal_retry",
                json_response=True,
            )
            text = self._adapter.extract_response_text(response)
            try:
                proposal = self._adapter.parse_json_response(text)
                last_proposal = proposal
                normalized = self._normalize_proposal(proposal)
                self._validate_proposal(normalized, capabilities)
                return normalized
            except Exception as exc:
                validation_error = str(exc)
                if attempt + 1 < 2:
                    continue

        if isinstance(last_proposal, dict):
            return self._fallback_proposal(user_request, dataset_context, capabilities, validation_error)
        raise ValueError(validation_error or "Unable to build analysis proposal.")

    def _build_messages(
        self,
        *,
        user_request: str,
        dataset_context: Dict[str, Any],
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
        capabilities: Dict[str, Any],
        validation_error: Optional[str],
    ) -> Dict[str, Any]:
        retry_note = (
            f"\nYour previous proposal failed validation: {validation_error}. "
            "Revise the proposal so it only uses available dataset variables and executable analysis types.\n"
            if validation_error
            else ""
        )
        system = (
            "You create user-visible analysis proposals for OceanMind before expensive ocean-data execution.\n"
            "Do not reveal hidden chain-of-thought. Return only a concise JSON object.\n"
            "Use the active dataset, workspace geometry, and skill capabilities to translate a broad public question "
            "into one skill-backed executable scientific query. Prefer the provided analysis_templates. The "
            "proposed_query must be specific enough for the existing skill planner to run after user approval.\n"
            "Always include selected_skills and a compact skill_plan when a matching template or skill capability exists. "
            "Do not invent skill ids.\n"
            "Prefer evidence-bounded language: explain what the data can support and what it cannot prove.\n"
            "If a variable or depth capability is unavailable, do not propose analyses that require it; mention the "
            "limitation instead.\n"
            f"{retry_note}"
        )
        payload = {
            "user_request": user_request,
            "dataset_context": dataset_context,
            "dataset_capabilities": capabilities,
            "structured_inputs": extracted_params,
            "workspace_context": additional_context.get("workspace_context", {}),
            "analysis_templates": self._analysis_templates(capabilities),
            "skill_capabilities": self._compact_skill_capabilities(),
            "required_output_schema": {
                "title": "string",
                "public_question": "string",
                "proposed_query": "string",
                "analysis_steps": ["string"],
                "expected_outputs": ["string"],
                "limitations": ["string"],
                "approval_prompt": "string",
                "synthesis_profile_id": "optional string, e.g. ocean_integrated_assessment",
                "selected_skills": ["string"],
                "skill_plan": {
                    "primary_skill": "string",
                    "skills_used": ["string"],
                    "synthesis_profile_id": "optional string for the summary agent only",
                    "planned_tools": ["string"],
                    "planned_steps": [
                        {"label": "string", "tool": "string", "purpose": "string"}
                    ],
                },
            },
        }
        return {
            "system": system,
            "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}],
        }

    def _normalize_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for field in ("title", "public_question", "proposed_query", "approval_prompt"):
            value = proposal.get(field)
            normalized[field] = str(value).strip() if isinstance(value, str) else ""
        profile_id = proposal.get("synthesis_profile_id")
        if isinstance(profile_id, str) and profile_id.strip():
            normalized["synthesis_profile_id"] = profile_id.strip()
        for field in ("analysis_steps", "expected_outputs", "limitations"):
            normalized[field] = self._coerce_str_list(proposal.get(field))
        normalized["selected_skills"] = self._coerce_str_list(proposal.get("selected_skills"))
        normalized["skill_plan"] = self._normalize_skill_plan(proposal.get("skill_plan"))
        plan = proposal.get("plan")
        if isinstance(plan, dict):
            normalized["plan"] = plan
        if not normalized.get("synthesis_profile_id"):
            skill_plan_profile = normalized["skill_plan"].get("synthesis_profile_id")
            if isinstance(skill_plan_profile, str) and skill_plan_profile.strip():
                normalized["synthesis_profile_id"] = skill_plan_profile.strip()
        if not normalized["selected_skills"] and normalized["skill_plan"].get("skills_used"):
            normalized["selected_skills"] = list(normalized["skill_plan"]["skills_used"])
        return normalized

    def _validate_proposal(self, proposal: Dict[str, Any], capabilities: Dict[str, Any]) -> None:
        missing = [
            field
            for field in self.REQUIRED_FIELDS
            if not proposal.get(field)
        ]
        if missing:
            raise ValueError(f"Analysis proposal is missing required field(s): {', '.join(sorted(missing))}.")

        proposed_query = str(proposal["proposed_query"]).strip()
        if len(proposed_query) < 40:
            raise ValueError("proposed_query must be a concrete executable analysis request.")

        lower = proposed_query.lower()
        variables: Set[str] = set(capabilities.get("variables", set()))
        depth_count = int(capabilities.get("depth_level_count") or 0)

        plan = proposal.get("plan")
        if plan is not None:
            if not isinstance(plan, dict):
                raise ValueError("Analysis proposal plan must be a JSON object.")
            steps = plan.get("steps")
            if not isinstance(steps, list) or not steps:
                raise ValueError("Analysis proposal plan must include non-empty steps.")

        if not self._has_any(variables, {"oxygen", "o2"}) and re.search(r"\b(oxygen|dissolved oxygen|hypoxia|hypoxic)\b", lower):
            raise ValueError("Dataset lacks oxygen, so proposed_query must not request oxygen or hypoxia analysis.")
        if "temp" not in variables and "temperature" not in variables and re.search(r"\b(sst|temperature|warming|heat)\b", lower):
            raise ValueError("Dataset lacks temperature, so proposed_query must not request SST or warming analysis.")
        if not {"u", "v"}.issubset(variables) and re.search(r"\b(transport|current|circulation|vorticity)\b", lower):
            raise ValueError("Dataset lacks both u and v currents, so proposed_query must not request transport or circulation diagnostics.")
        if (depth_count <= 1 or not self._has_any(variables, {"temp", "salt"})) and re.search(
            r"\b(stratification|density|pycnocline|thermocline|bottom|time-depth|full-depth)\b",
            lower,
        ):
            raise ValueError("Dataset lacks multi-depth temperature/salinity support for bottom, stratification, or time-depth analysis.")

        selected_skills = self._coerce_str_list(proposal.get("selected_skills"))
        if selected_skills:
            available_skills = set(list_skill_ids(self.skills_root))
            unknown = [skill_id for skill_id in selected_skills if skill_id not in available_skills]
            if unknown:
                raise ValueError(f"Analysis proposal selected unknown skill id(s): {', '.join(unknown)}.")

    def _fallback_proposal(
        self,
        user_request: str,
        dataset_context: Dict[str, Any],
        capabilities: Dict[str, Any],
        validation_error: Optional[str],
    ) -> Dict[str, Any]:
        deterministic = self._deterministic_environment_health_proposal(
            user_request=user_request,
            dataset_context=dataset_context,
            capabilities=capabilities,
        )
        if deterministic is not None:
            return deterministic

        dataset = dataset_context.get("dataset") if isinstance(dataset_context.get("dataset"), dict) else {}
        name = str(dataset.get("name") or dataset.get("id") or "the active dataset")
        variables: Set[str] = set(capabilities.get("variables", set()))
        has_temp = self._has_any(variables, {"temp", "temperature"})
        has_oxygen = self._has_any(variables, {"oxygen", "o2"})
        has_salt = "salt" in variables or "salinity" in variables
        depth_count = int(capabilities.get("depth_level_count") or 0)

        if has_temp and has_oxygen and has_salt and depth_count > 1:
            proposed_query = (
                "The current request needs a validated executable tool plan before it can be run. "
                "Please revise the question with the region, time window, and environmental evidence you want prioritized."
            )
            steps = [
                "Review the available environment-health evidence options.",
                "Revise the request with the desired region, period, and risk endpoints.",
                "Generate a fresh executable tool plan after the revised request.",
            ]
            selected_skills = ["ocean_environment_health_assessment"]
            skill_plan = {
                "primary_skill": "ocean_environment_health_assessment",
                "skills_used": selected_skills,
                "planned_tools": [],
                "planned_steps": [],
            }
        elif has_temp:
            proposed_query = (
                f"Using {name}, compute a surface temperature spatial trend and regional-mean temperature trend over "
                f"the active dataset domain and time range."
            )
            steps = [
                "Use surface temperature as the primary available variable.",
                "Compute a surface temperature trend diagnostic over the active dataset coverage.",
                "Report what cannot be assessed without oxygen, chlorophyll, or multi-depth data.",
            ]
            selected_skills = ["ocean_trend_analysis"]
            skill_plan = {
                "primary_skill": "ocean_trend_analysis",
                "skills_used": selected_skills,
                "planned_tools": ["load_dataset", "extract_regional_mean", "compute_trend", "compute_field_trend"],
                "planned_steps": [
                    {"label": "Load surface temperature", "tool": "load_dataset", "purpose": "Open the SST field."},
                    {"label": "Compute trend", "tool": "compute_trend", "purpose": "Estimate regional SST change."},
                    {"label": "Optionally map trend", "tool": "compute_field_trend", "purpose": "Show spatial trend patterns."},
                ],
            }
        else:
            proposed_query = (
                f"Using {name}, summarize the strongest executable environmental diagnostics supported by the active "
                f"dataset variables and propose a focused follow-up analysis."
            )
            steps = [
                "Inspect available variables and coverage.",
                "Choose diagnostics that do not require missing variables.",
                "Report limitations clearly before any management interpretation.",
            ]
            selected_skills = []
            skill_plan = {}

        limitations = self._dataset_limitations(capabilities)
        if validation_error:
            limitations.append(f"Automatic proposal fallback was used because LLM validation failed: {validation_error}")

        return {
            "title": "Suggested Analysis Plan",
            "public_question": user_request.strip(),
            "proposed_query": proposed_query,
            "analysis_steps": steps,
            "expected_outputs": [
                "A short evidence-based summary.",
                "Maps or time-series diagnostics when supported by the selected variables.",
                "Clear limitations about what the dataset cannot prove.",
            ],
            "limitations": limitations or ["The proposal is limited to variables and coverage in the active dataset."],
            "approval_prompt": (
                "I could not build a validated executable tool plan yet. "
                "Please revise the query with more specific analysis details."
            ),
            "selected_skills": selected_skills,
            "skill_plan": skill_plan,
            "executable": False,
            "requires_revision": True,
        }

    def _generate_planner_backed_plan(
        self,
        *,
        user_request: str,
        dataset_context: Dict[str, Any],
        extracted_params: Dict[str, Any],
        additional_context: Dict[str, Any],
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        planner = self._planner or SkillPlanner(**self._planner_kwargs)
        planner_context = self._proposal_planner_context(dataset_context, additional_context)
        planner_max_tokens = max(max_tokens, self.PLANNER_BACKED_MIN_TOKENS)
        plan = planner.generate_plan_for_query(
            user_request=user_request,
            extracted_params=extracted_params,
            additional_context=planner_context,
            max_tokens=planner_max_tokens,
            temperature=temperature,
            allow_multiple_skills=True,
        )
        if not isinstance(plan, dict):
            raise ValueError("Planner did not return a JSON plan.")
        if plan.get("status") == "clarification_needed":
            question = plan.get("clarification_question") or "Planner needs a more specific analysis request."
            raise ValueError(str(question))
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("Planner returned a plan without executable steps.")
        plan = self._canonicalize_planner_backed_plan(plan, user_request=user_request)
        self._validate_executable_plan(plan)
        return plan

    @staticmethod
    def _canonicalize_planner_backed_plan(
        plan: Dict[str, Any],
        *,
        user_request: str = "",
    ) -> Dict[str, Any]:
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return plan
        skills_used = {
            str(skill).strip()
            for skill in plan.get("skills_used", [])
            if str(skill).strip()
        } if isinstance(plan.get("skills_used"), list) else set()
        uses_environment_health = (
            str(plan.get("skill_id") or "").strip() == "ocean_environment_health_assessment"
            or "ocean_environment_health_assessment" in skills_used
        )
        has_hypoxia_step = any(
            isinstance(step, dict) and step.get("tool") == "detect_hypoxia"
            for step in steps
        )
        if not (uses_environment_health or has_hypoxia_step):
            return plan
        if has_hypoxia_step and not uses_environment_health:
            return SkillPlanner._canonicalize_hypoxia_standard_step_params(plan)
        return SkillPlanner._canonicalize_environment_health_standard_step_params(
            plan,
            user_request=user_request,
        )

    def _deterministic_environment_health_proposal(
        self,
        *,
        user_request: str,
        dataset_context: Dict[str, Any],
        capabilities: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self._should_generate_planner_backed_proposal(user_request, capabilities):
            return None
        plan = self._build_deterministic_environment_health_plan(
            user_request=user_request,
            dataset_context=dataset_context,
            capabilities=capabilities,
        )
        if plan is None:
            return None
        try:
            skill_markdown = load_skill_markdown("ocean_environment_health_assessment", self.skills_root)
            validator = SkillPlanner(client=object(), skills_root=self.skills_root)
            validator._validate_plan_for_execution(
                plan,
                expected_skill_id="ocean_environment_health_assessment",
                skill_markdowns={"ocean_environment_health_assessment": skill_markdown},
                user_request=user_request,
                extracted_params={},
                additional_context=dataset_context,
            )
            self._validate_executable_plan(plan)
            proposal = self.build_proposal_from_plan(
                user_request=user_request,
                dataset_context=dataset_context,
                plan=plan,
                capabilities=capabilities,
            )
            self._validate_proposal(proposal, capabilities)
            return proposal
        except Exception:
            return None

    def _build_deterministic_environment_health_plan(
        self,
        *,
        user_request: str,
        dataset_context: Dict[str, Any],
        capabilities: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        load_context = self._environment_health_load_context(user_request, dataset_context)
        if not {"lon_range", "lat_range", "time_range"}.issubset(load_context):
            return None

        requested = SkillPlanner._environment_health_requested_branch_keys(user_request)
        requested.difference_update(SkillPlanner._environment_health_suppressed_branch_keys(user_request))
        requested = self._filter_environment_health_branches_for_capabilities(requested, capabilities)
        if not requested:
            return None

        seed_steps: List[Dict[str, Any]] = []
        for save_as in self._environment_health_seed_steps_for_branches(requested):
            step = SkillPlanner._environment_health_default_step(save_as)
            if step is None:
                continue
            SkillPlanner._apply_environment_health_load_context(
                step,
                load_context,
                overwrite_keys=set(load_context),
            )
            seed_steps.append(step)
        if not seed_steps:
            return None

        plan = {
            "status": "ready",
            "skill_id": "ocean_environment_health_assessment",
            "skills_used": ["ocean_environment_health_assessment"],
            "steps": seed_steps,
        }
        plan = SkillPlanner._canonicalize_environment_health_standard_step_params(
            plan,
            user_request=user_request,
            requested_branch_keys=requested,
        )
        return SkillPlanner._repair_step_dependency_order(plan)

    @staticmethod
    def _environment_health_seed_steps_for_branches(requested: Set[str]) -> List[str]:
        seed_save_as: List[str] = []

        def add(save_as: str) -> None:
            if save_as not in seed_save_as:
                seed_save_as.append(save_as)

        if requested & {
            "bottom_oxygen_trend",
            "bottom_hypoxia_burden",
            "hypoxic_days",
            "hypoxia_statistics",
        }:
            add("bottom_oxygen_field")
        if requested & {"bloom_frequency_change", "bloom_burden", "bloom_event_days"}:
            add("bloom_field")
        if "eutrophication_context" in requested:
            add("chlorophyll_context_field")
        if "sst_trend" in requested:
            add("sst_field")
        if requested & {"heatwave_burden", "heatwave_days"}:
            add("heatwave_field")
        if "upwelling_days" in requested:
            add("upwelling_field")
        if "stratification_strength_change" in requested:
            add("temp_field")
            add("salt_field")

        return seed_save_as

    @staticmethod
    def _filter_environment_health_branches_for_capabilities(
        requested: Set[str],
        capabilities: Dict[str, Any],
    ) -> Set[str]:
        supported = set(requested)
        if not capabilities.get("has_oxygen"):
            supported.difference_update({
                "bottom_oxygen_trend",
                "bottom_hypoxia_burden",
                "hypoxic_days",
                "hypoxia_statistics",
            })
        if not capabilities.get("has_temperature"):
            supported.difference_update({"sst_trend", "heatwave_burden", "heatwave_days", "upwelling_days"})
        if not capabilities.get("has_chlorophyll"):
            supported.difference_update({
                "bloom_frequency_change",
                "bloom_burden",
                "bloom_event_days",
                "eutrophication_context",
            })
        if not (
            capabilities.get("has_temperature")
            and capabilities.get("has_salinity")
            and capabilities.get("has_multi_depth")
        ):
            supported.discard("stratification_strength_change")
        return supported

    def _environment_health_load_context(
        self,
        user_request: str,
        dataset_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        region_bounds = self._region_bounds_from_request(user_request)
        if region_bounds is None:
            region_bounds = self._dataset_bounds(dataset_context)
        if region_bounds is not None:
            context["lon_range"], context["lat_range"] = region_bounds

        time_range = SkillPlanner._time_range_from_request_text(user_request)
        if time_range is None:
            time_range = self._dataset_time_range(dataset_context)
        if time_range is not None:
            context["time_range"] = time_range

        season_filter = SkillPlanner._infer_requested_season_filter(user_request)
        if season_filter is not None:
            context["season_filter"] = season_filter
        return context

    @staticmethod
    def _region_bounds_from_request(user_request: str) -> Optional[tuple[List[float], List[float]]]:
        lowered = str(user_request or "").lower()
        region_bounds = {
            "yellow sea": ([119.0, 126.0], [33.0, 39.0]),
            "bohai sea": ([117.5, 121.5], [37.0, 41.0]),
            "east china sea": ([121.0, 126.0], [27.0, 32.0]),
            "south china sea": ([105.0, 122.0], [5.0, 23.0]),
        }
        aliases = [
            ("yellow sea", ("yellow sea", "黄海")),
            ("bohai sea", ("bohai sea", "bohai", "渤海")),
            ("east china sea", ("east china sea", "东海")),
            ("south china sea", ("south china sea", "南海")),
        ]
        for canonical, names in aliases:
            if any(name in lowered or name in user_request for name in names):
                return region_bounds[canonical]
        if any(
            phrase in lowered or phrase in user_request
            for phrase in (
                "china coastal seas",
                "china's coastal seas",
                "chinese coastal seas",
                "china seas",
                "中国近海",
                "中国沿海",
                "中国海域",
            )
        ):
            return [105.0, 126.0], [5.0, 42.0]
        return None

    @staticmethod
    def _dataset_bounds(dataset_context: Dict[str, Any]) -> Optional[tuple[List[float], List[float]]]:
        dataset = dataset_context.get("dataset") if isinstance(dataset_context.get("dataset"), dict) else {}
        spatial = dataset.get("spatial_extent") if isinstance(dataset, dict) else None
        if not isinstance(spatial, dict):
            return None
        lon = spatial.get("lon_range") or spatial.get("lon") or spatial.get("longitude")
        lat = spatial.get("lat_range") or spatial.get("lat") or spatial.get("latitude")
        lon_range = AnalysisProposalSynthesizer._coerce_range(lon)
        lat_range = AnalysisProposalSynthesizer._coerce_range(lat)
        if lon_range is None or lat_range is None:
            return None
        return lon_range, lat_range

    @staticmethod
    def _dataset_time_range(dataset_context: Dict[str, Any]) -> Optional[List[str]]:
        dataset = dataset_context.get("dataset") if isinstance(dataset_context.get("dataset"), dict) else {}
        temporal = dataset.get("temporal_extent") if isinstance(dataset, dict) else None
        if isinstance(temporal, dict):
            start = str(temporal.get("start") or "").strip()
            end = str(temporal.get("end") or "").strip()
            if start and end:
                return [start[:10], end[:10]]
        if isinstance(temporal, list) and len(temporal) >= 2:
            start = str(temporal[0] or "").strip()
            end = str(temporal[1] or "").strip()
            if start and end:
                return [start[:10], end[:10]]
        return None

    @staticmethod
    def _coerce_range(value: Any) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            first = float(value[0])
            second = float(value[1])
        except (TypeError, ValueError):
            return None
        return [min(first, second), max(first, second)]

    @staticmethod
    def _proposal_planner_context(
        dataset_context: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        context = dict(additional_context or {})
        for key, value in (dataset_context or {}).items():
            context.setdefault(key, value)
        return context

    def build_proposal_from_plan(
        self,
        *,
        user_request: str,
        dataset_context: Dict[str, Any],
        plan: Dict[str, Any],
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        capabilities = capabilities or self._dataset_capabilities(dataset_context)
        skill_plan = self.compact_plan_for_proposal(plan)
        profile_id = (
            self.INTEGRATED_ASSESSMENT_PROFILE_ID
            if "ocean_environment_health_assessment" in set(skill_plan.get("skills_used", []))
            else None
        )
        limitations = self._dataset_limitations(capabilities)
        if profile_id == self.INTEGRATED_ASSESSMENT_PROFILE_ID:
            limitations.extend([
                "This is environmental risk evidence, not a direct fish-yield or economic-profit forecast.",
                "Future suitability is inferred from available historical evidence, not from a future climate projection.",
            ])
        if profile_id and not skill_plan.get("synthesis_profile_id"):
            skill_plan["synthesis_profile_id"] = profile_id

        proposed_query = (
            "Execute the attached validated tool plan to answer: "
            f"{str(user_request or '').strip()}"
        )
        return {
            "title": self._proposal_title(user_request, plan, skill_plan),
            "public_question": str(user_request or "").strip(),
            "proposed_query": proposed_query,
            "analysis_steps": self._proposal_analysis_steps(skill_plan),
            "expected_outputs": self._proposal_expected_outputs(plan),
            "limitations": limitations or ["The proposal is limited to variables and coverage in the active dataset."],
            "approval_prompt": "If this validated tool plan looks right, reply OK or click Run proposed analysis.",
            "synthesis_profile_id": profile_id,
            "selected_skills": list(skill_plan.get("skills_used", [])),
            "skill_plan": skill_plan,
            "plan": plan,
            "executable": True,
            "requires_revision": False,
        }

    def compact_plan_for_proposal(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        steps = plan.get("steps") if isinstance(plan, dict) else []
        planned_tools: List[str] = []
        planned_steps: List[Dict[str, str]] = []
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                tool = str(step.get("tool") or "").strip()
                if tool and tool not in planned_tools:
                    planned_tools.append(tool)
                save_as = str(step.get("save_as") or step.get("step_id") or tool or "step").strip()
                planned_steps.append({
                    "label": save_as.replace("_", " ").strip().title(),
                    "tool": tool,
                    "purpose": self._proposal_step_purpose(step),
                })
        skills_used = self._coerce_str_list(plan.get("skills_used")) if isinstance(plan, dict) else []
        primary_skill = str(plan.get("skill_id") or (skills_used[0] if skills_used else "")).strip()
        if primary_skill and primary_skill not in skills_used:
            skills_used.insert(0, primary_skill)
        return {
            "primary_skill": primary_skill,
            "skills_used": skills_used,
            "planned_tools": planned_tools,
            "planned_steps": planned_steps,
        }

    @staticmethod
    def _proposal_step_purpose(step: Dict[str, Any]) -> str:
        tool = str(step.get("tool") or "")
        save_as = str(step.get("save_as") or step.get("step_id") or "result")
        if tool == "load_dataset":
            params = step.get("params") if isinstance(step.get("params"), dict) else {}
            variable = params.get("variable") or "field"
            return f"Load {variable} evidence for downstream diagnostics."
        if tool == "compute_trend":
            return f"Estimate the trend for {save_as.replace('_', ' ')}."
        if tool == "detect_hypoxia":
            return "Detect hypoxia events from bottom oxygen evidence."
        if tool == "compute_event_summary_map":
            return f"Build a spatial exposure map for {save_as.replace('_', ' ')}."
        if tool == "compute_event_statistics":
            return "Summarize event counts and changes over time."
        if tool == "compute_vertical_stability_timeseries":
            return "Summarize water-column stability as stratification evidence."
        return f"Produce {save_as.replace('_', ' ')}."

    def _proposal_title(
        self,
        user_request: str,
        plan: Dict[str, Any],
        skill_plan: Dict[str, Any],
    ) -> str:
        if "ocean_environment_health_assessment" in set(skill_plan.get("skills_used", [])):
            return f"Ocean Environment Health Assessment for {self._infer_public_region(user_request)}"
        primary = str(skill_plan.get("primary_skill") or plan.get("skill_id") or "analysis").replace("_", " ").title()
        return f"Planner-Generated {primary} Plan"

    @staticmethod
    def _proposal_analysis_steps(skill_plan: Dict[str, Any]) -> List[str]:
        steps = skill_plan.get("planned_steps")
        if isinstance(steps, list) and steps:
            return [
                f"{step.get('label')}: {step.get('purpose')}"
                for step in steps[:8]
                if isinstance(step, dict) and (step.get("label") or step.get("purpose"))
            ]
        return ["Review and approve the planner-generated executable tool chain."]

    @staticmethod
    def _proposal_expected_outputs(plan: Dict[str, Any]) -> List[str]:
        steps = plan.get("steps") if isinstance(plan, dict) else []
        result_ids = [
            str(step.get("save_as") or step.get("step_id"))
            for step in steps
            if isinstance(step, dict) and (step.get("save_as") or step.get("step_id"))
        ]
        if not result_ids:
            return ["Validated scientific evidence from the approved tool chain."]
        return [
            "Validated intermediate evidence from the approved tool chain.",
            "Maps, time-series, trends, or event summaries produced by the planned tools when applicable.",
            "Final synthesis grounded in completed results: " + ", ".join(result_ids[-4:]),
        ]

    def _dataset_capabilities(self, dataset_context: Dict[str, Any]) -> Dict[str, Any]:
        dataset = dataset_context.get("dataset") if isinstance(dataset_context.get("dataset"), dict) else {}
        raw_variables = dataset.get("variables") if isinstance(dataset, dict) else []
        variable_names = dataset.get("variable_names") if isinstance(dataset, dict) else {}
        variables = {
            str(item).strip().lower()
            for item in (raw_variables if isinstance(raw_variables, list) else [])
            if str(item).strip()
        }
        if isinstance(variable_names, dict):
            variables.update(str(key).strip().lower() for key in variable_names if str(key).strip())
        depth_levels = dataset.get("depth_levels") if isinstance(dataset, dict) else []
        depth_count = len(depth_levels) if isinstance(depth_levels, list) else 0
        return {
            "dataset_name": dataset.get("name") or dataset.get("id") or "active dataset",
            "variables": sorted(variables),
            "has_temperature": self._has_any(variables, {"temp", "temperature"}),
            "has_salinity": self._has_any(variables, {"salt", "salinity"}),
            "has_oxygen": self._has_any(variables, {"oxygen", "o2"}),
            "has_chlorophyll": self._has_any(variables, {"chlorophyll", "chl"}),
            "has_currents": {"u", "v"}.issubset(variables),
            "depth_level_count": depth_count,
            "has_multi_depth": depth_count > 1,
            "spatial_extent": dataset.get("spatial_extent"),
            "temporal_extent": dataset.get("temporal_extent"),
            "depth_range": dataset.get("depth_range"),
            "resolution": dataset.get("resolution"),
        }

    def _analysis_templates(self, capabilities: Dict[str, Any]) -> List[Dict[str, Any]]:
        templates: List[Dict[str, Any]] = [
            {
                "id": "public_capability_translation",
                "use_when": "The user asks broadly what can be analyzed with the available data.",
                "pattern": "Translate the public question into one concrete region/time/variable analysis.",
            }
        ]
        if capabilities.get("has_temperature") and capabilities.get("has_oxygen") and capabilities.get("has_salinity") and capabilities.get("has_multi_depth"):
            templates.append(
                {
                    "id": "aquaculture_marine_ranching_risk",
                    "use_when": "fish farming, aquaculture, marine ranching, environmental health, hypoxia risk",
                    "pattern": (
                        "Use bottom oxygen/hypoxia as endpoints; SST, heatwaves, stratification, and supported "
                        "chlorophyll/bloom screening as vulnerability or ecological-pressure evidence; "
                        "management guidance must be evidence-bounded."
                    ),
                    "skills_used": ["ocean_environment_health_assessment"],
                    "synthesis_profile_id": self.INTEGRATED_ASSESSMENT_PROFILE_ID,
                    "planned_tools": [
                        "load_dataset",
                        "compute_area_weighted_mean",
                        "compute_trend",
                        "detect_hypoxia",
                        "compute_event_statistics",
                        "compute_event_summary_map",
                        "assemble_dataset",
                        "compute_density",
                        "compute_vertical_stability_timeseries",
                    ],
                }
            )
        if capabilities.get("has_temperature"):
            templates.append(
                {
                    "id": "surface_temperature_variability",
                    "use_when": "ENSO, warming, SST trend, climate variability, EOF-style surface analysis",
                    "pattern": "Use surface temperature trend, anomaly, climatology, EOF, or regional mean diagnostics.",
                    "skills_used": ["ocean_trend_analysis"],
                }
            )
        if capabilities.get("has_currents"):
            templates.append(
                {
                    "id": "circulation_transport_context",
                    "use_when": "currents, exchange, circulation, transport, vorticity",
                    "pattern": "Use u/v-supported circulation diagnostics only when the public question needs physical transport evidence.",
                }
            )
        return templates

    @staticmethod
    def _should_generate_planner_backed_proposal(
        user_request: str,
        capabilities: Dict[str, Any],
    ) -> bool:
        lower = str(user_request or "").lower()
        if SkillPlanner._policy_query_requires_new_environment_evidence(user_request):
            requested = SkillPlanner._environment_health_requested_branch_keys(user_request)
            requested.difference_update(SkillPlanner._environment_health_suppressed_branch_keys(user_request))
            requested = AnalysisProposalSynthesizer._filter_environment_health_branches_for_capabilities(
                requested,
                capabilities,
            )
            return bool(requested)

        public_environment_health = re.search(
            r"\b(aquaculture|marine ranching|fish farming|fishery|fisheries|"
            r"environmental risk|environmental health|marine health|suitab|remain suitable)\b|"
            r"养鱼|海洋牧场|水产|渔业|适合|风险|环境健康",
            lower,
        )
        return bool(
            public_environment_health
            and capabilities.get("has_temperature")
            and capabilities.get("has_oxygen")
            and capabilities.get("has_salinity")
            and capabilities.get("has_multi_depth")
        )

    @staticmethod
    def _validate_executable_plan(plan: Dict[str, Any]) -> None:
        if not isinstance(plan, dict):
            raise ValueError("Planner did not return a JSON plan.")

        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("Planner returned a plan without executable steps.")

        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                raise ValueError(f"Planner returned a non-object step at index {index}.")
            tool_name = step.get("tool")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(f"Planner step {index} is missing a tool name.")
            if get_tool_contract(tool_name) is None:
                raise ValueError(f"Planner step {index} uses unknown tool '{tool_name}'.")
            params = step.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(f"Planner step {index} params must be a JSON object.")
            errors = [
                issue["message"]
                for issue in validate_tool_params(tool_name, params)
                if issue.get("level") == "error"
            ]
            if errors:
                raise ValueError(f"Planner step {index} failed tool-contract validation: {'; '.join(errors)}")

    def _skill_backed_template_proposal(
        self,
        *,
        user_request: str,
        dataset_context: Dict[str, Any],
        capabilities: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Deprecated compatibility hook; planner-backed proposals own this path."""
        return None

    def _aquaculture_template_proposal(
        self,
        user_request: str,
        dataset_context: Dict[str, Any],
        capabilities: Dict[str, Any],
    ) -> Dict[str, Any]:
        region = self._infer_public_region(user_request)
        time_window = self._proposal_time_window(dataset_context, default="2011-2022")
        proposed_query = (
            f"For the {region} during {time_window}, run an ocean environment health assessment for aquaculture "
            "and marine-ranching environmental risk. Use bottom hypoxia burden, hypoxic days, and bottom dissolved "
            "oxygen trend as primary endpoint evidence; use SST warming, marine heatwave exposure, stratification, "
            "and supported chlorophyll/bloom screening as risk-amplifier or ecological-pressure evidence. Identify "
            "risk hotspots and named places from the completed spatial evidence itself, such as map extrema or "
            "ranked event cells, and leave aquaculture suitability, "
            "economic-development implications, environmental-protection implications, and management guidance to the "
            "summary synthesis. Do not generate a fixed environment-health report card. Treat chlorophyll/bloom evidence "
            "as ecological-pressure or red-tide/HAB screening only; without nutrient/source/pathway data, do not attribute "
            "risk to pollution sources or prescribe direct source-control actions."
        )
        skill_plan = {
            "primary_skill": "ocean_environment_health_assessment",
            "skills_used": ["ocean_environment_health_assessment"],
            "synthesis_profile_id": self.INTEGRATED_ASSESSMENT_PROFILE_ID,
            "planned_tools": [
                "load_dataset",
                "compute_area_weighted_mean",
                "compute_trend",
                "detect_hypoxia",
                "compute_event_statistics",
                "compute_event_summary_map",
                "detect_heatwaves",
                "detect_algal_blooms",
                "assemble_dataset",
                "compute_density",
                "compute_vertical_stability_timeseries",
            ],
            "planned_steps": [
                {
                    "label": "Bottom oxygen evidence",
                    "tool": "load_dataset",
                    "purpose": "Load bottom dissolved oxygen and compute the bottom-oxygen trend.",
                },
                {
                    "label": "Hypoxia burden evidence",
                    "tool": "detect_hypoxia",
                    "purpose": "Detect hypoxic exposure and summarize hypoxic days or oxygen-deficit burden.",
                },
                {
                    "label": "Thermal vulnerability",
                    "tool": "detect_heatwaves",
                    "purpose": "Use surface temperature trends and marine heatwave exposure as vulnerability or timing evidence.",
                },
                {
                    "label": "Stratification vulnerability",
                    "tool": "compute_vertical_stability_timeseries",
                    "purpose": "Use density-derived stratification trends as oxygen-risk context.",
                },
                {
                    "label": "Bloom screening",
                    "tool": "detect_algal_blooms",
                    "purpose": "Use chlorophyll-supported bloom evidence as ecological-pressure screening when available.",
                },
            ],
        }
        return {
            "title": f"Aquaculture and Marine-Ranching Environmental Risk in the {region}",
            "public_question": user_request.strip(),
            "proposed_query": proposed_query,
            "analysis_steps": [
                "Use the environment-health assessment skill as the execution backbone.",
                "Treat bottom hypoxia and low bottom oxygen as the primary environmental endpoints.",
                "Use SST warming, heatwaves, stratification, and supported chlorophyll/bloom evidence as vulnerability, timing, or ecological-pressure context.",
                "Use completed spatial maps and hotspot coordinates as the only basis for named regional risk claims.",
                "Generate hotspot/risk ranking and evidence-bounded management guidance after the data analysis.",
            ],
            "expected_outputs": [
                "Bottom oxygen trend and hypoxia-burden evidence.",
                "SST, heatwave, stratification, and supported bloom/chlorophyll diagnostics.",
                "Risk hotspot ranking for aquaculture and marine-ranching suitability based on computed maps.",
                "Management guidance that distinguishes supported evidence from screening suggestions.",
            ],
            "limitations": self._dataset_limitations(capabilities)
            + [
                "This is not a direct fish-yield or economic-profit forecast.",
                "Future suitability is inferred from 2011-2022 environmental risk evidence, not from a future climate projection.",
                "Acidification, typhoon/storm-surge/wave exposure, nutrient sources, species tolerance, and in-situ validation are reported as gaps unless those data are supplied.",
            ],
            "approval_prompt": "If this skill-backed plan looks right, reply OK or click Run proposed analysis.",
            "synthesis_profile_id": self.INTEGRATED_ASSESSMENT_PROFILE_ID,
            "selected_skills": ["ocean_environment_health_assessment"],
            "skill_plan": skill_plan,
        }

    def _normalize_skill_plan(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        normalized: Dict[str, Any] = {}
        primary_skill = value.get("primary_skill")
        if isinstance(primary_skill, str) and primary_skill.strip():
            normalized["primary_skill"] = primary_skill.strip()
        skills_used = self._coerce_str_list(value.get("skills_used"))
        if skills_used:
            normalized["skills_used"] = skills_used
        synthesis_profile_id = value.get("synthesis_profile_id")
        if isinstance(synthesis_profile_id, str) and synthesis_profile_id.strip():
            normalized["synthesis_profile_id"] = synthesis_profile_id.strip()
        planned_tools = self._coerce_str_list(value.get("planned_tools"))
        if planned_tools:
            normalized["planned_tools"] = planned_tools
        planned_steps = []
        raw_steps = value.get("planned_steps")
        if isinstance(raw_steps, list):
            for step in raw_steps:
                if not isinstance(step, dict):
                    continue
                label = str(step.get("label") or "").strip()
                tool = str(step.get("tool") or "").strip()
                purpose = str(step.get("purpose") or "").strip()
                if label or tool or purpose:
                    planned_steps.append({"label": label, "tool": tool, "purpose": purpose})
        if planned_steps:
            normalized["planned_steps"] = planned_steps
        return normalized

    def _infer_public_region(self, user_request: str) -> str:
        lower = user_request.lower()
        if "yellow sea" in lower or "黄海" in user_request:
            return "Yellow Sea"
        if "bohai sea" in lower or "bohai" in lower or "渤海" in user_request:
            return "Bohai Sea"
        if "east china sea" in lower or "东海" in user_request:
            return "East China Sea"
        if "south china sea" in lower or "南海" in user_request:
            return "South China Sea"
        if "china seas" in lower or "中国海" in user_request:
            return "China Seas"
        return "China Seas"

    def _proposal_time_window(self, dataset_context: Dict[str, Any], *, default: str) -> str:
        dataset = dataset_context.get("dataset") if isinstance(dataset_context.get("dataset"), dict) else {}
        temporal = dataset.get("temporal_extent") if isinstance(dataset, dict) else None
        if isinstance(temporal, dict):
            start = str(temporal.get("start") or "")[:4]
            end = str(temporal.get("end") or "")[:4]
            if start and end:
                return f"{start}-{end}"
        if isinstance(temporal, list) and len(temporal) >= 2:
            start = str(temporal[0])[:4]
            end = str(temporal[1])[:4]
            if start and end:
                return f"{start}-{end}"
        return default

    def _compact_skill_capabilities(self) -> List[Dict[str, Any]]:
        hints = load_all_planner_hints(self.skills_root)
        preferred = {
            "ocean_environment_health_assessment",
            "ocean_hypoxia_detection",
            "ocean_trend_analysis",
            "ocean_spatial_field_analysis",
            "ocean_timeseries",
            "ocean_eof_analysis",
            "ocean_stratification_diagnostics",
            "ocean_transport_analysis",
            "ocean_derived_hovmoller_analysis",
        }
        selected_ids = [skill_id for skill_id in sorted(hints) if skill_id in preferred]
        return [
            {
                "skill_id": skill_id,
                "intent_summary": str(hints[skill_id].get("intent_summary") or ""),
                "positive_query_examples": list(hints[skill_id].get("positive_query_examples", [])[:2]),
                "required_entities": list(hints[skill_id].get("required_entities", [])),
                "result_types": list(hints[skill_id].get("result_types", [])),
            }
            for skill_id in selected_ids
        ]

    def _dataset_limitations(self, capabilities: Dict[str, Any]) -> List[str]:
        limitations: List[str] = []
        if not capabilities.get("has_oxygen"):
            limitations.append("No oxygen variable is available, so hypoxia or dissolved-oxygen burden cannot be assessed.")
        if not capabilities.get("has_chlorophyll"):
            limitations.append("No chlorophyll variable is available, so eutrophication or bloom evidence can only be discussed as unavailable.")
        if not capabilities.get("has_multi_depth"):
            limitations.append("Only one or no depth level is available, so bottom-layer and stratification diagnostics are not supported.")
        if not capabilities.get("has_currents"):
            limitations.append("Both u and v currents are not available, so transport or circulation diagnostics are not supported.")
        limitations.append("No nutrient/source, pH/alkalinity, wave, storm-surge, typhoon, species-tolerance, production, or economic data are assumed unless explicitly present.")
        return limitations

    @staticmethod
    def _coerce_str_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        result: List[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned:
                result.append(cleaned)
        return result

    @staticmethod
    def _has_any(values: Set[str], candidates: Set[str]) -> bool:
        return any(candidate in values for candidate in candidates)
