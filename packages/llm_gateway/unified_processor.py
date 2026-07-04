from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from packages.llm_gateway.skill_planner import SkillPlanner
from packages.llm_gateway.config import load_model_name
from packages.llm_gateway.processor_prompts import UNIFIED_ROUTER_SYSTEM_PROMPT


logger = logging.getLogger(__name__)


def _has_lon_lat_bounds(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    lon_range = payload.get("lon_range") or payload.get("longitude_range")
    lat_range = payload.get("lat_range") or payload.get("latitude_range")
    if _looks_like_range(lon_range) and _looks_like_range(lat_range):
        return True
    region = payload.get("region")
    if isinstance(region, dict):
        lon_range = region.get("lon_range") or region.get("longitude_range")
        lat_range = region.get("lat_range") or region.get("latitude_range")
        if _looks_like_range(lon_range) and _looks_like_range(lat_range):
            return True
    region_bounds = payload.get("region_bounds") or payload.get("current_region_bounds")
    if isinstance(region_bounds, dict):
        lon_range = region_bounds.get("lon") or region_bounds.get("lon_range")
        lat_range = region_bounds.get("lat") or region_bounds.get("lat_range")
        return _looks_like_range(lon_range) and _looks_like_range(lat_range)
    return False


def _looks_like_range(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


class UnifiedQueryProcessor:
    DEFAULT_MODEL = load_model_name("QUERY_ROUTER_MODEL", default=SkillPlanner.DEFAULT_MODEL)
    LOW_CONFIDENCE_THRESHOLD = 0.6

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
        model: str = DEFAULT_MODEL,
        planner: Optional[SkillPlanner] = None,
        skills_root: Optional[str] = None,
        trust_env: bool = False,
        base_url: Optional[str] = None,
        request_retries: int = 2,
    ):
        self.api_key = api_key
        self.client = client
        self.model = model
        self.planner = planner
        self.skills_root = skills_root
        self.trust_env = trust_env
        self.base_url = base_url
        self.request_retries = request_retries

    def process(
        self,
        query: str,
        dataset_context: Dict[str, Any],
        conversation_context: Optional[Dict[str, Any]] = None,
        extracted_params: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 1200,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        helper = self._helper()
        client = self._get_client(helper)
        payload = {
            "query": query,
            "dataset_context": dataset_context,
            "conversation_context": conversation_context or {},
        }
        prior_queries_text = (additional_context or {}).get("prior_queries_text")
        if isinstance(prior_queries_text, str) and prior_queries_text.strip():
            payload["prior_queries_text"] = prior_queries_text
        conversation_memory = (additional_context or {}).get("conversation_memory")
        if isinstance(conversation_memory, dict):
            payload["conversation_memory"] = conversation_memory
        response = helper._create_message(
            client=client,
            max_tokens=max_tokens,
            temperature=temperature,
            system=UNIFIED_ROUTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}],
            request_name="unified_query_processor",
            json_response=True,
        )
        text = helper._extract_response_text(response)
        try:
            decision = helper._parse_json_response(text)
        except Exception:
            logger.exception("Router returned an unparseable decision; using deterministic fallback.")
            decision = self._fallback_decision(query=query, dataset_context=dataset_context)
        return self._normalize_decision(decision, query=query)

    def _normalize_decision(self, decision: Dict[str, Any], query: str) -> Dict[str, Any]:
        routing_mode = decision.get("routing_mode")
        needs_dataset = decision.get("needs_dataset")
        if routing_mode not in {"dataset_analysis", "general_answer"} and isinstance(needs_dataset, bool):
            routing_mode = "dataset_analysis" if needs_dataset else "general_answer"

        extracted_entities = (
            decision.get("extracted_entities")
            if isinstance(decision.get("extracted_entities"), dict)
            else {}
        )
        extracted_entities = self._strip_parameter_entities(extracted_entities)

        normalized: Dict[str, Any] = {
            "action": decision.get("action"),
            "routing_mode": routing_mode,
            "needs_dataset": bool(needs_dataset) if isinstance(needs_dataset, bool) else routing_mode == "dataset_analysis",
            "confidence": self._coerce_confidence(decision.get("confidence")),
            "reason": str(decision.get("reason") or "").strip(),
            "inferred_intent": str(decision.get("inferred_intent") or query).strip(),
            "extracted_entities": extracted_entities,
        }

        valid_modes = {"dataset_analysis", "general_answer"}
        if normalized["routing_mode"] not in valid_modes:
            raise ValueError(f"Unsupported routing mode: {normalized['routing_mode']}")

        if normalized["routing_mode"] == "dataset_analysis":
            normalized["action"] = "route_to_executor"
            normalized["needs_dataset"] = True
        else:
            normalized["action"] = "handle_directly"
            normalized["needs_dataset"] = False
        return normalized

    @staticmethod
    def _strip_parameter_entities(extracted_entities: Dict[str, Any]) -> Dict[str, Any]:
        blocked = {
            "lon_range",
            "lat_range",
            "longitude_range",
            "latitude_range",
            "time_range",
            "date_range",
            "temporal_range",
            "region_name",
            "region_source",
            "named_regions",
            "known_named_region_bounds",
            "region",
            "region_bounds",
            "current_region_bounds",
            "selected_point",
            "transect_points",
            "drawn_transect_points",
            "mask_polygon",
            "drawn_polygon_points",
            "workspace_selection",
        }
        return {
            key: value
            for key, value in dict(extracted_entities or {}).items()
            if key not in blocked
        }

    def _helper(self) -> SkillPlanner:
        return SkillPlanner(
            api_key=self.api_key,
            model=self.model,
            skills_root=self.skills_root,
            client=self.client,
            trust_env=self.trust_env,
            base_url=self.base_url,
            request_retries=self.request_retries,
        )

    def _get_client(self, helper: SkillPlanner) -> Any:
        if self.client is not None:
            return self.client
        return helper._get_client()

    def _fallback_decision(self, *, query: str, dataset_context: Dict[str, Any]) -> Dict[str, Any]:
        needs_dataset = self._fallback_needs_dataset(query, dataset_context)
        return {
            "action": "route_to_executor" if needs_dataset else "handle_directly",
            "routing_mode": "dataset_analysis" if needs_dataset else "general_answer",
            "needs_dataset": needs_dataset,
            "confidence": 0.55,
            "reason": "Router response was not valid JSON, so a deterministic fallback classified this request.",
            "inferred_intent": query,
            "extracted_entities": {},
        }

    @staticmethod
    def _fallback_needs_dataset(query: str, dataset_context: Dict[str, Any]) -> bool:
        lowered = (query or "").lower()
        dataset = dataset_context.get("dataset") if isinstance(dataset_context, dict) else {}
        if not isinstance(dataset, dict):
            dataset = {}

        dataset_markers = (
            "this dataset", "current dataset", "active dataset", "dataset config",
            "dataset metadata", "dataset variables", "available variables",
            "data path", "data_path", "zarr", "zarr store", "backend",
            "coverage", "spatial extent", "temporal extent", "depth range",
            "数据集", "当前数据", "这个数据", "变量", "时间范围", "空间范围",
            "深度范围", "数据路径", "数据后端",
        )
        if any(marker in lowered for marker in dataset_markers):
            return True

        for field in ("id", "name"):
            value = str(dataset.get(field) or "").strip().lower()
            if value and value in lowered:
                return True

        variables = list(dataset.get("variables") or [])
        variable_names = dataset.get("variable_names")
        if isinstance(variable_names, dict):
            variables.extend(str(key) for key in variable_names)
            variables.extend(str(value) for value in variable_names.values())
        variable_hit = False
        for variable in variables:
            token = str(variable).strip().lower()
            if not token:
                continue
            if re.fullmatch(r"[a-z0-9_]+", token):
                if re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", lowered):
                    variable_hit = True
                    break
            elif token in lowered:
                variable_hit = True
                break

        analysis_markers = (
            "compute", "calculate", "plot", "map", "show", "visualize", "extract",
            "trend", "correlation", "profile", "section", "time series", "timeseries",
            "mean", "average", "anomaly", "event", "计算", "画", "绘制", "展示",
            "可视化", "提取", "趋势", "相关", "剖面", "断面", "时间序列", "平均", "异常", "事件",
        )
        return variable_hit and any(marker in lowered for marker in analysis_markers)

    @staticmethod
    def _coerce_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _clean_optional_string(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned or None

    @staticmethod
    def _coerce_str_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned:
                normalized.append(cleaned)
        return normalized
