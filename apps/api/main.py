from __future__ import annotations

import asyncio
import calendar
import json
import sys
import time
import logging
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple
from uuid import uuid4

import numpy as np
import xarray as xr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from apps.api.failures import FailureKind, _public_failure, _public_failure_message
from apps.api.request_geometry import (
    hydrate_workspace_geometry_extracted_params as _hydrate_workspace_geometry_extracted_params,
)
from domain.ocean.analysis.profile.extract import (
    identify_mixed_layer_depth,
    identify_pycnocline_depth,
    identify_thermocline_depth,
)
from domain.ocean.analysis.spatial.analysis import compute_spatial_field
from domain.ocean.analysis.timeseries.extract import compute_layer_mean, extract_regional_mean
from domain.ocean.data_access.assemble import assemble_dataset
from domain.ocean.data_access.load import load_dataset
from domain.ocean.data_access.partitioned import PartitionedDataArray
from domain.ocean.dask_utils import is_dask_backed
from domain.ocean.diagnostics.compute import compute_density
from domain.ocean.result_payload import (
    as_numeric_array,
    json_safe_array,
    matrix_sample_indices,
    workspace_max_matrix_points,
)
from domain.ocean.visualization.contourf import render_contourf_image, render_multiregion_contourf_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packages.agent_core.executor import SkillExecutor
from packages.harness import OceanHarnessExecutor
from packages.harness.planner_agent import PlannerAgent
from packages.llm_gateway import (
    AnalysisProposalSynthesizer,
    ResultSynthesizer,
    SkillPlanner,
    WebAnswerSynthesizer,
    UnifiedQueryProcessor,
)
from packages.llm_gateway.config import load_llm_api_key, load_llm_base_url, load_model_name
from packages.runtime import get_active_dataset_context, get_active_dataset_public_config
from packages.web_search import WebSearchService
from packages.conversation_memory import MemoryAgent, record_turn_memory
from apps.api.report_export import ConversationReportRequest, export_report_response


REPORT_PATH = PROJECT_ROOT / "skill_execution_report.json"
CONVERSATION_STATE: Dict[str, Dict[str, Any]] = {}
PROJECT_MEMORY_STATE: Dict[str, Any] = {}
REFERENCE_MEMORY_STATE: Dict[str, Any] = {}
_POST_EXEC_POOL = ThreadPoolExecutor(max_workers=3)
_logger = logging.getLogger("ocean_api")
OCEAN_INTEGRATED_ASSESSMENT_PROFILE_ID = "ocean_integrated_assessment"
LLM_SYNTHESIS_MAX_ATTEMPTS = 2
HARNESS_PLANNING_TIMEOUT_S = float(os.getenv("HARNESS_PLANNING_TIMEOUT_S", "0"))
_INTEGRATED_ASSESSMENT_PROFILE_RE = re.compile(
    r"\b(aquaculture|marine ranching|fish farming|fishery|fisheries|"
    r"environmental risk|marine health|environmental health|suitab|remain suitable)\b|"
    r"养鱼|海洋牧场|水产|渔业|适合|风险|环境健康",
    re.IGNORECASE,
)
_GENERAL_SEARCH_TAG_RE = re.compile(r"^\s*<search>(.*?)</search>\s*$", re.DOTALL | re.IGNORECASE)


def _run_tool_partition_aware(tool_name: str, tool_func, **params):
    return tool_func(**params)

TOOL_TITLES: Dict[str, str] = {
    "get_dataset_info": "Dataset info",
    "load_dataset": "Load data",
    "compute_spatial_field": "Spatial field",
    "compute_spatial_vorticity_map": "Vorticity map",
    "extract_regional_mean": "Regional mean",
    "extract_point_timeseries": "Point time series",
    "extract_timeseries": "Time series",
    "compute_area_weighted_mean": "Area-weighted mean",
    "compute_trend": "Trend analysis",
    "compute_climatology": "Climatology",
    "compute_anomaly": "Anomaly analysis",
    "compute_histogram": "Distribution",
    "extract_vertical_profile": "Vertical profile",
    "identify_mixed_layer_depth": "Mixed-layer depth",
    "identify_thermocline_depth": "Thermocline depth",
    "identify_pycnocline_depth": "Pycnocline depth",
    "compute_layer_mean": "Layer mean",
    "compute_hovmoller": "Hovmoller",
    "compute_ts_diagram": "T-S diagram",
    "compute_watermass_event_association": "Water-mass classification",
    "build_watermass_tile_map": "Water-mass tile map",
    "build_watermass_ts_diagram": "Water-mass T-S diagram",
    "perform_eof_analysis": "EOF analysis",
    "compute_density": "Density field",
    "compute_derived_field": "Derived field",
    "compute_kinetic_energy": "Kinetic energy",
    "detect_heatwaves": "Heatwave detection",
    "detect_hypoxia": "Hypoxia detection",
    "detect_algal_blooms": "Bloom detection",
    "detect_upwelling": "Upwelling detection",
    "detect_eutrophication": "Eutrophication detection",
    "compute_event_frequency_map": "Event frequency map",
    "compute_event_summary_map": "Event summary map",
    "compute_event_timeseries_count": "Event count timeseries",
    "compute_stratification_index": "Stratification index",
    "compute_brunt_vaisala_frequency": "Brunt-Vaisala frequency",
    "compute_density_gradient_profile": "Density-gradient profile",
    "compute_mld_thermocline_offset": "MLD-thermocline offset",
    "compute_vertical_stability_timeseries": "Vertical stability timeseries",
    "compute_tracer_horizontal_advection_timeseries": "Tracer advection timeseries",
    "compute_tracer_advection_map": "Tracer advection map",
    "compute_partial_tracer_budget": "Partial tracer budget",
    "compute_budget_residual": "Budget residual",
    "compare_budget_term_magnitudes": "Budget mechanism ranking",
    "compute_front_proximity_index": "Front proximity index",
    "compute_eddy_influence_mask": "Eddy influence mask",
    "compute_tracer_gradient_alignment": "Tracer-gradient alignment",
    "compute_mesoscale_background_separation": "Mesoscale separation",
    "compute_flow_structure_context": "Flow structure context",
    "compute_event_precursor_composite": "Event precursor composite",
    "compute_event_lead_lag_regression": "Event lead-lag regression",
    "compute_oxygen_chla_coupling_metrics": "Oxygen-chlorophyll coupling",
    "compute_stratification_response_index": "Stratification response index",
    "compute_event_condition_contrast": "Event condition contrast",
    "replace_field_with_climatology": "Replace with climatology",
    "remove_field_anomaly_component": "Remove anomaly component",
    "filter_mesoscale_component": "Filter mesoscale component",
    "run_proxy_counterfactual_experiment": "Proxy counterfactual",
    "compare_counterfactual_outcome": "Counterfactual evidence",
    "rank_mechanism_support": "Mechanism support ranking",
    "grade_evidence_strength": "Evidence grading",
    "assemble_mechanism_evidence_report": "Mechanism evidence report",
    "assemble_environment_health_report": "Environment health assessment",
    "assemble_policy_recommendation_report": "Policy recommendation report",
    "check_claim_support_level": "Claim support check",
}

TOOL_TITLES_ZH: Dict[str, str] = {
    "get_dataset_info": "数据集信息",
    "load_dataset": "读取数据",
    "compute_spatial_field": "空间场",
    "compute_spatial_vorticity_map": "涡度图",
    "extract_regional_mean": "区域平均",
    "extract_point_timeseries": "点位时间序列",
    "extract_timeseries": "时间序列",
    "compute_area_weighted_mean": "面积加权平均",
    "compute_trend": "趋势分析",
    "compute_climatology": "气候态",
    "compute_anomaly": "异常分析",
    "compute_histogram": "分布统计",
    "extract_vertical_profile": "垂向剖面",
    "identify_mixed_layer_depth": "混合层深度",
    "identify_thermocline_depth": "温跃层深度",
    "identify_pycnocline_depth": "密跃层深度",
    "compute_layer_mean": "层平均",
    "compute_hovmoller": "霍夫默勒图",
    "compute_ts_diagram": "T-S 图",
    "compute_watermass_event_association": "水团分类",
    "build_watermass_tile_map": "水团格网图",
    "build_watermass_ts_diagram": "水团 T-S 图",
    "perform_eof_analysis": "EOF 分析",
    "compute_density": "密度场",
    "compute_derived_field": "派生场",
    "compute_kinetic_energy": "动能",
    "detect_heatwaves": "热浪检测",
    "detect_hypoxia": "低氧检测",
    "detect_algal_blooms": "藻华检测",
    "detect_upwelling": "上升流检测",
    "detect_eutrophication": "富营养化检测",
    "compute_event_frequency_map": "事件频率图",
    "compute_event_summary_map": "事件汇总图",
    "compute_event_timeseries_count": "事件计数时间序列",
    "compute_stratification_index": "层化指标",
    "compute_brunt_vaisala_frequency": "布伦特-魏萨拉频率",
    "compute_density_gradient_profile": "密度梯度剖面",
    "compute_mld_thermocline_offset": "混合层-温跃层偏移",
    "compute_vertical_stability_timeseries": "垂向稳定性时间序列",
    "compute_tracer_horizontal_advection_timeseries": "示踪物平流时间序列",
    "compute_tracer_advection_map": "示踪物平流图",
    "compute_partial_tracer_budget": "部分示踪物收支",
    "compute_budget_residual": "收支残差",
    "compare_budget_term_magnitudes": "收支机制排序",
    "compute_front_proximity_index": "锋面邻近指标",
    "compute_eddy_influence_mask": "涡旋影响掩膜",
    "compute_tracer_gradient_alignment": "示踪物梯度-流向对齐",
    "compute_mesoscale_background_separation": "中尺度分离",
    "compute_flow_structure_context": "流场结构背景",
    "compute_event_precursor_composite": "事件前兆合成",
    "compute_event_lead_lag_regression": "事件领先-滞后回归",
    "compute_oxygen_chla_coupling_metrics": "氧气-叶绿素耦合",
    "compute_stratification_response_index": "层化响应指标",
    "compute_event_condition_contrast": "事件条件对比",
    "replace_field_with_climatology": "替换为气候态",
    "remove_field_anomaly_component": "移除异常分量",
    "filter_mesoscale_component": "过滤中尺度分量",
    "run_proxy_counterfactual_experiment": "代理反事实实验",
    "compare_counterfactual_outcome": "反事实证据",
    "rank_mechanism_support": "机制支持排序",
    "grade_evidence_strength": "证据分级",
    "assemble_mechanism_evidence_report": "机制证据报告",
    "assemble_environment_health_report": "环境健康评估",
    "assemble_policy_recommendation_report": "政策建议报告",
    "check_claim_support_level": "结论支持度检查",
}


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    continue_pending: bool = False
    extracted_params: Dict[str, Any] = Field(default_factory=dict)
    additional_context: Dict[str, Any] = Field(default_factory=dict)
    synthesize: bool = True
    trust_env: bool = False
    model: str = Field(default_factory=lambda: load_model_name("PLANNER_MODEL", default=PlannerAgent.DEFAULT_MODEL))


class QueryResponse(BaseModel):
    status: Literal["completed", "clarification_needed", "failed"]
    query: str
    language: Literal["en"] = "en"
    conversation_id: Optional[str] = None
    routing_mode: Optional[
        Literal["dataset_analysis", "general_answer"]
    ] = None
    router_confidence: Optional[float] = None
    router_reason: Optional[str] = None
    skill_id: Optional[str] = None
    skills_used: List[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None
    missing_fields: List[str] = Field(default_factory=list)
    analysis_proposal: Optional[Dict[str, Any]] = None
    dataset_info: Dict[str, Any] = Field(default_factory=dict)
    plan_summary: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    plan_steps: List[Dict[str, Any]] = Field(default_factory=list)
    step_cards: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    result_cards: List[Dict[str, Any]] = Field(default_factory=list)
    result_summaries: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    synthesis: Optional[Dict[str, Any]] = None
    summary_status: Literal["pending", "completed", "failed"] = "completed"
    source_cards: List[Dict[str, Any]] = Field(default_factory=list)
    active_result_id: Optional[str] = None
    active_map_step_id: Optional[str] = None
    workspace_data: Dict[str, Any] = Field(default_factory=dict)
    workspace_data_by_result: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    error: Optional[str] = None
    failure_kind: Optional[FailureKind] = None
    recoverable: Optional[bool] = None
    timings: Dict[str, float] = Field(default_factory=dict)


def _record_timing(timings: Dict[str, float], name: str, started_at: float) -> float:
    elapsed_s = round(time.perf_counter() - started_at, 3)
    timings[name] = elapsed_s
    return elapsed_s


def _timing_event(name: str, elapsed_s: float, **metadata: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": "timing",
        "name": name,
        "elapsed_s": round(float(elapsed_s), 3),
    }
    payload.update({key: value for key, value in metadata.items() if value is not None})
    return payload


def _planning_phase_event(name: str, status: str = "started", **metadata: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": "planning_phase",
        "name": name,
        "status": status,
    }
    payload.update({key: value for key, value in metadata.items() if value is not None})
    return payload


def _final_response_payload(response: QueryResponse, timings: Dict[str, float]) -> Dict[str, Any]:
    payload = response.model_dump(mode="json")
    payload["timings"] = {key: round(float(value), 3) for key, value in timings.items()}
    return payload


class VisualizationRegion(BaseModel):
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


class VisualizationPoint(BaseModel):
    lat: float
    lon: float


class VisualizationRequest(BaseModel):
    dataset: str = Field(..., min_length=1)
    variable: str = Field(..., min_length=1)
    time_range: tuple[str, str]
    region: VisualizationRegion
    depth_mode: Literal["fixed", "feature", "layer_mean"]
    depth_range: tuple[float, float] = (0.0, -100.0)
    feature: Literal["mixed_layer", "thermocline", "pycnocline"] = "thermocline"
    layer_mean_label: str = "surface -> thermocline"
    selected_point: Optional[VisualizationPoint] = None
    search_depth_range: tuple[float, float] = (0.0, -300.0)


class VisualizationResponse(BaseModel):
    status: Literal["completed", "failed"]
    result_cards: List[Dict[str, Any]] = Field(default_factory=list)
    active_result_id: Optional[str] = None
    workspace_data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


app = FastAPI(
    title="OceanMind API",
    version="0.1.0",
    description="Minimal API for frontend integration and LLM-backed ocean analysis."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/dataset")
def get_active_dataset() -> Dict[str, Any]:
    return get_active_dataset_public_config()


@app.post("/report/export")
def export_report(request: ConversationReportRequest):
    return export_report_response(request)


@app.get("/demo/report")
def get_demo_report() -> Dict[str, Any]:
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Demo report not found.")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@app.get("/demo/cases")
def get_demo_cases() -> Dict[str, Any]:
    report = _load_demo_report()
    cases = []
    for item in report.get("skills", []):
        cases.append(
            {
                "case_id": item.get("case_id") or item.get("skill_id"),
                "skill_id": item.get("skill_id"),
                "query": item.get("query"),
                "status": item.get("status"),
                "result_type": item.get("result", {}).get("output_type"),
            }
        )
    return {"cases": cases}


@app.get("/demo/cases/{case_id}")
def get_demo_case(case_id: str) -> Dict[str, Any]:
    report = _load_demo_report()
    for item in report.get("skills", []):
        item_case_id = item.get("case_id") or item.get("skill_id")
        if item_case_id == case_id:
            return item
    raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")


@app.post("/visualize", response_model=VisualizationResponse)
async def run_visualization(request: VisualizationRequest) -> VisualizationResponse:
    try:
        workspace_data, result_card = _run_manual_visualization(request)
        return VisualizationResponse(
            status="completed",
            result_cards=[_json_safe(result_card)],
            active_result_id=result_card["id"],
            workspace_data=_json_safe(workspace_data),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest) -> QueryResponse:
    try:
        final_payload: Optional[QueryResponse] = None
        async for item in _iterate_query_execution(request):
            if item["event"] == "final":
                final_payload = QueryResponse.model_validate(item["payload"])
        if final_payload is None:
            raise RuntimeError("Query execution finished without a final payload.")
        return final_payload
    except Exception as exc:
        _logger.exception("Query execution failed before a final payload could be returned.")
        failure = _public_failure(exc, query=request.query, default_kind="transport")
        raise HTTPException(status_code=500, detail=failure["message"]) from exc


@app.post("/query/stream")
async def run_query_stream(request: QueryRequest):
    async def generate():
        pending_next: Optional[asyncio.Task] = None
        try:
            iterator = _iterate_query_execution(request).__aiter__()
            heartbeat_interval_sec = 15.0
            pending_next = asyncio.create_task(iterator.__anext__())
            while True:
                done, _ = await asyncio.wait({pending_next}, timeout=heartbeat_interval_sec)
                if not done:
                    yield json.dumps(
                        {
                            "event": "execution_event",
                            "payload": {
                                "type": "heartbeat",
                                "detail": "Stream is still active while the backend is waiting on a long-running step.",
                            },
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    ) + "\n"
                    continue

                try:
                    item = pending_next.result()
                except StopAsyncIteration:
                    break
                finally:
                    pending_next = None

                yield json.dumps(_json_safe(item), ensure_ascii=False, allow_nan=False) + "\n"
                pending_next = asyncio.create_task(iterator.__anext__())
        except Exception as exc:  # pragma: no cover - stream failure path
            _logger.exception("Streaming query failed before a final payload could be returned.")
            failure = _public_failure(exc, query=request.query, default_kind="transport")
            yield json.dumps(
                {
                    "event": "error",
                    "payload": {
                        "detail": failure["message"],
                        "failure_kind": failure["failure_kind"],
                        "recoverable": failure["recoverable"],
                    },
                },
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n"
        finally:
            if pending_next is not None and not pending_next.done():
                pending_next.cancel()

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def _load_demo_report() -> Dict[str, Any]:
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Demo report not found.")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


async def _iterate_query_execution(request: QueryRequest):
    request_started_at = time.perf_counter()
    timings: Dict[str, float] = {}
    conversation_id = request.conversation_id or str(uuid4())
    dataset_context = get_active_dataset_context()
    dataset_info = dataset_context["dataset"]

    llm_api_key = load_llm_api_key()
    llm_base_url = load_llm_base_url()
    planner_model = request.model or load_model_name("PLANNER_MODEL", default=PlannerAgent.DEFAULT_MODEL)
    router_model = load_model_name("QUERY_ROUTER_MODEL", default=UnifiedQueryProcessor.DEFAULT_MODEL)
    web_answer_model = load_model_name("WEB_ANSWER_MODEL", default=planner_model)
    result_synth_model = load_model_name("RESULT_SYNTHESIZER_MODEL", default=planner_model)

    planner = SkillPlanner(
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=planner_model,
        trust_env=request.trust_env,
    )
    harness_planner_agent = PlannerAgent(
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=planner_model,
        trust_env=request.trust_env,
    )
    processor = UnifiedQueryProcessor(
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=router_model,
        trust_env=request.trust_env,
    )
    web_answerer = WebAnswerSynthesizer(planner=planner, model=web_answer_model)
    proposal_synthesizer = AnalysisProposalSynthesizer(
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=result_synth_model,
        planner=planner,
        trust_env=request.trust_env,
    )
    synthesizer = (
        ResultSynthesizer(planner=planner, model=result_synth_model)
        if request.synthesize
        else None
    )
    search_service = WebSearchService()

    effective_query, effective_context, effective_extracted = _resolve_query_turn(
        conversation_id=conversation_id,
        request=request,
        planner=planner,
    )
    effective_context = _deep_merge_dicts(dataset_context, effective_context)
    effective_extracted = _hydrate_workspace_geometry_extracted_params(effective_extracted, effective_context)
    effective_extracted = _merge_named_region_resolved_entities(
        effective_extracted,
        effective_query,
        planner,
    )
    synthesis_profile_id = _infer_synthesis_profile_id(
        query=effective_query,
        additional_context=effective_context,
    )
    planner_safe_context = _strip_synthesis_profile_context(effective_context)

    events: List[Dict[str, Any]] = []
    yield {"event": "execution_event", "payload": {"type": "planning_started"}}
    await asyncio.sleep(0)

    loop = asyncio.get_event_loop()
    conv_state = CONVERSATION_STATE.setdefault(conversation_id, {})
    chinese_mode = False

    phase_event = _planning_phase_event("router")
    events.append(phase_event)
    yield {"event": "execution_event", "payload": phase_event}
    await asyncio.sleep(0)
    router_started_at = time.perf_counter()
    routing = _json_safe(
        await loop.run_in_executor(
            _POST_EXEC_POOL,
            lambda: processor.process(
                query=effective_query,
                dataset_context=dataset_context,
                conversation_context={},
                extracted_params=effective_extracted,
                additional_context=planner_safe_context,
            )
        )
    )
    router_elapsed_s = _record_timing(timings, "router", router_started_at)
    timing_event = _timing_event(
        "router",
        router_elapsed_s,
        model=router_model,
        routing_mode=routing.get("routing_mode"),
    )
    events.append(timing_event)
    yield {"event": "execution_event", "payload": timing_event}
    await asyncio.sleep(0)
    routing_event = {
        "type": "routing_decision",
        "action": routing.get("action"),
        "routing_mode": routing.get("routing_mode"),
        "needs_dataset": routing.get("needs_dataset"),
        "confidence": routing.get("confidence"),
        "reason": routing.get("reason"),
    }
    events.append(routing_event)
    yield {"event": "execution_event", "payload": routing_event}
    await asyncio.sleep(0)

    routing_mode = routing.get("routing_mode")
    if _should_offer_analysis_proposal(
        effective_query,
        routing=routing,
        additional_context=effective_context,
        dataset_context=dataset_context,
    ):
        phase_event = _planning_phase_event("analysis_proposal")
        events.append(phase_event)
        yield {"event": "execution_event", "payload": phase_event}
        await asyncio.sleep(0)
        proposal_started_at = time.perf_counter()
        try:
            analysis_proposal = _json_safe(
                await loop.run_in_executor(
                    _POST_EXEC_POOL,
                    lambda: proposal_synthesizer.propose(
                        user_request=effective_query,
                        dataset_context=dataset_context,
                        extracted_params=effective_extracted,
                        additional_context=effective_context,
                    ),
                )
            )
        except Exception as exc:
            analysis_proposal = _fallback_analysis_proposal(
                query=effective_query,
                dataset_context=dataset_context,
                reason=str(exc),
            )
        proposal_elapsed_s = _record_timing(timings, "analysis_proposal", proposal_started_at)
        timing_event = _timing_event("analysis_proposal", proposal_elapsed_s)
        events.append(timing_event)
        yield {"event": "execution_event", "payload": timing_event}
        await asyncio.sleep(0)
        proposal_event = {
            "type": "analysis_proposal_ready",
            "analysis_proposal": analysis_proposal,
        }
        events.append(proposal_event)
        yield {"event": "execution_event", "payload": proposal_event}
        await asyncio.sleep(0)

        final_response = QueryResponse(
            status="clarification_needed",
            query=request.query,
            language="en",
            conversation_id=conversation_id,
            routing_mode="dataset_analysis",
            router_confidence=routing.get("confidence"),
            router_reason=routing.get("reason"),
            clarification_question=analysis_proposal.get("approval_prompt")
            or "Review the suggested analysis plan before I run it.",
            missing_fields=["analysis_proposal_approval"],
            analysis_proposal=analysis_proposal,
            dataset_info=dataset_info,
            events=events,
            workspace_data=_empty_workspace_data(),
            workspace_data_by_result={},
        )
        state = CONVERSATION_STATE.setdefault(conversation_id, {})
        state["pending_analysis_proposal"] = {
            "original_query": _original_query_for_turn(request),
            "analysis_proposal": analysis_proposal,
            "additional_context": effective_context,
            "extracted_params": effective_extracted,
        }
        state.pop("pending_clarification", None)
        record_memory_started_at = time.perf_counter()
        await _record_turn_memory_async(
            loop=loop,
            conversation_id=conversation_id,
            planner=planner,
            turn_packet=_build_memory_turn_packet(
                user_query=request.query,
                effective_query=effective_query,
                status=final_response.status,
                routing_mode=final_response.routing_mode,
                response=final_response,
            ),
        )
        _record_timing(timings, "memory.record", record_memory_started_at)
        _record_timing(timings, "total", request_started_at)
        yield {"event": "final", "payload": _final_response_payload(final_response, timings)}
        await asyncio.sleep(0)
        return

    if routing_mode == "general_answer":
        memory_agent = MemoryAgent()
        general_memory_started_at = time.perf_counter()
        general_answer_memory_packet = await loop.run_in_executor(
            _POST_EXEC_POOL,
            lambda: memory_agent.recall(
                conversation_state=conv_state,
                project_memory_state=PROJECT_MEMORY_STATE,
                reference_memory_state=REFERENCE_MEMORY_STATE,
                query=effective_query,
                planner=planner,
                role="general_answer",
            ),
        )
        general_memory_elapsed_s = _record_timing(timings, "memory.general_answer", general_memory_started_at)
        timing_event = _timing_event("memory.general_answer", general_memory_elapsed_s)
        events.append(timing_event)
        yield {"event": "execution_event", "payload": timing_event}
        await asyncio.sleep(0)
        general_answer_context = {
            "conversation_memory": general_answer_memory_packet,
        }
        final_response, emitted_events = await _handle_direct_query(
            request=request,
            conversation_id=conversation_id,
            effective_query=effective_query,
            effective_context=general_answer_context,
            dataset_info=dataset_info,
            routing=routing,
            search_service=search_service,
            web_answerer=web_answerer,
            events=events,
            trust_env=request.trust_env,
        )
        for event in emitted_events:
            payload = _json_safe(event)
            events.append(payload)
            yield {"event": "execution_event", "payload": payload}
            await asyncio.sleep(0)
        state = CONVERSATION_STATE.setdefault(conversation_id, {})
        state.pop("pending_clarification", None)
        state.pop("pending_analysis_proposal", None)
        record_memory_started_at = time.perf_counter()
        await _record_turn_memory_async(
            loop=loop,
            conversation_id=conversation_id,
            planner=planner,
            turn_packet=_build_memory_turn_packet(
                user_query=request.query,
                effective_query=effective_query,
                status=final_response.status,
                routing_mode=final_response.routing_mode,
                response=final_response,
            ),
        )
        _record_timing(timings, "memory.record", record_memory_started_at)
        _record_timing(timings, "total", request_started_at)
        yield {"event": "final", "payload": _final_response_payload(final_response, timings)}
        await asyncio.sleep(0)
        return

    if routing_mode != "dataset_analysis":
        raise RuntimeError(f"Unsupported routing mode from dataset router: {routing_mode}")

    memory_agent = MemoryAgent()
    merged_extracted = dict(effective_extracted)
    phase_event = _planning_phase_event("memory.planner")
    events.append(phase_event)
    yield {"event": "execution_event", "payload": phase_event}
    await asyncio.sleep(0)
    planner_memory_started_at = time.perf_counter()
    planner_memory_packet = await loop.run_in_executor(
        _POST_EXEC_POOL,
        lambda: memory_agent.recall(
            conversation_state=conv_state,
            project_memory_state=PROJECT_MEMORY_STATE,
            reference_memory_state=REFERENCE_MEMORY_STATE,
            query=effective_query,
            planner=planner,
            role="planner",
        ),
    )
    planner_memory_elapsed_s = _record_timing(timings, "memory.planner", planner_memory_started_at)
    timing_event = _timing_event("memory.planner", planner_memory_elapsed_s)
    events.append(timing_event)
    yield {"event": "execution_event", "payload": timing_event}
    await asyncio.sleep(0)
    planner_memory_entities = (
        planner_memory_packet.get("resolved_entities")
        if isinstance(planner_memory_packet, dict)
        else None
    )
    merged_extracted = _merge_memory_resolved_entities(merged_extracted, planner_memory_entities)
    planner_context = _context_with_role_text(
        planner_safe_context,
        conversation_memory=planner_memory_packet,
    )
    phase_event = _planning_phase_event("memory.synthesizer")
    events.append(phase_event)
    yield {"event": "execution_event", "payload": phase_event}
    await asyncio.sleep(0)
    synthesizer_memory_started_at = time.perf_counter()
    synthesizer_memory_packet = await loop.run_in_executor(
        _POST_EXEC_POOL,
        lambda: memory_agent.recall(
            conversation_state=conv_state,
            project_memory_state=PROJECT_MEMORY_STATE,
            reference_memory_state=REFERENCE_MEMORY_STATE,
            query=effective_query,
            planner=planner,
            role="synthesizer",
        ),
    )
    synthesizer_memory_elapsed_s = _record_timing(timings, "memory.synthesizer", synthesizer_memory_started_at)
    timing_event = _timing_event("memory.synthesizer", synthesizer_memory_elapsed_s)
    events.append(timing_event)
    yield {"event": "execution_event", "payload": timing_event}
    await asyncio.sleep(0)
    synthesizer_context = _context_with_role_text(
        effective_context,
        conversation_memory=synthesizer_memory_packet,
    )

    generated_plan: Optional[Dict[str, Any]] = None
    skills_used: List[str] = []
    clarification_question: Optional[str] = None
    missing_fields: List[str] = []
    failed_error: Optional[str] = None
    failed_kind: Optional[FailureKind] = None
    failed_recoverable: Optional[bool] = None
    plan_summary: Optional[str] = None
    step_cards_by_id: Dict[str, Dict[str, Any]] = {}
    step_order: List[str] = []
    visible_result_cards: List[Dict[str, Any]] = []

    executor = OceanHarnessExecutor()
    phase_event = _planning_phase_event("planning", timeout_s=HARNESS_PLANNING_TIMEOUT_S or None)
    events.append(phase_event)
    yield {"event": "execution_event", "payload": phase_event}
    await asyncio.sleep(0)
    runtime_events = executor.execute_query(
        user_request=effective_query,
        extracted_params=merged_extracted,
        additional_context=planner_context,
        planner=harness_planner_agent,
        planner_kwargs={"planning_timeout_s": HARNESS_PLANNING_TIMEOUT_S},
        supervised=True,
    )

    async for raw_event in runtime_events:
        translated_events = _translate_runtime_event(
            raw_event=raw_event,
            generated_plan=generated_plan,
            step_cards_by_id=step_cards_by_id,
            step_order=step_order,
            executor=executor,
            visible_result_cards=visible_result_cards,
            request=request,
            chinese=chinese_mode,
        )
        for translated in translated_events:
            event = _json_safe(translated)
            events.append(event)
            yield {"event": "execution_event", "payload": event}
            await asyncio.sleep(0)

            event_type = event.get("type")
            if event_type == "plan_ready":
                generated_plan = raw_event.get("plan")
                skills_used = list(raw_event.get("skills_used", []) or [])
                plan_summary = _build_plan_summary(effective_query, generated_plan, chinese=chinese_mode)
                planning_elapsed_s = raw_event.get("planning_elapsed_s")
                if isinstance(planning_elapsed_s, (int, float)):
                    timings["planning"] = round(float(planning_elapsed_s), 3)
                    timing_event = _timing_event(
                        "planning",
                        timings["planning"],
                        skill_id=raw_event.get("skill_id"),
                        skills_used=skills_used,
                    )
                    events.append(timing_event)
                    yield {"event": "execution_event", "payload": timing_event}
                    await asyncio.sleep(0)
                planner_agent_timings = raw_event.get("planner_agent_timings")
                planner_prompt_sizes = raw_event.get("planner_agent_prompt_sizes")
                if isinstance(planner_agent_timings, dict):
                    for timing_name, timing_value in planner_agent_timings.items():
                        if not isinstance(timing_value, (int, float)):
                            continue
                        normalized_name = str(timing_name)
                        timings[normalized_name] = round(float(timing_value), 3)
                        metadata = (
                            {"prompt_sizes": planner_prompt_sizes}
                            if normalized_name == "planner.total" and isinstance(planner_prompt_sizes, dict)
                            else {}
                        )
                        timing_event = _timing_event(normalized_name, timings[normalized_name], **metadata)
                        events.append(timing_event)
                        yield {"event": "execution_event", "payload": timing_event}
                        await asyncio.sleep(0)
            elif event_type == "plan_replanned":
                generated_plan = raw_event.get("plan")
                skills_used = list(raw_event.get("skills_used", []) or [])
                plan_summary = _build_plan_summary(effective_query, generated_plan, chinese=chinese_mode)
                failed_error = None
            elif event_type == "clarification_needed":
                clarification_question = event.get("question") or ""
                missing_fields = list(event.get("missing_fields", []))
                planning_elapsed_s = raw_event.get("planning_elapsed_s")
                if isinstance(planning_elapsed_s, (int, float)):
                    timings["planning"] = round(float(planning_elapsed_s), 3)
                    timing_event = _timing_event("planning", timings["planning"], status="clarification_needed")
                    events.append(timing_event)
                    yield {"event": "execution_event", "payload": timing_event}
                    await asyncio.sleep(0)
            elif event_type == "step_failed":
                failure = _public_failure(
                    raw_event.get("error") or event.get("error") or "Step execution failed.",
                    query=effective_query,
                    default_kind="execution",
                )
                failed_error = _friendly_step_error(raw_event, events, request)
                failed_kind = failure["failure_kind"]
                failed_recoverable = failure["recoverable"]
            elif event_type == "planning_failed":
                failure = _public_failure(
                    raw_event.get("error") or event.get("error") or "Planning failed.",
                    query=effective_query,
                    default_kind="planning",
                )
                failed_error = failure["message"]
                failed_kind = failure["failure_kind"]
                failed_recoverable = failure["recoverable"]
                planning_elapsed_s = raw_event.get("planning_elapsed_s")
                if isinstance(planning_elapsed_s, (int, float)):
                    timings["planning"] = round(float(planning_elapsed_s), 3)
                    timing_event = _timing_event("planning", timings["planning"], status="failed")
                    events.append(timing_event)
                    yield {"event": "execution_event", "payload": timing_event}
                    await asyncio.sleep(0)

        if raw_event.get("type") == "plan_aborted" and not failed_error:
            failure = _public_failure(
                raw_event.get("last_error") or raw_event.get("reason") or "Plan aborted.",
                query=effective_query,
                default_kind="planning",
            )
            failed_error = failure["message"]
            failed_kind = failure["failure_kind"]
            failed_recoverable = failure["recoverable"]

    if clarification_question:
        failure = _public_failure(clarification_question, query=effective_query, default_kind="planning")
        if failure["failure_kind"] == "capability_boundary":
            final_response = QueryResponse(
                status="completed",
                query=request.query,
                language="en",
                conversation_id=conversation_id,
                routing_mode="dataset_analysis",
                router_confidence=routing.get("confidence"),
                router_reason=routing.get("reason"),
                skill_id=generated_plan.get("skill_id") if generated_plan else None,
                skills_used=skills_used,
                dataset_info=dataset_info,
                plan_summary=plan_summary,
                plan=generated_plan,
                plan_steps=_compact_plan_steps(generated_plan, chinese=chinese_mode),
                step_cards=_ordered_step_cards(step_cards_by_id, step_order),
                events=events,
                synthesis={"summary": failure["message"]},
                summary_status="completed",
                workspace_data=_empty_workspace_data(),
                workspace_data_by_result={},
                failure_kind=failure["failure_kind"],
                recoverable=failure["recoverable"],
            )
            state = CONVERSATION_STATE.setdefault(conversation_id, {})
            state.pop("pending_clarification", None)
            state.pop("pending_analysis_proposal", None)
            await _record_turn_memory_async(
                loop=loop,
                conversation_id=conversation_id,
                planner=planner,
                turn_packet=_build_memory_turn_packet(
                    user_query=request.query,
                    effective_query=effective_query,
                    status=final_response.status,
                    routing_mode=final_response.routing_mode,
                    response=final_response,
                    generated_plan=generated_plan,
                ),
            )
            _record_timing(timings, "total", request_started_at)
            yield {"event": "final", "payload": _final_response_payload(final_response, timings)}
            await asyncio.sleep(0)
            return
        final_response = QueryResponse(
            status="clarification_needed",
            query=request.query,
            language="en",
            conversation_id=conversation_id,
            routing_mode="dataset_analysis",
            router_confidence=routing.get("confidence"),
            router_reason=routing.get("reason"),
            skill_id=generated_plan.get("skill_id") if generated_plan else None,
            skills_used=skills_used,
            clarification_question=clarification_question,
            missing_fields=missing_fields,
            dataset_info=dataset_info,
            plan_summary=plan_summary,
            plan=generated_plan,
            plan_steps=_compact_plan_steps(generated_plan, chinese=chinese_mode),
            step_cards=_ordered_step_cards(step_cards_by_id, step_order),
            events=events,
            workspace_data=_empty_workspace_data(),
            workspace_data_by_result={},
        )
        CONVERSATION_STATE.setdefault(conversation_id, {})["pending_clarification"] = {
            "original_query": _original_query_for_turn(request),
            "clarification_question": clarification_question,
            "missing_fields": missing_fields,
            "additional_context": effective_context,
            "extracted_params": effective_extracted,
        }
        await _record_turn_memory_async(
            loop=loop,
            conversation_id=conversation_id,
            planner=planner,
            turn_packet=_build_memory_turn_packet(
                user_query=request.query,
                effective_query=effective_query,
                status=final_response.status,
                routing_mode=final_response.routing_mode,
                response=final_response,
                generated_plan=generated_plan,
            ),
        )
        _record_timing(timings, "total", request_started_at)
        yield {"event": "final", "payload": _final_response_payload(final_response, timings)}
        await asyncio.sleep(0)
        return

    if failed_error or generated_plan is None:
        failure = _public_failure(
            failed_error or "No plan was generated.",
            query=effective_query,
            default_kind=failed_kind or "planning",
        )
        if failed_kind:
            failure["failure_kind"] = failed_kind
        if failed_recoverable is not None:
            failure["recoverable"] = failed_recoverable
        if failure["failure_kind"] == "capability_boundary":
            final_response = QueryResponse(
                status="completed",
                query=request.query,
                language="en",
                conversation_id=conversation_id,
                routing_mode="dataset_analysis",
                router_confidence=routing.get("confidence"),
                router_reason=routing.get("reason"),
                skill_id=generated_plan.get("skill_id") if generated_plan else None,
                skills_used=skills_used,
                dataset_info=dataset_info,
                plan_summary=plan_summary,
                plan=generated_plan,
                plan_steps=_compact_plan_steps(generated_plan, chinese=chinese_mode),
                step_cards=_ordered_step_cards(step_cards_by_id, step_order),
                events=events,
                synthesis={"summary": failure["message"]},
                summary_status="completed",
                workspace_data=_empty_workspace_data(),
                workspace_data_by_result={},
                failure_kind=failure["failure_kind"],
                recoverable=failure["recoverable"],
            )
            state = CONVERSATION_STATE.setdefault(conversation_id, {})
            state.pop("pending_clarification", None)
            state.pop("pending_analysis_proposal", None)
            await _record_turn_memory_async(
                loop=loop,
                conversation_id=conversation_id,
                planner=planner,
                turn_packet=_build_memory_turn_packet(
                    user_query=request.query,
                    effective_query=effective_query,
                    status=final_response.status,
                    routing_mode=final_response.routing_mode,
                    response=final_response,
                    generated_plan=generated_plan,
                ),
            )
            _record_timing(timings, "total", request_started_at)
            yield {"event": "final", "payload": _final_response_payload(final_response, timings)}
            await asyncio.sleep(0)
            return
        final_response = QueryResponse(
            status="failed",
            query=request.query,
            language="en",
            conversation_id=conversation_id,
            routing_mode="dataset_analysis",
            router_confidence=routing.get("confidence"),
            router_reason=routing.get("reason"),
            skill_id=generated_plan.get("skill_id") if generated_plan else None,
            skills_used=skills_used,
            dataset_info=dataset_info,
            plan_summary=plan_summary,
            plan=generated_plan,
            plan_steps=_compact_plan_steps(generated_plan, chinese=chinese_mode),
            step_cards=_ordered_step_cards(step_cards_by_id, step_order),
            events=events,
            workspace_data=_empty_workspace_data(),
            workspace_data_by_result={},
            error=failure["message"],
            failure_kind=failure["failure_kind"],
            recoverable=failure["recoverable"],
        )
        state = CONVERSATION_STATE.setdefault(conversation_id, {})
        state.pop("pending_clarification", None)
        state.pop("pending_analysis_proposal", None)
        await _record_turn_memory_async(
            loop=loop,
            conversation_id=conversation_id,
            planner=planner,
            turn_packet=_build_memory_turn_packet(
                user_query=request.query,
                effective_query=effective_query,
                status=final_response.status,
                routing_mode=final_response.routing_mode,
                response=final_response,
                generated_plan=generated_plan,
            ),
        )
        _record_timing(timings, "total", request_started_at)
        yield {"event": "final", "payload": _final_response_payload(final_response, timings)}
        await asyncio.sleep(0)
        return

    result_summaries = executor.get_result_summaries()
    if chinese_mode:
        result_summaries = {
            result_id: _localize_environment_assessment_summary(summary, chinese=True)
            if isinstance(summary, dict)
            else summary
            for result_id, summary in result_summaries.items()
        }
    result_cards = list(visible_result_cards)
    if _should_suppress_default_policy_report(effective_query, generated_plan):
        suppressed_policy_result_ids = {
            str(card.get("id"))
            for card in result_cards
            if card.get("type") == "policy_recommendation_result"
        }
        if suppressed_policy_result_ids:
            result_cards = [
                card
                for card in result_cards
                if str(card.get("id")) not in suppressed_policy_result_ids
            ]
            result_summaries = {
                result_id: summary
                for result_id, summary in result_summaries.items()
                if str(result_id) not in suppressed_policy_result_ids
            }
            for step_card in step_cards_by_id.values():
                results = step_card.get("results")
                if isinstance(results, list):
                    step_card["results"] = [
                        result
                        for result in results
                        if not (
                            isinstance(result, dict)
                            and str(result.get("id")) in suppressed_policy_result_ids
                        )
                    ]
    active_result_id = _pick_active_result_id(generated_plan, result_cards)

    # ── Build workspace data immediately ──
    _t0 = time.monotonic()
    workspace_data_by_result = await loop.run_in_executor(
        _POST_EXEC_POOL,
        lambda: _build_workspace_data_by_result(
            executor=executor,
            result_cards=result_cards,
            chinese=chinese_mode,
        ),
    )
    workspace_data = workspace_data_by_result.get(active_result_id or "", _empty_workspace_data())
    step_cards = _ordered_step_cards(step_cards_by_id, step_order)
    _t1 = time.monotonic()
    _logger.info("workspace build %.2fs", _t1 - _t0)

    # ── Send results_ready as execution_event so frontend can show results immediately ──
    results_ready_event = {
        "type": "results_ready",
        "result_cards": _json_safe(result_cards),
        "workspace_data": workspace_data,
        "workspace_data_by_result": _json_safe(workspace_data_by_result),
        "active_result_id": active_result_id,
    }
    events.append(results_ready_event)
    yield {"event": "execution_event", "payload": results_ready_event}
    await asyncio.sleep(0)

    # ── Run synthesis (no longer blocks results display) ──
    synthesis_payload: Optional[Dict[str, Any]] = None
    summary_status: Literal["pending", "completed", "failed"] = "pending"
    synthesis_error: Optional[str] = None
    synthesis_is_fallback = False
    source_cards: List[Dict[str, Any]] = []
    synthesis_required = _requires_llm_result_synthesis(
        active_plan=generated_plan,
        synthesis_profile_id=synthesis_profile_id,
        user_request=effective_query,
    )
    if synthesizer is not None:
        completed_steps = [
            {
                "step_id": event.get("step_id"),
                "tool": event.get("tool"),
                "result_id": event.get("result_id"),
                "output_type": event.get("output_type"),
                "result_summary": event.get("result_summary"),
            }
            for event in events
            if event.get("type") == "step_completed"
        ]
        synthesis_context = dict(synthesizer_context or {})
        if synthesis_profile_id:
            synthesis_context["synthesis_profile_id"] = synthesis_profile_id
        planner_policy_making_intent = _planner_policy_making_intent(generated_plan)
        if planner_policy_making_intent is not None:
            synthesis_context["policy_making_intent"] = planner_policy_making_intent
        evidence_packets = _build_synthesis_evidence_packets(
            executor=executor,
            completed_steps=completed_steps,
            result_summaries=result_summaries,
            active_result_id=active_result_id,
        )
        synthesis_context["evidence_packets"] = evidence_packets
        assessment_context_packet = _build_assessment_context_packet(
            user_request=effective_query,
            evidence_packets=evidence_packets,
            result_summaries=result_summaries,
            synthesis_profile_id=synthesis_profile_id,
        )
        if assessment_context_packet:
            synthesis_context["assessment_context_packet"] = assessment_context_packet
        policy_context_packet = _build_policy_context_packet(
            user_request=effective_query,
            evidence_packets=evidence_packets,
            result_summaries=result_summaries,
            synthesis_profile_id=synthesis_profile_id,
            policy_making_intent=planner_policy_making_intent,
        )
        if policy_context_packet:
            synthesis_context["policy_context_packet"] = policy_context_packet
        _t2 = time.monotonic()
        for attempt in range(1, LLM_SYNTHESIS_MAX_ATTEMPTS + 1):
            attempt_context = dict(synthesis_context)
            if synthesis_error:
                attempt_context["previous_synthesis_error"] = synthesis_error
                attempt_context["retry_instruction"] = (
                    "Regenerate the final result synthesis. Do not return a placeholder or fallback; "
                    "produce a schema-valid synthesis grounded only in completed tool results and evidence packets."
                )
            synthesis_started_event = {
                "type": "synthesis_started" if attempt == 1 else "synthesis_retry_started",
                "attempt": attempt,
                "max_attempts": LLM_SYNTHESIS_MAX_ATTEMPTS,
            }
            events.append(synthesis_started_event)
            yield {"event": "execution_event", "payload": synthesis_started_event}
            await asyncio.sleep(0)
            try:
                synthesis_payload = await loop.run_in_executor(
                    _POST_EXEC_POOL,
                    lambda: _json_safe(
                        synthesizer.synthesize(
                            user_request=effective_query,
                            active_plan=generated_plan,
                            completed_steps=completed_steps,
                            result_summaries=result_summaries,
                            additional_context=attempt_context,
                        )
                    ),
                )
                if not synthesis_payload:
                    raise ValueError("LLM synthesis returned an empty payload.")
                synthesis_payload = _augment_environment_assessment_synthesis(
                    synthesis_payload=synthesis_payload,
                    result_summaries=result_summaries,
                    active_result_id=active_result_id,
                    user_request=effective_query,
                )
                synthesis_payload = _filter_synthesis_findings_by_language(
                    synthesis_payload,
                    chinese=chinese_mode,
                )
                synthesis_payload.pop("recommended_followups", None)
                result_summaries = _apply_lag_selection_rewrite(
                    result_summaries,
                    result_cards,
                    step_cards_by_id,
                    synthesis_payload,
                    chinese=chinese_mode,
                )
                summary_status = "completed"
                synthesis_error = None
                break
            except Exception as exc:
                synthesis_payload = None
                synthesis_error = str(exc)
                summary_status = "failed"
                failure = _public_failure(synthesis_error, query=effective_query, default_kind="synthesis")
                display_error = failure["message"]
                synthesis_failed_event = {
                    "type": "synthesis_attempt_failed"
                    if attempt < LLM_SYNTHESIS_MAX_ATTEMPTS
                    else "synthesis_failed",
                    "attempt": attempt,
                    "max_attempts": LLM_SYNTHESIS_MAX_ATTEMPTS,
                    "fatal": synthesis_required,
                    "error": display_error,
                    "failure_kind": failure["failure_kind"],
                    "recoverable": failure["recoverable"],
                }
                events.append(synthesis_failed_event)
                yield {"event": "execution_event", "payload": synthesis_failed_event}
                await asyncio.sleep(0)
        _t3 = time.monotonic()
        _logger.info("synthesis %.2fs", _t3 - _t2)

        if not synthesis_payload and synthesis_error:
            synthesis_payload = _build_result_synthesis_fallback(
                user_request=effective_query,
                active_plan=generated_plan,
                completed_steps=completed_steps,
                result_summaries=result_summaries,
                synthesis_error=synthesis_error,
            )
            if synthesis_payload:
                synthesis_is_fallback = True
                summary_status = "completed"
                synthesis_error = None

        if synthesis_payload:
            _attach_result_interpretations(list(step_cards_by_id.values()), synthesis_payload, chinese=chinese_mode)
            step_cards = _ordered_step_cards(step_cards_by_id, step_order)

            if not synthesis_is_fallback:
                post_analysis_query = _build_post_analysis_search_query(
                    effective_query,
                    synthesis_payload,
                    dataset_info,
                )
                post_search_started = {
                    "type": "post_analysis_search_started",
                    "query": post_analysis_query,
                }
                events.append(post_search_started)
                yield {"event": "execution_event", "payload": post_search_started}
                await asyncio.sleep(0)

                try:
                    search_results = await loop.run_in_executor(
                        _POST_EXEC_POOL,
                        lambda: search_service.search(
                            post_analysis_query,
                            max_results=5,
                            trust_env=request.trust_env,
                        ),
                    )
                    source_cards = _build_source_cards(
                        search_results,
                        default_reason="Relevant external background for the generated scientific synthesis.",
                        chinese=chinese_mode,
                    )
                    try:
                        source_cards = await loop.run_in_executor(
                            _POST_EXEC_POOL,
                            lambda: _json_safe(web_answerer.translate_source_cards_to_english(source_cards)),
                        )
                    except Exception as exc:
                        failure = _public_failure(exc, query=effective_query, default_kind="transport")
                        source_cards = _build_source_fallback_card(
                            title="External source translation unavailable",
                            snippet=failure["message"],
                            reason="The analysis completed, but translating external source cards failed.",
                        )
                    if not source_cards:
                        source_cards = _build_source_fallback_card(
                            title="No usable external sources found",
                            snippet=f"Search query: {post_analysis_query}",
                            reason="The post-analysis search completed, but no parseable references were returned.",
                        )
                    post_search_completed = {
                        "type": "post_analysis_search_completed",
                        "result_count": len(search_results),
                        "provider": _search_provider_from_results(search_results, search_service),
                    }
                    events.append(post_search_completed)
                    yield {"event": "execution_event", "payload": post_search_completed}
                    await asyncio.sleep(0)
                except Exception as exc:
                    failure = _public_failure(exc, query=effective_query, default_kind="transport")
                    source_cards = _build_source_fallback_card(
                        title="External source search unavailable",
                        snippet=failure["message"],
                        reason="The analysis completed, but the optional post-analysis web search failed.",
                    )
                    post_search_failed = {
                        "type": "post_analysis_search_failed",
                        "error": failure["message"],
                        "failure_kind": failure["failure_kind"],
                        "recoverable": failure["recoverable"],
                    }
                    events.append(post_search_failed)
                    yield {"event": "execution_event", "payload": post_search_failed}
                    await asyncio.sleep(0)

            synthesis_ready_event = {
                "type": "synthesis_ready",
                "synthesis": synthesis_payload,
                "step_cards": _json_safe(step_cards),
                "result_cards": _json_safe(result_cards),
                "result_summaries": _json_safe(result_summaries),
                "summary_status": summary_status,
                "source_cards": source_cards,
            }
            events.append(synthesis_ready_event)
            yield {"event": "execution_event", "payload": synthesis_ready_event}
            await asyncio.sleep(0)
    else:
        if synthesis_required:
            summary_status = "failed"
            synthesis_error = "LLM synthesis is required for this assessment, but synthesis was disabled."
            failure = _public_failure(synthesis_error, query=effective_query, default_kind="synthesis")
            synthesis_failed_event = {
                "type": "synthesis_failed",
                "attempt": 0,
                "max_attempts": 0,
                "fatal": True,
                "error": failure["message"],
                "failure_kind": failure["failure_kind"],
                "recoverable": failure["recoverable"],
            }
            events.append(synthesis_failed_event)
            yield {"event": "execution_event", "payload": synthesis_failed_event}
            await asyncio.sleep(0)
        else:
            summary_status = "completed"

    final_status: Literal["completed", "failed"] = (
        "failed" if synthesis_required and summary_status == "failed" else "completed"
    )
    final_failure = (
        _public_failure(synthesis_error, query=effective_query, default_kind="synthesis")
        if final_status == "failed"
        else None
    )
    final_response = QueryResponse(
        status=final_status,
        query=request.query,
        language="en",
        conversation_id=conversation_id,
        routing_mode="dataset_analysis",
        router_confidence=routing.get("confidence"),
        router_reason=routing.get("reason"),
        skill_id=generated_plan.get("skill_id"),
        skills_used=skills_used or list(generated_plan.get("skills_used", []) or []),
        dataset_info=dataset_info,
        plan_summary=plan_summary,
        plan=generated_plan,
        plan_steps=_compact_plan_steps(generated_plan, chinese=chinese_mode),
        step_cards=_json_safe(step_cards),
        events=events,
        result_cards=_json_safe(result_cards),
        result_summaries=_json_safe(result_summaries),
        synthesis=synthesis_payload,
        summary_status=summary_status,
        source_cards=source_cards,
        active_result_id=active_result_id,
        active_map_step_id=None,
        workspace_data=workspace_data,
        workspace_data_by_result=_json_safe(workspace_data_by_result),
        error=final_failure["message"] if final_failure else None,
        failure_kind=final_failure["failure_kind"] if final_failure else None,
        recoverable=final_failure["recoverable"] if final_failure else None,
    )
    state = CONVERSATION_STATE.setdefault(conversation_id, {})
    state.pop("pending_clarification", None)
    state.pop("pending_analysis_proposal", None)
    _persist_turn_results(
        conversation_id=conversation_id,
        query=_original_query_for_turn(request),
        skill_id=generated_plan.get("skill_id") if generated_plan else None,
        result_summaries=result_summaries,
        synthesis=synthesis_payload,
        active_result_id=active_result_id,
    )
    await _record_turn_memory_async(
        loop=loop,
        conversation_id=conversation_id,
        planner=planner,
        turn_packet=_build_memory_turn_packet(
            user_query=request.query,
            effective_query=effective_query,
            status=final_response.status,
            routing_mode=final_response.routing_mode,
            response=final_response,
            generated_plan=generated_plan,
            result_summaries=result_summaries,
        ),
    )
    _record_timing(timings, "total", request_started_at)
    yield {"event": "final", "payload": _final_response_payload(final_response, timings)}
    await asyncio.sleep(0)


async def _handle_direct_query(
    request: QueryRequest,
    conversation_id: str,
    effective_query: str,
    effective_context: Dict[str, Any],
    dataset_info: Dict[str, Any],
    routing: Dict[str, Any],
    search_service: WebSearchService,
    web_answerer: WebAnswerSynthesizer,
    events: List[Dict[str, Any]],
    trust_env: bool,
) -> tuple[QueryResponse, List[Dict[str, Any]]]:
    chinese_mode = False
    loop = asyncio.get_event_loop()

    emitted_events: List[Dict[str, Any]] = [
        {"type": "general_answer_started", "phase": "first_pass"}
    ]
    first_pass_text = await loop.run_in_executor(
        _POST_EXEC_POOL,
        lambda: web_answerer.answer_or_request_search(
            user_request=effective_query,
            additional_context=effective_context,
        ),
    )
    search_match = _GENERAL_SEARCH_TAG_RE.fullmatch(first_pass_text)
    if search_match is None:
        emitted_events.append({"type": "general_answer_completed", "used_web_search": False})
        return QueryResponse(
            status="completed",
            query=request.query,
            language="en",
            conversation_id=conversation_id,
            routing_mode="general_answer",
            router_confidence=routing.get("confidence"),
            router_reason=routing.get("reason"),
            dataset_info=dataset_info,
            synthesis={"summary": first_pass_text},
            events=events + emitted_events,
            workspace_data=_empty_workspace_data(),
            workspace_data_by_result={},
        ), emitted_events

    search_query = (search_match.group(1) or "").strip() or effective_query
    emitted_events.append({"type": "search_requested", "query": search_query})
    emitted_events.append({"type": "search_started", "query": search_query})
    search_error: Optional[str] = None
    search_results: List[Dict[str, Any]] = []
    try:
        search_results = await loop.run_in_executor(
            _POST_EXEC_POOL,
            lambda: search_service.search(
                search_query,
                max_results=5,
                trust_env=trust_env,
            ),
        )
        emitted_events.append(
            {
                "type": "search_completed",
                "result_count": len(search_results),
                "provider": _search_provider_from_results(search_results, search_service),
            }
        )
    except Exception as exc:
        search_error = str(exc)
        failure = _public_failure(exc, query=effective_query, default_kind="transport")
        emitted_events.append(
            {
                "type": "search_failed",
                "error": failure["message"],
                "failure_kind": failure["failure_kind"],
                "recoverable": failure["recoverable"],
            }
        )

    synthesized = await loop.run_in_executor(
        _POST_EXEC_POOL,
        lambda: _json_safe(
            web_answerer.synthesize_answer(
                user_request=effective_query,
                search_query=search_query,
                search_results=search_results,
                search_error=search_error,
                additional_context=effective_context,
            )
        ),
    )

    source_cards: List[Dict[str, Any]] = []
    if search_results:
        source_cards = _build_source_cards(search_results, chinese=chinese_mode)
        if not source_cards:
            source_cards = _build_source_fallback_card(
                title="No usable external sources found",
                snippet=f"Search query: {search_query}",
                reason="The web search finished, but no parseable external references were returned.",
            )
    elif search_error:
        failure = _public_failure(search_error, query=effective_query, default_kind="transport")
        source_cards = _build_source_fallback_card(
            title="External source search unavailable",
            snippet=failure["message"],
            reason="The answer was generated without usable web search results.",
        )
    emitted_events.append({"type": "general_answer_completed", "used_web_search": True})
    return QueryResponse(
        status="completed",
        query=request.query,
        language="en",
        conversation_id=conversation_id,
        routing_mode="general_answer",
        router_confidence=routing.get("confidence"),
        router_reason=routing.get("reason"),
        dataset_info=dataset_info,
        synthesis={"summary": synthesized["summary"]},
        source_cards=source_cards,
        events=events + emitted_events,
        workspace_data=_empty_workspace_data(),
        workspace_data_by_result={},
    ), emitted_events


def _normalize_step_progress(progress: Any) -> Dict[str, Any]:
    if not isinstance(progress, dict):
        return {}

    raw_percent = progress.get("percent")
    percent: Optional[float]
    try:
        percent = float(raw_percent)
    except (TypeError, ValueError):
        percent = None
    if percent is not None:
        if percent > 1.0:
            percent = percent / 100.0
        percent = max(0.0, min(1.0, percent))

    payload: Dict[str, Any] = {
        "phase": str(progress.get("phase") or "running"),
        "message": str(progress.get("message") or ""),
    }
    if percent is not None and math.isfinite(percent):
        payload["percent"] = percent

    for key in (
        "completed_units",
        "total_units",
        "unit_label",
        "current_unit",
        "storage_backend",
        "compute_backend",
        "chunks",
    ):
        if key in progress:
            payload[key] = progress[key]

    if "completed_units" not in payload and "completed_files" in progress:
        payload["completed_units"] = progress["completed_files"]
    if "total_units" not in payload and "total_files" in progress:
        payload["total_units"] = progress["total_files"]
    if "current_unit" not in payload and "current_file" in progress:
        payload["current_unit"] = progress["current_file"]
    if "unit_label" not in payload and (
        "completed_files" in progress or "total_files" in progress or "current_file" in progress
    ):
        payload["unit_label"] = "data source"

    for key in ("completed_files", "total_files", "current_file"):
        if key in progress:
            payload[key] = progress[key]

    current_unit = _sanitize_progress_current_unit(payload.get("current_unit"), payload.get("unit_label"))
    if current_unit:
        payload["current_unit"] = current_unit
    else:
        payload.pop("current_unit", None)
    return payload


def _sanitize_progress_current_unit(value: Any, unit_label: Any = None) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    label = str(unit_label or "").strip().lower()
    if label == "task" and _looks_like_internal_task_key(text):
        return None
    if len(text) > 96:
        return f"{text[:93]}..."
    return text


def _looks_like_internal_task_key(text: str) -> bool:
    if text.startswith("(") and ("getitem-" in text or "finalize-" in text or "concatenate-" in text):
        return True
    if re.search(r"\b(getitem|finalize|concatenate|transpose|rechunk|mean|sum)-[0-9a-f]{8,}", text):
        return True
    return False


def _merge_monotonic_step_progress(previous: Any, incoming: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(previous, dict) or "percent" not in incoming:
        return incoming
    previous_percent = previous.get("percent")
    incoming_percent = incoming.get("percent")
    if not isinstance(previous_percent, (int, float)) or not isinstance(incoming_percent, (int, float)):
        return incoming
    if not math.isfinite(float(previous_percent)) or not math.isfinite(float(incoming_percent)):
        return incoming
    if float(incoming_percent) >= float(previous_percent):
        return incoming
    return {
        **incoming,
        "percent": float(previous_percent),
    }


def _translate_runtime_event(
    raw_event: Dict[str, Any],
    generated_plan: Optional[Dict[str, Any]],
    step_cards_by_id: Dict[str, Dict[str, Any]],
    step_order: List[str],
    executor: SkillExecutor,
    visible_result_cards: List[Dict[str, Any]],
    request: QueryRequest,
    *,
    chinese: bool = False,
) -> List[Dict[str, Any]]:
    event_type = raw_event.get("type")
    translated: List[Dict[str, Any]] = []

    if event_type == "planning_start":
        translated.append({"type": "planning_started"})
        return translated

    if event_type == "planning_failed":
        raw_error = raw_event.get("error") or "Planning failed."
        failure = _public_failure(raw_error, query=request.query, default_kind="planning")
        friendly_error = failure["message"]
        translated.append(
            {
                "type": "planning_failed",
                "error": friendly_error,
                "error_category": raw_event.get("error_category") or "planning_failed",
                "failure_kind": failure["failure_kind"],
                "recoverable": failure["recoverable"],
                "planning_elapsed_s": raw_event.get("planning_elapsed_s"),
            }
        )
        return translated

    if event_type == "plan_generated":
        plan = raw_event.get("plan") or {}
        initial_step_cards = _reseed_step_cards_from_plan(
            plan,
            step_cards_by_id,
            step_order,
            chinese=chinese,
            preserve_completed=False,
        )
        translated.append(
            {
                "type": "plan_ready",
                "skill_id": raw_event.get("skill_id"),
                "skills_used": raw_event.get("skills_used", [raw_event.get("skill_id")]),
                "plan": plan,
                "plan_summary": _build_plan_summary("", plan, chinese=chinese),
                "plan_steps": _compact_plan_steps(plan, chinese=chinese),
                "initial_step_cards": initial_step_cards,
                "planning_elapsed_s": raw_event.get("planning_elapsed_s"),
            }
        )
        return translated

    if event_type == "plan_replanned":
        plan = raw_event.get("plan") or {}
        initial_step_cards = _reseed_step_cards_from_plan(
            plan,
            step_cards_by_id,
            step_order,
            chinese=chinese,
            preserve_completed=True,
        )
        translated.append(
            {
                "type": "plan_replanned",
                "skill_id": raw_event.get("skill_id"),
                "skills_used": raw_event.get("skills_used", [raw_event.get("skill_id")]),
                "reason": raw_event.get("reason"),
                "plan": plan,
                "plan_summary": _build_plan_summary("", plan, chinese=chinese),
                "plan_steps": _compact_plan_steps(plan, chinese=chinese),
                "initial_step_cards": initial_step_cards,
            }
        )
        return translated

    if event_type == "clarification_needed":
        translated.append(
            {
                "type": "clarification_needed",
                "question": raw_event.get("question", ""),
                "missing_fields": raw_event.get("missing_fields", []),
                "skills_used": raw_event.get("skills_used", [raw_event.get("skill_id")]),
                "planning_elapsed_s": raw_event.get("planning_elapsed_s"),
            }
        )
        return translated

    if event_type == "step_start":
        step_id = str(raw_event.get("step_id", ""))
        step_meta = _find_plan_step(generated_plan, step_id)
        tool_name = raw_event.get("tool")
        # Build descriptive label from plan params when available
        plan_params = step_meta.get("params", {}) if step_meta and isinstance(step_meta, dict) else {}
        param_summary = _build_descriptive_label_from_params(tool_name, plan_params, chinese=chinese)
        card = {
            "step_id": step_id,
            "human_label": _humanize_tool_name(tool_name, chinese=chinese),
            "technical_label": param_summary,
            "status": "running",
            "results_hidden_by_default": True,
            "results": [],
            "interpretation": "",
            "actions": [],
            "is_map_bound": False,
            "is_expanded": False,
        }
        if step_meta:
            card["human_label"] = str(step_meta.get("human_label") or card["human_label"])
            if step_meta.get("technical_label"):
                card["technical_label"] = str(step_meta["technical_label"])
        step_cards_by_id[step_id] = card
        if step_id not in step_order:
            step_order.append(step_id)
        translated.append(
            {
                "type": "step_started",
                "step_id": step_id,
                "tool": raw_event.get("tool"),
                "step_card": dict(card),
                "current_step_index": len(step_order),
                "total_steps": len((generated_plan or {}).get("steps", [])),
            }
        )
        return translated

    if event_type == "step_progress":
        step_id = str(raw_event.get("step_id", ""))
        tool_name = raw_event.get("tool")
        step_meta = _find_plan_step(generated_plan, step_id)
        progress = _normalize_step_progress(raw_event.get("progress"))
        step_card = step_cards_by_id.setdefault(
            step_id,
            {
                "step_id": step_id,
                "human_label": _humanize_tool_name(tool_name, chinese=chinese),
                "technical_label": str(tool_name or ""),
                "status": "running",
                "results_hidden_by_default": True,
                "results": [],
                "interpretation": "",
                "actions": [],
                "is_map_bound": False,
                "is_expanded": False,
            },
        )
        progress = _merge_monotonic_step_progress(step_card.get("progress"), progress)
        if step_meta:
            step_card["human_label"] = str(step_meta.get("human_label") or step_card["human_label"])
            if step_meta.get("technical_label"):
                step_card["technical_label"] = str(step_meta["technical_label"])
        step_card["status"] = "running"
        step_card["progress"] = progress
        if step_id not in step_order:
            step_order.append(step_id)
        translated.append(
            {
                "type": "step_progress",
                "step_id": step_id,
                "tool": tool_name,
                "progress": progress,
                "step_card": _json_safe(dict(step_card)),
            }
        )
        return translated

    if event_type == "step_complete":
        step_id = str(raw_event.get("step_id", ""))
        result_id = str(raw_event.get("result_id", ""))
        output_type = str(raw_event.get("output_type") or "generic_result")
        summary = raw_event.get("result_summary")
        if isinstance(summary, dict) and output_type == "environment_assessment_result":
            summary = _localize_environment_assessment_summary(summary, chinese=chinese)
        step_card = step_cards_by_id.setdefault(
            step_id,
            {
                "step_id": step_id,
                "human_label": _humanize_tool_name(raw_event.get("tool"), chinese=chinese),
                "technical_label": _build_descriptive_label(raw_event.get("tool"), summary, chinese=chinese),
                "status": "running",
                "results_hidden_by_default": True,
                "results": [],
                "interpretation": "",
                "actions": [],
                "is_map_bound": False,
                "is_expanded": False,
            },
        )
        # Update technical_label with result summary info
        step_card["technical_label"] = _build_descriptive_label(raw_event.get("tool"), summary, chinese=chinese)
        result_card = _build_result_card(result_id, summary or {}, step_tool=raw_event.get("tool"), chinese=chinese)
        result_card["owner_step_id"] = step_id
        result_card["surface"] = _result_type_to_surface(output_type)
        result_card["actions"] = _build_result_actions(result_card["surface"], chinese=chinese)
        visible_result_cards.append(result_card)
        step_card["status"] = "completed"
        if isinstance(step_card.get("progress"), dict):
            step_card["progress"] = {
                **step_card["progress"],
                "percent": 1.0,
                "message": "已完成" if chinese else "Completed",
            }
        step_card["results"].append(result_card)
        step_card["is_map_bound"] = step_card["is_map_bound"] or result_card["surface"] == "map"
        step_card["actions"] = result_card["actions"]
        step_card["is_expanded"] = False

        translated.append(
            {
                "type": "step_completed",
                "step_id": step_id,
                "tool": raw_event.get("tool"),
                "result_id": result_id,
                "output_type": output_type,
                "result_summary": summary,
                "step_card": _json_safe(dict(step_card)),
            }
        )
        translated.append(
            {
                "type": "step_result_attached",
                "step_id": step_id,
                "result_id": result_id,
                "step_card": _json_safe(dict(step_card)),
            }
        )
        return translated

    if event_type == "step_reflection_started":
        step_id = str(raw_event.get("step_id", ""))
        tool_name = raw_event.get("tool")
        step_meta = _find_plan_step(generated_plan, step_id)
        step_card = step_cards_by_id.setdefault(
            step_id,
            {
                "step_id": step_id,
                "human_label": _humanize_tool_name(tool_name, chinese=chinese),
                "technical_label": str(tool_name or ""),
                "status": "running",
                "results_hidden_by_default": True,
                "results": [],
                "interpretation": "",
                "actions": [],
                "is_map_bound": False,
                "is_expanded": False,
            },
        )
        if step_meta:
            step_card["human_label"] = str(step_meta.get("human_label") or step_card["human_label"])
            if step_meta.get("technical_label"):
                step_card["technical_label"] = str(step_meta["technical_label"])
        step_card["status"] = "running"
        step_card["is_expanded"] = False
        step_card.pop("error", None)
        step_card["progress"] = _normalize_step_progress(
            {
                "phase": "reflection",
                "message": (
                    "正在反思这一步并更新工作流"
                    if chinese
                    else "Reflecting on this step and updating the workflow"
                ),
            }
        )
        if step_id not in step_order:
            step_order.append(step_id)
        translated.append(
            {
                "type": "step_reflection_started",
                "step_id": step_id,
                "tool": tool_name,
                "reason": raw_event.get("reason"),
                "replans_used": raw_event.get("replans_used"),
                "max_replans": raw_event.get("max_replans"),
                "step_card": _json_safe(dict(step_card)),
            }
        )
        return translated

    if event_type == "step_error":
        step_id = str(raw_event.get("step_id", ""))
        step_card = step_cards_by_id.setdefault(
            step_id,
            {
                "step_id": step_id,
                "human_label": _humanize_tool_name(raw_event.get("tool"), chinese=chinese),
                "technical_label": str(raw_event.get("tool") or ""),
                "status": "running",
                "results_hidden_by_default": True,
                "results": [],
                "interpretation": "",
                "actions": [],
                "is_map_bound": False,
                "is_expanded": False,
            },
        )
        step_card["status"] = "failed"
        friendly_error = _friendly_step_error(raw_event, translated, request)
        failure = _public_failure(raw_event.get("error") or friendly_error, query=request.query, default_kind="execution")
        step_card["error"] = friendly_error
        translated.append(
            {
                "type": "step_failed",
                "step_id": step_id,
                "tool": raw_event.get("tool"),
                "error": friendly_error,
                "failure_kind": failure["failure_kind"],
                "recoverable": failure["recoverable"],
                "step_card": _json_safe(dict(step_card)),
            }
        )
        return translated

    return translated


def _compact_plan_steps(plan: Optional[Dict[str, Any]], *, chinese: bool = False) -> List[Dict[str, Any]]:
    if not isinstance(plan, dict):
        return []
    steps = []
    for step in plan.get("steps", []):
        if not isinstance(step, dict):
            continue
        steps.append(
            {
                "step_id": step.get("step_id"),
                "tool": step.get("tool"),
                "save_as": step.get("save_as"),
                "human_label": step.get("human_label") or _humanize_tool_name(step.get("tool"), chinese=chinese),
                "technical_label": step.get("technical_label") or step.get("tool"),
            }
        )
    return steps


def _resolve_query_turn(
    conversation_id: str,
    request: QueryRequest,
    planner: Optional[SkillPlanner] = None,
) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    state = CONVERSATION_STATE.get(conversation_id, {})
    pending_proposal = state.get("pending_analysis_proposal")
    pending = state.get("pending_clarification")
    additional_context = dict(request.additional_context)
    extracted_params = dict(request.extracted_params)

    if request.continue_pending and isinstance(pending_proposal, dict):
        previous_context = pending_proposal.get("additional_context")
        previous_extracted = pending_proposal.get("extracted_params")
        if isinstance(previous_context, dict):
            additional_context = _deep_merge_dicts(previous_context, additional_context)
        if isinstance(previous_extracted, dict):
            extracted_params = {**previous_extracted, **extracted_params}

        original_query = str(pending_proposal.get("original_query") or request.query)
        proposal = pending_proposal.get("analysis_proposal") if isinstance(pending_proposal.get("analysis_proposal"), dict) else {}
        proposed_query = str(proposal.get("proposed_query") or "").strip()

        if _is_general_question_clarification(request.query):
            state.pop("pending_analysis_proposal", None)
            general_query = (
                f"Original question: {original_query}\n"
                f"User clarification: {request.query}\n\n"
                "Answer this as a general conceptual ocean-science question. "
                "Do not run the pending analysis proposal unless the user explicitly asks for it."
            )
            return general_query, additional_context, extracted_params

        if proposed_query and _is_analysis_proposal_approval(request.query):
            state.pop("pending_analysis_proposal", None)
            state.pop("pending_clarification", None)
            approved_plan = proposal.get("plan")
            if not (
                isinstance(approved_plan, dict)
                and isinstance(approved_plan.get("steps"), list)
                and approved_plan.get("steps")
            ):
                additional_context["analysis_proposal_failure_context"] = {
                    "original_query": original_query,
                    "failed_proposal": proposal,
                    "user_approval": request.query,
                    "instruction": (
                        "The user approved a proposal that did not contain a validated executable tool plan. "
                        "Do not run tools. Ask the user to provide a more specific analysis query or revise the scope."
                    ),
                }
                clarification_query = (
                    "The previous suggested analysis is not executable because it does not contain a validated tool plan. "
                    "Ask the user to revise the request with a specific region, time window, and priority evidence; "
                    "do not run dataset tools."
                )
                return clarification_query, additional_context, extracted_params
            additional_context["analysis_proposal_context"] = {
                "original_query": original_query,
                "approved_query": proposed_query,
                "approved_proposal": proposal,
                "user_approval": request.query,
                "instruction": (
                    "The user approved this suggested analysis plan. Execute the approved tool plan as the current task."
                ),
            }
            if isinstance(approved_plan, dict):
                additional_context["analysis_proposal_context"]["approved_plan"] = approved_plan
            return proposed_query, additional_context, extracted_params

        if not _is_analysis_proposal_revision(request.query):
            state.pop("pending_analysis_proposal", None)
            state.pop("pending_clarification", None)
            return request.query, additional_context, extracted_params

        additional_context["force_analysis_proposal"] = True
        additional_context["analysis_proposal_revision_context"] = {
            "original_query": original_query,
            "previous_proposal": proposal,
            "user_revision": request.query,
            "instruction": "Revise the suggested analysis proposal. Do not execute tools until the user approves.",
        }
        revised_query = (
            f"Original broad request: {original_query}\n"
            f"Previous proposed query: {proposed_query or '(missing)'}\n"
            f"User requested revision: {request.query}"
        )
        return revised_query, additional_context, extracted_params

    context_proposal = additional_context.get("approved_analysis_proposal")
    if _is_analysis_proposal_approval(request.query) and isinstance(context_proposal, dict):
        proposed_query = str(context_proposal.get("proposed_query") or "").strip()
        approved_plan = context_proposal.get("plan")
        if (
            proposed_query
            and isinstance(approved_plan, dict)
            and isinstance(approved_plan.get("steps"), list)
            and approved_plan.get("steps")
        ):
            state.pop("pending_analysis_proposal", None)
            state.pop("pending_clarification", None)
            original_query = str(context_proposal.get("public_question") or proposed_query)
            additional_context["analysis_proposal_context"] = {
                "original_query": original_query,
                "approved_query": proposed_query,
                "approved_proposal": context_proposal,
                "approved_plan": approved_plan,
                "user_approval": request.query,
                "instruction": (
                    "The user approved this suggested analysis plan. Execute the approved tool plan as the current task."
                ),
            }
            return proposed_query, additional_context, extracted_params

    if not request.continue_pending or not isinstance(pending, dict):
        return request.query, additional_context, extracted_params

    if _pending_message_starts_new_turn(
        planner=planner,
        latest_query=request.query,
        pending=pending,
    ):
        state.pop("pending_clarification", None)
        return request.query, additional_context, extracted_params

    previous_context = pending.get("additional_context")
    previous_extracted = pending.get("extracted_params")
    if isinstance(previous_context, dict):
        additional_context = _deep_merge_dicts(previous_context, additional_context)
    if isinstance(previous_extracted, dict):
        extracted_params = {**previous_extracted, **extracted_params}

    original_query = str(pending.get("original_query") or request.query)
    clarification_question = str(pending.get("clarification_question") or "")
    missing_fields = pending.get("missing_fields", [])

    if _is_general_question_clarification(request.query):
        CONVERSATION_STATE.get(conversation_id, {}).pop("pending_clarification", None)
        general_query = (
            f"Original question: {original_query}\n"
            f"User clarification: {request.query}\n\n"
            "Answer this as a general conceptual ocean-science question. "
            "Do not require a geographic region, time period, or dataset computation unless the user asks for one."
        )
        return general_query, additional_context, extracted_params

    additional_context["clarification_context"] = {
        "original_query": original_query,
        "clarification_question": clarification_question,
        "missing_fields": missing_fields,
        "user_answer": request.query,
        "instruction": (
            "Treat user_answer as the answer to the pending clarification question. "
            "Use it to fill the missing fields and continue the original task "
            "instead of interpreting it as a new standalone task."
        ),
    }
    resumed_query = (
        f"Original request: {original_query}\n"
        f"Pending clarification question: {clarification_question}\n"
        f"Missing fields: {missing_fields}\n"
        f"User clarification answer: {request.query}"
    )
    return resumed_query, additional_context, extracted_params


def _infer_synthesis_profile_id(
    *,
    query: str,
    additional_context: Dict[str, Any],
) -> Optional[str]:
    direct = additional_context.get("synthesis_profile_id") if isinstance(additional_context, dict) else None
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    proposal_context = (
        additional_context.get("analysis_proposal_context")
        if isinstance(additional_context, dict)
        else None
    )
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

    if isinstance(query, str) and _INTEGRATED_ASSESSMENT_PROFILE_RE.search(query):
        return OCEAN_INTEGRATED_ASSESSMENT_PROFILE_ID
    return None


def _requires_llm_result_synthesis(
    *,
    active_plan: Optional[Dict[str, Any]],
    synthesis_profile_id: Optional[str],
    user_request: str,
) -> bool:
    if synthesis_profile_id == OCEAN_INTEGRATED_ASSESSMENT_PROFILE_ID:
        return True
    if _EXPLICIT_POLICY_REPORT_RE.search(user_request or ""):
        return True
    planner_policy_making_intent = _planner_policy_making_intent(active_plan)
    if planner_policy_making_intent is True:
        return True
    if planner_policy_making_intent is None and _POLICY_GUIDANCE_INTENT_RE.search(user_request or ""):
        return True
    if _INTEGRATED_ASSESSMENT_PROFILE_RE.search(user_request or ""):
        return True
    if not isinstance(active_plan, dict):
        return False
    raw_skills_used = active_plan.get("skills_used") or []
    plan_skills_used = [raw_skills_used] if isinstance(raw_skills_used, str) else raw_skills_used
    skill_ids = {
        str(active_plan.get("skill_id") or "").strip(),
        *(
            str(skill).strip()
            for skill in plan_skills_used
            if str(skill).strip()
        ),
    }
    return "ocean_environment_health_assessment" in skill_ids


def _planner_policy_making_intent(active_plan: Optional[Dict[str, Any]]) -> Optional[bool]:
    if not isinstance(active_plan, dict):
        return None

    def read_from(payload: Any) -> Optional[bool]:
        if not isinstance(payload, dict):
            return None
        value = _coerce_optional_bool(payload.get("policy_making_intent"))
        if value is not None:
            return value
        intent_flags = payload.get("intent_flags")
        if isinstance(intent_flags, dict):
            value = _coerce_optional_bool(
                intent_flags.get("policy_making")
                if "policy_making" in intent_flags
                else intent_flags.get("policy_making_intent")
            )
            if value is not None:
                return value
        return None

    for payload in (
        active_plan.get("planner_llm_decision"),
        active_plan,
    ):
        value = read_from(payload)
        if value is not None:
            return value
        if isinstance(payload, dict):
            value = read_from(payload.get("planner_skill_selection"))
            if value is not None:
                return value
    return None


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


def _strip_synthesis_profile_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Keep summary-only profile hints out of planner/router prompts."""
    if not isinstance(context, dict):
        return {}
    try:
        stripped = json.loads(json.dumps(context, default=str))
    except Exception:
        stripped = dict(context)
    if not isinstance(stripped, dict):
        return {}
    stripped.pop("synthesis_profile_id", None)
    proposal_context = stripped.get("analysis_proposal_context")
    if isinstance(proposal_context, dict):
        proposal = proposal_context.get("approved_proposal")
        if isinstance(proposal, dict):
            proposal.pop("synthesis_profile_id", None)
            skill_plan = proposal.get("skill_plan")
            if isinstance(skill_plan, dict):
                skill_plan.pop("synthesis_profile_id", None)
    return stripped


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _original_query_for_turn(request: QueryRequest) -> str:
    if request.continue_pending and request.conversation_id:
        pending_proposal = CONVERSATION_STATE.get(request.conversation_id, {}).get("pending_analysis_proposal")
        if isinstance(pending_proposal, dict):
            original_query = pending_proposal.get("original_query")
            if isinstance(original_query, str) and original_query.strip():
                return original_query
        pending = CONVERSATION_STATE.get(request.conversation_id, {}).get("pending_clarification")
        if isinstance(pending, dict):
            original_query = pending.get("original_query")
            if isinstance(original_query, str) and original_query.strip():
                return original_query
    return request.query


_ANALYSIS_PROPOSAL_APPROVAL_RE = re.compile(
    r"^\s*(ok|okay|yes|y|approve|approved|run|run it|run proposed analysis|execute|proceed|go ahead|"
    r"use this|use this plan|looks good|continue|start|start it|do it)\s*[.!。！]*\s*$|"
    r"^\s*(可以|好|好的|同意|确认|执行|开始|继续|就这个|按这个|没问题|运行|跑吧|可以执行)\s*[。！!]*\s*$",
    re.IGNORECASE,
)
_ANALYSIS_PROPOSAL_NEGATIVE_RE = re.compile(
    r"\b(no|not|don't|do not|revise|change|edit|modify|instead|wait|stop)\b|"
    r"不要|不行|不是|修改|改成|换成|先不|等等|停止"
)
_ANALYSIS_PROPOSAL_REVISION_RE = re.compile(
    r"\b(revise|revision|change|edit|modify|instead|rather than|only|limit to|focus on|narrow to|"
    r"switch to|replace|exclude|include)\b|"
    r"修改|改成|改为|换成|只看|只要|限定|聚焦|不要|排除|加入|增加|减少|换一个",
    re.IGNORECASE,
)
_BROAD_ANALYSIS_MARKER_RE = re.compile(
    r"\b(what can (i|we) (analy[sz]e|do)|what analysis|can (i|we)|could (i|we)|is it suitable|"
    r"suitable for|risk assessment|environmental risk|environmental health|marine health|aquaculture|"
    r"fish farming|marine ranching|fisher(y|ies)|policy|management|economic benefit|environmental benefit|"
    r"demo|demonstration|public question|broad question)\b|"
    r"能做什么|做什么分析|可以做什么|适合.*分析|能不能|还能不能|可不可以|适不适合|养鱼|海洋牧场|"
    r"水产|渔业|生态风险|环境风险|环境健康|环保效益|经济效益|大众|演示|管理建议|政策",
    re.IGNORECASE,
)
_DIRECT_EXECUTION_MARKER_RE = re.compile(
    r"^\s*(plot|map|show|compute|calculate|extract|detect|make|build|draw|compare|run)\b|"
    r"画|绘制|计算|提取|检测|生成|比较"
)
_CONCRETE_ANALYSIS_MARKER_RE = re.compile(
    r"\b(trend|timeseries|time series|hovmoller|time-depth|map|section|transect|eof|hypoxia|oxygen|"
    r"temperature|sst|salinity|chlorophyll|vorticity|transport|stratification|201\d|202\d|\d+\s*[°]?[EWNS])\b|"
    r"趋势|时间序列|剖面|断面|缺氧|氧气|温度|盐度|叶绿素|涡度|输运|层化|南海|中国海|东海|黄海|渤海",
    re.IGNORECASE,
)
_COORDINATE_OR_RANGE_RE = re.compile(
    r"\d{4}\s*[-–]\s*\d{4}|\d+\s*[°]?\s*[EWNS]\s*[-–]\s*\d+\s*[°]?\s*[EWNS]|\d+\s*[°]?[EWNS]"
)


def _is_analysis_proposal_approval(query: str) -> bool:
    cleaned = " ".join(str(query or "").strip().split())
    if not cleaned or _ANALYSIS_PROPOSAL_NEGATIVE_RE.search(cleaned):
        return False
    return bool(_ANALYSIS_PROPOSAL_APPROVAL_RE.search(cleaned))


def _is_analysis_proposal_revision(query: str) -> bool:
    cleaned = " ".join(str(query or "").strip().split())
    if not cleaned or _is_analysis_proposal_approval(cleaned):
        return False
    if _is_specific_execution_query(cleaned):
        return False
    if _ANALYSIS_PROPOSAL_REVISION_RE.search(cleaned):
        return True
    return len(cleaned) < 80


def _should_offer_analysis_proposal(
    query: str,
    *,
    routing: Dict[str, Any],
    additional_context: Dict[str, Any],
    dataset_context: Dict[str, Any],
) -> bool:
    if additional_context.get("analysis_proposal_context"):
        return False
    if additional_context.get("force_analysis_proposal"):
        return True

    dataset = dataset_context.get("dataset") if isinstance(dataset_context.get("dataset"), dict) else {}
    variables = dataset.get("variables") if isinstance(dataset, dict) else []
    if not isinstance(variables, list) or not variables:
        return False

    query_text = str(query or "").strip()
    if not query_text:
        return False

    routing_mode = routing.get("routing_mode")

    if _is_specific_execution_query(query_text):
        return False

    if not _BROAD_ANALYSIS_MARKER_RE.search(query_text):
        return False

    return routing_mode == "dataset_analysis"


def _is_specific_execution_query(query: str) -> bool:
    lower = query.lower()
    if not _DIRECT_EXECUTION_MARKER_RE.search(query):
        return False
    concrete_score = 0
    if _CONCRETE_ANALYSIS_MARKER_RE.search(query):
        concrete_score += 1
    if _COORDINATE_OR_RANGE_RE.search(query):
        concrete_score += 1
    if "drawn" in lower or "polygon" in lower or "transect" in lower or "区域" in query or "多边形" in query:
        concrete_score += 1
    return concrete_score >= 2


def _fallback_analysis_proposal(
    *,
    query: str,
    dataset_context: Dict[str, Any],
    reason: str,
) -> Dict[str, Any]:
    query_text = str(query or "")
    dataset = dataset_context.get("dataset") if isinstance(dataset_context.get("dataset"), dict) else {}
    variables = {
        str(item).strip().lower()
        for item in (dataset.get("variables") if isinstance(dataset.get("variables"), list) else [])
        if str(item).strip()
    }
    depth_levels = dataset.get("depth_levels") if isinstance(dataset.get("depth_levels"), list) else []
    has_full_health_stack = {"temp", "salt", "oxygen"}.issubset(variables) and len(depth_levels) > 1
    if has_full_health_stack:
        proposed_query = (
            "The current request needs a validated executable tool plan before it can be run. Please revise the "
            "question with the region, time window, and environmental evidence you want prioritized."
        )
    elif "temp" in variables:
        proposed_query = (
            "Analyze surface temperature variability and trend over the active dataset domain and time range, "
            "including a spatial trend map and an EOF-style variability summary if supported."
        )
    else:
        proposed_query = (
            "Summarize the strongest executable diagnostics supported by the active dataset variables and propose "
            "a focused follow-up analysis."
        )
    limitations = [f"LLM proposal generation failed, so a conservative fallback proposal was used: {reason}"]
    if "oxygen" not in variables:
        limitations.append("No oxygen variable is available, so hypoxia analysis should not be proposed.")
    if len(depth_levels) <= 1:
        limitations.append("The dataset has one or no depth levels, so bottom-layer and stratification diagnostics are limited.")
    return {
        "title": "Suggested Analysis Plan",
        "public_question": query,
        "proposed_query": proposed_query,
        "analysis_steps": [
            "Translate the broad question into a concrete dataset-supported diagnostic.",
            "Run only analyses supported by the active variables and coverage.",
            "Summarize results with explicit evidence limits.",
        ],
        "expected_outputs": [
            "A concise scientific answer.",
            "Maps, time series, or variability diagnostics when supported.",
            "Evidence-bounded interpretation for public or management use.",
        ],
        "limitations": limitations,
        "approval_prompt": "If this plan looks right, reply OK or click Run proposed analysis.",
        "executable": not (has_full_health_stack and _INTEGRATED_ASSESSMENT_PROFILE_RE.search(query_text)),
        "requires_revision": bool(has_full_health_stack and _INTEGRATED_ASSESSMENT_PROFILE_RE.search(query_text)),
        **(
            {
                "approval_prompt": (
                    "I could not build a validated executable tool plan yet. "
                    "Please revise the query with more specific analysis details."
                ),
                "selected_skills": ["ocean_environment_health_assessment"],
                "skill_plan": {
                    "primary_skill": "ocean_environment_health_assessment",
                    "skills_used": ["ocean_environment_health_assessment"],
                    "planned_tools": [],
                    "planned_steps": [],
                },
            }
            if has_full_health_stack and _INTEGRATED_ASSESSMENT_PROFILE_RE.search(query_text)
            else {}
        ),
    }


def _friendly_step_error(event: Dict[str, Any], events: List[Dict[str, Any]], request: QueryRequest) -> str:
    raw_error = event.get("error") or "Step execution failed."
    tool_name = event.get("tool")

    if tool_name in {"extract_regional_mean", "extract_timeseries", "compute_spatial_field"}:
        empty_reason = _detect_empty_data_reason(events, request)
        if empty_reason:
            return empty_reason

    if (
        raw_error == "zero-size array to reduction operation minimum which has no identity"
        or "All-NaN slice encountered" in str(raw_error)
    ):
        empty_reason = _detect_empty_data_reason(events, request)
        if empty_reason:
            return empty_reason
        return (
            "The selected depth, time, or region produced no finite values. "
            "Try a broader selection or check that the requested layer exists in this dataset."
        )

    return _public_failure(raw_error, query=request.query, default_kind="execution")["message"]


def _build_result_synthesis_fallback(
    *,
    user_request: str,
    active_plan: Dict[str, Any],
    completed_steps: List[Dict[str, Any]],
    result_summaries: Dict[str, Dict[str, Any]],
    synthesis_error: str,
) -> Dict[str, Any]:
    result_count = len(result_summaries)
    step_count = len(completed_steps)
    skill_id = str(active_plan.get("skill_id") or "ocean analysis").replace("_", " ")
    failure = _public_failure(synthesis_error, query=user_request, default_kind="synthesis")
    findings: List[Dict[str, Any]] = []

    for result_id, summary in list(result_summaries.items())[:5]:
        if not isinstance(summary, dict):
            continue
        title = str(
            summary.get("title")
            or summary.get("name")
            or summary.get("analysis_type")
            or str(result_id).replace("_", " ")
        ).strip()
        output_type = str(summary.get("type") or summary.get("output_type") or "result").strip()
        evidence = _fallback_evidence_lines(result_id=result_id, summary=summary, output_type=output_type)
        findings.append(
            {
                "finding": f"Computed result available: {title}",
                "evidence": evidence or [f"Result id {result_id}; output type {output_type}."],
                "result_id": str(result_id),
            }
        )

    summary = (
        f"The {skill_id} workflow completed {step_count} step(s) and produced {result_count} result set(s), "
        "but the final LLM-written interpretation did not pass the required format checks. "
        "The result cards and maps contain the computed evidence; this fallback summary does not add new scientific "
        "or management conclusions beyond the tool outputs."
    )
    return {
        "summary": summary,
        "scientific_findings": findings,
        "notable_patterns": [],
        "anomalies": [],
        "significance_assessment": [],
        "uncertainties": [
            "This is a deterministic fallback summary because natural-language synthesis failed validation.",
            "Evidence strength and policy interpretation should be read from a successful synthesis or inspected directly in the result cards.",
        ],
        "synthesis_warnings": [
            failure["message"],
            "Fallback summary generated from completed step metadata only.",
        ],
    }


def _fallback_evidence_lines(*, result_id: Any, summary: Dict[str, Any], output_type: str) -> List[str]:
    evidence = [f"Result id: {result_id}; output type: {output_type}."]
    preferred_keys = [
        "variable",
        "analysis_type",
        "event_type",
        "top_associated_watermass_name",
        "top_associated_watermass",
        "association_score",
        "hotspot_tile_count",
        "valid_tile_count",
        "dominant_watermass_name",
        "mean",
        "max",
        "min",
        "slope",
        "p_value",
        "units",
    ]
    for key in preferred_keys:
        if key not in summary:
            continue
        value = _compact_fallback_value(summary.get(key))
        if value is not None:
            evidence.append(f"{key}: {value}.")
        if len(evidence) >= 4:
            break
    if len(evidence) < 4:
        statistics = summary.get("statistics")
        if isinstance(statistics, dict):
            for key, value in statistics.items():
                compact_value = _compact_fallback_value(value)
                if compact_value is None:
                    continue
                evidence.append(f"statistics.{key}: {compact_value}.")
                if len(evidence) >= 4:
                    break
    return evidence


def _compact_fallback_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return f"{value:.4g}" if isinstance(value, float) else str(value)
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned[:120] if cleaned else None
    if isinstance(value, (list, tuple)):
        items = [_compact_fallback_value(item) for item in list(value)[:4]]
        compact_items = [item for item in items if item]
        return ", ".join(compact_items) if compact_items else None
    if isinstance(value, dict):
        parts = []
        for key, item in list(value.items())[:4]:
            compact_item = _compact_fallback_value(item)
            if compact_item:
                parts.append(f"{key}={compact_item}")
        return ", ".join(parts) if parts else None
    return str(value)[:120]


def _detect_empty_data_reason(events: List[Dict[str, Any]], request: QueryRequest) -> Optional[str]:
    for event in reversed(events):
        if event.get("type") != "step_complete":
            continue
        summary = event.get("result_summary")
        if not isinstance(summary, dict):
            continue
        if summary.get("type") != "data_container_result":
            continue
        shape = summary.get("shape")
        dims = summary.get("dims")
        if not isinstance(shape, list) or not isinstance(dims, list):
            continue
        if "time" in dims:
            time_index = dims.index("time")
            if time_index < len(shape) and shape[time_index] == 0:
                variable = summary.get("variable") or "selected variable"
                time_range = request.additional_context.get("workspace_context", {}).get("time_range")
                if isinstance(time_range, (list, tuple)) and len(time_range) == 2:
                    time_label = f"{time_range[0]} to {time_range[1]}"
                else:
                    time_label = "the requested time range"
                return (
                    f"No data is available for {variable} in the selected region during {time_label}. "
                    "Try a different time range or a region with coverage."
                )
    return None


_MAX_CACHED_TURNS = 4
_MAX_REUSABLE_RESULTS_PER_TURN = 2
_MAX_FINDINGS_PER_TURN = 3


def _extract_anchors(summary: Dict[str, Any], output_type: str) -> Dict[str, Any]:
    anchors: Dict[str, Any] = {}
    statistics = summary.get("statistics")
    if isinstance(statistics, dict):
        for key, value in statistics.items():
            if "threshold" not in str(key):
                continue
            compact_value = _compact_anchor_value(value)
            if compact_value is not None:
                anchors[str(key)] = compact_value

    if output_type == "event_detection_result":
        sd = summary.get("spatial_distribution") or {}
        if sd.get("centroid"):
            anchors["centroid"] = sd["centroid"]
        lon_range = _compact_anchor_value(sd.get("lon_range"))
        lat_range = _compact_anchor_value(sd.get("lat_range"))
        if lon_range is not None:
            anchors["lon_range"] = lon_range
        if lat_range is not None:
            anchors["lat_range"] = lat_range
        time_index_range = _compact_anchor_value(summary.get("time_index_range"))
        if time_index_range is not None:
            anchors["time_index_range"] = time_index_range
        focus_event = summary.get("strongest_event") or summary.get("largest_event") or {}
        if isinstance(focus_event, dict):
            if focus_event.get("center"):
                anchors["focus_coordinates"] = focus_event["center"]
            for key in ("event_id", "timestamp", "end_timestamp", "duration_days", "area_km2", "metric", "value"):
                compact_value = _compact_anchor_value(focus_event.get(key))
                if compact_value is not None:
                    anchors[key] = compact_value
    elif output_type == "timeseries_result":
        time_range = summary.get("time_range")
        compact_time_range = _compact_anchor_value(time_range)
        if compact_time_range is None and summary.get("time_start") and summary.get("time_end"):
            compact_time_range = {
                "start": summary["time_start"],
                "end": summary["time_end"],
            }
        if compact_time_range is not None:
            anchors["time_range"] = compact_time_range
        extrema = summary.get("extrema") or {}
        if extrema.get("max_time"):
            anchors["extrema_time"] = extrema["max_time"]
        if extrema.get("max_value") is not None:
            anchors["extrema_value"] = extrema["max_value"]
    elif output_type == "profile_result":
        point = _compact_anchor_value(summary.get("point") or summary.get("requested_point"))
        if point is not None:
            anchors["profile_point"] = point
        depth_range = _compact_anchor_value(summary.get("depth_range"))
        if depth_range is not None:
            anchors["depth_range"] = depth_range
        strongest_gradient = summary.get("strongest_gradient") or {}
        if strongest_gradient.get("depth_midpoint") is not None:
            anchors["representative_depth"] = strongest_gradient["depth_midpoint"]
    elif output_type == "spatial_field_result":
        extrema = summary.get("extrema") or {}
        if extrema.get("max_location") is not None:
            anchors["hotspot_coordinates"] = extrema["max_location"]
        if extrema.get("min_location") is not None:
            anchors["coldspot_coordinates"] = extrema["min_location"]
        lon_range = _compact_anchor_value(summary.get("lon_range"))
        lat_range = _compact_anchor_value(summary.get("lat_range"))
        if lon_range is not None:
            anchors["lon_range"] = lon_range
        if lat_range is not None:
            anchors["lat_range"] = lat_range
    elif output_type == "lag_correlation_result":
        if summary.get("optimal_lag") is not None:
            anchors["optimal_lag"] = summary["optimal_lag"]
        if summary.get("peak_correlation") is not None:
            anchors["peak_correlation"] = summary["peak_correlation"]
        elif summary.get("max_correlation") is not None:
            anchors["peak_correlation"] = summary["max_correlation"]
    return anchors


def _persist_turn_results(
    conversation_id: str,
    query: str,
    skill_id: Optional[str],
    result_summaries: Dict[str, Dict[str, Any]],
    synthesis: Optional[Dict[str, Any]],
    active_result_id: Optional[str],
) -> None:
    state = CONVERSATION_STATE.setdefault(conversation_id, {})
    turn_history: list = state.setdefault("turn_history", [])
    turn_id = int(state.get("next_turn_id") or 1)
    state["next_turn_id"] = turn_id + 1

    turn_history.append({
        "turn_id": turn_id,
        "query": query,
        "skill_id": skill_id,
        "summary": str((synthesis or {}).get("summary") or "").strip(),
        "findings": _compact_scientific_findings(synthesis),
        "reusable_results": _build_reusable_results(
            result_summaries=result_summaries,
            synthesis=synthesis,
            active_result_id=active_result_id,
        ),
    })

    while len(turn_history) > _MAX_CACHED_TURNS:
        turn_history.pop(0)

    state.pop("reusable_result_manifest", None)


async def _record_turn_memory_async(
    *,
    loop: asyncio.AbstractEventLoop,
    conversation_id: str,
    planner: SkillPlanner,
    turn_packet: Dict[str, Any],
) -> None:
    state = CONVERSATION_STATE.setdefault(conversation_id, {})
    turn_id = int(state.get("memory_next_turn_id") or 1)
    state["memory_next_turn_id"] = turn_id + 1
    packet = dict(turn_packet)
    packet["turn_id"] = turn_id
    try:
        await loop.run_in_executor(
            _POST_EXEC_POOL,
            lambda: record_turn_memory(
                conversation_state=state,
                turn_packet=packet,
                planner=planner,
            ),
        )
    except Exception:
        _logger.exception("Failed to record conversation memory.")


def _build_memory_turn_packet(
    *,
    user_query: str,
    effective_query: str,
    status: str,
    routing_mode: Optional[str],
    response: QueryResponse,
    generated_plan: Optional[Dict[str, Any]] = None,
    result_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    synthesis = response.synthesis if isinstance(response.synthesis, dict) else {}
    return _json_safe(
        {
            "user_query": user_query,
            "effective_query": effective_query,
            "status": status,
            "routing_mode": routing_mode,
            "skill_id": response.skill_id,
            "skills_used": response.skills_used,
            "clarification_question": response.clarification_question,
            "missing_fields": response.missing_fields,
            "failure_kind": response.failure_kind,
            "error": response.error,
            "plan_summary": response.plan_summary,
            "plan": _compact_plan_for_memory(generated_plan),
            "synthesis_summary": str(synthesis.get("summary") or "").strip(),
            "source_cards_count": len(response.source_cards or []),
            "result_summaries": _compact_result_summaries_for_memory(result_summaries or {}),
        }
    )


def _compact_plan_for_memory(plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    steps = []
    for step in plan.get("steps", [])[:8]:
        if not isinstance(step, dict):
            continue
        steps.append(
            {
                "step_id": step.get("step_id"),
                "tool": step.get("tool"),
                "save_as": step.get("save_as"),
                "params": _memory_safe_params(step.get("params")),
            }
        )
    return {
        "skill_id": plan.get("skill_id"),
        "skills_used": plan.get("skills_used"),
        "status": plan.get("status"),
        "steps": steps,
    }


def _memory_safe_params(params: Any) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    allowed_keys = {
        "variable",
        "variables",
        "lon_range",
        "lat_range",
        "time_range",
        "depth_range",
        "depth_value",
        "vertical_mode",
        "field_type",
        "event_type",
        "threshold",
        "season_filter",
        "aggregation",
    }
    return {
        key: value
        for key, value in params.items()
        if key in allowed_keys
    }


def _compact_result_summaries_for_memory(
    result_summaries: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for result_id, summary in list(result_summaries.items())[:8]:
        if not isinstance(summary, dict):
            continue
        item: Dict[str, Any] = {
            "result_id": result_id,
            "type": summary.get("type"),
        }
        for key in (
            "variable",
            "field_type",
            "event_type",
            "time_range",
            "lon_range",
            "lat_range",
            "depth_range",
            "point",
            "requested_point",
            "event_count",
            "statistics",
            "extrema",
            "spatial_distribution",
        ):
            if key in summary:
                item[key] = summary.get(key)
        compact.append(item)
    return compact


def _compact_scientific_findings(synthesis: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(synthesis, dict):
        return []

    compact_findings: List[Dict[str, Any]] = []
    for item in synthesis.get("scientific_findings", [])[:_MAX_FINDINGS_PER_TURN]:
        if not isinstance(item, dict):
            continue
        finding = str(item.get("finding") or "").strip()
        if not finding:
            continue
        compact_item: Dict[str, Any] = {"finding": finding}
        evidence = item.get("evidence")
        if isinstance(evidence, list):
            compact_item["evidence"] = [
                str(entry).strip()
                for entry in evidence
                if isinstance(entry, str) and entry.strip()
            ][:2]
        result_ids = item.get("result_ids")
        if isinstance(result_ids, list):
            compact_item["result_ids"] = [
                str(result_id).strip()
                for result_id in result_ids
                if isinstance(result_id, str) and result_id.strip()
            ][:2]
        compact_findings.append(compact_item)
    return compact_findings


def _build_reusable_results(
    result_summaries: Dict[str, Dict[str, Any]],
    synthesis: Optional[Dict[str, Any]],
    active_result_id: Optional[str],
) -> List[Dict[str, Any]]:
    source_result_ids: List[str] = []
    if active_result_id:
        source_result_ids.append(active_result_id)

    if isinstance(synthesis, dict):
        for finding in synthesis.get("scientific_findings", []):
            if not isinstance(finding, dict):
                continue
            for result_id in finding.get("result_ids", []):
                if isinstance(result_id, str) and result_id not in source_result_ids:
                    source_result_ids.append(result_id)

    if not source_result_ids:
        for result_id, summary in result_summaries.items():
            if _is_reusable_summary(summary):
                source_result_ids.append(result_id)
                break

    reusable_results: List[Dict[str, Any]] = []
    for result_id in source_result_ids:
        summary = result_summaries.get(result_id)
        if not _is_reusable_summary(summary):
            continue
        reusable_results.extend(_build_reusable_result_cards(result_id, summary))
        if len(reusable_results) >= _MAX_REUSABLE_RESULTS_PER_TURN:
            break

    return reusable_results[:_MAX_REUSABLE_RESULTS_PER_TURN]


def _is_reusable_summary(summary: Any) -> bool:
    return (
        isinstance(summary, dict)
        and str(summary.get("type") or "").strip() != "data_container_result"
    )


def _build_reusable_result_cards(result_id: str, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    output_type = str(summary.get("type") or "").strip()
    if output_type == "event_detection_result":
        cards = _build_event_reusable_result_cards(result_id, summary)
        if cards:
            return cards

    anchors = _extract_anchors(summary, output_type)
    return [
        {
            "label": _default_reusable_result_label(output_type),
            "source_result_id": result_id,
            "output_type": output_type,
            "one_sentence_summary": _build_generic_reusable_result_summary(summary, output_type),
            "anchors": anchors,
            "reuse_hint": _default_reuse_hint(output_type),
        }
    ]


def _build_event_reusable_result_cards(result_id: str, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    event_type = str(summary.get("event_type") or "event").strip()
    aggregate_anchors = _extract_anchors(summary, "event_detection_result")
    cards: List[Dict[str, Any]] = [
        {
            "label": "Event overview",
            "source_result_id": result_id,
            "output_type": "event_detection_result",
            "one_sentence_summary": _build_event_overview_summary(summary, event_type),
            "anchors": aggregate_anchors,
            "reuse_hint": (
                "Reuse when the user refers to the previously detected event region, centroid, timing, or threshold."
            ),
        }
    ]

    focus_source = summary.get("strongest_event")
    focus_label = "Focused event"
    if not isinstance(focus_source, dict) or not focus_source:
        focus_source = summary.get("largest_event")
        focus_label = "Largest event"

    if isinstance(focus_source, dict) and focus_source:
        focus_anchors = {
            key: compact_value
            for key in ("center", "event_id", "timestamp", "end_timestamp", "duration_days", "area_km2", "metric", "value")
            if (compact_value := _compact_anchor_value(focus_source.get(key))) is not None
        }
        if "center" in focus_anchors:
            focus_anchors["focus_coordinates"] = focus_anchors.pop("center")
        cards.append(
            {
                "label": focus_label,
                "source_result_id": result_id,
                "output_type": "event_detection_result",
                "one_sentence_summary": _build_event_focus_summary(focus_source, event_type),
                "anchors": focus_anchors,
                "reuse_hint": (
                    "Reuse when the user asks about the highlighted event center, duration, area, or intensity."
                ),
            }
        )

    return cards[:_MAX_REUSABLE_RESULTS_PER_TURN]


def _default_reusable_result_label(output_type: str) -> str:
    labels = {
        "spatial_field_result": "Spatial hotspot",
        "profile_result": "Profile anchor",
        "timeseries_result": "Time-series anchor",
        "event_comparison_result": "Event comparison",
        "event_statistics_result": "Event statistics",
        "lag_correlation_result": "Lag-correlation anchor",
    }
    return labels.get(output_type, output_type.replace("_", " ") or "Reusable result")


def _build_event_overview_summary(summary: Dict[str, Any], event_type: str) -> str:
    parts = [f"Detected {summary.get('event_count', 0)} {event_type} events"]
    centroid = ((summary.get("spatial_distribution") or {}).get("centroid") if isinstance(summary.get("spatial_distribution"), dict) else None)
    if centroid:
        parts.append(f"centroid near {_format_point(centroid)}")
    lon_range = _format_range((summary.get("spatial_distribution") or {}).get("lon_range"))
    lat_range = _format_range((summary.get("spatial_distribution") or {}).get("lat_range"))
    if lon_range and lat_range:
        parts.append(f"spanning lon {lon_range} and lat {lat_range}")
    time_index_range = _format_range(summary.get("time_index_range"))
    if time_index_range:
        parts.append(f"time indices {time_index_range}")
    threshold_text = _format_thresholds(summary.get("statistics"))
    if threshold_text:
        parts.append(threshold_text)
    return "; ".join(parts) + "."


def _build_event_focus_summary(event: Dict[str, Any], event_type: str) -> str:
    parts: List[str] = []
    event_id = str(event.get("event_id") or "").strip()
    if event_id:
        parts.append(f"{event_type} event {event_id}")
    else:
        parts.append(f"Highlighted {event_type} event")
    if event.get("center"):
        parts.append(f"centered near {_format_point(event['center'])}")
    if event.get("timestamp"):
        parts.append(f"starting around {event['timestamp']}")
    if event.get("duration_days") is not None:
        parts.append(f"duration {_format_number(event['duration_days'])} days")
    if event.get("area_km2") is not None:
        parts.append(f"area {_format_number(event['area_km2'])} km^2")
    metric = str(event.get("metric") or "").strip()
    if metric and event.get("value") is not None:
        parts.append(f"{metric}={_format_number(event['value'])}")
    return "; ".join(parts) + "."


def _build_generic_reusable_result_summary(summary: Dict[str, Any], output_type: str) -> str:
    if output_type == "spatial_field_result":
        extrema = summary.get("extrema") or {}
        parts = ["Previously computed spatial field"]
        if extrema.get("max_location") is not None:
            parts.append(f"hotspot near {_format_point(extrema['max_location'])}")
        if extrema.get("max_value") is not None:
            parts.append(f"max value {_format_number(extrema['max_value'])}")
        lon_range = _format_range(summary.get("lon_range"))
        lat_range = _format_range(summary.get("lat_range"))
        if lon_range and lat_range:
            parts.append(f"map spans lon {lon_range} and lat {lat_range}")
        return "; ".join(parts) + "."

    if output_type == "profile_result":
        parts = ["Previously extracted vertical profile"]
        point = summary.get("point") or summary.get("requested_point")
        if point is not None:
            parts.append(f"at {_format_point(point)}")
        depth_range = _format_range(summary.get("depth_range"))
        if depth_range:
            parts.append(f"depth range {depth_range}")
        strongest_gradient = summary.get("strongest_gradient") or {}
        if strongest_gradient.get("depth_midpoint") is not None:
            parts.append(
                f"strongest transition near {_format_number(strongest_gradient['depth_midpoint'])} m"
            )
        return "; ".join(parts) + "."

    if output_type == "timeseries_result":
        parts = ["Previously computed time series"]
        time_range = _format_range(summary.get("time_range"))
        if time_range:
            parts.append(f"covering {time_range}")
        extrema = summary.get("extrema") or {}
        if extrema.get("max_time"):
            parts.append(f"peak near {extrema['max_time']}")
        if extrema.get("max_value") is not None:
            parts.append(f"peak value {_format_number(extrema['max_value'])}")
        return "; ".join(parts) + "."

    summary_bits: List[str] = [f"Reusable {output_type.replace('_', ' ')} result"]
    threshold_text = _format_thresholds(summary.get("statistics"))
    if threshold_text:
        summary_bits.append(threshold_text)
    return "; ".join(summary_bits) + "."


def _default_reuse_hint(output_type: str) -> str:
    hints = {
        "spatial_field_result": "Reuse when the user refers to the hotspot, coldspot, or map location identified earlier.",
        "profile_result": "Reuse when the user refers to the previous profile location, depth range, or strongest transition depth.",
        "timeseries_result": "Reuse when the user refers to the previous time window, peak timing, or extrema.",
        "event_comparison_result": "Reuse when the user refers to the previously compared periods or their contrast.",
        "event_statistics_result": "Reuse when the user refers to the previously identified dominant event group or ranking.",
        "lag_correlation_result": "Reuse when the user refers to the previously identified optimal lag or correlation strength.",
    }
    return hints.get(output_type, "Reuse when the current query refers to this earlier analysis result.")


def _context_with_role_text(base_context: Dict[str, Any], **role_payloads: Any) -> Dict[str, Any]:
    merged = {
        key: value
        for key, value in base_context.items()
        if key not in {
            "available_prior_results",
            "prior_queries_text",
            "planner_prior_context_text",
            "synthesizer_prior_context_text",
            "conversation_context",
            "conversation_memory",
        }
    }
    for key, value in role_payloads.items():
        if isinstance(value, str) and value.strip():
            merged[key] = value
        elif isinstance(value, dict):
            merged[key] = value
        elif isinstance(value, list):
            merged[key] = value
    return merged


def _merge_memory_resolved_entities(
    extracted_params: Mapping[str, Any],
    resolved_entities: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged = dict(extracted_params or {})
    if not isinstance(resolved_entities, Mapping):
        return merged

    for key in ("named_regions", "known_named_region_bounds", "region_name", "region_source"):
        value = resolved_entities.get(key)
        if value is not None and key not in merged:
            merged[key] = value

    has_explicit_bounds = "lon_range" in merged and "lat_range" in merged
    if has_explicit_bounds:
        return merged

    lon_range = resolved_entities.get("lon_range")
    lat_range = resolved_entities.get("lat_range")
    if "lon_range" not in merged and isinstance(lon_range, list) and len(lon_range) == 2:
        merged["lon_range"] = list(lon_range)
    if "lat_range" not in merged and isinstance(lat_range, list) and len(lat_range) == 2:
        merged["lat_range"] = list(lat_range)

    for key in ("region", "region_bounds"):
        value = resolved_entities.get(key)
        if value is not None and key not in merged:
            merged[key] = value
    return merged


def _merge_named_region_resolved_entities(
    extracted_params: Mapping[str, Any],
    user_query: str,
    planner: Any,
) -> Dict[str, Any]:
    resolver = getattr(planner, "resolve_named_region_entities", None)
    if not callable(resolver):
        return dict(extracted_params or {})

    resolved_entities = resolver(user_query)
    return _merge_memory_resolved_entities(extracted_params, resolved_entities)


def _build_prior_queries_text(turns: List[Dict[str, Any]]) -> str:
    lines = [
        "Possible prior user queries that may help interpret follow-up intent. Ignore them if the current query is standalone.",
    ]
    for turn in reversed(turns[-4:]):
        query = str(turn.get("query") or "").strip()
        if not query:
            continue
        lines.append(f"- Turn {turn.get('turn_id', '?')}: {query}")
    return "\n".join(lines)


def _build_planner_prior_context_text(turns: List[Dict[str, Any]]) -> str:
    blocks = [
        "Possible prior analysis results that may be reusable for the current query. "
        "These are compressed summaries, not full outputs. Use them only when they help resolve references or missing parameters.",
    ]
    for turn in reversed(turns[-4:]):
        turn_block = _render_planner_turn_block(turn)
        if turn_block:
            blocks.append(turn_block)
    return "\n\n".join(blocks)


def _build_synthesizer_prior_context_text(turns: List[Dict[str, Any]]) -> str:
    blocks = [
        "Possible earlier findings and supporting results from prior turns. Use them only when building cumulative explanation, "
        "and distinguish prior evidence from current-turn evidence.",
    ]
    for turn in reversed(turns[-4:]):
        turn_block = _render_synthesizer_turn_block(turn)
        if turn_block:
            blocks.append(turn_block)
    return "\n\n".join(blocks)


def _render_planner_turn_block(turn: Dict[str, Any]) -> str:
    query = str(turn.get("query") or "").strip()
    results = turn.get("reusable_results")
    summary = str(turn.get("summary") or "").strip()
    if not query and not results:
        return ""

    lines = [f"Turn {turn.get('turn_id', '?')}", f"Query: {query or '(missing query)'}"]
    if summary:
        lines.append(f"Summary: {summary}")
    if isinstance(results, list) and results:
        lines.append("Reusable results:")
        for card in results[:_MAX_REUSABLE_RESULTS_PER_TURN]:
            if not isinstance(card, dict):
                continue
            lines.append(f"- {_render_reusable_result_line(card)}")
    return "\n".join(lines)


def _render_synthesizer_turn_block(turn: Dict[str, Any]) -> str:
    query = str(turn.get("query") or "").strip()
    findings = turn.get("findings")
    results = turn.get("reusable_results")
    summary = str(turn.get("summary") or "").strip()
    if not query and not findings and not results:
        return ""

    lines = [f"Turn {turn.get('turn_id', '?')}", f"Query: {query or '(missing query)'}"]
    if summary:
        lines.append(f"Summary: {summary}")
    if isinstance(findings, list) and findings:
        lines.append("Key findings:")
        for finding in findings[:_MAX_FINDINGS_PER_TURN]:
            if not isinstance(finding, dict):
                continue
            text = str(finding.get("finding") or "").strip()
            if not text:
                continue
            lines.append(f"- {text}")
    if isinstance(results, list) and results:
        lines.append("Supporting reusable results:")
        for card in results[:_MAX_REUSABLE_RESULTS_PER_TURN]:
            if not isinstance(card, dict):
                continue
            lines.append(f"- {_render_reusable_result_line(card)}")
    return "\n".join(lines)


def _render_reusable_result_line(card: Dict[str, Any]) -> str:
    label = str(card.get("label") or "Reusable result").strip()
    source_result_id = str(card.get("source_result_id") or "").strip()
    output_type = str(card.get("output_type") or "").strip()
    summary = str(card.get("one_sentence_summary") or "").strip()
    reuse_hint = str(card.get("reuse_hint") or "").strip()
    anchors = card.get("anchors")

    parts = [label]
    metadata = [item for item in (source_result_id, output_type) if item]
    if metadata:
        parts.append(f"[{' / '.join(metadata)}]")
    if summary:
        parts.append(summary)
    if reuse_hint:
        parts.append(f"Reuse hint: {reuse_hint}")
    anchor_text = _format_anchor_summary(anchors)
    if anchor_text:
        parts.append(f"Anchors: {anchor_text}")
    return " ".join(parts)


def _format_anchor_summary(anchors: Any) -> str:
    if not isinstance(anchors, dict) or not anchors:
        return ""

    chunks: List[str] = []
    for key, value in anchors.items():
        if len(chunks) >= 4:
            break
        if key in {"centroid", "focus_coordinates", "hotspot_coordinates", "coldspot_coordinates", "profile_point"}:
            chunks.append(f"{key}={_format_point(value)}")
            continue
        if isinstance(value, (list, tuple)):
            formatted = _format_range(value)
            if formatted:
                chunks.append(f"{key}={formatted}")
            continue
        if isinstance(value, dict) and {"start", "end"} <= set(value.keys()):
            chunks.append(f"{key}={value['start']} to {value['end']}")
            continue
        compact = _compact_anchor_value(value)
        if compact is not None:
            chunks.append(f"{key}={compact}")
    return "; ".join(chunks)


def _format_thresholds(statistics: Any) -> str:
    if not isinstance(statistics, dict):
        return ""

    threshold_parts = []
    for key, value in statistics.items():
        if "threshold" not in str(key):
            continue
        if not isinstance(value, (int, float, str)):
            continue
        threshold_parts.append(f"{key}={_format_number(value)}")
    return ", ".join(threshold_parts[:2])


def _compact_anchor_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        compact: Dict[str, Any] = {}
        for key in ("lon", "lat", "start", "end", "event_id", "timestamp", "end_timestamp", "duration_days", "area_km2", "metric", "value"):
            if key in value and isinstance(value.get(key), (bool, int, float, str, list, tuple, dict)):
                compact[key] = value[key]
        return compact or None
    if isinstance(value, (list, tuple)) and len(value) <= 2:
        if all(isinstance(item, (int, float, str)) for item in value):
            return list(value)
    return None


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if abs(value) >= 100:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        if abs(value) >= 1:
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{value:.3g}"
    return str(value)


def _format_point(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value)
    lon = value.get("lon")
    lat = value.get("lat")
    if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
        return f"({_format_number(lon)}, {_format_number(lat)})"
    return str(value)


def _format_range(value: Any) -> str:
    if isinstance(value, dict):
        start = value.get("start")
        end = value.get("end")
        if start is not None and end is not None:
            return f"{start} to {end}"
        return ""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{_format_number(value[0])} to {_format_number(value[1])}"
    return ""


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _is_chinese_query(text: str) -> bool:
    return _contains_cjk(text or "")


def _localized(english: str, chinese: str, *, chinese_mode: bool) -> str:
    return chinese if chinese_mode else english


def _localize_enum_value(value: Any, *, chinese: bool) -> str:
    if not isinstance(value, str):
        return str(value)
    lowered = value.strip().lower()
    mapping = {
        "supported": "支持较强",
        "limited": "证据有限",
        "untestable": "现有变量下不可检验",
        "not testable with current variables": "现有变量下不可检验",
        "deteriorating": "恶化",
        "improving": "改善",
        "stable": "基本稳定",
        "background like": "接近背景态",
        "background_like": "接近背景态",
        "event condition shift": "事件条件偏移",
        "event_condition_shift": "事件条件偏移",
        "stratification control": "层化控制",
        "stratification_control": "层化控制",
        "weak response": "响应较弱",
        "weak_response": "响应较弱",
        "transport dominated": "以输运为主",
        "transport_dominated": "以输运为主",
        "local accumulation": "以局地累积为主",
        "local_accumulation": "以局地累积为主",
        "front proximity": "锋面代理",
        "front_proximity": "锋面代理",
        "eddy influence": "涡旋代理",
        "eddy_influence": "涡旋代理",
        "gradient alignment": "梯度-流向对齐代理",
        "gradient_alignment": "梯度-流向对齐代理",
        "flow context": "背景流场代理",
        "flow_context": "背景流场代理",
        "raw": "原始序列",
        "deseasoned": "去季节序列",
        "mixed": "混合输入",
        "month of year": "按月份去季节",
        "month_of_year": "按月份去季节",
        "calendar day": "按日历日去季节",
        "calendar_day": "按日历日去季节",
        "uncertain": "不确定",
    }
    if chinese:
        return mapping.get(lowered, value)
    return value.replace("_", " ")


def _localize_metric_label(label: str, *, chinese: bool) -> str:
    if not chinese:
        return label
    mapping = {
        "period": "周期",
        "n_labels": "标签数",
        "amplitude": "振幅",
        "positive_n": "正位相样本数",
        "negative_n": "负位相样本数",
        "diff_mean": "差值均值",
        "significant_pct": "显著比例",
        "slope_min": "最小斜率",
        "slope_max": "最大斜率",
        "corr_max": "最大相关",
        "peak_period": "峰值周期",
        "peak_freq": "峰值频率",
        "peak_power": "峰值功率",
        "n_freq": "频率数",
        "n_points": "样本点数",
        "temp_range": "温度范围",
        "salt_range": "盐度范围",
        "total_count": "总事件数",
        "group_by": "分组方式",
        "n_groups": "分组数",
        "top_group": "最高发分组",
        "period_1_count": "时期一事件数",
        "period_2_count": "时期二事件数",
        "count_change_pct": "事件数变化率",
        "count_change": "事件数变化",
        "intensity_change": "强度变化",
        "lag_mode": "滞后分析口径",
        "optimal_lag": "最优滞后",
        "optimal_lag_days": "最优滞后天数",
        "lag_step_days": "时间步长",
        "max_corr": "最大相关",
        "zero_lag": "零滞后相关",
        "support": "支持度",
        "valid_tiles": "有效格子数",
        "hotspot_tiles": "热点格子数",
        "assoc_score": "关联分数",
        "top_score": "最高分",
        "candidates": "候选数",
        "supported_n": "支持项数",
        "limited_n": "有限支持项数",
        "untestable_n": "不可检验项数",
        "verdict": "结论",
        "branches": "分支数",
        "subregions": "子区数",
        "valid_subregions": "有效子区数",
        "shape": "形状",
        "units": "单位",
        "time": "时间",
        "depth": "深度",
        "lon": "经度",
        "lat": "纬度",
        "mean": "均值",
        "std": "标准差",
        "min": "最小值",
        "max": "最大值",
        "trend": "趋势",
        "p_value": "P 值",
        "r_squared": "R²",
        "event_count": "事件数",
        "max_value": "最大值",
        "min_value": "最小值",
    }
    return mapping.get(label, label)


def _localize_metric_entries(metrics: List[Dict[str, str]], *, chinese: bool) -> List[Dict[str, str]]:
    if not chinese:
        return metrics
    localized_metrics: List[Dict[str, str]] = []
    for metric in metrics:
        localized_metrics.append(
            {
                **metric,
                "label": _localize_metric_label(str(metric.get("label") or ""), chinese=True),
                "value": _localize_enum_value(metric.get("value"), chinese=True),
            }
        )
    return localized_metrics


def _localize_explicit_title(title: str, *, chinese: bool) -> str:
    if not chinese:
        return title
    mapping = {
        "Bloom Chlorophyll Burden": "藻华叶绿素累积负荷图",
        "Bloom Event Days": "藻华事件天数图",
        "Marine Heatwave Burden": "海洋热浪累积强度图",
        "Marine Heatwave Days": "海洋热浪天数图",
        "Hypoxia Oxygen Deficit Burden": "低氧氧亏累积负荷图",
        "Hypoxic Days": "低氧天数图",
        "Upwelling Cold-Anomaly Burden": "上升流冷异常累积强度图",
        "Upwelling Days": "上升流天数图",
        "Eutrophication Chlorophyll Burden": "富营养化叶绿素累积负荷图",
        "Eutrophic Days": "富营养化天数图",
        "Environment Health Assessment": "海洋环境健康评估",
        "Evidence Report": "证据报告",
        "Mechanism Ranking": "机制排序",
        "Event-Watermass Tile Association": "事件-水团格网关联",
        "Bloom Hotspot Tile Map": "藻华热点格网图",
        "Heatwave Hotspot Tile Map": "热浪热点格网图",
        "Hypoxia Hotspot Tile Map": "低氧热点格网图",
        "Upwelling Hotspot Tile Map": "上升流热点格网图",
        "Dominant Watermass Tile Map": "主导水团格网图",
        "Watermass Definition T-S Diagram": "水团定义温盐图",
    }
    return mapping.get(title, title)


def _select_environment_assessment_summary(
    result_summaries: Dict[str, Dict[str, Any]],
    active_result_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    if active_result_id:
        active_summary = result_summaries.get(active_result_id)
        if isinstance(active_summary, dict) and active_summary.get("type") == "environment_assessment_result":
            return active_summary
    for summary in result_summaries.values():
        if isinstance(summary, dict) and summary.get("type") == "environment_assessment_result":
            return summary
    return None


def _environment_indicator_label(label: Any, name: Any = None, *, chinese: bool) -> str:
    raw = str(label or name or "").strip()
    if not raw:
        return ""
    if not chinese:
        return raw
    mapping = {
        "sea-surface temperature trend": "海表温度趋势",
        "sst trend": "海表温度趋势",
        "sst_trend": "海表温度趋势",
        "bloom-event frequency change": "藻华事件频率变化",
        "bloom frequency change": "藻华事件频率变化",
        "bloom_frequency_change": "藻华事件频率变化",
        "bottom-oxygen trend": "底层氧气趋势",
        "bottom oxygen trend": "底层氧气趋势",
        "bottom_oxygen_trend": "底层氧气趋势",
        "stratification-strength change": "层化强度变化",
        "stratification strength change": "层化强度变化",
        "stratification_strength_change": "层化强度变化",
        "sst": "海表温度",
        "temperature": "温度",
        "oxygen": "氧气",
        "stratification": "层化",
        "bloom": "藻华",
    }
    return mapping.get(raw.strip().lower(), raw)


def _environment_branch_priority_local(branch: Dict[str, Any]) -> tuple[float, float]:
    support_rank = {"supported": 2.0, "limited": 1.0, "untestable": 0.0}
    return (
        support_rank.get(str(branch.get("support_strength") or "").strip().lower(), 0.0),
        abs(float(branch.get("score_contribution") or 0.0)),
    )


def _extract_environment_trend_numbers(summary_text: str) -> Dict[str, str]:
    extracted: Dict[str, str] = {}
    slope_match = re.search(r"slope ([^,)\s]+)", summary_text)
    p_match = re.search(r"p=([^)]+)", summary_text)
    fitted_match = re.search(r"fitted change ([^)\s]+)", summary_text)
    if slope_match:
        extracted["slope"] = slope_match.group(1)
    if p_match:
        extracted["p"] = p_match.group(1)
    if fitted_match:
        extracted["fitted_change"] = fitted_match.group(1)
    return extracted


def _extract_environment_event_numbers(summary_text: str) -> Dict[str, str]:
    extracted: Dict[str, str] = {}
    earlier_match = re.search(r"earlier-half mean of ([^\s]+)", summary_text)
    later_match = re.search(r"later-half mean of ([^\s]+)", summary_text)
    change_match = re.search(r"\(([-+0-9.eE]+)% relative change\)", summary_text)
    bins_match = re.search(r"based on only ([0-9]+) yearly bins", summary_text)
    if earlier_match:
        extracted["earlier_mean"] = earlier_match.group(1)
    if later_match:
        extracted["later_mean"] = later_match.group(1)
    if change_match:
        extracted["relative_change_pct"] = change_match.group(1)
    if bins_match:
        extracted["n_bins"] = bins_match.group(1)
    return extracted


def _localize_environment_branch_summary(branch: Dict[str, Any], *, chinese: bool) -> str:
    summary_text = str(branch.get("summary") or "").strip()
    if not chinese or not summary_text:
        return summary_text
    if _contains_cjk(summary_text):
        return summary_text

    label = _environment_indicator_label(branch.get("indicator_label"), branch.get("name"), chinese=True) or "该指标"
    direction = str(branch.get("direction") or "").strip().lower()
    output_type = str(branch.get("output_type") or "").strip().lower()
    lowered = summary_text.lower()

    if output_type == "trend_result":
        values = _extract_environment_trend_numbers(summary_text)
        if "could not be evaluated because no usable result was supplied" in lowered:
            return f"{label}无法评估，因为没有可用结果。"
        if "lacks the required slope or fitted-change diagnostics" in lowered:
            return f"{label}缺少必要的斜率或拟合变化诊断量。"
        direction_text = {
            "deteriorating": "呈恶化方向变化",
            "improving": "呈改善方向变化",
            "stable": "方向性变化不明显",
            "untestable": "当前变量下不可判断变化方向",
        }.get(direction, "变化方向待判定")
        if "slope" in values:
            if "p" in values:
                return f"{label}{direction_text}（斜率 {values['slope']}，p={values['p']}）。"
            return f"{label}{direction_text}（斜率 {values['slope']}）。"
        if "fitted_change" in values:
            return f"{label}{direction_text}（拟合变化量 {values['fitted_change']}）。"
        return f"{label}{direction_text}。"

    if output_type == "event_statistics_result":
        values = _extract_environment_event_numbers(summary_text)
        if "statistics were not grouped by year" in lowered:
            return f"{label}无法比较，因为统计结果未按年份分组。"
        if "no yearly event counts were available" in lowered:
            return f"{label}无法比较，因为没有可用的年度事件计数。"
        if "fewer than two yearly bins were available" in lowered:
            return f"{label}无法比较，因为年度分箱少于两个。"
        if "could not be split into earlier and later periods" in lowered:
            return f"{label}无法拆分为前后两个时期进行比较。"
        if "shows little early-versus-late change" in lowered:
            return f"{label}在前后两个时期之间变化较小。"
        if {"earlier_mean", "later_mean", "relative_change_pct"} <= set(values):
            return (
                f"{label}前半段平均值为 {values['earlier_mean']}，后半段平均值为 {values['later_mean']}，"
                f"相对变化为 {values['relative_change_pct']}%。"
            )
        return f"{label}的年度事件统计变化已完成评估。"

    if "not yet supported as an environment-health branch" in lowered:
        return f"{label}暂不支持作为环境健康评估分支。"
    return summary_text


def _trim_environment_summary_prefix(summary_text: str, label: str) -> str:
    trimmed = summary_text.strip()
    if not trimmed or not label:
        return trimmed
    prefixes = (
        label,
        f"{label}：",
        f"{label}:",
    )
    for prefix in prefixes:
        if trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix):].lstrip("：:，,；; ")
            break
    return trimmed


def _trim_sentence_ending(text: str) -> str:
    return str(text).strip().rstrip("。；;,. ")


def _localize_environment_pressure_list(branches: List[Dict[str, Any]], *, chinese: bool) -> List[str]:
    if not chinese:
        return [
            f"{str(item.get('indicator_label') or item.get('name') or 'Indicator').strip()}: {str(item.get('summary') or '').strip()}"
            for item in branches
        ]
    localized_items: List[str] = []
    for item in branches:
        label = _environment_indicator_label(item.get("indicator_label"), item.get("name"), chinese=True)
        summary_text = _localize_environment_branch_summary(item, chinese=True)
        trimmed_summary = _trim_environment_summary_prefix(summary_text, label)
        if summary_text.startswith(label):
            localized_items.append(_trim_sentence_ending(summary_text))
        elif trimmed_summary:
            localized_items.append(f"{label}：{_trim_sentence_ending(trimmed_summary)}")
        else:
            localized_items.append(label)
    return localized_items


def _build_environment_monitoring_priorities_localized(
    branches: List[Dict[str, Any]],
    *,
    overall_verdict: str,
    overall_support_strength: str,
    chinese: bool,
) -> List[str]:
    if not chinese:
        return [str(item) for item in []]
    priorities: List[str] = []
    watch_list = sorted(
        [
            branch for branch in branches
            if branch.get("support_strength") in {"limited", "untestable"}
        ],
        key=_environment_branch_priority_local,
        reverse=True,
    )
    for branch in watch_list[:3]:
        label = _environment_indicator_label(branch.get("indicator_label"), branch.get("name"), chinese=True) or "该指标"
        direction = str(branch.get("direction") or "").strip().lower()
        support = str(branch.get("support_strength") or "").strip().lower()
        if support == "untestable":
            priorities.append(f"建议加强{label}监测，因为当前时间窗还不足以支持明确的方向判断。")
        elif direction == "deteriorating":
            priorities.append(f"建议优先跟踪{label}，因为它指向环境恶化，但当前证据仍然有限。")
        else:
            priorities.append(f"建议持续关注{label}，以确认当前弱信号是否会持续。")

    if not priorities:
        verdict_label = _localize_enum_value(overall_verdict, chinese=True)
        if overall_verdict == "Deteriorating":
            priorities.append("建议保持跨指标常规监测，以便进一步识别恶化信号出现的季节和子区域。")
        elif overall_verdict == "Improving":
            priorities.append("建议维持基线监测，以确认改善信号具有持续性而非短期波动。")
        else:
            priorities.append(f"建议维持各指标基线监测，因为当前时间窗显示海洋环境总体{verdict_label}。")

    if str(overall_support_strength).strip().lower() == "untestable":
        priorities.append("在做出高成本管理调整前，应优先提高时间覆盖率并补齐关键指标观测。")

    return priorities[:4]


def _build_environment_policy_recommendations_localized(
    branches: List[Dict[str, Any]],
    *,
    overall_verdict: str,
    overall_support_strength: str,
    chinese: bool,
) -> List[str]:
    if not chinese:
        return [str(item) for item in []]
    recommendations: List[str] = []
    for branch in sorted(branches, key=_environment_branch_priority_local, reverse=True):
        if branch.get("direction") != "deteriorating":
            continue
        if branch.get("support_strength") not in {"supported", "limited"}:
            continue
        label = str(branch.get("indicator_label") or "").strip()
        name = str(branch.get("name") or "").strip()
        summary_text = str(branch.get("summary") or "").strip()
        search_text = f"{label} {name} {summary_text}".lower()
        if any(token in search_text for token in ("temperature", "sst", "heatwave", "海表温度", "温度趋势", "热浪")):
            recommendations.append("应加强海洋热风险应对，扩大暖季监测，并面向渔业和养殖业制定适应性管理与高温预警。")
        elif any(token in search_text for token in ("bloom", "chlorophyll", "eutroph", "藻华", "叶绿素", "富营养")):
            recommendations.append("应优先推进营养盐管理和藻华监测，重点关注径流影响显著季节及高风险热点海区。")
        elif any(token in search_text for token in ("oxygen", "hypoxia", "底层氧气", "低氧", "氧气趋势")):
            recommendations.append("应扩大底层氧气监测，并减少脆弱近岸海域的营养盐或有机负荷，以提前干预低氧风险。")
        elif any(token in search_text for token in ("stratification", "层化")):
            recommendations.append("应将层化增强作为管理预警信号，在高风险季节加强垂向剖面观测，并与低氧和藻华响应预案联动。")

    if overall_verdict == "Deteriorating":
        if str(overall_support_strength).strip().lower() == "supported":
            recommendations.append("建议将该区域列为更高优先级的海洋环境管理区域，并围绕主要恶化指标配置季节性监测、减缓和执法资源。")
        else:
            recommendations.append("建议将本次评估作为预警依据，优先采取针对性监测和低后悔成本减缓措施，而非立即实施不可逆政策调整。")
    elif overall_verdict == "Improving":
        recommendations.append("建议维持当前与改善信号一致的保护和管理措施，同时继续验证积极趋势是否能够持续。")
    else:
        recommendations.append("建议保持基础保护措施，并通过分指标监测及时识别环境状态是否转向更明确的恶化或恢复。")

    deduped: List[str] = []
    seen: set[str] = set()
    for item in recommendations:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped[:5]


def _build_environment_narrative_localized(branches: List[Dict[str, Any]], *, verdict: str, support: str, chinese: bool) -> str:
    if not chinese:
        return ""
    ordered = sorted(branches, key=_environment_branch_priority_local, reverse=True)
    deteriorating = [
        _environment_indicator_label(item.get("indicator_label"), item.get("name"), chinese=True)
        for item in ordered
        if item.get("direction") == "deteriorating" and item.get("support_strength") in {"supported", "limited"}
    ][:2]
    improving = [
        _environment_indicator_label(item.get("indicator_label"), item.get("name"), chinese=True)
        for item in ordered
        if item.get("direction") == "improving" and item.get("support_strength") in {"supported", "limited"}
    ][:2]

    if verdict == "Deteriorating":
        lead = "综合指标整体指向海洋环境恶化"
    elif verdict == "Improving":
        lead = "综合指标整体指向海洋环境改善"
    else:
        lead = "综合指标显示海洋环境总体较为稳定"

    sentence = f"{lead}，总体证据强度为{_localize_enum_value(support, chinese=True)}。"
    if deteriorating:
        sentence += f" 主要压力来自{ '、'.join(deteriorating) }。"
    if improving:
        sentence += f" 同时，{ '、'.join(improving) }提供了一定的改善信号。"
    return sentence


def _localize_environment_assessment_summary(summary: Dict[str, Any], *, chinese: bool) -> Dict[str, Any]:
    if not isinstance(summary, dict) or summary.get("type") != "environment_assessment_result" or not chinese:
        return summary
    if str(summary.get("localized_language") or "").strip().lower() == "zh":
        return summary

    localized = dict(summary)
    branches_raw = summary.get("branch_assessments")
    localized_branches: List[Dict[str, Any]] = []
    if isinstance(branches_raw, list):
        for item in branches_raw:
            if not isinstance(item, dict):
                continue
            branch_copy = dict(item)
            branch_copy["indicator_label"] = _environment_indicator_label(item.get("indicator_label"), item.get("name"), chinese=True)
            branch_copy["summary"] = _localize_environment_branch_summary(item, chinese=True)
            localized_branches.append(branch_copy)
        localized["branch_assessments"] = localized_branches

    verdict = str(summary.get("overall_verdict") or "").strip()
    support = str(summary.get("overall_support_strength") or "").strip()
    localized["overall_narrative"] = _build_environment_narrative_localized(
        localized_branches,
        verdict=verdict,
        support=support,
        chinese=True,
    )

    deteriorating = [
        item for item in localized_branches
        if item.get("direction") == "deteriorating" and item.get("support_strength") in {"supported", "limited"}
    ]
    improving = [
        item for item in localized_branches
        if item.get("direction") == "improving" and item.get("support_strength") in {"supported", "limited"}
    ]
    localized["key_pressures"] = _localize_environment_pressure_list(
        sorted(deteriorating, key=_environment_branch_priority_local, reverse=True)[:4],
        chinese=True,
    )
    localized["stabilizing_signals"] = _localize_environment_pressure_list(
        sorted(improving, key=_environment_branch_priority_local, reverse=True)[:3],
        chinese=True,
    )
    localized["monitoring_priorities"] = _build_environment_monitoring_priorities_localized(
        localized_branches,
        overall_verdict=verdict,
        overall_support_strength=support,
        chinese=True,
    )
    localized["policy_recommendations"] = _build_environment_policy_recommendations_localized(
        localized_branches,
        overall_verdict=verdict,
        overall_support_strength=support,
        chinese=True,
    )
    localized["localized_language"] = "zh"
    return localized


def _format_environment_branch_line(branch: Dict[str, Any], *, chinese: bool) -> Optional[str]:
    label = str(branch.get("indicator_label") or branch.get("name") or "").strip()
    direction = str(branch.get("direction") or "").strip()
    support = str(branch.get("support_strength") or "").strip()
    summary = str(branch.get("summary") or "").strip()
    if not label:
        return None
    if chinese:
        line = f"{label}：{_localize_enum_value(direction, chinese=True)}"
        if support:
            line += f"（{_localize_enum_value(support, chinese=True)}）"
        if summary:
            line += f"，{_trim_sentence_ending(_trim_environment_summary_prefix(summary, label))}"
        return line
    line = f"{label}: {direction}"
    if support:
        line += f" ({support})"
    if summary:
        line += f"; {summary}"
    return line


def _build_environment_assessment_analysis_summary(
    summary: Dict[str, Any],
    *,
    user_request: str,
    existing_summary: Optional[str] = None,
) -> str:
    chinese = False
    summary = _localize_environment_assessment_summary(summary, chinese=chinese)
    verdict = str(summary.get("overall_verdict") or "").strip()
    support = str(summary.get("overall_support_strength") or "").strip()
    narrative = str(summary.get("overall_narrative") or "").strip()
    branches = summary.get("branch_assessments") if isinstance(summary.get("branch_assessments"), list) else []
    branch_lines = [
        _trim_sentence_ending(line)
        for line in (
            _format_environment_branch_line(item, chinese=chinese)
            for item in branches[:4]
            if isinstance(item, dict)
        )
        if line
    ]
    key_pressures = [_trim_sentence_ending(item) for item in summary.get("key_pressures", []) if str(item).strip()][:3]
    stabilizing_signals = [_trim_sentence_ending(item) for item in summary.get("stabilizing_signals", []) if str(item).strip()][:2]

    if chinese:
        sentences: List[str] = []
        if narrative:
            sentences.append(narrative)
        elif verdict and support:
            sentences.append(f"综合评估结论为 {_localize_enum_value(verdict, chinese=True)}，总体证据强度为 {_localize_enum_value(support, chinese=True)}。")
        elif verdict:
            sentences.append(f"综合评估结论为 {_localize_enum_value(verdict, chinese=True)}。")
        if branch_lines:
            sentences.append("各分支结果：" + "；".join(branch_lines) + "。")
        if key_pressures:
            sentences.append("主要环境压力包括：" + "；".join(key_pressures) + "。")
        if stabilizing_signals:
            sentences.append("相对稳定或改善信号包括：" + "；".join(stabilizing_signals) + "。")
        return " ".join(sentence for sentence in sentences if sentence).strip()

    sentences = []
    if narrative:
        sentences.append(narrative)
    elif verdict and support:
        sentences.append(f"Overall environmental-health verdict is {verdict} with {support} support.")
    elif verdict:
        sentences.append(f"Overall environmental-health verdict is {verdict}.")
    if branch_lines:
        sentences.append("Branch results: " + "; ".join(branch_lines) + ".")
    if key_pressures:
        sentences.append("Key pressures: " + "; ".join(key_pressures) + ".")
    if stabilizing_signals:
        sentences.append("Stabilizing signals: " + "; ".join(stabilizing_signals) + ".")
    return " ".join(sentence for sentence in sentences if sentence).strip()


def _augment_environment_assessment_synthesis(
    synthesis_payload: Optional[Dict[str, Any]],
    result_summaries: Dict[str, Dict[str, Any]],
    active_result_id: Optional[str],
    user_request: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(synthesis_payload, dict):
        return synthesis_payload
    env_summary = _select_environment_assessment_summary(result_summaries, active_result_id)
    if env_summary is None:
        return synthesis_payload
    synthesis_payload = dict(synthesis_payload)
    synthesis_payload["summary"] = _build_environment_assessment_analysis_summary(
        env_summary,
        user_request=user_request,
        existing_summary=str(synthesis_payload.get("summary") or "").strip() or None,
    )
    chinese = False
    synthesis_payload["scientific_findings"] = _build_environment_assessment_scientific_findings(
        env_summary,
        active_result_id=active_result_id,
        chinese=chinese,
    )
    return synthesis_payload


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


def _should_suppress_default_policy_report(user_request: str, active_plan: Optional[Dict[str, Any]] = None) -> bool:
    if not isinstance(user_request, str):
        return False
    if _EXPLICIT_POLICY_REPORT_RE.search(user_request):
        return False
    planner_policy_making_intent = _planner_policy_making_intent(active_plan)
    if planner_policy_making_intent is not None:
        return True
    return bool(_POLICY_GUIDANCE_INTENT_RE.search(user_request))


_EVIDENCE_PACKET_MAX_PACKETS = 14
_EVIDENCE_PACKET_MAX_ARRAY_POINTS = 250_000
_POLICY_CONTEXT_MAX_EVIDENCE_ROLES = 8
_POLICY_CONTEXT_MAX_HOTSPOTS = 4
_POLICY_CONTEXT_MAX_TREND_SIGNALS = 5

_POLICY_CONTEXT_INTENT_RE = re.compile(
    r"\b(policy|policies|management|recommendation|recommendations|governance|decision support|"
    r"action|actions|measure|measures|strategy|strategies|planning|zoning|monitoring|"
    r"regulation|regulatory|economic|economics|source[- ]?control|nutrient[- ]?control|"
    r"pollution[- ]?control|discharge[- ]?control)\b|"
    r"政策|管理|建议|治理|决策|措施|行动|策略|规划|分区|监测|监管|经济",
    re.IGNORECASE,
)


def _build_policy_context_packet(
    *,
    user_request: str,
    evidence_packets: List[Dict[str, Any]],
    result_summaries: Dict[str, Dict[str, Any]],
    synthesis_profile_id: Optional[str],
    policy_making_intent: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Build a compact, professional policy context for the summary LLM.

    The packet is intentionally small and schema-like. It tells the synthesizer
    which evidence may support which policy language, without turning this into
    a fixed policy report or a planner-facing skill.
    """
    if not _policy_context_requested(
        user_request=user_request,
        synthesis_profile_id=synthesis_profile_id,
        policy_making_intent=policy_making_intent,
    ):
        return None

    evidence_roles: List[Dict[str, Any]] = []
    hotspots: List[Dict[str, Any]] = []
    trend_signals: List[Dict[str, Any]] = []
    driver_responses: List[Dict[str, Any]] = []
    action_opportunities: List[Dict[str, Any]] = []
    timing_signals: List[str] = []
    evidence_anchors: List[str] = []
    evidence_texts: List[str] = []

    for packet in evidence_packets:
        if not isinstance(packet, dict):
            continue
        result_id = str(packet.get("result_id") or "").strip()
        if not result_id:
            continue
        evidence_text = _policy_packet_text(packet)
        evidence_texts.append(evidence_text)
        role, role_reason = _classify_policy_evidence_role(evidence_text)
        if len(evidence_roles) < _POLICY_CONTEXT_MAX_EVIDENCE_ROLES:
            evidence_roles.append(
                {
                    "result_id": result_id,
                    "role": role,
                    "basis": role_reason,
                    "variable": packet.get("variable"),
                    "output_type": packet.get("output_type"),
                }
            )

        driver_response = _policy_driver_response_from_evidence_packet(packet, evidence_text=evidence_text, role=role)
        if driver_response:
            _append_unique_policy_signal(driver_responses, driver_response, key="driver", limit=5)
        for opportunity in _policy_action_opportunities_from_evidence_packet(
            packet,
            evidence_text=evidence_text,
            role=role,
        ):
            _append_unique_policy_signal(action_opportunities, opportunity, key="policy_lever", limit=8)

        for hotspot in _policy_hotspots_from_evidence_packet(packet):
            if len(hotspots) >= _POLICY_CONTEXT_MAX_HOTSPOTS:
                break
            _append_policy_hotspot(hotspots, hotspot)

        trend_signal = _policy_trend_signal_from_evidence_packet(packet)
        if trend_signal and len(trend_signals) < _POLICY_CONTEXT_MAX_TREND_SIGNALS:
            trend_signals.append(trend_signal)

        timing_signal = _policy_timing_signal_from_evidence_packet(packet)
        if timing_signal and timing_signal not in timing_signals:
            timing_signals.append(timing_signal)

        for anchor in _policy_evidence_anchors_from_packet(packet):
            if anchor not in evidence_anchors:
                evidence_anchors.append(anchor)

    if not timing_signals:
        timing_signals = _infer_policy_timing_signals_from_request(user_request)

    has_source_evidence = any(_policy_text_mentions_source_pathway(text) for text in evidence_texts) or _summary_text_mentions_any(
        result_summaries,
        {
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
    has_economic_data = any(_policy_text_mentions_economic_data(text) for text in evidence_texts) or _policy_text_mentions_economic_data(
        _jsonish_compact_text(result_summaries).lower()
    )

    return {
        "version": "compact_policy_context_v3",
        "intended_use": (
            "Use only in summary synthesis. It supports evidence-bounded policy language; "
            "it is not a fixed report and not a planner tool."
        ),
        "output_budget": {
            "max_policy_synthesis_rows": 6,
            "max_evidence_ids_per_row": 3,
            "recommended_row_mix": [
                "spatial hotspot or ranked hotspot portfolio",
                "oxygen endpoint response",
                "stratification or ventilation response when present",
                "heat-stress response when present",
                "source/pathway screening or management review when supported",
                "economic/data assessment when requested or implied",
            ],
        },
        "row_detail_contract": [
            (
                "Each decision row should name the decision place, time window, monitored variable or "
                "management object, and the policy lever/action it supports."
            ),
            (
                "`trigger_evidence` should quote one concrete anchor from risk_signals.evidence_anchors, "
                "risk_signals.hotspots[*].evidence_anchor, or result_summaries: coordinates, rank, extrema, "
                "event count, trend slope/p-value, timing window, or result ID."
            ),
            (
                "`evidence_basis` should explain why that evidence supports the policy lever; "
                "`trigger_evidence` should hold the computed facts. Do not make the two fields identical."
            ),
            (
                "Do not invent policy cutoffs or trigger thresholds. Use exact computed values as evidence "
                "descriptors only unless an explicit policy threshold exists in the evidence."
            ),
        ],
        "evidence_roles": [
            {key: value for key, value in item.items() if value not in (None, "", [], {})}
            for item in evidence_roles
        ],
        "risk_signals": {
            "hotspots": _policy_hotspots_with_anchors(hotspots),
            "timing": timing_signals[:4],
            "trends_or_amplifiers": trend_signals,
            "driver_responses": driver_responses,
            "action_opportunities": action_opportunities,
            "evidence_anchors": evidence_anchors[:10],
        },
        "policy_lexicon": {
            "planning_governance": [
                "marine spatial planning",
                "ecosystem-based management",
                "adaptive management",
                "precautionary approach",
                "cumulative impact assessment",
                "environmental carrying capacity",
                "zoning / buffer zones",
            ],
            "aquaculture_marine_ranching": [
                "site suitability screening",
                "carrying-capacity control",
                "species tolerance matching",
                "stocking-density adjustment",
                "seasonal operation window",
                "real-time dissolved oxygen early warning",
            ],
            "environmental_protection": [
                "habitat restoration targeting",
                "water-quality compliance monitoring",
                "source investigation",
                "nutrient-load screening",
                "watershed-estuary linkage assessment",
            ],
            "risk_management": [
                "exposure reduction",
                "evidence-triggered early warning",
                "contingency response",
                "monitoring network design",
                "validation with in-situ observations",
            ],
        },
        "policy_design_framework": {
            "purpose": (
                "Use for broad aquaculture, marine-ranching, and coastal policy synthesis. "
                "It should turn computed environmental evidence into a system design layer, not a generic essay."
            ),
            "governing_principle": (
                "Move from single-pressure control toward risk-based marine management that coordinates "
                "climate exposure, ecosystem condition, and aquaculture or marine-ranching operations."
            ),
            "system_transition": [
                "average-state management -> extreme-event and recurrent-stress management",
                "single-variable control -> coupled temperature, oxygen, stratification, and ecological-pressure evidence",
                "static administrative zoning -> dynamic, evidence-triggered zoning and operating windows",
            ],
            "recommended_modules": [
                {
                    "module": "risk-based marine zoning",
                    "policy_logic": "Use hotspot maps, trend evidence, and endpoint oxygen risk to classify strict-protection, optimized-use, and expansion-screening areas.",
                    "model_or_data_link": "bottom oxygen, hypoxia burden or days, SST stress, stratification, and hotspot coordinates",
                    "implementation_note": "Tie zoning updates to map review and validation rather than administrative boundaries alone.",
                    "guardrail": "Do not name unsupported areas; use computed hotspots or explicitly mark map-review needs.",
                },
                {
                    "module": "land-sea nutrient and source-pathway screening",
                    "policy_logic": "Treat nutrient, river, estuary, chlorophyll, and source-pathway evidence as a watershed-coast linkage question.",
                    "model_or_data_link": "nutrient/source evidence when present; otherwise endpoint stress plus source-screening data gaps",
                    "implementation_note": "Recommend source investigation or pulse-input monitoring when direct source evidence is missing.",
                    "guardrail": "Do not attribute oxygen or bloom stress to pollution sources without source/pathway evidence.",
                },
                {
                    "module": "climate adaptation and early warning",
                    "policy_logic": "Use warming, heat-stress, stratification, and low-oxygen timing as operational risk signals.",
                    "model_or_data_link": "SST trends, heat-stress metrics, stratification, bottom oxygen, hypoxia events, and seasonal timing",
                    "implementation_note": "Connect evidence to early warning, seasonal operation windows, and contingency planning.",
                    "guardrail": "Frame future claims as historical trend based inference unless projection or scenario results are present.",
                },
                {
                    "module": "aquaculture structure transition",
                    "policy_logic": "Use environmental exposure to guide carrying-capacity review, species-tolerance matching, IMTA screening, and offshore expansion screening.",
                    "model_or_data_link": "site suitability, oxygen exposure, heat stress, stratification, and missing species or production data",
                    "implementation_note": "Prefer screening and adaptive operations when production, species tolerance, or economic data are absent.",
                    "guardrail": "Do not infer economic viability, losses, or benefits without economic and production data.",
                },
                {
                    "module": "data-driven governance and digital twin",
                    "policy_logic": "Combine numerical model output, remote sensing, in-situ validation, and LLM tool-use traces for auditable decisions.",
                    "model_or_data_link": "evidence packets, result maps, trend outputs, validation gaps, and monitoring-network needs",
                    "implementation_note": "Expose which tool result supports each decision and what data would change the recommendation.",
                    "guardrail": "Keep computed findings separate from policy-design recommendations and data gaps.",
                },
            ],
        },
        "spatial_policy_rules": [
            (
                "Policy rows should name computed hotspot labels/coordinates or the exact result map used. "
                "If multiple ranked hotspots are present, treat them as a spatial portfolio: distinguish the "
                "primary computed hotspot from other ranked hotspot candidates and avoid a single-site-only policy."
            ),
            (
                "For China coastal management questions, do not name Bohai Sea, Yellow Sea, East China Sea, "
                "Pearl River Estuary, or South China Sea unless the hotspot coordinate is inside that named "
                "region's bounding box. Hotspots marked outside_china_coastal_management_domain should be "
                "reported as data-domain or map-review warnings, not as Chinese coastal priority zones."
            ),
            (
                "Do not invent numeric policy cutoffs such as >1,000 hypoxic days. Use exact computed extrema "
                "or qualitative map-derived priority wording such as highest-burden mapped cells unless an explicit "
                "threshold is present in evidence."
            ),
            (
                "If evidence does not identify named subregions beyond the max hotspot, recommend inspecting "
                "the event-days or burden map for additional priority cells instead of naming unsupported areas."
            ),
            (
                "When evidence includes a spatial field with statistics such as max, min, mean, or hotspot coordinates, "
                "quote the exact computed values in trigger_evidence and use them to justify the spatial scope."
            ),
            (
                "If evidence covers multiple variables, each decision row should explain how its specific evidence "
                "subset leads to its specific lever, rather than listing all variables generically."
            ),
        ],
        "action_framework": [
            {
                "policy_lever": "monitoring network design",
                "use_when": "endpoint evidence identifies hotspots, recurrent events, or strong trends",
                "composable_with": ["seasonal operation window", "evidence-triggered early warning"],
                "detail_hint": "specify monitored variable, depth, hotspot label or coordinates, and validation need",
            },
            {
                "policy_lever": "seasonal operation window",
                "use_when": "events or user intent identify seasonal risk timing",
                "composable_with": ["stocking-density / carrying-capacity review", "contingency response"],
                "detail_hint": "specify months/season, endpoint variable, depth range, and operational implication",
            },
            {
                "policy_lever": "evidence-triggered early warning",
                "use_when": "endpoint evidence shows recurrent events, low oxygen, or rapid deterioration risk",
                "composable_with": ["monitoring network design", "seasonal operation window"],
                "detail_hint": "state the evidence trigger qualitatively or with exact computed values only",
            },
            {
                "policy_lever": "bottom-oxygen trend response",
                "use_when": "bottom oxygen is decreasing or hypoxia/oxygen-deficit metrics are elevated",
                "composable_with": ["monitoring network design", "evidence-triggered early warning", "site suitability re-evaluation"],
                "detail_hint": "increase near-bottom DO sampling frequency, validate with in-situ profiles, and connect worsening trends to seasonal contingency planning without estimating production losses",
            },
            {
                "policy_lever": "stratification and ventilation screening",
                "use_when": "stratification, density stability, weak mixing, or ventilation-sensitive conditions are detected",
                "composable_with": ["seasonal operation window", "site suitability re-evaluation", "monitoring network design"],
                "detail_hint": "add temperature-salinity-oxygen vertical profiles, identify poorly ventilated seasons/areas, and use stratification as vulnerability or timing evidence rather than source attribution",
            },
            {
                "policy_lever": "heat-stress adaptation screening",
                "use_when": "SST warming or heat-stress evidence is present",
                "composable_with": ["seasonal operation window", "species tolerance matching", "contingency response"],
                "detail_hint": "treat warming as exposure and timing evidence; recommend tolerance screening or operational timing review only if species/tolerance data are absent",
            },
            {
                "policy_lever": "mechanism-aware monitoring transects",
                "use_when": "circulation, transport, upwelling, fronts, eddies, or water-mass evidence is present",
                "composable_with": ["monitoring network design", "site suitability re-evaluation", "source investigation / nutrient-load screening"],
                "detail_hint": "use mechanism evidence to place transects, separate retention from advection-prone areas, and validate hotspot mechanisms before attributing causes",
            },
            {
                "policy_lever": "bloom and ecological-pressure surveillance",
                "use_when": "chlorophyll, bloom, or eutrophication evidence is present",
                "composable_with": ["water-quality compliance monitoring", "source investigation / nutrient-load screening"],
                "detail_hint": "recommend bloom surveillance and nutrient-source screening only; do not claim nutrient control is supported unless nutrient/load/source evidence exists",
            },
            {
                "policy_lever": "source-pathway management review",
                "use_when": "nutrient, organic loading, river, estuary, outfall, or discharge evidence is present",
                "composable_with": ["water-quality compliance monitoring", "watershed-estuary linkage assessment"],
                "detail_hint": "allow stronger source-management language only when direct source/pathway evidence is present; otherwise frame as investigation or screening",
            },
            {
                "policy_lever": "stocking-density / carrying-capacity review",
                "use_when": "oxygen deficit or hypoxia burden suggests elevated exposure for culture operations",
                "composable_with": ["seasonal operation window", "site suitability re-evaluation"],
                "detail_hint": "reference computed burden/days and species tolerance only if tolerance evidence exists",
            },
            {
                "policy_lever": "site suitability re-evaluation",
                "use_when": "spatial evidence shows persistent risk at existing or proposed sites",
                "composable_with": ["marine spatial planning / zoning", "environmental carrying-capacity screening"],
                "detail_hint": "name the specific map cell, hotspot, or coordinate and the risk metric",
            },
            {
                "policy_lever": "marine spatial planning / zoning",
                "use_when": "spatial maps identify risk gradients or hotspots",
                "composable_with": ["site suitability re-evaluation", "buffer zones"],
                "detail_hint": "distinguish primary computed hotspot from areas that require map review",
            },
            {
                "policy_lever": "environmental carrying-capacity screening",
                "use_when": "aquaculture expansion is requested but ecological exposure evidence is incomplete",
                "composable_with": ["economic assessment", "site suitability re-evaluation"],
                "detail_hint": "frame as screening if production, species, or tolerance data are missing",
            },
            {
                "policy_lever": "economic assessment",
                "use_when": "economic implications are requested but no economic dataset is present",
                "composable_with": ["site suitability re-evaluation", "environmental carrying-capacity screening"],
                "detail_hint": "discuss exposure/operational risk only; do not estimate losses or benefits",
            },
            {
                "policy_lever": "source investigation / nutrient-load screening",
                "use_when": "endpoint stress is present but source evidence is absent",
                "composable_with": ["water-quality compliance monitoring", "watershed-estuary linkage assessment"],
                "detail_hint": "frame source actions as investigation or screening unless source evidence exists",
            },
            {
                "policy_lever": "habitat / water-quality protection targeting",
                "use_when": "environmental protection implications are requested and endpoint risk is spatially localized",
                "composable_with": ["monitoring network design", "marine spatial planning / zoning"],
                "detail_hint": "tie protection priority to the computed hotspot or risk gradient",
            },
        ],
        "guardrails": [
            "Bottom oxygen and hypoxia are endpoint evidence for monitoring, seasonal warning, siting review, and zoning.",
            "SST and stratification are vulnerability or timing amplifiers, not source attribution.",
            "Chlorophyll is auxiliary ecological-pressure context unless direct bloom/eutrophication evidence is present.",
            (
                "Without economic data, discuss exposure, operational risk, and the need for economic assessment; "
                "do not invent stock loss, economic viability, costs, benefits, revenue, damages, or cost-benefit conclusions."
            ),
            (
                "Without nutrient, river, source-inventory, or discharge-outlet evidence, frame source-control actions "
                "as screening, investigation, or low-regret review."
            ),
        ],
        "data_availability_flags": {
            "has_source_evidence": has_source_evidence,
            "has_economic_data": has_economic_data,
        },
    }


def _policy_context_requested(
    *,
    user_request: str,
    synthesis_profile_id: Optional[str],
    policy_making_intent: Optional[bool] = None,
) -> bool:
    if policy_making_intent is not None:
        return policy_making_intent
    if not isinstance(user_request, str):
        return False
    return bool(_POLICY_CONTEXT_INTENT_RE.search(user_request))


def _build_assessment_context_packet(
    *,
    user_request: str,
    evidence_packets: List[Dict[str, Any]],
    result_summaries: Dict[str, Dict[str, Any]],
    synthesis_profile_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Build the narrative-first integrated assessment context.

    This packet groups completed evidence by environmental thread so the
    synthesizer can write a coherent answer before policy cards. It does not add
    a fixed regional portfolio or external factual claims.
    """
    if synthesis_profile_id != OCEAN_INTEGRATED_ASSESSMENT_PROFILE_ID:
        return None

    grouped: Dict[str, Dict[str, Any]] = {}
    evidence_anchors: List[str] = []
    places: List[str] = []
    for packet in evidence_packets:
        if not isinstance(packet, dict):
            continue
        result_id = str(packet.get("result_id") or "").strip()
        if not result_id:
            continue
        text = _policy_packet_text(packet)
        theme = _assessment_thread_theme(text)
        if not theme:
            continue
        entry = grouped.setdefault(
            theme,
            {
                "theme": theme,
                "status": "computed",
                "evidence_summary": "",
                "evidence_result_ids": [],
                "anchors": [],
            },
        )
        entry["evidence_result_ids"].append(result_id)
        anchors = _policy_evidence_anchors_from_packet(packet)
        for anchor in anchors[:3]:
            if anchor not in entry["anchors"]:
                entry["anchors"].append(anchor)
            if anchor not in evidence_anchors:
                evidence_anchors.append(anchor)
        for hotspot in _policy_hotspots_from_evidence_packet(packet)[:2]:
            label = str(hotspot.get("label") or "").strip()
            lon = hotspot.get("lon")
            lat = hotspot.get("lat")
            if label and lon is not None and lat is not None:
                place = f"{label} (lon={_format_policy_scalar(lon)}, lat={_format_policy_scalar(lat)})"
                if place not in places:
                    places.append(place)

    evidence_threads: List[Dict[str, Any]] = []
    for theme in (
        "bottom_oxygen/hypoxia",
        "warming/heatwave",
        "stratification",
        "chlorophyll/bloom",
    ):
        entry = grouped.get(theme)
        if not entry:
            continue
        anchors = entry.get("anchors") if isinstance(entry.get("anchors"), list) else []
        ids = [
            str(item)
            for item in entry.get("evidence_result_ids", [])
            if isinstance(item, str) and item.strip()
        ]
        evidence_threads.append(
            {
                "theme": theme,
                "status": "computed",
                "evidence_summary": _assessment_thread_summary(theme, anchors, ids),
                "evidence_result_ids": ids[:4],
            }
        )

    data_gap_summary = _assessment_data_gap_summary(
        user_request=user_request,
        evidence_threads=evidence_threads,
        result_summaries=result_summaries,
    )
    if data_gap_summary:
        evidence_threads.append(
            {
                "theme": "data_gap",
                "status": "data_gap",
                "evidence_summary": data_gap_summary,
                "evidence_result_ids": [],
            }
        )

    return {
        "version": "integrated_assessment_context_v1",
        "intended_use": (
            "Use only in result synthesis. Write a direct answer and narrative first; "
            "use policy recommendations only after the assessment narrative."
        ),
        "answer_shape": [
            "direct answer",
            "integrated narrative",
            "evidence threads",
            "optional policy recommendations",
            "uncertainty and data gaps",
        ],
        "narrative_guidance": [
            "Start with the overall suitability/risk judgment, not with tool names or policy cards.",
            "Explain how computed environmental signals make suitability more stable, fragile, uneven, or insufficiently evidenced.",
            "Use the concrete anchors below inside the narrative when available: coordinates, event days, burden values, extrema, trend slopes/p-values, seasonal timing, and result IDs.",
            "Do not merely duplicate evidence_threads; use them to write a fuller causal narrative linking stressors, places, and uncertainty.",
            "For each substantive paragraph, write data interpretation first, then the scientific implication for the user's question.",
            "Scientific implications should explain mechanisms in general terms without pretending they are new computed results.",
            "For aquaculture or marine-ranching questions, explicitly explain how each stressor affects farmed organisms or operations: low oxygen can constrain respiration, feeding, growth, survival, and bottom habitat; warming raises metabolic demand and reduces oxygen solubility; blooms can indicate ecological pressure and may add toxin, light-limitation, or decay-related oxygen stress when supported.",
            "For higher-risk region tables, prefer bay, estuary, nearshore shelf, or hotspot-coordinate labels over broad sea names; if only a broad region is supported, include the severe coordinate(s) in the evidence cell.",
            "Name places only from hotspot labels, coordinates, map extrema, or result summaries; do not use a fixed regional portfolio.",
            "Use domain reasoning only for mechanisms such as heat stress, oxygen exposure, stratification, and ecological pressure.",
            "Do not state production dominance, mortality events, coral bleaching, species impacts, or economic outcomes unless present in evidence.",
        ],
        "evidence_threads": evidence_threads[:6],
        "supported_places": places[:6],
        "evidence_anchors": evidence_anchors[:12],
        "data_gaps": _assessment_data_gaps(
            user_request=user_request,
            evidence_threads=evidence_threads,
            result_summaries=result_summaries,
        ),
    }


def _assessment_thread_theme(text: str) -> Optional[str]:
    if re.search(r"\b(hypoxia|hypoxic|oxygen|o2|oxygen_deficit|bottom oxygen)\b|缺氧|低氧", text, re.IGNORECASE):
        return "bottom_oxygen/hypoxia"
    if re.search(r"\b(heatwave|marine heatwave|sst|temperature|heat stress|thermal)\b|温度|热浪", text, re.IGNORECASE):
        return "warming/heatwave"
    if re.search(r"\b(stratification|density|stability|vertical stability)\b|层化|密度", text, re.IGNORECASE):
        return "stratification"
    if re.search(r"\b(chlorophyll|chl|bloom|algal|eutroph)\b|叶绿素|藻华|富营养", text, re.IGNORECASE):
        return "chlorophyll/bloom"
    return None


def _assessment_thread_summary(theme: str, anchors: List[str], evidence_ids: List[str]) -> str:
    if anchors:
        return f"{theme} evidence is computed from {', '.join(evidence_ids[:3])}: {'; '.join(anchors[:2])}."
    if evidence_ids:
        return f"{theme} evidence is computed from {', '.join(evidence_ids[:3])}; use the result summaries for direction and magnitude."
    return f"{theme} evidence is present, but the compact packet does not include a detailed anchor."


def _assessment_data_gap_summary(
    *,
    user_request: str,
    evidence_threads: List[Dict[str, Any]],
    result_summaries: Dict[str, Dict[str, Any]],
) -> str:
    gaps = _assessment_data_gaps(
        user_request=user_request,
        evidence_threads=evidence_threads,
        result_summaries=result_summaries,
    )
    if not gaps:
        return ""
    return f"Additional data are needed for: {', '.join(gaps[:4])}."


def _assessment_data_gaps(
    *,
    user_request: str,
    evidence_threads: List[Dict[str, Any]],
    result_summaries: Dict[str, Dict[str, Any]],
) -> List[str]:
    themes = {
        str(item.get("theme"))
        for item in evidence_threads
        if isinstance(item, dict) and item.get("status") == "computed"
    }
    gaps: List[str] = []
    request_text = str(user_request or "").lower()
    summary_text = _jsonish_compact_text(result_summaries).lower()
    if "warming/heatwave" not in themes:
        gaps.append("SST or marine heatwave evidence")
    if "bottom_oxygen/hypoxia" not in themes:
        gaps.append("bottom oxygen or hypoxia endpoint evidence")
    if "stratification" not in themes:
        gaps.append("water-column stratification evidence")
    if "chlorophyll/bloom" not in themes:
        gaps.append("chlorophyll, bloom, or eutrophication screening evidence")
    if re.search(r"\b(aquaculture|marine ranching|fish farming|suitable|economic|coming years|future)\b|养殖|海洋牧场|适合|经济|未来", request_text):
        if not _policy_text_mentions_economic_data(summary_text):
            gaps.append("production, cost, revenue, or other economic data")
        if not re.search(r"\b(species|tolerance|mortality|survival|culture species)\b|物种|耐受|死亡", summary_text):
            gaps.append("species-specific tolerance or mortality data")
        if not re.search(r"\b(projection|forecast|scenario|ssp|rcp|future_projection)\b|预测|情景", summary_text):
            gaps.append("future projection or scenario evidence")
    return _dedupe_preserving_order(gaps)[:6]


def _dedupe_preserving_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _policy_packet_text(packet: Dict[str, Any]) -> str:
    values = [
        packet.get("result_id"),
        packet.get("tool"),
        packet.get("output_type"),
        packet.get("title"),
        packet.get("variable"),
        packet.get("units"),
        packet.get("key_statistics"),
        packet.get("interpretation_hints"),
    ]
    return " ".join(_jsonish_compact_text(value) for value in values if value is not None).lower()


def _jsonish_compact_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _classify_policy_evidence_role(text: str) -> Tuple[str, str]:
    if re.search(r"\b(hypoxia|hypoxic|oxygen|o2|oxygen_deficit|bottom oxygen)\b|缺氧|低氧", text):
        return "endpoint", "bottom oxygen / hypoxia can support exposure, siting, monitoring, and seasonal risk actions"
    if re.search(r"\b(sst|temperature|heatwave|stratification|density|stability)\b|温度|热浪|层化", text):
        return "risk_amplifier", "warming, heat stress, or stratification can support vulnerability or timing language"
    if re.search(r"\b(chlorophyll|chl|bloom|eutroph)\b|叶绿素|藻华|富营养", text):
        return "auxiliary_context", "chlorophyll or bloom evidence is ecological-pressure context unless directly detected"
    if re.search(r"\b(current|transport|upwelling|front|eddy|watermass|water mass)\b|流|输运|上升流|锋|涡|水团", text):
        return "mechanism_context", "circulation or water-mass evidence can support mechanism-aware monitoring design"
    return "context", "use as supporting scientific context only"


def _policy_hotspots_from_evidence_packet(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    stats = packet.get("key_statistics")
    if not isinstance(stats, dict):
        return []
    hotspots: List[Dict[str, Any]] = []
    ranked_locations = stats.get("top_hotspots")
    if isinstance(ranked_locations, list):
        for index, location in enumerate(ranked_locations):
            raw_rank = location.get("rank") if isinstance(location, dict) else None
            numeric_rank = _safe_float(raw_rank)
            hotspot = _policy_hotspot_from_location(
                packet,
                stats=stats,
                location=location,
                rank=int(numeric_rank) if numeric_rank and numeric_rank > 0 else index + 1,
            )
            if hotspot:
                hotspots.append(hotspot)
    if hotspots:
        return hotspots

    location = stats.get("max_location")
    hotspot = _policy_hotspot_from_location(packet, stats=stats, location=location, rank=1)
    return [hotspot] if hotspot else []


def _policy_hotspot_from_location(
    packet: Dict[str, Any],
    *,
    stats: Dict[str, Any],
    location: Any,
    rank: int,
) -> Optional[Dict[str, Any]]:
    if not isinstance(location, dict):
        return None
    lon = _safe_float(location.get("lon"))
    lat = _safe_float(location.get("lat"))
    value = _safe_float(location.get("value"))
    raw_label = location.get("label") or stats.get("hotspot_region_label")
    label = _validated_ocean_region_label(raw_label, lon, lat) or _label_ocean_region(lon, lat)
    domain_status = (
        "inside_china_coastal_management_domain"
        if _is_china_coastal_management_domain(lon, lat)
        else "outside_china_coastal_management_domain"
    )
    if not label:
        label = "outside China coastal management domain" if domain_status.startswith("outside") else "mapped hotspot"
    hotspot: Dict[str, Any] = {
        "result_id": packet.get("result_id"),
        "rank": rank,
        "label": label,
        "lon": lon,
        "lat": lat,
        "value": value,
        "variable": packet.get("variable"),
        "domain_status": domain_status,
    }
    summary_mode = stats.get("summary_mode")
    if isinstance(summary_mode, str) and summary_mode.strip():
        hotspot["metric"] = summary_mode.strip()
    nonzero_fraction = _safe_float(stats.get("nonzero_fraction"))
    if nonzero_fraction is not None:
        hotspot["footprint"] = (
            "localized" if nonzero_fraction < 0.10 else "widespread" if nonzero_fraction > 0.60 else "patchy"
        )
    return {key: value for key, value in hotspot.items() if value is not None}


def _append_policy_hotspot(hotspots: List[Dict[str, Any]], hotspot: Dict[str, Any]) -> None:
    lon = _safe_float(hotspot.get("lon"))
    lat = _safe_float(hotspot.get("lat"))
    result_id = str(hotspot.get("result_id") or "").strip()
    for existing in hotspots:
        existing_lon = _safe_float(existing.get("lon"))
        existing_lat = _safe_float(existing.get("lat"))
        if lon is None or lat is None or existing_lon is None or existing_lat is None:
            continue
        if abs(lon - existing_lon) <= 0.2 and abs(lat - existing_lat) <= 0.2:
            result_ids = existing.setdefault("result_ids", [])
            if isinstance(result_ids, list) and result_id and result_id not in result_ids:
                result_ids.append(result_id)
            existing.pop("result_id", None)
            return
    if result_id:
        hotspot["result_ids"] = [result_id]
    hotspots.append(hotspot)


def _append_unique_policy_signal(
    signals: List[Dict[str, Any]],
    signal: Dict[str, Any],
    *,
    key: str,
    limit: int,
) -> None:
    value = str(signal.get(key) or "").strip().lower()
    if not value:
        return
    for existing in signals:
        if str(existing.get(key) or "").strip().lower() == value:
            result_id = str(signal.get("result_id") or "").strip()
            result_ids = existing.setdefault("result_ids", [])
            if isinstance(result_ids, list) and result_id and result_id not in result_ids:
                result_ids.append(result_id)
            existing.pop("result_id", None)
            return
    if len(signals) < limit:
        result_id = str(signal.get("result_id") or "").strip()
        if result_id:
            signal["result_ids"] = [result_id]
            signal.pop("result_id", None)
        signals.append(signal)


def _policy_hotspots_with_anchors(hotspots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    anchored: List[Dict[str, Any]] = []
    for hotspot in hotspots:
        if not isinstance(hotspot, dict):
            continue
        item = dict(hotspot)
        anchor = _format_policy_hotspot_anchor(item)
        if anchor:
            item["evidence_anchor"] = anchor
        anchored.append(item)
    return anchored


def _policy_evidence_anchors_from_packet(packet: Dict[str, Any]) -> List[str]:
    result_id = str(packet.get("result_id") or "").strip()
    if not result_id:
        return []
    stats = packet.get("key_statistics")
    if not isinstance(stats, dict):
        stats = {}

    anchors: List[str] = []
    for hotspot in _policy_hotspots_from_evidence_packet(packet)[:3]:
        anchor = _format_policy_hotspot_anchor(hotspot)
        if anchor:
            anchors.append(anchor)

    for key in ("event_count", "total_count"):
        if key in stats:
            anchors.append(f"{result_id}: {key}={_format_policy_scalar(stats.get(key))}")
            break

    trend = _policy_trend_signal_from_evidence_packet(packet)
    if trend:
        parts = [f"{result_id}: trend"]
        direction = trend.get("direction")
        if direction:
            parts.append(f"direction={direction}")
        if trend.get("slope") is not None:
            parts.append(f"slope={_format_policy_scalar(trend.get('slope'))}")
        if trend.get("p_value") is not None:
            parts.append(f"p={_format_policy_scalar(trend.get('p_value'))}")
        anchors.append(", ".join(parts))

    for key in ("max", "min", "mean", "total", "nonzero_fraction"):
        if key in stats:
            anchors.append(f"{result_id}: {key}={_format_policy_scalar(stats.get(key))}")

    ranges = packet.get("ranges")
    if isinstance(ranges, dict):
        time_range = ranges.get("time_range")
        depth_range = ranges.get("depth_range")
        if time_range is not None:
            anchors.append(f"{result_id}: time_range={_format_policy_scalar(time_range)}")
        if depth_range is not None:
            anchors.append(f"{result_id}: depth_range={_format_policy_scalar(depth_range)}")

    unique: List[str] = []
    for anchor in anchors:
        if anchor and anchor not in unique:
            unique.append(anchor)
    return unique[:5]


def _format_policy_hotspot_anchor(hotspot: Dict[str, Any]) -> Optional[str]:
    label = str(hotspot.get("label") or "mapped hotspot").strip()
    rank = hotspot.get("rank")
    lon = _safe_float(hotspot.get("lon"))
    lat = _safe_float(hotspot.get("lat"))
    value = hotspot.get("value")
    metric = str(hotspot.get("metric") or hotspot.get("variable") or "mapped_value").strip()
    result_ids = hotspot.get("result_ids")
    if not isinstance(result_ids, list) or not result_ids:
        single_result_id = hotspot.get("result_id")
        result_ids = [single_result_id] if single_result_id else []

    parts: List[str] = []
    if rank is not None:
        parts.append(f"rank {_format_policy_scalar(rank)}")
    parts.append(label)
    if lon is not None and lat is not None:
        parts.append(f"lon={_format_policy_scalar(lon)}, lat={_format_policy_scalar(lat)}")
    if value is not None:
        parts.append(f"{metric}={_format_policy_scalar(value)}")
    domain_status = str(hotspot.get("domain_status") or "").strip()
    if domain_status:
        parts.append(f"domain_status={domain_status}")
    clean_result_ids = [str(item).strip() for item in result_ids if str(item).strip()]
    if clean_result_ids:
        parts.append("result_ids=" + ",".join(clean_result_ids[:3]))
    return "; ".join(part for part in parts if part)


def _format_policy_scalar(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_policy_scalar(item) for item in value[:4]) + "]"
    numeric_value = _safe_float(value)
    if numeric_value is not None:
        return f"{numeric_value:.8g}"
    if isinstance(value, str):
        return value
    return str(value)


def _policy_driver_response_from_evidence_packet(
    packet: Dict[str, Any],
    *,
    evidence_text: str,
    role: str,
) -> Optional[Dict[str, Any]]:
    trend = _policy_trend_signal_from_evidence_packet(packet)
    direction = str((trend or {}).get("direction") or "").strip().lower()
    base: Dict[str, Any] = {
        "result_id": packet.get("result_id"),
        "evidence_role": role,
        "variable": packet.get("variable"),
    }
    if re.search(r"\b(hypoxia|hypoxic|oxygen|o2|oxygen_deficit|bottom oxygen)\b|缺氧|低氧", evidence_text):
        response = dict(base)
        response.update(
            {
                "driver": "bottom oxygen decline or hypoxia endpoint risk",
                "policy_response": (
                    "Expand near-bottom dissolved-oxygen monitoring, add evidence-triggered early warning, "
                    "and use declining or low-oxygen areas for seasonal contingency and site-suitability review."
                ),
                "guardrail": "Endpoint oxygen evidence supports exposure management, monitoring, and siting review, not economic-loss estimates without production data.",
            }
        )
        if direction:
            response["direction"] = direction
        return response
    if re.search(r"\b(stratification|density|stability|mixed layer|pycnocline|vertical mixing|ventilation)\b|层化|密度|混合|通风", evidence_text):
        response = dict(base)
        response.update(
            {
                "driver": "stratification or weak vertical ventilation",
                "policy_response": (
                    "Add temperature-salinity-oxygen vertical profiles, map poorly ventilated seasons and areas, "
                    "and use stratification evidence to adjust seasonal operations or require ventilation screening before siting expansion."
                ),
                "guardrail": "Stratification is vulnerability and timing evidence; it must not be treated as direct pollutant-source attribution.",
            }
        )
        if direction:
            response["direction"] = direction
        return response
    if re.search(r"\b(sst|temperature|heatwave|heat stress|warming)\b|温度|升温|热浪|热压力", evidence_text):
        response = dict(base)
        response.update(
            {
                "driver": "warming or heat-stress exposure",
                "policy_response": (
                    "Use warming evidence for seasonal timing review, species-tolerance screening, and contingency planning, "
                    "paired with oxygen monitoring where heat can amplify low-oxygen exposure."
                ),
                "guardrail": "Thermal evidence supports exposure adaptation and timing, not source attribution or economic quantification by itself.",
            }
        )
        if direction:
            response["direction"] = direction
        return response
    if re.search(r"\b(current|transport|upwelling|front|eddy|watermass|water mass|circulation)\b|流|输运|上升流|锋|涡|水团", evidence_text):
        response = dict(base)
        response.update(
            {
                "driver": "circulation or water-mass mechanism context",
                "policy_response": (
                    "Use circulation evidence to place monitoring transects, distinguish retention from advection-prone areas, "
                    "and target follow-up mechanism checks around event hotspots."
                ),
                "guardrail": "Mechanism context should guide monitoring design and hypothesis testing, not replace endpoint risk evidence.",
            }
        )
        if direction:
            response["direction"] = direction
        return response
    return None


def _policy_action_opportunities_from_evidence_packet(
    packet: Dict[str, Any],
    *,
    evidence_text: str,
    role: str,
) -> List[Dict[str, Any]]:
    base: Dict[str, Any] = {
        "result_id": packet.get("result_id"),
        "evidence_role": role,
        "variable": packet.get("variable"),
    }
    opportunities: List[Dict[str, Any]] = []

    def add(
        *,
        policy_lever: str,
        target: str,
        action_type: str,
        action: str,
        guardrail: str,
        evidence_strength: str = "limited",
    ) -> None:
        item = dict(base)
        item.update(
            {
                "policy_lever": policy_lever,
                "target": target,
                "action_type": action_type,
                "recommended_action": action,
                "guardrail": guardrail,
                "evidence_strength": evidence_strength,
            }
        )
        opportunities.append({key: value for key, value in item.items() if value not in (None, "", [], {})})

    if re.search(r"\b(hypoxia|hypoxic|oxygen|o2|oxygen_deficit|bottom oxygen)\b|缺氧|低氧", evidence_text):
        add(
            policy_lever="bottom-oxygen trend response",
            target="bottom oxygen risk areas and hypoxia-prone seasons",
            action_type="monitoring",
            action=(
                "Increase near-bottom dissolved oxygen monitoring, add real-time early warning where feasible, "
                "and use low-oxygen trends for seasonal contingency, site-suitability review, and carrying-capacity screening."
            ),
            guardrail="Use oxygen endpoints for exposure management and siting review, not economic-loss estimates without production data.",
            evidence_strength="supported",
        )
    if re.search(r"\b(stratification|density|stability|mixed layer|pycnocline|vertical mixing|ventilation)\b|层化|密度|混合|通风", evidence_text):
        add(
            policy_lever="stratification and ventilation screening",
            target="stratified water-column windows and poorly ventilated areas",
            action_type="seasonal_management",
            action=(
                "Add repeated temperature-salinity-oxygen vertical profiles, map poorly ventilated windows, "
                "and require ventilation screening before expansion in stratified seasons."
            ),
            guardrail="Treat stratification as vulnerability and timing evidence, not direct pollutant-source attribution.",
        )
    if re.search(r"\b(sst|temperature|heatwave|heat stress|warming)\b|温度|升温|热浪|热压力", evidence_text):
        add(
            policy_lever="heat-stress adaptation screening",
            target="warm-season exposure windows",
            action_type="seasonal_management",
            action=(
                "Use warming evidence for heat-risk advisories, seasonal operation timing review, "
                "and species-tolerance screening when species/tolerance data are unavailable."
            ),
            guardrail="Thermal evidence supports exposure adaptation and timing, not source attribution or economic quantification by itself.",
        )
    if re.search(r"\b(current|transport|upwelling|front|eddy|watermass|water mass|circulation)\b|流|输运|上升流|锋|涡|水团", evidence_text):
        add(
            policy_lever="mechanism-aware monitoring transects",
            target="circulation-linked risk gradients and hotspot mechanisms",
            action_type="coastal_planning",
            action=(
                "Use circulation or water-mass evidence to place adaptive monitoring transects, "
                "distinguish retention from advection-prone areas, and validate hotspot mechanisms."
            ),
            guardrail="Mechanism context should guide sampling and hypothesis testing, not replace endpoint evidence.",
        )
    if re.search(r"\b(chlorophyll|chl|bloom|eutroph)\b|叶绿素|藻华|富营养", evidence_text):
        add(
            policy_lever="bloom and ecological-pressure surveillance",
            target="chlorophyll, bloom, or eutrophication screening areas",
            action_type="governance",
            action=(
                "Track chlorophyll exposure and bloom indicators, and use them to prioritize nutrient-source screening "
                "only where source/load evidence is collected."
            ),
            guardrail="Chlorophyll is ecological-pressure context unless direct bloom/eutrophication and source evidence support stronger action.",
        )
    if _policy_text_mentions_source_pathway(evidence_text):
        add(
            policy_lever="source-pathway management review",
            target="nutrient, river-estuary, outfall, or discharge pathways",
            action_type="source_control",
            action=(
                "Use direct pathway evidence to review source-management measures, compliance monitoring, "
                "and watershed-estuary linkage actions."
            ),
            guardrail="Without direct source/pathway evidence, keep source actions framed as screening or investigation.",
            evidence_strength="supported",
        )
    if _policy_text_mentions_economic_context(evidence_text):
        add(
            policy_lever="economic and production data assessment",
            target="economic-development evidence gaps",
            action_type="economic_assessment",
            action=(
                "Collect production, cost, revenue, and exposure data before quantifying financial implications; "
                "use current ocean evidence only for exposure screening."
            ),
            guardrail="Do not infer stock loss, revenue, viability, damages, or cost-benefit outcomes from ocean variables alone.",
            evidence_strength="screening",
        )
    return opportunities


def _policy_text_mentions_source_pathway(text: str) -> bool:
    return bool(
        re.search(
            r"\b(nutrient_load|nutrient loading|organic loading|river discharge|river input|"
            r"estuary input|outfall|discharge outlet|source inventory|wastewater|point source|"
            r"watershed|catchment)\b|营养盐|有机负荷|河流|河口|排口|污水|流域",
            text,
            re.IGNORECASE,
        )
    )


def _policy_text_mentions_economic_context(text: str) -> bool:
    return bool(
        re.search(
            r"\b(economic|economics|production|cost|benefit|revenue|loss|damage|valuation|"
            r"cost-benefit|viability)\b|经济|产量|成本|收益|收入|损失|估值",
            text,
            re.IGNORECASE,
        )
    )


def _policy_text_mentions_economic_data(text: str) -> bool:
    if not isinstance(text, str):
        return False
    if re.search(
        r"\b(?:no|without|missing|lack(?:ing)?|absent|insufficient|unavailable|"
        r"not supplied|not provided|not available|did not include|does not include|not include)\b.{0,140}"
        r"\b(?:economic|production|cost|benefit|revenue|loss|damage|valuation)\w*\b.{0,80}"
        r"\b(?:data|dataset|evidence|valuation|assessment)\b|"
        r"\b(?:economic|production|cost|benefit|revenue|loss|damage|valuation)\w*\b.{0,80}"
        r"\b(?:data|dataset|evidence|valuation|assessment)\b.{0,140}"
        r"\b(?:no|without|missing|lack(?:ing)?|absent|insufficient|unavailable|"
        r"not supplied|not provided|not available|did not include|does not include|not include|needed|required)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    return bool(
        re.search(
            r"\b(economic_data|economic dataset|production dataset|cost dataset|benefit dataset|"
            r"revenue dataset|valuation dataset|economic valuation|cost-benefit data)\b",
            text,
            re.IGNORECASE,
        )
    )


def _policy_trend_signal_from_evidence_packet(packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    stats = packet.get("key_statistics")
    if not isinstance(stats, dict):
        return None
    slope = _safe_float(stats.get("slope"))
    direction = stats.get("trend_direction")
    if slope is None and not isinstance(direction, str):
        return None
    signal: Dict[str, Any] = {
        "result_id": packet.get("result_id"),
        "variable": packet.get("variable"),
        "direction": direction or ("increasing" if slope and slope > 0 else "decreasing" if slope and slope < 0 else "flat"),
        "slope": slope,
        "p_value": _safe_float(stats.get("p_value")),
    }
    return {key: value for key, value in signal.items() if value is not None}


def _policy_timing_signal_from_evidence_packet(packet: Dict[str, Any]) -> Optional[str]:
    ranges = packet.get("ranges")
    title_text = _policy_packet_text(packet)
    if "summer" in title_text:
        return "summer risk window"
    if isinstance(ranges, dict) and ranges.get("time_range"):
        return f"evidence time window {ranges.get('time_range')}"
    return None


def _infer_policy_timing_signals_from_request(user_request: str) -> List[str]:
    if not isinstance(user_request, str):
        return []
    lowered = user_request.lower()
    signals: List[str] = []
    if re.search(r"\bsummer\b|夏季|夏天", lowered):
        signals.append("summer risk window")
    if re.search(r"\bwinter\b|冬季|冬天", lowered):
        signals.append("winter risk window")
    if re.search(r"\bseasonal\b|季节", lowered):
        signals.append("seasonal management window")
    return signals


def _summary_text_mentions_any(result_summaries: Dict[str, Dict[str, Any]], terms: Set[str]) -> bool:
    text = _jsonish_compact_text(result_summaries).lower()
    return any(term.lower() in text for term in terms)


def _build_synthesis_evidence_packets(
    *,
    executor: Any,
    completed_steps: List[Dict[str, Any]],
    result_summaries: Dict[str, Dict[str, Any]],
    active_result_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Build compact, science-oriented evidence for the summary LLM.

    These packets intentionally avoid full arrays. They give the synthesizer
    locations, extrema, counts, trends, and interpretation hints that are easy
    to lose in generic result summaries.
    """
    step_by_result: Dict[str, Dict[str, Any]] = {}
    ordered_result_ids: List[str] = []

    def add_result_id(result_id: Any) -> None:
        normalized = str(result_id or "").strip()
        if normalized and normalized not in ordered_result_ids:
            ordered_result_ids.append(normalized)

    add_result_id(active_result_id)
    for step in completed_steps:
        if not isinstance(step, dict):
            continue
        result_id = str(step.get("result_id") or "").strip()
        if result_id:
            step_by_result[result_id] = step
            add_result_id(result_id)
    for result_id in result_summaries:
        add_result_id(result_id)

    packets: List[Dict[str, Any]] = []
    for result_id in ordered_result_ids:
        summary = result_summaries.get(result_id)
        if not isinstance(summary, dict):
            continue
        result = None
        try:
            result = executor.get_result(result_id)
        except Exception:
            result = None
        packet = _build_synthesis_evidence_packet(
            result_id=result_id,
            summary=summary,
            result=result,
            step=step_by_result.get(result_id, {}),
        )
        if packet:
            packets.append(packet)
        if len(packets) >= _EVIDENCE_PACKET_MAX_PACKETS:
            break
    return packets


def _build_synthesis_evidence_packet(
    *,
    result_id: str,
    summary: Dict[str, Any],
    result: Any,
    step: Dict[str, Any],
) -> Dict[str, Any]:
    output_type = str(
        summary.get("type")
        or (result.get("output_type") if isinstance(result, dict) else "")
        or ""
    ).strip()
    metadata = result.get("metadata") if isinstance(result, dict) and isinstance(result.get("metadata"), dict) else {}
    ranges = _packet_ranges(summary, metadata, result)
    statistics = _packet_key_statistics(summary, metadata)
    hints: List[str] = []

    packet: Dict[str, Any] = {
        "result_id": result_id,
        "tool": step.get("tool"),
        "output_type": output_type or None,
        "title": summary.get("title") or metadata.get("title"),
        "variable": summary.get("variable") or metadata.get("variable") or metadata.get("source_variable"),
        "units": summary.get("units") or metadata.get("units"),
        "ranges": ranges,
        "key_statistics": statistics,
        "interpretation_hints": hints,
    }

    if output_type in {"spatial_field_result", "field_trend_result", "regression_map_result", "composite_result"}:
        _add_spatial_evidence(packet, summary=summary, result=result, metadata=metadata)
    elif output_type == "event_detection_result":
        _add_event_detection_evidence(packet, summary=summary, result=result)
    elif output_type in {"timeseries_result", "climatology_result"}:
        _add_timeseries_evidence(packet, summary=summary, result=result)
    elif output_type == "trend_result":
        _add_trend_evidence(packet, summary=summary, result=result)
    elif output_type == "hovmoller_result":
        _add_hovmoller_evidence(packet, summary=summary, result=result)
    elif output_type == "section_result":
        _add_section_evidence(packet, summary=summary, result=result)
    elif output_type == "eof_result":
        _add_eof_evidence(packet, summary=summary, result=result)
    else:
        _add_generic_summary_hints(packet, summary)

    packet["key_statistics"] = {
        key: value
        for key, value in packet.get("key_statistics", {}).items()
        if value is not None
    }
    packet["ranges"] = {
        key: value
        for key, value in packet.get("ranges", {}).items()
        if value is not None
    }
    packet["interpretation_hints"] = [
        str(item)
        for item in packet.get("interpretation_hints", [])
        if isinstance(item, str) and item.strip()
    ][:8]
    return {
        key: value
        for key, value in packet.items()
        if value not in (None, {}, [])
    }


def _packet_ranges(summary: Dict[str, Any], metadata: Dict[str, Any], result: Any) -> Dict[str, Any]:
    ranges: Dict[str, Any] = {}
    for key in ("time_range", "depth_range", "lon_range", "lat_range", "spatial_range", "distance_range_km"):
        value = summary.get(key) if key in summary else metadata.get(key)
        if value is not None:
            ranges[key] = _compact_evidence_value(value)
    coord_ranges = summary.get("coord_ranges") or metadata.get("coord_ranges")
    if isinstance(coord_ranges, dict):
        for key in ("time_range", "depth_range", "lon_range", "lat_range"):
            if key in coord_ranges and key not in ranges:
                ranges[key] = _compact_evidence_value(coord_ranges[key])
    if isinstance(result, dict):
        for key, coord_key in (("lon", "lon_range"), ("lat", "lat_range"), ("depth", "depth_range")):
            if coord_key in ranges:
                continue
            coord = _bounded_numeric_array(result.get(key), max_points=10_000)
            if coord.size:
                ranges[coord_key] = [float(np.nanmin(coord)), float(np.nanmax(coord))]
        times = result.get("times")
        if times is None:
            times = result.get("time")
        if "time_range" not in ranges and isinstance(times, list) and times:
            ranges["time_range"] = [times[0], times[-1]]
    return ranges


def _packet_key_statistics(summary: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for source in (metadata.get("statistics"), summary.get("statistics"), summary):
        if isinstance(source, dict):
            stats.update(_compact_scalar_dict(source))
    return stats


def _add_spatial_evidence(
    packet: Dict[str, Any],
    *,
    summary: Dict[str, Any],
    result: Any,
    metadata: Dict[str, Any],
) -> None:
    stats = packet.setdefault("key_statistics", {})
    hints = packet.setdefault("interpretation_hints", [])

    for key in (
        "total",
        "n_nonzero",
        "nonzero_fraction",
        "max_location",
        "min_location",
        "top_hotspots",
        "hotspot_region_label",
        "coldspot_region_label",
    ):
        value = metadata.get(key)
        if value is None:
            value = summary.get(key)
        if value is not None:
            stats[key] = _compact_evidence_value(value)

    extrema = summary.get("extrema") if isinstance(summary.get("extrema"), dict) else {}
    if extrema:
        stats.update(_compact_scalar_dict(extrema))

    if not isinstance(result, dict):
        _add_spatial_hints_from_stats(stats, hints)
        return

    values = _bounded_numeric_array(result.get("values"), max_points=_EVIDENCE_PACKET_MAX_ARRAY_POINTS)
    lon = _bounded_numeric_array(result.get("lon"), max_points=50_000)
    lat = _bounded_numeric_array(result.get("lat"), max_points=50_000)
    if values.size:
        computed = _spatial_value_statistics(values, lat=lat, lon=lon)
        stats.update({key: value for key, value in computed.items() if value is not None})
    _add_spatial_hints_from_stats(stats, hints)


def _add_spatial_hints_from_stats(stats: Dict[str, Any], hints: List[str]) -> None:
    max_location = stats.get("max_location")
    if isinstance(max_location, dict):
        label = stats.get("hotspot_region_label") or _label_ocean_region(
            _safe_float(max_location.get("lon")),
            _safe_float(max_location.get("lat")),
        )
        suffix = f" near {label}" if label else ""
        hints.append(
            f"Highest mapped value at lon={max_location.get('lon')}, lat={max_location.get('lat')}{suffix}."
        )
    top_hotspots = stats.get("top_hotspots")
    if isinstance(top_hotspots, list) and len(top_hotspots) > 1:
        labels: List[str] = []
        for hotspot in top_hotspots[1:4]:
            if not isinstance(hotspot, dict):
                continue
            label = hotspot.get("label") or _label_ocean_region(
                _safe_float(hotspot.get("lon")),
                _safe_float(hotspot.get("lat")),
            )
            if label:
                labels.append(str(label))
        if labels:
            hints.append("Additional ranked hotspot candidates include " + ", ".join(labels) + ".")
    min_location = stats.get("min_location")
    if isinstance(min_location, dict):
        label = stats.get("coldspot_region_label") or _label_ocean_region(
            _safe_float(min_location.get("lon")),
            _safe_float(min_location.get("lat")),
        )
        suffix = f" near {label}" if label else ""
        hints.append(
            f"Lowest mapped value at lon={min_location.get('lon')}, lat={min_location.get('lat')}{suffix}."
        )
    nonzero_fraction = _safe_float(stats.get("nonzero_fraction"))
    if nonzero_fraction is not None:
        if nonzero_fraction <= 0:
            hints.append("The summary map is spatially empty or all zero.")
        elif nonzero_fraction < 0.10:
            hints.append("Nonzero values are concentrated in a small hotspot footprint.")
        elif nonzero_fraction > 0.60:
            hints.append("Nonzero values are widespread across the mapped domain.")


def _add_event_detection_evidence(packet: Dict[str, Any], *, summary: Dict[str, Any], result: Any) -> None:
    stats = packet.setdefault("key_statistics", {})
    hints = packet.setdefault("interpretation_hints", [])
    event_count = summary.get("event_count") or summary.get("total_count")
    if event_count is None and isinstance(result, dict):
        event_count = result.get("event_count") or len(result.get("events", []) or [])
    if event_count is not None:
        stats["event_count"] = event_count

    spatial = summary.get("spatial_distribution")
    if spatial is None and isinstance(result, dict):
        spatial = result.get("spatial_distribution")
    if isinstance(spatial, dict):
        stats["spatial_distribution"] = _compact_evidence_value(spatial)
        centroid = spatial.get("centroid")
        if isinstance(centroid, dict):
            label = _label_ocean_region(_safe_float(centroid.get("lon")), _safe_float(centroid.get("lat")))
            suffix = f" near {label}" if label else ""
            hints.append(
                f"Detected-event centroid at lon={centroid.get('lon')}, lat={centroid.get('lat')}{suffix}."
            )

    events = result.get("events") if isinstance(result, dict) else None
    if isinstance(events, list) and events:
        strongest = _select_event_record(events, keys=("max_intensity", "mean_intensity", "intensity", "severity"))
        largest = _select_event_record(events, keys=("area_km2", "duration_days"))
        if strongest:
            stats["strongest_event"] = strongest
        if largest:
            stats["largest_or_longest_event"] = largest


def _add_timeseries_evidence(packet: Dict[str, Any], *, summary: Dict[str, Any], result: Any) -> None:
    stats = packet.setdefault("key_statistics", {})
    hints = packet.setdefault("interpretation_hints", [])
    for key in (
        "n_points",
        "finite_count",
        "start_value",
        "end_value",
        "absolute_change",
        "relative_change_pct",
        "peak_phase",
        "trough_phase",
    ):
        if key in summary:
            stats[key] = _compact_evidence_value(summary[key])
    extrema = summary.get("extrema")
    if isinstance(extrema, dict):
        stats["extrema"] = _compact_evidence_value(extrema)
        if extrema.get("max_time") is not None:
            hints.append(f"Peak occurs near {extrema.get('max_time')} with value {extrema.get('max_value')}.")
        if extrema.get("min_time") is not None:
            hints.append(f"Minimum occurs near {extrema.get('min_time')} with value {extrema.get('min_value')}.")
    if not isinstance(result, dict):
        return
    values = _bounded_numeric_array(result.get("values"), max_points=_EVIDENCE_PACKET_MAX_ARRAY_POINTS)
    times = result.get("times")
    if times is None:
        times = result.get("time")
    if values.size and isinstance(times, list):
        computed = _timeseries_value_statistics(values, times)
        stats.update(computed)


def _add_trend_evidence(packet: Dict[str, Any], *, summary: Dict[str, Any], result: Any) -> None:
    stats = packet.setdefault("key_statistics", {})
    hints = packet.setdefault("interpretation_hints", [])
    source = result if isinstance(result, dict) else summary
    for key in (
        "slope",
        "intercept",
        "r_squared",
        "p_value",
        "std_err",
        "is_significant",
        "confidence_level",
        "n_points",
        "n_valid_points",
        "trend_direction",
        "fitted_change_over_period",
    ):
        value = summary.get(key) if key in summary else source.get(key)
        if value is not None:
            stats[key] = _compact_evidence_value(value)
    slope = _safe_float(stats.get("slope"))
    p_value = _safe_float(stats.get("p_value"))
    if slope is not None:
        direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        significance = " statistically significant" if p_value is not None and p_value < 0.05 else ""
        hints.append(f"Trend is {direction}{significance}: slope={slope}.")


def _add_hovmoller_evidence(packet: Dict[str, Any], *, summary: Dict[str, Any], result: Any) -> None:
    stats = packet.setdefault("key_statistics", {})
    hints = packet.setdefault("interpretation_hints", [])
    for key in ("diagram_type", "spatial_dim", "value_shape", "extrema", "time_mean_extrema", "spatial_mean_extrema"):
        if key in summary:
            stats[key] = _compact_evidence_value(summary[key])
    if not isinstance(result, dict):
        return
    values = _bounded_numeric_array(result.get("values"), max_points=_EVIDENCE_PACKET_MAX_ARRAY_POINTS)
    if values.ndim != 2 or not values.size:
        return
    times = result.get("time")
    if times is None:
        times = result.get("times")
    raw_spatial_coord = result.get("spatial_coord")
    if raw_spatial_coord is None:
        raw_spatial_coord = result.get("depth")
    spatial_coord = _bounded_numeric_array(raw_spatial_coord, max_points=100_000)
    extrema = _matrix_extrema(values, row_labels=times if isinstance(times, list) else None, col_values=spatial_coord)
    if extrema:
        stats["matrix_extrema"] = extrema
        max_item = extrema.get("max")
        min_item = extrema.get("min")
        if isinstance(max_item, dict):
            hints.append(
                f"Hovmoller maximum at time={max_item.get('row_label')} and coord={max_item.get('col_value')}."
            )
        if isinstance(min_item, dict):
            hints.append(
                f"Hovmoller minimum at time={min_item.get('row_label')} and coord={min_item.get('col_value')}."
            )


def _add_section_evidence(packet: Dict[str, Any], *, summary: Dict[str, Any], result: Any) -> None:
    stats = packet.setdefault("key_statistics", {})
    hints = packet.setdefault("interpretation_hints", [])
    for key in ("value_shape", "distance_range_km", "depth_range", "transect_endpoints"):
        if key in summary:
            stats[key] = _compact_evidence_value(summary[key])
    if not isinstance(result, dict):
        return
    values = _bounded_numeric_array(result.get("values"), max_points=_EVIDENCE_PACKET_MAX_ARRAY_POINTS)
    distance = _bounded_numeric_array(result.get("distance_km"), max_points=100_000)
    depth = _bounded_numeric_array(result.get("depth"), max_points=100_000)
    if values.ndim == 2 and values.size:
        extrema = _matrix_extrema(values, row_labels=None, row_values=depth, col_values=distance)
        if extrema:
            stats["section_extrema"] = extrema
            max_item = extrema.get("max")
            if isinstance(max_item, dict):
                hints.append(
                    f"Section maximum near distance={max_item.get('col_value')} km and depth={max_item.get('row_value')} m."
                )


def _add_eof_evidence(packet: Dict[str, Any], *, summary: Dict[str, Any], result: Any) -> None:
    stats = packet.setdefault("key_statistics", {})
    hints = packet.setdefault("interpretation_hints", [])
    for key in ("n_modes", "cumulative_variance", "total_variance", "leading_modes"):
        if key in summary:
            stats[key] = _compact_evidence_value(summary[key])
    leading_modes = summary.get("leading_modes")
    if isinstance(leading_modes, list) and leading_modes:
        first = leading_modes[0]
        if isinstance(first, dict):
            hints.append(
                f"Leading EOF mode explains {first.get('variance_explained')} of variance."
            )


def _add_generic_summary_hints(packet: Dict[str, Any], summary: Dict[str, Any]) -> None:
    hints = packet.setdefault("interpretation_hints", [])
    for key in ("summary", "narrative_summary", "description"):
        value = summary.get(key)
        if isinstance(value, str) and value.strip():
            hints.append(value.strip()[:300])
            break


def _spatial_value_statistics(values: np.ndarray, *, lat: np.ndarray, lon: np.ndarray) -> Dict[str, Any]:
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    if finite.size == 0:
        return {"count": 0}
    n_nonzero = int(np.count_nonzero(np.abs(finite) > 0.0))
    stats: Dict[str, Any] = {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
        "total": float(np.nansum(finite)),
        "n_nonzero": n_nonzero,
        "nonzero_fraction": float(n_nonzero / finite.size),
    }
    if n_nonzero > 0:
        extrema = _grid_extrema(values, lat=lat, lon=lon)
        stats.update(extrema)
    return stats


def _grid_extrema(values: np.ndarray, *, lat: np.ndarray, lon: np.ndarray) -> Dict[str, Any]:
    if values.ndim != 2 or not np.any(np.isfinite(values)):
        return {}
    if lat.ndim != 1 or lon.ndim != 1 or values.shape != (lat.size, lon.size):
        return {}
    max_index = tuple(int(idx) for idx in np.unravel_index(int(np.nanargmax(values)), values.shape))
    min_index = tuple(int(idx) for idx in np.unravel_index(int(np.nanargmin(values)), values.shape))
    max_location = {
        "lon": float(lon[max_index[1]]),
        "lat": float(lat[max_index[0]]),
        "value": float(values[max_index]),
    }
    min_location = {
        "lon": float(lon[min_index[1]]),
        "lat": float(lat[min_index[0]]),
        "value": float(values[min_index]),
    }
    result: Dict[str, Any] = {
        "max_location": max_location,
        "min_location": min_location,
    }
    top_hotspots = _top_spatial_hotspots(values, lat=lat, lon=lon, limit=4)
    if top_hotspots:
        result["top_hotspots"] = top_hotspots
    max_label = _label_ocean_region(max_location["lon"], max_location["lat"])
    min_label = _label_ocean_region(min_location["lon"], min_location["lat"])
    if max_label:
        result["hotspot_region_label"] = max_label
    if min_label:
        result["coldspot_region_label"] = min_label
    return result


def _top_spatial_hotspots(
    values: np.ndarray,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    limit: int,
) -> List[Dict[str, Any]]:
    if values.ndim != 2 or lat.ndim != 1 or lon.ndim != 1 or values.shape != (lat.size, lon.size):
        return []
    finite_mask = np.isfinite(values) & (np.abs(values) > 0.0)
    if not np.any(finite_mask):
        return []
    candidate_indices = np.argwhere(finite_mask)
    candidate_values = values[finite_mask]
    order = np.argsort(candidate_values)[::-1]
    min_separation_deg = 1.0
    hotspots: List[Dict[str, Any]] = []
    for ordered_index in order:
        lat_idx, lon_idx = (int(idx) for idx in candidate_indices[int(ordered_index)])
        hotspot_lon = float(lon[lon_idx])
        hotspot_lat = float(lat[lat_idx])
        if any(
            math.hypot(hotspot_lon - float(existing["lon"]), hotspot_lat - float(existing["lat"]))
            < min_separation_deg
            for existing in hotspots
        ):
            continue
        label = _label_ocean_region(hotspot_lon, hotspot_lat)
        hotspot: Dict[str, Any] = {
            "rank": len(hotspots) + 1,
            "lon": hotspot_lon,
            "lat": hotspot_lat,
            "value": float(values[lat_idx, lon_idx]),
            "domain_status": (
                "inside_china_coastal_management_domain"
                if _is_china_coastal_management_domain(hotspot_lon, hotspot_lat)
                else "outside_china_coastal_management_domain"
            ),
        }
        if label:
            hotspot["label"] = label
        hotspots.append(hotspot)
        if len(hotspots) >= limit:
            break
    return hotspots


def _timeseries_value_statistics(values: np.ndarray, times: List[Any]) -> Dict[str, Any]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite_mask = np.isfinite(values)
    if values.size == 0 or not np.any(finite_mask):
        return {}
    max_index = int(np.nanargmax(values))
    min_index = int(np.nanargmin(values))
    stats: Dict[str, Any] = {
        "max_value": float(values[max_index]),
        "max_time": times[max_index] if max_index < len(times) else None,
        "min_value": float(values[min_index]),
        "min_time": times[min_index] if min_index < len(times) else None,
    }
    finite_indices = np.where(finite_mask)[0]
    first_idx = int(finite_indices[0])
    last_idx = int(finite_indices[-1])
    stats["first_finite"] = {"time": times[first_idx] if first_idx < len(times) else None, "value": float(values[first_idx])}
    stats["last_finite"] = {"time": times[last_idx] if last_idx < len(times) else None, "value": float(values[last_idx])}
    stats["finite_change"] = float(values[last_idx] - values[first_idx])
    return {key: value for key, value in stats.items() if value is not None}


def _matrix_extrema(
    values: np.ndarray,
    *,
    row_labels: Optional[List[Any]] = None,
    row_values: Optional[np.ndarray] = None,
    col_values: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if values.ndim != 2 or not np.any(np.isfinite(values)):
        return {}

    def entry(index: Tuple[int, int]) -> Dict[str, Any]:
        row, col = index
        item: Dict[str, Any] = {"value": float(values[row, col]), "row_index": int(row), "col_index": int(col)}
        if row_labels is not None and row < len(row_labels):
            item["row_label"] = row_labels[row]
        if row_values is not None and row_values.size > row:
            item["row_value"] = float(row_values[row])
        if col_values is not None and col_values.size > col:
            item["col_value"] = float(col_values[col])
        return item

    max_index = tuple(int(idx) for idx in np.unravel_index(int(np.nanargmax(values)), values.shape))
    min_index = tuple(int(idx) for idx in np.unravel_index(int(np.nanargmin(values)), values.shape))
    return {"max": entry(max_index), "min": entry(min_index)}


def _select_event_record(events: List[Dict[str, Any]], *, keys: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    best_event: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    best_key: Optional[str] = None
    for event in events:
        if not isinstance(event, dict):
            continue
        for key in keys:
            score = _safe_float(event.get(key))
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_event = event
                best_key = key
            break
    if not isinstance(best_event, dict):
        return None
    compact = {
        key: _compact_evidence_value(best_event.get(key))
        for key in ("event_id", "timestamp", "duration_days", "area_km2", "center", "max_intensity", "mean_intensity", "min_oxygen")
        if best_event.get(key) is not None
    }
    if best_key:
        compact["ranking_metric"] = best_key
    return compact


def _bounded_numeric_array(value: Any, *, max_points: int) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, dict) and value.get("__array_omitted__"):
        return np.asarray([], dtype=float)
    if hasattr(value, "chunks") or is_dask_backed(value):
        return np.asarray([], dtype=float)
    if hasattr(value, "data") and hasattr(value.data, "chunks"):
        return np.asarray([], dtype=float)
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            size = int(np.prod([int(dim) for dim in shape]))
            if size > max_points:
                return np.asarray([], dtype=float)
        except Exception:
            return np.asarray([], dtype=float)
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    if array.size > max_points:
        return np.asarray([], dtype=float)
    return array


def _compact_scalar_dict(source: Dict[str, Any]) -> Dict[str, Any]:
    scalar_keys = {
        "count",
        "n",
        "n_points",
        "n_valid",
        "finite_count",
        "n_valid_points",
        "n_missing_points",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "total",
        "n_nonzero",
        "nonzero_fraction",
        "slope",
        "intercept",
        "r_squared",
        "p_value",
        "std_err",
        "is_significant",
        "confidence_level",
        "event_count",
        "total_count",
        "max_duration_days",
        "mean_duration_days",
        "max_area_km2",
        "total_area_km2",
        "summary_mode",
        "event_type",
        "trend_direction",
        "fitted_change_over_period",
    }
    compact: Dict[str, Any] = {}
    for key, value in source.items():
        if key not in scalar_keys and key not in {
            "max_location",
            "min_location",
            "top_hotspots",
            "hotspot_region_label",
            "coldspot_region_label",
        }:
            continue
        compact_value = _compact_evidence_value(value)
        if compact_value is not None:
            compact[key] = compact_value
    return compact


def _compact_evidence_value(value: Any, *, max_items: int = 8, max_depth: int = 3) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        return number
    if max_depth <= 0:
        return None
    if isinstance(value, (list, tuple)):
        items = [
            _compact_evidence_value(item, max_items=max_items, max_depth=max_depth - 1)
            for item in list(value)[:max_items]
        ]
        items = [item for item in items if item is not None]
        if len(value) > max_items:
            items.append(f"... ({len(value) - max_items} more)")
        return items
    if isinstance(value, dict):
        compact: Dict[str, Any] = {}
        for key, item in list(value.items())[:max_items]:
            compact_item = _compact_evidence_value(item, max_items=max_items, max_depth=max_depth - 1)
            if compact_item is not None:
                compact[str(key)] = compact_item
        if len(value) > max_items:
            compact["..."] = f"{len(value) - max_items} more keys"
        return compact
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _validated_ocean_region_label(label: Any, lon: Optional[float], lat: Optional[float]) -> Optional[str]:
    if not isinstance(label, str) or not label.strip():
        return None
    cleaned = label.strip()
    lowered = cleaned.lower()
    known_region_checks = (
        (("liaodong", "辽东"), _is_liaodong_bay),
        (("bohai bay", "渤海湾"), _is_bohai_bay),
        (("laizhou", "莱州"), _is_laizhou_bay),
        (("hangzhou", "杭州"), _is_hangzhou_bay),
        (("southern yellow sea", "south yellow sea", "南黄海"), _is_southern_yellow_sea_shelf),
        (("north yellow sea", "northern yellow sea", "北黄海"), _is_northern_yellow_sea_shelf),
        (("shandong peninsula", "山东半岛"), _is_shandong_peninsula_shelf),
        (("bohai", "渤海"), _is_bohai_sea),
        (("yellow sea", "黄海"), _is_yellow_sea),
        (("yangtze", "长江"), _is_yangtze_estuary_shelf),
        (("east china sea", "东海"), _is_east_china_sea_shelf),
        (("pearl river", "珠江"), _is_pearl_river_estuary),
        (("tonkin", "北部湾"), _is_gulf_of_tonkin),
        (("south china sea", "南海"), _is_south_china_sea_shelf_or_basin),
        (("taiwan strait", "台湾海峡"), _is_taiwan_strait_or_luzon_approach),
        (("luzon", "吕宋"), _is_taiwan_strait_or_luzon_approach),
    )
    for terms, check in known_region_checks:
        if any(term in lowered for term in terms):
            return cleaned if check(lon, lat) else None
    return cleaned


def _is_china_coastal_management_domain(lon: Optional[float], lat: Optional[float]) -> bool:
    return any(
        check(lon, lat)
        for check in (
            _is_pearl_river_estuary,
            _is_gulf_of_tonkin,
            _is_taiwan_strait_or_luzon_approach,
            _is_south_china_sea_shelf_or_basin,
            _is_hangzhou_bay,
            _is_yangtze_estuary_shelf,
            _is_east_china_sea_shelf,
            _is_southern_yellow_sea_shelf,
            _is_northern_yellow_sea_shelf,
            _is_shandong_peninsula_shelf,
            _is_yellow_sea,
            _is_liaodong_bay,
            _is_bohai_bay,
            _is_laizhou_bay,
            _is_bohai_sea,
        )
    )


def _coord_in_box(
    lon: Optional[float],
    lat: Optional[float],
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> bool:
    return (
        lon is not None
        and lat is not None
        and lon_min <= lon <= lon_max
        and lat_min <= lat <= lat_max
    )


def _is_pearl_river_estuary(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=112.0, lon_max=116.5, lat_min=20.0, lat_max=24.0)


def _is_gulf_of_tonkin(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=105.0, lon_max=110.5, lat_min=17.0, lat_max=22.0)


def _is_taiwan_strait_or_luzon_approach(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=119.0, lon_max=123.5, lat_min=22.0, lat_max=26.5)


def _is_south_china_sea_shelf_or_basin(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=108.0, lon_max=121.5, lat_min=5.0, lat_max=23.5)


def _is_liaodong_bay(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=120.0, lon_max=122.4, lat_min=39.5, lat_max=41.3)


def _is_bohai_bay(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=117.0, lon_max=119.8, lat_min=37.4, lat_max=39.5)


def _is_laizhou_bay(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=119.0, lon_max=121.2, lat_min=36.8, lat_max=38.4)


def _is_hangzhou_bay(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=120.0, lon_max=122.7, lat_min=29.5, lat_max=31.3)


def _is_yangtze_estuary_shelf(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=120.0, lon_max=124.0, lat_min=29.0, lat_max=33.0)


def _is_east_china_sea_shelf(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=118.0, lon_max=126.0, lat_min=24.0, lat_max=33.0)


def _is_yellow_sea(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=119.0, lon_max=126.0, lat_min=32.0, lat_max=39.5)


def _is_northern_yellow_sea_shelf(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=122.0, lon_max=126.0, lat_min=37.0, lat_max=39.5)


def _is_southern_yellow_sea_shelf(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=120.0, lon_max=126.0, lat_min=32.0, lat_max=35.5)


def _is_shandong_peninsula_shelf(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=119.0, lon_max=123.5, lat_min=35.0, lat_max=38.5)


def _is_bohai_sea(lon: Optional[float], lat: Optional[float]) -> bool:
    return _coord_in_box(lon, lat, lon_min=117.0, lon_max=122.2, lat_min=37.0, lat_max=41.3)


def _label_ocean_region(lon: Optional[float], lat: Optional[float]) -> Optional[str]:
    if lon is None or lat is None:
        return None
    if _is_pearl_river_estuary(lon, lat):
        return "Pearl River Estuary / northern South China Sea shelf"
    if _is_gulf_of_tonkin(lon, lat):
        return "Gulf of Tonkin / western northern South China Sea"
    if _is_taiwan_strait_or_luzon_approach(lon, lat):
        return "Taiwan Strait / Luzon Strait approach"
    if _is_south_china_sea_shelf_or_basin(lon, lat):
        return "South China Sea shelf/basin"
    if _is_hangzhou_bay(lon, lat):
        return "Hangzhou Bay / inner East China Sea coast"
    if _is_yangtze_estuary_shelf(lon, lat):
        return "Yangtze River Estuary shelf"
    if _is_liaodong_bay(lon, lat):
        return "Liaodong Bay"
    if _is_bohai_bay(lon, lat):
        return "Bohai Bay"
    if _is_laizhou_bay(lon, lat):
        return "Laizhou Bay"
    if _is_bohai_sea(lon, lat):
        return "Bohai Sea / Bohai Strait"
    if _is_northern_yellow_sea_shelf(lon, lat):
        return "Northern Yellow Sea shelf"
    if _is_southern_yellow_sea_shelf(lon, lat):
        return "Southern Yellow Sea shelf"
    if _is_shandong_peninsula_shelf(lon, lat):
        return "Shandong Peninsula shelf"
    if _is_yellow_sea(lon, lat):
        return "Yellow Sea shelf"
    if _is_east_china_sea_shelf(lon, lat):
        return "East China Sea shelf"
    return None


def _apply_lag_selection_rewrite(
    result_summaries: Dict[str, Dict[str, Any]],
    result_cards: List[Dict[str, Any]],
    step_cards_by_id: Dict[str, Dict[str, Any]],
    synthesis_payload: Optional[Dict[str, Any]],
    *,
    chinese: bool,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(synthesis_payload, dict):
        return result_summaries

    overrides_by_result = _index_lag_selection_overrides(synthesis_payload.get("lag_selection_overrides"))
    rewritten_result_ids: List[str] = []
    for result_id, summary in list(result_summaries.items()):
        if not isinstance(summary, dict) or summary.get("type") != "lag_correlation_result":
            continue
        rewritten_summary = _rewrite_lag_correlation_summary(
            summary,
            overrides_by_result.get(result_id),
        )
        result_summaries[result_id] = rewritten_summary
        rewritten_result_ids.append(result_id)

    if not rewritten_result_ids:
        return result_summaries

    rewritten_result_id_set = set(rewritten_result_ids)
    for result_card in result_cards:
        result_id = str(result_card.get("id") or "").strip()
        if result_id in rewritten_result_id_set:
            _refresh_result_card_from_summary(
                result_card,
                result_summaries[result_id],
                chinese=chinese,
            )

    for step_card in step_cards_by_id.values():
        results = step_card.get("results")
        if not isinstance(results, list):
            continue
        for result_card in results:
            if not isinstance(result_card, dict):
                continue
            result_id = str(result_card.get("id") or "").strip()
            if result_id in rewritten_result_id_set:
                _refresh_result_card_from_summary(
                    result_card,
                    result_summaries[result_id],
                    chinese=chinese,
                )
        step_card["is_map_bound"] = any(
            isinstance(result, dict) and result.get("surface") == "map"
            for result in results
        )
        if results:
            actions = results[-1].get("actions")
            if isinstance(actions, list):
                step_card["actions"] = actions

    return result_summaries


def _index_lag_selection_overrides(overrides: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(overrides, list):
        return {}
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in overrides:
        if not isinstance(item, dict):
            continue
        result_id = str(item.get("result_id") or "").strip()
        if result_id:
            indexed[result_id] = item
    return indexed


def _rewrite_lag_correlation_summary(
    summary: Dict[str, Any],
    selection: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    rewritten = dict(summary)
    selected_mode = "symmetric"
    selection_reason = "No clear directional ordering was retained, so the symmetric raw optimum remains official."

    if isinstance(selection, dict) and selection.get("has_clear_directionality") is True:
        requested_mode = str(selection.get("selected_mode") or "").strip().lower()
        if requested_mode in {"positive", "negative"} and _lag_candidate_for_mode(rewritten, requested_mode) is not None:
            selected_mode = requested_mode
            selection_reason = str(selection.get("reason") or "").strip() or (
                "Directionality was judged meaningful, so the directional candidate was promoted."
            )
        else:
            selection_reason = (
                str(selection.get("reason") or "").strip()
                or "Directionality was requested, but a valid directional candidate was unavailable, so the symmetric raw optimum remains official."
            )
    elif isinstance(selection, dict):
        selection_reason = str(selection.get("reason") or "").strip() or selection_reason

    candidate = _lag_candidate_for_mode(rewritten, selected_mode) or _lag_candidate_for_mode(rewritten, "symmetric")
    if candidate is None:
        return rewritten

    rewritten["optimal_lag"] = candidate["lag"]
    rewritten["max_correlation"] = candidate["correlation"]
    rewritten["peak_correlation"] = candidate["correlation"]
    rewritten["lag_selection_mode"] = selected_mode
    rewritten["lag_selection_reason"] = selection_reason

    if candidate.get("p_value") is not None:
        rewritten["optimal_lag_p_value"] = candidate["p_value"]
    elif "optimal_lag_p_value" in rewritten:
        rewritten["optimal_lag_p_value"] = None

    step_days = rewritten.get("median_step_days")
    if isinstance(step_days, (int, float)) and isinstance(candidate["lag"], (int, float)):
        rewritten["optimal_lag_days"] = float(candidate["lag"]) * float(step_days)
    else:
        rewritten["optimal_lag_days"] = None

    return rewritten


def _lag_candidate_for_mode(summary: Dict[str, Any], mode: str) -> Optional[Dict[str, Any]]:
    if mode == "positive":
        lag = summary.get("best_positive_lag")
        correlation = summary.get("best_positive_correlation")
        p_value = summary.get("best_positive_p_value")
    elif mode == "negative":
        lag = summary.get("best_negative_lag")
        correlation = summary.get("best_negative_correlation")
        p_value = summary.get("best_negative_p_value")
    else:
        lag = summary.get("symmetric_optimal_lag", summary.get("optimal_lag"))
        correlation = summary.get("symmetric_max_correlation", summary.get("max_correlation"))
        p_value = summary.get("optimal_lag_p_value")

    if lag is None or correlation is None:
        return None

    candidate = {
        "lag": lag,
        "correlation": correlation,
    }
    if p_value is not None:
        candidate["p_value"] = p_value
    return candidate


def _refresh_result_card_from_summary(
    result_card: Dict[str, Any],
    summary: Dict[str, Any],
    *,
    chinese: bool,
) -> None:
    owner_step_id = result_card.get("owner_step_id")
    refreshed = _build_result_card(
        str(result_card.get("id") or ""),
        summary,
        chinese=chinese,
    )
    refreshed["actions"] = _build_result_actions(refreshed["surface"], chinese=chinese)
    if owner_step_id is not None:
        refreshed["owner_step_id"] = owner_step_id
    result_card.clear()
    result_card.update(refreshed)


def _build_environment_assessment_scientific_findings(
    summary: Dict[str, Any],
    *,
    active_result_id: Optional[str],
    chinese: bool,
) -> List[Dict[str, Any]]:
    if not isinstance(summary, dict) or summary.get("type") != "environment_assessment_result":
        return []
    localized_summary = _localize_environment_assessment_summary(summary, chinese=chinese)
    result_ids = [active_result_id] if isinstance(active_result_id, str) and active_result_id.strip() else []

    findings: List[Dict[str, Any]] = []
    verdict = str(localized_summary.get("overall_verdict") or "").strip()
    support = str(localized_summary.get("overall_support_strength") or "").strip()
    narrative = str(localized_summary.get("overall_narrative") or "").strip()
    if narrative:
        findings.append(
            {
                "finding": narrative,
                "evidence": [narrative],
                "result_ids": result_ids,
            }
        )
    elif verdict:
        verdict_line = (
            f"Overall marine environmental health is assessed as {verdict} with {support} support."
            if not chinese
            else f"海洋环境健康总体评估为{_localize_enum_value(verdict, chinese=True)}，证据强度为{_localize_enum_value(support, chinese=True)}。"
        )
        findings.append({"finding": verdict_line, "evidence": [verdict_line], "result_ids": result_ids})

    branches = localized_summary.get("branch_assessments")
    if isinstance(branches, list):
        deteriorating_branches = [
            item for item in branches
            if isinstance(item, dict) and item.get("direction") == "deteriorating" and item.get("support_strength") in {"supported", "limited"}
        ]
        if deteriorating_branches:
            top_branches = sorted(deteriorating_branches, key=_environment_branch_priority_local, reverse=True)[:3]
            labels = [
                str(item.get("indicator_label") or item.get("name") or "").strip()
                for item in top_branches
                if str(item.get("indicator_label") or item.get("name") or "").strip()
            ]
            if labels:
                branch_finding = (
                    f"The main pressures are {', '.join(labels)}."
                    if not chinese
                    else f"主要环境压力来自{ '、'.join(labels) }。"
                )
                branch_evidence = [
                    _trim_sentence_ending(str(item.get("summary") or "").strip())
                    for item in top_branches
                    if str(item.get("summary") or "").strip()
                ]
                findings.append(
                    {
                        "finding": branch_finding,
                        "evidence": branch_evidence[:3] or [branch_finding],
                        "result_ids": result_ids,
                    }
                )

    return findings[:3]


def _language_matches_query(text: str, *, chinese: bool) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    has_cjk = _contains_cjk(stripped)
    return has_cjk if chinese else not has_cjk


def _filter_synthesis_findings_by_language(
    synthesis_payload: Optional[Dict[str, Any]],
    *,
    chinese: bool,
) -> Optional[Dict[str, Any]]:
    if not isinstance(synthesis_payload, dict):
        return synthesis_payload
    findings = synthesis_payload.get("scientific_findings")
    if not isinstance(findings, list):
        return synthesis_payload
    filtered_findings: List[Dict[str, Any]] = []
    for item in findings:
        if not isinstance(item, dict):
            continue
        finding_text = str(item.get("finding") or "").strip()
        if not _language_matches_query(finding_text, chinese=chinese):
            continue
        item_copy = dict(item)
        evidence = item.get("evidence")
        if isinstance(evidence, list):
            item_copy["evidence"] = [
                str(entry).strip()
                for entry in evidence
                if isinstance(entry, str) and str(entry).strip() and _language_matches_query(str(entry), chinese=chinese)
            ]
        filtered_findings.append(item_copy)
    synthesis_payload = dict(synthesis_payload)
    synthesis_payload["scientific_findings"] = filtered_findings
    return synthesis_payload


def _pick_active_result_id(plan: Dict[str, Any], result_cards: List[Dict[str, Any]]) -> Optional[str]:
    for step in reversed(plan.get("steps", [])):
        result_id = step.get("save_as")
        if not result_id:
            continue
        if any(card["id"] == result_id for card in result_cards):
            return result_id
    return result_cards[0]["id"] if result_cards else None


def _build_plan_summary(user_request: str, plan: Optional[Dict[str, Any]], *, chinese: bool = False) -> str:
    if isinstance(plan, dict):
        steps = plan.get("steps", [])
        skills_used = plan.get("skills_used") or ([plan.get("skill_id")] if plan.get("skill_id") else [])
        if isinstance(steps, list) and steps:
            if chinese:
                return (
                    f"已生成 {len(steps)} 步分析流程"
                    + (f"，使用技能 {', '.join(str(skill) for skill in skills_used if skill)}" if skills_used else "")
                )
            return (
                f"{len(steps)}-step workflow ready"
                + (f" using {', '.join(str(skill) for skill in skills_used if skill)}" if skills_used else "")
            )
    if user_request.strip():
        return f"正在为以下问题准备分析：{user_request.strip()}" if chinese else f"Preparing analysis for: {user_request.strip()}"
    return "分析计划已就绪。" if chinese else "Plan ready."


def _find_plan_step(plan: Optional[Dict[str, Any]], step_id: str) -> Optional[Dict[str, Any]]:
    if not isinstance(plan, dict):
        return None
    for step in plan.get("steps", []):
        if isinstance(step, dict) and step.get("step_id") == step_id:
            return step
    return None


def _ordered_step_cards(step_cards_by_id: Dict[str, Dict[str, Any]], step_order: List[str]) -> List[Dict[str, Any]]:
    return [_json_safe(step_cards_by_id[step_id]) for step_id in step_order if step_id in step_cards_by_id]


def _build_result_card(
    result_id: str,
    summary: Dict[str, Any],
    step_tool: Optional[str] = None,
    *,
    chinese: bool = False,
) -> Dict[str, Any]:
    output_type = summary.get("type") or "generic_result"
    title = _build_result_title(summary, step_tool=step_tool, result_id=result_id, chinese=chinese)
    renderer = _result_type_to_renderer(output_type)
    headline = _build_headline(summary, chinese=chinese)
    description = _build_description(summary, chinese=chinese)
    metrics = _build_metrics(summary, chinese=chinese)
    detail_sections = _build_detail_sections(summary, chinese=chinese)
    interpretation = _build_card_interpretation(summary, chinese=chinese)

    card = {
        "id": result_id,
        "title": title,
        "type": output_type,
        "headline": headline,
        "description": description,
        "renderer": renderer,
        "metrics": metrics,
        "surface": _result_surface_for_summary(summary, output_type),
        "interpretation": interpretation,
        "detail_sections": detail_sections,
    }
    if output_type == "timeseries_result":
        finite_count = int(summary.get("finite_count") or 0)
        has_finite = summary.get("has_finite_values")
        card["finiteCount"] = finite_count
        card["hasFiniteValues"] = bool(has_finite) if has_finite is not None else finite_count > 0
    return card


VARIABLE_LABELS: Dict[str, str] = {
    "temp": "Temperature",
    "sst": "SST",
    "salt": "Salinity",
    "salinity": "Salinity",
    "u": "Zonal Current",
    "v": "Meridional Current",
    "oxygen": "Oxygen",
    "density": "Density",
    "vorticity": "Relative Vorticity",
    "relative_vorticity": "Relative Vorticity",
    "current_speed": "Current Speed",
    "speed": "Current Speed",
    "chlorophyll": "Chlorophyll",
    "chla": "Chlorophyll",
    "mixed_layer_depth": "Mixed-Layer Depth",
}

VARIABLE_LABELS_ZH: Dict[str, str] = {
    "temp": "温度",
    "sst": "海表温度",
    "salt": "盐度",
    "salinity": "盐度",
    "u": "纬向流速",
    "v": "经向流速",
    "oxygen": "氧气",
    "density": "密度",
    "vorticity": "相对涡度",
    "relative_vorticity": "相对涡度",
    "current_speed": "流速",
    "speed": "流速",
    "chlorophyll": "叶绿素",
    "chla": "叶绿素",
    "mixed_layer_depth": "混合层深度",
}

AGGREGATION_LABELS: Dict[str, str] = {
    "layer_mean": "Layer Mean",
    "horizontal_advection": "Horizontal Advection",
    "vertical_advection": "Vertical Advection",
    "horizontal_gradient": "Horizontal Gradient",
    "vertical_gradient": "Vertical Gradient",
    "buoyancy_frequency": "Buoyancy Frequency",
    "climatology": "Climatology",
    "anomaly": "Anomaly",
    "mean": "Mean",
}

AGGREGATION_LABELS_ZH: Dict[str, str] = {
    "layer_mean": "层平均",
    "horizontal_advection": "水平平流",
    "vertical_advection": "垂向平流",
    "horizontal_gradient": "水平梯度",
    "vertical_gradient": "垂向梯度",
    "buoyancy_frequency": "浮力频率",
    "climatology": "气候态",
    "anomaly": "异常",
    "mean": "平均",
}


def _humanize_data_label(value: Any, *, chinese: bool = False) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in {"unknown", "none", "null", "nan"}:
        return None
    variable_labels = VARIABLE_LABELS_ZH if chinese else VARIABLE_LABELS
    aggregation_labels = AGGREGATION_LABELS_ZH if chinese else AGGREGATION_LABELS
    if lowered in variable_labels:
        return variable_labels[lowered]
    if lowered in aggregation_labels:
        return aggregation_labels[lowered]
    for suffix, label in aggregation_labels.items():
        token = f"_{suffix}"
        if lowered.endswith(token):
            prefix = cleaned[: -len(token)]
            prefix_label = _humanize_data_label(prefix, chinese=chinese) or prefix.replace("_", " ").strip().title()
            if chinese:
                return f"{prefix_label}{label}"
            return f"{label} of {prefix_label}"
    return cleaned.replace("_", " ").strip().title()


def _data_container_variable_label(summary: Dict[str, Any], *, chinese: bool = False) -> Optional[str]:
    variable_label = _humanize_data_label(summary.get("variable"), chinese=chinese)
    if variable_label:
        return variable_label

    variables = summary.get("variables")
    if isinstance(variables, list):
        cleaned = [str(item).strip() for item in variables if str(item).strip()]
        if len(cleaned) == 1:
            return _humanize_data_label(cleaned[0], chinese=chinese) or cleaned[0]
        if len(cleaned) > 1:
            return f"{len(cleaned)} 个变量" if chinese else f"{len(cleaned)} variables"
    return None


def _format_data_container_shape(summary: Dict[str, Any]) -> str:
    dims = summary.get("dims")
    shape = summary.get("shape")
    if isinstance(dims, list) and isinstance(shape, list) and len(dims) == len(shape) and dims and shape:
        dim_label = "×".join(str(dim) for dim in dims)
        shape_label = "×".join(_format_metric(size) for size in shape)
        return f"{dim_label} = {shape_label}"

    dim_sizes = summary.get("dim_sizes")
    if isinstance(dim_sizes, dict) and dim_sizes:
        ordered_dims = [str(dim) for dim in dims if str(dim) in dim_sizes] if isinstance(dims, list) else []
        ordered_dims.extend(str(dim) for dim in dim_sizes.keys() if str(dim) not in ordered_dims)
        return " × ".join(f"{dim} {_format_metric(dim_sizes[dim])}" for dim in ordered_dims)

    if isinstance(shape, list) and shape:
        return "×".join(_format_metric(size) for size in shape)
    return ""


def _format_data_coord_value(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if "T" in text and len(text) >= 10:
            return text[:10]
        return text
    return _format_number(value)


def _format_data_coord_range(value: Any) -> str:
    if isinstance(value, dict):
        start = value.get("start")
        end = value.get("end")
        if start is not None and end is not None:
            return f"{_format_data_coord_value(start)} to {_format_data_coord_value(end)}"
        return ""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{_format_data_coord_value(value[0])} to {_format_data_coord_value(value[1])}"
    return ""


def _data_coord_ranges(summary: Dict[str, Any]) -> Dict[str, Any]:
    coord_ranges = summary.get("coord_ranges")
    return coord_ranges if isinstance(coord_ranges, dict) else {}


def _data_container_depth_label(summary: Dict[str, Any], *, chinese: bool = False) -> str:
    vertical_mode = str(summary.get("vertical_mode") or "").strip().lower()
    bottom_coord = str(summary.get("bottom_depth_coordinate") or "").strip().lower()
    bottom_selection = str(summary.get("bottom_selection") or "").strip().lower()
    if vertical_mode == "bottom" and (
        bottom_coord == "per_cell" or bottom_selection == "local_deepest_finite"
    ):
        return "逐格点海底有效层" if chinese else "per-cell bottom valid layer"
    return ""


def _mechanism_metadata(summary: Dict[str, Any]) -> Dict[str, Any]:
    metadata = summary.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _mechanism_source_proxy_label(summary: Dict[str, Any], *, chinese: bool = False) -> Optional[str]:
    metadata = _mechanism_metadata(summary)
    proxy_name = metadata.get("source_proxy")
    if not isinstance(proxy_name, str):
        return None
    labels = {
        "front_proximity": "锋面代理" if chinese else "Front Proxy",
        "eddy_influence": "涡旋代理" if chinese else "Eddy Proxy",
        "gradient_alignment": "梯度-流向对齐代理" if chinese else "Gradient-Flow Alignment Proxy",
        "flow_context": "背景流场代理" if chinese else "Background-Flow Context Proxy",
    }
    return labels.get(proxy_name.strip().lower())


def _subregion_status_label(status: str, *, chinese: bool = False) -> str:
    normalized = str(status or "").strip().lower()
    labels = {
        "ok": "有效" if chinese else "valid",
        "skipped_no_valid_ocean": "跳过：无有效海域" if chinese else "skipped: no valid ocean cells",
        "skipped_no_valid_samples": "跳过：无有效事件/背景样本" if chinese else "skipped: no valid event/background samples",
    }
    return labels.get(normalized, status)


def _format_subregion_breakdown_line(item: Dict[str, Any], *, chinese: bool = False) -> str:
    label = str(item.get("label") or item.get("subregion_id") or "subregion").strip()
    status = str(item.get("status") or "").strip()
    if status and status != "ok":
        return (
            f"{label}: {_subregion_status_label(status, chinese=chinese)}"
            if chinese
            else f"{label}: {_subregion_status_label(status, chinese=chinese)}"
        )

    contrast = item.get("standardized_contrast")
    claim = str(item.get("claim_strength") or "").strip()
    weight = item.get("area_weight")
    parts = [f"{label}: {_format_metric(contrast)}"] if contrast is not None else [f"{label}"]
    if claim:
        parts.append(_localize_enum_value(claim, chinese=chinese))
    if isinstance(weight, (int, float)):
        parts.append(
            f"权重 {_format_metric(weight)}" if chinese else f"weight {_format_metric(weight)}"
        )
    separator = "，" if chinese else ", "
    return separator.join(parts)


def _format_proxy_breakdown_line(item: Dict[str, Any], *, chinese: bool = False) -> str:
    name = _humanize_mechanism_name(item.get("name"), chinese=chinese)
    score = item.get("score")
    claim = str(item.get("claim_strength") or "").strip()
    base_parts = [f"{name}: {_format_metric(score)}" if score is not None else (name or "proxy")]
    if claim:
        base_parts.append(_localize_enum_value(claim, chinese=chinese))

    subregions = item.get("subregion_breakdown")
    valid_lines = []
    if isinstance(subregions, list):
        valid_lines = [
            _format_subregion_breakdown_line(subitem, chinese=chinese)
            for subitem in subregions[:4]
            if isinstance(subitem, dict)
        ]
    if valid_lines:
        joiner = "；" if chinese else "; "
        breakdown_text = ("子区 " if chinese else "subregions ") + joiner.join(valid_lines)
        base_parts.append(breakdown_text)

    separator = "，" if chinese else ", "
    return separator.join(part for part in base_parts if part)


def _build_result_title(summary: Dict[str, Any], step_tool: Optional[str], result_id: str, *, chinese: bool = False) -> str:
    output_type = str(summary.get("type") or "generic_result")
    explicit_title = summary.get("title")
    if isinstance(explicit_title, str) and explicit_title.strip():
        return _localize_explicit_title(explicit_title.strip(), chinese=chinese)
    variable_label = _humanize_data_label(summary.get("variable"), chinese=chinese)
    aggregation_label = _humanize_data_label(summary.get("aggregation"), chinese=chinese)
    feature_label = _humanize_data_label(summary.get("feature"), chinese=chinese)
    base_title = _humanize_tool_name(step_tool, chinese=chinese) if step_tool else _humanize_result_id(result_id)

    if output_type == "data_container_result":
        if _is_spatial_data_container_summary(summary):
            if variable_label and aggregation_label:
                return f"{aggregation_label}{variable_label}图" if chinese else f"{aggregation_label} {variable_label} Map"
            if variable_label:
                return f"{variable_label}图" if chinese else f"{variable_label} Map"
        if variable_label and aggregation_label:
            return f"{aggregation_label}{variable_label}场" if chinese else f"{aggregation_label} {variable_label} Field"
        if variable_label:
            return f"已加载{variable_label}数据" if chinese else f"Loaded {variable_label} Data"
    if output_type == "climatology_result":
        period_label = _climatology_period_label(summary.get("period"))
        if variable_label and period_label:
            return f"{period_label}{variable_label}气候态" if chinese else f"{period_label} {variable_label} Climatology"
        if variable_label:
            return f"{variable_label}气候态" if chinese else f"{variable_label} Climatology"
        if period_label:
            return f"{period_label}气候态" if chinese else f"{period_label} Climatology"
        return "气候态" if chinese else "Climatology"
    if output_type in {"timeseries_result", "trend_result"}:
        if variable_label and aggregation_label:
            return f"{aggregation_label}{variable_label}时间序列" if chinese else f"{aggregation_label} {variable_label} Time Series"
        if variable_label:
            return f"{variable_label}时间序列" if chinese else f"{variable_label} Time Series"
    if output_type == "spectrum_result":
        if variable_label:
            return f"{variable_label}功率谱" if chinese else f"{variable_label} Power Spectrum"
        return "功率谱" if chinese else "Power Spectrum"
    if output_type == "field_trend_result":
        if variable_label:
            return f"{variable_label}趋势斜率图" if chinese else f"{variable_label} Trend Slope Map"
        return "趋势斜率图" if chinese else "Trend Slope Map"
    if output_type == "spatial_field_result":
        if variable_label and aggregation_label in {
            "Horizontal Advection",
            "Vertical Advection",
            "Horizontal Gradient",
            "Vertical Gradient",
            "Buoyancy Frequency",
            "水平平流",
            "垂向平流",
            "水平梯度",
            "垂向梯度",
            "浮力频率",
        }:
            return f"{variable_label}{aggregation_label}" if chinese else f"{aggregation_label} of {variable_label}"
        if variable_label and aggregation_label:
            return f"{aggregation_label}{variable_label}图" if chinese else f"{aggregation_label} {variable_label} Map"
        if variable_label:
            return f"{variable_label}图" if chinese else f"{variable_label} Map"
    if output_type == "regression_map_result":
        if variable_label:
            return f"{variable_label}回归图" if chinese else f"{variable_label} Regression Map"
        return "回归图" if chinese else "Regression Map"
    if output_type == "profile_result" and variable_label:
        return f"{variable_label}垂向剖面" if chinese else f"{variable_label} Vertical Profile"
    if output_type == "hovmoller_result" and variable_label:
        return f"{variable_label}霍夫默勒图" if chinese else f"{variable_label} Hovmoller Diagram"
    if output_type == "ts_diagram_result":
        return "温盐图" if chinese else "Temperature-Salinity Diagram"
    if output_type == "watermass_event_association_result":
        return "事件-水团格网关联" if chinese else "Event-Watermass Tile Association"
    if output_type in {"histogram_result", "histogram_2d_result"}:
        label = feature_label or variable_label
        if label:
            return f"{label}分布" if chinese else f"{label} Distribution"
    if output_type == "eof_result" and variable_label:
        return f"{variable_label}EOF分析" if chinese else f"{variable_label} EOF Analysis"
    if output_type == "composite_result":
        if variable_label:
            return f"{variable_label}合成分析" if chinese else f"{variable_label} Composite Analysis"
        return "合成分析" if chinese else "Composite Analysis"
    if output_type == "lag_correlation_result":
        return "领先-滞后关系" if chinese else "Lead-Lag Relationship"
    if output_type == "mechanism_score_result":
        metadata = _mechanism_metadata(summary)
        proxy_label = _mechanism_source_proxy_label(summary, chinese=chinese)
        if metadata.get("comparison") == "mesoscale_proxy_ranking":
            return "中尺度代理机制排序" if chinese else "Mesoscale Proxy Ranking"
        if metadata.get("comparison") == "event_condition_contrast" and proxy_label:
            return f"{proxy_label}事件对比" if chinese else f"{proxy_label} Event Contrast"
        return "机制排序" if chinese else "Mechanism Ranking"
    if output_type == "evidence_report_result":
        return "证据报告" if chinese else "Evidence Report"
    if output_type == "environment_assessment_result":
        return "海洋环境健康评估" if chinese else "Environment Health Assessment"
    if output_type == "policy_recommendation_result":
        return "政策建议报告" if chinese else "Policy Recommendation Report"

    return base_title


def _result_type_to_renderer(output_type: str) -> str:
    if output_type in {"spatial_field_result", "data_container_result", "regression_map_result"}:
        return "reference"
    if output_type in {"timeseries_result", "trend_result", "climatology_result", "spectrum_result", "lag_correlation_result"}:
        return "timeseries"
    if output_type == "ts_diagram_result":
        return "ts_diagram"
    if output_type == "profile_result":
        return "profile"
    if output_type == "hovmoller_result":
        return "hovmoller"
    if output_type == "section_result":
        return "section"
    if output_type in {"histogram_result", "histogram_2d_result"}:
        return "histogram"
    if output_type == "eof_result":
        return "eof"
    if output_type == "composite_result":
        return "composite"
    if output_type in {"event_statistics_result", "mechanism_score_result", "watermass_event_association_result", "evidence_report_result", "environment_assessment_result", "policy_recommendation_result"}:
        return "summary"
    if output_type == "event_comparison_result":
        return "summary"
    if output_type in {
        "event_detection_result",
        "event_spatial_distribution_result",
    }:
        return "event"
    return "reference"


def _result_type_to_surface(output_type: str) -> str:
    if output_type in {"timeseries_result", "trend_result", "climatology_result", "histogram_result", "histogram_2d_result", "ts_diagram_result", "spectrum_result", "lag_correlation_result", "hovmoller_result"}:
        return "inline"
    if output_type in {"spatial_field_result", "field_trend_result", "event_detection_result", "event_spatial_distribution_result", "regression_map_result"}:
        return "map"
    if output_type == "composite_result":
        return "drawer"
    if output_type == "section_result":
        return "drawer"
    if output_type == "eof_result":
        return "inline"
    if output_type in {"event_comparison_result", "mechanism_score_result", "watermass_event_association_result", "evidence_report_result", "environment_assessment_result", "policy_recommendation_result"}:
        return "summary"
    return "summary"


def _is_spatial_data_container_summary(summary: Dict[str, Any]) -> bool:
    if summary.get("type") != "data_container_result":
        return False
    dims = summary.get("dims")
    return isinstance(dims, list) and "lat" in dims and "lon" in dims


def _result_surface_for_summary(summary: Dict[str, Any], output_type: str) -> str:
    if output_type == "mechanism_score_result" and _build_mechanism_subregion_map_field(summary, title="Mechanism Subregion Diagnosis") is not None:
        return "map"
    if _is_spatial_data_container_summary(summary):
        return "map"
    return _result_type_to_surface(output_type)


def _build_result_actions(surface: str, *, chinese: bool = False) -> List[Dict[str, str]]:
    if surface == "map":
        return [{"id": "focus_map", "label": "在主地图中显示" if chinese else "Show on main map"}]
    if surface == "drawer":
        return [{"id": "open_detail", "label": "展开详情" if chinese else "Expand Details"}]
    if surface == "modal":
        return [{"id": "open_modal", "label": "打开详细分析" if chinese else "Open Modal Analysis"}]
    return []


def _build_source_cards(
    search_results: List[Dict[str, Any]],
    default_reason: Optional[str] = None,
    *,
    chinese: bool = False,
) -> List[Dict[str, Any]]:
    reason = default_reason or ("相关外部背景来源。" if chinese else "Relevant background source.")
    source_cards: List[Dict[str, Any]] = []
    for result in search_results:
        url = str(result.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        source_cards.append(
            {
                "title": result.get("title", ""),
                "source": result.get("source", "web"),
                "url": url,
                "short_snippet": result.get("short_snippet", ""),
                "why_it_matters": result.get("why_it_matters") or reason,
                "provider": result.get("provider"),
                "search_query": result.get("search_query"),
                "rank": result.get("rank"),
            }
        )
    return source_cards


def _build_source_fallback_card(title: str, snippet: str, reason: str) -> List[Dict[str, Any]]:
    return []


def _search_provider_from_results(search_results: List[Dict[str, Any]], search_service: WebSearchService) -> Optional[str]:
    for result in search_results:
        if isinstance(result, dict) and isinstance(result.get("provider"), str) and result["provider"].strip():
            return result["provider"].strip()
    provider = getattr(search_service, "last_provider", None)
    return provider if isinstance(provider, str) and provider.strip() else None


def _build_post_analysis_search_query(
    user_request: str,
    synthesis_payload: Optional[Dict[str, Any]],
    dataset_info: Dict[str, Any],
) -> str:
    """Build search query from analysis summary."""
    summary = (synthesis_payload or {}).get("summary")

    # Use the synthesis summary directly if available
    if isinstance(summary, str) and summary.strip():
        # Extract key phrases from the summary for better search results
        summary_clean = summary.strip()
        # If the summary is very long, use the first sentence or first 150 characters
        if len(summary_clean) > 150:
            # Try to get the first sentence
            first_sentence_end = summary_clean.find('. ')
            if first_sentence_end > 0 and first_sentence_end < 150:
                return summary_clean[:first_sentence_end + 1].strip()
            else:
                return summary_clean[:150].strip() + "..."
        return summary_clean

    # Fallback: use the user request with dataset context
    dataset_name = dataset_info.get("name", "ocean dataset")
    return f"{user_request} {dataset_name} ocean science"


def _attach_result_interpretations(
    step_cards: List[Dict[str, Any]],
    synthesis_payload: Dict[str, Any],
    *,
    chinese: bool,
) -> None:
    findings = synthesis_payload.get("scientific_findings", [])
    result_cards_by_id: Dict[str, Dict[str, Any]] = {}
    for card in step_cards:
        for result in card.get("results", []):
            result_id = result.get("id")
            if isinstance(result_id, str):
                result_cards_by_id[result_id] = result

    for finding in findings:
        if not isinstance(finding, dict):
            continue
        result_ids = finding.get("result_ids") or []
        finding_text = str(finding.get("finding") or "").strip()
        if not finding_text:
            continue
        if not _language_matches_query(finding_text, chinese=chinese):
            continue
        matching_result_cards: List[Dict[str, Any]] = []
        for result_id in result_ids:
            result_card = result_cards_by_id.get(result_id)
            if not result_card:
                continue
            matching_result_cards.append(result_card)
        if len(matching_result_cards) == 1:
            result_type = str(matching_result_cards[0].get("type") or "").strip()
            if result_type in {"event_statistics_result", "environment_assessment_result", "policy_recommendation_result", "lag_correlation_result"}:
                continue
            existing = str(matching_result_cards[0].get("interpretation") or "").strip()
            matching_result_cards[0]["interpretation"] = (
                f"{existing} {finding_text}".strip() if existing else finding_text
            )
            continue
        if step_cards:
            existing = str(step_cards[-1].get("interpretation") or "").strip()
            step_cards[-1]["interpretation"] = f"{existing} {finding_text}".strip() if existing else finding_text


def _is_general_question_clarification(query: str) -> bool:
    query_lower = " ".join(str(query or "").lower().split())
    if not query_lower:
        return False

    escape_markers = (
        "general question",
        "general circumstances",
        "in general",
        "generally",
        "conceptual question",
        "conceptually",
        "without considering region",
        "without considering time",
        "without region",
        "without time",
        "not considering region",
        "not considering time",
        "no region",
        "no time",
        "web search",
        "web-search",
        "search can solve",
        "right or wrong",
        "correct or incorrect",
        "not dataset",
        "no dataset",
    )
    return any(marker in query_lower for marker in escape_markers)


def _pending_message_starts_new_turn(
    *,
    planner: Optional[SkillPlanner],
    latest_query: str,
    pending: Dict[str, Any],
) -> bool:
    if planner is None:
        return False
    payload = {
        "latest_user_message": latest_query,
        "pending_original_query": pending.get("original_query"),
        "pending_clarification_question": pending.get("clarification_question"),
        "pending_missing_fields": pending.get("missing_fields"),
        "decision_schema": {
            "relationship": "answers_pending | starts_new_turn",
            "confidence": 0.0,
            "reason": "brief explanation",
        },
    }
    system = (
        "You decide whether the latest user message answers a pending clarification or starts a new standalone turn.\n"
        "Choose answers_pending only when the latest message directly supplies the missing information, confirms an "
        "offered option, or clearly continues the pending task. Choose starts_new_turn when it asks a different question, "
        "contains a complete new request, changes topic, asks about the assistant, or is otherwise not a direct answer "
        "to the pending clarification. Be conservative about carrying pending context forward: if relevance is unclear, "
        "choose starts_new_turn to avoid contaminating a new query with stale task memory.\n"
        "Return JSON only."
    )
    try:
        client = planner._get_client()
        response = planner._create_message(
            client=client,
            max_tokens=300,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}],
            request_name="pending_turn_relationship",
            json_response=True,
        )
        decision = planner._parse_json_response(planner._extract_response_text(response))
    except Exception:
        _logger.exception("Pending turn relationship check failed; starting a fresh turn to avoid stale context.")
        return True

    relationship = str(decision.get("relationship") or "").strip()
    try:
        confidence = float(decision.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    if relationship == "answers_pending" and confidence >= 0.55:
        return False
    return True


def _humanize_result_id(result_id: str) -> str:
    return result_id.replace("_", " ").strip().title()


def _humanize_tool_name(tool_name: Optional[str], *, chinese: bool = False) -> str:
    if not tool_name:
        return "结果" if chinese else "Result"
    title_map = TOOL_TITLES_ZH if chinese else TOOL_TITLES
    if tool_name in title_map:
        return title_map[tool_name]
    return tool_name.replace("_", " ").strip().title()


def _build_descriptive_label(
    tool_name: Optional[str],
    summary: Optional[Dict[str, Any]],
    *,
    chinese: bool = False,
) -> str:
    """Build natural language description from tool and result summary."""
    if not summary or not isinstance(summary, dict):
        return _humanize_tool_name(tool_name, chinese=chinese)

    if tool_name == "compute_event_summary_map":
        explicit_title = summary.get("title")
        if isinstance(explicit_title, str) and explicit_title.strip():
            localized_title = _localize_explicit_title(explicit_title.strip(), chinese=chinese)
            return f"正在计算{localized_title}" if chinese else f"Computing {localized_title.lower()}"
    variable = _humanize_data_label(summary.get("variable"), chinese=chinese)
    aggregation = _humanize_data_label(summary.get("aggregation"), chinese=chinese)

    if tool_name == "load_dataset" and variable:
        return f"正在读取{variable}数据" if chinese else f"Loading {variable} data"
    if tool_name == "compute_event_condition_contrast":
        proxy_label = _mechanism_source_proxy_label(summary, chinese=chinese)
        if proxy_label:
            return f"正在比较事件期间的{proxy_label}" if chinese else f"Comparing {proxy_label.lower()} during event times"
    if tool_name == "extract_point_timeseries" and variable:
        return f"正在提取{variable}点位时间序列" if chinese else f"Extracting point time series of {variable.lower()}"
    if tool_name in {"extract_regional_mean", "compute_area_weighted_mean", "compute_layer_mean"} and variable:
        if aggregation:
            return f"正在计算{variable}的{aggregation}" if chinese else f"Computing {aggregation.lower()} of {variable.lower()}"
        return f"正在计算{variable}的面积加权平均" if chinese else f"Computing area-weighted mean of {variable.lower()}"
    if tool_name == "compute_spatial_field" and variable:
        if aggregation:
            return f"正在计算{variable}{aggregation}空间场" if chinese else f"Computing spatial field of {variable.lower()} {aggregation.lower()}"
        return f"正在计算{variable}空间场" if chinese else f"Computing spatial field of {variable.lower()}"
    if tool_name == "compute_spatial_vorticity_map":
        return "正在计算涡度空间场" if chinese else "Computing spatial vorticity field"
    if tool_name == "compute_climatology" and variable:
        return f"正在计算{variable}气候态" if chinese else f"Computing climatology of {variable.lower()}"
    if tool_name == "compute_anomaly" and variable:
        return f"正在计算{variable}异常" if chinese else f"Computing anomaly of {variable.lower()}"
    if tool_name == "compute_trend" and variable:
        return f"正在分析{variable}趋势" if chinese else f"Analyzing trend of {variable.lower()}"
    if tool_name == "extract_vertical_profile" and variable:
        return f"正在提取{variable}垂向剖面" if chinese else f"Extracting vertical profile of {variable.lower()}"
    if tool_name == "compute_hovmoller" and variable:
        return f"正在计算{variable}霍夫默勒图" if chinese else f"Computing Hovmoller diagram of {variable.lower()}"
    if tool_name == "compute_derived_field":
        if aggregation:
            return f"正在计算{aggregation}" if chinese else f"Computing {aggregation.lower()}"
        if variable:
            return f"正在计算{variable}派生场" if chinese else f"Computing derived field of {variable.lower()}"

    return _humanize_tool_name(tool_name, chinese=chinese)


def _build_descriptive_label_from_params(
    tool_name: Optional[str],
    params: Dict[str, Any],
    *,
    chinese: bool = False,
) -> str:
    """Build natural language description from tool name and plan params (before result is available)."""
    if not params:
        return _humanize_tool_name(tool_name, chinese=chinese)

    variable = _humanize_data_label(params.get("variable"), chinese=chinese)
    field_type = _humanize_data_label(params.get("field_type"), chinese=chinese)
    aggregation = _humanize_data_label(params.get("aggregation"), chinese=chinese)

    if tool_name == "load_dataset" and variable:
        return f"正在读取{variable}数据" if chinese else f"Loading {variable} data"
    if tool_name == "extract_point_timeseries" and variable:
        return f"正在提取{variable}点位时间序列" if chinese else f"Extracting point time series of {variable.lower()}"
    if tool_name in {"extract_regional_mean", "compute_area_weighted_mean", "compute_layer_mean"} and variable:
        if aggregation:
            return f"正在计算{variable}的{aggregation}" if chinese else f"Computing {aggregation.lower()} of {variable.lower()}"
        return f"正在计算{variable}的面积加权平均" if chinese else f"Computing area-weighted mean of {variable.lower()}"
    if tool_name == "compute_spatial_field" and variable:
        return f"正在计算{variable}空间场" if chinese else f"Computing spatial field of {variable.lower()}"
    if tool_name == "compute_spatial_vorticity_map":
        return "正在计算涡度空间场" if chinese else "Computing spatial vorticity field"
    if tool_name == "compute_event_summary_map":
        summary_mode = str(params.get("summary_mode") or "")
        if summary_mode == "burden":
            return "正在计算事件累积负荷图" if chinese else "Computing event burden map"
        if summary_mode == "event_days":
            return "正在计算事件天数图" if chinese else "Computing event-days map"
    if tool_name == "compute_climatology":
        return "正在计算气候态" if chinese else "Computing climatology"
    if tool_name == "compute_anomaly":
        return "正在计算异常" if chinese else "Computing anomaly"
    if tool_name == "compute_trend" and variable:
        return f"正在分析{variable}趋势" if chinese else f"Analyzing trend of {variable.lower()}"
    if tool_name == "extract_vertical_profile" and variable:
        return f"正在提取{variable}垂向剖面" if chinese else f"Extracting vertical profile of {variable.lower()}"
    if tool_name == "compute_hovmoller" and variable:
        return f"正在计算{variable}霍夫默勒图" if chinese else f"Computing Hovmoller diagram of {variable.lower()}"
    if tool_name == "compute_derived_field":
        if field_type:
            return f"正在计算{field_type}" if chinese else f"Computing {field_type.lower()}"
        if variable:
            return f"正在计算{variable}派生场" if chinese else f"Computing derived field of {variable.lower()}"

    return _humanize_tool_name(tool_name, chinese=chinese)


def _build_headline(summary: Dict[str, Any], *, chinese: bool = False) -> str:
    output_type = summary.get("type")
    if output_type == "data_container_result":
        variable = _data_container_variable_label(summary, chinese=chinese)
        shape = _format_data_container_shape(summary)
        if variable and shape:
            return f"已加载{variable}数据（{shape}）" if chinese else f"Loaded {variable} data ({shape})"
        if variable:
            return f"已加载{variable}数据" if chinese else f"Loaded {variable} data"
        if shape:
            return f"已加载数据（{shape}）" if chinese else f"Loaded data ({shape})"
        return "已加载数据，可用于后续分析" if chinese else "Loaded data for downstream analysis"
    if output_type == "spatial_field_result":
        statistics = summary.get("statistics", {})
        mean_value = statistics.get("mean") if isinstance(statistics, dict) else None
        if mean_value is not None:
            return f"所选区域空间场均值为 {mean_value:.4g}" if chinese else f"Spatial field mean {mean_value:.4g} over the selected region"
        return "所选区域的空间场结果" if chinese else "Spatial field over the selected region"
    if output_type == "trend_result":
        direction = summary.get("trend_direction", "trend")
        slope = summary.get("slope")
        if isinstance(slope, (int, float)) and math.isfinite(float(slope)):
            direction_label = _localize_enum_value(direction, chinese=chinese)
            return f"{direction_label}趋势，斜率为 {slope:.4g}" if chinese else f"{direction.title()} trend with slope {slope:.4g}"
        if slope is not None:
            return "趋势无法计算：有效时间序列点不足或存在缺测" if chinese else "Trend unavailable: insufficient valid time-series points"
        return f"{_localize_enum_value(direction, chinese=chinese)}趋势" if chinese else f"{direction.title()} trend"
    if output_type == "timeseries_result":
        start = summary.get("start_value")
        end = summary.get("end_value")
        if start is not None and end is not None:
            return f"所选时间窗内从 {start:.3g} 变化到 {end:.3g}" if chinese else f"From {start:.3g} to {end:.3g} over the selected window"
        return "时间序列摘要" if chinese else "Time series summary"
    if output_type == "climatology_result":
        period = str(summary.get("period") or "").strip().lower()
        peak_phase = summary.get("peak_phase")
        if isinstance(peak_phase, dict):
            peak_label = _format_climatology_label(peak_phase.get("label"), period)
            peak_value = peak_phase.get("value")
            if peak_label and isinstance(peak_value, (int, float)):
                return f"{period or '气候态'}均值峰值出现在 {peak_label}，数值为 {float(peak_value):.3g}" if chinese else f"Peak {period or 'climatology'} mean occurs in {peak_label} at {float(peak_value):.3g}"
            if peak_label:
                return f"{period or '气候态'}均值峰值出现在 {peak_label}" if chinese else f"Peak {period or 'climatology'} mean occurs in {peak_label}"
        return "所选时间窗的气候态变化" if chinese else "Climatological cycle over the selected window"
    if output_type == "profile_result":
        gradient = summary.get("strongest_gradient")
        if isinstance(gradient, dict):
            depth = gradient.get("depth")
            return f"最强垂向变化出现在约 {depth} 米附近" if chinese else f"Strongest vertical transition near {depth} m"
        return "垂向剖面摘要" if chinese else "Vertical profile summary"
    if output_type == "hovmoller_result":
        return "时空演变矩阵" if chinese else "Space-time evolution matrix"
    if output_type == "regression_map_result":
        significant_fraction = summary.get("significant_fraction")
        slope_extrema = summary.get("slope_extrema")
        if isinstance(significant_fraction, (int, float)):
            return f"有效格点中有 {float(significant_fraction) * 100.0:.1f}% 达到回归显著性" if chinese else f"Regression is significant over {float(significant_fraction) * 100.0:.1f}% of valid cells"
        if isinstance(slope_extrema, dict):
            max_branch = slope_extrema.get("max")
            if isinstance(max_branch, dict) and isinstance(max_branch.get("value"), (int, float)):
                return f"最强正斜率为 {float(max_branch['value']):.4g}" if chinese else f"Strongest positive slope is {float(max_branch['value']):.4g}"
        return "所选区域的空间回归系数分布" if chinese else "Spatial regression coefficients over the selected region"
    if output_type == "spectrum_result":
        global_peak = summary.get("global_peak")
        if isinstance(global_peak, dict):
            period = global_peak.get("period")
            power = global_peak.get("power")
            if isinstance(period, (int, float)) and np.isfinite(period):
                if isinstance(power, (int, float)):
                    return f"主谱峰周期约为 {float(period):.3g}，功率为 {float(power):.3g}" if chinese else f"Dominant spectral peak near period {float(period):.3g} with power {float(power):.3g}"
                return f"主谱峰周期约为 {float(period):.3g}" if chinese else f"Dominant spectral peak near period {float(period):.3g}"
            frequency = global_peak.get("frequency")
            if isinstance(frequency, (int, float)):
                return f"主谱峰频率约为 {float(frequency):.3g}" if chinese else f"Dominant spectral peak near frequency {float(frequency):.3g}"
        return "频域功率谱" if chinese else "Frequency-domain power spectrum"
    if output_type == "ts_diagram_result":
        n_points = summary.get("n_points")
        temp_range = summary.get("temperature_range")
        salt_range = summary.get("salinity_range")
        if (
            isinstance(n_points, (int, float))
            and isinstance(temp_range, (list, tuple))
            and len(temp_range) == 2
            and isinstance(salt_range, (list, tuple))
            and len(salt_range) == 2
        ):
            return (
                f"共 {int(n_points)} 个样本点，温度范围 {float(temp_range[0]):.3g}–{float(temp_range[1]):.3g}，"
                f"盐度范围 {float(salt_range[0]):.3g}–{float(salt_range[1]):.3g}"
                if chinese
                else f"{int(n_points)} sampled points spanning T {float(temp_range[0]):.3g}–{float(temp_range[1]):.3g} and S {float(salt_range[0]):.3g}–{float(salt_range[1]):.3g}"
            )
        if isinstance(n_points, (int, float)):
            return f"共 {int(n_points)} 个温盐样本点" if chinese else f"{int(n_points)} sampled temperature-salinity pairs"
        return "温盐散点分布" if chinese else "Temperature-salinity point cloud"
    if output_type == "watermass_event_association_result":
        top_name = str(summary.get("top_associated_watermass_name") or summary.get("top_associated_watermass") or "").strip()
        hotspot_count = summary.get("hotspot_tile_count")
        valid_count = summary.get("valid_tile_count")
        score = summary.get("association_score")
        strength = str(summary.get("evidence_strength") or "").strip()
        if top_name and isinstance(hotspot_count, (int, float)) and isinstance(valid_count, (int, float)):
            if isinstance(score, (int, float)):
                return (
                    f"{int(hotspot_count)}/{int(valid_count)} 个热点格子最偏向 {top_name}，关联分数为 {_format_metric(score)}"
                    if chinese
                    else f"{int(hotspot_count)}/{int(valid_count)} hotspot tiles are most enriched in {top_name} with association score {_format_metric(score)}"
                )
            return (
                f"{int(hotspot_count)}/{int(valid_count)} 个热点格子最偏向 {top_name}"
                if chinese
                else f"{int(hotspot_count)}/{int(valid_count)} hotspot tiles are most enriched in {top_name}"
            )
        if strength:
            return (
                f"事件热点与水团格网关联的证据等级为 {_localize_enum_value(strength, chinese=chinese)}"
                if chinese
                else f"Tile-level event-watermass linkage is graded as {strength}"
            )
        return "事件热点与水团格网关联摘要" if chinese else "Tile-level event-watermass association summary"
    if output_type == "histogram_result":
        peak = summary.get("peak_bin")
        if isinstance(peak, dict):
            return f"分布峰值位于 {peak.get('center')} 附近" if chinese else f"Peak density near {peak.get('center')}"
        return "分布摘要" if chinese else "Distribution summary"
    if output_type == "eof_result":
        modes = summary.get("leading_modes")
        if isinstance(modes, list) and modes:
            variance = modes[0].get("variance_explained_pct")
            if variance is not None:
                return f"第一模态解释了 {variance:.1f}% 的方差" if chinese else f"Leading mode explains {variance:.1f}% variance"
        return "EOF 模态摘要" if chinese else "EOF mode summary"
    if output_type == "event_detection_result":
        count = summary.get("event_count")
        return f"检测到 {count} 个事件" if count is not None and chinese else (f"{count} detected events" if count is not None else ("事件检测摘要" if chinese else "Event detection summary"))
    if output_type == "event_statistics_result":
        total_count = summary.get("total_count")
        group_by = summary.get("group_by")
        top_group = summary.get("top_group_by_count")
        if isinstance(top_group, dict):
            peak_group = top_group.get("group")
            peak_count = top_group.get("count")
            if peak_group is not None and peak_count is not None and group_by:
                return f"{peak_count} events in the busiest {group_by} group ({peak_group})"
        if total_count is not None and group_by:
            return f"按 {group_by} 分组后共有 {total_count} 个事件" if chinese else f"{total_count} events grouped by {group_by}"
        if total_count is not None:
            return f"所选时间窗内共检测到 {total_count} 个事件" if chinese else f"{total_count} detected events in the selected window"
        return "分组事件统计摘要" if chinese else "Grouped event statistics summary"
    if output_type == "event_comparison_result":
        period_1 = summary.get("period1_label")
        period_2 = summary.get("period2_label")
        changes = summary.get("changes", {})
        if isinstance(changes, dict):
            count_change_pct = changes.get("count_change_percent")
            if (
                isinstance(period_1, str)
                and period_1
                and isinstance(period_2, str)
                and period_2
                and isinstance(count_change_pct, (int, float))
            ):
                direction = "up" if count_change_pct > 0 else "down" if count_change_pct < 0 else "flat"
                if direction == "flat":
                    return f"{period_1} 与 {period_2} 之间事件数基本不变" if chinese else f"Event counts were unchanged between {period_1} and {period_2}"
                return f"从 {period_1} 到 {period_2}，事件数{('上升' if direction == 'up' else '下降')}了 {abs(float(count_change_pct)):.1f}%" if chinese else f"Event counts were {direction} {abs(float(count_change_pct)):.1f}% from {period_1} to {period_2}"
            count_change = changes.get("count_change")
            if isinstance(count_change, (int, float)):
                return f"两个时期之间事件数变化 {_format_metric(count_change)}" if chinese else f"Event count changed by {_format_metric(count_change)} between the two periods"
        return "事件时期对比摘要" if chinese else "Event period comparison summary"
    if output_type == "composite_result":
        sample_counts = summary.get("sample_counts", {})
        if isinstance(sample_counts, dict):
            positive = sample_counts.get("positive")
            negative = sample_counts.get("negative")
            if positive is not None and negative is not None:
                return f"正位相样本数 {positive}，负位相样本数 {negative}" if chinese else f"Positive phase n={positive} versus negative phase n={negative}"
        return "正位相、负位相及差值合成场" if chinese else "Positive, negative, and difference composite fields"
    if output_type == "lag_correlation_result":
        optimal_lag = summary.get("optimal_lag")
        optimal_lag_days = summary.get("optimal_lag_days")
        max_correlation = summary.get("max_correlation")
        analysis_mode = str(summary.get("analysis_mode") or "").strip()
        mode_label = _localize_enum_value(analysis_mode, chinese=chinese) if analysis_mode else None
        if isinstance(optimal_lag, (int, float)) and isinstance(max_correlation, (int, float)):
            if isinstance(optimal_lag_days, (int, float)):
                return (
                    f"{mode_label or '该结果'}中最强领先-滞后耦合出现在滞后 {_format_metric(optimal_lag)} 步（约 {_format_metric(optimal_lag_days)} 天），相关系数为 {_format_metric(max_correlation)}"
                    if chinese
                    else f"Peak lead-lag coupling in the {mode_label or 'current'} series occurs at lag {_format_metric(optimal_lag)} step(s) (~{_format_metric(optimal_lag_days)} days) with correlation {_format_metric(max_correlation)}"
                )
            return (
                f"{mode_label or '该结果'}中最强领先-滞后耦合出现在滞后 {_format_metric(optimal_lag)}，相关系数为 {_format_metric(max_correlation)}"
                if chinese
                else f"Peak lead-lag coupling in the {mode_label or 'current'} series occurs at lag {_format_metric(optimal_lag)} with correlation {_format_metric(max_correlation)}"
            )
        return "领先-滞后相关摘要" if chinese else "Lead-lag correlation summary"
    if output_type == "mechanism_score_result":
        subregion_pattern = _summarize_mechanism_subregion_pattern(summary)
        if subregion_pattern is not None:
            top_mechanism_key = str(subregion_pattern.get("topMechanism") or "").strip()
            top_mechanism = _humanize_mechanism_name(top_mechanism_key, chinese=chinese)
            valid_count = int(subregion_pattern.get("validCount") or 0)
            top_count = int(subregion_pattern.get("topCount") or 0)
            mean_top_score = float(subregion_pattern.get("meanTopScore") or 0.0)
            if subregion_pattern.get("isMostlyBackground") or subregion_pattern.get("isMixed"):
                return (
                    f"子区诊断显示空间不一致，且有 {int(subregion_pattern.get('backgroundLikeCount') or 0)}/{valid_count} 个有效子区接近背景态"
                    if chinese
                    else f"Subregion diagnosis is spatially mixed, with {int(subregion_pattern.get('backgroundLikeCount') or 0)}/{valid_count} valid tiles remaining background-like"
                )
            if top_mechanism and subregion_pattern.get("isConsistent") and mean_top_score >= 0.75:
                return (
                    f"{valid_count}/{valid_count} 个有效子区一致指向{top_mechanism}，证据较强"
                    if chinese
                    else f"All {valid_count} valid tiles consistently favor {top_mechanism} with strong evidence"
                )
            if top_mechanism and top_count > 0:
                return (
                    f"{top_count}/{valid_count} 个有效子区主要指向{top_mechanism}"
                    if chinese
                    else f"{top_count}/{valid_count} valid tiles mainly favor {top_mechanism}"
                )
        top_mechanism = _humanize_mechanism_name(summary.get("top_mechanism"), chinese=chinese)
        claim_strength = summary.get("claim_strength")
        proxy_breakdowns = summary.get("proxy_breakdowns")
        if top_mechanism and claim_strength:
            if isinstance(proxy_breakdowns, list) and proxy_breakdowns:
                return (
                    f"首要机制：{top_mechanism}（{_localize_enum_value(claim_strength, chinese=chinese)}），含子区明细"
                    if chinese
                    else f"Top mechanism: {top_mechanism} ({claim_strength}) with subregion breakdown"
                )
            return f"首要机制：{top_mechanism}（{_localize_enum_value(claim_strength, chinese=chinese)}）" if chinese else f"Top mechanism: {top_mechanism} ({claim_strength})"
        if top_mechanism:
            return f"首要机制：{top_mechanism}" if chinese else f"Top mechanism: {top_mechanism}"
        return "机制排序摘要" if chinese else "Mechanism ranking summary"
    if output_type == "evidence_report_result":
        for key in ("supported_claims", "limited_claims", "untestable_claims"):
            claims = summary.get(key)
            if isinstance(claims, list):
                first = next((str(item).strip() for item in claims if str(item).strip()), "")
                if first:
                    return first
        return "证据报告摘要" if chinese else "Evidence report summary"
    if output_type == "environment_assessment_result":
        verdict = str(summary.get("overall_verdict") or "").strip()
        support = str(summary.get("overall_support_strength") or "").strip()
        if verdict and support:
            return f"总体结论：{_localize_enum_value(verdict, chinese=chinese)}（{_localize_enum_value(support, chinese=chinese)}）" if chinese else f"Overall verdict: {verdict} ({support})"
        if verdict:
            return f"总体结论：{_localize_enum_value(verdict, chinese=chinese)}" if chinese else f"Overall verdict: {verdict}"
        return "环境健康评估摘要" if chinese else "Environment health assessment summary"
    if output_type == "policy_recommendation_result":
        priority = str(summary.get("priority_level") or "").strip()
        if priority:
            return f"政策优先级：{_localize_enum_value(priority, chinese=chinese)}" if chinese else f"Policy priority: {priority}"
        return "政策建议摘要" if chinese else "Policy recommendation summary"
    return "分析结果摘要" if chinese else "Analysis result summary"


def _build_description(summary: Dict[str, Any], *, chinese: bool = False) -> str:
    output_type = summary.get("type")
    if output_type == "data_container_result":
        variable = _data_container_variable_label(summary, chinese=chinese)
        coord_ranges = _data_coord_ranges(summary)
        range_items: List[str] = []
        for key, label in (
            ("time_range", "时间" if chinese else "time"),
            ("depth_range", "深度" if chinese else "depth"),
            ("lat_range", "纬度" if chinese else "lat"),
            ("lon_range", "经度" if chinese else "lon"),
        ):
            formatted = (
                _data_container_depth_label(summary, chinese=chinese)
                if key == "depth_range"
                else _format_data_coord_range(coord_ranges.get(key))
            )
            if formatted:
                range_items.append(f"{label} {formatted}")
        if range_items:
            prefix = variable or ("数据" if chinese else "Data")
            separator = "；" if chinese else "; "
            return (
                f"{prefix}已读取，覆盖范围：{separator.join(range_items)}。"
                if chinese
                else f"{prefix} loaded with coverage: {separator.join(range_items)}."
            )

        shape = _format_data_container_shape(summary)
        if shape:
            prefix = variable or ("数据" if chinese else "Data")
            return (
                f"{prefix}已读取，数据形状为 {shape}。"
                if chinese
                else f"{prefix} loaded with data shape {shape}."
            )
        return "数据已读取，可作为后续步骤的输入。" if chinese else "Data loaded and available as input for later steps."
    if output_type == "climatology_result":
        variable = _humanize_data_label(summary.get("variable"), chinese=chinese)
        period_label = _climatology_period_label(summary.get("period")) or ("气候态" if chinese else "Climatology")
        metadata = summary.get("metadata", {})
        n_years = metadata.get("n_years") if isinstance(metadata, dict) else None
        if variable and isinstance(n_years, (int, float)):
            return f"{variable}的{period_label}基于 {int(n_years)} 个样本年计算得到。" if chinese else f"{period_label} climatology of {variable.lower()} computed across {int(n_years)} sampled year(s)."
        if variable:
            return f"所选分析时间窗内的{variable}{period_label}结果。" if chinese else f"{period_label} climatology of {variable.lower()} across the selected analysis window."
        return f"所选分析时间窗内的{period_label}结果。" if chinese else f"{period_label} climatology across the selected analysis window."
    if output_type == "composite_result":
        variable = _humanize_data_label(summary.get("variable"), chinese=chinese)
        if variable:
            return f"{variable}在正负指数位相下的合成图。" if chinese else f"Composite maps for {variable.lower()} across positive and negative index phases."
        return "正负指数位相下的合成图。" if chinese else "Composite maps across positive and negative index phases."
    if output_type == "regression_map_result":
        variable = _humanize_data_label(summary.get("variable"), chinese=chinese)
        if variable:
            return f"{variable}相对于所选指数的回归斜率图，显著性和相关信息已在卡片中汇总。" if chinese else f"Regression slope map for {variable.lower()} against the selected index; significance and correlation are summarized in the card."
        return "相对于所选指数的回归斜率图，显著性和相关信息已在卡片中汇总。" if chinese else "Regression slope map against the selected index; significance and correlation are summarized in the card."
    if output_type == "spectrum_result":
        variable = _humanize_data_label(summary.get("variable"), chinese=chinese)
        if variable:
            return f"{variable}在采样频率上的功率谱，主峰信息已在卡片中汇总。" if chinese else f"Power spectrum for {variable.lower()} across the sampled frequencies, with dominant peaks summarized in the card."
        return "采样频率上的功率谱，主峰信息已在卡片中汇总。" if chinese else "Power spectrum across the sampled frequencies, with dominant peaks summarized in the card."
    if output_type == "ts_diagram_result":
        color_variable = _humanize_data_label(summary.get("color_variable"), chinese=chinese)
        if color_variable:
            return f"温盐散点展示了采样水团结构，并按 {color_variable} 着色。" if chinese else f"Temperature-salinity scatter showing sampled water-mass structure, colored by {color_variable.lower()}."
        return "温盐散点展示了采样水团结构。" if chinese else "Temperature-salinity scatter showing sampled water-mass structure."
    if output_type == "watermass_event_association_result":
        return (
            "该结果按等分格网汇总事件热点强度、主导水团和热点相对背景分布的偏移程度；配套的热点格网图、主导水团格网图和 T-S 图会展示同一份诊断结果。"
            if chinese
            else "This result aggregates tile-level event intensity, dominant watermass, and the hotspot-versus-background distribution shift. Companion hotspot maps, dominant-watermass maps, and the T-S diagram expose the same diagnosis."
        )
    if output_type == "event_statistics_result":
        group_by = summary.get("group_by")
        n_groups = summary.get("n_groups")
        if group_by and isinstance(n_groups, (int, float)) and n_groups > 0:
            return (
                f"检测到的事件按 {group_by} 分组，并汇总了 {int(n_groups)} 个分组中的数量和强度/面积统计。"
                if chinese
                else f"Summary statistics for detected events, grouped by {group_by}, including count and intensity/area aggregates across {int(n_groups)} groups."
            )
        return "所选分析时间窗内的事件统计摘要。" if chinese else "Summary statistics for detected events across the selected analysis window."
    if output_type == "event_comparison_result":
        period_1 = summary.get("period1_label")
        period_2 = summary.get("period2_label")
        if isinstance(period_1, str) and period_1 and isinstance(period_2, str) and period_2:
            return f"{period_1} 与 {period_2} 之间的事件数和平均强度对比。" if chinese else f"Comparison of detected event counts and mean intensity between {period_1} and {period_2}."
        return "两个所选时期之间的事件数和平均强度对比。" if chinese else "Comparison of detected event counts and mean intensity between two selected periods."
    if output_type == "lag_correlation_result":
        labels = summary.get("labels", {})
        ts1 = _humanize_mechanism_name(labels.get("ts1") if isinstance(labels, dict) else None, chinese=chinese)
        ts2 = _humanize_mechanism_name(labels.get("ts2") if isinstance(labels, dict) else None, chinese=chinese)
        analysis_mode = str(summary.get("analysis_mode") or "").strip()
        mode_label = _localize_enum_value(analysis_mode, chinese=chinese) if analysis_mode else None
        if ts1 and ts2:
            if mode_label:
                return (
                    f"{ts1} 与 {ts2} 在不同滞后下的相关扫描结果，当前显示的是{mode_label}口径。"
                    if chinese
                    else f"Lag-correlation sweep between {ts1.lower()} and {ts2.lower()} across the tested offsets, currently shown for the {mode_label} mode."
                )
            return f"{ts1} 与 {ts2} 在不同滞后下的相关扫描结果。" if chinese else f"Lag-correlation sweep between {ts1.lower()} and {ts2.lower()} across the tested offsets."
        return "不同滞后下的相关扫描结果。" if chinese else "Lag-correlation sweep across the tested offsets."
    if output_type == "mechanism_score_result":
        subregion_pattern = _summarize_mechanism_subregion_pattern(summary)
        if subregion_pattern is not None:
            valid_count = int(subregion_pattern.get("validCount") or 0)
            skipped_count = int(subregion_pattern.get("diagnosis", {}).get("skippedCount") or 0)
            background_like_count = int(subregion_pattern.get("backgroundLikeCount") or 0)
            top_mechanism = _humanize_mechanism_name(subregion_pattern.get("topMechanism"), chinese=chinese)
            mean_top_score = float(subregion_pattern.get("meanTopScore") or 0.0)
            if subregion_pattern.get("isMostlyBackground") or subregion_pattern.get("isMixed"):
                return (
                    f"最终图显示的是 2×2 子区诊断而非原始高分辨率空间场。{valid_count} 个有效子区中，主导机制并不一致，且有 {background_like_count} 个子区更接近宽尺度背景场；这说明证据有限或空间不一致。"
                    if chinese
                    else f"The final map is a 2×2 subregion diagnosis rather than a raw high-resolution field. Across {valid_count} valid tiles, dominant mechanisms are not spatially consistent and {background_like_count} tile(s) stay closer to the broad background field, so the evidence remains limited or spatially uneven."
                )
            if top_mechanism:
                skipped_fragment = (
                    f" 另有 {skipped_count} 个子区因缺少有效海域或样本被跳过。"
                    if chinese and skipped_count > 0
                    else (f" {skipped_count} additional tile(s) were skipped because valid ocean or event/background samples were unavailable." if skipped_count > 0 else "")
                )
                return (
                    f"最终图显示的是 2×2 子区诊断而非原始高分辨率空间场。{valid_count} 个有效子区中，多数子区指向{top_mechanism}，平均顶层标准化对比约为 {_format_metric(mean_top_score)}。{skipped_fragment}"
                    if chinese
                    else f"The final map shows a 2×2 subregion diagnosis instead of a raw high-resolution field. Across {valid_count} valid tiles, most tiles favor {top_mechanism} with a mean top standardized contrast near {_format_metric(mean_top_score)}.{skipped_fragment}"
                )
        n_candidates = len(summary.get("candidate_mechanisms", [])) if isinstance(summary.get("candidate_mechanisms"), list) else 0
        claim_strength = str(summary.get("claim_strength") or "").strip()
        metadata = _mechanism_metadata(summary)
        proxy_label = _mechanism_source_proxy_label(summary, chinese=chinese)
        subregion_breakdown = summary.get("subregion_breakdown")
        proxy_breakdowns = summary.get("proxy_breakdowns")
        prefix = ""
        if metadata.get("comparison") == "event_condition_contrast" and proxy_label:
            prefix = f"{proxy_label}的事件条件对比。 " if chinese else f"Event-conditioned contrast for {proxy_label.lower()}. "
            if isinstance(subregion_breakdown, list) and subregion_breakdown:
                prefix += (
                    f"聚合比较按 {len(subregion_breakdown)} 个子区执行。 "
                    if chinese
                    else f"Aggregation is partitioned across {len(subregion_breakdown)} subregions. "
                )
        elif metadata.get("comparison") == "mesoscale_proxy_ranking" and isinstance(proxy_breakdowns, list) and proxy_breakdowns:
            prefix = (
                f"中尺度代理排序汇总了 {len(proxy_breakdowns)} 条 proxy 证据，并保留子区 breakdown。 "
                if chinese
                else f"Mesoscale proxy ranking aggregates {len(proxy_breakdowns)} proxy evidence items with subregion breakdowns. "
            )
        if n_candidates > 0 and claim_strength:
            return (
                f"{prefix}候选机制基于当前可用的代理诊断结果完成排序，总体支持度评为{_localize_enum_value(claim_strength, chinese=True)}。"
                if chinese
                else f"{prefix}Candidate mechanisms are ranked using the currently available proxy diagnostics. Overall support is graded as {claim_strength}."
            )
        if prefix:
            return f"{prefix}候选机制基于当前可用的代理诊断结果完成排序。" if chinese else f"{prefix}Candidate mechanisms are ranked using the currently available proxy diagnostics."
        return "候选机制基于当前可用的代理诊断结果完成排序。" if chinese else "Candidate mechanisms are ranked using the currently available proxy diagnostics."
    if output_type == "evidence_report_result":
        claim_strength = str(summary.get("claim_strength") or "").strip()
        if claim_strength:
            return (
                f"报告明确区分了支持较强、证据有限和不可检验的结论，总体证据等级为{_localize_enum_value(claim_strength, chinese=True)}。"
                if chinese
                else f"Supported, limited, and untestable claims are separated explicitly. Overall evidence grade is {claim_strength}."
            )
        return "报告明确区分了支持较强、证据有限和不可检验的结论。" if chinese else "Supported, limited, and untestable claims are separated explicitly."
    if output_type == "environment_assessment_result":
        verdict = str(summary.get("overall_verdict") or "").strip()
        support = str(summary.get("overall_support_strength") or "").strip()
        narrative = str(summary.get("overall_narrative") or "").strip()
        if narrative:
            return narrative
        if verdict and support:
            return (
                f"系统将多个环境指标整合为加权证据评估，整体结论为{_localize_enum_value(verdict, chinese=True)}，总体支持度为{_localize_enum_value(support, chinese=True)}。"
                if chinese
                else f"Multiple environmental indicators are combined into an evidence-weighted state assessment. Overall verdict is {verdict.lower()} with an overall support grade of {support}."
            )
        return "系统将多个环境指标整合为加权证据评估。" if chinese else "Multiple environmental indicators are combined into an evidence-weighted state assessment."
    if output_type == "policy_recommendation_result":
        policy_summary = str(summary.get("policy_summary") or "").strip()
        if policy_summary:
            return policy_summary
        priority = str(summary.get("priority_level") or "").strip()
        if priority:
            return (
                f"系统将上游海洋诊断结果转译为政策建议，当前优先级为{_localize_enum_value(priority, chinese=True)}。"
                if chinese
                else f"Upstream ocean diagnostics are translated into policy guidance. Current priority level is {priority}."
            )
        return "系统将上游海洋诊断结果转译为政策建议。" if chinese else "Upstream ocean diagnostics are translated into policy guidance."

    variable = _humanize_data_label(summary.get("variable"), chinese=chinese)
    feature = _humanize_data_label(summary.get("feature"), chinese=chinese)
    aggregation = _humanize_data_label(summary.get("aggregation"), chinese=chinese)
    parts = []
    if variable:
        parts.append(variable)
    if feature:
        parts.append(feature)
    if aggregation:
        parts.append(aggregation)
    if not parts:
        return "来自当前分析结果。" if chinese else "Derived from the current analysis result."
    return "由以下要素计算得到：" + " · ".join(parts) if chinese else "Computed from " + " · ".join(parts)


def _build_metrics(summary: Dict[str, Any], *, chinese: bool = False) -> List[Dict[str, str]]:
    metrics: List[Dict[str, str]] = []
    if summary.get("type") == "data_container_result":
        shape = _format_data_container_shape(summary)
        if shape:
            metrics.append({"label": "shape", "value": shape})
        coord_ranges = _data_coord_ranges(summary)
        time_range = _format_data_coord_range(coord_ranges.get("time_range"))
        if time_range:
            metrics.append({"label": "time", "value": time_range})
        lon_range = _format_data_coord_range(coord_ranges.get("lon_range"))
        lat_range = _format_data_coord_range(coord_ranges.get("lat_range"))
        if lon_range and lat_range:
            metrics.append({"label": "spatial", "value": f"lon {lon_range}; lat {lat_range}"})
        elif lon_range:
            metrics.append({"label": "spatial", "value": f"lon {lon_range}"})
        elif lat_range:
            metrics.append({"label": "spatial", "value": f"lat {lat_range}"})
        depth_range = _data_container_depth_label(summary, chinese=chinese) or _format_data_coord_range(
            coord_ranges.get("depth_range")
        )
        if depth_range:
            metrics.append({"label": "depth", "value": depth_range})
        units = summary.get("units")
        if units:
            metrics.append({"label": "units", "value": str(units)})
        return _localize_metric_entries(metrics[:5], chinese=chinese)
    if summary.get("type") == "climatology_result":
        period_label = _climatology_period_label(summary.get("period"))
        if period_label:
            metrics.append({"label": "period", "value": period_label})
        n_labels = summary.get("n_labels")
        if n_labels is not None and len(metrics) < 3:
            metrics.append({"label": "n_labels", "value": _format_metric(n_labels)})
        amplitude = summary.get("amplitude")
        if amplitude is not None and len(metrics) < 3:
            metrics.append({"label": "amplitude", "value": _format_metric(amplitude)})
        return _localize_metric_entries(metrics[:3], chinese=chinese)
    if summary.get("type") == "composite_result":
        sample_counts = summary.get("sample_counts", {})
        if isinstance(sample_counts, dict):
            positive = sample_counts.get("positive")
            negative = sample_counts.get("negative")
            if positive is not None:
                metrics.append({"label": "positive_n", "value": _format_metric(positive)})
            if negative is not None:
                metrics.append({"label": "negative_n", "value": _format_metric(negative)})
        difference_statistics = summary.get("difference_statistics", {})
        if isinstance(difference_statistics, dict):
            mean_value = difference_statistics.get("mean")
            if mean_value is not None:
                metrics.append({"label": "diff_mean", "value": _format_metric(mean_value)})
        return _localize_metric_entries(metrics[:3], chinese=chinese)
    if summary.get("type") == "regression_map_result":
        significant_fraction = summary.get("significant_fraction")
        if significant_fraction is not None:
            metrics.append({"label": "significant_pct", "value": f"{float(significant_fraction) * 100.0:.1f}%"})
        slope_statistics = summary.get("slope_statistics")
        if isinstance(slope_statistics, dict):
            if slope_statistics.get("min") is not None:
                metrics.append({"label": "slope_min", "value": _format_metric(slope_statistics["min"])})
            if slope_statistics.get("max") is not None and len(metrics) < 3:
                metrics.append({"label": "slope_max", "value": _format_metric(slope_statistics["max"])})
        correlation_statistics = summary.get("correlation_statistics")
        if isinstance(correlation_statistics, dict) and correlation_statistics.get("max") is not None and len(metrics) < 3:
            metrics.append({"label": "corr_max", "value": _format_metric(correlation_statistics["max"])})
        return _localize_metric_entries(metrics[:3], chinese=chinese)
    if summary.get("type") == "spectrum_result":
        global_peak = summary.get("global_peak")
        if isinstance(global_peak, dict):
            period = global_peak.get("period")
            if isinstance(period, (int, float)) and np.isfinite(period):
                metrics.append({"label": "peak_period", "value": _format_metric(period)})
            frequency = global_peak.get("frequency")
            if isinstance(frequency, (int, float)) and len(metrics) < 3:
                metrics.append({"label": "peak_freq", "value": _format_metric(frequency)})
            power = global_peak.get("power")
            if isinstance(power, (int, float)) and len(metrics) < 3:
                metrics.append({"label": "peak_power", "value": _format_metric(power)})
        n_frequencies = summary.get("n_frequencies")
        if n_frequencies is not None and len(metrics) < 3:
            metrics.append({"label": "n_freq", "value": _format_metric(n_frequencies)})
        return _localize_metric_entries(metrics[:3], chinese=chinese)
    if summary.get("type") == "ts_diagram_result":
        n_points = summary.get("n_points")
        if n_points is not None:
            metrics.append({"label": "n_points", "value": _format_metric(n_points)})
        temp_range = summary.get("temperature_range")
        if isinstance(temp_range, (list, tuple)) and len(temp_range) == 2:
            metrics.append({"label": "temp_range", "value": f"{_format_metric(temp_range[0])}–{_format_metric(temp_range[1])}"})
        salt_range = summary.get("salinity_range")
        if isinstance(salt_range, (list, tuple)) and len(salt_range) == 2:
            metrics.append({"label": "salt_range", "value": f"{_format_metric(salt_range[0])}–{_format_metric(salt_range[1])}"})
        return _localize_metric_entries(metrics[:3], chinese=chinese)
    if summary.get("type") == "event_statistics_result":
        total_count = summary.get("total_count")
        if total_count is not None:
            metrics.append({"label": "total_count", "value": _format_metric(total_count)})
        group_by = summary.get("group_by")
        if group_by:
            metrics.append({"label": "group_by", "value": str(group_by)})
        n_groups = summary.get("n_groups")
        if n_groups is not None:
            metrics.append({"label": "n_groups", "value": _format_metric(n_groups)})
        elif isinstance(summary.get("groups"), dict):
            metrics.append({"label": "n_groups", "value": _format_metric(len(summary["groups"]))})

        top_group = summary.get("top_group_by_count")
        if len(metrics) < 3 and isinstance(top_group, dict):
            group_label = top_group.get("group")
            group_count = top_group.get("count")
            if group_label is not None and group_count is not None:
                metrics.append({"label": "top_group", "value": f"{group_label} ({_format_metric(group_count)})"})
        return _localize_metric_entries(metrics[:3], chinese=chinese)
    if summary.get("type") == "event_comparison_result":
        period_1_count = summary.get("period1_total_count")
        if period_1_count is not None:
            metrics.append({"label": "period_1_count", "value": _format_metric(period_1_count)})
        period_2_count = summary.get("period2_total_count")
        if period_2_count is not None:
            metrics.append({"label": "period_2_count", "value": _format_metric(period_2_count)})
        changes = summary.get("changes", {})
        if isinstance(changes, dict):
            count_change_pct = changes.get("count_change_percent")
            if count_change_pct is not None:
                metrics.append({"label": "count_change_pct", "value": f"{_format_metric(count_change_pct)}%"})
            elif changes.get("count_change") is not None:
                metrics.append({"label": "count_change", "value": _format_metric(changes["count_change"])})
            elif changes.get("intensity_change") is not None:
                metrics.append({"label": "intensity_change", "value": _format_metric(changes["intensity_change"])})
        return _localize_metric_entries(metrics[:3], chinese=chinese)
    if summary.get("type") == "lag_correlation_result":
        analysis_mode = summary.get("analysis_mode")
        if analysis_mode:
            metrics.append({"label": "lag_mode", "value": _localize_enum_value(analysis_mode, chinese=chinese)})
        optimal_lag = summary.get("optimal_lag")
        if optimal_lag is not None:
            metrics.append({"label": "optimal_lag", "value": _format_metric(optimal_lag)})
        optimal_lag_days = summary.get("optimal_lag_days")
        if optimal_lag_days is not None:
            metrics.append({"label": "optimal_lag_days", "value": _format_metric(optimal_lag_days)})
        max_correlation = summary.get("max_correlation")
        if max_correlation is not None:
            metrics.append({"label": "max_corr", "value": _format_metric(max_correlation)})
        step_days = summary.get("median_step_days")
        if step_days is not None:
            metrics.append({"label": "lag_step_days", "value": _format_metric(step_days)})
        zero_lag = summary.get("zero_lag_correlation")
        if zero_lag is not None:
            metrics.append({"label": "zero_lag", "value": _format_metric(zero_lag)})
        return _localize_metric_entries(metrics[:4], chinese=chinese)
    if summary.get("type") == "mechanism_score_result":
        claim_strength = summary.get("claim_strength")
        if claim_strength:
            metrics.append({"label": "support", "value": str(claim_strength)})
        n_valid_subregions = summary.get("n_valid_subregions")
        if n_valid_subregions is not None:
            metrics.append({"label": "valid_subregions", "value": _format_metric(n_valid_subregions)})
        n_subregions = summary.get("n_subregions")
        if n_subregions is not None:
            metrics.append({"label": "subregions", "value": _format_metric(n_subregions)})
        candidates = summary.get("candidate_mechanisms")
        if isinstance(candidates, list) and candidates:
            top = candidates[0]
            if isinstance(top, dict) and top.get("score") is not None:
                metrics.append({"label": "top_score", "value": _format_metric(top["score"])})
            metrics.append({"label": "candidates", "value": _format_metric(len(candidates))})
        return _localize_metric_entries(metrics[:4], chinese=chinese)
    if summary.get("type") == "watermass_event_association_result":
        evidence_strength = summary.get("evidence_strength")
        if evidence_strength:
            metrics.append({"label": "support", "value": str(evidence_strength)})
        valid_tiles = summary.get("valid_tile_count")
        if valid_tiles is not None:
            metrics.append({"label": "valid_tiles", "value": _format_metric(valid_tiles)})
        hotspot_tiles = summary.get("hotspot_tile_count")
        if hotspot_tiles is not None:
            metrics.append({"label": "hotspot_tiles", "value": _format_metric(hotspot_tiles)})
        association_score = summary.get("association_score")
        if association_score is not None:
            metrics.append({"label": "assoc_score", "value": _format_metric(association_score)})
        return _localize_metric_entries(metrics[:4], chinese=chinese)
    if summary.get("type") == "evidence_report_result":
        claim_strength = summary.get("claim_strength")
        if claim_strength:
            metrics.append({"label": "support", "value": str(claim_strength)})
        for label, key in (
            ("supported_n", "supported_claims"),
            ("limited_n", "limited_claims"),
            ("untestable_n", "untestable_claims"),
        ):
            items = summary.get(key)
            if isinstance(items, list):
                metrics.append({"label": label, "value": _format_metric(len(items))})
            if len(metrics) >= 3:
                break
        return _localize_metric_entries(metrics[:3], chinese=chinese)
    if summary.get("type") == "environment_assessment_result":
        verdict = summary.get("overall_verdict")
        if verdict:
            metrics.append({"label": "verdict", "value": str(verdict)})
        support = summary.get("overall_support_strength")
        if support:
            metrics.append({"label": "support", "value": str(support)})
        n_branches = summary.get("n_branches")
        if n_branches is not None:
            metrics.append({"label": "branches", "value": _format_metric(n_branches)})
        return _localize_metric_entries(metrics[:3], chinese=chinese)

    statistics = summary.get("statistics")
    extrema = summary.get("extrema")

    if isinstance(statistics, dict):
        for label in ("mean", "std", "min", "max"):
            value = statistics.get(label)
            if value is None:
                continue
            metrics.append({"label": label, "value": _format_metric(value)})
            if len(metrics) >= 3:
                return metrics

    if summary.get("trend_direction"):
        metrics.append({"label": "trend", "value": str(summary["trend_direction"])})
    p_value = summary.get("p_value")
    if isinstance(p_value, (int, float)) and math.isfinite(float(p_value)):
        metrics.append({"label": "p_value", "value": _format_metric(p_value)})
    r_squared = summary.get("r_squared")
    if isinstance(r_squared, (int, float)) and math.isfinite(float(r_squared)):
        metrics.append({"label": "r_squared", "value": _format_metric(r_squared)})
    if summary.get("event_count") is not None:
        metrics.append({"label": "event_count", "value": str(summary["event_count"])})

    if len(metrics) < 3 and isinstance(extrema, dict):
        for label in ("max_value", "min_value"):
            value = extrema.get(label)
            if value is None:
                continue
            metrics.append({"label": label, "value": _format_metric(value)})
            if len(metrics) >= 3:
                break

    return _localize_metric_entries(metrics[:3], chinese=chinese)


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _format_fraction(value: Any) -> str:
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value) * 100.0:.1f}%"


def _humanize_mechanism_name(value: Any, *, chinese: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    if not cleaned:
        return ""
    return _localize_enum_value(cleaned, chinese=chinese)


def _format_policy_action_matrix_item(item: Dict[str, Any], *, chinese: bool = False) -> str:
    priority = str(item.get("priority") or "").strip()
    action_type = str(item.get("action_type") or "").strip()
    target = str(item.get("target") or "").strip()
    action = str(item.get("action") or "").strip()
    evidence_basis = str(item.get("evidence_basis") or item.get("rationale") or "").strip()
    guardrail = str(item.get("guardrail") or "").strip()
    prefix_parts = [part for part in (priority, action_type) if part]
    prefix = f"[{'/'.join(prefix_parts)}] " if prefix_parts else ""
    core = f"{target} — {action}" if target and action else action or target
    if chinese:
        details = []
        if evidence_basis:
            details.append(f"依据：{evidence_basis}")
        if guardrail:
            details.append(f"边界：{guardrail}")
        return prefix + core + ("；" + "；".join(details) if details else "")
    details = []
    if evidence_basis:
        details.append(f"basis: {evidence_basis}")
    if guardrail:
        details.append(f"guardrail: {guardrail}")
    return prefix + core + ("; " + "; ".join(details) if details else "")


def _build_detail_sections(summary: Dict[str, Any], *, chinese: bool = False) -> List[Dict[str, Any]]:
    output_type = summary.get("type")
    if output_type == "mechanism_score_result":
        sections: List[Dict[str, Any]] = []
        candidates = summary.get("candidate_mechanisms")
        if isinstance(candidates, list):
            candidate_items = []
            for item in candidates[:4]:
                if not isinstance(item, dict):
                    continue
                name = _humanize_mechanism_name(item.get("name"), chinese=chinese)
                score = item.get("score")
                if name and isinstance(score, (int, float)):
                    candidate_items.append(f"{name}: {_format_metric(score)}")
            if candidate_items:
                sections.append({"title": "候选机制" if chinese else "Candidate Mechanisms", "items": candidate_items})
        proxy_breakdowns = summary.get("proxy_breakdowns")
        if isinstance(proxy_breakdowns, list) and proxy_breakdowns:
            proxy_items = [
                _format_proxy_breakdown_line(item, chinese=chinese)
                for item in proxy_breakdowns[:4]
                if isinstance(item, dict)
            ]
            if proxy_items:
                sections.append({"title": "子区明细" if chinese else "Subregion Breakdown", "items": proxy_items})
        subregion_breakdown = summary.get("subregion_breakdown")
        if isinstance(subregion_breakdown, list) and subregion_breakdown:
            subregion_items = [
                _format_subregion_breakdown_line(item, chinese=chinese)
                for item in subregion_breakdown[:6]
                if isinstance(item, dict)
            ]
            if subregion_items:
                sections.append({"title": "子区明细" if chinese else "Subregion Breakdown", "items": subregion_items})
        supporting = [str(item).strip() for item in summary.get("supporting_evidence", []) if str(item).strip()]
        conflicting = [str(item).strip() for item in summary.get("conflicting_evidence", []) if str(item).strip()]
        if supporting:
            sections.append({"title": "支持性证据" if chinese else "Supporting Evidence", "items": supporting[:4]})
        if conflicting:
            sections.append({"title": "冲突证据" if chinese else "Conflicting Evidence", "items": conflicting[:3]})
        return sections

    if output_type == "watermass_event_association_result":
        sections: List[Dict[str, Any]] = []
        top_name = str(summary.get("top_associated_watermass_name") or summary.get("top_associated_watermass") or "").strip()
        score = summary.get("association_score")
        hotspot_count = summary.get("hotspot_tile_count")
        valid_count = summary.get("valid_tile_count")
        exact_match_count = summary.get("exact_match_count")
        fallback_count = summary.get("nearest_fallback_count")
        overview_items: List[str] = []
        if top_name:
            overview_items.append(
                f"热点格子最偏向 {top_name}" if chinese else f"Hotspot tiles are most enriched in {top_name}"
            )
        if isinstance(score, (int, float)):
            overview_items.append(
                f"关联分数 {_format_metric(score)}" if chinese else f"Association score {_format_metric(score)}"
            )
        if isinstance(hotspot_count, (int, float)) and isinstance(valid_count, (int, float)):
            overview_items.append(
                f"{int(hotspot_count)}/{int(valid_count)} 个有效格子被标记为热点"
                if chinese
                else f"{int(hotspot_count)}/{int(valid_count)} valid tiles are flagged as hotspots"
            )
        if isinstance(exact_match_count, (int, float)) and isinstance(fallback_count, (int, float)):
            overview_items.append(
                f"严格命中 {int(exact_match_count)} 个样本，最近邻回退 {int(fallback_count)} 个样本"
                if chinese
                else f"Strict-match samples: {int(exact_match_count)}; nearest-fallback samples: {int(fallback_count)}"
            )
        if overview_items:
            sections.append({"title": "关联摘要" if chinese else "Association Summary", "items": overview_items[:4]})

        background_distribution = summary.get("background_distribution")
        if isinstance(background_distribution, dict):
            items = [
                f"{str(key)}: {_format_fraction(value)}"
                for key, value in background_distribution.items()
                if isinstance(value, (int, float))
            ]
            if items:
                sections.append({"title": "背景分布" if chinese else "Background Distribution", "items": items[:6]})

        hotspot_distribution = summary.get("hotspot_distribution")
        if isinstance(hotspot_distribution, dict):
            items = [
                f"{str(key)}: {_format_fraction(value)}"
                for key, value in hotspot_distribution.items()
                if isinstance(value, (int, float))
            ]
            if items:
                sections.append({"title": "热点分布" if chinese else "Hotspot Distribution", "items": items[:6]})
        return sections

    if output_type == "evidence_report_result":
        sections = []
        for title, key, limit in (
            ("Supported Claims", "supported_claims", 4),
            ("Limited Claims", "limited_claims", 4),
            ("Untestable Claims", "untestable_claims", 3),
            ("Residual Or Uncertainty", "residual_or_uncertainty", 4),
        ):
            items = [str(item).strip() for item in summary.get(key, []) if str(item).strip()]
            if items:
                localized_title = {
                    "Supported Claims": "支持性结论",
                    "Limited Claims": "有限支持结论",
                    "Untestable Claims": "不可检验结论",
                    "Residual Or Uncertainty": "残差或不确定性",
                }.get(title, title) if chinese else title
                sections.append({"title": localized_title, "items": items[:limit]})
        return sections
    if output_type == "environment_assessment_result":
        sections = []
        branches = summary.get("branch_assessments")
        if isinstance(branches, list):
            branch_items = []
            for item in branches:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("indicator_label") or item.get("name") or "").strip()
                direction = str(item.get("direction") or "").strip()
                support = str(item.get("support_strength") or "").strip()
                branch_summary = str(item.get("summary") or "").strip()
                if not label:
                    continue
                line = f"{label}: {_localize_enum_value(direction, chinese=chinese)} ({_localize_enum_value(support, chinese=chinese)})"
                if branch_summary:
                    line = f"{line}. {branch_summary}"
                branch_items.append(line)
            if branch_items:
                sections.append({"title": "分支评估" if chinese else "Branch Assessments", "items": branch_items[:6]})
        key_pressures = [str(item).strip() for item in summary.get("key_pressures", []) if str(item).strip()]
        if key_pressures:
            sections.append({"title": "主要压力" if chinese else "Key Pressures", "items": key_pressures[:4]})
        stabilizing = [str(item).strip() for item in summary.get("stabilizing_signals", []) if str(item).strip()]
        if stabilizing:
            sections.append({"title": "稳定或改善信号" if chinese else "Stabilizing Signals", "items": stabilizing[:3]})
        supporting = [str(item).strip() for item in summary.get("supporting_evidence", []) if str(item).strip()]
        uncertainties = [str(item).strip() for item in summary.get("uncertainties", []) if str(item).strip()]
        if supporting:
            sections.append({"title": "支持性证据" if chinese else "Supporting Evidence", "items": supporting[:4]})
        if uncertainties:
            sections.append({"title": "不确定性" if chinese else "Uncertainties", "items": uncertainties[:4]})
        return sections
    if output_type == "policy_recommendation_result":
        sections = []
        actions = summary.get("recommended_actions")
        if isinstance(actions, list):
            action_items = []
            for item in actions:
                if isinstance(item, dict):
                    action = str(item.get("action") or "").strip()
                    if action:
                        action_items.append(_format_policy_action_matrix_item(item, chinese=chinese))
                elif str(item).strip():
                    action_items.append(str(item).strip())
            if action_items:
                sections.append({"title": "政策行动矩阵" if chinese else "Policy Action Matrix", "items": action_items[:8]})
        monitoring = [str(item).strip() for item in summary.get("monitoring_priorities", []) if str(item).strip()]
        if monitoring:
            sections.append({"title": "监测重点" if chinese else "Monitoring Priorities", "items": monitoring[:5]})
        governance = [str(item).strip() for item in summary.get("governance_notes", []) if str(item).strip()]
        if governance:
            sections.append({"title": "治理说明" if chinese else "Governance Notes", "items": governance[:4]})
        constraints = [str(item).strip() for item in summary.get("evidence_constraints", []) if str(item).strip()]
        if constraints:
            sections.append({"title": "证据边界" if chinese else "Evidence Boundaries", "items": constraints[:4]})
        evidence = summary.get("evidence_table")
        if isinstance(evidence, list):
            evidence_items = []
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "").strip()
                signal = str(item.get("signal") or "").strip()
                support = str(item.get("support_strength") or "").strip()
                if label:
                    evidence_items.append(f"{label}: {_localize_enum_value(signal, chinese=chinese)} ({_localize_enum_value(support, chinese=chinese)})")
            if evidence_items:
                sections.append({"title": "证据条目" if chinese else "Evidence Items", "items": evidence_items[:6]})
        return sections

    return []


def _build_card_interpretation(summary: Dict[str, Any], *, chinese: bool = False) -> str:
    output_type = summary.get("type")
    claim_strength = str(summary.get("claim_strength") or "").strip()
    if output_type == "lag_correlation_result":
        selection_mode = str(summary.get("lag_selection_mode") or "").strip().lower()
        selection_reason = str(summary.get("lag_selection_reason") or "").strip()
        mode_labels = {
            "positive": "正滞后候选" if chinese else "the positive-lag candidate",
            "negative": "负滞后候选" if chinese else "the negative-lag candidate",
            "symmetric": "对称原始最优值" if chinese else "the symmetric raw optimum",
        }
        if selection_mode and selection_reason:
            if chinese:
                return f"官方 optimal lag 采用{mode_labels.get(selection_mode, '当前候选')}。{selection_reason}"
            return f"The official optimal lag uses {mode_labels.get(selection_mode, 'the current candidate')}. {selection_reason}"
        if selection_reason:
            return selection_reason
    if output_type == "event_statistics_result":
        group_by = str(summary.get("group_by") or "").strip()
        top_group = summary.get("top_group_by_count")
        top_ranked = summary.get("groups_ranked_by_count")
        label_mode = str(summary.get("group_label_mode") or "").strip()
        if isinstance(top_group, dict):
            group_label = top_group.get("group")
            count = top_group.get("count")
            share = top_group.get("share_of_total_count_pct")
            count_gap = summary.get("count_gap_top_vs_second")
            if group_label is not None and isinstance(count, (int, float)):
                if chinese:
                    line = f"事件活动主要集中在 {group_by or '首要分组'} {group_label}，共 {int(count)} 个事件"
                    if isinstance(share, (int, float)):
                        line += f"（占总事件数 {float(share):.1f}%）"
                    if isinstance(count_gap, (int, float)):
                        line += f"，比第二位分组多 {float(count_gap):.0f} 个事件"
                    line += "。"
                    if label_mode == "index_proxy":
                        line += " 分组标签是顺序代理分箱，不应直接解释为具体月份或季节。"
                    return line
                line = f"Event activity is concentrated in the leading {group_by or 'group'} {group_label} with {int(count)} events"
                if isinstance(share, (int, float)):
                    line += f" ({float(share):.1f}% of the total)"
                if isinstance(count_gap, (int, float)):
                    line += f", exceeding the second-ranked group by {float(count_gap):.0f} events"
                line += "."
                if label_mode == "index_proxy":
                    line += " Group labels are ordered proxy bins and should not be read as explicit calendar labels."
                return line
        if isinstance(top_ranked, list) and top_ranked:
            if chinese:
                return "事件统计展示了不同分组之间的活跃度差异，可用于判断活动是否集中在少数高发分组。"
            return "The grouped event statistics show how strongly activity is concentrated across the ranked groups."
    if output_type == "mechanism_score_result" and claim_strength:
        metadata = _mechanism_metadata(summary)
        proxy_label = _mechanism_source_proxy_label(summary, chinese=chinese)
        if metadata.get("comparison") == "event_condition_contrast" and proxy_label:
            return (
                f"应将这项{proxy_label}事件对比结果视为{_localize_enum_value(claim_strength, chinese=True)}的证据；它仍属于代理型解释，而不是完整的因果证明。"
                if chinese
                else f"Treat this {proxy_label.lower()} event-contrast ranking as {claim_strength}; it is still a proxy-based explanation, not a full causal proof."
            )
        return (
            f"应将这项机制排序视为{_localize_enum_value(claim_strength, chinese=True)}的证据；它仍属于代理型解释，而不是完整的因果证明。"
            if chinese
            else f"Treat this mechanism ranking as {claim_strength}; it is still a proxy-based explanation, not a full causal proof."
        )
    if output_type == "watermass_event_association_result":
        strength = str(summary.get("evidence_strength") or "").strip()
        if strength:
            return (
                f"应将这项事件-水团格网关联视为{_localize_enum_value(strength, chinese=True)}证据；它说明热点格子的主导水团分布是否偏离背景格网分布，但不单独证明因果关系。"
                if chinese
                else f"Treat this event-watermass tile association as {strength}-grade evidence. It shows whether hotspot-tile watermass composition departs from the background tile distribution, but it does not by itself establish causality."
            )
    if output_type == "evidence_report_result" and claim_strength:
        return (
            f"应将这份证据报告视为{_localize_enum_value(claim_strength, chinese=True)}的证据；代理反事实和部分收支结果不应被解读为完整因果证明。"
            if chinese
            else f"Treat this evidence report as {claim_strength}; proxy counterfactuals and partial budgets should not be read as full causal proof."
        )
    if output_type == "environment_assessment_result":
        support = str(summary.get("overall_support_strength") or "").strip()
        verdict = str(summary.get("overall_verdict") or "").strip().lower()
        if support:
            return (
                f"应将这项环境健康评估视为用于政策优先级判断的{_localize_enum_value(support, chinese=True)}证据；它是对{_localize_enum_value(verdict or 'environmental conditions', chinese=True)}的加权状态诊断，而不是完整的因果归因或政策设计。"
                if chinese
                else f"Treat this environment-health assessment as {support}-grade evidence for policy prioritization; it is an evidence-weighted state diagnosis of {verdict or 'environmental conditions'}, not a full causal attribution or a complete policy design."
            )
    if output_type == "policy_recommendation_result":
        priority = str(summary.get("priority_level") or "").strip()
        if priority:
            return (
                f"应将这份政策建议视为{_localize_enum_value(priority, chinese=True)}优先级的决策支持；它依赖上游诊断证据，不等同于完整法规文本或成本收益评估。"
                if chinese
                else f"Treat this policy recommendation report as {priority}-priority decision support. It depends on upstream diagnostic evidence and is not a full regulatory design or cost-benefit assessment."
            )
    return ""


def _build_pending_step_card(plan_step: Dict[str, Any], *, chinese: bool = False) -> Dict[str, Any]:
    sid = str(plan_step.get("step_id", ""))
    tool = plan_step.get("tool", "")
    plan_params = plan_step.get("params", {}) if isinstance(plan_step.get("params"), dict) else {}
    param_summary = _build_descriptive_label_from_params(tool, plan_params, chinese=chinese)
    return {
        "step_id": sid,
        "human_label": str(plan_step.get("human_label") or _humanize_tool_name(tool, chinese=chinese)),
        "technical_label": str(plan_step.get("technical_label") or param_summary),
        "status": "pending",
        "results_hidden_by_default": True,
        "results": [],
        "interpretation": "",
        "actions": [],
        "is_map_bound": False,
        "is_expanded": False,
    }


def _reseed_step_cards_from_plan(
    plan: Optional[Dict[str, Any]],
    step_cards_by_id: Dict[str, Dict[str, Any]],
    step_order: List[str],
    *,
    chinese: bool = False,
    preserve_completed: bool,
) -> List[Dict[str, Any]]:
    next_cards_by_id: Dict[str, Dict[str, Any]] = {}
    next_step_order: List[str] = []

    if preserve_completed:
        for step_id in step_order:
            existing = step_cards_by_id.get(step_id)
            if isinstance(existing, dict) and existing.get("status") == "completed":
                next_cards_by_id[step_id] = existing
                next_step_order.append(step_id)

    for plan_step in (plan or {}).get("steps", []):
        if not isinstance(plan_step, dict):
            continue
        card = _build_pending_step_card(plan_step, chinese=chinese)
        step_id = card["step_id"]
        if not step_id or step_id in next_cards_by_id:
            continue
        next_cards_by_id[step_id] = card
        next_step_order.append(step_id)

    step_cards_by_id.clear()
    step_cards_by_id.update(next_cards_by_id)
    step_order[:] = next_step_order
    return _ordered_step_cards(step_cards_by_id, step_order)


def _empty_workspace_data() -> Dict[str, Any]:
    return {
        "referenceSeries": [],
        "resultSeries": [],
        "anomalySeries": [],
        "timeseriesDisplayInfo": None,
        "tsDiagramPoints": [],
        "tsDiagramTemperatureLabel": "Temperature",
        "tsDiagramSalinityLabel": "Salinity",
        "tsDiagramColorLabel": None,
        "tsDiagramColorRange": None,
        "tsDiagramPointClasses": [],
        "tsDiagramClassColorMap": {},
        "tsDiagramWatermassBins": [],
        "profileSeries": [],
        "profileMarkers": [],
        "hovmollerRows": [],
        "hovmollerTimeLabels": [],
        "hovmollerDisplayInfo": None,
        "hovmollerDepthIntegratedSeries": [],
        "sectionRows": [],
        "sectionDistanceKm": [],
        "sectionAxisTitle": "",
        "sectionSliceLabel": "",
        "sectionDisplayInfo": None,
        "overlaySeries": [],
        "histogramBins": [],
        "eofVariance": [],
        "eofPcSeries": [],
        "eofModes": [],
        "compositeFields": [],
        "mapField": None,
        "eventOverlays": [],
    }


def _run_manual_visualization(request: VisualizationRequest) -> tuple[Dict[str, Any], Dict[str, Any]]:
    lon_range = (request.region.lon_min, request.region.lon_max)
    lat_range = (request.region.lat_min, request.region.lat_max)

    if request.depth_mode == "fixed":
        source_field = load_dataset(
            variable=request.variable,
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=request.time_range,
        )
        depth_aggregation = _pick_depth_aggregation(request.depth_range)
        spatial_field = _run_tool_partition_aware(
            "compute_spatial_field",
            compute_spatial_field,
            data=source_field,
            time_aggregation="mean",
            depth_range=request.depth_range,
            depth_aggregation=depth_aggregation,
        )
        reference_series = _build_manual_reference_series(
            data=source_field,
            lon_range=lon_range,
            lat_range=lat_range,
            depth_range=request.depth_range,
            depth_aggregation=depth_aggregation,
        )
        title = f"{request.variable.title()} Map"
        description = (
            f"Direct manual visualization for {request.variable} over "
            f"{request.region.lon_min:.2f}-{request.region.lon_max:.2f}E and "
            f"{request.region.lat_min:.2f}-{request.region.lat_max:.2f}N."
        )
        summary = {
            "type": "spatial_field_result",
            "variable": request.variable,
            "aggregation": depth_aggregation,
            "statistics": spatial_field.get("metadata", {}).get("statistics", {}),
        }
    elif request.depth_mode == "feature":
        source_field = _load_feature_field(
            dataset=request.dataset,
            feature=request.feature,
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=request.time_range,
            search_depth_range=request.search_depth_range,
        )
        spatial_field = _run_tool_partition_aware(
            "compute_spatial_field",
            compute_spatial_field,
            data=source_field,
            time_aggregation="mean",
        )
        reference_series = _build_manual_reference_series(
            data=source_field,
            lon_range=lon_range,
            lat_range=lat_range,
        )
        feature_label = request.feature.replace("_", " ").title()
        title = f"{feature_label} Depth"
        description = f"Direct depth-field visualization for the diagnosed {request.feature}."
        summary = {
            "type": "spatial_field_result",
            "variable": request.feature,
            "feature": request.feature,
            "aggregation": "depth_field",
            "statistics": spatial_field.get("metadata", {}).get("statistics", {}),
        }
    else:
        source_field = load_dataset(
            variable=request.variable,
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=request.time_range,
            depth_range=request.search_depth_range,
        )
        layer_mean_field, upper_source, lower_source = _build_layer_mean_field(
            dataset=request.dataset,
            variable=request.variable,
            source_field=source_field,
            layer_mean_label=request.layer_mean_label,
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=request.time_range,
            search_depth_range=request.search_depth_range,
        )
        spatial_field = _run_tool_partition_aware(
            "compute_spatial_field",
            compute_spatial_field,
            data=layer_mean_field,
            time_aggregation="mean",
        )
        reference_series = _build_manual_reference_series(
            data=layer_mean_field,
            lon_range=lon_range,
            lat_range=lat_range,
        )
        title = f"{request.variable.title()} Layer Mean"
        description = f"Direct layer-mean visualization for {request.layer_mean_label}."
        summary = {
            "type": "spatial_field_result",
            "variable": request.variable,
            "aggregation": "layer_mean",
            "statistics": spatial_field.get("metadata", {}).get("statistics", {}),
            "upper_bound_source": upper_source,
            "lower_bound_source": lower_source,
        }

    workspace_data = _empty_workspace_data()
    workspace_data["referenceSeries"] = reference_series
    workspace_data["resultSeries"] = list(reference_series)
    workspace_data["anomalySeries"] = list(reference_series)
    workspace_data["mapField"] = _build_map_field_payload(
        spatial_field=spatial_field,
        title=title,
        time_range=request.time_range,
    )

    result_card = _build_result_card("manual_visualization", summary)
    result_card["title"] = title
    result_card["headline"] = _build_headline(summary)
    result_card["description"] = description
    result_card["metrics"] = _build_metrics(summary)

    return workspace_data, result_card


def _pick_depth_aggregation(depth_range: tuple[float, float]) -> str:
    if all(abs(value) < 1e-6 for value in depth_range):
        return "surface"
    return "mean"


def _build_manual_reference_series(
    data: xr.DataArray,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    depth_range: Optional[tuple[float, float]] = None,
    depth_aggregation: str = "mean",
) -> List[Dict[str, Any]]:
    if "time" not in data.dims:
        return []

    result = _run_tool_partition_aware(
        "extract_regional_mean",
        extract_regional_mean,
        data=data,
        lon_range=lon_range,
        lat_range=lat_range,
        depth_range=depth_range,
        depth_aggregation=depth_aggregation,
    )
    return _build_named_series(
        labels=result.get("times", []),
        values=result.get("values", []),
        label_key="label",
        value_key="value",
        limit=18,
    )


def _load_feature_field(
    dataset: str,
    feature: str,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    time_range: tuple[str, str],
    search_depth_range: tuple[float, float],
) -> xr.DataArray:
    if feature == "thermocline":
        temp = load_dataset(
            dataset=dataset,
            variable="temp",
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=time_range,
            depth_range=search_depth_range,
        )
        return _run_tool_partition_aware(
            "identify_thermocline_depth",
            identify_thermocline_depth,
            temp=temp,
        )

    temp = load_dataset(
        dataset=dataset,
        variable="temp",
        lon_range=lon_range,
        lat_range=lat_range,
        time_range=time_range,
        depth_range=search_depth_range,
    )
    salt = load_dataset(
        dataset=dataset,
        variable="salt",
        lon_range=lon_range,
        lat_range=lat_range,
        time_range=time_range,
        depth_range=search_depth_range,
    )
    dataset_field = _run_tool_partition_aware(
        "assemble_dataset",
        assemble_dataset,
        variables={"temp": temp, "salt": salt},
    )
    density = _run_tool_partition_aware("compute_density", compute_density, data=dataset_field)
    if feature == "mixed_layer":
        return _run_tool_partition_aware(
            "identify_mixed_layer_depth",
            identify_mixed_layer_depth,
            density=density,
        )
    if feature == "pycnocline":
        return _run_tool_partition_aware(
            "identify_pycnocline_depth",
            identify_pycnocline_depth,
            density=density,
        )
    raise ValueError(f"Unsupported feature: {feature}")


def _build_layer_mean_field(
    dataset: str,
    variable: str,
    source_field: xr.DataArray,
    layer_mean_label: str,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
    time_range: tuple[str, str],
    search_depth_range: tuple[float, float],
) -> tuple[xr.DataArray, str, str]:
    upper_spec, lower_spec = _parse_layer_mean_label(layer_mean_label)
    feature_cache: Dict[str, xr.DataArray] = {}

    def resolve_feature(feature_name: str) -> xr.DataArray:
        if feature_name not in feature_cache:
            feature_cache[feature_name] = _load_feature_field(
                dataset=dataset,
                feature=feature_name,
                lon_range=lon_range,
                lat_range=lat_range,
                time_range=time_range,
                search_depth_range=search_depth_range,
            )
        return feature_cache[feature_name]

    layer_kwargs: Dict[str, Any] = {"data": source_field}
    upper_source = "surface"
    lower_source = "unknown"

    if upper_spec["kind"] == "fixed_depth":
        layer_kwargs["upper_bound_value"] = upper_spec["value"]
        upper_source = f"fixed:{upper_spec['value']}"
    else:
        upper_feature = str(upper_spec["feature"])
        layer_kwargs["upper_bound_field"] = resolve_feature(upper_feature)
        upper_source = upper_feature

    if lower_spec["kind"] == "fixed_depth":
        layer_kwargs["lower_bound_value"] = lower_spec["value"]
        lower_source = f"fixed:{lower_spec['value']}"
    else:
        lower_feature = str(lower_spec["feature"])
        layer_kwargs["lower_bound_field"] = resolve_feature(lower_feature)
        lower_source = lower_feature

    layer_mean_field = _run_tool_partition_aware("compute_layer_mean", compute_layer_mean, **layer_kwargs)
    layer_mean_field = _with_layer_mean_metadata(
        layer_mean_field,
        variable=variable,
        upper_source=upper_source,
        lower_source=lower_source,
    )
    return layer_mean_field, upper_source, lower_source


def _with_layer_mean_metadata(
    field: xr.DataArray,
    *,
    variable: str,
    upper_source: str,
    lower_source: str,
):
    def _apply(part: xr.DataArray) -> xr.DataArray:
        part = part.copy()
        part.attrs = {
            **part.attrs,
            "aggregation": "layer_mean",
            "upper_bound_source": upper_source,
            "lower_bound_source": lower_source,
        }
        part.name = f"{variable}_layer_mean"
        return part

    if isinstance(field, PartitionedDataArray):
        return PartitionedDataArray(
            tuple(_apply(partition) for partition in field.partitions),
            partition_labels=field.partition_labels,
        )
    return _apply(field)


def _parse_layer_mean_label(layer_mean_label: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    normalized = layer_mean_label.strip().lower()
    if "->" not in normalized:
        raise ValueError(f"Unsupported layer mean label: {layer_mean_label}")

    upper_text, lower_text = [part.strip() for part in normalized.split("->", 1)]

    def parse_boundary(text: str) -> Dict[str, Any]:
        if text == "surface":
            return {"kind": "fixed_depth", "value": 0.0}
        if text in {"mixed_layer", "thermocline", "pycnocline"}:
            return {"kind": "feature", "feature": text}
        raise ValueError(f"Unsupported layer boundary: {text}")

    return parse_boundary(upper_text), parse_boundary(lower_text)


def _build_map_field_payload(
    spatial_field: Dict[str, Any],
    title: str,
    time_range: tuple[str, str],
    max_lon_points: int = 100,
    max_lat_points: int = 100,
) -> Dict[str, Any]:
    lon = as_numeric_array(spatial_field.get("lon", []))
    lat = as_numeric_array(spatial_field.get("lat", []))
    values = as_numeric_array(spatial_field.get("values", []))

    if lon.size == 0 or lat.size == 0 or values.size == 0 or values.ndim != 2:
        return {
            "lon": [],
            "lat": [],
            "values": [],
            "label": title,
            "variable": spatial_field.get("metadata", {}).get("variable", "unknown"),
            "units": spatial_field.get("metadata", {}).get("units", ""),
            "statistics": {},
            "timeLabel": _format_time_label(time_range),
        }

    required_lat_indices, required_lon_indices = _map_extreme_sample_indices(values)
    lon_indices = _sample_indices_with_required(
        len(lon),
        limit=min(len(lon), max_lon_points),
        required_indices=required_lon_indices,
    )
    lat_indices = _sample_indices_with_required(
        len(lat),
        limit=min(len(lat), max_lat_points),
        required_indices=required_lat_indices,
    )
    sampled_values = values[np.ix_(lat_indices, lon_indices)]

    sampled_lon = [float(lon[i]) for i in lon_indices]
    sampled_lat = [float(lat[i]) for i in lat_indices]
    sampled_list = sampled_values.tolist()

    metadata = spatial_field.get("metadata", {}) or {}
    discrete_legend = spatial_field.get("discrete_legend") or metadata.get("discrete_legend")
    has_discrete_legend = isinstance(discrete_legend, list) and len(discrete_legend) > 0
    subregion_grid = spatial_field.get("subregion_grid") or metadata.get("subregion_grid")
    has_subregion_grid = isinstance(subregion_grid, dict)
    tile_map_kind = str(metadata.get("tile_map_kind") or spatial_field.get("tile_map_kind") or "").strip().lower()
    is_event_hotspot_tile_map = has_subregion_grid and tile_map_kind == "event_hotspot"
    is_categorical_tile_map = has_subregion_grid and not is_event_hotspot_tile_map
    is_transport_streamfunction = str(metadata.get("variable") or "").strip().lower() == "transport_streamfunction"
    has_finite_values = bool(sampled_values.size and np.any(np.isfinite(sampled_values)))
    color_scale = None if (
        is_categorical_tile_map
        or (has_discrete_legend and not is_event_hotspot_tile_map)
        or not has_finite_values
    ) else _build_map_color_scale(
        values,
        label=str(_humanize_data_label(metadata.get("variable")) or title),
        units=str(metadata.get("units") or ""),
    )
    disable_contour_preview = bool(
        metadata.get("ocean_mask_applied") or metadata.get("disable_contour_preview")
    ) or is_categorical_tile_map or is_transport_streamfunction

    contour_b64 = None
    if has_finite_values and not disable_contour_preview:
        try:
            contour_b64 = render_contourf_image(
                lon=sampled_lon,
                lat=sampled_lat,
                values=sampled_list,
                variable=metadata.get("variable", ""),
                units=metadata.get("units", ""),
                colormap=str(color_scale.get("colormap") or "ocean_diverging") if color_scale else "ocean_diverging",
                vmin=color_scale.get("min") if color_scale else None,
                vmax=color_scale.get("max") if color_scale else None,
            )
        except Exception:
            contour_b64 = None

    depth_range = spatial_field.get("metadata", {}).get("depth_range")
    depth_aggregation = spatial_field.get("metadata", {}).get("depth_aggregation")
    if depth_range is not None:
        depth_label = f"{depth_range[0]} to {depth_range[1]} m"
    elif depth_aggregation:
        depth_label = str(depth_aggregation)
    else:
        depth_label = None

    result: Dict[str, Any] = {
        "lon": sampled_lon,
        "lat": sampled_lat,
        "values": sampled_list,
        "label": title,
        "variable": metadata.get("variable", "unknown"),
        "units": metadata.get("units", ""),
        "statistics": metadata.get("statistics", {}),
        "depthLabel": depth_label,
        "timeLabel": _format_time_label(time_range),
        "valueShape": [int(dim) for dim in values.shape],
        "sampledShape": [int(dim) for dim in sampled_values.shape],
        "bounds": [
            [float(lat[lat_indices[0]]), float(lon[lon_indices[0]])],
            [float(lat[lat_indices[-1]]), float(lon[lon_indices[-1]])],
        ],
    }
    if color_scale is not None:
        result["colorScale"] = color_scale
    regional_color_scales = metadata.get("regional_color_scales")
    transport_rendering = _sample_transport_rendering_payload(
        metadata.get("transport_rendering"),
        lat_indices=lat_indices,
        lon_indices=lon_indices,
        full_shape=values.shape,
    )
    if (
        contour_b64 is None
        and is_transport_streamfunction
        and transport_rendering is not None
        and isinstance(regional_color_scales, list)
    ):
        contour_b64 = _render_transport_multiregion_contourf(
            sampled_lon=sampled_lon,
            sampled_lat=sampled_lat,
            sampled_values=sampled_values,
            transport_rendering=transport_rendering,
            regional_color_scales=regional_color_scales,
        )

    if isinstance(regional_color_scales, list) and regional_color_scales:
        result["regionalColorScales"] = _json_safe(regional_color_scales)
    if transport_rendering is not None:
        result["transportRendering"] = transport_rendering
    if tile_map_kind:
        result["tileMapKind"] = tile_map_kind
    if has_subregion_grid:
        result["subregionGrid"] = _json_safe(subregion_grid)
    if isinstance(discrete_legend, list):
        result["discreteLegend"] = _json_safe(discrete_legend)
    if contour_b64:
        result["contourImage"] = contour_b64
    return result


def _render_transport_multiregion_contourf(
    *,
    sampled_lon: List[float],
    sampled_lat: List[float],
    sampled_values: np.ndarray,
    transport_rendering: Dict[str, Any],
    regional_color_scales: List[Any],
) -> Optional[str]:
    filled_regions = transport_rendering.get("filledRegions")
    if not isinstance(filled_regions, list) or not filled_regions:
        return None

    regions: List[Dict[str, Any]] = []
    for region in filled_regions:
        if not isinstance(region, dict):
            continue
        mask = region.get("mask")
        scale = _find_transport_region_scale(region, regional_color_scales)
        if not isinstance(scale, dict):
            continue
        regions.append({
            "mask": mask,
            "vmin": scale.get("min"),
            "vmax": scale.get("max"),
            "colormap": scale.get("colormap") or "ocean_diverging",
        })
    if not regions:
        return None

    try:
        rendered = render_multiregion_contourf_image(
            lon=sampled_lon,
            lat=sampled_lat,
            values=sampled_values,
            regions=regions,
            n_levels=22,
        )
    except Exception:
        return None
    return rendered or None


def _find_transport_region_scale(region: Dict[str, Any], scales: List[Any]) -> Optional[Dict[str, Any]]:
    strategy = str(region.get("scaleStrategy") or region.get("scale_strategy") or "").strip().lower()
    region_id = str(region.get("id") or region.get("region") or "").strip().lower()

    def matches(scale: Dict[str, Any]) -> bool:
        scale_strategy = str(scale.get("scaleStrategy") or scale.get("scale_strategy") or "").strip().lower()
        label = str(scale.get("label") or "").strip().lower()
        if strategy and scale_strategy == strategy:
            return True
        if region_id == "wpo":
            return "wpo" in scale_strategy or "wpo" in label or "global" in label
        if region_id in {"china_seas", "china seas", "cs"}:
            return (
                "china_seas" in scale_strategy
                or "china seas" in label
                or "cs regional" in label
            )
        return bool(region_id and (region_id in scale_strategy or region_id in label))

    for scale in scales:
        if isinstance(scale, dict) and matches(scale):
            return scale
    return None


def _sample_transport_rendering_payload(
    rendering: Any,
    *,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    full_shape: tuple[int, ...],
) -> Optional[Dict[str, Any]]:
    if not isinstance(rendering, dict) or len(full_shape) != 2:
        return None

    filled_mask = _sample_rendering_mask(rendering.get("filled_mask"), lat_indices, lon_indices, full_shape)
    contour_mask = _sample_rendering_mask(rendering.get("contour_mask"), lat_indices, lon_indices, full_shape)
    filled_regions = _sample_transport_filled_regions(
        rendering.get("filled_regions"),
        lat_indices=lat_indices,
        lon_indices=lon_indices,
        full_shape=full_shape,
    )
    if filled_mask is None and contour_mask is None and not filled_regions:
        return None

    levels = rendering.get("contour_levels")
    contour_levels: List[float] = []
    if isinstance(levels, list):
        contour_levels = [float(value) for value in levels if _is_finite_number(value)]

    return _json_safe({
        "mode": rendering.get("mode"),
        "filledRegion": rendering.get("filled_region"),
        "contourRegion": rendering.get("contour_region"),
        "filledMask": filled_mask,
        "contourMask": contour_mask,
        "contourLevels": contour_levels,
        "filledRegions": filled_regions,
        "filledColormap": rendering.get("filled_colormap"),
        "contourColor": rendering.get("contour_color"),
        "zeroContourColor": rendering.get("zero_contour_color"),
    })


def _sample_transport_filled_regions(
    regions: Any,
    *,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    full_shape: tuple[int, ...],
) -> List[Dict[str, Any]]:
    if not isinstance(regions, list):
        return []
    sampled_regions: List[Dict[str, Any]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        sampled_mask = _sample_rendering_mask(region.get("mask"), lat_indices, lon_indices, full_shape)
        if sampled_mask is None:
            continue
        sampled_regions.append({
            "id": region.get("id"),
            "region": region.get("region"),
            "label": region.get("label"),
            "scaleStrategy": region.get("scaleStrategy") or region.get("scale_strategy"),
            "mask": sampled_mask,
        })
    return sampled_regions


def _sample_rendering_mask(
    mask: Any,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
    full_shape: tuple[int, ...],
) -> Optional[List[List[bool]]]:
    try:
        mask_array = np.asarray(mask, dtype=bool)
    except Exception:
        return None
    if tuple(mask_array.shape) != tuple(full_shape):
        return None
    return mask_array[np.ix_(lat_indices, lon_indices)].tolist()


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _xarray_time_range(data: Any) -> tuple[str, str]:
    if isinstance(data, PartitionedDataArray):
        first_partition = data.partitions[0]
        last_partition = data.partitions[-1]
        if "time" not in first_partition.coords or "time" not in last_partition.coords:
            return "", ""
        first_values = np.asarray(first_partition["time"].values)
        last_values = np.asarray(last_partition["time"].values)
        if first_values.size == 0 or last_values.size == 0:
            return "", ""
        return str(first_values[0]), str(last_values[-1])

    if hasattr(data, "coords") and "time" in data.coords:
        time_values = np.asarray(data["time"].values)
        if time_values.size:
            return str(time_values[0]), str(time_values[-1])
    return "", ""


def _mean_xarray_like(data: Any, dims: Sequence[str]) -> xr.DataArray:
    reduce_dims = [dim for dim in dims if dim in getattr(data, "dims", ())]
    if not reduce_dims:
        return data
    if isinstance(data, PartitionedDataArray):
        return _mean_partitioned_dataarray(data, reduce_dims)
    return data.mean(dim=reduce_dims, skipna=True)


def _mean_partitioned_dataarray(data: PartitionedDataArray, dims: Sequence[str]) -> xr.DataArray:
    reduce_dims = [dim for dim in dims if dim in data.dims]
    if not reduce_dims:
        if "time" in data.dims:
            return xr.concat(data.partitions, dim="time", coords="minimal", compat="override")
        return data.partitions[0]

    preserves_time = "time" in data.dims and "time" not in reduce_dims
    if preserves_time:
        reduced_parts = [
            partition.mean(
                dim=[dim for dim in reduce_dims if dim in partition.dims],
                skipna=True,
            )
            for partition in data.partitions
        ]
        collapsed = xr.concat(reduced_parts, dim="time", coords="minimal", compat="override")
        collapsed.name = data.name
        collapsed.attrs = dict(data.attrs)
        return collapsed

    total_sum: Optional[xr.DataArray] = None
    total_count: Optional[xr.DataArray] = None
    for partition in data.partitions:
        partition_dims = [dim for dim in reduce_dims if dim in partition.dims]
        if not partition_dims:
            current_sum = partition
            current_count = xr.ones_like(partition)
        else:
            current_sum = partition.sum(dim=partition_dims, skipna=True)
            current_count = partition.count(dim=partition_dims)

        if total_sum is None or total_count is None:
            total_sum = current_sum
            total_count = current_count
            continue

        total_sum, current_sum = xr.align(total_sum, current_sum, join="inner")
        total_count, current_count = xr.align(total_count, current_count, join="inner")
        total_sum = total_sum + current_sum
        total_count = total_count + current_count

    if total_sum is None or total_count is None:
        raise ValueError("Cannot compute mean for empty partitioned data")

    collapsed = xr.where(total_count > 0, total_sum / total_count, np.nan)
    collapsed.name = data.name
    collapsed.attrs = dict(data.attrs)
    return collapsed


def _build_map_color_scale(
    values: np.ndarray,
    *,
    label: str,
    units: str,
    colormap: str = "ocean_diverging",
) -> Optional[Dict[str, Any]]:
    finite_values = np.asarray(values, dtype=float)
    valid = finite_values[np.isfinite(finite_values)]
    if valid.size == 0:
        return None

    raw_min = float(np.nanmin(valid))
    raw_max = float(np.nanmax(valid))
    crosses_zero = raw_min < 0.0 < raw_max
    if crosses_zero:
        raw_limit = max(abs(raw_min), abs(raw_max))
        robust_limit = float(np.nanquantile(np.abs(valid), 0.98))
        scale_limit = robust_limit if np.isfinite(robust_limit) and robust_limit > 0.0 else raw_limit
        scale_min = -float(scale_limit)
        scale_max = float(scale_limit)
        scale_strategy = "symmetric_p98_abs"
    else:
        robust_min = float(np.nanquantile(valid, 0.02))
        robust_max = float(np.nanquantile(valid, 0.98))
        if not np.isfinite(robust_min) or not np.isfinite(robust_max) or robust_max <= robust_min:
            scale_min = raw_min
            scale_max = raw_max
            scale_strategy = "raw_extent"
        else:
            scale_min = robust_min
            scale_max = robust_max
            scale_strategy = "p02_p98"

    if not np.isfinite(scale_min) or not np.isfinite(scale_max) or scale_max <= scale_min:
        scale_min = raw_min
        scale_max = raw_max
        scale_strategy = "raw_extent"

    return {
        "min": float(scale_min),
        "max": float(scale_max),
        "rawMin": raw_min,
        "rawMax": raw_max,
        "colormap": colormap,
        "units": units,
        "label": label,
        "symmetric": bool(crosses_zero),
        "scaleStrategy": scale_strategy,
    }


def _map_extreme_sample_indices(values: np.ndarray) -> tuple[List[int], List[int]]:
    """Return row/column indices that preserve sparse map extrema during preview sampling."""
    if values.ndim != 2 or values.size == 0:
        return [], []

    finite_mask = np.isfinite(values)
    if not bool(np.any(finite_mask)):
        return [], []

    finite_values = np.where(finite_mask, values, np.nan)
    required_rows: set[int] = set()
    required_cols: set[int] = set()
    for flat_index in (
        int(np.nanargmin(finite_values)),
        int(np.nanargmax(finite_values)),
    ):
        row_index, col_index = np.unravel_index(flat_index, values.shape)
        required_rows.add(int(row_index))
        required_cols.add(int(col_index))
    return sorted(required_rows), sorted(required_cols)


def _sample_indices_with_required(
    length: int,
    *,
    limit: int,
    required_indices: Sequence[int],
) -> List[int]:
    if length <= 0:
        return []
    if length <= limit:
        return list(range(length))

    required = sorted({int(index) for index in required_indices if 0 <= int(index) < length})
    if not required:
        return _sample_indices(length, limit)

    target_count = max(1, min(int(limit), length))
    sample_count = max(0, target_count - len(required))
    sampled = set(_sample_indices(length, sample_count)) if sample_count else set()
    sampled.update(required)

    if len(sampled) > target_count:
        keep = set(required[:target_count])
        for index in sorted(sampled):
            if len(keep) >= target_count:
                break
            keep.add(index)
        sampled = keep

    return sorted(sampled)


def _build_map_field_from_data_container_result(
    result: Dict[str, Any],
    title: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return None

    data = result.get("data")
    if data is None or not hasattr(data, "dims"):
        return None

    if hasattr(data, "data_vars"):
        data_vars = list(getattr(data, "data_vars", {}).keys())
        if not data_vars:
            return None
        data = data[data_vars[0]]

    if _skip_auto_workspace_data_container_compute(data):
        return None

    if "lat" not in getattr(data, "dims", ()) or "lon" not in getattr(data, "dims", ()):
        return None

    field = data
    time_range = _xarray_time_range(field)
    if "time" in field.dims:
        if isinstance(field, PartitionedDataArray):
            field = _mean_xarray_like(field, ["time"])
        elif int(field.sizes.get("time", 0)) <= 1:
            field = field.isel(time=0, drop=True)
        else:
            field = _mean_xarray_like(field, ["time"])

    depth_label = None
    depth_dim = next((dim for dim in field.dims if str(dim).lower() in {"depth", "lev", "level", "z"}), None)
    if depth_dim is not None:
        depth_values = np.asarray(field[depth_dim].values, dtype=float)
        if depth_values.size == 1:
            depth_label = f"{depth_values[0]:g} m"
            field = field.isel({depth_dim: 0}, drop=True)
        elif depth_values.size > 1:
            depth_label = f"{depth_values.min():g} to {depth_values.max():g} m"
            field = _mean_xarray_like(field, [depth_dim])

    extra_dims = [dim for dim in field.dims if dim not in {"lat", "lon"}]
    if extra_dims:
        field = _mean_xarray_like(field, extra_dims)

    if "lat" not in getattr(field, "dims", ()) or "lon" not in getattr(field, "dims", ()):
        return None

    values = as_numeric_array(field.values)
    if values.ndim != 2 or values.size == 0:
        return None

    valid = values[np.isfinite(values)]
    statistics: Dict[str, float] = {}
    if valid.size:
        statistics = {
            "mean": float(np.nanmean(valid)),
            "std": float(np.nanstd(valid)),
            "min": float(np.nanmin(valid)),
            "max": float(np.nanmax(valid)),
        }

    result_metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    map_field = _build_map_field_payload(
        spatial_field={
            "lon": as_numeric_array(field["lon"].values),
            "lat": as_numeric_array(field["lat"].values),
            "values": values,
            "metadata": {
                "variable": str(field.name or result_metadata.get("variable") or "unknown"),
                "units": str(field.attrs.get("units", result_metadata.get("units", ""))),
                "statistics": statistics,
                "time_range": list(time_range) if any(time_range) else None,
                "disable_contour_preview": True,
            },
        },
        title=title,
        time_range=time_range,
        max_lon_points=80,
        max_lat_points=80,
    )
    if depth_label and not map_field.get("depthLabel"):
        map_field["depthLabel"] = depth_label
    return map_field


def _format_time_label(time_range: tuple[str, str]) -> str:
    start, end = time_range
    if not start and not end:
        return ""
    if start == end:
        return start
    return f"{start} ~ {end} mean"


def _parse_subregion_indices(subregion_id: Any) -> Optional[tuple[int, int]]:
    if not isinstance(subregion_id, str):
        return None
    match = re.fullmatch(r"r(\d+)_c(\d+)", subregion_id.strip().lower())
    if not match:
        return None
    row = int(match.group(1))
    col = int(match.group(2))
    if row <= 0 or col <= 0:
        return None
    return row, col


def _subregion_display_label(
    subregion_id: str,
    label: str,
    *,
    grid_shape: tuple[int, int],
) -> str:
    indices = _parse_subregion_indices(subregion_id)
    if indices is None:
        return label or subregion_id
    row, col = indices
    if grid_shape == (2, 2):
        return {
            (2, 1): "NW",
            (2, 2): "NE",
            (1, 1): "SW",
            (1, 2): "SE",
        }.get((row, col), label or subregion_id)
    return label or subregion_id


def _mechanism_subregion_status(item: Dict[str, Any]) -> str:
    return str(item.get("status") or "").strip().lower()


def _mechanism_subregion_bounds(item: Dict[str, Any]) -> Optional[Dict[str, float]]:
    lon_range = item.get("lon_range")
    lat_range = item.get("lat_range")
    if not (
        isinstance(lon_range, (list, tuple))
        and len(lon_range) == 2
        and isinstance(lat_range, (list, tuple))
        and len(lat_range) == 2
    ):
        return None
    try:
        lon_min, lon_max = sorted((float(lon_range[0]), float(lon_range[1])))
        lat_min, lat_max = sorted((float(lat_range[0]), float(lat_range[1])))
    except (TypeError, ValueError):
        return None
    return {
        "lonMin": lon_min,
        "lonMax": lon_max,
        "latMin": lat_min,
        "latMax": lat_max,
    }


def _iter_mechanism_subregion_breakdowns(
    result: Dict[str, Any],
) -> List[tuple[str, str, Dict[str, Any]]]:
    proxy_breakdowns = result.get("proxy_breakdowns")
    records: List[tuple[str, str, Dict[str, Any]]] = []

    if isinstance(proxy_breakdowns, list) and proxy_breakdowns:
        for proxy_item in proxy_breakdowns:
            if not isinstance(proxy_item, dict):
                continue
            proxy_name = str(proxy_item.get("name") or "").strip().lower()
            proxy_claim = str(proxy_item.get("claim_strength") or "").strip()
            subregions = proxy_item.get("subregion_breakdown")
            if not isinstance(subregions, list):
                continue
            for subitem in subregions:
                if isinstance(subitem, dict):
                    records.append((proxy_name, proxy_claim, subitem))
        if records:
            return records

    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    fallback_proxy = str(metadata.get("source_proxy") or result.get("top_mechanism") or "").strip().lower()
    fallback_claim = str(result.get("claim_strength") or "").strip()
    breakdown = result.get("subregion_breakdown")
    if not isinstance(breakdown, list):
        return records
    for subitem in breakdown:
        if isinstance(subitem, dict):
            records.append((fallback_proxy, fallback_claim, subitem))
    return records


def _mechanism_subregion_consensus(
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    records = _iter_mechanism_subregion_breakdowns(result)
    if not records:
        return None

    grouped: Dict[str, Dict[str, Any]] = {}
    max_row = 0
    max_col = 0

    for proxy_name, proxy_claim, subitem in records:
        subregion_id = str(subitem.get("subregion_id") or "").strip()
        if not subregion_id:
            continue
        entry = grouped.setdefault(
            subregion_id,
            {
                "subregionId": subregion_id,
                "label": str(subitem.get("label") or subregion_id).strip(),
                "bounds": None,
                "candidates": [],
                "statuses": [],
            },
        )
        if entry.get("bounds") is None:
            entry["bounds"] = _mechanism_subregion_bounds(subitem)

        status = _mechanism_subregion_status(subitem)
        if status:
            entry["statuses"].append(status)

        indices = _parse_subregion_indices(subregion_id)
        if indices is not None:
            row, col = indices
            max_row = max(max_row, row)
            max_col = max(max_col, col)

        contrast = subitem.get("standardized_contrast")
        if status == "ok" and isinstance(contrast, (int, float)) and np.isfinite(float(contrast)):
            entry["candidates"].append(
                {
                    "mechanism": proxy_name,
                    "score": float(contrast),
                    "claimStrength": str(subitem.get("claim_strength") or proxy_claim or "").strip(),
                }
            )

    if not grouped:
        return None

    grid_shape = (max_col or 1, max_row or 1)
    cells: List[Dict[str, Any]] = []
    row_centers: Dict[int, float] = {}
    col_centers: Dict[int, float] = {}
    lon_bounds: List[float] = []
    lat_bounds: List[float] = []

    for subregion_id, entry in grouped.items():
        bounds = entry.get("bounds") if isinstance(entry.get("bounds"), dict) else None
        indices = _parse_subregion_indices(subregion_id)
        candidates = sorted(
            entry.get("candidates", []),
            key=lambda item: (-float(item.get("score") or 0.0), str(item.get("mechanism") or "")),
        )
        statuses = [str(item).strip().lower() for item in entry.get("statuses", []) if str(item).strip()]
        cell_status = "ok" if candidates else (statuses[0] if statuses else "skipped_no_valid_samples")
        dominant_score = float(candidates[0]["score"]) if candidates else float("nan")
        dominant_mechanism = (
            str(candidates[0].get("mechanism") or "").strip().lower()
            if candidates and dominant_score >= 0.5
            else ("background_like" if candidates else None)
        )
        runner_up = candidates[1] if len(candidates) > 1 else None
        cells.append(
            {
                "subregionId": subregion_id,
                "label": str(entry.get("label") or subregion_id).strip(),
                "shortLabel": _subregion_display_label(
                    subregion_id,
                    str(entry.get("label") or subregion_id).strip(),
                    grid_shape=grid_shape,
                ),
                "bounds": bounds,
                "dominantMechanism": dominant_mechanism,
                "dominantScore": dominant_score if np.isfinite(dominant_score) else None,
                "runnerUpMechanism": str(runner_up.get("mechanism") or "").strip().lower() if runner_up else None,
                "runnerUpScore": float(runner_up.get("score")) if runner_up and isinstance(runner_up.get("score"), (int, float)) else None,
                "claimStrength": str(candidates[0].get("claimStrength") or "").strip() if candidates else None,
                "status": cell_status,
            }
        )

        if bounds is not None:
            lon_bounds.extend([bounds["lonMin"], bounds["lonMax"]])
            lat_bounds.extend([bounds["latMin"], bounds["latMax"]])
            if indices is not None:
                row, col = indices
                row_centers[row] = (bounds["latMin"] + bounds["latMax"]) / 2.0
                col_centers[col] = (bounds["lonMin"] + bounds["lonMax"]) / 2.0

    if not cells or not lon_bounds or not lat_bounds:
        return None

    valid_cells = [cell for cell in cells if cell.get("status") == "ok" and isinstance(cell.get("dominantScore"), (int, float))]
    dominant_counter = Counter(
        str(cell.get("dominantMechanism") or "").strip()
        for cell in valid_cells
        if str(cell.get("dominantMechanism") or "").strip()
    )

    return {
        "gridShape": [int(grid_shape[0]), int(grid_shape[1])],
        "cells": sorted(
            cells,
            key=lambda item: (
                -(_parse_subregion_indices(str(item.get("subregionId") or "")) or (0, 0))[0],
                (_parse_subregion_indices(str(item.get("subregionId") or "")) or (0, 0))[1],
            ),
        ),
        "lonCenters": [float(col_centers[index]) for index in sorted(col_centers)],
        "latCenters": [float(row_centers[index]) for index in sorted(row_centers)],
        "bounds": {
            "lonMin": float(min(lon_bounds)),
            "lonMax": float(max(lon_bounds)),
            "latMin": float(min(lat_bounds)),
            "latMax": float(max(lat_bounds)),
        },
        "validCount": len(valid_cells),
        "skippedCount": len([cell for cell in cells if cell.get("status") != "ok"]),
        "dominantCounts": dict(sorted(dominant_counter.items())),
    }


def _build_mechanism_subregion_map_field(
    result: Dict[str, Any],
    title: str,
) -> Optional[Dict[str, Any]]:
    diagnosis = _mechanism_subregion_consensus(result)
    if diagnosis is None:
        return None

    grid_shape = diagnosis.get("gridShape")
    lon_centers = diagnosis.get("lonCenters")
    lat_centers = diagnosis.get("latCenters")
    bounds = diagnosis.get("bounds")
    cells = diagnosis.get("cells")
    if not (
        isinstance(grid_shape, list)
        and len(grid_shape) == 2
        and isinstance(lon_centers, list)
        and isinstance(lat_centers, list)
        and isinstance(bounds, dict)
        and isinstance(cells, list)
    ):
        return None

    n_cols = int(grid_shape[0])
    n_rows = int(grid_shape[1])
    if len(lon_centers) != n_cols or len(lat_centers) != n_rows:
        return None

    values = np.full((n_rows, n_cols), np.nan, dtype=float)
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        indices = _parse_subregion_indices(cell.get("subregionId"))
        if indices is None:
            continue
        row, col = indices
        score = cell.get("dominantScore")
        if 1 <= row <= n_rows and 1 <= col <= n_cols and isinstance(score, (int, float)) and np.isfinite(float(score)):
            values[row - 1, col - 1] = float(score)

    valid = values[np.isfinite(values)]
    statistics: Dict[str, float] = {}
    if valid.size:
        statistics = {
            "mean": float(np.nanmean(valid)),
            "std": float(np.nanstd(valid)),
            "min": float(np.nanmin(valid)),
            "max": float(np.nanmax(valid)),
        }

    return {
        "lon": [float(value) for value in lon_centers],
        "lat": [float(value) for value in lat_centers],
        "values": values.tolist(),
        "label": title,
        "variable": "mechanism_subregion_score",
        "units": "standardized contrast",
        "statistics": statistics,
        "timeLabel": _format_time_label(_infer_time_range_from_result(result)),
        "bounds": [
            [float(bounds["latMin"]), float(bounds["lonMin"])],
            [float(bounds["latMax"]), float(bounds["lonMax"])],
        ],
        "subregionGrid": diagnosis,
    }


def _summarize_mechanism_subregion_pattern(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    diagnosis = _mechanism_subregion_consensus(summary)
    if diagnosis is None:
        return None
    cells = diagnosis.get("cells")
    if not isinstance(cells, list) or not cells:
        return None
    valid_cells = [cell for cell in cells if isinstance(cell, dict) and cell.get("status") == "ok"]
    if not valid_cells:
        return None

    dominant_counter = Counter(
        str(cell.get("dominantMechanism") or "").strip()
        for cell in valid_cells
        if str(cell.get("dominantMechanism") or "").strip()
    )
    if not dominant_counter:
        return None

    top_name, top_count = dominant_counter.most_common(1)[0]
    mean_top_score = float(np.nanmean([
        float(cell["dominantScore"])
        for cell in valid_cells
        if isinstance(cell.get("dominantScore"), (int, float))
    ]))
    background_like_count = dominant_counter.get("background_like", 0)
    return {
        "diagnosis": diagnosis,
        "validCount": len(valid_cells),
        "backgroundLikeCount": background_like_count,
        "dominantCounts": dominant_counter,
        "topMechanism": top_name,
        "topCount": top_count,
        "meanTopScore": mean_top_score,
        "isMixed": len(dominant_counter) > 1,
        "isMostlyBackground": background_like_count >= max(1, math.ceil(len(valid_cells) / 2)),
        "isConsistent": top_name != "background_like" and top_count == len(valid_cells),
    }


def _build_composite_field_entries(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        return []

    variable_label = _humanize_data_label(result.get("metadata", {}).get("variable")) or "Composite"
    time_range = _infer_time_range_from_result(result)
    entries: List[Dict[str, Any]] = []
    field_specs = [
        ("positive", f"Positive-phase {variable_label}", result.get("positive_composite")),
        ("negative", f"Negative-phase {variable_label}", result.get("negative_composite")),
        ("difference", f"{variable_label} Difference", result.get("difference")),
    ]

    for field_id, title, field in field_specs:
        if not isinstance(field, dict):
            continue
        map_field = _build_map_field_payload(
            spatial_field=field,
            title=title,
            time_range=time_range,
            max_lon_points=48,
            max_lat_points=48,
        )
        entries.append(
            {
                "id": field_id,
                "title": title,
                "mapField": map_field,
            }
        )

    return entries


def _timeseries_label(ts_result: Dict[str, Any], *, chinese: bool = False) -> str:
    """Build a descriptive label for a timeseries result from its metadata."""
    metadata = ts_result.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    variable = _humanize_data_label(metadata.get("variable"), chinese=chinese)
    aggregation = _humanize_data_label(metadata.get("aggregation"), chinese=chinese)
    if variable and aggregation:
        return f"{variable}{aggregation}" if chinese else f"{aggregation} of {variable}"
    if variable:
        return variable
    return "时间序列" if chinese else "Timeseries"


def _climatology_period_label(period: Any) -> Optional[str]:
    if not isinstance(period, str):
        return None
    lowered = period.strip().lower()
    if lowered == "monthly":
        return "Monthly"
    if lowered == "seasonal":
        return "Seasonal"
    return lowered.title() if lowered else None


def _format_climatology_label(label: Any, period: Any) -> str:
    lowered = str(period or "").strip().lower()
    if lowered == "monthly":
        try:
            month = int(label)
        except (TypeError, ValueError):
            return str(label)
        if 1 <= month <= 12:
            return calendar.month_abbr[month]
        return str(label)
    if lowered == "seasonal":
        try:
            quarter = int(label)
        except (TypeError, ValueError):
            return str(label)
        return f"Q{quarter}"
    return str(label)


def _build_climatology_series(result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    period = result.get("period")
    labels = [_format_climatology_label(label, period) for label in result.get("labels", [])]
    return _build_named_series(
        labels=labels,
        values=result.get("values", []),
        label_key="label",
        value_key="value",
        limit=None,
    )


def _build_lag_correlation_series(result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    return _build_named_series(
        labels=result.get("lags", []),
        values=result.get("correlations", []),
        label_key="label",
        value_key="value",
        limit=None,
    )


def _build_event_statistics_series(stats_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(stats_result, dict):
        return []

    groups = stats_result.get("groups")
    if not isinstance(groups, dict) or not groups:
        return []

    group_by = str(stats_result.get("group_by") or "").strip().lower()
    month_order = {label: index for index, label in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )}
    season_order = {label: index for index, label in enumerate(["DJF", "MAM", "JJA", "SON", "Winter", "Spring", "Summer", "Fall"])}

    def sort_key(item: tuple[str, Any]) -> tuple[int, Any]:
        label, _payload = item
        if group_by == "month":
            return (0, month_order.get(str(label), 999))
        if group_by == "season":
            return (0, season_order.get(str(label), 999))
        if group_by == "year":
            try:
                return (0, int(label))
            except (TypeError, ValueError):
                return (1, str(label))
        return (0, str(label))

    series: List[Dict[str, Any]] = []
    for label, payload in sorted(groups.items(), key=sort_key):
        if not isinstance(payload, dict):
            continue
        count = payload.get("count")
        if not isinstance(count, (int, float)):
            continue
        series.append({"label": str(label), "value": float(count)})
    return series


def _build_workspace_data(
    executor: SkillExecutor,
    result_cards: List[Dict[str, Any]],
    active_result_id: Optional[str],
    *,
    chinese: bool = False,
) -> Dict[str, Any]:
    workspace_data = _empty_workspace_data()
    ordered_results = _collect_ordered_results(executor, result_cards)
    result_card_by_id = {card.get("id"): card for card in result_cards if isinstance(card, dict) and card.get("id")}

    timeseries_results = [result for _, result in ordered_results if result.get("output_type") == "timeseries_result"]
    climatology_results = [result for _, result in ordered_results if result.get("output_type") == "climatology_result"]
    profile_results = [result for _, result in ordered_results if result.get("output_type") == "profile_result"]
    hovmoller_results = [result for _, result in ordered_results if result.get("output_type") == "hovmoller_result"]
    section_results = [result for _, result in ordered_results if result.get("output_type") == "section_result"]
    histogram_results = [result for _, result in ordered_results if result.get("output_type") == "histogram_result"]
    ts_diagram_results = [result for _, result in ordered_results if result.get("output_type") == "ts_diagram_result"]
    spectrum_results = [result for _, result in ordered_results if result.get("output_type") == "spectrum_result"]
    lag_correlation_results = [result for _, result in ordered_results if result.get("output_type") == "lag_correlation_result"]
    eof_results = [result for _, result in ordered_results if result.get("output_type") == "eof_result"]
    data_container_results = [
        result for _, result in ordered_results if result.get("output_type") == "data_container_result"
    ]
    feature_results = [result for result in data_container_results if _get_feature_name(result) is not None]
    active_result = None
    active_output_type = None
    if active_result_id is not None:
        try:
            active_result = executor.get_result(active_result_id)
            active_output_type = active_result.get("output_type") if isinstance(active_result, dict) else None
        except KeyError:
            active_result = None
            active_output_type = None

    reference_series = []
    if active_output_type == "data_container_result" and active_result is not None:
        reference_series = _build_reference_series(active_result)
        workspace_data["seriesLabels"] = {
            "reference": "参考场" if chinese else "Reference field",
            "result": "结果" if chinese else "Result",
            "compare": "对比" if chinese else "Compare",
        }
        active_card = result_card_by_id.get(active_result_id or "", {})
        map_field = _build_map_field_from_data_container_result(
            active_result,
            title=str(active_card.get("title") or _humanize_result_id(active_result_id or "result")),
        )
        if map_field is not None:
            workspace_data["mapField"] = map_field
    elif active_output_type == "trend_result" and isinstance(active_result, dict):
        active_trend = active_result
        source_times = active_trend.get("times") if isinstance(active_trend.get("times"), list) else []
        source_values = active_trend.get("values") if isinstance(active_trend.get("values"), list) else []
        has_source_series = bool(source_times and source_values)
        base_timeseries = None

        if has_source_series:
            workspace_data["resultSeries"] = _build_named_series(
                labels=source_times,
                values=source_values,
                label_key="label",
                value_key="value",
                limit=None,
            )
        else:
            base_timeseries = _nearest_prior_result(ordered_results, active_result_id, "timeseries_result")

        if base_timeseries is not None and not workspace_data["resultSeries"]:
            workspace_data["resultSeries"] = _build_named_series(
                labels=base_timeseries.get("times", []),
                values=base_timeseries.get("values", []),
                label_key="label",
                value_key="value",
                limit=None,
            )
        trend_line = active_trend.get("trend_line", [])
        trend_labels = (
            source_times
            if has_source_series
            else (base_timeseries.get("times", []) if base_timeseries is not None else [f"T{i + 1}" for i in range(len(trend_line))])
        )
        workspace_data["anomalySeries"] = _build_named_series(
            labels=trend_labels,
            values=trend_line,
            label_key="label",
            value_key="value",
            limit=None,
        )
        workspace_data["seriesLabels"] = {
            "result": "观测值" if chinese else "Observed",
            "compare": "趋势线" if chinese else "Trend line",
        }
    elif active_output_type == "mechanism_score_result" and isinstance(active_result, dict):
        active_metadata = active_result.get("metadata", {}) if isinstance(active_result.get("metadata"), dict) else {}
        active_card = result_card_by_id.get(active_result_id or "", {})
        mechanism_map_field = _build_mechanism_subregion_map_field(
            active_result,
            title=str(active_card.get("title") or _humanize_result_id(active_result_id or "mechanism")),
        )
        if mechanism_map_field is not None:
            workspace_data["mapField"] = mechanism_map_field
        if active_metadata.get("comparison") == "event_condition_contrast" and {"times", "values"}.issubset(active_result.keys()):
            workspace_data["resultSeries"] = _build_named_series(
                labels=active_result.get("times", []),
                values=active_result.get("values", []),
                label_key="label",
                value_key="value",
                limit=None,
            )
            proxy_label = _mechanism_source_proxy_label({"metadata": active_metadata}, chinese=chinese)
            workspace_data["seriesLabels"] = {
                "result": proxy_label or ("事件条件对比来源" if chinese else "Event-condition contrast source"),
            }
    elif timeseries_results and active_output_type not in {
        "event_statistics_result",
        "spectrum_result",
        "climatology_result",
        "mechanism_score_result",
        "evidence_report_result",
        "event_comparison_result",
        "environment_assessment_result",
        "policy_recommendation_result",
    }:
        # Build id-indexed lookup for timeseries results
        timeseries_by_id = {
            rid: result
            for rid, result in ordered_results
            if result.get("output_type") == "timeseries_result"
        }

        if active_result_id and active_result_id in timeseries_by_id:
            # Show only the active timeseries in resultSeries
            primary_timeseries = timeseries_by_id[active_result_id]
            primary_label = _timeseries_label(primary_timeseries, chinese=chinese)
            primary_payload = _build_timeseries_display_payload(primary_timeseries)
            workspace_data["resultSeries"] = primary_payload["series"]
            workspace_data["timeseriesDisplayInfo"] = primary_payload["display_info"]
            workspace_data["seriesLabels"] = {
                "result": primary_label,
            }
        else:
            # No specific active timeseries — show first as primary
            primary_timeseries = timeseries_results[0]
            primary_label = _timeseries_label(primary_timeseries, chinese=chinese)
            primary_payload = _build_timeseries_display_payload(primary_timeseries)
            workspace_data["resultSeries"] = primary_payload["series"]
            workspace_data["timeseriesDisplayInfo"] = primary_payload["display_info"]
            comparison_result = timeseries_results[1] if len(timeseries_results) > 1 else None
            if comparison_result is not None:
                compare_label = _timeseries_label(comparison_result, chinese=chinese)
                compare_payload = _build_timeseries_display_payload(comparison_result)
                workspace_data["anomalySeries"] = compare_payload["series"]
                workspace_data["seriesLabels"] = {
                    "result": primary_label,
                    "compare": compare_label,
                }
            else:
                workspace_data["seriesLabels"] = {
                    "result": primary_label,
                }
    elif active_output_type == "event_statistics_result" and isinstance(active_result, dict):
        event_count_series = _build_event_statistics_series(active_result)
        if event_count_series:
            workspace_data["resultSeries"] = event_count_series
            group_by = str(active_result.get("group_by") or "group")
            workspace_data["seriesLabels"] = {
                "result": f"按 {group_by} 统计的事件数" if chinese else f"Event count by {group_by}",
            }
    elif active_output_type == "spectrum_result" and isinstance(active_result, dict):
        spectrum_series = _build_spectrum_series(active_result)
        if spectrum_series:
            workspace_data["resultSeries"] = spectrum_series
            workspace_data["seriesLabels"] = {
                "result": "谱功率" if chinese else "Spectral power",
            }
    elif active_output_type == "lag_correlation_result" and isinstance(active_result, dict):
        lag_series = _build_lag_correlation_series(active_result)
        if lag_series:
            workspace_data["resultSeries"] = lag_series
            workspace_data["seriesLabels"] = {
                "result": "滞后相关" if chinese else "Lag correlation",
            }
    elif active_output_type == "climatology_result" and isinstance(active_result, dict):
        climatology_series = _build_climatology_series(active_result)
        if climatology_series:
            workspace_data["resultSeries"] = climatology_series
            period_label = _climatology_period_label(active_result.get("period")) or "Climatology"
            workspace_data["seriesLabels"] = {
                "result": f"{period_label}气候态" if chinese else f"{period_label} climatology",
            }
    else:
        reference_series = (
            _build_reference_series(data_container_results[0])
            if data_container_results
            else _build_reference_series_from_structured_result(
                timeseries_results[0]
                if timeseries_results
                else (climatology_results[0] if climatology_results else None)
            )
        )

    workspace_data["referenceSeries"] = reference_series

    if not workspace_data["resultSeries"] and reference_series:
        workspace_data["resultSeries"] = list(reference_series)
    if not workspace_data["anomalySeries"]:
        workspace_data["anomalySeries"] = []

    if profile_results:
        workspace_data["profileSeries"] = _build_profile_series(profile_results[0])

    if feature_results:
        workspace_data["profileMarkers"] = _build_profile_markers(feature_results)
        workspace_data["overlaySeries"] = _build_overlay_series(feature_results[0])

    if hovmoller_results:
        hovmoller_payload = _build_hovmoller_payload(hovmoller_results[0])
        workspace_data["hovmollerRows"] = hovmoller_payload["rows"]
        workspace_data["hovmollerTimeLabels"] = hovmoller_payload["time_labels"]
        workspace_data["hovmollerDisplayInfo"] = hovmoller_payload["display_info"]
        workspace_data["hovmollerDepthIntegratedSeries"] = hovmoller_payload["depth_integrated_series"]

    section_source = None
    if active_output_type == "section_result" and isinstance(active_result, dict):
        section_source = active_result
    elif section_results:
        section_source = section_results[0]

    if section_source:
        section_payload = _build_section_payload(section_source)
        workspace_data["sectionRows"] = section_payload["rows"]
        workspace_data["sectionDistanceKm"] = section_payload["distance_km"]
        workspace_data["sectionAxisTitle"] = section_payload["axis_title"]
        workspace_data["sectionSliceLabel"] = section_payload["slice_label"]
        workspace_data["sectionDisplayInfo"] = section_payload.get("display_info")

    if histogram_results:
        workspace_data["histogramBins"] = _build_histogram_bins(histogram_results[0])

    if active_output_type == "ts_diagram_result" and isinstance(active_result, dict):
        workspace_data.update(_build_ts_diagram_payload(active_result))
    elif ts_diagram_results:
        workspace_data.update(_build_ts_diagram_payload(ts_diagram_results[0]))
    elif active_output_type == "spectrum_result" and isinstance(active_result, dict) and not workspace_data["resultSeries"]:
        workspace_data["resultSeries"] = _build_spectrum_series(active_result)
        workspace_data["seriesLabels"] = {"result": "谱功率" if chinese else "Spectral power"}
    elif spectrum_results and not workspace_data["resultSeries"]:
        workspace_data["resultSeries"] = _build_spectrum_series(spectrum_results[0])
        workspace_data["seriesLabels"] = {"result": "谱功率" if chinese else "Spectral power"}
    elif active_output_type == "lag_correlation_result" and isinstance(active_result, dict) and not workspace_data["resultSeries"]:
        workspace_data["resultSeries"] = _build_lag_correlation_series(active_result)
        workspace_data["seriesLabels"] = {"result": "滞后相关" if chinese else "Lag correlation"}
    elif lag_correlation_results and not workspace_data["resultSeries"]:
        workspace_data["resultSeries"] = _build_lag_correlation_series(lag_correlation_results[0])
        workspace_data["seriesLabels"] = {"result": "滞后相关" if chinese else "Lag correlation"}
    elif active_output_type == "climatology_result" and isinstance(active_result, dict) and not workspace_data["resultSeries"]:
        workspace_data["resultSeries"] = _build_climatology_series(active_result)
        period_label = _climatology_period_label(active_result.get("period")) or "Climatology"
        workspace_data["seriesLabels"] = {"result": f"{period_label}气候态" if chinese else f"{period_label} climatology"}
    elif climatology_results and not workspace_data["resultSeries"]:
        workspace_data["resultSeries"] = _build_climatology_series(climatology_results[0])
        period_label = _climatology_period_label(climatology_results[0].get("period")) or "Climatology"
        workspace_data["seriesLabels"] = {"result": f"{period_label}气候态" if chinese else f"{period_label} climatology"}

    if eof_results:
        eof_variance, eof_pc_series, eof_modes = _build_eof_payload(eof_results[0])
        workspace_data["eofVariance"] = eof_variance
        workspace_data["eofPcSeries"] = eof_pc_series
        workspace_data["eofModes"] = eof_modes
        if active_output_type == "eof_result" and eof_modes:
            workspace_data["mapField"] = eof_modes[0].get("mapField")

    if active_result and active_result.get("output_type") == "data_container_result" and not reference_series:
        workspace_data["referenceSeries"] = _build_reference_series(active_result)

    if active_output_type == "spatial_field_result" and isinstance(active_result, dict) and active_result_id:
        active_card = result_card_by_id.get(active_result_id, {})
        workspace_data["mapField"] = _build_map_field_payload(
            spatial_field=active_result,
            title=str(active_card.get("title") or _humanize_result_id(active_result_id)),
            time_range=_infer_time_range_from_result(active_result),
        )

    if active_output_type == "field_trend_result" and isinstance(active_result, dict) and active_result_id:
        active_card = result_card_by_id.get(active_result_id, {})
        workspace_data["mapField"] = _build_field_trend_map_payload(
            result=active_result,
            title=str(active_card.get("title") or _humanize_result_id(active_result_id)),
        )

    if active_output_type == "regression_map_result" and isinstance(active_result, dict) and active_result_id:
        active_card = result_card_by_id.get(active_result_id, {})
        workspace_data["mapField"] = _build_regression_map_payload(
            result=active_result,
            title=str(active_card.get("title") or _humanize_result_id(active_result_id)),
        )

    if active_output_type == "composite_result" and isinstance(active_result, dict):
        composite_fields = _build_composite_field_entries(active_result)
        workspace_data["compositeFields"] = composite_fields
        difference_source = active_result.get("difference")
        if isinstance(difference_source, dict):
            workspace_data["mapField"] = _build_map_field_payload(
                spatial_field=difference_source,
                title=f"{_humanize_data_label(active_result.get('metadata', {}).get('variable')) or 'Composite'} Difference",
                time_range=_infer_time_range_from_result(active_result),
            )

    if active_output_type == "event_detection_result" and isinstance(active_result, dict):
        workspace_data["eventOverlays"] = _build_event_overlays(active_result)

    if (
        workspace_data.get("mapField") is None
        and active_output_type in {"mechanism_score_result", "watermass_event_association_result", "evidence_report_result", "environment_assessment_result", "policy_recommendation_result"}
    ):
        preferred_spatial = _preferred_spatial_context_result(ordered_results)
        if preferred_spatial is not None:
            spatial_result_id, spatial_result = preferred_spatial
            spatial_card = result_card_by_id.get(spatial_result_id, {})
            workspace_data["mapField"] = _build_map_field_payload(
                spatial_field=spatial_result,
                title=str(spatial_card.get("title") or _humanize_result_id(spatial_result_id)),
                time_range=_infer_time_range_from_result(spatial_result),
            )

    return workspace_data


def _build_workspace_data_by_result(
    executor: SkillExecutor,
    result_cards: List[Dict[str, Any]],
    *,
    chinese: bool = False,
) -> Dict[str, Dict[str, Any]]:
    workspace_by_result: Dict[str, Dict[str, Any]] = {}
    for card in result_cards:
        result_id = card.get("id")
        if not isinstance(result_id, str) or not result_id:
            continue
        workspace_by_result[result_id] = _build_workspace_data(
            executor=executor,
            result_cards=result_cards,
            active_result_id=result_id,
            chinese=chinese,
        )
    return workspace_by_result


def _collect_ordered_results(
    executor: SkillExecutor,
    result_cards: List[Dict[str, Any]],
) -> List[tuple[str, Dict[str, Any]]]:
    ordered_results: List[tuple[str, Dict[str, Any]]] = []
    seen_objects: set[int] = set()

    for card in result_cards:
        result_id = card.get("id")
        if not result_id:
            continue
        try:
            result = executor.get_result(result_id)
        except KeyError:
            continue
        object_id = id(result)
        if object_id in seen_objects:
            continue
        seen_objects.add(object_id)
        ordered_results.append((result_id, result))

    return ordered_results


def _nearest_prior_result(
    ordered_results: List[Tuple[str, Dict[str, Any]]],
    active_result_id: Optional[str],
    output_type: str,
) -> Optional[Dict[str, Any]]:
    if not ordered_results:
        return None

    active_index = None
    if isinstance(active_result_id, str):
        for index, (result_id, _result) in enumerate(ordered_results):
            if result_id == active_result_id:
                active_index = index
                break

    if active_index is None:
        return None

    candidates = ordered_results[:active_index]
    for _result_id, result in reversed(candidates):
        if result.get("output_type") == output_type:
            return result
    return None


def _preferred_spatial_context_result(
    ordered_results: List[Tuple[str, Dict[str, Any]]],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    watermass_hotspot_candidates: List[Tuple[str, Dict[str, Any]]] = []
    watermass_dominant_candidates: List[Tuple[str, Dict[str, Any]]] = []
    burden_candidates: List[Tuple[str, Dict[str, Any]]] = []
    event_day_candidates: List[Tuple[str, Dict[str, Any]]] = []
    spatial_candidates: List[Tuple[str, Dict[str, Any]]] = []

    for result_id, result in ordered_results:
        if result.get("output_type") != "spatial_field_result":
            continue
        spatial_candidates.append((result_id, result))
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        tile_map_kind = str(metadata.get("tile_map_kind") or "").strip().lower()
        if tile_map_kind == "event_hotspot":
            watermass_hotspot_candidates.append((result_id, result))
        elif tile_map_kind == "dominant_watermass":
            watermass_dominant_candidates.append((result_id, result))
        summary_mode = str(metadata.get("summary_mode") or "").strip().lower()
        if summary_mode == "burden":
            burden_candidates.append((result_id, result))
        elif summary_mode == "event_days":
            event_day_candidates.append((result_id, result))

    if watermass_hotspot_candidates:
        return watermass_hotspot_candidates[-1]
    if watermass_dominant_candidates:
        return watermass_dominant_candidates[-1]
    if burden_candidates:
        return burden_candidates[-1]
    if event_day_candidates:
        return event_day_candidates[-1]
    if spatial_candidates:
        return spatial_candidates[-1]
    return None


def _build_reference_series(result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        return []

    data = result.get("data")
    if data is None or not hasattr(data, "dims"):
        return []

    if hasattr(data, "data_vars"):
        data_vars = list(getattr(data, "data_vars", {}).keys())
        if not data_vars:
            return []
        data = data[data_vars[0]]

    if _skip_auto_workspace_data_container_compute(data):
        return []

    if not getattr(data, "dims", None):
        value = float(np.asarray(data.values, dtype=float))
        return [{"label": getattr(data, "name", "value") or "value", "value": value}]

    preferred_dim = "time" if "time" in data.dims else ("depth" if "depth" in data.dims else data.dims[0])
    collapsed = data
    collapse_dims = [dim for dim in data.dims if dim != preferred_dim]
    if collapse_dims:
        collapsed = _mean_xarray_like(data, collapse_dims)
    return _build_named_series(
        labels=_coord_labels(collapsed, preferred_dim),
        values=np.asarray(collapsed.values, dtype=float).tolist(),
        label_key="label",
        value_key="value",
        limit=18,
    )


def _skip_auto_workspace_data_container_compute(data: Any) -> bool:
    """Avoid silently computing large/lazy intermediate xarray results for previews."""
    if _contains_dask_backed_xarray(data):
        return True

    value_count = _xarray_value_count(data)
    if value_count is None:
        return False
    return value_count > _workspace_data_container_preview_limit()


def _contains_dask_backed_xarray(data: Any) -> bool:
    if isinstance(data, PartitionedDataArray):
        return any(is_dask_backed(partition) for partition in data.partitions)
    return is_dask_backed(data)


def _xarray_value_count(data: Any) -> Optional[int]:
    try:
        if isinstance(data, PartitionedDataArray):
            return int(data.size)
        if hasattr(data, "data_vars"):
            return int(sum(int(getattr(variable, "size", 0)) for variable in data.data_vars.values()))
        if hasattr(data, "size"):
            return int(data.size)
    except Exception:
        return None
    return None


def _workspace_data_container_preview_limit() -> int:
    raw = os.environ.get("OCEAN_WORKSPACE_DATA_CONTAINER_PREVIEW_LIMIT")
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return max(100_000, int(workspace_max_matrix_points()))


def _build_reference_series_from_structured_result(result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    if {"times", "values"}.issubset(result.keys()):
        return _build_named_series(
            labels=result.get("times", []),
            values=result.get("values", []),
            label_key="label",
            value_key="value",
            limit=18,
        )
    if {"labels", "values"}.issubset(result.keys()):
        return _build_climatology_series(result)
    return []


def _build_profile_series(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _build_pair_series(
        x_values=result.get("depth", []),
        y_values=result.get("values", []),
        x_key="depth",
        y_key="value",
        limit=40,
    )


def _build_profile_markers(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    markers: List[Dict[str, Any]] = []
    seen_labels: set[str] = set()

    for result in results:
        label = _humanize_feature_name(_get_feature_name(result))
        if not label or label in seen_labels:
            continue
        depth_value = _extract_feature_depth_scalar(result)
        if depth_value is None:
            continue
        seen_labels.add(label)
        markers.append({"label": label, "depth": round(depth_value, 3)})
        if len(markers) >= 3:
            break

    return markers


def _build_spectrum_series(result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        return []

    frequency = np.asarray(result.get("frequency", []), dtype=float)
    power = np.asarray(result.get("power", []), dtype=float)
    if frequency.size == 0 or power.size == 0:
        return []

    size = min(frequency.size, power.size)
    labels = [_format_coord_label(value, "frequency") for value in frequency[:size]]
    return _build_named_series(
        labels=labels,
        values=power[:size].tolist(),
        label_key="label",
        value_key="value",
        limit=None,
    )


def _build_overlay_series(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = result.get("data")
    if data is None or not hasattr(data, "dims") or "time" not in data.dims:
        return []
    if _skip_auto_workspace_data_container_compute(data):
        return []

    collapsed = data
    collapse_dims = [dim for dim in data.dims if dim != "time"]
    if collapse_dims:
        collapsed = _mean_xarray_like(data, collapse_dims)

    return _build_named_series(
        labels=_coord_labels(collapsed, "time"),
        values=np.asarray(collapsed.values, dtype=float).tolist(),
        label_key="day",
        value_key="depth",
        limit=18,
    )


def _safe_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _parse_hovmoller_time(value: Any) -> Optional[datetime]:
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("T", " ").replace("Z", "")
    text = re.sub(r"\.\d+", "", text)
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(text[: len(datetime.now().strftime(fmt))], fmt)
            if fmt == "%Y-%m":
                return parsed.replace(day=1)
            if fmt == "%Y":
                return parsed.replace(month=1, day=1)
            return parsed
        except ValueError:
            continue
    try:
        dt64 = np.datetime64(str(value))
        if np.isnat(dt64):
            return None
        normalized = np.datetime_as_string(dt64.astype("datetime64[s]"), unit="s")
        return _parse_hovmoller_time(normalized)
    except Exception:
        return None


def _time_group_mean(values: np.ndarray, indices: List[int]) -> np.ndarray:
    if not indices:
        return np.full(values.shape[1:], np.nan, dtype=float)
    block = values[indices, ...]
    valid_count = np.isfinite(block).sum(axis=0)
    summed = np.nansum(block, axis=0)
    return np.where(valid_count > 0, summed / np.maximum(valid_count, 1), np.nan)


def _smooth_hovmoller_time_values(values: np.ndarray, *, window: int = 30, min_periods: Optional[int] = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return values
    resolved_window = max(int(window), 1)
    if resolved_window <= 1:
        return values.copy()

    required = int(min_periods if min_periods is not None else max(1, np.ceil(resolved_window / 2)))
    before = resolved_window // 2
    after = resolved_window - before
    smoothed = np.full_like(values, np.nan, dtype=float)
    for index in range(values.shape[0]):
        start = max(0, index - before)
        end = min(values.shape[0], index + after)
        block = values[start:end, :]
        valid_count = np.isfinite(block).sum(axis=0)
        summed = np.nansum(block, axis=0)
        smoothed[index, :] = np.where(valid_count >= required, summed / np.maximum(valid_count, 1), np.nan)
    return smoothed


def _choose_hovmoller_display_aggregation(parsed_times: List[Optional[datetime]]) -> str:
    valid_times = [value for value in parsed_times if value is not None]
    if len(valid_times) < 2:
        return "none"
    duration_days = (max(valid_times) - min(valid_times)).days
    if duration_days <= 366:
        return "none"
    return "daily_climatology"


def _aggregate_hovmoller_display_values(
    values: np.ndarray,
    times: List[str],
) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
    parsed_times = [_parse_hovmoller_time(value) for value in times]
    aggregation = _choose_hovmoller_display_aggregation(parsed_times)
    original_columns = int(values.shape[0])

    if aggregation == "none" or any(value is None for value in parsed_times[: values.shape[0]]):
        return values, times[: values.shape[0]], {
            "aggregation": "none",
            "aggregationLabel": "Original time steps",
            "originalColumns": original_columns,
            "displayColumns": original_columns,
        }

    smoothed_values = _smooth_hovmoller_time_values(values, window=30)
    groups: Dict[Any, List[int]] = {}
    labels: Dict[Any, str] = {}
    for index, parsed in enumerate(parsed_times[: values.shape[0]]):
        if parsed is None:
            continue
        if parsed.month == 2 and parsed.day == 29:
            continue
        noleap_day = int(parsed.strftime("%j"))
        if parsed.month > 2 and _is_leap_year(parsed.year):
            noleap_day -= 1
        key = noleap_day
        labels.setdefault(key, parsed.strftime("%b %d"))
        groups.setdefault(key, []).append(index)

    ordered_keys = sorted(groups.keys())
    aggregation_label = "30-day smoothed daily climatology for display"

    if not ordered_keys:
        return values, times[: values.shape[0]], {
            "aggregation": "none",
            "aggregationLabel": "Original time steps",
            "originalColumns": original_columns,
            "displayColumns": original_columns,
        }

    display_values = np.vstack([_time_group_mean(smoothed_values, groups[key]) for key in ordered_keys])
    display_labels = [labels[key] for key in ordered_keys]
    return display_values, display_labels, {
        "aggregation": aggregation,
        "aggregationLabel": aggregation_label,
        "originalColumns": original_columns,
        "displayColumns": int(display_values.shape[0]),
        "smoothingWindowDays": 30,
        "smoothingMinPeriods": 15,
        "climatologyInput": "smoothed_daily",
        "leapDayPolicy": "drop_feb29_noleap_dayofyear",
    }


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _should_climatologize_timeseries_for_display(result: Dict[str, Any]) -> bool:
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    variable = str(metadata.get("variable") or result.get("variable") or "").strip().lower()
    return variable in {
        "volume_transport",
        "heat_transport",
        "salt_transport",
        "freshwater_transport",
    } or variable.endswith("_transport")


def _aggregate_timeseries_display_values(
    values: np.ndarray,
    times: List[str],
    *,
    enabled: bool,
) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
    values = np.asarray(values, dtype=float).reshape(-1)
    parsed_times = [_parse_hovmoller_time(value) for value in times]
    aggregation = _choose_hovmoller_display_aggregation(parsed_times) if enabled else "none"
    original_points = int(values.shape[0])

    if aggregation == "none" or any(value is None for value in parsed_times[:original_points]):
        return values, times[:original_points], {
            "aggregation": "none",
            "aggregationLabel": "Original time steps",
            "originalPoints": original_points,
            "displayPoints": original_points,
        }

    groups: Dict[Any, List[int]] = {}
    labels: Dict[Any, str] = {}
    for index, parsed in enumerate(parsed_times[:original_points]):
        if parsed is None:
            continue
        key = (parsed.month, parsed.day)
        labels.setdefault(key, parsed.strftime("%b %d"))
        groups.setdefault(key, []).append(index)

    ordered_keys = sorted(groups.keys())
    if not ordered_keys:
        return values, times[:original_points], {
            "aggregation": "none",
            "aggregationLabel": "Original time steps",
            "originalPoints": original_points,
            "displayPoints": original_points,
        }

    display_values = np.asarray([_time_group_mean(values, groups[key]) for key in ordered_keys], dtype=float)
    display_labels = [labels[key] for key in ordered_keys]
    return display_values, display_labels, {
        "aggregation": aggregation,
        "aggregationLabel": "Daily climatology for display",
        "originalPoints": original_points,
        "displayPoints": int(display_values.shape[0]),
    }


def _build_timeseries_display_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    values = np.asarray(result.get("values", []), dtype=float)
    times = [str(value) for value in result.get("times", [])]
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    finite_count = int(np.count_nonzero(np.isfinite(values))) if values.size else 0
    has_finite_values = finite_count > 0

    if values.size == 0:
        return {
            "series": [],
            "display_info": {
                "aggregation": "none",
                "aggregationLabel": "Original time steps",
                "originalPoints": 0,
                "displayPoints": 0,
                "variable": metadata.get("variable") or "unknown",
                "units": metadata.get("units") or metadata.get("unit") or "",
                "finiteCount": 0,
                "hasFiniteValues": False,
            },
        }

    display_values, display_labels, display_info = _aggregate_timeseries_display_values(
        values,
        times,
        enabled=_should_climatologize_timeseries_for_display(result),
    )
    display_info.update(
        {
            "variable": metadata.get("variable") or "unknown",
            "units": metadata.get("units") or metadata.get("unit") or "",
            "finiteCount": finite_count,
            "hasFiniteValues": has_finite_values,
        }
    )
    return {
        "series": _build_named_series(
            labels=display_labels,
            values=display_values.tolist(),
            label_key="label",
            value_key="value",
            limit=None,
        ),
        "display_info": display_info,
    }


def _configured_depth_levels_for_hovmoller() -> List[float]:
    try:
        levels = get_active_dataset_config().depth_levels
    except Exception:
        return []
    configured: List[float] = []
    for value in levels:
        numeric = _safe_float(value)
        if numeric is not None and np.isfinite(numeric) and abs(numeric) < 9000:
            configured.append(float(numeric))
    return configured


def _looks_like_positional_depth_axis(coords: List[Optional[float]], configured_levels: List[float]) -> bool:
    if not coords or not configured_levels or len(coords) > len(configured_levels):
        return False
    finite_coords = [coord for coord in coords if coord is not None and np.isfinite(coord)]
    if len(finite_coords) != len(coords):
        return False
    coord_values = np.asarray(finite_coords, dtype=float)
    positional = np.arange(len(coords), dtype=float)
    if not np.allclose(coord_values, positional, rtol=0.0, atol=1e-9):
        return False
    configured_slice = np.asarray(configured_levels[: len(coords)], dtype=float)
    return not np.allclose(coord_values, configured_slice, rtol=0.0, atol=1e-9)


def _snap_depth_to_config(value: float, configured_levels: List[float]) -> float:
    if not configured_levels:
        return float(value)
    configured = np.asarray(configured_levels, dtype=float)
    distances = np.abs(configured - float(value))
    nearest_index = int(np.nanargmin(distances))
    tolerance = max(1e-6, abs(float(configured[nearest_index])) * 1e-9)
    if float(distances[nearest_index]) <= tolerance:
        return float(configured[nearest_index])
    return float(value)


def _is_hovmoller_sentinel_depth(value: float) -> bool:
    configured_levels = _configured_depth_levels_for_hovmoller()
    if configured_levels:
        max_configured_depth = max(abs(level) for level in configured_levels)
        return abs(float(value)) > max_configured_depth + 1e-6
    return abs(float(value)) >= 9000


def _resolve_hovmoller_depth_coordinates(result: Dict[str, Any], n_levels: int) -> tuple[List[Optional[float]], str]:
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    spatial_dim = metadata.get("spatial_dim", "depth")
    if spatial_dim not in {"depth", "z"} or n_levels <= 0:
        return [], "not_depth"

    configured_levels = _configured_depth_levels_for_hovmoller()
    spatial_coord = result.get("spatial_coord", [])
    raw_coords = [
        _safe_float(spatial_coord[index]) if index < len(spatial_coord) else None
        for index in range(n_levels)
    ]

    if _looks_like_positional_depth_axis(raw_coords, configured_levels):
        return [float(configured_levels[index]) for index in range(n_levels)], "dataset_config"

    resolved: List[Optional[float]] = []
    used_config = False
    for index, coord in enumerate(raw_coords):
        if coord is None and index < len(configured_levels):
            resolved.append(float(configured_levels[index]))
            used_config = True
            continue
        if coord is None:
            resolved.append(None)
            continue
        resolved.append(_snap_depth_to_config(float(coord), configured_levels))

    source = "result_coordinates_with_dataset_config_fallback" if used_config else "result_coordinates"
    return resolved, source


def _build_hovmoller_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    values = as_numeric_array(result.get("values", []))
    times = [str(value) for value in result.get("time", [])]
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    depth_coordinates, depth_axis_source = _resolve_hovmoller_depth_coordinates(result, values.shape[1] if values.ndim == 2 else 0)
    if values.size == 0 or values.ndim != 2:
        return {
            "rows": [],
            "time_labels": [],
            "depth_integrated_series": [],
            "display_info": {
                "aggregation": "none",
                "aggregationLabel": "Original time steps",
                "originalColumns": 0,
                "displayColumns": 0,
                "variable": metadata.get("variable") or "unknown",
                "units": metadata.get("units") or "",
                "depthIntegratedUnits": _depth_integrated_units(metadata.get("units") or ""),
                "depthAxisSource": depth_axis_source,
                "depthLevels": [value for value in depth_coordinates if value is not None],
            },
        }

    display_values, display_labels, display_info = _aggregate_hovmoller_display_values(values, times)
    original_display_shape = [int(dim) for dim in display_values.shape]
    sampled_result = result
    sampled_depth_coordinates = depth_coordinates
    if display_values.ndim == 2 and display_values.size:
        time_indices, spatial_indices = matrix_sample_indices(display_values.shape)
        if len(time_indices) != display_values.shape[0] or len(spatial_indices) != display_values.shape[1]:
            display_values = display_values[np.ix_(time_indices, spatial_indices)]
            display_labels = [display_labels[index] for index in time_indices if index < len(display_labels)]
            sampled_depth_coordinates = [
                depth_coordinates[index] if index < len(depth_coordinates) else None
                for index in spatial_indices
            ]
            spatial_coord = result.get("spatial_coord", [])
            sampled_result = dict(result)
            sampled_result["spatial_coord"] = [
                spatial_coord[index] if index < len(spatial_coord) else index
                for index in spatial_indices
            ]
            display_info.update(
                {
                    "workspaceSampled": True,
                    "workspaceOriginalShape": original_display_shape,
                    "workspaceSampledShape": [int(dim) for dim in display_values.shape],
                    "workspaceMaxMatrixPoints": workspace_max_matrix_points(),
                }
            )
    display_info.update(
        {
            "variable": metadata.get("variable") or "unknown",
            "units": metadata.get("units") or "",
            "depthIntegratedUnits": _depth_integrated_units(metadata.get("units") or ""),
            "depthAxisSource": depth_axis_source,
            "depthLevels": [value for value in sampled_depth_coordinates if value is not None],
        }
    )
    return {
        "rows": _build_hovmoller_rows_from_values(sampled_result, display_values, depth_coordinates=sampled_depth_coordinates),
        "time_labels": display_labels,
        "depth_integrated_series": _build_hovmoller_depth_integrated_series(
            sampled_result,
            display_values,
            display_labels,
            depth_coordinates=sampled_depth_coordinates,
        ),
        "display_info": display_info,
    }


def _build_hovmoller_rows_from_values(
    result: Dict[str, Any],
    values: np.ndarray,
    *,
    depth_coordinates: Optional[List[Optional[float]]] = None,
) -> List[Dict[str, Any]]:
    spatial_coord = result.get("spatial_coord", [])
    spatial_dim = result.get("metadata", {}).get("spatial_dim", "depth")
    if values.size == 0 or values.ndim != 2:
        return []

    if spatial_dim in {"depth", "z"} and depth_coordinates is None:
        depth_coordinates, _ = _resolve_hovmoller_depth_coordinates(result, values.shape[1])

    rows: List[Dict[str, Any]] = []
    for spatial_index in range(values.shape[1]):
        if spatial_dim in {"depth", "z"} and depth_coordinates is not None:
            coord_numeric = depth_coordinates[spatial_index] if spatial_index < len(depth_coordinates) else None
            coord_value = coord_numeric if coord_numeric is not None else (
                spatial_coord[spatial_index] if spatial_index < len(spatial_coord) else spatial_index
            )
        else:
            coord_value = spatial_coord[spatial_index] if spatial_index < len(spatial_coord) else spatial_index
            coord_numeric = _safe_float(coord_value)
        if spatial_dim in {"depth", "z"} and (coord_numeric is None or _is_hovmoller_sentinel_depth(coord_numeric)):
            continue
        row_array = np.asarray(values[:, spatial_index], dtype=float)
        if spatial_dim in {"depth", "z"} and not np.any(np.isfinite(row_array)):
            continue
        row_values = [
            float(value) if np.isfinite(value) else None
            for value in row_array.tolist()
        ]
        rows.append(
            {
                "depthLabel": _format_coord_label(coord_value, spatial_dim),
                "depthValue": coord_numeric,
                "values": row_values,
            }
        )

    return rows


def _build_hovmoller_depth_integrated_series(
    result: Dict[str, Any],
    values: np.ndarray,
    labels: List[str],
    *,
    depth_coordinates: Optional[List[Optional[float]]] = None,
) -> List[Dict[str, Any]]:
    spatial_dim = result.get("metadata", {}).get("spatial_dim", "depth")
    if spatial_dim not in {"depth", "z"} or values.size == 0 or values.ndim != 2:
        return []

    if depth_coordinates is None:
        depth_coordinates, _ = _resolve_hovmoller_depth_coordinates(result, values.shape[1])
    valid_indices: List[int] = []
    depths: List[float] = []
    for index in range(values.shape[1]):
        coord_numeric = depth_coordinates[index] if index < len(depth_coordinates) else None
        if coord_numeric is None or not np.isfinite(coord_numeric) or _is_hovmoller_sentinel_depth(coord_numeric):
            continue
        valid_indices.append(index)
        depths.append(abs(float(coord_numeric)))
    if not valid_indices:
        return []

    order = np.argsort(np.asarray(depths, dtype=float))
    depth_axis = np.asarray(depths, dtype=float)[order]
    matrix = np.asarray(values[:, valid_indices], dtype=float)[:, order]
    if matrix.shape[1] == 1:
        integrated = matrix[:, 0]
    else:
        interval_valid = np.isfinite(matrix[:, :-1]) & np.isfinite(matrix[:, 1:])
        dz = np.diff(depth_axis)
        interval_values = 0.5 * (matrix[:, :-1] + matrix[:, 1:]) * dz[None, :]
        integrated = np.sum(np.where(interval_valid, interval_values, 0.0), axis=1)
        integrated = np.where(np.any(interval_valid, axis=1), integrated, np.nan)

    return _build_named_series(
        labels=labels[: integrated.shape[0]],
        values=integrated.tolist(),
        label_key="label",
        value_key="value",
        limit=None,
    )


def _depth_integrated_units(units: Any) -> str:
    text = str(units or "").strip()
    return f"{text} m" if text else "value m"


def _build_section_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    values = as_numeric_array(result.get("values", []))
    distance_km = [float(value) for value in result.get("distance_km", [])]
    depth = [float(value) for value in result.get("depth", [])]
    times = [str(value) for value in result.get("time", [])]
    original_shape = [int(dim) for dim in values.shape]

    if values.ndim == 3:
        values = values[0]
        slice_label = f"Time slice: {times[0]}" if times else "Time slice: first"
    else:
        slice_label = ""

    rows: List[Dict[str, Any]] = []
    axis_title = "Section index"
    sampled = False

    if values.ndim == 2 and depth:
        row_indices, col_indices = matrix_sample_indices(values.shape)
        if len(row_indices) != values.shape[0] or len(col_indices) != values.shape[1]:
            sampled = True
            values = values[np.ix_(row_indices, col_indices)]
            depth = [depth[index] for index in row_indices if index < len(depth)]
            distance_km = [distance_km[index] for index in col_indices if index < len(distance_km)]
        axis_title = "Depth (m)"
        for index, depth_value in enumerate(depth):
            if index >= values.shape[0]:
                break
            rows.append(
                {
                    "label": f"{depth_value:g} m",
                    "coordValue": float(depth_value),
                    "values": [float(value) if np.isfinite(value) else float("nan") for value in values[index]],
                }
            )
    elif values.ndim == 2 and times:
        row_indices, col_indices = matrix_sample_indices(values.shape)
        if len(row_indices) != values.shape[0] or len(col_indices) != values.shape[1]:
            sampled = True
            values = values[np.ix_(row_indices, col_indices)]
            times = [times[index] for index in row_indices if index < len(times)]
            distance_km = [distance_km[index] for index in col_indices if index < len(distance_km)]
        axis_title = "Time"
        for index, label in enumerate(times):
            if index >= values.shape[0]:
                break
            rows.append(
                {
                    "label": label,
                    "values": [float(value) if np.isfinite(value) else float("nan") for value in values[index]],
                }
            )
    elif values.ndim == 1:
        rows.append(
            {
                "label": "Section",
                "values": [float(value) if np.isfinite(value) else float("nan") for value in values],
            }
        )
        axis_title = "Section"

    return {
        "rows": rows,
        "distance_km": distance_km,
        "axis_title": axis_title,
        "slice_label": slice_label,
        "display_info": {
            "workspaceSampled": sampled,
            "workspaceOriginalShape": original_shape,
            "workspaceSampledShape": [int(dim) for dim in values.shape],
            "workspaceMaxMatrixPoints": workspace_max_matrix_points(),
        },
    }


def _build_histogram_bins(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    labels = result.get("bin_centers", [])
    values = result.get("density", [])
    return _build_named_series(
        labels=[_format_coord_label(label, "value") for label in labels],
        values=values,
        label_key="label",
        value_key="value",
        limit=10,
    )


def _build_ts_diagram_payload(
    result: Dict[str, Any],
    max_points: int = 2500,
) -> Dict[str, Any]:
    temperature = np.asarray(result.get("temperature", []), dtype=float)
    salinity = np.asarray(result.get("salinity", []), dtype=float)
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    point_classes = result.get("point_classes")

    payload: Dict[str, Any] = {
        "tsDiagramPoints": [],
        "tsDiagramTemperatureLabel": _humanize_data_label(metadata.get("temperature_variable")) or "Temperature",
        "tsDiagramSalinityLabel": _humanize_data_label(metadata.get("salinity_variable")) or "Salinity",
        "tsDiagramColorLabel": _humanize_data_label(metadata.get("color_variable")) if metadata.get("color_variable") else None,
        "tsDiagramColorRange": None,
        "tsDiagramPointClasses": [],
        "tsDiagramClassColorMap": _json_safe(metadata.get("class_color_map") or {}),
        "tsDiagramWatermassBins": _json_safe(metadata.get("watermass_bins") or []),
    }

    if temperature.size == 0 or salinity.size == 0 or temperature.size != salinity.size:
        return payload

    valid_mask = np.isfinite(temperature) & np.isfinite(salinity)
    color_values = None
    if isinstance(result.get("color_values"), list):
        color_array = np.asarray(result.get("color_values", []), dtype=float)
        if color_array.size == temperature.size:
            valid_mask &= np.isfinite(color_array)
            color_values = color_array

    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return payload

    if valid_indices.size > max_points:
        sampled_positions = _sample_indices(valid_indices.size, limit=max_points)
        valid_indices = np.asarray([valid_indices[index] for index in sampled_positions], dtype=int)

    points: List[Dict[str, Any]] = []
    for index in valid_indices:
        point: Dict[str, Any] = {
            "temperature": float(temperature[index]),
            "salinity": float(salinity[index]),
        }
        if color_values is not None:
            point["colorValue"] = float(color_values[index])
        if isinstance(point_classes, list) and index < len(point_classes) and isinstance(point_classes[index], str):
            point["pointClass"] = point_classes[index]
        points.append(point)

    payload["tsDiagramPoints"] = points
    payload["tsDiagramPointClasses"] = [
        point["pointClass"]
        for point in points
        if isinstance(point.get("pointClass"), str)
    ]
    color_range = metadata.get("color_range")
    if isinstance(color_range, (list, tuple)) and len(color_range) == 2:
        payload["tsDiagramColorRange"] = [float(color_range[0]), float(color_range[1])]
    return payload


def _build_regression_map_payload(
    result: Dict[str, Any],
    title: str,
) -> Optional[Dict[str, Any]]:
    lon = as_numeric_array(result.get("lon", []))
    lat = as_numeric_array(result.get("lat", []))
    slope = as_numeric_array(result.get("slope", []))
    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}

    if lon.size == 0 or lat.size == 0 or slope.size == 0:
        return None

    valid = slope[np.isfinite(slope)]
    statistics: Dict[str, float] = {}
    if valid.size:
        statistics = {
            "mean": float(np.nanmean(valid)),
            "std": float(np.nanstd(valid)),
            "min": float(np.nanmin(valid)),
            "max": float(np.nanmax(valid)),
        }

    variable_label = _humanize_data_label(metadata.get("variable")) or "Regression"
    return _build_map_field_payload(
        spatial_field={
            "lon": lon,
            "lat": lat,
            "values": slope,
            "metadata": {
                "variable": f"{metadata.get('variable', 'regression')}_slope",
                "units": metadata.get("units", ""),
                "statistics": statistics,
                "time_range": metadata.get("time_range"),
            },
        },
        title=title or f"{variable_label} Regression Slope",
        time_range=_infer_time_range_from_result(result),
    )


def _build_field_trend_map_payload(
    result: Dict[str, Any],
    title: str,
) -> Optional[Dict[str, Any]]:
    slope_field = result.get("data")
    if slope_field is None or not hasattr(slope_field, "dims"):
        return None
    if "lat" not in slope_field.dims or "lon" not in slope_field.dims:
        return None

    field = slope_field
    extra_dims = [dim for dim in field.dims if dim not in {"lat", "lon"}]
    if extra_dims:
        field = _mean_xarray_like(field, extra_dims)
    if "lat" not in getattr(field, "dims", ()) or "lon" not in getattr(field, "dims", ()):
        return None

    values = as_numeric_array(field.values)
    if values.ndim != 2 or values.size == 0:
        return None

    valid = values[np.isfinite(values)]
    statistics: Dict[str, float] = {}
    if valid.size:
        statistics = {
            "mean": float(np.nanmean(valid)),
            "std": float(np.nanstd(valid)),
            "min": float(np.nanmin(valid)),
            "max": float(np.nanmax(valid)),
        }

    metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
    variable = str(metadata.get("variable") or getattr(field, "name", None) or "field").strip()
    variable_label = _humanize_data_label(variable) or variable.replace("_", " ").title()
    units = str(field.attrs.get("units") or metadata.get("units") or "").strip()
    return _build_map_field_payload(
        spatial_field={
            "lon": as_numeric_array(field["lon"].values),
            "lat": as_numeric_array(field["lat"].values),
            "values": values,
            "metadata": {
                "variable": f"{variable}_trend_slope",
                "units": units,
                "statistics": statistics,
                "time_range": metadata.get("time_range"),
                "confidence_level": metadata.get("confidence_level"),
                "time_unit": metadata.get("time_unit"),
            },
        },
        title=title or f"{variable_label} Trend Slope Map",
        time_range=_infer_time_range_from_result(result),
        max_lon_points=100,
        max_lat_points=100,
    )


def _build_eof_mode_map(
    spatial_pattern: Any,
    title: str,
    variance_label: str,
) -> Optional[Dict[str, Any]]:
    if spatial_pattern is None or not hasattr(spatial_pattern, "dims"):
        return None
    if "lat" not in spatial_pattern.dims or "lon" not in spatial_pattern.dims:
        return None

    values = as_numeric_array(spatial_pattern.values)
    valid = values[np.isfinite(values)]
    statistics: Dict[str, float] = {}
    if valid.size:
        statistics = {
            "mean": float(np.nanmean(valid)),
            "std": float(np.nanstd(valid)),
            "min": float(np.nanmin(valid)),
            "max": float(np.nanmax(valid)),
        }

    map_field = _build_map_field_payload(
        spatial_field={
            "lon": as_numeric_array(spatial_pattern.lon.values),
            "lat": as_numeric_array(spatial_pattern.lat.values),
            "values": values,
            "metadata": {
                "variable": str(spatial_pattern.name or "eof_mode"),
                "units": str(spatial_pattern.attrs.get("units", "")),
                "statistics": statistics,
            },
        },
        title=title,
        time_range=("", ""),
        max_lon_points=80,
        max_lat_points=80,
    )
    map_field["timeLabel"] = variance_label
    return map_field


def _build_eof_payload(result: Dict[str, Any]) -> tuple[List[Dict[str, str]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    modes = result.get("modes", [])
    variance = []
    pc_series: List[Dict[str, Any]] = []
    eof_modes: List[Dict[str, Any]] = []

    for mode in modes[:3]:
        mode_number = mode.get("mode_number")
        variance_explained = mode.get("variance_explained")
        if mode_number is None or variance_explained is None:
            continue
        variance_label = f"{float(variance_explained):.1f}% variance"
        variance.append(
            {
                "label": f"Mode {mode_number}",
                "value": variance_label,
            }
        )

        mode_pc_series: List[Dict[str, Any]] = []
        time_series = mode.get("time_series")
        if time_series is not None and hasattr(time_series, "dims") and "time" in time_series.dims:
            mode_pc_series = _build_named_series(
                labels=_coord_labels(time_series, "time"),
                values=np.asarray(time_series.values, dtype=float).tolist(),
                label_key="day",
                value_key="value",
                limit=None,
            )
            if mode_number == 1:
                pc_series = mode_pc_series

        spatial_pattern = mode.get("spatial_pattern")
        map_field = _build_eof_mode_map(
            spatial_pattern=spatial_pattern,
            title=f"EOF Mode {mode_number}",
            variance_label=variance_label,
        )
        if map_field is not None:
            eof_modes.append(
                {
                    "id": f"mode_{mode_number}",
                    "title": f"EOF Mode {mode_number}",
                    "varianceLabel": variance_label,
                    "mapField": map_field,
                    "pcSeries": mode_pc_series,
                }
            )

    return variance, pc_series, eof_modes


def _infer_time_range_from_result(result: Dict[str, Any]) -> tuple[str, str]:
    metadata = result.get("metadata", {})
    if isinstance(metadata, dict):
        time_range = metadata.get("time_range")
        if isinstance(time_range, (list, tuple)) and len(time_range) == 2:
            return str(time_range[0]), str(time_range[1])
    return "", ""


def _build_event_overlays(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = result.get("events", [])
    event_type = str(result.get("event_type") or "event")
    overlays: List[Dict[str, Any]] = []

    if not isinstance(events, list):
        return overlays

    # Preserve the full event catalog so event cards, summaries, and main-map overlays
    # reflect the same total count instead of a truncated sample.
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        center = _extract_event_center(event)
        if center is None:
            continue
        lon = _finite_event_float(center.get("lon"))
        lat = _finite_event_float(center.get("lat"))
        if lon is None or lat is None:
            continue

        overlay: Dict[str, Any] = {
            "id": str(event.get("event_id") or event.get("track_id") or f"{event_type}_{index + 1}"),
            "eventType": event_type,
            "title": _event_overlay_title(event_type, event, index),
            "center": {
                "lat": lat,
                "lon": lon,
            },
            "details": _event_detail_lines(event),
        }

        if event_type in {"front", "jet", "meander"}:
            shape_payload = _extract_event_shape(event_type, event)
            if shape_payload:
                overlay.update(shape_payload)
            else:
                bounds = _extract_event_bounds(event)
                if bounds is not None:
                    overlay["shape"] = "rectangle"
                    overlay["bounds"] = bounds
        else:
            bounds = _extract_event_bounds(event)
            if bounds is not None:
                overlay["shape"] = "rectangle"
                overlay["bounds"] = bounds
            else:
                overlay.update(_extract_event_shape(event_type, event))

        if isinstance(event.get("severity"), str):
            overlay["severity"] = str(event["severity"])
        if isinstance(event.get("timestamp"), str):
            overlay["timestamp"] = str(event["timestamp"])
        if isinstance(event.get("end_timestamp"), str):
            overlay["endTimestamp"] = str(event["end_timestamp"])

        overlays.append(overlay)

    return overlays


def _extract_event_shape(event_type: str, event: Dict[str, Any]) -> Dict[str, Any]:
    if event_type == "eddy":
        radius_km = event.get("radius_km")
        if isinstance(radius_km, (int, float)) and float(radius_km) > 0:
            return {"shape": "circle", "radiusKm": float(radius_km)}

    if event_type in {"front", "jet", "meander", "eddy_track"}:
        path = _extract_event_path(event_type, event)
        if path:
            return {"shape": "polyline", "path": path}

    default_symbol = {
        "front": "diamond",
        "jet": "square",
        "meander": "diamond",
        "eddy": "diamond",
    }
    return {"shape": "point", "symbol": default_symbol.get(event_type, "triangle")}


def _extract_event_path(event_type: str, event: Dict[str, Any]) -> Optional[List[Dict[str, float]]]:
    if event_type == "eddy_track":
        raw_path = event.get("path")
        if isinstance(raw_path, list):
            path: List[Dict[str, float]] = []
            for point in raw_path:
                if not isinstance(point, dict):
                    continue
                lon = point.get("lon")
                lat = point.get("lat")
                if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
                    path.append({"lon": float(lon), "lat": float(lat)})
            if len(path) >= 2:
                return path

    raw_path = event.get("path_coordinates")
    if isinstance(raw_path, list):
        path: List[Dict[str, float]] = []
        for point in raw_path:
            if not isinstance(point, dict):
                continue
            lon = point.get("lon")
            lat = point.get("lat")
            if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
                path.append({"lon": float(lon), "lat": float(lat)})
        if len(path) >= 2:
            return path

    center = event.get("center")
    orientation = event.get("orientation_deg")
    length_km = event.get("length_km")
    if (
        isinstance(center, dict)
        and isinstance(center.get("lon"), (int, float))
        and isinstance(center.get("lat"), (int, float))
        and isinstance(orientation, (int, float))
        and isinstance(length_km, (int, float))
        and float(length_km) > 0
    ):
        bounded_length_km = float(length_km)
        bbox = event.get("bbox")
        if isinstance(bbox, dict):
            diagonal_km = _bbox_diagonal_km(bbox)
            if diagonal_km is not None:
                bounded_length_km = min(bounded_length_km, diagonal_km * 1.05)
        segment = _oriented_segment(
            center_lon=float(center["lon"]),
            center_lat=float(center["lat"]),
            length_km=bounded_length_km,
            orientation_deg=float(orientation),
        )
        if isinstance(bbox, dict):
            return _clip_path_to_bbox(segment, bbox)
        return segment

    if event_type == "meander" and isinstance(center, dict):
        lon = center.get("lon")
        lat = center.get("lat")
        if isinstance(lon, (int, float)) and isinstance(lat, (int, float)) and isinstance(length_km, (int, float)):
            amplitude_km = event.get("amplitude_km")
            return _wavy_segment(
                center_lon=float(lon),
                center_lat=float(lat),
                length_km=float(length_km),
                amplitude_km=float(amplitude_km) if isinstance(amplitude_km, (int, float)) else max(float(length_km) * 0.1, 10.0),
            )

    return None


def _finite_event_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _first_finite_event_value(mapping: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key not in mapping:
            continue
        numeric = _finite_event_float(mapping.get(key))
        if numeric is not None:
            return numeric
    return None


def _extract_bbox_edges(bbox: Dict[str, Any]) -> Optional[tuple[float, float, float, float]]:
    lon_min = _first_finite_event_value(bbox, ("lon_min", "lonMin", "min_lon", "minLon"))
    lon_max = _first_finite_event_value(bbox, ("lon_max", "lonMax", "max_lon", "maxLon"))
    lat_min = _first_finite_event_value(bbox, ("lat_min", "latMin", "min_lat", "minLat"))
    lat_max = _first_finite_event_value(bbox, ("lat_max", "latMax", "max_lat", "maxLat"))
    if lon_min is None or lon_max is None or lat_min is None or lat_max is None:
        return None
    return lon_min, lon_max, lat_min, lat_max


def _extract_point_from_mapping(point: Dict[str, Any]) -> Optional[Dict[str, float]]:
    lon = _first_finite_event_value(point, ("lon", "lng", "longitude", "center_lon", "centroid_lon"))
    lat = _first_finite_event_value(point, ("lat", "latitude", "center_lat", "centroid_lat"))
    if lon is None or lat is None:
        return None
    return {"lon": lon, "lat": lat}


def _extract_event_center(event: Dict[str, Any]) -> Optional[Dict[str, float]]:
    center = event.get("center")
    if isinstance(center, dict):
        point = _extract_point_from_mapping(center)
        if point is not None:
            return point

    centroid = event.get("centroid")
    if isinstance(centroid, dict):
        point = _extract_point_from_mapping(centroid)
        if point is not None:
            return point

    direct_point = _extract_point_from_mapping(event)
    if direct_point is not None:
        return direct_point

    path = event.get("path") or event.get("path_coordinates")
    if isinstance(path, list) and path:
        points = [_extract_point_from_mapping(point) for point in path if isinstance(point, dict)]
        points = [point for point in points if point is not None]
        if points:
            midpoint = points[len(points) // 2]
            return {"lon": midpoint["lon"], "lat": midpoint["lat"]}

    bbox = event.get("bbox")
    if isinstance(bbox, dict):
        edges = _extract_bbox_edges(bbox)
        if edges is not None:
            lon_min, lon_max, lat_min, lat_max = edges
            return {"lon": (lon_min + lon_max) / 2.0, "lat": (lat_min + lat_max) / 2.0}

    return None


def _event_overlay_title(event_type: str, event: Dict[str, Any], index: int) -> str:
    if event_type == "eddy_track" and isinstance(event.get("track_id"), str):
        return str(event["track_id"])
    return f"{_humanize_feature_name(event_type) or 'Event'} {index + 1}"


def _bbox_diagonal_km(bbox: Dict[str, Any]) -> Optional[float]:
    edges = _extract_bbox_edges(bbox)
    if edges is None:
        return None

    lon_min, lon_max, lat_min, lat_max = edges
    mean_lat = (lat_min + lat_max) / 2.0
    dx_km = abs(lon_max - lon_min) * max(111.32 * np.cos(np.deg2rad(mean_lat)), 1e-6)
    dy_km = abs(lat_max - lat_min) * 110.57
    return float(np.hypot(dx_km, dy_km))


def _clip_path_to_bbox(path: List[Dict[str, float]], bbox: Dict[str, Any]) -> List[Dict[str, float]]:
    edges = _extract_bbox_edges(bbox)
    if edges is None:
        return path

    lon_min, lon_max, lat_min, lat_max = edges
    clipped: List[Dict[str, float]] = []
    for point in path:
        lon = _finite_event_float(point.get("lon"))
        lat = _finite_event_float(point.get("lat"))
        if lon is None or lat is None:
            continue
        clipped.append(
            {
                "lon": float(min(max(lon, lon_min), lon_max)),
                "lat": float(min(max(lat, lat_min), lat_max)),
            }
        )
    return clipped


def _oriented_segment(center_lon: float, center_lat: float, length_km: float, orientation_deg: float) -> List[Dict[str, float]]:
    half_length = max(length_km / 2.0, 1.0)
    angle_rad = np.deg2rad(orientation_deg)
    dx_km = half_length * np.cos(angle_rad)
    dy_km = half_length * np.sin(angle_rad)
    lon_scale = max(111.32 * np.cos(np.deg2rad(center_lat)), 1e-6)
    lat_scale = 110.57
    lon_offset = dx_km / lon_scale
    lat_offset = dy_km / lat_scale
    return [
        {"lon": center_lon - lon_offset, "lat": center_lat - lat_offset},
        {"lon": center_lon + lon_offset, "lat": center_lat + lat_offset},
    ]


def _wavy_segment(center_lon: float, center_lat: float, length_km: float, amplitude_km: float) -> List[Dict[str, float]]:
    lon_scale = max(111.32 * np.cos(np.deg2rad(center_lat)), 1e-6)
    lat_scale = 110.57
    half_length_deg = (length_km / 2.0) / lon_scale
    amplitude_deg = amplitude_km / lat_scale
    samples = np.linspace(-1.0, 1.0, 7)
    return [
        {
            "lon": float(center_lon + sample * half_length_deg),
            "lat": float(center_lat + np.sin(sample * np.pi) * amplitude_deg),
        }
        for sample in samples
    ]


def _extract_event_bounds(event: Dict[str, Any]) -> Optional[Dict[str, float]]:
    bbox = event.get("bbox")
    if not isinstance(bbox, dict):
        return None

    edges = _extract_bbox_edges(bbox)
    if edges is None:
        return None

    lon_min, lon_max, lat_min, lat_max = edges
    return {
        "lonMin": lon_min,
        "lonMax": lon_max,
        "latMin": lat_min,
        "latMax": lat_max,
    }


def _event_detail_lines(event: Dict[str, Any]) -> List[str]:
    detail_fields = [
        ("timestamp", "Date"),
        ("end_timestamp", "End"),
        ("duration_days", "Duration (days)"),
        ("severity", "Severity"),
        ("type", "Type"),
        ("bloom_type", "Bloom type"),
        ("area_km2", "Area (km²)"),
        ("radius_km", "Radius (km)"),
        ("length_km", "Length (km)"),
        ("width_km", "Width (km)"),
        ("amplitude_km", "Amplitude (km)"),
        ("aspect_ratio", "Aspect ratio"),
        ("orientation_deg", "Orientation (deg)"),
        ("mean_intensity", "Mean intensity"),
        ("max_intensity", "Max intensity"),
        ("intensity", "Intensity"),
        ("max_gradient", "Max gradient"),
        ("mean_gradient", "Mean gradient"),
        ("value_range", "Value range"),
        ("mean_speed", "Mean speed"),
        ("max_speed", "Max speed"),
        ("mean_direction_deg", "Direction (deg)"),
        ("mean_curvature", "Mean curvature"),
        ("max_curvature", "Max curvature"),
        ("max_vorticity", "Max vorticity"),
        ("mean_oxygen", "Mean oxygen"),
        ("min_oxygen", "Min oxygen"),
        ("mean_temp", "Mean temp"),
        ("max_temp", "Max temp"),
        ("min_temp", "Min temp"),
        ("mean_chlorophyll", "Mean chlorophyll"),
        ("max_chlorophyll", "Max chlorophyll"),
        ("n_pixels", "Pixels"),
    ]

    lines: List[str] = []
    for field_name, label in detail_fields:
        value = event.get(field_name)
        if value is None:
            continue
        if isinstance(value, float):
            rendered = f"{value:.4g}"
        else:
            rendered = str(value)
        lines.append(f"{label}: {rendered}")
    return lines[:8]


def _build_named_series(
    labels: List[Any],
    values: List[Any],
    label_key: str,
    value_key: str,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    labels_list = list(labels)
    values_array = np.asarray(values, dtype=float).reshape(-1) if len(values) else np.asarray([], dtype=float)
    if values_array.size == 0:
        return []

    sample_limit = min(len(labels_list), values_array.size) if limit is None else limit
    sample_indices = _sample_indices(min(len(labels_list), values_array.size), limit=sample_limit)
    rows: List[Dict[str, Any]] = []
    for index in sample_indices:
        value = values_array[index]
        if not np.isfinite(value):
            continue
        label = labels_list[index] if index < len(labels_list) else f"{index + 1}"
        rows.append({label_key: _short_label(label), value_key: float(value)})
    return rows


def _build_pair_series(
    x_values: List[Any],
    y_values: List[Any],
    x_key: str,
    y_key: str,
    limit: int,
) -> List[Dict[str, Any]]:
    x_list = list(x_values)
    y_array = np.asarray(y_values, dtype=float).reshape(-1) if len(y_values) else np.asarray([], dtype=float)
    if y_array.size == 0:
        return []

    sample_indices = _sample_indices(min(len(x_list), y_array.size), limit=limit)
    rows: List[Dict[str, Any]] = []
    for index in sample_indices:
        value = y_array[index]
        if not np.isfinite(value):
            continue
        rows.append({x_key: float(x_list[index]), y_key: float(value)})
    return rows


def _sample_indices(length: int, limit: int) -> List[int]:
    if length <= 0:
        return []
    if length <= limit:
        return list(range(length))
    raw = np.linspace(0, length - 1, num=limit)
    return sorted({int(round(value)) for value in raw})


def _coord_labels(data: Any, dim: str) -> List[str]:
    if not hasattr(data, "coords") or dim not in data.coords:
        return [str(index + 1) for index in range(int(np.asarray(data.values).size))]
    return [_format_coord_label(value, dim) for value in np.asarray(data[dim].values)]


def _format_coord_label(value: Any, dim: str) -> str:
    if dim == "time":
        if isinstance(value, np.datetime64):
            return np.datetime_as_string(value, unit="D")
        return _short_label(value)
    if dim in {"depth", "z", "lev", "level"}:
        try:
            return f"{float(value):.0f} m"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (float, int)):
        return f"{float(value):.3g}"
    return str(value)


def _short_label(value: Any) -> str:
    text = str(value)
    if "T" in text:
        text = text.split("T", 1)[0]
    return text


def _get_feature_name(result: Dict[str, Any]) -> Optional[str]:
    metadata = result.get("metadata")
    if isinstance(metadata, dict) and metadata.get("feature"):
        return str(metadata["feature"])

    data = result.get("data")
    attrs = getattr(data, "attrs", {})
    if isinstance(attrs, dict) and attrs.get("feature"):
        return str(attrs["feature"])
    return None


def _humanize_feature_name(feature_name: Optional[str]) -> Optional[str]:
    if not feature_name:
        return None
    return feature_name.replace("_", " ").strip().title()


def _extract_feature_depth_scalar(result: Dict[str, Any]) -> Optional[float]:
    data = result.get("data")
    if data is None or not hasattr(data, "values"):
        return None
    if _skip_auto_workspace_data_container_compute(data):
        return None
    values = np.asarray(data.values, dtype=float)
    if values.size == 0:
        return None
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return None
    return float(np.nanmean(valid))


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return _json_safe(json_safe_array(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.datetime64):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return json.loads(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False))
