"""Analysis-DAG decision layer for OceanMind harness planning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from packages.harness.data_scope import infer_lag_variables
from packages.harness.manual_loader import WorkflowTemplate


class PlanRoute(str, Enum):
    DATASET_INFO = "dataset_info"
    SKILL_WORKFLOW = "skill_workflow"
    MANUAL_RECIPE = "manual_recipe"
    CONDITION_MASK_SPATIAL_MAP = "condition_mask_spatial_map"
    PAIR_LAG_RELATIONSHIP = "pair_lag_relationship"
    HYPOXIA_DRIVER = "hypoxia_driver"
    GENERIC_TIMESERIES = "generic_timeseries"
    GENERATED_CODE = "generated_code"
    GENERIC_MAP = "generic_map"


@dataclass(frozen=True)
class AnalysisDAGDecision:
    route: PlanRoute
    selected_workflow: Optional[WorkflowTemplate] = None
    include_trend: bool = False
    include_spectrum: bool = False
    reason: str = ""

    @property
    def selected_recipe(self) -> Optional[WorkflowTemplate]:
        return self.selected_workflow


class AnalysisDAGBuilder:
    """Choose the analysis graph family before binding tools or code."""

    def decide(
        self,
        query: str,
        *,
        selected_workflow: Optional[WorkflowTemplate] = None,
        selected_recipe: Optional[WorkflowTemplate] = None,
    ) -> AnalysisDAGDecision:
        workflow = selected_workflow or selected_recipe
        if looks_like_dataset_info_request(query):
            return AnalysisDAGDecision(PlanRoute.DATASET_INFO, reason="dataset metadata request")
        if workflow is not None:
            return AnalysisDAGDecision(PlanRoute.SKILL_WORKFLOW, selected_workflow=workflow, reason="strong skill workflow match")
        if looks_like_condition_mask_spatial_map_request(query):
            return AnalysisDAGDecision(PlanRoute.CONDITION_MASK_SPATIAL_MAP, reason="condition mask spatial projection")
        if looks_like_pair_lag_relationship_request(query):
            return AnalysisDAGDecision(PlanRoute.PAIR_LAG_RELATIONSHIP, reason="two-variable lag relationship")
        if looks_like_hypoxia_driver_request(query):
            return AnalysisDAGDecision(PlanRoute.HYPOXIA_DRIVER, include_trend=True, include_spectrum=True, reason="complex hypoxia driver diagnosis")
        if looks_like_trend_request(query):
            return AnalysisDAGDecision(
                PlanRoute.GENERIC_TIMESERIES,
                include_trend=True,
                include_spectrum=looks_like_spectrum_request(query),
                reason="trend request over a data field",
            )
        if looks_like_timeseries_request(query):
            return AnalysisDAGDecision(
                PlanRoute.GENERIC_TIMESERIES,
                include_trend=False,
                include_spectrum=looks_like_spectrum_request(query),
                reason="time-series request over a data field",
            )
        if looks_like_code_fallback_request(query):
            return AnalysisDAGDecision(PlanRoute.GENERATED_CODE, reason="analysis request without covered tool recipe")
        return AnalysisDAGDecision(PlanRoute.GENERIC_MAP, reason="default spatial field projection")


def looks_like_dataset_info_request(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(r"\b(dataset|metadata|available variables|data info)\b|数据集|变量有哪些|元数据", lowered))


def looks_like_hypoxia_driver_request(text: str) -> bool:
    lowered = text.lower()
    has_hypoxia = bool(re.search(r"\b(hypoxia|hypoxic|low oxygen|oxygen)\b|缺氧|低氧|溶解氧", lowered))
    has_complex = bool(
        re.search(
            r"\b(trend|spectrum|power spectrum|correlation|relationship|driver|temperature|salinity|speed)\b|趋势|功率谱|谱|相关|关系|温度|盐度|流速",
            lowered,
        )
    )
    return has_hypoxia and has_complex


def looks_like_pair_lag_relationship_request(text: str) -> bool:
    lowered = text.lower()
    has_lag = bool(re.search(r"\b(lag|lags|lead|leads|time lag|cross[- ]correlation|delayed)\b|滞后|领先", lowered))
    has_relationship = bool(re.search(r"\b(related|relationship|correlation|coupling|between|with)\b|相关|关系|联系|耦合", lowered))
    variable1, variable2 = infer_lag_variables(text)
    return has_lag and has_relationship and bool(variable1 and variable2 and variable1 != variable2)


def looks_like_condition_mask_spatial_map_request(text: str) -> bool:
    lowered = text.lower()
    has_condition = bool(
        re.search(
            r"\b(?:oxygen|chlorophyll|chl|temp|temperature|sst|salt|salinity)\b\s*"
            r"(?:<=|>=|<|>|below|under|less than|above|over|greater than)\s*"
            r"(?:p\d{1,3}|threshold|[-+]?\d+(?:\.\d+)?)",
            lowered,
        )
    )
    if not has_condition:
        return False
    if re.search(
        r"\b(trend|time[- ]?series|timeseries|spectrum|correlation|lag|custom|index|detect|detection|event|events|bloom|heatwave|hypoxia|upwelling|eutrophication)\b",
        lowered,
    ):
        return False
    return bool(
        re.search(r"\b(show|map|plot|display|visualize|region|regions|area|areas)\b|显示|区域|地图|画图|可视化", lowered)
    )


def looks_like_code_fallback_request(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\b(map|plot|show|visualize|display)\b|画图|地图|显示|可视化", lowered):
        return False
    return bool(
        re.search(
            r"\b(custom|index|metric|diagnose|diagnostic|estimate|calculate|compute|derive|relationship|related|compare|association)\b|"
            r"自定义|指标|诊断|估计|计算|推导|关系|比较|关联",
            lowered,
        )
    )


def looks_like_trend_request(text: str) -> bool:
    return bool(re.search(r"\btrend|linear trend|long[- ]term\b|趋势|长期变化", text.lower()))


def looks_like_spectrum_request(text: str) -> bool:
    return bool(re.search(r"\bspectrum|power spectrum|periodic|周期\b|功率谱|谱", text.lower()))


def looks_like_timeseries_request(text: str) -> bool:
    return bool(re.search(r"\btime series|timeseries|变化序列|时间序列|随时间", text.lower()))
