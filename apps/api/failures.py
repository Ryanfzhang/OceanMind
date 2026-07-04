from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional


FailureKind = Literal["capability_boundary", "llm_format", "planning", "execution", "synthesis", "transport"]


def _query_prefers_chinese(query: Optional[str]) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", query or ""))


def _looks_like_llm_format_error(raw_error: Any) -> bool:
    text = str(raw_error or "").lower()
    return any(
        marker in text
        for marker in (
            "does not contain valid json",
            "not valid json",
            "valid json",
            "jsondecodeerror",
            "expecting value",
            "expecting ',' delimiter",
            "unterminated string",
            "does not contain text content",
            "response is not valid json",
            "previous response failed json",
            "failed json/schema validation",
            "schema validation",
            "fenced yaml/json",
            "fenced yaml",
            "fenced json",
            "fenced contract",
            "non-object payload",
            "task graph contract",
            "structured workflow contract",
            "planner agent response did not include",
            "planner skill selector returned",
            "planner skill selector failed",
            "planner task-graph generator failed",
            "final routing contract",
            "final workflow contract",
            "reasoning_content",
        )
    )


def _looks_like_code_agent_error(raw_error: Any) -> bool:
    text = str(raw_error or "").lower()
    return any(
        marker in text
        for marker in (
            "codeagent",
            "generated code",
            "generated-code",
            "generated python",
            "generated_python_analysis",
        )
    )


def _looks_like_capability_boundary(raw_error: Any) -> bool:
    text = str(raw_error or "").lower()
    return any(
        marker in text
        for marker in (
            "not supported",
            "unsupported",
            "no applicable skill",
            "unknown skill",
            "unknown tool",
            "not among the supported",
            "supported derived fields",
            "could not be accommodated",
            "cannot be accommodated",
            "outside the supported",
            "unable to build an executable plan",
            "cannot be executed by the available tools",
        )
    )


def _classify_failure_kind(raw_error: Any, default_kind: FailureKind) -> FailureKind:
    if _looks_like_llm_format_error(raw_error):
        return "llm_format"
    if _looks_like_capability_boundary(raw_error):
        return "capability_boundary"
    return default_kind


def _normalize_missing_field_label(value: Any) -> Optional[str]:
    text = str(value or "").strip().strip("'\"`.,;:")
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"resolved_scope", "required", "keys", "field", "fields", "contract", "planner"}:
        return None
    if "lon_range" in lowered or "lat_range" in lowered or "region" in lowered:
        return "region.lon_range/lat_range"
    if "time_range" in lowered or lowered in {"time", "date_range", "date"}:
        return "time_range"
    if "variable1" in lowered or "variable2" in lowered:
        return "variable1/variable2"
    if "variable" in lowered or "variables" in lowered:
        return "variable/variables"
    if "depth" in lowered or "vertical" in lowered or "layer" in lowered:
        return "depth/layer"
    if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_./-]*", text):
        return text
    return None


def _missing_fields_from_error(raw_error: Any) -> List[str]:
    text = str(raw_error or "")
    if not text.strip():
        return []
    lowered = text.lower()
    if _looks_like_code_agent_error(raw_error):
        return []
    if re.search(r"\bnot\s+a\s+missing\s+(?:variable|field|region|time|time range)\b", lowered):
        return []
    if "did not report any specific missing fields" in lowered:
        return []
    if "invalid workflow python block" in lowered or "workflow_code" in lowered and "invalid syntax" in lowered:
        return []

    candidates: List[str] = []
    for pattern in (
        r"missing_fields?\s*[:=]\s*\[([^\]]+)\]",
        r"missing required keys?:\s*([^.;\n]+)",
        r"missing resolved_scope\s+([a-zA-Z0-9_./-]+)",
        r"missing\s+([a-zA-Z0-9_./-]+)",
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            chunk = match.group(1)
            candidates.extend(re.split(r"[,，]\s*", chunk))

    if "lon_range/lat_range" in lowered or ("lon_range" in lowered and "lat_range" in lowered):
        candidates.append("region.lon_range/lat_range")
    if "time_range" in lowered:
        candidates.append("time_range")
    if (
        "variable/variables" in lowered
        or "missing resolved_scope variable" in lowered
        or "missing resolved_scope variables" in lowered
        or re.search(r"\bmissing\s+variable(?:s)?\b", lowered)
    ):
        candidates.append("variable/variables")
    if ("depth" in lowered or "vertical" in lowered or "layer" in lowered) and "missing" in lowered:
        candidates.append("depth/layer")

    normalized: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        label = _normalize_missing_field_label(candidate)
        if label and label not in seen:
            seen.add(label)
            normalized.append(label)
    return normalized


def _missing_field_display(field: str, *, chinese: bool) -> str:
    mapping_en = {
        "region.lon_range/lat_range": "region bounds (lon_range and lat_range)",
        "time_range": "time_range",
        "variable/variables": "variable or variables",
        "variable1/variable2": "both variables to compare",
        "depth/layer": "depth or layer",
    }
    mapping_zh = {
        "region.lon_range/lat_range": "空间范围 lon_range/lat_range",
        "time_range": "时间范围 time_range",
        "variable/variables": "变量 variable/variables",
        "variable1/variable2": "需要比较的两个变量 variable1/variable2",
        "depth/layer": "深度或层位 depth/layer",
    }
    return (mapping_zh if chinese else mapping_en).get(field, field)


def _missing_fields_failure_message(fields: List[str], *, chinese: bool) -> str:
    display = ", ".join(_missing_field_display(field, chinese=chinese) for field in fields)
    if chinese:
        return f"这个请求还缺少后端实际需要的字段：{display}。请只补充这些缺失项后再试。"
    return f"This request is missing the fields the backend actually needs: {display}. Please provide those fields and try again."


def _planner_format_failure_message(raw_error: Any, *, chinese: bool) -> str:
    text = str(raw_error or "").strip()
    lowered = text.lower()
    if "skill selector" in lowered:
        component_en = "Planner skill selector"
        component_zh = "planner 的 skill selector"
    elif "task-graph generator" in lowered:
        component_en = "Planner task-graph generator"
        component_zh = "planner 的 task-graph generator"
    else:
        component_en = "Planner"
        component_zh = "planner"

    if "reasoning_content" in lowered and "final text" in lowered:
        if "finish_reason=length" in lowered:
            reason_en = "the LLM used the whole output budget for reasoning and returned no final JSON/YAML contract"
            reason_zh = "LLM 把输出额度都用在 reasoning 上了，没有返回最终 JSON/YAML contract"
        else:
            reason_en = "the LLM returned reasoning content but no final JSON/YAML contract"
            reason_zh = "LLM 只返回了 reasoning 内容，没有返回最终 JSON/YAML contract"
    elif "does not contain final text content" in lowered or "does not contain text content" in lowered:
        reason_en = "the LLM response had no final text content to parse"
        reason_zh = "LLM 响应没有可解析的最终文本"
    elif "did not include a parseable" in lowered:
        reason_en = "the LLM response did not include a parseable YAML/JSON contract"
        reason_zh = "LLM 响应里没有可解析的 YAML/JSON contract"
    elif "non-object payload" in lowered:
        reason_en = "the parsed planner contract was not a JSON/YAML object"
        reason_zh = "解析出来的 planner contract 不是 JSON/YAML object"
    else:
        reason_en = "the planning output did not satisfy the required contract"
        reason_zh = "规划输出不满足后端要求的 contract"

    if chinese:
        return (
            f"{component_zh} 在生成可执行工作流前失败：{reason_zh}。"
            "这不是你的 query 缺少变量、时间、空间或深度字段。请重试一次；如果重复出现，需要调整 selector/planner 的模型或输出 token 设置。"
        )
    return (
        f"{component_en} failed before execution: {reason_en}. "
        "No variable, time range, region, or depth field appears to be missing from your request. "
        "Please try again; if it repeats, adjust the selector/planner model or output-token budget."
    )


def _is_safe_public_failure_text(raw_error: Any) -> bool:
    text = str(raw_error or "").strip()
    if not text:
        return False
    lowered = text.lower()
    technical_markers = (
        "traceback",
        "jsondecodeerror",
        "valid json",
        "llm request failed",
        "notfounderror",
        "apiconnectionerror",
        "filenotfounderror",
        "keyerror",
        "valueerror",
        "runtimeerror",
        "httpx",
        "base_url=",
        "error_type=",
        "stack",
    )
    if any(marker in lowered for marker in technical_markers):
        return False
    safe_prefixes = (
        "No data is available",
        "The selected depth, time, or region produced no finite values",
        "Current results can support",
        "The policy summary tried",
        "The synthesis tried",
        "Generated code returned output_type",
        "Generated code result is missing required",
    )
    return text.startswith(safe_prefixes)


def _public_failure(
    raw_error: Any,
    *,
    query: Optional[str],
    default_kind: FailureKind,
) -> Dict[str, Any]:
    kind = _classify_failure_kind(raw_error, default_kind)
    return {
        "message": _public_failure_message(kind, raw_error, query=query),
        "failure_kind": kind,
        "recoverable": kind != "transport",
    }


def _public_failure_message(kind: FailureKind, raw_error: Any, *, query: Optional[str]) -> str:
    chinese = _query_prefers_chinese(query)
    if kind == "capability_boundary":
        if chinese:
            return (
                "这个请求目前超出了 OceanMind 可执行能力边界。OceanMind 主要支持当前海洋数据集的介绍、"
                "变量和覆盖范围查询，以及时间序列、空间场、剖面/断面、趋势、相关和事件类分析；"
                "通用问题可以直接回答，需要实时外部信息时可以搜索。你可以试着问："
                "“介绍一下当前数据集有哪些变量”、“画 2015 年 1 月表层温度空间分布”、"
                "或“计算某区域 2015 年 1-3 月 temp 时间序列”。"
            )
        return (
            "This request is outside OceanMind's current executable capability boundary. OceanMind supports active "
            "ocean-dataset descriptions, variables and coverage, time series, spatial fields, profiles/sections, "
            "trends, correlations, and event-style analyses; it can also answer general questions and search for "
            "current external information when needed. Try asking: “What variables are in the current dataset?”, "
            "“Map surface temperature for January 2015”, or “Compute a temp time series for this region”."
        )

    if kind == "llm_format":
        missing_fields = _missing_fields_from_error(raw_error)
        if missing_fields:
            return _missing_fields_failure_message(missing_fields, chinese=chinese)
        code_agent_error = _looks_like_code_agent_error(raw_error)
        if code_agent_error:
            if chinese:
                return (
                    "CodeAgent 没能生成有效的分析设计或 Python 入口函数，所以 generated-code 步骤没有执行完成。"
                    "这不是你的请求缺少变量、时间或区域，而是内部代码生成格式问题；请重试一次。"
                )
            return (
                "CodeAgent could not produce a valid analysis design or Python entrypoint for the generated-code step. "
                "This is an internal code-generation format issue, not a missing variable, time range, or region in your request. Please try again."
            )
        return _planner_format_failure_message(raw_error, chinese=chinese)

    if kind == "planning":
        missing_fields = _missing_fields_from_error(raw_error)
        if missing_fields:
            return _missing_fields_failure_message(missing_fields, chinese=chinese)
        if chinese:
            return (
                "planner 没能生成可执行的 OceanMind 工作流，但错误里没有报告具体缺失字段。"
                "请重试一次；如果仍失败，再补充你要分析的变量、时间范围、空间范围和输出类型。"
            )
        return (
            "The planner could not generate an executable OceanMind workflow, and it did not report any specific "
            "missing fields. Please try again; if it repeats, include the exact variable, time range, region, and output type."
        )

    if kind == "execution":
        if _is_safe_public_failure_text(raw_error):
            return str(raw_error).strip()
        if chinese:
            return (
                "分析步骤执行时没有得到可靠结果。请检查变量、时间范围、区域和深度是否在当前数据集覆盖范围内，"
                "或尝试放宽筛选条件。"
            )
        return (
            "The analysis step could not produce a reliable result. Please check that the variable, time range, "
            "region, and depth are covered by the current dataset, or try a broader selection."
        )

    if kind == "synthesis":
        if _is_safe_public_failure_text(raw_error):
            return str(raw_error).strip()
        if chinese:
            return (
                "数据分析步骤已经运行，但生成最终自然语言总结时没有得到可靠的结构化结果。"
                "你可以查看已生成的图表/步骤结果，或重试一次生成总结。"
            )
        return (
            "The analysis steps ran, but the final natural-language summary could not be generated reliably. "
            "You can still inspect the generated step results or try the summary again."
        )

    if chinese:
        return (
            "实时连接或后端处理过程没有返回完整结果。请稍后重试；如果刚才已经开始执行，"
            "可以保留当前进度后重新提交。"
        )
    return (
        "The live connection or backend process did not return a complete result. Please try again; if execution "
        "had already started, keep the current progress and resubmit."
    )
