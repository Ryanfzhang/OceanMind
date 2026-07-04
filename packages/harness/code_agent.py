"""Generated-code planning agent for uncovered analysis nodes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from packages.llm_gateway.config import (
    DEFAULT_OPENAI_MODEL,
    load_llm_api_key,
    load_llm_base_url,
    load_model_name,
)
from packages.llm_gateway.openai_compatible_client import OpenAICompatibleClientAdapter


DesignWriter = Callable[[Mapping[str, Any]], Any]
CodeWriter = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class GeneratedCodePlan:
    code: str
    params: Mapping[str, Any]
    contract: Mapping[str, Any]
    analysis_design: Mapping[str, Any]
    code_steps: Tuple[Mapping[str, Any], ...]


class CodeAgent:
    """Create generated-code plans under a fixed I/O contract.

    The agent is intentionally split into two loops:
    1. decide the generated step's input/output contract and analysis design
       from the user query plus resolved input artifact schemas;
    2. turn that design into executable Python.

    Planner-provided analysis details are treated as hints, not as the source
    of truth, because skill selection can bias the planner toward the wrong
    output shape.
    """

    allowed_libraries = ("numpy", "pandas", "xarray", "scipy")
    blocked_capabilities = ("file", "network", "shell")

    DEFAULT_MODEL = load_model_name(
        "CODE_AGENT_MODEL",
        default=load_model_name("PLANNER_MODEL", default=DEFAULT_OPENAI_MODEL),
    )

    def __init__(
        self,
        *,
        design_writer: Optional[DesignWriter] = None,
        code_writer: Optional[CodeWriter] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        client: Optional[Any] = None,
        trust_env: bool = False,
        request_retries: int = 0,
    ) -> None:
        self.design_writer = design_writer
        self.code_writer = code_writer
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = client
        self.trust_env = trust_env
        self.request_retries = request_retries
        self._adapter: Optional[OpenAICompatibleClientAdapter] = None

    def make_plan(
        self,
        *,
        user_request: str,
        input_refs: Mapping[str, str],
        code_params: Mapping[str, Any],
        generate_code: bool = True,
    ) -> GeneratedCodePlan:
        analysis_design = self.design_analysis(
            user_request=user_request,
            input_refs=input_refs,
            code_params=code_params,
        )
        expected_output_type = _analysis_design_output_type(analysis_design)
        contract: Dict[str, Any] = {
            "entrypoint": "run(inputs, params) -> dict",
            "inputs": {"field": "OceanArtifact safe data view"},
            "output": "normalized frontend result dict",
            "expected_output_type": expected_output_type,
            "required_fields_by_output_type": {
                "spatial_field_result": ["output_type", "lon", "lat", "values", "metadata"],
                "timeseries_result": ["output_type", "times", "values", "metadata"],
            },
            "allowed_libraries": list(self.allowed_libraries),
            "blocked_capabilities": list(self.blocked_capabilities),
            "repair_loop": {
                "max_attempts": 3,
                "feedback": "traceback + input artifact schema + expected output schema",
                "status": "llm_code_writer",
            },
        }
        code_steps = self.write_code_steps(analysis_design, code_params=code_params)
        code_params_payload = dict(code_params)
        code_params_payload["analysis_design"] = analysis_design
        params = {
            "input_refs": dict(input_refs),
            "code_params": code_params_payload,
            "analysis_design": analysis_design,
            "code_steps": [dict(step) for step in code_steps],
            "io_contract": contract,
        }
        code = ""
        if generate_code:
            code = self.write_code(
                user_request=user_request,
                input_refs=input_refs,
                code_params=code_params_payload,
                analysis_design=analysis_design,
                code_steps=code_steps,
                contract=contract,
            )
        return GeneratedCodePlan(
            code=code,
            params=params,
            contract=contract,
            analysis_design=analysis_design,
            code_steps=code_steps,
        )

    def design_analysis(
        self,
        *,
        user_request: str,
        input_refs: Mapping[str, str],
        code_params: Mapping[str, Any],
        input_schemas: Optional[Mapping[str, Any]] = None,
        planner_analysis_design: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        planner_hint = planner_analysis_design if isinstance(planner_analysis_design, Mapping) else None
        if planner_hint is None and isinstance(code_params.get("analysis_design"), Mapping):
            planner_hint = code_params.get("analysis_design")

        payload = {
            "user_request": user_request,
            "input_refs": dict(input_refs),
            "input_schemas": dict(input_schemas or {}),
            "code_params": dict(code_params),
            "planner_analysis_design_hint": dict(planner_hint) if isinstance(planner_hint, Mapping) else None,
        }
        if self.design_writer is not None:
            return _normalize_analysis_design(
                _coerce_design_writer_result(self.design_writer(payload)),
                user_request=user_request,
                input_refs=input_refs,
                code_params=code_params,
            )
        if self._has_llm_credentials():
            return self._design_analysis_with_llm(payload)

        request = str(user_request or "")
        lowered = request.lower()
        variable = str(code_params.get("variable") or "").strip() or _infer_variable_from_request(lowered)
        data = {
            "input_ref": dict(input_refs).get("field"),
            "variable": variable,
            "lon_range": code_params.get("lon_range"),
            "lat_range": code_params.get("lat_range"),
            "time_range": code_params.get("time_range"),
            "vertical_mode": _default_vertical_mode(code_params.get("vertical_mode")),
        }
        output_type = _infer_generated_output_type_from_request(lowered)
        analysis = _default_generated_analysis_from_request(lowered, output_type)
        frontend = "spatial_field" if output_type == "spatial_field_result" else "interactive_series"
        output = {
            "output_type": output_type,
            "frontend": frontend,
            "title": f"{variable.title() if variable else 'Field'} generated analysis",
            "unit": "percent" if "variability" in lowered else code_params.get("unit"),
        }
        return {
            "data": data,
            "analysis": analysis,
            "output": output,
            "assumptions": [
                "Generated code operates only on provided input artifacts.",
                "No file, network, or shell access is allowed.",
                "Do not replace a custom generated-code analysis with a regional/domain mean unless the user asks for a mean.",
            ],
        }

    def build_contract(self, analysis_design: Mapping[str, Any]) -> Dict[str, Any]:
        expected_output_type = _analysis_design_output_type(analysis_design)
        return {
            "entrypoint": "run(inputs, params) -> dict",
            "inputs": {"field": "OceanArtifact safe data view"},
            "output": "normalized frontend result dict",
            "expected_output_type": expected_output_type,
            "required_fields_by_output_type": {
                "spatial_field_result": ["output_type", "lon", "lat", "values", "metadata"],
                "timeseries_result": ["output_type", "times", "values", "metadata"],
            },
            "allowed_libraries": list(self.allowed_libraries),
            "blocked_capabilities": list(self.blocked_capabilities),
            "repair_loop": {
                "max_attempts": 3,
                "feedback": "traceback + input artifact schema + expected output schema",
                "status": "llm_code_writer",
            },
        }

    def write_code_steps(
        self,
        analysis_design: Mapping[str, Any],
        *,
        code_params: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Mapping[str, Any], ...]:
        supplied_steps = None
        if isinstance(code_params, Mapping) and isinstance(code_params.get("code_steps"), list):
            supplied_steps = code_params.get("code_steps")
        elif isinstance(analysis_design.get("code_steps"), list):
            supplied_steps = analysis_design.get("code_steps")
        if supplied_steps is not None:
            return tuple(dict(step) for step in supplied_steps if isinstance(step, Mapping))

        output = analysis_design.get("output") if isinstance(analysis_design.get("output"), Mapping) else {}
        output_type = str(output.get("output_type") or "normalized_result")
        return (
            {"id": "prepare_field", "purpose": "Read field artifact and metadata."},
            {"id": "write_python_analysis", "purpose": "Implement the planner-provided analysis design in Python."},
            {"id": "return_result", "purpose": f"Return frontend-ready {output_type}."},
        )

    def write_code(
        self,
        *,
        user_request: str,
        input_refs: Mapping[str, str],
        code_params: Mapping[str, Any],
        analysis_design: Mapping[str, Any],
        code_steps: Tuple[Mapping[str, Any], ...],
        contract: Mapping[str, Any],
        input_schemas: Optional[Mapping[str, Any]] = None,
    ) -> str:
        payload = {
            "user_request": user_request,
            "input_refs": dict(input_refs),
            "input_schemas": dict(input_schemas or {}),
            "analysis_design": dict(analysis_design),
            "code_steps": [dict(step) for step in code_steps],
            "code_params": dict(code_params),
            "contract": dict(contract),
        }
        if self.code_writer is not None:
            return _extract_python_code(_coerce_code_writer_result(self.code_writer(payload)))
        return self._write_code_with_llm(payload)

    def _write_code_with_llm(self, payload: Mapping[str, Any]) -> str:
        adapter = self._get_adapter()
        system = (
            "You are OceanMind CodeAgent. Write restricted Python code for one generated analysis step.\n"
            "The CodeAgent design loop has already decided what data to use, what analysis to run, and what frontend "
            "result shape to return. Follow analysis_design; do not replace it with a generic mean unless the design "
            "asks for that.\n"
            "The code must define exactly run(inputs, params) -> dict. inputs['field'] is an xarray-like DataArray. "
            "params contains user_request, analysis_design, ranges, variable, and vertical settings.\n"
            "Allowed imports: numpy, pandas, xarray, scipy. No file, network, shell, subprocess, pathlib, or OS access.\n"
            "Return exactly analysis_design.output.output_type. A different output_type is a contract error.\n"
            "For spatial_field_result, preserve lon/lat axes and return one value per grid cell; reduce time/depth "
            "only when the analysis design says to. Do not compute a domain/regional mean for a spatial output.\n"
            "Return a normalized result dict. For spatial maps use output_type='spatial_field_result' with lon, lat, "
            "values, metadata. For time series use output_type='timeseries_result' with times, values, metadata. "
            "Record assumptions and formulas in metadata. Return only Python code, no prose."
        )
        user_content = (
            "Write the generated analysis code for this payload.\n\n"
            f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}\n```"
        )
        client = adapter.get_client()
        response = adapter.create_message(
            client=client,
            max_tokens=2600,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            request_name="ocean_harness_code_agent",
            json_response=False,
        )
        return _extract_python_code(adapter.extract_response_text(response))

    def _design_analysis_with_llm(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        adapter = self._get_adapter()
        system = (
            "You are OceanMind CodeAgent design loop. Decide the generated Python step's input/output contract and "
            "analysis design before any code is written.\n"
            "Use the user query and resolved input artifact schemas as the source of truth. Planner analysis hints may "
            "come from skill selection and can be wrong; treat them only as non-authoritative suggestions.\n"
            "Do not write Python code. Return exactly one JSON object with keys: data, analysis, output, assumptions.\n"
            "For gridded ocean fields, choose which axes to preserve and which axes to reduce. If the user asks for a "
            "custom/ad-hoc index or variability metric and does not explicitly ask for a regional/domain mean or a "
            "time series, prefer preserving lat/lon and reducing time/depth as needed. Only choose timeseries_result "
            "when the user asks for values over time, evolution, daily/monthly series, or a regional/domain mean series.\n"
            "For spatial_field_result, output must be lon/lat/values/metadata. For timeseries_result, output must be "
            "times/values/metadata."
        )
        user_content = (
            "Decide the analysis design for this generated-code step. Return only one compact JSON object.\n\n"
            f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}\n```"
        )
        client = adapter.get_client()
        response = adapter.create_message(
            client=client,
            max_tokens=1800,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            request_name="ocean_harness_code_agent_design",
            json_response=True,
        )
        try:
            text = adapter.extract_response_text(response)
        except ValueError as exc:
            raise ValueError(f"CodeAgent design output was not a valid JSON analysis design: {exc}") from exc
        try:
            design = _parse_json_object(text)
        except Exception as exc:
            raise ValueError(f"CodeAgent design output was not a valid JSON analysis design: {exc}") from exc
        return _normalize_analysis_design(
            design,
            user_request=str(payload.get("user_request") or ""),
            input_refs=payload.get("input_refs") if isinstance(payload.get("input_refs"), Mapping) else {},
            code_params=payload.get("code_params") if isinstance(payload.get("code_params"), Mapping) else {},
        )

    def _has_llm_credentials(self) -> bool:
        if self.client is not None:
            return True
        try:
            return bool(self.api_key or load_llm_api_key())
        except Exception:
            return False

    def _get_adapter(self) -> OpenAICompatibleClientAdapter:
        if self._adapter is None:
            self._adapter = OpenAICompatibleClientAdapter(
                api_key=self.api_key or load_llm_api_key(),
                base_url=self.base_url or load_llm_base_url(),
                model=self.model,
                client=self.client,
                trust_env=self.trust_env,
                request_retries=self.request_retries,
            )
        return self._adapter


def _normalize_analysis_design(
    value: Mapping[str, Any],
    *,
    user_request: str,
    input_refs: Mapping[str, str],
    code_params: Mapping[str, Any],
) -> Dict[str, Any]:
    design = dict(value)
    raw_data = design.get("data")
    raw_analysis = design.get("analysis")
    raw_output = design.get("output")
    data = dict(raw_data) if isinstance(raw_data, Mapping) else {}
    analysis = dict(raw_analysis) if isinstance(raw_analysis, Mapping) else {}
    output = dict(raw_output) if isinstance(raw_output, Mapping) else {}
    if raw_data is not None and not isinstance(raw_data, Mapping):
        data["description"] = str(raw_data)
    if raw_analysis is not None and not isinstance(raw_analysis, Mapping):
        analysis["description"] = str(raw_analysis)
    if raw_output is not None and not isinstance(raw_output, Mapping):
        output["description"] = str(raw_output)
    variable = str(data.get("variable") or code_params.get("variable") or _infer_variable_from_request(user_request)).strip()
    data.setdefault("input_ref", dict(input_refs).get("field"))
    data.setdefault("variable", variable)
    data.setdefault("lon_range", code_params.get("lon_range"))
    data.setdefault("lat_range", code_params.get("lat_range"))
    data.setdefault("time_range", code_params.get("time_range"))
    data.setdefault("vertical_mode", _default_vertical_mode(code_params.get("vertical_mode")))
    analysis.setdefault("description", "Implement the planner-provided generated-code analysis.")
    output_type = str(output.get("output_type") or "").strip()
    if not output_type:
        frontend = str(output.get("frontend") or output.get("artifact_type") or output.get("description") or "").lower()
        if any(term in frontend for term in ("spatial", "map", "xarray.dataarray", "grid", "field")):
            output["output_type"] = "spatial_field_result"
            output.setdefault("frontend", "spatial_field")
        elif any(term in frontend for term in ("time series", "timeseries", "series", "daily", "monthly")):
            output["output_type"] = "timeseries_result"
            output.setdefault("frontend", "interactive_series")
        else:
            output["output_type"] = "generic_result"
    output.setdefault("title", f"{variable.title() if variable else 'Field'} generated analysis")
    return {
        "data": data,
        "analysis": analysis,
        "output": output,
        "code_steps": list(design.get("code_steps") or []),
        "assumptions": list(design.get("assumptions") or []),
    }


def _analysis_design_output_type(analysis_design: Mapping[str, Any]) -> str:
    output = analysis_design.get("output") if isinstance(analysis_design.get("output"), Mapping) else {}
    return str(output.get("output_type") or "generic_result")


def _infer_generated_output_type_from_request(lowered_request: str) -> str:
    if re.search(r"\b(time series|timeseries|daily|monthly|evolution|over time)\b|时间序列|随时间", lowered_request):
        return "timeseries_result"
    if re.search(r"\b(map|spatial|field|grid|hotspot|distribution)\b|地图|空间|分布", lowered_request):
        return "spatial_field_result"
    if re.search(r"\b(variability|index|metric|diagnostic)\b|指标|诊断", lowered_request):
        return "spatial_field_result"
    return "generic_result"


def _default_generated_analysis_from_request(lowered_request: str, output_type: str) -> Dict[str, Any]:
    if "variability" in lowered_request and output_type == "spatial_field_result":
        return {
            "formula": "temporal coefficient of variation per lon/lat grid cell: std(time) / abs(mean(time)) * 100",
            "reduction_axes": ["time"],
            "preserve_axes": ["lat", "lon"],
            "description": (
                "Compute a custom variability index over the requested time window while preserving the spatial grid. "
                "Do not reduce lon/lat to a regional mean unless the user explicitly asks for a domain mean."
            ),
        }
    if output_type == "spatial_field_result":
        return {
            "description": (
                "Implement the user's custom generated-code analysis as a spatial field, preserving lon/lat axes. "
                "Reduce only the axes required by the requested diagnostic."
            ),
            "preserve_axes": ["lat", "lon"],
        }
    if output_type == "timeseries_result":
        return {
            "description": (
                "Implement the user's custom generated-code analysis as a time series. If a spatial reduction is "
                "required, record it explicitly in metadata."
            ),
            "preserve_axes": ["time"],
        }
    return {
        "description": (
            "Implement the user's custom generated-code analysis from the supplied field. Do not substitute a "
            "generic regional mean unless the request explicitly asks for a mean."
        )
    }


def _infer_variable_from_request(text: str) -> str:
    lowered = str(text or "").lower()
    aliases = {
        "oxygen": ("oxygen", "o2", "dissolved oxygen"),
        "chlorophyll": ("chlorophyll", "chla", "chlorophyll-a"),
        "temp": ("temperature", "sst", "temp"),
        "salt": ("salinity", "salt"),
    }
    for variable, terms in aliases.items():
        if any(term in lowered for term in terms):
            return variable
    return "field"


def _default_vertical_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if not mode or mode in {"none", "null", "unspecified"}:
        return "surface"
    return mode


def _coerce_code_writer_result(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("code", "python", "python_code"):
            if isinstance(value.get(key), str):
                return str(value[key])
    return str(value or "")


def _coerce_design_writer_result(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return _parse_json_object(str(value or ""))


def _parse_json_object(text: str) -> Mapping[str, Any]:
    candidate = str(text or "").strip()
    fence = re.search(r"```(?:json|yaml|yml)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, Mapping):
        raise ValueError("CodeAgent design output must be a JSON object.")
    return parsed


def _extract_python_code(text: str) -> str:
    candidate = str(text or "").strip()
    fence = re.search(r"```(?:python|py)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
    if "def run(" not in candidate:
        raise ValueError("CodeAgent output must define run(inputs, params) -> dict.")
    return candidate
