"""
Tool Orchestrator - 工具执行与中间结果编排

统一处理：
1. 工具执行
2. 中间结果缓存
3. `$ref:result.field` 引用解析
4. 工具输出标准化
"""

import os
import threading
from typing import Any, Callable, Dict, List, Optional

from packages.tool_loader.introspect import get_tool_by_name, get_tools_cached
from packages.tool_loader.progress import reset_tool_progress_callback, set_tool_progress_callback
from packages.tool_loader.registry import get_tool_output_type
from domain.ocean.data_access.partitioned import (
    is_partitioned_xarray,
    partitioned_metadata,
)
from domain.ocean.result_payload import as_numeric_array, json_safe_array, result_full_list_limit


_SENTINEL_DEPTH_ABS_THRESHOLD = 9000.0


class ToolOrchestrator:
    """
    工具编排器

    对外暴露统一的工具执行接口，并负责中间结果管理。
    """

    def __init__(
        self,
        workspace: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Callable]] = None
    ):
        self.workspace = workspace if workspace is not None else {}
        self.tools = tools if tools is not None else get_tools_cached()
        self.results: Dict[str, Dict[str, Any]] = {}
        self._counter = 0
        self._lock = threading.RLock()

    def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        save_as: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        执行工具并缓存标准化结果。

        Args:
            tool_name: 工具名
            params: 原始参数，支持 `$ref:` 和旧 `$name` 引用
            save_as: 可选的保存别名

        Returns:
            包含 `ref_id`、`result_id`、`output_type` 和摘要的字典
        """
        tool_func = get_tool_by_name(tool_name, self.tools)
        resolved_params = self.resolve_references(params)
        token = set_tool_progress_callback(progress_callback)
        try:
            raw_result = tool_func(**resolved_params)
        finally:
            reset_tool_progress_callback(token)
        normalized_result = self.normalize_result(tool_name, raw_result)

        with self._lock:
            ref_id = self._next_ref_id()
            result_id = save_as or ref_id

            self.results[ref_id] = normalized_result
            self.results[result_id] = normalized_result

            if isinstance(self.workspace, dict):
                self.workspace[result_id] = normalized_result

        return {
            "ref_id": ref_id,
            "result_id": result_id,
            "output_type": normalized_result.get("output_type", "generic_result"),
            "result": normalized_result,
            "summary": self.summarize_result(normalized_result)
        }

    def resolve_references(self, obj: Any) -> Any:
        """递归解析参数中的 `$ref:` 和旧 `$name` 引用。"""
        if isinstance(obj, dict):
            return {key: self.resolve_references(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self.resolve_references(value) for value in obj]
        if isinstance(obj, tuple):
            return tuple(self.resolve_references(value) for value in obj)
        if isinstance(obj, str) and obj.startswith("$ref:"):
            return self._lookup_ref(obj[5:])
        if isinstance(obj, str) and obj.startswith("$"):
            return self._lookup_legacy_ref(obj[1:])
        return obj

    def normalize_result(self, tool_name: str, result: Any) -> Dict[str, Any]:
        """
        将工具输出统一包装为 dict。

        - 已经是 dict：补齐 `output_type`
        - `xr.DataArray/xr.Dataset`：包装成 `data_container_result`
        - 标量/列表：包装成 `generic_result`
        """
        if self._is_normalized_result(result):
            return self._normalize_temporal_axes(dict(result))

        output_type = self._infer_output_type(tool_name, result)
        xarray_types = self._get_xarray_types()

        if is_partitioned_xarray(result):
            metadata = partitioned_metadata(result, tool_name)
            return {
                "output_type": output_type,
                "data": result,
                "metadata": metadata
            }

        if isinstance(result, dict):
            normalized = dict(result)
            normalized.setdefault("output_type", output_type)
            return self._normalize_temporal_axes(normalized)

        if xarray_types and isinstance(result, xarray_types):
            metadata = self._build_data_metadata(result, tool_name)
            return {
                "output_type": output_type,
                "data": result,
                "metadata": metadata
            }

        return {
            "output_type": output_type,
            "value": result,
            "metadata": {
                "source": tool_name,
                "python_type": type(result).__name__
            }
        }

    def summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """生成适合事件流和 LLM 的结果摘要。"""
        import numpy as np

        output_type = result.get("output_type", "generic_result")
        xarray_types = self._get_xarray_types()

        if output_type == "data_container_result" and "data" in result:
            data = result["data"]
            if is_partitioned_xarray(data):
                metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
                summary = {
                    "type": output_type,
                    "dims": list(getattr(data, "dims", [])),
                    "shape": list(getattr(data, "shape", [])) if hasattr(data, "shape") else [],
                    "variable": metadata.get("variable"),
                    "units": metadata.get("units"),
                }
                summary.update(self._summarize_lightweight_data_container(data, metadata))
                return self._make_json_safe(summary)
            if xarray_types and isinstance(data, xarray_types):
                metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
                summary = {
                    "type": output_type,
                    "dims": list(getattr(data, "dims", [])),
                    "shape": list(getattr(data, "shape", [])) if hasattr(data, "shape") else [],
                    "variable": metadata.get("variable"),
                    "units": metadata.get("units"),
                }
                if self._should_use_lightweight_data_summary(data, metadata):
                    summary.update(self._summarize_lightweight_data_container(data, metadata))
                else:
                    summary.update(self._summarize_data_container(data))
                return self._make_json_safe(summary)

        if isinstance(result, dict):
            summary = self._summarize_structured_result(output_type, result)
            if summary is not None:
                return self._make_json_safe(summary)
            return self._make_json_safe(summary)

        if isinstance(result, (list, np.ndarray)):
            return self._make_json_safe({
                "type": output_type,
                "length": len(result)
            })

        return self._make_json_safe({
            "type": output_type,
            "python_type": type(result).__name__
        })

    def get_result(self, result_id: str) -> Dict[str, Any]:
        """获取标准化结果。"""
        with self._lock:
            if result_id not in self.results:
                raise KeyError(f"Result not found: {result_id}")
            return self.results[result_id]

    def clear_results(self):
        """清空结果缓存。"""
        with self._lock:
            self.results.clear()
            if isinstance(self.workspace, dict):
                self.workspace.clear()

    def _lookup_ref(self, expr: str) -> Any:
        parts = expr.split(".")
        root = self._get_normalized_reference(parts[0])
        value: Any = root

        for key in parts[1:]:
            if not isinstance(value, dict):
                raise ValueError(f"Reference path is invalid: {expr}")
            if key not in value:
                raise ValueError(f"Reference field not found: {expr}")
            value = value[key]

        return value

    def _lookup_legacy_ref(self, expr: str) -> Any:
        if "." in expr:
            return self._lookup_ref(expr)

        result = self._get_normalized_reference(expr)
        if self._is_data_container_result(result):
            return result["data"]
        return result

    def _get_normalized_reference(self, name: str) -> Dict[str, Any]:
        with self._lock:
            if name in self.results:
                return self.results[name]

            if isinstance(self.workspace, dict) and name in self.workspace:
                workspace_value = self.workspace[name]
                if self._is_normalized_result(workspace_value):
                    normalized = self._normalize_temporal_axes(dict(workspace_value))
                else:
                    normalized = self.normalize_result(name, workspace_value)
                self.results[name] = normalized
                return normalized

        raise ValueError(f"Reference not found: {name}")

    def _infer_output_type(self, tool_name: str, result: Any) -> str:
        registered_output_type = get_tool_output_type(tool_name)
        if registered_output_type:
            return registered_output_type

        xarray_types = self._get_xarray_types()
        if xarray_types and isinstance(result, xarray_types):
            return "data_container_result"
        if is_partitioned_xarray(result):
            return "data_container_result"

        if isinstance(result, dict):
            keys = set(result.keys())
            if {"times", "values"}.issubset(keys):
                return "timeseries_result"
            if {"period", "labels", "values"}.issubset(keys):
                return "climatology_result"
            if {"data", "p_value", "significance_mask"}.issubset(keys):
                return "field_trend_result"
            if {"lags", "correlations", "p_values"}.issubset(keys):
                return "lag_correlation_result"
            if {"distance_km", "sample_points", "values"}.issubset(keys):
                return "section_result"
            if {"temperature", "salinity"}.issubset(keys):
                return "ts_diagram_result"
            if {"comparison", "changes"}.issubset(keys):
                return "event_comparison_result"
            if {"event_type", "events"}.issubset(keys):
                return "event_detection_result"
            if {"event_count", "centroid"}.issubset(keys):
                return "event_spatial_distribution_result"
            if {"slope", "correlation", "p_value", "significant_mask"}.issubset(keys):
                return "regression_map_result"
            if {"positive_composite", "negative_composite", "difference", "sample_counts"}.issubset(keys):
                return "composite_result"
            if {"frequency", "period", "power"}.issubset(keys):
                return "spectrum_result"
            if {"times", "layers"}.issubset(keys):
                return "layer_transport_result"
            if {"lon", "lat", "values"}.issubset(keys):
                return "spatial_field_result"
            if {"depth", "values"}.issubset(keys):
                return "profile_result"
            return "generic_result"

        return "generic_result"

    def _build_data_metadata(self, result: Any, tool_name: str) -> Dict[str, Any]:
        metadata = {
            "source": tool_name,
            "python_type": type(result).__name__,
        }

        xarray_types = self._get_xarray_types()
        if not xarray_types:
            return metadata

        if is_partitioned_xarray(result):
            metadata.update(partitioned_metadata(result, tool_name))
            return metadata

        data_array_type, dataset_type = xarray_types

        if isinstance(result, data_array_type):
            metadata.update({
                "variable": result.name or "unknown",
                "dims": list(result.dims),
                "shape": list(result.shape),
                "units": result.attrs.get("units", ""),
                "feature": result.attrs.get("feature"),
                "method": result.attrs.get("method"),
                "aggregation": result.attrs.get("aggregation"),
                "upper_bound_source": result.attrs.get("upper_bound_source"),
                "lower_bound_source": result.attrs.get("lower_bound_source"),
            })
            return metadata

        if isinstance(result, dataset_type):
            metadata.update({
                "variables": list(result.data_vars.keys()),
                "dims": {name: int(size) for name, size in result.sizes.items()},
            })
            return metadata

        return metadata

    def _summarize_data_container(self, data: Any) -> Dict[str, Any]:
        import numpy as np

        summary: Dict[str, Any] = {}
        if is_partitioned_xarray(data):
            if hasattr(data, "data_vars"):
                summary["variables"] = list(data.data_vars.keys())
                summary["dim_sizes"] = {name: int(size) for name, size in data.sizes.items()}
            coord_ranges = data.coord_ranges()
            if coord_ranges:
                summary["coord_ranges"] = coord_ranges
            attrs = getattr(data, "attrs", {})
            if isinstance(attrs, dict):
                for key in (
                    "feature",
                    "method",
                    "aggregation",
                    "vertical_mode",
                    "bottom_selection",
                    "bottom_depth_coordinate",
                    "upper_bound_source",
                    "lower_bound_source",
                    "mixed_layer_depth_source",
                    "threshold",
                    "season_filter",
                    "season_months",
                ):
                    if key in attrs:
                        summary[key] = attrs[key]
            summary["statistics_skipped"] = "large data_container summaries avoid reading full arrays"
            return summary

        if hasattr(data, "data_vars"):
            summary["variables"] = list(data.data_vars.keys())
            summary["dim_sizes"] = {name: int(size) for name, size in data.sizes.items()}
            analysis_data = data.to_array()
        else:
            analysis_data = data

        coord_ranges = self._extract_coord_ranges(data)
        if coord_ranges:
            summary["coord_ranges"] = coord_ranges

        attrs = getattr(data, "attrs", {})
        if isinstance(attrs, dict):
            for key in (
                "feature",
                "method",
                "aggregation",
                "vertical_mode",
                "bottom_selection",
                "bottom_depth_coordinate",
                "upper_bound_source",
                "lower_bound_source",
                "mixed_layer_depth_source",
                "threshold",
            ):
                if key in attrs:
                    summary[key] = attrs[key]

        values = np.asarray(analysis_data.values, dtype=float)
        numeric_stats = self._compute_numeric_statistics(values)
        if numeric_stats:
            summary["statistics"] = numeric_stats

        extrema = self._extract_dataarray_extrema(analysis_data)
        if extrema:
            summary["extrema"] = extrema

        return summary

    def _should_use_lightweight_data_summary(self, data: Any, metadata: Dict[str, Any]) -> bool:
        source = metadata.get("source") if isinstance(metadata, dict) else None
        if source in {"load_dataset", "assemble_dataset", "compute_derived_field"}:
            return True

        estimated_values = self._estimate_xarray_value_count(data)
        if estimated_values is None:
            return False
        return estimated_values > self._summary_value_limit()

    def _summary_value_limit(self) -> int:
        raw_value = os.environ.get("OCEAN_SUMMARY_VALUE_LIMIT", "1000000")
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            return 1_000_000
        return max(0, parsed)

    def _estimate_xarray_value_count(self, data: Any) -> Optional[int]:
        try:
            if hasattr(data, "data_vars"):
                total = 0
                for variable in data.data_vars.values():
                    total += int(getattr(variable, "size", 0))
                return total
            if hasattr(data, "size"):
                return int(data.size)
        except Exception:
            return None
        return None

    def _summarize_lightweight_data_container(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize large/lazy fields without forcing array values to read from disk."""
        summary: Dict[str, Any] = {}
        if hasattr(data, "data_vars"):
            summary["variables"] = list(data.data_vars.keys())
            summary["dim_sizes"] = {name: int(size) for name, size in data.sizes.items()}

        coord_ranges = self._extract_coord_ranges(data)
        if coord_ranges:
            summary["coord_ranges"] = coord_ranges

        attrs = getattr(data, "attrs", {})
        if isinstance(attrs, dict):
            for key in (
                "feature",
                "method",
                "aggregation",
                "upper_bound_source",
                "lower_bound_source",
                "mixed_layer_depth_source",
                "threshold",
                "season_filter",
                "season_months",
            ):
                if key in attrs:
                    summary[key] = attrs[key]

        source = (metadata or {}).get("source") if isinstance(metadata, dict) else None
        if source == "load_dataset":
            reason = "load_dataset summaries avoid reading full arrays"
        elif isinstance(source, str) and source:
            reason = f"{source} summaries avoid reading full arrays"
        else:
            reason = "large data_container summaries avoid reading full arrays"
        summary["statistics_skipped"] = reason
        return summary

    def _summarize_loaded_data_container(self, data: Any) -> Dict[str, Any]:
        return self._summarize_lightweight_data_container(
            data,
            {"source": "load_dataset"},
        )

    def _summarize_structured_result(self, output_type: str, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        summary: Dict[str, Any] = {
            "type": output_type,
            "keys": list(result.keys()),
            "size": len(result),
        }

        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            for key in ("variable", "units", "unit", "source", "x_variable", "y_variable"):
                if key in metadata:
                    summary[key] = metadata[key]

        if output_type == "metadata_result":
            for key in (
                "summary",
                "variables",
                "variable_names",
                "spatial_extent",
                "temporal_extent",
                "depth_range",
                "resolution",
                "backend",
                "data_path_redacted",
                "data_path_policy",
                "data_stores",
            ):
                if key in result:
                    summary[key] = result[key]
            dataset = result.get("dataset")
            if isinstance(dataset, dict):
                for key in ("id", "name", "description", "backend", "data_path_redacted", "data_path_policy"):
                    if key in dataset and key not in summary:
                        summary[key] = dataset[key]
            return summary

        if output_type == "timeseries_result":
            summary.update(self._summarize_timeseries_result(result))
            return summary
        if output_type == "climatology_result":
            summary.update(self._summarize_climatology_result(result))
            return summary
        if output_type == "trend_result":
            summary.update(self._summarize_trend_result(result))
            return summary
        if output_type == "field_trend_result":
            summary.update(self._summarize_field_trend_result(result))
            return summary
        if output_type == "lag_correlation_result":
            summary.update(self._summarize_lag_correlation_result(result))
            return summary
        if output_type == "profile_result":
            summary.update(self._summarize_profile_result(result))
            return summary
        if output_type == "section_result":
            summary.update(self._summarize_section_result(result))
            return summary
        if output_type == "eof_result":
            summary.update(self._summarize_eof_result(result))
            return summary
        if output_type == "spatial_field_result":
            summary.update(self._summarize_spatial_field_result(result))
            return summary
        if output_type == "hovmoller_result":
            summary.update(self._summarize_hovmoller_result(result))
            return summary
        if output_type == "ts_diagram_result":
            summary.update(self._summarize_ts_diagram_result(result))
            return summary
        if output_type == "watermass_event_association_result":
            summary.update(self._summarize_watermass_event_association_result(result))
            return summary
        if output_type == "histogram_result":
            summary.update(self._summarize_histogram_result(result))
            return summary
        if output_type == "histogram_2d_result":
            summary.update(self._summarize_histogram_2d_result(result))
            return summary
        if output_type == "regression_map_result":
            summary.update(self._summarize_regression_map_result(result))
            return summary
        if output_type == "composite_result":
            summary.update(self._summarize_composite_result(result))
            return summary
        if output_type == "spectrum_result":
            summary.update(self._summarize_spectrum_result(result))
            return summary
        if output_type == "layer_transport_result":
            summary.update(self._summarize_layer_transport_result(result))
            return summary
        if output_type == "event_detection_result":
            summary.update(self._summarize_event_detection_result(result))
            return summary
        if output_type == "event_statistics_result":
            summary.update(self._summarize_event_statistics_result(result))
            return summary
        if output_type == "event_spatial_distribution_result":
            summary.update(self._summarize_event_spatial_distribution_result(result))
            return summary
        if output_type == "event_comparison_result":
            summary.update(self._summarize_event_comparison_result(result))
            return summary
        if output_type == "mechanism_score_result":
            summary.update(self._summarize_mechanism_score_result(result))
            return summary
        if output_type == "evidence_report_result":
            summary.update(self._summarize_evidence_report_result(result))
            return summary
        if output_type == "environment_assessment_result":
            summary.update(self._summarize_environment_assessment_result(result))
            return summary
        if output_type == "policy_recommendation_result":
            summary.update(self._summarize_policy_recommendation_result(result))
            return summary

        if "values" in result:
            values = result["values"]
            if isinstance(values, list):
                summary["value_length"] = len(values)
                if values and isinstance(values[0], list):
                    summary["value_shape"] = [len(values), len(values[0])]

        if "events" in result and isinstance(result["events"], list):
            summary["event_count"] = len(result["events"])

        scalar_fields = self._extract_scalar_fields(result)
        if scalar_fields:
            summary["scalar_fields"] = scalar_fields

        metadata_stats = self._extract_metadata_statistics(metadata)
        if metadata_stats:
            summary["metadata_statistics"] = metadata_stats

        return summary

    def _summarize_timeseries_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        times = result.get("times", [])
        values = np.asarray(result.get("values", []), dtype=float)
        if values.ndim > 1:
            values = values.reshape(-1)
        finite_mask = np.isfinite(values) if values.size else np.asarray([], dtype=bool)
        finite_count = int(np.count_nonzero(finite_mask))
        has_finite = finite_count > 0

        summary: Dict[str, Any] = {
            "n_points": int(values.size),
            "finite_count": finite_count,
            "has_finite_values": has_finite,
            "time_start": times[0] if times else None,
            "time_end": times[-1] if times else None,
        }

        stats = self._compute_numeric_statistics(values)
        if stats:
            summary["statistics"] = stats

        if values.size:
            if np.isfinite(values[0]):
                summary["start_value"] = float(values[0])
            if np.isfinite(values[-1]):
                summary["end_value"] = float(values[-1])
            if np.isfinite(values[0]) and np.isfinite(values[-1]):
                summary["absolute_change"] = float(values[-1] - values[0])
                if abs(values[0]) > 1e-12:
                    summary["relative_change_pct"] = float((values[-1] - values[0]) / values[0] * 100.0)

            if has_finite:
                max_idx = int(np.nanargmax(values))
                min_idx = int(np.nanargmin(values))
                max_abs_idx = int(np.nanargmax(np.abs(values)))
                summary["extrema"] = {
                    "max_value": float(values[max_idx]),
                    "max_time": times[max_idx] if max_idx < len(times) else None,
                    "min_value": float(values[min_idx]),
                    "min_time": times[min_idx] if min_idx < len(times) else None,
                    "max_abs_value": float(values[max_abs_idx]),
                    "max_abs_time": times[max_abs_idx] if max_abs_idx < len(times) else None,
                }
            summary["value_preview"] = self._preview_list(result.get("values", []))

        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            flags = {
                "is_anomaly": metadata.get("is_anomaly"),
                "is_deseasoned": metadata.get("is_deseasoned"),
                "climatology_period": metadata.get("climatology_period"),
                "mode": metadata.get("mode"),
                "location": metadata.get("location"),
                "region": metadata.get("region"),
                "depth_range": metadata.get("depth_range"),
            }
            flags = {key: value for key, value in flags.items() if value is not None}
            if flags:
                summary["analysis_context"] = flags

        return summary

    def _summarize_climatology_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        labels = result.get("labels", [])
        values = np.asarray(result.get("values", []), dtype=float)
        summary: Dict[str, Any] = {
            "period": result.get("period"),
            "n_labels": len(labels),
            "finite_count": int(np.count_nonzero(np.isfinite(values))) if values.size else 0,
            "has_finite_values": bool(values.size and np.any(np.isfinite(values))),
        }

        if values.size:
            summary["statistics"] = self._compute_numeric_statistics(values)

        if values.size and np.any(np.isfinite(values)):
            peak_idx = int(np.nanargmax(values))
            trough_idx = int(np.nanargmin(values))
            summary["peak_phase"] = {
                "label": labels[peak_idx] if peak_idx < len(labels) else peak_idx,
                "value": float(values[peak_idx]),
            }
            summary["trough_phase"] = {
                "label": labels[trough_idx] if trough_idx < len(labels) else trough_idx,
                "value": float(values[trough_idx]),
            }
            summary["amplitude"] = float(values[peak_idx] - values[trough_idx])
            if len(labels) <= 12:
                summary["cycle"] = [
                    {
                        "label": labels[index] if index < len(labels) else index,
                        "value": float(value) if np.isfinite(value) else None,
                    }
                    for index, value in enumerate(values.tolist())
                ]

        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            summary["metadata"] = self._extract_scalar_fields(metadata)
            source_variable = metadata.get("source_variable")
            if isinstance(source_variable, str) and source_variable.strip():
                summary["variable"] = source_variable.strip()

        return summary

    def _summarize_trend_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        summary = {
            "method": result.get("method"),
            "slope": result.get("slope"),
            "intercept": result.get("intercept"),
            "r_squared": result.get("r_squared"),
            "p_value": result.get("p_value"),
            "std_err": result.get("std_err"),
            "is_significant": result.get("is_significant"),
            "confidence_level": result.get("confidence_level"),
            "n_points": result.get("n_points"),
            "n_valid_points": result.get("n_valid_points"),
            "n_missing_points": result.get("n_missing_points"),
        }
        slope = result.get("slope")
        if isinstance(slope, (int, float)) and np.isfinite(float(slope)):
            if slope > 0:
                summary["trend_direction"] = "positive"
            elif slope < 0:
                summary["trend_direction"] = "negative"
            else:
                summary["trend_direction"] = "flat"

        trend_line = result.get("trend_line")
        if isinstance(trend_line, list) and trend_line:
            try:
                start = float(trend_line[0])
                end = float(trend_line[-1])
            except (TypeError, ValueError):
                start = end = float("nan")
            if np.isfinite(start) and np.isfinite(end):
                summary["fitted_change_over_period"] = float(end - start)
            summary["trend_line_preview"] = self._preview_list(trend_line)

        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            source_variable = metadata.get("source_variable") or metadata.get("variable")
            if source_variable:
                summary["variable"] = source_variable
            for key in ("region", "time_range", "depth_range"):
                value = metadata.get(key)
                if value is not None:
                    summary[key] = value
            context = {
                "variable": source_variable,
                "region": metadata.get("region"),
                "time_range": metadata.get("time_range"),
                "depth_range": metadata.get("depth_range"),
                "mode": metadata.get("mode"),
                "spatial_aggregation": metadata.get("spatial_aggregation"),
            }
            context = {key: value for key, value in context.items() if value is not None}
            if context:
                summary["analysis_context"] = context

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_field_trend_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        summary: Dict[str, Any] = {
            "metadata": self._extract_scalar_fields(result.get("metadata", {})),
        }
        slope_field = result.get("data")
        p_value_field = result.get("p_value")
        significance_mask = result.get("significance_mask")
        xarray_types = self._get_xarray_types()

        if xarray_types and isinstance(slope_field, xarray_types):
            summary["dims"] = list(getattr(slope_field, "dims", []))
            summary["shape"] = list(getattr(slope_field, "shape", []))
            summary["statistics"] = self._compute_numeric_statistics(np.asarray(slope_field.values, dtype=float))
            extrema = self._extract_dataarray_extrema(slope_field)
            if extrema:
                summary["extrema"] = extrema

        if xarray_types and isinstance(p_value_field, xarray_types):
            summary["p_value_statistics"] = self._compute_numeric_statistics(np.asarray(p_value_field.values, dtype=float))

        if xarray_types and isinstance(significance_mask, xarray_types):
            mask_values = np.asarray(significance_mask.values, dtype=bool)
            if mask_values.size:
                summary["significant_fraction"] = float(np.mean(mask_values))

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_lag_correlation_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        lags = [int(lag) for lag in result.get("lags", [])]
        correlations = np.asarray(result.get("correlations", []), dtype=float)
        p_values = np.asarray(result.get("p_values", []), dtype=float)
        summary: Dict[str, Any] = {
            "optimal_lag": result.get("optimal_lag"),
            "optimal_lag_days": result.get("optimal_lag_days"),
            "median_step_days": result.get("median_step_days"),
            "max_correlation": result.get("max_correlation"),
            "peak_correlation": result.get("max_correlation"),
            "confidence_bound": result.get("confidence_bound"),
            "confidence_level": result.get("confidence_level"),
            "n_points": result.get("n_points"),
            "analysis_mode": result.get("analysis_mode"),
            "ts1_is_deseasoned": result.get("ts1_is_deseasoned"),
            "ts2_is_deseasoned": result.get("ts2_is_deseasoned"),
            "ts1_seasonal_cycle_method": result.get("ts1_seasonal_cycle_method"),
            "ts2_seasonal_cycle_method": result.get("ts2_seasonal_cycle_method"),
            "labels": {
                "ts1": result.get("ts1_label"),
                "ts2": result.get("ts2_label"),
            },
            "symmetric_optimal_lag": result.get("optimal_lag"),
            "symmetric_max_correlation": result.get("max_correlation"),
            "best_positive_lag": None,
            "best_positive_correlation": None,
            "best_negative_lag": None,
            "best_negative_correlation": None,
            "lag_curve": self._build_lag_curve(lags, correlations, p_values),
        }

        if correlations.size:
            zero_lag_idx = next((idx for idx, lag in enumerate(lags) if lag == 0), None)
            if zero_lag_idx is not None:
                summary["zero_lag_correlation"] = float(correlations[zero_lag_idx])
                if zero_lag_idx < p_values.size:
                    summary["zero_lag_p_value"] = float(p_values[zero_lag_idx])

            opt_idx = next((idx for idx, lag in enumerate(lags) if lag == result.get("optimal_lag")), None)
            if opt_idx is not None and opt_idx < p_values.size:
                summary["optimal_lag_p_value"] = float(p_values[opt_idx])

            positive_candidate = self._derive_lag_candidate(
                lags,
                correlations,
                p_values,
                include_lag=lambda lag: lag > 0,
            )
            if positive_candidate:
                summary["best_positive_lag"] = positive_candidate["lag"]
                summary["best_positive_correlation"] = positive_candidate["correlation"]
                if "p_value" in positive_candidate:
                    summary["best_positive_p_value"] = positive_candidate["p_value"]

            negative_candidate = self._derive_lag_candidate(
                lags,
                correlations,
                p_values,
                include_lag=lambda lag: lag < 0,
            )
            if negative_candidate:
                summary["best_negative_lag"] = negative_candidate["lag"]
                summary["best_negative_correlation"] = negative_candidate["correlation"]
                if "p_value" in negative_candidate:
                    summary["best_negative_p_value"] = negative_candidate["p_value"]

        return summary

    def _build_lag_curve(
        self,
        lags: List[int],
        correlations: Any,
        p_values: Any,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        lag_curve: List[Dict[str, Any]] = []
        for index, lag in enumerate(lags):
            correlation = float(correlations[index]) if index < len(correlations) and np.isfinite(correlations[index]) else None
            p_value = float(p_values[index]) if index < len(p_values) and np.isfinite(p_values[index]) else None
            lag_curve.append(
                {
                    "lag": int(lag),
                    "correlation": correlation,
                    "p_value": p_value,
                }
            )
        return lag_curve

    def _derive_lag_candidate(
        self,
        lags: List[int],
        correlations: Any,
        p_values: Any,
        *,
        include_lag: Callable[[int], bool],
    ) -> Optional[Dict[str, Any]]:
        import numpy as np

        candidate_indices = [
            index
            for index, lag in enumerate(lags)
            if include_lag(int(lag))
            and index < len(correlations)
            and np.isfinite(correlations[index])
        ]
        if not candidate_indices:
            return None

        max_abs_correlation = float(
            np.nanmax(np.abs(np.asarray([correlations[index] for index in candidate_indices], dtype=float)))
        )
        best_indices = [
            index
            for index in candidate_indices
            if np.isclose(
                abs(float(correlations[index])),
                max_abs_correlation,
                rtol=1e-6,
                atol=1e-12,
            )
        ]
        best_index = min(best_indices, key=lambda index: abs(int(lags[index])))
        candidate = {
            "lag": int(lags[best_index]),
            "correlation": float(correlations[best_index]),
        }
        if best_index < len(p_values) and np.isfinite(p_values[best_index]):
            candidate["p_value"] = float(p_values[best_index])
        return candidate

    def _summarize_profile_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        depth = np.asarray(result.get("depth", []), dtype=float)
        values = np.asarray(result.get("values", []), dtype=float)
        summary: Dict[str, Any] = {
            "variable": result.get("variable"),
            "n_levels": result.get("n_levels"),
            "depth_range": result.get("depth_range"),
            "statistics": self._extract_scalar_fields(result.get("statistics", {})),
            "point": result.get("point"),
            "requested_point": result.get("requested_point"),
        }

        if values.size and depth.size == values.size and np.any(np.isfinite(values)):
            if np.isfinite(values[0]):
                summary["surface_value"] = float(values[0])
            if np.isfinite(values[-1]):
                summary["bottom_value"] = float(values[-1])
            if np.isfinite(values[0]) and np.isfinite(values[-1]):
                summary["surface_to_bottom_change"] = float(values[-1] - values[0])
            max_idx = int(np.nanargmax(values))
            min_idx = int(np.nanargmin(values))
            summary["extrema"] = {
                "max_value": float(values[max_idx]),
                "max_depth": float(depth[max_idx]),
                "min_value": float(values[min_idx]),
                "min_depth": float(depth[min_idx]),
            }

            if values.size > 1:
                gradients = np.diff(values) / np.where(np.diff(depth) == 0, np.nan, np.diff(depth))
                if np.any(np.isfinite(gradients)):
                    grad_idx = int(np.nanargmax(np.abs(gradients)))
                    summary["strongest_gradient"] = {
                        "depth_midpoint": float((depth[grad_idx] + depth[grad_idx + 1]) / 2.0),
                        "gradient": float(gradients[grad_idx]),
                    }

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_section_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        distance = as_numeric_array(result.get("distance_km", []))
        values = as_numeric_array(result.get("values", []))
        summary: Dict[str, Any] = {
            "value_shape": list(values.shape),
            "distance_range_km": [float(distance.min()), float(distance.max())] if distance.size else None,
            "statistics": self._extract_metadata_statistics(result.get("metadata")),
        }

        if "time" in result and isinstance(result["time"], list) and result["time"]:
            summary["time_range"] = [result["time"][0], result["time"][-1]]
        if "depth" in result:
            depth = np.asarray(result.get("depth", []), dtype=float)
            if depth.size:
                summary["depth_range"] = [float(depth.min()), float(depth.max())]

        sample_points = result.get("sample_points", [])
        if isinstance(sample_points, list) and sample_points:
            summary["transect_endpoints"] = {
                "start": sample_points[0],
                "end": sample_points[-1],
            }

        if not summary["statistics"] and values.size <= result_full_list_limit():
            summary["statistics"] = self._compute_numeric_statistics(values)

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_eof_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "n_modes": result.get("n_modes"),
            "cumulative_variance": result.get("cumulative_variance"),
            "total_variance": result.get("total_variance"),
            "metadata": self._extract_scalar_fields(result.get("metadata", {})),
        }

        modes = result.get("modes", [])
        leading_modes = []
        for mode in modes[:3]:
            if not isinstance(mode, dict):
                continue
            mode_summary: Dict[str, Any] = {
                "mode_number": mode.get("mode_number"),
                "variance_explained": mode.get("variance_explained"),
                "eigenvalue": mode.get("eigenvalue"),
            }

            time_series = mode.get("time_series")
            if hasattr(time_series, "values") and hasattr(time_series, "coords"):
                mode_summary["pc_summary"] = self._summarize_dataarray_series(time_series)

            spatial_pattern = mode.get("spatial_pattern")
            if hasattr(spatial_pattern, "values") and hasattr(spatial_pattern, "coords"):
                mode_summary["pattern_extrema"] = self._extract_dataarray_extrema(spatial_pattern)

            leading_modes.append(mode_summary)

        if leading_modes:
            summary["leading_modes"] = leading_modes

        return summary

    def _summarize_spatial_field_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        lon = as_numeric_array(result.get("lon", []))
        lat = as_numeric_array(result.get("lat", []))
        values = as_numeric_array(result.get("values", []))
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        summary: Dict[str, Any] = {
            "value_shape": list(values.shape),
            "title": metadata.get("title"),
            "variable": metadata.get("variable"),
            "units": metadata.get("units"),
            "aggregation": metadata.get("aggregation") or metadata.get("summary_mode"),
            "feature": metadata.get("feature"),
            "event_type": metadata.get("event_type"),
            "summary_mode": metadata.get("summary_mode"),
            "lon_range": [float(lon.min()), float(lon.max())] if lon.size else None,
            "lat_range": [float(lat.min()), float(lat.max())] if lat.size else None,
            "statistics": self._extract_metadata_statistics(metadata),
        }
        if not summary["statistics"] and values.size <= result_full_list_limit():
            summary["statistics"] = self._compute_numeric_statistics(values)

        if values.size <= result_full_list_limit():
            extrema = self._extract_grid_extrema(values, lat, lon)
            if extrema:
                summary["extrema"] = extrema

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_hovmoller_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        times = result.get("time", [])
        spatial_coord = as_numeric_array(result.get("spatial_coord", []))
        values = as_numeric_array(result.get("values", []))
        summary: Dict[str, Any] = {
            "diagram_type": result.get("metadata", {}).get("diagram_type") if isinstance(result.get("metadata"), dict) else None,
            "spatial_dim": result.get("metadata", {}).get("spatial_dim") if isinstance(result.get("metadata"), dict) else None,
            "value_shape": list(values.shape),
            "time_range": [times[0], times[-1]] if times else None,
            "spatial_range": [float(spatial_coord.min()), float(spatial_coord.max())] if spatial_coord.size else None,
            "statistics": self._extract_metadata_statistics(result.get("metadata")),
        }
        if not summary["statistics"] and values.size <= result_full_list_limit():
            summary["statistics"] = self._compute_numeric_statistics(values)

        if values.size <= result_full_list_limit():
            extrema = self._extract_hovmoller_extrema(values, times, spatial_coord)
            if extrema:
                summary["extrema"] = extrema

        if values.ndim == 2 and values.size and values.size <= result_full_list_limit():
            time_means = np.nanmean(values, axis=1)
            spatial_means = np.nanmean(values, axis=0)
            if np.any(np.isfinite(time_means)):
                max_time_idx = int(np.nanargmax(time_means))
                min_time_idx = int(np.nanargmin(time_means))
                summary["time_mean_extrema"] = {
                    "max_time": times[max_time_idx] if max_time_idx < len(times) else None,
                    "max_mean_value": float(time_means[max_time_idx]),
                    "min_time": times[min_time_idx] if min_time_idx < len(times) else None,
                    "min_mean_value": float(time_means[min_time_idx]),
                }
            if np.any(np.isfinite(spatial_means)) and spatial_coord.size:
                max_space_idx = int(np.nanargmax(spatial_means))
                min_space_idx = int(np.nanargmin(spatial_means))
                summary["spatial_mean_extrema"] = {
                    "max_coord": float(spatial_coord[max_space_idx]),
                    "max_mean_value": float(spatial_means[max_space_idx]),
                    "min_coord": float(spatial_coord[min_space_idx]),
                    "min_mean_value": float(spatial_means[min_space_idx]),
                }

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_ts_diagram_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        temperature = result.get("temperature", [])
        salinity = result.get("salinity", [])
        metadata = result.get("metadata", {})
        summary: Dict[str, Any] = {
            "title": metadata.get("title") if isinstance(metadata, dict) else None,
            "n_points": len(temperature),
            "temperature_range": metadata.get("temperature_range") if isinstance(metadata, dict) else None,
            "salinity_range": metadata.get("salinity_range") if isinstance(metadata, dict) else None,
            "color_variable": metadata.get("color_variable") if isinstance(metadata, dict) else None,
            "color_range": metadata.get("color_range") if isinstance(metadata, dict) else None,
            "watermass_bins": metadata.get("watermass_bins") if isinstance(metadata, dict) else None,
        }
        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_watermass_event_association_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        association = result.get("watermass_event_association")
        assignment = result.get("assignment_method")
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        summary: Dict[str, Any] = {
            "type": "watermass_event_association_result",
            "title": metadata.get("title"),
            "event_type": result.get("event_type"),
            "grid_shape": result.get("grid_shape"),
            "evidence_strength": result.get("evidence_strength"),
            "n_bins": metadata.get("n_bins"),
        }
        if isinstance(assignment, dict):
            summary["exact_match_count"] = assignment.get("exact_match_count")
            summary["nearest_fallback_count"] = assignment.get("nearest_fallback_count")
        if isinstance(association, dict):
            for key in (
                "valid_tile_count",
                "hotspot_tile_count",
                "association_score",
                "top_associated_watermass",
                "top_associated_watermass_name",
                "top_associated_enrichment",
                "organized_more_than_background",
                "background_distribution",
                "hotspot_distribution",
                "enrichment_by_watermass",
            ):
                if key in association:
                    summary[key] = association.get(key)
        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_histogram_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "statistics": self._extract_metadata_statistics(result.get("metadata")),
            "n_bins": result.get("metadata", {}).get("n_bins") if isinstance(result.get("metadata"), dict) else None,
            "normalized": result.get("metadata", {}).get("normalized") if isinstance(result.get("metadata"), dict) else None,
        }

        if isinstance(result.get("density"), list) and isinstance(result.get("bin_centers"), list):
            density = result["density"]
            if density:
                peak_idx = max(range(len(density)), key=lambda idx: density[idx])
                summary["peak_bin"] = {
                    "center": result["bin_centers"][peak_idx] if peak_idx < len(result["bin_centers"]) else None,
                    "density": density[peak_idx],
                }

        stats = summary.get("statistics") or {}
        mean = stats.get("mean")
        median = stats.get("median")
        std = stats.get("std")
        if isinstance(mean, (int, float)) and isinstance(median, (int, float)) and isinstance(std, (int, float)) and std > 0:
            delta = mean - median
            if delta > 0.1 * std:
                summary["shape_hint"] = "right_skewed"
            elif delta < -0.1 * std:
                summary["shape_hint"] = "left_skewed"
            else:
                summary["shape_hint"] = "roughly_symmetric"

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_histogram_2d_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        density = np.asarray(result.get("density", []), dtype=float)
        summary: Dict[str, Any] = {
            "statistics": self._extract_metadata_statistics(result.get("metadata")),
            "n_bins": result.get("metadata", {}).get("n_bins") if isinstance(result.get("metadata"), dict) else None,
            "normalized": result.get("metadata", {}).get("normalized") if isinstance(result.get("metadata"), dict) else None,
        }

        if density.ndim == 2 and density.size and np.any(np.isfinite(density)):
            peak_idx = np.unravel_index(int(np.nanargmax(density)), density.shape)
            x_centers = result.get("x_bin_centers", [])
            y_centers = result.get("y_bin_centers", [])
            summary["peak_bin"] = {
                "x_center": x_centers[peak_idx[0]] if peak_idx[0] < len(x_centers) else None,
                "y_center": y_centers[peak_idx[1]] if peak_idx[1] < len(y_centers) else None,
                "density": float(density[peak_idx]),
            }

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_regression_map_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        lat = np.asarray(result.get("lat", []), dtype=float)
        lon = np.asarray(result.get("lon", []), dtype=float)
        slope = np.asarray(result.get("slope", []), dtype=float)
        correlation = np.asarray(result.get("correlation", []), dtype=float)
        significant_mask = np.asarray(result.get("significant_mask", []), dtype=bool)
        summary: Dict[str, Any] = {
            "value_shape": list(slope.shape),
            "lat_range": [float(lat.min()), float(lat.max())] if lat.size else None,
            "lon_range": [float(lon.min()), float(lon.max())] if lon.size else None,
            "metadata": self._extract_scalar_fields(result.get("metadata", {})),
            "slope_statistics": self._compute_numeric_statistics(slope),
            "correlation_statistics": self._compute_numeric_statistics(correlation),
        }

        slope_extrema = self._extract_grid_extrema(slope, lat, lon)
        if slope_extrema:
            summary["slope_extrema"] = slope_extrema
        correlation_extrema = self._extract_grid_extrema(correlation, lat, lon)
        if correlation_extrema:
            summary["correlation_extrema"] = correlation_extrema

        if significant_mask.size:
            valid_mask = np.isfinite(np.asarray(result.get("p_value", []), dtype=float))
            if np.any(valid_mask):
                summary["significant_fraction"] = float(np.mean(significant_mask[valid_mask]))

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_composite_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        difference = result.get("difference", {})
        lat = difference.get("lat", []) if isinstance(difference, dict) else []
        lon = difference.get("lon", []) if isinstance(difference, dict) else []
        values = np.asarray(difference.get("values", []), dtype=float) if isinstance(difference, dict) else np.asarray([])
        summary: Dict[str, Any] = {
            "sample_counts": self._extract_scalar_fields(result.get("sample_counts", {})),
            "metadata": self._extract_scalar_fields(result.get("metadata", {})),
            "difference_statistics": self._extract_metadata_statistics(difference.get("metadata")) if isinstance(difference, dict) else None,
        }

        extrema = self._extract_grid_extrema(values, np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
        if extrema:
            summary["difference_extrema"] = extrema
        if not summary["difference_statistics"]:
            summary["difference_statistics"] = self._compute_numeric_statistics(values)

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_spectrum_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        frequency = np.asarray(result.get("frequency", []), dtype=float)
        power = np.asarray(result.get("power", []), dtype=float)
        summary: Dict[str, Any] = {
            "n_frequencies": int(frequency.size),
            "metadata": self._extract_scalar_fields(result.get("metadata", {})),
            "dominant_peaks": result.get("dominant_peaks"),
        }

        if power.size and np.any(np.isfinite(power)):
            peak_index = int(np.nanargmax(power))
            periods = result.get("period", [])
            summary["global_peak"] = {
                "frequency": float(frequency[peak_index]),
                "period": periods[peak_index] if peak_index < len(periods) else None,
                "power": float(power[peak_index]),
            }

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_layer_transport_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        layers = result.get("layers", [])
        summary: Dict[str, Any] = {
            "times": self._preview_list(result.get("times", [])),
            "layer_count": len(layers),
            "metadata": self._extract_scalar_fields(result.get("metadata", {})),
        }

        if isinstance(layers, list) and layers:
            summary["layer_labels"] = [layer.get("label") for layer in layers[:5]]
            layer_means = []
            for layer in layers:
                values = np.asarray(layer.get("values", []), dtype=float)
                valid = values[np.isfinite(values)]
                if valid.size:
                    layer_means.append(
                        {
                            "label": layer.get("label"),
                            "mean": float(np.mean(valid)),
                        }
                    )
            if layer_means:
                dominant = max(layer_means, key=lambda item: abs(item["mean"]))
                summary["layer_mean_preview"] = layer_means[:5]
                summary["dominant_layer_by_mean"] = dominant

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_event_detection_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        events = result.get("events", [])
        summary: Dict[str, Any] = {
            "event_type": result.get("event_type"),
            "event_count": len(events),
            "statistics": self._extract_scalar_fields(result.get("statistics", {})),
        }

        if events:
            centers = [event.get("center") for event in events if isinstance(event.get("center"), dict)]
            if centers:
                lons = [center["lon"] for center in centers if "lon" in center]
                lats = [center["lat"] for center in centers if "lat" in center]
                if lons and lats:
                    summary["spatial_distribution"] = {
                        "centroid": {
                            "lon": float(sum(lons) / len(lons)),
                            "lat": float(sum(lats) / len(lats)),
                        },
                        "lon_range": [float(min(lons)), float(max(lons))],
                        "lat_range": [float(min(lats)), float(max(lats))],
                    }

            time_indices = [event.get("time_index") for event in events if isinstance(event.get("time_index"), (int, float))]
            if time_indices:
                summary["time_index_range"] = [int(min(time_indices)), int(max(time_indices))]

            duration_values = [event.get("duration_days") for event in events if isinstance(event.get("duration_days"), (int, float))]
            if duration_values:
                summary["duration_days"] = {
                    "mean": float(sum(duration_values) / len(duration_values)),
                    "max": float(max(duration_values)),
                }

            by_area = self._select_top_event(events, "area_km2")
            if by_area is not None:
                summary["largest_event"] = by_area

            intensity_field = self._select_event_intensity_field(events)
            if intensity_field:
                by_intensity = self._select_top_event(events, intensity_field)
                if by_intensity is not None:
                    summary["strongest_event"] = by_intensity

        return summary

    def _summarize_event_statistics_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        groups = result.get("groups", {})
        summary: Dict[str, Any] = {
            "total_count": result.get("total_count"),
            "group_by": result.get("group_by"),
        }

        if isinstance(groups, dict) and groups:
            normalized_groups = {
                str(key): self._extract_scalar_fields(value)
                for key, value in groups.items()
                if isinstance(value, dict)
            }
            summary["n_groups"] = len(normalized_groups)
            summary["group_label_mode"] = self._infer_event_group_label_mode(
                normalized_groups.keys(),
                result.get("group_by"),
            )
            if summary["group_label_mode"] == "index_proxy":
                summary["group_label_note"] = (
                    "Group labels are index-derived proxy bins, not explicit calendar labels."
                )

            total_count = result.get("total_count")
            if isinstance(total_count, (int, float)) and total_count > 0:
                for group_summary in normalized_groups.values():
                    count = group_summary.get("count")
                    if isinstance(count, (int, float)):
                        group_summary["share_of_total_count_pct"] = float(count / total_count * 100.0)

            nonzero_groups = {
                key: value
                for key, value in normalized_groups.items()
                if isinstance(value.get("count"), (int, float)) and value.get("count", 0) > 0
            }
            summary["n_nonzero_groups"] = len(nonzero_groups)
            zero_groups = [key for key, value in normalized_groups.items() if value.get("count", 0) == 0]
            if zero_groups:
                summary["zero_groups"] = zero_groups

            if nonzero_groups:
                ranked_by_count = sorted(
                    nonzero_groups.items(),
                    key=lambda item: (
                        item[1].get("count", 0),
                        item[1].get("share_of_total_count_pct", 0),
                    ),
                    reverse=True,
                )
                top_count_group = ranked_by_count[0]
                summary["top_group_by_count"] = {
                    "group": top_count_group[0],
                    "count": top_count_group[1].get("count"),
                    "share_of_total_count_pct": top_count_group[1].get("share_of_total_count_pct"),
                }
                summary["groups_ranked_by_count"] = [
                    {
                        "group": key,
                        "count": value.get("count"),
                        "share_of_total_count_pct": value.get("share_of_total_count_pct"),
                        "mean_intensity": value.get("mean_intensity"),
                        "mean_area_km2": value.get("mean_area_km2"),
                        "total_area_km2": value.get("total_area_km2"),
                    }
                    for key, value in ranked_by_count[:3]
                ]
                if len(ranked_by_count) > 1:
                    count_gap = ranked_by_count[0][1].get("count", 0) - ranked_by_count[1][1].get("count", 0)
                    summary["count_gap_top_vs_second"] = float(count_gap)

                intensity_groups = [
                    (key, value) for key, value in nonzero_groups.items()
                    if isinstance(value.get("mean_intensity"), (int, float))
                ]
                if intensity_groups:
                    ranked_by_mean_intensity = sorted(
                        intensity_groups,
                        key=lambda item: item[1].get("mean_intensity", 0),
                        reverse=True,
                    )
                    top_intensity_group = ranked_by_mean_intensity[0]
                    summary["top_group_by_mean_intensity"] = {
                        "group": top_intensity_group[0],
                        "mean_intensity": top_intensity_group[1].get("mean_intensity"),
                    }
                    summary["groups_ranked_by_mean_intensity"] = [
                        {
                            "group": key,
                            "mean_intensity": value.get("mean_intensity"),
                            "max_intensity": value.get("max_intensity"),
                            "count": value.get("count"),
                        }
                        for key, value in ranked_by_mean_intensity[:3]
                    ]

                ordered_groups = self._order_event_groups(normalized_groups.keys(), result.get("group_by"))
                active_groups_in_order = [
                    (key, normalized_groups[key])
                    for key in ordered_groups
                    if key in nonzero_groups
                ]
                if active_groups_in_order:
                    first_group, first_values = active_groups_in_order[0]
                    last_group, last_values = active_groups_in_order[-1]
                    summary["first_active_group"] = {
                        "group": first_group,
                        "count": first_values.get("count"),
                        "mean_intensity": first_values.get("mean_intensity"),
                    }
                    summary["last_active_group"] = {
                        "group": last_group,
                        "count": last_values.get("count"),
                        "mean_intensity": last_values.get("mean_intensity"),
                    }
                    if len(active_groups_in_order) > 1:
                        first_count = first_values.get("count")
                        last_count = last_values.get("count")
                        if isinstance(first_count, (int, float)) and isinstance(last_count, (int, float)):
                            summary["count_change_first_to_last_active"] = {
                                "from_group": first_group,
                                "to_group": last_group,
                                "absolute_change": float(last_count - first_count),
                                "relative_change_pct": (
                                    float((last_count - first_count) / first_count * 100.0)
                                    if first_count
                                    else None
                                ),
                            }
                        first_intensity = first_values.get("mean_intensity")
                        last_intensity = last_values.get("mean_intensity")
                        if isinstance(first_intensity, (int, float)) and isinstance(last_intensity, (int, float)):
                            summary["mean_intensity_change_first_to_last_active"] = {
                                "from_group": first_group,
                                "to_group": last_group,
                                "absolute_change": float(last_intensity - first_intensity),
                            }

                if len(normalized_groups) <= 12:
                    summary["groups"] = {
                        key: normalized_groups[key]
                        for key in ordered_groups
                    }

        return summary

    def _summarize_event_spatial_distribution_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        summary = self._extract_scalar_fields(result)
        summary["type"] = "event_spatial_distribution_result"
        return summary

    def _summarize_event_comparison_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        summary = self._extract_scalar_fields(result)
        summary["type"] = "event_comparison_result"
        comparison = result.get("comparison")
        if isinstance(comparison, dict) and comparison:
            normalized_comparison = {
                str(label): self._extract_scalar_fields(stats)
                for label, stats in comparison.items()
                if isinstance(stats, dict)
            }
            if normalized_comparison:
                summary["comparison"] = normalized_comparison
                period_labels = list(normalized_comparison.keys())
                summary["period_labels"] = period_labels[:2]
                if period_labels:
                    summary["period1_label"] = period_labels[0]
                    first = normalized_comparison.get(period_labels[0], {})
                    if isinstance(first.get("total_count"), (int, float)):
                        summary["period1_total_count"] = first["total_count"]
                    if isinstance(first.get("mean_intensity"), (int, float)):
                        summary["period1_mean_intensity"] = first["mean_intensity"]
                if len(period_labels) > 1:
                    summary["period2_label"] = period_labels[1]
                    second = normalized_comparison.get(period_labels[1], {})
                    if isinstance(second.get("total_count"), (int, float)):
                        summary["period2_total_count"] = second["total_count"]
                    if isinstance(second.get("mean_intensity"), (int, float)):
                        summary["period2_mean_intensity"] = second["mean_intensity"]
        changes = result.get("changes")
        if isinstance(changes, dict):
            summary["changes"] = self._extract_scalar_fields(changes)
        return summary

    def _summarize_mechanism_score_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        summary = {
            "type": "mechanism_score_result",
            "top_mechanism": result.get("top_mechanism"),
            "claim_strength": result.get("claim_strength"),
            "supporting_evidence": self._preview_list(result.get("supporting_evidence", [])),
            "conflicting_evidence": self._preview_list(result.get("conflicting_evidence", [])),
            "metadata": self._extract_scalar_fields(result.get("metadata", {})),
        }
        subregion_breakdown = self._extract_scalar_object_list(result.get("subregion_breakdown"))
        if subregion_breakdown:
            summary["subregion_breakdown"] = subregion_breakdown
            summary["n_subregions"] = len(subregion_breakdown)
            summary["n_valid_subregions"] = sum(
                1 for item in subregion_breakdown if item.get("status") == "ok"
            )
        proxy_breakdowns = self._extract_scalar_object_list(result.get("proxy_breakdowns"))
        if proxy_breakdowns:
            summary["proxy_breakdowns"] = proxy_breakdowns
            summary["n_proxy_breakdowns"] = len(proxy_breakdowns)
        candidates = result.get("candidate_mechanisms", [])
        if isinstance(candidates, list) and candidates:
            normalized_candidates = []
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                score = item.get("score")
                normalized_candidates.append({
                    "name": name,
                    "score": score,
                })
            if normalized_candidates:
                summary["candidate_mechanisms"] = normalized_candidates
                summary["n_candidates"] = len(normalized_candidates)
                top_candidate = normalized_candidates[0]
                if top_candidate.get("name") is not None:
                    summary.setdefault("top_mechanism", top_candidate.get("name"))
                if top_candidate.get("score") is not None:
                    summary["top_score"] = top_candidate.get("score")
        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_evidence_report_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        summary = {
            "type": "evidence_report_result",
            "claim_strength": result.get("claim_strength"),
            "supported_claims": self._preview_list(result.get("supported_claims", [])),
            "limited_claims": self._preview_list(result.get("limited_claims", [])),
            "untestable_claims": self._preview_list(result.get("untestable_claims", [])),
            "residual_or_uncertainty": self._preview_list(result.get("residual_or_uncertainty", [])),
            "metadata": self._extract_scalar_fields(result.get("metadata", {})),
        }
        supported_claims = result.get("supported_claims")
        limited_claims = result.get("limited_claims")
        untestable_claims = result.get("untestable_claims")
        summary["n_supported_claims"] = len(supported_claims) if isinstance(supported_claims, list) else 0
        summary["n_limited_claims"] = len(limited_claims) if isinstance(limited_claims, list) else 0
        summary["n_untestable_claims"] = len(untestable_claims) if isinstance(untestable_claims, list) else 0
        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_environment_assessment_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        summary = {
            "type": "environment_assessment_result",
            "overall_verdict": result.get("overall_verdict"),
            "overall_support_strength": result.get("overall_support_strength"),
            "overall_narrative": result.get("overall_narrative"),
            "key_pressures": self._preview_list(result.get("key_pressures", [])),
            "stabilizing_signals": self._preview_list(result.get("stabilizing_signals", [])),
            "supporting_evidence": self._preview_list(result.get("supporting_evidence", [])),
            "uncertainties": self._preview_list(result.get("uncertainties", [])),
            "metadata": self._extract_scalar_fields(result.get("metadata", {})),
        }

        branches = result.get("branch_assessments")
        if isinstance(branches, list):
            normalized_branches = []
            for item in branches:
                if not isinstance(item, dict):
                    continue
                normalized_branches.append({
                    "name": item.get("name"),
                    "indicator_label": item.get("indicator_label"),
                    "direction": item.get("direction"),
                    "support_strength": item.get("support_strength"),
                    "summary": item.get("summary"),
                    "output_type": item.get("output_type"),
                    "score_contribution": item.get("score_contribution"),
                })
            if normalized_branches:
                summary["branch_assessments"] = normalized_branches
                summary["n_branches"] = len(normalized_branches)
                summary["n_supported_branches"] = sum(
                    1 for item in normalized_branches if item.get("support_strength") == "supported"
                )
                summary["n_limited_branches"] = sum(
                    1 for item in normalized_branches if item.get("support_strength") == "limited"
                )
                summary["n_untestable_branches"] = sum(
                    1 for item in normalized_branches if item.get("support_strength") == "untestable"
                )

        return {key: value for key, value in summary.items() if value is not None}

    def _summarize_policy_recommendation_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        actions = result.get("recommended_actions")
        evidence_table = result.get("evidence_table")
        summary = {
            "type": "policy_recommendation_result",
            "region_scope": result.get("region_scope"),
            "policy_context": result.get("policy_context"),
            "management_objective": result.get("management_objective"),
            "priority_level": result.get("priority_level"),
            "policy_summary": result.get("policy_summary"),
            "recommended_actions": self._preview_list(actions if isinstance(actions, list) else []),
            "monitoring_priorities": self._preview_list(result.get("monitoring_priorities", [])),
            "governance_notes": self._preview_list(result.get("governance_notes", [])),
            "evidence_constraints": self._preview_list(result.get("evidence_constraints", [])),
            "evidence_table": self._preview_list(evidence_table if isinstance(evidence_table, list) else []),
            "metadata": self._extract_scalar_fields(result.get("metadata", {})),
        }
        if isinstance(actions, list):
            summary["n_recommended_actions"] = len(actions)
        if isinstance(evidence_table, list):
            summary["n_evidence_entries"] = len(evidence_table)
        return {key: value for key, value in summary.items() if value is not None}

    def _extract_coord_ranges(self, data: Any) -> Dict[str, Any]:
        if is_partitioned_xarray(data):
            return dict(data.coord_ranges())

        coord_ranges: Dict[str, Any] = {}
        for coord_name in ("time", "depth", "lat", "lon"):
            if coord_name not in getattr(data, "coords", {}):
                continue
            coord = data.coords[coord_name].values
            if getattr(coord, "size", 0) == 0:
                continue
            if coord_name == "time":
                coord_ranges["time_range"] = [str(coord[0]), str(coord[-1])]
            else:
                numeric = self._clean_numeric_coord_values(coord_name, coord)
                if numeric.size == 0:
                    continue
                coord_ranges[f"{coord_name}_range"] = [
                    float(numeric.min()),
                    float(numeric.max()),
                ]
        return coord_ranges

    def _clean_numeric_coord_values(self, coord_name: str, values: Any) -> Any:
        import numpy as np

        try:
            numeric = np.asarray(values, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return np.asarray([], dtype=float)

        finite = numeric[np.isfinite(numeric)]
        if coord_name == "depth":
            finite = finite[np.abs(finite) < _SENTINEL_DEPTH_ABS_THRESHOLD]
        return finite

    def _compute_numeric_statistics(self, values: Any) -> Optional[Dict[str, Any]]:
        import numpy as np

        try:
            array = np.asarray(values, dtype=float)
        except Exception:
            return None

        if array.size == 0:
            return None

        valid = array[np.isfinite(array)]
        if valid.size == 0:
            return {
                "n_total": int(array.size),
                "n_valid": 0,
                "nan_fraction": 1.0,
            }

        return {
            "mean": float(np.nanmean(valid)),
            "std": float(np.nanstd(valid)),
            "min": float(np.nanmin(valid)),
            "max": float(np.nanmax(valid)),
            "median": float(np.nanmedian(valid)),
            "p05": float(np.nanpercentile(valid, 5)),
            "p95": float(np.nanpercentile(valid, 95)),
            "n_valid": int(valid.size),
            "n_total": int(array.size),
            "nan_fraction": float(1.0 - valid.size / array.size),
        }

    def _extract_dataarray_extrema(self, data: Any) -> Optional[Dict[str, Any]]:
        import numpy as np

        try:
            values = np.asarray(data.values, dtype=float)
        except Exception:
            return None

        if values.size == 0 or not np.any(np.isfinite(values)):
            return None

        max_index = np.unravel_index(int(np.nanargmax(values)), values.shape)
        min_index = np.unravel_index(int(np.nanargmin(values)), values.shape)

        return {
            "max": {
                "value": float(values[max_index]),
                "coords": self._coords_from_index(data, max_index),
            },
            "min": {
                "value": float(values[min_index]),
                "coords": self._coords_from_index(data, min_index),
            },
        }

    def _summarize_dataarray_series(self, data: Any) -> Dict[str, Any]:
        import numpy as np

        values = np.asarray(data.values, dtype=float)
        summary: Dict[str, Any] = {
            "statistics": self._compute_numeric_statistics(values),
        }
        if values.size and np.any(np.isfinite(values)) and "time" in getattr(data, "coords", {}):
            times = [str(value) for value in data.coords["time"].values]
            max_idx = int(np.nanargmax(values))
            min_idx = int(np.nanargmin(values))
            summary["extrema"] = {
                "max_value": float(values[max_idx]),
                "max_time": times[max_idx] if max_idx < len(times) else None,
                "min_value": float(values[min_idx]),
                "min_time": times[min_idx] if min_idx < len(times) else None,
            }
        return summary

    def _coords_from_index(self, data: Any, index: Any) -> Dict[str, Any]:
        import numpy as np

        coords: Dict[str, Any] = {}
        for dim, dim_index in zip(getattr(data, "dims", []), index):
            try:
                value = data.coords[dim].values[dim_index]
            except Exception:
                continue
            if isinstance(value, np.datetime64):
                coords[dim] = str(value)
                continue
            if hasattr(value, "item"):
                value = value.item()
            if dim == "time":
                coords[dim] = str(value)
            elif isinstance(value, (int, float)):
                coords[dim] = float(value)
            else:
                coords[dim] = str(value)
        return coords

    def _extract_grid_extrema(self, values: Any, lat: Any, lon: Any) -> Optional[Dict[str, Any]]:
        import numpy as np

        if values.size == 0 or values.ndim != 2 or not np.any(np.isfinite(values)):
            return None

        max_idx = np.unravel_index(int(np.nanargmax(values)), values.shape)
        min_idx = np.unravel_index(int(np.nanargmin(values)), values.shape)

        return {
            "max": {
                "value": float(values[max_idx]),
                "lat": float(lat[max_idx[0]]) if lat.size > max_idx[0] else None,
                "lon": float(lon[max_idx[1]]) if lon.size > max_idx[1] else None,
            },
            "min": {
                "value": float(values[min_idx]),
                "lat": float(lat[min_idx[0]]) if lat.size > min_idx[0] else None,
                "lon": float(lon[min_idx[1]]) if lon.size > min_idx[1] else None,
            },
        }

    def _extract_hovmoller_extrema(self, values: Any, times: Any, spatial_coord: Any) -> Optional[Dict[str, Any]]:
        import numpy as np

        if values.size == 0 or values.ndim != 2 or not np.any(np.isfinite(values)):
            return None

        max_idx = np.unravel_index(int(np.nanargmax(values)), values.shape)
        min_idx = np.unravel_index(int(np.nanargmin(values)), values.shape)

        return {
            "max": {
                "value": float(values[max_idx]),
                "time": times[max_idx[0]] if max_idx[0] < len(times) else None,
                "coord": float(spatial_coord[max_idx[1]]) if spatial_coord.size > max_idx[1] else None,
            },
            "min": {
                "value": float(values[min_idx]),
                "time": times[min_idx[0]] if min_idx[0] < len(times) else None,
                "coord": float(spatial_coord[min_idx[1]]) if spatial_coord.size > min_idx[1] else None,
            },
        }

    def _extract_scalar_fields(self, value: Any) -> Dict[str, Any]:
        scalar_fields: Dict[str, Any] = {}
        if not isinstance(value, dict):
            return scalar_fields

        for key, item in value.items():
            if self._is_json_scalar(item):
                scalar_fields[key] = self._make_json_safe(item)
            elif isinstance(item, dict):
                nested = {
                    nested_key: self._make_json_safe(nested_item)
                    for nested_key, nested_item in item.items()
                    if self._is_json_scalar(nested_item)
                }
                if nested:
                    scalar_fields[key] = nested
            elif isinstance(item, list) and item and len(item) <= 8 and all(self._is_json_scalar(entry) for entry in item):
                scalar_fields[key] = [self._make_json_safe(entry) for entry in item]
        return scalar_fields

    def _extract_scalar_object_list(self, value: Any) -> List[Dict[str, Any]]:
        normalized_items: List[Dict[str, Any]] = []
        if not isinstance(value, list):
            return normalized_items

        for item in value:
            if not isinstance(item, dict):
                continue
            normalized: Dict[str, Any] = {}
            for key, nested_value in item.items():
                if self._is_json_scalar(nested_value):
                    normalized[key] = self._make_json_safe(nested_value)
                elif isinstance(nested_value, dict):
                    scalar_nested = self._extract_scalar_fields(nested_value)
                    if scalar_nested:
                        normalized[key] = scalar_nested
                elif isinstance(nested_value, list):
                    if nested_value and all(self._is_json_scalar(entry) for entry in nested_value):
                        normalized[key] = [self._make_json_safe(entry) for entry in nested_value]
                    elif nested_value and all(isinstance(entry, dict) for entry in nested_value):
                        nested_items = self._extract_scalar_object_list(nested_value)
                        if nested_items:
                            normalized[key] = nested_items
            if normalized:
                normalized_items.append(normalized)
        return normalized_items

    def _extract_metadata_statistics(self, metadata: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(metadata, dict):
            return None
        statistics = metadata.get("statistics")
        if isinstance(statistics, dict):
            return self._extract_scalar_fields(statistics)
        return None

    def _preview_list(self, values: Any, n: int = 3) -> Any:
        if not isinstance(values, list):
            return values
        if len(values) <= n * 2:
            return values
        return values[:n] + ["..."] + values[-n:]

    def _normalize_temporal_axes(self, result: Dict[str, Any]) -> Dict[str, Any]:
        output_type = result.get("output_type")
        if output_type not in {
            "timeseries_result",
            "trend_result",
            "hovmoller_result",
            "layer_transport_result",
        }:
            return result

        for key in ("times", "time"):
            if key in result:
                result[key] = self._normalize_time_axis_values(result[key])
        return result

    def _normalize_time_axis_values(self, values: Any) -> Any:
        if values is None or isinstance(values, (str, bytes, dict)):
            return values
        if hasattr(values, "tolist"):
            try:
                values = values.tolist()
            except Exception:
                pass
        if isinstance(values, tuple):
            values = list(values)
        if not isinstance(values, list):
            return values
        return [self._format_time_axis_value(value) for value in values]

    def _format_time_axis_value(self, value: Any) -> Any:
        import math
        from datetime import date, datetime

        import numpy as np

        if value is None:
            return None
        if isinstance(value, np.datetime64):
            return np.datetime_as_string(value.astype("datetime64[s]"), unit="s")
        if isinstance(value, datetime):
            return value.replace(tzinfo=None).isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if hasattr(value, "to_pydatetime"):
            try:
                return value.to_pydatetime().replace(tzinfo=None).isoformat()
            except Exception:
                pass

        if isinstance(value, str):
            epoch_label = self._format_epoch_like_time(value.strip())
            return epoch_label or value

        if isinstance(value, (bool, np.bool_)):
            return value
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if not math.isfinite(numeric):
                return None
            epoch_label = self._format_epoch_like_time(value)
            return epoch_label if epoch_label is not None else value
        return value

    def _format_epoch_like_time(self, value: Any) -> Optional[str]:
        import math

        numeric_text: Optional[str] = None
        if isinstance(value, str):
            numeric_text = value
            if not numeric_text:
                return None
            try:
                numeric = float(numeric_text)
            except ValueError:
                return None
        else:
            numeric = float(value)

        if not math.isfinite(numeric):
            return None
        abs_value = abs(numeric)
        if abs_value < 1_000_000_000:
            return None
        if abs_value >= 100_000_000_000_000_000:
            unit = "ns"
        elif abs_value >= 100_000_000_000_000:
            unit = "us"
        elif abs_value >= 100_000_000_000:
            unit = "ms"
        else:
            unit = "s"

        try:
            import pandas as pd

            raw_value = int(numeric_text) if numeric_text and numeric_text.lstrip("-").isdigit() else numeric
            timestamp = pd.to_datetime(raw_value, unit=unit, errors="coerce")
            if pd.isna(timestamp):
                return None
            if getattr(timestamp, "tzinfo", None) is not None:
                timestamp = timestamp.tz_convert(None)
            return timestamp.isoformat()
        except Exception:
            return None

    def _is_json_scalar(self, value: Any) -> bool:
        import numpy as np

        scalar_types = (str, bool, int, float, np.bool_, np.integer, np.floating)
        return value is None or isinstance(value, scalar_types)

    def _make_json_safe(self, value: Any) -> Any:
        import numpy as np

        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.datetime64):
            return str(value)
        if isinstance(value, np.ndarray):
            return self._make_json_safe(json_safe_array(value))
        if isinstance(value, dict):
            return {str(key): self._make_json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._make_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self._make_json_safe(item) for item in value]
        if hasattr(value, "item"):
            try:
                return self._make_json_safe(value.item())
            except Exception:
                pass
        return str(value)

    def _select_event_intensity_field(self, events: Any) -> Optional[str]:
        candidate_fields = (
            "max_intensity",
            "mean_intensity",
            "intensity",
            "max_gradient",
            "max_vorticity",
            "max_chlorophyll",
            "min_oxygen",
        )
        for field in candidate_fields:
            if any(isinstance(event.get(field), (int, float)) for event in events if isinstance(event, dict)):
                return field
        return None

    def _select_top_event(self, events: Any, field: str) -> Optional[Dict[str, Any]]:
        numeric_events = [
            event for event in events
            if isinstance(event, dict) and isinstance(event.get(field), (int, float))
        ]
        if not numeric_events:
            return None

        top_event = max(numeric_events, key=lambda event: event.get(field, float("-inf")))
        summary = {
            "event_id": top_event.get("event_id"),
            "metric": field,
            "value": top_event.get(field),
        }
        for key in ("time_index", "timestamp", "end_timestamp", "duration_days", "area_km2", "n_pixels", "center"):
            if key in top_event:
                summary[key] = top_event[key]
        return summary

    def _order_event_groups(self, group_labels: Any, group_by: Any) -> list[str]:
        labels = [str(label) for label in group_labels]
        month_order = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        }
        season_order = {
            "DJF": 1, "Winter": 1,
            "MAM": 2, "Spring": 2,
            "JJA": 3, "Summer": 3,
            "SON": 4, "Fall": 4, "Autumn": 4,
        }

        def _sort_key(label: str) -> tuple[int, Any]:
            if group_by == "month":
                if label in month_order:
                    return (0, month_order[label])
                if label.isdigit():
                    return (0, int(label))
            elif group_by == "season":
                if label in season_order:
                    return (0, season_order[label])
            elif group_by == "year" and label.isdigit():
                return (0, int(label))
            return (1, label)

        return sorted(labels, key=_sort_key)

    def _infer_event_group_label_mode(self, group_labels: Any, group_by: Any) -> str:
        labels = [str(label) for label in group_labels]
        if not labels:
            return "unknown"

        if group_by == "month":
            if all(label.isdigit() for label in labels):
                return "index_proxy"
            return "calendar"

        if group_by == "season":
            known_labels = {"DJF", "MAM", "JJA", "SON", "Winter", "Spring", "Summer", "Fall", "Autumn"}
            if all(label in known_labels for label in labels):
                return "calendar"
            if all(label.isdigit() for label in labels):
                return "index_proxy"
            return "ordered_groups"

        if group_by == "year":
            if all(label.isdigit() for label in labels):
                return "year_or_index"
            return "ordered_groups"

        return "ordered_groups"

    def _is_data_container_result(self, value: Any) -> bool:
        return isinstance(value, dict) and value.get("output_type") == "data_container_result" and "data" in value

    def _is_normalized_result(self, value: Any) -> bool:
        return isinstance(value, dict) and "output_type" in value

    def _next_ref_id(self) -> str:
        self._counter += 1
        return f"result_{self._counter}"

    def _get_xarray_types(self):
        try:
            import xarray as xr
        except ImportError:
            return None
        return (xr.DataArray, xr.Dataset)
