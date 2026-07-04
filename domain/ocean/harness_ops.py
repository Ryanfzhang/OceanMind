"""Generic harness operations for shape-first OceanMind task graphs."""

from __future__ import annotations

import ast
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from domain.ocean.data_access.load import (
    _normalize_depth_range,
    get_depth_dim,
    resolve_pending_bottom_selection,
)
from domain.ocean.data_access.partitioned import (
    find_partitioned_values,
    materialize_partitioned_xarray,
)
from domain.ocean.dask_utils import dataarray_to_numpy, report_phase


def select_vertical(
    data: xr.DataArray,
    mode: str = "as_is",
    depth_value: Optional[float] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    relative_to: Optional[str] = None,
    band_thickness_m: Optional[float] = None,
    aggregation: Optional[str] = None,
    retain_depth: bool = False,
) -> xr.DataArray:
    """
    Apply a semantic vertical selection to a field.

    This is intentionally generic: planner-generated task graphs can map user
    language to this operation before choosing downstream diagnostics.
    """
    if find_partitioned_values((data,)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="select_vertical",
            tool_func=select_vertical,
            params={
                "data": data,
                "mode": mode,
                "depth_value": depth_value,
                "depth_range": depth_range,
                "relative_to": relative_to,
                "band_thickness_m": band_thickness_m,
                "aggregation": aggregation,
                "retain_depth": retain_depth,
            },
        )

    field = materialize_partitioned_xarray(data)
    depth_dim = get_depth_dim(field)
    if depth_dim is None:
        return field

    normalized_mode = str(mode or "as_is").strip().lower()
    if normalized_mode in {"as_is", "none", "unspecified"}:
        return field
    if normalized_mode == "surface":
        selected = _select_nearest_depth(field, depth_dim, 0.0)
        selected.attrs.update({"vertical_mode": "surface", "vertical_aggregation": aggregation})
        return _maybe_aggregate_depth(selected, depth_dim, aggregation, retain_depth)
    if normalized_mode == "fixed_depth":
        if depth_value is None:
            raise ValueError("fixed_depth selection requires depth_value")
        selected = _select_nearest_depth(field, depth_dim, float(depth_value))
        selected.attrs.update({"vertical_mode": "fixed_depth", "depth_value": float(depth_value)})
        return _maybe_aggregate_depth(selected, depth_dim, aggregation, retain_depth)
    if normalized_mode == "depth_range":
        if depth_range is None:
            raise ValueError("depth_range selection requires depth_range")
        selected = _select_depth_range(field, depth_dim, depth_range)
        selected.attrs.update({"vertical_mode": "depth_range", "depth_range": list(depth_range)})
        return _maybe_aggregate_depth(selected, depth_dim, aggregation, retain_depth)
    if normalized_mode == "bottom":
        selected = resolve_pending_bottom_selection(field)
        if depth_dim in selected.dims and selected.sizes.get(depth_dim, 0) != 1:
            selected = _select_bottom(field, depth_dim)
        selected.attrs.update({"vertical_mode": "bottom"})
        return _maybe_aggregate_depth(selected, depth_dim, aggregation, retain_depth)
    if normalized_mode in {"bottom_band", "relative_to_bottom"} or str(relative_to or "").lower() == "bottom":
        thickness = float(band_thickness_m or 0.0)
        if thickness <= 0.0:
            raise ValueError("bottom_band selection requires a positive band_thickness_m")
        selected = _select_bottom_band(field, depth_dim, thickness)
        selected.attrs.update(
            {
                "vertical_mode": "bottom_band",
                "relative_to": "bottom",
                "band_thickness_m": thickness,
            }
        )
        return _maybe_aggregate_depth(selected, depth_dim, aggregation, retain_depth)

    raise ValueError(f"Unsupported vertical selection mode: {mode}")


def build_threshold_mask(
    data: xr.DataArray,
    threshold: float,
    comparison: str = "lt",
    mask_name: str = "threshold_mask",
) -> xr.DataArray:
    """Build a boolean diagnostic mask from a numeric field."""
    field = materialize_partitioned_xarray(data)
    comp = str(comparison or "lt").strip().lower()
    if comp in {"lt", "<", "less_than", "below"}:
        mask = field < float(threshold)
    elif comp in {"le", "<=", "less_equal", "at_or_below"}:
        mask = field <= float(threshold)
    elif comp in {"gt", ">", "greater_than", "above"}:
        mask = field > float(threshold)
    elif comp in {"ge", ">=", "greater_equal", "at_or_above"}:
        mask = field >= float(threshold)
    else:
        raise ValueError(f"Unsupported threshold comparison: {comparison}")
    mask.name = mask_name
    mask.attrs = {
        "mask_type": "threshold",
        "threshold": float(threshold),
        "comparison": comp,
        "source_variable": field.name or "unknown",
    }
    return mask


def build_condition_mask(
    fields: Mapping[str, xr.DataArray],
    expression: str,
    mask_name: str = "condition_mask",
) -> xr.DataArray:
    """Build a boolean mask from a constrained xarray expression DSL."""
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("build_condition_mask requires at least one named field")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("build_condition_mask requires a non-empty expression")

    normalized_fields: Dict[str, xr.DataArray] = {}
    for name, value in fields.items():
        field_name = str(name)
        if not field_name.isidentifier():
            raise ValueError(f"Condition field name must be a valid identifier: {field_name}")
        field = materialize_partitioned_xarray(value)
        if not hasattr(field, "dims"):
            raise ValueError(f"Condition field {field_name} is not an xarray DataArray")
        normalized_fields[field_name] = field

    try:
        aligned_values = xr.align(*normalized_fields.values(), join="inner")
    except Exception as exc:
        raise ValueError(f"Condition fields could not be aligned: {exc}") from exc
    aligned_fields = dict(zip(normalized_fields.keys(), aligned_values))
    if any(any(size == 0 for size in field.sizes.values()) for field in aligned_fields.values()):
        raise ValueError("Condition fields aligned to an empty coordinate intersection")

    tree = ast.parse(expression, mode="eval")
    evaluator = _ConditionMaskEvaluator(aligned_fields)
    result = evaluator.visit(tree)
    if not hasattr(result, "dims"):
        raise ValueError("Condition expression must evaluate to an xarray boolean mask")
    if result.dtype != bool:
        raise ValueError("Condition expression must evaluate to boolean data")

    mask = result.astype(bool)
    mask.name = mask_name
    mask.attrs = {
        "long_name": "Condition mask",
        "mask_type": "condition",
        "expression": expression,
        "source_variables": sorted(evaluator.source_variables),
    }
    return mask


class _ConditionMaskEvaluator(ast.NodeVisitor):
    _COMPARE_OPS = {
        ast.Lt: lambda left, right: left < right,
        ast.LtE: lambda left, right: left <= right,
        ast.Gt: lambda left, right: left > right,
        ast.GtE: lambda left, right: left >= right,
        ast.Eq: lambda left, right: left == right,
        ast.NotEq: lambda left, right: left != right,
    }
    _BIN_OPS = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.Pow: lambda left, right: left**right,
    }

    def __init__(self, fields: Mapping[str, xr.DataArray]) -> None:
        self.fields = dict(fields)
        self.source_variables: set[str] = set()

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> xr.DataArray:
        if node.id not in self.fields:
            raise ValueError(f"Unknown condition variable: {node.id}")
        self.source_variables.add(node.id)
        return self.fields[node.id]

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, bool, str)) or node.value is None:
            return node.value
        raise ValueError(f"Unsupported constant in condition expression: {node.value!r}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return ~self._as_bool_mask(value, "not")
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("Unsupported unary operator in condition expression")

    def visit_BoolOp(self, node: ast.BoolOp) -> xr.DataArray:
        if not node.values:
            raise ValueError("Boolean operation requires operands")
        values = [self._as_bool_mask(self.visit(value), "boolean operation") for value in node.values]
        result = values[0]
        for value in values[1:]:
            if isinstance(node.op, ast.And):
                result = result & value
            elif isinstance(node.op, ast.Or):
                result = result | value
            else:
                raise ValueError("Unsupported boolean operator in condition expression")
        return result

    def visit_Compare(self, node: ast.Compare) -> xr.DataArray:
        left = self.visit(node.left)
        comparisons = []
        for operator, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            op_type = type(operator)
            if op_type not in self._COMPARE_OPS:
                raise ValueError("Unsupported comparison operator in condition expression")
            comparisons.append(self._COMPARE_OPS[op_type](left, right))
            left = right
        result = self._as_bool_mask(comparisons[0], "comparison")
        for comparison in comparisons[1:]:
            result = result & self._as_bool_mask(comparison, "comparison")
        return result

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_type = type(node.op)
        if op_type not in self._BIN_OPS:
            raise ValueError("Unsupported arithmetic operator in condition expression")
        return self._BIN_OPS[op_type](self.visit(node.left), self.visit(node.right))

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple allow-listed function calls are allowed in condition expressions")
        name = node.func.id
        if name == "abs":
            self._require_arg_count(node, 1)
            return abs(self.visit(node.args[0]))
        if name == "isfinite":
            self._require_arg_count(node, 1)
            return xr.apply_ufunc(np.isfinite, self.visit(node.args[0]))
        if name == "mean":
            self._require_arg_count(node, 1)
            field = self.visit(node.args[0])
            return field.mean(dim=self._optional_dim_kwarg(node), skipna=True)
        if name == "percentile":
            if len(node.args) != 2:
                raise ValueError("percentile(field, q, dim=...) requires field and q")
            field = self.visit(node.args[0])
            q = float(self.visit(node.args[1]))
            if q > 1.0:
                q = q / 100.0
            if q < 0.0 or q > 1.0:
                raise ValueError("percentile q must be in [0, 1] or [0, 100]")
            return field.quantile(q, dim=self._optional_dim_kwarg(node), skipna=True)
        raise ValueError(f"Function is not allowed in condition expressions: {name}")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")

    def _as_bool_mask(self, value: Any, context: str) -> xr.DataArray:
        if not hasattr(value, "dims"):
            raise ValueError(f"{context} must produce an xarray boolean mask")
        if value.dtype != bool:
            raise ValueError(f"{context} must produce boolean data")
        return value

    @staticmethod
    def _require_arg_count(node: ast.Call, count: int) -> None:
        if len(node.args) != count:
            raise ValueError(f"{getattr(node.func, 'id', 'function')} expects {count} positional argument(s)")
        if any(keyword.arg is None for keyword in node.keywords):
            raise ValueError("Starred keyword arguments are not allowed in condition expressions")
        unexpected = [keyword.arg for keyword in node.keywords if keyword.arg != "dim"]
        if unexpected:
            raise ValueError(f"Unsupported keyword argument(s): {unexpected}")

    @staticmethod
    def _optional_dim_kwarg(node: ast.Call) -> str | Sequence[str] | None:
        dim_value = None
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("Starred keyword arguments are not allowed in condition expressions")
            if keyword.arg != "dim":
                raise ValueError(f"Unsupported keyword argument: {keyword.arg}")
            if not isinstance(keyword.value, (ast.Constant, ast.List, ast.Tuple)):
                raise ValueError("dim must be a string or list of strings")
            if isinstance(keyword.value, ast.Constant):
                if keyword.value.value is None:
                    dim_value = None
                elif isinstance(keyword.value.value, str):
                    dim_value = keyword.value.value
                else:
                    raise ValueError("dim must be a string or list of strings")
            else:
                dims = []
                for item in keyword.value.elts:
                    if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                        raise ValueError("dim list must contain only strings")
                    dims.append(item.value)
                dim_value = dims
        return dim_value


def compute_masked_area_fraction_timeseries(
    event_mask: xr.DataArray,
    analysis_mask: Optional[xr.DataArray] = None,
    min_valid_fraction: float = 0.0,
) -> Dict:
    """Reduce an event mask to a fractional coverage time series."""
    event = materialize_partitioned_xarray(event_mask).astype(float)
    analysis_mask = materialize_partitioned_xarray(analysis_mask)
    if "time" not in event.dims:
        raise ValueError("compute_masked_area_fraction_timeseries requires a time dimension")

    if analysis_mask is not None:
        event = event.where(analysis_mask)

    reduce_dims = [dim for dim in event.dims if dim != "time"]
    if not reduce_dims:
        raise ValueError("event_mask must include at least one non-time dimension")

    valid = xr.ones_like(event, dtype=float)
    valid = valid.where(np.isfinite(event))
    numerator = event.fillna(0.0).sum(dim=reduce_dims)
    denominator = valid.fillna(0.0).sum(dim=reduce_dims)
    fraction = xr.where(denominator > 0, numerator / denominator, np.nan)

    if min_valid_fraction > 0.0:
        max_count = float(denominator.max(skipna=True).values)
        if np.isfinite(max_count) and max_count > 0:
            fraction = fraction.where((denominator / max_count) >= float(min_valid_fraction))

    values = dataarray_to_numpy(
        fraction,
        label="masked area fraction time series",
        dtype=float,
        start=0.2,
        end=0.9,
    )
    return {
        "times": [str(value) for value in fraction.time.values],
        "values": values.tolist(),
        "metadata": {
            "variable": event_mask.name or "event_mask",
            "unit": "fraction",
            "units": "fraction",
            "reduction": "masked_area_fraction",
            "reduced_dims": reduce_dims,
            "statistics": _series_statistics(values),
        },
    }


def compute_masked_mean_timeseries(
    data: xr.DataArray,
    analysis_mask: Optional[xr.DataArray] = None,
    event_mask: Optional[xr.DataArray] = None,
) -> Dict:
    """Reduce a field to a masked mean time series."""
    field = materialize_partitioned_xarray(data)
    analysis_mask = materialize_partitioned_xarray(analysis_mask)
    event_mask = materialize_partitioned_xarray(event_mask)
    if "time" not in field.dims:
        raise ValueError("compute_masked_mean_timeseries requires a time dimension")

    if analysis_mask is not None:
        field = field.where(analysis_mask)
    if event_mask is not None:
        field = field.where(event_mask)
    reduce_dims = [dim for dim in field.dims if dim != "time"]
    if not reduce_dims:
        raise ValueError("data must include at least one non-time dimension")
    series = field.mean(dim=reduce_dims, skipna=True)
    values = dataarray_to_numpy(
        series,
        label="masked mean time series",
        dtype=float,
        start=0.2,
        end=0.9,
    )
    return {
        "times": [str(value) for value in series.time.values],
        "values": values.tolist(),
        "metadata": {
            "variable": field.name or "unknown",
            "unit": field.attrs.get("units", ""),
            "units": field.attrs.get("units", ""),
            "reduction": "masked_mean",
            "reduced_dims": reduce_dims,
            "statistics": _series_statistics(values),
        },
    }


def compute_speed_from_uv(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Compute current speed magnitude from u/v components."""
    u_field, v_field = xr.align(materialize_partitioned_xarray(u), materialize_partitioned_xarray(v), join="inner")
    speed = np.sqrt(u_field ** 2 + v_field ** 2)
    speed.name = "speed"
    speed.attrs = {
        "long_name": "Current speed magnitude",
        "units": u_field.attrs.get("units") or v_field.attrs.get("units") or "m s-1",
    }
    return speed


def _select_nearest_depth(data: xr.DataArray, depth_dim: str, target: float) -> xr.DataArray:
    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    index = int(np.nanargmin(np.abs(np.abs(depth_values) - abs(float(target)))))
    return data.isel({depth_dim: slice(index, index + 1)})


def _select_depth_range(
    data: xr.DataArray,
    depth_dim: str,
    depth_range: Tuple[float, float],
) -> xr.DataArray:
    normalized = _normalize_depth_range(np.asarray(data[depth_dim].values, dtype=float), depth_range)
    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    if depth_values[0] <= depth_values[-1]:
        indexer = slice(normalized[0], normalized[1])
    else:
        indexer = slice(normalized[1], normalized[0])
    selected = data.sel({depth_dim: indexer})
    if selected.sizes.get(depth_dim, 0) == 0:
        raise ValueError("Selected depth_range does not include any levels")
    return selected


def _select_bottom(data: xr.DataArray, depth_dim: str) -> xr.DataArray:
    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    valid_depth = xr.DataArray(
        np.isfinite(depth_values),
        coords={depth_dim: data[depth_dim]},
        dims=(depth_dim,),
    )
    depth_score = xr.DataArray(
        np.abs(depth_values),
        coords={depth_dim: data[depth_dim]},
        dims=(depth_dim,),
    )
    finite = xr.apply_ufunc(np.isfinite, data, dask="parallelized", output_dtypes=[bool]) & valid_depth
    score = depth_score.where(finite, -np.inf)
    bottom_depth = score.idxmax(dim=depth_dim)
    selected = data.sel({depth_dim: bottom_depth}).where(finite.any(dim=depth_dim))
    selected = selected.drop_vars(depth_dim, errors="ignore").expand_dims({depth_dim: [0.0]})
    return _transpose_like(selected, data)


def _select_bottom_band(data: xr.DataArray, depth_dim: str, thickness_m: float) -> xr.DataArray:
    report_phase(
        phase="selecting_bottom_band",
        message=f"Selecting bottom band thickness {thickness_m:g} m",
        percent=0.05,
    )
    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    valid_depth = xr.DataArray(
        np.isfinite(depth_values),
        coords={depth_dim: data[depth_dim]},
        dims=(depth_dim,),
    )
    depth_abs = xr.DataArray(
        np.abs(depth_values),
        coords={depth_dim: data[depth_dim]},
        dims=(depth_dim,),
    )
    finite = xr.apply_ufunc(np.isfinite, data, dask="parallelized", output_dtypes=[bool]) & valid_depth
    horizontal_valid = finite
    if "time" in horizontal_valid.dims:
        horizontal_valid = horizontal_valid.any(dim="time")
    bottom_depth = depth_abs.where(horizontal_valid, np.nan).max(dim=depth_dim, skipna=True)
    bottom_band_mask = (depth_abs >= (bottom_depth - float(thickness_m))) & (depth_abs <= bottom_depth)
    selected = data.where(bottom_band_mask)
    selected.attrs = dict(data.attrs)
    selected.attrs["bottom_depth_available"] = True
    selected.attrs["valid_bottom_band_mask"] = True
    return selected


def _maybe_aggregate_depth(
    data: xr.DataArray,
    depth_dim: str,
    aggregation: Optional[str],
    retain_depth: bool,
) -> xr.DataArray:
    if retain_depth or depth_dim not in data.dims:
        return data
    agg = str(aggregation or "").strip().lower()
    if agg in {"", "retain", "none"}:
        return data
    if agg == "mean":
        return data.mean(dim=depth_dim, skipna=True)
    if agg == "min":
        return data.min(dim=depth_dim, skipna=True)
    if agg == "max":
        return data.max(dim=depth_dim, skipna=True)
    if agg == "integral":
        return data.integrate(coord=depth_dim)
    raise ValueError(f"Unsupported depth aggregation: {aggregation}")


def _transpose_like(selected: xr.DataArray, template: xr.DataArray) -> xr.DataArray:
    ordered = [dim for dim in template.dims if dim in selected.dims]
    extras = [dim for dim in selected.dims if dim not in ordered]
    return selected.transpose(*(ordered + extras))


def _series_statistics(values: np.ndarray) -> Dict[str, float | int | bool]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "n_points": int(np.asarray(values).size),
            "finite_count": 0,
            "has_finite_values": False,
        }
    return {
        "n_points": int(np.asarray(values).size),
        "finite_count": int(finite.size),
        "has_finite_values": True,
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "median": float(np.median(finite)),
    }
