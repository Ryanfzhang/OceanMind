"""Partition-aware execution helpers for tool orchestration."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from domain.ocean.data_access.partitioned import (
    PartitionedDataArray,
    PartitionedDataset,
    PartitionedXarray,
    find_partitioned_values,
    partition_labels,
    replace_partitioned_at_index,
    wrap_partitioned,
)
from domain.ocean.result_payload import as_numeric_array
from packages.tool_loader.progress import report_tool_progress


_DEFAULT_PARTITION_MAX_WORKERS = 8


class PartitionedExecutionError(ValueError):
    """Raised when a tool cannot safely consume partitioned inputs."""


def execute_partition_aware(
    *,
    tool_name: str,
    tool_func: Callable[..., Any],
    params: Dict[str, Any],
) -> Any:
    partitioned_values = find_partitioned_values(params)
    if not partitioned_values:
        return tool_func(**params)

    _validate_compatible_partitions(partitioned_values, tool_name)

    if tool_name == "assemble_dataset":
        return _map_xarray_output(tool_func, params, partitioned_values)
    if tool_name == "compute_field_climatology":
        return _compute_partitioned_field_climatology(params)
    if tool_name == "compute_field_anomaly":
        return _compute_partitioned_field_anomaly(tool_func, params)
    if tool_name == "compute_spatial_field":
        return _compute_partitioned_spatial_field(params)
    if tool_name == "compute_field_trend":
        return _compute_partitioned_field_trend(params)
    if tool_name in {"replace_field_with_climatology", "remove_field_anomaly_component"}:
        return _compute_partitioned_climatology_replacement(tool_name, params)
    if tool_name == "compute_histogram":
        return _compute_partitioned_histogram(params)
    if tool_name == "compute_2d_histogram":
        return _compute_partitioned_2d_histogram(params)
    if tool_name == "compute_regression_map":
        return _compute_partitioned_regression_map(params)
    if tool_name == "compute_composite_field":
        return _compute_partitioned_composite_field(params)

    mapped_results = _map_results(tool_func, params, partitioned_values)
    if _all_xarray(mapped_results):
        return wrap_partitioned(mapped_results, labels=partition_labels(partitioned_values[0]))
    if _all_timeseries_results(mapped_results):
        return _merge_timeseries_results(mapped_results)
    if _all_spatial_field_results(mapped_results):
        return _combine_spatial_result_maps(mapped_results, params, partitioned_values)
    if _all_hovmoller_results(mapped_results):
        return _merge_hovmoller_results(mapped_results)
    if _all_event_results(mapped_results):
        return _merge_event_results(mapped_results)
    if _all_layer_transport_results(mapped_results):
        return _merge_layer_transport_results(mapped_results)

    raise PartitionedExecutionError(f"Tool {tool_name} is not partition-aware yet")


def _map_results(
    tool_func: Callable[..., Any],
    params: Dict[str, Any],
    partitioned_values: Sequence[PartitionedXarray],
) -> List[Any]:
    n_partitions = len(partitioned_values[0].partitions)
    workers = _partition_worker_count(n_partitions, partitioned_values)
    labels = partition_labels(partitioned_values[0])

    if workers <= 1:
        results = []
        for index in range(n_partitions):
            _report_partition_progress(index, labels[index], n_partitions, phase="partition_started")
            results.append(_run_partition_task(tool_func, params, index, labels[index]))
            _report_partition_progress(index + 1, labels[index], n_partitions, phase="partition_complete")
        return results

    results: List[Any] = [None] * n_partitions
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocean-partition") as executor:
        futures = [
            executor.submit(_run_partition_task, tool_func, params, index, labels[index])
            for index in range(n_partitions)
        ]
        for index, future in enumerate(futures):
            _report_partition_progress(index, labels[index], n_partitions, phase="partition_started")
            results[index] = future.result()
            _report_partition_progress(index + 1, labels[index], n_partitions, phase="partition_complete")
    return results


def _partition_worker_count(
    n_partitions: int,
    partitioned_values: Optional[Sequence[PartitionedXarray]] = None,
) -> int:
    if n_partitions <= 1:
        return 1
    if partitioned_values and _contains_dask_backed_partition(partitioned_values):
        return 1

    raw_value = os.environ.get("OCEAN_PARTITION_MAX_WORKERS", str(_DEFAULT_PARTITION_MAX_WORKERS))
    try:
        configured = int(raw_value)
    except (TypeError, ValueError):
        configured = _DEFAULT_PARTITION_MAX_WORKERS
    configured = max(1, configured)
    return min(configured, n_partitions)


def _contains_dask_backed_partition(partitioned_values: Sequence[PartitionedXarray]) -> bool:
    for value in partitioned_values:
        for partition in value.partitions:
            if isinstance(partition, xr.DataArray):
                if getattr(partition.data, "chunks", None) is not None:
                    return True
            elif isinstance(partition, xr.Dataset):
                for data_array in partition.data_vars.values():
                    if getattr(data_array.data, "chunks", None) is not None:
                        return True
    return False


def _run_partition_task(
    tool_func: Callable[..., Any],
    params: Dict[str, Any],
    index: int,
    label: str,
) -> Any:
    try:
        return tool_func(**replace_partitioned_at_index(params, index))
    except Exception as exc:
        raise PartitionedExecutionError(
            f"Partition {index} ({label}) failed: {exc}"
        ) from exc


def _report_partition_progress(completed: int, label: str, total: int, *, phase: str) -> None:
    report_tool_progress(
        phase=phase,
        message="Computing partitioned data",
        percent=(completed / total) if total else None,
        completed_units=completed,
        total_units=total,
        unit_label="partition",
        current_unit=label,
    )


def _map_xarray_output(
    tool_func: Callable[..., Any],
    params: Dict[str, Any],
    partitioned_values: Sequence[PartitionedXarray],
) -> PartitionedXarray:
    mapped_results = _map_results(tool_func, params, partitioned_values)
    if not _all_xarray(mapped_results):
        raise PartitionedExecutionError("Partitioned map expected xarray outputs")
    return wrap_partitioned(mapped_results, labels=partition_labels(partitioned_values[0]))


def _validate_compatible_partitions(
    partitioned_values: Sequence[PartitionedXarray],
    tool_name: str,
) -> None:
    expected_count = len(partitioned_values[0].partitions)
    expected_ranges = _partition_time_ranges(partitioned_values[0])
    for value in partitioned_values[1:]:
        if len(value.partitions) != expected_count:
            raise PartitionedExecutionError(
                f"Tool {tool_name} received partitioned inputs with different partition counts"
            )
        ranges = _partition_time_ranges(value)
        if expected_ranges and ranges and ranges != expected_ranges:
            raise PartitionedExecutionError(
                f"Tool {tool_name} received partitioned inputs with different time partitions"
            )


def _partition_time_ranges(value: PartitionedXarray) -> Tuple[Tuple[str, str], ...]:
    ranges = []
    for partition in value.partitions:
        if "time" not in getattr(partition, "coords", {}):
            return tuple()
        time_values = partition["time"].values
        if getattr(time_values, "size", 0) == 0:
            ranges.append(("", ""))
        else:
            ranges.append((str(time_values[0]), str(time_values[-1])))
    return tuple(ranges)


def _all_xarray(values: Sequence[Any]) -> bool:
    return bool(values) and all(isinstance(value, (xr.DataArray, xr.Dataset)) for value in values)


def _all_timeseries_results(values: Sequence[Any]) -> bool:
    return bool(values) and all(
        isinstance(value, dict) and {"times", "values"}.issubset(value.keys())
        for value in values
    )


def _all_spatial_field_results(values: Sequence[Any]) -> bool:
    return bool(values) and all(
        isinstance(value, dict) and {"lon", "lat", "values"}.issubset(value.keys())
        for value in values
    )


def _all_hovmoller_results(values: Sequence[Any]) -> bool:
    return bool(values) and all(
        isinstance(value, dict) and {"time", "spatial_coord", "values"}.issubset(value.keys())
        for value in values
    )


def _all_event_results(values: Sequence[Any]) -> bool:
    return bool(values) and all(
        isinstance(value, dict) and isinstance(value.get("events"), list)
        for value in values
    )


def _all_layer_transport_results(values: Sequence[Any]) -> bool:
    return bool(values) and all(
        isinstance(value, dict) and {"times", "layers"}.issubset(value.keys())
        for value in values
    )


def _merge_timeseries_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for result in results:
        for time_value, value in zip(result.get("times", []), result.get("values", [])):
            rows.append((pd.to_datetime(time_value), str(time_value), value))
    rows.sort(key=lambda item: item[0])

    deduped = []
    seen = set()
    for timestamp, original, value in rows:
        key = timestamp.to_datetime64()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((original, value))

    times = [item[0] for item in deduped]
    values = [item[1] for item in deduped]
    merged = {
        "times": times,
        "values": values,
        "metadata": dict(results[0].get("metadata", {})),
    }
    merged["metadata"]["statistics"] = _series_statistics(values)
    return merged


def _merge_layer_transport_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise PartitionedExecutionError("No layer transport results to merge")
    first = results[0]
    layer_count = len(first.get("layers", []))
    merged_times = []
    for result in results:
        merged_times.extend(result.get("times", []))
    merged_layers = []
    for layer_index in range(layer_count):
        first_layer = first["layers"][layer_index]
        values = []
        for result in results:
            layers = result.get("layers", [])
            if layer_index >= len(layers):
                raise PartitionedExecutionError("Layer transport partitions produced inconsistent layers")
            values.extend(layers[layer_index].get("values", []))
        layer = dict(first_layer)
        layer["values"] = values
        layer["statistics"] = _series_statistics(values)
        merged_layers.append(layer)
    return {
        "times": merged_times,
        "layers": merged_layers,
        "metadata": dict(first.get("metadata", {})),
    }


def _merge_hovmoller_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first = results[0]
    rows = []
    for result in results:
        values_array = as_numeric_array(result.get("values"))
        for time_index, time_value in enumerate(result.get("time", [])):
            values = values_array[time_index] if time_index < values_array.shape[0] else []
            rows.append((pd.to_datetime(time_value), str(time_value), values))
    rows.sort(key=lambda item: item[0])
    return {
        "time": [row[1] for row in rows],
        "spatial_coord": first.get("spatial_coord", []),
        "values": np.asarray([row[2] for row in rows], dtype=float),
        "metadata": dict(first.get("metadata", {})),
    }


def _merge_event_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(results[0])
    events = []
    for result in results:
        events.extend(result.get("events", []))
    merged["events"] = events
    statistics = dict(merged.get("statistics", {})) if isinstance(merged.get("statistics"), dict) else {}
    statistics["n_events"] = len(events)
    statistics["event_count"] = len(events)
    merged["statistics"] = statistics
    return merged


def _series_statistics(values: Sequence[Any]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"n_points": 0}
    return {
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array)),
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "median": float(np.nanmedian(array)),
        "n_points": int(array.size),
    }


def _compute_partitioned_spatial_field(params: Dict[str, Any]) -> Dict[str, Any]:
    from domain.ocean.analysis.spatial.analysis import _aggregate_depth, _compute_statistics

    data = params.get("data")
    if not isinstance(data, PartitionedDataArray):
        raise PartitionedExecutionError("compute_spatial_field requires partitioned data")

    time_aggregation = str(params.get("time_aggregation") or "mean")
    if time_aggregation not in {"mean", "max", "min", "std", "median"}:
        raise PartitionedExecutionError(
            f"compute_spatial_field with partitioned data does not support time_aggregation={time_aggregation}"
        )

    reduced_fields = []
    weighted_sum = None
    valid_count = None
    sum_squares = None
    median_fields = []
    for index, partition in enumerate(data.partitions):
        field = partition
        time_range = params.get("time_range")
        if time_range is not None and "time" in field.coords:
            field = field.sel(time=slice(*time_range))
        mask = replace_partitioned_at_index(params.get("mask"), index)
        if mask is not None:
            field = field.where(mask)
        field = _aggregate_depth(
            field,
            params.get("depth_range"),
            str(params.get("depth_aggregation") or "mean"),
        )
        if "time" not in field.dims:
            reduced_fields.append(field)
            continue
        if time_aggregation == "mean":
            current_sum = field.sum(dim="time", skipna=True)
            current_count = field.count(dim="time")
            if weighted_sum is None:
                weighted_sum = current_sum
                valid_count = current_count
            else:
                weighted_sum, current_sum = xr.align(weighted_sum, current_sum, join="inner")
                valid_count, current_count = xr.align(valid_count, current_count, join="inner")
                weighted_sum = weighted_sum + current_sum
                valid_count = valid_count + current_count
        elif time_aggregation == "std":
            current_sum = field.sum(dim="time", skipna=True)
            current_sum_squares = (field ** 2).sum(dim="time", skipna=True)
            current_count = field.count(dim="time")
            if weighted_sum is None:
                weighted_sum = current_sum
                sum_squares = current_sum_squares
                valid_count = current_count
            else:
                weighted_sum, current_sum = xr.align(weighted_sum, current_sum, join="inner")
                sum_squares, current_sum_squares = xr.align(sum_squares, current_sum_squares, join="inner")
                valid_count, current_count = xr.align(valid_count, current_count, join="inner")
                weighted_sum = weighted_sum + current_sum
                sum_squares = sum_squares + current_sum_squares
                valid_count = valid_count + current_count
        elif time_aggregation == "median":
            median_fields.append(field)
        elif time_aggregation == "max":
            reduced_fields.append(field.max(dim="time", skipna=True))
        elif time_aggregation == "min":
            reduced_fields.append(field.min(dim="time", skipna=True))

    if time_aggregation == "mean" and weighted_sum is not None and valid_count is not None:
        combined = xr.where(valid_count > 0, weighted_sum / valid_count, np.nan)
    elif time_aggregation == "std" and weighted_sum is not None and valid_count is not None and sum_squares is not None:
        mean = xr.where(valid_count > 0, weighted_sum / valid_count, np.nan)
        variance = xr.where(valid_count > 0, (sum_squares / valid_count) - (mean ** 2), np.nan)
        combined = xr.where(variance >= 0.0, np.sqrt(variance), np.nan)
    elif time_aggregation == "median":
        if not median_fields:
            raise PartitionedExecutionError("compute_spatial_field produced no partition fields")
        combined = xr.concat(median_fields, dim="time").median(dim="time", skipna=True)
    else:
        if not reduced_fields:
            raise PartitionedExecutionError("compute_spatial_field produced no partition fields")
        stacked = xr.concat(reduced_fields, dim="_partition")
        if time_aggregation == "max":
            combined = stacked.max(dim="_partition", skipna=True)
        elif time_aggregation == "min":
            combined = stacked.min(dim="_partition", skipna=True)
        else:
            combined = stacked.mean(dim="_partition", skipna=True)

    if "lat" not in combined.dims or "lon" not in combined.dims:
        raise ValueError("Result must retain lat and lon dimensions after aggregation")
    values = np.asarray(combined.values, dtype=float)
    return {
        "lon": combined.lon.values.tolist(),
        "lat": combined.lat.values.tolist(),
        "values": values,
        "metadata": {
            "variable": data.name or "unknown",
            "units": data.attrs.get("units", "") if isinstance(data.attrs, Mapping) else "",
            "time_range": list(params.get("time_range")) if params.get("time_range") is not None else None,
            "time_aggregation": time_aggregation,
            "depth_range": list(params.get("depth_range")) if params.get("depth_range") is not None else None,
            "depth_aggregation": params.get("depth_aggregation") if _partition_has_depth_dim(data) else None,
            "statistics": _compute_statistics(values),
        },
    }


def _combine_spatial_result_maps(
    results: Sequence[Dict[str, Any]],
    params: Dict[str, Any],
    partitioned_values: Sequence[PartitionedXarray],
) -> Dict[str, Any]:
    first = results[0]
    aggregation = str(params.get("time_aggregation") or "mean")
    arrays = [as_numeric_array(result.get("values")) for result in results]
    stack = np.stack(arrays, axis=0)
    if aggregation == "max":
        values = np.nanmax(stack, axis=0)
    elif aggregation == "min":
        values = np.nanmin(stack, axis=0)
    else:
        weights = _partition_time_weights(partitioned_values[0])
        if len(weights) == stack.shape[0] and np.sum(weights) > 0:
            weight_grid = weights[:, None, None]
            valid = np.isfinite(stack)
            weighted_sum = np.nansum(np.where(valid, stack * weight_grid, 0.0), axis=0)
            valid_weight_sum = np.sum(np.where(valid, weight_grid, 0.0), axis=0)
            values = np.full_like(weighted_sum, np.nan, dtype=float)
            np.divide(weighted_sum, valid_weight_sum, out=values, where=valid_weight_sum > 0.0)
        else:
            values = np.nanmean(stack, axis=0)
    merged = dict(first)
    merged["values"] = values
    metadata = dict(first.get("metadata", {})) if isinstance(first.get("metadata"), dict) else {}
    metadata["statistics"] = _map_statistics(values)
    merged["metadata"] = metadata
    return merged


def _partition_time_weights(value: PartitionedXarray) -> np.ndarray:
    weights = []
    for partition in value.partitions:
        weights.append(float(partition.sizes.get("time", 1)))
    return np.asarray(weights, dtype=float)


def _map_statistics(values: np.ndarray) -> Dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    valid = finite[np.isfinite(finite)]
    if valid.size == 0:
        return {"n_valid": 0, "n_total": int(finite.size)}
    return {
        "mean": float(np.nanmean(valid)),
        "std": float(np.nanstd(valid)),
        "min": float(np.nanmin(valid)),
        "max": float(np.nanmax(valid)),
        "median": float(np.nanmedian(valid)),
        "n_valid": int(valid.size),
        "n_total": int(finite.size),
    }


def _compute_partitioned_field_climatology(params: Dict[str, Any]) -> xr.DataArray:
    data = params.get("data")
    if not isinstance(data, PartitionedDataArray):
        raise PartitionedExecutionError("compute_field_climatology requires partitioned data")
    period = str(params.get("period") or "monthly")
    if period == "monthly":
        labels = list(range(1, 13))
        grouped_dim = "month"
        canonical_time = pd.date_range("2001-01-01", periods=12, freq="MS")
        label_getter = lambda field: field["time"].dt.month
    elif period == "seasonal":
        labels = [1, 2, 3, 4]
        grouped_dim = "quarter"
        canonical_time = pd.to_datetime(["2001-01-01", "2001-04-01", "2001-07-01", "2001-10-01"])
        label_getter = lambda field: field["time"].dt.quarter
    else:
        raise ValueError(f"Unsupported climatology period: {period}")

    sums: Dict[int, xr.DataArray] = {}
    counts: Dict[int, xr.DataArray] = {}
    for partition in data.partitions:
        if "time" not in partition.dims:
            raise ValueError("compute_field_climatology requires a time dimension")
        label_values = label_getter(partition)
        for label in labels:
            selected = partition.where(label_values == label, drop=True)
            if int(selected.sizes.get("time", 0)) == 0:
                continue
            current_sum = selected.sum(dim="time", skipna=True)
            current_count = selected.count(dim="time")
            if label in sums:
                sums[label], current_sum = xr.align(sums[label], current_sum, join="inner")
                counts[label], current_count = xr.align(counts[label], current_count, join="inner")
                sums[label] = sums[label] + current_sum
                counts[label] = counts[label] + current_count
            else:
                sums[label] = current_sum
                counts[label] = current_count

    present_labels = [label for label in labels if label in sums]
    if not present_labels:
        raise ValueError("No time samples available for field climatology")
    climatology_fields = [
        xr.where(counts[label] > 0, sums[label] / counts[label], np.nan)
        for label in present_labels
    ]
    climatology = xr.concat(
        climatology_fields,
        dim=xr.DataArray(present_labels, dims=(grouped_dim,), name=grouped_dim),
    )
    resolved_time = pd.to_datetime([canonical_time[labels.index(label)] for label in present_labels])
    climatology = climatology.rename({grouped_dim: "time"}).assign_coords(time=resolved_time)
    climatology.name = f"{data.name or 'field'}_{period}_climatology"
    climatology.attrs = {
        **dict(data.attrs),
        "aggregation": "field_climatology",
        "climatology_period": period,
        "climatology_labels": present_labels,
        "canonical_time_year": 2001,
    }
    return climatology


def _compute_partitioned_field_anomaly(
    tool_func: Callable[..., Any],
    params: Dict[str, Any],
) -> PartitionedDataArray:
    data = params.get("data")
    if not isinstance(data, PartitionedDataArray):
        raise PartitionedExecutionError("compute_field_anomaly requires partitioned data")
    climatology = params.get("climatology")
    if climatology is None:
        climatology = _compute_partitioned_field_climatology(
            {"data": data, "period": params.get("period") or "monthly"}
        )
    mapped = []
    for partition in data.partitions:
        part_params = dict(params)
        part_params["data"] = partition
        part_params["climatology"] = climatology
        mapped.append(tool_func(**part_params))
    return PartitionedDataArray(
        tuple(mapped),
        partition_labels=data.partition_labels,
        attrs=dict(getattr(mapped[0], "attrs", {})),
    )


def _compute_partitioned_climatology_replacement(
    tool_name: str,
    params: Dict[str, Any],
) -> PartitionedDataArray:
    from domain.ocean.interpretation.explain import _project_climatology_to_time

    data = params.get("data")
    if not isinstance(data, PartitionedDataArray):
        raise PartitionedExecutionError(f"{tool_name} requires partitioned data")
    period = str(params.get("period") or "monthly")
    climatology = _compute_partitioned_field_climatology({"data": data, "period": period})
    mapped = []
    for partition in data.partitions:
        projected = _project_climatology_to_time(
            climatology=climatology,
            source_time=partition["time"],
            period=period,
        )
        if tool_name == "replace_field_with_climatology":
            projected.name = f"{partition.name or 'field'}_climatology_replaced"
            aggregation = "replace_field_with_climatology"
        else:
            projected.name = f"{partition.name or 'field'}_without_anomaly"
            aggregation = "remove_field_anomaly_component"
        projected.attrs = {
            **dict(partition.attrs),
            "aggregation": aggregation,
            "climatology_period": period,
        }
        mapped.append(projected)
    return PartitionedDataArray(
        tuple(mapped),
        partition_labels=data.partition_labels,
        attrs=dict(getattr(mapped[0], "attrs", {})),
    )


def _compute_partitioned_field_trend(params: Dict[str, Any]) -> Dict[str, Any]:
    from scipy import stats
    from domain.ocean.analysis.field_tools import _combine_units, _time_axis_for_regression

    data = params.get("data")
    if not isinstance(data, PartitionedDataArray):
        raise PartitionedExecutionError("compute_field_trend requires partitioned data")
    method = str(params.get("method") or "linear")
    if method != "linear":
        raise ValueError("Only linear trends are supported in v1")

    first = data.partitions[0]
    if "time" not in first.dims:
        raise ValueError("compute_field_trend requires a time dimension")
    ordered_first = first.transpose(*[dim for dim in first.dims if dim != "time"], "time")
    non_time_dims = tuple(dim for dim in ordered_first.dims if dim != "time")
    non_time_shape = tuple(ordered_first.sizes[dim] for dim in non_time_dims)
    n_cells = int(np.prod(non_time_shape))

    n_valid = np.zeros(n_cells, dtype=float)
    sum_x = np.zeros(n_cells, dtype=float)
    sum_y = np.zeros(n_cells, dtype=float)
    sum_xx = np.zeros(n_cells, dtype=float)
    sum_yy = np.zeros(n_cells, dtype=float)
    sum_xy = np.zeros(n_cells, dtype=float)
    time_unit = "year"
    total_time_steps = 0
    time_start = None
    time_end = None

    for partition in data.partitions:
        if "time" not in partition.dims:
            raise ValueError("compute_field_trend requires a time dimension")
        ordered = partition.transpose(*non_time_dims, "time")
        values = np.asarray(ordered.values, dtype=float).reshape(n_cells, ordered.sizes["time"])
        x, time_unit = _time_axis_for_regression(partition["time"].values)
        valid = np.isfinite(values) & np.isfinite(x)[None, :]
        x_broadcast = np.broadcast_to(x, values.shape)
        y_valid = np.where(valid, values, 0.0)
        x_valid = np.where(valid, x_broadcast, 0.0)
        n_valid += valid.sum(axis=1)
        sum_x += x_valid.sum(axis=1)
        sum_y += y_valid.sum(axis=1)
        sum_xx += (x_valid * x_valid).sum(axis=1)
        sum_yy += (y_valid * y_valid).sum(axis=1)
        sum_xy += (x_valid * y_valid).sum(axis=1)
        total_time_steps += int(ordered.sizes["time"])
        if ordered.sizes["time"]:
            if time_start is None:
                time_start = str(partition["time"].values[0])
            time_end = str(partition["time"].values[-1])

    with np.errstate(invalid="ignore", divide="ignore"):
        x_mean = sum_x / n_valid
        y_mean = sum_y / n_valid
        ss_x = sum_xx - (sum_x * sum_x / n_valid)
        ss_y = sum_yy - (sum_y * sum_y / n_valid)
        ss_xy = sum_xy - (sum_x * sum_y / n_valid)

    slope_values = np.full(n_cells, np.nan, dtype=float)
    intercept_values = np.full(n_cells, np.nan, dtype=float)
    r_squared_values = np.full(n_cells, np.nan, dtype=float)
    p_value_values = np.full(n_cells, np.nan, dtype=float)

    good = (n_valid >= 3) & np.isfinite(ss_x) & (np.abs(ss_x) > 0.0)
    if np.any(good):
        slope_values[good] = ss_xy[good] / ss_x[good]
        intercept_values[good] = y_mean[good] - slope_values[good] * x_mean[good]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = ss_xy[good] / np.sqrt(ss_x[good] * ss_y[good])
        r = np.clip(r, -1.0, 1.0)
        r_squared_values[good] = r ** 2
        df = np.maximum(n_valid[good] - 2, 1)
        denom = np.maximum(1.0 - r ** 2, 1e-12)
        t_stat = r * np.sqrt(df / denom)
        p_value_values[good] = 2.0 * stats.t.sf(np.abs(t_stat), df)

    confidence_level = float(params.get("confidence_level") or 0.95)
    alpha = 1.0 - confidence_level
    significant_values = np.isfinite(p_value_values) & (p_value_values < alpha)
    coords = {dim: ordered_first.coords[dim] for dim in non_time_dims}

    slope = xr.DataArray(
        slope_values.reshape(non_time_shape),
        coords=coords,
        dims=non_time_dims,
        name=f"{data.name or 'field'}_trend_slope",
        attrs={
            "long_name": f"Linear trend of {data.name or 'field'}",
            "units": _combine_units(data.attrs.get("units", ""), time_unit),
            "aggregation": "field_trend",
            "time_unit": time_unit,
            "confidence_level": confidence_level,
        },
    )
    intercept = xr.DataArray(intercept_values.reshape(non_time_shape), coords=coords, dims=non_time_dims, name=f"{data.name or 'field'}_trend_intercept")
    r_squared = xr.DataArray(r_squared_values.reshape(non_time_shape), coords=coords, dims=non_time_dims, name=f"{data.name or 'field'}_trend_r_squared")
    p_value = xr.DataArray(p_value_values.reshape(non_time_shape), coords=coords, dims=non_time_dims, name=f"{data.name or 'field'}_trend_p_value")
    significance_mask = xr.DataArray(
        significant_values.reshape(non_time_shape),
        coords=coords,
        dims=non_time_dims,
        name=f"{data.name or 'field'}_trend_significance",
        attrs={
            "long_name": f"Significant trend mask of {data.name or 'field'}",
            "confidence_level": confidence_level,
        },
    )
    n_valid_field = xr.DataArray(n_valid.astype(int).reshape(non_time_shape), coords=coords, dims=non_time_dims, name=f"{data.name or 'field'}_trend_n_valid")
    return {
        "data": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "p_value": p_value,
        "significance_mask": significance_mask,
        "n_valid": n_valid_field,
        "metadata": {
            "variable": data.name or "unknown",
            "units": data.attrs.get("units", ""),
            "method": method,
            "confidence_level": confidence_level,
            "time_unit": time_unit,
            "n_time_steps": total_time_steps,
            "time_range": [time_start, time_end],
        },
    }


def _compute_partitioned_histogram(params: Dict[str, Any]) -> Dict[str, Any]:
    data = params.get("data")
    if not isinstance(data, PartitionedDataArray):
        raise PartitionedExecutionError("compute_histogram requires partitioned data")
    n_bins = int(params.get("n_bins") or 50)
    normalize = bool(params.get("normalize", True))
    bin_range = params.get("bin_range")
    mask = params.get("mask")

    value_parts = [
        _flatten_partition_values(partition, replace_partitioned_at_index(mask, index))
        for index, partition in enumerate(data.partitions)
    ]
    values = np.concatenate([part for part in value_parts if part.size]) if value_parts else np.asarray([])
    if values.size == 0:
        raise ValueError("No valid values available for histogram analysis")
    if bin_range is None:
        bin_range = (float(np.nanmin(values)), float(np.nanmax(values)))

    counts = np.zeros(n_bins, dtype=float)
    for part in value_parts:
        if part.size == 0:
            continue
        current, bin_edges = np.histogram(part, bins=n_bins, range=bin_range, density=False)
        counts += current
    bin_edges = np.linspace(float(bin_range[0]), float(bin_range[1]), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if normalize:
        widths = np.diff(bin_edges)
        total = np.sum(counts)
        density = np.divide(counts, total * widths, out=np.zeros_like(counts), where=(total * widths) > 0)
    else:
        density = counts

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "density": density.tolist(),
        "metadata": {
            "variable": data.name or "unknown",
            "units": data.attrs.get("units", "") if isinstance(data.attrs, Mapping) else "",
            "n_bins": n_bins,
            "normalized": normalize,
            "statistics": _histogram_1d_statistics(values, density, bin_centers),
        },
    }


def _compute_partitioned_2d_histogram(params: Dict[str, Any]) -> Dict[str, Any]:
    data_x = params.get("data_x")
    data_y = params.get("data_y")
    if not isinstance(data_x, PartitionedDataArray) or not isinstance(data_y, PartitionedDataArray):
        raise PartitionedExecutionError("compute_2d_histogram requires partitioned data_x and data_y")
    n_bins = int(params.get("n_bins") or 50)
    normalize = bool(params.get("normalize", True))
    mask = params.get("mask")

    pairs = []
    for index, (part_x, part_y) in enumerate(zip(data_x.partitions, data_y.partitions)):
        values_x, values_y = _flatten_partition_value_pair(
            part_x,
            part_y,
            replace_partitioned_at_index(mask, index),
        )
        if values_x.size:
            pairs.append((values_x, values_y))
    if not pairs:
        raise ValueError("No valid paired values available for 2D histogram analysis")

    all_x = np.concatenate([pair[0] for pair in pairs])
    all_y = np.concatenate([pair[1] for pair in pairs])
    range_x = params.get("range_x") or (float(np.nanmin(all_x)), float(np.nanmax(all_x)))
    range_y = params.get("range_y") or (float(np.nanmin(all_y)), float(np.nanmax(all_y)))

    counts = np.zeros((n_bins, n_bins), dtype=float)
    for values_x, values_y in pairs:
        current, x_edges, y_edges = np.histogram2d(
            values_x,
            values_y,
            bins=n_bins,
            range=[range_x, range_y],
            density=False,
        )
        counts += current
    x_edges = np.linspace(float(range_x[0]), float(range_x[1]), n_bins + 1)
    y_edges = np.linspace(float(range_y[0]), float(range_y[1]), n_bins + 1)
    if normalize:
        x_widths = np.diff(x_edges)
        y_widths = np.diff(y_edges)
        area = x_widths[:, None] * y_widths[None, :]
        total = np.sum(counts)
        density = np.divide(counts, total * area, out=np.zeros_like(counts), where=(total * area) > 0)
    else:
        density = counts

    return {
        "x_bin_edges": x_edges.tolist(),
        "y_bin_edges": y_edges.tolist(),
        "x_bin_centers": (0.5 * (x_edges[:-1] + x_edges[1:])).tolist(),
        "y_bin_centers": (0.5 * (y_edges[:-1] + y_edges[1:])).tolist(),
        "density": density.tolist(),
        "metadata": {
            "x_variable": data_x.name or "x",
            "y_variable": data_y.name or "y",
            "x_units": data_x.attrs.get("units", "") if isinstance(data_x.attrs, Mapping) else "",
            "y_units": data_y.attrs.get("units", "") if isinstance(data_y.attrs, Mapping) else "",
            "n_bins": n_bins,
            "normalized": normalize,
            "statistics": {
                "n_samples": int(all_x.size),
                "x_mean": float(np.mean(all_x)),
                "y_mean": float(np.mean(all_y)),
                "x_std": float(np.std(all_x)),
                "y_std": float(np.std(all_y)),
            },
        },
    }


def _compute_partitioned_regression_map(params: Dict[str, Any]) -> Dict[str, Any]:
    from scipy import stats

    field = params.get("field")
    if not isinstance(field, PartitionedDataArray):
        raise PartitionedExecutionError("compute_regression_map requires partitioned field")

    parts = _prepare_partitioned_field_index_parts(
        field=field,
        index_timeseries=params.get("index_timeseries"),
        lag=int(params.get("lag") or 0),
        remove_seasonal_cycle=bool(params.get("remove_seasonal_cycle", False)),
    )
    if not parts:
        raise ValueError("Field and index_timeseries do not share enough overlapping time steps")

    first_field = parts[0][0]
    lat = np.asarray(first_field.lat.values, dtype=float)
    lon = np.asarray(first_field.lon.values, dtype=float)
    non_time_shape = (lat.size, lon.size)
    n_cells = int(np.prod(non_time_shape))

    if bool(params.get("remove_seasonal_cycle", False)):
        month_means, index_month_means = _partitioned_monthly_means(parts, n_cells)
    else:
        month_means = {}
        index_month_means = {}

    n_valid = np.zeros(n_cells, dtype=float)
    sum_x = np.zeros(n_cells, dtype=float)
    sum_y = np.zeros(n_cells, dtype=float)
    sum_xx = np.zeros(n_cells, dtype=float)
    sum_yy = np.zeros(n_cells, dtype=float)
    sum_xy = np.zeros(n_cells, dtype=float)
    for part_field, index_values, times in parts:
        values = np.asarray(part_field.transpose("time", "lat", "lon").values, dtype=float).reshape(part_field.sizes["time"], n_cells).T
        x = np.asarray(index_values, dtype=float)
        if month_means:
            months = pd.to_datetime(times).month
            for row_index, month in enumerate(months):
                values[:, row_index] = values[:, row_index] - month_means[int(month)]
                x[row_index] = x[row_index] - index_month_means[int(month)]
        valid = np.isfinite(values) & np.isfinite(x)[None, :]
        x_broadcast = np.broadcast_to(x, values.shape)
        y_valid = np.where(valid, values, 0.0)
        x_valid = np.where(valid, x_broadcast, 0.0)
        n_valid += valid.sum(axis=1)
        sum_x += x_valid.sum(axis=1)
        sum_y += y_valid.sum(axis=1)
        sum_xx += (x_valid * x_valid).sum(axis=1)
        sum_yy += (y_valid * y_valid).sum(axis=1)
        sum_xy += (x_valid * y_valid).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        ss_x = sum_xx - (sum_x * sum_x / n_valid)
        ss_y = sum_yy - (sum_y * sum_y / n_valid)
        ss_xy = sum_xy - (sum_x * sum_y / n_valid)

    slope = np.full(n_cells, np.nan, dtype=float)
    correlation = np.full(n_cells, np.nan, dtype=float)
    p_value = np.full(n_cells, np.nan, dtype=float)
    good = (n_valid >= 3) & np.isfinite(ss_x) & (np.abs(ss_x) > 0.0)
    if np.any(good):
        slope[good] = ss_xy[good] / ss_x[good]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = ss_xy[good] / np.sqrt(ss_x[good] * ss_y[good])
        correlation[good] = np.clip(r, -1.0, 1.0)
        df = np.maximum(n_valid[good] - 2, 1)
        denom = np.maximum(1.0 - correlation[good] ** 2, 1e-12)
        t_stat = correlation[good] * np.sqrt(df / denom)
        p_value[good] = 2.0 * stats.t.sf(np.abs(t_stat), df)

    significant_mask = p_value <= float(params.get("significance_level") or 0.05)
    all_times = [time for _, _, times in parts for time in times]
    return {
        "lon": lon.tolist(),
        "lat": lat.tolist(),
        "slope": slope.reshape(non_time_shape),
        "correlation": correlation.reshape(non_time_shape),
        "p_value": p_value.reshape(non_time_shape),
        "significant_mask": significant_mask.reshape(non_time_shape),
        "metadata": {
            "variable": first_field.name or field.name or "unknown",
            "units": field.attrs.get("units", "") if isinstance(field.attrs, Mapping) else "",
            "lag": int(params.get("lag") or 0),
            "positive_lag_convention": "positive lag means the index leads the field",
            "remove_seasonal_cycle": bool(params.get("remove_seasonal_cycle", False)),
            "significance_level": float(params.get("significance_level") or 0.05),
            "n_time_samples": int(sum(part[0].sizes["time"] for part in parts)),
            "time_range": [str(all_times[0]), str(all_times[-1])] if all_times else None,
        },
    }


def _compute_partitioned_composite_field(params: Dict[str, Any]) -> Dict[str, Any]:
    from domain.ocean.analysis.statistics import _dataarray_to_spatial_payload

    field = params.get("field")
    if not isinstance(field, PartitionedDataArray):
        raise PartitionedExecutionError("compute_composite_field requires partitioned field")
    quantile = float(params.get("quantile") or 0.2)
    if quantile <= 0.0 or quantile >= 0.5:
        raise ValueError("quantile must be between 0 and 0.5")

    parts = _prepare_partitioned_field_index_parts(
        field=field,
        index_timeseries=params.get("index_timeseries"),
        lag=int(params.get("lag") or 0),
        remove_seasonal_cycle=False,
    )
    if not parts:
        raise ValueError("Field and index_timeseries do not share enough overlapping time steps")

    all_index = np.concatenate([np.asarray(index_values, dtype=float) for _, index_values, _ in parts])
    lower_threshold = float(np.nanquantile(all_index, quantile))
    upper_threshold = float(np.nanquantile(all_index, 1.0 - quantile))

    n_cells = int(np.prod((parts[0][0].sizes["lat"], parts[0][0].sizes["lon"])))
    if bool(params.get("anomaly", True)):
        field_mean = _global_field_mean(parts, n_cells)
    else:
        field_mean = np.zeros(n_cells, dtype=float)

    positive_sum = np.zeros(n_cells, dtype=float)
    positive_count = np.zeros(n_cells, dtype=float)
    negative_sum = np.zeros(n_cells, dtype=float)
    negative_count = np.zeros(n_cells, dtype=float)
    positive_samples = 0
    negative_samples = 0

    for part_field, index_values, _ in parts:
        values = np.asarray(part_field.transpose("time", "lat", "lon").values, dtype=float).reshape(part_field.sizes["time"], n_cells)
        values = values - field_mean[None, :]
        positive_mask = np.asarray(index_values, dtype=float) >= upper_threshold
        negative_mask = np.asarray(index_values, dtype=float) <= lower_threshold
        positive_samples += int(np.sum(positive_mask))
        negative_samples += int(np.sum(negative_mask))
        if np.any(positive_mask):
            selected = values[positive_mask, :]
            valid = np.isfinite(selected)
            positive_sum += np.where(valid, selected, 0.0).sum(axis=0)
            positive_count += valid.sum(axis=0)
        if np.any(negative_mask):
            selected = values[negative_mask, :]
            valid = np.isfinite(selected)
            negative_sum += np.where(valid, selected, 0.0).sum(axis=0)
            negative_count += valid.sum(axis=0)

    if positive_samples == 0 or negative_samples == 0:
        raise ValueError("Composite selection produced an empty sample; adjust quantile or lag")

    shape = (parts[0][0].sizes["lat"], parts[0][0].sizes["lon"])
    coords = {"lat": parts[0][0].lat, "lon": parts[0][0].lon}
    positive = xr.DataArray(
        np.divide(positive_sum, positive_count, out=np.full_like(positive_sum, np.nan), where=positive_count > 0).reshape(shape),
        coords=coords,
        dims=("lat", "lon"),
        name=parts[0][0].name,
        attrs=dict(parts[0][0].attrs),
    )
    negative = xr.DataArray(
        np.divide(negative_sum, negative_count, out=np.full_like(negative_sum, np.nan), where=negative_count > 0).reshape(shape),
        coords=coords,
        dims=("lat", "lon"),
        name=parts[0][0].name,
        attrs=dict(parts[0][0].attrs),
    )
    difference = positive - negative
    all_times = [time for _, _, times in parts for time in times]
    return {
        "positive_composite": _dataarray_to_spatial_payload(positive),
        "negative_composite": _dataarray_to_spatial_payload(negative),
        "difference": _dataarray_to_spatial_payload(difference),
        "sample_counts": {"positive": positive_samples, "negative": negative_samples},
        "metadata": {
            "variable": parts[0][0].name or field.name or "unknown",
            "units": field.attrs.get("units", "") if isinstance(field.attrs, Mapping) else "",
            "lag": int(params.get("lag") or 0),
            "positive_lag_convention": "positive lag means the index leads the field",
            "quantile": quantile,
            "anomaly": bool(params.get("anomaly", True)),
            "upper_threshold": upper_threshold,
            "lower_threshold": lower_threshold,
            "time_range": [str(all_times[0]), str(all_times[-1])] if all_times else None,
        },
    }


def _flatten_partition_values(data: xr.DataArray, mask: Optional[xr.DataArray] = None) -> np.ndarray:
    field = data.where(mask) if mask is not None else data
    values = np.asarray(field.values, dtype=float).ravel()
    return values[np.isfinite(values)]


def _flatten_partition_value_pair(
    data_x: xr.DataArray,
    data_y: xr.DataArray,
    mask: Optional[xr.DataArray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    x_field, y_field = xr.align(data_x, data_y, join="inner")
    if mask is not None:
        x_field, y_field, mask = xr.align(x_field, y_field, mask, join="inner")
        x_field = x_field.where(mask)
        y_field = y_field.where(mask)
    x_values = np.asarray(x_field.values, dtype=float).ravel()
    y_values = np.asarray(y_field.values, dtype=float).ravel()
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[valid], y_values[valid]


def _histogram_1d_statistics(values: np.ndarray, counts: np.ndarray, bin_centers: np.ndarray) -> Dict[str, Any]:
    peak_index = int(np.nanargmax(counts))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "peak_value": float(bin_centers[peak_index]),
        "n_samples": int(values.size),
    }


def _prepare_partitioned_field_index_parts(
    *,
    field: PartitionedDataArray,
    index_timeseries: Dict[str, Any],
    lag: int,
    remove_seasonal_cycle: bool,
) -> List[Tuple[xr.DataArray, np.ndarray, pd.DatetimeIndex]]:
    if not isinstance(index_timeseries, dict):
        raise ValueError("index_timeseries is required")
    index_times = pd.to_datetime(index_timeseries["times"])
    index_values = np.asarray(index_timeseries["values"], dtype=float)
    index_series = pd.Series(index_values, index=index_times)

    common_times = []
    for partition in field.partitions:
        prepared = _select_surface_partition(partition)
        if "time" not in prepared.dims or "lat" not in prepared.dims or "lon" not in prepared.dims:
            raise ValueError("Statistical map tools require time/lat/lon dimensions")
        part_times = pd.to_datetime(prepared.time.values)
        common_times.extend([time for time in part_times if time in index_series.index])
    common_times = pd.DatetimeIndex(common_times).unique().sort_values()
    if len(common_times) < 3:
        raise ValueError("Field and index_timeseries do not share enough overlapping time steps")

    common_index_values = index_series.loc[common_times].astype(float).to_numpy()
    time_position = {time: position for position, time in enumerate(common_times)}
    parts = []
    for partition in field.partitions:
        prepared = _select_surface_partition(partition)
        part_times = pd.to_datetime(prepared.time.values)
        selected_times = [time for time in part_times if time in time_position]
        if not selected_times:
            continue
        positions = np.asarray([time_position[time] for time in selected_times], dtype=int)
        paired_positions = positions - int(lag)
        valid = (paired_positions >= 0) & (paired_positions < len(common_times))
        if not np.any(valid):
            continue
        selected_times = pd.DatetimeIndex(np.asarray(selected_times, dtype="datetime64[ns]")[valid])
        paired_positions = paired_positions[valid]
        selected_field = prepared.sel(time=selected_times)
        selected_index = common_index_values[paired_positions]
        if selected_field.sizes.get("time", 0) == 0:
            continue
        parts.append((selected_field, selected_index.astype(float), pd.to_datetime(selected_field.time.values)))

    if remove_seasonal_cycle and any(not np.issubdtype(part[2].dtype, np.datetime64) for part in parts):
        raise ValueError("remove_seasonal_cycle requires datetime-like time coordinates")
    return parts


def _select_surface_partition(field: xr.DataArray) -> xr.DataArray:
    if "depth" in field.dims:
        return field.isel(depth=0)
    if "z" in field.dims:
        return field.isel(z=0)
    return field


def _partitioned_monthly_means(
    parts: Sequence[Tuple[xr.DataArray, np.ndarray, pd.DatetimeIndex]],
    n_cells: int,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    sums = {month: np.zeros(n_cells, dtype=float) for month in range(1, 13)}
    counts = {month: np.zeros(n_cells, dtype=float) for month in range(1, 13)}
    index_sums = {month: 0.0 for month in range(1, 13)}
    index_counts = {month: 0 for month in range(1, 13)}
    for part_field, index_values, times in parts:
        values = np.asarray(part_field.transpose("time", "lat", "lon").values, dtype=float).reshape(part_field.sizes["time"], n_cells)
        months = pd.to_datetime(times).month
        for row_index, month in enumerate(months):
            row = values[row_index]
            valid = np.isfinite(row)
            sums[int(month)] += np.where(valid, row, 0.0)
            counts[int(month)] += valid
            if np.isfinite(index_values[row_index]):
                index_sums[int(month)] += float(index_values[row_index])
                index_counts[int(month)] += 1
    field_means = {
        month: np.divide(sums[month], counts[month], out=np.zeros_like(sums[month]), where=counts[month] > 0)
        for month in range(1, 13)
    }
    index_means = {
        month: (index_sums[month] / index_counts[month] if index_counts[month] else 0.0)
        for month in range(1, 13)
    }
    return field_means, index_means


def _global_field_mean(
    parts: Sequence[Tuple[xr.DataArray, np.ndarray, pd.DatetimeIndex]],
    n_cells: int,
) -> np.ndarray:
    sums = np.zeros(n_cells, dtype=float)
    counts = np.zeros(n_cells, dtype=float)
    for part_field, _, _ in parts:
        values = np.asarray(part_field.transpose("time", "lat", "lon").values, dtype=float).reshape(part_field.sizes["time"], n_cells)
        valid = np.isfinite(values)
        sums += np.where(valid, values, 0.0).sum(axis=0)
        counts += valid.sum(axis=0)
    return np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)


def _partition_has_depth_dim(data: PartitionedDataArray) -> bool:
    return any("depth" in partition.dims or "z" in partition.dims for partition in data.partitions)
