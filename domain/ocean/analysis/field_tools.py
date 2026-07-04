"""
Field-based temporal diagnostics for gridded ocean model output.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr

from domain.ocean.data_access.partitioned import (
    PartitionedDataArray,
    materialize_partitioned_xarray,
)
from domain.ocean.dask_utils import dataarray_to_numpy, is_dask_backed, report_phase


def compute_field_climatology(
    data: xr.DataArray,
    period: Literal["monthly", "seasonal"] = "monthly",
) -> xr.DataArray:
    """
    Compute a gridded climatology while preserving spatial and depth dimensions.

    The climatology is returned on a canonical `time` axis so downstream tools can
    continue to treat the result like a time-varying field.
    """
    if isinstance(data, PartitionedDataArray):
        from packages.tool_loader.partitioned_execution import _compute_partitioned_field_climatology

        return _compute_partitioned_field_climatology({"data": data, "period": period})
    data = materialize_partitioned_xarray(data)

    if "time" not in data.dims:
        raise ValueError("compute_field_climatology requires a time dimension")

    grouped, labels, canonical_time = _group_field_by_period(data, period)
    climatology = grouped.mean(dim="time", skipna=True)
    grouped_dim = "month" if period == "monthly" else "quarter"
    present_labels = [int(label) for label in np.asarray(climatology[grouped_dim].values).tolist()]
    canonical_lookup = {
        int(label): timestamp
        for label, timestamp in zip(labels, canonical_time)
    }
    resolved_time = pd.to_datetime([canonical_lookup[label] for label in present_labels])
    climatology = climatology.rename({grouped_dim: "time"}).assign_coords(time=resolved_time)
    climatology.name = f"{data.name or 'field'}_{period}_climatology"
    climatology.attrs = {
        **data.attrs,
        "aggregation": "field_climatology",
        "climatology_period": period,
        "climatology_labels": present_labels,
        "canonical_time_year": 2001,
    }
    return climatology


def compute_field_anomaly(
    data: xr.DataArray,
    climatology: Optional[xr.DataArray] = None,
    period: Literal["monthly", "seasonal"] = "monthly",
) -> xr.DataArray:
    """
    Remove the monthly or seasonal climatology from a gridded field.
    """
    if isinstance(data, PartitionedDataArray):
        from packages.tool_loader.partitioned_execution import _compute_partitioned_field_anomaly

        return _compute_partitioned_field_anomaly(
            compute_field_anomaly,
            {"data": data, "climatology": climatology, "period": period},
        )
    data = materialize_partitioned_xarray(data)
    climatology = materialize_partitioned_xarray(climatology)

    if "time" not in data.dims:
        raise ValueError("compute_field_anomaly requires a time dimension")

    if climatology is None:
        climatology = compute_field_climatology(data, period=period)

    resolved_period = str(climatology.attrs.get("climatology_period", period))
    if resolved_period not in {"monthly", "seasonal"}:
        raise ValueError(f"Unsupported climatology period: {resolved_period}")

    if resolved_period == "monthly":
        month_values = _canonical_time_to_labels(climatology["time"].values, resolved_period)
        climatology_index = (
            climatology.assign_coords(month=("time", month_values))
            .swap_dims({"time": "month"})
            .drop_vars("time")
        )
        anomaly = data.groupby("time.month") - climatology_index
    else:
        quarter_values = _canonical_time_to_labels(climatology["time"].values, resolved_period)
        climatology_index = (
            climatology.assign_coords(quarter=("time", quarter_values))
            .swap_dims({"time": "quarter"})
            .drop_vars("time")
        )
        anomaly = data.groupby("time.quarter") - climatology_index

    anomaly.name = f"{data.name or 'field'}_{resolved_period}_anomaly"
    anomaly.attrs = {
        **data.attrs,
        "aggregation": "field_anomaly",
        "is_anomaly": True,
        "climatology_period": resolved_period,
    }
    return anomaly


def compute_field_trend(
    data: xr.DataArray,
    method: Literal["linear"] = "linear",
    confidence_level: float = 0.95,
) -> Dict:
    """
    Fit an independent linear trend at every non-time grid point.

    Returns a dict so the slope field can be used as the primary downstream
    product while p-values and significance masks remain accessible.
    """
    if isinstance(data, PartitionedDataArray):
        from packages.tool_loader.partitioned_execution import _compute_partitioned_field_trend

        return _compute_partitioned_field_trend(
            {"data": data, "method": method, "confidence_level": confidence_level}
        )
    data = materialize_partitioned_xarray(data)

    if "time" not in data.dims:
        raise ValueError("compute_field_trend requires a time dimension")
    if method != "linear":
        raise ValueError("Only linear trends are supported in v1")

    alpha = 1.0 - float(confidence_level)
    ordered = data.transpose(*[dim for dim in data.dims if dim != "time"], "time")
    non_time_dims = tuple(dim for dim in ordered.dims if dim != "time")
    non_time_shape = tuple(ordered.sizes[dim] for dim in non_time_dims)
    report_phase(
        phase="preparing_field_trend",
        message="Preparing field trend matrix",
        percent=0.05,
        compute_backend="dask" if is_dask_backed(ordered) else "xarray",
    )
    values = dataarray_to_numpy(
        ordered,
        label="field trend matrix",
        dtype=float,
        start=0.1,
        end=0.75,
    ).reshape(-1, ordered.sizes["time"])

    x, time_unit = _time_axis_for_regression(data["time"].values)
    report_phase(
        phase="solving_field_trend",
        message="Solving field trend",
        percent=0.82,
        compute_backend="numpy",
    )
    (
        slope_values,
        intercept_values,
        r_squared_values,
        p_value_values,
        significant_values,
        n_valid_values,
    ) = _vectorized_linear_trend(values, x, alpha)

    report_phase(
        phase="assembling_field_trend",
        message="Assembling field trend result",
        percent=0.94,
        compute_backend="numpy",
    )
    coords = {dim: ordered.coords[dim] for dim in non_time_dims}
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
            "confidence_level": float(confidence_level),
        },
    )
    intercept = xr.DataArray(
        intercept_values.reshape(non_time_shape),
        coords=coords,
        dims=non_time_dims,
        name=f"{data.name or 'field'}_trend_intercept",
    )
    r_squared = xr.DataArray(
        r_squared_values.reshape(non_time_shape),
        coords=coords,
        dims=non_time_dims,
        name=f"{data.name or 'field'}_trend_r_squared",
    )
    p_value = xr.DataArray(
        p_value_values.reshape(non_time_shape),
        coords=coords,
        dims=non_time_dims,
        name=f"{data.name or 'field'}_trend_p_value",
    )
    significance_mask = xr.DataArray(
        significant_values.reshape(non_time_shape),
        coords=coords,
        dims=non_time_dims,
        name=f"{data.name or 'field'}_trend_significance",
        attrs={
            "long_name": f"Significant trend mask of {data.name or 'field'}",
            "confidence_level": float(confidence_level),
        },
    )
    n_valid = xr.DataArray(
        n_valid_values.reshape(non_time_shape),
        coords=coords,
        dims=non_time_dims,
        name=f"{data.name or 'field'}_trend_n_valid",
    )

    report_phase(
        phase="field_trend_complete",
        message="Field trend complete",
        percent=1.0,
        compute_backend="numpy",
    )

    return {
        "data": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "p_value": p_value,
        "significance_mask": significance_mask,
        "n_valid": n_valid,
        "metadata": {
            "variable": data.name or "unknown",
            "units": data.attrs.get("units", ""),
            "method": method,
            "confidence_level": float(confidence_level),
            "time_unit": time_unit,
            "n_time_steps": int(data.sizes["time"]),
            "time_range": [str(data["time"].values[0]), str(data["time"].values[-1])],
        },
    }


def compute_local_tendency(data: xr.DataArray) -> xr.DataArray:
    """
    Compute the local time tendency d(field)/dt along the time axis.
    """
    data = materialize_partitioned_xarray(data)

    if "time" not in data.dims:
        raise ValueError("compute_local_tendency requires a time dimension")

    time_axis, time_unit = _time_axis_for_gradient(data["time"].values)
    tendency = xr.apply_ufunc(
        _gradient_1d,
        data,
        xr.DataArray(time_axis, coords={"time": data["time"]}, dims=("time",)),
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    ).transpose(*data.dims)
    tendency.name = f"{data.name or 'field'}_local_tendency"
    tendency.attrs = {
        **data.attrs,
        "long_name": f"Local tendency of {data.name or 'field'}",
        "units": _combine_units(data.attrs.get("units", ""), time_unit),
        "aggregation": "local_tendency",
        "time_unit": time_unit,
    }
    return tendency


def _group_field_by_period(
    data: xr.DataArray,
    period: str,
) -> tuple[xr.core.groupby.DataArrayGroupBy, list, pd.DatetimeIndex]:
    if period == "monthly":
        labels = list(range(1, 13))
        canonical_time = pd.date_range("2001-01-01", periods=12, freq="MS")
        return data.groupby("time.month"), labels, canonical_time
    if period == "seasonal":
        labels = [1, 2, 3, 4]
        canonical_time = pd.to_datetime(
            ["2001-01-01", "2001-04-01", "2001-07-01", "2001-10-01"]
        )
        return data.groupby("time.quarter"), labels, canonical_time
    raise ValueError(f"Unsupported climatology period: {period}")


def _canonical_time_to_labels(time_values, period: str) -> np.ndarray:
    canonical_time = pd.to_datetime(time_values)
    if period == "monthly":
        return canonical_time.month.to_numpy()
    if period == "seasonal":
        return canonical_time.quarter.to_numpy()
    raise ValueError(f"Unsupported climatology period: {period}")


def _time_axis_for_regression(time_values) -> tuple[np.ndarray, str]:
    values = np.asarray(time_values)
    if np.issubdtype(values.dtype, np.datetime64):
        times = pd.to_datetime(values)
        decimal_years = times.year + (times.dayofyear - 1) / 365.25
        return decimal_years.astype(float), "year"

    try:
        numeric = values.astype(float)
        return numeric, "time_unit"
    except (TypeError, ValueError):
        return np.arange(values.size, dtype=float), "step"


def _time_axis_for_gradient(time_values) -> tuple[np.ndarray, str]:
    values = np.asarray(time_values)
    if np.issubdtype(values.dtype, np.datetime64):
        times = pd.to_datetime(values)
        axis = (times - times[0]).total_seconds() / 86400.0
        return axis.astype(float), "day"

    try:
        numeric = values.astype(float)
        return numeric, "time_unit"
    except (TypeError, ValueError):
        return np.arange(values.size, dtype=float), "step"


def _gradient_1d(values: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(time_axis)
    if np.sum(valid) < 2:
        return np.full(values.shape, np.nan, dtype=float)

    valid_values = values[valid].astype(float)
    valid_axis = time_axis[valid].astype(float)
    if np.unique(valid_axis).size < 2:
        return np.full(values.shape, np.nan, dtype=float)

    gradient = np.gradient(valid_values, valid_axis)
    result = np.full(values.shape, np.nan, dtype=float)
    result[valid] = gradient
    return result


def _vectorized_linear_trend(values: np.ndarray, x: np.ndarray, alpha: float):
    from scipy import stats

    valid = np.isfinite(values)
    n_valid = valid.sum(axis=1)
    x_broadcast = np.broadcast_to(x, values.shape)
    x_valid = np.where(valid, x_broadcast, 0.0)
    y_valid = np.where(valid, values, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        x_mean = x_valid.sum(axis=1) / n_valid
        y_mean = y_valid.sum(axis=1) / n_valid

    x_dev = np.where(valid, x_broadcast - x_mean[:, None], 0.0)
    y_dev = np.where(valid, values - y_mean[:, None], 0.0)
    ss_x = np.sum(x_dev ** 2, axis=1)
    ss_y = np.sum(y_dev ** 2, axis=1)
    ss_xy = np.sum(x_dev * y_dev, axis=1)

    slope = np.full(values.shape[0], np.nan, dtype=float)
    intercept = np.full(values.shape[0], np.nan, dtype=float)
    r_squared = np.full(values.shape[0], np.nan, dtype=float)
    p_value = np.full(values.shape[0], np.nan, dtype=float)

    good = (n_valid >= 3) & np.isfinite(ss_x) & (np.abs(ss_x) > 0.0)
    if np.any(good):
        slope[good] = ss_xy[good] / ss_x[good]
        intercept[good] = y_mean[good] - slope[good] * x_mean[good]

        with np.errstate(invalid="ignore", divide="ignore"):
            r = ss_xy[good] / np.sqrt(ss_x[good] * ss_y[good])
        r = np.clip(r, -1.0, 1.0)
        r_squared[good] = r ** 2

        df = np.maximum(n_valid[good] - 2, 1)
        denom = np.maximum(1.0 - r ** 2, 1e-12)
        t_stat = r * np.sqrt(df / denom)
        p_value[good] = 2.0 * stats.t.sf(np.abs(t_stat), df)

    significance_mask = np.isfinite(p_value) & (p_value < alpha)
    return slope, intercept, r_squared, p_value, significance_mask, n_valid.astype(int)


def _combine_units(base_units: str, time_unit: str) -> str:
    if not base_units:
        return f"per_{time_unit}"
    return f"{base_units} / {time_unit}"
