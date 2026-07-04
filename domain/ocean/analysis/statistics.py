"""
Statistical diagnostics for regression, composites, and spectra.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.signal import find_peaks, welch

from domain.ocean.data_access.partitioned import find_partitioned_values, materialize_partitioned_xarray
from domain.ocean.dask_utils import (
    compute_together_with_progress,
    dataarray_to_numpy,
    is_dask_backed,
    report_phase,
)


def compute_regression_map(
    field: xr.DataArray,
    index_timeseries: Dict,
    lag: int = 0,
    remove_seasonal_cycle: bool = False,
    significance_level: float = 0.05,
) -> Dict:
    """
    Regress a time-varying field onto an internally generated index time series.
    """
    if find_partitioned_values((field,)):
        from packages.tool_loader.partitioned_execution import _compute_partitioned_regression_map

        return _compute_partitioned_regression_map(
            {
                "field": field,
                "index_timeseries": index_timeseries,
                "lag": lag,
                "remove_seasonal_cycle": remove_seasonal_cycle,
                "significance_level": significance_level,
            }
        )
    field = materialize_partitioned_xarray(field)

    prepared_field, index_values, times = _prepare_field_and_index(
        field=field,
        index_timeseries=index_timeseries,
        lag=lag,
        remove_seasonal_cycle=remove_seasonal_cycle,
    )
    lat = np.asarray(prepared_field.lat.values, dtype=float)
    lon = np.asarray(prepared_field.lon.values, dtype=float)
    report_phase(
        phase="preparing_regression_map",
        message="Preparing regression map input field",
        percent=0.05,
        compute_backend="dask" if is_dask_backed(prepared_field) else "xarray",
    )
    values = dataarray_to_numpy(
        prepared_field.transpose("time", "lat", "lon"),
        label="regression map input field",
        dtype=float,
        start=0.1,
        end=0.75,
    )

    slope = np.full((lat.size, lon.size), np.nan, dtype=float)
    correlation = np.full((lat.size, lon.size), np.nan, dtype=float)
    p_value = np.full((lat.size, lon.size), np.nan, dtype=float)

    report_phase(
        phase="solving_regression_map",
        message="Solving regression map",
        percent=0.82,
        compute_backend="numpy",
    )
    for lat_index in range(lat.size):
        for lon_index in range(lon.size):
            series = values[:, lat_index, lon_index]
            valid = np.isfinite(series) & np.isfinite(index_values)
            if np.sum(valid) < 3:
                continue
            fit = stats.linregress(index_values[valid], series[valid])
            slope[lat_index, lon_index] = float(fit.slope)
            correlation[lat_index, lon_index] = float(fit.rvalue)
            p_value[lat_index, lon_index] = float(fit.pvalue)

    significant_mask = p_value <= float(significance_level)
    report_phase(
        phase="regression_map_complete",
        message="Regression map complete",
        percent=1.0,
        compute_backend="numpy",
    )
    return {
        "lon": lon.tolist(),
        "lat": lat.tolist(),
        "slope": slope,
        "correlation": correlation,
        "p_value": p_value,
        "significant_mask": significant_mask,
        "metadata": {
            "variable": prepared_field.name or field.name or "unknown",
            "units": field.attrs.get("units", ""),
            "lag": int(lag),
            "positive_lag_convention": "positive lag means the index leads the field",
            "remove_seasonal_cycle": bool(remove_seasonal_cycle),
            "significance_level": float(significance_level),
            "n_time_samples": int(len(times)),
            "time_range": [str(times[0]), str(times[-1])] if len(times) else None,
        },
    }


def compute_composite_field(
    field: xr.DataArray,
    index_timeseries: Dict,
    quantile: float = 0.2,
    lag: int = 0,
    anomaly: bool = True,
) -> Dict:
    """
    Compute positive/negative composites and their difference from an internal index.
    """
    if find_partitioned_values((field,)):
        from packages.tool_loader.partitioned_execution import _compute_partitioned_composite_field

        return _compute_partitioned_composite_field(
            {
                "field": field,
                "index_timeseries": index_timeseries,
                "quantile": quantile,
                "lag": lag,
                "anomaly": anomaly,
            }
        )
    field = materialize_partitioned_xarray(field)

    if quantile <= 0.0 or quantile >= 0.5:
        raise ValueError("quantile must be between 0 and 0.5")

    prepared_field, index_values, times = _prepare_field_and_index(
        field=field,
        index_timeseries=index_timeseries,
        lag=lag,
        remove_seasonal_cycle=False,
    )
    if anomaly:
        prepared_field = prepared_field - prepared_field.mean(dim="time", skipna=True)

    lower_threshold = float(np.nanquantile(index_values, quantile))
    upper_threshold = float(np.nanquantile(index_values, 1.0 - quantile))
    positive_mask = index_values >= upper_threshold
    negative_mask = index_values <= lower_threshold

    if int(np.sum(positive_mask)) == 0 or int(np.sum(negative_mask)) == 0:
        raise ValueError("Composite selection produced an empty sample; adjust quantile or lag")

    positive = prepared_field.isel(time=np.where(positive_mask)[0]).mean(dim="time", skipna=True)
    negative = prepared_field.isel(time=np.where(negative_mask)[0]).mean(dim="time", skipna=True)
    difference = positive - negative
    report_phase(
        phase="preparing_composite_fields",
        message="Preparing composite fields",
        percent=0.1,
        compute_backend="dask" if is_dask_backed((positive, negative, difference)) else "xarray",
    )
    positive, negative, difference = compute_together_with_progress(
        [positive, negative, difference],
        label="composite fields",
        start=0.2,
        end=0.85,
    )
    report_phase(
        phase="assembling_composite_fields",
        message="Assembling composite field payload",
        percent=0.92,
        compute_backend="numpy",
    )

    return {
        "positive_composite": _dataarray_to_spatial_payload(positive, label="positive composite field"),
        "negative_composite": _dataarray_to_spatial_payload(negative, label="negative composite field"),
        "difference": _dataarray_to_spatial_payload(difference, label="composite difference field"),
        "sample_counts": {
            "positive": int(np.sum(positive_mask)),
            "negative": int(np.sum(negative_mask)),
        },
        "metadata": {
            "variable": prepared_field.name or field.name or "unknown",
            "units": field.attrs.get("units", ""),
            "lag": int(lag),
            "positive_lag_convention": "positive lag means the index leads the field",
            "quantile": float(quantile),
            "anomaly": bool(anomaly),
            "upper_threshold": upper_threshold,
            "lower_threshold": lower_threshold,
            "time_range": [str(times[0]), str(times[-1])] if len(times) else None,
        },
    }


def compute_spectrum(
    timeseries: Dict,
    method: Literal["welch"] = "welch",
    detrend: Literal["linear", "constant"] = "linear",
    window: str = "hann",
) -> Dict:
    """
    Compute a one-dimensional power spectrum from a time-series result.
    """
    if method != "welch":
        raise ValueError("Only Welch spectra are supported in v1")

    times = pd.to_datetime(timeseries["times"])
    values = np.asarray(timeseries["values"], dtype=float)
    valid = np.isfinite(values)
    times = times[valid]
    values = values[valid]

    if values.size < 4:
        metadata = timeseries.get("metadata", {}) if isinstance(timeseries.get("metadata"), dict) else {}
        original_values = timeseries.get("values", [])
        try:
            original_time_steps = int(len(original_values))
        except TypeError:
            original_time_steps = 0
        return {
            "frequency": [],
            "period": [],
            "power": [],
            "dominant_peaks": [],
            "metadata": {
                "variable": metadata.get("variable", "unknown"),
                "method": method,
                "window": window,
                "detrend": detrend,
                "n_points": int(values.size),
                "validation": {
                    "status": "insufficient_data",
                    "message": "Power spectrum requires at least four valid time steps.",
                    "required_valid_time_steps": 4,
                    "valid_time_steps": int(values.size),
                    "original_time_steps": original_time_steps,
                },
            },
        }

    if len(times) >= 2:
        delta_days = float(np.median(np.diff(times.values).astype("timedelta64[s]").astype(float)) / 86400.0)
        fs = 1.0 / delta_days if delta_days > 0.0 else 1.0
    else:
        delta_days = 1.0
        fs = 1.0

    nperseg = min(256, values.size)
    frequency, power = welch(values, fs=fs, detrend=detrend, window=window, nperseg=nperseg)
    period = np.full(frequency.shape, np.inf, dtype=float)
    positive_mask = frequency > 0.0
    period[positive_mask] = 1.0 / frequency[positive_mask]

    peak_indices, _ = find_peaks(power)
    ranked_peaks = sorted(peak_indices.tolist(), key=lambda idx: power[idx], reverse=True)[:3]
    dominant_peaks = [
        {
            "frequency": float(frequency[index]),
            "period": float(period[index]),
            "power": float(power[index]),
        }
        for index in ranked_peaks
    ]

    return {
        "frequency": frequency.tolist(),
        "period": period.tolist(),
        "power": power.tolist(),
        "dominant_peaks": dominant_peaks,
        "metadata": {
            "variable": timeseries.get("metadata", {}).get("variable", "unknown"),
            "method": method,
            "window": window,
            "detrend": detrend,
            "sample_interval_days": delta_days,
            "n_points": int(values.size),
        },
    }


def _prepare_field_and_index(
    field: xr.DataArray,
    index_timeseries: Dict,
    lag: int,
    remove_seasonal_cycle: bool,
) -> tuple[xr.DataArray, np.ndarray, pd.DatetimeIndex]:
    if "time" not in field.dims:
        raise ValueError("Statistical map tools require a time dimension")
    if "lat" not in field.dims or "lon" not in field.dims:
        raise ValueError("Statistical map tools require lat/lon dimensions")

    prepared_field = _select_surface_if_needed(field)
    field_times = pd.to_datetime(prepared_field.time.values)
    index_times = pd.to_datetime(index_timeseries["times"])
    index_values = np.asarray(index_timeseries["values"], dtype=float)

    common_times = field_times.intersection(index_times)
    if len(common_times) < 3:
        raise ValueError("Field and index_timeseries do not share enough overlapping time steps")

    prepared_field = prepared_field.sel(time=common_times)
    aligned_index = (
        pd.Series(index_values, index=index_times)
        .loc[common_times]
        .astype(float)
        .to_numpy()
    )

    if lag > 0:
        prepared_field = prepared_field.isel(time=slice(lag, None))
        aligned_index = aligned_index[:-lag]
    elif lag < 0:
        prepared_field = prepared_field.isel(time=slice(None, lag))
        aligned_index = aligned_index[-lag:]

    times = pd.to_datetime(prepared_field.time.values)
    if remove_seasonal_cycle:
        if not np.issubdtype(times.dtype, np.datetime64):
            raise ValueError("remove_seasonal_cycle requires datetime-like time coordinates")
        prepared_field = prepared_field.groupby("time.month") - prepared_field.groupby("time.month").mean("time")
        month_index = pd.Index(times.month)
        month_means = pd.Series(aligned_index).groupby(month_index).transform("mean").to_numpy()
        aligned_index = aligned_index - month_means

    if prepared_field.sizes["time"] != aligned_index.size:
        raise ValueError("Field and index alignment produced inconsistent lengths")

    return prepared_field, aligned_index.astype(float), times


def _select_surface_if_needed(field: xr.DataArray) -> xr.DataArray:
    if "depth" in field.dims:
        return field.isel(depth=0)
    if "z" in field.dims:
        return field.isel(z=0)
    return field


def _dataarray_to_spatial_payload(field: xr.DataArray, *, label: str = "spatial statistics field") -> Dict:
    values = dataarray_to_numpy(
        field,
        label=label,
        dtype=float,
        start=0.0,
        end=1.0,
    )
    return {
        "lon": [float(value) for value in field.lon.values],
        "lat": [float(value) for value in field.lat.values],
        "values": values,
        "metadata": {
            "variable": field.name or "unknown",
            "units": field.attrs.get("units", ""),
            "statistics": {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            },
        },
    }
