"""
Transect-based transport diagnostics.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import xarray as xr

from domain.ocean.data_access.partitioned import (
    PartitionedDataArray,
    find_partitioned_values,
    materialize_partitioned_xarray,
    validate_compatible_partitioned_inputs,
)
from domain.ocean.analysis._transect import (
    _compute_left_normal_components,
    _distance_axis_m,
    _interp_along_transect,
)
from domain.ocean.data_access.load import get_depth_dim
from domain.ocean.dask_utils import (
    compute_together_with_progress,
    dataarray_to_numpy,
    report_phase,
)


DEFAULT_TRANSPORT_TRANSECT_SAMPLES = 120


def compute_volume_transport(
    u: xr.DataArray,
    v: xr.DataArray,
    transect_points,
    depth_range: Optional[Tuple[float, float]] = None,
    n_samples: int = DEFAULT_TRANSPORT_TRANSECT_SAMPLES,
    method: Literal["linear", "nearest"] = "linear",
) -> Dict:
    """
    Compute normal volume transport through a sampled transect.
    """
    if find_partitioned_values((u, v)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_volume_transport",
            tool_func=compute_volume_transport,
            params={
                "u": u,
                "v": v,
                "transect_points": transect_points,
                "depth_range": depth_range,
                "n_samples": n_samples,
                "method": method,
            },
        )
    u = materialize_partitioned_xarray(u)
    v = materialize_partitioned_xarray(v)

    u_field, v_field = xr.align(u, v, join="inner")
    if "time" not in u_field.dims:
        raise ValueError("compute_volume_transport requires a time dimension")

    depth_dim = get_depth_dim(u_field)
    if depth_dim is None:
        raise ValueError("compute_volume_transport requires velocity fields with a depth dimension")

    u_field = _select_depth_range(u_field, depth_range)
    v_field = _select_depth_range(v_field, depth_range)

    normal_velocity, sample_info = _compute_normal_velocity_section(
        u_field=u_field,
        v_field=v_field,
        transect_points=transect_points,
        n_samples=n_samples,
        method=method,
    )

    transport_series = _integrate_transport_xarray(normal_velocity, sample_info["distance_km"], depth_dim) / 1e6
    transport_values = dataarray_to_numpy(
        transport_series,
        label="volume transport time series",
        dtype=float,
        start=0.2,
        end=0.9,
    )
    times = [str(value) for value in transport_series.time.values]

    return {
        "times": times,
        "values": transport_values.tolist(),
        "metadata": {
            "variable": "volume_transport",
            "unit": "Sv",
            "units": "Sv",
            "transect_points": sample_info["transect_points"],
            "n_samples": int(n_samples),
            "method": method,
            "depth_range": list(depth_range) if depth_range is not None else None,
            "statistics": _compute_series_statistics(transport_values),
            "sign_convention": "positive values indicate transport to the left of the transect orientation",
        },
    }


def compute_heat_transport(
    u: xr.DataArray,
    v: xr.DataArray,
    temp: xr.DataArray,
    transect_points,
    depth_range: Optional[Tuple[float, float]] = None,
    rho0: float = 1025.0,
    cp: float = 3990.0,
    n_samples: int = DEFAULT_TRANSPORT_TRANSECT_SAMPLES,
    method: Literal["linear", "nearest"] = "linear",
) -> Dict:
    """
    Compute a transect-integrated advective heat transport proxy.
    """
    if find_partitioned_values((u, v, temp)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_heat_transport",
            tool_func=compute_heat_transport,
            params={
                "u": u,
                "v": v,
                "temp": temp,
                "transect_points": transect_points,
                "depth_range": depth_range,
                "rho0": rho0,
                "cp": cp,
                "n_samples": n_samples,
                "method": method,
            },
        )
    u = materialize_partitioned_xarray(u)
    v = materialize_partitioned_xarray(v)
    temp = materialize_partitioned_xarray(temp)

    u_field, v_field, temp_field = xr.align(u, v, temp, join="inner")
    if "time" not in u_field.dims:
        raise ValueError("compute_heat_transport requires a time dimension")

    depth_dim = get_depth_dim(u_field)
    if depth_dim is None or get_depth_dim(temp_field) is None:
        raise ValueError("compute_heat_transport requires u, v, and temp fields with a depth dimension")

    u_field = _select_depth_range(u_field, depth_range)
    v_field = _select_depth_range(v_field, depth_range)
    temp_field = _select_depth_range(temp_field, depth_range)

    normal_velocity, sample_info = _compute_normal_velocity_section(
        u_field=u_field,
        v_field=v_field,
        transect_points=transect_points,
        n_samples=n_samples,
        method=method,
    )
    temp_section, _ = _interp_along_transect(temp_field, transect_points, n_samples=n_samples, method=method)
    heat_flux = rho0 * cp * temp_section * normal_velocity

    transport_series = _integrate_transport_xarray(heat_flux, sample_info["distance_km"], depth_dim)
    transport_values = dataarray_to_numpy(
        transport_series,
        label="heat transport time series",
        dtype=float,
        start=0.2,
        end=0.9,
    )
    times = [str(value) for value in transport_series.time.values]

    return {
        "times": times,
        "values": transport_values.tolist(),
        "metadata": {
            "variable": "heat_transport",
            "unit": "W",
            "units": "W",
            "scaled_units": "PW",
            "scaled_values_preview": (transport_values / 1e15).tolist()[:5],
            "transect_points": sample_info["transect_points"],
            "n_samples": int(n_samples),
            "method": method,
            "depth_range": list(depth_range) if depth_range is not None else None,
            "rho0": float(rho0),
            "cp": float(cp),
            "formula": "rho0 * cp * temp * normal_velocity",
            "statistics": _compute_series_statistics(transport_values),
            "sign_convention": "positive values indicate transport to the left of the transect orientation",
        },
    }


def compute_salt_transport(
    u: xr.DataArray,
    v: xr.DataArray,
    salt: xr.DataArray,
    transect_points,
    depth_range: Optional[Tuple[float, float]] = None,
    n_samples: int = DEFAULT_TRANSPORT_TRANSECT_SAMPLES,
    method: Literal["linear", "nearest"] = "linear",
) -> Dict:
    """
    Compute a transect-integrated advective salt-transport proxy.
    """
    if find_partitioned_values((u, v, salt)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_salt_transport",
            tool_func=compute_salt_transport,
            params={
                "u": u,
                "v": v,
                "salt": salt,
                "transect_points": transect_points,
                "depth_range": depth_range,
                "n_samples": n_samples,
                "method": method,
            },
        )
    u = materialize_partitioned_xarray(u)
    v = materialize_partitioned_xarray(v)
    salt = materialize_partitioned_xarray(salt)

    u_field, v_field, salt_field = xr.align(u, v, salt, join="inner")
    if "time" not in u_field.dims:
        raise ValueError("compute_salt_transport requires a time dimension")

    depth_dim = get_depth_dim(u_field)
    if depth_dim is None or get_depth_dim(salt_field) is None:
        raise ValueError("compute_salt_transport requires u, v, and salt fields with a depth dimension")

    u_field = _select_depth_range(u_field, depth_range)
    v_field = _select_depth_range(v_field, depth_range)
    salt_field = _select_depth_range(salt_field, depth_range)

    normal_velocity, sample_info = _compute_normal_velocity_section(
        u_field=u_field,
        v_field=v_field,
        transect_points=transect_points,
        n_samples=n_samples,
        method=method,
    )
    salt_section, _ = _interp_along_transect(salt_field, transect_points, n_samples=n_samples, method=method)
    salt_flux = salt_section * normal_velocity

    transport_series = _integrate_transport_xarray(salt_flux, sample_info["distance_km"], depth_dim)
    transport_values = dataarray_to_numpy(
        transport_series,
        label="salt transport time series",
        dtype=float,
        start=0.2,
        end=0.9,
    )
    times = [str(value) for value in transport_series.time.values]

    return {
        "times": times,
        "values": transport_values.tolist(),
        "metadata": {
            "variable": "salt_transport",
            "unit": "psu m^3 s^-1",
            "units": "psu m^3 s^-1",
            "scaled_units": "psu Sv",
            "scaled_values_preview": (transport_values / 1e6).tolist()[:5],
            "transect_points": sample_info["transect_points"],
            "n_samples": int(n_samples),
            "method": method,
            "depth_range": list(depth_range) if depth_range is not None else None,
            "formula": "salinity * normal_velocity",
            "statistics": _compute_series_statistics(transport_values),
            "sign_convention": "positive values indicate transport to the left of the transect orientation",
        },
    }


def compute_freshwater_transport(
    u: xr.DataArray,
    v: xr.DataArray,
    salt: xr.DataArray,
    transect_points,
    depth_range: Optional[Tuple[float, float]] = None,
    s_ref: float = 35.0,
    n_samples: int = DEFAULT_TRANSPORT_TRANSECT_SAMPLES,
    method: Literal["linear", "nearest"] = "linear",
) -> Dict:
    """
    Compute freshwater transport relative to a reference salinity.
    """
    if find_partitioned_values((u, v, salt)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_freshwater_transport",
            tool_func=compute_freshwater_transport,
            params={
                "u": u,
                "v": v,
                "salt": salt,
                "transect_points": transect_points,
                "depth_range": depth_range,
                "s_ref": s_ref,
                "n_samples": n_samples,
                "method": method,
            },
        )
    u = materialize_partitioned_xarray(u)
    v = materialize_partitioned_xarray(v)
    salt = materialize_partitioned_xarray(salt)

    u_field, v_field, salt_field = xr.align(u, v, salt, join="inner")
    if "time" not in u_field.dims:
        raise ValueError("compute_freshwater_transport requires a time dimension")
    if s_ref == 0.0:
        raise ValueError("s_ref must be non-zero")

    depth_dim = get_depth_dim(u_field)
    if depth_dim is None or get_depth_dim(salt_field) is None:
        raise ValueError("compute_freshwater_transport requires u, v, and salt fields with a depth dimension")

    u_field = _select_depth_range(u_field, depth_range)
    v_field = _select_depth_range(v_field, depth_range)
    salt_field = _select_depth_range(salt_field, depth_range)

    normal_velocity, sample_info = _compute_normal_velocity_section(
        u_field=u_field,
        v_field=v_field,
        transect_points=transect_points,
        n_samples=n_samples,
        method=method,
    )
    salt_section, _ = _interp_along_transect(salt_field, transect_points, n_samples=n_samples, method=method)
    freshwater_fraction = 1.0 - (salt_section / float(s_ref))
    freshwater_flux = freshwater_fraction * normal_velocity

    transport_series = _integrate_transport_xarray(freshwater_flux, sample_info["distance_km"], depth_dim) / 1e6
    transport_values = dataarray_to_numpy(
        transport_series,
        label="freshwater transport time series",
        dtype=float,
        start=0.2,
        end=0.9,
    )
    times = [str(value) for value in transport_series.time.values]

    return {
        "times": times,
        "values": transport_values.tolist(),
        "metadata": {
            "variable": "freshwater_transport",
            "unit": "Sv",
            "units": "Sv",
            "transect_points": sample_info["transect_points"],
            "n_samples": int(n_samples),
            "method": method,
            "depth_range": list(depth_range) if depth_range is not None else None,
            "s_ref": float(s_ref),
            "formula": "(1 - salinity / s_ref) * normal_velocity",
            "statistics": _compute_series_statistics(transport_values),
            "sign_convention": "positive values indicate transport to the left of the transect orientation",
        },
    }


def compute_transport_by_layer(
    u: xr.DataArray,
    v: xr.DataArray,
    transect_points,
    layer_bounds,
    transport_type: Literal["volume", "heat", "salt", "freshwater"] = "volume",
    temp: Optional[xr.DataArray] = None,
    salt: Optional[xr.DataArray] = None,
    rho0: float = 1025.0,
    cp: float = 3990.0,
    s_ref: float = 35.0,
    n_samples: int = DEFAULT_TRANSPORT_TRANSECT_SAMPLES,
    method: Literal["linear", "nearest"] = "linear",
) -> Dict:
    """
    Compute layer-by-layer transport time series across a transect.
    """
    if find_partitioned_values((u, v, temp, salt)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_transport_by_layer",
            tool_func=compute_transport_by_layer,
            params={
                "u": u,
                "v": v,
                "transect_points": transect_points,
                "layer_bounds": layer_bounds,
                "transport_type": transport_type,
                "temp": temp,
                "salt": salt,
                "rho0": rho0,
                "cp": cp,
                "s_ref": s_ref,
                "n_samples": n_samples,
                "method": method,
            },
        )
    u = materialize_partitioned_xarray(u)
    v = materialize_partitioned_xarray(v)
    temp = materialize_partitioned_xarray(temp)
    salt = materialize_partitioned_xarray(salt)

    if not layer_bounds:
        raise ValueError("compute_transport_by_layer requires at least one depth layer")

    layer_results = []
    reference_times = None
    reference_units = None
    for bounds in layer_bounds:
        if len(bounds) != 2:
            raise ValueError("Each layer bound must contain exactly two depth values")
        depth_range = (float(bounds[0]), float(bounds[1]))

        if transport_type == "volume":
            result = compute_volume_transport(
                u=u,
                v=v,
                transect_points=transect_points,
                depth_range=depth_range,
                n_samples=n_samples,
                method=method,
            )
        elif transport_type == "heat":
            if temp is None:
                raise ValueError("temp is required when transport_type == 'heat'")
            result = compute_heat_transport(
                u=u,
                v=v,
                temp=temp,
                transect_points=transect_points,
                depth_range=depth_range,
                rho0=rho0,
                cp=cp,
                n_samples=n_samples,
                method=method,
            )
        elif transport_type == "salt":
            if salt is None:
                raise ValueError("salt is required when transport_type == 'salt'")
            result = compute_salt_transport(
                u=u,
                v=v,
                salt=salt,
                transect_points=transect_points,
                depth_range=depth_range,
                n_samples=n_samples,
                method=method,
            )
        elif transport_type == "freshwater":
            if salt is None:
                raise ValueError("salt is required when transport_type == 'freshwater'")
            result = compute_freshwater_transport(
                u=u,
                v=v,
                salt=salt,
                transect_points=transect_points,
                depth_range=depth_range,
                s_ref=s_ref,
                n_samples=n_samples,
                method=method,
            )
        else:
            raise ValueError(f"Unsupported transport_type: {transport_type}")

        if reference_times is None:
            reference_times = list(result["times"])
            reference_units = result["metadata"].get("units")
        layer_results.append(
            {
                "label": _format_layer_label(depth_range),
                "depth_range": list(depth_range),
                "values": list(result["values"]),
                "statistics": result["metadata"].get("statistics", {}),
            }
        )

    return {
        "times": reference_times or [],
        "layers": layer_results,
        "metadata": {
            "transport_type": transport_type,
            "units": reference_units,
            "transect_points": transect_points,
            "n_layers": len(layer_results),
            "layer_bounds": [list(bounds) for bounds in layer_bounds],
            "n_samples": int(n_samples),
            "method": method,
            "rho0": float(rho0) if transport_type == "heat" else None,
            "cp": float(cp) if transport_type == "heat" else None,
            "s_ref": float(s_ref) if transport_type == "freshwater" else None,
        },
    }


def compute_transport_streamfunction_map(
    u: xr.DataArray,
    v: xr.DataArray,
    depth_range: Optional[Tuple[float, float]] = None,
    time_aggregation: Literal["mean", "max", "min", "median", "std"] = "mean",
    regional_gauge: Optional[str] = None,
) -> Dict:
    """
    Estimate a depth-integrated volume transport streamfunction map.

    The implementation follows the Gan et al. (2016) Figure 10 sign convention:
    northward transport corresponds to a negative cross-shelf/zonal gradient of
    the streamfunction, i.e. ``v = -d(psi)/dx``. The resulting map is returned in
    Sverdrup for map plotting.
    """
    if find_partitioned_values((u, v)):
        return _compute_partitioned_transport_streamfunction_map(
            u=u,
            v=v,
            depth_range=depth_range,
            time_aggregation=time_aggregation,
            regional_gauge=regional_gauge,
        )
    u = materialize_partitioned_xarray(u)
    v = materialize_partitioned_xarray(v)

    depth_integrated_u, depth_integrated_v, had_time_dim = _prepare_depth_integrated_transport_pair(
        u=u,
        v=v,
        depth_range=depth_range,
        time_aggregation=time_aggregation,
    )

    return _build_transport_streamfunction_result(
        depth_integrated_u=depth_integrated_u,
        depth_integrated_v=depth_integrated_v,
        depth_range=depth_range,
        time_aggregation=time_aggregation if had_time_dim else None,
        partition_strategy=None,
        regional_gauge=regional_gauge,
    )


def _compute_partitioned_transport_streamfunction_map(
    *,
    u,
    v,
    depth_range: Optional[Tuple[float, float]],
    time_aggregation: Literal["mean", "max", "min", "median", "std"],
    regional_gauge: Optional[str],
) -> Dict:
    validate_compatible_partitioned_inputs((u, v), context="compute_transport_streamfunction_map inputs")
    if not isinstance(u, PartitionedDataArray) or not isinstance(v, PartitionedDataArray):
        return compute_transport_streamfunction_map(
            u=materialize_partitioned_xarray(u),
            v=materialize_partitioned_xarray(v),
            depth_range=depth_range,
            time_aggregation=time_aggregation,
            regional_gauge=regional_gauge,
        )
    if time_aggregation != "mean":
        return compute_transport_streamfunction_map(
            u=materialize_partitioned_xarray(u),
            v=materialize_partitioned_xarray(v),
            depth_range=depth_range,
            time_aggregation=time_aggregation,
            regional_gauge=regional_gauge,
        )

    sum_u = None
    sum_v = None
    weight_u = None
    weight_v = None
    template_u = None
    template_v = None
    had_time_dim = False
    total_time_steps = 0

    for u_part, v_part in zip(u.partitions, v.partitions):
        part_time_steps = max(int(u_part.sizes.get("time", 1)), int(v_part.sizes.get("time", 1)))
        total_time_steps += part_time_steps
        part_u, part_v, part_had_time = _prepare_depth_integrated_transport_pair(
            u=u_part,
            v=v_part,
            depth_range=depth_range,
            time_aggregation="mean",
        )
        had_time_dim = had_time_dim or part_had_time
        if template_u is None:
            template_u = part_u
            template_v = part_v
            shape = tuple(part_u.transpose("lat", "lon").shape)
            sum_u = np.zeros(shape, dtype=float)
            sum_v = np.zeros(shape, dtype=float)
            weight_u = np.zeros(shape, dtype=float)
            weight_v = np.zeros(shape, dtype=float)
        else:
            _require_same_horizontal_grid(template_u, part_u, field_name="u")
            _require_same_horizontal_grid(template_v, part_v, field_name="v")

        u_values = np.asarray(part_u.transpose("lat", "lon").values, dtype=float)
        v_values = np.asarray(part_v.transpose("lat", "lon").values, dtype=float)
        u_valid = np.isfinite(u_values)
        v_valid = np.isfinite(v_values)
        sum_u += np.where(u_valid, u_values * part_time_steps, 0.0)
        sum_v += np.where(v_valid, v_values * part_time_steps, 0.0)
        weight_u += np.where(u_valid, part_time_steps, 0.0)
        weight_v += np.where(v_valid, part_time_steps, 0.0)

    mean_u_values = np.full_like(sum_u, np.nan, dtype=float)
    mean_v_values = np.full_like(sum_v, np.nan, dtype=float)
    np.divide(sum_u, weight_u, out=mean_u_values, where=weight_u > 0.0)
    np.divide(sum_v, weight_v, out=mean_v_values, where=weight_v > 0.0)

    assert template_u is not None and template_v is not None
    mean_u = xr.DataArray(
        mean_u_values,
        coords={"lat": template_u["lat"].values, "lon": template_u["lon"].values},
        dims=("lat", "lon"),
        attrs=dict(template_u.attrs),
        name=template_u.name,
    )
    mean_v = xr.DataArray(
        mean_v_values,
        coords={"lat": template_v["lat"].values, "lon": template_v["lon"].values},
        dims=("lat", "lon"),
        attrs=dict(template_v.attrs),
        name=template_v.name,
    )

    return _build_transport_streamfunction_result(
        depth_integrated_u=mean_u,
        depth_integrated_v=mean_v,
        depth_range=depth_range,
        time_aggregation="mean" if had_time_dim else None,
        partition_strategy="mean_depth_integrated_transport_then_2d_solve",
        regional_gauge=regional_gauge,
        extra_metadata={
            "n_partitions": len(u.partitions),
            "time_steps_averaged": int(total_time_steps),
        },
    )


def _prepare_depth_integrated_transport_pair(
    *,
    u: xr.DataArray,
    v: xr.DataArray,
    depth_range: Optional[Tuple[float, float]],
    time_aggregation: Literal["mean", "max", "min", "median", "std"],
) -> Tuple[xr.DataArray, xr.DataArray, bool]:
    u_field, v_field = xr.align(u, v, join="inner")
    if "lat" not in u_field.dims or "lon" not in u_field.dims or "lat" not in v_field.dims or "lon" not in v_field.dims:
        raise ValueError("compute_transport_streamfunction_map requires lat/lon dimensions")

    u_depth_dim = get_depth_dim(u_field)
    v_depth_dim = get_depth_dim(v_field)
    if u_depth_dim is None or v_depth_dim is None:
        raise ValueError("compute_transport_streamfunction_map requires velocity fields with a depth dimension")

    u_field = _select_depth_range(u_field, depth_range)
    v_field = _select_depth_range(v_field, depth_range)
    u_field, v_field = xr.align(u_field, v_field, join="inner")
    had_time_dim = "time" in u_field.dims or "time" in v_field.dims
    if time_aggregation == "mean" and had_time_dim:
        u_field = _aggregate_optional_time(u_field, aggregation=time_aggregation)
        v_field = _aggregate_optional_time(v_field, aggregation=time_aggregation)

    depth_integrated_u = _vertical_integrate_field(u_field, depth_dim=u_depth_dim)
    depth_integrated_v = _vertical_integrate_field(v_field, depth_dim=v_depth_dim)
    depth_integrated_u = _aggregate_optional_time(depth_integrated_u, aggregation=time_aggregation)
    depth_integrated_v = _aggregate_optional_time(depth_integrated_v, aggregation=time_aggregation)
    depth_integrated_u, depth_integrated_v = xr.align(depth_integrated_u, depth_integrated_v, join="inner")
    return depth_integrated_u, depth_integrated_v, had_time_dim


def _build_transport_streamfunction_result(
    *,
    depth_integrated_u: xr.DataArray,
    depth_integrated_v: xr.DataArray,
    depth_range: Optional[Tuple[float, float]],
    time_aggregation: Optional[str],
    partition_strategy: Optional[str],
    regional_gauge: Optional[str] = None,
    extra_metadata: Optional[Dict] = None,
) -> Dict:
    depth_integrated_u, depth_integrated_v = xr.align(depth_integrated_u, depth_integrated_v, join="inner")
    u_field = depth_integrated_u.transpose("lat", "lon")
    v_field = depth_integrated_v.transpose("lat", "lon")
    lat = np.asarray(v_field["lat"].values, dtype=float)
    lon = np.asarray(v_field["lon"].values, dtype=float)
    u_field, v_field = compute_together_with_progress(
        (u_field, v_field),
        label="depth-integrated transport vector",
        start=0.1,
        end=0.6,
    )
    zonal_transport = np.asarray(u_field.values, dtype=float)
    meridional_transport = np.asarray(v_field.values, dtype=float)
    report_phase(
        phase="building_mask",
        message="Building transport streamfunction wet/land mask",
        percent=0.62,
    )
    land_mask = _build_land_mask(lat=lat, lon=lon)
    wet_mask = np.isfinite(zonal_transport) & np.isfinite(meridional_transport)
    if land_mask is not None:
        wet_mask = wet_mask & ~land_mask
        zonal_transport = np.where(wet_mask, zonal_transport, np.nan)
        meridional_transport = np.where(wet_mask, meridional_transport, np.nan)

    report_phase(
        phase="solving_streamfunction",
        message="Solving global transport streamfunction",
        percent=0.7,
    )
    streamfunction_sv = _compute_streamfunction_from_transport_vector_field(
        zonal_transport=zonal_transport,
        meridional_transport=meridional_transport,
        lat=lat,
        lon=lon,
        wet_mask=wet_mask,
    ) / 1e6
    streamfunction_sv = np.where(wet_mask, streamfunction_sv, np.nan)
    report_phase(
        phase="solving_streamfunction",
        message="Solved global transport streamfunction",
        percent=0.82,
    )
    base_streamfunction_sv = streamfunction_sv
    streamfunction_sv, regional_gauge_metadata = _apply_transport_streamfunction_regional_gauge(
        streamfunction_sv=streamfunction_sv,
        lat=lat,
        lon=lon,
        wet_mask=wet_mask,
        regional_gauge=regional_gauge,
    )
    land_mask_applied = land_mask is not None
    metadata = {
        "variable": "transport_streamfunction",
        "units": "Sv",
        "unit": "Sv",
        "time_aggregation": time_aggregation,
        "depth_range": list(depth_range) if depth_range is not None else None,
        "statistics": _compute_field_statistics(streamfunction_sv),
        "sign_convention": (
            "Gan et al. (2016) Figure 10 convention: v = -dpsi/dx and u = dpsi/dy; "
            "northward transport corresponds to a negative zonal/cross-shelf "
            "streamfunction gradient"
        ),
        "estimation_method": (
            "two-dimensional least-squares streamfunction fit to depth-integrated "
            "zonal and meridional transports"
        ),
        "integration_origin": "southeast_valid_wet_point_zero",
        "gauge": "southeast_valid_wet_point_zero_per_connected_component",
        "gauge_reference": "right-bottom/southeast valid wet grid point in each connected wet component",
        "mask_policy": "break_at_nan_and_land_segments",
        "bathymetry_mask_applied": False,
        "land_mask_applied": bool(land_mask_applied),
        "land_mask_source": "Natural Earth 10m land polygons" if land_mask_applied else None,
        "disable_contour_preview": True,
    }
    if regional_gauge_metadata:
        metadata.update(regional_gauge_metadata)
        metadata["raw_global_statistics"] = _compute_field_statistics(base_streamfunction_sv)
    if partition_strategy is not None:
        metadata["partition_strategy"] = partition_strategy
    if extra_metadata:
        metadata.update(extra_metadata)
    report_phase(
        phase="preparing_map_payload",
        message="Preparing transport streamfunction map payload",
        percent=0.96,
    )

    return {
        "lon": lon.tolist(),
        "lat": lat.tolist(),
        "values": streamfunction_sv,
        "metadata": metadata,
    }


def compute_transect_normal_flux_hovmoller(
    u: xr.DataArray,
    v: xr.DataArray,
    transect_points,
    depth_range: Optional[Tuple[float, float]] = None,
    n_samples: int = DEFAULT_TRANSPORT_TRANSECT_SAMPLES,
    method: Literal["linear", "nearest"] = "linear",
) -> Dict:
    """
    Build a time-depth Hovmoller of along-transect integrated normal volume flux.

    The retained vertical coordinate is depth; values are the normal flux
    integrated across the sampled transect length, with units m^2 s^-1.
    """
    if find_partitioned_values((u, v)):
        from packages.tool_loader.partitioned_execution import execute_partition_aware

        return execute_partition_aware(
            tool_name="compute_transect_normal_flux_hovmoller",
            tool_func=compute_transect_normal_flux_hovmoller,
            params={
                "u": u,
                "v": v,
                "transect_points": transect_points,
                "depth_range": depth_range,
                "n_samples": n_samples,
                "method": method,
            },
        )
    u = materialize_partitioned_xarray(u)
    v = materialize_partitioned_xarray(v)

    u_field, v_field = xr.align(u, v, join="inner")
    if "time" not in u_field.dims:
        raise ValueError("compute_transect_normal_flux_hovmoller requires a time dimension")

    depth_dim = get_depth_dim(u_field)
    if depth_dim is None:
        raise ValueError("compute_transect_normal_flux_hovmoller requires velocity fields with a depth dimension")

    u_field = _select_depth_range(u_field, depth_range)
    v_field = _select_depth_range(v_field, depth_range)

    normal_velocity, sample_info = _compute_normal_velocity_section(
        u_field=u_field,
        v_field=v_field,
        transect_points=transect_points,
        n_samples=n_samples,
        method=method,
    )

    ordered = normal_velocity.transpose("time", depth_dim, "distance")
    distance_axis_m = _distance_axis_m(sample_info["distance_km"])
    flux_field = _integrate_valid_distance_xarray(
        ordered,
        distance_axis_m,
    )
    flux_per_depth = dataarray_to_numpy(
        flux_field,
        label="normal flux Hovmoller matrix",
        dtype=float,
        start=0.2,
        end=0.9,
    )

    return {
        "time": [str(value) for value in flux_field.time.values],
        "spatial_coord": flux_field[depth_dim].values.tolist(),
        "values": flux_per_depth,
        "metadata": {
            "diagram_type": "time_depth",
            "spatial_dim": depth_dim,
            "variable": "normal_volume_flux",
            "units": "m^2 s^-1",
            "unit": "m^2 s^-1",
            "transect_points": sample_info["transect_points"],
            "n_samples": int(n_samples),
            "method": method,
            "depth_range": list(depth_range) if depth_range is not None else None,
            "statistics": _compute_field_statistics(flux_per_depth),
            "sign_convention": "positive values indicate flux to the left of the transect orientation",
        },
    }


def _select_depth_range(data: xr.DataArray, depth_range: Optional[Tuple[float, float]]) -> xr.DataArray:
    depth_dim = get_depth_dim(data)
    if depth_dim is None or depth_range is None:
        return data
    from domain.ocean.data_access.load import _normalize_depth_range
    depth_values = np.asarray(data[depth_dim].values, dtype=float)
    nr = _normalize_depth_range(depth_values, depth_range)
    return data.sel({depth_dim: _build_coord_slice(data[depth_dim].values, nr)})


def _compute_normal_velocity_section(
    u_field: xr.DataArray,
    v_field: xr.DataArray,
    transect_points,
    n_samples: int,
    method: Literal["linear", "nearest"],
) -> Tuple[xr.DataArray, Dict]:
    u_section, sample_info = _interp_along_transect(
        u_field,
        transect_points,
        n_samples=n_samples,
        method=method,
    )
    v_section, _ = _interp_along_transect(
        v_field,
        transect_points,
        n_samples=n_samples,
        method=method,
    )

    nx, ny = _compute_left_normal_components(sample_info["lon"], sample_info["lat"])
    normal_velocity = u_section * xr.DataArray(nx, dims="distance") + v_section * xr.DataArray(ny, dims="distance")
    return normal_velocity, sample_info


def _integrate_transport(field: xr.DataArray, distance_km, depth_dim: str) -> np.ndarray:
    distance_axis_m = _distance_axis_m(distance_km)
    depth_axis_m = np.abs(np.asarray(field[depth_dim].values, dtype=float))
    ordered = field.transpose("time", depth_dim, "distance")
    values = np.asarray(ordered.values, dtype=float)

    if depth_axis_m.size < 2:
        raise ValueError("Transport integration requires at least two depth levels")

    depth_order = np.argsort(depth_axis_m)
    depth_sorted_values = values[:, depth_order, :]
    depth_sorted_values = np.moveaxis(depth_sorted_values, 1, -1)
    integrated_depth = _integrate_valid_depth_prefix(depth_sorted_values, depth_axis_m[depth_order])
    integrated_section = _integrate_valid_distance(integrated_depth, distance_axis_m)
    return integrated_section


def _integrate_transport_xarray(field: xr.DataArray, distance_km, depth_dim: str) -> xr.DataArray:
    """Lazy time-series transport integration over depth and transect distance."""
    distance_axis_m = _distance_axis_m(distance_km)
    depth_axis_m = np.abs(np.asarray(field[depth_dim].values, dtype=float))
    if depth_axis_m.size < 2:
        raise ValueError("Transport integration requires at least two depth levels")

    ordered = field.transpose("time", depth_dim, "distance")
    depth_order = np.argsort(depth_axis_m)
    depth_sorted = ordered.isel({depth_dim: depth_order})
    depth_integrated = _integrate_valid_depth_prefix_xarray(
        depth_sorted,
        depth_axis_m[depth_order],
        depth_dim,
    )
    return _integrate_valid_distance_xarray(depth_integrated, distance_axis_m)


def _vertical_integrate_field(field: xr.DataArray, depth_dim: str) -> xr.DataArray:
    depth_axis_m = np.abs(np.asarray(field[depth_dim].values, dtype=float))
    if depth_axis_m.size == 0:
        raise ValueError("Depth integration requires at least one depth level")
    if depth_axis_m.size == 1:
        valid = field.notnull().any(dim=depth_dim)
        integrated = field.isel({depth_dim: 0})
        return integrated.where(valid)

    ordered = field.transpose(*[dim for dim in field.dims if dim != depth_dim], depth_dim)
    depth_order = np.argsort(depth_axis_m)
    ordered = ordered.isel({depth_dim: depth_order})
    integrated = _integrate_valid_depth_prefix_xarray(
        ordered,
        depth_axis_m[depth_order],
        depth_dim,
    )
    integrated.attrs = dict(field.attrs)
    integrated.name = field.name
    return integrated


def _integrate_valid_depth_prefix_xarray(
    field: xr.DataArray,
    depth_axis_m: np.ndarray,
    depth_dim: str,
) -> xr.DataArray:
    """Integrate lazily from the surface until the first invalid depth level."""
    if field.sizes.get(depth_dim, 0) != depth_axis_m.size:
        raise ValueError("Depth axis length must match field depth dimension")
    if depth_axis_m.size < 2:
        return field.isel({depth_dim: 0}).where(field.isel({depth_dim: 0}).notnull())

    interval_coord = np.arange(depth_axis_m.size - 1)
    finite = field.notnull()
    valid_prefix = finite.astype("int8").cumprod(dim=depth_dim) > 0
    left = field.isel({depth_dim: slice(None, -1)}).assign_coords({depth_dim: interval_coord})
    right = field.isel({depth_dim: slice(1, None)}).assign_coords({depth_dim: interval_coord})
    interval_valid = valid_prefix.isel({depth_dim: slice(1, None)}).assign_coords({depth_dim: interval_coord})
    delta = xr.DataArray(
        np.diff(depth_axis_m),
        coords={depth_dim: interval_coord},
        dims=(depth_dim,),
    )
    interval_values = 0.5 * (left + right) * delta
    integrated = xr.where(interval_valid, interval_values, 0.0).sum(dim=depth_dim, skipna=True)
    valid_count = valid_prefix.sum(dim=depth_dim)
    return integrated.where(valid_count >= 2)


def _integrate_valid_depth_prefix(values: np.ndarray, depth_axis_m: np.ndarray) -> np.ndarray:
    """Integrate each column from the surface until the first invalid depth value."""
    if values.shape[-1] != depth_axis_m.size:
        raise ValueError("Depth axis length must match the last dimension of values")
    if depth_axis_m.size < 2:
        return np.where(np.isfinite(values[..., 0]), values[..., 0], np.nan)

    finite = np.isfinite(values)
    valid_prefix = np.logical_and.accumulate(finite, axis=-1)
    valid_count = np.sum(valid_prefix, axis=-1)
    interval_valid = valid_prefix[..., 1:]
    interval_values = 0.5 * (values[..., :-1] + values[..., 1:]) * np.diff(depth_axis_m)
    integrated = np.sum(np.where(interval_valid, interval_values, 0.0), axis=-1)
    return np.where(valid_count >= 2, integrated, np.nan)


def _integrate_valid_distance_xarray(field: xr.DataArray, distance_axis_m: np.ndarray) -> xr.DataArray:
    """Integrate lazily along distance while breaking at invalid segments."""
    if "distance" not in field.dims:
        raise ValueError("Distance integration requires a distance dimension")
    if field.sizes.get("distance", 0) != distance_axis_m.size:
        raise ValueError("Distance axis length must match field distance dimension")
    if distance_axis_m.size < 2:
        return field.isel(distance=0).where(field.isel(distance=0).notnull())

    interval_coord = np.arange(distance_axis_m.size - 1)
    left = field.isel(distance=slice(None, -1)).assign_coords(distance=interval_coord)
    right = field.isel(distance=slice(1, None)).assign_coords(distance=interval_coord)
    interval_valid = left.notnull() & right.notnull()
    delta = xr.DataArray(
        np.diff(distance_axis_m),
        coords={"distance": interval_coord},
        dims=("distance",),
    )
    interval_values = 0.5 * (left + right) * delta
    integrated = xr.where(interval_valid, interval_values, 0.0).sum(dim="distance", skipna=True)
    valid_intervals = interval_valid.sum(dim="distance")
    return integrated.where(valid_intervals >= 1)


def _integrate_valid_distance(values: np.ndarray, distance_axis_m: np.ndarray) -> np.ndarray:
    """Integrate along the last axis while skipping invalid land/outside-domain gaps."""
    if values.shape[-1] != distance_axis_m.size:
        raise ValueError("Distance axis length must match the last dimension of values")
    if distance_axis_m.size < 2:
        return np.where(np.isfinite(values[..., 0]), values[..., 0], np.nan)

    finite = np.isfinite(values)
    interval_valid = finite[..., :-1] & finite[..., 1:]
    interval_values = 0.5 * (values[..., :-1] + values[..., 1:]) * np.diff(distance_axis_m)
    integrated = np.sum(np.where(interval_valid, interval_values, 0.0), axis=-1)
    valid_intervals = np.sum(interval_valid, axis=-1)
    return np.where(valid_intervals >= 1, integrated, np.nan)


def _aggregate_optional_time(
    field: xr.DataArray,
    aggregation: Literal["mean", "max", "min", "median", "std"],
) -> xr.DataArray:
    if "time" not in field.dims:
        return field
    if aggregation == "mean":
        return field.mean(dim="time", skipna=True)
    if aggregation == "max":
        return field.max(dim="time", skipna=True)
    if aggregation == "min":
        return field.min(dim="time", skipna=True)
    if aggregation == "median":
        return field.median(dim="time", skipna=True)
    if aggregation == "std":
        return field.std(dim="time", skipna=True)
    raise ValueError(f"Unsupported time_aggregation: {aggregation}")


def _compute_streamfunction_from_meridional_transport(
    *,
    meridional_transport: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    if lon.size < 2:
        raise ValueError("Streamfunction estimation requires at least two longitude points")

    psi = np.full_like(meridional_transport, np.nan, dtype=float)
    earth_radius_m = 6_371_000.0

    for lat_index, latitude in enumerate(lat):
        cos_lat = np.cos(np.deg2rad(float(latitude)))
        row = meridional_transport[lat_index]
        finite = np.isfinite(row)
        for lon_index in range(lon.size):
            if not finite[lon_index]:
                continue
            if lon_index == 0 or not finite[lon_index - 1]:
                psi[lat_index, lon_index] = 0.0
                continue
            delta_lon_rad = np.deg2rad(float(lon[lon_index] - lon[lon_index - 1]))
            delta_x = earth_radius_m * cos_lat * abs(delta_lon_rad)
            left = row[lon_index - 1]
            right = row[lon_index]
            segment_flux = -0.5 * (left + right) * delta_x
            psi[lat_index, lon_index] = psi[lat_index, lon_index - 1] + segment_flux

    return psi


def _compute_streamfunction_from_transport_vector_field(
    *,
    zonal_transport: np.ndarray,
    meridional_transport: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    wet_mask: np.ndarray,
) -> np.ndarray:
    """Fit psi so that meridional transport = -dpsi/dx and zonal transport = dpsi/dy."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    if zonal_transport.shape != meridional_transport.shape:
        raise ValueError("zonal and meridional transport arrays must have the same shape")
    if zonal_transport.shape != wet_mask.shape:
        raise ValueError("wet_mask must match transport array shape")
    if zonal_transport.shape != (lat.size, lon.size):
        raise ValueError("transport array shape must match lat/lon coordinates")
    if lon.size < 2 or lat.size < 1:
        raise ValueError("Streamfunction estimation requires a horizontal grid")

    wet = np.asarray(wet_mask, dtype=bool)
    index_map, components = _wet_component_indices(wet)
    n_unknowns = int(np.max(index_map) + 1) if np.any(wet) else 0
    psi = np.full_like(meridional_transport, np.nan, dtype=float)
    if n_unknowns == 0:
        return psi

    earth_radius_m = 6_371_000.0
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row_index = 0

    for lat_index, latitude in enumerate(lat):
        cos_lat = np.cos(np.deg2rad(float(latitude)))
        for lon_index in range(1, lon.size):
            if not (wet[lat_index, lon_index] and wet[lat_index, lon_index - 1]):
                continue
            left = meridional_transport[lat_index, lon_index - 1]
            right = meridional_transport[lat_index, lon_index]
            if not (np.isfinite(left) and np.isfinite(right)):
                continue
            delta_lon_rad = np.deg2rad(float(lon[lon_index] - lon[lon_index - 1]))
            delta_x = earth_radius_m * cos_lat * delta_lon_rad
            rows.extend([row_index, row_index])
            cols.extend([int(index_map[lat_index, lon_index]), int(index_map[lat_index, lon_index - 1])])
            data.extend([1.0, -1.0])
            rhs.append(float(-0.5 * (left + right) * delta_x))
            row_index += 1

    for lat_index in range(1, lat.size):
        delta_lat_rad = np.deg2rad(float(lat[lat_index] - lat[lat_index - 1]))
        delta_y = earth_radius_m * delta_lat_rad
        for lon_index in range(lon.size):
            if not (wet[lat_index, lon_index] and wet[lat_index - 1, lon_index]):
                continue
            south = zonal_transport[lat_index - 1, lon_index]
            north = zonal_transport[lat_index, lon_index]
            if not (np.isfinite(south) and np.isfinite(north)):
                continue
            rows.extend([row_index, row_index])
            cols.extend([int(index_map[lat_index, lon_index]), int(index_map[lat_index - 1, lon_index])])
            data.extend([1.0, -1.0])
            rhs.append(float(0.5 * (south + north) * delta_y))
            row_index += 1

    for component in components:
        anchor = int(component[0])
        rows.append(row_index)
        cols.append(anchor)
        data.append(1.0)
        rhs.append(0.0)
        row_index += 1

    matrix = coo_matrix((data, (rows, cols)), shape=(row_index, n_unknowns)).tocsr()
    solution = lsqr(matrix, np.asarray(rhs, dtype=float), atol=1e-8, btol=1e-8, iter_lim=2000)[0]
    row_by_variable = np.full(n_unknowns, -1, dtype=int)
    col_by_variable = np.full(n_unknowns, -1, dtype=int)
    wet_rows, wet_cols = np.where(index_map >= 0)
    wet_variables = index_map[wet_rows, wet_cols]
    row_by_variable[wet_variables] = wet_rows
    col_by_variable[wet_variables] = wet_cols

    for component in components:
        component_indices = np.asarray(component, dtype=int)
        reference_index = _southeast_reference_variable_index(
            component_indices=component_indices,
            row_by_variable=row_by_variable,
            col_by_variable=col_by_variable,
            lat=lat,
            lon=lon,
        )
        solution[component_indices] = solution[component_indices] - float(solution[reference_index])

    psi_values = np.full(n_unknowns, np.nan, dtype=float)
    psi_values[:] = solution
    psi[wet] = psi_values[index_map[wet]]
    return psi


def _southeast_reference_variable_index(
    *,
    component_indices: np.ndarray,
    row_by_variable: np.ndarray,
    col_by_variable: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> int:
    """Select the wet point closest to the domain southeast corner for a component."""
    if component_indices.size == 0:
        raise ValueError("Cannot choose a streamfunction gauge reference from an empty component")
    rows = row_by_variable[component_indices]
    cols = col_by_variable[component_indices]
    lat_values = np.asarray(lat, dtype=float)[rows]
    lon_values = np.asarray(lon, dtype=float)[cols]
    target_lat = float(np.nanmin(lat))
    target_lon = float(np.nanmax(lon))
    lat_scale = max(float(np.nanmax(lat) - np.nanmin(lat)), 1.0)
    lon_scale = max(float(np.nanmax(lon) - np.nanmin(lon)), 1.0)
    distance = ((lat_values - target_lat) / lat_scale) ** 2 + ((lon_values - target_lon) / lon_scale) ** 2
    order = np.lexsort((lat_values, -lon_values, distance))
    return int(component_indices[int(order[0])])


def _wet_component_indices(wet_mask: np.ndarray) -> Tuple[np.ndarray, list[list[int]]]:
    index_map = np.full(wet_mask.shape, -1, dtype=int)
    wet_positions = np.argwhere(wet_mask)
    for variable_index, (lat_index, lon_index) in enumerate(wet_positions):
        index_map[int(lat_index), int(lon_index)] = variable_index

    visited = np.zeros(wet_mask.shape, dtype=bool)
    components: list[list[int]] = []
    n_lat, n_lon = wet_mask.shape
    for start_lat, start_lon in wet_positions:
        start = (int(start_lat), int(start_lon))
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: list[int] = []
        while stack:
            lat_index, lon_index = stack.pop()
            component.append(int(index_map[lat_index, lon_index]))
            for next_lat, next_lon in (
                (lat_index - 1, lon_index),
                (lat_index + 1, lon_index),
                (lat_index, lon_index - 1),
                (lat_index, lon_index + 1),
            ):
                if next_lat < 0 or next_lat >= n_lat or next_lon < 0 or next_lon >= n_lon:
                    continue
                if visited[next_lat, next_lon] or not wet_mask[next_lat, next_lon]:
                    continue
                visited[next_lat, next_lon] = True
                stack.append((next_lat, next_lon))
        components.append(component)
    return index_map, components


def _require_same_horizontal_grid(reference: xr.DataArray, candidate: xr.DataArray, *, field_name: str) -> None:
    ref_lat = np.asarray(reference["lat"].values, dtype=float)
    ref_lon = np.asarray(reference["lon"].values, dtype=float)
    cand_lat = np.asarray(candidate["lat"].values, dtype=float)
    cand_lon = np.asarray(candidate["lon"].values, dtype=float)
    if ref_lat.shape != cand_lat.shape or ref_lon.shape != cand_lon.shape:
        raise ValueError(f"Partitioned {field_name} transport grids do not match")
    if not np.allclose(ref_lat, cand_lat, equal_nan=True) or not np.allclose(ref_lon, cand_lon, equal_nan=True):
        raise ValueError(f"Partitioned {field_name} transport coordinates do not match")


def _build_land_mask(*, lat: np.ndarray, lon: np.ndarray) -> Optional[np.ndarray]:
    """Return True over land using cached Natural Earth polygons when available."""
    try:
        lat_key = tuple(float(value) for value in np.round(np.asarray(lat, dtype=float), 6))
        lon_key = tuple(float(value) for value in np.round(np.asarray(lon, dtype=float), 6))
        return _cached_land_mask(lat_key, lon_key)
    except Exception:
        return None


@lru_cache(maxsize=16)
def _cached_land_mask(lat_key: Tuple[float, ...], lon_key: Tuple[float, ...]) -> np.ndarray:
    from cartopy.io import shapereader
    import shapely

    lat = np.asarray(lat_key, dtype=float)
    lon = np.asarray(lon_key, dtype=float)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mask = np.zeros(lon_grid.shape, dtype=bool)
    domain_bounds = (float(np.nanmin(lon)), float(np.nanmin(lat)), float(np.nanmax(lon)), float(np.nanmax(lat)))

    land_path = shapereader.natural_earth(resolution="10m", category="physical", name="land")
    for geometry in shapereader.Reader(land_path).geometries():
        if not _bounds_overlap(geometry.bounds, domain_bounds):
            continue
        contains_xy = getattr(shapely, "contains_xy", None)
        if contains_xy is not None:
            mask |= contains_xy(geometry, lon_grid, lat_grid)
        else:
            from shapely import vectorized

            mask |= vectorized.contains(geometry, lon_grid, lat_grid)
    return mask


def _bounds_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    a_min_lon, a_min_lat, a_max_lon, a_max_lat = a
    b_min_lon, b_min_lat, b_max_lon, b_max_lat = b
    return not (
        a_max_lon < b_min_lon
        or a_min_lon > b_max_lon
        or a_max_lat < b_min_lat
        or a_min_lat > b_max_lat
    )


def _apply_transport_streamfunction_regional_gauge(
    *,
    streamfunction_sv: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    wet_mask: np.ndarray,
    regional_gauge: Optional[str],
) -> Tuple[np.ndarray, Dict]:
    mode = _normalize_transport_regional_gauge(regional_gauge)
    if mode is None:
        return streamfunction_sv, {}
    if mode != "gan_fig10_china_seas":
        raise ValueError(f"Unsupported regional_gauge: {regional_gauge}")

    report_phase(
        phase="applying_regional_gauge",
        message="Evaluating Fig10 China Seas regional gauge",
        percent=0.84,
    )
    if not _domain_spans_gan_fig10_wpo_and_china_seas(lat=lat, lon=lon):
        report_phase(
            phase="applying_regional_gauge",
            message="Regional gauge skipped for small/non-WPO domain",
            percent=0.9,
        )
        return streamfunction_sv, {
            "regional_gauge": mode,
            "regional_gauge_applied": False,
            "regional_gauge_reason": (
                "domain does not span both the western Pacific and China Seas; "
                "kept the default southeast valid wet-point zero gauge"
            ),
        }

    regions = _gan_fig10_display_area_masks(lat=lat, lon=lon, wet_mask=wet_mask)
    rendering_regions: list[Dict] = []
    summaries: list[Dict] = []
    for region in regions:
        region_name = str(region["label"])
        region_mask = np.asarray(region["mask"], dtype=bool)
        valid = wet_mask & region_mask & np.isfinite(streamfunction_sv)
        if int(np.count_nonzero(valid)) < 1:
            continue
        rendering_regions.append({
            **region,
            "mask": valid,
        })
        summaries.append(_regional_display_summary(region_name, streamfunction_sv, valid))

    if not summaries:
        report_phase(
            phase="applying_regional_gauge",
            message="Regional gauge skipped because no valid China Seas mask intersected the domain",
            percent=0.92,
        )
        return streamfunction_sv, {
            "regional_gauge": mode,
            "regional_gauge_applied": False,
            "regional_gauge_reason": "no valid China Seas regional mask intersected the selected domain",
        }

    area1_region = next((region for region in rendering_regions if region.get("id") == "area1"), None)
    if area1_region is None:
        report_phase(
            phase="applying_regional_gauge",
            message="Regional gauge skipped because no valid Area 1 reference region intersected the domain",
            percent=0.92,
        )
        return streamfunction_sv, {
            "regional_gauge": mode,
            "regional_gauge_applied": False,
            "regional_gauge_reason": "no valid Area 1 reference region intersected the selected domain",
        }

    reference_mask = _gan_fig10_area1_northwest_reference_mask(
        lat=lat,
        lon=lon,
        wet_mask=wet_mask,
        area1_mask=np.asarray(area1_region["mask"], dtype=bool),
    )
    reference_valid = reference_mask & np.isfinite(streamfunction_sv)
    if not np.any(reference_valid):
        reference_valid = _northwest_valid_reference_mask(
            streamfunction_sv=streamfunction_sv,
            lat=lat,
            lon=lon,
            mask=np.asarray(area1_region["mask"], dtype=bool),
        )
    if not np.any(reference_valid):
        report_phase(
            phase="applying_regional_gauge",
            message="Regional gauge skipped because no Area 1 northwest reference wet point was available",
            percent=0.92,
        )
        return streamfunction_sv, {
            "regional_gauge": mode,
            "regional_gauge_applied": False,
            "regional_gauge_reason": "no valid Area 1 northwest reference wet point intersected the selected domain",
        }

    reference_offset = float(np.nanmedian(streamfunction_sv[reference_valid]))
    display_values = np.where(np.isfinite(streamfunction_sv), streamfunction_sv - reference_offset, np.nan)
    rendering_regions = [
        {
            **region,
            "mask": np.asarray(region["mask"], dtype=bool) & np.isfinite(display_values),
        }
        for region in rendering_regions
    ]

    report_phase(
        phase="applying_regional_gauge",
        message="Applied Area 1 northwest wet-point zero reference to global streamfunction",
        percent=0.92,
    )
    return display_values, {
        "regional_gauge": mode,
        "regional_gauge_applied": True,
        "regional_gauge_scope": "global_streamfunction_area1_northwest_reference_zero",
        "regional_gauge_regions": summaries,
        "regional_gauge_reference": _reference_gauge_summary(
            name="Area 1 northwest wet-point reference",
            streamfunction_sv=streamfunction_sv,
            reference_valid=reference_valid,
            offset_sv=reference_offset,
        ),
        "regional_color_scales": _gan_fig10_regional_color_scales(
            display_values=display_values,
            regions=rendering_regions,
        ),
        "transport_rendering": _gan_fig10_transport_rendering_metadata(
            regions=rendering_regions,
        ),
        "gauge": "area1_northwest_reference_zero",
        "base_gauge": "southeast_valid_wet_point_zero_per_connected_component",
        "integration_origin": "area1_northwest_reference_zero",
        "display_note": (
            "Gan Fig10-style diagnostic display: the streamfunction is solved once over "
            "the full selected wet domain, then shifted by one constant so the northwest "
            "valid wet patch in Area 1 is zero; Area 1 and Area 2 are rendering masks only."
        ),
    }


def _gan_fig10_regional_color_scales(
    *,
    display_values: np.ndarray,
    regions: list[Dict],
) -> list[Dict]:
    scales: list[Dict] = []
    for region in regions:
        region_mask = np.asarray(region.get("mask"), dtype=bool)
        region_values = display_values[region_mask & np.isfinite(display_values)]
        if not region_values.size:
            continue
        region_limit = float(np.nanmax(np.abs(region_values)))
        scales.append({
            "label": region.get("label", region.get("id", "Area")),
            "units": "Sv",
            "min": -region_limit,
            "max": region_limit,
            "rawMin": float(np.nanmin(region_values)),
            "rawMax": float(np.nanmax(region_values)),
            "colormap": region.get("colormap", "ocean_diverging"),
            "symmetric": True,
            "scaleStrategy": region.get("scaleStrategy", region.get("id", "gan_fig10_area")),
            "renderMode": "filled",
        })
    return scales


def _gan_fig10_transport_rendering_metadata(
    *,
    regions: list[Dict],
) -> Dict:
    first_mask = np.asarray(regions[0]["mask"], dtype=bool) if regions else None
    return {
        "mode": "gan_fig10_china_seas",
        "filled_region": "area1",
        "filled_mask": first_mask,
        "filled_colormap": "ocean_diverging",
        "filled_regions": [
            {
                "id": region.get("id"),
                "region": region.get("id"),
                "label": region.get("label"),
                "scaleStrategy": region.get("scaleStrategy"),
                "mask": region.get("mask"),
            }
            for region in regions
        ],
    }


def _gan_fig10_display_area_masks(*, lat: np.ndarray, lon: np.ndarray, wet_mask: np.ndarray) -> list[Dict]:
    area1_mask = _gan_fig10_area1_display_mask(lat=lat, lon=lon, wet_mask=wet_mask)
    area2_mask = _gan_fig10_area2_display_mask(
        lat=lat,
        lon=lon,
        wet_mask=wet_mask,
        applied_mask=area1_mask,
    )
    return [
        {
            "id": "area1",
            "label": "Area 1",
            "scaleStrategy": "gan_fig10_area1",
            "colormap": "ocean_diverging",
            "mask": area1_mask,
        },
        {
            "id": "area2",
            "label": "Area 2",
            "scaleStrategy": "gan_fig10_area2",
            "colormap": "transport_blue_red",
            "mask": area2_mask,
        },
    ]


def _gan_fig10_area1_northwest_reference_mask(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    wet_mask: np.ndarray,
    area1_mask: np.ndarray,
) -> np.ndarray:
    """Reference the Fig10-style display to the upper-left valid wet patch in Area 1."""
    valid = np.asarray(wet_mask, dtype=bool) & np.asarray(area1_mask, dtype=bool)
    reference = np.zeros_like(valid, dtype=bool)
    if not np.any(valid):
        return reference

    lat_order = np.argsort(-np.asarray(lat, dtype=float))
    lon_order = np.argsort(np.asarray(lon, dtype=float))
    for i in lat_order:
        row_valid = valid[int(i), :]
        if not np.any(row_valid):
            continue
        for j in lon_order:
            if not row_valid[int(j)]:
                continue
            i0 = max(int(i) - 1, 0)
            i1 = min(int(i) + 2, valid.shape[0])
            j0 = max(int(j) - 1, 0)
            j1 = min(int(j) + 2, valid.shape[1])
            reference[i0:i1, j0:j1] = valid[i0:i1, j0:j1]
            return reference
    return reference


def _northwest_valid_reference_mask(
    *,
    streamfunction_sv: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(streamfunction_sv) & np.asarray(mask, dtype=bool)
    reference = np.zeros_like(valid, dtype=bool)
    if not np.any(valid):
        return reference
    lat_order = np.argsort(-np.asarray(lat, dtype=float))
    lon_order = np.argsort(np.asarray(lon, dtype=float))
    for i in lat_order:
        row_valid = valid[int(i), :]
        if not np.any(row_valid):
            continue
        for j in lon_order:
            if not row_valid[int(j)]:
                continue
            reference[int(i), int(j)] = True
            return reference
    return reference


def _nearest_valid_reference_mask(
    *,
    streamfunction_sv: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> np.ndarray:
    valid = np.isfinite(streamfunction_sv)
    reference = np.zeros_like(valid, dtype=bool)
    if not np.any(valid):
        return reference
    lon_grid, lat_grid = np.meshgrid(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    lat_scale = max(float(np.nanmax(lat) - np.nanmin(lat)), 1.0)
    lon_scale = max(float(np.nanmax(lon) - np.nanmin(lon)), 1.0)
    distance = ((lat_grid - float(target_lat)) / lat_scale) ** 2 + ((lon_grid - float(target_lon)) / lon_scale) ** 2
    distance = np.where(valid, distance, np.inf)
    flat_index = int(np.nanargmin(distance))
    reference[np.unravel_index(flat_index, valid.shape)] = True
    return reference


def _gan_fig10_area1_display_mask(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    wet_mask: np.ndarray,
) -> np.ndarray:
    """Mask the coastal side enclosed by the Fig10 transect boundaries.

    This is intentionally broader than the SCS basin polygon: Area 1 follows
    the coastal side of the published Fig10 transects and includes the South
    China Sea, East China Sea, Yellow Sea, and adjacent China Seas shelves.
    The Taiwan Strait transect is not used as a divider here.
    """
    lon_grid, lat_grid = np.meshgrid(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    boundary_lon = _gan_fig10_transect_boundary_lon(lat_grid)
    china_seas_window = (
        (lat_grid >= 0.0)
        & (lat_grid <= 42.5)
        & (lon_grid >= 99.0)
        & (lon_grid <= boundary_lon)
    )
    return wet_mask & china_seas_window


def _gan_fig10_area2_display_mask(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    wet_mask: np.ndarray,
    applied_mask: np.ndarray,
) -> np.ndarray:
    """Mask the open-ocean side outside the Fig10 transect boundaries."""
    lon_grid, lat_grid = np.meshgrid(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    boundary_lon = _gan_fig10_transect_boundary_lon(lat_grid)
    area2_window = (
        (lat_grid >= 0.0)
        & (lat_grid <= 42.5)
        & (lon_grid > boundary_lon)
    )
    return wet_mask & ~applied_mask & area2_window


def _gan_fig10_transect_boundary_lon(lat_grid: np.ndarray) -> np.ndarray:
    """Approximate the Fig10 published transect envelope as lon=f(lat)."""
    boundary_lats = np.array(
        [0.0, 3.0, 6.0, 10.0, 15.0, 20.0, 24.0, 28.0, 32.0, 36.0, 41.0, 50.0],
        dtype=float,
    )
    boundary_lons = np.array(
        [118.2, 119.0, 120.0, 121.3, 122.0, 121.8, 123.4, 125.2, 126.4, 127.4, 129.2, 132.0],
        dtype=float,
    )
    return np.interp(
        lat_grid,
        boundary_lats,
        boundary_lons,
        left=boundary_lons[0],
        right=boundary_lons[-1],
    )


def _symmetric_contour_levels(values: np.ndarray) -> list[float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return []
    limit = float(np.nanmax(np.abs(valid)))
    if not np.isfinite(limit) or limit <= 0.0:
        return [0.0]
    step = _nice_integer_contour_step(limit)
    max_level = int(math.ceil(limit / step) * step)
    return [float(level) for level in range(-max_level, max_level + step, step)]


def _nice_integer_contour_step(limit: float) -> int:
    if not np.isfinite(limit) or limit <= 6.0:
        return 1
    target_step = max(1.0, limit / 5.0)
    magnitude = 10 ** math.floor(math.log10(target_step))
    for multiplier in (1, 2, 5, 10):
        step = max(1, int(round(multiplier * magnitude)))
        if target_step <= step:
            return step
    return max(1, int(round(10 * magnitude)))


def _normalize_transport_regional_gauge(value: Optional[str]) -> Optional[str]:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "none", "global", "default"}:
        return None
    if normalized in {"gan_fig10", "gan_fig10_china_seas", "cmoms_gan_fig10", "cmoms_china_seas"}:
        return "gan_fig10_china_seas"
    return normalized


def _domain_spans_gan_fig10_wpo_and_china_seas(*, lat: np.ndarray, lon: np.ndarray) -> bool:
    if lat.size == 0 or lon.size == 0:
        return False
    lon_min = float(np.nanmin(lon))
    lon_max = float(np.nanmax(lon))
    lat_min = float(np.nanmin(lat))
    lat_max = float(np.nanmax(lat))
    covers_china_seas = lon_min <= 112.0 and lat_min <= 10.0 and lat_max >= 22.0
    covers_wpo = lon_max >= 130.0 and lat_min <= 15.0 and lat_max >= 30.0
    return covers_china_seas and covers_wpo


def _gan_fig10_china_seas_region_masks(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    wet_mask: np.ndarray,
) -> list[Tuple[str, np.ndarray]]:
    return [
        (region_name, region_mask & wet_mask)
        for region_name, region_mask in _gan_fig10_china_seas_polygon_masks(lat=lat, lon=lon)
    ]


def _china_seas_mask(*, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    masks = _gan_fig10_china_seas_polygon_masks(lat=lat, lon=lon)
    if not masks:
        lon_grid, _lat_grid = np.meshgrid(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        return np.zeros(lon_grid.shape, dtype=bool)
    combined = np.zeros_like(masks[0][1], dtype=bool)
    for _region_name, region_mask in masks:
        combined |= region_mask
    return combined


def _gan_fig10_china_seas_polygon_masks(*, lat: np.ndarray, lon: np.ndarray) -> list[Tuple[str, np.ndarray]]:
    lon_grid, lat_grid = np.meshgrid(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    polygons: list[Tuple[str, Tuple[Tuple[float, float], ...]]] = [
        (
            "gulf_of_tonkin",
            (
                (105.2, 17.0),
                (107.5, 17.4),
                (109.2, 19.8),
                (108.7, 22.2),
                (106.2, 22.6),
                (104.9, 20.8),
                (105.0, 18.5),
                (105.2, 17.0),
            ),
        ),
        (
            "south_china_sea",
            (
                (108.2, 4.3),
                (112.3, 4.2),
                (116.7, 5.8),
                (119.6, 8.8),
                (121.3, 12.8),
                (121.0, 18.4),
                (119.7, 21.9),
                (116.0, 22.6),
                (112.1, 21.1),
                (109.1, 17.5),
                (107.8, 12.6),
                (107.9, 7.4),
                (108.2, 4.3),
            ),
        ),
        (
            "east_china_sea",
            (
                (118.4, 22.0),
                (121.8, 22.8),
                (124.8, 25.2),
                (126.2, 29.4),
                (126.1, 32.2),
                (124.3, 33.0),
                (121.3, 31.5),
                (119.5, 28.6),
                (118.1, 25.1),
                (118.4, 22.0),
            ),
        ),
        (
            "yellow_sea",
            (
                (119.1, 31.1),
                (125.6, 31.2),
                (125.8, 36.8),
                (124.0, 39.4),
                (121.0, 39.5),
                (119.0, 37.3),
                (118.3, 33.6),
                (119.1, 31.1),
            ),
        ),
    ]
    masks: list[Tuple[str, np.ndarray]] = []
    for region_name, polygon in polygons:
        mask = _points_in_polygon(points, polygon).reshape(lon_grid.shape)
        masks.append((region_name, mask))
    return masks


def _south_china_sea_basin_mask(*, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lon_grid, lat_grid = np.meshgrid(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    polygon = _load_scs_basin_polygon()
    return _points_in_polygon(points, polygon).reshape(lon_grid.shape)


@lru_cache(maxsize=1)
def _load_scs_basin_polygon() -> Tuple[Tuple[float, float], ...]:
    candidates = [
        Path.cwd() / "CS_ocean_boundary_SCSbasin(1).mat",
        Path(__file__).resolve().parents[3] / "CS_ocean_boundary_SCSbasin(1).mat",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            from scipy.io import loadmat

            mat = loadmat(candidate, squeeze_me=True, struct_as_record=False)
            output = mat.get("output")
            lon_bound = np.asarray(getattr(output, "lon_bound"), dtype=float).ravel()
            lat_bound = np.asarray(getattr(output, "lat_bound"), dtype=float).ravel()
            if lon_bound.size >= 3 and lon_bound.size == lat_bound.size:
                return tuple((float(x), float(y)) for x, y in zip(lon_bound, lat_bound))
        except Exception:
            continue
    return _default_scs_basin_polygon()


def _default_scs_basin_polygon() -> Tuple[Tuple[float, float], ...]:
    return (
        (108.8, 4.2),
        (113.0, 4.1),
        (117.0, 5.4),
        (120.0, 8.2),
        (121.2, 12.0),
        (120.8, 16.5),
        (119.8, 20.6),
        (117.2, 22.0),
        (113.6, 20.5),
        (110.2, 17.2),
        (108.7, 13.4),
        (108.7, 8.5),
        (108.8, 4.2),
    )


def _points_in_polygon(points: np.ndarray, polygon: Tuple[Tuple[float, float], ...]) -> np.ndarray:
    poly = np.asarray(polygon, dtype=float)
    x = np.asarray(points[:, 0], dtype=float)
    y = np.asarray(points[:, 1], dtype=float)
    inside = np.zeros(x.shape, dtype=bool)
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        intersects = ((y1 > y) != (y0 > y)) & (
            x < (x0 - x1) * (y - y1) / ((y0 - y1) if abs(y0 - y1) > 1e-12 else 1e-12) + x1
        )
        inside ^= intersects
        x0, y0 = x1, y1
    return inside


def _regional_display_summary(
    region_name: str,
    streamfunction_sv: np.ndarray,
    valid: np.ndarray,
) -> Dict:
    values = streamfunction_sv[valid]
    return {
        "name": region_name,
        "n_valid": int(values.size),
        "global_mean_before_reference_shift": float(np.nanmean(values)),
        "global_min_before_reference_shift": float(np.nanmin(values)),
        "global_max_before_reference_shift": float(np.nanmax(values)),
    }


def _reference_gauge_summary(
    *,
    name: str,
    streamfunction_sv: np.ndarray,
    reference_valid: np.ndarray,
    offset_sv: float,
) -> Dict:
    reference_values = streamfunction_sv[reference_valid]
    return {
        "name": name,
        "n_valid": int(reference_values.size),
        "mean_before_shift_sv": float(np.nanmean(reference_values)),
        "offset_subtracted_sv": float(offset_sv),
        "mean_after_shift_sv": 0.0,
    }


def _build_coord_slice(values, coord_range: Tuple[float, float]) -> slice:
    start, end = coord_range
    values = np.asarray(values, dtype=float)
    ascending = len(values) < 2 or values[0] <= values[1]
    if ascending:
        return slice(min(start, end), max(start, end))
    return slice(max(start, end), min(start, end))


def _compute_series_statistics(values: np.ndarray) -> Dict:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"n_valid": 0}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "n_valid": int(valid.size),
    }


def _compute_field_statistics(values: np.ndarray) -> Dict:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n_valid": 0}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "n_valid": int(valid.size),
    }


def _format_layer_label(depth_range: Tuple[float, float]) -> str:
    upper, lower = depth_range
    return f"{upper:g} to {lower:g} m"
