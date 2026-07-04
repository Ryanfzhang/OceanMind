"""
Water-mass and isopycnal diagnostics.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import numpy as np
import xarray as xr

from domain.ocean.analysis.timeseries.extract import compute_layer_mean
from domain.ocean.data_access.load import get_depth_dim
from domain.ocean.data_access.partitioned import materialize_partitioned_xarray


def compute_ts_diagram(
    temp: xr.DataArray,
    salt: xr.DataArray,
    color_field: Optional[xr.DataArray] = None,
    max_points: int = 20000,
    sampling: Literal["random", "head"] = "random",
) -> Dict:
    """
    Flatten temperature and salinity fields into a T-S diagram result.
    """
    temp = materialize_partitioned_xarray(temp)
    salt = materialize_partitioned_xarray(salt)
    color_field = materialize_partitioned_xarray(color_field)

    aligned = [temp, salt]
    if color_field is not None:
        aligned.append(color_field)
    aligned_arrays = xr.align(*aligned, join="inner")
    temp_arr, salt_arr = aligned_arrays[0], aligned_arrays[1]
    color_arr = aligned_arrays[2] if color_field is not None else None

    temp_values = np.asarray(temp_arr.values, dtype=float).reshape(-1)
    salt_values = np.asarray(salt_arr.values, dtype=float).reshape(-1)
    valid_mask = np.isfinite(temp_values) & np.isfinite(salt_values)

    color_values = None
    color_variable = None
    if color_arr is not None:
        flat_color = np.asarray(color_arr.values, dtype=float).reshape(-1)
        valid_mask &= np.isfinite(flat_color)
        color_values = flat_color
        color_variable = color_arr.name or "color_field"

    indices = np.where(valid_mask)[0]
    if indices.size == 0:
        raise ValueError("No valid temperature-salinity pairs are available")

    if indices.size > max_points:
        if sampling == "random":
            rng = np.random.default_rng(0)
            indices = np.sort(rng.choice(indices, size=max_points, replace=False))
        elif sampling == "head":
            indices = indices[:max_points]
        else:
            raise ValueError(f"Unsupported sampling method: {sampling}")

    sampled_temp = temp_values[indices]
    sampled_salt = salt_values[indices]

    result = {
        "temperature": sampled_temp.tolist(),
        "salinity": sampled_salt.tolist(),
        "metadata": {
            "variable": "ts_diagram",
            "temperature_variable": temp_arr.name or "temp",
            "salinity_variable": salt_arr.name or "salt",
            "n_total_points": int(np.sum(valid_mask)),
            "n_sampled_points": int(indices.size),
            "sampling": sampling,
            "temperature_range": [float(np.min(sampled_temp)), float(np.max(sampled_temp))],
            "salinity_range": [float(np.min(sampled_salt)), float(np.max(sampled_salt))],
            "sigma0_contours": {
                "levels": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                "temperature_range": [float(np.min(sampled_temp)), float(np.max(sampled_temp))],
                "salinity_range": [float(np.min(sampled_salt)), float(np.max(sampled_salt))],
                "note": "Contour levels are provided as plotting metadata; the result does not embed a sigma0 grid.",
            },
        },
    }

    if color_values is not None:
        sampled_color = color_values[indices]
        result["color_values"] = sampled_color.tolist()
        result["metadata"]["color_variable"] = color_variable
        result["metadata"]["color_range"] = [float(np.min(sampled_color)), float(np.max(sampled_color))]

    return result


def extract_isopycnal_surface(
    density: xr.DataArray,
    target_sigma0: float,
) -> xr.DataArray:
    """
    Interpolate the depth of a target sigma0 surface from a density field.
    """
    density = materialize_partitioned_xarray(density)

    depth_dim = get_depth_dim(density)
    if depth_dim is None:
        raise ValueError("extract_isopycnal_surface requires a depth dimension")

    sigma0 = density - 1000.0
    depth_values = np.asarray(density[depth_dim].values, dtype=float)

    def interpolate_depth(profile: np.ndarray) -> float:
        valid = np.isfinite(profile) & np.isfinite(depth_values)
        sigma_profile = profile[valid]
        depth_profile = depth_values[valid]
        if sigma_profile.size < 2:
            return np.nan

        for index in range(sigma_profile.size - 1):
            s0 = sigma_profile[index]
            s1 = sigma_profile[index + 1]
            if s0 == s1:
                if s0 == target_sigma0:
                    return float(depth_profile[index])
                continue
            if min(s0, s1) <= target_sigma0 <= max(s0, s1):
                frac = (target_sigma0 - s0) / (s1 - s0)
                return float(depth_profile[index] + frac * (depth_profile[index + 1] - depth_profile[index]))
        return np.nan

    surface = xr.apply_ufunc(
        interpolate_depth,
        sigma0,
        input_core_dims=[[depth_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    surface.name = f"sigma0_{str(target_sigma0).replace('.', '_')}_surface_depth"
    surface.attrs = {
        "long_name": f"Depth of sigma0={target_sigma0}",
        "units": density[depth_dim].attrs.get("units", "m"),
        "feature": "isopycnal_surface",
        "method": "linear_interpolation",
        "target_sigma0": float(target_sigma0),
    }
    return surface


def compute_isopycnal_layer_mean(
    data: xr.DataArray,
    density: xr.DataArray,
    sigma0_upper: float,
    sigma0_lower: float,
) -> xr.DataArray:
    """
    Compute a mean field between two sigma0 surfaces.
    """
    data = materialize_partitioned_xarray(data)
    density = materialize_partitioned_xarray(density)

    upper_surface = extract_isopycnal_surface(density, sigma0_upper)
    lower_surface = extract_isopycnal_surface(density, sigma0_lower)
    layer_mean = compute_layer_mean(
        data=data,
        upper_bound_field=upper_surface,
        lower_bound_field=lower_surface,
    )
    layer_mean.name = f"{data.name or 'field'}_sigma0_layer_mean"
    layer_mean.attrs = {
        **layer_mean.attrs,
        "aggregation": "isopycnal_layer_mean",
        "sigma0_upper": float(sigma0_upper),
        "sigma0_lower": float(sigma0_lower),
    }
    return layer_mean

