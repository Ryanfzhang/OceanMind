"""
Advanced ocean diagnostics.

This module provides diagnostic quantities derived from velocity and
stratification fields and is designed to work directly with orchestrated
`xr.DataArray` inputs.
"""

from typing import Any, Optional, Tuple

import numpy as np
import xarray as xr

from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.data_access.load import get_depth_dim
from .compute import _calculate_dx, _calculate_dy


EARTH_ROTATION = 7.292e-5  # rad/s
GRAVITY = 9.81  # m/s^2
RHO0 = 1025.0  # kg/m^3


def compute_kinetic_energy(
    u_data: Optional[xr.DataArray] = None,
    v_data: Optional[xr.DataArray] = None,
    data: Optional[xr.Dataset] = None,
    dataset: Optional[xr.Dataset] = None,
) -> xr.DataArray:
    """
    Compute kinetic energy: KE = 0.5 * (u^2 + v^2).

    Args:
        u_data: Eastward velocity component.
        v_data: Northward velocity component.
        data: Optional Dataset containing `u` and `v`.
        dataset: Optional alias for `data`.

    Returns:
        Kinetic energy field.
    """
    u, v = _resolve_velocity_inputs(u_data, v_data, data, dataset)
    ke = 0.5 * (u ** 2 + v ** 2)
    ke.name = "kinetic_energy"
    ke.attrs = {
        "long_name": "Kinetic Energy",
        "units": "m^2 s^-2",
    }
    return ke


def compute_eddy_kinetic_energy(
    u_data: Optional[xr.DataArray] = None,
    v_data: Optional[xr.DataArray] = None,
    data: Optional[xr.Dataset] = None,
    dataset: Optional[xr.Dataset] = None,
) -> xr.DataArray:
    """
    Compute eddy kinetic energy: EKE = 0.5 * (u'^2 + v'^2).

    The eddy components are defined relative to the time mean, so the input must
    retain a time dimension.
    """
    u, v = _resolve_velocity_inputs(u_data, v_data, data, dataset)
    if "time" not in u.dims:
        raise ValueError("compute_eddy_kinetic_energy requires a time dimension")

    u_prime = u - u.mean(dim="time", skipna=True)
    v_prime = v - v.mean(dim="time", skipna=True)
    eke = 0.5 * (u_prime ** 2 + v_prime ** 2)
    eke.name = "eddy_kinetic_energy"
    eke.attrs = {
        "long_name": "Eddy Kinetic Energy",
        "units": "m^2 s^-2",
        "mean_reference": "time_mean",
    }
    return eke


def compute_vertical_shear(
    u_data: Optional[xr.DataArray] = None,
    v_data: Optional[xr.DataArray] = None,
    data: Optional[xr.Dataset] = None,
    dataset: Optional[xr.Dataset] = None,
) -> xr.DataArray:
    """
    Compute vertical shear magnitude: S = sqrt((du/dz)^2 + (dv/dz)^2).

    Args:
        u_data: Eastward velocity component.
        v_data: Northward velocity component.
        data: Optional Dataset containing `u` and `v`.
        dataset: Optional alias for `data`.

    Returns:
        Vertical shear magnitude.
    """
    u, v = _resolve_velocity_inputs(u_data, v_data, data, dataset)
    depth_dim = get_depth_dim(u)
    if depth_dim is None:
        raise ValueError("compute_vertical_shear requires a depth dimension")

    du_dz = u.differentiate(depth_dim)
    dv_dz = v.differentiate(depth_dim)
    shear = np.sqrt(du_dz ** 2 + dv_dz ** 2)
    shear.name = "vertical_shear"
    shear.attrs = {
        "long_name": "Vertical Shear Magnitude",
        "units": "s^-1",
    }
    return shear


def compute_strain_rate(
    u_data: Optional[xr.DataArray] = None,
    v_data: Optional[xr.DataArray] = None,
    data: Optional[xr.Dataset] = None,
    dataset: Optional[xr.Dataset] = None,
) -> xr.DataArray:
    """
    Compute total horizontal strain rate magnitude.

    alpha = sqrt(Sn^2 + Ss^2)
    Sn = du/dx - dv/dy
    Ss = dv/dx + du/dy
    """
    u, v = _resolve_velocity_inputs(u_data, v_data, data, dataset)
    _require_horizontal_coords(u)
    dx, dy = _get_horizontal_spacing(u)

    du_dx = u.differentiate("lon") / dx
    du_dy = u.differentiate("lat") / dy
    dv_dx = v.differentiate("lon") / dx
    dv_dy = v.differentiate("lat") / dy

    normal_strain = du_dx - dv_dy
    shear_strain = dv_dx + du_dy
    strain = np.sqrt(normal_strain ** 2 + shear_strain ** 2)
    strain.name = "strain_rate"
    strain.attrs = {
        "long_name": "Horizontal Strain Rate",
        "units": "s^-1",
    }
    return strain


def compute_rossby_number(
    u_data: Optional[xr.DataArray] = None,
    v_data: Optional[xr.DataArray] = None,
    data: Optional[xr.Dataset] = None,
    dataset: Optional[xr.Dataset] = None,
) -> xr.DataArray:
    """
    Compute Rossby number: Ro = zeta / f.

    Args:
        u_data: Eastward velocity component.
        v_data: Northward velocity component.
        data: Optional Dataset containing `u` and `v`.
        dataset: Optional alias for `data`.

    Returns:
        Rossby number field.
    """
    u, v = _resolve_velocity_inputs(u_data, v_data, data, dataset)
    vorticity = _compute_vorticity_from_uv(u, v)
    coriolis = _build_coriolis_parameter(u)
    rossby = xr.where(np.abs(coriolis) > 1e-12, vorticity / coriolis, np.nan)
    rossby.name = "rossby_number"
    rossby.attrs = {
        "long_name": "Rossby Number",
        "units": "dimensionless",
    }
    return rossby


def compute_divergence(
    u_data: Optional[xr.DataArray] = None,
    v_data: Optional[xr.DataArray] = None,
    data: Optional[xr.Dataset] = None,
    dataset: Optional[xr.Dataset] = None,
) -> xr.DataArray:
    """
    Compute horizontal divergence: div = du/dx + dv/dy.

    Args:
        u_data: Eastward velocity component.
        v_data: Northward velocity component.
        data: Optional Dataset containing `u` and `v`.
        dataset: Optional alias for `data`.

    Returns:
        Horizontal divergence field.
    """
    u, v = _resolve_velocity_inputs(u_data, v_data, data, dataset)
    _require_horizontal_coords(u)
    dx, dy = _get_horizontal_spacing(u)

    du_dx = u.differentiate("lon") / dx
    dv_dy = v.differentiate("lat") / dy
    divergence = du_dx + dv_dy
    divergence.name = "divergence"
    divergence.attrs = {
        "long_name": "Horizontal Divergence",
        "units": "s^-1",
    }
    return divergence


def compute_richardson_number(
    n2_data: xr.DataArray,
    u_data: xr.DataArray,
    v_data: xr.DataArray,
) -> xr.DataArray:
    """
    Compute gradient Richardson number: Ri = N^2 / S^2.

    Args:
        n2_data: Buoyancy frequency squared field.
        u_data: Eastward velocity component.
        v_data: Northward velocity component.

    Returns:
        Richardson number field.
    """
    n2_data = materialize_partitioned_xarray(n2_data)
    u_data = materialize_partitioned_xarray(u_data)
    v_data = materialize_partitioned_xarray(v_data)

    shear = compute_vertical_shear(u_data, v_data)
    n2, shear = xr.align(n2_data, shear, join="inner")
    shear_sq = shear ** 2
    ri = xr.where(shear_sq > 1e-16, n2 / shear_sq, np.nan)
    ri.name = "richardson_number"
    ri.attrs = {
        "long_name": "Gradient Richardson Number",
        "units": "dimensionless",
    }
    return ri


def _resolve_velocity_inputs(
    u_data: Optional[xr.DataArray],
    v_data: Optional[xr.DataArray],
    data: Optional[Any],
    dataset: Optional[Any],
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Accept either explicit u/v fields or a Dataset containing u and v."""
    u_data = materialize_partitioned_xarray(u_data)
    v_data = materialize_partitioned_xarray(v_data)
    data = materialize_partitioned_xarray(data)
    dataset = materialize_partitioned_xarray(dataset)

    if u_data is not None and v_data is not None:
        return _align_velocity_fields(u_data, v_data)

    source = _unwrap_velocity_source(dataset if dataset is not None else data)
    if source is None:
        raise TypeError(
            "Velocity diagnostics require either 'u_data' and 'v_data', or a "
            "'dataset'/'data' Dataset containing both variables."
        )
    if not isinstance(source, xr.Dataset):
        raise TypeError(
            "Velocity diagnostics expect 'dataset'/'data' to be an xarray Dataset "
            "containing 'u' and 'v'."
        )
    if "u" not in source.data_vars or "v" not in source.data_vars:
        raise ValueError("Velocity dataset must contain both 'u' and 'v' variables.")
    return _align_velocity_fields(source["u"], source["v"])


def _unwrap_velocity_source(source: Optional[Any]) -> Optional[Any]:
    """Accept common orchestrator-normalized wrappers around xarray objects."""
    if source is None:
        return None

    if isinstance(source, dict):
        if "data" in source:
            return _unwrap_velocity_source(source["data"])
        if "u" in source and "v" in source:
            u_data = source["u"]
            v_data = source["v"]
            if isinstance(u_data, xr.DataArray) and isinstance(v_data, xr.DataArray):
                return xr.Dataset({"u": u_data, "v": v_data})

    return source


def _align_velocity_fields(u_data: xr.DataArray, v_data: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """Align two velocity fields on a shared grid."""
    u, v = xr.align(u_data, v_data, join="inner")
    _require_horizontal_coords(u)
    return u, v


def _require_horizontal_coords(data: xr.DataArray) -> None:
    """Ensure the field contains the required horizontal coordinates."""
    missing = [name for name in ("lon", "lat") if name not in data.coords]
    if missing:
        raise ValueError(f"Missing required horizontal coordinates: {missing}")


def _get_horizontal_spacing(data: xr.DataArray) -> Tuple[Any, float]:
    """Return meters per native horizontal coordinate unit."""
    dx = _calculate_dx(data.lon, data.lat)
    dy = _calculate_dy(data.lat)
    return dx, dy


def _compute_vorticity_from_uv(u_data: xr.DataArray, v_data: xr.DataArray) -> xr.DataArray:
    """Compute relative vorticity from aligned velocity fields."""
    dx, dy = _get_horizontal_spacing(u_data)
    dvdx = v_data.differentiate("lon") / dx
    dudy = u_data.differentiate("lat") / dy
    return dvdx - dudy


def _build_coriolis_parameter(data: xr.DataArray) -> xr.DataArray:
    """Create a Coriolis parameter field broadcast across the input grid."""
    lat = xr.DataArray(
        np.asarray(data.lat.values, dtype=float),
        coords={"lat": data.lat.values},
        dims=("lat",),
    )
    f = 2.0 * EARTH_ROTATION * np.sin(np.deg2rad(lat))
    return f.broadcast_like(data)
