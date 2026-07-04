"""
Process-oriented tracer diagnostics.
"""

from __future__ import annotations

from typing import Optional

import xarray as xr

from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.data_access.load import get_depth_dim
from .advanced import _require_horizontal_coords
from .compute import _calculate_dx, _calculate_dy


EARTH_RADIUS_M = 6_371_000.0
SECONDS_PER_DAY = 86_400.0


def compute_horizontal_advection(
    data: xr.DataArray,
    u_data: xr.DataArray,
    v_data: xr.DataArray,
) -> xr.DataArray:
    """
    Compute horizontal advective tendency on a daily tendency scale.

    The native transport term ``-(u * dC/dx + v * dC/dy)`` is in tracer-units
    per second. We rescale it to tracer-units per day so it is directly
    comparable to ``compute_local_tendency`` and downstream budget diagnostics.
    """
    data = materialize_partitioned_xarray(data)
    u_data = materialize_partitioned_xarray(u_data)
    v_data = materialize_partitioned_xarray(v_data)

    tracer, u_field, v_field = xr.align(data, u_data, v_data, join="inner")
    _require_horizontal_coords(tracer)

    dcdx = _differentiate_longitude(tracer)
    dcdy = _differentiate_latitude(tracer)
    advection_per_second = -(u_field * dcdx + v_field * dcdy)
    advection = advection_per_second * SECONDS_PER_DAY
    advection.name = f"{tracer.name or 'field'}_horizontal_advection"
    advection.attrs = {
        **tracer.attrs,
        "long_name": f"Horizontal advection of {tracer.name or 'field'}",
        "units": _combine_units(tracer.attrs.get("units", ""), "day"),
        "aggregation": "horizontal_advection",
        "formula": "86400 * -(u*dC/dx + v*dC/dy)",
        "native_formula": "-(u*dC/dx + v*dC/dy)",
        "native_units": _combine_units(tracer.attrs.get("units", ""), "s"),
        "time_unit": "day",
        "native_time_unit": "s",
        "scaling_seconds_per_day": float(SECONDS_PER_DAY),
    }
    return advection


def compute_vertical_advection(
    data: xr.DataArray,
    w_data: xr.DataArray,
) -> xr.DataArray:
    """
    Compute vertical advective tendency on a daily tendency scale.
    """
    data = materialize_partitioned_xarray(data)
    w_data = materialize_partitioned_xarray(w_data)

    tracer, w_field = xr.align(data, w_data, join="inner")
    depth_dim = get_depth_dim(tracer)
    if depth_dim is None:
        raise ValueError("compute_vertical_advection requires a depth dimension")

    dcdz = tracer.differentiate(depth_dim)
    advection_per_second = -(w_field * dcdz)
    advection = advection_per_second * SECONDS_PER_DAY
    advection.name = f"{tracer.name or 'field'}_vertical_advection"
    advection.attrs = {
        **tracer.attrs,
        "long_name": f"Vertical advection of {tracer.name or 'field'}",
        "units": _combine_units(tracer.attrs.get("units", ""), "day"),
        "aggregation": "vertical_advection",
        "formula": "86400 * -(w*dC/dz)",
        "native_formula": "-(w*dC/dz)",
        "native_units": _combine_units(tracer.attrs.get("units", ""), "s"),
        "depth_coordinate": depth_dim,
        "time_unit": "day",
        "native_time_unit": "s",
        "scaling_seconds_per_day": float(SECONDS_PER_DAY),
    }
    return advection


def _differentiate_longitude(data: xr.DataArray) -> xr.DataArray:
    lon_metric = _calculate_dx(data.lon, data.lat)
    return data.differentiate("lon") / lon_metric


def _differentiate_latitude(data: xr.DataArray) -> xr.DataArray:
    return data.differentiate("lat") / _calculate_dy(data.lat)


def _combine_units(field_units: Optional[str], time_unit: str) -> str:
    field_units = (field_units or "").strip()
    if not field_units:
        return f"per_{time_unit}"
    return f"{field_units} / {time_unit}"
