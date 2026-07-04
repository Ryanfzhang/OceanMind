"""
Interpretation-oriented ocean diagnostics built on top of the existing toolset.

These helpers stay within the currently available variables
(`temp`, `salt`, `u`, `v`, `oxygen`, `chla`) and focus on mechanism ranking,
partial-budget attribution, proxy counterfactuals, evidence grading, and
environment-health assessment.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from domain.ocean.analysis.field_tools import compute_field_climatology, compute_local_tendency
from domain.ocean.analysis.profile.extract import identify_mixed_layer_depth, identify_thermocline_depth
from domain.ocean.analysis.spatial.analysis import compute_spatial_field
from domain.ocean.analysis.timeseries.advanced import (
    _lag_candidates,
    _select_optimal_lag_index,
    compute_lag_correlation,
)
from domain.ocean.analysis.timeseries.extract import compute_layer_mean
from domain.ocean.analysis.weighted_means import (
    _aggregate_depth,
    _horizontal_cell_area,
    _normalize_to_timeseries_field,
    _subset_horizontal,
    compute_area_weighted_mean,
    compute_volume_weighted_mean,
)
from domain.ocean.data_access.load import get_depth_dim, _normalize_depth_range
from domain.ocean.data_access.partitioned import find_partitioned_values, materialize_partitioned_xarray
from domain.ocean.dask_utils import chunk_summary, is_dask_backed, report_phase
from domain.ocean.diagnostics.advanced import (
    compute_eddy_kinetic_energy,
    compute_kinetic_energy,
)
from domain.ocean.diagnostics.compute import _calculate_dx, _calculate_dy, compute_density, compute_derived_field
from domain.ocean.diagnostics.process import compute_horizontal_advection
from domain.ocean.preprocessing.filter import filter_data


CLAIM_SUPPORTED = "supported by available variables"
CLAIM_LIMITED = "consistent with but not identified"
CLAIM_UNTESTABLE = "not testable with current variables"

ENV_SUPPORT_SUPPORTED = "supported"
ENV_SUPPORT_LIMITED = "limited"
ENV_SUPPORT_UNTESTABLE = "untestable"

ENV_DIRECTION_DETERIORATING = "deteriorating"
ENV_DIRECTION_IMPROVING = "improving"
ENV_DIRECTION_STABLE = "stable"
ENV_DIRECTION_UNTESTABLE = "untestable"

ENV_ROLE_PRIMARY_ENDPOINT = "primary_endpoint"
ENV_ROLE_PRIMARY_SUPPORT = "primary_support"
ENV_ROLE_RISK_FACTOR = "risk_factor"
ENV_ROLE_AUXILIARY_CONTEXT = "auxiliary_context"

ENV_EVIDENCE_TREND = "trend"
ENV_EVIDENCE_EVENT_STATISTICS = "event_statistics"
ENV_EVIDENCE_EVENT_DETECTION = "event_detection"
ENV_EVIDENCE_SPATIAL_FIELD = "spatial_field"
ENV_EVIDENCE_EVENT_SPATIAL_FIELD = "event_spatial_field"
ENV_EVIDENCE_EVENT_SPATIAL_DISTRIBUTION = "event_spatial_distribution"

ENV_SPATIAL_METRICS = {"mean", "max", "total", "positive_fraction", "nonzero_fraction"}


def compute_stratification_index(
    data: Optional[xr.Dataset] = None,
    temp: Optional[xr.DataArray] = None,
    salt: Optional[xr.DataArray] = None,
    density: Optional[xr.DataArray] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    surface_depth: float = 0.0,
    bottom_depth: Optional[float] = None,
    method: str = "surface_bottom_density_difference",
) -> xr.DataArray:
    """
    Compute a simple stratification index from density structure.

    The default index is the density difference between the surface and a lower
    level (or the deepest available level within `depth_range`).
    """
    density_field = _resolve_density_field(data=data, temp=temp, salt=salt, density=density)
    depth_dim = get_depth_dim(density_field)
    if depth_dim is None:
        raise ValueError("compute_stratification_index requires a depth dimension")

    field = density_field
    if depth_range is not None:
        depth_values = np.asarray(field[depth_dim].values, dtype=float)
        nr = _normalize_depth_range(depth_values, depth_range)
        field = field.sel({depth_dim: _build_coord_slice(field[depth_dim].values, nr)})
    if field.sizes.get(depth_dim, 0) < 2:
        raise ValueError("compute_stratification_index requires at least two depth levels")

    upper_index = _nearest_index(field[depth_dim].values, surface_depth)
    upper = field.isel({depth_dim: upper_index})
    upper_depth_value = float(field[depth_dim].values[upper_index])

    if bottom_depth is not None:
        lower_index = _nearest_index(field[depth_dim].values, bottom_depth)
        lower = field.isel({depth_dim: lower_index})
        lower_depth = xr.full_like(lower, float(field[depth_dim].values[lower_index]), dtype=float)
        bottom_depth_attr: Any = float(field[depth_dim].values[lower_index])
    else:
        # Use the deepest valid level for each profile instead of the globally deepest
        # model level. This keeps shelf-sea regions from becoming entirely NaN when the
        # loaded field includes abyssal levels that are invalid everywhere locally.
        lower, lower_depth = _deepest_valid_profile_value(field, depth_dim)
        bottom_depth_attr = "deepest_valid_level"

    dz = np.abs(lower_depth - upper_depth_value)
    if bottom_depth is not None and float(dz.values.flat[0]) <= 0:
        raise ValueError("surface and lower levels collapse to the same depth")

    if method == "surface_bottom_density_difference":
        index = lower - upper
        units = field.attrs.get("units", "kg/m^3")
    elif method == "density_gradient":
        index = xr.where(dz > 0, (lower - upper) / dz, np.nan)
        units = _combine_units(field.attrs.get("units", "kg/m^3"), "m^-1")
    else:
        raise ValueError(f"Unsupported method for compute_stratification_index: {method}")

    index.name = "stratification_index"
    index.attrs = {
        "long_name": "Stratification Index",
        "units": units,
        "aggregation": "stratification_index",
        "method": method,
        "surface_depth": upper_depth_value,
        "bottom_depth": bottom_depth_attr,
        "source_variable": density_field.name or "density",
    }
    if is_dask_backed(index):
        report_phase(
            phase="lazy_stratification_index_prepared",
            message="Prepared lazy stratification index",
            percent=0.15,
            compute_backend="dask",
            chunks=chunk_summary(index),
        )
    return index


def compute_brunt_vaisala_frequency(
    data: Optional[xr.Dataset] = None,
    temp: Optional[xr.DataArray] = None,
    salt: Optional[xr.DataArray] = None,
    density: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Compute buoyancy frequency from density."""
    density_field = _resolve_density_field(data=data, temp=temp, salt=salt, density=density)
    dataset = xr.Dataset({"density": density_field})
    n2 = compute_derived_field(data=dataset, field_type="buoyancy_frequency")
    n2.name = "brunt_vaisala_frequency"
    n2.attrs = {
        **n2.attrs,
        "long_name": "Brunt-Vaisala Frequency Squared",
        "aggregation": "buoyancy_frequency",
        "source_variable": density_field.name or "density",
    }
    return n2


def compute_density_gradient_profile(
    data: Optional[xr.Dataset] = None,
    temp: Optional[xr.DataArray] = None,
    salt: Optional[xr.DataArray] = None,
    density: Optional[xr.DataArray] = None,
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    method: str = "nearest",
) -> Dict[str, Any]:
    """Return a vertical profile of density gradient at a point or regional mean."""
    density_field = _resolve_density_field(data=data, temp=temp, salt=salt, density=density)
    depth_dim = get_depth_dim(density_field)
    if depth_dim is None:
        raise ValueError("compute_density_gradient_profile requires a depth dimension")

    gradient = density_field.differentiate(depth_dim)
    gradient.name = "density_gradient"
    gradient.attrs = {
        "long_name": "Vertical Density Gradient",
        "units": _combine_units(density_field.attrs.get("units", "kg/m^3"), "m^-1"),
        "aggregation": "density_gradient_profile",
        "source_variable": density_field.name or "density",
    }

    profile = gradient
    point: Optional[Dict[str, float]] = None
    requested_point: Optional[Dict[str, float]] = None
    if "time" in profile.dims:
        profile = profile.mean(dim="time", skipna=True)
    if lon is not None and lat is not None:
        if method == "nearest":
            profile = profile.sel(lon=lon, lat=lat, method="nearest")
        else:
            profile = profile.interp(lon=lon, lat=lat, method=method)
        point = {"lon": float(profile.lon.values), "lat": float(profile.lat.values)}
        requested_point = {"lon": float(lon), "lat": float(lat)}
    else:
        remaining = [dim for dim in ("lat", "lon") if dim in profile.dims]
        if remaining:
            profile = profile.mean(dim=remaining, skipna=True)

    depth = np.asarray(profile[depth_dim].values, dtype=float)
    values = np.asarray(profile.values, dtype=float)
    valid = np.isfinite(depth) & np.isfinite(values)
    depth = depth[valid]
    values = values[valid]
    if depth.size == 0:
        raise ValueError("No valid density-gradient profile values found")

    return {
        "depth": depth.tolist(),
        "values": values.tolist(),
        "variable": "density_gradient",
        "point": point,
        "requested_point": requested_point,
        "n_levels": int(depth.size),
        "depth_range": [float(np.min(depth)), float(np.max(depth))],
        "statistics": _simple_statistics(values),
        "metadata": {
            "units": gradient.attrs.get("units", ""),
            "source_variable": density_field.name or "density",
        },
    }


def compute_mld_thermocline_offset(
    mixed_layer_depth: xr.DataArray,
    thermocline_depth: xr.DataArray,
) -> xr.DataArray:
    """Compute absolute separation between the MLD and thermocline depth."""
    mixed_layer_depth = materialize_partitioned_xarray(mixed_layer_depth)
    thermocline_depth = materialize_partitioned_xarray(thermocline_depth)

    mld, thermo = xr.align(mixed_layer_depth, thermocline_depth, join="inner")
    offset = np.abs(thermo - mld)
    offset.name = "mld_thermocline_offset"
    offset.attrs = {
        "long_name": "MLD-Thermocline Separation",
        "units": "m",
        "aggregation": "mld_thermocline_offset",
        "feature": "layer_separation",
    }
    return offset


def compute_vertical_stability_timeseries(
    data: Optional[xr.Dataset] = None,
    temp: Optional[xr.DataArray] = None,
    salt: Optional[xr.DataArray] = None,
    density: Optional[xr.DataArray] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    weighting: str = "area_weighted",
) -> Dict[str, Any]:
    """Reduce the stratification index to a regional time series."""
    data, temp, salt, density = _subset_stability_inputs(
        data=data,
        temp=temp,
        salt=salt,
        density=density,
        lon_range=lon_range,
        lat_range=lat_range,
    )
    stratification = compute_stratification_index(
        data=data,
        temp=temp,
        salt=salt,
        density=density,
        depth_range=depth_range,
    )
    metadata = {
        "variable": "stratification_index",
        "units": stratification.attrs.get("units", ""),
        "source_variable": stratification.attrs.get("source_variable", "density"),
    }
    return _reduce_field_to_timeseries(
        field=stratification,
        lon_range=lon_range,
        lat_range=lat_range,
        depth_range=None,
        depth_aggregation="mean",
        weighting=weighting,
        metadata_updates=metadata,
    )


def compute_tracer_horizontal_advection_timeseries(
    data: xr.DataArray,
    u_data: xr.DataArray,
    v_data: xr.DataArray,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: str = "mean",
    weighting: str = "area_weighted",
) -> Dict[str, Any]:
    """Compute a regional timeseries of horizontal tracer advection."""
    advection = compute_horizontal_advection(data=data, u_data=u_data, v_data=v_data)
    return _reduce_field_to_timeseries(
        field=advection,
        lon_range=lon_range,
        lat_range=lat_range,
        depth_range=depth_range,
        depth_aggregation=depth_aggregation,
        weighting=weighting,
        metadata_updates={"aggregation": "horizontal_advection"},
    )


def compute_tracer_advection_map(
    data: xr.DataArray,
    u_data: xr.DataArray,
    v_data: xr.DataArray,
    time_range: Optional[Tuple[str, str]] = None,
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: str = "mean",
    time_aggregation: str = "mean",
) -> Dict[str, Any]:
    """Compute a spatial map of horizontal tracer advection."""
    advection = compute_horizontal_advection(data=data, u_data=u_data, v_data=v_data)
    result = compute_spatial_field(
        data=advection,
        time_range=time_range,
        time_aggregation=time_aggregation,
        depth_range=depth_range,
        depth_aggregation=depth_aggregation,
    )
    result.setdefault("metadata", {})
    result["metadata"].update({
        "variable": advection.name or data.name or "horizontal_advection",
        "aggregation": "horizontal_advection",
    })
    return result


def compute_partial_tracer_budget(
    data: xr.DataArray,
    u_data: xr.DataArray,
    v_data: xr.DataArray,
) -> xr.Dataset:
    """
    Build a compact partial budget dataset containing local tendency and
    horizontal advection for the tracer.
    """
    local_tendency = compute_local_tendency(data)
    horizontal_advection = compute_horizontal_advection(data=data, u_data=u_data, v_data=v_data)
    residual_proxy = local_tendency - horizontal_advection
    residual_proxy.name = f"{data.name or 'field'}_budget_residual_proxy"
    residual_proxy.attrs = {
        **local_tendency.attrs,
        "long_name": f"Residual proxy of {data.name or 'field'} budget",
        "aggregation": "budget_residual_proxy",
        "formula": "local_tendency - horizontal_advection",
    }
    return xr.Dataset(
        {
            "local_tendency": local_tendency,
            "horizontal_advection": horizontal_advection,
            "budget_residual_proxy": residual_proxy,
        }
    )


def compute_budget_residual(
    local_tendency: Dict[str, Any],
    horizontal_advection: Optional[Dict[str, Any]] = None,
    vertical_advection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the residual of reduced budget terms as a timeseries."""
    times, tendency_values = _extract_series(local_tendency)
    residual = tendency_values.copy()

    used_terms = []
    for name, term in (
        ("horizontal_advection", horizontal_advection),
        ("vertical_advection", vertical_advection),
    ):
        if term is None:
            continue
        _, term_values = _extract_series(term, target_times=times)
        residual = residual - term_values
        used_terms.append(name)

    metadata = {
        **_series_metadata(local_tendency),
        "variable": "budget_residual",
        "aggregation": "budget_residual",
        "used_terms": used_terms,
        "statistics": _simple_statistics(residual),
    }
    return {
        "times": [str(value) for value in times],
        "values": residual.tolist(),
        "metadata": metadata,
    }


def compare_budget_term_magnitudes(
    local_tendency: Dict[str, Any],
    horizontal_advection: Optional[Dict[str, Any]] = None,
    vertical_advection: Optional[Dict[str, Any]] = None,
    residual: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Rank the dominant budget term and attach evidence-strength grading."""
    times, tendency_values = _extract_series(local_tendency)
    candidate_scores: Dict[str, float] = {
        "local_tendency": _mean_abs(tendency_values),
    }
    supporting_evidence = [
        f"Mean absolute local tendency is {_format_number(candidate_scores['local_tendency'])}.",
    ]

    if horizontal_advection is not None:
        _, horizontal_values = _extract_series(horizontal_advection, target_times=times)
        candidate_scores["horizontal_advection"] = _mean_abs(horizontal_values)
        supporting_evidence.append(
            f"Mean absolute horizontal advection is {_format_number(candidate_scores['horizontal_advection'])}."
        )
    if vertical_advection is not None:
        _, vertical_values = _extract_series(vertical_advection, target_times=times)
        candidate_scores["vertical_advection"] = _mean_abs(vertical_values)
        supporting_evidence.append(
            f"Mean absolute vertical advection is {_format_number(candidate_scores['vertical_advection'])}."
        )
    if residual is not None:
        _, residual_values = _extract_series(residual, target_times=times)
        candidate_scores["residual"] = _mean_abs(residual_values)
        supporting_evidence.append(
            f"Mean absolute residual is {_format_number(candidate_scores['residual'])}."
        )

    top_name, top_score = max(candidate_scores.items(), key=lambda item: item[1])
    residual_share = 0.0
    if "residual" in candidate_scores:
        total = sum(value for key, value in candidate_scores.items() if key != "local_tendency")
        if total > 0:
            residual_share = candidate_scores["residual"] / total

    claim_strength = _claim_from_residual_share(residual_share if "residual" in candidate_scores else 0.15)
    conflicting = []
    if "residual" in candidate_scores and residual_share >= 0.5:
        conflicting.append("Residual remains comparable to or larger than resolved terms.")
    if len(candidate_scores) > 1:
        sorted_scores = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        if len(sorted_scores) > 1:
            ratio = sorted_scores[0][1] / max(sorted_scores[1][1], 1e-12)
            supporting_evidence.append(
                f"Top-to-second magnitude ratio is {_format_number(ratio)}."
            )
            if ratio < 1.2:
                conflicting.append("Top-ranked process is only marginally larger than the next candidate.")

    return _mechanism_score_result(
        candidate_scores=candidate_scores,
        top_mechanism=top_name,
        claim_strength=claim_strength,
        supporting_evidence=supporting_evidence,
        conflicting_evidence=conflicting,
        metadata={"comparison": "budget_term_magnitude"},
    )


def compute_front_proximity_index(
    data: xr.DataArray,
    percentile: float = 90.0,
) -> xr.DataArray:
    """
    Compute a normalized front-proximity proxy from horizontal tracer gradients.
    """
    gradient = compute_derived_field(field=data, variable=data.name or "field", field_type="horizontal_gradient")
    values = np.asarray(gradient.values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        scale = 1.0
    else:
        scale = float(np.nanpercentile(np.abs(finite), percentile))
        if scale <= 0:
            scale = float(np.nanmax(np.abs(finite))) or 1.0
    proximity = xr.where(np.isfinite(gradient), np.clip(np.abs(gradient) / scale, 0.0, 1.0), np.nan)
    proximity.name = "front_proximity_index"
    proximity.attrs = {
        "long_name": "Front Proximity Index",
        "units": "dimensionless",
        "aggregation": "front_proximity_index",
        "source_variable": data.name or "field",
        "percentile": float(percentile),
    }
    return proximity


def compute_eddy_influence_mask(
    u_data: xr.DataArray,
    v_data: xr.DataArray,
    percentile: float = 90.0,
) -> xr.DataArray:
    """
    Build an eddy-influence mask from eddy kinetic energy or kinetic energy.

    Non-ocean / invalid cells are preserved as NaN so downstream map rendering
    can leave land unshaded instead of drawing it as zero influence.
    """
    try:
        proxy = compute_eddy_kinetic_energy(u_data=u_data, v_data=v_data)
    except ValueError:
        proxy = compute_kinetic_energy(u_data=u_data, v_data=v_data)
    threshold = float(np.nanpercentile(np.abs(proxy.values), percentile))
    mask = xr.where(
        np.isfinite(proxy),
        (np.abs(proxy) >= threshold).astype(float),
        np.nan,
    )
    mask.name = "eddy_influence_mask"
    mask.attrs = {
        "long_name": "Eddy Influence Mask",
        "units": "dimensionless",
        "aggregation": "eddy_influence_mask",
        "threshold_percentile": float(percentile),
        "proxy": proxy.name or "eddy_kinetic_energy",
    }
    return mask


def compute_tracer_gradient_alignment(
    data: xr.DataArray,
    u_data: xr.DataArray,
    v_data: xr.DataArray,
) -> xr.DataArray:
    """
    Compute cosine alignment between tracer gradient and horizontal flow vector.
    """
    data = materialize_partitioned_xarray(data)
    u_data = materialize_partitioned_xarray(u_data)
    v_data = materialize_partitioned_xarray(v_data)

    tracer, u_field, v_field = xr.align(data, u_data, v_data, join="inner")
    dcdx = _differentiate_longitude(tracer)
    dcdy = _differentiate_latitude(tracer)
    speed = np.sqrt(u_field ** 2 + v_field ** 2)
    grad_mag = np.sqrt(dcdx ** 2 + dcdy ** 2)
    denom = speed * grad_mag
    alignment = xr.where(denom > 1e-12, (u_field * dcdx + v_field * dcdy) / denom, np.nan)
    alignment.name = "tracer_gradient_alignment"
    alignment.attrs = {
        "long_name": "Tracer Gradient Alignment",
        "units": "dimensionless",
        "aggregation": "tracer_gradient_alignment",
        "source_variable": tracer.name or "field",
    }
    return alignment


def compute_mesoscale_background_separation(
    data: xr.DataArray,
    cutoff_period: float = 2.0,
    component: str = "mesoscale",
) -> xr.DataArray:
    """Separate mesoscale/high-pass and background/low-pass spatial components."""
    if component == "mesoscale":
        separated = filter_data(
            data=data,
            filter_type="highpass",
            cutoff_period=cutoff_period,
            dimension="spatial",
            method="gaussian",
        )
    elif component == "background":
        separated = filter_data(
            data=data,
            filter_type="lowpass",
            cutoff_period=cutoff_period,
            dimension="spatial",
            method="gaussian",
        )
    else:
        raise ValueError("component must be either 'mesoscale' or 'background'")

    separated.name = f"{data.name or 'field'}_{component}"
    separated.attrs = {
        **data.attrs,
        "aggregation": "mesoscale_background_separation",
        "component": component,
        "cutoff_period": float(cutoff_period),
    }
    return separated


def compute_flow_structure_context(
    u_data: xr.DataArray,
    v_data: xr.DataArray,
) -> xr.DataArray:
    """Build a single background-flow context proxy from the velocity field."""
    flow_context = compute_kinetic_energy(u_data=u_data, v_data=v_data)
    flow_context.name = "flow_context"
    flow_context.attrs = {
        **flow_context.attrs,
        "long_name": "Background Flow Context Proxy",
        "aggregation": "flow_structure_context",
        "source_diagnostics": "kinetic_energy",
    }
    return flow_context


def compute_event_precursor_composite(
    field: xr.DataArray,
    events: List[Dict[str, Any]],
    lead_steps: int = 1,
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: str = "mean",
) -> Dict[str, Any]:
    """
    Composite a field at the time steps immediately preceding detected events.
    """
    field = materialize_partitioned_xarray(field)

    if "time" not in field.dims:
        raise ValueError("compute_event_precursor_composite requires a time dimension")

    precursor_indices = _event_time_indices(events, len(field["time"]))
    precursor_indices = sorted({index - lead_steps for index in precursor_indices if index - lead_steps >= 0})
    if not precursor_indices:
        raise ValueError("No valid precursor time steps could be inferred from events")

    subset = field.isel(time=precursor_indices)
    composite = compute_spatial_field(
        data=subset,
        time_aggregation="mean",
        depth_range=depth_range,
        depth_aggregation=depth_aggregation,
    )
    composite.setdefault("metadata", {})
    composite["metadata"].update({
        "aggregation": "event_precursor_composite",
        "lead_steps": int(lead_steps),
        "n_precursor_steps": int(len(precursor_indices)),
    })
    return composite


def compute_event_lead_lag_regression(
    field: xr.DataArray,
    events: List[Dict[str, Any]],
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: str = "mean",
    max_lag: int = 3,
) -> Dict[str, Any]:
    """Compute lag correlation between a field timeseries and event-count timeseries."""
    field_series = _reduce_field_to_timeseries(
        field=field,
        lon_range=lon_range,
        lat_range=lat_range,
        depth_range=depth_range,
        depth_aggregation=depth_aggregation,
        weighting="area_weighted",
        metadata_updates={"aggregation": "event_lead_lag_regression_source"},
    )
    event_indices = _event_time_indices(events, len(field_series.get("times", [])))
    event_values = np.zeros(len(field_series.get("times", [])), dtype=float)
    for index in event_indices:
        event_values[index] += 1.0
    event_series = {
        "times": list(field_series.get("times", [])),
        "values": event_values.tolist(),
        "metadata": {
            "variable": "event_count",
            "event_count": int(sum(event_values)),
            "time_key": "time_index",
        },
    }

    available_points = len(field_series.get("values", []))
    if available_points < max_lag + 10:
        adjusted_max_lag = max(1, min(int(max_lag), max(available_points // 2, 1)))
        return _compute_short_lag_correlation(
            field_series,
            event_series,
            max_lag=adjusted_max_lag,
            confidence_level=0.95,
        )
    return compute_lag_correlation(field_series, event_series, max_lag=max_lag, confidence_level=0.95)


def compute_oxygen_chla_coupling_metrics(
    oxygen_timeseries: Dict[str, Any],
    chla_timeseries: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize oxygen-chlorophyll co-variability as a mechanism score."""
    times, oxygen_values = _extract_series(oxygen_timeseries)
    _, chla_values = _extract_series(chla_timeseries, target_times=times)
    correlation = _safe_correlation(oxygen_values, chla_values)
    lag = _best_lag(oxygen_values, chla_values, max_lag=min(3, max(len(oxygen_values) // 3, 1)))

    candidate_scores = {
        "synchronous_coupling": abs(correlation),
        "lagged_coupling": abs(lag["correlation"]),
        "decoupled_behavior": max(0.0, 1.0 - max(abs(correlation), abs(lag["correlation"]))),
    }
    if abs(lag["correlation"]) >= abs(correlation):
        top = "lagged_coupling" if lag["lag"] != 0 else "synchronous_coupling"
    else:
        top = "synchronous_coupling"

    supporting = [
        f"Zero-lag correlation is {_format_number(correlation)}.",
        f"Best short-lag correlation is {_format_number(lag['correlation'])} at lag {lag['lag']}.",
    ]
    conflicting = []
    if max(abs(correlation), abs(lag["correlation"])) < 0.2:
        conflicting.append("Coupling metrics remain weak across tested short lags.")

    claim_strength = _claim_from_correlation(max(abs(correlation), abs(lag["correlation"])))
    return _mechanism_score_result(
        candidate_scores=candidate_scores,
        top_mechanism=top,
        claim_strength=claim_strength,
        supporting_evidence=supporting,
        conflicting_evidence=conflicting,
        metadata={"pair": ["oxygen", "chla"]},
    )


def compute_stratification_response_index(
    stratification: Dict[str, Any],
    response: Dict[str, Any],
) -> Dict[str, Any]:
    """Estimate whether a response variable tracks stratification changes."""
    times, strat_values = _extract_series(stratification)
    _, response_values = _extract_series(response, target_times=times)
    correlation = _safe_correlation(strat_values, response_values)

    candidate_scores = {
        "stratification_control": abs(correlation),
        "weak_response": max(0.0, 1.0 - abs(correlation)),
    }
    top = "stratification_control" if abs(correlation) >= 0.3 else "weak_response"
    supporting = [f"Response vs stratification correlation is {_format_number(correlation)}."]
    conflicting = []
    if abs(correlation) < 0.2:
        conflicting.append("Response remains weakly correlated with the stratification proxy.")

    return _mechanism_score_result(
        candidate_scores=candidate_scores,
        top_mechanism=top,
        claim_strength=_claim_from_correlation(abs(correlation)),
        supporting_evidence=supporting,
        conflicting_evidence=conflicting,
        metadata={"relationship": "stratification_response"},
    )


def compute_event_condition_contrast(
    field: xr.DataArray,
    events: List[Dict[str, Any]],
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: str = "mean",
    partition_mode: Optional[str] = None,
    subregion_grid: Optional[Tuple[int, int]] = None,
    subregion_weighting: str = "area_weighted",
) -> Dict[str, Any]:
    """Compare regional field means during event times versus the background."""
    field = materialize_partitioned_xarray(field)

    if "time" not in field.dims:
        raise ValueError("compute_event_condition_contrast requires a time dimension")
    source_proxy = _infer_mesoscale_proxy_name(field)
    regional = _reduce_field_to_timeseries(
        field=field,
        lon_range=lon_range,
        lat_range=lat_range,
        depth_range=depth_range,
        depth_aggregation=depth_aggregation,
        weighting="area_weighted",
        metadata_updates={"aggregation": "event_condition_contrast_source"},
    )
    times, values = _extract_series(regional)
    event_indices = _event_time_indices(events, len(times))
    event_mask = np.zeros(len(times), dtype=bool)
    event_mask[event_indices] = True
    partition_active = partition_mode is not None or subregion_grid is not None
    if not partition_active:
        event_mean, background_mean, delta, standardized = _compute_event_background_contrast(values, event_mask)
        candidate_scores = {
            source_proxy: standardized,
            "background_like": max(0.0, 1.0 - min(standardized, 1.0)),
        }
        top = source_proxy if standardized >= 0.5 else "background_like"
        supporting = [
            f"During-event mean is {_format_number(event_mean)} versus {_format_number(background_mean)} outside events.",
            f"Standardized {source_proxy} contrast is {_format_number(standardized)}.",
        ]
        conflicting = []
        if standardized < 0.5:
            conflicting.append("Event-time conditions are close to the background state.")

        return _mechanism_score_result(
            candidate_scores=candidate_scores,
            top_mechanism=top,
            claim_strength=_claim_from_standardized_difference(standardized),
            supporting_evidence=supporting,
            conflicting_evidence=conflicting,
            times=[str(t) for t in times.tolist()],
            values=np.asarray(values, dtype=float).tolist(),
            metadata={
                "comparison": "event_condition_contrast",
                "source_proxy": source_proxy,
                "source_field": field.name or field.attrs.get("long_name") or "field",
            },
        )

    normalized_partition_mode = (partition_mode or "lon_lat_grid").strip().lower()
    if normalized_partition_mode != "lon_lat_grid":
        raise ValueError(f"Unsupported partition_mode for compute_event_condition_contrast: {partition_mode}")

    grid = _normalize_subregion_grid(subregion_grid)
    weighting = (subregion_weighting or "area_weighted").strip().lower()
    if weighting not in {"area_weighted", "equal"}:
        raise ValueError(f"Unsupported subregion_weighting: {subregion_weighting}")

    subregion_breakdown: List[Dict[str, Any]] = []
    weighted_score = 0.0
    total_weight = 0.0
    valid_subregion_count = 0

    for subregion in _build_rectangular_subregions(lon_range=lon_range, lat_range=lat_range, grid=grid):
        record: Dict[str, Any] = {
            "subregion_id": subregion["subregion_id"],
            "label": subregion["label"],
            "lon_range": [float(subregion["lon_range"][0]), float(subregion["lon_range"][1])],
            "lat_range": [float(subregion["lat_range"][0]), float(subregion["lat_range"][1])],
        }
        area_weight = _compute_subregion_area_weight(
            field=field,
            lon_range=subregion["lon_range"],
            lat_range=subregion["lat_range"],
            depth_range=depth_range,
            depth_aggregation=depth_aggregation,
        )
        record["area_weight"] = float(area_weight)
        if area_weight <= 0.0:
            record["status"] = "skipped_no_valid_ocean"
            subregion_breakdown.append(record)
            continue

        try:
            subregional = _reduce_field_to_timeseries(
                field=field,
                lon_range=subregion["lon_range"],
                lat_range=subregion["lat_range"],
                depth_range=depth_range,
                depth_aggregation=depth_aggregation,
                weighting="area_weighted",
                metadata_updates={"aggregation": "event_condition_contrast_source"},
            )
        except ValueError:
            record["status"] = "skipped_no_valid_ocean"
            subregion_breakdown.append(record)
            continue

        _, subregion_values = _extract_series(subregional)
        subregion_event_indices = _event_time_indices(events, len(subregion_values))
        subregion_event_mask = np.zeros(len(subregion_values), dtype=bool)
        subregion_event_mask[subregion_event_indices] = True

        try:
            event_mean, background_mean, delta, standardized = _compute_event_background_contrast(
                subregion_values,
                subregion_event_mask,
            )
        except ValueError:
            record["status"] = "skipped_no_valid_samples"
            subregion_breakdown.append(record)
            continue

        effective_weight = area_weight if weighting == "area_weighted" else 1.0
        weighted_score += effective_weight * standardized
        total_weight += effective_weight
        valid_subregion_count += 1

        record.update({
            "event_mean": float(event_mean),
            "background_mean": float(background_mean),
            "delta": float(delta),
            "standardized_contrast": float(standardized),
            "claim_strength": _claim_from_standardized_difference(standardized),
            "status": "ok",
        })
        subregion_breakdown.append(record)

    if total_weight <= 0.0:
        raise ValueError("Partitioned event/background contrast requires at least one valid subregion")

    standardized = weighted_score / total_weight
    candidate_scores = {
        source_proxy: standardized,
        "background_like": max(0.0, 1.0 - min(standardized, 1.0)),
    }
    top = source_proxy if standardized >= 0.5 else "background_like"

    supporting = [
        (
            f"Area-weighted {source_proxy} contrast across {valid_subregion_count}/"
            f"{len(subregion_breakdown)} subregions is {_format_number(standardized)}."
        )
    ]
    strongest_subregion = max(
        (item for item in subregion_breakdown if item.get("status") == "ok"),
        key=lambda item: float(item.get("standardized_contrast") or 0.0),
        default=None,
    )
    if strongest_subregion is not None:
        supporting.append(
            f"Strongest valid subregion is {strongest_subregion['label']} with standardized contrast "
            f"{_format_number(strongest_subregion.get('standardized_contrast'))}."
        )

    conflicting = []
    skipped_count = len([item for item in subregion_breakdown if item.get("status") != "ok"])
    if standardized < 0.5:
        conflicting.append("Most subregions stay close to the broad background state.")
    if skipped_count > 0:
        conflicting.append(f"{skipped_count} subregions were skipped because valid ocean or event samples were unavailable.")

    return _mechanism_score_result(
        candidate_scores=candidate_scores,
        top_mechanism=top,
        claim_strength=_claim_from_standardized_difference(standardized),
        supporting_evidence=supporting,
        conflicting_evidence=conflicting,
        times=[str(t) for t in times.tolist()],
        values=np.asarray(values, dtype=float).tolist(),
        subregion_breakdown=subregion_breakdown,
        metadata={
            "comparison": "event_condition_contrast",
            "source_proxy": source_proxy,
            "source_field": field.name or field.attrs.get("long_name") or "field",
            "partition_mode": normalized_partition_mode,
            "subregion_grid": [int(grid[0]), int(grid[1])],
            "subregion_weighting": weighting,
            "n_subregions": len(subregion_breakdown),
            "n_valid_subregions": valid_subregion_count,
        },
    )


def replace_field_with_climatology(
    data: xr.DataArray,
    period: str = "monthly",
) -> xr.DataArray:
    """Replace each time step with the corresponding climatological mean."""
    if find_partitioned_values((data,)):
        from packages.tool_loader.partitioned_execution import _compute_partitioned_climatology_replacement

        return _compute_partitioned_climatology_replacement(
            "replace_field_with_climatology",
            {"data": data, "period": period},
        )
    data = materialize_partitioned_xarray(data)

    climatology = compute_field_climatology(data, period=period)
    replaced = _project_climatology_to_time(climatology=climatology, source_time=data["time"], period=period)

    replaced.name = f"{data.name or 'field'}_climatology_replaced"
    replaced.attrs = {
        **data.attrs,
        "aggregation": "replace_field_with_climatology",
        "climatology_period": period,
    }
    return replaced


def remove_field_anomaly_component(
    data: xr.DataArray,
    period: str = "monthly",
) -> xr.DataArray:
    """Remove the anomaly component by reconstructing the climatological field."""
    if find_partitioned_values((data,)):
        from packages.tool_loader.partitioned_execution import _compute_partitioned_climatology_replacement

        return _compute_partitioned_climatology_replacement(
            "remove_field_anomaly_component",
            {"data": data, "period": period},
        )
    data = materialize_partitioned_xarray(data)

    climatology = compute_field_climatology(data, period=period)
    reconstructed = _project_climatology_to_time(climatology=climatology, source_time=data["time"], period=period)
    reconstructed.name = f"{data.name or 'field'}_without_anomaly"
    reconstructed.attrs = {
        **data.attrs,
        "aggregation": "remove_field_anomaly_component",
        "climatology_period": period,
    }
    return reconstructed


def filter_mesoscale_component(
    data: xr.DataArray,
    cutoff_period: float = 2.0,
    component: str = "mesoscale",
) -> xr.DataArray:
    """Alias wrapper for mesoscale/background filtering."""
    return compute_mesoscale_background_separation(
        data=data,
        cutoff_period=cutoff_period,
        component=component,
    )


def run_proxy_counterfactual_experiment(
    baseline: xr.DataArray,
    counterfactual: xr.DataArray,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    depth_range: Optional[Tuple[float, float]] = None,
    depth_aggregation: str = "mean",
) -> Dict[str, Any]:
    """
    Reduce baseline-minus-counterfactual regional differences to a timeseries.
    """
    baseline = materialize_partitioned_xarray(baseline)
    counterfactual = materialize_partitioned_xarray(counterfactual)

    baseline_aligned, counter_aligned = xr.align(baseline, counterfactual, join="inner")
    difference = baseline_aligned - counter_aligned
    difference.name = f"{baseline.name or 'field'}_counterfactual_difference"
    difference.attrs = {
        **baseline.attrs,
        "aggregation": "proxy_counterfactual_difference",
    }
    result = _reduce_field_to_timeseries(
        field=difference,
        lon_range=lon_range,
        lat_range=lat_range,
        depth_range=depth_range,
        depth_aggregation=depth_aggregation,
        weighting="area_weighted",
        metadata_updates={
            "variable": difference.name or "counterfactual_difference",
            "comparison": "baseline_minus_counterfactual",
        },
    )
    return result


def compare_counterfactual_outcome(
    baseline: Dict[str, Any],
    counterfactual: Dict[str, Any],
    mechanism_name: str = "target_mechanism",
) -> Dict[str, Any]:
    """Turn baseline-vs-counterfactual outcome differences into an evidence report."""
    times, baseline_values = _extract_series(baseline)
    _, counter_values = _extract_series(counterfactual, target_times=times)
    delta = baseline_values - counter_values
    mean_abs_delta = _mean_abs(delta)
    baseline_scale = max(_mean_abs(baseline_values), 1e-12)
    relative_delta = mean_abs_delta / baseline_scale

    if relative_delta >= 0.35:
        claim_strength = CLAIM_SUPPORTED
        supported = [
            f"Removing or replacing {mechanism_name} changes the regional response by about {relative_delta * 100.0:.1f}%."
        ]
        limited = []
    elif relative_delta >= 0.15:
        claim_strength = CLAIM_LIMITED
        supported = []
        limited = [
            f"{mechanism_name} sensitivity is detectable ({relative_delta * 100.0:.1f}% relative change) but not dominant."
        ]
    else:
        claim_strength = CLAIM_UNTESTABLE
        supported = []
        limited = [f"{mechanism_name} sensitivity stays small ({relative_delta * 100.0:.1f}% relative change)."]

    uncertainty = [
        "This is a proxy counterfactual in data space, not a fully dynamical intervention experiment.",
        f"Mean absolute baseline-minus-counterfactual change is {_format_number(mean_abs_delta)}.",
    ]
    return _evidence_report_result(
        supported_claims=supported,
        limited_claims=limited,
        untestable_claims=[] if claim_strength != CLAIM_UNTESTABLE else [f"{mechanism_name} is not strongly identifiable with the current setup."],
        residual_or_uncertainty=uncertainty,
        claim_strength=claim_strength,
        metadata={"mechanism_name": mechanism_name},
    )


def rank_mechanism_support(
    evidence_items: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate multiple mechanism-score style results into one ranked result."""
    aggregate: Dict[str, float] = {}
    supporting: List[str] = []
    conflicting: List[str] = []
    claim_strength = CLAIM_UNTESTABLE
    proxy_breakdowns: List[Dict[str, Any]] = []
    for item in evidence_items:
        result = _unwrap_normalized_result(item)
        if result.get("output_type") != "mechanism_score_result":
            continue
        for mechanism in result.get("candidate_mechanisms", []):
            name = str(mechanism.get("name") or "").strip()
            score = float(mechanism.get("score") or 0.0)
            if not name:
                continue
            aggregate[name] = aggregate.get(name, 0.0) + score
        supporting.extend(_normalize_string_list(result.get("supporting_evidence")))
        conflicting.extend(_normalize_string_list(result.get("conflicting_evidence")))
        claim_strength = _max_claim_strength(claim_strength, str(result.get("claim_strength") or CLAIM_UNTESTABLE))

        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        source_proxy = str(metadata.get("source_proxy") or "").strip()
        if source_proxy:
            proxy_score = next(
                (
                    float(mechanism.get("score") or 0.0)
                    for mechanism in result.get("candidate_mechanisms", [])
                    if str(mechanism.get("name") or "").strip() == source_proxy
                ),
                None,
            )
            if proxy_score is not None:
                proxy_breakdowns.append({
                    "name": source_proxy,
                    "score": float(proxy_score),
                    "claim_strength": str(result.get("claim_strength") or CLAIM_UNTESTABLE),
                    "subregion_breakdown": result.get("subregion_breakdown", []),
                })

    if not aggregate:
        raise ValueError("rank_mechanism_support requires at least one mechanism_score_result")

    top = max(aggregate.items(), key=lambda item: item[1])[0]
    return _mechanism_score_result(
        candidate_scores=aggregate,
        top_mechanism=top,
        claim_strength=claim_strength,
        supporting_evidence=_dedupe_strings(supporting)[:6],
        conflicting_evidence=_dedupe_strings(conflicting)[:4],
        proxy_breakdowns=proxy_breakdowns,
        metadata={
            "aggregation": "rank_mechanism_support",
            "comparison": "mesoscale_proxy_ranking" if proxy_breakdowns else None,
            "n_proxy_breakdowns": len(proxy_breakdowns) if proxy_breakdowns else None,
        },
    )


def grade_evidence_strength(
    evidence_items: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Convert multiple mechanism/evidence items into a consolidated evidence grade."""
    supported: List[str] = []
    limited: List[str] = []
    untestable: List[str] = []
    uncertainties: List[str] = []
    claim_strength = CLAIM_UNTESTABLE

    for item in evidence_items:
        result = _unwrap_normalized_result(item)
        output_type = result.get("output_type")
        if output_type == "mechanism_score_result":
            top = result.get("top_mechanism")
            strength = str(result.get("claim_strength") or CLAIM_UNTESTABLE)
            if strength == CLAIM_SUPPORTED:
                supported.append(f"{top} is the strongest supported mechanism candidate.")
            elif strength == CLAIM_LIMITED:
                limited.append(f"{top} is consistent with the available evidence but remains weakly identified.")
            else:
                untestable.append(f"{top} cannot be strongly identified with the available variables.")
            uncertainties.extend(_normalize_string_list(result.get("conflicting_evidence")))
            claim_strength = _max_claim_strength(claim_strength, strength)
        elif output_type == "evidence_report_result":
            supported.extend(_normalize_string_list(result.get("supported_claims")))
            limited.extend(_normalize_string_list(result.get("limited_claims")))
            untestable.extend(_normalize_string_list(result.get("untestable_claims")))
            uncertainties.extend(_normalize_string_list(result.get("residual_or_uncertainty")))
            claim_strength = _max_claim_strength(claim_strength, str(result.get("claim_strength") or CLAIM_UNTESTABLE))

    return _evidence_report_result(
        supported_claims=_dedupe_strings(supported)[:5],
        limited_claims=_dedupe_strings(limited)[:5],
        untestable_claims=_dedupe_strings(untestable)[:5],
        residual_or_uncertainty=_dedupe_strings(uncertainties)[:6],
        claim_strength=claim_strength,
        metadata={"aggregation": "grade_evidence_strength"},
    )


def assemble_mechanism_evidence_report(
    mechanism_scores: Sequence[Dict[str, Any]],
    context_note: str = "",
) -> Dict[str, Any]:
    """Assemble a user-facing evidence report from mechanism scores."""
    normalized_scores = []
    for item in mechanism_scores:
        result = _unwrap_normalized_result(item)
        if result.get("output_type") == "mechanism_score_result":
            normalized_scores.append(result)
    if not normalized_scores:
        uncertainty = [
            "No mechanism-score diagnostics were available to rank.",
            "Trend, event-count, and other non-mechanism outputs cannot be passed directly into assemble_mechanism_evidence_report.",
        ]
        if context_note.strip():
            uncertainty.append(context_note.strip())
        return _evidence_report_result(
            supported_claims=[],
            limited_claims=[],
            untestable_claims=["Mechanism ranking was skipped because no mechanism_score_result inputs were available."],
            residual_or_uncertainty=uncertainty,
            claim_strength=CLAIM_UNTESTABLE,
            metadata={"aggregation": "assemble_mechanism_evidence_report", "mechanism_input_count": 0},
        )

    ranked = rank_mechanism_support(normalized_scores)
    graded = grade_evidence_strength(list(normalized_scores) + [ranked])
    report = _unwrap_normalized_result(graded)
    if context_note.strip():
        report["residual_or_uncertainty"] = _normalize_string_list(report.get("residual_or_uncertainty")) + [context_note.strip()]
    return report


def assemble_environment_health_report(
    branches: Sequence[Dict[str, Any]],
    context_note: str = "",
) -> Dict[str, Any]:
    """Assemble a multi-branch marine environment health assessment."""
    branch_assessments: List[Dict[str, Any]] = []
    supporting: List[str] = []
    uncertainties: List[str] = []
    overall_score = 0.0
    supported_direction_counts = {
        ENV_DIRECTION_DETERIORATING: 0,
        ENV_DIRECTION_IMPROVING: 0,
    }

    for index, branch in enumerate(branches):
        if not isinstance(branch, dict):
            continue

        name = str(branch.get("name") or f"branch_{index + 1}").strip() or f"branch_{index + 1}"
        indicator_label = str(
            branch.get("indicator_label")
            or branch.get("indicator")
            or _humanize_environment_indicator(name)
        ).strip() or _humanize_environment_indicator(name)
        worse_when = str(branch.get("worse_when") or "increase").strip().lower()
        raw_result = branch.get("result")
        role = _normalize_environment_branch_role(branch.get("role"))
        evidence_kind = _normalize_environment_evidence_kind(branch.get("evidence_kind"))
        metric = _normalize_environment_metric(branch.get("metric"))

        assessment = _assess_environment_health_branch(
            name=name,
            indicator_label=indicator_label,
            worse_when=worse_when,
            result=raw_result,
            role=role,
            evidence_kind=evidence_kind,
            metric=metric,
        )
        branch_assessments.append(assessment)
        supporting.extend(_normalize_string_list(assessment.get("supporting_evidence")))
        uncertainties.extend(_normalize_string_list(assessment.get("uncertainties")))

        contributes_to_overall = str(assessment.get("role") or role) != ENV_ROLE_AUXILIARY_CONTEXT
        score = float(assessment.get("score_contribution") or 0.0) if contributes_to_overall else 0.0
        overall_score += score

        if (
            contributes_to_overall
            and
            assessment.get("support_strength") == ENV_SUPPORT_SUPPORTED
            and assessment.get("direction") in supported_direction_counts
        ):
            supported_direction_counts[str(assessment["direction"])] += 1

    if overall_score >= 1.0:
        overall_verdict = "Deteriorating"
    elif overall_score <= -1.0:
        overall_verdict = "Improving"
    else:
        overall_verdict = "Stable"

    effective_branches = [
        branch for branch in branch_assessments
        if branch.get("support_strength") in {ENV_SUPPORT_SUPPORTED, ENV_SUPPORT_LIMITED}
        and branch.get("role") != ENV_ROLE_AUXILIARY_CONTEXT
    ]
    if not effective_branches:
        overall_support_strength = ENV_SUPPORT_UNTESTABLE
    elif overall_verdict == "Deteriorating":
        if (
            supported_direction_counts[ENV_DIRECTION_DETERIORATING] >= 2
            and overall_score >= 2.0
        ):
            overall_support_strength = ENV_SUPPORT_SUPPORTED
        else:
            overall_support_strength = ENV_SUPPORT_LIMITED
    elif overall_verdict == "Improving":
        if (
            supported_direction_counts[ENV_DIRECTION_IMPROVING] >= 2
            and overall_score <= -2.0
        ):
            overall_support_strength = ENV_SUPPORT_SUPPORTED
        else:
            overall_support_strength = ENV_SUPPORT_LIMITED
    else:
        overall_support_strength = ENV_SUPPORT_LIMITED

    key_pressures = _environment_key_pressures(branch_assessments)
    stabilizing_signals = _environment_stabilizing_signals(branch_assessments)
    monitoring_priorities = _environment_monitoring_priorities(
        branch_assessments,
        overall_verdict=overall_verdict,
        overall_support_strength=overall_support_strength,
    )
    policy_action_matrix = _environment_policy_action_matrix(
        branch_assessments,
        overall_verdict=overall_verdict,
        overall_support_strength=overall_support_strength,
    )
    policy_recommendations = _environment_policy_recommendations(
        branch_assessments,
        overall_verdict=overall_verdict,
        overall_support_strength=overall_support_strength,
        policy_action_matrix=policy_action_matrix,
    )
    overall_narrative = _environment_overall_narrative(
        branch_assessments,
        overall_verdict=overall_verdict,
        overall_support_strength=overall_support_strength,
    )

    if context_note.strip():
        uncertainties.append(context_note.strip())

    return _environment_assessment_result(
        overall_verdict=overall_verdict,
        overall_support_strength=overall_support_strength,
        branch_assessments=branch_assessments,
        supporting_evidence=_dedupe_strings(supporting)[:8],
        uncertainties=_dedupe_strings(uncertainties)[:8],
        overall_narrative=overall_narrative,
        key_pressures=key_pressures,
        stabilizing_signals=stabilizing_signals,
        monitoring_priorities=monitoring_priorities,
        policy_recommendations=policy_recommendations,
        policy_action_matrix=policy_action_matrix,
        metadata={
            "aggregation": "assemble_environment_health_report",
            "n_input_branches": len([branch for branch in branches if isinstance(branch, dict)]),
            "overall_score": float(overall_score),
        },
    )


def check_claim_support_level(
    claim: str,
    evidence: Dict[str, Any],
    requested_strength: str = CLAIM_SUPPORTED,
) -> Dict[str, Any]:
    """Check whether a textual claim exceeds the evidence strength of an upstream result."""
    normalized = _unwrap_normalized_result(evidence)
    actual = str(normalized.get("claim_strength") or CLAIM_UNTESTABLE)
    claim_text = claim.strip() or "The requested claim"
    if _claim_rank(actual) >= _claim_rank(requested_strength):
        supported = [f"{claim_text} does not exceed the available support level ({actual})."]
        limited: List[str] = []
    else:
        supported = []
        limited = [f"{claim_text} is stronger than the available support level ({actual})."]

    return _evidence_report_result(
        supported_claims=supported,
        limited_claims=limited,
        untestable_claims=[],
        residual_or_uncertainty=_normalize_string_list(normalized.get("conflicting_evidence")) + _normalize_string_list(normalized.get("residual_or_uncertainty")),
        claim_strength=actual,
        metadata={"requested_strength": requested_strength},
    )


def assemble_policy_recommendation_report(
    evidence_items: Sequence[Dict[str, Any]],
    region_scope: str = "China Seas and western Pacific",
    policy_context: str = "",
    management_objective: str = "",
    context_note: str = "",
) -> Dict[str, Any]:
    """Translate existing ocean diagnostics into evidence-bounded policy guidance."""
    normalized_items = [
        item for item in evidence_items
        if isinstance(item, dict) and str(item.get("output_type") or "").strip()
    ]

    evidence_table = [
        _policy_evidence_entry(item, index)
        for index, item in enumerate(normalized_items, start=1)
    ]
    priority_level = _policy_priority_level(evidence_table)

    recommended_actions: List[Dict[str, str]] = []
    monitoring_priorities: List[str] = []
    evidence_constraints: List[str] = []

    for item, entry in zip(normalized_items, evidence_table):
        recommended_actions.extend(_policy_actions_for_evidence(item, entry))
        monitoring_priorities.extend(_policy_monitoring_for_evidence(item, entry))
        evidence_constraints.extend(_policy_constraints_for_evidence(item, entry))

    if not recommended_actions:
        recommended_actions.append({
            "theme": "evidence_development",
            "priority": "near_term",
            "action_type": "governance",
            "target": "minimum policy evidence package",
            "where_when": region_scope.strip() or "selected management region",
            "evidence_basis": "No upstream diagnostic result was strong enough to support targeted intervention.",
            "action": "Build a minimum evidence package before adopting high-cost measures: combine indicator trends, event exposure, and mechanism diagnostics for the same region and period.",
            "guardrail": "Do not adopt high-cost or source-specific measures until the evidence package becomes stronger.",
            "rationale": "The supplied outputs do not yet provide a policy-diagnostic signal strong enough for targeted intervention.",
            "evidence_strength": "limited",
        })

    monitoring_priorities.extend(_policy_default_monitoring_priorities(evidence_table))
    evidence_constraints.extend(_policy_default_constraints(evidence_table, context_note=context_note))
    governance_notes = _policy_governance_notes(
        region_scope=region_scope,
        policy_context=policy_context,
        management_objective=management_objective,
    )

    policy_summary = _policy_summary_sentence(
        priority_level=priority_level,
        evidence_table=evidence_table,
        management_objective=management_objective,
    )

    return {
        "output_type": "policy_recommendation_result",
        "region_scope": region_scope.strip() or "China Seas and western Pacific",
        "policy_context": policy_context.strip(),
        "management_objective": management_objective.strip(),
        "priority_level": priority_level,
        "policy_summary": policy_summary,
        "recommended_actions": _dedupe_policy_actions(recommended_actions)[:8],
        "monitoring_priorities": _dedupe_strings(monitoring_priorities)[:6],
        "governance_notes": _dedupe_strings(governance_notes)[:5],
        "evidence_constraints": _dedupe_strings(evidence_constraints)[:6],
        "evidence_table": evidence_table[:12],
        "metadata": {
            "aggregation": "assemble_policy_recommendation_report",
            "n_input_items": len(normalized_items),
            "n_evidence_entries": len(evidence_table),
            "domain_limit": "Use for China Seas and western Pacific datasets unless the user supplies a narrower region.",
        },
    }


def _policy_evidence_entry(result: Dict[str, Any], index: int) -> Dict[str, Any]:
    output_type = str(result.get("output_type") or "unknown").strip()
    label = _policy_evidence_label(result, index=index)
    theme = _policy_theme_from_result(result, label)
    signal, support_strength, rationale, score = _policy_signal_from_result(result, label=label, theme=theme)
    return {
        "label": label,
        "output_type": output_type,
        "theme": theme,
        "signal": signal,
        "support_strength": support_strength,
        "priority_score": float(score),
        "rationale": rationale,
    }


def _policy_evidence_label(result: Dict[str, Any], *, index: int) -> str:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    for key in ("title", "variable", "source_variable", "aggregation"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return _humanize_policy_label(value)

    output_type = str(result.get("output_type") or "").strip()
    if output_type == "environment_assessment_result":
        return "Marine environmental health assessment"
    if output_type == "event_detection_result":
        return _humanize_policy_label(result.get("event_type") or "event detection")
    if output_type == "event_statistics_result":
        return "Event statistics"
    if output_type == "mechanism_score_result":
        return _humanize_policy_label(result.get("top_mechanism") or "mechanism support")
    if output_type == "evidence_report_result":
        return "Mechanism evidence synthesis"
    if output_type == "trend_result":
        return "Indicator trend"
    if output_type == "field_trend_result":
        return "Spatial trend field"
    if output_type == "watermass_event_association_result":
        return "Watermass-event association"
    return _humanize_policy_label(output_type or f"evidence item {index}")


def _policy_signal_from_result(
    result: Dict[str, Any],
    *,
    label: str,
    theme: str,
) -> Tuple[str, str, str, float]:
    output_type = str(result.get("output_type") or "").strip()

    if output_type == "environment_assessment_result":
        verdict = str(result.get("overall_verdict") or "Unknown").strip()
        support = str(result.get("overall_support_strength") or "limited").strip().lower()
        narrative = str(result.get("overall_narrative") or "").strip()
        score = 4.0 if verdict == "Deteriorating" and support == "supported" else 3.0 if verdict == "Deteriorating" else 1.0
        return verdict.lower(), support, narrative or f"Overall environmental-health verdict is {verdict}.", score

    if output_type == "trend_result":
        slope = _safe_float(result.get("slope"))
        significant = result.get("is_significant") is True
        direction = _policy_direction_from_value(slope)
        support = "supported" if significant else "limited"
        risk_score = _policy_risk_score(theme=theme, direction=direction, supported=significant)
        if slope is None:
            return "trend unavailable", "untestable", f"{label} does not include a usable slope.", 0.0
        rationale = f"{label} is {direction} with slope {_format_number(slope)}."
        return direction, support, rationale, risk_score

    if output_type == "field_trend_result":
        summary = _policy_field_trend_summary(result)
        direction = str(summary.get("direction") or "mixed").strip()
        support = str(summary.get("support_strength") or "limited").strip()
        score = _policy_risk_score(theme=theme, direction=direction, supported=support == "supported")
        rationale = str(summary.get("rationale") or f"{label} has a {direction} spatial trend pattern.")
        return direction, support, rationale, score

    if output_type == "event_detection_result":
        event_type = str(result.get("event_type") or label).strip()
        count = _policy_event_count(result)
        if count > 0:
            support = "supported"
            score = 3.0 if theme in {"oxygen", "chlorophyll", "heat"} else 2.0
            return "events detected", support, f"{_humanize_policy_label(event_type)} detection found {count} event(s).", score
        return "no events detected", "limited", f"{_humanize_policy_label(event_type)} detection found no events under the selected thresholds.", 0.5

    if output_type == "event_statistics_result":
        count = _policy_event_count(result)
        group_by = str(result.get("group_by") or "").strip()
        groups = result.get("groups")
        n_groups = len(groups) if isinstance(groups, dict) else 0
        support = "supported" if count > 0 and n_groups >= 2 else "limited"
        score = 2.5 if count > 0 else 0.5
        return "event exposure summarized", support, f"Event statistics contain {count} event(s) across {n_groups} {group_by or 'group'} bin(s).", score

    if output_type == "event_comparison_result":
        return "period comparison", "limited", f"{label} compares event behavior between supplied periods.", 2.0

    if output_type == "mechanism_score_result":
        top = str(result.get("top_mechanism") or "candidate mechanism").strip()
        strength = str(result.get("claim_strength") or "limited").strip()
        score = 2.5 if "supported" in strength.lower() else 1.5
        return "mechanism evidence", strength, f"Top mechanism is {_humanize_policy_label(top)} with {strength} support.", score

    if output_type == "evidence_report_result":
        supported = _normalize_string_list(result.get("supported_claims"))
        limited = _normalize_string_list(result.get("limited_claims"))
        strength = str(result.get("claim_strength") or "limited").strip()
        score = 2.5 if supported else 1.5 if limited else 0.5
        rationale = supported[0] if supported else limited[0] if limited else "Evidence synthesis did not identify a strongly supported claim."
        return "evidence synthesis", strength, rationale, score

    if output_type == "watermass_event_association_result":
        return "watermass association", "limited", f"{label} identifies where event behavior is associated with watermass structure.", 2.0

    if output_type == "spatial_field_result":
        return _policy_signal_from_spatial_field(result, label=label, theme=theme)

    if output_type == "event_spatial_distribution_result":
        count = _safe_float(result.get("event_count"))
        if count is not None and count > 0:
            return "events spatially distributed", "supported", f"{label} contains {_format_number(count)} spatially distributed event(s).", 2.5
        return "no spatial events", "limited", f"{label} does not contain a positive spatial event count.", 0.5

    if output_type in {"profile_result", "section_result", "hovmoller_result", "eof_result", "lag_correlation_result", "regression_map_result", "composite_result", "spectrum_result", "histogram_result", "histogram_2d_result", "timeseries_result", "climatology_result"}:
        return "diagnostic context", "limited", f"{label} provides supporting diagnostic context rather than a direct policy trigger.", 1.0

    return "context only", "limited", f"{label} is retained as context, but it is not a direct management trigger.", 0.5


def _policy_signal_from_spatial_field(
    result: Dict[str, Any],
    *,
    label: str,
    theme: str,
) -> Tuple[str, str, str, float]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    event_type = str(metadata.get("event_type") or "").strip()
    summary_mode = str(metadata.get("summary_mode") or "").strip()
    values = _spatial_values_from_result(result)
    if values.size == 0:
        return "spatial evidence unavailable", "untestable", f"{label} has no finite spatial values.", 0.0

    total = float(np.nansum(values))
    max_value = float(np.nanmax(values))
    if event_type or summary_mode:
        if total > 0 or max_value > 0:
            score = 3.0 if theme in {"oxygen", "chlorophyll", "heat"} else 2.0
            return (
                "spatial exposure present",
                "supported",
                f"{label} has positive spatial exposure (total {_format_number(total)}, max {_format_number(max_value)}).",
                score,
            )
        return "no spatial exposure", "limited", f"{label} has no positive spatial exposure under the configured event threshold.", 0.5

    return "spatial diagnostic context", "limited", f"{label} is a spatial diagnostic field; use a configured health branch to score it as policy evidence.", 1.0


def _policy_field_trend_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    data = result.get("data")
    if not isinstance(data, xr.DataArray):
        return {
            "direction": "mixed",
            "support_strength": "limited",
            "rationale": "Spatial trend result lacks a map-like slope field.",
        }

    values = np.asarray(data.values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {
            "direction": "trend unavailable",
            "support_strength": "untestable",
            "rationale": "Spatial trend field has no finite slope values.",
        }

    mean_slope = float(np.nanmean(values[finite]))
    positive_fraction = float(np.count_nonzero(values[finite] > 0) / np.count_nonzero(finite))
    negative_fraction = float(np.count_nonzero(values[finite] < 0) / np.count_nonzero(finite))
    sig = result.get("significance_mask")
    significant_fraction = 0.0
    if isinstance(sig, xr.DataArray):
        sig_values = np.asarray(sig.values, dtype=bool)
        if sig_values.shape == finite.shape:
            significant_fraction = float(np.count_nonzero(sig_values & finite) / np.count_nonzero(finite))

    if positive_fraction >= 0.6:
        direction = "increasing"
    elif negative_fraction >= 0.6:
        direction = "decreasing"
    else:
        direction = "mixed"

    support = "supported" if significant_fraction >= 0.25 else "limited"
    rationale = (
        f"Spatial trend pattern is {direction}; mean slope {_format_number(mean_slope)}, "
        f"significant-cell fraction {_format_number(significant_fraction)}."
    )
    return {
        "direction": direction,
        "support_strength": support,
        "rationale": rationale,
    }


def _policy_actions_for_evidence(result: Dict[str, Any], entry: Dict[str, Any]) -> List[Dict[str, str]]:
    output_type = str(entry.get("output_type") or "").strip()
    theme = str(entry.get("theme") or "general").strip()
    priority = _policy_action_priority(float(entry.get("priority_score") or 0.0))
    strength = str(entry.get("support_strength") or "limited").strip()
    rationale = str(entry.get("rationale") or "").strip()

    if output_type == "environment_assessment_result":
        action_matrix = result.get("policy_action_matrix")
        if isinstance(action_matrix, list) and action_matrix:
            return [
                _policy_action_record_from_matrix_item(item, priority=priority, strength=strength, rationale=rationale)
                for item in action_matrix
                if isinstance(item, dict)
            ]
        actions = []
        for recommendation in _normalize_string_list(result.get("policy_recommendations")):
            actions.append({
                "theme": "integrated_environment",
                "priority": priority,
                "action_type": "governance",
                "target": "integrated environmental-health management",
                "where_when": "selected region and assessment window",
                "evidence_basis": rationale or "Integrated environmental-health assessment.",
                "action": recommendation,
                "guardrail": _policy_guardrail_for_strength(strength),
                "rationale": rationale or "The integrated environmental-health assessment combines multiple indicators.",
                "evidence_strength": strength,
            })
        return actions

    action = _policy_action_text(theme=theme, signal=str(entry.get("signal") or ""))
    return [{
        "theme": theme,
        "priority": priority,
        "action_type": _policy_action_type_for_theme(theme),
        "target": _policy_target_for_theme(theme),
        "where_when": "hotspots, vulnerable seasons, and the selected management region",
        "evidence_basis": rationale,
        "action": action,
        "guardrail": _policy_guardrail_for_strength(strength),
        "rationale": rationale,
        "evidence_strength": strength,
    }]


def _policy_action_record_from_matrix_item(
    item: Dict[str, Any],
    *,
    priority: str,
    strength: str,
    rationale: str,
) -> Dict[str, str]:
    row_priority = str(item.get("priority") or priority or "routine").strip()
    action_type = str(item.get("action_type") or "governance").strip()
    target = str(item.get("target") or "environmental-health management").strip()
    evidence_basis = str(item.get("evidence_basis") or rationale).strip()
    guardrail = str(item.get("guardrail") or _policy_guardrail_for_strength(strength)).strip()
    return {
        "theme": str(item.get("theme") or "integrated_environment").strip(),
        "priority": row_priority,
        "action_type": action_type,
        "target": target,
        "where_when": str(item.get("where_when") or "").strip(),
        "evidence_basis": evidence_basis,
        "action": str(item.get("action") or "").strip(),
        "guardrail": guardrail,
        "rationale": evidence_basis,
        "evidence_strength": strength,
    }


def _policy_action_type_for_theme(theme: str) -> str:
    if theme == "oxygen":
        return "monitoring"
    if theme == "chlorophyll":
        return "source_control"
    if theme in {"heat", "stratification"}:
        return "seasonal_management"
    if theme in {"dynamics", "watermass"}:
        return "coastal_planning"
    return "governance"


def _policy_target_for_theme(theme: str) -> str:
    if theme == "oxygen":
        return "bottom oxygen risk areas"
    if theme == "chlorophyll":
        return "nutrient-sensitive coastal waters"
    if theme == "heat":
        return "warm-season marine heat risk"
    if theme == "stratification":
        return "stratified water-column windows"
    if theme == "dynamics":
        return "circulation-linked hotspots"
    if theme == "watermass":
        return "watermass pathway boundaries"
    return "coastal management evidence package"


def _policy_guardrail_for_strength(strength: str) -> str:
    lowered = str(strength or "").strip().lower()
    if "supported" in lowered:
        return "Use as prioritization evidence while retaining local source and mechanism checks."
    if "untestable" in lowered:
        return "Use only for evidence development until the signal is testable."
    return "Treat as screening and low-regret guidance, not standalone causal proof."


def _policy_monitoring_for_evidence(result: Dict[str, Any], entry: Dict[str, Any]) -> List[str]:
    theme = str(entry.get("theme") or "general").strip()
    output_type = str(entry.get("output_type") or "").strip()
    monitoring: List[str] = []

    if output_type == "environment_assessment_result":
        monitoring.extend(_normalize_string_list(result.get("monitoring_priorities")))

    if theme == "chlorophyll":
        monitoring.append("Track chlorophyll-a exposure, bloom duration, and nutrient-sensitive coastal hotspots with seasonal reporting.")
    elif theme == "oxygen":
        monitoring.append("Maintain near-bottom dissolved oxygen monitoring in vulnerable seasons, paired with chlorophyll, stratification, river/estuary input, and discharge-outlet observations.")
    elif theme == "heat":
        monitoring.append("Maintain warm-season SST and marine heatwave early-warning indicators for fisheries, aquaculture, and protected-area managers.")
    elif theme == "stratification":
        monitoring.append("Add repeated vertical profiles of temperature, salinity, density, oxygen, and chlorophyll in high-risk seasons.")
    elif theme == "dynamics":
        monitoring.append("Use fronts, eddies, upwelling, jets, and transport diagnostics to target adaptive sampling and hotspot surveillance.")
    elif theme == "watermass":
        monitoring.append("Track source-water and watermass boundaries where event exposure changes sharply across watermass classes.")
    else:
        monitoring.append("Keep the upstream diagnostic on a repeatable schedule so policy triggers can be compared across years.")

    return monitoring


def _policy_constraints_for_evidence(result: Dict[str, Any], entry: Dict[str, Any]) -> List[str]:
    constraints: List[str] = []
    support = str(entry.get("support_strength") or "").lower()
    output_type = str(entry.get("output_type") or "").strip()

    if "untestable" in support:
        constraints.append(f"{entry.get('label')} is not yet testable enough to justify strong policy attribution.")
    elif "limited" in support or "consistent" in support:
        constraints.append(f"{entry.get('label')} should be treated as screening evidence, not as standalone causal proof.")

    if output_type in {"event_detection_result", "event_statistics_result"}:
        constraints.append("Event-based recommendations depend on detection thresholds, minimum duration, minimum area, and sampling frequency.")
    if output_type in {"trend_result", "field_trend_result"}:
        constraints.append("Trend-based recommendations should account for window length, seasonal aliasing, and statistical significance.")
    if output_type in {"mechanism_score_result", "evidence_report_result"}:
        constraints.append("Mechanism evidence is policy-relevant for prioritization but should not be read as complete causal attribution.")

    return constraints


def _policy_default_monitoring_priorities(evidence_table: Sequence[Dict[str, Any]]) -> List[str]:
    if evidence_table:
        return [
            "Use the same spatial mask, depth convention, and seasonal window when updating the policy evidence package.",
        ]
    return [
        "Start with a compact indicator package: chlorophyll-a, SST, bottom oxygen, stratification, and event-frequency diagnostics.",
    ]


def _policy_default_constraints(
    evidence_table: Sequence[Dict[str, Any]],
    *,
    context_note: str,
) -> List[str]:
    constraints = [
        "Recommendations are evidence-bounded decision support, not a complete regulatory design or formal cost-benefit analysis.",
        "Do not extrapolate beyond the China Seas and western Pacific data domain unless additional data are supplied.",
    ]
    if not evidence_table:
        constraints.append("No upstream analysis result was supplied to the policy recommendation step.")
    if context_note.strip():
        constraints.append(context_note.strip())
    return constraints


def _policy_governance_notes(
    *,
    region_scope: str,
    policy_context: str,
    management_objective: str,
) -> List[str]:
    notes = [
        "Frame actions as adaptive management: update priorities when new events, trends, or mechanism evidence become available.",
    ]
    scope_text = f"{region_scope} {policy_context}".lower()
    if any(token in scope_text for token in ("south china sea", "east china sea", "yellow sea", "bohai", "taiwan strait", "western pacific")):
        notes.append("For transboundary or multi-jurisdictional waters, separate scientific evidence from jurisdictional claims and emphasize monitoring coordination, risk communication, and shared early warning.")
    if management_objective.strip():
        notes.append(f"Align recommendations with the stated management objective: {management_objective.strip()}.")
    return notes


def _policy_summary_sentence(
    *,
    priority_level: str,
    evidence_table: Sequence[Dict[str, Any]],
    management_objective: str,
) -> str:
    if not evidence_table:
        return "Policy guidance is evidence-limited because no upstream diagnostic result was supplied."

    strongest = sorted(
        evidence_table,
        key=lambda item: float(item.get("priority_score") or 0.0),
        reverse=True,
    )[:2]
    labels = [str(item.get("label") or "").strip() for item in strongest if str(item.get("label") or "").strip()]
    lead = f"Policy priority is {priority_level.lower()}"
    if labels:
        lead += " based mainly on " + " and ".join(labels)
    if management_objective.strip():
        lead += f" for {management_objective.strip()}"
    return lead + "."


def _policy_priority_level(evidence_table: Sequence[Dict[str, Any]]) -> str:
    if not evidence_table:
        return "Evidence-limited"
    max_score = max(float(item.get("priority_score") or 0.0) for item in evidence_table)
    if max_score >= 3.5:
        return "High"
    if max_score >= 2.0:
        return "Moderate"
    return "Routine"


def _policy_theme_from_result(result: Dict[str, Any], label: str) -> str:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    pieces = [
        label,
        str(result.get("event_type") or ""),
        str(result.get("top_mechanism") or ""),
        str(metadata.get("event_type") or ""),
        str(metadata.get("summary_mode") or ""),
        str(metadata.get("variable") or ""),
        str(metadata.get("source_variable") or ""),
        str(metadata.get("title") or ""),
    ]
    text = " ".join(piece.lower() for piece in pieces if piece)

    if any(token in text for token in ("chlorophyll", "chla", "bloom", "eutroph")):
        return "chlorophyll"
    if any(token in text for token in ("oxygen", "hypoxia", "low oxygen", "o2")):
        return "oxygen"
    if any(token in text for token in ("temperature", "sst", "heatwave", "heat")):
        return "heat"
    if any(token in text for token in ("stratification", "density", "mixed layer", "thermocline")):
        return "stratification"
    if any(token in text for token in ("front", "eddy", "upwelling", "jet", "meander", "advection", "transport", "current", "mesoscale")):
        return "dynamics"
    if "watermass" in text or "water mass" in text:
        return "watermass"
    if any(token in text for token in ("salinity", "salt", "fresh")):
        return "salinity"
    return "general"


def _policy_action_text(*, theme: str, signal: str) -> str:
    if theme == "chlorophyll":
        return "Prioritize nutrient-load management, bloom surveillance, and aquaculture risk communication in hotspots and seasons where chlorophyll-a or bloom exposure is elevated."
    if theme == "oxygen":
        return "Strengthen bottom-water oxygen early warning and reduce nutrient or organic loading, with attention to river and estuary inputs, discharge outlets, and outfalls where low-oxygen exposure is recurring or increasing."
    if theme == "heat":
        return "Prepare marine heat-risk advisories and adaptation measures for fisheries, aquaculture, and conservation areas during high-risk warm seasons."
    if theme == "stratification":
        return "Use stronger stratification as an early-warning condition and link vertical-profile monitoring to bloom and low-oxygen response planning."
    if theme == "dynamics":
        return "Use circulation, fronts, eddies, upwelling, and transport diagnostics to target patrols, adaptive sampling, and seasonal risk maps."
    if theme == "watermass":
        return "Manage risk by watermass pathway: monitor source-water shifts and prioritize boundaries where events concentrate."
    if theme == "salinity":
        return "Track freshwater influence and salinity shifts together with river discharge, nutrient inputs, and coastal ecosystem exposure."
    if "no events" in signal:
        return "Maintain baseline monitoring and revisit thresholds before concluding that the risk is absent."
    return "Use the diagnostic as a screening signal for targeted monitoring, periodic review, and low-regret management actions."


def _policy_risk_score(*, theme: str, direction: str, supported: bool) -> float:
    direction = direction.lower()
    worsening = False
    if theme in {"chlorophyll", "heat", "stratification"} and direction == "increasing":
        worsening = True
    if theme == "oxygen" and direction == "decreasing":
        worsening = True
    if theme in {"salinity", "dynamics", "watermass", "general"} and direction in {"increasing", "decreasing", "mixed"}:
        return 1.5 if supported else 1.0
    if worsening:
        return 3.0 if supported else 2.0
    if direction in {"increasing", "decreasing"}:
        return 1.5 if supported else 1.0
    return 0.5


def _policy_action_priority(score: float) -> str:
    if score >= 3.0:
        return "high"
    if score >= 1.5:
        return "near_term"
    return "routine"


def _policy_direction_from_value(value: Optional[float]) -> str:
    if value is None:
        return "trend unavailable"
    if value > 0:
        return "increasing"
    if value < 0:
        return "decreasing"
    return "stable"


def _policy_event_count(result: Dict[str, Any]) -> int:
    statistics = result.get("statistics") if isinstance(result.get("statistics"), dict) else {}
    for source in (result, statistics):
        count = source.get("total_count") if isinstance(source, dict) else None
        if isinstance(count, (int, np.integer)):
            return int(count)
        if isinstance(count, (float, np.floating)) and np.isfinite(count):
            return int(count)
    events = result.get("events")
    if isinstance(events, list):
        return len(events)
    return 0


def _humanize_policy_label(value: Any) -> str:
    text = str(value or "").strip().replace("_", " ")
    return " ".join(text.split()).title() if text else "Evidence Item"


def _dedupe_policy_actions(actions: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[str] = set()
    deduped: List[Dict[str, str]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_text = str(action.get("action") or "").strip()
        if not action_text or action_text in seen:
            continue
        seen.add(action_text)
        deduped.append({
            "theme": str(action.get("theme") or "general").strip(),
            "priority": str(action.get("priority") or "routine").strip(),
            "action_type": str(action.get("action_type") or "governance").strip(),
            "target": str(action.get("target") or "").strip(),
            "where_when": str(action.get("where_when") or "").strip(),
            "evidence_basis": str(action.get("evidence_basis") or action.get("rationale") or "").strip(),
            "action": action_text,
            "guardrail": str(action.get("guardrail") or "").strip(),
            "rationale": str(action.get("rationale") or "").strip(),
            "evidence_strength": str(action.get("evidence_strength") or "limited").strip(),
        })
    return deduped


def _resolve_density_field(
    data: Optional[xr.Dataset],
    temp: Optional[xr.DataArray],
    salt: Optional[xr.DataArray],
    density: Optional[xr.DataArray],
) -> xr.DataArray:
    if density is not None:
        return _unwrap_dataarray(density, fallback_name="density")

    dataset = _unwrap_dataset(data)
    if dataset is None:
        variables: Dict[str, xr.DataArray] = {}
        if temp is not None:
            variables["temp"] = _unwrap_dataarray(temp, fallback_name="temp")
        if salt is not None:
            variables["salt"] = _unwrap_dataarray(salt, fallback_name="salt")
        if not variables:
            raise ValueError("Density-related diagnostics require either density or temp/salt inputs.")
        dataset = xr.Dataset(variables)

    return compute_density(dataset)


def _subset_stability_inputs(
    *,
    data: Optional[xr.Dataset],
    temp: Optional[xr.DataArray],
    salt: Optional[xr.DataArray],
    density: Optional[xr.DataArray],
    lon_range: Optional[Tuple[float, float]],
    lat_range: Optional[Tuple[float, float]],
) -> tuple[Optional[xr.Dataset], Optional[xr.DataArray], Optional[xr.DataArray], Optional[xr.DataArray]]:
    return (
        _subset_horizontal_if_present(data, lon_range, lat_range),
        _subset_horizontal_if_present(temp, lon_range, lat_range),
        _subset_horizontal_if_present(salt, lon_range, lat_range),
        _subset_horizontal_if_present(density, lon_range, lat_range),
    )


def _subset_horizontal_if_present(
    value: Optional[xr.DataArray | xr.Dataset],
    lon_range: Optional[Tuple[float, float]],
    lat_range: Optional[Tuple[float, float]],
) -> Optional[xr.DataArray | xr.Dataset]:
    if value is None:
        return None
    if lon_range is None or lat_range is None:
        return value
    if "lon" not in value.coords or "lat" not in value.coords:
        return value
    return _subset_horizontal(value, lon_range, lat_range)


def _reduce_field_to_timeseries(
    field: xr.DataArray,
    lon_range: Optional[Tuple[float, float]],
    lat_range: Optional[Tuple[float, float]],
    depth_range: Optional[Tuple[float, float]],
    depth_aggregation: str,
    weighting: str,
    metadata_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if weighting == "volume_weighted":
        if depth_range is None:
            raise ValueError("volume_weighted reduction requires depth_range")
        result = compute_volume_weighted_mean(
            data=field,
            lon_range=lon_range,
            lat_range=lat_range,
            depth_range=depth_range,
        )
    else:
        result = compute_area_weighted_mean(
            data=field,
            lon_range=lon_range,
            lat_range=lat_range,
            depth_range=depth_range,
            depth_aggregation=depth_aggregation,
        )
    result.setdefault("metadata", {})
    result["metadata"].setdefault("variable", field.name or "unknown")
    result["metadata"].setdefault("units", field.attrs.get("units", ""))
    if metadata_updates:
        result["metadata"].update(metadata_updates)
    return result


def _mechanism_score_result(
    candidate_scores: Dict[str, float],
    top_mechanism: str,
    claim_strength: str,
    supporting_evidence: Sequence[str],
    conflicting_evidence: Sequence[str],
    times: Optional[Sequence[Any]] = None,
    values: Optional[Sequence[float]] = None,
    subregion_breakdown: Optional[Sequence[Dict[str, Any]]] = None,
    proxy_breakdowns: Optional[Sequence[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    result = {
        "output_type": "mechanism_score_result",
        "candidate_mechanisms": [
            {"name": name, "score": float(score)}
            for name, score in ranked
        ],
        "top_mechanism": top_mechanism,
        "claim_strength": claim_strength,
        "supporting_evidence": _dedupe_strings(supporting_evidence)[:6],
        "conflicting_evidence": _dedupe_strings(conflicting_evidence)[:4],
        "metadata": metadata or {},
    }
    if times is not None and values is not None:
        result["times"] = [str(value) for value in times]
        result["values"] = [float(value) if np.isfinite(value) else float("nan") for value in values]
    if subregion_breakdown is not None:
        result["subregion_breakdown"] = [
            _normalize_breakdown_record(item)
            for item in subregion_breakdown
            if isinstance(item, dict)
        ]
    if proxy_breakdowns is not None:
        result["proxy_breakdowns"] = [
            {
                **_normalize_breakdown_record(item),
                "subregion_breakdown": [
                    _normalize_breakdown_record(subitem)
                    for subitem in item.get("subregion_breakdown", [])
                    if isinstance(subitem, dict)
                ],
            }
            for item in proxy_breakdowns
            if isinstance(item, dict)
        ]
    return result


def _normalize_breakdown_record(item: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, (str, bool, int, np.integer)):
            normalized[key] = value
        elif isinstance(value, (float, np.floating)):
            normalized[key] = float(value)
        elif isinstance(value, list):
            normalized[key] = [
                float(entry) if isinstance(entry, (float, int, np.floating, np.integer)) else entry
                for entry in value
            ]
    return normalized


def _infer_mesoscale_proxy_name(field: xr.DataArray) -> str:
    candidates = [
        field.name,
        field.attrs.get("aggregation"),
        field.attrs.get("long_name"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        lowered = candidate.strip().lower()
        if not lowered:
            continue
        if "front_proximity" in lowered:
            return "front_proximity"
        if "eddy_influence" in lowered or (lowered.startswith("eddy") and "mask" in lowered):
            return "eddy_influence"
        if "tracer_gradient_alignment" in lowered or ("alignment" in lowered and "gradient" in lowered):
            return "gradient_alignment"
        if "flow_context" in lowered or "flow_structure_context" in lowered:
            return "flow_context"
    return "unknown_proxy"


def _normalize_subregion_grid(subregion_grid: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if subregion_grid is None:
        return (2, 2)
    if len(subregion_grid) != 2:
        raise ValueError("subregion_grid must contain exactly two integers")
    nx = int(subregion_grid[0])
    ny = int(subregion_grid[1])
    if nx < 1 or ny < 1:
        raise ValueError("subregion_grid dimensions must be positive integers")
    return (nx, ny)


def _build_rectangular_subregions(
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    grid: Tuple[int, int],
) -> List[Dict[str, Any]]:
    lon_min, lon_max = sorted((float(lon_range[0]), float(lon_range[1])))
    lat_min, lat_max = sorted((float(lat_range[0]), float(lat_range[1])))
    lon_edges = np.linspace(lon_min, lon_max, grid[0] + 1)
    lat_edges = np.linspace(lat_min, lat_max, grid[1] + 1)

    subregions: List[Dict[str, Any]] = []
    for row in range(grid[1]):
        for col in range(grid[0]):
            subregions.append({
                "subregion_id": f"r{row + 1}_c{col + 1}",
                "label": f"R{row + 1}C{col + 1}",
                "lon_range": (float(lon_edges[col]), float(lon_edges[col + 1])),
                "lat_range": (float(lat_edges[row]), float(lat_edges[row + 1])),
            })
    return subregions


def _compute_subregion_area_weight(
    field: xr.DataArray,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    depth_range: Optional[Tuple[float, float]],
    depth_aggregation: str,
) -> float:
    try:
        subset = _subset_horizontal(field, lon_range, lat_range)
    except ValueError:
        return 0.0

    subset = _aggregate_depth(subset, depth_range, depth_aggregation)
    subset = _normalize_to_timeseries_field(subset)
    weights = _horizontal_cell_area(subset["lon"].values, subset["lat"].values)
    representative = subset.mean(dim="time", skipna=True) if "time" in subset.dims else subset
    valid = np.isfinite(representative)
    effective_weights = xr.where(valid, weights, 0.0)
    return float(effective_weights.sum(skipna=True).item())


def _compute_event_background_contrast(
    values: np.ndarray,
    event_mask: np.ndarray,
) -> Tuple[float, float, float, float]:
    event_values = values[event_mask]
    background_values = values[~event_mask]
    event_values = event_values[np.isfinite(event_values)]
    background_values = background_values[np.isfinite(background_values)]
    all_values = values[np.isfinite(values)]
    if event_values.size == 0 or background_values.size == 0 or all_values.size == 0:
        raise ValueError("Event/background contrast requires both event and non-event samples")

    event_mean = float(np.nanmean(event_values))
    background_mean = float(np.nanmean(background_values))
    delta = event_mean - background_mean
    pooled = max(float(np.nanstd(all_values)), 1e-12)
    standardized = abs(delta) / pooled
    return event_mean, background_mean, delta, standardized


def _evidence_report_result(
    supported_claims: Sequence[str],
    limited_claims: Sequence[str],
    untestable_claims: Sequence[str],
    residual_or_uncertainty: Sequence[str],
    claim_strength: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "output_type": "evidence_report_result",
        "supported_claims": _dedupe_strings(supported_claims)[:6],
        "limited_claims": _dedupe_strings(limited_claims)[:6],
        "untestable_claims": _dedupe_strings(untestable_claims)[:6],
        "residual_or_uncertainty": _dedupe_strings(residual_or_uncertainty)[:6],
        "claim_strength": claim_strength,
        "metadata": metadata or {},
    }


def _environment_assessment_result(
    overall_verdict: str,
    overall_support_strength: str,
    branch_assessments: Sequence[Dict[str, Any]],
    supporting_evidence: Sequence[str],
    uncertainties: Sequence[str],
    overall_narrative: str = "",
    key_pressures: Sequence[str] = (),
    stabilizing_signals: Sequence[str] = (),
    monitoring_priorities: Sequence[str] = (),
    policy_recommendations: Sequence[str] = (),
    policy_action_matrix: Sequence[Dict[str, Any]] = (),
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "output_type": "environment_assessment_result",
        "overall_verdict": overall_verdict,
        "overall_support_strength": overall_support_strength,
        "overall_narrative": overall_narrative.strip(),
        "branch_assessments": [
            {
                "name": str(item.get("name") or "").strip(),
                "indicator_label": str(item.get("indicator_label") or "").strip(),
                "role": _normalize_environment_branch_role(item.get("role")),
                "evidence_kind": _normalize_environment_evidence_kind(item.get("evidence_kind")),
                "metric": _normalize_environment_metric(item.get("metric")),
                "direction": str(item.get("direction") or "").strip(),
                "support_strength": str(item.get("support_strength") or "").strip(),
                "summary": str(item.get("summary") or "").strip(),
                "output_type": str(item.get("output_type") or "").strip(),
                "score_contribution": float(item.get("score_contribution") or 0.0),
                "supporting_evidence": _normalize_string_list(item.get("supporting_evidence")),
                "uncertainties": _normalize_string_list(item.get("uncertainties")),
            }
            for item in branch_assessments
            if isinstance(item, dict)
        ],
        "key_pressures": _dedupe_strings(key_pressures)[:5],
        "stabilizing_signals": _dedupe_strings(stabilizing_signals)[:4],
        "monitoring_priorities": _dedupe_strings(monitoring_priorities)[:5],
        "policy_recommendations": _dedupe_strings(policy_recommendations)[:6],
        "policy_action_matrix": _dedupe_environment_policy_actions(policy_action_matrix)[:8],
        "supporting_evidence": _dedupe_strings(supporting_evidence)[:8],
        "uncertainties": _dedupe_strings(uncertainties)[:8],
        "metadata": metadata or {},
    }


def _unwrap_dataset(value: Any) -> Optional[xr.Dataset]:
    value = materialize_partitioned_xarray(value)

    if value is None:
        return None
    if isinstance(value, xr.Dataset):
        return value
    if isinstance(value, dict):
        data = value.get("data")
        if isinstance(data, xr.Dataset):
            return data
    return None


def _unwrap_dataarray(value: Any, fallback_name: str) -> xr.DataArray:
    value = materialize_partitioned_xarray(value)

    if isinstance(value, xr.DataArray):
        return value if value.name else value.rename(fallback_name)
    if isinstance(value, dict):
        data = value.get("data")
        if isinstance(data, xr.DataArray):
            return data if data.name else data.rename(fallback_name)
        if isinstance(data, xr.Dataset):
            if fallback_name in data:
                return data[fallback_name]
            if len(data.data_vars) == 1:
                name = list(data.data_vars)[0]
                return data[name].rename(fallback_name)
    raise ValueError(f"Expected a DataArray-like input for '{fallback_name}'")


def _unwrap_normalized_result(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValueError("Expected a normalized result dictionary")


def _assess_environment_health_branch(
    name: str,
    indicator_label: str,
    worse_when: str,
    result: Any,
    role: str = ENV_ROLE_PRIMARY_SUPPORT,
    evidence_kind: str = "",
    metric: str = "",
) -> Dict[str, Any]:
    role = _normalize_environment_branch_role(role)
    evidence_kind = _normalize_environment_evidence_kind(evidence_kind)
    metric = _normalize_environment_metric(metric)
    try:
        normalized = _unwrap_normalized_result(result)
    except ValueError:
        return _environment_branch_assessment(
            name=name,
            indicator_label=indicator_label,
            role=role,
            evidence_kind=evidence_kind,
            metric=metric,
            direction=ENV_DIRECTION_UNTESTABLE,
            support_strength=ENV_SUPPORT_UNTESTABLE,
            output_type="unknown",
            summary=f"{indicator_label} could not be evaluated because no usable result was supplied.",
            supporting_evidence=[],
            uncertainties=[f"{indicator_label} result is missing or malformed."],
        )

    output_type = str(normalized.get("output_type") or "").strip()
    inferred_kind = evidence_kind or _infer_environment_evidence_kind(output_type)
    if output_type == "trend_result" and inferred_kind == ENV_EVIDENCE_TREND:
        return _assess_environment_trend_branch(
            name=name,
            indicator_label=indicator_label,
            worse_when=worse_when,
            result=normalized,
            role=role,
            evidence_kind=inferred_kind,
            metric=metric,
        )
    if output_type == "event_statistics_result" and inferred_kind == ENV_EVIDENCE_EVENT_STATISTICS:
        return _assess_environment_event_statistics_branch(
            name=name,
            indicator_label=indicator_label,
            worse_when=worse_when,
            result=normalized,
            role=role,
            evidence_kind=inferred_kind,
            metric=metric,
        )
    if output_type == "event_detection_result" and inferred_kind == ENV_EVIDENCE_EVENT_DETECTION:
        return _assess_environment_event_detection_branch(
            name=name,
            indicator_label=indicator_label,
            worse_when=worse_when,
            result=normalized,
            role=role,
            evidence_kind=inferred_kind,
            metric=metric,
        )
    if output_type == "event_spatial_distribution_result" and inferred_kind == ENV_EVIDENCE_EVENT_SPATIAL_DISTRIBUTION:
        return _assess_environment_event_spatial_distribution_branch(
            name=name,
            indicator_label=indicator_label,
            worse_when=worse_when,
            result=normalized,
            role=role,
            evidence_kind=inferred_kind,
            metric=metric,
        )
    if output_type == "spatial_field_result" and inferred_kind in {
        ENV_EVIDENCE_SPATIAL_FIELD,
        ENV_EVIDENCE_EVENT_SPATIAL_FIELD,
    }:
        return _assess_environment_spatial_field_branch(
            name=name,
            indicator_label=indicator_label,
            worse_when=worse_when,
            result=normalized,
            role=role,
            evidence_kind=inferred_kind,
            metric=metric,
        )

    return _environment_branch_assessment(
        name=name,
        indicator_label=indicator_label,
        role=role,
        evidence_kind=inferred_kind,
        metric=metric,
        direction=ENV_DIRECTION_UNTESTABLE,
        support_strength=ENV_SUPPORT_UNTESTABLE,
        output_type=output_type or "unknown",
        summary=f"{indicator_label} is not yet supported as an environment-health branch.",
        supporting_evidence=[],
        uncertainties=[f"Unsupported branch result type: {output_type or 'unknown'}."],
    )


def _assess_environment_trend_branch(
    name: str,
    indicator_label: str,
    worse_when: str,
    result: Dict[str, Any],
    role: str = ENV_ROLE_PRIMARY_SUPPORT,
    evidence_kind: str = ENV_EVIDENCE_TREND,
    metric: str = "",
) -> Dict[str, Any]:
    slope = _safe_float(result.get("slope"))
    fitted_change = _safe_float(result.get("fitted_change_over_period"))
    p_value = _safe_float(result.get("p_value"))
    direction_hint = str(result.get("trend_direction") or "").strip().lower()
    is_significant = result.get("is_significant")

    change_sign = 0
    if slope is not None and slope > 0:
        change_sign = 1
    elif slope is not None and slope < 0:
        change_sign = -1
    elif fitted_change is not None and fitted_change > 0:
        change_sign = 1
    elif fitted_change is not None and fitted_change < 0:
        change_sign = -1
    elif direction_hint == "positive":
        change_sign = 1
    elif direction_hint == "negative":
        change_sign = -1

    if change_sign == 0:
        direction = ENV_DIRECTION_STABLE
    elif worse_when == "decrease":
        direction = ENV_DIRECTION_DETERIORATING if change_sign < 0 else ENV_DIRECTION_IMPROVING
    else:
        direction = ENV_DIRECTION_DETERIORATING if change_sign > 0 else ENV_DIRECTION_IMPROVING

    if slope is None and fitted_change is None and not direction_hint:
        support_strength = ENV_SUPPORT_UNTESTABLE
    elif is_significant is True:
        support_strength = ENV_SUPPORT_SUPPORTED
    else:
        support_strength = ENV_SUPPORT_LIMITED

    summary_parts = [indicator_label]
    if direction == ENV_DIRECTION_STABLE:
        summary_parts.append("shows little directional change")
    elif direction == ENV_DIRECTION_DETERIORATING:
        summary_parts.append("moves in the worsening direction")
    else:
        summary_parts.append("moves in the improving direction")
    if slope is not None:
        summary_parts.append(f"(slope {_format_number(slope)}")
        if p_value is not None:
            summary_parts[-1] += f", p={_format_number(p_value)})"
        else:
            summary_parts[-1] += ")"
    elif fitted_change is not None:
        summary_parts.append(f"(fitted change {_format_number(fitted_change)})")

    supporting = [" ".join(summary_parts).strip() + "."]
    uncertainties: List[str] = []
    if support_strength == ENV_SUPPORT_LIMITED:
        uncertainties.append(f"{indicator_label} trend is detectable but not statistically significant.")
    if support_strength == ENV_SUPPORT_UNTESTABLE:
        uncertainties.append(f"{indicator_label} trend lacks the required slope or fitted-change diagnostics.")

    return _environment_branch_assessment(
        name=name,
        indicator_label=indicator_label,
        role=role,
        evidence_kind=evidence_kind,
        metric=metric,
        direction=direction,
        support_strength=support_strength,
        output_type="trend_result",
        summary=supporting[0],
        supporting_evidence=supporting,
        uncertainties=uncertainties,
    )


def _assess_environment_event_statistics_branch(
    name: str,
    indicator_label: str,
    worse_when: str,
    result: Dict[str, Any],
    role: str = ENV_ROLE_PRIMARY_SUPPORT,
    evidence_kind: str = ENV_EVIDENCE_EVENT_STATISTICS,
    metric: str = "",
) -> Dict[str, Any]:
    if str(result.get("group_by") or "").strip().lower() != "year":
        return _environment_branch_assessment(
            name=name,
            indicator_label=indicator_label,
            role=role,
            evidence_kind=evidence_kind,
            metric=metric,
            direction=ENV_DIRECTION_UNTESTABLE,
            support_strength=ENV_SUPPORT_UNTESTABLE,
            output_type="event_statistics_result",
            summary=f"{indicator_label} could not be compared because the statistics were not grouped by year.",
            supporting_evidence=[],
            uncertainties=[f"{indicator_label} requires `group_by=\"year\"` for early-versus-late comparison."],
        )

    groups = result.get("groups")
    if not isinstance(groups, dict) or not groups:
        return _environment_branch_assessment(
            name=name,
            indicator_label=indicator_label,
            role=role,
            evidence_kind=evidence_kind,
            metric=metric,
            direction=ENV_DIRECTION_UNTESTABLE,
            support_strength=ENV_SUPPORT_UNTESTABLE,
            output_type="event_statistics_result",
            summary=f"{indicator_label} could not be compared because no yearly event counts were available.",
            supporting_evidence=[],
            uncertainties=[f"{indicator_label} returned no yearly event groups."],
        )

    ordered_counts: List[Tuple[str, float]] = []
    for label, payload in groups.items():
        if not isinstance(payload, dict):
            continue
        count = _safe_float(payload.get("count"))
        if count is None:
            continue
        ordered_counts.append((str(label), float(count)))

    if len(ordered_counts) < 2:
        return _environment_branch_assessment(
            name=name,
            indicator_label=indicator_label,
            role=role,
            evidence_kind=evidence_kind,
            metric=metric,
            direction=ENV_DIRECTION_UNTESTABLE,
            support_strength=ENV_SUPPORT_UNTESTABLE,
            output_type="event_statistics_result",
            summary=f"{indicator_label} could not be compared because fewer than two yearly bins were available.",
            supporting_evidence=[],
            uncertainties=[f"{indicator_label} needs at least two yearly event-count bins."],
        )

    ordered_counts.sort(key=lambda item: _sortable_event_group(item[0]))
    split_index = len(ordered_counts) // 2
    earlier = ordered_counts[:split_index]
    later = ordered_counts[split_index:]
    if not earlier or not later:
        return _environment_branch_assessment(
            name=name,
            indicator_label=indicator_label,
            role=role,
            evidence_kind=evidence_kind,
            metric=metric,
            direction=ENV_DIRECTION_UNTESTABLE,
            support_strength=ENV_SUPPORT_UNTESTABLE,
            output_type="event_statistics_result",
            summary=f"{indicator_label} could not be split into earlier and later periods.",
            supporting_evidence=[],
            uncertainties=[f"{indicator_label} yearly groups did not support a half-window comparison."],
        )

    earlier_mean = float(np.mean([count for _, count in earlier]))
    later_mean = float(np.mean([count for _, count in later]))
    baseline = max(abs(earlier_mean), 1.0)
    relative_change_pct = float((later_mean - earlier_mean) / baseline * 100.0)

    if abs(relative_change_pct) < 10.0:
        direction = ENV_DIRECTION_STABLE
        support_strength = ENV_SUPPORT_UNTESTABLE
    else:
        if worse_when == "decrease":
            direction = ENV_DIRECTION_DETERIORATING if later_mean < earlier_mean else ENV_DIRECTION_IMPROVING
        else:
            direction = ENV_DIRECTION_DETERIORATING if later_mean > earlier_mean else ENV_DIRECTION_IMPROVING

        if len(ordered_counts) >= 4 and abs(relative_change_pct) >= 25.0:
            support_strength = ENV_SUPPORT_SUPPORTED
        else:
            support_strength = ENV_SUPPORT_LIMITED

    supporting = [
        (
            f"{indicator_label} changes from an earlier-half mean of {_format_number(earlier_mean)} "
            f"to a later-half mean of {_format_number(later_mean)} "
            f"({_format_number(relative_change_pct)}% relative change)."
        )
    ]
    uncertainties: List[str] = []
    if len(ordered_counts) < 4 and support_strength != ENV_SUPPORT_UNTESTABLE:
        uncertainties.append(
            f"{indicator_label} is based on only {len(ordered_counts)} yearly bins, so support remains limited."
        )
    if support_strength == ENV_SUPPORT_UNTESTABLE:
        uncertainties.append(f"{indicator_label} shows little early-versus-late change in yearly event counts.")

    return _environment_branch_assessment(
        name=name,
        indicator_label=indicator_label,
        role=role,
        evidence_kind=evidence_kind,
        metric=metric,
        direction=direction,
        support_strength=support_strength,
        output_type="event_statistics_result",
        summary=supporting[0],
        supporting_evidence=supporting,
        uncertainties=uncertainties,
    )


def _assess_environment_event_detection_branch(
    name: str,
    indicator_label: str,
    worse_when: str,
    result: Dict[str, Any],
    role: str = ENV_ROLE_PRIMARY_SUPPORT,
    evidence_kind: str = ENV_EVIDENCE_EVENT_DETECTION,
    metric: str = "",
) -> Dict[str, Any]:
    count = _event_count_from_result(result)
    if count is None:
        return _environment_branch_assessment(
            name=name,
            indicator_label=indicator_label,
            role=role,
            evidence_kind=evidence_kind,
            metric=metric,
            direction=ENV_DIRECTION_UNTESTABLE,
            support_strength=ENV_SUPPORT_UNTESTABLE,
            output_type="event_detection_result",
            summary=f"{indicator_label} could not be evaluated because event counts were unavailable.",
            supporting_evidence=[],
            uncertainties=[f"{indicator_label} event detection lacks a usable event count."],
        )

    if count > 0:
        direction = ENV_DIRECTION_DETERIORATING if worse_when in {"presence", "increase"} else ENV_DIRECTION_IMPROVING
        support_strength = ENV_SUPPORT_SUPPORTED
    else:
        direction = ENV_DIRECTION_STABLE
        support_strength = ENV_SUPPORT_LIMITED

    supporting = [f"{indicator_label} detected {count} event(s) under the configured threshold and persistence rules."]
    uncertainties: List[str] = []
    if count == 0:
        uncertainties.append(f"{indicator_label} absence is threshold-dependent and should be treated as screening evidence.")

    return _environment_branch_assessment(
        name=name,
        indicator_label=indicator_label,
        role=role,
        evidence_kind=evidence_kind,
        metric=metric,
        direction=direction,
        support_strength=support_strength,
        output_type="event_detection_result",
        summary=supporting[0],
        supporting_evidence=supporting,
        uncertainties=uncertainties,
    )


def _assess_environment_event_spatial_distribution_branch(
    name: str,
    indicator_label: str,
    worse_when: str,
    result: Dict[str, Any],
    role: str = ENV_ROLE_PRIMARY_SUPPORT,
    evidence_kind: str = ENV_EVIDENCE_EVENT_SPATIAL_DISTRIBUTION,
    metric: str = "",
) -> Dict[str, Any]:
    count = _safe_float(result.get("event_count"))
    if count is None:
        statistics = result.get("statistics") if isinstance(result.get("statistics"), dict) else {}
        count = _safe_float(statistics.get("event_count") or statistics.get("total_count"))
    if count is None:
        return _environment_branch_assessment(
            name=name,
            indicator_label=indicator_label,
            role=role,
            evidence_kind=evidence_kind,
            metric=metric,
            direction=ENV_DIRECTION_UNTESTABLE,
            support_strength=ENV_SUPPORT_UNTESTABLE,
            output_type="event_spatial_distribution_result",
            summary=f"{indicator_label} spatial distribution could not be evaluated because event counts were unavailable.",
            supporting_evidence=[],
            uncertainties=[f"{indicator_label} spatial distribution lacks a usable event count."],
        )

    if count > 0:
        direction = ENV_DIRECTION_DETERIORATING if worse_when in {"presence", "increase"} else ENV_DIRECTION_IMPROVING
        support_strength = ENV_SUPPORT_SUPPORTED
    else:
        direction = ENV_DIRECTION_STABLE
        support_strength = ENV_SUPPORT_LIMITED

    supporting = [f"{indicator_label} spatial distribution contains {_format_number(count)} event(s)."]
    return _environment_branch_assessment(
        name=name,
        indicator_label=indicator_label,
        role=role,
        evidence_kind=evidence_kind,
        metric=metric,
        direction=direction,
        support_strength=support_strength,
        output_type="event_spatial_distribution_result",
        summary=supporting[0],
        supporting_evidence=supporting,
        uncertainties=[] if count > 0 else [f"{indicator_label} absence is threshold-dependent."],
    )


def _assess_environment_spatial_field_branch(
    name: str,
    indicator_label: str,
    worse_when: str,
    result: Dict[str, Any],
    role: str = ENV_ROLE_PRIMARY_SUPPORT,
    evidence_kind: str = ENV_EVIDENCE_SPATIAL_FIELD,
    metric: str = "",
) -> Dict[str, Any]:
    metric = _normalize_environment_metric(metric)
    if metric not in ENV_SPATIAL_METRICS:
        return _environment_branch_assessment(
            name=name,
            indicator_label=indicator_label,
            role=role,
            evidence_kind=evidence_kind,
            metric=metric,
            direction=ENV_DIRECTION_UNTESTABLE,
            support_strength=ENV_SUPPORT_UNTESTABLE,
            output_type="spatial_field_result",
            summary=f"{indicator_label} spatial evidence is not scored because no valid metric was configured.",
            supporting_evidence=[],
            uncertainties=[f"{indicator_label} requires metric in {sorted(ENV_SPATIAL_METRICS)}."],
        )

    values = _spatial_values_from_result(result)
    if values.size == 0:
        return _environment_branch_assessment(
            name=name,
            indicator_label=indicator_label,
            role=role,
            evidence_kind=evidence_kind,
            metric=metric,
            direction=ENV_DIRECTION_UNTESTABLE,
            support_strength=ENV_SUPPORT_UNTESTABLE,
            output_type="spatial_field_result",
            summary=f"{indicator_label} spatial evidence contains no finite values.",
            supporting_evidence=[],
            uncertainties=[f"{indicator_label} has no finite spatial values after masking."],
        )

    metric_value = _spatial_metric_value(values, metric)
    direction, support_strength = _environment_direction_from_metric(
        metric_value=metric_value,
        worse_when=worse_when,
        evidence_kind=evidence_kind,
    )
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    title = str(metadata.get("title") or indicator_label).strip()
    units = str(metadata.get("units") or "").strip()
    unit_text = f" {units}" if units else ""
    supporting = [
        f"{title} spatial evidence has {metric.replace('_', ' ')} {_format_number(metric_value)}{unit_text}."
    ]
    uncertainties: List[str] = []
    if support_strength == ENV_SUPPORT_LIMITED:
        uncertainties.append(f"{indicator_label} is spatial exposure evidence and does not by itself establish a trend or causal mechanism.")

    return _environment_branch_assessment(
        name=name,
        indicator_label=indicator_label,
        role=role,
        evidence_kind=evidence_kind,
        metric=metric,
        direction=direction,
        support_strength=support_strength,
        output_type="spatial_field_result",
        summary=supporting[0],
        supporting_evidence=supporting,
        uncertainties=uncertainties,
    )


def _infer_environment_evidence_kind(output_type: str) -> str:
    if output_type == "trend_result":
        return ENV_EVIDENCE_TREND
    if output_type == "event_statistics_result":
        return ENV_EVIDENCE_EVENT_STATISTICS
    return ""


def _normalize_environment_branch_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {
        ENV_ROLE_PRIMARY_ENDPOINT,
        ENV_ROLE_PRIMARY_SUPPORT,
        ENV_ROLE_RISK_FACTOR,
        ENV_ROLE_AUXILIARY_CONTEXT,
    }:
        return role
    return ENV_ROLE_PRIMARY_SUPPORT


def _normalize_environment_evidence_kind(value: Any) -> str:
    evidence_kind = str(value or "").strip().lower()
    aliases = {
        "event_count": ENV_EVIDENCE_EVENT_STATISTICS,
        "event_counts": ENV_EVIDENCE_EVENT_STATISTICS,
        "event_summary_map": ENV_EVIDENCE_EVENT_SPATIAL_FIELD,
        "event_spatial_map": ENV_EVIDENCE_EVENT_SPATIAL_FIELD,
        "spatial": ENV_EVIDENCE_SPATIAL_FIELD,
    }
    evidence_kind = aliases.get(evidence_kind, evidence_kind)
    if evidence_kind in {
        ENV_EVIDENCE_TREND,
        ENV_EVIDENCE_EVENT_STATISTICS,
        ENV_EVIDENCE_EVENT_DETECTION,
        ENV_EVIDENCE_SPATIAL_FIELD,
        ENV_EVIDENCE_EVENT_SPATIAL_FIELD,
        ENV_EVIDENCE_EVENT_SPATIAL_DISTRIBUTION,
    }:
        return evidence_kind
    return ""


def _normalize_environment_metric(value: Any) -> str:
    metric = str(value or "").strip().lower()
    aliases = {
        "sum": "total",
        "count_fraction": "nonzero_fraction",
        "presence_fraction": "nonzero_fraction",
    }
    return aliases.get(metric, metric)


def _spatial_values_from_result(result: Dict[str, Any]) -> np.ndarray:
    data = result.get("data")
    if isinstance(data, xr.DataArray):
        raw = data.values
    else:
        raw = result.get("values", [])
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    finite = values[np.isfinite(values)]
    return finite.reshape(-1)


def _spatial_metric_value(values: np.ndarray, metric: str) -> float:
    if metric == "mean":
        return float(np.nanmean(values))
    if metric == "max":
        return float(np.nanmax(values))
    if metric == "total":
        return float(np.nansum(values))
    if metric == "positive_fraction":
        return float(np.count_nonzero(values > 0) / values.size)
    if metric == "nonzero_fraction":
        return float(np.count_nonzero(np.abs(values) > 0) / values.size)
    raise ValueError(f"Unsupported spatial metric: {metric}")


def _environment_direction_from_metric(
    *,
    metric_value: float,
    worse_when: str,
    evidence_kind: str,
) -> Tuple[str, str]:
    if not np.isfinite(metric_value):
        return ENV_DIRECTION_UNTESTABLE, ENV_SUPPORT_UNTESTABLE
    if worse_when == "presence":
        if metric_value > 0:
            return ENV_DIRECTION_DETERIORATING, ENV_SUPPORT_SUPPORTED
        return ENV_DIRECTION_STABLE, ENV_SUPPORT_LIMITED
    if abs(metric_value) <= 0:
        return ENV_DIRECTION_STABLE, ENV_SUPPORT_LIMITED
    if worse_when == "decrease":
        direction = ENV_DIRECTION_DETERIORATING if metric_value < 0 else ENV_DIRECTION_IMPROVING
    else:
        direction = ENV_DIRECTION_DETERIORATING if metric_value > 0 else ENV_DIRECTION_IMPROVING
    support = ENV_SUPPORT_LIMITED if evidence_kind == ENV_EVIDENCE_SPATIAL_FIELD else ENV_SUPPORT_SUPPORTED
    return direction, support


def _event_count_from_result(result: Dict[str, Any]) -> Optional[int]:
    statistics = result.get("statistics") if isinstance(result.get("statistics"), dict) else {}
    for source in (result, statistics):
        count = source.get("total_count") if isinstance(source, dict) else None
        if isinstance(count, (int, np.integer)):
            return int(count)
        if isinstance(count, (float, np.floating)) and np.isfinite(count):
            return int(count)
    events = result.get("events")
    if isinstance(events, list):
        return len(events)
    return None


def _environment_key_pressures(branch_assessments: Sequence[Dict[str, Any]]) -> List[str]:
    pressures = [
        branch for branch in branch_assessments
        if isinstance(branch, dict)
        and branch.get("role") != ENV_ROLE_AUXILIARY_CONTEXT
        and branch.get("direction") == ENV_DIRECTION_DETERIORATING
        and branch.get("support_strength") in {ENV_SUPPORT_SUPPORTED, ENV_SUPPORT_LIMITED}
    ]
    ordered = sorted(pressures, key=_environment_branch_priority, reverse=True)
    return [
        f"{str(item.get('indicator_label') or item.get('name') or 'Indicator').strip()}: {str(item.get('summary') or '').strip()}"
        for item in ordered[:4]
    ]


def _environment_stabilizing_signals(branch_assessments: Sequence[Dict[str, Any]]) -> List[str]:
    stabilizing = [
        branch for branch in branch_assessments
        if isinstance(branch, dict)
        and branch.get("role") != ENV_ROLE_AUXILIARY_CONTEXT
        and branch.get("direction") == ENV_DIRECTION_IMPROVING
        and branch.get("support_strength") in {ENV_SUPPORT_SUPPORTED, ENV_SUPPORT_LIMITED}
    ]
    ordered = sorted(stabilizing, key=_environment_branch_priority, reverse=True)
    return [
        f"{str(item.get('indicator_label') or item.get('name') or 'Indicator').strip()}: {str(item.get('summary') or '').strip()}"
        for item in ordered[:3]
    ]


def _environment_monitoring_priorities(
    branch_assessments: Sequence[Dict[str, Any]],
    *,
    overall_verdict: str,
    overall_support_strength: str,
) -> List[str]:
    priorities: List[str] = []
    watch_list = sorted(
        [
            branch for branch in branch_assessments
            if isinstance(branch, dict)
            and branch.get("role") != ENV_ROLE_AUXILIARY_CONTEXT
            and branch.get("support_strength") in {ENV_SUPPORT_LIMITED, ENV_SUPPORT_UNTESTABLE}
        ],
        key=_environment_branch_priority,
        reverse=True,
    )
    for branch in watch_list[:3]:
        label = str(branch.get("indicator_label") or branch.get("name") or "Indicator").strip()
        direction = str(branch.get("direction") or ENV_DIRECTION_UNTESTABLE).strip()
        if branch.get("support_strength") == ENV_SUPPORT_UNTESTABLE:
            priorities.append(f"Improve monitoring for {label.lower()} because the current window does not yet support a firm directional assessment.")
        elif direction == ENV_DIRECTION_DETERIORATING:
            priorities.append(f"Track {label.lower()} more closely because it points toward worsening conditions but the current evidence remains limited.")
        else:
            priorities.append(f"Maintain closer surveillance of {label.lower()} to confirm whether the current weak signal persists.")

    deteriorating_branches = sorted(
        [
            branch for branch in branch_assessments
            if isinstance(branch, dict)
            and branch.get("role") != ENV_ROLE_AUXILIARY_CONTEXT
            and branch.get("direction") == ENV_DIRECTION_DETERIORATING
            and branch.get("support_strength") in {ENV_SUPPORT_SUPPORTED, ENV_SUPPORT_LIMITED}
        ],
        key=_environment_branch_priority,
        reverse=True,
    )
    for branch in deteriorating_branches[:3]:
        monitoring = _monitoring_guidance_for_environment_branch(branch)
        if monitoring:
            priorities.append(monitoring)

    if not priorities:
        if overall_verdict == "Deteriorating":
            priorities.append("Sustain routine cross-indicator monitoring so the worsening signal can be localized in season and subregion.")
        elif overall_verdict == "Improving":
            priorities.append("Maintain baseline monitoring to verify that the improving signal remains durable rather than temporary.")
        else:
            priorities.append("Preserve baseline monitoring across the selected indicators because the current window does not show a strong net directional shift.")

    if overall_support_strength == ENV_SUPPORT_UNTESTABLE:
        priorities.append("Prioritize denser time coverage and more complete indicator sampling before making high-cost management changes.")

    return priorities[:4]


def _environment_policy_action_matrix(
    branch_assessments: Sequence[Dict[str, Any]],
    *,
    overall_verdict: str,
    overall_support_strength: str,
) -> List[Dict[str, str]]:
    branches = [branch for branch in branch_assessments if isinstance(branch, dict)]
    oxygen_branches = [
        branch for branch in branches
        if _environment_branch_matches(branch, ("oxygen", "hypoxia", "hypoxic"))
        and branch.get("direction") == ENV_DIRECTION_DETERIORATING
        and branch.get("support_strength") in {ENV_SUPPORT_SUPPORTED, ENV_SUPPORT_LIMITED}
    ]
    oxygen_branches = sorted(oxygen_branches, key=_environment_branch_priority, reverse=True)
    risk_branches = [
        branch for branch in branches
        if (
            _environment_branch_matches(branch, ("temperature", "sst", "heat", "stratification"))
            and branch.get("direction") == ENV_DIRECTION_DETERIORATING
            and branch.get("support_strength") in {ENV_SUPPORT_SUPPORTED, ENV_SUPPORT_LIMITED}
        )
    ]
    risk_branches = sorted(risk_branches, key=_environment_branch_priority, reverse=True)
    auxiliary_chlorophyll = [
        branch for branch in branches
        if branch.get("role") == ENV_ROLE_AUXILIARY_CONTEXT
        and _environment_branch_matches(branch, ("chlorophyll", "eutroph", "bloom"))
        and branch.get("support_strength") in {ENV_SUPPORT_SUPPORTED, ENV_SUPPORT_LIMITED}
    ]

    actions: List[Dict[str, str]] = []
    if oxygen_branches:
        oxygen = oxygen_branches[0]
        oxygen_basis = _environment_action_evidence_basis(oxygen)
        risk_basis = "; ".join(_environment_action_evidence_basis(branch) for branch in risk_branches[:2])
        combined_basis = oxygen_basis if not risk_basis else f"{oxygen_basis}; risk timing context: {risk_basis}"
        priority = _environment_policy_priority(oxygen)

        actions.extend([
            _environment_policy_action(
                priority=priority,
                action_type="monitoring",
                target="bottom oxygen hotspots",
                where_when="recurrent burden hotspots and warm or stratified seasons",
                evidence_basis=oxygen_basis,
                action="Maintain targeted near-bottom dissolved oxygen monitoring, including vertical profiles and repeat stations around the strongest oxygen-deficit hotspots.",
                guardrail=_environment_action_guardrail(oxygen),
                theme="oxygen",
            ),
            _environment_policy_action(
                priority=priority,
                action_type="source_control",
                target="nutrient and organic loading",
                where_when="catchments and coastal waters upstream of recurrent low-oxygen areas",
                evidence_basis=oxygen_basis,
                action="Prioritize low-regret nutrient or organic loading reduction where recurrent bottom hypoxia burden overlaps plausible coastal source pathways.",
                guardrail="Use the oxygen evidence for prioritization and screening; it is not standalone causal proof of any single nutrient or organic source.",
                theme="oxygen",
            ),
            _environment_policy_action(
                priority=priority,
                action_type="river_estuary",
                target="river and estuary inputs",
                where_when="river-plume and estuary discharge windows before and during hypoxia-prone seasons",
                evidence_basis=combined_basis,
                action="Screen river and estuary inputs for nutrient, organic matter, and freshwater-stratification pulses that can precondition bottom oxygen loss.",
                guardrail="Treat river and estuary screening as source-pathway triage unless paired with load observations and hydrodynamic attribution.",
                theme="oxygen",
            ),
            _environment_policy_action(
                priority=priority,
                action_type="discharge_outlet",
                target="discharge outlets and outfalls",
                where_when="outlets near oxygen-deficit hotspots and during low-mixing periods",
                evidence_basis=oxygen_basis,
                action="Review discharge outlets, outfalls, and timing of high-organic or high-nutrient releases near recurring bottom-oxygen stress zones.",
                guardrail="Use this as a compliance and siting review trigger, not as proof that an outlet caused the observed hypoxia.",
                theme="oxygen",
            ),
            _environment_policy_action(
                priority=priority,
                action_type="seasonal_management",
                target="hypoxia early-warning window",
                where_when="warm, stratified, or weak-exchange seasons",
                evidence_basis=combined_basis,
                action="Use SST and stratification risk factors to time seasonal advisories, field sampling, and response readiness before bottom oxygen reaches ecological thresholds.",
                guardrail="Temperature and stratification indicate vulnerability and timing; they should not be used alone to assign pollution-source responsibility.",
                theme="oxygen",
            ),
            _environment_policy_action(
                priority=priority,
                action_type="coastal_planning",
                target="coastal management zones",
                where_when="persistent or recurrent hypoxia-burden hotspots",
                evidence_basis=oxygen_basis,
                action="Prioritize recurrent burden hotspots for coastal management zoning, restoration targeting, and cumulative-impact review.",
                guardrail=_environment_action_guardrail(oxygen),
                theme="oxygen",
            ),
        ])

    for branch in risk_branches:
        label = _environment_branch_label(branch).lower()
        if "stratification" in label:
            actions.append(_environment_policy_action(
                priority=_environment_policy_priority(branch),
                action_type="seasonal_management",
                target="stratified water-column risk windows",
                where_when="periods with stronger density stratification",
                evidence_basis=_environment_action_evidence_basis(branch),
                action="Use stronger stratification as an early-warning condition for oxygen stress, sampling design, and response timing.",
                guardrail="Stratification is a vulnerability signal; pair it with oxygen and load evidence before adopting source-specific controls.",
                theme="stratification",
            ))
        elif "temperature" in label or "sst" in label or "heat" in label:
            actions.append(_environment_policy_action(
                priority=_environment_policy_priority(branch),
                action_type="seasonal_management",
                target="warm-season exposure window",
                where_when="months or subregions with increasing SST or heat stress",
                evidence_basis=_environment_action_evidence_basis(branch),
                action="Use SST warming as a seasonal early-warning layer for oxygen-risk communication and adaptive sampling.",
                guardrail="SST supports timing and vulnerability assessment, not direct attribution of hypoxia to local pollution sources.",
                theme="heat",
            ))

    for branch in auxiliary_chlorophyll[:1]:
        actions.append(_environment_policy_action(
            priority="routine",
            action_type="governance",
            target="chlorophyll and eutrophication screening context",
            where_when="same region and season as oxygen-risk assessment",
            evidence_basis=_environment_action_evidence_basis(branch),
            action="Use chlorophyll or eutrophication information as supporting context for oxygen-risk triage, not as the primary evidence branch.",
            guardrail="Auxiliary chlorophyll evidence can strengthen screening hypotheses but should not by itself justify nutrient-control conclusions.",
            theme="chlorophyll",
        ))

    if not actions:
        if overall_verdict == "Deteriorating":
            actions.append(_environment_policy_action(
                priority="near_term",
                action_type="governance",
                target="integrated coastal management",
                where_when="selected assessment region and future update windows",
                evidence_basis=f"overall verdict: {overall_verdict}, support: {overall_support_strength}",
                action="Use the assessment as a precautionary screening package and update management priorities as stronger branch-level evidence becomes available.",
                guardrail=_policy_guardrail_for_strength(overall_support_strength),
                theme="general",
            ))
        else:
            actions.append(_environment_policy_action(
                priority="routine",
                action_type="monitoring",
                target="baseline environmental-health indicators",
                where_when="selected assessment region and repeated seasonal updates",
                evidence_basis=f"overall verdict: {overall_verdict}, support: {overall_support_strength}",
                action="Maintain a stable baseline indicator package so future deterioration or recovery can be detected consistently.",
                guardrail=_policy_guardrail_for_strength(overall_support_strength),
                theme="general",
            ))

    return _dedupe_environment_policy_actions(actions)[:8]


def _environment_policy_action(
    *,
    priority: str,
    action_type: str,
    target: str,
    where_when: str,
    evidence_basis: str,
    action: str,
    guardrail: str,
    theme: str,
) -> Dict[str, str]:
    return {
        "priority": priority,
        "action_type": action_type,
        "target": target,
        "where_when": where_when,
        "evidence_basis": evidence_basis,
        "action": action,
        "guardrail": guardrail,
        "theme": theme,
    }


def _environment_branch_label(branch: Dict[str, Any]) -> str:
    return str(branch.get("indicator_label") or branch.get("name") or "indicator").strip()


def _environment_branch_matches(branch: Dict[str, Any], tokens: Sequence[str]) -> bool:
    text = " ".join(
        str(branch.get(key) or "")
        for key in ("indicator_label", "name", "summary", "evidence_kind")
    ).lower()
    return any(token in text for token in tokens)


def _environment_action_evidence_basis(branch: Dict[str, Any]) -> str:
    label = _environment_branch_label(branch)
    direction = str(branch.get("direction") or ENV_DIRECTION_UNTESTABLE).strip()
    support = str(branch.get("support_strength") or ENV_SUPPORT_LIMITED).strip()
    return f"{label}: {direction}, {support}"


def _environment_policy_priority(branch: Dict[str, Any]) -> str:
    support = str(branch.get("support_strength") or "").strip()
    role = str(branch.get("role") or "").strip()
    if support == ENV_SUPPORT_SUPPORTED and role == ENV_ROLE_PRIMARY_ENDPOINT:
        return "high"
    if support in {ENV_SUPPORT_SUPPORTED, ENV_SUPPORT_LIMITED}:
        return "near_term"
    return "routine"


def _environment_action_guardrail(branch: Dict[str, Any]) -> str:
    support = str(branch.get("support_strength") or ENV_SUPPORT_LIMITED).strip()
    if support == ENV_SUPPORT_SUPPORTED:
        return "Use as prioritization evidence, while pairing it with local source, hydrodynamic, and field-observation context."
    if support == ENV_SUPPORT_UNTESTABLE:
        return "Use only for evidence development until the branch becomes testable."
    return "Treat as screening and low-regret guidance, not standalone causal proof."


def _dedupe_environment_policy_actions(actions: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    seen: set[Tuple[str, str, str]] = set()
    deduped: List[Dict[str, str]] = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip()
        action_type = str(item.get("action_type") or "").strip()
        target = str(item.get("target") or "").strip()
        key = (action_type, target, action)
        if not action or key in seen:
            continue
        seen.add(key)
        deduped.append({
            "priority": str(item.get("priority") or "routine").strip(),
            "action_type": action_type or "governance",
            "target": target or "environmental-health management",
            "where_when": str(item.get("where_when") or "").strip(),
            "evidence_basis": str(item.get("evidence_basis") or "").strip(),
            "action": action,
            "guardrail": str(item.get("guardrail") or "").strip(),
            "theme": str(item.get("theme") or "general").strip(),
        })
    return deduped


def _environment_policy_recommendations(
    branch_assessments: Sequence[Dict[str, Any]],
    *,
    overall_verdict: str,
    overall_support_strength: str,
    policy_action_matrix: Sequence[Dict[str, Any]] = (),
) -> List[str]:
    if policy_action_matrix:
        return _dedupe_strings(
            [
                f"{str(item.get('target') or 'Management action').strip()}: {str(item.get('action') or '').strip()}"
                for item in policy_action_matrix
                if isinstance(item, dict) and str(item.get("action") or "").strip()
            ]
        )[:6]

    recommendations: List[str] = []

    for branch in sorted(branch_assessments, key=_environment_branch_priority, reverse=True):
        if not isinstance(branch, dict):
            continue
        if branch.get("role") == ENV_ROLE_AUXILIARY_CONTEXT:
            continue
        if branch.get("direction") != ENV_DIRECTION_DETERIORATING:
            continue
        if branch.get("support_strength") not in {ENV_SUPPORT_SUPPORTED, ENV_SUPPORT_LIMITED}:
            continue
        recommendation = _policy_guidance_for_environment_branch(branch)
        if recommendation:
            recommendations.append(recommendation)

    if overall_verdict == "Deteriorating":
        if overall_support_strength == ENV_SUPPORT_SUPPORTED:
            recommendations.append(
                "Treat the assessed region as a higher-priority management area and align seasonal monitoring, mitigation, and compliance attention around the strongest deteriorating indicators."
            )
        else:
            recommendations.append(
                "Use this assessment as an early-warning basis for precautionary management, with near-term emphasis on targeted monitoring and low-regret mitigation rather than irreversible policy changes."
            )
    elif overall_verdict == "Improving":
        recommendations.append(
            "Maintain the current protection and management measures that are consistent with the improving indicators, while verifying that the positive signal persists in future windows."
        )
    else:
        recommendations.append(
            "Keep baseline protections in place and use indicator-specific monitoring to detect whether the current stable assessment shifts toward clearer deterioration or recovery."
        )

    return _dedupe_strings(recommendations)[:5]


def _environment_overall_narrative(
    branch_assessments: Sequence[Dict[str, Any]],
    *,
    overall_verdict: str,
    overall_support_strength: str,
) -> str:
    deteriorating = [
        str(item.get("indicator_label") or item.get("name") or "").strip()
        for item in sorted(branch_assessments, key=_environment_branch_priority, reverse=True)
        if isinstance(item, dict)
        and item.get("role") != ENV_ROLE_AUXILIARY_CONTEXT
        and item.get("direction") == ENV_DIRECTION_DETERIORATING
        and item.get("support_strength") in {ENV_SUPPORT_SUPPORTED, ENV_SUPPORT_LIMITED}
    ][:2]
    improving = [
        str(item.get("indicator_label") or item.get("name") or "").strip()
        for item in sorted(branch_assessments, key=_environment_branch_priority, reverse=True)
        if isinstance(item, dict)
        and item.get("role") != ENV_ROLE_AUXILIARY_CONTEXT
        and item.get("direction") == ENV_DIRECTION_IMPROVING
        and item.get("support_strength") in {ENV_SUPPORT_SUPPORTED, ENV_SUPPORT_LIMITED}
    ][:2]

    if overall_verdict == "Deteriorating":
        lead = "The combined indicators point toward environmental deterioration"
    elif overall_verdict == "Improving":
        lead = "The combined indicators point toward environmental improvement"
    else:
        lead = "The combined indicators suggest broadly stable environmental conditions"

    sentence = f"{lead} with an overall support grade of {overall_support_strength}."
    if deteriorating:
        sentence += f" The main pressures come from {', '.join(deteriorating)}."
    if improving:
        sentence += f" At the same time, {', '.join(improving)} provide partial counter-signals."
    return sentence


def _policy_guidance_for_environment_branch(branch: Dict[str, Any]) -> str:
    label = str(branch.get("indicator_label") or branch.get("name") or "indicator").strip().lower()
    if "temperature" in label or "sst" in label or "heat" in label:
        return (
            "Strengthen marine heat-risk preparedness by expanding warm-season monitoring, adaptation planning for fisheries and aquaculture, and targeted advisories during peak heat-stress periods."
        )
    if "bloom" in label or "chlorophyll" in label or "eutroph" in label:
        return (
            "Prioritize nutrient-management and bloom-surveillance measures, especially in runoff-influenced seasons and hotspots where repeated high-chlorophyll conditions can affect coastal water quality and aquaculture."
        )
    if "oxygen" in label or "hypoxia" in label:
        return (
            "Expand bottom-oxygen monitoring and reduce nutrient or organic loading in vulnerable coastal waters, with targeted attention to river and estuary inputs, discharge outlets, and outfalls so low-oxygen stress can be managed before it reaches ecologically damaging levels."
        )
    if "stratification" in label:
        return (
            "Use stronger stratification as a management warning signal by increasing vertical-profile observations in high-risk seasons and linking them to low-oxygen and bloom-response planning."
        )
    return ""


def _monitoring_guidance_for_environment_branch(branch: Dict[str, Any]) -> str:
    label = str(branch.get("indicator_label") or branch.get("name") or "indicator").strip().lower()
    if "oxygen" in label or "hypoxia" in label:
        return (
            "Maintain near-bottom dissolved oxygen monitoring in vulnerable seasons, paired with chlorophyll, stratification, river/estuary input, and discharge-outlet observations."
        )
    if "bloom" in label or "chlorophyll" in label or "eutroph" in label:
        return "Track chlorophyll-a exposure, bloom duration, and nutrient-sensitive coastal hotspots with seasonal reporting."
    if "temperature" in label or "sst" in label or "heat" in label:
        return "Maintain warm-season SST and marine heatwave early-warning indicators for fisheries, aquaculture, and protected-area managers."
    if "stratification" in label:
        return "Add repeated vertical profiles of temperature, salinity, density, oxygen, and chlorophyll in high-risk seasons."
    if "upwelling" in label or "front" in label or "eddy" in label:
        return "Use circulation and event-summary maps to target adaptive sampling and hotspot surveillance."
    return ""


def _environment_branch_assessment(
    name: str,
    indicator_label: str,
    direction: str,
    support_strength: str,
    output_type: str,
    summary: str,
    supporting_evidence: Sequence[str],
    uncertainties: Sequence[str],
    *,
    role: str = ENV_ROLE_PRIMARY_SUPPORT,
    evidence_kind: str = "",
    metric: str = "",
) -> Dict[str, Any]:
    role = _normalize_environment_branch_role(role)
    evidence_kind = _normalize_environment_evidence_kind(evidence_kind)
    metric = _normalize_environment_metric(metric)
    score_contribution = (
        0.0
        if role == ENV_ROLE_AUXILIARY_CONTEXT
        else _environment_direction_sign(direction) * _environment_support_weight(support_strength)
    )
    return {
        "name": name,
        "indicator_label": indicator_label,
        "role": role,
        "evidence_kind": evidence_kind,
        "metric": metric,
        "direction": direction,
        "support_strength": support_strength,
        "output_type": output_type,
        "summary": summary,
        "score_contribution": float(score_contribution),
        "supporting_evidence": _dedupe_strings(supporting_evidence)[:3],
        "uncertainties": _dedupe_strings(uncertainties)[:3],
    }


def _environment_branch_priority(branch: Dict[str, Any]) -> Tuple[float, float]:
    support_rank = {
        ENV_SUPPORT_SUPPORTED: 2.0,
        ENV_SUPPORT_LIMITED: 1.0,
        ENV_SUPPORT_UNTESTABLE: 0.0,
    }
    return (
        support_rank.get(str(branch.get("support_strength") or ""), 0.0),
        abs(float(branch.get("score_contribution") or 0.0)),
    )


def _environment_direction_sign(direction: str) -> float:
    if direction == ENV_DIRECTION_DETERIORATING:
        return 1.0
    if direction == ENV_DIRECTION_IMPROVING:
        return -1.0
    return 0.0


def _environment_support_weight(support_strength: str) -> float:
    if support_strength == ENV_SUPPORT_SUPPORTED:
        return 1.0
    if support_strength == ENV_SUPPORT_LIMITED:
        return 0.5
    return 0.0


def _humanize_environment_indicator(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _sortable_event_group(label: str) -> tuple[int, Any]:
    try:
        return (0, int(label))
    except (TypeError, ValueError):
        return (1, str(label))


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return None


def _deepest_valid_profile_value(
    field: xr.DataArray,
    depth_dim: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    if is_dask_backed(field):
        return _deepest_valid_profile_value_chunked(field, depth_dim)

    depth_values = xr.DataArray(
        np.asarray(field[depth_dim].values, dtype=float),
        coords={depth_dim: field[depth_dim].values},
        dims=(depth_dim,),
    )
    depth_grid = depth_values.broadcast_like(field)
    deepest_abs_depth = np.abs(depth_grid).where(np.isfinite(field)).max(dim=depth_dim, skipna=True)
    deepest_mask = np.abs(depth_grid) == deepest_abs_depth
    deepest_value = field.where(deepest_mask).max(dim=depth_dim, skipna=True)
    deepest_depth = depth_grid.where(deepest_mask).max(dim=depth_dim, skipna=True)
    return deepest_value, deepest_depth


def _deepest_valid_profile_value_chunked(
    field: xr.DataArray,
    depth_dim: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    depth_values = np.asarray(field[depth_dim].values, dtype=float)
    report_phase(
        phase="preparing_deepest_valid_profile_selection",
        message="Preparing chunked deepest-valid profile selection",
        percent=0.05,
        compute_backend="dask",
        chunks=chunk_summary(field),
    )
    field = field.chunk({depth_dim: -1})
    deepest_value, deepest_depth = xr.apply_ufunc(
        _deepest_valid_profile_value_block,
        field,
        input_core_dims=[[depth_dim]],
        output_core_dims=[[], []],
        kwargs={"depth_values": depth_values},
        dask="parallelized",
        output_dtypes=[float, float],
        dask_gufunc_kwargs={"allow_rechunk": False},
    )
    deepest_value.name = field.name
    deepest_depth.name = f"{depth_dim}_of_deepest_valid_profile_value"
    report_phase(
        phase="deepest_valid_profile_selection_prepared",
        message="Prepared chunked deepest-valid profile selection",
        percent=0.1,
        compute_backend="dask",
        chunks=chunk_summary(deepest_value),
    )
    return deepest_value, deepest_depth


def _deepest_valid_profile_value_block(values: np.ndarray, *, depth_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    depth = np.asarray(depth_values, dtype=float)
    if values.shape[-1] != depth.size:
        raise ValueError("Depth coordinate size does not match the density block depth dimension")

    finite = np.isfinite(values)
    has_valid = finite.any(axis=-1)
    abs_depth = np.abs(depth)
    deepest_abs_depth = np.where(finite, abs_depth, -np.inf).max(axis=-1)
    deepest_mask = finite & (abs_depth == np.expand_dims(deepest_abs_depth, axis=-1))

    deepest_value = np.where(deepest_mask, values, -np.inf).max(axis=-1)
    deepest_depth = np.where(deepest_mask, depth, -np.inf).max(axis=-1)
    deepest_value = np.where(has_valid, deepest_value, np.nan)
    deepest_depth = np.where(has_valid, deepest_depth, np.nan)
    return deepest_value, deepest_depth


def _build_coord_slice(coord_values: Sequence[float], bounds: Tuple[float, float]) -> slice:
    lower, upper = bounds
    if len(coord_values) == 0:
        return slice(lower, upper)
    coord_values = np.asarray(coord_values, dtype=float)
    ascending = len(coord_values) < 2 or coord_values[0] <= coord_values[1]
    if ascending:
        return slice(min(lower, upper), max(lower, upper))
    return slice(max(lower, upper), min(lower, upper))


def _nearest_index(coord_values: Sequence[float], target: float) -> int:
    coord = np.asarray(coord_values, dtype=float)
    return int(np.nanargmin(np.abs(coord - float(target))))


def _extract_series(result: Dict[str, Any], target_times: Optional[Sequence[Any]] = None) -> Tuple[pd.Index, np.ndarray]:
    if "times" not in result or "values" not in result:
        raise ValueError("Expected a timeseries-like result with 'times' and 'values'")
    series = pd.Series(np.asarray(result["values"], dtype=float), index=pd.Index([str(t) for t in result["times"]]))
    if target_times is None:
        return series.index, series.to_numpy(dtype=float)
    aligned = series.reindex(pd.Index([str(t) for t in target_times]))
    return aligned.index, aligned.to_numpy(dtype=float)


def _series_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    metadata = result.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _simple_statistics(values: np.ndarray) -> Dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "n_valid": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n_valid": int(finite.size),
        "mean": float(np.nanmean(finite)),
        "std": float(np.nanstd(finite)),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
    }


def _mean_abs(values: np.ndarray) -> float:
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.nanmean(finite))


def _safe_correlation(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3:
        return 0.0
    a = a[mask]
    b = b[mask]
    if np.nanstd(a) <= 1e-12 or np.nanstd(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _best_lag(
    values_a: np.ndarray,
    values_b: np.ndarray,
    max_lag: int,
) -> Dict[str, Any]:
    best = {"lag": 0, "correlation": _safe_correlation(values_a, values_b)}
    lags = _lag_candidates(max_lag)
    correlations = [best["correlation"] if lag == 0 else float("nan") for lag in lags]
    for index, lag in enumerate(lags):
        if lag == 0:
            continue
        if lag < 0:
            corr = _safe_correlation(values_a[:lag], values_b[-lag:])
        else:
            corr = _safe_correlation(values_a[lag:], values_b[:-lag])
        correlations[index] = float(corr)

    best_index = _select_optimal_lag_index(
        lags,
        np.asarray(correlations, dtype=float),
    )
    best = {"lag": int(lags[best_index]), "correlation": float(correlations[best_index])}
    return best


def _compute_short_lag_correlation(
    timeseries1: Dict[str, Any],
    timeseries2: Dict[str, Any],
    max_lag: int,
    confidence_level: float,
) -> Dict[str, Any]:
    values1 = np.asarray(timeseries1.get("values", []), dtype=float)
    values2 = np.asarray(timeseries2.get("values", []), dtype=float)
    min_len = min(len(values1), len(values2))
    values1 = values1[:min_len]
    values2 = values2[:min_len]

    mask = np.isfinite(values1) & np.isfinite(values2)
    values1 = values1[mask]
    values2 = values2[mask]
    if values1.size == 0:
        raise ValueError("Lag correlation requires at least one valid paired sample")

    adjusted_max_lag = max(0, min(int(max_lag), max(values1.size - 1, 0)))
    lags = _lag_candidates(adjusted_max_lag)
    correlations: List[float] = []
    p_values: List[float] = []

    for lag in lags:
        if lag < 0:
            left = values1[:lag]
            right = values2[-lag:]
        elif lag > 0:
            left = values1[lag:]
            right = values2[:-lag]
        else:
            left = values1
            right = values2

        corr = _safe_correlation(left, right)
        correlations.append(float(corr))
        p_values.append(float("nan"))

    correlations_array = np.asarray(correlations, dtype=float)
    max_idx = _select_optimal_lag_index(
        lags,
        correlations_array,
    )
    optimal_lag = int(lags[max_idx])
    max_correlation = float(correlations[max_idx])
    confidence_bound = float(1.96 / np.sqrt(values1.size)) if values1.size > 0 else float("nan")

    return {
        "lags": lags,
        "correlations": correlations,
        "p_values": p_values,
        "optimal_lag": optimal_lag,
        "max_correlation": max_correlation,
        "confidence_bound": confidence_bound,
        "confidence_level": confidence_level,
        "ts1_label": timeseries1.get("metadata", {}).get("variable", "ts1"),
        "ts2_label": timeseries2.get("metadata", {}).get("variable", "ts2"),
        "n_points": int(values1.size),
        "metadata": {
            "method": "short_series_fallback",
        },
    }


def _project_climatology_to_time(
    climatology: xr.DataArray,
    source_time: xr.DataArray,
    period: str,
) -> xr.DataArray:
    if period == "monthly":
        labels = pd.to_datetime(source_time.values).month
        lookup = (
            climatology.assign_coords(month=("time", pd.to_datetime(climatology.time.values).month))
            .swap_dims({"time": "month"})
            .drop_vars("time")
        )
        projected = lookup.sel(month=xr.DataArray(labels, dims="time", coords={"time": source_time}))
        if "month" in projected.coords:
            projected = projected.reset_coords("month", drop=True)
    elif period == "seasonal":
        labels = pd.to_datetime(source_time.values).quarter
        lookup = (
            climatology.assign_coords(quarter=("time", pd.to_datetime(climatology.time.values).quarter))
            .swap_dims({"time": "quarter"})
            .drop_vars("time")
        )
        projected = lookup.sel(quarter=xr.DataArray(labels, dims="time", coords={"time": source_time}))
        if "quarter" in projected.coords:
            projected = projected.reset_coords("quarter", drop=True)
    else:
        raise ValueError(f"Unsupported climatology period: {period}")

    projected = projected.transpose(*_time_first_dims(climatology))
    return projected


def _time_first_dims(climatology: xr.DataArray) -> Tuple[str, ...]:
    dims = ["time"]
    dims.extend(dim for dim in climatology.dims if dim != "time")
    return tuple(dims)


def _claim_from_correlation(abs_corr: float) -> str:
    if abs_corr >= 0.6:
        return CLAIM_SUPPORTED
    if abs_corr >= 0.3:
        return CLAIM_LIMITED
    return CLAIM_UNTESTABLE


def _claim_from_standardized_difference(value: float) -> str:
    if value >= 1.0:
        return CLAIM_SUPPORTED
    if value >= 0.5:
        return CLAIM_LIMITED
    return CLAIM_UNTESTABLE


def _claim_from_residual_share(value: float) -> str:
    if value <= 0.25:
        return CLAIM_SUPPORTED
    if value <= 0.5:
        return CLAIM_LIMITED
    return CLAIM_UNTESTABLE


def _claim_rank(value: str) -> int:
    order = {
        CLAIM_UNTESTABLE: 0,
        CLAIM_LIMITED: 1,
        CLAIM_SUPPORTED: 2,
    }
    return order.get(value, 0)


def _max_claim_strength(left: str, right: str) -> str:
    return left if _claim_rank(left) >= _claim_rank(right) else right


def _normalize_string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.append(text)
    return seen


def _event_time_indices(events: Sequence[Dict[str, Any]], max_len: int) -> List[int]:
    indices: List[int] = []
    for event in events:
        raw_index = event.get("time_index")
        if isinstance(raw_index, (int, float)):
            index = int(raw_index)
            if 0 <= index < max_len:
                indices.append(index)
    return sorted(set(indices))


def _differentiate_longitude(data: xr.DataArray) -> xr.DataArray:
    return data.differentiate("lon") / _calculate_dx(data.lon, data.lat)


def _differentiate_latitude(data: xr.DataArray) -> xr.DataArray:
    return data.differentiate("lat") / _calculate_dy(data.lat)


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _combine_units(base_units: Optional[str], suffix: str) -> str:
    base = (base_units or "").strip()
    return suffix if not base else f"{base} {suffix}"
