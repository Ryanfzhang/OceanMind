from __future__ import annotations

from math import sqrt
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from domain.ocean.data_access.partitioned import materialize_partitioned_xarray
from domain.ocean.events.analysis.statistics import (
    _horizontal_valid_mask_as_dataarray,
    _prepare_summary_field,
    _resolve_event_mask,
    _resolve_summary_threshold,
    _time_step_days,
)
from domain.ocean.events.vertical import prepare_event_vertical_field
from packages.runtime import get_active_watermass_config


def compute_watermass_event_association(
    event_field: xr.DataArray,
    temp: xr.DataArray,
    salt: xr.DataArray,
    density: xr.DataArray,
    event_detection: Dict[str, Any],
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    subregion_grid: Tuple[int, int] = (3, 3),
    hotspot_quantile: float = 0.75,
    max_ts_points: int = 4000,
    sampling: Literal["random", "head"] = "random",
    watermass_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Diagnose whether event hotspots are preferentially organized by named water masses.

    The workflow:
    1. Apply the same vertical semantics used during event detection to temp/salt/density.
    2. Classify each valid sample into a named water-mass bin using configured axes
       plus nearest-center fallback.
    3. Aggregate event burden/coverage and dominant water mass in each equal lon-lat tile.
    4. Compare hotspot-tile water-mass composition against the background tile distribution.
    """
    event_field = materialize_partitioned_xarray(event_field)
    temp = materialize_partitioned_xarray(temp)
    salt = materialize_partitioned_xarray(salt)
    density = materialize_partitioned_xarray(density)

    grid = _normalize_grid(subregion_grid)
    watermass_config = get_active_watermass_config(watermass_config_path)
    bins = watermass_config.bins
    classification_axes = tuple(watermass_config.classification_axes)
    if not bins:
        raise ValueError("Watermass configuration must define at least one bin")

    prepared_event = _prepare_summary_field(event_field, event_detection)
    default_mode = "bottom" if str(event_detection.get("event_type") or "").strip().lower() == "hypoxia" else "surface"
    params = _detection_params(event_detection)
    prepared_temp = _prepare_tracer_like_event_field(temp, params=params, default_mode=default_mode)
    prepared_salt = _prepare_tracer_like_event_field(salt, params=params, default_mode=default_mode)
    prepared_density = _prepare_tracer_like_event_field(density, params=params, default_mode=default_mode)

    prepared_event, prepared_temp, prepared_salt, prepared_density = xr.align(
        prepared_event,
        prepared_temp,
        prepared_salt,
        prepared_density,
        join="inner",
    )
    event_mask = _resolve_event_mask(event_detection, prepared_event)
    event_mask, prepared_event = xr.align(event_mask, prepared_event, join="inner")
    valid_ocean_mask = _horizontal_valid_mask_as_dataarray(prepared_event)

    sigma0 = prepared_density - 1000.0 if "sigma0" in classification_axes else None
    class_indices, exact_match_count, nearest_fallback_count = _classify_watermass_samples(
        temp=prepared_temp,
        salt=prepared_salt,
        sigma0=sigma0,
        bins=bins,
        classification_axes=classification_axes,
    )

    event_score_map, event_coverage_map, event_score_unit = _build_event_tile_inputs(
        field=prepared_event,
        event_mask=event_mask,
        event_detection=event_detection,
    )

    tile_records = _aggregate_tile_records(
        class_indices=class_indices,
        event_score_map=event_score_map,
        event_coverage_map=event_coverage_map,
        valid_ocean_mask=valid_ocean_mask,
        lon_range=lon_range,
        lat_range=lat_range,
        grid=grid,
        bins=bins,
    )
    hotspot_threshold, hotspot_count = _assign_hotspot_flags(tile_records, hotspot_quantile=hotspot_quantile)
    association = _build_watermass_association(tile_records, bins=bins, hotspot_threshold=hotspot_threshold)
    ts_payload = _build_watermass_ts_payload(
        temp=prepared_temp,
        salt=prepared_salt,
        class_indices=class_indices,
        bins=bins,
        classification_axes=classification_axes,
        max_points=max_ts_points,
        sampling=sampling,
    )

    event_type = str(event_detection.get("event_type") or "").strip().lower() or "event"
    return {
        "output_type": "watermass_event_association_result",
        "event_type": event_type,
        "grid_shape": [int(grid[0]), int(grid[1])],
        "tile_bounds": [
            {
                "subregion_id": record["subregion_id"],
                "label": record["label"],
                "short_label": record["short_label"],
                "row": int(record["row"]),
                "col": int(record["col"]),
                "bounds": dict(record["bounds"]),
            }
            for record in tile_records
        ],
        "event_tile_metrics": [
            {
                "subregion_id": record["subregion_id"],
                "label": record["label"],
                "short_label": record["short_label"],
                "row": int(record["row"]),
                "col": int(record["col"]),
                "bounds": dict(record["bounds"]),
                "status": record["status"],
                "event_score": record["event_score"],
                "event_score_unit": event_score_unit,
                "event_score_label": _format_event_score_label(record["event_score"], event_score_unit),
                "event_coverage_fraction": record["event_coverage_fraction"],
                "is_hotspot": bool(record["is_hotspot"]),
                "dominant_watermass": record.get("dominant_watermass"),
                "dominant_watermass_name": record.get("dominant_watermass_name"),
            }
            for record in tile_records
        ],
        "dominant_watermass_by_tile": [
            {
                "subregion_id": record["subregion_id"],
                "label": record["label"],
                "short_label": record["short_label"],
                "row": int(record["row"]),
                "col": int(record["col"]),
                "bounds": dict(record["bounds"]),
                "status": record["status"],
                "dominant_watermass": record.get("dominant_watermass"),
                "dominant_watermass_name": record.get("dominant_watermass_name"),
                "dominant_watermass_short_name": record.get("dominant_watermass_short_name"),
                "dominant_watermass_color": record.get("dominant_watermass_color"),
                "dominant_share": record.get("dominant_share"),
            }
            for record in tile_records
        ],
        "watermass_fractions_by_tile": [
            {
                "subregion_id": record["subregion_id"],
                "label": record["label"],
                "short_label": record["short_label"],
                "row": int(record["row"]),
                "col": int(record["col"]),
                "bounds": dict(record["bounds"]),
                "status": record["status"],
                "sample_count": int(record["sample_count"]),
                "fractions": dict(record["fractions"]),
            }
            for record in tile_records
        ],
        "watermass_event_association": association,
        "evidence_strength": association["evidence_strength"],
        "assignment_method": {
            "classification_axes": list(classification_axes),
            "strict_rule": _strict_rule_name(classification_axes),
            "fallback_rule": "nearest_center",
            "overlap_resolution": "nearest_center",
            "exact_match_count": int(exact_match_count),
            "nearest_fallback_count": int(nearest_fallback_count),
        },
        "watermass_bins": [_serialize_bin(item, classification_axes=classification_axes) for item in bins],
        "class_color_map": {item.id: item.color for item in bins},
        "watermass_ts_diagram": ts_payload,
        "metadata": {
            "title": "Event-Watermass Tile Association",
            "event_type": event_type,
            "hotspot_quantile": float(hotspot_quantile),
            "event_score_unit": event_score_unit,
            "watermass_config_id": watermass_config.id,
            "watermass_config_name": watermass_config.name,
            "classification_axes": list(classification_axes),
            "n_bins": len(bins),
        },
    }


def build_watermass_tile_map(
    association_result: Dict[str, Any],
    map_kind: Literal["event_hotspot", "dominant_watermass"] = "event_hotspot",
) -> Dict[str, Any]:
    """
    Convert a watermass-event association result into a map-ready tile field.
    """
    grid = _extract_grid_shape(association_result)
    bins = _extract_bins(association_result)
    tile_bounds = association_result.get("tile_bounds")
    if not isinstance(tile_bounds, list) or not tile_bounds:
        raise ValueError("association_result must include tile_bounds")

    event_type = str(association_result.get("event_type") or "").strip().lower() or "event"
    metadata = association_result.get("metadata", {}) if isinstance(association_result.get("metadata"), dict) else {}
    event_unit = str(metadata.get("event_score_unit") or "")
    rows = grid[1]
    cols = grid[0]

    lon_centers = _axis_centers(tile_bounds, axis="lon", count=cols)
    lat_centers = _axis_centers(tile_bounds, axis="lat", count=rows)

    if map_kind == "event_hotspot":
        metrics = association_result.get("event_tile_metrics")
        if not isinstance(metrics, list):
            raise ValueError("association_result must include event_tile_metrics")
        values = np.full((rows, cols), np.nan, dtype=float)
        cells = []
        for item in metrics:
            if not isinstance(item, dict):
                continue
            row = int(item.get("row") or 0)
            col = int(item.get("col") or 0)
            if 1 <= row <= rows and 1 <= col <= cols and isinstance(item.get("event_score"), (int, float)):
                values[row - 1, col - 1] = float(item["event_score"])
            bounds = item.get("bounds")
            cells.append(
                {
                    "subregionId": str(item.get("subregion_id") or f"r{row}_c{col}"),
                    "label": str(item.get("label") or f"R{row}C{col}"),
                    "shortLabel": str(item.get("short_label") or _tile_short_label(row, col, grid)).strip(),
                    "bounds": _json_bounds(bounds),
                    "status": str(item.get("status") or "ok"),
                    "category": "hotspot" if item.get("is_hotspot") else "background",
                    "categoryLabel": "Hotspot" if item.get("is_hotspot") else "Non-hotspot",
                    "categoryShortLabel": "HOT" if item.get("is_hotspot") else "BG",
                    "color": "#d97706" if item.get("is_hotspot") else "#2563eb",
                    "value": _coerce_optional_float(item.get("event_score")),
                    "valueLabel": _format_event_score_label(item.get("event_score"), event_unit),
                    "details": [
                        f"Coverage: {_format_fraction(item.get('event_coverage_fraction'))}",
                        f"Dominant watermass: {item.get('dominant_watermass_name') or 'NA'}",
                    ],
                }
            )
        return {
            "lon": lon_centers,
            "lat": lat_centers,
            "values": values.tolist(),
            "metadata": {
                "title": f"{_event_type_label(event_type)} Hotspot Tile Map",
                "variable": "event_hotspot_tile_score",
                "units": event_unit,
                "tile_map_kind": map_kind,
                "statistics": _numeric_statistics(values),
                "subregion_grid": _subregion_grid_payload(grid, cells),
                "discrete_legend": [
                    {
                        "value": 0.0,
                        "category": "background",
                        "label": "Non-hotspot",
                        "short_label": "BG",
                        "color": "#2563eb",
                    },
                    {
                        "value": 1.0,
                        "category": "hotspot",
                        "label": "Hotspot",
                        "short_label": "HOT",
                        "color": "#d97706",
                    },
                ],
                "depth_range": metadata.get("depth_range"),
                "depth_aggregation": metadata.get("depth_aggregation"),
            },
        }

    if map_kind != "dominant_watermass":
        raise ValueError(f"Unsupported map_kind: {map_kind}")

    dominant = association_result.get("dominant_watermass_by_tile")
    if not isinstance(dominant, list):
        raise ValueError("association_result must include dominant_watermass_by_tile")

    bin_index = {item.id: idx + 1 for idx, item in enumerate(bins)}
    values = np.full((rows, cols), np.nan, dtype=float)
    cells = []
    for item in dominant:
        if not isinstance(item, dict):
            continue
        row = int(item.get("row") or 0)
        col = int(item.get("col") or 0)
        watermass_id = str(item.get("dominant_watermass") or "").strip()
        if 1 <= row <= rows and 1 <= col <= cols and watermass_id in bin_index:
            values[row - 1, col - 1] = float(bin_index[watermass_id])
        bounds = item.get("bounds")
        cells.append(
            {
                "subregionId": str(item.get("subregion_id") or f"r{row}_c{col}"),
                "label": str(item.get("label") or f"R{row}C{col}"),
                "shortLabel": str(item.get("short_label") or _tile_short_label(row, col, grid)).strip(),
                "bounds": _json_bounds(bounds),
                "status": str(item.get("status") or "ok"),
                "category": watermass_id or None,
                "categoryLabel": str(item.get("dominant_watermass_name") or "NA"),
                "categoryShortLabel": str(item.get("dominant_watermass_short_name") or "NA"),
                "color": str(item.get("dominant_watermass_color") or "#475569"),
                "value": _coerce_optional_float(item.get("dominant_share")),
                "valueLabel": f"Dominant share {_format_fraction(item.get('dominant_share'))}",
                "details": [],
            }
        )

    return {
        "lon": lon_centers,
        "lat": lat_centers,
        "values": values.tolist(),
        "metadata": {
            "title": "Dominant Watermass Tile Map",
            "variable": "dominant_watermass_index",
            "units": "named classes",
            "tile_map_kind": map_kind,
            "statistics": _numeric_statistics(values),
            "subregion_grid": _subregion_grid_payload(grid, cells),
            "discrete_legend": [
                {
                    "value": float(idx + 1),
                    "category": item.id,
                    "label": item.name,
                    "short_label": item.short_name or item.id.upper(),
                    "color": item.color,
                }
                for idx, item in enumerate(bins)
            ],
        },
    }


def build_watermass_ts_diagram(association_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expose the precomputed watermass-definition T-S diagram payload as a standard result.
    """
    payload = association_result.get("watermass_ts_diagram")
    if not isinstance(payload, dict):
        raise ValueError("association_result must include watermass_ts_diagram")
    return {
        "output_type": "ts_diagram_result",
        "temperature": list(payload.get("temperature", [])),
        "salinity": list(payload.get("salinity", [])),
        "point_classes": list(payload.get("point_classes", [])),
        "metadata": dict(payload.get("metadata") or {}),
    }


def _prepare_tracer_like_event_field(
    field: xr.DataArray,
    *,
    params: Dict[str, Any],
    default_mode: str,
) -> xr.DataArray:
    return prepare_event_vertical_field(
        field,
        default_mode=default_mode,  # type: ignore[arg-type]
        vertical_mode=str(params.get("vertical_mode") or default_mode),
        depth_value=_normalize_optional_float(params.get("depth_value")),
        depth_range=_normalize_optional_range(params.get("depth_range")),
        depth_aggregation=str(params.get("depth_aggregation") or "mean"),
    )


def _detection_params(event_detection: Dict[str, Any]) -> Dict[str, Any]:
    statistics = event_detection.get("statistics")
    if isinstance(statistics, dict):
        detection_params = statistics.get("detection_params")
        if isinstance(detection_params, dict):
            return detection_params
    return {}


def _normalize_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_optional_range(value: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _normalize_grid(subregion_grid: Sequence[int]) -> Tuple[int, int]:
    if len(subregion_grid) != 2:
        raise ValueError("subregion_grid must contain [nx, ny]")
    nx = max(int(subregion_grid[0]), 1)
    ny = max(int(subregion_grid[1]), 1)
    return nx, ny


def _classify_watermass_samples(
    *,
    temp: xr.DataArray,
    salt: xr.DataArray,
    sigma0: Optional[xr.DataArray],
    bins: Sequence[Any],
    classification_axes: Sequence[str],
) -> Tuple[xr.DataArray, int, int]:
    temp_values = np.asarray(temp.values, dtype=float)
    salt_values = np.asarray(salt.values, dtype=float)
    sigma_values = np.asarray(sigma0.values, dtype=float) if sigma0 is not None else None

    values_by_axis: Dict[str, Optional[np.ndarray]] = {
        "temp": temp_values,
        "salt": salt_values,
        "sigma0": sigma_values,
    }
    valid_mask = np.ones(temp_values.shape, dtype=bool)
    for axis in classification_axes:
        axis_values = values_by_axis.get(axis)
        valid_mask &= axis_values is not None and np.isfinite(axis_values)

    class_indices = np.full(temp_values.shape, -1, dtype=int)
    if not np.any(valid_mask):
        return (
            xr.DataArray(class_indices, coords=temp.coords, dims=temp.dims, name="watermass_index"),
            0,
            0,
        )

    contains_by_bin = []
    distance_by_bin = []
    for item in bins:
        contains = valid_mask.copy()
        distance = np.zeros(temp_values.shape, dtype=float)
        for axis in classification_axes:
            axis_values = values_by_axis.get(axis)
            bounds = list(_bounds_for_axis(item, axis))
            if axis_values is None or len(bounds) != 2:
                contains &= False
                continue
            lower, upper = float(bounds[0]), float(bounds[1])
            contains &= (axis_values >= lower) & (axis_values <= upper)
            center = 0.5 * (lower + upper)
            scale = max(0.5 * abs(upper - lower), 1e-6)
            distance += ((axis_values - center) / scale) ** 2
        contains_by_bin.append(contains)
        distance_by_bin.append(np.sqrt(distance))

    contains_stack = np.stack(contains_by_bin, axis=0)
    distance_stack = np.stack(distance_by_bin, axis=0)
    exact_mask = np.any(contains_stack, axis=0) & valid_mask
    fallback_mask = valid_mask & ~exact_mask

    exact_distances = np.where(contains_stack, distance_stack, np.inf)
    nearest_exact = np.argmin(exact_distances, axis=0)
    nearest_any = np.argmin(distance_stack, axis=0)
    class_indices[exact_mask] = nearest_exact[exact_mask]
    class_indices[fallback_mask] = nearest_any[fallback_mask]

    return (
        xr.DataArray(class_indices, coords=temp.coords, dims=temp.dims, name="watermass_index"),
        int(np.count_nonzero(exact_mask)),
        int(np.count_nonzero(fallback_mask)),
    )


def _sample_is_valid(
    classification_axes: Sequence[str],
    *,
    sigma0: Optional[float],
    temp: float,
    salt: float,
) -> bool:
    for axis in classification_axes:
        value = _value_for_axis(axis, sigma0=sigma0, temp=temp, salt=salt)
        if value is None or not np.isfinite(value):
            return False
    return True


def _bin_contains(
    item: Any,
    *,
    classification_axes: Sequence[str],
    sigma0: Optional[float],
    temp: float,
    salt: float,
) -> bool:
    for axis in classification_axes:
        bounds = _bounds_for_axis(item, axis)
        value = _value_for_axis(axis, sigma0=sigma0, temp=temp, salt=salt)
        if len(bounds) != 2 or value is None or not (bounds[0] <= value <= bounds[1]):
            return False
    return True


def _normalized_center_distance(
    item: Any,
    *,
    classification_axes: Sequence[str],
    sigma0: Optional[float],
    temp: float,
    salt: float,
) -> float:
    def term(value: float, bounds: Sequence[float]) -> float:
        lower, upper = float(bounds[0]), float(bounds[1])
        center = 0.5 * (lower + upper)
        scale = max(0.5 * abs(upper - lower), 1e-6)
        return ((value - center) / scale) ** 2

    distance = 0.0
    for axis in classification_axes:
        bounds = _bounds_for_axis(item, axis)
        value = _value_for_axis(axis, sigma0=sigma0, temp=temp, salt=salt)
        if len(bounds) != 2 or value is None:
            continue
        distance += term(value, bounds)
    return sqrt(distance)


def _bounds_for_axis(item: Any, axis: str) -> Sequence[float]:
    return list(getattr(item, f"{axis}_range", []) or [])


def _value_for_axis(
    axis: str,
    *,
    sigma0: Optional[float],
    temp: float,
    salt: float,
) -> Optional[float]:
    if axis == "sigma0":
        return sigma0
    if axis == "temp":
        return float(temp)
    if axis == "salt":
        return float(salt)
    raise ValueError(f"Unsupported classification axis: {axis}")


def _strict_rule_name(classification_axes: Sequence[str]) -> str:
    if not classification_axes:
        return "no_strict_rule"
    return "_and_".join(classification_axes) + "_ranges"


def _build_event_tile_inputs(
    *,
    field: xr.DataArray,
    event_mask: xr.DataArray,
    event_detection: Dict[str, Any],
) -> Tuple[xr.DataArray, xr.DataArray, str]:
    if "time" in event_mask.dims:
        coverage = event_mask.astype(float).mean(dim="time", skipna=True)
    else:
        coverage = event_mask.astype(float)
    coverage = coverage.where(np.isfinite(coverage))

    event_type = str(event_detection.get("event_type") or "").strip().lower()
    threshold = _resolve_summary_threshold(event_detection, field)
    if event_type in {"algal_bloom", "heatwave", "eutrophication"}:
        severity = xr.where(field > threshold, field - threshold, 0.0)
    elif event_type in {"hypoxia", "upwelling"}:
        severity = xr.where(field < threshold, threshold - field, 0.0)
    else:
        return coverage, coverage, "event_fraction"

    contribution = severity.where(event_mask, 0.0)
    if "time" in contribution.dims:
        step_days = _time_step_days(np.asarray(field["time"].values))
        burden = (contribution * step_days).sum(dim="time", skipna=True)
    else:
        burden = contribution

    source_units = str(field.attrs.get("units", "") or "").strip()
    unit = f"{source_units} day" if source_units else "burden_day"
    return burden, coverage, unit


def _aggregate_tile_records(
    *,
    class_indices: xr.DataArray,
    event_score_map: xr.DataArray,
    event_coverage_map: xr.DataArray,
    valid_ocean_mask: xr.DataArray,
    lon_range: Tuple[float, float],
    lat_range: Tuple[float, float],
    grid: Tuple[int, int],
    bins: Sequence[Any],
) -> list[Dict[str, Any]]:
    lon_edges = np.linspace(float(min(lon_range)), float(max(lon_range)), grid[0] + 1)
    lat_edges = np.linspace(float(min(lat_range)), float(max(lat_range)), grid[1] + 1)

    tile_records: list[Dict[str, Any]] = []
    for row in range(1, grid[1] + 1):
        for col in range(1, grid[0] + 1):
            lon_min = float(lon_edges[col - 1])
            lon_max = float(lon_edges[col])
            lat_min = float(lat_edges[row - 1])
            lat_max = float(lat_edges[row])

            lon_mask = _coord_mask(np.asarray(class_indices["lon"].values, dtype=float), col - 1, lon_edges)
            lat_mask = _coord_mask(np.asarray(class_indices["lat"].values, dtype=float), row - 1, lat_edges)
            if not lon_mask.any() or not lat_mask.any():
                tile_records.append(
                    {
                        "subregion_id": f"r{row}_c{col}",
                        "label": f"R{row}C{col}",
                        "short_label": _tile_short_label(row, col, grid),
                        "row": row,
                        "col": col,
                        "bounds": {
                            "lonMin": lon_min,
                            "lonMax": lon_max,
                            "latMin": lat_min,
                            "latMax": lat_max,
                        },
                        "sample_count": 0,
                        "event_score": float("nan"),
                        "event_coverage_fraction": float("nan"),
                        "fractions": {},
                        "status": "skipped_no_grid_cell",
                        "is_hotspot": False,
                    }
                )
                continue

            tile_class = class_indices.isel(lat=np.where(lat_mask)[0], lon=np.where(lon_mask)[0])
            tile_event_score = event_score_map.isel(lat=np.where(lat_mask)[0], lon=np.where(lon_mask)[0])
            tile_event_coverage = event_coverage_map.isel(lat=np.where(lat_mask)[0], lon=np.where(lon_mask)[0])
            tile_ocean = valid_ocean_mask.isel(lat=np.where(lat_mask)[0], lon=np.where(lon_mask)[0])

            sample_values = np.asarray(tile_class.values, dtype=int)
            valid_samples = sample_values[sample_values >= 0]
            event_score_values = np.asarray(tile_event_score.values, dtype=float)
            coverage_values = np.asarray(tile_event_coverage.values, dtype=float)
            tile_has_ocean = bool(np.any(np.asarray(tile_ocean.values, dtype=bool)))

            record: Dict[str, Any] = {
                "subregion_id": f"r{row}_c{col}",
                "label": f"R{row}C{col}",
                "short_label": _tile_short_label(row, col, grid),
                "row": row,
                "col": col,
                "bounds": {
                    "lonMin": lon_min,
                    "lonMax": lon_max,
                    "latMin": lat_min,
                    "latMax": lat_max,
                },
                "sample_count": int(valid_samples.size),
                "event_score": float(np.nanmean(event_score_values)) if np.isfinite(event_score_values).any() else float("nan"),
                "event_coverage_fraction": float(np.nanmean(coverage_values)) if np.isfinite(coverage_values).any() else float("nan"),
                "fractions": {},
                "status": "ok" if tile_has_ocean and valid_samples.size > 0 else "skipped_no_valid_ocean",
                "is_hotspot": False,
            }

            if tile_has_ocean and valid_samples.size > 0:
                counts = np.bincount(valid_samples, minlength=len(bins))
                fractions = {
                    bins[index].id: float(count / valid_samples.size)
                    for index, count in enumerate(counts)
                    if count > 0
                }
                dominant_index = int(np.argmax(counts))
                dominant_bin = bins[dominant_index]
                dominant_share = float(counts[dominant_index] / valid_samples.size)
                record.update(
                    {
                        "fractions": fractions,
                        "dominant_watermass": dominant_bin.id,
                        "dominant_watermass_name": dominant_bin.name,
                        "dominant_watermass_short_name": dominant_bin.short_name or dominant_bin.id.upper(),
                        "dominant_watermass_color": dominant_bin.color,
                        "dominant_share": dominant_share,
                    }
                )
            tile_records.append(record)

    return tile_records


def _coord_mask(values: np.ndarray, cell_index: int, edges: np.ndarray) -> np.ndarray:
    lower = float(edges[cell_index])
    upper = float(edges[cell_index + 1])
    if cell_index == len(edges) - 2:
        return (values >= lower) & (values <= upper)
    return (values >= lower) & (values < upper)


def _assign_hotspot_flags(
    tile_records: list[Dict[str, Any]],
    *,
    hotspot_quantile: float,
) -> Tuple[Optional[float], int]:
    positive_scores = np.asarray(
        [
            float(item["event_score"])
            for item in tile_records
            if item.get("status") == "ok" and isinstance(item.get("event_score"), (int, float)) and float(item["event_score"]) > 0
        ],
        dtype=float,
    )
    if positive_scores.size == 0:
        return None, 0

    if positive_scores.size == 1:
        threshold = float(positive_scores[0])
    else:
        threshold = float(np.nanquantile(positive_scores, min(max(float(hotspot_quantile), 0.0), 1.0)))

    hotspot_count = 0
    for item in tile_records:
        score = item.get("event_score")
        is_hotspot = (
            item.get("status") == "ok"
            and isinstance(score, (int, float))
            and np.isfinite(float(score))
            and float(score) > 0
            and float(score) >= threshold
        )
        item["is_hotspot"] = bool(is_hotspot)
        hotspot_count += 1 if is_hotspot else 0
    return threshold, hotspot_count


def _build_watermass_association(
    tile_records: Sequence[Dict[str, Any]],
    *,
    bins: Sequence[Any],
    hotspot_threshold: Optional[float],
) -> Dict[str, Any]:
    valid_tiles = [item for item in tile_records if item.get("status") == "ok" and item.get("dominant_watermass")]
    hotspot_tiles = [item for item in valid_tiles if item.get("is_hotspot")]
    total_valid = len(valid_tiles)
    total_hotspots = len(hotspot_tiles)

    background_distribution = _distribution_by_key(valid_tiles, bins=bins)
    hotspot_distribution = _distribution_by_key(hotspot_tiles, bins=bins) if hotspot_tiles else {item.id: 0.0 for item in bins}
    enrichment = {
        item.id: float(hotspot_distribution.get(item.id, 0.0) - background_distribution.get(item.id, 0.0))
        for item in bins
    }
    association_score = 0.5 * sum(abs(value) for value in enrichment.values())
    top_associated = max(enrichment.items(), key=lambda entry: entry[1], default=("", 0.0))
    top_bin = next((item for item in bins if item.id == top_associated[0]), None)
    evidence_strength = _association_strength(
        score=association_score,
        valid_tile_count=total_valid,
        hotspot_tile_count=total_hotspots,
    )

    return {
        "valid_tile_count": total_valid,
        "hotspot_tile_count": total_hotspots,
        "hotspot_threshold": hotspot_threshold,
        "association_score": float(association_score),
        "evidence_strength": evidence_strength,
        "top_associated_watermass": top_associated[0] or None,
        "top_associated_watermass_name": top_bin.name if top_bin is not None else None,
        "top_associated_enrichment": float(top_associated[1]),
        "background_distribution": background_distribution,
        "hotspot_distribution": hotspot_distribution,
        "enrichment_by_watermass": enrichment,
        "organized_more_than_background": bool(total_hotspots > 0 and association_score >= 0.2),
    }


def _distribution_by_key(records: Iterable[Dict[str, Any]], *, bins: Sequence[Any]) -> Dict[str, float]:
    materialized = list(records)
    total = max(len(materialized), 1)
    counts = {item.id: 0 for item in bins}
    for item in materialized:
        watermass_id = str(item.get("dominant_watermass") or "").strip()
        if watermass_id in counts:
            counts[watermass_id] += 1
    return {key: float(value / total) for key, value in counts.items()}


def _association_strength(*, score: float, valid_tile_count: int, hotspot_tile_count: int) -> str:
    if valid_tile_count < 2 or hotspot_tile_count == 0:
        return "insufficient"
    # A single hotspot tile can still be highly informative when it departs
    # sharply from the background composition, so keep a high-score escape hatch.
    if score >= 0.65:
        return "strong"
    if score >= 0.35 and hotspot_tile_count >= 3:
        return "strong"
    if score >= 0.2:
        return "moderate"
    return "weak"


def _build_watermass_ts_payload(
    *,
    temp: xr.DataArray,
    salt: xr.DataArray,
    class_indices: xr.DataArray,
    bins: Sequence[Any],
    classification_axes: Sequence[str],
    max_points: int,
    sampling: Literal["random", "head"],
) -> Dict[str, Any]:
    temp_values = np.asarray(temp.values, dtype=float).reshape(-1)
    salt_values = np.asarray(salt.values, dtype=float).reshape(-1)
    class_values = np.asarray(class_indices.values, dtype=int).reshape(-1)
    valid = np.isfinite(temp_values) & np.isfinite(salt_values) & (class_values >= 0)
    indices = np.where(valid)[0]
    if indices.size == 0:
        raise ValueError("No valid watermass-classified temperature-salinity samples are available")

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
    sampled_classes = [bins[int(class_values[index])].id for index in indices]

    return {
        "temperature": sampled_temp.tolist(),
        "salinity": sampled_salt.tolist(),
        "point_classes": sampled_classes,
        "metadata": {
            "title": "Watermass Definition T-S Diagram",
            "temperature_variable": temp.name or "temp",
            "salinity_variable": salt.name or "salt",
            "n_total_points": int(np.sum(valid)),
            "n_sampled_points": int(indices.size),
            "sampling": sampling,
            "classification_axes": list(classification_axes),
            "temperature_range": [float(np.min(sampled_temp)), float(np.max(sampled_temp))],
            "salinity_range": [float(np.min(sampled_salt)), float(np.max(sampled_salt))],
            "watermass_bins": [_serialize_bin(item, classification_axes=classification_axes) for item in bins],
            "class_color_map": {item.id: item.color for item in bins},
        },
    }


def _serialize_bin(item: Any, *, classification_axes: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    payload = {
        "id": item.id,
        "name": item.name,
        "short_name": item.short_name or item.id.upper(),
        "color": item.color,
        "temp_range": [float(item.temp_range[0]), float(item.temp_range[1])],
        "salt_range": [float(item.salt_range[0]), float(item.salt_range[1])],
    }
    include_sigma0 = classification_axes is None or "sigma0" in classification_axes
    if include_sigma0 and isinstance(getattr(item, "sigma0_range", None), list) and len(item.sigma0_range) == 2:
        payload["sigma0_range"] = [float(item.sigma0_range[0]), float(item.sigma0_range[1])]
    return payload


def _extract_grid_shape(association_result: Dict[str, Any]) -> Tuple[int, int]:
    grid_shape = association_result.get("grid_shape")
    if not isinstance(grid_shape, list) or len(grid_shape) != 2:
        raise ValueError("association_result must include grid_shape")
    return max(int(grid_shape[0]), 1), max(int(grid_shape[1]), 1)


def _extract_bins(association_result: Dict[str, Any]) -> list[Any]:
    bins = association_result.get("watermass_bins")
    if not isinstance(bins, list) or not bins:
        raise ValueError("association_result must include watermass_bins")
    return [
        type(
            "SerializedWatermassBin",
            (),
            {
                "id": str(item.get("id")),
                "name": str(item.get("name")),
                "short_name": str(item.get("short_name") or item.get("id")).strip(),
                "color": str(item.get("color") or "#475569"),
                "temp_range": [float(item["temp_range"][0]), float(item["temp_range"][1])],
                "salt_range": [float(item["salt_range"][0]), float(item["salt_range"][1])],
                "sigma0_range": (
                    [float(item["sigma0_range"][0]), float(item["sigma0_range"][1])]
                    if isinstance(item.get("sigma0_range"), list) and len(item.get("sigma0_range")) == 2
                    else []
                ),
            },
        )()
        for item in bins
        if isinstance(item, dict)
    ]


def _axis_centers(tile_bounds: list[Dict[str, Any]], *, axis: Literal["lon", "lat"], count: int) -> list[float]:
    key_min = f"{axis}Min"
    key_max = f"{axis}Max"
    centers = {}
    for item in tile_bounds:
        if not isinstance(item, dict):
            continue
        row = int(item.get("row") or 0)
        col = int(item.get("col") or 0)
        bounds = item.get("bounds")
        if not isinstance(bounds, dict):
            continue
        center = 0.5 * (float(bounds[key_min]) + float(bounds[key_max]))
        index = col if axis == "lon" else row
        centers[index] = float(center)
    return [float(centers[index]) for index in range(1, count + 1)]


def _subregion_grid_payload(grid: Tuple[int, int], cells: list[Dict[str, Any]]) -> Dict[str, Any]:
    bounds_lon: list[float] = []
    bounds_lat: list[float] = []
    valid_count = 0
    skipped_count = 0
    for cell in cells:
        bounds = cell.get("bounds")
        if isinstance(bounds, dict):
            bounds_lon.extend([float(bounds["lonMin"]), float(bounds["lonMax"])])
            bounds_lat.extend([float(bounds["latMin"]), float(bounds["latMax"])])
        if str(cell.get("status") or "").strip().lower() == "ok":
            valid_count += 1
        else:
            skipped_count += 1
    result: Dict[str, Any] = {
        "gridShape": [int(grid[0]), int(grid[1])],
        "cells": cells,
        "validCount": valid_count,
        "skippedCount": skipped_count,
    }
    if bounds_lon and bounds_lat:
        result["bounds"] = {
            "lonMin": float(min(bounds_lon)),
            "lonMax": float(max(bounds_lon)),
            "latMin": float(min(bounds_lat)),
            "latMax": float(max(bounds_lat)),
        }
    return result


def _numeric_statistics(values: np.ndarray) -> Dict[str, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {}
    return {
        "mean": float(np.nanmean(valid)),
        "std": float(np.nanstd(valid)),
        "min": float(np.nanmin(valid)),
        "max": float(np.nanmax(valid)),
    }


def _json_bounds(bounds: Any) -> Optional[Dict[str, float]]:
    if not isinstance(bounds, dict):
        return None
    try:
        return {
            "lonMin": float(bounds["lonMin"]),
            "lonMax": float(bounds["lonMax"]),
            "latMin": float(bounds["latMin"]),
            "latMax": float(bounds["latMax"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        return None
    return float(value)


def _format_fraction(value: Any) -> str:
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value) * 100.0:.1f}%"


def _format_event_score_label(value: Any, unit: str) -> str:
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        return f"NA {unit}".strip()
    return f"{float(value):.3g} {unit}".strip()


def _tile_short_label(row: int, col: int, grid: Tuple[int, int]) -> str:
    if grid == (2, 2):
        return {
            (2, 1): "NW",
            (2, 2): "NE",
            (1, 1): "SW",
            (1, 2): "SE",
        }.get((row, col), f"R{row}C{col}")
    return f"R{row}C{col}"


def _event_type_label(event_type: str) -> str:
    mapping = {
        "algal_bloom": "Bloom",
        "heatwave": "Heatwave",
        "hypoxia": "Hypoxia",
        "upwelling": "Upwelling",
    }
    return mapping.get(event_type, event_type.replace("_", " ").title() or "Event")
