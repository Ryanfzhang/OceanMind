from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


def hydrate_workspace_geometry_extracted_params(
    extracted_params: Mapping[str, Any],
    additional_context: Mapping[str, Any],
) -> Dict[str, Any]:
    hydrated = dict(extracted_params or {})

    polygon_points = _find_polygon_points(hydrated, additional_context)
    if polygon_points is not None:
        hydrated["mask_polygon"] = polygon_points
        hydrated.setdefault("drawn_polygon_points", polygon_points)
        hydrated.setdefault("polygon_points", polygon_points)
        hydrated.setdefault("region_selection_type", "polygon")
        if "region" not in hydrated and "lon_range" not in hydrated and "lat_range" not in hydrated:
            bounds = _geometry_bounds(polygon_points, min_points=3)
            if bounds is not None:
                lon_range, lat_range = bounds
                hydrated["region"] = {"lon_range": lon_range, "lat_range": lat_range}

    transect_points = _find_transect_points(hydrated, additional_context)
    if transect_points is not None:
        hydrated["transect_points"] = transect_points
        hydrated.setdefault("drawn_transect_points", transect_points)
        hydrated.setdefault("transect_selection_type", "transect")

    return hydrated


def _find_polygon_points(*candidates: Any) -> Optional[List[List[float]]]:
    seen: Set[int] = set()
    for candidate in candidates:
        polygon = _extract_geometry_points(
            candidate,
            seen=seen,
            direct_keys=("mask_polygon", "drawn_polygon_points", "polygon_points"),
            selection_key="selected_region",
            selection_type="polygon",
            min_points=3,
        )
        if polygon is not None:
            return polygon
    return None


def _find_transect_points(*candidates: Any) -> Optional[List[List[float]]]:
    seen: Set[int] = set()
    for candidate in candidates:
        transect = _extract_geometry_points(
            candidate,
            seen=seen,
            direct_keys=("transect_points", "drawn_transect_points"),
            selection_key="selected_transect",
            selection_type="transect",
            min_points=2,
        )
        if transect is not None:
            return transect
    return None


def _extract_geometry_points(
    candidate: Any,
    *,
    seen: Set[int],
    direct_keys: Sequence[str],
    selection_key: str,
    selection_type: str,
    min_points: int,
) -> Optional[List[List[float]]]:
    points = _coerce_geometry_points(candidate, min_points=min_points)
    if points is not None:
        return points
    if not isinstance(candidate, Mapping):
        return None

    marker = id(candidate)
    if marker in seen:
        return None
    seen.add(marker)

    for key in direct_keys:
        points = _coerce_geometry_points(candidate.get(key), min_points=min_points)
        if points is not None:
            return points

    selected_geometry = candidate.get(selection_key)
    if isinstance(selected_geometry, Mapping):
        geometry_type = str(selected_geometry.get("type") or "").lower()
        points = _coerce_geometry_points(selected_geometry.get("points"), min_points=min_points)
        if points is not None and (geometry_type in {"", selection_type}):
            return points

    selected_region = candidate.get("selected_region")
    if selection_key != "selected_region" and isinstance(selected_region, Mapping):
        geometry_type = str(selected_region.get("type") or "").lower()
        points = _coerce_geometry_points(selected_region.get("points"), min_points=min_points)
        if points is not None and geometry_type == selection_type:
            return points

    workspace_selection = candidate.get("workspace_selection")
    if isinstance(workspace_selection, Mapping):
        points = _extract_geometry_points(
            workspace_selection,
            seen=seen,
            direct_keys=direct_keys,
            selection_key=selection_key,
            selection_type=selection_type,
            min_points=min_points,
        )
        if points is not None:
            return points

    for key in ("workspace_context", "additional_context"):
        nested = candidate.get(key)
        if isinstance(nested, Mapping):
            points = _extract_geometry_points(
                nested,
                seen=seen,
                direct_keys=direct_keys,
                selection_key=selection_key,
                selection_type=selection_type,
                min_points=min_points,
            )
            if points is not None:
                return points

    return None


def _coerce_geometry_points(value: Any, *, min_points: int) -> Optional[List[List[float]]]:
    if not isinstance(value, (list, tuple)) or len(value) < min_points:
        return None
    points: List[List[float]] = []
    for point in value:
        lon: Any
        lat: Any
        if isinstance(point, Mapping):
            lon = point.get("lon")
            lat = point.get("lat")
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            lon = point[0]
            lat = point[1]
        else:
            return None
        try:
            lon_float = float(lon)
            lat_float = float(lat)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(lon_float) and math.isfinite(lat_float)):
            return None
        points.append([lon_float, lat_float])
    return points if len(points) >= min_points else None


def _geometry_bounds(points: Sequence[Sequence[float]], *, min_points: int) -> Optional[Tuple[List[float], List[float]]]:
    if len(points) < min_points:
        return None
    lons = [float(point[0]) for point in points]
    lats = [float(point[1]) for point in points]
    return [min(lons), max(lons)], [min(lats), max(lats)]
