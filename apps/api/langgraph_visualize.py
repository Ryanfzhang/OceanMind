"""Manual map and PDF endpoints retained alongside the LangGraph query loop."""

from __future__ import annotations

import logging
import math
from typing import Literal

import numpy as np
import xarray as xr
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from apps.api.report_export import ConversationReportRequest, export_report_response
from apps.api.langgraph_query import QueryService, get_query_service
from domain.ocean.analysis.profile.extract import (
    identify_mixed_layer_depth,
    identify_pycnocline_depth,
    identify_thermocline_depth,
)
from domain.ocean.analysis.spatial.analysis import compute_spatial_field, extract_timeseries
from domain.ocean.analysis.sections import extract_transect_section
from domain.ocean.analysis.timeseries.extract import (
    compute_layer_mean,
    extract_point_timeseries,
    extract_regional_mean,
)
from domain.ocean.data_access.assemble import assemble_dataset
from domain.ocean.data_access.load import load_dataset
from domain.ocean.diagnostics.compute import compute_density
from domain.ocean.preprocessing.filter import build_polygon_mask


router = APIRouter()
LOGGER = logging.getLogger(__name__)
FEATURES = {"mixed_layer", "thermocline", "pycnocline"}


class Region(BaseModel):
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


class Point(BaseModel):
    lat: float
    lon: float


class VisualizationRequest(BaseModel):
    dataset: str = Field(min_length=1)
    variable: str = Field(min_length=1)
    time_range: tuple[str, str]
    region: Region
    depth_mode: Literal["fixed", "feature", "layer_mean"]
    depth_range: tuple[float, float] = (0.0, 0.0)
    feature: Literal["mixed_layer", "thermocline", "pycnocline"] = "thermocline"
    layer_mean_label: str = "surface -> thermocline"
    selected_point: Point | None = None
    selection_mode: Literal["box", "point", "transect", "polygon", "none"] = "box"
    transect_points: list[tuple[float, float]] | None = None
    polygon_points: list[tuple[float, float]] | None = None
    search_depth_range: tuple[float, float] = (0.0, -300.0)


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load(request: VisualizationRequest, variable: str, *, search_depth: bool = False):
    return load_dataset(
        dataset=request.dataset,
        variable=variable,
        lon_range=(request.region.lon_min, request.region.lon_max),
        lat_range=(request.region.lat_min, request.region.lat_max),
        time_range=request.time_range,
        depth_range=request.search_depth_range if search_depth else None,
    )


def _feature(request: VisualizationRequest, name: str):
    temp = _load(request, "temp", search_depth=True)
    if name == "thermocline":
        return identify_thermocline_depth(temp)
    salt = _load(request, "salt", search_depth=True)
    density = compute_density(assemble_dataset({"temp": temp, "salt": salt}))
    if name == "mixed_layer":
        return identify_mixed_layer_depth(density)
    if name == "pycnocline":
        return identify_pycnocline_depth(density)
    raise ValueError(f"Unknown vertical feature: {name}")


def _layer(request: VisualizationRequest):
    parts = [part.strip().lower() for part in request.layer_mean_label.split("->")]
    if len(parts) != 2 or parts[0] not in FEATURES | {"surface"} or parts[1] not in FEATURES:
        raise ValueError("Choose a supported layer, such as surface -> thermocline")
    if parts[0] == parts[1]:
        raise ValueError("Layer boundaries must differ")
    field = _load(request, request.variable, search_depth=True)
    known: dict[str, object] = {}

    def boundary(name: str):
        if name not in known:
            known[name] = _feature(request, name)
        return known[name]

    kwargs = {"upper_bound_value": 0.0} if parts[0] == "surface" else {"upper_bound_field": boundary(parts[0])}
    kwargs["lower_bound_field"] = boundary(parts[1])
    return compute_layer_mean(field, **kwargs)


def _map_field(result: dict, title: str, dates: tuple[str, str], depth_label: str) -> dict:
    lon = np.asarray(result["lon"], dtype=float)
    lat = np.asarray(result["lat"], dtype=float)
    values = np.asarray(result["values"], dtype=float)
    if values.shape != (len(lat), len(lon)) or not len(lat) or not len(lon):
        raise ValueError("The selection has no spatial field")
    # The browser renders this grid directly; cap the response, including nulls for land.
    rows = np.unique(np.linspace(0, len(lat) - 1, min(len(lat), 100), dtype=int))
    cols = np.unique(np.linspace(0, len(lon) - 1, min(len(lon), 100), dtype=int))
    sampled = values[np.ix_(rows, cols)]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("The selection contains no finite ocean values")
    low, high = float(finite.min()), float(finite.max())
    if low == high:
        pad = max(abs(low) * 1e-6, 1e-9)
        low, high = low - pad, high + pad
    metadata = result.get("metadata") or {}
    return {
        "lon": lon[cols].tolist(),
        "lat": lat[rows].tolist(),
        "values": [[_finite(value) for value in row] for row in sampled],
        "label": title,
        "variable": metadata.get("variable") or title,
        "units": metadata.get("units") or "",
        "statistics": {key: _finite(value) for key, value in (metadata.get("statistics") or {}).items()},
        "depthLabel": depth_label,
        "timeLabel": dates[0] if dates[0] == dates[1] else f"{dates[0]} to {dates[1]}",
        "colorScale": {"min": low, "max": high, "label": title,
                       "units": metadata.get("units") or "", "colormap": "ocean_diverging"},
        "bounds": [[float(lat[rows[0]]), float(lon[cols[0]])],
                   [float(lat[rows[-1]]), float(lon[cols[-1]])]],
    }


def _series(result: dict) -> list[dict]:
    return [
        {"label": str(day)[:10], "value": number}
        for day, value in zip(result.get("times", []), result.get("values", []))
        if (number := _finite(value)) is not None
    ][:18]


def _visualize(request: VisualizationRequest) -> dict:
    if request.region.lon_min >= request.region.lon_max or request.region.lat_min >= request.region.lat_max:
        raise ValueError("Region minimum must be below maximum")
    if request.depth_mode == "fixed":
        field = _load(request, request.variable)
        depth_label = f"{request.depth_range[0]:g} m" if request.depth_range[0] == request.depth_range[1] else f"{request.depth_range[0]:g} to {request.depth_range[1]:g} m"
        depth_aggregation = "surface" if request.depth_range == (0.0, 0.0) else "mean"
        options = {"depth_range": request.depth_range, "depth_aggregation": depth_aggregation}
        title = f"{request.variable.title()} Map"
    elif request.depth_mode == "feature":
        field = _feature(request, request.feature)
        depth_label = request.feature.replace("_", " ")
        options = {}
        title = f"{depth_label.title()} Depth"
    else:
        field = _layer(request)
        depth_label = request.layer_mean_label
        options = {}
        title = f"{request.variable.title()} Layer Mean"

    mode = request.selection_mode
    if mode == "point" and request.selected_point is None:
        raise ValueError("Point selection requires selected_point")
    if mode == "polygon" and (request.polygon_points is None or len(request.polygon_points) < 3):
        raise ValueError("Polygon selection requires at least three [lon, lat] points")
    if mode == "transect" and (request.transect_points is None or len(request.transect_points) < 2):
        raise ValueError("Transect selection requires at least two [lon, lat] points")

    mask = build_polygon_mask(field, request.polygon_points) if mode == "polygon" else None
    spatial = compute_spatial_field(field, time_aggregation="mean", mask=mask, **options)
    map_field = _map_field(spatial, title, request.time_range, depth_label)
    series: list[dict] = []
    workspace_data: dict = {"mapField": map_field}
    overlays: list[dict] = []
    if "time" in field.dims:
        if mode == "point":
            point = request.selected_point
            assert point is not None
            series = _series(extract_point_timeseries(field, lon=point.lon, lat=point.lat, **options))
        elif mode == "polygon":
            series = _series(extract_timeseries(
                field,
                lon_range=(request.region.lon_min, request.region.lon_max),
                lat_range=(request.region.lat_min, request.region.lat_max),
                mask=mask,
                **options,
            ))
        elif mode != "transect":
            series = _series(extract_regional_mean(
                field,
                (request.region.lon_min, request.region.lon_max),
                (request.region.lat_min, request.region.lat_max),
                **options,
            ))

    if mode == "point":
        point = request.selected_point
        assert point is not None
        overlays.append({"id": "manual_point", "eventType": "selection", "title": "Selected point",
                         "shape": "point", "center": point.model_dump(), "details": ["Point time series"]})
    elif mode == "transect":
        points = request.transect_points
        assert points is not None
        mean_field = xr.DataArray(
            spatial["values"], coords={"lat": spatial["lat"], "lon": spatial["lon"]},
            dims=("lat", "lon"), name=request.variable,
        )
        section = extract_transect_section(mean_field, points, n_samples=100)
        distances = section["distance_km"]
        values = np.asarray(section["values"], dtype=float)
        workspace_data.update({
            "sectionDistanceKm": distances,
            "sectionRows": [{"label": "Time mean", "values": [_finite(value) for value in values]}],
            "sectionAxisTitle": "Distance along transect (km)",
            "sectionSliceLabel": depth_label,
        })
        series = [
            {"label": f"{distance:.1f} km", "value": number}
            for distance, value in zip(distances, values)
            if (number := _finite(value)) is not None
        ]
        path = [{"lon": lon, "lat": lat} for lon, lat in points]
        overlays.append({"id": "manual_transect", "eventType": "selection", "title": "Selected transect",
                         "shape": "polyline", "center": path[0], "path": path,
                         "details": [f"{len(points)} vertices", f"{distances[-1]:.1f} km"]})

    workspace_data.update({"referenceSeries": series, "resultSeries": series, "eventOverlays": overlays})
    selection = {"point": "Point", "transect": "Transect", "polygon": "Polygon"}.get(mode)
    if selection:
        title = f"{title} · {selection}"
    card = {
        "id": "manual_visualization",
        "title": title,
        "type": "spatial_field_result",
        "headline": f"{title}: {depth_label}",
        "description": f"{selection or 'Region'} selection for {request.time_range[0]} to {request.time_range[1]}.",
        "renderer": "reference",
        "metrics": [{"label": "Depth", "value": depth_label}],
        "surface": "map",
    }
    return {
        "status": "completed",
        "result_cards": [card],
        "active_result_id": card["id"],
        "workspace_data": workspace_data,
    }


@router.post("/visualize")
def run_visualization(request: VisualizationRequest) -> dict:
    try:
        return _visualize(request)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Manual visualization failed")
        raise HTTPException(status_code=500, detail="Manual visualization failed; inspect backend logs") from exc


@router.post("/report/export")
def export_report(
    request: ConversationReportRequest,
    service: QueryService = Depends(get_query_service),
):
    session = None
    if request.conversation_id:
        try:
            record = service.store.resolve(request.conversation_id)
            session = service.store.open_analysis(record)
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Conversation is unavailable") from exc
    try:
        return export_report_response(request, session)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail="Saved analysis is unavailable") from exc
