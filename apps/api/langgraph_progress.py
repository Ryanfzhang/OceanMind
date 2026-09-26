"""Project analysis runtime stages onto the existing workspace event protocol."""

from __future__ import annotations

import base64
import json
import math
import time
from copy import deepcopy
from collections import OrderedDict
from typing import Any, Callable

from packages.agent_loop.analysis import AnalysisSession
from packages.analysis_runtime.artifacts import _stored_payload


PREVIEW_JSON_BYTES = 4 * 1024 * 1024
PREVIEW_GRID_SIDE = 64


def _map_values(values: Any) -> list[list[float | None]]:
    return [[float(value) if math.isfinite(value) else None for value in row]
            for row in values]


def _loaded_field_metrics(path: Any, summary: dict) -> list[dict[str, str]]:
    """Describe a loaded array without turning it into an analysis map."""
    import numpy as np
    import xarray as xr

    dims = summary.get("dims") or []
    shape = summary.get("shape") or []
    metrics = [
        {"label": "Dimensions", "value": " × ".join(map(str, dims))},
        {"label": "Shape", "value": " × ".join(map(str, shape))},
    ]
    if summary.get("units"):
        metrics.append({"label": "Units", "value": str(summary["units"])})
    with xr.open_dataarray(path) as field:
        sampled = field.isel(**{
            dim: slice(None, None, max(1, math.ceil(size / (64 if dim in {"lat", "lon"} else 8))))
            for dim, size in field.sizes.items()
        })
        values = np.asarray(sampled.values, dtype=float)
    finite = values[np.isfinite(values)]
    metrics.append({"label": "Sample valid", "value": f"{finite.size}/{values.size}"})
    if finite.size:
        metrics.extend({"label": label, "value": f"{value:.4g}"} for label, value in (
            ("Sample min", float(finite.min())),
            ("Sample mean", float(finite.mean())),
            ("Sample max", float(finite.max())),
        ))
    return metrics


def _array_preview(path: Any, name: str, *, spatial_slice_only: bool = False) -> tuple[str, dict] | None:
    """Read at most a small slice of a saved field for the existing UI charts."""
    import numpy as np
    import xarray as xr
    from domain.ocean.visualization.landmask import mask_land_for_map_preview

    with xr.open_dataarray(path) as source:
        field = source.squeeze(drop=True)
        if {"lat", "lon"}.issubset(field.dims):
            extras = [dim for dim in field.dims if dim not in {"lat", "lon"}]
            if spatial_slice_only and extras:
                return None
            is_mask = (source.dtype.kind == "b" or
                       str(source.name or name).lower().endswith("_mask"))
            field = field.isel(**{
                dim: slice(None, None, max(1, math.ceil(field.sizes[dim] / PREVIEW_GRID_SIDE)))
                for dim in ("lat", "lon")
            })
            if extras:
                if is_mask:
                    field = field.any(extras) if field.dtype.kind == "b" else field.max(extras)
                else:
                    field = field.isel(**{
                        dim: slice(None, None, max(1, math.ceil(field.sizes[dim] / 8)))
                        for dim in extras
                    }).mean(extras, skipna=True)
                    name = f"{name} (sampled mean preview)"
            field = field.transpose("lat", "lon")
            if field.sizes["lat"] < 2 or field.sizes["lon"] < 2:
                return None
            lon = np.asarray(field["lon"].values, dtype=float)
            lat = np.asarray(field["lat"].values, dtype=float)
            if is_mask:
                mask_field = _mask_map_payload(
                    field.values, lon.tolist(), lat.tolist(), None,
                    f"{name.replace('_', ' ').title()} footprint", str(source.name or name))
                return ("summary", {"mapField": mask_field}) if mask_field else None
            values = np.asarray(field.values, dtype=float)
            if (lon.ndim != 1 or lat.ndim != 1 or values.shape != (len(lat), len(lon))
                    or not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat))
                    or not np.any(np.isfinite(values))):
                return None
            values, land_image = mask_land_for_map_preview(lon, lat, values)
            map_field = {
                "lon": lon.tolist(), "lat": lat.tolist(),
                "values": _map_values(values),
                "label": name.replace("_", " ").title(), "variable": str(source.name or name),
                "units": str(source.attrs.get("units") or ""),
                "bounds": [[float(lat.min()), float(lon.min())],
                           [float(lat.max()), float(lon.max())]],
            }
            if land_image:
                map_field["landMaskImage"] = land_image
            return "summary", {"mapField": map_field}
        if field.ndim == 1 and field.dims[0] == "time":
            stride = max(1, math.ceil(field.sizes["time"] / 100))
            field = field.isel(time=slice(None, None, stride))
            values = np.asarray(field.values, dtype=float)
            times = field["time"].values
            series = [{"label": str(t)[:19], "value": float(v)}
                      for t, v in zip(times, values) if math.isfinite(v)]
            if series:
                return "timeseries", {"resultSeries": series,
                                      "seriesLabels": {"result": name.replace("_", " ")}}
        if field.ndim == 1 and field.dims[0] == "depth":
            stride = max(1, math.ceil(field.sizes["depth"] / 100))
            field = field.isel(depth=slice(None, None, stride))
            depths = np.asarray(field["depth"].values, dtype=float)
            values = np.asarray(field.values, dtype=float)
            points = [{"depth": float(depth), "value": float(value)}
                      for depth, value in zip(depths, values)
                      if math.isfinite(depth) and math.isfinite(value)]
            if points:
                return "profile", {"profileSeries": points}
    return None


def _preview_values(value: Any, root: Any) -> Any:
    """Decode a numeric array lazily when the artifact stores it in a sidecar."""
    import numpy as np

    if isinstance(value, dict) and set(value) == {"__ndarray_file__"}:
        return np.load(_stored_payload(root, value["__ndarray_file__"]),
                       allow_pickle=False, mmap_mode="r")
    if isinstance(value, dict) and value.get("__ndarray__") is True:
        raw = base64.b64decode(value["data_b64"])
        return np.frombuffer(raw, dtype=value["dtype"]).reshape(value["shape"])
    return value


def _transport_map_style(value: Any, metadata: dict, root: Any,
                         lat: Any, lon: Any, rows: Any, cols: Any) -> dict:
    """Carry tool rendering metadata, or derive regional display scales for a broad China Seas map."""
    import numpy as np

    if str(metadata.get("variable") or "").lower() != "transport_streamfunction":
        return {}
    scales = metadata.get("regional_color_scales")
    rendering = metadata.get("transport_rendering")
    if not scales or not rendering:
        from domain.ocean.analysis.transports import (
            _domain_spans_gan_fig10_wpo_and_china_seas,
            _gan_fig10_display_area_masks,
            _gan_fig10_regional_color_scales,
            _gan_fig10_transport_rendering_metadata,
        )
    if (not scales or not rendering) and _domain_spans_gan_fig10_wpo_and_china_seas(
            lat=lat, lon=lon):
        regions = _gan_fig10_display_area_masks(
            lat=lat, lon=lon, wet_mask=np.isfinite(value))
        if len(regions) >= 2:
            # The original Fig. 10 masks stop at 42.5 N; keep the rest of a
            # broader requested domain visible in the open-ocean color scale.
            covered = np.logical_or.reduce([region["mask"] for region in regions])
            regions[1] = {**regions[1], "mask": regions[1]["mask"] | (np.isfinite(value) & ~covered)}
        scales = _gan_fig10_regional_color_scales(display_values=value, regions=regions)
        rendering = _gan_fig10_transport_rendering_metadata(regions=regions)
    if not isinstance(scales, list) or not isinstance(rendering, dict):
        return {}

    def sample_mask(mask: Any) -> list[list[bool]] | None:
        decoded = np.asarray(_preview_values(mask, root), dtype=bool)
        if decoded.shape != value.shape:
            return None
        return decoded[np.ix_(rows, cols)].tolist()

    regions = []
    for region in rendering.get("filled_regions") or []:
        if not isinstance(region, dict):
            continue
        mask = sample_mask(region.get("mask"))
        if mask is not None:
            regions.append({
                "id": region.get("id"), "region": region.get("region"),
                "label": region.get("label"),
                "scaleStrategy": region.get("scaleStrategy") or region.get("scale_strategy"),
                "mask": mask,
            })
    if not regions:
        return {}
    return {
        "regionalColorScales": scales,
        "transportRendering": {
            "mode": rendering.get("mode"),
            "filledRegion": rendering.get("filled_region"),
            "filledMask": sample_mask(rendering.get("filled_mask")),
            "filledRegions": regions,
            "filledColormap": rendering.get("filled_colormap"),
        },
    }


def _map_payload(value: dict, root: Any, name: str) -> dict | None:
    import numpy as np
    from domain.ocean.visualization.landmask import mask_land_for_map_preview

    lon, lat = value.get("lon"), value.get("lat")
    field = value.get("values", value.get("slope"))
    if not isinstance(lon, list) or not isinstance(lat, list) or field is None:
        return None
    values = _preview_values(field, root)
    if isinstance(values, list):
        values = np.asarray(values)
    if (np.ndim(values) != 2 or len(lon) < 2 or len(lat) < 2
            or np.shape(values) != (len(lat), len(lon))):
        return None
    row_step = max(1, math.ceil(len(lat) / PREVIEW_GRID_SIDE))
    col_step = max(1, math.ceil(len(lon) / PREVIEW_GRID_SIDE))
    rows = np.arange(0, len(lat), row_step)
    cols = np.arange(0, len(lon), col_step)
    x = np.asarray(lon, dtype=float)[cols]
    y = np.asarray(lat, dtype=float)[rows]
    sample = np.asarray(values[np.ix_(rows, cols)], dtype=float)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.any(np.isfinite(sample))):
        return None
    metadata = value.get("metadata") or {}
    sample, land_image = mask_land_for_map_preview(x, y, sample)
    result = {"lon": x.tolist(), "lat": y.tolist(),
            "values": _map_values(sample),
            "label": name.replace("_", " ").title(),
            "variable": str(metadata.get("variable") or name),
            "units": str(metadata.get("units") or metadata.get("unit") or ""),
            "bounds": [[float(y.min()), float(x.min())],
                       [float(y.max()), float(x.max())]]}
    if land_image:
        result["landMaskImage"] = land_image
    result.update(_transport_map_style(np.asarray(values, dtype=float), metadata, root,
                                       np.asarray(lat, dtype=float), np.asarray(lon, dtype=float),
                                       rows, cols))
    return result


def _mask_map_payload(mask_value: Any, lon: Any, lat: Any, root: Any,
                      label: str, variable: str) -> dict | None:
    """Project any georeferenced binary mask onto a sparse map layer."""
    import numpy as np

    if not isinstance(lon, list) or not isinstance(lat, list) or len(lon) < 2 or len(lat) < 2:
        return None
    x, y = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None
    if isinstance(mask_value, dict) and "__dataarray_file__" in mask_value:
        import xarray as xr

        with xr.open_dataarray(_stored_payload(root, mask_value["__dataarray_file__"])) as array:
            mask = np.asarray(array.values)
    else:
        mask = np.asarray(_preview_values(mask_value, root))
    if (mask.ndim < 2 or mask.shape[-2:] != (len(lat), len(lon))
            or mask.dtype.kind not in "bifu"):
        return None
    row_step = max(1, math.ceil(len(lat) / PREVIEW_GRID_SIDE))
    col_step = max(1, math.ceil(len(lon) / PREVIEW_GRID_SIDE))
    rows = np.arange(0, len(lat), row_step)
    cols = np.arange(0, len(lon), col_step)
    occupied = np.zeros((len(rows), len(cols)), dtype=bool)
    for field in mask.reshape((-1, len(lat), len(lon))):
        if field.dtype.kind == "b":
            hits = field
        else:
            valid = np.isfinite(field)
            if not np.all(~valid | (field == 0) | (field == 1)):
                return None
            hits = valid & (field == 1)
        occupied |= np.logical_or.reduceat(
            np.logical_or.reduceat(hits, rows, axis=0), cols, axis=1)
    if not np.any(occupied):
        return None
    return {
        "lon": x[cols].tolist(), "lat": y[rows].tolist(),
        "values": [[1.0 if hit else None for hit in row] for row in occupied],
        "label": label, "variable": variable, "units": "mask cells",
        "bounds": [[float(y.min()), float(x.min())], [float(y.max()), float(x.max())]],
        "discreteLegend": [{"value": 1, "label": "Mask cells", "color": "#dc2626"}],
    }


def _structured_mask_payload(value: dict, root: Any, name: str) -> dict | None:
    """Use a saved mask, preferring accepted events over candidate masks."""
    coordinates = value.get("coordinates")
    if not isinstance(coordinates, dict):
        return None
    mask_names = sorted((key for key in value
                         if key == "mask" or key.endswith("_mask")),
                        key=lambda key: (key != "event_mask", key))
    for mask_name in mask_names:
        title = name.replace("_", " ").title()
        kind = ("event footprint" if mask_name == "event_mask" else
                "candidate mask" if isinstance(value.get("events"), list) else "mask footprint")
        try:
            field = _mask_map_payload(value[mask_name], coordinates.get("lon"),
                                      coordinates.get("lat"), root,
                                      f"{title} {kind} (any selected slice)", mask_name)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if field:
            return field
    return None


def _json_preview(root: Any, path: Any, name: str) -> tuple[str, dict] | None:
    import numpy as np

    if path.stat().st_size > PREVIEW_JSON_BYTES:
        return None
    with path.open(encoding="utf-8") as file:
        value = json.load(file).get("value")
    if not isinstance(value, dict):
        return None
    for field_name in ("data", "slope"):
        field = value.get(field_name)
        if isinstance(field, dict) and "__dataarray_file__" in field:
            preview = _array_preview(_stored_payload(root, field["__dataarray_file__"]),
                                     f"{name} {field_name}" if field_name == "slope" else name)
            if preview:
                return preview
    mask_field = _structured_mask_payload(value, root, str(value.get("event_type") or name))
    if mask_field:
        return "summary", {"mapField": mask_field}
    if all(isinstance(value.get(key), dict)
           for key in ("positive_composite", "negative_composite", "difference")):
        fields = []
        for key in ("positive_composite", "negative_composite", "difference"):
            map_field = _map_payload(value[key], root, f"{name} {key}")
            if map_field:
                fields.append({"id": key, "title": map_field["label"], "mapField": map_field})
        if fields:
            return "composite", {"compositeFields": fields,
                                 "mapField": fields[-1]["mapField"]}
    modes = value.get("modes")
    if isinstance(modes, list):
        eof_modes = []
        variance = []
        for mode in modes[:3]:
            if not isinstance(mode, dict):
                continue
            pattern, series = mode.get("spatial_pattern"), mode.get("time_series")
            if not (isinstance(pattern, dict) and "__dataarray_file__" in pattern
                    and isinstance(series, dict) and "__dataarray_file__" in series):
                continue
            number = mode.get("mode_number", len(eof_modes) + 1)
            map_preview = _array_preview(_stored_payload(root, pattern["__dataarray_file__"]),
                                         f"EOF Mode {number}")
            pc_preview = _array_preview(_stored_payload(root, series["__dataarray_file__"]),
                                        f"PC {number}")
            if not map_preview or not pc_preview:
                continue
            map_field = map_preview[1].get("mapField")
            pc_points = pc_preview[1].get("resultSeries")
            if not map_field or not pc_points:
                continue
            label = f"{float(mode.get('variance_explained', 0)):.1f}% variance"
            variance.append({"label": f"Mode {number}", "value": label})
            eof_modes.append({"id": f"mode_{number}", "title": f"EOF Mode {number}",
                              "varianceLabel": label, "mapField": map_field,
                              "pcSeries": pc_points})
        if eof_modes:
            return "eof", {"eofModes": eof_modes, "eofVariance": variance,
                           "eofPcSeries": [{"day": point["label"], "value": point["value"]}
                                           for point in eof_modes[0]["pcSeries"]],
                           "mapField": eof_modes[0]["mapField"]}
    labels = value.get("times")
    if isinstance(labels, list) and "values" in value:
        values = _preview_values(value["values"], root)
        if np.ndim(values) == 1 and len(labels) == len(values):
            stride = max(1, math.ceil(len(labels) / 100))
            series = [{"label": str(label)[:19], "value": float(number)}
                      for label, number in zip(labels[::stride], values[::stride])
                      if isinstance(number, (int, float, np.number)) and math.isfinite(number)]
            if series:
                workspace = {"resultSeries": series,
                             "seriesLabels": {"result": name.replace("_", " ")}}
                trend = value.get("trend_line")
                if isinstance(trend, list) and len(trend) == len(labels):
                    workspace["anomalySeries"] = [
                        {"label": str(label)[:19], "value": float(number)}
                        for label, number in zip(labels[::stride], trend[::stride])
                        if isinstance(number, (int, float)) and math.isfinite(number)]
                    workspace["seriesLabels"]["compare"] = "Trend line"
                return "timeseries", workspace
    for label_key, value_key in (("labels", "values"), ("lags", "correlations"),
                                 ("frequency", "power")):
        labels = value.get(label_key)
        numbers = value.get(value_key)
        if isinstance(labels, list) and numbers is not None:
            numbers = _preview_values(numbers, root)
            if np.ndim(numbers) == 1 and len(labels) == len(numbers):
                stride = max(1, math.ceil(len(labels) / 100))
                series = [{"label": str(label)[:19], "value": float(number)}
                          for label, number in zip(labels[::stride], numbers[::stride])
                          if isinstance(number, (int, float, np.number)) and math.isfinite(number)]
                if series:
                    return "timeseries", {"resultSeries": series,
                                          "seriesLabels": {"result": name.replace("_", " ")}}
    if isinstance(value.get("bin_centers"), list) and isinstance(value.get("density"), list):
        bins = [{"label": f"{center:.4g}", "value": float(density)}
                for center, density in zip(value["bin_centers"], value["density"])
                if isinstance(center, (int, float)) and isinstance(density, (int, float))
                and math.isfinite(center) and math.isfinite(density)]
        if bins:
            return "histogram", {"histogramBins": bins[:100]}
    if isinstance(value.get("depth"), list) and "values" in value and "distance_km" not in value:
        values = _preview_values(value["values"], root)
        if np.ndim(values) == 1 and len(value["depth"]) == len(values):
            points = [{"depth": float(depth), "value": float(number)}
                      for depth, number in zip(value["depth"], values)
                      if math.isfinite(depth) and math.isfinite(number)]
            if points:
                return "profile", {"profileSeries": points[:100]}
    if isinstance(value.get("temperature"), list) and isinstance(value.get("salinity"), list):
        temperature, salinity = value["temperature"], value["salinity"]
        if len(temperature) == len(salinity):
            stride = max(1, math.ceil(len(temperature) / 2500))
            colors = value.get("color_values")
            classes = value.get("point_classes")
            points = []
            for index in range(0, len(temperature), stride):
                t, s = temperature[index], salinity[index]
                if not (isinstance(t, (int, float)) and isinstance(s, (int, float))
                        and math.isfinite(t) and math.isfinite(s)):
                    continue
                point = {"temperature": float(t), "salinity": float(s)}
                if isinstance(colors, list) and index < len(colors):
                    color = colors[index]
                    if isinstance(color, (int, float)) and math.isfinite(color):
                        point["colorValue"] = float(color)
                if isinstance(classes, list) and index < len(classes) and isinstance(classes[index], str):
                    point["pointClass"] = classes[index]
                points.append(point)
            if points:
                metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
                color_range = metadata.get("color_range")
                return "ts_diagram", {
                    "tsDiagramPoints": points,
                    "tsDiagramTemperatureLabel": str(metadata.get("temperature_variable") or "Temperature"),
                    "tsDiagramSalinityLabel": str(metadata.get("salinity_variable") or "Salinity"),
                    "tsDiagramColorLabel": metadata.get("color_variable"),
                    "tsDiagramColorRange": color_range if isinstance(color_range, list) and len(color_range) == 2 else None,
                    "tsDiagramPointClasses": [point["pointClass"] for point in points if "pointClass" in point],
                    "tsDiagramClassColorMap": metadata.get("class_color_map") or {},
                    "tsDiagramWatermassBins": metadata.get("watermass_bins") or [],
                }
    if isinstance(value.get("time"), list) and isinstance(value.get("spatial_coord"), list) and "values" in value:
        values = _preview_values(value["values"], root)
        time, coordinate = value["time"], value["spatial_coord"]
        if np.ndim(values) == 2 and np.shape(values) == (len(time), len(coordinate)):
            time_step = max(1, math.ceil(len(time) / PREVIEW_GRID_SIDE))
            coord_step = max(1, math.ceil(len(coordinate) / PREVIEW_GRID_SIDE))
            sampled = np.asarray(values[::time_step, ::coord_step], dtype=float)
            rows = [{"depthLabel": str(label), "depthValue": float(label),
                     "values": [float(number) if math.isfinite(number) else None for number in sampled[:, i]]}
                    for i, label in enumerate(coordinate[::coord_step])]
            if rows:
                return "hovmoller", {"hovmollerRows": rows,
                                     "hovmollerTimeLabels": [str(t)[:19] for t in time[::time_step]]}
    if isinstance(value.get("distance_km"), list) and "values" in value:
        values = _preview_values(value["values"], root)
        if np.ndim(values) == 3:
            values = values[0]
        if np.ndim(values) == 2:
            distances = value["distance_km"]
            depth = value.get("depth") or value.get("time") or []
            if np.shape(values) == (len(depth), len(distances)):
                row_step = max(1, math.ceil(len(depth) / PREVIEW_GRID_SIDE))
                col_step = max(1, math.ceil(len(distances) / PREVIEW_GRID_SIDE))
                sample = np.asarray(values[::row_step, ::col_step], dtype=float)
                rows = [{"label": str(label), "coordValue": float(label) if isinstance(label, (int, float)) else None,
                         "values": [float(number) if math.isfinite(number) else None for number in row]}
                        for label, row in zip(depth[::row_step], sample)]
                if rows:
                    return "section", {"sectionRows": rows,
                                       "sectionDistanceKm": [float(d) for d in distances[::col_step]],
                                       "sectionAxisTitle": "Depth (m)" if value.get("depth") else "Time"}
    map_field = _map_payload(value, root, name)
    if map_field:
        return "summary", {"mapField": map_field}
    return None


def _numeric_metrics(path: Any) -> list[dict[str, str]]:
    if path.stat().st_size > 64 * 1024:
        return []
    with path.open(encoding="utf-8") as file:
        value = json.load(file).get("value")
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        value = {"Value": value}
    if not isinstance(value, dict):
        return []
    return [{"label": str(key).replace("_", " ")[:40], "value": f"{number:.5g}"}
            for key, number in value.items()
            if isinstance(number, (int, float)) and not isinstance(number, bool)
            and math.isfinite(number)][:6]


def _current_unit(value: Any) -> str | None:
    if isinstance(value, dict):
        return ", ".join(f"{key}: {item}" for key, item in value.items())[:120]
    return str(value)[:120] if value is not None else None


def _overlay(event: dict, artifact_id: str, number: int, kind: str,
             when: Any, depth: Any) -> dict | None:
    center = event.get("center") or event.get("centroid")
    if not isinstance(center, dict):
        center = event
    lon, lat = center.get("lon"), center.get("lat")
    path = event.get("path") or event.get("path_coordinates")
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lon, lat))
            and isinstance(path, list) and path):
        midpoint = path[len(path) // 2]
        if isinstance(midpoint, dict):
            lon, lat = midpoint.get("lon"), midpoint.get("lat")
    bbox = event.get("bbox")
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lon, lat))
            and isinstance(bbox, dict)):
        edges = [bbox.get(key) for key in ("lon_min", "lon_max", "lat_min", "lat_max")]
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in edges):
            lon, lat = (edges[0] + edges[1]) / 2, (edges[2] + edges[3]) / 2
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lon, lat)):
        return None
    if not -180 <= lon <= 360 or not -90 <= lat <= 90:
        return None
    details = []
    radius = event.get("radius_km")
    if isinstance(radius, (int, float)) and math.isfinite(radius) and radius > 0:
        details.append(f"Radius: {radius:.1f} km")
    if when is not None:
        details.append(f"Time: {when}")
    if depth is not None:
        details.append(f"Depth: {depth}")
    overlay = {"id": str(event.get("event_id") or event.get("track_id") or f"{artifact_id}_{number}"),
               "eventType": kind, "title": f"{kind.replace('_', ' ').title()} {number}",
               "center": {"lat": lat, "lon": lon}, "details": details,
               "timestamp": str(event.get("timestamp") or when) if event.get("timestamp") or when else None}
    if isinstance(path, list):
        points = [{"lon": point["lon"], "lat": point["lat"]} for point in path
                  if isinstance(point, dict) and all(isinstance(point.get(key), (int, float))
                  and math.isfinite(point[key]) for key in ("lon", "lat"))]
        if len(points) >= 2:
            overlay.update(shape="polyline", path=points)
    if "shape" not in overlay and isinstance(radius, (int, float)) and math.isfinite(radius) and radius > 0:
        overlay.update(shape="circle", radiusKm=float(radius))
    if "shape" not in overlay:
        overlay.update(shape="point", symbol="diamond" if kind in {"front", "meander"} else "triangle")
    if isinstance(event.get("severity"), str):
        overlay["severity"] = event["severity"]
    return overlay


def _selection_overlay(geometry: dict, artifact_id: str) -> dict | None:
    kind, vertices = geometry.get("type"), geometry.get("points")
    if kind not in {"transect", "polygon"} or not isinstance(vertices, list):
        return None
    points = []
    for vertex in vertices:
        if (not isinstance(vertex, list) or len(vertex) != 2
                or not all(isinstance(value, (int, float)) and math.isfinite(value)
                           for value in vertex)):
            return None
        points.append({"lon": float(vertex[0]), "lat": float(vertex[1])})
    if len(points) < (2 if kind == "transect" else 3):
        return None
    path = points + [points[0]] if kind == "polygon" and points[-1] != points[0] else points
    return {"id": f"{artifact_id}_{kind}_selection", "eventType": "selection",
            "title": "Analyzed transect" if kind == "transect" else "Analyzed polygon",
            "shape": "polyline", "center": points[0], "path": path,
            "details": [f"{len(points)} calculation vertices"]}


class ProgressAdapter:
    """Maintain bounded UI previews while the full runtime index stays on disk."""

    def __init__(self, session: AnalysisSession,
                 emit: Callable[[dict[str, Any]], None] | None = None,
                 *, interval_seconds: float = 0.25) -> None:
        self.session = session
        self.emit = emit or (lambda envelope: None)
        self.interval_seconds = interval_seconds
        self.cards: OrderedDict[str, dict] = OrderedDict()
        self.result_cards: list[dict] = []
        self.workspace_by_result: dict[str, dict] = {}
        self.counts: dict[str, dict[str, int]] = {}
        self.last_progress: dict[str, float] = {}
        self.active_result_id: str | None = None
        self.attempt_order: dict[str, int] = {}

    def _attempt_index(self, attempt_id: Any) -> int | None:
        if not isinstance(attempt_id, str) or not attempt_id:
            return None
        if attempt_id not in self.attempt_order:
            self.attempt_order[attempt_id] = len(self.attempt_order)
        return self.attempt_order[attempt_id]

    def _send(self, event_type: str, payload: dict[str, Any]) -> None:
        self.emit({"event": "execution_event", "payload": deepcopy({"type": event_type, **payload})})

    def _result_geometry(self, metadata: dict) -> list[dict]:
        """Follow saved input references to the vertices actually used by tools."""
        if not hasattr(self.session, "records"):
            return []
        queue = [metadata]
        seen: set[str] = set()
        overlays = []
        while queue and len(seen) < 128:
            item = queue.pop(0)
            artifact_id = item.get("artifact_id")
            if not isinstance(artifact_id, str) or artifact_id in seen:
                continue
            seen.add(artifact_id)
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                try:
                    call = self.session.records.read("call", call_id)
                    geometry = call.get("parameters", {}).get("analysis_geometry")
                    if isinstance(geometry, dict):
                        overlay = _selection_overlay(geometry, metadata["artifact_id"])
                        if overlay and not any(other["path"] == overlay["path"]
                                               for other in overlays):
                            overlays.append(overlay)
                except (KeyError, OSError, ValueError):
                    pass
            for ref in item.get("inputs", []):
                if not isinstance(ref, str) or ref in seen:
                    continue
                try:
                    source = self.session.artifacts.read_artifact(ref)
                    if source.get("run_id") == self.session.run_id:
                        queue.append(source)
                except (OSError, ValueError):
                    pass
        return overlays

    def _backfill_input_geometry(self, metadata: dict) -> None:
        """A later summary can identify geometry used by an earlier saved map."""
        call_id = metadata.get("call_id")
        if not isinstance(call_id, str) or not hasattr(self.session, "records"):
            return
        try:
            call = self.session.records.read("call", call_id)
        except (KeyError, OSError, ValueError):
            return
        geometry = call.get("parameters", {}).get("analysis_geometry")
        if not isinstance(geometry, dict):
            return
        pending = list(metadata.get("inputs", []))
        seen: set[str] = set()
        while pending and len(seen) < 128:
            ref = pending.pop(0)
            if not isinstance(ref, str) or ref in seen:
                continue
            seen.add(ref)
            try:
                source = self.session.artifacts.read_artifact(ref)
            except (OSError, ValueError):
                continue
            if source.get("run_id") != self.session.run_id:
                continue
            workspace = self.workspace_by_result.get(ref)
            if workspace and (workspace.get("mapField") or workspace.get("sectionRows")):
                overlay = _selection_overlay(geometry, ref)
                if overlay and not any(item.get("id") == overlay["id"]
                                       for item in workspace.get("eventOverlays", [])):
                    workspace["eventOverlays"] = [*workspace.get("eventOverlays", []), overlay]
            pending.extend(source.get("inputs", []))

    def _card(self, stage_id: str, title: str | None = None,
              attempt_id: str | None = None) -> dict:
        if stage_id not in self.cards:
            self.cards[stage_id] = {
                "step_id": stage_id, "human_label": title or "Run analysis",
                "technical_label": "analysis stage", "status": "running",
                "results_hidden_by_default": False, "results": [], "actions": [],
                "is_map_bound": False, "is_expanded": False,
                "attempt_id": attempt_id,
                "attempt_index": self._attempt_index(attempt_id),
            }
            self.counts[stage_id] = {"completed": 0, "failed": 0}
        elif attempt_id and self.cards[stage_id]["attempt_id"] is None:
            self.cards[stage_id]["attempt_id"] = attempt_id
            self.cards[stage_id]["attempt_index"] = self._attempt_index(attempt_id)
        return self.cards[stage_id]

    def _attach(self, stage_id: str, entry: dict, *, visible_step: bool = True) -> None:
        card = self._card(stage_id) if visible_step else None
        status = entry.get("status")
        if card is not None and status in ("completed", "failed"):
            self.counts[stage_id][status] += 1
        artifact_id = entry.get("artifact_id")
        if status != "completed" or not isinstance(artifact_id, str):
            return
        try:
            metadata = self.session.artifacts.read_artifact(artifact_id)
        except (OSError, ValueError):
            return
        if (metadata.get("status") != "completed" or metadata.get("run_id") != self.session.run_id
                or metadata.get("stage_id") != stage_id):
            return
        is_figure = metadata.get("kind") == "image_png"
        when, depth = entry.get("time"), entry.get("depth")
        scope = ", ".join(f"{label}: {value}" for label, value in (("time", when), ("depth", depth))
                          if value is not None)
        name = str(metadata.get("name", "Result"))[:120]
        display_name = name.removeprefix("publish:")
        result = {"id": artifact_id, "title": display_name.replace("_", " ").title(),
                  "type": str(metadata.get("kind", "result")),
                  "headline": scope or display_name,
                  "description": "Saved analysis figure" if is_figure else
                                 "Saved field" if metadata.get("kind") == "dataarray_netcdf" else
                                 "Saved calculation result",
                  "renderer": "summary", "metrics": [], "surface": "inline",
                  "attemptId": metadata.get("attempt_id"),
                  "attemptIndex": self._attempt_index(metadata.get("attempt_id"))}
        if card is not None:
            result["ownerStepId"] = stage_id
        try:
            if metadata.get("kind") == "dataarray_netcdf":
                path = _stored_payload(self.session.root, metadata["payload"])
                if metadata.get("presentation") == "summary":
                    result["metrics"] = _loaded_field_metrics(path, metadata.get("summary") or {})
                    result["headline"] = "Loaded data dimensions and sampled statistics"
                    # A saved 2D georeferenced field is already a map result, even
                    # when the producer also requested a statistical summary.
                    preview = _array_preview(path, display_name, spatial_slice_only=True)
                    if preview and preview[1].get("mapField"):
                        result["renderer"], workspace = preview
                        result["workspaceData"] = workspace
                        self.workspace_by_result[artifact_id] = workspace
                else:
                    preview = _array_preview(path, display_name)
                    if preview:
                        result["renderer"], workspace = preview
                        result["workspaceData"] = workspace
                        self.workspace_by_result[artifact_id] = workspace
            elif is_figure and len(metadata.get("inputs") or []) == 1:
                source_id = metadata["inputs"][0]
                source = self.session.artifacts.read_artifact(source_id)
                if (source.get("status") == "completed" and source.get("run_id") == self.session.run_id
                        and source.get("kind") == "dataarray_netcdf"):
                    preview = _array_preview(
                        _stored_payload(self.session.root, source["payload"]),
                        str(source.get("name") or display_name).removeprefix("publish:"),
                        spatial_slice_only=True,
                    )
                    if preview and preview[1].get("mapField"):
                        result["workspaceData"] = preview[1]
                        self.workspace_by_result[artifact_id] = preview[1]
            elif metadata.get("kind") == "json":
                path = _stored_payload(self.session.root, metadata["payload"])
                result["metrics"] = _numeric_metrics(path)
                if result["metrics"]:
                    first = result["metrics"][0]
                    result["headline"] = f"{first['label']}: {first['value']}"
                preview = _json_preview(self.session.root, path, display_name)
                if preview:
                    result["renderer"], workspace = preview
                    result["workspaceData"] = workspace
                    self.workspace_by_result[artifact_id] = workspace
        except (OSError, KeyError, TypeError, ValueError):
            pass
        if metadata.get("kind") == "json":
            try:
                path = _stored_payload(self.session.root, metadata["payload"])
                if path.stat().st_size <= PREVIEW_JSON_BYTES:
                    with path.open(encoding="utf-8") as file:
                        value = json.load(file).get("value")
                    events = value.get("events") if isinstance(value, dict) else None
                    if isinstance(events, list):
                        event_type = str(value.get("event_type") or
                                         ("eddy" if name == "detect_eddies" else
                                          name.removeprefix("detect_").rstrip("s")) or "event")
                        existing_field = (result.get("workspaceData") or {}).get("mapField")
                        mask_field = (existing_field if existing_field and
                                      str(existing_field.get("variable", "")).endswith("mask")
                                      else _structured_mask_payload(value, self.session.root, event_type))
                        overlays = [overlay for i, item in enumerate(events, 1)
                                    if isinstance(item, dict)
                                    if (overlay := _overlay(item, artifact_id, i, event_type,
                                                            when, depth))]
                        workspace = {"eventOverlays": overlays}
                        if mask_field:
                            workspace["mapField"] = mask_field
                        coordinates = value.get("coordinates")
                        if isinstance(coordinates, dict) and "mapField" not in workspace:
                            for field_name in ("ow_field", "gradient_field", "vorticity_field"):
                                if field_name not in value:
                                    continue
                                map_field = _map_payload({**coordinates, "values": value[field_name]},
                                                         self.session.root, field_name)
                                if map_field:
                                    workspace["mapField"] = map_field
                                    break
                        for input_id in metadata.get("inputs", []):
                            source = self.workspace_by_result.get(input_id, {})
                            if source.get("mapField") and "mapField" not in workspace:
                                workspace["mapField"] = source["mapField"]
                                break
                        result.update(type="eddy_detection" if event_type == "eddy" else "event_detection",
                                      renderer="event", surface="map",
                                      metrics=[{"label": "Detected events", "value": str(len(overlays))}],
                                      workspaceData=workspace,
                                      actions=[{"id": "focus_map", "label": "Show on map"}])
                        self.workspace_by_result[artifact_id] = workspace
                        if card is not None:
                            card["is_map_bound"] = True
            except (OSError, KeyError, TypeError, ValueError):
                pass
        workspace = result.get("workspaceData") or {}
        if workspace.get("mapField") or workspace.get("sectionRows"):
            geometry_overlays = self._result_geometry(metadata)
            if geometry_overlays:
                workspace["eventOverlays"] = [*workspace.get("eventOverlays", []),
                                              *geometry_overlays]
                result["workspaceData"] = workspace
                self.workspace_by_result[artifact_id] = workspace
        if workspace.get("mapField") or workspace.get("eventOverlays"):
            result.update(surface="map", actions=[{"id": "focus_map", "label": "Show on main map"}])
            if card is not None:
                card["is_map_bound"] = True
        if card is not None:
            card["results"].append(result)
        self.result_cards.append(result)
        self._backfill_input_geometry(metadata)
        if workspace.get("mapField") or workspace.get("eventOverlays"):
            self.active_result_id = artifact_id
        if card is not None:
            self._send("step_result_attached", {"step_id": stage_id,
                                                "attempt_id": metadata.get("attempt_id"),
                                                "step_card": {**card, "results": [result]}})
        else:
            self._send("attempt_result_attached", {"attempt_id": metadata.get("attempt_id"),
                                                   "result_card": result})

    def on_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "synthesis_started":
            self._send("synthesis_started", {})
            return
        if kind == "attempt_reconciled":
            attempt_id = event.get("attempt_id")
            for stage in event.get("stages", []):
                if not isinstance(stage, dict):
                    continue
                stage_id = stage.get("stage_id")
                card = self.cards.get(stage_id) if isinstance(stage_id, str) else None
                status = stage.get("status")
                if (card is None or card.get("attempt_id") not in (None, attempt_id)
                        or status not in {"completed", "failed"}
                        or card["status"] == status):
                    continue
                self.on_event({"type": "step_completed" if status == "completed" else "step_failed",
                               "attempt_id": attempt_id, "stage_id": stage_id,
                               "title": stage.get("title"),
                               "completed_units": stage.get("completed_units"),
                               "error": stage.get("error") or event.get("error")})
            return
        stage_id = event.get("stage_id") or event.get("step_id")
        if not isinstance(stage_id, str):
            return
        if kind == "stage_result_indexed":
            if isinstance(event.get("entry"), dict):
                self._attach(stage_id, event["entry"],
                             visible_step=event.get("visible_step") is not False)
            return
        if kind not in {"step_started", "step_progress", "step_completed", "step_failed"}:
            return
        if event.get("visible_step") is False:
            return
        card = self._card(stage_id, event.get("title"), event.get("attempt_id"))
        if event.get("title"):
            card["human_label"] = str(event["title"])[:120]
        if kind == "step_progress":
            now = time.monotonic()
            if now - self.last_progress.get(stage_id, float("-inf")) < self.interval_seconds:
                return
            self.last_progress[stage_id] = now
        count = self.counts[stage_id]
        completed = event.get("completed_units")
        total = event.get("total_units")
        progress: dict[str, Any] = {"phase": "complete" if kind == "step_completed" else
                                    "failed" if kind == "step_failed" else "computing",
                                    "message": f"{count['completed']} results saved"
                                    + (f", {count['failed']} failed" if count["failed"] else "")}
        if isinstance(completed, int):
            progress["completed_units"] = completed
        if isinstance(total, int):
            progress["total_units"] = total
        if isinstance(event.get("unit"), str):
            progress["unit_label"] = event["unit"]
        current = _current_unit(event.get("current_unit"))
        if current:
            progress["current_unit"] = current
        if isinstance(event.get("percent"), (int, float)):
            progress["percent"] = max(0.0, min(1.0, event["percent"] / 100))
        card["progress"] = progress
        if kind in ("step_completed", "step_failed"):
            card["status"] = "completed" if kind == "step_completed" else "failed"
            if kind == "step_failed":
                card["error"] = str(event.get("error") or "Analysis stage failed")[:500]
        self._send(kind, {"step_id": stage_id, "attempt_id": event.get("attempt_id"),
                          "step_card": card.copy(),
                          "error": card.get("error"), "recoverable": False})

    def finalize(self, state: Any = None) -> dict[str, Any]:
        cards = list(self.cards.values())
        active = self.active_result_id
        return {
            "step_cards": cards, "result_cards": list(self.result_cards),
            "workspace_data_by_result": dict(self.workspace_by_result),
            "workspace_data": self.workspace_by_result.get(active, {}) if active else {},
            "active_result_id": active,
            "active_map_step_id": next((card["step_id"] for card in cards
                                        if any(result["id"] == active for result in card["results"])), None),
            "plan_steps": [{"step_id": card["step_id"], "tool": "run_analysis",
                            "human_label": card["human_label"],
                            "technical_label": card["technical_label"],
                            "status": card["status"]} for card in cards],
            "result_summaries": {card["step_id"]: {"completed": self.counts[card["step_id"]]["completed"],
                                                 "failed": self.counts[card["step_id"]]["failed"],
                                                 "preview_count": len(card["results"])} for card in cards},
        }
