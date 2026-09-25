"""The retained manual endpoints work with the new, small API module."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api import langgraph_visualize


@pytest.fixture
def client(tmp_path, monkeypatch):
    time = np.array(["2022-01-01", "2022-01-02", "2022-01-03"], dtype="datetime64[ns]")
    depth = np.array([0.0, -10.0, -20.0, -30.0])
    lat = np.array([20.0, 21.0, 22.0])
    lon = np.array([120.0, 121.0, 122.0])
    shape = (len(time), len(depth), len(lat), len(lon))
    base = np.broadcast_to(
        2 + np.arange(len(time))[:, None, None, None]
        + np.arange(len(depth))[None, :, None, None] * 0.5
        + np.arange(len(lat))[None, None, :, None] * 0.2
        + np.arange(len(lon))[None, None, None, :] * 0.1,
        shape,
    )
    for name, values in {
        "temp": 25 - base - np.array([0, 1, 5, 6])[None, :, None, None],
        "salt": 35 + base * 0.05,
        "chlorophyll": base,
    }.items():
        xr.Dataset({name: (("time", "depth", "lat", "lon"), values)},
                   coords={"time": time, "depth": depth, "lat": lat, "lon": lon}).to_netcdf(
            tmp_path / f"CMOMS_{name}_Zlev_2022.nc", engine="scipy"
        )
    original_load = langgraph_visualize.load_dataset
    monkeypatch.setattr(
        langgraph_visualize, "load_dataset",
        lambda **kwargs: original_load(**kwargs, data_path=str(tmp_path)),
    )
    app = FastAPI()
    app.include_router(langgraph_visualize.router)
    return TestClient(app)


@pytest.mark.parametrize(
    ("mode", "extras", "expected_depth"),
    [
        ("fixed", {"depth_range": [0, 0]}, "0 m"),
        ("feature", {"feature": "thermocline"}, "thermocline"),
        ("feature", {"feature": "mixed_layer"}, "mixed layer"),
        ("feature", {"feature": "pycnocline"}, "pycnocline"),
        ("layer_mean", {"layer_mean_label": "surface -> thermocline"}, "surface -> thermocline"),
    ],
)
def test_manual_visualization_modes_return_map(client, mode, extras, expected_depth):
    payload = {
        "dataset": "CMOMS", "variable": "chlorophyll",
        "time_range": ["2022-01-01", "2022-01-03"],
        "region": {"lon_min": 120, "lon_max": 122, "lat_min": 20, "lat_max": 22},
        "depth_mode": mode,
        **extras,
    }
    response = client.post("/visualize", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "completed"
    assert result["active_result_id"] == result["result_cards"][0]["id"]
    field = result["workspace_data"]["mapField"]
    assert field["depthLabel"] == expected_depth
    assert field["lon"] == [120, 121, 122]
    assert field["lat"] == [20, 21, 22]
    assert len(field["values"]) == len(field["lat"])
    assert all(len(row) == len(field["lon"]) for row in field["values"])
    assert len(result["workspace_data"]["referenceSeries"]) == 3


def test_invalid_layer_is_rejected(client):
    response = client.post("/visualize", json={
        "dataset": "CMOMS", "variable": "chlorophyll",
        "time_range": ["2022-01-01", "2022-01-03"],
        "region": {"lon_min": 120, "lon_max": 122, "lat_min": 20, "lat_max": 22},
        "depth_mode": "layer_mean", "layer_mean_label": "surface -> nonsense",
    })
    assert response.status_code == 400


def test_point_selection_uses_point_timeseries_and_marks_location(client):
    response = client.post("/visualize", json={
        "dataset": "CMOMS", "variable": "chlorophyll",
        "time_range": ["2022-01-01", "2022-01-03"],
        "region": {"lon_min": 120, "lon_max": 122, "lat_min": 20, "lat_max": 22},
        "depth_mode": "fixed", "depth_range": [0, 0],
        "selection_mode": "point", "selected_point": {"lon": 120, "lat": 20},
    })
    assert response.status_code == 200, response.text
    result = response.json()
    data = result["workspace_data"]
    assert [item["value"] for item in data["referenceSeries"]] == [2, 3, 4]
    assert data["eventOverlays"][0]["center"] == {"lat": 20, "lon": 120}
    assert "Point" in result["result_cards"][0]["title"]
    assert data["mapField"]["values"][1][1] == pytest.approx(3.3)


def test_polygon_selection_masks_map_and_averages_only_inside(client):
    response = client.post("/visualize", json={
        "dataset": "CMOMS", "variable": "chlorophyll",
        "time_range": ["2022-01-01", "2022-01-03"],
        "region": {"lon_min": 120, "lon_max": 122, "lat_min": 20, "lat_max": 22},
        "depth_mode": "fixed", "depth_range": [0, 0],
        "selection_mode": "polygon",
        "polygon_points": [[120.4, 20.4], [121.6, 20.4], [121.6, 21.6], [120.4, 21.6]],
    })
    assert response.status_code == 200, response.text
    data = response.json()["workspace_data"]
    assert data["mapField"]["values"][0] == [None, None, None]
    assert data["mapField"]["values"][1] == [None, pytest.approx(3.3), None]
    assert [item["value"] for item in data["referenceSeries"]] == pytest.approx([2.3, 3.3, 4.3])
    assert data["eventOverlays"][0]["shape"] == "polyline"
    assert data["eventOverlays"][0]["path"][0] == data["eventOverlays"][0]["path"][-1]


def test_transect_selection_returns_sampled_profile_and_path(client):
    response = client.post("/visualize", json={
        "dataset": "CMOMS", "variable": "chlorophyll",
        "time_range": ["2022-01-01", "2022-01-03"],
        "region": {"lon_min": 120, "lon_max": 122, "lat_min": 20, "lat_max": 22},
        "depth_mode": "fixed", "depth_range": [0, 0],
        "selection_mode": "transect", "transect_points": [[120, 20], [122, 22]],
    })
    assert response.status_code == 200, response.text
    data = response.json()["workspace_data"]
    assert len(data["sectionDistanceKm"]) == len(data["sectionRows"][0]["values"]) == 100
    assert data["sectionRows"][0]["values"][0] == pytest.approx(3)
    assert data["sectionRows"][0]["values"][-1] == pytest.approx(3.6)
    assert data["eventOverlays"][0]["shape"] == "polyline"
    assert len(data["resultSeries"]) == 100


@pytest.mark.parametrize("mode", ["point", "transect", "polygon"])
def test_missing_active_geometry_is_rejected(client, mode):
    response = client.post("/visualize", json={
        "dataset": "CMOMS", "variable": "chlorophyll",
        "time_range": ["2022-01-01", "2022-01-03"],
        "region": {"lon_min": 120, "lon_max": 122, "lat_min": 20, "lat_max": 22},
        "depth_mode": "fixed", "selection_mode": mode,
    })
    assert response.status_code == 400


def test_report_endpoint_returns_notebook_without_analysis_code(client):
    response = client.post("/report/export", json={
        "exported_at": "2022-01-03T00:00:00Z",
        "turns": [{"user_query": "What changed?", "assistant_status": "completed",
                   "assistant_summary": "The field changed."}],
    })
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ipynb+json")
    assert response.headers["content-disposition"].endswith('.ipynb"')
    notebook = response.json()
    assert notebook["nbformat"] == 4
    assert notebook["cells"][1]["cell_type"] == "markdown"
    assert "The field changed." in notebook["cells"][1]["source"]
    assert not any(cell["cell_type"] == "code" for cell in notebook["cells"])
