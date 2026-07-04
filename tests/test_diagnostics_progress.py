import numpy as np
import xarray as xr

from domain.ocean.diagnostics.compute import compute_spatial_vorticity_map
from packages.tool_loader.progress import reset_tool_progress_callback, set_tool_progress_callback


def test_spatial_vorticity_map_reports_progress_phases():
    time = np.array(["2011-01-01", "2011-01-02"], dtype="datetime64[D]")
    depth = np.array([0.0, 5.0])
    lat = np.array([10.0, 11.0, 12.0])
    lon = np.array([110.0, 111.0, 112.0])
    shape = (time.size, depth.size, lat.size, lon.size)

    lon_grid = np.broadcast_to(lon.reshape(1, 1, 1, lon.size), shape)
    lat_grid = np.broadcast_to(lat.reshape(1, 1, lat.size, 1), shape)
    u = xr.DataArray(
        lat_grid,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        name="u",
    )
    v = xr.DataArray(
        lon_grid,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
        name="v",
    )

    events = []
    token = set_tool_progress_callback(events.append)
    try:
        result = compute_spatial_vorticity_map(
            u,
            v,
            time_range=("2011-01-01", "2011-01-02"),
            depth_range=(0, 5),
        )
    finally:
        reset_tool_progress_callback(token)

    phases = [event.get("phase") for event in events]
    assert phases[0] == "prepare_vorticity_inputs"
    assert "align_vorticity_inputs" in phases
    assert "aggregate_vorticity_depth" in phases
    assert "aggregate_vorticity_time" in phases
    assert "build_vorticity_field" in phases
    assert "package_vorticity_map" in phases
    assert events[0]["percent"] < events[-1]["percent"]
    assert result["metadata"]["variable"] == "relative_vorticity"
    assert np.asarray(result["values"]).shape == (lat.size, lon.size)
