"""Eddy tools must use the time and depth selected by their caller."""

import numpy as np
import pytest
import xarray as xr

from domain.ocean.events.eddy.detect import detect_eddies
from domain.ocean.events.eddy.track import track_eddies


def velocity_fields():
    coords = {
        "time": [0, 1],
        "depth": [0.0, 50.0],
        "lat": np.linspace(18, 20, 5),
        "lon": np.linspace(110, 112, 5),
    }
    data = np.zeros((2, 2, 5, 5), dtype=float)
    u = xr.DataArray(data, dims=tuple(coords), coords=coords)
    v = xr.DataArray(data.copy(), dims=tuple(coords), coords=coords)
    return u, v


def test_detect_requires_explicit_single_time_and_depth():
    u, v = velocity_fields()
    with pytest.raises(ValueError, match="select one time and depth explicitly"):
        detect_eddies(u, v)
    with pytest.raises(ValueError, match="select one time and depth explicitly"):
        detect_eddies(u.isel(time=0), v.isel(time=0))

    for depth in (0.0, 50.0):
        result = detect_eddies(u.sel(time=0, depth=depth), v.sel(time=0, depth=depth))
        assert result["statistics"]["total_count"] == 0
        assert result["mask"].shape == (5, 5)


def test_track_requires_explicit_depth_but_keeps_time_series():
    u, v = velocity_fields()
    with pytest.raises(ValueError, match="select one depth explicitly"):
        track_eddies(u, v)
    result = track_eddies(u.sel(depth=50.0), v.sel(depth=50.0))
    assert result["statistics"]["total_count"] == 0
