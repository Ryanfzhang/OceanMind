import numpy as np
import pandas as pd
import xarray as xr

from domain.ocean.events.bloom import detect_algal_blooms


def test_detect_algal_blooms_accepts_absolute_threshold():
    data = xr.DataArray(
        np.asarray(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[1.2, 0.5], [0.5, 0.5]],
                [[1.2, 0.5], [0.5, 0.5]],
                [[1.2, 0.5], [0.5, 0.5]],
                [[1.2, 0.5], [0.5, 0.5]],
                [[1.2, 0.5], [0.5, 0.5]],
            ],
            dtype=float,
        ),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2011-01-01", periods=6, freq="D"),
            "lat": [18.0, 18.1],
            "lon": [110.0, 110.1],
        },
        name="chlorophyll",
    )

    result = detect_algal_blooms(
        data,
        threshold=1.0,
        percentile_threshold=None,
        min_duration_days=5,
        min_area_km2=0,
    )

    assert result["statistics"]["detection_params"]["threshold"] == 1.0
    assert np.asarray(result["threshold_field"]).shape == (2, 2)
    assert np.allclose(np.asarray(result["threshold_field"]), 1.0)
    assert np.count_nonzero(np.asarray(result["event_mask"])) == 5


def test_detect_algal_blooms_defaults_to_absolute_threshold():
    data = xr.DataArray(
        np.full((6, 2, 2), 1.2, dtype=float),
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2011-01-01", periods=6, freq="D"),
            "lat": [18.0, 18.1],
            "lon": [110.0, 110.1],
        },
        name="chlorophyll",
    )

    result = detect_algal_blooms(data, min_duration_days=1, min_area_km2=0)

    assert result["statistics"]["detection_params"]["threshold"] == 1.0
    assert result["statistics"]["detection_params"]["percentile_threshold"] is None
