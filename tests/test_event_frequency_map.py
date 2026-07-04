import numpy as np
import xarray as xr

from domain.ocean.events.analysis.statistics import compute_event_frequency_map


def test_event_frequency_map_prefers_event_mask_when_detection_is_available():
    data = xr.DataArray(
        np.ones((2, 3), dtype=float),
        coords={"lat": [18.0, 18.25], "lon": [110.0, 110.25, 110.5]},
        dims=("lat", "lon"),
        name="temp",
    )
    event_mask = np.asarray(
        [
            [True, True, False],
            [False, True, False],
        ],
        dtype=bool,
    )
    event_detection = {
        "event_type": "heatwave",
        "event_mask": event_mask,
        "events": [{"center": {"lon": 110.5, "lat": 18.25}}],
        "statistics": {"total_count": 1, "detection_params": {"vertical_mode": "surface"}},
    }

    result = compute_event_frequency_map(
        event_detection=event_detection,
        data=data,
    )

    values = np.asarray(result["values"], dtype=float)
    assert result["metadata"]["frequency_source"] == "event_mask"
    assert result["metadata"]["title"] == "Event frequency map"
    assert values.shape == (2, 3)
    assert np.nansum(values) == 3.0
    assert values[0, 0] == 1.0
    assert values[1, 1] == 1.0


def test_event_frequency_map_defaults_to_source_grid():
    data = xr.DataArray(
        np.ones((2, 3), dtype=float),
        coords={"lat": [18.0, 18.25], "lon": [110.0, 110.25, 110.5]},
        dims=("lat", "lon"),
    )
    events = [
        {"center": {"lon": 110.1, "lat": 18.1}},
        {"center": {"lon": 110.4, "lat": 18.3}},
    ]

    result = compute_event_frequency_map(
        events=events,
        data=data,
        lon_range=(110.0, 110.5),
        lat_range=(18.0, 18.25),
    )

    values = np.asarray(result["values"], dtype=float)
    assert result["metadata"]["grid_mode"] == "native"
    assert result["metadata"]["resolution_deg"] is None
    assert result["lon"] == [110.0, 110.25, 110.5]
    assert result["lat"] == [18.0, 18.25]
    assert values.shape == (2, 3)
    assert np.nansum(values) == 2.0


def test_event_frequency_map_uses_explicit_coarse_resolution():
    data = xr.DataArray(
        np.ones((2, 3), dtype=float),
        coords={"lat": [18.0, 18.25], "lon": [110.0, 110.25, 110.5]},
        dims=("lat", "lon"),
    )

    result = compute_event_frequency_map(
        events=[{"center": {"lon": 110.1, "lat": 18.1}}],
        data=data,
        lon_range=(110.0, 111.0),
        lat_range=(18.0, 19.0),
        resolution_deg=1.0,
    )

    values = np.asarray(result["values"], dtype=float)
    assert result["metadata"]["grid_mode"] == "binned"
    assert result["metadata"]["resolution_deg"] == 1.0
    assert result["lon"] == [110.5]
    assert result["lat"] == [18.5]
    assert values.shape == (1, 1)
    assert np.nansum(values) == 1.0
