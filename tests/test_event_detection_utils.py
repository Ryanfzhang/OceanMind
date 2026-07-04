from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from domain.ocean.events.detection_utils import (
    compute_duration_mask,
    estimate_area_km2,
    estimate_grid_spacing_km,
    extract_bbox_from_indices,
    extract_bbox_from_mask,
    iter_time_slices,
    time_batch_slices,
)


def test_extract_bbox_from_non_empty_mask() -> None:
    mask = np.array([[False, True, False], [False, True, True]])
    lon = np.array([110.0, 111.0, 112.0])
    lat = np.array([20.0, 21.0])

    assert extract_bbox_from_mask(mask, lon, lat) == {
        "lon_min": 111.0,
        "lon_max": 112.0,
        "lat_min": 20.0,
        "lat_max": 21.0,
    }


def test_extract_bbox_from_empty_mask_is_safe() -> None:
    mask = np.zeros((2, 3), dtype=bool)
    lon = np.array([110.0, 111.0, 112.0])
    lat = np.array([20.0, 21.0])

    assert extract_bbox_from_mask(mask, lon, lat) == {
        "lon_min": None,
        "lon_max": None,
        "lat_min": None,
        "lat_max": None,
    }


def test_extract_bbox_from_index_arrays() -> None:
    lon = np.array([110.0, 111.0, 112.0])
    lat = np.array([20.0, 21.0])

    assert extract_bbox_from_indices(np.array([0, 1]), np.array([2, 1]), lon, lat) == {
        "lon_min": 111.0,
        "lon_max": 112.0,
        "lat_min": 20.0,
        "lat_max": 21.0,
    }


def test_estimate_area_km2_handles_false_and_partial_masks() -> None:
    lon = np.array([110.0, 111.0])
    lat = np.array([20.0, 21.0])

    assert estimate_area_km2(np.zeros((2, 2), dtype=bool), lon, lat) == 0.0

    area = estimate_area_km2(np.array([[True, False], [False, True]]), lon, lat)
    expected_cell_area = 6371.0 * np.deg2rad(1.0) * np.cos(np.deg2rad(20.5))
    expected_cell_area *= 6371.0 * np.deg2rad(1.0)
    assert area == pytest.approx(2 * expected_cell_area)


def test_estimate_grid_spacing_with_1d_lon_lat() -> None:
    spacing = estimate_grid_spacing_km(np.array([110.0, 111.0]), np.array([20.0, 21.0]))

    dx = 6371.0 * np.deg2rad(1.0) * np.cos(np.deg2rad(20.5))
    dy = 6371.0 * np.deg2rad(1.0)
    assert spacing == pytest.approx((dx + dy) / 2.0)


def test_compute_duration_mask_marks_full_persistent_run() -> None:
    mask = np.array(
        [
            [[True, False]],
            [[True, True]],
            [[False, True]],
        ]
    )

    expected = np.array(
        [
            [[True, False]],
            [[True, True]],
            [[False, True]],
        ]
    )
    assert np.array_equal(compute_duration_mask(mask, 2, event_label="test"), expected)


def test_time_batch_slices_without_chunks() -> None:
    no_time = xr.DataArray(np.zeros((2, 3)), dims=("lat", "lon"))
    short = xr.DataArray(np.zeros((5, 2, 3)), dims=("time", "lat", "lon"))
    long = xr.DataArray(np.zeros((65, 2, 3)), dims=("time", "lat", "lon"))

    assert time_batch_slices(no_time) == []
    assert time_batch_slices(short) == [slice(0, 5)]
    assert time_batch_slices(long) == [slice(0, 30), slice(30, 60), slice(60, 65)]


def test_iter_time_slices_with_and_without_time_dimension() -> None:
    with_time = xr.DataArray(
        np.zeros((2, 3, 4)),
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": [10, 11, 12], "lon": [100, 101, 102, 103]},
    )
    without_time = xr.DataArray(
        np.zeros((3, 4)),
        dims=("lat", "lon"),
        coords={"lat": [10, 11, 12], "lon": [100, 101, 102, 103]},
    )

    timed = list(iter_time_slices(with_time, with_time))
    untimed = list(iter_time_slices(without_time, without_time))

    assert [item[0] for item in timed] == [0, 1]
    assert all(item[1].dims == ("lat", "lon") for item in timed)
    assert len(untimed) == 1
    assert untimed[0][0] is None
    assert untimed[0][1].identical(without_time)
    assert untimed[0][2].identical(without_time)
