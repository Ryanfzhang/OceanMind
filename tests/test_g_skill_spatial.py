"""Keep the spatial skill example aligned with the real Python tool signatures."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import xarray as xr

from domain.ocean.analysis.spatial.analysis import compute_spatial_field
from domain.ocean.data_access.load import load_dataset
from packages.analysis_runtime.artifacts import ArtifactStore
from packages.analysis_runtime.records import RunRecords
from packages.analysis_runtime.stages import StageManager
from packages.analysis_runtime.tools import AnalysisTools


SKILL = Path(__file__).resolve().parents[1] / "skills/ocean_spatial_field_analysis/SKILL.md"


def _example():
    source = SKILL.read_text()
    block = re.search(r"```python\n(.*?)\n```", source, re.DOTALL)
    assert block is not None
    namespace = {}
    exec(compile(block.group(1), str(SKILL), "exec"), namespace)
    return namespace["run_spatial_map"]


def _runtime(tmp_path):
    field = xr.DataArray(
        np.arange(2 * 2 * 2 * 3, dtype=float).reshape(2, 2, 2, 3),
        dims=("time", "depth", "lat", "lon"),
        coords={
            "time": ["2022-01-01", "2022-01-02"],
            "depth": [0.0, 50.0],
            "lat": [18.0, 19.0],
            "lon": [110.0, 111.0, 112.0],
        },
        name="chlorophyll",
        attrs={"units": "mg m-3"},
    )
    calls = []

    def loaded(**kwargs):
        inspect.signature(load_dataset).bind(**kwargs)
        calls.append(("load_dataset", kwargs))
        assert kwargs["variable"] == "chlorophyll"
        mode = kwargs["vertical_mode"]
        if mode == "surface":
            return field.sel(depth=0.0)
        if mode == "bottom":
            return field.sel(depth=50.0)
        if mode == "fixed_depth":
            return field.sel(depth=kwargs["depth_value"])
        if mode == "depth_range":
            return field.sel(depth=slice(*kwargs["depth_range"]))
        raise AssertionError(f"unexpected vertical mode: {mode}")

    def mapped(**kwargs):
        inspect.signature(compute_spatial_field).bind(**kwargs)
        calls.append(("compute_spatial_field", kwargs))
        return compute_spatial_field(**kwargs)

    records = RunRecords(tmp_path)
    stages = StageManager(records.run_id, records.new_attempt(), records=records)
    tools = AnalysisTools(
        records, ArtifactStore(tmp_path), stages,
        functions={"load_dataset": loaded, "compute_spatial_field": mapped},
    )
    return tools, stages, calls


def test_surface_example_runs_and_saves_both_results(tmp_path):
    tools, stages, calls = _runtime(tmp_path)
    with stages.activate():
        result = _example()(
            tools, variable="chlorophyll", lon_range=(110, 112),
            lat_range=(18, 19), time_range=("2022-01-01", "2022-01-02"),
            vertical_mode="surface",
        )

    assert [name for name, _ in calls] == ["load_dataset", "compute_spatial_field"]
    assert calls[1][1]["depth_range"] is None
    assert result["values"].shape == (2, 3)
    assert result["metadata"]["units"] == "mg m-3"
    assert [item.title for item in stages.stages] == ["读取原始变量", "计算二维空间场"]
    assert all(item.results[0]["artifact_id"] for item in stages.stages)


def test_layer_mean_and_bottom_keep_distinct_vertical_meaning(tmp_path):
    run = _example()
    layer_tools, layer_stages, layer_calls = _runtime(tmp_path / "layer")
    with layer_stages.activate():
        layer = run(
            layer_tools, variable="chlorophyll", lon_range=(110, 112),
            lat_range=(18, 19), time_range=("2022-01-01", "2022-01-02"),
            vertical_mode="depth_range", depth_range=(0.0, 50.0),
            depth_aggregation="mean",
        )
    bottom_tools, bottom_stages, bottom_calls = _runtime(tmp_path / "bottom")
    with bottom_stages.activate():
        bottom = run(
            bottom_tools, variable="chlorophyll", lon_range=(110, 112),
            lat_range=(18, 19), time_range=("2022-01-01", "2022-01-02"),
            vertical_mode="bottom",
        )

    assert layer_calls[1][1]["depth_range"] == (0.0, 50.0)
    assert bottom_calls[1][1]["depth_range"] is None
    assert layer["values"].shape == bottom["values"].shape == (2, 3)
    assert not np.array_equal(layer["values"], bottom["values"])


def test_skill_preserves_scientific_options_without_old_dsl():
    source = SKILL.read_text()
    for term in ("inspect_data", "find_tools", "bottom", "fixed_depth",
                 "depth_range", "integral", "polygon", "isobath",
                 "mask=...", "feature depth"):
        assert term in source
    assert "$ref" not in source
    assert "workflow_code" not in source
    assert "field.data" not in source
