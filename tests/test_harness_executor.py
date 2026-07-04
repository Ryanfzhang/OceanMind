import asyncio

import numpy as np
import xarray as xr

from domain.ocean.analysis.statistics import compute_spectrum
from packages.harness.code_agent import CodeAgent
from packages.harness.executor import OceanHarnessExecutor
from packages.harness.planner import OceanHarnessPlanner
from packages.tool_loader.progress import report_tool_progress
from packages.tool_loader.orchestrator import ToolOrchestrator


def fake_load_dataset(variable, lon_range, lat_range, time_range=None, dataset="current", **kwargs):
    report_tool_progress(
        phase="fake_load",
        message=f"Loading {variable}",
        percent=0.25,
        completed_units=0,
        total_units=1,
    )
    times = np.array(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"], dtype="datetime64[ns]")
    depth = np.array([0.0, 5.0])
    lat = np.array([lat_range[0], lat_range[1]])
    lon = np.array([lon_range[0], lon_range[1]])
    base = {
        "oxygen": 50.0,
        "temp": 20.0,
        "salt": 34.0,
        "u": 0.2,
        "v": 0.1,
    }.get(variable, 1.0)
    values = np.full((times.size, depth.size, lat.size, lon.size), base, dtype=float)
    if variable == "oxygen":
        values[:, 1, :, :] = np.array([55.0, 58.0, 65.0, 70.0])[:, None, None]
    data = xr.DataArray(
        values,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": times, "depth": depth, "lat": lat, "lon": lon},
        name=variable,
        attrs={"units": "unit"},
    )
    return data


def fake_generated_code_writer(payload):
    return """
import numpy as np

def run(inputs, params):
    field = inputs["field"]
    design = params.get("analysis_design") if isinstance(params.get("analysis_design"), dict) else {}
    output = design.get("output") if isinstance(design.get("output"), dict) else {}
    output_type = output.get("output_type")
    dims = list(getattr(field, "dims", []))
    if output_type == "spatial_field_result" and {"lat", "lon"}.issubset(set(dims)):
        reduce_dims = [dim for dim in dims if dim not in ("lat", "lon")]
        mean = field.mean(dim=reduce_dims, skipna=True) if reduce_dims else field
        std = field.std(dim=reduce_dims, skipna=True) if reduce_dims else field * 0
        values = (std / np.maximum(np.abs(mean), 1e-12) * 100.0).values
        return {
            "output_type": "spatial_field_result",
            "lon": [float(value) for value in field["lon"].values],
            "lat": [float(value) for value in field["lat"].values],
            "values": np.asarray(values, dtype=float).tolist(),
            "metadata": {
                "analysis_type": "custom_variability_index_map",
                "analysis_design": design,
                "generated_code_summary": "Computed temporal coefficient of variation while preserving lat/lon.",
                "unit": output.get("unit") or "percent",
            },
        }
    if hasattr(field, "mean") and "time" in dims:
        reduce_dims = [dim for dim in dims if dim != "time"]
        series = field.mean(dim=reduce_dims, skipna=True) if reduce_dims else field
        values = np.asarray(series.values, dtype=float)
        times = [str(value) for value in series["time"].values]
        return {
            "output_type": "timeseries_result",
            "times": times,
            "values": values.tolist(),
            "metadata": {
                "analysis_type": "regional_mean_timeseries",
                "analysis_design": design,
                "generated_code_summary": "Computed a regional mean time series from fake CodeAgent.",
            },
        }
    return {"output_type": "generic_result", "value": None, "metadata": {"analysis_design": design}}
"""


def fake_generated_design_writer(payload):
    return {
        "data": {
            "variable": "oxygen",
            "input_ref": payload["input_refs"]["field"],
            "vertical_mode": "surface",
        },
        "analysis": {
            "formula": "temporal coefficient of variation per grid cell: std(time) / abs(mean(time)) * 100",
            "reduction_axes": ["time", "depth"],
            "preserve_axes": ["lat", "lon"],
            "description": "Compute a custom oxygen variability index map while preserving spatial axes.",
        },
        "output": {
            "output_type": "spatial_field_result",
            "frontend": "spatial_field",
            "unit": "percent",
            "title": "Oxygen variability index map",
        },
        "assumptions": ["Surface/depth dimension is reduced for the synthetic test field."],
    }


def fake_wrong_timeseries_code_writer(payload):
    return """
def run(inputs, params):
    return {
        "output_type": "timeseries_result",
        "times": ["2020-01-01"],
        "values": [1.0],
        "metadata": {"analysis_type": "wrong_timeseries"},
    }
"""


BAD_SPATIAL_CODE = """
def run(inputs, params):
    return {
        "output_type": "timeseries_result",
        "times": ["2020-01-01"],
        "values": [1.0],
        "metadata": {"analysis_type": "wrong_timeseries"},
    }
"""


GOOD_SPATIAL_CODE = """
import numpy as np

def run(inputs, params):
    field = inputs["field"]
    reduce_dims = [dim for dim in field.dims if dim not in ("lat", "lon")]
    spatial = field.mean(dim=reduce_dims, skipna=True) if reduce_dims else field
    return {
        "output_type": "spatial_field_result",
        "lon": [float(value) for value in field["lon"].values],
        "lat": [float(value) for value in field["lat"].values],
        "values": np.asarray(spatial.values, dtype=float).tolist(),
        "metadata": {"analysis_type": "recovered_spatial_field", "unit": field.attrs.get("units", "")},
}
"""


def fake_deep_velocity_dataset(variable, lon_range, lat_range, time_range=None, dataset="current", **kwargs):
    times = np.array(["2011-01-01", "2011-02-01", "2011-03-01"], dtype="datetime64[ns]")
    depth = np.array([0.0, 100.0, 500.0])
    lat = np.array([lat_range[0], (lat_range[0] + lat_range[1]) / 2.0, lat_range[1]], dtype=float)
    lon = np.array([lon_range[0], (lon_range[0] + lon_range[1]) / 2.0, lon_range[1]], dtype=float)
    base = 0.2 if variable == "u" else 0.1
    values = np.full((times.size, depth.size, lat.size, lon.size), base, dtype=float)
    values += np.arange(lon.size, dtype=float)[None, None, None, :] * 0.01
    values += np.arange(lat.size, dtype=float)[None, None, :, None] * 0.005
    data = xr.DataArray(
        values,
        dims=("time", "depth", "lat", "lon"),
        coords={"time": times, "depth": depth, "lat": lat, "lon": lon},
        name=variable,
        attrs={"units": "m s-1"},
    )
    return data


def fake_build_polygon_mask_for_hovmoller(data, polygon_points, invert=False):
    values = np.ones((data.sizes["lat"], data.sizes["lon"]), dtype=bool)
    if invert:
        values = ~values
    return xr.DataArray(values, coords={"lat": data.lat, "lon": data.lon}, dims=("lat", "lon"), name="polygon_mask")


def fake_build_isobath_mask_for_hovmoller(data, isobath_depth, comparison="deeper_or_equal", **kwargs):
    values = np.ones((data.sizes["lat"], data.sizes["lon"]), dtype=bool)
    return xr.DataArray(values, coords={"lat": data.lat, "lon": data.lon}, dims=("lat", "lon"), name="isobath_mask")


def fake_combine_masks_for_hovmoller(masks, operation="and", invert=False):
    combined = masks[0].astype(bool)
    for mask in masks[1:]:
        combined = combined & mask.astype(bool)
    if invert:
        combined = ~combined
    combined.name = "analysis_mask"
    return combined


def fake_compute_derived_vorticity_field(u, v, field_type="vorticity", **kwargs):
    field = (v - u).rename("relative_vorticity")
    field.attrs["units"] = "s-1"
    field.attrs["field_type"] = field_type
    return field


def fake_apply_mask_for_hovmoller(data, mask, fill_value=np.nan):
    return data.where(mask, fill_value)


class FlakyHovmollerTool:
    def __init__(self):
        self.calls = 0

    def __call__(
        self,
        data,
        diagram_type="time_depth",
        fixed_lon_range=None,
        fixed_lat_range=None,
        aggregate_dim="mean",
        spatial_weighting="area_weighted",
        depth_range=None,
        **kwargs,
    ):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated first Hovmoller time-depth batch materialization failure")
        reduced = data.mean(dim=[dim for dim in data.dims if dim not in ("time", "depth")], skipna=True)
        return {
            "time": [str(value) for value in reduced.time.values],
            "spatial_coord": [float(value) for value in reduced.depth.values],
            "values": np.asarray(reduced.values, dtype=float).tolist(),
            "metadata": {
                "diagram_type": diagram_type,
                "spatial_dim": "depth",
                "aggregate_dim": aggregate_dim,
                "spatial_weighting": spatial_weighting,
                "variable": "relative_vorticity",
                "units": "s-1",
            },
        }


def generated_code_recovery_plan(code, *, code_step_id="generated_analysis"):
    analysis_design = {
        "data": {"variable": "oxygen", "input_ref": "field"},
        "analysis": {"description": "Compute a spatial field from the loaded test data."},
        "output": {
            "output_type": "spatial_field_result",
            "frontend": "spatial_field",
            "unit": "unit",
            "title": "Recovered spatial field",
        },
    }
    return {
        "status": "ready",
        "skill_id": "ocean_harness",
        "skills_used": ["ocean_harness"],
        "steps": [
            {
                "step_id": "load_field",
                "tool": "load_dataset",
                "params": {
                    "variable": "oxygen",
                    "lon_range": [110, 120],
                    "lat_range": [18, 23],
                    "time_range": ["2020-01-01", "2020-04-01"],
                },
                "save_as": "field",
                "human_label": "Load data",
                "technical_label": "Loading oxygen data",
            },
            {
                "step_id": code_step_id,
                "tool": "generated_python_analysis",
                "params": {
                    "input_refs": {"field": "$ref:field.data"},
                    "code": code,
                    "analysis_design": analysis_design,
                    "io_contract": CodeAgent().build_contract(analysis_design),
                    "code_params": {"user_request": "Compute a custom oxygen spatial field."},
                },
                "save_as": "generated_analysis",
                "human_label": "Generated analysis",
                "technical_label": "Run generated spatial analysis",
                "harness_node": {
                    "execution": {
                        "strategy": "generated_code",
                        "output_type": "spatial_field_result",
                    }
                },
            },
        ],
    }


class RecoveringGeneratedCodePlanner(OceanHarnessPlanner):
    def __init__(self):
        super().__init__()
        self.replan_calls = 0

    def generate_plan_for_query(self, **kwargs):
        return generated_code_recovery_plan(BAD_SPATIAL_CODE)

    def replan_after_code_error(self, **kwargs):
        self.replan_calls += 1
        return generated_code_recovery_plan(GOOD_SPATIAL_CODE, code_step_id="generated_analysis_retry")


def hovmoller_runtime_recovery_plan():
    shared_params = {
        "lon_range": [110.0, 121.5],
        "lat_range": [5.0, 23.5],
        "time_range": ["2011-01-01", "2022-12-31"],
    }
    return {
        "status": "ready",
        "skill_id": "ocean_derived_hovmoller_analysis",
        "skills_used": ["ocean_derived_hovmoller_analysis"],
        "steps": [
            {
                "step_id": "u_field",
                "tool": "load_dataset",
                "params": {"variable": "u", **shared_params},
                "save_as": "u_field",
                "human_label": "Load eastward velocity",
            },
            {
                "step_id": "v_field",
                "tool": "load_dataset",
                "params": {"variable": "v", **shared_params},
                "save_as": "v_field",
                "human_label": "Load northward velocity",
            },
            {
                "step_id": "polygon_mask",
                "tool": "build_polygon_mask",
                "params": {
                    "data": "$ref:u_field.data",
                    "polygon_points": [[110.0, 5.0], [121.5, 5.0], [121.5, 23.5], [110.0, 23.5]],
                },
                "save_as": "polygon_mask",
                "human_label": "Build polygon mask",
            },
            {
                "step_id": "isobath_mask",
                "tool": "build_isobath_mask",
                "params": {
                    "data": "$ref:u_field.data",
                    "isobath_depth": 100,
                    "comparison": "deeper_or_equal",
                },
                "save_as": "isobath_mask",
                "human_label": "Build isobath mask",
            },
            {
                "step_id": "analysis_mask",
                "tool": "combine_masks",
                "params": {"masks": ["$ref:polygon_mask.data", "$ref:isobath_mask.data"], "operation": "and"},
                "save_as": "analysis_mask",
                "human_label": "Combine masks",
            },
            {
                "step_id": "derived_field",
                "tool": "compute_derived_field",
                "params": {"u": "$ref:u_field.data", "v": "$ref:v_field.data", "field_type": "vorticity"},
                "save_as": "derived_field",
                "human_label": "Compute relative vorticity",
            },
            {
                "step_id": "masked_derived_field",
                "tool": "apply_mask",
                "params": {"data": "$ref:derived_field.data", "mask": "$ref:analysis_mask.data"},
                "save_as": "masked_derived_field",
                "human_label": "Apply analysis mask",
            },
            {
                "step_id": "hovmoller_result",
                "tool": "compute_hovmoller",
                "params": {
                    "data": "$ref:masked_derived_field.data",
                    "diagram_type": "time_depth",
                    "fixed_lon_range": shared_params["lon_range"],
                    "fixed_lat_range": shared_params["lat_range"],
                    "aggregate_dim": "mean",
                    "spatial_weighting": "area_weighted",
                    "depth_range": None,
                },
                "save_as": "hovmoller_result",
                "human_label": "Build Hovmoller diagram",
            },
        ],
        "task_graph": {"final_artifacts": ["hovmoller_result"]},
    }


class RecoveringRuntimeHovmollerPlanner(OceanHarnessPlanner):
    def __init__(self):
        super().__init__()
        self.replan_calls = 0
        self.failed_events = []

    def generate_plan_for_query(self, **kwargs):
        return hovmoller_runtime_recovery_plan()

    def replan_after_step_error(self, **kwargs):
        self.replan_calls += 1
        self.failed_events.append(kwargs["failed_event"])
        return hovmoller_runtime_recovery_plan()


def fake_select_vertical(data, **params):
    return data.isel(depth=1, drop=True)


def fake_build_threshold_mask(data, threshold, comparison="lt", mask_name="threshold_mask"):
    mask = data < threshold
    mask.name = mask_name
    return mask


def fake_compute_masked_area_fraction_timeseries(event_mask, analysis_mask=None, min_valid_fraction=0.0):
    series = event_mask.mean(dim=[dim for dim in event_mask.dims if dim != "time"])
    return {"times": [str(t) for t in series.time.values], "values": series.values.tolist(), "metadata": {"unit": "fraction"}}


def fake_compute_masked_mean_timeseries(data, analysis_mask=None, event_mask=None):
    series = data.mean(dim=[dim for dim in data.dims if dim != "time"])
    return {"times": [str(t) for t in series.time.values], "values": series.values.tolist(), "metadata": {"unit": data.attrs.get("units", "")}}


def fake_compute_speed_from_uv(u, v):
    speed = np.sqrt(u ** 2 + v ** 2)
    speed.name = "speed"
    return speed


def fake_compute_trend(timeseries, method="linear", confidence_level=0.95):
    return {
        "method": method,
        "slope": 1.0,
        "p_value": 0.01,
        "trend_line": list(timeseries["values"]),
        "times": list(timeseries["times"]),
        "values": list(timeseries["values"]),
    }


def fake_compute_spectrum(timeseries, method="welch", detrend="linear", window="hann"):
    return {"frequency": [0.0, 1.0], "period": [None, 1.0], "power": [0.0, 2.0], "metadata": {}}


def fake_compute_lag_correlation(timeseries1, timeseries2, max_lag=12, confidence_level=0.95):
    return {"lags": [0], "correlations": [0.5], "p_values": [0.1], "optimal_lag": 0}


def test_trend_summary_preserves_requested_scope_metadata():
    summary = ToolOrchestrator().summarize_result(
        {
            "output_type": "trend_result",
            "method": "linear",
            "slope": 0.5,
            "p_value": 0.01,
            "trend_line": [1.0, 2.0],
            "metadata": {
                "source_variable": "temp",
                "region": {"lon_range": [110.0, 120.0], "lat_range": [18.0, 23.0]},
                "time_range": ["2011-01-01", "2012-12-31"],
            },
        }
    )

    assert summary["variable"] == "temp"
    assert summary["region"] == {"lon_range": [110.0, 120.0], "lat_range": [18.0, 23.0]}
    assert summary["time_range"] == ["2011-01-01", "2012-12-31"]
    assert summary["analysis_context"]["region"]["lon_range"] == [110.0, 120.0]


def fake_detect_heatwaves(temp, **params):
    return {
        "event_type": "heatwave",
        "events": [
            {"center": {"lon": float(temp.lon.values[0]), "lat": float(temp.lat.values[0])}, "duration_days": 5}
        ],
        "statistics": {"total_count": 1, "detection_params": params},
    }


def fake_compute_event_summary_map(event_detection, data, summary_mode="burden"):
    return {
        "lon": data.lon.values.tolist(),
        "lat": data.lat.values.tolist(),
        "values": np.ones((data.sizes["lat"], data.sizes["lon"])).tolist(),
        "metadata": {
            "title": f"Heatwave {summary_mode}",
            "event_type": event_detection["event_type"],
            "summary_mode": summary_mode,
            "variable": "temp",
            "units": "days" if summary_mode == "event_days" else "degree_days",
        },
    }


async def _collect_events():
    tools = {
        "load.load_dataset": fake_load_dataset,
        "harness_ops.select_vertical": fake_select_vertical,
        "harness_ops.build_threshold_mask": fake_build_threshold_mask,
        "harness_ops.compute_masked_area_fraction_timeseries": fake_compute_masked_area_fraction_timeseries,
        "harness_ops.compute_masked_mean_timeseries": fake_compute_masked_mean_timeseries,
        "harness_ops.compute_speed_from_uv": fake_compute_speed_from_uv,
        "extract.compute_trend": fake_compute_trend,
        "statistics.compute_spectrum": fake_compute_spectrum,
        "advanced.compute_lag_correlation": fake_compute_lag_correlation,
    }
    executor = OceanHarnessExecutor(tools=tools)
    events = []
    async for event in executor.execute_query(
        "底层缺氧5m以内趋势、时间序列、功率谱，以及与温度盐度流速强度是否有关",
        extracted_params={"lon_range": [110, 120], "lat_range": [18, 23]},
        additional_context={},
    ):
        events.append(event)
    return events, executor


async def _collect_heatwave_events():
    tools = {
        "load.load_dataset": fake_load_dataset,
        "heatwave.detect_heatwaves": fake_detect_heatwaves,
        "statistics.compute_event_summary_map": fake_compute_event_summary_map,
    }
    executor = OceanHarnessExecutor(tools=tools)
    events = []
    async for event in executor.execute_query(
        "Detect marine heatwave events in the region (113 ° E-117 ° E, 19 °N-22 °N) from January to March 2015",
        extracted_params={},
        additional_context={},
    ):
        events.append(event)
    return events, executor


async def _collect_generated_code_events(code_writer=fake_generated_code_writer, *, max_replans=1):
    tools = {
        "load.load_dataset": fake_load_dataset,
    }
    executor = OceanHarnessExecutor(tools=tools)
    planner = OceanHarnessPlanner(
        code_agent=CodeAgent(
            design_writer=fake_generated_design_writer,
            code_writer=code_writer,
        )
    )
    events = []
    async for event in executor.execute_query(
        "Compute a custom oxygen variability index for this region from 2010 to 2012",
        extracted_params={"lon_range": [110, 120], "lat_range": [18, 23]},
        additional_context={},
        planner=planner,
        planner_kwargs={"max_replans": max_replans},
    ):
        events.append(event)
    return events, executor


async def _collect_recovering_generated_code_events():
    executor = OceanHarnessExecutor(tools={"load.load_dataset": fake_load_dataset})
    planner = RecoveringGeneratedCodePlanner()
    events = []
    async for event in executor.execute_query(
        "Compute a custom oxygen spatial field",
        extracted_params={"lon_range": [110, 120], "lat_range": [18, 23]},
        additional_context={},
        planner=planner,
        planner_kwargs={"max_replans": 1},
    ):
        events.append(event)
    return events, executor, planner


async def _collect_recovering_hovmoller_tool_events():
    flaky_hovmoller = FlakyHovmollerTool()
    tools = {
        "load.load_dataset": fake_deep_velocity_dataset,
        "filter.build_polygon_mask": fake_build_polygon_mask_for_hovmoller,
        "filter.build_isobath_mask": fake_build_isobath_mask_for_hovmoller,
        "filter.combine_masks": fake_combine_masks_for_hovmoller,
        "compute.compute_derived_field": fake_compute_derived_vorticity_field,
        "filter.apply_mask": fake_apply_mask_for_hovmoller,
        "analysis.compute_hovmoller": flaky_hovmoller,
    }
    executor = OceanHarnessExecutor(tools=tools)
    planner = RecoveringRuntimeHovmollerPlanner()
    events = []
    async for event in executor.execute_query(
        (
            "make a time-depth diagram of area-averaged relative vorticity in the deep South China Sea basin "
            "for 2011-2022. Use the drawn polygon as the analysis mask, and keep only areas deeper than or "
            "equal to 100 m."
        ),
        extracted_params={
            "lon_range": [110.0, 121.5],
            "lat_range": [5.0, 23.5],
            "time_range": ["2011-01-01", "2022-12-31"],
            "mask_polygon": [[110.0, 5.0], [121.5, 5.0], [121.5, 23.5], [110.0, 23.5]],
            "mask_isobath_depth": 100,
            "mask_isobath_comparison": "deeper_or_equal",
        },
        additional_context={},
        planner=planner,
        planner_kwargs={"max_replans": 1},
    ):
        events.append(event)
    return events, executor, planner, flaky_hovmoller


def test_harness_executor_emits_legacy_event_protocol():
    events, executor = asyncio.run(_collect_events())
    event_types = [event["type"] for event in events]
    assert "plan_generated" in event_types
    assert "step_progress" in event_types
    assert "step_complete" in event_types
    assert event_types[-1] == "plan_complete"
    progress_events = [event for event in events if event["type"] == "step_progress"]
    assert any(event["progress"].get("phase") == "fake_load" for event in progress_events)
    summaries = executor.get_result_summaries()
    assert "hypoxia_timeseries" in summaries
    assert "hypoxia_spectrum" in summaries
    assert "speed_lag_correlation" in summaries


def test_harness_executor_runs_manual_heatwave_recipe():
    events, executor = asyncio.run(_collect_heatwave_events())
    completed_tools = [event["tool"] for event in events if event["type"] == "step_complete"]

    assert completed_tools == [
        "load_dataset",
        "detect_heatwaves",
        "compute_event_summary_map",
        "compute_event_summary_map",
    ]
    assert "compute_spatial_field" not in completed_tools
    summaries = executor.get_result_summaries()
    assert "heatwave_detection" in summaries
    assert "marine_heatwave_days" in summaries
    assert "marine_heatwave_burden" in summaries


def test_harness_executor_runs_generated_code_node():
    events, executor = asyncio.run(_collect_generated_code_events())
    completed = [(event["tool"], event["result_id"], event["output_type"]) for event in events if event["type"] == "step_complete"]

    assert completed[-1] == ("generated_python_analysis", "generated_analysis", "spatial_field_result")
    progress_events = [event for event in events if event["type"] == "step_progress"]
    assert any(event["progress"].get("phase") == "design_generated_code" for event in progress_events)
    assert any(event["progress"].get("phase") == "run_generated_code" for event in progress_events)
    summaries = executor.get_result_summaries()
    assert "generated_analysis" in summaries
    assert summaries["generated_analysis"]["type"] == "spatial_field_result"
    result = executor.workspace["generated_analysis"]
    assert result["metadata"]["analysis_type"] == "custom_variability_index_map"
    assert "preserving lat/lon" in result["metadata"]["generated_code_summary"]
    assert np.asarray(result["values"]).shape == (2, 2)


def test_generated_code_contract_rejects_wrong_output_type():
    events, _executor = asyncio.run(
        _collect_generated_code_events(fake_wrong_timeseries_code_writer, max_replans=0)
    )
    step_errors = [event for event in events if event["type"] == "step_error"]

    assert step_errors
    assert "expected spatial_field_result" in step_errors[-1]["error"]


def test_generated_code_recovery_hides_initial_error_until_replan_finishes():
    events, executor, planner = asyncio.run(_collect_recovering_generated_code_events())
    event_types = [event["type"] for event in events]

    assert planner.replan_calls == 1
    assert "step_reflection_started" in event_types
    assert "plan_replanned" in event_types
    assert event_types.index("step_reflection_started") < event_types.index("plan_replanned")
    assert "step_error" not in event_types[: event_types.index("plan_replanned")]

    replanned = [event for event in events if event["type"] == "plan_replanned"]
    assert replanned[-1]["reason"] == "generated_code_error"
    assert event_types[-1] == "plan_complete"
    assert "generated_analysis" in executor.get_result_summaries()


def test_runtime_hovmoller_recovery_hides_initial_tool_error_until_replan_finishes():
    events, executor, planner, flaky_hovmoller = asyncio.run(_collect_recovering_hovmoller_tool_events())
    event_types = [event["type"] for event in events]

    assert flaky_hovmoller.calls == 2
    assert planner.replan_calls == 1
    assert planner.failed_events[0]["step_id"] == "hovmoller_result"
    assert "simulated first Hovmoller" in planner.failed_events[0]["error"]
    assert "step_reflection_started" in event_types
    assert "plan_replanned" in event_types
    assert event_types.index("step_reflection_started") < event_types.index("plan_replanned")
    assert "step_error" not in event_types[: event_types.index("plan_replanned")]

    replanned = [event for event in events if event["type"] == "plan_replanned"]
    assert replanned[-1]["reason"] == "runtime_step_error"
    assert event_types[-1] == "plan_complete"
    assert "hovmoller_result" in executor.get_result_summaries()


def test_timeseries_normalization_formats_nanosecond_time_axis():
    orchestrator = ToolOrchestrator()
    datetime_axis = np.array(["2011-01-01T12:00:00", "2011-12-31T12:00:00"], dtype="datetime64[ns]")

    normalized_from_datetime = orchestrator.normalize_result(
        "generated_code",
        {
            "output_type": "timeseries_result",
            "times": datetime_axis,
            "values": [1.0, 2.0],
        },
    )
    normalized_from_epoch = orchestrator.normalize_result(
        "generated_code",
        {
            "output_type": "timeseries_result",
            "times": datetime_axis.astype("int64").tolist(),
            "values": [1.0, 2.0],
        },
    )

    assert normalized_from_datetime["times"] == ["2011-01-01T12:00:00", "2011-12-31T12:00:00"]
    assert normalized_from_epoch["times"] == ["2011-01-01T12:00:00", "2011-12-31T12:00:00"]


def test_compute_spectrum_returns_validation_result_for_short_series():
    result = compute_spectrum(
        {
            "times": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "values": [0.2, np.nan, 0.4],
            "metadata": {"variable": "hypoxia_fraction"},
        }
    )

    assert result["frequency"] == []
    assert result["power"] == []
    assert result["metadata"]["validation"]["status"] == "insufficient_data"
    assert result["metadata"]["validation"]["valid_time_steps"] == 2


class ExplodingPlanner:
    def generate_plan_for_query(self, **kwargs):
        raise AssertionError("approved proposal execution should not re-plan")


async def _collect_approved_proposal_plan_events():
    approved_plan = {
        "status": "ready",
        "skill_id": "approved_proposal_skill",
        "skills_used": ["approved_proposal_skill"],
        "steps": [
            {
                "step_id": "validated_step",
                "tool": "already_validated_tool",
                "params": {},
                "save_as": "validated_step",
            }
        ],
    }
    executor = OceanHarnessExecutor(tools={})
    events = []
    async for event in executor.execute_query(
        "Run proposed analysis",
        additional_context={"analysis_proposal_context": {"approved_plan": approved_plan}},
        planner=ExplodingPlanner(),
    ):
        events.append(event)
        if event["type"] == "plan_generated":
            break
    return events


def test_harness_executor_uses_approved_proposal_plan_without_replanning():
    events = asyncio.run(_collect_approved_proposal_plan_events())
    plan_events = [event for event in events if event["type"] == "plan_generated"]

    assert plan_events
    assert plan_events[0]["plan"]["skill_id"] == "approved_proposal_skill"
    assert plan_events[0]["plan"]["steps"][0]["tool"] == "already_validated_tool"
