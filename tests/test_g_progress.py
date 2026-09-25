from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import xarray as xr

from apps.api.langgraph_progress import ProgressAdapter
from packages.agent_loop.analysis import AnalysisSession
from packages.analysis_runtime.stages import StageManager
from packages.analysis_runtime.tools import AnalysisTools


def test_many_runtime_updates_remain_one_stage_card_and_keep_final_count():
    envelopes = []
    session = SimpleNamespace(root=None, run_id="run_test", artifacts=None)
    adapter = ProgressAdapter(session, envelopes.append, interval_seconds=60)
    adapter.on_event({"type": "step_started", "stage_id": "stage_1", "title": "涡旋检测",
                      "completed_units": 0, "total_units": 100, "unit": "切片"})
    for count in range(1, 101):
        adapter.on_event({"type": "step_progress", "stage_id": "stage_1",
                          "completed_units": count, "total_units": 100,
                          "percent": count, "current_unit": {"depth": count}})
    adapter.on_event({"type": "step_completed", "stage_id": "stage_1",
                      "completed_units": 100, "total_units": 100, "percent": 100})
    payload = adapter.finalize()
    assert len(payload["step_cards"]) == 1
    assert payload["step_cards"][0]["progress"]["completed_units"] == 100
    assert payload["step_cards"][0]["progress"]["percent"] == 1
    assert [event["payload"]["type"] for event in envelopes].count("step_progress") == 1
    assert envelopes[-1]["payload"]["step_card"]["status"] == "completed"
    assert envelopes[0]["payload"]["step_card"]["status"] == "running"


class _Artifacts:
    def read_artifact(self, artifact_id: str) -> dict:
        return {"status": "completed", "run_id": "run_test", "stage_id": "stage_1",
                "artifact_id": artifact_id, "name": "computed_field", "kind": "json"}


def test_every_result_is_visible_without_resending_the_whole_stage():
    envelopes = []
    session = SimpleNamespace(root=None, run_id="run_test", artifacts=_Artifacts())
    adapter = ProgressAdapter(session, envelopes.append)
    adapter.on_event({"type": "step_started", "stage_id": "stage_1", "title": "Compute"})
    for index in range(100):
        adapter.on_event({"type": "stage_result_indexed", "stage_id": "stage_1",
                          "entry": {"status": "completed", "artifact_id": f"artifact_{index}",
                                    "summary": "done", "time": str(index)}})
    payload = adapter.finalize()
    assert payload["result_summaries"]["stage_1"] == {
        "completed": 100, "failed": 0, "preview_count": 100,
    }
    assert len(payload["step_cards"][0]["results"]) == 100
    assert len(payload["result_cards"]) == 100
    attached = [event for event in envelopes if event["payload"]["type"] == "step_result_attached"]
    assert len(attached) == 100
    assert all(len(event["payload"]["step_card"]["results"]) == 1 for event in attached)


def test_loaded_field_shows_stats_and_computed_field_shows_map(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    stage = session.records.new_stage(attempt, "Mean chlorophyll")
    adapter = ProgressAdapter(session)
    field = xr.DataArray(np.array([[0.5, 0.8], [1.0, 1.3]]), dims=("lat", "lon"),
                         coords={"lat": [20.0, 20.1], "lon": [120.0, 120.1]},
                         name="chlorophyll", attrs={"units": "mg m-3"})
    field_id = session.artifacts.publish("load_dataset", field, run_id=session.run_id,
                                         attempt_id=attempt, stage_id=stage, inputs=[],
                                         presentation="summary")
    mean_id = session.artifacts.publish("mean", {"mean": 0.9}, run_id=session.run_id,
                                        attempt_id=attempt, stage_id=stage, inputs=[field_id])
    map_id = session.artifacts.publish("compute_spatial_field", {
        "lon": [120.0, 120.1], "lat": [20.0, 20.1],
        "values": [[0.5, 0.8], [1.0, 1.3]],
        "metadata": {"variable": "chlorophyll", "units": "mg m-3"},
    }, run_id=session.run_id, attempt_id=attempt, stage_id=stage, inputs=[field_id])
    for artifact_id in (field_id, mean_id, map_id):
        adapter.on_event({"type": "stage_result_indexed", "stage_id": stage,
                          "entry": {"status": "completed", "artifact_id": artifact_id}})
    field_card, mean_card, map_card = adapter.finalize()["result_cards"]
    assert "workspaceData" not in field_card
    assert {item["label"]: item["value"] for item in field_card["metrics"]} == {
        "Dimensions": "lat × lon", "Shape": "2 × 2", "Units": "mg m-3",
        "Sample valid": "4/4", "Sample min": "0.5", "Sample mean": "0.9",
        "Sample max": "1.3",
    }
    assert mean_card["metrics"] == [{"label": "mean", "value": "0.9"}]
    assert "workspaceData" not in mean_card
    assert map_card["workspaceData"]["mapField"]["variable"] == "chlorophyll"
    assert map_card["surface"] == "map"
    assert adapter.finalize()["active_result_id"] == map_id


def test_publish_outside_stage_keeps_result_without_visible_step(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    stage = session.records.new_stage(attempt, "Hidden default")
    adapter = ProgressAdapter(session)
    artifact = session.artifacts.publish("publish:computed", {
        "lon": [120.0, 121.0], "lat": [20.0, 21.0],
        "values": [[1.0, 2.0], [3.0, 4.0]],
    }, run_id=session.run_id, attempt_id=attempt, stage_id=stage, inputs=[])
    adapter.on_event({"type": "stage_result_indexed", "stage_id": stage,
                      "attempt_id": attempt, "visible_step": False,
                      "entry": {"status": "completed", "artifact_id": artifact}})
    adapter.on_event({"type": "step_progress", "stage_id": stage,
                      "attempt_id": attempt, "visible_step": False,
                      "completed_units": 1})
    payload = adapter.finalize()
    assert payload["step_cards"] == []
    assert payload["result_cards"][0]["attemptId"] == attempt
    assert payload["result_cards"][0]["attemptIndex"] == 0
    assert payload["result_cards"][0]["title"] == "Computed"
    assert payload["result_cards"][0]["surface"] == "map"


def test_saved_tool_maps_and_timeseries_use_interactive_renderers(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    stage = session.records.new_stage(attempt, "Visual results")
    adapter = ProgressAdapter(session)
    artifacts = [
        session.artifacts.publish("metadata", {"status": "ready"}, run_id=session.run_id,
                                  attempt_id=attempt, stage_id=stage, inputs=[]),
        session.artifacts.publish("event_days_map", {
            "lon": [120.0, 121.0], "lat": [20.0, 21.0],
            "values": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "metadata": {"variable": "event_days", "units": "days"},
        }, run_id=session.run_id, attempt_id=attempt, stage_id=stage, inputs=[]),
        session.artifacts.publish("extract_timeseries", {
            "times": ["2022-01-01", "2022-01-02"], "values": [1.0, 2.0],
        }, run_id=session.run_id, attempt_id=attempt, stage_id=stage, inputs=[]),
    ]
    for artifact_id in artifacts:
        adapter.on_event({"type": "stage_result_indexed", "stage_id": stage,
                          "entry": {"status": "completed", "artifact_id": artifact_id}})
    result = adapter.finalize()
    assert result["result_summaries"][stage]["completed"] == 3
    assert [card["id"] for card in result["result_cards"]] == artifacts
    map_card, series_card = result["result_cards"][1:]
    assert map_card["surface"] == "map"
    assert map_card["workspaceData"]["mapField"]["values"] == [[1.0, 2.0], [3.0, 4.0]]
    assert result["active_result_id"] == artifacts[1]
    assert series_card["renderer"] == "timeseries"
    assert series_card["workspaceData"]["resultSeries"][1]["value"] == 2.0


def test_eof_modes_keep_interactive_map_and_pc_series(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    stage = session.records.new_stage(attempt, "EOF")
    pattern = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"),
                           coords={"lat": [20, 21], "lon": [120, 121]})
    pc = xr.DataArray([0.5, -0.5], dims=("time",),
                      coords={"time": np.array(["2022-01-01", "2022-01-02"],
                                               dtype="datetime64[ns]")})
    artifact_id = session.artifacts.publish("eof", {"modes": [{
        "mode_number": 1, "variance_explained": 70.0,
        "spatial_pattern": pattern, "time_series": pc,
    }]}, run_id=session.run_id, attempt_id=attempt, stage_id=stage, inputs=[])
    adapter = ProgressAdapter(session)
    adapter.on_event({"type": "stage_result_indexed", "stage_id": stage,
                      "entry": {"status": "completed", "artifact_id": artifact_id}})
    card = adapter.finalize()["result_cards"][0]
    assert card["renderer"] == "eof"
    assert card["workspaceData"]["eofModes"][0]["mapField"]["values"][0] == [1.0, 2.0]
    assert card["workspaceData"]["eofModes"][0]["pcSeries"][1]["value"] == -0.5


def test_saved_result_keeps_actual_polygon_outline_through_inputs(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    manager = StageManager(session.run_id, attempt, records=session.records)
    tools = AnalysisTools(session.records, session.artifacts, manager, functions={
        "build_polygon_mask": lambda polygon_points: {"mask": "ready"},
    })
    vertices = [[120, 20], [121, 20], [121, 21], [120, 21]]
    with manager.activate():
        mask = tools.call("build_polygon_mask", polygon_points=vertices)
        mask_ref = tools.ref(mask)
        field = xr.DataArray([[1.0, 2.0], [3.0, 4.0]],
                             coords={"lat": [20, 21], "lon": [120, 121]},
                             dims=("lat", "lon"))
        result_ref = tools.publish("polygon_mean", field, inputs=[mask_ref])
    adapter = ProgressAdapter(session)
    result = session.artifacts.read_artifact(result_ref)
    adapter.on_event({"type": "stage_result_indexed", "stage_id": result["stage_id"],
                      "entry": {"status": "completed", "artifact_id": result_ref}})
    card = adapter.finalize()["result_cards"][0]
    path = card["workspaceData"]["eventOverlays"][0]["path"]
    assert [[point["lon"], point["lat"]] for point in path] == [*vertices, vertices[0]]


def test_custom_transect_geometry_stays_with_saved_result(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    manager = StageManager(session.run_id, attempt, records=session.records)
    tools = AnalysisTools(session.records, session.artifacts, manager, functions={})
    vertices = [[120, 20], [121, 20.5], [122, 21]]
    with manager.activate():
        field = xr.DataArray([[1.0, 2.0], [3.0, 4.0]],
                             coords={"lat": [20, 21], "lon": [120, 121]},
                             dims=("lat", "lon"))
        result_ref = tools.publish("section_map", field, geometry={
            "type": "transect", "points": vertices,
        })
    adapter = ProgressAdapter(session)
    result = session.artifacts.read_artifact(result_ref)
    adapter.on_event({"type": "stage_result_indexed", "stage_id": result["stage_id"],
                      "entry": {"status": "completed", "artifact_id": result_ref}})
    path = adapter.finalize()["workspace_data_by_result"][result_ref]["eventOverlays"][0]["path"]
    assert [[point["lon"], point["lat"]] for point in path] == vertices


def test_multidimensional_saved_field_gets_bounded_spatial_preview(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    stage = session.records.new_stage(attempt, "Load 4D field")
    values = np.arange(2 * 2 * 3 * 3, dtype=float).reshape(2, 2, 3, 3)
    field = xr.DataArray(values, dims=("time", "depth", "lat", "lon"),
                         coords={"time": [0, 1], "depth": [0, -10],
                                 "lat": [20, 21, 22], "lon": [120, 121, 122]})
    artifact_id = session.artifacts.publish("load", field, run_id=session.run_id,
                                           attempt_id=attempt, stage_id=stage, inputs=[])
    adapter = ProgressAdapter(session)
    adapter.on_event({"type": "stage_result_indexed", "stage_id": stage,
                      "entry": {"status": "completed", "artifact_id": artifact_id}})
    card = adapter.finalize()["result_cards"][0]
    assert card["workspaceData"]["mapField"]["values"][0][0] == 13.5
    assert card["surface"] == "map"


def test_eddy_maps_keep_their_intermediate_results(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    stage = session.records.new_stage(attempt, "Detect two depths")
    adapter = ProgressAdapter(session)
    adapter.on_event({"type": "step_started", "stage_id": stage,
                      "title": "Detect two depths", "total_units": 2})
    for index, name in enumerate(("u_slice", "v_slice", "detect_eddies", "detect_eddies")):
        call = session.records.new_call(attempt, stage, name)
        value = ({"events": [{"type": "cyclonic", "center": {"lon": 120.5, "lat": 20.5},
                              "radius_km": 12.0}], "statistics": {"total_count": 1}}
                 if name == "detect_eddies" else {"field": name})
        artifact = session.artifacts.publish(name, value, run_id=session.run_id,
                                             attempt_id=attempt, stage_id=stage,
                                             inputs=[], call_id=call)
        adapter.on_event({"type": "stage_result_indexed", "stage_id": stage,
                          "entry": {"status": "completed", "artifact_id": artifact,
                                    "time": "2022-07-01", "depth": 0 if index == 2 else -50}})
    payload = adapter.finalize()
    eddy_cards = [result for result in payload["result_cards"] if result["type"] == "eddy_detection"]
    assert len(payload["result_cards"]) == 4
    assert len(eddy_cards) == 2
    assert {result["headline"] for result in eddy_cards} == {
        "time: 2022-07-01, depth: 0", "time: 2022-07-01, depth: -50",
    }
    assert all(payload["workspace_data_by_result"][result["id"]]["eventOverlays"]
               for result in eddy_cards)
    assert payload["result_summaries"][stage]["completed"] == 4


def test_ts_diagram_keeps_color_and_watermass_classes(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    stage = session.records.new_stage(attempt, "T-S diagram")
    artifact = session.artifacts.publish("build_watermass_ts_diagram", {
        "temperature": [18.0, 19.0], "salinity": [34.0, 35.0],
        "color_values": [20.0, 21.0], "point_classes": ["a", "b"],
        "metadata": {"temperature_variable": "temp", "salinity_variable": "salt",
                     "color_variable": "density", "color_range": [20.0, 21.0],
                     "class_color_map": {"a": "#ff0000", "b": "#0000ff"},
                     "watermass_bins": [{"id": "a", "name": "Water mass A", "color": "#ff0000"}]},
    }, run_id=session.run_id, attempt_id=attempt, stage_id=stage, inputs=[])
    adapter = ProgressAdapter(session)
    adapter.on_event({"type": "stage_result_indexed", "stage_id": stage,
                      "entry": {"status": "completed", "artifact_id": artifact}})
    card = adapter.finalize()["result_cards"][0]
    assert card["renderer"] == "ts_diagram"
    assert card["workspaceData"]["tsDiagramPoints"][1] == {
        "temperature": 19.0, "salinity": 35.0, "colorValue": 21.0, "pointClass": "b"}
    assert card["workspaceData"]["tsDiagramClassColorMap"]["a"] == "#ff0000"
    assert card["workspaceData"]["tsDiagramWatermassBins"][0]["name"] == "Water mass A"


def test_front_and_eddy_track_keep_interactive_map_shapes(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    stage = session.records.new_stage(attempt, "Detect events")
    adapter = ProgressAdapter(session)
    for name, value in (
        ("detect_fronts", {"event_type": "front", "events": [{
            "center": {"lon": 120.5, "lat": 20.5},
            "bbox": {"lon_min": 120.0, "lon_max": 121.0,
                     "lat_min": 20.0, "lat_max": 21.0}}]}),
        ("track_eddies", {"event_type": "eddy_track", "events": [{
            "track_id": "track_1", "path": [{"lon": 120.0, "lat": 20.0},
                                           {"lon": 121.0, "lat": 21.0}]}]}),
    ):
        artifact = session.artifacts.publish(name, value, run_id=session.run_id,
                                             attempt_id=attempt, stage_id=stage, inputs=[])
        adapter.on_event({"type": "stage_result_indexed", "stage_id": stage,
                          "entry": {"status": "completed", "artifact_id": artifact}})
    front, track = adapter.finalize()["result_cards"]
    assert front["renderer"] == track["renderer"] == "event"
    assert front["workspaceData"]["eventOverlays"][0]["shape"] == "rectangle"
    assert front["workspaceData"]["eventOverlays"][0]["bounds"] == {
        "lonMin": 120.0, "lonMax": 121.0, "latMin": 20.0, "latMax": 21.0}
    assert track["workspaceData"]["eventOverlays"][0]["shape"] == "polyline"
    assert track["workspaceData"]["eventOverlays"][0]["path"][1] == {"lon": 121.0, "lat": 21.0}


def test_eddy_overlay_keeps_its_input_map_context(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    stage = session.records.new_stage(attempt, "Eddies")
    adapter = ProgressAdapter(session)
    source = session.artifacts.publish("velocity", xr.DataArray(
        [[0.1, 0.2], [0.3, 0.4]], dims=("lat", "lon"),
        coords={"lat": [20, 21], "lon": [120, 121]}),
        run_id=session.run_id, attempt_id=attempt, stage_id=stage, inputs=[])
    adapter.on_event({"type": "stage_result_indexed", "stage_id": stage,
                      "entry": {"status": "completed", "artifact_id": source}})
    eddies = session.artifacts.publish("detect_eddies", {
        "event_type": "eddy", "events": [{"center": {"lon": 120.5, "lat": 20.5},
                                          "radius_km": 12.0}],
        "coordinates": {"lat": [20, 21], "lon": [120, 121]},
        "ow_field": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "vorticity_field": np.ones((300, 300)),
    }, run_id=session.run_id, attempt_id=attempt, stage_id=stage, inputs=[source])
    adapter.on_event({"type": "stage_result_indexed", "stage_id": stage,
                      "entry": {"status": "completed", "artifact_id": eddies}})
    card = adapter.finalize()["result_cards"][-1]
    assert card["workspaceData"]["eventOverlays"][0]["shape"] == "circle"
    assert card["workspaceData"]["mapField"]["values"] == [[1.0, 2.0], [3.0, 4.0]]


def test_nested_trend_field_and_trend_line_keep_map_and_chart(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    stage = session.records.new_stage(attempt, "Trend")
    slope = xr.DataArray([[0.1, 0.2], [0.3, 0.4]], dims=("lat", "lon"),
                         coords={"lat": [20, 21], "lon": [120, 121]},
                         attrs={"units": "°C/year"})
    artifacts = [
        session.artifacts.publish("compute_field_trend", {"slope": slope},
                                  run_id=session.run_id, attempt_id=attempt,
                                  stage_id=stage, inputs=[]),
        session.artifacts.publish("compute_trend", {
            "times": ["2020-01-01", "2021-01-01"], "values": [1.0, 2.0],
            "trend_line": [1.1, 1.9]}, run_id=session.run_id,
            attempt_id=attempt, stage_id=stage, inputs=[]),
    ]
    adapter = ProgressAdapter(session)
    for artifact in artifacts:
        adapter.on_event({"type": "stage_result_indexed", "stage_id": stage,
                          "entry": {"status": "completed", "artifact_id": artifact}})
    map_card, trend_card = adapter.finalize()["result_cards"]
    assert map_card["workspaceData"]["mapField"]["values"][1] == [0.3, 0.4]
    assert trend_card["renderer"] == "timeseries"
    assert trend_card["workspaceData"]["anomalySeries"][1]["value"] == 1.9
