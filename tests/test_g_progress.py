from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import xarray as xr

from apps.api.langgraph_progress import ProgressAdapter
from packages.agent_loop.analysis import AnalysisSession


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


def test_saved_field_and_derived_number_are_visible_as_map_and_metric(tmp_path):
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    stage = session.records.new_stage(attempt, "Mean chlorophyll")
    adapter = ProgressAdapter(session)
    field = xr.DataArray(np.array([[0.5, 0.8], [1.0, 1.3]]), dims=("lat", "lon"),
                         coords={"lat": [20.0, 20.1], "lon": [120.0, 120.1]},
                         name="chlorophyll", attrs={"units": "mg m-3"})
    field_id = session.artifacts.publish("load_dataset", field, run_id=session.run_id,
                                         attempt_id=attempt, stage_id=stage, inputs=[])
    mean_id = session.artifacts.publish("mean", {"mean": 0.9}, run_id=session.run_id,
                                        attempt_id=attempt, stage_id=stage, inputs=[field_id])
    for artifact_id in (field_id, mean_id):
        adapter.on_event({"type": "stage_result_indexed", "stage_id": stage,
                          "entry": {"status": "completed", "artifact_id": artifact_id}})
    field_card, mean_card = adapter.finalize()["result_cards"]
    assert field_card["workspaceData"]["mapField"]["values"] == [[0.5, 0.8], [1.0, 1.3]]
    assert mean_card["metrics"] == [{"label": "mean", "value": "0.9"}]
    assert mean_card["workspaceData"]["mapField"]["variable"] == "chlorophyll"
    assert mean_card["surface"] == "map"
    assert adapter.finalize()["active_result_id"] == mean_id


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
