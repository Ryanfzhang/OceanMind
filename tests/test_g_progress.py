from __future__ import annotations

from types import SimpleNamespace

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


def test_result_previews_are_bounded_but_counts_cover_all_results():
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
        "completed": 100, "failed": 0, "preview_count": 2,
    }
    assert len(payload["step_cards"][0]["results"]) == 2
    assert len(payload["result_cards"]) == 2
    assert [event["payload"]["type"] for event in envelopes].count("step_result_attached") == 2


def test_two_eddy_maps_replace_intermediate_field_previews(tmp_path):
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
    assert len(payload["result_cards"]) == 2
    assert all(result["type"] == "eddy_detection" for result in payload["result_cards"])
    assert {result["headline"] for result in payload["result_cards"]} == {
        "time: 2022-07-01, depth: 0", "time: 2022-07-01, depth: -50",
    }
    assert all(payload["workspace_data_by_result"][result["id"]]["eventOverlays"]
               for result in payload["result_cards"])
    assert payload["result_summaries"][stage]["completed"] == 4
