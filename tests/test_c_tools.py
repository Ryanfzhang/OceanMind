"""Tool calls retain raw values while saving calls, progress, and results."""

import pytest
import xarray as xr

from packages.analysis_runtime.artifacts import ArtifactStore
from packages.analysis_runtime.records import RunRecords
from packages.analysis_runtime.stages import StageManager
from packages.analysis_runtime.tools import AnalysisTools
from packages.tool_loader.progress import report_tool_progress


def runtime(tmp_path, functions):
    records = RunRecords(tmp_path)
    attempt_id = records.new_attempt()
    artifacts = ArtifactStore(tmp_path)
    stages = StageManager(records.run_id, attempt_id, records=records)
    return records, artifacts, stages, AnalysisTools(records, artifacts, stages, functions=functions)


def test_call_progress_raw_result_and_reloaded_snapshot(tmp_path):
    def field():
        report_tool_progress(phase="read", percent=0.4)
        return xr.DataArray(
            [[1.0, 2.0]], dims=("lat", "lon"),
            coords={"lat": [18.0], "lon": [110.0, 111.0]}, attrs={"units": "m/s"},
        )

    records, artifacts, stages, tools = runtime(tmp_path, {"field": field})
    with stages.activate():
        data = tools.field()
        artifact_id = tools.ref(data)
        assert isinstance(data, xr.DataArray)
        data.values[0, 0] = 99

    restored = ArtifactStore(tmp_path).load_result(artifact_id)
    assert artifacts.read_artifact(artifact_id)["presentation"] == "summary"
    assert restored.values[0, 0] == 1
    assert restored.attrs["units"] == "m/s"
    assert restored.lon.values.tolist() == [110.0, 111.0]
    events = [event["type"] for event in stages.events]
    assert events.index("call_started") < events.index("call_progress") < events.index("call_completed")
    assert events.index("call_progress") < events.index("call_completed")
    assert not any(event.startswith("step_") for event in events)
    assert stages.stages[0].completed == 0
    call_id = stages.result_index[0]["call_id"]
    assert records.read("call", call_id)["artifact_ids"] == [artifact_id]


def test_failed_third_call_keeps_first_two_results(tmp_path):
    def item(value):
        if value == 3:
            raise ValueError("bad third input")
        return {"value": value}

    records, artifacts, stages, tools = runtime(tmp_path, {"item": item})
    with pytest.raises(ValueError, match="bad third input"):
        with stages.activate():
            first = tools.item(1)
            second = tools.item(value=2)
            tools.item(value=3)

    assert first == {"value": 1} and second == {"value": 2}
    assert [entry["status"] for entry in stages.result_index] == [
        "completed", "completed", "failed"
    ]
    first_id, second_id = [entry["artifact_id"] for entry in stages.result_index[:2]]
    assert ArtifactStore(tmp_path).load_result(first_id) == first
    assert ArtifactStore(tmp_path).load_result(second_id) == second
    assert records.read("call", stages.result_index[-1]["call_id"])["status"] == "failed"
    assert stages.stages[0].status == "failed"


def test_source_field_and_derived_field_have_different_presentation(tmp_path):
    def field():
        return xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"),
                            coords={"lat": [20.0, 21.0], "lon": [120.0, 121.0]})

    _, artifacts, stages, tools = runtime(tmp_path, {"field": field})
    with stages.activate():
        with stages.stage("Read data"):
            source = tools.field()
        with stages.stage("Compute field"):
            derived_id = tools.publish("doubled", source * 2, inputs=[tools.ref(source)])
    assert artifacts.read_artifact(tools.ref(source))["presentation"] == "summary"
    assert artifacts.read_artifact(derived_id)["presentation"] == "auto"
    assert [event["type"] for event in stages.events if event["type"].startswith("step_")] == [
        "step_started", "step_completed", "step_started", "step_completed",
    ]


def test_computed_tool_field_from_unsaved_array_stays_a_map(tmp_path):
    _, artifacts, stages, tools = runtime(tmp_path, {"double": lambda data: data * 2})
    raw = xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=("lat", "lon"),
                       coords={"lat": [20.0, 21.0], "lon": [120.0, 121.0]})
    with stages.activate():
        computed = tools.double(data=raw)
    assert artifacts.read_artifact(tools.ref(computed))["presentation"] == "auto"


def test_custom_publish_and_reload_between_contexts(tmp_path):
    records, artifacts, stages, tools = runtime(tmp_path, {})
    with stages.activate():
        artifact_id = tools.publish("custom_mean", {"mean": 1.5}, inputs=[])
        unlinked_id = tools.publish("standalone_note", {"note": "ok"})
    assert artifacts.read_artifact(unlinked_id)["inputs"] == []
    reopened = RunRecords(tmp_path, records.run_id)
    next_attempt = reopened.new_attempt()
    next_stages = StageManager(reopened.run_id, next_attempt, records=reopened)
    next_tools = AnalysisTools(reopened, ArtifactStore(tmp_path), next_stages, functions={})
    with next_stages.activate():
        prior = next_tools.load_result(artifact_id)
        followup_id = next_tools.publish("doubled_mean", {"mean": prior["mean"] * 2},
                                         inputs=[artifact_id])
    other = ArtifactStore(tmp_path)
    assert other.load_result(artifact_id) == {"mean": 1.5}
    assert other.load_result(followup_id) == {"mean": 3.0}
    assert other.read_artifact(followup_id)["inputs"] == [artifact_id]
    assert stages.result_index[0]["artifact_id"] == artifact_id
    assert not any(event["type"].startswith("step_") for event in stages.events)
    assert all(event.get("attempt_id") == stages.attempt_id for event in stages.events)
