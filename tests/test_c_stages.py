"""Stage grouping and lightweight call index checks."""

import pytest

from packages.analysis_runtime.stages import (
    StageCountError,
    StageLimitError,
    StageManager,
    current_stage,
    stage,
)
from packages.analysis_runtime.records import RunRecords


def manager(**kwargs):
    return StageManager("run_a", "attempt_a", **kwargs)


def test_lifecycle_and_context_reset():
    seen = []
    stages = manager(callback=seen.append)
    with stages.stage("读取数据") as first:
        assert current_stage() is first
        assert first.id == stages.events[0]["step_id"]
    assert current_stage() is None
    assert [event["type"] for event in seen] == ["step_started", "step_completed"]
    with pytest.raises(RuntimeError, match="broken"):
        with stages.stage("绘图"):
            raise RuntimeError("broken")
    assert [event["type"] for event in seen[-2:]] == ["step_started", "step_failed"]
    assert stages.stages[0].id != stages.stages[1].id


def test_two_calls_per_slice_advance_only_once():
    stages = manager()
    with stages.stage("检测", total=3, unit="切片") as progress:
        for depth in range(3):
            progress.set_current(time="2024-01-01", depth=depth)
            for call in range(2):
                stages.register_call_result(
                    f"call_{depth}_{call}", status="completed", artifact_id=f"art_{depth}_{call}"
                )
            assert progress.completed == depth
            progress.advance()
    assert progress.completed == 3
    assert len(progress.results) == 6
    assert {entry["stage_id"] for entry in progress.results} == {progress.id}
    assert stages.events[-1]["percent"] == 100


def test_unknown_total_and_index_does_not_contain_results():
    stages = manager()
    with stages.stage("批量计算") as progress:
        for n in range(100):
            progress.set_current(time=f"2024-01-{n + 1:03d}", depth=n)
            stages.register_call_result(f"call_{n}", status="completed", artifact_id=f"art_{n}")
            progress.advance()
    assert len(stages.result_index) == 100
    assert stages.result_index[0]["time"] == "2024-01-001"
    assert stages.result_index[-1]["depth"] == 99
    assert all("total_units" not in event and "percent" not in event for event in stages.events)
    assert all("result" not in entry for entry in stages.result_index)


def test_count_mismatch_overrun_empty_and_failure_result():
    stages = manager()
    with pytest.raises(StageCountError, match="1/2"):
        with stages.stage("检测", total=2) as progress:
            progress.advance()
            stages.register_call_result("bad", status="failed", summary="input mismatch")
    assert progress.status == "failed"
    assert progress.completed == 1
    assert stages.result_index[0]["status"] == "failed"
    with pytest.raises(StageCountError, match="exceed"):
        with stages.stage("越界", total=0) as empty:
            empty.advance()
    with stages.stage("空输入", total=0) as empty:
        pass
    assert empty.results == []
    assert stages.events[-1]["empty_input"] is True


def test_default_stage_reused_and_limit_keeps_prior_results():
    stages = manager(max_stages=1)
    first = stages.ensure_stage()
    for n in range(100):
        assert stages.ensure_stage() is first
        stages.register_call_result(f"call_{n}", status="completed", artifact_id=f"art_{n}")
    with pytest.raises(StageLimitError, match="outside the loop"):
        with stages.stage("too many"):
            pass
    stages.close()
    assert len(stages.stages) == 1
    assert len(stages.result_index) == 100
    assert [event["type"] for event in stages.events if event["type"].startswith("step_")] == [
        "step_started", "step_completed"
    ]


def test_invalid_coordinates_and_summary_reject_large_values():
    stages = manager()
    with stages.stage("safe") as progress:
        with pytest.raises(TypeError, match="scalar"):
            progress.set_current(depth=[1, 2, 3])
        with pytest.raises(ValueError, match="finite"):
            progress.set_current(depth=float("nan"))
        with pytest.raises(ValueError, match="500"):
            stages.register_call_result("call_1", status="completed", summary="x" * 501)
        assert progress.results == []


def test_stage_ids_and_result_index_survive_record_reload(tmp_path):
    records = RunRecords(tmp_path)
    attempt_id = records.new_attempt()
    stages = StageManager(records.run_id, attempt_id, records=records)
    with stages.stage("检测", total=2) as progress:
        for depth in (0, 100):
            progress.set_current(time="2024-01-01", depth=depth)
            stages.register_call_result(
                f"call_{depth}", status="completed", artifact_id=f"art_{depth}"
            )
            progress.advance()
    saved = RunRecords(tmp_path, records.run_id).read("stage", progress.id)
    assert saved["status"] == "completed"
    assert [item["depth"] for item in saved["result_index"]] == [0, 100]
    assert saved["completed_units"] == 2


def test_script_facing_stage_and_default_close():
    stages = manager()
    with pytest.raises(RuntimeError, match="active analysis script"):
        with stage("outside"):
            pass
    with stages.activate():
        with stage("inside") as progress:
            assert stages.ensure_stage() is progress
        stages.ensure_stage()
    assert [item.status for item in stages.stages] == ["completed", "completed"]
