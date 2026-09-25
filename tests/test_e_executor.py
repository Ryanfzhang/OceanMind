"""Child execution, persistence, event delivery, and bounded failure behavior."""

import os
import time
from pathlib import Path

import pytest

from packages.analysis_runtime.artifacts import ArtifactStore
from packages.analysis_runtime.code_store import CodeStore
from packages.analysis_runtime.executor import run_script
from packages.analysis_runtime.records import RunRecords


def _local_test_launcher(python: str, args: list[str], root: Path, read_roots):
    """Explicit unsandboxed seam for disposable tests, never used by the agent graph."""
    return [python, *args]


def _setup(tmp_path):
    records = RunRecords(tmp_path)
    return records, CodeStore(tmp_path, records.run_id)


def _run(tmp_path, records, store, code_id, **kwargs):
    return run_script(root=tmp_path, run_id=records.run_id, code_id=code_id,
                      code_store=store, launcher=_local_test_launcher, **kwargs)


def test_saved_script_runs_chain_and_sends_events_before_exit(tmp_path):
    records, store = _setup(tmp_path)
    sentinel = tmp_path / "finished.txt"
    source = f'''from oceanmind_runtime import tools, stage
import time
from pathlib import Path
with stage("计算", total=2, unit="项") as progress:
    first = tools.publish("first", {{"value": 2}}, inputs=[])
    progress.advance()
    time.sleep(0.2)
    value = tools.load_result(first)
    tools.publish("second", {{"value": value["value"] * 3}}, inputs=[first])
    progress.advance()
print("ordinary stdout with {{braces}}")
Path({str(sentinel)!r}).write_text("done")
'''
    code_id = store.write_analysis(source)["code_id"]
    observations = []
    result = _run(tmp_path, records, store, code_id,
                  on_event=lambda event: observations.append((event, sentinel.exists())))
    assert result["status"] == "completed" and result["exit_code"] == 0
    assert result["call_count"] == result["result_count"] == 2
    assert result["stage_count"] == 1
    assert result["code_id"] == records.read("attempt", result["attempt_id"])["code_version"]
    assert any(event["type"] == "step_progress" and not finished
               for event, finished in observations)
    assert "ordinary stdout with {braces}" in result["stdout"]
    second = result["result_preview"][1]
    assert ArtifactStore(tmp_path).load_result(second["artifact_id"]) == {"value": 6}


def test_revision_reuses_prior_artifact_without_repeating_call(tmp_path):
    records, store = _setup(tmp_path)
    first_code = store.write_analysis(
        'from oceanmind_runtime import tools\n'
        'tools.publish("saved", [1, 2, 3], inputs=[])\n')
    first = _run(tmp_path, records, store, first_code["code_id"])
    assert first["status"] == "completed"
    artifact_id = first["result_preview"][0]["artifact_id"]
    second_code = store.write_analysis(
        'from oceanmind_runtime import tools\n'
        f'prior = tools.load_result({artifact_id!r})\n'
        f'tools.publish("derived", sum(prior), inputs=[{artifact_id!r}])\n',
        previous_version=first_code["code_id"])
    second = _run(tmp_path, records, store, second_code["code_id"])
    assert second["status"] == "completed"
    assert second["call_count"] == 1
    assert ArtifactStore(tmp_path).load_result(second["result_preview"][0]["artifact_id"]) == 6
    assert records.read("attempt", second["attempt_id"])["code_version"] == second_code["code_id"]


def test_crash_closes_running_call_and_stage_but_keeps_prior_result(tmp_path):
    records, store = _setup(tmp_path)
    code_id = store.write_analysis('''from oceanmind_runtime import tools, stage
import os
with stage("工作"):
    tools.publish("good", {"ok": True}, inputs=[])
    def crash():
        os._exit(23)
    tools.functions["crash"] = crash
    tools.crash()
''')["code_id"]
    result = _run(tmp_path, records, store, code_id)
    assert result["status"] == "failed" and result["exit_code"] == 23
    assert result["failed_call_preview"][0]["name"] == "crash"
    assert result["stages"][0]["status"] == "failed"
    assert ArtifactStore(tmp_path).load_result(result["result_preview"][0]["artifact_id"]) == {"ok": True}
    assert records.read("attempt", result["attempt_id"])["status"] == "failed"


def test_timeout_closes_stage_and_logs_are_bounded_and_secrets_absent(tmp_path, monkeypatch):
    records, store = _setup(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-must-not-pass")
    code_id = store.write_analysis('''from oceanmind_runtime import stage
import os, time
print("key=" + str(os.getenv("DEEPSEEK_API_KEY")))
print("x" * 100000)
with stage("等待"):
    time.sleep(30)
''')["code_id"]
    result = _run(tmp_path, records, store, code_id,
                  timeout_seconds=3, max_log_bytes=1024)
    assert result["status"] == "timed_out" and result["timed_out"]
    assert result["stdout_truncated"] and len(result["stdout"]) < 1100
    assert "key=None" in result["stdout"]
    assert "secret-must-not-pass" not in result["stdout"]
    assert all(stage["status"] == "failed" for stage in result["stages"])


def test_executor_requires_launcher_and_verified_code(tmp_path):
    records, store = _setup(tmp_path)
    code_id = store.write_analysis('print("ok")\n')["code_id"]
    with pytest.raises(ValueError, match="launcher"):
        run_script(root=tmp_path, run_id=records.run_id, code_id=code_id,
                   code_store=store, launcher=None)
    path = store.get_path(code_id)
    path.write_text('print("tampered")\n')
    with pytest.raises(ValueError, match="changed"):
        _run(tmp_path, records, store, code_id)


def test_self_modified_code_cannot_be_reported_as_completed(tmp_path):
    records, store = _setup(tmp_path)
    code_id = store.write_analysis(
        'from pathlib import Path\nPath(__file__).write_text("print(2)\\n")\n'
    )["code_id"]
    result = _run(tmp_path, records, store, code_id)
    assert result["exit_code"] == 0
    assert result["status"] == "failed"
    assert "Saved code changed" in result["error"]
    assert records.read("attempt", result["attempt_id"])["status"] == "failed"


def test_zero_exit_with_missing_saved_result_is_not_reported_as_valid(tmp_path):
    records, store = _setup(tmp_path)
    code_id = store.write_analysis('''from pathlib import Path
from oceanmind_runtime import tools
ref = tools.publish("missing_later", {"value": 1}, inputs=[])
metadata = tools.read_artifact(ref)
(Path(tools.records.root) / metadata["payload"]).unlink()
''')["code_id"]
    result = _run(tmp_path, records, store, code_id)
    assert result["exit_code"] == 0
    assert result["status"] == "failed"
    assert result["invalid_result_count"] == 1
    assert result["invalid_result_preview"] == [result["result_preview"][0]["artifact_id"]]


def test_zero_exit_with_redirected_artifact_directory_is_not_valid(tmp_path):
    records, store = _setup(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    code_id = store.write_analysis(f'''from pathlib import Path
from oceanmind_runtime import tools
ref = tools.publish("value", {{"count": 1}}, inputs=[])
root = Path(tools.records.root)
data = root / "artifacts" / "data"
data.rename(root / "artifacts" / "old_data")
data.symlink_to({str(outside)!r}, target_is_directory=True)
''')["code_id"]
    result = _run(tmp_path, records, store, code_id)
    assert result["exit_code"] == 0
    assert result["status"] == "failed"
    assert result["invalid_result_count"] == 1


def test_attempt_cannot_claim_another_code_version(tmp_path):
    records, store = _setup(tmp_path)
    code_id = store.write_analysis('''from oceanmind_runtime import tools
tools.records.update("attempt", tools.stages.attempt_id,
                     code_version="code_00000000000000000000000000000000")
''')["code_id"]
    result = _run(tmp_path, records, store, code_id)
    assert result["exit_code"] == 0
    assert result["status"] == "failed"
    assert "code version" in result["error"]


def test_large_script_stage_title_is_rejected_before_model_observation(tmp_path):
    records, store = _setup(tmp_path)
    code_id = store.write_analysis('''from oceanmind_runtime import stage
with stage("x" * 1000000):
    pass
''')["code_id"]
    result = _run(tmp_path, records, store, code_id)
    assert result["status"] == "failed"
    assert result["stage_count"] == 0
    assert len(result["stderr"]) < 2000


def test_many_calls_return_recent_bounded_preview(tmp_path):
    records, store = _setup(tmp_path)
    code_id = store.write_analysis('''from oceanmind_runtime import tools, stage
with stage("批量", total=8) as progress:
    for index in range(8):
        tools.publish("item", index, inputs=[])
        progress.advance()
''')["code_id"]
    result = _run(tmp_path, records, store, code_id)
    assert result["status"] == "completed"
    assert result["call_count"] == result["result_count"] == 8
    assert len(result["result_preview"]) == 5
    last_id = result["result_preview"][-1]["artifact_id"]
    assert ArtifactStore(tmp_path).load_result(last_id) == 7


def test_real_ocean_tool_and_custom_calculation_share_one_attempt(tmp_path):
    records, store = _setup(tmp_path)
    code_id = store.write_analysis('''from oceanmind_runtime import tools, stage
with stage("趋势分析"):
    series = {"times": ["2020-01-01", "2021-01-01", "2022-01-01"],
              "values": [1.0, 2.0, 3.0]}
    trend = tools.compute_trend(timeseries=series)
    tools.publish("slope_summary", {"slope": trend["slope"]},
                  inputs=[tools.ref(trend)])
''')["code_id"]
    result = _run(tmp_path, records, store, code_id)
    assert result["status"] == "completed", result["stderr"]
    assert result["call_count"] == 2
    assert result["result_count"] == 2
    assert result["result_preview"][0]["artifact_id"]
    summary = ArtifactStore(tmp_path).load_result(result["result_preview"][1]["artifact_id"])
    assert 0.9 < summary["slope"] < 1.1


def test_script_cannot_load_other_runs_result(tmp_path):
    first_run = RunRecords(tmp_path)
    first_store = CodeStore(tmp_path, first_run.run_id)
    first_code = first_store.write_analysis(
        'from oceanmind_runtime import tools\n'
        'tools.publish("private", 42, inputs=[])\n')["code_id"]
    first = _run(tmp_path, first_run, first_store, first_code)
    artifact_id = first["result_preview"][0]["artifact_id"]

    second_run = RunRecords(tmp_path)
    second_store = CodeStore(tmp_path, second_run.run_id)
    second_code = second_store.write_analysis(
        'from oceanmind_runtime import tools\n'
        f'tools.load_result({artifact_id!r})\n')["code_id"]
    second = _run(tmp_path, second_run, second_store, second_code)
    assert second["status"] == "failed"
    assert "another run" in second["stderr"]


def test_finished_worker_stops_inherited_grandchild(tmp_path):
    records, store = _setup(tmp_path)
    started = tmp_path / "child-started"
    escaped = tmp_path / "child-escaped"
    child_code = ("from pathlib import Path; import time; "
                  f"Path({str(started)!r}).write_text('started'); "
                  "time.sleep(0.8); "
                  f"Path({str(escaped)!r}).write_text('escaped')")
    source = ("from pathlib import Path\nimport subprocess, sys, time\n"
              f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
              f"for _ in range(100):\n"
              f"    if Path({str(started)!r}).exists(): break\n"
              "    time.sleep(0.01)\n")
    code_id = store.write_analysis(source)["code_id"]
    result = _run(tmp_path, records, store, code_id)
    assert result["status"] == "completed"
    assert started.exists()
    time.sleep(1)
    assert not escaped.exists()
