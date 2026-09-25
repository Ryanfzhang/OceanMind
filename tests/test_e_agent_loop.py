"""Offline acceptance paths through the single model → tool → model loop."""

import json

import numpy as np
import pytest
import xarray as xr

from packages.agent_loop.analysis import AnalysisSession
from packages.agent_loop.graph import _configured_data_prompt, build_graph
from packages.agent_loop.run import run_query
from packages.agent_loop.state import initial_state
from packages.analysis_runtime.artifacts import ArtifactStore
from packages.runtime.dataset_config import DatasetConfig


def call(name, arguments, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def action(name, arguments, call_id):
    return {"role": "assistant", "content": None,
            "tool_calls": [call(name, arguments, call_id)]}


def observation(messages):
    assert messages[-1]["role"] == "tool"
    return json.loads(messages[-1]["content"])


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen = []

    def complete(self, messages, *, tools, timeout):
        self.seen.append(messages)
        assert {"write_analysis", "read_artifact", "run_analysis"}.issubset(
            {schema["function"]["name"] for schema in tools})
        response = next(self.responses)
        return response(messages) if callable(response) else response


@pytest.fixture
def local_worker(monkeypatch):
    # The production AnalysisSession still uses its verified sandbox builder.
    # These disposable offline tests only replace the process launch command.
    monkeypatch.setattr(
        "packages.agent_loop.analysis.build_sandbox_command",
        lambda python, args, root, read_roots: [python, *args],
    )


def run_loop(model, session, query):
    return build_graph(model, analysis_session=session, max_rounds=10).invoke(
        initial_state(query, run_id=session.run_id, run_root=str(session.root)))


def test_configured_dataset_guidance_only_applies_to_its_source(tmp_path, monkeypatch):
    configured = tmp_path / "cmoms"
    configured.mkdir()
    monkeypatch.setattr("packages.agent_loop.graph.get_active_dataset_config", lambda: DatasetConfig(
        name="CMOMS", data_path=str(configured), backend="zarr",
        variables=["temp", "salt"], depth_levels=[0, -100],
    ))
    known = _configured_data_prompt((configured,))
    assert "tools.load_dataset uses this configured source" in known
    assert "temp, salt" in known
    assert _configured_data_prompt((tmp_path / "other",)) == ""
    model = ScriptedModel([{"role": "assistant", "content": "done"}])
    run_loop(model, AnalysisSession(tmp_path / "runs", data_roots=[configured]),
             "Plot a T-S diagram")
    prompt = model.seen[0][0]["content"]
    assert known in prompt
    assert "do not write and run probe scripts" in prompt
    assert "interactive T-S chart" in prompt
    assert "never pass their `.data` arrays" in prompt
    assert "without rereading large arrays" in prompt
    assert "English stage titles" in prompt


def test_direct_entry_point_creates_run_scoped_analysis_session(tmp_path):
    model = ScriptedModel([{"role": "assistant", "content": "No data needed."}])
    state = run_query("Hello", model=model, analysis_workspace=tmp_path / "runs")
    assert state["status"] == "completed"
    assert state["run_id"].startswith("run_")
    assert state["run_root"].startswith(str(tmp_path / "runs"))


def test_agent_writes_reads_and_runs_one_eddy_script(tmp_path, local_worker):
    events = []
    session = AnalysisSession(tmp_path / "work", on_event=events.append)
    source = '''import numpy as np
import xarray as xr
from oceanmind_runtime import stage, tools
coords = {"lat": np.linspace(18, 20, 5), "lon": np.linspace(110, 112, 5)}
u = xr.DataArray(np.zeros((5, 5)), dims=("lat", "lon"), coords=coords)
v = xr.DataArray(np.zeros((5, 5)), dims=("lat", "lon"), coords=coords)
with stage("涡旋检测"):
    result = tools.detect_eddies(u=u, v=v)
'''

    def read_code(messages):
        saved = observation(messages)
        assert saved["code_id"].startswith("code_")
        return action("read_artifact", {"ref": saved["code_id"]}, "read")

    def run_code(messages):
        read = observation(messages)
        assert read["content"] == source
        return action("run_analysis", {"code_id": read["code_id"]}, "run")

    def answer(messages):
        result = observation(messages)
        assert result["status"] == "completed", result.get("stderr")
        assert result["stage_count"] == result["call_count"] == result["result_count"] == 1
        return {"role": "assistant", "content": "脚本已运行；涡旋事件数以保存结果为准。"}

    model = ScriptedModel([
        action("write_analysis", {"code": source}, "write"),
        read_code, run_code, answer,
    ])
    state = run_loop(model, session, "运行涡旋检测并保存脚本和结果")
    assert state["status"] == "completed" and state["rounds"] == 4
    result = json.loads(state["messages"][-2]["content"])
    saved = ArtifactStore(session.root).load_result(result["result_preview"][0]["artifact_id"])
    assert saved["event_type"] == "eddy" and saved["statistics"]["total_count"] == 0
    assert [event["type"] for event in events if event["type"].startswith("step_")] == [
        "step_started", "step_completed"]


def test_unknown_tracer_is_inspected_and_analyzed_without_skill(tmp_path, local_worker):
    source = tmp_path / "cruise_random_name.nc"
    xr.Dataset({"new_tracer": (("time", "lat", "lon"),
                              np.array([[[1.0, 2.0], [3.0, 4.0]]]))}).to_netcdf(source)
    session = AnalysisSession(tmp_path / "work", data_roots=[source])
    script = f'''import numpy as np
import xarray as xr
from oceanmind_runtime import stage, publish
with stage("新示踪剂统计"):
    with xr.open_dataset({str(source)!r}) as data:
        values = data["new_tracer"].values
    valid = values[np.isfinite(values)]
    publish("new_tracer_summary", {{"valid_count": int(valid.size),
        "mean": float(valid.mean()) if valid.size else None}}, inputs=[])
'''

    def write(messages):
        inspected = observation(messages)
        assert inspected["variables"][0]["name"] == "new_tracer"
        assert inspected["dimensions"] == {"time": 1, "lat": 2, "lon": 2}
        return action("write_analysis", {"code": script}, "write")

    def execute(messages):
        return action("run_analysis", {"code_id": observation(messages)["code_id"]}, "run")

    def read_result(messages):
        ran = observation(messages)
        assert ran["status"] == "completed" and ran["result_count"] == 1
        return action("read_artifact", {"ref": ran["result_preview"][0]["artifact_id"]}, "read")

    def answer(messages):
        content = observation(messages)["content"]
        assert json.loads(content)["value"] == {"valid_count": 4, "mean": 2.5}
        return {"role": "assistant", "content": "new_tracer 有 4 个有效值，均值为 2.5。"}

    model = ScriptedModel([
        action("inspect_data", {"source": str(source), "variable": "new_tracer"}, "inspect"),
        write, execute, read_result, answer,
    ])
    state = run_loop(model, session, "分析此文件中的 new_tracer 平均值")
    assert state["status"] == "completed"
    assert "2.5" in state["messages"][-1]["content"]
    assert "read_skill" not in [message["tool_calls"][0]["function"]["name"]
                                for message in state["messages"]
                                if message["role"] == "assistant" and message.get("tool_calls")]


def test_agent_revises_failed_code_and_retries(tmp_path, local_worker):
    session = AnalysisSession(tmp_path / "work")
    broken = 'from oceanmind_runtime import stage, publish\nwith stage("计算"):\n    value = 1 / 0\n'
    repaired = ('from oceanmind_runtime import stage, publish\n'
                'with stage("计算"):\n    publish("answer", {"value": 2}, inputs=[])\n')
    versions = []

    def first_run(messages):
        code_id = observation(messages)["code_id"]
        versions.append(code_id)
        return action("run_analysis", {"code_id": code_id}, "run_bad")

    def revise(messages):
        result = observation(messages)
        assert result["status"] == "failed" and "ZeroDivisionError" in result["stderr"]
        assert result["stages"][0]["status"] == "failed"
        return action("write_analysis", {"code": repaired, "previous_version": versions[0]}, "fix")

    def second_run(messages):
        saved = observation(messages)
        assert saved["previous_version"] == versions[0]
        assert "-    value = 1 / 0" in saved["diff"]
        versions.append(saved["code_id"])
        return action("run_analysis", {"code_id": saved["code_id"]}, "run_fixed")

    def answer(messages):
        result = observation(messages)
        assert result["status"] == "completed" and result["result_count"] == 1
        assert result["code_id"] == versions[1]
        return {"role": "assistant", "content": "修正错误后已成功运行第二版代码。"}

    model = ScriptedModel([
        action("write_analysis", {"code": broken}, "write_bad"),
        first_run, revise, second_run, answer,
    ])
    state = run_loop(model, session, "执行计算，出错后修正")
    assert state["status"] == "completed" and len(set(versions)) == 2
    assert ArtifactStore(session.root).load_result(
        json.loads(state["messages"][-2]["content"])["result_preview"][0]["artifact_id"]
    ) == {"value": 2}


def test_repeating_same_failed_script_can_be_repaired(tmp_path, local_worker):
    session = AnalysisSession(tmp_path / "work")
    bad_code = 'raise ValueError("same failure")\n'
    code_id = None
    attempts = []

    def run_once(messages):
        nonlocal code_id
        code_id = observation(messages)["code_id"]
        return action("run_analysis", {"code_id": code_id}, "first_run")

    def repeat(messages):
        first = observation(messages)
        assert first["status"] == "failed" and "same failure" in first["stderr"]
        attempts.append(first["attempt_id"])
        return action("run_analysis", {"code_id": code_id}, "second_run")

    def repair(messages):
        second = observation(messages)
        attempts.append(second["attempt_id"])
        assert second["status"] == "failed"
        return action("write_analysis", {"code": 'print("fixed")\n',
                                          "previous_version": code_id}, "write_fixed")

    def run_fixed(messages):
        return action("run_analysis", {"code_id": observation(messages)["code_id"]}, "third_run")

    model = ScriptedModel([
        action("write_analysis", {"code": bad_code}, "write"), run_once, repeat,
        repair, run_fixed, {"role": "assistant", "content": "The corrected script ran."},
    ])
    state = run_loop(model, session, "运行分析脚本")
    assert len(set(attempts)) == 2
    assert state["status"] == "completed"
    assert state["rounds"] == 6


def test_successful_run_with_no_valid_values_is_not_called_a_valid_mean(tmp_path, local_worker):
    source = tmp_path / "missing.nc"
    xr.Dataset({"new_tracer": (("time", "lat", "lon"),
                              np.full((1, 2, 2), np.nan))}).to_netcdf(source)
    session = AnalysisSession(tmp_path / "work", data_roots=[source])
    script = f'''import numpy as np
import xarray as xr
from oceanmind_runtime import stage, publish
with stage("缺测检查"):
    with xr.open_dataset({str(source)!r}) as data:
        values = data["new_tracer"].values
    valid = values[np.isfinite(values)]
    publish("quality", {{"valid_count": int(valid.size),
        "mean": float(valid.mean()) if valid.size else None}}, inputs=[])
'''

    def write(messages):
        inspected = observation(messages)
        assert inspected["sample"]["valid_count"] == 0
        return action("write_analysis", {"code": script}, "write")

    def run(messages):
        return action("run_analysis", {"code_id": observation(messages)["code_id"]}, "run")

    def read(messages):
        ran = observation(messages)
        assert ran["status"] == "completed" and ran["exit_code"] == 0
        return action("read_artifact", {"ref": ran["result_preview"][0]["artifact_id"]}, "read")

    def honest_answer(messages):
        quality = json.loads(observation(messages)["content"])["value"]
        assert quality == {"valid_count": 0, "mean": None}
        return {"role": "assistant", "content": "脚本运行成功，但 new_tracer 全部缺测，没有有效均值。"}

    model = ScriptedModel([
        action("inspect_data", {"source": str(source), "variable": "new_tracer"}, "inspect"),
        write, run, read, honest_answer,
    ])
    state = run_loop(model, session, "计算 new_tracer 的均值")
    assert state["status"] == "completed"
    assert "没有有效均值" in state["messages"][-1]["content"]
