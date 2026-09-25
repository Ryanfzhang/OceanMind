"""One complete analysis attempt under the production macOS launcher."""

import sys

import numpy as np
import pytest
import xarray as xr

from packages.agent_loop.analysis import AnalysisSession


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt runtime")
def test_saved_code_runs_in_verified_sandbox(tmp_path):
    session = AnalysisSession(tmp_path / "work")
    code_id = session.write_analysis('''from oceanmind_runtime import stage, publish
with stage("计算"):
    publish("quality", {"valid_count": 2, "mean": 3.0}, inputs=[])
''')["code_id"]
    result = session.run_analysis(code_id, timeout_seconds=30)
    assert result["status"] == "completed", result.get("error")
    assert result["stage_count"] == result["result_count"] == 1
    assert result["log_ref"].startswith("log_")
    content = session.read_artifact(result["result_preview"][0]["artifact_id"])["content"]
    assert '"valid_count": 2' in content


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Seatbelt runtime")
def test_unknown_variable_can_be_computed_under_verified_sandbox(tmp_path):
    source = tmp_path / "unregistered_file.nc"
    xr.Dataset({"new_tracer": ("sample", np.array([1.0, 2.0, 3.0]))}).to_netcdf(source)
    session = AnalysisSession(tmp_path / "work", data_roots=[source])
    code_id = session.write_analysis(f'''import xarray as xr
from oceanmind_runtime import stage, publish
with stage("未知变量统计"):
    with xr.open_dataset({str(source)!r}) as data:
        mean = float(data["new_tracer"].mean().item())
    publish("new_tracer_mean", {{"mean": mean}}, inputs=[])
''')["code_id"]
    result = session.run_analysis(code_id, timeout_seconds=30)
    assert result["status"] == "completed", result.get("error") or result["stderr"]
    content = session.read_artifact(result["result_preview"][0]["artifact_id"])["content"]
    assert '"mean": 2.0' in content
