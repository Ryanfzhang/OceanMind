"""Native Windows acceptance checks for AppContainer and default execution."""

import base64
import sys

import pytest

from packages.agent_loop.analysis import AnalysisSession
from packages.analysis_runtime.artifacts import ArtifactStore


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows AppContainer test")
def test_native_windows_analysis_is_isolated_and_keeps_state(tmp_path, monkeypatch):
    monkeypatch.setenv("OCEANMIND_WINDOWS_ANALYSIS_MODE", "appcontainer")
    session = AnalysisSession(tmp_path / "runs")
    first = session.write_analysis("saved = 7\n")["code_id"]
    second = session.write_analysis('''from oceanmind_runtime import stage, publish
with stage("Use saved value"):
    publish("result", {"value": saved + 1}, inputs=[])
''')["code_id"]
    try:
        initial = session.run_analysis(first, timeout_seconds=60)
        final = session.run_analysis(second, timeout_seconds=60)
        assert initial["status"] == "completed", initial
        assert final["status"] == "completed", final
        result = final["result_preview"][0]["artifact_id"]
        assert ArtifactStore(session.root).load_result(result) == {"value": 8}
    finally:
        session.close()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows analysis test")
def test_native_windows_default_runs_without_appcontainer(tmp_path, monkeypatch):
    from packages.analysis_runtime import windows_appcontainer

    monkeypatch.delenv("OCEANMIND_WINDOWS_ANALYSIS_MODE", raising=False)

    def reject_appcontainer(*_args, **_kwargs):
        raise AssertionError("Default Windows analysis must not start AppContainer")

    monkeypatch.setattr(windows_appcontainer, "WindowsAppContainer", reject_appcontainer)
    source = tmp_path / "input.txt"
    source.write_text("from outside the task", encoding="utf-8")
    session = AnalysisSession(tmp_path / "runs", data_roots=(source,))
    code_id = session.write_analysis(
        "from pathlib import Path\n"
        "from oceanmind_runtime import publish\n"
        f'publish("check", {{"value": Path({str(source)!r}).read_text()}}, inputs=[])\n'
    )["code_id"]
    try:
        result = session.run_analysis(code_id, timeout_seconds=60)
        assert result["status"] == "completed", result
        artifact_id = result["result_preview"][0]["artifact_id"]
        assert ArtifactStore(session.root).load_result(artifact_id) == {
            "value": "from outside the task"}
    finally:
        session.close()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows binary artifact test")
def test_native_windows_saves_png_array_and_dataarray(tmp_path, monkeypatch):
    monkeypatch.delenv("OCEANMIND_WINDOWS_ANALYSIS_MODE", raising=False)
    session = AnalysisSession(tmp_path / "runs")
    code_id = session.write_analysis('''import base64
import numpy as np
import xarray as xr
from oceanmind_runtime import stage, publish
from packages.analysis_runtime.figures import PngFigure

png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg=="
)
with stage("Save binary results"):
    publish("figure", PngFigure(png, {}), inputs=[])
    publish("large array", {"values": np.arange(30_000)}, inputs=[])
    publish("field", xr.DataArray([[1.0, 2.0]], dims=("lat", "lon")), inputs=[])
''')["code_id"]
    try:
        result = session.run_analysis(code_id, timeout_seconds=60)
        assert result["status"] == "completed", result
        assert result["result_count"] == 3
        saved = {session.artifacts.read_artifact(item["artifact_id"])["name"]: item["artifact_id"]
                 for item in result["result_preview"]}
        assert session.artifacts.read_image(saved["figure"]) == base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
            "AAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg==")
        assert len(ArtifactStore(session.root).load_result(saved["large array"])["values"]) == 30_000
        assert ArtifactStore(session.root).load_result(saved["field"]).shape == (1, 2)
    finally:
        session.close()
