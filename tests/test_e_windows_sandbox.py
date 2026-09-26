"""Native Windows acceptance check; runs a real AppContainer, not a mock."""

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
