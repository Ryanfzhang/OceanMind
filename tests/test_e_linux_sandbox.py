"""A real Linux Bubblewrap run must retain data access, events and results."""

import sys

import pytest

from packages.agent_loop.analysis import AnalysisSession


@pytest.mark.skipif(sys.platform != "linux", reason="Linux Bubblewrap runtime")
def test_linux_worker_runs_with_one_authorized_source(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("42", encoding="utf-8")
    session = AnalysisSession(tmp_path / "work", data_roots=[source])
    code_id = session.write_analysis(f'''from oceanmind_runtime import stage, publish
with stage("Linux smoke"):
    value = int(open({str(source)!r}, encoding="utf-8").read())
    publish("answer", {{"value": value}}, inputs=[])
''')["code_id"]

    result = session.run_analysis(code_id, timeout_seconds=30)
    assert result["status"] == "completed", result.get("error") or result["stderr"]
    assert result["event_count"] > 0
    assert result["result_count"] == 1
    content = session.read_artifact(result["result_preview"][0]["artifact_id"])["content"]
    assert '"value": 42' in content
