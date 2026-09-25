"""Successive analysis cells reuse their sandboxed Python environment."""

from packages.agent_loop.analysis import AnalysisSession
from packages.analysis_runtime.artifacts import ArtifactStore


def test_later_cell_uses_prior_values_without_repeating_completed_work(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "packages.agent_loop.analysis.build_sandbox_command",
        lambda python, args, root, read_roots: [python, *args],
    )
    session = AnalysisSession(tmp_path / "runs")
    marker = session.root / "load_count.txt"
    first_code = session.write_analysis(f'''from pathlib import Path
from oceanmind_runtime import stage, publish
with stage("Load data"):
    marker = Path({str(marker)!r})
    marker.write_text(str(int(marker.read_text()) + 1 if marker.exists() else 1))
    mask = {{"coverage": 4}}
    publish("mask", mask, inputs=[])
''')["code_id"]
    second_code = session.write_analysis('''from oceanmind_runtime import stage, publish
with stage("Compute coverage"):
    publish("coverage", {"cells": mask["coverage"] * 2}, inputs=[])
''', previous_version=first_code)["code_id"]
    try:
        first = session.run_analysis(first_code, timeout_seconds=20)
        second = session.run_analysis(second_code, timeout_seconds=20)
        assert first["status"] == second["status"] == "completed"
        assert [item["title"] for item in second["stages"]] == ["Compute coverage"]
        assert marker.read_text() == "1"
        result = second["result_preview"][0]["artifact_id"]
        assert ArtifactStore(session.root).load_result(result) == {"cells": 8}
    finally:
        session.close()


def test_python_error_keeps_prior_values_for_repair(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "packages.agent_loop.analysis.build_sandbox_command",
        lambda python, args, root, read_roots: [python, *args],
    )
    session = AnalysisSession(tmp_path / "runs")
    first = session.write_analysis("saved_value = 7\n")["code_id"]
    broken = session.write_analysis('raise ValueError("bad formula")\n')["code_id"]
    repaired = session.write_analysis('''from oceanmind_runtime import publish
publish("repaired", {"value": saved_value + 1}, inputs=[])
''')["code_id"]
    try:
        assert session.run_analysis(first, timeout_seconds=20)["status"] == "completed"
        failed = session.run_analysis(broken, timeout_seconds=20)
        assert failed["status"] == "failed"
        assert "bad formula" in failed["stderr"]
        result = session.run_analysis(repaired, timeout_seconds=20)
        assert result["status"] == "completed"
        assert ArtifactStore(session.root).load_result(
            result["result_preview"][0]["artifact_id"]
        ) == {"value": 8}
    finally:
        session.close()


def test_timeout_restarts_sandbox_and_saved_result_can_be_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "packages.agent_loop.analysis.build_sandbox_command",
        lambda python, args, root, read_roots: [python, *args],
    )
    session = AnalysisSession(tmp_path / "runs")
    first = session.write_analysis('''from oceanmind_runtime import publish
publish("saved", {"count": 3}, inputs=[])
''')["code_id"]
    timeout = session.write_analysis("import time\ntime.sleep(20)\n")["code_id"]
    try:
        saved = session.run_analysis(first, timeout_seconds=20)
        artifact_id = saved["result_preview"][0]["artifact_id"]
        interrupted = session.run_analysis(timeout, timeout_seconds=1)
        assert interrupted["status"] == "timed_out"
        assert interrupted["environment_lost"]
        recovery = session.write_analysis(f'''from oceanmind_runtime import load_result, publish
value = load_result({artifact_id!r})
publish("recovered", {{"count": value["count"] + 1}}, inputs=[{artifact_id!r}])
''')["code_id"]
        result = session.run_analysis(recovery, timeout_seconds=20)
        assert result["status"] == "completed"
        assert ArtifactStore(session.root).load_result(
            result["result_preview"][0]["artifact_id"]
        ) == {"count": 4}
    finally:
        session.close()
