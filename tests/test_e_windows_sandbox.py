"""Native Windows acceptance check; runs a real AppContainer, not a mock."""

import sys

import pytest

from packages.agent_loop.analysis import AnalysisSession
from packages.analysis_runtime.artifacts import ArtifactStore


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows AppContainer test")
def test_native_windows_analysis_is_isolated_and_keeps_state(tmp_path):
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
