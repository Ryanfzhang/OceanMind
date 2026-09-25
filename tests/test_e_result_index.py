"""A late failure leaves early results discoverable without flooding the model."""

import json

import pytest

from packages.analysis_runtime.artifacts import ArtifactStore
from packages.analysis_runtime.index import list_results
from packages.analysis_runtime.records import RunRecords
from packages.analysis_runtime.stages import StageManager, stage
from packages.analysis_runtime.tools import AnalysisTools


def test_page_early_result_after_late_failure(tmp_path):
    records = RunRecords(tmp_path)
    attempt = records.new_attempt()
    stages = StageManager(records.run_id, attempt, records=records)
    artifacts = ArtifactStore(tmp_path)

    def calculate(number):
        if number == 101:
            raise ValueError("bad final slice")
        return {"number": number}

    tools = AnalysisTools(records, artifacts, stages, functions={"calculate": calculate})
    with pytest.raises(ValueError, match="bad final slice"):
        with stages.activate():
            with stage("Many slices", total=101) as progress:
                for number in range(1, 102):
                    progress.set_current(time=number)
                    tools.calculate(number)
                    progress.advance()

    first = list_results(tmp_path, records.run_id, attempt, limit=1)
    last = list_results(tmp_path, records.run_id, attempt, offset=100, limit=1)
    assert first["total"] == 101
    assert first["next_offset"] == 1
    assert len(json.dumps(first)) < 1500
    assert first["results"][0]["time"] == 1
    assert artifacts.load_result(first["results"][0]["artifact_id"]) == {"number": 1}
    assert last["results"][0]["status"] == "failed"
    assert last["results"][0]["artifact_id"] is None

    other = RunRecords(tmp_path)
    with pytest.raises(ValueError, match="another run"):
        list_results(tmp_path, other.run_id, attempt)
