"""Two-depth real tools and a 100-slice failure use the same runtime."""

import numpy as np
import pytest
import xarray as xr

from examples.c_eddy_two_depths import run_two_depths
from packages.analysis_runtime.artifacts import ArtifactStore
from packages.analysis_runtime.records import RunRecords
from packages.analysis_runtime.stages import StageManager
from packages.analysis_runtime.tools import AnalysisTools


def test_real_load_and_detect_at_two_depths(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    coords = {
        "time": np.array(["2022-07-01"], dtype="datetime64[D]"),
        "depth": [0.0, 50.0],
        "lat": np.linspace(18, 20, 5),
        "lon": np.linspace(110, 112, 5),
    }
    for name in ("u", "v"):
        field = xr.DataArray(
            np.zeros((1, 2, 5, 5)), dims=tuple(coords), coords=coords,
            name=name, attrs={"units": "m/s"},
        )
        field.to_dataset().to_netcdf(source / f"sample_{name}.nc")

    records, artifacts, stages = run_two_depths(
        source, tmp_path / "output", date="2022-07-01", depths=(0.0, 50.0),
        lon_range=(110, 112), lat_range=(18, 20),
    )
    assert [item.title for item in stages.stages] == ["读取数据", "涡旋检测"]
    assert [len(item.results) for item in stages.stages] == [2, 2]
    detections = stages.stages[1].results
    assert [item["depth"] for item in detections] == [0.0, 50.0]
    assert len({item["artifact_id"] for item in detections}) == 2
    for entry in detections:
        result = ArtifactStore(tmp_path / "output").load_result(entry["artifact_id"])
        assert result["event_type"] == "eddy"
        assert result["mask"].shape == (5, 5)
        assert records.read("call", entry["call_id"])["status"] == "completed"


def test_slice_37_failure_preserves_prior_results(tmp_path):
    def load(name):
        return {"name": name}

    def detect(index):
        if index == 37:
            raise ValueError("bad slice 37")
        return {"index": index}

    records = RunRecords(tmp_path)
    attempt_id = records.new_attempt()
    artifacts = ArtifactStore(tmp_path)
    stages = StageManager(records.run_id, attempt_id, records=records)
    tools = AnalysisTools(records, artifacts, stages, functions={"load": load, "detect": detect})

    with pytest.raises(ValueError, match="bad slice 37"):
        with stages.activate():
            with stages.stage("读取数据"):
                u, v = tools.load(name="u"), tools.load(name="v")
            with stages.stage("批量检测", total=100, unit="切片") as progress:
                for index in range(1, 101):
                    progress.set_current(time=f"2022-07-{index:03d}", depth=50)
                    tools.call("detect", input_refs=[tools.ref(u), tools.ref(v)], index=index)
                    progress.advance()

    assert len(stages.stages) == 2
    assert [len(item.results) for item in stages.stages] == [2, 37]
    assert progress.completed == 36 and progress.total == 100
    assert stages.stages[1].status == "failed"
    assert [item["status"] for item in stages.stages[1].results].count("completed") == 36
    assert stages.stages[1].results[-1]["status"] == "failed"
    first, last = stages.stages[1].results[0], stages.stages[1].results[-2]
    assert artifacts.load_result(first["artifact_id"])["index"] == 1
    assert ArtifactStore(tmp_path).load_result(last["artifact_id"])["index"] == 36
    assert records.read("call", stages.stages[1].results[-1]["call_id"])["status"] == "failed"
