"""Run the Python example in the eddy skill against the actual stage/tool wrapper."""

import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from domain.ocean.events.eddy.detect import detect_eddies
from packages.analysis_runtime.artifacts import ArtifactStore
from packages.analysis_runtime.records import RunRecords
from packages.analysis_runtime.stages import StageManager
from packages.analysis_runtime.tools import AnalysisTools


SKILL = Path(__file__).resolve().parents[1] / "skills/ocean_eddy_detection/SKILL.md"


def _example():
    source = SKILL.read_text()
    code = re.search(r"```python\n(.*?)\n```", source, re.DOTALL)
    assert code is not None
    namespace = {}
    exec(compile(code.group(1), str(SKILL), "exec"), namespace)
    return namespace["run_eddy_detection"]


def _runtime(tmp_path):
    coords = {
        "time": ["2022-01-01"],
        "depth": [0.0, 50.0],
        "lat": np.linspace(18, 20, 5),
        "lon": np.linspace(110, 112, 5),
    }
    field = xr.DataArray(np.zeros((1, 2, 5, 5)), dims=tuple(coords), coords=coords)
    calls = []

    def load_dataset(variable, **scope):
        assert variable in {"east_current", "north_current"}
        assert scope["time_range"] == ("2022-01-01", "2022-01-01")
        return field.copy()

    def recorded_detection(**kwargs):
        calls.append(kwargs)
        return detect_eddies(**kwargs)

    records = RunRecords(tmp_path)
    attempt = records.new_attempt()
    artifacts = ArtifactStore(tmp_path)
    stages = StageManager(records.run_id, attempt, records=records)
    tools = AnalysisTools(
        records, artifacts, stages,
        functions={"load_dataset": load_dataset, "detect_eddies": recorded_detection},
    )
    return tools, stages, calls


def test_two_depths_share_one_stage_and_keep_two_results(tmp_path):
    run = _example()
    tools, stages, calls = _runtime(tmp_path)
    with stages.activate():
        summary = run(
            tools, u_name="east_current", v_name="north_current",
            scope={"lon_range": (110, 112), "lat_range": (18, 20),
                   "time_range": ("2022-01-01", "2022-01-01")},
            slices=[("2022-01-01", 0.0), ("2022-01-01", 50.0)],
        )

    assert len(stages.stages) == 2
    assert [item.title for item in stages.stages] == ["读取流速", "涡旋检测"]
    assert stages.stages[1].completed == 2
    assert [item["depth"] for item in stages.stages[1].results] == [0.0, 50.0]
    assert all(item["artifact_id"] for item in stages.stages[1].results)
    assert len(calls) == 2
    assert all(set(call["u"].dims) == {"lat", "lon"} for call in calls)
    assert [item["depth"] for item in summary] == [0.0, 50.0]


def test_unselected_extra_dimensions_fail_instead_of_using_first_slice(tmp_path):
    run = _example()
    tools, stages, calls = _runtime(tmp_path)
    with pytest.raises(ValueError, match="Select one time and depth"):
        with stages.activate():
            run(tools, u_name="east_current", v_name="north_current",
                scope={"time_range": ("2022-01-01", "2022-01-01")},
                slices=[(None, None)])
    assert calls == []
    assert stages.stages[1].status == "failed"


def test_skill_keeps_algorithm_semantics_and_has_no_old_reference_syntax():
    source = SKILL.read_text()
    for detail in ("Okubo–Weiss", "ow_threshold=-2e-12", "min_radius_km=30",
                   "max_radius_km=300", "min_pixels=10", "inspect_data",
                   "find_tools", "mask", "events"):
        assert detail in source
    assert "$ref" not in source
    assert "u_field.data" not in source
