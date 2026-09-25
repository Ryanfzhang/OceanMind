"""Manual C-stage example: load two fields and detect two explicit depths."""

from __future__ import annotations

from pathlib import Path

from domain.ocean.data_access.load import load_dataset
from domain.ocean.events.eddy.detect import detect_eddies
from packages.analysis_runtime.artifacts import ArtifactStore
from packages.analysis_runtime.records import RunRecords
from packages.analysis_runtime.stages import StageManager, stage
from packages.analysis_runtime.tools import AnalysisTools


def run_two_depths(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    date: str,
    depths: tuple[float, float],
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> tuple[RunRecords, ArtifactStore, StageManager]:
    """Run four recorded tool calls across two UI stages using real tool functions."""
    records = RunRecords(output_dir)
    attempt_id = records.new_attempt()
    artifacts = ArtifactStore(output_dir)
    stages = StageManager(records.run_id, attempt_id, records=records)
    tools = AnalysisTools(
        records, artifacts, stages,
        functions={"load_dataset": load_dataset, "detect_eddies": detect_eddies},
    )
    scope = {
        "lon_range": lon_range,
        "lat_range": lat_range,
        "time_range": (date, date),
        "depth_range": (min(depths), max(depths)),
        "data_path": str(data_dir),
    }
    with stages.activate():
        with stage("读取数据"):
            u = tools.load_dataset(variable="u", **scope)
            v = tools.load_dataset(variable="v", **scope)

        with stage("涡旋检测", total=2, unit="深度") as progress:
            for depth in depths:
                progress.set_current(time=date, depth=depth)
                tools.detect_eddies(
                    input_refs=[tools.ref(u), tools.ref(v)],
                    u=u.sel(time=date, depth=depth),
                    v=v.sel(time=date, depth=depth),
                )
                progress.advance()
    return records, artifacts, stages
