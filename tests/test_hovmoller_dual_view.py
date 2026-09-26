"""The transect Hovmöller option saves two usable views from one computation."""

import numpy as np
import xarray as xr

from apps.api.langgraph_progress import ProgressAdapter
from domain.ocean.analysis.transports import (
    _with_hovmoller_climatology,
    compute_transect_normal_flux_hovmoller,
)
from domain.ocean.result_payload import ResultWithCompanions
from packages.agent_loop.analysis import AnalysisSession
from packages.analysis_runtime.stages import StageManager
from packages.analysis_runtime.tools import AnalysisTools


def test_transect_tool_saves_raw_and_monthly_climatology_as_two_cards(tmp_path):
    months = np.arange("2011-01", "2013-01", dtype="datetime64[M]").astype("datetime64[D]")
    shape = (len(months), 2, 2, 2)
    coords = {"time": months, "depth": [0., 10.], "lat": [0., 1.], "lon": [0., 1.]}
    u = xr.DataArray(np.zeros(shape), dims=("time", "depth", "lat", "lon"), coords=coords)
    v = xr.DataArray(
        np.broadcast_to(np.arange(len(months))[:, None, None, None] + 1., shape),
        dims=u.dims, coords=coords,
    )
    session = AnalysisSession(tmp_path)
    attempt = session.records.new_attempt()
    stages = StageManager(session.run_id, attempt, records=session.records)
    tools = AnalysisTools(session.records, session.artifacts, stages, functions={
        "compute_transect_normal_flux_hovmoller": compute_transect_normal_flux_hovmoller,
    })

    with stages.activate():
        with stages.stage("Time-depth transport"):
            result = tools.compute_transect_normal_flux_hovmoller(
                u, v, [[0.2, 0.5], [0.8, 0.5]], n_samples=3,
                include_climatology=True,
            )
    assert isinstance(result, ResultWithCompanions)
    artifact_ids = [entry["artifact_id"] for entry in stages.result_index]
    assert len(artifact_ids) == 2
    assert session.artifacts.load_result(artifact_ids[0])["values"].shape == (24, 2)
    climate = session.artifacts.load_result(artifact_ids[1])
    assert climate["values"].shape == (12, 2)
    assert climate["metadata"]["aggregation"] == "monthly_climatology"
    assert np.allclose(climate["values"][0],
                       (result["values"][0] + result["values"][12]) / 2)
    assert session.artifacts.read_artifact(artifact_ids[1])["inputs"] == [artifact_ids[0]]

    adapter = ProgressAdapter(session)
    for event in stages.events:
        adapter.on_event(event)
    assert [card["renderer"] for card in adapter.finalize()["result_cards"]] == [
        "hovmoller", "hovmoller",
    ]


def test_daily_climatology_has_365_calendar_days_and_keeps_raw_values():
    days = np.arange("2011-01-01", "2013-01-01", dtype="datetime64[D]")
    values = np.where(days < np.datetime64("2012-01-01"), 1., 3.)[:, None]
    original = {"time": [str(day) for day in days], "spatial_coord": [0.],
                "values": values, "metadata": {"units": "m^2 s^-1"}}
    result = _with_hovmoller_climatology(original)
    climate = result.companions["climatology"]

    assert len(result["time"]) == 731
    assert len(climate["time"]) == 365
    assert climate["time"][0] == "Jan 01"
    assert "Feb 29" not in climate["time"]
    assert climate["values"][0, 0] == 2.
    assert climate["metadata"]["source_time_steps"] == 731
    assert climate["metadata"]["aggregation"] == "daily_climatology"
    assert original["metadata"] == {"units": "m^2 s^-1"}
