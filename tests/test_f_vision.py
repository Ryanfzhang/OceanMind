"""A saved detector figure becomes real model pixels, then analysis can continue."""

from __future__ import annotations

import base64
import json
from io import BytesIO

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from domain.ocean.events.eddy.detect import detect_eddies
from packages.agent_loop.analysis import AnalysisSession
from packages.agent_loop.graph import build_graph
from packages.agent_loop.state import initial_state
from packages.agent_loop.vision import hydrate_vision_messages, make_view_image
from packages.analysis_runtime.figures import PngFigure, render_eddy_figure
from packages.analysis_runtime.code_store import CodeStore
from packages.analysis_runtime.executor import run_script
from packages.analysis_runtime.records import RunRecords


def _saved_eddy_plot(tmp_path):
    lon, lat = np.linspace(120, 121, 51), np.linspace(20, 21, 51)
    x, y = np.meshgrid(lon - 120.5, lat - 20.5)
    gaussian = np.exp(-(x * x + y * y) / 0.08**2)
    u = xr.DataArray(-y * gaussian * 0.01, coords={"lat": lat, "lon": lon},
                     dims=("lat", "lon"))
    v = xr.DataArray(x * gaussian * 0.01, coords={"lat": lat, "lon": lon},
                     dims=("lat", "lon"))
    result = detect_eddies(u, v, ow_threshold=-2e-12,
                           min_radius_km=2, min_pixels=4)
    assert result["statistics"]["total_count"] == 1
    assert int(np.count_nonzero(result["mask"])) > 1
    image = render_eddy_figure(result, date="2022-07-01", depth="50 m")

    session = AnalysisSession(tmp_path)
    attempt_id = session.records.new_attempt()
    stage_id = session.records.new_stage(attempt_id, "Plot eddies")
    artifact_id = session.artifacts.publish(
        "eddy_plot", image, run_id=session.run_id, attempt_id=attempt_id,
        stage_id=stage_id, inputs=[],
    )
    return session, artifact_id


def test_real_detector_figure_has_axes_context_and_owned_png(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    session, artifact_id = _saved_eddy_plot(tmp_path)
    metadata = session.artifacts.read_artifact(artifact_id)
    assert metadata["kind"] == "image_png"
    assert metadata["run_id"] == session.run_id
    context = metadata["summary"]["context"]
    assert context["date"] == "2022-07-01"
    assert context["depth"] == "50 m"
    assert context["extent"] == {"lon": [120.0, 121.0], "lat": [20.0, 21.0]}
    assert context["accepted_events"] == {"cyclonic": 1, "anticyclonic": 0}
    assert "candidates" in context["overlay"]
    with Image.open(BytesIO(session.artifacts.read_image(artifact_id))) as image:
        assert image.format == "PNG"
        assert image.width > 800 and image.height > 500


def test_view_image_hydrates_actual_pixels_without_persisting_them(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    session, artifact_id = _saved_eddy_plot(tmp_path)
    observation = make_view_image(session)(artifact_id)
    assert "png_bytes" not in json.dumps(observation)
    messages = [
        {"role": "user", "content": "Inspect the spatial pattern"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "view1", "type": "function", "function": {
                "name": "view_image", "arguments": json.dumps({"artifact_id": artifact_id})},
        }]},
        {"role": "tool", "tool_call_id": "view1", "content": json.dumps(observation)},
    ]
    wire_messages = hydrate_vision_messages(messages, session)
    assert len(messages) == 3  # The graph's durable messages remain references.
    assert len(wire_messages) == 4
    assert wire_messages[-1]["role"] == "user"
    text, image = wire_messages[-1]["content"]
    assert artifact_id in text["text"] and "2022-07-01" in text["text"]
    encoded = image["image_url"]["url"]
    assert encoded.startswith("data:image/png;base64,")
    assert base64.b64decode(encoded.partition(",")[2]) == session.artifacts.read_image(artifact_id)
    assert hydrate_vision_messages([messages[0]], session) == [messages[0]]


def test_view_image_rejects_other_run_and_tampered_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    session, artifact_id = _saved_eddy_plot(tmp_path)
    other = AnalysisSession(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        make_view_image(other)(artifact_id)
    metadata = session.artifacts.read_artifact(artifact_id)
    (session.root / metadata["payload"]).write_bytes(b"\x89PNG\r\n\x1a\ncorrupt")
    with pytest.raises(ValueError):
        make_view_image(session)(artifact_id)


def test_same_agent_views_known_pattern_then_calls_calculator(tmp_path):
    canvas = Image.new("RGB", (40, 40), "#204ac0")
    for x in range(20):
        for y in range(20):
            canvas.putpixel((x, y), (230, 30, 30))
    output = BytesIO()
    canvas.save(output, format="PNG")
    session = AnalysisSession(tmp_path)
    attempt_id = session.records.new_attempt()
    stage_id = session.records.new_stage(attempt_id, "Known pattern")
    artifact_id = session.artifacts.publish(
        "known_pattern", PngFigure(output.getvalue(), {
            "variable": "synthetic patch", "date": "2022-07-01",
            "extent": {"lon": [120, 121], "lat": [20, 21]},
            "color_scale": {"red": "high", "blue": "low"},
        }), run_id=session.run_id, attempt_id=attempt_id,
        stage_id=stage_id, inputs=[],
    )

    class Model:
        turns = 0
        observed_pixels = False

        def complete(self, messages, *, tools, timeout):
            self.turns += 1
            if self.turns == 1:
                assert all(not isinstance(item["content"], list) for item in messages)
                return {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "view", "type": "function", "function": {
                        "name": "view_image",
                        "arguments": json.dumps({"artifact_id": artifact_id}),
                    },
                }]}
            if self.turns == 2:
                image_part = messages[-1]["content"][1]["image_url"]["url"]
                with Image.open(BytesIO(base64.b64decode(image_part.partition(",")[2]))) as pixels:
                    self.observed_pixels = (pixels.getpixel((5, 5)) == (230, 30, 30)
                                            and pixels.getpixel((35, 35)) == (32, 74, 192))
                assert self.observed_pixels
                return {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "calculate", "type": "function", "function": {
                        "name": "calculator", "arguments": json.dumps({
                            "operation": "multiply", "a": 20, "b": 20,
                        }),
                    },
                }]}
            assert all(not isinstance(item["content"], list) for item in messages)
            assert json.loads(messages[-1]["content"])["result"] == 400
            return {"role": "assistant", "content": "High patch: northwest; 400 pixels."}

    model = Model()
    graph = build_graph(model, max_rounds=4, analysis_session=session)
    state = initial_state("Which quadrant is high, and how many pixels?",
                          run_id=session.run_id, run_root=str(session.root))
    done = graph.invoke(state, config={"recursion_limit": 12})
    assert done["status"] == "completed"
    assert model.turns == 3 and model.observed_pixels
    assert "northwest" in done["messages"][-1]["content"]
    assert all("data:image/png" not in json.dumps(item) for item in done["messages"])


def test_analysis_script_publishes_real_eddy_figure(tmp_path):
    """The figure path works through the same recorded child as ordinary code."""
    records = RunRecords(tmp_path)
    codes = CodeStore(tmp_path, records.run_id)
    code_id = codes.write_analysis('''
import numpy as np
import xarray as xr
from oceanmind_runtime import stage, tools
from packages.analysis_runtime.figures import render_eddy_figure
lon, lat = np.linspace(120, 121, 51), np.linspace(20, 21, 51)
x, y = np.meshgrid(lon - 120.5, lat - 20.5)
vortex = np.exp(-(x*x + y*y) / 0.08**2)
u = xr.DataArray(-y * vortex * 0.01, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
v = xr.DataArray(x * vortex * 0.01, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
with stage("Detect eddies"):
    eddies = tools.detect_eddies(u, v, ow_threshold=-2e-12, min_radius_km=2, min_pixels=4)
with stage("Plot accepted events"):
    figure = render_eddy_figure(eddies, date="2022-07-01", depth="50 m")
    tools.publish("eddy_figure", figure, inputs=[tools.ref(eddies)])
''')["code_id"]
    result = run_script(
        root=tmp_path, run_id=records.run_id, code_id=code_id,
        code_store=codes, launcher=lambda python, args, _root, _reads: [python, *args],
        timeout_seconds=90,
    )
    assert result["status"] == "completed", result.get("error") or result["stderr"]
    assert result["stage_count"] == 2 and result["result_count"] == 2
    image_ref = result["result_preview"][1]["artifact_id"]
    from packages.analysis_runtime.artifacts import ArtifactStore

    image_store = ArtifactStore(tmp_path)
    assert image_store.read_artifact(image_ref)["kind"] == "image_png"
    with Image.open(BytesIO(image_store.read_image(image_ref))) as image:
        assert image.width > 800
