"""A notebook contains report conclusions and verified code from its own conversation."""

import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.langgraph_query import QueryService, get_query_service
from apps.api.langgraph_visualize import router
from packages.analysis_runtime.figures import PngFigure


def test_notebook_export_uses_conversation_owned_code_and_bounded_results(tmp_path):
    service = QueryService(tmp_path, model_factory=lambda: None, data_roots=lambda: ())
    record, session = service.store.create()
    other, other_session = service.store.create()
    other_code = other_session.write_analysis("print('other conversation')")["code_id"]
    secret = "sk-abcdefghijklmnopqrstuvwxyz12345"
    code_id = session.write_analysis(f"API_KEY = '{secret}'\nprint(42)")["code_id"]
    attempt_id = session.records.new_attempt(code_version=code_id)
    stage_id = session.records.new_stage(attempt_id, "Calculate")
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVQIHWP4z8DwHwAFgAI/ScL/nwAAAABJRU5ErkJggg=="
    )
    figure_id = session.artifacts.publish(
        "figure", PngFigure(png, {"description": "One pixel"}),
        run_id=session.run_id, attempt_id=attempt_id, stage_id=stage_id, inputs=[],
    )
    session.records.update("stage", stage_id, status="completed", result_index=[{
        "stage_id": stage_id, "call_id": "call_1", "artifact_id": "artifact_1",
        "status": "completed", "summary": "Mean temperature: 42", "depth": 0,
    }, {"stage_id": stage_id, "call_id": "call_2", "artifact_id": figure_id,
        "status": "completed", "summary": "One-pixel figure"}])
    session.records.update("attempt", attempt_id, status="completed")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_query_service] = lambda: service
    client = TestClient(app)
    response = client.post("/report/export", json={
        "conversation_id": record.conversation_id,
        "exported_at": "2022-01-03T00:00:00Z",
        "turns": [{"user_query": "Compute mean", "assistant_status": "completed",
                   "assistant_summary": "The mean is 42."}],
    })
    assert response.status_code == 200, response.text
    notebook = response.json()
    assert [cell["cell_type"] for cell in notebook["cells"]] == [
        "markdown", "markdown", "markdown", "code", "markdown",
    ]
    assert "The mean is 42." in notebook["cells"][1]["source"]
    assert "Mean temperature: 42" in notebook["cells"][2]["source"]
    code = notebook["cells"][3]
    assert code["metadata"]["oceanmind"] == {"code_id": code_id, "attempt_id": attempt_id}
    assert "print(42)" in code["source"]
    figure = notebook["cells"][4]
    assert "attachment:figure-1.png" in figure["source"]
    assert base64.b64decode(figure["attachments"]["figure-1.png"]["image/png"]) == png
    assert secret not in response.text
    assert other_code not in response.text
    assert "other conversation" not in response.text
    assert client.post("/report/export", json={
        "conversation_id": "conv_" + "0" * 32,
        "exported_at": "2022-01-03T00:00:00Z", "turns": [],
    }).status_code == 404
    assert other.conversation_id != record.conversation_id

    session.codes.get_path(code_id).write_text("modified", encoding="utf-8")
    assert client.post("/report/export", json={
        "conversation_id": record.conversation_id,
        "exported_at": "2022-01-03T00:00:00Z", "turns": [],
    }).status_code == 409
