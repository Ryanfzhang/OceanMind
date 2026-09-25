"""Session-owned pagination and artifact reading without result downloads."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.langgraph_query import QueryService, get_query_service
from apps.api.langgraph_results import router


def test_page_and_cross_conversation_boundary(tmp_path):
    service = QueryService(tmp_path, model_factory=lambda: None, data_roots=lambda: ())
    record, session = service.store.create()
    other, _ = service.store.create()
    code_id = session.write_analysis("print('ok')")['code_id']
    attempt_id = session.records.new_attempt(code_version=code_id)
    stage_id = session.records.new_stage(attempt_id, "batch")
    entries = []
    for number in range(1, 101):
        call_id = session.records.new_call(attempt_id, stage_id, "publish")
        artifact_id = session.artifacts.publish(
            "sample", {"slice": number}, run_id=session.run_id,
            attempt_id=attempt_id, stage_id=stage_id, inputs=[], call_id=call_id,
        )
        session.records.update("call", call_id, status="completed", artifact_ids=[artifact_id])
        entries.append({"stage_id": stage_id, "call_id": call_id, "artifact_id": artifact_id,
                        "time": f"2022-01-{number:03d}", "depth": number,
                        "status": "completed", "summary": "saved"})
    session.records.update("stage", stage_id, status="completed", result_index=entries)
    session.records.update("attempt", attempt_id, status="completed")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_query_service] = lambda: service
    client = TestClient(app)
    base = f"/results/{record.conversation_id}/attempts/{attempt_id}"
    page = client.get(base, params={"offset": 36, "limit": 1})
    assert page.status_code == 200
    assert page.json()["total"] == 100
    assert page.json()["results"][0]["depth"] == 37
    assert client.get(base + "/export").status_code == 404
    artifact = client.get(f"/results/{record.conversation_id}/artifacts/{entries[36]['artifact_id']}")
    assert artifact.status_code == 200
    assert '"slice": 37' in artifact.json()["content"]
    code = client.get(f"/results/{record.conversation_id}/code/{code_id}")
    assert code.status_code == 200
    assert code.text == "print('ok')"
    assert client.get(f"/results/{other.conversation_id}/attempts/{attempt_id}").status_code == 404
    assert client.get(f"/results/{other.conversation_id}/artifacts/{entries[0]['artifact_id']}").status_code == 404
    assert client.get(f"/results/{other.conversation_id}/code/{code_id}").status_code == 404
