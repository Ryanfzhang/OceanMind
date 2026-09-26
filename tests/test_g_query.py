"""Resumable API contract and live NDJSON delivery."""

import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.langgraph_query import QueryRequest, QueryService, get_query_service, router
from packages.agent_loop.conversations import ConversationStore


def test_run_workspace_can_use_another_volume(tmp_path, monkeypatch):
    workspace = tmp_path / "runs"
    monkeypatch.setenv("OCEANMIND_RUNS_DIR", str(workspace))
    get_query_service.cache_clear()
    try:
        assert get_query_service().store.workspace == workspace
    finally:
        get_query_service.cache_clear()


def _clarify(question: str):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": "clarify_1", "type": "function",
        "function": {"name": "request_clarification", "arguments": json.dumps({"question": question})},
    }]}


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.seen = []

    def complete(self, messages, *, tools, timeout):
        self.seen.append(messages)
        return next(self.replies)


def test_clarification_resumes_same_run_and_keeps_other_session_out(tmp_path):
    model = ScriptedModel([
        _clarify("Which year?"),
        {"role": "assistant", "content": "Use 2022."},
        {"role": "assistant", "content": "A separate answer."},
    ])
    service = QueryService(tmp_path, model_factory=lambda: model, data_roots=lambda: ())
    first = service.execute(QueryRequest(query="Analyze chlorophyll", conversation_id="conversation-browser"))
    assert first["status"] == "clarification_needed"
    assert first["clarification_question"] == "Which year?"
    assert first["conversation_id"].startswith("conv_")

    first_record = service.store.resolve(first["conversation_id"])
    code_id = service.store.open_analysis(first_record).write_analysis("print(2022)")["code_id"]
    resumed = service.execute(QueryRequest(
        query="2022", conversation_id=first["conversation_id"], continue_pending=True,
    ))
    assert resumed["status"] == "completed"
    assert resumed["conversation_id"] == first["conversation_id"]
    assert len(model.seen[1]) > len(model.seen[0])
    assert any(item.get("content") == "Analyze chlorophyll" for item in model.seen[1])

    separate = service.execute(QueryRequest(query="Hello", conversation_id="conversation-other"))
    assert separate["conversation_id"] != first["conversation_id"]
    assert not any(item.get("content") == "Analyze chlorophyll" for item in model.seen[2])
    second_record = service.store.resolve(separate["conversation_id"])
    assert second_record.run_root != first_record.run_root
    with pytest.raises((FileNotFoundError, ValueError)):
        service.store.open_analysis(second_record).read_artifact(code_id)

    # A new process can recover the same validated run and saved history.
    restored = ConversationStore(tmp_path).resolve(first["conversation_id"])
    assert restored.run_id == first_record.run_id
    assert restored.messages[-1]["content"] == "Use 2022."


def test_backend_exception_uses_model_authored_failure_response(tmp_path, monkeypatch):
    def broken_graph(*_args, **_kwargs):
        raise RuntimeError("optional analysis setup failed")

    monkeypatch.setattr("apps.api.langgraph_query.build_graph", broken_graph)
    explanation = (
        "I could not start the calculation because analysis setup failed before "
        "any data was read. Please retry after the service is repaired."
    )
    model = ScriptedModel([{"role": "assistant", "content": explanation}])
    service = QueryService(tmp_path, model_factory=lambda: model, data_roots=lambda: ())
    result = service.execute(QueryRequest(query="Compute transport across the selected line"))
    assert result["status"] == "failed"
    assert result["failure_explained"] is True
    assert result["synthesis"]["summary"] == explanation
    assert "optional analysis setup failed" in model.seen[0][-1]["content"]


def test_ndjson_starts_before_model_completes(tmp_path):
    release = threading.Event()

    class SlowModel:
        def complete(self, messages, *, tools, timeout):
            assert release.wait(5)
            return {"role": "assistant", "content": "The sea is blue."}

    service = QueryService(tmp_path, model_factory=SlowModel, data_roots=lambda: ())
    stream = service.stream(QueryRequest(query="Say one sentence"))
    first = json.loads(next(stream))
    assert first == {"event": "execution_event", "payload": {"type": "planning_started"}}
    release.set()
    events = [json.loads(line) for line in stream]
    assert events[-1]["event"] == "final"
    assert events[-1]["payload"]["status"] == "completed"
    assert events[-1]["payload"]["synthesis"]["summary"] == "The sea is blue."


def test_runtime_event_reaches_stream_before_final(tmp_path):
    class Adapter:
        def __init__(self, session, emit):
            self.emit = emit

        def on_event(self, event):
            self.emit({"event": "execution_event", "payload": event})

        def finalize(self, state):
            return {"step_cards": [{"step_id": "stage_one", "status": "completed"}]}

    class EventModel:
        def complete(self, messages, *, tools, timeout):
            session_events[0].on_event({"type": "step_started", "step_id": "stage_one"})
            return {"role": "assistant", "content": "Done."}

    session_events = []

    def progress_factory(session, emit):
        adapter = Adapter(session, emit)
        session_events.append(adapter)
        return adapter

    service = QueryService(
        tmp_path, model_factory=EventModel, data_roots=lambda: (),
        progress_factory=progress_factory,
    )
    events = [json.loads(line) for line in service.stream(QueryRequest(query="Run"))]
    assert [event["event"] for event in events] == [
        "execution_event", "execution_event", "final",
    ]
    assert events[1]["payload"]["type"] == "step_started"
    assert events[-1]["payload"]["step_cards"][0]["step_id"] == "stage_one"


def test_unknown_server_conversation_cannot_be_continued(tmp_path):
    service = QueryService(tmp_path, model_factory=lambda: None, data_roots=lambda: ())
    with pytest.raises(ValueError, match="Unknown conversation"):
        service.execute(QueryRequest(query="Continue", conversation_id="conv_" + "a" * 32))


def test_query_routes_use_existing_frontend_envelope(tmp_path):
    model = ScriptedModel([
        {"role": "assistant", "content": "Hello from LangGraph."},
        {"role": "assistant", "content": "Hello again."},
    ])
    service = QueryService(tmp_path, model_factory=lambda: model, data_roots=lambda: ())
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_query_service] = lambda: service
    client = TestClient(app)
    response = client.post("/query", json={"query": "Hello"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["synthesis"]["summary"] == "Hello from LangGraph."
    assert payload["step_cards"] == []

    response = client.post("/query/stream", json={
        "query": "Again", "conversation_id": payload["conversation_id"],
    })
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["event"] == "final"
    assert events[-1]["payload"]["conversation_id"] == payload["conversation_id"]


def test_main_app_exposes_query_and_result_routes(tmp_path):
    from apps.api.langgraph_app import app

    model = ScriptedModel([{"role": "assistant", "content": "Ready."}])
    service = QueryService(tmp_path, model_factory=lambda: model, data_roots=lambda: ())
    app.dependency_overrides[get_query_service] = lambda: service
    try:
        client = TestClient(app)
        assert client.get("/health").json() == {"status": "ok"}
        answer = client.post("/query", json={"query": "Hello"})
        assert answer.status_code == 200
        conversation = answer.json()["conversation_id"]
        attempts = client.get(f"/results/{conversation}/attempts")
        assert attempts.status_code == 200
        assert attempts.json()["attempts"] == []
    finally:
        app.dependency_overrides.clear()
