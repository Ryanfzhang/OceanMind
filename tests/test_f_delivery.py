"""Final delivery validates cited refs and exposes verified saved artifacts."""

from packages.agent_loop.analysis import AnalysisSession
from packages.agent_loop.graph import build_graph
from packages.agent_loop.state import initial_state


class _Replies:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.seen = []

    def complete(self, messages, *, tools, timeout):
        self.seen.append(messages)
        return {"role": "assistant", "content": next(self.replies)}


def test_invalid_attachment_is_returned_to_agent_and_repaired(tmp_path):
    session = AnalysisSession(tmp_path)
    wrong = "code_" + "0" * 32
    model = _Replies([f"Done: {wrong}", "The result is still being prepared."])
    state = build_graph(model, max_rounds=3, analysis_session=session).invoke(
        initial_state("Analyze it", run_id=session.run_id, run_root=str(session.root)))
    assert state["status"] == "completed"
    assert len(model.seen) == 2
    assert "Delivery check failed" in model.seen[-1][-1]["content"]
    assert state["attachments"] == []


def test_latest_successful_code_and_result_are_attached(tmp_path):
    session = AnalysisSession(tmp_path)
    code_id = session.write_analysis("print(2)")["code_id"]
    attempt_id = session.records.new_attempt(code_version=code_id)
    stage_id = session.records.new_stage(attempt_id, "Compute")
    call_id = session.records.new_call(attempt_id, stage_id, "publish")
    artifact_id = session.artifacts.publish(
        "answer", {"value": 2}, run_id=session.run_id,
        attempt_id=attempt_id, stage_id=stage_id, inputs=[], call_id=call_id)
    session.records.update("call", call_id, status="completed", artifact_ids=[artifact_id])
    session.records.update("stage", stage_id, status="completed")
    session.records.update("attempt", attempt_id, status="completed")
    model = _Replies([f"Computed 2; code {code_id}; result {artifact_id}."])
    state = build_graph(model, analysis_session=session).invoke(
        initial_state("Compute", run_id=session.run_id, run_root=str(session.root)))
    assert state["status"] == "completed"
    assert state["attachments"] == [
        {"kind": "code", "ref": code_id},
        {"kind": "json", "ref": artifact_id},
    ]
