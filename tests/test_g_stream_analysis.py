"""The query stream carries live bounded cards for a many-result script."""

import json

from apps.api.langgraph_progress import ProgressAdapter
from apps.api.langgraph_query import QueryRequest, QueryService


def _call(name, args, call_id):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": name, "arguments": json.dumps(args),
        },
    }]}


def test_stream_keeps_two_stage_cards_for_one_hundred_results(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "packages.agent_loop.analysis.build_sandbox_command",
        lambda python, args, root, read_roots: [python, *args],
    )
    script = '''from oceanmind_runtime import stage, publish
with stage("Read data"):
    publish("source_summary", {"count": 100}, inputs=[])
with stage("Batch", total=100, unit="slice") as progress:
    for index in range(100):
        progress.set_current(depth=index)
        publish("slice", {"index": index}, inputs=[])
        progress.advance()
'''

    class Model:
        turn = 0

        def complete(self, messages, *, tools, timeout):
            self.turn += 1
            if self.turn == 1:
                return _call("write_analysis", {"code": script}, "write")
            if self.turn == 2:
                code_id = json.loads(messages[-1]["content"])["code_id"]
                return _call("run_analysis", {"code_id": code_id}, "run")
            assert json.loads(messages[-1]["content"])["result_count"] == 101
            return {"role": "assistant", "content": "Saved all 100 slice results."}

    service = QueryService(
        tmp_path, model_factory=Model, data_roots=lambda: (),
        progress_factory=ProgressAdapter, timeout_seconds=120,
    )
    events = [json.loads(line) for line in service.stream(QueryRequest(query="Compute batch"))]
    final = events[-1]
    assert final["event"] == "final" and final["payload"]["status"] == "completed"
    cards = final["payload"]["step_cards"]
    assert len(cards) == 2
    assert cards[1]["progress"]["completed_units"] == 100
    assert final["payload"]["result_summaries"][cards[1]["step_id"]]["completed"] == 100
    assert len(cards[1]["results"]) == 100
    assert [event["payload"]["type"] for event in events[:-1]].count("step_started") == 2
    assert [event["payload"]["type"] for event in events[:-1]].count("step_result_attached") == 101
