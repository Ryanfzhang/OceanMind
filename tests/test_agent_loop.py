"""Offline checks for the model → tool → model path."""

import json
import time

from packages.agent_loop.graph import build_graph
from packages.agent_loop.run import run_query
from packages.agent_loop.state import initial_state


def call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen = []
        self.tools_seen = []

    def complete(self, messages, *, tools, timeout):
        self.seen.append(messages)
        self.tools_seen.append(tools)
        response = next(self.responses)
        return response(messages) if callable(response) else response


def test_dependent_calculations_return_to_one_agent():
    def second(messages):
        first_result = json.loads(messages[-1]["content"])["result"]
        return {"role": "assistant", "content": None, "tool_calls": [
            call("calculator", {"operation": "multiply", "a": first_result, "b": 4}, "call_2")
        ]}

    model = ScriptedModel([
        {"role": "assistant", "content": None, "tool_calls": [
            call("calculator", {"operation": "add", "a": 2, "b": 3})
        ]},
        second,
        {"role": "assistant", "content": "The result is 20."},
    ])
    state = run_query("Compute (2+3)*4", model=model)
    assert state["status"] == "completed"
    assert state["rounds"] == 3
    assert [message["role"] for message in state["messages"]] == [
        "user", "assistant", "tool", "assistant", "tool", "assistant"
    ]
    assert state["messages"][4]["tool_call_id"] == "call_2"
    assert json.loads(state["messages"][4]["content"])["result"] == 20


def test_tool_error_is_observed_and_repaired():
    def repair(messages):
        assert json.loads(messages[-1]["content"])["error"]["type"] == "invalid_arguments"
        return {"role": "assistant", "content": None, "tool_calls": [
            call("calculator", {"operation": "add", "a": 2, "b": 3}, "fixed")
        ]}

    model = ScriptedModel([
        {"role": "assistant", "content": None, "tool_calls": [
            call("calculator", {"operation": "add", "a": "bad", "b": 3})
        ]},
        repair,
        {"role": "assistant", "content": "The sum is 5."},
    ])
    state = build_graph(model).invoke(initial_state("Add the values"))
    assert state["status"] == "completed"
    assert json.loads(state["messages"][4]["content"])["result"] == 5


def test_search_source_reaches_final_answer():
    def answer(messages):
        source = json.loads(messages[-1]["content"])["results"][0]["url"]
        return {"role": "assistant", "content": f"Found evidence: {source}"}

    model = ScriptedModel([
        {"role": "assistant", "content": None, "tool_calls": [
            call("web_search", {"query": "ocean temperature"})
        ]}, answer,
    ])
    fake_search = lambda **kwargs: {"query": kwargs["query"], "provider": "exa", "results": [
        {"title": "Source", "url": "https://example.org/source", "snippet": "Evidence"}
    ]}
    state = build_graph(model, web_search=fake_search).invoke(initial_state("Search the web"))
    assert state["status"] == "completed"
    assert "https://example.org/source" in state["messages"][-1]["content"]


def test_answer_agent_can_return_a_conflict_for_verification():
    executor = ScriptedModel([
        {"role": "assistant", "content": "Draft says 34 days."},
        {"role": "assistant", "content": "Mask verifies 10 days."},
    ])
    answer = ScriptedModel([
        {"role": "assistant", "content": None, "tool_calls": [
            call("request_verification", {"issue": "Check duration against the saved mask."})
        ]},
        {"role": "assistant", "content": "The event lasted 10 days."},
    ])
    state = build_graph(executor, answer_model=answer).invoke(
        initial_state("How long did the event last?")
    )
    assert state["status"] == "completed"
    assert state["messages"][-1]["content"] == "The event lasted 10 days."
    assert state["rounds"] == 4
    assert "verification_needed" in executor.seen[1][-1]["content"]
    assert "run_analysis" not in {item["function"]["name"] for item in answer.tools_seen[0]}
    assert "request_verification" in {item["function"]["name"] for item in answer.tools_seen[0]}


def test_clarification_and_limits_have_distinct_statuses():
    clarification = {"role": "assistant", "content": None, "tool_calls": [
        call("request_clarification", {"question": "Which dataset?"})
    ]}
    state = build_graph(ScriptedModel([clarification])).invoke(initial_state("Analyze it"))
    assert state["status"] == "needs_input"
    assert json.loads(state["messages"][-1]["content"])["question"] == "Which dataset?"
    assert state["messages"][-1]["tool_call_id"] == "call_1"

    expired = build_graph(ScriptedModel([])).invoke(
        initial_state("Analyze it", deadline=time.monotonic() - 1)
    )
    assert (expired["status"], expired["termination_reason"]) == (
        "completed", "deadline_exceeded"
    )
    assert "could not complete" in expired["messages"][-1]["content"]

    one_round = build_graph(ScriptedModel([clarification]), max_rounds=1).invoke(
        initial_state("Analyze it")
    )
    assert one_round["status"] == "needs_input"

    tool_call = {"role": "assistant", "content": None, "tool_calls": [
        call("calculator", {"operation": "add", "a": 1, "b": 2})
    ]}
    capped = build_graph(ScriptedModel([tool_call]), max_rounds=1).invoke(initial_state("Add"))
    assert (capped["status"], capped["termination_reason"]) == (
        "completed", "max_rounds_exceeded"
    )


def test_repeated_identical_tool_failure_stays_recoverable():
    bad_call = {"role": "assistant", "content": None, "tool_calls": [
        call("calculator", {"operation": "add", "a": "bad", "b": 3})
    ]}
    repaired = {"role": "assistant", "content": None, "tool_calls": [
        call("calculator", {"operation": "add", "a": 2, "b": 3}, "fixed")
    ]}
    state = build_graph(ScriptedModel([
        bad_call, bad_call, repaired, {"role": "assistant", "content": "The sum is 5."},
    ])).invoke(initial_state("Add"))
    assert (state["status"], state["rounds"]) == ("completed", 4)
    assert json.loads(state["messages"][-2]["content"])["result"] == 5


def test_transient_model_error_retries_agent_without_replaying_tool():
    class FlakyModel:
        calls = 0

        def complete(self, messages, *, tools, timeout):
            self.calls += 1
            if self.calls == 1:
                return {"role": "assistant", "content": None, "tool_calls": [
                    call("calculator", {"operation": "add", "a": 2, "b": 3})]}
            if self.calls == 2:
                raise ConnectionError("temporary model connection")
            assert json.loads(messages[-1]["content"])["result"] == 5
            return {"role": "assistant", "content": "The sum is 5."}

    model = FlakyModel()
    tool_calls = []
    state = build_graph(model, tool_registry={"calculator": lambda **kwargs: (
        tool_calls.append(kwargs) or kwargs["a"] + kwargs["b"]
    )}).invoke(initial_state("Add"))
    assert state["status"] == "completed"
    assert model.calls == 3
    assert len(tool_calls) == 1
