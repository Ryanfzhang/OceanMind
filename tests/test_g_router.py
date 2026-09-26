"""The source router makes external answers search before synthesis."""

import json

from apps.api.langgraph_query import QueryRequest, QueryService
from packages.agent_loop.router import route_query


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.seen = []
        self.tools_seen = []

    def complete(self, messages, *, tools, timeout):
        self.seen.append(messages)
        self.tools_seen.append(tools)
        reply = next(self.replies)
        return reply(messages) if callable(reply) else reply


def decision(mode, *, search_query=None, question=None):
    return {"role": "assistant", "content": json.dumps({
        "mode": mode, "search_query": search_query, "question": question,
    }, ensure_ascii=False)}


def call(name, arguments, call_id="call_search"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
    }}


def test_external_information_searches_even_if_executor_would_skip_it(tmp_path):
    router = ScriptedModel([decision("web_information", search_query="今天香港天气")])

    def answer(messages):
        assert all(not message.get("tool_calls") for message in messages)
        result = json.loads(messages[-1]["content"].split("Web search results: ", 1)[1])
        assert result["results"][0]["url"] == "https://weather.example/hong-kong"
        assert len(result["results"][0]["snippet"]) == 1200
        return {"role": "assistant", "content": "香港天气来源：https://weather.example/hong-kong"}

    executor = ScriptedModel([])
    synthesis = ScriptedModel([answer])
    searched = []

    def search(**kwargs):
        searched.append(kwargs["query"])
        return {"query": kwargs["query"], "provider": "test", "results": [{
            "title": "Hong Kong weather", "url": "https://weather.example/hong-kong",
            "snippet": "Observed conditions " * 200,
        }]}

    service = QueryService(tmp_path, model_factory=lambda: executor,
                           answer_model_factory=lambda: synthesis,
                           router_model_factory=lambda: router, web_search=search,
                           data_roots=lambda: ())
    response = service.execute(QueryRequest(
        query="今天香港天气如何", additional_context={"workspace_context": {
            "region": [98.8, 143.2, 0.9, 48.6],
        }},
    ))
    assert response["status"] == "completed"
    assert response["routing_mode"] == "general_answer"
    assert response["router_reason"] == "web_information"
    assert searched == ["今天香港天气"]
    assert response["source_cards"][0]["url"] == "https://weather.example/hong-kong"
    assert len(response["source_cards"][0]["short_snippet"]) == 600
    assert not executor.seen
    assert len(synthesis.seen) == 1
    assert synthesis.tools_seen == [[]]


def test_search_answer_stays_on_latest_question_after_router_fallback(tmp_path):
    router = ScriptedModel([
        decision("conversation"),
        {"role": "assistant", "content": "invalid routing output"},
    ])
    executor = ScriptedModel([
        {"role": "assistant", "content": "CMOMS ends in December 2022."},
        {"role": "assistant", "content": None, "tool_calls": [
            call("web_search", {"query": "香港今天天气 香港天文台"}),
        ]},
        {"role": "assistant", "content": "The search returned a Hong Kong weather bulletin."},
    ])

    def weather_answer(messages):
        assert len(messages) == 2
        assert "CMOMS" not in json.dumps(messages, ensure_ascii=False)
        assert "香港今天的天气" in messages[-1]["content"]
        assert "https://weather.example/hk" in messages[-1]["content"]
        assert "Current UTC time:" in messages[0]["content"]
        return {"role": "assistant", "content": "香港今天的天气见天文台预报：https://weather.example/hk"}

    synthesis = ScriptedModel([
        {"role": "assistant", "content": "CMOMS ends in December 2022."},
        weather_answer,
    ])
    service = QueryService(
        tmp_path, model_factory=lambda: executor,
        answer_model_factory=lambda: synthesis,
        router_model_factory=lambda: router,
        web_search=lambda **kwargs: {"query": kwargs["query"], "provider": "test", "results": [{
            "title": "Hong Kong weather", "url": "https://weather.example/hk",
            "snippet": "Today: sunny, 28 to 32 C", "published_at": "today",
        }]},
        data_roots=lambda: (),
    )
    first = service.execute(QueryRequest(query="CMOMS 数据到什么时候？"))
    second = service.execute(QueryRequest(
        query="我想知道香港今天的天气", conversation_id=first["conversation_id"],
    ))
    assert second["status"] == "completed"
    assert second["router_reason"] is None
    assert "香港今天的天气" in second["synthesis"]["summary"]
    assert "CMOMS" not in second["synthesis"]["summary"]
    assert second["source_cards"][0]["url"] == "https://weather.example/hk"


def test_missing_location_clarifies_then_searches_on_resume(tmp_path):
    def resumed_route(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["pending_request"] == "今天天气如何"
        assert payload["latest_request"] == "香港"
        assert "城市" in payload["pending_question"]
        return decision("web_information", search_query="今天香港天气")

    router = ScriptedModel([
        decision("clarification", question="请问你要查询哪个城市的天气？"),
        resumed_route,
    ])
    executor = ScriptedModel([{"role": "assistant", "content": "香港天气见来源。"}])
    searched = []

    def search(**kwargs):
        searched.append(kwargs["query"])
        return {"query": kwargs["query"], "provider": "test", "results": [{
            "title": "Weather", "url": "https://weather.example/hk", "snippet": "Conditions",
        }]}

    service = QueryService(tmp_path, model_factory=lambda: executor,
                           router_model_factory=lambda: router, web_search=search,
                           data_roots=lambda: ())
    first = service.execute(QueryRequest(query="今天天气如何"))
    assert first["status"] == "clarification_needed"
    assert first["clarification_question"] == "请问你要查询哪个城市的天气？"
    assert not executor.seen and not searched

    second = service.execute(QueryRequest(
        query="香港", conversation_id=first["conversation_id"], continue_pending=True,
    ))
    assert second["status"] == "completed"
    assert searched == ["今天香港天气"]
    assert second["source_cards"][0]["url"] == "https://weather.example/hk"
    assert not any(message.get("tool_calls") for message in executor.seen[0])


def test_workspace_request_does_not_force_web_search(tmp_path):
    router = ScriptedModel([decision("workspace_analysis")])
    executor = ScriptedModel([{"role": "assistant", "content": "工作区分析完成。"}])
    service = QueryService(tmp_path, model_factory=lambda: executor,
                           router_model_factory=lambda: router,
                           web_search=lambda **_: (_ for _ in ()).throw(AssertionError("searched")),
                           data_roots=lambda: ())
    response = service.execute(QueryRequest(query="可视化2011年叶绿素"))
    assert response["status"] == "completed"
    assert response["router_reason"] == "workspace_analysis"
    assert len(executor.seen) == 1


def test_llm_can_answer_stable_knowledge_without_web_search(tmp_path):
    router = ScriptedModel([decision("conversation")])
    executor = ScriptedModel([{"role": "assistant", "content": "盐度是海水中溶解盐的含量。"}])
    service = QueryService(tmp_path, model_factory=lambda: executor,
                           router_model_factory=lambda: router,
                           web_search=lambda **_: (_ for _ in ()).throw(AssertionError("searched")),
                           data_roots=lambda: ())
    response = service.execute(QueryRequest(query="盐度是什么意思？"))
    assert response["status"] == "completed"
    assert response["router_reason"] == "conversation"
    assert response["source_cards"] == []


def test_provider_validation_error_is_reported_without_request_dump(tmp_path):
    class BadRequestError(Exception):
        body = {"error": {"message": "Tool call message is invalid"}}

    router = ScriptedModel([decision("web_information", search_query="今天香港天气")])
    answer = ScriptedModel([lambda _messages: (_ for _ in ()).throw(BadRequestError("secret"))])
    service = QueryService(tmp_path, model_factory=lambda: answer,
                           router_model_factory=lambda: router,
                           web_search=lambda **_: {"provider": "test", "results": []},
                           data_roots=lambda: ())
    response = service.execute(QueryRequest(query="今天香港天气"))
    assert response["status"] == "failed"
    assert response["error"] == "answer_model_error: BadRequestError: Tool call message is invalid"
    assert "secret" not in response["error"]


def test_invalid_router_output_falls_back_to_agent():
    router = ScriptedModel([{"role": "assistant", "content": "maybe"}])
    assert route_query(router, "hello") is None
