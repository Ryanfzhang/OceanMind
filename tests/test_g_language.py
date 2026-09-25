"""The final answer follows the current question, even after foreign-language tools."""

from apps.api.langgraph_query import QueryRequest, QueryService
from packages.agent_loop.graph import build_graph
from packages.agent_loop.language import preferred_language
from packages.agent_loop.model import OpenAIChatModel
from packages.agent_loop.state import initial_state


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.seen = []

    def complete(self, messages, *, tools, timeout):
        self.seen.append((messages, tools))
        return {"role": "assistant", "content": next(self.replies)}


def test_english_answer_is_repaired_after_chinese_model_output():
    model = ScriptedModel([
        "## 结果\n海表温度为 22 °C。",
        "## Result\nThe sea surface temperature is 22 °C.",
    ])
    state = build_graph(model).invoke(initial_state("What is the sea surface temperature?"))
    assert state["status"] == "completed"
    assert state["messages"][-1]["content"] == (
        "## Result\nThe sea surface temperature is 22 °C."
    )
    assert "in English" in model.seen[0][0][0]["content"]
    assert model.seen[1][1] == []


def test_question_language_ignores_technical_english_and_sets_api_metadata(tmp_path):
    assert preferred_language("请分析 SST 和 CMOMS 的温度") == "zh"
    assert preferred_language("Explain 中国海 depth using sources") == "en"
    model = ScriptedModel(["## 结果\n海表温度为 22 °C。"])
    service = QueryService(tmp_path, model_factory=lambda: model, data_roots=lambda: ())
    result = service.execute(QueryRequest(query="请解释 SST 是什么"))
    assert result["language"] == "zh"
    assert result["synthesis"]["summary"].startswith("## 结果")
    assert "in Chinese" in model.seen[0][0][0]["content"]


def test_new_english_turn_overrides_chinese_conversation_history(tmp_path):
    model = ScriptedModel([
        "这是之前的中文答复。",
        "这次仍然错误地用了中文。",
        "This answer is in English.",
    ])
    service = QueryService(tmp_path, model_factory=lambda: model, data_roots=lambda: ())
    first = service.execute(QueryRequest(query="请解释海流"))
    second = service.execute(QueryRequest(
        query="Explain the ocean current in English",
        conversation_id=first["conversation_id"],
    ))
    assert second["language"] == "en"
    assert second["synthesis"]["summary"] == "This answer is in English."
    assert "in English" in model.seen[1][0][0]["content"]


def test_query_service_uses_separate_answer_model_when_configured(tmp_path):
    executor = ScriptedModel(["Draft from the executor."])
    answer = ScriptedModel(["Verified final answer."])
    service = QueryService(
        tmp_path, model_factory=lambda: executor,
        answer_model_factory=lambda: answer, data_roots=lambda: (),
    )
    result = service.execute(QueryRequest(query="Give the verified answer"))
    assert result["synthesis"]["summary"] == "Verified final answer."
    assert len(executor.seen) == len(answer.seen) == 1


def test_deepseek_reasoning_content_survives_tool_turns():
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return {"choices": [{"message": {
                        "content": None, "reasoning_content": "synthetic reasoning",
                        "tool_calls": [{"id": "call_1", "type": "function", "function": {
                            "name": "web_search", "arguments": '{"query":"ocean"}',
                        }}],
                    }}]}

    model = OpenAIChatModel(api_key="test", base_url="http://test", model="test", client=Client())
    reply = model.complete([{"role": "user", "content": "Search"}], tools=[], timeout=None)
    assert reply["reasoning_content"] == "synthetic reasoning"
    assert reply["tool_calls"][0]["function"]["name"] == "web_search"
