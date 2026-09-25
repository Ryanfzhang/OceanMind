"""A skill is optional guidance inside the existing agent loop."""

import json
import re

from packages.agent_loop.graph import build_graph
from packages.agent_loop.skills import DEFAULT_SKILLS_ROOT
from packages.agent_loop.state import initial_state


def call(name, arguments, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen = []

    def complete(self, messages, *, tools, timeout):
        self.seen.append((messages, tools))
        response = next(self.responses)
        return response(messages, tools) if callable(response) else response


def test_skill_examples_pass_results_without_stripping_xarray_coordinates():
    for path in DEFAULT_SKILLS_ROOT.glob("*/SKILL.md"):
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"\b[A-Za-z_]\w*\.data\b", source), path
    watermass = (DEFAULT_SKILLS_ROOT / "ocean_watermass_analysis" / "SKILL.md").read_text()
    assert "temp=temp_field" in watermass and "salt=salt_field" in watermass


def test_index_is_metadata_and_body_arrives_only_after_read(tmp_path):
    folder = tmp_path / "sample"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nskill_id: sample\ndescription: Sample guidance\n---\n"
        "# Secret body marker\nUse a measured field.\n"
    )

    def first(messages, tools):
        prompt = messages[0]["content"]
        assert "sample: Sample guidance" in prompt
        assert "Secret body marker" not in prompt
        assert {"read_skill", "find_tools"}.issubset(
            {item["function"]["name"] for item in tools}
        )
        return {"role": "assistant", "content": None, "tool_calls": [
            call("read_skill", {"skill_id": "sample"}, "read_1")
        ]}

    def finish(messages, tools):
        observation = json.loads(messages[-1]["content"])
        assert observation["skill_id"] == "sample"
        assert "Secret body marker" in observation["content"]
        return {"role": "assistant", "content": "I read the guidance."}

    model = ScriptedModel([first, finish])
    state = build_graph(model, skills_root=tmp_path).invoke(initial_state("Use sample"))
    assert state["status"] == "completed"
    assert [item["role"] for item in state["messages"]] == [
        "user", "assistant", "tool", "assistant"
    ]


def test_environment_guidance_leads_to_relevant_calculation_skill():
    def after_environment(messages, tools):
        body = json.loads(messages[-1]["content"])["content"]
        assert "ocean_hypoxia_detection" in body
        assert "four-branch recipe" in body
        return {"role": "assistant", "content": None, "tool_calls": [
            call("read_skill", {"skill_id": "ocean_hypoxia_detection"}, "read_hypoxia")
        ]}

    def after_hypoxia(messages, tools):
        body = json.loads(messages[-1]["content"])["content"]
        assert "detect_hypoxia" in body
        return {"role": "assistant", "content": None, "tool_calls": [
            call("find_tools", {"query": "detect_hypoxia"}, "find_hypoxia")
        ]}

    def finish(messages, tools):
        result = json.loads(messages[-1]["content"])
        assert any(tool["name"] == "detect_hypoxia" for tool in result["tools"])
        return {"role": "assistant", "content": "I can calculate the requested hypoxia measure."}

    model = ScriptedModel([
        {"role": "assistant", "content": None, "tool_calls": [
            call("read_skill", {"skill_id": "ocean_environment_health_assessment"}, "read_environment")
        ]},
        after_environment,
        after_hypoxia,
        finish,
    ])
    state = build_graph(model).invoke(initial_state("Assess bottom hypoxia risk"))
    assert state["status"] == "completed"
    assert [item["tool_call_id"] for item in state["messages"] if item["role"] == "tool"] == [
        "read_environment", "read_hypoxia", "find_hypoxia"
    ]


def test_unknown_task_can_discover_tools_without_reading_a_skill(tmp_path):
    def finish(messages, tools):
        result = json.loads(messages[-1]["content"])
        assert any(tool["name"] == "load_dataset" for tool in result["tools"])
        return {"role": "assistant", "content": "I found a data loading tool."}

    model = ScriptedModel([
        {"role": "assistant", "content": None, "tool_calls": [
            call("find_tools", {"query": "load_dataset"}, "find_loader")
        ]},
        finish,
    ])
    state = build_graph(model, skills_root=tmp_path).invoke(
        initial_state("Inspect an unknown model variable")
    )
    assert state["status"] == "completed"
    assert all(
        item["function"]["name"] != "read_skill"
        for message in state["messages"]
        for item in message.get("tool_calls", [])
    )
