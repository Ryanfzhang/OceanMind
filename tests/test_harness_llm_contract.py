import pytest

from packages.harness.llm_contract import (
    LLMContractError,
    XMLIOContract,
    parse_json_from_xml_output,
    render_xml_io_contract,
)
from packages.harness.planner import OceanHarnessPlanner


def test_xml_contract_renders_input_and_output_blocks():
    prompt = render_xml_io_contract(
        XMLIOContract(
            task="Decompose an ocean request",
            input_payload={"query": "Detect heatwaves", "manuals": ["ocean_heatwave_detection"]},
            output_schema={"planner_kind": "ocean_harness", "steps": []},
        )
    )

    assert "<input format=\"json\">" in prompt
    assert "<output format=\"json\">" in prompt
    assert "Detect heatwaves" in prompt
    assert "planner_kind" in prompt


def test_parse_json_from_xml_output_ignores_input_block_and_repairs_common_slips():
    response = """
    <input>{"not": "trusted", "broken": }</input>
    The useful answer is below.
    <output>
    ```json
    {
      "planner_kind": "ocean_harness" // comment from model
      "steps": [],
      "needs_clarification": False,
      "missing_fields": None,
    }
    ```
    </output>
    """

    parsed = parse_json_from_xml_output(response, required_keys=["planner_kind", "steps"])

    assert parsed == {
        "planner_kind": "ocean_harness",
        "steps": [],
        "needs_clarification": False,
        "missing_fields": None,
    }


def test_planner_exposes_xml_output_json_parser():
    parsed = OceanHarnessPlanner.parse_llm_output_json(
        """
        <output>
        Some prefix:
        {"task_graph": {"nodes": []}, "status": "ready"}
        </output>
        """,
        required_keys=["task_graph", "status"],
    )

    assert parsed["status"] == "ready"
    assert parsed["task_graph"]["nodes"] == []


def test_parse_json_from_xml_output_requires_output_tag():
    with pytest.raises(LLMContractError):
        parse_json_from_xml_output('{"planner_kind": "ocean_harness"}')
