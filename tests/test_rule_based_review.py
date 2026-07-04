from benchmarks.rule_based_review import _depth_requirement_satisfied
from benchmarks.trace import ToolCall


def _successful_load(params):
    return ToolCall(
        tool_name="load_dataset",
        params=params,
        success=True,
        error=None,
        latency_ms=0.0,
    )


def test_fixed_depth_zero_satisfies_surface_requirement():
    calls = [
        _successful_load(
            {
                "vertical_mode": "fixed_depth",
                "depth_value": 0.0,
            }
        )
    ]

    assert _depth_requirement_satisfied(("surface", 0.0), calls)


def test_fixed_depth_value_satisfies_exact_depth_requirement():
    calls = [
        _successful_load(
            {
                "vertical_mode": "fixed_depth",
                "depth_value": 50.0,
            }
        )
    ]

    assert _depth_requirement_satisfied(("exact", 50.0), calls)
