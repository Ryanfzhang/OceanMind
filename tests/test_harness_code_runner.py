import pytest

from packages.harness import CodeSafetyError, run_code_node


def test_code_runner_executes_safe_run_function():
    result = run_code_node(
        """
import numpy as np

def run(inputs, params):
    values = np.asarray(inputs["values"], dtype=float)
    return {"mean": float(np.nanmean(values)), "scale": params["scale"]}
""",
        {"values": [1, 2, 3]},
        {"scale": 10},
    )
    assert result == {"mean": 2.0, "scale": 10}


def test_code_runner_blocks_file_access():
    with pytest.raises(CodeSafetyError):
        run_code_node(
            """
def run(inputs, params):
    open("/tmp/x", "w")
    return {}
""",
            {},
            {},
        )

