"""The Python-tool path should not inherit legacy DSL contracts."""

import json
import subprocess
import sys

from packages.analysis_runtime.catalog import find_tools


def test_catalog_import_does_not_load_legacy_registry():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import packages.analysis_runtime.catalog, sys; "
            "assert 'packages.tool_loader.registry' not in sys.modules; "
            "assert 'packages.tool_loader.orchestrator' not in sys.modules"
        )],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_tool_metadata_describes_python_calls_without_dsl_refs():
    loader = find_tools("load_dataset", limit=1)["tools"][0]
    assert loader["discovery_name"].endswith(".load_dataset")
    assert loader["python_return_type"]
    assert "variable" in {param["name"] for param in loader["parameters"]}
    encoded = json.dumps(loader)
    assert "$ref:" not in encoded
    assert "x-orchestration" not in encoded
    assert "legacy_registry_output_type" not in loader


def test_legacy_schema_exports_remain_available():
    from packages.tool_loader import ToolOrchestrator, get_tool_schema
    from domain.ocean.events.eddy.detect import detect_eddies

    assert ToolOrchestrator.__name__ == "ToolOrchestrator"
    assert get_tool_schema(detect_eddies)["name"] == "detect_eddies"
