"""Acceptance checks for the two decision-oriented D skills."""

from __future__ import annotations

import inspect
import re
from contextlib import contextmanager
from pathlib import Path

import xarray as xr

from domain.ocean.data_access.load import load_dataset
from domain.ocean.events.hypoxia.detect import detect_hypoxia
from domain.ocean.events.analysis.statistics import compute_event_summary_map
from domain.ocean.interpretation.explain import (
    assemble_environment_health_report,
    assemble_policy_recommendation_report,
)


ROOT = Path(__file__).resolve().parents[1]
ENV_SKILL = ROOT / "skills/ocean_environment_health_assessment/SKILL.md"
POLICY_SKILL = ROOT / "skills/ocean_policy_recommendation/SKILL.md"


def _first_python_block(path: Path) -> str:
    match = re.search(r"```python\n(.*?)\n```", path.read_text(), re.DOTALL)
    assert match, f"No executable Python example in {path.name}"
    return match.group(1)


def test_bottom_hypoxic_days_example_runs_only_requested_diagnostic():
    """The skill's narrow worked example uses real signatures, no DSL or extra branches."""
    field = xr.DataArray(
        [[[45.0]]], dims=("time", "lat", "lon"),
        coords={"time": ["2022-07-01"], "lat": [18.0], "lon": [112.0]},
        name="oxygen", attrs={"units": "mmol/m³"},
    )
    detection = {"event_type": "hypoxia", "events": []}
    calls: list[tuple[str, dict]] = []
    stages: list[str] = []

    class FakeTools:
        def ref(self, value):
            return "oxygen_artifact" if value is field else "detection_artifact"

        def load_dataset(self, **kwargs):
            inspect.signature(load_dataset).bind(**kwargs)
            calls.append(("load_dataset", kwargs))
            return field

        def detect_hypoxia(self, **kwargs):
            inspect.signature(detect_hypoxia).bind(**kwargs)
            calls.append(("detect_hypoxia", kwargs))
            return detection

        def compute_event_summary_map(self, **kwargs):
            refs = kwargs.pop("input_refs")
            inspect.signature(compute_event_summary_map).bind(**kwargs)
            assert refs == ["detection_artifact", "oxygen_artifact"]
            calls.append(("compute_event_summary_map", kwargs))
            return {"metadata": {"summary_mode": "event_days"}}

    @contextmanager
    def stage(title):
        stages.append(title)
        yield

    scope = {
        "tools": FakeTools(), "stage": stage,
        "lon_range": (110, 115), "lat_range": (16, 22),
        "time_range": ("2022-07-01", "2022-07-31"),
        "oxygen_threshold": 60,
    }
    exec(compile(_first_python_block(ENV_SKILL), str(ENV_SKILL), "exec"), scope)

    assert [name for name, _ in calls] == [
        "load_dataset", "detect_hypoxia", "compute_event_summary_map"
    ]
    assert len(stages) == 2
    assert calls[0][1]["vertical_mode"] == "bottom"
    assert calls[1][1]["oxygen"] is field
    assert calls[2][1]["event_detection"] is detection
    assert scope["hypoxic_days"]["metadata"]["summary_mode"] == "event_days"


def test_legacy_reports_do_not_create_observed_evidence_from_empty_input():
    """Document why both old report builders need an evidence guard."""
    environment = assemble_environment_health_report([])
    policy = assemble_policy_recommendation_report([], region_scope="Example region")
    raw_detection = assemble_policy_recommendation_report(
        [{"event_type": "hypoxia", "events": []}], region_scope="Example region"
    )

    assert environment["metadata"]["n_input_branches"] == 0
    assert environment["overall_support_strength"] == "untestable"
    assert policy["metadata"]["n_input_items"] == 0
    assert policy["evidence_table"] == []
    assert raw_detection["metadata"]["n_input_items"] == 0

    env_text = ENV_SKILL.read_text()
    policy_text = POLICY_SKILL.read_text()
    assert "Never pass an empty branch list" in env_text
    assert "evidence_items=[]" in policy_text
    assert "filtered out" in policy_text


def test_examples_do_not_use_legacy_reference_syntax():
    for path in (ENV_SKILL, POLICY_SKILL):
        text = path.read_text()
        assert "$ref" not in text
        assert "workflow_code" not in text
