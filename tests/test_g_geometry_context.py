"""The map selection must reach the LangGraph model without duplicate vertices."""

import json

from apps.api.langgraph_query import QueryRequest, QueryService
from packages.analysis_runtime.catalog import find_tools


class CaptureModel:
    def __init__(self):
        self.messages = None

    def complete(self, messages, *, tools, timeout):
        self.messages = messages
        return {"role": "assistant", "content": "Selection received."}


def _sent_context(model):
    content = model.messages[-1]["content"]
    assert "Workspace context supplied by the user:\n" in content
    return json.loads(content.split("Workspace context supplied by the user:\n", 1)[1])


def test_polygon_vertices_survive_without_duplicate_payloads(tmp_path):
    points = [[110 + index / 1000, 20 + index / 1000] for index in range(500)]
    selection = {"selected_region": {"type": "polygon", "points": points}}
    model = CaptureModel()
    service = QueryService(tmp_path, model_factory=lambda: model, data_roots=lambda: ())
    result = service.execute(QueryRequest(
        query="Analyze the drawn polygon",
        extracted_params={
            "mask_polygon": points, "drawn_polygon_points": points,
            "region": {"lon_range": [110, 111], "lat_range": [20, 21]},
            "region_selection_type": "polygon", "workspace_selection": selection,
        },
        additional_context={"workspace_context": {
            "mask_polygon": points, "drawn_polygon_points": points,
            "workspace_selection": selection, "variable": "temp",
        }},
    ))
    assert result["status"] == "completed"
    context = _sent_context(model)
    assert context["extracted_params"]["mask_polygon"] == points
    assert context["extracted_params"]["region_selection_type"] == "polygon"
    assert context["additional_context"]["workspace_context"]["variable"] == "temp"
    assert "drawn_polygon_points" not in model.messages[-1]["content"]
    assert "workspace_selection" not in model.messages[-1]["content"]
    assert "polygon mask" in model.messages[0]["content"]


def test_transect_and_point_use_lon_lat_contract_with_query_priority(tmp_path):
    model = CaptureModel()
    service = QueryService(tmp_path, model_factory=lambda: model, data_roots=lambda: ())
    points = [[119.0, 21.0], [120.0, 22.0], [121.0, 21.5]]
    service.execute(QueryRequest(
        query="Use a transect from 118 E, 20 N instead",
        extracted_params={
            "transect_points": points, "drawn_transect_points": points,
            "selected_point": {"lon": 120.0, "lat": 21.0},
        },
        additional_context={"workspace_context": {
            "transect_points": points, "drawn_transect_points": points,
            "selected_point": {"lon": 120.0, "lat": 21.0},
        }},
    ))
    assert model.messages[-1]["content"].startswith("Use a transect from 118 E, 20 N instead")
    context = _sent_context(model)
    assert context["extracted_params"]["transect_points"] == points
    assert context["extracted_params"]["selected_point"] == {"lon": 120.0, "lat": 21.0}
    assert "drawn_transect_points" not in model.messages[-1]["content"]
    assert "explicit coordinates in the current query take precedence" in model.messages[0]["content"]


def test_selection_only_context_is_kept(tmp_path):
    model = CaptureModel()
    service = QueryService(tmp_path, model_factory=lambda: model, data_roots=lambda: ())
    selection = {"selected_region": {"type": "box", "region": {
        "lon_range": [110, 120], "lat_range": [18, 23],
    }}}
    service.execute(QueryRequest(
        query="Use the box",
        additional_context={"workspace_context": {"workspace_selection": selection}},
    ))
    assert _sent_context(model)["additional_context"]["workspace_context"]["workspace_selection"] == selection


def test_nested_polygon_is_not_dropped_when_only_box_bounds_are_extracted(tmp_path):
    model = CaptureModel()
    service = QueryService(tmp_path, model_factory=lambda: model, data_roots=lambda: ())
    selection = {"selected_region": {"type": "polygon", "points": [
        [110, 20], [111, 20], [111, 21],
    ]}}
    service.execute(QueryRequest(
        query="Analyze my selected polygon",
        extracted_params={"region": {"lon_range": [110, 111], "lat_range": [20, 21]}},
        additional_context={"workspace_context": {"workspace_selection": selection}},
    ))
    assert _sent_context(model)["additional_context"]["workspace_context"]["workspace_selection"] == selection


def test_geometry_tools_are_found_by_natural_multiword_queries():
    for query, name, argument in (
        ("polygon mask", "build_polygon_mask", "polygon_points"),
        ("transect section", "extract_transect_section", "transect_points"),
        ("point timeseries", "extract_point_timeseries", "lon"),
    ):
        matches = find_tools(query)["tools"]
        tool = next(item for item in matches if item["name"] == name)
        assert argument in {item["name"] for item in tool["parameters"]}
