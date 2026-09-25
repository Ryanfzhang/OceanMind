"""Tool loader package exports."""

from packages.tool_loader.introspect import (
    discover_tools,
    get_all_tool_schemas,
    get_all_planner_tool_specs,
    get_planner_specs_cached,
    get_planner_tool_spec,
    get_schemas_cached,
    get_tool_schema,
    get_tool_by_name,
    get_tools_cached,
    reload_tools,
)


def __getattr__(name: str):
    """Keep legacy exports available without loading DSL contracts for discovery."""
    if name == "ToolOrchestrator":
        from packages.tool_loader.orchestrator import ToolOrchestrator
        return ToolOrchestrator
    if name in {"TOOL_CONTRACTS", "get_tool_contract", "get_tool_output_type"}:
        from packages.tool_loader import registry
        return getattr(registry, name)
    if name == "validate_tool_params":
        from packages.tool_loader.validation import validate_tool_params
        return validate_tool_params
    raise AttributeError(name)


__all__ = [
    "ToolOrchestrator",
    "TOOL_CONTRACTS",
    "discover_tools",
    "get_all_planner_tool_specs",
    "get_all_tool_schemas",
    "get_planner_specs_cached",
    "get_planner_tool_spec",
    "get_schemas_cached",
    "get_tool_contract",
    "get_tool_by_name",
    "get_tool_output_type",
    "get_tool_schema",
    "get_tools_cached",
    "reload_tools",
    "validate_tool_params",
]
