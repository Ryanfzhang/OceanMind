"""Compile task graph nodes into executable step dictionaries."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from packages.harness.ir import ExecutionSpec, ExecutionStrategy, TaskGraph, TaskNode


def compile_task_graph(graph: TaskGraph) -> List[Dict[str, Any]]:
    """Compile a TaskGraph into the current event/step protocol."""

    return [compile_node(node) for node in graph.nodes]


def compile_node(node: TaskNode) -> Dict[str, Any]:
    execution = node.execution or ExecutionSpec(strategy=ExecutionStrategy.HARNESS)
    save_as = node.output.artifact_id if node.output is not None else node.node_id
    return {
        "step_id": node.node_id,
        "tool": execution.tool_name or node.operation,
        "params": dict(execution.params),
        "save_as": save_as,
        "harness_node": {
            "node_id": node.node_id,
            "node_type": node.node_type.value,
            "operation": node.operation,
            "strategy": execution.strategy.value,
            "inputs": dict(node.inputs),
            "output": save_as,
            "execution": {
                "strategy": execution.strategy.value,
                "tool_name": execution.tool_name,
                "code": execution.code,
                "params": dict(execution.params),
            },
        },
    }
