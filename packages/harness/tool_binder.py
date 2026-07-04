"""Tool/code binding policy for harness DAG nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from packages.harness.ir import ExecutionStrategy, TaskNode


@dataclass(frozen=True)
class BindingReport:
    tool_nodes: Tuple[str, ...]
    code_nodes: Tuple[str, ...]
    missing_bindings: Tuple[str, ...]


class ToolBinder:
    """Validate that analysis nodes are executable by a tool or code strategy."""

    def bind(self, nodes: Iterable[TaskNode]) -> BindingReport:
        tool_nodes = []
        code_nodes = []
        missing = []
        for node in nodes:
            execution = node.execution
            if execution is None:
                missing.append(node.node_id)
            elif execution.strategy == ExecutionStrategy.CODE:
                code_nodes.append(node.node_id)
            elif execution.strategy == ExecutionStrategy.TOOL:
                tool_nodes.append(node.node_id)
            else:
                missing.append(node.node_id)
        return BindingReport(
            tool_nodes=tuple(tool_nodes),
            code_nodes=tuple(code_nodes),
            missing_bindings=tuple(missing),
        )
