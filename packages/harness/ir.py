"""Task graph intermediate representation for OceanMind harness planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from packages.harness.shapes import ShapeClass
from packages.harness.specs import ArtifactSpec, ReadSpec


class NodeType(str, Enum):
    READ = "read"
    SELECT = "select"
    DERIVE = "derive"
    REDUCE = "reduce"
    DIAGNOSE = "diagnose"
    PROJECT = "project"
    SYNTHESIZE = "synthesize"


class ExecutionStrategy(str, Enum):
    TOOL = "tool"
    CODE = "generated_code"
    HARNESS = "harness"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class ExecutionSpec:
    strategy: ExecutionStrategy
    tool_name: Optional[str] = None
    code: Optional[str] = None
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskNode:
    node_id: str
    node_type: NodeType
    intent: str
    operation: str
    inputs: Mapping[str, str] = field(default_factory=dict)
    output: Optional[ArtifactSpec] = None
    execution: Optional[ExecutionSpec] = None
    read_spec: Optional[ReadSpec] = None
    expected_input_shapes: Mapping[str, ShapeClass] = field(default_factory=dict)
    validation_rules: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskGraph:
    graph_id: str
    nodes: Tuple[TaskNode, ...]
    final_artifacts: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def node_ids(self) -> Tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    def output_ids(self) -> Tuple[str, ...]:
        return tuple(
            node.output.artifact_id
            for node in self.nodes
            if node.output is not None
        )


def task_graph_from_nodes(
    graph_id: str,
    nodes: Iterable[TaskNode],
    *,
    final_artifacts: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> TaskGraph:
    node_tuple = tuple(nodes)
    return TaskGraph(
        graph_id=graph_id,
        nodes=node_tuple,
        final_artifacts=tuple(final_artifacts or ()),
        metadata=metadata or {},
    )

