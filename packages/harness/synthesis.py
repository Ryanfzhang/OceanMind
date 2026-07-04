"""Build compact evidence packets for final LLM synthesis."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from packages.harness.artifact_store import ArtifactStore
from packages.harness.ir import TaskGraph


def build_synthesis_packet(
    *,
    user_question: str,
    graph: Optional[TaskGraph],
    store: ArtifactStore,
    data_requirements: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "user_question": user_question,
        "data_requirements": dict(data_requirements or {}),
        "task_graph": _summarize_graph(graph),
        "artifact_summaries": {
            record.artifact_id: {
                "kind": record.spec.kind.value,
                "shape_class": record.spec.shape_class.value,
                "dims": list(record.spec.dims),
                "units": record.spec.units,
                "frontend_type": record.spec.frontend_type.value,
                "summary": dict(record.summary or {}),
                "validation": [
                    {
                        "level": issue.level,
                        "message": issue.message,
                        "path": issue.path,
                    }
                    for issue in record.validation.issues
                ],
                "provenance": dict(record.provenance or record.spec.provenance or {}),
            }
            for record in store.values()
        },
    }


def _summarize_graph(graph: Optional[TaskGraph]) -> Dict[str, Any]:
    if graph is None:
        return {}
    return {
        "graph_id": graph.graph_id,
        "nodes": [
            {
                "id": node.node_id,
                "type": node.node_type.value,
                "intent": node.intent,
                "operation": node.operation,
                "inputs": dict(node.inputs),
                "output": node.output.artifact_id if node.output is not None else None,
            }
            for node in graph.nodes
        ],
        "final_artifacts": list(graph.final_artifacts),
    }

