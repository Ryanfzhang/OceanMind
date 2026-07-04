"""Result gathering policy for final synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple


@dataclass(frozen=True)
class SynthesisPacketSpec:
    final_artifacts: Tuple[str, ...]
    supporting_artifacts: Tuple[str, ...]
    image_artifacts: Tuple[str, ...]
    table_artifacts: Tuple[str, ...]
    series_artifacts: Tuple[str, ...]


class ArtifactGatherer:
    """Choose which artifacts should be sent to the final synthesis step."""

    def plan_packet(self, semantic_task_graph: Mapping[str, Any]) -> SynthesisPacketSpec:
        final_artifacts = tuple(str(item) for item in semantic_task_graph.get("final_artifacts", []) or [])
        artifacts = semantic_task_graph.get("artifacts", []) or []
        image_artifacts = []
        table_artifacts = []
        series_artifacts = []
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            artifact_id = str(artifact.get("artifact_id") or "")
            frontend_type = str(artifact.get("frontend_type") or "")
            kind = str(artifact.get("kind") or "")
            if frontend_type == "spatial_field_result" or kind == "map":
                image_artifacts.append(artifact_id)
            elif frontend_type in {"trend_result", "lag_correlation_result", "generic_result"} or kind == "table":
                table_artifacts.append(artifact_id)
            elif frontend_type == "timeseries_result" or kind == "series":
                series_artifacts.append(artifact_id)
        supporting = tuple(
            artifact_id
            for artifact_id in [*image_artifacts, *table_artifacts, *series_artifacts]
            if artifact_id and artifact_id not in final_artifacts
        )
        return SynthesisPacketSpec(
            final_artifacts=final_artifacts,
            supporting_artifacts=supporting,
            image_artifacts=tuple(image_artifacts),
            table_artifacts=tuple(table_artifacts),
            series_artifacts=tuple(series_artifacts),
        )


def packet_spec_to_dict(packet: SynthesisPacketSpec) -> Dict[str, Any]:
    return {
        "final_artifacts": list(packet.final_artifacts),
        "supporting_artifacts": list(packet.supporting_artifacts),
        "image_artifacts": list(packet.image_artifacts),
        "table_artifacts": list(packet.table_artifacts),
        "series_artifacts": list(packet.series_artifacts),
        "multimodal_policy": "send compact numeric summaries plus selected map/series/spectrum images when a multimodal model is available",
    }
