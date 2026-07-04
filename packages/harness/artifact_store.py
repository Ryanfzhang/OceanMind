"""Artifact storage for harness execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from packages.harness.contracts import ValidationResult
from packages.harness.specs import ArtifactSpec


@dataclass
class ArtifactRecord:
    artifact_id: str
    spec: ArtifactSpec
    value: Any
    summary: Mapping[str, Any] = field(default_factory=dict)
    validation: ValidationResult = field(default_factory=ValidationResult)
    provenance: Mapping[str, Any] = field(default_factory=dict)


class ArtifactStore:
    def __init__(self) -> None:
        self._records: Dict[str, ArtifactRecord] = {}

    def put(self, record: ArtifactRecord) -> ArtifactRecord:
        if not record.artifact_id:
            raise ValueError("ArtifactRecord requires artifact_id")
        self._records[record.artifact_id] = record
        return record

    def put_value(
        self,
        artifact_id: str,
        value: Any,
        spec: ArtifactSpec,
        *,
        summary: Optional[Mapping[str, Any]] = None,
        validation: Optional[ValidationResult] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> ArtifactRecord:
        return self.put(
            ArtifactRecord(
                artifact_id=artifact_id,
                value=value,
                spec=spec,
                summary=summary or {},
                validation=validation or ValidationResult(),
                provenance=provenance or {},
            )
        )

    def get(self, artifact_id: str) -> ArtifactRecord:
        try:
            return self._records[artifact_id]
        except KeyError as exc:
            raise KeyError(f"Artifact not found: {artifact_id}") from exc

    def has(self, artifact_id: str) -> bool:
        return artifact_id in self._records

    def values(self) -> Iterable[ArtifactRecord]:
        return self._records.values()

    def summaries(self) -> Dict[str, Mapping[str, Any]]:
        return {artifact_id: record.summary for artifact_id, record in self._records.items()}

    def specs(self) -> Dict[str, ArtifactSpec]:
        return {artifact_id: record.spec for artifact_id, record in self._records.items()}

    def clear(self) -> None:
        self._records.clear()

