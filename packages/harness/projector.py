"""Projection helpers from harness artifacts to current frontend payloads."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from packages.harness.artifact_store import ArtifactRecord, ArtifactStore
from packages.harness.specs import FrontendType


def project_record_to_result(record: ArtifactRecord) -> Dict[str, Any]:
    """Return a normalized result dict close to the existing API contract."""

    if isinstance(record.value, dict) and "output_type" in record.value:
        return dict(record.value)
    output_type = record.spec.frontend_type.value
    if output_type == FrontendType.DATA_CONTAINER.value:
        return {
            "output_type": output_type,
            "data": record.value,
            "metadata": _metadata_from_record(record),
        }
    return {
        "output_type": output_type,
        "value": record.value,
        "metadata": _metadata_from_record(record),
    }


def project_store_summaries(store: ArtifactStore) -> Dict[str, Mapping[str, Any]]:
    return store.summaries()


def _metadata_from_record(record: ArtifactRecord) -> Dict[str, Any]:
    return {
        "artifact_id": record.artifact_id,
        "kind": record.spec.kind.value,
        "shape_class": record.spec.shape_class.value,
        "dims": list(record.spec.dims),
        "units": record.spec.units,
        "variable": record.spec.variable,
        "provenance": dict(record.provenance or record.spec.provenance or {}),
    }

