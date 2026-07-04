from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_config.yaml"


class DatasetConfig(BaseModel):
    id: str = "current"
    name: str = "CMOMS"
    data_path: str = "./data"
    backend: str = "netcdf"
    description: str = (
        "CMOMS provides gridded upper-ocean state variables for map-first exploration "
        "and agent-driven scientific analysis."
    )
    variable_names: Optional[Dict[str, str]] = None
    variables: list[str] = Field(default_factory=list)
    chunks: Optional[Dict[str, Any]] = None
    zarr_store_pattern: Optional[str] = None
    spatial_extent: Optional[Dict[str, Any]] = None
    temporal_extent: Optional[Dict[str, Any]] = None
    depth_levels: list[float] = Field(default_factory=list)
    depth_range: Optional[list[float]] = None
    resolution: Optional[str | Dict[str, Any]] = None

    def resolve_variable(self, canonical: str) -> str:
        if self.variable_names and canonical in self.variable_names:
            return self.variable_names[canonical]
        return canonical


def _resolve_config_path(config_path: Optional[str] = None) -> Path:
    return Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH


def _read_config_payload(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset config must be a mapping: {config_path}")
    return payload


def _resolve_data_path(raw_path: str, config_path: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if config_path == DEFAULT_CONFIG_PATH:
        return (PROJECT_ROOT / candidate).resolve()
    return (config_path.parent / candidate).resolve()


def _build_dataset_config(payload: Dict[str, Any], config_path: Path) -> DatasetConfig:
    normalized_payload = dict(payload)
    data_path = str(_resolve_data_path(str(normalized_payload.get("data_path") or "./data"), config_path))
    backend = str(normalized_payload.get("backend") or _infer_backend_from_path(Path(data_path))).strip().lower()
    depth_levels = [float(level) for level in normalized_payload.get("depth_levels") or []]
    depth_range = [depth_levels[0], depth_levels[-1]] if depth_levels else None

    normalized_payload["data_path"] = data_path
    normalized_payload["backend"] = backend
    normalized_payload["depth_levels"] = depth_levels
    normalized_payload["depth_range"] = depth_range
    return DatasetConfig.model_validate(normalized_payload)


def _infer_backend_from_path(path: Path) -> str:
    if path.suffix.lower() == ".zarr" or path.name.lower().endswith(".zarr"):
        return "zarr"
    if path.suffix.lower() == ".nc":
        return "netcdf"
    if path.is_dir():
        try:
            if any(path.glob("*.zarr")):
                return "zarr"
            if any(path.glob("*.nc")):
                return "netcdf"
        except OSError:
            pass
    return "netcdf"


@lru_cache(maxsize=8)
def get_active_dataset_config(config_path: Optional[str] = None) -> DatasetConfig:
    resolved_config_path = _resolve_config_path(config_path)
    payload = _read_config_payload(resolved_config_path)
    return _build_dataset_config(payload, resolved_config_path)


@lru_cache(maxsize=8)
def get_dataset_config_for_path(
    dataset_path: str,
    config_path: Optional[str] = None,
) -> DatasetConfig:
    resolved_config_path = _resolve_config_path(config_path)
    config = get_active_dataset_config(config_path)
    resolved_dataset_path = str(_resolve_data_path(dataset_path, resolved_config_path))
    backend = _infer_backend_from_path(Path(resolved_dataset_path))
    return config.model_copy(
        update={"data_path": resolved_dataset_path, "backend": backend},
        deep=True,
    )


def get_active_dataset_context(config_path: Optional[str] = None) -> Dict[str, Any]:
    config = get_active_dataset_config(config_path)
    payload = get_active_dataset_public_config(config_path)
    return {
        "dataset": payload,
        "dataset_brief": {
            "id": config.id,
            "name": config.name,
            "description": config.description,
        },
    }


def get_active_dataset_public_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Return dataset config fields that are safe to expose through API/LLM context."""
    config = get_active_dataset_config(config_path)
    payload = config.model_dump(mode="json")
    payload.pop("data_path", None)
    payload["data_path_redacted"] = True
    payload["data_path_policy"] = "Local server data paths are hidden from public dataset descriptions."
    return payload


def clear_dataset_config_cache() -> None:
    get_dataset_config_for_path.cache_clear()
    get_active_dataset_config.cache_clear()
