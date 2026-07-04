from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WATERMASS_CONFIG_PATH = PROJECT_ROOT / "configs" / "watermass_config.yaml"


class WatermassBin(BaseModel):
    id: str
    name: str
    short_name: Optional[str] = None
    color: str = "#475569"
    sigma0_range: list[float] = Field(default_factory=list)
    temp_range: list[float] = Field(default_factory=list)
    salt_range: list[float] = Field(default_factory=list)

    def normalized(self) -> "WatermassBin":
        return WatermassBin.model_validate(
            {
                **self.model_dump(mode="json"),
                "sigma0_range": _normalize_optional_range(self.sigma0_range),
                "temp_range": _normalize_optional_range(self.temp_range),
                "salt_range": _normalize_optional_range(self.salt_range),
            }
        )


class WatermassConfig(BaseModel):
    id: str = "default_watermass_bins"
    name: str = "Default Watermass Bins"
    description: str = "Named water-mass bins used for tile-level event association diagnostics."
    classification_axes: list[str] = Field(default_factory=list)
    bins: list[WatermassBin] = Field(default_factory=list)


def _normalize_range(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("Watermass bin ranges must be two-element arrays")
    lower, upper = sorted((float(values[0]), float(values[1])))
    return [lower, upper]


def _normalize_optional_range(values: Any) -> list[float]:
    if values in (None, ""):
        return []
    if isinstance(values, (list, tuple)) and len(values) == 0:
        return []
    return _normalize_range(values)


def _normalize_classification_axes(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise ValueError("classification_axes must be an array")
    normalized: list[str] = []
    supported_axes = {"sigma0", "temp", "salt"}
    for axis in values:
        axis_name = str(axis).strip().lower()
        if not axis_name:
            continue
        if axis_name not in supported_axes:
            raise ValueError(f"Unsupported watermass classification axis: {axis_name}")
        if axis_name not in normalized:
            normalized.append(axis_name)
    if not normalized:
        raise ValueError("classification_axes must contain at least one axis")
    return normalized


def _resolve_classification_axes(config: WatermassConfig) -> list[str]:
    if config.classification_axes:
        return _normalize_classification_axes(config.classification_axes)
    if any(len(item.sigma0_range) == 2 for item in config.bins):
        return ["sigma0", "temp", "salt"]
    return ["temp", "salt"]


def _validate_bins_for_axes(config: WatermassConfig, axes: list[str]) -> None:
    for item in config.bins:
        for axis in axes:
            bounds = getattr(item, f"{axis}_range", [])
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"Watermass bin '{item.id}' must define {axis}_range for axes {axes}")


def _resolve_config_path(config_path: Optional[str] = None) -> Path:
    return Path(config_path).resolve() if config_path else DEFAULT_WATERMASS_CONFIG_PATH


def _read_config_payload(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Watermass config must be a mapping: {config_path}")
    return payload


def _build_watermass_config(payload: Dict[str, Any]) -> WatermassConfig:
    config = WatermassConfig.model_validate(payload)
    axes = _resolve_classification_axes(config)
    bins = [item.normalized() for item in config.bins]
    normalized_config = config.model_copy(
        update={
            "classification_axes": axes,
            "bins": bins,
        },
        deep=True,
    )
    _validate_bins_for_axes(normalized_config, axes)
    return normalized_config


@lru_cache(maxsize=8)
def get_active_watermass_config(config_path: Optional[str] = None) -> WatermassConfig:
    resolved_config_path = _resolve_config_path(config_path)
    payload = _read_config_payload(resolved_config_path)
    return _build_watermass_config(payload)


def get_active_watermass_context(config_path: Optional[str] = None) -> Dict[str, Any]:
    config = get_active_watermass_config(config_path)
    payload = config.model_dump(mode="json")
    return {
        "watermass_config": payload,
        "watermass_brief": {
            "id": config.id,
            "name": config.name,
            "description": config.description,
            "bin_count": len(config.bins),
        },
    }


def clear_watermass_config_cache() -> None:
    get_active_watermass_config.cache_clear()
