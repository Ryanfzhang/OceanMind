"""Read-only dataset metadata for the LangGraph API."""

from typing import Any

from fastapi import APIRouter

from packages.runtime import get_active_dataset_public_config


router = APIRouter()


@router.get("/dataset")
def get_active_dataset() -> dict[str, Any]:
    return get_active_dataset_public_config()
