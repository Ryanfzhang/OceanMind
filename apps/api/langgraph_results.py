"""Read run-owned result indexes and artifacts without exposing task paths."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from apps.api.langgraph_query import QueryService, get_query_service
from packages.analysis_runtime.records import SAFE_ID


router = APIRouter()


def _session(service: QueryService, conversation_id: str):
    try:
        record = service.store.resolve(conversation_id)
        return service.store.open_analysis(record)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Conversation is unavailable") from exc


def _attempt(session: Any, attempt_id: str) -> dict:
    if not SAFE_ID.fullmatch(attempt_id):
        raise HTTPException(status_code=404, detail="Attempt is unavailable")
    try:
        attempt = session.records.read("attempt", attempt_id)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Attempt is unavailable") from exc
    if attempt.get("run_id") != session.run_id:
        raise HTTPException(status_code=404, detail="Attempt is unavailable")
    return attempt


@router.get("/results/{conversation_id}/attempts")
def list_attempts(conversation_id: str,
                  service: QueryService = Depends(get_query_service)) -> dict:
    session = _session(service, conversation_id)
    directory = session.root / "records" / "attempt"
    attempts = []
    if directory.exists():
        for path in directory.glob(f"{session.run_id}_attempt_*.json"):
            attempt = _attempt(session, path.stem)
            attempts.append({key: attempt.get(key) for key in (
                "attempt_id", "code_version", "status", "created_at", "error")})
    attempts.sort(key=lambda item: item.get("created_at") or "")
    return {"conversation_id": conversation_id, "attempts": attempts}


@router.get("/results/{conversation_id}/attempts/{attempt_id}")
def page_results(conversation_id: str, attempt_id: str, offset: int = 0,
                 limit: int = 20,
                 service: QueryService = Depends(get_query_service)) -> dict:
    session = _session(service, conversation_id)
    _attempt(session, attempt_id)
    try:
        return session.list_results(attempt_id, offset=offset, limit=limit)
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/results/{conversation_id}/artifacts/{artifact_id}")
def read_artifact(conversation_id: str, artifact_id: str, offset: int = 0,
                  max_chars: int = 2000,
                  service: QueryService = Depends(get_query_service)) -> dict:
    session = _session(service, conversation_id)
    try:
        result = session.read_artifact(artifact_id, offset=offset, max_chars=max_chars)
        if result.get("status") == "missing":
            raise FileNotFoundError(artifact_id)
        return result
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Artifact is unavailable") from exc


@router.get("/results/{conversation_id}/images/{artifact_id}")
def read_image(conversation_id: str, artifact_id: str,
               service: QueryService = Depends(get_query_service)) -> Response:
    session = _session(service, conversation_id)
    try:
        metadata = session.artifacts.read_artifact(artifact_id)
        if metadata.get("run_id") != session.run_id or metadata.get("kind") != "image_png":
            raise ValueError("Image belongs to another run")
        return Response(content=session.artifacts.read_image(artifact_id), media_type="image/png")
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Image is unavailable") from exc


@router.get("/results/{conversation_id}/code/{code_id}")
def read_code(conversation_id: str, code_id: str,
              service: QueryService = Depends(get_query_service)) -> Response:
    session = _session(service, conversation_id)
    try:
        path = session.codes.get_path(code_id)
        return Response(content=path.read_bytes(), media_type="text/x-python",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{code_id}.py"'})
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Code is unavailable") from exc
