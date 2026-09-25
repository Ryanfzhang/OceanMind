from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, List, Optional

from fastapi.responses import Response
from pydantic import BaseModel, Field

from packages.agent_loop.analysis import AnalysisSession


NOTEBOOK_RESULT_LIMIT = 50
NOTEBOOK_ATTEMPT_LIMIT = 20
NOTEBOOK_CODE_LIMIT = 1024 * 1024
NOTEBOOK_FIGURE_LIMIT = 3
NOTEBOOK_FIGURE_BYTES = 4 * 1024 * 1024
_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:api_?key|access_?token|secret|password)\s*=\s*)(['\"])[^'\"\r\n]+\2"
)


def _redact(value: str) -> str:
    return _TOKEN.sub("[REDACTED]", _SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]\2", value))


def _markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _redact(source)}


def _code(source: str, *, code_id: str, attempt_id: str) -> dict[str, Any]:
    return {"cell_type": "code", "metadata": {"oceanmind": {
        "code_id": code_id, "attempt_id": attempt_id,
    }}, "source": _redact(source), "execution_count": None, "outputs": []}


def _turn_cell(turn: "ReportTurnPayload", number: int) -> dict[str, Any]:
    lines = [f"## Turn {number}", f"**Question:** {turn.user_query}",
             f"**Status:** {turn.assistant_status}", "", turn.assistant_summary]
    for finding in turn.findings:
        lines.append(f"\n### {finding.title}")
        lines.extend(f"- {item}" for item in finding.evidence)
    for step in turn.step_cards:
        lines.append(f"\n### {step.human_label} ({step.status})")
        if step.interpretation:
            lines.append(step.interpretation)
        for result in step.results:
            lines.append(f"\n**{result.title}:** {result.headline or result.description}")
            lines.extend(f"- {metric.label}: {metric.value}" for metric in result.metrics)
            if result.interpretation:
                lines.append(result.interpretation)
            for section in result.detail_sections:
                lines.append(f"\n**{section.title}**")
                lines.extend(f"- {item}" for item in section.items)
    if turn.source_cards:
        lines.append("\n### Sources")
        lines.extend(f"- {source.title}: {source.url}" for source in turn.source_cards)
    return _markdown("\n".join(lines))


def _analysis_cells(session: AnalysisSession) -> list[dict[str, Any]]:
    directory = session.root / "records" / "attempt"
    if not directory.exists():
        return []
    attempts = []
    for path in directory.glob(f"{session.run_id}_attempt_*.json"):
        attempt = session.records.read("attempt", path.stem)
        if (attempt.get("run_id") == session.run_id
                and attempt.get("attempt_id") == path.stem
                and attempt.get("status") == "completed"):
            attempts.append(attempt)
    attempts.sort(key=lambda item: (item.get("created_at") or "", item["attempt_id"]))
    skipped = max(0, len(attempts) - NOTEBOOK_ATTEMPT_LIMIT)
    attempts = attempts[-NOTEBOOK_ATTEMPT_LIMIT:]
    cells: list[dict[str, Any]] = []
    if skipped:
        cells.append(_markdown(f"{skipped} earlier completed attempts were omitted from this notebook."))
    code_bytes = 0
    figure_bytes = 0
    figure_count = 0
    for attempt in attempts:
        code_id = attempt.get("code_version")
        if not isinstance(code_id, str):
            continue
        # CodeStore checks the run ID, file location, length and SHA-256 here.
        source = session.codes.get_path(code_id).read_text(encoding="utf-8")
        encoded_size = len(source.encode("utf-8"))
        if code_bytes + encoded_size > NOTEBOOK_CODE_LIMIT:
            cells.append(_markdown("Additional analysis code was omitted because the notebook code limit was reached."))
            break
        code_bytes += encoded_size
        index = session.list_results(attempt["attempt_id"], offset=0, limit=20)
        entries = list(index["results"])
        for offset in range(20, min(index["total"], NOTEBOOK_RESULT_LIMIT), 20):
            entries.extend(session.list_results(
                attempt["attempt_id"], offset=offset,
                limit=min(20, NOTEBOOK_RESULT_LIMIT - offset),
            )["results"])
        lines = [f"## Analysis attempt {attempt['attempt_id']}",
                 f"Saved results: {index['total']}. The first {len(entries)} result references are listed below."]
        for entry in entries:
            lines.append(
                f"- {entry['stage']}: {entry.get('summary') or entry.get('status')} "
                f"(artifact `{entry.get('artifact_id') or 'none'}`; "
                f"time {entry.get('time') or 'unspecified'}; "
                f"depth {entry.get('depth') if entry.get('depth') is not None else 'unspecified'})"
            )
        cells.extend([_markdown("\n".join(lines)), _code(
            source, code_id=code_id, attempt_id=attempt["attempt_id"],
        )])
        for entry in entries:
            if figure_count >= NOTEBOOK_FIGURE_LIMIT or figure_bytes >= NOTEBOOK_FIGURE_BYTES:
                break
            artifact_id = entry.get("artifact_id")
            if not artifact_id:
                continue
            metadata = session.read_artifact(artifact_id)
            if (metadata.get("status") != "completed"
                    or metadata.get("kind") != "image_png"
                    or metadata.get("attempt_id") != attempt["attempt_id"]):
                continue
            png = session.artifacts.read_image(artifact_id)
            if figure_bytes + len(png) > NOTEBOOK_FIGURE_BYTES:
                continue
            figure_count += 1
            figure_bytes += len(png)
            filename = f"figure-{figure_count}.png"
            cell = _markdown(f"### {entry['stage']} figure\n\n![{entry['stage']}](attachment:{filename})")
            cell["attachments"] = {filename: {"image/png": base64.b64encode(png).decode("ascii")}}
            cells.append(cell)
    return cells


class ReportMetricPayload(BaseModel):
    label: str
    value: str


class ReportDetailSectionPayload(BaseModel):
    title: str
    items: List[str] = Field(default_factory=list)


class ReportResultCardPayload(BaseModel):
    title: str
    headline: str = ""
    description: str = ""
    metrics: List[ReportMetricPayload] = Field(default_factory=list)
    interpretation: Optional[str] = None
    detail_sections: List[ReportDetailSectionPayload] = Field(default_factory=list)


class ReportStepCardPayload(BaseModel):
    human_label: str
    technical_label: str = ""
    status: str
    interpretation: Optional[str] = None
    error: Optional[str] = None
    results: List[ReportResultCardPayload] = Field(default_factory=list)


class ReportFindingPayload(BaseModel):
    title: str
    evidence: List[str] = Field(default_factory=list)


class ReportSourcePayload(BaseModel):
    title: str
    source: str = ""
    url: str = ""
    short_snippet: str = ""
    why_it_matters: str = ""


class ReportFigurePayload(BaseModel):
    title: Optional[str] = None
    mime_type: str
    data_base64: str
    width: Optional[int] = None
    height: Optional[int] = None


class ReportTurnPayload(BaseModel):
    user_query: str = ""
    assistant_status: str
    assistant_summary: str = ""
    plan_steps: List[Dict[str, Any]] = Field(default_factory=list)
    step_cards: List[ReportStepCardPayload] = Field(default_factory=list)
    findings: List[ReportFindingPayload] = Field(default_factory=list)
    source_cards: List[ReportSourcePayload] = Field(default_factory=list)
    primary_figure: Optional[ReportFigurePayload] = None


class ConversationReportRequest(BaseModel):
    conversation_id: Optional[str] = None
    exported_at: str
    dataset_info: Dict[str, Any] = Field(default_factory=dict)
    turns: List[ReportTurnPayload] = Field(default_factory=list)


def export_report_response(
    request: ConversationReportRequest, session: AnalysisSession | None = None,
) -> Response:
    cells = [_markdown("# OceanMind analysis\n\nExported: " + request.exported_at)]
    cells.extend(_turn_cell(turn, number) for number, turn in enumerate(request.turns, 1))
    if session is not None:
        cells.extend(_analysis_cells(session))
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    safe_date = re.sub(r"[^A-Za-z0-9._-]", "-", request.exported_at)[:80]
    filename = f"oceanmind-conversation-{safe_date}.ipynb"
    return Response(
        content=json.dumps(notebook, ensure_ascii=False, allow_nan=False),
        media_type="application/x-ipynb+json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
