from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi.responses import Response
from pydantic import BaseModel, Field

from packages.reporting import build_conversation_report_pdf


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


def export_report_response(request: ConversationReportRequest) -> Response:
    report_payload = request.model_dump(mode="json")
    pdf_bytes = build_conversation_report_pdf(report_payload)
    filename = f"oceanmind-conversation-{request.exported_at.replace(':', '-').replace('.', '-')}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
