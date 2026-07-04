from __future__ import annotations

import base64
import html
import io
import logging
import textwrap
from typing import Any, Dict, List, Optional

from PIL import Image

LOGGER = logging.getLogger(__name__)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _wrap_lines(text: str, width: int = 100) -> List[str]:
    normalized = _as_text(text)
    if not normalized:
        return []
    wrapped: List[str] = []
    for paragraph in normalized.splitlines() or [""]:
        stripped = paragraph.strip()
        if not stripped:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(stripped, width=width) or [""])
    return wrapped


def build_report_outline_lines(report: Dict[str, Any]) -> List[str]:
    lines = ["OceanMind Conversation Report"]
    conversation_id = _as_text(report.get("conversation_id"))
    exported_at = _as_text(report.get("exported_at"))
    if conversation_id:
        lines.append(f"Conversation ID: {conversation_id}")
    if exported_at:
        lines.append(f"Exported At: {exported_at}")

    dataset_info = report.get("dataset_info") or {}
    dataset_name = _as_text(dataset_info.get("name"))
    if dataset_name:
        lines.append(f"Dataset: {dataset_name}")
    dataset_description = _as_text(dataset_info.get("description"))
    if dataset_description:
        lines.extend(_wrap_lines(f"Dataset Description: {dataset_description}", width=96))

    turns = report.get("turns") or []
    for index, turn in enumerate(turns, start=1):
        lines.append("")
        lines.append(f"Turn {index}")
        user_query = _as_text(turn.get("user_query"))
        if user_query:
            lines.extend(_wrap_lines(f"Question: {user_query}", width=96))
        assistant_status = _as_text(turn.get("assistant_status"))
        if assistant_status:
            lines.append(f"Assistant Status: {assistant_status}")
        summary = _as_text(turn.get("assistant_summary"))
        if summary:
            lines.extend(_wrap_lines(f"Summary: {summary}", width=96))

        for step in turn.get("plan_steps") or []:
            label = _as_text(step.get("humanLabel") or step.get("tool"))
            status = _as_text(step.get("status"))
            if label:
                lines.append(f"Plan Step: {label} [{status}]".strip())

        for step_card in turn.get("step_cards") or []:
            human_label = _as_text(step_card.get("human_label"))
            if human_label:
                lines.append(f"Executed Step: {human_label}")
            interpretation = _as_text(step_card.get("interpretation"))
            if interpretation:
                lines.extend(_wrap_lines(f"Interpretation: {interpretation}", width=92))
            for result in step_card.get("results") or []:
                title = _as_text(result.get("title"))
                headline = _as_text(result.get("headline"))
                if title:
                    lines.append(f"Result: {title}")
                if headline:
                    lines.extend(_wrap_lines(headline, width=92))
                for metric in result.get("metrics") or []:
                    label = _as_text(metric.get("label"))
                    value = _as_text(metric.get("value"))
                    if label or value:
                        lines.append(f"Metric: {label}: {value}".strip(": "))
                for section in result.get("detail_sections") or []:
                    section_title = _as_text(section.get("title"))
                    if section_title:
                        lines.append(f"Detail Section: {section_title}")
                    for item in section.get("items") or []:
                        item_text = _as_text(item)
                        if item_text:
                            lines.extend(_wrap_lines(f"- {item_text}", width=88))

        for finding in turn.get("findings") or []:
            finding_title = _as_text(finding.get("title"))
            if finding_title:
                lines.extend(_wrap_lines(f"Finding: {finding_title}", width=92))
            for evidence in finding.get("evidence") or []:
                evidence_text = _as_text(evidence)
                if evidence_text:
                    lines.extend(_wrap_lines(f"- {evidence_text}", width=88))

        for source in turn.get("source_cards") or []:
            title = _as_text(source.get("title"))
            url = _as_text(source.get("url"))
            if title:
                lines.append(f"Source: {title}")
            if url:
                lines.append(f"URL: {url}")

    return lines


def _pdf_metadata_subject(report: Dict[str, Any]) -> str:
    turns = report.get("turns") or []
    if not turns:
        return "OceanMind conversation report"
    first_turn = turns[0]
    question = _as_text(first_turn.get("user_query"))
    summary = _as_text(first_turn.get("assistant_summary"))
    fragments = [fragment for fragment in [question, summary] if fragment]
    return " | ".join(fragments)[:240] or "OceanMind conversation report"


def _decode_snapshot_image(snapshot: Optional[Dict[str, Any]]) -> Optional[Image.Image]:
    if not snapshot:
        return None
    payload = _as_text(snapshot.get("data_base64"))
    if not payload:
        return None
    try:
        raw = base64.b64decode(payload)
    except Exception:
        return None
    image = Image.open(io.BytesIO(raw))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _append_plaintext_trailer(pdf_bytes: bytes, report: Dict[str, Any]) -> bytes:
    lines = build_report_outline_lines(report)
    trailer_lines = ["", "% OceanMind Report Outline"]
    for line in lines[:80]:
        trailer_lines.append(f"% {line}")
    trailer = "\n".join(trailer_lines).encode("utf-8", "ignore")
    return pdf_bytes + trailer


def _build_pdf_with_reportlab(report: Dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Image as ReportlabImage
    from reportlab.platypus import HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    from reportlab.platypus import Table, TableStyle

    page_buffer = io.BytesIO()
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    base_font = "STSong-Light"
    navy = colors.HexColor("#12314f")
    slate = colors.HexColor("#42566f")
    muted = colors.HexColor("#6d7c8f")
    line = colors.HexColor("#d7dee8")
    wash = colors.HexColor("#f4f7fb")
    warm = colors.HexColor("#fbf8ef")
    accent = colors.HexColor("#1d6f92")

    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=10.5,
            leading=14,
            spaceAfter=5,
            textColor=slate,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading",
            parent=styles["Heading2"],
            fontName=base_font,
            fontSize=16,
            leading=20,
            spaceBefore=8,
            spaceAfter=7,
            textColor=navy,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName=base_font,
            fontSize=25,
            leading=30,
            alignment=TA_CENTER,
            spaceAfter=6,
            textColor=navy,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            spaceAfter=14,
            textColor=muted,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportEyebrow",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=8.5,
            leading=11,
            textColor=accent,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSmall",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=9,
            leading=12,
            textColor=muted,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeaderCell",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=9,
            leading=12,
            textColor=colors.white,
            spaceAfter=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportCaption",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=muted,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportResultTitle",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=11.5,
            leading=15,
            textColor=navy,
            spaceBefore=3,
            spaceAfter=4,
        )
    )

    doc = SimpleDocTemplate(
        page_buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="OceanMind Conversation Report",
        author="OceanMind",
        subject=_pdf_metadata_subject(report),
        pageCompression=0,
    )

    def safe(value: Any) -> str:
        return html.escape(_as_text(value)).replace("\n", "<br/>")

    def para(value: Any, style_name: str = "ReportBody") -> Paragraph:
        return Paragraph(safe(value), styles[style_name])

    def labelled(label: str, value: Any, style_name: str = "ReportBody") -> Paragraph:
        return Paragraph(f'<font color="#12314f">{html.escape(label)}:</font> {safe(value)}', styles[style_name])

    def header_footer(canvas, built_doc) -> None:
        canvas.saveState()
        page_width, page_height = A4
        page_number = canvas.getPageNumber()
        canvas.setFont(base_font, 8)
        canvas.setFillColor(muted)
        if page_number > 1:
            canvas.drawString(built_doc.leftMargin, page_height - 10 * mm, "OceanMind Conversation Report")
            canvas.setStrokeColor(line)
            canvas.setLineWidth(0.5)
            canvas.line(built_doc.leftMargin, page_height - 12 * mm, page_width - built_doc.rightMargin, page_height - 12 * mm)
        canvas.drawRightString(page_width - built_doc.rightMargin, 9 * mm, f"Page {page_number}")
        canvas.restoreState()

    def metadata_table() -> Optional[Table]:
        dataset_info = report.get("dataset_info") or {}
        rows: List[List[Any]] = []
        for label, value in (
            ("Conversation", report.get("conversation_id")),
            ("Exported", report.get("exported_at")),
            ("Dataset", dataset_info.get("name")),
            ("Dataset description", dataset_info.get("description")),
        ):
            text = _as_text(value)
            if text:
                rows.append([para(label, "ReportSmall"), para(text, "ReportBody")])
        if not rows:
            return None
        table = Table(rows, colWidths=[38 * mm, doc.width - 38 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), warm),
                    ("BOX", (0, 0), (-1, -1), 0.6, line),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e7dcc8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return table

    def callout(title: str, body: Any, background=wash) -> Optional[Table]:
        body_text = _as_text(body)
        if not body_text:
            return None
        content = [
            para(title.upper(), "ReportEyebrow"),
            para(body_text, "ReportBody"),
        ]
        table = Table([[content]], colWidths=[doc.width])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), background),
                    ("BOX", (0, 0), (-1, -1), 0.5, line),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        return table

    def metric_table(metrics: List[Dict[str, Any]]) -> Optional[Table]:
        rows: List[List[Any]] = []
        for metric in metrics:
            label = _as_text(metric.get("label"))
            value = _as_text(metric.get("value"))
            if label or value:
                rows.append([para(label, "ReportSmall"), para(value, "ReportBody")])
        if not rows:
            return None
        table = Table(rows, colWidths=[doc.width * 0.34, doc.width * 0.66])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("BOX", (0, 0), (-1, -1), 0.4, line),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#edf1f5")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return table

    def plan_table(plan_steps: List[Dict[str, Any]]) -> Optional[Table]:
        rows: List[List[Any]] = [
            [para("#", "ReportHeaderCell"), para("Plan step", "ReportHeaderCell"), para("Status", "ReportHeaderCell")]
        ]
        for step_index, step in enumerate(plan_steps, start=1):
            label = _as_text(step.get("humanLabel") or step.get("tool"))
            status = _as_text(step.get("status"))
            if label:
                rows.append([para(str(step_index), "ReportSmall"), para(label, "ReportBody"), para(status, "ReportSmall")])
        if len(rows) == 1:
            return None
        table = Table(rows, colWidths=[14 * mm, doc.width - 44 * mm, 30 * mm], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), navy),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("BOX", (0, 0), (-1, -1), 0.5, line),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e7edf4")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return table

    def figure_block(snapshot: Optional[Dict[str, Any]], index: int) -> List[Any]:
        image = _decode_snapshot_image(snapshot)
        if image is None:
            return []
        width, height = image.size
        if width <= 0 or height <= 0:
            return []
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        max_width = doc.width
        max_height = 116 * mm
        scale = min(max_width / width, max_height / height)
        draw_width = width * scale
        draw_height = height * scale
        report_image = ReportlabImage(image_buffer, width=draw_width, height=draw_height)
        report_image.hAlign = "CENTER"
        title = _as_text((snapshot or {}).get("title")) or "Primary figure"
        image_table = Table([[report_image]], colWidths=[doc.width])
        image_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.6, line),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return [
            Spacer(1, 4),
            KeepTogether(
                [
                    Paragraph(f"Figure {index}. {safe(title)}", styles["ReportCaption"]),
                    image_table,
                ]
            ),
            Spacer(1, 10),
        ]

    def add_result(story_items: List[Any], result: Dict[str, Any]) -> None:
        title = _as_text(result.get("title"))
        if title:
            story_items.append(Paragraph(safe(title), styles["ReportResultTitle"]))
        for field in ("headline", "description", "interpretation"):
            text = _as_text(result.get(field))
            if text:
                story_items.append(para(text, "ReportBody"))
        metrics = metric_table(result.get("metrics") or [])
        if metrics is not None:
            story_items.append(metrics)
            story_items.append(Spacer(1, 5))
        for section in result.get("detail_sections") or []:
            section_title = _as_text(section.get("title"))
            if section_title:
                story_items.append(labelled("Detail", section_title, "ReportSmall"))
            for item in section.get("items") or []:
                item_text = _as_text(item)
                if item_text:
                    story_items.append(Paragraph(f"&bull; {safe(item_text)}", styles["ReportBody"]))

    story: List[Any] = [
        Spacer(1, 22),
        Paragraph("OceanMind Conversation Report", styles["ReportTitle"]),
        Paragraph("Conversation export with analysis context, results, and figures", styles["ReportSubtitle"]),
    ]

    metadata = metadata_table()
    if metadata is not None:
        story.append(metadata)
        story.append(Spacer(1, 16))

    turns = report.get("turns") or []
    if turns:
        story.append(Paragraph("Report Contents", styles["ReportHeading"]))
        story.append(para(f"{len(turns)} assistant turn(s) exported. Figures are placed inside bounded figure panels and scaled to the page.", "ReportBody"))
        story.append(PageBreak())

    for index, turn in enumerate(turns, start=1):
        story.append(Paragraph(f"Turn {index}", styles["ReportHeading"]))
        user_query = _as_text(turn.get("user_query"))
        question_box = callout("Question", user_query, background=warm)
        if question_box is not None:
            story.append(question_box)
            story.append(Spacer(1, 7))
        assistant_status = _as_text(turn.get("assistant_status"))
        if assistant_status:
            story.append(labelled("Status", assistant_status, "ReportSmall"))
        assistant_summary = _as_text(turn.get("assistant_summary"))
        summary_box = callout("Summary", assistant_summary)
        if summary_box is not None:
            story.append(summary_box)
            story.append(Spacer(1, 7))

        story.extend(figure_block(turn.get("primary_figure"), index))

        plan = plan_table(turn.get("plan_steps") or [])
        if plan is not None:
            story.append(Paragraph("Execution Plan", styles["ReportResultTitle"]))
            story.append(plan)
            story.append(Spacer(1, 9))

        for step_card in turn.get("step_cards") or []:
            story.append(HRFlowable(width="100%", thickness=0.4, color=line, spaceBefore=5, spaceAfter=8))
            human_label = _as_text(step_card.get("human_label")) or _as_text(step_card.get("technical_label")) or "Executed step"
            status = _as_text(step_card.get("status"))
            story.append(Paragraph(safe(human_label), styles["ReportResultTitle"]))
            if status:
                story.append(labelled("Step status", status, "ReportSmall"))
            if step_card.get("error"):
                story.append(labelled("Error", step_card.get("error"), "ReportBody"))
            if step_card.get("interpretation"):
                story.append(labelled("Interpretation", step_card.get("interpretation"), "ReportBody"))
            for result in step_card.get("results") or []:
                add_result(story, result)
            story.append(Spacer(1, 5))

        findings = turn.get("findings") or []
        if findings:
            story.append(Paragraph("Scientific Findings", styles["ReportResultTitle"]))
            for finding in findings:
                title = _as_text(finding.get("title"))
                if title:
                    story.append(Paragraph(f"&bull; {safe(title)}", styles["ReportBody"]))
                for evidence in finding.get("evidence") or []:
                    evidence_text = _as_text(evidence)
                    if evidence_text:
                        story.append(Paragraph(f"&nbsp;&nbsp;- {safe(evidence_text)}", styles["ReportSmall"]))

        sources = turn.get("source_cards") or []
        if sources:
            story.append(Paragraph("Sources", styles["ReportResultTitle"]))
            for source in sources:
                source_title = _as_text(source.get("title"))
                if source_title:
                    story.append(labelled("Source", source_title, "ReportSmall"))
                url = _as_text(source.get("url"))
                if url:
                    story.append(para(url, "ReportSmall"))

        if index != len(turns):
            story.append(PageBreak())

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return page_buffer.getvalue()


def _build_pdf_with_matplotlib(report: Dict[str, Any]) -> bytes:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    def save_text_page(pdf, title: str, lines: List[str]) -> None:
        max_lines = 44
        chunks = [lines[index:index + max_lines] for index in range(0, len(lines), max_lines)] or [[]]
        for chunk_index, chunk in enumerate(chunks, start=1):
            figure = Figure(figsize=(8.27, 11.69))
            ax = figure.subplots()
            ax.axis("off")
            page_title = title if len(chunks) == 1 else f"{title} ({chunk_index}/{len(chunks)})"
            ax.text(0.06, 0.96, page_title, fontsize=16, fontweight="bold", va="top", color="#12314f")
            ax.text(0.06, 0.89, "\n".join(chunk), fontsize=9.2, va="top", linespacing=1.28, color="#42566f")
            pdf.savefig(figure)

    def save_image_page(pdf, title: str, image: Image.Image) -> None:
        figure = Figure(figsize=(8.27, 11.69))
        title_ax = figure.add_axes([0.06, 0.91, 0.88, 0.06])
        title_ax.axis("off")
        title_ax.text(0.0, 0.85, title, fontsize=14, fontweight="bold", va="top", color="#12314f")
        image_ax = figure.add_axes([0.08, 0.12, 0.84, 0.74])
        image_ax.imshow(image)
        image_ax.axis("off")
        pdf.savefig(figure)

    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "OceanMind Conversation Report"
        metadata["Author"] = "OceanMind"
        metadata["Subject"] = _pdf_metadata_subject(report)

        overview_lines = build_report_outline_lines({
            "conversation_id": report.get("conversation_id"),
            "exported_at": report.get("exported_at"),
            "dataset_info": report.get("dataset_info"),
            "turns": [],
        })
        save_text_page(pdf, "OceanMind Conversation Report", overview_lines[1:])

        turns = report.get("turns") or []
        for index, turn in enumerate(turns, start=1):
            lines = [f"Turn {index}"]
            lines.extend(build_report_outline_lines({"turns": [turn]})[2:])
            save_text_page(pdf, f"Turn {index}", lines)

            image = _decode_snapshot_image(turn.get("primary_figure"))
            if image is not None:
                title = _as_text((turn.get("primary_figure") or {}).get("title")) or f"Turn {index} figure"
                save_image_page(pdf, f"Figure {index}. {title}", image)

    return pdf_buffer.getvalue()


def build_conversation_report_pdf(report: Dict[str, Any]) -> bytes:
    try:
        pdf_bytes = _build_pdf_with_reportlab(report)
    except Exception:
        LOGGER.warning("ReportLab PDF export failed; falling back to Matplotlib PDF export.", exc_info=True)
        pdf_bytes = _build_pdf_with_matplotlib(report)
    return _append_plaintext_trailer(pdf_bytes, report)
