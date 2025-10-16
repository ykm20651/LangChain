from typing import Optional, Dict, Any, List
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from datetime import datetime

def _kv_table_from_incident(incident_data: Optional[Dict[str, Any]]) -> Optional[Table]:
    if not incident_data:
        return None

    # 한국어 라벨 매핑
    label_map = {
        "incident_type": "사고유형",
        "description": "설명",
        "location": "발생위치",
        "report_type": "보고서 유형",
        "language": "언어",
    }

    data: List[List[str]] = [["항목", "값"]]
    order = ["incident_type", "description", "location", "report_type", "language"]
    for k in order:
        v = incident_data.get(k)
        if v is not None and str(v).strip() != "":
            data.append([label_map.get(k, k), str(v)])

    if len(data) == 1:
        return None

    table = Table(data, colWidths=[100, 370])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F0F0F0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,-1), 'HYSMyeongJo-Medium'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('INNERGRID', (0,0), (-1,-1), 0.4, colors.grey),
        ('BOX', (0,0), (-1,-1), 0.6, colors.black),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
    ]))
    return table


def save_report_pdf(
    path: str,
    title: str,
    answer: str,
    incident_data: Optional[Dict[str, Any]] = None,
    subtitle: Optional[str] = None,
):
    """
    보고서 텍스트를 PDF로 저장.
    - 상단: 제목 / 생성시각
    - 기본정보 표: incident_data로 구성
    - 본문: LLM이 생성한 보험 청구서 섹션 텍스트 (Markdown 유사 규칙 그대로)
    """
    # ✅ 한글 폰트 등록
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))

    # ✅ 문서 객체 생성
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=50,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CustomTitle", fontName="HYSMyeongJo-Medium", fontSize=18, leading=24, spaceAfter=12, alignment=1))
    styles.add(ParagraphStyle(name="CustomSub", fontName="HYSMyeongJo-Medium", fontSize=11, leading=16, spaceAfter=4, alignment=1, textColor=colors.grey))
    styles.add(ParagraphStyle(name="CustomHeading1", fontName="HYSMyeongJo-Medium", fontSize=14, leading=20, spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="CustomHeading2", fontName="HYSMyeongJo-Medium", fontSize=12, leading=18, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="CustomBody", fontName="HYSMyeongJo-Medium", fontSize=11, leading=16, spaceAfter=6))
    styles.add(ParagraphStyle(name="CustomCode", fontName="HYSMyeongJo-Medium", fontSize=10, leading=14, backColor="#f4f4f4"))

    content = []

    # 🔹 제목/서브타이틀/생성시각
    content.append(Paragraph(title, styles["CustomTitle"]))
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    content.append(Paragraph(subtitle or "보험 청구 보고서 (자동 생성)", styles["CustomSub"]))
    content.append(Paragraph(f"Generated at: {ts}", styles["CustomSub"]))
    content.append(Spacer(1, 10))

    # 🔹 기본정보 표 (incident_data)
    info_table = _kv_table_from_incident(incident_data)
    if info_table:
        content.append(Paragraph("<b>기본정보</b>", styles["CustomHeading1"]))
        content.append(info_table)
        content.append(Spacer(1, 8))

    # 🔹 본문
    content.append(Paragraph("<b>본문</b>", styles["CustomHeading1"]))

    # Markdown-like 처리 (제목/소제목/리스트/기타)
    for raw in answer.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# "):
            content.append(Paragraph(line[2:], styles["CustomHeading1"]))
        elif line.startswith("## "):
            content.append(Paragraph(line[3:], styles["CustomHeading2"]))
        elif line.startswith("- "):
            content.append(Paragraph(f"• {line[2:]}", styles["CustomBody"]))
        elif line.startswith("**"):
            content.append(Paragraph(f"<b>{line.strip('*')}</b>", styles["CustomBody"]))
        else:
            content.append(Paragraph(line, styles["CustomBody"]))

    content.append(Spacer(1, 16))
    content.append(Paragraph("──────────────────────────────", styles["CustomBody"]))
    content.append(Paragraph("본 보고서는 LangChain 기반 RAG+LLM으로 자동 생성되었습니다.", styles["CustomBody"]))

    # ✅ PDF 빌드
    doc.build(content)
