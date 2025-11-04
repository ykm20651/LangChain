from typing import Optional, Dict, Any, List
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os


def _register_korean_font() -> str:
    """Windows/macOS/Linux 환경에 맞춰 한글 폰트를 안전하게 등록"""
    try:
        if os.name == "nt" and os.path.exists(r"C:\Windows\Fonts\malgun.ttf"):
            pdfmetrics.registerFont(TTFont("Malgun", r"C:\Windows\Fonts\malgun.ttf"))
            return "Malgun"
        elif os.path.exists("/System/Library/Fonts/AppleSDGothicNeo.ttc"):
            pdfmetrics.registerFont(TTFont("AppleSDGothicNeo", "/System/Library/Fonts/AppleSDGothicNeo.ttc"))
            return "AppleSDGothicNeo"
        elif os.path.exists("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
            pdfmetrics.registerFont(TTFont("NotoSansCJK", "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"))
            return "NotoSansCJK"
    except Exception as e:
        print(f"[WARN] 폰트 등록 실패: {e}")
    return "Helvetica"


def _kv_table_from_incident(incident_data: Optional[Dict[str, Any]], font_name: str) -> Optional[Table]:
    if not incident_data:
        return None

    # Paragraph 스타일
    cell_style = ParagraphStyle(
        "Cell",
        fontName=font_name,
        fontSize=10,
        leading=14,
        wordWrap="CJK",
    )

    label_map = {
        "incident_type": "사고유형",
        "description": "설명",
        "location": "발생위치",
        "report_type": "보고서 유형",
        "language": "언어",
    }

    # ✅ 데이터 구성 (Paragraph로 감싸서 줄바꿈 가능)
    data: List[List[Any]] = []
    data.append([Paragraph("<b>항목</b>", cell_style), Paragraph("<b>값</b>", cell_style)])

    order = ["incident_type", "description", "location", "report_type", "language"]
    for k in order:
        v = incident_data.get(k)
        if v is not None and str(v).strip() != "":
            key_p = Paragraph(label_map.get(k, k), cell_style)
            val_p = Paragraph(str(v), cell_style)
            data.append([key_p, val_p])

    # ✅ 표 생성
    table = Table(data, colWidths=[100, 380])
    table.setStyle(TableStyle([
        # 헤더 스타일
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#D9D9D9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),

        # 셀 스타일
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 1), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 10),

        # 테두리 및 그리드
        ('INNERGRID', (0, 0), (-1, -1), 0.4, colors.HexColor("#A0A0A0")),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.black),
    ]))
    return table


def save_report_pdf(path, title, answer, incident_data, subtitle=""):
    """PDF 보고서 저장 (표 + 본문 + 디자인 정돈 버전)"""
    font_name = _register_korean_font()

    doc = SimpleDocTemplate(path, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()

    # 제목 / 본문 스타일
    title_style = ParagraphStyle(
        'Title',
        fontName=font_name,
        fontSize=18,
        leading=22,
        alignment=1,  # Center
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        fontName=font_name,
        fontSize=10,
        leading=12,
        alignment=1,
        textColor=colors.gray,
    )
    normal = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        wordWrap='CJK',
    )
    section_title = ParagraphStyle(
        'Heading2',
        fontName=font_name,
        fontSize=13,
        leading=18,
        spaceBefore=10,
        spaceAfter=6,
    )

    # 제목
    elements.append(Paragraph(title, title_style))
    elements.append(Paragraph(subtitle, subtitle_style))
    elements.append(Spacer(1, 20))

    # ✅ 기본정보 섹션
    elements.append(Paragraph("기본정보", section_title))
    table = _kv_table_from_incident(incident_data, font_name)
    elements.append(table)
    elements.append(Spacer(1, 20))

    # ✅ 본문 (개행 포함)
    elements.append(Paragraph(answer.replace("\n", "<br/>"), normal))

    doc.build(elements)
