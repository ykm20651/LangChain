from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from datetime import datetime


def save_report_pdf(path: str, title: str, question: str, answer: str):
    """
    보고서 텍스트를 PDF로 보기 좋게 저장하는 함수.
    Markdown 비슷한 형식을 유지하며, 한글 폰트 적용.
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
    # 기존 스타일 이름과 충돌 방지 → "Custom..." 으로 변경
    styles.add(ParagraphStyle(name="CustomTitle", fontName="HYSMyeongJo-Medium", fontSize=18, leading=24, spaceAfter=12, alignment=1))
    styles.add(ParagraphStyle(name="CustomHeading1", fontName="HYSMyeongJo-Medium", fontSize=14, leading=20, spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="CustomHeading2", fontName="HYSMyeongJo-Medium", fontSize=12, leading=18, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="CustomBody", fontName="HYSMyeongJo-Medium", fontSize=11, leading=16, spaceAfter=6))
    styles.add(ParagraphStyle(name="CustomCode", fontName="HYSMyeongJo-Medium", fontSize=10, leading=14, backColor="#f4f4f4"))

    content = []

    # 🔹 제목
    content.append(Paragraph(title, styles["CustomTitle"]))
    content.append(Paragraph(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["CustomBody"]))
    content.append(Spacer(1, 12))

    # 🔹 질문 섹션
    content.append(Paragraph("<b>[질문]</b>", styles["CustomHeading1"]))
    for line in question.split("\n"):
        if line.strip():
            content.append(Paragraph(line.strip(), styles["CustomBody"]))

    content.append(Spacer(1, 12))

    # 🔹 답변 섹션
    content.append(Paragraph("<b>[답변]</b>", styles["CustomHeading1"]))

    # Markdown-like 처리
    for line in answer.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            content.append(Paragraph(line[2:], styles["CustomHeading1"]))
        elif line.startswith("## "):
            content.append(Paragraph(line[3:], styles["CustomHeading2"]))
        elif line.startswith("- "):
            content.append(Paragraph(f"• {line[2:]}", styles["CustomBody"]))
        elif line.startswith("**"):
            content.append(Paragraph(f"<b>{line}</b>", styles["CustomBody"]))
        else:
            content.append(Paragraph(line, styles["CustomBody"]))

    content.append(Spacer(1, 20))
    content.append(Paragraph("──────────────────────────────", styles["CustomBody"]))
    content.append(Paragraph("본 보고서는 LangChain 기반 AI 분석 결과입니다.", styles["CustomBody"]))

    # ✅ PDF 빌드
    doc.build(content)
