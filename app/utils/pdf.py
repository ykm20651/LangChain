"""
pdf.py
- 보고서 텍스트를 PDF로 저장하는 유틸리티
- LangChain 결과 텍스트를 사람이 읽기 좋은 형식으로 렌더링
"""

import os
import textwrap
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def _draw_wrapped_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width_chars: int = 85,
    leading: int = 16,
) -> float:
    """
    긴 텍스트를 PDF 페이지 폭에 맞게 자동 줄바꿈하여 출력.
    Returns: 마지막 y 좌표 (다음 줄 그릴 위치)
    """
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width_chars) or [""])
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
        if y < 2 * cm:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = A4[1] - 2 * cm
    return y


def save_report_pdf(
    path: str,
    title: str,
    question: str,
    answer: str,
    font: str = "Helvetica",
):
    """
    LangChain LLM이 생성한 보고서를 PDF 파일로 저장.
    Args:
        path: 저장 경로 (e.g., ./storage/reports/{task_id}.pdf)
        title: 보고서 제목
        question: 입력 프롬프트 (질문 또는 사고정보)
        answer: LLM의 생성 결과 (보고서 본문)
    """
    # 디렉토리 없으면 생성
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # PDF 캔버스 초기화
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    x = 2 * cm
    y = h - 2 * cm

    # 보고서 제목
    c.setFont(f"{font}-Bold", 16)
    c.drawString(x, y, title)
    y -= 22

    # 생성일시
    c.setFont(font, 10)
    c.drawString(x, y, f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 18

    # 질문/입력 데이터
    c.setFont(f"{font}-Bold", 12)
    c.drawString(x, y, "[입력 데이터]")
    y -= 16
    c.setFont(font, 11)
    y = _draw_wrapped_text(c, question, x, y)

    # 구분선
    y -= 10
    c.line(x, y, w - 2 * cm, y)
    y -= 18

    # 답변 / 보고서 본문
    c.setFont(f"{font}-Bold", 12)
    c.drawString(x, y, "[생성된 보고서]")
    y -= 18
    c.setFont(font, 11)
    y = _draw_wrapped_text(c, answer, x, y)

    # PDF 완료
    c.showPage()
    c.save()
