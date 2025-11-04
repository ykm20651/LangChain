"""
InsuranceReportService (최종 안정화 버전)
- RAG + Pydantic + SecretStr 완전 대응
"""

from __future__ import annotations
import os
import json
from typing import Optional, Dict, Any
from pydantic.json import pydantic_encoder
from pydantic import SecretStr

from app.chains.rag_chain import RAGChain
from app.chains.report_chain import ReportChain
from app.utils.pdf import save_report_pdf
from app.chains.report_structured_chain import StructuredReportChain
from app.models.report_models import InsurancePydanticReportResponse


# ---------------------------------------------------------------------
# 안전 직렬화 유틸 (최종 완성본)
# ---------------------------------------------------------------------
def to_safe_json(obj):
    """
    dict / list / pydantic / SecretStr / callable / class / LangChain Runnable 등
    모든 타입을 안전하게 직렬화 가능한 형태로 변환
    """
    from pydantic import SecretStr

    # 클래스 자체
    if isinstance(obj, type):
        return str(obj)

    # LangChain Runnable 계열 (예: RunnableLambda, RunnableSequence 등)
    if "Runnable" in obj.__class__.__name__:
        return f"<{obj.__class__.__name__}>"

    # SecretStr (보안 문자열)
    if isinstance(obj, SecretStr):
        try:
            return obj.get_secret_value()
        except Exception:
            return "********"

    # dict
    if isinstance(obj, dict):
        return {k: to_safe_json(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [to_safe_json(i) for i in obj]

    # Pydantic 객체 (BaseModel)
    if hasattr(obj, "model_dump") and not isinstance(obj, type):
        try:
            return to_safe_json(obj.model_dump())
        except Exception:
            return str(obj)

    # callable (함수/메서드)
    if callable(obj):
        return f"<function {obj.__name__}>"

    # 기본형
    return obj



# ---------------------------------------------------------------------
class InsuranceReportService:
    """
    사건 데이터(incident_data) → (선택) RAG 컨텍스트 → 사고유형별 템플릿 보고서 → PDF 저장
    """

    def __init__(self):
        self.reports_dir = os.getenv("REPORTS_DIR", "./storage/reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        self.default_collection = os.getenv("RAG_DEFAULT_COLLECTION", "marine_laws")

    # ---------------------------------------------------------------------
    def _build_seed_query(self, incident_data: Dict[str, Any], incident_type: str) -> str:
        desc = str(incident_data.get("description", incident_data.get("title", "")) or "").strip()
        itype = (incident_type or incident_data.get("incident_type", "generic")).lower()

        hints = {
            "fire": "선박 화재사고 처리 기준, 보험 약관 화재조항, 선박안전법 관련 조항",
            "oil_spill": "유류유출 방제 기준, 해양환경관리법, MARPOL 협약, 보험 약관 오염조항",
            "collision": "선박 충돌 관련 해사안전법, 국제충돌예방규칙 COLREGS, 보험 약관 충돌조항",
            "crew_injury": "선원 재해 보상, 선원법, 산재보험 관련 규정, 보험 약관 인적사고 조항",
        }

        hint = hints.get(itype, "해상보험 일반 약관, 선박사고 일반 규정")
        return f"{desc}\n\n[검색 힌트]\n{hint}".strip()

    # ---------------------------------------------------------------------
    def _rag_context(
        self,
        seed_query: str,
        top_k: int,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        collection: Optional[str] = None,
    ) -> str:
        rag = RAGChain(
            collection=collection or self.default_collection,
            top_k=top_k,
            model=model,
            temperature=temperature,
            include_sources=True,
        )
        return rag.run(seed_query)

    # ---------------------------------------------------------------------
    # 1. 보험 보고서 PDF 생성
    # ---------------------------------------------------------------------
    def generate_insurance_report_pdf(
        self,
        task_id: str,
        incident_data: Dict[str, Any],
        incident_type: Optional[str] = None,
        use_rag: bool = True,
        top_k: int = 5,
        title: str = "해양 보험 청구 보고서",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        rag_ctx = ""
        if use_rag:
            seed = self._build_seed_query(incident_data, incident_type or "")
            rag_ctx = self._rag_context(seed_query=seed, top_k=top_k, model=model)

        chain = ReportChain(model=model, temperature=temperature)
        resolved_type = (incident_type or incident_data.get("incident_type", "generic")).lower()

        report_text = chain.generate_report(
            incident_data=incident_data,
            rag_context=rag_ctx,
            incident_type=resolved_type,
        )

        if hasattr(report_text, "content"):
            report_text = report_text.content
        elif not isinstance(report_text, str):
            report_text = str(report_text)

        path = os.path.join(self.reports_dir, f"{task_id}.pdf")
        save_report_pdf(
            path=path,
            title=title,
            answer=report_text,
            incident_data=incident_data,
            subtitle=f"Task ID: {task_id}",
        )
        print(f"보고서 <PDF> 저장 완료: {path}")
        return path

    # ---------------------------------------------------------------------
    def generate_structured_report_background(
        self,
        task_id: str,
        incident_data: Dict[str, Any],
        use_rag: bool = True,
        top_k: int = 5,
        model: str = "gpt-4o-mini",
        title: str = "해양 보험 청구 보고서",
        temperature: float = 0.1,
        collection: Optional[str] = None,
    ):
        """
        구조화된 보고서를 JSON + PDF 모두 저장 (2~3페이지 완성본)
        """
        report: InsurancePydanticReportResponse = self.generate_structured_report(
            incident_data=incident_data,
            use_rag=use_rag,
            top_k=top_k,
            model=model,
            title=title,
            temperature=temperature,
            collection=collection,
        )

        # JSON 저장 (SecretStr 등 안전 변환)
        json_path = os.path.join(self.reports_dir, f"{task_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            safe_report = to_safe_json(report)
            json.dump(safe_report, f, indent=2, ensure_ascii=False)
        print(f"구조화 보고서 <JSON> 저장 완료: {json_path}")

        # PDF 저장
        pdf_path = os.path.join(self.reports_dir, f"{task_id}.pdf")
        formatted_answer = (
            f"[제목]\n{report.title}\n\n"
            f"[사고 개요]\n{report.incident_summary}\n\n"
            f"[사고 경위]\n{report.sequence_of_events}\n\n"
            f"[피해 내역]\n{report.damages}\n\n"
            f"[법령 근거]\n" + "\n".join(report.legal_basis) + "\n\n"
            f"[산정 기준]\n{report.calculation_basis}\n\n"
            f"[첨부서류]\n" + ", ".join(report.attachments) + "\n\n"
            f"[결론]\n{report.conclusion}\n\n"
            f"[조사관 의견]\n"
            f"{getattr(report, 'recommendations', '보험사 검토 및 환경 당국의 후속 조치 필요.')}\n\n"
        )

        save_report_pdf(
            path=pdf_path,
            title=title,
            answer=formatted_answer,
            incident_data=incident_data,
            subtitle=f"Task ID: {task_id}",
        )
        print(f"구조화 보고서 <PDF> 저장 완료: {pdf_path}")
        return {"json": json_path, "pdf": pdf_path}

    # ---------------------------------------------------------------------
    def generate_structured_report(
        self,
        incident_data: Dict[str, Any],
        use_rag: bool = True,
        top_k: int = 5,
        model: str = "gpt-4o-mini",
        title: str = "해양 보험 청구 보고서",
        temperature: float = 0.1,
        collection: Optional[str] = None,
    ) -> InsurancePydanticReportResponse:
        rag_ctx = ""
        if use_rag:
            seed = self._build_seed_query(incident_data, incident_data.get("incident_type", "generic"))
            rag_ctx = self._rag_context(
                seed_query=seed, top_k=top_k, model=model, temperature=0.0, collection=collection
            )

        chain = StructuredReportChain(model=model, temperature=temperature)
        return chain.generate_structured_report(
            incident_data=incident_data,
            rag_context=rag_ctx,
        )
