"""
ReportService / InsuranceReportService
- (옵션) QA + RAG
- 사고 데이터 기반: RAG 컨텍스트 생성 → 사고유형별 템플릿 보고서 생성 → PDF 저장

환경변수:
- REPORTS_DIR: PDF 저장 디렉터리 (기본: ./storage/reports)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

from app.chains.rag_chain import RAGChain
from app.chains.report_chain import ReportChain
from app.services.memory_manager import MemoryManager  # (선택) QA 히스토리용
from app.utils.pdf import save_report_pdf


# -----------------------------------------------------------
# 핵심: 사건 데이터 기반 보험 보고서 생성
# -----------------------------------------------------------
class InsuranceReportService:
    """
    사건 데이터(incident_data) → (선택) RAG 컨텍스트 → 사고유형별 템플릿 보고서 → PDF 저장
    - 사고유형: fire | oil_spill | collision | crew_injury
    """

    def __init__(self):
        self.reports_dir = os.getenv("REPORTS_DIR", "./storage/reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    def _build_seed_query(self, incident_data: Dict[str, Any], incident_type: str) -> str:
        """
        RAG 검색 시 사용할 seed query를 사고유형에 맞게 생성.
        - description/title이 있으면 우선 포함
        - 유형별 키워드로 법령/약관을 더 잘 당겨오도록 힌트 제공
        """
        desc = str(
            incident_data.get("description", incident_data.get("title", "")) or ""
        ).strip()

        itype = (incident_type or incident_data.get("incident_type", "generic")).lower()

        if itype == "fire":
            hint = "선박 화재사고 처리 기준, 보험 약관 화재조항, 선박안전법 관련 조항"
        elif itype == "oil_spill":
            hint = "유류유출 방제 기준, 해양환경관리법, MARPOL 협약, 보험 약관 오염조항"
        elif itype == "collision":
            hint = "선박 충돌 관련 해사안전법, 국제충돌예방규칙 COLREGS, 보험 약관 충돌조항"
        elif itype == "crew_injury":
            hint = "선원 재해 보상, 선원법, 산재보험 관련 규정, 보험 약관 인적사고 조항"
        else:
            hint = "해상보험 일반 약관, 선박사고 일반 규정"

        seed = f"{desc}\n\n[검색 힌트]\n{hint}"
        return seed.strip()

    def _rag_context(
        self,
        collection: str,
        top_k: int,
        seed_query: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> str:
        """
        RAGChain으로 법령/약관 컨텍스트 생성.
        - temperature는 낮게: 사실기반 요약/편집에 유리
        """
        rag = RAGChain(
            collection=collection,
            top_k=top_k,
            model=model,
            temperature=temperature,
            include_sources=True,
        )
        return rag.run(seed_query)

    # ---------- 퍼블릭 API ----------
    def generate_insurance_report_pdf(
        self,
        task_id: str,
        incident_data: Dict[str, Any],
        incident_type: Optional[str] = None,
        use_rag: bool = True,
        collection: str = "default",
        top_k: int = 5,
        title: str = "해양 보험 청구 보고서",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        """
        1) (옵션) RAG 컨텍스트 생성
        2) 사고유형별 템플릿으로 보고서 작성
        3) PDF 저장
        """

        # 1. RAG 컨텍스트
        rag_ctx = ""
        if use_rag:
            seed = self._build_seed_query(incident_data, incident_type or "")
            rag_ctx = self._rag_context(
                collection=collection,
                top_k=top_k,
                seed_query=seed,
                model=model,
                temperature=0.0,  # 사실요약은 낮게
            )

        # 2. 보고서 생성 (보험 청구 양식)
        chain = ReportChain(model=model, temperature=temperature)
        resolved_type = (incident_type or incident_data.get("incident_type", "generic")).lower()

        report_text = chain.generate_report(
            incident_data=incident_data,
            rag_context=rag_ctx,
            incident_type=resolved_type,
        )

        # LangChain 0.2.x 이상 대응
        if hasattr(report_text, "content"):
            report_text = report_text.content
        elif not isinstance(report_text, str):
            report_text = str(report_text)

        # 3. PDF 저장 (질문 섹션 제거, incident_data를 표로 삽입)
        path = os.path.join(self.reports_dir, f"{task_id}.pdf")

        try:
            save_report_pdf(
                path=path,
                title=title,
                answer=report_text,
                incident_data=incident_data,
                subtitle=f"Task ID: {task_id}",
            )
            print(f"✅ 보고서 PDF 저장 완료: {path}")
        except Exception as e:
            print(f"❌ PDF 저장 중 오류 발생: {e}")
            raise e

        return path

    def get_report_path(self, task_id: str) -> Optional[str]:
        path = os.path.join(self.reports_dir, f"{task_id}.pdf")
        return path if os.path.exists(path) else None


# -----------------------------------------------------------
# (선택) 기존 QA + RAG
# -----------------------------------------------------------
class ReportService:
    """
    자유 질문 → RAG QA → 답변 or PDF
    - 시스템 목표가 '보고서 자동 생성'이라면 이 클래스는 사용하지 않아도 됨.
    """

    def __init__(self):
        self.memory = MemoryManager()
        self.reports_dir = os.getenv("REPORTS_DIR", "./storage/reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    def qa_with_memory(
        self,
        question: str,
        session_id: Optional[str],
        collection: str,
        top_k: int,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> Tuple[str, str]:
        sid = self.memory.ensure(session_id)
        history = self.memory.history_text(sid)
        prompt_text = f"{history}\n\n현재 질문: {question}\n"

        rag = RAGChain(
            collection=collection,
            top_k=top_k,
            model=model,
            temperature=temperature,
            include_sources=True,
        )
        answer = rag.run(prompt_text)

        self.memory.add_user(sid, question)
        self.memory.add_ai(sid, answer)
        return answer, sid

    def generate_and_save_report(
        self,
        task_id: str,
        question: str,
        session_id: Optional[str],
        collection: str,
        top_k: int,
        title: str = "자동 생성 보고서",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ):
        answer, _sid = self.qa_with_memory(
            question=question,
            session_id=session_id,
            collection=collection,
            top_k=top_k,
            model=model,
            temperature=temperature,
        )
        path = os.path.join(self.reports_dir, f"{task_id}.pdf")
        # 질문 기반 PDF는 유지하되, 신규 보험청구서 흐름엔 사용 안 함
        from app.utils.pdf import save_report_pdf as _save
        _save(path, title=title, answer=answer, incident_data=None, subtitle="QA Report")
