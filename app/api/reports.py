
"""
reports.py
- FastAPI 엔드포인트 정의
- 자유 질문 QA (선택)
- 보험 청구 보고서 생성 (비동기)
- 보고서 PDF 다운로드
"""

import os
import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from app.models.report_models import (
    ReportGenerationRequest,
    ReportResponse,
    IncidentReportRequest,
    ReportTaskResponse,
)
from app.services.report_service import ReportService, InsuranceReportService

# FastAPI 라우터 객체
router = APIRouter()

# 서비스 인스턴스
qa_service = ReportService()
insurance_service = InsuranceReportService()


# ---------------------------------------------------------------------
# 1️⃣ 자유 질문 QA (RAG 기반)
# ---------------------------------------------------------------------
@router.post("/chat", response_model=ReportResponse)
async def chat(req: ReportGenerationRequest):
    """
    자유 질문에 대해 RAG 기반 QA를 수행.
    """
    try:
        answer, _sid = qa_service.qa_with_memory(
            question=req.question,
            session_id=None,
            collection=req.collection,
            top_k=req.top_k,
            model=req.model,
            temperature=req.temperature,
        )
        return ReportResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# 2️⃣ 자유 질문 → 보고서 PDF 생성 (비동기)
# ---------------------------------------------------------------------
@router.post("/generate-report", response_model=ReportTaskResponse)
async def generate_report(req: ReportGenerationRequest, background_tasks: BackgroundTasks):
    """
    자유 질문 기반 PDF 보고서 생성 (비동기)
    """
    try:
        task_id = str(uuid.uuid4())
        background_tasks.add_task(
            qa_service.generate_and_save_report,
            task_id=task_id,
            question=req.question,
            session_id=None,
            collection=req.collection,
            top_k=req.top_k,
            title="자유 질의 기반 보고서",
            model=req.model,
            temperature=req.temperature,
        )
        return ReportTaskResponse(task_id=task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# 3️⃣ 사고 데이터 기반 해양 보험 청구 보고서 생성 (RAG + ReportChain)
# ---------------------------------------------------------------------
@router.post("/generate/insurance", response_model=ReportTaskResponse)
async def generate_insurance_report(req: IncidentReportRequest, background_tasks: BackgroundTasks):
    """
    해양 사고 데이터를 기반으로 보험 청구 보고서를 자동 생성한다.
    사고 유형별 템플릿 및 법령 기반 RAG 검색을 포함한다.
    """
    try:
        task_id = str(uuid.uuid4())
        background_tasks.add_task(
            insurance_service.generate_insurance_report_pdf,
            task_id=task_id,
            incident_data=req.incident_data,
            incident_type=req.incident_data.get("incident_type", None),
            use_rag=req.use_rag,
            collection=req.collection,
            top_k=req.top_k,
            title=req.title,
            model=req.model,
            temperature=0.1,
        )
        return ReportTaskResponse(task_id=task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# 4️⃣ 보고서 PDF 다운로드
# ---------------------------------------------------------------------
@router.get("/download/{task_id}.pdf")
async def download_report(task_id: str):
    """
    생성 완료된 PDF 보고서를 다운로드.
    """
    # 보험 보고서 → QA 보고서 순으로 경로 확인
    path = insurance_service.get_report_path(task_id) or qa_service.get_report_path(task_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="보고서를 찾을 수 없습니다.")
    return FileResponse(path, media_type="application/pdf", filename=os.path.basename(path))
