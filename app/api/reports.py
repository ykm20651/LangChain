
"""
reports.py
- FastAPI 엔드포인트 정의
- 1. 해양 사고 데이터를 기반으로 보험 청구 보고서를 자동 생성 (비동기) -> 안다미로 핵심 ai 보고서 생성 라우터
- 2. 보고서 PDF 다운로드 
"""

import os
import uuid
import time

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from app.models.report_models import (
    InsurancePydanticReportResponse,
    IncidentReportRequest,
    ReportTaskResponse,
)
from app.services.report_service import InsuranceReportService

# FastAPI 라우터 객체
router = APIRouter()

# 서비스 인스턴스
insurance_service = InsuranceReportService()

# ---------------------------------------------------------------------
# 1. 사고 데이터 기반 해양 보험 청구 보고서 생성 (RAG + ReportChain)
# ---------------------------------------------------------------------
@router.post("/generate/insurance", response_model=ReportTaskResponse)
async def generate_insurance_report(req: IncidentReportRequest, background_tasks: BackgroundTasks):
    """
    해양 사고 데이터를 기반으로 보험 청구 보고서를 자동 생성한다.
    사고 유형별 템플릿 및 법령 기반 RAG 검색을 포함한다.
    """
    try:
        task_id = str(uuid.uuid4())
        incident_data = {
            "incident_type": req.incident_type,
            "description": req.description,
            "location": req.location,
            "report_type": req.report_type,
            "language": req.language,
        }

        background_tasks.add_task(
        insurance_service.generate_insurance_report_pdf,
        task_id=task_id,
        incident_data=incident_data,
        incident_type=req.incident_type,
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
# 2. 보고서 PDF 다운로드
# ---------------------------------------------------------------------
@router.get("/download/{task_id}.pdf")
async def download_report(task_id: str):
    for _ in range(10):  # 최대 10번 재시도
        path = insurance_service.get_report_path(task_id)
        if path and os.path.exists(path):
            return FileResponse(path, media_type="application/pdf", filename=os.path.basename(path))
        time.sleep(0.5)  # 0.5초 간격으로 기다림
    raise HTTPException(status_code=404, detail="보고서를 찾을 수 없습니다.")


# ---------------------------------------------------------------------
# 3. Pydantic 활용 
# ---------------------------------------------------------------------

@router.post("/generate/structured", response_model=ReportTaskResponse)
async def generate_structured_insurance_report_async(
    req: IncidentReportRequest,
    background_tasks: BackgroundTasks,
):
    """
    사고 데이터를 기반으로 구조화된 보험 보고서를 비동기로 생성.
    보고서는 JSON 파일로 저장되며, task_id로 조회 가능.
    """
    try:
        task_id = str(uuid.uuid4())

        incident_data = {
            "incident_type": req.incident_type,
            "description": req.description,
            "location": req.location,
            "report_type": req.report_type,
            "language": req.language,
        }

        background_tasks.add_task(
            insurance_service.generate_structured_report_background,
            task_id=task_id,
            incident_data=incident_data,
            use_rag=req.use_rag,
            collection=req.collection,
            top_k=req.top_k,
            model=req.model,
            title=req.title,
            temperature=0.1,
        )

        return ReportTaskResponse(task_id=task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


