
"""
report_models.py
- FastAPI용 Pydantic 데이터 모델 정의
- 자유 질문 QA / 사고데이터 기반 보험 보고서 / 태스크 응답

Spring의 @RequestBody DTO랑 역할 똑같음.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


# ------------------------------------------------------------
# 사건 데이터 기반 해양 보험 보고서 생성 요청 -> 핵심 
# ------------------------------------------------------------
from pydantic import BaseModel, Field
from typing import Optional


# ------------------------------------------------------------
# 구조화된 보고서 출력 모델 (Pydantic 기반)
# ------------------------------------------------------------
class InsurancePydanticReportResponse(BaseModel):
    """
    LLM이 생성한 보험 보고서의 구조화된 출력 형식.
    """
    title: str
    incident_summary: str
    sequence_of_events: str
    damages: str
    legal_basis: List[str]
    calculation_basis: Optional[str] = None
    attachments: Optional[List[str]] = []
    conclusion: Optional[str] = None


class IncidentReportRequest(BaseModel):
    """
    해양 사고 데이터를 기반으로 보고서를 생성하는 요청 모델.
    Spring의 FastApiReportRequest와 동일한 필드 구조로 수정됨.
    내부적으로 RAGChain + ReportChain을 사용하여
    보험약관 및 법령 기반 보고서를 생성한다.
    """
    incident_type: str = Field(..., description="사고 유형 (예: fire, oil_spill, collision, crew_injury 등)")
    description: str = Field(..., description="사고 설명 또는 세부 내용")
    location: str = Field(..., description="사고 발생 위치")
    report_type: Optional[str] = Field(default="INITIAL", description="보고서 유형 (예: INITIAL, UPDATE 등)")
    language: Optional[str] = Field(default="ko", description="생성할 보고서 언어 코드 (예: ko, en)")

    # LangChain/RAG 설정
    use_rag: bool = Field(default=True, description="RAG 컨텍스트 사용 여부")
    collection: Optional[str] = Field(default="default", description="RAG 벡터스토어 컬렉션명")
    top_k: Optional[int] = Field(default=5, description="RAG 검색 문서 수")
    model: Optional[str] = Field(default="gpt-4o-mini", description="사용할 OpenAI 모델명")
    title: Optional[str] = Field(default="해양 보험 청구 보고서", description="PDF 제목")



# ------------------------------------------------------------
# 비동기 태스크 응답
# ------------------------------------------------------------
class ReportTaskResponse(BaseModel):
    """
    백그라운드에서 실행된 보고서 생성 태스크의 식별자 응답.
    """
    task_id: str = Field(..., description="비동기 보고서 생성 태스크 ID (UUID)")
    message: str = Field(default="보고서 생성이 시작되었습니다.", description="현재 상태 메시지")


# ------------------------------------------------------------
# (선택) 사고유형 Enum 정의 (확장용)
# ------------------------------------------------------------
class IncidentType(str):
    """
    사고 유형 (Enum처럼 문자열로 관리)
    """
    FIRE = "fire"
    OIL_SPILL = "oil_spill"
    COLLISION = "collision"
    CREW_INJURY = "crew_injury"
    GENERIC = "generic"
