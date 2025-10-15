"""
main.py
- FastAPI 애플리케이션 실행 루트
- 라우터 등록 (ingest, reports)
- CORS, 환경변수 설정
"""

import os
from fastapi import FastAPI # 이거로 앱 인스턴스 생성 
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# API 라우터 import
from app.api import ingest, reports

# 환경변수 로드 (.env 파일)
load_dotenv()

# FastAPI 앱 생성
app = FastAPI(
    title="Marine Insurance Report Generator",
    description="LangChain 기반 해양 보험 청구 보고서 자동 생성 시스템",
    version="1.0.0"
)

# -------------------------------------------------------------
# CORS 설정 (프론트엔드, Swagger, 외부 접근 허용)
# -------------------------------------------------------------
origins = os.getenv("CORS_ORIGINS", "*").split(",")  # 환경변수에서 CORS_ORIGINS 지정 가능

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# 라우터 등록
# -------------------------------------------------------------
app.include_router(ingest.router)
app.include_router(reports.router)


# -------------------------------------------------------------
# 기본 엔드포인트
# -------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "🌊 Marine Insurance AI API is running",
        "routes": ["/ingest", "/chat", "/generate/insurance", "/download/{task_id}.pdf"],
        "version": "1.0.0"
    }
