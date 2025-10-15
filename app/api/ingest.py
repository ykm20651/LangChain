
"""
ingest.py
- 문서(RAG용 지식) 수집 API
- URL / 텍스트 / PDF 파일 업로드를 지원
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from app.services.vectorstore import VectorStoreService


# FastAPI 라우터
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# VectorStore 서비스 (Chroma + Embeddings)
vs = VectorStoreService()


# -------------------------------------------------------------
# 요청 모델 정의 - DTO
# -------------------------------------------------------------
class IngestRequest(BaseModel):
    """
    URL 또는 텍스트를 통한 문서 수집 요청 모델
    """
    urls: Optional[List[str]] = Field(default=None, description="수집할 웹 문서 URL 리스트")
    texts: Optional[List[str]] = Field(default=None, description="수집할 원시 텍스트 리스트")
    collection: str = Field(default="default", description="Chroma 컬렉션 이름 (벡터 DB 명)")


# -------------------------------------------------------------
# 1️⃣ URL 수집 API
# -------------------------------------------------------------
@router.post("/urls")
async def ingest_urls(req: IngestRequest):
    """
    지정된 URL에서 문서를 로드하여 분할/임베딩 후 벡터스토어에 추가.
    """
    try:
        if not req.urls:
            raise HTTPException(status_code=400, detail="urls 필드가 필요합니다.")
        count = vs.add_from_urls(req.collection, req.urls)
        return {"ok": True, "added": count, "collection": req.collection}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# 2️⃣ 텍스트 수집 API
# -------------------------------------------------------------
@router.post("/texts")
async def ingest_texts(req: IngestRequest):
    """
    원시 텍스트 리스트를 벡터스토어에 추가.
    """
    try:
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts 필드가 필요합니다.")
        count = vs.add_from_texts(req.collection, req.texts)
        return {"ok": True, "added": count, "collection": req.collection}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# 3️⃣ PDF 파일 업로드 API
# -------------------------------------------------------------
@router.post("/pdf")
async def ingest_pdf(
    collection: str = "default",
    file: UploadFile = File(..., description="업로드할 PDF 파일"),
):
    """
    PDF 파일을 업로드 받아, 분할/임베딩 후 벡터스토어에 추가.
    """
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="PDF 파일만 허용됩니다.")
        content = await file.read()
        count = vs.add_from_pdf_bytes(collection, content, file.filename)
        return {
            "ok": True,
            "added": count,
            "collection": collection,
            "filename": file.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------
# 4️⃣ (선택) 로컬 폴더 수집 API
# -------------------------------------------------------------
@router.post("/directory")
async def ingest_directory(
    dir_path: str,
    collection: str = "default",
):
    """
    로컬 폴더 내 모든 문서를 일괄 수집.
    주로 관리자/개발자용 (운영서버에선 안전한 경로만 허용)
    """
    try:
        count = vs.add_from_directory(collection, dir_path)
        return {"ok": True, "added": count, "collection": collection, "dir": dir_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/count-all")
async def count_all(collection: str = Query(...)):
    try:
        chunk_count = vs.count(collection)
        doc_count = vs.count_documents(collection)
        return {
            "collection": collection,
            "document_count": doc_count,
            "chunk_count": chunk_count
        }
    except Exception as e:
        return {"error": str(e)}

