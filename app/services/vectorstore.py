
"""
RAG 벡터스토어 서비스
- 문서 수집(로더)
- 텍스트 분할(TextSplitter)
- 임베딩(OpenAIEmbeddings)
- 벡터DB(Chroma) 저장/조회
- 검색기(Retriever) 생성

환경변수:
- OPENAI_API_KEY         : OpenAI API 키
- RAG_PERSIST_DIR        : Chroma 영속 저장 위치 (기본: ./storage/chroma)

RAG의 핵심은 “LLM이 모르는 사실을 외부에서 끌어다 쓴다”는 것임.
GPT는 해양사고보험법이나 MARPOL 협약 전문을 통째로 기억하지 않기 때문에, 벡터스토어에 넣은 텍스트 조각들을 불러와서 
요약 컨텍스트로 제공해야 → “출처 기반, 사실 일관된 보고서”를 만들 수 있음.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Iterable

from dotenv import load_dotenv

# LangChain - Vector DB / Embeddings / Loaders / Splitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain.schema import Document

# (선택) LLM 기반 필터/압축 검색기 추가 옵션
# - 비용/지연이 늘 수 있어 기본은 False로 사용 권장
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI


load_dotenv()


class VectorStoreService:
    """
    문서 → [분할] → [임베딩] → [Chroma] → [Retriever]
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        separators: Optional[List[str]] = None,
    ):
        # 1) 저장 경로
        self.persist_dir = persist_dir or os.getenv("RAG_PERSIST_DIR", "./storage/chroma")
        os.makedirs(self.persist_dir, exist_ok=True)

        # 2) 임베딩 모델 (OpenAI)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # 3) 분할기
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""],
        )

    
    #임베딩된 벡터 저장 + 유사도 검색 지원
    def _chroma(self, collection: str) -> Chroma:
        """
        컬렉션명에 해당하는 Chroma 핸들러 생성/연결
        """
        return Chroma(
            collection_name=collection,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
    
    """
    ex) 컬렉션은 DB에 저장될 내용 분류느낌 
    -- 벡터 Database (Chroma)
    Database: ./storage/chroma
        ├─ Collection: marine_laws      ← 해양 법령
        ├─ Collection: shipping_terms   ← 해운 약관
        └─ Collection: company_rules    ← 회사 규정
    
    """

    def _split(self, docs: List[Document]) -> List[Document]:
        """
        긴 문서를 청크 단위로 분할 (문맥 손실 최소화를 위해 overlap 사용)
        """
        return self.splitter.split_documents(docs)

    def _persist(self, collection: str):
        """
        현재 컬렉션 상태를 디스크에 영속화
        """
        self._chroma(collection).persist()

    # -------------------------
    # 수집(Ingest)
    # -------------------------

    #웹 문서 URL을 넣고 HTML 파싱 
    def add_from_urls(self, collection: str, urls: List[str]) -> int:
        """
        웹 URL에서 문서를 로드 → 분할 → 임베딩/저장
        """
        all_docs: List[Document] = []
        for url in urls:
            loader = WebBaseLoader(url)
            docs = loader.load()
            # 출처 기록
            for d in docs:
                d.metadata = {**d.metadata, "source": url}
            all_docs.extend(docs)
 
        chunks = self._split(all_docs) #문서 분할하고
        store = self._chroma(collection) #벡터 DB연결하고 
        store.add_documents(chunks) #각 청크의 텍스트를 OpenAI 임베딩 API로 전송 
        self._persist(collection)
        return len(chunks)

    def add_from_texts(self, collection: str, texts: List[str], source: str = "text") -> int:
        """
        원시 텍스트 리스트를 Document로 감싸서 → 분할 → 임베딩/저장
        """
        docs = [Document(page_content=t, metadata={"source": source}) for t in texts]
        chunks = self._split(docs) #청크 단위로 나누고,
        store = self._chroma(collection) #지정한 컬렉션, OpenAI 임베딩 모델과 저장경로에 맞추어 벡터 DB 연결.
        store.add_documents(chunks) # 각 청크를 OpenAI API로 전송, 벡터 + 원시 텍스트 + 메타데이터를 ChromaDB에 저장
        self._persist(collection) # 디스크 동기화 
        return len(chunks)

    #웹에서 업로드된 PDF 파일을 벡터 DB에 저장할 때
    """
    <파일 업로드는 2가지 방식임>
    1. 사용자가 웹에서 업로드시킨 바이트 데이터 -> add_from_pdf_bytes
    ex) 
        service.add_from_pdf_bytes(
            collection="laws",
            content=uploaded_file_bytes,  # 바이트 데이터
            filename="법령.pdf"
        )
    
    2. 서버에 이미 저장된 파일 - add_from_pdf_paths 
    ex)"./storage/법령.pdf"
    """
    def add_from_pdf_bytes(self, collection: str, content: bytes, filename: str) -> int:
        """
        바이트로 올라온 PDF 파일을 임시 경로에 저장해서 → PyPDFLoader 로드 → 분할/저장
        FastAPI 업로드와 연결되는 경로
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for d in docs:
                d.metadata = {**d.metadata, "source": filename}
            chunks = self._split(docs)
            store = self._chroma(collection)
            store.add_documents(chunks)
            self._persist(collection)
            return len(chunks)
        finally:
            os.remove(tmp_path)


    def add_from_pdf_paths(self, collection: str, pdf_paths: List[str]) -> int:
        """
        로컬 PDF 경로들에서 로드 → 분할/저장
        """
        all_docs: List[Document] = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs = loader.load()
            for d in docs:
                d.metadata = {**d.metadata, "source": os.path.basename(path)}
            all_docs.extend(docs)

        chunks = self._split(all_docs)
        store = self._chroma(collection)
        store.add_documents(chunks)
        self._persist(collection)
        return len(chunks)

    def add_from_directory(
        self,
        collection: str,
        dir_path: str,
        glob: str = "**/*",
        use_multithreading: bool = True,
    ) -> int:
        """
        폴더 전체 일괄 수집 (PDF/텍스트 등)
        - DirectoryLoader가 내부적으로 파일 확장자에 맞는 Loader를 선택
        """
        loader = DirectoryLoader(
            dir_path,
            glob=glob,
            show_progress=True,
            use_multithreading=use_multithreading,
        )
        docs = loader.load()
        # source 메타 채우기
        for d in docs:
            if "source" not in d.metadata:
                d.metadata["source"] = d.metadata.get("file_path", "directory")
        chunks = self._split(docs)
        store = self._chroma(collection)
        store.add_documents(chunks)
        self._persist(collection)
        return len(chunks)

    def add_from_text_files(self, collection: str, paths: List[str]) -> int:
        """
        로컬 .txt 파일들을 TextLoader로 수집
        """
        all_docs: List[Document] = []
        for path in paths:
            loader = TextLoader(path)
            docs = loader.load()
            for d in docs:
                d.metadata = {**d.metadata, "source": os.path.basename(path)}
            all_docs.extend(docs)

        chunks = self._split(all_docs)
        store = self._chroma(collection)
        store.add_documents(chunks)
        self._persist(collection)
        return len(chunks)

    # -------------------------
    # 검색(Retrieve)
    # -------------------------
    def get_retriever( #벡터 DB에서 유사한 문서를 검색하는 객체(Retriever) 생성
        self,
        collection: str,
        k: int = 4,
        search_type: str = "similarity", #검색 알고리즘 종류 
        score_threshold: Optional[float] = None, #유사도 임계값 0.0 ~ 1.0
        enable_llm_filter: bool = False, 
        llm_model: str = "gpt-4o-mini",
    ):
        """
        컬렉션에서 Retriever를 생성.
        - search_type: "similarity" | "mmr" | "similarity_score_threshold"
        - k: 가져올 문서 수
        - score_threshold: search_type이 similarity_score_threshold일 때 임계값
        """
        store = self._chroma(collection)

        search_kwargs = {"k": k}
        if search_type == "similarity_score_threshold" and score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        base_retriever = store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

        if not enable_llm_filter:
            return base_retriever

        # 2차 LLM 필터링 단계 (문맥 압축): 반환 문서 수를 줄여 노이즈 감소 (비용/지연↑) 일단 false로 해놓자. 
        compressor = LLMChainFilter.from_llm( #검색된 문서를 ChatGPT에게 보여주고 "이게 쿼리랑 관련 있어?" 물어봄
            llm=ChatOpenAI(model=llm_model, temperature=0.0)
        )
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )

    # -------------------------
    # 문서 개수 조회용  
    # -------------------------
    def count(self, collection: str) -> int:
        """
        컬렉션에 저장된 벡터 문서 수
        """
        return self._chroma(collection)._collection.count()  # 내부 핸들 접근 (커뮤니티 드라이버 관례)

    def list_collections(self) -> List[str]:
        """
        현재 persist_dir에 존재하는 컬렉션 목록 (간단 버전)
        """
        # Chroma API가 컬렉션 나열을 직접 제공하지 않으므로, 디렉토리 스캔 또는 메타 유지가 필요.
        # 여기선 간단히 디렉토리 스캔(컬렉션 당 별도 메타를 쓰지 않으므로 best-effort).
        names: List[str] = []
        # chroma는 내부적으로 sqlite/duckdb 등 저장하므로 파일명으로 추론이 어렵다.
        # 실서비스에선 별도의 메타파일을 관리할 것을 권장.
        # 일단은 None 리턴로 두거나, 필요한 경우 확장.
        return names
