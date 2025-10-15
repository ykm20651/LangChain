
"""
RAGChain - Retrieval-Augmented Generation (검색 기반 생성 체인)

랭체인 표준 구조로 Retriever → PromptTemplate → ChatOpenAI → StrOutputParser

1) Retriever를 이용해 관련 문서를 검색
2) 검색된 문서들을 하나의 context 문자열로 병합
3) PromptTemplate으로 프롬프트 구성
4) ChatOpenAI로 LLM 호출
5) StrOutputParser로 결과 텍스트 반환

사용 예시:
    rag = RAGChain(collection="marine_laws", top_k=5)
    answer = rag.run("유류유출 사고의 보험 처리 절차를 요약해줘")

"""

from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from app.services.vectorstore import VectorStoreService


class RAGChain:
    def __init__(
        self,
        collection: str = "default",
        top_k: int = 4,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_context_len: int = 3000,
        include_sources: bool = True,
        debug: bool = False,
    ):
        """
        RAG 체인 초기화
        Args:
            collection: 벡터스토어 컬렉션명
            top_k: 검색 문서 개수
            model: 사용할 OpenAI LLM 모델명
            temperature: 창의성 (낮을수록 객관적)
            max_context_len: 프롬프트에 넣을 최대 문맥 길이
            include_sources: 결과 하단에 문서 출처 요약 포함 여부
            debug: True면 검색된 문서 및 context 콘솔 출력
        """
        self.collection = collection
        self.top_k = top_k
        self.include_sources = include_sources
        self.debug = debug
        self.max_context_len = max_context_len

        # (1) 벡터스토어 서비스 로드
        self.vs = VectorStoreService()
        self.retriever = self.vs.get_retriever(collection=collection, k=top_k)

        # (2) LLM (OpenAI)
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # (3) 출력 파서
        self.parser = StrOutputParser()

        # (4) 프롬프트 템플릿
        self.prompt = ChatPromptTemplate.from_template(
            """
너는 해양 보험 청구 보고서 작성 보조 AI다.
아래 [검색 문맥]에 포함된 정보를 사실 기반으로 사용하여 질문에 답하라.
만약 관련 정보가 없다면 "근거 부족"이라고 답하라.
가능하면 문서 출처를 요약해서 하단에 정리하라.

[검색 문맥]
{context}

[질문]
{question}
"""
        )

        # (5) Runnable 시퀀스 구성
        #    question → retriever → prompt → llm → parser
        self.chain = (
            {"context": self._context_from_retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | self.parser
        )

    # -----------------------------
    # 내부 메서드
    # -----------------------------

    def _context_from_retriever(self, question: str) -> str:
        """
        Retriever로 문서 검색 → context 문자열 생성
        """
        docs: List[Document] = self.retriever.get_relevant_documents(question)

        if self.debug:
            print("\n🔎 [RAG 검색 결과] ----------------------")
            for d in docs:
                print(f"• {d.metadata.get('source', 'unknown')}: {d.page_content[:120]}...")
            print("----------------------------------------\n")

        # 문서 내용 합치기
        merged = "\n\n".join([d.page_content for d in docs])
        if len(merged) > self.max_context_len:
            merged = merged[: self.max_context_len] + "\n\n...(문맥 길이 초과로 일부 생략됨)"

        # 문서 출처 추가
        if self.include_sources:
            sources = "\n".join(
                [f"- {d.metadata.get('source', 'unknown')}" for d in docs]
            )
            merged += f"\n\n[참고 문서 출처]\n{sources}"

        return merged

    # -----------------------------
    # 퍼블릭 메서드
    # -----------------------------

    def run(self, question: str) -> str:
        """
        단일 질문에 대해 RAG 파이프라인 실행.
        검색 + 생성 결과 문자열 반환.
        """
        return self.chain.invoke(question)

    def run_with_sources(self, question: str) -> dict:
        """
        검색된 문서 출처까지 함께 반환하는 버전.
        """
        docs: List[Document] = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])
        answer = self.llm.invoke(self.prompt.format(context=context, question=question)).content

        return {
            "answer": answer,
            "sources": [d.metadata.get("source", "unknown") for d in docs],
        }
