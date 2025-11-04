"""
RAGChain - Retrieval-Augmented Generation (검색 기반 생성 체인)
보험사 내부 문체(공문체)로 강화된 버전
"""

from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
        """
        self.collection = collection
        self.top_k = top_k
        self.include_sources = include_sources
        self.debug = debug
        self.max_context_len = max_context_len

        # (1) 벡터스토어 서비스
        self.vs = VectorStoreService()
        self.retriever = self.vs.get_retriever(collection=collection, k=top_k)

        # (2) LLM
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # (3) 출력 파서
        self.parser = StrOutputParser()

        # (4) 프롬프트 템플릿 (보험심사 문체)
        self.prompt = ChatPromptTemplate.from_template(
            """
당신은 해상보험 심사관이며, 아래 [검색 문맥]은 관련 법령·약관·협약에서 추출한 내용이다.
이 정보를 바탕으로 사고 보고서 및 보험금 청구서 작성에 필요한 "법적 근거 문단"을 작성하라.

작성 규칙:
1. 문체는 공식 보고서체(예: "~에 의거하여", "~으로 판단된다")로 통일한다.
2. 사실 확인이 불가능한 내용은 "확인 불가"로 명시한다.
3. 법령, 협약, 약관 명칭은 정확히 표기하고, 조항이 있다면 번호를 함께 적는다.
4. 객관적 사실·조문 중심으로 기술하고, 주관적 판단·권유 표현은 피한다.
5. 원문 문맥 중 중요 문구는 ‘“...”’로 인용 표시한다.
6. 결론 문장은 항상 “따라서 본 사고는 위 조항에 의거하여 ...로 분류된다.” 형식으로 마무리한다.

[검색 문맥]
{context}

[질문/요청 사항]
{question}
"""
        )

        # (5) 체인 구성
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
                src = d.metadata.get('source', 'unknown')
                print(f"• {src}: {d.page_content[:120]}...")
            print("----------------------------------------\n")

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
        answer = self.llm.invoke(
            self.prompt.format(context=context, question=question)
        ).content

        return {
            "answer": answer,
            "sources": [d.metadata.get("source", "unknown") for d in docs],
        }
