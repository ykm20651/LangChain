
"""
ReportChain
- RAG 컨텍스트 + 사고 데이터(JSON) → 보험 청구 보고서 자동 생성
- 사고 유형별 템플릿 체인 구성 (화재, 유류유출, 충돌, 선원 부상)
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ReportChain:
    """
    사고유형별 보험 청구 보고서 생성용 LangChain 체인
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        debug: bool = False,
    ):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.parser = StrOutputParser()
        self.debug = debug

        # 사고유형별 프롬프트 템플릿 미리 준비
        self.templates = {
            "fire": self._fire_template(),
            "oil_spill": self._oil_template(),
            "collision": self._collision_template(),
            "crew_injury": self._injury_template(),
        }

    # --------------------------------------------------
    # 1️⃣ 공용 인터페이스
    # --------------------------------------------------
    def generate_report(
        self,
        incident_data: Dict[str, Any],
        rag_context: str = "",
        incident_type: str = "generic",
    ) -> str:
        """
        사고유형에 맞는 프롬프트 선택 후 LLM 실행
        """
        formatted_incident = self._format_incident_data(incident_data)
        prompt_template = self._select_template(incident_type)

        # 프롬프트 생성
        prompt = prompt_template.format(
            rag_context=rag_context,
            incident_data=formatted_incident,
        )

        if self.debug:
            print("🧾 [프롬프트 시작] ----------------------")
            print(prompt[:1200])
            print("... (이하 생략)")
            print("-----------------------------------------")

        # LLM 실행
        response = self.llm.invoke(prompt)
        return self.parser.parse(response)

    # --------------------------------------------------
    # 2️⃣ 템플릿 선택 로직
    # --------------------------------------------------
    def _select_template(self, incident_type: str) -> PromptTemplate:
        key = incident_type.lower()
        if key in self.templates:
            return self.templates[key]
        else:
            return self._generic_template()

    # --------------------------------------------------
    # 3️⃣ 사고유형별 템플릿 정의
    # --------------------------------------------------
    def _fire_template(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            """
당신은 선박 보험 전문가입니다. 아래 [컨텍스트]를 근거로 선박 화재사고 보험 청구 보고서를 작성하세요.
모르면 "근거 부족"이라고 명시하세요.

[컨텍스트]
{rag_context}

[사고 정보]
{incident_data}

# 선박 화재사고 보험 청구 보고서

## 1. 사고 개요
- 화재 발생 일시, 위치, 선박명, 선주, 화재 원인(전기, 연료, 인화물질 등)
## 2. 사고 경위
- 화재 발생부터 진압까지의 경과 요약
## 3. 피해 현황
- 인적 피해, 선체 손상, 화물 피해, 환경 피해
## 4. 법령 및 약관 근거
- 관련 법률, IMO 기준, 보험약관 화재조항 근거
## 5. 보험금 산정 근거
- 손해액, 공제액, 면책사유 여부
## 6. 결론 및 권고사항
(마지막에 참고 문서 출처 요약)
"""
        )

    def _oil_template(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            """
당신은 해양오염 사고 처리 전문가입니다.
아래 [컨텍스트]를 기반으로 유류유출 사고의 보험 청구 보고서를 작성하세요.

[컨텍스트]
{rag_context}

[사고 정보]
{incident_data}

# 유류유출 사고 보험 청구 보고서

## 1. 사고 개요
- 발생 일시, 선박 정보, 해상 위치, 기상조건, 유류 종류/유출량
## 2. 사고 경위
- 유출 원인, 방제 조치, 환경 영향
## 3. 피해 현황
- 해양오염 범위, 해양생물 피해, 방제비용
## 4. 법령 및 근거
- 해양환경관리법, MARPOL 협약 관련 조항
## 5. 보험금 산정 근거
- 복구비, 방제비, 피해보상항목
## 6. 결론 및 권고사항
(참고 문서 출처 포함)
"""
        )

    def _collision_template(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            """
당신은 해사안전 전문가입니다. 아래 [컨텍스트]를 참조하여 선박 충돌사고 보험 보고서를 작성하세요.

[컨텍스트]
{rag_context}

[사고 정보]
{incident_data}

# 선박 충돌사고 보험 청구 보고서

## 1. 사고 개요
- 사고 일시, 위치, 충돌 선박, 항로, 기상
## 2. 사고 경위
- 충돌 전후 상황, 항해기록 요약
## 3. 피해 현황
- 선체 손상, 화물 파손, 인명피해
## 4. 법령 및 약관 근거
- 해사안전법, 국제충돌예방규칙(COLREGS) 관련 조항
## 5. 보험금 산정 근거
- 수리비, 정박손실비용 등
## 6. 결론 및 향후 예방조치
"""
        )

    def _injury_template(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            """
당신은 선원 재해 보상 전문 심사관입니다.
아래 [컨텍스트]를 바탕으로 선원 부상사고 보험 청구 보고서를 작성하세요.

[컨텍스트]
{rag_context}

[사고 정보]
{incident_data}

# 선원 부상사고 보험 청구 보고서

## 1. 사고 개요
- 부상자, 발생장소, 작업상황, 선박정보
## 2. 사고 경위
- 부상 원인, 구조조치, 의료조치
## 3. 피해 현황
- 부상 정도, 치료기간, 복귀 가능 여부
## 4. 법령 및 약관 근거
- 선원법, 산업재해보상보험법 관련 조항
## 5. 보험금 산정 근거
- 치료비, 휴업급여, 장해급여 산정
## 6. 결론 및 권고사항
"""
        )

    def _generic_template(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            """
아래 [컨텍스트]와 [사고 정보]를 바탕으로 일반 보험 청구 보고서를 작성하세요.

[컨텍스트]
{rag_context}

[사고 정보]
{incident_data}

# 일반 보험 청구 보고서

## 1. 사고 개요
## 2. 사고 경위
## 3. 피해 현황
## 4. 법령 및 약관 근거
## 5. 보험금 산정 근거
## 6. 결론
"""
        )

    # --------------------------------------------------
    # 4️⃣ 데이터 포맷팅
    # --------------------------------------------------
    def _format_incident_data(self, data: Dict[str, Any]) -> str:
        """
        JSON 사고데이터를 사람이 읽기 쉬운 라인 포맷으로 변환
        """
        lines = []
        for key, val in data.items():
            if val is not None and val != "":
                lines.append(f"- {key}: {val}")
        return "\n".join(lines) if lines else "(제공된 데이터 없음)"
