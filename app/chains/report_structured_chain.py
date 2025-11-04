"""
StructuredReportChain
- RAG 컨텍스트 + 사고 데이터(JSON) → 구조화된 보험 청구 보고서 자동 생성
- 실제 보험사 제출 양식에 기반한 상세 보고서 (2~3페이지 수준)
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.models.report_models import InsurancePydanticReportResponse


class StructuredReportChain:
    """
    해양보험 사고 보고서를 구조화된 JSON으로 생성하는 LangChain 체인.
    - 공문서형 어투, 사고조사관 수준의 디테일
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        self.parser = PydanticOutputParser(pydantic_object=InsurancePydanticReportResponse)
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # 사고유형별 세부 가이드
        self.guides = {
            "fire": "- 화재 원인(전기, 연료, 인화물질 등), 진압 경과, 화물·선체 손상, 선박안전법 관련 조항",
            "oil_spill": "- 유류 종류/유출량, 기상해황, 방제 조치, 해양환경관리법·MARPOL 조항",
            "collision": "- 충돌 선박 정보, 항로/시야/속력/통신 내역, COLREGS 및 해사안전법 조항",
            "crew_injury": "- 부상자 신원, 근무 중 여부, 보호장구 착용, 선원법/산재보험 조항",
            "generic": "- 손해내역과 법적 근거를 구체적으로 연결, 공란 없이 작성"
        }

        # 🔥 개선된 고품질 보고서 프롬프트
        self.prompt_template = PromptTemplate(
            template="""
당신은 해양사고 및 보험 손해사정 분야의 전문 조사관이며,
보험사 제출용 **공식 해양 보험 청구 보고서**를 작성해야 합니다.

이 보고서는 법적·기술적 사실관계의 정리, 손해 산정, 보험금 청구 근거를 모두 포함해야 합니다.
단순 요약이 아닌, 실제 조사관의 시각에서 신뢰성 있는 전문 문체로 작성하십시오.

[입력 데이터]
사고 기본정보:
{incident_data}

관련 법령 및 약관 정보:
{rag_context}

사고유형별 참고 가이드:
{guide}

작성 지침:
1. 사고 개요에는 **배경, 기상상태, 운항경로, 선박 제원, 당시 조건**을 포함하세요.
2. 사고 경위는 **시각별 전개 과정**을 서술하세요.
3. 피해 내역은 **인명 / 환경 / 선체 / 화물 / 제3자 피해**로 나누어 기술하세요.
4. 법령 근거는 **조문 번호 + 문장 인용 + 해석**으로 작성하세요.
5. 산정 기준은 **금액, 근거 문서, 복구 절차, 보험금 처리 절차**를 구체적으로 기술하세요.
6. 결론에는 **사고 원인 판단, 향후 조치 권고, 보험금 지급 판단 요약**을 포함하세요.
7. 각 항목은 최소 5문장 이상 작성하며, 전체 분량은 약 2~3페이지를 목표로 합니다.
8. 공식 보고서 어투 (“확인됨”, “판단됨”, “이행되었음”)을 사용하세요.

JSON 스키마 명세:
{format_instructions}
""",
            input_variables=["incident_data", "rag_context", "guide"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def generate_structured_report(
        self,
        incident_data: Dict[str, Any],
        rag_context: str = "",
    ) -> InsurancePydanticReportResponse:
        formatted_incident = "\n".join(f"{k}: {v}" for k, v in incident_data.items())
        incident_type = (incident_data.get("incident_type") or "generic").lower()
        guide = self.guides.get(incident_type, self.guides["generic"])

        prompt = self.prompt_template.format(
            incident_data=formatted_incident,
            rag_context=rag_context,
            guide=guide
        )

        response = self.llm.invoke(prompt)

        # 결과를 Pydantic 모델로 파싱
        parsed = self.parser.parse(response.content if hasattr(response, "content") else str(response))

        return parsed  # InsurancePydanticReportResponse 객체 반환

