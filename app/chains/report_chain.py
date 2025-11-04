"""
ReportChain
- RAG 컨텍스트 + 사고 데이터(JSON) → 보험 청구 보고서 자동 생성
- 사고 유형별 템플릿 체인 구성 (화재, 유류유출, 충돌, 선원 부상)
- 실제 보험 청구 양식(기본정보/사고개요/손해내역/법령근거/산정근거/첨부) 형태로 개편
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
    # 1) 공용 인터페이스
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
            print(prompt[:2000])
            print("... (이하 생략)")
            print("-----------------------------------------")

        # LLM 실행
        response = self.llm.invoke(prompt)
        return self.parser.parse(response)

    # --------------------------------------------------
    # 2) 템플릿 선택 로직
    # --------------------------------------------------
    def _select_template(self, incident_type: str) -> PromptTemplate:
        key = (incident_type or "generic").lower()
        if key in self.templates:
            return self.templates[key]
        else:
            return self._generic_template()

    # --------------------------------------------------
    # 3) 사고유형별 템플릿 정의 (보험청구 양식)
    # --------------------------------------------------
    def _common_header(self) -> str:
        # 모든 유형에서 공유하는 “보험청구서 양식” 골격     
        return """
당신은 해상보험 심사관입니다.
아래 [컨텍스트]와 [사고 기본정보]를 바탕으로
'보험 청구서 양식'에 맞춰 정확하고 간결한 공식 문서체로 보고서를 작성하세요.
가능한 경우 법령/약관 조항은 조항번호와 조문명을 병기하고, 근거가 부족하면 '근거 부족'이라고 명확히 표기합니다.
모든 금액은 숫자 단위(원/달러 등)를 명시하십시오.

[컨텍스트]
{rag_context}

[사고 기본정보]
{incident_data}

# 보험 청구서

## 1. 기본정보
- 보험사명: (확정 불가 시 공란)
- 청구인(선주/회사): (확정 불가 시 공란)
- 사고유형: (incident_type 반영)
- 사고일시: (컨텍스트/설명에서 추론 가능 시 기입, 불가 시 공란)
- 발생위치: (location 반영)
- 보고서 유형: (report_type 반영)
- 언어: (language 반영)

## 2. 사고 개요
- 한 문단 요약
- 주요 사실(언제/어디서/무엇/왜/어떻게)

## 3. 사고 경위
- 발생 전후 경과, 즉시 조치 내용, 확인 가능한 로그/기록(가능 시)

## 4. 피해 및 손해내역
- 인적/선체/화물/환경 피해 항목별 요약
- 방제/복구/수리 관련 항목(해당 시)

## 5. 법령·국제기준·약관 근거
- 관련 국내법/국제협약/보험약관 조항 요약
- 각 조항의 적용 이유 및 시사점

## 6. 보험금 산정 근거 및 청구금액
- 항목별 산정(예: 수리비, 방제비, 정박손실 등)
- 공제/면책 사유 검토
- 최종 청구금액 제시(가능 시 추정치 포함)

## 7. 첨부서류 목록(필요 시)
- 선박등록증/선원명부/사진/수리견적서/조사보고서/해경·기관 문서 등

## 8. 결론 및 권고사항
- 책임소지/재발방지/후속조치 권고
"""

    def _fire_template(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            self._common_header()
            + """
[유형별 보강 가이드]
- 화재 원인(전기/연료/인화물질 등)과 진압 경과를 정확히 정리
- 선박안전법/보험 약관의 화재 관련 조항 인용
- 화물 손상, 연소·열 손상 범위와 수리 가능성 명시
"""
        )

    def _oil_template(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            self._common_header()
            + """
[유형별 보강 가이드]
- 유류 종류/유출량, 기상해황, 오염 범위, 방제 조치 상세
- 해양환경관리법·MARPOL 협약 근거 및 방제비 산정 논리
- 환경피해 산정 시 불확실성/추정치 범위 명시
"""
        )

    def _collision_template(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            self._common_header()
            + """
[유형별 보강 가이드]
- 충돌 상대선/항로/가시거리/속력/기관상태 등 상태 기록
- 국제충돌예방규칙(COLREGS) 및 해사안전법 적용 검토
- 선체 손상 부위/수리범위/운항중단 손실 고려
"""
        )

    def _injury_template(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            self._common_header()
            + """
[유형별 보강 가이드]
- 부상자 신원/근무상황/보호구/안전절차 준수 여부
- 선원법·산재보험 관련 조항으로 급여/보상 항목 정리
- 치료기간/후유장해 가능성/복귀 전망 명시
"""
        )

    def _generic_template(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            self._common_header()
            + """
[유형별 보강 가이드]
- 사건 특성에 맞춰 4~6항의 손해내역·근거·산정 영역을 충실히 기술
"""
        )

    # --------------------------------------------------
    # 4) 데이터 포맷팅
    # --------------------------------------------------
    def _format_incident_data(self, data: Dict[str, Any]) -> str:
        """
        JSON 사고데이터를 사람이 읽기 쉬운 라인 포맷으로 변환
        - 한국어 라벨 매핑
        """
        label_map = {
            "incident_type": "사고유형",
            "description": "설명",
            "location": "발생위치",
            "report_type": "보고서 유형",
            "language": "언어",
        }
        lines = []
        for key in ["incident_type", "description", "location", "report_type", "language"]:
            val = data.get(key)
            if val is not None and val != "":
                label = label_map.get(key, key)
                lines.append(f"- {label}: {val}")
        return "\n".join(lines) if lines else "(제공된 데이터 없음)"
