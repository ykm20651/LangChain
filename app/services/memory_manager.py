from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict #JSON 직렬화/복원할 수 있게 도와주는 함수들
from typing import Dict, Optional 
import uuid

class MemoryManager:
    """
    LangChain 기반 멀티 세션 대화 메모리 관리자.
    각 세션별로 ConversationBufferMemory 인스턴스를 보관한다.
    """

    def __init__(self):
        # 세션ID별 메모리 저장소 (딕셔너리)
        self.sessions: Dict[str, ConversationBufferMemory] = {}
        """
        { -> 이렇게 저장됨 
            "session_1": <ConversationBufferMemory>,
            "session_2": <ConversationBufferMemory>
        }

        """

    def ensure(self, session_id: Optional[str]) -> str: # '-> str' 리턴 타입 힌트구나.
        """
        세션ID가 주어지지 않으면 새로 생성하고, 해당 세션용 메모리를 준비한다.
        """
        sid = session_id or str(uuid.uuid4())
        if sid not in self.sessions:
            # 새 세션 생성 시 LangChain 메모리 초기화
            self.sessions[sid] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return sid

    def add_user(self, sid: str, text: str):
        """
        특정 세션의 메모리에 사용자 메시지를 추가한다.
        """
        if sid not in self.sessions:
            self.ensure(sid)
        self.sessions[sid].chat_memory.add_user_message(text)

    def add_ai(self, sid: str, text: str):
        """
        특정 세션의 메모리에 AI 메시지를 추가한다.
        """
        if sid not in self.sessions:
            self.ensure(sid)
        self.sessions[sid].chat_memory.add_ai_message(text)

    def history_text(self, sid: str) -> str:
        """
        특정 세션의 대화 히스토리를 문자열로 반환한다.
        LangChain 내부 메시지 리스트를 사람이 읽기 쉽게 포맷팅.
        """
        if sid not in self.sessions:
            return ""
        messages = self.sessions[sid].chat_memory.messages
        formatted = []
        for m in messages[-20:]:  # 최근 20턴만 유지
            role = "사용자" if m.type == "human" else "AI"
            formatted.append(f"{role}: {m.content}")
        return "\n".join(formatted)

    def export_session(self, sid: str):
        """
        (선택) 세션 기록을 JSON 직렬화된 딕셔너리 형태로 내보내기.
        FastAPI에서 파일로 저장하거나 복원할 때 유용.
        """
        if sid not in self.sessions:
            return []
        return messages_to_dict(self.sessions[sid].chat_memory.messages)

    def import_session(self, sid: str, data):
        """
        (선택) 외부에서 JSON 형태로 불러온 대화 기록을 복원.
        """
        self.sessions[sid] = ConversationBufferMemory(return_messages=True)
        self.sessions[sid].chat_memory.messages = messages_from_dict(data)

    def clear(self, sid: str):
        """
        특정 세션 메모리 초기화.
        """
        if sid in self.sessions:
            del self.sessions[sid]
