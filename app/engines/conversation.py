"""대화 엔진 (Conversation Engine) — 페르소나 및 인터페이스.

server.mermaid 매핑:
  CE_Input       → receive_input()
  CE_Latency     → generate_filler()
  CE_Tone        → apply_tone()
  CE_Conversation_Core → synthesize_response()
  CE_Response    → build_response()
"""
import logging
import random
import re
from datetime import datetime
from typing import Optional

from app.core.config import settings
from app.schemas.engine_contracts import (
    ConversationComposeRequest,
    ConversationComposeResponse,
    ReasoningMode,
)

logger = logging.getLogger(__name__)

FILLER_RESPONSES = [
    "잠시만요, 기록을 확인하고 있어요.",
    "네, 알겠습니다. 잠깐만 기다려 주세요.",
    "어르신, 제가 꼼꼼히 살펴보고 있어요~",
    "확인 중이에요. 금방 알려드릴게요.",
    "말씀하신 부분 확인하고 있습니다, 조금만 기다려 주세요.",
]

SMALLTALK_PATTERNS = {
    "greeting": [
        "어르신, 안녕하세요! 오늘 기분은 어떠세요?",
        "안녕하세요, 반갑습니다! 무엇을 도와드릴까요?",
    ],
    "feeling_bad": [
        "아이고, 오늘 컨디션이 안 좋으시군요. 많이 불편하셨겠어요.",
        "걱정되시겠네요. 제가 도울 수 있는 게 있으면 말씀해 주세요.",
    ],
    "feeling_good": [
        "그렇게 좋으시다니 정말 다행이에요!",
        "건강하게 지내고 계시니 정말 기쁘네요!",
    ],
    "thanks": [
        "별말씀을요, 언제든 편하게 물어보세요!",
        "도움이 되셨다니 다행이에요!",
    ],
}

GREETING_KEYWORDS = ["안녕", "반가", "여보세요", "하이", "hello"]
FEELING_BAD_KEYWORDS = ["아프", "어지럽", "안 좋", "힘들", "피곤", "아파", "쑤시"]
FEELING_GOOD_KEYWORDS = ["좋아", "괜찮", "건강해", "기분 좋"]
THANKS_KEYWORDS = ["고마", "감사", "땡큐", "thank"]


class ConversationEngine:
    """대화 엔진: 페르소나 적용, Latency Hiding, 톤앤매너 최적화."""

    def __init__(self):
        self.system_prompt = (
            "당신은 어르신에게 복약 안내를 전달하는 따뜻한 AI 도우미입니다.\n"
            "- 존댓말을 사용하고, 짧고 쉬운 문장으로 말합니다.\n"
            "- '어르신', '할머니', '할아버지' 등 친근한 호칭을 씁니다.\n"
            "- 의학 전문 용어를 쉬운 말로 바꿉니다.\n"
            "- 답변 끝에 항상 '정확한 판단은 의사·약사 상담이 필요합니다'를 포함합니다."
        )

    # ── CE_Input: STT 해석 데이터 수신 ──

    def receive_input(self, stt_text: str, speaker_id: Optional[str] = None) -> dict:
        """로컬 에이전트로부터 STT 결과를 수신하고 초기 분석."""
        return {
            "text": stt_text.strip(),
            "speaker_id": speaker_id,
            "timestamp": datetime.now().isoformat(),
            "is_smalltalk": self._detect_smalltalk(stt_text),
            "smalltalk_type": self._classify_smalltalk(stt_text),
        }

    # ── CE_Latency: 즉시 응답 처리 및 스몰토크 ──

    def generate_filler(self, input_data: dict) -> Optional[str]:
        """Latency Hiding: 처리 시간이 필요할 때 즉시 내보낼 filler 메시지 생성."""
        smalltalk_type = input_data.get("smalltalk_type")
        if smalltalk_type and smalltalk_type in SMALLTALK_PATTERNS:
            return random.choice(SMALLTALK_PATTERNS[smalltalk_type])
        return random.choice(FILLER_RESPONSES)

    def generate_smalltalk(self, input_data: dict) -> Optional[str]:
        """스몰토크만 필요한 경우 최종 응답 반환."""
        smalltalk_type = input_data.get("smalltalk_type")
        if smalltalk_type and smalltalk_type in SMALLTALK_PATTERNS:
            return random.choice(SMALLTALK_PATTERNS[smalltalk_type])
        return None

    # ── CE_Tone: 환자 맞춤형 언어 순화 및 최적화 ──

    def apply_tone(
        self,
        fact_data: str,
        user_profile: Optional[dict] = None,
        flash_context: Optional[str] = None,
    ) -> str:
        """추론 엔진이 전달한 팩트 데이터를 어르신 친화적 언어로 변환."""
        if not fact_data or not fact_data.strip():
            return "어르신, 죄송해요. 지금은 답변을 드리기 어렵습니다. 잠시 후 다시 말씀해 주세요."

        text = fact_data.strip()

        honorific = "어르신"
        if user_profile:
            name = user_profile.get("name", "")
            if name:
                honorific = f"{name} 어르신"

        replacements = {
            "병용 금기": "같이 드시면 안 되는 약",
            "병용금기": "같이 드시면 안 되는 약",
            "상호작용": "서로 영향을 줄 수 있는 약",
            "부작용": "몸에 안 좋은 반응",
            "금기": "주의하셔야 하는",
            "복용": "드시는 것",
            "투여": "드시는 것",
            "처방": "의사 선생님이 정해주신",
            "용량": "드시는 양",
            "효능": "약의 효과",
        }
        for medical_term, friendly_term in replacements.items():
            text = text.replace(medical_term, friendly_term)

        if not any(text.startswith(prefix) for prefix in ["네,", "어르신", honorific]):
            text = f"{honorific}, {text[0].lower() + text[1:]}" if len(text) > 1 else text

        if "의사" not in text and "약사" not in text:
            if not text.rstrip().endswith("."):
                text += "."
            text += " 정확한 판단은 의사·약사 상담이 필요합니다."

        return text

    # ── CE_Conversation_Core: 대화 정책 코어 에이전트 ──

    def synthesize_response(
        self,
        input_data: dict,
        fact_data: Optional[str] = None,
        filler_sent: bool = False,
        user_profile: Optional[dict] = None,
        flash_context: Optional[str] = None,
        apply_tone: bool = True,
    ) -> dict:
        """최종 응답을 합성. 팩트 데이터가 없으면 스몰토크로 처리."""
        if input_data.get("is_smalltalk") and not fact_data:
            response_text = self.generate_smalltalk(input_data) or "네, 듣고 있어요."
            return {
                "text": response_text,
                "type": "smalltalk",
                "requires_tts": True,
            }

        if fact_data:
            toned_text = (
                self.apply_tone(fact_data, user_profile, flash_context)
                if apply_tone
                else fact_data.strip()
            )
            return {
                "text": toned_text,
                "type": "medical_response",
                "requires_tts": True,
            }

        return {
            "text": "네, 말씀해 주세요. 듣고 있어요.",
            "type": "fallback",
            "requires_tts": True,
        }

    # ── CE_Response: 최종 응답 빌드 ──

    def build_response(self, synthesis_result: dict) -> dict:
        """WebSocket 으로 전송할 최종 응답 객체."""
        return {
            "response_text": synthesis_result["text"],
            "response_type": synthesis_result["type"],
            "requires_tts": synthesis_result.get("requires_tts", True),
            "timestamp": datetime.now().isoformat(),
        }

    def compose_from_contract(
        self,
        contract: ConversationComposeRequest,
    ) -> ConversationComposeResponse:
        """Compose elder-facing output from typed engine contracts.

        Runtime policy:
        - Never expose ``<think>`` blocks.
        - Conversation engine consumes user profile + reasoning decision
          + reviewed/delivery text candidates.
        """
        if contract.decision.mode == ReasoningMode.ASK_USER_CLARIFY:
            text = (
                "어르신, 확인이 필요한 약 이름이나 증상을 조금 더 자세히 말씀해 주세요. "
                "예: 약 이름, 하루 몇 번 드시는지, 언제부터 불편하셨는지."
            )
            return ConversationComposeResponse(
                response_text=text,
                response_type="clarify",
                requires_tts=True,
            )

        source = (
            contract.delivery_message.strip()
            or contract.reviewed_message.strip()
            or contract.core_message.strip()
        )
        source = self._strip_think_tags(source)

        if not source:
            fallback = "어르신, 현재 기록만으로는 정확한 판단이 어렵습니다."
            if contract.decision.mode == ReasoningMode.MEMORY_ONLY:
                fallback = (
                    "어르신, 지금은 가벼운 안내만 가능한 상태예요. "
                    "복용 중인 약 이름이나 처방전을 알려주시면 더 정확히 도와드릴 수 있어요."
                )
            return ConversationComposeResponse(
                response_text=self.apply_tone(
                    fallback,
                    user_profile=contract.user_profile,
                    flash_context=contract.evidence.summary if contract.evidence else None,
                ),
                response_type="fallback",
                requires_tts=True,
            )

        response_type = (
            "smalltalk"
            if contract.decision.mode == ReasoningMode.MEMORY_ONLY
            and contract.decision.intent == "smalltalk"
            else "medical_response"
        )
        text = self.apply_tone(
            source,
            user_profile=contract.user_profile,
            flash_context=contract.evidence.summary if contract.evidence else None,
        )
        return ConversationComposeResponse(
            response_text=text,
            response_type=response_type,
            requires_tts=True,
        )

    # ── 내부 유틸 ──

    def _detect_smalltalk(self, text: str) -> bool:
        text_lower = text.lower().strip()
        all_keywords = (
            GREETING_KEYWORDS
            + FEELING_BAD_KEYWORDS
            + FEELING_GOOD_KEYWORDS
            + THANKS_KEYWORDS
        )
        return any(kw in text_lower for kw in all_keywords)

    def _classify_smalltalk(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        if any(kw in text_lower for kw in GREETING_KEYWORDS):
            return "greeting"
        if any(kw in text_lower for kw in FEELING_BAD_KEYWORDS):
            return "feeling_bad"
        if any(kw in text_lower for kw in FEELING_GOOD_KEYWORDS):
            return "feeling_good"
        if any(kw in text_lower for kw in THANKS_KEYWORDS):
            return "thanks"
        return None

    def _strip_think_tags(self, text: str) -> str:
        """Remove training-only longCoT blocks from runtime responses."""
        if not text:
            return ""
        cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
        return cleaned.strip()
