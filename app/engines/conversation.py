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
from app.services.medication_extraction import is_ocr_capture_request_text
from app.services.patient_safety import classify_patient_safety_situation

logger = logging.getLogger(__name__)

FILLER_RESPONSES = [
    "잠시만요, 필요한 기록을 확인하고 있어요.",
    "확인하고 있습니다. 잠시만 기다려주세요.",
]

MEDICATION_FILLERS = [
    "복용 정보를 확인하고 있습니다. 잠시만 기다려주세요.",
    "약 정보와 기록을 확인하고 있어요.",
]

DUR_FILLERS = [
    "약 정보를 확인하고 있습니다. 잠시만 기다려주세요.",
    "같이 먹어도 되는지 확인하고 있어요.",
]

OCR_FILLERS = [
    "카메라 촬영을 준비하고 있어요.",
    "약봉투나 처방전을 잘 읽을 수 있게 촬영 준비를 하고 있어요.",
    "잠시 후 약봉투를 카메라 앞에 보여주시면 됩니다.",
]

OCR_PROCESSING_FILLERS = [
    "사진에서 읽힌 약 이름을 확인하고 있어요.",
    "약봉투 글자가 제대로 읽혔는지 살펴보고 있어요.",
    "추측해서 저장하지 않도록, 인식된 내용을 먼저 확인하고 있어요.",
]

REMINDER_FILLERS = [
    "알림 시간을 확인하고 있어요.",
    "복약 알림을 확인하고 있습니다. 잠시만 기다려주세요.",
]

RECORD_FILLERS = [
    "복용 기록을 확인하고 있어요.",
    "방금 말씀하신 내용을 기록과 맞춰보고 있어요.",
]

MEAL_FILLERS = [
    "식후에 드실 약이 있는지 저장된 기록을 확인하고 있어요.",
    "식사 후 복용 안내를 확인하고 있습니다. 잠시만 기다려주세요.",
]

SMALLTALK_PATTERNS = {
    "greeting": [
        "안녕하세요. 오늘 복약이나 컨디션 관련해서 도와드릴게요.",
        "안녕하세요, 반갑습니다. 무엇을 확인해드릴까요?",
    ],
    "feeling_bad": [
        "오늘 컨디션이 좋지 않으시군요. 많이 불편하셨겠어요.",
        "걱정되시겠어요. 증상이나 복용 중인 약을 말씀해 주시면 같이 확인해볼게요.",
    ],
    "feeling_good": [
        "좋으시다니 다행이에요. 오늘 복약도 무리 없이 챙겨볼게요.",
        "컨디션이 괜찮으시다니 좋네요. 필요한 것이 있으면 편하게 말씀해 주세요.",
    ],
    "thanks": [
        "별말씀을요. 언제든 편하게 물어보세요.",
        "도움이 되셨다니 다행이에요.",
    ],
    "acknowledgement": [
        "네, 필요하시면 또 말씀해 주세요.",
        "알겠습니다. 더 필요한 것이 있으면 말씀해 주세요.",
    ],
}

GREETING_KEYWORDS = ["안녕", "반가", "여보세요", "하이", "hello"]
FEELING_BAD_KEYWORDS = ["아프", "어지럽", "안 좋", "힘들", "피곤", "아파", "쑤시"]
FEELING_GOOD_KEYWORDS = ["좋아", "괜찮", "건강해", "기분 좋"]
THANKS_KEYWORDS = ["고마", "감사", "땡큐", "thank"]
ACKNOWLEDGEMENT_KEYWORDS = [
    "잘했",
    "수고",
    "됐어",
    "됐네",
    "됐습니다",
    "알겠",
    "알았",
    "오케이",
    "okay",
]
FAST_SMALLTALK_TYPES = {"greeting", "feeling_good", "thanks", "acknowledgement"}
FAST_SMALLTALK_RESPONSES = {
    "greeting": "안녕하세요. 무엇을 도와드릴까요?",
    "feeling_good": "다행이에요. 필요한 것이 있으면 편하게 말씀해 주세요.",
    "thanks": "별말씀을요. 언제든 편하게 말씀해 주세요.",
    "acknowledgement": "네, 필요하시면 또 말씀해 주세요.",
}


class ConversationEngine:
    """대화 엔진: 페르소나 적용, Latency Hiding, 톤앤매너 최적화."""

    def __init__(self):
        self.system_prompt = (
            "당신은 나이와 무관하게 복약 관리가 필요한 사용자를 돕는 따뜻한 AI 도우미입니다.\n"
            "- 존댓말을 사용하고, 짧고 쉬운 문장으로 말합니다.\n"
            "- 이름이 있으면 이름을 사용하고, 없으면 '사용자님'이라고 부릅니다.\n"
            "- 사용자가 직접 말했거나 프로필에 저장된 경우가 아니면 나이, 질환, 호칭을 추측하지 않습니다.\n"
            "- 보호자가 대신 말하면 사용자 본인 정보와 복약 관리 대상자 정보를 섞지 않습니다.\n"
            "- 만성질환자나 복약 사용자라는 이유만으로 '어르신'이라고 부르지 않습니다.\n"
            "- 의학 전문 용어를 쉬운 말로 바꿉니다.\n"
            "- 의료 안전 판단이 필요한 경우에만 '정확한 판단은 의사·약사 상담이 필요합니다'를 포함합니다."
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
        if input_data.get("is_smalltalk"):
            return None
        text = str(input_data.get("text") or "")
        safety = classify_patient_safety_situation(text)
        if safety and safety.severity == "emergency":
            return None
        if self._filler_category(text) == "general":
            return None
        category = self._filler_category(text)
        if category == "ocr":
            return random.choice(OCR_FILLERS)
        if category == "dur":
            return random.choice(DUR_FILLERS)
        if category == "reminder":
            return random.choice(REMINDER_FILLERS)
        if category == "record":
            return random.choice(RECORD_FILLERS)
        if category == "meal":
            return random.choice(MEAL_FILLERS)
        if category == "medication":
            return random.choice(MEDICATION_FILLERS)
        return random.choice(FILLER_RESPONSES)

    def generate_smalltalk(self, input_data: dict) -> Optional[str]:
        """스몰토크만 필요한 경우 최종 응답 반환."""
        smalltalk_type = input_data.get("smalltalk_type")
        if smalltalk_type and smalltalk_type in SMALLTALK_PATTERNS:
            return random.choice(SMALLTALK_PATTERNS[smalltalk_type])
        return None

    def fast_smalltalk_type(self, text: str) -> Optional[str]:
        """Return deterministic fast-path smalltalk type for low-risk turns."""
        if classify_patient_safety_situation(text):
            return None
        input_data = self.receive_input(text)
        smalltalk_type = input_data.get("smalltalk_type")
        if not input_data.get("is_smalltalk") or smalltalk_type not in FAST_SMALLTALK_TYPES:
            return None
        return str(smalltalk_type)

    def build_smalltalk_fast_response(
        self,
        text: str,
        user_profile: Optional[dict] = None,
    ) -> str:
        """Build a deterministic TTS-ready response without LLM polishing."""
        smalltalk_type = self.fast_smalltalk_type(text) or "greeting"
        response = FAST_SMALLTALK_RESPONSES.get(
            smalltalk_type,
            "네, 말씀해 주세요.",
        )
        return self._ensure_user_prefix(response, user_profile)

    def build_wake_word_response(self, user_profile: Optional[dict] = None) -> str:
        """Return a deterministic wake-word acknowledgement.

        Wake-word-only turns should never go through LLM generation because they
        are not medical questions. Keep the response short and profile-aware.
        """
        return f"네, {self._honorific(user_profile)}. 말씀하세요."

    # ── CE_Tone: 사용자 맞춤형 언어 순화 및 최적화 ──

    def apply_tone(
        self,
        fact_data: str,
        user_profile: Optional[dict] = None,
        flash_context: Optional[str] = None,
        require_disclaimer: Optional[bool] = None,
    ) -> str:
        """추론 엔진이 전달한 팩트 데이터를 사용자 친화적 언어로 변환."""
        if not fact_data or not fact_data.strip():
            return "사용자님, 죄송해요. 지금은 답변을 드리기 어렵습니다. 잠시 후 다시 말씀해 주세요."

        text = fact_data.strip()
        safety_source = text
        del flash_context

        honorific = self._honorific(user_profile)
        text = self._replace_generic_honorific(text, honorific)

        replacements = {
            "병용 금기": "같이 드시면 안 되는 약",
            "병용금기": "같이 드시면 안 되는 약",
            "상호작용": "서로 영향을 줄 수 있는 약",
            "부작용": "몸에 안 좋은 반응",
            "용량주의": "드시는 양 주의",
            "투여기간주의": "복용 기간 주의",
            "복용량": "드시는 양",
            "용량": "드시는 양",
            "효능": "약의 효과",
        }
        for medical_term, friendly_term in replacements.items():
            text = text.replace(medical_term, friendly_term)

        if not any(text.startswith(prefix) for prefix in ["네,", "어르신", "사용자님", honorific]):
            text = f"{honorific}, {text}" if text else text

        should_disclaim = (
            self._looks_like_medical_safety_answer(f"{safety_source}\n{text}")
            if require_disclaimer is None
            else require_disclaimer
        )
        if should_disclaim and "의사·약사 상담" not in text and "전문가" not in text:
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
        """Compose user-facing output from typed engine contracts.

        Runtime policy:
        - Never expose ``<think>`` blocks.
        - Conversation engine consumes user profile + reasoning decision
          + reviewed/delivery text candidates.
        """
        if contract.decision.mode == ReasoningMode.ASK_USER_CLARIFY:
            if self._is_ocr_capture_request(contract.input_text):
                text = (
                    f"{self._honorific(contract.user_profile)}, 알겠습니다. "
                    "카메라 앞으로 약봉투나 처방전을 잘 보이게 보여주세요. "
                    "글자가 흔들리지 않도록 잠시만 멈춰주세요. "
                    "5, 4, 3, 2, 1. 촬영하겠습니다."
                )
            else:
                text = (
                    f"{self._honorific(contract.user_profile)}, 확인이 필요한 약 이름이나 증상을 조금 더 자세히 말씀해 주세요. "
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
        if self._is_profile_recall_text(contract.input_text):
            return ConversationComposeResponse(
                response_text=self._profile_recall_response(contract.user_profile),
                response_type="profile_recall",
                requires_tts=True,
            )

        if not source:
            fallback = "사용자님, 현재 기록만으로는 정확한 판단이 어렵습니다."
            if contract.decision.mode == ReasoningMode.MEMORY_ONLY:
                fallback = (
                    "사용자님, 지금은 가벼운 안내만 가능한 상태예요. "
                    "복용 중인 약 이름이나 처방전을 알려주시면 더 정확히 도와드릴 수 있어요."
                )
            return ConversationComposeResponse(
                response_text=self.apply_tone(
                    fallback,
                    user_profile=contract.user_profile,
                    flash_context=contract.evidence.summary if contract.evidence else None,
                    require_disclaimer=contract.decision.mode != ReasoningMode.MEMORY_ONLY,
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
        if response_type == "smalltalk":
            return ConversationComposeResponse(
                response_text=self._ensure_user_prefix(source, contract.user_profile),
                response_type=response_type,
                requires_tts=True,
            )
        if (
            contract.decision.mode == ReasoningMode.MEMORY_ONLY
            and self._is_memory_ack_or_recall(contract.input_text)
        ):
            return ConversationComposeResponse(
                response_text=self._ensure_user_prefix(source, contract.user_profile),
                response_type=response_type,
                requires_tts=True,
            )
        text = self.apply_tone(
            source,
            user_profile=contract.user_profile,
            flash_context=contract.evidence.summary if contract.evidence else None,
            require_disclaimer=self._requires_disclaimer_for_contract(contract, source),
        )
        return ConversationComposeResponse(
            response_text=text,
            response_type=response_type,
            requires_tts=True,
        )

    # ── 내부 유틸 ──

    def _detect_smalltalk(self, text: str) -> bool:
        from app.services.medication_extraction import is_wake_word_only

        if is_wake_word_only(text):
            return True

        text_lower = text.lower().strip()
        if self._has_medication_or_task_signal(text_lower):
            return False
        all_keywords = (
            GREETING_KEYWORDS
            + FEELING_BAD_KEYWORDS
            + FEELING_GOOD_KEYWORDS
            + THANKS_KEYWORDS
            + ACKNOWLEDGEMENT_KEYWORDS
        )
        return any(kw in text_lower for kw in all_keywords)

    def _classify_smalltalk(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        if self._has_medication_or_task_signal(text_lower):
            return None
        if any(kw in text_lower for kw in GREETING_KEYWORDS):
            return "greeting"
        if any(kw in text_lower for kw in FEELING_BAD_KEYWORDS):
            return "feeling_bad"
        if any(kw in text_lower for kw in FEELING_GOOD_KEYWORDS):
            return "feeling_good"
        if any(kw in text_lower for kw in THANKS_KEYWORDS):
            return "thanks"
        if any(kw in text_lower for kw in ACKNOWLEDGEMENT_KEYWORDS):
            return "acknowledgement"
        return None

    @staticmethod
    def _has_medication_or_task_signal(text: str) -> bool:
        return any(
            token in text
            for token in (
                "약",
                "복용",
                "처방",
                "먹어도",
                "먹으면",
                "먹을게",
                "먹을께",
                "먹겠습니다",
                "먹을게요",
                "같이 먹",
                "드셔도",
                "부작용",
                "금기",
                "주의",
                "용량",
                "식후",
                "식전",
                "알림",
                "기록",
                "ocr",
                "사진",
                "약봉투",
                "처방전",
                "촬영",
                "찍",
                "와파린",
                "아스피린",
                "영양제",
                "건강기능식품",
            )
        )

    @staticmethod
    def _filler_category(text: str) -> str:
        lowered = text.lower()
        if is_ocr_capture_request_text(text):
            return "ocr"
        if ConversationEngine._is_after_meal_completion_signal(text):
            return "meal"
        if any(token in lowered for token in ("사진", "ocr", "약봉투", "처방전", "촬영", "찍")):
            return "ocr"
        if any(token in lowered for token in ("알림", "알람", "예약", "깨워", "챙겨", "몇 시", "시간 바꿔", "시간 변경")) or (
            any(meal in lowered for meal in ("아침", "점심", "저녁"))
            and re.search(r"\d{1,2}\s*시", lowered)
        ):
            return "reminder"
        if any(token in lowered for token in ("먹었어", "먹었나", "복용했", "기록")):
            return "record"
        if any(token in lowered for token in ("같이 먹", "병용", "상호작용", "두 번", "더 빨리", "녹용", "오메가3", "건강기능식품", "영양제", "dur")):
            return "dur"
        if any(token in lowered for token in ("약", "복용", "처방", "식후", "식전", "밥", "아침", "점심", "저녁")):
            return "medication"
        return "general"

    @staticmethod
    def _is_after_meal_completion_signal(text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "").lower()
        if not compact:
            return False
        if any(token in compact for token in ("약먹", "약복용", "복용했")):
            return False
        meal_signal = any(
            token in compact
            for token in ("밥", "식사", "아침", "점심", "저녁", "식후")
        )
        done_signal = any(
            token in compact
            for token in (
                "먹었",
                "먹고왔",
                "먹고옴",
                "다먹",
                "먹음",
                "식사했",
                "식사끝",
                "식사마쳤",
                "먹고나",
            )
        )
        return meal_signal and done_signal

    @staticmethod
    def _is_ocr_capture_request(text: str) -> bool:
        return is_ocr_capture_request_text(text)

    def _strip_think_tags(self, text: str) -> str:
        """Remove training-only longCoT blocks from runtime responses."""
        if not text:
            return ""
        cleaned = re.sub(r"<think\b[^>]*>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<think\b[^>]*>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    def _ensure_user_prefix(self, text: str, user_profile: Optional[dict] = None) -> str:
        text = (text or "").strip()
        if not text:
            return f"{self._honorific(user_profile)}, 듣고 있어요."
        honorific = self._honorific(user_profile)
        text = self._replace_generic_honorific(text, honorific)
        if text.startswith(("어르신", "사용자님", "네,", honorific)):
            return text
        return f"{honorific}, {text}"

    def _is_memory_ack_or_recall(self, text: str) -> bool:
        lowered = text.lower()
        compact = re.sub(r"\s+", "", lowered)
        if "ocr" in lowered or "읽힌 처방 약 이름" in text:
            return False
        if self._is_profile_recall_text(text):
            return True
        return any(
            token in lowered
            for token in (
                "기록해",
                "기록한",
                "뭐였",
                "다시 말",
                "누구인지",
                "내가 누구",
                "먹었어",
                "먹었나",
                "알림",
                "저장",
            )
        )

    @staticmethod
    def _is_profile_recall_text(text: str) -> bool:
        compact = re.sub(r"\s+", "", (text or "").lower())
        return any(
            token in compact
            for token in ("누군지", "누구인지", "내가누구", "나누구", "내이름", "내프로필")
        )

    def _profile_recall_response(self, user_profile: Optional[dict]) -> str:
        profile = user_profile or {}
        name = str(profile.get("name") or "").strip()
        if not name:
            return "아직 등록된 이름이 없어요. 이름, 성별, 나이를 말씀해 주시면 기억하겠습니다."
        details: list[str] = []
        gender = str(profile.get("gender") or "").strip()
        age = str(profile.get("age") or "").strip()
        if gender:
            details.append(gender)
        if age:
            details.append(f"{age}세")
        conditions = profile.get("conditions") or []
        if conditions:
            details.append("기저질환 " + ", ".join(str(item) for item in conditions))
        if details:
            return f"{name}님이십니다. 저장된 정보는 {', '.join(details)}입니다."
        return f"{name}님이십니다."

    @staticmethod
    def _honorific(user_profile: Optional[dict] = None) -> str:
        name = str((user_profile or {}).get("name") or "").strip()
        return f"{name}님" if name else "사용자님"

    @staticmethod
    def _replace_unconfirmed_elder_honorific(text: str, honorific: str) -> str:
        return ConversationEngine._replace_generic_honorific(text, honorific)

    @staticmethod
    def _replace_generic_honorific(text: str, honorific: str) -> str:
        cleaned = (text or "").replace("어르신", honorific)
        if honorific and honorific != "사용자님":
            cleaned = cleaned.replace("사용자님", honorific)
        return cleaned

    @staticmethod
    def _looks_like_medical_safety_answer(text: str) -> bool:
        lowered = text.lower()
        return any(
            token in lowered
            for token in (
                "위험",
                "금기",
                "주의",
                "부작용",
                "저혈압",
                "출혈",
                "상담",
                "전문가",
                "임의",
                "중단",
                "용량",
                "복용량",
                "dur",
            )
        )

    def _requires_disclaimer_for_contract(
        self,
        contract: ConversationComposeRequest,
        source: str,
    ) -> bool:
        lowered = contract.input_text.lower()
        if contract.decision.intent == "smalltalk":
            return False
        if contract.decision.mode == ReasoningMode.MEMORY_ONLY:
            return False
        if any(token in lowered for token in ("알림", "찍", "촬영", "사진", "저장", "먹었어")):
            return False
        if any(token in lowered for token in ("두 번", "많이", "더 빨리", "녹용", "건강기능식품", "영양제", "같이 먹")):
            return True
        return self._looks_like_medical_safety_answer(source)
