"""의사소통 전문 AI 에이전트: 검증된 데이터를 사용자 친화 문구로 가공."""
from typing import Optional

SYSTEM_PROMPT_FRIENDLY = """당신은 복약 관리가 필요한 사용자에게 안내를 전달하는 말하기 전문가입니다.
- 딱딱한 문장을 부드럽고 친근한 말투로 바꿉니다.
- 이름을 알면 이름으로 부르고, 없으면 "사용자님"이라고 부릅니다.
- 문장은 짧게, 한 번에 하나씩 전달하듯이 작성합니다.
- 내용은 바꾸지 말고, 표현만 사용자가 이해하기 좋게 바꿉니다."""


def to_senior_friendly_text(verified_answer: str) -> str:
    """
    검증된 답변을 사용자 친화 문구로 변환.
    LLM API 미호출 시 단순 치환(예: 마침표 뒤 공백, '~' 추가)만 적용.
    """
    if not verified_answer.strip():
        return verified_answer
    # 간단 후처리: 문장 끝을 부드럽게
    text = verified_answer.strip()
    if text.endswith(".") and "사용자님" not in text[:50]:
        text = "네, 사용자님. " + text[0].lower() + text[1:] if len(text) > 1 else text
    return text
