"""의사소통 전문 AI 에이전트: 검증된 데이터를 시니어 친화 문구로 가공."""
from typing import Optional

SYSTEM_PROMPT_FRIENDLY = """당신은 어르신에게 복약 안내를 전달하는 말하기 전문가입니다.
- 딱딱한 문장을 부드럽고 친근한 말투로 바꿉니다.
- "네, 어르신~", "~하시면 됩니다" 같은 존댓말을 사용합니다.
- 문장은 짧게, 한 번에 하나씩 전달하듯이 작성합니다.
- 내용은 바꾸지 말고, 표현만 시니어가 듣기 좋게 바꿉니다."""


def to_senior_friendly_text(verified_answer: str) -> str:
    """
    검증된 답변을 시니어 친화 문구로 변환.
    LLM API 미호출 시 단순 치환(예: 마침표 뒤 공백, '~' 추가)만 적용.
    """
    if not verified_answer.strip():
        return verified_answer
    # 간단 후처리: 문장 끝을 부드럽게
    text = verified_answer.strip()
    if text.endswith(".") and "어르신" not in text[:50]:
        text = "네, 어르신. " + text[0].lower() + text[1:] if len(text) > 1 else text
    return text
