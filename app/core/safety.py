"""안전·품질: 면책 문구, DUR 주의사항 강조, 개인정보 미외부유출 검증."""
import re

DISCLAIMER = "정확한 판단은 의사·약사 상담이 필요합니다."
DISCLAIMER_VARIANTS = [
    "정확한 판단은 의사",
    "약사 상담",
    "의사·약사 상담",
]


def ensure_disclaimer(text: str) -> str:
    """답변 끝에 면책 문구가 없으면 추가."""
    if not text or not text.strip():
        return DISCLAIMER
    t = text.strip()
    if any(v in t for v in DISCLAIMER_VARIANTS):
        return text
    if not t.endswith("."):
        t += "."
    return t + " " + DISCLAIMER


def contains_pii_candidates(text: str) -> bool:
    """주민번호·전화번호·이메일 등이 남아있는지 검사 (외부 전송 전 검증용)."""
    if re.search(r"\d{6}\s*[-]?\s*\d{7}", text):
        return True
    if re.search(r"01[0-9]\s*[-]?\s*\d{3,4}\s*[-]?\s*\d{4}", text):
        return True
    if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text):
        return True
    return False
