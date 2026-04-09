"""개인정보 검열: 외부 모델로 보내기 전 식별 가능 정보 마스킹·삭제."""
import re
from typing import Optional


# 주민번호 패턴 (일부 마스킹)
REGEX_SSN = re.compile(r"\d{6}\s*[-]?\s*\d{7}")
# 전화번호 유사
REGEX_PHONE = re.compile(r"01[0-9]\s*[-]?\s*\d{3,4}\s*[-]?\s*\d{4}")
# 이메일
REGEX_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def censor_text(text: str) -> str:
    """
    규칙 기반으로 개인정보를 마스킹.
    이름·병원명 등은 별도 NER 또는 내부 LLM으로 제거하는 것을 권장.
    """
    out = text
    out = REGEX_SSN.sub("[주민번호 마스킹]", out)
    out = REGEX_PHONE.sub("[전화번호 마스킹]", out)
    out = REGEX_EMAIL.sub("[이메일 마스킹]", out)
    return out


def extract_censored_for_external(query_text: str, llm_doc: str, internal_answer: str) -> str:
    """
    내부 LLM 응답 + 질문 + 문서 중 개인정보를 검열한 뒤 외부 모델에 넘길 페이로드 생성.
    """
    combined = f"[질문]\n{query_text}\n\n[복약 요약]\n{llm_doc}\n\n[1차 답변]\n{internal_answer}"
    return censor_text(combined)
