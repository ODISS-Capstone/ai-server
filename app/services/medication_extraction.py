"""Medication name extraction helpers with wake-word / non-drug filtering."""
from __future__ import annotations

import re

# Keep in sync with local_agent/src/edge_node/vad.py WAKE_WORDS
WAKE_WORDS: tuple[str, ...] = (
    "오디스야",
    "오디스",
    "저기",
    "얘야",
    "오디",
    "어디스",
)

NON_MEDICATION_TOKENS: frozenset[str] = frozenset(
    {
        *WAKE_WORDS,
        "네",
        "예",
        "응",
        "그래",
        "맞아",
        "맞습니다",
        "아니",
        "아니요",
        "어르신",
        "사용자",
        "사용자님",
    }
)

_MEDICATION_SUFFIX_RE = re.compile(
    r"([가-힣A-Za-z0-9]+(?:장용정|정|캡슐|시럽))"
)


def strip_wake_words(text: str) -> str:
    """Remove wake-word tokens from user text before medication/DUR routing."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    for wake in sorted(WAKE_WORDS, key=len, reverse=True):
        cleaned = cleaned.replace(wake, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.!?~")
    return cleaned


def is_wake_word_only(text: str) -> bool:
    """True when the utterance is only a wake word (optionally with punctuation)."""
    normalized = strip_wake_words(text)
    return not normalized


def is_non_medication_token(token: str) -> bool:
    normalized = re.sub(r"[\s\t\r\n]+", "", (token or "").strip())
    normalized = re.sub(r"[.,;:!?~'\"`]+", "", normalized)
    if not normalized or len(normalized) < 2:
        return True
    if normalized in NON_MEDICATION_TOKENS:
        return True
    return any(normalized == wake or normalized.startswith(wake) for wake in WAKE_WORDS)


def filter_drug_name_candidates(names: list[str]) -> list[str]:
    unique: list[str] = []
    for name in names:
        cleaned = (name or "").strip(".,!?()[]{} ")
        if not cleaned or is_non_medication_token(cleaned):
            continue
        if cleaned not in unique:
            unique.append(cleaned)
    return unique


def extract_medication_suffix_tokens(text: str) -> list[str]:
    """Extract tokens that look like drug product names (정/캡슐/시럽 suffix)."""
    return filter_drug_name_candidates(
        match.group(1) for match in _MEDICATION_SUFFIX_RE.finditer(text or "")
    )
