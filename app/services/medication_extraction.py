"""Medication name extraction helpers with wake-word / non-drug filtering."""
from __future__ import annotations

import re

# Keep in sync with local_agent/src/edge_node/vad.py WAKE_WORDS
WAKE_WORDS: tuple[str, ...] = (
    "오디스야",
    "오디세이",
    "오딧세이",
    "오디스요",
    "오디스아",
    "오디스여",
    "오디스",
    "오디세",
    "오딧스",
    "오딧세",
    "오디즈",
    "오디쓰",
    "오디수",
    "오티스",
    "오티즈",
    "오티쓰",
    "오티세",
    "오티세이",
    "오지스",
    "오지즈",
    "오지쓰",
    "우디스",
    "우디즈",
    "우디",
    "우디야",
    "워디스",
    "워디즈",
    "워디",
    "아디스",
    "아디즈",
    "아디",
    "오리스",
    "보리스",
    "보리쓰",
    "보디스",
    "보디즈",
    "저기",
    "얘야",
    "어디스",
    "어딧스",
    "어디쓰",
    "어디즈",
)

WAKE_WORD_ONLY_ALIASES: tuple[str, ...] = (
    "야",
    "오디",
    "우디",
    "워디",
    "아디",
    "여보세요",
    "들려",
    "잘들려",
    "듣고있어",
    "듣고있니",
    "내말들려",
    "어디서",
    "오디서",
    "우디서",
    "워디서",
)

_WAKE_WORD_FUZZY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[오어우워아]\s*디\s*[스즈쓰수]?"),
    re.compile(r"오\s*딧\s*[스세]"),
    re.compile(r"오\s*티\s*[스즈쓰]"),
    re.compile(r"오\s*지\s*[스즈쓰]"),
    re.compile(r"[오우워]\s*디\s*서"),
    re.compile(r"보\s*리\s*[스쓰]"),
    re.compile(r"보\s*디\s*[스즈]"),
)

NON_MEDICATION_TOKENS: frozenset[str] = frozenset(
    {
        *WAKE_WORDS,
        *WAKE_WORD_ONLY_ALIASES,
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
    if _compact_call_text(cleaned) in WAKE_WORD_ONLY_ALIASES:
        return ""
    for wake in sorted(WAKE_WORDS, key=len, reverse=True):
        cleaned = cleaned.replace(wake, " ")
    cleaned = re.sub(r"^\s*(야|오디)[\s,.!?~，。]+", " ", cleaned)
    for pattern in _WAKE_WORD_FUZZY_PATTERNS:
        cleaned = pattern.sub(" ", cleaned, count=1)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.!?~")
    return cleaned


def is_wake_word_only(text: str) -> bool:
    """True when the utterance is only a wake word (optionally with punctuation)."""
    raw = (text or "").strip()
    if not raw:
        return False
    if _compact_call_text(raw) in WAKE_WORD_ONLY_ALIASES:
        return True
    if not any(wake and wake in raw for wake in WAKE_WORDS) and not any(
        pattern.search(raw) for pattern in _WAKE_WORD_FUZZY_PATTERNS
    ):
        return False
    normalized = strip_wake_words(raw)
    return not normalized


def _compact_call_text(text: str) -> str:
    return re.sub(r"[\s\t\r\n.,;:!?~'\"`，。]+", "", (text or "").strip().lower())


OCR_CAPTURE_OBJECT_TOKENS: tuple[str, ...] = (
    "처방전",
    "약봉투",
    "약봉지",
    "약 사진",
    "약사진",
    "사진",
    "카메라",
    "ocr",
)

OCR_CAPTURE_ACTION_TOKENS: tuple[str, ...] = (
    "읽어서",
    "읽어",
    "읽혀",
    "찍",
    "촬영",
    "보여",
    "등록",
    "저장",
    "켜",
    "준비",
    "대",
    "ocr",
)

OCR_CAPTURE_RESULT_TOKENS: tuple[str, ...] = (
    "결과",
    "인식",
    "읽힌",
    "읽혔",
    "추출",
)

NEW_MEDICATION_CAPTURE_PATTERNS: tuple[str, ...] = (
    "새약받",
    "새약타",
    "새약가져",
    "새약처방",
    "새로운약받",
    "새로운약타",
    "약받아왔",
    "약받아옴",
    "약을받아왔",
    "약을받아옴",
    "약받았",
    "약을받았",
    "약타왔",
    "약타옴",
    "약을타왔",
    "약을타옴",
    "약새로받",
    "약새로타",
    "처방받아왔",
    "처방받았",
    "처방받음",
    "처방나왔",
    "처방전받",
    "새처방받",
    "새처방나왔",
    "약봉투받",
    "약봉지받",
)


def is_ocr_capture_request_text(text: str) -> bool:
    """Detect utterances that should start prescription/package OCR capture.

    Elderly users often say "새 약 받아왔어" or "약 타왔어" without saying
    "사진" or "카메라". Treat those as OCR setup requests when no concrete drug
    name has been provided yet.
    """
    lowered = (text or "").lower().strip()
    if not lowered:
        return False

    compact = re.sub(r"[\s\t\r\n.,;:!?~'\"`]+", "", lowered)
    if not compact:
        return False

    result_context = any(token in lowered for token in OCR_CAPTURE_RESULT_TOKENS) or any(
        token in compact for token in OCR_CAPTURE_RESULT_TOKENS
    )
    wants_recapture = any(
        token in lowered for token in ("다시 찍", "다시 촬영", "재촬영", "한 번 더", "한번 더")
    ) or any(token in compact for token in ("다시찍", "다시촬영", "재촬영", "한번더"))
    if result_context and not wants_recapture:
        return False

    explicit_capture = (
        any(token in lowered for token in OCR_CAPTURE_OBJECT_TOKENS)
        and any(token in lowered for token in OCR_CAPTURE_ACTION_TOKENS)
    )
    new_medication_capture = any(
        pattern in compact for pattern in NEW_MEDICATION_CAPTURE_PATTERNS
    )
    return explicit_capture or new_medication_capture


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
