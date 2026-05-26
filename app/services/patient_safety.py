"""Deterministic safety handling for common medication-use mistakes."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PatientSafetySituation:
    key: str
    severity: str
    response_text: str
    should_record_incident: bool = True


EMERGENCY_SYMPTOMS = (
    "숨이 차",
    "숨쉬기 힘",
    "숨을 못",
    "호흡 곤란",
    "호흡곤란",
    "의식이 없",
    "의식 저하",
    "의식을 잃",
    "쓰러",
    "경련",
    "가슴 통증",
    "가슴이 아파",
    "가슴 아파",
    "가슴이 아프",
    "가슴 아프",
    "가슴이 아픔",
    "가슴 아픔",
    "가슴이 답답",
    "가슴 답답",
    "가슴이 꽉",
    "가슴이 조",
    "가슴 조",
    "가슴이 눌",
    "가슴 눌",
    "가슴 압박",
    "가슴이 무겁",
    "가슴 무겁",
    "가슴이 뻐근",
    "가슴 뻐근",
    "가슴이 찌릿",
    "가슴 찌릿",
    "가슴이 쑤",
    "가슴 쑤",
    "명치가 답답",
    "명치 답답",
    "흉통",
    "식은땀",
    "왼팔",
    "턱이 아",
    "턱 통증",
    "얼굴이 붓",
    "얼굴 붓",
    "혀가 붓",
    "목이 붓",
    "입술이 붓",
    "심한 출혈",
    "피가 멈추지",
    "검은 변",
    "피 토",
)

MEDICATION_SIGNALS = (
    "약",
    "정",
    "캡슐",
    "시럽",
    "알약",
    "혈압약",
    "당뇨약",
    "인슐린",
    "와파린",
    "아스피린",
    "타이레놀",
    "아세트아미노펜",
    "진통제",
    "해열제",
    "감기약",
    "수면제",
    "항생제",
)

ACETAMINOPHEN_SIGNALS = (
    "타이레놀",
    "아세트아미노펜",
    "acetaminophen",
    "tylenol",
)

EXCESS_DOSE_TOKENS = (
    "두 번",
    "2번",
    "한 번 더",
    "한번 더",
    "또 먹",
    "많이 먹",
    "과다",
    "초과",
    "정해진 양보다",
    "두 알",
    "2알",
    "두 개",
    "두개",
    "두 정",
    "두정",
    "두 캡슐",
    "두캡슐",
    "세 알",
    "세알",
    "세 개",
    "세개",
    "세 정",
    "세정",
    "여러 알",
    "여러알",
    "여러 개",
    "여러개",
    "한 번에",
    "한번에",
    "한꺼번에",
)

INGESTION_TOKENS = (
    "먹었",
    "먹어도",
    "먹을까",
    "먹으면",
    "먹으려",
    "먹은",
    "먹어버",
    "복용",
    "삼켜",
    "드셔도",
    "드실까",
    "드시려",
    "드시면",
)

MULTI_UNIT_DOSE_RE = re.compile(r"(?:[2-9]|[1-9]\d+|두|세|네|다섯|여섯|일곱|여덟|아홉|여러)\s*(?:개|알|정|캡슐)")


def classify_patient_safety_situation(text: str) -> PatientSafetySituation | None:
    """Classify high-risk user utterances without relying on an LLM."""
    normalized = " ".join((text or "").strip().split())
    lowered = normalized.lower()
    if not normalized:
        return None

    if _has_any(normalized, EMERGENCY_SYMPTOMS):
        return PatientSafetySituation(
            key="emergency_symptom_after_medication",
            severity="emergency",
            response_text=(
                "가슴 통증이나 답답함은 응급 신호일 수 있습니다. 어떤 약을 드실지 고르기보다 지금 즉시 119에 연락하거나 가까운 응급실로 이동하세요. "
                "혼자 이동하지 말고, 가능하면 보호자에게 알리고 드신 약봉투나 약통을 함께 가져가세요."
            ),
        )

    if not _has_medication_signal(normalized):
        return None

    if _has_any(normalized, ("다른 사람 약", "남의 약", "아내 약", "남편 약", "어머니 약", "아버지 약", "엄마 약", "아빠 약")):
        return PatientSafetySituation(
            key="wrong_person_medication",
            severity="urgent",
            response_text=(
                "다른 사람의 약을 드셨다면 더 이상 복용하지 마세요. 약 이름, 먹은 시간, 먹은 양을 적고 약봉투를 확인하세요. "
                "어지러움, 숨참, 얼굴이나 입술 부음, 심한 두근거림 같은 증상이 있으면 즉시 119에 연락하세요. "
                "증상이 없어도 의사나 약사에게 바로 확인하는 것이 안전합니다."
            ),
        )

    if _has_any(lowered, ("깜빡", "놓쳤", "못 먹", "빼먹", "잊었", "시간 지났")):
        return PatientSafetySituation(
            key="missed_dose",
            severity="caution",
            response_text=(
                "복용을 놓쳤다고 해서 다음에 두 번 드시면 안 됩니다. "
                "약봉투나 처방 안내에 '놓쳤을 때' 지시가 있으면 그 지시를 우선으로 따르세요. "
                "다음 복용 시간이 가까우면 건너뛰는 경우가 많지만, 약마다 달라서 확실하지 않으면 의사나 약사에게 확인하세요."
            ),
        )

    if _has_any(
        lowered,
        (
            "먹었는지 기억",
            "먹었는지 모르",
            "먹었나 모르",
            "먹은 것 같",
            "먹은거 같",
            "먹은 것같",
            "먹은거같",
            "먹은 듯",
            "먹은듯",
            "복용했는지 모르",
            "헷갈",
            "기억 안 나",
        ),
    ):
        return PatientSafetySituation(
            key="uncertain_taken",
            severity="caution",
            response_text=(
                "복용했는지 확실하지 않을 때는 바로 한 번 더 드시지 마세요. "
                "약통, 약봉투, 복용 기록, 알림 기록을 먼저 확인하세요. "
                "확인해도 모르겠거나 꼭 시간 맞춰 먹어야 하는 약이라면 의사나 약사에게 문의하는 것이 안전합니다."
            ),
        )

    if (_has_any(lowered, EXCESS_DOSE_TOKENS) or _has_multi_unit_dose(normalized)) and _has_any(
        lowered,
        INGESTION_TOKENS,
    ):
        if _has_any(lowered, ACETAMINOPHEN_SIGNALS):
            return PatientSafetySituation(
                key="acetaminophen_excess_dose",
                severity="urgent",
                response_text=(
                    "타이레놀이나 아세트아미노펜 성분 약을 한 번에 여러 알 드시는 것은 위험할 수 있습니다. "
                    "간 손상 위험이 있으니 정해진 용법보다 한 번에 많이 드시지 마세요. "
                    "이미 많이 드셨거나 몇 알까지 가능한지 확실하지 않다면 약봉투나 제품 포장을 들고 의사나 약사에게 바로 확인하세요. "
                    "심한 메스꺼움, 구토, 복통, 의식 저하가 있거나 이미 많이 복용했다면 즉시 119나 응급실에 연락하세요."
                ),
            )
        return PatientSafetySituation(
            key="extra_or_double_dose",
            severity="urgent",
            response_text=(
                "정해진 양보다 많이 드셨을 수 있습니다. 지금은 추가로 더 드시지 마세요. "
                "약 이름, 먹은 시간, 먹은 양을 적고 약봉투를 옆에 두세요. "
                "숨참, 의식 저하, 심한 어지러움, 가슴 통증, 얼굴이나 입술 부음이 있으면 즉시 119에 연락하세요. "
                "증상이 없어도 복용량을 스스로 맞추려 하지 말고 의사나 약사에게 확인하세요."
            ),
        )

    if _has_any(lowered, ("공복", "밥 안 먹고", "식전 약을 식후", "식후 약을 식전", "술", "음주", "자몽", "우유랑")):
        return PatientSafetySituation(
            key="wrong_food_or_timing",
            severity="caution",
            response_text=(
                "복용 시간이나 음식 조건이 달랐다고 해서 임의로 약을 더 드시지는 마세요. "
                "약봉투의 식전·식후 안내와 주의 스티커를 확인하고, 속쓰림이나 어지러움 같은 증상이 있으면 약사나 의사에게 확인하세요. "
                "호흡 곤란, 심한 어지러움, 의식 저하가 있으면 119에 연락하세요."
            ),
        )

    if _has_any(lowered, ("끊어도", "중단", "안 먹을래", "용량 바꿔", "양 줄", "양 늘", "반으로", "쪼개")):
        return PatientSafetySituation(
            key="self_stop_or_dose_change",
            severity="caution",
            response_text=(
                "만성질환 약은 증상이 괜찮아 보여도 임의로 끊거나 양을 바꾸면 위험할 수 있습니다. "
                "오늘 복용이 불편했던 이유를 기록해두고, 처방한 의사나 약사와 먼저 상의하세요. "
                "심한 알레르기 증상이나 호흡 곤란이 있으면 즉시 119에 연락하세요."
            ),
        )

    if _has_any(lowered, ("유통기한", "기한 지난", "오래된 약", "무슨 약인지 모르", "정체를 모르", "라벨이 없어", "약봉투가 없어")):
        return PatientSafetySituation(
            key="unknown_or_expired_medication",
            severity="caution",
            response_text=(
                "정체를 모르거나 유효기간이 지난 약은 드시지 않는 것이 안전합니다. "
                "약봉투, 처방전, 약 모양을 확인하고 약사에게 확인받으세요. "
                "이미 드셨고 이상 증상이 있으면 119 또는 가까운 의료기관에 바로 연락하세요."
            ),
        )

    return None


def _has_medication_signal(text: str) -> bool:
    return _has_any(text, MEDICATION_SIGNALS)


def _has_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _has_multi_unit_dose(text: str) -> bool:
    return bool(MULTI_UNIT_DOSE_RE.search(text or ""))
