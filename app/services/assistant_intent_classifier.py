"""State-aware routing contract for ODISS assistant turns."""
from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any

from app.services.medication_extraction import (
    is_ocr_capture_request_text,
    is_wake_word_only,
    strip_wake_words,
)
from app.services.patient_safety import classify_patient_safety_situation
from app.services.reminders import ReminderService


RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_EMERGENCY = "emergency"


@dataclass(frozen=True)
class AssistantRouteDecision:
    route_label: str
    engine_scope: str
    risk_level: str = RISK_LOW
    fast_path: str = ""
    ui_action: str = ""
    db_write_expected: bool = False
    active_flow: str = "none"
    route_reason: str = ""
    response_text: str = ""
    paused_flow: str = ""


class AssistantIntentClassifier:
    """Classify a user utterance before identity, workflow, or LLM routing.

    The classifier is intentionally deterministic. It is used both by the
    runtime and by route-contract tests, so it must not call external services.
    """

    def __init__(self, *, allow_wakeless_emergency: bool | None = None) -> None:
        if allow_wakeless_emergency is None:
            allow_wakeless_emergency = os.getenv("ODISS_WAKELESS_EMERGENCY", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.allow_wakeless_emergency = allow_wakeless_emergency

    def classify(
        self,
        text: str,
        *,
        active_flow: str = "none",
        active_session: bool = True,
        client_context: dict[str, Any] | None = None,
        odiss_directed: bool = False,
    ) -> AssistantRouteDecision:
        raw = (text or "").strip()
        compact = _compact(raw)
        flow = active_flow or "none"
        client_context = client_context or {}
        if not compact:
            return AssistantRouteDecision(
                route_label="ignored_noise",
                engine_scope="ignored",
                risk_level=RISK_LOW,
                active_flow=flow,
                route_reason="empty_or_silence",
            )

        has_wake = _has_wake_signal(raw)
        session_active = bool(
            active_session
            or odiss_directed
            or has_wake
            or flow not in {"", "none", "assistant_social"}
            or _client_context_active(client_context)
        )

        safety = self._safety_precheck(raw, active_flow=flow)
        if safety and (
            session_active
            or has_wake
            or odiss_directed
            or (self.allow_wakeless_emergency and _is_high_confidence_emergency(raw))
        ):
            return safety

        if not session_active and not has_wake:
            return AssistantRouteDecision(
                route_label="ignored_background",
                engine_scope="ignored",
                risk_level=RISK_LOW,
                active_flow=flow,
                route_reason="inactive_no_wake",
            )

        if is_wake_word_only(raw):
            return AssistantRouteDecision(
                route_label="wake_word",
                engine_scope="conversation_engine",
                fast_path="wake_word",
                active_flow="assistant_social",
                route_reason="wake_word_only",
            )

        wake_stripped = strip_wake_words(raw)
        if has_wake and _is_cancel_or_stop(raw) and not _has_medication_or_symptom_signal(wake_stripped):
            return AssistantRouteDecision(
                route_label="wake_cancel",
                engine_scope="conversation_engine",
                fast_path="assistant_stop",
                active_flow="none",
                route_reason="wake_cancel",
            )
        if has_wake and _is_wake_status_or_plain_call(raw):
            return AssistantRouteDecision(
                route_label="wake_status",
                engine_scope="conversation_engine",
                fast_path="wake_word",
                active_flow="assistant_social",
                route_reason="wake_status_or_stt_variant",
            )

        reasoning = self._reasoning_precheck(raw, active_flow=flow)
        if reasoning:
            return reasoning

        stateful = self._active_flow_short_utterance(raw, flow)
        if stateful:
            return stateful

        workflow = self._workflow_route(raw, active_flow=flow)
        if workflow:
            return workflow

        social = self._conversation_route(raw, active_flow=flow)
        if social:
            return social

        return AssistantRouteDecision(
            route_label="unclear",
            engine_scope="conversation_engine",
            risk_level=RISK_LOW,
            active_flow=flow,
            route_reason="short_unclear_without_context",
            response_text="어떤 약이나 알림을 말씀하시는지 한 번만 더 알려주세요.",
        )

    def _safety_precheck(self, text: str, *, active_flow: str) -> AssistantRouteDecision | None:
        compact = _compact(text)
        emergency_response = (
            "응급 신호일 수 있습니다. 지금은 약을 고르지 말고 즉시 119에 연락하거나 응급실로 가세요."
        )
        if _is_emergency(text):
            return AssistantRouteDecision(
                route_label="emergency",
                engine_scope="safety",
                risk_level=RISK_EMERGENCY,
                fast_path="global_safety_precheck",
                active_flow="emergency",
                route_reason="emergency_phrase",
                response_text=emergency_response,
                paused_flow=active_flow if active_flow not in {"", "none"} else "",
            )
        if _is_third_party_medication_risk(text):
            return AssistantRouteDecision(
                route_label="third_party_medication",
                engine_scope="safety",
                risk_level=RISK_HIGH,
                fast_path="global_safety_precheck",
                active_flow="emergency",
                route_reason="third_party_medication",
                response_text="다른 사람 약은 드시면 안 됩니다. 약봉투 이름을 확인하고, 이미 드셨다면 약사나 119에 바로 확인하세요.",
                paused_flow=active_flow if active_flow not in {"", "none"} else "",
            )
        situation = classify_patient_safety_situation(text)
        if not situation:
            if _is_excess_or_duplicate_medication(compact):
                return AssistantRouteDecision(
                    route_label="medication_safety",
                    engine_scope="safety",
                    risk_level=RISK_HIGH,
                    fast_path="global_safety_precheck",
                    active_flow="emergency",
                    route_reason="duplicate_or_excess_dose",
                    response_text="지금은 한 번 더 드시지 마세요. 약봉투의 1회 용량과 마지막 복용 시간을 먼저 확인하세요.",
                    paused_flow=active_flow if active_flow not in {"", "none"} else "",
                )
            return None
        if situation.severity == "emergency":
            route_label = "emergency"
            risk_level = RISK_EMERGENCY
        elif situation.key in {"wrong_person_medication", "acetaminophen_excess_dose", "extra_or_double_dose"}:
            route_label = "medication_safety" if situation.key != "wrong_person_medication" else "third_party_medication"
            risk_level = RISK_HIGH
        else:
            return None
        return AssistantRouteDecision(
            route_label=route_label,
            engine_scope="safety",
            risk_level=risk_level,
            fast_path="global_safety_precheck",
            active_flow="emergency",
            route_reason=situation.key,
            response_text=_shorten_safety_response(situation.response_text),
            paused_flow=active_flow if active_flow not in {"", "none"} else "",
        )

    def _reasoning_precheck(self, text: str, *, active_flow: str) -> AssistantRouteDecision | None:
        compact = _compact(strip_wake_words(text))
        if not compact:
            return None
        if _is_emergency(text):
            return self._safety_precheck(text, active_flow=active_flow)
        if _is_third_party_medication_risk(text):
            return self._safety_precheck(text, active_flow=active_flow)
        if _is_missed_or_uncertain_dose(compact):
            return AssistantRouteDecision(
                route_label="missed_or_uncertain_dose",
                engine_scope="reasoning_engine",
                risk_level=RISK_HIGH,
                active_flow=active_flow,
                route_reason="missed_or_uncertain_dose",
            )
        if _is_excess_or_duplicate_medication(compact):
            return AssistantRouteDecision(
                route_label="medication_safety",
                engine_scope="reasoning_engine",
                risk_level=RISK_HIGH,
                active_flow=active_flow,
                route_reason="duplicate_or_excess_dose",
            )
        if _is_medication_can_take_question(compact):
            return AssistantRouteDecision(
                route_label="medication_can_take",
                engine_scope="reasoning_engine",
                risk_level=RISK_HIGH if _has_chronic_or_high_risk_med(compact) else RISK_MEDIUM,
                active_flow=active_flow,
                route_reason="medication_can_take",
            )
        if _is_meal_medication_question(compact):
            return AssistantRouteDecision(
                route_label="meal_medication_guidance",
                engine_scope="reasoning_engine",
                risk_level=RISK_MEDIUM,
                active_flow="medication_guidance",
                route_reason="meal_medication",
            )
        if _has_symptom_signal(compact):
            return AssistantRouteDecision(
                route_label="symptom_medication_question",
                engine_scope="reasoning_engine",
                risk_level=RISK_MEDIUM,
                active_flow=active_flow,
                route_reason="symptom_signal",
            )
        return None

    def _active_flow_short_utterance(self, text: str, active_flow: str) -> AssistantRouteDecision | None:
        compact = _compact(text)
        kind = _short_reply_kind(compact)
        if active_flow == "ocr_confirm":
            if _is_recapture(text):
                return _workflow("ocr_retry", "ocr_retry", RISK_LOW, "ocr_confirm", ui_action="open_camera")
            if kind == "negative" or _is_save_reject(text):
                return _workflow("ocr_reject", "ocr_save_reject", RISK_MEDIUM, "none", ui_action="close_camera")
            if kind == "affirmative" or _is_save_confirm(text):
                return _workflow("ocr_confirm", "ocr_save_confirm", RISK_MEDIUM, "none", db_write=True)
            return _workflow("ocr_confirmation_followup", "", RISK_MEDIUM, "ocr_confirm")
        if active_flow in {"ocr_camera", "ocr"} and (_is_camera_cancel(text) or kind in {"negative", "stop"}):
            return _workflow("ocr_cancel", "assistant_camera_cancel", RISK_LOW, "none", ui_action="close_camera")
        if active_flow == "identity":
            if kind in {"affirmative", "negative"}:
                return _workflow("identity_confirm", "identity_followup", RISK_MEDIUM, "identity")
            if _looks_like_identity_registration(text):
                return _workflow("identity_register", "identity_register", RISK_MEDIUM, "identity", db_write=True)
            if _is_identity_switch(text):
                return _workflow("identity_switch", "identity_switch", RISK_MEDIUM, "identity", db_write=True)
            return _workflow("identity_followup", "identity_followup", RISK_MEDIUM, "identity")
        if active_flow == "medication_guidance":
            if _is_taken_record(text):
                return _workflow("medication_taken_record", "medication_taken_record", RISK_LOW, "medication_record", db_write=True)
            if _is_taken_recall(text):
                return _workflow("medication_taken_recall", "medication_taken_recall", RISK_MEDIUM, "medication_guidance")
            if _has_symptom_signal(compact):
                return AssistantRouteDecision("symptom_medication_question", "reasoning_engine", RISK_MEDIUM, active_flow=active_flow, route_reason="post_taken_symptom")
            if any(token in compact for token in ("기록", "체크", "먹긴먹", "먹은거", "먹는소리")):
                return _workflow("medication_taken_record", "medication_taken_record", RISK_LOW, "medication_record", db_write=True)
            return _workflow("medication_guidance", "stored_medication_guidance", RISK_MEDIUM, "medication_guidance")
        if active_flow == "medication_record":
            if _is_time_correction(text):
                return _workflow("medication_taken_correction", "medication_taken_time_correction", RISK_MEDIUM, "medication_record", db_write=True)
            if kind in {"negative", "stop"}:
                return _workflow("medication_record_cancel", "medication_record_cancel", RISK_LOW, "none")
            if _is_taken_recall(text) or any(token in compact for token in ("기록", "복약표", "지난주", "이번달", "보여", "읽어", "확인")):
                return _workflow("medication_taken_recall", "medication_taken_recall", RISK_MEDIUM, "medication_record")
            if any(token in compact for token in ("어제", "날짜", "먹은", "먹었", "병원가기", "빠진")):
                if any(token in compact for token in ("적었", "바꿔", "고쳐", "수정")):
                    return _workflow("medication_taken_correction", "medication_taken_time_correction", RISK_MEDIUM, "medication_record", db_write=True)
                return _workflow("medication_taken_recall", "medication_taken_recall", RISK_MEDIUM, "medication_record")
        if active_flow == "reminder":
            if ReminderService.is_missed_one_shot_check(text):
                return _workflow("missed_reminder_check", "missed_one_shot_check", RISK_LOW, "reminder")
            if kind == "negative" or _is_reminder_cancel(text):
                return _workflow("reminder_cancel", "reminder_control", RISK_LOW, "none", db_write=True)
            if kind == "affirmative":
                return _workflow("reminder_confirm", "reminder_confirm", RISK_LOW, "reminder", db_write=True)
            if _is_reminder_create(text):
                return _workflow("reminder_create", "reminder_setup", RISK_LOW, "reminder", db_write=True)
            if any(token in compact for token in ("소리", "크게", "작게", "진동", "주말", "일요일", "새벽", "밤", "울리")):
                return _workflow("reminder_update", "reminder_setup", RISK_LOW, "reminder", db_write=True)
            if any(token in compact for token in ("9시로", "알림목록", "알람목록", "울린알람", "취소되돌", "되돌려")):
                return _workflow("reminder_update", "reminder_setup", RISK_LOW, "reminder", db_write=True)
        if kind == "repeat":
            return AssistantRouteDecision("assistant_repeat", "conversation_engine", RISK_LOW, "assistant_repeat", active_flow=active_flow, route_reason="repeat")
        if kind == "stop":
            return AssistantRouteDecision("assistant_stop", "conversation_engine", RISK_LOW, "assistant_stop", active_flow="none", route_reason="stop")
        if kind in {"affirmative", "negative"}:
            return AssistantRouteDecision("assistant_acknowledgement", "conversation_engine", RISK_LOW, "assistant_acknowledgement", active_flow="assistant_social", route_reason=kind)
        return None

    def _workflow_route(self, text: str, *, active_flow: str) -> AssistantRouteDecision | None:
        compact = _compact(strip_wake_words(text))
        if is_ocr_capture_request_text(text) or _is_ocr_capture_like(compact):
            return _workflow("ocr_capture", "ocr_capture", RISK_LOW, "ocr_camera", ui_action="open_camera")
        if _looks_like_identity_registration(text):
            return _workflow("identity_register", "identity_register", RISK_MEDIUM, "identity", db_write=True)
        if _is_identity_recall(text):
            return _workflow("identity_recall", "profile_recall", RISK_LOW, "identity")
        if _is_identity_switch(text):
            return _workflow("identity_switch", "identity_switch", RISK_MEDIUM, "identity", db_write=True)
        if ReminderService.is_relative_alarm_request(text) or _is_reminder_create(text):
            return _workflow("reminder_create", "relative_alarm" if ReminderService.extract_relative_delay(text) else "reminder_setup", RISK_LOW, "reminder", db_write=True)
        if ReminderService.is_missed_one_shot_check(text):
            return _workflow("missed_reminder_check", "missed_one_shot_check", RISK_LOW, "reminder")
        if _is_reminder_cancel(text):
            return _workflow("reminder_cancel", "reminder_control", RISK_LOW, "none", db_write=True)
        if _is_time_correction(text):
            return _workflow("medication_taken_correction", "medication_taken_time_correction", RISK_MEDIUM, "medication_record", db_write=True)
        if _is_taken_recall(text):
            return _workflow("medication_taken_recall", "medication_taken_recall", RISK_MEDIUM, "medication_guidance")
        if _is_taken_record(text):
            return _workflow("medication_taken_record", "medication_taken_record", RISK_LOW, "medication_record", db_write=True)
        if _is_current_medication_query(compact):
            return _workflow("current_medication_lookup", "stored_medication_guidance", RISK_MEDIUM, "medication_guidance")
        if _is_caregiver_workflow(text):
            return _workflow("caregiver_workflow", "caregiver_workflow", RISK_MEDIUM, active_flow)
        return None

    def _conversation_route(self, text: str, *, active_flow: str) -> AssistantRouteDecision | None:
        compact = _compact(text)
        if _is_capability(text):
            return AssistantRouteDecision("assistant_capability", "conversation_engine", RISK_LOW, "smalltalk", active_flow="assistant_social", route_reason="capability")
        if _is_suggestion(text):
            return AssistantRouteDecision("assistant_suggestion", "conversation_engine", RISK_LOW, "smalltalk", active_flow="assistant_social", route_reason="suggestion")
        if _is_companion(text):
            return AssistantRouteDecision("assistant_social", "conversation_engine", RISK_MEDIUM if _is_emotional_support(text) else RISK_LOW, "smalltalk", active_flow="assistant_social", route_reason="assistant_social")
        if _is_unsupported(text):
            risk = RISK_HIGH if _is_dangerous_unsupported(text) else RISK_LOW
            return AssistantRouteDecision("unsupported_but_answered", "conversation_engine", risk, "smalltalk", active_flow="assistant_social", route_reason="unsupported")
        if any(token in compact for token in ("안녕", "고마", "감사", "잘했", "그래", "좋아", "어딨어", "뭐해", "듣고있")):
            return AssistantRouteDecision("assistant_social", "conversation_engine", RISK_LOW, "smalltalk", active_flow="assistant_social", route_reason="social")
        return None


def _workflow(
    route_label: str,
    fast_path: str,
    risk_level: str,
    active_flow: str,
    *,
    ui_action: str = "",
    db_write: bool = False,
) -> AssistantRouteDecision:
    return AssistantRouteDecision(
        route_label=route_label,
        engine_scope="workflow",
        risk_level=risk_level,
        fast_path=fast_path,
        ui_action=ui_action,
        db_write_expected=db_write,
        active_flow=active_flow,
        route_reason=route_label,
    )


def _compact(text: str) -> str:
    return re.sub(r"[\s\t\r\n.,;:!?~'\"`，。…]+", "", (text or "").strip().lower())


def _has_wake_signal(text: str) -> bool:
    raw = text or ""
    return is_wake_word_only(raw) or strip_wake_words(raw) != raw or _is_wakeish(_compact(raw))


def _is_wakeish(compact: str) -> bool:
    return any(
        token in compact
        for token in (
            "오디스",
            "오디야",
            "오디씨",
            "오디세",
            "오디즈",
            "오딧",
            "오티스",
            "오지스",
            "어디스",
            "보리스",
            "보디스",
            "약비서",
            "약도우미",
            "비서야",
            "복약아",
            "오디쓰",
        )
    )


def _is_wake_status_or_plain_call(text: str) -> bool:
    compact = _compact(text)
    if _is_excess_or_duplicate_medication(compact) or _is_medication_can_take_question(compact):
        return False
    return _is_wakeish(compact) or any(token in compact for token in ("들리", "대답안", "켜진거", "나말한다", "말좀들어", "깨워"))


def _client_context_active(context: dict[str, Any]) -> bool:
    if context.get("active_session") is False or context.get("voice_armed") is False:
        return False
    camera_mode = str(context.get("camera_mode") or "")
    return bool(
        context.get("active_session")
        or context.get("voice_armed")
        or context.get("listening")
        or context.get("ocr_busy")
        or camera_mode not in {"", "idle"}
    )


def _is_high_confidence_emergency(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("숨이안쉬", "숨을못쉬", "가슴이너무아파", "혀가부", "목이부", "119불러", "응급실"))


def _is_emergency(text: str) -> bool:
    compact = _compact(text)
    emergency_tokens = (
        "숨이안쉬",
        "숨을못쉬",
        "숨쉬기힘",
        "호흡곤란",
        "숨이잘안쉬",
        "가슴이꽉",
        "가슴이아파",
        "가슴아파",
        "가슴답답",
        "가슴통증",
        "흉통",
        "혀가부",
        "목이부",
        "입술이부",
        "입술이갑자기부",
        "두드러기",
        "피를토",
        "피토",
        "변이새까",
        "소변에피",
        "머리를부딪",
        "정신이몽롱",
        "두명으로보",
        "심장이너무빨리",
        "맥박이상",
        "맥박이이상",
        "혈압이너무낮",
        "혈압이200",
        "혈당이50",
        "열이39",
        "숨차고",
        "가래에피",
        "계속토",
        "물도못마셨",
        "소변이하루종일안",
        "소변이안나",
        "다리가갑자기",
        "종아리가한쪽",
        "대답하기가힘",
        "말이꼬",
        "한쪽팔",
        "얼굴이한쪽",
        "눈앞이안보",
        "의식",
        "쓰러",
        "경련",
        "119",
        "응급실",
    )
    return any(token in compact for token in emergency_tokens)


def _has_symptom_signal(compact: str) -> bool:
    return any(token in compact for token in ("아파", "통증", "어지", "속쓰", "두통", "열나", "토할", "토해서", "목에걸", "메스꺼", "잠안와", "기침", "복통", "울렁", "졸려", "설사"))


def _has_medication_or_symptom_signal(text: str) -> bool:
    compact = _compact(text)
    return _has_medication_signal(compact) or _has_symptom_signal(compact)


def _has_medication_signal(compact: str) -> bool:
    return any(token in compact for token in ("약", "복용", "처방", "타이레놀", "아세트아미노펜", "혈압", "당뇨", "와파린", "아스피린", "감기약", "수면제", "진통제"))


def _is_third_party_medication_risk(text: str) -> bool:
    compact = _compact(text)
    other_person = any(token in compact for token in ("남편약", "아내약", "엄마약", "어머니약", "아빠약", "아버지약", "남의약", "다른사람약", "손자감기약", "할아버지약", "아이한테", "아기가", "어른약", "강아지"))
    ingestion = any(token in compact for token in ("내가먹", "먹어도", "먹여도", "먹이면", "먹었", "먹으면", "섞였", "내약", "복용"))
    return other_person and ingestion


def _is_excess_or_duplicate_medication(compact: str) -> bool:
    dose = any(token in compact for token in ("두번", "또먹", "한번더", "더먹", "두알", "두개", "세알", "세개", "네알", "4개", "여러알", "한번에", "동시에", "중복", "같이먹", "함께먹", "섞어먹", "두봉투", "겹치", "해열제", "하루몇", "두배", "하나더", "대신", "자몽", "두봉투섞", "두군데"))
    return dose and any(token in compact for token in ("먹", "복용", "삼켜", "되", "괜찮"))


def _is_missed_or_uncertain_dose(compact: str) -> bool:
    uncertainty_context = any(token in compact for token in ("물컵", "손에들", "손에들고", "남았", "잠결", "바닥", "봉투가그대로", "뜯겨", "하얀약", "하얀약하나", "파란약", "흰약", "어제약", "어제약을오늘", "토했는데", "다시먹어야"))
    if not _has_medication_signal(compact) and "그거" not in compact and not uncertainty_context:
        return False
    return any(token in compact for token in ("먹었나", "먹었는지모르", "기억안", "헷갈", "까먹", "놓쳤", "못먹", "빼먹", "뜯겨있", "그대로"))


def _is_medication_can_take_question(compact: str) -> bool:
    food_or_form = any(token in compact for token in ("우유랑", "커피", "술", "운전", "운동", "캡슐", "까서", "쪼개", "검사", "병원가기전", "병원가기", "혈압이낮", "혈압약먹어", "떨어졌", "소화제", "피묽게", "피를묽게"))
    if not _has_medication_signal(compact) and not food_or_form:
        return False
    return any(token in compact for token in ("먹어도돼", "먹어도되", "먹어도될까", "먹어도되나", "복용해도", "먹을까", "괜찮", "문제없", "먹나"))


def _has_chronic_or_high_risk_med(compact: str) -> bool:
    return any(token in compact for token in ("혈압", "당뇨", "인슐린", "수면제", "와파린", "아스피린", "항응고", "심장"))


def _is_meal_medication_question(compact: str) -> bool:
    return any(token in compact for token in ("밥먹", "아침먹", "점심먹", "저녁먹", "식후", "식전", "공복", "밥안먹", "식사", "죽한", "간식", "아침안먹", "밥늦", "명절", "삼십분")) and any(
        token in compact for token in ("약", "먹", "복용", "뭐", "되", "쳐", "해놔", "시간")
    )


def _short_reply_kind(compact: str) -> str:
    if compact in {"네", "예", "응", "어", "그래", "맞아", "맞습니다", "어맞아", "응맞아", "네맞아", "예맞아"}:
        return "affirmative"
    if compact in {"아니", "아냐", "아니야", "아니요"} or compact.startswith(("아니", "아냐")):
        return "negative"
    if compact in {"그만", "됐어", "잠깐만", "잠깐", "멈춰", "취소"}:
        return "stop"
    if any(token in compact for token in ("다시말", "한번더", "못들", "방금뭐", "다시알려")):
        return "repeat"
    return ""


def _is_cancel_or_stop(text: str) -> bool:
    return _short_reply_kind(_compact(text)) in {"negative", "stop"} or any(token in _compact(text) for token in ("꺼져", "끄고", "안불렀", "부른거아냐"))


def _is_camera_cancel(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("사진안찍", "안찍", "찍지마", "카메라꺼", "카메라닫", "촬영취소", "사진취소", "사진그만", "필요없"))


def _is_save_confirm(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("저장", "맞아", "확인", "그대로")) or _short_reply_kind(compact) == "affirmative"


def _is_save_reject(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("저장하지", "저장안", "틀렸", "삭제", "취소", "아니", "아냐"))


def _is_recapture(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("다시찍", "재촬영", "흐려", "다시"))


def _looks_like_identity_registration(text: str) -> bool:
    compact = _compact(text)
    return bool(re.search(r"\d{1,3}세", text or "")) or any(token in compact for token in ("내이름", "처음쓰", "등록", "남성이", "여성이", "남자", "여자", "생년월일", "생일", "전화번호", "번호", "할머니", "순자", "박씨", "김씨", "이름", "목소리"))


def _is_identity_recall(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("나누구", "누군지알", "내가누구", "등록된사람"))


def _is_identity_switch(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("나아니", "다른사람", "새로등록", "사람바꿔", "이름틀렸", "나말고", "아니고", "지워", "삭제"))


def _is_taken_record(text: str) -> bool:
    if ReminderService.is_taken_confirmation(text):
        return True
    compact = _compact(text)
    if _is_medication_can_take_question(compact) or _is_missed_or_uncertain_dose(compact):
        return False
    return any(token in compact for token in ("먹었어", "먹었다", "방금먹", "지금먹", "삼켰", "복용했", "먹은걸로", "기록해"))


def _is_taken_recall(text: str) -> bool:
    if ReminderService.is_taken_recall(text):
        return True
    compact = _compact(text)
    return any(token in compact for token in ("언제먹", "몇시에먹", "기록돼", "기록되", "오늘몇번", "먹은거맞", "먹었다고했"))


def _is_time_correction(text: str) -> bool:
    if ReminderService.is_taken_time_correction(text):
        return True
    compact = _compact(text)
    return any(token in compact for token in ("시간틀", "아니", "수정", "고쳐", "7시", "8시", "분이야", "중복", "하나지워")) and any(
        token in compact for token in ("기록", "먹", "시", "분", "지워", "수정")
    )


def _is_reminder_create(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("알림", "알람", "알려", "말해", "깨워", "챙겨", "울려", "울린")) and any(
        token in compact for token in ("약", "복용", "아침", "점심", "저녁", "식후", "매일", "시간", "뒤", "9시", "까먹", "폰잠", "인터넷", "글자")
    )


def _is_reminder_cancel(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("알람꺼", "알림꺼", "알람좀꺼", "알림좀꺼", "알람빼", "알림빼", "알람취소", "알림취소", "시끄러워서꺼"))


def _is_current_medication_query(compact: str) -> bool:
    return any(token in compact for token in ("무슨약", "저장된약", "먹어야되는약", "오늘약", "약있나", "뭐먹지", "뭐먹어"))


def _is_ocr_capture_like(compact: str) -> bool:
    return any(token in compact for token in ("약봉투", "약봉지", "처방전", "카메라", "사진찍", "찍을게", "이약이름", "봉다리", "봉투글씨", "글자가", "찰칵", "앞면", "뒷면", "알약색", "알약", "읽어봐", "종이이거찍", "손떨", "불좀켜", "한장씩", "빛이번쩍", "화면에서찍", "멀리둬", "가까이가야"))


def _is_caregiver_workflow(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("엄마약시간", "아버지약먹었", "딸한테", "아들한테", "보호자", "엄마약", "아빠약", "딸이볼", "누가내약기록", "요양보호사", "간병인", "아들말", "딸이약", "아이한테", "강아지", "아들번호지워", "가족끼리약공유")) and not _is_third_party_medication_risk(text)


def _is_capability(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("뭐할수있", "뭐도와", "무엇을도와", "뭐해줘", "사용법", "어떻게써", "말하면알아듣", "뭘말하면", "약알려면", "사진은어디", "마이크", "기록도볼", "알람은어떻게", "잘못말하면", "화면이뭔말", "설명해", "하나씩말", "못하겠", "버튼", "dur", "복약이뭐", "너의사", "병원대신", "위험하면", "저장돼", "소리안", "매일써", "돈내"))


def _is_suggestion(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("뭐하면좋", "뭐하지", "뭘하면", "도와줘", "뭐해야", "어떻게하면"))


def _is_companion(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("심심", "얘기좀", "이야기", "말동무", "외로", "허전", "기분이별로", "칭찬", "지겹", "잠이잘안와", "혼자", "귀찮", "눈물이", "화가나", "말좀들", "다행", "고맙", "무섭", "겁나", "서럽", "젊을때", "전화안하", "사는게", "하기싫", "바보된", "이해했", "천천히말", "큰소리", "작게말"))


def _is_emotional_support(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("외로", "기분이별로", "허전", "지겹", "잠이잘안와", "무섭", "불안"))


def _is_unsupported(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("택시", "치킨", "유튜브", "노래", "tv", "티비", "불좀꺼", "송금", "돈보내", "로또", "병원예약", "뉴스", "날씨", "주식", "비밀번호", "카톡", "문자", "버스", "장보기", "냉장고", "녹음", "약값", "수면제더구", "거짓말", "병원속이", "보험청구", "진단서", "병명정", "처방해", "항생제", "주사", "대신버려", "물떠다", "문잠가", "응급이라고"))


def _is_dangerous_unsupported(text: str) -> bool:
    compact = _compact(text)
    return any(token in compact for token in ("처방전없이", "병원속이", "약안먹고", "남몰래", "불법"))


def _shorten_safety_response(text: str) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。])\s+", text or "") if part.strip()]
    if not sentences:
        return "지금은 더 드시지 말고 약봉투를 확인하세요. 심한 증상이 있으면 119에 연락하세요."
    return " ".join(sentences[:2])
