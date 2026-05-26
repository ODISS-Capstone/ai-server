"""Patient safety classifier false-positive regression tests."""
from __future__ import annotations

from app.engines.llm_judge import LLMJudgeEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import IntentType, ReasoningEngine
from app.schemas.engine_contracts import ReasoningMode, ReasoningRouteInput
from app.services.patient_safety import classify_patient_safety_situation


def make_reasoning() -> ReasoningEngine:
    return ReasoningEngine(MemoryEngine(), LLMJudgeEngine())


def test_bleeding_risk_question_is_not_forced_to_emergency():
    engine = make_reasoning()

    decision = engine.route_execution(
        ReasoningRouteInput(
            text="와파린이랑 아스피린 같이 먹으면 출혈 위험이 있는지 알려줘",
            context={},
        )
    )

    assert decision.intent == IntentType.MEDICATION_QUERY
    assert decision.mode == ReasoningMode.TOOL_FIRST


def test_general_breathing_exercise_question_is_not_emergency():
    engine = make_reasoning()

    assert classify_patient_safety_situation("호흡 운동 방법 알려줘") is None
    assert engine.classify_intent("호흡 운동 방법 알려줘") == IntentType.SMALLTALK


def test_actual_breathing_distress_after_medication_is_emergency():
    engine = make_reasoning()

    situation = classify_patient_safety_situation("약 먹고 호흡 곤란이 있어")
    decision = engine.route_execution(
        ReasoningRouteInput(text="약 먹고 호흡 곤란이 있어", context={})
    )

    assert situation is not None
    assert situation.severity == "emergency"
    assert decision.intent == IntentType.EMERGENCY
    assert decision.mode == ReasoningMode.FRONTIER_FIRST


def test_chest_tightness_medicine_request_is_emergency_not_drug_recommendation():
    engine = make_reasoning()

    text = "내가 가슴이 답답해서 그런데 어떤 약을 먹는게 좋지"
    situation = classify_patient_safety_situation(text)
    decision = engine.route_execution(ReasoningRouteInput(text=text, context={}))

    assert situation is not None
    assert situation.severity == "emergency"
    assert "119" in situation.response_text
    assert decision.intent == IntentType.EMERGENCY
    assert decision.mode == ReasoningMode.FRONTIER_FIRST


def test_acetaminophen_multi_tablet_question_uses_deterministic_safety():
    engine = make_reasoning()

    text = "혹시 내가 머리가 아파서그런데 타이레놀 4개를 한번에 먹어도 문제가 없을까"
    situation = classify_patient_safety_situation(text)
    decision = engine.route_execution(ReasoningRouteInput(text=text, context={}))

    assert situation is not None
    assert situation.key == "acetaminophen_excess_dose"
    assert "간 손상" in situation.response_text
    assert decision.intent == IntentType.MEDICATION_QUERY
    assert decision.mode == ReasoningMode.MEMORY_ONLY


def test_korean_unit_count_double_blood_pressure_dose_uses_safety_route():
    engine = make_reasoning()

    text = "나 혈압약 두 개 동시에 먹어도 돼?"
    situation = classify_patient_safety_situation(text)
    decision = engine.route_execution(ReasoningRouteInput(text=text, context={}))

    assert situation is not None
    assert situation.key == "extra_or_double_dose"
    assert "추가로 더 드시지" in situation.response_text
    assert decision.intent == IntentType.MEDICATION_QUERY
    assert decision.mode == ReasoningMode.MEMORY_ONLY
