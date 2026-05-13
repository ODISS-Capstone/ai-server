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
