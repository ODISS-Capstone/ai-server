"""Scenario regression tests for elderly-style imperfect ODISS utterances."""
from __future__ import annotations

import asyncio

import pytest

from app.database.md_store import MDStore
from app.engines.conversation import ConversationEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import ReasoningEngine
from app.memory import StructuredMemoryService
from app.schemas.engine_contracts import ReasoningMode
from app.services.engine_orchestrator import EngineOrchestrator


def run(coro):
    return asyncio.run(coro)


def make_memory(tmp_path) -> MemoryEngine:
    engine = MemoryEngine()
    engine.store = MDStore(str(tmp_path / "md_database"))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))
    run(engine.initialize())
    return engine


async def _disable_local_llm_route(**kwargs):
    return {"usable": False, "source": "scenario_test_no_llm"}


def make_orchestrator(memory: MemoryEngine) -> EngineOrchestrator:
    judge = LLMJudgeEngine()
    return EngineOrchestrator(
        memory_engine=memory,
        reasoning_engine=ReasoningEngine(memory, judge),
        conversation_engine=ConversationEngine(),
        llm_judge=judge,
    )


def seed_elderly_user(memory: MemoryEngine, speaker_id: str) -> None:
    run(
        memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "gender": "남성", "age": "72", "conditions": ["고혈압"]},
            mark_verified=True,
        )
    )
    run(
        memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n- 와파린정\n",
        )
    )


def remember_turn(memory: MemoryEngine, speaker_id: str, result) -> None:
    run(
        memory.update_and_compress(
            {
                "query": result.input_data["text"],
                "answer": result.conversation.response_text,
                "type": result.decision.intent,
                "core_message": result.core_message,
                "dur_results": result.execution_results.get("task_results", {}).get("dur"),
            },
            speaker_id=speaker_id,
        )
    )


ELDERLY_DIALOG_SCENARIOS = [
    {
        "id": "new_medication_with_wake_word",
        "text": "오디스 나 새약 받아왔어",
        "mode": ReasoningMode.TOOL_FIRST,
        "rationale": "ocr_capture_requested",
        "tasks": ["request_ocr"],
        "must": ["카메라 앞으로", "5, 4, 3, 2, 1"],
    },
    {
        "id": "new_medication_colloquial_pharmacy",
        "text": "병원에서 약 타왔는데",
        "mode": ReasoningMode.TOOL_FIRST,
        "rationale": "ocr_capture_requested",
        "tasks": ["request_ocr"],
        "must": ["카메라 앞으로", "약봉투"],
    },
    {
        "id": "new_prescription_colloquial",
        "text": "오늘 처방 나왔어",
        "mode": ReasoningMode.TOOL_FIRST,
        "rationale": "ocr_capture_requested",
        "tasks": ["request_ocr"],
        "must": ["카메라 앞으로", "약봉투"],
    },
    {
        "id": "where_to_hold_medication_package",
        "text": "약봉투 어디다 대면 돼?",
        "mode": ReasoningMode.TOOL_FIRST,
        "rationale": "ocr_capture_requested",
        "tasks": ["request_ocr"],
        "must": ["카메라 앞으로", "약봉투"],
    },
    {
        "id": "package_text_not_visible",
        "text": "약봉투 글씨가 잘 안 보여",
        "mode": ReasoningMode.TOOL_FIRST,
        "rationale": "ocr_capture_requested",
        "tasks": ["request_ocr"],
        "must": ["카메라 앞으로", "약봉투"],
    },
    {
        "id": "meal_done_short",
        "text": "점심 먹었어",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale": "stored_medication_meal_guidance",
        "tasks": [],
        "must": ["점심 식후", "혈압약"],
    },
    {
        "id": "meal_done_ultra_short",
        "text": "점심 다 먹음",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale": "stored_medication_meal_guidance",
        "tasks": [],
        "must": ["점심 식후", "혈압약"],
    },
    {
        "id": "meal_done_with_omitted_medication",
        "text": "밥 먹었는데 그거 뭐였지",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale": "stored_medication_meal_guidance",
        "tasks": [],
        "must": ["식후", "혈압약"],
    },
    {
        "id": "missed_morning_dose",
        "text": "아침약 못 먹었어 지금 먹어?",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale_prefix": "deterministic_patient_safety:missed_dose",
        "tasks": [],
        "must": ["두 번", "약봉투"],
    },
    {
        "id": "double_blood_pressure_dose",
        "text": "혈압약 두 번 먹으면 더 빨리 좋아져?",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale_prefix": "deterministic_patient_safety:extra_or_double_dose",
        "tasks": [],
        "must": ["추가로 더 드시지", "119"],
    },
    {
        "id": "wrong_person_medication",
        "text": "남편 약 하나 먹어도 돼?",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale_prefix": "deterministic_patient_safety:wrong_person_medication",
        "tasks": [],
        "must": ["다른 사람의 약", "더 이상 복용하지"],
    },
    {
        "id": "expired_medication",
        "text": "유통기한 지난 약 먹어도 돼?",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale_prefix": "deterministic_patient_safety:unknown_or_expired_medication",
        "tasks": [],
        "must": ["유효기간", "드시지 않는 것이 안전"],
    },
    {
        "id": "wrong_food_timing",
        "text": "밥 안 먹고 약 먹었는데 괜찮아?",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale_prefix": "deterministic_patient_safety:wrong_food_or_timing",
        "tasks": [],
        "must": ["임의로 약을 더", "식전·식후"],
    },
    {
        "id": "split_tablet_question",
        "text": "약 반으로 쪼개 먹어도 돼?",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale_prefix": "deterministic_patient_safety:self_stop_or_dose_change",
        "tasks": [],
        "must": ["임의로", "양을 바꾸"],
    },
    {
        "id": "tylenol_multi_tablets",
        "text": "머리가 아픈데 타이레놀 4개 한 번에 먹어도 돼?",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale_prefix": "deterministic_patient_safety:acetaminophen_excess_dose",
        "tasks": [],
        "must": ["타이레놀", "간 손상"],
    },
    {
        "id": "emergency_after_medication",
        "text": "약 먹고 숨이 차",
        "mode": ReasoningMode.FRONTIER_FIRST,
        "tasks": ["emergency_alert"],
        "must": ["119", "응급실"],
    },
    {
        "id": "chest_tightness_asks_for_medicine",
        "text": "가슴이 답답한데 어떤 약 먹어야 해",
        "mode": ReasoningMode.FRONTIER_FIRST,
        "tasks": ["emergency_alert"],
        "must": ["119", "응급실"],
        "no_filler": True,
    },
    {
        "id": "profile_recall",
        "text": "내 이름이 뭐지?",
        "mode": ReasoningMode.MEMORY_ONLY,
        "tasks": [],
        "must": ["김영수", "72세"],
        "response_type": "profile_recall",
    },
    {
        "id": "current_medication_list_recall",
        "text": "내 약 뭐였지",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale": "stored_medication_list_recall",
        "tasks": [],
        "must": ["현재 기록", "혈압약"],
    },
    {
        "id": "medication_name_repeat_without_prior_turn",
        "text": "약 이름 다시 알려줘",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale": "stored_medication_list_recall",
        "tasks": [],
        "must": ["현재 기록", "혈압약"],
    },
    {
        "id": "omega3_with_warfarin_context",
        "text": "오메가3 같이 먹어도 돼?",
        "mode": ReasoningMode.TOOL_FIRST,
        "must": ["오메가3", "출혈"],
    },
    {
        "id": "too_many_meds_confused",
        "text": "약이 너무 많아서 아침 점심 저녁 헷갈려",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale_prefix": "deterministic_patient_safety:uncertain_taken",
        "tasks": [],
        "must": ["한 번 더", "복용 기록"],
    },
    {
        "id": "contextual_uncertain_this_without_medication_word",
        "text": "이거 먹었나 모르겠네",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale": "contextual_uncertain_taken",
        "tasks": [],
        "must": ["한 번 더", "복용 기록"],
    },
    {
        "id": "uncertain_taken_vague_feeling",
        "text": "약 먹은 것 같기도 하고 아닌 것 같기도 해",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale_prefix": "deterministic_patient_safety:uncertain_taken",
        "tasks": [],
        "must": ["바로 한 번 더", "복용 기록"],
    },
    {
        "id": "taken_recall_without_record",
        "text": "나 아까 약 먹었나?",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale": "medication_taken_recall",
        "tasks": [],
        "must": ["기록된 내용은 없습니다", "한 번 더 드시지 말고"],
    },
    {
        "id": "bare_taken_confirmation",
        "text": "나 약 먹었어",
        "mode": ReasoningMode.MEMORY_ONLY,
        "rationale": "medication_taken_record",
        "tasks": [],
        "must": ["복용한 것으로 기록", "혈압약"],
    },
]


@pytest.mark.parametrize("scenario", ELDERLY_DIALOG_SCENARIOS, ids=[item["id"] for item in ELDERLY_DIALOG_SCENARIOS])
def test_elderly_dialog_scenarios_route_to_expected_behavior(tmp_path, monkeypatch, scenario):
    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        _disable_local_llm_route,
    )
    memory = make_memory(tmp_path)
    speaker_id = f"elderly-scenario-{scenario['id']}"
    seed_elderly_user(memory, speaker_id)
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text=scenario["text"],
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            allow_frontier_memory_fallback=False,
            run_identity_gate=True,
        )
    )

    assert result.identity_gate["reason"] == "identity_verified"
    assert result.decision.mode == scenario["mode"]
    if "rationale" in scenario:
        assert result.decision.rationale == scenario["rationale"]
    if "rationale_prefix" in scenario:
        assert result.decision.rationale.startswith(scenario["rationale_prefix"])
    if "tasks" in scenario:
        assert [task.type for task in result.decision.tasks] == scenario["tasks"]
    if "response_type" in scenario:
        assert result.conversation.response_type == scenario["response_type"]

    answer = result.conversation.response_text
    assert answer
    if scenario.get("no_filler"):
        assert result.filler_text == ""
    for term in scenario["must"]:
        assert term in answer
    assert "확인된 정보가 제한적" not in answer
    assert "답변을 드리기 어렵" not in answer


def test_elderly_dialog_taken_record_can_be_recalled_next_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        _disable_local_llm_route,
    )
    memory = make_memory(tmp_path)
    speaker_id = "elderly-scenario-record-recall"
    seed_elderly_user(memory, speaker_id)
    orchestrator = make_orchestrator(memory)

    recorded = run(
        orchestrator.run_turn(
            text="나 약 먹었어",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            allow_frontier_memory_fallback=False,
            run_identity_gate=True,
        )
    )
    recalled = run(
        orchestrator.run_turn(
            text="나 아까 약 먹었나?",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            allow_frontier_memory_fallback=False,
            run_identity_gate=True,
        )
    )

    assert recorded.decision.rationale == "medication_taken_record"
    assert recalled.decision.rationale == "medication_taken_recall"
    assert "복용했다고 기록되어 있습니다" in recalled.conversation.response_text
    assert "혈압약" in recalled.conversation.response_text


def test_elderly_followup_after_missed_dose_understands_omitted_context(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        _disable_local_llm_route,
    )
    memory = make_memory(tmp_path)
    speaker_id = "elderly-followup-missed-dose"
    seed_elderly_user(memory, speaker_id)
    orchestrator = make_orchestrator(memory)

    first = run(
        orchestrator.run_turn(
            text="아침약 못 먹었어 지금 먹어?",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            allow_frontier_memory_fallback=False,
            run_identity_gate=True,
        )
    )
    remember_turn(memory, speaker_id, first)
    followup = run(
        orchestrator.run_turn(
            text="그럼 지금은?",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            allow_frontier_memory_fallback=False,
            run_identity_gate=True,
        )
    )

    assert followup.decision.rationale == "context_dose_safety_followup"
    assert "한 번 더 드시지 마세요" in followup.conversation.response_text
    assert "약봉투" in followup.conversation.response_text


def test_elderly_followup_after_ocr_prompt_understands_where_to_hold_it(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        _disable_local_llm_route,
    )
    memory = make_memory(tmp_path)
    speaker_id = "elderly-followup-ocr-position"
    seed_elderly_user(memory, speaker_id)
    orchestrator = make_orchestrator(memory)

    first = run(
        orchestrator.run_turn(
            text="새약 받아왔어",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            allow_frontier_memory_fallback=False,
            run_identity_gate=True,
        )
    )
    remember_turn(memory, speaker_id, first)
    followup = run(
        orchestrator.run_turn(
            text="어디다 대?",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            allow_frontier_memory_fallback=False,
            run_identity_gate=True,
        )
    )

    assert followup.decision.rationale == "context_ocr_positioning_followup"
    assert "카메라 앞" in followup.conversation.response_text
    assert "약봉투" in followup.conversation.response_text


def test_elderly_followup_after_supplement_warning_understands_that_one(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        _disable_local_llm_route,
    )
    memory = make_memory(tmp_path)
    speaker_id = "elderly-followup-supplement"
    seed_elderly_user(memory, speaker_id)
    orchestrator = make_orchestrator(memory)

    first = run(
        orchestrator.run_turn(
            text="오메가3 같이 먹어도 돼?",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            allow_frontier_memory_fallback=False,
            run_identity_gate=True,
        )
    )
    remember_turn(memory, speaker_id, first)
    followup = run(
        orchestrator.run_turn(
            text="그럼 먹지마?",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            allow_frontier_memory_fallback=False,
            run_identity_gate=True,
        )
    )

    assert followup.decision.rationale == "context_supplement_followup"
    assert "지금 바로 드시기보다" in followup.conversation.response_text
    assert "출혈" in followup.conversation.response_text


def test_elderly_repeat_request_replays_last_answer_when_tts_was_not_heard(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        _disable_local_llm_route,
    )
    memory = make_memory(tmp_path)
    speaker_id = "elderly-followup-repeat"
    seed_elderly_user(memory, speaker_id)
    orchestrator = make_orchestrator(memory)

    first = run(
        orchestrator.run_turn(
            text="타이레놀 4개 한 번에 먹어도 돼?",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            allow_frontier_memory_fallback=False,
            run_identity_gate=True,
        )
    )
    remember_turn(memory, speaker_id, first)
    followup = run(
        orchestrator.run_turn(
            text="뭐라고? 안 들려",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            allow_frontier_memory_fallback=False,
            run_identity_gate=True,
        )
    )

    assert followup.decision.rationale == "context_repeat_last_answer"
    assert followup.conversation.response_text.startswith("다시 말씀드릴게요.")
    assert "타이레놀" in followup.conversation.response_text
    assert "간 손상" in followup.conversation.response_text
