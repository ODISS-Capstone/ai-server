"""Regression coverage for ODISS demo story conversation flows."""
from __future__ import annotations

import asyncio
from datetime import datetime

from app.database.md_store import MDStore
from app.engines.conversation import ConversationEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import ReasoningEngine
from app.memory import StructuredMemoryService
from app.schemas.engine_contracts import ConversationComposeRequest, ReasoningMode, ReasoningRouteDecision, ReasoningRouteInput
from app.services import identity_guard
from app.services.engine_orchestrator import EngineOrchestrator
from app.services.reminders import ReminderService


def run(coro):
    return asyncio.run(coro)


def make_memory(tmp_path) -> MemoryEngine:
    engine = MemoryEngine()
    engine.store = MDStore(str(tmp_path / "md_database"))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))
    run(engine.initialize())
    return engine


def test_identity_registration_completes_without_extra_confirmation(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)

    async def fake_extract(current_text: str, **kwargs):
        return {"profile": {"name": "김영수"} if "김영수" in current_text else {}, "source": "test"}

    monkeypatch.setattr(identity_guard, "extract_identity_profile_with_llm", fake_extract)

    first = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="오디스.",
            speaker_id="demo-user",
        )
    )
    assert first.reason == "needs_registration"
    assert "이름" in first.response_text

    second = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="김영수 남자고, 72살이야.",
            speaker_id="demo-user",
        )
    )
    assert second.reason == "identity_registered"
    assert "김영수님" in second.response_text
    assert "남성" in second.response_text
    assert "72세" in second.response_text
    assert "등록해도 될까요" not in second.response_text

    state = run(memory.load_identity_state("demo-user"))
    assert state["profile"]["name"] == "김영수"
    assert state["profile"]["gender"] == "남성"
    assert state["profile"]["age"] == "72"


def test_identity_registration_accepts_arbitrary_young_profile(tmp_path):
    memory = make_memory(tmp_path)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="처음 왔어요. 저는 홍길동이고 23살 남자예요.",
            speaker_id="young-user",
        )
    )

    assert result.reason == "identity_registered"
    assert "홍길동님" in result.response_text
    assert "23세" in result.response_text
    assert "어르신" not in result.response_text

    state = run(memory.load_identity_state("young-user"))
    assert state["profile"]["name"] == "홍길동"
    assert state["profile"]["age"] == "23"
    assert state["profile"]["gender"] == "남성"


def test_identity_registration_accepts_caregiver_target_profile(tmp_path):
    memory = make_memory(tmp_path)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="저는 딸이고 아버지는 박철수 68세 남자예요. 고혈압이 있어요.",
            speaker_id="caregiver-user",
        )
    )

    assert result.reason == "identity_registered"
    assert "박철수님" in result.response_text
    assert "68세" in result.response_text
    assert "딸님" not in result.response_text

    state = run(memory.load_identity_state("caregiver-user"))
    assert state["profile"]["name"] == "박철수"
    assert state["profile"]["age"] == "68"
    assert state["profile"]["gender"] == "남성"
    assert "고혈압" in state["profile"]["conditions"]


def test_conversation_memory_ack_does_not_force_medical_disclaimer():
    engine = ConversationEngine()
    decision = ReasoningRouteDecision(
        mode=ReasoningMode.MEMORY_ONLY,
        intent="medication_query",
        rationale="record",
        tasks=[],
    )
    result = engine.compose_from_contract(
        ConversationComposeRequest(
            input_text="먹었어",
            user_profile={"name": "김영수"},
            decision=decision,
            core_message="점심 식후 약을 복용한 것으로 기록해두겠습니다.",
            reviewed_message="",
            delivery_message="",
        )
    )
    assert result.response_text.startswith("김영수님")
    assert "의사·약사 상담" not in result.response_text


def test_conversation_uses_neutral_default_honorific_for_non_elder_users():
    engine = ConversationEngine()
    decision = ReasoningRouteDecision(
        mode=ReasoningMode.MEMORY_ONLY,
        intent="smalltalk",
        rationale="smalltalk",
        tasks=[],
    )
    result = engine.compose_from_contract(
        ConversationComposeRequest(
            input_text="고마워",
            user_profile={},
            decision=decision,
            core_message="어르신, 언제든 편하게 물어보세요.",
            reviewed_message="",
            delivery_message="",
        )
    )
    assert result.response_text.startswith("사용자님")
    assert "어르신" not in result.response_text


def test_reasoning_routes_demo_ocr_capture_request():
    engine = ReasoningEngine(MemoryEngine(), LLMJudgeEngine())
    decision = engine.route_execution(
        ReasoningRouteInput(text="오디스. 내가 먹는 약 사진 좀 찍을게.", context={})
    )
    assert decision.mode == ReasoningMode.ASK_USER_CLARIFY
    assert decision.intent == "medication_query"
    assert "5, 4, 3, 2, 1" in engine.request_ocr()["message"]


def test_reminder_service_override_dispatch_and_taken_record(tmp_path):
    memory = make_memory(tmp_path)
    current = datetime(2026, 5, 13, 11, 59)

    def now_provider():
        return current

    sent: list[dict] = []
    service = ReminderService(now_provider=now_provider, start_background_tasks=False)
    service.register_connection("demo-user", lambda payload: sent.append(payload))

    proposal = service.start_setup(
        speaker_id="demo-user",
        user_profile={"name": "김영수"},
        prescription_log="# 현재 복용 약 요약\n- 혈압약\n",
    )
    assert "오전 8시" in proposal
    assert "오후 1시" in proposal

    confirm = run(
        service.finalize_pending(
            memory_engine=memory,
            speaker_id="demo-user",
            text="점심은 내가 일찍 먹으니까 알림을 12시로 설정해줘.",
            user_profile={"name": "김영수"},
            start_tasks=False,
        )
    )
    assert "점심 약 알림은 오후 12시" in confirm

    current = datetime(2026, 5, 13, 12, 0)
    dispatched = run(service.dispatch_due_reminders())
    assert dispatched
    assert sent[-1]["type"] == "reminder"
    assert "김영수님" in sent[-1]["text"]
    assert "먹었어" in sent[-1]["text"]

    recorded = run(
        service.record_taken(
            memory_engine=memory,
            speaker_id="demo-user",
            text="먹었어",
            user_profile={"name": "김영수"},
        )
    )
    assert "점심" in recorded
    assert "혈압약" in recorded

    recalled = run(
        service.recall_last_taken(
            memory_engine=memory,
            speaker_id="demo-user",
            user_profile={"name": "김영수"},
        )
    )
    assert "복용했다고 말씀하셨습니다" in recalled


def make_orchestrator(memory: MemoryEngine) -> EngineOrchestrator:
    judge = LLMJudgeEngine()
    return EngineOrchestrator(
        memory_engine=memory,
        reasoning_engine=ReasoningEngine(memory, judge),
        conversation_engine=ConversationEngine(),
        llm_judge=judge,
    )


def test_orchestrator_identity_gate_blocks_before_reasoning(tmp_path):
    memory = make_memory(tmp_path)
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="오디스.",
            speaker_id="new-demo-user",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert result.identity_gate["allowed"] is False
    assert result.identity_gate["reason"] == "needs_registration"
    assert result.decision.intent == "identity_check"
    assert "이름" in result.conversation.response_text
    assert not any(event.stage == "RE_Intent" for event in result.engine_trace)


def test_stt_ocr_result_is_normalized_into_prescription_memory(tmp_path):
    memory = make_memory(tmp_path)

    meds = run(
        memory.store_ocr_text_result(
            "처방전 OCR 결과가 와파린정, 아스피린장용정, 오메프라졸캡슐로 나왔어.",
            speaker_id="ocr-demo-user",
        )
    )

    assert meds == ["와파린정", "아스피린장용정", "오메프라졸캡슐"]
    prescription_log = run(memory.store.read_flash("prescription_log"))
    assert "와파린정" in prescription_log
    assert "아스피린장용정" in prescription_log
    ocr_entries = run(memory.store.list_entries("ocr_history"))
    prescription_entries = run(memory.store.list_entries("prescriptions"))
    assert ocr_entries
    assert prescription_entries
    structured = run(
        memory.structured_memory.build_context(
            "와파린 아스피린",
            speaker_id="ocr-demo-user",
        )
    )
    assert "최신 복약 및 DUR 요약" in structured["memory_prompt"]


def test_date_medication_event_is_typed_and_recalled_next_turn(tmp_path):
    memory = make_memory(tmp_path)
    orchestrator = make_orchestrator(memory)
    speaker_id = "date-demo-user"

    run(
        memory.update_and_compress(
            {
                "query": "2026년 5월 12일 화요일 밤 9시에 로사르탄정을 복용했다고 기록해줘.",
                "answer": "기록했습니다.",
                "type": "medication_query",
            },
            speaker_id=speaker_id,
        )
    )

    events = run(memory.store.read_user_file(speaker_id, "medication_events.md"))
    assert '"date": "2026-05-12"' in events
    assert '"time": "21:00"' in events
    assert '"medication": "로사르탄정"' in events

    result = run(
        orchestrator.run_turn(
            text="어제 밤에 먹었다고 기록한 약이 뭐였지? 시간도 같이 말해줘.",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
        )
    )

    assert result.decision.mode == ReasoningMode.MEMORY_ONLY
    assert "로사르탄정" in result.conversation.response_text
    assert "밤 9시" in result.conversation.response_text


def test_missing_medication_event_recall_does_not_fall_back_to_demo_drug(tmp_path):
    memory = make_memory(tmp_path)
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="어제 밤에 먹었다고 기록한 약이 뭐였지? 시간도 같이 말해줘.",
            speaker_id="empty-event-user",
            include_judge=False,
            include_delivery_llm=False,
        )
    )

    assert "로사르탄" not in result.conversation.response_text
    assert "찾지 못했습니다" in result.conversation.response_text


def test_schedule_and_dur_answers_do_not_inject_demo_medications(tmp_path):
    memory = make_memory(tmp_path)
    orchestrator = make_orchestrator(memory)
    context = run(memory.load_context("neutral-prescription-user"))
    context["prescription_log"] = "# 현재 복용 약 요약\n\n## 약품 목록\n- DrugA정\n- DrugB캡슐\n"
    context["context_memory"] = context["prescription_log"]

    schedule = run(
        orchestrator.run_turn(
            text="아침 점심 저녁 약을 어떻게 먹어야 해?",
            speaker_id="neutral-prescription-user",
            include_judge=False,
            include_delivery_llm=False,
            preloaded_context=context,
        )
    )
    dur = run(
        orchestrator.run_turn(
            text="dur 기준으로 확인해줘.",
            speaker_id="neutral-prescription-user",
            include_judge=False,
            include_delivery_llm=False,
            preloaded_context=context,
        )
    )

    combined = schedule.conversation.response_text + "\n" + dur.conversation.response_text
    assert "DrugA정" in combined
    assert "DrugB캡슐" in combined
    for demo_term in ("와파린", "아스피린", "오메프라졸", "로사르탄"):
        assert demo_term not in combined


def test_common_medication_mistakes_use_deterministic_safety_responses(tmp_path):
    memory = make_memory(tmp_path)
    orchestrator = make_orchestrator(memory)
    speaker_id = "safety-demo-user"

    cases = [
        (
            "아침 혈압약을 깜빡했어. 지금 두 번 먹어도 돼?",
            ReasoningMode.MEMORY_ONLY,
            ["두 번 드시면 안 됩니다", "약사"],
        ),
        (
            "내가 약 먹었는지 기억 안 나. 한 번 더 먹을까?",
            ReasoningMode.MEMORY_ONLY,
            ["바로 한 번 더", "복용 기록"],
        ),
        (
            "아내 약을 실수로 먹었어.",
            ReasoningMode.MEMORY_ONLY,
            ["다른 사람의 약", "119"],
        ),
        (
            "혈압약을 공복에 먹었어.",
            ReasoningMode.MEMORY_ONLY,
            ["임의로 약을 더", "식전·식후"],
        ),
        (
            "이제 괜찮으니까 당뇨약 중단해도 돼?",
            ReasoningMode.MEMORY_ONLY,
            ["임의로 끊거나", "의사나 약사"],
        ),
        (
            "유통기한 지난 약을 먹어도 돼?",
            ReasoningMode.MEMORY_ONLY,
            ["유효기간", "드시지 않는"],
        ),
        (
            "아스피린 먹고 숨이 차고 얼굴이 부었어.",
            ReasoningMode.FRONTIER_FIRST,
            ["119", "응급실"],
        ),
    ]

    for text, expected_mode, expected_terms in cases:
        result = run(
            orchestrator.run_turn(
                text=text,
                speaker_id=speaker_id,
                include_judge=False,
                include_delivery_llm=False,
            )
        )
        assert result.decision.mode == expected_mode
        for term in expected_terms:
            assert term in result.conversation.response_text
        if expected_mode == ReasoningMode.MEMORY_ONLY:
            assert not result.tool_trace

        run(
            memory.update_and_compress(
                {
                    "query": text,
                    "answer": result.conversation.response_text,
                    "type": result.decision.intent,
                },
                speaker_id=speaker_id,
            )
        )

    incidents = run(memory.store.read_user_file(speaker_id, "safety_incidents.md"))
    assert "missed_dose" in incidents
    assert "wrong_person_medication" in incidents
    assert "emergency_symptom_after_medication" in incidents
