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
