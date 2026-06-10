"""Regression coverage for ODISS demo story conversation flows."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

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


def test_identity_registration_accepts_colloquial_nan_prefix(tmp_path):
    memory = make_memory(tmp_path)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="난 김영수고 72살 남자야",
            speaker_id="demo-nan-user",
        )
    )

    assert result.reason == "identity_registered"
    assert "김영수" in result.response_text
    assert "72" in result.response_text


def test_identity_extraction_includes_gout_condition(tmp_path):
    memory = make_memory(tmp_path)

    profile = memory.extract_identity_from_text("나는 통풍이 있어서 통풍약을 먹고 있어")

    assert profile["conditions"] == ["통풍"]


def test_identity_gate_merges_new_condition_into_existing_profile(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "condition-user",
            {"name": "김영수", "gender": "남성", "age": "72", "conditions": ["고혈압"]},
            mark_verified=True,
        )
    )

    async def fake_judge_identity_conflict(**kwargs):
        return {"conflict": False, "source": "test"}

    monkeypatch.setattr(identity_guard, "judge_identity_conflict", fake_judge_identity_conflict)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="나는 통풍이 있어서 통풍약을 먹고 있어",
            speaker_id="condition-user",
        )
    )

    state = run(memory.load_identity_state("condition-user"))
    assert result.allowed is True
    assert state["profile"]["conditions"] == ["고혈압", "통풍"]


def test_medication_turns_do_not_trigger_identity_conflict_judge(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "med-turn-user",
            {"name": "김영수", "gender": "남성", "age": "72", "conditions": ["고혈압"]},
            mark_verified=True,
        )
    )
    calls = []

    async def fake_judge_identity_conflict(**kwargs):
        calls.append(kwargs)
        return {"conflict": True, "source": "should_not_be_called"}

    monkeypatch.setattr(identity_guard, "judge_identity_conflict", fake_judge_identity_conflict)

    for text in (
        "오디스. 나 혈압약 먹었어.",
        "내가 아까 약 먹었나?",
        "오디스. 혈압약 두 번 먹으면 더 빨리 좋아져?",
        "그럼 그냥 원래대로 먹어야겠네?",
    ):
        result = run(
            identity_guard.evaluate_identity_gate(
                memory_engine=memory,
                text=text,
                speaker_id="med-turn-user",
            )
        )
        assert result.allowed is True
        assert result.reason == "identity_verified"

    assert calls == []


def test_medication_guidance_turn_does_not_overwrite_profile_name(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    speaker_id = "profile-contamination-user"
    run(
        memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(
        memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )
    orchestrator = make_orchestrator(memory)

    async def fake_classify_route(**kwargs):
        return {
            "usable": True,
            "route_label": "meal_medication_prep",
            "mode": "MEMORY_ONLY",
            "intent": "medication_query",
            "task_types": [],
            "rationale": "test",
            "source": "test_local_llm",
        }

    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        fake_classify_route,
    )

    result = run(
        orchestrator.run_turn(
            text="그래 밥 먹고 나면 내가 나중에 뭐 먹어야 되는지 알려 줘",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )
    run(
        memory.update_and_compress(
            {
                "query": "그래 밥 먹고 나면 내가 나중에 뭐 먹어야 되는지 알려 줘",
                "answer": result.conversation.response_text,
                "type": result.decision.intent,
            },
            speaker_id=speaker_id,
        )
    )

    state = run(memory.load_identity_state(speaker_id))
    assert state["profile"]["name"] == "김영수"
    assert "먹어님" not in result.conversation.response_text
    assert "김영수님" in result.conversation.response_text


def test_medical_followup_is_answered_without_silent_suppression(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "followup-user",
            {"name": "김영수", "gender": "남성", "age": "72", "conditions": ["고혈압"]},
            mark_verified=True,
        )
    )
    run(
        memory.store.write_flash(
            "context_memory",
            "# 대화 컨텍스트 메모리\n\n- 질문: 녹용 먹어도 될까?\n- 핵심 응답: 혈압약 복용 중이면 먼저 의사나 약사에게 확인 권장.\n",
        )
    )
    orchestrator = make_orchestrator(memory)
    calls = []

    async def fake_classify_route(**kwargs):
        return {"usable": False, "source": "test_route_unavailable"}

    async def fake_recover(**kwargs):
        calls.append(kwargs)
        return {
            "is_medical_followup": True,
            "intent": "confirm_avoid_now",
            "response": "네, 지금 바로 드시기보다는 의사나 약사에게 먼저 확인하시는 것이 안전합니다.",
            "source": "test_local_llm",
        }

    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        fake_classify_route,
    )
    monkeypatch.setattr(
        "app.services.engine_orchestrator.recover_medical_followup_with_llm",
        fake_recover,
    )

    result = run(
        orchestrator.run_turn(
            text="그럼 당장은 안 먹는 게 낫겠네?",
            speaker_id="followup-user",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert not calls
    assert result.decision.intent == "medication_query"
    assert "먼저 확인" in result.conversation.response_text
    assert result.conversation.requires_tts is True


def test_identity_registration_accepts_stt_filler_prefix(tmp_path):
    memory = make_memory(tmp_path)

    run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="오디스.",
            speaker_id="demo-filler-prefix-user",
        )
    )
    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="어 김영수 72살 남자야",
            speaker_id="demo-filler-prefix-user",
        )
    )

    assert result.reason == "identity_registered"
    assert "김영수" in result.response_text
    assert "72세" in result.response_text


def test_prior_conversation_ambiguous_does_not_repeat_prior_question(tmp_path):
    memory = make_memory(tmp_path)
    run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="오디스.",
            speaker_id="demo-ambig-user",
        )
    )
    second = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="음 뭐라고요? 잘 안 들렸어요.",
            speaker_id="demo-ambig-user",
        )
    )
    assert second.reason == "needs_registration"
    assert "일전에 대화" not in second.response_text
    assert "이전에 대화" not in second.response_text
    state = run(memory.load_identity_state("demo-ambig-user"))
    assert state.get("pending_identity_action") == "registration"


def test_prior_conversation_no_leads_to_registration_prompt(tmp_path):
    memory = make_memory(tmp_path)

    run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="오디스.",
            speaker_id="demo-no-user",
        )
    )
    second = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="아니요, 처음이에요.",
            speaker_id="demo-no-user",
        )
    )
    assert second.reason == "needs_registration"
    assert "이름" in second.response_text
    assert "처음 뵙는" not in second.response_text


def test_unknown_profile_recall_says_unknown_before_registration_prompt(tmp_path):
    memory = make_memory(tmp_path)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="나 누군지 알아",
            speaker_id="unknown-profile-recall-user",
        )
    )

    assert result.allowed is False
    assert result.reason == "needs_registration"
    assert "아직" in result.response_text
    assert "누구신지 모릅니다" in result.response_text
    assert "이름, 나이, 성별" in result.response_text


def test_initial_profile_recall_skips_prior_conversation_llm(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    speaker_id = "initial-profile-recall-user"
    run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="오디스",
            speaker_id=speaker_id,
        )
    )

    async def fail_if_identity_llm_called(**kwargs):
        raise AssertionError("profile recall should not call identity extraction LLM")

    async def fail_if_prior_judge_called(**kwargs):
        raise AssertionError("profile recall should not call prior conversation judge")

    monkeypatch.setattr(identity_guard, "extract_identity_profile_with_llm", fail_if_identity_llm_called)
    monkeypatch.setattr(identity_guard, "judge_prior_conversation_turn", fail_if_prior_judge_called)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="내가 누구야",
            speaker_id=speaker_id,
        )
    )

    assert result.allowed is False
    assert result.reason == "needs_registration"
    assert "아직" in result.response_text
    assert "누구신지 모릅니다" in result.response_text
    assert "일전에 대화" not in result.response_text


def test_new_speaker_uses_flash_profile_before_registration_prompt(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.update_flash_profile(
            "previous-speaker",
            {"name": "김영수", "age": "72", "gender": "남성", "conditions": ["고혈압"]},
        )
    )

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="오디스.",
            speaker_id="new-speaker-same-room",
        )
    )

    assert result.allowed is False
    assert result.reason == "confirm_flash_identity"
    assert "김영수님" in result.response_text
    assert "맞으신가요" in result.response_text
    state = run(memory.load_identity_state("new-speaker-same-room"))
    assert state.get("pending_identity_action") == "confirm_new_identity"
    assert state.get("pending_identity_candidate", {}).get("name") == "김영수"

    confirmed = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="맞아.",
            speaker_id="new-speaker-same-room",
        )
    )
    assert confirmed.reason == "identity_candidate_registered"
    flash = run(memory.store.read_flash("current_user_profile"))
    manual = run(memory.store.read_flash("current_manual"))
    assert "김영수" in flash
    assert "김영수" in manual


def test_name_only_mismatch_starts_new_registration_without_loop(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "demo-kim",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(memory.mark_identity_pending("demo-kim", "identity_conflict"))

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="난 이재석이야.",
            speaker_id="demo-kim",
        )
    )
    assert result.allowed is False
    assert result.reason == "identity_rejected_needs_registration"
    assert "김영수님으로 보지 않겠습니다" in result.response_text

    recall = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="이재석 45세 남자야.",
            speaker_id="demo-kim",
        )
    )
    assert recall.allowed is False
    assert recall.reason == "identity_registered"


def test_explicit_reregistration_request_prompts_for_new_identity(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "reregister-user",
            {"name": "홍길동", "gender": "남성", "age": "23"},
            mark_verified=True,
        )
    )
    calls = []

    async def fake_judge_identity_conflict(**kwargs):
        calls.append(kwargs)
        return {"conflict": False, "source": "should_not_be_called"}

    monkeypatch.setattr(identity_guard, "judge_identity_conflict", fake_judge_identity_conflict)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="아냐 새로 정보등록할게",
            speaker_id="reregister-user",
        )
    )

    state = run(memory.load_identity_state("reregister-user"))
    assert result.allowed is False
    assert result.reason == "identity_rejected_needs_registration"
    assert "새로 등록할 이름, 나이, 성별" in result.response_text
    assert state.get("pending_identity_action") == "registration"
    assert calls == []

    recall = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="내 이름이 누군지 알아?",
            speaker_id="reregister-user",
        )
    )

    assert recall.allowed is False
    assert recall.reason == "needs_registration"
    assert "아직" in recall.response_text
    assert "누구신지 모릅니다" in recall.response_text
    assert "홍길동" not in recall.response_text


def test_different_person_reregistration_request_does_not_return_blank(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "different-person-user",
            {"name": "홍길동", "gender": "남성", "age": "23"},
            mark_verified=True,
        )
    )

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="사실은 다른사람인데 새로 등록 가능할까",
            speaker_id="different-person-user",
        )
    )

    state = run(memory.load_identity_state("different-person-user"))
    assert result.allowed is False
    assert result.response_text
    assert result.reason == "identity_rejected_needs_registration"
    assert state.get("pending_identity_action") == "registration"


def test_current_profile_name_negation_starts_registration_without_llm(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "profile-negation-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    calls = []

    async def fake_judge_identity_conflict(**kwargs):
        calls.append(kwargs)
        return {"conflict": False, "source": "should_not_be_called"}

    monkeypatch.setattr(identity_guard, "judge_identity_conflict", fake_judge_identity_conflict)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="아니야 나 김영수 아니야",
            speaker_id="profile-negation-user",
        )
    )

    state = run(memory.load_identity_state("profile-negation-user"))
    assert result.allowed is False
    assert result.reason == "identity_rejected_needs_registration"
    assert "김영수님으로 보지 않겠습니다" in result.response_text
    assert "새로 등록할 이름, 나이, 성별" in result.response_text
    assert state.get("pending_identity_action") == "registration"
    assert calls == []


def test_profile_consistency_conflict_uses_llm_judge_not_keyword_heuristic(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "demo-consistency-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    calls = []

    async def fake_judge_identity_conflict(**kwargs):
        calls.append(kwargs)
        return {"conflict": False, "source": "test_llm_judge"}

    monkeypatch.setattr(identity_guard, "judge_identity_conflict", fake_judge_identity_conflict)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="나는 임신 중인데 이 약 먹어도 돼?",
            speaker_id="demo-consistency-user",
        )
    )

    assert calls
    assert result.allowed is True
    assert result.reason == "identity_verified"
    assert result.metadata["judge"]["source"] == "test_llm_judge"


def test_stale_identity_pending_expires_instead_of_repeating_reverify(tmp_path):
    memory = make_memory(tmp_path)
    now = datetime(2026, 5, 18, 6, 20, 0)
    run(
        memory.save_identity_profile(
            "demo-kim",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
            now=now - timedelta(minutes=20),
        )
    )
    run(
        memory.save_identity_profile(
            "demo-kim",
            {},
            pending_identity_action="reverification",
            mark_seen=False,
            now=now - timedelta(minutes=6),
        )
    )

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="고혈압약이랑 녹용 같이 먹어도 돼?",
            speaker_id="demo-kim",
            now=now,
        )
    )

    assert result.allowed is True
    assert result.reason == "identity_verified"
    assert "본인이 맞으신가요" not in result.response_text
    state = run(memory.load_identity_state("demo-kim"))
    assert state.get("pending_identity_action") == ""


def test_expired_reverification_accepts_affirmative_reply_instead_of_blank(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    now = datetime(2026, 5, 18, 6, 20, 0)
    run(
        memory.save_identity_profile(
            "expired-reverify-yes-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
            now=now - timedelta(minutes=20),
        )
    )
    run(
        memory.save_identity_profile(
            "expired-reverify-yes-user",
            {},
            pending_identity_action="reverification",
            mark_seen=False,
            now=now - timedelta(minutes=6),
        )
    )
    calls = []

    async def fake_pending_identity_judge(**kwargs):
        calls.append(kwargs)
        raise AssertionError("expired affirmative should use identity fast path")

    monkeypatch.setattr(identity_guard, "judge_pending_identity_reply_with_llm", fake_pending_identity_judge)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="어 맞아",
            speaker_id="expired-reverify-yes-user",
            now=now,
        )
    )

    state = run(memory.load_identity_state("expired-reverify-yes-user"))
    assert result.allowed is False
    assert result.reason == "identity_reverified"
    assert result.response_text
    assert "김영수님으로 확인했습니다" in result.response_text
    assert state.get("pending_identity_action") == ""
    assert calls == []


def test_orchestrator_expired_reverification_affirmative_returns_tts(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    speaker_id = "expired-reverify-orchestrator-user"
    now = datetime.now()
    run(
        memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
            now=now - timedelta(minutes=20),
        )
    )
    run(
        memory.save_identity_profile(
            speaker_id,
            {},
            pending_identity_action="reverification",
            mark_seen=False,
            now=now - timedelta(minutes=6),
        )
    )

    async def fake_pending_identity_judge(**kwargs):
        raise AssertionError("expired affirmative should not enter pending identity LLM")

    monkeypatch.setattr(identity_guard, "judge_pending_identity_reply_with_llm", fake_pending_identity_judge)

    orchestrator = make_orchestrator(memory)
    result = run(
        orchestrator.run_turn(
            text="어 맞아",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert result.identity_gate["reason"] == "identity_reverified"
    assert result.conversation.response_type == "identity_check"
    assert result.conversation.requires_tts is True
    assert result.conversation.response_text
    assert "김영수님으로 확인했습니다" in result.conversation.response_text
    assert result.execution_results.get("suppressed") is not True


def test_negative_reverification_starts_registration_instead_of_repeating_prompt(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "demo-kim",
            {"name": "김양수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(memory.mark_identity_pending("demo-kim", "reverification"))

    async def fake_pending_identity_judge(**kwargs):
        return {"decision": "rejected", "profile": {}, "source": "test_local_llm"}

    monkeypatch.setattr(identity_guard, "judge_pending_identity_reply_with_llm", fake_pending_identity_judge)

    rejected = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="아니",
            speaker_id="demo-kim",
        )
    )

    assert rejected.allowed is False
    assert rejected.reason == "identity_rejected_needs_registration"
    assert "김양수님으로 보지 않겠습니다" in rejected.response_text
    assert "본인이 맞으신가요" not in rejected.response_text
    state = run(memory.load_identity_state("demo-kim"))
    assert state.get("pending_identity_action") == "registration"

    registered = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="김영수 72살 남자야",
            speaker_id="demo-kim",
        )
    )

    assert registered.reason == "identity_registered"
    assert "김영수님" in registered.response_text
    final_state = run(memory.load_identity_state("demo-kim"))
    assert final_state["profile"]["name"] == "김영수"
    assert final_state.get("pending_identity_action") == ""


def test_reverification_accepts_spoken_affirmative_without_repeating_prompt(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "reverify-yes-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(memory.mark_identity_pending("reverify-yes-user", "reverification"))
    calls = []

    async def fake_pending_identity_judge(**kwargs):
        calls.append(kwargs)
        raise AssertionError("spoken affirmative should use identity fast path")

    monkeypatch.setattr(identity_guard, "judge_pending_identity_reply_with_llm", fake_pending_identity_judge)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="어 맞아",
            speaker_id="reverify-yes-user",
        )
    )

    state = run(memory.load_identity_state("reverify-yes-user"))
    assert result.allowed is False
    assert result.reason == "identity_reverified"
    assert "김영수님으로 확인했습니다" in result.response_text
    assert "본인이 맞으신가요" not in result.response_text
    assert state.get("pending_identity_action") == ""
    assert calls == []


def test_reverification_accepts_self_name_confirmation_without_repeating_prompt(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "reverify-name-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(memory.mark_identity_pending("reverify-name-user", "reverification"))
    calls = []

    async def fake_pending_identity_judge(**kwargs):
        calls.append(kwargs)
        raise AssertionError("matching self-name should use identity fast path")

    monkeypatch.setattr(identity_guard, "judge_pending_identity_reply_with_llm", fake_pending_identity_judge)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="내가 김영수야",
            speaker_id="reverify-name-user",
        )
    )

    state = run(memory.load_identity_state("reverify-name-user"))
    assert result.allowed is False
    assert result.reason == "identity_reverified"
    assert "김영수님으로 확인했습니다" in result.response_text
    assert "본인이 맞으신가요" not in result.response_text
    assert state.get("pending_identity_action") == ""
    assert calls == []


def test_reverification_noise_is_ignored_without_repeating_prompt(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "reverify-noise-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(memory.mark_identity_pending("reverify-noise-user", "reverification"))

    async def fake_pending_identity_judge(**kwargs):
        return {"decision": "noise", "profile": {}, "source": "test_local_llm"}

    monkeypatch.setattr(identity_guard, "judge_pending_identity_reply_with_llm", fake_pending_identity_judge)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="아자",
            speaker_id="reverify-noise-user",
        )
    )

    assert result.allowed is False
    assert result.reason == "identity_pending_noise"
    assert result.response_type == "ignored"
    assert result.response_text == ""


def test_reverification_mixed_affirmative_with_other_name_becomes_identity_candidate(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "reverify-other-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(memory.mark_identity_pending("reverify-other-user", "reverification"))

    async def fake_pending_identity_judge(**kwargs):
        return {
            "decision": "provided_identity",
            "profile": {"name": "김향수", "gender": "남성", "age": "72"},
            "source": "test_local_llm",
        }

    monkeypatch.setattr(identity_guard, "judge_pending_identity_reply_with_llm", fake_pending_identity_judge)

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="맞아 난 김향수야",
            speaker_id="reverify-other-user",
        )
    )

    assert result.allowed is False
    assert result.reason == "confirm_new_identity"
    assert "김향수" in result.response_text
    assert "김영수님 본인이 맞으신가요" not in result.response_text


def test_registration_does_not_treat_korean_age_ne_as_affirmative(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "demo-kim",
            {"name": "김양수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(memory.mark_identity_pending("demo-kim", "registration"))

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="이재석 스물네 살 남성",
            speaker_id="demo-kim",
        )
    )

    assert result.reason == "identity_registered"
    assert "김양수님으로 확인" not in result.response_text
    assert "이재석님" in result.response_text
    state = run(memory.load_identity_state("demo-kim"))
    assert state["profile"]["name"] == "이재석"
    assert state["profile"]["age"] == "24"
    assert state["profile"]["gender"] == "남성"


def test_prior_conversation_recognizes_existing_profile_by_name(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "returning-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            pending_identity_action="prior_conversation_check",
        )
    )
    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="김영수야.",
            speaker_id="returning-user",
        )
    )
    assert result.reason == "identity_recognized"
    assert "김영수" in result.response_text


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
    assert first.reason == "prior_conversation_check"
    assert "일전에 대화" in first.response_text
    assert "처음 뵙는" not in first.response_text

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


def test_identity_registration_keeps_name_when_gender_precedes_age(tmp_path):
    memory = make_memory(tmp_path)

    run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="오디스.",
            speaker_id="demo-gender-age-user",
        )
    )

    result = run(
        identity_guard.evaluate_identity_gate(
            memory_engine=memory,
            text="김영수 남성 72세 야",
            speaker_id="demo-gender-age-user",
        )
    )

    assert result.reason == "identity_registered"
    assert "김영수님" in result.response_text
    assert "남성님" not in result.response_text

    state = run(memory.load_identity_state("demo-gender-age-user"))
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


def test_medication_question_with_profile_words_does_not_become_registration_ack(tmp_path):
    memory = make_memory(tmp_path)
    orchestrator = EngineOrchestrator(
        memory_engine=memory,
        reasoning_engine=ReasoningEngine(memory, LLMJudgeEngine()),
        conversation_engine=ConversationEngine(),
        llm_judge=LLMJudgeEngine(),
    )
    decision = ReasoningRouteDecision(
        mode=ReasoningMode.TOOL_FIRST,
        intent="medication_query",
        rationale="deterministic_tools_available",
        tasks=[],
    )

    core = orchestrator._deterministic_core_message(
        text="내가 지금 그 고혈압약이랑 그리고 녹용을 같이 먹을 수 있나",
        decision=decision,
        context={"user_profile": {"name": "김영수", "age": "72", "gender": "남성", "conditions": ["고혈압"]}},
        execution_results={"task_results": {"supplements": ""}},
    )

    assert "프로필을 등록" not in core


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
    assert decision.mode == ReasoningMode.TOOL_FIRST
    assert decision.intent == "medication_query"
    assert [task.type for task in decision.tasks] == ["request_ocr"]
    results = run(
        engine.execute_tasks(
            "오디스. 내가 먹는 약 사진 좀 찍을게.",
            decision.intent,
            {},
            decision.tasks,
        )
    )
    assert results["task_results"]["ocr_requested"] is True
    assert "5, 4, 3, 2, 1" in engine.request_ocr()["message"]

    register_decision = engine.route_execution(
        ReasoningRouteInput(text="처방전 등록하고 싶어", context={})
    )
    assert register_decision.mode == ReasoningMode.TOOL_FIRST
    assert [task.type for task in register_decision.tasks] == ["request_ocr"]


def test_ocr_capture_request_uses_pre_capture_language(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "ocr-capture-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="나 약받아왔는데 사진 찍고싶거든 준비좀 해줄 수 있어?",
            speaker_id="ocr-capture-user",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert result.decision.mode == ReasoningMode.TOOL_FIRST
    assert [task.type for task in result.decision.tasks] == ["request_ocr"]
    assert result.execution_results["task_results"]["ocr_requested"] is True
    assert "촬영" in result.filler_text or "카메라" in result.filler_text
    assert "읽혔" not in result.filler_text
    assert "인식" not in result.filler_text
    assert "카메라 앞으로" in result.conversation.response_text
    assert "5, 4, 3, 2, 1" in result.conversation.response_text
    assert "약물 식별을 위해" not in result.conversation.response_text


def test_new_medication_received_routes_to_ocr_capture(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "new-medication-capture-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="오디스 나 새약 받아왔어",
            speaker_id="new-medication-capture-user",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert result.decision.mode == ReasoningMode.TOOL_FIRST
    assert result.decision.rationale == "ocr_capture_requested"
    assert [task.type for task in result.decision.tasks] == ["request_ocr"]
    assert result.execution_results["task_results"]["ocr_requested"] is True
    assert "촬영" in result.filler_text or "카메라" in result.filler_text or "약봉투" in result.filler_text
    assert "읽혔" not in result.filler_text
    assert "인식" not in result.filler_text
    assert "카메라 앞으로" in result.conversation.response_text
    assert "5, 4, 3, 2, 1" in result.conversation.response_text
    assert "확인된 정보가 제한적" not in result.conversation.response_text


def test_meal_medication_guidance_uses_deterministic_fast_path_without_ocr(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    speaker_id = "meal-route-user"
    run(
        memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(
        memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )
    orchestrator = make_orchestrator(memory)

    async def fake_classify_route(**kwargs):
        raise AssertionError(f"meal guidance should not call local LLM route: {kwargs!r}")

    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        fake_classify_route,
    )

    prep = run(
        orchestrator.run_turn(
            text="나중에 내가 밥을 먹고 나면 무슨 약을 먹어야 되는지 알려 줘",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )
    after_meal = run(
        orchestrator.run_turn(
            text="오디스. 밥 먹고 왔는데 약 뭐 먹어야 하지?",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert prep.decision.rationale == "stored_medication_meal_guidance"
    assert after_meal.decision.rationale == "stored_medication_meal_guidance"
    assert prep.decision.tasks == []
    assert after_meal.decision.tasks == []
    assert "저장된 약은 혈압약" in prep.conversation.response_text
    assert "먹었어" in prep.conversation.response_text
    assert "혈압약" in after_meal.conversation.response_text
    assert not prep.execution_results.get("task_results", {}).get("ocr_requested")
    assert not after_meal.execution_results.get("task_results", {}).get("ocr_requested")
    assert any(
        event.action == "skip_route_classification"
        and event.metadata.get("route_skipped_reason") == "stored_medication_meal_guidance"
        for event in prep.engine_trace
    )


def test_after_meal_question_with_wake_word_uses_stored_medication_first(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    speaker_id = "meal-memory-user"
    run(
        memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(
        memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )

    async def fake_classify_route(**kwargs):
        return {"usable": False, "source": "test_no_local_route"}

    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        fake_classify_route,
    )

    orchestrator = make_orchestrator(memory)
    result = run(
        orchestrator.run_turn(
            text="오디스 내가 밥을 먹고 난후 무슨 약을 먹어야 하는지 알려줘",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    answer = result.conversation.response_text
    assert result.decision.rationale == "stored_medication_meal_guidance"
    assert result.decision.tasks == []
    assert "김영수님" in answer
    assert "혈압약" in answer
    assert "저장된 약은" in answer
    assert "먹었어" in answer
    assert "어르신" not in answer
    assert "사용자님" not in answer
    assert "구체적인 정보" not in answer
    assert "아직 기록" not in answer


def test_short_meal_completion_guides_stored_after_meal_medication(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    speaker_id = "short-meal-user"
    run(
        memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(
        memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )

    async def fake_classify_route(**kwargs):
        return {"usable": False, "source": "test_no_local_route"}

    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        fake_classify_route,
    )

    orchestrator = make_orchestrator(memory)
    result = run(
        orchestrator.run_turn(
            text="나 밥먹었어",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert result.decision.mode == ReasoningMode.MEMORY_ONLY
    assert result.decision.rationale == "stored_medication_meal_guidance"
    assert result.decision.tasks == []
    assert not result.tool_trace
    assert "방금 드셨다는" not in result.filler_text
    assert "복용 기록" not in result.filler_text
    assert "혈압약" in result.conversation.response_text
    assert "드시면 됩니다" in result.conversation.response_text
    assert "먹었어" in result.conversation.response_text
    assert "확인된 정보가 제한적" not in result.conversation.response_text


def test_short_meal_completion_without_medication_context_asks_for_package(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    speaker_id = "short-meal-no-med-user"
    run(
        memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )

    async def fake_classify_route(**kwargs):
        return {"usable": False, "source": "test_no_local_route"}

    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        fake_classify_route,
    )

    orchestrator = make_orchestrator(memory)
    result = run(
        orchestrator.run_turn(
            text="점심 먹었어",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert result.decision.mode == ReasoningMode.MEMORY_ONLY
    assert result.decision.rationale == "meal_guidance_missing_medication_context"
    assert result.decision.tasks == []
    assert "약봉투" in result.conversation.response_text
    assert "처방전" in result.conversation.response_text
    assert "확인된 정보가 제한적" not in result.conversation.response_text


@pytest.mark.parametrize(
    "user_text",
    [
        "나 밥 먹고 나서 타이레놀 먹어야 되는데 알림 해 줄 수 있어",
        "어 나 밥 먹고 오면은 타이레놀 먹으라고 알려 줘",
    ],
)
def test_named_after_meal_medication_guidance_infers_breakfast_without_stored_med(tmp_path, monkeypatch, user_text):
    memory = make_memory(tmp_path)
    speaker_id = "named-meal-med-user"
    run(
        memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )

    async def fake_classify_route(**kwargs):
        raise AssertionError(f"named meal guidance should not call local LLM route: {kwargs!r}")

    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        fake_classify_route,
    )
    monkeypatch.setattr(
        EngineOrchestrator,
        "_meal_hint_from_current_time",
        staticmethod(lambda now=None: "아침"),
    )
    monkeypatch.setattr(
        EngineOrchestrator,
        "_current_time_phrase",
        staticmethod(lambda now=None: "오전 8시 9분"),
    )

    orchestrator = make_orchestrator(memory)
    result = run(
        orchestrator.run_turn(
            text=user_text,
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert result.decision.rationale == "named_medication_meal_guidance"
    assert result.decision.tasks == []
    answer = result.conversation.response_text
    assert "오전 8시 9분" in answer
    assert "아침 식사 후" in answer
    assert "타이레놀" in answer
    assert "밥 먹었어" in answer
    assert "먹었어" in answer
    assert "현재 저장된 식후 복용약 기록이 없습니다" not in answer


def test_medication_record_challenge_confirms_existing_record(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    speaker_id = "record-challenge-user"
    run(
        memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(
        memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )

    async def fake_classify_route(**kwargs):
        return {"usable": False, "source": "test_no_local_route"}

    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        fake_classify_route,
    )

    orchestrator = make_orchestrator(memory)
    result = run(
        orchestrator.run_turn(
            text="나 혈압약 먹고 있다고 했는데 기록 남아있지 않나?",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    answer = result.conversation.response_text
    assert result.decision.rationale == "stored_medication_record_recall"
    assert "김영수님" in answer
    assert "맞아요" in answer
    assert "혈압약" in answer
    assert "어르신" not in answer
    assert "사용자님" not in answer
    assert "아직 기록" not in answer


def test_llm_routed_noise_is_ignored_but_ack_uses_smalltalk_fast_path(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "llm-noise-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    orchestrator = make_orchestrator(memory)
    labels = iter(["non_actionable_ack", "noise_fragment"])

    async def fake_classify_route(**kwargs):
        return {
            "usable": True,
            "route_label": next(labels),
            "mode": "MEMORY_ONLY",
            "intent": "unknown",
            "task_types": [],
            "rationale": "test_ignore",
            "source": "test_local_llm",
        }

    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        fake_classify_route,
    )

    ack = run(
        orchestrator.run_turn(
            text="네, 알겠습니다",
            speaker_id="llm-noise-user",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )
    fragment = run(
        orchestrator.run_turn(
            text="나중에 밤",
            speaker_id="llm-noise-user",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert ack.conversation.response_type == "smalltalk"
    assert ack.conversation.requires_tts is True
    assert ack.decision.rationale == "smalltalk_detected"
    assert fragment.conversation.response_type == "smalltalk"
    assert fragment.conversation.response_text
    assert fragment.conversation.requires_tts is True
    assert fragment.execution_results.get("suppressed") is not True


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
    assert "오후 12시" in recalled


def test_reminder_story_setup_wait_taken_and_recall(tmp_path):
    memory = make_memory(tmp_path)
    current = datetime(2026, 5, 18, 11, 55)

    def now_provider():
        return current

    sent: list[dict] = []
    service = ReminderService(now_provider=now_provider, start_background_tasks=False)
    service.register_connection("demo-kim", lambda payload: sent.append(payload))

    setup = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="demo-kim",
            text="식후 복용 알림 설정해줘",
            user_profile={"name": "김영수"},
            prescription_log="# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )
    assert "아침은 오전 8시" in setup
    assert "점심은 오후 1시" in setup

    confirm = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="demo-kim",
            text="점심은 내가 일찍 먹으니까 알림을 12시로 설정해줘.",
            user_profile={"name": "김영수"},
            prescription_log="# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )
    assert "아침 약 알림은 오전 8시" in confirm
    assert "점심 약 알림은 오후 12시" in confirm
    assert "저녁 약 알림은 오후 7시" in confirm

    current = datetime(2026, 5, 18, 12, 0)
    dispatched = run(service.dispatch_due_reminders())
    assert dispatched
    assert "김영수님" in sent[-1]["text"]
    assert "오후 12시" in sent[-1]["text"]
    assert "점심 혈압약" in sent[-1]["text"]

    wait_ack = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="demo-kim",
            text="알았어. 기다려봐.",
            user_profile={"name": "김영수"},
        )
    )
    assert wait_ack is None

    current = datetime(2026, 5, 18, 12, 3)
    taken = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="demo-kim",
            text="먹었어.",
            user_profile={"name": "김영수"},
        )
    )
    assert "오늘 오후 12시 3분" in taken
    assert "점심 혈압약" in taken

    direct_taken = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="demo-kim",
            text="오디스. 나 혈압약 먹었어.",
            user_profile={"name": "김영수"},
        )
    )
    assert "혈압약을 복용한 것으로 기록" in direct_taken

    recall = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="demo-kim",
            text="내가 아까 약 먹었나?",
            user_profile={"name": "김영수"},
        )
    )
    assert "오늘 오후 12시 3분에 식후 혈압약을 복용했다고 말씀하셨습니다" in recall

    time_recall = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="demo-kim",
            text="몇 시에 먹었지",
            user_profile={"name": "김영수"},
        )
    )
    assert "오늘 오후 12시 3분에 식후 혈압약" in time_recall


def make_orchestrator(memory: MemoryEngine) -> EngineOrchestrator:
    judge = LLMJudgeEngine()
    return EngineOrchestrator(
        memory_engine=memory,
        reasoning_engine=ReasoningEngine(memory, judge),
        conversation_engine=ConversationEngine(),
        llm_judge=judge,
    )


def test_spoken_medication_registration_updates_prescription_memory(tmp_path):
    memory = make_memory(tmp_path)

    assert memory.extract_spoken_medications_from_text("나 디오반정 가지고 있거든 한번 확인해 줘") == ["디오반정"]
    assert memory.extract_spoken_medications_from_text("나 혈압 양 디오반") == ["디오반정"]
    assert memory.extract_spoken_medications_from_text("혈압약 디오반 먹어도 돼?") == []

    merged = run(
        memory.store_spoken_medication_result(
            "나 디오반정 가지고 있거든 한번 확인해 줘",
            ["디오반정"],
            speaker_id="speaker-hyun",
        )
    )
    prescription_log = run(memory.store.read_flash("prescription_log"))

    assert "디오반정" in merged
    assert "디오반정" in prescription_log


def test_orchestrator_medication_safety_question_skips_delivery_llm(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    speaker_id = "medication-safety-user"
    run(
        memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "age": "72", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(
        memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 디오반정\n",
        )
    )

    async def fail_if_route_llm_called(**kwargs):
        raise AssertionError("medication safety fast path should not call route LLM")

    async def fail_if_delivery_llm_called(**kwargs):
        raise AssertionError("medication safety fast path should not call delivery LLM")

    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        fail_if_route_llm_called,
    )
    monkeypatch.setattr(
        "app.services.engine_orchestrator.call_local_delivery_llm",
        fail_if_delivery_llm_called,
    )

    orchestrator = make_orchestrator(memory)
    result = run(
        orchestrator.run_turn(
            text="오디스 내가 디오반정을 내게 동시에 먹어도 될까",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=True,
            run_identity_gate=True,
        )
    )

    assert result.decision.mode == ReasoningMode.MEMORY_ONLY
    assert result.decision.rationale == "medication_safety_fast_path"
    assert result.decision.tasks == []
    assert result.filler_text == ""
    answer = result.conversation.response_text
    assert "김영수님" in answer
    assert "디오반정" in answer
    assert "한 번에" in answer
    assert "119" in answer
    assert "디곡신" not in answer
    assert "方才" not in answer
    assert "主治" not in answer


def test_orchestrator_simple_tylenol_suitability_is_not_overdose_warning(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    speaker_id = "medication-safety-user"
    run(
        memory.save_identity_profile(
            speaker_id,
            {"name": "김영수", "age": "72", "gender": "남성"},
            mark_verified=True,
        )
    )

    async def fail_if_route_llm_called(**kwargs):
        raise AssertionError("simple medication suitability should not call route LLM")

    async def fail_if_delivery_llm_called(**kwargs):
        raise AssertionError("simple medication suitability should not call delivery LLM")

    monkeypatch.setattr(
        "app.services.engine_orchestrator.classify_reasoning_route_with_llm",
        fail_if_route_llm_called,
    )
    monkeypatch.setattr(
        "app.services.engine_orchestrator.call_local_delivery_llm",
        fail_if_delivery_llm_called,
    )

    orchestrator = make_orchestrator(memory)
    result = run(
        orchestrator.run_turn(
            text="내가 지금 타이레놀 먹어도 될까",
            speaker_id=speaker_id,
            include_judge=False,
            include_delivery_llm=True,
            run_identity_gate=True,
        )
    )

    assert result.decision.mode == ReasoningMode.MEMORY_ONLY
    assert result.decision.rationale == "medication_safety_fast_path"
    assert result.filler_text == ""
    answer = result.conversation.response_text
    assert "타이레놀" in answer
    assert "용량" in answer
    assert "시간" in answer
    assert "여러 알" not in answer
    assert "저혈압" not in answer
    assert "119" not in answer
    assert len(answer) < 150


def test_orchestrator_spoken_medication_registration_and_vague_followup(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "speaker-hyun",
            {"name": "정현기", "age": "23", "gender": "남성"},
            mark_verified=True,
        )
    )
    orchestrator = make_orchestrator(memory)

    add_result = run(
        orchestrator.run_turn(
            text="나 디오반정 가지고 있거든 한번 확인해 줘",
            speaker_id="speaker-hyun",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert add_result.decision.rationale == "spoken_medication_registration"
    assert "디오반정" in add_result.conversation.response_text
    prescription_log = run(memory.store.read_flash("prescription_log"))
    assert "디오반정" in prescription_log

    followup_result = run(
        orchestrator.run_turn(
            text="오늘 그거 먹어야 돼",
            speaker_id="speaker-hyun",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert followup_result.decision.rationale == "stored_medication_vague_guidance"
    assert "디오반정" in followup_result.conversation.response_text
    assert "한 번 더 드시지 마세요" in followup_result.conversation.response_text


def test_orchestrator_profile_memory_ack_ignores_medication_context(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "speaker-kim",
            {"name": "김영수", "age": "72", "gender": "남성"},
            mark_verified=True,
        )
    )
    run(
        memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 타이레놀\n",
        )
    )
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="알았어 나 잘 기억해 줘",
            speaker_id="speaker-kim",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert result.decision.rationale == "profile_memory_ack"
    assert result.conversation.response_type == "profile_memory_ack"
    assert "앞으로 김영수님 정보로 잘 기억하겠습니다" in result.conversation.response_text
    assert "남성, 72세" not in result.conversation.response_text
    assert "타이레놀" not in result.conversation.response_text
    assert "의사·약사" not in result.conversation.response_text


def test_orchestrator_profile_recall_after_name_mismatch(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "recall-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="내가 누구인지 말해봐바.",
            speaker_id="recall-user",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert result.identity_gate["allowed"] is True
    assert "김영수" in result.conversation.response_text
    assert "복약 질문" not in result.conversation.response_text


def test_orchestrator_profile_recall_accepts_spaced_stt_variant(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "recall-spaced-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="내가 누군 지 알아?",
            speaker_id="recall-spaced-user",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert result.identity_gate["allowed"] is True
    assert "김영수" in result.conversation.response_text


def test_profile_recall_does_not_expose_history_reference(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "recall-history-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(
        memory.store.save_user_file(
            "recall-history-user",
            "history.md",
            "\n---\n### 이전 상담\n- Q: 고혈압약 녹용 문의\n- A: 주의가 필요하다고 안내\n",
        )
    )
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="내가 누군 지 알아?",
            speaker_id="recall-history-user",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert "김영수" in result.conversation.response_text
    assert "과거 상담" not in result.conversation.response_text
    assert "상담이력" not in result.conversation.response_text
    assert "참고" not in result.conversation.response_text


def test_smalltalk_does_not_claim_history_reference(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "smalltalk-history-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(
        memory.store.save_user_file(
            "smalltalk-history-user",
            "history.md",
            "\n---\n### 이전 상담\n- Q: 고혈압약 녹용 문의\n- A: 주의가 필요하다고 안내\n",
        )
    )
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="잘 먹었습니다",
            speaker_id="smalltalk-history-user",
            include_judge=False,
            include_delivery_llm=False,
        )
    )

    assert "과거 상담" not in result.conversation.response_text
    assert "참고" not in result.conversation.response_text
    assert "듣고 있어요" not in result.conversation.response_text


def test_dur_summary_hides_internal_tool_labels(tmp_path):
    memory = make_memory(tmp_path)
    judge = LLMJudgeEngine()
    reasoning = ReasoningEngine(memory, judge)

    summary = reasoning._summarize_dur(
        "아스피린장용정",
        {
            "combination_contraindication": {
                "endpoint": "병용 금기 정보조회 (T2)",
                "items": [
                    {
                        "ITEM_NAME": "아스피린장용정",
                        "PROHBT_CONTENT": "혈액학적 독성",
                    }
                ],
            }
        },
    )

    assert "T2" not in summary
    assert "DUR" not in summary
    assert "정보조회" not in summary
    assert "함께 먹으면 안 되는 조합" in summary


def test_vitamin_question_does_not_replay_old_deer_antler_context(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "vitamin-user",
            {"name": "김영수", "gender": "남성", "age": "72"},
            mark_verified=True,
        )
    )
    run(
        memory.store.write_flash(
            "context_memory",
            "# 대화 컨텍스트 메모리\n\n- 질문: 고혈압약 녹용 먹어도 돼?\n- 핵심 응답: 녹용은 주의가 필요합니다.\n",
        )
    )
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="비타민은?",
            speaker_id="vitamin-user",
            include_judge=False,
            include_delivery_llm=False,
        )
    )

    assert "비타민" in result.conversation.response_text
    assert "녹용" not in result.conversation.response_text
    assert "고혈압약" not in result.conversation.response_text


def test_wake_word_only_is_suppressed_without_tts(tmp_path):
    memory = make_memory(tmp_path)
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="오디스",
            speaker_id="noise-user",
            include_judge=False,
            include_delivery_llm=False,
        )
    )

    assert result.conversation.response_type == "ignored"
    assert result.conversation.response_text == ""
    assert result.conversation.requires_tts is False
    assert result.execution_results.get("suppressed") is True


def test_out_of_scope_stt_noise_is_answered_without_llm_search(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 와파린정\n",
        )
    )
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="기상캐스터 설탕 얘기",
            speaker_id="noise-user",
            include_judge=False,
            include_delivery_llm=False,
        )
    )

    assert result.decision.rationale in {
        "assistant_general_smalltalk",
        "smalltalk_detected",
        "assistant_answered_former_suppressed_turn",
        "local_llm_route:noise_fragment",
        "local_llm_route:unknown",
        "local_llm_route:non_actionable_ack",
    }
    assert result.conversation.response_type == "smalltalk"
    assert result.conversation.response_text
    assert result.conversation.requires_tts is True
    # 일반 대화(unclear smalltalk)는 delivery LLM 자연 응답을 허용하지만,
    # 비활성/실패 시에도 외부 LLM 검색 없이 결정적 응답으로 마무리되어야 한다.
    assert not any(event.action == "call_local_delivery_llm" for event in result.engine_trace)


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
    assert result.identity_gate["reason"] == "prior_conversation_check"
    assert result.decision.intent == "identity_check"
    assert "일전에 대화" in result.conversation.response_text
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


def test_generic_blood_pressure_medication_overview_does_not_call_dur(tmp_path):
    memory = make_memory(tmp_path)
    run(
        memory.save_identity_profile(
            "bp-overview-user",
            {"name": "김철수", "gender": "남성", "age": "55"},
            mark_verified=True,
        )
    )
    orchestrator = make_orchestrator(memory)

    result = run(
        orchestrator.run_turn(
            text="내가 혈압약 먹고 있는데 혈압약에 어떤거들이 있는지 확인해줄 수 있어?",
            speaker_id="bp-overview-user",
            include_judge=False,
            include_delivery_llm=False,
            run_identity_gate=True,
        )
    )

    assert result.decision.mode == ReasoningMode.MEMORY_ONLY
    assert result.decision.rationale == "generic_blood_pressure_medication_overview"
    assert not result.tool_trace
    assert "ACE" in result.conversation.response_text
    assert "ARB" in result.conversation.response_text
    assert "약봉투" in result.conversation.response_text
    assert "확인된 정보가 제한적" not in result.conversation.response_text


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
        (
            "혹시 내가 머리가 아파서그런데 타이레놀 4개를 한번에 먹어도 문제가 없을까",
            ReasoningMode.MEMORY_ONLY,
            ["타이레놀", "간 손상", "의사나 약사"],
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


def test_after_meal_guidance_prefers_current_prescription_over_recent_context(tmp_path):
    memory = make_memory(tmp_path)
    orchestrator = make_orchestrator(memory)

    context = {
        "user_profile": {"name": "김영수", "age": "72", "gender": "남성"},
        "prescription_log": (
            "# 현재 복용 약 요약\n\n"
            "## 약품 목록\n"
            "- 타이레놀정500밀리그람\n"
            "- 아스피린프로텍트정100밀리그람\n"
        ),
        "context_memory": "- 최근 질문: 혈압약 두 개 동시에 먹어도 돼?\n",
    }

    result = run(
        orchestrator.run_turn(
            text="나 밥 먹었어",
            speaker_id="meal-current-prescription-user",
            include_judge=False,
            include_delivery_llm=False,
            allow_frontier_memory_fallback=False,
            preloaded_context=context,
        )
    )

    assert "타이레놀정500밀리그람" in result.conversation.response_text
    assert "아스피린프로텍트정100밀리그람" in result.conversation.response_text
    assert "혈압약" not in result.conversation.response_text
