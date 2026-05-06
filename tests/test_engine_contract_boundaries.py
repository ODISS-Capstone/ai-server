"""Regression tests for per-engine responsibility boundaries."""
from __future__ import annotations

import asyncio

from app.engines.conversation import ConversationEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import IntentType, ReasoningEngine
from app.schemas.engine_contracts import (
    ConversationComposeRequest,
    MemoryEvidenceRequest,
    ReasoningMode,
    ReasoningRouteDecision,
    ReasoningRouteInput,
)


def run(coro):
    return asyncio.run(coro)


def test_reasoning_route_execution_modes_are_contractual():
    engine = ReasoningEngine(MemoryEngine(), LLMJudgeEngine())

    clarify = engine.route_execution(
        ReasoningRouteInput(text="", context={})
    )
    assert clarify.mode == ReasoningMode.ASK_USER_CLARIFY

    smalltalk = engine.route_execution(
        ReasoningRouteInput(text="안녕하세요", is_smalltalk=True, context={})
    )
    assert smalltalk.intent == IntentType.SMALLTALK
    assert smalltalk.mode == ReasoningMode.MEMORY_ONLY

    tool_first = engine.route_execution(
        ReasoningRouteInput(text="와파린이랑 아스피린 같이 먹어도 돼?", context={})
    )
    assert tool_first.mode == ReasoningMode.TOOL_FIRST
    assert tool_first.tasks

    frontier = engine.route_execution(
        ReasoningRouteInput(text="오늘 뉴스 요약해줘", is_smalltalk=False, context={})
    )
    assert frontier.mode == ReasoningMode.FRONTIER_FIRST


def test_conversation_contract_never_exposes_think_token():
    ce = ConversationEngine()
    decision = ReasoningRouteDecision(
        mode=ReasoningMode.TOOL_FIRST,
        intent=IntentType.MEDICATION_QUERY,
        rationale="test",
        tasks=[],
    )
    result = ce.compose_from_contract(
        ConversationComposeRequest(
            input_text="이 약 같이 먹어도 돼?",
            user_profile={"name": "홍길동"},
            decision=decision,
            core_message="",
            reviewed_message="",
            delivery_message=(
                "<think>1. internal reasoning</think>\n"
                "와파린과 아스피린은 출혈 위험이 올라갈 수 있습니다."
            ),
        )
    )
    assert "<think>" not in result.response_text
    assert "의사·약사 상담" in result.response_text


def test_memory_engine_fallback_activates_only_when_dur_not_searchable(monkeypatch):
    engine = MemoryEngine()

    async def fake_search_history(query: str, speaker_id=None):
        return {"structured_memory": {"items": [], "briefs": [], "prompt": "테스트 메모리"}}

    async def fake_llm_search(query: str, context=None):
        return {"success": True, "answer": "검색 API 미지원 항목이라 LLM 검색으로 보완했습니다."}

    monkeypatch.setattr(engine, "search_history", fake_search_history)
    monkeypatch.setattr("app.engines.memory.llm_search.llm_search", fake_llm_search)

    evidence = run(
        engine.prepare_evidence_bundle(
            MemoryEvidenceRequest(
                query="이 약 괜찮아?",
                speaker_id="speaker-test",
                ocr_payload=None,
                allow_frontier_fallback=True,
            )
        )
    )
    assert evidence.dur_searchable is False
    assert evidence.used_frontier_fallback is True
    assert "LLM 검색" in evidence.frontier_answer_preview
