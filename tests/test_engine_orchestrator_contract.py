"""EngineOrchestrator contract-order tests."""
from __future__ import annotations

import asyncio

from app.schemas.engine_contracts import (
    ConversationComposeResponse,
    MemoryEvidenceBundle,
    ReasoningMode,
    ReasoningRouteDecision,
)
from app.services.engine_orchestrator import EngineOrchestrator


def run(coro):
    return asyncio.run(coro)


def test_engine_orchestrator_runs_in_contract_order(monkeypatch):
    calls: list[str] = []

    class FakeMemory:
        async def initialize(self):
            calls.append("memory.initialize")

        async def load_context(self, speaker_id=None):
            calls.append("memory.load_context")
            return {"speaker_id": speaker_id, "user_profile": {"name": "테스터"}}

        async def prepare_evidence_bundle(self, request):
            calls.append("memory.prepare_evidence_bundle")
            return MemoryEvidenceBundle(
                normalized_query=request.query,
                normalized_medications=["아스피린정"],
                dur_searchable=True,
                used_frontier_fallback=False,
                frontier_answer_preview="",
                artifact_refs=[],
                summary="메모리 요약",
                memory_prompt="prompt",
            )

    class FakeReasoning:
        def route_execution(self, route_input):
            calls.append("reasoning.route_execution")
            return ReasoningRouteDecision(
                mode=ReasoningMode.TOOL_FIRST,
                intent="medication_query",
                rationale="test",
                tasks=[],
            )

        async def execute_tasks(self, **kwargs):
            calls.append("reasoning.execute_tasks")
            return {
                "intent": "medication_query",
                "query": kwargs["text"],
                "task_results": {},
                "emergency": False,
            }

        async def synthesize_core_message(self, execution_results, verify_with_judge=False):
            calls.append("reasoning.synthesize_core_message")
            return "핵심 메시지"

    class FakeConversation:
        def receive_input(self, text, speaker_id=None):
            calls.append("conversation.receive_input")
            return {"text": text, "is_smalltalk": False, "speaker_id": speaker_id}

        def generate_filler(self, input_data):
            calls.append("conversation.generate_filler")
            return "잠시만요"

        def compose_from_contract(self, contract):
            calls.append("conversation.compose_from_contract")
            return ConversationComposeResponse(
                response_text=contract.delivery_message or contract.reviewed_message,
                response_type="medical_response",
                requires_tts=True,
            )

    class FakeJudge:
        async def review_final_answer(self, core_message, original_query, additional_context=None):
            calls.append("judge.review_final_answer")
            return {"reviewed_text": f"[judge]{core_message}"}

    async def fake_local_delivery_llm(**kwargs):
        calls.append("llm.call_local_delivery")
        return f"[delivery]{kwargs['reviewed_message']}"

    monkeypatch.setattr(
        "app.services.engine_orchestrator.call_local_delivery_llm",
        fake_local_delivery_llm,
    )

    orchestrator = EngineOrchestrator(
        memory_engine=FakeMemory(),
        reasoning_engine=FakeReasoning(),
        conversation_engine=FakeConversation(),
        llm_judge=FakeJudge(),
    )
    result = run(orchestrator.run_turn(text="질문", speaker_id="spk-1"))

    assert result.conversation.response_text.startswith("[delivery][judge]핵심")
    assert calls == [
        "memory.initialize",
        "conversation.receive_input",
        "conversation.generate_filler",
        "memory.load_context",
        "reasoning.route_execution",
        "memory.prepare_evidence_bundle",
        "reasoning.execute_tasks",
        "reasoning.synthesize_core_message",
        "judge.review_final_answer",
        "llm.call_local_delivery",
        "conversation.compose_from_contract",
    ]
