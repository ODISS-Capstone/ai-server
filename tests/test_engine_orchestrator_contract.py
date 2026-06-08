"""EngineOrchestrator contract-order tests."""
from __future__ import annotations

import asyncio

from app.engines.conversation import ConversationEngine
from app.schemas.engine_contracts import (
    ConversationComposeResponse,
    MemoryEvidenceBundle,
    ReasoningMode,
    ReasoningRouteDecision,
    ReasoningTask,
)
from app.services.engine_orchestrator import EngineOrchestrator


def run(coro):
    return asyncio.run(coro)


async def _disable_local_llm_route(*args, **kwargs):
    return {"source": "test_disabled", "usable": False}


def test_engine_orchestrator_runs_in_contract_order(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        EngineOrchestrator,
        "_classify_route_with_local_llm",
        _disable_local_llm_route,
    )

    class FakeMemory:
        async def initialize(self):
            calls.append("memory.initialize")

        async def load_context(self, speaker_id=None):
            calls.append("memory.load_context")
            return {"speaker_id": speaker_id, "user_profile": {"name": "Tester"}}

        async def prepare_evidence_bundle(self, request):
            calls.append("memory.prepare_evidence_bundle")
            return MemoryEvidenceBundle(
                normalized_query=request.query,
                normalized_medications=["aspirin"],
                dur_searchable=True,
                used_frontier_fallback=False,
                frontier_answer_preview="",
                artifact_refs=[],
                summary="memory summary",
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
            return "core message"

    class FakeConversation:
        def receive_input(self, text, speaker_id=None):
            calls.append("conversation.receive_input")
            return {"text": text, "is_smalltalk": False, "speaker_id": speaker_id}

        def generate_filler(self, input_data):
            calls.append("conversation.generate_filler")
            return "checking"

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
    result = run(orchestrator.run_turn(text="can I take aspirin together?", speaker_id="spk-1"))

    assert result.conversation.response_text.startswith("[delivery][judge]core")
    assert [event.stage for event in result.engine_trace[:6]] == [
        "ME_Initialize",
        "CE_Input",
        "CE_Latency",
        "ME_Context",
        "RE_Intent",
        "ME_RAG",
    ]
    assert any(event.logical_file == "Patient.md" for event in result.memory_trace)
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


def test_engine_orchestrator_smalltalk_fast_path_skips_rag_history_and_delivery(monkeypatch):
    calls: list[str] = []

    class FakeMemory:
        async def initialize(self):
            calls.append("memory.initialize")

        async def load_context(self, speaker_id=None):
            calls.append("memory.load_context")
            return {"speaker_id": speaker_id, "user_profile": {"name": "Tester"}}

        async def prepare_evidence_bundle(self, request):
            raise AssertionError("smalltalk fast path should not prepare evidence")

        async def search_history(self, query_text, speaker_id=None):
            raise AssertionError("smalltalk fast path should not search history")

    class FakeReasoning:
        def route_execution(self, route_input):
            raise AssertionError("smalltalk fast path should not call route execution")

    class FakeJudge:
        async def review_final_answer(self, core_message, original_query, additional_context=None):
            raise AssertionError("smalltalk fast path should not call judge")

    async def fail_delivery(**kwargs):
        raise AssertionError("smalltalk fast path should not call delivery LLM")

    monkeypatch.setattr(
        "app.services.engine_orchestrator.call_local_delivery_llm",
        fail_delivery,
    )

    orchestrator = EngineOrchestrator(
        memory_engine=FakeMemory(),
        reasoning_engine=FakeReasoning(),
        conversation_engine=ConversationEngine(),
        llm_judge=FakeJudge(),
    )
    result = run(
        orchestrator.run_turn(
            text="안녕",
            speaker_id="spk-1",
            include_judge=True,
            include_delivery_llm=True,
        )
    )

    assert calls == ["memory.initialize", "memory.load_context"]
    assert result.decision.intent == "smalltalk"
    assert result.decision.rationale == "smalltalk_detected"
    assert result.conversation.response_type == "smalltalk"
    assert result.conversation.response_text.count("Tester님") == 1
    assert "확인된 정보가 제한적" not in result.conversation.response_text
    assert any(
        event.stage == "DeliveryLLM"
        and event.status == "skipped"
        and event.metadata.get("delivery_skipped_reason") == "smalltalk_fast_path"
        for event in result.engine_trace
    )


def test_engine_orchestrator_skips_delivery_for_safety_fast_path(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        EngineOrchestrator,
        "_classify_route_with_local_llm",
        _disable_local_llm_route,
    )

    class FakeMemory:
        async def initialize(self):
            calls.append("memory.initialize")

        async def load_context(self, speaker_id=None):
            calls.append("memory.load_context")
            return {"speaker_id": speaker_id, "user_profile": {"name": "Tester"}}

        async def prepare_evidence_bundle(self, request):
            calls.append("memory.prepare_evidence_bundle")
            return MemoryEvidenceBundle(
                normalized_query=request.query,
                normalized_medications=[],
                dur_searchable=False,
                used_frontier_fallback=False,
                frontier_answer_preview="",
                artifact_refs=[],
                summary="",
                memory_prompt="",
            )

        async def search_history(self, query_text, speaker_id=None):
            calls.append("memory.search_history")
            return {}

    class FakeReasoning:
        def route_execution(self, route_input):
            calls.append("reasoning.route_execution")
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent="medication_query",
                rationale="deterministic_patient_safety:acetaminophen_excess_dose",
                tasks=[],
            )

        async def synthesize_core_message(self, execution_results, verify_with_judge=False):
            calls.append("reasoning.synthesize_core_message")
            return "Tylenol safety core"

    class FakeConversation:
        def receive_input(self, text, speaker_id=None):
            calls.append("conversation.receive_input")
            return {"text": text, "is_smalltalk": False, "speaker_id": speaker_id}

        def generate_filler(self, input_data):
            calls.append("conversation.generate_filler")
            return "checking"

        def compose_from_contract(self, contract):
            calls.append("conversation.compose_from_contract")
            return ConversationComposeResponse(
                response_text=contract.delivery_message or contract.reviewed_message,
                response_type="medical_response",
                requires_tts=True,
            )

    class FakeJudge:
        async def review_final_answer(self, core_message, original_query, additional_context=None):
            raise AssertionError("patient safety fast path should not call judge")

    async def fail_delivery(**kwargs):
        raise AssertionError("patient safety fast path should not call delivery LLM")

    monkeypatch.setattr(
        "app.services.engine_orchestrator.call_local_delivery_llm",
        fail_delivery,
    )

    orchestrator = EngineOrchestrator(
        memory_engine=FakeMemory(),
        reasoning_engine=FakeReasoning(),
        conversation_engine=FakeConversation(),
        llm_judge=FakeJudge(),
    )
    result = run(orchestrator.run_turn(text="Can I take Tylenol 4 tablets at once?", speaker_id="spk-1"))

    assert result.conversation.response_text == "Tylenol safety core"
    assert "llm.call_local_delivery" not in calls
    assert any(
        event.stage == "DeliveryLLM"
        and event.status == "skipped"
        and event.metadata.get("delivery_skipped_reason") == "patient_safety_fast_path"
        for event in result.engine_trace
    )


def test_engine_orchestrator_skips_delivery_for_tool_safety_fast_path(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        EngineOrchestrator,
        "_classify_route_with_local_llm",
        _disable_local_llm_route,
    )

    class FakeMemory:
        async def initialize(self):
            calls.append("memory.initialize")

        async def load_context(self, speaker_id=None):
            calls.append("memory.load_context")
            return {"speaker_id": speaker_id, "user_profile": {"name": "Tester"}}

        async def prepare_evidence_bundle(self, request):
            calls.append("memory.prepare_evidence_bundle")
            return MemoryEvidenceBundle(
                normalized_query=request.query,
                normalized_medications=["aspirin"],
                dur_searchable=True,
                used_frontier_fallback=False,
                frontier_answer_preview="",
                artifact_refs=[],
                summary="",
                memory_prompt="",
            )

    class FakeReasoning:
        def route_execution(self, route_input):
            calls.append("reasoning.route_execution")
            return ReasoningRouteDecision(
                mode=ReasoningMode.TOOL_FIRST,
                intent="medication_query",
                rationale="deterministic_tools_available",
                tasks=[
                    ReasoningTask(
                        type="dur_check",
                        priority=1,
                        description="DUR safety check",
                    )
                ],
            )

        async def execute_tasks(self, **kwargs):
            calls.append("reasoning.execute_tasks")
            return {
                "intent": "medication_query",
                "query": kwargs["text"],
                "task_results": {"dur": {}},
                "emergency": False,
            }

        async def synthesize_core_message(self, execution_results, verify_with_judge=False):
            calls.append("reasoning.synthesize_core_message")
            return "DUR safety core"

    class FakeConversation:
        def receive_input(self, text, speaker_id=None):
            calls.append("conversation.receive_input")
            return {"text": text, "is_smalltalk": False, "speaker_id": speaker_id}

        def generate_filler(self, input_data):
            calls.append("conversation.generate_filler")
            return "checking"

        def compose_from_contract(self, contract):
            calls.append("conversation.compose_from_contract")
            return ConversationComposeResponse(
                response_text=contract.delivery_message or contract.reviewed_message,
                response_type="medical_response",
                requires_tts=True,
            )

    class FakeJudge:
        async def review_final_answer(self, core_message, original_query, additional_context=None):
            raise AssertionError("non-emergency DUR turn should not call judge")

    async def recording_delivery(**kwargs):
        calls.append("llm.call_local_delivery")
        return "정리된 DUR 안내"

    monkeypatch.setattr(
        "app.services.engine_orchestrator.call_local_delivery_llm",
        recording_delivery,
    )

    orchestrator = EngineOrchestrator(
        memory_engine=FakeMemory(),
        reasoning_engine=FakeReasoning(),
        conversation_engine=FakeConversation(),
        llm_judge=FakeJudge(),
    )
    result = run(orchestrator.run_turn(text="can I take these together?", speaker_id="spk-1"))

    # DUR/약물 안전 턴은 이제 delivery LLM(Together)으로 자연어 가공한다.
    assert "llm.call_local_delivery" in calls
    assert result.delivery_message == "정리된 DUR 안내"
    assert not any(
        event.stage == "DeliveryLLM"
        and event.status == "skipped"
        and event.metadata.get("delivery_skipped_reason") == "tool_safety_fast_path"
        for event in result.engine_trace
    )
    assert any(
        event.stage == "DeliveryLLM" and event.status != "skipped"
        for event in result.engine_trace
    )
