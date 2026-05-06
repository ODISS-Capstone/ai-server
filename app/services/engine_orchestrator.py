"""Shared runtime orchestrator for Conversation/Memory/Reasoning engines."""
from __future__ import annotations

import json
from typing import Any, Optional

from app.engines.conversation import ConversationEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import ReasoningEngine
from app.schemas.engine_contracts import (
    ConversationComposeRequest,
    EnginePipelineResult,
    MemoryEvidenceRequest,
    ReasoningMode,
    ReasoningRouteInput,
)
from app.services.llm import call_local_delivery_llm


class EngineOrchestrator:
    """Runs one end-to-end turn with explicit per-engine contracts."""

    def __init__(
        self,
        *,
        memory_engine: MemoryEngine,
        reasoning_engine: ReasoningEngine,
        conversation_engine: ConversationEngine,
        llm_judge: LLMJudgeEngine,
    ) -> None:
        self.memory = memory_engine
        self.reasoning = reasoning_engine
        self.conversation = conversation_engine
        self.llm_judge = llm_judge

    async def run_turn(
        self,
        *,
        text: str,
        speaker_id: Optional[str] = None,
        include_judge: bool = True,
        include_delivery_llm: bool = True,
        allow_frontier_memory_fallback: bool = True,
    ) -> EnginePipelineResult:
        await self.memory.initialize()

        input_data = self.conversation.receive_input(text, speaker_id)
        filler_text = self.conversation.generate_filler(input_data) or ""
        context = await self.memory.load_context(speaker_id)

        decision = self.reasoning.route_execution(
            ReasoningRouteInput(
                text=text,
                speaker_id=speaker_id,
                is_smalltalk=input_data.get("is_smalltalk", False),
                context=context,
            )
        )

        evidence = await self.memory.prepare_evidence_bundle(
            MemoryEvidenceRequest(
                query=text,
                speaker_id=speaker_id,
                ocr_payload=None,
                allow_frontier_fallback=allow_frontier_memory_fallback,
            )
        )

        execution_results: dict[str, Any] = {
            "intent": decision.intent,
            "query": text,
            "task_results": {},
            "emergency": False,
        }
        if decision.mode == ReasoningMode.TOOL_FIRST:
            execution_results = await self.reasoning.execute_tasks(
                text=text,
                intent=decision.intent,
                context=context,
                tasks=decision.tasks,
            )
        elif decision.mode == ReasoningMode.MEMORY_ONLY:
            history = await self.memory.search_history(text, speaker_id=speaker_id)
            execution_results["task_results"]["history"] = history
        elif decision.mode == ReasoningMode.ASK_USER_CLARIFY:
            execution_results["task_results"]["clarify_required"] = True

        core_message = await self._build_core_message(
            text=text,
            decision_mode=decision.mode,
            execution_results=execution_results,
            evidence=evidence,
        )

        judge_review: dict[str, Any] = {}
        reviewed_message = core_message
        if include_judge and core_message:
            judge_review = await self.llm_judge.review_final_answer(
                core_message=core_message,
                original_query=text,
                additional_context=self._build_review_context(context, execution_results),
            )
            reviewed_message = judge_review.get("reviewed_text") or core_message

        delivery_message = reviewed_message
        if include_delivery_llm and reviewed_message:
            delivery_message = await call_local_delivery_llm(
                original_query=text,
                reviewed_message=reviewed_message,
                user_profile=context.get("user_profile"),
                conversation_context=context.get("context_memory", ""),
            )

        conversation = self.conversation.compose_from_contract(
            ConversationComposeRequest(
                input_text=text,
                user_profile=context.get("user_profile", {}) or {},
                decision=decision,
                evidence=evidence,
                core_message=core_message,
                reviewed_message=reviewed_message,
                delivery_message=delivery_message,
            )
        )

        return EnginePipelineResult(
            input_data=input_data,
            context=context,
            decision=decision,
            evidence=evidence,
            execution_results=execution_results,
            filler_text=filler_text,
            core_message=core_message,
            judge_review=judge_review,
            reviewed_message=reviewed_message,
            delivery_message=delivery_message,
            conversation=conversation,
        )

    async def _build_core_message(
        self,
        *,
        text: str,
        decision_mode: ReasoningMode,
        execution_results: dict[str, Any],
        evidence,
    ) -> str:
        if decision_mode == ReasoningMode.ASK_USER_CLARIFY:
            return "약 이름이나 복용 상황을 조금 더 구체적으로 알려주시면 확인해 드릴 수 있습니다."

        if decision_mode == ReasoningMode.FRONTIER_FIRST and evidence.frontier_answer_preview:
            return evidence.frontier_answer_preview

        if decision_mode in (ReasoningMode.TOOL_FIRST, ReasoningMode.MEMORY_ONLY):
            return await self.reasoning.synthesize_core_message(
                execution_results,
                verify_with_judge=False,
            )

        # fallback: no tool execution result
        if evidence.frontier_answer_preview:
            return evidence.frontier_answer_preview
        return f"'{text}'에 대해 확인 가능한 근거를 더 모은 뒤 안내드리겠습니다."

    def _build_review_context(
        self,
        context: dict[str, Any],
        execution_results: dict[str, Any],
    ) -> str:
        task_results = execution_results.get("task_results", {})
        parts: list[str] = [f"의도: {execution_results.get('intent', '')}"]

        prescription_log = context.get("prescription_log")
        if prescription_log:
            parts.append(f"[현재 복약 요약]\n{prescription_log[:1200]}")

        memory_prompt = context.get("memory_prompt")
        if memory_prompt:
            parts.append(f"[관련 메모리]\n{memory_prompt[:1200]}")

        for key in ("dur", "supplements", "hira"):
            payload = task_results.get(key)
            if payload:
                parts.append(
                    f"[{key} 결과]\n"
                    + json.dumps(payload, ensure_ascii=False, default=str)[:1800]
                )
        return "\n\n".join(parts)
