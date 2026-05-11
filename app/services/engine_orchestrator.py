"""Shared runtime orchestrator for Conversation/Memory/Reasoning engines."""
from __future__ import annotations

import json
import logging
from time import perf_counter
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

logger = logging.getLogger(__name__)


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
        turn_started = perf_counter()
        logger.info(
            "[EnginePipeline] turn_start speaker_id=%s text_chars=%d include_judge=%s include_delivery_llm=%s",
            speaker_id or "-",
            len(text or ""),
            include_judge,
            include_delivery_llm,
        )

        stage_started = perf_counter()
        await self.memory.initialize()
        logger.info(
            "[MemoryEngine] initialized elapsed_ms=%.1f",
            (perf_counter() - stage_started) * 1000,
        )

        stage_started = perf_counter()
        input_data = self.conversation.receive_input(text, speaker_id)
        filler_text = self.conversation.generate_filler(input_data) or ""
        logger.info(
            "[ConversationEngine] input_received smalltalk=%s smalltalk_type=%s filler=%s elapsed_ms=%.1f",
            input_data.get("is_smalltalk"),
            input_data.get("smalltalk_type") or "-",
            bool(filler_text),
            (perf_counter() - stage_started) * 1000,
        )

        stage_started = perf_counter()
        context = await self.memory.load_context(speaker_id)
        logger.info(
            "[MemoryEngine] context_loaded speaker_id=%s new_user=%s memory_items=%d elapsed_ms=%.1f",
            speaker_id or "-",
            context.get("is_new_user", False),
            len(context.get("relevant_memories") or []),
            (perf_counter() - stage_started) * 1000,
        )

        stage_started = perf_counter()
        decision = self.reasoning.route_execution(
            ReasoningRouteInput(
                text=text,
                speaker_id=speaker_id,
                is_smalltalk=input_data.get("is_smalltalk", False),
                context=context,
            )
        )
        logger.info(
            "[ReasoningEngine] route_decided mode=%s intent=%s tasks=%d rationale=%s elapsed_ms=%.1f",
            decision.mode,
            decision.intent,
            len(decision.tasks),
            decision.rationale,
            (perf_counter() - stage_started) * 1000,
        )

        stage_started = perf_counter()
        evidence = await self.memory.prepare_evidence_bundle(
            MemoryEvidenceRequest(
                query=text,
                speaker_id=speaker_id,
                ocr_payload=None,
                allow_frontier_fallback=allow_frontier_memory_fallback,
            )
        )
        logger.info(
            "[MemoryEngine] evidence_prepared meds=%d dur_searchable=%s frontier_fallback=%s artifacts=%d elapsed_ms=%.1f",
            len(evidence.normalized_medications),
            evidence.dur_searchable,
            evidence.used_frontier_fallback,
            len(evidence.artifact_refs),
            (perf_counter() - stage_started) * 1000,
        )

        execution_results: dict[str, Any] = {
            "intent": decision.intent,
            "query": text,
            "task_results": {},
            "emergency": False,
        }
        if decision.mode == ReasoningMode.TOOL_FIRST:
            stage_started = perf_counter()
            execution_results = await self.reasoning.execute_tasks(
                text=text,
                intent=decision.intent,
                context=context,
                tasks=decision.tasks,
            )
            logger.info(
                "[ReasoningEngine] tasks_executed result_keys=%s emergency=%s elapsed_ms=%.1f",
                sorted(execution_results.get("task_results", {}).keys()),
                execution_results.get("emergency", False),
                (perf_counter() - stage_started) * 1000,
            )
        elif decision.mode == ReasoningMode.MEMORY_ONLY:
            stage_started = perf_counter()
            history = await self.memory.search_history(text, speaker_id=speaker_id)
            execution_results["task_results"]["history"] = history
            logger.info(
                "[MemoryEngine] history_loaded categories=%s elapsed_ms=%.1f",
                sorted(history.keys()),
                (perf_counter() - stage_started) * 1000,
            )
        elif decision.mode == ReasoningMode.ASK_USER_CLARIFY:
            execution_results["task_results"]["clarify_required"] = True

        stage_started = perf_counter()
        core_message = await self._build_core_message(
            text=text,
            decision_mode=decision.mode,
            execution_results=execution_results,
            evidence=evidence,
        )
        logger.info(
            "[ReasoningEngine] core_message_built chars=%d elapsed_ms=%.1f",
            len(core_message or ""),
            (perf_counter() - stage_started) * 1000,
        )

        judge_review: dict[str, Any] = {}
        reviewed_message = core_message
        if include_judge and core_message:
            stage_started = perf_counter()
            judge_review = await self.llm_judge.review_final_answer(
                core_message=core_message,
                original_query=text,
                additional_context=self._build_review_context(context, execution_results),
            )
            reviewed_message = judge_review.get("reviewed_text") or core_message
            logger.info(
                "[LLMJudgeEngine] final_review reviewed=%s model=%s chars=%d elapsed_ms=%.1f",
                judge_review.get("reviewed", False),
                judge_review.get("model", "-"),
                len(reviewed_message or ""),
                (perf_counter() - stage_started) * 1000,
            )

        delivery_message = reviewed_message
        if include_delivery_llm and reviewed_message:
            stage_started = perf_counter()
            delivery_message = await call_local_delivery_llm(
                original_query=text,
                reviewed_message=reviewed_message,
                user_profile=context.get("user_profile"),
                conversation_context=context.get("context_memory", ""),
            )
            logger.info(
                "[DeliveryLLM] message_rewritten chars=%d elapsed_ms=%.1f",
                len(delivery_message or ""),
                (perf_counter() - stage_started) * 1000,
            )

        stage_started = perf_counter()
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
        logger.info(
            "[ConversationEngine] response_composed type=%s chars=%d elapsed_ms=%.1f total_elapsed_ms=%.1f",
            conversation.response_type,
            len(conversation.response_text or ""),
            (perf_counter() - stage_started) * 1000,
            (perf_counter() - turn_started) * 1000,
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
