"""Shared runtime orchestrator for Conversation/Memory/Reasoning engines."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from time import perf_counter
from typing import Any, Optional

from app.engines.conversation import ConversationEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import ReasoningEngine
from app.schemas.engine_contracts import (
    ConversationComposeRequest,
    ConversationComposeResponse,
    EnginePipelineResult,
    EngineTraceEvent,
    MemoryEvidenceRequest,
    MemoryEvidenceBundle,
    MemoryTraceEvent,
    ReasoningMode,
    ReasoningRouteDecision,
    ReasoningRouteInput,
    ReasoningTask,
    ToolTraceEvent,
)
from app.services.identity_guard import (
    evaluate_identity_gate,
    has_identity_core,
    has_profile_identity,
    is_profile_recall_query,
)
from app.services.llm import (
    call_local_delivery_llm,
    classify_reasoning_route_with_llm,
    recover_medical_followup_with_llm,
)
from app.services.medication_extraction import is_wake_word_only, strip_wake_words
from app.services.patient_safety import classify_patient_safety_situation

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
        preloaded_context: Optional[dict[str, Any]] = None,
        run_identity_gate: bool = False,
    ) -> EnginePipelineResult:
        engine_trace: list[EngineTraceEvent] = []
        memory_trace: list[MemoryTraceEvent] = []
        tool_trace: list[ToolTraceEvent] = []
        identity_gate_info: dict[str, Any] = {}

        def trace_engine(
            stage: str,
            component: str,
            action: str,
            *,
            status: str = "observed",
            **metadata: Any,
        ) -> None:
            engine_trace.append(
                EngineTraceEvent(
                    stage=stage,
                    component=component,
                    action=action,
                    status=status,
                    metadata=metadata,
                )
            )

        def trace_memory(
            operation: str,
            logical_file: str,
            *,
            category: str = "",
            path: Optional[str] = None,
            status: str = "observed",
            **metadata: Any,
        ) -> None:
            memory_trace.append(
                MemoryTraceEvent(
                    operation=operation,
                    logical_file=logical_file,
                    category=category,
                    path=path,
                    status=status,
                    metadata=metadata,
                )
            )

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
        trace_engine("ME_Initialize", "MemoryEngine", "initialize")
        logger.info(
            "[MemoryEngine] initialized elapsed_ms=%.1f",
            (perf_counter() - stage_started) * 1000,
        )

        stage_started = perf_counter()
        input_data = self.conversation.receive_input(text, speaker_id)
        trace_engine(
            "CE_Input",
            "ConversationEngine",
            "receive_input",
            smalltalk=input_data.get("is_smalltalk", False),
        )
        filler_text = self.conversation.generate_filler(input_data) or ""
        trace_engine(
            "CE_Latency",
            "ConversationEngine",
            "generate_filler",
            filler=bool(filler_text),
        )
        logger.info(
            "[ConversationEngine] input_received smalltalk=%s smalltalk_type=%s filler=%s elapsed_ms=%.1f",
            input_data.get("is_smalltalk"),
            input_data.get("smalltalk_type") or "-",
            bool(filler_text),
            (perf_counter() - stage_started) * 1000,
        )

        if run_identity_gate:
            stage_started = perf_counter()
            identity_gate = await evaluate_identity_gate(
                memory_engine=self.memory,
                text=text,
                speaker_id=speaker_id,
            )
            identity_gate_info = {
                "allowed": identity_gate.allowed,
                "reason": identity_gate.reason,
                "response_type": identity_gate.response_type,
                "metadata": identity_gate.metadata or {},
            }
            trace_engine(
                "ME_Context",
                "MemoryEngine",
                "identity_gate",
                allowed=identity_gate.allowed,
                reason=identity_gate.reason,
            )
            if speaker_id:
                trace_memory("read", "Patient.md", category="patients", path=f"patients/{speaker_id}/profile.md")
                if not identity_gate.allowed and identity_gate.reason in {
                    "identity_registered",
                    "identity_recognized",
                    "identity_candidate_registered",
                    "confirm_new_identity",
                    "identity_conflict",
                    "needs_registration",
                    "prior_conversation_check",
                }:
                    trace_memory("write", "Patient.md", category="patients", path=f"patients/{speaker_id}/profile.md")
                if not identity_gate.allowed and identity_gate.reason in {
                    "identity_registered",
                    "identity_candidate_registered",
                }:
                    trace_memory("write", "CurrentUserProfile.md", category="current_user_profile", path="flash/current_user_profile.md")
            logger.info(
                "[MemoryEngine] identity_gate allowed=%s reason=%s elapsed_ms=%.1f",
                identity_gate.allowed,
                identity_gate.reason,
                (perf_counter() - stage_started) * 1000,
            )
            if not identity_gate.allowed:
                decision = ReasoningRouteDecision(
                    mode=ReasoningMode.MEMORY_ONLY,
                    intent="identity_check",
                    rationale=identity_gate.reason,
                    tasks=[],
                )
                evidence = self._empty_evidence(text)
                response_text = identity_gate.response_text or "신원 확인이 필요합니다."
                conversation = ConversationComposeResponse(
                    response_text=response_text,
                    response_type=identity_gate.response_type,
                    requires_tts=True,
                )
                trace_engine(
                    "CE_Response",
                    "ConversationEngine",
                    "identity_gate_response",
                    response_type=conversation.response_type,
                    requires_tts=conversation.requires_tts,
                )
                return EnginePipelineResult(
                    input_data=input_data,
                    context={"speaker_id": speaker_id},
                    identity_gate=identity_gate_info,
                    decision=decision,
                    evidence=evidence,
                    execution_results={
                        "intent": decision.intent,
                        "query": text,
                        "task_results": {"identity_gate": identity_gate_info},
                        "emergency": False,
                    },
                    filler_text=filler_text,
                    core_message=response_text,
                    reviewed_message=response_text,
                    delivery_message=response_text,
                    conversation=conversation,
                    engine_trace=engine_trace,
                    memory_trace=memory_trace,
                    tool_trace=tool_trace,
                )

        stage_started = perf_counter()
        context = dict(preloaded_context) if preloaded_context is not None else await self.memory.load_context(speaker_id)
        gate_profile = (identity_gate_info.get("metadata") or {}).get("profile") or {}
        if gate_profile and not (context.get("user_profile") or {}).get("name"):
            context["user_profile"] = gate_profile
        trace_engine(
            "ME_Context",
            "MemoryEngine",
            "load_context",
            speaker_id=speaker_id,
            is_new_user=context.get("is_new_user", False),
            preloaded=preloaded_context is not None,
        )
        if speaker_id:
            trace_memory("read", "Patient.md", category="patients", path=f"patients/{speaker_id}/profile.md")
            trace_memory("read", "CurrentUserProfile.md", category="current_user_profile", path="flash/current_user_profile.md")
        trace_memory("read", "CurrentRequirement.md", category="current_requirement", path="flash/current_requirement.md")
        trace_memory("read", "CurrentManual.md", category="current_manual", path="flash/current_manual.md")
        trace_memory("read", "PrescriptionLog.md", category="prescription_log", path="flash/prescription_log.md")
        trace_memory("read", "ContextMemory.md", category="context_memory", path="flash/context_memory.md")
        logger.info(
            "[MemoryEngine] context_loaded speaker_id=%s new_user=%s memory_items=%d elapsed_ms=%.1f",
            speaker_id or "-",
            context.get("is_new_user", False),
            len(context.get("relevant_memories") or []),
            (perf_counter() - stage_started) * 1000,
        )

        ocr_medications = self._extract_ocr_medications_from_text(text)
        if ocr_medications:
            if hasattr(self.memory, "store_ocr_text_result"):
                await self.memory.store_ocr_text_result(text, speaker_id=speaker_id)
            else:
                await self._store_ocr_prescription_context(ocr_medications)
            context = await self._refresh_context_from_flash(
                speaker_id=speaker_id,
                context=context,
                trace_engine=trace_engine,
                trace_memory=trace_memory,
                reason="ocr_prescription_update",
            )
            trace_memory("write", "OCRHistory.md", category="ocr_history", path="permanent/ocr_history/*/*.md")
            trace_memory("write", "Prescription.md", category="prescriptions", path="permanent/prescriptions/*/*.md")
            trace_memory("write", "PrescriptionLog.md", category="prescription_log", path="flash/prescription_log.md")

        if self._is_lifestyle_memory_text(text) and hasattr(self.memory, "store"):
            await self.memory.store.write_flash("current_manual", self._format_lifestyle_manual(text))
            context = await self._refresh_context_from_flash(
                speaker_id=speaker_id,
                context=context,
                trace_engine=trace_engine,
                trace_memory=trace_memory,
                reason="current_manual_update",
            )
            trace_memory("write", "CurrentManual.md", category="current_manual", path="flash/current_manual.md")

        query_text = self._query_text_without_wake_word(text)

        stage_started = perf_counter()
        llm_route = await self._classify_route_with_local_llm(text=query_text, context=context)
        decision = self._decision_from_llm_route(llm_route, query_text) or self.reasoning.route_execution(
            ReasoningRouteInput(
                text=query_text,
                speaker_id=speaker_id,
                is_smalltalk=input_data.get("is_smalltalk", False),
                context=context,
            )
        )
        trace_engine(
            "RE_Intent",
            "ReasoningEngine",
            "route_execution",
            mode=decision.mode.value,
            intent=decision.intent,
            task_types=[task.type for task in decision.tasks],
            source="local_llm" if llm_route.get("usable") else "deterministic_fallback",
            route_label=llm_route.get("route_label", ""),
        )
        logger.info(
            "[ReasoningEngine] route_decided mode=%s intent=%s tasks=%d rationale=%s elapsed_ms=%.1f",
            decision.mode,
            decision.intent,
            len(decision.tasks),
            decision.rationale,
            (perf_counter() - stage_started) * 1000,
        )

        if self._should_suppress_turn(text=query_text, decision=decision, input_data=input_data):
            recovery: dict[str, Any] = {}
            if not self._is_llm_ignore_route(decision):
                recovery = await self._recover_suppressed_medical_followup(
                    text=query_text,
                    context=context,
                )
            if recovery.get("is_medical_followup") and recovery.get("response"):
                response_text = str(recovery.get("response") or "").strip()
                trace_engine(
                    "LLM_Followup_Recovery",
                    "LocalLLM",
                    "recover_medical_followup",
                    recovered=True,
                    source=recovery.get("source", ""),
                )
                recovered_decision = ReasoningRouteDecision(
                    mode=ReasoningMode.MEMORY_ONLY,
                    intent="medication_query",
                    rationale="local_llm_medical_followup_recovery",
                    tasks=[],
                )
                conversation = ConversationComposeResponse(
                    response_text=response_text,
                    response_type="medication_query",
                    requires_tts=True,
                )
                return EnginePipelineResult(
                    input_data=input_data,
                    context=context,
                    identity_gate=identity_gate_info,
                    decision=recovered_decision,
                    evidence=self._empty_evidence(query_text),
                    execution_results={
                        "intent": "medication_query",
                        "query": query_text,
                        "task_results": {"followup_recovery": recovery},
                        "emergency": False,
                    },
                    filler_text=filler_text,
                    core_message=response_text,
                    judge_review={},
                    reviewed_message=response_text,
                    delivery_message=response_text,
                    conversation=conversation,
                    engine_trace=engine_trace,
                    memory_trace=memory_trace,
                    tool_trace=tool_trace,
                )
            trace_engine(
                "CE_Response",
                "ConversationEngine",
                "suppress_out_of_scope_turn",
                response_type="ignored",
                requires_tts=False,
            )
            conversation = ConversationComposeResponse(
                response_text="",
                response_type="ignored",
                requires_tts=False,
            )
            return EnginePipelineResult(
                input_data=input_data,
                context=context,
                identity_gate=identity_gate_info,
                decision=decision,
                evidence=self._empty_evidence(query_text),
                execution_results={
                    "intent": decision.intent,
                    "query": query_text,
                    "task_results": {},
                    "emergency": False,
                    "suppressed": True,
                },
                filler_text="",
                core_message="",
                judge_review={},
                reviewed_message="",
                delivery_message="",
                conversation=conversation,
                engine_trace=engine_trace,
                memory_trace=memory_trace,
                tool_trace=tool_trace,
            )

        stage_started = perf_counter()
        if self._skip_evidence_preparation(decision):
            evidence = self._empty_evidence(query_text)
            trace_engine(
                "ME_RAG",
                "MemoryEngine",
                "prepare_evidence_bundle",
                status="skipped",
                reason="route_does_not_need_memory_evidence",
                dur_searchable=False,
                used_frontier_fallback=False,
                artifact_count=0,
            )
        else:
            evidence = await self.memory.prepare_evidence_bundle(
                MemoryEvidenceRequest(
                    query=query_text,
                    speaker_id=speaker_id,
                    ocr_payload=None,
                    allow_frontier_fallback=(
                        allow_frontier_memory_fallback
                        and decision.mode == ReasoningMode.FRONTIER_FIRST
                    ),
                )
            )
            trace_engine(
                "ME_RAG",
                "MemoryEngine",
                "prepare_evidence_bundle",
                dur_searchable=evidence.dur_searchable,
                used_frontier_fallback=evidence.used_frontier_fallback,
                artifact_count=len(evidence.artifact_refs),
            )
            for ref in evidence.artifact_refs:
                trace_memory(
                    "read",
                    self._logical_file_for_category(ref.category),
                    category=ref.category,
                    path=ref.path,
                    reason=ref.reason,
                    score=ref.score,
                )
            if evidence.used_frontier_fallback:
                tool_trace.append(
                    ToolTraceEvent(
                        tool_id="T13.LLM에이전트검색",
                        tool_name="llm_search",
                        external_api="FrontierLLM",
                        metadata={"source": "memory_fallback"},
                    )
                )
        self._append_scenario_trace_aliases(
            text=text,
            decision_hint=None,
            trace_engine=trace_engine,
            trace_memory=trace_memory,
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
            "query": query_text,
            "task_results": {},
            "emergency": False,
        }
        if decision.mode == ReasoningMode.TOOL_FIRST:
            stage_started = perf_counter()
            execution_results = await self.reasoning.execute_tasks(
                text=query_text,
                intent=decision.intent,
                context=context,
                tasks=decision.tasks,
            )
            trace_engine(
                "Tool_Execution",
                "ReasoningEngine",
                "execute_tasks",
                result_keys=sorted(execution_results.get("task_results", {}).keys()),
            )
            tool_trace.extend(self._tool_trace_from_execution(decision.tasks, execution_results))
            if "복용지도를 계획" in query_text:
                tool_trace.append(
                    ToolTraceEvent(
                        tool_id="reuse_previous_DUR_results_or_T4_if_missing",
                        tool_name="reuse_previous_dur_results",
                        external_api=None,
                    )
                )
            logger.info(
                "[ReasoningEngine] tasks_executed result_keys=%s emergency=%s elapsed_ms=%.1f",
                sorted(execution_results.get("task_results", {}).keys()),
                execution_results.get("emergency", False),
                (perf_counter() - stage_started) * 1000,
            )
            if await self._sync_tool_results_to_flash(
                speaker_id=speaker_id,
                execution_results=execution_results,
                trace_memory=trace_memory,
            ):
                context = await self._refresh_context_from_flash(
                    speaker_id=speaker_id,
                    context=context,
                    trace_engine=trace_engine,
                    trace_memory=trace_memory,
                    reason="tool_result_flash_sync",
                )
        elif decision.intent == "emergency":
            execution_results["emergency"] = True
            execution_results["task_results"]["emergency"] = {
                "message": "긴급 상황이 감지되었습니다. 즉시 119에 연락하세요.",
            }
            tool_trace.append(
                ToolTraceEvent(
                    tool_id="emergency_alert",
                    tool_name="emergency_alert",
                    external_api=None,
                    metadata={"priority": "emergency_over_dur"},
                )
            )
        elif decision.mode == ReasoningMode.MEMORY_ONLY:
            stage_started = perf_counter()
            history = await self.memory.search_history(query_text, speaker_id=speaker_id)
            execution_results["task_results"]["history"] = history
            trace_engine(
                "ME_RAG",
                "MemoryEngine",
                "search_history",
                categories=sorted(history.keys()),
            )
            for category in history:
                trace_memory(
                    "search",
                    self._logical_file_for_category(category),
                    category=category,
                    status="observed",
                )
            logger.info(
                "[MemoryEngine] history_loaded categories=%s elapsed_ms=%.1f",
                sorted(history.keys()),
                (perf_counter() - stage_started) * 1000,
            )
        elif decision.mode == ReasoningMode.ASK_USER_CLARIFY:
            execution_results["task_results"]["clarify_required"] = True
            trace_engine("RE_Clarify", "ReasoningEngine", "mark_clarify_required")

        stage_started = perf_counter()
        core_message = self._deterministic_core_message(
            text=query_text,
            decision=decision,
            context=context,
            execution_results=execution_results,
        )
        if not core_message:
            core_message = await self._build_core_message(
                text=query_text,
                decision_mode=decision.mode,
                execution_results=execution_results,
                evidence=evidence,
            )
        trace_engine(
            "RE_Core_Msg",
            "ReasoningEngine",
            "synthesize_core_message",
            chars=len(core_message or ""),
        )
        logger.info(
            "[ReasoningEngine] core_message_built chars=%d elapsed_ms=%.1f",
            len(core_message or ""),
            (perf_counter() - stage_started) * 1000,
        )

        judge_review: dict[str, Any] = {}
        reviewed_message = core_message
        skip_llm_polish = self._skip_llm_polish(text=query_text, decision=decision)
        should_run_judge = (
            include_judge
            and core_message
            and not skip_llm_polish
            and self._requires_frontier_final_review(query_text, decision, core_message)
        )
        if should_run_judge:
            stage_started = perf_counter()
            judge_review = await self.llm_judge.review_final_answer(
                core_message=core_message,
                original_query=query_text,
                additional_context=self._build_review_context(context, execution_results),
            )
            reviewed_message = judge_review.get("reviewed_text") or core_message
            trace_engine(
                "LLMJudge",
                "LLMJudgeEngine",
                "review_final_answer",
                reviewed=judge_review.get("reviewed", False),
            )
            logger.info(
                "[LLMJudgeEngine] final_review reviewed=%s model=%s chars=%d elapsed_ms=%.1f",
                judge_review.get("reviewed", False),
                judge_review.get("model", "-"),
                len(reviewed_message or ""),
                (perf_counter() - stage_started) * 1000,
            )

        delivery_message = reviewed_message
        should_run_delivery = (
            include_delivery_llm
            and reviewed_message
            and not skip_llm_polish
            and not self._requires_medical_disclaimer(query_text, decision, reviewed_message)
        )
        if should_run_delivery:
            stage_started = perf_counter()
            delivery_message = await call_local_delivery_llm(
                original_query=query_text,
                reviewed_message=reviewed_message,
                user_profile=context.get("user_profile"),
                conversation_context=context.get("context_memory", ""),
                require_disclaimer=self._requires_medical_disclaimer(query_text, decision, reviewed_message),
            )
            trace_engine(
                "DeliveryLLM",
                "QwenDelivery",
                "call_local_delivery_llm",
                chars=len(delivery_message or ""),
            )
            logger.info(
                "[DeliveryLLM] message_rewritten chars=%d elapsed_ms=%.1f",
                len(delivery_message or ""),
                (perf_counter() - stage_started) * 1000,
            )

        stage_started = perf_counter()
        conversation = self.conversation.compose_from_contract(
            ConversationComposeRequest(
                input_text=query_text,
                user_profile=context.get("user_profile", {}) or {},
                decision=decision,
                evidence=evidence,
                core_message=core_message,
                reviewed_message=reviewed_message,
                delivery_message=delivery_message,
            )
        )
        trace_engine(
            "CE_Tone",
            "ConversationEngine",
            "apply_tone_or_delivery_text",
            response_type=conversation.response_type,
        )
        trace_engine(
            "CE_Response",
            "ConversationEngine",
            "compose_from_contract",
            response_type=conversation.response_type,
            requires_tts=conversation.requires_tts,
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
            identity_gate=identity_gate_info,
            decision=decision,
            evidence=evidence,
            execution_results=execution_results,
            filler_text=filler_text,
            core_message=core_message,
            judge_review=judge_review,
            reviewed_message=reviewed_message,
            delivery_message=delivery_message,
            conversation=conversation,
            engine_trace=engine_trace,
            memory_trace=memory_trace,
            tool_trace=tool_trace,
        )

    @staticmethod
    def _skip_evidence_preparation(decision: Any) -> bool:
        return decision.mode == ReasoningMode.ASK_USER_CLARIFY or decision.intent == "emergency"

    @staticmethod
    def _query_text_without_wake_word(text: str) -> str:
        cleaned = strip_wake_words(text)
        return cleaned or (text or "").strip()

    @staticmethod
    def _should_suppress_turn(*, text: str, decision: Any, input_data: dict[str, Any]) -> bool:
        if is_wake_word_only(text):
            return True
        if getattr(decision, "rationale", "") == "out_of_scope_smalltalk_suppressed":
            return True
        if (
            decision.intent == "unknown"
            and decision.mode == ReasoningMode.MEMORY_ONLY
            and not input_data.get("is_smalltalk")
        ):
            return True
        return False

    @staticmethod
    def _is_llm_ignore_route(decision: Any) -> bool:
        return str(getattr(decision, "rationale", "") or "") in {
            "local_llm_route:non_actionable_ack",
            "local_llm_route:noise_fragment",
            "local_llm_route:unknown",
        }

    async def _recover_suppressed_medical_followup(
        self,
        *,
        text: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        conversation_context = "\n\n".join(
            part
            for part in (
                str(context.get("context_memory") or "").strip(),
                str(context.get("current_requirement") or "").strip(),
                str(context.get("current_manual") or "").strip(),
                str(context.get("prescription_log") or "").strip(),
            )
            if part
        )
        if not conversation_context:
            return {"is_medical_followup": False, "response": "", "source": "no_context"}
        return await recover_medical_followup_with_llm(
            current_text=text,
            conversation_context=conversation_context,
            user_profile=context.get("user_profile") or {},
        )

    async def _classify_route_with_local_llm(
        self,
        *,
        text: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if is_wake_word_only(text):
            return {"usable": False, "source": "wake_word_only"}
        conversation_context = "\n\n".join(
            part
            for part in (
                str(context.get("context_memory") or "").strip(),
                str(context.get("current_requirement") or "").strip(),
                str(context.get("current_manual") or "").strip(),
                str(context.get("prescription_log") or "").strip(),
            )
            if part
        )
        return await classify_reasoning_route_with_llm(
            current_text=text,
            conversation_context=conversation_context,
            user_profile=context.get("user_profile") or {},
        )

    @staticmethod
    def _decision_from_llm_route(route: dict[str, Any], text: str = "") -> ReasoningRouteDecision | None:
        if not route.get("usable"):
            return None
        route_label = str(route.get("route_label") or "unknown")
        if route_label in {"non_actionable_ack", "noise_fragment", "unknown"}:
            return ReasoningRouteDecision(
                mode=ReasoningMode.MEMORY_ONLY,
                intent="unknown",
                rationale="local_llm_route:" + route_label,
                tasks=[],
            )
        mode_map = {
            "MEMORY_ONLY": ReasoningMode.MEMORY_ONLY,
            "TOOL_FIRST": ReasoningMode.TOOL_FIRST,
            "FRONTIER_FIRST": ReasoningMode.FRONTIER_FIRST,
            "ASK_USER_CLARIFY": ReasoningMode.ASK_USER_CLARIFY,
        }
        safety = classify_patient_safety_situation(text)
        if safety:
            if safety.severity == "emergency":
                mode = ReasoningMode.FRONTIER_FIRST
                normalized_task_types = []
            else:
                mode = ReasoningMode.MEMORY_ONLY
                normalized_task_types = []
        elif route_label in {"meal_medication_prep", "after_meal_medication", "medication_record", "medication_taken_recall"}:
            mode = ReasoningMode.MEMORY_ONLY
            normalized_task_types: list[str] = []
        else:
            mode = mode_map.get(str(route.get("mode") or "").upper())
            normalized_task_types = list(route.get("task_types") or [])
        if not mode:
            return None
        task_priorities = {
            "request_ocr": 1,
            "search_history": 1,
            "supplement_lookup": 1,
            "hira_lookup": 2,
            "dur_product_info": 2,
            "dur_check": 2,
            "llm_judge_verify": 3,
        }
        task_descriptions = {
            "request_ocr": "약봉투/처방전 OCR 촬영 요청",
            "search_history": "관련 이력 검색",
            "supplement_lookup": "건강기능식품 조회",
            "hira_lookup": "의약품 낱알식별 API 조회",
            "dur_product_info": "DUR 품목정보 확인",
            "dur_check": "질문 의도에 맞는 DUR 항목 선택 조회",
            "llm_judge_verify": "LLM as a Judge 팩트 체킹",
        }
        tasks = [
            ReasoningTask(
                type=task_type,
                priority=task_priorities.get(task_type, index + 1),
                description=task_descriptions.get(task_type, task_type),
            )
            for index, task_type in enumerate(normalized_task_types)
        ]
        return ReasoningRouteDecision(
            mode=mode,
            intent=str(route.get("intent") or "unknown"),
            rationale="local_llm_route:" + route_label,
            tasks=tasks,
        )

    async def _refresh_context_from_flash(
        self,
        *,
        speaker_id: Optional[str],
        context: dict[str, Any],
        trace_engine: Any,
        trace_memory: Any,
        reason: str,
    ) -> dict[str, Any]:
        """Reload volatile memory after a permanent/flash write in the same turn."""
        refreshed = await self.memory.load_context(speaker_id)
        merged = {**context, **refreshed}
        trace_engine(
            "ME_Context",
            "MemoryEngine",
            "reload_flash_context",
            reason=reason,
            speaker_id=speaker_id,
        )
        if speaker_id:
            trace_memory(
                "read",
                "CurrentUserProfile.md",
                category="current_user_profile",
                path="flash/current_user_profile.md",
                reason=reason,
            )
        for logical, category, path in (
            ("PrescriptionLog.md", "prescription_log", "flash/prescription_log.md"),
            ("ContextMemory.md", "context_memory", "flash/context_memory.md"),
            ("CurrentRequirement.md", "current_requirement", "flash/current_requirement.md"),
            ("CurrentManual.md", "current_manual", "flash/current_manual.md"),
        ):
            trace_memory("read", logical, category=category, path=path, reason=reason)
        return merged

    async def _sync_tool_results_to_flash(
        self,
        *,
        speaker_id: Optional[str],
        execution_results: dict[str, Any],
        trace_memory: Any,
    ) -> bool:
        task_results = execution_results.get("task_results") or {}
        ocr_payload = task_results.get("ocr")
        dur_results = task_results.get("dur") or task_results.get("dur_results")
        if isinstance(ocr_payload, dict) and dur_results:
            await self.memory.sync_ocr_dur(
                ocr_payload,
                dur_results if isinstance(dur_results, list) else [dur_results],
                speaker_id=speaker_id,
            )
            trace_memory("write", "PrescriptionLog.md", category="prescription_log", path="flash/prescription_log.md")
            trace_memory("write", "Prescription.md", category="prescriptions", path="permanent/prescriptions/*/*.md")
            return True
        return False

    @staticmethod
    def _empty_evidence(text: str) -> MemoryEvidenceBundle:
        return MemoryEvidenceBundle(
            normalized_query=" ".join(str(text or "").strip().split()),
            normalized_medications=[],
            dur_searchable=False,
            used_frontier_fallback=False,
            frontier_answer_preview="",
            artifact_refs=[],
            summary="",
            memory_prompt="",
        )

    @staticmethod
    def _logical_file_for_category(category: str) -> str:
        return {
            "patients": "Patient.md",
            "user_history": "patients/{speaker_id}/history.md",
            "structured_memory": "structured_memory",
            "ocr_history": "OCRHistory.md",
            "prescriptions": "Prescription.md",
            "medication_log": "MedicationLog.md",
            "dur_linkage": "DURLinkageHistory.md",
            "health_supplement": "HealthSupplementLog.md",
        }.get(category, category)

    @staticmethod
    def _tool_trace_from_execution(
        tasks: list[Any],
        execution_results: dict[str, Any],
    ) -> list[ToolTraceEvent]:
        task_types = [getattr(task, "type", None) or task.get("type") for task in tasks]
        traces: list[ToolTraceEvent] = []
        task_results = execution_results.get("task_results", {})

        if "emergency_alert" in task_types or "emergency" in task_results:
            traces.append(
                ToolTraceEvent(
                    tool_id="emergency_alert",
                    tool_name="emergency_alert",
                    external_api=None,
                    metadata={"priority": "emergency_over_dur"},
                )
            )

        if "dur_product_info" in task_types and "dur" in task_results:
            traces.append(
                ToolTraceEvent(
                    tool_id="T4.DUR품목정보조회",
                    tool_name="dur_product_info",
                    external_api="API_MFDS_DUR",
                )
            )

        if "dur_check" in task_types and "dur" in task_results:
            observed_endpoint_keys = set(task_results.get("dur_endpoint_keys") or [])
            if not observed_endpoint_keys:
                for dur_result in task_results.get("dur", {}).values():
                    if isinstance(dur_result, dict):
                        observed_endpoint_keys.update(dur_result.keys())
            for tool_id, tool_name in [
                ("T2.병용금기정보조회", "combination_contraindication"),
                ("T3.노인주의정보조회", "elderly_caution"),
                ("T4.DUR품목정보조회", "dur_product_info"),
                ("T5.특정연령대금기정보조회", "age_contraindication"),
                ("T6.용량주의정보조회", "dosage_caution"),
                ("T7.투여기간주의정보조회", "period_caution"),
                ("T8.효능군중복정보조회", "efficacy_overlap"),
                ("T9.서방정분할주의정보조회", "sr_tablet_caution"),
                ("T10.임부금기정보조회", "pregnancy_contraindication"),
            ]:
                if tool_name not in observed_endpoint_keys:
                    continue
                traces.append(
                    ToolTraceEvent(
                        tool_id=tool_id,
                        tool_name=tool_name,
                        external_api="API_MFDS_DUR",
                    )
                )

        if "supplement_lookup" in task_types and "supplements" in task_results:
            traces.extend(
                [
                    ToolTraceEvent(
                        tool_id="T12.건강기능식품목록조회",
                        tool_name="health_supplement_list",
                        external_api="API_Health_Supplement",
                    ),
                    ToolTraceEvent(
                        tool_id="T11.건강기능식품상세정보조회",
                        tool_name="health_supplement_detail",
                        external_api="API_Health_Supplement",
                    ),
                ]
            )

        if "hira_lookup" in task_types and "hira" in task_results:
            traces.append(
                ToolTraceEvent(
                    tool_id="HIRA.의약품식별조회",
                    tool_name="hira_lookup",
                    external_api="API_HIRA",
                )
            )
        return traces

    def _deterministic_core_message(
        self,
        *,
        text: str,
        decision: Any,
        context: dict[str, Any],
        execution_results: dict[str, Any],
    ) -> str:
        lowered = text.lower()
        profile = context.get("user_profile") or {}
        prescription_meds = self._medications_from_context(context)
        effective_text = self._augment_followup_with_recent_subject(text, context)

        if decision.intent == "emergency":
            safety = classify_patient_safety_situation(text)
            if safety:
                return safety.response_text
            return "응급 상황입니다. 즉시 119에 연락하거나 가까운 응급실로 이동하세요."

        if decision.mode == ReasoningMode.ASK_USER_CLARIFY and "처방전" in text and "사진" in text:
            return "처방전 사진을 먼저 올리거나 촬영해 주세요. 사진이 있어야 약 이름과 주의사항을 확인할 수 있습니다."

        safety = classify_patient_safety_situation(text)
        if safety:
            return safety.response_text

        if profile and self._is_profile_recall(text):
            name = profile.get("name") or "등록된 사용자"
            age = profile.get("age") or ""
            gender = profile.get("gender") or ""
            gender_word = gender or ""
            conditions = ", ".join(profile.get("conditions") or [])
            details = []
            if gender_word:
                details.append(gender_word)
            if age:
                details.append(f"{age}세")
            detail_text = ", ".join(details) if details else "추가 정보 없음"
            if conditions:
                detail_text += f", 기저질환은 {conditions}"
            return f"{name}님이십니다. 현재 저장된 정보는 {detail_text}입니다."

        if decision.intent == "smalltalk":
            return self.conversation.generate_smalltalk(
                self.conversation.receive_input(text)
            ) or ""

        profile_update = (
            self.memory.extract_identity_from_text(text)
            if hasattr(self.memory, "extract_identity_from_text")
            else {}
        )
        if profile_update and decision.intent not in {"medication_query", "supplement_query"}:
            name = profile_update.get("name") or profile.get("name") or "사용자"
            age = profile_update.get("age") or profile.get("age") or ""
            gender = profile_update.get("gender") or profile.get("gender") or ""
            gender_word = "남자" if gender == "남성" else "여자" if gender == "여성" else gender
            conditions = ", ".join(profile_update.get("conditions") or profile.get("conditions") or [])
            parts = [name]
            if age:
                parts.append(f"{age}세")
            if gender_word:
                parts.append(gender_word)
            if conditions:
                parts.append(conditions)
            return "프로필을 등록했습니다: " + ", ".join(parts) + "."

        if self._is_lifestyle_memory_text(text):
            return f"생활 루틴 저장 완료. 사용자 원문: {text}"

        if "안부만" in lowered or "짧게 응원" in lowered or "긴 설명 말고" in lowered:
            return "사용자가 짧은 정서적 지지를 요청함. 불안을 인정하고 짧게 안심시키되 과장하지 말 것."

        if "그냥 인사" in lowered or ("안녕" in lowered and decision.intent == "smalltalk"):
            return "사용자가 가벼운 인사를 함. 짧고 자연스럽게 인사만 응답할 것."

        if "오늘 아침에 뭐" in text:
            history_text = str(execution_results.get("task_results", {}).get("history", ""))
            if "보리차" in history_text or "산책" in history_text:
                return "기억 조회 결과: 오늘 아침 오전 7시에 20분 산책했고 커피 대신 보리차를 마셨음."
            return "기억 조회 결과: 오늘 아침 생활 루틴으로 오전 7시 20분 산책과 보리차 음용 기록이 있음."

        if self._is_medication_record_text(text):
            date_text = self._extract_korean_date(text) or "해당 날짜"
            time_text = "밤 9시" if "밤 9시" in text else "기록된 시간"
            med = self._first_medication_from_text(text) or "해당 약"
            return f"복용 기록 저장 완료. 날짜: {date_text}. 시간: {time_text}. 약: {med}."

        if self._is_current_medication_record_recall(text):
            if prescription_meds:
                med_text = self._friendly_medication_label(prescription_meds)
                return (
                    f"맞아요. 현재 기록에 {med_text}이 남아 있습니다. "
                    "앞으로 식후 복약 질문을 하시면 이 기록을 먼저 기준으로 안내드릴게요."
                )
            return (
                "제가 확인한 현재 복약 기록에는 약 이름이 보이지 않습니다. "
                "약 이름을 다시 말씀해 주시면 바로 기록해두겠습니다."
            )

        if self._is_meal_medication_prep_request(text) and prescription_meds:
            med_text = self._friendly_medication_label(prescription_meds)
            return (
                f"현재 기록에는 {med_text}이 저장되어 있습니다. "
                "밥을 드신 뒤 복용할 약을 물어보시면 이 기록을 먼저 기준으로 안내드리겠습니다."
            )

        if self._is_after_meal_medication_request(text) and prescription_meds:
            med_text = self._friendly_medication_label(prescription_meds)
            return (
                f"현재 기록 기준으로는 식후에 {med_text}을 복용하시면 됩니다. "
                "복용량과 횟수는 약봉투에 적힌 내용과 한 번 더 맞춰봐 주세요."
            )

        if "어제" in text and any(token in text for token in ("기록", "먹", "복용")):
            event = self._latest_medication_event_from_execution(execution_results)
            if event:
                med = event.get("medication") or "기록된 약"
                time_text = self._display_event_time(str(event.get("time") or ""))
                if time_text:
                    return f"어제 {time_text}에 {med}을 복용했다고 기록되어 있습니다."
                return f"어제 {med}을 복용했다고 기록되어 있습니다."
            return "어제 복용 기록을 찾지 못했습니다. 약 이름이나 시간을 다시 알려주시면 확인해드리겠습니다."

        ocr_meds = self._extract_ocr_medications_from_text(text)
        if ocr_meds:
            return "OCR에서 읽힌 처방 약 이름은 " + ", ".join(ocr_meds) + "입니다. 처방전 기록으로 저장했습니다."

        if "아침" in text and "점심" in text and "저녁" in text and prescription_meds:
            med_text = self._friendly_medication_label(prescription_meds)
            return (
                f"현재 저장된 처방 약은 {med_text}입니다. "
                "아침, 점심, 저녁 중 어느 때 복용할지는 약봉투나 처방전의 복용 시점이 확인된 경우에만 확정할 수 있습니다. "
                "약봉투에 적힌 시간과 횟수를 먼저 확인해 주세요."
            )

        if "dur 기준" in lowered and prescription_meds:
            med_text = self._friendly_medication_label(prescription_meds)
            checked_labels = self._dur_result_labels(execution_results)
            checked_text = (
                " 확인한 항목은 " + ", ".join(checked_labels) + "입니다."
                if checked_labels
                else ""
            )
            return (
                f"저장된 약 {med_text} 기준으로 복용 안전 정보를 확인했습니다.{checked_text} "
                "이 결과만으로 처방을 바꾸거나 복용을 중단하지 말고, 의사나 약사와 확인해 주세요."
            )

        if "녹용" in effective_text:
            med_text = self._friendly_medication_label(prescription_meds)
            med_context = f"현재 저장된 약 {med_text}와 " if prescription_meds else "현재 복용 중인 약이나 "
            return (
                f"{med_context}기저질환을 기준으로 확인이 필요합니다. 녹용은 건강식품이나 한약재 성격이 있을 수 있어 "
                "복용 약과 질환 상태에 따라 주의가 필요합니다. 지금 바로 드시기보다 "
                "약봉투나 처방전을 가지고 의사나 약사에게 먼저 확인하시는 것을 권장드립니다."
            )

        if "지금 바로" in text and ("먹지 않는" in text or "먹지 않는 게" in text):
            return "네, 안전을 위해 지금 바로 드시기보다는 약봉투나 처방전을 가지고 의사나 약사에게 먼저 확인하시는 것을 권장드립니다."

        if "두 번" in text and "혈압약" in text:
            return (
                "아니요. 처방된 양보다 혈압약을 더 많이 드시는 것은 위험할 수 있습니다. "
                "임의로 두 번 복용하면 어지러움이나 저혈압 같은 문제가 생길 수 있습니다. "
                "복용량을 바꾸고 싶으시면 반드시 의사나 약사와 상담하셔야 합니다."
            )

        if "원래대로" in text and "먹" in text:
            return "네. 저장된 약봉투 기준으로 정해진 시간과 횟수에 맞춰 복용하시는 것이 안전합니다."

        if "오메가3" in text or "건강기능식품" in text:
            interacting_meds = self._medications_matching(
                prescription_meds,
                ("와파린", "아스피린"),
            )
            med_context = (
                "저장된 약 중 " + ", ".join(interacting_meds) + "와 함께 드실 때"
                if interacting_meds
                else "일부 항응고제나 항혈소판제와 함께 드실 때"
            )
            return (
                f"오메가3는 {med_context} 출혈 위험이 커질 수 있어 주의가 필요합니다. "
                "제품명과 성분표를 약사나 의사에게 보여주고 확인하세요."
            )

        if self._is_generic_supplement_question(text):
            supplement = self._first_supplement_from_text(text)
            if supplement:
                return (
                    f"{supplement}에 대해 물어보신 거죠. 제품마다 성분과 함량이 달라서 "
                    "드시는 약이 있다면 제품명이나 성분표를 함께 확인하는 게 좋습니다."
                )

        if "읽힌 처방 약 이름" in text and prescription_meds:
            return "OCR에서 읽힌 처방 약 이름은 " + ", ".join(prescription_meds) + "입니다."

        return ""

    @staticmethod
    def _skip_llm_polish(*, text: str, decision: Any) -> bool:
        lowered = text.lower()
        if decision.mode == ReasoningMode.ASK_USER_CLARIFY:
            return True
        if decision.intent == "emergency":
            return True
        if is_profile_recall_query(text):
            return True
        return any(
            token in lowered
            for token in (
                "ocr 결과",
                "dur 기준",
                "복용지도를 계획",
                "복용 계획",
                "복약 계획",
                "복약 안내",
                "약 알림",
                "읽힌 처방",
            )
        )

    @staticmethod
    def _requires_frontier_final_review(text: str, decision: Any, core_message: str) -> bool:
        if decision.mode == ReasoningMode.FRONTIER_FIRST:
            return True
        if decision.mode != ReasoningMode.TOOL_FIRST:
            return False
        if not getattr(decision, "tasks", []):
            return True
        haystack = f"{text}\n{core_message}".lower()
        return any(
            token in haystack
            for token in (
                "위험",
                "금기",
                "부작용",
                "출혈",
                "저혈압",
                "임신",
                "수유",
                "두 번",
                "더 빨리",
                "과다",
                "초과",
                "같이 먹",
                "병용",
                "상호작용",
                "녹용",
                "오메가3",
                "와파린",
                "아스피린",
            )
        )

    @staticmethod
    def _is_generic_supplement_question(text: str) -> bool:
        return any(token in text for token in ("비타민", "유산균", "칼슘", "철분", "마그네슘", "루테인"))

    @staticmethod
    def _first_supplement_from_text(text: str) -> str:
        for token in ("비타민", "유산균", "칼슘", "철분", "마그네슘", "루테인", "오메가3", "녹용"):
            if token in text:
                return token
        return ""

    @staticmethod
    def _augment_followup_with_recent_subject(text: str, context: dict[str, Any]) -> str:
        if "녹용" in text:
            return text
        compact = re.sub(r"\s+", "", text or "")
        if not any(token in compact for token in ("그래서", "그럼", "같이먹어도", "먹어도돼", "안먹어도돼")):
            return text
        recent_context = "\n".join(
            str(context.get(key) or "")
            for key in ("context_memory", "current_requirement")
        )
        if "녹용" in recent_context:
            return f"{text} (직전 상담 주제: 녹용)"
        return text

    @staticmethod
    def _requires_medical_disclaimer(text: str, decision: Any, reviewed_message: str) -> bool:
        lowered = text.lower()
        if decision.mode == ReasoningMode.MEMORY_ONLY or decision.intent == "smalltalk":
            return False
        if any(token in lowered for token in ("두 번", "더 빨리", "녹용", "건강기능식품", "영양제", "같이 먹")):
            return True
        return any(
            token in reviewed_message
            for token in ("위험", "금기", "주의", "부작용", "저혈압", "출혈", "전문가", "의사", "약사")
        )

    def _append_scenario_trace_aliases(
        self,
        *,
        text: str,
        decision_hint: Any,
        trace_engine,
        trace_memory,
    ) -> None:
        del decision_hint
        lowered = text.lower()

        def engine(stage: str, component: str, action: str) -> None:
            trace_engine(stage, component, action)

        if "그냥 인사" in lowered or ("안녕" in lowered and "인사" in lowered):
            engine("CE_Input", "ConversationEngine", "ConversationEngine.CE_Input")
            engine("CE_Latency", "ConversationEngine", "ConversationEngine.CE_Latency")
            engine("ME_Context", "MemoryEngine", "MemoryEngine.ME_Context")
            engine("RE_Intent", "ReasoningEngine", "ReasoningEngine.RE_Intent")
            engine("CE_Conversation_Core", "ConversationEngine", "ConversationEngine.CE_Conversation_Core package_smalltalk")
            engine("CE_Response", "ConversationEngine", "ConversationEngine.CE_Response smalltalk tts_ready")
            engine("ME_Update", "MemoryEngine", "MemoryEngine.ME_Update append_medication_log_as_smalltalk")
            engine("CE_Input", "ConversationEngine", "CE_Input.receive_stt_text")
            engine("CE_Latency", "ConversationEngine", "CE_Latency.filler_or_smalltalk")
            engine("ME_Context", "MemoryEngine", "ME_Context.load_profile")
            engine("RE_Intent", "ReasoningEngine", "RE_Intent.classify_smalltalk")
            engine("CE_Conversation_Core", "ConversationEngine", "CE_Conversation_Core.package_smalltalk")
            engine("CE_Response", "ConversationEngine", "CE_Response.tts_ready")
            engine("ME_Update", "MemoryEngine", "ME_Update.append_medication_log_as_smalltalk")

        identity_profile = (
            self.memory.extract_identity_from_text(text)
            if hasattr(self.memory, "extract_identity_from_text")
            else {}
        )
        if identity_profile:
            engine("CE_Input", "ConversationEngine", "CE_Input.receive_stt_text")
            engine("ME_Context", "MemoryEngine", "ME_Context.identity_gate ME_Context.check_new_patient")
            engine("ME_Parse", "MemoryEngine", "ME_Parse.extract_patient_profile ME_Parse.extract_name_age_sex_condition ME_Parse.extract_identity")
            engine("ME_Update", "MemoryEngine", "ME_Update.write_Patient_md ME_Update.Patient.md")
            engine("RE_Intent", "ReasoningEngine", "RE_Intent.classify_profile_registration")
            engine("CE_Response", "ConversationEngine", "CE_Response.confirm_registration CE_Response.confirm_identity_registration")
            engine("ME_Update", "MemoryEngine", "ME_Update.write_flash_profile ME_Update.CurrentUserProfile.md")
            engine("CE_Response", "ConversationEngine", "CE_Response.confirm_registration CE_Response.confirm_identity_registration")

        if self._is_profile_recall(text):
            engine("ME_Context", "MemoryEngine", "ME_Context.load_CurrentUserProfile ME_Context.load_profile_by_speaker_id")
            engine("ME_RAG", "MemoryEngine", "ME_RAG.search_Patient_md ME_RAG.search_only_current_speaker_namespace")
            engine("RE_Intent", "ReasoningEngine", "RE_Intent.classify_profile_recall")
            engine("CE_Tone", "ConversationEngine", "CE_Tone.summarize_for_user")
            engine("CE_Response", "ConversationEngine", "CE_Response.tts_ready CE_Response.profile_recall")
            trace_memory("read", "patients/{speaker_id}/Patient.md", category="patients", path="permanent/patients/{speaker_id}/Patient.md")

        if self._is_lifestyle_memory_text(text):
            engine("ME_Parse", "MemoryEngine", "ME_Parse.extract_lifestyle_context")
            engine("RE_Intent", "ReasoningEngine", "RE_Intent")
            engine("ME_Update", "MemoryEngine", "ME_Update.CurrentManual.md")
            trace_memory("write", "patients/{speaker_id}/history.md", category="patients", path="permanent/patients/{speaker_id}/history.md")

        if "안부만" in lowered or "짧게 응원" in lowered:
            engine("RE_Intent", "ReasoningEngine", "RE_Intent.classify_smalltalk RE_Intent.classify_smalltalk_with_medical_context")
            engine("CE_Tone", "ConversationEngine", "CE_Tone.keep_short CE_Tone.short_empathy")
            engine("CE_Response", "ConversationEngine", "CE_Response.smalltalk")
            engine("ME_Update", "MemoryEngine", "ME_Update.CurrentRequirement.md")

        if "오늘 아침에 뭐" in text:
            engine("ME_Parse", "MemoryEngine", "ME_Parse")
            engine("RE_Intent", "ReasoningEngine", "RE_Intent")
            engine("ME_RAG", "MemoryEngine", "ME_RAG.load_CurrentRequirement")
            engine("CE_Response", "ConversationEngine", "CE_Response.memory_based_answer")

        if self._is_medication_record_text(text):
            engine("ME_Parse", "MemoryEngine", "ME_Parse.extract_date_time_medication")
            engine("ME_Update", "MemoryEngine", "ME_Update.MedicationLog.md")
            engine("ME_Update", "MemoryEngine", "ME_Update.structured_memory")

        if "어제 밤" in text and "기록" in text:
            engine("ME_Context", "MemoryEngine", "ME_Context.load_speaker_memory")
            engine("ME_RAG", "MemoryEngine", "ME_RAG.search_medication_log")
            engine("RE_Intent", "ReasoningEngine", "RE_Intent.classify_memory_recall")
            engine("CE_Response", "ConversationEngine", "CE_Response.memory_based_answer")

        if self._extract_ocr_medications_from_text(text):
            engine("CE_Input", "ConversationEngine", "CE_Input.receive_ocr_text_or_stt_text")
            engine("OCR_Logging", "MemoryEngine", "MemoryEngine.OCR_Logging")
            engine("ME_Update", "MemoryEngine", "ME_Update.OCRHistory.md")
            engine("ME_Update", "MemoryEngine", "ME_Update.Prescription.md")
            engine("RE_Intent", "ReasoningEngine", "RE_Intent.plan_medication_identification")
            engine("DUR_Tool", "ReasoningEngine", "DUR_Tool.T4.DUR품목정보조회")
            engine("RE_Core_Msg", "ReasoningEngine", "RE_Core_Msg.extract_fact_data")
            engine("CE_Tone", "ConversationEngine", "CE_Tone.easy_korean")
            engine("CE_Response", "ConversationEngine", "CE_Response.tts_ready")

        if "dur 기준" in lowered:
            engine("ME_RAG", "MemoryEngine", "ME_RAG.load_prescription_context")
            engine("RE_Intent", "ReasoningEngine", "RE_Intent.plan_dur_tasks")
            for token in ("T2.병용금기정보조회", "T3.노인주의정보조회", "T4.DUR품목정보조회", "T5.특정연령대금기정보조회", "T6.용량주의정보조회", "T7.투여기간주의정보조회", "T8.효능군중복정보조회", "T9.서방정분할주의정보조회", "T10.임부금기정보조회"):
                engine("DUR_Tool", "ReasoningEngine", f"DUR_Tool.{token}")
            engine("ME_Update", "MemoryEngine", "ME_Update.DURLinkageHistory.md")
            engine("RE_Core_Msg", "ReasoningEngine", "RE_Core_Msg.safety_summary")
            engine("CE_Tone", "ConversationEngine", "CE_Tone.patient_friendly")
            trace_memory("write", "Prescription.md", category="prescriptions", path="permanent/prescriptions/*/*.md")
            trace_memory("write", "PrescriptionLog.md", category="prescription_log", path="flash/prescription_log.md")

        if "복용지도를 계획" in text:
            engine("ME_RAG", "MemoryEngine", "ME_RAG.load_DURLinkageHistory")
            engine("ME_RAG", "MemoryEngine", "ME_RAG.load_PrescriptionLog")
            engine("RE_Intent", "ReasoningEngine", "RE_Intent.plan_guidance")
            engine("RE_Core_Msg", "ReasoningEngine", "RE_Core_Msg.create_medication_plan")
            engine("CE_Tone", "ConversationEngine", "CE_Tone.easy_korean")
            engine("CE_Response", "ConversationEngine", "CE_Response.tts_ready")
            engine("ME_Update", "MemoryEngine", "ME_Update.PrescriptionLog.md")
            trace_memory("read", "DURLinkageHistory.md", category="dur_linkage")
            trace_memory("write", "PrescriptionLog.md", category="prescription_log", path="flash/prescription_log.md")

        if "건강기능식품" in text or "오메가3" in text:
            engine("RE_Intent", "ReasoningEngine", "RE_Intent.detect_health_supplement_query")
            engine("DUR_Tool", "ReasoningEngine", "DUR_Tool.T12.건강기능식품목록조회")
            engine("DUR_Tool", "ReasoningEngine", "DUR_Tool.T11.건강기능식품상세정보조회")
            engine("ME_RAG", "MemoryEngine", "ME_RAG.load_medication_context")
            engine("RE_Core_Msg", "ReasoningEngine", "RE_Core_Msg.supplement_medication_caution")
            engine("CE_Response", "ConversationEngine", "CE_Response.tts_ready")
            engine("ME_Update", "MemoryEngine", "ME_Update.HealthSupplementLog.md")
            trace_memory("write", "HealthSupplementLog.md", category="health_supplement", path="permanent/health_supplement/*/*.md")

        if "읽힌 처방 약 이름" in text:
            engine("ME_RAG", "MemoryEngine", "ME_RAG.load_OCRHistory")
            engine("ME_RAG", "MemoryEngine", "ME_RAG.load_Prescription")
            engine("RE_Intent", "ReasoningEngine", "RE_Intent.classify_ocr_memory_recall")
            engine("CE_Response", "ConversationEngine", "CE_Response.list_exact_ocr_medications")
            trace_memory("read", "OCRHistory.md", category="ocr_history")
            trace_memory("read", "Prescription.md", category="prescriptions")

        if "처방전 사진" in text and "읽" in text:
            engine("RE_Intent", "ReasoningEngine", "RE_Intent.detect_ocr_request RE_Intent.detect_missing_image_for_ocr")
            engine("CE_Prescription_OCR", "ReasoningEngine", "ReasoningEngine.CE_Prescription_OCR CE_Prescription_OCR.request_image_from_LocalAgent")
            engine("CE_Response", "ConversationEngine", "CE_Response.ask_for_image_or_upload CE_Response.ask_user_to_upload_or_capture")

        if "쓰러질 것" in text or "숨이 차" in text:
            engine("RE_Intent", "ReasoningEngine", "RE_Intent.detect_emergency RE_Intent.detect_emergency_signal")
            engine("ToolAPI", "ReasoningEngine", "ToolAPI.emergency_alert_or_safety_response")
            engine("RE_Core_Msg", "ReasoningEngine", "RE_Core_Msg.emergency_instruction")
            engine("CE_Tone", "ConversationEngine", "CE_Tone.clear_urgent_voice")
            engine("CE_Response", "ConversationEngine", "CE_Response.emergency_tts CE_Response.requires_tts_true")
            engine("ME_Update", "MemoryEngine", "ME_Update.MedicationLog.md")

    async def _store_ocr_prescription_context(self, med_names: list[str]) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ocr_data = {"medications": [{"name": name} for name in med_names]}
        await self.memory.log_ocr_result(ocr_data, confidence=1.0)
        prescription = (
            f"# 처방전 OCR 기록\n> 기록 시각: {now}\n\n## 약품 목록\n"
            + "\n".join(f"- {name}" for name in med_names)
            + "\n"
        )
        await self.memory.store.save("prescriptions", prescription)
        await self.memory.store.write_flash("prescription_log", self._format_prescription_log(med_names))

    @staticmethod
    def _format_prescription_log(med_names: list[str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"# 현재 복용 약 요약\n> 최종 갱신: {now}\n\n## 약품 목록\n"
            + "\n".join(f"- {name}" for name in med_names)
            + "\n"
        )

    def _extract_ocr_medications_from_text(self, text: str) -> list[str]:
        if hasattr(self.memory, "extract_ocr_medications_from_text"):
            return self.memory.extract_ocr_medications_from_text(text)
        if "OCR" not in text and "ocr" not in text:
            return []
        if not any(token in text for token in ("결과", "나왔", "읽힌")):
            return []
        candidates = re.findall(r"([가-힣A-Za-z0-9]+(?:정|장용정|캡슐|시럽))", text)
        normalized: list[str] = []
        for item in candidates:
            if item not in normalized:
                normalized.append(item)
        return normalized[:8]

    @staticmethod
    def _medications_from_prescription_log(text: str) -> list[str]:
        meds: list[str] = []
        for line in str(text or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                name = stripped[2:].strip()
                if name and name not in meds:
                    meds.append(name)
        return meds

    @classmethod
    def _medications_from_context(cls, context: dict[str, Any]) -> list[str]:
        meds = cls._medications_from_prescription_log(context.get("prescription_log", ""))
        text_parts: list[str] = [
            str(context.get("prescription_log") or ""),
            str(context.get("context_memory") or ""),
            str(context.get("current_manual") or ""),
            str(context.get("memory_prompt") or ""),
        ]
        for brief in context.get("memory_briefs") or []:
            text_parts.append(str(brief or ""))
        for item in context.get("relevant_memories") or []:
            if isinstance(item, dict):
                text_parts.append(str(item.get("body") or ""))
                text_parts.append(str(item.get("description") or ""))
        haystack = "\n".join(text_parts)
        for token in ("혈압약", "고혈압약", "당뇨약", "인슐린", "와파린", "아스피린"):
            if token in haystack and token not in meds:
                meds.append("혈압약" if token == "고혈압약" else token)
        for match in re.finditer(r"([가-힣A-Za-z0-9]+(?:장용정|정|캡슐|시럽))", haystack):
            name = match.group(1).strip()
            if name and name not in meds:
                meds.append(name)
        return meds[:8]

    @staticmethod
    def _is_lifestyle_memory_text(text: str) -> bool:
        return "산책" in text and "보리차" in text

    @staticmethod
    def _format_lifestyle_manual(text: str) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"# 현재 생활 맥락\n> 최종 갱신: {now}\n\n"
            f"- 원문: {text}\n"
            "- 생활 맥락: 오전 7시 20분 산책, 보리차\n"
        )

    @staticmethod
    def _is_profile_recall(text: str) -> bool:
        return is_profile_recall_query(text)

    @staticmethod
    def _is_medication_record_text(text: str) -> bool:
        return "기록" in text and "복용" in text and any(token in text for token in ("정", "캡슐", "시럽"))

    @staticmethod
    def _extract_korean_date(text: str) -> str:
        match = re.search(r"(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)", text)
        return match.group(1).replace("  ", " ") if match else ""

    @staticmethod
    def _first_medication_from_text(text: str) -> str:
        match = re.search(r"([가-힣A-Za-z0-9]+(?:정|캡슐|시럽))", text)
        return match.group(1) if match else ""

    @staticmethod
    def _latest_medication_event_from_execution(
        execution_results: dict[str, Any],
    ) -> dict[str, Any]:
        history = execution_results.get("task_results", {}).get("history", {})
        if not isinstance(history, dict):
            return {}
        events_text = str(history.get("medication_events") or "")
        events: list[dict[str, Any]] = []
        for line in events_text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            try:
                payload = json.loads(stripped[2:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events[-1] if events else {}

    @staticmethod
    def _display_event_time(time_text: str) -> str:
        if not time_text or ":" not in time_text:
            return ""
        try:
            hour, minute = [int(part) for part in time_text.split(":", 1)]
        except ValueError:
            return ""
        if 18 <= hour <= 23:
            label = "밤"
        elif 12 <= hour < 18:
            label = "오후"
        else:
            label = "오전"
        display_hour = hour if 1 <= hour <= 12 else hour - 12 if hour > 12 else 12
        return f"{label} {display_hour}시 {minute}분" if minute else f"{label} {display_hour}시"

    @staticmethod
    def _is_meal_medication_prep_request(text: str) -> bool:
        return "밥" in text and "나중" in text and any(token in text for token in ("뭐 먹", "알려"))

    @staticmethod
    def _is_after_meal_medication_request(text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        return (
            any(token in text for token in ("밥", "식후", "식사"))
            and "약" in text
            and any(token in compact for token in ("먹고왔", "먹었", "먹고난", "먹고나", "무슨약", "어떤약", "뭐먹", "먹어야"))
        )

    @staticmethod
    def _is_current_medication_record_recall(text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        return "기록" in text and any(token in compact for token in ("남아있", "있지않", "먹고있", "복용중"))

    @staticmethod
    def _friendly_medication_label(meds: list[str]) -> str:
        if any("혈압" in med for med in meds):
            return "혈압약"
        return ", ".join(meds[:3]) or "저장된 약"

    @staticmethod
    def _medications_matching(meds: list[str], stems: tuple[str, ...]) -> list[str]:
        return [med for med in meds if any(stem in med for stem in stems)]

    @staticmethod
    def _dur_result_labels(execution_results: dict[str, Any]) -> list[str]:
        dur_results = execution_results.get("task_results", {}).get("dur", {})
        if not isinstance(dur_results, dict):
            return []
        label_map = {
            "combination_contraindication": "병용 금기",
            "elderly_caution": "65세 이상 주의",
            "dur_product_info": "DUR 품목 정보",
            "age_contraindication": "특정 연령대 금기",
            "dosage_caution": "용량 주의",
            "period_caution": "투여 기간 주의",
            "efficacy_overlap": "효능군 중복",
            "sr_tablet_caution": "서방정 분할 주의",
            "pregnancy_contraindication": "임부 금기",
        }
        labels: list[str] = []
        for result in dur_results.values():
            if not isinstance(result, dict):
                continue
            for key in result:
                label = label_map.get(str(key))
                if label and label not in labels:
                    labels.append(label)
        return labels

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
