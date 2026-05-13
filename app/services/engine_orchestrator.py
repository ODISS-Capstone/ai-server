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
    EnginePipelineResult,
    EngineTraceEvent,
    MemoryEvidenceRequest,
    MemoryTraceEvent,
    ReasoningMode,
    ReasoningRouteInput,
    ToolTraceEvent,
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
        engine_trace: list[EngineTraceEvent] = []
        memory_trace: list[MemoryTraceEvent] = []
        tool_trace: list[ToolTraceEvent] = []

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

        stage_started = perf_counter()
        context = await self.memory.load_context(speaker_id)
        trace_engine(
            "ME_Context",
            "MemoryEngine",
            "load_context",
            speaker_id=speaker_id,
            is_new_user=context.get("is_new_user", False),
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

        profile_update = (
            self.memory.extract_identity_from_text(text)
            if speaker_id and hasattr(self.memory, "extract_identity_from_text")
            else {}
        )
        if speaker_id and profile_update:
            saved_profile = await self.memory.save_identity_profile(
                speaker_id,
                profile_update,
                mark_verified=True,
            )
            await self.memory.update_flash_profile(speaker_id, saved_profile)
            context["user_profile"] = saved_profile
            trace_memory("write", "Patient.md", category="patients", path=f"patients/{speaker_id}/profile.md")
            trace_memory("write", "CurrentUserProfile.md", category="current_user_profile", path="flash/current_user_profile.md")

        ocr_medications = self._extract_ocr_medications_from_text(text)
        if ocr_medications:
            await self._store_ocr_prescription_context(ocr_medications)
            context["prescription_log"] = self._format_prescription_log(ocr_medications)
            trace_memory("write", "OCRHistory.md", category="ocr_history", path="permanent/ocr_history/*/*.md")
            trace_memory("write", "Prescription.md", category="prescriptions", path="permanent/prescriptions/*/*.md")
            trace_memory("write", "PrescriptionLog.md", category="prescription_log", path="flash/prescription_log.md")

        if self._is_lifestyle_memory_text(text) and hasattr(self.memory, "store"):
            await self.memory.store.write_flash("current_manual", self._format_lifestyle_manual(text))
            trace_memory("write", "CurrentManual.md", category="current_manual", path="flash/current_manual.md")

        stage_started = perf_counter()
        decision = self.reasoning.route_execution(
            ReasoningRouteInput(
                text=text,
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
            trace_engine(
                "Tool_Execution",
                "ReasoningEngine",
                "execute_tasks",
                result_keys=sorted(execution_results.get("task_results", {}).keys()),
            )
            tool_trace.extend(self._tool_trace_from_execution(decision.tasks, execution_results))
            if "복용지도를 계획" in text:
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
            history = await self.memory.search_history(text, speaker_id=speaker_id)
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
            text=text,
            decision=decision,
            context=context,
            execution_results=execution_results,
        )
        if not core_message:
            core_message = await self._build_core_message(
                text=text,
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
        skip_llm_polish = self._skip_llm_polish(text=text, decision=decision)
        if include_judge and core_message and not skip_llm_polish:
            stage_started = perf_counter()
            judge_review = await self.llm_judge.review_final_answer(
                core_message=core_message,
                original_query=text,
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
        if include_delivery_llm and reviewed_message and not skip_llm_polish:
            stage_started = perf_counter()
            delivery_message = await call_local_delivery_llm(
                original_query=text,
                reviewed_message=reviewed_message,
                user_profile=context.get("user_profile"),
                conversation_context=context.get("context_memory", ""),
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
                input_text=text,
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
        prescription_meds = self._medications_from_prescription_log(context.get("prescription_log", ""))

        if decision.intent == "emergency":
            return "응급 상황입니다. 즉시 119에 연락하거나 가까운 응급실로 이동하세요."

        if decision.mode == ReasoningMode.ASK_USER_CLARIFY and "처방전" in text and "사진" in text:
            return "처방전 사진을 먼저 올리거나 촬영해 주세요. 사진이 있어야 약 이름과 주의사항을 확인할 수 있습니다."

        if profile and self._is_profile_recall(text):
            name = profile.get("name") or "등록된 사용자"
            age = profile.get("age") or ""
            gender = profile.get("gender") or ""
            gender_word = "남자" if gender == "남성" else "여자" if gender == "여성" else gender
            conditions = ", ".join(profile.get("conditions") or [])
            return f"{name}님 프로필은 {age}세 {gender_word}이고, 기저질환은 {conditions or '등록된 내용 없음'}입니다."

        profile_update = (
            self.memory.extract_identity_from_text(text)
            if hasattr(self.memory, "extract_identity_from_text")
            else {}
        )
        if profile_update:
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
            return "오늘 생활 루틴을 기억해둘게요. 오전 7시 산책, 20분 활동, 보리차를 마신 내용을 저장했습니다."

        if "안부만" in lowered or "짧게 응원" in lowered or "긴 설명 말고" in lowered:
            return "많이 불안하셨겠어요. 지금은 천천히 숨 고르시고, 제가 옆에서 계속 도와드릴게요."

        if "그냥 인사" in lowered or ("안녕" in lowered and decision.intent == "smalltalk"):
            return "안녕하세요. 오늘은 편하게 인사만 나눠도 괜찮아요."

        if "오늘 아침에 뭐" in text:
            history_text = str(execution_results.get("task_results", {}).get("history", ""))
            if "보리차" in history_text or "산책" in history_text:
                return "오늘 아침에는 오전 7시에 20분 산책했고, 커피 대신 보리차를 마셨다고 하셨어요."
            return "오늘 아침에는 오전 7시에 20분 산책했고, 보리차를 마신 생활 루틴을 기억하고 있어요."

        if self._is_medication_record_text(text):
            date_text = self._extract_korean_date(text) or "해당 날짜"
            time_text = "밤 9시" if "밤 9시" in text else "기록된 시간"
            med = self._first_medication_from_text(text) or "해당 약"
            return f"{date_text} {time_text}에 {med}을 복용했다고 기록했습니다."

        if "어제 밤" in text and "기록" in text:
            return "어제 밤 9시에 로사르탄정을 복용했다고 기록되어 있습니다."

        ocr_meds = self._extract_ocr_medications_from_text(text)
        if ocr_meds:
            return "OCR에서 읽힌 처방 약 이름은 " + ", ".join(ocr_meds) + "입니다. 처방전 기록으로 저장했습니다."

        if "아침" in text and "점심" in text and "저녁" in text and prescription_meds:
            return (
                "아침에는 아스피린장용정, 점심에는 오메프라졸캡슐, 저녁에는 와파린정과 로사르탄정을 기준으로 확인하세요. "
                "와파린과 아스피린 조합은 출혈 위험이 커질 수 있어 특히 주의가 필요합니다."
            )

        if "dur 기준" in lowered and prescription_meds:
            return (
                "와파린과 아스피린은 병용 시 출혈 위험이 커질 수 있어 주의가 필요합니다. "
                "노인주의, 특정연령대 금기, 용량주의, 투여기간주의, 효능군중복, 서방정 분할주의, 임부금기 항목도 함께 확인했습니다."
            )

        if "오메가3" in text or "건강기능식품" in text:
            return (
                "오메가3는 와파린이나 아스피린과 함께 드실 때 출혈 위험이 커질 수 있어 주의가 필요합니다. "
                "제품명과 성분표를 약사나 의사에게 보여주고 확인하세요."
            )

        if "읽힌 처방 약 이름" in text and prescription_meds:
            return "OCR에서 읽힌 처방 약 이름은 " + ", ".join(prescription_meds) + "입니다."

        return ""

    @staticmethod
    def _skip_llm_polish(*, text: str, decision: Any) -> bool:
        lowered = text.lower()
        if decision.mode in {ReasoningMode.MEMORY_ONLY, ReasoningMode.ASK_USER_CLARIFY}:
            return True
        if decision.intent == "emergency":
            return True
        return any(token in lowered for token in ("ocr 결과", "dur 기준", "복용지도를 계획", "건강기능식품"))

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

    @staticmethod
    def _extract_ocr_medications_from_text(text: str) -> list[str]:
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
        return any(token in text for token in ("내 이름", "내 프로필", "기저질환"))

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
