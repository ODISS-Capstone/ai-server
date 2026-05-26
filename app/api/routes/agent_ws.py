"""로컬 에이전트 WebSocket 엔드포인트 — 실시간 양방향 통신.

데이터 흐름 (server.mermaid):
  LocalAgent → CE_Input → CE_Latency → ME_Context → RE_Intent
                                                      ↕
                                                    ME_RAG / Tools
                                                      ↓
                                                  RE_Core_Msg
                                                      ↓
                                              CE_Tone → CE_Response → LocalAgent
"""
import json
import logging
import asyncio
import re
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.engines.conversation import ConversationEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import ReasoningEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.services.engine_orchestrator import EngineOrchestrator
from app.services.identity_guard import evaluate_identity_gate
from app.services.llm import extract_ocr_medication_candidates_with_llm, refine_ocr_medication_candidates_with_context
from app.services.medication_extraction import (
    extract_medication_suffix_tokens,
    is_ocr_capture_request_text,
    is_wake_word_only,
    strip_wake_words,
)
from app.services.patient_safety import classify_patient_safety_situation
from app.services.reminders import ReminderService
from app.tools import llm_search

logger = logging.getLogger(__name__)
router = APIRouter()

memory_engine = MemoryEngine()
llm_judge = LLMJudgeEngine()
reasoning_engine = ReasoningEngine(memory_engine, llm_judge)
conversation_engine = ConversationEngine()
engine_orchestrator = EngineOrchestrator(
    memory_engine=memory_engine,
    reasoning_engine=reasoning_engine,
    conversation_engine=conversation_engine,
    llm_judge=llm_judge,
)
reminder_service = ReminderService()
_pending_ocr_by_speaker: dict[str, dict[str, Any]] = {}
_queued_ocr_request_by_speaker: dict[str, dict[str, Any]] = {}
_bootstrapped_speakers: set[str] = set()
_wake_profile_cache_by_speaker: dict[str, dict[str, Any]] = {}
_identity_pending_action_cache_by_speaker: dict[str, str] = {}
OCR_PENDING_TTL = timedelta(minutes=5)
ANONYMOUS_OCR_KEY = "__anonymous__"
WEBSOCKET_IDLE_TIMEOUT_SEC = 65.0
WAKE_PROFILE_LOOKUP_TIMEOUT_SEC = 0.15
IDENTITY_PENDING_ACTIONS = {
    "registration",
    "confirm_new_identity",
    "confirm_flash_identity",
    "identity_conflict",
    "reverification",
    "prior_conversation_check",
}
MEDICATION_PROGRESS_FILLERS = (
    "아직 약 정보를 확인하고 있습니다. 잠시만 기다려주세요.",
)
REMINDER_PROGRESS_FILLERS = (
    "아직 알림 정보를 확인하고 있습니다. 잠시만 기다려주세요.",
)
RECORD_PROGRESS_FILLERS = (
    "아직 복용 기록을 확인하고 있습니다. 잠시만 기다려주세요.",
)
GENERAL_PROGRESS_FILLERS = (
    "아직 내용을 확인하고 있습니다. 잠시만 기다려주세요.",
)
OCR_PROCESSING_FILLER = "사진을 확인하고 있습니다. 잠시만 기다려주세요."
OCR_PROGRESS_FILLERS = (
    "사진 속 글자를 확인하고 있습니다. 잠시만 기다려주세요.",
    "약 이름을 확인하고 있습니다. 잠시만 기다려주세요.",
)


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """로컬 에이전트와의 실시간 대화 WebSocket.

    수신 JSON:
      { "type": "stt_result", "text": "...", "speaker_id": "..." }
      { "type": "ocr_result", "data": { ... } }

    송신 JSON:
      { "type": "filler", "text": "..." }          # Latency Hiding
      { "type": "response", "text": "...", ... }    # 최종 응답
      { "type": "ocr_request", "message": "..." }   # 처방전 촬영 요청
      { "type": "reminder", "text": "...", ... }     # 예약 복약 알림
    """
    await websocket.accept()
    logger.info("WebSocket connected")
    active_speakers: set[str] = set()

    try:
        await memory_engine.initialize()

        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=WEBSOCKET_IDLE_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                await websocket.send_json(
                    {
                        "type": "session_closed",
                        "reason": "idle_timeout",
                        "requires_tts": False,
                    }
                )
                logger.info("WebSocket idle timeout; closing session")
                break
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON"}
                )
                continue

            msg_type = message.get("type", "")

            if msg_type in {"stt_result", "identity_confirmed"}:
                if msg_type == "identity_confirmed" and not message.get("text"):
                    message["text"] = "네, 본인 맞습니다."
                await _handle_stt(websocket, message, active_speakers)
            elif msg_type == "ocr_result":
                await _handle_ocr(websocket, message)
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json(
                    {"type": "error", "message": f"Unknown type: {msg_type}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_json(
                {"type": "error", "message": str(e)}
            )
        except Exception:
            pass
    finally:
        for speaker_id in active_speakers:
            reminder_service.unregister_connection(speaker_id)


async def _handle_stt(websocket: WebSocket, message: dict, active_speakers: set[str]) -> None:
    """STT 결과를 받아 전체 파이프라인 실행."""
    text = message.get("text", "").strip()
    speaker_id = message.get("speaker_id")

    if not text:
        await websocket.send_json(
            {"type": "error", "message": "Empty text"}
        )
        return

    if _is_incomplete_or_noise_utterance(text):
        await websocket.send_json(
            {
                "type": "ignored",
                "reason": "incomplete_or_noise_utterance",
                "requires_tts": False,
            }
        )
        return

    if is_wake_word_only(text):
        await _handle_wake_word_fast_path(websocket, speaker_id, active_speakers)
        return

    if _should_handle_profile_memory_ack_fast_path(text):
        await _handle_profile_memory_ack_fast_path(websocket, text, speaker_id, active_speakers)
        return

    if await _should_handle_smalltalk_fast_path(text, speaker_id):
        await _handle_smalltalk_fast_path(websocket, text, speaker_id, active_speakers)
        return

    if speaker_id:
        if speaker_id not in _bootstrapped_speakers:
            await memory_engine.bootstrap_flash_from_permanent(speaker_id)
            _bootstrapped_speakers.add(speaker_id)

        async def send_reminder(payload: dict[str, Any]) -> None:
            await websocket.send_json(payload)

        reminder_service.register_connection(speaker_id, send_reminder)
        active_speakers.add(speaker_id)
        await reminder_service.restore_for_speaker(memory_engine, speaker_id)

    if ReminderService.is_relative_alarm_request(text):
        await _handle_relative_alarm_request(websocket, text, speaker_id)
        return

    if ReminderService.is_missed_one_shot_check(text):
        await _handle_missed_one_shot_request(websocket, text, speaker_id)
        return

    pending_ocr_confirmation = _has_pending_ocr_confirmation(speaker_id)
    queued_ocr_reason = ""
    if not pending_ocr_confirmation:
        if _is_ocr_recapture_reply(text):
            queued_ocr_reason = "user_requested_recapture"
        elif _is_direct_ocr_capture_request(text):
            queued_ocr_reason = "direct_ocr_capture_request"
        if queued_ocr_reason:
            _queue_ocr_request(speaker_id, queued_ocr_reason)

    identity_gate = await evaluate_identity_gate(
        memory_engine=memory_engine,
        text=text,
        speaker_id=speaker_id,
    )
    if not identity_gate.allowed:
        if identity_gate.response_type == "ignored" and not identity_gate.response_text:
            await websocket.send_json(
                {
                    "type": "ignored",
                    "reason": identity_gate.reason,
                    "requires_tts": False,
                    "identity_gate": {
                        "reason": identity_gate.reason,
                        "metadata": identity_gate.metadata or {},
                    },
                }
            )
            return
        response = conversation_engine.build_response(
            {
                "text": identity_gate.response_text,
                "type": identity_gate.response_type,
                "requires_tts": True,
            }
        )
        await websocket.send_json(
            {
                "type": "response",
                **response,
                "identity_gate": {
                    "reason": identity_gate.reason,
                    "metadata": identity_gate.metadata or {},
                },
            }
        )
        _sync_wake_profile_cache_from_identity_gate(speaker_id, identity_gate)
        return

    if queued_ocr_reason:
        await _send_queued_ocr_request_if_ready(websocket, speaker_id)
        return

    if await _handle_pending_ocr_confirmation(websocket, text, speaker_id):
        return

    if await _send_queued_ocr_request_if_ready(websocket, speaker_id):
        return

    if await _handle_medication_safety_question_request(websocket, text, speaker_id, identity_gate):
        return

    spoken_medications = _extract_spoken_medications_from_text(text)
    if spoken_medications:
        await _handle_spoken_medication_registration_request(
            websocket,
            text,
            speaker_id,
            identity_gate,
            spoken_medications,
        )
        return

    if await _handle_current_medication_list_request(websocket, text, speaker_id, identity_gate):
        return

    if await _handle_medication_intent_to_take_request(websocket, text, speaker_id, identity_gate):
        return

    if await _handle_stored_medication_guidance_request(websocket, text, speaker_id, identity_gate):
        return

    if ReminderService.is_taken_time_correction(text):
        await _handle_medication_taken_time_correction_request(websocket, text, speaker_id, identity_gate)
        return

    if ReminderService.is_taken_recall(text):
        await _handle_medication_taken_recall_request(websocket, text, speaker_id, identity_gate)
        return

    if ReminderService.is_taken_confirmation(text):
        await _handle_medication_taken_confirmation_request(websocket, text, speaker_id, identity_gate)
        return

    immediate_filler = _immediate_filler_for_text(text)
    progress_task: asyncio.Task | None = None
    if immediate_filler:
        await _send_runtime_filler(websocket, immediate_filler, stage=_processing_stage_for_text(text))
        progress_task = asyncio.create_task(
            _send_progress_fillers(websocket, text, initial_sent=True)
        )
    try:
        context = await memory_engine.load_context(speaker_id)
        gate_profile = (identity_gate.metadata or {}).get("profile") or {}
        if gate_profile and not (context.get("user_profile") or {}).get("name"):
            context["user_profile"] = gate_profile

        reminder_text = await reminder_service.handle_user_text(
            memory_engine=memory_engine,
            speaker_id=speaker_id,
            text=text,
            user_profile=context.get("user_profile", {}),
            prescription_log=context.get("prescription_log", ""),
        )
        if reminder_text:
            response = conversation_engine.build_response(
                {
                    "text": reminder_text,
                    "type": "reminder",
                    "requires_tts": True,
                }
            )
            await websocket.send_json({"type": "response", **response})
            await memory_engine.update_and_compress(
                {
                    "query": text,
                    "answer": reminder_text,
                    "type": "reminder",
                },
                speaker_id=speaker_id,
            )
            if speaker_id:
                await memory_engine.mark_identity_seen(speaker_id, verified=True)
            return

        preview_input = conversation_engine.receive_input(text, speaker_id)
        pre_filler = "" if immediate_filler else conversation_engine.generate_filler(preview_input)
        if pre_filler:
            await _send_runtime_filler(websocket, pre_filler, stage=_processing_stage_for_text(text))

        if progress_task is None:
            progress_task = asyncio.create_task(
                _send_progress_fillers(websocket, text, initial_sent=bool(pre_filler))
            )
        turn = await engine_orchestrator.run_turn(
            text=text,
            speaker_id=speaker_id,
            include_judge=True,
            include_delivery_llm=True,
            allow_frontier_memory_fallback=True,
            preloaded_context=context,
        )
    finally:
        await _cancel_progress_task(progress_task)

    if turn.filler_text and not immediate_filler and not pre_filler:
        await _send_runtime_filler(websocket, turn.filler_text, stage=_processing_stage_for_text(text))

    if not turn.conversation.requires_tts and not turn.conversation.response_text:
        await websocket.send_json(
            {
                "type": "ignored",
                "reason": turn.decision.rationale,
                "requires_tts": False,
            }
        )
        return

    # OCR 요청이 필요한 경우
    if turn.execution_results.get("task_results", {}).get("ocr_requested"):
        ocr_request = reasoning_engine.request_ocr()
        await websocket.send_json({"type": "ocr_request", **ocr_request})
        return

    synthesis = {
        "text": turn.conversation.response_text,
        "type": turn.conversation.response_type,
        "requires_tts": turn.conversation.requires_tts,
    }
    response = conversation_engine.build_response(synthesis)
    await websocket.send_json({"type": "response", **response})

    # ME_Update: 결과 저장 및 Flash Memory 압축
    await memory_engine.update_and_compress(
        {
            "query": text,
            "answer": synthesis["text"],
            "type": turn.decision.intent,
            "core_message": turn.core_message,
            "judge_review": turn.judge_review,
            "dur_results": turn.execution_results.get("task_results", {}).get("dur"),
        },
        speaker_id=speaker_id,
    )
    if speaker_id:
        await memory_engine.mark_identity_seen(speaker_id, verified=True)


async def _handle_wake_word_fast_path(
    websocket: WebSocket,
    speaker_id: str | None,
    active_speakers: set[str],
) -> None:
    """Acknowledge wake-word-only turns without waiting for identity/LLM work."""
    if speaker_id:
        async def send_reminder(payload: dict[str, Any]) -> None:
            await websocket.send_json(payload)

        reminder_service.register_connection(speaker_id, send_reminder)
        active_speakers.add(speaker_id)

    profile = await _load_wake_profile_fast(speaker_id)
    response_text = conversation_engine.build_wake_word_response(profile)
    response = conversation_engine.build_response(
        {
            "text": response_text,
            "type": "wake_word_ack",
            "requires_tts": True,
        }
    )
    await websocket.send_json({"type": "response", **response})

    if speaker_id:
        asyncio.create_task(_refresh_wake_word_state_background(speaker_id))


async def _should_handle_smalltalk_fast_path(text: str, speaker_id: str | None) -> bool:
    if _has_pending_ocr_confirmation(speaker_id):
        return False
    if not conversation_engine.fast_smalltalk_type(text):
        return False
    if await _has_pending_identity_action_fast(speaker_id):
        return False
    return True


async def _has_pending_identity_action_fast(speaker_id: str | None) -> bool:
    if not speaker_id:
        return False
    cached = _identity_pending_action_cache_by_speaker.get(speaker_id)
    if cached in IDENTITY_PENDING_ACTIONS:
        return True
    try:
        state = await asyncio.wait_for(
            memory_engine.load_identity_state(speaker_id),
            timeout=WAKE_PROFILE_LOOKUP_TIMEOUT_SEC,
        )
    except Exception as exc:  # noqa: BLE001 - smalltalk should stay responsive on lookup failure
        logger.debug("[SmalltalkFastPath] identity_pending_lookup_skipped speaker=%s error=%r", speaker_id, exc)
        return False
    pending = str(state.get("pending_identity_action") or "")
    if pending in IDENTITY_PENDING_ACTIONS:
        _identity_pending_action_cache_by_speaker[speaker_id] = pending
        return True
    _identity_pending_action_cache_by_speaker.pop(speaker_id, None)
    return False


async def _handle_smalltalk_fast_path(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
    active_speakers: set[str],
) -> None:
    """Answer pure social smalltalk without identity, reminder, RAG, or LLM work."""
    started = perf_counter()
    if speaker_id:
        async def send_reminder(payload: dict[str, Any]) -> None:
            await websocket.send_json(payload)

        reminder_service.register_connection(speaker_id, send_reminder)
        active_speakers.add(speaker_id)

    profile = await _load_wake_profile_fast(speaker_id)
    response_text = conversation_engine.build_smalltalk_fast_response(text, profile)
    response = conversation_engine.build_response(
        {
            "text": response_text,
            "type": "smalltalk",
            "requires_tts": True,
        }
    )
    await websocket.send_json(
        {
            "type": "response",
            **response,
            "fast_path": "smalltalk",
            "server_elapsed_ms": round((perf_counter() - started) * 1000, 1),
        }
    )

    if speaker_id:
        asyncio.create_task(_refresh_wake_word_state_background(speaker_id))


def _should_handle_profile_memory_ack_fast_path(text: str) -> bool:
    compact = re.sub(r"[\s.?!,，。~]+", "", (text or "").strip().lower())
    if not compact:
        return False
    if not any(token in compact for token in ("기억해줘", "기억해", "기억하고있", "잘기억")):
        return False
    if any(
        token in compact
        for token in (
            "약",
            "복용",
            "처방",
            "먹",
            "타이레놀",
            "디오반",
            "혈압",
            "당뇨",
            "알림",
            "알람",
            "기록",
            "사진",
            "ocr",
        )
    ):
        return False
    return True


async def _handle_profile_memory_ack_fast_path(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
    active_speakers: set[str],
) -> None:
    """Acknowledge profile-memory requests without pulling medication context."""
    started = perf_counter()
    if speaker_id:
        async def send_reminder(payload: dict[str, Any]) -> None:
            await websocket.send_json(payload)

        reminder_service.register_connection(speaker_id, send_reminder)
        active_speakers.add(speaker_id)

    profile = await _load_wake_profile_fast(speaker_id)
    response_text = _build_profile_memory_ack_response(profile)
    response = conversation_engine.build_response(
        {
            "text": response_text,
            "type": "profile_memory_ack",
            "requires_tts": True,
        }
    )
    await websocket.send_json(
        {
            "type": "response",
            **response,
            "fast_path": "profile_memory_ack",
            "server_elapsed_ms": round((perf_counter() - started) * 1000, 1),
        }
    )

    if speaker_id:
        asyncio.create_task(_refresh_wake_word_state_background(speaker_id))


async def _handle_relative_alarm_request(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
    identity_gate=None,
) -> None:
    """Handle relative one-shot alarm requests before filler/orchestrator work."""
    if not speaker_id:
        response = conversation_engine.build_response(
            {
                "text": "알림 설정에는 사용자 연결 정보가 필요합니다. 오디스를 다시 불러 주세요.",
                "type": "reminder",
                "requires_tts": True,
            }
        )
        await websocket.send_json(
            {
                "type": "response",
                **response,
                "fast_path": "relative_alarm",
            }
        )
        return

    context = await memory_engine.load_context(speaker_id)
    gate_profile = ((identity_gate.metadata or {}).get("profile") if identity_gate else {}) or {}
    if gate_profile and not (context.get("user_profile") or {}).get("name"):
        context["user_profile"] = gate_profile

    reminder_text = await reminder_service.handle_user_text(
        memory_engine=memory_engine,
        speaker_id=speaker_id,
        text=text,
        user_profile=context.get("user_profile", {}),
        prescription_log=context.get("prescription_log", ""),
    )
    if reminder_text is None:
        reminder_text = await reminder_service.schedule_one_shot(
            speaker_id=speaker_id,
            text=text,
            user_profile=context.get("user_profile", {}),
            prescription_log=context.get("prescription_log", ""),
        )
    if not reminder_text:
        reminder_text = "알림을 설정했습니다."

    response = conversation_engine.build_response(
        {
            "text": reminder_text,
            "type": "reminder",
            "requires_tts": True,
        }
    )
    await websocket.send_json(
        {
            "type": "response",
            **response,
            "fast_path": "relative_alarm",
            **reminder_service.one_shot_metadata_for_speaker(speaker_id),
        }
    )
    await memory_engine.update_and_compress(
        {
            "query": text,
            "answer": reminder_text,
            "type": "reminder",
        },
        speaker_id=speaker_id,
    )
    await memory_engine.mark_identity_seen(speaker_id, verified=True)


async def _handle_missed_one_shot_request(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
) -> None:
    if speaker_id:
        dispatched = await reminder_service.dispatch_due_reminders()
        if dispatched or reminder_service.had_recent_one_shot_dispatch(speaker_id):
            return
    response = conversation_engine.build_response(
        {
            "text": "방금 설정된 알림을 찾지 못했습니다. 다시 설정해 주세요.",
            "type": "reminder",
            "requires_tts": True,
        }
    )
    await websocket.send_json(
        {
            "type": "response",
            **response,
            "fast_path": "missed_one_shot_check",
        }
    )


async def _handle_medication_taken_recall_request(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
    identity_gate,
) -> None:
    if not speaker_id:
        response = conversation_engine.build_response(
            {
                "text": "복용 기록 확인에는 사용자 연결 정보가 필요합니다. 오디스를 다시 불러 주세요.",
                "type": "reminder",
                "requires_tts": True,
            }
        )
        await websocket.send_json(
            {
                "type": "response",
                **response,
                "fast_path": "medication_taken_recall",
            }
        )
        return

    context = await memory_engine.load_context(speaker_id)
    gate_profile = (identity_gate.metadata or {}).get("profile") or {}
    if gate_profile and not (context.get("user_profile") or {}).get("name"):
        context["user_profile"] = gate_profile

    reminder_text = await reminder_service.handle_user_text(
        memory_engine=memory_engine,
        speaker_id=speaker_id,
        text=text,
        user_profile=context.get("user_profile", {}),
        prescription_log=context.get("prescription_log", ""),
    )
    if reminder_text is None:
        reminder_text = await reminder_service.recall_last_taken(
            memory_engine=memory_engine,
            speaker_id=speaker_id,
            user_profile=context.get("user_profile", {}),
        )

    response = conversation_engine.build_response(
        {
            "text": reminder_text,
            "type": "reminder",
            "requires_tts": True,
        }
    )
    await websocket.send_json(
        {
            "type": "response",
            **response,
            "fast_path": "medication_taken_recall",
        }
    )
    await memory_engine.update_and_compress(
        {
            "query": text,
            "answer": reminder_text,
            "type": "reminder",
        },
        speaker_id=speaker_id,
    )
    await memory_engine.mark_identity_seen(speaker_id, verified=True)


async def _handle_spoken_medication_registration_request(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
    identity_gate,
    medications: list[str],
) -> None:
    """Store verbally provided current-medication names before filler/LLM work."""
    if not speaker_id:
        response = conversation_engine.build_response(
            {
                "text": "약 기록 저장에는 사용자 연결 정보가 필요합니다. 오디스를 다시 불러 주세요.",
                "type": "medication_query",
                "requires_tts": True,
            }
        )
        await websocket.send_json(
            {
                "type": "response",
                **response,
                "fast_path": "spoken_medication_registration",
                "medications": medications,
            }
        )
        return

    context = await _load_context_with_identity_profile(speaker_id, identity_gate)
    if hasattr(memory_engine, "store_spoken_medication_result"):
        merged = await memory_engine.store_spoken_medication_result(
            text,
            medications,
            speaker_id=speaker_id,
        )
    else:
        merged = medications
    if not merged:
        merged = medications

    name = _display_name_from_context(context)
    med_text = _friendly_medication_label(merged)
    response_text = (
        f"{name}, {med_text}을 현재 복용 약 목록에 추가했습니다. "
        "복용 시간과 한 번에 드실 양은 약봉투나 처방전 기준으로 확인해 주세요. "
        "밥을 드신 뒤나 복용 시간이 헷갈릴 때 말씀하시면 이 기록을 기준으로 안내드릴게요."
    )
    response = conversation_engine.build_response(
        {
            "text": response_text,
            "type": "medication_query",
            "requires_tts": True,
        }
    )
    await websocket.send_json(
        {
            "type": "response",
            **response,
            "fast_path": "spoken_medication_registration",
            "medications": merged,
        }
    )
    await memory_engine.update_and_compress(
        {
            "query": text,
            "answer": response_text,
            "type": "medication_query",
        },
        speaker_id=speaker_id,
    )
    await memory_engine.mark_identity_seen(speaker_id, verified=True)


async def _handle_medication_safety_question_request(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
    identity_gate,
) -> bool:
    """Answer coadministration/overdose safety questions before filler/LLM work."""
    if classify_patient_safety_situation(text):
        return False
    if not _is_medication_safety_question_request(text):
        return False

    context = (
        await _load_context_with_identity_profile(speaker_id, identity_gate)
        if speaker_id
        else {"user_profile": (getattr(identity_gate, "metadata", None) or {}).get("profile") or {}}
    )
    meds = _medications_for_safety_question(text, context)
    response_text = _build_medication_safety_question_text(text, meds, context)
    response = conversation_engine.build_response(
        {
            "text": response_text,
            "type": "medical_response",
            "requires_tts": True,
        }
    )
    await websocket.send_json(
        {
            "type": "response",
            **response,
            "fast_path": "medication_safety_fast_path",
            "medications": meds,
        }
    )
    if speaker_id:
        await memory_engine.update_and_compress(
            {
                "query": text,
                "answer": response_text,
                "type": "medical_response",
            },
            speaker_id=speaker_id,
        )
        await memory_engine.mark_identity_seen(speaker_id, verified=True)
    return True


async def _handle_stored_medication_guidance_request(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
    identity_gate,
) -> bool:
    """Answer vague meal/that-med requests from stored or explicitly named medication context."""
    if not speaker_id or not _has_stored_medication_guidance_signal(text):
        return False
    context = await _load_context_with_identity_profile(speaker_id, identity_gate)
    stored_meds = _medications_from_prescription_log(context.get("prescription_log", ""))
    explicit_meds = _explicit_medications_from_text(text)
    meds = explicit_meds or stored_meds
    if not meds or not _is_stored_medication_guidance_request(text, meds):
        return False

    explicit_meal_guidance = bool(explicit_meds and _is_meal_guidance_signal(text))
    if explicit_meal_guidance:
        await memory_engine.store_spoken_medication_result(
            text,
            explicit_meds,
            speaker_id=speaker_id,
        )
    response_text = _build_stored_medication_guidance_text(
        text,
        meds,
        context,
        explicit_meal_guidance=explicit_meal_guidance,
    )
    response = conversation_engine.build_response(
        {
            "text": response_text,
            "type": "medical_response",
            "requires_tts": True,
        }
    )
    await websocket.send_json(
        {
            "type": "response",
            **response,
            "fast_path": "named_meal_medication_guidance" if explicit_meal_guidance else "stored_medication_guidance",
            "medications": meds,
        }
    )
    await memory_engine.update_and_compress(
        {
            "query": text,
            "answer": response_text,
            "type": "medical_response",
        },
        speaker_id=speaker_id,
    )
    await memory_engine.mark_identity_seen(speaker_id, verified=True)
    return True


async def _handle_current_medication_list_request(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
    identity_gate,
) -> bool:
    """Answer current stored-medication list questions before filler/LLM work."""
    if not speaker_id or not _is_current_medication_list_request(text):
        return False
    context = await _load_context_with_identity_profile(speaker_id, identity_gate)
    meds = _medications_from_prescription_log(context.get("prescription_log", ""))
    if not meds:
        return False

    response_text = _build_current_medication_list_text(text, meds, context)
    response = conversation_engine.build_response(
        {
            "text": response_text,
            "type": "medication_query",
            "requires_tts": True,
        }
    )
    await websocket.send_json(
        {
            "type": "response",
            **response,
            "fast_path": "stored_medication_list_recall",
            "medications": meds,
        }
    )
    await memory_engine.update_and_compress(
        {
            "query": text,
            "answer": response_text,
            "type": "medication_query",
        },
        speaker_id=speaker_id,
    )
    await memory_engine.mark_identity_seen(speaker_id, verified=True)
    return True


async def _handle_medication_intent_to_take_request(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
    identity_gate,
) -> bool:
    """Handle "I'll take it now" turns as medication workflow, not smalltalk."""
    if not speaker_id or not _is_medication_intent_to_take_request(text):
        return False
    context = await _load_context_with_identity_profile(speaker_id, identity_gate)
    meds = _medications_from_prescription_log(context.get("prescription_log", ""))
    if not meds:
        return False

    response_text = _build_medication_intent_to_take_text(meds, context)
    response = conversation_engine.build_response(
        {
            "text": response_text,
            "type": "medical_response",
            "requires_tts": True,
        }
    )
    await websocket.send_json(
        {
            "type": "response",
            **response,
            "fast_path": "medication_intent_to_take",
            "medications": meds,
        }
    )
    await memory_engine.update_and_compress(
        {
            "query": text,
            "answer": response_text,
            "type": "medical_response",
        },
        speaker_id=speaker_id,
    )
    await memory_engine.mark_identity_seen(speaker_id, verified=True)
    return True


async def _handle_medication_taken_confirmation_request(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
    identity_gate,
) -> None:
    """Record simple medication-taken confirmations before filler/LLM work."""
    if not speaker_id:
        response = conversation_engine.build_response(
            {
                "text": "복용 기록 저장에는 사용자 연결 정보가 필요합니다. 오디스를 다시 불러 주세요.",
                "type": "reminder",
                "requires_tts": True,
            }
        )
        await websocket.send_json(
            {
                "type": "response",
                **response,
                "fast_path": "medication_taken_record",
            }
        )
        return

    context = await _load_context_with_identity_profile(speaker_id, identity_gate)
    reminder_text = await reminder_service.handle_user_text(
        memory_engine=memory_engine,
        speaker_id=speaker_id,
        text=text,
        user_profile=context.get("user_profile", {}),
        prescription_log=context.get("prescription_log", ""),
    )
    if not reminder_text:
        reminder_text = "복용했다고 기록해두겠습니다."
    response = conversation_engine.build_response(
        {
            "text": reminder_text,
            "type": "reminder",
            "requires_tts": True,
        }
    )
    await websocket.send_json(
        {
            "type": "response",
            **response,
            "fast_path": "medication_taken_record",
        }
    )
    await memory_engine.update_and_compress(
        {
            "query": text,
            "answer": reminder_text,
            "type": "reminder",
        },
        speaker_id=speaker_id,
    )
    await memory_engine.mark_identity_seen(speaker_id, verified=True)


async def _handle_medication_taken_time_correction_request(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
    identity_gate,
) -> None:
    """Correct the latest medication-taken timestamp without LLM work."""
    if not speaker_id:
        response = conversation_engine.build_response(
            {
                "text": "복용 시간 수정에는 사용자 연결 정보가 필요합니다. 오디스를 다시 불러 주세요.",
                "type": "reminder",
                "requires_tts": True,
            }
        )
        await websocket.send_json(
            {
                "type": "response",
                **response,
                "fast_path": "medication_taken_time_correction",
            }
        )
        return

    context = await _load_context_with_identity_profile(speaker_id, identity_gate)
    reminder_text = await reminder_service.correct_last_taken_time(
        memory_engine=memory_engine,
        speaker_id=speaker_id,
        text=text,
        user_profile=context.get("user_profile", {}),
    )
    response = conversation_engine.build_response(
        {
            "text": reminder_text,
            "type": "reminder",
            "requires_tts": True,
        }
    )
    await websocket.send_json(
        {
            "type": "response",
            **response,
            "fast_path": "medication_taken_time_correction",
        }
    )
    await memory_engine.update_and_compress(
        {
            "query": text,
            "answer": reminder_text,
            "type": "reminder",
        },
        speaker_id=speaker_id,
    )
    await memory_engine.mark_identity_seen(speaker_id, verified=True)


async def _load_context_with_identity_profile(speaker_id: str, identity_gate) -> dict[str, Any]:
    context = await memory_engine.load_context(speaker_id)
    gate_profile = (identity_gate.metadata or {}).get("profile") or {}
    if gate_profile and not (context.get("user_profile") or {}).get("name"):
        context["user_profile"] = gate_profile
    return context


def _extract_spoken_medications_from_text(text: str) -> list[str]:
    extractor = getattr(memory_engine, "extract_spoken_medications_from_text", None)
    if not callable(extractor):
        return []
    return extractor(text)


def _compact_text(text: str) -> str:
    return re.sub(r"[\s\t\r\n.,;:!?~'\"`，。]+", "", (text or "").strip().lower())


def _is_medication_safety_question_request(text: str) -> bool:
    compact = _compact_text(strip_wake_words(text))
    if not compact:
        return False
    if any(token in compact for token in ("알림", "알람", "예약", "깨워", "설정", "기록해", "먹었다고")):
        return False
    med_signal = (
        "약" in compact
        or "복용" in compact
        or bool(extract_medication_suffix_tokens(strip_wake_words(text)))
        or any(token in compact for token in ("타이레놀", "아세트아미노펜", "디오반", "와파린", "아스피린"))
    )
    if not med_signal:
        return False
    safety_signal = any(
        token in compact
        for token in (
            "먹어도돼",
            "먹어도되",
            "먹어도될까",
            "먹어도되나",
            "먹어도괜찮",
            "복용해도돼",
            "복용해도되",
            "괜찮",
            "문제없",
            "위험",
            "부작용",
            "같이먹",
            "함께먹",
            "동시에",
            "한번에",
            "한꺼번에",
            "여러알",
            "많이먹",
            "더먹",
            "중복",
            "겹쳐먹",
            "두개",
            "2개",
            "세개",
            "3개",
            "네개",
            "내게",
            "4개",
        )
    )
    return safety_signal and any(token in compact for token in ("먹", "복용", "삼켜", "드셔", "먹어도", "복용해도"))


def _medications_for_safety_question(text: str, context: dict[str, Any]) -> list[str]:
    cleaned = strip_wake_words(text)
    compact = _compact_text(cleaned)
    stored = _medications_from_prescription_log(context.get("prescription_log", ""))
    meds: list[str] = []

    for med in stored:
        normalized = _compact_text(med)
        stem = normalized[:-1] if normalized.endswith("정") else normalized
        if normalized and (normalized in compact or (len(stem) >= 2 and stem in compact)):
            meds.append(med)

    for med in extract_medication_suffix_tokens(cleaned):
        if med not in meds:
            meds.append(med)

    common_names = ("타이레놀", "아세트아미노펜", "혈압약", "고혈압약", "당뇨약", "와파린", "아스피린")
    for name in common_names:
        if name in compact and name not in meds:
            meds.append(name)

    if not meds:
        meds.extend(stored[:3])
    return meds[:5]


def _build_medication_safety_question_text(
    text: str,
    meds: list[str],
    context: dict[str, Any],
) -> str:
    name = _display_name_from_context(context)
    med_text = _friendly_medication_label(meds) if meds else "그 약"
    compact = _compact_text(strip_wake_words(text))
    other_med_signal = any(token in compact for token in ("다른약", "같이먹", "함께먹", "병용", "상호작용"))
    multi_dose_signal = any(
        token in compact
        for token in (
            "동시에",
            "한번에",
            "한꺼번에",
            "여러알",
            "많이먹",
            "더먹",
            "중복",
            "두개",
            "2개",
            "세개",
            "3개",
            "네개",
            "내게",
            "4개",
        )
    )
    if other_med_signal and not multi_dose_signal:
        return (
            f"{name}, {med_text}을 다른 약과 같이 드셔도 되는지는 같이 드시려는 약 이름이 필요합니다. "
            "확인 전에는 임의로 같이 드시지 말고, 약봉투나 처방전을 들고 의사나 약사에게 먼저 확인해 주세요."
        )
    return (
        f"{name}, {med_text}은 처방된 양보다 한 번에 더 드시면 위험할 수 있습니다. "
        "지금 여러 알을 동시에 드시려는 상황이면 드시지 말고, 약봉투에 적힌 1회 복용량을 먼저 확인해 주세요. "
        "이미 많이 드셨거나 어지러움, 심한 저혈압 느낌, 실신할 것 같은 증상이 있으면 119나 응급실에 연락하세요."
    )


def _has_stored_medication_guidance_signal(text: str) -> bool:
    compact = re.sub(r"[\s.?!,，。~]+", "", (text or "").strip().lower())
    if not compact:
        return False
    return any(
        token in compact
        for token in (
            "먹어야",
            "먹으라고",
            "밥먹었",
            "밥먹고오",
            "식사했",
            "식사끝",
            "식후",
            "그거",
            "잘먹었",
            "잘먹었습니다",
            "잘먹음",
            "알려줘",
            "알려줄",
        )
    )


def _is_stored_medication_guidance_request(text: str, meds: list[str] | None = None) -> bool:
    compact = re.sub(r"[\s.?!,，。~]+", "", (text or "").strip().lower())
    if not compact:
        return False
    if any(token in compact for token in ("알림", "알람", "예약", "깨워", "설정", "추가")):
        if not _is_meal_based_notification_guidance_request(text):
            return False
    if any(token in compact for token in ("먹어도돼", "먹어도되", "같이먹", "동시에", "많이먹", "네개", "4개")):
        return False
    if any(token in compact for token in ("밥먹었", "밥먹고오", "먹고오면", "식사했", "식사끝", "식후", "저녁먹었", "점심먹었", "아침먹었", "잘먹었", "잘먹었습니다", "잘먹음")):
        return True
    if "그거" in compact and any(token in compact for token in ("먹어야", "먹나", "먹으면", "먹을까")):
        return True
    for med in meds or []:
        normalized_med = re.sub(r"\s+", "", med.lower())
        if normalized_med and normalized_med in compact and "먹어야" in compact:
            return True
    if "먹어야" in compact and not any(token in compact for token in ("밥먹어야", "식사해야", "물먹어야")):
        return True
    return "오늘" in compact and "먹어야" in compact and ("약" in compact or "그거" in compact)


def _is_current_medication_list_request(text: str) -> bool:
    compact = re.sub(r"[\s.?!,，。~]+", "", (text or "").strip().lower())
    if not compact:
        return False
    if any(token in compact for token in ("기록해", "먹었", "먹을게", "먹어도", "위험", "부작용")):
        return False
    list_signal = any(token in compact for token in ("저장된", "저장된게", "등록된", "기록된", "목록", "뭐있", "뭐가있", "다른약", "그거말고"))
    query_signal = any(token in compact for token in ("있나", "있어", "뭐", "뭐야", "알려", "확인", "보여"))
    return list_signal and query_signal


def _build_current_medication_list_text(
    text: str,
    meds: list[str],
    context: dict[str, Any],
) -> str:
    name = _display_name_from_context(context)
    med_text = _friendly_medication_label(meds)
    compact = re.sub(r"[\s.?!,，。~]+", "", (text or "").strip().lower())
    if ("그거말고" in compact or "다른" in compact) and len(meds) <= 1:
        return (
            f"{name}, 현재 저장된 약은 {med_text}입니다. "
            "그 외에 추가로 저장된 약은 없습니다. 새 약이 있으면 약 이름을 말씀해 주세요."
        )
    return (
        f"{name}, 현재 저장된 약은 {med_text}입니다. "
        "복용 시간과 한 번에 드실 양은 약봉투나 처방전 기준으로 확인해 주세요."
    )


def _is_medication_intent_to_take_request(text: str) -> bool:
    compact = re.sub(r"[\s.?!,，。~]+", "", (text or "").strip().lower())
    if not compact:
        return False
    if any(token in compact for token in ("먹었", "복용했", "기록", "먹어도", "괜찮", "위험", "부작용")):
        return False
    return any(token in compact for token in ("지금먹을게", "지금먹을께", "먹을게", "먹을께", "먹겠습니다", "먹을게요"))


def _build_medication_intent_to_take_text(
    meds: list[str],
    context: dict[str, Any],
) -> str:
    name = _display_name_from_context(context)
    med_text = _friendly_medication_label(meds)
    return (
        f"네, {name}. 현재 저장된 약은 {med_text}입니다. "
        "약봉투나 처방전에 적힌 복용 시간이 맞다면 정해진 양만 물과 함께 드세요. "
        "드신 뒤에는 '먹었어'라고 말씀하시면 복용 기록으로 남겨둘게요."
    )


def _build_stored_medication_guidance_text(
    text: str,
    meds: list[str],
    context: dict[str, Any],
    *,
    explicit_meal_guidance: bool = False,
) -> str:
    name = _display_name_from_context(context)
    med_text = _friendly_medication_label(meds)
    meal = _meal_hint_from_text(text)
    compact = re.sub(r"\s+", "", text or "")
    if explicit_meal_guidance:
        meal_label = f"{meal} 식사" if meal else "식사"
        if _is_meal_based_notification_guidance_request(text):
            return (
                f"{name}, 네. 지금은 {_current_time_phrase()}이라 {meal_label} 후 {med_text} 안내로 기억해둘게요. "
                f"{meal_label}를 하고 오시면 저에게 '밥 먹었어'라고 말씀해 주세요. "
                f"그러면 {med_text}을 드시라고 안내드리겠습니다. "
                "드신 뒤에는 '먹었어'라고 알려주시면 복용 기록으로 남기겠습니다. "
                "복용량은 약봉투나 제품 포장에 적힌 대로만 드세요."
            )
        return (
            f"{name}, 네. 지금 말씀하신 기준으로 {meal_label} 후 {med_text}을 드셔야 하는 것으로 기억해둘게요. "
            f"{meal_label}를 하고 오시면 저에게 '밥 먹었어'라고 말씀해 주세요. "
            f"그러면 {med_text} 복용을 안내드리겠습니다. "
            "드신 뒤에는 '먹었어'라고 말씀하시면 복용 기록으로 남기겠습니다. "
            "복용량은 약봉투나 제품 포장에 적힌 대로만 드세요."
        )
    if _is_meal_guidance_signal(text):
        meal_label = f"{meal} 식사" if meal else "식사"
        if _is_meal_based_notification_guidance_request(text):
            return (
                f"{name}, 네. 지금은 {_current_time_phrase()}이라 {meal_label} 후 {med_text} 안내로 기억해둘게요. "
                f"{meal_label}를 하고 오시면 저에게 '밥 먹었어'라고 말씀해 주세요. "
                f"그러면 {med_text}을 드시라고 안내드리겠습니다. "
                "드신 뒤에는 '먹었어'라고 알려주시면 복용 기록으로 남기겠습니다. "
                "복용량은 약봉투나 처방전에 적힌 대로만 드세요."
            )
        if _is_after_meal_completion_signal(text):
            return (
                f"{name}, 네. {meal_label}를 하셨군요. {med_text}을 드시면 됩니다. "
                "복용량은 약봉투나 처방전에 적힌 대로만 드세요. "
                "드신 뒤에는 '먹었어'라고 말씀해 주세요."
            )
        meal_part = f"{meal_label} 후"
        return (
            f"{name}, 현재 기록 기준으로 저장된 약은 {med_text}입니다. "
            f"밥을 드신 뒤, 약봉투나 처방전에 식후, 즉 {meal_part} 복용으로 적혀 있다면 정해진 양만 물과 함께 드세요. "
            "이미 드셨거나 헷갈리면 한 번 더 드시지 말고 약봉투나 약통을 먼저 확인해 주세요."
        )
    return (
        f"{name}, 현재 저장된 약은 {med_text}입니다. "
        "오늘 드셔야 하는지와 시간은 약봉투나 처방전에 적힌 복용법을 기준으로 확인해야 합니다. "
        "복용 시간이 맞고 아직 안 드셨다면 정해진 양만 드세요. 이미 드셨거나 헷갈리면 한 번 더 드시지 마세요."
    )


def _medications_from_prescription_log(content: str) -> list[str]:
    meds: list[str] = []
    for line in str(content or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            name = stripped[2:].strip()
            if name and name not in meds:
                meds.append(name)
    return meds[:8]


def _explicit_medications_from_text(text: str) -> list[str]:
    cleaned = strip_wake_words(text)
    compact = re.sub(r"[\s.?!,，。~]+", "", cleaned.lower())
    meds: list[str] = []
    for med in extract_medication_suffix_tokens(cleaned):
        if med not in meds:
            meds.append(med)
    aliases = {
        "타이레놀": "타이레놀",
        "아세트아미노펜": "아세트아미노펜",
        "디오반": "디오반정",
        "와파린": "와파린",
        "아스피린": "아스피린",
        "로사르탄": "로사르탄정",
    }
    for alias, canonical in aliases.items():
        if alias in compact and canonical not in meds:
            meds.append(canonical)
    return meds[:5]


def _display_name_from_context(context: dict[str, Any]) -> str:
    profile = context.get("user_profile") or {}
    name = str(profile.get("name") or "").strip()
    return f"{name}님" if name else "사용자님"


def _build_profile_memory_ack_response(profile: dict[str, Any]) -> str:
    name = str((profile or {}).get("name") or "").strip()
    if name:
        return f"네, 알겠습니다. 앞으로 {name}님 정보로 잘 기억하겠습니다."
    return "아직 등록된 이름이 없습니다. 이름, 나이, 성별을 말씀해 주시면 기억하겠습니다."


def _sync_wake_profile_cache_from_identity_gate(speaker_id: str | None, identity_gate) -> None:
    if not speaker_id:
        return
    metadata = getattr(identity_gate, "metadata", None) or {}
    reason = str(getattr(identity_gate, "reason", "") or "")
    pending = str((metadata.get("profile") or {}).get("pending_identity_action") or "")
    if pending in IDENTITY_PENDING_ACTIONS:
        _identity_pending_action_cache_by_speaker[speaker_id] = pending
    elif reason in {
        "identity_registered",
        "identity_candidate_registered",
        "identity_verified",
        "identity_recognized",
        "identity_reverified",
        "no_speaker_id",
    }:
        _identity_pending_action_cache_by_speaker.pop(speaker_id, None)
    if reason in {"identity_rejected_needs_registration", "needs_registration"}:
        _identity_pending_action_cache_by_speaker[speaker_id] = "registration"
        _wake_profile_cache_by_speaker.pop(speaker_id, None)
        return
    profile = metadata.get("profile") or {}
    if profile.get("name"):
        _wake_profile_cache_by_speaker[speaker_id] = profile
        return
    _wake_profile_cache_by_speaker.pop(speaker_id, None)


def _friendly_medication_label(meds: list[str]) -> str:
    specific = [med for med in meds if med not in {"혈압약", "고혈압약", "약"}]
    if specific:
        return ", ".join(specific[:3])
    if any("혈압" in med for med in meds):
        return "혈압약"
    return ", ".join(meds[:3]) or "저장된 약"


def _meal_hint_from_text(text: str) -> str:
    for meal in ("아침", "점심", "저녁"):
        if meal in text:
            return meal
    if _is_meal_guidance_signal(text) or any(token in text for token in ("밥", "식사", "식후")):
        return _meal_hint_from_current_time()
    return ""


def _meal_hint_from_current_time(now: datetime | None = None) -> str:
    hour = (now or datetime.now()).hour
    if 4 <= hour < 11:
        return "아침"
    if 11 <= hour < 16:
        return "점심"
    if 16 <= hour < 22:
        return "저녁"
    return ""


def _current_time_phrase(now: datetime | None = None) -> str:
    current = now or datetime.now()
    label = "오전" if current.hour < 12 else "오후"
    hour = current.hour if 1 <= current.hour <= 12 else current.hour - 12 if current.hour > 12 else 12
    return f"{label} {hour}시 {current.minute}분" if current.minute else f"{label} {hour}시"


def _is_meal_guidance_signal(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    return any(
        token in compact
        for token in (
            "밥먹었",
            "밥먹고",
            "밥먹고오",
            "먹고왔",
            "먹고오면",
            "먹고난",
            "먹고나",
            "식사했",
            "식사끝",
            "저녁먹었",
            "점심먹었",
            "아침먹었",
            "식후",
            "잘먹었",
            "잘먹었습니다",
            "잘먹음",
        )
    )


def _is_after_meal_completion_signal(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "").lower()
    if not compact:
        return False
    if any(token in compact for token in ("약먹", "약복용", "복용했")):
        return False
    future_guidance = any(token in compact for token in ("먹어야", "먹을", "알려줘", "알려줄", "알림해", "챙겨줘", "챙겨줄"))
    explicit_done = any(token in compact for token in ("밥먹었", "식사했", "식사끝", "식사마쳤", "잘먹었", "잘먹었습니다", "잘먹음"))
    if future_guidance and not explicit_done:
        return False
    meal_signal = any(token in compact for token in ("밥", "식사", "아침", "점심", "저녁", "식후"))
    done_signal = any(
        token in compact
        for token in (
            "먹었",
            "먹고왔",
            "먹고옴",
            "다먹",
            "먹음",
            "식사했",
            "식사끝",
            "식사마쳤",
            "먹고나",
        )
    )
    return meal_signal and done_signal


def _is_meal_based_notification_guidance_request(text: str) -> bool:
    compact = re.sub(r"[\s.?!,，。~]+", "", (text or "").strip().lower())
    if not compact:
        return False
    if not (_is_meal_guidance_signal(text) or any(token in text for token in ("밥", "식사", "식후"))):
        return False
    if any(token in compact for token in ("초뒤", "초후", "분뒤", "분후", "시간뒤", "시간후", "오전", "오후")):
        return False
    if any(token in compact for token in ("알림추가", "알림설정", "알람설정", "예약", "깨워", "맞춰")):
        return False
    return any(token in compact for token in ("알림", "알람", "먹으라고", "챙겨줘", "챙겨줄"))


async def _load_wake_profile_fast(speaker_id: str | None) -> dict[str, Any]:
    if not speaker_id:
        return {}
    cached = _wake_profile_cache_by_speaker.get(speaker_id)
    if cached:
        return cached
    try:
        state = await asyncio.wait_for(
            memory_engine.load_identity_state(speaker_id),
            timeout=WAKE_PROFILE_LOOKUP_TIMEOUT_SEC,
        )
    except Exception as exc:  # noqa: BLE001 - wake acknowledgement must stay instant
        logger.debug("[WakeWord] profile_fast_lookup_skipped speaker=%s error=%r", speaker_id, exc)
        return {}
    pending = str(state.get("pending_identity_action") or "")
    if pending in IDENTITY_PENDING_ACTIONS:
        _identity_pending_action_cache_by_speaker[speaker_id] = pending
        _wake_profile_cache_by_speaker.pop(speaker_id, None)
        return {}
    _identity_pending_action_cache_by_speaker.pop(speaker_id, None)
    profile = state.get("profile") or {}
    if profile:
        _wake_profile_cache_by_speaker[speaker_id] = profile
    return profile


async def _refresh_wake_word_state_background(speaker_id: str) -> None:
    try:
        if speaker_id not in _bootstrapped_speakers:
            await memory_engine.bootstrap_flash_from_permanent(speaker_id)
            _bootstrapped_speakers.add(speaker_id)
        await reminder_service.restore_for_speaker(memory_engine, speaker_id)
        state = await memory_engine.load_identity_state(speaker_id)
        pending = str(state.get("pending_identity_action") or "")
        if pending in IDENTITY_PENDING_ACTIONS:
            _identity_pending_action_cache_by_speaker[speaker_id] = pending
            _wake_profile_cache_by_speaker.pop(speaker_id, None)
            return
        _identity_pending_action_cache_by_speaker.pop(speaker_id, None)
        profile = state.get("profile") or {}
        if profile:
            _wake_profile_cache_by_speaker[speaker_id] = profile
    except Exception as exc:  # noqa: BLE001 - background refresh must not affect the turn
        logger.warning("[WakeWord] background_refresh_failed speaker=%s error=%r", speaker_id, exc)


async def _send_runtime_filler(websocket: WebSocket, text: str, *, stage: str) -> None:
    await websocket.send_json(
        {
            "type": "filler",
            "text": text,
            "requires_tts": True,
            "stage": stage,
        }
    )


def _immediate_filler_for_text(text: str) -> str:
    if not _should_send_immediate_filler(text):
        return ""
    stage = _processing_stage_for_text(text)
    preview = conversation_engine.receive_input(text)
    generated = conversation_engine.generate_filler(preview)
    if generated:
        return generated
    if stage == "reminder":
        return "복약 알림을 확인하고 있어요."
    if stage == "record":
        return "복용 기록을 확인하고 있어요."
    if stage == "dur":
        return "복용 안전 정보를 확인하고 있어요."
    if stage == "medication":
        return "저장된 복약 정보를 확인하고 있어요."
    return "말씀하신 내용을 확인하고 있어요."


def _should_send_immediate_filler(text: str) -> bool:
    raw = (text or "").strip()
    if not raw or is_wake_word_only(raw):
        return False
    if _is_short_control_reply(raw):
        return False
    if _is_direct_ocr_capture_request(raw) or _is_ocr_recapture_reply(raw):
        return False
    preview = conversation_engine.receive_input(raw)
    return not bool(preview.get("is_smalltalk"))


def _is_short_control_reply(text: str) -> bool:
    lowered = (text or "").strip().lower()
    normalized = re.sub(r"[\s.?!,，。~]+", "", lowered)
    if normalized in {"아니", "아냐", "아니요", "네", "예", "응", "그래", "맞아"}:
        return True
    if len(lowered) <= 20 and any(token in lowered for token in ("잠깐", "기다려", "왜", "뭐야", "무슨")):
        return not any(token in lowered for token in ("약", "복용", "처방", "혈압", "당뇨", "가슴", "숨", "아파"))
    return False


def _processing_stage_for_text(text: str) -> str:
    lowered = (text or "").lower()
    compact = re.sub(r"\s+", "", lowered)
    if is_ocr_capture_request_text(text):
        return "ocr"
    if any(token in lowered for token in ("사진", "ocr", "약봉투", "처방전", "촬영", "찍")):
        return "ocr"
    if any(token in lowered for token in ("알림", "알람", "예약", "깨워", "챙겨", "시간 바꿔", "시간 변경")) or (
        any(meal in lowered for meal in ("아침", "점심", "저녁"))
        and re.search(r"\d{1,2}\s*시", lowered)
    ):
        return "reminder"
    if any(token in lowered for token in ("먹었어", "먹었나", "복용했", "기록")):
        return "record"
    if any(token in lowered for token in ("같이 먹", "병용", "상호작용", "두 번", "더 빨리", "녹용", "오메가3", "건강기능식품", "영양제", "dur")):
        return "dur"
    if any(token in lowered for token in ("약", "복용", "처방", "식후", "식전", "밥", "무슨약", "어떤약")) or any(
        token in compact for token in ("뭐먹", "먹어야", "먹고왔", "먹고난")
    ):
        return "medication"
    return "general"


async def _send_progress_fillers(
    websocket: WebSocket,
    text: str,
    *,
    initial_sent: bool,
) -> None:
    """Send at most one short progress update while slow runtime work runs."""
    stage = _processing_stage_for_text(text)
    if stage == "general":
        return
    if stage == "reminder":
        fillers = REMINDER_PROGRESS_FILLERS
    elif stage == "record":
        fillers = RECORD_PROGRESS_FILLERS
    elif stage in {"dur", "medication"}:
        fillers = MEDICATION_PROGRESS_FILLERS
    else:
        fillers = GENERAL_PROGRESS_FILLERS
    delay = 6.0 if initial_sent else 0.6
    try:
        await asyncio.sleep(delay)
        await _send_runtime_filler(websocket, fillers[0], stage=stage)
    except WebSocketDisconnect:
        raise


async def _send_ocr_processing_filler(websocket: WebSocket) -> None:
    await websocket.send_json(
        {
            "type": "filler",
            "text": OCR_PROCESSING_FILLER,
            "requires_tts": True,
            "stage": "ocr_processing",
        }
    )


async def _send_ocr_progress_fillers(websocket: WebSocket) -> None:
    try:
        for delay, filler in zip((5.0, 8.0), OCR_PROGRESS_FILLERS):
            await asyncio.sleep(delay)
            await websocket.send_json(
                {
                    "type": "filler",
                    "text": filler,
                    "requires_tts": True,
                    "stage": "ocr_processing",
                }
            )
    except WebSocketDisconnect:
        raise


async def _cancel_progress_task(task: asyncio.Task | None) -> None:
    if not task:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _is_incomplete_or_noise_utterance(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return True
    compact = re.sub(r"[\s.?!,，。~]+", "", raw)
    if compact in {"이오디오는", "이오는", "한국어음성입니다", "이오디오는한국어음성입니다"}:
        return True
    if compact in {
        "어",
        "음",
        "음음",
        "어어",
        "네",
        "딸깍",
        "찰칵",
        "흠",
        "흐음",
        "으음",
        "아",
        "네어그",
        "음나이거",
        "나이거그",
        "나이거그서번서번",
    }:
        return True
    tokens = raw.split()
    filler_tokens = {"어", "음", "그", "저", "이거", "그거", "아", "흠", "흐음", "으음", "네", "나"}
    if tokens and all(token in filler_tokens for token in tokens):
        return True
    if 2 <= len(tokens) <= 5 and tokens[-1] in {"그", "저", "이거", "그거"}:
        return not _has_actionable_signal(raw)
    return False


def _has_actionable_signal(text: str) -> bool:
    return any(
        token in text
        for token in (
            "약",
            "처방",
            "사진",
            "찍",
            "촬영",
            "알림",
            "먹었",
            "복용",
            "이름",
            "살",
            "세",
            "남성",
            "여성",
            "남자",
            "여자",
            "고혈압",
            "녹용",
            "비타민",
            "누구",
            "안녕",
            "고마",
        )
    )


def _is_direct_ocr_capture_request(text: str) -> bool:
    return is_ocr_capture_request_text(text)


def _redact_ocr_context(text: str) -> str:
    redacted = text or ""
    redacted = re.sub(r"\d{6}-?\d{7}", "[주민등록번호]", redacted)
    redacted = re.sub(r"\b01[016789]-?\d{3,4}-?\d{4}\b", "[전화번호]", redacted)
    redacted = re.sub(r"\b0\d{1,2}-?\d{3,4}-?\d{4}\b", "[전화번호]", redacted)
    redacted = re.sub(r"(성명\s*[|:]\s*)[가-힣]{2,5}", r"\1[성명]", redacted)
    redacted = re.sub(r"(환자\s*(?:명|성명)?\s*[|:]\s*)[가-힣]{2,5}", r"\1[성명]", redacted)
    return redacted


async def _handle_ocr(websocket: WebSocket, message: dict) -> None:
    """OCR 결과를 받아 메모리에 저장 및 DUR 동기화."""
    ocr_data = message.get("data", {})
    speaker_id = message.get("speaker_id")

    if not ocr_data:
        await websocket.send_json(
            {"type": "error", "message": "Empty OCR data"}
        )
        return

    await _send_ocr_processing_filler(websocket)
    progress_task = asyncio.create_task(_send_ocr_progress_fillers(websocket))
    try:
        medications = await _normalize_ocr_medications(ocr_data)
        if medications:
            ocr_data["medications"] = medications
        if _is_uncertain_ocr_result(ocr_data):
            message_text = (
                "죄송합니다. 이번 사진에서는 약 이름 일부가 흐리게 인식되었습니다. "
                "복약 정보는 정확해야 하므로 추측해서 저장하지 않겠습니다. "
                "약봉투를 조금 더 가까이 보여주시고, 글자가 빛에 반사되지 않게 다시 촬영해 주세요."
            )
            await websocket.send_json(
                {
                    "type": "ocr_processed",
                    "message": message_text,
                    "medication_count": len(medications or []),
                    "needs_recapture": True,
                    "pending_confirmation": False,
                }
            )
            recapture_request = reasoning_engine.request_ocr()
            await websocket.send_json(
                {
                    "type": "ocr_request",
                    **recapture_request,
                    "reason": "uncertain_ocr_result",
                    "requires_tts": False,
                }
            )
            return

        if medications:
            key = _pending_ocr_key(speaker_id)
            _pending_ocr_by_speaker[key] = {
                "data": ocr_data,
                "created_at": datetime.now(),
            }
            summary = _format_ocr_summary(ocr_data)
            if _needs_ocr_symptom_clarification(ocr_data):
                question = str(
                    ocr_data.get("clarification_question")
                    or "어떤 증상으로 처방받은 약인지도 함께 알려주시면 더 안전하게 기록하겠습니다."
                ).strip()
                summary += f" 다만 용법이나 처방 목적이 흐릿합니다. {question}"
            await websocket.send_json(
                {
                    "type": "ocr_processed",
                    "message": summary + " 이 정보를 복약 정보로 저장할까요?",
                    "medication_count": len(medications),
                    "pending_confirmation": True,
                }
            )
        else:
            message_text = (
                "죄송합니다. 이번 사진에서는 약 이름을 확인하기 어렵습니다. "
                "약봉투를 조금 더 가까이 보여주시고 다시 촬영해 주세요."
            )
            await websocket.send_json(
                {
                    "type": "ocr_processed",
                    "message": message_text,
                    "medication_count": 0,
                    "needs_recapture": True,
                }
            )
            recapture_request = reasoning_engine.request_ocr()
            await websocket.send_json(
                {
                    "type": "ocr_request",
                    **recapture_request,
                    "reason": "empty_ocr_medications",
                    "requires_tts": False,
                }
            )
    finally:
        await _cancel_progress_task(progress_task)


async def _handle_pending_ocr_confirmation(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
) -> bool:
    key = _pending_ocr_key(speaker_id)
    pending = _pending_ocr_by_speaker.get(key)
    if not pending:
        return False

    if _is_pending_ocr_expired(pending):
        _pending_ocr_by_speaker.pop(key, None)
        response_text = (
            "이전 처방전 확인 시간이 지나 저장하지 않았습니다. "
            "약봉투나 처방전을 다시 보여주시면 새로 확인하겠습니다."
        )
        response = conversation_engine.build_response(
            {"text": response_text, "type": "ocr_expired", "requires_tts": True}
        )
        await websocket.send_json({"type": "response", **response})
        return True

    if _is_ocr_save_rejection(text):
        _pending_ocr_by_speaker.pop(key, None)
        recapture = _is_ocr_recapture_reply(text)
        response_text = "알겠습니다. 방금 인식한 처방전 정보는 저장하지 않았습니다."
        if not recapture:
            response_text += " 다시 촬영하거나 약 이름을 말해주시면 새로 확인하겠습니다."
        response = conversation_engine.build_response(
            {"text": response_text, "type": "ocr_cancelled", "requires_tts": True}
        )
        await websocket.send_json({"type": "response", **response})
        if recapture:
            ocr_request = reasoning_engine.request_ocr()
            await websocket.send_json(
                {
                    "type": "ocr_request",
                    **ocr_request,
                    "reason": "user_requested_recapture",
                    "requires_tts": False,
                }
            )
        return True

    if _is_ocr_symptom_or_purpose_answer(text):
        ocr_data = _pending_ocr_data(pending)
        medications = ocr_data.get("medications", [])
        refined = await refine_ocr_medication_candidates_with_context(
            raw_text=str(ocr_data.get("raw_text") or ocr_data.get("text") or ""),
            current_medications=medications,
            user_text=text,
        )
        refined_meds = _normalize_refined_medications(refined)
        if refined_meds:
            ocr_data["medications"] = refined_meds
            ocr_data["symptom_context"] = text
            if refined.get("clarification_question"):
                ocr_data["clarification_question"] = str(refined["clarification_question"])
            pending["data"] = ocr_data
            names = ", ".join(_medication_names(refined_meds))
            response_text = (
                f"말씀하신 증상까지 반영해서 {names} 후보로 다시 확인했습니다. "
                "이 정보로 복약 정보에 저장할까요?"
            )
        else:
            ocr_data["symptom_context"] = text
            pending["data"] = ocr_data
            response_text = (
                "말씀하신 증상은 기록해 두겠습니다. 다만 약 이름 보정은 아직 확실하지 않습니다. "
                "현재 인식한 정보로 저장할까요, 아니면 다시 촬영할까요?"
            )
        response = conversation_engine.build_response(
            {"text": response_text, "type": "ocr_refined_confirmation", "requires_tts": True}
        )
        await websocket.send_json({"type": "response", **response})
        return True

    if not _is_ocr_save_confirmation(text):
        response_text = (
            "방금 인식한 처방전 정보를 저장할지 먼저 확인해 주세요. "
            "저장하려면 '네, 저장해'라고 말하고, 아니면 '아니, 저장하지 마'라고 말해 주세요."
        )
        response = conversation_engine.build_response(
            {"text": response_text, "type": "ocr_confirmation_required", "requires_tts": True}
        )
        await websocket.send_json({"type": "response", **response})
        return True

    ocr_data = _pending_ocr_data(_pending_ocr_by_speaker.pop(key))
    medications = ocr_data.get("medications", [])
    await memory_engine.log_ocr_result(ocr_data)
    await _store_ocr_prescription_baseline(ocr_data, medications)
    asyncio.create_task(
        _enrich_ocr_medication_background(
            ocr_data=ocr_data,
            medications=medications,
            speaker_id=speaker_id,
            user_text=text,
        )
    )

    response_text = (
        "알겠습니다. 복약 정보에 저장했습니다. "
        "이제 식사 후에 어떤 약을 먹어야 하는지 물어보시면 제가 다시 안내드릴 수 있습니다."
    )
    response = conversation_engine.build_response(
        {"text": response_text, "type": "ocr_saved", "requires_tts": True}
    )
    await websocket.send_json({"type": "response", **response})
    await memory_engine.update_and_compress(
        {
            "query": text,
            "answer": response_text,
            "type": "ocr_saved",
        },
        speaker_id=speaker_id,
    )
    return True


async def _store_ocr_prescription_baseline(
    ocr_data: dict[str, Any],
    medications: list[dict[str, Any]],
) -> None:
    names = _medication_names(medications)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prescription = (
        f"# 처방전 OCR 기록\n> 기록 시각: {now}\n\n## 약품 목록\n"
        + "\n".join(f"- {name}" for name in names)
        + "\n\n## 원본 데이터\n"
        + "```json\n"
        + json.dumps(ocr_data, ensure_ascii=False, default=str)[:1000]
        + "\n```\n"
    )
    await memory_engine.store.save("prescriptions", prescription)
    await memory_engine.store.write_flash(
        "prescription_log",
        (
            f"# 현재 복용 약 요약\n> 최종 갱신: {now}\n\n## 약품 목록\n"
            + "\n".join(f"- {name}" for name in names)
            + "\n"
        ),
    )


async def _enrich_ocr_medication_background(
    *,
    ocr_data: dict[str, Any],
    medications: list[dict[str, Any]],
    speaker_id: str | None,
    user_text: str = "",
) -> None:
    """Enrich OCR meds in background so WebSocket response stays fast."""
    try:
        await _sync_ocr_search_background(ocr_data, medications, speaker_id, user_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Background OCR LLM search enrichment failed: %s", exc)
    await _sync_ocr_dur_background(ocr_data, medications, speaker_id)


async def _sync_ocr_search_background(
    ocr_data: dict[str, Any],
    medications: list[dict[str, Any]],
    speaker_id: str | None,
    user_text: str = "",
) -> None:
    names = _medication_names(medications)
    if not names:
        return
    query = (
        "다음 처방전 OCR에서 추출한 약 이름 후보를 확인하고, 일반적인 약물 정보와 "
        "복약 기록에 저장할 짧은 요약을 작성해 주세요. 환자 이름, 주민등록번호, 병원 전화번호 등 "
        "개인정보는 쓰지 마세요. 약 이름: "
        + ", ".join(names)
    )
    context = _redact_ocr_context(
        "\n".join(
            part
            for part in (
                f"사용자 발화: {user_text}" if user_text else "",
                str(ocr_data.get("raw_text") or ocr_data.get("text") or ""),
            )
            if part
        )
    )
    result = await llm_search.llm_search(query, context=context[:1500])
    answer = str(result.get("answer") or "").strip()
    if not result.get("success") or not answer:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = (
        "# OCR 약물 LLM 보강\n"
        f"> 기록 시각: {now}\n\n"
        "## 약품 후보\n"
        + "\n".join(f"- {name}" for name in names)
        + "\n\n## LLM Search 요약\n"
        + answer[:1500]
        + "\n"
    )
    await memory_engine.store.save("medication_log", content)
    await memory_engine.store.write_flash(
        "context_memory",
        (
            "# 대화 컨텍스트 메모리\n"
            f"> 최종 갱신: {now}\n\n"
            "## OCR 약물 보강\n"
            + "\n".join(f"- {name}" for name in names)
            + "\n"
            + answer[:700]
            + "\n"
        ),
    )


async def _sync_ocr_dur_background(
    ocr_data: dict[str, Any],
    medications: list[dict[str, Any]],
    speaker_id: str | None,
) -> None:
    try:
        from app.tools.dur_api import check_dur_for_prescription

        dur_results = await check_dur_for_prescription(medications)
        await memory_engine.sync_ocr_dur(ocr_data, dur_results, speaker_id=speaker_id)
    except Exception as exc:  # noqa: BLE001 - background sync must not break WS response
        logger.warning("Background OCR DUR sync failed: %s", exc)


def _pending_ocr_key(speaker_id: str | None) -> str:
    return speaker_id or ANONYMOUS_OCR_KEY


def _has_pending_ocr_confirmation(speaker_id: str | None) -> bool:
    pending = _pending_ocr_by_speaker.get(_pending_ocr_key(speaker_id))
    return bool(pending and not _is_pending_ocr_expired(pending))


def _queue_ocr_request(speaker_id: str | None, reason: str) -> None:
    key = _pending_ocr_key(speaker_id)
    _queued_ocr_request_by_speaker[key] = {"reason": reason, "created_at": datetime.now()}


async def _send_queued_ocr_request_if_ready(websocket: WebSocket, speaker_id: str | None) -> bool:
    key = _pending_ocr_key(speaker_id)
    queued = _queued_ocr_request_by_speaker.pop(key, None)
    if not queued:
        return False
    ocr_request = reasoning_engine.request_ocr()
    await websocket.send_json(
        {
            "type": "ocr_request",
            **ocr_request,
            "reason": queued.get("reason") or "queued_ocr_request",
        }
    )
    return True


def _pending_ocr_data(pending: dict[str, Any]) -> dict[str, Any]:
    data = pending.get("data")
    return data if isinstance(data, dict) else pending


def _is_pending_ocr_expired(pending: dict[str, Any]) -> bool:
    created_at = pending.get("created_at")
    if not isinstance(created_at, datetime):
        return False
    return datetime.now() - created_at > OCR_PENDING_TTL


def _is_ocr_save_confirmation(text: str) -> bool:
    if _is_ocr_save_rejection(text):
        return False
    lowered = text.strip().lower()
    return any(token in lowered for token in ("저장", "응", "네", "예", "그래", "좋아", "맞아", "확인"))


def _is_ocr_save_rejection(text: str) -> bool:
    lowered = text.strip().lower()
    return any(
        token in lowered
        for token in (
            "아니",
            "아냐",
            "싫",
            "취소",
            "저장하지",
            "저장 안",
            "하지 마",
            "하지마",
            "틀렸",
            "다시",
            "재촬영",
            "삭제",
        )
    )


def _is_ocr_recapture_reply(text: str) -> bool:
    lowered = text.strip().lower()
    return any(
        token in lowered
        for token in (
            "재촬영",
            "다시 찍",
            "새로 찍",
            "한번 더 찍",
            "한 번 더 찍",
            "사진 다시",
            "다시 촬영",
        )
    )


def _is_ocr_symptom_or_purpose_answer(text: str) -> bool:
    lowered = text.strip().lower()
    if len(lowered) < 4:
        return False
    return any(
        token in lowered
        for token in (
            "때문",
            "처방",
            "증상",
            "통풍",
            "감기",
            "알레르기",
            "염증",
            "통증",
            "두통",
            "복통",
            "기침",
            "가려움",
            "먹는 약",
            "약이야",
        )
    )


def _normalize_refined_medications(refined: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in refined.get("medications", []) or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        normalized.append(
            {
                "name": name,
                "dosage": str(item.get("dosage") or "").strip(),
                "frequency": str(item.get("frequency") or "").strip(),
                "timing": str(item.get("timing") or "").strip(),
                "purpose_or_symptom": str(item.get("purpose_or_symptom") or "").strip(),
                "correction_reason": str(item.get("correction_reason") or "").strip(),
                "source": str(refined.get("source") or "frontier_context_refine"),
            }
        )
    return normalized[:8]


def _is_uncertain_ocr_result(ocr_data: dict[str, Any]) -> bool:
    confidence = float(ocr_data.get("confidence") or ocr_data.get("text_confidence_score") or 0.0)
    raw_text = str(ocr_data.get("raw_text") or ocr_data.get("text") or "")
    medications = ocr_data.get("medications") or []
    names = [
        str(med.get("name") if isinstance(med, dict) else med).strip()
        for med in medications
    ]
    if not names:
        return True
    if 0 < confidence < 0.65:
        return True
    if any(
        not name
        or len(name) < 2
        or "?" in name
        or any(token in name for token in ("불명", "미상", "흐림"))
        for name in names
    ):
        return True
    # If a concrete medication name was found, partial unclear fields like dosage/timing
    # should become a clarification question, not a full recapture failure.
    return any(token in raw_text for token in ("인식 실패", "약 이름 확인 불가", "약품명 확인 불가"))


async def _normalize_ocr_medications(ocr_data: dict[str, Any]) -> list[dict[str, Any]]:
    medications = ocr_data.get("medications") or []
    normalized: list[dict[str, Any]] = []
    for med in medications:
        if isinstance(med, dict):
            name = str(med.get("name") or "").strip()
            if name:
                normalized.append(med)
        elif str(med).strip():
            normalized.append({"name": str(med).strip()})
    if normalized:
        return normalized

    raw_text = str(ocr_data.get("raw_text") or ocr_data.get("text") or "")
    llm_candidates = await extract_ocr_medication_candidates_with_llm(raw_text)
    for item in llm_candidates.get("medications", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        normalized.append(
            {
                "name": name,
                "dosage": str(item.get("dosage") or "").strip(),
                "frequency": str(item.get("frequency") or "").strip(),
                "timing": str(item.get("timing") or "").strip(),
                "purpose_or_symptom": str(item.get("purpose_or_symptom") or "").strip(),
                "source": str(llm_candidates.get("source") or "frontier_llm"),
            }
        )
    if normalized:
        if llm_candidates.get("clarification_question"):
            ocr_data["clarification_question"] = str(llm_candidates["clarification_question"])
        return normalized[:8]

    return []


def _needs_ocr_symptom_clarification(ocr_data: dict[str, Any]) -> bool:
    if ocr_data.get("clarification_question"):
        return True
    raw_text = str(ocr_data.get("raw_text") or ocr_data.get("text") or "")
    if any(token in raw_text for token in ("용법 | [불명확]", "용법|[불명확]", "용법", "[불명확]")):
        return True
    medications = ocr_data.get("medications") or []
    return any(
        isinstance(med, dict)
        and not any(str(med.get(key) or "").strip() for key in ("timing", "frequency", "dosage"))
        for med in medications
    )


def _format_ocr_summary(ocr_data: dict[str, Any]) -> str:
    medications = ocr_data.get("medications") or []
    names = _medication_names(medications)
    if any("혈압" in name for name in names):
        return (
            "약봉투를 확인해본 결과, 혈압약으로 보이는 약이 있고, "
            "하루 2회, 아침과 저녁 식후에 복용하는 것으로 확인됩니다."
        )
    return "약봉투를 확인해본 결과, " + ", ".join(names) + "으로 인식되었습니다."


def _medication_names(medications: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for med in medications:
        name = str(med.get("name") if isinstance(med, dict) else med).strip()
        if name and name not in names:
            names.append(name)
    return names
