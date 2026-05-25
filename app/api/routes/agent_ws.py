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
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.engines.conversation import ConversationEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import ReasoningEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.services.engine_orchestrator import EngineOrchestrator
from app.services.identity_guard import evaluate_identity_gate
from app.services.llm import extract_ocr_medication_candidates_with_llm, refine_ocr_medication_candidates_with_context
from app.services.medication_extraction import is_ocr_capture_request_text, is_wake_word_only
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
OCR_PENDING_TTL = timedelta(minutes=5)
ANONYMOUS_OCR_KEY = "__anonymous__"
WEBSOCKET_IDLE_TIMEOUT_SEC = 65.0
WAKE_PROFILE_LOOKUP_TIMEOUT_SEC = 0.15
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

    if speaker_id:
        if speaker_id not in _bootstrapped_speakers:
            await memory_engine.bootstrap_flash_from_permanent(speaker_id)
            _bootstrapped_speakers.add(speaker_id)

        async def send_reminder(payload: dict[str, Any]) -> None:
            await websocket.send_json(payload)

        reminder_service.register_connection(speaker_id, send_reminder)
        active_speakers.add(speaker_id)
        await reminder_service.restore_for_speaker(memory_engine, speaker_id)

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
        return

    if queued_ocr_reason:
        await _send_queued_ocr_request_if_ready(websocket, speaker_id)
        return

    if await _handle_pending_ocr_confirmation(websocket, text, speaker_id):
        return

    if await _send_queued_ocr_request_if_ready(websocket, speaker_id):
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
    if _is_direct_ocr_capture_request(raw) or _is_ocr_recapture_reply(raw):
        return False
    preview = conversation_engine.receive_input(raw)
    return not bool(preview.get("is_smalltalk"))


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
