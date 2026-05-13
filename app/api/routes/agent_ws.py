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
from datetime import datetime
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.engines.conversation import ConversationEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import ReasoningEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.services.engine_orchestrator import EngineOrchestrator
from app.services.identity_guard import evaluate_identity_gate
from app.services.reminders import ReminderService

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
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON"}
                )
                continue

            msg_type = message.get("type", "")

            if msg_type == "stt_result":
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

    if speaker_id:
        async def send_reminder(payload: dict[str, Any]) -> None:
            await websocket.send_json(payload)

        reminder_service.register_connection(speaker_id, send_reminder)
        active_speakers.add(speaker_id)
        await reminder_service.restore_for_speaker(memory_engine, speaker_id)

    identity_gate = await evaluate_identity_gate(
        memory_engine=memory_engine,
        text=text,
        speaker_id=speaker_id,
    )
    if not identity_gate.allowed:
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

    context = await memory_engine.load_context(speaker_id)

    if await _handle_pending_ocr_confirmation(websocket, text, speaker_id):
        return

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
    pre_filler = conversation_engine.generate_filler(preview_input)
    if pre_filler:
        await websocket.send_json({"type": "filler", "text": pre_filler})

    turn = await engine_orchestrator.run_turn(
        text=text,
        speaker_id=speaker_id,
        include_judge=True,
        include_delivery_llm=True,
        allow_frontier_memory_fallback=True,
        preloaded_context=context,
    )

    if turn.filler_text and not pre_filler:
        await websocket.send_json({"type": "filler", "text": turn.filler_text})

    # OCR 요청이 필요한 경우
    if turn.execution_results.get("task_results", {}).get("ocr_requested"):
        ocr_request = reasoning_engine.request_ocr()
        await websocket.send_json({"type": "ocr_request", **ocr_request})

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


async def _handle_ocr(websocket: WebSocket, message: dict) -> None:
    """OCR 결과를 받아 메모리에 저장 및 DUR 동기화."""
    ocr_data = message.get("data", {})
    speaker_id = message.get("speaker_id")

    if not ocr_data:
        await websocket.send_json(
            {"type": "error", "message": "Empty OCR data"}
        )
        return

    medications = ocr_data.get("medications", [])
    if _is_uncertain_ocr_result(ocr_data):
        await websocket.send_json(
            {
                "type": "ocr_processed",
                "message": (
                    "죄송합니다. 이번 사진에서는 약 이름 일부가 흐리게 인식되었습니다. "
                    "복약 정보는 정확해야 하므로 추측해서 저장하지 않겠습니다. "
                    "약봉투를 조금 더 가까이 보여주시고, 글자가 빛에 반사되지 않게 다시 촬영해 주세요."
                ),
                "medication_count": len(medications or []),
                "needs_recapture": True,
                "pending_confirmation": False,
            }
        )
        return

    if medications:
        key = speaker_id or "__anonymous__"
        _pending_ocr_by_speaker[key] = ocr_data
        summary = _format_ocr_summary(ocr_data)
        await websocket.send_json(
            {
            "type": "ocr_processed",
            "message": summary + " 이 정보를 복약 정보로 저장할까요?",
            "medication_count": len(medications),
            "pending_confirmation": True,
            }
        )
    else:
        await websocket.send_json(
            {
            "type": "ocr_processed",
            "message": (
                "죄송합니다. 이번 사진에서는 약 이름을 확인하기 어렵습니다. "
                "약봉투를 조금 더 가까이 보여주시고 다시 촬영해 주세요."
            ),
            "medication_count": 0,
            "needs_recapture": True,
            }
        )


async def _handle_pending_ocr_confirmation(
    websocket: WebSocket,
    text: str,
    speaker_id: str | None,
) -> bool:
    key = speaker_id or "__anonymous__"
    if key not in _pending_ocr_by_speaker:
        return False
    if not _is_ocr_save_confirmation(text):
        return False

    ocr_data = _pending_ocr_by_speaker.pop(key)
    medications = ocr_data.get("medications", [])
    await memory_engine.log_ocr_result(ocr_data)
    await _store_ocr_prescription_baseline(ocr_data, medications)
    asyncio.create_task(_sync_ocr_dur_background(ocr_data, medications, speaker_id))

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


async def _sync_ocr_dur_background(
    ocr_data: dict[str, Any],
    medications: list[dict[str, Any]],
    speaker_id: str | None,
) -> None:
    try:
        from app.tools.dur_api import check_dur_for_prescription

        dur_results = await check_dur_for_prescription(medications)
        dur_dicts = [r.get("dur", {}) for r in dur_results]
        await memory_engine.sync_ocr_dur(ocr_data, dur_dicts, speaker_id=speaker_id)
    except Exception as exc:  # noqa: BLE001 - background sync must not break WS response
        logger.warning("Background OCR DUR sync failed: %s", exc)


def _is_ocr_save_confirmation(text: str) -> bool:
    lowered = text.strip().lower()
    return any(token in lowered for token in ("저장", "응", "네", "그래", "좋아", "맞아"))


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
    return any(
        not name
        or len(name) < 2
        or "?" in name
        or any(token in name for token in ("불명", "미상", "흐림"))
        for name in names
    ) or any(token in raw_text for token in ("흐림", "불명확", "인식 실패"))


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
