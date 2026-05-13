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
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.engines.conversation import ConversationEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import ReasoningEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.services.engine_orchestrator import EngineOrchestrator
from app.services.identity_guard import evaluate_identity_gate

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
    """
    await websocket.accept()
    logger.info("WebSocket connected")

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
                await _handle_stt(websocket, message)
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


async def _handle_stt(websocket: WebSocket, message: dict) -> None:
    """STT 결과를 받아 전체 파이프라인 실행."""
    text = message.get("text", "").strip()
    speaker_id = message.get("speaker_id")

    if not text:
        await websocket.send_json(
            {"type": "error", "message": "Empty text"}
        )
        return

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

    turn = await engine_orchestrator.run_turn(
        text=text,
        speaker_id=speaker_id,
        include_judge=True,
        include_delivery_llm=True,
        allow_frontier_memory_fallback=True,
    )

    if turn.filler_text:
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

    # OCR_Logging: OCRHistory.md에 기록
    await memory_engine.log_ocr_result(ocr_data)

    # OCR_DUR_Interaction: 처방전 약품에 대해 DUR 동기화
    medications = ocr_data.get("medications", [])
    if medications:
        from app.tools.dur_api import check_dur_for_prescription

        dur_results = await check_dur_for_prescription(medications)
        dur_dicts = [r.get("dur", {}) for r in dur_results]
        await memory_engine.sync_ocr_dur(ocr_data, dur_dicts, speaker_id=speaker_id)

        await websocket.send_json(
            {
            "type": "ocr_processed",
            "message": f"{len(medications)}개 약품의 DUR 확인이 완료되었습니다.",
            "medication_count": len(medications),
            "dur_check_count": len(dur_results),
            }
        )
    else:
        await websocket.send_json(
            {
            "type": "ocr_processed",
            "message": "OCR 결과가 저장되었습니다.",
            "medication_count": 0,
            }
        )
