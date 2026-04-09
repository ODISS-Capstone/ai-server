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

logger = logging.getLogger(__name__)
router = APIRouter()

memory_engine = MemoryEngine()
llm_judge = LLMJudgeEngine()
reasoning_engine = ReasoningEngine(memory_engine, llm_judge)
conversation_engine = ConversationEngine()


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

    # 1. CE_Input: 입력 수신 및 초기 분석
    input_data = conversation_engine.receive_input(text, speaker_id)

    # 2. CE_Latency: 즉시 filler 응답 (Latency Hiding)
    filler = conversation_engine.generate_filler(input_data)
    if filler:
        await websocket.send_json({"type": "filler", "text": filler})

    # 3. 스몰토크만인 경우 바로 응답
    if input_data["is_smalltalk"]:
        intent = reasoning_engine.classify_intent(text)
        if intent == "smalltalk":
            result = conversation_engine.synthesize_response(input_data)
            response = conversation_engine.build_response(result)
            await websocket.send_json({"type": "response", **response})
            return

    # 4. ME_Context: 사용자 식별 및 컨텍스트 로드
    context = await memory_engine.load_context(speaker_id)

    # 5. RE_Intent: 의도 파악 및 태스크 설계
    intent = reasoning_engine.classify_intent(text)
    tasks = reasoning_engine.plan_tasks(intent, context)

    # 6. 태스크 실행 (DUR, HIRA, RAG 등)
    execution_results = await reasoning_engine.execute_tasks(
        text, intent, context, tasks
    )

    # 6-1. OCR 요청이 필요한 경우
    if execution_results.get("task_results", {}).get("ocr_requested"):
        ocr_request = reasoning_engine.request_ocr()
        await websocket.send_json(
            {"type": "ocr_request", **ocr_request}
        )

    # 7. RE_Core_Msg: 핵심 답변 생성
    core_message = await reasoning_engine.synthesize_core_message(
        execution_results
    )

    # 8. CE_Tone + CE_Conversation_Core: 톤앤매너 적용 및 최종 합성
    flash_context = context.get("context_memory", "")
    user_profile = context.get("user_profile")

    synthesis = conversation_engine.synthesize_response(
        input_data,
        fact_data=core_message,
        filler_sent=True,
        user_profile=user_profile,
        flash_context=flash_context,
    )

    # 9. CE_Response: 최종 응답 빌드 및 전송
    response = conversation_engine.build_response(synthesis)
    await websocket.send_json({"type": "response", **response})

    # 10. ME_Update: 결과 저장 및 Flash Memory 압축
    await memory_engine.update_and_compress(
        {
            "query": text,
            "answer": synthesis["text"],
            "type": intent,
            "dur_results": execution_results.get("task_results", {}).get("dur"),
        },
        speaker_id=speaker_id,
    )


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
        await memory_engine.sync_ocr_dur(ocr_data, dur_dicts)

        await websocket.send_json({
            "type": "ocr_processed",
            "message": f"{len(medications)}개 약품의 DUR 확인이 완료되었습니다.",
            "medication_count": len(medications),
            "dur_check_count": len(dur_results),
        })
    else:
        await websocket.send_json({
            "type": "ocr_processed",
            "message": "OCR 결과가 저장되었습니다.",
            "medication_count": 0,
        })
