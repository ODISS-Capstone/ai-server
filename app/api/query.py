"""질의·파이프라인 API: 이미지 업로드 → OCR → DUR → MD 저장 및 답변 생성."""
import logging
import uuid
from time import perf_counter
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.database.md_store import md_store
from app.engines.conversation import ConversationEngine
from app.engines.llm_judge import LLMJudgeEngine
from app.engines.memory import MemoryEngine
from app.engines.reasoning import ReasoningEngine
from app.schemas.answer import AskRequest, AskResponse
from app.schemas.ocr import OcrResponse
from app.schemas.query import PipelineResponse
from app.services import dur as dur_service
from app.services import ocr as ocr_service
from app.services.device_api import send_to_device
from app.services.documentation import build_llm_doc
from app.services.engine_orchestrator import EngineOrchestrator
from app.services.mcp_client import send_verified_to_mcp

router = APIRouter(prefix="/query", tags=["query"])
logger = logging.getLogger(__name__)
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


@router.post("/pipeline", response_model=PipelineResponse)
async def pipeline(
    file: UploadFile = File(...),
    query_text: Optional[str] = Form(None),
) -> PipelineResponse:
    """이미지 업로드 → OCR → DUR → LLM용 문서 생성 후 MD 파일로 저장."""
    started = perf_counter()
    logger.info(
        "[QueryPipeline] start filename=%s content_type=%s query_chars=%d",
        file.filename or "-",
        file.content_type or "-",
        len(query_text or ""),
    )
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image file required")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    await memory_engine.initialize()

    stage_started = perf_counter()
    ocr_result = await ocr_service.run_ocr_image(
        contents, content_type=file.content_type or "image/jpeg"
    )
    logger.info(
        "[QueryPipeline] ocr_done medications=%d elapsed_ms=%.1f",
        len(ocr_result.medications),
        (perf_counter() - stage_started) * 1000,
    )
    stage_started = perf_counter()
    dur_result = await dur_service.check_dur(ocr_result.medications)
    logger.info(
        "[QueryPipeline] dur_done items=%d elapsed_ms=%.1f",
        len(dur_result.items),
        (perf_counter() - stage_started) * 1000,
    )
    llm_doc = build_llm_doc(ocr_result, dur_result)

    session_id = str(uuid.uuid4())

    # OCR 결과 → MD 파일 저장
    await memory_engine.log_ocr_result(ocr_result.model_dump())
    await memory_engine.sync_ocr_dur(
        ocr_result.model_dump(),
        [item.model_dump() for item in dur_result.items],
    )

    # 파이프라인 세션 데이터 → MD 파일 저장
    import json
    from datetime import datetime

    session_content = (
        f"# 파이프라인 세션\n"
        f"> 세션 ID: {session_id}\n"
        f"> 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## 사용자 질문\n{query_text or '(없음)'}\n\n"
        f"## OCR 결과\n```json\n{json.dumps(ocr_result.model_dump(), ensure_ascii=False, default=str)[:2000]}\n```\n\n"
        f"## DUR 결과\n```json\n{json.dumps(dur_result.model_dump(), ensure_ascii=False, default=str)[:2000]}\n```\n\n"
        f"## LLM 문서\n{llm_doc}\n"
    )
    await md_store.save("medication_log", session_content)
    logger.info(
        "[QueryPipeline] stored session_id=%s total_elapsed_ms=%.1f",
        session_id,
        (perf_counter() - started) * 1000,
    )

    return PipelineResponse(
        session_id=session_id,
        query_text=query_text,
        ocr=ocr_result,
        dur=dur_result,
        llm_doc=llm_doc,
    )


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """추론 파이프라인: 계약형 엔진 오케스트레이션 결과를 반환."""
    started = perf_counter()
    logger.info(
        "[QueryAsk] start session_id=%s query_chars=%d device_id=%s",
        req.session_id,
        len(req.query_text or ""),
        req.device_id or "-",
    )
    await memory_engine.initialize()

    llm_doc = await _load_pipeline_llm_doc(req.session_id)

    query_text = req.query_text or ""
    if not llm_doc:
        raise HTTPException(status_code=404, detail="Session not found or no data available")
    if not query_text and llm_doc:
        query_text = "이전 세션의 복약 정보를 다시 쉽게 설명해줘."

    preloaded_context = await memory_engine.load_context(None)
    preloaded_context["prescription_log"] = llm_doc
    preloaded_context["context_memory"] = (
        f"# 파이프라인 세션 복약 문서\n> 세션 ID: {req.session_id}\n\n{llm_doc}"
    )
    preloaded_context["memory_prompt"] = llm_doc

    turn = await engine_orchestrator.run_turn(
        text=query_text,
        speaker_id=None,
        include_judge=True,
        include_delivery_llm=True,
        allow_frontier_memory_fallback=True,
        preloaded_context=preloaded_context,
    )
    answer_internal = turn.core_message
    answer_external = turn.evidence.frontier_answer_preview or None
    answer_verified = turn.reviewed_message or turn.core_message
    answer_final = turn.conversation.response_text

    sent_to_mcp = await send_verified_to_mcp(
        answer_final, req.session_id, {"query_text": query_text}
    )
    sent_to_device = False
    if req.device_id:
        sent_to_device = await send_to_device(
            req.device_id, answer_final, tts_requested=True
        )

    # 응답 결과를 MD 파일로 저장
    await memory_engine.update_and_compress({
        "query": query_text,
        "answer": answer_final,
        "type": turn.decision.intent or "ask_pipeline",
        "dur_results": turn.execution_results.get("task_results", {}).get("dur"),
        "core_message": turn.core_message,
        "judge_review": turn.judge_review,
    })
    logger.info(
        "[QueryAsk] done session_id=%s intent=%s sent_to_mcp=%s sent_to_device=%s elapsed_ms=%.1f",
        req.session_id,
        turn.decision.intent,
        sent_to_mcp,
        sent_to_device,
        (perf_counter() - started) * 1000,
    )

    return AskResponse(
        session_id=req.session_id,
        query_text=query_text,
        answer_internal=answer_internal,
        answer_external=answer_external,
        answer_verified=answer_verified,
        answer_final=answer_final,
        sent_to_mcp=sent_to_mcp,
        sent_to_device=sent_to_device,
    )


async def _load_pipeline_llm_doc(session_id: str) -> str:
    """Load the exact pipeline session document instead of relying on global flash memory."""
    hits = await md_store.search("medication_log", session_id, limit=3)
    for hit in hits:
        content = await md_store.read_entry(hit["path"])
        if f"> 세션 ID: {session_id}" not in content:
            continue
        if "## LLM 문서\n" not in content:
            continue
        return content.split("## LLM 문서\n", 1)[1].strip()
    return ""
