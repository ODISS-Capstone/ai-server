"""질의·파이프라인 API: 이미지 업로드 → OCR → DUR → MD 저장 및 답변 생성."""
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.database.md_store import md_store
from app.engines.memory import MemoryEngine
from app.schemas.answer import AskRequest, AskResponse
from app.schemas.dur import DurItem, DurResponse
from app.schemas.ocr import OcrResponse
from app.schemas.query import PipelineResponse
from app.services import dur as dur_service
from app.services import ocr as ocr_service
from app.services.censoring import extract_censored_for_external
from app.services.communication_agent import to_senior_friendly_text
from app.services.device_api import send_to_device
from app.services.documentation import build_llm_doc
from app.services.llm import call_external_llm, call_internal_llm
from app.services.mcp_client import send_verified_to_mcp
from app.services.verification import verify_answer_against_dur
from app.core.safety import ensure_disclaimer

router = APIRouter(prefix="/query", tags=["query"])
memory_engine = MemoryEngine()


@router.post("/pipeline", response_model=PipelineResponse)
async def pipeline(
    file: UploadFile = File(...),
    query_text: Optional[str] = Form(None),
) -> PipelineResponse:
    """이미지 업로드 → OCR → DUR → LLM용 문서 생성 후 MD 파일로 저장."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image file required")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    await memory_engine.initialize()

    ocr_result = await ocr_service.run_ocr_image(
        contents, content_type=file.content_type or "image/jpeg"
    )
    dur_result = await dur_service.check_dur(ocr_result.medications)
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

    return PipelineResponse(
        session_id=session_id,
        query_text=query_text,
        ocr=ocr_result,
        dur=dur_result,
        llm_doc=llm_doc,
    )


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """추론 파이프라인: 내부 LLM → 검열 → 외부 LLM → 팩트 검증 → 시니어 친화 가공 → 전송."""
    await memory_engine.initialize()

    # 최근 파이프라인 기록에서 llm_doc 및 dur 복원
    latest = await md_store.read_latest("medication_log", n=5)
    llm_doc = ""
    dur_response = DurResponse(items=[])

    for entry in latest:
        content = entry.get("content", "")
        if req.session_id in content:
            for line in content.split("\n"):
                if line.startswith("> 세션 ID:") and req.session_id in line:
                    # LLM 문서 섹션 추출
                    if "## LLM 문서" in content:
                        llm_doc = content.split("## LLM 문서\n")[-1].strip()
                    # DUR 결과 섹션 추출
                    if "## DUR 결과" in content:
                        import json as _json
                        try:
                            dur_block = content.split("## DUR 결과\n```json\n")[1].split("\n```")[0]
                            dur_dict = _json.loads(dur_block)
                            items = dur_dict.get("items", [])
                            dur_response = DurResponse(
                                items=[DurItem(**x) if isinstance(x, dict) else x for x in items]
                            )
                        except (IndexError, _json.JSONDecodeError, KeyError):
                            pass
                    break

    query_text = req.query_text or ""
    if not query_text and not llm_doc:
        raise HTTPException(status_code=404, detail="Session not found or no data available")

    memory_prompt = await memory_engine.build_query_memory_context(query_text)
    llm_doc_with_memory = llm_doc
    if memory_prompt:
        llm_doc_with_memory = (
            f"{llm_doc}\n\n[구조화 메모리]\n{memory_prompt}"
            if llm_doc
            else f"[구조화 메모리]\n{memory_prompt}"
        )

    answer_internal = await call_internal_llm(query_text, llm_doc_with_memory)
    censored = extract_censored_for_external(query_text, llm_doc, answer_internal)
    answer_external = await call_external_llm(censored)
    answer_verified, _ = verify_answer_against_dur(answer_external, dur_response)
    answer_final = ensure_disclaimer(to_senior_friendly_text(answer_verified))

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
        "type": "ask_pipeline",
    })

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
