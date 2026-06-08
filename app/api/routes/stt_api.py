"""로컬 에이전트 STT(Instruction_Log) 수신 HTTP 엔드포인트.

server.mermaid 매핑::

    STT --> Instruction_Log --> DB_Medication_Log / ocr_history sidecar

로컬 에이전트는 사용자 발화를 ``Instruction_Log`` 노드를 통해 비동기로
클라우드에 보낸다.  여기서는 가벼운 감사 로그 저장만 담당하고,
대화 파이프라인은 ``/ws/chat`` WebSocket의 ``stt_result`` 메시지가
맡는다.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import settings
from app.database.md_store import md_store
from app.services.gemini_stt import GeminiSttError, transcribe_audio_with_gemini
from app.services.whisper_stt import WhisperSttError, transcribe_audio_with_whisper

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stt", tags=["stt"])


class InstructionLogInput(BaseModel):
    text: str = Field(..., description="STT 변환 텍스트")
    timestamp: Optional[float] = Field(None, description="에이전트 측 epoch 초")
    source: str = Field("stt", description="원본 소스 (기본값 stt)")
    speaker_id: Optional[str] = Field(None, description="화자/복약 관리 대상자 ID")


class InstructionLogResponse(BaseModel):
    success: bool = True
    received_chars: int = 0
    stored_at: str = ""


class SttTranscribeResponse(BaseModel):
    success: bool = True
    text: str = ""
    provider: str = "gemini"
    model: str = ""
    audio_bytes: int = 0


@router.post("/transcribe", response_model=SttTranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    speaker_id: Optional[str] = Form(None),
    language: str = Form("ko-KR"),
) -> SttTranscribeResponse:
    """모바일 음성 파일을 서버 STT provider로 전사한다.

    ``STT_PROVIDER=whisper``이면 로컬 faster-whisper 모델을 사용하고,
    그 외에는 기존 Gemini STT를 사용한다.
    """
    audio = await file.read()
    mime_type = file.content_type or "audio/mp4"
    provider = (settings.stt_provider or "gemini").strip().lower()
    try:
        if provider == "whisper":
            text = await transcribe_audio_with_whisper(audio, language=language)
        else:
            provider = "gemini"
            text = await transcribe_audio_with_gemini(audio, mime_type=mime_type, language=language)
    except GeminiSttError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except WhisperSttError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    logger.info(
        "[STT] provider=%s model=%s speaker=%s audio_bytes=%d chars=%d text=%r",
        provider,
        settings.whisper_model if provider == "whisper" else settings.gemini_stt_model,
        speaker_id or "",
        len(audio),
        len(text.strip()),
        text.strip()[:200],
    )
    if text.strip():
        await ingest_instruction_log(
            InstructionLogInput(
                text=text.strip(),
                source=f"android_{provider}_stt",
                speaker_id=speaker_id,
            )
        )
    return SttTranscribeResponse(
        success=True,
        text=text.strip(),
        provider=provider,
        model=settings.whisper_model if provider == "whisper" else settings.gemini_stt_model,
        audio_bytes=len(audio),
    )


@router.post("/log", response_model=InstructionLogResponse)
async def ingest_instruction_log(payload: InstructionLogInput) -> InstructionLogResponse:
    """에이전트의 ``Instruction_Log --> DB`` 엣지를 수용한다."""
    await md_store.initialize()

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    content = (
        "# STT Instruction Log\n"
        f"> 기록 시각: {now}\n"
        f"> 소스: {payload.source}\n"
        f"> 화자: {payload.speaker_id or '미지정'}\n"
        f"> agent 타임스탬프: {payload.timestamp}\n\n"
        "## 전사 텍스트\n"
        f"{payload.text}\n\n"
        "## raw\n"
        "```json\n"
        f"{json.dumps(payload.model_dump(), ensure_ascii=False)}\n"
        "```\n"
    )

    await md_store.save("stt_instruction_log", content)

    return InstructionLogResponse(
        success=True,
        received_chars=len(payload.text),
        stored_at=now,
    )
