"""처방전 OCR 결과 수신 HTTP 엔드포인트.

로컬 에이전트가 처방전 이미지를 OCR 처리한 결과를 서버로 전송할 때 사용.
OCR_Logging → DB_OCR_History, OCR_DUR_Interaction → 필요한 DUR만 조회 → DB_Prescription_Raw_History
"""
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.engines.memory import MemoryEngine
from app.tools.dur_api import check_dur_for_prescription

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ocr", tags=["ocr"])

memory_engine = MemoryEngine()


class MedicationItemInput(BaseModel):
    name: str = Field(..., description="약품명")
    strength: Optional[str] = Field(None, description="용량")
    dosage: Optional[str] = Field(None, description="1회 복용량")
    frequency: Optional[str] = Field(None, description="복용 빈도")
    timing: Optional[str] = Field(None, description="복용 시점")


class OCRResultInput(BaseModel):
    raw_text: str = Field("", description="OCR 원문 텍스트")
    medications: list[MedicationItemInput] = Field(
        default_factory=list, description="구조화된 약품 목록"
    )
    confidence: float = Field(0.0, description="OCR 신뢰도 (0~1)")
    speaker_id: Optional[str] = Field(None, description="화자/복약 관리 대상자 ID")


class OCRProcessedResponse(BaseModel):
    success: bool = True
    message: str = ""
    medication_count: int = 0
    dur_results: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/analyze", response_model=OCRProcessedResponse)
async def receive_ocr_result(payload: OCRResultInput) -> OCRProcessedResponse:
    """처방전 OCR 결과를 수신하여 로깅 및 DUR 동기화."""
    await memory_engine.initialize()

    ocr_data = {
        "raw_text": payload.raw_text,
        "medications": [m.model_dump() for m in payload.medications],
        "confidence": payload.confidence,
    }

    # OCR_Logging → DB_OCR_History
    await memory_engine.log_ocr_result(ocr_data, payload.confidence)

    # OCR_DUR_Interaction → 기본 T4 품목 정보만 우선 조회
    dur_results: list[dict[str, Any]] = []
    if payload.medications:
        med_dicts = [m.model_dump() for m in payload.medications]
        dur_results = await check_dur_for_prescription(med_dicts)

        await memory_engine.sync_ocr_dur(
            ocr_data,
            dur_results,
            speaker_id=payload.speaker_id,
        )

    return OCRProcessedResponse(
        success=True,
        message=f"{len(payload.medications)}개 약품 처리 완료",
        medication_count=len(payload.medications),
        dur_results=dur_results,
    )


@router.get("/history")
async def get_ocr_history() -> dict[str, Any]:
    """OCR 이력 조회."""
    await memory_engine.initialize()
    history = await memory_engine.store.read_latest("ocr_history", n=20)
    return {"history": history}
