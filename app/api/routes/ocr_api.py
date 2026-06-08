"""처방전 OCR 결과 수신 HTTP 엔드포인트.

로컬 에이전트가 처방전 이미지를 OCR 처리한 결과를 서버로 전송할 때 사용.
OCR_Logging → DB_OCR_History, OCR_DUR_Interaction → 필요한 DUR만 조회 → DB_Prescription_Raw_History
"""
import logging
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.engines.memory import MemoryEngine
from app.services import ocr as ocr_service
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


class OCRImageAnalyzeResponse(OCRProcessedResponse):
    raw_text: str = ""
    response_text: str = ""
    needs_recapture: bool = False


@router.post("/analyze-image", response_model=OCRImageAnalyzeResponse)
async def analyze_ocr_image(
    file: UploadFile = File(...),
    speaker_id: Optional[str] = Form(None),
) -> OCRImageAnalyzeResponse:
    """모바일 촬영 이미지를 서버에서 OCR 처리하고 복약 응답까지 만든다."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image file required")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image")

    ocr = await ocr_service.run_ocr_image(contents, content_type=file.content_type or "image/jpeg")
    payload = OCRResultInput(
        raw_text=ocr.raw_text,
        medications=[
            MedicationItemInput(
                name=item.name,
                strength=item.strength,
                dosage=item.dosage,
                frequency=item.frequency,
                timing=item.timing,
            )
            for item in ocr.medications
        ],
        confidence=0.9 if ocr.success and ocr.raw_text.strip() else 0.0,
        speaker_id=speaker_id,
    )
    processed = await receive_ocr_result(payload)

    if not ocr.success or not payload.medications:
        response_text = (
            "죄송합니다. 이번 사진에서는 약 이름을 확인하기 어렵습니다. "
            "약봉투나 처방전을 조금 더 가까이, 빛 반사가 없게 다시 촬영해 주세요."
        )
        return OCRImageAnalyzeResponse(
            success=False,
            message=ocr.message or response_text,
            response_text=response_text,
            medication_count=0,
            dur_results=[],
            raw_text=ocr.raw_text,
            needs_recapture=True,
        )

    med_names = ", ".join(m.name for m in payload.medications[:5])
    response_text = (
        f"사진에서 {len(payload.medications)}개 약 정보를 확인했습니다. "
        f"{med_names} 정보가 보입니다. 이 정보를 복약 정보로 저장했습니다."
    )
    return OCRImageAnalyzeResponse(
        success=True,
        message=processed.message,
        response_text=response_text,
        medication_count=processed.medication_count,
        dur_results=processed.dur_results,
        raw_text=ocr.raw_text,
        needs_recapture=False,
    )


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
