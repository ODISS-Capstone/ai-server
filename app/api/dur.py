"""DUR(의약품 안전사용) 검증 API."""
from fastapi import APIRouter, HTTPException

from app.schemas.dur import DurResponse
from app.schemas.ocr import MedicationItem
from app.services import dur as dur_service

router = APIRouter(prefix="/dur", tags=["dur"])


@router.post("", response_model=DurResponse)
async def check_dur_api(medications: list[MedicationItem]) -> DurResponse:
    """OCR로 추출한 약품 목록을 DUR API에 보내 검증·금기·상호작용 조회."""
    if not medications:
        raise HTTPException(status_code=400, detail="medications list required")
    return await dur_service.check_dur(medications)
