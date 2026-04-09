"""이미지/음성 업로드 및 OCR·STT 연동 API."""
from fastapi import APIRouter, File, UploadFile, HTTPException

from app.schemas.ocr import OcrResponse
from app.schemas.voice import SttResponse
from app.services import ocr as ocr_service
from app.services import voice as voice_service

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/image", response_model=OcrResponse)
async def upload_image_for_ocr(file: UploadFile = File(...)) -> OcrResponse:
    """처방전 또는 약 봉투 이미지를 업로드하면 OCR로 텍스트 추출 후 구조화된 약품 목록 반환."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Image file required (e.g. image/jpeg, image/png)")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    return await ocr_service.run_ocr_image(contents, content_type=file.content_type or "image/jpeg")


@router.post("/voice", response_model=SttResponse)
async def upload_voice_for_stt(file: UploadFile = File(...)) -> SttResponse:
    """음성 파일을 업로드하면 STT로 질문 텍스트 반환."""
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Audio file required (e.g. audio/wav, audio/mpeg)")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    return await voice_service.run_stt(contents, content_type=file.content_type or "audio/wav")
