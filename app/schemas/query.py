"""질의·세션·파이프라인 요청/응답."""
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.dur import DurResponse
from app.schemas.ocr import OcrResponse


class PipelineResponse(BaseModel):
    """이미지 업로드 → OCR → DUR → 문서화 후 저장한 결과."""
    session_id: str = Field(..., description="세션 ID")
    query_text: Optional[str] = Field(None, description="사용자 질문 텍스트")
    ocr: OcrResponse = Field(..., description="OCR 결과")
    dur: DurResponse = Field(..., description="DUR 결과")
    llm_doc: str = Field(..., description="LLM용 요약 문서")
