"""OCR 요청/응답 스키마."""
from typing import Optional

from pydantic import BaseModel, Field


class MedicationItem(BaseModel):
    """OCR로 추출된 단일 약품 정보."""
    name: str = Field(..., description="약품명")
    strength: Optional[str] = Field(None, description="용량 (예: 2mg, 60mg)")
    dosage: Optional[str] = Field(None, description="1회 복용량 (예: 0.5정, 1.0정)")
    frequency: Optional[str] = Field(None, description="복용 빈도 (예: 1일 3회)")
    timing: Optional[str] = Field(None, description="복용 시점 (예: 식후 30분)")
    raw_line: Optional[str] = Field(None, description="원문 한 줄")


class OcrResponse(BaseModel):
    """OCR 결과."""
    raw_text: str = Field(..., description="추출된 전체 텍스트")
    medications: list[MedicationItem] = Field(default_factory=list, description="구조화된 약품 목록")
    success: bool = True
    message: Optional[str] = None
