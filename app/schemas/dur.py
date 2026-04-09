"""DUR(의약품 안전사용) API 요청/응답 스키마."""
from typing import Optional

from pydantic import BaseModel, Field


class DurItem(BaseModel):
    """단일 약품에 대한 DUR 검증 결과."""
    name: str = Field(..., description="약품명")
    ingredient: Optional[str] = Field(None, description="성분명")
    efficacy: Optional[str] = Field(None, description="효능")
    contraindications: list[str] = Field(default_factory=list, description="금기 사항")
    interactions: list[str] = Field(default_factory=list, description="병용 금기/상호작용")
    precautions: list[str] = Field(default_factory=list, description="주의사항")
    verified: bool = Field(True, description="DB 매칭 여부")


class DurResponse(BaseModel):
    """DUR 일괄 조회 결과."""
    items: list[DurItem] = Field(default_factory=list)
    success: bool = True
    message: Optional[str] = None
