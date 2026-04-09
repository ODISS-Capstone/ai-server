"""음성(STT) 요청/응답 스키마."""
from typing import Optional

from pydantic import BaseModel, Field


class SttResponse(BaseModel):
    """STT 결과."""
    text: str = Field(..., description="인식된 질문 텍스트")
    success: bool = True
    message: Optional[str] = None
