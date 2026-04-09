"""답변 생성 파이프라인 요청/응답."""
from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """질의 답변 요청."""
    session_id: str = Field(..., description="파이프라인으로 생성된 세션 ID")
    query_text: Optional[str] = Field(None, description="사용자 질문 (없으면 기존 로그에서 사용)")
    device_id: Optional[str] = Field(None, description="답변 전달할 단말 ID (선택)")


class AskResponse(BaseModel):
    """답변 결과."""
    session_id: str
    query_text: str
    answer_internal: Optional[str] = None
    answer_external: Optional[str] = None
    answer_verified: str = Field(..., description="팩트 검증 후 답변")
    answer_final: str = Field(..., description="시니어 친화 최종 답변")
    sent_to_mcp: bool = False
    sent_to_device: bool = False
