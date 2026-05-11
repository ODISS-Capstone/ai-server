"""Health check endpoint."""
from fastapi import APIRouter

from app.services.llm import check_internal_llm_health

router = APIRouter()


@router.get("")
async def health() -> dict[str, str]:
    """서버 및 의존성 상태 확인."""
    return {"status": "ok", "service": "senior-medication-guidance"}


@router.get("/llm")
async def llm_health() -> dict:
    """현재 설정된 내부 LLM 서버 호출 상태를 확인."""
    return await check_internal_llm_health()
