"""Health check endpoint."""
from fastapi import APIRouter

from app.services.frontier_llm import check_frontier_llm_health
from app.services.llm import check_internal_llm_health

router = APIRouter()


@router.get("")
async def health() -> dict[str, str]:
    """서버 및 의존성 상태 확인."""
    return {"status": "ok", "service": "odiss-medication-guidance"}


@router.get("/llm")
async def llm_health() -> dict:
    """현재 설정된 내부 LLM 서버 호출 상태를 확인."""
    internal = await check_internal_llm_health()
    frontier = await check_frontier_llm_health()
    return {
        "internal": internal,
        "frontier": frontier,
    }


@router.get("/frontier-llm")
async def frontier_llm_health() -> dict:
    """외부 frontier provider(OpenAI/Together) 설정 상태를 확인."""
    return await check_frontier_llm_health()
