"""Health check endpoint."""
from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def health() -> dict[str, str]:
    """서버 및 의존성 상태 확인."""
    return {"status": "ok", "service": "senior-medication-guidance"}
