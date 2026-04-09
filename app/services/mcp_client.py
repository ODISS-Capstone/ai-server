"""MCP Host/Client: 검증 결과 전송 및 데이터 소스(MySQL·NFS·HTTPS) 연동."""
from typing import Any, Optional

import httpx

from app.core.config import settings


async def send_verified_to_mcp(
    verified_answer: str,
    session_id: str,
    meta: Optional[dict[str, Any]] = None,
) -> bool:
    """
    검증된 최종 답변을 MCP 서버로 전송.
    MCP 미설정 시 True 반환(스킵).
    """
    url = settings.mcp_server_url
    if not url:
        return True
    payload = {
        "answer": verified_answer,
        "session_id": session_id,
        "meta": meta or {},
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return True
        except Exception:
            return False


def get_mcp_data_sources() -> dict[str, str]:
    """MCP로 연결된 데이터 소스 설명 (MySQL, NFS, HTTPS)."""
    return {
        "mysql": "환자 복용 이력, 상담 로그",
        "nfs": "처방전 이미지, 분석 문서",
        "https": "실시간 약학 정보, 의학 가이드라인",
    }
