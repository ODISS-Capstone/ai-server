"""사용자 식별 레지스트리 조회/등록 API.

파일 기반 리스트(identity_registry.json)에 정리된 사용자(speaker) 목록을
관리자 토큰으로 조회하고, 웹 등 클라이언트가 최초 1회 명시적으로 등록(touch)할
수 있는 엔드포인트를 제공한다.
"""
from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.core.config import settings
from app.api.routes.assistant_auth import validate_assistant_token
from app.services.identity_registry import get_identity, list_identities, touch_identity

router = APIRouter(prefix="/api/identity", tags=["identity"])


async def verify_admin_token(
    authorization: Annotated[Optional[str], Header()] = None,
) -> None:
    if settings.app_env.lower() in {"development", "dev", "local"} and not authorization:
        return
    expected_tokens = {
        token
        for token in (
            (settings.memory_browser_token or "").strip(),
            (settings.assistant_web_token or "").strip(),
        )
        if token
    }
    if not expected_tokens:
        if settings.app_env.lower() in {"development", "dev", "local"}:
            return
        raise HTTPException(status_code=503, detail="Admin token is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if not validate_assistant_token(token, include_memory_browser=True):
        raise HTTPException(status_code=403, detail="Invalid bearer token")


class IdentityTouchRequest(BaseModel):
    speaker_id: str = Field(..., min_length=1, description="클라이언트 영구 키")
    platform: str = Field("unknown", description="android | web | unknown")
    app_version: str = Field("", description="앱/클라이언트 버전")
    display_name: str = Field("", description="표시 이름(선택)")


class IdentityTouchResponse(BaseModel):
    success: bool = True
    speaker_id: str
    is_new: bool


def _client_ip(request: Request) -> str:
    xff = str(request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if xff:
        return xff
    return request.client.host if request.client else ""


@router.post("/touch", response_model=IdentityTouchResponse)
async def touch_endpoint(payload: IdentityTouchRequest, request: Request) -> IdentityTouchResponse:
    """클라이언트가 자신의 영구 키를 명시적으로 등록/갱신한다(인증 불필요)."""
    _, is_new = touch_identity(
        payload.speaker_id,
        platform=payload.platform or "unknown",
        ip=_client_ip(request),
        app_version=payload.app_version,
        source="touch_api",
        display_name=payload.display_name,
    )
    return IdentityTouchResponse(success=True, speaker_id=payload.speaker_id, is_new=is_new)


@router.get("/list")
async def list_endpoint(_: None = Depends(verify_admin_token)) -> dict:
    records = list_identities()
    return {
        "total": len(records),
        "identities": [record.to_dict() for record in records],
    }


@router.get("/{speaker_id}")
async def get_endpoint(speaker_id: str, _: None = Depends(verify_admin_token)) -> dict:
    record = get_identity(speaker_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Identity not found")
    return record.to_dict()
