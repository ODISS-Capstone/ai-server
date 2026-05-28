"""Shared token checks for the deployable assistant web app."""
from __future__ import annotations

from typing import Annotated, Optional

from fastapi import Header, HTTPException, WebSocket

from app.core.config import settings


def _configured_tokens(*, include_memory_browser: bool = False) -> set[str]:
    tokens = {str(settings.assistant_web_token or "").strip()}
    if include_memory_browser:
        tokens.add(str(settings.memory_browser_token or "").strip())
    return {token for token in tokens if token}


def _is_local_env() -> bool:
    return settings.app_env.lower() in {"development", "dev", "local"}


def _extract_bearer(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        return ""
    return authorization.removeprefix("Bearer ").strip()


def validate_assistant_token(
    token: str,
    *,
    include_memory_browser: bool = False,
) -> bool:
    """Return whether a token is allowed for browser-facing assistant APIs."""
    if _is_local_env() and not token:
        return True
    configured = _configured_tokens(include_memory_browser=include_memory_browser)
    if not configured:
        return _is_local_env()
    return token in configured


async def verify_assistant_web_token(
    authorization: Annotated[Optional[str], Header()] = None,
) -> None:
    bearer_token = _extract_bearer(authorization)
    if _is_local_env() and not bearer_token:
        return
    configured = _configured_tokens()
    if not configured:
        if _is_local_env():
            return
        raise HTTPException(status_code=503, detail="Assistant web token is not configured")
    if bearer_token not in configured:
        raise HTTPException(status_code=401, detail="Missing or invalid assistant token")


def websocket_assistant_token_allowed(websocket: WebSocket) -> bool:
    token = str(websocket.query_params.get("token") or "").strip()
    return validate_assistant_token(token)
