"""Feedback capture API for the deployable ODISS web assistant."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.routes.assistant_auth import verify_assistant_web_token
from app.database.md_store import md_store

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class FeedbackLatency(BaseModel):
    stt_ms: Optional[int] = Field(None, ge=0)
    first_message_ms: Optional[int] = Field(None, ge=0)
    final_response_ms: Optional[int] = Field(None, ge=0)
    tts_ms: Optional[int] = Field(None, ge=0)


class TurnFeedbackInput(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    speaker_id: str = Field(..., min_length=1, max_length=128)
    turn_id: str = Field(..., min_length=1, max_length=128)
    rating: Literal["up", "down"]
    tags: list[str] = Field(default_factory=list, max_length=12)
    comment: str = Field("", max_length=1000)
    user_text: str = Field("", max_length=4000)
    response_text: str = Field("", max_length=8000)
    response_type: str = Field("", max_length=80)
    fast_path: str = Field("", max_length=120)
    latency: FeedbackLatency = Field(default_factory=FeedbackLatency)
    raw: dict[str, Any] = Field(default_factory=dict)
    user_agent: str = Field("", max_length=500)


class SessionFeedbackInput(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    speaker_id: str = Field(..., min_length=1, max_length=128)
    satisfaction: int = Field(..., ge=1, le=5)
    comment: str = Field("", max_length=2000)
    problem_tags: list[str] = Field(default_factory=list, max_length=16)
    turn_count: int = Field(0, ge=0)
    user_agent: str = Field("", max_length=500)


class FeedbackSavedResponse(BaseModel):
    success: bool = True
    stored_at: str
    path: str


@router.post("/turn", response_model=FeedbackSavedResponse)
async def save_turn_feedback(
    payload: TurnFeedbackInput,
    _: None = Depends(verify_assistant_web_token),
) -> FeedbackSavedResponse:
    await md_store.initialize()
    stored_at = _now()
    content = _feedback_markdown(
        title="ODISS Turn Feedback",
        stored_at=stored_at,
        payload=payload.model_dump(),
    )
    path = await md_store.save("feedback", content)
    return FeedbackSavedResponse(stored_at=stored_at, path=_relative_path(path))


@router.post("/session", response_model=FeedbackSavedResponse)
async def save_session_feedback(
    payload: SessionFeedbackInput,
    _: None = Depends(verify_assistant_web_token),
) -> FeedbackSavedResponse:
    await md_store.initialize()
    stored_at = _now()
    content = _feedback_markdown(
        title="ODISS Session Feedback",
        stored_at=stored_at,
        payload=payload.model_dump(),
    )
    path = await md_store.save("feedback", content)
    return FeedbackSavedResponse(stored_at=stored_at, path=_relative_path(path))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _feedback_markdown(*, title: str, stored_at: str, payload: dict[str, Any]) -> str:
    return (
        f"# {title}\n"
        f"> stored_at: {stored_at}\n"
        f"> session_id: {payload.get('session_id', '')}\n"
        f"> speaker_id: {payload.get('speaker_id', '')}\n\n"
        "## Summary\n"
        f"- rating: {payload.get('rating', payload.get('satisfaction', ''))}\n"
        f"- tags: {', '.join(payload.get('tags') or payload.get('problem_tags') or [])}\n\n"
        "## Comment\n"
        f"{payload.get('comment') or '(none)'}\n\n"
        "## Raw\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}\n"
        "```\n"
    )


def _relative_path(path) -> str:
    try:
        return path.relative_to(md_store.base).as_posix()
    except ValueError:
        return str(path)
