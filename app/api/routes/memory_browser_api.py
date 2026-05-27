"""Read-only API for browsing patient memory records."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.core.config import settings
from app.api.routes.assistant_auth import validate_assistant_token
from app.services.memory_browser import MemoryBrowserService

router = APIRouter(prefix="/api/memory", tags=["memory-browser"])
memory_browser = MemoryBrowserService()


async def verify_memory_browser_token(
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
        raise HTTPException(status_code=503, detail="Memory browser token is not configured")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if not validate_assistant_token(token, include_memory_browser=True):
        raise HTTPException(status_code=403, detail="Invalid bearer token")


@router.get("/patients")
async def search_patients(
    name: str = Query(..., min_length=1, description="환자명 (부분 일치)"),
    limit: int = Query(20, ge=1, le=100),
    _: None = Depends(verify_memory_browser_token),
) -> dict:
    await memory_browser.initialize()
    patients = await memory_browser.search_patients(name, limit=limit)
    return {"query": name, "patients": patients, "total": len(patients)}


@router.get("/patients/{speaker_id}")
async def get_patient_detail(
    speaker_id: str,
    _: None = Depends(verify_memory_browser_token),
) -> dict:
    await memory_browser.initialize()
    if not await memory_browser.store.user_exists(speaker_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    return await memory_browser.get_patient_detail(speaker_id)


@router.get("/patients/{speaker_id}/records")
async def get_patient_records(
    speaker_id: str,
    categories: Optional[str] = Query(
        None,
        description="Comma-separated categories, e.g. ocr_history,prescriptions",
    ),
    query: str = Query("", description="추가 검색어"),
    start: Optional[date] = Query(None),
    end: Optional[date] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    _: None = Depends(verify_memory_browser_token),
) -> dict:
    await memory_browser.initialize()
    if not await memory_browser.store.user_exists(speaker_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    category_list = [item.strip() for item in (categories or "").split(",") if item.strip()]
    return await memory_browser.search_patient_records(
        speaker_id,
        categories=category_list or None,
        query=query,
        start=start,
        end=end,
        limit=limit,
    )


@router.get("/entry")
async def read_memory_entry(
    path: str = Query(..., min_length=1, description="Relative markdown path"),
    _: None = Depends(verify_memory_browser_token),
) -> dict:
    await memory_browser.initialize()
    try:
        return await memory_browser.read_entry(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
