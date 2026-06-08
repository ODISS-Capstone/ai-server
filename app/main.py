"""ODISS 서버엔진 — FastAPI 애플리케이션 엔트리포인트.

클라우드 AI 서버 아키텍처:
  - 대화 엔진 (Conversation Engine): 페르소나 및 인터페이스
  - 메모리 엔진 (Memory Engine): 지식 창고 및 사서
  - 추론 엔진 (Reasoning Engine): 지휘통제실
  - LLM as a Judge Engine: 성능 증강 및 검증
  - Tool Execution Engine: DUR, HIRA, 건기식 등 외부 API

데이터 저장소: MD 파일시스템 기반 (SQL 미사용)
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import dur, health, query, upload
from app.api.routes import (
    agent_ws,
    device_api,
    feedback_api,
    memory_browser_api,
    ocr_api,
    stt_api,
)
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.database.md_store import md_store
from app.engines.memory import MemoryEngine
from app.services import turboquant_runtime
from app.services.llm import get_internal_llm_provider
from app.services.whisper_stt import preload_whisper_model

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: 디렉토리 생성, MD 파일시스템 데이터베이스 초기화."""
    logger.info("ODISS 서버엔진 시작 중...")
    logger.info(
        "로그 설정 완료: level=%s file_enabled=%s file_path=%s",
        settings.log_level.upper(),
        settings.log_to_file,
        settings.log_file_path,
    )
    logger.info(
        "[InternalLLM] config provider=%s url=%s model=%s api_key_set=%s",
        get_internal_llm_provider(),
        settings.internal_llm_api_url or "-",
        settings.internal_llm_model,
        bool(settings.internal_llm_api_key),
    )

    os.makedirs("data", exist_ok=True)
    os.makedirs(settings.storage_path, exist_ok=True)

    await md_store.initialize()
    logger.info("MD Database Layer 초기화 완료 (%s)", settings.md_database_path)

    await MemoryEngine().bootstrap_flash_from_permanent()
    logger.info("Flash Memory bootstrap 완료 (no active dialogue)")

    turboquant_runtime.install()

    if (settings.stt_provider or "").strip().lower() == "whisper":
        try:
            await preload_whisper_model()
            logger.info(
                "[WhisperSTT] preload complete model=%s device=%s compute_type=%s",
                settings.whisper_model,
                settings.whisper_device,
                settings.whisper_compute_type,
            )
        except Exception as exc:  # noqa: BLE001 - keep server bootable for diagnostics
            logger.warning("[WhisperSTT] preload failed: %s", exc)

    logger.info("ODISS 서버엔진 준비 완료")
    yield
    logger.info("ODISS 서버엔진 종료")


app = FastAPI(
    title="ODISS 서버엔진 — OCR 기반 멀티모달 복약관리",
    description=(
        "추론 엔진, 메모리 엔진, 대화 엔진, LLM Judge 기반의 "
        "만성질환자와 복약 관리가 필요한 사용자를 위한 복약 상담 AI 서버입니다. "
        "WebSocket을 통해 로컬 에이전트와 실시간 통신합니다.\n\n"
        "데이터 저장소: MD 파일시스템 (날짜별 디렉토리 + 개별 Markdown 파일)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in settings.memory_browser_cors_origins.split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

# ── 기존 HTTP API 라우터 (하위 호환) ──
app.include_router(health.router, prefix="/health")
app.include_router(upload.router)
app.include_router(dur.router)
app.include_router(query.router)

# ── ODISS 신규 엔드포인트 ──
app.include_router(agent_ws.router, tags=["websocket"])
app.include_router(ocr_api.router)
app.include_router(stt_api.router)
app.include_router(device_api.router)
app.include_router(memory_browser_api.router)
app.include_router(feedback_api.router)

web_dist = Path(settings.assistant_web_dist_path)
if web_dist.exists() and web_dist.is_dir():
    app.mount("/app", StaticFiles(directory=str(web_dist), html=True), name="assistant-web")
    web_assets = web_dist / "assets"
    if web_assets.exists() and web_assets.is_dir():
        # Backward-compatible fallback for older/cached web builds that referenced
        # assets from the domain root instead of /app/assets.
        app.mount("/assets", StaticFiles(directory=str(web_assets)), name="assistant-web-assets")
else:
    logger.info("Assistant web dist not mounted; directory not found: %s", web_dist)


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "ODISS — 만성질환 복약관리 AI 서버엔진",
        "version": "1.0.0",
        "database": "MD filesystem (no SQL)",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws/chat",
        "engines": [
            "Conversation Engine",
            "Memory Engine",
            "Reasoning Engine",
            "LLM as a Judge Engine",
            "Tool Execution Engine",
        ],
    }
