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
from typing import Any

from fastapi import FastAPI

from app.api import dur, health, query, upload
from app.api.routes import agent_ws, ocr_api, stt_api
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.database.md_store import md_store
from app.services import turboquant_runtime

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
        "[InternalLLM] config url=%s model=%s api_key_set=%s",
        settings.internal_llm_api_url or "-",
        settings.internal_llm_model,
        bool(settings.internal_llm_api_key),
    )

    os.makedirs("data", exist_ok=True)
    os.makedirs(settings.storage_path, exist_ok=True)

    await md_store.initialize()
    logger.info("MD Database Layer 초기화 완료 (%s)", settings.md_database_path)

    turboquant_runtime.install()

    logger.info("ODISS 서버엔진 준비 완료")
    yield
    logger.info("ODISS 서버엔진 종료")


app = FastAPI(
    title="ODISS 서버엔진 — OCR 기반 멀티모달 시니어 복약지도",
    description=(
        "추론 엔진, 메모리 엔진, 대화 엔진, LLM Judge 기반의 "
        "시니어 복약 상담 AI 서버입니다. "
        "WebSocket을 통해 로컬 에이전트와 실시간 통신합니다.\n\n"
        "데이터 저장소: MD 파일시스템 (날짜별 디렉토리 + 개별 Markdown 파일)"
    ),
    version="1.0.0",
    lifespan=lifespan,
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


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "ODISS — 시니어 복약지도 AI 서버엔진",
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
