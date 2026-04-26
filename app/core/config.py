"""ODISS 서버엔진 설정 — 환경 변수 기반."""
from functools import lru_cache
from typing import Any, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    database_url: str = "sqlite+aiosqlite:///./data/senior_med.db"
    md_database_path: str = "./data/md_database"
    structured_memory_path: str = "./data/md_database/structured_memory"

    # OCR — DeepSeek
    deepseek_ocr_api_url: Optional[str] = None
    deepseek_ocr_api_key: Optional[str] = None

    # 공공데이터포털 서비스 키 (data.go.kr)
    data_go_kr_service_key: Optional[str] = None

    # DUR API (식약처 — T2~T10)
    dur_api_base_url: str = "http://apis.data.go.kr/1471000"

    # 의약품 낱알식별 API (HIRA — T1)
    hira_api_base_url: str = "http://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03"

    # 건강기능식품 API (T11, T12)
    health_supplement_api_base_url: str = "http://apis.data.go.kr/1471000/HtfsSttusIdntfcInfoService01"

    # KPIC DUR
    kpic_dur_api_url: Optional[str] = None
    kpic_dur_api_key: Optional[str] = None

    # Internal LLM (Qwen / EXAONE)
    internal_llm_api_url: Optional[str] = None
    internal_llm_api_key: Optional[str] = None

    # External LLM (OpenAI — LLM as a Judge + LLM Search)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    google_ai_api_key: Optional[str] = None

    # MCP
    mcp_server_url: Optional[str] = None
    mcp_transport: str = "stdio"

    # File storage
    storage_path: str = "./data/storage"
    nfs_mount_path: Optional[str] = None

    # TTS / STT (optional — Clova)
    clova_stt_client_id: Optional[str] = None
    clova_stt_client_secret: Optional[str] = None
    clova_tts_client_id: Optional[str] = None
    clova_tts_client_secret: Optional[str] = None

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production", "false", "0", "off", "no"}:
                return False
            if normalized in {"debug", "dev", "development", "true", "1", "on", "yes"}:
                return True
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
