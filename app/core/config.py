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
    log_to_file: bool = True
    log_file_path: str = "./logs/ai-server.log"

    database_url: str = "sqlite+aiosqlite:///./data/senior_med.db"
    md_database_path: str = "./data/md_database"
    structured_memory_path: str = "./data/md_database/structured_memory"

    # Identity gate
    identity_reverify_window_seconds: int = 60 * 60
    identity_pending_timeout_seconds: int = 5 * 60

    # OCR — DeepSeek
    deepseek_ocr_api_url: Optional[str] = None
    deepseek_ocr_api_key: Optional[str] = None

    # 공공데이터포털 서비스 키 (data.go.kr)
    data_go_kr_service_key: Optional[str] = None

    # DUR API (식약처 — T2~T10)
    dur_api_base_url: str = "http://apis.data.go.kr/1471000"
    dur_api_timeout_seconds: float = 8.0
    dur_api_max_concurrency: int = 8

    # 의약품 낱알식별 API (HIRA — T1)
    hira_api_base_url: str = "http://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03"
    hira_api_timeout_seconds: float = 8.0

    # 건강기능식품 API (T11, T12)
    health_supplement_api_base_url: str = "http://apis.data.go.kr/1471000/HtfsSttusIdntfcInfoService01"
    health_supplement_api_timeout_seconds: float = 8.0

    # KPIC DUR
    kpic_dur_api_url: Optional[str] = None
    kpic_dur_api_key: Optional[str] = None
    kpic_dur_api_timeout_seconds: float = 8.0

    # Internal LLM (Ollama — OpenAI-compatible /v1/chat/completions)
    internal_llm_provider: str = "ollama"  # ollama | vllm | openai_compatible
    internal_llm_api_url: Optional[str] = "http://127.0.0.1:11434/v1/chat/completions"
    internal_llm_api_key: Optional[str] = None
    internal_llm_model: str = "qwen3:4b"
    llm_prompts_path: str = "./app/prompts/llm_prompts.json"
    llm_tools_path: str = "./app/prompts/llm_tools.json"
    internal_llm_timeout_seconds: float = 60.0
    local_delivery_llm_timeout_seconds: float = 4.0
    internal_llm_temperature: float = 0.0
    internal_llm_route_temperature: float = 0.0
    internal_llm_delivery_temperature: float = 0.0
    internal_llm_reasoning_temperature: float = 0.25
    internal_llm_tool_temperature: float = 0.0
    internal_llm_memory_temperature: float = 0.0
    frontier_llm_judge_temperature: float = 0.0
    conversation_llm_backend: str = "local"  # local | together | auto
    conversation_llm_fallback_enabled: bool = True
    together_conversation_model: Optional[str] = None
    together_conversation_timeout_seconds: float = 10.0
    llm_engine_max_concurrency_internal: int = 1
    llm_engine_max_concurrency_external: int = 1
    llm_engine_max_concurrency_judge: int = 1
    llm_engine_max_concurrency_search: int = 1
    llm_engine_max_concurrency_tool: int = 4
    llm_engine_max_concurrency_dur: int = 4

    # External LLM (OpenAI — LLM as a Judge + LLM Search)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_judge_model: Optional[str] = "gpt-5"
    openai_timeout_seconds: float = 6.0
    openai_search_timeout_seconds: float = 6.0
    anthropic_api_key: Optional[str] = None
    google_ai_api_key: Optional[str] = None

    # Together AI (OpenAI-compatible frontier provider)
    together_api_key: Optional[str] = None
    together_base_url: str = "https://api.together.ai/v1/chat/completions"
    together_model: str = "Qwen/Qwen3.5-9B"
    together_judge_model: Optional[str] = None
    together_search_model: Optional[str] = None
    together_timeout_seconds: float = 6.0

    # Frontier provider routing (OpenAI / Together)
    frontier_llm_enabled_providers: str = "openai,together"
    frontier_llm_primary_provider: str = "openai"
    frontier_llm_fallback_enabled: bool = True

    # OCR
    ocr_api_timeout_seconds: float = 8.0

    # MCP
    mcp_server_url: Optional[str] = None
    mcp_transport: str = "stdio"

    # File storage
    storage_path: str = "./data/storage"
    nfs_mount_path: Optional[str] = None

    # Memory browser (read-only patient memory web UI)
    memory_browser_token: Optional[str] = None
    memory_browser_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # TTS / STT (optional — Clova)
    clova_stt_client_id: Optional[str] = None
    clova_stt_client_secret: Optional[str] = None
    clova_tts_client_id: Optional[str] = None
    clova_tts_client_secret: Optional[str] = None

    # TurboQuant compressed KV cache — enabled by default on all Transformers
    # loads; see turboquant.runtime for details.
    turboquant_auto_wrap: bool = True
    turboquant_key_bits: int = 3
    turboquant_value_bits: int = 3
    turboquant_compress_values: bool = False
    turboquant_require_cuda: bool = False

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
