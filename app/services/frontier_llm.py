"""OpenAI-compatible frontier LLM provider router (OpenAI, Together)."""
from __future__ import annotations

import logging
import re
from typing import Any, Literal, Optional

import httpx

from app.core.config import settings
from app.services.llm_queue import run_with_engine_queue

logger = logging.getLogger(__name__)

FrontierProvider = Literal["openai", "together"]
FrontierTask = Literal["judge", "search", "external", "conversation"]

_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_ALL_PROVIDERS: tuple[FrontierProvider, ...] = ("openai", "together")


def _parse_enabled_providers() -> list[FrontierProvider]:
    raw = (settings.frontier_llm_enabled_providers or "").strip()
    if not raw:
        return []
    parsed: list[FrontierProvider] = []
    for item in raw.split(","):
        normalized = item.strip().lower()
        if normalized in _ALL_PROVIDERS and normalized not in parsed:
            parsed.append(normalized)  # type: ignore[arg-type]
    return parsed


def _provider_order() -> list[FrontierProvider]:
    enabled = _parse_enabled_providers()
    if not enabled:
        return []

    primary = (settings.frontier_llm_primary_provider or "openai").strip().lower()
    order: list[FrontierProvider] = []
    if primary in enabled:
        order.append(primary)  # type: ignore[arg-type]
    for provider in enabled:
        if provider not in order:
            order.append(provider)
    return order


def is_provider_configured(provider: FrontierProvider) -> bool:
    if provider == "openai":
        return bool(settings.openai_api_key)
    if provider == "together":
        return bool(settings.together_api_key)
    return False


def has_configured_frontier_provider() -> bool:
    return any(
        is_provider_configured(provider)
        for provider in _provider_order()
    )


def _model_for_provider_task(provider: FrontierProvider, task: FrontierTask) -> str:
    if provider == "openai":
        if task == "judge":
            return settings.openai_judge_model or settings.openai_model
        return settings.openai_model

    if task == "conversation":
        return settings.together_conversation_model or settings.together_model
    if task == "judge":
        return settings.together_judge_model or settings.together_model
    if task == "search":
        return settings.together_search_model or settings.together_model
    return settings.together_model


def _parse_csv_models(raw: str | None) -> list[str]:
    if not raw:
        return []
    models: list[str] = []
    for item in raw.split(","):
        model = item.strip()
        if model and model not in models:
            models.append(model)
    return models


def _models_for_provider_task(provider: FrontierProvider, task: FrontierTask) -> list[str]:
    primary = _model_for_provider_task(provider, task)
    models = [primary] if primary else []
    if provider == "together":
        if task == "conversation":
            models.extend(_parse_csv_models(settings.together_conversation_fallback_models))
        elif task == "judge":
            models.extend(_parse_csv_models(settings.together_judge_fallback_models))
        elif task == "search":
            models.extend(_parse_csv_models(settings.together_search_fallback_models))
    deduped: list[str] = []
    for model in models:
        if model and model not in deduped:
            deduped.append(model)
    return deduped


def _timeout_for_provider_task(provider: FrontierProvider, task: FrontierTask) -> float:
    if task == "conversation":
        if provider == "openai":
            return settings.openai_timeout_seconds
        return settings.together_conversation_timeout_seconds
    if provider == "openai":
        if task == "search":
            return settings.openai_search_timeout_seconds
        return settings.openai_timeout_seconds
    return settings.together_timeout_seconds


def _reasoning_for_provider_task(provider: FrontierProvider, task: FrontierTask) -> Optional[bool]:
    if provider != "together":
        return None
    if task == "conversation" and settings.together_conversation_reasoning_enabled is not None:
        return settings.together_conversation_reasoning_enabled
    if task == "judge" and settings.together_judge_reasoning_enabled is not None:
        return settings.together_judge_reasoning_enabled
    if task == "search" and settings.together_search_reasoning_enabled is not None:
        return settings.together_search_reasoning_enabled
    return settings.together_reasoning_enabled


def _chat_url_for_provider(provider: FrontierProvider) -> str:
    if provider == "openai":
        return _OPENAI_CHAT_URL
    return settings.together_base_url or "https://api.together.ai/v1/chat/completions"


def _api_key_for_provider(provider: FrontierProvider) -> Optional[str]:
    if provider == "openai":
        return settings.openai_api_key
    if provider == "together":
        return settings.together_api_key
    return None


def _supports_temperature(model: str) -> bool:
    normalized = (model or "").lower().strip()
    return not normalized.startswith("gpt-5")


def _default_temperature_for_task(task: FrontierTask) -> float:
    if task == "judge":
        return settings.frontier_llm_judge_temperature
    if task == "search":
        return settings.internal_llm_memory_temperature
    if task == "conversation":
        return settings.internal_llm_delivery_temperature
    return settings.internal_llm_temperature


def _build_chat_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    reasoning_enabled: Optional[bool] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if (model or "").lower().strip().startswith("gpt-5"):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    if _supports_temperature(model):
        payload["temperature"] = temperature
    if reasoning_enabled is not None:
        payload["reasoning"] = {"enabled": reasoning_enabled}
    return payload


async def _post_chat_completion(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def _queue_engine_for_task(task: FrontierTask) -> str:
    if task == "conversation":
        return "internal"
    if task == "judge":
        return "judge"
    if task == "search":
        return "search"
    return "external"


def _extract_content(data: dict[str, Any]) -> str:
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        or ""
    )
    return _strip_reasoning_tags(content)


def _strip_reasoning_tags(content: str) -> str:
    if "<think" not in (content or "").lower():
        return content
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>\s*", "", content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think\b[^>]*>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


async def chat_completion(
    *,
    task: FrontierTask,
    messages: list[dict[str, Any]],
    max_tokens: int = 256,
    temperature: Optional[float] = None,
) -> dict[str, Any]:
    """Call the first available frontier provider with optional fallback."""
    providers = _provider_order()
    if not providers:
        return {
            "success": False,
            "content": "",
            "provider": None,
            "model": None,
            "message": "frontier provider 미설정",
        }

    errors: list[str] = []
    for index, provider in enumerate(providers):
        if not is_provider_configured(provider):
            continue

        api_key = _api_key_for_provider(provider)
        url = _chat_url_for_provider(provider)
        timeout = _timeout_for_provider_task(provider, task)
        resolved_temperature = (
            _default_temperature_for_task(task)
            if temperature is None
            else temperature
        )
        queue_engine = _queue_engine_for_task(task)

        for model in _models_for_provider_task(provider, task):
            payload = _build_chat_payload(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=resolved_temperature,
                reasoning_enabled=_reasoning_for_provider_task(provider, task),
            )

            async def _post(
                provider_name: FrontierProvider = provider,
                selected_model: str = model,
            ) -> dict[str, Any]:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    data = await _post_chat_completion(
                        client,
                        url=url,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        payload=payload,
                    )
                    data["_frontier_provider"] = provider_name
                    data["_frontier_model"] = selected_model
                    return data

            try:
                data = await run_with_engine_queue(queue_engine, _post)
                return {
                    "success": True,
                    "content": _extract_content(data),
                    "provider": data.get("_frontier_provider", provider),
                    "model": data.get("_frontier_model", model),
                    "message": "ok",
                }
            except httpx.HTTPStatusError as exc:
                message = f"{provider} {model} HTTP {exc.response.status_code}"
                logger.error("[FrontierLLM] %s task=%s", message, task)
                errors.append(message)
            except Exception as exc:  # noqa: BLE001 - provider fallback must continue
                message = f"{provider} {model} error: {exc}"
                logger.error("[FrontierLLM] %s task=%s", message, task)
                errors.append(message)

        if not settings.frontier_llm_fallback_enabled:
            break
        if index == len(providers) - 1:
            break

    return {
        "success": False,
        "content": "",
        "provider": None,
        "model": None,
        "message": "; ".join(errors) if errors else "사용 가능한 frontier provider 없음",
    }


async def together_conversation_completion(
    *,
    messages: list[dict[str, Any]],
    max_tokens: int = 256,
    temperature: Optional[float] = None,
) -> dict[str, Any]:
    """Call Together directly for conversation LLM work."""
    if not settings.together_api_key:
        return {
            "success": False,
            "content": "",
            "provider": "together",
            "model": None,
            "message": "TOGETHER_API_KEY is not set",
        }

    url = _chat_url_for_provider("together")
    resolved_temperature = (
        settings.internal_llm_delivery_temperature
        if temperature is None
        else temperature
    )

    errors: list[str] = []
    for model in _models_for_provider_task("together", "conversation"):
        payload = _build_chat_payload(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=resolved_temperature,
            reasoning_enabled=_reasoning_for_provider_task("together", "conversation"),
        )

        async def _post(selected_model: str = model) -> dict[str, Any]:
            async with httpx.AsyncClient(
                timeout=settings.together_conversation_timeout_seconds
            ) as client:
                data = await _post_chat_completion(
                    client,
                    url=url,
                    headers={
                        "Authorization": f"Bearer {settings.together_api_key}",
                        "Content-Type": "application/json",
                    },
                    payload=payload,
                )
                data["_frontier_provider"] = "together"
                data["_frontier_model"] = selected_model
                return data

        try:
            data = await run_with_engine_queue("internal", _post)
            return {
                "success": True,
                "content": _extract_content(data),
                "provider": "together",
                "model": model,
                "message": "ok",
            }
        except httpx.HTTPStatusError as exc:
            message = f"together {model} HTTP {exc.response.status_code}"
            logger.error("[ConversationLLM] %s", message)
            errors.append(message)
        except Exception as exc:  # noqa: BLE001 - caller decides fallback behavior
            message = f"together {model} error: {exc}"
            logger.error("[ConversationLLM] %s", message)
            errors.append(message)

    return {
        "success": False,
        "content": "",
        "provider": "together",
        "model": None,
        "message": "; ".join(errors) if errors else "Together conversation call failed",
    }


async def check_frontier_llm_health() -> dict[str, Any]:
    """Return configured/enabled status for frontier providers."""
    enabled = _parse_enabled_providers()
    providers: dict[str, dict[str, Any]] = {}
    for provider in _ALL_PROVIDERS:
        providers[provider] = {
            "enabled": provider in enabled,
            "configured": is_provider_configured(provider),
            "primary": provider == (settings.frontier_llm_primary_provider or "").strip().lower(),
            "model_judge": _model_for_provider_task(provider, "judge"),
            "model_search": _model_for_provider_task(provider, "search"),
            "model_conversation": _model_for_provider_task(provider, "conversation"),
            "models_judge": _models_for_provider_task(provider, "judge"),
            "models_search": _models_for_provider_task(provider, "search"),
            "models_conversation": _models_for_provider_task(provider, "conversation"),
            "reasoning_judge": _reasoning_for_provider_task(provider, "judge"),
            "reasoning_search": _reasoning_for_provider_task(provider, "search"),
            "reasoning_conversation": _reasoning_for_provider_task(provider, "conversation"),
        }

    return {
        "enabled_providers": list(enabled),
        "primary_provider": settings.frontier_llm_primary_provider,
        "fallback_enabled": settings.frontier_llm_fallback_enabled,
        "any_available": has_configured_frontier_provider(),
        "providers": providers,
    }
