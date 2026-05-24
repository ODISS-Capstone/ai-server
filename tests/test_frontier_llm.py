"""Frontier provider router tests."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from app.core.config import settings
from app.services import frontier_llm


class _FakeResponse:
    status_code = 200

    def __init__(self, content: str = "Frontier answer"):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeAsyncClient:
    def __init__(self, *, captured: list[dict], **kwargs):
        self.captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self.captured.append({"url": url, **kwargs})
        return _FakeResponse("Search answer.")


def test_provider_order_respects_primary_provider(monkeypatch):
    monkeypatch.setattr(settings, "frontier_llm_enabled_providers", "openai,together")
    monkeypatch.setattr(settings, "frontier_llm_primary_provider", "together")

    order = frontier_llm._provider_order()

    assert order == ["together", "openai"]


def test_chat_completion_uses_openai_when_configured(monkeypatch):
    captured: list[dict] = []

    def fake_client(**kwargs):
        return _FakeAsyncClient(captured=captured, **kwargs)

    monkeypatch.setattr(settings, "openai_api_key", "openai-key")
    monkeypatch.setattr(settings, "together_api_key", None)
    monkeypatch.setattr(settings, "frontier_llm_enabled_providers", "openai,together")
    monkeypatch.setattr(settings, "frontier_llm_primary_provider", "openai")
    monkeypatch.setattr(frontier_llm.httpx, "AsyncClient", fake_client)

    result = asyncio.run(
        frontier_llm.chat_completion(
            task="search",
            messages=[{"role": "user", "content": "query"}],
            max_tokens=128,
        )
    )

    assert result["success"] is True
    assert result["provider"] == "openai"
    assert captured[0]["url"] == "https://api.openai.com/v1/chat/completions"


def test_chat_completion_falls_back_to_together(monkeypatch):
    captured: list[dict] = []

    class _FailOpenAIClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, **kwargs):
            captured.append({"url": url, **kwargs})
            if "api.openai.com" in url:
                request = httpx.Request("POST", url)
                response = httpx.Response(503, request=request)
                raise httpx.HTTPStatusError("fail", request=request, response=response)
            return _FakeResponse("Together answer.")

    monkeypatch.setattr(settings, "openai_api_key", "openai-key")
    monkeypatch.setattr(settings, "together_api_key", "together-key")
    monkeypatch.setattr(settings, "frontier_llm_enabled_providers", "openai,together")
    monkeypatch.setattr(settings, "frontier_llm_primary_provider", "openai")
    monkeypatch.setattr(settings, "frontier_llm_fallback_enabled", True)
    monkeypatch.setattr(frontier_llm.httpx, "AsyncClient", lambda **kwargs: _FailOpenAIClient())

    result = asyncio.run(
        frontier_llm.chat_completion(
            task="search",
            messages=[{"role": "user", "content": "query"}],
            max_tokens=128,
        )
    )

    assert result["success"] is True
    assert result["provider"] == "together"
    assert any("together.ai" in item["url"] for item in captured)


def test_chat_completion_returns_error_when_no_provider_configured(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(settings, "together_api_key", None)
    monkeypatch.setattr(settings, "frontier_llm_enabled_providers", "openai,together")

    result = asyncio.run(
        frontier_llm.chat_completion(
            task="judge",
            messages=[{"role": "user", "content": "query"}],
        )
    )

    assert result["success"] is False
    assert "frontier provider" in result["message"].lower() or "사용 가능" in result["message"]


@pytest.mark.parametrize("provider", ["openai", "together"])
def test_check_frontier_llm_health_reports_provider_status(monkeypatch, provider):
    monkeypatch.setattr(settings, "openai_api_key", "openai-key" if provider == "openai" else None)
    monkeypatch.setattr(settings, "together_api_key", "together-key" if provider == "together" else None)
    monkeypatch.setattr(settings, "frontier_llm_enabled_providers", provider)
    monkeypatch.setattr(settings, "frontier_llm_primary_provider", provider)

    result = asyncio.run(frontier_llm.check_frontier_llm_health())

    assert result["any_available"] is True
    assert result["providers"][provider]["configured"] is True
    assert result["providers"][provider]["enabled"] is True
