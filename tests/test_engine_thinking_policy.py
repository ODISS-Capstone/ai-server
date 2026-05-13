"""Thinking-mode policy tests for engine LLM calls."""

from __future__ import annotations

import asyncio

import pytest

from app.core.config import settings
from app.engines.llm_judge import LLMJudgeEngine
from app.services import llm as llm_service


class _FakeResponse:
    status_code = 200

    def __init__(self, content: str = "OK"):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeAsyncClient:
    def __init__(self, *, captured: list[dict], content: str = "OK", **kwargs):
        self.captured = captured
        self.content = content

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, **kwargs):
        self.captured.append({"url": url, **kwargs})
        return _FakeResponse(self.content)


def _user_messages(payload: dict) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in payload["json"]["messages"]
        if message.get("role") == "user"
    ]


@pytest.mark.parametrize("content", ["OK", "<think>internal</think>\nOK"])
def test_delivery_llm_disables_thinking_on_user_facing_pass(monkeypatch, content):
    captured: list[dict] = []

    def fake_client(**kwargs):
        return _FakeAsyncClient(captured=captured, content=content, **kwargs)

    monkeypatch.setattr(settings, "internal_llm_api_url", "http://local/v1/chat/completions")
    monkeypatch.setattr(settings, "internal_llm_api_key", None)
    monkeypatch.setattr(settings, "internal_llm_model", "qwen3-4b")
    monkeypatch.setattr(llm_service.httpx, "AsyncClient", fake_client)

    answer = asyncio.run(
        llm_service.call_local_delivery_llm(
            original_query="이 두 약 같이 먹어도 돼?",
            reviewed_message="출혈 위험이 커질 수 있습니다.",
        )
    )

    assert captured
    users = _user_messages(captured[0])
    assert captured[0]["json"].get("chat_template_kwargs") == {"enable_thinking": False}
    assert not users[-1].startswith("/no_think")
    assert "<think>" not in answer
    assert "</think>" not in answer


def test_internal_reasoning_llm_does_not_force_no_think(monkeypatch):
    captured: list[dict] = []

    def fake_client(**kwargs):
        return _FakeAsyncClient(captured=captured, content="<think>reason</think>\n핵심", **kwargs)

    monkeypatch.setattr(settings, "internal_llm_api_url", "http://local/v1/chat/completions")
    monkeypatch.setattr(settings, "internal_llm_api_key", None)
    monkeypatch.setattr(settings, "internal_llm_model", "qwen3-4b")
    monkeypatch.setattr(llm_service.httpx, "AsyncClient", fake_client)

    answer = asyncio.run(
        llm_service.call_internal_llm(
            query_text="와파린과 아스피린 같이 먹어도 돼?",
            llm_doc="복약 근거",
            use_tools=False,
        )
    )

    assert captured
    assert "chat_template_kwargs" not in captured[0]["json"]
    assert not any(message.startswith("/no_think") for message in _user_messages(captured[0]))
    assert answer == "핵심"


def test_judge_llm_does_not_force_no_think(monkeypatch):
    captured: list[dict] = []

    def fake_client(**kwargs):
        return _FakeAsyncClient(captured=captured, content="VERIFIED", **kwargs)

    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "openai_model", "gpt-5.5")
    monkeypatch.setattr(settings, "openai_judge_model", "gpt-5.5")
    monkeypatch.setattr("app.engines.llm_judge.httpx.AsyncClient", fake_client)

    result = asyncio.run(
        LLMJudgeEngine().verify_fact(
            statement="출혈 위험 확인 필요",
            original_query="같이 먹어도 돼?",
        )
    )

    assert captured
    assert "chat_template_kwargs" not in captured[0]["json"]
    assert not any(message.startswith("/no_think") for message in _user_messages(captured[0]))
    assert result["verified"] is True
