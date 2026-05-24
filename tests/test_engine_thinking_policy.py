"""Thinking-mode policy tests for engine LLM calls."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.core.config import settings
from app.engines.llm_judge import LLMJudgeEngine
from app.services import llm as llm_service
from app.tools import llm_search


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


def test_conversation_llm_together_backend_uses_together_provider(monkeypatch):
    captured: list[dict] = []

    def fake_client(**kwargs):
        return _FakeAsyncClient(captured=captured, content="투게더 응답", **kwargs)

    monkeypatch.setattr(settings, "conversation_llm_backend", "together")
    monkeypatch.setattr(settings, "together_api_key", "together-key")
    monkeypatch.setattr(settings, "together_model", "Qwen/Qwen3.5-9B")
    monkeypatch.setattr(settings, "together_conversation_model", "Qwen/Qwen3.5-9B")
    monkeypatch.setattr("app.services.frontier_llm.httpx.AsyncClient", fake_client)

    answer = asyncio.run(
        llm_service.call_internal_llm(
            query_text="안녕",
            llm_doc="",
            use_tools=False,
        )
    )

    assert answer == "투게더 응답"
    assert captured
    assert captured[0]["url"] == "https://api.together.ai/v1/chat/completions"
    assert captured[0]["json"]["model"] == "Qwen/Qwen3.5-9B"


def test_conversation_llm_auto_falls_back_to_together(monkeypatch):
    captured_local: list[dict] = []
    captured_together: list[dict] = []

    class DispatchingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, **kwargs):
            if "api.together.ai" in url:
                captured_together.append({"url": url, **kwargs})
                return _FakeResponse("fallback 응답")
            captured_local.append({"url": url, **kwargs})
            raise httpx.ConnectError("local down")

    monkeypatch.setattr(settings, "conversation_llm_backend", "auto")
    monkeypatch.setattr(settings, "conversation_llm_fallback_enabled", True)
    monkeypatch.setattr(settings, "internal_llm_api_url", "http://local/v1/chat/completions")
    monkeypatch.setattr(settings, "internal_llm_model", "qwen3-4b")
    monkeypatch.setattr(settings, "together_api_key", "together-key")
    monkeypatch.setattr(settings, "together_conversation_model", "Qwen/Qwen3.5-9B")
    monkeypatch.setattr(llm_service.httpx, "AsyncClient", lambda **kwargs: DispatchingClient())

    answer = asyncio.run(
        llm_service.call_internal_llm(
            query_text="안녕",
            llm_doc="",
            use_tools=False,
        )
    )

    assert answer == "fallback 응답"
    assert captured_local
    assert captured_together
    assert captured_together[0]["json"]["model"] == "Qwen/Qwen3.5-9B"


def test_identity_conflict_judge_uses_local_llm(monkeypatch):
    captured: list[dict] = []

    def fake_client(**kwargs):
        return _FakeAsyncClient(captured=captured, content="TRUE", **kwargs)

    monkeypatch.setattr(settings, "internal_llm_api_url", "http://local/v1/chat/completions")
    monkeypatch.setattr(settings, "internal_llm_api_key", None)
    monkeypatch.setattr(settings, "internal_llm_model", "qwen3-4b")
    monkeypatch.setattr(llm_service.httpx, "AsyncClient", fake_client)

    result = asyncio.run(
        llm_service.judge_identity_conflict(
            current_text="나는 최서연이고 스물네 살 여성이야",
            patient_profile={"name": "박민준", "age": "68", "gender": "남성"},
        )
    )

    assert captured
    assert captured[0]["url"] == "http://local/v1/chat/completions"
    assert captured[0]["json"]["model"] == "qwen3-4b"
    assert captured[0]["json"].get("chat_template_kwargs") == {"enable_thinking": False}
    assert captured[0]["json"]["temperature"] == 0.0
    assert result == {"conflict": True, "source": "local_llm", "raw": "TRUE"}


def test_route_classifier_uses_zero_temperature(monkeypatch):
    captured: list[dict] = []

    def fake_client(**kwargs):
        return _FakeAsyncClient(
            captured=captured,
            content='{"route_label":"noise_fragment","mode":"MEMORY_ONLY","intent":"unknown","task_types":[],"rationale":"test"}',
            **kwargs,
        )

    monkeypatch.setattr(settings, "internal_llm_api_url", "http://local/v1/chat/completions")
    monkeypatch.setattr(settings, "internal_llm_api_key", None)
    monkeypatch.setattr(settings, "internal_llm_model", "qwen3-4b")
    monkeypatch.setattr(llm_service.httpx, "AsyncClient", fake_client)

    result = asyncio.run(
        llm_service.classify_reasoning_route_with_llm(
            current_text="스읍",
            conversation_context="",
        )
    )

    assert result["route_label"] == "noise_fragment"
    assert captured[0]["json"]["temperature"] == 0.0
    assert captured[0]["json"].get("chat_template_kwargs") == {"enable_thinking": False}


def test_identity_conflict_judge_does_not_fallback_to_keyword_heuristic(monkeypatch):
    monkeypatch.setattr(settings, "internal_llm_api_url", None)

    result = asyncio.run(
        llm_service.judge_identity_conflict(
            current_text="나는 최서연이고 스물네 살 여성이야",
            patient_profile={"name": "박민준", "age": "68", "gender": "남성"},
        )
    )

    assert result["conflict"] is False
    assert result["source"] == "local_llm_not_configured"


def test_ocr_medication_extraction_uses_frontier_openai(monkeypatch):
    captured: list[dict] = []

    def fake_client(**kwargs):
        return _FakeAsyncClient(
            captured=captured,
            content='{"medications":[{"name":"무브록정40mg"}],"clarification_question":"증상을 알려주세요."}',
            **kwargs,
        )

    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "openai_model", "gpt-test")
    monkeypatch.setattr(settings, "google_ai_api_key", None)
    monkeypatch.setattr(llm_service.httpx, "AsyncClient", fake_client)

    result = asyncio.run(
        llm_service.extract_ocr_medication_candidates_with_llm(
            "처방 의약품 목록 | 무브록정40mg | 용법 [불명확]"
        )
    )

    assert captured
    assert captured[0]["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured[0]["json"]["model"] == "gpt-test"
    assert result["source"] == "frontier_openai"
    assert result["medications"][0]["name"] == "무브록정40mg"


def test_ocr_medication_refinement_uses_frontier_openai(monkeypatch):
    captured: list[dict] = []

    def fake_client(**kwargs):
        return _FakeAsyncClient(
            captured=captured,
            content='{"medications":[{"name":"페브릭정","purpose_or_symptom":"통풍"}],"clarification_question":""}',
            **kwargs,
        )

    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "openai_model", "gpt-test")
    monkeypatch.setattr(settings, "google_ai_api_key", None)
    monkeypatch.setattr(llm_service.httpx, "AsyncClient", fake_client)

    result = asyncio.run(
        llm_service.refine_ocr_medication_candidates_with_context(
            raw_text="처방 의약품 | 페니라민정 | 록소나정 60mg",
            current_medications=[{"name": "페니라민정"}, {"name": "록소나정 60mg"}],
            user_text="통풍 때문에 처방받은 약이야",
        )
    )

    assert captured
    assert captured[0]["url"] == "https://api.openai.com/v1/chat/completions"
    assert "통풍 때문에 처방받은 약이야" in captured[0]["json"]["messages"][1]["content"]
    assert result["source"] == "frontier_openai_context_refine"
    assert result["medications"][0]["name"] == "페브릭정"


def test_reasoning_tag_stripper_removes_embedded_and_trailing_think_blocks():
    content = 'Visible answer. <think data-source="qwen">late reasoning</think>\nNext sentence.<THINK>unfinished'

    stripped = llm_service._strip_reasoning_tags(content)

    assert stripped == "Visible answer. Next sentence."
    assert "<think" not in stripped.lower()
    assert "reasoning" not in stripped


def test_judge_llm_does_not_force_no_think(monkeypatch):
    captured: list[dict] = []

    def fake_client(**kwargs):
        return _FakeAsyncClient(captured=captured, content="VERIFIED", **kwargs)

    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "openai_model", "gpt-5.5")
    monkeypatch.setattr(settings, "openai_judge_model", "gpt-5.5")
    monkeypatch.setattr(settings, "frontier_llm_enabled_providers", "openai")
    monkeypatch.setattr(settings, "frontier_llm_primary_provider", "openai")
    monkeypatch.setattr("app.services.frontier_llm.httpx.AsyncClient", fake_client)

    result = asyncio.run(
        LLMJudgeEngine().verify_fact(
            statement="출혈 위험 확인 필요",
            original_query="같이 먹어도 돼?",
        )
    )

    assert captured
    assert "chat_template_kwargs" not in captured[0]["json"]
    assert "temperature" not in captured[0]["json"]
    assert not any(message.startswith("/no_think") for message in _user_messages(captured[0]))
    assert result["verified"] is True


def test_external_llm_fallback_does_not_echo_payload(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None)

    answer = asyncio.run(
        llm_service.call_external_llm(
            "SECRET_PATIENT_CONTEXT: user takes aspirin and warfarin",
        )
    )

    assert "SECRET_PATIENT_CONTEXT" not in answer
    assert "aspirin" not in answer
    assert "warfarin" not in answer


def test_llm_search_strips_reasoning_tags_from_answer(monkeypatch):
    captured: list[dict] = []

    def fake_client(**kwargs):
        return _FakeAsyncClient(
            captured=captured,
            content='<think data-source="search">internal search reasoning</think>\nSearch answer.',
            **kwargs,
        )

    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "frontier_llm_enabled_providers", "openai")
    monkeypatch.setattr(settings, "frontier_llm_primary_provider", "openai")
    monkeypatch.setattr("app.services.frontier_llm.httpx.AsyncClient", fake_client)

    result = asyncio.run(llm_search.llm_search("search query"))

    assert result["success"] is True
    assert result["answer"] == "Search answer."
    assert "think" not in result["answer"].lower()
