"""HTTP query API session-context regression tests."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import query
from app.database.md_store import MDStore
from app.engines.memory import MemoryEngine
from app.memory import StructuredMemoryService
from app.schemas.answer import AskRequest


def run(coro):
    return asyncio.run(coro)


def make_memory(tmp_path) -> MemoryEngine:
    engine = MemoryEngine()
    engine.store = MDStore(base_path=str(tmp_path / "memory_database"))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))
    return engine


class FakeOrchestrator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run_turn(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            core_message="세션 문서를 기준으로 답했습니다.",
            evidence=SimpleNamespace(frontier_answer_preview=""),
            reviewed_message="세션 문서를 기준으로 답했습니다.",
            conversation=SimpleNamespace(response_text="DrugA 기준 답변입니다."),
            decision=SimpleNamespace(intent="medication_query"),
            execution_results={"task_results": {}},
            judge_review={},
        )


def test_query_ask_passes_exact_session_llm_doc_to_orchestrator(tmp_path, monkeypatch):
    session_id = "session-ctx-123"
    llm_doc = "[현재 복용 중인 약]\n\n - DrugA, 1정\n\n[DUR 검증 및 주의사항]\n - DrugA"
    store = MDStore(base_path=str(tmp_path / "md_database"))
    run(store.initialize())
    run(
        store.save(
            "medication_log",
            (
                "# 파이프라인 세션\n"
                f"> 세션 ID: {session_id}\n\n"
                "## 사용자 질문\n(없음)\n\n"
                f"## LLM 문서\n{llm_doc}\n"
            ),
        )
    )
    memory = make_memory(tmp_path)
    fake_orchestrator = FakeOrchestrator()

    async def fake_send_verified_to_mcp(*args, **kwargs):
        return False

    monkeypatch.setattr(query, "md_store", store)
    monkeypatch.setattr(query, "memory_engine", memory)
    monkeypatch.setattr(query, "engine_orchestrator", fake_orchestrator)
    monkeypatch.setattr(query, "send_verified_to_mcp", fake_send_verified_to_mcp)

    response = run(query.ask(AskRequest(session_id=session_id, query_text="이 약 설명해줘")))

    assert response.answer_final == "DrugA 기준 답변입니다."
    call = fake_orchestrator.calls[0]
    assert call["preloaded_context"]["prescription_log"] == llm_doc
    assert session_id in call["preloaded_context"]["context_memory"]
    assert call["preloaded_context"]["memory_prompt"] == llm_doc


def test_query_ask_rejects_unknown_session_even_with_query_text(tmp_path, monkeypatch):
    store = MDStore(base_path=str(tmp_path / "md_database"))
    run(store.initialize())
    monkeypatch.setattr(query, "md_store", store)
    monkeypatch.setattr(query, "memory_engine", make_memory(tmp_path))

    with pytest.raises(HTTPException) as exc_info:
        run(query.ask(AskRequest(session_id="missing-session", query_text="아무 질문")))

    assert exc_info.value.status_code == 404
