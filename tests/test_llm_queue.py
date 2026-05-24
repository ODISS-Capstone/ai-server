"""LLM queueing tests."""
import asyncio

from app.core.config import settings
from app.services.llm_queue import engine_slot, run_with_engine_queue


def test_engine_queue_serializes_when_limit_is_one(monkeypatch):
    monkeypatch.setattr(settings, "llm_engine_max_concurrency_internal", 1)
    active = 0
    max_active = 0

    async def worker():
        nonlocal active, max_active
        async with engine_slot("internal"):
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1

    async def run_workers():
        await asyncio.gather(*(worker() for _ in range(3)))

    asyncio.run(run_workers())

    assert max_active == 1


def test_run_with_engine_queue_returns_operation_result(monkeypatch):
    monkeypatch.setattr(settings, "llm_engine_max_concurrency_search", 1)

    async def operation():
        return "queued-result"

    result = asyncio.run(run_with_engine_queue("search", operation))

    assert result == "queued-result"


def test_tool_engine_queue_serializes_when_limit_is_one(monkeypatch):
    monkeypatch.setattr(settings, "llm_engine_max_concurrency_tool", 1)
    active = 0
    max_active = 0

    async def worker():
        nonlocal active, max_active
        async with engine_slot("tool"):
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1

    async def run_workers():
        await asyncio.gather(*(worker() for _ in range(3)))

    asyncio.run(run_workers())

    assert max_active == 1


def test_dur_engine_queue_serializes_when_limit_is_one(monkeypatch):
    monkeypatch.setattr(settings, "llm_engine_max_concurrency_dur", 1)
    active = 0
    max_active = 0

    async def worker():
        nonlocal active, max_active
        async with engine_slot("dur"):
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1

    async def run_workers():
        await asyncio.gather(*(worker() for _ in range(3)))

    asyncio.run(run_workers())

    assert max_active == 1
