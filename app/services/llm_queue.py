"""Per-engine async queueing for LLM backends."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter
from typing import TypeVar
from weakref import WeakKeyDictionary

from app.core.config import settings

T = TypeVar("T")
logger = logging.getLogger(__name__)

_ENGINE_LIMIT_FIELDS = {
    "internal": "llm_engine_max_concurrency_internal",
    "external": "llm_engine_max_concurrency_external",
    "judge": "llm_engine_max_concurrency_judge",
    "search": "llm_engine_max_concurrency_search",
    "tool": "llm_engine_max_concurrency_tool",
    "dur": "llm_engine_max_concurrency_dur",
}
_SEMAPHORES_BY_LOOP: WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Semaphore]
] = WeakKeyDictionary()


def get_engine_limit(engine: str) -> int:
    """Return configured max concurrency for an LLM logical engine."""
    field_name = _ENGINE_LIMIT_FIELDS.get(engine)
    if not field_name:
        return 1
    configured = getattr(settings, field_name, 1)
    return max(1, int(configured))


def get_engine_semaphore(engine: str) -> asyncio.Semaphore:
    """Return an event-loop-local semaphore for the given engine."""
    loop = asyncio.get_running_loop()
    semaphores = _SEMAPHORES_BY_LOOP.setdefault(loop, {})
    if engine not in semaphores:
        semaphores[engine] = asyncio.Semaphore(get_engine_limit(engine))
    return semaphores[engine]


@asynccontextmanager
async def engine_slot(engine: str):
    """Wait for capacity on an engine queue and release it after use."""
    semaphore = get_engine_semaphore(engine)
    async with semaphore:
        yield


async def run_with_engine_queue(
    engine: str,
    operation: Callable[[], Awaitable[T]],
) -> T:
    """Run an awaitable factory while holding the engine queue slot."""
    queued_at = perf_counter()
    async with engine_slot(engine):
        wait_ms = (perf_counter() - queued_at) * 1000
        logger.info(
            "[LLMQueue] slot_acquired engine=%s limit=%d wait_ms=%.1f",
            engine,
            get_engine_limit(engine),
            wait_ms,
        )
        started = perf_counter()
        try:
            return await operation()
        finally:
            logger.info(
                "[LLMQueue] operation_done engine=%s elapsed_ms=%.1f",
                engine,
                (perf_counter() - started) * 1000,
            )
