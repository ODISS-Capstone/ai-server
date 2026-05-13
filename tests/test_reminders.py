"""Reminder pending-state regression tests."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from app.database.md_store import MDStore
from app.engines.memory import MemoryEngine
from app.memory import StructuredMemoryService
from app.services.reminders import REMINDER_PENDING_TTL, ReminderService


def run(coro):
    return asyncio.run(coro)


def make_memory(tmp_path) -> MemoryEngine:
    engine = MemoryEngine()
    engine.store = MDStore(base_path=str(tmp_path / "md_database"))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))
    run(engine.initialize())
    return engine


def test_reminder_pending_rejection_clears_setup(tmp_path):
    memory = make_memory(tmp_path)
    service = ReminderService(start_background_tasks=False)
    service.start_setup(
        speaker_id="speaker-reminder",
        user_profile={"name": "홍길동"},
        prescription_log="# 약품 목록\n- DrugA\n",
    )

    cancelled = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="아니 취소해",
            user_profile={"name": "홍길동"},
        )
    )
    later_yes = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="네",
            user_profile={"name": "홍길동"},
        )
    )

    assert "취소" in cancelled
    assert "speaker-reminder" not in service._pending
    assert "speaker-reminder" not in service._active
    assert later_yes is None


def test_reminder_pending_expiry_blocks_stale_confirmation(tmp_path):
    memory = make_memory(tmp_path)
    current = datetime(2026, 5, 13, 12, 0)

    def now_provider():
        return current

    service = ReminderService(now_provider=now_provider, start_background_tasks=False)
    service.start_setup(
        speaker_id="speaker-reminder",
        user_profile={"name": "홍길동"},
        prescription_log="# 약품 목록\n- DrugA\n",
    )
    current = current + REMINDER_PENDING_TTL + timedelta(seconds=1)

    expired = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="네",
            user_profile={"name": "홍길동"},
        )
    )

    assert "시간이 지나" in expired
    assert "speaker-reminder" not in service._pending
    assert "speaker-reminder" not in service._active
