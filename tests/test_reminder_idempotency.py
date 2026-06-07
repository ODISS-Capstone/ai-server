from __future__ import annotations

import asyncio
from datetime import datetime

from app.database.md_store import MDStore
from app.engines.memory import MemoryEngine
from app.memory import StructuredMemoryService
from app.services.reminders import ReminderService


def run(coro):
    return asyncio.run(coro)


def make_memory(tmp_path) -> MemoryEngine:
    engine = MemoryEngine()
    engine.store = MDStore(base_path=str(tmp_path / "md_database"))
    engine.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))
    run(engine.initialize())
    return engine


def test_one_shot_relative_alarm_is_idempotent_for_same_run_at() -> None:
    current = datetime(2026, 5, 27, 8, 0, 0)

    def now_provider() -> datetime:
        return current

    service = ReminderService(now_provider=now_provider, start_background_tasks=False)

    first = run(
        service.schedule_one_shot(
            speaker_id="speaker-kim",
            text="3초 뒤에 타이레놀 먹으라고 알려 줘",
            user_profile={"name": "김영수"},
            prescription_log="",
            start_tasks=False,
        )
    )
    second = run(
        service.schedule_one_shot(
            speaker_id="speaker-kim",
            text="3초 뒤에 타이레놀 먹으라고 알려 줘",
            user_profile={"name": "김영수"},
            prescription_log="",
            start_tasks=False,
        )
    )

    assert first == second
    assert len(service._one_shots) == 1


def test_medication_taken_record_is_idempotent_for_duplicate_turn(tmp_path) -> None:
    memory = make_memory(tmp_path)
    current = datetime(2026, 5, 27, 8, 10, 0)

    def now_provider() -> datetime:
        return current

    service = ReminderService(now_provider=now_provider, start_background_tasks=False)
    kwargs = {
        "memory_engine": memory,
        "speaker_id": "speaker-kim",
        "text": "약 먹었어",
        "user_profile": {"name": "김영수"},
        "prescription_log": "# 현재 복용 약 요약\n\n## 약품 목록\n- 타이레놀\n",
    }

    first = run(service.handle_user_text(**kwargs))
    second = run(service.handle_user_text(**kwargs))
    saved = run(memory.store.read_user_file("speaker-kim", "medication_taken.md"))

    assert first == second
    assert saved.count('"medication_label": "타이레놀"') == 1
    assert saved.count('"idempotency_key"') == 1
