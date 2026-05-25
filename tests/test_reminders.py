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


def test_reminder_understands_elderly_wake_me_language_and_restores(tmp_path):
    memory = make_memory(tmp_path)
    current = datetime(2026, 5, 18, 11, 55)

    def now_provider():
        return current

    service = ReminderService(now_provider=now_provider, start_background_tasks=False)

    setup = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="약 먹을 때 깨워줘",
            user_profile={"name": "김영수"},
            prescription_log="# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )
    assert "오전 8시" in setup
    assert "오후 1시" in setup

    confirm = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="점심은 12시로 해줘",
            user_profile={"name": "김영수"},
            prescription_log="# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )
    assert "점심 약 알림은 오후 12시" in confirm

    restored_sent: list[dict] = []
    restored = ReminderService(now_provider=now_provider, start_background_tasks=False)
    restored.register_connection("speaker-reminder", lambda payload: restored_sent.append(payload))
    run(restored.restore_for_speaker(memory, "speaker-reminder"))

    current = datetime(2026, 5, 18, 12, 0)
    dispatched = run(restored.dispatch_due_reminders())

    assert dispatched
    assert restored_sent[-1]["type"] == "reminder"
    assert "김영수님" in restored_sent[-1]["text"]
    assert "점심 혈압약" in restored_sent[-1]["text"]

    saved = run(memory.store.read_user_file("speaker-reminder", "reminders.md"))
    assert "2026-05-19T12:00:00" in saved


def test_reminder_setup_uses_existing_prescription_context_for_vague_forgetful_request(tmp_path):
    memory = make_memory(tmp_path)
    service = ReminderService(start_background_tasks=False)

    setup = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="까먹으니까 좀 알려줘",
            user_profile={"name": "김영수"},
            prescription_log="# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )

    assert "식후 복용 알림" in setup
    assert "오전 8시" in setup


def test_reminder_dispatch_uses_current_prescription_names(tmp_path):
    memory = make_memory(tmp_path)
    current = datetime(2026, 5, 26, 11, 59)

    def now_provider():
        return current

    sent: list[dict] = []
    service = ReminderService(now_provider=now_provider, start_background_tasks=False)
    service.register_connection("speaker-reminder", lambda payload: sent.append(payload))
    prescription_log = (
        "# 현재 복용 약 요약\n\n"
        "## 약품 목록\n"
        "- 타이레놀정500밀리그람\n"
        "- 아스피린프로텍트정100밀리그람\n"
    )

    setup = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="밥 먹고 나서 약 먹어야 한다는 알림 추가해줘",
            user_profile={"name": "김영수"},
            prescription_log=prescription_log,
        )
    )
    confirm = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="점심은 12시로 해줘",
            user_profile={"name": "김영수"},
            prescription_log=prescription_log,
        )
    )

    current = datetime(2026, 5, 26, 12, 0)
    dispatched = run(service.dispatch_due_reminders())

    assert "식후 복용 알림" in setup
    assert "점심 약 알림은 오후 12시" in confirm
    assert dispatched
    assert "점심 타이레놀정500밀리그람, 아스피린프로텍트정100밀리그람" in sent[-1]["text"]


def test_reminder_default_timezone_is_kst_and_legacy_naive_times_are_localized():
    service = ReminderService(start_background_tasks=False)
    now = datetime(2026, 5, 26, 11, 59, tzinfo=service._timezone)

    parsed = service._parse_datetime("2026-05-26T12:00:00", now=now)

    assert service._now().utcoffset() == timedelta(hours=9)
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(hours=9)
