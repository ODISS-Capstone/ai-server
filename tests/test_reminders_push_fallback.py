"""Reminder push fallback when websocket is offline."""
from __future__ import annotations

import asyncio
from datetime import datetime

from app.database.md_store import MDStore
from app.engines.memory import MemoryEngine
from app.memory import StructuredMemoryService
from app.services import reminders as reminders_module
from app.services.reminders import ReminderService


def run(coro):
    return asyncio.run(coro)


def make_memory(tmp_path) -> MemoryEngine:
    memory = MemoryEngine()
    memory.store = MDStore(base_path=str(tmp_path / "md_database"))
    memory.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))
    run(memory.initialize())
    return memory


def test_dispatch_due_reminder_uses_device_push_without_ws(tmp_path, monkeypatch):
    memory = make_memory(tmp_path)
    clock = {"now": datetime(2026, 5, 27, 8, 0)}

    def now_provider():
        return clock["now"]

    sent = []

    async def fake_send(speaker_id, text, tts_requested=True, meta=None):
        sent.append(
            {
                "speaker_id": speaker_id,
                "text": text,
                "tts_requested": tts_requested,
                "meta": meta or {},
            }
        )
        return 1

    monkeypatch.setattr(reminders_module, "send_to_speaker_devices", fake_send)

    service = ReminderService(now_provider=now_provider, start_background_tasks=False)
    run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-push",
            text="30초 뒤에 혈압 약 먹으라고 알람 설정해줘",
            user_profile={"name": "김영수"},
            prescription_log="# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )
    clock["now"] = datetime(2026, 5, 27, 8, 1)
    dispatched = run(service.dispatch_due_reminders())
    assert dispatched
    assert sent
    assert sent[-1]["speaker_id"] == "speaker-push"
    assert sent[-1]["meta"]["type"] == "reminder"

