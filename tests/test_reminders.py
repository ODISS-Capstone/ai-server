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


def test_relative_one_shot_medication_reminder_dispatches_after_delay(tmp_path):
    memory = make_memory(tmp_path)
    current = datetime(2026, 5, 26, 19, 47)

    def now_provider():
        return current

    sent: list[dict] = []
    service = ReminderService(now_provider=now_provider, start_background_tasks=False)
    service.register_connection("speaker-reminder", lambda payload: sent.append(payload))

    setup = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="30초 뒤에 혈압 약 먹으라고 알람 설정해 줄 수 있어",
            user_profile={"name": "정현기"},
            prescription_log="# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )

    assert "30초 뒤" in setup
    assert "혈압약" in setup
    assert service._one_shots
    assert not ReminderService.is_setup_request(
        "30초 뒤에 혈압 약 먹으라고 알람 설정해 줄 수 있어",
        prescription_log="# 현재 복용 약 요약\n- 혈압약\n",
    )

    current = datetime(2026, 5, 26, 19, 47, 29)
    assert run(service.dispatch_due_reminders()) == []
    assert sent == []

    current = datetime(2026, 5, 26, 19, 47, 30)
    dispatched = run(service.dispatch_due_reminders())

    assert dispatched
    assert sent[-1]["type"] == "reminder"
    assert sent[-1]["reminder_kind"] == "one_shot"
    assert "정현기님" in sent[-1]["text"]
    assert "혈압약" in sent[-1]["text"]
    assert not service._one_shots


def test_relative_one_shot_tylenol_reminder_uses_explicit_medication_name(tmp_path):
    memory = make_memory(tmp_path)
    current = datetime(2026, 5, 26, 20, 36, 19)

    def now_provider():
        return current

    sent: list[dict] = []
    service = ReminderService(now_provider=now_provider, start_background_tasks=False)
    service.register_connection("speaker-reminder", lambda payload: sent.append(payload))

    setup = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="10초 뒤에 타이레놀 먹으라고 알려 줘",
            user_profile={"name": "정현기"},
            prescription_log="",
        )
    )

    assert "10초 뒤" in setup
    assert "타이레놀" in setup
    assert "의사·약사" not in setup
    assert "확인된 정보가 제한적" not in setup

    current = datetime(2026, 5, 26, 20, 36, 29)
    dispatched = run(service.dispatch_due_reminders())

    assert dispatched
    assert sent[-1]["reminder_kind"] == "one_shot"
    assert "타이레놀" in sent[-1]["text"]


def test_taken_confirmation_uses_prescription_context_and_colloquial_ack(tmp_path):
    memory = make_memory(tmp_path)
    current = datetime(2026, 5, 26, 21, 40, 43)

    def now_provider():
        return current

    service = ReminderService(now_provider=now_provider, start_background_tasks=False)
    prescription_log = "# 현재 복용 약 요약\n\n## 약품 목록\n- 디오반정\n"

    assert ReminderService.is_taken_confirmation("어 먹었어") is True

    response = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="약 먹었어",
            user_profile={"name": "정현기"},
            prescription_log=prescription_log,
        )
    )

    assert "식후 디오반정" in response
    assert "식후 식후 약" not in response

    current = datetime(2026, 5, 26, 21, 41, 0)
    colloquial = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="어 먹었어",
            user_profile={"name": "정현기"},
            prescription_log=prescription_log,
        )
    )

    assert "식후 디오반정" in colloquial

    current = datetime(2026, 5, 26, 21, 42, 0)
    explicit_record = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="어 먹었어 기록해 줘",
            user_profile={"name": "정현기"},
            prescription_log=prescription_log,
        )
    )

    assert "식후 디오반정" in explicit_record
    saved = run(memory.store.read_user_file("speaker-reminder", "medication_taken.md"))
    assert '"medication_label": "디오반정"' in saved


def test_taken_time_record_command_and_minute_correction(tmp_path):
    memory = make_memory(tmp_path)
    current = datetime(2026, 5, 26, 22, 30, 56)

    def now_provider():
        return current

    service = ReminderService(now_provider=now_provider, start_background_tasks=False)
    prescription_log = "# 현재 복용 약 요약\n\n## 약품 목록\n- 디오반정\n"

    assert ReminderService.is_taken_confirmation("알았어 지금 먹을게 시간 기록해 줘") is True
    recorded = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="알았어 지금 먹을게 시간 기록해 줘",
            user_profile={"name": "김영수"},
            prescription_log=prescription_log,
        )
    )
    assert "오후 10시 30분" in recorded
    assert "식후 디오반정" in recorded
    assert "나지금" not in recorded

    current = datetime(2026, 5, 26, 22, 30, 58)
    colloquial_recorded = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="알았어 나 지금 먹을테니까 기록해 줘",
            user_profile={"name": "김영수"},
            prescription_log=prescription_log,
        )
    )
    assert "식후 디오반정" in colloquial_recorded
    assert "나지금" not in colloquial_recorded

    assert ReminderService.is_taken_recall("지금 먹었다며") is True
    assert ReminderService.is_taken_recall("내가 언제 뭘 먹었다고") is True
    recalled = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="내가 언제 뭘 먹었다고",
            user_profile={"name": "김영수"},
            prescription_log=prescription_log,
        )
    )
    assert "오후 10시 30분" in recalled

    current = datetime(2026, 5, 26, 22, 31, 25)
    assert ReminderService.is_taken_time_correction("지금은 31분") is True
    corrected = run(
        service.correct_last_taken_time(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="지금은 31분",
            user_profile={"name": "김영수"},
        )
    )
    assert "오후 10시 31분" in corrected

    recalled_after = run(
        service.recall_last_taken(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            user_profile={"name": "김영수"},
        )
    )
    assert "오후 10시 31분" in recalled_after


def test_relative_one_shot_generic_alarm_does_not_need_prescription_context(tmp_path):
    memory = make_memory(tmp_path)
    current = datetime(2026, 5, 26, 20, 14, 57)

    def now_provider():
        return current

    sent: list[dict] = []
    service = ReminderService(now_provider=now_provider, start_background_tasks=False)
    service.register_connection("speaker-reminder", lambda payload: sent.append(payload))

    setup = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="30초 뒤에 알람 설정해 줘",
            user_profile={"name": "정현기"},
            prescription_log="",
        )
    )

    assert "30초 뒤" in setup
    assert "알림" in setup
    assert "아침" not in setup
    assert "점심" not in setup
    assert ReminderService.is_one_shot_request("30초 뒤에 알람 설정해 줘", prescription_log="")
    assert not ReminderService.is_setup_request("30초 뒤에 알람 설정해 줘", prescription_log="")

    current = datetime(2026, 5, 26, 20, 15, 27)
    dispatched = run(service.dispatch_due_reminders())

    assert dispatched
    assert sent[-1]["reminder_kind"] == "one_shot"
    assert "요청하신 알림 시간" in sent[-1]["text"]


def test_one_shot_reminder_restores_and_dispatches_when_due(tmp_path):
    memory = make_memory(tmp_path)
    current = datetime(2026, 5, 26, 20, 14, 57)

    def now_provider():
        return current

    service = ReminderService(now_provider=now_provider, start_background_tasks=False)
    run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="30초 뒤에 알람 설정해 줘",
            user_profile={"name": "정현기"},
            prescription_log="",
        )
    )
    saved = run(memory.store.read_user_file("speaker-reminder", "one_shot_reminders.md"))
    assert "one_shot" not in saved
    assert "30초 뒤에 알람 설정해 줘" in saved

    current = datetime(2026, 5, 26, 20, 15, 27)
    restored_sent: list[dict] = []
    restored = ReminderService(now_provider=now_provider, start_background_tasks=False)
    restored.register_connection("speaker-reminder", lambda payload: restored_sent.append(payload))
    run(restored.restore_for_speaker(memory, "speaker-reminder"))

    assert restored_sent
    assert restored_sent[-1]["reminder_kind"] == "one_shot"
    assert "요청하신 알림 시간" in restored_sent[-1]["text"]
    after = run(memory.store.read_user_file("speaker-reminder", "one_shot_reminders.md"))
    assert '"items": []' in after


def test_pending_reminder_confusion_cancels_instead_of_reprompting(tmp_path):
    memory = make_memory(tmp_path)
    service = ReminderService(start_background_tasks=False)
    service.start_setup(
        speaker_id="speaker-reminder",
        user_profile={"name": "정현기"},
        prescription_log="# 현재 복용 약 요약\n- 혈압약\n",
    )

    response = run(
        service.handle_user_text(
            memory_engine=memory,
            speaker_id="speaker-reminder",
            text="잠깐만 그걸 왜 얘기하지",
            user_profile={"name": "정현기"},
        )
    )

    assert "중단" in response
    assert "speaker-reminder" not in service._pending


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
