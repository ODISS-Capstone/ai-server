"""Regression coverage for wake-word-only WebSocket turns."""
from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.api.routes import agent_ws
from app.database.md_store import MDStore
from app.engines.memory import MemoryEngine
from app.memory import StructuredMemoryService
from app.services.identity_guard import IdentityGateResult
from app.services.reminders import ReminderService


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


class FakeMemoryEngine:
    def __init__(self) -> None:
        self.seen: list[tuple[str, bool]] = []

    async def bootstrap_flash_from_permanent(self, speaker_id: str) -> None:
        return None

    async def load_context(self, speaker_id: str) -> dict:
        return {
            "user_profile": {
                "name": "김영수",
                "age": "72",
                "gender": "남성",
            },
            "prescription_log": "# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        }

    async def mark_identity_seen(self, speaker_id: str, *, verified: bool = False, now=None) -> dict:
        self.seen.append((speaker_id, verified))
        return {}

    async def update_and_compress(self, response_data: dict, speaker_id: str | None = None) -> None:
        return None


class FakeReminderService:
    def register_connection(self, speaker_id: str, callback) -> None:
        return None

    async def restore_for_speaker(self, memory_engine, speaker_id: str) -> None:
        return None

    async def handle_user_text(self, **kwargs):
        return None


def run(coro):
    return asyncio.run(coro)


def setup_function() -> None:
    agent_ws._bootstrapped_speakers.clear()
    agent_ws._pending_ocr_by_speaker.clear()
    agent_ws._queued_ocr_request_by_speaker.clear()


def make_real_memory(tmp_path) -> MemoryEngine:
    memory = MemoryEngine()
    memory.store = MDStore(str(tmp_path / "md_database"))
    memory.structured_memory = StructuredMemoryService(base_path=str(tmp_path / "structured_memory"))
    run(memory.initialize())
    run(
        memory.store.write_flash(
            "prescription_log",
            "# 현재 복용 약 요약\n\n## 약품 목록\n- 혈압약\n",
        )
    )
    return memory


def test_registered_wake_word_uses_profile_and_skips_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={
                "profile": {
                    "name": "김영수",
                    "age": "72",
                    "gender": "남성",
                }
            },
        )

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("wake-word-only turn should not call the orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "오디스",
            },
            set(),
        )
    )

    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "wake_word_ack"
    assert websocket.sent[-1]["requires_tts"] is True
    assert websocket.sent[-1]["response_text"] == "네, 김영수님. 말씀하세요."
    assert "어르신" not in websocket.sent[-1]["response_text"]
    assert "약" not in websocket.sent[-1]["response_text"]
    assert fake_memory.seen == [("speaker-kim", True)]


def test_reminder_setup_sends_filler_before_reminder_response(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "김영수", "age": "72", "gender": "남성"}},
        )

    class SlowReminderService(FakeReminderService):
        async def handle_user_text(self, **kwargs):
            assert websocket.sent
            assert websocket.sent[0]["type"] == "filler"
            await asyncio.sleep(0)
            return "네, 김영수님. 현재 저장된 복약 정보 기준으로 식후 복용 알림을 설정할 수 있습니다."

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", SlowReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "오디스, 내가 밥 먹고 나서 약 먹어야 한다는 걸 알림 추가해줘.",
            },
            set(),
        )
    )

    assert websocket.sent[0]["type"] == "filler"
    assert websocket.sent[0]["stage"] == "reminder"
    assert websocket.sent[0]["requires_tts"] is True
    assert "알림" in websocket.sent[0]["text"]
    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "reminder"
    assert "김영수님" in websocket.sent[-1]["response_text"]


def test_agent_ws_reminder_setup_confirm_and_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    current = datetime(2026, 5, 18, 11, 55)

    def now_provider() -> datetime:
        return current

    fake_memory = make_real_memory(tmp_path)
    websocket = FakeWebSocket()
    reminder_service = ReminderService(now_provider=now_provider, start_background_tasks=False)

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "김영수", "age": "72", "gender": "남성"}},
        )

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", reminder_service)
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "오디스, 약 먹을 때 깨워줘.",
            },
            set(),
        )
    )
    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "점심은 12시로 해줘.",
            },
            set(),
        )
    )

    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "reminder"
    assert "점심 약 알림은 오후 12시" in websocket.sent[-1]["response_text"]

    current = datetime(2026, 5, 18, 12, 0)
    dispatched = run(reminder_service.dispatch_due_reminders())

    assert dispatched
    assert websocket.sent[-1]["type"] == "reminder"
    assert "점심 혈압약" in websocket.sent[-1]["text"]


def test_medication_reasoning_sends_filler_before_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_memory = FakeMemoryEngine()
    websocket = FakeWebSocket()

    async def fake_identity_gate(**kwargs):
        return IdentityGateResult(
            allowed=True,
            reason="identity_verified",
            metadata={"profile": {"name": "김영수", "age": "72", "gender": "남성"}},
        )

    async def fake_run_turn(**kwargs):
        assert websocket.sent
        assert websocket.sent[0]["type"] == "filler"
        return SimpleNamespace(
            filler_text="",
            conversation=SimpleNamespace(
                requires_tts=True,
                response_text="김영수님, 현재 기록 기준으로 혈압약을 확인하고 답변드릴게요.",
                response_type="medical_response",
            ),
            execution_results={"task_results": {}},
            decision=SimpleNamespace(rationale="test", intent="medication_query"),
            core_message="",
            judge_review={},
        )

    monkeypatch.setattr(agent_ws, "memory_engine", fake_memory)
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fake_identity_gate)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fake_run_turn)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-kim",
                "text": "오디스, 혈압약 두 번 먹으면 더 빨리 좋아져?",
            },
            set(),
        )
    )

    assert websocket.sent[0]["type"] == "filler"
    assert websocket.sent[0]["stage"] == "dur"
    assert websocket.sent[0]["requires_tts"] is True
    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["response_type"] == "medical_response"


def test_progress_fillers_continue_until_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    websocket = FakeWebSocket()
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= 6:
            raise asyncio.CancelledError()

    monkeypatch.setattr(agent_ws.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        run(
            agent_ws._send_progress_fillers(
                websocket,
                "오디스, 혈압약 두 번 먹으면 더 빨리 좋아져?",
                initial_sent=True,
            )
        )

    assert delays == [2.5, 4.0, 5.0, 7.0, 7.0, 7.0]
    assert len(websocket.sent) == 5
    assert {payload["stage"] for payload in websocket.sent} == {"dur"}
    assert all(payload["type"] == "filler" for payload in websocket.sent)
    assert len({payload["text"] for payload in websocket.sent}) >= 3


def test_ocr_progress_fillers_continue_until_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    websocket = FakeWebSocket()
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= 6:
            raise asyncio.CancelledError()

    monkeypatch.setattr(agent_ws.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        run(agent_ws._send_ocr_progress_fillers(websocket))

    assert delays == [2.5, 4.0, 5.0, 7.0, 7.0, 7.0]
    assert len(websocket.sent) == 5
    assert {payload["stage"] for payload in websocket.sent} == {"ocr_processing"}
    assert all(payload["type"] == "filler" for payload in websocket.sent)
