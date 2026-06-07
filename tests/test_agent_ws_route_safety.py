from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from app.api.routes import agent_ws


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


class FakeMemoryEngine:
    async def bootstrap_flash_from_permanent(self, speaker_id: str) -> None:
        return None

    async def mark_identity_seen(self, speaker_id: str, *, verified: bool = False, now=None) -> dict:
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
    agent_ws._wake_profile_cache_by_speaker.clear()
    agent_ws._identity_pending_action_cache_by_speaker.clear()
    agent_ws._dialog_state_by_speaker.clear()


def test_agent_ws_global_safety_precheck_beats_pending_ocr_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket()
    speaker_id = "speaker-kim"
    agent_ws._speaker_state(speaker_id).active_flow = "ocr_confirm"
    agent_ws._pending_ocr_by_speaker[speaker_id] = {
        "data": {"medications": [{"name": "타이레놀"}]},
        "created_at": datetime.now(),
    }

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("global safety precheck should run before identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("global safety precheck should not call orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", FakeMemoryEngine())
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": speaker_id,
                "text": "타이레놀 4개 먹어도 돼",
            },
            set(),
        )
    )

    assert websocket.sent[-1]["type"] == "response"
    assert websocket.sent[-1]["route_label"] == "medication_safety"
    assert websocket.sent[-1]["engine_scope"] == "safety"
    assert websocket.sent[-1]["risk_level"] == "high"
    assert websocket.sent[-1]["db_write_expected"] is False
    assert "ocr_saved" not in {payload.get("response_type") for payload in websocket.sent}
    assert speaker_id in agent_ws._pending_ocr_by_speaker


def test_agent_ws_inactive_background_without_wake_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    websocket = FakeWebSocket()

    async def fail_if_identity_gate_called(**kwargs):
        raise AssertionError("inactive background should not call identity gate")

    async def fail_if_orchestrator_called(**kwargs):
        raise AssertionError("inactive background should not call orchestrator")

    monkeypatch.setattr(agent_ws, "memory_engine", FakeMemoryEngine())
    monkeypatch.setattr(agent_ws, "reminder_service", FakeReminderService())
    monkeypatch.setattr(agent_ws, "evaluate_identity_gate", fail_if_identity_gate_called)
    monkeypatch.setattr(agent_ws.engine_orchestrator, "run_turn", fail_if_orchestrator_called)

    run(
        agent_ws._handle_stt(
            websocket,
            {
                "type": "stt_result",
                "speaker_id": "speaker-bg",
                "text": "티비 소리야 약 광고 나온 거야",
                "client_context": {"active_session": False, "voice_armed": False},
            },
            set(),
        )
    )

    assert websocket.sent[-1]["type"] == "ignored"
    assert websocket.sent[-1]["reason"] == "ignored_background"
